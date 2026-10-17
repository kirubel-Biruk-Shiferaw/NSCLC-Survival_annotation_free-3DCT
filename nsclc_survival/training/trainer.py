"""
Training loop and experiment runner (faithful port of the thesis protocol).

Protocol (identical across CNN experiments E1–E15)
--------------------------------------------------
  - Optimiser : AdamW, lr=1.2e-3 (3e-4 x 4 linear-scaling), weight_decay=1e-2
  - Schedule  : CosineAnnealingWarmRestarts, T_0=20, T_mult=2, eta_min=1e-6
  - Grad accum: 4 physical steps (batch 8 -> effective batch 32) for CNNs
  - Early stop: patience=20 on validation C-index (not on loss)
  - Best ckpt : restored before test evaluation

COLIPRI heads (E16a/b, E17a/b/c/d)
----------------------------------
  - lr=1e-3, up to 150 epochs, patience=30, grad_accum=1, no augmentation,
    10-fold CV. Batch 32 for pooled heads, 4 for patch heads.

MedicalNet (E14/E15) 3-phase gradual unfreeze
---------------------------------------------
  Phase 1 (10 ep): freeze backbone, train head only
  Phase 2 (20 ep): unfreeze layer4 + head
  Phase 3 (rest) : unfreeze all; the optimiser/scheduler are rebuilt at each
                   phase boundary over the newly trainable parameters.

Inner CV split
--------------
  For each outer fold, the training pool is split 5:1 with
  StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), taking the
  FIRST split. This inner split is fixed across experiments.

Sign convention
---------------
  Models output log-hazard eta (higher = higher risk). concordance_index()
  expects higher score = longer survival, so risks are negated: -eta.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from ..utils.seed import set_seed, seed_worker, make_generator, SEED
from ..data.clinical import normalise_age_per_fold
from ..losses.cox import LOSS_REGISTRY

# ── Canonical hyperparameters (thesis Section 2.7) ──────────────────────────
CNN_LR            = 3e-4 * 4    # 1.2e-3, linear-scaling rule
CNN_EPOCHS        = 100
CNN_PATIENCE      = 20
CNN_GRAD_ACCUM    = 4           # effective batch = 8 x 4 = 32
CNN_BATCH_SIZE    = 8
WEIGHT_DECAY      = 1e-2

COLIPRI_LR        = 1e-3
COLIPRI_EPOCHS    = 150
COLIPRI_PATIENCE  = 30
COLIPRI_BATCH     = 32          # patch heads override to 4

MAX_GRAD_NORM     = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def forward_model(
    model:        nn.Module,
    vol:          torch.Tensor = None,
    clinical:     torch.Tensor = None,
    patch:        torch.Tensor = None,
    emb:          torch.Tensor = None,
    r1:           torch.Tensor = None,
    r2:           torch.Tensor = None,
    r3:           torch.Tensor = None,
    use_clinical: bool = False,
) -> torch.Tensor:
    """
    Route the forward pass to the correct model signature. All model forward()
    methods accept **kw, so only the keys the model needs are supplied.
    """
    kw: Dict[str, torch.Tensor] = {}
    if vol   is not None: kw["vol"]   = vol
    if patch is not None: kw["patch"] = patch
    if emb   is not None: kw["emb"]   = emb
    if r1    is not None: kw["r1"]    = r1
    if r2    is not None: kw["r2"]    = r2
    if r3    is not None: kw["r3"]    = r3
    if use_clinical and clinical is not None:
        kw["clinical"] = clinical
    elif vol is None and patch is None and emb is None:
        # Clinical-only model (E10/E11)
        kw["clinical"] = clinical
    return model(**kw)


def _predict(model, loader, device, use_clinical):
    """Return (risks, times, events) lists over a loader."""
    risks, times, events = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            log_h = forward_model(
                model,
                vol=batch.get("vol"), clinical=batch.get("clinical"),
                patch=batch.get("patch"), emb=batch.get("emb"),
                r1=batch.get("r1"), r2=batch.get("r2"), r3=batch.get("r3"),
                use_clinical=use_clinical)
            risks.extend(log_h.cpu().tolist())
            times.extend(batch["t"].cpu().tolist())
            events.extend(batch["e"].cpu().tolist())
    return risks, times, events


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    model:          nn.Module,
    train_loader:   DataLoader,
    valid_loader:   DataLoader,
    test_loader:    DataLoader,
    *,
    use_clinical:   bool = False,
    loss_fn_name:   str = "breslow",
    num_epochs:     int = CNN_EPOCHS,
    patience:       int = CNN_PATIENCE,
    lr:             float = CNN_LR,
    weight_decay:   float = WEIGHT_DECAY,
    grad_accum:     int = CNN_GRAD_ACCUM,
    phase_schedule: Optional[list] = None,
    device:         Optional[torch.device] = None,
) -> Tuple[float, float, int, dict, list, list, list]:
    """
    Train one CV fold with early stopping, restore the best-validation
    checkpoint, and evaluate on the test fold.

    Returns
    -------
    (test_ci, best_val_ci, epochs_run, best_state, test_risks, test_times,
     test_events)
    """
    if device is None:
        device = _get_device()
    set_seed(SEED)

    model = model.to(device)
    loss_fn = LOSS_REGISTRY[loss_fn_name]

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_ci = 0.0
    best_state  = copy.deepcopy(model.state_dict())
    no_improve  = 0
    epochs_run  = 0

    for epoch in range(1, num_epochs + 1):

        # ── MedicalNet 3-phase gradual unfreeze ─────────────────────────────
        if phase_schedule:
            cumulative = 0
            for n_ep, unfreeze_fn in phase_schedule:
                if epoch == cumulative + 1 and unfreeze_fn is not None:
                    unfreeze_fn()
                    opt = AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, weight_decay=weight_decay)
                    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        opt, T_0=20, T_mult=2, eta_min=1e-6)
                cumulative += n_ep

        # ── Training ────────────────────────────────────────────────────────
        model.train()
        opt.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = _batch_to_device(batch, device)
            log_h = forward_model(
                model,
                vol=batch.get("vol"), clinical=batch.get("clinical"),
                patch=batch.get("patch"), emb=batch.get("emb"),
                r1=batch.get("r1"), r2=batch.get("r2"), r3=batch.get("r3"),
                use_clinical=use_clinical)
            loss = loss_fn(log_h, batch["t"], batch["e"].float())
            # Scale by 1/grad_accum so the accumulated gradient matches a
            # single large-batch step.
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                opt.step()
                opt.zero_grad()
                sched.step()

        # ── Validation (early stop on C-index) ──────────────────────────────
        v_risks, v_times, v_events = _predict(model, valid_loader, device, use_clinical)
        try:
            val_ci = concordance_index(v_times, [-r for r in v_risks], v_events)
        except Exception:
            val_ci = 0.5

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1

        epochs_run = epoch
        if no_improve >= patience:
            break

    # ── Test on the best checkpoint ─────────────────────────────────────────
    model.load_state_dict(best_state)
    t_risks, t_times, t_events = _predict(model, test_loader, device, use_clinical)
    try:
        test_ci = concordance_index(t_times, [-r for r in t_risks], t_events)
    except Exception:
        test_ci = 0.5

    return test_ci, best_val_ci, epochs_run, best_state, t_risks, t_times, t_events


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp_name:       str,
    model_cls,
    model_kwargs:   dict,
    dataset_cls,
    dataset_kwargs: dict,
    df,
    splits:         List[Tuple[np.ndarray, np.ndarray]],
    *,
    use_clinical:   bool = False,
    loss_fn_name:   str = "breslow",
    is_colipri:     bool = False,
    mednet_phases:  bool = False,
    n_epochs:       Optional[int] = None,
    patience:       Optional[int] = None,
    lr:             Optional[float] = None,
    weight_decay:   float = WEIGHT_DECAY,
    grad_accum:     Optional[int] = None,
    batch_size:     Optional[int] = None,
    ckpt_dir:       Optional[Path] = None,
    device:         Optional[torch.device] = None,
    verbose:        bool = True,
    **_ignored,
) -> Dict[str, Any]:
    """
    Run one experiment across all folds and collect OOF predictions.

    Hyperparameters default to the canonical CNN values, or the COLIPRI values
    when ``is_colipri=True``; any argument explicitly passed overrides them.
    ``dataset_kwargs`` may contain a ``df_ref`` key (ignored here; datasets that
    need it read it themselves).
    """
    device = device or _get_device()

    # Canonical hyperparameter resolution
    n_epochs   = n_epochs   or (COLIPRI_EPOCHS   if is_colipri else CNN_EPOCHS)
    patience   = patience   or (COLIPRI_PATIENCE if is_colipri else CNN_PATIENCE)
    lr         = lr         or (COLIPRI_LR       if is_colipri else CNN_LR)
    grad_accum = grad_accum or (1               if is_colipri else CNN_GRAD_ACCUM)
    batch_size = batch_size or (COLIPRI_BATCH    if is_colipri else CNN_BATCH_SIZE)
    augment    = not is_colipri       # CNN experiments use augmentation

    set_seed(SEED)
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_eta   = np.zeros(len(df))
    oof_surv  = np.zeros(len(df))
    oof_event = np.zeros(len(df))
    fold_ci: List[float] = []
    ckpt_paths: List[str] = []

    n_params = sum(p.numel() for p in model_cls(**model_kwargs).parameters())
    if verbose:
        print(f"\n{'='*70}\n  {exp_name}  "
              f"params={n_params:,}  folds={len(splits)}  loss={loss_fn_name}"
              f"\n{'='*70}")

    t0 = time.time()

    for fn, (tv_idx, test_idx) in enumerate(splits, start=1):
        tv_idx   = np.asarray(tv_idx)
        test_idx = np.asarray(test_idx)

        # Inner split — fixed FIRST stratified split of the training pool
        tr_s, va_s = next(inner_skf.split(df.iloc[tv_idx],
                                          df.iloc[tv_idx]["status"].values))
        train_idx = tv_idx[tr_s]
        valid_idx = tv_idx[va_s]

        # Age z-score using ONLY training statistics (no leakage)
        df_fold = normalise_age_per_fold(df, train_idx)

        tr_ds = dataset_cls(df_fold, train_idx, augment=augment, **dataset_kwargs)
        va_ds = dataset_cls(df_fold, valid_idx, augment=False,   **dataset_kwargs)
        te_ds = dataset_cls(df_fold, test_idx,  augment=False,   **dataset_kwargs)

        g = make_generator()
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, drop_last=True,
                           generator=g, worker_init_fn=seed_worker)
        va_dl = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=0)
        te_dl = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=0)

        # Fresh model each fold; set_seed for identical initialisation
        model = model_cls(**model_kwargs).to(device)
        set_seed(SEED)

        # MedicalNet 3-phase schedule
        phases = None
        if mednet_phases:
            phases = [
                (10,              None),                  # Phase 1: frozen
                (20,              model.unfreeze_layer4),  # Phase 2: layer4
                (n_epochs - 30,   model.unfreeze_all),     # Phase 3: all
            ]
            model.freeze_backbone()

        (test_ci, best_val_ci, n_ep, best_state,
         t_risks, t_times, t_events) = train_one_fold(
            model, tr_dl, va_dl, te_dl,
            use_clinical=use_clinical, loss_fn_name=loss_fn_name,
            num_epochs=n_epochs, patience=patience, lr=lr,
            weight_decay=weight_decay, grad_accum=grad_accum,
            phase_schedule=phases, device=device)

        fold_ci.append(test_ci)
        oof_eta[test_idx]   = t_risks
        oof_surv[test_idx]  = t_times
        oof_event[test_idx] = t_events

        if verbose:
            print(f"  Fold {fn}/{len(splits)}: test_CI={test_ci:.4f}  "
                  f"best_val_CI={best_val_ci:.4f}  epochs={n_ep}")

        if ckpt_dir is not None:
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{exp_name}_fold{fn:02d}.pt"
            torch.save({
                "model_state_dict": best_state,
                "fold": fn, "epoch": n_ep,
                "test_cindex": test_ci, "val_cindex": best_val_ci,
                "exp_name": exp_name, "model_cls": model_cls.__name__,
                "model_kwargs": model_kwargs,
            }, str(ckpt_path))
            ckpt_paths.append(str(ckpt_path))

    elapsed = time.time() - t0
    oof_ci  = concordance_index(oof_surv, -oof_eta, oof_event)

    if verbose:
        print(f"\n  {exp_name} COMPLETE: OOF C-index={oof_ci:.4f}  "
              f"(mean fold {np.mean(fold_ci):.4f} +/- {np.std(fold_ci):.4f})  "
              f"elapsed {elapsed/60:.1f} min")
        print(f"  Per-fold: {[f'{c:.4f}' for c in fold_ci]}")

    return {
        "exp":              exp_name,
        "oof_eta":          oof_eta,
        "oof_surv":         oof_surv,
        "oof_event":        oof_event,
        "fold_cindices":    fold_ci,
        "mean_ci":          float(np.mean(fold_ci)),
        "std_ci":           float(np.std(fold_ci)),
        "oof_cindex":       float(oof_ci),
        "n_params":         int(n_params),
        "elapsed_s":        elapsed,
        "checkpoint_paths": ckpt_paths,
    }
