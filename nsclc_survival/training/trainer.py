"""
Training loop utilities.

Key design decisions
--------------------
1. forward_model() dispatches batch keys → model keyword arguments via
   **batch, so each dataset class (DS_Clinical, DS_MAAS, DS_COLIPRI_Patch, …)
   only needs to return the keys its model expects — no if/else per experiment.

2. Age z-scoring is applied INSIDE the loop (normalise_age_per_fold) on every
   fold, preventing leakage of test-set statistics into the training process.

3. MedicalNet (E14/E15) uses a 3-phase gradual-unfreeze schedule:
     Phase 1: freeze backbone, train head for 10 epochs
     Phase 2: unfreeze layer4+head for 20 epochs
     Phase 3: unfreeze all for remaining epochs

4. Checkpoints are saved at the end of every fold (not just the best validation
   epoch) because we evaluate on OOF predictions, not a single held-out set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lifelines.statistics import concordance_index

from ..utils.seed import set_seed, seed_worker, make_generator, SEED
from ..data.clinical import normalise_age_per_fold
from ..losses.cox import LOSS_REGISTRY


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
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def forward_model(model: nn.Module, batch: dict) -> torch.Tensor:
    """
    Call model(**batch_keys_that_model_accepts).

    This works because all model forward() signatures use **kw to absorb
    unused keys.  The batch dict may contain 'surv', 'event', 'pid' (metadata
    only) alongside tensor inputs; these are ignored via **kw.
    """
    return model(**batch)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    fold: int,
    epoch: int,
    oof_cindex: float,
    out_dir: Path,
    exp_name: str,
) -> Path:
    """Save model state + metadata to {out_dir}/{exp_name}_fold{fold}.pt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_name}_fold{fold:02d}.pt"
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "fold":       fold,
            "epoch":      epoch,
            "oof_cindex": oof_cindex,
            "exp_name":   exp_name,
        },
        str(path),
    )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MedicalNet unfreeze schedule
# ─────────────────────────────────────────────────────────────────────────────

def _apply_medicalnet_unfreeze(model: nn.Module, epoch: int) -> None:
    """
    3-phase gradual unfreezing for MedicalNetSurv_E14/E15.
    Only called when model has freeze_backbone / unfreeze_layer4 / unfreeze_all.
    """
    if not hasattr(model, "freeze_backbone"):
        return
    if epoch == 0:
        model.freeze_backbone()
    elif epoch == 10:
        model.unfreeze_layer4()
    elif epoch == 30:
        model.unfreeze_all()


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    loss_fn_name: str = "efron",
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, List[float]]:
    """
    Train model for n_epochs.  Returns trained model and per-epoch val C-index.

    Parameters
    ----------
    model        : uninitialised (or pretrained) PyTorch model
    train_loader : DataLoader yielding batch dicts with 'surv' and 'event'
    val_loader   : DataLoader for in-fold validation (not test)
    loss_fn_name : 'breslow' or 'efron' (from LOSS_REGISTRY)
    """
    if device is None:
        device = _get_device()

    model = model.to(device)
    loss_fn = LOSS_REGISTRY[loss_fn_name]
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)

    val_cindices: List[float] = []

    for epoch in range(n_epochs):
        _apply_medicalnet_unfreeze(model, epoch)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            eta = forward_model(model, batch)
            loss = loss_fn(eta, batch["surv"].float(), batch["event"].float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        etas, survs, events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _batch_to_device(batch, device)
                etas.append(forward_model(model, batch).cpu().numpy())
                survs.append(batch["surv"].cpu().numpy())
                events.append(batch["event"].cpu().numpy())

        etas   = np.concatenate(etas)
        survs  = np.concatenate(survs)
        events = np.concatenate(events)

        try:
            ci = concordance_index(survs, -etas, events)
        except Exception:
            ci = 0.5

        val_cindices.append(ci)

        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(len(train_loader), 1)
            print(
                f"    Epoch {epoch+1:3d}/{n_epochs}  "
                f"loss={avg_loss:.4f}  val_CI={ci:.4f}"
            )

    return model, val_cindices


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    model_cls,
    model_kwargs: dict,
    dataset_cls,
    dataset_kwargs: dict,
    df,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    loss_fn_name: str = "efron",
    ckpt_dir: Optional[Path] = None,
    num_workers: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one experiment across all folds, collecting OOF predictions.

    Parameters
    ----------
    exp_name     : identifier string, e.g. 'E2', 'E16a'
    model_cls    : class (not instance) to instantiate per fold
    model_kwargs : kwargs passed to model_cls()
    dataset_cls  : Dataset class (not instance)
    dataset_kwargs : static kwargs passed to dataset_cls(); 'df' is added per fold
    df           : cohort DataFrame (from load_labels)
    splits       : list of (tv_idx, test_idx)  — from make_splits()

    Returns
    -------
    dict with keys: oof_eta, oof_surv, oof_event, oof_pid, fold_cindices,
                    oof_cindex, elapsed_s, checkpoint_paths
    """
    device = _get_device()
    set_seed(SEED)

    oof_eta    = np.zeros(len(df))
    oof_surv   = np.zeros(len(df))
    oof_event  = np.zeros(len(df))
    oof_pid    = [""] * len(df)
    fold_ci    = []
    ckpt_paths = []

    t0 = time.time()

    for fold_idx, (tv_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"\n  {exp_name} fold {fold_idx+1}/{len(splits)}")

        set_seed(SEED + fold_idx)     # deterministic per fold

        # Age z-score on training fold only
        df_fold = normalise_age_per_fold(df, tv_idx)

        # Train/val split (80/20 of tv_idx)
        rng      = np.random.default_rng(SEED + fold_idx)
        perm     = rng.permutation(len(tv_idx))
        n_val    = max(1, int(len(tv_idx) * 0.2))
        val_pos  = perm[:n_val]
        train_pos = perm[n_val:]
        train_idx = tv_idx[train_pos]
        val_idx   = tv_idx[val_pos]

        # Build datasets
        ds_train = dataset_cls(df=df_fold, indices=train_idx,
                               augment=True, **dataset_kwargs)
        ds_val   = dataset_cls(df=df_fold, indices=val_idx,
                               augment=False, **dataset_kwargs)
        ds_test  = dataset_cls(df=df_fold, indices=test_idx,
                               augment=False, **dataset_kwargs)

        g = make_generator()
        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, worker_init_fn=seed_worker,
            generator=g, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"))
        test_loader = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"))

        # Model
        model = model_cls(**model_kwargs)

        # Train
        model, _ = train_one_fold(
            model, train_loader, val_loader,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            loss_fn_name=loss_fn_name, device=device, verbose=verbose)

        # OOF inference
        model.eval()
        etas_fold, survs_fold, events_fold, pids_fold = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_dev = _batch_to_device(batch, device)
                eta = forward_model(model, batch_dev).cpu().numpy()
                etas_fold.append(eta)
                survs_fold.append(batch["surv"].numpy())
                events_fold.append(batch["event"].numpy())
                if "pid" in batch:
                    pids_fold.extend(batch["pid"])

        etas_fold   = np.concatenate(etas_fold)
        survs_fold  = np.concatenate(survs_fold)
        events_fold = np.concatenate(events_fold)

        oof_eta[test_idx]   = etas_fold
        oof_surv[test_idx]  = survs_fold
        oof_event[test_idx] = events_fold
        if pids_fold:
            for pos, pid in zip(test_idx, pids_fold):
                oof_pid[pos] = pid

        try:
            ci = concordance_index(survs_fold, -etas_fold, events_fold)
        except Exception:
            ci = 0.5
        fold_ci.append(ci)

        if verbose:
            print(f"    Fold {fold_idx+1} test C-index: {ci:.4f}")

        # Save checkpoint
        if ckpt_dir is not None:
            ckpt_path = save_checkpoint(
                model, torch.optim.Adam(model.parameters()),  # re-use signature
                fold_idx + 1, n_epochs, ci, ckpt_dir, exp_name)
            ckpt_paths.append(str(ckpt_path))

    elapsed = time.time() - t0
    oof_ci  = concordance_index(oof_surv, -oof_eta, oof_event)

    if verbose:
        print(
            f"\n  {exp_name} OOF C-index: {oof_ci:.4f}  "
            f"(mean fold: {np.mean(fold_ci):.4f} ± {np.std(fold_ci):.4f})  "
            f"elapsed: {elapsed/60:.1f} min"
        )

    return {
        "oof_eta":          oof_eta,
        "oof_surv":         oof_surv,
        "oof_event":        oof_event,
        "oof_pid":          oof_pid,
        "fold_cindices":    fold_ci,
        "oof_cindex":       oof_ci,
        "elapsed_s":        elapsed,
        "checkpoint_paths": ckpt_paths,
    }
