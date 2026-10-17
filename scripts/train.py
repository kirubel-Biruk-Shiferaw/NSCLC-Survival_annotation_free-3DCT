#!/usr/bin/env python3
"""
Train NSCLC survival models.

Usage
-----
# Train all experiments (E1–E17d):
python scripts/train.py --config configs/default.yaml

# Train a single experiment:
python scripts/train.py --config configs/default.yaml --exp E2

# Override output directory:
NSCLC_OUTPUT_DIR=/path/to/outputs python scripts/train.py

Environment variables
---------------------
NSCLC_ARTIFACT_DIR : root of TCIA Lung1 data (R2 volumes, CSV, COLIPRI embeddings)
NSCLC_OUTPUT_DIR   : where checkpoints, OOF predictions, and results are saved
MEDICALNET_CKPT    : path to MedicalNet ResNet10 checkpoint (.pth)

See README.md for full setup instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsclc_survival.utils.config import load_config
from nsclc_survival.utils.seed import set_seed
from nsclc_survival.data.clinical import load_labels
from nsclc_survival.data.splits import make_splits, load_splits, print_split_stats
from nsclc_survival.training.trainer import run_experiment

# ── Model imports ──────────────────────────────────────────────────────────
from nsclc_survival.models.cnn import (
    CNNSurv_E1, CNNSurv_E2, CNNSurv_E3, CNNSurv_E4,
    CNNSurv_E5, CNNSurv_E6, CNNSurv_E7, CNNSurv_E8, CNNSurv_E9,
    ClinicalMLP_E10, DeepSurv_E11,
    CNNSurv_E12, CNNSurv_MAAS_E13,
    MedicalNetSurv_E14, MedicalNetSurv_E15,
)
from nsclc_survival.models.colipri import (
    COLIPRI_Pooled_MLP_E16a, COLIPRI_Pooled_Clinical_E16b,
    COLIPRI_Patch_AttnPool_E17a, COLIPRI_Patch_Conv_GAP_E17b,
    COLIPRI_Patch_Conv_AttnPool_E17c,
    COLIPRI_Patch_Conv_AttnPool_Clinical_E17d,
)

# ── Dataset imports ────────────────────────────────────────────────────────
from nsclc_survival.data.datasets import (
    DS_Base, DS_HeavyAug, DS_LungMasked, DS_Clinical, DS_ClinicalOnly,
    DS_LungCrop_HU, DS_MAAS, DS_COLIPRI_Pooled, DS_COLIPRI_Patch,
)


def get_experiment_registry(cfg, medicalnet_ckpt: str | None = None) -> dict:
    """
    Return a registry mapping experiment name → (model_cls, model_kwargs,
    dataset_cls, dataset_kwargs, training_kwargs).

    All paths come from cfg (loaded from default.yaml + env vars).
    """
    data_dir  = Path(cfg.data.artifact_dir)
    vol_dir   = data_dir / "R2"
    emb_dir   = data_dir / "colipri_embeddings"

    ds_common = dict(vol_dir=vol_dir)

    return {
        # ── CNN ablation (Breslow loss, 5-fold) ──────────────────────────────
        "E1":  (CNNSurv_E1,  {},  DS_Base,        ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E2":  (CNNSurv_E2,  {},  DS_Base,        ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E3":  (CNNSurv_E3,  {},  DS_Base,        ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E4":  (CNNSurv_E4,  {},  DS_Base,        ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E5":  (CNNSurv_E5,  {},  DS_Base,        ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E6":  (CNNSurv_E6,  {},  DS_Base,        ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E7":  (CNNSurv_E7,  {},  DS_LungMasked,  ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E8":  (CNNSurv_E8,  {},  DS_HeavyAug,    ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),
        "E9":  (CNNSurv_E9,  {},  DS_Clinical,    ds_common,  {"loss_fn_name": "breslow", "n_epochs": cfg.training.n_epochs_cnn}),

        # ── Clinical baselines (Efron loss, 5-fold) ───────────────────────────
        "E10": (ClinicalMLP_E10, {}, DS_ClinicalOnly, {}, {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_cnn}),
        "E11": (DeepSurv_E11,    {}, DS_ClinicalOnly, {}, {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_cnn}),

        # ── Spatial correction (Efron, 5-fold) ──────────────────────────────
        "E12": (CNNSurv_E12,      {}, DS_LungCrop_HU, ds_common, {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_cnn}),
        "E13": (CNNSurv_MAAS_E13, {}, DS_MAAS,        ds_common, {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_cnn}),

        # ── MedicalNet pretrained (Efron, 5-fold) ─────────────────────────
        "E14": (MedicalNetSurv_E14,
                {"ckpt_path": medicalnet_ckpt} if medicalnet_ckpt else {},
                DS_Base, ds_common,
                {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_medicalnet,
                 "lr": cfg.training.lr_medicalnet}),
        "E15": (MedicalNetSurv_E15,
                {"ckpt_path": medicalnet_ckpt} if medicalnet_ckpt else {},
                DS_LungCrop_HU, ds_common,
                {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_medicalnet,
                 "lr": cfg.training.lr_medicalnet}),

        # ── COLIPRI (Efron, 10-fold) ─────────────────────────────────────
        "E16a": (COLIPRI_Pooled_MLP_E16a,      {}, DS_COLIPRI_Pooled,
                 {"emb_dir": emb_dir},
                 {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_colipri}),
        "E16b": (COLIPRI_Pooled_Clinical_E16b, {}, DS_COLIPRI_Pooled,
                 {"emb_dir": emb_dir},
                 {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_colipri}),
        "E17a": (COLIPRI_Patch_AttnPool_E17a,               {}, DS_COLIPRI_Patch,
                 {"emb_dir": emb_dir},
                 {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_colipri}),
        "E17b": (COLIPRI_Patch_Conv_GAP_E17b,               {}, DS_COLIPRI_Patch,
                 {"emb_dir": emb_dir},
                 {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_colipri}),
        "E17c": (COLIPRI_Patch_Conv_AttnPool_E17c,          {}, DS_COLIPRI_Patch,
                 {"emb_dir": emb_dir},
                 {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_colipri}),
        "E17d": (COLIPRI_Patch_Conv_AttnPool_Clinical_E17d, {}, DS_COLIPRI_Patch,
                 {"emb_dir": emb_dir},
                 {"loss_fn_name": "efron", "n_epochs": cfg.training.n_epochs_colipri}),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NSCLC survival models")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--exp",     default=None, help="Run only this experiment (e.g. E2)")
    parser.add_argument("--medicalnet-ckpt", default=None,
                        help="Path to MedicalNet ResNet10 checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    out_dir  = Path(cfg.output.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    oof_dir  = out_dir / "oof_predictions"
    splits_dir = out_dir / "splits"

    for d in [ckpt_dir, oof_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load cohort ──────────────────────────────────────────────────────────
    csv_path = Path(cfg.data.artifact_dir) / cfg.data.csv_filename
    print(f"Loading cohort from {csv_path}")
    df = load_labels(csv_path)
    print(f"  N={len(df)}, event_rate={df['status'].mean():.3f}")

    # ── Splits ───────────────────────────────────────────────────────────────
    splits_5_path  = splits_dir / "cv_splits_5fold.json"
    splits_10_path = splits_dir / "cv_splits_10fold.json"

    if splits_5_path.exists() and splits_10_path.exists():
        print("Loading existing splits from disk…")
        spl_5, spl_10 = load_splits(splits_dir)
    else:
        print("Creating new splits…")
        spl_5, spl_10 = make_splits(
            df,
            n_folds_5=cfg.training.n_folds_cnn,
            n_folds_10=cfg.training.n_folds_colipri,
            save_dir=splits_dir,
        )

    print_split_stats(df, spl_5,  "5-fold")
    print_split_stats(df, spl_10, "10-fold")

    # ── Experiment registry ──────────────────────────────────────────────────
    registry = get_experiment_registry(cfg, args.medicalnet_ckpt)

    colipri_exps = {"E16a", "E16b", "E17a", "E17b", "E17c", "E17d"}

    exps_to_run = [args.exp] if args.exp else list(registry.keys())
    all_results = {}

    for exp in exps_to_run:
        if exp not in registry:
            print(f"WARNING: Unknown experiment '{exp}', skipping.")
            continue

        model_cls, model_kwargs, dataset_cls, ds_kwargs, train_kwargs = registry[exp]
        splits = spl_10 if exp in colipri_exps else spl_5

        print(f"\n{'='*60}")
        print(f"  Running {exp}  ({len(splits)}-fold)")
        print(f"{'='*60}")

        result = run_experiment(
            exp_name=exp,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            dataset_cls=dataset_cls,
            dataset_kwargs={**ds_kwargs, "df_ref": df},
            df=df,
            splits=splits,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            batch_size=cfg.training.batch_size,
            ckpt_dir=ckpt_dir,
            **train_kwargs,
        )

        all_results[exp] = result

        # Save OOF predictions
        np.savez(
            oof_dir / f"{exp}_oof.npz",
            eta=result["oof_eta"],
            surv=result["oof_surv"],
            event=result["oof_event"],
            fold_cindices=result["fold_cindices"],
        )
        print(f"  Saved OOF predictions to {oof_dir}/{exp}_oof.npz")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for exp, res in all_results.items():
        print(f"  {exp:<6} OOF C-index: {res['oof_cindex']:.4f}  "
              f"  elapsed: {res['elapsed_s']/60:.1f} min")

    # Save summary JSON
    summary = {
        exp: {
            "oof_cindex":    float(r["oof_cindex"]),
            "fold_cindices": [float(c) for c in r["fold_cindices"]],
            "elapsed_s":     float(r["elapsed_s"]),
        }
        for exp, r in all_results.items()
    }
    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
