#!/usr/bin/env python3
"""
GradCAM interpretability and ATR computation for trained survival models.

Usage
-----
python scripts/interpret.py \\
    --config configs/default.yaml \\
    --exp E3 \\
    --fold 1 \\
    --out-dir outputs/

Outputs
-------
outputs/gradcam/E3_fold01_gradcam.npz  — (N_test, D, H, W) heatmap array
outputs/gradcam/E3_atr.json            — per-patient ATR values
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nsclc_survival.utils.config import load_config
from nsclc_survival.utils.seed import set_seed
from nsclc_survival.data.clinical import load_labels, normalise_age_per_fold
from nsclc_survival.data.splits import load_splits
from nsclc_survival.data.datasets import DS_Base, DS_Clinical
from nsclc_survival.interpretability.gradcam import GradCAM3D, compute_atr


# Model class / target layer map for supported experiments
MODEL_MAP = {
    "E2":  ("cnn", "nsclc_survival.models.cnn", "CNNSurv_E2",  "enc[7]"),
    "E3":  ("cnn", "nsclc_survival.models.cnn", "CNNSurv_E3",  "enc[7]"),
    "E6":  ("cnn", "nsclc_survival.models.cnn", "CNNSurv_E6",  "enc[7]"),
    "E9":  ("cnn", "nsclc_survival.models.cnn", "CNNSurv_E9",  "img_enc[7]"),
    "E17c": ("colipri", "nsclc_survival.models.colipri",
             "COLIPRI_Patch_Conv_AttnPool_E17c", "pool"),
    "E17d": ("colipri", "nsclc_survival.models.colipri",
             "COLIPRI_Patch_Conv_AttnPool_Clinical_E17d", "pool"),
}


def get_target_layer(model, layer_spec: str):
    """Resolve a dot-notation layer spec (e.g. 'enc[7]') to an nn.Module."""
    parts = layer_spec.replace("[", ".").replace("]", "").split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--exp",     required=True, help="E.g. E3, E17d")
    parser.add_argument("--fold",    type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    out_dir  = Path(args.out_dir or cfg.output.output_dir)
    cam_dir  = out_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir   = out_dir / "checkpoints"
    splits_dir = out_dir / "splits"

    csv_path = Path(cfg.data.artifact_dir) / cfg.data.csv_filename
    df  = load_labels(csv_path)
    spl_5, spl_10 = load_splits(splits_dir)

    exp = args.exp
    if exp not in MODEL_MAP:
        print(f"ERROR: {exp} not in interpretability model map.")
        print(f"Supported: {list(MODEL_MAP.keys())}")
        sys.exit(1)

    _, module_name, cls_name, layer_spec = MODEL_MAP[exp]

    # ── Load model ────────────────────────────────────────────────────────
    import importlib
    module = importlib.import_module(module_name)
    model_cls = getattr(module, cls_name)
    model = model_cls()

    ckpt_path = ckpt_dir / f"{exp}_fold{args.fold:02d}.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_layer = get_target_layer(model, layer_spec)

    # ── Test set for this fold ─────────────────────────────────────────────
    colipri_exps = {"E16a","E16b","E17a","E17b","E17c","E17d"}
    splits = spl_10 if exp in colipri_exps else spl_5
    fold_idx = args.fold - 1
    tv_idx, test_idx = splits[fold_idx]

    df_fold = normalise_age_per_fold(df, tv_idx)
    vol_dir = Path(cfg.data.artifact_dir) / "R2"
    ds_test = DS_Base(df=df_fold, indices=test_idx, vol_dir=vol_dir, augment=False)

    gradcam = GradCAM3D(model, target_layer)

    # ── Compute heatmaps ──────────────────────────────────────────────────
    print(f"Computing GradCAM for {exp} fold {args.fold}, {len(ds_test)} patients…")
    heatmaps = []
    pids     = []

    for i in range(len(ds_test)):
        sample = ds_test[i]
        vol_t  = sample["vol"].unsqueeze(0).to(device)
        clin_t = sample.get("clinical")
        if clin_t is not None:
            clin_t = clin_t.unsqueeze(0).to(device)

        hmap = gradcam(vol_t, clinical=clin_t)   # (1, D, H, W)
        heatmaps.append(hmap.squeeze(0).cpu().numpy())
        pids.append(sample.get("pid", f"patient_{i}"))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds_test)}")

    heatmaps = np.stack(heatmaps, axis=0)  # (N, D, H, W)

    # Save heatmaps
    cam_path = cam_dir / f"{exp}_fold{args.fold:02d}_gradcam.npz"
    np.savez_compressed(str(cam_path), heatmaps=heatmaps, pids=pids)
    print(f"Saved heatmaps: {cam_path}")

    # ── ATR computation (if GTV masks available) ──────────────────────────
    gtv_dir = Path(cfg.data.artifact_dir) / cfg.data.get("gtv_dir", "GTV_masks")
    if gtv_dir.exists():
        atrs = []
        for i, pid in enumerate(pids):
            mask_path = gtv_dir / f"{pid}_gtv.npy"
            if mask_path.exists():
                gtv_mask = np.load(str(mask_path))
                atr_val  = compute_atr(heatmaps[i], gtv_mask)
            else:
                atr_val = float("nan")
            atrs.append(atr_val)

        atr_dict = dict(zip(pids, [float(a) for a in atrs]))
        atr_path = cam_dir / f"{exp}_atr.json"
        atr_path.write_text(json.dumps(atr_dict, indent=2))

        valid_atrs = [a for a in atrs if not np.isnan(a)]
        if valid_atrs:
            print(f"ATR: mean={np.mean(valid_atrs):.3f}  "
                  f"median={np.median(valid_atrs):.3f}  "
                  f"n={len(valid_atrs)}/{len(atrs)}")
        print(f"ATR values saved to {atr_path}")
    else:
        print(f"GTV masks not found at {gtv_dir} — skipping ATR computation.")
        print("To compute ATR, place per-patient GTV masks as {pid}_gtv.npy "
              f"in {gtv_dir}")


if __name__ == "__main__":
    main()
