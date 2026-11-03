#!/usr/bin/env python3
"""
End-to-end reproduction script for the NSCLC survival analysis pipeline.

This script orchestrates all steps required to reproduce the results from
scratch, starting from raw TCIA data, and produces a Markdown results table
that matches the thesis values.

What it does
------------
1. Environment check      — Python version, GPU, required packages
2. Unit tests             — pytest tests/  (no data or GPU required)
3. Preprocessing          — CT → R2 numpy cache + COLIPRI embeddings
                            (skipped if --skip-preprocess or caches exist)
4. CV splits              — creates outputs/splits/{5,10}fold.json
5. Training               — runs experiments E1–E17d
                            (or a subset via --exps)
6. Evaluation             — computes OOF C-indices, bootstrap CIs,
                            Wilcoxon tests, IBS → results_table.md

Usage
-----
# Full reproduction from raw TCIA data (GPU recommended):
python reproduce.py \\
    --artifact-dir  /path/to/data/artifacts \\
    --dicom-dir     /path/to/data/raw/NSCLC-Radiomics \\
    --output-dir    outputs/

# Skip preprocessing (data already cached):
python reproduce.py \\
    --artifact-dir  /path/to/data/artifacts \\
    --output-dir    outputs/ \\
    --skip-preprocess

# CNN-only (skip COLIPRI; no HuggingFace access needed):
python reproduce.py \\
    --artifact-dir  /path/to/data/artifacts \\
    --output-dir    outputs/ \\
    --skip-preprocess \\
    --exps E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15

# Evaluate only (training already done):
python reproduce.py \\
    --output-dir outputs/ \\
    --artifact-dir /path/to/data/artifacts \\
    --skip-preprocess \\
    --skip-train

Environment variables (alternatives to flags)
---------------------------------------------
NSCLC_ARTIFACT_DIR   — path to preprocessed data root
NSCLC_OUTPUT_DIR     — path to outputs root
MEDICALNET_CKPT      — path to MedicalNet ResNet-10 checkpoint (.pth)
HF_TOKEN             — HuggingFace token for microsoft/colipri

Expected outputs
----------------
outputs/
  splits/
    cv_splits_5fold.json        (stratified 5-fold for E1–E15)
    cv_splits_10fold.json       (stratified 10-fold for E16–E17)
  checkpoints/
    {exp}_fold{k}.pt            (best model checkpoint per fold)
  oof_predictions/
    {exp}_oof.npz               (OOF risk scores, survival, events)
  training_summary.json         (OOF C-index + elapsed time per experiment)
  evaluation_results.json       (C-indices, bootstrap CIs, Wilcoxon p-values)
  results_table.md              (Markdown table for thesis / paper)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output to the terminal."""
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def _check_package(name: str, import_name: str | None = None) -> bool:
    return importlib.util.find_spec(import_name or name) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Environment check
# ─────────────────────────────────────────────────────────────────────────────

def check_environment(skip_colipri: bool) -> None:
    _print_header("Step 1 — Environment check")

    # Python version
    pv = sys.version_info
    print(f"\n  Python {pv.major}.{pv.minor}.{pv.micro}")
    if pv < (3, 10):
        sys.exit("Python 3.10+ is required.")

    # Core packages
    required = [
        ("torch",         "torch"),
        ("numpy",         "numpy"),
        ("pandas",        "pandas"),
        ("nibabel",       "nibabel"),
        ("skimage",       "skimage"),
        ("sklearn",       "sklearn"),
        ("lifelines",     "lifelines"),
        ("sksurv",        "sksurv"),
        ("scipy",         "scipy"),
        ("matplotlib",    "matplotlib"),
        ("yaml",          "yaml"),
        ("tqdm",          "tqdm"),
    ]
    if not skip_colipri:
        required += [
            ("transformers", "transformers"),
            ("huggingface_hub", "huggingface_hub"),
        ]

    missing = []
    for pkg_name, import_name in required:
        ok = _check_package(pkg_name, import_name)
        status = "✓" if ok else "✗ MISSING"
        print(f"    {pkg_name:<20} {status}")
        if not ok:
            missing.append(pkg_name)

    if missing:
        sys.exit(
            f"\nMissing packages: {', '.join(missing)}\n"
            "Install with:\n"
            "  conda env create -f environment.yml && conda activate nsclc_survival\n"
            "  # or: pip install -r requirements.txt && pip install -e ."
        )

    # GPU
    import torch
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  GPU: not available (CPU-only — training will be slow)")

    print("\n  Environment OK.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(repo_root: Path) -> None:
    _print_header("Step 2 — Unit tests")
    _run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
         check=True)
    print("\n  All tests passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocess(
    artifact_dir: Path,
    dicom_dir: Path | None,
    skip_colipri: bool,
    hf_token: str | None,
) -> None:
    _print_header("Step 3 — Preprocessing")

    cache_dir = artifact_dir / "global_cache_128"
    n_cached  = len(list(cache_dir.glob("*_global.npy"))) if cache_dir.exists() else 0

    pool_dir  = artifact_dir / "colipri_nifti" / "pooled"
    n_pooled  = len(list(pool_dir.glob("*.npy"))) if pool_dir.exists() else 0

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if n_cached >= 422:
        print(f"\n  CT cache: {n_cached} volumes found in {cache_dir} — skipping Stage 1.")
    elif dicom_dir is None:
        sys.exit(
            f"\nCT cache has only {n_cached}/422 volumes and --dicom-dir was not set.\n"
            "Either:\n"
            "  (a) provide --dicom-dir pointing to your TCIA download, or\n"
            "  (b) pass --skip-preprocess if preprocessing is already done."
        )
    else:
        cmd = [
            sys.executable, "scripts/preprocess.py",
            "--stage", "ct",
            "--dicom-dir", str(dicom_dir),
            "--artifact-dir", str(artifact_dir),
        ]
        _run(cmd)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if skip_colipri:
        print("\n  COLIPRI embedding extraction skipped (--skip-colipri or CNN-only run).")
        return

    if n_pooled >= 422:
        print(f"\n  COLIPRI embeddings: {n_pooled} found in {pool_dir} — skipping Stage 2.")
        return

    cmd = [
        sys.executable, "scripts/preprocess.py",
        "--stage", "colipri",
        "--artifact-dir", str(artifact_dir),
    ]
    if hf_token:
        cmd += ["--hf-token", hf_token]
    _run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4+5 — Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    repo_root: Path,
    config_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    exps: list[str] | None,
    medicalnet_ckpt: str | None,
) -> None:
    _print_header("Step 4+5 — CV splits + Training")

    os.environ["NSCLC_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["NSCLC_OUTPUT_DIR"]   = str(output_dir)

    if exps:
        # Train one experiment at a time so we can log progress cleanly
        for exp in exps:
            _print_header(f"  Training {exp}")
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", str(config_path),
                "--exp", exp,
            ]
            if medicalnet_ckpt:
                cmd += ["--medicalnet-ckpt", medicalnet_ckpt]
            _run(cmd)
    else:
        # Train all experiments
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", str(config_path),
        ]
        if medicalnet_ckpt:
            cmd += ["--medicalnet-ckpt", medicalnet_ckpt]
        _run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    config_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    no_ibs: bool,
) -> None:
    _print_header("Step 6 — Evaluation")

    os.environ["NSCLC_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["NSCLC_OUTPUT_DIR"]   = str(output_dir)

    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--config", str(config_path),
        "--out-dir", str(output_dir),
    ]
    if no_ibs:
        cmd.append("--no-ibs")
    _run(cmd)

    table_path = output_dir / "results_table.md"
    if table_path.exists():
        print(f"\n{'─'*70}")
        print(table_path.read_text())
        print(f"{'─'*70}")
        print(f"\n  Results table: {table_path}")

    eval_path = output_dir / "evaluation_results.json"
    if eval_path.exists():
        print(f"  Evaluation JSON: {eval_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="End-to-end NSCLC survival reproduction script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Root of preprocessed data (default: $NSCLC_ARTIFACT_DIR or data/artifacts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root for outputs (default: $NSCLC_OUTPUT_DIR or outputs/)",
    )
    parser.add_argument(
        "--dicom-dir",
        type=Path,
        default=None,
        help="Root of raw TCIA DICOM download — required unless --skip-preprocess",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "default.yaml",
        help="Config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--medicalnet-ckpt",
        default=None,
        help="Path to MedicalNet ResNet-10 checkpoint (E14/E15; or set MEDICALNET_CKPT)",
    )

    # ── Skip flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests (not recommended)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (CT cache and embeddings already exist)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (OOF predictions already exist); go straight to evaluation",
    )
    parser.add_argument(
        "--skip-colipri",
        action="store_true",
        help="Skip COLIPRI embedding extraction and E16/E17 experiments",
    )
    parser.add_argument(
        "--no-ibs",
        action="store_true",
        help="Skip Integrated Brier Score computation (slow)",
    )

    # ── Experiment selection ──────────────────────────────────────────────
    ALL_EXPS = [
        "E1","E2","E3","E4","E5","E6","E7","E8","E9",
        "E10","E11","E12","E13","E14","E15",
        "E16a","E16b","E17a","E17b","E17c","E17d",
    ]
    parser.add_argument(
        "--exps",
        nargs="+",
        choices=ALL_EXPS,
        default=None,
        metavar="EXP",
        help=(
            "Run only specific experiments, e.g. --exps E1 E2 E10. "
            "Default: all experiments."
        ),
    )

    # ── HuggingFace ────────────────────────────────────────────────────────
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for microsoft/colipri (or set HF_TOKEN)",
    )

    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────
    artifact_dir = (
        args.artifact_dir
        or Path(os.environ.get("NSCLC_ARTIFACT_DIR", "data/artifacts"))
    ).expanduser().resolve()

    output_dir = (
        args.output_dir
        or Path(os.environ.get("NSCLC_OUTPUT_DIR", "outputs"))
    ).expanduser().resolve()

    medicalnet_ckpt = (
        args.medicalnet_ckpt
        or os.environ.get("MEDICALNET_CKPT")
    )
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Decide whether COLIPRI experiments are included
    colipri_exps = {"E16a", "E16b", "E17a", "E17b", "E17c", "E17d"}
    selected_exps = args.exps or ALL_EXPS
    has_colipri   = any(e in colipri_exps for e in selected_exps)
    skip_colipri  = args.skip_colipri or not has_colipri

    # ── Banner ─────────────────────────────────────────────────────────────
    _print_header("NSCLC Survival Prediction — Reproduction")
    print(f"\n  Repo        : {repo_root}")
    print(f"  Artifact dir: {artifact_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Config      : {args.config}")
    print(f"  Experiments : {selected_exps}")
    if medicalnet_ckpt:
        print(f"  MedicalNet  : {medicalnet_ckpt}")
    if not skip_colipri:
        print(f"  HF token    : {'set' if hf_token else 'not set (may fail for gated model)'}")

    t0 = time.time()

    # ── Run steps ──────────────────────────────────────────────────────────
    if not args.skip_tests:
        run_tests(repo_root)

    if not args.skip_preprocess:
        run_preprocess(
            artifact_dir  = artifact_dir,
            dicom_dir     = args.dicom_dir,
            skip_colipri  = skip_colipri,
            hf_token      = hf_token,
        )
    else:
        _print_header("Step 3 — Preprocessing  [SKIPPED]")
        print("  Assuming data already preprocessed in", artifact_dir)

    if not args.skip_train:
        run_training(
            repo_root       = repo_root,
            config_path     = args.config,
            artifact_dir    = artifact_dir,
            output_dir      = output_dir,
            exps            = selected_exps if args.exps else None,
            medicalnet_ckpt = medicalnet_ckpt,
        )
    else:
        _print_header("Steps 4+5 — Training  [SKIPPED]")
        print("  Assuming OOF predictions exist in", output_dir / "oof_predictions")

    run_evaluation(
        config_path  = args.config,
        output_dir   = output_dir,
        artifact_dir = artifact_dir,
        no_ibs       = args.no_ibs,
    )

    # ── Done ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_header("Reproduction complete")
    print(f"\n  Total elapsed: {elapsed/3600:.1f} h")
    print(f"  Results table: {output_dir / 'results_table.md'}")
    print(f"  Eval JSON    : {output_dir / 'evaluation_results.json'}")
    print()


if __name__ == "__main__":
    main()
