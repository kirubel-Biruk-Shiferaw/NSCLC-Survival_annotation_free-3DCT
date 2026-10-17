"""
Cross-validation split generation and persistence.

Two split objects are created and immediately saved to disk:
  spl_5  — 5-fold  (E1–E15: CNN and clinical experiments)
  spl_10 — 10-fold (E16–E17: COLIPRI experiments, matches Paolo et al. 2026)

Both use stratified splitting on the event indicator for equal censoring rates.

IMPORTANT: These splits must be saved and reloaded for any subsequent analysis
(interpretability, ATR, KM plots) to guarantee that test patients were never
seen during the corresponding training run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..utils.seed import SEED


Split = List[Tuple[np.ndarray, np.ndarray]]


def make_splits(
    df: pd.DataFrame,
    n_folds_5: int = 5,
    n_folds_10: int = 10,
    save_dir: Path | None = None,
) -> Tuple[Split, Split]:
    """
    Create stratified 5-fold and 10-fold CV splits.

    Parameters
    ----------
    df          : cohort DataFrame with 'status' column
    n_folds_5   : outer folds for CNN/clinical experiments (default 5)
    n_folds_10  : outer folds for COLIPRI experiments (default 10)
    save_dir    : if not None, save JSON files here for reproducibility

    Returns
    -------
    (spl_5, spl_10) — each is a list of (tv_idx, test_idx) arrays
    """
    labels = df["status"].values

    skf_5  = StratifiedKFold(n_splits=n_folds_5,  shuffle=True, random_state=SEED)
    skf_10 = StratifiedKFold(n_splits=n_folds_10, shuffle=True, random_state=SEED)

    spl_5  = list(skf_5.split(df, labels))
    spl_10 = list(skf_10.split(df, labels))

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _save_splits(spl_5,  save_dir, "cv_splits_5fold.json")
        _save_splits(spl_10, save_dir, "cv_splits_10fold.json")
        print(f"Splits saved to {save_dir}")

    return spl_5, spl_10


def _save_splits(splits: Split, directory: Path, filename: str) -> None:
    out = [
        {"tv": tv.tolist(), "test": test.tolist()}
        for tv, test in splits
    ]
    (Path(directory) / filename).write_text(json.dumps(out, indent=2))


def load_splits(directory: Path) -> Tuple[Split, Split]:
    """
    Reload saved splits from JSON files in directory.
    Use this to guarantee identical test sets for interpretability analyses.
    """
    def _load(fname):
        data = json.loads((Path(directory) / fname).read_text())
        return [
            (np.array(d["tv"]), np.array(d["test"]))
            for d in data
        ]

    spl_5  = _load("cv_splits_5fold.json")
    spl_10 = _load("cv_splits_10fold.json")
    return spl_5, spl_10


def print_split_stats(df: pd.DataFrame, splits: Split, name: str = "Split") -> None:
    """Print per-fold sizes and event rates for quality control."""
    print(f"{name} verification:")
    for i, (tv, test) in enumerate(splits):
        ev_rate = df.iloc[test]["status"].mean()
        print(
            f"  Fold {i+1:2d}: train/val={len(tv):3d}  "
            f"test={len(test):3d}  event_rate={ev_rate:.3f}"
        )
    print(
        f"  Total: {len(df)}  "
        f"(check: {sum(len(t) for _, t in splits)} across folds)"
    )
