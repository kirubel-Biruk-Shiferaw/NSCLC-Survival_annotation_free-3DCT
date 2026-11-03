"""
Cross-validation split generation and persistence.

All experiments (E1–E17) use a single 5-fold stratified CV split:
  spl_5  — 5-fold  (all experiments: CNN, clinical, and COLIPRI)

Stratified splitting on the event indicator ensures balanced censoring rates
across folds (~84 test patients / fold on the 422-patient Lung 1 cohort).

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
    save_dir: Path | None = None,
) -> Split:
    """
    Create stratified 5-fold CV splits for all experiments (E1–E17).

    Parameters
    ----------
    df          : cohort DataFrame with 'status' column
    n_folds_5   : number of outer folds (default 5)
    save_dir    : if not None, save JSON file here for reproducibility

    Returns
    -------
    spl_5 — list of (tv_idx, test_idx) arrays, one per fold
    """
    labels = df["status"].values
    skf_5  = StratifiedKFold(n_splits=n_folds_5, shuffle=True, random_state=SEED)
    spl_5  = list(skf_5.split(df, labels))

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _save_splits(spl_5, save_dir, "cv_splits_5fold.json")
        print(f"Splits saved to {save_dir}")

    return spl_5


def _save_splits(splits: Split, directory: Path, filename: str) -> None:
    out = [
        {"tv": tv.tolist(), "test": test.tolist()}
        for tv, test in splits
    ]
    (Path(directory) / filename).write_text(json.dumps(out, indent=2))


def load_splits(directory: Path) -> Split:
    """
    Reload saved 5-fold splits from JSON in directory.
    Use this to guarantee identical test sets for interpretability analyses.
    """
    data = json.loads((Path(directory) / "cv_splits_5fold.json").read_text())
    return [(np.array(d["tv"]), np.array(d["test"])) for d in data]


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
