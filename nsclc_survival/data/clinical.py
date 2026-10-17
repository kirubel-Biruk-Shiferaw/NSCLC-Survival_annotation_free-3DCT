"""
Clinical data loading and feature engineering for the Lung 1 cohort.

The TCIA Lung 1 CSV contains (n=422):
  PatientID, Survival.time (days), deadstatus.event (0/1),
  clinical.T.Stage (float 1.0–4.0), Clinical.N.Stage, Clinical.M.Stage,
  Histology (string), age, gender

Key implementation decisions
-----------------------------
1. Stage columns are stored as FLOATS in this CSV (1.0, 2.0, …), not strings.
   encode_numeric_stage() handles this correctly. A string-map would produce
   all-NaN one-hot vectors silently.

2. Age is z-scored INSIDE the CV loop using only training fold statistics.
   Global z-scoring leaks test-set information.  Call normalise_age_per_fold()
   on every fold before building datasets; the 'age_z' column written here is a
   placeholder (= 0.0) that gets overwritten per fold.

3. Patients with missing stage (3/422) are RETAINED with zero one-hot vectors.
   Exclusion would introduce selection bias toward patients with complete
   clinical documentation.

4. Gender: 290 male (68.7 %), 132 female (31.3 %), no missing values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_numeric_stage(series: pd.Series) -> pd.Series:
    """
    Convert float stage values (1.0, 2.0, 3.0, 4.0) to nullable Int64.

    The Lung 1 CSV stores T/N/M stage as floats, not strings.
    Returns Int64 (nullable) with pd.NA for missing/non-numeric entries.
    """
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def encode_histology(series: pd.Series) -> pd.Series:
    """
    Map histology strings → numeric codes via longest-first substring match.
    Returns float with NaN for unknown/missing.

    Code mapping: 0=squamous, 1=adenocarcinoma, 2=large_cell, 3=nos/other
    """
    HIST = {
        "squamous":       0,
        "adenocarcinoma": 1,
        "large cell":     2,
        "nos":            3,
        "other":          3,
        "not otherwise":  3,
        "large_cell":     2,
    }
    keys = sorted(HIST.keys(), key=len, reverse=True)  # longest first

    def find(x: str) -> float:
        x = str(x).strip().lower()
        if x in ("", "nan", "none", "unknown", "-", "na"):
            return np.nan
        for k in keys:
            if k in x:
                return float(HIST[k])
        return np.nan

    return series.map(find).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold age normalisation  (call INSIDE the CV loop)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_age_per_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Z-score age using ONLY training fold statistics (prevents leakage).

    Parameters
    ----------
    df        : cohort DataFrame (must have 'age_raw' column)
    train_idx : integer indices of the training patients in this fold

    Returns
    -------
    A copy of df with 'age_z' updated for all rows using train-fold μ and σ.
    """
    df = df.copy()
    ages = df.iloc[train_idx]["age_raw"]
    mu   = float(ages.mean())
    sig  = float(ages.std())
    df["age_z"] = (df["age_raw"] - mu) / (sig + 1e-6)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "age_z",
    "t1", "t2", "t3", "t4",
    "n0", "n1", "n2", "n3",
    "m0", "m1",
    "hist_squamous", "hist_adenocarcinoma", "hist_large_cell", "hist_nos",
    "gender_male",
]


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and encode the Lung 1 clinical CSV.

    Returns
    -------
    DataFrame with columns:
      PatientID, surv (days), status (0/1),
      age_raw, age_z (placeholder — overwritten per fold),
      t1..t4, n0..n3, m0..m1 (one-hot stage),
      hist_squamous..hist_nos (one-hot histology),
      gender_male (binary)

    Patients with missing surv/status are dropped (~0 in Lung 1).
    Patients with missing stage are retained with zero one-hot vectors.
    """
    df = pd.read_csv(csv_path)

    # Harmonise column names (different CSV exports use different spellings)
    df.rename(
        columns={
            "Survival.time":    "surv",
            "deadstatus.event": "status",
        },
        inplace=True,
        errors="ignore",
    )

    # Survival time and event indicator
    df["surv"]   = pd.to_numeric(df["surv"],   errors="coerce")
    df["status"] = pd.to_numeric(df["status"], errors="coerce")
    n_before = len(df)
    df.dropna(subset=["surv", "status"], inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} patients with missing surv/status")

    # ── Stage encoding ────────────────────────────────────────────────────────
    df["t_num"] = encode_numeric_stage(df["clinical.T.Stage"])
    df["n_num"] = encode_numeric_stage(df["Clinical.N.Stage"])
    df["m_num"] = encode_numeric_stage(df["Clinical.M.Stage"])

    for lvl, col in [(1, "t1"), (2, "t2"), (3, "t3"), (4, "t4")]:
        df[col] = (df["t_num"] == lvl).astype(float)
    for lvl, col in [(0, "n0"), (1, "n1"), (2, "n2"), (3, "n3")]:
        df[col] = (df["n_num"] == lvl).astype(float)
    for lvl, col in [(0, "m0"), (1, "m1")]:
        df[col] = (df["m_num"] == lvl).astype(float)

    # ── Histology ─────────────────────────────────────────────────────────────
    df["hist_num"]              = encode_histology(df["Histology"])
    df["hist_squamous"]         = (df["hist_num"] == 0).astype(float)
    df["hist_adenocarcinoma"]   = (df["hist_num"] == 1).astype(float)
    df["hist_large_cell"]       = (df["hist_num"] == 2).astype(float)
    df["hist_nos"]              = (df["hist_num"] == 3).astype(float)

    # ── Age ───────────────────────────────────────────────────────────────────
    df["age_raw"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_z"]   = 0.0  # placeholder — overwritten inside CV loop

    # ── Sex / gender ──────────────────────────────────────────────────────────
    df["gender_male"] = (
        df["gender"].str.strip().str.lower()
        .map({"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0})
        .fillna(0.5)  # 0.5 for genuinely unknown (none in Lung 1)
    )

    return df.reset_index(drop=True)
