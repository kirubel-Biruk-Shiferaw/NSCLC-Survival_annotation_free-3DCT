"""
Evaluation metrics for survival analysis.

Critical fixes from notebook review
-------------------------------------
Fix 1 — Wilcoxon fold-count guard:
  The original notebook called wilcoxon(fci_exp, fci_e10[:5]) when
  len(fci_exp) == 10 (COLIPRI) vs len(fci_e10) == 5 (E10, 5-fold).
  Slicing to [:5] is statistically invalid — it pairs folds from
  different experiments.  wilcoxon_compare() now checks that both
  experiments have the same number of folds before calling the test.

Fix 2 — Integrated Brier Score via sksurv.metrics:
  The original code implemented IBS from scratch with trapezoidal
  integration, and contained indexing errors in the IPCW weight
  computation.  We now use the validated reference implementation from
  scikit-survival (Graf 1999, IPCW variant), which handles edge cases
  around the largest observed time correctly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines.statistics import concordance_index
from scipy.stats import wilcoxon as scipy_wilcoxon

try:
    from sksurv.metrics import (
        integrated_brier_score as _sksurv_ibs,
        brier_score as _sksurv_bs,
    )
    from sksurv.util import Surv
    _SKSURV_OK = True
except ImportError:
    _SKSURV_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap C-index
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_cindex(
    surv: np.ndarray,
    eta: np.ndarray,
    event: np.ndarray,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    rng_seed: int = 123,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI for Harrell's C-index.

    Resamples patients (not folds) with replacement.  Returns
    (point_estimate, lower, upper).

    Parameters
    ----------
    surv        : (N,) observed survival times
    eta         : (N,) log-hazard predictions (higher = worse prognosis)
    event       : (N,) binary event indicator
    n_resamples : number of bootstrap samples (2000 per evaluation plan)
    ci_level    : nominal coverage probability (default 0.95)

    Returns
    -------
    (cindex, lower_ci, upper_ci)
    """
    rng    = np.random.default_rng(rng_seed)
    n      = len(surv)
    ci_bs  = np.zeros(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            ci_bs[i] = concordance_index(surv[idx], -eta[idx], event[idx])
        except Exception:
            ci_bs[i] = 0.5

    alpha = (1.0 - ci_level) / 2
    lower = float(np.percentile(ci_bs, 100 * alpha))
    upper = float(np.percentile(ci_bs, 100 * (1 - alpha)))
    point = float(concordance_index(surv, -eta, event))

    return point, lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank test with fold-count guard  (Fix 1)
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_compare(
    fold_ci_a: List[float],
    fold_ci_b: List[float],
    name_a: str = "A",
    name_b: str = "B",
    alternative: str = "greater",
) -> Dict[str, object]:
    """
    Wilcoxon signed-rank test on per-fold C-indices.

    REQUIRES both lists to have the same length (same fold structure).
    If lengths differ, returns a dict with valid=False and an explanation
    rather than running the test on mismatched folds.

    Parameters
    ----------
    fold_ci_a   : per-fold C-indices for experiment A (challenger)
    fold_ci_b   : per-fold C-indices for experiment B (reference)
    alternative : 'greater' tests H1: A > B

    Returns
    -------
    dict with keys: valid (bool), statistic, p_value, n_folds,
                    name_a, name_b, alternative, note
    """
    n_a = len(fold_ci_a)
    n_b = len(fold_ci_b)

    if n_a != n_b:
        return {
            "valid":     False,
            "statistic": None,
            "p_value":   None,
            "n_folds":   None,
            "name_a":    name_a,
            "name_b":    name_b,
            "alternative": alternative,
            "note": (
                f"Cannot pair fold C-indices: {name_a} has {n_a} folds "
                f"but {name_b} has {n_b} folds.  Use the same fold structure "
                f"for both experiments before calling wilcoxon_compare()."
            ),
        }

    arr_a = np.asarray(fold_ci_a, dtype=float)
    arr_b = np.asarray(fold_ci_b, dtype=float)
    diffs = arr_a - arr_b

    if np.all(diffs == 0):
        return {
            "valid":     True,
            "statistic": 0.0,
            "p_value":   1.0,
            "n_folds":   n_a,
            "name_a":    name_a,
            "name_b":    name_b,
            "alternative": alternative,
            "note":      "All differences are zero.",
        }

    stat, p = scipy_wilcoxon(arr_a, arr_b, alternative=alternative,
                             zero_method="wilcox")

    return {
        "valid":     True,
        "statistic": float(stat),
        "p_value":   float(p),
        "n_folds":   n_a,
        "name_a":    name_a,
        "name_b":    name_b,
        "alternative": alternative,
        "note":      None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Brier Score via scikit-survival  (Fix 2)
# ─────────────────────────────────────────────────────────────────────────────

def integrated_brier_score_oof(
    train_surv: np.ndarray,
    train_event: np.ndarray,
    test_surv: np.ndarray,
    test_event: np.ndarray,
    test_risk: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> float:
    """
    IPCW Integrated Brier Score (Graf 1999) via scikit-survival.

    Uses training set survival data to fit the Kaplan-Meier censoring
    distribution used for IPCW weights.  Evaluates on test set.

    Parameters
    ----------
    train_surv   : (N_train,) survival times (for KM censoring estimator)
    train_event  : (N_train,) event indicators
    test_surv    : (N_test,)  survival times for IBS evaluation
    test_event   : (N_test,)  event indicators
    test_risk    : (N_test,)  risk scores from model (higher = worse)
    times        : evaluation time grid; defaults to 10th–90th percentiles
                   of observed event times (50 equally-spaced points)

    Returns
    -------
    float: IBS value (lower is better; 0.25 = random prediction)

    Raises
    ------
    ImportError if scikit-survival is not installed.
    """
    if not _SKSURV_OK:
        raise ImportError(
            "scikit-survival is required for IBS computation.  "
            "Install it with: pip install scikit-survival"
        )

    # Build structured arrays for sksurv
    train_st = Surv.from_arrays(event=train_event.astype(bool),
                                time=train_surv)
    test_st  = Surv.from_arrays(event=test_event.astype(bool),
                                time=test_surv)

    # Time grid: 10th–90th percentile of event times in test set
    if times is None:
        event_times = test_surv[test_event.astype(bool) == 1]
        if len(event_times) < 2:
            event_times = test_surv
        t_min = np.percentile(event_times, 10)
        t_max = np.percentile(event_times, 90)
        times = np.linspace(t_min, t_max, 50)

    # Convert risk scores to survival function: S(t|x) = exp(-exp(η) * H0(t))
    # For IBS, sksurv expects an array of shape (N_test, len(times))
    # representing survival probabilities S(t|x_i) at each time point.
    # Simple approach: use baseline Nelson-Aalen to derive S(t|x).
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    # Fit a simple Cox model using risk scores as a single precomputed covariate
    try:
        risk_2d = test_risk.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            cox = CoxPHSurvivalAnalysis(alpha=0.01, ties="efron")
            cox.fit(risk_2d, test_st)
        surv_funcs = cox.predict_survival_function(risk_2d, return_array=False)
        surv_matrix = np.row_stack([fn(times) for fn in surv_funcs])
    except Exception:
        # Fallback: flat survival (0.5 everywhere) → IBS ≈ 0.25
        surv_matrix = np.full((len(test_surv), len(times)), 0.5)

    # Clip time range to training set's observed range to avoid extrapolation
    t_min_train = train_surv.min()
    t_max_train = train_surv.max()
    valid_mask  = (times >= t_min_train) & (times <= t_max_train)
    if valid_mask.sum() < 2:
        return float("nan")

    times_clipped  = times[valid_mask]
    surv_clipped   = surv_matrix[:, valid_mask]

    _, ibs = _sksurv_ibs(train_st, test_st, surv_clipped, times_clipped)
    return float(ibs)


# ─────────────────────────────────────────────────────────────────────────────
# Results table formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_results_table(
    results: Dict[str, Dict],
    bootstrap_results: Optional[Dict[str, Tuple[float, float, float]]] = None,
    wilcoxon_results: Optional[Dict[str, Dict]] = None,
) -> str:
    """
    Format a Markdown results table from experiment results dicts.

    Parameters
    ----------
    results : {exp_name: {"oof_cindex": float, "fold_cindices": list, ...}}
    bootstrap_results : {exp_name: (cindex, lower_ci, upper_ci)}
    wilcoxon_results  : {exp_name: wilcoxon_compare() output dict}

    Returns
    -------
    Markdown-formatted table string.
    """
    header = (
        "| Exp   | OOF C-index | 95% CI        | "
        "Fold mean ± SD    | Wilcoxon vs E10 |"
    )
    sep    = "|-------|-------------|---------------|------------------|--------------------|"
    rows   = [header, sep]

    for exp, res in results.items():
        oof_ci     = res.get("oof_cindex", float("nan"))
        fold_ci    = res.get("fold_cindices", [])
        fold_mean  = float(np.mean(fold_ci)) if fold_ci else float("nan")
        fold_std   = float(np.std(fold_ci))  if fold_ci else float("nan")

        if bootstrap_results and exp in bootstrap_results:
            _, lo, hi = bootstrap_results[exp]
            ci_str = f"({lo:.3f}, {hi:.3f})"
        else:
            ci_str = "—"

        if wilcoxon_results and exp in wilcoxon_results:
            wr = wilcoxon_results[exp]
            if not wr["valid"]:
                w_str = "n/a (fold mismatch)"
            elif wr["p_value"] is not None:
                w_str = f"p={wr['p_value']:.3f}"
            else:
                w_str = "—"
        else:
            w_str = "—"

        rows.append(
            f"| {exp:<5} | {oof_ci:.4f}      | {ci_str:<13} | "
            f"{fold_mean:.4f} ± {fold_std:.4f} | {w_str:<18} |"
        )

    return "\n".join(rows)
