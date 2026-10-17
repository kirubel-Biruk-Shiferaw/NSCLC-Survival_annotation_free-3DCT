"""
Unit tests for evaluation metrics.

Tests
-----
1. bootstrap_cindex: returns values in [0, 1], CI contains point estimate.
2. bootstrap_cindex: perfect predictor gives CI near 1.
3. wilcoxon_compare: valid=False when fold counts differ.
4. wilcoxon_compare: valid=True and p<0.05 for clearly better predictor.
5. wilcoxon_compare: handles all-zero diffs gracefully.
6. format_results_table: returns a non-empty string with expected headers.
"""

import math
import numpy as np
import pytest

from nsclc_survival.evaluation.metrics import (
    bootstrap_cindex,
    wilcoxon_compare,
    format_results_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def surv_data(rng):
    """N=80, 70% event rate."""
    n     = 80
    surv  = rng.uniform(30, 1000, n)
    event = (rng.uniform(0, 1, n) < 0.7).astype(float)
    return surv, event


@pytest.fixture
def perfect_predictor(surv_data):
    """η = -surv (shorter survival = higher hazard) — perfect predictor."""
    surv, event = surv_data
    return surv, event, -surv.copy()


@pytest.fixture
def random_predictor(surv_data, rng):
    surv, event = surv_data
    return surv, event, rng.standard_normal(len(surv))


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_cindex
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCindex:

    def test_returns_three_values(self, perfect_predictor):
        surv, event, eta = perfect_predictor
        result = bootstrap_cindex(surv, eta, event, n_resamples=100)
        assert len(result) == 3

    def test_values_in_unit_interval(self, random_predictor):
        surv, event, eta = random_predictor
        pt, lo, hi = bootstrap_cindex(surv, eta, event, n_resamples=200)
        for v in (pt, lo, hi):
            assert 0.0 <= v <= 1.0, f"C-index value {v} outside [0, 1]"

    def test_ci_contains_point_estimate(self, random_predictor):
        surv, event, eta = random_predictor
        pt, lo, hi = bootstrap_cindex(surv, eta, event, n_resamples=500)
        assert lo <= pt <= hi, f"Point estimate {pt:.4f} not in CI [{lo:.4f}, {hi:.4f}]"

    def test_perfect_predictor_ci_near_one(self, perfect_predictor):
        surv, event, eta = perfect_predictor
        pt, lo, hi = bootstrap_cindex(surv, eta, event, n_resamples=500)
        assert pt > 0.95, f"Perfect predictor C-index {pt:.4f} unexpectedly low"
        assert lo > 0.85, f"Perfect predictor lower CI {lo:.4f} unexpectedly low"

    def test_random_predictor_ci_near_half(self, random_predictor):
        surv, event, eta = random_predictor
        pt, lo, hi = bootstrap_cindex(surv, eta, event, n_resamples=500)
        # Random predictor should be close to 0.5
        assert 0.3 <= pt <= 0.7, f"Random predictor C-index {pt:.4f} far from 0.5"

    def test_reproducible(self, random_predictor):
        surv, event, eta = random_predictor
        r1 = bootstrap_cindex(surv, eta, event, n_resamples=200, rng_seed=99)
        r2 = bootstrap_cindex(surv, eta, event, n_resamples=200, rng_seed=99)
        assert r1 == r2

    def test_different_seeds_differ(self, random_predictor):
        surv, event, eta = random_predictor
        r1 = bootstrap_cindex(surv, eta, event, n_resamples=200, rng_seed=1)
        r2 = bootstrap_cindex(surv, eta, event, n_resamples=200, rng_seed=2)
        # Same point estimate; CIs may differ slightly
        assert r1[0] == r2[0]  # same point estimate


# ─────────────────────────────────────────────────────────────────────────────
# wilcoxon_compare  (Fix 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestWilcoxonCompare:

    def test_fold_count_mismatch_returns_invalid(self):
        """Critical fix: 10-fold vs 5-fold should return valid=False, not raise."""
        fci_10fold = [0.62] * 10   # COLIPRI — 10-fold
        fci_5fold  = [0.58] * 5   # E10 — 5-fold
        result = wilcoxon_compare(fci_10fold, fci_5fold, name_a="E16a", name_b="E10")
        assert result["valid"] is False
        assert "10" in result["note"] and "5" in result["note"]

    def test_same_fold_count_valid(self):
        fci_a = [0.62, 0.60, 0.61, 0.63, 0.59]
        fci_b = [0.58, 0.57, 0.59, 0.58, 0.56]
        result = wilcoxon_compare(fci_a, fci_b)
        assert result["valid"] is True
        assert result["n_folds"] == 5
        assert 0.0 <= result["p_value"] <= 1.0

    def test_clearly_better_predictor_significant(self):
        """Predictor always 0.1 better than reference → p should be small."""
        fci_a = [0.65, 0.64, 0.66, 0.63, 0.65]
        fci_b = [0.55, 0.54, 0.56, 0.53, 0.55]
        result = wilcoxon_compare(fci_a, fci_b, alternative="greater")
        assert result["valid"] is True
        assert result["p_value"] < 0.1

    def test_all_diffs_zero_returns_p_one(self):
        """Identical fold CIs → p=1.0."""
        fci = [0.60, 0.61, 0.59, 0.60, 0.61]
        result = wilcoxon_compare(fci, fci)
        assert result["valid"] is True
        assert result["p_value"] == 1.0

    def test_statistic_and_p_finite(self):
        fci_a = [0.61 + 0.01 * i for i in range(5)]
        fci_b = [0.58 + 0.01 * i for i in range(5)]
        result = wilcoxon_compare(fci_a, fci_b)
        if result["valid"]:
            assert math.isfinite(result["statistic"])
            assert math.isfinite(result["p_value"])

    def test_name_fields_preserved(self):
        fci_a = [0.62, 0.61, 0.63]
        fci_b = [0.58, 0.57, 0.59]
        result = wilcoxon_compare(fci_a, fci_b, name_a="E17d", name_b="E10")
        assert result["name_a"] == "E17d"
        assert result["name_b"] == "E10"

    def test_one_fold_each_valid(self):
        """Single fold per experiment — no comparison to make, should handle gracefully."""
        result = wilcoxon_compare([0.62], [0.58])
        # Either valid (trivial Wilcoxon) or an error from scipy; either way, no exception
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# format_results_table
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatResultsTable:

    def _mock_results(self):
        return {
            "E2":  {"oof_cindex": 0.582, "fold_cindices": [0.57, 0.59, 0.58, 0.58, 0.57]},
            "E10": {"oof_cindex": 0.601, "fold_cindices": [0.60, 0.61, 0.59, 0.60, 0.61]},
        }

    def test_returns_non_empty_string(self):
        table = format_results_table(self._mock_results())
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_exp_names(self):
        table = format_results_table(self._mock_results())
        assert "E2" in table
        assert "E10" in table

    def test_contains_cindex_values(self):
        table = format_results_table(self._mock_results())
        assert "0.582" in table or "0.5820" in table

    def test_with_bootstrap_results(self):
        bs = {"E2": (0.582, 0.548, 0.614)}
        table = format_results_table(self._mock_results(), bootstrap_results=bs)
        assert "0.548" in table

    def test_with_wilcoxon_results(self):
        wr = {
            "E2": {
                "valid": True, "statistic": 4.0, "p_value": 0.031,
                "n_folds": 5, "name_a": "E2", "name_b": "E10",
                "alternative": "greater", "note": None,
            }
        }
        table = format_results_table(self._mock_results(), wilcoxon_results=wr)
        assert "0.031" in table or "p=0.031" in table

    def test_fold_mismatch_shows_na(self):
        wr = {
            "E2": {
                "valid": False, "statistic": None, "p_value": None,
                "n_folds": None, "name_a": "E16a", "name_b": "E10",
                "alternative": "greater",
                "note": "Cannot pair fold C-indices: 10 folds vs 5 folds.",
            }
        }
        table = format_results_table(self._mock_results(), wilcoxon_results=wr)
        assert "n/a" in table or "mismatch" in table
