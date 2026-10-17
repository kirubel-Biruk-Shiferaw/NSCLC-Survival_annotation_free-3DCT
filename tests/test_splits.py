"""
Unit tests for cross-validation split generation.

Tests
-----
1. 5-fold and 10-fold: correct total patient counts.
2. No patient appears in both train/val and test for the same fold.
3. Every patient appears in exactly one test fold.
4. Splits are reproducible (same seed → same splits).
5. Save and load round-trip produces identical splits.
6. Per-fold event rates are approximately balanced (stratification check).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nsclc_survival.data.splits import make_splits, load_splits, print_split_stats


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_df():
    """Minimal cohort DataFrame mimicking Lung1 structure."""
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "PatientID": [f"LUNG1-{i:03d}" for i in range(n)],
        "surv":      rng.integers(30, 1000, n).astype(float),
        "status":    rng.integers(0, 2, n).astype(float),  # ~50% events
    })


@pytest.fixture
def splits_5_10(fake_df):
    return make_splits(fake_df, n_folds_5=5, n_folds_10=10, save_dir=None)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitCounts:

    def test_five_fold_total(self, fake_df, splits_5_10):
        spl_5, _ = splits_5_10
        n = len(fake_df)
        total_test = sum(len(test) for _, test in spl_5)
        assert total_test == n, f"5-fold: expected {n} total test patients, got {total_test}"

    def test_ten_fold_total(self, fake_df, splits_5_10):
        _, spl_10 = splits_5_10
        n = len(fake_df)
        total_test = sum(len(test) for _, test in spl_10)
        assert total_test == n, f"10-fold: expected {n} total test patients, got {total_test}"

    def test_five_fold_count(self, splits_5_10):
        spl_5, _ = splits_5_10
        assert len(spl_5) == 5

    def test_ten_fold_count(self, splits_5_10):
        _, spl_10 = splits_5_10
        assert len(spl_10) == 10


class TestSplitDisjointness:

    def test_no_overlap_per_fold_5(self, fake_df, splits_5_10):
        spl_5, _ = splits_5_10
        for i, (tv, test) in enumerate(spl_5):
            overlap = set(tv) & set(test)
            assert len(overlap) == 0, \
                f"5-fold {i+1}: {len(overlap)} patients in both tv and test"

    def test_no_overlap_per_fold_10(self, fake_df, splits_5_10):
        _, spl_10 = splits_5_10
        for i, (tv, test) in enumerate(spl_10):
            overlap = set(tv) & set(test)
            assert len(overlap) == 0, \
                f"10-fold {i+1}: {len(overlap)} patients in both tv and test"


class TestCoverage:

    def test_every_patient_in_exactly_one_test_fold(self, fake_df, splits_5_10):
        for spl, name in zip(splits_5_10, ["5-fold", "10-fold"]):
            n = len(fake_df)
            counts = np.zeros(n, dtype=int)
            for _, test in spl:
                counts[test] += 1
            assert np.all(counts == 1), \
                f"{name}: some patients appear {np.unique(counts)} times in test folds"


class TestReproducibility:

    def test_same_seed_same_splits(self, fake_df):
        s5a, s10a = make_splits(fake_df, save_dir=None)
        s5b, s10b = make_splits(fake_df, save_dir=None)
        for (tv_a, te_a), (tv_b, te_b) in zip(s5a, s5b):
            assert np.array_equal(tv_a, tv_b)
            assert np.array_equal(te_a, te_b)


class TestSaveLoad:

    def test_roundtrip(self, fake_df, splits_5_10):
        spl_5, spl_10 = splits_5_10
        with tempfile.TemporaryDirectory() as tmpdir:
            make_splits(fake_df, save_dir=Path(tmpdir))
            spl_5_loaded, spl_10_loaded = load_splits(Path(tmpdir))

        for (tv_a, te_a), (tv_b, te_b) in zip(spl_5, spl_5_loaded):
            assert np.array_equal(tv_a, tv_b)
            assert np.array_equal(te_a, te_b)

        for (tv_a, te_a), (tv_b, te_b) in zip(spl_10, spl_10_loaded):
            assert np.array_equal(tv_a, tv_b)
            assert np.array_equal(te_a, te_b)


class TestStratification:

    def test_event_rates_balanced(self, fake_df, splits_5_10):
        """Per-fold event rates should be within ±0.15 of global rate."""
        global_rate = fake_df["status"].mean()
        for spl, k in zip(splits_5_10, [5, 10]):
            for i, (_, test) in enumerate(spl):
                fold_rate = fake_df.iloc[test]["status"].mean()
                diff = abs(fold_rate - global_rate)
                assert diff < 0.15, \
                    (f"{k}-fold {i+1}: fold event rate {fold_rate:.3f} deviates "
                     f"{diff:.3f} from global {global_rate:.3f}")
