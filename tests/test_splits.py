"""
Unit tests for cross-validation split generation.

All experiments (E1–E17) use a single 5-fold stratified CV split.

Tests
-----
1. 5-fold: correct total patient count.
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
def spl_5(fake_df):
    return make_splits(fake_df, save_dir=None)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitCounts:

    def test_five_fold_total(self, fake_df, spl_5):
        n = len(fake_df)
        total_test = sum(len(test) for _, test in spl_5)
        assert total_test == n, f"5-fold: expected {n} total test patients, got {total_test}"

    def test_five_fold_count(self, spl_5):
        assert len(spl_5) == 5


class TestSplitDisjointness:

    def test_no_overlap_per_fold(self, fake_df, spl_5):
        for i, (tv, test) in enumerate(spl_5):
            overlap = set(tv) & set(test)
            assert len(overlap) == 0, \
                f"Fold {i+1}: {len(overlap)} patients in both tv and test"


class TestCoverage:

    def test_every_patient_in_exactly_one_test_fold(self, fake_df, spl_5):
        n = len(fake_df)
        counts = np.zeros(n, dtype=int)
        for _, test in spl_5:
            counts[test] += 1
        assert np.all(counts == 1), \
            f"Some patients appear {np.unique(counts)} times in test folds"


class TestReproducibility:

    def test_same_seed_same_splits(self, fake_df):
        s5a = make_splits(fake_df, save_dir=None)
        s5b = make_splits(fake_df, save_dir=None)
        for (tv_a, te_a), (tv_b, te_b) in zip(s5a, s5b):
            assert np.array_equal(tv_a, tv_b)
            assert np.array_equal(te_a, te_b)


class TestSaveLoad:

    def test_roundtrip(self, fake_df, spl_5):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_splits(fake_df, save_dir=Path(tmpdir))
            spl_5_loaded = load_splits(Path(tmpdir))

        for (tv_a, te_a), (tv_b, te_b) in zip(spl_5, spl_5_loaded):
            assert np.array_equal(tv_a, tv_b)
            assert np.array_equal(te_a, te_b)


class TestStratification:

    def test_event_rates_balanced(self, fake_df, spl_5):
        """Per-fold event rates should be within ±0.15 of global rate."""
        global_rate = fake_df["status"].mean()
        for i, (_, test) in enumerate(spl_5):
            fold_rate = fake_df.iloc[test]["status"].mean()
            diff = abs(fold_rate - global_rate)
            assert diff < 0.15, \
                (f"Fold {i+1}: fold event rate {fold_rate:.3f} deviates "
                 f"{diff:.3f} from global {global_rate:.3f}")
