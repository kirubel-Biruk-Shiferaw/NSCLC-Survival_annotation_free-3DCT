"""
PyTorch Dataset classes for all NSCLC survival experiments.

All datasets preload the full R2 cache (~2.6 GB) into RAM on first use.
Subsequent instantiations (different folds, different experiments) share the
same cache via class-level dictionaries, so the I/O overhead is paid once.

Dataset hierarchy
-----------------
DS_Base           — standard R2 volumes (64×128×128 float32), flip + brightness
DS_HeavyAug       — DS_Base + noise + gamma (E8)
DS_LungMasked     — DS_Base + zero non-lung voxels via HU threshold (E7)
DS_Clinical       — DS_Base + 16-dim clinical vector (E9, E12 uses DS_LungCrop_HU)
DS_ClinicalOnly   — clinical features only, no CT (E10, E11)
DS_LungCrop_HU    — HU-threshold bounding-box crop resized to VOL_SHAPE (E12)
DS_MAAS           — 3-region axial decomposition (E13)
DS_COLIPRI_Pooled — 768-dim pooled COLIPRI embeddings (E16a/b)
DS_COLIPRI_Patch  — 864×24³ COLIPRI patch tensors (E17a/b/c/d)

Sign convention
---------------
Model outputs are log-hazard ratios η(x).  Higher η = higher risk = shorter OS.
concordance_index() from lifelines expects higher score = longer survival,
so pass -η when calling it.  All datasets return raw t and e for the loss.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom as scipy_zoom
from torch.utils.data import Dataset

from .clinical import FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract clinical vector
# ─────────────────────────────────────────────────────────────────────────────

def _get_clinical(row: pd.Series) -> np.ndarray:
    """Extract 16-dim clinical feature vector for one patient row."""
    return np.array(
        [
            float(row.get(c, 0.0) if not pd.isna(row.get(c, np.nan)) else 0.0)
            for c in FEATURE_COLS
        ],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standard R2 volume datasets  (E1–E9, E12–E15 use subclasses)
# ─────────────────────────────────────────────────────────────────────────────

class DS_Base(Dataset):
    """
    Standard dataset for pre-cached R2 volumes (64×128×128, float32, [0,1]).

    RAM cache: all 422 volumes loaded once on first instantiation.
    Subsequent calls for the same cache_dir skip I/O entirely.
    ThreadPoolExecutor provides parallel I/O (releases GIL during np.load).

    Augmentation (training only):
      - Random flip along each axis (p=0.5 each): anatomy is flip-invariant
      - Brightness jitter ±0.05: simulates scanner intensity variation
    """

    _ram_cache: dict = {}

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        cache_dir: str | Path,
        augment: bool = False,
    ):
        self.df        = df.iloc[indices].reset_index(drop=True)
        self.augment   = augment
        self.cache_dir = str(cache_dir)
        key            = self.cache_dir

        if key not in DS_Base._ram_cache:
            files = sorted(Path(cache_dir).glob("*_global.npy"))
            print(
                f"  RAM preload: {len(files)} volumes "
                f"from {Path(cache_dir).name}/",
                end="",
                flush=True,
            )
            t0 = time.time()

            def _load_one(p):
                pid = p.stem.replace("_global", "")
                return pid, np.load(str(p)).astype(np.float32)

            ram = {}
            with ThreadPoolExecutor(max_workers=8) as ex:
                for pid, arr in ex.map(_load_one, files):
                    ram[pid] = arr

            DS_Base._ram_cache[key] = ram
            gb = sum(v.nbytes for v in ram.values()) / 1e9
            print(f" {len(ram)} vols  {gb:.2f} GB  {time.time()-t0:.1f}s")

        self.ram = DS_Base._ram_cache[key]

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        """Standard augmentation: random axis flips + brightness jitter."""
        if np.random.rand() > 0.5:
            vol = vol[:, ::-1, :].copy()  # flip H (anterior-posterior)
        if np.random.rand() > 0.5:
            vol = vol[:, :, ::-1].copy()  # flip W (left-right)
        if np.random.rand() > 0.5:
            vol = vol[::-1, :, :].copy()  # flip D (superior-inferior)
        if np.random.rand() > 0.5:
            delta = np.random.uniform(-0.05, 0.05)
            vol = np.clip(vol + delta, 0.0, 1.0).astype(np.float32)
        return vol

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        vol = self.ram[row["PatientID"]].copy()  # copy: don't mutate cache
        if self.augment:
            vol = self._augment(vol)
        return {
            "vol": torch.from_numpy(vol[np.newaxis]).float(),  # (1, D, H, W)
            "t":   torch.tensor(float(row["surv"]),   dtype=torch.float32),
            "e":   torch.tensor(int(row["status"]),   dtype=torch.int32),
            "pid": row["PatientID"],
        }


class DS_HeavyAug(DS_Base):
    """
    E8: Heavy augmentation — noise, gamma, contrast jitter on top of DS_Base.

    Additional augmentations:
      - Gaussian noise σ~U(0, 0.03): simulates quantum noise
      - Random gamma γ~U(0.8, 1.2): simulates contrast variation
    """

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        vol = super()._augment(vol)
        if np.random.rand() > 0.5:
            sigma = np.random.uniform(0.0, 0.03)
            vol = np.clip(
                vol + np.random.normal(0.0, sigma, vol.shape), 0.0, 1.0
            ).astype(np.float32)
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            vol = np.power(np.clip(vol, 1e-6, 1.0), gamma).astype(np.float32)
        return vol


class DS_LungMasked(DS_Base):
    """
    E7: Zero all voxels at or above the HU-threshold lung boundary.

    Threshold 0.35 in [0,1] normalised space ≈ -510 HU:
    separates lung parenchyma + air (below) from tissue (above).

    Unlike DS_LungCrop_HU (E12) which crops, this zeroes the abdominal
    region but preserves the full volume shape.
    """

    LUNG_THRESHOLD = 0.35

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        vol  = item["vol"].numpy().squeeze(0)
        vol[vol >= self.LUNG_THRESHOLD] = 0.0
        item["vol"] = torch.from_numpy(vol[np.newaxis]).float()
        return item


class DS_Clinical(DS_Base):
    """
    DS_Base + 16-dim clinical feature vector. Used for E9 (late fusion) etc.
    """

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["clinical"] = torch.from_numpy(
            _get_clinical(self.df.iloc[idx])
        )
        return item


# ─────────────────────────────────────────────────────────────────────────────
# Clinical-only  (E10, E11 — DeepSurv)
# ─────────────────────────────────────────────────────────────────────────────

class DS_ClinicalOnly(Dataset):
    """
    E10, E11: Clinical features ONLY. Does not load any CT volumes.
    Avoids triggering the 2.6 GB R2 RAM preload for clinical-only experiments.
    """

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, **kw):
        self.df = df.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "clinical": torch.from_numpy(_get_clinical(row)),
            "t":        torch.tensor(float(row["surv"]),  dtype=torch.float32),
            "e":        torch.tensor(int(row["status"]),  dtype=torch.int32),
            "pid":      row["PatientID"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Spatial correction datasets  (E12, E13)
# ─────────────────────────────────────────────────────────────────────────────

class DS_LungCrop_HU(Dataset):
    """
    E12: HU-threshold bounding-box lung crop.

    Algorithm:
      1. Find voxels below threshold 0.35 (lung/air region)
      2. Compute axis-aligned bounding box with 5% proportional margin
      3. Resize to VOL_SHAPE (64×128×128) via trilinear scipy_zoom

    Crops are computed once from the R2 RAM cache and cached.
    Includes 16-dim clinical features for late fusion.

    Note: scipy_zoom factor convention is output/input (scalar or per-axis).
    The code computes target/source per axis, which is correct.
    """

    _crop_cache: dict = {}

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        cache_dir: str | Path,
        augment: bool = False,
        vol_shape: tuple = (64, 128, 128),
        **kw,
    ):
        self.df        = df.iloc[indices].reset_index(drop=True)
        self.augment   = augment
        self.vol_shape = vol_shape
        key            = str(cache_dir) + "_lungcrop_hu"

        if key not in DS_LungCrop_HU._crop_cache:
            # Ensure base R2 cache is available
            base_key = str(cache_dir)
            if base_key not in DS_Base._ram_cache:
                _ = DS_Base(df, np.arange(len(df)), cache_dir)
            base_ram = DS_Base._ram_cache[base_key]

            print("  Computing HU-threshold lung crops...", end="", flush=True)
            t0  = time.time()
            ram = {}
            for pid, vol in base_ram.items():
                mask   = vol < 0.35
                coords = np.argwhere(mask)
                if len(coords) > 0:
                    lo = coords.min(0)
                    hi = coords.max(0) + 1
                    # 5% proportional margin
                    mg = np.maximum(1, (np.array(vol.shape) * 0.05).astype(int))
                    lo = np.maximum(0, lo - mg)
                    hi = np.minimum(np.array(vol.shape), hi + mg)
                    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                else:
                    crop = vol  # fallback: full volume

                zf       = tuple(t / s for t, s in zip(vol_shape, crop.shape))
                ram[pid] = scipy_zoom(crop, zf, order=1).astype(np.float32)

            DS_LungCrop_HU._crop_cache[key] = ram
            print(f" done ({time.time()-t0:.1f}s)")

        self.ram = DS_LungCrop_HU._crop_cache[key]

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5: vol = vol[:, ::-1, :].copy()
        if np.random.rand() > 0.5: vol = vol[:, :, ::-1].copy()
        if np.random.rand() > 0.5: vol = vol[::-1, :, :].copy()
        return vol

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        vol = self.ram[row["PatientID"]].copy()
        if self.augment:
            vol = self._augment(vol)
        return {
            "vol":      torch.from_numpy(vol[np.newaxis]).float(),
            "clinical": torch.from_numpy(_get_clinical(row)),
            "t":        torch.tensor(float(row["surv"]),  dtype=torch.float32),
            "e":        torch.tensor(int(row["status"]),  dtype=torch.int32),
            "pid":      row["PatientID"],
        }


class DS_MAAS(Dataset):
    """
    E13: MAAS 3-region anatomical decomposition.

    Fixed axial fractions applied to the full R2 volume:
      r1 — upper 40% (lung parenchyma + mediastinum + primary tumour)
      r2 — middle 30% (thoracic body composition: muscle, fat, bone)
      r3 — lower 30% (upper abdomen: liver, spleen, stomach)

    Each region is resized to VOL_SHAPE and returned as a separate tensor.

    Limitation: fixed fractions don't account for patient-specific anatomy.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        cache_dir: str | Path,
        augment: bool = False,
        vol_shape: tuple = (64, 128, 128),
        **kw,
    ):
        self.df        = df.iloc[indices].reset_index(drop=True)
        self.augment   = augment
        self.vol_shape = vol_shape

        base_key = str(cache_dir)
        if base_key not in DS_Base._ram_cache:
            _ = DS_Base(df, np.arange(len(df)), cache_dir)
        self.ram = DS_Base._ram_cache[base_key]

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5: vol = vol[:, ::-1, :].copy()
        if np.random.rand() > 0.5: vol = vol[:, :, ::-1].copy()
        if np.random.rand() > 0.5: vol = vol[::-1, :, :].copy()
        return vol

    def _resize(self, v: np.ndarray) -> np.ndarray:
        zf = tuple(t / s for t, s in zip(self.vol_shape, v.shape))
        return scipy_zoom(v, zf, order=1).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        vol = self.ram[row["PatientID"]].copy()
        if self.augment:
            vol = self._augment(vol)
        D  = vol.shape[0]
        r1 = vol[: int(D * 0.40)]
        r2 = vol[int(D * 0.40) : int(D * 0.70)]
        r3 = vol[int(D * 0.70) :]
        return {
            "r1":       torch.from_numpy(self._resize(r1)[np.newaxis]).float(),
            "r2":       torch.from_numpy(self._resize(r2)[np.newaxis]).float(),
            "r3":       torch.from_numpy(self._resize(r3)[np.newaxis]).float(),
            "clinical": torch.from_numpy(_get_clinical(row)),
            "t":        torch.tensor(float(row["surv"]),  dtype=torch.float32),
            "e":        torch.tensor(int(row["status"]),  dtype=torch.int32),
            "pid":      row["PatientID"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# COLIPRI embedding datasets  (E16, E17)
# ─────────────────────────────────────────────────────────────────────────────

class DS_COLIPRI_Pooled(Dataset):
    """
    E16a/b: Pre-computed COLIPRI 768-dim pooled embeddings.

    All 422 embeddings (~2.5 MB total) are preloaded into RAM.
    E16b additionally provides 16-dim clinical features.

    NOTE: COLIPRI was pretrained on CT-RATE (CT + radiology-report pairs) and
    NLST (image-only). Neither corpus includes the NSCLC-Radiomics / Lung1
    cohort (MAASTRO/TCIA), so there is no pretraining/test overlap here.
    Domain shift and single-site evaluation still apply; external validation
    on an independent cohort remains necessary.
    """

    _cache: dict = {}

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        colipri_pool_dir: str | Path,
        use_clin: bool = False,
        **kw,
    ):
        self.df       = df.iloc[indices].reset_index(drop=True)
        self.use_clin = use_clin

        if "pool" not in DS_COLIPRI_Pooled._cache:
            print(
                "  RAM preload: COLIPRI pooled embeddings...",
                end="",
                flush=True,
            )
            t0    = time.time()
            files = list(Path(colipri_pool_dir).glob("*.npy"))
            cache = {}

            def _load(f):
                return f.stem, np.load(str(f)).astype(np.float32)

            with ThreadPoolExecutor(max_workers=8) as ex:
                for pid, arr in ex.map(_load, files):
                    cache[pid] = arr

            DS_COLIPRI_Pooled._cache["pool"] = cache
            mb = sum(v.nbytes for v in cache.values()) / 1e6
            print(f" {len(cache)} files  {mb:.1f} MB  {time.time()-t0:.1f}s")

        self.cache = DS_COLIPRI_Pooled._cache["pool"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        emb  = self.cache[row["PatientID"]].copy()
        item = {
            "emb": torch.from_numpy(emb),
            "t":   torch.tensor(float(row["surv"]),  dtype=torch.float32),
            "e":   torch.tensor(int(row["status"]),  dtype=torch.int32),
            "pid": row["PatientID"],
        }
        if self.use_clin:
            item["clinical"] = torch.from_numpy(_get_clinical(row))
        return item


class DS_COLIPRI_Patch(Dataset):
    """
    E17a/b/c/d: Pre-computed COLIPRI patch embeddings (864, 24, 24, 24).

    Total RAM footprint: 422 × 864 × 24³ × 4 bytes ≈ 22.8 GB.
    Set preload=True only if system RAM > 25 GB; otherwise loads on-demand.

    Note: augmentation is not applied to COLIPRI patch embeddings because the
    embeddings are pre-computed from the frozen COLIPRI encoder; augmenting in
    patch space would corrupt the semantic structure of the representation.
    The trainer always passes augment=False for COLIPRI experiments.
    """

    _cache: dict = {}

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        colipri_patch_dir: str | Path,
        augment: bool = False,  # always False for COLIPRI; kept for API compat
        use_clin: bool = False,
        preload: bool = True,
        **kw,
    ):
        self.df                = df.iloc[indices].reset_index(drop=True)
        self.use_clin          = use_clin
        self.colipri_patch_dir = Path(colipri_patch_dir)

        if preload and "patch" not in DS_COLIPRI_Patch._cache:
            print(
                "  RAM preload: COLIPRI patch embeddings...",
                end="",
                flush=True,
            )
            t0    = time.time()
            files = list(self.colipri_patch_dir.glob("*.npy"))
            cache = {}

            def _load(f):
                return f.stem, np.load(str(f)).astype(np.float32)

            with ThreadPoolExecutor(max_workers=4) as ex:
                for pid, arr in ex.map(_load, files):
                    cache[pid] = arr

            DS_COLIPRI_Patch._cache["patch"] = cache
            gb = sum(v.nbytes for v in cache.values()) / 1e9
            print(f" {len(cache)} files  {gb:.1f} GB  {time.time()-t0:.1f}s")

        self.preload = preload and "patch" in DS_COLIPRI_Patch._cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        pid = row["PatientID"]

        if self.preload:
            patch = DS_COLIPRI_Patch._cache["patch"][pid].copy()
        else:
            patch = np.load(
                str(self.colipri_patch_dir / f"{pid}.npy")
            ).astype(np.float32)

        item = {
            "patch": torch.from_numpy(patch),
            "t":     torch.tensor(float(row["surv"]),  dtype=torch.float32),
            "e":     torch.tensor(int(row["status"]),  dtype=torch.int32),
            "pid":   pid,
        }
        if self.use_clin:
            item["clinical"] = torch.from_numpy(_get_clinical(row))
        return item
