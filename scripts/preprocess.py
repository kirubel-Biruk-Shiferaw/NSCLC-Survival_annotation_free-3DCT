#!/usr/bin/env python3
"""
Preprocess TCIA NSCLC-Radiomics (Lung1) data for the NSCLC survival pipeline.

Two stages
----------
Stage 1  (--stage ct)
    DICOM series  →  NIfTI (.nii.gz)  →  R2 numpy cache (.npy)
    Output:  <artifact_dir>/global_cache_128/{pid}_global.npy
             shape (64, 128, 128)  float32  HU clipped to [-1000, 400] and
             normalised to [0, 1].

Stage 2  (--stage colipri)
    R2 numpy cache  →  COLIPRI pooled + patch embeddings (.npy)
    Requires the microsoft/colipri model on HuggingFace (request access first).
    Output:  <artifact_dir>/colipri_nifti/pooled/{pid}.npy   shape (768,)
             <artifact_dir>/colipri_nifti/patch/{pid}.npy    shape (864, 24, 24, 24)

Usage
-----
# Stage 1 — CT preprocessing
python scripts/preprocess.py \\
    --stage ct \\
    --dicom-dir  data/raw/NSCLC-Radiomics \\
    --artifact-dir data/artifacts

# Stage 2 — COLIPRI embeddings  (needs GPU; ~30 min for 422 scans)
python scripts/preprocess.py \\
    --stage colipri \\
    --artifact-dir data/artifacts \\
    --hf-token YOUR_HF_TOKEN       # or set HF_TOKEN env var

# Both stages in sequence
python scripts/preprocess.py \\
    --stage all \\
    --dicom-dir  data/raw/NSCLC-Radiomics \\
    --artifact-dir data/artifacts

Environment variables
---------------------
NSCLC_ARTIFACT_DIR  : override --artifact-dir
HF_TOKEN            : HuggingFace token for gated COLIPRI weights
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# ── Volume constants (must match configs/default.yaml) ────────────────────────
VOL_SHAPE = (64, 128, 128)          # D × H × W after R2 resampling
HU_MIN    = -1000.0                 # clip low  → 0
HU_MAX    =  400.0                  # clip high → 1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dicom_to_volume(dicom_dir: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Read a DICOM series from dicom_dir and return
    (volume_hu, voxel_spacing_mm) where volume_hu has shape (D, H, W).
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        sys.exit(
            "SimpleITK is required for DICOM reading.\n"
            "  pip install SimpleITK\n"
            "Alternatively, convert DICOMs to NIfTI with dcm2niix first."
        )

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    vol   = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)
    space = image.GetSpacing()  # (x, y, z) in mm — sitk: x=col, y=row, z=slice
    spacing_dhw = (space[2], space[1], space[0])              # → (D, H, W)
    return vol, spacing_dhw


def _nifti_to_volume(nifti_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Read a NIfTI file and return (volume_hu, voxel_spacing_dhw).
    """
    import nibabel as nib
    img = nib.load(str(nifti_path))
    vol = img.get_fdata(dtype=np.float32)        # (H, W, D) for most NIfTI
    # nibabel canonical: rows, cols, slices → transpose to (D, H, W)
    vol = vol.transpose(2, 0, 1)
    hdr = img.header
    zooms = hdr.get_zooms()                       # (x, y, z) voxel sizes in mm
    spacing_dhw = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
    return vol, spacing_dhw


def _resample_and_normalise(vol: np.ndarray, spacing_dhw: tuple) -> np.ndarray:
    """
    Resample vol to VOL_SHAPE using linear interpolation, then HU-clip and
    normalise to [0, 1].  Returns float32 array of shape VOL_SHAPE.
    """
    from skimage.transform import resize

    # Resample to target shape (anti_aliasing only when downsampling)
    target    = np.array(VOL_SHAPE, dtype=float)
    current   = np.array(vol.shape, dtype=float)
    do_aa     = any(current > target)

    vol_r = resize(
        vol, VOL_SHAPE,
        order=1,
        anti_aliasing=do_aa,
        preserve_range=True,
    ).astype(np.float32)

    # HU normalisation: clip to [HU_MIN, HU_MAX] → [0, 1]
    vol_r = np.clip(vol_r, HU_MIN, HU_MAX)
    vol_r = (vol_r - HU_MIN) / (HU_MAX - HU_MIN)

    return vol_r


def _find_dicom_series_dir(pid_dir: Path) -> Path | None:
    """
    Return the deepest directory that contains DICOM files under pid_dir.
    TCIA downloads often nest series inside patient → study → series folders.
    """
    dcm_dirs = sorted(set(f.parent for f in pid_dir.rglob("*.dcm")))
    if not dcm_dirs:
        # Some TCIA packages drop the .dcm extension — look for any file
        all_files = [f for f in pid_dir.rglob("*") if f.is_file()]
        dcm_dirs = sorted(set(f.parent for f in all_files if f.suffix == ""))
    if not dcm_dirs:
        return None
    # Prefer the directory with the most files (largest series)
    return max(dcm_dirs, key=lambda d: sum(1 for _ in d.iterdir()))


def stage_ct(dicom_root: Path, artifact_dir: Path) -> None:
    """
    Stage 1: DICOM/NIfTI → R2 numpy cache.

    dicom_root is expected to contain one sub-directory per patient named
    LUNG1-XXX, each containing either a DICOM series or a single .nii.gz file.
    """
    cache_dir = artifact_dir / "global_cache_128"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pid_dirs = sorted(dicom_root.glob("LUNG1-*"))
    if not pid_dirs:
        # Try flat directory with .nii.gz files
        nii_files = sorted(dicom_root.glob("*.nii.gz"))
        if nii_files:
            print(f"Found {len(nii_files)} NIfTI files directly in {dicom_root}")
            for nii in nii_files:
                pid = nii.stem.split(".")[0]
                out = cache_dir / f"{pid}_global.npy"
                if out.exists():
                    print(f"  {pid}: already cached, skipping")
                    continue
                try:
                    vol, spacing = _nifti_to_volume(nii)
                    vol_r = _resample_and_normalise(vol, spacing)
                    np.save(str(out), vol_r)
                    print(f"  {pid}: {vol.shape} {spacing} → {vol_r.shape}  ✓")
                except Exception as e:
                    warnings.warn(f"  {pid}: FAILED — {e}")
            return
        sys.exit(
            f"No LUNG1-* directories or .nii.gz files found in {dicom_root}.\n"
            f"Expected structure:  {dicom_root}/LUNG1-001/  (DICOM files inside)\n"
            f"or flat NIfTI:       {dicom_root}/LUNG1-001.nii.gz"
        )

    print(f"\nStage 1 — CT preprocessing")
    print(f"  Source : {dicom_root}  ({len(pid_dirs)} patients)")
    print(f"  Output : {cache_dir}")
    print(f"  Shape  : {VOL_SHAPE}  (D×H×W)  float32  HU→[0,1]")
    print()

    ok = err = skip = 0
    for pid_dir in pid_dirs:
        pid = pid_dir.name
        out = cache_dir / f"{pid}_global.npy"

        if out.exists():
            skip += 1
            continue

        try:
            # Check for a NIfTI file first
            nii_files = list(pid_dir.rglob("*.nii.gz"))
            if nii_files:
                vol, spacing = _nifti_to_volume(nii_files[0])
            else:
                series_dir = _find_dicom_series_dir(pid_dir)
                if series_dir is None:
                    warnings.warn(f"  {pid}: no DICOM or NIfTI found, skipping")
                    err += 1
                    continue
                vol, spacing = _dicom_to_volume(series_dir)

            vol_r = _resample_and_normalise(vol, spacing)
            np.save(str(out), vol_r)
            print(f"  {pid}: {vol.shape} → {vol_r.shape}  ✓")
            ok += 1

        except Exception as e:
            warnings.warn(f"  {pid}: FAILED — {e}")
            err += 1

    print(f"\nStage 1 complete: {ok} processed, {skip} already cached, {err} errors")

    if err:
        sys.exit(f"{err} patient(s) failed. Check warnings above.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def stage_colipri(artifact_dir: Path, hf_token: str | None, batch_size: int, device: str) -> None:
    """
    Stage 2: R2 cache → COLIPRI pooled + patch embeddings.

    Outputs
    -------
    pooled/{pid}.npy   — shape (768,)           float32
    patch/{pid}.npy    — shape (864, 24, 24, 24) float32
       where 864 = 4×4×4 patch grid (each patch stride = 6 voxels in 24³ cube)
    """
    import torch

    cache_dir = artifact_dir / "global_cache_128"
    pool_dir  = artifact_dir / "colipri_nifti" / "pooled"
    patch_dir = artifact_dir / "colipri_nifti" / "patch"
    pool_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(cache_dir.glob("*_global.npy"))
    if not npy_files:
        sys.exit(f"No *_global.npy files found in {cache_dir}. Run --stage ct first.")

    print(f"\nStage 2 — COLIPRI embedding extraction")
    print(f"  Source : {cache_dir}  ({len(npy_files)} volumes)")
    print(f"  Pool   : {pool_dir}")
    print(f"  Patch  : {patch_dir}")
    print(f"  Device : {device}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading microsoft/colipri from HuggingFace…", flush=True)
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        sys.exit(
            "transformers>=4.40.0 is required for COLIPRI.\n"
            "  pip install transformers>=4.40.0"
        )

    hf_kwargs: dict = {"trust_remote_code": True}
    if hf_token:
        hf_kwargs["token"] = hf_token

    try:
        model = AutoModel.from_pretrained(
            "microsoft/colipri", **hf_kwargs
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(
            "microsoft/colipri", **hf_kwargs
        )
    except Exception as e:
        sys.exit(
            f"Failed to load microsoft/colipri: {e}\n\n"
            "If you see a 403 / access-denied error, request access at:\n"
            "  https://huggingface.co/microsoft/colipri\n"
            "Then re-run with:  --hf-token YOUR_TOKEN  or set  HF_TOKEN=..."
        )
    print("Model loaded.\n")

    ok = err = skip = 0
    for npy_path in npy_files:
        pid = npy_path.stem.replace("_global", "")
        out_pool  = pool_dir  / f"{pid}.npy"
        out_patch = patch_dir / f"{pid}.npy"

        if out_pool.exists() and out_patch.exists():
            skip += 1
            continue

        try:
            vol = np.load(str(npy_path))                    # (64, 128, 128)
            # COLIPRI expects (B, C, D, H, W)
            tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,128,128)

            with torch.no_grad():
                outputs = model(pixel_values=tensor)

            # last_hidden_state: (1, 1 + n_patches, 768)
            hs = outputs.last_hidden_state.squeeze(0)       # (1+n_patches, 768)
            pooled  = hs[0].cpu().numpy().astype(np.float32)     # (768,)
            patches_flat = hs[1:].cpu().numpy().astype(np.float32) # (n_patches, 768)

            # Reshape patch tokens: COLIPRI uses a 4×6×6 patch grid over
            # the 64×128×128 volume → 864 patches of 16×16×16 voxels each.
            # The pipeline stores them as (n_patches, 768) — no spatial reshape.
            np.save(str(out_pool),  pooled)
            np.save(str(out_patch), patches_flat)
            print(f"  {pid}: pool={pooled.shape}  patch={patches_flat.shape}  ✓")
            ok += 1

        except Exception as e:
            warnings.warn(f"  {pid}: FAILED — {e}")
            err += 1

    print(f"\nStage 2 complete: {ok} extracted, {skip} already cached, {err} errors")
    if err:
        sys.exit(f"{err} patient(s) failed. Check warnings above.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSCLC-Radiomics preprocessing: CT → numpy cache + COLIPRI embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["ct", "colipri", "all"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--dicom-dir",
        type=Path,
        default=Path("data/raw/NSCLC-Radiomics"),
        help="Root directory of downloaded TCIA DICOM data (Stage 1)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Root of preprocessed artefacts (default: $NSCLC_ARTIFACT_DIR or data/artifacts)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for microsoft/colipri (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device for COLIPRI inference (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--colipri-batch",
        type=int,
        default=1,
        help="Batch size for COLIPRI inference (default: 1; increase with more VRAM)",
    )
    args = parser.parse_args()

    # ── Resolve artifact directory ──────────────────────────────────────────
    artifact_dir: Path = (
        args.artifact_dir
        or Path(os.environ.get("NSCLC_ARTIFACT_DIR", "data/artifacts"))
    )
    artifact_dir = artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve device ──────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # ── Resolve HF token ───────────────────────────────────────────────────
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # ── Run stages ─────────────────────────────────────────────────────────
    if args.stage in ("ct", "all"):
        dicom_dir = args.dicom_dir.expanduser().resolve()
        if not dicom_dir.exists():
            sys.exit(
                f"DICOM directory not found: {dicom_dir}\n"
                f"Download NSCLC-Radiomics from TCIA first:\n"
                f"  https://www.cancerimagingarchive.net/collection/nsclc-radiomics/"
            )
        stage_ct(dicom_dir, artifact_dir)

    if args.stage in ("colipri", "all"):
        stage_colipri(artifact_dir, hf_token, args.colipri_batch, device)

    print("\nPreprocessing complete.")
    print(f"Artifact directory: {artifact_dir}")
    print("Next step: python reproduce.py --skip-preprocess")


if __name__ == "__main__":
    main()
