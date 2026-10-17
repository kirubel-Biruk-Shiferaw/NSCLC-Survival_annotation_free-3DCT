"""
Configuration loader.

Priority (highest first):
  1. Environment variables  (NSCLC_ARTIFACT_DIR, NSCLC_OUTPUT_DIR)
  2. User-supplied YAML     (passed to load_config)
  3. Default YAML           (configs/default.yaml in repo root)

Usage:
    from nsclc_survival.utils import CFG, load_config
    CFG = load_config("configs/default.yaml")   # or call with custom path
    print(CFG.data.r2_cache)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass schema (mirrors configs/default.yaml)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    artifact_dir: Path = Path("data/artifacts")
    label_csv: Path = Path("data/artifacts/clinical_data.csv")
    r2_cache: Path = Path("data/artifacts/global_cache_128")
    nifti_dir: Path = Path("data/artifacts/nifti_ct_rtstruct")
    colipri_pool: Path = Path("data/artifacts/colipri_nifti/pooled")
    colipri_patch: Path = Path("data/artifacts/colipri_nifti/patch")
    mednet_dir: Path = Path("data/artifacts/medicalnet")
    mednet_ckpt: Path = Path("data/artifacts/medicalnet/resnet_10.pth")
    vol_shape: tuple = (64, 128, 128)
    feature_cols: List[str] = field(default_factory=lambda: [
        "age_z", "t1", "t2", "t3", "t4",
        "n0", "n1", "n2", "n3",
        "m0", "m1",
        "hist_squamous", "hist_adenocarcinoma", "hist_large_cell", "hist_nos",
        "gender_male",
    ])

    @property
    def n_clinical(self) -> int:
        return len(self.feature_cols)


@dataclass
class OutputConfig:
    model_dir: Path = Path("outputs/models")
    results_dir: Path = Path("outputs/results")
    fig_dir: Path = Path("outputs/figures")
    ckpt_dir: Path = Path("outputs/checkpoints")


@dataclass
class TrainingConfig:
    seed: int = 123
    n_folds_cnn: int = 5
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 1.2e-3
    weight_decay: float = 1e-2
    num_epochs: int = 100
    patience: int = 20
    n_folds_colipri: int = 10
    colipri_lr: float = 1e-3
    colipri_epochs: int = 150
    colipri_patience: int = 30
    colipri_batch: int = 32
    mednet_phase1_epochs: int = 10
    mednet_phase2_epochs: int = 20


@dataclass
class EvaluationConfig:
    bootstrap_n_resamples: int = 2000
    bootstrap_seed: int = 123
    key_exps_for_ci: List[str] = field(
        default_factory=lambda: ["E2", "E10", "E12", "E16a", "E17d"])
    brier_t_min_percentile: int = 10
    brier_t_max_percentile: int = 90


@dataclass
class SotaConfig:
    paolo_2026_ci: float = 0.584
    paolo_2026_ci_std: float = 0.054
    nguyen_2024_ci: float = 0.612


@dataclass
class NSCLCConfig:
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sota: SotaConfig = field(default_factory=SotaConfig)

    def make_output_dirs(self) -> None:
        """Create all output directories."""
        for d in [
            self.output.model_dir,
            self.output.results_dir,
            self.output.fig_dir,
            self.output.ckpt_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path: Optional[str] = None) -> NSCLCConfig:
    """
    Load config from YAML and apply environment-variable overrides.

    Environment variables (take highest priority):
      NSCLC_ARTIFACT_DIR  — overrides all data.* paths prefix
      NSCLC_OUTPUT_DIR    — overrides all output.* paths prefix
    """
    cfg = NSCLCConfig()

    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        if raw.get("data"):
            d = raw["data"]
            cfg.data.artifact_dir   = Path(d.get("artifact_dir", cfg.data.artifact_dir))
            cfg.data.label_csv      = Path(d.get("label_csv",    cfg.data.label_csv))
            cfg.data.r2_cache       = Path(d.get("r2_cache",     cfg.data.r2_cache))
            cfg.data.nifti_dir      = Path(d.get("nifti_dir",    cfg.data.nifti_dir))
            cfg.data.colipri_pool   = Path(d.get("colipri_pool", cfg.data.colipri_pool))
            cfg.data.colipri_patch  = Path(d.get("colipri_patch",cfg.data.colipri_patch))
            cfg.data.mednet_dir     = Path(d.get("mednet_dir",   cfg.data.mednet_dir))
            cfg.data.mednet_ckpt    = Path(d.get("mednet_ckpt",  cfg.data.mednet_ckpt))
            if "vol_shape" in d:
                cfg.data.vol_shape = tuple(d["vol_shape"])
            if "feature_cols" in d:
                cfg.data.feature_cols = d["feature_cols"]

        if raw.get("output"):
            o = raw["output"]
            cfg.output.model_dir   = Path(o.get("model_dir",   cfg.output.model_dir))
            cfg.output.results_dir = Path(o.get("results_dir", cfg.output.results_dir))
            cfg.output.fig_dir     = Path(o.get("fig_dir",     cfg.output.fig_dir))
            cfg.output.ckpt_dir    = Path(o.get("ckpt_dir",    cfg.output.ckpt_dir))

        if raw.get("training"):
            t = raw["training"]
            for attr in TrainingConfig.__dataclass_fields__:
                if attr in t:
                    setattr(cfg.training, attr, t[attr])

        if raw.get("evaluation"):
            ev = raw["evaluation"]
            for attr in EvaluationConfig.__dataclass_fields__:
                if attr in ev:
                    setattr(cfg.evaluation, attr, ev[attr])

        if raw.get("sota"):
            s = raw["sota"]
            for attr in SotaConfig.__dataclass_fields__:
                if attr in s:
                    setattr(cfg.sota, attr, s[attr])

    # ── Environment variable overrides ────────────────────────────────────
    art_dir = os.environ.get("NSCLC_ARTIFACT_DIR")
    if art_dir:
        art = Path(art_dir)
        cfg.data.artifact_dir  = art
        cfg.data.label_csv     = art / "clinical_data.csv"
        cfg.data.r2_cache      = art / "global_cache_128"
        cfg.data.nifti_dir     = art / "nifti_ct_rtstruct"
        cfg.data.colipri_pool  = art / "colipri_nifti" / "pooled"
        cfg.data.colipri_patch = art / "colipri_nifti" / "patch"
        cfg.data.mednet_dir    = art / "medicalnet"
        cfg.data.mednet_ckpt   = art / "medicalnet" / "resnet_10.pth"

    out_dir = os.environ.get("NSCLC_OUTPUT_DIR")
    if out_dir:
        out = Path(out_dir)
        cfg.output.model_dir   = out / "models"
        cfg.output.results_dir = out / "results"
        cfg.output.fig_dir     = out / "figures"
        cfg.output.ckpt_dir    = out / "checkpoints"

    return cfg


# Module-level default config (lazy, uses environment)
_DEFAULT_CFG_PATH = Path(__file__).parents[3] / "configs" / "default.yaml"
CFG = load_config(str(_DEFAULT_CFG_PATH) if _DEFAULT_CFG_PATH.exists() else None)
