"""
COLIPRI-based survival models (E16a/b, E17a/b/c/d).

COLIPRI (microsoft/colipri; Wald et al., ECCV 2026) is a 3D vision-language
transformer pretrained on chest CT paired with radiology reports. It is used
here purely as a FROZEN feature extractor; only the lightweight survival heads
below are trained. Embeddings are precomputed and cached per patient:

    pooled : (768,)              global embedding            -> E16a/b
    patch  : (864, 24, 24, 24)   dense spatial embedding     -> E17a/b/c/d

Pretraining data
----------------
COLIPRI is pretrained on CT-RATE (25,692 CT-report pairs) and NLST (~72,000
image-only chest CT). Neither corpus includes the NSCLC-Radiomics / Lung1
cohort used here (Lung1 is from MAASTRO/TCIA), so these results are not
affected by pretraining/test overlap. As with any single-cohort study,
external validation remains necessary.

Argument / batch-key convention
--------------------------------
Dataset batches use the keys ``emb`` (pooled) and ``patch`` (dense), plus
``clinical`` for the fusion heads. Model ``forward`` signatures below match
those keys exactly so the trainer can dispatch ``model(**inputs)``.

Sign convention
---------------
All models output log-hazard ratios eta(x); higher eta = higher risk.
concordance_index() expects higher score = longer survival, so negate: -eta.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cnn import AttentionPool3d


# ─────────────────────────────────────────────────────────────────────────────
# E16: pooled-embedding heads  (input key: "emb", shape (B, 768))
# ─────────────────────────────────────────────────────────────────────────────

class COLIPRI_Pooled_MLP_E16a(nn.Module):
    """
    E16a: COLIPRI 768-dim pooled embedding -> 2-layer MLP Cox head.

    Architecture: 768 -> Linear(128) -> BN(128) -> ReLU -> Dropout -> Linear(1)
    Params: 98,817
    """

    def __init__(self, emb_dim: int = 768, hidden: int = 128,
                 dropout: float = 0.3, **kw):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=False), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        nn.init.normal_(self.head[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, emb: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(emb).squeeze(-1)


class COLIPRI_Pooled_Clinical_E16b(nn.Module):
    """
    E16b: COLIPRI pooled embedding + 16-dim clinical, late fusion.

    Imaging branch:  768 -> Linear(128) -> BN -> ReLU -> Dropout -> (128,)
    Clinical branch: 16  -> Linear(64) -> ReLU -> Dropout -> Linear(16) -> ReLU
    Fusion head:     (128 + 16) -> Linear(1)
    Params: 100,961
    """

    def __init__(self, emb_dim: int = 768, n_clinical: int = 16,
                 hidden: int = 128, dropout: float = 0.3, **kw):
        super().__init__()
        self.img = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=False), nn.Dropout(dropout))
        self.clin = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False), nn.Dropout(0.2),
            nn.Linear(64, 16), nn.ReLU(inplace=False))
        self.head = nn.Linear(hidden + 16, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, emb: torch.Tensor = None,
                clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        z = self.img(emb)
        c = (self.clin(clinical) if clinical is not None
             else torch.zeros(z.shape[0], 16, device=z.device))
        return self.head(torch.cat([z, c], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# E17: dense-patch heads  (input key: "patch", shape (B, 864, 24, 24, 24))
# ─────────────────────────────────────────────────────────────────────────────

class COLIPRI_Patch_AttnPool_E17a(nn.Module):
    """
    E17a: COLIPRI patch tensor (864, 24, 24, 24) -> AttentionPool3d -> Cox.

    AttentionPool3d gates each of the 24^3 spatial positions with a learned
    1x1x1 conv, then takes the attention-weighted mean over space. Smallest
    head in the study and the attention map is directly interpretable.
    Params: 1,730
    """

    def __init__(self, patch_ch: int = 864, dropout: float = 0.3, **kw):
        super().__init__()
        self.pool = AttentionPool3d(patch_ch)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(patch_ch, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.drop(self.pool(patch))).squeeze(-1)

    def get_attention_map(self, patch: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, 24, 24, 24) spatial attention weights."""
        return self.pool.get_attention_map(patch)


class COLIPRI_Patch_Conv_GAP_E17b(nn.Module):
    """
    E17b: COLIPRI patch -> Conv3d(864->256, k=3, s=1) -> BN -> ReLU -> GAP -> Cox.

    A 3x3x3 kernel mixes neighbouring patches before global average pooling.
    Params: 5,972,737
    """

    def __init__(self, in_ch: int = 864, mid: int = 256,
                 dropout: float = 0.3, **kw):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=False))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(mid, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch: torch.Tensor = None, **kw) -> torch.Tensor:
        z = self.pool(self.conv(patch)).flatten(1)
        return self.head(self.drop(z)).squeeze(-1)


class COLIPRI_Patch_Conv_AttnPool_E17c(nn.Module):
    """
    E17c: COLIPRI patch -> Conv3d(864->256, k=3) -> AttentionPool3d -> Cox.

    Local spatial reasoning (Conv3d) followed by learned attention pooling.
    Attention map available via get_attention_map() for interpretability.
    Params: 5,972,994
    """

    def __init__(self, in_ch: int = 864, mid: int = 256,
                 dropout: float = 0.3, **kw):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=False))
        self.pool = AttentionPool3d(mid)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(mid, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.drop(self.pool(self.conv(patch)))).squeeze(-1)

    def get_attention_map(self, patch: torch.Tensor) -> torch.Tensor:
        return self.pool.get_attention_map(self.conv(patch))


class COLIPRI_Patch_Conv_AttnPool_Clinical_E17d(nn.Module):
    """
    E17d: E17c + 16-dim clinical late fusion (best overall model).

    Imaging branch:  patch -> Conv3d -> AttentionPool3d -> (256,)
    Clinical branch: 16 -> Linear(64) -> ReLU -> Dropout -> Linear(16) -> ReLU
    Fusion head:     (256 + 16) -> Linear(1)
    Params: 5,975,138
    """

    def __init__(self, in_ch: int = 864, mid: int = 256,
                 n_clinical: int = 16, dropout: float = 0.3, **kw):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=False))
        self.pool = AttentionPool3d(mid)
        self.drop = nn.Dropout(dropout)
        self.clin = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False), nn.Dropout(0.2),
            nn.Linear(64, 16), nn.ReLU(inplace=False))
        self.head = nn.Linear(mid + 16, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch: torch.Tensor = None,
                clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        z = self.drop(self.pool(self.conv(patch)))
        c = (self.clin(clinical) if clinical is not None
             else torch.zeros(z.shape[0], 16, device=z.device))
        return self.head(torch.cat([z, c], dim=-1)).squeeze(-1)

    def get_attention_map(self, patch: torch.Tensor) -> torch.Tensor:
        return self.pool.get_attention_map(self.conv(patch))
