"""
COLIPRI-based survival models (E16a/b, E17a/b/c/d).

COLIPRI (microsoft/colipri) is a vision-language model pretrained on
chest CT scans paired with radiology reports.  We use it purely as a
feature extractor (frozen weights) and train lightweight survival heads.

Pretraining caveat
------------------
microsoft/colipri was pretrained on a large chest CT corpus that MAY
include The Cancer Imaging Archive (TCIA) data, including the Lung1
cohort used here.  If Lung1 scans were part of the pretraining set, the
COLIPRI features already "saw" the outcome labels indirectly through the
paired radiology reports (which often include survival information).
This would constitute data leakage and would invalidate the C-index
improvement from ~0.58 (CNN) to ~0.65+ (COLIPRI) as an estimate of
true generalisation.  We include these experiments for completeness and
comparison with Paolo et al. 2026, but all results should be interpreted
with this caveat.  See docs/model_card.md for more details.

Sign convention
---------------
All models output log-hazard ratios η(x).  Higher η = higher risk.
concordance_index() expects higher score = longer survival, so negate:
    concordance_index(t, -η, e)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import AttentionPool3d


# ─────────────────────────────────────────────────────────────────────────────
# Shared MLP building block
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(dims: list[int], dropout: float = 0.3) -> nn.Sequential:
    """Build a fully-connected tower with BN + ReLU + Dropout on hidden layers."""
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:          # hidden layers only
            layers += [nn.BatchNorm1d(d_out), nn.ReLU(inplace=False),
                       nn.Dropout(dropout)]
    seq = nn.Sequential(*layers)
    nn.init.normal_(seq[-1].weight, 0.0, 0.01)
    nn.init.zeros_(seq[-1].bias)
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# E16: Pooled-embedding models (768-dim cls-token)
# ─────────────────────────────────────────────────────────────────────────────

class COLIPRI_Pooled_MLP_E16a(nn.Module):
    """
    E16a: COLIPRI 768-dim pooled embedding → MLP Cox.  Params: ~304,129.

    Architecture: 768 → BN(512) → ReLU → Drop → BN(256) → ReLU → Drop →
                       BN(128) → ReLU → Drop → 64 → ReLU → 1
    Input: (B, 768) cls-token from DS_COLIPRI_Pooled.
    """

    def __init__(self, emb_dim: int = 768, dropout: float = 0.3, **kw):
        super().__init__()
        self.net = _mlp([emb_dim, 512, 256, 128, 64, 1], dropout=dropout)

    def forward(self, embedding: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)


class COLIPRI_Pooled_Clinical_E16b(nn.Module):
    """
    E16b: COLIPRI 768-dim embedding + 16-dim clinical.  Params: ~305,697.

    Late fusion: two separate towers, concatenated before the final head.
      COLIPRI tower: 768 → BN(512) → ReLU → Drop → 256 → ReLU → (256,)
      Clinical tower: 16 → 64 → ReLU → Drop → 32 → ReLU → (32,)
      Head: (288,) → 64 → ReLU → 1
    """

    def __init__(self, emb_dim: int = 768, n_clinical: int = 16,
                 dropout: float = 0.3, **kw):
        super().__init__()
        self.emb_tower = nn.Sequential(
            nn.Linear(emb_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),  nn.Dropout(dropout),
            nn.Linear(512, 256),     nn.ReLU(inplace=False))
        self.clin_tower = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(inplace=False))
        self.head = nn.Sequential(
            nn.Linear(256 + 32, 64), nn.ReLU(inplace=False),
            nn.Linear(64, 1))
        nn.init.normal_(self.head[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, embedding: torch.Tensor = None,
                clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        e = self.emb_tower(embedding)
        c = self.clin_tower(clinical)
        return self.head(torch.cat([e, c], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# E17: Patch-level models (864 × 24³ voxel patches)
# ─────────────────────────────────────────────────────────────────────────────

class COLIPRI_Patch_AttnPool_E17a(nn.Module):
    """
    E17a: COLIPRI patch embeddings → learned attention pooling.

    Input: (B, N_patches, emb_dim) where N_patches ≤ 864, emb_dim = 768.
    Processing:
      Per-patch projection: 768 → 256 (shared linear)
      Attention score: 256 → 1 (scalar per patch)
      Weighted sum over patches → (B, 256)
      Head: 256 → 64 → 1

    Attention pooling here is over the N-patch sequence (not spatial),
    analogous to multiple-instance learning aggregation.
    get_attention_weights() returns (B, N_patches) for interpretability.
    """

    def __init__(self, emb_dim: int = 768, proj_dim: int = 256,
                 dropout: float = 0.3, **kw):
        super().__init__()
        self.proj  = nn.Linear(emb_dim, proj_dim)
        self.attn  = nn.Linear(proj_dim, 1)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.ReLU(inplace=False),
            nn.Linear(64, 1))
        nn.init.normal_(self.head[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def _aggregate(self, patches: torch.Tensor):
        """patches: (B, N, emb_dim) → pooled (B, proj_dim), weights (B, N)."""
        z   = F.relu(self.proj(patches), inplace=False)   # (B, N, proj_dim)
        w   = torch.softmax(self.attn(z).squeeze(-1), dim=-1)  # (B, N)
        agg = (z * w.unsqueeze(-1)).sum(dim=1)            # (B, proj_dim)
        return agg, w

    def forward(self, patches: torch.Tensor = None, **kw) -> torch.Tensor:
        agg, _ = self._aggregate(patches)
        return self.head(self.drop(agg)).squeeze(-1)

    def get_attention_weights(self, patches: torch.Tensor) -> torch.Tensor:
        """Return (B, N_patches) attention weights (sums to 1 over N)."""
        _, w = self._aggregate(patches)
        return w


class COLIPRI_Patch_Conv_GAP_E17b(nn.Module):
    """
    E17b: COLIPRI patch embeddings → 3-block conv backbone → GAP.

    Each patch is a (emb_dim, D, H, W) tensor = (768, 24, 24, 24).
    We treat the embedding dimension as channels and apply 3D convolutions.
      Block 1: 768→ 64, stride=2 → (B, 64, 12, 12, 12)
      Block 2:  64→128, stride=2 → (B,128,  6,  6,  6)
      Block 3: 128→256, stride=2 → (B,256,  3,  3,  3)
      GAP → (B, 256) → Dropout → Linear(1)
    """

    def __init__(self, emb_dim: int = 768, dropout: float = 0.3, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            nn.Conv3d(emb_dim, 64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),  nn.ReLU(inplace=False),
            nn.Conv3d(64,  128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=False),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(256, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch_vol: torch.Tensor = None, **kw) -> torch.Tensor:
        """patch_vol: (B, emb_dim, D, H, W) pre-stacked by DS_COLIPRI_Patch."""
        return self.head(self.enc(patch_vol)).squeeze(-1)


class COLIPRI_Patch_Conv_AttnPool_E17c(nn.Module):
    """
    E17c: COLIPRI patch volume → 3-block conv → AttentionPool3d.

    Same architecture as E17b but replaces AdaptiveAvgPool3d with learned
    AttentionPool3d so the network highlights diagnostically relevant
    spatial regions.  get_attention_map() returns the (B,1,D',H',W') volume
    for interpretability (ATR computation).
    """

    def __init__(self, emb_dim: int = 768, dropout: float = 0.3, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            nn.Conv3d(emb_dim, 64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),  nn.ReLU(inplace=False),
            nn.Conv3d(64,  128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=False),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=False))
        self.pool = AttentionPool3d(256)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(256, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, patch_vol: torch.Tensor = None, **kw) -> torch.Tensor:
        feat = self.enc(patch_vol)
        return self.head(self.drop(self.pool(feat))).squeeze(-1)

    def get_attention_map(self, patch_vol: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, D', H', W') spatial attention weights."""
        return self.pool.get_attention_map(self.enc(patch_vol))


class COLIPRI_Patch_Conv_AttnPool_Clinical_E17d(nn.Module):
    """
    E17d: E17c + clinical late fusion (best overall model).

    Extends E17c with a clinical branch fused after attention pooling.
    Architecture:
      COLIPRI branch:  (B, emb_dim, D, H, W) → conv → AttnPool → (B, 256)
      Clinical branch: (B, 16) → 64 → ReLU → Drop → 32 → ReLU → (B, 32)
      Head:            (B, 288) → 128 → ReLU → 64 → ReLU → 1
    """

    def __init__(self, emb_dim: int = 768, n_clinical: int = 16,
                 dropout: float = 0.3, **kw):
        super().__init__()
        # COLIPRI branch
        self.enc  = nn.Sequential(
            nn.Conv3d(emb_dim, 64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),  nn.ReLU(inplace=False),
            nn.Conv3d(64,  128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=False),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=False))
        self.pool = AttentionPool3d(256)
        self.drop = nn.Dropout(dropout)

        # Clinical branch
        self.clin = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(inplace=False))

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(inplace=False),
            nn.Linear(64, 1))
        nn.init.normal_(self.head[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, patch_vol: torch.Tensor = None,
                clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        feat = self.enc(patch_vol)
        img  = self.drop(self.pool(feat))          # (B, 256)
        clin = self.clin(clinical)                 # (B, 32)
        return self.head(torch.cat([img, clin], dim=-1)).squeeze(-1)

    def get_attention_map(self, patch_vol: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, D', H', W') spatial attention weights."""
        return self.pool.get_attention_map(self.enc(patch_vol))
