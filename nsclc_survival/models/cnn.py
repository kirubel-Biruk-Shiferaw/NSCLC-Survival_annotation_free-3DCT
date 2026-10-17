"""
Model architectures for experiments E1–E15.

Critical implementation notes
------------------------------
1. ReLU(inplace=False) everywhere — required for GradCAM backward pass.
   In-place ReLU overwrites stored pre-activation tensors that backward
   needs to compute gradients, causing silent NaN gradients.

2. bias=False in all Conv3d — BatchNorm immediately follows and applies
   its own learnable shift β, making the Conv bias redundant.

3. Head weights init ~ N(0, 0.01) — prevents large initial log-hazards
   that would saturate the Cox loss denominator.

Sign convention
---------------
All models output log-hazard ratios η(x).  Higher η = higher risk.
concordance_index() from lifelines expects higher score = longer survival,
so negate when computing C-index: concordance_index(t, -η, e).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared components
# ─────────────────────────────────────────────────────────────────────────────

def cnn_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Standard 3D strided convolution block.

    Architecture: Conv3d(3×3×3, stride=2) → BatchNorm3d → ReLU

    Spatial effect at R2 input (1, 64, 128, 128):
      Block 1 → (out_ch, 32,  64,  64)
      Block 2 → (out_ch, 16,  32,  32)
      Block 3 → (out_ch,  8,  16,  16)  ← GradCAM target (before GAP)
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=False),
    )


class AttentionPool3d(nn.Module):
    """
    Learned spatial attention pooling (replaces AdaptiveAvgPool3d).

    Standard GAP: z_c = (1/DHW) Σ_{d,h,w} x_c(d,h,w)
    AttentionPool:
        w(d,h,w) = σ(Conv_{1×1×1}(x)(d,h,w))
        z_c      = Σ w(d,h,w)·x_c(d,h,w) / Σ w(d,h,w)

    Adds only (C_in + 1) parameters (weight + bias of 1×1×1 conv).
    get_attention_map() returns the exact mechanism weight volume
    (not a GradCAM approximation) — used for interpretability in E17a/c/d.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)  # (B, 1, D, H, W)
        return (x * w).sum(dim=(2, 3, 4)) / (w.sum(dim=(2, 3, 4)) + 1e-7)

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, D, H, W) spatial attention weights."""
        return self.attn(x)


# ─────────────────────────────────────────────────────────────────────────────
# E1–E9: CNN ablation family
# ─────────────────────────────────────────────────────────────────────────────

class CNNSurv_E1(nn.Module):
    """E1: 1-block CNN (1→16f).  Params: ~481."""

    def __init__(self, dropout: float = 0.3, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            cnn_block(1, 16),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.head = nn.Linear(16, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.enc(vol)).squeeze(-1)


class CNNSurv_E2(nn.Module):
    """E2: 2-block CNN (1→16→32f).  Params: ~14,385."""

    def __init__(self, dropout: float = 0.3, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            cnn_block(1, 16), cnn_block(16, 32),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.head = nn.Linear(32, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.enc(vol)).squeeze(-1)


class CNNSurv_E3(nn.Module):
    """
    E3: 3-block baseline (1→16→32→64f).  Params: ~69,841.

    GradCAM target: enc[7] = BatchNorm3d(64) immediately before GAP.
    enc layout: [0]Conv [1]BN [2]ReLU [3]Conv [4]BN [5]ReLU
                [6]Conv [7]BN [8]ReLU [9]GAP [10]Flatten [11]Drop
    """

    def __init__(self, base_filters: int = 16, dropout: float = 0.3, **kw):
        super().__init__()
        f = base_filters
        self.enc  = nn.Sequential(
            cnn_block(1, f), cnn_block(f, f * 2), cnn_block(f * 2, f * 4),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.head = nn.Linear(f * 4, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.enc(vol)).squeeze(-1)


class CNNSurv_E4(nn.Module):
    """E4: 4-block capacity probe (1→32→64→128→256f).  Params: ~1.16M."""

    def __init__(self, dropout: float = 0.4, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            cnn_block(1, 32), cnn_block(32, 64),
            cnn_block(64, 128), cnn_block(128, 256),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.head = nn.Linear(256, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.enc(vol)).squeeze(-1)


class CNNSurv_E5(nn.Module):
    """
    E5: 3-block + residual shortcuts (ResNet-style).  Params: ~72,641.
    Projection shortcuts for channel/spatial dimension matching.
    """

    def __init__(self, base_filters: int = 16, dropout: float = 0.3, **kw):
        super().__init__()
        f = base_filters
        self.b1 = cnn_block(1,     f)
        self.b2 = cnn_block(f,     f * 2)
        self.b3 = cnn_block(f * 2, f * 4)
        self.s1 = nn.Sequential(
            nn.Conv3d(1,     f,     1, stride=2, bias=False), nn.BatchNorm3d(f))
        self.s2 = nn.Sequential(
            nn.Conv3d(f,     f * 2, 1, stride=2, bias=False), nn.BatchNorm3d(f * 2))
        self.s3 = nn.Sequential(
            nn.Conv3d(f * 2, f * 4, 1, stride=2, bias=False), nn.BatchNorm3d(f * 4))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(f * 4, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        x1 = F.relu(self.b1(vol) + self.s1(vol),  inplace=False)
        x2 = F.relu(self.b2(x1)  + self.s2(x1),  inplace=False)
        x3 = F.relu(self.b3(x2)  + self.s3(x2),  inplace=False)
        return self.head(self.drop(self.pool(x3).flatten(1))).squeeze(-1)


class CNNSurv_E6(nn.Module):
    """
    E6: 3-block wider filters (1→32→64→128f).  Params: ~277,921.

    GradCAM target: enc[7] = BatchNorm3d(128) before GAP.
    """

    def __init__(self, dropout: float = 0.3, **kw):
        super().__init__()
        self.enc  = nn.Sequential(
            cnn_block(1, 32), cnn_block(32, 64), cnn_block(64, 128),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.head = nn.Linear(128, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.head(self.enc(vol)).squeeze(-1)


class CNNSurv_E7(CNNSurv_E3):
    """E7: E3 + lung masking (DS_LungMasked applies it). Model is identical to E3."""
    pass


class CNNSurv_E8(CNNSurv_E3):
    """E8: E3 + heavy augmentation (DS_HeavyAug applies it). Model is identical to E3."""
    pass


class CNNSurv_E9(nn.Module):
    """
    E9: 3-block CNN + clinical late fusion.  Params: ~71,921.

    Late fusion: imaging and clinical branches combined only at the head.
    Imaging: R2 volume → 3 conv blocks → GAP → (64,)
    Clinical: 16-dim → Linear(64) → ReLU → Linear(16) → ReLU → (16,)
    Fusion: concat (80,) → Linear(1)
    """

    def __init__(self, n_clinical: int = 16, base_filters: int = 16,
                 dropout: float = 0.3, **kw):
        super().__init__()
        f = base_filters
        self.img_enc = nn.Sequential(
            cnn_block(1, f), cnn_block(f, f * 2), cnn_block(f * 2, f * 4),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(p=dropout))
        self.clin_enc = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 16),         nn.ReLU(inplace=False))
        self.head = nn.Linear(f * 4 + 16, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, vol: torch.Tensor = None,
                clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        img  = self.img_enc(vol)
        clin = self.clin_enc(clinical)
        return self.head(torch.cat([img, clin], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# E10–E11: Clinical baselines
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalMLP_E10(nn.Module):
    """
    E10: Clinical-only MLP Cox.  Params: ~3,137.
    Architecture: 16 → 64 → ReLU → Drop → 32 → ReLU → 1
    Loss: Efron (tied integer-day times)
    """

    def __init__(self, n_clinical: int = 16, dropout: float = 0.3, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(inplace=False),
            nn.Linear(32, 1))
        nn.init.normal_(self.net[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.net(clinical).squeeze(-1)


class DeepSurv_E11(nn.Module):
    """
    E11: DeepSurv (Katzman et al. 2018).  Params: ~12,801.
    Architecture: 16 → BN(128) → ReLU → Drop → BN(64) → ReLU → Drop → 32 → ReLU → 1
    """

    def __init__(self, n_clinical: int = 16, dropout: float = 0.3, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_clinical, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),     nn.Dropout(dropout),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),     nn.Dropout(dropout),
            nn.Linear(64, 32),          nn.ReLU(inplace=False),
            nn.Linear(32, 1))
        nn.init.normal_(self.net[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, clinical: torch.Tensor = None, **kw) -> torch.Tensor:
        return self.net(clinical).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# E12–E13: Spatial correction
# ─────────────────────────────────────────────────────────────────────────────

class CNNSurv_E12(CNNSurv_E9):
    """
    E12: E9 architecture on HU-threshold lung crops (DS_LungCrop_HU).
    Motivated by GradCAM: CNN attends to abdomen, not tumour.
    Dataset handles the crop; model is identical to E9.
    """
    pass


class CNNSurv_MAAS_E13(nn.Module):
    """
    E13: MAAS 3-region decomposition + clinical.  Params: ~47,377.

    Three independent 2-block CNN branches:
      r1 (lung/mediastinum) + r2 (body composition) + r3 (abdomen)
    Each → (32,); concatenated with clinical (16,) → Linear(1).
    """

    def __init__(self, n_clinical: int = 16, dropout: float = 0.3, **kw):
        super().__init__()

        def branch():
            return nn.Sequential(
                cnn_block(1, 16), cnn_block(16, 32),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout))

        self.b1 = branch()
        self.b2 = branch()
        self.b3 = branch()
        self.clin = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(64, 16),         nn.ReLU(inplace=False))
        self.head = nn.Linear(32 * 3 + 16, 1)  # 96 + 16 = 112
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, r1: torch.Tensor, r2: torch.Tensor,
                r3: torch.Tensor, clinical: torch.Tensor, **kw) -> torch.Tensor:
        return self.head(
            torch.cat([self.b1(r1), self.b2(r2), self.b3(r3), self.clin(clinical)], dim=-1)
        ).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# E14–E15: MedicalNet pretrained encoder
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock3D(nn.Module):
    """ResNet BasicBlock for 3D volumes (MedicalNet backbone building block)."""

    def __init__(self, ic: int, oc: int, stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        self.c1 = nn.Conv3d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm3d(oc)
        self.c2 = nn.Conv3d(oc, oc, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm3d(oc)
        self.ds = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.b1(self.c1(x)), inplace=False)
        out = self.b2(self.c2(out))
        if self.ds is not None:
            residual = self.ds(x)
        return F.relu(out + residual, inplace=False)


class ResNet10_3D(nn.Module):
    """
    MedicalNet ResNet10 backbone (Chen et al. 2019).
    Pretrained on 8 segmentation datasets (~1,695 scans, multi-anatomy).
    Output: (B, 512, D', H', W') feature map.
    """

    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        def make_layer(ic, oc, stride=1):
            ds = nn.Sequential(
                nn.Conv3d(ic, oc, 1, stride=stride, bias=False),
                nn.BatchNorm3d(oc),
            ) if stride != 1 or ic != oc else None
            return nn.Sequential(BasicBlock3D(ic, oc, stride, ds))

        self.layer1 = make_layer(64,  64,  stride=1)
        self.layer2 = make_layer(64,  128, stride=2)
        self.layer3 = make_layer(128, 256, stride=2)
        self.layer4 = make_layer(256, 512, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x)), inplace=False))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def load_medicalnet_weights(
    backbone: ResNet10_3D,
    ckpt_path: str | Path,
) -> ResNet10_3D:
    """
    Load pretrained MedicalNet weights, ignoring segmentation head keys.
    Missing keys (segmentation head) are expected and acceptable.
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        clean = {
            k.replace("module.", ""): v
            for k, v in state.items()
            if not k.replace("module.", "").startswith(("conv_seg", "seg"))
        }
        missing, unexpected = backbone.load_state_dict(clean, strict=False)
        print(
            f"  MedicalNet weights loaded.  "
            f"Missing: {len(missing)}  Unexpected: {len(unexpected)}"
        )
    else:
        print(
            f"  WARNING: MedicalNet checkpoint not found at {ckpt_path}. "
            f"Using random initialisation."
        )
    return backbone


class MedicalNetSurv_E14(nn.Module):
    """
    E14: MedicalNet ResNet10 + AttentionPool3d survival head.  Params: ~14M.

    Fine-tuning protocol (3 phases, prevents catastrophic forgetting):
      Phase 1 (10 ep):  freeze backbone, train head only
      Phase 2 (20 ep):  unfreeze layer4 + head
      Phase 3 (rest):   unfreeze all (full fine-tuning)
    """

    def __init__(self, ckpt_path: str | Path = None, dropout: float = 0.3, **kw):
        super().__init__()
        self.backbone = ResNet10_3D()
        if ckpt_path:
            load_medicalnet_weights(self.backbone, ckpt_path)
        self.pool = AttentionPool3d(512)
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(512, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_layer4(self):
        for p in self.backbone.layer4.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def forward(self, vol: torch.Tensor = None, **kw) -> torch.Tensor:
        feat = self.backbone(vol)          # (B, 512, D', H', W')
        z    = self.pool(feat)             # (B, 512)
        return self.head(self.drop(z)).squeeze(-1)

    def get_attention_map(self, vol: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, D', H', W') attention weight volume."""
        return self.pool.get_attention_map(self.backbone(vol))


class MedicalNetSurv_E15(MedicalNetSurv_E14):
    """
    E15: MedicalNet-guided lung crop → MedicalNet encoder.
    Crop boundary is determined by MedicalNet layer4 activation thresholding
    (more anatomically precise than HU-threshold in E12).
    The survival model class is identical to E14; the dataset handles the crop.
    """
    pass
