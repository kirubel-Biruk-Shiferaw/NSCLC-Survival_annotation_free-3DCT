"""
GradCAM for 3D survival CNNs + Attention-weighted Tumour Ratio (ATR).

GradCAM (Selvaraju et al. 2017, adapted for 3D)
-----------------------------------------------
For a model output scalar η (log-hazard), GradCAM computes:
  α_c = (1/DHW) Σ_{d,h,w} ∂η/∂A_c(d,h,w)    (global-average of gradient)
  CAM = ReLU( Σ_c α_c · A_c )                  (weighted feature map)
  CAM is up-sampled to input resolution with trilinear interpolation.

Only layers where the gradient flows cleanly are valid GradCAM targets:
  · The final Conv3d output before the pooling layer (AdaptiveAvgPool3d).
  · inplace=False on all ReLU in this package is essential — inplace ReLU
    overwrites the saved pre-activation tensors needed for the backward pass.

ATR (Attention-weighted Tumour Ratio)
--------------------------------------
A cohort-level interpretability metric:
  1. For each patient, extract their GradCAM heatmap (or attention map).
  2. Binarise at 50th percentile threshold.
  3. Count voxels that are BOTH in the high-attention region AND inside
     the GTV (Gross Tumour Volume) segmentation mask.
  ATR = (attention ∩ GTV) / |GTV|   ∈ [0, 1]
  ATR → 1 means the network focuses on the tumour; → 0 means elsewhere.

Usage
-----
    gradcam = GradCAM3D(model, target_layer=model.enc[7])
    heatmap  = gradcam(vol.unsqueeze(0))   # (1, D, H, W)
    atr_val  = compute_atr(heatmap, gtv_mask)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM implementation
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    GradCAM for 3D CNNs.

    Parameters
    ----------
    model        : trained PyTorch model
    target_layer : the Conv/BN layer whose feature maps are used.
                   For CNNSurv_E3:  model.enc[7]  (BN3d(64) before GAP)
                   For CNNSurv_E6:  model.enc[7]  (BN3d(128) before GAP)
                   For MedicalNet:  model.backbone.layer4[-1]  (last BasicBlock3D)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._handles: list = []

    def _register_hooks(self):
        def _save_act(module, input, output):
            self._activations = output.detach()

        def _save_grad(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles = [
            self.target_layer.register_forward_hook(_save_act),
            self.target_layer.register_full_backward_hook(_save_grad),
        ]

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __call__(
        self,
        vol: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Compute GradCAM heatmap.

        Parameters
        ----------
        vol      : (1, 1, D, H, W) input volume tensor
        clinical : (1, n_clinical) optional clinical features for late-fusion models

        Returns
        -------
        (1, D, H, W) heatmap up-sampled to input resolution, values in [0, 1].
        """
        self._register_hooks()
        self.model.eval()

        # Forward pass with gradients
        vol = vol.requires_grad_(False)
        out: torch.Tensor

        try:
            if clinical is not None:
                out = self.model(vol=vol, clinical=clinical, **model_kwargs)
            else:
                out = self.model(vol=vol, **model_kwargs)
            # Scalar log-hazard — backward w.r.t. this output
            out.sum().backward()
        finally:
            self._remove_hooks()

        # α_c = global-average of gradient over spatial dims
        grad = self._gradients     # (1, C, D', H', W')
        act  = self._activations   # (1, C, D', H', W')

        alpha = grad.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam   = (alpha * act).sum(dim=1, keepdim=True)  # (1, 1, D', H', W')
        cam   = F.relu(cam, inplace=False)

        # Up-sample to input spatial dims
        d, h, w = vol.shape[-3:]
        cam = F.interpolate(cam, size=(d, h, w), mode="trilinear",
                            align_corners=False)

        # Normalise to [0, 1]
        cam_min = cam.flatten(1).min(dim=1).values.view(-1, 1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1).values.view(-1, 1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)

        return cam.squeeze(1)  # (1, D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# ATR (Attention-weighted Tumour Ratio)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(
    heatmap: torch.Tensor | np.ndarray,
    gtv_mask: torch.Tensor | np.ndarray,
    threshold_pct: float = 50.0,
) -> float:
    """
    Attention-weighted Tumour Ratio.

    ATR = |{attention_high} ∩ {GTV}| / |{GTV}|

    Parameters
    ----------
    heatmap       : (D, H, W) or (1, D, H, W) GradCAM or attention map,
                    values in [0, 1]
    gtv_mask      : (D, H, W) binary GTV segmentation (1 = tumour, 0 = background)
    threshold_pct : percentile for binarising the heatmap (default: 50th)

    Returns
    -------
    float in [0, 1]; NaN if GTV mask is empty.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    if isinstance(gtv_mask, torch.Tensor):
        gtv_mask = gtv_mask.cpu().numpy()

    heatmap  = np.squeeze(heatmap)
    gtv_mask = (np.squeeze(gtv_mask) > 0.5).astype(bool)

    n_gtv = gtv_mask.sum()
    if n_gtv == 0:
        return float("nan")

    threshold = np.percentile(heatmap, threshold_pct)
    attn_high = heatmap >= threshold

    intersection = (attn_high & gtv_mask).sum()
    return float(intersection) / float(n_gtv)


def compute_cohort_atr(
    model: nn.Module,
    dataset,
    gtv_masks: List[np.ndarray],
    target_layer: nn.Module,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute ATR for every patient in dataset.

    Parameters
    ----------
    model        : trained model
    dataset      : dataset class with __getitem__ returning dicts with 'vol'
    gtv_masks    : list of (D, H, W) binary GTV arrays, one per patient
    target_layer : GradCAM target layer (see GradCAM3D docstring)

    Returns
    -------
    (N,) array of ATR values, NaN for missing GTV.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gradcam = GradCAM3D(model.to(device), target_layer)
    atrs    = np.full(len(dataset), float("nan"))

    for i in range(len(dataset)):
        sample = dataset[i]
        vol    = sample.get("vol")
        if vol is None:
            continue

        vol_t = vol.unsqueeze(0).to(device)     # (1, 1, D, H, W)
        clin  = sample.get("clinical")
        clin_t = clin.unsqueeze(0).to(device) if clin is not None else None

        try:
            heatmap = gradcam(vol_t, clinical=clin_t)  # (1, D, H, W)
            atrs[i] = compute_atr(heatmap.squeeze(0).cpu().numpy(),
                                  gtv_masks[i] if i < len(gtv_masks) else None)
        except Exception:
            atrs[i] = float("nan")

    return atrs
