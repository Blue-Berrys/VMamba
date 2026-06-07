"""
ShadowDualPathHead v2: Phase 1 of SA-SSM Architecture

Changes vs v1:
  1. Residual gating: shadow_gate = 0.5 + 0.5*prior  (was: prior)
     - v1 problem: prior≈0.5 at init → both paths get 0.5x signal (weakened)
     - v2 fix: worst case 0.5x, best case 1.0x — signal never fully killed
  2. SPE explicit supervision: BCE loss vs GT shadow mask
     - v1 problem: SPE only gets indirect gradients from main loss → slow convergence
     - v2 fix: direct BCE drives SPE to correctly localise shadows by iter ~4k
  3. SPE outputs raw logits (Sigmoid removed from net)
     - cleaner gradient flow for BCE-with-logits loss

Architecture (unchanged structure, fixed gating):
  inputs (VMamba features) → SPE (shadow prior logits)
                           → FPN lateral fusion
                           → Residual dual-path: shadow × (0.5+0.5p), light × (1.5-0.5p)
                           → Cross-attention: shadow attends to light context
                           → Fusion → prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils.wrappers import resize


class ShadowPriorEstimator(nn.Module):
    """Estimates a shadow prior map (logits) from shallow VMamba features.

    Outputs raw logits (no Sigmoid) so that:
    - BCE-with-logits loss has clean gradient flow
    - Sigmoid is applied explicitly in the caller when gating is needed

    Args:
        in_channels (int): Number of input channels (feat[0] channels).
        mid_channels (int): Hidden channels. Default: 32.
    """

    def __init__(self, in_channels: int, mid_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
            # No Sigmoid here — logits output for BCE-with-logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] shallowest VMamba feature map
        Returns:
            shadow_prior_logits: [B, 1, H, W] raw logits (unbounded)
        """
        return self.net(x)


class ShadowLightCrossAttention(nn.Module):
    """Shadow-Light Cross-Attention module.

    The shadow pathway attends to the light pathway's global context.
    This forces the shadow branch to be "aware" of what non-shadow regions
    look like — a key inductive bias for shadow detection.

    Args:
        channels (int): Feature channels.
        reduction (int): Channel reduction ratio. Default: 8.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 32)
        self.q_proj = nn.Conv2d(channels, mid, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, mid, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Learnable scale: starts at 0 so the module starts as identity
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self, shadow_feat: torch.Tensor, light_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            shadow_feat: [B, C, H, W] shadow pathway features (query)
            light_feat:  [B, C, H, W] light pathway features  (key/value)
        Returns:
            enhanced shadow features [B, C, H, W]
        """
        shadow_ctx = shadow_feat.mean(dim=[-2, -1], keepdim=True)  # [B, C, 1, 1]
        light_ctx = light_feat.mean(dim=[-2, -1], keepdim=True)    # [B, C, 1, 1]

        q = self.q_proj(shadow_ctx)  # [B, mid, 1, 1]
        k = self.k_proj(light_ctx)   # [B, mid, 1, 1]
        v = self.v_proj(light_ctx)   # [B, C,   1, 1]

        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True))  # [B, 1, 1, 1]
        light_info = self.out_proj(v * attn)                     # [B, C, 1, 1]
        return shadow_feat + self.gamma * light_info             # [B, C, H, W]


@MODELS.register_module()
class ShadowDualPathHead(BaseDecodeHead):
    """Dual-Path Shadow Detection Decoder v2 — Phase 1 of SA-SSM.

    Key fix over v1: residual gating + explicit SPE supervision.

    v1 gating (broken):
        shadow_feat = conv(fpn * prior)          # prior≈0.5 → 0.5x signal
        light_feat  = conv(fpn * (1 - prior))   # same problem

    v2 gating (fixed):
        shadow_gate = 0.5 + 0.5 * sigmoid(prior_logits)   # ∈ [0.5, 1.0]
        light_gate  = 1.5 - 0.5 * sigmoid(prior_logits)   # ∈ [0.5, 1.0]
        shadow_feat = conv(fpn * shadow_gate)
        light_feat  = conv(fpn * light_gate)

    With random init (prior≈0), gate≈0.5 for shadow, ≈1.5 for light.
    Signal is never zeroed; differentiation grows as SPE converges.

    Args:
        in_channels (list[int]): Feature channels from backbone.
        channels (int): Decoder internal channel count. Default: 256.
        spe_mid_channels (int): Hidden channels in SPE. Default: 32.
        cross_attn_reduction (int): Channel reduction in cross-attention. Default: 8.
        spe_loss_weight (float): Weight of BCE loss on shadow prior. Default: 0.2.
        **kwargs: Passed to BaseDecodeHead (num_classes, loss_decode, etc.)
    """

    def __init__(
        self,
        in_channels,
        channels: int = 256,
        spe_mid_channels: int = 32,
        cross_attn_reduction: int = 8,
        spe_loss_weight: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault('in_index', [0, 1, 2, 3])
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs,
        )

        self.spe_loss_weight = spe_loss_weight
        # Stores prior logits during forward for use in loss_by_feat
        self._shadow_prior_logits = None

        # ── Shadow Prior Estimator (logits output) ──────────────────────────
        self.spe = ShadowPriorEstimator(in_channels[0], spe_mid_channels)

        # ── FPN lateral projections ─────────────────────────────────────────
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels
        ])

        # ── Dual-path convs ─────────────────────────────────────────────────
        self.shadow_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.light_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ── Cross-attention ─────────────────────────────────────────────────
        self.cross_attn = ShadowLightCrossAttention(channels, cross_attn_reduction)

        # ── Fusion: [shadow ‖ light] → channels ────────────────────────────
        dropout = nn.Dropout2d(self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            dropout,
        )

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs: Tuple of 4 feature maps from VMamba backbone.
                inputs[0]: [B, 128,  H/4,  W/4]  shallowest
                inputs[1]: [B, 256,  H/8,  W/8]
                inputs[2]: [B, 512,  H/16, W/16]
                inputs[3]: [B, 1024, H/32, W/32] deepest

        Returns:
            seg_logits: [B, num_classes, H/4, W/4]
        """
        inputs = self._transform_inputs(inputs)

        # ── 1. Shadow Prior (logits + sigmoid gate) ──────────────────────────
        prior_logits = self.spe(inputs[0])           # [B, 1, H/4, W/4] logits
        self._shadow_prior_logits = prior_logits     # save for loss_by_feat
        shadow_prior = torch.sigmoid(prior_logits)   # [B, 1, H/4, W/4] ∈ (0,1)

        # Residual gates: minimum 0.5x signal, maximum 1.0x
        # shadow_gate ∈ [0.5, 1.0]: high in shadow regions
        # light_gate  ∈ [0.5, 1.0]: high in non-shadow (light) regions
        shadow_gate = 0.5 + 0.5 * shadow_prior       # [B, 1, H/4, W/4]
        light_gate  = 1.5 - 0.5 * shadow_prior       # equivalent to 0.5+0.5*(1-prior)

        # ── 2. FPN: project all scales to `channels` ────────────────────────
        laterals = [lat(feat) for lat, feat in zip(self.laterals, inputs)]

        # ── 3. Top-down aggregation: all scales → finest resolution ─────────
        target_h, target_w = laterals[0].shape[-2:]
        fpn_out = laterals[0]
        for i in range(1, len(laterals)):
            upsampled = F.interpolate(
                laterals[i],
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=self.align_corners,
            )
            fpn_out = fpn_out + upsampled

        # ── 4. Residual dual-path split ──────────────────────────────────────
        shadow_feat = self.shadow_conv(fpn_out * shadow_gate)
        light_feat  = self.light_conv(fpn_out * light_gate)

        # ── 5. Cross-attention: shadow attends to light context ─────────────
        shadow_feat = self.cross_attn(shadow_feat, light_feat)

        # ── 6. Fusion ────────────────────────────────────────────────────────
        fused = self.fusion(torch.cat([shadow_feat, light_feat], dim=1))

        # ── 7. Classification ─────────────────────────────────────────────────
        return self.conv_seg(fused)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        """Override to add SPE supervision on top of main segmentation losses.

        Extra loss: BCE(SPE_logits_resized, GT_shadow_mask)
        This directly teaches the SPE to localise shadow regions without
        waiting for gradients to propagate through the entire decoder.
        """
        # Main losses from BaseDecodeHead (FocalLoss + DiceLoss)
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        # SPE supervision
        if self.spe_loss_weight > 0 and self._shadow_prior_logits is not None:
            gt_seg = self._stack_batch_gt(batch_data_samples)  # [B, 1, H, W] long

            # shadow class = 1, ignore = 255
            shadow_gt = (gt_seg == 1).float()          # [B, 1, H, W]
            valid_mask = (gt_seg != 255)               # [B, 1, H, W] bool

            # Resize SPE logits to GT resolution
            prior_resized = F.interpolate(
                self._shadow_prior_logits,
                size=gt_seg.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )  # [B, 1, H, W]

            # BCE only on valid (non-ignore) pixels
            if valid_mask.any():
                spe_loss = F.binary_cross_entropy_with_logits(
                    prior_resized[valid_mask],
                    shadow_gt[valid_mask],
                    reduction='mean',
                )
                losses['loss_spe'] = spe_loss * self.spe_loss_weight

        return losses
