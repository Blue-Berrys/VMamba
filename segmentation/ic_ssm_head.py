"""
IC-SSM Head: Illumination Contrast State Space Module for Shadow Detection
==========================================================================

论文创新点:
    1. 首个将 SSM (State Space Model) 引入阴影检测 decoder 的方法
    2. 物理驱动的光照对比建模:
         阴影本质 = 局部光照衰减, shadow_pixel ≈ scene_illumination × α, α ∈ (0,1)
         → 通过计算每个像素与场景全局光照参考的偏差来定位阴影
    3. IC-SSM 扫描: 用 VMamba VSSBlock 对光照对比图进行 4 方向 SSM 扫描
         → SSM 状态累积光照历史, 在从亮区扫描到暗区时自然产生响应
         → O(n) 复杂度, 对比 ShadowFormer 的 O(n²) attention
    4. 两级先验监督 (SPE + BCE): 对比图预测的先验图直接用 GT mask 监督
    5. Shadow Boundary Supervision (SBS, 新增):
         阴影边界 = 光照快速过渡区 (penumbra), 是误检/漏检的主要来源
         → 边界预测分支 + 边界 BCE loss (边界 GT 由 mask 膨胀/腐蚀自动生成)
         → 边界置信度图反向增强主流特征, 提高边界像素精度

架构图:
    VMamba [C1,C2,C3,C4]
         ↓  FPN 融合
       F_fpn [B, 256, H/4, W/4]
         ↓  IC-SSM 模块
         |  1. SIR: 场景光照参考 μ = GlobalPool(F_fpn)
         |  2. ICM: 光照对比图 ΔF = F_fpn - μ
         |  3. DCP: VSSBlock(proj(ΔF)) → 方向感知对比特征
         |  4. SPE: prior_logits = Conv(DCP), BCE 监督
         |  5. CGE: F_enh = F_fpn + sigmoid(prior) * DCP
         ↓  SBS 模块 (新增)
         |  6. 边界预测: boundary_logits = BoundaryConv(F_enh)
         |  7. 边界增强: F_final = F_enh + γ_b * sigmoid(boundary) * F_enh
         |  8. 边界 GT = dilate(mask) - erode(mask), 自动生成
         ↓
       分类头 → seg_logits

参考: VMamba (VSSBlock), ShadowFormer (illumination attention), BDRAR (bidirectional),
      CASENet (boundary supervision), SegFix (boundary refinement)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils.wrappers import resize

# ── 导入 VMamba VSSBlock ─────────────────────────────────────────────────────
def _get_vssblock():
    """动态导入 VSSBlock，支持在 VMamba segmentation 项目中运行。"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(this_dir, "../classification")),
        os.path.normpath(os.path.join(this_dir, "../../VMamba/classification")),
        os.path.normpath(os.path.join(this_dir, "../../../classification")),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from models.vmamba import VSSBlock
        return VSSBlock
    except ImportError:
        return None

VSSBlock = _get_vssblock()


# ── 后备: 纯 PyTorch 近似扫描 (VSSBlock 不可用时) ────────────────────────────
class _ApproxSSMScan(nn.Module):
    """
    近似 SSM 扫描: 用因果卷积 + 深度可分卷积模拟方向性序列建模。

    对每个方向 d, 等价于:
        h_k = λ * h_{k-1} + (1-λ) * x_k   (离散 SSM 递推)
    用大核因果卷积近似 (按方向 permute 后做 depthwise conv1d)。
    """
    def __init__(self, channels: int, n_dirs: int = 4):
        super().__init__()
        self.n_dirs = n_dirs
        self.decay = nn.Parameter(torch.full((n_dirs, channels, 1), 0.9))
        self.proj_in  = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_out = nn.Sequential(
            nn.Conv2d(channels * n_dirs, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _scan_1d(self, x: torch.Tensor, dim: int, reverse: bool,
                 decay: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if reverse:
            x = x.flip(-1)
        out = torch.zeros_like(x)
        h = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        lam = torch.sigmoid(decay).squeeze(-1)
        for t in range(L):
            h = lam * h + (1 - lam) * x[:, :, t]
            out[:, :, t] = h
        if reverse:
            out = out.flip(-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj_in(x)
        outs = []
        for d_idx, (dim, reverse) in enumerate(
            [(3, False), (3, True), (2, False), (2, True)]
        ):
            seq = x.flatten(2) if dim == 3 else x.permute(0, 1, 3, 2).flatten(2)
            scanned = self._scan_1d(seq, dim, reverse, self.decay[d_idx])
            if dim == 3:
                out = scanned.reshape(B, C, H, W)
            else:
                out = scanned.reshape(B, C, W, H).permute(0, 1, 3, 2)
            outs.append(out)
        return self.proj_out(torch.cat(outs, dim=1))


# ── BrightnessGuidedSIR (BG-SIR, IC-SSM 光照先验增强) ────────────────────────
class BrightnessGuidedSIR(nn.Module):
    """
    亮度引导场景光照参考 (Brightness-Guided Scene Illumination Reference)

    问题:
        原始 IC-SSM 用全局平均池化 μ = mean(F) 作为场景光照参考。
        但 μ 包含了阴影像素, 使参考值被压低, 导致阴影的光照对比度偏小, FNR 偏高。

    物理依据 (Retinex):
        真实光照 ≈ 场景中亮像素区域的亮度 (非阴影区接近真实光照强度)。
        因此应用亮度加权均值代替均匀均值:
            bright_ref = Σ(F * w) / Σ(w),  w = sigmoid(亮度估计器(F))

    设计:
        1. 轻量亮度估计器: C → C//8 → C → Sigmoid, 输出逐像素权重 w ∈ (0,1)
        2. 亮度加权均值: bright_ref = Σ(F*w) / Σ(w)  [B,C,1,1]
           - 高权重区域 (亮/非阴影) 主导参考值 → 参考更接近真实光照
        3. 局部光照参考: local_ref = AvgPool(large-kernel)(F)  [B,C,H,W]
           - 捕获空间光照梯度 (光照不均匀场景)
        4. 残差融合: 两者由可学习参数 alpha, beta 混合 (初始=0, 退化为原始均值)
           ref = global_mean + alpha*(bright_ref - global_mean)
                             + beta*(local_ref - global_mean)

    论文贡献:
        首次在 SSM 阴影检测中引入物理驱动的光照参考估计:
        全局亮度加权 + 局部空间光照建模, 使 IC-SSM 对比图更具判别性
    """

    def __init__(self, channels: int):
        super().__init__()
        # 亮度估计器: 轻量级, 参数极少
        self.brightness_est = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),   # 输出权重 ∈ (0, 1)
        )
        # 局部光照估计: 大核均值池化 (空间光照梯度)
        self.local_est = nn.Sequential(
            nn.AvgPool2d(kernel_size=15, stride=1, padding=7),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # 残差系数: 从 0 开始学, 保证训练初期等价于原始 IC-SSM
        self.alpha = nn.Parameter(torch.zeros(1))  # 亮度加权参考权重
        self.beta  = nn.Parameter(torch.zeros(1))  # 局部光照参考权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            ref: [B, C, H, W] 空间自适应光照参考 (可广播)
                 训练初期 → global_mean [B, C, 1, 1]
                 训练后期 → 亮度引导 + 局部空间参考的混合
        """
        # 基线: 全局均值 (原始 SIR)
        global_mean = x.mean(dim=[2, 3], keepdim=True)   # [B, C, 1, 1]

        # 亮度加权均值: 亮区 (非阴影) 贡献更大 → 更接近真实光照
        w = self.brightness_est(x)                        # [B, C, H, W]
        w_sum   = w.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        bright_ref = (x * w).sum(dim=[2, 3], keepdim=True) / w_sum  # [B, C, 1, 1]

        # 局部光照参考: 大核均值捕获空间光照梯度
        local_ref = self.local_est(x)                    # [B, C, H, W]

        # 残差混合: alpha, beta 从 0 开始学习
        ref = (global_mean
               + self.alpha * (bright_ref - global_mean)
               + self.beta  * (local_ref  - global_mean))
        return ref   # [B, C, H, W] (local_ref 使 ref 具有空间变化)


# ── IlluminationContrastModule (IC-SSM 核心) ─────────────────────────────────
class IlluminationContrastModule(nn.Module):
    """
    光照对比状态空间模块 (Illumination Contrast SSM, IC-SSM)

    物理动机:
        - 场景光照参考 μ = 全局平均池化(特征) ≈ 场景平均亮度
        - 光照对比图 ΔF = F - μ:  阴影像素 ΔF < 0 (比平均暗)
        - SSM 扫描 ΔF: 从亮区到暗区扫描时, 状态累积亮区信息,
          遇到暗区时产生大偏差 → 自然检测阴影边界

    BG-SIR 增强 (可选):
        当 use_bg_sir=True 时, 用 BrightnessGuidedSIR 代替全局均值 SIR,
        实现物理驱动的自适应光照参考估计, 提升对比图判别性。
    """

    def __init__(
        self,
        channels: int,
        use_vssblock: bool = True,
        ssm_d_state: int = 16,
        ssm_ratio: float = 1.0,
        drop_path: float = 0.1,
        use_bg_sir: bool = False,
    ):
        super().__init__()
        self.channels = channels

        # BG-SIR: 亮度引导光照参考 (光照先验增强)
        self.bg_sir = BrightnessGuidedSIR(channels) if use_bg_sir else None

        self.contrast_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        if use_vssblock and VSSBlock is not None:
            self.vss_block = VSSBlock(
                hidden_dim=channels,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm,
                channel_first=False,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_conv=3,
                ssm_conv_bias=True,
                ssm_drop_rate=0.0,
                ssm_init="v0",
                forward_type="v2",
                mlp_ratio=0.0,
            )
            self._use_vss = True
        else:
            self.vss_block = None
            self._use_vss = False

        self.approx_ssm = _ApproxSSMScan(channels)

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.prior_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # 1. 场景光照参考 (SIR)
        # BG-SIR: 亮度加权 + 局部空间参考 (更接近真实光照)
        # 原始: 全局均值 (阴影像素拉低参考, 对比度偏小)
        if self.bg_sir is not None:
            scene_ref = self.bg_sir(x)   # [B, C, H, W] 空间自适应
        else:
            scene_ref = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # 2. 光照对比图 (ICM)
        contrast = x - scene_ref
        contrast = self.contrast_proj(contrast)

        # 3. 方向感知对比传播 (DCP via SSM)
        use_vss = self._use_vss and contrast.is_cuda
        if use_vss:
            contrast_cl = contrast.permute(0, 2, 3, 1).contiguous()
            dcp_cl = self.vss_block(contrast_cl)
            dcp = dcp_cl.permute(0, 3, 1, 2).contiguous()
        else:
            dcp = self.approx_ssm(contrast)

        dcp = self.out_proj(dcp)

        # 4. 阴影先验估计 (SPE)
        prior_logits = self.prior_head(dcp)

        # 5. 对比引导特征增强 (CGE)
        shadow_gate = torch.sigmoid(prior_logits)
        x_enhanced = x + self.gamma * (shadow_gate * dcp)

        return x_enhanced, prior_logits


# ── ScaleAwareFusion (SASF, 第三个创新点) ────────────────────────────────────
class ScaleAwareFusion(nn.Module):
    """
    尺度感知阴影融合 (Scale-Aware Shadow Fusion, SASF)

    创新动机:
        阴影的尺度差异极大: 路灯投影可能只有 10px 宽，建筑阴影可能覆盖 300px。
        传统 FPN 对所有尺度特征简单相加, 大尺度特征掩盖了细粒度边缘信息,
        导致小阴影和阴影边界漏检 (FNR 高的另一个来源)。

    设计:
        1. Scale Estimator: 从最细粒度特征预测每个位置的尺度偏好权重
           scale_weights = softmax(Conv(laterals[0]))  [B, N_scales, H, W]
        2. 空间自适应加权融合: 不同位置用不同尺度权重组合 FPN 各层
           fused = Σ_i (scale_weight_i * lateral_i)
        3. 残差连接: fused = scale_fused + simple_sum (保留原 FPN 信息)

    与 FPN 的区别:
        - 传统 FPN: 固定相加 (每个位置权重相同)
        - SASF: 位置自适应权重 (小阴影区域偏向精细尺度, 大阴影区域偏向粗尺度)

    计算开销: 仅增加一个 Conv(C, N/4, 3) + Conv(C/4, N, 1), 参数量极少
    """

    def __init__(self, channels: int, n_scales: int = 4):
        super().__init__()
        self.n_scales = n_scales

        # 尺度权重预测器: 用最细粒度特征预测每个位置偏向哪个尺度
        self.scale_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, n_scales, 1),   # [B, N, H, W] logits
        )

        # 残差权重 (初始 0, 训练初期退化为普通 FPN)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, laterals: list) -> torch.Tensor:
        """
        Args:
            laterals: list of N [B, C, H, W], 已上采样到同一分辨率

        Returns:
            fused: [B, C, H, W] 尺度感知融合特征
        """
        # 普通 FPN 和 (基线)
        simple_sum = sum(laterals)

        # 尺度权重: [B, N, H, W] → softmax → [B, N, 1, H, W]
        scale_logits  = self.scale_estimator(laterals[0])
        scale_weights = F.softmax(scale_logits, dim=1).unsqueeze(2)

        # 堆叠并加权: [B, N, C, H, W] * [B, N, 1, H, W] → sum → [B, C, H, W]
        stacked = torch.stack(laterals, dim=1)        # [B, N, C, H, W]
        scale_fused = (stacked * scale_weights).sum(dim=1)

        # 残差融合: alpha 从 0 开始学, 保证训练初期稳定
        return simple_sum + self.alpha * (scale_fused - simple_sum)


# ── ShadowBoundaryModule (SBS 核心, 新创新点) ────────────────────────────────
class ShadowBoundaryModule(nn.Module):
    """
    阴影边界感知模块 (Shadow Boundary Supervision, SBS)

    创新动机:
        阴影边界 = 光照从受遮挡到未遮挡的过渡带 (penumbra region)。
        该区域像素值处于阴影/非阴影之间, 是现有方法 FNR 高的主要来源。
        IC-SSM 建模全局光照对比, 但对局部边界过渡感知不足。
        SBS 专门学习边界过渡模式, 与 IC-SSM 互补。

    设计:
        1. 扩张卷积 (dilation=2) 捕获边界上下文 (感受野覆盖过渡带)
        2. 边界预测 logits → BCE loss 监督 (GT 由 mask 膨胀-腐蚀自动生成)
        3. 边界置信度图门控: 在边界区域加强特征表达
           F_out = F_in + γ_b * sigmoid(boundary_logits) * F_in

    边界 GT 生成 (无需额外标注):
        boundary_mask = max_pool(shadow_mask, k=5) - min_pool(shadow_mask, k=5)
        即膨胀结果 - 腐蚀结果 = 边界环 (宽度约 5px)
    """

    def __init__(self, channels: int, boundary_kernel: int = 5):
        super().__init__()
        self.boundary_kernel = boundary_kernel

        # 扩张卷积感知边界过渡 (dilation=2 使感受野覆盖过渡带两侧)
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),  # 边界置信度 logits
        )

        # 连续半影/软阴影置信分支: 与 binary boundary branch 共享输入特征,
        # 但预测 0-1 连续阴影强度, 避免把 penumbra 压成硬边界。
        self.penumbra_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
        )

        # Boundary gate keeps the pre-penumbra checkpoint behavior.  The new
        # penumbra gate starts from zero so loading an old BG-SIR checkpoint is
        # functionally neutral until the auxiliary task learns useful signal.
        self.gamma_b = nn.Parameter(torch.zeros(1))
        self.gamma_p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] IC-SSM 增强后的特征

        Returns:
            x_out: [B, C, H, W] 边界感知增强特征
            boundary_logits: [B, 1, H, W] 边界预测 (训练时 BCE 监督)
            penumbra_logits: [B, 1, H, W] 连续半影/软阴影置信预测
        """
        boundary_logits = self.boundary_conv(x)          # [B, 1, H, W]
        penumbra_logits = self.penumbra_conv(x)          # [B, 1, H, W]
        boundary_attn = torch.sigmoid(boundary_logits)
        penumbra_attn = torch.sigmoid(penumbra_logits)

        # Keep the old binary boundary enhancement and add a zero-initialized
        # learnable penumbra residual path.
        x_out = (x + self.gamma_b * boundary_attn * x +
                 self.gamma_p * penumbra_attn * x)

        return x_out, boundary_logits, penumbra_logits

    @staticmethod
    def get_boundary_gt(shadow_mask: torch.Tensor,
                        kernel_size: int = 5) -> torch.Tensor:
        """
        从 shadow_mask 自动生成边界 GT。

        Args:
            shadow_mask: [B, 1, H, W] float, 值为 0.0 或 1.0
            kernel_size: 膨胀/腐蚀核大小, 决定边界宽度

        Returns:
            boundary: [B, 1, H, W] float, 1 表示边界像素
        """
        pad = kernel_size // 2
        # 膨胀: max pool
        dilated = F.max_pool2d(
            shadow_mask, kernel_size, stride=1, padding=pad)
        # 腐蚀: -max_pool(-x)
        eroded = -F.max_pool2d(
            -shadow_mask, kernel_size, stride=1, padding=pad)
        # 边界 = 膨胀 - 腐蚀 (shadow 边界环)
        boundary = (dilated - eroded).clamp(0, 1)
        return boundary

    @staticmethod
    def get_soft_penumbra_gt(shadow_mask: torch.Tensor,
                             band_width: int = 8,
                             tau: float = 2.0):
        """
        从二值 shadow mask 生成连续 soft-shadow / penumbra 伪标签。

        signed distance > 0 表示阴影内部, < 0 表示非阴影外部。
        sigmoid(signed_distance / tau) 给出 0-1 连续阴影强度;
        abs(signed_distance) <= band_width 作为半影监督带。
        """
        mask_cpu = shadow_mask.detach().float().cpu()
        soft_maps, band_maps, signed_maps = [], [], []

        try:
            import cv2
            cv2_distance = True
        except Exception:
            cv2_distance = False
            from scipy import ndimage

        import numpy as np

        for b in range(mask_cpu.shape[0]):
            mask = (mask_cpu[b, 0].numpy() > 0.5).astype('uint8')
            if mask.max() == 0:
                signed = np.full(mask.shape, -10.0 * band_width, dtype=np.float32)
            elif mask.min() == 1:
                signed = np.full(mask.shape, 10.0 * band_width, dtype=np.float32)
            else:
                if cv2_distance:
                    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
                else:
                    dist_in = ndimage.distance_transform_edt(mask)
                    dist_out = ndimage.distance_transform_edt(1 - mask)
                signed = (dist_in - dist_out).astype(np.float32)

            scaled = np.clip(signed / max(tau, 1e-6), -20.0, 20.0)
            soft = 1.0 / (1.0 + np.exp(-scaled))
            band = (np.abs(signed) <= float(band_width)).astype(np.float32)
            soft_maps.append(soft.astype(np.float32))
            band_maps.append(band)
            signed_maps.append(signed.astype(np.float32))

        device = shadow_mask.device
        dtype = shadow_mask.dtype
        soft = torch.from_numpy(np.stack(soft_maps))[:, None].to(device=device, dtype=dtype)
        band = torch.from_numpy(np.stack(band_maps))[:, None].to(device=device, dtype=torch.bool)
        signed = torch.from_numpy(np.stack(signed_maps))[:, None].to(device=device, dtype=dtype)
        return soft, band, signed


# ── IC-SSM Head (含 SBS) ──────────────────────────────────────────────────────
@MODELS.register_module()
class ICShadowHead(BaseDecodeHead):
    """
    Illumination Contrast SSM Shadow Detection Head
    with Shadow Boundary Supervision (IC-SSM + SBS)

    论文贡献:
        1. IC-SSM: 首次在阴影检测 decoder 引入 VMamba VSSBlock
           - 物理驱动的光照对比建模, O(n) 复杂度
           - 先验 BCE 监督 (SPE loss)
        2. SBS: 阴影边界显式监督
           - 边界 GT 自动生成 (无需额外标注)
           - 扩张卷积感知 penumbra 过渡带
           - 边界门控增强主流特征
           - 与 IC-SSM 互补: IC-SSM 管全局, SBS 管局部边界

    Args:
        in_channels (list[int]): backbone 各层通道数
        channels (int): decoder 内部通道数. Default: 256
        ssm_d_state (int): SSM state 维度. Default: 16
        ssm_ratio (float): SS2D expansion ratio. Default: 1.0
        ssm_drop_path (float): SSM drop path rate. Default: 0.1
        spe_loss_weight (float): 先验 BCE loss 权重. Default: 0.3
        boundary_loss_weight (float): 边界 BCE loss 权重. Default: 0.4
        boundary_kernel (int): 边界 GT 生成核大小. Default: 5
        soft_boundary_loss_weight (float): soft penumbra 回归 loss 权重.
        penumbra_grad_loss_weight (float): penumbra 梯度匹配 loss 权重.
        penumbra_mono_loss_weight (float): signed-distance 单调约束权重.
        boundary_consistency_loss_weight (float): penumbra 与边界一致性权重.
        penumbra_refine_logits (bool): use the penumbra confidence map to
            calibrate final segmentation logits. Default: False.
        penumbra_refine_boundary_gate (bool): gate penumbra logit refinement
            by predicted boundary confidence. Default: True.
        penumbra_refine_uncertain_gate (bool): gate refinement by low
            binary-logit margin so confident pixels are left unchanged. Default: False.
        penumbra_refine_margin (float): logit margin below which refinement is active. Default: 1.5.
        penumbra_refine_sharpness (float): uncertainty gate sigmoid sharpness. Default: 2.0.
        penumbra_refine_max_delta (float): maximum absolute logit shift after
            tanh-bounded scaling. Default: 2.0.
        tversky_loss_weight (float): Tversky loss 权重, 0 表示关闭. Default: 0.0
        tversky_alpha (float): Tversky loss FP 惩罚系数. Default: 0.3
        tversky_beta (float): Tversky loss FN 惩罚系数 (>alpha 则更重视 FNR). Default: 0.7
        use_bg_sir (bool): 是否启用 BG-SIR 光照先验增强. Default: False
        **kwargs: 传给 BaseDecodeHead
    """

    def __init__(
        self,
        in_channels,
        channels: int = 256,
        ssm_d_state: int = 16,
        ssm_ratio: float = 1.0,
        ssm_drop_path: float = 0.1,
        spe_loss_weight: float = 0.3,
        boundary_loss_weight: float = 0.4,
        boundary_kernel: int = 5,
        soft_boundary_loss_weight: float = 0.0,
        penumbra_grad_loss_weight: float = 0.0,
        penumbra_mono_loss_weight: float = 0.0,
        boundary_consistency_loss_weight: float = 0.0,
        penumbra_band_width: int = 8,
        penumbra_tau: float = 2.0,
        penumbra_refine_logits: bool = False,
        penumbra_refine_boundary_gate: bool = True,
        penumbra_refine_uncertain_gate: bool = False,
        penumbra_refine_margin: float = 1.5,
        penumbra_refine_sharpness: float = 2.0,
        penumbra_refine_max_delta: float = 2.0,
        use_sasf: bool = True,
        tversky_loss_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        use_bg_sir: bool = False,
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
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_kernel = boundary_kernel
        self.soft_boundary_loss_weight = soft_boundary_loss_weight
        self.penumbra_grad_loss_weight = penumbra_grad_loss_weight
        self.penumbra_mono_loss_weight = penumbra_mono_loss_weight
        self.boundary_consistency_loss_weight = boundary_consistency_loss_weight
        self.penumbra_band_width = penumbra_band_width
        self.penumbra_tau = penumbra_tau
        self.penumbra_refine_logits = penumbra_refine_logits
        self.penumbra_refine_boundary_gate = penumbra_refine_boundary_gate
        self.penumbra_refine_uncertain_gate = penumbra_refine_uncertain_gate
        self.penumbra_refine_margin = penumbra_refine_margin
        self.penumbra_refine_sharpness = penumbra_refine_sharpness
        self.penumbra_refine_max_delta = penumbra_refine_max_delta
        self.tversky_loss_weight = tversky_loss_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.use_bg_sir = use_bg_sir
        self._prior_logits = None
        self._boundary_logits = None
        self._penumbra_logits = None
        self.penumbra_logit_scale_raw = nn.Parameter(torch.zeros(1))

        n_scales = (len(in_channels) if isinstance(in_channels, (list, tuple))
                    else 4)

        # ── FPN 侧向投影 ────────────────────────────────────────────────────
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in (in_channels if isinstance(in_channels, (list, tuple))
                          else [in_channels])
        ])

        # ── SASF 模块 (创新点 1: 尺度感知融合) ──────────────────────────────
        self.sasf = ScaleAwareFusion(channels, n_scales=n_scales) if use_sasf else None

        # ── IC-SSM 模块 (创新点 2: 光照对比 SSM + 可选 BG-SIR 先验增强) ────
        self.ic_ssm = IlluminationContrastModule(
            channels=channels,
            use_vssblock=True,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            drop_path=ssm_drop_path,
            use_bg_sir=use_bg_sir,
        )

        # ── SBS 模块 (创新点 3: 边界显式监督) ───────────────────────────────
        self.boundary_module = ShadowBoundaryModule(
            channels=channels,
            boundary_kernel=boundary_kernel,
        )

        # ── Fusion: 增强特征 → 预测 ─────────────────────────────────────────
        dropout = (nn.Dropout2d(self.dropout_ratio)
                   if self.dropout_ratio > 0 else nn.Identity())
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            dropout,
        )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        target_h, target_w = inputs[0].shape[-2:]

        # 1. FPN: 各尺度投影到 channels
        laterals = [lat(feat) for lat, feat in zip(self.laterals, inputs)]

        # 2. 上采样到最细粒度
        aligned = [laterals[0]]
        for i in range(1, len(laterals)):
            up = F.interpolate(
                laterals[i],
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=self.align_corners,
            )
            aligned.append(up)

        # 3. SASF: 尺度感知融合 (或退化为简单求和)
        if self.sasf is not None:
            fpn_out = self.sasf(aligned)
        else:
            fpn_out = sum(aligned)

        # 4. IC-SSM: 全局光照对比增强
        fpn_enh, prior_logits = self.ic_ssm(fpn_out)
        self._prior_logits = prior_logits

        # 5. SBS: 局部边界感知增强
        fpn_boundary, boundary_logits, penumbra_logits = self.boundary_module(fpn_enh)
        self._boundary_logits = boundary_logits
        self._penumbra_logits = penumbra_logits

        # 6. Fusion + 分类
        fused = self.fusion(fpn_boundary)
        seg_logits = self.conv_seg(fused)

        if self.penumbra_refine_logits:
            penumbra_prob = torch.sigmoid(penumbra_logits)
            delta = penumbra_prob - 0.5
            if self.penumbra_refine_boundary_gate:
                delta = delta * torch.sigmoid(boundary_logits)
            if self.penumbra_refine_uncertain_gate:
                if seg_logits.shape[1] >= 2:
                    margin = (seg_logits[:, 1:2] - seg_logits[:, 0:1]).abs()
                else:
                    margin = seg_logits.abs()
                uncertain_gate = torch.sigmoid(
                    (self.penumbra_refine_margin - margin) *
                    self.penumbra_refine_sharpness)
                delta = delta * uncertain_gate

            scale = (torch.tanh(self.penumbra_logit_scale_raw) *
                     self.penumbra_refine_max_delta)
            delta = delta * scale

            if seg_logits.shape[1] >= 2:
                seg_logits = seg_logits.clone()
                seg_logits[:, 0:1] = seg_logits[:, 0:1] - delta
                seg_logits[:, 1:2] = seg_logits[:, 1:2] + delta
            else:
                seg_logits = seg_logits + delta

        return seg_logits

    def loss_by_feat(self, seg_logits, batch_data_samples):
        """主分割 loss + SPE 先验 loss + SBS 边界 loss。"""
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        gt_seg = self._stack_batch_gt(batch_data_samples)  # [B, 1, H, W]
        shadow_gt  = (gt_seg == 1).float()
        valid_mask = (gt_seg != 255)
        valid_bool = valid_mask.bool()
        boundary_gt = None

        # ── SPE loss (先验 BCE) ──────────────────────────────────────────────
        if self.spe_loss_weight > 0 and self._prior_logits is not None:
            prior_up = F.interpolate(
                self._prior_logits,
                size=gt_seg.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
            if valid_mask.any():
                spe_loss = F.binary_cross_entropy_with_logits(
                    prior_up[valid_mask],
                    shadow_gt[valid_mask],
                    reduction='mean',
                )
                losses['loss_spe'] = spe_loss * self.spe_loss_weight

        # ── Tversky loss (FNR 专项惩罚) ─────────────────────────────────────
        # 针对当前 FNR >> FPR 问题 (FNR≈7.5%, FPR≈1%):
        # beta=0.7 > alpha=0.3, 使 FN 受到 2.3x 惩罚, 迫使模型减少漏检
        if self.tversky_loss_weight > 0 and seg_logits is not None:
            # shadow 类通道 logit (num_classes=2, index=1)
            if seg_logits.shape[1] >= 2:
                shadow_logit = seg_logits[:, 1:2, :, :]
            else:
                shadow_logit = seg_logits
            # 上采样到 GT 分辨率
            if shadow_logit.shape[-2:] != gt_seg.shape[-2:]:
                shadow_logit = F.interpolate(
                    shadow_logit, size=gt_seg.shape[-2:],
                    mode='bilinear', align_corners=self.align_corners)
            prob = torch.sigmoid(shadow_logit)
            if valid_mask.any():
                prob_v = prob[valid_mask].view(-1)
                gt_v   = shadow_gt[valid_mask].view(-1)
                smooth = 1e-5
                TP = (prob_v * gt_v).sum()
                FP = (prob_v * (1.0 - gt_v)).sum()
                FN = ((1.0 - prob_v) * gt_v).sum()
                tversky_idx = (TP + smooth) / (
                    TP + self.tversky_alpha * FP + self.tversky_beta * FN + smooth)
                losses['loss_tversky'] = (1.0 - tversky_idx) * self.tversky_loss_weight

        # ── SBS loss (边界 BCE) ──────────────────────────────────────────────
        if self.boundary_loss_weight > 0 and self._boundary_logits is not None:
            # 边界 GT: 由 shadow mask 自动生成 (膨胀 - 腐蚀)
            boundary_gt = ShadowBoundaryModule.get_boundary_gt(
                shadow_gt, self.boundary_kernel)  # [B, 1, H, W]

            boundary_up = F.interpolate(
                self._boundary_logits,
                size=gt_seg.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

            if valid_mask.any():
                b_pred = boundary_up[valid_mask]
                b_gt   = boundary_gt[valid_mask]

                # 边界像素稀少, 用 pos_weight 平衡正负样本
                pos_ratio = b_gt.mean().clamp(1e-4, 1 - 1e-4)
                pos_weight = ((1 - pos_ratio) / pos_ratio).clamp(2.0, 15.0)

                pw = torch.full((1,), pos_weight,
                                device=b_pred.device, dtype=b_pred.dtype)
                boundary_loss = F.binary_cross_entropy_with_logits(
                    b_pred,
                    b_gt,
                    pos_weight=pw,
                    reduction='mean',
                )
                losses['loss_boundary'] = boundary_loss * self.boundary_loss_weight

        # ── Penumbra-aware soft shadow confidence losses ───────────────────
        use_penumbra = (
            self._penumbra_logits is not None and
            (self.soft_boundary_loss_weight > 0 or
             self.penumbra_grad_loss_weight > 0 or
             self.penumbra_mono_loss_weight > 0 or
             self.boundary_consistency_loss_weight > 0)
        )
        if use_penumbra:
            penumbra_gt, penumbra_band, signed_dist = (
                ShadowBoundaryModule.get_soft_penumbra_gt(
                    shadow_gt,
                    band_width=self.penumbra_band_width,
                    tau=self.penumbra_tau,
                )
            )
            penumbra_up = F.interpolate(
                self._penumbra_logits,
                size=gt_seg.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
            penumbra_prob = torch.sigmoid(penumbra_up)
            band_valid = penumbra_band & valid_bool

            if self.soft_boundary_loss_weight > 0 and band_valid.any():
                soft_loss = F.smooth_l1_loss(
                    penumbra_prob[band_valid],
                    penumbra_gt[band_valid],
                    reduction='mean',
                )
                losses['loss_soft_boundary'] = (
                    soft_loss * self.soft_boundary_loss_weight)

            if self.penumbra_grad_loss_weight > 0:
                dx_p = penumbra_prob[:, :, :, 1:] - penumbra_prob[:, :, :, :-1]
                dy_p = penumbra_prob[:, :, 1:, :] - penumbra_prob[:, :, :-1, :]
                dx_g = penumbra_gt[:, :, :, 1:] - penumbra_gt[:, :, :, :-1]
                dy_g = penumbra_gt[:, :, 1:, :] - penumbra_gt[:, :, :-1, :]
                mask_x = ((penumbra_band[:, :, :, 1:] | penumbra_band[:, :, :, :-1]) &
                          (valid_bool[:, :, :, 1:] & valid_bool[:, :, :, :-1]))
                mask_y = ((penumbra_band[:, :, 1:, :] | penumbra_band[:, :, :-1, :]) &
                          (valid_bool[:, :, 1:, :] & valid_bool[:, :, :-1, :]))
                grad_terms = []
                if mask_x.any():
                    grad_terms.append(F.l1_loss(dx_p[mask_x], dx_g[mask_x]))
                if mask_y.any():
                    grad_terms.append(F.l1_loss(dy_p[mask_y], dy_g[mask_y]))
                if grad_terms:
                    losses['loss_penumbra_grad'] = (
                        sum(grad_terms) / len(grad_terms) *
                        self.penumbra_grad_loss_weight)

            if self.penumbra_mono_loss_weight > 0:
                dx_p = penumbra_prob[:, :, :, 1:] - penumbra_prob[:, :, :, :-1]
                dy_p = penumbra_prob[:, :, 1:, :] - penumbra_prob[:, :, :-1, :]
                dx_s = signed_dist[:, :, :, 1:] - signed_dist[:, :, :, :-1]
                dy_s = signed_dist[:, :, 1:, :] - signed_dist[:, :, :-1, :]
                mask_x = ((penumbra_band[:, :, :, 1:] | penumbra_band[:, :, :, :-1]) &
                          (valid_bool[:, :, :, 1:] & valid_bool[:, :, :, :-1]) &
                          (dx_s.abs() > 1e-6))
                mask_y = ((penumbra_band[:, :, 1:, :] | penumbra_band[:, :, :-1, :]) &
                          (valid_bool[:, :, 1:, :] & valid_bool[:, :, :-1, :]) &
                          (dy_s.abs() > 1e-6))
                mono_terms = []
                if mask_x.any():
                    mono_terms.append(F.relu(-(dx_p * torch.sign(dx_s))[mask_x]).mean())
                if mask_y.any():
                    mono_terms.append(F.relu(-(dy_p * torch.sign(dy_s))[mask_y]).mean())
                if mono_terms:
                    losses['loss_penumbra_mono'] = (
                        sum(mono_terms) / len(mono_terms) *
                        self.penumbra_mono_loss_weight)

            if self.boundary_consistency_loss_weight > 0:
                if boundary_gt is None:
                    boundary_gt = ShadowBoundaryModule.get_boundary_gt(
                        shadow_gt, self.boundary_kernel)
                # p(1-p) peaks at soft transitions; scaling by 4 maps peak to 1.
                penumbra_edge = (4.0 * penumbra_prob * (1.0 - penumbra_prob)).clamp(0, 1)
                if valid_bool.any():
                    consistency_loss = F.smooth_l1_loss(
                        penumbra_edge[valid_bool],
                        boundary_gt[valid_bool],
                        reduction='mean',
                    )
                    losses['loss_boundary_consistency'] = (
                        consistency_loss * self.boundary_consistency_loss_weight)

        return losses
