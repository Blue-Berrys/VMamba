"""
双流VMamba模块 - 用于阴影检测任务

实现双流架构：
- 流1：主干流（原始RGB图像）- 负责语义信息
- 流2：对比感知流（Mean Subtraction图像）- 负责纹理信息
- 融合：门控调制机制
"""

import os
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

# 导入基础模块
import sys
import os

# 确保能导入vmamba模块
try:
    from vmamba import SS2D, VSSBlock, VSSM, Backbone_VSSM, Linear2d, LayerNorm2d, PatchMerging2D, Permute
except ImportError:
    # 如果直接导入失败，尝试从同级目录导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from vmamba import SS2D, VSSBlock, VSSM, Backbone_VSSM, Linear2d, LayerNorm2d, PatchMerging2D, Permute


# =====================================================
# 双流VSS Block - 门控融合版本
# =====================================================

class DualStreamVSSBlock(nn.Module):
    """
    双流VSS Block with Gating Mechanism

    核心思想：
    1. 流1（主干流）：使用标准SS2D处理原始RGB特征，捕获语义信息
    2. 流2（对比流）：使用轻量SS2D处理Mean Subtraction特征，捕获纹理信息
    3. 门控融合：流2生成门控权重，调制流1的特征响应

    Args:
        hidden_dim_s1: 流1的隐藏维度
        hidden_dim_s2: 流2的隐藏维度（可以小于流1，以降低计算量）
        drop_path: DropPath概率
        norm_layer: 归一化层类型
        channel_first: 是否channel-first格式
        ssm_d_state: SSM状态维度
        ssm_ratio_s1: 流1的SSM扩展比例
        ssm_ratio_s2: 流2的SSM扩展比例（可以较小）
        gate_type: 门控类型 ('channel', 'spatial', 'channel_spatial')
        gate_ratio: 门控网络压缩比例
        ...其他参数同VSSBlock
    """

    def __init__(
        self,
        hidden_dim_s1: int = 128,
        hidden_dim_s2: int = 64,  # 流2可以更小
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # ==================== 流1参数 ====================
        ssm_d_state_s1: int = 16,
        ssm_ratio_s1: float = 2.0,
        ssm_dt_rank_s1: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # ==================== 流2参数（轻量化）====================
        ssm_d_state_s2: int = 8,
        ssm_ratio_s2: float = 1.5,
        ssm_dt_rank_s2: Any = "auto",
        # ==================== 门控参数 ====================
        gate_type: str = "channel_spatial",  # 'channel', 'spatial', 'channel_spatial'
        gate_ratio: float = 0.25,  # 门控网络压缩比例
        # ==================== MLP参数 ====================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # ==================== 其他 ====================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = True  # 双流始终使用SSM
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.gate_type = gate_type

        # 流1：主干流（语义特征）
        self.norm_s1 = norm_layer(hidden_dim_s1)
        self.op_s1 = SS2D(
            d_model=hidden_dim_s1,
            d_state=ssm_d_state_s1,
            ssm_ratio=ssm_ratio_s1,
            dt_rank=ssm_dt_rank_s1,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )

        # 流2：对比流（纹理特征）
        self.norm_s2 = norm_layer(hidden_dim_s2)
        self.op_s2 = SS2D(
            d_model=hidden_dim_s2,
            d_state=ssm_d_state_s2,
            ssm_ratio=ssm_ratio_s2,
            dt_rank=ssm_dt_rank_s2,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )

        # 特征对齐：将流2的特征对齐到流1的维度
        if hidden_dim_s2 != hidden_dim_s1:
            self.align_s2_to_s1 = nn.Linear(hidden_dim_s2, hidden_dim_s1) if not channel_first else \
                                  Linear2d(hidden_dim_s2, hidden_dim_s1)
        else:
            self.align_s2_to_s1 = nn.Identity()

        # 门控网络：从流2特征生成门控权重
        gate_dim = int(hidden_dim_s1 * gate_ratio)

        if gate_type in ['channel', 'channel_spatial']:
            # Channel门控：学习通道级别的权重
            self.gate_channel = nn.Sequential(
                nn.Linear(hidden_dim_s1, gate_dim) if not channel_first else Linear2d(hidden_dim_s1, gate_dim),
                nn.SiLU(),
                nn.Linear(gate_dim, hidden_dim_s1) if not channel_first else Linear2d(gate_dim, hidden_dim_s1),
                nn.Sigmoid()
            )

        if gate_type in ['spatial', 'channel_spatial']:
            # Spatial门控：学习空间位置的权重
            self.gate_spatial = nn.Sequential(
                nn.Conv2d(hidden_dim_s1, gate_dim, 1) if channel_first else nn.Linear(hidden_dim_s1, gate_dim),
                nn.SiLU(),
                nn.Conv2d(gate_dim, 1, 1) if channel_first else nn.Linear(gate_dim, 1),
                nn.Sigmoid()
            )

        self.drop_path = DropPath(drop_path)

        # MLP分支（仅应用于流1）
        if self.mlp_branch:
            self.norm_mlp = norm_layer(hidden_dim_s1)
            mlp_hidden_dim = int(hidden_dim_s1 * mlp_ratio)
            Linear = Linear2d if channel_first else nn.Linear
            self.mlp = nn.Sequential(
                Linear(hidden_dim_s1, mlp_hidden_dim),
                mlp_act_layer(),
                nn.Dropout(mlp_drop_rate),
                Linear(mlp_hidden_dim, hidden_dim_s1),
                nn.Dropout(mlp_drop_rate)
            )

    def _apply_gate(self, feat_s1, feat_s2_aligned):
        """
        应用门控机制

        Args:
            feat_s1: 流1特征 [B, H, W, C1] 或 [B, C1, H, W]
            feat_s2_aligned: 对齐后的流2特征 [B, H, W, C1] 或 [B, C1, H, W]

        Returns:
            gated_feat: 门控调制后的特征
        """
        channel_first = (len(feat_s1.shape) == 4 and feat_s1.shape[1] == feat_s1.shape[2] * feat_s1.shape[3] // 100) or \
                       (hasattr(self, 'op_s1') and self.op_s1.channel_first)

        # 融合两个流的特征
        fused_feat = feat_s1 + feat_s2_aligned

        if self.gate_type == 'channel':
            # Channel-wise门控
            if channel_first:
                # [B, C, H, W] -> [B, H, W, C] -> gate -> [B, C, H, W]
                gate = fused_feat.permute(0, 2, 3, 1)
                gate = self.gate_channel(gate)
                gate = gate.permute(0, 3, 1, 2)
            else:
                gate = self.gate_channel(fused_feat)
            gated_feat = feat_s1 * gate

        elif self.gate_type == 'spatial':
            # Spatial-wise门控
            if channel_first:
                gate = self.gate_spatial(fused_feat)  # [B, 1, H, W]
            else:
                # [B, H, W, C] -> [B, C, H, W] -> gate -> [B, 1, H, W] -> [B, H, W, 1]
                x_permuted = fused_feat.permute(0, 3, 1, 2)
                gate = self.gate_spatial(x_permuted)
                gate = gate.permute(0, 2, 3, 1)
            gated_feat = feat_s1 * gate

        elif self.gate_type == 'channel_spatial':
            # Channel + Spatial联合门控
            if channel_first:
                # Channel gate
                gate_c = fused_feat.permute(0, 2, 3, 1)
                gate_c = self.gate_channel(gate_c)
                gate_c = gate_c.permute(0, 3, 1, 2)
                # Spatial gate
                gate_s = self.gate_spatial(fused_feat)
                # 联合
                gate = gate_c * gate_s
            else:
                # Channel gate
                gate_c = self.gate_channel(fused_feat)
                # Spatial gate
                x_permuted = fused_feat.permute(0, 3, 1, 2)
                gate_s = self.gate_spatial(x_permuted)
                gate_s = gate_s.permute(0, 2, 3, 1)
                # 联合
                gate = gate_c * gate_s
            gated_feat = feat_s1 * gate

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        return gated_feat

    def _forward(self, x_s1, x_s2):
        """
        前向传播

        Args:
            x_s1: 流1输入特征 [B, H, W, C1] 或 [B, C1, H, W]
            x_s2: 流2输入特征 [B, H, W, C2] 或 [B, C2, H, W]

        Returns:
            out_s1: 流1输出特征（融合了流2信息）
            out_s2: 流2输出特征（仅用于后续融合）
        """
        # 保存输入用于残差
        residual_s1 = x_s1
        residual_s2 = x_s2

        # SSM分支
        if self.ssm_branch:
            if not self.post_norm:
                x_s1 = self.norm_s1(x_s1)
                x_s2 = self.norm_s2(x_s2)

            # 流1和流2分别进行选择性扫描
            feat_s1 = self.op_s1(x_s1)
            feat_s2 = self.op_s2(x_s2)

            # 对齐流2特征到流1维度
            feat_s2_aligned = self.align_s2_to_s1(feat_s2)

            # 门控融合：用流2调制流1
            gated_feat_s1 = self._apply_gate(feat_s1, feat_s2_aligned)

            if self.post_norm:
                gated_feat_s1 = self.norm_s1(gated_feat_s1)
                feat_s2 = self.norm_s2(feat_s2)

            # 残差连接（仅对流1）
            x_s1 = residual_s1 + self.drop_path(gated_feat_s1)
            x_s2 = residual_s2 + self.drop_path(feat_s2)  # 流2也保留残差

        # MLP分支（仅应用于流1）
        if self.mlp_branch:
            if not self.post_norm:
                x_s1 = x_s1 + self.drop_path(self.mlp(self.norm_mlp(x_s1)))
            else:
                x_s1 = x_s1 + self.drop_path(self.norm_mlp(self.mlp(x_s1)))

        return x_s1, x_s2

    def forward(self, x_s1, x_s2):
        """
        前向传播包装器（支持checkpoint）
        """
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_s1, x_s2)
        else:
            return self._forward(x_s1, x_s2)


# =====================================================
# 双流VSSM Backbone
# =====================================================

class DualStreamVSSM(nn.Module):
    """
    双流VMamba Backbone for Shadow Detection

    架构：
    - 流1：处理原始RGB图像（Base配置）
    - 流2：处理Mean Subtraction图像（轻量化配置）
    - 融合：在每个VSS Block内部进行门控融合

    Args:
        depths: 各阶段的block数量 [2, 2, 27, 2] for Base
        dims_s1: 流1各阶段的通道数 [128, 256, 512, 1024] for Base
        dims_s2: 流2各阶段的通道数 [64, 128, 256, 512] (轻量化)
        ...其他参数
    """

    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        # ==================== Stage配置 ====================
        depths=[2, 2, 27, 2],  # Base配置
        dims_s1=None,  # 流1：[128, 256, 512, 1024] for Base
        dims_s2=None,  # 流2：[64, 128, 256, 512] (轻量化)
        # ==================== 流1参数（Base配置）====================
        ssm_d_state_s1=16,
        ssm_ratio_s1=2.0,
        ssm_dt_rank_s1="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        # ==================== 流2参数（轻量化）====================
        ssm_d_state_s2=8,
        ssm_ratio_s2=1.5,
        ssm_dt_rank_s2="auto",
        # ==================== 门控参数 ====================
        gate_type="channel_spatial",  # 'channel', 'spatial', 'channel_spatial'
        gate_ratio=0.25,
        # ==================== MLP参数 ====================
        mlp_ratio=0.0,  # VMamba默认不使用MLP
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # ==================== 其他 ====================
        drop_path_rate=0.6,  # Base配置
        patch_norm=True,
        norm_layer="ln",  # LayerNorm
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)

        # 设置默认维度
        if dims_s1 is None:
            dims_s1 = [128, 256, 512, 1024]  # Base配置
        if dims_s2 is None:
            dims_s2 = [64, 128, 256, 512]  # 轻量化（约为流1的一半）

        self.dims_s1 = dims_s1
        self.dims_s2 = dims_s2
        self.num_features_s1 = dims_s1[-1]

        # Drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 归一化层
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        norm_layer = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        # Positional embedding
        self.pos_embed_s1 = self._pos_embed(dims_s1[0], patch_size, imgsize) if posembed else None
        self.pos_embed_s2 = self._pos_embed(dims_s2[0], patch_size, imgsize) if posembed else None

        # Patch Embedding (双流)
        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)

        self.patch_embed_s1 = _make_patch_embed(in_chans, dims_s1[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)
        self.patch_embed_s2 = _make_patch_embed(in_chans, dims_s2[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        # 下采样模块
        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        # 构建各阶段
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 流1的下采样
            downsample_s1 = _make_downsample(
                self.dims_s1[i_layer],
                self.dims_s1[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            # 流2的下采样
            downsample_s2 = _make_downsample(
                self.dims_s2[i_layer],
                self.dims_s2[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            # 构建双流Layer
            layer = DualStreamVSSLayer(
                dim_s1=self.dims_s1[i_layer],
                dim_s2=self.dims_s2[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample_s1=downsample_s1,
                downsample_s2=downsample_s2,
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                channel_first=self.channel_first,
                # 流1参数
                ssm_d_state_s1=ssm_d_state_s1,
                ssm_ratio_s1=ssm_ratio_s1,
                ssm_dt_rank_s1=ssm_dt_rank_s1,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # 流2参数
                ssm_d_state_s2=ssm_d_state_s2,
                ssm_ratio_s2=ssm_ratio_s2,
                ssm_dt_rank_s2=ssm_dt_rank_s2,
                # 门控参数
                gate_type=gate_type,
                gate_ratio=gate_ratio,
                # MLP参数
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            self.layers.append(layer)

        # 分类器（用于分类任务，分割时不需要）
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features_s1),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features_s1, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _make_patch_embed(self, in_chans, embed_dim, patch_size, patch_norm, norm_layer, channel_first=False):
        """Patch embedding v1"""
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def _make_patch_embed_v2(self, in_chans, embed_dim, patch_size, patch_norm, norm_layer, channel_first=False):
        """Patch embedding v2 (with 4x4 stem)"""
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2) if channel_first else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim) if channel_first else nn.Identity(),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def _make_downsample(self, dim_in, dim_out, norm_layer, channel_first=False):
        """下采样 v2"""
        Linear = Linear2d if channel_first else nn.Linear
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2, bias=False),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(dim_out) if not channel_first else nn.Identity()),
        )

    def _make_downsample_v3(self, dim_in, dim_out, norm_layer, channel_first=False):
        """下采样 v3 (更高效)"""
        Linear = Linear2d if channel_first else nn.Linear
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.BatchNorm2d(dim_in) if channel_first else nn.Identity(),
            nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2, bias=False),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(dim_out) if not channel_first else nn.Identity()),
        )

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed_s1", "pos_embed_s2"}

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            outs: 各阶段输出特征列表 [(B, C1, H1, W1), ...]
        """
        B, C, H, W = x.shape

        # Patch embedding
        x_s1 = self.patch_embed_s1(x)  # 流1
        x_s2 = self.patch_embed_s2(x)  # 流2

        # Positional embedding
        if self.pos_embed_s1 is not None:
            x_s1 = x_s1 + self.pos_embed_s1
        if self.pos_embed_s2 is not None:
            x_s2 = x_s2 + self.pos_embed_s2

        # 通过各阶段
        outs = []
        for i, layer in enumerate(self.layers):
            x_s1, x_s2 = layer(x_s1, x_s2)

            # 保存流1的输出（融合了流2信息）
            if not self.channel_first:
                out = x_s1.permute(0, 3, 1, 2).contiguous()
            else:
                out = x_s1.contiguous()
            outs.append(out)

        return outs


class DualStreamVSSLayer(nn.Module):
    """双流VSS Layer（包含多个Block和下采样）"""

    def __init__(
        self,
        dim_s1,
        dim_s2,
        depth,
        drop_path,
        downsample_s1,
        downsample_s2,
        use_checkpoint,
        norm_layer,
        channel_first,
        **block_kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DualStreamVSSBlock(
                hidden_dim_s1=dim_s1,
                hidden_dim_s2=dim_s2,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                channel_first=channel_first,
                use_checkpoint=use_checkpoint,
                **block_kwargs
            )
            for i in range(depth)
        ])
        self.downsample_s1 = downsample_s1
        self.downsample_s2 = downsample_s2

    def forward(self, x_s1, x_s2):
        """通过所有blocks并进行下采样"""
        for block in self.blocks:
            x_s1, x_s2 = block(x_s1, x_s2)

        # 下采样
        x_s1_down = self.downsample_s1(x_s1)
        x_s2_down = self.downsample_s2(x_s2)

        return x_s1_down, x_s2_down


# =====================================================
# MMSegmentation兼容的Backbone
# =====================================================

class Backbone_DualStreamVSSM(DualStreamVSSM):
    """
    双流VMamba Backbone - MMSegmentation版本

    与单流版本兼容，可以直接在分割任务中使用
    """

    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims_s1[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # 删除分类器（分割任务不需要）
        del self.classifier

        # 加载预训练权重
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        """加载预训练权重（仅加载流1）"""
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")

            # 仅加载流1的权重（流2从头训练）
            state_dict = _ckpt[key]
            incompatibleKeys = self.load_state_dict(state_dict, strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint from {ckpt}: {e}")

    def forward(self, x):
        """
        前向传播 - MMSegmentation版本

        在内部生成Mean Subtraction图像，保持接口兼容性

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            outs: list of tensors, each shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 流1：原始RGB图像
        x_s1 = x

        # 流2：Mean Subtraction图像（在backbone内部生成）
        x_ms = self._compute_mean_subtraction(x)
        x_s2 = x_ms

        # Patch embedding
        x_s1 = self.patch_embed_s1(x_s1)
        x_s2 = self.patch_embed_s2(x_s2)

        # Positional embedding
        if self.pos_embed_s1 is not None:
            x_s1 = x_s1 + self.pos_embed_s1
        if self.pos_embed_s2 is not None:
            x_s2 = x_s2 + self.pos_embed_s2

        # 通过各阶段
        outs = []
        for i, layer in enumerate(self.layers):
            x_s1, x_s2 = layer(x_s1, x_s2)

            # 保存输出
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x_s1)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        return outs

    def _compute_mean_subtraction(self, x, kernel_size=15):
        """
        计算Mean Subtraction图像

        Args:
            x: 输入图像 [B, 3, H, W]，范围[0, 1]或[0, 255]
            kernel_size: 高斯滤波核大小

        Returns:
            x_ms: Mean Subtraction图像 [B, 3, H, W]
        """
        import math

        # 确保输入在[0, 1]范围
        if x.max() > 1.0:
            x_norm = x / 255.0
        else:
            x_norm = x

        # 使用PyTorch原生的高斯模糊
        # 创建1D高斯核
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        gauss = torch.Tensor([
            math.exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(kernel_size)
        ])
        gauss = gauss / gauss.sum()
        gauss = gauss.to(x.device, dtype=x.dtype)

        # 创建2D高斯核（可分离卷积）
        # 对每个通道分别处理（保持独立性）
        padding = kernel_size // 2
        local_mean = x_norm.clone()

        # 对每个通道应用高斯模糊
        for c in range(x_norm.shape[1]):
            # 水平方向
            gauss_h = gauss.view(1, 1, 1, kernel_size)
            channel = x_norm[:, c:c+1, :, :]
            channel_h = F.conv2d(channel, gauss_h, padding=(0, padding))
            # 垂直方向
            gauss_v = gauss.view(1, 1, kernel_size, 1)
            channel_hv = F.conv2d(channel_h, gauss_v, padding=(padding, 0))
            local_mean[:, c:c+1, :, :] = channel_hv

        # 减去均值
        x_ms = x_norm - local_mean

        # 归一化到[0, 1]
        x_ms_min = x_ms.amin(dim=(2, 3), keepdim=True)
        x_ms_max = x_ms.amax(dim=(2, 3), keepdim=True)
        x_ms = (x_ms - x_ms_min) / (x_ms_max - x_ms_min + 1e-8)

        # 恢复到原始范围
        if x.max() > 1.0:
            x_ms = x_ms * 255.0

        return x_ms


# 导出
__all__ = [
    'DualStreamVSSBlock',
    'DualStreamVSSM',
    'Backbone_DualStreamVSSM',
]
