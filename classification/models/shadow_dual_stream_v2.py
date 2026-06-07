"""
改进的双流VMamba阴影检测模型 - VMamba项目集成版本
==========================================================

基于VMamba官方框架的改进双流实现

集成路径: /home/xjx/CodeProject/PycharmProject/VMamba/classification/models/

修改说明:
1. 继承VMamba的Backbone_VSSM作为全局流
2. 添加改进的局部流
3. 实现Cross-Attention渐进式融合
4. 支持渐进式训练(set_training_stage方法)

作者: AgentLaboratory
日期: 2026-02-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


# ============================================================================
# 多头交叉注意力模块
# ============================================================================

class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力模块

    全局特征作为Query，局部特征作为Key/Value
    让全局流能够"查询"局部细节信息
    """
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        assert dim_q % num_heads == 0, f'dim_q {dim_q} should be divisible by num_heads {num_heads}'

        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_q, dim_q)

        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, dim_q * 4),
            nn.GELU(),
            nn.Linear(dim_q * 4, dim_q)
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_q: Query特征 [B, H, W, C]
            x_kv: Key/Value特征 [B, H, W, C]
        Returns:
            融合后的特征 [B, H, W, C]
        """
        B, H, W, C = x_q.shape

        q = x_q.reshape(B, H * W, C)
        kv = x_kv.reshape(B, H * W, C)

        Q = self.q_proj(q).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ V
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)

        q = q + out
        q = self.norm1(q)
        out = self.ffn(q)
        out = self.norm2(q + out)

        return out.reshape(B, H, W, C)


class CrossAttentionFusion(nn.Module):
    """Cross-Attention融合模块"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bidirectional: bool = True,
        use_shadow_map: bool = False
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_shadow_map = use_shadow_map

        self.g2l_attn = MultiHeadCrossAttention(dim, dim, num_heads)

        if bidirectional:
            self.l2g_attn = MultiHeadCrossAttention(dim, dim, num_heads)

        self.fusion_conv = nn.Conv2d(dim * (2 if bidirectional else 1), dim, 1)

        if use_shadow_map:
            self.shadow_proj = nn.Sequential(
                nn.Conv2d(1, dim // 4, 3, padding=1),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        global_feat: torch.Tensor,
        local_feat: torch.Tensor,
        shadow_map: torch.Tensor = None
    ) -> torch.Tensor:
        B, C, H, W = global_feat.shape

        # 如果特征图太大,使用简单的融合而不是Cross-Attention
        if H * W > 32 * 32:  # 如果空间维度大于32x32
            # 简单的通道注意力融合
            g_mean = global_feat.mean(dim=[2, 3], keepdim=True)
            l_mean = local_feat.mean(dim=[2, 3], keepdim=True)

            g_att = torch.sigmoid(g_mean)
            l_att = torch.sigmoid(l_mean)

            # bidirectional=True 时 fusion_conv 期望 dim*2 通道，需要 cat 而非相加
            if self.bidirectional:
                fused = torch.cat([global_feat * g_att, local_feat * l_att], dim=1)
            else:
                fused = global_feat * g_att + local_feat * l_att
            fused = self.fusion_conv(fused)

            if shadow_map is not None and self.use_shadow_map:
                shadow_gate = self.shadow_proj(shadow_map)
                fused = fused * shadow_gate

            return fused

        # 对于较小的特征图,使用Cross-Attention
        global_feat_bhcw = global_feat.permute(0, 2, 3, 1)
        local_feat_bhcw = local_feat.permute(0, 2, 3, 1)

        g_attended = self.g2l_attn(global_feat_bhcw, local_feat_bhcw)
        g_attended = g_attended.permute(0, 3, 1, 2)

        if self.bidirectional:
            l_attended = self.l2g_attn(local_feat_bhcw, global_feat_bhcw)
            l_attended = l_attended.permute(0, 3, 1, 2)
            fused = torch.cat([g_attended, l_attended], dim=1)
        else:
            fused = g_attended

        fused = self.fusion_conv(fused)

        if shadow_map is not None and self.use_shadow_map:
            shadow_gate = self.shadow_proj(shadow_map)
            fused = fused * shadow_gate

        return fused


# ============================================================================
# 改进的局部流
# ============================================================================

class SobelFilter(nn.Module):
    """Sobel边缘检测"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edge


class LocalResBlock(nn.Module):
    """每3个DW+PW卷积对加一个残差shortcut，解决深层纯卷积梯度爆炸"""
    def __init__(self, dim: int, num_convs: int = 3):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=max(1, dim // 4)),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
            ])
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class LocalStream(nn.Module):
    """
    改进的局部流 - 专注于边缘和纹理细节

    Sobel边缘检测在原始RGB图像上进行，然后下采样到与patch_embed输出
    相同的空间分辨率后，通过投影层融合到特征中。
    每3个卷积块加一个残差连接，防止深层网络梯度爆炸。
    """
    def __init__(
        self,
        in_chans: int = 3,
        dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 9, 2]
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)

        # Sobel边缘检测（在原始RGB上操作）
        self.sobel = SobelFilter()

        # 边缘特征投影：将3通道边缘图下采样并投影到dims[0]通道
        self.edge_proj = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )

        # 融合patch特征和边缘特征
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )

        # 各stage的局部流模块：nn.ModuleList[nn.ModuleList]
        # 外层按stage索引，内层每个元素是下采样层或LocalResBlock
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            stage_modules = nn.ModuleList()
            # 下采样（除第一个stage外）
            if i > 0:
                stage_modules.append(nn.Sequential(
                    nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                ))
            # 每3个卷积对构成一个残差块
            num_blocks = math.ceil(depth / 3)
            for b in range(num_blocks):
                convs_in_block = min(3, depth - b * 3)
                stage_modules.append(LocalResBlock(dim, num_convs=convs_in_block))
            self.stages.append(stage_modules)

        # 阴影候选预测头
        self.shadow_proposals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, 1, 1),
                nn.Sigmoid()
            ) for dim in dims
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: 原始RGB图像 [B, 3, H, W]
        Returns:
            features: 多尺度局部特征
            shadow_maps: 多尺度阴影候选图
        """
        features = []
        shadow_maps = []

        # 在原始RGB图像上提取边缘
        edge = self.sobel(x)  # [B, 3, H, W]
        edge_feat = self.edge_proj(edge)  # [B, dims[0], H/4, W/4]

        # Patch embedding
        x = self.patch_embed(x)  # [B, dims[0], H/4, W/4]

        # 融合patch特征和边缘特征
        x = self.fuse_conv(torch.cat([x, edge_feat], dim=1))  # [B, dims[0], H/4, W/4]

        for stage_modules, shadow_head in zip(self.stages, self.shadow_proposals):
            for module in stage_modules:
                x = module(x)
            features.append(x)
            shadow_maps.append(shadow_head(x))

        return features, shadow_maps


# ============================================================================
# 改进的双流VMamba Backbone
# ============================================================================

class ShadowDualStreamVSSM(nn.Module):
    """
    改进的双流VMamba阴影检测模型

    改进点:
    1. Cross-Attention渐进式融合 - 每个stage都融合
    2. 改进的局部流 - 专门提取边缘/纹理
    3. 支持渐进式训练 - set_training_stage方法

    兼容MMSegmentation框架
    """
    def __init__(
        self,
        depths: List[int] = [2, 2, 27, 2],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        use_bidirectional_attn: bool = True,
        use_shadow_map: bool = True,
        pretrained: str = None,
        frozen_stages: int = -1
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.in_chans = in_chans
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.training_stage = 2  # 默认联合训练

        # 导入VMamba Backbone
        try:
            from vmamba import Backbone_VSSM
            self.has_vmamba = True
        except ImportError:
            self.has_vmamba = False
            print("Warning: VMamba not available, using placeholder")

        # 全局流：VMamba骨干
        if self.has_vmamba:
            self.global_stream = Backbone_VSSM(
                in_chans=in_chans,
                depths=list(depths),
                dims=list(dims),
                drop_path_rate=drop_path_rate,
                out_indices=out_indices
            )
        else:
            # 占位符
            self.global_stream = self._build_placeholder_backbone()

        # 局部流
        self.local_stream = LocalStream(
            in_chans=in_chans,
            dims=dims,
            depths=depths
        )

        # Cross-Attention融合模块（每个stage一个）
        self.fusion_modules = nn.ModuleList([
            CrossAttentionFusion(
                dim=dim,
                num_heads=max(1, dim // 32),
                bidirectional=use_bidirectional_attn,
                use_shadow_map=use_shadow_map
            ) for dim in dims
        ])

        # 辅助头（用于训练）
        self.auxiliary_heads = nn.ModuleList([
            nn.Conv2d(dim, 2, 1) for dim in dims
        ])

        self.global_aux_heads = nn.ModuleList([
            nn.Conv2d(dim, 2, 1) for dim in dims
        ])

        self.local_aux_heads = nn.ModuleList([
            nn.Conv2d(dim, 2, 1) for dim in dims
        ])

        self._init_weights()

        if pretrained:
            self._load_pretrained(pretrained)

    def _build_placeholder_backbone(self):
        """构建占位符backbone（当VMamba不可用时）"""
        class PlaceholderBackbone(nn.Module):
            def __init__(self, dims, depths):
                super().__init__()
                self.dims = dims
                self.depths = depths

                # Patch embedding
                self.patch_embed = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)

                # 各stage，包含下采样和特征提取
                self.stages = nn.ModuleList()
                for i in range(4):
                    stage_layers = []

                    # 下采样（除了第一个stage）
                    if i > 0:
                        stage_layers.append(nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2))
                        stage_layers.append(nn.BatchNorm2d(dims[i]))
                        stage_layers.append(nn.ReLU(inplace=True))

                    # 特征提取块
                    for _ in range(depths[i]):
                        stage_layers.append(nn.Conv2d(dims[i], dims[i], kernel_size=3, padding=1))
                        stage_layers.append(nn.BatchNorm2d(dims[i]))
                        stage_layers.append(nn.ReLU(inplace=True))

                    self.stages.append(nn.Sequential(*stage_layers))

            def forward(self, x):
                x = self.patch_embed(x)
                features = []
                for i, stage in enumerate(self.stages):
                    x = stage(x)
                    features.append(x)
                return features
        return PlaceholderBackbone(self.dims, self.depths)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, path: str):
        """加载预训练权重到全局流"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # 只加载全局流的权重
            if self.has_vmamba:
                global_dict = self.global_stream.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items()
                                 if k in global_dict and v.shape == global_dict[k].shape}
                global_dict.update(pretrained_dict)
                self.global_stream.load_state_dict(global_dict)
                print(f"Loaded {len(pretrained_dict)} pretrained weights to global stream")
        except Exception as e:
            print(f"Failed to load pretrained: {e}")

    def set_training_stage(self, stage: int):
        """
        设置训练阶段（用于渐进式训练）

        Args:
            stage: 0=只训练全局流, 1=只训练局部流, 2=联合训练
        """
        self.training_stage = stage

        for name, param in self.named_parameters():
            param.requires_grad = True

        if stage == 0:
            # 只训练全局流
            for name, param in self.named_parameters():
                if 'local_stream' in name or 'fusion' in name:
                    param.requires_grad = False
        elif stage == 1:
            # 只训练局部流和融合模块
            for name, param in self.named_parameters():
                if 'global_stream' in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            包含多尺度特征的字典
        """
        # 全局流前向
        if self.has_vmamba:
            global_output = self.global_stream(x)
            if isinstance(global_output, (tuple, list)):
                # 展平可能的嵌套结构
                global_features = []
                for item in global_output:
                    if isinstance(item, (tuple, list)):
                        global_features.extend(list(item))
                    else:
                        global_features.append(item)
            elif isinstance(global_output, dict):
                global_features = [global_output[i] for i in self.out_indices]
            else:
                global_features = [global_output]
        else:
            global_features = self.global_stream(x)

        # Stage 0: 跳过局部流和融合，防止随机初始化的局部流噪声污染全局特征
        if getattr(self, 'training_stage', 2) == 0:
            return {
                'features': global_features,
                'global_features': global_features,
                'local_features': [],
                'shadow_maps': []
            }

        # 局部流前向
        local_features, shadow_maps = self.local_stream(x)

        # 渐进式融合
        fused_features = []
        for i, (g_feat, l_feat, fusion) in enumerate(
            zip(global_features, local_features, self.fusion_modules)
        ):
            # 调整尺度
            if g_feat.shape[-2:] != l_feat.shape[-2:]:
                g_feat = F.interpolate(g_feat, size=l_feat.shape[-2:],
                                     mode='bilinear', align_corners=False)

            shadow_map = shadow_maps[i] if i < len(shadow_maps) else None
            fused = fusion(g_feat, l_feat, shadow_map)
            fused_features.append(fused)

        return {
            'features': fused_features,
            'global_features': global_features,
            'local_features': local_features,
            'shadow_maps': shadow_maps
        }


# ============================================================================
# 工厂函数
# ============================================================================

def shadow_dual_stream_tiny(**kwargs):
    kwargs.setdefault('depths', [2, 2, 9, 2])
    kwargs.setdefault('dims', [96, 192, 384, 768])
    kwargs.setdefault('drop_path_rate', 0.2)
    return ShadowDualStreamVSSM(**kwargs)


def shadow_dual_stream_small(**kwargs):
    kwargs.setdefault('depths', [2, 2, 27, 2])
    kwargs.setdefault('dims', [96, 192, 384, 768])
    kwargs.setdefault('drop_path_rate', 0.3)
    return ShadowDualStreamVSSM(**kwargs)


def shadow_dual_stream_base(**kwargs):
    kwargs.setdefault('depths', [2, 2, 27, 2])
    kwargs.setdefault('dims', [128, 256, 512, 1024])
    kwargs.setdefault('drop_path_rate', 0.6)
    return ShadowDualStreamVSSM(**kwargs)


# 导出
__all__ = [
    'ShadowDualStreamVSSM',
    'shadow_dual_stream_tiny',
    'shadow_dual_stream_small',
    'shadow_dual_stream_base',
    'CrossAttentionFusion',
    'LocalStream'
]
