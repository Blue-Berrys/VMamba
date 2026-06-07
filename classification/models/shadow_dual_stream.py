'''
双流VMamba阴影检测模型
- 全局流: VMamba提取全局上下文信息
- 局部流: Conv提取局部细节信息
- 门控融合: 自适应融合全局和局部特征
支持Tiny和Base配置
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# 阴影候选注意力模块
class ShadowCandidateAttention(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.attention_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H > 64 or W > 64:
            x_small = F.adaptive_avg_pool2d(x, (H // 8, W // 8))
            attn_small = self.attention_head(x_small)
            attn = F.interpolate(attn_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            attn = self.attention_head(x)
        return attn

# 局部流骨干网络
class LocalStreamBackbone(nn.Module):
    def __init__(self, in_chans: int = 3, out_channels: int = 64):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, out_channels // 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.shadow_attn = ShadowCandidateAttention(out_channels * 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [out_channels // 2, out_channels, out_channels * 2, out_channels * 4]
        self.out_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_channels, self.out_channels[0], 1), nn.BatchNorm2d(self.out_channels[0])),
            nn.Sequential(nn.Conv2d(out_channels, self.out_channels[1], 3, stride=2, padding=1), nn.BatchNorm2d(self.out_channels[1])),
            nn.Sequential(nn.Conv2d(out_channels, self.out_channels[2], 3, stride=2, padding=1), nn.BatchNorm2d(self.out_channels[2])),
            nn.Sequential(nn.Conv2d(out_channels, self.out_channels[3], 3, stride=2, padding=1), nn.BatchNorm2d(self.out_channels[3])),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Stage 1
        feat_s1 = self.stem[0](x)
        feat_s1 = self.stem[1](feat_s1)
        feat_s1 = self.stem[2](feat_s1)

        # Stage 2
        feat_s2 = self.stem[3](feat_s1)
        feat_s2 = self.stem[4](feat_s2)
        feat_s2 = self.stem[5](feat_s2)

        # Stage 3: with shadow attention
        feat_s3 = self.stem[6](feat_s2)
        feat_s3 = self.stem[7](feat_s3)
        feat_s3 = self.stem[8](feat_s3)

        attn = self.shadow_attn(feat_s3)
        feat_s3 = feat_s3 * attn

        feat = self.fusion(feat_s3)

        # Generate multi-scale outputs
        out1 = self.out_layers[0](feat_s2)
        out2 = self.out_layers[1](feat)
        out3 = self.out_layers[2](feat)
        out4 = self.out_layers[3](feat)

        return [out1, out2, out3, out4]

# 门控融合模块
class GatingFusionModule(nn.Module):
    def __init__(self, global_channels: int, local_channels: int, gate_type: str = 'channel_spatial'):
        super().__init__()
        self.gate_type = gate_type

        if global_channels != local_channels:
            self.align_local = nn.Conv2d(local_channels, global_channels, 1)
        else:
            self.align_local = nn.Identity()

        hidden_dim = max(global_channels // 4, 16)

        if 'channel' in gate_type:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(global_channels, hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, global_channels, 1),
                nn.Sigmoid()
            )

        if 'spatial' in gate_type:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(global_channels, hidden_dim, 7, padding=3),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid()
            )

        self.output_conv = nn.Sequential(
            nn.Conv2d(global_channels, global_channels, 1),
            nn.BatchNorm2d(global_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        local_feat = self.align_local(local_feat)

        if global_feat.shape[2:] != local_feat.shape[2:]:
            local_feat = F.interpolate(local_feat, size=global_feat.shape[2:], mode='bilinear', align_corners=False)

        if self.gate_type == 'channel':
            gate_g = self.channel_gate(global_feat)
            gate_l = 1 - gate_g
        elif self.gate_type == 'spatial':
            gate_g = self.spatial_gate(global_feat)
            gate_l = 1 - gate_g
        else:  # channel_spatial
            gate_g = self.channel_gate(global_feat) * self.spatial_gate(global_feat)
            gate_l = 1 - gate_g

        fused = gate_g * global_feat + gate_l * local_feat
        return self.output_conv(fused)

# 多尺度门控融合
class MultiScaleGatingFusion(nn.Module):
    def __init__(self, global_channels: List[int], local_channels: List[int], gate_type: str = 'channel_spatial'):
        super().__init__()
        self.fusion_modules = nn.ModuleList([
            GatingFusionModule(g, l, gate_type)
            for g, l in zip(global_channels, local_channels)
        ])

    def forward(self, global_feats: List[torch.Tensor], local_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for i, (g_feat, l_feat) in enumerate(zip(global_feats, local_feats)):
            fused.append(self.fusion_modules[i](g_feat, l_feat))
        return fused

# 双流VMamba骨干网络
class DualStreamVMambaBackbone(nn.Module):
    def __init__(self, global_stream, global_channels: List[int], local_channels: int = 64, gate_type: str = 'channel_spatial'):
        super().__init__()
        self.global_stream = global_stream
        self.local_stream = LocalStreamBackbone(in_chans=3, out_channels=local_channels)

        # 根据global_channels动态计算local_out_channels
        local_out_channels = [
            local_channels // 2,
            local_channels,
            local_channels * 2,
            local_channels * 4
        ]

        self.gating_fusion = MultiScaleGatingFusion(
            global_channels=global_channels,
            local_channels=local_out_channels,
            gate_type=gate_type
        )

    def forward(self, x: torch.Tensor):
        global_feats = self.global_stream(x)
        local_feats = self.local_stream(x)
        fused_feats = self.gating_fusion(global_feats, local_feats)
        return fused_feats

# MMSegmentation包装类
try:
    from mmseg.registry import MODELS as MMSEG_MODELS

    @MMSEG_MODELS.register_module()
    class ShadowDualStreamVSSM(nn.Module):
        def __init__(self,
                     depths=[2, 2, 9, 2],
                     dims=[96, 192, 384, 768],
                     drop_path_rate=0.2,
                     local_channels=64,
                     gate_type='channel_spatial',
                     pretrained=None,
                     **kwargs):
            super().__init__()

            # 导入VMamba
            import sys
            import os
            models_dir = os.path.dirname(os.path.abspath(__file__))
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            from vmamba import Backbone_VSSM

            # 创建全局流
            global_backbone = Backbone_VSSM(
                in_chans=3,
                patch_size=4,
                depths=list(depths),
                dims=list(dims),
                drop_path_rate=drop_path_rate,
                out_indices=(0, 1, 2, 3)
            )

            self.backbone = DualStreamVMambaBackbone(
                global_stream=global_backbone,
                global_channels=list(dims),
                local_channels=local_channels,
                gate_type=gate_type
            )

            # MMSeg需要的属性
            self.depths = list(depths)
            self.dims = list(dims)
            self.out_indices = [0, 1, 2, 3]
            self._out_indices = [0, 1, 2, 3]
            self.drop_path_rate = drop_path_rate
            self.local_channels = local_channels
            self.gate_type = gate_type

        def forward(self, x):
            feats = self.backbone(x)
            return tuple(feats)

        def init_weights(self, pretrained=None):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    print('ShadowDualStreamVSSM registered to MMSEG successfully!')

except Exception as e:
    print(f'Warning: MMSEG registration failed: {e}')
    import traceback
    traceback.print_exc()
