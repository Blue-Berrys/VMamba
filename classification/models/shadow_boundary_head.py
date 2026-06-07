"""
ShadowBoundaryHead: 阴影边界辅助预测头

功能：在训练时对阴影边界区域施加额外监督，强迫 decoder 精确感知边界，
     从而降低 FNR（漏检率），改善 FNR >> FPR 的类别不平衡问题。

边界GT在线生成：dilate(mask) - erode(mask) → 5px宽边界带
损失函数：加权BCE（pos_weight=10，边界像素约占5-10%）

注册为 mmseg 模型，可直接在 config 的 auxiliary_head 列表中使用：
    dict(type='ShadowBoundaryHead', in_channels=256, in_index=1, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.registry import MODELS
from mmseg.models.utils.wrappers import resize


@MODELS.register_module()
class ShadowBoundaryHead(FCNHead):
    """阴影检测边界辅助监督头。

    继承 FCNHead，重写 loss_by_feat 以在线生成边界GT并计算加权BCE损失。
    推理时不使用该头（仅参与训练loss）。

    Args:
        boundary_width (int): 边界带宽度（像素）。默认5。
        bce_pos_weight (float): BCE正样本权重，补偿边界像素稀疏问题。默认10.0。
        boundary_loss_weight (float): 整体 loss 权重。默认0.3。
        **kwargs: 传给 FCNHead 的其他参数（in_channels, in_index, channels等）。
    """

    def __init__(self,
                 boundary_width: int = 5,
                 bce_pos_weight: float = 10.0,
                 boundary_loss_weight: float = 0.3,
                 **kwargs):
        # 强制二分类（1通道）
        kwargs['num_classes'] = 1
        kwargs.setdefault('num_convs', 2)
        kwargs.setdefault('concat_input', False)
        super().__init__(**kwargs)

        self.boundary_width = boundary_width
        self.bce_pos_weight = bce_pos_weight
        self.boundary_loss_weight = boundary_loss_weight

        # 覆盖 FCNHead 的 conv_seg（channels → 1，二值边界输出）
        self.conv_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

    # ------------------------------------------------------------------
    # 边界GT生成（形态学梯度）
    # ------------------------------------------------------------------

    def _get_boundary_gt(self, gt_seg_map: torch.Tensor) -> torch.Tensor:
        """从语义分割GT在线生成边界GT。

        使用形态学梯度：boundary = dilate(shadow) - erode(shadow)

        Args:
            gt_seg_map: [B, 1, H, W] 分割mask（0=非阴影, 1=阴影, 255=忽略）

        Returns:
            boundary: [B, H, W] float32（1.0=边界像素, 0.0=非边界）
        """
        # 只取阴影区域（忽略255）
        shadow_mask = (gt_seg_map == 1).float()  # [B, 1, H, W]

        k = self.boundary_width
        p = k // 2

        # 膨胀：最大池化
        dilated = F.max_pool2d(shadow_mask, kernel_size=k, stride=1, padding=p)
        # 腐蚀：最小池化 = -max_pool(-x)
        eroded = -F.max_pool2d(-shadow_mask, kernel_size=k, stride=1, padding=p)

        # 边界 = 膨胀 - 腐蚀 > 0
        boundary = (dilated - eroded).squeeze(1)  # [B, H, W]
        return (boundary > 0.5).float()

    # ------------------------------------------------------------------
    # Loss 计算（重写 FCNHead 的 loss_by_feat）
    # ------------------------------------------------------------------

    def loss_by_feat(self, seg_logits: torch.Tensor,
                     batch_data_samples) -> dict:
        """计算边界监督损失。

        1. 从 batch_data_samples 提取 GT 分割 mask
        2. 在线生成边界 GT
        3. 计算加权BCE

        Args:
            seg_logits: [B, 1, H', W'] 边界预测 logits
            batch_data_samples: MMSeg SegDataSample 列表

        Returns:
            dict: {'loss_boundary': tensor, 'acc_boundary': tensor}
        """
        # 获取GT mask：[B, 1, H, W]
        gt_seg = self._stack_batch_gt(batch_data_samples)

        # 生成边界GT：[B, H, W] float
        boundary_gt = self._get_boundary_gt(gt_seg)  # [B, H, W]

        # 将logits上采样到GT分辨率
        seg_logits = resize(
            seg_logits,
            size=boundary_gt.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        seg_logits = seg_logits.squeeze(1)  # [B, H, W]

        # 加权BCE损失（边界像素稀少，需要高权重）
        pos_weight = torch.tensor(
            [self.bce_pos_weight], device=seg_logits.device, dtype=seg_logits.dtype
        )
        loss = F.binary_cross_entropy_with_logits(
            seg_logits,
            boundary_gt,
            pos_weight=pos_weight,
            reduction='mean'
        )

        losses = dict()
        losses['loss_boundary'] = loss * self.boundary_loss_weight

        # 边界预测精度（监控用，不参与梯度）
        with torch.no_grad():
            pred = (seg_logits > 0).float()
            acc = (pred == boundary_gt).float().mean() * 100.0
            losses['acc_boundary'] = acc

        return losses

    def predict_by_feat(self, seg_logits: torch.Tensor,
                        batch_img_metas: list) -> torch.Tensor:
        """推理时不使用，返回 sigmoid 概率（仅为接口兼容）。"""
        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return torch.sigmoid(seg_logits)
