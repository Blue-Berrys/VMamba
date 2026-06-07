"""
ISTD 数据集加载器 (用于多数据集预训练)
=====================================

ISTD 目录结构 (标准格式):
    ISTD_Dataset/
        train/
            img/     ← 原始图像 (1330 张)
            mask/    ← 阴影 mask (白=阴影, 黑=非阴影)
        test/
            img/     (540 张)
            mask/

注意:
    - ISTD mask: 白色(255) = 阴影, 黑色(0) = 非阴影
    - SBU  mask: 白色(255) = 阴影, 黑色(0) = 非阴影 (格式相同, 已归一化为 0/1)
    - 两者格式统一后可混合训练

MMSeg 集成:
    在 __init__.py 中注册, 配置文件中 dataset_type='ISTDDataset' 使用
"""

import os
import numpy as np
from PIL import Image

from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets import BaseSegDataset
import mmengine.fileio as fileio


@TRANSFORMS.register_module()
class ISTDLabelTransform:
    """将 ISTD mask 转为 mmseg 格式 (0=非阴影, 1=阴影)。

    ISTD mask: PNG, 灰度图或 RGB 图。
    - 像素值 > 127 → 阴影 → 标签 1
    - 像素值 ≤ 127 → 非阴影 → 标签 0
    """

    def __call__(self, results: dict) -> dict:
        if 'gt_seg_map' in results:
            seg = results['gt_seg_map']
            if isinstance(seg, np.ndarray):
                # 转换: >127 为阴影=1, ≤127 为非阴影=0
                results['gt_seg_map'] = (seg > 127).astype(np.uint8)
        return results


@DATASETS.register_module()
class ISTDDataset(BaseSegDataset):
    """ISTD 阴影检测数据集。

    METAINFO 与 SBUDataset 一致, 便于混合训练。

    Args:
        data_root (str): 数据集根目录
        split (str): 'train' 或 'test'
        **kwargs: 传给 BaseSegDataset
    """

    # 必须与 SBUDataset.METAINFO 完全一致，ConcatDataset 才能混合
    METAINFO = dict(
        classes=('non-shadow', 'shadow'),
        palette=[[255, 255, 255], [0, 0, 0]],
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs,
        )
