"""
SBU阴影检测数据集
用于像素级别的阴影检测任务
数据集结构:
- sbu/img: 输入图像
- sbu/label: 二值化标注图 (0=非阴影, 1或255=阴影)
"""

import os.path as osp
import numpy as np
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SBUDataset(BaseSegDataset):
    """SBU Shadow Detection Dataset.

    数据集包含阴影区域的像素级标注
    类别: 0-背景/非阴影, 1-阴影

    注意: 标注图像可能是 0-255 的灰度值，其中 255 表示阴影。
    本类会自动将 255 转换为 1，确保标签值在 [0, 1] 范围内。
    """

    # 类别元信息
    METAINFO = dict(
        classes=('non-shadow', 'shadow'),  # 类别名称
        palette=[[255, 255, 255], [0, 0, 0]]  # 可视化颜色: 白色=非阴影, 黑色=阴影
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        """初始化SBU数据集

        Args:
            img_suffix: 图像文件后缀，默认.jpg
            seg_map_suffix: 标注文件后缀，默认.png
            reduce_zero_label: 是否将标签0视为ignore，默认False（保留0作为非阴影类）
        """
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def _load_seg_map(self, seg_path):
        """加载并转换分割标注图

        将标注图中的值转换为 0 或 1：
        - 0 保持为 0（非阴影）
        - 非 0 的值（如 255）转换为 1（阴影）

        Args:
            seg_path: 标注文件路径

        Returns:
            numpy.ndarray: 转换后的标注图，值为 0 或 1
        """
        # 调用父类方法加载标注图
        seg_map = super()._load_seg_map(seg_path)

        # 将非 0 的值转换为 1（处理 255 或其他非 0 值）
        # 确保标签值在 [0, 1] 范围内
        seg_map = (seg_map >= 125).astype(np.uint8)  # SDDNet threshold 125

        return seg_map
