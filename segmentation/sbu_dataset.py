"""
SBU阴影检测数据集
用于像素级别的阴影检测任务
数据集结构:
- sbu/img: 输入图像
- sbu/label: 二值化标注图 (0=非阴影, 1=阴影)
"""

import os.path as osp
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SBUDataset(BaseSegDataset):
    """SBU Shadow Detection Dataset.

    数据集包含阴影区域的像素级标注
    类别: 0-背景/非阴影, 1-阴影
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
