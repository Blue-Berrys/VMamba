"""
自定义组件注册模块
注册SBU数据集、BER评估指标、IC-SSM Head、ISTD数据集到MMSegmentation框架
"""

# 导入并注册SBU数据集
from .sbu_dataset import SBUDataset

# 导入并注册BER评估指标
from .ber_metric import BERMetric

# 导入并注册 IC-SSM Shadow Detection Head (创新模块)
from .ic_ssm_head import ICShadowHead

# 导入并注册 ISTD 数据集 (多数据集训练用)
from .istd_dataset import ISTDDataset, ISTDLabelTransform
from .transforms.sbu_label_transform import SBULabelTransform, RefineAnnTransform

__all__ = [
    'SBUDataset',
    'BERMetric',
    'ICShadowHead',
    'ISTDDataset',
    'ISTDLabelTransform',
    'RefineAnnTransform',
]
