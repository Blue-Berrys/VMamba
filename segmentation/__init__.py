"""
自定义组件注册模块
注册SBU数据集和BER评估指标到MMSegmentation框架
"""

# 导入并注册SBU数据集
from .sbu_dataset import SBUDataset

# 导入并注册BER评估指标
from .ber_metric import BERMetric

__all__ = ['SBUDataset', 'BERMetric']
