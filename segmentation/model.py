"""
VMamba 模型注册模块
该模块用于将 VMamba 模型注册到 MMSegmentation 和 MMDetection 框架中
"""
import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET  # MMDetection 模型注册器
from mmseg.registry import MODELS as MODELS_MMSEG  # MMSegmentation 模型注册器

def import_abspy(name="models", path="classification/"):
    """
    动态导入指定路径下的Python模块

    Args:
        name: 模块名称，默认为"models"
        path: 模块所在路径，默认为"classification/"

    Returns:
        导入的模块对象
    """
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)  # 临时将路径添加到系统路径
    module = importlib.import_module(name)  # 导入模块
    sys.path.pop(0)  # 移除临时路径
    return module

# 导入分类任务中定义的 VMamba backbone
build = import_abspy(
    "models",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM  # 获取 VMamba 骨干网络

@MODELS_MMSEG.register_module()  # 注册到 MMSegmentation
@MODELS_MMDET.register_module()  # 注册到 MMDetection
class MM_VSSM(BaseModule, Backbone_VSSM):
    """
    VMamba 模型的 MMSegmentation/MMDetection 适配器

    该类继承自 MMEngine 的 BaseModule 和 VMamba 的 Backbone_VSSM，
    使得 VMamba 模型可以在 MMSegmentation 和 MMDetection 框架中使用。
    """
    def __init__(self, *args, **kwargs):
        """
        初始化 MM_VSSM 模型

        Args:
            *args: 传递给 Backbone_VSSM 的位置参数
            **kwargs: 传递给 Backbone_VSSM 的关键字参数
        """
        BaseModule.__init__(self)  # 初始化 MMEngine 基类
        Backbone_VSSM.__init__(self, *args, **kwargs)  # 初始化 VMamba 骨干网络

