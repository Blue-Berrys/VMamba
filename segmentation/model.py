"""VMamba 模型注册模块"""
import os
import sys
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG
from mmengine.model import BaseModule

def import_abspy(name="models", path="classification/"):
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

# Import VMamba backbone
build = import_abspy(
    "models",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
Backbone_VSSM = build.vmamba.Backbone_VSSM

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_VSSM(BaseModule, Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)

# Import improved dual stream model
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/models")
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from shadow_dual_stream_v2 import ShadowDualStreamVSSM

    @MODELS_MMSEG.register_module()
    @MODELS_MMDET.register_module()
    class MM_ShadowDualStream(BaseModule, ShadowDualStreamVSSM):
        """改进的双流VMamba - MMSeg注册版本"""
        def __init__(self, init_cfg=None, **kwargs):
            pretrained = None
            if init_cfg is not None and isinstance(init_cfg, dict):
                pretrained = init_cfg.get('checkpoint', None)
            BaseModule.__init__(self)
            ShadowDualStreamVSSM.__init__(self, pretrained=pretrained, **kwargs)

        def init_weights(self):
            if hasattr(super(), 'init_weights'):
                super().init_weights()

        def forward(self, x):
            outputs = ShadowDualStreamVSSM.forward(self, x)
            return tuple(outputs['features'])

    print("Successfully registered MM_ShadowDualStream")

except Exception as e:
    print(f"Failed to register MM_ShadowDualStream: {e}")
    import traceback
    traceback.print_exc()

# Import progressive training hook to register it
try:
    from core.hooks.progressive_training_hook import (
        ProgressiveTrainingHook, AuxiliaryLossHook, TrainOnlyPenumbraHook)
    print("Successfully registered ProgressiveTrainingHook")
except Exception as e:
    print(f"Failed to register ProgressiveTrainingHook: {e}")


# Import ShadowBoundaryHead and register to mmseg
try:
    from shadow_boundary_head import ShadowBoundaryHead
    MODELS_MMSEG.register_module()(ShadowBoundaryHead)
    print("Successfully registered ShadowBoundaryHead")
except Exception as e:
    print(f"Failed to register ShadowBoundaryHead: {e}")
    import traceback
    traceback.print_exc()

# Import ShadowDualPathHead and register
try:
    from shadow_dual_path_head import ShadowDualPathHead
    print("Successfully registered ShadowDualPathHead")
except Exception as e:
    print(f"Failed to register ShadowDualPathHead: {e}")
    import traceback; traceback.print_exc()
