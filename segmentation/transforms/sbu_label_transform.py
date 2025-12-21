"""
SBU 标签转换 Transform
将标签值从 0-255 转换为 0-1（二分类）
"""

import numpy as np
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms import LoadAnnotations


@TRANSFORMS.register_module()
class SBULabelTransform(LoadAnnotations):
    """SBU 标签转换 Transform
    
    继承 LoadAnnotations，在加载后自动将标签值转换为 0 或 1：
    - 0 保持为 0（非阴影）
    - 非 0 的值（如 255）转换为 1（阴影）
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def transform(self, results):
        """转换标签值"""
        # 调用父类方法加载标注
        results = super().transform(results)
        
        # 获取标注数据
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            
            # 将非 0 的值转换为 1（处理 255 或其他非 0 值）
            # 确保标签值在 [0, 1] 范围内
            if isinstance(seg_map, np.ndarray):
                seg_map = (seg_map > 0).astype(np.uint8)
                results['gt_seg_map'] = seg_map
        
        return results

