"""
SBU label transforms.

SBU uses binary 0/255 masks. SBU-Refine additionally provides 0-255 soft
masks, where intermediate values carry boundary confidence. For penumbra
training, keep the binary target for the main detector and expose the soft mask
as a separate PixelData field.
"""

import numpy as np
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData
from mmseg.datasets.transforms import LoadAnnotations, PackSegInputs
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class SBULabelTransform(LoadAnnotations):
    """Convert SBU labels from 0/255 or non-zero values to binary 0/1."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, results):
        results = super().transform(results)
        if "gt_seg_map" in results:
            seg_map = results["gt_seg_map"]
            if isinstance(seg_map, np.ndarray):
                results["gt_seg_map"] = (seg_map > 0).astype(np.uint8)
        return results


@TRANSFORMS.register_module(force=True)
class RefineAnnTransform(LoadAnnotations):
    """Load SBU-Refine annotations as both binary and soft targets.

    SBU-Refine masks are soft labels in [0, 255]. The binary detector still uses
    a thresholded mask, while the penumbra confidence head can regress the
    original continuous mask through gt_soft_seg_map.
    """

    def transform(self, results):
        results = super().transform(results)
        if "gt_seg_map" in results:
            seg_map = results["gt_seg_map"]
            if isinstance(seg_map, np.ndarray):
                soft_map = seg_map.astype(np.float32) / 255.0
                results["gt_soft_seg_map"] = np.clip(soft_map, 0.0, 1.0)
                if "seg_fields" in results and "gt_soft_seg_map" not in results["seg_fields"]:
                    results["seg_fields"].append("gt_soft_seg_map")
                results["gt_seg_map"] = (seg_map >= 128).astype(np.uint8)
        return results


@TRANSFORMS.register_module(force=True)
class PackSegInputsWithSoft(PackSegInputs):
    """Pack segmentation inputs and optional SBU-Refine soft mask."""

    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        if "gt_soft_seg_map" not in results:
            return packed_results

        soft_map = results["gt_soft_seg_map"]
        if len(soft_map.shape) == 2:
            data = to_tensor(soft_map[None, ...].astype(np.float32))
        else:
            data = to_tensor(soft_map.astype(np.float32))
        packed_results["data_samples"].set_data(
            dict(gt_soft_seg=PixelData(data=data)))
        return packed_results
