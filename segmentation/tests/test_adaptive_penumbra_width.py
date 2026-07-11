import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from segmentation.ic_ssm_head import ShadowBoundaryModule
from segmentation.transforms.sbu_label_transform import (
    PackSegInputsWithSoft,
    RefineAdaptivePenumbraAnnTransform,
)


def _run_teacher(luma_profile):
    height, width = 48, len(luma_profile)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, : width // 2] = 255
    image = np.repeat(np.asarray(luma_profile, dtype=np.uint8)[None, :, None], height, axis=0)
    image = np.repeat(image, 3, axis=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mask.png")
        Image.fromarray(mask).save(path)
        results = dict(
            img=image,
            seg_map_path=path,
            reduce_zero_label=False,
            seg_fields=[],
        )
        return RefineAdaptivePenumbraAnnTransform(
            reduce_zero_label=False,
            band_width=12,
            min_width=1.0,
            max_width=16.0,
            reliability_min=0.05,
        ).transform(results)


def test_rgb_teacher_assigns_wider_target_to_soft_transition():
    x = np.arange(64, dtype=np.float32)
    hard = np.where(x < 32, 40.0, 210.0)
    soft = 40.0 + 170.0 * np.clip((x - 22.0) / 20.0, 0.0, 1.0)

    hard_out = _run_teacher(hard)
    soft_out = _run_teacher(soft)
    hard_band = hard_out["gt_soft_weight_map"] > 0.5
    soft_band = soft_out["gt_soft_weight_map"] > 0.5

    hard_width = hard_out["gt_penumbra_width_map"][hard_band].mean()
    soft_width = soft_out["gt_penumbra_width_map"][soft_band].mean()
    assert soft_width > hard_width + 0.20
    assert soft_band.sum() > hard_band.sum()
    assert soft_out["gt_soft_seg_map"][:, 24:40].std() > 0.15


def test_pack_seg_inputs_includes_penumbra_width():
    results = dict(
        img=np.zeros((2, 2, 3), dtype=np.uint8),
        gt_seg_map=np.array([[1, 1], [0, 0]], dtype=np.uint8),
        gt_soft_seg_map=np.array([[1.0, 0.7], [0.3, 0.0]], dtype=np.float32),
        gt_soft_weight_map=np.ones((2, 2), dtype=np.float32),
        gt_penumbra_width_map=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        img_path="dummy.jpg",
        seg_map_path="dummy.png",
        ori_shape=(2, 2),
        img_shape=(2, 2),
        pad_shape=(2, 2),
        scale_factor=(1.0, 1.0),
        reduce_zero_label=False,
    )

    data = PackSegInputsWithSoft().transform(results)["data_samples"]
    assert hasattr(data, "gt_penumbra_width")
    np.testing.assert_allclose(
        data.gt_penumbra_width.data.numpy(),
        np.array([[[0.2, 0.4], [0.6, 0.8]]], dtype=np.float32),
    )


def test_predicted_width_changes_boundary_context_scale():
    torch.manual_seed(7)
    module = ShadowBoundaryModule(channels=32).eval()
    module.gamma_width.data.fill_(1.0)
    x = torch.randn(2, 32, 16, 16)

    for parameter in module.width_conv.parameters():
        parameter.data.zero_()
    module.width_conv[-1].bias.data.fill_(-8.0)
    narrow_out, _, _, narrow_width = module(x)
    module.width_conv[-1].bias.data.fill_(8.0)
    wide_out, _, _, wide_width = module(x)

    assert narrow_width.shape == (2, 1, 16, 16)
    assert wide_width.sigmoid().mean() > narrow_width.sigmoid().mean() + 0.9
    assert not torch.allclose(narrow_out, wide_out)


def test_width_context_is_suppressed_away_from_predicted_boundary():
    torch.manual_seed(11)
    module = ShadowBoundaryModule(channels=32).eval()
    module.gamma_width.data.fill_(1.0)
    module.gamma_b.data.zero_()
    module.gamma_p.data.zero_()
    x = torch.randn(2, 32, 16, 16)

    for parameter in module.boundary_conv.parameters():
        parameter.data.zero_()
    module.boundary_conv[-1].bias.data.fill_(-10.0)
    output, _, _, _ = module(x)

    assert torch.allclose(output, x, atol=1e-4, rtol=1e-4)
