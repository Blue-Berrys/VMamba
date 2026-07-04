import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from transforms.sbu_label_transform import (
    PackSegInputsWithSoft,
    RefineAnnTransform,
    RefineHardNegativeAnnTransform,
    RefineIlluminationSoftAnnTransform,
)


def test_refine_transform_keeps_binary_mask_and_soft_mask():
    mask = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mask.png")
        Image.fromarray(mask).save(path)
        results = dict(
            seg_map_path=path,
            reduce_zero_label=False,
            seg_fields=[],
        )

        out = RefineAnnTransform(reduce_zero_label=False).transform(results)

    np.testing.assert_array_equal(out["gt_seg_map"], np.array([[0, 0], [1, 1]], dtype=np.uint8))
    assert "gt_soft_seg_map" in out
    np.testing.assert_allclose(out["gt_soft_seg_map"], mask.astype(np.float32) / 255.0)
    assert "gt_soft_seg_map" in out["seg_fields"]


def test_illumination_transform_suppresses_outer_soft_target_only():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 3] = 96
    mask[:, 4] = 160
    mask[:, 5:] = 255
    img = np.full((8, 8, 3), 128, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mask.png")
        Image.fromarray(mask).save(path)
        results = dict(
            img=img,
            seg_map_path=path,
            reduce_zero_label=False,
            seg_fields=[],
        )

        out = RefineIlluminationSoftAnnTransform(
            reduce_zero_label=False,
            band_width=4,
            outer_suppress=0.40,
            distance_weight=0.0,
        ).transform(results)

    np.testing.assert_array_equal(out["gt_seg_map"], (mask >= 128).astype(np.uint8))
    assert out["gt_soft_seg_map"][:, 3].mean() < (96.0 / 255.0)
    np.testing.assert_allclose(out["gt_soft_seg_map"][:, 4], 160.0 / 255.0)
    assert "gt_soft_seg_map" in out["seg_fields"]


def test_hard_negative_transform_marks_dark_non_shadow_regions():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:, 8:] = 255
    img = np.full((12, 12, 3), 210, dtype=np.uint8)
    img[3:9, 1:4] = 30      # dark non-shadow distractor, far from boundary
    img[3:9, 9:11] = 30     # dark true-shadow interior, must not be negative

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mask.png")
        Image.fromarray(mask).save(path)
        results = dict(
            img=img,
            seg_map_path=path,
            reduce_zero_label=False,
            seg_fields=[],
        )

        out = RefineHardNegativeAnnTransform(
            reduce_zero_label=False,
            boundary_exclude=2,
            dark_percentile=35.0,
            dark_score_threshold=0.35,
        ).transform(results)

    hard_neg = out["gt_dark_neg_map"]
    assert hard_neg[3:9, 1:4].mean() > 0.5
    assert hard_neg[3:9, 9:11].max() == 0.0
    assert hard_neg[:, 6:9].max() == 0.0
    assert "gt_dark_neg_map" in out["seg_fields"]


def test_pack_seg_inputs_with_soft_adds_soft_pixel_data():
    results = dict(
        img=np.zeros((2, 2, 3), dtype=np.uint8),
        gt_seg_map=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        gt_soft_seg_map=np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32),
        gt_dark_neg_map=np.array([[0.0, 0.8], [0.4, 0.0]], dtype=np.float32),
        img_path="dummy.jpg",
        seg_map_path="dummy.png",
        ori_shape=(2, 2),
        img_shape=(2, 2),
        pad_shape=(2, 2),
        scale_factor=(1.0, 1.0),
        reduce_zero_label=False,
    )

    packed = PackSegInputsWithSoft().transform(results)
    data = packed["data_samples"]

    assert hasattr(data, "gt_soft_seg")
    assert hasattr(data, "gt_dark_neg")
    assert data.gt_soft_seg.data.shape == (1, 2, 2)
    assert data.gt_dark_neg.data.shape == (1, 2, 2)
    np.testing.assert_allclose(
        data.gt_soft_seg.data.numpy(),
        np.array([[[0.0, 0.25], [0.5, 1.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        data.gt_dark_neg.data.numpy(),
        np.array([[[0.0, 0.8], [0.4, 0.0]]], dtype=np.float32),
    )
