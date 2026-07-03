import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from segmentation.ic_ssm_head import ShadowBoundaryModule


def test_soft_penumbra_gt_prefers_refine_soft_mask():
    shadow_gt = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    sample = SegDataSample()
    sample.set_data(dict(gt_soft_seg=PixelData(data=torch.tensor([[[0.0, 0.25], [0.5, 1.0]]]))))

    soft, band, signed = ShadowBoundaryModule.get_soft_penumbra_gt(
        shadow_gt,
        band_width=0,
        tau=2.0,
        batch_data_samples=[sample],
    )

    np.testing.assert_allclose(
        soft.cpu().numpy(),
        np.array([[[[0.0, 0.25], [0.5, 1.0]]]], dtype=np.float32),
    )
    assert band[0, 0, 0, 1].item() is True
    assert band[0, 0, 1, 0].item() is True
    assert signed.shape == shadow_gt.shape
