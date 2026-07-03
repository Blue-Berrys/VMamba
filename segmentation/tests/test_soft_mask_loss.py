import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from segmentation.ic_ssm_head import ICShadowHead


def test_soft_mask_loss_targets_shadow_probability_in_penumbra_band():
    seg_logits = torch.tensor(
        [[
            [[-4.0, -4.0]],
            [[-4.0, 0.0]],
        ]]
    )
    soft_target = torch.tensor([[[[0.25, 0.75]]]])
    band_valid = torch.tensor([[[[False, True]]]])

    loss = ICShadowHead.compute_soft_mask_loss(
        seg_logits,
        soft_target,
        band_valid,
        align_corners=False,
    )

    expected = F.smooth_l1_loss(torch.sigmoid(torch.tensor([0.0])), torch.tensor([0.75]))
    torch.testing.assert_close(loss, expected)
