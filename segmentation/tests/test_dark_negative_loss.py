import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import segmentation  # noqa: F401
from mmseg.registry import MODELS


ICShadowHead = MODELS.get("ICShadowHead")


def test_dark_negative_margin_loss_penalizes_shadow_logits_on_hard_negatives():
    seg_logits = torch.zeros(1, 2, 2, 2)
    seg_logits[:, 1] = 1.0
    seg_logits[:, 0] = 0.0
    hard_neg = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])

    loss = ICShadowHead.compute_dark_negative_margin_loss(
        seg_logits,
        hard_neg,
        margin=0.25,
    )

    assert loss.item() > 1.0


def test_dark_negative_margin_loss_is_zero_when_background_margin_is_satisfied():
    seg_logits = torch.zeros(1, 2, 2, 2)
    seg_logits[:, 1] = -0.75
    seg_logits[:, 0] = 0.0
    hard_neg = torch.ones(1, 1, 2, 2)

    loss = ICShadowHead.compute_dark_negative_margin_loss(
        seg_logits,
        hard_neg,
        margin=0.25,
    )

    assert loss.item() == 0.0
