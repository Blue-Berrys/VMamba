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


def test_dark_negative_ranking_loss_compares_hard_negatives_to_shadow_core():
    seg_logits = torch.zeros(1, 2, 2, 3)
    seg_logits[:, 1] = torch.tensor([[[2.0, 0.2, 0.0], [2.0, 0.2, 0.0]]])
    seg_logits[:, 0] = 0.0
    hard_neg = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]])
    shadow_core = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])

    loss = ICShadowHead.compute_dark_negative_ranking_loss(
        seg_logits,
        hard_neg,
        shadow_core,
        rank_margin=2.0,
    )

    assert loss.item() > 0.0


def test_dark_negative_ranking_loss_is_zero_when_rank_margin_is_satisfied():
    seg_logits = torch.zeros(1, 2, 2, 3)
    seg_logits[:, 1] = torch.tensor([[[3.0, 0.2, 0.0], [3.0, 0.2, 0.0]]])
    seg_logits[:, 0] = 0.0
    hard_neg = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]])
    shadow_core = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])

    loss = ICShadowHead.compute_dark_negative_ranking_loss(
        seg_logits,
        hard_neg,
        shadow_core,
        rank_margin=0.5,
    )

    assert loss.item() == 0.0


def test_online_dark_fp_loss_only_penalizes_confident_hard_negative_predictions():
    seg_logits = torch.zeros(1, 2, 2, 3)
    seg_logits[:, 1] = torch.tensor([[[2.0, 0.1, 2.0], [2.0, 0.1, 2.0]]])
    seg_logits[:, 0] = 0.0
    hard_neg = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])

    loss = ICShadowHead.compute_online_dark_fp_loss(
        seg_logits,
        hard_neg,
        fp_threshold=0.7,
        fp_gamma=1.0,
    )

    assert loss.item() > 0.0

    no_hard_negative = torch.zeros_like(hard_neg)
    empty_loss = ICShadowHead.compute_online_dark_fp_loss(
        seg_logits,
        no_hard_negative,
        fp_threshold=0.7,
    )

    assert empty_loss.item() == 0.0


def test_online_dark_fp_loss_is_zero_below_prediction_threshold():
    seg_logits = torch.zeros(1, 2, 2, 2)
    seg_logits[:, 1] = -0.25
    seg_logits[:, 0] = 0.0
    hard_neg = torch.ones(1, 1, 2, 2)

    loss = ICShadowHead.compute_online_dark_fp_loss(
        seg_logits,
        hard_neg,
        fp_threshold=0.7,
    )

    assert loss.item() == 0.0


def test_shadow_core_recall_anchor_penalizes_low_shadow_core_margin():
    seg_logits = torch.zeros(1, 2, 1, 2)
    seg_logits[:, 1] = torch.tensor([[[0.1, 2.0]]])
    seg_logits[:, 0] = 0.0
    shadow_core = torch.ones(1, 1, 1, 2)

    loss = ICShadowHead.compute_shadow_core_recall_loss(
        seg_logits,
        shadow_core,
        recall_margin=1.0,
    )

    assert loss.item() > 0.0

    satisfied = ICShadowHead.compute_shadow_core_recall_loss(
        seg_logits + torch.tensor([[[[0.0, 0.0]], [[2.0, 0.0]]]]),
        shadow_core,
        recall_margin=1.0,
    )
    assert satisfied.item() == 0.0
