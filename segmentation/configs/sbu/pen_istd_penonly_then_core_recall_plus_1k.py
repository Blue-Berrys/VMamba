# ISTD recall-margin calibration from the existing ISTD checkpoint.
# This diagnostic removes background-dominated BCE and trains only conv_seg with
# Tversky plus shadow-core recall margin.

_base_ = "./pen_istd_from_istd_segcal_core_recall_1k.py"

load_from = "work_dirs/pen_istd_from_istd_penonly_2k/best_BER_iter_500.pth"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=("decode_head.conv_seg",),
        priority="VERY_LOW",
    ),
]

model = dict(
    decode_head=dict(
        spe_loss_weight=0.0,
        boundary_loss_weight=0.0,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.0),
            dict(type="DiceLoss", use_sigmoid=True, loss_weight=0.05),
        ],
        tversky_loss_weight=0.55,
        tversky_alpha=0.15,
        tversky_beta=0.85,
        soft_mask_loss_weight=0.0,
        soft_boundary_loss_weight=0.0,
        penumbra_grad_loss_weight=0.0,
        penumbra_mono_loss_weight=0.0,
        boundary_consistency_loss_weight=0.0,
        inner_shadow_margin_loss_weight=0.0,
        dark_negative_loss_weight=0.0,
        dark_negative_rank_loss_weight=0.0,
        online_dark_fp_loss_weight=0.0,
        shadow_core_recall_loss_weight=0.15,
        shadow_core_recall_margin=0.95,
        dark_negative_core_kernel=9,
        penumbra_refine_logits=False,
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=2,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type="AdamW", lr=3e-5, betas=(0.9, 0.999), weight_decay=0.0),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=1.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.5, by_epoch=False, begin=0, end=50),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=50, end=1000, eta_min=5e-7),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=1000, val_interval=100)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=100,
        max_keep_ckpts=5,
        save_best="BER",
        rule="less",
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

work_dir = "work_dirs/pen_istd_penonly_then_core_recall_plus_1k"
