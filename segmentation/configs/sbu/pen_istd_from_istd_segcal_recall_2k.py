# ISTD fixed-threshold calibration from the existing ISTD checkpoint.
# Only the final segmentation classifier is trainable; the objective is recall-biased
# because threshold sweep shows the old ISTD model is slightly under-confident for shadows.

_base_ = "./pen_istd_from_istd_head_5k.py"

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
            dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.6),
            dict(type="DiceLoss", use_sigmoid=True, loss_weight=0.2),
        ],
        tversky_loss_weight=0.35,
        tversky_alpha=0.20,
        tversky_beta=0.80,
        soft_mask_loss_weight=0.0,
        soft_boundary_loss_weight=0.0,
        penumbra_grad_loss_weight=0.0,
        penumbra_mono_loss_weight=0.0,
        boundary_consistency_loss_weight=0.0,
        inner_shadow_margin_loss_weight=0.0,
        dark_negative_loss_weight=0.0,
        dark_negative_rank_loss_weight=0.0,
        online_dark_fp_loss_weight=0.0,
        shadow_core_recall_loss_weight=0.0,
        penumbra_refine_logits=False,
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=2,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type="AdamW", lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=1.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=100),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=100, end=2000, eta_min=3e-7),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=2000, val_interval=250)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=250,
        max_keep_ckpts=4,
        save_best="BER",
        rule="less",
    ),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
)

work_dir = "work_dirs/pen_istd_from_istd_segcal_recall_2k"
