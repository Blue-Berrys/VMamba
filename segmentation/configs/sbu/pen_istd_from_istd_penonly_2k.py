# Pure penumbra-branch adaptation from the existing ISTD checkpoint.
# This keeps the binary detector functionally fixed: only penumbra_conv learns.

_base_ = "./pen_istd_from_istd_head_5k.py"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=("decode_head.boundary_module.penumbra_conv",),
        priority="VERY_LOW",
    ),
]

model = dict(
    decode_head=dict(
        spe_loss_weight=0.0,
        boundary_loss_weight=0.0,
        tversky_loss_weight=0.0,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.0),
            dict(type="DiceLoss", use_sigmoid=True, loss_weight=0.0),
        ],
        soft_mask_loss_weight=0.0,
        inner_shadow_margin_loss_weight=0.0,
        dark_negative_loss_weight=0.0,
        dark_negative_rank_loss_weight=0.0,
        online_dark_fp_loss_weight=0.0,
        shadow_core_recall_loss_weight=0.0,
        soft_boundary_loss_weight=0.4,
        penumbra_grad_loss_weight=0.15,
        penumbra_mono_loss_weight=0.10,
        boundary_consistency_loss_weight=0.20,
        penumbra_refine_logits=False,
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=1,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=1.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=100),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=100, end=2000, eta_min=1e-6),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=2000, val_interval=500)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        save_best="BER",
        rule="less",
    ),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
)

work_dir = "work_dirs/pen_istd_from_istd_penonly_2k"
