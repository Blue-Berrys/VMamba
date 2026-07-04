# Add the current penumbra-confidence supervision on top of the existing
# ISTD-specific checkpoint. This preserves the strong ISTD detector and learns
# the missing penumbra branch/soft constraints for a full-method ISTD result.

_base_ = "./pen_istd_fullft_global_10k.py"

load_from = "work_dirs/shadow_icssm_istd/best_BER_iter_21000.pth"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=(
            "decode_head.boundary_module.penumbra_conv",
            "decode_head.boundary_module.gamma_p",
            "decode_head.penumbra_logit_scale_raw",
            "decode_head.conv_seg",
        ),
        priority="VERY_LOW",
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=5000, val_interval=500)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=2,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type="AdamW", lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "auxiliary_head": dict(lr_mult=0.0),
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.boundary_module.gamma_p": dict(lr_mult=10.0, decay_mult=0.0),
            "decode_head.penumbra_logit_scale_raw": dict(lr_mult=10.0, decay_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=100),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=100, end=5000, eta_min=1e-7),
]

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

work_dir = "work_dirs/pen_istd_from_istd_head_5k"
