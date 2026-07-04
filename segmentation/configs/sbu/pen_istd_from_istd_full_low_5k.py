# Low-LR full fine-tuning from the existing ISTD-specific checkpoint. This tests
# whether the full penumbra-aware objective can improve the strong ISTD detector
# without the slow adaptation needed from SBU checkpoints.

_base_ = "./pen_istd_fullft_global_10k.py"

load_from = "work_dirs/shadow_icssm_istd/best_BER_iter_21000.pth"

custom_hooks = []

train_cfg = dict(type="IterBasedTrainLoop", max_iters=5000, val_interval=500)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=4,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type="AdamW", lr=5e-6, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.02),
            "decode_head.sasf": dict(lr_mult=0.5),
            "decode_head.ic_ssm": dict(lr_mult=0.5),
            "decode_head.ic_ssm.bg_sir": dict(lr_mult=0.75),
            "decode_head.ic_ssm.prior_head": dict(lr_mult=0.75),
            "decode_head.boundary_module": dict(lr_mult=1.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.penumbra_logit_scale_raw": dict(lr_mult=10.0, decay_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=0.5),
            "auxiliary_head": dict(lr_mult=0.25),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=100),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=100, end=5000, eta_min=5e-8),
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

work_dir = "work_dirs/pen_istd_from_istd_full_low_5k"
