# Low-LR decoder/BG-SIR refine from the current best soft0.29/bw4 checkpoint.
#
# Purpose: test whether the local 2.734 BER plateau comes from over-freezing.
# It still freezes the backbone, but lets the illumination/reference and
# boundary-facing decoder components adapt with a conservative LR.

_base_ = "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_sbu_refine_4090d_ddp.py"

load_from = (
    "work_dirs/shadow_icssm_penumbra_softmask_outer_soft029_bw4_gpu0_5k/"
    "best_BER_iter_4000.pth"
)

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=(
            "decode_head.sasf",
            "decode_head.ic_ssm.bg_sir",
            "decode_head.boundary_module",
            "decode_head.fusion",
            "decode_head.conv_seg",
        ),
        priority="VERY_LOW",
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=8000, val_interval=500)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3,
                    save_best="BER", rule="less"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(type="AdamW", lr=6e-6, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.sasf": dict(lr_mult=0.5),
            "decode_head.ic_ssm.bg_sir": dict(lr_mult=1.0),
            "decode_head.boundary_module": dict(lr_mult=1.0),
            "decode_head.boundary_module.gamma_p": dict(lr_mult=20.0, decay_mult=0.0),
            "decode_head.boundary_module.gamma_b": dict(lr_mult=5.0, decay_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=1.5),
            "decode_head.fusion": dict(lr_mult=1.0),
            "decode_head.conv_seg": dict(lr_mult=1.0),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=200),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=200,
         end=8000, eta_min=1e-8),
]

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft029_bw4_decoder_lowlr8k_sbu_refine_4090d_ddp"
)
