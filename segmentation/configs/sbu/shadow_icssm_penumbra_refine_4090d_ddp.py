# Penumbra-guided logit refinement on top of the fixed BG-SIR detector.
#
# This is the next-step experiment after head-only confidence learning:
# keep the loaded binary detector frozen, learn a continuous penumbra map, and
# let it calibrate final shadow/background logits only through a zero-initialized
# bounded scale. Loading an old BG-SIR checkpoint is therefore functionally
# identical at iteration 0.

_base_ = "./shadow_icssm_penumbra_bgsir_4090d_ddp.py"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=(
            "decode_head.boundary_module.penumbra_conv",
            "decode_head.penumbra_logit_scale_raw",
        ),
        priority="VERY_LOW",
    ),
]

model = dict(
    auxiliary_head=dict(
        loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.0),
    ),
    decode_head=dict(
        penumbra_refine_logits=True,
        penumbra_refine_boundary_gate=True,
        penumbra_refine_max_delta=2.0,
        spe_loss_weight=0.0,
        boundary_loss_weight=0.0,
        tversky_loss_weight=0.0,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.7),
            dict(type="DiceLoss", use_sigmoid=True, loss_weight=0.3),
        ],
        soft_boundary_loss_weight=0.3,
        penumbra_grad_loss_weight=0.10,
        penumbra_mono_loss_weight=0.05,
        boundary_consistency_loss_weight=0.10,
    ),
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

optim_wrapper = dict(
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.penumbra_logit_scale_raw": dict(
                lr_mult=20.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=5000,
    val_interval=1000,
)

work_dir = "./work_dirs/shadow_icssm_penumbra_refine_4090d_ddp"
