# Head-only penumbra confidence training on top of the fixed BG-SIR detector.
#
# Purpose: test whether the penumbra branch learns useful soft-boundary signal
# without perturbing the binary mask detector.  TrainOnlyPenumbraHook keeps the
# loaded detector in eval mode and only trains boundary_module.penumbra_conv.

_base_ = "./shadow_icssm_penumbra_sbu_refine_4090d_ddp.py"

custom_hooks = [
    dict(type="TrainOnlyPenumbraHook", priority="VERY_LOW"),
]

model = dict(
    decode_head=dict(
        # Disable detector losses for this diagnostic run.  The main detector is
        # frozen; only the confidence head receives the penumbra losses below.
        spe_loss_weight=0.0,
        boundary_loss_weight=0.0,
        tversky_loss_weight=0.0,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.0),
            dict(type="DiceLoss", use_sigmoid=True, loss_weight=0.0),
        ],
        soft_boundary_loss_weight=0.4,
        penumbra_grad_loss_weight=0.15,
        penumbra_mono_loss_weight=0.10,
        boundary_consistency_loss_weight=0.20,
    ),
)

optim_wrapper = dict(
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=5000,
    val_interval=1000,
)

work_dir = "./work_dirs/shadow_icssm_penumbra_headonly_sbu_refine_4090d_ddp"
