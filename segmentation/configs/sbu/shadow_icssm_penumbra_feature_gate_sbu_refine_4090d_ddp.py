# Feature-level penumbra gate on top of the fixed BG-SIR detector.
#
# The logit-refine experiments showed that the penumbra map learns a real
# soft-boundary signal, but direct logit suppression improves FPR at the cost
# of recall. This variant keeps final logits untouched and trains only the
# continuous penumbra branch plus the zero-initialized feature gate gamma_p.
# It tests whether penumbra information can help the frozen decoder through
# a softer feature modulation path.

_base_ = "./shadow_icssm_penumbra_sbu_refine_4090d_ddp.py"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=(
            "decode_head.boundary_module.penumbra_conv",
            "decode_head.boundary_module.gamma_p",
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
        penumbra_refine_logits=False,
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
            "decode_head.boundary_module.gamma_p": dict(
                lr_mult=100.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=3000,
    val_interval=500,
)

work_dir = "./work_dirs/shadow_icssm_penumbra_feature_gate_sbu_refine_4090d_ddp"
