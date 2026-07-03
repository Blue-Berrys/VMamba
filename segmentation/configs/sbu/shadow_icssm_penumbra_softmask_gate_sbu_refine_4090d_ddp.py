# Penumbra soft-mask feature gate from the SBU-Refine best checkpoint.
#
# The earlier logit-refine variants only learned a scalar post-hoc shift and
# produced tiny BER changes on the strong SBU-Refine baseline.  This variant
# connects the continuous soft-mask supervision to the final shadow probability
# in the penumbra band while keeping the detector mostly frozen: train only the
# penumbra branch, its zero-initialized feature gate, and the final classifier.
# Inference still outputs the standard binary shadow mask.

_base_ = "./shadow_icssm_penumbra_sbu_refine_4090d_ddp.py"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=(
            "decode_head.boundary_module.penumbra_conv",
            "decode_head.boundary_module.gamma_p",
            "decode_head.conv_seg",
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
        soft_mask_loss_weight=0.25,
        soft_boundary_loss_weight=0.20,
        penumbra_grad_loss_weight=0.05,
        penumbra_mono_loss_weight=0.02,
        boundary_consistency_loss_weight=0.05,
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
            "decode_head.conv_seg": dict(lr_mult=1.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=5000,
    val_interval=500,
)

work_dir = "./work_dirs/shadow_icssm_penumbra_softmask_gate_sbu_refine_4090d_ddp"
