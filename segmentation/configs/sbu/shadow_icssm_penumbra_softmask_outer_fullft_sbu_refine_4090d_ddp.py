# Full-decoder adaptation for outer-side penumbra soft-mask supervision.
#
# Gate-only training proved that soft supervision on the non-shadow side of the
# boundary band lowers false positives, but the tiny trainable subset plateaus
# around 2.74 BER.  This config keeps the same outer soft-mask objective and a
# small Tversky recall guard, while allowing the IC-SSM decoder to adapt at a
# conservative learning rate.

_base_ = "./shadow_icssm_penumbra_sbu_refine_4090d_ddp.py"

model = dict(
    auxiliary_head=dict(
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    decode_head=dict(
        penumbra_refine_logits=False,
        spe_loss_weight=0.0,
        boundary_loss_weight=0.0,
        tversky_loss_weight=0.10,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.7),
            dict(type="DiceLoss", use_sigmoid=True, loss_weight=0.3),
        ],
        soft_mask_region="outer",
        soft_mask_loss_weight=0.20,
        soft_boundary_loss_weight=0.20,
        penumbra_grad_loss_weight=0.05,
        penumbra_mono_loss_weight=0.02,
        boundary_consistency_loss_weight=0.05,
    ),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(lr=1.0e-5),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.05),
            "decode_head.sasf": dict(lr_mult=0.5),
            "decode_head.ic_ssm": dict(lr_mult=0.5),
            "decode_head.ic_ssm.bg_sir": dict(lr_mult=0.75),
            "decode_head.ic_ssm.prior_head": dict(lr_mult=0.75),
            "decode_head.boundary_module": dict(lr_mult=1.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.conv_seg": dict(lr_mult=1.0),
            "auxiliary_head": dict(lr_mult=0.2),
        },
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=8000,
    val_interval=500,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=300),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=300, end=8000, eta_min=1e-7),
]

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_fullft_sbu_refine_4090d_ddp"
)
