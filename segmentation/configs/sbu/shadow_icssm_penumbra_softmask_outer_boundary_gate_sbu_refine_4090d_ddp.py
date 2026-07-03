# Boundary-module-only outer penumbra soft-mask gate.
#
# Gate-only training improves dark false positives but plateaus around 2.74 BER.
# Full-decoder adaptation hurts recall. This middle-capacity sweep trains the
# whole boundary module plus the final classifier while keeping the IC-SSM/SASF
# detector frozen. A small boundary loss gives the binary boundary path a direct
# signal, while the penumbra branch keeps the outer soft-mask objective.

_base_ = "./shadow_icssm_penumbra_softmask_outer_gate_sbu_refine_4090d_ddp.py"

custom_hooks = [
    dict(
        type="TrainOnlyPenumbraHook",
        trainable_keywords=(
            "decode_head.boundary_module",
            "decode_head.conv_seg",
        ),
        priority="VERY_LOW",
    ),
]

model = dict(
    decode_head=dict(
        soft_mask_region="outer",
        soft_mask_loss_weight=0.25,
        tversky_loss_weight=0.15,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        boundary_loss_weight=0.05,
    ),
)

optim_wrapper = dict(
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module": dict(lr_mult=1.0),
            "decode_head.boundary_module.gamma_b": dict(
                lr_mult=20.0, decay_mult=0.0),
            "decode_head.boundary_module.gamma_p": dict(
                lr_mult=100.0, decay_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=1.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_boundary_gate_sbu_refine_4090d_ddp"
)
