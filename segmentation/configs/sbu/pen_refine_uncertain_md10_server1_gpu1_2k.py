# Less conservative uncertainty-gated penumbra logit refinement on the current
# soft-mask mainline. This tests whether a wider uncertain-margin and larger
# bounded shift can improve recall/precision balance without destabilizing BER.

_base_ = "./pen_main_soft0275_bw4_tv025_b075_server1_gpu1_2k.py"

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

model = dict(
    decode_head=dict(
        penumbra_refine_logits=True,
        penumbra_refine_boundary_gate=True,
        penumbra_refine_uncertain_gate=True,
        penumbra_refine_margin=1.5,
        penumbra_refine_sharpness=2.0,
        penumbra_refine_max_delta=1.0,
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "auxiliary_head": dict(lr_mult=0.0),
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module": dict(lr_mult=1.5),
            "decode_head.boundary_module.gamma_p": dict(
                lr_mult=100.0, decay_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.penumbra_logit_scale_raw": dict(
                lr_mult=100.0, decay_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=1.0),
            "decode_head.ic_ssm": dict(lr_mult=1.0),
            "decode_head.ic_ssm.bg_sir": dict(lr_mult=1.5),
            "decode_head.ic_ssm.prior_head": dict(lr_mult=1.5),
            "decode_head.sasf": dict(lr_mult=1.0),
        },
    ),
)

work_dir = "work_dirs/pen_refine_uncertain_md10_server1_gpu1_2k"
