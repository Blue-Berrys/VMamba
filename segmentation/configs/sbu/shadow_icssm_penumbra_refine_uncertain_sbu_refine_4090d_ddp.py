# Uncertainty-gated penumbra logit refinement.
#
# Previous logit-refine runs reduced false positives but increased false
# negatives because every boundary prediction could be shifted. This variant
# applies the penumbra correction only where the frozen binary detector has a
# small shadow/background logit margin, preserving confident predictions while
# still letting penumbra confidence clean uncertain boundary pixels.

_base_ = "./shadow_icssm_penumbra_refine_sbu_refine_4090d_ddp.py"

model = dict(
    decode_head=dict(
        penumbra_refine_uncertain_gate=True,
        penumbra_refine_margin=1.5,
        penumbra_refine_sharpness=2.0,
        penumbra_refine_max_delta=1.0,
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.penumbra_logit_scale_raw": dict(
                lr_mult=10.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

work_dir = "./work_dirs/shadow_icssm_penumbra_refine_uncertain_sbu_refine_4090d_ddp"
