# Conservative penumbra-guided logit refinement.
#
# The first refine run improved boundary/dark false positives but increased
# false negatives because the learned logit scale became too large. This safe
# variant keeps the same mechanism but lowers the maximum logit shift and the
# scale learning-rate multiplier.

_base_ = "./shadow_icssm_penumbra_refine_4090d_ddp.py"

model = dict(
    decode_head=dict(
        penumbra_refine_max_delta=0.5,
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.penumbra_logit_scale_raw": dict(
                lr_mult=5.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

work_dir = "./work_dirs/shadow_icssm_penumbra_refine_safe_4090d_ddp"
