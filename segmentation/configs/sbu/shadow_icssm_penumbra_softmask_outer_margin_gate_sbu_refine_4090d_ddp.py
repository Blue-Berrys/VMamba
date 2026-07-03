# Outer-side penumbra soft-mask gate with inner-side recall margin.
#
# The outer-only variant keeps improving BER by suppressing boundary false
# positives, but FNR rises as the learned gate becomes suppressive.  This
# variant keeps soft-mask regression on the non-shadow side and adds a small
# shadow-vs-background logit-margin loss on the shadow-side penumbra band.

_base_ = "./shadow_icssm_penumbra_softmask_outer_gate_sbu_refine_4090d_ddp.py"

model = dict(
    decode_head=dict(
        inner_shadow_margin_loss_weight=0.05,
        inner_shadow_margin=0.5,
    ),
)

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_margin_gate_sbu_refine_4090d_ddp"
)
