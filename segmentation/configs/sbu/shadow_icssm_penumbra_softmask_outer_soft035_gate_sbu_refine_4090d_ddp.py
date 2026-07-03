# Recall-preserving penumbra soft-mask gate.
#
# The symmetric soft-mask gate reliably lowers FPR but can increase FNR because
# soft labels also pull down shadow-side boundary pixels. This variant keeps
# the penumbra confidence head and feature gate, but applies final-mask soft
# regression only on the non-shadow side of the boundary band. A small Tversky
# term is restored as a recall guard.

_base_ = "./shadow_icssm_penumbra_softmask_gate_sbu_refine_4090d_ddp.py"

model = dict(
    decode_head=dict(
        soft_mask_region="outer",
        soft_mask_loss_weight=0.35,
        tversky_loss_weight=0.15,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    ),
)

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft035_gate_sbu_refine_4090d_ddp"
)
