# Narrow-band outer penumbra soft-mask gate (soft=0.29, band_width=4).
#
# Best short SBU sweep setting observed on 2026-07-04. It keeps binary mask
# supervision for detection, and uses the penumbra confidence branch as
# outer-boundary soft-mask regression plus consistency regularization.

_base_ = "./shadow_icssm_penumbra_softmask_gate_sbu_refine_4090d_ddp.py"

model = dict(
    decode_head=dict(
        soft_mask_region="outer",
        soft_mask_loss_weight=0.29,
        penumbra_band_width=4,
        tversky_loss_weight=0.15,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    ),
)

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_sbu_refine_4090d_ddp"
)
