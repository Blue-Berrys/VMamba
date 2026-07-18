"""P0 2x2 row 11: full BG-SIR plus PaSCL model."""

_base_ = "./pen_p0_2x2_common_40k.py"

model = dict(
    decode_head=dict(
        use_bg_sir=True,
        soft_mask_loss_weight=0.29,
        soft_boundary_loss_weight=0.20,
        penumbra_grad_loss_weight=0.05,
        penumbra_mono_loss_weight=0.02,
        boundary_consistency_loss_weight=0.05,
    ),
)

work_dir = "./work_dirs/pen_p0_2x2_11_bg_pascl_seed20260718_40k"
