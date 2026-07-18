"""P0 2x2 row 10: mean reference plus PaSCL supervision."""

_base_ = "./pen_p0_2x2_common_40k.py"

model = dict(
    decode_head=dict(
        soft_mask_loss_weight=0.29,
        soft_boundary_loss_weight=0.20,
        penumbra_grad_loss_weight=0.05,
        penumbra_mono_loss_weight=0.02,
        boundary_consistency_loss_weight=0.05,
    ),
)

work_dir = "./work_dirs/pen_p0_2x2_10_nobg_pascl_seed20260718_40k"
