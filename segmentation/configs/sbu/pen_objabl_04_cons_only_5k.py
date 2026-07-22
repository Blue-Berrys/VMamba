"""Pruned PaSCL objective: soft mask, confidence, and consistency only."""

_base_ = "./pen_objabl_02_conf_5k.py"

model = dict(
    decode_head=dict(
        # Gradient and monotonicity losses are intentionally removed.
        penumbra_grad_loss_weight=0.0,
        penumbra_mono_loss_weight=0.0,
        boundary_consistency_loss_weight=0.05,
    ),
)

work_dir = "./work_dirs/pen_objabl_04_cons_only_5k"
