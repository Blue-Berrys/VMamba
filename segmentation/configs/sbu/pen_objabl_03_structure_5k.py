"""Objective ablation 3: add penumbra gradient and monotonic constraints."""

_base_ = (
    "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_"
    "sbu_refine_4090d_ddp.py"
)

model = dict(
    decode_head=dict(
        # Keep the host Tversky term fixed across every objective-ablation row.
        tversky_loss_weight=0.15,
        soft_mask_loss_weight=0.29,
        soft_boundary_loss_weight=0.20,
        penumbra_grad_loss_weight=0.05,
        penumbra_mono_loss_weight=0.02,
        boundary_consistency_loss_weight=0.0,
    ),
)

work_dir = "./work_dirs/pen_objabl_03_structure_5k"
