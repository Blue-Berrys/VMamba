"""Objective ablation 0: matched binary-only fine-tuning control."""

_base_ = (
    "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_"
    "sbu_refine_4090d_ddp.py"
)

model = dict(
    decode_head=dict(
        # Keep the host objective and architecture fixed while disabling every
        # PaSCL supervision term.
        tversky_loss_weight=0.15,
        soft_mask_loss_weight=0.0,
        soft_boundary_loss_weight=0.0,
        penumbra_grad_loss_weight=0.0,
        penumbra_mono_loss_weight=0.0,
        boundary_consistency_loss_weight=0.0,
    ),
)

work_dir = "./work_dirs/pen_objabl_00_binary_5k"
