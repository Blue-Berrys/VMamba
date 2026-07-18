"""Strict matched P0 component ablation for the IG-PaSCL paper.

All four variants inherit this file and start from the same pre-BG-SIR
checkpoint.  They use identical data, optimization, validation, and random
seed.  Tversky is disabled so the objective matches the paper's BCE/Dice plus
PaSCL formulation.
"""

_base_ = "./shadow_icssm_penumbra_sbu_refine_4090d_ddp.py"

load_from = "work_dirs/shadow_icssm_tversky/best_BER_iter_30000.pth"

randomness = dict(seed=20260718, deterministic=False)

model = dict(
    decode_head=dict(
        use_bg_sir=False,
        tversky_loss_weight=0.0,
        soft_mask_loss_weight=0.0,
        soft_boundary_loss_weight=0.0,
        penumbra_grad_loss_weight=0.0,
        penumbra_mono_loss_weight=0.0,
        boundary_consistency_loss_weight=0.0,
        penumbra_width_loss_weight=0.0,
        inner_shadow_margin_loss_weight=0.0,
        soft_mask_region="outer",
        penumbra_band_width=4,
        penumbra_tau=2.0,
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=40000,
    val_interval=2000,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=300),
    dict(
        type="CosineAnnealingLR",
        by_epoch=False,
        begin=300,
        end=40000,
        eta_min=1e-7,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best="BER",
        rule="less",
    ),
)
