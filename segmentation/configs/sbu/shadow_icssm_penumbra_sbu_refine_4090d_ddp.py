# MMSegmentation config: IC-SSM + BG-SIR + Penumbra-aware Shadow Confidence
# ============================================================================
#
# This experiment replaces the previous Tversky add-on with an auxiliary
# continuous soft-shadow/penumbra confidence task.
#
# Main mask remains binary shadow detection. The new penumbra branch is
# supervised by a signed-distance soft target generated from the binary GT mask,
# plus gradient, monotonicity, and boundary-consistency constraints.
# ============================================================================

_base_ = './shadow_icssm_refine.py'

load_from = 'work_dirs/shadow_icssm_refine/best_BER_iter_25000.pth'

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

model = dict(
    decode_head=dict(
        # Replace ordinary Tversky with penumbra-aware auxiliary constraints.
        tversky_loss_weight=0.0,
        soft_boundary_loss_weight=0.4,
        penumbra_grad_loss_weight=0.15,
        penumbra_mono_loss_weight=0.10,
        boundary_consistency_loss_weight=0.20,
        penumbra_band_width=8,
        penumbra_tau=2.0,
        # Keep a light binary boundary loss so the continuous map remains
        # anchored to the existing boundary cue.
        boundary_loss_weight=0.2,
        loss_decode=[
            # Stable BCEWithLogits path for the binary main mask.
            # CUDA focal loss can return NaN after BG-SIR warm-start.
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.7),
            dict(type='DiceLoss', use_sigmoid=True, loss_weight=0.3),
        ],
    ),
)

optim_wrapper = dict(
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'decode_head.sasf': dict(lr_mult=1.0),
            'decode_head.ic_ssm': dict(lr_mult=1.0),
            'decode_head.ic_ssm.bg_sir': dict(lr_mult=1.5),
            'decode_head.ic_ssm.prior_head': dict(lr_mult=1.5),
            'decode_head.boundary_module': dict(lr_mult=1.5),
            'decode_head.boundary_module.penumbra_conv': dict(lr_mult=2.0),
            'auxiliary_head': dict(lr_mult=0.5),
        },
    ),
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=30000,
    val_interval=5000,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.3,
         by_epoch=False, begin=0, end=300),
    dict(type='CosineAnnealingLR', by_epoch=False,
         begin=300, end=30000, eta_min=1e-7),
]

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=30000,
        log_stage_switch=True,
    ),
]

work_dir = './work_dirs/shadow_icssm_penumbra_sbu_refine_4090d_ddp'
