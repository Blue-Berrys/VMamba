# VMamba + FocalLoss for SBU Shadow Detection
# Uses SDDNet threshold 125

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# Override model config to use FocalLoss
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.75,
            loss_weight=1.0
        )
    )
)

# Use AdamW optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0006,
        weight_decay=0.01
    )
)

work_dir = './work_dirs/vssm_tiny_sbu_focal'
