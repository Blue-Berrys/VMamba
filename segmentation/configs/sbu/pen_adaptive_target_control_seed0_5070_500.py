_base_ = './pen_adaptive_width_rgb_boundarysafe_5070_500.py'

randomness = dict(seed=20260711, deterministic=False)
work_dir = 'work_dirs/pen_adaptive_target_control_seed0_5070_500'

model = dict(
    decode_head=dict(
        penumbra_width_loss_weight=0.0,
    ),
)

# Same adaptive RGB/GT soft targets and binary fine-tuning as the joint model,
# but no learned width representation or width-conditioned feature feedback.
custom_hooks = [
    dict(
        type='TrainOnlyPenumbraHook',
        trainable_keywords=(
            'decode_head.boundary_module.penumbra_conv',
            'decode_head.boundary_module.gamma_p',
            'decode_head.conv_seg',
        ),
        priority='VERY_LOW',
    ),
]
