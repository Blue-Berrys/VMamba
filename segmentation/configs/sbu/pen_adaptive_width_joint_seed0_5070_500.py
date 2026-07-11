_base_ = './pen_adaptive_width_rgb_boundarysafe_5070_500.py'

randomness = dict(seed=20260711, deterministic=False)
work_dir = 'work_dirs/pen_adaptive_width_joint_seed0_5070_500'

# This differs from the target-only control only by the width predictor, width
# regression loss, and zero-initialized width-conditioned residual scalar.
custom_hooks = [
    dict(
        type='TrainOnlyPenumbraHook',
        trainable_keywords=(
            'decode_head.boundary_module.penumbra_conv',
            'decode_head.boundary_module.gamma_p',
            'decode_head.boundary_module.width_conv',
            'decode_head.boundary_module.gamma_width',
            'decode_head.conv_seg',
        ),
        priority='VERY_LOW',
    ),
]
