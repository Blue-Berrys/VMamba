_base_ = './pen_adaptive_width_rgb_boundarysafe_5070_500.py'

randomness = dict(seed=20260711, deterministic=False)
work_dir = 'work_dirs/pen_adaptive_width_only_5070_500'

# Learn the physical-width representation while keeping gamma_width frozen at
# zero. The binary detector must therefore remain identical to the initializer.
custom_hooks = [
    dict(
        type='TrainOnlyPenumbraHook',
        trainable_keywords=(
            'decode_head.boundary_module.width_conv',
        ),
        priority='VERY_LOW',
    ),
]
