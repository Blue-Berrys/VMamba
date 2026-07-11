_base_ = './pen_adaptive_width_rgb_boundarysafe_5070_500.py'

randomness = dict(seed=20260711, deterministic=False)
load_from = 'work_dirs/pen_adaptive_width_only_5070_500/iter_500.pth'
work_dir = 'work_dirs/pen_adaptive_width_feedback_only_5070_500'

model = dict(
    decode_head=dict(
        penumbra_width_loss_weight=0.0,
    ),
)

# The width predictor and binary detector stay frozen. Segmentation losses can
# change the prediction only through the scalar adaptive-context residual.
custom_hooks = [
    dict(
        type='TrainOnlyPenumbraHook',
        trainable_keywords=(
            'decode_head.boundary_module.gamma_width',
        ),
        priority='VERY_LOW',
    ),
]
