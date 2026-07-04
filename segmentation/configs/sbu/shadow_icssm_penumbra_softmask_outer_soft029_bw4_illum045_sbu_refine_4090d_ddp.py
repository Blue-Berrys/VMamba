# Stronger illumination-corrected outer penumbra soft-mask target.
#
# This uses the same checkpoint and training regime as the illum025 variant,
# but applies a stronger outer-band suppression to test whether the current
# 2.734 BER plateau is primarily false-positive limited.

_base_ = "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_illum025_sbu_refine_4090d_ddp.py"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RefineIlluminationSoftAnnTransform",
        reduce_zero_label=False,
        band_width=4,
        tau=2.0,
        distance_weight=0.10,
        outer_suppress=0.45,
        inner_distance_weight=0.0,
        illum_sigma=3.0,
        illum_tau=0.08,
        grad_percentile=90.0,
    ),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomRotate", prob=0.3, degree=15),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=40,
        contrast_range=(0.4, 1.6),
        saturation_range=(0.4, 1.6),
        hue_delta=20,
    ),
    dict(type="PackSegInputsWithSoft"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft029_bw4_illum045_sbu_refine_4090d_ddp"
)
