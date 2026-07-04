# Strong illumination-transition reliability weighting.

_base_ = "./pen_main_soft0275_bw4_tv025_b075_rel_wonly_server1_gpu0_2k.py"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RefineIlluminationSoftAnnTransform",
        reduce_zero_label=False,
        band_width=4,
        tau=2.0,
        distance_weight=0.0,
        outer_suppress=0.0,
        inner_distance_weight=0.0,
        reliability_min=0.15,
        reliability_power=2.0,
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
    "pen_main_soft0275_bw4_tv025_b075_rel_strong_server1_gpu0_2k"
)
