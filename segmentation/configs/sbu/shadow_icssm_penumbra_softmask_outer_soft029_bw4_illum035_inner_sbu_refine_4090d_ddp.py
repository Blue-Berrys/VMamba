# Balanced illumination-corrected penumbra target with inner recall protection.
#
# The first illumination sweep showed that stronger outer suppression reduces
# FPR but increases FNR. This variant keeps a moderate outer suppression and
# anchors the inner penumbra side with distance-aware soft labels plus a small
# inner-margin loss.

_base_ = "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_illum025_sbu_refine_4090d_ddp.py"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RefineIlluminationSoftAnnTransform",
        reduce_zero_label=False,
        band_width=4,
        tau=2.0,
        distance_weight=0.10,
        outer_suppress=0.35,
        inner_distance_weight=0.15,
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

model = dict(
    decode_head=dict(
        inner_shadow_margin_loss_weight=0.03,
        inner_shadow_margin=0.20,
    ),
)

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft029_bw4_illum035_inner_sbu_refine_4090d_ddp"
)
