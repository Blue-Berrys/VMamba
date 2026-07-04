# Illumination-corrected outer penumbra soft-mask target.
#
# Starts from the current best soft0.29/bw4 checkpoint and keeps the same
# frozen-detector training regime. The only target-side change is replacing raw
# SBU-Refine soft masks with a signed-distance + local illumination/gradient
# corrected soft map, mainly suppressing unsupported outer-band soft targets.

_base_ = "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_sbu_refine_4090d_ddp.py"

load_from = (
    "work_dirs/shadow_icssm_penumbra_softmask_outer_soft029_bw4_gpu0_5k/"
    "best_BER_iter_4000.pth"
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RefineIlluminationSoftAnnTransform",
        reduce_zero_label=False,
        band_width=4,
        tau=2.0,
        distance_weight=0.10,
        outer_suppress=0.25,
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

train_cfg = dict(type="IterBasedTrainLoop", max_iters=5000, val_interval=500)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3,
                    save_best="BER", rule="less"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft029_bw4_illum025_sbu_refine_4090d_ddp"
)
