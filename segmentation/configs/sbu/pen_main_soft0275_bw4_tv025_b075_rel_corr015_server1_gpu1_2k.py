# Illumination-transition reliability weighting with mild target correction.
#
# Compared with the reliability-only variant, this suppresses unsupported
# outer-band soft targets mildly while preserving the main binary mask target.

_base_ = "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_sbu_refine_4090d_ddp.py"

load_from = (
    "work_dirs/shadow_icssm_penumbra_softmask_outer_soft029_bw4_gpu0_5k/"
    "best_BER_iter_4000.pth"
)

model = dict(
    decode_head=dict(
        soft_mask_region="outer",
        soft_mask_loss_weight=0.275,
        penumbra_band_width=4,
        tversky_loss_weight=0.25,
        tversky_alpha=0.25,
        tversky_beta=0.75,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RefineIlluminationSoftAnnTransform",
        reduce_zero_label=False,
        band_width=4,
        tau=2.0,
        distance_weight=0.10,
        outer_suppress=0.15,
        inner_distance_weight=0.0,
        reliability_min=0.30,
        reliability_power=1.0,
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

train_cfg = dict(type="IterBasedTrainLoop", max_iters=2000, val_interval=250)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3,
                    save_best="BER", rule="less"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

work_dir = (
    "./work_dirs/"
    "pen_main_soft0275_bw4_tv025_b075_rel_corr015_server1_gpu1_2k"
)
