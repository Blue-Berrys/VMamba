# Full IG-PaSCL / penumbra-aware ISTD fine-tuning from the current SBU
# global-BER best checkpoint. This is intended for the paper's ISTD in-domain
# result, not for SBU->ISTD direct-transfer.

_base_ = "./pen_main_soft0275_bw4_tv025_b075_server1_gpu1_2k.py"

load_from = (
    "work_dirs/pen_main_soft0275_bw4_tv025_b075_server1_gpu1_2k/"
    "best_BER_iter_250.pth"
)

# Replace the SBU head-only fine-tuning hook. ISTD needs domain adaptation, so
# use conservative full-model fine-tuning instead of training only penumbra head.
custom_hooks = []

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="SBULabelTransform", reduce_zero_label=False),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomRotate", prob=0.3, degree=15),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18,
    ),
    dict(type="PackSegInputsWithSoft"),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="SBULabelTransform", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="ISTDDataset",
        data_root="data/ISTD_binary",
        data_prefix=dict(img_path="train/img", seg_map_path="train/mask"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ISTDDataset",
        data_root="data/ISTD_binary",
        data_prefix=dict(img_path="test/img", seg_map_path="test/mask"),
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    accumulative_counts=4,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type="AdamW", lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.05),
            "decode_head.sasf": dict(lr_mult=1.0),
            "decode_head.ic_ssm": dict(lr_mult=1.0),
            "decode_head.ic_ssm.bg_sir": dict(lr_mult=1.5),
            "decode_head.ic_ssm.prior_head": dict(lr_mult=1.5),
            "decode_head.boundary_module": dict(lr_mult=1.5),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.penumbra_logit_scale_raw": dict(lr_mult=10.0, decay_mult=0.0),
            "decode_head.conv_seg": dict(lr_mult=1.0),
            "auxiliary_head": dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=200),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=200, end=10000, eta_min=1e-7),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best="BER",
        rule="less",
    ),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
)

work_dir = "work_dirs/pen_istd_fullft_global_10k"
