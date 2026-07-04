crop_size = (
    416,
    416,
)
custom_hooks = [
    dict(
        priority='VERY_LOW',
        trainable_keywords=(
            'decode_head.boundary_module.penumbra_conv',
            'decode_head.boundary_module.gamma_p',
            'decode_head.conv_seg',
        ),
        type='TrainOnlyPenumbraHook'),
]
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=0,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/SBU-shadow'
dataset_type = 'SBUDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        rule='less',
        save_best='BER',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'none'
load_from = 'work_dirs/pen_main_soft0285_bw4_tv025_b075_5090_2k/best_BER_iter_500.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=512,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        depths=(
            2,
            2,
            15,
            2,
        ),
        dims=128,
        downsample_version='v3',
        drop_path_rate=0.3,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        norm_layer='ln2d',
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patchembed_version='v2',
        pretrained='',
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        type='MM_VSSM'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        boundary_consistency_loss_weight=0.05,
        boundary_kernel=5,
        boundary_loss_weight=0.0,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(loss_weight=0.7, type='CrossEntropyLoss', use_sigmoid=True),
            dict(loss_weight=0.3, type='DiceLoss', use_sigmoid=True),
        ],
        num_classes=2,
        penumbra_band_width=4,
        penumbra_grad_loss_weight=0.05,
        penumbra_mono_loss_weight=0.02,
        penumbra_refine_logits=False,
        penumbra_tau=2.0,
        soft_boundary_loss_weight=0.2,
        soft_mask_loss_weight=0.285,
        soft_mask_region='outer',
        spe_loss_weight=0.0,
        ssm_d_state=16,
        ssm_drop_path=0.05,
        ssm_ratio=1.0,
        tversky_alpha=0.25,
        tversky_beta=0.75,
        tversky_loss_weight=0.25,
        type='ICShadowHead',
        use_bg_sir=True,
        use_sasf=True),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=8e-06, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict({
            'auxiliary_head':
            dict(lr_mult=0.0),
            'backbone':
            dict(lr_mult=0.0),
            'decode_head':
            dict(lr_mult=0.0),
            'decode_head.boundary_module':
            dict(lr_mult=1.5),
            'decode_head.boundary_module.gamma_p':
            dict(decay_mult=0.0, lr_mult=100.0),
            'decode_head.boundary_module.penumbra_conv':
            dict(lr_mult=2.0),
            'decode_head.conv_seg':
            dict(lr_mult=1.0),
            'decode_head.ic_ssm':
            dict(lr_mult=1.0),
            'decode_head.ic_ssm.bg_sir':
            dict(lr_mult=1.5),
            'decode_head.ic_ssm.prior_head':
            dict(lr_mult=1.5),
            'decode_head.sasf':
            dict(lr_mult=1.0)
        })),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=150, start_factor=0.5, type='LinearLR'),
    dict(
        begin=150,
        by_epoch=False,
        end=2000,
        eta_min=1e-07,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='SBU-Test/ShadowImages',
            seg_map_path='SBU-Test/ShadowMasks'),
        data_root='data/SBU-shadow',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='SBULabelTransform'),
            dict(type='PackSegInputs'),
        ],
        type='SBUDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='BERMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        416,
        416,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='SBULabelTransform'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=2000, type='IterBasedTrainLoop', val_interval=250)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='SBUTrain4KRecoveredSmall/ShadowImages',
            seg_map_path=
            '../SBU-Refine/refined_sbu_train/refined_sbu_train/ShadowMasks'),
        data_root='data/SBU-shadow',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='RefineAnnTransform'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(degree=15, prob=0.3, type='RandomRotate'),
            dict(
                brightness_delta=40,
                contrast_range=(
                    0.4,
                    1.6,
                ),
                hue_delta=20,
                saturation_range=(
                    0.4,
                    1.6,
                ),
                type='PhotoMetricDistortion'),
            dict(type='PackSegInputsWithSoft'),
        ],
        type='SBUDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='RefineAnnTransform'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(degree=15, prob=0.3, type='RandomRotate'),
    dict(
        brightness_delta=40,
        contrast_range=(
            0.4,
            1.6,
        ),
        hue_delta=20,
        saturation_range=(
            0.4,
            1.6,
        ),
        type='PhotoMetricDistortion'),
    dict(type='PackSegInputsWithSoft'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(reduce_zero_label=False, type='SBULabelTransform'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='SBU-Test/ShadowImages',
            seg_map_path='SBU-Test/ShadowMasks'),
        data_root='data/SBU-shadow',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='SBULabelTransform'),
            dict(type='PackSegInputs'),
        ],
        type='SBUDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='BERMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/pen_main_soft0285_bw4_tv025_b075_from500_lr8e6_5090_2k'
