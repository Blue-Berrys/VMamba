crop_size = (
    416,
    416,
)
custom_hooks = [
    dict(
        log_stage_switch=True,
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=30000,
        type='ProgressiveTrainingHook'),
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
        interval=3000,
        max_keep_ckpts=3,
        rule='less',
        save_best='BER',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
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
istd_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ISTDLabelTransform'),
    dict(keep_ratio=False, scale=(
        416,
        416,
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
    dict(type='PackSegInputs'),
]
load_from = 'work_dirs/shadow_icssm_sbu/best_BER_iter_40000.pth'
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
            class_weight=[
                1.0,
                3.0,
            ],
            loss_weight=0.4,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        depths=[
            2,
            2,
            27,
            2,
        ],
        dims=[
            128,
            256,
            512,
            1024,
        ],
        drop_path_rate=0.3,
        in_chans=3,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        type='MM_ShadowDualStream',
        use_bidirectional_attn=True,
        use_shadow_map=True),
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
            416,
            416,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
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
            dict(
                alpha=0.88,
                gamma=2.0,
                loss_weight=0.7,
                type='FocalLoss',
                use_sigmoid=True),
            dict(loss_weight=0.3, type='DiceLoss', use_sigmoid=True),
        ],
        num_classes=2,
        spe_loss_weight=0.3,
        ssm_d_state=16,
        ssm_drop_path=0.05,
        ssm_ratio=1.0,
        type='ICShadowHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=2,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=3e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict({
            'auxiliary_head': dict(lr_mult=0.5),
            'backbone.fusion_modules': dict(lr_mult=0.2),
            'backbone.global_stream': dict(lr_mult=0.2),
            'backbone.local_stream': dict(lr_mult=0.2),
            'decode_head.ic_ssm.prior_head': dict(lr_mult=1.5)
        })),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.3, type='LinearLR'),
    dict(
        begin=500,
        by_epoch=False,
        end=30000,
        eta_min=1e-07,
        type='CosineAnnealingLR'),
]
resume = False
sbu_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='SBULabelTransform'),
    dict(keep_ratio=False, scale=(
        416,
        416,
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
    dict(type='PackSegInputs'),
]
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
                416,
                416,
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
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=3000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(
                    img_path='SBUTrain4KRecoveredSmall/ShadowImages',
                    seg_map_path='SBUTrain4KRecoveredSmall/ShadowMasks'),
                data_root='data/SBU-shadow',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(reduce_zero_label=False, type='SBULabelTransform'),
                    dict(keep_ratio=False, scale=(
                        416,
                        416,
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
                    dict(type='PackSegInputs'),
                ],
                type='SBUDataset'),
            dict(
                data_prefix=dict(
                    img_path='train/img', seg_map_path='train/mask'),
                data_root='data/ISTD_Dataset',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='ISTDLabelTransform'),
                    dict(keep_ratio=False, scale=(
                        416,
                        416,
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
                    dict(type='PackSegInputs'),
                ],
                type='ISTDDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='SBULabelTransform'),
    dict(keep_ratio=False, scale=(
        416,
        416,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
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
                416,
                416,
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
work_dir = './work_dirs/shadow_icssm_multidataset'
