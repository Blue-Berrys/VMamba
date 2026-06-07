# 消融实验 V2 方案A：所有配置从 ImageNet 预训练权重开始
# Row 1: IC-SSM only (no BG-SIR, no Tversky)
# Focal(0.7) + Dice(0.3) only
_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# 从 ImageNet 预训练权重加载（只加载 backbone）
# 不使用任何 shadow 检测的预训练权重，确保公平对比
backbone_pretrained = 'pretrained/vssm_base_0229_ckpt_epoch_237.pth'

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
    ),
    backbone=dict(
        _delete_=True,
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=backbone_pretrained,  # ImageNet 预训练
        dims=128,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank='auto',
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.3,
        norm_layer='ln2d',
    ),
    decode_head=dict(
        _delete_=True,
        type='ICShadowHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        use_sasf=True,
        ssm_d_state=16,
        ssm_ratio=1.0,
        ssm_drop_path=0.05,
        spe_loss_weight=0.3,
        boundary_loss_weight=0.4,
        boundary_kernel=5,
        tversky_loss_weight=0.0,   # Tversky 关闭
        tversky_alpha=0.5,
        tversky_beta=0.5,
        use_bg_sir=False,          # BG-SIR 关闭
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss', use_sigmoid=True,
                 gamma=2.5, alpha=0.92, loss_weight=0.7),
            dict(type='DiceLoss', use_sigmoid=True, loss_weight=0.3),
        ],
    ),
    auxiliary_head=dict(
        _delete_=True,
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RefineAnnTransform', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=15),
    dict(type='PhotoMetricDistortion',
         brightness_delta=40, contrast_range=(0.4, 1.6),
         saturation_range=(0.4, 1.6), hue_delta=20),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='SBUDataset',
        data_root='data/SBU-shadow',
        data_prefix=dict(
            img_path='SBUTrain4KRecoveredSmall/ShadowImages',
            seg_map_path='../SBU-Refine/refined_sbu_train/refined_sbu_train/ShadowMasks',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SBUDataset',
        data_root='data/SBU-shadow',
        data_prefix=dict(
            img_path='SBU-Test/ShadowImages',
            seg_map_path='SBU-Test/ShadowMasks'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='SBULabelTransform', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ],
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='BERMetric')
test_evaluator = val_evaluator

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
                 max_iters=40000, val_interval=5000)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=0.00006,
                   betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'decode_head': dict(lr_mult=1.0),
        'auxiliary_head': dict(lr_mult=0.5),
    }),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.3, by_epoch=False, begin=0, end=400),
    dict(type='CosineAnnealingLR', by_epoch=False, begin=400, end=40000, eta_min=1e-7),
]

custom_hooks = [dict(type='ProgressiveTrainingHook',
                     stage1_iters=0, stage2_iters=0,
                     stage3_iters=40000, log_stage_switch=True)]

log_processor = dict(by_epoch=False)
work_dir = './work_dirs/shadow_icssm_abl_v2_base'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000,
                    max_keep_ckpts=2, save_best='BER', rule='less'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
