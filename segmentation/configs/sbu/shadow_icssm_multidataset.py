# MMSegmentation 配置: IC-SSM + 多数据集训练 (SBU + ISTD)
# ===========================================================
#
# 接续: work_dirs/shadow_icssm_sbu/best_BER_iter_40000.pth (BER=6.74%)
#
# 改进:
#   1. 混合 ISTD (1330) + SBU (4089) 训练 → 更丰富的阴影场景
#   2. 从最佳 IC-SSM checkpoint 续训, backbone+head 权重都保留
#   3. 30k iters 精调, 低 LR 保护已学到的 IC-SSM 表示
#   4. 加强数据增强: RandomRotate + 更强 PhotoMetricDistortion
#
# 目标: BER < 5% (ISTD 验证保持参考, SBU test 为主评估)
# ===========================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# 从 IC-SSM 最佳 checkpoint 续训 (backbone + ICShadowHead 权重全部保留)
load_from = 'work_dirs/shadow_icssm_sbu/best_BER_iter_40000.pth'

# ===========================================================
# 模型 (与 shadow_icssm_sbu.py 完全相同)
# ===========================================================

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(416, 416),
    ),

    backbone=dict(
        _delete_=True,
        type='MM_ShadowDualStream',
        depths=[2, 2, 27, 2],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.3,
        in_chans=3,
        out_indices=(0, 1, 2, 3),
        use_bidirectional_attn=True,
        use_shadow_map=True,
    ),

    decode_head=dict(
        _delete_=True,
        type='ICShadowHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        ssm_d_state=16,
        ssm_ratio=1.0,
        ssm_drop_path=0.05,       # 续训时降低 drop_path, 减少正则化
        spe_loss_weight=0.3,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss', use_sigmoid=True,
                 gamma=2.0, alpha=0.88, loss_weight=0.7),
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
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),

    test_cfg=dict(mode='whole'),
)

# ===========================================================
# 数据: SBU + ISTD 混合
# ===========================================================

# SBU 训练 pipeline (加强增强)
sbu_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SBULabelTransform', reduce_zero_label=False),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=15),
    dict(type='PhotoMetricDistortion',
         brightness_delta=40,
         contrast_range=(0.4, 1.6),
         saturation_range=(0.4, 1.6),
         hue_delta=20),
    dict(type='PackSegInputs'),
]

# ISTD 训练 pipeline (mask 格式与 SBU 相同: 0=非阴影, 255=阴影, 直接复用 SBULabelTransform)
istd_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SBULabelTransform', reduce_zero_label=False),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=15),
    dict(type='PhotoMetricDistortion',
         brightness_delta=40,
         contrast_range=(0.4, 1.6),
         saturation_range=(0.4, 1.6),
         hue_delta=20),
    dict(type='PackSegInputs'),
]

# 混合数据集: SBU (4089) + ISTD (1330), 合计 5419 张
train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='SBUDataset',
                data_root='data/SBU-shadow',
                data_prefix=dict(
                    img_path='SBUTrain4KRecoveredSmall/ShadowImages',
                    seg_map_path='SBUTrain4KRecoveredSmall/ShadowMasks'),
                pipeline=sbu_train_pipeline,
            ),
            dict(
                type='ISTDDataset',
                data_root='data/ISTD_binary',
                data_prefix=dict(
                    img_path='train/img',
                    seg_map_path='train/mask'),
                pipeline=istd_train_pipeline,
            ),
        ],
    ),
)

# 验证仍用 SBU test set (论文主评估集)
val_dataloader = dict(
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
            dict(type='Resize', scale=(416, 416), keep_ratio=False),
            dict(type='SBULabelTransform', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ],
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='BERMetric')
test_evaluator = val_evaluator

# ===========================================================
# 训练: 30k iters 精调
# ===========================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=30000,
    val_interval=3000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===========================================================
# 优化器: 低 LR 保护已学表示
# ===========================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW',
        lr=0.00003,           # 比初训 0.0001 低 3x, 精调模式
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.fusion_modules': dict(lr_mult=0.2),
            'backbone.local_stream':   dict(lr_mult=0.2),
            'backbone.global_stream':  dict(lr_mult=0.2),
            'decode_head.ic_ssm.prior_head': dict(lr_mult=1.5),
            'auxiliary_head':          dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.3,
         by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=False,
         begin=500, end=30000, eta_min=1e-7),
]

# ===========================================================
# Hook: 直接全参数训练
# ===========================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=30000,
        log_stage_switch=True,
    ),
]

# ===========================================================
# 日志
# ===========================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_icssm_multidataset'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=3000,
        max_keep_ckpts=3,
        save_best='BER',
        rule='less',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
