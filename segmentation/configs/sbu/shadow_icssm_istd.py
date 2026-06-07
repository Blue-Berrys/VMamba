# MMSegmentation 配置: IC-SSM + BG-SIR — ISTD 专项训练
# ============================================================================
#
# 实验目标: 在 ISTD 数据集上做域内训练+测试，与顶刊指标对齐
#
# ISTD 数据集:
#   train: 1330 张 (室内, 受控场景)
#   test:  540  张
#   mask:  {0, 255} 二值图 (shadow=255, bg=0)
#
# 策略:
#   - 从 SBU-Refine 最优 checkpoint 迁移学习 (BER=2.81%, iter 25000)
#     → backbone 已学习通用阴影特征, 在 ISTD 上快速收敛
#   - 训练 30k iters (数据量 1330 < SBU 4089, 不需太长)
#   - 评测指标与顶刊一致: BER (全局聚合, 非 per-image 平均)
#
# 预期目标: ISTD BER < 1.5% (顶刊区间 1.0~1.8%)
# ============================================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',         # 继承基础结构, 数据部分全部 _delete_
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

load_from = 'work_dirs/shadow_icssm_refine/best_BER_iter_25000.pth'

# ============================================================================
# 模型: 与 SBU-Refine 完全相同 (迁移权重后域内微调)
# ============================================================================

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
        pretrained='',
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
        tversky_loss_weight=0.5,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        use_bg_sir=True,
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
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),

    test_cfg=dict(mode='whole'),
)

# ============================================================================
# 数据: ISTD train → ISTD test
# ============================================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SBULabelTransform', reduce_zero_label=False),  # {0,255}→{0,1}
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=15),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.6, 1.4),
         saturation_range=(0.6, 1.4),
         hue_delta=18),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ISTDDataset',
        data_root='data/ISTD_Dataset',
        data_prefix=dict(
            img_path='train/img',
            seg_map_path='train/mask'),
        pipeline=train_pipeline,
    ),
)

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='SBULabelTransform', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ISTDDataset',
        data_root='data/ISTD_Dataset',
        data_prefix=dict(
            img_path='test/img',
            seg_map_path='test/mask'),
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='BERMetric')
test_evaluator = val_evaluator

# ============================================================================
# 训练: 30k iters (数据量小，不需太长；迁移学习收敛快)
# ============================================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=30000,
    val_interval=3000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# 优化器: 迁移微调，backbone LR 更低
# ============================================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW',
        lr=0.00002,           # 迁移微调，LR 稍低
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':                          dict(lr_mult=0.05),  # 更保守
            'decode_head.sasf':                  dict(lr_mult=1.0),
            'decode_head.ic_ssm':                dict(lr_mult=1.0),
            'decode_head.ic_ssm.bg_sir':         dict(lr_mult=1.5),
            'decode_head.ic_ssm.prior_head':     dict(lr_mult=1.5),
            'decode_head.boundary_module':       dict(lr_mult=1.5),
            'auxiliary_head':                    dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.3,
         by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', by_epoch=False,
         begin=200, end=30000, eta_min=1e-7),
]

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=30000,
        log_stage_switch=True,
    ),
]

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_icssm_istd'

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
