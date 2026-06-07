# MMSegmentation 配置: IC-SSM + Shadow Boundary Supervision (SBS)
# ==================================================================
#
# 续训自: work_dirs/shadow_icssm_sbu/best_BER_iter_40000.pth (BER=6.74%)
#
# 新增创新点 (SBS - Shadow Boundary Supervision):
#   - ShadowBoundaryModule: 扩张卷积边界预测头
#   - 边界 GT 自动生成: dilate(mask) - erode(mask) → 无需额外标注
#   - 边界 BCE loss (pos_weight 自适应平衡正负样本)
#   - 边界门控增强: 边界置信度图反向加权主流特征
#
# 动机: 当前 FNR=12%, 主要来自阴影边界区域漏检
#       SBS 显式监督边界, 与 IC-SSM 全局光照对比建模互补
#
# 目标: BER < 5% (从 6.74% 改善)
# ==================================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# 从 IC-SSM 最佳 checkpoint 续训
load_from = 'work_dirs/shadow_icssm_sbu/best_BER_iter_40000.pth'

# ==================================================================
# 模型: IC-SSM + SBS
# ==================================================================

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
        # IC-SSM 参数 (不变)
        ssm_d_state=16,
        ssm_ratio=1.0,
        ssm_drop_path=0.05,
        spe_loss_weight=0.3,
        # SBS 参数 (新增)
        boundary_loss_weight=0.4,
        boundary_kernel=5,
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

# ==================================================================
# 数据: SBU 单数据集 (干净 baseline 对比)
# ==================================================================

train_pipeline = [
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

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline),
)

# ==================================================================
# 训练: 40k iters 精调
# ==================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=4000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==================================================================
# 优化器: SBS 新参数用较高 LR, backbone 低 LR 保护
# ==================================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW',
        lr=0.00005,          # 比初训 0.0001 低 2x, 精调模式
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # Backbone 保护 (已充分训练)
            'backbone.fusion_modules': dict(lr_mult=0.2),
            'backbone.local_stream':   dict(lr_mult=0.2),
            'backbone.global_stream':  dict(lr_mult=0.2),
            # IC-SSM 先验头: 维持
            'decode_head.ic_ssm.prior_head': dict(lr_mult=1.0),
            # SBS 边界模块: 新初始化, 较高 LR 快速收敛
            'decode_head.boundary_module':   dict(lr_mult=2.0),
            # 辅助头
            'auxiliary_head': dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.2,
         by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=False,
         begin=500, end=40000, eta_min=1e-7),
]

# ==================================================================
# Hook
# ==================================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=40000,
        log_stage_switch=True,
    ),
]

# ==================================================================
# 日志
# ==================================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_icssm_boundary_sbu'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
        save_best='BER',
        rule='less',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
