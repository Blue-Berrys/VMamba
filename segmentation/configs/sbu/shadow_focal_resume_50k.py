# MMSegmentation配置文件: Phase 1 改进 - Focal+Dice Loss + Resume from iter_50000
# ===================================================================
#
# 目标: BER 7.92% → <5%
# 核心改动:
#   1. FocalLoss(gamma=2, alpha=0.85) + DiceLoss 替换 CrossEntropyLoss
#      → 解决 FNR=12.80% >> FPR=3.04% 的类别不平衡问题
#   2. 从 iter_50000.pth 继续训练（LR 完全重置）
#   3. 直接进入 Stage 2（全参数联合训练）
#   4. LR = 0.0008（低于原始 0.0025，保护已收敛权重）
#
# 集成路径: /home/xjx/CodeProject/PycharmProject/VMamba/segmentation/configs/sbu/
# 日志路径: work_dirs/shadow_focal_resume_50k/
#
# 日期: 2026-02-21

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# ===================================================================
# 加载预训练权重（从上一轮 50k 训练的最优 checkpoint 继续）
# ===================================================================

load_from = 'work_dirs/improved_shadow_dual_stream_sbu_base/iter_50000.pth'

# ===================================================================
# 模型配置
# ===================================================================

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(416, 416)
    ),

    backbone=dict(
        _delete_=True,
        type='MM_ShadowDualStream',
        depths=[2, 2, 27, 2],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.6,
        in_chans=3,
        out_indices=(0, 1, 2, 3),
        use_bidirectional_attn=True,
        use_shadow_map=True
    ),

    decode_head=dict(
        _delete_=True,
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        # Focal + Dice: 同时解决 FNR >> FPR 问题
        # FocalLoss: alpha=0.85 给阴影类(少数类)更高权重，gamma=2聚焦漏检的难样本
        # DiceLoss: 几何上直接优化 IoU，对类别不平衡不敏感
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.85,
                loss_weight=0.7
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                loss_weight=0.3
            ),
        ]
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
        # 辅助头用加权 CE，给阴影类 3x 权重（shadow:non-shadow ≈ 25%:75%）
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, 3.0],
            loss_weight=0.4
        )
    ),

    test_cfg=dict(mode='whole')
)

# ===================================================================
# 训练配置：直接进入 Stage 2，50k iters
# ===================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=50000,
    val_interval=4000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===================================================================
# 优化器：LR 重置为 0.0008（低于原始 0.0025，保护已收敛权重）
# ===================================================================

optim_wrapper = dict(
    _delete_=True,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        type='SGD',
        lr=0.0008,
        momentum=0.9,
        weight_decay=0.0005
    ),
    type='AmpOptimWrapper',
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            'fusion_modules': dict(lr_mult=0.4),
            'local_stream': dict(lr_mult=0.4),
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,       # 从 0.0008*0.1=8e-5 热身，避免冷启动震荡
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=500,
        end=50000,
        eta_min=1e-7
    )
]

# ===================================================================
# Hook：直接进入 Stage 2（stage1_iters=0, stage2_iters=0）
# ===================================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=50000,
        log_stage_switch=True
    )
]

# ===================================================================
# 日志与其他配置
# ===================================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_focal_resume_50k'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# ===================================================================
# 说明
# ===================================================================
#
# 阶段对应：
#   iter 0-50000: Stage 2（全参数，LR 0.0008 → 1e-7 CosineAnnealing）
#
# 改动摘要 vs 上一轮训练：
#   - load_from: iter_50000.pth（接续已有50k训练权重）
#   - loss: CrossEntropyLoss → FocalLoss(γ=2,α=0.85) + DiceLoss
#   - aux loss: CrossEntropyLoss → weighted CE (shadow weight=3x)
#   - LR: 0.0025 → 0.0008（保护已收敛权重）
#   - Stage: 直接 Stage 2，跳过 Stage 0/1
#   - 数据增强: 待文献综述报告确认后再决定是否添加
