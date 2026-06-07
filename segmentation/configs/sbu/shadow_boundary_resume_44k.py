# MMSegmentation配置文件: Phase 2 - Boundary Auxiliary Head
# ===================================================================
#
# 目标: BER 7.76% → <6.5%
# 核心改动:
#   1. 新增 ShadowBoundaryHead（边界辅助监督）
#      → 强迫 decoder 感知阴影边界，降低 FNR（12.55% → 目标 <10%）
#   2. 从 iter_44000.pth 继续训练（Run 4 最佳 BER=7.76%）
#   3. LR = 0.0004（保护已收敛权重），boundary_head LR × 2.0
#   4. 训练 30k iters
#
# 集成路径: /home/xjx/CodeProject/PycharmProject/VMamba/segmentation/configs/sbu/
# 日期: 2026-02-21

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# ===================================================================
# 加载预训练权重（Run 4 最佳 checkpoint）
# ===================================================================

load_from = 'work_dirs/shadow_focal_resume_50k/iter_44000.pth'

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
        # 保留 Run 4 的 loss 配置
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

    # 两个辅助头：原有 FCNHead + 新增 ShadowBoundaryHead
    auxiliary_head=[
        # 辅助头1：保留原 FCNHead（stage2特征，加权CE）
        dict(
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
                class_weight=[1.0, 3.0],
                loss_weight=0.4
            )
        ),
        # 辅助头2：新增边界监督头（stage1特征，256ch，更高分辨率）
        dict(
            type='ShadowBoundaryHead',
            in_channels=256,       # stage1 特征通道数
            in_index=1,            # 使用 stage1 高分辨率特征（52×52）
            channels=128,          # 内部通道数
            num_convs=2,
            concat_input=False,
            dropout_ratio=0.1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            boundary_width=5,      # 边界带宽度（像素）
            bce_pos_weight=10.0,   # 边界像素稀少，需高权重
            boundary_loss_weight=0.3,
            # loss_decode 传入但不实际使用（接口兼容）
            loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.3)
        ),
    ],

    test_cfg=dict(mode='whole')
)

# ===================================================================
# 训练配置：30k iters fine-tune
# ===================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=30000,
    val_interval=3000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===================================================================
# 优化器：低 LR 保护已收敛权重，boundary_head LR × 2.0
# ===================================================================

optim_wrapper = dict(
    _delete_=True,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        type='SGD',
        lr=0.0004,
        momentum=0.9,
        weight_decay=0.0005
    ),
    type='AmpOptimWrapper',
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            # 已收敛模块：低 LR 保护
            'fusion_modules': dict(lr_mult=0.3),
            'local_stream': dict(lr_mult=0.3),
            'global_stream': dict(lr_mult=0.3),
            # 新增边界头：高 LR 快速收敛（随机初始化）
            'auxiliary_head.1': dict(lr_mult=2.0),
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,        # 热身：0.0004 × 0.1 → 0.0004
        by_epoch=False,
        begin=0,
        end=300
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=300,
        end=30000,
        eta_min=1e-7
    )
]

# ===================================================================
# Hook：直接 Stage 2（全参数联合训练）
# ===================================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=30000,
        log_stage_switch=True
    )
]

# ===================================================================
# 日志与其他配置
# ===================================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_boundary_resume_44k'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=3000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# ===================================================================
# 说明
# ===================================================================
#
# 边界监督原理：
#   boundary_gt = dilate(shadow_mask) - erode(shadow_mask)  （5px宽边界带）
#   loss = BCE(boundary_pred, boundary_gt, pos_weight=10)
#   → 强迫模型在阴影边界处做精确预测 → 降低 FNR
#
# 参数量变化：
#   原 FCNHead aux: 512→256→2  ≈ 525K
#   新 BoundaryHead: 256→128→1 ≈ 444K
#   总增加: ~0.2%，可忽略
#
# 预期效果：
#   BER: 7.76% → ~6.5% (FNR: 12.55% → ~10%)
