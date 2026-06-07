# MMSegmentation 配置文件: Phase 1 v2 — ShadowDualPathHead v2
# ===================================================================
#
# 目标: BER 7.76% → <6% (Phase 1 验证方向)
#
# v1 问题 (BER 卡在 10-12%):
#   1. 乘法门控 prior≈0.5 → 两路信号被砍半
#   2. SPE 无直接监督, 收敛极慢
#
# v2 修复:
#   1. 残差门控: shadow_gate=0.5+0.5*p, light_gate=1.5-0.5*p
#      最差情况信号保留 0.5x, 不再归零
#   2. SPE 显式 BCE 监督 (spe_loss_weight=0.2)
#      直接用 GT mask 监督先验图, 4k iter 内快速收敛
#   3. Backbone lr_mult 0.3→0.5, 更快重新适应新 decoder
#
# 起点: work_dirs/shadow_focal_resume_50k/iter_44000.pth (BER=7.76%)
# Decoder 权重重新初始化 (新架构), Backbone 权重继承
#
# 日期: 2026-02-25
# ===================================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# 从 iter_12000 续训 (v2 Phase 1 第12000轮模型权重)
# 使用 load_from 加载模型权重, 优化器重新初始化
load_from = 'work_dirs/shadow_phase1_dualpath_v2/iter_12000.pth'

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
        size=(416, 416),
    ),

    backbone=dict(
        _delete_=True,
        type='MM_ShadowDualStream',
        depths=[2, 2, 27, 2],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.3,       # 降低 drop_path 节省激活值显存
        in_chans=3,
        out_indices=(0, 1, 2, 3),
        use_bidirectional_attn=True,
        use_shadow_map=True,
    ),

    # 用 ShadowDualPathHead 替换 UPerHead
    decode_head=dict(
        _delete_=True,
        type='ShadowDualPathHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        spe_mid_channels=32,
        cross_attn_reduction=8,
        spe_loss_weight=0.2,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.85,
                loss_weight=0.7,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                loss_weight=0.3,
            ),
        ],
    ),

    # 辅助头保持不变 (权重从 checkpoint 继承)
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
            class_weight=[1.0, 3.0],
            loss_weight=0.4,
        ),
    ),

    test_cfg=dict(mode='whole'),
)

# ===================================================================
# 训练配置: 40k iters
# ===================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=28000,   # 40000 - 12000 已完成
    val_interval=4000,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===================================================================
# 优化器
# 策略:
#   - Backbone/Fusion/LocalStream: LR × 0.5 (v1=0.3 过低，上调以加速重新适应)
#   - Decoder (ShadowDualPathHead): 全 LR (新初始化)
#   - AuxHead: LR × 0.5
#   - SPE: LR × 2.0 (需要快速学习先验图)
# ===================================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=2,   # 等效 batch_size×2，节省显存
    optimizer=dict(
        type='SGD',           # SGD 比 AdamW 节省 ~50% 优化器显存
        lr=0.0004,
        momentum=0.9,
        weight_decay=0.0005,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # 已收敛模块: 适度 LR (v1=0.3 过低，decoder 换新后 backbone 需要重新适应)
            'backbone.fusion_modules': dict(lr_mult=0.5),
            'backbone.local_stream':   dict(lr_mult=0.5),
            'backbone.global_stream':  dict(lr_mult=0.5),
            # 辅助头: 中等 LR
            'auxiliary_head':          dict(lr_mult=0.5),
            # SPE: 略高 LR (随机初始化, 需快速收敛)
            'decode_head.spe':         dict(lr_mult=2.0),
        },
    ),
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=0,
        end=28000,
        eta_min=1e-6,
    ),
]

# ===================================================================
# Hook: 直接 Stage 2 (全参数联合训练)
# ===================================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=28000,
        log_stage_switch=True,
    )
]

# ===================================================================
# 日志与 Checkpoint
# ===================================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_phase1_dualpath_v2'

# batch_size 从 4 降到 2 节省显存 (accumulative_counts=2 保持等效梯度)
train_dataloader = dict(batch_size=2, num_workers=4)

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
