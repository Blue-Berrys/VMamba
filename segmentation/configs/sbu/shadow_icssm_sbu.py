# MMSegmentation 配置: IC-SSM Shadow Detection Head
# ===========================================================
#
# 论文创新:
#   IC-SSM (Illumination Contrast State Space Module)
#   - 首个 SSM-based shadow detection decoder
#   - 物理驱动: 阴影 = 局部光照衰减, 用 SSM 建模光照对比的方向性传播
#   - 先验监督: shadow prior BCE loss 快速定位阴影区域
#
# 起点: work_dirs/shadow_focal_resume_50k/iter_44000.pth (BER=7.76%)
#   - Backbone 权重: 完全继承 (44k iter 积累, 不丢弃)
#   - ICShadowHead 权重: 随机初始化
#   - FPN laterals 与 UPerHead 不共享权重, 需要重新学习
#
# 目标: BER < 5% (Phase 1 验证), 最终 < 3% (SOTA)
#
# 数据: SBU-shadow (4089 train, 638 test)
#       后续扩展: ISTD (1330 train) 混合预训练
#
# 日期: 2026-02-26
# ===========================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# 从最好的 baseline checkpoint 继承 backbone 权重
load_from = 'work_dirs/shadow_focal_resume_50k/iter_44000.pth'

# ===========================================================
# 模型配置
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

    # Backbone: 与 baseline 相同的 MM_ShadowDualStream
    # 完全继承 iter_44000.pth 的权重, 只更新 backbone LR
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

    # Decode Head: IC-SSM (核心创新)
    decode_head=dict(
        _delete_=True,
        type='ICShadowHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        # IC-SSM 参数
        ssm_d_state=16,
        ssm_ratio=1.0,
        ssm_drop_path=0.1,
        spe_loss_weight=0.3,      # 先验 BCE loss 权重 (高于 SPE v2 的 0.2, 因为 IC-SSM 更依赖先验)
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.88,       # 比 baseline 0.85 略高, 进一步抑制 FNR
                loss_weight=0.7,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                loss_weight=0.3,
            ),
        ],
    ),

    # 辅助头: FCNHead on C3 (512 channels)
    # 权重随机初始化 (与 ICShadowHead 不共享)
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

# ===========================================================
# 训练配置: 50k iters (比原 baseline 多 6k, 给新 head 充分收敛)
# ===========================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=50000,
    val_interval=4000,
)

val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===========================================================
# 优化器策略
# 核心思路:
#   - Backbone: LR × 0.3 (已收敛, 保护)
#   - ICShadowHead FPN laterals + IC-SSM: 全 LR (随机初始化)
#   - IC-SSM prior head: LR × 2.0 (需要快速收敛)
#   - 辅助头: LR × 0.5
# ===========================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=2,        # 等效 batch_size=4, 节省显存
    optimizer=dict(
        type='AdamW',             # AdamW 对新初始化的 decoder 收敛更稳定
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # Backbone: 已收敛, 低 LR
            'backbone.fusion_modules': dict(lr_mult=0.3),
            'backbone.local_stream':   dict(lr_mult=0.3),
            'backbone.global_stream':  dict(lr_mult=0.3),
            # IC-SSM 先验头: 快速学习
            'decode_head.ic_ssm.prior_head': dict(lr_mult=2.0),
            # 辅助头: 中等 LR
            'auxiliary_head':          dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000,                 # 1k iter 热身
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=1000,
        end=50000,
        eta_min=1e-6,
    ),
]

# ===========================================================
# Hook: 跳过 Stage 0/1, 直接全参数训练
# ===========================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=50000,
        log_stage_switch=True,
    ),
]

# ===========================================================
# 数据增强: 加入随机颜色抖动 (阴影检测重要)
# ===========================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SBULabelTransform', reduce_zero_label=False),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',    # 随机亮度/对比度扰动
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline),
)

# ===========================================================
# 日志与 Checkpoint
# ===========================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_icssm_sbu'

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
