# MMSegmentation 配置: IC-SSM + SBS + SASF 完整架构 (Plan B)
# ============================================================
#
# 策略: 从头训练完整三模块架构
#   - 起点: shadow_focal_resume_50k/iter_44000.pth (backbone 已充分预热,
#           decoder 全部随机初始化, 相当于完整架构从头联合训练)
#   - 分辨率: 512×512 (比 416×416 提升 51% 像素, 有助于小阴影和边界)
#   - 80k iters: 给三个模块充分联合优化的时间
#   - batch: 1 sample × accumulate 4 = effective batch 4 (适配 16GB VRAM)
#
# 三个创新模块:
#   1. SASF (Scale-Aware Shadow Fusion): FPN 尺度自适应加权融合
#   2. IC-SSM (Illumination Contrast SSM): 全局光照对比 SSM 建模
#   3. SBS (Shadow Boundary Supervision): 边界显式监督
#
# 目标: BER < 5%
# ============================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# Backbone 起点: 早期阴影训练 checkpoint (backbone 预热但 decoder 未与新模块联合优化)
# decoder 中 IC-SSM/SBS/SASF 的权重在 checkpoint 中不存在 → 自动随机初始化
load_from = 'work_dirs/shadow_focal_resume_50k/iter_44000.pth'

# ============================================================
# 模型: 完整 SASF + IC-SSM + SBS 架构
# ============================================================

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),          # 高分辨率: 比 416 提升 51% 像素
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
        # SASF (尺度感知融合)
        use_sasf=True,
        # IC-SSM 参数
        ssm_d_state=16,
        ssm_ratio=1.0,
        ssm_drop_path=0.1,
        spe_loss_weight=0.3,
        # SBS 参数
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

# ============================================================
# 数据: SBU, 512×512
# ============================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SBULabelTransform', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
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
    batch_size=1,          # 512×512 显存适配
    num_workers=4,
    dataset=dict(pipeline=train_pipeline),
)

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
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
            dict(type='SBULabelTransform', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ],
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='BERMetric')
test_evaluator = val_evaluator

# ============================================================
# 训练: 80k iters 完整联合训练
# ============================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_interval=5000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================
# 优化器: 完整训练 LR, backbone 适度保护
# ============================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=4,      # effective batch = 4, 稳定梯度
    optimizer=dict(
        type='AdamW',
        lr=0.0001,              # 完整训练 LR
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # Backbone: 已在阴影数据上预热, 低 LR 保护
            'backbone.fusion_modules': dict(lr_mult=0.3),
            'backbone.local_stream':   dict(lr_mult=0.3),
            'backbone.global_stream':  dict(lr_mult=0.3),
            # SASF: 全新模块, 全 LR
            'decode_head.sasf':              dict(lr_mult=1.0),
            # IC-SSM 先验头: 需要快速收敛
            'decode_head.ic_ssm.prior_head': dict(lr_mult=1.5),
            # SBS 边界模块: 全新模块
            'decode_head.boundary_module':   dict(lr_mult=1.5),
            # 辅助头
            'auxiliary_head': dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1,
         by_epoch=False, begin=0, end=1500),   # 1.5k warmup
    dict(type='CosineAnnealingLR', by_epoch=False,
         begin=1500, end=80000, eta_min=1e-7),
]

# ============================================================
# Hook
# ============================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=80000,
        log_stage_switch=True,
    ),
]

# ============================================================
# 日志
# ============================================================

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_icssm_full_512'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=3,
        save_best='BER',
        rule='less',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
