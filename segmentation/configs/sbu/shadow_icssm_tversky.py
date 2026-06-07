# MMSegmentation 配置: IC-SSM + SASF + SBS + Tversky Loss (FNR 专项优化)
# ============================================================================
#
# 问题: FNR=7.46% >> FPR=0.97% (漏检是主要误差来源)
#
# 改进:
#   1. 新增 Tversky Loss (beta=0.7 > alpha=0.3): FN 受 2.3x 惩罚, 直接降 FNR
#   2. Focal loss alpha: 0.88 → 0.92 (更激进地加权 shadow 类)
#   3. Focal gamma: 2.0 → 2.5 (更聚焦难样本)
#   4. 从 best_BER_iter_45000.pth (BER=4.22%) 续训
#
# 数学原理 (Tversky):
#   T(α,β) = TP / (TP + α·FP + β·FN)
#   Loss = 1 - T(0.3, 0.7)
#   FN 权重 β=0.7 >> α=0.3 → 强制模型减少漏检
#
# 总 Loss = Focal(0.7) + Dice(0.3) + Tversky(0.5) + aux_CE(0.4) + SPE(0.3) + SBS(0.4)
# ============================================================================

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# 从 b2 最优 checkpoint 续训
load_from = 'work_dirs/shadow_icssm_pretrained_b2/best_BER_iter_45000.pth'

# ============================================================================
# 模型: VMamba-Base (pretrained) + ICShadowHead + Tversky Loss
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
        pretrained='',   # load_from 已含 backbone 权重
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
        # Tversky loss: beta=0.7 > alpha=0.3, 惩罚 FN 2.3x 于 FP
        tversky_loss_weight=0.5,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[
            # alpha: 0.88 → 0.92 (更激进加权 shadow 类)
            # gamma: 2.0 → 2.5 (更聚焦难样本, 多为漏检的低对比度阴影)
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
# 数据: SBU, 512×512, batch_size=2
# ============================================================================

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
    batch_size=2,
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

# ============================================================================
# 训练: 40k iters 精调 (总计约 85k)
# ============================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=5000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# 优化器: 进一步降低 LR, 精细调整
# ============================================================================

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    clip_grad=dict(max_norm=10, norm_type=2),
    accumulative_counts=4,    # effective batch = 2×4 = 8
    optimizer=dict(
        type='AdamW',
        lr=0.00003,           # 进一步降低 LR (精调阶段)
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':                      dict(lr_mult=0.1),
            'decode_head.sasf':              dict(lr_mult=1.0),
            'decode_head.ic_ssm':            dict(lr_mult=1.0),
            'decode_head.ic_ssm.prior_head': dict(lr_mult=1.5),
            'decode_head.boundary_module':   dict(lr_mult=1.5),
            'auxiliary_head':                dict(lr_mult=0.5),
        },
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.3,
         by_epoch=False, begin=0, end=300),
    dict(type='CosineAnnealingLR', by_epoch=False,
         begin=300, end=40000, eta_min=1e-7),
]

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=40000,
        log_stage_switch=True,
    ),
]

log_processor = dict(by_epoch=False)

work_dir = './work_dirs/shadow_icssm_tversky'

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
