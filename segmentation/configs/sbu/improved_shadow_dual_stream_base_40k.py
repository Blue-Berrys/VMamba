# MMSegmentation配置文件: 改进的双流VMamba-Base (SBU数据集)
# ===================================================================
#
# 集成路径: /home/xjx/CodeProject/PycharmProject/VMamba/segmentation/configs/sbu/improved_shadow_dual_stream_base_40k.py
#
# 使用方法:
#   cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
#   bash tools/dist_train.sh configs/sbu/improved_shadow_dual_stream_base_40k.py 2
#
# 作者: AgentLaboratory
# 日期: 2026-02-19

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# ===================================================================
# 自定义导入
# ===================================================================

# model.py in segmentation already registers MM_ShadowDualStream and ProgressiveTrainingHook
# custom_imports are NOT needed and cause ModuleNotFoundError on the server

# ===================================================================
# 模型配置 - 继承 base 的 EncoderDecoder，只覆盖需要修改的部分
# ===================================================================

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=0,
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
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),

    auxiliary_head=dict(
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
            loss_weight=0.4
        )
    ),

    test_cfg=dict(mode='whole')
)

# 数据集配置完全继承自 _base_/datasets/sbu.py，无需在此覆盖

# ===================================================================
# 训练配置
# ===================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=50000,
    val_interval=4000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        type='SGD',
        lr=0.0025,
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
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        begin=1500,
        end=50000,
        eta_min=1e-6
    )
]

# ===================================================================
# 渐进式训练Hook配置
# ===================================================================

custom_hooks = [
    dict(
        type='ProgressiveTrainingHook',
        stage1_iters=20000,
        stage2_iters=10000,
        stage3_iters=20000,
        log_stage_switch=True
    )
]

# ===================================================================
# 日志与其他配置
# ===================================================================

log_processor = dict(
    by_epoch=False
)

work_dir = './work_dirs/improved_shadow_dual_stream_sbu_base'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# ===================================================================
# 注意事项
# ===================================================================
#
# 1. 确保以下文件已正确集成:
#    - classification/models/shadow_dual_stream_v2.py
#    - core/hooks/progressive_training_hook.py
#
# 2. 训练阶段说明:
#    - 阶段0 (0-20000):   只训练全局流 (VMamba) + 分割头；局部流被跳过（stage-aware routing）
#    - 阶段1 (20000-30000): 只训练局部流 (Conv+边缘) + 融合 + 分割头
#    - 阶段2 (30000-50000): 联合训练所有参数（fusion/local_stream LR=0.4×）
#
# 3. 显存不足时调整:
#    - batch_size: 4 -> 2
#    - accumulative_counts: 1 -> 2
