"""
双流VMamba-Base 阴影检测配置文件 - SBU数据集

创新点：
- 双流架构：原始RGB + Mean Subtraction图像
- 流1：Base配置的主干流（语义信息）
- 流2：轻量化的对比感知流（纹理信息）
- 门控融合：在每个VSS Block内部进行自适应特征融合

预期效果：
- 更好地区分真正的阴影和暗色物体
- BER降低2-5个百分点
- 对边界和复杂场景的鲁棒性提升
"""

_base_ = [
    '../_base_/datasets/sbu.py',                 # SBU数据集配置
    '../_base_/default_runtime.py',              # 默认运行时配置
    '../_base_/schedules/schedule_40k.py'        # 40k训练调度
]

# ================== 模型配置 ==================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet均值
        std=[58.395, 57.12, 57.375],      # ImageNet标准差
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    ),
    # Backbone: 双流VMamba-Base
    backbone=dict(
        type='MM_DualStreamVSSM',
        out_indices=(0, 1, 2, 3),  # 输出4个阶段的特征图

        # ==================== 通用配置 ====================
        depths=[2, 2, 27, 2],  # Base配置
        drop_path_rate=0.6,     # Base配置
        patch_size=4,
        in_chans=3,
        patch_norm=True,
        norm_layer='ln',        # LayerNorm
        downsample_version='v3',
        patchembed_version='v2',
        use_checkpoint=False,   # 如果GPU内存不足，设置为True
        posembed=False,

        # ==================== 流1配置（Base）====================
        dims_s1=[128, 256, 512, 1024],  # Base维度
        ssm_d_state_s1=16,
        ssm_ratio_s1=2.0,
        ssm_dt_rank_s1='auto',
        ssm_act_layer='silu',
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init='v0',
        forward_type='v2',

        # ==================== 流2配置（轻量化）====================
        dims_s2=[64, 128, 256, 512],  # 轻量化（约为流1的一半）
        ssm_d_state_s2=8,             # 状态维度减半
        ssm_ratio_s2=1.5,             # SSM扩展比例降低
        ssm_dt_rank_s2='auto',

        # ==================== 门控配置 ====================
        gate_type='channel_spatial',  # channel + spatial联合门控
        gate_ratio=0.25,              # 门控网络压缩比例

        # ==================== MLP配置 ====================
        mlp_ratio=0.0,  # VMamba默认不使用MLP（节省计算）
        mlp_act_layer='gelu',
        mlp_drop_rate=0.0,

        # ==================== 预训练权重 ====================
        pretrained=None,  # 如果有预训练权重，设置路径
        # pretrained='path/to/vssm_base_pretrained.pth'
    ),

    # Decode Head: UPerNet（修改输入通道）
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],  # Base输出通道数
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,              # 中间通道数
        dropout_ratio=0.1,
        num_classes=2,             # 二分类: 0=非阴影, 1=阴影
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),

    # Auxiliary Head
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,           # Stage3通道
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),

    # 训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ================== 运行时配置优化 ==================
# 可视化后端
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 工作目录
work_dir = './work_dirs/sbu_shadow_detection_dual_stream_base'

# ================== 训练配置 ==================
# 优化器
optimizer = dict(
    type='SGD',
    lr=0.01,                     # 初始学习率
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True                # 使用Nesterov动量
)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        # 对不同层设置不同的学习率衰减
        custom_keys={
            'pos_embed_s1': dict(decay_mult=0.),
            'pos_embed_s2': dict(decay_mult=0.),
            'patch_embed_s1': dict(lr_mult=0.1),
            'patch_embed_s2': dict(lr_mult=0.1),
        }
    ),
    _delete_=True,  # 删除基础配置中的clip_grad
    clip_grad=dict(max_norm=1, norm_type=2)  # 梯度裁剪
)

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,             # 40k iterations
        by_epoch=False
    )
]

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,           # 最大迭代次数
    val_interval=4000          # 每4000次迭代验证一次
)

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,          # 每4000次迭代保存一次
        max_keep_ckpts=5        # 最多保留5个checkpoint
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# ================== 数据加载器配置 ==================
# SBU数据集路径配置
# 实际路径: data/SBU-shadow/SBUTrain4KRecoveredSmall
# 通过符号链接: data/sbu -> data/SBU-shadow

# 训练数据加载器
train_dataloader = dict(
    batch_size=4,               # 每GPU的batch size（Base模型需要较大显存）
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='SBUDataset',
        data_root='data/sbu',  # 会解析为 data/SBU-shadow
        data_prefix=dict(
            img_path='img',           # 图像路径
            seg_map_path='label'      # 标注路径
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='RandomResize',
                scale=(640, 480),
                ratio_range=(0.5, 2.0),
                keep_ratio=True
            ),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]
    )
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SBUDataset',
        data_root='data/sbu',
        data_prefix=dict(
            img_path='img',
            seg_map_path='label'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 480), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]
    )
)

test_dataloader = val_dataloader

# 评估指标
val_evaluator = dict(type='BERMetric')
test_evaluator = val_evaluator

# ================== 使用说明 ==================
# 1. 准备数据集：
#    确保SBU数据集在 data/sbu/ 目录下
#    目录结构：data/sbu/img/ 和 data/sbu/label/
#
# 2. 训练命令（单GPU）：
#    python tools/train.py configs/sbu/sbu_shadow_dual_stream_base_40k.py
#
# 3. 训练命令（多GPU，推荐4-8卡）：
#    bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4
#
# 4. 评估命令：
#    bash tools/dist_test.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py \
#        work_dirs/sbu_shadow_detection_dual_stream_base/iter_40000.pth 4
#
# 5. 如果显存不足：
#    方案1：设置 use_checkpoint=True
#    方案2：减小 batch_size（4 → 2）
#    方案3：使用梯度累积
