# ================== 模型配置 - UPerNet + VMamba 二分类 ==================
# 用于阴影检测任务的二分类模型

# 归一化配置
norm_cfg = dict(type='SyncBN', requires_grad=True)

# 数据预处理配置
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet均值
    std=[58.395, 57.12, 57.375],      # ImageNet标准差
    bgr_to_rgb=True,                  # BGR转RGB
    pad_val=0,                         # 图像padding值
    seg_pad_val=255)                   # 分割图padding值

# 模型配置
model = dict(
    type='EncoderDecoder',            # 编码器-解码器架构
    data_preprocessor=data_preprocessor,

    # Backbone: VMamba
    backbone=dict(
        type='MM_VSSM',               # VMamba骨干网络
        out_indices=(0, 1, 2, 3),     # 输出4个阶段的特征图
        # VMamba-Tiny参数配置
        dims=96,                       # 基础通道数
        depths=(2, 2, 9, 2),          # 每个阶段的block数量
        ssm_d_state=16,               # 状态空间模型的状态维度
        ssm_dt_rank="auto",           # delta时间步的秩
        ssm_ratio=2.0,                # SSM扩展比例
        mlp_ratio=0.0,                # MLP扩展比例
        downsample_version="v1",      # 下采样版本
        patchembed_version="v1",      # patch embedding版本
    ),

    # Decode Head: UPerNet (修改为2分类)
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],  # VMamba-Tiny各阶段输出通道数
        in_index=[0, 1, 2, 3],            # 使用所有4个阶段的特征
        pool_scales=(1, 2, 3, 6),         # 金字塔池化尺度
        channels=512,                      # 中间通道数
        dropout_ratio=0.1,                 # dropout比例
        num_classes=2,                     # 二分类: 0=非阴影, 1=阴影
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,             # 使用softmax（多分类形式）
            loss_weight=1.0)),

    # 辅助Head: FCN Head (修改为2分类)
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,                   # 使用Stage3的特征
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,                     # 二分类
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),             # 辅助损失权重

    # 训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))           # 整图预测模式
