# ================== SBU阴影检测数据集配置 ==================
# 数据集基本设置
dataset_type = 'SBUDataset'  # 数据集类型
data_root = 'data/SBU-shadow'  # 数据集根目录
crop_size = (416, 416)  # 训练时的裁剪尺寸 (高度, 宽度) - 与现有阴影检测工作保持一致

# 训练数据处理流程
# 参考 BDRAR: Resize(416) + Flip(0.5)
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='SBULabelTransform', reduce_zero_label=False),  # 加载并转换标注（255->1），保留标签0作为非阴影类
    dict(type='Resize', scale=(416, 416), keep_ratio=False),  # 直接 resize 到 416x416，不保持宽高比（参考 BDRAR）
    dict(type='RandomFlip', prob=0.5),  # 50%概率随机水平翻转
    dict(type='PackSegInputs')  # 打包输入数据
]

# 测试/验证数据处理流程
# 注意：SBULabelTransform 放在 Resize 之后，这样标签保持原始尺寸，与模型预测一致
# 模型在验证时会自动将预测 resize 回原始图像尺寸，所以标签也应该保持原始尺寸
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='Resize', scale=(416, 416), keep_ratio=False),  # 只 resize 图像到 416x416（标签还未加载）
    dict(type='SBULabelTransform', reduce_zero_label=False),  # 在 Resize 之后加载并转换标注（255->1），标签保持原始尺寸
    dict(type='PackSegInputs')  # 打包输入数据
]

# 测试时增强（TTA）的多尺度设置
img_ratios = [0.75, 1.0, 1.25]  # 多尺度测试的缩放比例

# TTA 数据处理流程
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),  # 加载图像
    dict(
        type='TestTimeAug',  # 测试时增强
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)  # 多尺度resize
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),  # 不翻转
                dict(type='RandomFlip', prob=1., direction='horizontal')  # 水平翻转
            ],
            [dict(type='SBULabelTransform', reduce_zero_label=False)],  # 加载并转换标注（255->1）
            [dict(type='PackSegInputs')]
        ])
]

# ================== 数据加载器配置 ==================
# 训练数据加载器
train_dataloader = dict(
    batch_size=4,  # 每个GPU的batch size
    num_workers=4,  # 数据加载的工作进程数
    persistent_workers=True,  # 保持数据加载进程活跃，提高效率
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 无限采样器，训练时打乱数据
    dataset=dict(
        type=dataset_type,  # 数据集类型
        data_root=data_root,  # 数据集根目录
        data_prefix=dict(
            img_path='SBUTrain4KRecoveredSmall/ShadowImages',  # 训练图像路径
            seg_map_path='SBUTrain4KRecoveredSmall/ShadowMasks'),  # 训练标注路径
        pipeline=train_pipeline))  # 数据处理流程

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,  # 验证时batch size为1
    num_workers=4,  # 数据加载的工作进程数
    persistent_workers=True,  # 保持数据加载进程活跃
    sampler=dict(type='DefaultSampler', shuffle=False),  # 默认采样器，不打乱数据
    dataset=dict(
        type=dataset_type,  # 数据集类型
        data_root=data_root,  # 数据集根目录
        data_prefix=dict(
            img_path='SBU-Test/ShadowImages',  # 测试图像路径
            seg_map_path='SBU-Test/ShadowMasks'),  # 测试标注路径
        pipeline=test_pipeline))  # 数据处理流程

# 测试数据加载器（与验证数据加载器相同）
test_dataloader = val_dataloader

# ================== 评估指标配置 ==================
# 使用自定义的BER评估指标
val_evaluator = dict(type='BERMetric')
test_evaluator = val_evaluator
