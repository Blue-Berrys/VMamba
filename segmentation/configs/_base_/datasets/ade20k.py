# ================== ADE20K 数据集配置 ==================
# 数据集基本设置
dataset_type = 'ADE20KDataset'  # 数据集类型
data_root = 'data/ade/ADEChallengeData2016'  # 数据集根目录
crop_size = (512, 512)  # 训练时的裁剪尺寸 (高度, 宽度)

# 训练数据处理流程
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='LoadAnnotations', reduce_zero_label=True),  # 加载标注，将标签0视为背景
    dict(
        type='RandomResize',  # 随机调整图像大小
        scale=(2048, 512),  # 目标尺寸 (宽度, 高度)
        ratio_range=(0.5, 2.0),  # 缩放比例范围 [0.5x, 2.0x]
        keep_ratio=True),  # 保持宽高比
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪到指定尺寸，单个类别最大占比75%
    dict(type='RandomFlip', prob=0.5),  # 50%概率随机水平翻转
    dict(type='PhotoMetricDistortion'),  # 光度失真增强（亮度、对比度、饱和度、色调）
    dict(type='PackSegInputs')  # 打包输入数据
]
# 测试/验证数据处理流程
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),  # 调整到固定尺寸，保持宽高比
    # 在Resize之后加载标注，因为真实标签不需要做resize变换
    dict(type='LoadAnnotations', reduce_zero_label=True),  # 加载标注
    dict(type='PackSegInputs')  # 打包输入数据
]

# 测试时增强（TTA）的多尺度设置
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # 多尺度测试的缩放比例

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
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
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
            img_path='images/training',  # 训练图像路径
            seg_map_path='annotations/training'),  # 训练标注路径
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
            img_path='images/validation',  # 验证图像路径
            seg_map_path='annotations/validation'),  # 验证标注路径
        pipeline=test_pipeline))  # 数据处理流程

# 测试数据加载器（与验证数据加载器相同）
test_dataloader = val_dataloader

# ================== 评估指标配置 ==================
# 验证评估器：使用 IoU（交并比）指标
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # 计算平均交并比 (mIoU)
test_evaluator = val_evaluator  # 测试评估器与验证评估器相同
