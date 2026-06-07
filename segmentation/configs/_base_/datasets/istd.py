# ISTD 阴影检测数据集基础配置
# mask: 0=非阴影, 255=阴影 → ISTDLabelTransform 转为 0/1

dataset_type = 'ISTDDataset'
data_root = 'data/ISTD_Dataset'

istd_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ISTDLabelTransform'),          # 255→1, 0→0
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='PackSegInputs'),
]

istd_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='ISTDLabelTransform'),
    dict(type='PackSegInputs'),
]
