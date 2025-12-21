# VMamba SBU阴影检测使用指南

本指南介绍如何使用VMamba在SBU数据集上进行像素级阴影检测。

## 目录结构

已创建的文件：
```
segmentation/
├── __init__.py                           # 自定义组件注册
├── sbu_dataset.py                        # SBU数据集类
├── ber_metric.py                         # BER评估指标
├── configs/
│   ├── _base_/
│   │   ├── datasets/sbu.py              # SBU数据集配置
│   │   └── models/upernet_vssm_binary.py # 二分类模型配置
│   └── sbu/
│       └── sbu_shadow_vssm_base_40k.py  # 完整训练配置 (VMamba-Base)
└── tools/
    ├── train.py                          # 训练脚本(已修改)
    └── test.py                           # 测试脚本(已修改)
```

## 数据集准备

### 1. 数据集目录结构

将SBU数据集放置在以下位置：
```
VMamba/
└── data/
    └── SBU-shadow/
        ├── SBUTrain4KRecoveredSmall/    # 训练集
        │   ├── ShadowImages/            # 训练图像 (.jpg)
        │   │   ├── lssd1.jpg
        │   │   ├── lssd2.jpg
        │   │   └── ...
        │   └── ShadowMasks/             # 训练标注 (.png)
        │       ├── lssd1.png
        │       ├── lssd2.png
        │       └── ...
        └── SBU-Test/                    # 测试集
            ├── ShadowImages/            # 测试图像 (.jpg)
            │   ├── IMG_8243.jpg
            │   └── ...
            └── ShadowMasks/             # 测试标注 (.png)
                ├── $_35.png
                └── ...
```

### 2. 标注格式要求

- 标注图像应为灰度图像
- 像素值: 0 = 非阴影(背景), 1或255 = 阴影
- 文件名应与对应的输入图像匹配

## 模型配置说明

### 关键修改点

#### 1. 二分类Segmentation Head
```python
# configs/_base_/models/upernet_vssm_binary.py
decode_head=dict(
    type='UPerHead',
    num_classes=2,  # 修改为2类: 非阴影/阴影
    ...
)
```

#### 2. SBU数据集配置
```python
# configs/_base_/datasets/sbu.py
dataset_type = 'SBUDataset'
data_root = 'data/SBU-shadow'
# 训练集
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='SBUTrain4KRecoveredSmall/ShadowImages',
            seg_map_path='SBUTrain4KRecoveredSmall/ShadowMasks'
        )
    )
)
# 测试集
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='SBU-Test/ShadowImages',
            seg_map_path='SBU-Test/ShadowMasks'
        )
    )
)
```

#### 3. BER评估指标
```python
# BER = (FPR + FNR) / 2 * 100
val_evaluator = dict(type='BERMetric')
```

## 训练

### 环境准备

确保已安装以下依赖：
```bash
pip install torch torchvision
pip install mmcv-full
pip install mmsegmentation
pip install mmengine
```

### 单GPU训练

```bash
cd segmentation
python tools/train.py configs/sbu/sbu_shadow_vssm_base_40k.py
```

### 多GPU训练 (推荐)

使用4个GPU训练：
```bash
cd segmentation
bash tools/dist_train.sh configs/sbu/sbu_shadow_vssm_base_40k.py 4
```

### 训练参数

默认配置：
- **模型**: VMamba-Base (depths=[2,2,27,2], dims=128)
- **Batch size**: 4 per GPU (4 GPUs = 16 total)
- **训练迭代**: 40k iterations
- **验证间隔**: 每4000次迭代
- **输入尺寸**: 416×416
- **数据增强**: Resize(416) + Flip(0.5) (参考 BDRAR)
- **学习率**: 0.01 (PolyLR调度)
- **优化器**: SGD (momentum=0.9, weight_decay=0.0005)

### 预训练权重

配置文件中指定的预训练权重路径（已注释，可根据需要取消注释）：
```python
backbone=dict(
    # pretrained="../../ckpts/classification/outs/vssm/vssmbasedp05/vssmbase_dp05_ckpt_epoch_260.pth"
)
```

确保预训练权重文件存在，或修改为正确的路径。如果不需要预训练，可以保持注释状态。

## 评估

### 单GPU测试

```bash
cd segmentation
python tools/test.py \
    configs/sbu/sbu_shadow_vssm_base_40k.py \
    work_dirs/sbu_shadow_detection_vssm_base/iter_40000.pth
```

### 多GPU测试

```bash
cd segmentation
bash tools/dist_test.sh \
    configs/sbu/sbu_shadow_vssm_base_40k.py \
    work_dirs/sbu_shadow_detection_vssm_base/iter_40000.pth \
    4
```

### 评估指标

训练和测试时会输出以下指标：
- **BER** (Balance Error Rate): 主要指标，越低越好
- FPR (False Positive Rate): 假阳性率
- FNR (False Negative Rate): 假阴性率
- Precision: 精确率
- Recall: 召回率
- F1: F1分数
- Accuracy: 准确率
- IoU: 交并比

## 可视化

### TensorBoard监控

训练过程中自动生成TensorBoard日志：
```bash
cd work_dirs/sbu_shadow_detection_vssm_base
tensorboard --logdir=./
```

在浏览器中打开 http://localhost:6006 查看：
- 训练曲线
- 学习率变化
- 损失函数
- 验证指标

### 预测结果可视化

使用test.py的可视化功能：
```bash
python tools/test.py \
    configs/sbu/sbu_shadow_vssm_base_40k.py \
    work_dirs/sbu_shadow_detection_vssm_base/iter_40000.pth \
    --show-dir ./vis_results
```

## 自定义配置

### 修改训练迭代次数

编辑配置文件 `configs/sbu/sbu_shadow_vssm_base_40k.py`:
```python
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,      # 修改为80k
    val_interval=8000     # 相应调整验证间隔
)

param_scheduler = [
    dict(
        type='PolyLR',
        end=80000,  # 同步修改
        ...
    )
]
```

### 修改Batch Size

```python
train_dataloader = dict(
    batch_size=8,  # 从4改为8
    ...
)
```

### 使用不同的VMamba模型

当前配置使用 **VMamba-Base**，如需切换到其他模型：

**VMamba-Tiny**:
```python
model = dict(
    backbone=dict(
        dims=96,                       # Tiny: 96
        depths=(2, 2, 9, 2),          # Tiny: 9 blocks in stage 3
        drop_path_rate=0.1,            # Tiny: 0.1
        ...
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],  # Tiny配置
        ...
    )
)
```

**VMamba-Small**:
```python
model = dict(
    backbone=dict(
        dims=128,                      # Small: 128
        depths=(2, 2, 27, 2),         # Small: 27 blocks in stage 3
        drop_path_rate=0.3,            # Small: 0.3
        ...
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],  # Small配置
        ...
    )
)
```

**VMamba-Base** (当前配置):
```python
model = dict(
    backbone=dict(
        dims=128,                      # Base: 128 (与Small相同)
        depths=(2, 2, 27, 2),         # Base: 27 blocks (与Small相同)
        drop_path_rate=0.6,            # Base: 0.6 (更大)
        ...
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],  # Base配置
        ...
    )
)
```

## 常见问题

### 1. 数据集路径错误

确保以下目录存在且包含图像文件：
- 训练集: `data/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages` 和 `ShadowMasks`
- 测试集: `data/SBU-shadow/SBU-Test/ShadowImages` 和 `ShadowMasks`

### 2. 预训练权重加载失败

检查预训练权重路径是否正确，或设置为None从头训练：
```python
backbone=dict(pretrained=None)
```

### 3. CUDA内存不足

降低batch size：
```bash
python tools/train.py configs/sbu/sbu_shadow_vssm_base_40k.py \
    --cfg-options train_dataloader.batch_size=2
```

### 4. 自定义组件注册失败

确保训练/测试脚本包含以下导入：
```python
from sbu_dataset import SBUDataset
from ber_metric import BERMetric
```

## 实现细节

### BER计算公式

```python
FPR = FP / (FP + TN)  # 假阳性率
FNR = FN / (FN + TP)  # 假阴性率
BER = (FPR + FNR) / 2 * 100
```

### 损失函数

使用标准的交叉熵损失：
```python
loss_decode=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,  # 使用softmax多分类形式
    loss_weight=1.0
)
```

### 数据增强

**当前配置使用简化的数据增强策略（参考 BDRAR）**：

训练时包含：
- **Resize (416×416)**: 固定尺寸resize，不保持宽高比
- **RandomFlip (50%)**: 随机水平翻转

**为什么使用简化的数据增强？**

阴影检测任务对数据增强有特殊要求，**不使用**以下增强：
- ❌ **RandomResize**: 改变图像scale会破坏阴影的几何特征
- ❌ **RandomCrop**: 随机裁剪可能切掉重要的阴影区域
- ❌ **PhotoMetricDistortion**: 改变光照条件会影响阴影检测（阴影依赖于光照信息）

**设计原则**：
1. **保持阴影几何特征**: 固定尺寸resize，避免改变阴影形状
2. **保持光照一致性**: 不使用光度失真，保持原始光照信息
3. **参考最佳实践**: BDRAR等经典阴影检测方法都使用相同策略
4. **固定输入尺寸**: 416×416是阴影检测任务的常用尺寸

**数据增强配置** (`configs/_base_/datasets/sbu.py`):
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SBULabelTransform', reduce_zero_label=False),  # 标签转换 (255→1)
    dict(type='Resize', scale=(416, 416), keep_ratio=False),  # 固定尺寸
    dict(type='RandomFlip', prob=0.5),                       # 水平翻转
    dict(type='PackSegInputs')
]
```

## 预期性能

基于VMamba-Base的预期结果：
- **BER**: < 8% (取决于数据集质量，Base模型通常优于Tiny)
- **IoU**: > 85%
- **训练时间**: ~6-8小时 (4x V100 GPUs, 40k iterations)
- **模型参数量**: ~90M (Base模型)

**性能对比** (参考):
| 模型 | 参数量 | BER | IoU | 训练时间 |
|------|--------|-----|-----|----------|
| VMamba-Tiny | ~28M | ~10% | ~80% | ~4-5小时 |
| VMamba-Base | ~90M | ~8% | ~85% | ~6-8小时 |

## 扩展建议

1. **测试时增强(TTA)**: 使用多尺度和翻转提升性能
2. **集成学习**: 训练多个模型并融合预测结果
3. **后处理**: 使用CRF或形态学操作优化边界
4. **损失函数**: 尝试Dice Loss或Focal Loss处理类别不平衡

## 参考资料

- VMamba论文: [链接]
- MMSegmentation文档: https://mmsegmentation.readthedocs.io/
- SBU数据集: [原始论文链接]
