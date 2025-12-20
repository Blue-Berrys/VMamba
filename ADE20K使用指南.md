# VMamba ADE20K 数据集使用指南

## 目录
1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [数据集准备](#数据集准备)
4. [项目结构](#项目结构)
5. [训练](#训练)
6. [测试与评估](#测试与评估)
7. [配置文件说明](#配置文件说明)
8. [常见问题](#常见问题)

---

## 项目概述

VMamba 是一个基于状态空间模型的视觉骨干网络，支持多种视觉任务，包括：
- **分类** (Classification)
- **检测** (Detection)
- **分割** (Segmentation)

本指南主要介绍如何在 ADE20K 数据集上使用 VMamba 进行语义分割任务。

### ADE20K 数据集简介
- **类别数**: 150 个语义类别
- **训练集**: 约 20,000 张图像
- **验证集**: 约 2,000 张图像
- **图像尺寸**: 可变，通常较大（如 2048x512）

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n vmamba python=3.10
conda activate vmamba
```

### 2. 安装基础依赖

**重要**: 必须按照以下顺序安装，因为 `selective_scan` 的 `setup.py` 需要先导入 `torch`。

```bash
# 步骤 1: 先安装 PyTorch (必须使用 PyTorch 2.2，不要使用 2.9+)
# 这是关键步骤，必须最先安装！
# 注意：PyTorch 2.9+ 与 mmengine 0.10.1 存在兼容性问题，会导致 Adafactor 重复注册错误
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# 如果使用 CUDA，可以指定 CUDA 版本（例如 CUDA 11.8）
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 步骤 2: 安装其他基础依赖（这些依赖可能被 selective_scan 需要）
pip install packaging ninja einops

# 步骤 3: 安装项目其他依赖
pip install -r requirements.txt

# 步骤 4: 安装核心模块 selective_scan
# 注意：使用 --no-build-isolation 避免构建环境隔离问题
cd kernels/selective_scan && pip install . --no-build-isolation
cd ../..
```

**如果遇到 "ModuleNotFoundError: No module named 'torch'" 错误**：
- 确保已经先安装了 `torch`
- 检查是否正确激活了 conda 环境
- 尝试重新安装：`pip install torch==2.2 torchvision torchaudio`，然后再安装 `selective_scan`

### 3. 安装分割任务相关依赖

**重要**: mmcv 需要安装带有 CUDA 扩展的版本，不能只安装纯 Python 版本。

```bash
# 方法 1: 安装预编译的 mmcv（推荐，根据你的 CUDA 和 PyTorch 版本选择）

# 对于 CUDA 12.x + PyTorch 2.9+
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.9/index.html

# 如果上面的链接不工作，尝试通用安装（会自动选择合适版本）
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.0/index.html

# 方法 2: 如果预编译版本不可用，从源码编译（较慢）
# pip uninstall mmcv -y
# pip install mmcv==2.1.0 --no-cache-dir

# 安装其他依赖
pip install mmengine==0.10.1 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

**验证 mmcv 安装**:
```bash
python -c "from mmcv.ops import point_sample; print('mmcv 扩展模块安装成功')"
```

### 4. 验证安装

确保以下模块可以正常导入：
```bash
python -c "import torch; import mmseg; print('安装成功')"
```

---

## 数据集准备

### 1. 下载 ADE20K 数据集

**方法 1: 从官网下载（推荐）**

1. 访问 [ADE20K 官网](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
2. 下载以下文件：
   - `ADE20K_2016_07_26.zip` - 完整数据集（约 3.5GB）
   - 或分别下载训练集和验证集

**方法 2: 使用 wget 下载**

```bash
# 创建数据目录
cd /home/xjx/CodeProject/PycharmProject/VMamba
mkdir -p data/ade
cd data/ade

# 下载数据集（需要注册获取下载链接）
# wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
# unzip ADEChallengeData2016.zip
```

**方法 3: 手动下载并解压**

1. 从官网下载 `ADEChallengeData2016.zip`
2. 解压到项目根目录下的 `data/ade/` 目录

```bash
cd /home/xjx/CodeProject/PycharmProject/VMamba
mkdir -p data/ade
# 将下载的 zip 文件放到 data/ade/ 目录
cd data/ade
unzip ADEChallengeData2016.zip
```

### 2. 数据集目录结构

**重要**: 数据集必须放在项目根目录下的 `data/ade/` 目录中。

将数据集组织成以下结构：

```
VMamba/                          # 项目根目录
├── data/
│   └── ade/
│       └── ADEChallengeData2016/
│           ├── images/
│           │   ├── training/          # 训练图像（约 20,210 张）
│           │   │   ├── ADE_train_00000001.jpg
│           │   │   ├── ADE_train_00000002.jpg
│           │   │   └── ...
│           │   └── validation/        # 验证图像（约 2,000 张）
│           │       ├── ADE_val_00000001.jpg
│           │       ├── ADE_val_00000002.jpg
│           │       └── ...
│           └── annotations/
│               ├── training/          # 训练标注（分割掩码）
│               │   ├── ADE_train_00000001.png
│               │   ├── ADE_train_00000002.png
│               │   └── ...
│               └── validation/        # 验证标注
│                   ├── ADE_val_00000001.png
│                   ├── ADE_val_00000002.png
│                   └── ...
```

**验证数据集路径**：

```bash
# 从项目根目录检查
cd /home/xjx/CodeProject/PycharmProject/VMamba
ls -la data/ade/ADEChallengeData2016/images/training/ | head -5
ls -la data/ade/ADEChallengeData2016/images/validation/ | head -5

# 如果从 segmentation/ 目录运行训练，需要创建符号链接
cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
ln -sf ../data data
ls -la data/ade/ADEChallengeData2016/images/training/ | head -5
```

**重要提示**：
- 数据集路径：相对于项目根目录的 `data/ade/ADEChallengeData2016`
- 图像文件格式：`.jpg` 或 `.png`
- 标注文件格式：`.png`（单通道，像素值对应类别ID）
- 标注文件需要与对应的图像文件同名（仅扩展名不同）
- 训练集约 20,210 张图像，验证集约 2,000 张图像

### 3. 验证数据集

检查数据集是否正确组织：

```bash
# 检查训练集图像数量
ls data/ade/ADEChallengeData2016/images/training | wc -l

# 检查训练集标注数量
ls data/ade/ADEChallengeData2016/annotations/training | wc -l

# 数量应该一致
```

---

## 项目结构

```
VMamba/
├── segmentation/                 # 分割任务主目录
│   ├── configs/                  # 配置文件目录
│   │   ├── _base_/              # 基础配置
│   │   │   ├── datasets/        # 数据集配置
│   │   │   │   └── ade20k.py   # ADE20K 数据集配置
│   │   │   ├── models/          # 模型配置
│   │   │   ├── schedules/       # 训练调度配置
│   │   │   └── default_runtime.py
│   │   ├── vssm/                # VMamba 模型配置
│   │   │   ├── upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
│   │   │   ├── upernet_vssm_4xb4-160k_ade20k-512x512_small.py
│   │   │   └── upernet_vssm_4xb4-160k_ade20k-512x512_base.py
│   │   └── ...
│   ├── tools/                    # 工具脚本
│   │   ├── train.py             # 训练脚本
│   │   ├── test.py              # 测试脚本
│   │   ├── dist_train.sh        # 分布式训练脚本
│   │   └── dist_test.sh         # 分布式测试脚本
│   └── model.py                 # 模型定义
├── kernels/                      # 核心模块
│   └── selective_scan/          # 选择性扫描实现
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
```

---

## 训练

### 1. 单 GPU 训练

**重要**: 必须设置 `PYTHONPATH` 或使用 `dist_train.sh` 脚本（即使只有 1 个 GPU），因为 `train.py` 需要导入 `segmentation/model.py`。

#### 方法 1: 使用 dist_train.sh（推荐，即使单 GPU）

```bash
cd segmentation

# 使用 1 个 GPU 训练 Tiny 模型
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    1 \
    --work-dir work_dirs/ade20k_tiny

# 使用 1 个 GPU 训练 Small 模型
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    1 \
    --work-dir work_dirs/ade20k_small

# 使用 1 个 GPU 训练 Base 模型
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py \
    1 \
    --work-dir work_dirs/ade20k_base
```

#### 方法 2: 手动设置 PYTHONPATH

```bash
cd segmentation

# 设置 PYTHONPATH 并运行
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/train.py \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    --work-dir work_dirs/ade20k_tiny
```

### 2. 多 GPU 分布式训练

```bash
cd segmentation

# 使用 4 个 GPU 训练
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    4

# 使用 8 个 GPU 训练
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py \
    8
```

### 3. 训练参数说明

- **配置文件**: 指定模型和训练配置
- **--work-dir**: 工作目录，用于保存日志和模型检查点
- **--resume**: 从最新的检查点恢复训练
- **--amp**: 启用自动混合精度训练（可加速训练并节省显存）

### 4. 训练输出和检查点保存

训练过程中会在 `work_dir` 目录下生成：
- `work_dirs/ade20k_xxx/`
  - `*.log`: 训练日志
  - `*.pth`: 模型检查点（按迭代次数保存）
  - `config.py`: 保存的训练配置
  - `时间戳目录/`: 每次运行会创建一个时间戳目录

#### 检查点保存策略

根据配置文件 `schedule_160k.py`，检查点保存设置如下：

- **保存间隔**: 每 **16000 次迭代** 保存一次检查点
- **总迭代次数**: 160000 次
- **保存次数**: 共保存 **10 个检查点**（16000, 32000, 48000, ..., 160000）
- **文件命名**: `iter_16000.pth`, `iter_32000.pth`, ..., `iter_160000.pth`
- **保存位置**: `work_dirs/ade20k_tiny/时间戳目录/iter_XXXXX.pth`

**示例**：
```bash
work_dirs/ade20k_tiny/20251218_220455/
├── iter_16000.pth    # 第 16000 次迭代的检查点
├── iter_32000.pth    # 第 32000 次迭代的检查点
├── iter_48000.pth    # 第 48000 次迭代的检查点
└── ...
```

#### 修改保存间隔（可选）

如果想更频繁地保存检查点，可以在配置文件中修改：

```python
# 在 configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 中添加
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),  # 改为每 8000 次迭代保存
)
```

### 5. 中断和恢复训练

#### 可以随时中断训练

训练可以随时中断（使用 `Ctrl+C`），不会损坏已保存的检查点。

#### 恢复训练

有两种方式恢复训练：

**方法 1: 自动恢复（推荐）**

使用 `--resume` 参数，会自动从最新的检查点恢复：

```bash
cd segmentation

bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    1 \
    --work-dir work_dirs/ade20k_tiny \
    --resume
```

**方法 2: 从指定检查点恢复**

如果需要从特定的检查点恢复，可以修改配置文件或使用 `--cfg-options`：

```bash
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    1 \
    --work-dir work_dirs/ade20k_tiny \
    --cfg-options resume=True load_from=work_dirs/ade20k_tiny/时间戳目录/iter_32000.pth
```

#### 恢复训练时的注意事项

1. **自动恢复**: 使用 `--resume` 时，系统会自动找到最新的检查点（按迭代次数）
2. **状态恢复**: 恢复时会恢复：
   - 模型权重
   - 优化器状态
   - 学习率调度器状态
   - 当前迭代次数
3. **日志继续**: 训练日志会追加到现有日志文件，不会覆盖
4. **检查点位置**: 确保 `--work-dir` 与之前训练时使用的目录一致

### 5. 监控训练

#### 方法 1: 使用 TensorBoard（推荐）

**重要**: 默认配置使用 `LocalVisBackend`，不会生成 TensorBoard 日志。要使用 TensorBoard，需要在配置文件中添加 TensorBoard 后端。

**步骤 1**: 在配置文件中添加 TensorBoard 支持

编辑你的配置文件（例如 `configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py`），添加：

```python
# 启用 TensorBoard 可视化后端
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)
```

**步骤 2**: 启动 TensorBoard

```bash
cd segmentation

# 监控特定训练目录
tensorboard --logdir work_dirs/ade20k_tiny --port 6006

# 或者监控所有训练目录
tensorboard --logdir work_dirs --port 6006
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

**注意**: 
- 如果训练已经启动，需要重新启动训练才能生成 TensorBoard 日志
- TensorBoard 日志文件（`.tfevents`）会保存在 `work_dirs/xxx/时间戳/` 目录下

#### 方法 2: 查看日志文件

如果使用默认的 `LocalVisBackend`，训练日志保存在：

```bash
# 查看最新的训练日志
tail -f work_dirs/ade20k_tiny/最新时间戳目录/时间戳.log

# 例如
tail -f work_dirs/ade20k_tiny/20251218_220455/20251218_220455.log
```

---

## 测试与评估

### 1. 单 GPU 测试

**重要**: 必须设置 `PYTHONPATH` 或使用 `dist_test.sh` 脚本。

#### 方法 1: 使用 dist_test.sh（推荐）

```bash
cd segmentation

# 使用 1 个 GPU 测试（单尺度）
bash tools/dist_test.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth \
    1

# 使用 1 个 GPU 测试（多尺度 + TTA）
bash tools/dist_test.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth \
    1 \
    --tta
```

#### 方法 2: 手动设置 PYTHONPATH

```bash
cd segmentation

# 测试模型（单尺度）
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/test.py \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth

# 测试模型（多尺度 + TTA，获得更高精度）
PYTHONPATH="$(pwd):$PYTHONPATH" python tools/test.py \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth \
    --tta
```

### 2. 多 GPU 分布式测试

```bash
cd segmentation

# 使用 4 个 GPU 测试
bash tools/dist_test.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth \
    4

# 使用 TTA 测试
bash tools/dist_test.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth \
    4 \
    --tta
```

### 3. 评估指标

测试完成后会输出以下指标：
- **mIoU (mean Intersection over Union)**: 平均交并比，主要评估指标
- **单尺度 (SS)**: 使用单一输入尺寸测试
- **多尺度 (MS)**: 使用多个输入尺寸测试（通常更高）

### 4. 可视化结果

```bash
# 保存可视化结果
python tools/test.py \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/ade20k_small/iter_160000.pth \
    --show-dir work_dirs/ade20k_small/vis_results
```

---

## 配置文件说明

### 1. 数据集配置

文件位置: `segmentation/configs/_base_/datasets/ade20k.py`

主要配置项：
- `data_root`: 数据集根目录（默认: `data/ade/ADEChallengeData2016`）
- `crop_size`: 训练时的裁剪尺寸（默认: `(512, 512)`）
- `batch_size`: 批次大小（默认: 4）
- `num_workers`: 数据加载线程数（默认: 4）

### 2. 模型配置

VMamba 提供了三个规模的模型：

#### Tiny 模型
- 配置文件: `configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py`
- 参数量: 约 62M
- FLOPs: 约 949G
- 预期 mIoU: ~47.9 (SS) / ~48.8 (MS)

#### Small 模型
- 配置文件: `configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py`
- 参数量: 约 82M
- FLOPs: 约 1028G
- 预期 mIoU: ~50.6 (SS) / ~51.2 (MS)

#### Base 模型
- 配置文件: `configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py`
- 参数量: 约 122M
- FLOPs: 约 1170G
- 预期 mIoU: ~51.0 (SS) / ~51.6 (MS)

### 3. 修改配置

如果需要修改数据集路径或其他配置，可以：

**方法 1**: 直接修改配置文件
```python
# 在配置文件中修改
data_root = 'your/path/to/ADEChallengeData2016'
```

**方法 2**: 使用命令行参数
```bash
python tools/train.py \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    --cfg-options data_root='your/path/to/ADEChallengeData2016' \
                  train_dataloader.batch_size=8
```

### 4. 预训练权重

模型配置中指定了预训练权重路径（用于初始化 backbone）：
- Tiny: `../../ckpts/classification/outs/vssm/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth`
- Small: `../../ckpts/classification/outs/vssm/vssmsmall/vssmsmall_dp03_ckpt_epoch_238.pth`
- Base: `../../ckpts/classification/outs/vssm/vssmbasedp05/vssmbase_dp05_ckpt_epoch_260.pth`

如果预训练权重路径不同，需要修改配置文件中的 `pretrained` 参数。

---

## 常见问题

### 1. 数据集路径错误

**问题**: `FileNotFoundError: data/ade/ADEChallengeData2016/...`

**解决**: 
- 检查数据集是否正确下载和组织
- 确认 `data_root` 配置是否正确
- 使用绝对路径或相对路径（相对于项目根目录）

### 2. 显存不足 (OOM)

**问题**: `RuntimeError: CUDA out of memory`

**解决**:
- 减小 `batch_size`（在配置文件中修改 `train_dataloader.batch_size`）
- 使用更小的模型（Tiny 而非 Base）
- 启用混合精度训练（添加 `--amp` 参数）
- 减小输入图像尺寸（修改 `crop_size`）

### 3. 预训练权重加载失败

**问题**: `FileNotFoundError: pretrained checkpoint not found`

**解决**:
- 从官方仓库下载预训练权重
- 修改配置文件中的 `pretrained` 路径
- 或设置为 `None` 从头训练（性能会下降）

### 4. 训练速度慢

**解决**:
- 增加 `num_workers`（但不要超过 CPU 核心数）
- 使用多 GPU 训练
- 启用混合精度训练（`--amp`）
- 检查数据加载是否成为瓶颈

### 5. 评估指标异常

**问题**: mIoU 为 0 或非常低

**解决**:
- 检查标注文件是否正确（像素值范围应为 0-149）
- 确认 `reduce_zero_label=True` 配置正确
- 检查数据集路径和文件命名


---

## 快速开始示例

### 完整训练流程

```bash
# 1. 激活环境
conda activate vmamba

# 2. 进入分割目录
cd segmentation

# 3. 开始训练（使用 4 个 GPU）
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    4 \
    --work-dir work_dirs/ade20k_tiny 

# 4. 训练完成后测试
bash tools/dist_test.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py \
    work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_small/iter_160000.pth \
    4 \
    --tta
```

### 单 GPU 快速测试

```bash
cd segmentation

# 如果只有 1 个 GPU，使用 dist_train.sh（会自动设置 PYTHONPATH）
bash tools/dist_train.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    1 \
    --work-dir work_dirs/ade20k_tiny

# 测试（使用 dist_test.sh）
bash tools/dist_test.sh \
    configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py \
    work_dirs/ade20k_tiny/iter_160000.pth \
    1
```

---

## 参考资源

- **项目主页**: https://github.com/MzeroMiko/VMamba
- **论文**: [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)
- **MMSegmentation 文档**: https://mmsegmentation.readthedocs.io/
- **ADE20K 数据集**: https://groups.csail.mit.edu/vision/datasets/ADE20K/

---

## 总结

本指南涵盖了在 ADE20K 数据集上使用 VMamba 进行语义分割的完整流程：

1. ✅ 环境配置和依赖安装
2. ✅ 数据集准备和组织
3. ✅ 模型训练（单 GPU / 多 GPU）
4. ✅ 模型测试和评估
5. ✅ 配置文件说明和修改
6. ✅ 常见问题排查

按照本指南操作，您应该能够成功训练和评估 VMamba 模型。如有问题，请参考常见问题部分或查看项目 Issues。

---

**祝训练顺利！** 🚀

