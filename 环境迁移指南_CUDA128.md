# VMamba 环境迁移指南 - CUDA 12.8 + PyTorch 2.9.1

## 📋 环境信息

基于您的训练日志，当前环境配置：
- **Python**: 3.12.12
- **CUDA**: 12.8
- **PyTorch**: 2.9.1+cu128
- **TorchVision**: 0.24.1+cu128
- **GPU**: NVIDIA GeForce RTX 5070 Ti
- **MMEngine**: 0.10.1
- **OpenCV**: 4.12.0

## 🚀 快速安装

### 方法 1: 使用自动安装脚本（推荐）

```bash
# 1. 创建 conda 环境
conda create -n vmamba python=3.12 -y
conda activate vmamba

# 2. 运行安装脚本
bash install_cuda128_pytorch29.sh
```

### 方法 2: 手动安装

```bash
# 1. 创建 conda 环境
conda create -n vmamba python=3.12 -y
conda activate vmamba

# 2. 安装 PyTorch (必须先从官方源安装)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. 安装基础编译工具
pip install packaging ninja einops

# 4. 安装 tokenizers（预编译版本，避免 Rust 编译）
pip install tokenizers --only-binary :all:

# 5. 安装其他依赖
pip install -r requirements_cuda128_pytorch29.txt

# 6. 安装 mmcv（尝试预编译版本）
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.9/index.html

# 如果预编译版本不可用，从源码编译：
# export CUDA_HOME=/usr/local/cuda-12.8
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export MMCV_WITH_OPS=1
# pip install mmcv==2.1.0 --no-cache-dir

# 7. 编译安装 selective_scan
cd kernels/selective_scan
pip install . --no-build-isolation
cd ../..
```

## ✅ 验证安装

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证 mmcv
python -c "from mmcv.ops import point_sample; print('mmcv OK')"

# 验证 selective_scan
python -c "from selective_scan import selective_scan_fn; print('selective_scan OK')"

# 验证 mmseg
python -c "import mmseg; print(f'mmseg: {mmseg.__version__}')"
```

## 📦 关键依赖版本

| 包名 | 版本 | 说明 |
|------|------|------|
| PyTorch | 2.9.1+cu128 | 必须从官方源安装 |
| TorchVision | 0.24.1+cu128 | 与 PyTorch 配套 |
| MMEngine | 0.10.1 | OpenMMLab 核心引擎 |
| mmcv | 2.1.0 | 需要 CUDA 扩展 |
| mmdet | 3.3.0 | 目标检测框架 |
| mmsegmentation | 1.2.2 | 语义分割框架 |
| mmpretrain | 1.2.0 | 预训练模型库 |
| transformers | 4.35.0 | ⚠️ 必须此版本，避免冲突 |
| opencv-python-headless | 4.12.0 | 计算机视觉库 |

## 🔧 多卡训练配置

### 单卡训练
```bash
cd segmentation
python tools/train.py configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
```

### 多卡训练（推荐）
```bash
cd segmentation
bash tools/dist_train.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 4
# 其中 4 是 GPU 数量
```

### 使用 torchrun（PyTorch 2.9+ 推荐）
```bash
cd segmentation
torchrun --nproc_per_node=4 tools/train.py configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
```

## ⚠️ 常见问题

### 1. mmcv 安装失败

**问题**: `ModuleNotFoundError: No module named 'mmcv._ext'`

**解决**:
```bash
# 方法1: 尝试预编译版本
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.9/index.html

# 方法2: 从源码编译（需要 10-30 分钟）
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MMCV_WITH_OPS=1
pip install mmcv==2.1.0 --no-cache-dir
```

### 2. tokenizers 编译错误

**问题**: `error: can't find Rust compiler`

**解决**:
```bash
pip install tokenizers --only-binary :all:
```

### 3. transformers 版本冲突

**问题**: `TypeError: NoneType takes no arguments`

**解决**:
```bash
pip install transformers==4.35.0
```

### 4. selective_scan 编译失败

**问题**: CUDA 编译错误

**解决**:
```bash
# 确保 CUDA 环境变量正确
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新编译
cd kernels/selective_scan
pip uninstall selective-scan -y
python setup.py clean --all
pip install . --no-build-isolation
```

## 📝 文件说明

- **`requirements_cuda128_pytorch29.txt`**: 完整的依赖列表（不含 PyTorch）
- **`install_cuda128_pytorch29.sh`**: 自动安装脚本
- **`环境迁移指南_CUDA128.md`**: 本文件

## 🔗 相关文档

- 完整训练指南: `VMamba分割任务指南.md`
- ADE20K 数据集准备: `ADE20K使用指南.md`

## 💡 提示

1. **PyTorch 必须单独安装**: 因为需要指定 `--index-url` 参数
2. **mmcv 需要编译**: 确保 CUDA 环境变量正确设置
3. **多卡训练**: 使用 `dist_train.sh` 或 `torchrun` 进行分布式训练
4. **TensorBoard**: 训练日志会自动保存，使用 `tensorboard --logdir work_dirs/` 查看

---

**最后更新**: 2025-12-18  
**适用环境**: CUDA 12.8, PyTorch 2.9.1, Python 3.12

