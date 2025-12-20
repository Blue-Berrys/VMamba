#!/bin/bash
# ============================================================================
# VMamba 环境安装脚本 - CUDA 12.8 + PyTorch 2.9.1
# 适用于 NVIDIA RTX 50 系列 GPU
# ============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VMamba 环境安装脚本"
echo "CUDA 12.8 + PyTorch 2.9.1"
echo "=========================================="

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  警告: 未检测到 conda 环境"
    echo "建议先创建并激活 conda 环境："
    echo "  conda create -n vmamba python=3.12 -y"
    echo "  conda activate vmamba"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本: $PYTHON_VERSION"

# 检查 CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c 1-4)
    echo "✓ CUDA 版本: $CUDA_VERSION"
else
    echo "⚠️  警告: 未找到 nvcc，请确保 CUDA 已正确安装"
fi

# 步骤 1: 安装 PyTorch
echo ""
echo "=========================================="
echo "步骤 1: 安装 PyTorch 2.9.1 (CUDA 12.8)"
echo "=========================================="
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 验证 PyTorch
echo "验证 PyTorch 安装..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA 可用: {torch.cuda.is_available()}')" || {
    echo "✗ PyTorch 安装失败"
    exit 1
}

# 步骤 2: 安装基础编译工具
echo ""
echo "=========================================="
echo "步骤 2: 安装基础编译工具"
echo "=========================================="
pip install packaging ninja einops setuptools wheel

# 步骤 3: 安装 tokenizers（预编译版本）
echo ""
echo "=========================================="
echo "步骤 3: 安装 tokenizers (预编译版本)"
echo "=========================================="
pip install tokenizers --only-binary :all: || {
    echo "⚠️  tokenizers 预编译版本安装失败，尝试普通安装..."
    pip install tokenizers
}

# 步骤 4: 安装其他依赖
echo ""
echo "=========================================="
echo "步骤 4: 安装其他依赖"
echo "=========================================="
if [ -f "requirements_cuda128_pytorch29.txt" ]; then
    pip install -r requirements_cuda128_pytorch29.txt
else
    echo "✗ 未找到 requirements_cuda128_pytorch29.txt"
    exit 1
fi

# 步骤 5: 安装 mmcv
echo ""
echo "=========================================="
echo "步骤 5: 安装 mmcv 2.1.0"
echo "=========================================="
echo "尝试安装预编译版本..."
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.9/index.html || {
    echo "预编译版本不可用，从源码编译..."
    echo "⚠️  这可能需要 10-30 分钟"
    
    # 卸载旧版本
    pip uninstall mmcv mmcv-full -y
    
    # 设置环境变量
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export MMCV_WITH_OPS=1
    
    # 从源码编译
    pip install mmcv==2.1.0 --no-cache-dir
}

# 验证 mmcv
echo "验证 mmcv 安装..."
python -c "from mmcv.ops import point_sample; print('✓ mmcv CUDA 扩展安装成功')" || {
    echo "✗ mmcv CUDA 扩展安装失败"
    echo "请检查编译日志或手动编译"
}

# 步骤 6: 编译安装 selective_scan
echo ""
echo "=========================================="
echo "步骤 6: 编译安装 selective_scan"
echo "=========================================="
if [ -d "kernels/selective_scan" ]; then
    cd kernels/selective_scan
    echo "正在编译 selective_scan..."
    pip install . --no-build-isolation
    cd ../..
    
    # 验证 selective_scan
    python -c "from selective_scan import selective_scan_fn; print('✓ selective_scan 安装成功')" || {
        echo "✗ selective_scan 安装失败"
        echo "请检查编译错误"
    }
else
    echo "⚠️  未找到 kernels/selective_scan 目录，跳过"
fi

# 最终验证
echo ""
echo "=========================================="
echo "最终验证"
echo "=========================================="
python -c "
import torch
import mmseg
from mmcv.ops import point_sample
from selective_scan import selective_scan_fn

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA 可用:', torch.cuda.is_available())
print('✓ mmseg:', mmseg.__version__)
print('✓ mmcv CUDA 扩展: OK')
print('✓ selective_scan: OK')
print('')
print('==========================================')
print('✓ 所有依赖安装成功！')
print('==========================================')
"

echo ""
echo "安装完成！可以开始训练了："
echo "  cd segmentation"
echo "  bash tools/dist_train.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 1"

