#!/bin/bash
# ============================================================================
# VMamba 环境安装脚本 - CUDA 12.8 + PyTorch 2.9.1
# 适用于 NVIDIA RTX 50 系列 GPU
# ============================================================================
#
# 使用方法：
#   方法1（推荐）: 先激活环境，再运行脚本
#     conda activate vmamba
#     bash install_cuda128_pytorch29.sh
#
#   方法2: 直接运行脚本（脚本会自动激活环境）
#     bash install_cuda128_pytorch29.sh
#
# ============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VMamba 环境安装脚本"
echo "CUDA 12.8 + PyTorch 2.9.1"
echo "=========================================="
echo ""
echo "📝 提示: 脚本会自动检查并激活 conda vmamba 环境"
echo "   如果遇到问题，请先手动激活: conda activate vmamba"
echo ""

# 初始化 conda（如果尚未初始化）
# 尝试多种方式初始化 conda
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
fi

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "✗ 错误: 未找到 conda 命令"
    echo "请先安装 Anaconda 或 Miniconda，或手动激活 conda 环境后运行此脚本"
    echo ""
    echo "建议的安装方式："
    echo "  1. conda create -n vmamba python=3.12 -y"
    echo "  2. conda activate vmamba"
    echo "  3. bash install_cuda128_pytorch29.sh"
    exit 1
fi

# 检查 vmamba 环境是否存在
ENV_EXISTS=$(conda env list | grep -E "^vmamba\s" || echo "")

if [ -z "$ENV_EXISTS" ]; then
    echo "⚠️  vmamba 环境不存在"
    read -p "是否创建 vmamba 环境？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在创建 conda 环境 vmamba (Python 3.12)..."
        conda create -n vmamba python=3.12 -y
        echo "✓ 环境创建成功"
    else
        echo "✗ 取消安装"
        exit 1
    fi
fi

# 检查是否在 vmamba 环境中
if [ "$CONDA_DEFAULT_ENV" != "vmamba" ]; then
    echo "正在激活 conda 环境: vmamba"
    # 确保 conda 已初始化（再次尝试，以防第一次失败）
    if ! command -v conda &> /dev/null; then
        if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            source "/opt/conda/etc/profile.d/conda.sh"
        elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
        fi
    fi
    
    # 尝试激活环境
    if conda activate vmamba 2>/dev/null; then
        echo "✓ 已激活 vmamba 环境"
    else
        # 如果 conda activate 失败，提示用户手动激活
        echo "⚠️  无法自动激活 conda 环境"
        echo "请手动激活环境后重新运行脚本："
        echo "  conda activate vmamba"
        echo "  bash install_cuda128_pytorch29.sh"
        exit 1
    fi
else
    echo "✓ 已在 vmamba 环境中"
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

# 询问是否使用国内镜像源（在开始安装前询问）
echo ""
echo "=========================================="
echo "镜像源配置"
echo "=========================================="
echo "是否使用国内镜像源加速下载？(强烈推荐，可提速 10-100 倍)"
read -p "使用镜像源？(y/n，默认y) " -n 1 -r
echo
USE_MIRROR=${REPLY:-y}
if [[ $USE_MIRROR =~ ^[Yy]$ ]]; then
    echo "✓ 将使用清华大学镜像源加速下载"
    # 配置 pip 使用清华镜像（临时）
    export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "✓ 将使用官方源（可能较慢）"
    unset PIP_INDEX_URL
fi

# 步骤 1: 安装 PyTorch
echo ""
echo "=========================================="
echo "步骤 1: 安装 PyTorch 2.9.1 (CUDA 12.8)"
echo "=========================================="
echo "⚠️  PyTorch 文件较大 (~900MB)，请耐心等待..."

if [[ $USE_MIRROR =~ ^[Yy]$ ]]; then
    echo "使用清华大学镜像源 + PyTorch 官方源..."
    # PyTorch 需要从官方源下载 CUDA 版本，但其他依赖可以从镜像下载
    # 使用 --extra-index-url 同时支持镜像和官方源
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://download.pytorch.org/whl/cu128 || {
        echo "清华镜像失败，尝试直接使用 PyTorch 官方源..."
        pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    }
else
    echo "使用 PyTorch 官方源（可能较慢）..."
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
fi

# 验证 PyTorch
echo "验证 PyTorch 安装..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA 可用: {torch.cuda.is_available()}')" || {
    echo "✗ PyTorch 安装失败"
}

# 步骤 2: 安装基础编译工具
echo ""
echo "=========================================="
echo "步骤 2: 安装基础编译工具"
echo "=========================================="
if [[ $USE_MIRROR =~ ^[Yy]$ ]]; then
    pip install packaging ninja einops setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    pip install packaging ninja einops setuptools wheel
fi

# 步骤 3: 安装 tokenizers（预编译版本）
echo ""
echo "=========================================="
echo "步骤 3: 安装 tokenizers (预编译版本)"
echo "=========================================="
if [[ $USE_MIRROR =~ ^[Yy]$ ]]; then
    pip install tokenizers --only-binary :all: -i https://pypi.tuna.tsinghua.edu.cn/simple || {
        echo "⚠️  tokenizers 预编译版本安装失败，尝试普通安装..."
        pip install tokenizers -i https://pypi.tuna.tsinghua.edu.cn/simple
    }
else
    pip install tokenizers --only-binary :all: || {
        echo "⚠️  tokenizers 预编译版本安装失败，尝试普通安装..."
        pip install tokenizers
    }
fi

# 步骤 4: 安装其他依赖
echo ""
echo "=========================================="
echo "步骤 4: 安装其他依赖"
echo "=========================================="
if [ -f "requirements_cuda128_pytorch29.txt" ]; then
    if [[ $USE_MIRROR =~ ^[Yy]$ ]]; then
        echo "使用清华大学镜像源安装依赖..."
        pip install -r requirements_cuda128_pytorch29.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    else
        pip install -r requirements_cuda128_pytorch29.txt
    fi
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
    
    # 自动检测 CUDA_HOME
    if [ -z "$CUDA_HOME" ]; then
        if [ -d "/usr/local/cuda-12.8" ]; then
            export CUDA_HOME=/usr/local/cuda-12.8
        elif [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME=/usr/local/cuda
        else
            # 尝试从 nvcc 推断
            NVCC_PATH=$(which nvcc 2>/dev/null)
            if [ -n "$NVCC_PATH" ]; then
                export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
            fi
        fi
    fi
    
    # 设置环境变量
    if [ -n "$CUDA_HOME" ]; then
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        echo "✓ 使用 CUDA_HOME: $CUDA_HOME"
    else
        echo "⚠️  警告: 未找到 CUDA_HOME，尝试使用系统默认路径"
        export CUDA_HOME=/usr/local/cuda-12.8
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    fi
    
    # 确保启用 CUDA 扩展
    export MMCV_WITH_OPS=1
    export FORCE_CUDA=1
    
    # 从源码编译
    pip install mmcv==2.1.0 --no-cache-dir
}

# 验证 mmcv
echo "验证 mmcv 安装..."
python -c "
try:
    import mmcv
    from mmcv.ops import point_sample
    # 检查 CUDA 扩展是否存在
    import mmcv._ext
    print('✓ mmcv:', mmcv.__version__)
    print('✓ mmcv CUDA 扩展安装成功')
except ImportError as e:
    print('✗ mmcv CUDA 扩展安装失败:', str(e))
    print('请检查编译日志或手动编译')
    exit(1)
" || {
    echo "✗ mmcv CUDA 扩展验证失败"
    echo "提示：如果预编译版本不可用，可能需要从源码编译："
    echo "  export CUDA_HOME=/usr/local/cuda-12.8"
    echo "  export MMCV_WITH_OPS=1"
    echo "  pip install mmcv==2.1.0 --no-cache-dir"
}

# 步骤 6: 编译安装 selective_scan
echo ""
echo "=========================================="
echo "步骤 6: 编译安装 selective_scan"
echo "=========================================="
if [ -d "kernels/selective_scan" ]; then
    cd kernels/selective_scan
    
    # 设置环境变量（如果 GPU 不可见，使用默认值）
    # RTX 50 系列 (Blackwell) 使用 compute capability 9.0
    # 如果您的 GPU 不同，请修改 CUDA_ARCH 值：
    # - RTX 40 系列: 89 (8.9)
    # - RTX 30 系列: 86 (8.6)
    # - RTX 20 系列: 75 (7.5)
    export CUDA_ARCH=${CUDA_ARCH:-90}  # 默认 9.0 (RTX 50 系列)
    
    echo "正在编译 selective_scan..."
    echo "使用 compute capability: ${CUDA_ARCH} (sm_${CUDA_ARCH})"
    echo "如果编译失败，请检查 CUDA_ARCH 是否匹配您的 GPU"
    
    # 确保 CUDA 环境变量已设置
    if [ -z "$CUDA_HOME" ]; then
        export CUDA_HOME=/usr/local/cuda-12.8
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    fi
    
    pip install . --no-build-isolation
    cd ../..
    
    # 验证 selective_scan（检查 CUDA 扩展模块）
    echo "验证 selective_scan 安装..."
    python -c "
try:
    import selective_scan_cuda_oflex
    print('✓ selective_scan_cuda_oflex 安装成功')
except ImportError:
    try:
        import selective_scan_cuda
        print('✓ selective_scan_cuda 安装成功')
    except ImportError:
        try:
            import selective_scan_cuda_core
            print('✓ selective_scan_cuda_core 安装成功')
        except ImportError:
            print('✗ selective_scan CUDA 扩展安装失败')
            print('请检查编译错误或 GPU 是否可用')
            exit(1)
" || {
        echo "✗ selective_scan 安装失败"
        echo "请检查编译错误"
        echo ""
        echo "如果是因为 GPU 不可见导致的错误，可以："
        echo "  1. 在有 GPU 的环境中编译"
        echo "  2. 或者手动设置 CUDA_ARCH 环境变量："
        echo "     export CUDA_ARCH=90  # RTX 50 系列"
        echo "     cd kernels/selective_scan"
        echo "     pip install . --no-build-isolation"
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
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA 设备数量:', torch.cuda.device_count())
    print('✓ CUDA 设备名称:', torch.cuda.get_device_name(0))

try:
    import mmseg
    print('✓ mmseg:', mmseg.__version__)
except ImportError:
    print('⚠️  mmseg 未安装（可选）')

try:
    import mmcv
    from mmcv.ops import point_sample
    import mmcv._ext
    print('✓ mmcv:', mmcv.__version__)
    print('✓ mmcv CUDA 扩展: OK')
except ImportError as e:
    print('✗ mmcv CUDA 扩展: 失败 -', str(e))

try:
    import selective_scan_cuda_oflex
    print('✓ selective_scan_cuda_oflex: OK')
except ImportError:
    try:
        import selective_scan_cuda
        print('✓ selective_scan_cuda: OK')
    except ImportError:
        try:
            import selective_scan_cuda_core
            print('✓ selective_scan_cuda_core: OK')
        except ImportError:
            print('⚠️  selective_scan CUDA 扩展: 未安装（可能影响性能）')

print('')
print('==========================================')
print('✓ 核心依赖安装完成！')
print('==========================================')
"

echo ""
echo "安装完成！可以开始训练了："
echo "  cd segmentation"
echo "  bash tools/dist_train.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 1"

