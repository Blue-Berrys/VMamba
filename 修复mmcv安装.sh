#!/bin/bash
# 快速修复 mmcv 导入问题

echo "=========================================="
echo "修复 mmcv 导入问题"
echo "=========================================="

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vim

echo ""
echo "步骤 1: 检查当前 mmcv 状态..."
pip show mmcv 2>&1 | head -10

echo ""
echo "步骤 2: 卸载损坏的 mmcv 安装..."
pip uninstall mmcv mmcv-full -y

echo ""
echo "步骤 3: 重新安装 mmcv（使用预编译版本）..."
echo "如果预编译版本不可用，将使用源码安装..."

# 尝试使用预编译版本（更快）
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.split('.')[0]+torch.version.cuda.split('.')[1])" 2>/dev/null || echo "118")
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "2.2.0")

echo "检测到 CUDA: $CUDA_VERSION, PyTorch: $PYTORCH_VERSION"

# 尝试从官方源安装预编译版本
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION}/torch${PYTORCH_VERSION}/index.html 2>&1 | tail -5

# 验证安装
echo ""
echo "步骤 4: 验证安装..."
python -c "
try:
    from mmcv.ops import point_sample
    print('✓ mmcv 扩展模块安装成功！')
    import mmcv
    print(f'✓ mmcv 版本: {mmcv.__version__}')
    print(f'✓ mmcv 位置: {mmcv.__file__}')
except Exception as e:
    print(f'✗ 预编译版本安装失败: {e}')
    print('将使用源码安装...')
    exit(1)
" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "预编译版本不可用，使用源码安装..."
    echo "运行: bash install_mmcv_from_git.sh"
fi

