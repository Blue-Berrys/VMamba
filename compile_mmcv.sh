#!/bin/bash
# 从源码编译安装 mmcv

echo "=========================================="
echo "从源码编译安装 mmcv 2.1.0"
echo "这可能需要 10-30 分钟，请耐心等待..."
echo "=========================================="

# 确保在正确的环境中
# conda activate vim

echo ""
echo "步骤 1: 卸载当前 mmcv..."
pip uninstall mmcv mmcv-full -y

echo ""
echo "步骤 2: 检查编译环境..."
echo "CUDA_HOME: $CUDA_HOME"
echo "检查 gcc..."
gcc --version | head -1
echo "检查 nvcc..."
nvcc --version | head -1

echo ""
echo "步骤 3: 开始从源码编译安装 mmcv..."
echo "这可能需要较长时间，请等待..."
echo ""

# 设置环境变量以确保使用正确的 CUDA
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 从源码编译安装
pip install mmcv==2.1.0 --no-cache-dir --verbose 2>&1 | tee /tmp/mmcv_compile.log

echo ""
echo "步骤 4: 验证安装..."
python -c "from mmcv.ops import point_sample; print('✓ mmcv 扩展模块安装成功！')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ mmcv 编译安装成功！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ mmcv 扩展模块仍然无法导入"
    echo "请检查编译日志: /tmp/mmcv_compile.log"
    echo "=========================================="
fi
