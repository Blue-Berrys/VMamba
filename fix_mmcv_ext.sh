#!/bin/bash
# 修复 mmcv._ext 缺失问题 - 从 GitHub 源码编译安装

echo "=========================================="
echo "修复 mmcv CUDA 扩展缺失问题"
echo "从 GitHub 源码编译安装 mmcv 2.1.0"
echo "=========================================="

# 步骤 1: 卸载当前 mmcv
echo ""
echo "步骤 1: 卸载当前 mmcv..."
pip uninstall mmcv mmcv-full -y

# 步骤 2: 设置环境变量
echo ""
echo "步骤 2: 设置编译环境变量..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MMCV_WITH_OPS=1
export FORCE_CUDA=1

echo "CUDA_HOME: $CUDA_HOME"
echo "MMCV_WITH_OPS: $MMCV_WITH_OPS"
echo "检查 nvcc..."
which nvcc
nvcc --version | head -1

# 步骤 3: 克隆 mmcv 源码
echo ""
echo "步骤 3: 克隆 mmcv 源码..."
cd /tmp
if [ -d "mmcv" ]; then
    echo "删除旧的 mmcv 目录..."
    rm -rf mmcv
fi

echo "正在克隆 mmcv v2.1.0..."
git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv

# 步骤 4: 安装构建依赖
echo ""
echo "步骤 4: 安装构建依赖..."
pip install -r requirements/runtime.txt -q
pip install wheel setuptools -q

# 步骤 5: 编译安装
echo ""
echo "步骤 5: 开始编译安装（这需要 10-30 分钟）..."
echo "使用 MMCV_WITH_OPS=1 确保编译 CUDA 扩展..."
MMCV_WITH_OPS=1 pip install -e . --no-build-isolation 2>&1 | tee /tmp/mmcv_fix_compile.log

# 步骤 6: 验证安装
echo ""
echo "步骤 6: 验证安装..."
python -c "
import sys
try:
    import mmcv
    from mmcv.ops import point_sample
    import mmcv._ext
    print('✓ mmcv:', mmcv.__version__)
    print('✓ mmcv CUDA 扩展安装成功')
    print('✓ _ext 模块可以正常导入')
    sys.exit(0)
except ImportError as e:
    print('✗ 验证失败:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print('✗ 验证失败:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ mmcv CUDA 扩展修复成功！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ mmcv CUDA 扩展仍然无法导入"
    echo "请检查编译日志: /tmp/mmcv_fix_compile.log"
    echo "=========================================="
    exit 1
fi

