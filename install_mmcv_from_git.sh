#!/bin/bash
# 从 GitHub 源码安装 mmcv（带 CUDA 扩展）

echo "=========================================="
echo "从 GitHub 源码安装 mmcv 2.1.0"
echo "=========================================="

# 确保在正确的环境中
# conda activate vim

echo ""
echo "步骤 1: 卸载当前 mmcv..."
pip uninstall mmcv mmcv-full -y

echo ""
echo "步骤 2: 设置环境变量..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MMCV_WITH_OPS=1

echo "CUDA_HOME: $CUDA_HOME"
echo "MMCV_WITH_OPS: $MMCV_WITH_OPS"

echo ""
echo "步骤 3: 克隆 mmcv 仓库..."
# 使用项目目录下的固定位置，而不是 /tmp（可能被系统清理）
MMCV_DIR="${HOME}/.cache/mmcv_src"
mkdir -p "$MMCV_DIR"
cd "$MMCV_DIR"
if [ -d "mmcv" ]; then
    echo "mmcv 目录已存在，删除..."
    rm -rf mmcv
fi

git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git
cd mmcv

echo ""
echo "步骤 4: 检查版本..."
git describe --tags

echo ""
echo "步骤 5: 安装依赖..."
pip install -r requirements/runtime.txt -q

echo ""
echo "步骤 6: 开始编译安装（这需要 10-30 分钟）..."
echo "使用 MMCV_WITH_OPS=1 确保编译 CUDA 扩展..."
echo "注意: 使用非可编辑模式安装，避免依赖源码目录"
MMCV_WITH_OPS=1 pip install . --no-build-isolation 2>&1 | tee /tmp/mmcv_git_install.log

echo ""
echo "步骤 7: 验证安装..."
python -c "
try:
    from mmcv.ops import point_sample
    print('✓ mmcv 扩展模块安装成功！')
    import mmcv
    print(f'✓ mmcv 版本: {mmcv.__version__}')
    print(f'✓ mmcv 安装位置: {mmcv.__file__}')
except Exception as e:
    print(f'✗ 验证失败: {e}')
    import traceback
    traceback.print_exc()
" 2>&1

echo ""
echo "步骤 8: 检查安装位置..."
python -c "
import mmcv
import os
mmcv_path = os.path.dirname(os.path.dirname(mmcv.__file__))
print(f'mmcv 安装路径: {mmcv_path}')
print(f'是否为可编辑安装: {os.path.exists(os.path.join(mmcv_path, \"setup.py\"))}')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ mmcv 安装成功！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ mmcv 扩展模块仍然无法导入"
    echo "请检查编译日志: /tmp/mmcv_git_install.log"
    echo "=========================================="
fi




