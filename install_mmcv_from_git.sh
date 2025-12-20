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
cd /tmp
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
MMCV_WITH_OPS=1 pip install -e . --no-build-isolation 2>&1 | tee /tmp/mmcv_git_install.log

echo ""
echo "步骤 7: 验证安装..."
python -c "
try:
    from mmcv.ops import point_sample
    print('✓ mmcv 扩展模块安装成功！')
    import mmcv
    print(f'mmcv 版本: {mmcv.__version__}')
except Exception as e:
    print(f'✗ 验证失败: {e}')
    import traceback
    traceback.print_exc()
" 2>&1

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




