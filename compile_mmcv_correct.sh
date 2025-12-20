#!/bin/bash
# 正确编译 mmcv 扩展模块

echo "=========================================="
echo "从源码编译安装 mmcv 2.1.0 (带 CUDA 扩展)"
echo "=========================================="

# 确保在正确的环境中
# conda activate vim

echo ""
echo "步骤 1: 卸载当前 mmcv..."
pip uninstall mmcv mmcv-full -y

echo ""
echo "步骤 2: 设置编译环境变量..."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 关键：设置 MMCV_WITH_OPS=1 强制编译扩展
export MMCV_WITH_OPS=1

echo "CUDA_HOME: $CUDA_HOME"
echo "MMCV_WITH_OPS: $MMCV_WITH_OPS"

echo ""
echo "步骤 3: 检查编译工具..."
which gcc
which nvcc
nvcc --version | head -1

echo ""
echo "步骤 4: 开始编译（这需要 10-30 分钟）..."
echo ""

# 使用环境变量强制编译扩展
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-cache-dir --verbose 2>&1 | tee /tmp/mmcv_compile_full.log

echo ""
echo "步骤 5: 验证安装..."
python -c "
try:
    from mmcv.ops import point_sample
    print('✓ mmcv 扩展模块安装成功！')
except Exception as e:
    print(f'✗ 验证失败: {e}')
    print('')
    print('检查已安装的模块:')
    import mmcv
    import os
    mmcv_path = os.path.dirname(mmcv.__file__)
    ext_path = os.path.join(mmcv_path, '_ext')
    if os.path.exists(ext_path):
        print(f'  _ext 目录存在: {ext_path}')
        print(f'  内容: {os.listdir(ext_path)}')
    else:
        print(f'  _ext 目录不存在: {ext_path}')
        print(f'  mmcv 路径: {mmcv_path}')
        print(f'  可用内容: {os.listdir(mmcv_path)[:10]}')
" 2>&1




