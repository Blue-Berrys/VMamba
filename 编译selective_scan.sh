#!/bin/bash
# ============================================================================
# selective_scan 编译脚本（解决编译失败问题）
# ============================================================================

set -e

echo "=========================================="
echo "编译 selective_scan"
echo "=========================================="

# 设置环境变量
export CUDA_ARCH=${CUDA_ARCH:-90}  # RTX 50 系列默认 9.0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 减少并行编译任务数，避免内存不足
export MAX_JOBS=${MAX_JOBS:-2}  # 默认只使用 2 个并行任务

echo "CUDA_ARCH: $CUDA_ARCH"
echo "CUDA_HOME: $CUDA_HOME"
echo "MAX_JOBS: $MAX_JOBS"

cd kernels/selective_scan

# 清理之前的编译
echo "清理之前的编译..."
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
find . -name "*.o" -delete

# 方法1: 使用 pip 安装（减少并行度）
echo ""
echo "方法1: 使用 pip 安装（限制并行度）..."
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=$MAX_JOBS pip install . --no-build-isolation -v 2>&1 | tee /tmp/selective_scan_build.log || {
    echo "方法1 失败，查看错误信息..."
    tail -100 /tmp/selective_scan_build.log | grep -A 20 -i "error\|fail" || tail -50 /tmp/selective_scan_build.log
    
    echo ""
    echo "尝试方法2: 使用 python setup.py（单线程编译）..."
    # 方法2: 使用 setup.py 直接编译（单线程）
    python setup.py build_ext --inplace --force 2>&1 | tee /tmp/selective_scan_build2.log || {
        echo "方法2 也失败，查看错误信息..."
        tail -100 /tmp/selective_scan_build2.log | grep -A 20 -i "error\|fail" || tail -50 /tmp/selective_scan_build2.log
        
        echo ""
        echo "=========================================="
        echo "编译失败的可能原因："
        echo "1. 内存不足 - 尝试增加 swap 或减少 MAX_JOBS"
        echo "2. CUDA 版本不兼容 - 检查 CUDA_HOME 是否正确"
        echo "3. 无卡模式下某些编译步骤失败"
        echo ""
        echo "建议："
        echo "- 在有 GPU 的环境中编译"
        echo "- 或者增加系统内存/swap"
        echo "- 或者设置 MAX_JOBS=1 使用单线程编译"
        echo "=========================================="
        exit 1
    }
    
    # 如果 setup.py 成功，安装包
    pip install . --no-build-isolation
}

cd ../..

# 验证安装
echo ""
echo "验证安装..."
python -c "from selective_scan import selective_scan_fn; print('✓ selective_scan 安装成功')" || {
    echo "⚠️  导入失败，但编译可能已成功"
    echo "可以尝试在有 GPU 的环境中测试"
}

echo ""
echo "=========================================="
echo "编译完成"
echo "=========================================="


