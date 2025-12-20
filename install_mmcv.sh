#!/bin/bash
# 安装 mmcv 扩展模块的脚本

echo "=========================================="
echo "安装 mmcv 扩展模块"
echo "=========================================="

# 激活 conda 环境（如果需要）
# conda activate vim

echo "步骤 1: 卸载当前的 mmcv..."
pip uninstall mmcv mmcv-full -y

echo ""
echo "步骤 2: 尝试安装预编译版本..."
echo "尝试 CUDA 12.8 + PyTorch 2.9..."

# 尝试多个可能的预编译版本
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.9/index.html 2>&1 | tee /tmp/mmcv_install.log

if grep -q "No matching distribution" /tmp/mmcv_install.log || grep -q "ERROR" /tmp/mmcv_install.log; then
    echo "预编译版本 cu128/torch2.9 不可用，尝试 cu128/torch2.0..."
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.0/index.html 2>&1 | tee /tmp/mmcv_install.log
    
    if grep -q "No matching distribution" /tmp/mmcv_install.log || grep -q "ERROR" /tmp/mmcv_install.log; then
        echo "预编译版本 cu128/torch2.0 也不可用，尝试 cu121/torch2.0..."
        pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html 2>&1 | tee /tmp/mmcv_install.log
        
        if grep -q "No matching distribution" /tmp/mmcv_install.log || grep -q "ERROR" /tmp/mmcv_install.log; then
            echo ""
            echo "所有预编译版本都不可用，需要从源码编译..."
            echo "这可能需要 10-30 分钟，请耐心等待..."
            echo ""
            read -p "是否继续从源码编译？(y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "开始从源码编译 mmcv..."
                pip install mmcv==2.1.0 --no-cache-dir
            else
                echo "取消安装"
                exit 1
            fi
        fi
    fi
fi

echo ""
echo "步骤 3: 验证安装..."
python -c "from mmcv.ops import point_sample; print('✓ mmcv 扩展模块安装成功！')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ mmcv 安装成功！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ mmcv 扩展模块仍然无法导入"
    echo "可能需要从源码编译"
    echo "=========================================="
    echo ""
    echo "手动编译命令："
    echo "  pip uninstall mmcv -y"
    echo "  pip install mmcv==2.1.0 --no-cache-dir"
fi




