#!/bin/bash
# 快速设置SBU数据集路径
# 根据用户需求，按照之前的SBU路径配置

echo "=========================================="
echo "SBU数据集路径配置"
echo "=========================================="

cd /root/autodl-tmp/code/VMamba

# 创建data目录
mkdir -p data

# 检查数据集是否已存在
if [ -d "data/sbu/img" ] && [ -d "data/sbu/label" ]; then
    echo "✓ SBU数据集已存在于: data/sbu/"
    echo ""
    echo "图像数量: $(ls data/sbu/img 2>/dev/null | wc -l)"
    echo "标注数量: $(ls data/sbu/label 2>/dev/null | wc -l)"
    echo ""
    echo "可以开始训练了！"
    exit 0
fi

echo "⚠ data/sbu/ 不存在"
echo ""
echo "请提供SBU数据集的实际位置："
echo ""
echo "常见位置："
echo "  1. /root/autodl-fs/dataset/SBU"
echo "  2. /root/autodl-tmp/dataset/SBU"
echo "  3. /root/datasets/SBU"
echo "  4. ~/datasets/SBU"
echo ""
echo "找到数据集后，运行以下命令创建软链接："
echo ""
echo "  ln -s /实际路径/SBU data/sbu"
echo ""
echo "例如："
echo "  ln -s /root/autodl-fs/dataset/SBU data/sbu"
echo ""

# 尝试自动搜索
echo "正在搜索SBU数据集..."
FOUND=0

SEARCH_PATHS=(
    "/root/autodl-fs"
    "/root/autodl-tmp"
    "/root/datasets"
    "~/datasets"
)

for base_path in "${SEARCH_PATHS[@]}"; do
    # 扩展波浪号
    base_path="${base_path/#\~/$HOME}"

    if [ -d "$base_path" ]; then
        # 搜索SBU相关目录
        sbu_path=$(find "$base_path" -maxdepth 3 -type d -iname "*SBU*" -o -iname "*shadow*" 2>/dev/null | head -1)

        if [ -n "$sbu_path" ] && [ -d "$sbu_path" ]; then
            # 检查是否包含img和label目录
            if [ -d "$sbu_path/img" ] || [ -d "$sbu_path/train" ]; then
                echo "✓ 找到可能的SBU数据集: $sbu_path"
                echo ""
                read -p "是否使用此路径? (y/n): " confirm
                if [ "$confirm" = "y" ]; then
                    ln -s "$sbu_path" data/sbu
                    echo "✓ 软链接创建成功"
                    FOUND=1
                    break
                fi
            fi
        fi
    fi
done

if [ $FOUND -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "手动设置步骤"
    echo "=========================================="
    echo ""
    echo "方案1: 创建软链接（推荐）"
    echo "----------------------------------------"
    echo "如果您的SBU数据集在 /path/to/SBU，运行："
    echo "  cd /root/autodl-tmp/code/VMamba"
    echo "  mkdir -p data"
    echo "  ln -s /path/to/SBU data/sbu"
    echo ""
    echo "方案2: 直接使用路径"
    echo "----------------------------------------"
    echo "修改配置文件中的 data_root 路径："
    echo "  vim segmentation/configs/sbu/sbu_shadow_dual_stream_base_40k.py"
    echo ""
    echo "找到这行（约209行）:"
    echo "  data_root='data/sbu',"
    echo ""
    echo "改为实际路径:"
    echo "  data_root='/your/actual/path',"
    echo ""
    echo "方案3: 创建临时测试数据集"
    echo "----------------------------------------"
    echo "用于快速验证模型配置："
    echo "  python3 create_test_dataset.py"
    echo ""
fi

echo "=========================================="
