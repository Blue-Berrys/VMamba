#!/bin/bash
# 快速修复：创建SBU数据集软链接

echo "=========================================="
echo "SBU数据集路径配置"
echo "=========================================="

# 检查当前目录
cd /root/autodl-tmp/code/VMamba

echo ""
echo "当前工作目录: $(pwd)"
echo ""

# 检查是否已有数据集
if [ -d "data/sbu/img" ] && [ -d "data/sbu/label" ]; then
    echo "✓ SBU数据集已存在于: data/sbu/"
    echo ""
    echo "数据集内容:"
    echo "  图像数量: $(ls data/sbu/img | wc -l)"
    echo "  标注数量: $(ls data/sbu/label | wc -l)"
    echo ""
    echo "可以开始训练！"
    exit 0
fi

# 如果不存在，尝试查找常见位置
echo "⚠ data/sbu/ 不存在"
echo ""
echo "正在搜索SBU数据集..."

# 常见的数据集位置
POSSIBLE_PATHS=(
    "/root/autodl-tmp/datasets/SBU"
    "/root/autodl-fs/datasets/SBU"
    "/root/datasets/SBU"
    "/datasets/SBU"
    "~/datasets/SBU"
    "/root/SBUShadow"
    "/root/SBU"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✓ 找到数据集: $path"
        echo ""
        echo "创建软链接..."
        mkdir -p data
        ln -s "$path" data/sbu
        echo "✓ 软链接创建成功: data/sbu -> $path"
        echo ""
        echo "现在可以开始训练了！"
        exit 0
    fi
done

echo ""
echo "❌ 未找到SBU数据集"
echo ""
echo "=========================================="
echo "解决方案"
echo "=========================================="
echo ""
echo "方案1: 下载数据集（推荐）"
echo "----------------------------------------"
echo "SBU数据集下载地址："
echo "  官方: https://www.cs.sbu.edu/stuliu/iPhone-shadow-dataset/"
echo "  或使用网盘链接"
echo ""
echo "下载后解压到: /root/autodl-tmp/datasets/SBU/"
echo "然后运行: ln -s /root/autodl-tmp/datasets/SBU data/sbu"
echo ""
echo "方案2: 使用其他数据集测试模型"
echo "----------------------------------------"
echo "临时使用合成数据测试："
echo "  mkdir -p data/sbu/img data/sbu/label"
echo "  cp /path/to/test/images/* data/sbu/img/"
echo "  cp /path/to/test/labels/* data/sbu/label/"
echo ""
echo "方案3: 修改配置文件的数据路径"
echo "----------------------------------------"
echo "在配置文件中修改:"
echo "  train_dataloader=dict("
echo "    dataset=dict("
echo "      data_root='/your/path/to/sbu',"
echo "      ..."
echo "    )"
echo "  )"
echo ""
echo "=========================================="
