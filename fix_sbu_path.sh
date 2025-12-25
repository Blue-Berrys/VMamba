#!/bin/bash
# 修复SBU数据集路径

echo "=========================================="
echo "SBU数据集路径修复"
echo "=========================================="

cd /root/autodl-tmp/code/VMamba

# 检查实际数据集结构
SBU_PATH="data/SBU-shadow"

if [ ! -d "$SBU_PATH" ]; then
    echo "❌ 错误: $SBU_PATH 不存在"
    echo ""
    echo "请检查实际路径，运行："
    echo "  ls -la data/"
    exit 1
fi

echo "✓ 找到SBU数据集: $SBU_PATH"
echo ""

# 检查子目录
echo "数据集结构:"
ls -la "$SBU_PATH"
echo ""

# 查找训练和测试目录
TRAIN_DIR=""
TEST_DIR=""

if [ -d "$SBU_PATH/SBUTrain4KRecoveredSmall" ]; then
    TRAIN_DIR="$SBU_PATH/SBUTrain4KRecoveredSmall"
    echo "✓ 找到训练集: $TRAIN_DIR"
elif [ -d "$SBU_PATH/train" ]; then
    TRAIN_DIR="$SBU_PATH/train"
    echo "✓ 找到训练集: $TRAIN_DIR"
fi

if [ -d "$SBU_PATH/SBU-Test" ]; then
    TEST_DIR="$SBU_PATH/SBU-Test"
    echo "✓ 找到测试集: $TEST_DIR"
elif [ -d "$SBU_PATH/test" ]; then
    TEST_DIR="$SBU_PATH/test"
    echo "✓ 找到测试集: $TEST_DIR"
fi

echo ""

# 检查训练集内部结构
if [ -n "$TRAIN_DIR" ]; then
    echo "训练集内部结构:"
    ls -la "$TRAIN_DIR" | head -20
    echo ""

    # 检查是否有img和label子目录
    if [ -d "$TRAIN_DIR/img" ] && [ -d "$TRAIN_DIR/label" ]; then
        echo "✓ 标准结构: img/ 和 label/ 子目录存在"
        IMG_DIR="$TRAIN_DIR/img"
        LABEL_DIR="$TRAIN_DIR/label"
    elif [ -d "$TRAIN_DIR/images" ] && [ -d "$TRAIN_DIR/labels" ]; then
        echo "⚠ 备选结构: images/ 和 labels/"
        IMG_DIR="$TRAIN_DIR/images"
        LABEL_DIR="$TRAIN_DIR/labels"
    else
        echo "⚠ 未找到标准子目录，检查文件..."
        # 检查是否直接在目录下
        IMG_COUNT=$(find "$TRAIN_DIR" -name "*.jpg" 2>/dev/null | wc -l)
        LABEL_COUNT=$(find "$TRAIN_DIR" -name "*.png" 2>/dev/null | wc -l)
        echo "  图像文件(.jpg): $IMG_COUNT"
        echo "  标注文件(.png): $LABEL_COUNT"
        IMG_DIR="$TRAIN_DIR"
        LABEL_DIR="$TRAIN_DIR"
    fi

    echo ""
    echo "图像目录: $IMG_DIR"
    echo "标注目录: $LABEL_DIR"
    echo ""
    echo "图像数量: $(ls "$IMG_DIR"/*.jpg 2>/dev/null | wc -l)"
    echo "标注数量: $(ls "$LABEL_DIR"/*.png 2>/dev/null | wc -l)"
fi

# 创建符号链接
echo ""
echo "=========================================="
echo "创建标准路径符号链接"
echo "=========================================="
echo ""

# 方案1: 创建 sbu -> SBU-shadow 的符号链接
if [ ! -e "data/sbu" ]; then
    echo "创建: data/sbu -> $SBU_PATH"
    ln -s "$SBU_PATH" data/sbu
    echo "✓ 符号链接创建成功"
else
    echo "⚠ data/sbu 已存在"
    echo "   当前链接: $(ls -la data/sbu | awk '{print $NF}')"
fi

echo ""
echo "=========================================="
echo "✅ 数据集路径配置完成"
echo "=========================================="
echo ""
echo "标准路径: data/sbu → $SBU_PATH"
echo "训练集: data/sbu/SBUTrain4KRecoveredSmall"
echo "测试集: data/sbu/SBU-Test"
echo ""
echo "现在可以开始训练了！"
echo ""
echo "启动命令:"
echo "  cd segmentation"
echo "  bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4"
echo ""
