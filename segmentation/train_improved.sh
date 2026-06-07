#!/bin/bash
# ===================================================================
# 改进的双流VMamba阴影检测模型 - 训练启动脚本
# ===================================================================
#
# 集成路径: /home/xjx/CodeProject/PycharmProject/VMamba/segmentation/train_improved.sh
#
# 使用方法:
#   bash train_improved.sh [gpu_ids]
#
# 示例:
#   bash train_improved.sh 0       # 单GPU训练
#   bash train_improved.sh 0,1    # 双GPU训练
#   bash train_improved.sh 0,1,2,3  # 四GPU训练
#
# 作者: AgentLaboratory
# 日期: 2026-02-19

# 设置脚本在出错时退出
set -e

# ============================================================
# 配置参数
# ============================================================

# GPU设置
GPUS=${1:-"0,1"}  # 默认使用GPU 0和1
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# 工作目录
WORK_DIR="work_dirs/improved_shadow_dual_stream_sbu_base"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="${WORK_DIR}_${TIMESTAMP}"

# 配置文件
CONFIG="configs/sbu/improved_shadow_dual_stream_base_40k.py"

# 显存相关
# batch_size=4 适合单张A100 40GB
# 如果显存不足，可以在配置文件中修改batch_size

# ============================================================
# 打印配置信息
# ============================================================

echo "=========================================="
echo "改进的双流VMamba阴影检测训练"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "GPU数量: $NUM_GPUS"
echo "GPU ID: $GPUS"
echo "工作目录: $WORK_DIR"
echo "=========================================="
echo ""
echo "训练阶段:"
echo "  阶段1 (0-20000):    全局流训练"
echo "  阶段2 (20000-30000): 局部流训练"
echo "  阶段3 (30000-40000): 联合训练"
echo "=========================================="
echo ""

# ============================================================
# 环境检查
# ============================================================

# 检查是否在正确的目录
if [ ! -f "tools/train.py" ]; then
    echo "错误: 请在VMamba/segmentation目录下运行此脚本"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查必要的模块
if [ ! -f "../classification/models/shadow_dual_stream_v2.py" ]; then
    echo "警告: shadow_dual_stream_v2.py 不存在"
    echo "请确保改进的模型已正确集成"
fi

# ============================================================
# 创建工作目录
# ============================================================

mkdir -p "$WORK_DIR"
echo "工作目录已创建: $WORK_DIR"

# ============================================================
# 训练启动
# ============================================================

if [ $NUM_GPUS -eq 1 ]; then
    # 单GPU训练
    echo "启动单GPU训练..."
    CUDA_VISIBLE_DEVICES=$GPUS python tools/train.py \
        $CONFIG \
        --work-dir $WORK_DIR \
        --seed 0 \
        --deterministic
else
    # 多GPU分布式训练
    echo "启动${NUM_GPUS}卡分布式训练..."

    # 使用torchrun启动分布式训练 (推荐)
    CUDA_VISIBLE_DEVICES=$GPUS torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        tools/train.py \
        $CONFIG \
        --work-dir $WORK_DIR \
        --launcher pytorch \
        --seed 0 \
        --deterministic
fi

# ============================================================
# 训练完成
# ============================================================

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "工作目录: $WORK_DIR"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir $WORK_DIR"
echo ""
echo "查看最佳模型:"
echo "  ls -lh $WORK_DIR/best_*.pth"
echo "=========================================="
