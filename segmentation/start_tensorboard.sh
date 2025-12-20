#!/bin/bash
# TensorBoard 启动脚本
# 使用方法: bash start_tensorboard.sh

cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation

# 激活 conda 环境（如果需要）
# conda activate vim

# 启动 TensorBoard
echo "正在启动 TensorBoard..."
echo "日志目录: work_dirs/ade20k_tiny"
echo "访问地址: http://localhost:6006"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""

tensorboard --logdir work_dirs/ade20k_tiny --port 6006 --host 0.0.0.0
