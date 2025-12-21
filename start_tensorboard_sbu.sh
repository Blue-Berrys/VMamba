#!/bin/bash
# 启动 TensorBoard 查看 SBU 阴影检测训练日志

# 进入 segmentation 目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/segmentation" || {
    echo "❌ 错误: 无法进入 segmentation 目录"
    exit 1
}

# 查找最新的训练日志目录
LATEST_DIR=$(ls -td work_dirs/sbu_shadow_detection_vssm_base/202* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 错误: 未找到训练日志目录"
    exit 1
fi

# 复制日志到 /root/tf-logs/
TF_LOGS_DIR="/root/tf-logs"
mkdir -p "$TF_LOGS_DIR"
echo "复制日志到 $TF_LOGS_DIR..."
cp -r "$LATEST_DIR" "$TF_LOGS_DIR/" || {
    echo "⚠️  警告: 复制失败，使用原始目录"
    TF_LOGS_DIR=$(cd "$LATEST_DIR" && pwd)
}

# 启动 TensorBoard
echo "启动 TensorBoard..."
echo "日志目录: $TF_LOGS_DIR"
echo "访问地址: http://localhost:6007"
echo ""

tensorboard --port 6007 --logdir "$TF_LOGS_DIR" --bind_all
