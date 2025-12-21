#!/bin/bash
# 启动 TensorBoard 查看 SBU 阴影检测训练日志

# 进入 segmentation 目录
cd "$(dirname "$0")/segmentation" || exit 1

# 结束现有的 TensorBoard 进程
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 2>/dev/null

# 查找最新的训练日志目录
LATEST_DIR=$(ls -td work_dirs/sbu_shadow_detection_vssm_base/202* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 错误: 未找到训练日志目录"
    exit 1
fi

# 获取绝对路径
LOG_DIR=$(cd "$LATEST_DIR" && pwd)

# 启动 TensorBoard
echo "启动 TensorBoard..."
echo "日志目录: $LOG_DIR"
echo "访问地址: http://localhost:6007"
echo ""

tensorboard --port 6007 --logdir "$LOG_DIR" --bind_all
