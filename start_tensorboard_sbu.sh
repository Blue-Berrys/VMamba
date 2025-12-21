#!/bin/bash
# 启动 TensorBoard 查看 SBU 阴影检测训练日志

echo "=========================================="
echo "启动 TensorBoard - SBU 阴影检测"
echo "=========================================="

# 进入 segmentation 目录
cd segmentation

# 检查 work_dirs 是否存在
if [ ! -d "work_dirs/sbu_shadow_detection_vssm_base" ]; then
    echo "❌ 错误: work_dirs/sbu_shadow_detection_vssm_base 目录不存在"
    echo "请先开始训练，或检查配置文件中的 work_dir 设置"
    exit 1
fi

# 查找最新的训练日志目录
LATEST_DIR=$(ls -td work_dirs/sbu_shadow_detection_vssm_base/202* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 错误: 未找到训练日志目录"
    echo "请先开始训练"
    exit 1
fi

echo "📁 日志目录: $LATEST_DIR"
echo ""

# 启动 TensorBoard，指定具体的日志目录，避免扫描整个目录树
# 使用 --logdir 指定具体目录，而不是当前目录
# 使用 --bind_all 允许从外部访问（如果需要）
echo "🚀 启动 TensorBoard..."
echo "📊 访问地址: http://localhost:6006"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""

tensorboard --logdir="$LATEST_DIR" --port=6006 --bind_all

