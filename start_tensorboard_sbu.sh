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

# 检查 vis_data 目录是否存在（TensorBoard 日志在这里）
VIS_DATA_DIR="$LATEST_DIR/vis_data"
if [ -d "$VIS_DATA_DIR" ]; then
    LOG_DIR="$VIS_DATA_DIR"
    echo "📁 找到 vis_data 目录: $LOG_DIR"
else
    # 如果没有 vis_data，使用时间戳目录
    LOG_DIR="$LATEST_DIR"
    echo "📁 使用日志目录: $LOG_DIR"
fi

# 检查是否有 events 文件
EVENTS_COUNT=$(find "$LOG_DIR" -name "events.*" 2>/dev/null | wc -l)
if [ "$EVENTS_COUNT" -eq 0 ]; then
    echo "⚠️  警告: 未找到 TensorBoard events 文件"
    echo "可能训练还未开始，或 TensorBoardVisBackend 未正确配置"
    echo "继续启动 TensorBoard，等待数据写入..."
fi

echo ""
echo "🚀 启动 TensorBoard..."
echo "📊 日志目录: $LOG_DIR"
echo "📊 访问地址: http://localhost:6006"
echo ""
echo "💡 提示: 如果显示 'No dashboards are active'，请检查："
echo "   1. 训练是否正在进行"
echo "   2. 配置文件中是否启用了 TensorboardVisBackend"
echo "   3. 等待几分钟让数据写入"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""

# 使用绝对路径，避免路径问题
ABS_LOG_DIR=$(cd "$LOG_DIR" && pwd)
tensorboard --logdir="$ABS_LOG_DIR" --port=6006 --bind_all

