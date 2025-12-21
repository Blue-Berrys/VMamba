#!/bin/bash
# 启动 TensorBoard 查看 SBU 阴影检测训练日志

# 设置错误处理：即使某些命令失败也继续执行
set +e

echo "=========================================="
echo "启动 TensorBoard - SBU 阴影检测"
echo "=========================================="

# 获取脚本所在目录的绝对路径（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEGMENTATION_DIR="$SCRIPT_DIR/segmentation"

# 进入 segmentation 目录
cd "$SEGMENTATION_DIR" || {
    echo "❌ 错误: 无法进入 segmentation 目录"
    exit 1
}

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
    # 检查 vis_data 中是否有 events 文件
    EVENTS_IN_VIS=$(find "$VIS_DATA_DIR" -name "events.*" 2>/dev/null | wc -l)
    if [ "$EVENTS_IN_VIS" -gt 0 ]; then
        LOG_DIR="$VIS_DATA_DIR"
        echo "📁 找到 vis_data 目录（包含 $EVENTS_IN_VIS 个 events 文件）: $LOG_DIR"
    else
        # vis_data 存在但没有 events 文件，使用时间戳目录（TensorBoard 会自动扫描子目录）
        LOG_DIR="$LATEST_DIR"
        echo "📁 vis_data 目录存在但无 events 文件，使用时间戳目录: $LOG_DIR"
    fi
else
    # 如果没有 vis_data，使用时间戳目录
    LOG_DIR="$LATEST_DIR"
    echo "📁 使用时间戳目录: $LOG_DIR"
fi

# 检查是否有 events 文件（在整个时间戳目录中搜索）
EVENTS_COUNT=$(find "$LATEST_DIR" -name "events.*" 2>/dev/null | wc -l)
if [ "$EVENTS_COUNT" -eq 0 ]; then
    echo "⚠️  警告: 未找到 TensorBoard events 文件"
    echo "可能原因："
    echo "   1. 训练还未开始或刚启动（等待几分钟）"
    echo "   2. TensorBoardVisBackend 未正确配置"
    echo "   3. 日志还未写入"
    echo ""
    echo "💡 建议："
    echo "   - 检查配置文件是否包含 TensorboardVisBackend"
    echo "   - 等待训练开始后几分钟再启动 TensorBoard"
    echo "   - TensorBoard 会在有数据时自动显示"
    echo ""
    echo "继续启动 TensorBoard，等待数据写入..."
else
    echo "✓ 找到 $EVENTS_COUNT 个 events 文件"
    find "$LATEST_DIR" -name "events.*" 2>/dev/null | head -3
fi

# 步骤 1: 结束现有的 TensorBoard 进程（如果存在）
echo ""
echo "步骤 1: 检查并结束现有的 TensorBoard 进程..."
TB_PIDS=$(ps -ef | grep "[t]ensorboard" | awk '{print $2}')
# 使用 [t]ensorboard 避免 grep 匹配到自己

if [ -n "$TB_PIDS" ]; then
    echo "发现运行中的 TensorBoard 进程: $TB_PIDS"
    # 使用 kill 而不是 kill -9，更安全
    # 使用 || true 确保即使 kill 失败也不会导致脚本退出
    for pid in $TB_PIDS; do
        kill "$pid" 2>/dev/null || true
        echo "  发送 SIGTERM 到进程 $pid"
    done
    sleep 2
    # 如果还有进程，再使用 kill -9
    REMAINING=$(ps -ef | grep "[t]ensorboard" | awk '{print $2}')
    if [ -n "$REMAINING" ]; then
        echo "  强制结束残留进程: $REMAINING"
        for pid in $REMAINING; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    echo "✓ 已结束现有 TensorBoard 进程"
else
    echo "✓ 没有运行中的 TensorBoard 进程"
fi

# 使用绝对路径，避免路径问题
ABS_LOG_DIR=$(cd "$LOG_DIR" && pwd) || {
    echo "❌ 错误: 无法获取日志目录的绝对路径"
    exit 1
}

# 也可以使用 work_dirs 目录（TensorBoard 会自动扫描所有子目录）
ABS_WORK_DIR=$(cd "work_dirs/sbu_shadow_detection_vssm_base" && pwd) || {
    echo "❌ 错误: 无法获取工作目录的绝对路径"
    exit 1
}

echo ""
echo "步骤 2: 启动 TensorBoard..."
echo "📊 日志目录（精确）: $ABS_LOG_DIR"
echo "📊 工作目录（扫描所有实验）: $ABS_WORK_DIR"
echo "📊 访问地址: http://localhost:6007"
echo ""
echo "💡 提示: 如果显示 'No dashboards are active'，请检查："
echo "   1. 训练是否正在进行（需要等待数据写入）"
echo "   2. 配置文件中是否启用了 TensorboardVisBackend"
echo "   3. 等待几分钟让训练写入数据"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""

# 检查 tensorboard 命令是否可用
if ! command -v tensorboard &> /dev/null; then
    echo "❌ 错误: 未找到 tensorboard 命令"
    echo "请安装 TensorBoard: pip install tensorboard"
    exit 1
fi

# 启动 TensorBoard
# 使用 work_dirs 目录，TensorBoard 会自动扫描所有子目录中的 events 文件
# 这样可以看到所有训练实验的日志
# 使用 exec 替换当前进程，确保 TensorBoard 在前台运行
echo "正在启动 TensorBoard..."
echo "执行命令: tensorboard --port=6007 --logdir=\"$ABS_WORK_DIR\" --bind_all"
echo ""

# 使用 exec 替换当前进程，确保 TensorBoard 在前台运行
# 注意：如果 exec 成功，当前进程会被 TensorBoard 替换，后续代码不会执行
# 如果 exec 失败（例如命令不存在），脚本会继续执行
exec tensorboard --port=6007 --logdir="$ABS_WORK_DIR" --bind_all

# 如果执行到这里，说明 exec 失败了
echo ""
echo "❌ 错误: TensorBoard 启动失败"
echo "请检查："
echo "   1. TensorBoard 是否已安装: pip install tensorboard"
echo "   2. 端口 6007 是否被占用: lsof -i:6007"
echo "   3. 日志目录是否存在: $ABS_WORK_DIR"
exit 1

