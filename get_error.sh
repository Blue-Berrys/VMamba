#!/bin/bash
# 获取详细错误信息的脚本

echo "=========================================="
echo "获取详细错误信息"
echo "=========================================="

cd /root/autodl-tmp/code/VMamba/segmentation

# 方法1: 单GPU运行（可以看到完整错误）
echo ""
echo "方法1: 单GPU运行测试..."
echo "----------------------------------------"

python tools/train.py \
    configs/sbu/sbu_shadow_dual_stream_base_40k.py \
    --work-dir work_dirs/debug \
    2>&1 | tee train_debug.log

echo ""
echo "错误日志已保存到: train_debug.log"
echo "=========================================="

# 显示最后50行错误
echo ""
echo "最后50行日志："
echo "----------------------------------------"
tail -50 train_debug.log
