#!/bin/bash
# 改进的双流模型单卡训练脚本

cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
source /home/xjx/miniconda3/etc/profile.d/conda.sh
conda activate vim
export PYTHONPATH=/home/xjx/CodeProject/PycharmProject/VMamba:$PYTHONPATH

echo '=========================================='
echo '改进的双流VMamba阴影检测训练 (单卡)'
echo '=========================================='
echo '配置文件: configs/sbu/improved_shadow_dual_stream_base_40k.py'
echo '工作目录: work_dirs/improved_shadow_dual_stream_sbu_base'
echo '=========================================='

python tools/train.py configs/sbu/improved_shadow_dual_stream_base_40k.py     --work-dir work_dirs/improved_shadow_dual_stream_sbu_base     --seed 0
