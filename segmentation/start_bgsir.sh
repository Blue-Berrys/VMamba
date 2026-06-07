#!/bin/bash
# 等 Tversky 训练完成后，启动 BG-SIR 训练
# 用法: bash start_bgsir.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vim
cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
mkdir -p work_dirs/shadow_icssm_bgsir
nohup python tools/train.py configs/sbu/shadow_icssm_bgsir.py   > work_dirs/shadow_icssm_bgsir/train.log 2>&1 &
echo "BG-SIR training started, PID=$!, log=work_dirs/shadow_icssm_bgsir/train.log"
