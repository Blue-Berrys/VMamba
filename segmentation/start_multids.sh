#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vim
cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
mkdir -p work_dirs/shadow_icssm_multids
nohup python tools/train.py configs/sbu/shadow_icssm_multids.py   > work_dirs/shadow_icssm_multids/train.log 2>&1 &
echo "Multi-dataset training started, PID=$!, log=work_dirs/shadow_icssm_multids/train.log"
