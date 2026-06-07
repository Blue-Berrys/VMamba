#!/bin/bash
set -e
LOG=/home/xjx/CodeProject/PycharmProject/VMamba/segmentation/work_dirs/abl_v2_queue.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vim
cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation

echo "[$(date)] Starting ablation v2 queue" >> $LOG
echo "[$(date)] abl_v2_base start" >> $LOG
python tools/train.py configs/sbu/shadow_icssm_abl_v2_base.py >> $LOG 2>&1
echo "[$(date)] abl_v2_base done" >> $LOG

echo "[$(date)] abl_v2_tv start" >> $LOG
python tools/train.py configs/sbu/shadow_icssm_abl_v2_tv.py >> $LOG 2>&1
echo "[$(date)] abl_v2_tv done" >> $LOG

echo "[$(date)] abl_v2_bg start" >> $LOG
python tools/train.py configs/sbu/shadow_icssm_abl_v2_bg.py >> $LOG 2>&1
echo "[$(date)] abl_v2_bg done" >> $LOG

echo "[$(date)] abl_v2_full start" >> $LOG
python tools/train.py configs/sbu/shadow_icssm_abl_v2_full.py >> $LOG 2>&1
echo "[$(date)] abl_v2_full done" >> $LOG

echo "[$(date)] All ablation v2 done!" >> $LOG
