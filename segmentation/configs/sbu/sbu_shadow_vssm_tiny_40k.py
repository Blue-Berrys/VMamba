"""
VMamba-Tiny 阴影检测配置文件 - SBU数据集

该配置文件用于在SBU数据集上训练VMamba-Tiny模型进行像素级阴影检测
- 任务: 二分类 (非阴影/阴影)
- 输入尺寸: 416x416 (与现有阴影检测工作保持一致)
- 训练迭代: 40k (4 GPUs)
- 评估指标: BER (Balance Error Rate)
"""

_base_ = [
    '../_base_/models/upernet_vssm_binary.py',  # 二分类UPerNet+VMamba模型
    '../_base_/datasets/sbu.py',                 # SBU数据集配置
    '../_base_/default_runtime.py',              # 默认运行时配置
    '../_base_/schedules/schedule_40k.py'        # 40k训练调度
]

# ================== 模型配置 ==================
# 指定预训练权重路径
# 明确设置 data_preprocessor 的 size 参数，避免与 size_divisor 冲突
# seg_pad_val 设置为 0，避免与标签值 255 冲突
model = dict(
    data_preprocessor=dict(
        size=(416, 416),  # 与 crop_size 保持一致，与现有阴影检测工作保持一致
        seg_pad_val=0  # 改为 0，避免与标签值 255 冲突
    ),
    backbone=dict(
        pretrained="../../ckpts/classification/outs/vssm/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
    )
)

# ================== 运行时配置 ==================
# 可视化后端：启用TensorBoard
vis_backends = [
    dict(type='LocalVisBackend'),             # 本地可视化（保存图片）
    dict(type='TensorboardVisBackend')        # TensorBoard（实时监控）
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 工作目录
work_dir = './work_dirs/sbu_shadow_detection_vssm_tiny'

# ================== 多GPU训练配置 ==================
# 4 GPUs × 4 batch_size = 16 total batch size
# 40k iterations保持与160k相同的总样本数（基于4倍GPU加速）

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,  # 40k iterations
        by_epoch=False)
]

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,      # 最大迭代次数
    val_interval=4000     # 每4000次迭代验证一次
)

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,      # 每4000次迭代保存一次
        max_keep_ckpts=5    # 最多保留5个checkpoint
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# ================== 注册自定义组件 ==================
# 需要在训练前注册SBUDataset和BERMetric
# 在训练脚本中添加以下import:
# from segmentation.sbu_dataset import SBUDataset
# from segmentation.ber_metric import BERMetric

# ================== 使用说明 ==================
# 1. 准备数据集:
#    - 将SBU数据集放置在 data/SBU-shadow/ 目录
#    - 训练集: SBUTrain4KRecoveredSmall/ShadowImages 和 ShadowMasks
#    - 测试集: SBU-Test/ShadowImages 和 ShadowMasks
#
# 2. 训练命令 (4 GPUs):
#    bash tools/dist_train.sh configs/sbu/sbu_shadow_vssm_tiny_40k.py 4
#
# 3. 单GPU训练:
#    python tools/train.py configs/sbu/sbu_shadow_vssm_tiny_40k.py
#
# 4. 评估命令:
#    bash tools/dist_test.sh configs/sbu/sbu_shadow_vssm_tiny_40k.py \\
#        work_dirs/sbu_shadow_detection_vssm_tiny/iter_40000.pth 4
#
# 5. 查看训练曲线:
#    cd work_dirs/sbu_shadow_detection_vssm_tiny
#    tensorboard --logdir=./
