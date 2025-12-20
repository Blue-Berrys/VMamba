"""
VMamba-Tiny 模型配置文件 - ADE20K 语义分割任务

该配置文件用于在 ADE20K 数据集上训练 VMamba-Tiny 模型，使用 UperNet 作为分割头。
- 输入尺寸: 512x512
- 训练迭代次数: 160k
- Batch size: 4 per GPU (4 GPUs = 16)
- 预期性能: mIoU ~47.9 (SS) / ~48.8 (MS)
"""

# 继承 Swin-Tiny 的基础配置
_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]

# 模型配置
model = dict(
    backbone=dict(
        type='MM_VSSM',  # 使用 VMamba 作为骨干网络
        out_indices=(0, 1, 2, 3),  # 输出4个阶段的特征图，用于多尺度特征融合
        pretrained="../../ckpts/classification/outs/vssm/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth",  # 预训练权重路径
        # 以下参数来自 classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,  # 基础通道数
        depths=(2, 2, 9, 2),  # 每个阶段的block数量 [Stage1, Stage2, Stage3, Stage4]
        ssm_d_state=16,  # 状态空间模型的状态维度
        ssm_dt_rank="auto",  # delta时间步的秩，自动计算
        ssm_ratio=2.0,  # SSM扩展比例
        mlp_ratio=0.0,  # MLP扩展比例 (0.0表示不使用MLP)
        downsample_version="v1",  # 下采样版本
        patchembed_version="v1",  # patch embedding版本
        # forward_type="v0", # 如果需要完全一致的前向传播，可以取消注释
    ),)

# 可视化配置：启用 TensorBoard 可视化后端
vis_backends = [
    dict(type='LocalVisBackend'),  # 本地可视化后端（保存图片）
    dict(type='TensorboardVisBackend')  # TensorBoard后端（实时监控训练曲线）
]
visualizer = dict(
    type='SegLocalVisualizer',  # 分割任务可视化器
    vis_backends=vis_backends,  # 使用的可视化后端列表
    name='visualizer'  # 可视化器名称
)

# 数据加载器配置（可选）
# train_dataloader = dict(batch_size=4)  # 每个GPU的batch size，默认为4

# ================== 多GPU训练优化配置 ==================
# 注意：如果要实现真正的4倍加速（时间缩短到1/4），需要保持总训练样本数不变
# 单卡：160k iterations × 4 samples/iter = 640k samples
# 4卡：40k iterations × 16 samples/iter = 640k samples（相同总样本数）
# 
# 取消下面的注释以启用4倍加速配置（训练时间缩短到1/4）：
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=40000,  # 160k / 4 = 40k（保持总样本数不变）
    val_interval=4000  # 16000 / 4 = 4000
)
# 
# 同时需要调整学习率调度器的结束迭代次数：
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,  # 160k / 4 = 40k
        by_epoch=False)
]
#
# 注意：如果保持160k iterations，4卡会训练4倍的数据量，ETA相同但效果可能更好

