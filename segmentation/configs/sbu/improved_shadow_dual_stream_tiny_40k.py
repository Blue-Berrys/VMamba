# 改进的双流VMamba-Tiny阴影检测 - SBU数据集训练配置
# ===============================================================

_base_ = [
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(imports=['model', 'mmseg.models'], allow_failed_imports=False)

# 模型配置 - Tiny版本
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=0,
        size_divisor=32,
    ),
    backbone=dict(
        type='MM_ShadowDualStreamVSSM_Reg',
        depths=[2, 2, 9, 2],      # Tiny配置
        dims=[96, 192, 384, 768], # Tiny配置
        drop_path_rate=0.2,
        in_chans=3,
        out_indices=(0, 1, 2, 3),
        use_bidirectional_attn=True,  # 双向Cross-Attention融合
        use_shadow_map=True,          # 使用阴影候选图引导
        pretrained=None,
        frozen_stages=-1,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],  # Tiny通道数
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,  # Stage3通道数
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 训练配置
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

work_dir = './work_dirs/improved_shadow_dual_stream_sbu_tiny'
