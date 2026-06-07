_base_ = [
    '../_base_/models/upernet_vssm_binary.py',
    '../_base_/datasets/sbu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=0,
        size=(416, 416)
    ),
    backbone=dict(
        type='ShadowDualStreamVSSM',
        depths=[2, 2, 27, 2],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.6,
        local_channels=128,
        gate_type='channel_spatial'
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
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
        in_channels=512,
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
    test_cfg=dict(mode='whole')
)

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True, sampler=dict(type='InfiniteSampler', shuffle=True))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=5), logger=dict(interval=50, log_metric_by_epoch=False))

param_scheduler = [dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=40000, by_epoch=False)]

optim_wrapper = dict(_delete_=True, clip_grad=dict(max_norm=1, norm_type=2), optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005), type='AmpOptimWrapper', accumulative_counts=4)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = './work_dirs/shadow_dual_stream_sbu_base'
