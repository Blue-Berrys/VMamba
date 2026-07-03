# 5070Ti short sweep: boosted feature-level penumbra gate.
#
# The first feature-gate run on server1 kept the detector stable but gamma_p
# changed too slowly to affect the mask. This 5070Ti-only short config uses the
# local 50-series environment/checkpoint naming and raises gamma_p LR to test
# whether the feature gate path has any useful headroom before spending server1
# time on a longer run.

_base_ = "./shadow_icssm_penumbra_feature_gate_4090d_ddp.py"

load_from = "work_dirs/shadow_icssm_bgsir/best_BER_iter_10000.pth"

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)

optim_wrapper = dict(
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "decode_head": dict(lr_mult=0.0),
            "decode_head.boundary_module.penumbra_conv": dict(lr_mult=2.0),
            "decode_head.boundary_module.gamma_p": dict(
                lr_mult=300.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0),
        },
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=2000,
    val_interval=500,
)

work_dir = "./work_dirs/shadow_icssm_penumbra_feature_gate_boost_5070"
