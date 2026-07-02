# 4090D dual-GPU DDP config for BG-SIR-initialized penumbra fine-tuning.
#
# This is the primary server1 experiment config for testing whether the
# penumbra-aware confidence head improves the strong IC-SSM + BG-SIR model,
# instead of measuring the weaker ImageNet-only warm start.

_base_ = "./shadow_icssm_penumbra.py"

load_from = "work_dirs/shadow_icssm_bgsir/best_BER_iter_10000_full.pth"

model_wrapper_cfg = dict(
    type="MMDistributedDataParallel",
    find_unused_parameters=True,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

optim_wrapper = dict(
    accumulative_counts=1,
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=30000,
    val_interval=5000,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=300),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=300, end=30000, eta_min=1e-7),
]

work_dir = "./work_dirs/shadow_icssm_penumbra_bgsir_4090d_ddp"
