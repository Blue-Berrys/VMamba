# 5070Ti single-GPU probe for outer-side penumbra full-decoder adaptation.
#
# This mirrors the 4090D DDP full-decoder config but removes the distributed
# wrapper and uses a smaller batch so the 16 GB 5070Ti can test whether the
# wider trainable decoder range helps beyond the gate-only plateau.

_base_ = "./shadow_icssm_penumbra_softmask_outer_fullft_sbu_refine_4090d_ddp.py"

model_wrapper_cfg = None

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=3000,
    val_interval=500,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=300),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=300, end=3000, eta_min=1e-7),
]

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_fullft_sbu_refine_5070"
)
