# Continue the current best soft0.29/bw4 penumbra gate with lower LR.
#
# Purpose: test whether the 5k sweep is schedule-limited. This keeps the
# detector mostly frozen and continues from the best 4000-iter checkpoint.

_base_ = "./shadow_icssm_penumbra_softmask_outer_soft029_bw4_gate_sbu_refine_4090d_ddp.py"

load_from = (
    "work_dirs/shadow_icssm_penumbra_softmask_outer_soft029_bw4_gpu0_5k/"
    "best_BER_iter_4000.pth"
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=500)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3,
                    save_best="BER", rule="less"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3, by_epoch=False, begin=0, end=200),
    dict(type="CosineAnnealingLR", by_epoch=False, begin=200,
         end=10000, eta_min=1e-8),
]

work_dir = (
    "./work_dirs/"
    "shadow_icssm_penumbra_softmask_outer_soft029_bw4_lowlr10k_sbu_refine_4090d_ddp"
)
