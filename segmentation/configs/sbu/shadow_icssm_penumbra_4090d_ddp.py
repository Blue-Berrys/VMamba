# 4090D dual-GPU DDP override for shadow_icssm_penumbra.py.
#
# Use this when the BG-SIR fine-tuning checkpoint is unavailable or incomplete
# on the 4090D server. It starts from the ImageNet-pretrained VMamba-Base
# backbone that is already present on server1, then trains the penumbra-aware
# head on SBU.

_base_ = "./shadow_icssm_penumbra.py"

load_from = None

model = dict(
    backbone=dict(
        pretrained="pretrained/vssm_base_0229_ckpt_epoch_237.pth",
    ),
)

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

work_dir = "./work_dirs/shadow_icssm_penumbra_4090d_ddp"
