# Evaluation-only wrapper for the existing ISTD-trained checkpoint.
# server1 currently has ISTD images under data/ISTD_binary rather than
# data/ISTD_Dataset, so this keeps the model/training recipe unchanged and only
# redirects validation/test data paths.

_base_ = "./shadow_icssm_istd.py"

val_dataloader = dict(
    dataset=dict(
        data_root="data/ISTD_binary",
        data_prefix=dict(
            img_path="test/img",
            seg_map_path="test/mask",
        ),
    ),
)

test_dataloader = val_dataloader

work_dir = "./work_dirs/shadow_icssm_istd_binary_eval"
