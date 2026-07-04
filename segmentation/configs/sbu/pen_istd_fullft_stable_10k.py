# Full IG-PaSCL / penumbra-aware ISTD fine-tuning from the slice-balanced SBU
# checkpoint. This tests whether the checkpoint with better boundary/dark slices
# transfers better to ISTD in-domain fine-tuning than the global-BER best.

_base_ = "./pen_istd_fullft_global_10k.py"

load_from = (
    "work_dirs/shadow_icssm_penumbra_softmask_outer_soft029_bw4_gpu0_5k/"
    "best_BER_iter_4000.pth"
)

work_dir = "work_dirs/pen_istd_fullft_stable_10k"
