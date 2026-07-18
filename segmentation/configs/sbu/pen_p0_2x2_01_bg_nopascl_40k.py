"""P0 2x2 row 01: BG-SIR reference correction, no PaSCL losses."""

_base_ = "./pen_p0_2x2_common_40k.py"

model = dict(decode_head=dict(use_bg_sir=True))

work_dir = "./work_dirs/pen_p0_2x2_01_bg_nopascl_seed20260718_40k"
