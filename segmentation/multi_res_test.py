#!/usr/bin/env python3
"""
多分辨率测试脚本 — VMamba-Shadow
测试 416/512/640/768 四档分辨率的 BER + GPU显存 + GMACs
运行:
  conda activate vim
  cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
  python multi_res_test.py
"""
import os, sys
import numpy as np
import cv2
import torch
from pathlib import Path

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SEG_DIR)
os.chdir(_SEG_DIR)

# ── 注册自定义模块 ────────────────────────────────────────────
try:
    import model  # noqa: F401  registers MM_VSSM
except Exception:
    pass
try:
    from ic_ssm_head import ICShadowHead  # noqa: F401
except Exception:
    pass
try:
    from sbu_dataset import SBUDataset  # noqa: F401
    from ber_metric import BERMetric    # noqa: F401
    from transforms.sbu_label_transform import SBULabelTransform  # noqa: F401
except Exception:
    pass

from mmseg.apis import init_model

# ── 配置 ──────────────────────────────────────────────────────
CONFIG = 'configs/sbu/shadow_icssm_refine.py'
CKPT   = 'work_dirs/shadow_icssm_refine/best_BER_iter_25000.pth'
SBU_IMG  = 'data/SBU-shadow/SBU-Test/ShadowImages'
SBU_MASK = 'data/SBU-shadow/SBU-Test/ShadowMasks'
RESOLUTIONS = [416, 512, 640, 768]
DEVICE = 'cuda:0'

_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,   57.375], dtype=np.float32)

# ── 数据 ─────────────────────────────────────────────────────
def load_pairs(img_dir, mask_dir):
    pairs = []
    for ip in sorted(Path(img_dir).glob('*')):
        if ip.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        for ext in ('.png', '.bmp', '.jpg'):
            mp = Path(mask_dir) / (ip.stem + ext)
            if mp.exists():
                pairs.append((str(ip), str(mp)))
                break
    return pairs

def preprocess(img_bgr, size):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()

def compute_ber(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    FN = (~pred &  gt).sum()
    FP = ( pred & ~gt).sum()
    FNR = FN / max(gt.sum(), 1)
    FPR = FP / max((~gt).sum(), 1)
    return 0.5 * (FNR + FPR) * 100

# ── FLOPs ─────────────────────────────────────────────────────
def get_flops_params(model, res):
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, (3, res, res),
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return macs * 2 / 1e9, params / 1e6   # GFLOPs, Params(M)
    except Exception as e:
        print(f'  [FLOPs] {e}')
        return None, None

# ── 主循环 ────────────────────────────────────────────────────
def main():
    mdl = init_model(CONFIG, CKPT, device=DEVICE)
    mdl.eval()

    pairs = load_pairs(SBU_IMG, SBU_MASK)
    assert pairs, f'No images found in {SBU_IMG}'
    print(f'SBU-Test: {len(pairs)} images\n')

    results = []
    for res in RESOLUTIONS:
        print(f'{"="*50}')
        print(f'Resolution: {res}x{res}')
        torch.cuda.reset_peak_memory_stats(DEVICE)
        bers = []

        with torch.no_grad():
            for img_path, mask_path in pairs:
                img_bgr = cv2.imread(img_path)
                x = preprocess(img_bgr, res).to(DEVICE)

                meta = {'img_shape': (res, res),
                         'ori_shape': (res, res),
                         'scale_factor': 1.0,
                         'flip': False}
                result = mdl.encode_decode(x, [meta])
                if result.shape[1] == 2:
                    pred_bin = (result[0,1] > result[0,0]).cpu().numpy()
                else:
                    pred_bin = (torch.sigmoid(result[0,0]) > 0.5).cpu().numpy()

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)
                gt_bin = mask >= 128
                bers.append(compute_ber(pred_bin, gt_bin))

        ber_mean = np.mean(bers)
        gpu_mem  = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
        flops, params = get_flops_params(mdl, res)

        print(f'  BER     : {ber_mean:.4f}%')
        print(f'  GPU Mem : {gpu_mem:.2f} GB')
        if flops:
            print(f'  Params  : {params:.1f} M')
            print(f'  GFLOPs  : {flops:.1f} G')
        results.append({'res': res, 'ber': ber_mean, 'mem': gpu_mem, 'flops': flops, 'params': params})

    print(f'\n{"="*50}')
    ps = f'{results[0]["params"]:.1f}M' if results[0]['params'] else 'N/A'
    print(f'Params: {ps} (fixed across resolutions)\n')
    print(f'{"Res":>6}  {"BER(%)":>8}  {"Mem(GB)":>8}  {"GFLOPs":>8}')
    print('-' * 40)
    for r in results:
        fs = f'{r["flops"]:.1f}' if r['flops'] else 'N/A'
        print(f'{r["res"]:>6}  {r["ber"]:>8.4f}  {r["mem"]:>8.2f}  {fs:>8}')

    out = Path('work_dirs/shadow_icssm_refine/multi_res_results.txt')
    with open(out, 'w') as f:
        f.write(f'VMamba-Shadow Multi-Resolution Test\nCkpt: {CKPT}\n\n')
        ps = f'{results[0]["params"]:.1f}M' if results[0]['params'] else 'N/A'
        f.write(f'Params: {ps}\n\n')
        f.write(f'{"Res":>6}  {"BER(%)":>8}  {"Mem(GB)":>8}  {"GFLOPs":>8}\n')
        f.write('-' * 40 + '\n')
        for r in results:
            fs = f'{r["flops"]:.1f}' if r['flops'] else 'N/A'
            f.write(f'{r["res"]:>6}  {r["ber"]:>8.4f}  {r["mem"]:>8.2f}  {fs:>8}\n')
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
