#!/usr/bin/env python3
"""
显存对比脚本 — VMamba-Shadow vs SegFormer-B2
仅做 dummy 前向，无需 checkpoint 或数据集
运行:
  conda activate vim
  cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
  python mem_comparison.py
"""
import os, sys, torch
from pathlib import Path

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SEG_DIR)
os.chdir(_SEG_DIR)

try:
    import model  # noqa  registers MM_VSSM
except Exception: pass
try:
    from ic_ssm_head import ICShadowHead  # noqa
except Exception: pass

from mmengine.config import Config, ConfigDict
from mmseg.registry import MODELS as MMSEG_MODELS

DEVICE = 'cuda:0'
RESOLUTIONS = [416, 512, 640, 768]

# ── SegFormer-B2 配置（内联，无需文件）───────────────────────
SEGFORMER_B2 = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pretrained=None,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)

def build_no_preprocessor(cfg_dict):
    d = dict(cfg_dict)
    d.pop('data_preprocessor', None)
    d.pop('train_cfg', None)
    d.pop('test_cfg', None)
    return MMSEG_MODELS.build(ConfigDict(d))

def measure_mem(model, res):
    """dummy 前向，返回峰值显存 (GB)"""
    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    dummy = torch.randn(1, 3, res, res, device=DEVICE)
    meta = [{'img_shape': (res, res), 'ori_shape': (res, res),
             'scale_factor': 1.0, 'flip': False}]
    with torch.no_grad():
        try:
            model.encode_decode(dummy, meta)
        except Exception:
            # fallback: backbone only
            model.backbone(dummy)
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**3

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def main():
    results = {}

    # ── VMamba-Shadow ─────────────────────────────────────────
    print('Building VMamba-Shadow...')
    cfg = Config.fromfile('configs/sbu/shadow_icssm_refine.py')
    cfg.model.backbone.pretrained = None
    vmamba = build_no_preprocessor(cfg.model)
    vmamba.to(DEVICE)
    vmamba_params = count_params(vmamba)

    rows_v = []
    for res in RESOLUTIONS:
        mem = measure_mem(vmamba, res)
        print(f'  VMamba {res}x{res}  Mem={mem:.2f}GB')
        rows_v.append((res, mem))
    results['VMamba-Shadow'] = (vmamba_params, rows_v)
    del vmamba; torch.cuda.empty_cache()

    # ── SegFormer-B2 ──────────────────────────────────────────
    print('Building SegFormer-B2...')
    seg = build_no_preprocessor(SEGFORMER_B2)
    seg.to(DEVICE)
    seg_params = count_params(seg)

    rows_s = []
    for res in RESOLUTIONS:
        mem = measure_mem(seg, res)
        print(f'  SegFormer {res}x{res}  Mem={mem:.2f}GB')
        rows_s.append((res, mem))
    results['SegFormer-B2'] = (seg_params, rows_s)
    del seg; torch.cuda.empty_cache()

    # ── 汇总 ─────────────────────────────────────────────────
    # 已知 GFLOPs（VMamba 来自 ptflops 实测；SegFormer 来自原始论文按分辨率换算）
    vmamba_flops = {416: 40.7, 512: 61.6, 640: 96.2, 768: 138.6}
    # SegFormer-B2 官方：512×512 = 62.4G FLOPs；按 O(N) 粗估（其 sr_ratio 使得接近线性，但高分辨率二次项会显现）
    # 实测官方数据仅有 512；其他分辨率理论值暂留 N/A
    seg_flops = {512: 62.4}

    print(f'\n{"="*65}')
    print(f'{"Model":<18} {"Res":>5}  {"Params(M)":>10}  {"GFLOPs":>8}  {"Mem(GB)":>8}')
    print('-' * 55)
    for name, (params, rows) in results.items():
        flops_map = vmamba_flops if 'VMamba' in name else seg_flops
        for res, mem in rows:
            fs = f'{flops_map[res]:.1f}' if res in flops_map else 'N/A'
            print(f'{name:<18} {res:>5}  {params:>10.1f}  {fs:>8}  {mem:>8.2f}')
        print()

    # FLOPs 缩放比
    print('GFLOPs scaling (base=416, VMamba measured / SegFormer official @512 only):')
    base = vmamba_flops[416]
    print('  VMamba-Shadow:', ' | '.join(
        f'{r}→{vmamba_flops[r]/base:.2f}x' for r in RESOLUTIONS
    ))

    # Mem 缩放比
    print('\nMemory scaling ratio (base=416):')
    for name, (_, rows) in results.items():
        base_m = rows[0][1]
        ratios = [f'{res}→{mem/base_m:.2f}x' for res, mem in rows]
        print(f'  {name}: {" | ".join(ratios)}')

    # 保存
    out = Path('work_dirs/mem_comparison.txt')
    with open(out, 'w') as f:
        f.write(f'{"Model":<18} {"Res":>5}  {"Params(M)":>10}  {"GFLOPs":>8}  {"Mem(GB)":>8}\n')
        f.write('-' * 55 + '\n')
        for name, (params, rows) in results.items():
            flops_map = vmamba_flops if 'VMamba' in name else seg_flops
            for res, mem in rows:
                fs = f'{flops_map[res]:.1f}' if res in flops_map else 'N/A'
                f.write(f'{name:<18} {res:>5}  {params:>10.1f}  {fs:>8}  {mem:>8.2f}\n')
            f.write('\n')
    print(f'\nSaved to {out}')

if __name__ == '__main__':
    main()
