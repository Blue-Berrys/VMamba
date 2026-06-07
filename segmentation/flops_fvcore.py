#!/usr/bin/env python3
"""
用 fvcore 准确测量 GFLOPs — VMamba-Shadow vs SegFormer-B2
不需要数据集，用 dummy tensor 即可
运行:
  conda activate vim
  cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
  python flops_fvcore.py
"""
import os, sys, torch
from pathlib import Path
from fvcore.nn import FlopCountAnalysis, flop_count_table

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SEG_DIR)
os.chdir(_SEG_DIR)

try:
    import model  # noqa
except Exception: pass
try:
    from ic_ssm_head import ICShadowHead  # noqa
except Exception: pass

from mmengine.config import Config, ConfigDict
from mmseg.registry import MODELS as MMSEG_MODELS

DEVICE = 'cuda:0'
RESOLUTIONS = [416, 512, 640, 768]

SEGFORMER_B2 = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3, embed_dims=64, num_stages=4,
        num_layers=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        pretrained=None,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512], in_index=[0,1,2,3],
        channels=256, dropout_ratio=0.1, num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)

def build_model(cfg_dict):
    d = dict(cfg_dict)
    for k in ('data_preprocessor', 'train_cfg', 'test_cfg'):
        d.pop(k, None)
    return MMSEG_MODELS.build(ConfigDict(d))

def measure(model, res, label):
    model.eval().to(DEVICE)
    dummy = torch.randn(1, 3, res, res, device=DEVICE)

    # fvcore: 只需 backbone+neck+head 的 forward
    # 用 backbone forward 作为主体（decode_head 单独测）
    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    total = flops.total()
    uncounted = flops.unsupported_ops()  # 未统计的算子列表

    params = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = total * 2 / 1e9  # MACs → FLOPs

    # 显存
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        meta = [{'img_shape': (res,res), 'ori_shape': (res,res),
                 'scale_factor': 1.0, 'flip': False}]
        try:
            model.encode_decode(dummy, meta)
        except Exception:
            model.backbone(dummy)
    mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3

    print(f'  {label} {res}×{res}: {gflops:.1f}G FLOPs | {params:.1f}M params | {mem:.2f}GB mem')
    if uncounted:
        ops_str = ', '.join(list(uncounted.keys())[:5])
        print(f'    ⚠ uncounted ops: {ops_str}')

    return gflops, params, mem

def main():
    all_results = {}

    # ── VMamba-Shadow ─────────────────────────────────────────
    print('='*55)
    print('VMamba-Shadow (IC-SSM)')
    cfg = Config.fromfile('configs/sbu/shadow_icssm_refine.py')
    cfg.model.backbone.pretrained = None
    vmamba = build_model(cfg.model)
    rows = []
    for res in RESOLUTIONS:
        g, p, m = measure(vmamba, res, 'VMamba')
        rows.append((res, g, p, m))
    all_results['VMamba-Shadow'] = rows
    del vmamba; torch.cuda.empty_cache()

    # ── SegFormer-B2 ──────────────────────────────────────────
    print('='*55)
    print('SegFormer-B2 (MiT)')
    seg = build_model(SEGFORMER_B2)
    rows = []
    for res in RESOLUTIONS:
        g, p, m = measure(seg, res, 'SegFormer')
        rows.append((res, g, p, m))
    all_results['SegFormer-B2'] = rows
    del seg; torch.cuda.empty_cache()

    # ── 汇总 ─────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'{"Model":<18} {"Res":>5}  {"Params(M)":>10}  {"GFLOPs":>8}  {"Mem(GB)":>8}')
    print('-'*58)
    for name, rows in all_results.items():
        for res, g, p, m in rows:
            print(f'{name:<18} {res:>5}  {p:>10.1f}  {g:>8.1f}  {m:>8.2f}')
        print()

    # 缩放比
    print('GFLOPs scaling ratio (relative to 416):')
    for name, rows in all_results.items():
        base = rows[0][1]
        s = ' | '.join(f'{res}:{g/base:.2f}x' for res,g,_,_ in rows)
        print(f'  {name}: {s}')

    # 理论缩放参考
    print('\nTheoretical: O(N) → 1.00x|1.52x|2.37x|3.40x  (N=H×W)')
    print('Theoretical: O(N²)→ 1.00x|2.30x|5.60x|11.6x  (N²)')

    out = Path('work_dirs/flops_fvcore.txt')
    with open(out, 'w') as f:
        f.write(f'{"Model":<18} {"Res":>5}  {"Params(M)":>10}  {"GFLOPs":>8}  {"Mem(GB)":>8}\n')
        f.write('-'*58+'\n')
        for name, rows in all_results.items():
            for res,g,p,m in rows:
                f.write(f'{name:<18} {res:>5}  {p:>10.1f}  {g:>8.1f}  {m:>8.2f}\n')
            f.write('\n')
        f.write('\nGFLOPs scaling ratio (base=416):\n')
        for name, rows in all_results.items():
            base = rows[0][1]
            s = ' | '.join(f'{res}:{g/base:.2f}x' for res,g,_,_ in rows)
            f.write(f'  {name}: {s}\n')
    print(f'\nSaved to {out}')

if __name__ == '__main__':
    main()
