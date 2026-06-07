#!/usr/bin/env python3
"""
FLOPs/Params/Mem 对比脚本 — VMamba-Shadow vs SegFormer-B2
仅测效率指标，不需要 checkpoint 或数据集
运行:
  conda activate vim
  cd /home/xjx/CodeProject/PycharmProject/VMamba/segmentation
  python flops_comparison.py
"""
import os, sys, torch
from pathlib import Path

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SEG_DIR)
os.chdir(_SEG_DIR)

# 注册自定义模块
try:
    import model  # noqa  registers MM_VSSM
except Exception:
    pass
try:
    from ic_ssm_head import ICShadowHead  # noqa
except Exception:
    pass

from mmengine.config import Config
from mmseg.models import build_segmentor


DEVICE = 'cuda:0'
RESOLUTIONS = [416, 512, 640, 768]

# ── SegFormer-B2 内联配置（不依赖外部文件）────────────────────
SEGFORMER_CFG = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
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
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ── VMamba-Shadow 配置 ────────────────────────────────────────
VMAMBA_CONFIG = 'configs/sbu/shadow_icssm_refine.py'

def measure(model_cfg_or_obj, res, label):
    """返回 (params_M, gflops, mem_GB)"""
    from mmseg.registry import MODELS as MMSEG_MODELS
    from mmengine.config import ConfigDict

    if isinstance(model_cfg_or_obj, dict):
        cfg_dict = dict(model_cfg_or_obj)
    else:
        cfg_dict = dict(model_cfg_or_obj.model)

    # data_preprocessor 不参与前向计算，去掉避免注册问题
    cfg_dict.pop('data_preprocessor', None)
    # ptflops 直接传 tensor，不走 train_cfg/test_cfg
    cfg_dict.pop('train_cfg', None)
    cfg_dict.pop('test_cfg', None)

    model = MMSEG_MODELS.build(ConfigDict(cfg_dict))

    model.eval().to(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    # FLOPs + Params via thop
    try:
        from thop import profile as thop_profile
        dummy_in = torch.randn(1, 3, res, res).to(DEVICE)
        macs, params = thop_profile(model, inputs=(dummy_in,), verbose=False)
        gflops = macs * 2 / 1e9
        params_m = params / 1e6
    except Exception as e:
        print(f'  [{label}@{res}] thop error: {e}')
        # fallback: count params manually
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = None

    # GPU Mem: dummy forward
    try:
        with torch.no_grad():
            dummy = torch.randn(1, 3, res, res, device=DEVICE)
            meta = [{'img_shape': (res, res), 'ori_shape': (res, res),
                     'scale_factor': 1.0, 'flip': False}]
            _ = model.encode_decode(dummy, meta)
        mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    except Exception as e:
        print(f'  [{label}@{res}] forward error: {e}')
        mem = None

    del model
    torch.cuda.empty_cache()
    return params_m, gflops, mem


def main():
    results = {'VMamba-Shadow': [], 'SegFormer-B2': []}

    # ── VMamba-Shadow ─────────────────────────────────────────
    print('=' * 55)
    print('VMamba-Shadow (IC-SSM, O(N))')
    cfg = Config.fromfile(VMAMBA_CONFIG)
    # disable pretrained to avoid download
    if hasattr(cfg.model, 'backbone'):
        cfg.model.backbone.pretrained = None
    for res in RESOLUTIONS:
        params_m, gflops, mem = measure(cfg, res, 'VMamba')
        results['VMamba-Shadow'].append((res, params_m, gflops, mem))
        print(f'  {res:>4}×{res:<4}  Params={params_m:.1f}M  GFLOPs={gflops:.1f}  Mem={mem:.2f}GB')

    # ── SegFormer-B2 ──────────────────────────────────────────
    print('=' * 55)
    print('SegFormer-B2 (Mix-Transformer, O(N²/R²))')
    for res in RESOLUTIONS:
        params_m, gflops, mem = measure(SEGFORMER_CFG, res, 'SegFormer')
        results['SegFormer-B2'].append((res, params_m, gflops, mem))
        print(f'  {res:>4}×{res:<4}  Params={params_m:.1f}M  GFLOPs={gflops:.1f}  Mem={mem:.2f}GB')

    # ── 汇总 ─────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'{"Model":<18} {"Res":>5}  {"Params(M)":>10}  {"GFLOPs":>8}  {"Mem(GB)":>8}')
    print('-' * 60)
    for name, rows in results.items():
        for res, p, f, m in rows:
            ps = f'{p:.1f}' if p else 'N/A'
            fs = f'{f:.1f}' if f else 'N/A'
            ms = f'{m:.2f}' if m else 'N/A'
            print(f'{name:<18} {res:>5}  {ps:>10}  {fs:>8}  {ms:>8}')
        print()

    # FLOPs 增长比（以 416 为基准）
    print('GFLOPs scaling ratio (base=416):')
    for name, rows in results.items():
        base = rows[0][2]
        if base:
            ratios = [f'{r[2]/base:.2f}x' if r[2] else 'N/A' for r in rows]
            print(f'  {name}: {" | ".join(ratios)}')

    # 保存
    out = Path('work_dirs/flops_comparison.txt')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        f.write(f'{"Model":<18} {"Res":>5}  {"Params(M)":>10}  {"GFLOPs":>8}  {"Mem(GB)":>8}\n')
        f.write('-' * 60 + '\n')
        for name, rows in results.items():
            for res, p, fl, m in rows:
                f.write(f'{name:<18} {res:>5}  {p or 0:>10.1f}  {fl or 0:>8.1f}  {m or 0:>8.2f}\n')
            f.write('\n')
    print(f'\nSaved to {out}')

if __name__ == '__main__':
    main()
