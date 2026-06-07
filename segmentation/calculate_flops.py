#!/usr/bin/env python3
"""
FLOPs 计算脚本 - VMamba-Shadow
支持 416x416 和 512x512 两种输入尺寸
"""

import torch
import torch.nn as nn
from mmseg.models import build_segmentor
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import sys
from ptflops import get_model_complexity_info


def count_flops(config_path, input_size=(512, 512), print_detail=True):
    """
    计算模型的 FLOPs 和参数量

    Args:
        config_path: 配置文件路径
        input_size: 输入尺寸 (height, width)
        print_detail: 是否打印详细信息

    Returns:
        dict: {'params': 参数量(M), 'flops': FLOPs(G), 'fps': 推理速度}
    """
    # 加载配置
    cfg = Config.fromfile(config_path)

    # 构建模型
    model = build_segmentor(cfg.model)
    model.eval()

    if print_detail:
        print(f"\n{'='*60}")
        print(f"Config: {config_path}")
        print(f"Input size: {input_size[0]} x {input_size[1]}")
        print(f"{'='*60}\n")

    # 使用 ptflops 计算 FLOPs
    # MMSeg 的模型输入是 dict: {'inputs': tensor, 'data_samples': list}
    # 我们需要包装一下
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # x: [B, 3, H, W]
            return self.model.decode_head(None, x)

    wrapper = ModelWrapper(model)

    # 计算 FLOPs（不包括数据预处理）
    flops, params = get_model_complexity_info(
        wrapper,
        (3, input_size[0], input_size[1]),  # (C, H, W)
        as_strings=False,
        print_per_layer_stat=print_detail,
        verbose=print_detail,
    )

    # 转换单位
    params_m = params / 1e6
    flops_g = flops / 1e9

    if print_detail:
        print(f"\n{'='*60}")
        print(f"Parameters: {params_m:.2f} M")
        print(f"FLOPs:      {flops_g:.2f} G")
        print(f"{'='*60}\n")

    return {
        'params': params_m,
        'flops': flops_g,
        'input_size': input_size
    }


def calculate_fps(config_path, input_size=(512, 512), num_runs=100, warmup=10):
    """
    计算推理速度 (FPS)

    Args:
        config_path: 配置文件路径
        input_size: 输入尺寸
        num_runs: 推理次数
        warmup: 预热次数

    Returns:
        float: FPS 值
    """
    import time

    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 创建假输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.decode_head(None, dummy_input)

    # 同步并计时
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.decode_head(None, dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed = end_time - start_time

    fps = num_runs / elapsed

    print(f"FPS: {fps:.2f} (on {device})")

    return fps


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_flops.py <config_path> [input_size]")
        print("Example:")
        print("  python calculate_flops.py configs/sbu/shadow_icssm_abl_v2_full.py")
        print("  python calculate_flops.py configs/sbu/shadow_icssm_abl_v2_full.py 416")
        sys.exit(1)

    config_path = sys.argv[1]

    # 解析输入尺寸
    if len(sys.argv) >= 3:
        size = int(sys.argv[2])
        input_size = (size, size)
    else:
        input_size = (512, 512)  # 默认

    # 检查 ptflops 是否安装
    try:
        import ptflops
    except ImportError:
        print("Please install ptflops: pip install ptflops")
        sys.exit(1)

    # 计算 FLOPs
    result = count_flops(config_path, input_size=input_size, print_detail=True)

    # 计算 FPS（可选）
    if '--fps' in sys.argv:
        fps = calculate_fps(config_path, input_size=input_size)
        result['fps'] = fps

    # 打印 LaTeX 表格格式
    print("\n" + "="*60)
    print("LaTeX format for Table 5:")
    print("="*60)
    print(f"VMamba-Shadow (Ours) & {result['params']:.1f} & {result['flops']:.1f} & ", end="")
    if 'fps' in result:
        print(f"{result['fps']:.1f} & ", end="")
    else:
        print("\\TODO{} & ", end="")
    print("\\textbf{2.87} \\\\")


if __name__ == '__main__':
    main()
