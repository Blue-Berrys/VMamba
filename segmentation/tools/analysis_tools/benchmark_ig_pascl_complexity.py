#!/usr/bin/env python3
"""Benchmark the active inference graphs used by IG-PaSCL.

The current research branch contains later adaptive-width experiments in the
same boundary module.  This benchmark deliberately excludes that abandoned
width path and measures three paper variants at batch size one:

1. Host: channel-mean illumination reference + binary boundary branch.
2. +BG-SIR: corrected illumination reference + binary boundary branch.
3. IG-PaSCL: BG-SIR + binary boundary and penumbra-confidence branches.

MACs are the operations counted by fvcore.  We also expose the common
``2 x MACs`` FLOP convention.  VMamba selective-scan kernels are reported as
unsupported, so both absolute values are lower bounds; measured latency is the
primary runtime result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        default=(
            "configs/sbu/"
            "shadow_icssm_penumbra_softmask_outer_soft029_bw4_"
            "gate_sbu_refine_4090d_ddp.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "work_dirs/shadow_icssm_penumbra_softmask_outer_"
            "soft029_bw4_gpu0_5k/best_BER_iter_4000.pth"
        ),
    )
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def boundary_only_forward(self, x: torch.Tensor):
    boundary_logits = self.boundary_conv(x)
    boundary_attn = torch.sigmoid(boundary_logits)
    x_out = x + self.gamma_b * boundary_attn * x
    return x_out, boundary_logits, None, None


def penumbra_forward(self, x: torch.Tensor):
    boundary_logits = self.boundary_conv(x)
    penumbra_logits = self.penumbra_conv(x)
    boundary_attn = torch.sigmoid(boundary_logits)
    penumbra_attn = torch.sigmoid(penumbra_logits)
    x_out = (
        x
        + self.gamma_b * boundary_attn * x
        + self.gamma_p * penumbra_attn * x
    )
    return x_out, boundary_logits, penumbra_logits, None


def active_parameter_count(model: torch.nn.Module, variant: str) -> int:
    excluded = (
        "decode_head.boundary_module.width_conv",
        "decode_head.boundary_module.gamma_width",
        "decode_head.penumbra_logit_scale_raw",
    )
    if variant != "IG-PaSCL":
        excluded += (
            "decode_head.boundary_module.penumbra_conv",
            "decode_head.boundary_module.gamma_p",
        )
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith(excluded)
    )


def build_variant(config_path: Path, checkpoint_path: Path, variant: str):
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmseg.registry import MODELS

    cfg = Config.fromfile(str(config_path))
    cfg.model.backbone.pretrained = None
    cfg.model.decode_head.use_bg_sir = variant != "Host"
    cfg.model.decode_head.penumbra_refine_logits = False
    cfg.model.pop("data_preprocessor", None)
    cfg.model.pop("train_cfg", None)
    cfg.model.pop("test_cfg", None)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=False)

    boundary_module = model.decode_head.boundary_module
    if variant == "IG-PaSCL":
        boundary_module.forward = types.MethodType(
            penumbra_forward, boundary_module
        )
    else:
        boundary_module.forward = types.MethodType(
            boundary_only_forward, boundary_module
        )
    return model


def count_flops(model: torch.nn.Module, dummy: torch.Tensor):
    from fvcore.nn import FlopCountAnalysis

    analysis = FlopCountAnalysis(model, dummy)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    counted_flops = analysis.total()
    unsupported = {
        str(name): int(count)
        for name, count in analysis.unsupported_ops().items()
    }
    return counted_flops, unsupported


def benchmark_latency(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    warmup: int,
    iters: int,
    repeats: int,
):
    model.eval().cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize()

        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                model(dummy)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iters)

    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    return samples, peak_gib


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    # Project imports register the custom backbone, head, transforms, and hooks.
    import model  # noqa: F401
    import ic_ssm_head  # noqa: F401

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")
    torch.backends.cudnn.benchmark = True
    dummy = torch.randn(1, 3, args.size, args.size, device=device)

    checkpoint = (repo / args.checkpoint).resolve()
    config = (repo / args.config).resolve()
    output = args.output or (
        repo / "work_dirs" / "ig_pascl_complexity_runtime.json"
    )

    report = {
        "protocol": {
            "input": [1, 3, args.size, args.size],
            "precision": "FP32",
            "warmup": args.warmup,
            "iterations_per_repeat": args.iters,
            "repeats": args.repeats,
            "timing": "CUDA events with synchronization",
            "complexity_note": (
                "fvcore-counted MACs plus the common 2x-MAC FLOP convention; "
                "absolute values are lower bounds when selective-scan kernels "
                "are unsupported"
            ),
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cudnn": torch.backends.cudnn.version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "config": str(config),
        "checkpoint": str(checkpoint),
        "variants": {},
    }

    for variant in ("Host", "+BG-SIR", "IG-PaSCL"):
        model = build_variant(config, checkpoint, variant)
        model.eval().cuda()
        counted_macs, unsupported = count_flops(model, dummy)
        samples, peak_gib = benchmark_latency(
            model, dummy, args.warmup, args.iters, args.repeats
        )
        median_ms = statistics.median(samples)
        row = {
            "active_params": active_parameter_count(model, variant),
            "fvcore_counted_macs": int(counted_macs),
            "gflops_2x_macs": 2.0 * counted_macs / 1e9,
            "unsupported_ops": unsupported,
            "latency_ms_samples": samples,
            "latency_ms_median": median_ms,
            "latency_ms_mean": statistics.mean(samples),
            "latency_ms_stdev": statistics.stdev(samples)
            if len(samples) > 1
            else 0.0,
            "fps_from_median": 1000.0 / median_ms,
            "peak_inference_memory_gib": peak_gib,
        }
        report["variants"][variant] = row
        print(
            f"{variant:10s} "
            f"Params={row['active_params']/1e6:.3f}M "
            f"MACs={row['fvcore_counted_macs']/1e9:.3f}G "
            f"FLOPs(2x)={row['gflops_2x_macs']:.3f}G "
            f"latency={median_ms:.3f}ms "
            f"FPS={row['fps_from_median']:.2f} "
            f"peak={peak_gib:.3f}GiB"
        )
        del model
        torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
