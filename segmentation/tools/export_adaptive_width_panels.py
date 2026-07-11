#!/usr/bin/env python3
"""Export aligned real-image panels for adaptive penumbra-width figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from PIL import Image


def register_project_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    try:
        import model  # noqa: F401
    except ImportError:
        pass
    from ic_ssm_head import ICShadowHead, ShadowBoundaryModule  # noqa: F401
    from ber_metric import BERMetric  # noqa: F401
    from sbu_dataset import SBUDataset  # noqa: F401
    from transforms.sbu_label_transform import (  # noqa: F401
        PackSegInputsWithSoft,
        RefineAdaptivePenumbraAnnTransform,
        SBULabelTransform,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def save_gray(path: Path, value: np.ndarray) -> None:
    value = np.clip(value, 0.0, 1.0)
    Image.fromarray(np.round(value * 255.0).astype(np.uint8)).save(path)


def save_heat(path: Path, value: np.ndarray, vmin=0.0, vmax=1.0) -> None:
    scaled = np.clip((value - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.round(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).save(path)


def save_error(path: Path, prediction: np.ndarray, gt: np.ndarray) -> None:
    canvas = np.full((*gt.shape, 3), 245, dtype=np.uint8)
    canvas[(prediction == 1) & (gt == 1)] = (40, 40, 40)
    canvas[(prediction == 1) & (gt == 0)] = (220, 61, 50)
    canvas[(prediction == 0) & (gt == 1)] = (45, 116, 210)
    Image.fromarray(canvas).save(path)


def resize_map(value: torch.Tensor, size) -> np.ndarray:
    resized = F.interpolate(
        value, size=size, mode="bilinear", align_corners=False)
    return resized[0, 0].detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    register_project_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.pop("model_wrapper_cfg", None)
    cfg.launcher = "none"
    cfg.work_dir = str(args.output_dir / "_runner")
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    selected = None
    with torch.no_grad():
        for data_batch in runner.test_dataloader:
            outputs = model.test_step(data_batch)
            if Path(outputs[0].img_path).stem == args.sample:
                selected = outputs[0]
                break
    if selected is None:
        raise RuntimeError(f"sample not found in test split: {args.sample}")

    image_path = Path(selected.img_path)
    mask_path = Path(str(image_path).replace("ShadowImages", "ShadowMasks")).with_suffix(".png")
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask_u8 = np.asarray(Image.open(mask_path).convert("L"))
    gt = mask_u8 > 127

    from ic_ssm_head import ShadowBoundaryModule
    from transforms.sbu_label_transform import RefineAdaptivePenumbraAnnTransform

    teacher = RefineAdaptivePenumbraAnnTransform(
        reduce_zero_label=False,
        band_width=12,
        min_width=1.0,
        max_width=16.0,
        reliability_min=0.05,
        contrast_tau=0.12,
    ).transform(dict(
        img=image,
        seg_map_path=str(mask_path),
        reduce_zero_label=False,
        seg_fields=[],
    ))

    decode_head = model.decode_head
    out_size = gt.shape
    width_prediction = torch.sigmoid(decode_head._width_logits)
    confidence_prediction = torch.sigmoid(decode_head._penumbra_logits)
    width_prediction = resize_map(width_prediction, out_size)
    confidence_prediction = resize_map(confidence_prediction, out_size)
    prediction = selected.pred_sem_seg.data.squeeze().detach().cpu().numpy() > 0

    signed = ShadowBoundaryModule.get_soft_penumbra_gt(
        torch.from_numpy(gt.astype(np.float32))[None, None],
        band_width=12,
        tau=2.0,
    )[2][0, 0].numpy()
    signed_display = np.clip((signed + 12.0) / 24.0, 0.0, 1.0)

    Image.fromarray(image).save(args.output_dir / "input.png")
    save_gray(args.output_dir / "gt.png", gt.astype(np.float32))
    save_heat(args.output_dir / "signed_distance.png", signed_display)
    support = teacher["gt_soft_weight_map"] > 0.05
    width_target_display = np.where(
        support, teacher["gt_penumbra_width_map"], 0.0)
    width_prediction_display = np.where(support, width_prediction, 0.0)
    save_heat(args.output_dir / "width_target.png", width_target_display)
    save_heat(args.output_dir / "width_prediction.png", width_prediction_display)
    save_gray(args.output_dir / "soft_target.png", teacher["gt_soft_seg_map"])
    save_gray(args.output_dir / "confidence_prediction.png", confidence_prediction)
    save_gray(args.output_dir / "reliability.png", teacher["gt_soft_weight_map"])
    save_gray(args.output_dir / "prediction.png", prediction.astype(np.float32))
    save_error(args.output_dir / "error.png", prediction.astype(np.uint8), gt.astype(np.uint8))
    print(f"exported {args.sample} to {args.output_dir}")


if __name__ == "__main__":
    main()
