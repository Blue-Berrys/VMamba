#!/usr/bin/env python3
"""Export binary shadow predictions and penumbra confidence maps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    try:
        from ic_ssm_head import ICShadowHead  # noqa: F401
        from istd_dataset import ISTDDataset, ISTDLabelTransform  # noqa: F401
    except ImportError:
        pass
    try:
        from ber_metric import BERMetric  # noqa: F401
        from sbu_dataset import SBUDataset  # noqa: F401
        from transforms.sbu_label_transform import SBULabelTransform  # noqa: F401
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--penumbra-dir")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def sample_name(data_sample, index: int) -> str:
    img_path = getattr(data_sample, "img_path", None)
    if img_path:
        return Path(img_path).stem
    return f"{index:06d}"


def save_u8(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    register_project_modules()

    cfg = Config.fromfile(args.config)
    if "model_wrapper_cfg" in cfg:
        cfg.pop("model_wrapper_cfg")
    cfg.launcher = "none"
    cfg.work_dir = str(Path(args.pred_dir).parent / "_export_work")
    cfg.load_from = None
    if args.split == "val":
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator

    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    pred_dir = Path(args.pred_dir)
    penumbra_dir = Path(args.penumbra_dir) if args.penumbra_dir else None
    pred_dir.mkdir(parents=True, exist_ok=True)
    if penumbra_dir is not None:
        penumbra_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with torch.no_grad():
        for data_batch in runner.test_dataloader:
            outputs = model.test_step(data_batch)
            decode_head = model.decode_head
            penumbra_logits = getattr(decode_head, "_penumbra_logits", None)
            penumbra_prob = None
            if penumbra_dir is not None and penumbra_logits is not None:
                out_shape = outputs[0].pred_sem_seg.data.shape[-2:]
                penumbra_prob = torch.sigmoid(
                    F.interpolate(
                        penumbra_logits,
                        size=out_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                )

            for idx, data_sample in enumerate(outputs):
                name = sample_name(data_sample, exported)
                pred = data_sample.pred_sem_seg.data.squeeze().detach().cpu().numpy()
                save_u8(pred_dir / f"{name}.png", (pred > 0).astype(np.uint8) * 255)

                if penumbra_prob is not None:
                    pen = penumbra_prob[idx, 0].detach().cpu().numpy()
                    save_u8(
                        penumbra_dir / f"{name}.png",
                        np.clip(pen * 255.0, 0, 255).astype(np.uint8),
                    )

                exported += 1
                if args.max_items is not None and exported >= args.max_items:
                    print(f"exported {exported}")
                    return

    print(f"exported {exported}")


if __name__ == "__main__":
    main()
