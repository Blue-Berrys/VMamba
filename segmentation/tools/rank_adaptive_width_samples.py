#!/usr/bin/env python3
"""Rank SBU test images by reliable teacher penumbra width."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from transforms.sbu_label_transform import RefineAdaptivePenumbraAnnTransform


def main():
    image_dir = Path("data/SBU-shadow/SBU-Test/ShadowImages")
    mask_dir = Path("data/SBU-shadow/SBU-Test/ShadowMasks")
    teacher = RefineAdaptivePenumbraAnnTransform(
        reduce_zero_label=False,
        band_width=12,
        min_width=1.0,
        max_width=16.0,
        reliability_min=0.05,
        contrast_tau=0.12,
    )
    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        out = teacher.transform(dict(
            img=image,
            seg_map_path=str(mask_path),
            reduce_zero_label=False,
            seg_fields=[],
        ))
        weight = out["gt_soft_weight_map"]
        support = weight > 0.5
        if support.sum() < 100:
            continue
        width = out["gt_penumbra_width_map"]
        rows.append((float(width[support].mean()), int(support.sum()), image_path.stem))
    for row in sorted(rows, reverse=True)[:20]:
        print(f"{row[0]:.4f} {row[1]:6d} {row[2]}")
    print("-- narrow --")
    for row in sorted(rows)[:20]:
        print(f"{row[0]:.4f} {row[1]:6d} {row[2]}")


if __name__ == "__main__":
    main()
