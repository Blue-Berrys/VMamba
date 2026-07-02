#!/usr/bin/env python3
"""Evaluate shadow-detection slices beyond global BER.

Metrics:
  - soft-boundary band BER/F1: pixels within a signed-distance band.
  - dark-distractor FPR: non-shadow dark pixels outside the boundary band.
  - global BER/F1 for sanity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _distance(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype("uint8")
    if mask_u8.max() == 0:
        return np.full(mask_u8.shape, -1e6, dtype=np.float32)
    if mask_u8.min() == 1:
        return np.full(mask_u8.shape, 1e6, dtype=np.float32)

    try:
        import cv2

        dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    except Exception:
        from scipy import ndimage

        dist_in = ndimage.distance_transform_edt(mask_u8)
        dist_out = ndimage.distance_transform_edt(1 - mask_u8)
    return (dist_in - dist_out).astype(np.float32)


def _load_binary(path: Path, threshold: int = 127) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > threshold


def _load_luma(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _counts(pred: np.ndarray, gt: np.ndarray,
            region: np.ndarray) -> tuple[int, int, int, int]:
    pred_r = pred[region]
    gt_r = gt[region]
    tp = int(np.logical_and(pred_r, gt_r).sum())
    fp = int(np.logical_and(pred_r, ~gt_r).sum())
    fn = int(np.logical_and(~pred_r, gt_r).sum())
    tn = int(np.logical_and(~pred_r, ~gt_r).sum())
    return tp, fp, fn, tn


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-9
    pos_err = fn / (tp + fn + eps)
    neg_err = fp / (tn + fp + eps)
    ber = 0.5 * (pos_err + neg_err) * 100.0
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)
    return {
        "BER": ber,
        "F1": f1,
        "FPR": fpr,
        "precision": precision,
        "recall": recall,
    }


def _match_file(directory: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"{stem}*"))
    return matches[0] if matches else None


def evaluate(pred_dir: Path, gt_dir: Path, image_dir: Path,
             band_width: float, dark_percentile: float,
             threshold: int) -> dict[str, dict[str, float]]:
    totals = {
        "global": [0, 0, 0, 0],
        "boundary_band": [0, 0, 0, 0],
        "dark_distractor": [0, 0, 0, 0],
    }
    num_images = 0

    for gt_path in sorted(gt_dir.iterdir()):
        if gt_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        pred_path = _match_file(pred_dir, gt_path.stem)
        img_path = _match_file(image_dir, gt_path.stem)
        if pred_path is None or img_path is None:
            continue

        gt = _load_binary(gt_path)
        pred = _load_binary(pred_path, threshold=threshold)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127

        signed = _distance(gt)
        boundary_band = np.abs(signed) <= band_width

        luma = _load_luma(img_path)
        if luma.shape != gt.shape:
            luma = np.asarray(Image.fromarray(luma.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.BILINEAR), dtype=np.float32)

        non_shadow = ~gt
        non_boundary = ~boundary_band
        dark_pool = luma[non_shadow & non_boundary]
        if dark_pool.size:
            dark_thr = np.percentile(dark_pool, dark_percentile)
            dark_distractor = non_shadow & non_boundary & (luma <= dark_thr)
        else:
            dark_distractor = np.zeros_like(gt, dtype=bool)

        for key, region in (
            ("global", np.ones_like(gt, dtype=bool)),
            ("boundary_band", boundary_band),
            ("dark_distractor", dark_distractor),
        ):
            counts = _counts(pred, gt, region)
            totals[key] = [a + b for a, b in zip(totals[key], counts)]
        num_images += 1

    result = {key: _metrics(*counts) for key, counts in totals.items()}
    result["meta"] = {"images": float(num_images)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--band-width", default=8.0, type=float)
    parser.add_argument("--dark-percentile", default=20.0, type=float)
    parser.add_argument("--threshold", default=127, type=int)
    args = parser.parse_args()

    result = evaluate(
        args.pred_dir,
        args.gt_dir,
        args.image_dir,
        args.band_width,
        args.dark_percentile,
        args.threshold,
    )
    for key, values in result.items():
        if key == "meta":
            print(f"images: {int(values['images'])}")
            continue
        print(
            f"{key}: BER={values['BER']:.4f} "
            f"F1={values['F1']:.4f} FPR={values['FPR']:.4f} "
            f"P={values['precision']:.4f} R={values['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
