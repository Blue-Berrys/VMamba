#!/usr/bin/env python3
"""Evaluate continuous penumbra confidence maps against soft pseudo labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def distance(mask: np.ndarray) -> np.ndarray:
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


def load_binary(path: Path, threshold: int = 127) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > threshold


def load_prob(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def match_file(directory: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"{stem}*"))
    return matches[0] if matches else None


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def evaluate(
    penumbra_dir: Path,
    gt_dir: Path,
    band_width: float,
    tau: float,
) -> dict[str, float]:
    abs_errors: list[np.ndarray] = []
    sq_errors: list[np.ndarray] = []
    pred_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    images = 0

    for gt_path in sorted(gt_dir.iterdir()):
        if gt_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        pred_path = match_file(penumbra_dir, gt_path.stem)
        if pred_path is None:
            continue

        gt = load_binary(gt_path)
        pred = load_prob(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray((pred * 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0

        signed = distance(gt)
        band = np.abs(signed) <= band_width
        if not band.any():
            continue
        target = 1.0 / (1.0 + np.exp(-np.clip(signed / max(tau, 1e-6), -20, 20)))

        p = pred[band]
        t = target[band]
        err = p - t
        abs_errors.append(np.abs(err))
        sq_errors.append(err * err)
        pred_values.append(p)
        target_values.append(t)
        images += 1

    if not abs_errors:
        return {"images": 0.0, "MAE": float("nan"), "RMSE": float("nan"), "Pearson": float("nan")}

    abs_all = np.concatenate(abs_errors)
    sq_all = np.concatenate(sq_errors)
    pred_all = np.concatenate(pred_values)
    target_all = np.concatenate(target_values)
    return {
        "images": float(images),
        "MAE": float(abs_all.mean()),
        "RMSE": float(np.sqrt(sq_all.mean())),
        "Pearson": pearson(pred_all, target_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--penumbra-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--band-width", default=8.0, type=float)
    parser.add_argument("--tau", default=2.0, type=float)
    args = parser.parse_args()

    metrics = evaluate(args.penumbra_dir, args.gt_dir, args.band_width, args.tau)
    print(
        f"images={int(metrics['images'])} "
        f"MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f} "
        f"Pearson={metrics['Pearson']:.6f}"
    )


if __name__ == "__main__":
    main()
