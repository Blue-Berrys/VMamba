#!/usr/bin/env python3
"""Evaluate penumbra width against ISTD shadow/shadow-free image pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


class TransitionMeasurement(NamedTuple):
    width: float
    t80: float
    t20: float
    monotonicity: float


def normalize_attenuation_profile(
    offsets: np.ndarray,
    attenuation: np.ndarray,
    endpoint_radius: float,
) -> tuple[np.ndarray, float]:
    """Normalize a local attenuation profile using its two endpoint regions."""
    offsets = np.asarray(offsets, dtype=np.float64)
    attenuation = np.asarray(attenuation, dtype=np.float64)
    inside = attenuation[offsets <= -endpoint_radius]
    outside = attenuation[offsets >= endpoint_radius]
    if inside.size == 0 or outside.size == 0:
        raise ValueError("profile does not cover both endpoint regions")
    inside_level = float(np.median(inside))
    outside_level = float(np.median(outside))
    contrast = inside_level - outside_level
    if contrast <= 1e-12:
        return np.full_like(attenuation, 0.5), contrast
    normalized = np.clip(
        (attenuation - outside_level) / contrast, 0.0, 1.0)
    return normalized, contrast


def _decreasing_crossing(
    offsets: np.ndarray,
    values: np.ndarray,
    level: float,
) -> float | None:
    for index in range(values.size - 1):
        left, right = values[index], values[index + 1]
        if left >= level >= right:
            if abs(right - left) < 1e-12:
                return float(offsets[index])
            fraction = (level - left) / (right - left)
            return float(offsets[index] + fraction * (
                offsets[index + 1] - offsets[index]))
    return None


def measure_transition_width(
    offsets: np.ndarray,
    confidence: np.ndarray,
) -> TransitionMeasurement | None:
    """Measure the outward 80%-to-20% transition width of a profile."""
    offsets = np.asarray(offsets, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if offsets.ndim != 1 or confidence.shape != offsets.shape:
        raise ValueError("offsets and confidence must be aligned 1D arrays")
    if confidence.size < 3 or np.ptp(confidence) < 0.6:
        return None
    t80 = _decreasing_crossing(offsets, confidence, 0.8)
    t20 = _decreasing_crossing(offsets, confidence, 0.2)
    if t80 is None or t20 is None or t20 <= t80:
        return None
    monotonicity = float(np.mean(np.diff(confidence) <= 1e-3))
    return TransitionMeasurement(t20 - t80, t80, t20, monotonicity)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def binary_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    return {
        "tp": int(np.logical_and(prediction, target).sum()),
        "tn": int(np.logical_and(~prediction, ~target).sum()),
        "fp": int(np.logical_and(prediction, ~target).sum()),
        "fn": int(np.logical_and(~prediction, target).sum()),
    }


def metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "ber": 50.0 * (fpr + fnr),
        "fpr": 100.0 * fpr,
        "fnr": 100.0 * fnr,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--istd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--profile-radius", type=int, default=16)
    parser.add_argument("--profile-stride", type=int, default=8)
    parser.add_argument("--min-contrast", type=float, default=0.04)
    parser.add_argument("--min-monotonicity", type=float, default=0.65)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def register_project_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    try:
        import model  # noqa: F401
    except ImportError:
        pass
    from ber_metric import BERMetric  # noqa: F401
    from ic_ssm_head import ICShadowHead  # noqa: F401
    from istd_dataset import ISTDDataset  # noqa: F401
    from transforms.sbu_label_transform import SBULabelTransform  # noqa: F401


def paired_log_attenuation(shadow_rgb: np.ndarray, clean_rgb: np.ndarray) -> np.ndarray:
    """Return a robust per-pixel log attenuation from an aligned image pair."""
    shadow = shadow_rgb.astype(np.float32) / 255.0
    clean = clean_rgb.astype(np.float32) / 255.0
    shadow = np.where(
        shadow <= 0.04045, shadow / 12.92, ((shadow + 0.055) / 1.055) ** 2.4)
    clean = np.where(
        clean <= 0.04045, clean / 12.92, ((clean + 0.055) / 1.055) ** 2.4)
    return np.median(np.log(clean + 1e-3) - np.log(shadow + 1e-3), axis=2)


def sample_line(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.remap(
        image.astype(np.float32),
        xs.astype(np.float32).reshape(1, -1),
        ys.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).reshape(-1)


def boundary_profiles(
    mask: np.ndarray,
    attenuation: np.ndarray,
    radius: int,
    stride: int,
    min_contrast: float,
    min_monotonicity: float,
) -> list[dict[str, object]]:
    import cv2
    from scipy.ndimage import gaussian_filter1d

    mask_u8 = mask.astype(np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    signed_distance = outside - inside
    grad_y, grad_x = np.gradient(signed_distance)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    height, width = mask.shape
    profiles: list[dict[str, object]] = []

    for contour in contours:
        points = contour[:, 0, :]
        for x, y in points[::max(stride, 1)]:
            nx, ny = float(grad_x[y, x]), float(grad_y[y, x])
            norm = np.hypot(nx, ny)
            if norm < 1e-6:
                continue
            nx, ny = nx / norm, ny / norm
            xs = x + offsets * nx
            ys = y + offsets * ny
            if (xs.min() < 1 or xs.max() >= width - 1 or
                    ys.min() < 1 or ys.max() >= height - 1):
                continue
            raw = sample_line(attenuation, xs, ys)
            raw = gaussian_filter1d(raw, sigma=1.0, mode="nearest")
            normalized, contrast = normalize_attenuation_profile(
                offsets, raw, endpoint_radius=0.65 * radius)
            if contrast < min_contrast:
                continue
            measurement = measure_transition_width(offsets, normalized)
            if measurement is None or measurement.monotonicity < min_monotonicity:
                continue
            profiles.append({
                "x": float(x),
                "y": float(y),
                "nx": nx,
                "ny": ny,
                "offsets": offsets.copy(),
                "physical_confidence": normalized,
                "physical_width": measurement.width,
                "contrast": contrast,
                "monotonicity": measurement.monotonicity,
            })
    return profiles


def resize_tensor_map(value, size: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    resized = functional.interpolate(
        value, size=size, mode="bilinear", align_corners=False)
    return resized[0, 0].detach().float().cpu().numpy()


def main() -> None:
    import cv2
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from PIL import Image

    args = parse_args()
    register_project_modules()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.pop("model_wrapper_cfg", None)
    cfg.launcher = "none"
    cfg.randomness = dict(seed=20260711, deterministic=False)
    cfg.work_dir = str(args.output.parent / "_istd_physical_width_runner")
    cfg.load_from = None
    dataset = cfg.test_dataloader.dataset
    dataset.type = "ISTDDataset"
    dataset.data_root = str(args.istd_root)
    dataset.data_prefix = dict(img_path="test/img", seg_map_path="test/mask")
    dataset.pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(args.image_size, args.image_size),
             keep_ratio=False),
        dict(type="SBULabelTransform", reduce_zero_label=False),
        dict(type="PackSegInputs"),
    ]
    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    image_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    all_physical_confidences: list[float] = []
    all_predicted_confidences: list[float] = []
    with torch.no_grad():
        for image_index, data_batch in enumerate(runner.test_dataloader):
            if args.max_images is not None and image_index >= args.max_images:
                break
            output = model.test_step(data_batch)[0]
            image_path = Path(output.img_path)
            name = image_path.name
            clean_path = args.istd_root / "test" / "test_C" / name
            mask_path = args.istd_root / "test" / "mask" / name
            shadow = np.asarray(Image.open(image_path).convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BILINEAR))
            clean = np.asarray(Image.open(clean_path).convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BILINEAR))
            mask = np.asarray(Image.open(mask_path).convert("L").resize(
                (args.image_size, args.image_size), Image.Resampling.NEAREST)) > 127
            attenuation = paired_log_attenuation(shadow, clean)
            profiles = boundary_profiles(
                mask, attenuation, args.profile_radius, args.profile_stride,
                args.min_contrast, args.min_monotonicity)

            width_prob = resize_tensor_map(
                torch.sigmoid(model.decode_head._width_logits), mask.shape)
            penumbra_prob = resize_tensor_map(
                torch.sigmoid(model.decode_head._penumbra_logits), mask.shape)
            prediction = output.pred_sem_seg.data.squeeze().detach().cpu().numpy()
            prediction = cv2.resize(
                prediction.astype(np.uint8), mask.shape[::-1],
                interpolation=cv2.INTER_NEAREST) > 0

            physical_widths = []
            predicted_widths = []
            physical_confidences = []
            predicted_confidences = []
            for profile in profiles:
                x, y = profile["x"], profile["y"]
                nx, ny = profile["nx"], profile["ny"]
                offsets = profile["offsets"]
                xs = x + offsets * nx
                ys = y + offsets * ny
                predicted_width = 1.0 + 15.0 * float(width_prob[int(y), int(x)])
                predicted_profile = sample_line(penumbra_prob, xs, ys)
                physical_widths.append(float(profile["physical_width"]))
                predicted_widths.append(predicted_width)
                physical_confidences.extend(profile["physical_confidence"])
                predicted_confidences.extend(predicted_profile)
                all_physical_confidences.extend(profile["physical_confidence"])
                all_predicted_confidences.extend(predicted_profile)
                profile_rows.append({
                    "sample": image_path.stem,
                    "x": x,
                    "y": y,
                    "physical_width": profile["physical_width"],
                    "predicted_width": predicted_width,
                    "contrast": profile["contrast"],
                    "monotonicity": profile["monotonicity"],
                })

            if not physical_widths:
                continue
            counts = binary_counts(prediction, mask)
            image_rows.append({
                "sample": image_path.stem,
                "profiles": len(physical_widths),
                "physical_width": float(np.median(physical_widths)),
                "predicted_width": float(np.median(predicted_widths)),
                "profile_width_mae": float(np.mean(np.abs(
                    np.asarray(physical_widths) - np.asarray(predicted_widths)))),
                "penumbra_pearson": correlation(
                    np.asarray(physical_confidences),
                    np.asarray(predicted_confidences)),
                **counts,
            })

    if not image_rows or not profile_rows:
        raise RuntimeError("no reliable physical penumbra profiles were found")

    physical_profile_width = np.asarray([
        row["physical_width"] for row in profile_rows], dtype=np.float64)
    predicted_profile_width = np.asarray([
        row["predicted_width"] for row in profile_rows], dtype=np.float64)
    physical_image_width = np.asarray([
        row["physical_width"] for row in image_rows], dtype=np.float64)
    predicted_image_width = np.asarray([
        row["predicted_width"] for row in image_rows], dtype=np.float64)
    q1, q2 = np.quantile(physical_image_width, [1 / 3, 2 / 3])

    groups = {}
    predicates = {
        "narrow": lambda value: value <= q1,
        "medium": lambda value: q1 < value <= q2,
        "wide": lambda value: value > q2,
    }
    for group, predicate in predicates.items():
        selected = [row for row in image_rows if predicate(row["physical_width"])]
        counts = {key: sum(int(row[key]) for row in selected)
                  for key in ("tp", "tn", "fp", "fn")}
        groups[group] = {
            "images": len(selected),
            "physical_width": float(np.mean([
                row["physical_width"] for row in selected])),
            "predicted_width": float(np.mean([
                row["predicted_width"] for row in selected])),
            **metrics(counts),
        }

    total_counts = {key: sum(int(row[key]) for row in image_rows)
                    for key in ("tp", "tn", "fp", "fn")}
    summary = {
        "checkpoint": str(args.checkpoint),
        "images": len(image_rows),
        "profiles": len(profile_rows),
        "physical_protocol": {
            "pair_signal": "median channel log(clean)-log(shadow)",
            "width": "outward t80-to-t20 transition at 512x512",
            "profile_radius": args.profile_radius,
            "min_contrast": args.min_contrast,
            "min_monotonicity": args.min_monotonicity,
        },
        "width": {
            "profile_pearson": correlation(
                physical_profile_width, predicted_profile_width),
            "profile_spearman": correlation(
                average_ranks(physical_profile_width),
                average_ranks(predicted_profile_width)),
            "profile_mae_px": float(np.mean(np.abs(
                physical_profile_width - predicted_profile_width))),
            "image_pearson": correlation(
                physical_image_width, predicted_image_width),
            "image_spearman": correlation(
                average_ranks(physical_image_width),
                average_ranks(predicted_image_width)),
            "physical_mean_px": float(physical_profile_width.mean()),
            "predicted_mean_px": float(predicted_profile_width.mean()),
        },
        "confidence": {
            "profile_pearson": correlation(
                np.asarray(all_physical_confidences),
                np.asarray(all_predicted_confidences)),
            "mae": float(np.mean(np.abs(
                np.asarray(all_physical_confidences) -
                np.asarray(all_predicted_confidences)))),
        },
        "global": metrics(total_counts),
        "bin_thresholds_px": {"narrow_max": float(q1), "medium_max": float(q2)},
        "groups": groups,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for rows, suffix in ((image_rows, ".images.csv"),
                         (profile_rows, ".profiles.csv")):
        csv_path = args.output.with_suffix(suffix)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
