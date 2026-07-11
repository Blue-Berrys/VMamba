#!/usr/bin/env python3
"""Build method-independent SBU hard/soft shadow subsets from RGB and GT."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or values.shape != weights.shape:
        raise ValueError("values and weights must be non-empty aligned arrays")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    positive = weights > 0
    if not positive.any():
        raise ValueError("at least one weight must be positive")
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    threshold = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")),
                values.size - 1)
    return float(values[index])


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


def measure_luminance_transition(
    offsets: np.ndarray,
    luminance: np.ndarray,
    endpoint_radius: float,
) -> dict[str, float] | None:
    """Measure an unclipped 20-80% shadow transition from one RGB profile."""
    offsets = np.asarray(offsets, dtype=np.float64)
    luminance = np.asarray(luminance, dtype=np.float64)
    inside = luminance[offsets <= -endpoint_radius]
    outside = luminance[offsets >= endpoint_radius]
    if inside.size == 0 or outside.size == 0:
        return None
    inside_level = float(np.median(inside))
    outside_level = float(np.median(outside))
    contrast = outside_level - inside_level
    if contrast <= 1e-6:
        return None
    confidence = np.clip(
        (outside_level - luminance) / contrast, 0.0, 1.0)
    if np.ptp(confidence) < 0.6:
        return None
    t80 = _decreasing_crossing(offsets, confidence, 0.8)
    t20 = _decreasing_crossing(offsets, confidence, 0.2)
    if t80 is None or t20 is None or t20 <= t80:
        return None
    return {
        "width": t20 - t80,
        "contrast": contrast,
        "monotonicity": float(np.mean(np.diff(confidence) <= 1e-3)),
    }


def select_extreme_groups(
    rows: list[dict[str, object]],
    fraction: float,
) -> dict[str, list[dict[str, object]]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between 0 and 0.5")
    count = max(1, int(round(len(rows) * fraction)))
    ordered = sorted(
        rows, key=lambda row: (float(row["softness_score"]), str(row["sample"])))
    return {
        "hard": ordered[:count],
        "soft": list(reversed(ordered[-count:])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--profile-radius", type=int, default=24)
    parser.add_argument("--profile-stride", type=int, default=8)
    parser.add_argument("--min-contrast", type=float, default=0.03)
    parser.add_argument("--min-monotonicity", type=float, default=0.65)
    return parser.parse_args()


def rgb_luminance(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    rgb = np.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def sample_line(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.remap(
        image.astype(np.float32),
        xs.astype(np.float32).reshape(1, -1),
        ys.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).reshape(-1)


def measure_boundary_widths(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int,
    stride: int,
    min_contrast: float,
    min_monotonicity: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    import cv2
    from scipy.ndimage import gaussian_filter1d

    luminance = rgb_luminance(image)
    mask_u8 = mask.astype(np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    grad_y, grad_x = np.gradient(outside - inside)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    height, width = mask.shape
    widths, reliabilities = [], []
    candidates = 0
    for contour in contours:
        for x, y in contour[:, 0, :][::max(stride, 1)]:
            nx, ny = float(grad_x[y, x]), float(grad_y[y, x])
            norm = np.hypot(nx, ny)
            if norm < 1e-6:
                continue
            nx, ny = nx / norm, ny / norm
            xs, ys = x + offsets * nx, y + offsets * ny
            if (xs.min() < 1 or xs.max() >= width - 1 or
                    ys.min() < 1 or ys.max() >= height - 1):
                continue
            candidates += 1
            profile = gaussian_filter1d(
                sample_line(luminance, xs, ys), sigma=1.0, mode="nearest")
            result = measure_luminance_transition(
                offsets, profile, endpoint_radius=0.68 * radius)
            if (result is None or result["contrast"] < min_contrast or
                    result["monotonicity"] < min_monotonicity):
                continue
            widths.append(result["width"])
            reliabilities.append(
                result["contrast"] * result["monotonicity"])
    return (np.asarray(widths, dtype=np.float64),
            np.asarray(reliabilities, dtype=np.float64), candidates)


def find_mask(mask_dir: Path, stem: str) -> Path:
    matches = sorted(
        path for path in mask_dir.glob(f"{stem}.*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not matches:
        raise FileNotFoundError(f"no mask found for {stem}")
    return matches[0]


def save_gallery_asset(image, mask: np.ndarray, output: Path) -> None:
    import cv2
    from PIL import Image

    rgb = np.asarray(image.convert("RGB").resize((384, 288)))
    resized_mask = cv2.resize(
        mask.astype(np.uint8), (384, 288), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(resized_mask, cv2.MORPH_GRADIENT, kernel) > 0
    overlay = rgb.copy()
    overlay[resized_mask > 0] = (
        0.72 * overlay[resized_mask > 0] +
        0.28 * np.array([35, 180, 95])).astype(np.uint8)
    overlay[boundary] = np.array([230, 55, 45], dtype=np.uint8)
    panel = np.concatenate([rgb, overlay], axis=1)
    Image.fromarray(panel).save(output, quality=88, optimize=True)


def write_gallery(
    groups: dict[str, list[dict[str, object]]],
    output_dir: Path,
) -> None:
    cards = []
    for group in ("soft", "hard"):
        cards.append(f'<h2 id="{group}">{group.title()} candidates</h2>')
        cards.append('<section class="grid">')
        for row in groups[group]:
            sample = html.escape(str(row["sample"]))
            cards.append(
                '<article class="card" data-group="%s" data-sample="%s">'
                '<img src="assets/%s.jpg" alt="%s">'
                '<div class="meta"><strong>%s</strong>'
                '<span>score %.4f</span><span>q50 %.4f</span>'
                '<span>q90 %.4f</span><span>reliability %.3f</span></div>'
                '<label><input type="checkbox" class="reject"> Exclude</label>'
                '</article>' % (
                    group, sample, sample, sample, sample,
                    row["softness_score"], row["width_q50"], row["width_q90"],
                    row["mean_reliability"],
                ))
        cards.append('</section>')
    body = "\n".join(cards)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SBU soft-shadow split review</title>
<style>
body{{font:14px Arial,sans-serif;margin:24px;background:#f4f5f6;color:#1e252b}}
header{{position:sticky;top:0;background:#fff;padding:12px 16px;border:1px solid #ccd2d7;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ccd2d7;padding:8px;border-radius:6px}}
.card.rejected{{opacity:.35}} img{{display:block;width:100%;height:auto}}
.meta{{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:8px 0}}
button{{padding:7px 12px}} h2{{margin-top:28px}}
</style></head><body>
<header><b>SBU automatic hard/soft split review</b>
<button id="export">Export review JSON</button>
<span>Left: RGB. Right: GT overlay and boundary.</span></header>
{body}
<script>
const boxes=[...document.querySelectorAll('.reject')];
boxes.forEach(box=>box.addEventListener('change',()=>box.closest('.card').classList.toggle('rejected',box.checked)));
document.getElementById('export').onclick=()=>{{
 const data={{soft:[],hard:[]}};
 document.querySelectorAll('.card').forEach(card=>{{if(!card.querySelector('.reject').checked)data[card.dataset.group].push(card.dataset.sample)}});
 const blob=new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='sbu_soft_split_reviewed.json';link.click();
}};
</script></body></html>"""
    (output_dir / "index.html").write_text(page)


def main() -> None:
    import cv2
    from PIL import Image

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = args.output_dir / "assets"
    asset_dir.mkdir(exist_ok=True)
    image_paths = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        mask_path = find_mask(args.mask_dir, image_path.stem)
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        resized_image = np.asarray(image.resize(
            (args.image_size, args.image_size), Image.Resampling.BILINEAR))
        resized_mask = cv2.resize(
            mask.astype(np.uint8), (args.image_size, args.image_size),
            interpolation=cv2.INTER_NEAREST) > 0
        widths, reliabilities, candidates = measure_boundary_widths(
            resized_image, resized_mask, args.profile_radius,
            args.profile_stride, args.min_contrast, args.min_monotonicity)
        if widths.size < 4:
            continue
        row = {
            "sample": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "softness_score": weighted_quantile(widths, reliabilities, 0.75),
            "width_q50": weighted_quantile(widths, reliabilities, 0.50),
            "width_q90": weighted_quantile(widths, reliabilities, 0.90),
            "mean_reliability": float(reliabilities.mean()),
            "reliable_profiles": int(widths.size),
            "candidate_profiles": int(candidates),
            "valid_profile_ratio": float(widths.size / max(candidates, 1)),
        }
        rows.append(row)
        save_gallery_asset(image, mask, asset_dir / f"{image_path.stem}.jpg")

    if not rows:
        raise RuntimeError("no valid SBU samples were scored")
    groups = select_extreme_groups(rows, args.fraction)
    membership = {
        row["sample"]: group for group, selected in groups.items()
        for row in selected
    }
    for row in rows:
        row["group"] = membership.get(row["sample"], "middle")
    csv_path = args.output_dir / "sbu_softness_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["softness_score"],
                                reverse=True))
    split = {
        "protocol": {
            "source": "RGB and binary GT only; no model predictions",
            "score": "reliability-weighted q75 raw luminance 20-80 width",
            "fraction": args.fraction,
            "image_size": args.image_size,
            "profile_radius": args.profile_radius,
            "profile_stride": args.profile_stride,
            "min_contrast": args.min_contrast,
            "min_monotonicity": args.min_monotonicity,
        },
        "images": len(rows),
        "soft": [row["sample"] for row in groups["soft"]],
        "hard": [row["sample"] for row in groups["hard"]],
    }
    (args.output_dir / "sbu_soft_hard_split.json").write_text(
        json.dumps(split, indent=2) + "\n")
    write_gallery(groups, args.output_dir)
    print(json.dumps({
        "images": len(rows),
        "soft": len(groups["soft"]),
        "hard": len(groups["hard"]),
        "soft_score_range": [groups["soft"][-1]["softness_score"],
                             groups["soft"][0]["softness_score"]],
        "hard_score_range": [groups["hard"][0]["softness_score"],
                             groups["hard"][-1]["softness_score"]],
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
