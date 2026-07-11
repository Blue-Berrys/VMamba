#!/usr/bin/env python3
"""Evaluate adaptive-width fidelity and width-stratified shadow metrics."""

from __future__ import annotations

import argparse
import csv
import json
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
    from ber_metric import BERMetric  # noqa: F401
    from ic_ssm_head import ICShadowHead  # noqa: F401
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rank-samples", type=int, default=500_000)
    return parser.parse_args()


def resize_map(value: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    value = F.interpolate(
        value, size=size, mode="bilinear", align_corners=False)
    return value[0, 0].detach().float().cpu().numpy()


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


def add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for key in total:
        total[key] += current[key]


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


def main() -> None:
    args = parse_args()
    register_project_modules()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.pop("model_wrapper_cfg", None)
    cfg.launcher = "none"
    cfg.work_dir = str(args.output.parent / "_width_eval_runner")
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    from transforms.sbu_label_transform import RefineAdaptivePenumbraAnnTransform

    teacher = RefineAdaptivePenumbraAnnTransform(
        reduce_zero_label=False,
        band_width=12,
        min_width=1.0,
        max_width=16.0,
        reliability_min=0.05,
        contrast_tau=0.12,
    )
    rng = np.random.default_rng(20260711)
    sampled_gt: list[np.ndarray] = []
    sampled_pred: list[np.ndarray] = []
    rows = []

    with torch.no_grad():
        for data_batch in runner.test_dataloader:
            output = model.test_step(data_batch)[0]
            image_path = Path(output.img_path)
            mask_path = Path(
                str(image_path).replace("ShadowImages", "ShadowMasks")
            ).with_suffix(".png")
            image = np.asarray(Image.open(image_path).convert("RGB"))
            gt = np.asarray(Image.open(mask_path).convert("L")) > 127

            teacher_output = teacher.transform(dict(
                img=image,
                seg_map_path=str(mask_path),
                reduce_zero_label=False,
                seg_fields=[],
            ))
            weight = teacher_output["gt_soft_weight_map"]
            support = weight > 0.05
            reliable = weight > 0.5
            width_gt = teacher_output["gt_penumbra_width_map"]
            width_pred = resize_map(
                torch.sigmoid(model.decode_head._width_logits), gt.shape)
            penumbra_prob = resize_map(
                torch.sigmoid(model.decode_head._penumbra_logits), gt.shape)
            prediction = (
                output.pred_sem_seg.data.squeeze().detach().cpu().numpy() > 0)

            valid_values = np.flatnonzero(reliable)
            if valid_values.size:
                take = min(
                    valid_values.size,
                    max(1, args.max_rank_samples // max(
                        len(runner.test_dataloader), 1)),
                )
                chosen = rng.choice(valid_values, size=take, replace=False)
                sampled_gt.append(width_gt.ravel()[chosen])
                sampled_pred.append(width_pred.ravel()[chosen])

            support_weight = weight[support].astype(np.float64)
            width_error = np.abs(width_pred[support] - width_gt[support])
            weighted_mae = float(
                np.sum(width_error * support_weight) /
                max(np.sum(support_weight), 1e-12))
            sample_width = float(
                np.average(width_gt[support], weights=support_weight)
                if support.any() else 0.0)
            sample_prediction = float(
                np.average(width_pred[support], weights=support_weight)
                if support.any() else 0.0)
            uncertainty = 4.0 * penumbra_prob * (1.0 - penumbra_prob)

            row = {
                "sample": image_path.stem,
                "teacher_width": sample_width,
                "predicted_width": sample_prediction,
                "width_mae_norm": weighted_mae,
                "width_mae_px": 15.0 * weighted_mae,
                "reliable_pixels": int(reliable.sum()),
                "support_pixels": int(support.sum()),
                "mean_uncertainty_support": float(
                    uncertainty[support].mean() if support.any() else 0.0),
                **binary_counts(prediction, gt),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("no test samples were evaluated")

    if not sampled_gt:
        raise RuntimeError("no reliable penumbra-width pixels were found")
    sampled_gt_array = np.concatenate(sampled_gt)
    sampled_pred_array = np.concatenate(sampled_pred)
    pearson = correlation(sampled_gt_array, sampled_pred_array)
    spearman = correlation(
        average_ranks(sampled_gt_array),
        average_ranks(sampled_pred_array),
    )

    teacher_means = np.asarray([row["teacher_width"] for row in rows])
    q1, q2 = np.quantile(teacher_means, [1.0 / 3.0, 2.0 / 3.0])
    groups = {
        "narrow": lambda value: value <= q1,
        "medium": lambda value: q1 < value <= q2,
        "wide": lambda value: value > q2,
    }
    grouped = {}
    for name, predicate in groups.items():
        counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
        selected = [row for row in rows if predicate(row["teacher_width"])]
        for row in selected:
            add_counts(counts, {key: row[key] for key in counts})
        grouped[name] = {
            "images": len(selected),
            "teacher_width_mean": float(np.mean([
                row["teacher_width"] for row in selected])),
            "predicted_width_mean": float(np.mean([
                row["predicted_width"] for row in selected])),
            "width_mae_px": float(np.mean([
                row["width_mae_px"] for row in selected])),
            **metrics(counts),
        }

    total_counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    for row in rows:
        add_counts(total_counts, {key: row[key] for key in total_counts})
    summary = {
        "checkpoint": str(args.checkpoint),
        "images": len(rows),
        "width": {
            "mae_norm": float(np.mean([
                row["width_mae_norm"] for row in rows])),
            "mae_px": float(np.mean([
                row["width_mae_px"] for row in rows])),
            "pearson": pearson,
            "spearman": spearman,
            "sampled_pixels": int(sampled_gt_array.size),
            "teacher_mean": float(sampled_gt_array.mean()),
            "prediction_mean": float(sampled_pred_array.mean()),
        },
        "global": metrics(total_counts),
        "bin_thresholds": {
            "narrow_max": float(q1),
            "medium_max": float(q2),
        },
        "groups": grouped,
    }

    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"per-image rows: {csv_path}")


if __name__ == "__main__":
    main()
