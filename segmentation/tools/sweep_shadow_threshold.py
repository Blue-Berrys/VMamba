#!/usr/bin/env python3
"""Sweep probability thresholds for binary shadow BER.

This is a diagnostic script: it does not change checkpoints or training code.
It evaluates class-1 shadow probability at a list of thresholds and reports
global BER/FPR/FNR/F1 on SBU test.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch


_SEG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SEG_DIR)

try:
    import model  # noqa: F401  Registers VMamba modules and hooks.
except ImportError:
    pass
try:
    from sbu_dataset import SBUDataset  # noqa: F401
    from ber_metric import BERMetric  # noqa: F401
    from transforms.sbu_label_transform import SBULabelTransform  # noqa: F401
except ImportError:
    pass

from mmseg.apis import init_model


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def parse_thresholds(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(x) for x in value.split(":")]
        vals = []
        cur = start
        while cur <= stop + 1e-9:
            vals.append(round(cur, 6))
            cur += step
        return vals
    return [float(x) for x in value.split(",") if x.strip()]


def preprocess(img_bgr: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(img_bgr, (size, size))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


@torch.no_grad()
def shadow_prob(model_obj, img_path: str, device: str, size: int, tta: bool) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    orig_h, orig_w = img_bgr.shape[:2]
    meta = {
        "ori_shape": (orig_h, orig_w),
        "img_shape": (size, size),
        "scale_factor": (size / orig_w, size / orig_h),
        "flip": False,
    }

    tensor = preprocess(img_bgr, size).to(device)
    logits = model_obj.encode_decode(tensor, [meta])
    prob = torch.softmax(logits, dim=1)

    if tta:
        img_flip = cv2.flip(img_bgr, 1)
        flip_tensor = preprocess(img_flip, size).to(device)
        flip_logits = model_obj.encode_decode(flip_tensor, [{**meta, "flip": True}])
        flip_prob = torch.softmax(flip_logits, dim=1)
        flip_prob = torch.flip(flip_prob, dims=[3])
        prob = (prob + flip_prob) / 2.0

    prob_np = prob[0, 1].detach().cpu().numpy()
    return prob_np


def load_mask(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return (arr >= 128).astype(np.uint8)


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    fpr = fp / (fp + tn + 1e-10) * 100.0
    fnr = fn / (fn + tp + 1e-10) * 100.0
    ber = (fpr + fnr) / 2.0
    precision = tp / (tp + fp + 1e-10) * 100.0
    recall = tp / (tp + fn + 1e-10) * 100.0
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {
        "BER": ber,
        "FPR": fpr,
        "FNR": fnr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--thresholds", default="0.35:0.65:0.01")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--img-dir", default="data/SBU-shadow/SBU-Test/ShadowImages")
    parser.add_argument("--mask-dir", default="data/SBU-shadow/SBU-Test/ShadowMasks")
    parser.add_argument("--img-suffix", default=".jpg")
    parser.add_argument("--mask-suffix", default=".png")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    model_obj = init_model(args.config, args.checkpoint, device=args.device)
    model_obj.eval()

    counts = {thr: [0, 0, 0, 0] for thr in thresholds}  # tp, tn, fp, fn
    img_files = sorted(
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith(args.img_suffix) and "Zone" not in f
    )

    used = 0
    for idx, fname in enumerate(img_files, 1):
        stem = fname[: -len(args.img_suffix)]
        img_path = os.path.join(args.img_dir, fname)
        mask_path = os.path.join(args.mask_dir, stem + args.mask_suffix)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.mask_dir, stem + ".png")
        if not os.path.exists(mask_path):
            continue

        prob = shadow_prob(model_obj, img_path, args.device, args.size, args.tta)
        gt = load_mask(mask_path)
        if prob.shape != gt.shape:
            prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt_b = gt.astype(bool)

        for thr in thresholds:
            pred = prob >= thr
            counts[thr][0] += int((pred & gt_b).sum())
            counts[thr][1] += int((~pred & ~gt_b).sum())
            counts[thr][2] += int((pred & ~gt_b).sum())
            counts[thr][3] += int((~pred & gt_b).sum())

        used += 1
        if idx % 100 == 0:
            print(f"processed {idx}/{len(img_files)}", flush=True)

    rows = []
    for thr in thresholds:
        tp, tn, fp, fn = counts[thr]
        row = {"threshold": thr, **compute_metrics(tp, tn, fp, fn)}
        rows.append(row)
    rows.sort(key=lambda x: x["BER"])

    print(f"images={used} tta={args.tta} checkpoint={args.checkpoint}")
    print("threshold,BER,FPR,FNR,F1,Precision,Recall")
    for row in rows:
        print(
            f"{row['threshold']:.4f},"
            f"{row['BER']:.4f},"
            f"{row['FPR']:.4f},"
            f"{row['FNR']:.4f},"
            f"{row['F1']:.4f},"
            f"{row['Precision']:.4f},"
            f"{row['Recall']:.4f}"
        )


if __name__ == "__main__":
    main()
