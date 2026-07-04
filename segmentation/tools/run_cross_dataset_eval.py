#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
if (THIS_FILE.parents[1] / "configs").exists():
    BASE = THIS_FILE.parents[1]
else:
    BASE = THIS_FILE.parents[2]
sys.path.insert(0, str(BASE))
import configs.sbu.test_cross_dataset as tcd

DATASETS = {
    "SBU": dict(
        img_dir=str(BASE / "data/SBU-shadow/SBU-Test/ShadowImages"),
        mask_dir=str(BASE / "data/SBU-shadow/SBU-Test/ShadowMasks"),
        img_suffix=".jpg",
        mask_suffix=".png",
        mask_loader=tcd.load_mask_binary,
    ),
    "UCF": dict(
        img_dir=str(BASE / "data/UCF/test/img"),
        mask_dir=str(BASE / "data/UCF/test/mask"),
        img_suffix=".jpg",
        mask_suffix=".png",
        mask_loader=tcd.load_mask_ucf,
    ),
    "ISTD": dict(
        img_dir=str(BASE / "data/ISTD_binary/test/img"),
        mask_dir=str(BASE / "data/ISTD_binary/test/mask"),
        img_suffix=".png",
        mask_suffix=".png",
        mask_loader=tcd.load_mask_binary,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--datasets", default="SBU,UCF")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    tcd.USE_TTA = args.tta
    tcd.VIS_N = 0
    out_root = BASE / "work_dirs" / "cross_dataset_20260704" / args.name
    pred_root = out_root / "pred_masks"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"name={args.name}")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"device={args.device} tta={args.tta}")

    model = tcd.init_model(
        str(BASE / args.config),
        str(BASE / args.checkpoint),
        device=args.device,
    )
    model.eval()
    results = {}
    for ds_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        ds = DATASETS[ds_name]
        img_count = 0
        if os.path.isdir(ds["img_dir"]):
            img_count = len([
                f for f in os.listdir(ds["img_dir"])
                if f.lower().endswith(ds["img_suffix"]) and "Zone" not in f
            ])
        print(f"\n=== {ds_name} ({img_count} images) ===")
        if img_count == 0:
            print(f"skip {ds_name}: no images")
            continue
        pred_dir = pred_root / ds_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        result = tcd.evaluate_dataset(
            [model],
            ds["img_dir"],
            ds["mask_dir"],
            ds["img_suffix"],
            ds["mask_suffix"],
            ds["mask_loader"],
            ds_name,
            args.device,
            vis_dir=None,
            pred_dir=str(pred_dir),
        )
        if result:
            results[ds_name] = result

    print("\nSUMMARY")
    print("Dataset\tBER\tFPR\tFNR\tF1")
    for name, result in results.items():
        print(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
                name,
                result["BER"],
                result["FPR"],
                result["FNR"],
                result["F1"],
            )
        )


if __name__ == "__main__":
    main()
