"""
Cross-dataset evaluation script
使用最优 checkpoint 在 SBU / UCF / ISTD 三个数据集上测试

运行 (在 segmentation 目录下):
  conda activate vim
  python configs/sbu/test_cross_dataset.py
"""

import os, sys
import numpy as np
import cv2
import torch

VIS_N = 99999       # 每个数据集保存前 N 张可视化（0 = 不保存）
VIS_SIZE = (512, 512)   # 单格大小
USE_TTA = True      # 水平翻转 TTA
ENSEMBLE_CKPT = ''   # 不做 ensemble，只用单模型

# ── 注册自定义模块（与 tools/train.py 保持一致）──
_SEG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SEG_DIR)
try:
    import model  # 注册 VMamba (MM_VSSM 等)
except ImportError:
    pass
try:
    from ic_ssm_head import ICShadowHead
    from istd_dataset import ISTDDataset, ISTDLabelTransform
except ImportError:
    pass
try:
    from sbu_dataset import SBUDataset
    from ber_metric import BERMetric
    from transforms.sbu_label_transform import SBULabelTransform
except ImportError:
    pass

from mmseg.apis import init_model


# ──────────────────────────── metric ─────────────────────────────
def compute_ber(pred_bin, gt_bin):
    """pred_bin, gt_bin: numpy uint8 (H,W), shadow=1, bg=0"""
    pred = pred_bin.astype(bool)
    gt   = gt_bin.astype(bool)
    shadow_px    = gt.sum()
    nonshadow_px = (~gt).sum()
    TP = (pred  &  gt).sum()
    FP = (pred  & ~gt).sum()
    FN = (~pred &  gt).sum()
    FNR = FN / (shadow_px    + 1e-8) * 100
    FPR = FP / (nonshadow_px + 1e-8) * 100
    BER = (FNR + FPR) / 2
    precision = TP / (TP + FP + 1e-8) * 100
    recall    = TP / (TP + FN + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return dict(BER=BER, FPR=FPR, FNR=FNR, Precision=precision,
                Recall=recall, F1=f1)


# ──────────────────────────── mask loaders ───────────────────────
def load_mask_binary(path):
    """SBU / ISTD test: {0,255} → {0,1}"""
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (arr >= 128).astype(np.uint8)


def load_mask_ucf(path):
    """UCF: soft/continuous 0-255, threshold at 127 (standard practice)"""
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (arr >= 128).astype(np.uint8)


# ──────────────────────────── inference ──────────────────────────
_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,   57.375], dtype=np.float32)


def _preprocess(img_bgr, size=512):
    """BGR uint8 → normalized RGB float tensor (1,3,H,W)"""
    img = cv2.resize(img_bgr, (size, size))
    img = img[:, :, ::-1].astype(np.float32)   # BGR→RGB
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def _encode_decode(models, tensor, meta):
    """在多个模型上做推理并平均 softmax logit"""
    logits = None
    for m in models:
        out = m.encode_decode(tensor, [meta])   # [1, C, H, W]
        out = torch.softmax(out, dim=1)
        logits = out if logits is None else logits + out
    return logits / len(models)


@torch.no_grad()
def infer_one(models, img_path, device, size=512):
    """
    TTA: 原图 + 水平翻转, logit 平均
    Ensemble: 多模型 logit 平均
    返回 (pred uint8 H×W, orig_h, orig_w)
    """
    img_bgr = cv2.imread(img_path)
    orig_h, orig_w = img_bgr.shape[:2]

    meta_orig = {
        'ori_shape':    (orig_h, orig_w),
        'img_shape':    (size, size),
        'scale_factor': (size / orig_w, size / orig_h),
        'flip': False,
    }

    t_orig = _preprocess(img_bgr, size).to(device)
    logit = _encode_decode(models, t_orig, meta_orig)

    if USE_TTA:
        img_flip = cv2.flip(img_bgr, 1)           # 水平翻转
        t_flip = _preprocess(img_flip, size).to(device)
        meta_flip = {**meta_orig, 'flip': True}
        logit_flip = _encode_decode(models, t_flip, meta_flip)
        # 翻转预测结果再翻转回来
        logit_flip = torch.flip(logit_flip, dims=[3])
        logit = (logit + logit_flip) / 2

    pred = logit[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
    return pred, orig_h, orig_w


# ──────────────────────────── evaluate ───────────────────────────
def make_vis(img_bgr, gt, pred, ber):
    """拼接 [原图 | GT | 预测 | 误差图] 四格，并标注 BER"""
    H, W = VIS_SIZE
    img_r = cv2.resize(img_bgr, (W, H))

    def mask_to_bgr(m):
        gray = (m * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gt_r   = cv2.resize(gt.astype(np.uint8),   (W, H), interpolation=cv2.INTER_NEAREST)
    pred_r = cv2.resize(pred.astype(np.uint8),  (W, H), interpolation=cv2.INTER_NEAREST)

    # 误差图: TP=白, FP=红, FN=蓝, TN=黑
    err = np.zeros((H, W, 3), dtype=np.uint8)
    g, p = gt_r.astype(bool), pred_r.astype(bool)
    err[ g &  p] = (255, 255, 255)   # TP 白
    err[~g &  p] = (0,   0,   200)   # FP 红 (BGR)
    err[ g & ~p] = (200, 0,   0  )   # FN 蓝 (BGR)

    row = np.hstack([img_r, mask_to_bgr(gt_r), mask_to_bgr(pred_r), err])
    label = f"BER={ber:.2f}%  |  orig   |   GT    |   pred  |   error"
    cv2.putText(row, label, (10, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return row


def evaluate_dataset(models, img_dir, mask_dir,
                     img_suffix, mask_suffix,
                     mask_loader, name, device='cuda:0',
                     vis_dir=None, pred_dir=None):
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(img_suffix) and 'Zone' not in f
    ])
    if not img_files:
        print(f"[{name}] No images found in {img_dir}")
        return None

    # 全局聚合（与 BERMetric 完全一致: 先累加 TP/TN/FP/FN，再统一计算）
    total_tp = total_tn = total_fp = total_fn = 0
    n = 0

    for i, fname in enumerate(img_files):
        img_path  = os.path.join(img_dir, fname)
        stem      = fname[: -len(img_suffix)]
        mask_path = os.path.join(mask_dir, stem + mask_suffix)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, stem + '.png')
        if not os.path.exists(mask_path):
            continue

        pred_seg, orig_h, orig_w = infer_one(models, img_path, device)
        gt = mask_loader(mask_path)

        # pred 尺寸对齐 gt（模型已 resize 回 ori_shape，二者通常相同）
        if pred_seg.shape != gt.shape:
            pred_seg = cv2.resize(pred_seg, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        pred = pred_seg.astype(bool)
        gt_b = gt.astype(bool)
        tp = int((pred  &  gt_b).sum())
        tn = int((~pred & ~gt_b).sum())
        fp = int((pred  & ~gt_b).sum())
        fn = int((~pred &  gt_b).sum())
        total_tp += tp; total_tn += tn
        total_fp += fp; total_fn += fn
        n += 1

        # 保存预测 mask（拉伸回原始尺寸）
        if pred_dir:
            pred_orig = cv2.resize(pred_seg, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, stem + '.png'),
                        (pred_orig * 255).astype(np.uint8))

        # 保存可视化
        if vis_dir and n <= VIS_N:
            img_bgr  = cv2.imread(img_path)
            img_fnr  = fn / (fn + tp + 1e-8) * 100
            img_fpr  = fp / (fp + tn + 1e-8) * 100
            img_ber  = (img_fnr + img_fpr) / 2
            vis = make_vis(img_bgr, gt, pred_seg, img_ber)
            out_path = os.path.join(vis_dir, f'{n:04d}_{stem}.jpg')
            cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if (i + 1) % 50 == 0:
            _fpr = total_fp / (total_fp + total_tn + 1e-10) * 100
            _fnr = total_fn / (total_fn + total_tp + 1e-10) * 100
            print(f"  [{name}] {i+1}/{len(img_files)}  running BER={(_fpr+_fnr)/2:.4f}%")

    fpr = total_fp / (total_fp + total_tn + 1e-10) * 100
    fnr = total_fn / (total_fn + total_tp + 1e-10) * 100
    ber = (fpr + fnr) / 2
    prec = total_tp / (total_tp + total_fp + 1e-10) * 100
    rec  = total_tp / (total_tp + total_fn + 1e-10) * 100
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    acc  = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-10) * 100
    iou  = total_tp / (total_tp + total_fp + total_fn + 1e-10) * 100

    print(f"\n[{name}] ======== FINAL ({n} images) ========")
    print(f"  BER       : {ber:.4f}%")
    print(f"  FPR       : {fpr:.4f}%")
    print(f"  FNR       : {fnr:.4f}%")
    print(f"  Precision : {prec:.4f}%")
    print(f"  Recall    : {rec:.4f}%")
    print(f"  F1        : {f1:.4f}%")
    print(f"  Accuracy  : {acc:.4f}%")
    print(f"  IoU       : {iou:.4f}%")
    return dict(BER=ber, FPR=fpr, FNR=fnr, Precision=prec, Recall=rec, F1=f1)


# ──────────────────────────── main ───────────────────────────────
def main():
    # ── paths ──
    # 本脚本位于 segmentation/configs/sbu/, 需要上三级到 segmentation/
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CFG  = os.path.join(BASE, 'configs/sbu/shadow_icssm_refine.py')
    CKPT = os.path.join(BASE, 'work_dirs/shadow_icssm_refine/best_BER_iter_25000.pth')

    datasets = dict(
        SBU=dict(
            img_dir     = os.path.join(BASE, 'data/SBU-shadow/SBU-Test/ShadowImages'),
            mask_dir    = os.path.join(BASE, 'data/SBU-shadow/SBU-Test/ShadowMasks'),
            img_suffix  = '.jpg',
            mask_suffix = '.png',
            mask_loader = load_mask_binary,
        ),
        UCF=dict(
            img_dir     = os.path.join(BASE, 'data/UCF/test/img'),
            mask_dir    = os.path.join(BASE, 'data/UCF/test/mask'),
            img_suffix  = '.jpg',
            mask_suffix = '.png',
            mask_loader = load_mask_ucf,
        ),
        ISTD=dict(
            img_dir     = os.path.join(BASE, 'data/ISTD_Dataset/test/img'),
            mask_dir    = os.path.join(BASE, 'data/ISTD_Dataset/test/mask'),
            img_suffix  = '.png',
            mask_suffix = '.png',
            mask_loader = load_mask_binary,
        ),
    )

    print(f"Config    : {CFG}")
    print(f"Checkpoint: {CKPT}")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device    : {device}\n")

    model1 = init_model(CFG, CKPT, device=device)
    model1.eval()
    models = [model1]

    if ENSEMBLE_CKPT:
        model2 = init_model(CFG, ENSEMBLE_CKPT, device=device)
        model2.eval()
        models.append(model2)
        print(f"Ensemble: {len(models)} models")

    VIS_ROOT = os.path.join(BASE, 'work_dirs/shadow_icssm_refine/vis_results')
    os.makedirs(VIS_ROOT, exist_ok=True)

    PRED_ROOT = os.path.join(BASE, 'work_dirs/shadow_icssm_refine/pred_masks')
    os.makedirs(PRED_ROOT, exist_ok=True)

    results = {}
    for ds_name, ds_cfg in datasets.items():
        print(f"\n{'='*55}")
        print(f"Dataset: {ds_name}")
        vis_dir = os.path.join(VIS_ROOT, ds_name)
        os.makedirs(vis_dir, exist_ok=True)
        pred_dir = os.path.join(PRED_ROOT, ds_name)
        os.makedirs(pred_dir, exist_ok=True)
        r = evaluate_dataset(
            models,
            ds_cfg['img_dir'], ds_cfg['mask_dir'],
            ds_cfg['img_suffix'], ds_cfg['mask_suffix'],
            ds_cfg['mask_loader'], ds_name, device,
            vis_dir=vis_dir,
            pred_dir=pred_dir,
        )
        if r:
            results[ds_name] = r

    # ── summary table ──
    print(f"\n{'='*55}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'Dataset':<8} {'BER':>7} {'FPR':>7} {'FNR':>7} {'F1':>7}")
    print("-"*38)
    for name, r in results.items():
        print(f"{name:<8} {r['BER']:>7.4f} {r['FPR']:>7.4f} "
              f"{r['FNR']:>7.4f} {r['F1']:>7.4f}")
    print(f"\nVisualization saved to: {VIS_ROOT}")
    print(f"Prediction masks saved to: {PRED_ROOT}")


if __name__ == '__main__':
    main()
