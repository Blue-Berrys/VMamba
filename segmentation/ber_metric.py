"""
BER (Balance Error Rate) 评估指标
用于阴影检测任务的性能评估

BER = (1 - (1 - FPR + 1 - FNR) / 2) * 100
其中:
- FPR (False Positive Rate) = FP / (FP + TN)
- FNR (False Negative Rate) = FN / (FN + TP)
"""

import numpy as np
import torch
from typing import List, Sequence

from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class BERMetric(BaseMetric):
    """Balance Error Rate (BER) metric for shadow detection.

    BER是阴影检测任务常用的评估指标，平衡了假阳性率和假阴性率。
    BER越低表示性能越好。
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: str = None,
                 **kwargs):
        """初始化BER评估器

        Args:
            collect_device: 收集数据的设备，默认'cpu'
            prefix: 指标名称前缀
        """
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """处理一个batch的数据

        Args:
            data_batch: 输入数据batch
            data_samples: 模型预测结果
        """
        for data_sample in data_samples:
            # 获取预测结果和真实标签
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()  # [H, W]
            label = data_sample['gt_sem_seg']['data'].squeeze()  # [H, W]

            # 转换为numpy数组
            if isinstance(pred_label, torch.Tensor):
                pred_label = pred_label.cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()

            # 确保预测和标签都是二值的 (0或1)
            pred_label = (pred_label > 0).astype(np.uint8)
            label = (label > 0).astype(np.uint8)

            # 计算混淆矩阵元素
            # TP: 真阳性 (预测为阴影且实际为阴影)
            # TN: 真阴性 (预测为非阴影且实际为非阴影)
            # FP: 假阳性 (预测为阴影但实际为非阴影)
            # FN: 假阴性 (预测为非阴影但实际为阴影)
            tp = np.sum((pred_label == 1) & (label == 1))
            tn = np.sum((pred_label == 0) & (label == 0))
            fp = np.sum((pred_label == 1) & (label == 0))
            fn = np.sum((pred_label == 0) & (label == 1))

            # 保存每个样本的混淆矩阵
            self.results.append({
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            })

    def compute_metrics(self, results: list) -> dict:
        """根据所有样本计算最终指标

        Args:
            results: 所有样本的混淆矩阵列表

        Returns:
            包含BER等指标的字典
        """
        # 累加所有样本的混淆矩阵
        total_tp = sum(r['tp'] for r in results)
        total_tn = sum(r['tn'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_fn = sum(r['fn'] for r in results)

        # 计算FPR和FNR
        # FPR = FP / (FP + TN) - 假阳性率
        # FNR = FN / (FN + TP) - 假阴性率
        fpr = total_fp / (total_fp + total_tn + 1e-10)
        fnr = total_fn / (total_fn + total_tp + 1e-10)

        # 计算BER
        # BER = (1 - (1 - FPR + 1 - FNR) / 2) * 100
        # 简化为: BER = (FPR + FNR) / 2 * 100
        ber = ((fpr + fnr) / 2.0) * 100.0

        # 计算其他常用指标
        # Precision = TP / (TP + FP)
        precision = total_tp / (total_tp + total_fp + 1e-10)

        # Recall = TP / (TP + FN) = 1 - FNR
        recall = total_tp / (total_tp + total_fn + 1e-10)

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-10)

        # IoU (Intersection over Union) = TP / (TP + FP + FN)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-10)

        metrics = {
            'BER': ber,
            'FPR': fpr * 100.0,
            'FNR': fnr * 100.0,
            'Precision': precision * 100.0,
            'Recall': recall * 100.0,
            'F1': f1 * 100.0,
            'Accuracy': accuracy * 100.0,
            'IoU': iou * 100.0
        }

        return metrics
