from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class MetricResult:
    auc: float
    pr_auc: float
    recall_at_precision: float
    threshold_at_precision: float
    precision_floor: float
    keep_ratio: float
    reject_ratio: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "auc": self.auc,
            "pr_auc": self.pr_auc,
            "recall_at_precision": self.recall_at_precision,
            "threshold_at_precision": self.threshold_at_precision,
            "precision_floor": self.precision_floor,
            "keep_ratio": self.keep_ratio,
            "reject_ratio": self.reject_ratio,
        }


def recall_at_precision_threshold(
    y_true: np.ndarray, y_score: np.ndarray, precision_floor: float
) -> tuple[float, float]:
    if np.unique(y_true).size < 2:
        # Single-class split: threshold metrics are not meaningful.
        if int(y_true[0]) == 1:
            return 1.0, 0.0
        return 0.0, 1.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision[:-1] >= precision_floor)[0]
    if len(valid) == 0:
        return 0.0, 1.0
    idx = valid[np.argmax(recall[valid])]
    return float(recall[idx]), float(thresholds[idx])


def evaluate_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, precision_floor: float = 0.5
) -> MetricResult:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    unique = np.unique(y_true)
    if unique.size < 2:
        # Keep pipeline running and expose that the metric is undefined.
        auc = float("nan")
        pr_auc = 1.0 if int(unique[0]) == 1 else 0.0
    else:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    recall_floor, threshold = recall_at_precision_threshold(
        y_true, y_score, precision_floor=precision_floor
    )
    keep_ratio = float((y_score >= threshold).mean())
    reject_ratio = 1.0 - keep_ratio
    return MetricResult(
        auc=auc,
        pr_auc=pr_auc,
        recall_at_precision=recall_floor,
        threshold_at_precision=threshold,
        precision_floor=precision_floor,
        keep_ratio=keep_ratio,
        reject_ratio=reject_ratio,
    )
