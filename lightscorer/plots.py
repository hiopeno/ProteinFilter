from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_curves(y_true: np.ndarray, y_score: np.ndarray, output_dir: Path, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {suffix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_{suffix}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {suffix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_{suffix}.png", dpi=180)
    plt.close()


def plot_savings_curve(savings: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=savings, x="reject_ratio", y="speedup")
    plt.xlabel("Reject ratio")
    plt.ylabel("Speedup")
    plt.title("LightScorer + AF2 speedup")
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_curve.png", dpi=180)
    plt.close()


def plot_distance_heatmaps(
    x: np.ndarray, y: np.ndarray, output_dir: Path, prefix: str = "sample"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    good_idx = int(np.where(y == 1)[0][0])
    bad_idx = int(np.where(y == 0)[0][0])
    for idx, tag in [(good_idx, "good"), (bad_idx, "bad")]:
        plt.figure(figsize=(4, 4))
        sns.heatmap(x[idx], cmap="viridis", cbar=False)
        plt.title(f"{prefix}_{tag}")
        plt.tight_layout()
        plt.savefig(output_dir / f"distance_{prefix}_{tag}.png", dpi=180)
        plt.close()


def plot_misclassified_heatmaps(
    x: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
    top_k: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_score >= threshold).astype(int)
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx) == 0:
        return
    confidence = np.abs(y_score[wrong_idx] - threshold)
    order = np.argsort(-confidence)
    chosen = wrong_idx[order[:top_k]]
    for i, idx in enumerate(chosen):
        plt.figure(figsize=(4, 4))
        sns.heatmap(x[idx], cmap="magma", cbar=False)
        plt.title(
            f"mis_{i}_true{int(y_true[idx])}_pred{int(y_pred[idx])}_s{y_score[idx]:.2f}"
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"misclassified_{i}.png", dpi=180)
        plt.close()
