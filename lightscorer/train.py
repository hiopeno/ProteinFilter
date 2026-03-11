from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from lightscorer.cli_log import info, key_values
from lightscorer.models import (
    ImprovedCNN,
    ImprovedCNN_GRN,
    ImprovedCNN_LK_GRN,
    ImprovedCNN_LargeKernel,
    ImprovedCNN_PConv,
    ImprovedCNN_PConv_05,
    ImprovedCNN_RepVGG,
    ImprovedCNN_RepVGG_PConv,
    ImprovedCNN_ShiftwiseConv,
    ImprovedCNN_ShiftwiseConv_S2,
    SimpleCNN,
)


@dataclass
class TrainConfig:
    output_dir: Path
    model_name: str = "simple_cnn"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    verbose: bool = True
    log_interval_steps: int = 0
    # 早停：连续 patience 个 epoch 无提升则停止，0 表示禁用
    early_stop_patience: int = 0
    # 早停监控指标：auc（越大越好）或 loss（越小越好）
    early_stop_metric: str = "auc"


def _build_torch_model(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN()
    if name == "improved_cnn":
        return ImprovedCNN()
    if name == "improved_cnn_grn":
        return ImprovedCNN_GRN()
    if name == "improved_cnn_largekernel":
        return ImprovedCNN_LargeKernel()
    if name == "improved_cnn_lk_grn":
        return ImprovedCNN_LK_GRN()
    if name == "improved_cnn_repvgg":
        return ImprovedCNN_RepVGG()
    if name == "improved_cnn_pconv":
        return ImprovedCNN_PConv()
    if name == "improved_cnn_pconv_05":
        return ImprovedCNN_PConv_05()
    if name == "improved_cnn_repvgg_pconv":
        return ImprovedCNN_RepVGG_PConv()
    if name == "improved_cnn_shiftwise":
        return ImprovedCNN_ShiftwiseConv()
    if name == "improved_cnn_shiftwise_s2":
        return ImprovedCNN_ShiftwiseConv_S2()
    raise ValueError(
        f"Unknown model: {name}. Supported: simple_cnn, improved_cnn, "
        "improved_cnn_grn, improved_cnn_largekernel, improved_cnn_lk_grn, "
        "improved_cnn_repvgg, improved_cnn_pconv, improved_cnn_pconv_05, "
        "improved_cnn_repvgg_pconv, improved_cnn_shiftwise, "
        "improved_cnn_shiftwise_s2"
    )


class _NumpyBinaryDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        # 按 batch 懒转换，避免一次性复制整套数据到 torch tensor。
        fx = torch.from_numpy(self.x[idx]).to(torch.float32).unsqueeze(0)
        fy = torch.tensor(self.y[idx], dtype=torch.float32)
        return fx, fy


def _compute_val_loss(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            end = min(start + batch_size, len(x_val))
            bx = torch.from_numpy(x_val[start:end]).to(torch.float32).unsqueeze(1).to(device)
            by = torch.from_numpy(y_val[start:end]).to(torch.float32).to(device)
            logits = model(bx)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            loss = loss_fn(logits, by)
            loss_sum += float(loss) * (end - start)
            n += end - start
    return loss_sum / max(n, 1)


def _predict_torch_scores(
    model: nn.Module, x_eval: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_eval), batch_size):
            end = min(start + batch_size, len(x_eval))
            ex = torch.from_numpy(x_eval[start:end]).to(torch.float32).unsqueeze(1).to(device)
            logits = model(ex)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            prob = torch.sigmoid(logits).cpu().numpy()
            scores.append(prob)
    return np.concatenate(scores, axis=0) if scores else np.asarray([], dtype=np.float32)


def _set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _resolve_device(device_name: str) -> torch.device:
    name = device_name.lower().strip()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda 但当前环境不可用 CUDA。请安装 CUDA 版 PyTorch，或改用 --device cpu。"
            )
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda")


def train_torch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> tuple[nn.Module, float]:
    device = _resolve_device(config.device)
    _set_global_seed(config.seed, deterministic=config.deterministic)
    model = _build_torch_model(config.model_name).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_ds = _NumpyBinaryDataset(x_train, y_train)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)
    loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, generator=loader_generator
    )

    if config.verbose:
        key_values(
            "训练运行信息",
            {
                "实际设备": str(device),
                "epoch数": config.epochs,
                "batch_size": config.batch_size,
                "训练步数/epoch": len(loader),
                "学习率": config.lr,
                "早停patience": config.early_stop_patience if config.early_stop_patience > 0 else "禁用",
                "早停指标": config.early_stop_metric if config.early_stop_patience > 0 else "-",
            },
        )

    metric_name = config.early_stop_metric.lower().strip()
    if metric_name not in ("auc", "loss"):
        raise ValueError(f"early_stop_metric must be 'auc' or 'loss', got: {config.early_stop_metric}")
    higher_is_better = metric_name == "auc"
    best_val = float("-inf") if higher_is_better else float("inf")
    best_model_state: Optional[dict] = None
    epochs_no_improve = 0

    train_start = time.perf_counter()
    for epoch_idx in range(config.epochs):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        loss_count = 0
        for step_idx, (bx, by) in enumerate(loader, start=1):
            bx = bx.to(device)
            by = by.to(device)
            optim.zero_grad()
            logits = model(bx)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            loss = loss_fn(logits, by)
            loss.backward()
            optim.step()
            loss_value = float(loss.detach().cpu().item())
            loss_sum += loss_value
            loss_count += 1

            if config.verbose and config.log_interval_steps > 0 and (step_idx % config.log_interval_steps == 0):
                info(
                    f"epoch {epoch_idx + 1}/{config.epochs} step {step_idx}/{len(loader)} "
                    f"loss={loss_value:.6f}"
                )

        # 每个 epoch 结束后，给出可观察的训练信号（loss + val AUC/PR-AUC）。
        train_loss_mean = loss_sum / max(loss_count, 1)
        epoch_seconds = time.perf_counter() - epoch_start
        msg = (
            f"epoch {epoch_idx + 1}/{config.epochs} done "
            f"train_loss={train_loss_mean:.6f} "
            f"time={epoch_seconds:.1f}s "
            f"lr={optim.param_groups[0]['lr']:.2e}"
        )
        if x_val is not None and y_val is not None and len(y_val) > 0:
            val_score = _predict_torch_scores(model, x_val, device=device, batch_size=config.batch_size)
            val_metric = _score_quality(y_val, val_score)
            msg += f" val_auc={val_metric['auc']:.4f} val_pr_auc={val_metric['pr_auc']:.4f}"
            val_auc = val_metric["auc"]
            val_loss = _compute_val_loss(model, x_val, y_val, device, config.batch_size, loss_fn)
            msg += f" val_loss={val_loss:.4f}"
            if config.early_stop_patience > 0:
                if metric_name == "auc":
                    current_val = val_auc
                    valid = not (current_val != current_val)  # 非 NaN
                else:
                    current_val = val_loss
                    valid = True
                if valid:
                    improved = (current_val > best_val) if higher_is_better else (current_val < best_val)
                    if improved:
                        best_val = current_val
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
        if config.verbose:
            info(msg)
        if config.early_stop_patience > 0 and epochs_no_improve >= config.early_stop_patience:
            if config.verbose and best_model_state is not None:
                info(f"早停: val_{metric_name} 连续 {config.early_stop_patience} 轮无提升，恢复最佳模型")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    training_seconds = time.perf_counter() - train_start
    if config.verbose:
        info(f"训练总耗时: {training_seconds:.1f}s ({training_seconds / 60:.2f}min)")
    return model, training_seconds


def _score_quality(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    unique = np.unique(y_true)
    if unique.size < 2:
        auc = float("nan")
        pr_auc = 1.0 if int(unique[0]) == 1 else 0.0
    else:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    return {"auc": auc, "pr_auc": pr_auc}


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig,
) -> Dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    torch_model, training_seconds = train_torch_model(
        x_train, y_train, config=config, x_val=x_val, y_val=y_val
    )
    device = _resolve_device(config.device)
    val_score_torch = _predict_torch_scores(
        torch_model, x_val, device=device, batch_size=config.batch_size
    )
    test_score_torch = _predict_torch_scores(
        torch_model, x_test, device=device, batch_size=config.batch_size
    )

    metric_rows = []
    for model_name, yv, yt in [(config.model_name, val_score_torch, test_score_torch)]:
        val_metric = _score_quality(y_val, yv)
        test_metric = _score_quality(y_test, yt)
        metric_rows.append({"model": model_name, "split": "val", **val_metric})
        metric_rows.append({"model": model_name, "split": "test", **test_metric})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(config.output_dir / "metrics.csv", index=False)

    # 记录训练时间，便于消融实验对比
    training_info = pd.DataFrame(
        [
            {
                "model": config.model_name,
                "training_seconds": round(training_seconds, 2),
                "training_time": f"{int(training_seconds // 60)}m {training_seconds % 60:.1f}s",
            }
        ]
    )
    training_info.to_csv(config.output_dir / "training_info.csv", index=False)

    pred_test = pd.DataFrame(
        {
            "y_true": y_test.astype(int),
            f"score_{config.model_name}": test_score_torch,
        }
    )
    pred_test.to_csv(config.output_dir / "predictions_test.csv", index=False)

    pred_val = pd.DataFrame(
        {
            "y_true": y_val.astype(int),
            f"score_{config.model_name}": val_score_torch,
        }
    )
    pred_val.to_csv(config.output_dir / "predictions_val.csv", index=False)
    return {
        "metrics": metrics_df,
        "predictions_test": pred_test,
        "predictions_val": pred_val,
        "training_info": training_info,
    }
