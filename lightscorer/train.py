from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from lightscorer.metrics import evaluate_binary_metrics
from lightscorer.models import SimpleCNN, build_resnet18_single_channel


@dataclass
class TrainConfig:
    output_dir: Path
    model_name: str = "resnet18"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    device: str = "cpu"
    precision_floor: float = 0.5
    seed: int = 42
    deterministic: bool = True


def train_logreg_baseline(
    x_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed)
    model.fit(x_train.reshape(len(x_train), -1), y_train.astype(int))
    return model


def _build_torch_model(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN()
    if name == "resnet18":
        return build_resnet18_single_channel()
    raise ValueError(f"Unknown model: {name}")


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


def train_torch_model(x_train: np.ndarray, y_train: np.ndarray, config: TrainConfig) -> nn.Module:
    device = torch.device(config.device)
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

    model.train()
    for _ in range(config.epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optim.zero_grad()
            logits = model(bx)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            loss = loss_fn(logits, by)
            loss.backward()
            optim.step()
    return model


def train_and_compare(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig,
) -> Dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lr_model = train_logreg_baseline(x_train, y_train, seed=config.seed)
    val_score_lr = lr_model.predict_proba(x_val.reshape(len(x_val), -1))[:, 1]
    test_score_lr = lr_model.predict_proba(x_test.reshape(len(x_test), -1))[:, 1]

    torch_model = train_torch_model(x_train, y_train, config=config)
    device = torch.device(config.device)
    val_score_torch = _predict_torch_scores(
        torch_model, x_val, device=device, batch_size=config.batch_size
    )
    test_score_torch = _predict_torch_scores(
        torch_model, x_test, device=device, batch_size=config.batch_size
    )

    metric_rows = []
    for model_name, yv, yt in [
        ("logreg", val_score_lr, test_score_lr),
        (config.model_name, val_score_torch, test_score_torch),
    ]:
        val_metric = evaluate_binary_metrics(y_val, yv, precision_floor=config.precision_floor)
        test_metric = evaluate_binary_metrics(y_test, yt, precision_floor=config.precision_floor)
        metric_rows.append({"model": model_name, "split": "val", **val_metric.as_dict()})
        metric_rows.append({"model": model_name, "split": "test", **test_metric.as_dict()})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(config.output_dir / "metrics.csv", index=False)

    pred_test = pd.DataFrame(
        {
            "y_true": y_test.astype(int),
            "score_logreg": test_score_lr,
            f"score_{config.model_name}": test_score_torch,
        }
    )
    pred_test.to_csv(config.output_dir / "predictions_test.csv", index=False)
    return {"metrics": metrics_df, "predictions_test": pred_test}
