from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, key_values, stage, success, suggest
from lightscorer.train import TrainConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="improved_cnn", choices=["simple_cnn", "improved_cnn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="早停: 连续 N 轮无提升则停止，0 表示禁用")
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "loss"], help="早停监控指标: auc 或 loss")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval-steps", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner("LightScorer 模型训练")

    stage(1, 2, "加载已准备数据")
    if not args.data_npz.exists():
        raise FileNotFoundError(f"Missing data npz: {args.data_npz}")
    with np.load(args.data_npz) as npz:
        required = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
        missing = [k for k in required if k not in npz]
        if missing:
            raise ValueError(f"data npz missing required keys: {missing}")
        data = {k: npz[k] for k in required}

    key_values(
        "训练参数",
        {
            "数据文件": args.data_npz,
            "模型": args.model_name,
            "轮数": args.epochs,
            "早停patience": args.early_stop_patience if args.early_stop_patience > 0 else "禁用",
            "早停指标": args.early_stop_metric if args.early_stop_patience > 0 else "-",
            "批大小": args.batch_size,
            "设备": args.device,
            "step日志间隔": args.log_interval_steps,
            "输出目录": args.output_dir,
        },
    )
    key_values(
        "数据形状",
        {
            "x_train": data["x_train"].shape,
            "x_val": data["x_val"].shape,
            "x_test": data["x_test"].shape,
        },
    )

    stage(2, 2, "训练并导出分数")
    result = train_and_evaluate(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        config=TrainConfig(
            output_dir=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            verbose=not args.quiet,
            log_interval_steps=args.log_interval_steps,
            early_stop_patience=args.early_stop_patience,
            early_stop_metric=args.early_stop_metric,
        ),
    )
    success("模型训练与分数导出完成。")

    success(f"指标文件: {args.output_dir / 'metrics.csv'}")
    success(f"验证集预测文件: {args.output_dir / 'predictions_val.csv'}")
    success(f"预测文件: {args.output_dir / 'predictions_test.csv'}")
    print(result["metrics"].to_string(index=False))
    suggest(
        f"下一步可执行: python scripts/evaluate_savings.py --predictions \"{args.output_dir / 'predictions_test.csv'}\" --output \"{args.output_dir / 'savings.csv'}\""
    )


if __name__ == "__main__":
    main()
