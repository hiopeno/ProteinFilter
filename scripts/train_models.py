from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, key_values, stage, success, suggest
from lightscorer.train import TrainConfig, train_and_evaluate


def _parse_seeds(seeds_str: Optional[str]) -> Optional[list[int]]:
    if not seeds_str or not seeds_str.strip():
        return None
    parts = seeds_str.replace(",", " ").split()
    return [int(p.strip()) for p in parts if p.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="improved_cnn",
        choices=[
            "simple_cnn",
            "improved_cnn",
            "improved_cnn_grn",
            "improved_cnn_largekernel",
            "improved_cnn_lk_grn",
            "improved_cnn_repvgg",
            "improved_cnn_pconv",
            "improved_cnn_pconv_05",
            "improved_cnn_repvgg_pconv",
            "improved_cnn_shiftwise",
            "improved_cnn_shiftwise_s2",
        ],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="早停: 连续 N 轮无提升则停止，0 表示禁用")
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "loss"], help="早停监控指标: auc 或 loss")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="多种子复现，逗号分隔如 42,43,44")
    parser.add_argument("--log-interval-steps", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    if seeds is not None and len(seeds) == 0:
        raise ValueError("--seeds 解析后为空")

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
            "seeds": seeds if seeds else [args.seed],
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

    seeds_to_run = seeds if seeds else [args.seed]
    all_metrics = []
    for i, seed in enumerate(seeds_to_run):
        run_output_dir = args.output_dir / f"seed_{seed}" if len(seeds_to_run) > 1 else args.output_dir
        run_output_dir.mkdir(parents=True, exist_ok=True)
        if len(seeds_to_run) > 1:
            info(f"[多种子 {i+1}/{len(seeds_to_run)}] seed={seed} -> {run_output_dir}")

        stage(2, 2, f"训练并导出分数 (seed={seed})" if len(seeds_to_run) > 1 else "训练并导出分数")
        result = train_and_evaluate(
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_val=data["x_val"],
            y_val=data["y_val"],
            x_test=data["x_test"],
            y_test=data["y_test"],
            config=TrainConfig(
                output_dir=run_output_dir,
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                verbose=not args.quiet,
                log_interval_steps=args.log_interval_steps,
                early_stop_patience=args.early_stop_patience,
                early_stop_metric=args.early_stop_metric,
            ),
        )
        success("模型训练与分数导出完成。")
        training_info = result["training_info"]
        training_seconds = float(training_info.iloc[0]["training_seconds"])
        success(f"训练耗时: {training_info.iloc[0]['training_time']} ({training_seconds:.1f}s)")

        m = result["metrics"]
        test_row = m[m["split"] == "test"].iloc[0]
        all_metrics.append({
            "seed": seed,
            "test_auc": float(test_row["auc"]),
            "test_pr_auc": float(test_row["pr_auc"]),
            "training_seconds": training_seconds,
        })

        success(f"指标文件: {run_output_dir / 'metrics.csv'}")
        if not args.quiet:
            print(m.to_string(index=False))

    if len(seeds_to_run) > 1:
        df = pd.DataFrame(all_metrics)
        summary = df.agg({
            "test_auc": ["mean", "std", "min", "max"],
            "test_pr_auc": ["mean", "std", "min", "max"],
            "training_seconds": ["mean", "std"],
        })
        summary_path = args.output_dir / "multi_seed_summary.csv"
        df.to_csv(summary_path, index=False)
        success(f"多种子汇总: {summary_path}")
        key_values("多种子汇总", {
            "test_auc": f"{df['test_auc'].mean():.4f} ± {df['test_auc'].std():.4f}",
            "test_pr_auc": f"{df['test_pr_auc'].mean():.4f} ± {df['test_pr_auc'].std():.4f}",
        })
    else:
        suggest(
            f"下一步可执行: python scripts/evaluate_savings.py --predictions \"{args.output_dir / 'predictions_test.csv'}\" --output \"{args.output_dir / 'savings.csv'}\""
        )


if __name__ == "__main__":
    main()
