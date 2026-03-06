from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, stage, success, suggest, warn
from lightscorer.data import MockDataConfig, load_mock_data
from lightscorer.plots import plot_curves, plot_distance_heatmaps, plot_savings_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--savings", type=Path, required=True)
    parser.add_argument("--score-column", type=str, default="score_resnet18")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner("LightScorer 图像生成")
    stage(1, 2, "读取输入数据")
    info(f"读取预测文件: {args.predictions}")
    pred = pd.read_csv(args.predictions)
    info(f"读取收益文件: {args.savings}")
    savings = pd.read_csv(args.savings)
    y_true = pred["y_true"].to_numpy()
    y_score = pred[args.score_column].to_numpy()

    stage(2, 2, "绘制图像")
    if len(set(y_true.tolist())) >= 2:
        info("绘制 ROC/PR 曲线...")
        plot_curves(y_true, y_score, output_dir=args.output_dir, suffix=args.score_column)
    else:
        warn("y_true 为单一类别，跳过 ROC/PR 曲线绘制。")
    info("绘制速度收益曲线...")
    plot_savings_curve(savings, output_dir=args.output_dir)

    info("绘制示例距离矩阵热图...")
    mock = load_mock_data(MockDataConfig(test_size=100))
    plot_distance_heatmaps(
        mock["x_test"], mock["y_test"], output_dir=args.output_dir, prefix="mock_test"
    )
    success(f"图像输出目录: {args.output_dir}")
    suggest("可打开目录查看: ROC、PR、speedup_curve、distance_heatmap。")


if __name__ == "__main__":
    main()
