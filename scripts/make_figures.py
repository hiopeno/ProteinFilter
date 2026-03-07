from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, stage, success, suggest, warn
from lightscorer.data import MockDataConfig, load_mock_data
from lightscorer.plots import (
    export_test_protein_images,
    plot_curves,
    plot_distance_heatmaps,
    plot_savings_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--savings", type=Path, required=True)
    parser.add_argument("--score-column", type=str, default="score_simple_cnn")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, default=None)
    parser.add_argument("--good-protein-dir", type=Path, default=Path(r"D:\proteinTest\outputs\good_protein"))
    parser.add_argument("--bad-protein-dir", type=Path, default=Path(r"D:\proteinTest\outputs\bad_protein"))
    parser.add_argument("--max-test-images-per-class", type=int, default=None)
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
    if args.data_npz is not None:
        if not args.data_npz.exists():
            warn(f"未找到 data npz，跳过 test 蛋白图片导出: {args.data_npz}")
        else:
            with np.load(args.data_npz) as npz:
                if "x_test" in npz and "y_test" in npz:
                    counts = export_test_protein_images(
                        npz["x_test"],
                        npz["y_test"],
                        good_output_dir=args.good_protein_dir,
                        bad_output_dir=args.bad_protein_dir,
                        prefix="test",
                        max_per_class=args.max_test_images_per_class,
                    )
                    success(f"好蛋白图片目录: {args.good_protein_dir} (共 {counts['good']} 张)")
                    success(f"坏蛋白图片目录: {args.bad_protein_dir} (共 {counts['bad']} 张)")
                else:
                    warn("data npz 缺少 x_test/y_test，已跳过 test 蛋白图片导出。")
    success(f"图像输出目录: {args.output_dir}")
    suggest("可打开目录查看: ROC、PR、speedup_curve、distance_heatmap。")


if __name__ == "__main__":
    main()
