from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, key_values, stage, success, suggest
from lightscorer.savings import simulate_savings_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-column", type=str, default="score_resnet18")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-candidates", type=int, default=10000)
    parser.add_argument("--af2-seconds", type=float, default=18.0)
    parser.add_argument("--ls-ms", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner("LightScorer 收益评估")
    stage(1, 2, "读取预测结果")
    info(f"正在读取预测文件: {args.predictions}")
    pred = pd.read_csv(args.predictions)
    if args.score_column not in pred.columns:
        raise ValueError(f"Missing score column: {args.score_column}")
    key_values(
        "评估参数",
        {
            "分数字段": args.score_column,
            "候选总量(模拟)": args.n_candidates,
            "AF2单样本秒数": args.af2_seconds,
            "LightScorer单样本毫秒": args.ls_ms,
        },
    )

    stage(2, 2, "计算收益曲线")
    info("正在扫描阈值并估算时间节省与加速比...")
    thresholds = np.linspace(0.05, 0.95, 19)
    savings = simulate_savings_curve(
        pred[args.score_column].to_numpy(),
        thresholds=thresholds,
        n_candidates=args.n_candidates,
        af2_seconds_per_sample=args.af2_seconds,
        lightscorer_ms_per_sample=args.ls_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savings.to_csv(args.output, index=False)
    success(f"收益结果已保存: {args.output}")
    print("收益预览(前5行):")
    print(savings.head().to_string(index=False))
    best_row = savings.sort_values("speedup", ascending=False).iloc[0].to_dict()
    key_values(
        "最佳速度点",
        {
            "threshold": round(float(best_row["threshold"]), 4),
            "keep_ratio": round(float(best_row["keep_ratio"]), 4),
            "reject_ratio": round(float(best_row["reject_ratio"]), 4),
            "speedup": round(float(best_row["speedup"]), 4),
            "hours_saved": round(float(best_row["hours_saved"]), 4),
        },
    )
    suggest(
        f"下一步可执行: python scripts/make_figures.py --predictions \"{args.predictions}\" --savings \"{args.output}\" --score-column {args.score_column} --output-dir \"{args.output.parent / 'figures'}\""
    )


if __name__ == "__main__":
    main()
