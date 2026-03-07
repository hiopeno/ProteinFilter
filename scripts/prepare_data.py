from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, key_values, stage, success, suggest
from lightscorer.data import MockDataConfig, RealDataConfig, load_mock_data, load_real_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-lmdb-dir", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument("--clip-max", type=float, default=30.0)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--feature-dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner("LightScorer 数据准备")
    use_real = args.manifest is not None and args.raw_lmdb_dir is not None

    stage(1, 2, "加载并构建训练数据")
    key_values(
        "数据参数",
        {
            "数据模式": "real" if use_real else "mock",
            "manifest": args.manifest,
            "raw_lmdb_dir": args.raw_lmdb_dir,
            "train/val/test 限额": f"{args.max_train_samples}/{args.max_val_samples}/{args.max_test_samples}",
            "每 split 限额": args.max_samples_per_split,
            "矩阵尺寸": args.matrix_size,
            "特征精度": args.feature_dtype if use_real else "float32(mock)",
            "输出文件": args.output_npz,
        },
    )
    if use_real:
        info("正在从真实 LMDB 加载并提取距离矩阵特征...")
        data = load_real_data(
            RealDataConfig(
                manifest_path=args.manifest,
                raw_lmdb_dir=args.raw_lmdb_dir,
                matrix_size=args.matrix_size,
                clip_max=args.clip_max,
                max_samples_per_split=args.max_samples_per_split,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                max_test_samples=args.max_test_samples,
                feature_dtype=args.feature_dtype,
                seed=args.seed,
            )
        )
    else:
        info("正在生成 mock 数据...")
        data = load_mock_data(
            MockDataConfig(
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                matrix_size=args.matrix_size,
                seed=args.seed,
            )
        )
    key_values(
        "数据形状",
        {
            "x_train": data["x_train"].shape,
            "x_val": data["x_val"].shape,
            "x_test": data["x_test"].shape,
        },
    )

    stage(2, 2, "保存为 npz")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
    )
    success(f"数据已保存: {args.output_npz}")
    suggest(
        f"下一步可执行: python scripts/train_models.py --data-npz \"{args.output_npz}\" --output-dir outputs/real --epochs 5 --device auto"
    )


if __name__ == "__main__":
    main()
