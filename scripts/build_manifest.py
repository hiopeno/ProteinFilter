from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, key_values, stage, success, suggest, warn
from lightscorer.manifest import ManifestBuildConfig, build_manifest, summarize_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--raw-lmdb-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, default=None)
    parser.add_argument("--label-policy", type=str, default="tm_threshold")
    parser.add_argument("--tm-threshold", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="train,val,test ratio, e.g. 0.8,0.1,0.1",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap for raw LMDB entries (debug/smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner("LightScorer 数据清单构建")
    split_ratio = tuple(float(x.strip()) for x in args.split_ratio.split(","))
    stage(1, 2, "解析参数")
    key_values(
        "运行参数",
        {
            "输出路径": args.output,
            "数据来源(data_root)": args.data_root,
            "数据来源(raw_lmdb_dir)": args.raw_lmdb_dir,
            "标签策略": args.label_policy,
            "TM阈值": args.tm_threshold,
            "划分比例(train,val,test)": split_ratio,
            "随机种子": args.split_seed,
            "样本上限(max_entries)": args.max_entries,
        },
    )
    config = ManifestBuildConfig(
        output_path=args.output,
        data_root=args.data_root,
        raw_lmdb_dir=args.raw_lmdb_dir,
        score_file=args.score_file,
        label_policy=args.label_policy,
        tm_threshold=args.tm_threshold,
        split_seed=args.split_seed,
        split_ratio=split_ratio,
        max_entries=args.max_entries,
    )
    stage(2, 2, "构建并汇总 manifest")
    info("正在读取数据并生成样本清单...")
    manifest = build_manifest(config)
    summary = summarize_manifest(manifest)
    success(f"Manifest 已保存: {args.output}")
    print("样本预览:")
    print(manifest.head().to_string(index=False))
    print("Split 计数:")
    print(manifest["split"].value_counts().to_string())
    print("标签计数:")
    print(manifest["label"].value_counts().to_string())
    key_values("摘要统计", summary)
    if summary["positive_ratio"] < 0.05:
        warn("正样本比例低于 5%，数据极度不平衡。")
        suggest("可考虑重采样、加权损失或重新设定阈值。")
    if summary["tm_missing_ratio"] > 0.0 and args.label_policy == "tm_threshold":
        warn("当前 tm_threshold 策略下存在缺失 tm 样本。")
        suggest("请检查原始数据完整性，或先过滤缺失样本。")


if __name__ == "__main__":
    main()
