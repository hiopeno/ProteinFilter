from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, key_values, stage, success, suggest, warn
from lightscorer.data import RealDataConfig, load_real_data
from lightscorer.manifest import ManifestBuildConfig, build_manifest, summarize_manifest
from lightscorer.metrics import evaluate_binary_metrics
from lightscorer.plots import (
    plot_curves,
    plot_distance_heatmaps,
    plot_misclassified_heatmaps,
    plot_savings_curve,
)
from lightscorer.savings import simulate_savings_curve
from lightscorer.train import TrainConfig, train_and_evaluate


def _confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n_good = int((y_true == 1).sum())
    n_bad = int((y_true == 0).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    keep_ratio = float((y_pred == 1).mean())
    reject_ratio = 1.0 - keep_ratio
    # 对优质样本（label=1）拦截率：被过滤掉的优质样本占全部优质样本比例（越低越好）
    good_reject_ratio = (fn / n_good) if n_good > 0 else 0.0
    # 对劣质样本（label=0）拦截率：被过滤掉的劣质样本占全部劣质样本比例（越高越好）
    bad_reject_ratio = (tn / n_bad) if n_bad > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "keep_ratio": keep_ratio,
        "reject_ratio": reject_ratio,
        "good_reject_ratio": float(good_reject_ratio),
        "bad_reject_ratio": float(bad_reject_ratio),
    }


def _format_reject_ratio_1_to_x(good_reject_ratio: float, bad_reject_ratio: float) -> str:
    if good_reject_ratio <= 0:
        return "1:infx" if bad_reject_ratio > 0 else "1:0.00x"
    return f"1:{(bad_reject_ratio / good_reject_ratio):.2f}x"


def _build_threshold_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    savings: pd.DataFrame,
    recall_target: float,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    rows = []
    for row in savings.itertuples(index=False):
        threshold = float(row.threshold)
        m = _confusion_at_threshold(y_true, y_score, threshold)
        rows.append(
            {
                "threshold": threshold,
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
                "keep_ratio": m["keep_ratio"],
                "reject_ratio": m["reject_ratio"],
                "good_reject_ratio": m["good_reject_ratio"],
                "bad_reject_ratio": m["bad_reject_ratio"],
                "good_reject_ratio/bad_reject_ratio(1:x)": _format_reject_ratio_1_to_x(
                    float(m["good_reject_ratio"]), float(m["bad_reject_ratio"])
                ),
                "speedup": float(row.speedup),
                "hours_saved": float(row.hours_saved),
                "meets_recall_target": bool(m["recall"] >= recall_target),
            }
        )
    report = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    constrained = report[report["meets_recall_target"]].copy()
    best_theoretical = report.sort_values("speedup", ascending=False).iloc[0]
    if constrained.empty:
        return report, None, best_theoretical
    constrained = constrained.sort_values(
        ["speedup", "precision", "threshold"], ascending=[False, False, False]
    )
    best_working = constrained.iloc[0]
    return report, best_working, best_theoretical


def _savings_from_keep_ratio(
    keep_ratio: float,
    n_candidates: int,
    af2_seconds_per_sample: float,
    lightscorer_ms_per_sample: float,
) -> dict[str, float]:
    base_total_seconds = n_candidates * af2_seconds_per_sample
    ls_total_seconds = n_candidates * (lightscorer_ms_per_sample / 1000.0)
    af2_after_seconds = n_candidates * keep_ratio * af2_seconds_per_sample
    total_seconds = ls_total_seconds + af2_after_seconds
    return {
        "baseline_hours": base_total_seconds / 3600.0,
        "pipeline_hours": total_seconds / 3600.0,
        "hours_saved": (base_total_seconds - total_seconds) / 3600.0,
        "speedup": base_total_seconds / max(total_seconds, 1e-9),
    }


def _round_sig_float(value: float, sig: int = 3) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return value
    if value == 0:
        return 0.0
    digits = sig - int(math.floor(math.log10(abs(float(value))))) - 1
    return round(float(value), digits)


def _round_numeric_sig(df: pd.DataFrame, sig: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_integer_dtype(out[col]):
            if pd.api.types.is_integer_dtype(out[col]):
                continue
            out[col] = out[col].map(lambda x: _round_sig_float(x, sig=sig))
    return out


def _print_split_label_check(manifest) -> None:
    print("Split 标签检查:")
    for split in ("train", "val", "test"):
        part = manifest[manifest["split"] == split]
        counts = part["label"].value_counts().to_dict()
        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))
        unique_classes = int((part["label"].nunique() if len(part) > 0 else 0))
        print(f"- {split}: n={len(part)}, label0={n0}, label1={n1}")
        if unique_classes < 2:
            warn("该 split 只有单一类别，AUC/PR 等指标可能失真或为 NaN。")
            suggest("建议提高 --max-entries 或调整 --split-seed / --split-ratio。")


def _parse_seeds(seeds_str: Optional[str]) -> Optional[list[int]]:
    """解析 --seeds 字符串为整数列表。"""
    if not seeds_str or not seeds_str.strip():
        return None
    parts = seeds_str.replace(",", " ").split()
    out = []
    for p in parts:
        try:
            out.append(int(p.strip()))
        except ValueError:
            raise ValueError(f"--seeds 含非法整数: {p!r}")
    return out if out else None


def _run_single_seed(
    args: argparse.Namespace,
    seed: int,
    run_output_dir: Path,
    data: dict,
    manifest_summary: dict,
) -> dict:
    """单次种子运行：训练、评估、保存到 run_output_dir，返回指标字典。"""
    fig_dir = run_output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    score_col = f"score_{args.model_name}"

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
            verbose=True,
            log_interval_steps=args.log_interval_steps,
            early_stop_patience=args.early_stop_patience,
            early_stop_metric=args.early_stop_metric,
        ),
    )
    pred = result["predictions_test"]
    pred_val = result["predictions_val"]
    training_seconds = float(result["training_info"].iloc[0]["training_seconds"])

    thresholds_arr = np.linspace(0.05, 0.95, 19)
    savings_val = simulate_savings_curve(
        y_score=pred_val[score_col].to_numpy(),
        thresholds=thresholds_arr,
        n_candidates=args.n_candidates,
        af2_seconds_per_sample=args.af2_seconds,
        lightscorer_ms_per_sample=args.ls_ms,
    )
    savings_test = simulate_savings_curve(
        y_score=pred[score_col].to_numpy(),
        thresholds=thresholds_arr,
        n_candidates=args.n_candidates,
        af2_seconds_per_sample=args.af2_seconds,
        lightscorer_ms_per_sample=args.ls_ms,
    )
    savings_val.to_csv(run_output_dir / "savings.csv", index=False)

    if pred["y_true"].nunique() >= 2:
        plot_curves(pred["y_true"].to_numpy(), pred[score_col].to_numpy(), fig_dir, score_col)
    plot_savings_curve(savings_val, fig_dir)
    try:
        plot_distance_heatmaps(data["x_test"], data["y_test"], fig_dir, prefix="real_test")
    except (ValueError, IndexError):
        pass
    metric = evaluate_binary_metrics(
        pred["y_true"].to_numpy(),
        pred[score_col].to_numpy(),
        precision_floor=0.0,
    )
    try:
        plot_misclassified_heatmaps(
            data["x_test"],
            pred["y_true"].to_numpy(),
            pred[score_col].to_numpy(),
            threshold=metric.threshold_at_precision,
            output_dir=fig_dir,
        )
    except (ValueError, IndexError):
        pass

    threshold_report, best_working, best_theoretical = _build_threshold_report(
        pred_val["y_true"].to_numpy(),
        pred_val[score_col].to_numpy(),
        savings=savings_val,
        recall_target=args.recall_target,
    )
    _round_numeric_sig(threshold_report, sig=3).to_csv(
        run_output_dir / "threshold_report_val.csv", index=False
    )
    test_threshold_report, _, _ = _build_threshold_report(
        pred["y_true"].to_numpy(),
        pred[score_col].to_numpy(),
        savings=savings_test,
        recall_target=args.recall_target,
    )
    _round_numeric_sig(test_threshold_report, sig=3).to_csv(
        run_output_dir / "threshold_report_test.csv", index=False
    )

    if best_working is None:
        test_working = pd.DataFrame(
            [
                {
                    "threshold_from_val": np.nan,
                    "precision_test": np.nan,
                    "recall_test": np.nan,
                    "keep_ratio_test": np.nan,
                    "reject_ratio_test": np.nan,
                    "good_reject_ratio_test": np.nan,
                    "bad_reject_ratio_test": np.nan,
                    "speedup_test": np.nan,
                    "hours_saved_test": np.nan,
                }
            ]
        )
    else:
        working_threshold = float(best_working["threshold"])
        test_conf = _confusion_at_threshold(
            pred["y_true"].to_numpy(),
            pred[score_col].to_numpy(),
            threshold=working_threshold,
        )
        test_savings = _savings_from_keep_ratio(
            keep_ratio=float(test_conf["keep_ratio"]),
            n_candidates=args.n_candidates,
            af2_seconds_per_sample=args.af2_seconds,
            lightscorer_ms_per_sample=args.ls_ms,
        )
        test_working = pd.DataFrame(
            [
                {
                    "threshold_from_val": working_threshold,
                    "precision_test": float(test_conf["precision"]),
                    "recall_test": float(test_conf["recall"]),
                    "keep_ratio_test": float(test_conf["keep_ratio"]),
                    "reject_ratio_test": float(test_conf["reject_ratio"]),
                    "good_reject_ratio_test": float(test_conf["good_reject_ratio"]),
                    "bad_reject_ratio_test": float(test_conf["bad_reject_ratio"]),
                    "speedup_test": float(test_savings["speedup"]),
                    "hours_saved_test": float(test_savings["hours_saved"]),
                }
            ]
        )
    _round_numeric_sig(test_working, sig=3).to_csv(
        run_output_dir / "test_working_point.csv", index=False
    )

    val_working = threshold_report[threshold_report["meets_recall_target"]]
    if val_working.empty:
        decision = "NO-GO"
        working_recall = np.nan
        working_speedup = np.nan
    else:
        row = test_working.iloc[0]
        working_recall = float(row["recall_test"])
        working_speedup = float(row["speedup_test"])
        decision = "GO" if (working_recall >= args.recall_target and working_speedup >= 1.05) else "NO-GO"

    r0 = test_working.iloc[0]
    def _fmt(v, fmt_s=".4f"):
        return f"{v:{fmt_s}}" if pd.notna(v) else "N/A"
    go_no_go_lines = [
        "# LightScorer Real Data Go/No-Go",
        "",
        f"- decision: **{decision}**",
        f"- test_auc: {metric.auc:.4f}",
        f"- test_pr_auc: {metric.pr_auc:.4f}",
        f"- recall_unconstrained: {metric.recall_at_precision:.4f}",
        f"- recall_target: {args.recall_target:.2f}",
        f"- working_threshold(from_val): {_fmt(r0['threshold_from_val'])}",
        f"- working_precision(test): {_fmt(r0['precision_test'])}",
        f"- working_recall(test): {_fmt(r0['recall_test'])}",
        f"- working_keep_ratio(test): {_fmt(r0['keep_ratio_test'])}",
        f"- working_reject_ratio(test): {_fmt(r0['reject_ratio_test'])}",
        f"- working_good_reject_ratio(test): {_fmt(r0['good_reject_ratio_test'])}",
        f"- working_bad_reject_ratio(test): {_fmt(r0['bad_reject_ratio_test'])}",
        f"- working_speedup(test): {_fmt(r0['speedup_test'])}",
        f"- working_hours_saved(test): {_fmt(r0['hours_saved_test'])}",
        f"- theoretical_max_speedup_on_val_curve: {float(best_theoretical['speedup']):.2f}x",
        f"- theoretical_max_threshold_on_val: {float(best_theoretical['threshold']):.4f}",
        f"- manifest_samples: {int(manifest_summary['n_samples'])}",
        f"- manifest_positive_ratio: {manifest_summary['positive_ratio']:.4f}",
        f"- training_time: {result['training_info'].iloc[0]['training_time']} ({training_seconds:.1f}s)",
        "",
        "## Notes",
        "- Threshold is selected on VAL under recall constraint, then fixed for TEST evaluation.",
        "- Theoretical max speedup on VAL is reference only and may be non-deployable.",
    ]
    (run_output_dir / "go_no_go.md").write_text("\n".join(go_no_go_lines), encoding="utf-8")

    r0 = test_working.iloc[0]
    def _f(k):
        v = r0.get(k, np.nan)
        return float(v) if pd.notna(v) else np.nan
    return {
        "seed": seed,
        "test_auc": float(metric.auc),
        "test_pr_auc": float(metric.pr_auc),
        "recall_test": _f("recall_test"),
        "precision_test": _f("precision_test"),
        "good_reject_ratio_test": _f("good_reject_ratio_test"),
        "bad_reject_ratio_test": _f("bad_reject_ratio_test"),
        "speedup_test": _f("speedup_test"),
        "hours_saved_test": _f("hours_saved_test"),
        "training_seconds": training_seconds,
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-lmdb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_pipeline"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--tm-threshold", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1")

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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="早停: 连续 N 轮无提升则停止，0 表示禁用")
    parser.add_argument("--early-stop-metric", type=str, default="auc", choices=["auc", "loss"], help="早停监控指标: auc 或 loss")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-interval-steps", type=int, default=200)
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument("--clip-max", type=float, default=30.0)
    parser.add_argument("--max-samples-per-split", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=1200)
    parser.add_argument("--max-test-samples", type=int, default=1200)
    parser.add_argument("--feature-dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="多种子复现，逗号分隔如 42,43,44；与 --seed 互斥，启用后每种子输出到 output_dir/seed_<n>/",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="多种子模式下数据加载用种子，默认取 seeds 首个；保证各 run 使用相同数据",
    )

    parser.add_argument("--n-candidates", type=int, default=10000)
    parser.add_argument("--af2-seconds", type=float, default=18.0)
    parser.add_argument("--ls-ms", type=float, default=5.0)
    parser.add_argument("--recall-target", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    if seeds is not None and args.seed != 42:
        warn("--seeds 与 --seed 同时指定时，--seed 将被忽略。")
    if seeds is not None and len(seeds) == 0:
        raise ValueError("--seeds 解析后为空，请提供有效种子如 42,43,44")

    banner("LightScorer 真实数据一键流程")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or (args.output_dir / "manifest_raw.csv")
    split_ratio = tuple(float(x.strip()) for x in args.split_ratio.split(","))

    stage(1, 5, "参数与目标")
    run_params = {
        "raw_lmdb_dir": args.raw_lmdb_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_metric": args.early_stop_metric,
        "batch_size": args.batch_size,
        "tm_threshold": args.tm_threshold,
        "split_ratio": split_ratio,
        "split_seed": args.split_seed,
        "recall_target": args.recall_target,
    }
    if seeds is not None:
        run_params["seeds"] = seeds
        run_params["data_seed"] = args.data_seed if args.data_seed is not None else seeds[0]
    else:
        run_params["seed"] = args.seed
    key_values("运行参数", run_params)

    stage(2, 5, "构建 manifest")
    info("正在从 raw LMDB 解码并生成样本清单...")
    manifest = build_manifest(
        ManifestBuildConfig(
            output_path=manifest_path,
            raw_lmdb_dir=args.raw_lmdb_dir,
            label_policy="tm_threshold",
            tm_threshold=args.tm_threshold,
            split_seed=args.split_seed,
            split_ratio=split_ratio,
            max_entries=args.max_entries,
        )
    )
    summary = summarize_manifest(manifest)
    success(f"Manifest 已生成: {manifest_path}")
    key_values("Manifest 摘要", summary)
    _print_split_label_check(manifest)

    data_seed = args.data_seed if args.data_seed is not None else (seeds[0] if seeds else args.seed)
    stage(3, 5, "加载特征")
    info("正在提取距离矩阵特征...")
    data = load_real_data(
        RealDataConfig(
            manifest_path=manifest_path,
            raw_lmdb_dir=args.raw_lmdb_dir,
            matrix_size=args.matrix_size,
            clip_max=args.clip_max,
            max_samples_per_split=args.max_samples_per_split,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            feature_dtype=args.feature_dtype,
            seed=data_seed,
        )
    )
    key_values(
        "数据形状",
        {"x_train": data["x_train"].shape, "x_val": data["x_val"].shape, "x_test": data["x_test"].shape},
    )
    success("特征加载完成。")

    if seeds is not None:
        all_metrics = []
        for i, seed in enumerate(seeds):
            run_output_dir = args.output_dir / f"seed_{seed}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            info(f"[多种子 {i+1}/{len(seeds)}] seed={seed} -> {run_output_dir}")
            stage(4, 5, f"训练与评估 (seed={seed})")
            m = _run_single_seed(args, seed, run_output_dir, data, summary)
            all_metrics.append(m)
            success(f"seed {seed} 完成: test_auc={m['test_auc']:.4f}, decision={m['decision']}")

        stage(5, 5, "多种子聚合")
        df = pd.DataFrame(all_metrics)
        numeric_cols = [
            "test_auc", "test_pr_auc", "recall_test", "precision_test",
            "good_reject_ratio_test", "bad_reject_ratio_test",
            "speedup_test", "hours_saved_test", "training_seconds",
        ]
        agg = {}
        for col in numeric_cols:
            if col in df.columns:
                agg[f"{col}_mean"] = df[col].mean()
                agg[f"{col}_std"] = df[col].std()
                agg[f"{col}_min"] = df[col].min()
                agg[f"{col}_max"] = df[col].max()
        agg["n_seeds"] = len(seeds)
        agg["seeds"] = ",".join(str(s) for s in seeds)
        go_count = int((df["decision"] == "GO").sum())
        agg["go_count"] = go_count
        agg["go_ratio"] = go_count / len(seeds)

        summary_df = pd.DataFrame([agg])
        summary_path = args.output_dir / "multi_seed_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        success(f"多种子汇总: {summary_path}")

        report_lines = [
            "# LightScorer 多种子复现报告",
            "",
            f"- 种子: {agg['seeds']}",
            f"- 运行次数: {agg['n_seeds']}",
            f"- GO 次数: {agg['go_count']} / {agg['n_seeds']}",
            "",
            "## 指标汇总 (mean ± std)",
            "",
        ]
        for col in numeric_cols:
            if f"{col}_mean" in agg:
                report_lines.append(f"- {col}: {agg[f'{col}_mean']:.4f} ± {agg[f'{col}_std']:.4f} [{agg[f'{col}_min']:.4f}, {agg[f'{col}_max']:.4f}]")
        report_path = args.output_dir / "multi_seed_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        success(f"多种子报告: {report_path}")

        key_values("多种子汇总", {k: v for k, v in agg.items() if not k.startswith("seeds") or k == "seeds"})
        suggest(f"各种子结果: {args.output_dir / 'seed_*'}")
    else:
        stage(4, 5, "训练与评估")
        info("正在训练模型并评估...")
        _run_single_seed(args, args.seed, args.output_dir, data, summary)
        stage(5, 5, "完成")
        success("真实数据流程执行完成。")
        suggest(
            f"可继续查看: {args.output_dir / 'threshold_report_val.csv'}、{args.output_dir / 'threshold_report_test.csv'} 与 {args.output_dir / 'go_no_go.md'}。"
        )


if __name__ == "__main__":
    main()

