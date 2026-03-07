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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-lmdb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_pipeline"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--tm-threshold", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1")

    parser.add_argument("--model-name", type=str, default="improved_cnn", choices=["simple_cnn", "improved_cnn"])
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

    parser.add_argument("--n-candidates", type=int, default=10000)
    parser.add_argument("--af2-seconds", type=float, default=18.0)
    parser.add_argument("--ls-ms", type=float, default=5.0)
    parser.add_argument("--recall-target", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner("LightScorer 真实数据一键流程")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    manifest_path = args.manifest_path or (args.output_dir / "manifest_raw.csv")
    split_ratio = tuple(float(x.strip()) for x in args.split_ratio.split(","))
    stage(1, 5, "参数与目标")
    key_values(
        "运行参数",
        {
            "raw_lmdb_dir": args.raw_lmdb_dir,
            "output_dir": args.output_dir,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_metric": args.early_stop_metric,
            "batch_size": args.batch_size,
            "log_interval_steps": args.log_interval_steps,
            "tm_threshold": args.tm_threshold,
            "split_ratio": split_ratio,
            "split_seed": args.split_seed,
            "max_entries": args.max_entries,
            "max_samples_per_split": args.max_samples_per_split,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "feature_dtype": args.feature_dtype,
            "recall_target": args.recall_target,
        },
    )

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

    stage(3, 5, "加载特征并训练模型")
    info("正在提取距离矩阵特征并拟合分数模型...")
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
            verbose=True,
            log_interval_steps=args.log_interval_steps,
            early_stop_patience=args.early_stop_patience,
            early_stop_metric=args.early_stop_metric,
        ),
    )
    success("模型训练与分数导出完成。")

    stage(4, 5, "收益评估与图像生成")
    info("正在计算速度收益并导出图像...")
    pred = result["predictions_test"]
    pred_val = result["predictions_val"]
    score_col = f"score_{args.model_name}"
    savings_val = simulate_savings_curve(
        y_score=pred_val[score_col].to_numpy(),
        thresholds=np.linspace(0.05, 0.95, 19),
        n_candidates=args.n_candidates,
        af2_seconds_per_sample=args.af2_seconds,
        lightscorer_ms_per_sample=args.ls_ms,
    )
    savings_path = args.output_dir / "savings.csv"
    savings_val.to_csv(savings_path, index=False)

    if pred["y_true"].nunique() >= 2:
        info("绘制 ROC/PR 曲线...")
        plot_curves(pred["y_true"].to_numpy(), pred[score_col].to_numpy(), fig_dir, score_col)
    else:
        warn("测试集只有单一类别，已跳过 ROC/PR 曲线绘制。")
    plot_savings_curve(savings_val, fig_dir)
    try:
        plot_distance_heatmaps(data["x_test"], data["y_test"], fig_dir, prefix="real_test")
    except (ValueError, IndexError):
        warn("测试集中缺少正类或负类样本，已跳过 real_test 热图。")
    metric = evaluate_binary_metrics(
        pred["y_true"].to_numpy(),
        pred[score_col].to_numpy(),
        precision_floor=0.0,
    )
    plot_misclassified_heatmaps(
        data["x_test"],
        pred["y_true"].to_numpy(),
        pred[score_col].to_numpy(),
        threshold=metric.threshold_at_precision,
        output_dir=fig_dir,
    )
    success(f"收益文件: {savings_path}")
    success(f"图像目录: {fig_dir}")

    threshold_report, best_working, best_theoretical = _build_threshold_report(
        pred_val["y_true"].to_numpy(),
        pred_val[score_col].to_numpy(),
        savings=savings_val,
        recall_target=args.recall_target,
    )
    threshold_report_path = args.output_dir / "threshold_report.csv"
    threshold_report_export = _round_numeric_sig(threshold_report, sig=3)
    threshold_report_export.to_csv(threshold_report_path, index=False)
    success(f"阈值报告(VAL): {threshold_report_path}")

    test_working_path = args.output_dir / "test_working_point.csv"
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
    test_working_export = _round_numeric_sig(test_working, sig=3)
    test_working_export.to_csv(test_working_path, index=False)
    success(f"固定阈值测试评估: {test_working_path}")

    stage(5, 5, "生成 Go/No-Go 结论")
    go_no_go = args.output_dir / "go_no_go.md"
    val_working = threshold_report[threshold_report["meets_recall_target"]]
    if val_working.empty:
        decision = "NO-GO"
        working_text = [
            f"- recall_target: {args.recall_target:.2f}",
            "- val_working_point: not_found",
            "- reason: no threshold on VAL satisfies recall target",
        ]
        warn("VAL 上未找到满足 Recall 目标的阈值工作点。")
        suggest("可降低 recall_target 或提升模型质量后重试。")
    else:
        row = test_working.iloc[0]
        working_threshold = float(row["threshold_from_val"])
        working_precision = float(row["precision_test"])
        working_recall = float(row["recall_test"])
        working_speedup = float(row["speedup_test"])
        decision = "GO" if (working_recall >= args.recall_target and working_speedup >= 1.05) else "NO-GO"
        working_text = [
            f"- recall_target: {args.recall_target:.2f}",
            f"- working_threshold(from_val): {working_threshold:.4f}",
            f"- working_precision(test): {working_precision:.4f}",
            f"- working_recall(test): {working_recall:.4f}",
            f"- working_keep_ratio(test): {float(row['keep_ratio_test']):.4f}",
            f"- working_reject_ratio(test): {float(row['reject_ratio_test']):.4f}",
            f"- working_good_reject_ratio(test): {float(row['good_reject_ratio_test']):.4f}",
            f"- working_bad_reject_ratio(test): {float(row['bad_reject_ratio_test']):.4f}",
            f"- working_speedup(test): {working_speedup:.4f}",
            f"- working_hours_saved(test): {float(row['hours_saved_test']):.4f}",
        ]
    go_no_go.write_text(
        "\n".join(
            [
                "# LightScorer Real Data Go/No-Go",
                "",
                f"- decision: **{decision}**",
                f"- test_auc: {metric.auc:.4f}",
                f"- test_pr_auc: {metric.pr_auc:.4f}",
                f"- recall_unconstrained: {metric.recall_at_precision:.4f}",
                *working_text,
                f"- theoretical_max_speedup_on_val_curve: {float(best_theoretical['speedup']):.2f}x",
                f"- theoretical_max_threshold_on_val: {float(best_theoretical['threshold']):.4f}",
                f"- manifest_samples: {int(summary['n_samples'])}",
                f"- manifest_positive_ratio: {summary['positive_ratio']:.4f}",
                "",
                "## Notes",
                "- Threshold is selected on VAL under recall constraint, then fixed for TEST evaluation.",
                "- Theoretical max speedup on VAL is reference only and may be non-deployable.",
                "- Recommend repeating with multiple random seeds and a full-size run.",
            ]
        ),
        encoding="utf-8",
    )
    success("真实数据流程执行完成。")
    key_values(
        "结果总览",
        {
            "Manifest": manifest_path,
            "Metrics": args.output_dir / "metrics.csv",
            "Predictions(VAL)": args.output_dir / "predictions_val.csv",
            "Predictions": args.output_dir / "predictions_test.csv",
            "Savings": savings_path,
            "ThresholdReport(VAL)": threshold_report_path,
            "TestWorkingPoint": test_working_path,
            "Figures": fig_dir,
            "Decision": go_no_go,
            "测试AUC": f"{metric.auc:.4f}",
            "测试PR-AUC": f"{metric.pr_auc:.4f}",
            "Recall@P": f"{metric.recall_at_precision:.4f}",
            "理论最大加速比(VAL,仅参考)": f"{float(best_theoretical['speedup']):.2f}x",
        },
    )
    if not val_working.empty:
        test_row = test_working.iloc[0]
        key_values(
            f"业务工作点(TEST评估, 阈值来自VAL, Recall>={args.recall_target:.2f})",
            {
                "threshold(from_val)": f"{float(test_row['threshold_from_val']):.4f}",
                "precision(test)": f"{float(test_row['precision_test']):.4f}",
                "recall(test)": f"{float(test_row['recall_test']):.4f}",
                "keep_ratio(test)": f"{float(test_row['keep_ratio_test']):.4f}",
                "reject_ratio(test)": f"{float(test_row['reject_ratio_test']):.4f}",
                "good_reject_ratio(test,优质拦截)": f"{float(test_row['good_reject_ratio_test']):.4f}",
                "bad_reject_ratio(test,劣质拦截)": f"{float(test_row['bad_reject_ratio_test']):.4f}",
                "speedup(test)": f"{float(test_row['speedup_test']):.4f}",
            },
        )
    suggest(
        f"可继续查看: {args.output_dir / 'threshold_report.csv'} 与 {args.output_dir / 'go_no_go.md'}。"
    )


if __name__ == "__main__":
    main()

