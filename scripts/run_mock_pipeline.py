from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightscorer.cli_log import banner, info, key_values, stage, success, suggest
from lightscorer.data import MockDataConfig, load_mock_data
from lightscorer.metrics import evaluate_binary_metrics
from lightscorer.plots import (
    plot_curves,
    plot_distance_heatmaps,
    plot_misclassified_heatmaps,
    plot_savings_curve,
)
from lightscorer.savings import simulate_savings_curve
from lightscorer.train import TrainConfig, train_and_compare


def main() -> None:
    banner("LightScorer Mock 一键流程")
    output_dir = Path("outputs")
    fig_dir = output_dir / "figures"

    stage(1, 4, "准备数据并训练")
    info("正在生成 mock 数据并训练模型...")
    data = load_mock_data(MockDataConfig())
    result = train_and_compare(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        config=TrainConfig(output_dir=output_dir, model_name="resnet18", epochs=3),
    )

    stage(2, 4, "计算收益曲线")
    pred = result["predictions_test"]
    score_col = "score_resnet18"
    savings = simulate_savings_curve(
        y_score=pred[score_col].to_numpy(),
        thresholds=np.linspace(0.05, 0.95, 19),
    )
    savings.to_csv(output_dir / "savings.csv", index=False)
    success(f"收益文件已保存: {output_dir / 'savings.csv'}")

    stage(3, 4, "生成图像")
    info("正在绘制 ROC/PR、收益曲线、热图与误判样本...")
    plot_curves(pred["y_true"].to_numpy(), pred[score_col].to_numpy(), fig_dir, score_col)
    plot_savings_curve(savings, fig_dir)
    plot_distance_heatmaps(data["x_test"], data["y_test"], fig_dir, prefix="mock_test")
    metric = evaluate_binary_metrics(pred["y_true"].to_numpy(), pred[score_col].to_numpy())
    plot_misclassified_heatmaps(
        data["x_test"],
        pred["y_true"].to_numpy(),
        pred[score_col].to_numpy(),
        threshold=metric.threshold_at_precision,
        output_dir=fig_dir,
    )
    success(f"图像目录已生成: {fig_dir}")

    stage(4, 4, "生成决策摘要")
    go_no_go = output_dir / "go_no_go.md"
    speedup_best = float(savings["speedup"].max())
    decision = "GO" if (metric.recall_at_precision >= 0.95 and speedup_best >= 2.0) else "NO-GO"
    go_no_go.write_text(
        "\n".join(
            [
                "# LightScorer Go/No-Go",
                "",
                f"- decision: **{decision}**",
                f"- test_auc: {metric.auc:.4f}",
                f"- test_pr_auc: {metric.pr_auc:.4f}",
                f"- recall_at_p{metric.precision_floor:.2f}: {metric.recall_at_precision:.4f}",
                f"- best_speedup_on_curve: {speedup_best:.2f}x",
                "",
                "## Recommendation",
                "- Continue to real-data phase if independent external set keeps recall >= 0.95.",
                "- Before production, replace mock labels with TM-score/GDT-based labels and add leakage checks by sequence similarity clusters.",
            ]
        ),
        encoding="utf-8",
    )

    success("Mock 流程执行完成。")
    key_values(
        "结果总览",
        {
            "指标文件": output_dir / "metrics.csv",
            "预测文件": output_dir / "predictions_test.csv",
            "收益文件": output_dir / "savings.csv",
            "图像目录": fig_dir,
            "决策文件": go_no_go,
            "最佳加速比": f"{speedup_best:.2f}x",
        },
    )
    print(result["metrics"].to_string(index=False))
    suggest("如果要切换真实数据，请运行 scripts/run_real_pipeline.py。")


if __name__ == "__main__":
    main()
