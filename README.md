# LightScorer

LightScorer 是一个面向蛋白质生成候选的前置筛选原型：在调用 AlphaFold 前，先用超轻量模型过滤明显不可折叠样本，降低后续计算成本。

## 当前仓库已实现内容

- 指标定义：AUC、PR-AUC、Recall@Precision、拦截率、推理时延。
- 数据清洗：基于 target 划分去泄漏检查，支持将 split 清单转换为 manifest。
- 特征管线：`C-alpha` 距离矩阵构造、归一化、固定尺寸插值。
- 模型训练：`simple_cnn`（轻量基线）、`improved_cnn`（加深加宽 + BatchNorm + Dropout，默认）；支持早停（`--early-stop-patience`）。
- 业务收益评估：模拟先筛后 AF2 的总耗时与算力节省曲线。
- 可视化：ROC/PR、时延收益曲线、距离矩阵热图、误判样本图。

## 数据说明

当前仓库支持两种数据路径：

1. **真实数据模式（推荐）**：直接读取 raw LMDB（例如 `dataSet_withScore/casp5_to_13/data`），并使用 `tm >= 0.5` 生成标签。
2. **Mock 模式**：自动生成可控的“好/坏结构”距离矩阵，用于端到端调试。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_mock_pipeline.py
```

## 主要命令

- 从 raw LMDB 构建 manifest（按 target 重划分）：
  - `python scripts/build_manifest.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output data/manifest_raw.csv --label-policy tm_threshold --tm-threshold 0.5 --split-seed 42 --split-ratio 0.8,0.1,0.1`
- 从已有 split 清单构建 manifest（兼容旧流程）：
  - `python scripts/build_manifest.py --data-root dataset/split-by-year --output data/manifest.csv --label-policy native_vs_decoy`
- 准备训练数据（真实数据）：
  - `python scripts/prepare_data.py --manifest outputs\real_pipeline\manifest_raw.csv --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output-npz outputs/real/train_data.npz --max-train-samples 20000 --max-val-samples 5000 --max-test-samples 5000 --feature-dtype float32`
- 训练模型（纯训练，仅使用已准备数据）：
  - `python scripts/train_models.py --data-npz outputs/real/train_data.npz --output-dir outputs/real --model-name improved_cnn --epochs 20 --early-stop-patience 5 --early-stop-metric auc --device auto`
- 训练模型（Mock）：
  - `python scripts/prepare_data.py --output-npz outputs/mock/train_data.npz`
  - `python scripts/train_models.py --data-npz outputs/mock/train_data.npz --output-dir outputs/mock --epochs 5 --device auto`
- 真实全流程一键运行（构建 manifest + 训练 + 收益评估 + 出图 + go/no-go）：
  - `python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output-dir outputs/real_pipeline --model-name improved_cnn --epochs 20 --early-stop-patience 5 --early-stop-metric auc --max-entries 20000 --max-train-samples 20000 --max-val-samples 5000 --max-test-samples 5000 --feature-dtype float32 --recall-target 0.90 --device auto`
- 评估算力收益：
  - `python scripts/evaluate_savings.py --predictions outputs/predictions_test.csv --output outputs/savings.csv`
- 生成结果图：
  - `python scripts/make_figures.py --predictions outputs/predictions_test.csv --savings outputs/savings.csv --output-dir outputs/figures --data-npz outputs/real/train_data.npz --good-protein-dir D:\proteinTest\outputs\good_protein --bad-protein-dir D:\proteinTest\outputs\bad_protein`

## 指标目标建议

- 主目标：高召回（避免错杀好结构）。
- 阈值策略：先在 `val` 集上按业务约束 `Recall >= 0.90` 选择工作点，再固定阈值到 `test` 集评估对应 `speedup`。
- 不建议直接用“曲线最大 speedup”做部署阈值（该点可能是极端全拒绝，不可用）。
- 工程目标：在召回可接受前提下，把进入 AF2 的比例压到 10%-30% 区间。

## 模型与训练

- **模型**：`--model-name simple_cnn`（轻量）、`--model-name improved_cnn`（默认，推荐）。
- **早停**：`--early-stop-patience N`，连续 N 轮无提升则停止并恢复最佳模型；0 表示禁用。`--early-stop-metric auc`（默认）或 `loss`。建议 `epochs=20` 配合 `early-stop-patience=5`。

## 常见坑

- `native_vs_decoy` 只适合快速打通流程，科学评估请优先使用 `tm_threshold`。
- 必须按 `target` 划分 train/val/test，避免 target 泄漏导致结果虚高。
- 如果 `positive_ratio` 过低（例如 <5%），训练时建议做重采样或代价敏感学习。
- 内存紧张时优先使用 `--feature-dtype float16`，并分别限制 `--max-train-samples/--max-val-samples/--max-test-samples`，不要只放大单一总限额参数。
