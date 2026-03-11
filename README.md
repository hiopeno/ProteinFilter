# LightScorer

LightScorer 是一个面向蛋白质生成候选的前置筛选原型：在调用 AlphaFold 前，先用超轻量模型过滤明显不可折叠样本，降低后续计算成本。

## 当前仓库已实现内容

- 指标定义：AUC、PR-AUC、Recall@Precision、拦截率、推理时延。
- 数据清洗：基于 target 划分去泄漏检查，从 Raw LMDB 构建 manifest。
- 特征管线：`C-alpha` 距离矩阵构造、归一化、固定尺寸插值。
- 模型训练：`simple_cnn`（轻量基线）、`improved_cnn`（加深加宽 + BatchNorm + Dropout，默认）、`improved_cnn_grn`（仅加 GRN）、`improved_cnn_largekernel`（仅加 Large-Kernel Context）、`improved_cnn_lk_grn`（Large-Kernel Context + GRN 组合版）；**消融变体**：`improved_cnn_repvgg`、`improved_cnn_pconv`、`improved_cnn_pconv_05`、`improved_cnn_repvgg_pconv`、`improved_cnn_shiftwise`（ShiftwiseConv，stride=1）、`improved_cnn_shiftwise_s2`（stride=2，等效感受野更大）；支持早停（`--early-stop-patience`）。
- 业务收益评估：模拟先筛后 AF2 的总耗时与算力节省曲线。
- 可视化：ROC/PR、时延收益曲线、距离矩阵热图、误判样本图。

## 数据说明

当前仓库支持两种数据路径：

1. **真实数据模式（推荐）**：从 Raw LMDB（`dataSet_withScore/casp5_to_13/data`）读取，使用 `tm >= 0.5` 生成标签。
2. **Mock 模式**：自动生成可控的“好/坏结构”距离矩阵，用于端到端调试。

## 快速开始

```bash
# 推荐使用 conda
conda activate d2l   # 或你的 Python 环境

# 或使用 venv
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python scripts/run_mock_pipeline.py
```

## 主要命令

- 从 raw LMDB 构建 manifest（按 target 重划分）：
  - `python scripts/build_manifest.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output data/manifest_raw.csv --label-policy tm_threshold --tm-threshold 0.5 --split-seed 42 --split-ratio 0.8,0.1,0.1`
- 准备训练数据（真实数据）：
  - `python scripts/prepare_data.py --manifest outputs\real_pipeline\manifest_raw.csv --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output-npz outputs/real/train_data.npz --max-train-samples 20000 --max-val-samples 5000 --max-test-samples 5000 --feature-dtype float32`
- 训练模型（纯训练，仅使用已准备数据）：
  - `python scripts/train_models.py --data-npz outputs/real/train_data.npz --output-dir outputs/real --model-name improved_cnn --epochs 20 --early-stop-patience 5 --early-stop-metric auc --device auto`
- 训练 GRN only 消融：
  - `python scripts/train_models.py --data-npz outputs/real/train_data.npz --output-dir outputs/real_grn --model-name improved_cnn_grn --epochs 20 --early-stop-patience 5 --early-stop-metric auc --device auto`
- 训练 LargeKernel only 消融：
  - `python scripts/train_models.py --data-npz outputs/real/train_data.npz --output-dir outputs/real_lk --model-name improved_cnn_largekernel --epochs 20 --early-stop-patience 5 --early-stop-metric auc --device auto`
- 训练稳妥增强版（推荐作为新主线对照）：
  - `python scripts/train_models.py --data-npz outputs/real/train_data.npz --output-dir outputs/real_lk_grn --model-name improved_cnn_lk_grn --epochs 20 --early-stop-patience 5 --early-stop-metric auc --device auto`
- 训练模型（Mock）：
  - `python scripts/prepare_data.py --output-npz outputs/mock/train_data.npz`
  - `python scripts/train_models.py --data-npz outputs/mock/train_data.npz --output-dir outputs/mock --epochs 5 --device auto`
- 真实全流程一键运行（构建 manifest + 训练 + 收益评估 + 出图 + go/no-go）：
  - `python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output-dir outputs/real_pipeline --model-name improved_cnn --epochs 20 --early-stop-patience 5 --early-stop-metric auc --max-entries 20000 --max-train-samples 20000 --max-val-samples 5000 --max-test-samples 5000 --feature-dtype float32 --recall-target 0.90 --device auto`
- 多种子复现（评估稳定性）：
  - `python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --output-dir outputs/multi_seed --model-name improved_cnn --seeds 42,43,44 --epochs 20 --early-stop-patience 5 --max-entries 20000 ...`
  - 输出：`outputs/multi_seed/seed_42/`、`seed_43/`、`seed_44/` 及 `multi_seed_summary.csv`、`multi_seed_report.md`
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

- **模型**：
  - `simple_cnn`：轻量基线
  - `improved_cnn`：默认推荐，加深加宽 + BatchNorm + Dropout
  - `improved_cnn_grn`：仅在 Stage3/4 加入 GRN，测试通道竞争增强本身是否能稳定提升泛化
  - `improved_cnn_largekernel`：仅在 Stage3/4 加入 Large-Kernel Context Block（7x7/9x9 depthwise），测试大感受野本身的收益
  - `improved_cnn_lk_grn`：以 `ImprovedCNN` 为骨架，在 Stage3/4 叠加 Large-Kernel Context Block（7x7/9x9 depthwise）与 GRN，增强中远程结构关系建模
  - `improved_cnn_repvgg`：RepVGG 结构重参数化，推理时可融合为单路 3×3
  - `improved_cnn_pconv`：PConv 部分卷积（r=0.25），降低内存访问
  - `improved_cnn_pconv_05`：PConv（r=0.5），半数通道做空间卷积
  - `improved_cnn_repvgg_pconv`：RepVGG + PConv 串联
  - `improved_cnn_shiftwise`：ShiftwiseConv（CVPR 2025），3×3 小核 + 四方向空间移位（stride=1），硬件友好
  - `improved_cnn_shiftwise_s2`：ShiftwiseConv 变体，shift_stride=2，等效感受野更大，用于消融
- **早停**：`--early-stop-patience N`，连续 N 轮无提升则停止并恢复最佳模型；0 表示禁用。`--early-stop-metric auc`（默认）或 `loss`。建议 `epochs=20` 配合 `early-stop-patience=5`。
- **稳妥优化建议**：若你当前消融变体在 `Recall >= 0.90` 约束下不稳定，建议按 `improved_cnn -> improved_cnn_grn -> improved_cnn_largekernel -> improved_cnn_lk_grn` 的顺序做消融。这样可以分别验证“通道竞争增强”“大感受野增强”以及“两者叠加”各自带来的收益。

## 常见坑

- `native_vs_decoy` 只适合快速打通流程，科学评估请优先使用 `tm_threshold`。
- 必须按 `target` 划分 train/val/test，避免 target 泄漏导致结果虚高。
- 如果 `positive_ratio` 过低（例如 <5%），训练时建议做重采样或代价敏感学习。
- 内存紧张时优先使用 `--feature-dtype float16`，并分别限制 `--max-train-samples/--max-val-samples/--max-test-samples`，不要只放大单一总限额参数。

## 消融实验

引入 RepVGG 与 PConv 的消融实验已实现，详见 `RepVGG与PConv消融实验指南.md`。运行示例：

```bash
# V0 基线
python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --model-name improved_cnn --output-dir outputs/ablation/V0_baseline --epochs 20 --early-stop-patience 5

# V1 +RepVGG
python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --model-name improved_cnn_repvgg --output-dir outputs/ablation/V1_repvgg --epochs 20 --early-stop-patience 5

# V2 +PConv
python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --model-name improved_cnn_pconv --output-dir outputs/ablation/V2_pconv --epochs 20 --early-stop-patience 5

# V3 +RepVGG+PConv
python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --model-name improved_cnn_repvgg_pconv --output-dir outputs/ablation/V3_repvgg_pconv --epochs 20 --early-stop-patience 5

# V4 +ShiftwiseConv
python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --model-name improved_cnn_shiftwise --output-dir outputs/ablation/V4_shiftwise --epochs 20 --early-stop-patience 5

# V5 +ShiftwiseConv_S2 (shift_stride=2，等效感受野更大)
python scripts/run_real_pipeline.py --raw-lmdb-dir dataSet_withScore/casp5_to_13/data --model-name improved_cnn_shiftwise_s2 --output-dir outputs/ablation/V5_shiftwise_s2 --epochs 20 --early-stop-patience 5
```

## 文档索引

| 文档 | 说明 |
|------|------|
| `数据流与处理流程详解.txt` | 端到端数据流与各阶段说明 |
| `项目模块处理流程梳理.md` | 模块职责与处理流程 |
| `RepVGG与PConv消融实验指南.md` | RepVGG + PConv 消融实验设计与实施 |
| `消融实验报告.md` | RepVGG + PConv 消融实验结果分析 |
| `前沿CNN具体优化指南.md` | 2022–2026 前沿 CNN 优化技术调研 |
