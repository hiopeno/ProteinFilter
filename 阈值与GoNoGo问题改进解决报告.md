# LightScorer 阈值与 Go/No-Go 问题改进解决报告

## 1. 背景与目标

本项目目标是为蛋白质生成候选提供前置轻量筛选，在尽量不漏掉潜在好结构的前提下，减少后续 AlphaFold 评估开销。

在真实数据流程中，曾暴露出两个核心问题：

1. 阈值选择导致 `keep_ratio=1.0`，几乎全放行，筛选器工程价值弱。  
2. `go/no-go` 结论被极端阈值（全拒绝样本）产生的虚高 speedup 误导。

本次改进目标：用业务约束来定义可部署阈值，并修复结论逻辑，使输出结果可解释、可落地。

---

## 2. 问题复现与根因

### 2.1 现象

历史结果中（修复前）常见：

- `simple_cnn,test` 指标存在可用区分能力（AUC/PR-AUC 尚可）
- 但阈值落点偏低，导致 `keep_ratio=1.0`、`reject_ratio=0.0`
- 同时 `savings.csv` 中高阈值极端点出现超大 speedup（如 `3600x`）

### 2.2 根因

- 原流程将“曲线最大 speedup”与“部署阈值价值”混在一起看。  
- 缺少业务约束（例如 Recall 下限）导致阈值可能落在不可部署区域。  
- `go/no-go` 文案未区分“理论上限”与“可用工作点”。

---

## 3. 改造方案

## 3.1 总体策略

采用约束优化思路：

- 先约束：`Recall >= 0.90`
- 再优化：在满足约束的阈值中最大化 `speedup`
- 并将“理论极值点”仅作为参考，不作为主结论

## 3.2 代码改造点

主要改造文件：

- `scripts/run_real_pipeline.py`
- `README.md`

具体实现内容：

1. 增加阈值扫描报告构建逻辑（按阈值计算）：
   - `precision, recall, tp, fp, tn, fn, keep_ratio, reject_ratio, speedup, hours_saved`
2. 新增约束选点：
   - 过滤 `recall >= --recall-target`（默认 `0.90`）
   - 在候选中按 `speedup` 最高优先（并列看 precision、threshold）
3. 新增产物：
   - `threshold_report.csv`
4. 重写 go/no-go 生成逻辑：
   - 主结论基于“业务工作点”
   - 单列“theoretical_max_speedup_on_curve”作为参考
5. 终端摘要升级：
   - 明确显示业务工作点阈值与关键指标
6. 文档同步：
   - `README.md` 增加 `--recall-target 0.90`
   - 明确“不建议用曲线最大 speedup 作为部署阈值”

---

## 4. 新结果解析（修复后）

数据来源目录：`D:\proteinTest\outputs\real_pipeline`

### 4.1 核心文件

- `metrics.csv`
- `threshold_report.csv`
- `savings.csv`
- `go_no_go.md`

### 4.2 指标摘要

来自 `metrics.csv`（`simple_cnn,test`）：

- AUC: `0.7099`
- PR-AUC: `0.7582`

来自 `go_no_go.md` 的业务工作点（Recall 约束）：

- recall_target: `0.90`
- working_threshold: `0.6000`
- working_precision: `0.5768`
- working_recall: `0.9630`
- working_keep_ratio: `0.8790`
- working_reject_ratio: `0.1210`
- working_speedup: `1.1373`
- working_hours_saved: `6.0361`（在 10k 样本、50h baseline 假设下）

理论极值（仅参考）：

- theoretical_max_speedup_on_curve: `3600.00x`
- theoretical_max_threshold: `0.9500`

---

## 5. 这次修复“解决了什么”

### 已解决

1. **结论逻辑纠偏**  
   `go/no-go` 不再被极端 speedup 主导，主结论改为业务约束工作点。

2. **阈值可解释性增强**  
   通过 `threshold_report.csv` 可审计每个阈值的精度/召回/过滤率/速度收益。

3. **工程决策可落地**  
   从“看曲线峰值”切换为“在 Recall 约束下选最优可部署点”。

### 尚未完全解决（新阶段问题）

1. **过滤比例仍偏低**  
   当前工作点 `reject_ratio=0.121`，节省有限（约 `1.14x`）。

2. **模型分离度仍需提升**  
   为了满足高召回，阈值无法再大幅抬高，否则召回快速下降。

---

## 6. 风险与注意事项

1. 当前结论受单次 split、单次训练随机性影响，建议多 seed 复验。  
2. `speedup` 依赖于 AF2 时间与样本规模设定，应在真实算力预算下复标。  
3. 若要进一步提升过滤率，应优先提升模型判别能力，而非单纯调阈值。

---

## 7. 后续行动建议（按优先级）

1. **稳定性复验**  
   固定 `recall_target=0.90`，跑 3-5 个 seed，对比工作点波动。

2. **模型能力提升**  
   - 更强 backbone（如单通道 ResNet18/多尺度输入）
   - 代价敏感训练或校准方法
   - 误判样本驱动的特征增强

3. **报告闭环完善**  
   将 `threshold_report.csv` 直接纳入答辩图表，展示“约束-收益曲线”。

---

## 8. 关键结论（一句话）

本次改进已把 LightScorer 从“可能被极端指标误导”修复为“基于业务约束可部署决策”，当前结果显示系统可用但收益仍有提升空间，下一阶段应聚焦提升模型分离度以进一步压低 `keep_ratio`。

