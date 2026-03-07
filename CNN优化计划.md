# LightScorer CNN 优化计划

## 一、当前模型概况

### 1.1 现有架构 (SimpleCNN)

- **输入**：128×128 单通道 C-alpha 距离矩阵
- **结构**：3 层 Conv2d (16→32→64) + MaxPool2d + AdaptiveAvgPool2d(1) + Linear(64, 1)
- **训练**：5 epochs，Adam lr=1e-3，BCEWithLogitsLoss，无正则化

### 1.2 当前表现

| 指标 | 值 |
|------|-----|
| test_auc | 0.7623 |
| test_pr_auc | 0.7810 |
| 工作点 recall | 0.9948 |
| 工作点 speedup | 1.1234x |
| keep_ratio | ~89% |

### 1.3 主要问题

1. **结构过浅**：仅 3 层卷积，表达能力有限
2. **空间信息丢失过早**：32×32 直接压成 1×1，距离矩阵的局部/全局模式被抹平
3. **无正则化**：无 BatchNorm、Dropout，易过拟合
4. **未利用对称性**：距离矩阵 `M[i,j]=M[j,i]`，当前 CNN 未显式利用
5. **训练配置保守**：5 epochs、固定 lr、无调度器

---

## 二、架构优化方案

### 2.1 加深加宽 + BatchNorm + Dropout（推荐优先）

```python
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
```

### 2.2 双路池化（Max + Avg）【消融实验】

> **不确定性高**：Max 池化在自然图像上有效，但距离矩阵的统计特性不同（平滑梯度、带状结构），对好/坏结构的判别是否依赖峰值特征尚不明确。

保留更多判别信息，避免单一 AvgPool 丢失峰值特征：

```python
feat_max = F.adaptive_max_pool2d(feat, 1).flatten(1)
feat_avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)
feat = torch.cat([feat_max, feat_avg], dim=1)
```

### 2.3 对称性约束【消融实验】

> **不确定性高**：数据管线可能已产出近似对称矩阵；显式对称化可能无增益，甚至削弱模型对非对称噪声的鲁棒性。

前向时强制输入对称，与距离矩阵物理意义一致：

```python
def forward(self, x):
    x_sym = (x + x.transpose(-1, -2)) / 2
    return self.net(x_sym)
```

### 2.4 轻量 ResNet 风格

引入残差块，便于加深网络：

```python
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
    def forward(self, x):
        return F.relu(x + self.conv(x))
```

---

## 三、训练策略优化

| 项目 | 当前 | 建议 |
|------|------|------|
| epochs | 5 | 15–30，配合 early stopping |
| 学习率 | 固定 1e-3 | CosineAnnealingLR 或 ReduceLROnPlateau |
| 优化器 | Adam | AdamW (weight_decay=1e-2) |
| 数据增强 | 无 | 高斯噪声、行列同序打乱 |

### 3.1 学习率调度示例

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.epochs)
# 或
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=3)
```

### 3.2 数据增强（距离矩阵专用）【消融实验】

> **不确定性高**：行列同序打乱在数学上保持结构等价，但 CNN 是否因此提升泛化尚无定论；高斯噪声强度需调参，过强可能破坏判别信息。

```python
# 1. 高斯噪声（模拟坐标误差）
x_aug = x + np.random.randn(*x.shape).astype(x.dtype) * 0.02

# 2. 行列同序打乱（保持结构不变，等价于重排残基顺序）
perm = np.random.permutation(x.shape[1])
x_aug = x[np.ix_(perm, perm)]
```

---

## 四、实施优先级

| 阶段 | 内容 | 预期收益 | 改动量 |
|------|------|----------|--------|
| 1 | epochs 15–20、CosineAnnealingLR、AdamW | 低-中 | 小 |
| 2 | BatchNorm + Dropout | 中 | 小 |
| 3 | 加深加宽、双路池化 | 中-高 | 中 |
| 4 | 对称性约束、数据增强 | 待验证 | 中 |

---

## 五、目标指标

| 指标 | 当前 | 目标 |
|------|------|------|
| test_auc | 0.76 | ≥ 0.80 |
| 工作点 speedup (Recall≥0.9) | 1.12x | 1.3–1.5x |
| keep_ratio | ~89% | 70–80%（在 recall 约束下） |

---

## 六、消融实验设计

以下三项优化**不确定性最大**，建议单独做消融对比：

| 优化项 | 所在章节 | 不确定性来源 |
|--------|----------|--------------|
| **双路池化（Max+Avg）** | 2.2 | 距离矩阵 vs 自然图像的统计差异，Max 是否有效待验证 |
| **对称性约束** | 2.3 | 输入可能已对称；显式约束可能无增益或带来副作用 |
| **行列同序打乱（数据增强）** | 3.2 | 结构等价增强对 CNN 泛化的影响尚无定论 |
| **早停指标（auc vs loss）** | - | 以 val_auc 或 val_loss 监控早停，停止时机与最终 test 表现可能不同 |

### 消融实验矩阵

| 实验组 | 双路池化 | 对称性约束 | 行列同序打乱 | 说明 |
|--------|----------|------------|--------------|------|
| baseline | ✗ | ✗ | ✗ | 当前 SimpleCNN + 基础训练 |
| A | ✓ | ✗ | ✗ | 仅双路池化 |
| B | ✗ | ✓ | ✗ | 仅对称性约束 |
| C | ✗ | ✗ | ✓ | 仅行列同序打乱 |
| A+B | ✓ | ✓ | ✗ | 双路池化 + 对称性 |
| A+C | ✓ | ✗ | ✓ | 双路池化 + 数据增强 |
| B+C | ✗ | ✓ | ✓ | 对称性 + 数据增强 |
| A+B+C | ✓ | ✓ | ✓ | 全开 |

每组建议跑 3 个随机种子（如 42, 123, 456），取 test_auc、speedup 均值与标准差。

---

## 七、验证方式

1. 每次改动后运行 `run_real_pipeline.py`，对比 `go_no_go.md` 与 `test_working_point.csv`
2. 关注 val_auc 曲线是否过拟合（train 升、val 降）
3. 多随机种子（如 42, 123, 456）取平均，评估稳定性

---

## 八、注意事项

- 保持 LightScorer 的**轻量**定位：推理时延需维持在 ~5ms/样本 量级
- 新增模型需在 `train.py` 的 `_build_torch_model` 中注册
- 若使用 `--model-name` 参数，需在 `run_real_pipeline.py` 的 `parse_args` 中扩展 choices
