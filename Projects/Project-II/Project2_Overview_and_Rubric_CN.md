# Project 2 简要说明与 Rubric 对照

## 这个项目在做什么

Project 2 的主题是 **Learning from Data**。项目没有要求处理原始图片，而是直接给出了已经提取好的 256 维 image feature vectors。我们需要基于这些特征完成三个任务：

1. **Image Classification**：给定图片特征，预测它属于 10 个类别中的哪一类。
2. **Image Retrieval**：给定查询图片特征，从 repository 中找出 5 张最相似的图片。
3. **Feature Selection**：从 256 个特征维度中选择不超过 30 个维度，让固定分类模型在 masked features 上尽量准确。

最终提交的核心文件是：

- `task1/classifier.py`
- `task2/retrieval.py`
- `task3/selector.py`

本项目还维护了 `submission/` 和 `submission.zip`，其中包含最终可提交版本。

## Rubric 是什么

三个子任务都是 **binary grading**：

| 子任务 | 提交文件 | Baseline | 得分规则 |
|---|---|---|---|
| Subtask 1 | `classifier.py` | Softmax Regression | 测试准确率超过 baseline 得满分，否则 0 分 |
| Subtask 2 | `retrieval.py` | Raw Euclidean Nearest Neighbor Search | 检索准确率超过 baseline 得满分，否则 0 分 |
| Subtask 3 | `selector.py` | Random 30-feature selection | 测试准确率超过 baseline 得满分，否则 0 分 |

Rubric 中没有看到明确 bonus 项。因此目标是三个子任务都超过 baseline，拿到 full score。

## 我做了什么

### Subtask 1: Image Classification

我没有继续使用线性的 Softmax Regression，而是训练了一个 **8-model MLP probability ensemble**：

- 4 个 hidden layer 为 `(512,)` 的 MLP
- 2 个 hidden layer 为 `(256,)` 的 MLP
- 2 个 hidden layer 为 `(512, 256)` 的 MLP

训练时使用 sklearn，提交时不直接 pickle sklearn 模型，而是把 MLP 权重保存到：

```text
project2_code/task1/classification_mlp_ensemble.npz
```

`classifier.py` 使用纯 NumPy 完成 forward inference，这样更不容易受到 sklearn 版本差异影响。

### Subtask 2: Image Retrieval

我把 baseline 的 raw Euclidean distance 改成了 **standardized Euclidean distance**：

1. 用 repository 的均值和标准差标准化每个特征维度。
2. 在标准化后的空间里计算 squared Euclidean distance。
3. 返回距离最小的 5 个 repository row indices。

注意：代码保持了原 baseline 的输出语义，即返回 repository 的行号，而不是第一列中不连续的 image id。

### Subtask 3: Feature Selection

我没有使用随机选 30 个特征，而是根据 validation set 和固定模型权重做了搜索：

- beam forward selection
- one-swap local check

最终选择的 30 个 feature indices 是：

```text
0, 2, 3, 4, 5, 6, 7, 9, 11, 12,
13, 14, 15, 16, 17, 19, 20, 24, 26, 28,
29, 42, 46, 47, 59, 62, 71, 120, 130, 221
```

`selector.py` 直接返回这个确定性的 binary mask，`mask_code.pkl` 也已生成。

## 本地实验结果

| 子任务 | Baseline / 对照 | 我的结果 | 预期 rubric 得分 |
|---|---:|---:|---|
| Subtask 1 Classification | Logistic/softmax-style local baseline 约 `0.5062` | MLP ensemble validation accuracy `0.587735` | 预计满分 |
| Subtask 2 Retrieval | Raw Euclidean proxy `0.09860` | Standardized Euclidean proxy `0.10016` | 预计满分，但 hidden retrieval labels 无法 100% 保证 |
| Subtask 3 Feature Selection | Random seed-42 mask `0.164866` | Selected mask validation accuracy `0.464586` | 预计满分 |

## 预计总得分

按照 rubric 的二元规则：

- Subtask 1：预计 full score
- Subtask 2：预计 full score
- Subtask 3：预计 full score
- Bonus：rubric 中未发现明确 bonus

因此整体预期是 **full score**。其中 Subtask 2 因为官方 hidden retrieval relevance 不公开，只能根据 repository self-query proxy 进行估计；当前实现至少在本地可观测 proxy 上超过 raw Euclidean baseline。

## 最终文件位置

最终打包文件：

```text
Projects/Project-II/submission.zip
```

最终提交目录：

```text
Projects/Project-II/submission/
```

主要报告文件：

```text
Projects/Project-II/Project2_Report.md
Projects/Project-II/Project2_Report.pdf
```

原始文件目录：

```text
Projects/Project-II/original_files/
```

该目录没有被修改。
