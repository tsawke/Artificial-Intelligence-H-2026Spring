# Project 2 简要说明与 Rubric 对照

## 这个项目在做什么

Project 2 的主题是 **Learning from Data**。项目没有要求处理原始图片，而是直接给出了已经提取好的 256 维 image feature vectors。我们需要基于这些特征完成三个任务：

1. **Image Classification**：给定图片特征，预测它属于 10 个类别中的哪一类。
2. **Image Retrieval**：给定查询图片特征，从 repository 中找出 5 张最相似的图片。
3. **Feature Selection**：从 256 个特征维度中选择不超过 30 个维度，让固定分类模型在 masked features 上尽量准确。

## Rubric 是什么

三个子任务都是 **binary grading**：

| 子任务 | 提交文件 | Baseline | 得分规则 |
|---|---|---|---|
| Subtask 1 | `classifier.py` | Softmax Regression | 测试准确率超过 baseline 得满分，否则 0 分 |
| Subtask 2 | `retrieval.py` | Raw Euclidean Nearest Neighbor Search | 检索准确率超过 baseline 得满分，否则 0 分 |
| Subtask 3 | `selector.py` | Random 30-feature selection | 测试准确率超过 baseline 得满分，否则 0 分 |

Rubric 中没有看到明确 bonus 项。因此目标是三个子任务都超过 baseline，拿到 full score。

## 我做了什么

- **Subtask 1**：训练了 8 个 MLP 组成的 probability ensemble，并把权重保存为 `classification_mlp_ensemble.npz`；提交时用纯 NumPy inference，避免 sklearn pickle 版本问题。
- **Subtask 2**：将 raw Euclidean baseline 改为 standardized Euclidean retrieval，先用 repository statistics 标准化特征，再做 nearest neighbor search。
- **Subtask 3**：用 beam forward selection + one-swap local check 搜索 30 维 feature mask，替代随机选择。

## 本地实验结果与预计得分

| 子任务 | Baseline / 对照 | 我的结果 | 预期 rubric 得分 |
|---|---:|---:|---|
| Subtask 1 Classification | Logistic/softmax-style local baseline 约 `0.5062` | MLP ensemble validation accuracy `0.587735` | 预计满分 |
| Subtask 2 Retrieval | Raw Euclidean proxy `0.09860` | Standardized Euclidean proxy `0.10016` | 预计满分，但 hidden retrieval labels 无法 100% 保证 |
| Subtask 3 Feature Selection | Random seed-42 mask `0.164866` | Selected mask validation accuracy `0.464586` | 预计满分 |

整体预期是 **full score**。其中 Subtask 2 因为官方 hidden retrieval relevance 不公开，只能根据 repository self-query proxy 进行估计。

## 最终文件位置

```text
Projects/Project-II/submission.zip
Projects/Project-II/submission/
Projects/Project-II/Project2_Report.md
Projects/Project-II/Project2_Report.pdf
```

`original_files/` 没有被修改。
