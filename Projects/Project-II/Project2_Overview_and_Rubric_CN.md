# Project 2 简要说明与 Rubric 对照

## 这个项目在做什么

Project 2 的主题是 **Learning from Data**。项目不需要处理原始图片，而是直接使用已经提取好的 256 维 image feature vectors。需要完成三个子任务：

1. **Image Classification**：给定图片特征，预测 10 个类别之一。
2. **Image Retrieval**：给定 query 图片特征，从 repository 中返回 5 张最相似图片。
3. **Feature Selection**：从 256 个特征中选择不超过 30 个维度，使固定识别模型在 masked features 上尽量准确。

## 更新后的评测方式

根据最新 Q&A，评测方式是：

- OJ 会 import 提交的 Python 文件。
- 每个类只会被实例化一次，对应方法也只会被调用一次。
- 数据文件会软链接到当前目录，训练/验证数据与下发数据一致。
- 不能提交本地训练好的 `.pkl`、`.npz` 权重或数据来绕过训练。
- 每个 task 时间限制为 600s。
- Task 2 返回的是 repository 的**行号**，不是第一列 image id。

因此最终实现保持 OJ-compliant：Task 1 在构造函数里现场训练，Task 2 纯 NumPy 检索，Task 3 直接返回验证过的固定 mask。

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

`Classifier.__init__` 读取：

```text
classification_train_data.pkl
classification_train_label.pkl
```

然后现场训练一个加权 ensemble：

- 3 个 `MLPClassifier`，使用不同 hidden layers 和 random seeds。
- 1 个低权重 `HistGradientBoostingClassifier`，作为 diversity member。
- 对 16x16 图片特征做 5-shift 数据增强。
- `inference` 时使用相同 5-shift test-time augmentation，并累加加权概率。
- 加入 soft deadline：如果 OJ 机器较慢、剩余时间不足，就停止训练后续 ensemble member，直接使用已经训练好的最优当前模型。

我测试过额外 MLP 和 9-shift 版本：额外模型没有提升，9-shift 版本训练接近 600s 且精度更低，所以最终采用更稳的 5-shift 版本。

### Subtask 2: Image Retrieval

最终方法是 hybrid retrieval：

1. 先计算 raw Euclidean top-5，保留强 baseline 信号。
2. 再计算 shift、D4 rotation/reflection、3x3 blur 下的最佳增强匹配。
3. 若增强候选不远于 raw 第 5 近邻，则最多替换第 5 个 slot。
4. 明确返回 repository row indices，并保证每行 5 个合法且不重复的行号。
5. 加入 chunk-level soft deadline：如果大批量 query 快到时间限制，则对剩余 query 退回 raw top-5 fallback，保证按时返回完整输出。

这个策略同时提升 same-class proxy，并显著提升常见 transform query 的召回率。

### Subtask 3: Feature Selection

最终 `Selector` 直接构造一个固定 30-feature mask。这个 mask 来自 full-validation beam search，再经过 1-swap 和受限 2-swap local refinement；验证后发现它比之前动态搜索版本更高，且构造时间几乎为 0。

最终特征为：

```text
0, 2, 3, 4, 5, 6, 7, 9, 11, 12,
13, 14, 15, 16, 17, 19, 20, 24, 26, 28,
29, 33, 46, 47, 54, 59, 62, 67, 71, 205
```

## 本地模拟评测结果

最新评测使用当前 `oj_submission` 的同一份代码，按 OJ 风格重新 import、实例化、调用。Task 1 因官方 hidden labels 不可见，使用 stratified 80/20 holdout；Task 2 因官方 hidden relevance 不可见，使用 repository self-query proxy 和 transform recall；Task 3 使用公开 validation/model 流程。

| 子任务 | Baseline / 对照 | 当前结果 | 预期 rubric 得分 |
|---|---:|---:|---|
| Subtask 1 Classification | Logistic/softmax-style local baseline `0.506202` | Stratified holdout accuracy `0.592637` | 预计满分 |
| Subtask 2 Retrieval | Raw same-class proxy `0.108594` | Same-class proxy `0.109375` | 预计满分 |
| Subtask 2 Transform | Raw transform recall: right `0.08594`, down `0.00781`, rot `0.00000`, blur `0.13281` | right `0.99609`, down `0.98828`, rot `1.00000`, blur `1.00000` | 明显优于 baseline |
| Subtask 3 Feature Selection | Random seed-42 mask `0.164866` | Selected mask accuracy `0.465886` | 预计满分 |

最新完整模拟中：

- Task 1 init 约 `322.8s`，inference 约 `2.7s`。
- Task 2 init 约 `0.036s`，1000 query 估计约 `1.8s`。
- Task 3 init 约 `0s`。

三项都低于单 task 600s 限制。

## 预计总得分

按照 rubric 的二元规则：

- Subtask 1：预计 full score
- Subtask 2：预计 full score
- Subtask 3：预计 full score
- Bonus：rubric 中未发现明确 bonus

因此整体预期是 **full score**。其中 Subtask 2 因为官方 hidden retrieval relevance 不公开，只能根据 repository self-query proxy 和 transform recall 估计；当前实现至少在可观测 proxy 上超过 raw Euclidean baseline。

## 正式提交目录

正式提交包：

```text
Projects/Project-II/submission.zip
```

正式提交目录：

```text
Projects/Project-II/submission/
```

其中包含：

```text
task1/classifier.py
task2/retrieval.py
task3/selector.py
Project2_Report.md
Project2_Report.pdf
Project2_Overview_and_Rubric_CN.md
experiment_results.json
environment.yml
README.md
```

`original_files/` 没有被修改，正式提交包不包含训练数据或本地预训练权重。
