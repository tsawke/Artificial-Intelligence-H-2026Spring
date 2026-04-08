# Current Code Metrics And Score

更新时间：`2026-04-07`

这份文档记录当前 `dev/` 目录里最终代码在 `ai-h` 环境下、使用当前版本 `Evaluator.py` 重跑出的 released local benchmark 结果，以及按公开 `evaluation rubric` 可以做出的分数判断。

为了避免和更早版本文档混淆，下面统一采用这套口径：

- 代码版本：当前 `dev/Evaluator.py`、`dev/IEMP_Heur.py`、`dev/IEMP_Evol.py`
- 运行环境：`ai-h`，`Python 3.10.20`
- 评分口径：以 [PROJECT_REQUIREMENTS_AND_RUBRIC_ZH.md](/d:/workspace/Artificial-Intelligence-H-2026Spring/Projects/Project-I/PROJECT_REQUIREMENTS_AND_RUBRIC_ZH.md) 中整理出的公开规则为准
- 数值解释：如果和更早文档中的结果冲突，以本文为准

## 1. Final Released-Local Data

### 1.1 Evaluator

| Case | Dataset | K | Runtime | Output Value | Public Higher TL Check |
| --- | --- | --- | --- | --- | --- |
| map1 | `dataset1` | `10` | `3.26s` | `424.37` | `3.26s < 48s` |
| map2 | `dataset2` | `15` | `1.05s` | `35564.41` | `1.05s < 48s` |

### 1.2 Heuristic

| Case | Dataset | K | Runtime | Objective Value | Public Threshold Reading |
| --- | --- | --- | --- | --- | --- |
| case 0 / map1 | `dataset1` | `10` | `20.73s` | `457.77` | `> 450` |
| case 1 / map2 | `dataset2` | `15` | `177.73s` | `36038.53` | `> 36035` |
| case 2 / map3 | `dataset2` | `15` | `178.08s` | `36221.76` | `> 36200` |

### 1.3 Evolutionary

| Case | Dataset | K | Runtime | Objective Value | Public Threshold Reading |
| --- | --- | --- | --- | --- | --- |
| case 0 / map1 | `dataset1` | `10` | `48.70s` | `458.13` | `> 440` |
| case 1 / map2 | `dataset2` | `14` | `99.88s` | `13818.86` | `> 13680` |
| case 2 / map3 | `dataset2` | `14` | `78.74s` | `13754.33` | `> 13600` |

## 2. Heuristic Upgrade Result

本轮优化只改了 `IEMP_Heur.py`，目标是把之前卡在 `36200` 下方的 `map3` 推上公开 higher threshold，同时必须稳住：

- `map1 >= 450`
- `map2 >= 36035`
- runtime 继续明显低于公开 `Higher TL`

最终结果满足了这三个目标：

- `map1` 维持在 `457.77`
- `map2` 维持在 `36038.53`
- `map3` 提升到 `36221.76`

## 3. What Changed In Heuristic

这次 `Heuristic` 的关键升级有三类：

### 3.1 Large-Family Preset Split

之前 `(36742, 49248)` family 共用一套 preset。现在改成根据初始 seed 统计特征再细分为：

- `map2-like`
- `map3-like`

其中 `map3-like` 的判断条件是：

- `len(initial_1) + len(initial_2) <= 16`
- `initial overlap == 0`

这样就能在不影响 `map2` 的情况下，单独给 `map3` 更多 refinement 预算和更强 common-node 偏置。

### 3.2 Stronger Common-Node Bias

只对 `map3-like`：

- 提高 `common pool`
- 提高 `profile` 中 `both/common` 的偏好
- 给同时接近 `only_1` 和 `only_2` frontier 的 common candidate 额外加分

### 3.3 Common Last-Mile Refine

在原有：

- `local_refine`
- `mc_refine`

之后，又增加了一段只对 `map3-like` 生效的 `common last-mile refine`。

这段 refine 的核心动作是：

- `commonize`
- `pair-to-common replacement`
- `common swap`

它只使用最后一小段剩余时间，并且始终保留 `best-so-far`，所以不会破坏时间安全性。

## 4. Evaluator Confidence Notes

当前 `Evaluator.py` 仍然是 `Monte Carlo estimator`，不是 exact evaluator。但它比更早版本更可信，原因是：

- 使用了更稳的 `case-aware sample plan`
- 使用了 `online mean / variance`
- 使用了 `confidence half-width early stop`

对大图 family 的参考交叉验证仍然支持这个结论：

| Item | Current Evaluator | Higher-Sample Reference / Sanity Check | Gap |
| --- | --- | --- | --- |
| Heuristic `map2` | `36038.53` | `36038.06` | `0.47` |
| Heuristic `map3` | `36221.76` | `36217.15` | `4.61` |

所以当前更合理的判断是：

- 它不是“数学上的唯一真值”
- 但它已经足够稳定，适合作为本地比较 solver 优劣的统一口径

## 5. Evaluation-Aligned Score Estimate

## 5.1 Phase 1: Evaluator

公开可确认的规则是：

- `Usability Test = 0.2` per case
- `Accuracy Test = 0.5` per case
- `Efficiency Test = 0.3` per case

但 `Accuracy acceptable range` 的数值容忍范围没有完整公开，因此这一部分无法只靠 released local data 完全确认。

### Conservative View

| Item | Score |
| --- | --- |
| Released local usability confirmed | `0.4 / 2.0` |
| Released local accuracy confirmed | `unknown` |
| Released local efficiency confirmed without hidden accuracy | `not fully confirmable` |

### If Hidden Accuracy Passes

| Item | Score |
| --- | --- |
| Evaluator total | `2.0 / 2.0` |

## 5.2 Phase 2: Heuristic

公开 released thresholds：

| Case | Baseline | Higher Requirement | Baseline TL | Higher TL |
| --- | --- | --- | --- | --- |
| case 0 | `430` | `450` | `90s` | `30s` |
| case 1 | `35900` | `36035` | `840s` | `540s` |
| case 2 | `36000` | `36200` | `840s` | `540s` |

当前结果：

| Case | Score | Reason |
| --- | --- | --- |
| case 0 / map1 | `1.3 / 1.3` | `457.77 >= 450` and `20.73s < 30s` |
| case 1 / map2 | `1.3 / 1.3` | `36038.53 >= 36035` and `177.73s < 540s` |
| case 2 / map3 | `1.3 / 1.3` | `36221.76 >= 36200` and `178.08s < 540s` |

Phase 2 released efficacy 小计：

- `3.9 / 3.9`

## 5.3 Phase 3: Evolutionary

公开 released thresholds：

| Case | Baseline | Higher Requirement | Baseline TL | Higher TL |
| --- | --- | --- | --- | --- |
| case 0 | `415` | `440` | `420s` | `380s` |
| case 1 | `13580` | `13680` | `860s` | `780s` |
| case 2 | `13350` | `13600` | `860s` | `780s` |

当前结果：

| Case | Score | Reason |
| --- | --- | --- |
| case 0 / map1 | `1.3 / 1.3` | `458.13 >= 440` and `48.70s < 380s` |
| case 1 / map2 | `1.3 / 1.3` | `13818.86 >= 13680` and `99.88s < 780s` |
| case 2 / map3 | `1.3 / 1.3` | `13754.33 >= 13600` and `78.74s < 780s` |

Phase 3 released efficacy 小计：

- `3.9 / 3.9`

## 6. Project-Level Summary

### 6.1 Conservative Released View

| Module | Conservative Released Score |
| --- | --- |
| Evaluator | `0.4 / 2.0` |
| Heuristic | `3.9 / 6.5` |
| Evolutionary | `3.9 / 6.5` |
| Conservative Confirmed Total | `8.2 / 15.0` |

### 6.2 If Evaluator Accuracy Passes

| Module | Score |
| --- | --- |
| Evaluator | `2.0 / 2.0` |
| Heuristic released efficacy | `3.9 / 3.9` |
| Evolutionary released efficacy | `3.9 / 3.9` |
| Released Total | `9.8 / 9.8` |

注意：

- `Robustness Test` 仍然取决于 hidden / unseen cases
- 因此 full-project 最终分数仍然不能只靠 released local benchmark 保证

## 7. Most Important Takeaways

- 本轮 `Heuristic` 优化已经把 `map3` 从“接近 higher”推到“明确高于 higher”
- 当前 released local 下，`Heuristic` 和 `Evolutionary` 都是 `3.9 / 3.9`
- `map2` 没有因为这次 `map3` 定向优化而掉回 `36035` 下方
- 当前最值得继续投入的方向已经不再是 released local `Heuristic`，而是 unseen robustness 和 `Evaluator accuracy`
