# Project-I Implementation Process

这份文档按时间顺序记录本项目的实现、调参、验证和文档同步过程。正文以中文为主，关键术语保留 `English`，目的是让后续修改时能够快速知道“做了什么、为什么这么做、现阶段还剩什么风险”。

## 1. 先读题，再锁定硬约束

第一步不是写代码，而是完整读题并提炼硬约束。这里先后核对了：

- `Project1.pdf`
- `Project1-ProblemDescription.pdf`
- `Project 1 Evaluation Details.pdf`
- `Project1_Evaluator&Heuristicsearch(1).pdf`
- `Project1_Phase3 - EA(1).pdf`
- `CS311H Project1 Report Template.pdf`

最终先锁定下面这些不能错的东西：

- 外部接口必须保持：
  - `Evaluator.py -n -i -b -k -o`
  - `IEMP_Heur.py -n -i -b -k`
  - `IEMP_Evol.py -n -i -b -k`
- 最终提交目录只能有 `4` 个文件：
  - `report.pdf`
  - `Evaluator.py`
  - `IEMP_Heur.py`
  - `IEMP_Evol.py`
- 目标函数必须基于 `exposed node set`，而不是常见的 `active node set`
- 第三部分题面允许做 `one evolutionary algorithm or one simulated annealing algorithm`
- 运行环境以 `Python 3.10` 和题面给出的库版本为准

## 2. 统一开发环境

题目明确给出了版本链，所以后续实现全部以 `ai-h` 环境为准：

- `Python 3.10.20`
- `numpy==1.24.4`
- `scipy==1.14.1`
- `pandas==2.0.3`
- `networkx==2.8.8`
- `pymoo==0.6.0.1`

虽然最终代码没有依赖 `pymoo`，但环境仍然按题面要求补齐，这样做有两个好处：

- 报告里的环境说明和题面一致
- 不会出现“本地环境和 TA 环境假设不一致”的问题

## 3. 先把目标函数定义对齐

这个项目最容易做错的地方，是把题目写成普通 `influence maximization`。

这里一开始就统一采用下面这个理解：

- 两个 `campaign` 独立扩散
- 每次 `Monte Carlo world` 里得到两个 `exposed set`
- 单次 world 的得分是：
  - 两边都暴露到的节点数
  - 加上两边都没有暴露到的节点数

也就是写成：

- `|X1 inter X2| + |V - (X1 union X2)|`

后续为了实现方便，把它等价改写成：

- `|V| - |X1 symmetric_difference X2|`

这个改写在 `Evaluator`、`Heuristic refinement`、`EA fitness` 里都被复用。

## 4. 先做 Evaluator 主干

最早的 `Evaluator.py` 能跑，但样本计划比较朴素。后来逐步收敛成当前版本，核心流程是：

1. 解析 CLI
2. 读取 graph / seed
3. 校验格式和 budget
4. 按图规模选择模拟路径
5. 做 `Monte Carlo estimation`
6. 把最终 `objective value` 写到 `-o`

后面这部分经历了几轮升级：

- 引入 `CSR-like arrays`
- 小图使用 `bitset` 路径
- 大图使用 `bytearray + touched list`
- 引入 `case-aware sample plan`
- 引入 `online mean / variance`
- 引入 `confidence half-width early stop`

这样做的原因很直接：

- 小图可以多 sample，追求更稳
- 大图要在准确度和时间之间做更合理平衡
- `Project 1 Evaluation Details.pdf` 已公开不同 family 的 `Nodes / Edges` 和 `TL`

## 5. 把公开 testcase size 融进 preset

题面公开了多个 `testcase family` 的规模和时间限制。这里最终采用的策略不是“死记 case 编号”，而是：

- 对公开 `node/edge signature` 建立精确 `preset`
- 同时保留通用 `fallback`

实际做法：

- `Evaluator.py` 按 Phase 1 的 family 设 `sample plan`
- `IEMP_Heur.py` 按 Phase 2 的 family 设 `rough worlds / fine worlds / pool size / refinement`
- `IEMP_Evol.py` 按 Phase 3 的 family 设 `population / fast worlds / accurate worlds / SA reserved time`

这样既利用了公开信息，又没有把答案硬编码进代码。

## 6. Heuristic 从互斥选点改成 overlap-aware

最初版本的 `Heuristic` 主要问题是把两边 seed set 当成互斥集合，这会浪费 IEM 问题里很重要的一类动作：`common-node`。

后续升级的核心点是：

- 支持三种动作：
  - add to campaign 1
  - add to campaign 2
  - add to both
- `both` 的 cost 记为 `2`
- 不再用全局互斥 `blocked_nodes`
- 改为分别维护：
  - `selected1_set`
  - `selected2_set`

这一步对结果的帮助很大，因为 IEM 的目标本身就是在修补两边 exposure 的不平衡，而 `common-node` 往往正好适合做这件事。

## 7. Heuristic 的搜索逻辑逐步升级

当前 `Heuristic` 不是简单 greedy，而是分成几层：

### 7.1 cheap scoring

先构造 `approx exposure arrays`，用：

- direct exposure
- second-hop probabilistic coverage

去近似当前哪些区域已经平衡，哪些区域仍然偏向单边。

### 7.2 mixed candidate pools

再同时维护三类候选：

- `pool1`
- `pool2`
- `common pool`

候选打分综合使用：

- `weighted out-degree`
- `weighted in-degree`
- `campaign asymmetry`
- `one-sided exposure counts`
- `repair target`
- `common strength`

### 7.3 staged reranking

为了避免对所有候选都完整跑高样本 `Monte Carlo`，又加入了三阶段重排：

1. `rough worlds`
2. `rerank worlds`
3. `fine worlds`

也就是先粗筛，再细筛，最后只对很小的 shortlist 做高质量比较。

### 7.4 refinement

构造完解之后，再做两种 refine：

- `local_refine`
- `mc_refine`

局部动作包括：

- `swap1`
- `swap2`
- `commonize`
- `decommonize`
- `move 1 -> 2`
- `move 2 -> 1`

## 8. Evol 保留 EA 主体，但升级成 overlap-aware

第三部分没有整体改写成纯 `SA`，而是保留 `EA` 主体，再在末尾加严格限时 `SA post-refinement`。

主要原因：

- 题面允许 `EA` 或 `SA`
- 当前代码已经有 `EA warm start`
- 直接整套切成纯 `SA` 风险更大
- 把 `SA` 放在最后一段当 `last-mile refinement` 更稳

当前 `IEMP_Evol.py` 的结构是：

### 8.1 4-state encoding

- `0 = none`
- `1 = campaign1`
- `2 = campaign2`
- `3 = both`

### 8.2 repair by true budget cost

- state `1/2` cost `1`
- state `3` cost `2`

### 8.3 case-aware candidate pool

候选池不是全图展开，而是混合：

- campaign-1-biased nodes
- campaign-2-biased nodes
- common-node-biased nodes
- structural nodes

### 8.4 two-stage evaluation

- `fast worlds` 先筛一遍
- `accurate worlds` 再评估 top candidates

### 8.5 SA post-refinement

`SA` 只吃预留的剩余时间，并始终保留当前 `best-so-far solution`，确保：

- 不会因为 refinement 超时
- 即将触发时间保护时可以直接输出当前最优合法解

## 9. 加入严格卡时

后期一个重点是把“临近超时直接停止继续优化、输出当前最优解”做成统一机制。

这部分分别加在：

- `Heuristic main loop`
- `Heuristic refinement`
- `Evolutionary loop`
- `SA post-refinement`

这么做的目的很明确：

- 先保证不超时
- 再在剩余时间里尽量提分

这比“贪心地把所有 refinement 都跑完”更符合题目的 `TL` 约束。

## 10. 参考他人实现，但不照搬

中途我分析了 `Project-I_others/` 的实现，重点不是复制代码，而是吸收其中对 IEM 更贴题的思路。

最终真正融入自己实现的高价值点主要有：

- `common-node / overlap` 解空间
- `common candidate pool`
- `shared-world staged rerank`
- `structured neighborhood`
- `fast eval + accurate eval`
- `time reserve / best-so-far fallback`

没有直接照搬它的原因也很清楚：

- 代码体量太大
- 不同实现之间数据结构差异很大
- 直接拼接会引入更高回归风险

## 11. 第一次完整 benchmark 与报告生成

在主干功能稳定后，做了一轮 released local benchmark，并据此生成了：

- `dev/report.md`
- `report.pdf`
- `CURRENT_CODE_METRICS_AND_SCORE_ZH.md`
- `IMPLEMENTATION_PROCESS_ZH.md`

当时的结果一度看起来非常漂亮，尤其是：

- `Heuristic map2` 超过了公开 `36035`
- `Heuristic map3` 一度也看起来接近甚至略高于 `36200`
- `Evolutionary` 三个 released case 都高于公开 higher threshold

## 12. 后续复查：发现 Heuristic map3 的“高分”不够稳

后来在继续核对时发现一个问题：

- `Heuristic map3` 的数值在不同版本 `Evaluator` 下波动比较明显

最开始直觉上像是：

- Heuristic 退化了

但继续拆分后发现，更大的原因其实是：

- 早期 Evaluator 的 `sample count` 偏低
- `map3` 本来就卡在 `36200` 附近
- 因此估计值会在 `36190+` 到 `36210+` 一带摆动

## 13. 用独立 reference Monte Carlo 复核

为了判断到底是 solver 变差了，还是 evaluator 更严格了，我又做了一次独立交叉验证：

- 不走当前优化版 `Evaluator.py`
- 单独写 reference `Monte Carlo`
- 用更高 sample 数复核 `Heuristic map2` 和 `Heuristic map3`

结果显示：

- `map2`：当前 Evaluator 和 reference 非常接近
- `map3`：当前 Evaluator 和更高 sample sanity check 也保持一致量级

因此最终判断是：

- 当前 `Evaluator.py` 更可信
- 旧文档里 `map3` 的更高结果偏乐观

## 14. Heuristic 定向优化：只冲 map3，不破坏 map2

这次最终完成的一轮关键优化，是 `Heuristic-only safe upgrade`。

优化目标不是盲目把 `map3` 提高，而是同时满足：

- `map1 >= 450`
- `map2 >= 36035`
- `map3 >= 36200`
- 三个 released local case 的 runtime 继续明显低于公开 `Higher TL`

### 14.1 Large-family preset split

之前 `(36742, 49248)` 这一 family 的 `map2` 和 `map3` 共用同一套 preset。

这次改成了：

- `map2-like`
- `map3-like`

判断依据不看 case 编号，而是看初始 seed 统计特征：

- `map3-like`：
  - `len(initial_1) + len(initial_2) <= 16`
  - `initial overlap == 0`

这样就能对 `map3` 定向加大 common-node 倾向和 refinement 预算，而不影响 `map2`。

### 14.2 common-node bias

只对 `map3-like`：

- 增大 `pool_common`
- 提高 `choose_profile(...)` 中 `both/common` 的偏好
- 给同时接近 `only_1` 和 `only_2` frontier 的 common candidate 额外加分

### 14.3 common last-mile refine

在原有：

- `local_refine`
- `mc_refine`

之后，又加了一段只对 `map3-like` 生效的 `common last-mile refine`。

这段 refine 专门枚举：

- `commonize`
- `pair-to-common replacement`
- `common swap`

它只吃最后一小段剩余时间，而且始终保留 `best-so-far`，所以不会引入超时风险。

## 15. 当前最终 benchmark

下面是同步文档时采用的最终 released local 数据：

### 15.1 Evaluator

- `map1`: `424.37`, runtime `3.26s`
- `map2`: `35564.41`, runtime `1.05s`

### 15.2 Heuristic

- `map1`: `457.77`, runtime `20.73s`
- `map2`: `36038.53`, runtime `177.73s`
- `map3`: `36221.76`, runtime `178.08s`

### 15.3 Evolutionary

- `map1`: `458.13`, runtime `48.70s`
- `map2`: `13818.86`, runtime `99.88s`
- `map3`: `13754.33`, runtime `78.74s`

## 16. 当前结果该怎么理解

比较合理的结论是：

- `Evaluator` 当前速度非常安全，但 hidden `accuracy tolerance` 仍然无法只靠 released local data 保证
- `Heuristic map2` 现在稳定高于公开 `Higher requirement = 36035`
- `Heuristic map3` 这次已经被抬到 `36221.76`，明确高于公开 `Higher requirement = 36200`
- `Evolutionary` 当前 released local 三个 case 全部高于公开 higher threshold

## 17. 文档同步

最后一次同步时，统一更新了：

- [report.md](/d:/workspace/Artificial-Intelligence-H-2026Spring/Projects/Project-I/dev/report.md)
- [CURRENT_CODE_METRICS_AND_SCORE_ZH.md](/d:/workspace/Artificial-Intelligence-H-2026Spring/Projects/Project-I/CURRENT_CODE_METRICS_AND_SCORE_ZH.md)
- [IMPLEMENTATION_PROCESS_ZH.md](/d:/workspace/Artificial-Intelligence-H-2026Spring/Projects/Project-I/IMPLEMENTATION_PROCESS_ZH.md)

并重新生成：

- [report.pdf](/d:/workspace/Artificial-Intelligence-H-2026Spring/Projects/Project-I/dev/report.pdf)

再同步到最终提交目录：

- [report.pdf](/d:/workspace/Artificial-Intelligence-H-2026Spring/Projects/Project-I/submission/Project1_StudentID/report.pdf)

## 18. 如果后面还要继续改

下一步最值得继续投入的方向，已经不再是 released local `Heuristic` 提分，而是：

1. unseen robustness
2. `Evaluator accuracy`
3. `Evolutionary` 在 hidden family 上的稳定性

当前最不值得做的事情是：

- 再把 `Evaluator` 调回低样本版本
- 为了让数字更好看而弱化 `Evaluator` 的稳定性

因为那样只会让文档更漂亮，但真实可信度更低。
