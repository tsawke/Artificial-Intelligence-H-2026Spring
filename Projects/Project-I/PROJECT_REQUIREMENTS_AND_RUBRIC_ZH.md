# Project-I 题目要求与评分标准提炼

这份文档是对课程 `Project 1: Information Exposure Maximization (IEM)` 的题目要求、提交约束和评分规则的精炼版整理，目标是让你在实现、调参、写 `report.pdf` 和最终打包时可以快速对照。

说明：
- 本文依据以下材料整理：`Project1.pdf`、`Project1-ProblemDescription.pdf`、`Project 1 Evaluation Details.pdf`、`Project1_Evaluator&Heuristicsearch(1).pdf`、`Project1_Phase3 - EA(1).pdf`、`CS311H Project1 Report Template.pdf`
- 若本文与原 PDF 有冲突，以原 PDF 为准
- 本文会把“公开信息能确认的内容”和“题面未公开的内容”明确区分开，避免误判

## 1. Project Goal

这个项目的主题是 `Information Exposure Maximization (IEM)`。

你需要完成三部分内容：
- 一个 `Evaluator`
- 一个 `Heuristic algorithm`
- 一个 `Evolutionary algorithm` 或一个 `Simulated Annealing algorithm`

注意：
- 即使你第三部分内部实现选择的是 `Simulated Annealing (SA)`，最终可执行文件名仍然必须是 `IEMP_Evol.py`

## 2. Core Definitions

### 2.1 Diffusion Model

题目采用 `Independent Cascade (IC)` 模型。

两条 campaign 的传播是相互独立的：
- 同一条边在 campaign 1 和 campaign 2 下有不同的传播概率
- 两条 campaign 的扩散过程分别模拟

### 2.2 Budget

你需要额外选择两个 balanced seed sets：
- `S1`
- `S2`

预算约束是：
- `|S1| + |S2| <= k`

### 2.3 Objective

这个项目最容易做错的地方是目标函数。

题目最大化的不是普通 `influence maximization` 里的 active nodes，而是 balanced information exposure。直观上，它奖励两类节点：
- 同时被两边 campaign 暴露到的节点
- 同时未被两边 campaign 暴露到的节点

因此评估时应该基于 `exposed node set`，而不是只看 `active node set`。

如果把题目误做成只统计最终激活节点的版本，`Evaluator.py`、`IEMP_Heur.py`、`IEMP_Evol.py` 都会一起偏掉。

## 3. Environment Requirements

题面明确给出的环境是：
- `Python 3.10`
- `pymoo == 0.6.0.1`
- `pandas == 2.0.3`
- `numpy == 1.24.4`
- `scipy == 1.14.1`
- `networkx == 2.8.8`

重要结论：
- 公开文档明确写了 Python 和库版本
- 公开文档没有承诺 `GPU`、`CUDA`、`PyTorch` 测评环境
- 因此最终提交版本应默认按 `CPU + Python 3.10 + 指定库版本` 可运行来设计

## 4. Executable Names

提交时必须严格使用以下文件名：
- `Evaluator.py`
- `IEMP_Heur.py`
- `IEMP_Evol.py`

这些名字是自动评测脚本依赖的接口，不能改。

## 5. CLI Interface

### 5.1 Evaluator

调用格式：

```bash
python Evaluator.py -n <social network> -i <initial seed set> -b <balanced seed set> -k <budget> -o <object value output path>
```

### 5.2 Heuristic Solver

调用格式：

```bash
python IEMP_Heur.py -n <social network> -i <initial seed set> -b <balanced seed set> -k <budget>
```

### 5.3 Evolutionary / Simulated Annealing Solver

调用格式：

```bash
python IEMP_Evol.py -n <social network> -i <initial seed set> -b <balanced seed set> -k <budget>
```

## 6. Input / Output Format

### 6.1 Social Network

图文件格式：
1. 第一行：`num_nodes num_edges`
2. 后续每行一条边：`src dst p_campaign1 p_campaign2`

### 6.2 Seed Set

初始 seed 文件和输出 balanced seed 文件格式一致：
1. 第一行：`count1 count2`
2. 接下来的 `count1` 行：campaign 1 的种子
3. 接下来的 `count2` 行：campaign 2 的种子

### 6.3 Output

- `Evaluator.py`：把目标值写到 `-o` 指定文件
- `IEMP_Heur.py` / `IEMP_Evol.py`：把 balanced seed set 写到 `-b` 指定文件

## 7. Submission Requirements

最终压缩包名称必须是：
- `Project1_[Student ID]`

压缩包内必须严格只包含 4 个文件：
- `report.pdf`
- `Evaluator.py`
- `IEMP_Heur.py`
- `IEMP_Evol.py`

题面明确警告：
- 文件数量错误可能导致自动评测失败

另外还有两个隐含但很重要的要求：
- `report.pdf` 是硬性要求，不交报告则代码不评测
- 最终提交物内不要包含额外测试文件、临时输出、`__pycache__` 或开发脚本

## 8. Report Requirements

必须提交 `report.pdf`。

题面和模板都强调：
- 没有提交项目报告，代码将不会被评测和计分
- 报告应该解释算法思路、结构、实验与结论
- 报告里不要直接粘贴 Python 源代码

推荐结构：
- `Introduction`
- `Preliminary`
- `Methodology`
- `Experiments`
- `Conclusion`

## 9. Overall Score Structure

项目总分为 `15 points`，代码部分分成三块：
- `Objective evaluation`: `2.0 points`
- `Heuristic algorithm`: `6.5 points`
- `Evolutionary algorithm / Simulated Annealing`: `6.5 points`

其中：
- `report.pdf` 更像是评测前置门槛
- 三个代码模块本身的分数相加为 `15`

## 10. Phase 1: Evaluator Rubric

Phase 1 总计 `2.0 points`，共有 `2` 个测试实例，每个实例 `1.0 point`。

每个实例拆分为：
- `Usability Test`: `0.2`
- `Accuracy Test`: `0.5`
- `Efficiency Test`: `0.3`

### 10.1 Publicly Known Rules

公开文档能确认的规则是：
1. 先检查接口和输入输出格式
2. 通过后再检查估计值是否落在官方可接受范围内
3. 通过后再检查运行速度

### 10.2 What Is Public And What Is Not

公开信息能确认：
- 有统一 cutoff time
- 有官方 acceptable range
- 有 baseline runtime 对比
- Released local table 中给出了 `Usability TL / Baseline TL / Higher TL`

公开信息不能确认：
- `Accuracy Test` 的具体数值容忍区间没有在公开 PDF 里完整给出
- 因此你无法仅凭公开文档精确判断自己的 `Evaluator.py` 一定拿到多少 accuracy points

### 10.3 Released Local Dataset Info

题面公布的 Evaluator released local 数据：
- Dataset 1：`475 nodes`, `13,289 edges`
- Dataset 2：`36,742 nodes`, `49,248 edges`

公开时间门槛：
- `Usability TL = 60s`
- `Baseline TL = 60s`
- `Higher TL = 48s`

## 11. Phase 2: Heuristic Rubric

Phase 2 总计 `6.5 points`。

结构如下：
- `Efficacy Test`: `3.9 points`
- `Robustness Test`: `2.6 points`

### 11.1 Per-Case Score Breakdown

每个 Heuristic case 总分 `1.3`，拆分为：
- `test a / Usability`: `0.1`
- `test b / Baseline quality`: `0.9`
- `test c / Advanced performance`: `0.3`

公开文档对应的判定顺序可整理为：
1. 接口正确，拿 `0.1`
2. 在 `Baseline TL` 内达到 `Baseline`，再拿 `0.9`
3. 如果达到 `Higher requirement`，或者在更严格的 `Higher TL` 内完成，再拿 `0.3`

### 11.2 Released Cases

公开的 Heuristic released cases 为 `case 0 ~ 2`：

| Case | K | Baseline | Higher Requirement | Baseline TL | Higher TL |
| --- | --- | --- | --- | --- | --- |
| case 0 | 10 | 430 | 450 | 90s | 30s |
| case 1 | 15 | 35900 | 36035 | 840s | 540s |
| case 2 | 15 | 36000 | 36200 | 840s | 540s |

### 11.3 Robustness

Heuristic 还有 `2` 个 unseen robustness cases，总计 `2.6 points`。

这意味着：
- 不能只对 released `case 0/1/2` 手工调参
- 不能把固定答案或固定图规模写死在代码里
- 更合理的做法是根据 `num_nodes / num_edges / budget` 自适应调参

## 12. Phase 3: Evolutionary / Simulated Annealing Rubric

Phase 3 总计 `6.5 points`，结构与 Heuristic 相同：
- `Efficacy Test`: `3.9 points`
- `Robustness Test`: `2.6 points`

### 12.1 Per-Case Score Breakdown

每个 Phase 3 case 总分 `1.3`，拆分为：
- `test a / Usability`: `0.1`
- `test b / Baseline quality`: `0.9`
- `test c / Advanced performance`: `0.3`

### 12.2 Released Cases

公开的 Phase 3 released cases 为 `case 0 ~ 2`：

| Case | K | Baseline | Higher Requirement | Baseline TL | Higher TL |
| --- | --- | --- | --- | --- | --- |
| case 0 | 10 | 415 | 440 | 420s | 380s |
| case 1 | 14 | 13580 | 13680 | 860s | 780s |
| case 2 | 14 | 13350 | 13600 | 860s | 780s |

### 12.3 Algorithm Choice

第三部分允许两种实现方向：
- `Evolutionary Algorithm`
- `Simulated Annealing`

但无论内部实现是哪一种，提交文件名都仍然必须是：
- `IEMP_Evol.py`

## 13. What Actually Matters For Full Score

如果目标是尽量冲高分，真正重要的是下面几件事：

### 13.1 Interface Must Be Perfect

因为每个 case 的第一步都是先看接口和输入输出格式。

只要接口错了：
- 对应 case 的后续得分直接没有

### 13.2 Objective Must Match The Problem Definition

必须按 `exposed node set` 评估。

如果误写成只看 `active node set`：
- `Evaluator.py` 会错
- `Heuristic` 的搜索导向会错
- `EA/SA` 的 fitness 也会错

### 13.3 Do Not Overfit Released Cases

因为 Heuristic 和 Phase 3 都有 `Robustness Test`。

所以更合理的策略是：
- 保持接口稳定
- 保持目标函数定义严格正确
- 参数对图规模和预算自适应
- 避免把 case 编号、固定节点数或固定答案写死

### 13.4 Full-Score Strategy Is Layered

更稳妥的拿分顺序应该是：
1. 三个脚本全部稳定通过 `Usability`
2. Released local cases 稳定过 `Baseline`
3. Released local cases 尽量冲 `Higher`
4. 同时保留对 unseen instances 的鲁棒性

## 14. Final Self-Check

提交前建议逐项检查：
- 文件名是否完全正确
- 压缩包名是否完全正确
- 压缩包里是否严格只有 4 个文件
- `report.pdf` 是否存在并可打开
- `Evaluator.py` 是否支持 `-n -i -b -k -o`
- `IEMP_Heur.py` 是否支持 `-n -i -b -k`
- `IEMP_Evol.py` 是否支持 `-n -i -b -k`
- graph / seed 文件格式读写是否符合题目要求
- 目标函数是否确实基于 `exposed node set`
- 代码是否默认在 `Python 3.10 + 指定库版本` 下可运行
- 提交目录里是否没有临时输出和 `__pycache__`

## 15. Quick Score Form

下面这份 `form` 是给你最后核对用的简洁版。

### 15.1 Evaluator Form

| Item | Tolerance / Range / Condition | Score |
| --- | --- | --- |
| Per-case usability | CLI、输入输出格式正确，并在 `Usability TL = 60s` 内完成 | `0.2` |
| Per-case accuracy | 输出值落在官方 `acceptable range` 内 | `0.5` |
| Accuracy numeric tolerance | 公开 PDF 只说明“有 acceptable range”，未完整公开具体数值区间 | `未公开` |
| Per-case efficiency | 通过 accuracy 后，再比较运行时间；released local 表里给出的更高时间门槛是 `48s` | `0.3` |
| Evaluator total | 2 个 case，每个 case 满分 `1.0` | `2.0` |

### 15.2 Heuristic Form

| Item | Tolerance / Range / Condition | Score |
| --- | --- | --- |
| Per-case test a | CLI、输入输出格式正确 | `0.1` |
| Per-case test b | `objective >= Baseline` 且 `runtime <= Baseline TL` | `0.9` |
| Per-case test c | `objective >= Higher requirement` 或 `runtime <= Higher TL` | `0.3` |
| Released efficacy total | `3` 个 released cases，每个 `1.3` | `3.9` |
| Robustness total | `2` 个 unseen cases，每个 `1.3` | `2.6` |
| Heuristic phase total | released + unseen | `6.5` |

Heuristic released thresholds：

| Case | Baseline Range | Higher Range | Time Range | Full Per-Case Score |
| --- | --- | --- | --- | --- |
| case 0 | `>= 430` | `>= 450` | `<= 90s`, advanced speed `<= 30s` | `1.3` |
| case 1 | `>= 35900` | `>= 36035` | `<= 840s`, advanced speed `<= 540s` | `1.3` |
| case 2 | `>= 36000` | `>= 36200` | `<= 840s`, advanced speed `<= 540s` | `1.3` |

### 15.3 Evolutionary / SA Form

| Item | Tolerance / Range / Condition | Score |
| --- | --- | --- |
| Per-case test a | CLI、输入输出格式正确 | `0.1` |
| Per-case test b | `objective >= Baseline` 且 `runtime <= Baseline TL` | `0.9` |
| Per-case test c | `objective >= Higher requirement` 或 `runtime <= Higher TL` | `0.3` |
| Released efficacy total | `3` 个 released cases，每个 `1.3` | `3.9` |
| Robustness total | `2` 个 unseen cases，每个 `1.3` | `2.6` |
| Phase 3 total | released + unseen | `6.5` |

Phase 3 released thresholds：

| Case | Baseline Range | Higher Range | Time Range | Full Per-Case Score |
| --- | --- | --- | --- | --- |
| case 0 | `>= 415` | `>= 440` | `<= 420s`, advanced speed `<= 380s` | `1.3` |
| case 1 | `>= 13580` | `>= 13680` | `<= 860s`, advanced speed `<= 780s` | `1.3` |
| case 2 | `>= 13350` | `>= 13600` | `<= 860s`, advanced speed `<= 780s` | `1.3` |

### 15.4 Total Score Form

| Module | Max Score | Publicly Confirmable Notes |
| --- | --- | --- |
| `Evaluator.py` | `2.0` | accuracy 的具体 numeric tolerance 未公开 |
| `IEMP_Heur.py` | `6.5` | released 可测，robustness 需 unseen 实例 |
| `IEMP_Evol.py` | `6.5` | released 可测，robustness 需 unseen 实例 |
| Project Total | `15.0` | `report.pdf` 是前置门槛 |
