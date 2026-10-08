# 论文实验完整执行手册

本手册对应 `run_paper_experiments.ps1`、`ATTACK.py` 和
`MAGIC_main/paper_outputs.py` 的当前实现。主实验统一报告四项指标：

1. `DirectTP-ASR`，越高越好；
2. `EvasionAmplification`，越高越好；
3. `GraphModifyRate = (deltaE+deltaV)/(E0+V0)`，越低越好；
4. `AverageAttackBudget = (deltaE+deltaV)/成功逃逸初始TP数`，越低越好。

## 1. 新增策略和消融参数

### Greedy动作策略

```text
--action-policy greedy
```

Greedy不额外查询检测器。Greedy、Random和PPO使用完全相同的宏动作协议：
先选择一次语义边类型，再连续插入5条同类型边，随后只查询和反馈一次。
Greedy先让每个合法动作至少被尝试一次，再选择历史平均宏动作奖励最高的
动作；三种策略的候选端点都按当前可观察异常分数、节点度和节点编号依次
排序。这保证对比只改变“语义动作选择策略”，不会让某种策略通过额外查询
或不同候选端点规则获得不公平优势。

PPO的actor只输出9维语义动作概率，不再输出候选列表下标。每个宏动作对应
一个reward和一个transition，在线PPO跨目标累计8个宏动作后更新；运行结束
时更新不足8个但已经完成的剩余transition。旧版非宏动作checkpoint会被
明确拒绝，正式实验必须重新训练。

PPO动作选择和计算复现性使用两个独立概念：

```text
--deterministic
--ppo-selection-mode auto|sample|argmax
```

`--deterministic`只固定随机种子和确定性计算，不改变策略语义。默认`auto`
强制Online PPO使用`sample`进行on-policy探索，Frozen PPO使用`argmax`
评估checkpoint学到的固定策略。若要研究冻结策略的随机部署，可显式使用
`--ppo-selection-mode sample`；Online PPO不允许argmax训练。

正式脚本使用 `--warmup-updates 0`。当前主预算最多约产生
`5轮 × 10目标 × 2宏动作 = 100` 个宏transition，约13次PPO更新；如果仍使用
旧值10，整次训练都会停留在强制均匀采样阶段，无法检验PPO是否学到策略。

### 奖励消融

```text
--reward-mode full
--reward-mode none
--reward-mode score_only
--reward-mode no_graph_penalty
```

预设权重：

| Reward mode | Target score | Target crossing | Indirect effect | Graph penalty |
|---|---:|---:|---:|---:|
| full | 1.0 | 0.5 | 0.2 | 0.05 |
| none | 0 | 0 | 0 | 0 |
| score_only | 1.0 | 0 | 0 | 0 |
| no_graph_penalty | 1.0 | 0.5 | 0.2 | 0 |

也可单独覆盖：

```text
--reward-score-weight FLOAT
--reward-crossing-weight FLOAT
--reward-indirect-weight FLOAT
--reward-perturbation-weight FLOAT
```

间接奖励不读取metadata或TP标签。候选集合固定为“初始MAGIC告警中尚未
被直接攻击的节点”。每次宏动作查询后，只统计上一状态仍高于阈值的候选：
其归一化正向分数下降取Top-5均值，跨阈值数量经`tanh(count/2)`有界化，
二者各占间接信号的一半，再乘0.2进入总奖励。已跨阈值节点不会在后续查询
重复获得跨阈值奖励。这使PPO可以感知图修改的旁路影响，同时保持黑盒设定。

最终日志和`summary.json.result.policy_health`会额外报告：

- `NormalizedEntropy`及其前后各8个宏动作均值；
- `MeanMaxActionProbability`；
- `Top1Top2ProbabilityGap`；
- `KLToUniform`；
- `IndirectRewardNonzeroRate`；
- 每种动作的次数、平均奖励、直接跨阈值率和间接奖励非零率。

若Online PPO已经更新但最终`NormalizedEntropy > 0.98`，日志会打印
`PolicyHealthWarning`，提示策略仍接近合法动作集合上的均匀分布。

### 检测器查询协议

每次运行只保留下列必要查询：

1. 攻击开始时查询一次原始图，固定阈值、初始告警和初始分数；
2. 每完成一个5边宏动作后查询一次，用于奖励、下一状态和跨阈值判断；
3. `--clean-repeat-checks N`显式要求的N次重复性审计。

轮首直接复用上一轮最后一次宏动作反馈；轮末不查询；最终指标也复用最后
一次反馈。无攻击基线因此只需要一次检测器查询。日志中的`QueryReuse`和
最终`reused_query_count`记录省去的同图重复查询数。该优化不减少任何奖励
反馈，也不改变动作、目标、最终分数或四项主指标。

### 候选节点选择器消融

```text
--candidate-policy semantic_mixed
--candidate-policy random_legal
```

`random_legal`只移除局部/全局混合候选选择策略，仍保留节点类型、边方向、
无自环和无重复边约束。消融实验不能删除动作合法性，否则比较对象不是同一
威胁模型。

### 消融标签

```text
--experiment-variant full_model
--experiment-variant no_reward
--experiment-variant no_graph_penalty
--experiment-variant no_candidate_selector
--experiment-variant no_ppo
```

代码会检查标签与实际参数是否一致，避免把错误配置写入表5。

## 2. 干净实验原则

脚本为每次实验启动一个独立Python进程，因此每次都会重新加载：

- 原始 `test0.pkl`；
- 检测器和固定阈值；
- 对应随机种子；
- 对应PPO checkpoint；
- 空的攻击修改记录。

脚本不会复用上一次攻击后的图，也不会覆盖已有运行目录。若实验根目录已经
存在，`smoke`阶段会直接报错，要求选择一个新目录。

推荐新目录：

```text
D:\Python\attack-main\attack_results\paper_suite_theia_v1
```

目录结构：

```text
paper_suite_theia_v1/
  experiment_manifest.txt
  smoke/             # 冒烟，不进入论文表
  runs/              # 正式实验，每次一个独立目录
  paper_outputs/     # CSV、PNG和PDF
```

## 3. 环境检查

推荐直接运行统一脚本的检查阶段：

```powershell
Set-Location D:\Python\attack-main
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage check
```

它会核对STGAN中的PyTorch、DGL、CUDA和绘图库，检查Python语法，并验证
CADETS、THEIA、TRACE三个数据集的动作空间、无标签间接奖励公式以及
自适应基线策略。
也可以手工执行等价命令：

```powershell
Set-Location D:\Python\attack-main

$py = "D:\App\anaconda\envs\stgan\python.exe"
& $py -c "import torch,dgl,sklearn,matplotlib; print(torch.__version__, dgl.__version__, torch.cuda.is_available())"
& $py -B MAGIC_main\validate_action_space.py --dataset cadets --test-index 0
& $py -B MAGIC_main\validate_action_space.py --dataset theia --test-index 0
& $py -B MAGIC_main\validate_action_space.py --dataset trace --test-index 0
```

三个验证命令都必须正常退出，并报告合法动作及边覆盖率。

## 4. 阶段一：冒烟测试

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage smoke `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds 42,43,44
```

运行内容：在相同Margin、K=1、R=1和seed=42下依次验证No Attack、
Random、Greedy、LinUCB、Double DQN、Online PPO和Frozen PPO。Online PPO
结束时强制刷新不足8条的rollout；Double DQN在冒烟阶段临时使用batch size=1，
确保即使第一个宏动作立即逃逸也能打印一次loss。正式实验仍使用batch size=16。

验证目标：

- PPO能找到合法动作，且每个`MacroActionStart`后恰有5条同动作`Action`；
- 每5条边只产生1条`MacroTransition`和1次`RewardBatch`；
- 至少出现1条`PPOMacroUpdate`，并成功保存新版宏动作checkpoint；
- LinUCB出现`LinUCBUpdate`，合法动作外的动作不会被选择；
- Double DQN出现`DoubleDQNUpdate`，`updated=True`且loss为有限数；
- Frozen PPO能够加载本次冒烟刚生成的checkpoint且不更新参数；
- Online PPO使用`sample`，Frozen PPO使用`argmax`；
- `policy_health`至少记录一次PPO选择并输出非均匀性指标；
- 最终四项指标能打印；
- score missing/nonfinite为0；
- clean repeat分数差和告警差为0；
- `summary.json`、`trajectory.json`、`scores.npz`和逐次图片生成成功。

检查：

```powershell
Get-ChildItem .\attack_results\paper_suite_theia_v1\smoke
Select-String `
  -Path .\attack_results\paper_suite_theia_v1\smoke\*\run.log `
  -Pattern "LinUCBUpdate|DoubleDQNUpdate|PPOMacroUpdate|FinalMetrics"
```

冒烟失败时立即停止，不运行正式实验。

## 5. 阶段二：分三步完成表2和表3

### 5.1 只运行表2：目标选择策略

先只运行seed 42：

```powershell
& .\run_paper_experiments.ps1 `
  -Stage table2 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds @(42)
```

固定Random动作、K=10、R=5，比较：

- `threshold_margin`；
- `highest_score`；
- `random_alert`。

验证目标：在完全相同动作策略下，Margin是否获得更高的
DirectTP-ASR、Amplification或更低的AAB。

生成：

```text
table02_target_policy.csv
fig05_table02_target_policy.png/pdf
```

表2只验证目标选择，不运行PPO，也不会产生PPO loss。确认Margin目标选择合理
后再进入下一阶段。补齐seed 43、44时执行同一个阶段并只传入新种子：

```powershell
& .\run_paper_experiments.ps1 `
  -Stage table2 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds @(43,44)
```

### 5.2 只运行正式PPO诊断对

```powershell
& .\run_paper_experiments.ps1 `
  -Stage ppo_probe `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds @(42)
```

该阶段以正式K=10、R=5配置依次运行Online PPO和Frozen PPO。Online产生
checkpoint并使用sample，Frozen加载同一checkpoint、使用argmax且不更新。
这两次结果直接进入表3，不是一次性调试数据。脚本会验证Online至少更新一次，
以及Frozen没有更新参数。

确认loss、entropy、动作概率、间接奖励和四项指标正常后，再对seed 43、44执行：

```powershell
& .\run_paper_experiments.ps1 `
  -Stage ppo_probe `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds @(43,44)
```

### 5.3 补齐表3：其余动作策略

```powershell
& .\run_paper_experiments.ps1 `
  -Stage table3 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds @(42)
```

该阶段只新增No Attack、Greedy、LinUCB和Double DQN。Random+Margin复用
表2结果，Online/Frozen PPO复用`ppo_probe`结果，不产生重复实验。

确认seed 42完整表3正常后补齐：

```powershell
& .\run_paper_experiments.ps1 `
  -Stage table3 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds @(43,44)
```

表3固定Margin、K=10、R=5，最终比较：

- No Attack；
- Random；
- Greedy；
- LinUCB；
- Double DQN；
- Online PPO；
- Frozen PPO。

验证目标：比较无学习启发式、上下文bandit、off-policy价值学习和
on-policy Actor-Critic在同一状态、动作、奖励、合法性约束与查询预算下的效果。
LinUCB用于判断问题是否主要依赖即时上下文；Double DQN用于判断经验回放和
长期Q值是否优于PPO。Frozen PPO仍作为最终部署策略，Online PPO展示在线训练过程。

生成：

```text
table03_action_policy.csv
fig06_table03_action_policy.png/pdf
fig09_adaptive_learning_curves.png/pdf
appendix_action_distribution.csv
fig08_action_distribution.png/pdf
```

旧的`-Stage core`仍被保留，但只适合一个尚未产生任何正式运行的全新实验目录；
它会依次执行`table2`、`ppo_probe`和`table3`。一旦已经按上述方式完成任一
阶段，就不要再运行`core`，因为脚本会拒绝覆盖已有运行目录。

### 第一个停止/总结点

三阶段结束后，先不要立即运行预算和消融。执行：

```powershell
Import-Csv .\attack_results\paper_suite_theia_v1\paper_outputs\table02_target_policy.csv | Format-Table -AutoSize
Import-Csv .\attack_results\paper_suite_theia_v1\paper_outputs\table03_action_policy.csv | Format-Table -AutoSize
```

满足以下条件时，才能初步认为想法是正确的并继续：

1. No Attack的GMR为0，其他无有效攻击分母的指标为N/A；
2. 每次正式攻击的DirectTP-ASR分母大于0；
3. Margin在多数种子上不弱于Random Target；
4. Frozen PPO在多数种子上至少在DirectTP-ASR或Amplification上优于Random；
5. Frozen PPO与Greedy、LinUCB、Double DQN相比存在可见优势，或以更低
   GMR/AAB达到相近效果；若其他算法最优，则如实把它报告为最佳策略；
6. PPO和Double DQN的更新次数增加，LinUCB累计了有效反馈，动作分布不再
   始终均匀；
7. score integrity错误全部为0。

如果Frozen PPO稳定弱于Random和Greedy，应在这里停止，先修改奖励或训练，
不应继续消耗时间运行预算与消融。

三个种子适合阶段性判断。核心结论确定后，如需补种子45、46，必须按相同顺序
补齐三个阶段：

```powershell
& .\run_paper_experiments.ps1 -Stage table2 -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" -Seeds @(45,46)
& .\run_paper_experiments.ps1 -Stage ppo_probe -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" -Seeds @(45,46)
& .\run_paper_experiments.ps1 -Stage table3 -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" -Seeds @(45,46)
```

不同seed会生成不同目录，不会覆盖42–44。

## 6. 阶段三：预算实验（表4-1和表4-2）

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage budget `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -BudgetPolicy ppo_frozen `
  -Seeds 42,43,44
```

`BudgetPolicy`应在表3完成后替换为表现最好的策略，可选
`greedy`、`linucb`、`double_dqn`、`ppo_online`或`ppo_frozen`。
预算表始终将该策略与Random在相同K、R和种子下比较。一个实验根目录只运行
一次budget阶段；若要比较第二个赢家候选，应使用新的完整实验根目录，避免重复
的Random运行目录发生冲突。

### 表4-1

固定R=5，K取1、5、10、20。K=10结果复用核心阶段，不重复运行。
每个K比较Random和Frozen PPO。

```text
table04_1_budget_r5.csv
fig07_table04_1_budget_r5.png/pdf
```

验证目标：每轮目标数量增加后，攻击效果和图修改成本如何变化；Frozen PPO
是否在相同预算下持续优于Random。表中额外报告 `DetectorQueries`
和 `RuntimeMinutes`，用于比较达到相应攻击效果所需的查询与时间成本。

### 表4-2

按用户最终设定固定R=10，K取1、3、5、10。每个K比较Random和Frozen PPO。

```text
table04_2_budget_r10.csv
fig08_table04_2_budget_r10.png/pdf
```

验证目标：在较长攻击过程下，不同每轮目标数量对四项指标、检测器查询次数
和运行时间的影响。

表4-1和表4-2的四项核心指标按多个种子分别报告mean、std和有效样本数；
查询次数和运行时间各占一列并写成 `mean +/- std`。这两列属于成本，
不计入四项攻击效果指标。

注意：表4-1和4-2都是Top-K扫描，并不能单独隔离“轮数变化”的因果效果；
它们能够比较R=5和R=10两个预算区域。如果以后要单独研究轮数，应固定K=10
并令R取1、3、5、10。

### 第二个停止/总结点

预算阶段完成后检查：

- 增加预算是否总体提高DirectTP-ASR或Amplification；
- Frozen PPO是否在多个预算点保持优势，而非只在K=10有效；
- GMR增长是否符合实际编辑数量；
- AAB是否下降或保持稳定；
- `Actual Unique Targets`与计划K×R是否存在较大差异。

若方法只在单个预算点有效，应把结论限定为该预算，不能声称预算鲁棒性。

## 7. 阶段四：总体消融（表5）

先运行seed 42并检查，再补seed 43、44：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage ablation `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_final_v1" `
  -Seeds 42

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage ablation `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_final_v1" `
  -Seeds 43,44
```

表5包含：

| Variant | 实际配置 |
|---|---|
| Full LinUCB | 复用表3的LinUCB + Margin结果，不重新运行 |
| No Context | 将50维状态替换为常量上下文，退化为无上下文UCB式决策 |
| No Exploration | 令LinUCB探索系数alpha=0 |
| No Candidate Selector | 在全部语义合法候选中均匀随机选择端点 |
| No Reward Feedback | 奖励权重全部置0，保留相同更新时机 |
| No Macro Action | 宏动作长度由5改为1，每添加一条边即查询反馈 |

生成：

```text
table05_ablation.csv
fig09_table05_ablation.png/pdf
fig10_reward_components.png/pdf
```

验证目标是分别确认上下文状态、UCB探索项、候选节点选择、奖励反馈和宏动作
批处理对LinUCB攻击效果与查询成本的贡献。No Macro Action会显著增加检测器查询，
因此固定R=5、K=10及每目标最大编辑数，并通过AQT单独衡量其时间代价。

### 第三个停止/总结点

完成表5后，如果出现以下整体模式，可以总结当前实现支持研究假设：

1. Full LinUCB相对至少多数移除项具有更高DASR或更低AAB；
2. 去掉上下文或探索项后效果下降，说明LinUCB不是固定启发式；
3. 随机合法候选使效果或成本恶化，说明候选选择器有效；
4. 去掉奖励反馈后学习退化；去掉宏动作后AQT或查询次数显著上升；
5. 上述方向在seed 42、43、44上总体一致。

此时可以完成MAGIC+THEIA案例研究的消融结论，再开展跨检测器实验。

## 8. 表1：跨数据集、跨检测器（最后完成）

表1按用户要求不含Strategy列。汇总脚本只选择：

```text
action_policy=ppo_frozen
target_policy=threshold_margin
experiment_variant=full_model
K=10, R=5
```

然后仅按Dataset和Detector分组：

| Dataset | Detector | 四项主指标 | AUC before/after | F1 before/after | Precision before/after | Recall before/after |
|---|---|---|---|---|---|---|

输出：

```text
table01_cross_detector_dataset.csv
fig03_table01_attack_metrics.png/pdf
fig04_table01_detector_metrics.png/pdf
```

每个数据集/检测器必须先独立训练自己的PPO checkpoint，再运行Frozen评估。
不能把CADETS/MAGIC的checkpoint直接用于其他检测器并声称泛化。FLASH还需要
对应的 `--flash-semantic-json`。

在正式运行表1前，应先对每个组合做K=1、R=1冒烟，检查模型、数据、阈值、
标签、候选类型和状态维度是否适配。表1的完整跨检测器矩阵应在这些资源准备
完毕后单独执行。

## 9. 随时重新汇总

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage aggregate `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_v1" `
  -Seeds 42,43,44
```

也可直接执行：

```powershell
$py = "D:\App\anaconda\envs\stgan\python.exe"
& $py MAGIC_main\compare_experiments.py `
  --results-dir attack_results\paper_suite_theia_v1\runs `
  --output attack_results\paper_suite_theia_v1\paper_outputs\comparison_all.csv

& $py MAGIC_main\paper_outputs.py `
  --results-dir attack_results\paper_suite_theia_v1\runs `
  --output-dir attack_results\paper_suite_theia_v1\paper_outputs `
  --main-top-k 10 `
  --main-rounds 5
```

## 10. 一次运行全部阶段

不建议第一次直接运行全部，因为耗时很长，且失去阶段性停止机会。确认核心
实验有效后才可使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage all `
  -ExperimentRoot "attack_results\paper_suite_macro_all_v1" `
  -Seeds 42,43,44
```

`all`必须使用一个从未存在的新目录。
