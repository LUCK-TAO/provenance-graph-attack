# 实验运行与复现说明

本文档只用于工程复现，不作为论文正文或附录内容。当前正式实验采用
THEIA 数据集、MAGIC 检测器和 `STGAN` Conda 环境，并暂不运行 Frozen PPO。

## 1. 基本路径

在 PowerShell 中进入项目目录：

```powershell
cd D:\Python\attack-main
```

统一使用以下解释器：

```text
D:\App\anaconda\envs\stgan\python.exe
```

当前正式实验根目录：

```text
attack_results\paper_suite_theia_final_v1
```

如需从零复现，建议使用新的目录，避免与现有结果混合：

```text
attack_results\paper_suite_theia_reproduction_v1
```

## 2. 运行前检查

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage check `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 42 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

该阶段检查Python语法、数据集动作空间、攻击目标、LinUCB、Double DQN、
Benign-Mimicry及PPO基础接口，不启动正式攻击。

## 3. 从零开始的冒烟测试

`smoke` 会创建实验根目录，因此只能对不存在的新目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage smoke `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 42 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

现有 `paper_suite_theia_final_v1` 已完成冒烟测试，不要再次对该目录执行
`-Stage smoke`。

## 4. 表2：攻击目标选择策略

固定Random动作策略、5轮攻击、每轮10个目标，比较Margin、Highest Score和
Random Target。脚本会按种子轮换策略执行顺序，降低运行时段和机器负载造成的
时间偏差。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage table2 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 42,43,44 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

## 5. 表3第一阶段：seed=42的Online PPO

该阶段要求同一根目录中已存在表2的Random+Margin结果。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage ppo_probe `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 42 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

完成后应检查对应 `summary.json` 和 `run.log` 中：PPO更新次数大于0、选择模式为
`sample`，并检查最终动作熵、动作概率、PPO loss和奖励分解。

## 6. 表3第二阶段：seed=42的其余基线

该命令复用表2的Random+Margin和上一步的Online PPO，只运行No Attack、
Reward-Greedy、LinUCB、Online Double DQN和Adapted Benign-Mimicry：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage table3 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 42 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

先比较seed=42的所有策略。确认奖励、查询、更新和四项核心指标合理后，再运行
另外两个种子。

## 7. 表3第三阶段：seed=43和44

先运行Online PPO：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage ppo_probe `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 43,44 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

再运行其余基线：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage table3 `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 43,44 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

## 8. 表4：攻击预算敏感性

当前不使用Frozen PPO，因此显式指定 `ppo_online`。该阶段包含固定5轮时
Top-K为1、5、10、20，以及固定10轮时Top-K为1、3、5、10的实验。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage budget `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -BudgetPolicy ppo_online `
  -Seeds 42,43,44 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

## 9. 消融实验注意事项

当前 `-Stage ablation` 仍采用“Online训练后Frozen评估”的旧流程。由于当前研究
阶段已决定暂不运行Frozen PPO，因此不要执行该阶段，也不要执行 `-Stage all`。
待消融协议改为统一的Online评价或重新确认Frozen预训练协议后，再补充正式命令。

## 10. 汇总已有结果

任何阶段完成后均可单独重新生成汇总表格和图片：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_paper_experiments.ps1 `
  -Stage aggregate `
  -Dataset theia `
  -ExperimentRoot "attack_results\paper_suite_theia_reproduction_v1" `
  -Seeds 42,43,44 `
  -PythonPath "D:\App\anaconda\envs\stgan\python.exe"
```

主要输出位于：

```text
<ExperimentRoot>\runs\<run_name>\summary.json
<ExperimentRoot>\runs\<run_name>\trajectory.json
<ExperimentRoot>\runs\<run_name>\modifications.json
<ExperimentRoot>\runs\<run_name>\run.log
<ExperimentRoot>\paper_outputs\
```

## 11. 中断与续跑

脚本以单个完整基线为断点：若运行目录中存在完整 `summary.json`，再次执行同一
阶段会复用该结果；若在单个基线内部强制停止而只留下残缺目录，脚本会拒绝覆盖。
此时应先检查残缺目录和日志，确认后再决定是否删除该单个目录并重跑。不要删除
整个实验根目录，也不要使用 `-Stage smoke` 覆盖正式结果。

