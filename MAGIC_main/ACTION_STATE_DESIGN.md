# Semantic action and state design

## Design contract

- Policy output width is fixed at 9.
- Each action slot represents one edge-event family, not an implementation
  mechanism such as "add a node".
- A candidate is legal only if the complete
  `(source node type, edge type, destination node type)` triple belongs to the
  active dataset schema.
- The target may be either the source or destination. Direction is resolved
  from its node type and the candidate node type.
- Invalid actions are masked before Random, Greedy, or PPO selects an action.
- Existing and pending duplicate directed edges are rejected.
- The threat model is edge-injection-only: every action connects two existing
  nodes, and node creation is not part of the policy or graph mutator.

## PPO宏动作与信用分配

PPO现在只学习“选择哪一种语义边类型”，不再学习候选节点列表中的临时
下标。候选下标在不同目标和不同动作下含义会变化，把它作为策略输出会造成
不稳定的信用分配，也是旧实现长期接近均匀探索的重要原因之一。

一次PPO决策严格对应一个宏动作：

1. 根据目标节点类型生成9维合法动作掩码；
2. PPO、Random或Greedy只选择一次语义动作；
3. 用所有策略共享的确定性规则选择5个合法端点；
4. 连续插入5条相同语义类型的边；
5. 第5条边完成后查询检测器一次；
6. 以这次查询计算一个奖励，并记录一个PPO transition。

因此“动作、反馈、奖励、transition”的粒度完全一致，不会再把一个批次奖励
只记到第5个微动作上，也不会把5种不同动作错误地串成同一段GAE轨迹。
每个目标节点最多执行2个宏动作（共10条边）；目标结束时写入episode边界。
在线PPO至少积累8个宏动作再更新，实验结束时会强制更新剩余的完整
transition。单样本批次不再被优势标准化为0。

在线训练和冻结评估采用显式分离的动作选择协议：Online PPO从掩码后的
Categorical分布采样，以保持PPO的on-policy探索；Frozen PPO默认对同一
分布取argmax，从而确定性展示checkpoint学到的动作偏好。固定随机种子的
`--deterministic`只负责计算复现，不代表argmax选择。

日志中的关键字段：

- `MacroActionStart`：被选动作、5步重复约束、合法掩码、动作概率和熵；
- `Action`：宏动作内部每一条边的具体源点、终点和语义；
- `RewardBatch` / `RewardBreakdown`：一次宏动作的分数变化及奖励分解；
- `MacroTransition`：确认5条边只形成一个transition；
- `PPORollout`：跨目标累积的宏动作数量；
- `PPOMacroUpdate`：actor/critic loss、熵、KL、裁剪率、梯度和优势统计。
- `PolicyHealth` / `PerActionReward`：归一化熵、最大动作概率、Top1与Top2
  概率差、到均匀分布的KL、间接奖励命中率和分动作收益。

## CADETS actions

| ID | Action | Edge | Valid source -> destination node types |
|---:|---|---|---|
| 0 | OPEN_OBJECT | 6 EVENT_OPEN | Process -> File, Process -> Directory |
| 1 | CREATE_OBJECT | 1 EVENT_CREATE_OBJECT | Process -> UnixSocket/Pipe/Directory |
| 2 | WRITE_OBJECT | 4 EVENT_WRITE | Process -> File/Pipe/NetFlow |
| 3 | READ_OBJECT | 5 EVENT_READ | File/UnixSocket/Pipe/NetFlow -> Process |
| 4 | CLOSE_OBJECT | 0 EVENT_CLOSE | Process -> File/UnixSocket/Pipe/NetFlow |
| 5 | CONNECT_NETWORK | 2 EVENT_CONNECT | Process -> UnixSocket/NetFlow |
| 6 | ACCEPT_NETWORK | 3 EVENT_ACCEPT | Process -> UnixSocket/NetFlow |
| 7 | EXECUTE_FILE | 9 EVENT_EXECUTE | Process -> File |
| 8 | FORK_PROCESS | 8 EVENT_FORK | Process -> Process |

On the current CADETS `test0.pkl`, these triples cover 802,641 of 840,300
edges (95.52%) and all six node types.

## THEIA and TRACE

Numeric type IDs are dataset-local. CADETS IDs are never reused for THEIA or
TRACE. Until named semantic profiles are supplied, the code builds a
dataset-specific observed schema:

1. Select legal edge families to cover every node type that actually occurs.
2. Fill remaining slots with the highest-frequency edge families.
3. Keep every observed endpoint pair for a selected edge family.
4. Store a schema fingerprint in PPO checkpoints and reject a mismatched
   checkpoint.

Validation on current `test0.pkl` files:

| Dataset | Node feature dimension | Edge feature dimension | Selected-action edge coverage |
|---|---:|---:|---:|
| CADETS | 6 | 27 | 95.52% |
| THEIA | 5 | 17 | 98.66% |
| TRACE | 11 | 23 | 98.16% |

The fixed output width makes the architecture comparable, but action-slot
semantics remain dataset-specific. Train and evaluate separate PPO
checkpoints. Direct frozen-policy transfer between datasets requires a future
shared CDM event-name ontology.

## State space

The shared state is 50 dimensions. Eleven node-type slots are reserved because
TRACE has the largest current node feature dimension; CADETS uses the first 6
and THEIA the first 5.

| Dimensions | Features |
|---|---|
| 0-10 | Current target node type one-hot |
| 11 | Current anomaly score |
| 12 | Current threshold-margin rank |
| 13-14 | Current target in-degree and out-degree |
| 15 | Added edges for the target |
| 16 | Fraction of target modifications that are incoming edges |
| 17 | Remaining target action budget |
| 18-28 | Previous feedback target node type one-hot |
| 29-37 | Previous feedback action one-hot |
| 38 | Previous target score drop |
| 39 | Previous target threshold-crossing flag |
| 40 | Previous batch reward |
| 41-49 | Recent action-frequency distribution |

Detector adapters may append private features. Flash appends five semantic
features, so its PPO input dimension is 55; MAGIC and adapters without extra
features use 50.

All previous-context fields are detector-observable and do not use metadata
labels.

## Validation

```powershell
python MAGIC_main\validate_action_space.py --dataset cadets --test-index 0
python MAGIC_main\validate_action_space.py --dataset theia --test-index 0
python MAGIC_main\validate_action_space.py --dataset trace --test-index 0
```
