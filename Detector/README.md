# Detector adapters

所有适配器实现 `BaseDetector.query(graph)`，返回：

```text
all_scores       节点 ID -> 连续异常分数
native_alerts    原生告警节点集合
top_k_nodes      按分数排序的节点
threshold        攻击前校准并冻结的阈值（如适用）
metadata         模型、数据集和告警模式
```

MAGIC 和 ProvGuard 使用表示空间距离分数。ProvGuard 的适配器读取
`ProvGuard_main/data/eval_result/*_trained_params.pkl` 中的均值、KMeans
中心和全局距离，并要求对应的原生模型 checkpoint 存在。

Flash 和 STGAN 使用原生分类 logits，不再使用伪造的 KNN 分数。Flash 的
适配器复用 notebook 中的 SAGE/GCN 结构、数据集 Word2Vec 和
`lword2vec_gnn_*_E3.pth` 权重；查询前必须通过 `attach_semantic_state()`
提供节点短语旁路信息。STGAN 复用 `STGAN_master/model` 和各数据集的
`gat0/tgn0/selfattention0` 权重；当前实现返回分类 NLL/无标签不确定性，
若需要严格原生误分类告警，输入图必须带 `y` 标签或提供原生标签映射。

攻击者只能使用 `all_scores`、`native_alerts` 和公开元数据；真实标签只用于
评估 `TP_0`、最终 ASR 和副作用指标。
