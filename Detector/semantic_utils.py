"""语义分类型检测器的公共黑盒工具。"""
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F


def graph_attr(graph, key: str, default=None):
    if hasattr(graph, 'ndata') and key in graph.ndata:
        return graph.ndata[key]
    if hasattr(graph, key):
        return getattr(graph, key)
    return default


def logits_to_result(logits: torch.Tensor, top_k: int, *, labels=None,
                     threshold: Optional[float] = None, model: str = '',
                     dataset: str = '', node_indices=None) -> Dict[str, Any]:
    """将原生分类 logits 转换为连续黑盒反馈。

    有标签时使用真实语义类别的 NLL；无标签时使用 ``1-max(p)``，并在
    metadata 中标注为 uncertainty-only，避免伪造“误分类”告警。
    """
    logits = logits.detach().float().cpu()
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    if labels is not None:
        labels = labels.detach().long().cpu() if torch.is_tensor(labels) else torch.as_tensor(labels).long()
        valid = (labels >= 0) & (labels < probs.shape[1])
    else:
        valid = None
    if labels is not None and bool(valid.all()):
        scores = -torch.log(probs[torch.arange(len(probs)), labels].clamp_min(1e-8))
        alerts = (pred != labels).nonzero(as_tuple=False).view(-1).tolist()
        mode = 'native_misclassification'
    else:
        scores = 1.0 - probs.max(dim=-1).values
        alerts = scores.topk(min(top_k, len(scores))).indices.tolist()
        mode = 'uncertainty_only_no_labels'
    all_scores = {int(i): float(v) for i, v in enumerate(scores)}
    visible = all_scores if node_indices is None else {int(i): all_scores[int(i)] for i in node_indices if int(i) in all_scores}
    ranked = sorted(visible.items(), key=lambda x: (-x[1], x[0]))
    return {
        'top_k_nodes': [i for i, _ in ranked[:top_k]],
        'anomaly_scores': [v for _, v in ranked[:top_k]],
        'all_scores': all_scores,
        'native_alerts': sorted(int(i) for i in alerts),
        'threshold': threshold,
        'score_direction': 'higher_is_more_anomalous',
        'detector_family': 'semantic_classifier',
        'metadata': {'model': model, 'dataset': dataset, 'alert_mode': mode},
    }
