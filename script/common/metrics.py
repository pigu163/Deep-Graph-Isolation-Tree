"""统一的无监督异常检测评估指标（异常分数越大越异常）。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["compute_metrics"]


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None,
                    review_rates: tuple[float, ...] = (0.001, 0.005, 0.01)) -> dict:
    """Report global ranking plus fixed review-budget metrics without threshold tuning."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("length mismatch: y_true and y_score must be aligned 1-D arrays")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains non-finite values")
    known = y_true != -1
    y_true, y_score = y_true[known], y_score[known]
    n_eval, n_pos = len(y_true), int((y_true == 1).sum())
    warnings: list[str] = []
    if n_eval == 0:
        return {"roc_auc": None, "pr_auc": None, "prevalence": None, "top_k": {},
                "precision_at_k": None, "recall_at_k": None, "f1_at_k": None,
                "n_eval": 0, "n_pos": 0, "warnings": ["no known labels in split"]}
    order = np.argsort(-y_score, kind="stable")
    prevalence = n_pos / n_eval

    def at(count: int) -> dict:
        count = max(1, min(int(count), n_eval))
        hits = int((y_true[order[:count]] == 1).sum())
        precision = hits / count
        recall = hits / n_pos if n_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"k": count, "precision": precision, "recall": recall, "f1": f1,
                "lift": precision / prevalence if prevalence else None}

    default = at(n_pos if k is None else k)
    top_k = {f"{rate:g}": at(int(np.ceil(n_eval * rate))) for rate in review_rates}
    out = {"precision_at_k": default["precision"], "recall_at_k": default["recall"],
           "f1_at_k": default["f1"], "k": default["k"], "prevalence": prevalence,
           "top_k": top_k, "n_eval": n_eval, "n_pos": n_pos, "warnings": warnings}
    if len(np.unique(y_true)) < 2:
        warnings.append("single class in evaluated split; ROC-AUC/PR-AUC are null")
        out.update(roc_auc=None, pr_auc=None)
    else:
        out.update(roc_auc=float(roc_auc_score(y_true, y_score)),
                   pr_auc=float(average_precision_score(y_true, y_score)))
    return out
