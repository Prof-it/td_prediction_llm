"""Decision-threshold tuning on the VALIDATION set.

The original thesis tuned the classification threshold by maximizing F1
on the test set — that leaks test information into the final metric.
Here we expose `tune_threshold(y_val, p_val)` and `tune_threshold_for(...)`
which only see validation data; the chosen threshold is then frozen for
the test-set evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve


@dataclass
class ThresholdChoice:
    threshold: float
    objective: str
    score: float
    precision: float
    recall: float


def tune_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    *,
    objective: str = "f1",
) -> ThresholdChoice:
    """Return the threshold maximizing `objective` on validation data.

    Supported objectives:
    - "f1"  — F1 on the Debt class.
    - "mcc" — Matthews correlation coefficient.
    - "youden" — Youden's J (tpr - fpr).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    if objective == "f1":
        scores = []
        for t in thresholds:
            scores.append(f1_score(y_val, (p_val >= t).astype(int), zero_division=0))
        idx = int(np.argmax(scores))
        return ThresholdChoice(
            threshold=float(thresholds[idx]),
            objective="f1",
            score=float(scores[idx]),
            precision=float(precisions[idx]),
            recall=float(recalls[idx]),
        )
    if objective == "mcc":
        scores = []
        for t in thresholds:
            scores.append(matthews_corrcoef(y_val, (p_val >= t).astype(int)))
        idx = int(np.argmax(scores))
        return ThresholdChoice(
            threshold=float(thresholds[idx]),
            objective="mcc",
            score=float(scores[idx]),
            precision=float(precisions[idx]),
            recall=float(recalls[idx]),
        )
    if objective == "youden":
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_val, p_val)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return ThresholdChoice(
            threshold=float(thresholds[idx]),
            objective="youden",
            score=float(j[idx]),
            precision=float("nan"),
            recall=float(tpr[idx]),
        )
    raise ValueError(f"unknown objective {objective}")
