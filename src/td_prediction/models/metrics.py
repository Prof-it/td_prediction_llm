"""Evaluation metrics reported per (model, split, feature_set, threshold).

The metric set is deliberately broad because technical-debt labels are
imbalanced. Reporting only accuracy or weighted-F1 would be misleading;
PR-AUC, MCC, and per-class F1 are the reviewer-defensible signals.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    model: str
    split: str
    feature_set: str
    label_col: str
    threshold: float
    p_debt: float
    r_debt: float
    f1_debt: float
    f1_macro: float
    f1_weighted: float
    f1_micro: float
    mcc: float
    balanced_acc: float
    roc_auc: float
    pr_auc_debt: float
    specificity: float
    tn: int
    fp: int
    fn: int
    tp: int
    n_samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    *,
    model,
    X,
    y_true,
    threshold: float = 0.5,
    split_name: str,
    model_name: str,
    feature_set: str,
    label_col: str,
    verbose: bool = False,
) -> EvalResult:
    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [0, 1])) > 1:
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
    else:
        y_pred = model.predict(X)
        y_proba = None
        roc_auc = float("nan")
        pr_auc = float("nan")

    p = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    res = EvalResult(
        model=model_name,
        split=split_name,
        feature_set=feature_set,
        label_col=label_col,
        threshold=float(threshold),
        p_debt=p, r_debt=r, f1_debt=f1,
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        balanced_acc=float(balanced_accuracy_score(y_true, y_pred)),
        roc_auc=roc_auc,
        pr_auc_debt=pr_auc,
        specificity=specificity,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n_samples=int(len(y_true)),
    )
    if verbose:
        print(f"\n[{model_name}] {split_name} | fs={feature_set} | t={threshold:.3f}")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        print(f"CM: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
        print(f"PR-AUC={pr_auc:.3f}  ROC-AUC={roc_auc:.3f}  MCC={res.mcc:.3f}  bal_acc={res.balanced_acc:.3f}")
    return res


def results_df(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in results])
