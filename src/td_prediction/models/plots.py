"""Plotting helpers for PR / ROC curves and class-balance visualizations.

Separated from training so notebooks can call these without pulling in
the full sklearn/imblearn stack when they only need to replot saved
predictions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def pr_curve(y_true, y_proba, *, title: str, out_path: Path, best_threshold: float | None = None) -> Path:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, label="PR curve")
    if best_threshold is not None and len(thresholds) > 0:
        # Find nearest index to the given threshold.
        idx = int(np.argmin(np.abs(thresholds - best_threshold)))
        ax.scatter(recalls[idx], precisions[idx], color="red",
                   label=f"t={best_threshold:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(); ax.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def roc_plot(y_true, y_proba, *, title: str, out_path: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend(); ax.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def class_balance_grid(
    counts_by_strategy: dict[str, pd.Series],
    *,
    out_path: Path,
    title_prefix: str = "",
) -> Path:
    """One grayscale bar chart per strategy in `counts_by_strategy`."""
    fig, axes = plt.subplots(1, len(counts_by_strategy), figsize=(4 * len(counts_by_strategy), 4))
    hatches = ["///", "\\\\\\", "xxx", "---", "+++", "..."]
    if len(counts_by_strategy) == 1:
        axes = [axes]
    for ax, (name, counts), hatch in zip(axes, counts_by_strategy.items(), hatches):
        ax.bar(counts.index.astype(str), counts.values,
               color="lightgray", hatch=hatch, edgecolor="black")
        ax.set_title(f"{title_prefix}{name}")
        ax.set_ylabel("Number of commits")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
