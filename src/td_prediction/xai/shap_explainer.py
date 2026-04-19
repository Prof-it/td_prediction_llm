"""SHAP explainability for tree models.

Produces:
- A global summary plot (beeswarm) per model.
- A feature-importance CSV (mean |SHAP|) so feature rankings can be
  compared against LIME and permutation importance.
- Optional local force plots for named instances.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def shap_values_for(model, X):
    """Compute SHAP values; normalize shape across RF / LGBM / XGB."""
    import shap
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    # RandomForestClassifier returns a list [class0, class1]; pick class1.
    if isinstance(values, list) and len(values) == 2:
        values = values[1]
    # Newer SHAP may return a 3D array (samples, features, classes).
    if isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[..., 1]
    return explainer, values


def summary_plot(model, X, *, title: str, out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import shap
    _, values = shap_values_for(model, X)
    shap.summary_plot(values, X, show=False)
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path


def feature_importance(model, X) -> pd.DataFrame:
    """Return mean absolute SHAP per feature, sorted descending."""
    _, values = shap_values_for(model, X)
    mean_abs = np.abs(values).mean(axis=0)
    return pd.DataFrame(
        {"feature": list(X.columns), "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
