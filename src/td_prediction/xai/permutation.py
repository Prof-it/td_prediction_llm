"""Permutation importance — a model-agnostic third lens alongside SHAP and LIME."""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from .. import config


def permutation_importance_df(model, X, y, *, n_repeats: int = 5, scoring: str = "average_precision") -> pd.DataFrame:
    """Return mean permutation importance per feature, sorted descending."""
    r = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=config.SEED,
        n_jobs=config.N_JOBS, scoring=scoring,
    )
    return pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compare_rankings(rankings: dict[str, pd.DataFrame], top_k: int = 15) -> pd.DataFrame:
    """Side-by-side top-K rankings from multiple XAI methods."""
    cols = {}
    for name, df in rankings.items():
        df = df.head(top_k).reset_index(drop=True)
        cols[f"{name}_feature"] = df["feature"].values
    return pd.DataFrame(cols)
