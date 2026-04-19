"""LIME local explanations + a sampled global LIME view.

LIME fits a local linear surrogate per instance. In the original
thesis LIME was an iteration-3 afterthought; here it becomes a first-
class citizen alongside SHAP. We also compute a "sampled global" view
by averaging |LIME weight| across a stratified sample of test instances,
which gives a second global-importance lens complementary to SHAP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_explainer(X_train: pd.DataFrame):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=["No Debt", "Debt"],
        mode="classification",
        discretize_continuous=False,
    )


def local_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    indices,
    out_dir: Path,
    prefix: str = "lime",
) -> list[Path]:
    explainer = make_explainer(X_train)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in indices:
        row = X_test.iloc[i].values
        exp = explainer.explain_instance(
            data_row=row,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=X_test.columns)),
        )
        p = out_dir / f"{prefix}_instance_{i}.html"
        exp.save_to_file(p)
        paths.append(p)
    return paths


def global_importance_sampled(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    n_sample: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Approximate global feature importance by averaging local |weight|
    across a random sample of `n_sample` test instances.

    This is the standard "aggregated LIME" view — slower than SHAP but a
    useful cross-check because the surrogate models are fundamentally
    different.
    """
    explainer = make_explainer(X_train)
    rng = np.random.default_rng(random_state)
    n_sample = min(n_sample, len(X_test))
    idx = rng.choice(len(X_test), size=n_sample, replace=False)

    weights = np.zeros(len(X_test.columns))
    counts = np.zeros(len(X_test.columns))
    col_to_idx = {c: i for i, c in enumerate(X_test.columns)}

    for i in idx:
        exp = explainer.explain_instance(
            data_row=X_test.iloc[i].values,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=X_test.columns)),
            num_features=len(X_test.columns),
        )
        for feat_label, w in exp.as_list():
            # LIME returns feature labels; with discretize_continuous=False
            # these are raw feature names.
            name = feat_label.split(" ")[0]
            if name in col_to_idx:
                weights[col_to_idx[name]] += abs(w)
                counts[col_to_idx[name]] += 1

    mean_abs = np.where(counts > 0, weights / np.maximum(counts, 1), 0.0)
    return pd.DataFrame(
        {"feature": list(X_test.columns), "mean_abs_lime": mean_abs, "n_support": counts.astype(int)}
    ).sort_values("mean_abs_lime", ascending=False).reset_index(drop=True)
