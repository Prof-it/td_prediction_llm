"""04 — Explainability: SHAP + LIME + permutation importance.

Three XAI methods are used as cross-checks. For the paper, agreement
across at least two of {SHAP, LIME, permutation} is the criterion we use
to call a feature "robustly important".
"""
import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 — Explainability

    - **SHAP** global beeswarm + mean |SHAP| ranking.
    - **LIME** 5 local case studies + sampled global ranking.
    - **Permutation importance** as a third, model-agnostic lens.
    - Top-15 side-by-side table for cross-method agreement.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    return (ROOT,)


@app.cell
def _():
    import joblib
    import pandas as pd
    from td_prediction import config
    from td_prediction.data.splits import FeatureSet, make_time_splits, prepare_xy
    from td_prediction.xai import shap_explainer, lime_explainer, permutation
    return (
        FeatureSet,
        config,
        joblib,
        lime_explainer,
        make_time_splits,
        pd,
        permutation,
        prepare_xy,
        shap_explainer,
    )


@app.cell
def _(config, joblib, pd):
    # Load a persisted bundle from 03_train_eval. Change model_name as needed.
    bundle_path = next(config.PATHS.models.glob("time__lgbm__class_weight__all.joblib"), None)
    if bundle_path is None:
        raise FileNotFoundError("Run 03_train_eval first to create a bundle.")
    bundle = joblib.load(bundle_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    meta = bundle["meta"]
    print("Loaded:", meta)
    labeled_csv = config.PATHS.data / "features_with_llm_labels.csv"
    df = pd.read_csv(labeled_csv)
    df["commit_dt"] = pd.to_datetime(df["commit_date"], utc=True)
    return (bundle, df, feature_cols, meta, model)


@app.cell
def _(FeatureSet, df, feature_cols, make_time_splits, prepare_xy):
    split = make_time_splits(df)
    X_tr, y_tr, _ = prepare_xy(split.train, label_col="label_llm",
                               feature_set=FeatureSet.ALL, feature_cols=feature_cols)
    X_te, y_te, _ = prepare_xy(split.test, label_col="label_llm",
                               feature_set=FeatureSet.ALL, feature_cols=feature_cols)
    return (X_te, X_tr, y_te, y_tr)


@app.cell
def _(X_te, config, meta, model, shap_explainer):
    # Global SHAP
    fig_path = config.PATHS.figures / f"shap_summary_{meta['model_name']}.png"
    shap_explainer.summary_plot(model, X_te, title=f"SHAP — {meta['model_name']}", out_path=fig_path)
    print(f"Saved {fig_path}")
    shap_ranking = shap_explainer.feature_importance(model, X_te)
    shap_ranking.to_csv(config.PATHS.results / f"shap_ranking_{meta['model_name']}.csv", index=False)
    shap_ranking.head(15)
    return (shap_ranking,)


@app.cell
def _(X_te, X_tr, config, lime_explainer, meta, model):
    # Local LIME for 5 instances
    local_paths = lime_explainer.local_explanations(
        model, X_tr, X_te, indices=[0, 1, 2, 3, 4],
        out_dir=config.PATHS.figures, prefix=f"lime_{meta['model_name']}",
    )
    print("Local LIME files:", local_paths)

    # Sampled global LIME
    lime_ranking = lime_explainer.global_importance_sampled(
        model, X_tr, X_te, n_sample=80,
    )
    lime_ranking.to_csv(config.PATHS.results / f"lime_ranking_{meta['model_name']}.csv", index=False)
    lime_ranking.head(15)
    return (lime_ranking,)


@app.cell
def _(X_te, config, meta, model, permutation, y_te):
    perm_ranking = permutation.permutation_importance_df(
        model, X_te, y_te, n_repeats=5, scoring="average_precision",
    )
    perm_ranking.to_csv(config.PATHS.results / f"perm_ranking_{meta['model_name']}.csv", index=False)
    perm_ranking.head(15)
    return (perm_ranking,)


@app.cell
def _(config, lime_ranking, meta, permutation, perm_ranking, shap_ranking):
    comparison = permutation.compare_rankings(
        {"shap": shap_ranking, "lime": lime_ranking, "perm": perm_ranking},
        top_k=15,
    )
    comparison.to_csv(config.PATHS.results / f"xai_topk_{meta['model_name']}.csv", index=False)
    comparison
    return (comparison,)


if __name__ == "__main__":
    app.run()
