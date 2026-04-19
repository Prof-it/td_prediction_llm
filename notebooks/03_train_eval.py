"""03 — Training, threshold tuning on VAL, test-set evaluation.

Fixes from the thesis pipeline:
- Adds a dedicated validation split (60/20/20 instead of 70/30).
- Tunes the decision threshold on VAL, freezes it, and reports on TEST.
- Sweeps class-imbalance strategies (none, class_weight, SMOTE).
- Sweeps feature sets (all, no_maturity, change_only) to diagnose
  shortcut learning from maturity features.
- Drops `satd_delta` everywhere to remove the obvious label leakage
  between the SATD regex label and the training features.
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
    # 03 — Training & Evaluation

    Grid:
    - Models: Random Forest, LightGBM, XGBoost
    - Imbalance: none, class_weight, SMOTE
    - Feature sets: all, no_maturity, change_only

    Threshold tuned on VAL, evaluated on TEST. Label column is `label_llm`.
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
    import pandas as pd
    from td_prediction import config
    from td_prediction.data.splits import FeatureSet, make_time_splits, make_lopo_splits
    from td_prediction.models import trainer
    return (
        FeatureSet,
        config,
        make_lopo_splits,
        make_time_splits,
        pd,
        trainer,
    )


@app.cell
def _(config, pd):
    # Load the LLM-labeled dataset produced by 02_label.
    labeled_csv = config.PATHS.data / "features_with_llm_labels.csv"
    df = pd.read_csv(labeled_csv)
    df["commit_dt"] = pd.to_datetime(df["commit_date"], utc=True)
    print("Shape:", df.shape, "| positive rate (label_llm):", df["label_llm"].mean())
    return (df,)


@app.cell
def _(FeatureSet, df, make_time_splits, trainer):
    # ----- Time-based split grid -----
    split = make_time_splits(df)
    print("Time split sizes:", split.sizes())
    bundles_time = trainer.sweep(
        split=split,
        label_col="label_llm",
        model_kinds=("rf", "lgbm", "xgb"),
        imbalance_strategies=("none", "class_weight", "smote"),
        feature_sets=(FeatureSet.ALL, FeatureSet.NO_MATURITY, FeatureSet.CHANGE_ONLY),
        verbose=False,
    )
    return (bundles_time, split)


@app.cell
def _(bundles_time, config, trainer):
    df_time = trainer.bundles_to_df(bundles_time)
    out_time = config.PATHS.results / "metrics_time.csv"
    df_time.to_csv(out_time, index=False)
    print(f"Saved {out_time}")
    df_time.sort_values(["feature_set", "imbalance", "model", "phase"])
    return (df_time,)


@app.cell
def _(FeatureSet, df, make_lopo_splits, trainer):
    # ----- Leave-one-project-out -----
    lopo_splits = make_lopo_splits(df)
    lopo_bundles = []
    for s in lopo_splits:
        bs = trainer.sweep(
            split=s,
            label_col="label_llm",
            model_kinds=("rf", "lgbm", "xgb"),
            imbalance_strategies=("none", "class_weight"),
            feature_sets=(FeatureSet.ALL, FeatureSet.NO_MATURITY),
        )
        lopo_bundles.extend(bs)
    return (lopo_bundles,)


@app.cell
def _(config, lopo_bundles, trainer):
    df_lopo = trainer.bundles_to_df(lopo_bundles)
    out_lopo = config.PATHS.results / "metrics_lopo.csv"
    df_lopo.to_csv(out_lopo, index=False)
    print(f"Saved {out_lopo}")
    df_lopo.sort_values(["split", "feature_set", "imbalance", "model", "phase"])
    return (df_lopo,)


@app.cell
def _(bundles_time, config, trainer):
    # Persist the single-best time-split bundle per (model, feature_set=all).
    saved = []
    for b in bundles_time:
        if b.feature_set == "all" and b.imbalance == "class_weight":
            p = trainer.save_bundle(b, config.PATHS.models)
            saved.append(p)
    print("Saved bundles:", saved)
    return


if __name__ == "__main__":
    app.run()
