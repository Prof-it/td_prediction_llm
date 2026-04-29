"""05 — Analysis: class imbalance, label distribution, gold-set review, token usage.

Collects the miscellaneous analyses that were scattered across the
thesis notebook into one place. None of this changes the models; it
produces tables and plots for the paper.
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
    # 05 — Analysis

    - Label distribution (SATD vs LLM) overall and per repo.
    - Class-imbalance visualization: unbalanced / SMOTE / undersample / SMOTEENN.
    - LOPO splits summary table.
    - Human gold-set sampler.
    - LLM token-usage accounting.
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
    import json
    import glob
    import pandas as pd
    from td_prediction import config
    from td_prediction.data.splits import FeatureSet, make_time_splits, make_lopo_splits, prepare_xy
    from td_prediction.labeling.human_review import sample_for_review, write_review_sheet
    from td_prediction.models import plots
    return (
        FeatureSet,
        config,
        glob,
        json,
        make_lopo_splits,
        make_time_splits,
        pd,
        plots,
        prepare_xy,
        sample_for_review,
        write_review_sheet,
    )


@app.cell
def _(config, pd):
    df = pd.read_csv(config.PATHS.data / "features_with_llm_labels.csv")
    df["commit_dt"] = pd.to_datetime(df["commit_date"], utc=True)
    summary = df.groupby("repo_id").agg(
        n=("commit_uid", "size"),
        satd_pos=("label_satd", "sum"),
        satd_rate=("label_satd", "mean"),
        llm_pos=("label_llm", "sum"),
        llm_rate=("label_llm", "mean"),
    )
    out = config.PATHS.results / "label_distribution.csv"
    summary.to_csv(out)
    print(f"Saved {out}")
    summary
    return (df, summary)


@app.cell
def _(FeatureSet, config, df, make_time_splits, plots, prepare_xy):
    # Class-imbalance visualization.
    split = make_time_splits(df)
    X_tr, y_tr, _ = prepare_xy(split.train, label_col="label_llm", feature_set=FeatureSet.ALL)
    counts = {}
    counts["Unbalanced"] = y_tr.value_counts().rename(index={0: "No Debt", 1: "Debt"}).sort_index()

    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    _, y_s = SMOTE(random_state=config.SEED).fit_resample(X_tr, y_tr)
    _, y_u = RandomUnderSampler(random_state=config.SEED).fit_resample(X_tr, y_tr)
    _, y_se = SMOTEENN(random_state=config.SEED).fit_resample(X_tr, y_tr)
    import pandas as pd
    counts["SMOTE"] = pd.Series(y_s).value_counts().rename(index={0: "No Debt", 1: "Debt"}).sort_index()
    counts["Undersample"] = pd.Series(y_u).value_counts().rename(index={0: "No Debt", 1: "Debt"}).sort_index()
    counts["SMOTEENN"] = pd.Series(y_se).value_counts().rename(index={0: "No Debt", 1: "Debt"}).sort_index()

    out_png = config.PATHS.figures / "class_balance.png"
    plots.class_balance_grid(counts, out_path=out_png)
    print(f"Saved {out_png}")
    return (counts,)


@app.cell
def _(config, df, make_lopo_splits, pd):
    # LOPO splits summary table.
    rows = []
    for s in make_lopo_splits(df):
        held_out = s.name.replace("lopo_", "")
        rows.append({
            "Fold": len(rows) + 1,
            "Test Project": held_out,
            "#Commits (Train)": len(s.train),
            "#Commits (Val)": len(s.val),
            "#Commits (Test)": len(s.test),
            "Train Projects": ", ".join(sorted(s.train["repo_id"].unique())),
        })
    lopo_df = pd.DataFrame(rows)
    out = config.PATHS.splits / "lopo_splits_summary_v2.csv"
    lopo_df.to_csv(out, index=False)
    print(f"Saved {out}")
    lopo_df
    return (lopo_df,)


@app.cell
def _(config, df, sample_for_review, write_review_sheet):
    # Human gold set sampler — produces 100 commits in a 2x2 stratified design.
    review = sample_for_review(df, n_per_cell=25)
    out = config.PATHS.results / "human_review_sheet.csv"
    write_review_sheet(review, out)
    print(f"Saved {out} — fill in 'label_human' (0/1) and return.")
    review.head()
    return (review,)


@app.cell
def _(config, glob, json):
    # LLM token accounting across all cached outputs.
    files = sorted(config.PATHS.llm_batch.glob("*_output.jsonl"))
    total_in = total_out = 0
    for path in files:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    u = json.loads(line).get("response", {}).get("body", {}).get("usage", {})
                    total_in += u.get("prompt_tokens", 0)
                    total_out += u.get("completion_tokens", 0)
                except json.JSONDecodeError:
                    pass
    print(f"Total prompt tokens: {total_in:,}")
    print(f"Total output tokens: {total_out:,}")
    return


if __name__ == "__main__":
    app.run()
