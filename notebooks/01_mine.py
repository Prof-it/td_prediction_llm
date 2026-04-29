"""01 — Repository mining & feature extraction.

This notebook is thin: it delegates to `td_prediction.mining`. The heavy
feature-extraction code lives in the package so it can also be called
from a CLI or a unit test.

Two modes:
- `FORCE = False` (default): reuse the CSVs under `data/features_*.csv`.
  Useful for reproducing paper results without re-cloning the 5 repos.
- `FORCE = True`: clone repos locally under `repos/<name>/` and re-mine.
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
    # 01 — Repository Mining

    Mines commits from 5 Python OSS repos (requests, fastapi, scrapy, flask, keras)
    and emits `data/features_<repo>.csv` containing size/complexity/process metrics
    plus the regex-based SATD label. **Defaults to cached mode** — existing CSVs
    are reused. Set `FORCE = True` to re-mine from local clones.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    # Make the `src/` package importable when running from notebooks/.
    ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    return (ROOT,)


@app.cell
def _():
    from td_prediction import config, mining
    return config, mining


@app.cell
def _(mining):
    FORCE = False  # Set True to re-mine; requires repos/<name>/ clones.
    paths = mining.mine_all(force=FORCE)
    for p in paths:
        print(p)
    return (paths,)


@app.cell
def _(mining):
    df = mining.load_all_features()
    print("Rows:", len(df))
    print("Per-repo:")
    print(df.groupby("repo_id").size())
    print("SATD label positive rate:", df["label_td_satd"].mean())
    return (df,)


if __name__ == "__main__":
    app.run()
