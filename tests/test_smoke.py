"""Smoke tests: the pipeline's non-I/O, non-network code paths must import
and execute on a tiny synthetic dataset without errors. Any reviewer
re-running the paper should be able to `pytest -q` and see green.

Run: `pytest -q tests/`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from td_prediction import config  # noqa: E402
from td_prediction.data.splits import (  # noqa: E402
    FeatureSet, feature_columns, make_lopo_splits, make_time_splits, prepare_xy,
)
from td_prediction.labeling.prompts import VARIANTS, v1_satd_filtered, v2_rubric_json  # noqa: E402
from td_prediction.labeling.satd import is_comment_or_docstring, strip_comments  # noqa: E402
from td_prediction.models.threshold import tune_threshold  # noqa: E402


def _synthetic_df(n_per_repo: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for repo in ("requests", "fastapi", "scrapy"):
        for i in range(n_per_repo):
            rows.append({
                "repo_id": repo,
                "commit_hash": f"{repo}{i:06x}",
                "commit_uid": f"{repo}#{i:06x}",
                "commit_date": f"2024-01-{(i % 28) + 1:02d}T00:00:00+00:00",
                "lines_added": int(rng.integers(0, 200)),
                "lines_deleted": int(rng.integers(0, 100)),
                "files_changed": int(rng.integers(1, 10)),
                "hunks": int(rng.integers(1, 30)),
                "n_methods_changed": int(rng.integers(0, 20)),
                "cc_delta_sum": float(rng.normal(0, 5)),
                "cc_delta_max": float(rng.normal(0, 3)),
                "complexity_current_sum": float(rng.normal(50, 20)),
                "churn_delta": int(rng.integers(0, 300)),
                "churn_cum": int(rng.integers(0, 5000)),
                "contributors_count": int(rng.integers(1, 5)),
                "contributors_cum": int(rng.integers(1, 30)),
                "n_authors_till_now": int(rng.integers(1, 30)),
                "n_commits_file_past90d": int(rng.integers(0, 20)),
                "commits_count_file": int(rng.integers(0, 100)),
                "contributors_experience": float(rng.random()),
                "history_complexity": float(rng.random()),
                "dmm_unit_complexity": float(rng.random()),
                "dmm_unit_size": float(rng.random()),
                "dmm_unit_interfacing": float(rng.random()),
                "satd_delta": int(rng.integers(-1, 3)),
                "label_satd": int(rng.integers(0, 2)),
                "label_llm": int(rng.integers(0, 2)),
            })
    df = pd.DataFrame(rows)
    df["commit_dt"] = pd.to_datetime(df["commit_date"], utc=True)
    return df


def test_config_paths_resolve():
    assert config.PATHS.root.exists()
    assert config.PATHS.data.exists()


def test_satd_helpers():
    assert is_comment_or_docstring("# TODO fix this")
    assert is_comment_or_docstring('"""docstring start')
    assert not is_comment_or_docstring("x = 1")
    diff = "+# TODO something\n+x = 1\n-y = 2\n-# HACK bad\n"
    stripped = strip_comments(diff)
    assert "TODO" not in stripped
    assert "x = 1" in stripped


def test_splits_and_feature_selection():
    df = _synthetic_df()
    split = make_time_splits(df, train_frac=0.6, val_frac=0.2)
    sizes = split.sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == len(df)
    assert sizes["train"] > sizes["val"] > 0
    cols_all = feature_columns(df, FeatureSet.ALL)
    assert "satd_delta" not in cols_all           # label leakage
    assert "commits_count_file" not in cols_all   # temporal leakage
    assert "contributors_experience" not in cols_all
    assert "history_complexity" not in cols_all
    assert "label_llm" not in cols_all
    cols_mat = feature_columns(df, FeatureSet.MATURITY_ONLY)
    assert "contributors_cum" in cols_mat         # still a maturity feature
    assert "commits_count_file" not in cols_mat   # moved to leakage


def test_lopo_splits_one_repo_in_test():
    df = _synthetic_df()
    splits = make_lopo_splits(df)
    for s in splits:
        assert len(s.test["repo_id"].unique()) == 1
        assert s.test["repo_id"].iloc[0] not in s.train["repo_id"].unique()


def test_prepare_xy_handles_nans():
    df = _synthetic_df()
    df.loc[0, "lines_added"] = np.nan
    X, y, cols, fill_vals = prepare_xy(df, label_col="label_llm", feature_set=FeatureSet.ALL)
    assert not X.isna().any().any()
    assert len(y) == len(df)
    assert "satd_delta" not in cols
    # val/test must use train fill values
    X2, _, _, _ = prepare_xy(df, label_col="label_llm", feature_cols=cols, fill_values=fill_vals)
    assert not X2.isna().any().any()


def test_tune_threshold_returns_reasonable_value():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    p = y_true + rng.normal(0, 0.3, size=200)  # monotone-ish
    p = (p - p.min()) / (p.max() - p.min())
    choice = tune_threshold(y_true, p, objective="f1")
    assert 0.0 <= choice.threshold <= 1.0
    assert 0.0 <= choice.score <= 1.0


@pytest.mark.parametrize("variant_name", list(VARIANTS))
def test_prompt_variants_build_messages(variant_name):
    df = _synthetic_df(n_per_repo=1)
    row = df.iloc[0].to_dict()
    diff_text = "+x = 1\n-y = 2\n"
    messages = VARIANTS[variant_name](row, diff_text)
    assert isinstance(messages, list) and messages
    assert all("content" in m and "role" in m for m in messages)


def test_v2_prompt_mentions_rubric_and_json():
    df = _synthetic_df(n_per_repo=1)
    messages = v2_rubric_json(df.iloc[0].to_dict(), "diff")
    system = messages[0]["content"]
    assert "Technical debt" in system
    assert "JSON" in system


def test_metrics_evaluate_on_synthetic():
    from sklearn.ensemble import RandomForestClassifier

    from td_prediction.models.metrics import evaluate

    df = _synthetic_df()
    split = make_time_splits(df)
    X_tr, y_tr, cols, fill_vals = prepare_xy(split.train, label_col="label_llm")
    X_te, y_te, _, _ = prepare_xy(split.test, label_col="label_llm", feature_cols=cols, fill_values=fill_vals)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_tr, y_tr)
    res = evaluate(
        model=model, X=X_te, y_true=y_te, threshold=0.5,
        split_name="smoke", model_name="rf",
        feature_set="all", label_col="label_llm",
    )
    assert res.n_samples == len(y_te)
    assert 0.0 <= res.f1_debt <= 1.0
