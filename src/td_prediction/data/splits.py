"""Train / validation / test splits.

Two split strategies are supported:

1. **Time-based, per-repo chronological** (`make_time_splits`).
   Within each repo, commits are sorted by date and the earliest
   `TRAIN_FRAC` form train, next `VAL_FRAC` form val, remainder is test.
   This avoids train/test contamination from future information within
   a project.

2. **Leave-One-Project-Out** (`make_lopo_splits`).
   For each fold one repo is the test set; the remaining four repos are
   split chronologically into train/val (no time-ordering across repos
   — each repo is sorted on its own timeline).

The classic thesis pipeline only had train/test. Having a separate val
set is critical because we tune the decision threshold and hyperparameters
on it: doing that on the test set (as iteration 3 did) is a form of
label leakage that inflates reported metrics.

Feature groups (configurable by `FeatureSet`):
- `all`          — every numeric column except metadata/labels/leakers.
- `change_only`  — change-size/complexity features (see config.CHANGE_COLS).
- `maturity_only` — history/maturity features (see config.MATURITY_COLS).
- `no_maturity`  — `all` minus maturity; a shortcut-learning ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


class FeatureSet(str, Enum):
    ALL = "all"
    CHANGE_ONLY = "change_only"
    MATURITY_ONLY = "maturity_only"
    NO_MATURITY = "no_maturity"


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _chrono_slice(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by commit_dt and slice into train/val/test."""
    df = df.sort_values("commit_dt")
    n = len(df)
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_va], df.iloc[n_tr + n_va:]


def make_time_splits(
    df: pd.DataFrame,
    *,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> Split:
    """Chronological train/val/test, sliced within each repo and concatenated."""
    if "commit_dt" not in df.columns:
        df = df.assign(commit_dt=pd.to_datetime(df["commit_date"], utc=True))
    tr_parts, va_parts, te_parts = [], [], []
    for _, g in df.groupby("repo_id"):
        tr, va, te = _chrono_slice(g, train_frac, val_frac)
        tr_parts.append(tr); va_parts.append(va); te_parts.append(te)
    return Split(
        train=pd.concat(tr_parts, ignore_index=True),
        val=pd.concat(va_parts, ignore_index=True),
        test=pd.concat(te_parts, ignore_index=True),
        name="time",
    )


def make_lopo_splits(
    df: pd.DataFrame,
    *,
    val_frac_within_train: float = 0.15,
) -> list[Split]:
    """One Split per held-out repo."""
    if "commit_dt" not in df.columns:
        df = df.assign(commit_dt=pd.to_datetime(df["commit_date"], utc=True))
    splits: list[Split] = []
    for held_out in sorted(df["repo_id"].unique()):
        test_df = df[df["repo_id"] == held_out].copy()
        train_pool = df[df["repo_id"] != held_out].copy()
        # Within each train repo, take the last `val_frac_within_train`
        # chronologically as validation.
        tr_parts, va_parts = [], []
        for _, g in train_pool.groupby("repo_id"):
            g = g.sort_values("commit_dt")
            n_val = max(1, int(val_frac_within_train * len(g)))
            tr_parts.append(g.iloc[:-n_val])
            va_parts.append(g.iloc[-n_val:])
        splits.append(Split(
            train=pd.concat(tr_parts, ignore_index=True),
            val=pd.concat(va_parts, ignore_index=True),
            test=test_df.reset_index(drop=True),
            name=f"lopo_{held_out}",
        ))
    return splits


def feature_columns(df: pd.DataFrame, feature_set: FeatureSet = FeatureSet.ALL) -> list[str]:
    """Return the list of columns to use as model input features."""
    forbidden = set(config.META_COLS) | set(config.LABEL_COLS) | set(config.LEAKAGE_COLS) | {
        "llm_agreement", "llm_confidence", "n_trials", "rationale_llm",
    }
    candidates = [c for c in df.columns if c not in forbidden]
    # Keep only numeric
    numeric = df[candidates].select_dtypes(include=[np.number]).columns.tolist()

    if feature_set is FeatureSet.ALL:
        return numeric
    if feature_set is FeatureSet.CHANGE_ONLY:
        return [c for c in numeric if c in config.CHANGE_COLS]
    if feature_set is FeatureSet.MATURITY_ONLY:
        return [c for c in numeric if c in config.MATURITY_COLS]
    if feature_set is FeatureSet.NO_MATURITY:
        return [c for c in numeric if c not in config.MATURITY_COLS]
    raise ValueError(f"unknown feature_set {feature_set}")


def prepare_xy(
    df: pd.DataFrame,
    *,
    label_col: str = "label_llm",
    feature_set: FeatureSet = FeatureSet.ALL,
    feature_cols: list[str] | None = None,
    fill_values: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """Return (X, y, cols, fill_values).

    `fill_values` maps column → mean used for NaN imputation. Pass the
    dict returned from the training call into val/test calls so imputation
    uses the training distribution, not the val/test distribution.
    """
    cols = feature_cols or feature_columns(df, feature_set)
    X = df[cols].copy()
    if fill_values is None:
        fill_values = X.mean(numeric_only=True).to_dict()
    X = X.fillna(fill_values)
    y = df[label_col].astype(int)
    return X, y, cols, fill_values


def save_split(split: Split, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        part.to_csv(directory / f"{split.name}_{name}.csv", index=False)


def load_split(directory: Path, name: str) -> Split:
    return Split(
        train=pd.read_csv(directory / f"{name}_train.csv"),
        val=pd.read_csv(directory / f"{name}_val.csv"),
        test=pd.read_csv(directory / f"{name}_test.csv"),
        name=name,
    )
