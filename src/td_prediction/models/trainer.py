"""Training pipeline with pluggable class-imbalance strategies.

Primary entry point: `train_and_evaluate(...)` which takes a `Split`, a
label column, and a feature set, and runs a sweep over model types and
imbalance strategies, returning a results DataFrame plus the trained
models.

Imbalance strategies:
- `none`        — train as-is.
- `class_weight` — pass `class_weight='balanced'` (or equivalent weight) to the model.
- `smote`       — oversample minority on the training set only.
- `smoteenn`    — SMOTE + ENN hybrid on the training set only.

IMPORTANT: resampling is applied to TRAIN only; val and test remain
natural-distribution so reported metrics reflect the true class balance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .. import config
from ..data.splits import FeatureSet, Split, prepare_xy
from .metrics import EvalResult, evaluate
from .threshold import ThresholdChoice, tune_threshold


IMBALANCE_STRATEGIES = ["none", "class_weight", "smote", "smoteenn"]


def _new_model(kind: str, *, imbalance: str, n_pos: int, n_neg: int) -> Any:
    """Return an untrained classifier. `imbalance='class_weight'` configures
    the model's built-in handling; resampling strategies use the default
    config here and the resampling is done outside.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    use_class_weight = (imbalance == "class_weight")
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=250, random_state=config.SEED, n_jobs=config.N_JOBS,
            class_weight="balanced" if use_class_weight else None,
        )
    if kind == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=250, random_state=config.SEED, n_jobs=config.N_JOBS,
            class_weight="balanced" if use_class_weight else None,
            verbose=-1,
        )
    if kind == "xgb":
        # XGBoost uses scale_pos_weight rather than class_weight.
        spw = (n_neg / max(n_pos, 1)) if use_class_weight else 1.0
        return xgb.XGBClassifier(
            n_estimators=250, random_state=config.SEED, n_jobs=config.N_JOBS,
            eval_metric="logloss", scale_pos_weight=spw,
        )
    raise ValueError(f"unknown model kind {kind}")


def _apply_resampling(X: pd.DataFrame, y: pd.Series, imbalance: str):
    """Return possibly-resampled (X, y). Only applied to training data."""
    if imbalance in {"none", "class_weight"}:
        return X, y
    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        X_r, y_r = SMOTE(random_state=config.SEED).fit_resample(X, y)
        return X_r, y_r
    if imbalance == "smoteenn":
        from imblearn.combine import SMOTEENN
        X_r, y_r = SMOTEENN(random_state=config.SEED).fit_resample(X, y)
        return X_r, y_r
    raise ValueError(f"unknown imbalance strategy {imbalance}")


@dataclass
class TrainedBundle:
    """Artifacts of one (model, imbalance, feature_set, split) run."""
    model_name: str
    imbalance: str
    feature_set: str
    split_name: str
    label_col: str
    model: Any
    feature_cols: list[str]
    threshold_val: ThresholdChoice
    val_result: EvalResult
    test_result: EvalResult
    lopo_results: list[EvalResult] = field(default_factory=list)


def train_bundle(
    *,
    split: Split,
    model_kind: str,
    imbalance: str,
    feature_set: FeatureSet,
    label_col: str,
    threshold_objective: str = "f1",
    verbose: bool = False,
) -> TrainedBundle:
    X_tr, y_tr, cols = prepare_xy(split.train, label_col=label_col, feature_set=feature_set)
    X_va, y_va, _ = prepare_xy(split.val, label_col=label_col, feature_set=feature_set, feature_cols=cols)
    X_te, y_te, _ = prepare_xy(split.test, label_col=label_col, feature_set=feature_set, feature_cols=cols)

    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    model = _new_model(model_kind, imbalance=imbalance, n_pos=n_pos, n_neg=n_neg)

    X_tr_r, y_tr_r = _apply_resampling(X_tr, y_tr, imbalance)
    model.fit(X_tr_r, y_tr_r)

    # Tune threshold on validation, then evaluate on both val and test.
    p_va = model.predict_proba(X_va)[:, 1]
    thr = tune_threshold(y_va.to_numpy(), p_va, objective=threshold_objective)

    val_res = evaluate(
        model=model, X=X_va, y_true=y_va, threshold=thr.threshold,
        split_name=f"{split.name}/val", model_name=model_kind,
        feature_set=feature_set.value, label_col=label_col, verbose=verbose,
    )
    test_res = evaluate(
        model=model, X=X_te, y_true=y_te, threshold=thr.threshold,
        split_name=f"{split.name}/test", model_name=model_kind,
        feature_set=feature_set.value, label_col=label_col, verbose=verbose,
    )

    return TrainedBundle(
        model_name=model_kind,
        imbalance=imbalance,
        feature_set=feature_set.value,
        split_name=split.name,
        label_col=label_col,
        model=model,
        feature_cols=cols,
        threshold_val=thr,
        val_result=val_res,
        test_result=test_res,
    )


def sweep(
    *,
    split: Split,
    label_col: str,
    model_kinds: tuple[str, ...] = ("rf", "lgbm", "xgb"),
    imbalance_strategies: tuple[str, ...] = ("none", "class_weight", "smote"),
    feature_sets: tuple[FeatureSet, ...] = (FeatureSet.ALL, FeatureSet.NO_MATURITY, FeatureSet.CHANGE_ONLY),
    verbose: bool = False,
) -> list[TrainedBundle]:
    """Cartesian-product sweep. Returns one TrainedBundle per combination."""
    bundles: list[TrainedBundle] = []
    for fs in feature_sets:
        for imb in imbalance_strategies:
            for mk in model_kinds:
                b = train_bundle(
                    split=split, model_kind=mk, imbalance=imb,
                    feature_set=fs, label_col=label_col, verbose=verbose,
                )
                bundles.append(b)
    return bundles


def bundles_to_df(bundles: list[TrainedBundle]) -> pd.DataFrame:
    rows = []
    for b in bundles:
        for kind, r in [("val", b.val_result), ("test", b.test_result)]:
            d = r.as_dict()
            d["imbalance"] = b.imbalance
            d["phase"] = kind
            d["tuned_threshold"] = b.threshold_val.threshold
            d["threshold_objective"] = b.threshold_val.objective
            rows.append(d)
    return pd.DataFrame(rows)


def save_bundle(b: TrainedBundle, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fname = f"{b.split_name}__{b.model_name}__{b.imbalance}__{b.feature_set}.joblib"
    path = directory / fname
    joblib.dump({
        "model": b.model,
        "feature_cols": b.feature_cols,
        "threshold": b.threshold_val.threshold,
        "meta": {
            "model_name": b.model_name,
            "imbalance": b.imbalance,
            "feature_set": b.feature_set,
            "split_name": b.split_name,
            "label_col": b.label_col,
        },
    }, path)
    return path
