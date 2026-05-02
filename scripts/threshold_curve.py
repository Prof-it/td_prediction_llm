"""
Generate the precision-recall trade-off curve for the headline model
(RF / all features / no rebalancing) on the time-split test set, and report
several useful operating points.

Why this matters:
  The current headline F1 = 0.41 at the F1-optimal threshold (≈0.28).
  But F1 is not the only operating point that makes sense — a deployment
  use case "flag commits for human review" might want higher precision
  (don't waste reviewer time), while a screening use case "catch all TD"
  wants higher recall.

Outputs:
  artifacts/results/pr_curve.csv           — full precision-recall curve data
  artifacts/results/operating_points.csv   — canonical operating points

  Operating points reported:
    - default_0.5            (default threshold)
    - f1_optimal             (current headline)
    - max_f0.5_optimal       (precision-weighted F-score)
    - max_f2_optimal         (recall-weighted F-score)
    - high_precision_p>=0.5  (lowest threshold giving precision ≥ 0.5)
    - high_recall_r>=0.8     (highest threshold giving recall ≥ 0.8)

Run from repo root:
    python scripts/threshold_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, fbeta_score,
    confusion_matrix, matthews_corrcoef,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_prediction import config
from td_prediction.data.splits import FeatureSet, Split, load_split, prepare_xy
from td_prediction.models import trainer

LABEL_COL = "label_consolidated"
FEATURES_CSV = Path("data/features_with_llm_labels.csv")
OUT_CURVE = Path("artifacts/results/pr_curve.csv")
OUT_OPS   = Path("artifacts/results/operating_points.csv")


def _refresh_labels(split: Split, labels_df: pd.DataFrame) -> Split:
    cols = [c for c in ("commit_uid", "label_llm", "label_satd", "label_consolidated", "label_human")
            if c in labels_df.columns]
    fresh = labels_df[cols]

    def _merge(df):
        df = df.drop(columns=[c for c in cols if c != "commit_uid" and c in df.columns])
        return df.merge(fresh, on="commit_uid", how="left")

    return Split(train=_merge(split.train), val=_merge(split.val),
                 test=_merge(split.test), name=split.name)


def _operating_point(name: str, y_true, y_score, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "name": name,
        "threshold": float(threshold),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "f0.5":      float(fbeta_score(y_true, pred, beta=0.5, zero_division=0)),
        "f2":        float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    labels_df = pd.read_csv(FEATURES_CSV)
    split = _refresh_labels(load_split(config.PATHS.splits, "time"), labels_df)

    # Train once with the headline configuration (uses config.SEED == 42)
    print("Training RF / all / none on label_consolidated...")
    bundle = trainer.train_bundle(
        split=split, model_kind="rf", imbalance="none",
        feature_set=FeatureSet.ALL, label_col=LABEL_COL,
        threshold_objective="f1",
    )

    X_te, y_te, _, _ = prepare_xy(
        split.test, label_col=LABEL_COL,
        feature_set=FeatureSet.ALL, feature_cols=bundle.feature_cols,
    )
    y_score = bundle.model.predict_proba(X_te)[:, 1]
    y_true = y_te.to_numpy()

    n_pos, n = int(y_true.sum()), len(y_true)
    print(f"Test set: {n:,} commits, {n_pos} positive ({n_pos/n*100:.2f}%)")

    # Full PR curve
    p_arr, r_arr, t_arr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one extra precision/recall point for the
    # corner — drop it so each row corresponds to a real threshold.
    f1_arr = np.where(p_arr + r_arr > 0, 2 * p_arr * r_arr / (p_arr + r_arr), 0.0)

    curve = pd.DataFrame({
        "threshold": np.concatenate([t_arr, [1.0]]),
        "precision": p_arr,
        "recall":    r_arr,
        "f1":        f1_arr,
    })
    curve.to_csv(OUT_CURVE, index=False)
    print(f"PR curve ({len(curve)} points) → {OUT_CURVE}")

    # Operating points
    ops = []

    # Default 0.5 threshold
    ops.append(_operating_point("default_0.5", y_true, y_score, 0.5))

    # F1-optimal (current headline) — same as bundle's tuned threshold
    ops.append(_operating_point("f1_optimal", y_true, y_score, bundle.threshold_val.threshold))

    # F0.5-optimal (precision-weighted): scan all thresholds, pick max F0.5
    f05_scores = []
    f2_scores  = []
    for t in t_arr:
        pred = (y_score >= t).astype(int)
        f05_scores.append(fbeta_score(y_true, pred, beta=0.5, zero_division=0))
        f2_scores.append(fbeta_score(y_true, pred, beta=2.0, zero_division=0))
    best_f05_t = float(t_arr[int(np.argmax(f05_scores))])
    best_f2_t  = float(t_arr[int(np.argmax(f2_scores))])
    ops.append(_operating_point("f0.5_optimal", y_true, y_score, best_f05_t))
    ops.append(_operating_point("f2_optimal",   y_true, y_score, best_f2_t))

    # Lowest threshold s.t. precision >= 0.5 (highest recall at that precision)
    mask = p_arr[:-1] >= 0.5
    if mask.any():
        # t_arr is sorted ascending; pick the smallest t where precision ≥ 0.5
        valid_thresholds = t_arr[mask]
        ops.append(_operating_point("high_precision_p>=0.5",
                                     y_true, y_score, float(valid_thresholds.min())))
    else:
        print("  (no threshold achieves precision ≥ 0.5)")

    # Highest threshold s.t. recall >= 0.8
    mask_r = r_arr[:-1] >= 0.8
    if mask_r.any():
        valid_thresholds = t_arr[mask_r]
        ops.append(_operating_point("high_recall_r>=0.8",
                                     y_true, y_score, float(valid_thresholds.max())))

    ops_df = pd.DataFrame(ops)
    ops_df.to_csv(OUT_OPS, index=False)

    print(f"\nOperating points → {OUT_OPS}\n")
    print(ops_df[["name", "threshold", "precision", "recall", "f1", "f0.5", "f2", "mcc",
                  "tp", "fp", "fn", "tn"]].to_string(index=False))


if __name__ == "__main__":
    main()
