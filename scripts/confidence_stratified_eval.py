"""
Stratified evaluation: does the ML model perform better on commits where
the LLM judge was confident vs uncertain?

This addresses the "targeted retraining on disagreement + high-uncertainty
slices" workflow item — even without retraining, we can quantify whether
LLM confidence is a meaningful signal of label quality (i.e., whether the
model performs measurably worse on the low-confidence slice). If yes, the
low-confidence slice is a strong candidate for additional human review.

Outputs:
  artifacts/results/confidence_stratified_metrics.csv

Run from repo root:
    python scripts/confidence_stratified_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_prediction import config
from td_prediction.data.splits import FeatureSet, Split, load_split, prepare_xy
from td_prediction.models import trainer

LABEL_COL  = "label_consolidated"
PARSED_CSV = Path("artifacts/results/parsed_labels_v2_rubric_json.csv")
FEATURES_CSV = Path("data/features_with_llm_labels.csv")
OUT = Path("artifacts/results/confidence_stratified_metrics.csv")


def _refresh_labels(split, labels_df):
    cols = [c for c in ("commit_uid", "label_llm", "label_satd", "label_consolidated", "label_human")
            if c in labels_df.columns]
    fresh = labels_df[cols]
    def _merge(df):
        df = df.drop(columns=[c for c in cols if c != "commit_uid" and c in df.columns])
        return df.merge(fresh, on="commit_uid", how="left")
    return Split(train=_merge(split.train), val=_merge(split.val),
                 test=_merge(split.test), name=split.name)


def main():
    print("Loading...")
    labels_df = pd.read_csv(FEATURES_CSV)
    parsed    = pd.read_csv(PARSED_CSV)[["commit_uid", "confidence"]]
    split = _refresh_labels(load_split(config.PATHS.splits, "time"), labels_df)

    test_with_conf = split.test.merge(parsed, on="commit_uid", how="left")
    n_missing = test_with_conf["confidence"].isna().sum()
    if n_missing:
        print(f"  {n_missing} test commits have no LLM confidence (defaulting to median)")
        test_with_conf["confidence"] = test_with_conf["confidence"].fillna(
            test_with_conf["confidence"].median()
        )

    # Train once with headline configuration
    print("Training RF / all / none on label_consolidated...")
    bundle = trainer.train_bundle(
        split=split, model_kind="rf", imbalance="none",
        feature_set=FeatureSet.ALL, label_col=LABEL_COL,
        threshold_objective="f1",
    )

    X_te, y_te, _, _ = prepare_xy(
        test_with_conf, label_col=LABEL_COL,
        feature_set=FeatureSet.ALL, feature_cols=bundle.feature_cols,
    )
    y_score = bundle.model.predict_proba(X_te)[:, 1]
    y_pred  = (y_score >= bundle.threshold_val.threshold).astype(int)
    y_true  = y_te.to_numpy()
    conf    = test_with_conf["confidence"].to_numpy()

    # Buckets — using fixed thresholds rather than quantiles so the
    # "uncertainty" interpretation is stable across runs.
    buckets = [
        ("low (<0.85)",      conf < 0.85),
        ("mid (0.85-0.95)", (conf >= 0.85) & (conf < 0.95)),
        ("high (>=0.95)",    conf >= 0.95),
        ("ALL",              np.ones_like(conf, dtype=bool)),
    ]

    rows = []
    print()
    print(f"{'bucket':<18} {'n':>6} {'pos%':>7} {'F1':>6} {'P':>6} {'R':>6} "
          f"{'AUC':>6} {'PR-AUC':>7} {'MCC':>6}")
    print("-" * 78)
    for name, mask in buckets:
        if mask.sum() == 0:
            continue
        yt = y_true[mask]; yp = y_pred[mask]; ys = y_score[mask]
        if len(set(yt)) > 1:
            auc = roc_auc_score(yt, ys)
            pr  = average_precision_score(yt, ys)
        else:
            auc = pr = float("nan")
        m = {
            "bucket": name,
            "n": int(mask.sum()),
            "pos_rate": float(yt.mean()),
            "f1":  f1_score(yt, yp, zero_division=0),
            "p":   precision_score(yt, yp, zero_division=0),
            "r":   recall_score(yt, yp, zero_division=0),
            "roc_auc":     auc,
            "pr_auc_debt": pr,
            "mcc":         matthews_corrcoef(yt, yp) if len(set(yt)) > 1 else float("nan"),
            "tp": int((yt & yp).sum()),
            "fp": int(((1 - yt) & yp).sum()),
            "fn": int((yt & (1 - yp)).sum()),
            "tn": int(((1 - yt) & (1 - yp)).sum()),
        }
        rows.append(m)
        print(f"{name:<18} {m['n']:>6} {m['pos_rate']*100:>6.2f}% "
              f"{m['f1']:>6.3f} {m['p']:>6.3f} {m['r']:>6.3f} "
              f"{m['roc_auc']:>6.3f} {m['pr_auc_debt']:>7.3f} {m['mcc']:>6.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  → {OUT}")

    # Headline interpretation
    if len(rows) >= 3:
        low  = next((r for r in rows if r["bucket"].startswith("low")),  None)
        high = next((r for r in rows if r["bucket"].startswith("high")), None)
        if low and high and not np.isnan(low["roc_auc"]) and not np.isnan(high["roc_auc"]):
            print(f"\n  AUC gap (high - low confidence): "
                  f"{high['roc_auc'] - low['roc_auc']:+.3f}")
            print(f"  F1  gap (high - low confidence): "
                  f"{high['f1']  - low['f1']:+.3f}")


if __name__ == "__main__":
    main()
