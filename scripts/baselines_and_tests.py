"""
Evaluation extras for paper-quality reporting:

  1. SATD-regex baseline — predict label_consolidated using only label_satd.
     Reports same metrics as the ML model on the same time-split test set.
  2. Multi-seed runs — re-train the headline model with N seeds, report
     mean ± std for F1 / AUC / MCC.
  3. McNemar's test — pairwise comparison of best ML model vs SATD baseline
     on the test set (per-commit predictions).

Reads:
  data/features_with_llm_labels.csv
  splits/time/{train,val,test}.csv

Writes:
  artifacts/results/baseline_metrics.csv      — SATD baseline metrics
  artifacts/results/multiseed_metrics.csv     — N-seed mean ± std
  artifacts/results/mcnemar_test.txt          — McNemar's test result

Run from repo root:
    python scripts/baselines_and_tests.py
    python scripts/baselines_and_tests.py --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_prediction import config
from td_prediction.data.splits import FeatureSet, Split, load_split, prepare_xy
from td_prediction.models import trainer

LABEL_COL = "label_consolidated"
FEATURES_CSV = Path("data/features_with_llm_labels.csv")

OUT_BASELINE = Path("artifacts/results/baseline_metrics.csv")
OUT_MULTI    = Path("artifacts/results/multiseed_metrics.csv")
OUT_MCNEMAR  = Path("artifacts/results/mcnemar_test.txt")


# ── shared utilities ──────────────────────────────────────────────────────────

def _refresh_labels(split: Split, labels_df: pd.DataFrame) -> Split:
    """Merge fresh labels from features CSV into cached split files."""
    cols = [c for c in ("commit_uid", "label_llm", "label_satd", "label_consolidated", "label_human")
            if c in labels_df.columns]
    fresh = labels_df[cols]

    def _merge(df):
        df = df.drop(columns=[c for c in cols if c != "commit_uid" and c in df.columns])
        return df.merge(fresh, on="commit_uid", how="left")

    return Split(
        train=_merge(split.train), val=_merge(split.val),
        test=_merge(split.test), name=split.name,
    )


def _metrics(y_true, y_pred, y_score=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "f1_debt":      f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "p_debt":       precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "r_debt":       recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        out["roc_auc"]     = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
        out["pr_auc_debt"] = average_precision_score(y_true, y_score)
    return out


# ── 1. SATD baseline ──────────────────────────────────────────────────────────

def satd_baseline(split: Split) -> dict:
    """
    The naive baseline: 'flag the commit if label_satd == 1, else don't'.
    No training, no features, just the regex output as the prediction.
    """
    print("\n=== SATD-REGEX BASELINE ===")
    rows = []
    for phase, part in [("val", split.val), ("test", split.test)]:
        y_true = part[LABEL_COL].astype(int).to_numpy()
        y_pred = part["label_satd"].astype(int).to_numpy()
        m = _metrics(y_true, y_pred, y_score=y_pred.astype(float))
        m["phase"] = phase
        m["model"] = "satd_regex"
        m["n"] = len(y_true)
        rows.append(m)
        print(f"  {phase}: F1={m['f1_debt']:.3f} P={m['p_debt']:.3f} R={m['r_debt']:.3f} "
              f"MCC={m['mcc']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})")

    pd.DataFrame(rows).to_csv(OUT_BASELINE, index=False)
    print(f"  → {OUT_BASELINE}")
    return rows[-1]  # test row


# ── 2. Multi-seed runs ────────────────────────────────────────────────────────

def multiseed(split: Split, n_seeds: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Re-train RF / all features / no rebalancing with `n_seeds` seeds.
    Returns: (metrics DataFrame, y_test, ML predictions from seed 0 for McNemar).
    """
    print(f"\n=== MULTI-SEED RUNS (n={n_seeds}, model=rf/all/none) ===")
    seeds = list(range(n_seeds))
    rows = []
    test_preds = None
    y_test = None

    for s in seeds:
        original_seed = config.SEED
        config.SEED = s   # the trainer reads SEED at model-construction time
        try:
            bundle = trainer.train_bundle(
                split=split, model_kind="rf", imbalance="none",
                feature_set=FeatureSet.ALL, label_col=LABEL_COL,
                threshold_objective="f1",
            )
        finally:
            config.SEED = original_seed

        for phase, res in [("val", bundle.val_result), ("test", bundle.test_result)]:
            rows.append({
                "seed": s, "phase": phase,
                "f1_debt":     res.f1_debt,
                "roc_auc":     res.roc_auc,
                "pr_auc_debt": res.pr_auc_debt,
                "p_debt":      res.p_debt,
                "r_debt":      res.r_debt,
                "mcc":         res.mcc,
            })
        if s == 0:
            X_te, y_te, _, _ = prepare_xy(
                split.test, label_col=LABEL_COL,
                feature_set=FeatureSet.ALL, feature_cols=bundle.feature_cols,
            )
            scores = bundle.model.predict_proba(X_te)[:, 1]
            test_preds = (scores >= bundle.threshold_val.threshold).astype(int)
            y_test = y_te.to_numpy()
            print(f"  seed=0 threshold={bundle.threshold_val.threshold:.3f}")
        print(f"  seed={s}: val F1={bundle.val_result.f1_debt:.3f}, "
              f"test F1={bundle.test_result.f1_debt:.3f}")

    df = pd.DataFrame(rows)
    print()
    print("  Aggregate (test):")
    test_only = df[df["phase"] == "test"]
    for col in ["f1_debt", "roc_auc", "pr_auc_debt", "p_debt", "r_debt", "mcc"]:
        mean = test_only[col].mean(); std = test_only[col].std()
        print(f"    {col:<14} {mean:.4f} ± {std:.4f}")

    df.to_csv(OUT_MULTI, index=False)
    print(f"  → {OUT_MULTI}")
    return df, y_test, test_preds


# ── 3. McNemar's test ─────────────────────────────────────────────────────────

def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
            name_a: str, name_b: str) -> str:
    """
    McNemar's test on paired predictions (binary).
    Tests whether two classifiers disagree systematically on test commits.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    # b00=both wrong, b11=both right (off-diagonal terms drive the test)
    b01 = int(((~correct_a) & correct_b).sum())   # A wrong, B right
    b10 = int((correct_a & (~correct_b)).sum())   # A right, B wrong

    from statsmodels.stats.contingency_tables import mcnemar as mcn
    result = mcn(np.array([[0, b01], [b10, 0]]), exact=False, correction=True)
    chi2 = float(result.statistic)
    p = float(result.pvalue)

    lines = [
        f"McNemar's test: {name_a} vs {name_b}",
        f"  n_test                                    = {len(y_true)}",
        f"  {name_a:<25} correct, other wrong: {b10}",
        f"  {name_b:<25} correct, other wrong: {b01}",
        f"  chi²                                      = {chi2:.4f}",
        f"  p-value                                   = {p:.4g}",
        f"  significant at α=0.05?                    = {'YES' if p < 0.05 else 'NO'}",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds for multi-seed runs.")
    args = p.parse_args()

    labels_df = pd.read_csv(FEATURES_CSV)
    split = _refresh_labels(load_split(config.PATHS.splits, "time"), labels_df)

    # 1. SATD baseline
    satd_test_metrics = satd_baseline(split)
    y_test_arr = split.test[LABEL_COL].astype(int).to_numpy()
    satd_pred = split.test["label_satd"].astype(int).to_numpy()

    # 2. Multi-seed
    multi_df, y_test_ml, ml_pred = multiseed(split, n_seeds=args.seeds)
    assert (y_test_arr == y_test_ml).all(), "test split mismatch — should never happen"

    # 3. McNemar's test
    print("\n=== McNEMAR'S TEST ===")
    out = mcnemar(y_test_arr, ml_pred, satd_pred, "rf_all_none", "satd_regex")
    print(out)
    OUT_MCNEMAR.write_text(out + "\n")
    print(f"  → {OUT_MCNEMAR}")


if __name__ == "__main__":
    main()
