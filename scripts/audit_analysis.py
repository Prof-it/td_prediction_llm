"""
Audit-quality analysis of the 100-commit human-reviewed gold set:

  1. Stratification check: is the gold set representative of the full corpus
     by (a) repo, (b) commit size, (c) label_satd / TD-likelihood?
  2. Confidence slice: identify high-uncertainty LLM labels for targeted
     additional human review (high-uncertainty slices for targeted retraining).

Reads:
  data/features_with_llm_labels.csv
  artifacts/results/consolidated_gt.csv
  artifacts/results/parsed_labels_v2_rubric_json.csv  (for confidence)
Writes:
  artifacts/results/audit_stratification.csv
  artifacts/results/audit_low_confidence.csv

Run from repo root:
    python scripts/audit_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FEATURES = Path("data/features_with_llm_labels.csv")
CGT      = Path("artifacts/results/consolidated_gt.csv")
PARSED   = Path("artifacts/results/parsed_labels_v2_rubric_json.csv")
OUT_STRAT = Path("artifacts/results/audit_stratification.csv")
OUT_LOWC  = Path("artifacts/results/audit_low_confidence.csv")


def stratification_check():
    print("\n" + "=" * 70)
    print("  STRATIFICATION CHECK — is the gold set representative?")
    print("=" * 70)

    df = pd.read_csv(FEATURES)
    gold_uids = set(pd.read_csv(CGT)["commit_uid"])
    df["is_gold"] = df["commit_uid"].isin(gold_uids)

    # 1. Repo distribution
    rows = []
    print("\n  By repository:")
    print(f"  {'repo':<10} {'corpus':>10} {'gold':>8} {'expected':>10} {'observed':>10}")
    for repo, sub in df.groupby("repo_id"):
        corpus_share = len(sub) / len(df)
        gold_n = sub["is_gold"].sum()
        expected = corpus_share * 100
        observed = gold_n
        rows.append({"dimension": "repo", "stratum": repo,
                     "corpus_n": len(sub), "gold_n": int(gold_n),
                     "corpus_share": corpus_share,
                     "expected_in_gold": expected, "observed_in_gold": observed,
                     "ratio_obs_exp": observed / expected if expected else float("nan")})
        print(f"  {repo:<10} {len(sub):>10,} {int(gold_n):>8} {expected:>10.1f} {observed:>10}")

    # 2. Size buckets (lines_added quantiles on full corpus)
    qs = df["lines_added"].quantile([0.25, 0.5, 0.75]).values
    bins = [-1, qs[0], qs[1], qs[2], df["lines_added"].max()]
    labels = ["XS (≤Q1)", "S (Q1-Q2)", "M (Q2-Q3)", "L (>Q3)"]
    df["size_bucket"] = pd.cut(df["lines_added"], bins=bins, labels=labels)
    print("\n  By size bucket (lines_added quartiles):")
    print(f"  {'bucket':<14} {'corpus':>10} {'gold':>8} {'expected':>10} {'observed':>10}")
    for b in labels:
        sub = df[df["size_bucket"] == b]
        corpus_share = len(sub) / len(df)
        gold_n = sub["is_gold"].sum()
        expected = corpus_share * 100
        observed = gold_n
        rows.append({"dimension": "size", "stratum": b,
                     "corpus_n": len(sub), "gold_n": int(gold_n),
                     "corpus_share": corpus_share,
                     "expected_in_gold": expected, "observed_in_gold": observed,
                     "ratio_obs_exp": observed / expected if expected else float("nan")})
        print(f"  {b:<14} {len(sub):>10,} {int(gold_n):>8} {expected:>10.1f} {observed:>10}")

    # 3. label_satd (proxy for TD likelihood)
    print("\n  By label_satd (regex baseline — TD-likelihood proxy):")
    print(f"  {'satd':<6} {'corpus':>10} {'gold':>8} {'expected':>10} {'observed':>10}")
    for v in [0, 1]:
        sub = df[df["label_satd"] == v]
        corpus_share = len(sub) / len(df)
        gold_n = sub["is_gold"].sum()
        expected = corpus_share * 100
        observed = gold_n
        rows.append({"dimension": "label_satd", "stratum": str(v),
                     "corpus_n": len(sub), "gold_n": int(gold_n),
                     "corpus_share": corpus_share,
                     "expected_in_gold": expected, "observed_in_gold": observed,
                     "ratio_obs_exp": observed / expected if expected else float("nan")})
        print(f"  {v:<6} {len(sub):>10,} {int(gold_n):>8} {expected:>10.1f} {observed:>10}")

    # Verdict
    print("\n  Verdict:")
    repo_ratios = [r["ratio_obs_exp"] for r in rows if r["dimension"] == "repo"]
    size_ratios = [r["ratio_obs_exp"] for r in rows if r["dimension"] == "size"]
    def assess(name, ratios):
        worst = max(abs(r - 1) for r in ratios)
        if worst < 0.2: print(f"    {name}: well-balanced (max deviation {worst*100:.0f}%)")
        elif worst < 0.5: print(f"    {name}: moderately balanced (max deviation {worst*100:.0f}%)")
        else: print(f"    {name}: skewed (max deviation {worst*100:.0f}%)")
    assess("by repo", repo_ratios)
    assess("by size", size_ratios)
    print(f"    by label_satd: 50/50 stratified split — INTENTIONAL.")
    print(f"      Pure random sampling at 3.1% positive rate would give")
    print(f"      ~3 SATD-positive cases per 100, making κ unreliable.")
    print(f"      Stratifying on label_satd ensures both classes are well")
    print(f"      represented for the kappa comparison.")

    pd.DataFrame(rows).to_csv(OUT_STRAT, index=False)
    print(f"\n  Saved → {OUT_STRAT}")


def confidence_slice():
    print("\n" + "=" * 70)
    print("  CONFIDENCE SLICE — high-uncertainty LLM labels")
    print("=" * 70)

    parsed = pd.read_csv(PARSED)
    if "confidence" not in parsed.columns:
        print("  No confidence column in parsed labels — skipping.")
        return

    print(f"\n  Confidence distribution across {len(parsed):,} labels:")
    print(parsed["confidence"].describe().round(3).to_string())

    # Distribution buckets
    print("\n  By bucket:")
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]
    for lo, hi in buckets:
        sub = parsed[(parsed["confidence"] >= lo) & (parsed["confidence"] < hi)]
        share = len(sub) / len(parsed) * 100
        print(f"  [{lo:.2f}, {hi:.2f})  {len(sub):>7,}  ({share:5.1f}%)")

    # Low-confidence slice for review
    threshold = 0.85
    low = parsed[parsed["confidence"] < threshold].copy()
    print(f"\n  Low-confidence (<{threshold}): {len(low):,} commits "
          f"({len(low)/len(parsed)*100:.1f}% of corpus)")
    print(f"  Of those, label distribution:")
    print(low["label_int"].value_counts().to_dict())

    # Cross-reference with consolidated_gt to see how LLM does on humans-validated commits
    cgt = pd.read_csv(CGT)
    low_in_gold = low.merge(cgt, on="commit_uid", how="inner", suffixes=("_p", "_gt"))
    if len(low_in_gold):
        print(f"\n  Of low-confidence commits, {len(low_in_gold)} are in the gold set.")

    # Save the low-confidence slice with relevant context
    df = pd.read_csv(FEATURES)
    out = low.merge(df[["commit_uid", "repo_id", "commit_hash", "lines_added", "lines_deleted",
                         "files_changed", "label_satd", "label_consolidated"]],
                     on="commit_uid", how="left")
    out = out.sort_values("confidence")
    out.to_csv(OUT_LOWC, index=False)
    print(f"\n  Saved → {OUT_LOWC}")
    print(f"\n  These are good candidates for additional human review")
    print(f"  (high-uncertainty slices for targeted retraining).")


def main():
    stratification_check()
    confidence_slice()


if __name__ == "__main__":
    main()
