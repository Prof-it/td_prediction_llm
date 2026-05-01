"""
Patch features_with_llm_labels.csv with:
  - Fresh v2.3 LLM labels for the 100 gold commits (from human_review_sheet_llm.csv)
  - label_human column: consolidated GT where humans agreed (from consolidated_gt.csv)

Reads:
  artifacts/results/human_review_sheet_llm.csv
  artifacts/results/consolidated_gt.csv
Writes:
  data/features_with_llm_labels.csv  (in-place update)

Run from repo root:
    python scripts/patch_labels.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

SHEET_LLM       = Path("artifacts/results/human_review_sheet_llm.csv")
CONSOLIDATED_GT = Path("artifacts/results/consolidated_gt.csv")
FEATURES_CSV    = Path("data/features_with_llm_labels.csv")


def norm(val: str) -> int | None:
    v = str(val).strip().lower()
    if v in ("1", "yes", "true"):  return 1
    if v in ("0", "no", "false"): return 0
    return None


def main():
    if not SHEET_LLM.exists():
        sys.exit(f"Missing {SHEET_LLM} — run llm_autoreview.py first")
    if not CONSOLIDATED_GT.exists():
        sys.exit(f"Missing {CONSOLIDATED_GT} — run kappa_analysis.py first")

    # Load fresh LLM labels for gold commits
    llm_labels: dict[str, int] = {}
    with SHEET_LLM.open() as f:
        for row in csv.DictReader(f):
            v = norm(row.get("label_llm", ""))
            if v is not None:
                llm_labels[row["commit_uid"]] = v

    # Load consolidated GT (only where humans agreed)
    gt_labels: dict[str, int] = {}
    with CONSOLIDATED_GT.open() as f:
        for row in csv.DictReader(f):
            v = norm(row.get("label_gt", ""))
            if v is not None:
                gt_labels[row["commit_uid"]] = v

    df = pd.read_csv(FEATURES_CSV)
    n_before = df["label_llm"].sum()

    df["label_llm"] = df.apply(
        lambda row: llm_labels.get(row["commit_uid"], row["label_llm"]), axis=1
    )

    if "label_human" not in df.columns:
        df["label_human"] = pd.NA
    df["label_human"] = df.apply(
        lambda row: gt_labels.get(row["commit_uid"], row["label_human"]), axis=1
    )

    df.to_csv(FEATURES_CSV, index=False)

    n_after = df["label_llm"].sum()
    print(f"Patched {len(llm_labels)} gold commits with v2.3 LLM labels")
    print(f"label_llm positive rate: {n_before/len(df):.3f} → {n_after/len(df):.3f}")
    print(f"label_human (GT) set for {df['label_human'].notna().sum()} commits")
    print(f"Written → {FEATURES_CSV}")


if __name__ == "__main__":
    main()
