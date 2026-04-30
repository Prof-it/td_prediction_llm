"""
Compute inter-rater agreement (Cohen's κ) across:
  - Human A (sheet_a) vs Human B (sheet_b)
  - LLM (sheet_llm) vs Human A
  - LLM (sheet_llm) vs Human B
  - LLM vs consolidated GT (A == B)

Also identifies disagreement slices for targeted review.

Run from repo root:
    python scripts/kappa_analysis.py
"""
import csv
import sys
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
except ImportError:
    sys.exit("Missing sklearn: pip install scikit-learn")

SHEET_A   = Path("artifacts/results/human_review_sheet_a_20260424_021700.csv")
SHEET_B   = Path("artifacts/results/human_review_sheet_b_20260428_183349.csv")
SHEET_LLM = Path("artifacts/results/human_review_sheet_llm.csv")
OUT_CSV   = Path("artifacts/results/disagreement_slice.csv")


def norm(val: str) -> int | None:
    """Normalize label to 0/1. Returns None if missing/error."""
    v = str(val).strip().lower()
    if v in ("1", "yes", "true"):
        return 1
    if v in ("0", "no", "false"):
        return 0
    return None


def load_labels(path: Path, label_col: str, delimiter: str = ",") -> dict[str, int]:
    """Return {commit_uid: 0|1} from a CSV file."""
    result = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            uid = row["commit_uid"]
            val = norm(row.get(label_col, ""))
            if val is not None:
                result[uid] = val
    return result


def aligned(a: dict, b: dict) -> tuple[list, list, list]:
    """Return (uids, a_labels, b_labels) for rows present in both."""
    common = sorted(set(a) & set(b))
    return common, [a[u] for u in common], [b[u] for u in common]


def kappa_report(name: str, uids: list, y1: list, y2: list, label1="rater1", label2="rater2"):
    k = cohen_kappa_score(y1, y2)
    cm = confusion_matrix(y1, y2)
    agree = sum(a == b for a, b in zip(y1, y2))
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  n={len(y1)}  agreement={agree/len(y1):.1%}  Cohen's κ={k:.4f}")
    print(f"\n  Confusion matrix ({label1} rows, {label2} cols):")
    print(f"           pred_0  pred_1")
    for i, row in enumerate(cm):
        print(f"  true_{i}   {row[0]:5d}   {row[1]:5d}")
    return k


def save_disagreements(a: dict, b: dict, llm: dict, all_rows: dict):
    """Write a CSV of commits where at least one pair disagrees."""
    rows = []
    for uid in sorted(set(a) & set(b) & set(llm)):
        la, lb, ll = a[uid], b[uid], llm[uid]
        if not (la == lb == ll):
            meta = all_rows.get(uid, {})
            rows.append({
                "commit_uid": uid,
                "label_human_a": la,
                "label_human_b": lb,
                "label_llm": ll,
                "a_b_agree": int(la == lb),
                "llm_agrees_a": int(ll == la),
                "llm_agrees_b": int(ll == lb),
                "rationale_human_a": meta.get("rationale_human_a", ""),
                "rationale_human_b": meta.get("rationale_human_b", ""),
                "rationale_llm": meta.get("rationale_llm", ""),
                "commit_url": meta.get("commit_url", ""),
            })
    with OUT_CSV.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  Disagreement slice: {len(rows)} commits → {OUT_CSV}")
    return rows


def main():
    print("Loading labels...")
    human_a  = load_labels(SHEET_A,   "label_human", delimiter=";")
    human_b  = load_labels(SHEET_B,   "label_human")
    llm      = load_labels(SHEET_LLM, "label_llm")

    print(f"  Human A:  {len(human_a)} labels")
    print(f"  Human B:  {len(human_b)} labels")
    print(f"  LLM:      {len(llm)} labels")

    # --- Human A vs Human B ---
    uids_ab, ya, yb = aligned(human_a, human_b)
    kappa_report("Human A vs Human B", uids_ab, ya, yb, "human_a", "human_b")

    # --- Consolidated GT (A == B) ---
    gt = {u: human_a[u] for u in uids_ab if human_a[u] == human_b[u]}
    print(f"\n  Consolidated GT: {len(gt)}/{len(uids_ab)} commits where A==B")

    # --- LLM vs Human A ---
    uids_la, yla, ylla = aligned(llm, human_a)
    kappa_report("LLM vs Human A", uids_la, yla, ylla, "llm", "human_a")

    # --- LLM vs Human B ---
    uids_lb, ylb, yllb = aligned(llm, human_b)
    kappa_report("LLM vs Human B", uids_lb, ylb, yllb, "llm", "human_b")

    # --- LLM vs Consolidated GT ---
    uids_gt, ygt, yllgt = aligned(llm, gt)
    kappa_report("LLM vs Consolidated GT", uids_gt, ygt, yllgt, "llm", "gt")

    # --- Collect metadata for disagreement slice ---
    meta: dict[str, dict] = {}
    with SHEET_A.open() as f:
        for row in csv.DictReader(f, delimiter=';'):
            meta[row["commit_uid"]] = {
                "commit_url": row.get("commit_url", ""),
                "rationale_human_a": row.get("rationale_human", ""),
            }
    with SHEET_B.open() as f:
        for row in csv.DictReader(f):
            uid = row["commit_uid"]
            if uid in meta:
                meta[uid]["rationale_human_b"] = row.get("rationale_human", "")
    with SHEET_LLM.open() as f:
        for row in csv.DictReader(f):
            uid = row["commit_uid"]
            if uid in meta:
                meta[uid]["rationale_llm"] = row.get("rationale_llm", "")

    save_disagreements(human_a, human_b, llm, meta)

    print("\nSummary:")
    print(f"  Inter-human κ tells you labeling consistency.")
    print(f"  LLM vs GT κ tells you how much you can trust LLM pre-labeling.")
    print(f"  Disagreement slice → investigate + use for fine-tuning signal.")


if __name__ == "__main__":
    main()
