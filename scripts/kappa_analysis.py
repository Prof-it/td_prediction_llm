"""
Compute inter-rater agreement (Cohen's κ) and save outputs:
  - Human A vs Human B
  - LLM vs Human A / B
  - LLM vs consolidated GT (where A == B)

Outputs:
  artifacts/results/consolidated_gt.csv   — all 100 commits with all labels + GT
  artifacts/results/disagreement_slice.csv — commits where any pair disagrees

Run from repo root:
    python scripts/kappa_analysis.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError:
    sys.exit("Missing sklearn: pip install scikit-learn")

SHEET_A          = Path("artifacts/results/human_review_sheet_a_20260424_021700.csv")
SHEET_B          = Path("artifacts/results/human_review_sheet_b_20260428_183349.csv")
SHEET_LLM        = Path("artifacts/results/human_review_sheet_llm.csv")
CONSOLIDATED_GT  = Path("artifacts/results/consolidated_gt.csv")
DISAGREEMENT     = Path("artifacts/results/disagreement_slice.csv")
KAPPA_CI         = Path("artifacts/results/kappa_ci.json")
N_BOOTSTRAP      = 10_000
BOOTSTRAP_SEED   = 42


def norm(val: str) -> int | None:
    v = str(val).strip().lower()
    if v in ("1", "yes", "true"):  return 1
    if v in ("0", "no", "false"): return 0
    return None


def load_labels(path: Path, label_col: str, delimiter: str = ",") -> dict[str, int]:
    result = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            v = norm(row.get(label_col, ""))
            if v is not None:
                result[row["commit_uid"]] = v
    return result


def load_rows(path: Path, delimiter: str = ",") -> dict[str, dict]:
    with path.open() as f:
        return {row["commit_uid"]: row for row in csv.DictReader(f, delimiter=delimiter)}


def aligned(a: dict, b: dict) -> tuple[list, list, list]:
    common = sorted(set(a) & set(b))
    return common, [a[u] for u in common], [b[u] for u in common]


def kappa_report(name: str, y1: list, y2: list) -> float:
    k = cohen_kappa_score(y1, y2)
    cm = confusion_matrix(y1, y2)
    agree = sum(a == b for a, b in zip(y1, y2))
    quality = "good" if k >= 0.6 else "moderate" if k >= 0.4 else "poor"
    print(f"\n  {'='*50}")
    print(f"  {name}")
    print(f"  n={len(y1)}  agreement={agree/len(y1):.1%}  κ={k:.4f}  ({quality})")
    print(f"  confusion matrix (rows=rater1, cols=rater2): {cm.tolist()}")
    return k


def save_consolidated_gt(rows_a, rows_b, rows_llm,
                          human_a: dict, human_b: dict, llm: dict):
    all_uids = sorted(set(rows_a) | set(rows_b))
    out, disagreements = [], []

    for uid in all_uids:
        la = human_a.get(uid)
        lb = human_b.get(uid)
        ll = llm.get(uid)
        agree = la is not None and lb is not None and la == lb
        gt = la if agree else None

        row = {
            "commit_uid":        uid,
            "commit_url":        rows_a.get(uid, rows_b.get(uid, {})).get("commit_url", ""),
            "label_human_a":     "" if la is None else la,
            "label_human_b":     "" if lb is None else lb,
            "label_llm":         "" if ll is None else ll,
            "label_gt":          "" if gt is None else gt,
            "humans_agree":      "" if (la is None or lb is None) else int(agree),
            "rationale_human_a": rows_a.get(uid, {}).get("rationale_human", ""),
            "rationale_human_b": rows_b.get(uid, {}).get("rationale_human", ""),
            "rationale_llm":     rows_llm.get(uid, {}).get("rationale_llm", ""),
        }
        out.append(row)
        if not agree or (ll is not None and gt is not None and ll != gt):
            disagreements.append(row)

    fields = list(out[0].keys())
    with CONSOLIDATED_GT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)

    with DISAGREEMENT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(disagreements)

    print(f"\n  Consolidated GT   → {CONSOLIDATED_GT} ({len(out)} rows)")
    print(f"  Disagreement slice → {DISAGREEMENT} ({len(disagreements)} rows)")


def main() -> dict[str, float]:
    print("Loading labels...")
    human_a = load_labels(SHEET_A,   "label_human", delimiter=";")
    human_b = load_labels(SHEET_B,   "label_human")
    llm     = load_labels(SHEET_LLM, "label_llm")
    print(f"  Human A: {len(human_a)}  Human B: {len(human_b)}  LLM: {len(llm)}")

    uids_ab, ya, yb    = aligned(human_a, human_b)
    gt = {u: human_a[u] for u in uids_ab if human_a[u] == human_b[u]}
    _, yla, ylla       = aligned(llm, human_a)
    _, ylb, yllb       = aligned(llm, human_b)
    _, ygt, yllgt      = aligned(llm, gt)

    print(f"\n  Consolidated GT: {len(gt)}/{len(uids_ab)} commits where A == B\n")

    kappas = {
        "human_a_vs_human_b":  kappa_report("Human A vs Human B",       ya,   yb),
        "llm_vs_human_a":      kappa_report("LLM vs Human A",           yla,  ylla),
        "llm_vs_human_b":      kappa_report("LLM vs Human B",           ylb,  yllb),
        "llm_vs_consolidated": kappa_report("LLM vs Consolidated GT",   ygt,  yllgt),
    }

    rows_a   = load_rows(SHEET_A,   delimiter=";")
    rows_b   = load_rows(SHEET_B)
    rows_llm = load_rows(SHEET_LLM)
    save_consolidated_gt(rows_a, rows_b, rows_llm, human_a, human_b, llm)

    # ── Bootstrap 95% CI for each κ ────────────────────────────────────────
    print(f"\n  Bootstrap 95% CI ({N_BOOTSTRAP:,} resamples, seed={BOOTSTRAP_SEED}):")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cis: dict[str, dict] = {}
    for name, (y1, y2) in [
        ("human_a_vs_human_b",  (ya,   yb)),
        ("llm_vs_human_a",      (yla,  ylla)),
        ("llm_vs_human_b",      (ylb,  yllb)),
        ("llm_vs_consolidated", (ygt,  yllgt)),
    ]:
        y1 = np.asarray(y1); y2 = np.asarray(y2)
        n = len(y1)
        boots = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            try:
                boots[i] = cohen_kappa_score(y1[idx], y2[idx])
            except Exception:
                boots[i] = np.nan
        boots = boots[~np.isnan(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cis[name] = {
            "kappa": kappas[name],
            "n": n,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "ci_width": float(hi - lo),
            "n_bootstrap": int(len(boots)),
        }
        print(f"    {name:<30} κ={kappas[name]:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    KAPPA_CI.write_text(json.dumps(cis, indent=2))
    print(f"  → {KAPPA_CI}")

    return kappas


if __name__ == "__main__":
    main()
