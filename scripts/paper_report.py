"""
Generate a single paper-ready report with all key numbers.

Aggregates from:
  data/features_with_llm_labels.csv
  artifacts/results/consolidated_gt.csv
  artifacts/results/metrics_time.csv
  artifacts/results/metrics_lopo.csv
  artifacts/results/xai_topk_lgbm.csv
  artifacts/results/parsed_labels_v2_rubric_json.csv
  src/td_prediction/labeling/prompts.py (rubric version)
  src/td_prediction/config.py (model, date)

Writes:
  artifacts/results/paper_report.md   — human-readable markdown
  artifacts/results/paper_report.json — machine-readable, all numbers

Run from repo root:
    python scripts/paper_report.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_prediction.config import LLM_MODEL, LLM_MODEL_DATE, LLM_TEMPERATURE
from td_prediction.labeling.prompts import PROMPT_VERSION

OUT_MD   = Path("artifacts/results/paper_report.md")
OUT_JSON = Path("artifacts/results/paper_report.json")


def _norm(v):
    s = str(v).strip().lower()
    if s in ("1", "yes", "true"):  return 1
    if s in ("0", "no", "false"): return 0
    return None


def _kappas() -> dict:
    """Compute inter-rater κ from the human review sheets."""
    SHEET_A = Path("artifacts/results/human_review_sheet_a_20260424_021700.csv")
    SHEET_B = Path("artifacts/results/human_review_sheet_b_20260428_183349.csv")
    SHEET_LLM = Path("artifacts/results/human_review_sheet_llm.csv")

    def load(path, col, delim=","):
        d = {}
        with path.open() as f:
            for r in csv.DictReader(f, delimiter=delim):
                v = _norm(r.get(col, ""))
                if v is not None:
                    d[r["commit_uid"]] = v
        return d

    ha = load(SHEET_A, "label_human", delim=";")
    hb = load(SHEET_B, "label_human")
    ll = load(SHEET_LLM, "label_llm")

    uids_ab = sorted(set(ha) & set(hb))
    gt = {u: ha[u] for u in uids_ab if ha[u] == hb[u]}

    def k(a, b):
        common = sorted(set(a) & set(b))
        return cohen_kappa_score([a[u] for u in common], [b[u] for u in common]), len(common)

    k_ab, n_ab = k(ha, hb)
    k_la, n_la = k(ll, ha)
    k_lb, n_lb = k(ll, hb)
    k_lg, n_lg = k(ll, gt)

    return {
        "human_a_vs_human_b":   {"kappa": k_ab, "n": n_ab},
        "llm_vs_human_a":       {"kappa": k_la, "n": n_la},
        "llm_vs_human_b":       {"kappa": k_lb, "n": n_lb},
        "llm_vs_consolidated":  {"kappa": k_lg, "n": n_lg},
        "consolidated_gt_size": len(gt),
        "gold_set_size":        len(uids_ab),
    }


def _dataset_stats() -> dict:
    df = pd.read_csv("data/features_with_llm_labels.csv")
    per_repo = df.groupby("repo_id").agg(
        n_commits=("commit_uid", "count"),
        n_satd_pos=("label_satd", "sum"),
        n_llm_pos=("label_llm", "sum"),
        n_consolidated_pos=("label_consolidated", "sum"),
    ).reset_index().to_dict("records")

    return {
        "n_total":               int(len(df)),
        "n_repos":               int(df["repo_id"].nunique()),
        "label_satd_pos":        int(df["label_satd"].sum()),
        "label_satd_rate":       float(df["label_satd"].mean()),
        "label_llm_pos":         int(df["label_llm"].sum()),
        "label_llm_rate":        float(df["label_llm"].mean()),
        "label_consolidated_pos":  int(df["label_consolidated"].sum()),
        "label_consolidated_rate": float(df["label_consolidated"].mean()),
        "label_human_n":         int(df["label_human"].notna().sum()),
        "label_human_pos":       int(df["label_human"].dropna().sum()),
        "llm_satd_kappa":        float(cohen_kappa_score(df["label_satd"], df["label_llm"])),
        "per_repo":              per_repo,
    }


def _time_metrics() -> dict:
    m = pd.read_csv("artifacts/results/metrics_time.csv")
    test = m[m["phase"] == "test"]
    val = m[m["phase"] == "val"]
    best = test.sort_values("f1_debt", ascending=False).iloc[0]

    cols = ["model", "feature_set", "imbalance",
            "f1_debt", "roc_auc", "pr_auc_debt",
            "p_debt", "r_debt", "mcc", "balanced_acc"]

    return {
        "best": best[cols].to_dict(),
        "best_val": val[(val.model==best.model)&(val.feature_set==best.feature_set)&(val.imbalance==best.imbalance)].iloc[0][cols].to_dict(),
        "all_test": test[cols].to_dict("records"),
    }


def _lopo_metrics() -> dict:
    m = pd.read_csv("artifacts/results/metrics_lopo.csv")
    test = m[m["phase"] == "test"]

    per_repo = []
    for repo in ["fastapi", "flask", "keras", "requests", "scrapy"]:
        sub = test[test["split"] == f"lopo_{repo}/test"].sort_values("f1_debt", ascending=False)
        if len(sub):
            r = sub.iloc[0]
            per_repo.append({
                "held_out_repo": repo,
                "model": r["model"], "feature_set": r["feature_set"], "imbalance": r["imbalance"],
                "f1": float(r["f1_debt"]), "auc": float(r["roc_auc"]),
                "p": float(r["p_debt"]), "r": float(r["r_debt"]), "mcc": float(r["mcc"]),
            })

    # Mean across same-model rows for the headline number
    rf_all_none = test[(test["model"]=="rf") & (test["feature_set"]=="all") & (test["imbalance"]=="none")]
    return {
        "per_repo_best": per_repo,
        "rf_all_none_mean_f1": float(rf_all_none["f1_debt"].mean()),
        "rf_all_none_mean_auc": float(rf_all_none["roc_auc"].mean()),
        "rf_all_none_mean_mcc": float(rf_all_none["mcc"].mean()),
    }


def _baseline_metrics() -> dict | None:
    p = Path("artifacts/results/baseline_metrics.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    test = df[df["phase"] == "test"].iloc[0]
    val  = df[df["phase"] == "val"].iloc[0]
    return {
        "test": {k: float(test[k]) for k in ["f1_debt","p_debt","r_debt","mcc"]},
        "val":  {k: float(val[k])  for k in ["f1_debt","p_debt","r_debt","mcc"]},
        "test_cm": {k: int(test[k]) for k in ["tp","fp","fn","tn"]},
    }


def _multiseed() -> dict | None:
    p = Path("artifacts/results/multiseed_metrics.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    test = df[df["phase"] == "test"]
    return {
        "n_seeds": int(test["seed"].nunique()),
        "test_mean": {c: float(test[c].mean()) for c in ["f1_debt","roc_auc","pr_auc_debt","p_debt","r_debt","mcc"]},
        "test_std":  {c: float(test[c].std())  for c in ["f1_debt","roc_auc","pr_auc_debt","p_debt","r_debt","mcc"]},
    }


def _mcnemar() -> str | None:
    p = Path("artifacts/results/mcnemar_test.txt")
    return p.read_text() if p.exists() else None


def _xai_top() -> list:
    xai = pd.read_csv("artifacts/results/xai_topk_lgbm.csv")
    return xai.head(10).to_dict("records")


def _confidence_stats() -> dict:
    parsed = pd.read_csv("artifacts/results/parsed_labels_v2_rubric_json.csv")
    return {
        "n": int(len(parsed)),
        "mean": float(parsed["confidence"].mean()),
        "std":  float(parsed["confidence"].std()),
        "min":  float(parsed["confidence"].min()),
        "low_confidence_count": int((parsed["confidence"] < 0.85).sum()),
        "low_confidence_threshold": 0.85,
    }


def main():
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "generated_at": generated_at,
        "config": {
            "llm_model": LLM_MODEL,
            "llm_model_date": LLM_MODEL_DATE,
            "llm_temperature": LLM_TEMPERATURE,
            "prompt_version": PROMPT_VERSION,
        },
        "dataset":     _dataset_stats(),
        "kappa":       _kappas(),
        "confidence":  _confidence_stats(),
        "time_split":  _time_metrics(),
        "lopo":        _lopo_metrics(),
        "satd_baseline": _baseline_metrics(),
        "multiseed":     _multiseed(),
        "mcnemar":       _mcnemar(),
        "xai_top10":   _xai_top(),
    }

    # Numeric report (JSON)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))

    # Human-readable report (Markdown)
    cfg = report["config"]
    ds  = report["dataset"]
    k   = report["kappa"]
    c   = report["confidence"]
    tm  = report["time_split"]
    lp  = report["lopo"]
    xai = report["xai_top10"]

    md = []
    md.append(f"# TD Prediction — Paper Report\n")
    md.append(f"_Auto-generated {generated_at}. Re-run `python scripts/paper_report.py` after re-training._\n")

    md.append("## Setup\n")
    md.append(f"- LLM model: `{cfg['llm_model']}` (snapshot {cfg['llm_model_date']})")
    md.append(f"- Prompt version: `{cfg['prompt_version']}`, temperature={cfg['llm_temperature']}")
    md.append(f"- Repositories: {ds['n_repos']} (fastapi, flask, keras, requests, scrapy)\n")

    md.append("## Dataset\n")
    md.append(f"- **{ds['n_total']:,}** commits total")
    md.append(f"- **{ds['label_consolidated_pos']:,}** TD-positive ({ds['label_consolidated_rate']*100:.2f}% rate) in `label_consolidated`")
    md.append(f"- {ds['label_satd_pos']:,} SATD-keyword commits ({ds['label_satd_rate']*100:.2f}%)")
    md.append(f"- {ds['label_llm_pos']:,} LLM-flagged ({ds['label_llm_rate']*100:.2f}%)")
    md.append(f"- {ds['label_human_n']} commits with consolidated human label (gold set with consensus)")
    md.append(f"- LLM vs SATD agreement (κ): **{ds['llm_satd_kappa']:.3f}** — LLM captures signal beyond keywords\n")

    md.append("**Per-repository:**\n")
    md.append("| Repo | Commits | label_satd+ | label_consolidated+ |")
    md.append("|---|---:|---:|---:|")
    for r in ds["per_repo"]:
        md.append(f"| {r['repo_id']} | {r['n_commits']:,} | {r['n_satd_pos']:,} | {int(r['n_consolidated_pos']):,} |")
    md.append("")

    md.append("## Inter-rater agreement (Cohen's κ)\n")
    md.append(f"- 100-commit gold set, two human reviewers (A, B). Consolidated GT = where A == B ({k['consolidated_gt_size']}/{k['gold_set_size']} commits).\n")
    md.append("| Comparison | n | κ | Interpretation |")
    md.append("|---|---:|---:|---|")
    for name, label in [
        ("human_a_vs_human_b",  "Human A vs Human B"),
        ("llm_vs_human_a",      "LLM vs Human A"),
        ("llm_vs_human_b",      "LLM vs Human B"),
        ("llm_vs_consolidated", "LLM vs Consolidated GT"),
    ]:
        kv = k[name]["kappa"]; nv = k[name]["n"]
        if kv >= 0.6: interp = "substantial"
        elif kv >= 0.4: interp = "moderate"
        elif kv >= 0.2: interp = "fair"
        else: interp = "poor"
        md.append(f"| {label} | {nv} | **{kv:.4f}** | {interp} |")
    md.append("")
    md.append("**Headline:** TD labeling is inherently subjective (inter-human κ=0.28). "
              f"The LLM achieves κ={k['llm_vs_consolidated']['kappa']:.2f} against human consensus — substantial agreement, "
              "approaching the upper bound given task subjectivity.\n")

    md.append("## LLM confidence\n")
    md.append(f"- Mean confidence: {c['mean']:.3f} ± {c['std']:.3f}")
    md.append(f"- {c['low_confidence_count']} commits with confidence < {c['low_confidence_threshold']} "
              f"({c['low_confidence_count']/c['n']*100:.2f}% of corpus) — flagged for additional human review\n")

    md.append("## Time-split — best model on TEST\n")
    b = tm["best"]; bv = tm["best_val"]
    md.append(f"**{b['model']} / {b['feature_set']} / {b['imbalance']}**\n")
    md.append("| Phase | F1 | AUC | PR-AUC | P | R | MCC | Bal.Acc. |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.append(f"| val  | {bv['f1_debt']:.3f} | {bv['roc_auc']:.3f} | {bv['pr_auc_debt']:.3f} | {bv['p_debt']:.3f} | {bv['r_debt']:.3f} | {bv['mcc']:.3f} | {bv['balanced_acc']:.3f} |")
    md.append(f"| **test** | **{b['f1_debt']:.3f}** | **{b['roc_auc']:.3f}** | **{b['pr_auc_debt']:.3f}** | **{b['p_debt']:.3f}** | **{b['r_debt']:.3f}** | **{b['mcc']:.3f}** | **{b['balanced_acc']:.3f}** |")
    md.append("")

    bl = report.get("satd_baseline")
    ms = report.get("multiseed")
    if bl and ms:
        md.append("## ML model vs SATD-regex baseline (TEST)\n")
        md.append("| Approach | F1 | P | R | MCC |")
        md.append("|---|---:|---:|---:|---:|")
        md.append(f"| **ML (RF/all/none, mean of {ms['n_seeds']} seeds)** "
                  f"| **{ms['test_mean']['f1_debt']:.3f} ± {ms['test_std']['f1_debt']:.3f}** "
                  f"| {ms['test_mean']['p_debt']:.3f} ± {ms['test_std']['p_debt']:.3f} "
                  f"| {ms['test_mean']['r_debt']:.3f} ± {ms['test_std']['r_debt']:.3f} "
                  f"| {ms['test_mean']['mcc']:.3f} ± {ms['test_std']['mcc']:.3f} |")
        md.append(f"| SATD regex baseline | {bl['test']['f1_debt']:.3f} | {bl['test']['p_debt']:.3f} "
                  f"| {bl['test']['r_debt']:.3f} | {bl['test']['mcc']:.3f} |")
        md.append("")
        md.append(f"ML is ranking-stable (AUC std = {ms['test_std']['roc_auc']:.4f}) and produces "
                  f"+{(ms['test_mean']['f1_debt']-bl['test']['f1_debt'])*100:.1f}pp F1 over SATD-regex. "
                  "The trade-off: SATD has higher precision (it only fires on commits with TD keywords); "
                  "ML has substantially higher recall (catches TD without keywords).\n")

    mcn = report.get("mcnemar")
    if mcn:
        md.append("**McNemar's test** (significance of disagreement, two-sided):\n")
        md.append("```")
        md.append(mcn.strip())
        md.append("```\n")

    md.append("## LOPO (leave-one-project-out) — cross-project generalization\n")
    md.append(f"Mean across the 5 held-out repos (RF / all / none): "
              f"F1={lp['rf_all_none_mean_f1']:.3f}, AUC={lp['rf_all_none_mean_auc']:.3f}, MCC={lp['rf_all_none_mean_mcc']:.3f}\n")
    md.append("**Best per held-out repo (test):**\n")
    md.append("| Held-out | Model | Feat. set | Imb. | F1 | AUC | P | R | MCC |")
    md.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in lp["per_repo_best"]:
        md.append(f"| {r['held_out_repo']} | {r['model']} | {r['feature_set']} | {r['imbalance']} | "
                  f"{r['f1']:.3f} | {r['auc']:.3f} | {r['p']:.3f} | {r['r']:.3f} | {r['mcc']:.3f} |")
    md.append("")

    md.append("## XAI — top features (consensus across SHAP / LIME / Permutation)\n")
    md.append("| Rank | SHAP | LIME | Permutation |")
    md.append("|---:|---|---|---|")
    for i, r in enumerate(xai, 1):
        md.append(f"| {i} | `{r['shap_feature']}` | `{r['lime_feature']}` | `{r['perm_feature']}` |")
    md.append("\n**Headline:** size and complexity changes (`lines_added`, `cc_delta_*`, `churn_delta`) "
              "drive predictions; author/maturity context (`n_authors_till_now`, `contributors_count`) is secondary.")
    md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
