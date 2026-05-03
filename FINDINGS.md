# Key Findings

A hand-curated summary of the most important results, each with the **script that produced it** so anyone can reproduce. Re-run any script from the repo root.

> **Note on freshness.** This file is **not auto-generated**. The numbers below are a snapshot taken from the most recent training run (see commit history). They are correct for the artifacts currently in this repository but will drift if the pipeline is re-trained on different data without updating this file. The auto-generated companion is **[artifacts/results/paper_report.md](artifacts/results/paper_report.md)** — that file always reflects the latest run because every number in it is derived from CSVs at generation time. If you re-run `python scripts/build_all_paper_artifacts.py` and want fresh prose, re-derive from `paper_report.md`.

---

## 1. LLM judge agreement with human consensus is "substantial"

**κ = 0.60** on the 69 commits where both human reviewers agreed (95% bootstrap CI [0.38, 0.79]).

By the Landis & Koch (1977) classification this is "substantial" agreement. Inter-human κ on the same 100-commit gold set is **0.28** (95% CI [0.07, 0.47]), indicating that TD labeling is itself subjective; the LLM-vs-consensus κ is higher than either individual reviewer's agreement with the consensus.

**Evidence**
- [scripts/kappa_analysis.py](scripts/kappa_analysis.py) — computes all four κs + bootstrap CIs
- Outputs: [artifacts/results/consolidated_gt.csv](artifacts/results/consolidated_gt.csv), [artifacts/results/kappa_ci.json](artifacts/results/kappa_ci.json)

---

## 2. LLM labels are NOT redundant with the SATD-keyword regex

**κ(LLM, SATD) = 0.34** on the full 21,038-commit corpus.

Of the 1,310 LLM-flagged TD commits, **947 have no SATD keywords** — the LLM identifies TD beyond what regex detection catches. This justifies using an LLM judge rather than relying on the regex alone.

**Evidence**
- [scripts/paper_report.py](scripts/paper_report.py) — `_dataset_stats()` computes this κ
- Source data: [data/features_with_llm_labels.csv](data/features_with_llm_labels.csv)

---

## 3. ML on mechanical metrics significantly beats the SATD baseline

| Approach | F1 | Precision | Recall | MCC |
|---|---:|---:|---:|---:|
| **ML (RF, 5-seed mean)** | **0.395 ± 0.017** | 0.292 | **0.619** | **0.378** |
| SATD-regex baseline | 0.317 | 0.470 | 0.239 | 0.309 |

McNemar's test on per-commit predictions: χ² = 40.1, p = 2.4 × 10⁻¹⁰. ML achieves higher recall (0.62 vs 0.24) at the cost of precision (0.29 vs 0.47).

**Evidence**
- [scripts/baselines_and_tests.py](scripts/baselines_and_tests.py) — runs SATD baseline, multi-seed (5×), McNemar's test
- Outputs: [artifacts/results/baseline_metrics.csv](artifacts/results/baseline_metrics.csv), [artifacts/results/multiseed_metrics.csv](artifacts/results/multiseed_metrics.csv), [artifacts/results/mcnemar_test.txt](artifacts/results/mcnemar_test.txt)

---

## 4. Model performance is highly stable across random seeds

**AUC σ = 0.0013** across 5 seeds. F1 σ = 0.017. The model is essentially deterministic in ranking quality and only marginally noisy in F1.

**Evidence**
- [scripts/baselines_and_tests.py](scripts/baselines_and_tests.py) — `--seeds` parameter (default 5)
- Output: [artifacts/results/multiseed_metrics.csv](artifacts/results/multiseed_metrics.csv)

---

## 5. LLM confidence correlates with model performance

| LLM confidence bucket | n | F1 | P | R | AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| low (<0.85) | 23 | — | — | — | — | — |
| mid (0.85 – 0.95) | 1,476 | 0.335 | 0.224 | 0.658 | 0.785 | 0.293 |
| **high (≥0.95)** | 2,713 | **0.630** | **0.744** | 0.547 | **0.898** | **0.626** |
| ALL | 4,212 | 0.408 | 0.307 | 0.606 | 0.862 | 0.387 |

On the 64% of test commits where the LLM judge confidence is ≥0.95, the model achieves F1=0.63, P=0.74, AUC=0.90, MCC=0.63. The 80 low-confidence commits across the full corpus are recorded as candidates for additional human review.

**Evidence**
- [scripts/confidence_stratified_eval.py](scripts/confidence_stratified_eval.py) — buckets test set by LLM confidence and reports metrics per bucket
- [scripts/audit_analysis.py](scripts/audit_analysis.py) — produces the low-confidence slice for review
- Outputs: [artifacts/results/confidence_stratified_metrics.csv](artifacts/results/confidence_stratified_metrics.csv), [artifacts/results/audit_low_confidence.csv](artifacts/results/audit_low_confidence.csv)

---

## 6. Cost-function analysis — the optimal threshold depends on what we assume errors actually cost

We evaluate the model under an explicit misclassification cost function:

> **Cost = C_FN · FN + C_FP · FP**

A false negative (a TD-introducing commit the model failed to flag) is generally more expensive than a false positive (an unnecessary entry on the reviewer's queue): the FN's cost compounds in maintenance debt, the FP's cost is bounded by review time. We do not commit to a single ratio; instead we report three:

| Ratio (C_FN:C_FP) | Threshold | TP | FP | FN | TD caught | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1:1 | 0.580 | 22 | 12 | 204 | **10%** | 0.169 |
| 5:1 | 0.312 | 133 | 265 | 93 | **59%** | 0.426 |
| 10:1 | 0.240 | 149 | 412 | 77 | **66%** | 0.379 |

Three observations:

- **At 1:1 the model becomes silent.** With a 5.4% positive rate in the test set, treating both errors as equally bad mathematically pushes the optimum toward not flagging anything; only 22 of 226 TD commits (10%) get caught. This is exactly why F1 — not raw accuracy — is the standard metric for imbalanced detection.
- **F1-optimal lies in the same neighbourhood as cost-optimal at 5:1.** The widely used F1 criterion implicitly assumes a cost ratio close to 5:1; reporting F1 alone is therefore a hidden cost-modelling choice rather than a neutral metric.
- **At 10:1 the cost-optimal threshold (≈0.24) lies below the F1-optimal threshold (≈0.28).** Catching 12 additional TD commits requires accepting 103 additional false positives. Whether this trade is worthwhile depends on the deployment context.

The 5:1 and 10:1 ratios are not derived from a published cost model — we adopt them as plausible bounds motivated by the gap between typical code-review effort (minutes per commit) and typical TD remediation effort (hours to days). The 1:1 row is included as a sensitivity check, not a recommended operating point.

**Evidence**
- [scripts/threshold_curve.py](scripts/threshold_curve.py) — computes cost across thresholds and identifies cost-optimal points for each ratio
- Outputs: [artifacts/results/cost_curve.csv](artifacts/results/cost_curve.csv), cost-rows in [artifacts/results/operating_points.csv](artifacts/results/operating_points.csv)
- Plot: [artifacts/figures/cost_vs_threshold.png](artifacts/figures/cost_vs_threshold.png)

---

## 7. Threshold trade-off — choose the operating point that fits the use case

| Operating point | Threshold | P | R | F1 |
|---|---:|---:|---:|---:|
| default 0.5 | 0.500 | 0.521 | 0.164 | 0.249 |
| **F1-optimal** (default reported point) | 0.284 | 0.307 | 0.606 | 0.408 |
| **F0.5-optimal (review queue)** | 0.388 | 0.385 | 0.438 | 0.410 |
| F2-optimal (screening) | 0.304 | 0.325 | 0.597 | 0.421 |
| high-precision (P≥0.5) | 0.492 | 0.500 | 0.168 | 0.252 |
| high-recall (R≥0.8) | 0.116 | 0.154 | 0.810 | 0.258 |

The same trained model can operate at P=0.50 / R=0.17 (high-precision flagging) or at R=0.81 / P=0.15 (broad screening) by changing the decision threshold.

**Evidence**
- [scripts/threshold_curve.py](scripts/threshold_curve.py) — generates the full PR curve and operating points
- Outputs: [artifacts/results/pr_curve.csv](artifacts/results/pr_curve.csv), [artifacts/results/operating_points.csv](artifacts/results/operating_points.csv)

---

## 8. XAI: three independent methods agree on the dominant features

`lines_added` is rank-1 across SHAP, LIME, and Permutation Importance. The complete top-3 differs slightly per method, but all three identify size and complexity deltas (`lines_added`, `cc_delta_max`, `churn_delta`) as the dominant signal, with author/maturity context (`n_authors_till_now`, `contributors_count`) as secondary.

**Evidence**
- [scripts/run_pipeline.py](scripts/run_pipeline.py) — `--stage xai`
- Output: [artifacts/results/xai_topk_lgbm.csv](artifacts/results/xai_topk_lgbm.csv)

---

## 9. Cross-project generalization is weaker than within-project temporal generalization

| Held-out repo | F1 | AUC | MCC |
|---|---:|---:|---:|
| keras | 0.453 | 0.861 | 0.406 |
| fastapi | 0.365 | 0.863 | 0.335 |
| requests | 0.355 | 0.862 | 0.342 |
| scrapy | 0.335 | 0.836 | 0.304 |
| flask | 0.224 | 0.823 | 0.231 |
| **mean (RF/all/none)** | **0.293** | **0.845** | **0.289** |

LOPO mean F1 = 0.29 vs time-split test F1 = 0.41. The model transfers reasonably across repos by ranking quality (AUC ≥ 0.82 everywhere) but absolute classification quality drops. Flask is hardest — likely due to size and domain differences.

**Evidence**
- [scripts/run_pipeline.py](scripts/run_pipeline.py) — `--stage train --lopo`
- Output: [artifacts/results/metrics_lopo.csv](artifacts/results/metrics_lopo.csv)

---

## 10. The gold set is stratified, by design

The 100-commit human-reviewed sample uses a 50/50 split by `label_satd`. Pure random sampling at the corpus's 3.1% SATD rate would yield ~3 SATD-positive cases — too few for stable κ estimates. By-repo balance is moderate (max deviation 36%); large commits are intentionally over-represented because SATD keywords correlate with diff size.

**Evidence**
- [scripts/audit_analysis.py](scripts/audit_analysis.py)
- Output: [artifacts/results/audit_stratification.csv](artifacts/results/audit_stratification.csv)

---

## How to reproduce all of this

```bash
# Single command — runs all 11 steps in order, idempotent (skips cached steps)
python scripts/build_all_paper_artifacts.py

# After it completes
open artifacts/results/paper_report.md
```

For finer control, see the individual scripts referenced above.
