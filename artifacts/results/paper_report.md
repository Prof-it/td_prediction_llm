# TD Prediction — Paper Report

_Auto-generated 2026-05-02T14:10:18+02:00. Re-run `python scripts/paper_report.py` after re-training._

## Setup

- LLM model: `gpt-5.4-mini-2026-03-17` (snapshot 2026-03-17)
- Prompt version: `v2.3`, temperature=0.0
- Repositories: 5 (fastapi, flask, keras, requests, scrapy)

## Dataset

- **21,038** commits total
- **1,304** TD-positive (6.20% rate) in `label_consolidated`
- 648 SATD-keyword commits (3.08%)
- 1,311 LLM-flagged (6.23%)
- 69 commits with consolidated human label (gold set with consensus)
- LLM vs SATD agreement (κ): **0.344** — LLM captures signal beyond keywords

**Per-repository:**

| Repo | Commits | label_satd+ | label_consolidated+ |
|---|---:|---:|---:|
| fastapi | 934 | 22 | 51 |
| flask | 1,967 | 31 | 34 |
| keras | 9,256 | 380 | 751 |
| requests | 2,749 | 75 | 167 |
| scrapy | 6,132 | 140 | 301 |

## Inter-rater agreement (Cohen's κ)

- 100-commit gold set, two human reviewers (A, B). Consolidated GT = where A == B (69/100 commits).
- 95% CIs from 10,000-resample non-parametric bootstrap.

| Comparison | n | κ | 95% CI | Interpretation |
|---|---:|---:|---|---|
| Human A vs Human B | 100 | **0.2757** | [0.073, 0.465] | fair |
| LLM vs Human A | 100 | **0.3993** | [0.207, 0.579] | fair |
| LLM vs Human B | 100 | **0.4105** | [0.215, 0.591] | moderate |
| LLM vs Consolidated GT | 69 | **0.5990** | [0.376, 0.791] | moderate |

Inter-human κ on this 100-commit gold set is 0.28; LLM-vs-consensus κ is 0.60 on the 69-commit consensus subset.

## LLM confidence

- Mean confidence: 0.957 ± 0.038
- 80 commits with confidence < 0.85 (0.38% of corpus) — flagged for additional human review

## Gold-set stratification

The 100-commit human-reviewed sample was stratified to ensure both label classes are well represented. Pure random sampling at the corpus's 3.1% SATD rate would yield ~3 SATD-positive cases out of 100, making κ unreliable.

| Dimension | Stratum | Corpus | Gold | Expected | Observed |
|---|---|---:|---:|---:|---:|
| repo | fastapi | 934 | 5 | 4.4 | 5 |
| repo | flask | 1,967 | 6 | 9.3 | 6 |
| repo | keras | 9,256 | 51 | 44.0 | 51 |
| repo | requests | 2,749 | 13 | 13.1 | 13 |
| repo | scrapy | 6,132 | 25 | 29.1 | 25 |
| size | XS (≤Q1) | 6,305 | 15 | 30.0 | 15 |
| size | S (Q1-Q2) | 4,625 | 13 | 22.0 | 13 |
| size | M (Q2-Q3) | 4,873 | 16 | 23.2 | 16 |
| size | L (>Q3) | 5,235 | 56 | 24.9 | 56 |
| label_satd | 0 | 20,390 | 50 | 96.9 | 50 |
| label_satd | 1 | 648 | 50 | 3.1 | 50 |

## Time-split — best model on TEST

**rf / all / none**, classification threshold tuned on validation: **0.284** (objective: f1).

| Phase | F1 | AUC | PR-AUC | P | R | MCC | Bal.Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| val  | 0.501 | 0.896 | 0.416 | 0.407 | 0.651 | 0.462 | 0.785 |
| **test** | **0.409** | **0.866** | **0.343** | **0.309** | **0.606** | **0.388** | **0.765** |

### Full ablation (all 18 configurations, TEST)

| Model | Features | Imbalance | F1 | AUC | PR-AUC | P | R | MCC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rf | all | none | 0.409 | 0.866 | 0.343 | 0.309 | 0.606 | 0.388 |
| rf | all | class_weight | 0.407 | 0.869 | 0.325 | 0.311 | 0.588 | 0.384 |
| lgbm | all | class_weight | 0.390 | 0.853 | 0.304 | 0.290 | 0.593 | 0.368 |
| xgb | all | none | 0.384 | 0.851 | 0.301 | 0.294 | 0.553 | 0.357 |
| xgb | all | class_weight | 0.383 | 0.852 | 0.301 | 0.284 | 0.584 | 0.360 |
| rf | change_only | class_weight | 0.377 | 0.862 | 0.317 | 0.279 | 0.580 | 0.354 |
| lgbm | all | none | 0.376 | 0.861 | 0.319 | 0.294 | 0.522 | 0.346 |
| rf | change_only | none | 0.375 | 0.865 | 0.328 | 0.271 | 0.611 | 0.357 |
| rf | all | smote | 0.350 | 0.856 | 0.265 | 0.276 | 0.478 | 0.316 |
| lgbm | change_only | none | 0.339 | 0.837 | 0.271 | 0.257 | 0.496 | 0.307 |
| lgbm | change_only | class_weight | 0.333 | 0.812 | 0.258 | 0.258 | 0.469 | 0.298 |
| lgbm | all | smote | 0.318 | 0.811 | 0.256 | 0.255 | 0.425 | 0.280 |
| xgb | change_only | none | 0.316 | 0.812 | 0.267 | 0.251 | 0.425 | 0.277 |
| xgb | change_only | smote | 0.308 | 0.781 | 0.237 | 0.251 | 0.398 | 0.267 |
| xgb | all | smote | 0.308 | 0.813 | 0.229 | 0.246 | 0.412 | 0.268 |
| rf | change_only | smote | 0.306 | 0.842 | 0.239 | 0.221 | 0.496 | 0.275 |
| lgbm | change_only | smote | 0.305 | 0.795 | 0.244 | 0.258 | 0.372 | 0.263 |
| xgb | change_only | class_weight | 0.297 | 0.795 | 0.252 | 0.215 | 0.482 | 0.265 |

## ML model vs SATD-regex baseline (TEST)

| Approach | F1 | P | R | MCC |
|---|---:|---:|---:|---:|
| **ML (RF/all/none, mean of 5 seeds)** | **0.395 ± 0.017** | 0.292 ± 0.026 | 0.619 ± 0.035 | 0.378 ± 0.013 |
| SATD regex baseline | 0.317 | 0.470 | 0.239 | 0.309 |

AUC σ across seeds = 0.0013. ML F1 exceeds SATD-regex F1 by 7.9pp. SATD has higher precision (it only fires on commits with TD keywords); ML has higher recall (flags commits with structural TD signals regardless of keywords).

**McNemar's test** (significance of disagreement, two-sided):

```
McNemar's test: rf_all_none vs satd_regex
  n_test                                    = 4212
  rf_all_none               correct, other wrong: 129
  satd_regex                correct, other wrong: 254
  chi²                                      = 40.1462
  p-value                                   = 2.356e-10
  significant at α=0.05?                    = YES
```

## Threshold trade-off — operating points (TEST)

Same model, different decision thresholds. Use case dictates the choice.

| Operating point | Threshold | P | R | F1 | F0.5 | F2 | MCC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_0.5 | 0.500 | 0.521 | 0.164 | 0.249 | 0.363 | 0.190 | 0.272 | 37 | 34 | 189 | 3952 |
| f1_optimal | 0.284 | 0.307 | 0.606 | 0.408 | 0.341 | 0.507 | 0.387 | 137 | 309 | 89 | 3677 |
| f0.5_optimal | 0.388 | 0.385 | 0.438 | 0.410 | 0.395 | 0.426 | 0.375 | 99 | 158 | 127 | 3828 |
| f2_optimal | 0.304 | 0.325 | 0.597 | 0.421 | 0.357 | 0.511 | 0.398 | 135 | 281 | 91 | 3705 |
| high_precision_p>=0.5 | 0.492 | 0.500 | 0.168 | 0.252 | 0.358 | 0.194 | 0.269 | 38 | 38 | 188 | 3948 |
| high_recall_r>=0.8 | 0.116 | 0.154 | 0.810 | 0.258 | 0.184 | 0.437 | 0.279 | 183 | 1007 | 43 | 2979 |

Note: F1-optimal is the default reported operating point. F0.5 weights precision more (review-queue use cases); F2 weights recall more (screening use cases).

## Stratified evaluation by LLM confidence (TEST)

Does the model perform better on commits where the LLM judge was confident?

| Confidence bucket | n | Pos% | F1 | P | R | AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low (<0.85) | 23 | 0.00% | 0.000 | 0.000 | 0.000 | — | — | — |
| mid (0.85-0.95) | 1,476 | 8.13% | 0.335 | 0.224 | 0.658 | 0.785 | 0.281 | 0.293 |
| high (>=0.95) | 2,713 | 3.91% | 0.630 | 0.744 | 0.547 | 0.898 | 0.611 | 0.626 |
| ALL | 4,212 | 5.37% | 0.408 | 0.307 | 0.606 | 0.862 | 0.342 | 0.387 |

Model performance increases monotonically with LLM judge confidence. Low-confidence commits are recorded for additional human review.

## LOPO (leave-one-project-out) — cross-project generalization

Mean across the 5 held-out repos (RF / all / none): F1=0.293, AUC=0.845, MCC=0.289

**Best per held-out repo (test):**

| Held-out | Model | Feat. set | Imb. | F1 | AUC | P | R | MCC |
|---|---|---|---|---:|---:|---:|---:|---:|
| fastapi | rf | all | class_weight | 0.365 | 0.863 | 0.278 | 0.529 | 0.335 |
| flask | rf | all | class_weight | 0.224 | 0.823 | 0.154 | 0.412 | 0.231 |
| keras | rf | all | none | 0.453 | 0.861 | 0.369 | 0.589 | 0.406 |
| requests | lgbm | all | class_weight | 0.355 | 0.862 | 0.500 | 0.275 | 0.342 |
| scrapy | rf | all | class_weight | 0.335 | 0.836 | 0.362 | 0.312 | 0.304 |

## XAI — top features (consensus across SHAP / LIME / Permutation)

| Rank | SHAP | LIME | Permutation |
|---:|---|---|---|
| 1 | `lines_added` | `lines_added` | `lines_added` |
| 2 | `n_authors_till_now` | `churn_delta` | `cc_delta_max` |
| 3 | `churn_cum` | `cc_delta_max` | `cc_delta_sum` |
| 4 | `contributors_count` | `n_authors_till_now` | `contributors_count` |
| 5 | `complexity_current_sum` | `churn_cum` | `churn_delta` |
| 6 | `cc_delta_max` | `cc_delta_sum` | `n_commits_file_past90d` |
| 7 | `contributors_cum` | `hunks` | `hunks` |
| 8 | `n_commits_file_past90d` | `dmm_unit_interfacing` | `contributors_cum` |
| 9 | `cc_delta_sum` | `n_methods_changed` | `dmm_unit_interfacing` |
| 10 | `churn_delta` | `n_commits_file_past90d` | `complexity_current_sum` |

`lines_added` is rank-1 across all three explainers. Size and complexity deltas (`lines_added`, `cc_delta_*`, `churn_delta`) appear consistently in the top-5; author/maturity features (`n_authors_till_now`, `contributors_count`) are next.
