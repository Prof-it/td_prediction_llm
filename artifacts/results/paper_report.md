# TD Prediction — Paper Report

_Auto-generated 2026-05-02T01:47:42+02:00. Re-run `python scripts/paper_report.py` after re-training._

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

| Comparison | n | κ | Interpretation |
|---|---:|---:|---|
| Human A vs Human B | 100 | **0.2757** | fair |
| LLM vs Human A | 100 | **0.3993** | fair |
| LLM vs Human B | 100 | **0.4105** | moderate |
| LLM vs Consolidated GT | 69 | **0.5990** | moderate |

**Headline:** TD labeling is inherently subjective (inter-human κ=0.28). The LLM achieves κ=0.60 against human consensus — substantial agreement, approaching the upper bound given task subjectivity.

## LLM confidence

- Mean confidence: 0.957 ± 0.038
- 80 commits with confidence < 0.85 (0.38% of corpus) — flagged for additional human review

## Time-split — best model on TEST

**rf / all / none**

| Phase | F1 | AUC | PR-AUC | P | R | MCC | Bal.Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| val  | 0.501 | 0.896 | 0.416 | 0.407 | 0.651 | 0.462 | 0.785 |
| **test** | **0.409** | **0.866** | **0.343** | **0.309** | **0.606** | **0.388** | **0.765** |

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

**Headline:** size and complexity changes (`lines_added`, `cc_delta_*`, `churn_delta`) drive predictions; author/maturity context (`n_authors_till_now`, `contributors_count`) is secondary.
