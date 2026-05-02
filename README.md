# td_prediction_llm

Predict whether a commit introduces technical debt, using mechanical code metrics trained against an LLM-as-judge ground truth that has been validated against a 100-commit human gold set.

![Workflow](artifacts/figures/workflowdiagram.drawio.svg)

**See [FINDINGS.md](FINDINGS.md) for a summary of results with reproduction pointers.**
**See [artifacts/results/paper_report.md](artifacts/results/paper_report.md) for the auto-generated full numeric report.**

## Quick start

```bash
make install                                       # install dependencies
python scripts/build_all_paper_artifacts.py        # run everything (idempotent)
open artifacts/results/paper_report.md             # final report
```

The orchestrator skips any step whose outputs are already cached. Use `--force` to re-run everything from scratch, `--skip-train` to skip the slow training steps and only re-run aggregations.

## Layout

### Source code (`src/td_prediction/`)

```
config.py                  paths, seeds, pinned LLM_MODEL + LLM_MODEL_DATE,
                           LABEL_COLS / FEATURE_COLS / LEAKAGE_COLS
mining.py                  PyDriller feature extraction
labeling/
    satd.py                SATD keyword regex (baseline label_satd)
    prompts.py             versioned LLM-as-judge prompts (single source of
                           truth — RUBRIC, SYSTEM_PROMPT, PROMPT_VERSION)
    llm_judge.py           batch build / submit / poll / parse
    human_review.py        gold-set sampler + agreement metrics
data/splits.py             time + LOPO splits, feature-set selection,
                           prepare_xy with leakage filtering
models/
    trainer.py             RF / LightGBM / XGBoost + imbalance handling
    metrics.py             PR-AUC, MCC, F1, balanced accuracy, etc.
    threshold.py           tune classification threshold on validation set
    plots.py               PR, ROC, class-balance plots
xai/
    shap_explainer.py      SHAP explanations
    lime_explainer.py      LIME explanations
    permutation.py         permutation importance
```

### Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| **`build_all_paper_artifacts.py`** | **End-to-end orchestrator — runs all 11 steps in order, idempotent** |
| `llm_autoreview.py`         | Sync API LLM labeling on the 100-commit gold set |
| `build_v2_batches.py`       | Build batch JSONL files for full-corpus LLM labeling |
| `submit_batches.py`         | Submit + poll OpenAI batches |
| `relabel_oversized.py`      | Re-label commits whose diffs exceed the batch context limit |
| `kappa_analysis.py`         | Cohen's κ across all rater pairs + bootstrap CIs |
| `patch_labels.py`           | Update features CSV with v2.3 labels and `label_consolidated` |
| `run_pipeline.py`           | Mine / label / train / xai / analysis stages |
| `audit_analysis.py`         | Gold-set stratification check + low-confidence slice |
| `baselines_and_tests.py`    | SATD-regex baseline + multi-seed runs + McNemar's test |
| `threshold_curve.py`        | Precision-recall curve + canonical operating points |
| `confidence_stratified_eval.py` | Test metrics bucketed by LLM confidence |
| `paper_report.py`           | Aggregate everything into `paper_report.{md,json}` |

### Artifacts (`artifacts/`)

```
results/
    paper_report.md / .json          ← the auto-generated unified report
    consolidated_gt.csv              100 commits with all 4 labels + GT
    disagreement_slice.csv           commits where reviewers disagree
    kappa_ci.json                    bootstrap 95% CIs for all κ values
    metrics_time.csv                 18 model configurations × val/test
    metrics_lopo.csv                 LOPO cross-project results
    baseline_metrics.csv             SATD-regex baseline metrics
    multiseed_metrics.csv            5-seed mean ± std
    mcnemar_test.txt                 ML vs SATD significance test
    pr_curve.csv                     full precision-recall curve data
    operating_points.csv             6 canonical operating points
    confidence_stratified_metrics.csv per-confidence-bucket test metrics
    audit_stratification.csv         gold-set stratification audit
    audit_low_confidence.csv         80 commits flagged for human review
    xai_topk_lgbm.csv                SHAP / LIME / Permutation top features
    parsed_labels_v2_rubric_json.csv per-commit LLM label + rationale + confidence
    human_review_sheet_a*.csv        human reviewer A's labels
    human_review_sheet_b*.csv        human reviewer B's labels
    human_review_sheet_llm.csv       LLM labels on the gold set
    human_review_sheet_review.html   HTML tool for new reviews
    adjudication_tool.html           side-by-side review for disagreements

figures/                              plots (SHAP summaries, etc.)
models/                               trained model bundles (joblib)
diffs/                                cached GitHub commit diffs
```

### Other directories

```
data/features_with_llm_labels.csv     21,038 commits × code metrics + labels
splits/                               cached train/val/test splits
llm_batch/                            LLM batch JSONL inputs + outputs
notebooks/                            thin Marimo orchestrator notebooks
tests/                                unit and integration tests
```

## Reproducibility

| What's pinned | Where |
|---|---|
| LLM model snapshot | `config.LLM_MODEL` (`gpt-5.4-mini-2026-03-17`) |
| LLM model date | `config.LLM_MODEL_DATE` |
| LLM temperature | `config.LLM_TEMPERATURE` (0.0) |
| Rubric version | `prompts.PROMPT_VERSION` (v2.3) |
| Random seed | `config.SEED` (42) |
| Mining cutoff date | `config.CUTOFF` |
| Train/val/test fractions | `config.TRAIN_FRAC` etc. |

`OPENAI_API_KEY` lives in `.env` (gitignored). All scripts call `load_dotenv(override=True)` so the project key always wins over shell environment.

The 21,038-commit feature CSV is committed at `data/features_with_llm_labels.csv` for reproducibility without re-mining.

## Reproducing from cached data

```bash
# One-liner (recommended) — uses cached data + features + LLM labels
python scripts/build_all_paper_artifacts.py

# Or skip the LLM autoreview step (uses cached gold-set labels)
python scripts/build_all_paper_artifacts.py --skip-autoreview

# Or kappa + reporting only (skips all training)
python scripts/build_all_paper_artifacts.py --skip-autoreview --skip-train
```

## Running end-to-end from scratch

1. Clone target repos into `repos/<name>/` (see `config.REPOS`).
2. Re-mine features:
   ```bash
   python scripts/run_pipeline.py --stage mine --force
   ```
3. Build and submit LLM-as-judge batches:
   ```bash
   # Put OPENAI_API_KEY in .env first
   python scripts/build_v2_batches.py --variant v2_rubric_json
   python scripts/submit_batches.py --glob 'llm_batch/*_v2_rubric_json.jsonl' --wait
   ```
4. Run the orchestrator:
   ```bash
   python scripts/build_all_paper_artifacts.py
   ```

## Reviewing changes / disagreements

- **Single-reviewer flow:** open `artifacts/results/human_review_sheet_review.html` in a browser served from `artifacts/results/` (loads the LLM sheet for a new reviewer to label).
- **Adjudication flow:** open `artifacts/results/adjudication_tool.html` (loads `consolidated_gt.csv` and shows Human A / Human B / LLM labels + rationales side-by-side, filterable by disagreement type).
