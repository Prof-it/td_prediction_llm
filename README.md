# td_prediction_llm

Source code and artifacts for the MOC2025 workshop paper (DECLARE conference) on
predicting technical debt per commit via mining software repositories, an
LLM-as-judge labeling workflow, and XAI-driven inspection.

The repository extends a prior bachelor's thesis in which the pipeline suffered
from label leakage, shortcut learning, and class-imbalance issues. The current
codebase fixes those problems and adds a human-in-the-loop gold set.

![Workflow](artifacts/figures/workflowdiagram.drawio.svg)

## Quick start

```bash
make install       # Install dependencies
make reproduce     # Run label → train → xai → analysis (uses cached data)
```

Outputs go to `artifacts/`. Next, follow [`HUMAN_TODOS.md`](HUMAN_TODOS.md) to label the gold set.

## Layout

```
src/td_prediction/         # Python package (all logic lives here)
    config.py              # paths, seeds, constants
    mining.py              # PyDriller feature extraction
    labeling/
        satd.py            # SATD regex baseline
        prompts.py         # versioned LLM-as-judge prompts (v1 legacy, v2 rubric+JSON)
        llm_judge.py       # batch build / submit / poll / parse
        human_review.py    # gold-set sampler + agreement metrics
    data/splits.py         # train/val/test + LOPO, feature-set selection
    models/
        trainer.py         # RF / LightGBM / XGBoost + imbalance handling
        metrics.py         # PR-AUC, MCC, per-class F1 reporting
        threshold.py       # threshold tuning on VAL only
        plots.py           # PR, ROC, class-balance plots
    xai/
        shap_explainer.py  # SHAP
        lime_explainer.py  # LIME (local + sampled global)
        permutation.py     # permutation importance + ranking comparison

notebooks/                 # Thin orchestrator notebooks (Marimo .py files)
    01_mine.py
    02_label.py
    03_train_eval.py
    04_xai.py
    05_analysis.py

scripts/                   # CLI entry points
    run_pipeline.py        # mine | label | train | xai | analysis | all
    build_v2_batches.py    # build new LLM batch JSONL files
    submit_batches.py      # submit + poll OpenAI batches

configs/                   # YAML configs (reserved; defaults live in config.py)
tests/                     # smoke tests
artifacts/
    figures/               # all generated PNG/PDF plots
    results/               # metric CSVs, ranking CSVs, human review sheet
    models/                # persisted .joblib bundles
    legacy/                # original thesis notebook.py / notebook.ipynb

data/                      # features_<repo>.csv (mined features per commit)
llm_batch/                 # LLM batch input + output JSONL (large)
splits/                    # cached CSV splits from legacy iterations
presentation/              # workshop slides and QR code
```

## Reproducing the paper from cached data

The repo ships with the mined feature CSVs (`data/features_*.csv`) and the
cached LLM-judge outputs (`llm_batch/*_output.jsonl`) from the thesis
iterations, so the model-training and XAI stages are reproducible without
re-cloning the 5 target repositories or calling the OpenAI API.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# End-to-end with caches (no mining, no API calls):
python scripts/run_pipeline.py --stage label
python scripts/run_pipeline.py --stage train --lopo
python scripts/run_pipeline.py --stage xai --model lgbm
python scripts/run_pipeline.py --stage analysis
```

Outputs land under `artifacts/`.

## Running end-to-end from scratch

1. Clone the target repos into `repos/<name>/` (see `config.REPOS` for the
   expected branch per repo). Any path works — update `REPOS` in
   `src/td_prediction/config.py` if you put them elsewhere.
2. Re-mine features:
   ```bash
   python scripts/run_pipeline.py --stage mine --force
   ```
3. Build v2 LLM prompts, submit to OpenAI Batches:
   ```bash
   export OPENAI_API_KEY=sk-...
   python scripts/build_v2_batches.py --variant v2_rubric_json --strip-satd-comments
   python scripts/submit_batches.py --glob 'llm_batch/*_v2_rubric_json.jsonl' --wait
   ```
4. Parse labels + train + explain:
   ```bash
   python scripts/run_pipeline.py --stage label --variant v2_rubric_json
   python scripts/run_pipeline.py --stage train --lopo
   python scripts/run_pipeline.py --stage xai --model lgbm
   ```

## Methodological choices (workshop paper)

| Issue | Fix |
|---|---|
| Label leakage from `satd_delta` | dropped from feature set (see `config.LEAKAGE_COLS`) |
| Label leakage via SATD keywords in the LLM prompt | `strip_comments()` removes comment lines from the diff before LLM review |
| Shortcut learning from maturity features | ablation via `FeatureSet.NO_MATURITY` and `FeatureSet.CHANGE_ONLY` |
| Threshold tuned on test | threshold tuned on a dedicated VAL split, frozen, reported on TEST |
| Class imbalance | sweep over `none / class_weight / smote / smoteenn`; PR-AUC and MCC reported alongside F1 |
| Prompt engineering | versioned rubric + structured JSON output, optional few-shot from human gold set |
| XAI surface area | SHAP + LIME + permutation importance; top-K agreement table in `artifacts/results/xai_topk_*.csv` |
| No human gold truth | `human_review.sample_for_review()` produces a stratified CSV for manual labeling; see `HUMAN_TODOS.md` |
| Reproducibility | fixed seeds (`config.SEED=42`), pinned cutoff date, deterministic splits, `scripts/run_pipeline.py` |

## Human-in-the-loop checklist

See [`HUMAN_TODOS.md`](HUMAN_TODOS.md) for the current list of tasks that
require your input (gold-set labeling, rubric approval, operating-point
selection).
