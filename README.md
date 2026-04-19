# td_prediction_llm

This repository contains source code for predicting technical debt introduced per commit. The pipeline mines repository history, labels commits via LLM-as-judge, trains classifiers (Random Forest, LightGBM, XGBoost), and explains predictions using SHAP, LIME, and permutation importance.

![Workflow](artifacts/figures/workflowdiagram.drawio.svg)

## Quick start

```bash
make install       # Install dependencies
make reproduce     # Run full pipeline (uses cached data)
```

Outputs go to `artifacts/`.

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
        metrics.py         # evaluation metrics (PR-AUC, MCC, F1, etc.)
        threshold.py       # threshold tuning on validation set
        plots.py           # PR, ROC, class-balance plots
    xai/
        shap_explainer.py  # SHAP explanations
        lime_explainer.py  # LIME explanations
        permutation.py     # permutation importance

notebooks/                 # Thin orchestrator notebooks (Marimo .py files)
    01_mine.py
    02_label.py
    03_train_eval.py
    04_xai.py
    05_analysis.py

scripts/                   # CLI entry points
    run_pipeline.py        # mine | label | train | xai | analysis
    build_v2_batches.py    # build LLM batch JSONL files
    submit_batches.py      # submit + poll OpenAI batches

tests/                     # unit and integration tests
artifacts/
    figures/               # plots (PR curves, SHAP, class balance, etc.)
    results/               # metric CSVs, XAI rankings
    models/                # trained model bundles

data/                      # features_*.csv (mined per commit)
llm_batch/                 # LLM batch JSONL (input + output)
splits/                    # cached train/val/test splits (time-based + LOPO)
```

## Reproducing from cached data

The repo includes cached features and LLM outputs, so training and XAI are reproducible without re-mining repositories or calling external APIs.

```bash
# One-liner
make reproduce

# Or step-by-step
python scripts/run_pipeline.py --stage label
python scripts/run_pipeline.py --stage train --lopo
python scripts/run_pipeline.py --stage xai --model lgbm
python scripts/run_pipeline.py --stage analysis
```

## Running end-to-end from scratch

1. Clone target repositories into `repos/<name>/` (see `config.REPOS`).
2. Re-mine features:
   ```bash
   python scripts/run_pipeline.py --stage mine --force
   ```
3. Build and submit LLM-as-judge batches:
   ```bash
   export OPENAI_API_KEY=sk-...
   python scripts/build_v2_batches.py --variant v2_rubric_json --strip-satd-comments
   python scripts/submit_batches.py --glob 'llm_batch/*_v2_rubric_json.jsonl' --wait
   ```
4. Train and explain:
   ```bash
   python scripts/run_pipeline.py --stage label --variant v2_rubric_json
   python scripts/run_pipeline.py --stage train --lopo
   python scripts/run_pipeline.py --stage xai --model lgbm
   ```
