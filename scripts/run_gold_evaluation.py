"""
Orchestrates the full gold-set evaluation pipeline (prof's workflow):

  1. llm_autoreview.py   — re-label 100 gold commits with LLM v2.3
  2. kappa_analysis.py   — Cohen's κ + consolidated_gt.csv + disagreement_slice.csv
  3. patch_labels.py     — update features_with_llm_labels.csv with fresh labels
  4. run_pipeline.py     — re-train ML models on updated labels

Each step can also be run independently:
    python scripts/llm_autoreview.py
    python scripts/kappa_analysis.py
    python scripts/patch_labels.py
    python scripts/run_pipeline.py --stage train

Run full pipeline from repo root:
    python scripts/run_gold_evaluation.py
    python scripts/run_gold_evaluation.py --skip-autoreview   # reuse existing LLM sheet
    python scripts/run_gold_evaluation.py --skip-train        # skip retraining
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-autoreview", action="store_true",
                   help="Skip LLM re-labeling (reuse existing human_review_sheet_llm.csv)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip ML re-training")
    args = p.parse_args()

    if not args.skip_autoreview:
        print("\n=== STEP 1: LLM auto-review ===")
        import llm_autoreview
        llm_autoreview.main()
    else:
        print("\n[skip] Step 1: autoreview")

    print("\n=== STEP 2: Kappa analysis + consolidate GT ===")
    import kappa_analysis
    kappas = kappa_analysis.main()

    print("\n=== STEP 3: Patch features CSV ===")
    import patch_labels
    patch_labels.main()

    if not args.skip_train:
        print("\n=== STEP 4: Re-train ML models ===")
        import types
        import run_pipeline
        run_pipeline.stage_train(types.SimpleNamespace(
            force=False, variant="v2_rubric_json", model="lgbm", lopo=False,
        ))
    else:
        print("\n[skip] Step 4: training")

    print("\n=== Summary ===")
    for name, k in kappas.items():
        quality = "good" if k >= 0.6 else "moderate" if k >= 0.4 else "poor"
        print(f"  {name:<35} κ={k:.4f}  ({quality})")

    metrics = Path("artifacts/results/metrics_time.csv")
    if metrics.exists():
        import pandas as pd
        best = pd.read_csv(metrics).sort_values("f1_debt", ascending=False).iloc[0]
        print(f"\n  Best model: {best['model']} / {best['feature_set']} / {best['imbalance']}")
        print(f"  F1={best['f1_debt']:.4f}  AUC={best['roc_auc']:.4f}  "
              f"P={best['p_debt']:.4f}  R={best['r_debt']:.4f}  MCC={best['mcc']:.4f}")


if __name__ == "__main__":
    main()
