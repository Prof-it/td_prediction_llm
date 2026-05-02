"""
End-to-end orchestrator: builds every artifact needed for the paper.

Runs (in order):
  1. llm_autoreview.py            — gold-set LLM labels (skippable; cached)
  2. kappa_analysis.py            — κ + consolidated_gt + disagreement_slice
  3. patch_labels.py              — apply gold + consolidated GT to features CSV
  4. run_pipeline.py --stage train (time split, label_consolidated)
  5. run_pipeline.py --stage train --lopo
  6. run_pipeline.py --stage xai
  7. audit_analysis.py            — stratification + low-confidence slice
  8. baselines_and_tests.py       — SATD baseline + multi-seed + McNemar
  9. paper_report.py              — single aggregated paper_report.{md,json}

Steps 1, 4–6, 8 are expensive; 2–3, 7, 9 are fast aggregations.

Usage:
    python scripts/build_all_paper_artifacts.py            # full run
    python scripts/build_all_paper_artifacts.py --skip-autoreview --skip-train
    python scripts/build_all_paper_artifacts.py --skip 1,4,5,6,8   # only fast steps

Each step lists its outputs; if all expected outputs already exist, the step
is logged as "(already exists)" and skipped — making the script idempotent.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


# Each step: (id, label, command, expected_output_paths)
STEPS = [
    (1, "LLM autoreview (gold set re-label)",
     ["python", "scripts/llm_autoreview.py"],
     ["artifacts/results/human_review_sheet_llm.csv"]),
    (2, "Kappa analysis + consolidated GT",
     ["python", "scripts/kappa_analysis.py"],
     ["artifacts/results/consolidated_gt.csv",
      "artifacts/results/disagreement_slice.csv"]),
    (3, "Patch features CSV with gold labels",
     ["python", "scripts/patch_labels.py"],
     ["data/features_with_llm_labels.csv"]),
    (4, "Train (time split, label_consolidated)",
     ["python", "scripts/run_pipeline.py", "--stage", "train",
      "--label-col", "label_consolidated"],
     ["artifacts/results/metrics_time.csv"]),
    (5, "Train (LOPO splits)",
     ["python", "scripts/run_pipeline.py", "--stage", "train", "--lopo",
      "--label-col", "label_consolidated"],
     ["artifacts/results/metrics_lopo.csv"]),
    (6, "XAI (SHAP / LIME / Permutation)",
     ["python", "scripts/run_pipeline.py", "--stage", "xai", "--model", "lgbm",
      "--label-col", "label_consolidated"],
     ["artifacts/results/xai_topk_lgbm.csv"]),
    (7, "Audit analysis (stratification + low-confidence)",
     ["python", "scripts/audit_analysis.py"],
     ["artifacts/results/audit_stratification.csv",
      "artifacts/results/audit_low_confidence.csv"]),
    (8, "Baselines + multi-seed + McNemar",
     ["python", "scripts/baselines_and_tests.py", "--seeds", "5"],
     ["artifacts/results/baseline_metrics.csv",
      "artifacts/results/multiseed_metrics.csv",
      "artifacts/results/mcnemar_test.txt"]),
    (9, "Paper report",
     ["python", "scripts/paper_report.py"],
     ["artifacts/results/paper_report.md",
      "artifacts/results/paper_report.json"]),
]


def run_step(step_id, label, cmd, outputs, skip_if_cached=True, force=False):
    print(f"\n{'='*72}")
    print(f"  STEP {step_id}: {label}")
    print(f"{'='*72}")

    paths = [REPO_ROOT / o for o in outputs]
    if not force and skip_if_cached and all(p.exists() for p in paths):
        print(f"  (already exists, skipping)")
        for p in paths:
            print(f"    {p.relative_to(REPO_ROOT)}")
        return True

    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}, {dt:.1f}s)")
        return False
    print(f"  OK ({dt:.1f}s)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", default="",
                   help="Comma-separated step IDs to skip (e.g. '1,4,5,6,8' to skip the slow steps).")
    p.add_argument("--skip-autoreview", action="store_true", help="Alias for --skip 1.")
    p.add_argument("--skip-train", action="store_true", help="Alias for --skip 4,5,6.")
    p.add_argument("--force", action="store_true",
                   help="Re-run every step even if outputs already exist.")
    args = p.parse_args()

    skipped = set()
    if args.skip:
        skipped.update(int(x) for x in args.skip.split(","))
    if args.skip_autoreview:
        skipped.add(1)
    if args.skip_train:
        skipped.update({4, 5, 6})

    results = {}
    for step_id, label, cmd, outputs in STEPS:
        if step_id in skipped:
            print(f"\n[skip] STEP {step_id}: {label}")
            results[step_id] = "skipped"
            continue
        ok = run_step(step_id, label, cmd, outputs, force=args.force)
        results[step_id] = "ok" if ok else "FAILED"
        if not ok:
            print("\nAborting — fix the failing step and rerun.")
            sys.exit(1)

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    for step_id, label, _, _ in STEPS:
        marker = {"ok": "  OK ", "FAILED": "FAIL ", "skipped": "skip "}[results[step_id]]
        print(f"  [{marker}] {step_id}: {label}")

    print(f"\n  Final report: artifacts/results/paper_report.md")


if __name__ == "__main__":
    main()
