"""Build v2 LLM batch JSONL files (rubric + structured JSON, optional few-shot).

This does NOT call the OpenAI API. It produces files in `llm_batch/` that
you can submit with `scripts/submit_batches.py --glob 'llm_batch/*_v2_*.jsonl'`.

Usage:
    python scripts/build_v2_batches.py \
        --variant v2_rubric_json \
        --strip-satd-comments \
        --few-shot artifacts/results/human_review_sheet.csv \
        --n-consistency 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_diffs_from_v1_inputs(batch_dir: Path) -> dict[str, str]:
    """Recover per-commit diffs from the v1 original-prompt batch inputs."""
    diffs: dict[str, str] = {}
    for p in sorted(batch_dir.glob("*_original_prompt.jsonl")):
        with p.open() as fh:
            for line in fh:
                obj = json.loads(line)
                cid = obj["custom_id"].replace("-original-prompt", "")
                for msg in obj["body"]["messages"]:
                    if "DIFF:" in msg["content"]:
                        diffs[cid] = msg["content"].split("DIFF:", 1)[1].strip()
                        break
    return diffs


def _few_shot_from_sheet(sheet_path: Path, diffs: dict[str, str], *, k: int = 4) -> list[dict]:
    """Build few-shot exemplars from a filled human-review sheet.

    Expects columns: commit_uid, label_human (0/1), rationale_human.
    Picks `k` high-confidence exemplars balanced across classes.
    """
    import pandas as pd
    df = pd.read_csv(sheet_path)
    df = df[df["label_human"].astype(str).str.strip().isin(["0", "1"])]
    if df.empty:
        print(f"[warn] no filled human labels in {sheet_path}; skipping few-shot.")
        return []
    df["label_human"] = df["label_human"].astype(int)
    picked = []
    for label in (1, 0):
        pool = df[df["label_human"] == label]
        picked.append(pool.head(k // 2))
    picked = pd.concat(picked, ignore_index=True)
    exemplars = []
    for _, r in picked.iterrows():
        diff = diffs.get(r["commit_uid"], "")
        exemplars.append({
            "diff": diff[:2000],
            "metrics": {k: r.get(k, 0) for k in [
                "lines_added", "lines_deleted", "files_changed", "hunks",
                "n_methods_changed", "cc_delta_sum", "cc_delta_max",
                "churn_delta", "dmm_unit_complexity", "dmm_unit_size",
                "dmm_unit_interfacing",
            ]},
            "label": "yes" if r["label_human"] == 1 else "no",
            "rationale": str(r.get("rationale_human", ""))[:200] or (
                "Clear structural shortcut / debt marker." if r["label_human"] == 1 else "Routine, well-scoped change."
            ),
            "confidence": 0.95,
        })
    return exemplars


def main() -> None:
    _ensure_src_on_path()
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v2_rubric_json",
                   choices=["v2_rubric_json", "v2_rubric_no_diff", "v2_rubric_diff_only"])
    p.add_argument("--strip-satd-comments", action="store_true",
                   help="Strip SATD-keyword comment lines from the diff sent to the LLM.")
    p.add_argument("--few-shot", type=Path,
                   help="Human-review sheet to build few-shot exemplars from.")
    p.add_argument("--n-consistency", type=int, default=1,
                   help="Number of stochastic samples per commit for self-consistency.")
    p.add_argument("--repos", nargs="*", help="Subset of repos; default = all.")
    args = p.parse_args()

    from td_prediction import config, mining
    from td_prediction.labeling.llm_judge import BatchPlan, build_batch_file

    df = mining.load_all_features(args.repos)
    diffs = _load_diffs_from_v1_inputs(config.PATHS.llm_batch)
    print(f"Loaded diffs for {len(diffs)} commits")

    few_shot = _few_shot_from_sheet(args.few_shot, diffs) if args.few_shot else None
    if few_shot:
        print(f"Using {len(few_shot)} few-shot exemplars.")

    repos = args.repos or sorted(df["repo_id"].unique())
    for repo in repos:
        repo_rows = df[df["repo_id"] == repo]
        plan = BatchPlan(
            variant=args.variant,
            rows=repo_rows,
            diffs=diffs,
            few_shot=few_shot,
            n_consistency=args.n_consistency,
        )
        out = config.PATHS.llm_batch / f"{repo}_{args.variant}.jsonl"
        build_batch_file(plan, out, strip_satd_comments=args.strip_satd_comments)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
