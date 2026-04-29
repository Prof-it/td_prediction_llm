"""02 — LLM-as-judge labeling.

This notebook prepares and submits LLM batch jobs, then parses their
outputs into `label_llm`. It supports:

- **Cached mode (default):** reads existing `llm_batch/*_output.jsonl`
  files (from the thesis iterations) and produces `label_llm`.
- **Re-labeling mode:** builds v2 batch files (rubric + structured JSON
  + optional few-shot), submits them, polls, and parses.

Toggle `SUBMIT_NEW_BATCHES` to switch modes. Token cost is non-trivial
(~5 x commits x prompt_tokens), so submission is opt-in.
"""
import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 — LLM-as-Judge Labeling

    Default is **cached**: uses `llm_batch/*_output.jsonl` from the thesis
    iterations. Flip the flags at the top of the next cell to generate or
    submit v2 prompts.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    return (ROOT,)


@app.cell
def _():
    from td_prediction import config, mining
    from td_prediction.labeling import llm_judge
    return config, llm_judge, mining


@app.cell
def _():
    # Flags
    BUILD_V2_BATCHES = False     # Writes fresh JSONL files with v2 prompts.
    SUBMIT_NEW_BATCHES = False   # Calls the OpenAI API (costs tokens).
    POLL_UNTIL_DONE = False      # Waits for submitted batches to finish.
    PARSE_VARIANT = "v1_satd_filtered"  # which variant to use for label_llm
    return (
        BUILD_V2_BATCHES,
        POLL_UNTIL_DONE,
        PARSE_VARIANT,
        SUBMIT_NEW_BATCHES,
    )


@app.cell
def _(BUILD_V2_BATCHES, config, llm_judge, mining):
    # Build v2 batch files if requested. Does not call the API.
    if BUILD_V2_BATCHES:
        import json
        from td_prediction.labeling.llm_judge import BatchPlan, build_batch_file
        df = mining.load_all_features()

        # Per-commit diff text is NOT stored in features_*.csv — you must
        # either (a) re-mine with diff preserved, or (b) reuse the diffs
        # embedded in the existing v1 batch INPUTS. We reuse (b) here.
        diffs = {}
        for p in sorted(config.PATHS.llm_batch.glob("*_original_prompt.jsonl")):
            with p.open() as fh:
                for line in fh:
                    obj = json.loads(line)
                    cid = obj["custom_id"].replace("-original-prompt", "")
                    for msg in obj["body"]["messages"]:
                        if "DIFF:" in msg["content"]:
                            diffs[cid] = msg["content"].split("DIFF:", 1)[1].strip()
                            break
        print(f"Loaded diffs for {len(diffs)} commits")

        for repo in df["repo_id"].unique():
            repo_rows = df[df["repo_id"] == repo]
            plan = BatchPlan(
                variant="v2_rubric_json",
                rows=repo_rows,
                diffs=diffs,
                few_shot=None,   # Populate after gold set is ready.
                n_consistency=1,
            )
            out = config.PATHS.llm_batch / f"{repo}_v2_rubric_json.jsonl"
            build_batch_file(plan, out, strip_satd_comments=True)
            print(f"Wrote {out}")
    return


@app.cell
def _(SUBMIT_NEW_BATCHES, config, llm_judge):
    if SUBMIT_NEW_BATCHES:
        files = sorted(config.PATHS.llm_batch.glob("*_v2_*.jsonl"))
        if not files:
            print("No v2 files to submit — set BUILD_V2_BATCHES=True first.")
        else:
            meta = llm_judge.submit_batches(files, description="TD v2 rubric+json")
            print(f"Submitted {len(meta)} batches.")
    return


@app.cell
def _(POLL_UNTIL_DONE, llm_judge):
    if POLL_UNTIL_DONE:
        status = llm_judge.poll_batches(wait=True, interval=60)
        for name, meta in status.items():
            print(f"{name}: {meta.get('status')}")
    return


@app.cell
def _(PARSE_VARIANT, config, llm_judge, mining):
    parsed = llm_judge.parse_results(variant_filter=PARSE_VARIANT)
    print(f"Parsed {len(parsed)} responses for variant {PARSE_VARIANT}.")
    df_feat = mining.load_all_features()
    df_labeled = llm_judge.attach_llm_labels(df_feat, variant=PARSE_VARIANT, parsed=parsed)
    out = config.PATHS.data / "features_with_llm_labels.csv"
    df_labeled.to_csv(out, index=False)
    print(f"Wrote {out} — positive rate: {df_labeled['label_llm'].mean():.3f}")
    return (df_labeled,)


if __name__ == "__main__":
    app.run()
