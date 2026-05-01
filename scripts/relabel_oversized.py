"""
Remediate commits whose diffs exceeded the batch context limit (272k tokens).

The batch API rejects them silently — attach_llm_labels then defaults them
to label_llm=0, which is wrong (we don't actually know the label).

This script:
  1. Reads the *_errors.jsonl files in llm_batch/
  2. Fetches each affected commit's diff from GitHub
  3. Truncates to a safe size (200k chars ≈ 50k tokens)
  4. Calls the sync API with the v2.3 rubric
  5. Updates features_with_llm_labels.csv in place

Run from repo root after the batch pipeline:
    python scripts/relabel_oversized.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(override=True)

import openai
import pandas as pd
from td_prediction.config import LLM_MODEL, LLM_TEMPERATURE
from td_prediction.labeling.prompts import SYSTEM_PROMPT, RUBRIC

DIFFS_DIR    = Path("artifacts/diffs/")
FEATURES_CSV = Path("data/features_with_llm_labels.csv")
MAX_DIFF_CHARS = 200_000  # ~50k tokens, well below 272k limit


def find_failed_commits() -> list[str]:
    uids = []
    for f in Path("llm_batch").glob("*_errors.jsonl"):
        for line in f.open():
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            if "|" in cid:
                uids.append(cid.split("|")[0])
    return uids


def fetch_diff(commit_uid: str) -> str | None:
    repo, sha = commit_uid.split("#", 1)
    repo_url_map = {
        "requests": "https://github.com/psf/requests",
        "fastapi":  "https://github.com/fastapi/fastapi",
        "scrapy":   "https://github.com/scrapy/scrapy",
        "flask":    "https://github.com/pallets/flask",
        "keras":    "https://github.com/keras-team/keras",
    }
    diff_url = f"{repo_url_map[repo]}/commit/{sha}.diff"
    diff_path = DIFFS_DIR / f"{sha}.diff"
    if diff_path.exists():
        return diff_path.read_text()
    try:
        with urllib.request.urlopen(diff_url) as r:
            content = r.read().decode("utf-8")
        DIFFS_DIR.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(content)
        return content
    except Exception as e:
        print(f"  fetch failed: {e}")
        return None


def filter_py_diff(diff_text: str) -> str:
    lines = diff_text.splitlines(keepends=True)
    result, include = [], False
    for line in lines:
        if line.startswith("diff --git"):
            include = line.endswith(".py\n") or ".py " in line
        if include:
            result.append(line)
    return "".join(result)


def call_llm(metrics: str, diff: str) -> dict:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RUBRIC},
            {"role": "user",   "content": f"Metrics: {metrics}\n\nDIFF:\n{diff}"},
        ],
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=128,
    )
    return json.loads(resp.choices[0].message.content)


def main():
    failed = find_failed_commits()
    if not failed:
        print("No failed commits in batch error files. Nothing to do.")
        return

    print(f"Found {len(failed)} failed commit(s). Re-labeling with truncated diffs.")
    df = pd.read_csv(FEATURES_CSV)

    updates = []
    for uid in failed:
        print(f"\n[{uid}]")
        diff = fetch_diff(uid)
        if diff is None:
            continue
        py_diff = filter_py_diff(diff)
        if len(py_diff) > MAX_DIFF_CHARS:
            print(f"  diff size {len(py_diff):,} chars → truncating to {MAX_DIFF_CHARS:,}")
            py_diff = py_diff[:MAX_DIFF_CHARS] + "\n\n[... DIFF TRUNCATED DUE TO SIZE ...]"

        row = df[df["commit_uid"] == uid].iloc[0]
        metrics = (
            f"LoC+{row['lines_added']}/-{row['lines_deleted']} | "
            f"files={row['files_changed']} | hunks={row['hunks']}"
        )
        try:
            result = call_llm(metrics, py_diff)
            label = 1 if str(result.get("label", "")).strip().lower() == "yes" else 0
            print(f"  → label={label}  conf={result.get('confidence')}  rationale={result.get('rationale','')[:80]}")
            updates.append({
                "commit_uid": uid,
                "label_llm": label,
                "rationale_llm": result.get("rationale", ""),
                "llm_confidence": result.get("confidence"),
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    if not updates:
        print("\nNo successful re-labels. features CSV not modified.")
        return

    # Apply updates
    upd = pd.DataFrame(updates).set_index("commit_uid")
    for col in upd.columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.set_index("commit_uid")
    df.update(upd)
    df = df.reset_index()

    # label_consolidated may also need update — only where humans haven't overridden
    # For these 4 commits, no human consensus exists (they're not in the gold set)
    df["label_consolidated"] = df.apply(
        lambda r: r["label_human"] if pd.notna(r.get("label_human")) else r["label_llm"], axis=1
    ).astype(int)

    df.to_csv(FEATURES_CSV, index=False)
    print(f"\nUpdated {len(updates)} rows in {FEATURES_CSV}")


if __name__ == "__main__":
    main()
