import os
import sys
from pathlib import Path

# allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
import urllib.request
import csv
import json
import openai
from openai.types.chat import ChatCompletionMessageParam
from td_prediction.config import LLM_MODEL, LLM_MODEL_DATE, LLM_TEMPERATURE
from td_prediction.labeling.prompts import SYSTEM_PROMPT, RUBRIC
from td_prediction.labeling.satd import strip_comments

load_dotenv()

# --- SETTINGS ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL   = LLM_MODEL          # pinned in config.py — do not override via env
INPUT_CSV  = Path("artifacts/results/human_review_sheet_a_20260424_021700.csv")
OUTPUT_CSV = Path("artifacts/results/human_review_sheet_llm.csv")
DIFFS_DIR  = Path("artifacts/diffs/")

# --- LLM CALL ---
def call_llm(metrics_row, diff_text):
    prompt = SYSTEM_PROMPT + "\n\n" + RUBRIC
    user_content = f"Metrics: {metrics_row}\n\nDIFF:\n{diff_text}"
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=128,
    )
    return response.choices[0].message.content

# --- MAIN ---

def fetch_diff_from_commit_url(commit_url, diff_path):
    """
    Attempt to download the .diff for a commit from GitHub using the commit_url and save it locally.
    commit_url: e.g., https://github.com/psf/requests/commit/8fbb1e6d97cda90d588d4263a18906a52d147fba
    """
    if commit_url.endswith('/'):
        commit_url = commit_url[:-1]
    # Replace /commit/{hash} with /commit/{hash}.diff
    if commit_url.endswith('.diff'):
        diff_url = commit_url
    else:
        diff_url = commit_url + '.diff'
    try:
        print(f"Fetching diff from {diff_url}")
        with urllib.request.urlopen(diff_url) as response:
            diff_content = response.read().decode('utf-8')
        with open(diff_path, 'w') as f:
            f.write(diff_content)
        print(f"Saved diff to {diff_path}")
        return True
    except Exception as e:
        print(f"Failed to fetch diff from {diff_url}: {e}")
        return False

def filter_py_diff(diff_text: str) -> str:
    """Keep only hunks belonging to .py files."""
    lines = diff_text.splitlines(keepends=True)
    result, include = [], False
    for line in lines:
        if line.startswith("diff --git"):
            include = line.endswith(".py\n") or ".py " in line
        if include:
            result.append(line)
    return "".join(result)


def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open() as fin:
        reader = csv.DictReader(fin, delimiter=';')
        base_fields = [f for f in reader.fieldnames if f not in ("label_llm", "rationale_llm", "confidence_llm")]
        fieldnames = base_fields + ["label_llm", "confidence_llm", "rationale_llm"]
        rows = list(reader)

    with OUTPUT_CSV.open("w", newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            commit_hash = row["commit_hash"]
            diff_path = DIFFS_DIR / f"{commit_hash}.diff"
            if not diff_path.exists():
                commit_url = row.get("commit_url")
                if commit_url:
                    fetched = fetch_diff_from_commit_url(commit_url, diff_path)
                    if not fetched:
                        print(f"[{i}/{len(rows)}] SKIP {commit_hash}: diff unavailable")
                        continue
                else:
                    print(f"[{i}/{len(rows)}] SKIP {commit_hash}: no commit_url")
                    continue
            with diff_path.open() as df:
                diff_text = strip_comments(filter_py_diff(df.read()))
            metrics_summary = f"LoC+{row['lines_added']}/-{row['lines_deleted']} | files={row['files_changed']} | hunks={row['hunks']}"
            try:
                llm_response = call_llm(metrics_summary, diff_text)
                llm_json = json.loads(llm_response)
                row["label_llm"] = llm_json.get("label", "")
                row["confidence_llm"] = llm_json.get("confidence", "")
                row["rationale_llm"] = llm_json.get("rationale", "")
                print(f"[{i}/{len(rows)}] {commit_hash[:8]} → {row['label_llm']} ({row['confidence_llm']})")
            except Exception as e:
                print(f"[{i}/{len(rows)}] ERROR {commit_hash}: {e}")
                row["label_llm"] = "error"
                row["confidence_llm"] = ""
                row["rationale_llm"] = str(e)
            writer.writerow(row)

    print(f"\nDone. Model={OPENAI_MODEL} (pinned {LLM_MODEL_DATE}). Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
