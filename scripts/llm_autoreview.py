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

load_dotenv()
# --- CONFIG ---
SYSTEM_PROMPT = (
    "You are a senior software engineer performing strict diff-based code review. "
    "Base your judgment ONLY on visible Python diff evidence. "
    "Do not infer or speculate. When unsure, answer 'no'. "
    "Return valid JSON only."
)

RUBRIC = """Technical debt (TD) is any design or implementation shortcut in this\n
specific commit that trades short-term delivery for higher future maintenance cost\n
(Cunningham 1992). The assessment is STRICTLY limited to the provided Python (.py)\n
diff only.\n\n

EPISTEMIC CONSTRAINTS:\n
- Assume ONLY the shown .py diff is available.\n
- Do NOT infer repository structure, unseen files, tests, or architecture.\n
- Do NOT speculate about missing tests or dependencies unless explicitly shown.\n
- If evidence is not directly visible in the diff, treat it as UNKNOWN (not TD).\n\n

Label a commit as TD-introducing (\"yes\") ONLY if at least one of the following\n
is clearly and directly observable in the diff:\n\n1. Local complexity increase:\n   
- visibly longer methods/functions OR\n   
- deeply nested conditionals OR\n   
- multiple responsibilities added into one function\n\n

2. Explicit duplication:\n   
- repeated or copy-pasted code blocks within the diff\n   
- near-identical logic with minor variations that should be abstracted\n\n

3. Local coupling within diff:\n   
- same concern spread across multiple modified Python files OR\n   
- new direct dependency between modules visible in the diff\n   
(Do NOT assume global architecture or cyclic dependencies)\n\n

4. Concrete code quality issues:\n   
- hard-coded values / magic numbers without explanation\n   
- commented-out production code left in place\n   
- clearly unused variables or dead code within the diff\n\n

5. Removed or weakened safeguards (ONLY if explicitly visible):\n   
- deletion of tests, assertions, validation, or error handling\n   
- silenced exceptions (e.g., bare except, pass)\n   
- explicit \"temporary workaround\" in code/comments\n\n

6. Self-admitted technical debt:\n   
- comments or messages explicitly stating the solution is temporary,\n     
incomplete, or suboptimal\n\n

Do NOT label as TD:\n
- Changes that reduce complexity or remove duplication\n
- Mechanical or formatting-only changes\n
- Self-contained feature additions with no visible shortcuts\n
- Absence of tests or safeguards UNLESS their removal is explicitly shown\n
- Any claim that relies on unseen files or assumed architecture\n\n

DECISION RULE:\n
- Default to \"no\" unless there is clear, direct evidence of TD in the diff.\n
- When uncertain or evidence is weak → label \"no\".\n\n

Respond ONLY with:\n
{"label": "yes" | "no", "confidence": 0.0-1.0, "rationale": "<= 25 words"}\n
"""

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
                diff_text = filter_py_diff(df.read())
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
