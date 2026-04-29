"""Versioned prompt templates for the LLM-as-judge.

Each template exposes the same interface:
    build(row: dict, diff_text: str, *, few_shot: list[dict] | None = None) -> str

where `row` is a CSV row with the commit's metrics, and `diff_text` is
the (optionally SATD-stripped) unified diff. `few_shot` is an optional
list of exemplar dicts with keys {diff, metrics, label, rationale}.

The rubric is the research-defensible definition of technical debt
used across all variants. If you change the rubric, bump PROMPT_VERSION.
"""
from __future__ import annotations

import json
from typing import Callable

PROMPT_VERSION = "v2.2"



RUBRIC = """Technical debt (TD) is any design or implementation shortcut in this
specific commit that trades short-term delivery for higher future maintenance cost
(Cunningham 1992). The assessment is STRICTLY limited to the provided Python (.py)
diff only.

EPISTEMIC CONSTRAINTS:
- Assume ONLY the shown .py diff is available.
- Do NOT infer repository structure, unseen files, tests, or architecture.
- Do NOT speculate about missing tests or dependencies unless explicitly shown.
- If evidence is not directly visible in the diff, treat it as UNKNOWN (not TD).

Label a commit as TD-introducing ("yes") ONLY if at least one of the following
is clearly and directly observable in the diff:

1. Local complexity increase:
    - visibly longer methods/functions OR
    - deeply nested conditionals OR
    - multiple responsibilities added into one function

2. Explicit duplication:
    - repeated or copy-pasted code blocks within the diff
    - near-identical logic with minor variations that should be abstracted

3. Local coupling within diff:
    - same concern spread across multiple modified Python files OR
    - new direct dependency between modules visible in the diff
    (Do NOT assume global architecture or cyclic dependencies)

4. Concrete code quality issues:
    - hard-coded values / magic numbers without explanation
    - commented-out production code left in place
    - clearly unused variables or dead code within the diff

5. Removed or weakened safeguards (ONLY if explicitly visible):
    - deletion of tests, assertions, validation, or error handling
    - silenced exceptions (e.g., bare except, pass)
    - explicit "temporary workaround" in code/comments

6. Self-admitted technical debt:
    - comments or messages explicitly stating the solution is temporary,
      incomplete, or suboptimal

Do NOT label as TD:
- Changes that reduce complexity or remove duplication
- Mechanical or formatting-only changes
- Self-contained feature additions with no visible shortcuts
- Absence of tests or safeguards UNLESS their removal is explicitly shown
- Any claim that relies on unseen files or assumed architecture

DECISION RULE:
- Default to "no" unless there is clear, direct evidence of TD in the diff.
- When uncertain or evidence is weak → label "no".

Respond ONLY with:
{"label": "yes" | "no", "confidence": 0.0-1.0, "rationale": "<= 25 words"}
"""


SYSTEM_PROMPT = (
    "You are a senior software engineer performing strict diff-based code review. "
    "Base your judgment ONLY on visible Python diff evidence. "
    "Do not infer or speculate. When unsure, answer 'no'. "
    "Return valid JSON only."
)


def _summary_line(row: dict) -> str:
    return " | ".join([
        f"LoC+{row['lines_added']}/-{row['lines_deleted']}",
        f"files={row['files_changed']}",
        f"hunks={row['hunks']}",
        f"methods_changed={row['n_methods_changed']}",
        f"ccΔ_sum/max={row['cc_delta_sum']}/{row['cc_delta_max']}",
        f"churnΔ={row['churn_delta']}",
        f"dmm_cx={row['dmm_unit_complexity']}",
        f"dmm_size={row['dmm_unit_size']}",
        f"dmm_if={row['dmm_unit_interfacing']}",
    ])


# --- Variants -------------------------------------------------------------

def v2_rubric_json(row: dict, diff_text: str, *, few_shot=None) -> list[dict]:
    """Rubric + structured JSON output. Primary variant for iteration 4."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RUBRIC}]
    if few_shot:
        for ex in few_shot:
            messages.append({
                "role": "user",
                "content": (
                    f"Metrics: {_summary_line(ex['metrics'])}\n\n"
                    f"DIFF:\n{ex['diff']}"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "label": ex["label"],
                    "confidence": ex.get("confidence", 0.9),
                    "rationale": ex["rationale"],
                }),
            })
    messages.append({
        "role": "user",
        "content": f"Metrics: {_summary_line(row)}\n\nDIFF:\n{diff_text}",
    })
    return messages


def v2_rubric_no_diff(row: dict, diff_text: str, *, few_shot=None) -> list[dict]:
    """Ablation: metrics only, no diff. Tests how much the diff adds."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RUBRIC}]
    messages.append({
        "role": "user",
        "content": f"Metrics: {_summary_line(row)}\n\n(No diff provided.)",
    })
    return messages


def v2_rubric_diff_only(row: dict, diff_text: str, *, few_shot=None) -> list[dict]:
    """Ablation: diff only, no metrics. Tests whether metrics bias the judge."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + RUBRIC}]
    messages.append({"role": "user", "content": f"DIFF:\n{diff_text}"})
    return messages


# --- Legacy (v1) — retained so existing llm_batch/ outputs can still be read ---

def v1_original(row: dict, diff_text: str, *, few_shot=None) -> list[dict]:
    prompt = (
        "You are a senior reviewer.\n\n"
        f"Commit Summary:\n{_summary_line(row)}\n\n"
        f"DIFF:\n{diff_text}\n\n"
        "Question: Does this commit introduce technical debt? Answer yes or no."
    )
    return [{"role": "user", "content": prompt}]


def v1_satd_filtered(row: dict, diff_text: str, *, few_shot=None) -> list[dict]:
    from .. import config
    filtered = config.SATD_PATTERN.sub("", diff_text)
    prompt = (
        "You are a senior code reviewer. Based on the code change and metrics "
        "summary below, assess if this change might lead to long-term "
        "maintainability issues. Answer with 'yes' or 'no'.\n\n"
        f"Commit Summary:\n{_summary_line(row)}\n\n"
        f"DIFF:\n{filtered}"
    )
    return [{"role": "user", "content": prompt}]


def v1_diff_removed(row: dict, diff_text: str, *, few_shot=None) -> list[dict]:
    prompt = (
        "You are a senior code reviewer. Based on the code change and metrics "
        "summary below, assess if this change might lead to long-term "
        "maintainability issues. Answer with 'yes' or 'no'.\n\n"
        f"Commit Summary:\n{_summary_line(row)}"
    )
    return [{"role": "user", "content": prompt}]


VARIANTS: dict[str, Callable[..., list[dict]]] = {
    # v2 — used for the workshop-paper iteration.
    "v2_rubric_json":     v2_rubric_json,
    "v2_rubric_no_diff":  v2_rubric_no_diff,
    "v2_rubric_diff_only": v2_rubric_diff_only,
    # v1 — kept for parsing cached outputs from the thesis iterations.
    "v1_original":        v1_original,
    "v1_satd_filtered":   v1_satd_filtered,
    "v1_diff_removed":    v1_diff_removed,
}


def max_tokens_for(variant: str) -> int:
    """v1 variants ask for a single yes/no token; v2 asks for a short JSON."""
    return 1 if variant.startswith("v1_") else 80
