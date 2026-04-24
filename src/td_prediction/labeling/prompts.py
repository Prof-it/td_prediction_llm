"""Versioned prompt templates for the LLM-as-judge.

Each template exposes the same interface:
    build(row: dict, diff_text: str, *, few_shot: list[dict] | None = None) -> str

where `row` is a CSV row with the commit's metrics, and `diff_text` is
the (optionally SATD-stripped) unified diff (only containing .py files). 
`few_shot` is an optional list of exemplar dicts 
with keys {diff, metrics, label, rationale}.

The rubric is the research-defensible definition of technical debt
used across all variants. If you change the rubric, bump PROMPT_VERSION.
"""
from __future__ import annotations

import json
from typing import Callable

PROMPT_VERSION = "v2.3"


RUBRIC = """Technical debt (TD) is any design or implementation shortcut in this
specific commit that trades short-term delivery for *higher future maintenance
cost* (Cunningham 1992). The focus is on *code debt* introduced by the change.
Label a commit as TD-introducing ('yes') when at least one of the following
is evident from the change itself:

1. Complexity shortcut: overly long methods, deeply nested conditionals,
   high cyclomatic-complexity delta, or God-class growth — anything that
   increases cognitive load on future maintainers.
2. Duplication: copy-pasted logic, code clones, or repeated fragments that
   should be abstracted — the canonical code duplication debt category.
3. Coupling / change scattering: a change that tightly couples unrelated
   modules, introduces cyclic dependencies, or scatters a single concern
   across many files (Shotgun Surgery pattern).
4. Brittle or unclear code: hard-coded values, magic numbers, unclear
   naming, dead code, or commented-out production code left in place.
5. Missing safeguards: removed or weakened tests, silenced exceptions,
   disabled checks, or explicit deferrals ("temporary workaround").

Do NOT label as TD:
- Routine refactors that *reduce* complexity or remove duplication.
- Mechanical changes (formatting, import reordering, dependency bumps).
- Feature additions that are self-contained, well-scoped, and low-coupling.
- Test-only additions with no production code change.
- Commits whose only signal is a TODO/FIXME/HACK keyword — label the code
  quality, not the comment. Keyword-based debt is captured separately by the
  SATD regex baseline and must not drive this label.

Respond ONLY with a JSON object of the shape:
{"label": "yes" | "no", "confidence": 0.0-1.0, "rationale": "<= 25 words"}
No prose outside the JSON."""


SYSTEM_PROMPT = (
    "You are a senior software engineer performing code review on behalf of "
    "a research study. Apply the rubric strictly and return JSON only."
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
