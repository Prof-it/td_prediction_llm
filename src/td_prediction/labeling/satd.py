"""SATD (Self-Admitted Technical Debt) regex labeling.

Counts SATD-keyword comments added minus removed in a commit's diff.
Produces an integer delta and a binary label `label_td_satd = 1 iff delta > 0`.

Weaknesses of this labeler (documented for the paper):
- Only fires on literal keywords, misses paraphrased debt.
- False positives: "TODO: support Py3.12" is not necessarily debt.
- Delta-based: a commit that adds TODO *and* removes TODO could net to 0.
- Comment detection is line-local and naive (no AST), which is why the
  LLM-as-judge step is primarily used.
"""
from __future__ import annotations

from .. import config


def is_comment_or_docstring(line: str) -> bool:
    s = line.strip()
    return (
        s.startswith("#")
        or s.startswith('"""') or s.startswith("'''")
        or s.endswith('"""') or s.endswith("'''")
    )


def satd_delta(mod) -> int:
    """Net SATD-keyword comments added in a PyDriller modification."""
    add = rem = 0
    for _, line in mod.diff_parsed["added"]:
        if is_comment_or_docstring(line) and config.SATD_PATTERN.search(line):
            add += 1
    for _, line in mod.diff_parsed["deleted"]:
        if is_comment_or_docstring(line) and config.SATD_PATTERN.search(line):
            rem += 1
    return add - rem


def is_py(m) -> bool:
    fp = m.new_path or m.old_path or ""
    return fp.endswith(".py")


def strip_comments(diff_text: str) -> str:
    """Remove comment lines from a unified diff.

    Used to build SATD-stripped prompts for the LLM-as-judge so the model
    cannot trivially shortcut to the regex label.
    """
    kept = []
    for line in diff_text.split("\n"):
        # Keep the leading +/- marker, check what follows.
        marker, rest = line[:1], line[1:]
        if marker in {"+", "-"} and is_comment_or_docstring(rest):
            continue
        kept.append(line)
    return "\n".join(kept)
