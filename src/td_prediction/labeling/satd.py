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
    """Remove SATD-keyword comments from a unified diff.

    Goal: the LLM-as-judge should not shortcut to the SATD-regex label
    via literal TODO/FIXME/etc. keywords. Strategy: remove only comments
    (full-line or inline) that contain an SATD keyword. Non-SATD comments
    are preserved so the diff remains readable code.

    This keeps string literals with '#' (URLs, etc.) intact, because we
    only strip when a `#` begins a comment AND that comment contains an
    SATD keyword.
    """
    kept = []
    for line in diff_text.split("\n"):
        marker, rest = line[:1], line[1:]
        if marker in {"+", "-"}:
            # Full-line comment with SATD keyword → drop entirely
            if is_comment_or_docstring(rest) and config.SATD_PATTERN.search(rest):
                continue
            # Inline comment with SATD keyword → strip comment, keep code
            hash_idx = _find_comment_hash(rest)
            if hash_idx is not None and config.SATD_PATTERN.search(rest[hash_idx:]):
                rest = rest[:hash_idx].rstrip()
            kept.append(marker + rest)
        else:
            kept.append(line)
    return "\n".join(kept)


def _find_comment_hash(code: str) -> int | None:
    """Return the index of the first `#` that starts a comment, or None.

    Tracks single/double-quote string state so `#` inside a string literal
    is not mistaken for a comment marker.
    """
    in_single = in_double = False
    i = 0
    while i < len(code):
        c = code[i]
        if c == "\\" and i + 1 < len(code):  # escape next char
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == "#" and not in_single and not in_double:
            return i
        i += 1
    return None
