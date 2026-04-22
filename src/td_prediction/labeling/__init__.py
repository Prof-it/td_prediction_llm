"""Labelers for technical-debt commits.

Three label sources exist:
- `label_satd`  — regex-based SATD keyword detection on added/removed comments.
- `label_llm`      — GPT-4.1-mini as judge on the full diff + metrics.
- `label_human`    — manually labeled gold set (a stratified sample).
"""
