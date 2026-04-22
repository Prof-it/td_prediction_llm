"""Human gold-labeling support.

Generates a stratified review spreadsheet (CSV) with GitHub commit
links, LLM and SATD labels, and empty columns for the human annotator
to fill in. Also computes agreement metrics (accuracy, precision,
recall, Cohen's κ) once the spreadsheet is returned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def sample_for_review(
    df: pd.DataFrame,
    *,
    n_per_cell: int = 25,
    llm_col: str = "label_llm",
    satd_col: str = "label_td_satd",
    random_state: int = config.SEED,
) -> pd.DataFrame:
    """Stratified sample across the (SATD × LLM) 2×2 label grid.

    `n_per_cell=25` gives 100 total (25 each for LLM-yes/SATD-yes,
    LLM-yes/SATD-no, LLM-no/SATD-yes, LLM-no/SATD-no).
    """
    rng = np.random.default_rng(random_state)
    parts = []
    for s in (0, 1):
        for l in (0, 1):
            cell = df[(df[satd_col] == s) & (df[llm_col] == l)]
            k = min(n_per_cell, len(cell))
            if k == 0:
                continue
            idx = rng.choice(cell.index.values, size=k, replace=False)
            parts.append(cell.loc[idx])
    sample = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state)

    sample["commit_url"] = sample.apply(
        lambda r: f"{config.REPO_URL_MAP[r['repo_id']]}/commit/{r['commit_hash']}",
        axis=1,
    )
    # Empty columns for the annotator.
    sample["label_human"] = ""
    sample["rationale_human"] = ""
    cols = [
        "commit_uid", "repo_id", "commit_hash", "commit_url",
        satd_col, llm_col,
        "label_human", "rationale_human",
        "lines_added", "lines_deleted", "files_changed", "hunks",
    ]
    if "rationale_llm" in sample.columns:
        cols.insert(cols.index(llm_col) + 1, "rationale_llm")
    return sample[cols]


def _safe_kappa(y_true, y_pred, fn) -> float:
    try:
        return float(fn(y_true, y_pred))
    except Exception:
        return float("nan")


def agreement_table(sample: pd.DataFrame, *, llm_col: str = "label_llm", satd_col: str = "label_td_satd") -> pd.DataFrame:
    """Return per-judge agreement vs human (after the sheet is filled in).

    Note: Cohen's κ is computed on a stratified 2×2 sample (balanced by design).
    This κ reflects agreement on the balanced gold set, not the natural class
    distribution (~5-10% positive). Do not compare directly to population-level κ.
    """
    from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score

    sheet = sample.dropna(subset=["label_human"]).copy()
    sheet = sheet[sheet["label_human"].astype(str).str.strip().isin(["0", "1"])]
    sheet["label_human"] = sheet["label_human"].astype(int)

    rows = []
    for name, col in [("SATD_regex", satd_col), ("LLM_judge", llm_col)]:
        y_true = sheet["label_human"]
        y_pred = sheet[col].astype(int)
        rows.append({
            "judge": name,
            "n": len(sheet),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "cohen_kappa": _safe_kappa(y_true, y_pred, cohen_kappa_score),
        })
    return pd.DataFrame(rows)


def write_review_sheet(sample: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    return out_path
