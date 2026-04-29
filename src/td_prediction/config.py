"""Central configuration for the TD-prediction pipeline.

Paths are computed from the repository root so the code works both when
invoked from the repo root and from inside `notebooks/`. A single
`Config` dataclass is the source of truth — any script or notebook that
needs paths, seeds, or hyperparameters should go through it.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until a marker file is found.

    Markers: `.git`, `README.md`, `requirements.txt`. Falls back to cwd.
    """
    p = (start or Path(__file__).resolve()).parent
    for parent in [p, *p.parents]:
        if (parent / ".git").exists() or (parent / "README.md").exists():
            return parent
    return Path.cwd()


REPO_ROOT: Path = _find_repo_root()


# Seeds and reproducibility
SEED: int = 42
N_JOBS: int = -1


# Mining cutoff — pinned so the dataset is reproducible
CUTOFF: dt.datetime = dt.datetime(2025, 6, 15, 23, 59, 59, tzinfo=dt.timezone.utc)


# SATD keyword regex (case-insensitive). Used for the regex baseline label
# AND for optional comment-stripping before LLM labeling.
SATD_PATTERN: re.Pattern[str] = re.compile(
    r"\b(TODO|FIXME|BUG|HACK|XXX|WORKAROUND|TEMP|KLUDGE|UGLY|DIRTY|BROKEN|FIX)\b",
    re.IGNORECASE,
)


# Repositories to mine. Paths are relative to REPO_ROOT.
REPOS: dict[str, dict[str, str]] = {
    "requests": {"path": "repos/requests", "branch": "main"},
    "fastapi":  {"path": "repos/fastapi",  "branch": "master"},
    "scrapy":   {"path": "repos/scrapy",   "branch": "master"},
    "flask":    {"path": "repos/flask",    "branch": "main"},
    "keras":    {"path": "repos/keras",    "branch": "master"},
}

REPO_URL_MAP: dict[str, str] = {
    "requests": "https://github.com/psf/requests",
    "fastapi":  "https://github.com/fastapi/fastapi",
    "scrapy":   "https://github.com/scrapy/scrapy",
    "flask":    "https://github.com/pallets/flask",
    "keras":    "https://github.com/keras-team/keras",
}


# CSV columns mined per commit.
FEATURE_COLS: list[str] = [
    "repo_id", "commit_hash", "commit_uid", "commit_date",
    "lines_added", "lines_deleted", "files_changed", "hunks",
    "n_methods_changed", "cc_delta_sum", "cc_delta_max",
    "complexity_current_sum", "churn_delta", "churn_cum",
    "contributors_count", "contributors_cum", "n_authors_till_now",
    "n_commits_file_past90d", "commits_count_file",
    "contributors_experience", "history_complexity",
    "dmm_unit_complexity", "dmm_unit_size", "dmm_unit_interfacing",
    "satd_delta", "label_satd",
]


# Non-feature columns — never passed to the model.
META_COLS: list[str] = [
    "repo_id", "commit_hash", "commit_uid", "commit_date", "commit_dt",
]

# Label columns — never used as input features.
LABEL_COLS: list[str] = [
    "label_satd", "label_llm", "label_human",
]

# Features always dropped — either direct label proxies or temporal leakage.
# commits_count_file / contributors_experience / history_complexity are computed
# from the full repo history up to the global cutoff, not up to each commit's
# date, so they encode future information for early commits (temporal leakage).
LEAKAGE_COLS: list[str] = [
    "satd_delta",
    "commits_count_file",
    "contributors_experience",
    "history_complexity",
]

# Maturity / history features flagged as shortcut-learning suspects.
# These are included by default but an ablation run removes them.
MATURITY_COLS: list[str] = [
    "contributors_cum", "n_authors_till_now",
    "n_commits_file_past90d", "churn_cum", "complexity_current_sum",
]

# Change-size / complexity features that describe THIS commit only.
CHANGE_COLS: list[str] = [
    "lines_added", "lines_deleted", "files_changed", "hunks",
    "n_methods_changed", "cc_delta_sum", "cc_delta_max",
    "churn_delta", "contributors_count",
    "dmm_unit_complexity", "dmm_unit_size", "dmm_unit_interfacing",
]


@dataclass
class Paths:
    """Filesystem paths, all anchored at REPO_ROOT."""
    root: Path = REPO_ROOT
    data: Path = field(default_factory=lambda: REPO_ROOT / "data")
    splits: Path = field(default_factory=lambda: REPO_ROOT / "splits")
    llm_batch: Path = field(default_factory=lambda: REPO_ROOT / "llm_batch")
    artifacts: Path = field(default_factory=lambda: REPO_ROOT / "artifacts")
    figures: Path = field(default_factory=lambda: REPO_ROOT / "artifacts" / "figures")
    results: Path = field(default_factory=lambda: REPO_ROOT / "artifacts" / "results")
    models: Path = field(default_factory=lambda: REPO_ROOT / "artifacts" / "models")
    repos: Path = field(default_factory=lambda: REPO_ROOT / "repos")

    def ensure(self) -> "Paths":
        for p in (self.data, self.splits, self.llm_batch,
                  self.artifacts, self.figures, self.results, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths().ensure()


# LLM-as-judge config
#
# Labeling strategy (three tiers, each building on the previous):
#   Tier 1 — gold set only (100 commits): run v2_rubric_json to get
#             human-vs-LLM agreement metrics. Cost: ~$0.05.
#   Tier 2 — ablation sample (2–5k commits): compare v1 vs. v2 label
#             distributions and agreement with human labels.  Cost: ~$1–3.
#   Tier 3 — full re-label (~70k commits): only if tier-2 shows v2 labels
#             meaningfully improve ML metrics over v1. Cost: ~$10–20.
#
# The main ML training pipeline uses the existing v1 cached labels by
# default (PARSE_VARIANT below). Upgrade to "v2_rubric_json" only after
# running at least tier-1 to validate that v2 agrees better with humans.
LLM_MODEL: str = "gpt-5.4-mini"  # v2 re-label ~$7 via batch API
LLM_MODEL_V1: str = "gpt-4.1-mini"  # model used for cached v1 outputs
LLM_TEMPERATURE: float = 0.0
PARSE_VARIANT: str = "v1_satd_filtered"  # variant driving label_llm today
# For self-consistency the pipeline can optionally re-query N times with
# temperature > 0 and majority-vote. Set N_CONSISTENCY=1 to disable.
N_CONSISTENCY: int = 1
BATCH_MAX_MB: int = 180
BATCH_MAX_REQ: int = 50_000


# Splits: chronological within each repo. Default train/val/test = 60/20/20.
TRAIN_FRAC: float = 0.60
VAL_FRAC: float = 0.20
TEST_FRAC: float = 0.20


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the LLM "
            "batch step, or skip that step and use cached llm_batch/ outputs."
        )
    return key
