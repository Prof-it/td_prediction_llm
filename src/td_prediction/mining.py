"""Repository mining: extract per-commit features via PyDriller.

The pipeline supports two modes:

1. **Cached (default).** If `data/features_<repo>.csv` already exists it is
   reused. This is the path used for reproducing paper results without
   re-cloning the 5 target repos.
2. **Fresh.** Pass `force=True` to re-mine from a local clone. Requires
   each repo to exist under `repos/<name>/` (see `config.REPOS`).

The feature set intentionally separates change-size/complexity signals
(see `config.CHANGE_COLS`) from file-history/maturity signals
(see `config.MATURITY_COLS`) so the training pipeline can run ablations
to diagnose shortcut learning.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from . import config
from .labeling.satd import satd_delta, is_py


def _quick_hunks(mod) -> int:
    """Approximate hunk count from a PyDriller modification's unified diff."""
    h = 0
    in_hunk = False
    for line in mod.diff.splitlines():
        if line.startswith(("+", "-")):
            if not in_hunk:
                in_hunk = True
                h += 1
        else:
            in_hunk = False
    return h


def _row_for_commit(
    repo: str,
    c,
    py_mods,
    file_prev_cc: dict,
    file_current_cc: dict,
    file_churn_cum: dict,
    file_authors: dict,
    file_times: dict,
    file_contributors: dict,
    commits_count_dict: dict,
    contrib_exp_dict: dict,
    hist_complexity_dict: dict,
) -> dict:
    la = sum(m.added_lines for m in py_mods)
    ld = sum(m.deleted_lines for m in py_mods)
    hunks = sum(_quick_hunks(m) for m in py_mods)
    n_methods = sum(len(m.changed_methods) for m in py_mods)

    cc_delta_sum = cc_delta_max = churn_delta = 0
    complexity_current_sum = churn_cum_sum = 0
    contributors_in_commit: set[str] = set()

    for m in py_mods:
        fp = m.new_path or m.old_path
        delta_cc = (m.complexity or 0) - file_prev_cc[fp]
        cc_delta_sum += delta_cc
        cc_delta_max = max(cc_delta_max, delta_cc)
        file_prev_cc[fp] = m.complexity or 0
        file_current_cc[fp] = m.complexity or 0
        complexity_current_sum += file_current_cc[fp]

        churn_this = m.added_lines + m.deleted_lines
        churn_delta += churn_this
        file_churn_cum[fp] += churn_this
        churn_cum_sum += file_churn_cum[fp]

        file_contributors[fp].add(c.author.email)
        contributors_in_commit.update(file_contributors[fp])
        file_authors[fp].add(c.author.email)
        file_times[fp].append(c.author_date)

    cutoff90 = c.author_date - dt.timedelta(days=90)
    py_fps = {m.new_path or m.old_path for m in py_mods}
    n_commits90 = sum(
        len([t for t in ts if t >= cutoff90])
        for fp, ts in file_times.items() if fp in py_fps
    )
    commits_count_file = sum(commits_count_dict.get(fp, 0) for fp in py_fps)
    contributors_experience = sum(contrib_exp_dict.get(fp, 0) for fp in py_fps)
    history_complexity = sum(hist_complexity_dict.get(fp, 0) for fp in py_fps)

    satd = sum(satd_delta(m) for m in py_mods)

    return {
        "repo_id": repo,
        "commit_hash": c.hash,
        "commit_uid": f"{repo}#{c.hash}",
        "commit_date": c.author_date.isoformat(),
        "lines_added": la,
        "lines_deleted": ld,
        "files_changed": len(py_mods),
        "hunks": hunks,
        "n_methods_changed": n_methods,
        "cc_delta_sum": cc_delta_sum,
        "cc_delta_max": cc_delta_max,
        "complexity_current_sum": complexity_current_sum,
        "churn_delta": churn_delta,
        "churn_cum": churn_cum_sum,
        "contributors_count": len(contributors_in_commit),
        "contributors_cum": sum(
            len(file_contributors[fp]) for fp in py_fps
        ),
        "n_authors_till_now": len({a for s in file_authors.values() for a in s}),
        "n_commits_file_past90d": n_commits90,
        "commits_count_file": commits_count_file,
        "contributors_experience": contributors_experience,
        "history_complexity": history_complexity,
        "dmm_unit_complexity": c.dmm_unit_complexity,
        "dmm_unit_size": c.dmm_unit_size,
        "dmm_unit_interfacing": c.dmm_unit_interfacing,
        "satd_delta": satd,
        "label_satd": 1 if satd > 0 else 0,
    }


def mine_repo(repo: str, *, force: bool = False) -> Path:
    """Mine one repository. Returns path to the features CSV.

    If `data/features_<repo>.csv` exists and `force` is False, no mining
    is performed and the existing CSV is used as-is.
    """
    from pydriller import Repository
    from pydriller.metrics.process.commits_count import CommitsCount
    from pydriller.metrics.process.contributors_experience import ContributorsExperience
    from pydriller.metrics.process.history_complexity import HistoryComplexity

    out = config.PATHS.data / f"features_{repo}.csv"
    if out.exists() and not force:
        return out

    cfg = config.REPOS[repo]
    repo_path = (config.PATHS.root / cfg["path"]).expanduser().resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(
            f"{repo_path} does not exist. Either clone the repo into "
            f"'repos/{repo}' or use the cached data/features_{repo}.csv."
        )

    start = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    print(f"[{repo}] computing process metrics ...")
    commits_count_dict = CommitsCount(str(repo_path), since=start, to=config.CUTOFF).count()
    contrib_exp_dict = ContributorsExperience(str(repo_path), since=start, to=config.CUTOFF).count()
    hist_complexity_dict = HistoryComplexity(str(repo_path), since=start, to=config.CUTOFF).count()

    file_prev_cc = collections.defaultdict(int)
    file_current_cc = collections.defaultdict(int)
    file_churn_cum = collections.defaultdict(int)
    file_authors = collections.defaultdict(set)
    file_times = collections.defaultdict(list)
    file_contributors = collections.defaultdict(set)

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=config.FEATURE_COLS)
        writer.writeheader()

        commits = Repository(
            path_to_repo=str(repo_path),
            only_in_branch=cfg["branch"],
            to=config.CUTOFF,
            only_modifications_with_file_types=[".py"],
            num_workers=64,
            skip_whitespaces=True,
            histogram_diff=True,
        ).traverse_commits()

        for c in tqdm(commits, desc=repo, leave=False):
            py = [m for m in c.modified_files if is_py(m)]
            if not py:
                continue
            row = _row_for_commit(
                repo, c, py,
                file_prev_cc, file_current_cc, file_churn_cum,
                file_authors, file_times, file_contributors,
                commits_count_dict, contrib_exp_dict, hist_complexity_dict,
            )
            writer.writerow(row)

    print(f"[{repo}] wrote {out}")
    return out


def mine_all(repos: Iterable[str] | None = None, *, force: bool = False) -> list[Path]:
    repos = list(repos) if repos else list(config.REPOS.keys())
    return [mine_repo(r, force=force) for r in tqdm(repos, desc="repos")]


def load_all_features(repos: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatenate all `features_<repo>.csv` into one DataFrame."""
    repos = list(repos) if repos else list(config.REPOS.keys())
    frames = []
    for r in repos:
        p = config.PATHS.data / f"features_{r}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run `mine_repo({r!r})` or restore from cache."
            )
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df["commit_dt"] = pd.to_datetime(df["commit_date"], utc=True)
    return df
