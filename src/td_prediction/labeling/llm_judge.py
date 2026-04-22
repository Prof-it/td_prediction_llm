"""LLM-as-judge: build batch files, submit, poll, and parse results.

Separation of concerns:
- `build_batch_file(rows, variant, out_path, ...)` — writes a single JSONL
  batch file for the OpenAI Batches API. Never calls the API.
- `submit_batches(files)` — submits existing JSONL files; returns metadata.
- `poll_batches(metadata_path)` — checks status, downloads outputs.
- `parse_results(output_dir, variant)` — parses label / confidence / rationale
  out of either v1 (yes/no) or v2 (JSON) responses.

This matches the thesis flow but cleanly separates mechanical from
cost-incurring operations — you can regenerate the JSONL files any time
at zero cost, and only spend tokens when you explicitly submit.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from tqdm import tqdm

from .. import config
from . import prompts
from .satd import strip_comments


def _custom_id(commit_uid: str, variant: str, trial: int = 0) -> str:
    """Stable, unique-per-(commit,variant,trial) ID.

    Self-consistency uses trial>0 to request multiple samples per commit.
    The commit_uid is hashed when composing so the ID length stays bounded.
    """
    base = f"{commit_uid}::{variant}::t{trial}"
    # Keep the commit_uid readable; just append a short digest if it is long.
    h = hashlib.sha1(base.encode()).hexdigest()[:10]
    return f"{commit_uid}|{variant}|t{trial}|{h}"


def _parse_custom_id(custom_id: str) -> tuple[str, str, int]:
    """Inverse of `_custom_id`. Returns (commit_uid, variant, trial).

    For legacy IDs produced by the thesis code (which used a single
    dash-joined suffix) falls back to best-effort parsing so old outputs
    still load.
    """
    if "|" in custom_id:
        parts = custom_id.split("|")
        commit_uid, variant, trial_str = parts[0], parts[1], parts[2]
        trial = int(trial_str.lstrip("t"))
        return commit_uid, variant, trial
    # Legacy format: "<repo>#<sha>-<variant>"
    for suffix, variant in [
        ("-original-prompt", "v1_original"),
        ("-satd-filtered",   "v1_satd_filtered"),
        ("-diff-removed",    "v1_diff_removed"),
        ("-original-diff",   "v1_original"),
        ("-no-diff",         "v1_diff_removed"),
    ]:
        if custom_id.endswith(suffix):
            return custom_id[: -len(suffix)], variant, 0
    # Further legacy: only the commit_uid, no suffix
    return custom_id, "v1_original", 0


@dataclass
class BatchPlan:
    """Describes one batch-file group: a prompt variant applied to some rows."""
    variant: str
    rows: pd.DataFrame
    diffs: dict[str, str]            # commit_uid -> diff text
    few_shot: list[dict] | None = None
    n_consistency: int = 1           # >1 enables self-consistency sampling
    temperature: float = field(init=False)

    def __post_init__(self) -> None:
        # Self-consistency requires nonzero temperature to get diverse samples.
        self.temperature = 0.7 if self.n_consistency > 1 else config.LLM_TEMPERATURE


def build_batch_file(plan: BatchPlan, out_path: Path, *, strip_satd_comments: bool = False) -> Path:
    """Write one .jsonl batch file for `plan`. Returns the path written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    builder = prompts.VARIANTS[plan.variant]
    max_tok = prompts.max_tokens_for(plan.variant)

    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in plan.rows.iterrows():
            diff_text = plan.diffs.get(row["commit_uid"], "")
            if strip_satd_comments:
                diff_text = strip_comments(diff_text)
            for trial in range(plan.n_consistency):
                messages = builder(row.to_dict(), diff_text, few_shot=plan.few_shot)
                request = {
                    "custom_id": _custom_id(row["commit_uid"], plan.variant, trial),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": config.LLM_MODEL,
                        "messages": messages,
                        "max_tokens": max_tok,
                        "temperature": plan.temperature,
                    },
                }
                fh.write(json.dumps(request, ensure_ascii=False) + "\n")
    return out_path


def submit_batches(files: Iterable[Path], *, description: str = "TD labeling") -> dict:
    """Upload and start batches for each file. Does not wait for completion."""
    from openai import OpenAI
    client = OpenAI(api_key=config.get_openai_api_key())

    metadata: dict = {}
    for f in tqdm(list(files), desc="uploading"):
        with open(f, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{description}: {f.name}"},
        )
        metadata[f.name] = {
            "file_id": uploaded.id,
            "batch_id": batch.id,
            "status": "submitted",
        }
    meta_path = config.PATHS.llm_batch / "batch_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def poll_batches(*, wait: bool = False, interval: int = 60) -> dict:
    """Check status for all known batches; download outputs when ready.

    If `wait` is True, block until all batches terminate.
    """
    from openai import OpenAI
    client = OpenAI(api_key=config.get_openai_api_key())
    meta_path = config.PATHS.llm_batch / "batch_metadata.json"
    metadata = json.loads(meta_path.read_text())

    while True:
        all_done = True
        for name, meta in metadata.items():
            batch = client.batches.retrieve(meta["batch_id"])
            meta["status"] = batch.status
            if batch.output_file_id:
                out_p = config.PATHS.llm_batch / f"{name}_output.jsonl"
                with open(out_p, "wb") as out_f:
                    out_f.write(client.files.content(batch.output_file_id).read())
            if batch.error_file_id:
                err_p = config.PATHS.llm_batch / f"{name}_errors.jsonl"
                with open(err_p, "wb") as err_f:
                    err_f.write(client.files.content(batch.error_file_id).read())
            if batch.status not in {"completed", "failed", "expired", "cancelled"}:
                all_done = False
        status_p = config.PATHS.llm_batch / "batch_status.json"
        status_p.write_text(json.dumps(metadata, indent=2, default=str))
        if not wait or all_done:
            return metadata
        time.sleep(interval)


def _parse_one_response(data: dict) -> tuple[str | None, float | None, str | None]:
    """Extract (label, confidence, rationale) from a single response record."""
    try:
        content = data["response"]["body"]["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return None, None, None

    # v2 structured JSON
    if content.startswith("{"):
        try:
            obj = json.loads(content)
            label = str(obj.get("label", "")).strip().lower()
            conf = float(obj.get("confidence", float("nan")))
            rationale = obj.get("rationale")
            if label in {"yes", "no"}:
                return label, conf, rationale
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # v1 single-token yes/no
    low = content.lower()
    if low.startswith("yes"):
        return "yes", None, None
    if low.startswith("no"):
        return "no", None, None
    return None, None, content


def parse_results(
    *,
    variant_filter: str | None = None,
    output_files: Iterable[Path] | None = None,
) -> pd.DataFrame:
    """Parse all LLM outputs into a long-format DataFrame.

    Returns one row per (commit_uid, variant, trial) with columns:
    [commit_uid, variant, trial, label, label_int, confidence, rationale].

    `variant_filter` restricts to a specific prompt variant.
    """
    if output_files is None:
        output_files = list(config.PATHS.llm_batch.glob("*_output.jsonl"))

    records = []
    for f in tqdm(list(output_files), desc="parsing"):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                cid = data.get("custom_id", "")
                commit_uid, variant, trial = _parse_custom_id(cid)
                if variant_filter and variant != variant_filter:
                    continue
                label, conf, rationale = _parse_one_response(data)
                records.append({
                    "commit_uid": commit_uid,
                    "variant": variant,
                    "trial": trial,
                    "label": label,
                    "label_int": {"yes": 1, "no": 0}.get(label),
                    "confidence": conf,
                    "rationale": rationale,
                })
    return pd.DataFrame.from_records(records)


def aggregate_self_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple trials per (commit_uid, variant) with majority vote.

    Returns one row per (commit_uid, variant).
    """
    df = df.dropna(subset=["label_int"])
    grouped = df.groupby(["commit_uid", "variant"], as_index=False).agg(
        label_int=("label_int", lambda s: int(s.mean() >= 0.5)),
        agreement=("label_int", lambda s: float(max(s.mean(), 1 - s.mean()))),
        n_trials=("label_int", "size"),
        mean_confidence=("confidence", "mean"),
        rationale=("rationale", "first"),
    )
    return grouped


def attach_llm_labels(
    features_df: pd.DataFrame,
    *,
    variant: str,
    parsed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach `label_llm` (0/1) and optional confidence/agreement columns.

    The legacy CSVs written by the thesis pipeline may already have a
    `label_llm` column baked in. Drop it before merging so pandas doesn't
    produce `label_llm_x` / `label_llm_y` suffixes.
    """
    if parsed is None:
        parsed = parse_results(variant_filter=variant)
    agg = aggregate_self_consistency(parsed[parsed["variant"] == variant])
    base = features_df.drop(
        columns=[c for c in ("label_llm", "llm_agreement", "llm_confidence", "n_trials", "rationale_llm") if c in features_df.columns]
    )
    merged = base.merge(
        agg.rename(columns={
            "label_int": "label_llm",
            "agreement": "llm_agreement",
            "mean_confidence": "llm_confidence",
            "rationale": "rationale_llm",
        }).drop(columns="variant"),
        on="commit_uid", how="left",
    )
    # Unlabeled commits default to 0 (No Debt) to match the legacy pipeline;
    # downstream code can filter on llm_agreement.notna() when stricter.
    n_unparseable = merged["label_llm"].isna().sum()
    if n_unparseable > 0:
        print(f"[warn] {n_unparseable} commits had unparseable LLM responses — defaulting to label_llm=0")
    merged["label_llm"] = merged["label_llm"].fillna(0).astype(int)
    return merged
