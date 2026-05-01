"""Submit already-built batch JSONL files to the OpenAI Batches API.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/submit_batches.py --glob 'llm_batch/*_v2_rubric_json.jsonl'
    python scripts/submit_batches.py --poll-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)  # .env wins over shell env — submission key must match poll key


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_src_on_path()
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="llm_batch/*.jsonl",
                   help="Glob (relative to repo root) of files to submit.")
    p.add_argument("--poll-only", action="store_true",
                   help="Skip submission; just check status / download outputs.")
    p.add_argument("--wait", action="store_true",
                   help="Poll until all batches finish.")
    args = p.parse_args()

    from td_prediction import config
    from td_prediction.labeling import llm_judge

    if not args.poll_only:
        files = sorted(config.PATHS.root.glob(args.glob))
        files = [f for f in files if "_output" not in f.name and "_errors" not in f.name]
        if not files:
            raise SystemExit(f"No files match {args.glob}")
        print(f"Submitting {len(files)} file(s):")
        for f in files:
            print(f"  {f}")
        llm_judge.submit_batches(files)

    status = llm_judge.poll_batches(wait=args.wait)
    for name, meta in status.items():
        print(f"{name}: {meta.get('status')}")


if __name__ == "__main__":
    main()
