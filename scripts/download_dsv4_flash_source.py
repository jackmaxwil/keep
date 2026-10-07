#!/usr/bin/env python3
"""Download the DeepSeek-V4-Flash-0731 source snapshot (Wave 1, ~172 GB).

Mixed precision on disk: FP4-in-I8 routed experts with E8M0 group-32 scales,
FP8 e4m3 residents with 128x128 block scales.

Resumable: huggingface_hub skips completed files on rerun. Writes a
completion marker with the pinned revision when every file is present.

The revision is pinned to the snapshot the family profile and policy were
measured against, so a rerun resumes the *same* snapshot instead of silently
switching to whatever ``main`` points at now (which would mix shards from two
revisions into one directory). ``--refresh-revision`` resolves the live head
instead, for the deliberate act of moving the pin.

Usage:
    uv run --group dev python scripts/download_dsv4_flash_source.py
    uv run --group dev python scripts/download_dsv4_flash_source.py --refresh-revision
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.hf_api import HfApi

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
# Must stay in sync with ``revision:`` in models/deepseek-v4-flash-0731.yaml.
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
LOCAL_DIR = Path.home() / "models" / "DeepSeek-V4-Flash-0731"
MARKER = LOCAL_DIR / "_KEEP_DOWNLOAD_COMPLETE.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-revision",
        action="store_true",
        help=(
            "Resolve the live repo head instead of the pinned REVISION. Moves "
            "the snapshot off the measured pin; update the profile to match."
        ),
    )
    args = parser.parse_args(argv)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_revision:
        revision = HfApi().model_info(MODEL_ID).sha
        if revision != REVISION:
            print(
                f"WARNING: live head {revision} differs from pinned {REVISION}; "
                "update REVISION here and revision: in the family profile.",
                flush=True,
            )
    else:
        revision = REVISION
    print(f"downloading {MODEL_ID} @ {revision} -> {LOCAL_DIR}", flush=True)

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        local_dir=LOCAL_DIR,
        max_workers=4,
    )

    marker = {
        "record_type": "keep_dsv4_source_download_v1",
        "model_id": MODEL_ID,
        "revision": revision,
        "local_path": str(path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    MARKER.write_text(json.dumps(marker, indent=2) + "\n")
    print(f"complete: {MARKER}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
