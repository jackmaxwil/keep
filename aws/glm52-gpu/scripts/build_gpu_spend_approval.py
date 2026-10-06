#!/usr/bin/env python3
"""Build the immutable Kon-approved GLM-5.2 GPU spend authority."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
SPEC = importlib.util.spec_from_file_location("_glm52_sky_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
SKY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SKY
SPEC.loader.exec_module(SKY)
build_gpu_spend_approval = SKY.build_gpu_spend_approval
validate_gpu_spend_approval = SKY.validate_gpu_spend_approval


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingested-at", required=True)
    parser.add_argument("--slack-permalink")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        ingested_at = datetime.fromisoformat(
            args.ingested_at.replace("Z", "+00:00")
        )
        approval = build_gpu_spend_approval(
            ingested_at=ingested_at,
            slack_permalink=args.slack_permalink,
        )
        validate_gpu_spend_approval(approval)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    _write_atomic(args.output, approval)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
