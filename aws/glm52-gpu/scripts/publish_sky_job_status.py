#!/usr/bin/env python3
"""Build one authenticated SkyPilot status marker without calling AWS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SKY = _load("_glm52_status_sky", "src/mlx_vq/quality/glm52_sky_campaign.py")
WATCHDOG = _load(
    "_glm52_status_watchdog",
    "src/mlx_vq/quality/glm52_campaign_watchdog.py",
)


def _parse_time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("observed-at must be timezone-aware")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument(
        "--status",
        choices=(
            "SUBMITTED",
            "PENDING",
            "RUNNING",
            "RECOVERING",
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
        ),
        required=True,
    )
    parser.add_argument("--instance-id")
    parser.add_argument("--observed-at")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    descriptor = SKY.validate_sky_campaign_descriptor(
        json.loads(args.descriptor.read_bytes())
    )
    marker = WATCHDOG.build_skypilot_job_status(
        run_id=str(descriptor["run_id"]),
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        sky_job_name=args.job_name,
        status=args.status,
        instance_id=args.instance_id,
        observed_at=_parse_time(args.observed_at),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.",
        dir=args.output.parent,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(marker, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
