#!/usr/bin/env python3
"""Read-only, throttling-aware monitor for earlier P5 capacity.

Each invocation checks one staged region (round-robin), makes at most seven EC2
API calls, and persists exponential backoff.  It never purchases or launches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlx_vq.quality.glm52_capacity_monitor import (  # noqa: E402
    INSTANCE_TYPES,
    is_more_appealing,
)

# The live EC2 API rejects 24-hour P5 searches; 48 hours is the shortest
# compatible P5-family block accepted for this campaign/account.
DURATIONS = (48,)
MAX_REQUESTS = 7


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("capacity time must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _aws_json(region: str, arguments: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"AWS_MAX_ATTEMPTS": "1", "AWS_RETRY_MODE": "standard"})
    completed = subprocess.run(
        ["aws", "ec2", *arguments, "--region", region, "--output", "json", "--no-cli-pager"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"AWS CLI exit {completed.returncode}"
        raise RuntimeError(detail)
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("AWS capacity response must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--record-jsonl", type=Path, required=True)
    parser.add_argument("--approved-start", required=True)
    parser.add_argument("--approved-max-fee", type=float, default=1993.34)
    parser.add_argument("--approved-duration-hours", type=int, default=48)
    parser.add_argument("--regions", nargs="+", default=["us-west-2", "us-east-1"])
    parser.add_argument(
        "--staged-az",
        action="append",
        default=["us-west-2a", "us-east-1a"],
        help="repeat for every availability zone backed by staged infrastructure",
    )
    args = parser.parse_args()
    approved_start = _utc(args.approved_start)
    approved_hourly = args.approved_max_fee / args.approved_duration_hours
    approved_end = approved_start + timedelta(hours=args.approved_duration_hours)
    now = datetime.now(timezone.utc)
    state: dict[str, Any] = {}
    if args.state.is_file():
        state = json.loads(args.state.read_bytes())
    next_not_before = state.get("next_not_before")
    if isinstance(next_not_before, str) and now < _utc(next_not_before):
        print(json.dumps({"status": "backoff", "next_not_before": next_not_before}))
        return 0
    region_index = int(state.get("next_region_index", 0)) % len(args.regions)
    region = args.regions[region_index]
    requests = 0
    offerings: list[dict[str, Any]] = []
    placement: dict[str, Any] = {}
    try:
        for instance_type in INSTANCE_TYPES:
            for duration in DURATIONS:
                result = _aws_json(
                    region,
                    [
                        "describe-capacity-block-offerings",
                        "--instance-type",
                        instance_type,
                        "--instance-count",
                        "1",
                        "--capacity-duration-hours",
                        str(duration),
                        "--end-date-range",
                        approved_end.isoformat().replace("+00:00", "Z"),
                        "--no-paginate",
                    ],
                )
                requests += 1
                values = result.get("CapacityBlockOfferings", [])
                if isinstance(values, list):
                    offerings.extend(value for value in values if isinstance(value, dict))
        placement = _aws_json(
            region,
            [
                "get-spot-placement-scores",
                "--instance-types",
                *INSTANCE_TYPES,
                "--target-capacity",
                "1",
                "--single-availability-zone",
                "--region-names",
                region,
            ],
        )
        requests += 1
        if requests > MAX_REQUESTS:
            raise RuntimeError("capacity monitor exceeded its request budget")
    except Exception as error:
        failures = int(state.get("consecutive_failures", 0)) + 1
        delay_minutes = min(30 * (2 ** min(failures, 5)), 16 * 60)
        next_time = now + timedelta(minutes=delay_minutes)
        _write_json(
            args.state,
            {
                "consecutive_failures": failures,
                "last_error": str(error),
                "last_region": region,
                "next_not_before": next_time.isoformat().replace("+00:00", "Z"),
                "next_region_index": region_index,
            },
        )
        print(json.dumps({"status": "error-backoff", "region": region, "error": str(error)}))
        return 2
    appealing = [
        offering
        for offering in offerings
        if is_more_appealing(
            offering,
            approved_start=approved_start,
            approved_hourly_cost=approved_hourly,
            staged_availability_zones=set(args.staged_az),
        )
    ]
    record = {
        "record_type": "glm52_gpu_capacity_monitor_v1",
        "observed_at": now.isoformat().replace("+00:00", "Z"),
        "region": region,
        "request_count": requests,
        "offering_count": len(offerings),
        "more_appealing": appealing,
        "spot_placement_scores": placement.get("SpotPlacementScores", []),
        "read_only": True,
        "purchase_attempted": False,
        "launch_attempted": False,
    }
    args.record_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.record_jsonl.open("ab") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _write_json(
        args.state,
        {
            "consecutive_failures": 0,
            "last_region": region,
            "last_observed_at": record["observed_at"],
            "next_region_index": (region_index + 1) % len(args.regions),
        },
    )
    print(json.dumps(record, sort_keys=True))
    return 3 if appealing else 0


if __name__ == "__main__":
    raise SystemExit(main())
