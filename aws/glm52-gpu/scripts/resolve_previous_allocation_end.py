#!/usr/bin/env python3
"""Resolve a replacement worker's prior billable end without counting pending time."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone


_TRANSITION = re.compile(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)")


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("allocation timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--previous-launch", required=True)
    parser.add_argument("--replacement-launch", required=True)
    args = parser.parse_args()
    previous_launch = _time(args.previous_launch)
    replacement_launch = _time(args.replacement_launch)
    if replacement_launch < previous_launch:
        parser.error("replacement launch precedes previous launch")
    try:
        response = json.load(sys.stdin)
    except json.JSONDecodeError:
        response = {}
    candidates = [
        item
        for reservation in response.get("Reservations", [])
        for item in reservation.get("Instances", [])
        if item.get("InstanceId") == args.instance_id
    ] if isinstance(response, dict) else []
    if len(candidates) == 1:
        instance = candidates[0]
        match = _TRANSITION.search(str(instance.get("StateTransitionReason", "")))
        if (
            instance.get("State", {}).get("Name") == "terminated"
            and match is not None
        ):
            exact = datetime.strptime(
                match.group(1),
                "%Y-%m-%d %H:%M:%S",
            ).replace(tzinfo=timezone.utc)
            if previous_launch <= exact <= replacement_launch:
                print(_iso(exact))
                return 0
    # Missing AWS history is charged conservatively through replacement launch.
    print(_iso(replacement_launch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
