#!/usr/bin/env python3
"""Find tagged prior P5 workers absent from the cumulative spend ledger."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


_TERMINATION = re.compile(
    r"\((?P<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}) GMT\)"
)


def _time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("EC2 allocation time must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _known_instances(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.is_symlink() or not path.is_file():
        raise ValueError("GPU spend ledger must be a regular file")
    known: set[str] = set()
    for line in path.read_text().splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(
            value.get("instance_id"), str
        ):
            raise ValueError("GPU spend ledger record is invalid")
        known.add(value["instance_id"])
    return known


def _termination_time(instance: dict[str, object]) -> datetime | None:
    match = _TERMINATION.search(str(instance.get("StateTransitionReason", "")))
    if match is None:
        return None
    return datetime.strptime(
        match.group("timestamp"),
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--current-instance-id", required=True)
    parser.add_argument("--replacement-launch", required=True)
    args = parser.parse_args()
    try:
        replacement_launch = _time(args.replacement_launch)
        response = json.load(sys.stdin)
        instances = [
            item
            for reservation in response.get("Reservations", [])
            for item in reservation.get("Instances", [])
        ]
        known = _known_instances(args.ledger)
        missing: list[tuple[datetime, dict[str, object]]] = []
        for instance in instances:
            instance_id = str(instance.get("InstanceId", ""))
            tags = {
                str(item.get("Key")): str(item.get("Value"))
                for item in instance.get("Tags", [])
            }
            if (
                not instance_id.startswith("i-")
                or instance.get("InstanceType") != "p5.48xlarge"
                or tags.get("project") != "keep-glm52"
                or tags.get("owner") != "jack.mazac"
                or tags.get("model") != "glm-5.2"
                or tags.get("cost-allocation") != "glm52-sky-campaign"
                or tags.get("campaign-run-id") != args.run_id
            ):
                raise ValueError("EC2 history contains a foreign campaign instance")
            if instance_id == args.current_instance_id or instance_id in known:
                continue
            state = str(instance.get("State", {}).get("Name", ""))
            if state in {"pending", "running"}:
                raise ValueError("unrecorded prior GPU is still active")
            launch = _time(instance.get("LaunchTime"))
            if launch >= replacement_launch:
                raise ValueError("prior GPU launch does not precede replacement")
            missing.append((launch, instance))
        missing.sort(key=lambda item: (item[0], str(item[1]["InstanceId"])))
        for index, (launch, instance) in enumerate(missing):
            next_launch = (
                missing[index + 1][0]
                if index + 1 < len(missing)
                else replacement_launch
            )
            ended = _termination_time(instance) or next_launch
            if ended < launch or ended > next_launch:
                raise ValueError("EC2 history contains overlapping GPU allocations")
            print(
                str(instance["InstanceId"]),
                _iso(launch),
                _iso(ended),
                sep="\t",
            )
        return 0
    except (AttributeError, json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
