#!/usr/bin/env python3
"""Re-query and purchase only the explicitly approved Oregon 48-hour p5 block."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from decimal import Decimal


def _aws(*arguments: str) -> dict:
    result = subprocess.run(
        ["aws", *arguments, "--output", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"AWS CLI exit {result.returncode}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("AWS CLI returned a non-object response")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("approved dates must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--approved-start", required=True)
    parser.add_argument("--approved-end", required=True)
    parser.add_argument("--approved-az", default="us-west-2a")
    parser.add_argument("--instance-type", default="p5.48xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--duration-hours", type=int, default=48)
    parser.add_argument("--maximum-upfront-fee", type=Decimal, default=Decimal("1993.34"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    approved_start = _utc(args.approved_start)
    approved_end = _utc(args.approved_end)
    if (approved_end - approved_start).total_seconds() != args.duration_hours * 3600:
        raise SystemExit("approved start/end do not equal the approved duration")
    # The API treats start-date-range as the earliest start and end-date-range
    # as the latest end. Duration is pinned, and the exact values are checked
    # again below before any purchase call.
    offerings = _aws(
        "ec2",
        "describe-capacity-block-offerings",
        "--region",
        args.region,
        "--instance-type",
        args.instance_type,
        "--instance-count",
        str(args.instance_count),
        "--capacity-duration-hours",
        str(args.duration_hours),
        "--start-date-range",
        approved_start.isoformat(),
        "--end-date-range",
        approved_end.isoformat(),
    ).get("CapacityBlockOfferings", [])
    matches = []
    for offering in offerings:
        if (
            offering.get("InstanceType") == args.instance_type
            and offering.get("InstanceCount") == args.instance_count
            and offering.get("AvailabilityZone") == args.approved_az
            and _utc(offering["StartDate"]) == approved_start
            and _utc(offering["EndDate"]) == approved_end
            and Decimal(offering["UpfrontFee"]) <= args.maximum_upfront_fee
        ):
            matches.append(offering)
    if len(matches) != 1:
        raise SystemExit(
            "guard refused purchase: exact start/AZ/duration/count/type/fee match was not unique"
        )
    selected = matches[0]
    print(json.dumps({"guard": "passed", "offering": selected}, sort_keys=True))
    if not args.execute:
        return 0
    purchase = _aws(
        "ec2",
        "purchase-capacity-block",
        "--region",
        args.region,
        "--capacity-block-offering-id",
        selected["CapacityBlockOfferingId"],
        "--instance-platform",
        "Linux/UNIX",
        "--tag-specifications",
        (
            "ResourceType=capacity-reservation,Tags=["
            f"{{Key=campaign-run-id,Value={args.run_id}}},"
            "{Key=project,Value=keep-glm52},{Key=owner,Value=jack.mazac}]"
        ),
    )
    print(json.dumps({"purchase": purchase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
