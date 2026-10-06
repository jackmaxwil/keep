#!/usr/bin/env python3
"""Safely re-invoke the idempotent controller or request a graceful stop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_PATH = (
    REPO_ROOT / "src/mlx_vq/quality/glm52_capacity_block_controller.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_glm52_capacity_block_controller", CONTROLLER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {CONTROLLER_PATH}")
_CONTROLLER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONTROLLER
_SPEC.loader.exec_module(_CONTROLLER)
validate_campaign_descriptor = _CONTROLLER.validate_campaign_descriptor


def _aws_json(arguments: Sequence[str], *, region: str) -> dict[str, Any]:
    result = subprocess.run(
        ["aws", *arguments, "--region", region, "--output", "json"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("AWS CLI returned a non-object")
    return value


def _same_time(left: object, right: object) -> bool:
    def parse(value: object) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    return parse(left) == parse(right)


def _event(descriptor: dict[str, object], *, action: str) -> dict[str, object]:
    detail_type = {
        "delivery": "Capacity Block Reservation Delivered",
        "graceful-stop": "Capacity Block Reservation Expiration Warning",
    }[action]
    return {
        "version": "0",
        "id": f"break-glass-{descriptor['run_id']}-{action}",
        "detail-type": detail_type,
        "source": "aws.ec2",
        "account": descriptor["account_id"],
        "region": descriptor["region"],
        "time": datetime.now().astimezone().isoformat(),
        "detail": {
            "capacity-reservation-id": descriptor["capacity_reservation_id"],
            "end-date": descriptor["capacity_block_end"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("delivery", "graceful-stop"))
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--parameter", default="/keep-glm52/campaign/active")
    parser.add_argument(
        "--function-name", default="keep-glm52-capacity-block-controller"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the Lambda invocation. Without this flag, only print the event.",
    )
    args = parser.parse_args()
    parameter = _aws_json(
        ["ssm", "get-parameter", "--name", args.parameter], region=args.region
    )
    descriptor = validate_campaign_descriptor(
        json.loads(str(parameter["Parameter"]["Value"]))
    )
    region = str(descriptor["region"])
    reservations = _aws_json(
        [
            "ec2",
            "describe-capacity-reservations",
            "--capacity-reservation-ids",
            str(descriptor["capacity_reservation_id"]),
        ],
        region=region,
    ).get("CapacityReservations", [])
    if len(reservations) != 1:
        raise SystemExit("capacity reservation lookup was not unique")
    reservation = reservations[0]
    checks = {
        "reservation_id": reservation.get("CapacityReservationId")
        == descriptor["capacity_reservation_id"],
        "instance_type": reservation.get("InstanceType") == descriptor["instance_type"],
        "availability_zone": reservation.get("AvailabilityZone")
        == descriptor["availability_zone"],
        "end_time": _same_time(reservation.get("EndDate"), descriptor["capacity_block_end"]),
        "active": reservation.get("State") == "active",
    }
    if not all(checks.values()):
        raise SystemExit("refusing break-glass action; reservation authority drift: " + json.dumps(checks))
    event = _event(descriptor, action=args.action)
    preview = {
        "execute": args.execute,
        "function_name": args.function_name,
        "reservation_checks": checks,
        "event": event,
    }
    if not args.execute:
        print(json.dumps(preview, sort_keys=True, indent=2))
        return 0
    with tempfile.NamedTemporaryFile() as response:
        invocation = subprocess.run(
            [
                "aws",
                "lambda",
                "invoke",
                "--function-name",
                args.function_name,
                "--invocation-type",
                "RequestResponse",
                "--cli-binary-format",
                "raw-in-base64-out",
                "--payload",
                json.dumps(event, separators=(",", ":")),
                response.name,
                "--region",
                region,
                "--output",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        response.seek(0)
        body = response.read().decode()
    if invocation.returncode != 0:
        raise SystemExit(invocation.stderr.strip() or invocation.stdout.strip())
    metadata = json.loads(invocation.stdout)
    result = json.loads(body) if body else None
    output = {**preview, "invocation": metadata, "controller_result": result}
    print(json.dumps(output, sort_keys=True, indent=2))
    if metadata.get("FunctionError"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
