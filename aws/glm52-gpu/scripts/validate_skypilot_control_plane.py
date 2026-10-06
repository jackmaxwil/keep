#!/usr/bin/env python3
"""Validate the task and rendered config using the pinned SkyPilot parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sky
from sky import skypilot_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if sky.__version__ != "0.13.0":
        parser.error(f"SkyPilot must be 0.13.0, got {sky.__version__}")

    task = sky.Task.from_yaml(str(args.task))
    config = skypilot_config.parse_and_validate_config_file(str(args.config))
    resources = list(task.resources)
    if len(resources) != 1:
        parser.error("task must declare exactly one resource alternative")
    resource = resources[0]
    rendered = task.to_yaml_config()
    expected = {
        "infra": "aws/us-west-2",
        "instance_type": "p5.48xlarge",
        "use_spot": False,
        "max_hourly_cost": 55.04,
    }
    raw_resource = rendered.get("resources", {})
    for field, value in expected.items():
        if raw_resource.get(field) != value:
            parser.error(f"task resource {field} does not match {value!r}")
    if rendered.get("num_nodes") != 1:
        parser.error("task must request exactly one worker")
    if config.get("allowed_clouds") != ["aws"]:
        parser.error("SkyPilot config must allow AWS only")
    if config.get_nested(("aws", "use_ssm"), None) is not True:
        parser.error("SkyPilot config must enable AWS SSM")
    if config.get_nested(("jobs", "controller", "resources", "instance_type"), None) != (
        "c6a.xlarge"
    ):
        parser.error("Managed Jobs controller type is not pinned")
    print(
        json.dumps(
            {
                "record_type": "glm52_skypilot_parser_validation_v1",
                "skypilot_version": sky.__version__,
                "task": task.name,
                "instance_type": resource.instance_type,
                "region": resource.region,
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
