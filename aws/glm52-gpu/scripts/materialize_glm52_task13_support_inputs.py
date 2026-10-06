#!/usr/bin/env python3
"""Collect and write authenticated Task 13 support pre-create inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task13_support_input_materialization import (  # noqa: E402
    PROFILE,
    REGION,
    SupportInputServices,
    collect_support_build_inputs,
    read_canonical_support_input_materialization_request,
    write_support_build_inputs_once,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize exact Task13 SupportBuildInputs from authenticated "
            "AWS facts"
        )
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _build_services(
    *,
    session_factory: Callable[..., object] | None = None,
) -> SupportInputServices:
    os.environ["AWS_MAX_ATTEMPTS"] = "1"
    os.environ["AWS_RETRY_MODE"] = "standard"
    if session_factory is None:
        import boto3

        session_factory = boto3.Session
    from botocore.config import Config

    config = Config(
        region_name=REGION,
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=15,
    )
    collection_deadline = time.monotonic() + 300
    session = session_factory(
        profile_name=PROFILE,
        region_name=REGION,
    )

    def client(name: str) -> object:
        factory = getattr(session, "client", None)
        if not callable(factory):
            raise TypeError("AWS session has no typed client factory")
        return factory(name, region_name=REGION, config=config)

    return SupportInputServices(
        sts=client("sts"),
        organizations=client("organizations"),
        cloudformation=client("cloudformation"),
        ec2=client("ec2"),
        kms=client("kms"),
        dynamodb=client("dynamodb"),
        s3=client("s3"),
        lambda_client=client("lambda"),
        total_max_attempts=1,
        remaining_time_in_millis=lambda: max(
            0,
            int((collection_deadline - time.monotonic()) * 1000),
        ),
    )


def main(
    argv: list[str] | None = None,
    *,
    session_factory: Callable[..., object] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    request = read_canonical_support_input_materialization_request(
        args.request
    )
    services = _build_services(session_factory=session_factory)
    inputs = collect_support_build_inputs(
        request=request,
        services=services,
    )
    write_support_build_inputs_once(args.output, inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
