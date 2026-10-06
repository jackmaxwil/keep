"""AWS Lambda shim for the import-light retained lifecycle core."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.retained_support_lifecycle_handler import (  # noqa: E402
    RetainedLifecycleServices,
    main as evaluate,
)


def _services() -> RetainedLifecycleServices:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        connect_timeout=5,
        read_timeout=10,
        region_name="us-west-2",
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    return RetainedLifecycleServices(
        cloudwatch=boto3.client("cloudwatch", config=config),
        ec2=boto3.client("ec2", config=config),
    )


def main(event: object, context: object) -> Mapping[str, object]:
    return evaluate(event, context, services=_services())
