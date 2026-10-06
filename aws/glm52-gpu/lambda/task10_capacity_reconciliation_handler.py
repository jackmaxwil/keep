"""Task 10 sole-sender Lambda for durable internal capacity outcomes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task10_capacity_reconciliation import (  # noqa: E402
    build_capacity_reconciliation,
    publish_capacity_reconciliation,
)


def main(event: object, context: object, *, s3_client: object | None = None) -> Mapping[str, object]:
    del context
    if type(event) is not dict:
        raise ValueError("Task 10 reconciliation event must be an object")
    value = build_capacity_reconciliation(**event)
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name="us-west-2")
    return publish_capacity_reconciliation(s3_client, value=value)
