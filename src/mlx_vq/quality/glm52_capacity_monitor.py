"""Pure qualification rule for the read-only P5 capacity monitor."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

INSTANCE_TYPES = ("p5.48xlarge", "p5e.48xlarge", "p5en.48xlarge")


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("capacity time must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def is_more_appealing(
    offering: Mapping[str, Any],
    *,
    approved_start: datetime,
    approved_hourly_cost: float,
    staged_availability_zones: set[str],
) -> bool:
    try:
        start = _utc(str(offering["StartDate"]))
        duration = int(offering["CapacityBlockDurationHours"])
        fee = float(offering["UpfrontFee"])
        zone = str(offering["AvailabilityZone"])
        instance_type = str(offering["InstanceType"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        start < approved_start
        and duration >= 24
        and instance_type in INSTANCE_TYPES
        and zone in staged_availability_zones
        and fee / duration <= approved_hourly_cost
    )


__all__ = ["INSTANCE_TYPES", "is_more_appealing"]
