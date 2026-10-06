from __future__ import annotations

from datetime import datetime, timezone

from mlx_vq.quality.glm52_capacity_monitor import is_more_appealing


def test_more_appealing_capacity_definition_is_fail_closed() -> None:
    approved_start = datetime(2026, 7, 17, 11, 30, tzinfo=timezone.utc)
    offering = {
        "StartDate": "2026-07-16T11:30:00Z",
        "CapacityBlockDurationHours": 24,
        "UpfrontFee": "900.00",
        "AvailabilityZone": "us-west-2a",
        "InstanceType": "p5.48xlarge",
    }
    assert is_more_appealing(
        offering,
        approved_start=approved_start,
        approved_hourly_cost=1993.34 / 48,
        staged_availability_zones={"us-west-2a"},
    )
    assert not is_more_appealing(
        {**offering, "StartDate": "2026-07-17T12:00:00Z"},
        approved_start=approved_start,
        approved_hourly_cost=1993.34 / 48,
        staged_availability_zones={"us-west-2a"},
    )
    assert not is_more_appealing(
        {**offering, "UpfrontFee": "2000.00"},
        approved_start=approved_start,
        approved_hourly_cost=1993.34 / 48,
        staged_availability_zones={"us-west-2a"},
    )
