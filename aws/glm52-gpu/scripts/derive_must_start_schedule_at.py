"""Derive EventBridge Scheduler at() time from one canonical UTC deadline."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone


def derive_schedule_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"must_start_by is not ISO-8601: {error}") from error
    if (
        parsed.tzinfo is None
        or parsed.astimezone(timezone.utc) != parsed
        or parsed.microsecond != 0
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise ValueError(
            "must_start_by must use exact second-resolution "
            "YYYY-MM-DDTHH:MM:SSZ"
        )
    return (
        (parsed - timedelta(minutes=1))
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("must_start_by")
    args = parser.parse_args()
    print(derive_schedule_at(args.must_start_by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
