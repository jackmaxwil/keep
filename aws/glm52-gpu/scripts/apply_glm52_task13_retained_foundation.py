#!/usr/bin/env python3
"""Apply the guarded Task 13 retained Phase-1 foundation once."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Callable
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

_foundation = importlib.import_module("glm52_enforcement.task13_retained_foundation")
PROFILE = _foundation.PROFILE
REGION = _foundation.REGION
FoundationError = _foundation.FoundationError
FoundationServices = _foundation.FoundationServices
apply_retained_foundation = _foundation.apply_retained_foundation
recover_retained_foundation = _foundation.recover_retained_foundation


def _build_services(
    *,
    session_factory: Callable[..., object] | None = None,
    config_factory: Callable[..., object] | None = None,
) -> FoundationServices:
    if session_factory is None or config_factory is None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("boto3 and botocore are required") from exc
        session_factory = boto3.Session
        config_factory = Config
    config = config_factory(
        region_name=REGION,
        connect_timeout=5,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = session_factory(
        profile_name=PROFILE,
        region_name=REGION,
    )
    if getattr(session, "region_name", None) != REGION:
        raise FoundationError("AWS session region drifted")
    client = getattr(session, "client", None)
    if not callable(client):
        raise FoundationError("AWS session client factory is absent")
    return FoundationServices(
        sts=client("sts", config=config),
        cloudformation=client("cloudformation", config=config),
        iam=client("iam", config=config),
        cloudtrail=client("cloudtrail", config=config),
        dynamodb=client("dynamodb", config=config),
        kms=client("kms", config=config),
        s3=client("s3", config=config),
        total_max_attempts=1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
        help="Evidence directory (empty for apply; existing for recovery)",
    )
    parser.add_argument(
        "--recover-change-set-evidence",
        type=Path,
        help=(
            "Recover the missing final readback from this authenticated "
            "change-set evidence without creating or executing a change set"
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    services_factory: Callable[[], FoundationServices] = _build_services,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = _parser().parse_args(argv)
    operation = "recovery" if args.recover_change_set_evidence else "update"
    try:
        services = services_factory()
        if args.recover_change_set_evidence is None:
            result = apply_retained_foundation(
                services,
                output_directory=args.output_directory,
                sleep=sleep,
                max_polls=180,
            )
        else:
            result = recover_retained_foundation(
                services,
                change_set_evidence=args.recover_change_set_evidence,
                readback_output=(
                    args.output_directory / "retained-foundation-recovery-readback.json"
                ),
            )
    except (
        BotoCoreError,
        ClientError,
        FoundationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"Task 13 retained-foundation {operation} refused: {exc}",
            file=sys.stderr,
        )
        return 64
    print(f"{result['status']} {result['stack_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
