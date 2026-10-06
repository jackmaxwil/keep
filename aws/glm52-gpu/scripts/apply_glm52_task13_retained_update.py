#!/usr/bin/env python3
"""Apply one guarded additive Task 13 retained-stack update."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

_update = importlib.import_module("glm52_enforcement.task13_retained_update")
PROFILE = _update.PROFILE
REGION = _update.REGION
RetainedUpdateError = _update.RetainedUpdateError
RetainedUpdateServices = _update.RetainedUpdateServices
apply_retained_update = _update.apply_retained_update
apply_intentional_drift_settlement = _update.apply_intentional_drift_settlement
canonical_json_bytes = _update.canonical_json_bytes


def _build_services(
    *,
    session_factory: Callable[..., object] | None = None,
    config_factory: Callable[..., object] | None = None,
) -> RetainedUpdateServices:
    if session_factory is None or config_factory is None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover
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
        raise RetainedUpdateError("AWS session region drifted")
    client = getattr(session, "client", None)
    if not callable(client):
        raise RetainedUpdateError("AWS session client factory is absent")
    return RetainedUpdateServices(
        sts=client("sts", config=config),
        cloudformation=client("cloudformation", config=config),
        ec2=client("ec2", config=config),
        ssm=client("ssm", config=config),
        iam=client("iam", config=config),
        s3=client("s3", config=config),
        total_max_attempts=1,
    )


def _read_fragment(path: Path) -> Mapping[str, object]:
    source = Path(path)
    if (
        not source.is_absolute()
        or not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
    ):
        raise RetainedUpdateError(
            "fragment must be one absolute regular non-symlink file"
        )
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RetainedUpdateError("fragment is not canonical UTF-8 JSON") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise RetainedUpdateError("fragment is not one canonical JSON object plus LF")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "pre-support",
            "intentional-drift-settlement-v1",
            "fence-bootstrap-v9",
            "final",
        ),
        help="Closed retained update phase and immutable template key",
    )
    parser.add_argument(
        "--fragment",
        type=Path,
        help="Absolute canonical additive fragment JSON path",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
        help=(
            "Production output root for drift settlement or fence-bootstrap-v9; "
            "existing phase evidence directory otherwise"
        ),
    )
    return parser


def _evidence_directory(phase: str, output_directory: Path) -> Path:
    root = Path(output_directory)
    stems = {
        "intentional-drift-settlement-v1": ("retained-intentional-drift-settlement-v1"),
        "fence-bootstrap-v9": "retained-fence-bootstrap-v9",
    }
    if phase not in stems:
        return root
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise RetainedUpdateError(
            f"{phase} production output root is not one safe directory"
        )
    evidence = root / stems[phase]
    evidence.mkdir(mode=0o700, exist_ok=True)
    if evidence.is_symlink() or not evidence.is_dir():
        raise RetainedUpdateError(
            f"{phase} evidence directory is not one safe directory"
        )
    return evidence


def main(
    argv: list[str] | None = None,
    *,
    services_factory: Callable[[], RetainedUpdateServices] = _build_services,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = _parser().parse_args(argv)
    try:
        output_directory = _evidence_directory(
            args.phase,
            args.output_directory,
        )
        if args.phase == "intentional-drift-settlement-v1":
            if args.fragment is not None:
                raise RetainedUpdateError(
                    "intentional drift settlement accepts no fragment"
                )
            result = apply_intentional_drift_settlement(
                services_factory(),
                output_directory=output_directory,
                sleep=sleep,
                max_polls=180,
            )
        else:
            if args.fragment is None:
                raise RetainedUpdateError(
                    "additive retained phase requires one fragment"
                )
            result = apply_retained_update(
                services_factory(),
                phase=args.phase,
                fragment=_read_fragment(args.fragment),
                output_directory=output_directory,
                sleep=sleep,
                max_polls=180,
            )
    except (
        BotoCoreError,
        ClientError,
        OSError,
        RetainedUpdateError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"Task 13 retained {args.phase} update refused: {exc}",
            file=sys.stderr,
        )
        return 64
    print(f"{result['status']} {result['stack_id']} {result['template_version_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
