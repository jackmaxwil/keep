#!/usr/bin/env python3
"""Fetch, build, and explicitly publish exact Task 13 support artifacts."""

from __future__ import annotations

import argparse
import sys
import urllib.error
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task13_fixed_artifacts import (  # noqa: E402
    PROFILE,
    REGION,
    Task13FixedArtifactError,
    Task13FixedArtifactServices,
    write_coordinate_once,
)
from glm52_enforcement.task13_support_artifacts import (  # noqa: E402
    Task13SupportArtifactError,
    fetch_pinned_wheels,
    materialize_support_artifacts,
    publish_support_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)

    fetch = operations.add_parser("fetch-wheels")
    fetch.add_argument("--wheelhouse", required=True, type=Path)

    materialize = operations.add_parser("materialize")
    materialize.add_argument("--wheelhouse", required=True, type=Path)
    materialize.add_argument("--support-output", required=True, type=Path)
    materialize.add_argument("--layer-output", required=True, type=Path)
    materialize.add_argument("--manifest-output", required=True, type=Path)

    publish = operations.add_parser("publish")
    publish.add_argument("--manifest", required=True, type=Path)
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--coordinate-output", required=True, type=Path)
    return parser


def _build_services() -> Task13FixedArtifactServices:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        region_name=REGION,
        connect_timeout=5,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )
    if session.region_name != REGION:
        raise Task13SupportArtifactError("AWS session region drifted")
    return Task13FixedArtifactServices(
        sts=session.client("sts", config=config),
        s3=session.client("s3", config=config),
        total_max_attempts=1,
    )


def main(
    argv: list[str] | None = None,
    *,
    services_factory: Callable[[], Task13FixedArtifactServices] = (_build_services),
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "fetch-wheels":
            fetch_pinned_wheels(args.wheelhouse)
            result_path = args.wheelhouse
        elif args.operation == "materialize":
            materialize_support_artifacts(
                wheelhouse=args.wheelhouse,
                support_output=args.support_output,
                layer_output=args.layer_output,
                manifest_output=args.manifest_output,
            )
            result_path = args.manifest_output
        else:
            coordinate = publish_support_artifacts(
                manifest_path=args.manifest,
                bucket=args.bucket,
                services=services_factory(),
            )
            write_coordinate_once(args.coordinate_output, coordinate)
            result_path = args.coordinate_output
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        Task13FixedArtifactError,
        Task13SupportArtifactError,
        TypeError,
        urllib.error.URLError,
        ValueError,
    ) as error:
        print(
            f"Task 13 support-artifact materialization refused: {error}",
            file=sys.stderr,
        )
        return 64
    print(str(result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
