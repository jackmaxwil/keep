#!/usr/bin/env python3
"""Publish one externally pinned kind from the closed Task 13 reviewed map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task13_reviewed_artifacts import (  # noqa: E402
    PROFILE,
    REGION,
    Task13ReviewedArtifactError,
    Task13ReviewedArtifactServices,
    publish_reviewed_artifact,
    write_coordinate_once,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-kind", required=True)
    parser.add_argument("--activation-id")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-body-sha256", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--coordinate-output", required=True, type=Path)
    return parser


def _build_services() -> Task13ReviewedArtifactServices:
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
        raise Task13ReviewedArtifactError("AWS session region drifted")
    return Task13ReviewedArtifactServices(
        sts=session.client("sts", config=config),
        s3=session.client("s3", config=config),
        total_max_attempts=1,
    )


def main(
    argv: list[str] | None = None,
    *,
    services_factory: Callable[
        [], Task13ReviewedArtifactServices
    ] = _build_services,
) -> int:
    args = _parser().parse_args(argv)
    try:
        output = args.coordinate_output
        if (
            not output.is_absolute()
            or not output.parent.is_dir()
            or output.parent.is_symlink()
            or output.exists()
            or output.is_symlink()
        ):
            raise Task13ReviewedArtifactError(
                "coordinate output path is not one new absolute file"
            )
        coordinate = publish_reviewed_artifact(
            artifact_kind=args.artifact_kind,
            source_path=args.source,
            expected_file_sha256=args.expected_file_sha256,
            expected_body_sha256=args.expected_body_sha256,
            bucket=args.bucket,
            services=services_factory(),
            activation_id=args.activation_id,
        )
        write_coordinate_once(output, coordinate)
    except (
        OSError,
        RuntimeError,
        Task13ReviewedArtifactError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "Task 13 reviewed-artifact publication refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.coordinate_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
