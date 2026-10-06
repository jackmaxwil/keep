#!/usr/bin/env python3
"""Accept the immutable current qualification-cache seed into a local record."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlx_vq.quality.glm52_qualification_cache_seed import (  # noqa: E402
    QualificationCacheSeedPins,
    authenticate_qualification_cache_seed,
    validate_approved_identity,
    validate_qualification_cache_seed_accepted,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--managed-mode", required=True, choices=["cache-seed"])
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--seed-descriptor-key", required=True)
    parser.add_argument("--seed-descriptor-file-sha256", required=True)
    parser.add_argument("--seed-descriptor-body-sha256", required=True)
    parser.add_argument("--campaign-identity-sha256", required=True)
    parser.add_argument("--repo-tar-sha256", required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--submission-key", required=True)
    parser.add_argument("--submission-file-sha256", required=True)
    parser.add_argument("--submission-body-sha256", required=True)
    parser.add_argument("--seed-ready-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_file(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _write_atomic(path: Path, value: object) -> None:
    if path.is_symlink():
        raise ValueError("output path must not be a symlink")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be an existing non-symlink directory")
    raw = _canonical_file(value)
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    written = path.read_bytes()
    if written != raw:
        raise ValueError("atomic cache-seed acceptance output changed after write")
    validate_qualification_cache_seed_accepted(json.loads(written))


def main(
    argv: list[str] | None = None,
    *,
    session_factory: Callable[..., object] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.profile != "keep-gpu":
        parser.error("--profile must be exactly keep-gpu")
    if args.job_id <= 0:
        parser.error("--job-id must be a positive integer")
    if session_factory is None:
        import boto3

        session_factory = boto3.Session

    pins = QualificationCacheSeedPins(
        account_id="246813579024",
        region=args.region,
        bucket=args.bucket,
        run_id=args.run_id,
        managed_mode=args.managed_mode,
        target_job_id=args.job_id,
        descriptor_key=args.seed_descriptor_key,
        descriptor_file_sha256=args.seed_descriptor_file_sha256,
        descriptor_body_sha256=args.seed_descriptor_body_sha256,
        campaign_identity_sha256=args.campaign_identity_sha256,
        repo_tar_sha256=args.repo_tar_sha256,
        approval_sha256=args.approval_sha256,
        submission_key=args.submission_key,
        submission_file_sha256=args.submission_file_sha256,
        submission_body_sha256=args.submission_body_sha256,
        ready_key=args.seed_ready_key,
    )
    pins.validate()
    session = session_factory(
        profile_name=args.profile,
        region_name=args.region,
    )
    # This is intentionally repeated here, immediately before the first S3
    # client is created or any S3 read can occur.
    sts = session.client("sts")  # type: ignore[attr-defined]
    validate_approved_identity(sts.get_caller_identity())
    s3 = session.client("s3")  # type: ignore[attr-defined]
    with tempfile.TemporaryDirectory(
        prefix="glm52-qualification-cache-seed-accept-"
    ) as directory:
        accepted = authenticate_qualification_cache_seed(
            s3_client=s3,
            pins=pins,
            work_dir=Path(directory),
            accepted_at=datetime.now(timezone.utc),
        )
    _write_atomic(args.output, accepted)
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(64) from error
