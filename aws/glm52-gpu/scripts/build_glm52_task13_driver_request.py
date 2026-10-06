#!/usr/bin/env python3
"""Build one exact local Task 13 repository-driver request."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_live_inputs import (  # noqa: E402
    Task13LiveInputError,
    build_cache_seed_driver_request,
    build_h100_driver_request,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    cache = subparsers.add_parser("cache-seed")
    cache.add_argument("--campaign-descriptor", required=True, type=Path)
    cache.add_argument("--approval", required=True, type=Path)
    cache.add_argument("--staged-ready", required=True, type=Path)
    cache.add_argument("--staged-ready-version-id", required=True)
    cache.add_argument("--rehearsal-evidence", required=True, type=Path)
    cache.add_argument("--task", required=True, type=Path)
    cache.add_argument("--config", required=True, type=Path)
    cache.add_argument("--sky-bin", required=True, type=Path)
    h100 = subparsers.add_parser("h100")
    h100.add_argument("--campaign-descriptor", required=True, type=Path)
    for child in (cache, h100):
        child.add_argument("--activation-id", required=True)
        child.add_argument("--output", required=True, type=Path)
    return parser


def _write_new(path: Path, value: object) -> None:
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise Task13LiveInputError(
            "output must be an absolute path under an existing real directory"
        )
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "cache-seed":
            value = build_cache_seed_driver_request(
                activation_id=args.activation_id,
                campaign_descriptor=args.campaign_descriptor,
                approval=args.approval,
                staged_ready=args.staged_ready,
                staged_ready_version_id=args.staged_ready_version_id,
                rehearsal_evidence=args.rehearsal_evidence,
                task=args.task,
                config=args.config,
                sky_bin=args.sky_bin,
            )
        else:
            value = build_h100_driver_request(
                activation_id=args.activation_id,
                campaign_descriptor=args.campaign_descriptor,
            )
        _write_new(args.output, value)
    except (OSError, Task13LiveInputError, ValueError) as error:
        print(
            "Task 13 driver-request build refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
