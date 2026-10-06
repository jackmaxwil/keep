#!/usr/bin/env python3
"""Build the two exact Task 13 drivers closed over one fresh archive."""

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
    build_fresh_archive_activation_driver_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--repository-archive", required=True, type=Path)
    parser.add_argument("--coordinator", required=True, type=Path)
    parser.add_argument(
        "--h100-campaign-descriptor",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cache-seed-campaign-descriptor",
        required=True,
        type=Path,
    )
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--staged-ready", required=True, type=Path)
    parser.add_argument("--staged-ready-version-id", required=True)
    parser.add_argument("--rehearsal-evidence", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sky-bin", required=True, type=Path)
    parser.add_argument("--h100-output", required=True, type=Path)
    parser.add_argument("--cache-seed-output", required=True, type=Path)
    return parser


def _validate_new_output(path: Path, label: str) -> Path:
    exact = Path(path)
    if (
        not exact.is_absolute()
        or not exact.parent.is_dir()
        or exact.parent.is_symlink()
    ):
        raise Task13LiveInputError(
            label
            + " must be an absolute path under an existing real directory"
        )
    if os.path.lexists(exact):
        raise Task13LiveInputError(label + " already exists")
    return exact


def _write_new_pair(
    *,
    h100_path: Path,
    h100_value: object,
    cache_seed_path: Path,
    cache_seed_value: object,
) -> None:
    h100 = _validate_new_output(h100_path, "H100 output")
    cache_seed = _validate_new_output(
        cache_seed_path,
        "cache-seed output",
    )
    if h100 == cache_seed:
        raise Task13LiveInputError("driver outputs must be distinct")
    values = (
        (h100, h100_value),
        (cache_seed, cache_seed_value),
    )
    descriptors: list[int] = []
    created_paths: list[Path] = []
    operation_error: BaseException | None = None
    try:
        for path, _value in values:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            descriptors.append(descriptor)
            created_paths.append(path)
            os.fchmod(descriptor, 0o600)
        for descriptor, (_path, value) in zip(descriptors, values):
            remaining = memoryview(canonical_json_bytes(value) + b"\n")
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("driver output write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
    except BaseException as error:
        operation_error = error

    cleanup_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as close_error:
            cleanup_error = cleanup_error or close_error
    if operation_error is not None or cleanup_error is not None:
        for path in created_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as unlink_error:
                cleanup_error = cleanup_error or unlink_error
    if cleanup_error is not None:
        if operation_error is not None:
            raise cleanup_error from operation_error
        raise cleanup_error
    if operation_error is not None:
        raise operation_error


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_fresh_archive_activation_driver_inputs(
            activation_id=args.activation_id,
            repository_archive_path=args.repository_archive,
            coordinator_path=args.coordinator,
            h100_campaign_descriptor=args.h100_campaign_descriptor,
            cache_seed_campaign_descriptor=(
                args.cache_seed_campaign_descriptor
            ),
            approval=args.approval,
            staged_ready=args.staged_ready,
            staged_ready_version_id=args.staged_ready_version_id,
            rehearsal_evidence=args.rehearsal_evidence,
            task=args.task,
            config=args.config,
            sky_bin=args.sky_bin,
        )
        _write_new_pair(
            h100_path=args.h100_output,
            h100_value=result.h100_driver_request,
            cache_seed_path=args.cache_seed_output,
            cache_seed_value=result.cache_seed_driver_request,
        )
    except (OSError, Task13LiveInputError, ValueError) as error:
        print(
            f"Task 13 fresh driver-input build refused: {error}",
            file=sys.stderr,
        )
        return 64
    print(str(args.h100_output))
    print(str(args.cache_seed_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
