#!/usr/bin/env python3
"""Build the exact additive 21-to-25 production reviewed-artifact list."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_live_inputs import (  # noqa: E402
    Task13LiveInputError,
    build_production_reviewed_artifacts,
)


def _read(path: Path, label: str) -> object:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Task13LiveInputError(
            label + " must be an absolute regular non-symlink file"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Task13LiveInputError(label + " is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise Task13LiveInputError(label + " is not canonical JSON plus LF")
    return value


def _write_new(path: Path, value: object) -> None:
    if (
        not path.is_absolute()
        or path.resolve(strict=False) != path
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or os.path.lexists(path)
    ):
        raise Task13LiveInputError("output must be one new absolute file")

    descriptor = None
    created = False
    operation_error: BaseException | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(canonical_json_bytes(value) + b"\n")
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0 or written > len(remaining):
                raise OSError("production reviewed output write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error

    close_error: BaseException | None = None
    if descriptor is not None:
        try:
            os.close(descriptor)
        except BaseException as error:
            close_error = error

    failure = close_error or operation_error
    if failure is None:
        return
    if created:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise cleanup_error from failure
    if close_error is not None:
        if operation_error is not None:
            raise close_error from operation_error
        raise close_error
    if operation_error is not None:
        raise operation_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prequalification-reviewed-artifacts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--clean-rehearsal-coordinate",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--task10-production-authority",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--task10-worker-descriptor",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--task10-task-inputs",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        reviewed = build_production_reviewed_artifacts(
            prequalification_reviewed_artifacts=_read(
                args.prequalification_reviewed_artifacts,
                "prequalification reviewed artifacts",
            ),
            clean_rehearsal_coordinate=_read(
                args.clean_rehearsal_coordinate,
                "clean rehearsal coordinate",
            ),
            task10_production_authority=_read(
                args.task10_production_authority,
                "Task10 production authority coordinate",
            ),
            task10_worker_descriptor=_read(
                args.task10_worker_descriptor,
                "Task10 worker descriptor coordinate",
            ),
            task10_task_inputs=_read(
                args.task10_task_inputs,
                "Task10 task inputs coordinate",
            ),
        )
        _write_new(args.output, reviewed)
    except (OSError, Task13LiveInputError, ValueError) as error:
        print(
            "production reviewed-artifact build refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
