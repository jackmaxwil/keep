#!/usr/bin/env python3
"""Build the production request from proven clean evidence and Task 10 inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_live_inputs import (  # noqa: E402
    Task13LiveInputError,
    build_production_successor_request,
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
        raise Task13LiveInputError(
            label + " is not canonical JSON plus LF"
        )
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
    operation_error = None
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        created = True
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(canonical_json_bytes(value) + b"\n")
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("successor output write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error

    close_error = None
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError as error:
            close_error = error
    if operation_error is not None or close_error is not None:
        if created:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as error:
                close_error = close_error or error
        if close_error is not None:
            if operation_error is not None:
                raise close_error from operation_error
            raise close_error
        if operation_error is not None:
            raise operation_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prequalification-request",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--prequalification-package",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--prequalification-reviewed-artifacts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--production-reviewed-artifacts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--clean-rehearsal-evidence",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        value = build_production_successor_request(
            prequalification_request=_read(
                args.prequalification_request,
                "prequalification request",
            ),
            prequalification_package=_read(
                args.prequalification_package,
                "prequalification package",
            ),
            prequalification_reviewed_artifacts=_read(
                args.prequalification_reviewed_artifacts,
                "prequalification reviewed artifacts",
            ),
            production_reviewed_artifacts=_read(
                args.production_reviewed_artifacts,
                "production reviewed artifacts",
            ),
            clean_rehearsal_evidence=_read(
                args.clean_rehearsal_evidence,
                "clean rehearsal evidence",
            ),
        )
        _write_new(args.output, value)
    except (OSError, Task13LiveInputError, ValueError) as error:
        print(
            "Task 13 production-successor build refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
