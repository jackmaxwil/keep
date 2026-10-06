#!/usr/bin/env python3
"""Build the exact Task 10 authority-materialization manifest."""

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
    build_task10_materialization_manifest,
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


def _journal(path: Path) -> list[dict[str, object]]:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Task13LiveInputError(
            "campaign journal must be an absolute regular non-symlink file"
        )
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise Task13LiveInputError("campaign journal is not canonical JSONL")
    rows = []
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise Task13LiveInputError(
                "campaign journal is not JSONL"
            ) from error
        if (
            type(value) is not dict
            or line != canonical_json_bytes(value) + b"\n"
        ):
            raise Task13LiveInputError(
                "campaign journal is not canonical JSONL"
            )
        rows.append(value)
    return rows


def _write_new(path: Path, value: object) -> None:
    if (
        not path.is_absolute()
        or path.resolve(strict=False) != path
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        raise Task13LiveInputError("output must be one new absolute file")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    operation_error: BaseException | None = None
    try:
        remaining = memoryview(canonical_json_bytes(value) + b"\n")
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(
                    "materialization output write made no progress"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error

    close_error: OSError | None = None
    try:
        os.close(descriptor)
    except OSError as error:
        close_error = error
    primary_error = operation_error or close_error
    if primary_error is not None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise primary_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--task10-task-inputs", required=True, type=Path)
    parser.add_argument(
        "--task10-worker-descriptor",
        required=True,
        type=Path,
    )
    parser.add_argument("--h100-stage-result", required=True, type=Path)
    parser.add_argument("--campaign-journal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        value = build_task10_materialization_manifest(
            prequalification_package=_read(
                args.prequalification_package,
                "prequalification package",
            ),
            prequalification_reviewed_artifacts=_read(
                args.prequalification_reviewed_artifacts,
                "prequalification reviewed artifacts",
            ),
            task10_task_inputs=_read(
                args.task10_task_inputs,
                "Task10 task inputs",
            ),
            task10_worker_descriptor=_read(
                args.task10_worker_descriptor,
                "Task10 worker descriptor",
            ),
            h100_stage_result=_read(
                args.h100_stage_result,
                "H100 stage result",
            ),
            campaign_journal_rows=_journal(args.campaign_journal),
            repository_root=REPO_ROOT,
        )
        _write_new(args.output, value)
    except (KeyError, OSError, Task13LiveInputError, ValueError) as error:
        print(
            "Task 10 materialization-manifest build refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
