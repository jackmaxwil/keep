#!/usr/bin/env python3
"""Build the Task 13 base request and exact 21-coordinate input set."""

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
    assemble_prequalification_inputs,
    build_prequalification_base_request,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--collector-version-arn", required=True)
    parser.add_argument("--task11-request", required=True, type=Path)
    parser.add_argument("--task11-boundary", required=True, type=Path)
    parser.add_argument(
        "--staged-infrastructure-evidence",
        required=True,
        type=Path,
    )
    parser.add_argument("--retained-stack-id", required=True)
    parser.add_argument("--fence-change-set-name", required=True)
    parser.add_argument("--support-change-set-name", required=True)
    parser.add_argument("--monitor-descriptor-path", required=True)
    parser.add_argument(
        "--coordinate-source",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--base-request-output", required=True, type=Path)
    parser.add_argument("--coordinates-output", required=True, type=Path)
    parser.add_argument("--request-output", required=True, type=Path)
    parser.add_argument(
        "--reviewed-artifacts-output",
        required=True,
        type=Path,
    )
    return parser


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


def _write_all(outputs: list[tuple[Path, object]]) -> None:
    paths = [path for path, _value in outputs]
    resolved = [path.resolve(strict=False) for path in paths]
    if (
        len(set(resolved)) != len(paths)
        or paths != resolved
        or any(
            not path.is_absolute()
            or not path.parent.is_dir()
            or path.parent.is_symlink()
            or path.exists()
            or path.is_symlink()
            for path in paths
        )
    ):
        raise Task13LiveInputError(
            "outputs must be distinct new absolute files"
        )
    descriptors: list[int] = []
    created: list[Path] = []
    operation_error: BaseException | None = None
    try:
        for path, value in outputs:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            descriptors.append(descriptor)
            created.append(path)
            remaining = memoryview(canonical_json_bytes(value) + b"\n")
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError(
                        "prequalification output write made no progress"
                    )
                remaining = remaining[written:]
            os.fsync(descriptor)
    except BaseException as error:
        operation_error = error

    close_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if close_error is None:
                close_error = error
    primary_error = operation_error or close_error
    if primary_error is not None:
        for path in created:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise primary_error


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        base = build_prequalification_base_request(
            activation_id=args.activation_id,
            collector_version_arn=args.collector_version_arn,
            task11_request=_read(
                args.task11_request,
                "Task11 request",
            ),
            task11_boundary=_read(
                args.task11_boundary,
                "Task11 boundary",
            ),
            staged_infrastructure_evidence=_read(
                args.staged_infrastructure_evidence,
                "staged infrastructure evidence",
            ),
            retained_stack_id=args.retained_stack_id,
            fence_change_set_name=args.fence_change_set_name,
            support_change_set_name=args.support_change_set_name,
            monitor_descriptor_path=args.monitor_descriptor_path,
        )
        result = assemble_prequalification_inputs(
            base_request=base,
            coordinate_documents=[
                _read(path, "coordinate source")
                for path in args.coordinate_source
            ],
        )
        _write_all(
            [
                (args.base_request_output, result["base_request"]),
                (args.coordinates_output, result["coordinates"]),
                (args.request_output, result["request"]),
                (
                    args.reviewed_artifacts_output,
                    result["reviewed_artifacts"],
                ),
            ]
        )
    except (OSError, Task13LiveInputError, ValueError) as error:
        print(
            "Task 13 prequalification-source build refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    for path in (
        args.base_request_output,
        args.coordinates_output,
        args.request_output,
        args.reviewed_artifacts_output,
    ):
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
