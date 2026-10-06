#!/usr/bin/env python3
"""Build three reviewed fresh-archive prequalification coordinates locally."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_live_inputs import (  # noqa: E402
    Task13LiveInputError,
    build_fresh_archive_activation_driver_inputs,
    build_fresh_archive_prequalification_coordinates,
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
    parser.add_argument(
        "--repository-archive-coordinate",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--repository-archive-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--h100-driver-coordinate",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cache-seed-driver-coordinate",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read_canonical(path: Path, label: str) -> tuple[object, bytes]:
    exact = Path(path)
    if (
        not exact.is_absolute()
        or not exact.is_file()
        or exact.is_symlink()
        or exact.resolve(strict=True) != exact
    ):
        raise Task13LiveInputError(
            label + " must be an absolute regular non-symlink file"
        )
    raw = exact.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Task13LiveInputError(label + " is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise Task13LiveInputError(
            label + " is not canonical JSON plus LF"
        )
    return value, raw


def _write_new(path: Path, value: object) -> None:
    exact = Path(path)
    if (
        not exact.is_absolute()
        or exact.resolve(strict=False) != exact
        or not exact.parent.is_dir()
        or exact.parent.is_symlink()
        or os.path.lexists(exact)
    ):
        raise Task13LiveInputError("output must be one new absolute file")
    descriptor: Optional[int] = None
    created = False
    operation_error: Optional[BaseException] = None
    try:
        descriptor = os.open(
            exact,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        created = True
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(canonical_json_bytes(value) + b"\n")
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("coordinate output write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error

    close_error: Optional[OSError] = None
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError as error:
            close_error = error
    if operation_error is not None or close_error is not None:
        if created:
            try:
                os.unlink(exact)
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


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        archive_coordinate, _archive_coordinate_raw = _read_canonical(
            args.repository_archive_coordinate,
            "repository archive coordinate",
        )
        archive_manifest, archive_manifest_raw = _read_canonical(
            args.repository_archive_manifest,
            "repository archive manifest",
        )
        h100_coordinate, _h100_coordinate_raw = _read_canonical(
            args.h100_driver_coordinate,
            "H100 driver coordinate",
        )
        cache_seed_coordinate, _cache_seed_coordinate_raw = _read_canonical(
            args.cache_seed_driver_coordinate,
            "cache-seed driver coordinate",
        )
        fresh_inputs = build_fresh_archive_activation_driver_inputs(
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
        coordinates = build_fresh_archive_prequalification_coordinates(
            fresh_inputs=fresh_inputs,
            repository_archive_coordinate=archive_coordinate,
            repository_archive_manifest=archive_manifest,
            repository_archive_manifest_bytes=archive_manifest_raw,
            h100_driver_coordinate=h100_coordinate,
            cache_seed_driver_coordinate=cache_seed_coordinate,
        )
        _write_new(args.output, coordinates)
    except (OSError, Task13LiveInputError, ValueError) as error:
        print(
            "Fresh archive coordinate build refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
