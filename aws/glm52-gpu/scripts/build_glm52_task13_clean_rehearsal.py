#!/usr/bin/env python3
"""Build one local canonical Task 13 clean-rehearsal evidence record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_clean_rehearsal import (  # noqa: E402
    CleanRehearsalEvidenceError,
    build_clean_rehearsal_evidence,
    canonical_clean_rehearsal_evidence_bytes,
)


def _read_canonical(path: Path, label: str, *, lf: bool) -> tuple[object, bytes]:
    if not path.is_file() or path.is_symlink():
        raise CleanRehearsalEvidenceError(
            label + " must be one regular non-symlink file"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CleanRehearsalEvidenceError(label + " is not JSON") from error
    expected = canonical_json_bytes(value) + (b"\n" if lf else b"")
    if raw != expected:
        raise CleanRehearsalEvidenceError(
            label + " is not canonical JSON" + (" plus LF" if lf else "")
        )
    return value, raw


def _write_new(path: Path, raw: bytes) -> None:
    if not path.is_absolute():
        raise CleanRehearsalEvidenceError("output must be an absolute path")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise CleanRehearsalEvidenceError(
            "output parent must be one existing regular directory"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-package", required=True, type=Path)
    parser.add_argument(
        "--predecessor-reviewed-artifacts", required=True, type=Path
    )
    parser.add_argument(
        "--repository-archive-coordinate", required=True, type=Path
    )
    parser.add_argument(
        "--repository-archive-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--finalize-invoke-metadata", required=True, type=Path
    )
    parser.add_argument("--finalize-payload", required=True, type=Path)
    parser.add_argument("--gate-readback", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        predecessor, predecessor_raw = _read_canonical(
            args.predecessor_package,
            "predecessor package",
            lf=True,
        )
        reviewed, _reviewed_raw = _read_canonical(
            args.predecessor_reviewed_artifacts,
            "predecessor reviewed artifacts",
            lf=True,
        )
        archive_coordinate, _coordinate_raw = _read_canonical(
            args.repository_archive_coordinate,
            "repository archive coordinate",
            lf=True,
        )
        archive_manifest, archive_manifest_raw = _read_canonical(
            args.repository_archive_manifest,
            "repository archive manifest",
            lf=True,
        )
        finalizer_metadata, _metadata_raw = _read_canonical(
            args.finalize_invoke_metadata,
            "Task11 FINALIZE invoke metadata",
            lf=True,
        )
        finalizer_payload, finalizer_payload_raw = _read_canonical(
            args.finalize_payload,
            "Task11 FINALIZE payload",
            lf=False,
        )
        if (
            type(finalizer_metadata) is not dict
            or set(finalizer_metadata) != {"StatusCode", "ExecutedVersion"}
            or type(finalizer_payload) is not dict
        ):
            raise CleanRehearsalEvidenceError(
                "Task11 FINALIZE invoke input schema drifted"
            )
        gate, gate_raw = _read_canonical(
            args.gate_readback,
            "immutable gate readback",
            lf=False,
        )
        evidence = build_clean_rehearsal_evidence(
            predecessor_package=predecessor,
            predecessor_package_bytes=predecessor_raw,
            predecessor_reviewed_artifacts=reviewed,
            repository_archive_coordinate=archive_coordinate,
            repository_archive_manifest=archive_manifest,
            repository_archive_manifest_bytes=archive_manifest_raw,
            finalize_invoke_result={
                **finalizer_metadata,
                "Payload": finalizer_payload_raw,
            },
            gate_readback=gate,
            gate_readback_bytes=gate_raw,
        )
        _write_new(
            args.output,
            canonical_clean_rehearsal_evidence_bytes(evidence),
        )
    except (CleanRehearsalEvidenceError, FileExistsError, OSError) as error:
        print("Task 13 clean rehearsal refused: %s" % error, file=sys.stderr)
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
