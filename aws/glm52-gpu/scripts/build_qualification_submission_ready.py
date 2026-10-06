#!/usr/bin/env python3
"""Build the exact local authority for one post-seed qualification submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlx_vq.quality.glm52_qualification_submission_ready import (  # noqa: E402
    build_qualification_submission_ready,
    qualification_submission_ready_s3_key,
    validate_qualification_submission_ready,
)

OUTPUT_FILENAME = "QUALIFICATION_SUBMISSION_READY.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--descriptor-file-sha256", required=True)
    parser.add_argument("--seed-descriptor", type=Path, required=True)
    parser.add_argument("--seed-descriptor-file-sha256", required=True)
    parser.add_argument("--staged-readiness", type=Path, required=True)
    parser.add_argument("--staged-readiness-key", required=True)
    parser.add_argument("--staged-readiness-file-sha256", required=True)
    parser.add_argument("--cache-seed-acceptance", type=Path, required=True)
    parser.add_argument("--cache-seed-acceptance-key", required=True)
    parser.add_argument(
        "--cache-seed-acceptance-file-sha256",
        required=True,
    )
    parser.add_argument("--gpu-spend-snapshot", type=Path, required=True)
    parser.add_argument("--gpu-spend-snapshot-key", required=True)
    parser.add_argument("--gpu-spend-snapshot-sha256", required=True)
    parser.add_argument("--rehearsal-evidence", type=Path, required=True)
    parser.add_argument("--rehearsal-evidence-key", required=True)
    parser.add_argument("--rehearsal-evidence-sha256", required=True)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _load_object(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> dict[str, object]:
    raw = _read_regular(path, label=label)
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} SHA-256 must be lowercase hexadecimal")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid finite JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if raw != _canonical(value) + b"\n":
        raise ValueError(f"{label} bytes must be exact canonical JSON plus newline")
    return value


def _write_atomic(path: Path, raw: bytes) -> None:
    if path.name != OUTPUT_FILENAME:
        raise ValueError(f"output filename must be {OUTPUT_FILENAME}")
    if path.is_symlink():
        raise ValueError("output path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be a regular directory")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        published = True
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _authorities(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    descriptor = _load_object(
        args.descriptor,
        label="descriptor",
        expected_sha256=args.descriptor_file_sha256,
    )
    seed_raw = _read_regular(
        args.seed_descriptor,
        label="seed descriptor",
    )
    staged = _load_object(
        args.staged_readiness,
        label="staged readiness",
        expected_sha256=args.staged_readiness_file_sha256,
    )
    acceptance = _load_object(
        args.cache_seed_acceptance,
        label="cache-seed acceptance",
        expected_sha256=args.cache_seed_acceptance_file_sha256,
    )
    snapshot = _load_object(
        args.gpu_spend_snapshot,
        label="GPU spend snapshot",
        expected_sha256=args.gpu_spend_snapshot_sha256,
    )
    rehearsal = _load_object(
        args.rehearsal_evidence,
        label="rehearsal evidence",
        expected_sha256=args.rehearsal_evidence_sha256,
    )
    build_arguments: dict[str, object] = {
        "descriptor": descriptor,
        "seed_descriptor_raw": seed_raw,
        "seed_descriptor_file_sha256": (args.seed_descriptor_file_sha256),
        "staged_readiness": staged,
        "cache_seed_acceptance": acceptance,
        "gpu_spend_snapshot": snapshot,
        "rehearsal_evidence": rehearsal,
        "descriptor_file_sha256": args.descriptor_file_sha256,
        "staged_readiness_key": args.staged_readiness_key,
        "staged_readiness_file_sha256": (args.staged_readiness_file_sha256),
        "cache_seed_acceptance_key": args.cache_seed_acceptance_key,
        "cache_seed_acceptance_file_sha256": (args.cache_seed_acceptance_file_sha256),
        "gpu_spend_snapshot_key": args.gpu_spend_snapshot_key,
        "gpu_spend_snapshot_sha256": args.gpu_spend_snapshot_sha256,
        "rehearsal_evidence_key": args.rehearsal_evidence_key,
        "rehearsal_evidence_sha256": args.rehearsal_evidence_sha256,
        "built_at": args.built_at,
    }
    validation_arguments = {
        key: value
        for key, value in build_arguments.items()
        if key
        not in {
            "seed_descriptor_raw",
            "seed_descriptor_file_sha256",
            "built_at",
        }
    }
    validation_arguments["now"] = args.built_at
    return build_arguments, validation_arguments


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output_written = False
    try:
        build_arguments, validation_arguments = _authorities(args)
        ready = build_qualification_submission_ready(**build_arguments)
        raw = _canonical(ready) + b"\n"
        _write_atomic(args.output, raw)
        output_written = True
        written_raw = _read_regular(
            args.output,
            label="qualification submission readiness",
        )
        if written_raw != raw:
            raise ValueError(
                "qualification submission readiness changed after atomic write"
            )
        written = json.loads(written_raw)
        if not isinstance(written, Mapping):
            raise ValueError(
                "qualification submission readiness output is not an object"
            )
        validate_qualification_submission_ready(
            written,
            **validation_arguments,
        )
    except (OSError, TypeError, ValueError) as error:
        if output_written:
            args.output.unlink(missing_ok=True)
        parser.error(str(error))
    receipt = {
        "output": str(args.output.resolve()),
        "qualification_submission_ready_key": (
            qualification_submission_ready_s3_key(ready)
        ),
        "qualification_submission_ready_sha256": hashlib.sha256(raw).hexdigest(),
        "readiness_body_sha256": ready["readiness_body_sha256"],
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
