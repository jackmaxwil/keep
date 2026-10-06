#!/usr/bin/env python3
"""Build an exact local post-seed cumulative GPU spend snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SNAPSHOT_MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_gpu_spend_snapshot.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_module(name: str, path: Path):  # noqa: ANN202
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


SNAPSHOT = _load_module("_glm52_gpu_spend_snapshot", SNAPSHOT_MODULE_PATH)
SKY = importlib.import_module("mlx_vq.quality.glm52_sky_campaign")


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _read_records(path: Path) -> dict[str, bytes]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(
            "GPU spend records directory must be a regular non-symlink directory"
        )
    records: dict[str, bytes] = {}
    for entry in path.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("GPU spend records directory contains a non-regular entry")
        records[entry.name] = entry.read_bytes()
    return records


def _guard_account(profile: str) -> None:
    if profile != "keep-gpu":
        raise ValueError("--profile must be exactly keep-gpu")
    result = subprocess.run(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--profile",
            profile,
            "--output",
            "json",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "AWS_PAGER": ""},
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(
            detail or "AWS STS account guard failed without diagnostic output"
        )
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("AWS STS account guard returned malformed JSON") from error
    if not isinstance(identity, dict):
        raise ValueError("AWS STS account guard returned a non-object")
    SKY.require_approved_aws_identity(identity)


def _write_atomic(path: Path, raw: bytes) -> None:
    if path.name != "GPU_SPEND_SNAPSHOT.json":
        raise ValueError("output filename must be GPU_SPEND_SNAPSHOT.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--descriptor-sha256", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--ledger-latest", type=Path, required=True)
    parser.add_argument("--ledger-latest-sha256", required=True)
    parser.add_argument("--ledger-records-dir", type=Path, required=True)
    parser.add_argument(
        "--ec2-allocation-history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ec2-allocation-history-sha256",
        required=True,
    )
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _guard_account(args.profile)
        descriptor_raw = _read_regular(
            args.descriptor,
            label="Sky campaign descriptor",
        )
        approval_raw = _read_regular(
            args.approval,
            label="GPU spend approval",
        )
        latest_raw = _read_regular(
            args.ledger_latest,
            label="GPU spend latest marker",
        )
        history_raw = _read_regular(
            args.ec2_allocation_history,
            label="EC2 allocation history",
        )
        records = _read_records(args.ledger_records_dir)
        observed_at = datetime.fromisoformat(args.observed_at.replace("Z", "+00:00"))
        snapshot = SNAPSHOT.build_gpu_spend_snapshot(
            descriptor_raw=descriptor_raw,
            expected_descriptor_sha256=args.descriptor_sha256,
            approval_raw=approval_raw,
            expected_approval_sha256=args.approval_sha256,
            latest_raw=latest_raw,
            expected_latest_sha256=args.ledger_latest_sha256,
            immutable_record_raw_by_key=records,
            ec2_allocation_history_raw=history_raw,
            expected_ec2_allocation_history_sha256=(args.ec2_allocation_history_sha256),
            observed_at=observed_at,
        )
        raw = (
            json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        _write_atomic(args.output, raw)
        final_raw = _read_regular(
            args.output,
            label="GPU spend snapshot",
        )
        if final_raw != raw:
            raise ValueError("GPU spend snapshot final bytes changed after write")
        final = json.loads(final_raw)
        SNAPSHOT.validate_gpu_spend_snapshot(final)
    except (
        json.JSONDecodeError,
        OSError,
        SKY.SkyCampaignValidationError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "gpu_spend_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                "snapshot_body_sha256": snapshot["snapshot_body_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
