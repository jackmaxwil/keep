#!/usr/bin/env python3
"""Start or end an authenticated cumulative GPU allocation for one Sky job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
SPEC = importlib.util.spec_from_file_location("_glm52_sky_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
SKY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SKY
SPEC.loader.exec_module(SKY)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("allocation timestamps must be timezone-aware")
    return parsed


def _iso_utc(value: datetime) -> str:
    return value.astimezone(SKY.timezone.utc).isoformat().replace("+00:00", "Z")


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_atomic(path, _canonical_bytes(value) + b"\n")


def _materialize_ledger_publication(ledger) -> None:  # noqa: ANN001
    """Write content-addressed records first and the local latest marker last."""

    root = ledger.path.parent / "spend-ledger"
    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    record_keys: list[str] = []
    for index, record in enumerate(ledger.records):
        record_sha = str(record["record_sha256"])
        name = f"{index:06d}-{record['event']}-{record_sha}.json"
        destination = records_dir / name
        raw = _canonical_bytes(record) + b"\n"
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != raw:
                raise ValueError(f"immutable GPU spend record drift: {name}")
        else:
            _write_atomic(destination, raw)
        record_keys.append(name)
    ledger_raw = ledger.path.read_bytes()
    _write_atomic(root / "GPU_SPEND_LEDGER.jsonl", ledger_raw)
    latest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_ledger_latest_v1",
        "run_id": ledger.run_id,
        "gpu_spend_authority_sha256": ledger.genesis_sha256,
        "record_count": len(ledger.records),
        "record_keys": record_keys,
        "latest_record_sha256": (
            ledger.records[-1]["record_sha256"] if ledger.records else None
        ),
        "ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
    }
    latest = {
        **latest_body,
        "latest_body_sha256": hashlib.sha256(
            _canonical_bytes(latest_body)
        ).hexdigest(),
    }
    _write_json_atomic(root / "latest.json", latest)


def _load_authority(
    descriptor_path: Path, approval_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    descriptor = _read_object(descriptor_path, label="Sky campaign descriptor")
    approval = _read_object(approval_path, label="GPU spend approval")
    SKY.validate_gpu_spend_approval(approval)
    if descriptor.get("record_type") == "glm52_sky_campaign_descriptor_v2":
        SKY.validate_sky_campaign_descriptor(descriptor)
    required = {
        "run_id",
        "approval_sha256",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "max_hourly_cost_usd",
    }
    if not required.issubset(descriptor):
        raise ValueError("Sky campaign descriptor lacks spend authority")
    if descriptor["approval_sha256"] != _sha256_file(approval_path):
        raise ValueError("GPU spend approval file SHA-256 mismatch")
    return descriptor, approval


def _ledger(
    descriptor: dict[str, object],
    *,
    approval_path: Path,
    root: Path,
):
    return SKY.GpuSpendLedger(
        root / "runtime/GPU_SPEND_LEDGER.jsonl",
        run_id=str(descriptor["run_id"]),
        approval_sha256=_sha256_file(approval_path),
        approved_gpu_runtime_seconds=int(
            descriptor["approved_gpu_runtime_seconds"]
        ),
        approved_gpu_cost_usd=float(descriptor["approved_gpu_cost_usd"]),
        hourly_cost_usd=float(descriptor["max_hourly_cost_usd"]),
    )


def _start(args: argparse.Namespace) -> int:
    descriptor, _approval = _load_authority(args.descriptor, args.approval)
    observed_at = _parse_time(args.observed_at)
    launched_at = _parse_time(args.launched_at)
    if launched_at > observed_at:
        raise ValueError("allocation launch cannot follow its observation time")
    ledger = _ledger(descriptor, approval_path=args.approval, root=args.root)
    remaining = ledger.remaining_gpu_seconds(now=observed_at)
    if remaining <= 60 * 60:
        raise ValueError(
            "approved GPU budget has one hour or less remaining; refusing new work"
        )
    record = ledger.start_allocation(
        job_id=args.job_id,
        instance_id=args.instance_id,
        launched_at=launched_at,
    )
    _materialize_ledger_publication(ledger)
    remaining = ledger.remaining_gpu_seconds(now=observed_at)
    execution_deadline = ledger.execution_deadline(now=observed_at)
    body: dict[str, object] = {
        "record_type": "glm52_gpu_runtime_allocation_v1",
        "run_id": descriptor["run_id"],
        "job_id": args.job_id,
        "instance_id": args.instance_id,
        "launched_at": _iso_utc(launched_at),
        "observed_at": _iso_utc(observed_at),
        "execution_deadline": _iso_utc(execution_deadline),
        "approval_sha256": _sha256_file(args.approval),
        "gpu_spend_authority_sha256": ledger.genesis_sha256,
        "gpu_spend_record_sha256": record["record_sha256"],
        "gpu_spend_ledger_sha256": _sha256_file(ledger.path),
        "remaining_gpu_seconds": int(remaining),
        "estimated_gpu_cost_usd": ledger.estimated_gpu_cost_usd(now=observed_at),
    }
    allocation = {
        **body,
        "allocation_body_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }
    runtime_dir = args.root / "runtime"
    allocation_path = runtime_dir / "GPU_RUNTIME_ALLOCATION.json"
    _write_json_atomic(allocation_path, allocation)
    environment = "\n".join(
        [
            f"GLM52_EXECUTION_DEADLINE={body['execution_deadline']}",
            (
                "GLM52_GPU_ALLOCATION_SHA256="
                f"{allocation['allocation_body_sha256']}"
            ),
            (
                "GLM52_GPU_SPEND_AUTHORITY_SHA256="
                f"{ledger.genesis_sha256}"
            ),
            "",
        ]
    ).encode("utf-8")
    _write_atomic(runtime_dir / "campaign.env", environment)
    print(json.dumps(allocation, sort_keys=True))
    return 0


def _end(args: argparse.Namespace) -> int:
    descriptor, _approval = _load_authority(args.descriptor, args.approval)
    ended_at = _parse_time(args.ended_at)
    ledger = _ledger(descriptor, approval_path=args.approval, root=args.root)
    record = ledger.end_allocation(
        instance_id=args.instance_id,
        ended_at=ended_at,
    )
    _materialize_ledger_publication(ledger)
    status = {
        "record_type": "glm52_gpu_spend_status_v1",
        "run_id": descriptor["run_id"],
        "instance_id": args.instance_id,
        "ended_at": _iso_utc(ended_at),
        "gpu_spend_record_sha256": record["record_sha256"],
        "gpu_spend_ledger_sha256": _sha256_file(ledger.path),
        "consumed_gpu_seconds": int(ledger.consumed_gpu_seconds(now=ended_at)),
        "remaining_gpu_seconds": int(ledger.remaining_gpu_seconds(now=ended_at)),
        "estimated_gpu_cost_usd": ledger.estimated_gpu_cost_usd(now=ended_at),
    }
    _write_json_atomic(args.root / "runtime/GPU_SPEND_STATUS.json", status)
    print(json.dumps(status, sort_keys=True))
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    descriptor, _approval = _load_authority(args.descriptor, args.approval)
    launched_at = _parse_time(args.launched_at)
    ended_at = _parse_time(args.ended_at)
    if ended_at < launched_at:
        raise ValueError("reconciled allocation end precedes launch")
    ledger = _ledger(descriptor, approval_path=args.approval, root=args.root)
    matching = [
        record
        for record in ledger.records
        if record["instance_id"] == args.instance_id
    ]
    if matching:
        if (
            len(matching) == 2
            and matching[0]["event"] == "allocation_started"
            and matching[1]["event"] == "allocation_ended"
        ):
            _materialize_ledger_publication(ledger)
            print(json.dumps(matching[-1], sort_keys=True))
            return 0
        raise ValueError("reconciled GPU allocation is only partially recorded")
    ledger.start_allocation(
        job_id=args.job_id,
        instance_id=args.instance_id,
        launched_at=launched_at,
    )
    record = ledger.end_allocation(
        instance_id=args.instance_id,
        ended_at=ended_at,
    )
    _materialize_ledger_publication(ledger)
    print(json.dumps(record, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "end", "reconcile"):
        command = subparsers.add_parser(name)
        command.add_argument("--descriptor", type=Path, required=True)
        command.add_argument("--approval", type=Path, required=True)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--instance-id", required=True)
    start = subparsers.choices["start"]
    start.add_argument("--job-id", required=True)
    start.add_argument("--launched-at", required=True)
    start.add_argument("--observed-at", required=True)
    end = subparsers.choices["end"]
    end.add_argument("--ended-at", required=True)
    reconcile = subparsers.choices["reconcile"]
    reconcile.add_argument("--job-id", required=True)
    reconcile.add_argument("--launched-at", required=True)
    reconcile.add_argument("--ended-at", required=True)
    args = parser.parse_args()
    try:
        if args.command == "start":
            return _start(args)
        if args.command == "end":
            return _end(args)
        return _reconcile(args)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
