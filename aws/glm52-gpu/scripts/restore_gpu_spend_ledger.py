#!/usr/bin/env python3
"""Reconstruct a spend ledger from an authenticated immutable-record marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_RECORD_NAME = re.compile(
    r"(?P<index>[0-9]{6})-"
    r"(?P<event>allocation_started|allocation_ended)-"
    r"(?P<sha>[0-9a-f]{64})\.json"
)
_START_RECORD_FIELDS = {
    "record_type",
    "run_id",
    "approval_sha256",
    "event",
    "job_id",
    "instance_id",
    "timestamp",
    "prior_record_sha256",
    "record_sha256",
}
_END_RECORD_FIELDS = _START_RECORD_FIELDS - {"job_id"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _load_latest(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("GPU spend latest marker must be a regular file")
    value = json.loads(path.read_bytes())
    required = {
        "schema_version",
        "record_type",
        "run_id",
        "gpu_spend_authority_sha256",
        "record_count",
        "record_keys",
        "latest_record_sha256",
        "ledger_sha256",
        "latest_body_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("record_type") != "glm52_gpu_spend_ledger_latest_v1"
    ):
        raise ValueError("GPU spend latest marker schema mismatch")
    if path.read_bytes() != _canonical(value) + b"\n":
        raise ValueError("GPU spend latest marker bytes are not canonical")
    body = dict(value)
    actual = body.pop("latest_body_sha256")
    if actual != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("GPU spend latest marker SHA-256 mismatch")
    keys = value.get("record_keys")
    count = value.get("record_count")
    if (
        not isinstance(keys, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(keys)
    ):
        raise ValueError("GPU spend latest record inventory mismatch")
    for index, name in enumerate(keys):
        match = _RECORD_NAME.fullmatch(str(name))
        if match is None or int(match.group("index")) != index:
            raise ValueError("GPU spend immutable record name is invalid")
    for field in (
        "gpu_spend_authority_sha256",
        "ledger_sha256",
    ):
        if _HEX64.fullmatch(str(value.get(field))) is None:
            raise ValueError(f"GPU spend latest {field} is invalid")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("GPU spend latest run_id is invalid")
    latest_record = value.get("latest_record_sha256")
    if count == 0:
        if latest_record is not None:
            raise ValueError("empty GPU spend latest marker has a record SHA")
    elif latest_record != _RECORD_NAME.fullmatch(str(keys[-1])).group("sha"):
        raise ValueError("GPU spend latest record SHA-256 mismatch")
    return value


def _compare_s3_key_inventory(
    latest: dict[str, object],
    inventory_path: Path,
    *,
    record_prefix: str,
) -> None:
    if (
        inventory_path.is_symlink()
        or not inventory_path.is_file()
        or not record_prefix
        or not record_prefix.endswith("/")
    ):
        raise ValueError("GPU spend immutable key inventory input is invalid")
    inventory = json.loads(inventory_path.read_bytes())
    if (
        not isinstance(inventory, dict)
        or inventory.get("IsTruncated") is not False
        or inventory.get("NextToken") is not None
        or inventory.get("NextContinuationToken") is not None
    ):
        raise ValueError("GPU spend immutable key inventory is incomplete")
    contents = inventory.get("Contents", [])
    if not isinstance(contents, list):
        raise ValueError("GPU spend immutable key inventory is malformed")
    names: list[str] = []
    for item in contents:
        key = item.get("Key") if isinstance(item, dict) else None
        if not isinstance(key, str) or not key.startswith(record_prefix):
            raise ValueError("GPU spend immutable key inventory is malformed")
        name = key[len(record_prefix) :]
        if "/" in name or _RECORD_NAME.fullmatch(name) is None:
            raise ValueError("GPU spend immutable key inventory is malformed")
        names.append(name)
    if (
        len(set(names)) != len(names)
        or set(names) != set(latest["record_keys"])
        or len(names) != int(latest["record_count"])
    ):
        raise ValueError(
            "GPU spend latest is stale or rollback: declared immutable "
            "chain/key set does not match S3"
        )


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("GPU spend immutable record timestamp is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GPU spend immutable record timestamp is invalid")
    return parsed.astimezone(timezone.utc)


def _reconstruct(latest: dict[str, object], records_dir: Path) -> bytes:
    output = bytearray()
    prior_record_sha = str(latest["gpu_spend_authority_sha256"])
    prior_timestamp: datetime | None = None
    approval_sha256: str | None = None
    active_instance: str | None = None
    for index, name_raw in enumerate(latest["record_keys"]):
        name = str(name_raw)
        path = records_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"GPU spend immutable record is missing: {name}")
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"GPU spend immutable record is invalid: {name}")
        event = value.get("event")
        expected_fields = (
            _START_RECORD_FIELDS
            if event == "allocation_started"
            else _END_RECORD_FIELDS
            if event == "allocation_ended"
            else set()
        )
        if (
            not expected_fields
            or set(value) != expected_fields
            or value.get("record_type") != "glm52_gpu_spend_event_v1"
        ):
            raise ValueError(f"GPU spend immutable record schema mismatch: {name}")
        if raw != _canonical(value) + b"\n":
            raise ValueError(
                f"GPU spend immutable record bytes are not canonical: {name}"
            )
        body = dict(value)
        record_sha = body.pop("record_sha256", None)
        expected = hashlib.sha256(_canonical(body)).hexdigest()
        match = _RECORD_NAME.fullmatch(name)
        assert match is not None
        if (
            record_sha != expected
            or record_sha != match.group("sha")
            or value.get("event") != match.group("event")
        ):
            raise ValueError(f"GPU spend immutable record identity mismatch: {name}")
        if value.get("run_id") != latest["run_id"]:
            raise ValueError("GPU spend immutable record contains a foreign run")
        current_approval = value.get("approval_sha256")
        if _HEX64.fullmatch(str(current_approval)) is None:
            raise ValueError("GPU spend immutable record approval SHA-256 is invalid")
        if approval_sha256 is None:
            approval_sha256 = str(current_approval)
        elif current_approval != approval_sha256:
            raise ValueError("GPU spend immutable records contain foreign approval")
        if value.get("prior_record_sha256") != prior_record_sha:
            raise ValueError("GPU spend immutable records are noncontiguous")
        current_timestamp = _timestamp(value.get("timestamp"))
        if prior_timestamp is not None and current_timestamp < prior_timestamp:
            raise ValueError("GPU spend immutable record timestamps are time-skewed")
        instance_id = value.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("GPU spend immutable record instance_id is invalid")
        if event == "allocation_started":
            if active_instance is not None:
                raise ValueError(
                    "GPU spend immutable records contain parallel allocation"
                )
            job_id = value.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("GPU spend immutable record job_id is invalid")
            active_instance = instance_id
        elif active_instance != instance_id:
            raise ValueError("GPU spend immutable allocation end is noncontiguous")
        else:
            active_instance = None
        prior_record_sha = str(record_sha)
        prior_timestamp = current_timestamp
        output.extend(raw)
    if latest["record_count"] and (prior_record_sha != latest["latest_record_sha256"]):
        raise ValueError("GPU spend immutable record tip SHA-256 mismatch")
    if hashlib.sha256(output).hexdigest() != latest["ledger_sha256"]:
        raise ValueError("reconstructed GPU spend ledger SHA-256 mismatch")
    return bytes(output)


def _write_atomic(path: Path, raw: bytes) -> None:
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-records", action="store_true")
    parser.add_argument("--s3-key-inventory", type=Path)
    parser.add_argument("--s3-record-prefix")
    parser.add_argument("--require-nonempty", action="store_true")
    args = parser.parse_args()
    try:
        latest = _load_latest(args.latest)
        if args.require_nonempty and latest["record_count"] == 0:
            raise ValueError(
                "required GPU spend latest marker contains no immutable records"
            )
        if (args.s3_key_inventory is None) != (args.s3_record_prefix is None):
            parser.error("--s3-key-inventory and --s3-record-prefix must be combined")
        if args.s3_key_inventory is not None:
            _compare_s3_key_inventory(
                latest,
                args.s3_key_inventory,
                record_prefix=args.s3_record_prefix,
            )
        if args.list_records:
            if args.records_dir is not None or args.output is not None:
                parser.error("--list-records cannot be combined with reconstruction")
            for name in latest["record_keys"]:
                print(name)
            return 0
        if args.records_dir is None or args.output is None:
            parser.error("--records-dir and --output are required")
        _write_atomic(args.output, _reconstruct(latest, args.records_dir))
        return 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
