#!/usr/bin/env python3
"""Abort only stale source uploads whose published objects already authenticate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location("_glm52_stale_multipart", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
validate_s3_artifact_inventory = AUDIT.validate_s3_artifact_inventory


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("multipart cutoff must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--checksum-authority", type=Path, required=True)
    parser.add_argument("--initiated-before", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile == "default":
        parser.error("the default AWS profile is forbidden")
    subprocess.run(
        [str(REPO_ROOT / "aws/glm52-gpu/scripts/assert_rnd_aws_account.sh")],
        check=True,
        env={**os.environ, "AWS_PROFILE": args.profile},
    )
    inventory = validate_s3_artifact_inventory(
        json.loads(args.inventory.read_bytes())
    )
    authority = json.loads(args.checksum_authority.read_bytes())
    if (
        not isinstance(authority, dict)
        or authority.get("record_type")
        != "glm52_s3_batch_checksum_authority_v1"
        or authority.get("account_id") != "246813579024"
        or authority.get("region") != args.region
        or authority.get("bucket") != inventory["bucket"]
        or authority.get("inventory_body_sha256")
        != inventory["inventory_body_sha256"]
        or authority.get("checksum_algorithm") != "SHA256"
        or authority.get("checksum_type") != "FULL_OBJECT"
        or not isinstance(authority.get("checksums"), dict)
    ):
        parser.error("S3 Batch checksum authority identity mismatch")
    authority_body = dict(authority)
    recorded = authority_body.pop("authority_body_sha256", None)
    if recorded != hashlib.sha256(_canonical(authority_body)).hexdigest():
        parser.error("S3 Batch checksum authority body SHA-256 mismatch")

    expected = {str(item["key"]): item for item in inventory["objects"]}
    checksums = {
        str(key): str(value) for key, value in authority["checksums"].items()
    }
    if set(checksums) != set(expected):
        parser.error("checksum authority does not cover the exact inventory")
    cutoff = _time(args.initiated_before)
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    candidates: list[dict[str, object]] = []
    paginator = s3.get_paginator("list_multipart_uploads")
    for page in paginator.paginate(Bucket=str(inventory["bucket"])):
        for upload in page.get("Uploads", []):
            key = str(upload["Key"])
            initiated = upload["Initiated"]
            if initiated.tzinfo is None:
                raise ValueError("S3 multipart initiation time is naive")
            if initiated.astimezone(timezone.utc) >= cutoff:
                continue
            if not (
                key.startswith("source-snapshot/model-")
                and key.endswith(".safetensors")
            ):
                raise ValueError(f"stale foreign multipart upload requires review: {key}")
            item = expected.get(key)
            if item is None or checksums.get(key) != item["sha256"]:
                raise ValueError(f"{key} lacks authenticated published authority")
            head = s3.head_object(Bucket=str(inventory["bucket"]), Key=key)
            if int(head["ContentLength"]) != item["size"]:
                raise ValueError(f"{key} published object size mismatch")
            candidates.append(
                {
                    "key": key,
                    "upload_id": str(upload["UploadId"]),
                    "initiated": initiated.astimezone(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )

    candidates.sort(key=lambda item: (str(item["key"]), str(item["upload_id"])))
    if args.apply:
        for item in candidates:
            s3.abort_multipart_upload(
                Bucket=str(inventory["bucket"]),
                Key=str(item["key"]),
                UploadId=str(item["upload_id"]),
            )
    report = {
        "schema_version": 1,
        "record_type": "glm52_stale_source_multipart_cleanup_v1",
        "bucket": inventory["bucket"],
        "inventory_body_sha256": inventory["inventory_body_sha256"],
        "checksum_authority_body_sha256": authority["authority_body_sha256"],
        "initiated_before": cutoff.isoformat().replace("+00:00", "Z"),
        "candidate_count": len(candidates),
        "applied": args.apply,
        "uploads": candidates,
    }
    _write_atomic(args.output, report)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
