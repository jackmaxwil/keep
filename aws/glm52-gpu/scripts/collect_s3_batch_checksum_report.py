#!/usr/bin/env python3
"""Collect a completed S3 Batch checksum report into an authenticated authority."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import boto3


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location("_glm52_s3_report_inventory", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


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


def _read_report_rows(raw: bytes, *, compressed: bool) -> list[list[str]]:
    if compressed:
        raw = gzip.decompress(raw)
    return list(csv.reader(io.StringIO(raw.decode("utf-8"))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--job-marker", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile == "default":
        parser.error("the default AWS profile is forbidden")
    subprocess.run(
        [str(REPO_ROOT / "aws/glm52-gpu/scripts/assert_rnd_aws_account.sh")],
        check=True,
        env={**os.environ, "AWS_PROFILE": args.profile},
    )
    marker = json.loads(args.job_marker.read_bytes())
    inventory = AUDIT.validate_s3_artifact_inventory(
        json.loads(args.inventory.read_bytes())
    )
    if (
        not isinstance(marker, dict)
        or marker.get("record_type") != "glm52_s3_batch_checksum_job_v1"
        or marker.get("account_id") != "246813579024"
        or marker.get("region") != args.region
        or marker.get("bucket") != inventory["bucket"]
        or marker.get("run_id") != inventory["run_id"]
        or marker.get("inventory_body_sha256")
        != inventory["inventory_body_sha256"]
        or marker.get("object_count") != len(inventory["objects"])
    ):
        parser.error("S3 Batch job marker identity mismatch")

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3control = session.client("s3control")
    response = s3control.describe_job(
        AccountId="246813579024",
        JobId=str(marker["job_id"]),
    )
    job = response.get("Job")
    if not isinstance(job, dict):
        parser.error("S3 Batch describe-job response is malformed")
    status = str(job.get("Status", ""))
    if status != "Complete":
        parser.error(f"S3 Batch checksum job is not complete: {status}")
    progress = job.get("ProgressSummary", {})
    total = int(progress.get("TotalNumberOfTasks", -1))
    succeeded = int(progress.get("NumberOfTasksSucceeded", -1))
    failed = int(progress.get("NumberOfTasksFailed", -1))
    if (
        total != len(inventory["objects"])
        or succeeded != total
        or failed != 0
    ):
        parser.error(
            "S3 Batch checksum job did not succeed for every inventory object"
        )
    report = job.get("Report", {})
    prefix = report.get("Prefix") if isinstance(report, dict) else None
    if not isinstance(prefix, str) or not prefix.startswith("campaign-audits/"):
        parser.error("S3 Batch completion report prefix is invalid")

    bucket = str(inventory["bucket"])
    job_id = str(marker["job_id"])
    s3 = session.client("s3")
    report_keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if job_id in key and (key.endswith(".csv") or key.endswith(".csv.gz")):
                report_keys.append(key)
    if not report_keys:
        parser.error("S3 Batch checksum completion CSV is missing")

    checksums: dict[str, str] = {}
    expected_versions = {
        str(item["key"]): str(item["version_id"])
        for item in inventory["objects"]
        if isinstance(item.get("version_id"), str)
        and item["version_id"] not in {"", "null", "None"}
    }
    if len(expected_versions) != len(inventory["objects"]):
        parser.error(
            "S3 Batch checksum inventory requires an exact VersionId "
            "for every object"
        )
    for report_key in sorted(report_keys):
        raw = s3.get_object(Bucket=bucket, Key=report_key)["Body"].read()
        for row in _read_report_rows(
            raw,
            compressed=report_key.endswith(".gz"),
        ):
            if len(row) != 7:
                parser.error("S3 Batch completion row is malformed")
            row_bucket, encoded_key, row_version_id, task_status = row[:4]
            if row_bucket != bucket:
                parser.error("S3 Batch completion row has a foreign bucket")
            if task_status != "succeeded":
                parser.error("S3 Batch completion row contains a failed task")
            key = urllib.parse.unquote(encoded_key)
            if expected_versions.get(key) != row_version_id:
                parser.error(
                    "S3 Batch completion row VersionId does not match inventory"
                )
            result = json.loads(row[-1])
            if (
                not isinstance(result, dict)
                or result.get("checksumAlgorithm") != "SHA256"
                or result.get("checksumType") != "FULL_OBJECT"
            ):
                parser.error("S3 Batch checksum result type mismatch")
            checksum = str(result.get("checksum_hex", "")).lower()
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                parser.error("S3 Batch checksum result is malformed")
            prior = checksums.setdefault(key, checksum)
            if prior != checksum:
                parser.error("S3 Batch report contains conflicting checksums")

    expected_keys = {str(item["key"]) for item in inventory["objects"]}
    if set(checksums) != expected_keys:
        parser.error("S3 Batch report does not cover the exact inventory")
    body: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "glm52_s3_batch_checksum_authority_v1",
        "account_id": "246813579024",
        "region": args.region,
        "bucket": bucket,
        "job_id": job_id,
        "inventory_body_sha256": inventory["inventory_body_sha256"],
        "checksum_algorithm": "SHA256",
        "checksum_type": "FULL_OBJECT",
        "checksums": dict(sorted(checksums.items())),
    }
    body["authority_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    _write_atomic(args.output, body)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
