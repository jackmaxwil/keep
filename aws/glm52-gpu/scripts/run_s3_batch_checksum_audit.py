#!/usr/bin/env python3
"""Submit an exact S3 Batch ComputeChecksum job for a GLM-5.2 inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location("_glm52_s3_batch_inventory", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
validate_s3_artifact_inventory = AUDIT.validate_s3_artifact_inventory

CHECKSUM_ROLE_NAME = "keep-glm52-s3-checksum-auditor"


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
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
    inventory = validate_s3_artifact_inventory(
        json.loads(args.inventory.read_bytes())
    )
    account_id = "246813579024"
    bucket = str(inventory["bucket"])
    run_id = str(inventory["run_id"])
    inventory_sha256 = str(inventory["inventory_body_sha256"])
    versioned = [
        item
        for item in inventory["objects"]
        if isinstance(item.get("version_id"), str)
        and item["version_id"] not in {"", "null", "None"}
    ]
    if len(versioned) != len(inventory["objects"]):
        parser.error(
            "S3 Batch checksum inventory requires an exact VersionId "
            "for every object"
        )

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for item in inventory["objects"]:
        writer.writerow(
            [
                bucket,
                urllib.parse.quote(str(item["key"]), safe="/"),
                item["version_id"],
            ]
        )
    manifest = buffer.getvalue().encode()
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    prefix = f"campaign-audits/{run_id}/{inventory_sha256}"
    manifest_key = f"{prefix}/s3-batch-manifest.csv"

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    try:
        existing = s3.head_object(Bucket=bucket, Key=manifest_key)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
        uploaded = s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=manifest,
            ContentType="text/csv",
            ChecksumAlgorithm="SHA256",
            IfNoneMatch="*",
            Metadata={
                "glm52-inventory-sha256": inventory_sha256,
                "glm52-manifest-sha256": manifest_sha256,
                "glm52-run-id": run_id,
            },
        )
        etag = str(uploaded["ETag"]).strip('"')
        manifest_version_id = uploaded.get("VersionId")
    else:
        metadata = existing.get("Metadata", {})
        if (
            int(existing["ContentLength"]) != len(manifest)
            or metadata.get("glm52-inventory-sha256") != inventory_sha256
            or metadata.get("glm52-manifest-sha256") != manifest_sha256
        ):
            raise ValueError("foreign or corrupt immutable S3 Batch manifest exists")
        etag = str(existing["ETag"]).strip('"')
        manifest_version_id = existing.get("VersionId")
    if (
        not isinstance(manifest_version_id, str)
        or not manifest_version_id
        or manifest_version_id in {"null", "None"}
    ):
        raise ValueError("S3 Batch manifest lacks an opaque VersionId")

    role_arn = str(
        session.client("iam").get_role(RoleName=CHECKSUM_ROLE_NAME)["Role"]["Arn"]
    )
    token = hashlib.sha256(
        f"glm52-s3-checksum-v1:{run_id}:{inventory_sha256}".encode()
    ).hexdigest()
    response = session.client("s3control").create_job(
        AccountId=account_id,
        ConfirmationRequired=False,
        Operation={
            "S3ComputeObjectChecksum": {
                "ChecksumAlgorithm": "SHA256",
                "ChecksumType": "FULL_OBJECT",
            }
        },
        Report={
            "Bucket": f"arn:aws:s3:::{bucket}",
            "Format": "Report_CSV_20180820",
            "Enabled": True,
            "Prefix": f"{prefix}/report",
            "ReportScope": "AllTasks",
            "ExpectedBucketOwner": account_id,
        },
        ClientRequestToken=token,
        Manifest={
            "Spec": {
                "Format": "S3BatchOperations_CSV_20180820",
                "Fields": ["Bucket", "Key", "VersionId"],
            },
            "Location": {
                "ObjectArn": f"arn:aws:s3:::{bucket}/{manifest_key}",
                "ObjectVersionId": manifest_version_id,
                "ETag": etag,
            },
        },
        Description=(
            f"GLM-5.2 full-object SHA256 audit for {run_id} "
            f"inventory {inventory_sha256}"
        ),
        Priority=10,
        RoleArn=role_arn,
        Tags=[
            {"Key": "project", "Value": "keep-glm52"},
            {"Key": "owner", "Value": "jack.mazac"},
            {"Key": "model", "Value": "glm-5.2"},
            {"Key": "campaign-run-id", "Value": run_id},
            {"Key": "cost-allocation", "Value": "glm52-sky-campaign"},
        ],
    )
    marker = {
        "schema_version": 1,
        "record_type": "glm52_s3_batch_checksum_job_v1",
        "account_id": account_id,
        "region": args.region,
        "bucket": bucket,
        "run_id": run_id,
        "inventory_body_sha256": inventory_sha256,
        "manifest_key": manifest_key,
        "manifest_version_id": manifest_version_id,
        "manifest_sha256": manifest_sha256,
        "object_count": len(inventory["objects"]),
        "job_id": str(response["JobId"]),
        "role_arn": role_arn,
    }
    _write_atomic(args.output, marker)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
