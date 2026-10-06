#!/usr/bin/env python3
"""Audit an exact GLM-5.2 S3 inventory before any paid GPU launch."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location("_glm52_s3_artifact_audit", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
S3ObjectHead = AUDIT.S3ObjectHead
audit_s3_artifact_inventory = AUDIT.audit_s3_artifact_inventory
validate_s3_artifact_inventory = AUDIT.validate_s3_artifact_inventory


def _aws(profile: str, *arguments: str, capture: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["aws", *arguments, "--profile", profile],
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
    )


def _json_command(profile: str, *arguments: str) -> object:
    return json.loads(_aws(profile, *arguments).stdout)


def _stream_sha256(profile: str, *, bucket: str, key: str, region: str) -> str:
    command = [
        "aws",
        "s3",
        "cp",
        f"s3://{bucket}/{key}",
        "-",
        "--region",
        region,
        "--only-show-errors",
        "--profile",
        profile,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    digest = hashlib.sha256()
    for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        digest.update(block)
    status = process.wait()
    if status:
        raise RuntimeError(f"streaming SHA-256 failed for s3://{bucket}/{key}")
    return digest.hexdigest()


def _range(
    profile: str,
    *,
    bucket: str,
    key: str,
    region: str,
    start: int,
    length: int,
    version_id: str | None,
) -> bytes:
    if length <= 0:
        return b""
    descriptor, raw_path = tempfile.mkstemp(prefix="glm52-s3-range-")
    os.close(descriptor)
    path = Path(raw_path)
    try:
        arguments = [
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
        ]
        if version_id is not None:
            arguments.extend(["--version-id", version_id])
        arguments.extend(
            [
                "--range",
                f"bytes={start}-{start + length - 1}",
                "--region",
                region,
                str(path),
            ]
        )
        _aws(profile, *arguments)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode() + b"\n"
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
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--checksum-authority",
        type=Path,
        help="authenticated S3 Batch full-object SHA-256 authority JSON",
    )
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
    bucket = str(inventory["bucket"])
    inventory_by_key = {
        str(item["key"]): item
        for item in inventory["objects"]
    }
    sha_cache: dict[str, str] = {}
    batch_checksums: dict[str, str] = {}
    if args.checksum_authority is not None:
        authority = json.loads(args.checksum_authority.read_bytes())
        expected_fields = {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "bucket",
            "job_id",
            "inventory_body_sha256",
            "checksum_algorithm",
            "checksum_type",
            "checksums",
            "authority_body_sha256",
        }
        if not isinstance(authority, dict) or set(authority) != expected_fields:
            parser.error("S3 Batch checksum authority schema mismatch")
        authority_body = dict(authority)
        authority_digest = authority_body.pop("authority_body_sha256")
        if authority_digest != hashlib.sha256(
            json.dumps(
                authority_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest():
            parser.error("S3 Batch checksum authority body SHA-256 mismatch")
        if (
            authority["schema_version"] != 1
            or authority["record_type"]
            != "glm52_s3_batch_checksum_authority_v1"
            or authority["account_id"] != "246813579024"
            or authority["region"] != args.region
            or authority["bucket"] != bucket
            or authority["checksum_algorithm"] != "SHA256"
            or authority["checksum_type"] != "FULL_OBJECT"
            or not isinstance(authority["checksums"], dict)
        ):
            parser.error("S3 Batch checksum authority identity mismatch")
        batch_checksums = {
            str(key): str(value)
            for key, value in authority["checksums"].items()
        }
        shared_items = [
            item
            for item in inventory["objects"]
            if item["run_scope"] == "shared"
        ]
        shared_inventory = AUDIT.build_s3_artifact_inventory(
            run_id=str(inventory["run_id"]),
            bucket=bucket,
            objects=shared_items,
        )
        shared_keys = {str(item["key"]) for item in shared_items}
        if (
            authority["inventory_body_sha256"]
            != shared_inventory["inventory_body_sha256"]
            or set(batch_checksums) != shared_keys
        ):
            parser.error(
                "S3 Batch checksum authority does not cover the exact "
                "shared inventory"
            )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in batch_checksums.values()
        ):
            parser.error("S3 Batch checksum authority inventory mismatch")

    def head_object(key: str) -> S3ObjectHead:
        item = inventory_by_key[key]
        version_id = item.get("version_id")
        arguments = [
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
        ]
        if isinstance(version_id, str):
            arguments.extend(["--version-id", version_id])
        arguments.extend(
            [
                "--checksum-mode",
                "ENABLED",
                "--region",
                args.region,
            ]
        )
        response = _json_command(
            args.profile,
            *arguments,
        )
        if not isinstance(response, dict):
            raise ValueError(f"malformed S3 HEAD response for {key}")
        checksum = response.get("ChecksumSHA256")
        checksum_type = response.get("ChecksumType")
        if key in batch_checksums:
            sha256 = batch_checksums[key]
            checksum_source = "s3_batch_full_object_sha256"
        elif isinstance(checksum, str) and checksum_type == "FULL_OBJECT":
            sha256 = base64.b64decode(checksum, validate=True).hex()
            checksum_source = "s3_full_object_sha256"
        elif isinstance(version_id, str):
            raise ValueError(
                f"{key} exact VersionId lacks an authenticated "
                "full-object SHA-256 authority"
            )
        else:
            sha256 = sha_cache.setdefault(
                key,
                _stream_sha256(
                    args.profile,
                    bucket=bucket,
                    key=key,
                    region=args.region,
                ),
            )
            checksum_source = "streamed_sha256"
        metadata = response.get("Metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"malformed S3 metadata for {key}")
        return S3ObjectHead(
            size=int(response["ContentLength"]),
            etag=str(response.get("ETag", "")),
            sha256=sha256,
            checksum_source=checksum_source,
            metadata={str(name): str(value) for name, value in metadata.items()},
            version_id=(
                str(response["VersionId"])
                if isinstance(response.get("VersionId"), str)
                else None
            ),
        )

    multipart = _json_command(
        args.profile,
        "s3api",
        "list-multipart-uploads",
        "--bucket",
        bucket,
        "--region",
        args.region,
    )
    if not isinstance(multipart, dict):
        raise ValueError("malformed S3 multipart upload response")
    active_uploads = tuple(
        str(upload["Key"])
        for upload in multipart.get("Uploads", [])
        if isinstance(upload, dict) and "Key" in upload
    )
    report = audit_s3_artifact_inventory(
        inventory,
        head_object=head_object,
        read_range=lambda key, start, length: _range(
            args.profile,
            bucket=bucket,
            key=key,
            region=args.region,
            start=start,
            length=length,
            version_id=(
                str(inventory_by_key[key]["version_id"])
                if isinstance(inventory_by_key[key].get("version_id"), str)
                else None
            ),
        ),
        active_multipart_uploads=active_uploads,
    )
    _write_atomic(args.output, report)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
