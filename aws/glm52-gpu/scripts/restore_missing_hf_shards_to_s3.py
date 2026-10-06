#!/usr/bin/env python3
"""Restore missing pinned GLM-5.2 source shards to regional S3 without local disk."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPOSITORY = "0xSero/glm-5.2-reap-504B-v2"
DEFAULT_PREFIX = "source-snapshot"
S3_MIN_PART_BYTES = 5 * 1024 * 1024
PART_BYTES = 64 * 1024 * 1024
HTTP_RETRY_ATTEMPTS = 6
TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}


def _drain_multipart_parts(
    pending: bytearray,
    *,
    part_bytes: int = PART_BYTES,
    final: bool,
) -> list[bytes]:
    if part_bytes < S3_MIN_PART_BYTES:
        raise ValueError("multipart part size must be at least 5 MiB")
    parts: list[bytes] = []
    while len(pending) >= part_bytes:
        parts.append(bytes(pending[:part_bytes]))
        del pending[:part_bytes]
    if final and pending:
        parts.append(bytes(pending))
        pending.clear()
    return parts


def _open_with_backoff(request: urllib.request.Request):
    for attempt in range(HTTP_RETRY_ATTEMPTS):
        retry_after: str | None = None
        try:
            return urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP_STATUS:
                raise
            retry_after = error.headers.get("Retry-After")
            error.close()
        except urllib.error.URLError:
            if attempt + 1 == HTTP_RETRY_ATTEMPTS:
                raise
        if attempt + 1 == HTTP_RETRY_ATTEMPTS:
            raise RuntimeError("pinned Hugging Face download retries exhausted")
        delay = min(60.0, float(2**attempt))
        if retry_after is not None:
            try:
                delay = min(60.0, max(delay, float(retry_after)))
            except ValueError:
                pass
        print(
            f"transient HTTP response; retrying in {delay:g}s "
            f"(attempt {attempt + 1}/{HTTP_RETRY_ATTEMPTS})",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("HTTP retry loop did not return or raise")


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


def _existing_keys(s3: Any, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def _head_or_none(s3: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _upload_one(
    *,
    profile: str,
    region: str,
    bucket: str,
    prefix: str,
    repository: str,
    revision: str,
    filename: str,
    authority: dict[str, object],
) -> dict[str, object]:
    expected_size = int(authority["lfs_size"])
    expected_sha256 = str(authority["lfs_sha256"])
    key = f"{prefix}/{filename}"
    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")
    head = _head_or_none(s3, bucket=bucket, key=key)
    if head is not None:
        metadata = head.get("Metadata", {})
        if (
            int(head["ContentLength"]) == expected_size
            and metadata.get("glm52-lfs-sha256") == expected_sha256
            and metadata.get("glm52-source-revision") == revision
        ):
            return {
                "key": key,
                "size": expected_size,
                "sha256": expected_sha256,
                "status": "already_restored",
            }
        raise ValueError(f"refusing to overwrite unauthenticated existing object {key}")

    multipart = s3.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ChecksumAlgorithm="SHA256",
        ContentType="application/octet-stream",
        Metadata={
            "glm52-lfs-sha256": expected_sha256,
            "glm52-source-revision": revision,
        },
        Tagging=(
            "project=keep-glm52&owner=jack.mazac&model=glm-5.2"
            "&cost-allocation=glm52-sky-campaign"
        ),
    )
    upload_id = str(multipart["UploadId"])
    parts: list[dict[str, object]] = []
    digest = hashlib.sha256()
    byte_count = 0
    pending = bytearray()
    url = (
        "https://huggingface.co/"
        f"{urllib.parse.quote(repository, safe='/')}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
        f"{urllib.parse.quote(filename, safe='')}"
    )
    try:
        while byte_count < expected_size:
            headers = {"User-Agent": "keep-glm52-pinned-s3-restore/1"}
            if byte_count:
                headers["Range"] = f"bytes={byte_count}-"
            request = urllib.request.Request(url, headers=headers)
            connection_start = byte_count
            with _open_with_backoff(request) as response:
                if connection_start:
                    content_range = str(response.headers.get("Content-Range", ""))
                    if response.status != 206 or not content_range.startswith(
                        f"bytes {connection_start}-"
                    ):
                        raise ValueError(
                            f"HTTP range continuation mismatch for {key}"
                        )
                while byte_count < expected_size:
                    incomplete = False
                    try:
                        block = response.read(
                            min(PART_BYTES, expected_size - byte_count)
                        )
                    except http.client.IncompleteRead as error:
                        block = error.partial
                        incomplete = True
                    if not block:
                        break
                    if byte_count + len(block) > expected_size:
                        raise ValueError(f"HTTP response exceeded pinned size for {key}")
                    digest.update(block)
                    byte_count += len(block)
                    pending.extend(block)
                    for part in _drain_multipart_parts(
                        pending,
                        final=byte_count == expected_size,
                    ):
                        part_sha256 = base64.b64encode(
                            hashlib.sha256(part).digest()
                        ).decode()
                        uploaded = s3.upload_part(
                            Bucket=bucket,
                            Key=key,
                            UploadId=upload_id,
                            PartNumber=len(parts) + 1,
                            Body=part,
                            ChecksumAlgorithm="SHA256",
                            ChecksumSHA256=part_sha256,
                        )
                        returned_checksum = str(
                            uploaded.get("ChecksumSHA256", "")
                        )
                        if returned_checksum != part_sha256:
                            raise ValueError(
                                f"S3 part checksum mismatch for {key}"
                            )
                        parts.append(
                            {
                                "PartNumber": len(parts) + 1,
                                "ETag": str(uploaded["ETag"]),
                                "ChecksumSHA256": returned_checksum,
                            }
                        )
                    if incomplete:
                        break
            if byte_count == connection_start:
                raise ValueError(f"HTTP range continuation made no progress for {key}")
        actual_sha256 = digest.hexdigest()
        if byte_count != expected_size:
            raise ValueError(
                f"{key} byte count mismatch: expected {expected_size}, got {byte_count}"
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{key} pinned LFS SHA-256 mismatch")
        completed = s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        upload_id = ""
        final_head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(final_head["ContentLength"]) != expected_size:
            raise ValueError(f"{key} completed S3 object size mismatch")
        return {
            "key": key,
            "size": expected_size,
            "sha256": actual_sha256,
            "etag": str(completed.get("ETag", "")),
            "status": "restored",
        }
    finally:
        if upload_id:
            s3.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile == "default":
        parser.error("the default AWS profile is forbidden")
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8")
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be positive")

    subprocess.run(
        [str(REPO_ROOT / "aws/glm52-gpu/scripts/assert_rnd_aws_account.sh")],
        check=True,
        env={**os.environ, "AWS_PROFILE": args.profile},
    )
    tree = json.loads(args.source_tree.read_bytes())
    if not isinstance(tree, dict) or not isinstance(tree.get("files"), dict):
        parser.error("source tree authority is malformed")
    files = {
        str(name): value
        for name, value in tree["files"].items()
        if str(name).endswith(".safetensors")
    }
    for name, value in files.items():
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("lfs_sha256"), str)
            or not isinstance(value.get("lfs_size"), int)
        ):
            parser.error(f"{name} lacks pinned LFS size/SHA-256 authority")

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    present = _existing_keys(session.client("s3"), args.bucket, args.prefix)
    missing = [
        name
        for name in sorted(files)
        if f"{args.prefix}/{name}" not in present
    ]
    selected = missing[: args.max_shards] if args.max_shards else missing
    if args.dry_run:
        report: dict[str, object] = {
            "schema_version": 1,
            "record_type": "glm52_source_shard_restore_v1",
            "bucket": args.bucket,
            "revision": args.revision,
            "expected_shards": len(files),
            "present_shards": len(files) - len(missing),
            "missing_shards": missing,
            "selected_shards": selected,
            "dry_run": True,
        }
        _write_atomic(args.output, report)
        print(args.output)
        return 0

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                _upload_one,
                profile=args.profile,
                region=args.region,
                bucket=args.bucket,
                prefix=args.prefix,
                repository=args.repository,
                revision=args.revision,
                filename=name,
                authority=files[name],
            ): name
            for name in selected
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"{result['status']} {result['key']} "
                    f"{result['size']} {result['sha256']}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    results.sort(key=lambda item: str(item["key"]))
    report = {
        "schema_version": 1,
        "record_type": "glm52_source_shard_restore_v1",
        "bucket": args.bucket,
        "revision": args.revision,
        "expected_shards": len(files),
        "initially_missing_shards": missing,
        "selected_shards": selected,
        "restored": results,
        "remaining_missing_shards": missing[len(selected) :],
        "dry_run": False,
    }
    _write_atomic(args.output, report)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
