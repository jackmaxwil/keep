#!/usr/bin/env python3
"""Exact-version, owner-bound Task 10 approval and intent downloads."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Mapping, NoReturn
from urllib.parse import urlsplit


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
APPROVAL = Path("/mnt/nvme/glm52-campaign/runtime/approval.json")
INTENT = Path("/mnt/nvme/glm52-campaign/runtime/intent.json")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")


def _fail(message: str) -> NoReturn:
    print("Task 10 runtime download: " + message, file=sys.stderr)
    raise SystemExit(70)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(name + " is absent")
    return value


def _coordinate(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path[1:]
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("runtime S3 coordinate is invalid")
    return parsed.netloc, parsed.path[1:]


def _download(
    client: object,
    *,
    uri: str,
    version_id: str,
    file_sha256: str,
) -> bytes:
    if (
        _VERSION.fullmatch(version_id) is None
        or version_id == "null"
        or _SHA.fullmatch(file_sha256) is None
    ):
        raise ValueError("runtime object authority is invalid")
    bucket, key = _coordinate(uri)
    get_object = getattr(client, "get_object", None)
    if not callable(get_object):
        raise ValueError("runtime S3 get boundary is absent")
    response = get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    if type(response) is not dict:
        raise ValueError("runtime S3 get response is invalid")
    body = response.get("Body")
    metadata = response.get("ResponseMetadata")
    if getattr(body, "read", None) is None:
        raise ValueError("runtime S3 response body is absent")
    raw = body.read()
    expected_checksum = base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii")
    if (
        response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != expected_checksum
        or type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or hashlib.sha256(raw).hexdigest() != file_sha256
    ):
        raise ValueError("runtime exact-version readback drifted")
    return raw


def _exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _client() -> object:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        ),
    )


def main(
    argv: list[str],
    *,
    client: object | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    if argv:
        _fail("this downloader accepts no arguments")
    environment = os.environ if environment is None else environment
    try:
        boundary = _client() if client is None else client
        approval_raw = _download(
            boundary,
            uri=_required(environment, "GLM52_APPROVAL_S3_URI"),
            version_id=_required(
                environment,
                "GLM52_APPROVAL_VERSION_ID",
            ),
            file_sha256=_required(
                environment,
                "GLM52_APPROVAL_FILE_SHA256",
            ),
        )
        intent_raw = _download(
            boundary,
            uri=_required(environment, "GLM52_SUBMISSION_INTENT_S3_URI"),
            version_id=_required(
                environment,
                "GLM52_SUBMISSION_INTENT_VERSION_ID",
            ),
            file_sha256=_required(
                environment,
                "GLM52_SUBMISSION_INTENT_FILE_SHA256",
            ),
        )
        _exclusive(APPROVAL, approval_raw)
        _exclusive(INTENT, intent_raw)
    except (OSError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
