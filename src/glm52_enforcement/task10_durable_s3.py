"""One-attempt immutable S3 publication/adoption for Task 10 glue."""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Mapping, Optional


ACCOUNT_ID = "246813579024"
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")


def _status(response: object, label: str) -> Mapping[str, object]:
    if type(response) is not dict:
        raise ValueError(label + " response is invalid")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(metadata.get("RetryAttempts")) is not int
        or metadata["RetryAttempts"] != 0
    ):
        raise ValueError(label + " response status is invalid")
    return response


def _version(response: Mapping[str, object], label: str) -> str:
    value = response.get("VersionId")
    if (
        type(value) is not str
        or value == "null"
        or _VERSION.fullmatch(value) is None
    ):
        raise ValueError(label + " VersionId is invalid")
    return value


def publish_or_adopt_exact(
    client: object,
    *,
    bucket: str,
    key: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> str:
    """Conditionally publish once, or adopt one exact existing version."""

    if (
        type(bucket) is not str
        or not bucket
        or type(key) is not str
        or not key
        or type(raw) is not bytes
        or not raw
        or type(metadata) is not dict
        or any(type(name) is not str for name in metadata)
        or any(type(value) is not str for value in metadata.values())
    ):
        raise ValueError("immutable S3 publication input is invalid")
    put_object = getattr(client, "put_object", None)
    head_object = getattr(client, "head_object", None)
    get_object = getattr(client, "get_object", None)
    if not all(callable(item) for item in (put_object, head_object, get_object)):
        raise ValueError("immutable S3 boundary is incomplete")
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    try:
        put_response = _status(
            put_object(
                Bucket=bucket,
                Key=key,
                Body=raw,
                ContentType="application/json",
                Metadata=dict(metadata),
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=checksum,
                ExpectedBucketOwner=ACCOUNT_ID,
                IfNoneMatch="*",
            ),
            "immutable S3 put",
        )
        version_id = _version(put_response, "immutable S3 put")
    except Exception:  # noqa: BLE001 - ambiguous/lost response is read back
        head_response = _status(
            head_object(
                Bucket=bucket,
                Key=key,
                ExpectedBucketOwner=ACCOUNT_ID,
                ChecksumMode="ENABLED",
            ),
            "immutable S3 adoption head",
        )
        version_id = _version(head_response, "immutable S3 adoption head")
    readback = _status(
        get_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        "immutable S3 exact readback",
    )
    body = readback.get("Body")
    if (
        readback.get("VersionId") != version_id
        or readback.get("ChecksumSHA256") != checksum
        or readback.get("Metadata") != dict(metadata)
        or getattr(body, "read", None) is None
        or body.read() != raw
    ):
        raise ValueError("immutable S3 exact readback drifted")
    return version_id


def exact_version_read(
    client: object,
    *,
    bucket: str,
    key: str,
    version_id: str,
    expected_raw: bytes,
    expected_metadata: Optional[Mapping[str, str]] = None,
) -> bytes:
    """Read one already-authenticated immutable version with checksum proof."""

    if _VERSION.fullmatch(version_id) is None or version_id == "null":
        raise ValueError("exact S3 read VersionId is invalid")
    if (
        expected_metadata is not None
        and (
            type(expected_metadata) is not dict
            or any(type(name) is not str for name in expected_metadata)
            or any(
                type(value) is not str
                for value in expected_metadata.values()
            )
        )
    ):
        raise ValueError("exact S3 read metadata is invalid")
    get_object = getattr(client, "get_object", None)
    if not callable(get_object):
        raise ValueError("exact S3 read boundary is absent")
    response = _status(
        get_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        "exact S3 readback",
    )
    body = response.get("Body")
    raw = body.read() if getattr(body, "read", None) is not None else None
    checksum = base64.b64encode(
        hashlib.sha256(expected_raw).digest()
    ).decode("ascii")
    if (
        response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != checksum
        or (
            expected_metadata is not None
            and response.get("Metadata") != dict(expected_metadata)
        )
        or raw != expected_raw
    ):
        raise ValueError("exact S3 readback drifted")
    return expected_raw


__all__ = ["exact_version_read", "publish_or_adopt_exact"]
