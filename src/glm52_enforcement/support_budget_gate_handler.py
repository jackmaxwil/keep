"""Trusted deployed-measurement gate Lambda entrypoint."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import os
import re
from typing import Mapping, Optional

from .canonical import canonical_sha256
from .decision_closure import parse_deployed_gate_document


_SHA = re.compile(r"^[0-9a-f]{64}$")
_MAX_VERSION_PAGES = 64
_LIST_PAGE_FIELDS = frozenset(
    {
        "Versions",
        "DeleteMarkers",
        "IsTruncated",
        "Name",
        "Prefix",
        "MaxKeys",
        "KeyMarker",
        "VersionIdMarker",
        "NextKeyMarker",
        "NextVersionIdMarker",
        "ResponseMetadata",
    }
)
_VERSION_FIELDS = frozenset(
    {
        "ETag",
        "ChecksumAlgorithm",
        "ChecksumType",
        "Size",
        "StorageClass",
        "Key",
        "VersionId",
        "IsLatest",
        "LastModified",
        "Owner",
        "RestoreStatus",
    }
)
_GET_OBJECT_FIELDS = frozenset(
    {
        "Body",
        "DeleteMarker",
        "AcceptRanges",
        "Expiration",
        "Restore",
        "ArchiveStatus",
        "LastModified",
        "ContentLength",
        "ChecksumCRC32",
        "ChecksumCRC32C",
        "ChecksumCRC64NVME",
        "ChecksumSHA1",
        "ChecksumSHA256",
        "ChecksumType",
        "ETag",
        "MissingMeta",
        "VersionId",
        "CacheControl",
        "ContentDisposition",
        "ContentEncoding",
        "ContentLanguage",
        "ContentRange",
        "ContentType",
        "Expires",
        "ExpiresString",
        "WebsiteRedirectLocation",
        "ServerSideEncryption",
        "Metadata",
        "SSECustomerAlgorithm",
        "SSECustomerKeyMD5",
        "SSEKMSKeyId",
        "BucketKeyEnabled",
        "StorageClass",
        "RequestCharged",
        "ReplicationStatus",
        "PartsCount",
        "TagCount",
        "ObjectLockMode",
        "ObjectLockRetainUntilDate",
        "ObjectLockLegalHoldStatus",
        "ResponseMetadata",
    }
)


@dataclass(frozen=True)
class BudgetGateConfig:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    bucket: str
    key: str
    expected_bucket_owner: str
    deployment_identity_sha256: str


def _validate_config(value: object) -> BudgetGateConfig:
    if type(value) is not BudgetGateConfig:
        raise ValueError("budget gate config must be exact and typed")
    if (
        value.account_id != "246813579024"
        or value.expected_bucket_owner != value.account_id
        or value.region != "us-west-2"
        or value.run_id != "glm52-sky-20260724"
        or type(value.activation_id) is not str
        or not value.activation_id
        or value.bucket
        != "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
        or value.key
        != (
            "rehearsal/gates/"
            + value.activation_id
            + "/CLOSURE_BUDGET.json"
        )
        or _SHA.fullmatch(value.deployment_identity_sha256) is None
    ):
        raise ValueError("budget gate config coordinates drifted")
    return value


def _environment_config() -> BudgetGateConfig:
    activation_id = os.environ.get("GLM52_ACTIVATION_ID", "")
    return _validate_config(
        BudgetGateConfig(
            account_id=os.environ.get("GLM52_ACCOUNT_ID", ""),
            region=os.environ.get("AWS_REGION", ""),
            run_id=os.environ.get("GLM52_RUN_ID", ""),
            activation_id=activation_id,
            bucket=os.environ.get("GLM52_REHEARSAL_BUCKET", ""),
            key=os.environ.get("GLM52_CLOSURE_GATE_KEY", ""),
            expected_bucket_owner=os.environ.get(
                "GLM52_EXPECTED_BUCKET_OWNER",
                "",
            ),
            deployment_identity_sha256=os.environ.get(
                "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
                "",
            ),
        )
    )


def _production_s3() -> object:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            connect_timeout=5,
            read_timeout=10,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )


def _success_metadata(value: object, label: str) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or type(value.get("ResponseMetadata")) is not dict
        or type(value["ResponseMetadata"].get("HTTPStatusCode")) is not int
        or value["ResponseMetadata"]["HTTPStatusCode"] != 200
        or type(value["ResponseMetadata"].get("RequestId")) is not str
        or not value["ResponseMetadata"]["RequestId"]
    ):
        raise ValueError(label + " response is not authenticated")
    return value


def _list_exact_version(
    *,
    list_versions: object,
    config: BudgetGateConfig,
) -> str:
    versions = []
    seen_versions = set()
    seen_markers = set()
    marker = None
    for _ in range(_MAX_VERSION_PAGES):
        request = {
            "Bucket": config.bucket,
            "Prefix": config.key,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": config.expected_bucket_owner,
        }
        if marker is not None:
            request["KeyMarker"] = marker[0]
            request["VersionIdMarker"] = marker[1]
        page = _success_metadata(
            list_versions(**request),
            "ListObjectVersions",
        )
        if (
            not {
                "Versions",
                "DeleteMarkers",
                "IsTruncated",
                "Name",
                "Prefix",
                "MaxKeys",
                "ResponseMetadata",
            }.issubset(page)
            or not set(page).issubset(_LIST_PAGE_FIELDS)
            or page["Name"] != config.bucket
            or page["Prefix"] != config.key
            or type(page["MaxKeys"]) is not int
            or page["MaxKeys"] != 1000
            or type(page["IsTruncated"]) is not bool
            or type(page["Versions"]) is not list
            or type(page["DeleteMarkers"]) is not list
        ):
            raise ValueError("budget gate version page drifted")
        if marker is None:
            if "KeyMarker" in page or "VersionIdMarker" in page:
                raise ValueError("budget gate marker echo drifted")
        elif (
            page.get("KeyMarker") != marker[0]
            or page.get("VersionIdMarker") != marker[1]
        ):
            raise ValueError("budget gate marker echo drifted")
        if page["DeleteMarkers"]:
            raise ValueError("budget gate version history has delete markers")
        for item in page["Versions"]:
            if (
                type(item) is not dict
                or not {
                    "Key",
                    "VersionId",
                    "IsLatest",
                    "Size",
                }.issubset(item)
                or not set(item).issubset(_VERSION_FIELDS)
                or item["Key"] != config.key
                or type(item["VersionId"]) is not str
                or not item["VersionId"]
                or type(item["IsLatest"]) is not bool
                or type(item["Size"]) is not int
                or item["Size"] < 0
            ):
                raise ValueError("budget gate version row drifted")
            identity = (item["Key"], item["VersionId"])
            if identity in seen_versions:
                raise ValueError("budget gate version row was duplicated")
            seen_versions.add(identity)
            versions.append(item)
        if page["IsTruncated"] is False:
            if (
                "NextKeyMarker" in page
                or "NextVersionIdMarker" in page
            ):
                raise ValueError("budget gate terminal markers drifted")
            break
        next_key = page.get("NextKeyMarker")
        next_version = page.get("NextVersionIdMarker")
        next_marker = (next_key, next_version)
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or next_marker == marker
            or next_marker in seen_markers
        ):
            raise ValueError("budget gate marker did not progress")
        seen_markers.add(next_marker)
        marker = next_marker
    else:
        raise ValueError("budget gate version pagination exceeded bound")
    if len(versions) != 1 or versions[0]["IsLatest"] is not True:
        raise ValueError("budget gate has no unique current VersionId")
    return str(versions[0]["VersionId"])


def main(
    event: object,
    context: object,
    *,
    services: Optional[object] = None,
    config: Optional[BudgetGateConfig] = None,
) -> Mapping[str, object]:
    """Authenticate the current immutable deployed rehearsal object."""

    del context
    if event != {
        "schema_version": 1,
        "record_type": "glm52_task11_budget_gate_request_v1",
    }:
        raise ValueError("budget gate request is not the exact static payload")
    config = _validate_config(
        _environment_config() if config is None else config
    )
    if services is None:
        services = _production_s3()
    list_versions = getattr(services, "list_object_versions", None)
    get_object = getattr(services, "get_object", None)
    if not callable(list_versions) or not callable(get_object):
        raise ValueError("budget gate S3 boundary is incomplete")
    version_id = _list_exact_version(
        list_versions=list_versions,
        config=config,
    )
    response = _success_metadata(
        get_object(
            Bucket=config.bucket,
            Key=config.key,
            VersionId=version_id,
            ExpectedBucketOwner=config.expected_bucket_owner,
            ChecksumMode="ENABLED",
        ),
        "GetObject",
    )
    if (
        not {
            "Body",
            "VersionId",
            "ContentLength",
            "ChecksumSHA256",
            "ResponseMetadata",
        }.issubset(response)
        or not set(response).issubset(_GET_OBJECT_FIELDS)
    ):
        raise ValueError("budget gate GetObject response drifted")
    body = response.get("Body")
    read = getattr(body, "read", None)
    if not callable(read):
        raise ValueError("budget gate body stream is absent")
    raw = read(2 * 1024 * 1024 + 1)
    if type(raw) is not bytes or len(raw) > 2 * 1024 * 1024:
        raise ValueError("budget gate body is absent or oversized")
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    metadata = response["ResponseMetadata"]
    headers = metadata.get("HTTPHeaders")
    if (
        response.get("VersionId") != version_id
        or response.get("ContentLength") != len(raw)
        or response.get("ChecksumSHA256") != checksum
        or type(headers) is not dict
        or headers.get("x-amz-version-id") != version_id
        or headers.get("x-amz-checksum-sha256") != checksum
    ):
        raise ValueError("budget gate VersionId/checksum readback drifted")
    gate = parse_deployed_gate_document(
        raw,
        activation_id=config.activation_id,
        deployment_identity_sha256=(
            config.deployment_identity_sha256
        ),
    )
    result = {
        "schema_version": 1,
        "record_type": "glm52_task11_trusted_budget_authority_v1",
        "status": "CLOSURE_BUDGET_PROVEN",
        "account_id": gate.account_id,
        "region": gate.region,
        "run_id": gate.run_id,
        "activation_id": gate.activation_id,
        "bucket": config.bucket,
        "key": config.key,
        "version_id": version_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": gate.canonical_body_sha256,
        "checksum_sha256_base64": checksum,
        "deployment_identity_sha256": (
            gate.deployment_identity_sha256
        ),
        "measurement_count": gate.measurement_count,
        "cold_environment_count": gate.cold_environment_count,
        "environment_set_sha256": gate.environment_set_sha256,
        "measurements_identity_sha256": (
            gate.measurements_identity_sha256
        ),
    }
    result["canonical_identity_sha256"] = canonical_sha256(result)
    return result


__all__ = ["BudgetGateConfig", "main"]
