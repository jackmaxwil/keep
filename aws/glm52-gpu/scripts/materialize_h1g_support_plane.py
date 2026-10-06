#!/usr/bin/env python3
"""Materialize Task 7 postcreate authority from exact read-only AWS truth."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.live_authority import (  # noqa: E402
    RUNTIME_REVALIDATION_SOURCE_KINDS,
    runtime_revalidation_sources_from_mapping,
    validate_runtime_revalidation_source_documents,
)
from glm52_enforcement.support_plane import (  # noqa: E402
    SupportMaterializationServices,
    coordinate_support_postcreate,
    coordinate_support_task12_postpublication,
    support_build_inputs_from_mapping,
    support_inputs_projection,
)
from glm52_enforcement.task11_boundary import (  # noqa: E402
    build_task11_input_coordinate,
)
from glm52_enforcement.task12_postpublication import (  # noqa: E402
    Task12OperationSourceCoordinate,
    Task12PostpublicationInputs,
    Task12PostpublicationServices,
)

EXPECTED_PROFILE = "keep-gpu"
EXPECTED_ACCOUNT_ID = "246813579024"
EXPECTED_REGION = "us-west-2"
EXPECTED_STACK_NAME = "keep-glm52-h1g-support"
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"keep-glm52-h1g-support/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_RETAINED_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"[A-Za-z0-9._-]+/[0-9a-fA-F-]{36}\Z"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KMS_KEY_ARN = re.compile(
    r"arn:aws:kms:us-west-2:246813579024:key/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_RUNTIME_RECORD_TYPES = {
    kind: "glm52_task11_" + kind.lower() + "_v1"
    for kind in RUNTIME_REVALIDATION_SOURCE_KINDS
}
_SOURCE_RUNTIME_KINDS = {
    "TASK6_MIGRATION_MANIFEST": (
        "task6_manifest",
        "glm52_h1g_stack_migration_manifest_v1",
    ),
    "TASK6_TEMPLATE_INVENTORY": (
        "task6_templates",
        "glm52_h1d_task6_task7_template_bundle_v1",
    ),
    "TASK7_POSTCREATE_MANIFEST": (
        "task7_postcreate_manifest",
        "glm52_h1g_support_postcreate_manifest_v1",
    ),
    "TASK7_SUPPORT_INVENTORY": (
        "task7_inventory",
        "glm52_h1g_support_postcreate_inventory_v1",
    ),
}
_H1D_LIVE_REQUEST_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "generation",
    "expected_state",
    "expected_state_authentication",
    "runtime_revalidation_sources",
    "spend_request_template",
    "sky_probe_request",
    "sky_probe_admission_identity_sha256",
    "canonical_identity_sha256",
}


def _read_canonical(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{path}: canonical JSON must end in exactly one LF")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid ASCII JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{path}: input must be one exact JSON object")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{path}: JSON bytes are not canonical")
    return value


def _success_response(value: object, operation: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{operation} returned no exact object")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise ValueError(f"{operation} did not return authenticated success")
    return value


class Boto3ReadRunner:
    """One no-SDK-retry session exposing only the required read clients."""

    def __init__(self, *, profile: str, region: str) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("boto3 and botocore are required") from exc
        config = Config(
            connect_timeout=5,
            read_timeout=30,
            region_name=region,
            retries={"mode": "standard", "total_max_attempts": 1},
        )
        session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )
        if session.region_name != region:
            raise ValueError("AWS session region drifted")
        self._sts = session.client("sts", config=config)
        self.cloudformation = session.client(
            "cloudformation", config=config
        )
        self.ec2 = session.client("ec2", config=config)
        self.s3 = session.client("s3", config=config)
        self.dynamodb = session.client("dynamodb", config=config)

    def get_caller_identity(self) -> Mapping[str, object]:
        return self._sts.get_caller_identity()


def _guard_caller(
    *,
    profile: str,
    region: str,
    support_stack_id: str,
    runner: object,
) -> None:
    if profile != EXPECTED_PROFILE:
        raise ValueError("postcreate materialization requires keep-gpu")
    if region != EXPECTED_REGION:
        raise ValueError("postcreate materialization requires us-west-2")
    if (
        type(support_stack_id) is not str
        or _STACK_ID.fullmatch(support_stack_id) is None
    ):
        raise ValueError("postcreate materialization requires the exact stack ID")
    identity_method = getattr(runner, "get_caller_identity", None)
    if not callable(identity_method):
        raise TypeError("postcreate runner has no STS identity boundary")
    identity = _success_response(
        identity_method(), "GetCallerIdentity"
    )
    if (
        identity.get("Account") != EXPECTED_ACCOUNT_ID
        or type(identity.get("Arn")) is not str
        or f"::{EXPECTED_ACCOUNT_ID}:" not in identity["Arn"]
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise ValueError("postcreate caller is not the exact campaign account")


def _guard_publication_caller(
    *,
    profile: str,
    region: str,
    runner: object,
) -> None:
    if profile != EXPECTED_PROFILE or region != EXPECTED_REGION:
        raise ValueError(
            "runtime publication requires keep-gpu/us-west-2"
        )
    identity_method = getattr(runner, "get_caller_identity", None)
    if not callable(identity_method):
        raise TypeError("runtime publication has no STS identity boundary")
    identity = _success_response(
        identity_method(), "GetCallerIdentity"
    )
    if (
        identity.get("Account") != EXPECTED_ACCOUNT_ID
        or type(identity.get("Arn")) is not str
        or f"::{EXPECTED_ACCOUNT_ID}:" not in identity["Arn"]
        or type(identity.get("UserId")) is not str
        or not identity["UserId"]
    ):
        raise ValueError(
            "runtime publication caller is not the campaign account"
        )


def _write_new_artifacts(
    output_dir: Path,
    artifacts: Mapping[str, object],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for file_name in sorted(artifacts):
        path = output_dir / file_name
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(artifacts[file_name]) + b"\n")


def _body_identity(document: Mapping[str, object]) -> str:
    for field in (
        "canonical_identity_sha256",
        "canonical_body_sha256",
    ):
        if field in document:
            body = dict(document)
            identity = body.pop(field)
            if (
                type(identity) is not str
                or identity != canonical_sha256(body)
            ):
                raise ValueError("runtime source self identity drifted")
            return identity
    return canonical_sha256(document)


def _runtime_document(
    *,
    kind: str,
    activation_id: str,
    generation: int,
    authority: Mapping[str, object],
) -> dict[str, object]:
    if (
        kind not in _RUNTIME_RECORD_TYPES
        or type(authority) is not dict
    ):
        raise ValueError("runtime source authority is not exact")
    body = {
        "schema_version": 1,
        "record_type": _RUNTIME_RECORD_TYPES[kind],
        "account_id": EXPECTED_ACCOUNT_ID,
        "region": EXPECTED_REGION,
        "run_id": "glm52-sky-20260724",
        "activation_id": activation_id,
        "generation": generation,
        "input_kind": kind,
        "authority": dict(authority),
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _source_projection(
    *,
    kind: str,
    document: Mapping[str, object],
    coordinate: Mapping[str, object],
) -> Mapping[str, object]:
    source_name, record_type = _SOURCE_RUNTIME_KINDS[kind]
    del source_name
    fields = {
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
    raw = canonical_json_bytes(document) + b"\n"
    body_identity = _body_identity(document)
    if (
        type(coordinate) is not dict
        or set(coordinate) != fields
        or document.get("record_type") != record_type
        or coordinate["bucket"]
        != "keep-glm52-models-246813579024-us-west-2"
        or type(coordinate["key"]) is not str
        or not coordinate["key"]
        or type(coordinate["version_id"]) is not str
        or not coordinate["version_id"]
        or coordinate["version_id"] == "null"
        or coordinate["file_sha256"]
        != hashlib.sha256(raw).hexdigest()
        or coordinate["body_sha256"] != body_identity
    ):
        raise ValueError(kind + " authenticated source drifted")
    common = {
        "source_coordinate": dict(coordinate),
        "source_record_type": record_type,
        "source_body_identity_sha256": body_identity,
    }
    if kind == "TASK6_MIGRATION_MANIFEST":
        stacks = document.get("stacks")
        artifacts = document.get("artifacts")
        ownership = document.get("ownership")
        if (
            document.get("account_id") != EXPECTED_ACCOUNT_ID
            or document.get("region") != EXPECTED_REGION
            or document.get("run_id") != "glm52-sky-20260724"
            or document.get("bucket_name") != coordinate["bucket"]
            or type(document.get("retained_deployment_role_id")) is not str
            or not document["retained_deployment_role_id"]
            or type(stacks) is not list
            or len(stacks) != 3
            or type(artifacts) is not list
            or not artifacts
            or type(ownership) is not list
            or not ownership
        ):
            raise ValueError("Task 6 migration semantics are incomplete")
        return {
            **common,
            "bucket_name": document["bucket_name"],
            "retained_deployment_role_id": document[
                "retained_deployment_role_id"
            ],
            "stacks": stacks,
            "artifacts_identity_sha256": canonical_sha256(artifacts),
            "ownership_identity_sha256": canonical_sha256(ownership),
        }
    if kind == "TASK6_TEMPLATE_INVENTORY":
        task6 = document.get("task6_templates")
        support = document.get("task7_support_template")
        if (
            type(task6) is not dict
            or set(task6)
            != {
                "retention-only",
                "post-retain",
                "fence-import",
                "final-fence",
            }
            or type(support) is not dict
        ):
            raise ValueError("Task 6/7 template inventory is incomplete")
        return {
            **common,
            "template_identities": {
                **{
                    name: canonical_sha256(template)
                    for name, template in task6.items()
                },
                "disabled-support": canonical_sha256(support),
            },
        }
    if kind == "TASK7_POSTCREATE_MANIFEST":
        if (
            document.get("account_id") != EXPECTED_ACCOUNT_ID
            or document.get("region") != EXPECTED_REGION
            or document.get("run_id") != "glm52-sky-20260724"
            or type(document.get("support_stack_id")) is not str
            or not document["support_stack_id"]
            or _SHA256.fullmatch(
                document.get("support_template_body_sha256", "")
            )
            is None
            or _SHA256.fullmatch(
                document.get("support_postcreate_inventory_sha256", "")
            )
            is None
        ):
            raise ValueError("Task 7 postcreate manifest is incomplete")
        return {
            **common,
            "support_stack_id": document["support_stack_id"],
            "support_template_body_sha256": document[
                "support_template_body_sha256"
            ],
            "support_postcreate_inventory_sha256": document[
                "support_postcreate_inventory_sha256"
            ],
        }
    stack_resources = document.get("stack_resources")
    action_resources = document.get("action_resources")
    if (
        type(document.get("support_stack_id")) is not str
        or not document["support_stack_id"]
        or _SHA256.fullmatch(
            document.get("support_template_body_sha256", "")
        )
        is None
        or type(stack_resources) is not list
        or not stack_resources
        or type(action_resources) is not dict
        or not action_resources
        or _SHA256.fullmatch(
            document.get("describe_stacks_response_sha256", "")
        )
        is None
        or type(document.get("list_stack_resources_response_sha256"))
        is not list
        or not document["list_stack_resources_response_sha256"]
    ):
        raise ValueError("Task 7 support inventory is incomplete")
    return {
        **common,
        "support_stack_id": document["support_stack_id"],
        "support_template_body_sha256": document[
            "support_template_body_sha256"
        ],
        "stack_resources_identity_sha256": canonical_sha256(
            stack_resources
        ),
        "action_resources_identity_sha256": canonical_sha256(
            action_resources
        ),
        "describe_stacks_response_sha256": document[
            "describe_stacks_response_sha256"
        ],
        "list_stack_resources_response_sha256": document[
            "list_stack_resources_response_sha256"
        ],
    }


def materialize_runtime_revalidation_documents(
    *,
    activation_id: str,
    generation: int,
    source_documents: Mapping[str, object],
    source_coordinates: Mapping[str, object],
    semantic_authorities: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    """Build all seven canonical Task 11 sources from authenticated truth."""

    if (
        type(activation_id) is not str
        or not activation_id
        or type(generation) is not int
        or generation <= 0
        or type(source_documents) is not dict
        or set(source_documents)
        != {item[0] for item in _SOURCE_RUNTIME_KINDS.values()}
        or type(source_coordinates) is not dict
        or set(source_coordinates) != set(source_documents)
        or type(semantic_authorities) is not dict
        or set(semantic_authorities)
        != (
            set(RUNTIME_REVALIDATION_SOURCE_KINDS)
            - set(_SOURCE_RUNTIME_KINDS)
        )
    ):
        raise ValueError(
            "runtime revalidation materialization input is incomplete"
        )
    result: dict[str, Mapping[str, object]] = {}
    for kind, (source_name, _record_type) in (
        _SOURCE_RUNTIME_KINDS.items()
    ):
        document = source_documents[source_name]
        if type(document) is not dict:
            raise ValueError("runtime source document is not exact")
        result[kind] = _runtime_document(
            kind=kind,
            activation_id=activation_id,
            generation=generation,
            authority=_source_projection(
                kind=kind,
                document=document,
                coordinate=source_coordinates[source_name],
            ),
        )
    for kind in (
        set(RUNTIME_REVALIDATION_SOURCE_KINDS)
        - set(_SOURCE_RUNTIME_KINDS)
    ):
        result[kind] = _runtime_document(
            kind=kind,
            activation_id=activation_id,
            generation=generation,
            authority=semantic_authorities[kind],
        )
    validate_runtime_revalidation_source_documents(
        {
            kind: result[kind]
            for kind in (
                "RUNTIME_CUTOFF_AUTHORITY",
                "RUNTIME_ATTACHMENT_INVENTORY",
                "CREDENTIAL_PROBE_INVENTORY",
            )
        },
        activation_id=activation_id,
        generation=generation,
        task7_inventory_identity_sha256=_body_identity(
            source_documents["task7_inventory"]
        ),
    )
    return {
        kind: result[kind]
        for kind in RUNTIME_REVALIDATION_SOURCE_KINDS
    }


def _runtime_revalidation_key(
    *,
    activation_id: str,
    generation: int,
    kind: str,
) -> str:
    index = RUNTIME_REVALIDATION_SOURCE_KINDS.index(kind) + 1
    return (
        "campaigns/glm52-sky-20260724/authorities/task11/"
        + activation_id
        + "/"
        + f"{generation:08d}"
        + "/runtime-revalidation/"
        + f"{index:02d}-"
        + kind.lower().replace("_", "-")
        + ".json"
    )


def _publication_response(
    value: object,
    *,
    operation: str,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(operation + " returned no exact response")
    metadata = value.get("ResponseMetadata")
    headers = (
        metadata.get("HTTPHeaders")
        if type(metadata) is dict
        else None
    )
    date_value = (
        headers.get("date")
        if type(headers) is dict
        else None
    )
    try:
        service_date = parsedate_to_datetime(date_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            operation + " service Date is absent"
        ) from exc
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(metadata.get("HostId")) is not str
        or not metadata["HostId"]
        or type(metadata.get("RetryAttempts")) is not int
        or metadata["RetryAttempts"] != 0
        or service_date.tzinfo is None
    ):
        raise ValueError(operation + " is not authenticated success")
    return value


def _publication_metadata(
    *,
    document: Mapping[str, object],
    input_kind: str,
    body_sha256: str,
) -> Mapping[str, str]:
    return {
        "glm52-account-id": EXPECTED_ACCOUNT_ID,
        "glm52-region": EXPECTED_REGION,
        "glm52-run-id": "glm52-sky-20260724",
        "glm52-activation-id": str(document["activation_id"]),
        "glm52-generation": f"{int(document['generation']):08d}",
        "glm52-input-kind": input_kind,
        "glm52-record-type": str(document["record_type"]),
        "glm52-body-sha256": body_sha256,
    }


def _list_exact_object_versions(
    *,
    s3: object,
    bucket: str,
    key: str,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]], tuple[str, ...]]:
    get_paginator = getattr(s3, "get_paginator", None)
    if not callable(get_paginator):
        raise ValueError("S3 ListObjectVersions paginator is absent")
    pages = get_paginator("list_object_versions").paginate(
        Bucket=bucket,
        Prefix=key,
        ExpectedBucketOwner=EXPECTED_ACCOUNT_ID,
    )
    versions: list[Mapping[str, object]] = []
    delete_markers: list[Mapping[str, object]] = []
    request_ids: list[str] = []
    page_identities: set[str] = set()
    for page_index, page in enumerate(pages):
        if page_index >= 64:
            raise ValueError("ListObjectVersions exceeded page bound")
        authenticated = _publication_response(
            page,
            operation="ListObjectVersions",
        )
        page_identity = canonical_sha256(
            {
                key: value
                for key, value in authenticated.items()
                if key != "ResponseMetadata"
            }
        )
        if page_identity in page_identities:
            raise ValueError("ListObjectVersions page cycle detected")
        page_identities.add(page_identity)
        request_ids.append(
            authenticated["ResponseMetadata"]["RequestId"]
        )
        for field, target in (
            ("Versions", versions),
            ("DeleteMarkers", delete_markers),
        ):
            values = authenticated.get(field, [])
            if type(values) is not list:
                raise ValueError(
                    "ListObjectVersions " + field + " drifted"
                )
            for item in values:
                if type(item) is not dict:
                    raise ValueError(
                        "ListObjectVersions entry drifted"
                    )
                if item.get("Key") == key:
                    target.append(item)
        if (
            authenticated.get("IsTruncated") is True
            and not any(
                type(authenticated.get(field)) is str
                and authenticated[field]
                for field in ("NextKeyMarker", "NextVersionIdMarker")
            )
        ):
            raise ValueError(
                "ListObjectVersions truncation marker is absent"
            )
    if not request_ids:
        raise ValueError("ListObjectVersions returned no page")
    return versions, delete_markers, tuple(request_ids)


def _exact_get_published_document(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: str,
    raw: bytes,
    checksum_sha256: str,
    etag: str,
    kms_key_arn: str,
    metadata: Mapping[str, str],
) -> str:
    get_object = getattr(s3, "get_object", None)
    if not callable(get_object):
        raise ValueError("S3 exact GetObject is absent")
    response = _publication_response(
        get_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=EXPECTED_ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        operation="GetObject",
    )
    stream = response.get("Body")
    read = getattr(stream, "read", None)
    if (
        response.get("VersionId") != version_id
        or response.get("ContentLength") != len(raw)
        or response.get("ChecksumSHA256") != checksum_sha256
        or response.get("ETag") != etag
        or response.get("ContentType") != "application/json"
        or response.get("Metadata") != metadata
        or response.get("ServerSideEncryption") != "aws:kms"
        or response.get("SSEKMSKeyId") != kms_key_arn
        or response.get("BucketKeyEnabled") is not True
        or not callable(read)
        or read(len(raw) + 1) != raw
    ):
        raise ValueError("exact published GetObject drifted")
    return response["ResponseMetadata"]["RequestId"]


def _publish_immutable_document(
    *,
    s3: object,
    bucket: str,
    key: str,
    kms_key_arn: str,
    input_kind: str,
    document: Mapping[str, object],
) -> Mapping[str, object]:
    raw = canonical_json_bytes(document) + b"\n"
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("immutable publication exceeds size bound")
    body_sha256 = _body_identity(document)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    checksum_sha256 = base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii")
    metadata = _publication_metadata(
        document=document,
        input_kind=input_kind,
        body_sha256=body_sha256,
    )
    versions_before, deletes_before, preflight_ids = (
        _list_exact_object_versions(
            s3=s3,
            bucket=bucket,
            key=key,
        )
    )
    if versions_before or deletes_before:
        raise ValueError(
            "immutable publication key was already materialized"
        )
    put_object = getattr(s3, "put_object", None)
    if not callable(put_object):
        raise ValueError("S3 conditional PutObject is absent")
    put_request = {
        "Bucket": bucket,
        "Key": key,
        "Body": raw,
        "ExpectedBucketOwner": EXPECTED_ACCOUNT_ID,
        "IfNoneMatch": "*",
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": checksum_sha256,
        "ContentType": "application/json",
        "Metadata": metadata,
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": kms_key_arn,
        "BucketKeyEnabled": True,
    }
    put_response: Optional[Mapping[str, object]] = None
    put_error: Optional[BaseException] = None
    try:
        raw_put_response = put_object(**put_request)
    except Exception as exc:
        put_error = exc
    else:
        put_response = _publication_response(
            raw_put_response,
            operation="PutObject",
        )
    versions, delete_markers, reconciliation_ids = (
        _list_exact_object_versions(
            s3=s3,
            bucket=bucket,
            key=key,
        )
    )
    if len(versions) != 1 or delete_markers:
        if put_error is not None:
            raise ValueError(
                "ambiguous PutObject could not be reconciled"
            ) from put_error
        raise ValueError(
            "immutable publication cardinality drifted"
        )
    version = versions[0]
    version_id = version.get("VersionId")
    etag = version.get("ETag")
    if (
        type(version_id) is not str
        or not version_id
        or version_id == "null"
        or type(etag) is not str
        or not etag
        or version.get("Size") != len(raw)
        or version.get("IsLatest") is not True
    ):
        raise ValueError("published S3 version evidence drifted")
    put_request_id = None
    if put_response is not None:
        if (
            put_response.get("VersionId") != version_id
            or put_response.get("ChecksumSHA256") != checksum_sha256
            or put_response.get("ETag") != etag
        ):
            raise ValueError("PutObject response identity drifted")
        put_request_id = put_response["ResponseMetadata"]["RequestId"]
    exact_get_request_id = _exact_get_published_document(
        s3=s3,
        bucket=bucket,
        key=key,
        version_id=version_id,
        raw=raw,
        checksum_sha256=checksum_sha256,
        etag=etag,
        kms_key_arn=kms_key_arn,
        metadata=metadata,
    )
    final_versions, final_delete_markers, final_ids = (
        _list_exact_object_versions(
            s3=s3,
            bucket=bucket,
            key=key,
        )
    )
    if (
        len(final_versions) != 1
        or final_delete_markers
        or final_versions[0].get("VersionId") != version_id
        or final_versions[0].get("ETag") != etag
        or final_versions[0].get("Size") != len(raw)
        or final_versions[0].get("IsLatest") is not True
    ):
        raise ValueError(
            "immutable publication changed after exact GetObject"
        )
    coordinate_body = {
        "input_kind": input_kind,
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "file_sha256": file_sha256,
        "body_sha256": body_sha256,
    }
    return {
        "coordinate": {
            **coordinate_body,
            "canonical_identity_sha256": canonical_sha256(
                coordinate_body
            ),
        },
        "evidence": {
            "put_request_id": put_request_id,
            "ambiguity_reconciled": put_error is not None,
            "preflight_request_ids": list(preflight_ids),
            "reconciliation_request_ids": list(
                reconciliation_ids
            ),
            "exact_get_request_id": exact_get_request_id,
            "final_request_ids": list(final_ids),
            "version_id": version_id,
            "checksum_sha256": checksum_sha256,
            "etag": etag,
        },
    }


def inject_runtime_sources_into_h1d_live_request(
    *,
    h1d_live_request: Mapping[str, object],
    runtime_revalidation_sources: Mapping[str, object],
    activation_id: str,
    generation: int,
) -> Mapping[str, object]:
    """Bind exact seven-source coordinates and refresh the H1D self identity."""

    if type(h1d_live_request) is not dict:
        raise ValueError("H1D live request is not exact")
    prior = dict(h1d_live_request)
    prior_identity = prior.pop("canonical_identity_sha256", None)
    if (
        set(h1d_live_request) != _H1D_LIVE_REQUEST_FIELDS
        or prior_identity != canonical_sha256(prior)
        or prior.get("schema_version") != 1
        or prior.get("record_type")
        != "glm52_task11_h1d_live_input_v1"
        or prior.get("account_id") != EXPECTED_ACCOUNT_ID
        or prior.get("region") != EXPECTED_REGION
        or prior.get("run_id") != "glm52-sky-20260724"
        or prior.get("activation_id") != activation_id
        or prior.get("generation") != generation
        or prior.get("runtime_revalidation_sources") != {}
        or type(prior.get("expected_state")) is not dict
        or type(prior.get("expected_state_authentication")) is not dict
        or type(prior.get("spend_request_template")) is not dict
        or type(prior.get("sky_probe_request")) is not dict
        or type(
            prior.get("sky_probe_admission_identity_sha256")
        )
        is not str
        or _SHA256.fullmatch(
            prior["sky_probe_admission_identity_sha256"]
        ) is None
    ):
        raise ValueError("H1D live request base identity drifted")
    runtime_revalidation_sources_from_mapping(
        runtime_revalidation_sources,
        activation_id=activation_id,
        generation=generation,
    )
    body = {
        **prior,
        "runtime_revalidation_sources": dict(
            runtime_revalidation_sources
        ),
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def publish_runtime_revalidation_bundle(
    *,
    s3: object,
    bucket: str,
    kms_key_arn: str,
    activation_id: str,
    generation: int,
    source_documents: Mapping[str, object],
    source_coordinates: Mapping[str, object],
    semantic_authorities: Mapping[str, object],
    h1d_live_request: Mapping[str, object],
) -> Mapping[str, object]:
    """Build, validate, conditionally publish, and exact-read Task 11 truth."""

    if (
        bucket != "keep-glm52-models-246813579024-us-west-2"
        or _KMS_KEY_ARN.fullmatch(kms_key_arn) is None
    ):
        raise ValueError("immutable publication destination drifted")
    documents = materialize_runtime_revalidation_documents(
        activation_id=activation_id,
        generation=generation,
        source_documents=source_documents,
        source_coordinates=source_coordinates,
        semantic_authorities=semantic_authorities,
    )
    # Validate the H1D carrier before any immutable source is written.  The
    # actual coordinates are injected only after S3 returns their versions.
    empty_h1d = dict(h1d_live_request)
    inject_runtime_sources_into_h1d_live_request(
        h1d_live_request=empty_h1d,
        runtime_revalidation_sources={
            kind: {
                "input_kind": kind,
                "bucket": bucket,
                "key": _runtime_revalidation_key(
                    activation_id=activation_id,
                    generation=generation,
                    kind=kind,
                ),
                "version_id": "preflight-version",
                "file_sha256": "0" * 64,
                "body_sha256": documents[kind][
                    "canonical_identity_sha256"
                ],
                "canonical_identity_sha256": canonical_sha256(
                    {
                        "input_kind": kind,
                        "bucket": bucket,
                        "key": _runtime_revalidation_key(
                            activation_id=activation_id,
                            generation=generation,
                            kind=kind,
                        ),
                        "version_id": "preflight-version",
                        "file_sha256": "0" * 64,
                        "body_sha256": documents[kind][
                            "canonical_identity_sha256"
                        ],
                    }
                ),
            }
            for kind in RUNTIME_REVALIDATION_SOURCE_KINDS
        },
        activation_id=activation_id,
        generation=generation,
    )
    source_result: dict[str, object] = {}
    publication_evidence: dict[str, object] = {}
    for kind in RUNTIME_REVALIDATION_SOURCE_KINDS:
        published = _publish_immutable_document(
            s3=s3,
            bucket=bucket,
            key=_runtime_revalidation_key(
                activation_id=activation_id,
                generation=generation,
                kind=kind,
            ),
            kms_key_arn=kms_key_arn,
            input_kind=kind,
            document=documents[kind],
        )
        source_result[kind] = published["coordinate"]
        publication_evidence[kind] = published["evidence"]
    runtime_revalidation_sources_from_mapping(
        source_result,
        activation_id=activation_id,
        generation=generation,
    )
    h1d_document = inject_runtime_sources_into_h1d_live_request(
        h1d_live_request=h1d_live_request,
        runtime_revalidation_sources=source_result,
        activation_id=activation_id,
        generation=generation,
    )
    h1d_key = (
        "campaigns/glm52-sky-20260724/authorities/task11/"
        + activation_id
        + "/"
        + f"{generation:08d}"
        + "/11-h1d-live-request.json"
    )
    h1d_published = _publish_immutable_document(
        s3=s3,
        bucket=bucket,
        key=h1d_key,
        kms_key_arn=kms_key_arn,
        input_kind="H1D_LIVE_REQUEST",
        document=h1d_document,
    )
    h1d_coordinate = build_task11_input_coordinate(
        **{
            field: h1d_published["coordinate"][field]
            for field in (
                "input_kind",
                "bucket",
                "key",
                "version_id",
                "file_sha256",
                "body_sha256",
            )
        }
    )
    result_body = {
        "runtime_revalidation_sources": source_result,
        "h1d_live_request": asdict(h1d_coordinate),
        "publication_evidence": {
            **publication_evidence,
            "H1D_LIVE_REQUEST": h1d_published["evidence"],
        },
    }
    return {
        **result_body,
        "canonical_identity_sha256": canonical_sha256(result_body),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--support-stack-id")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--support-template", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--publish-runtime-revalidation",
        action="store_true",
        help=(
            "explicitly enable immutable seven-source and H1D publication"
        ),
    )
    parser.add_argument(
        "--runtime-revalidation-request",
        type=Path,
    )
    parser.add_argument(
        "--runtime-revalidation-output",
        type=Path,
    )
    parser.add_argument(
        "--publish-task12-postpublication",
        action="store_true",
        help=(
            "read retained CFN versions and conditionally materialize the "
            "32 operation inputs plus nine Task 12 deployment records"
        ),
    )
    parser.add_argument("--task12-postpublication-request", type=Path)
    parser.add_argument("--task12-postpublication-output", type=Path)
    return parser


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    runner_factory: Callable[..., object] = Boto3ReadRunner,
) -> int:
    args = _parser().parse_args(argv)
    if args.publish_task12_postpublication:
        if (
            args.task12_postpublication_request is None
            or args.task12_postpublication_output is None
            or args.task12_postpublication_output.exists()
            or args.publish_runtime_revalidation
            or args.runtime_revalidation_request is not None
            or args.runtime_revalidation_output is not None
            or any(
                value is not None
                for value in (
                    args.support_stack_id,
                    args.inputs,
                    args.support_template,
                    args.output_dir,
                )
            )
        ):
            raise ValueError(
                "Task 12 postpublication mode has a closed argument surface"
            )
        if os.environ.get("AWS_PAGER", ""):
            raise ValueError("AWS_PAGER must be empty for Task 12 publication")
        request = _read_canonical(args.task12_postpublication_request)
        expected_fields = {
            "schema_version",
            "record_type",
            "retained_stack_id",
            "support_stack_id",
            "activation_id",
            "activation_ordinal",
            "generation",
            "dispatch_identity_sha256",
            "ledger_table_name",
            "authority_bucket",
            "campaign_bucket",
            "kms_key_id",
            "worker_drain_document_name",
            "worker_drain_document_version",
            "operation_source",
            "campaign_descriptor",
            "gpu_spend_approval",
            "canonical_identity_sha256",
        }
        request_body = dict(request)
        request_identity = request_body.pop("canonical_identity_sha256", None)
        operation_source = request.get("operation_source")
        campaign_descriptor = request.get("campaign_descriptor")
        gpu_spend_approval = request.get("gpu_spend_approval")
        if (
            set(request) != expected_fields
            or request.get("schema_version") != 1
            or request.get("record_type")
            != "glm52_task12_postpublication_request_v1"
            or request_identity != canonical_sha256(request_body)
            or type(operation_source) is not dict
            or set(operation_source)
            != {"bucket", "key", "version_id", "file_sha256"}
            or type(campaign_descriptor) is not dict
            or set(campaign_descriptor)
            != {"bucket", "key", "version_id", "file_sha256"}
            or type(gpu_spend_approval) is not dict
            or set(gpu_spend_approval)
            != {"bucket", "key", "version_id", "file_sha256"}
            or _RETAINED_STACK_ID.fullmatch(
                str(request.get("retained_stack_id"))
            )
            is None
            or _STACK_ID.fullmatch(
                str(request.get("support_stack_id"))
            )
            is None
        ):
            raise ValueError("Task 12 postpublication request is not exact")
        runner = runner_factory(profile=args.profile, region=args.region)
        _guard_publication_caller(
            profile=args.profile,
            region=args.region,
            runner=runner,
        )
        cloudformation = getattr(runner, "cloudformation", None)
        dynamodb = getattr(runner, "dynamodb", None)
        s3 = getattr(runner, "s3", None)
        if cloudformation is None or dynamodb is None or s3 is None:
            raise TypeError(
                "Task 12 postpublication runner lacks exact AWS clients"
            )
        materialization_inputs = Task12PostpublicationInputs(
            retained_stack_id=request["retained_stack_id"],
            support_stack_id=request["support_stack_id"],
            activation_id=request["activation_id"],
            activation_ordinal=request["activation_ordinal"],
            generation=request["generation"],
            dispatch_identity_sha256=request[
                "dispatch_identity_sha256"
            ],
            ledger_table_name=request["ledger_table_name"],
            authority_bucket=request["authority_bucket"],
            campaign_bucket=request["campaign_bucket"],
            kms_key_id=request["kms_key_id"],
            worker_drain_document_name=request[
                "worker_drain_document_name"
            ],
            worker_drain_document_version=request[
                "worker_drain_document_version"
            ],
            operation_source=Task12OperationSourceCoordinate(
                bucket=request["operation_source"]["bucket"],
                key=request["operation_source"]["key"],
                version_id=request["operation_source"]["version_id"],
                file_sha256=request["operation_source"]["file_sha256"],
            ),
            campaign_descriptor=Task12OperationSourceCoordinate(
                bucket=request["campaign_descriptor"]["bucket"],
                key=request["campaign_descriptor"]["key"],
                version_id=request["campaign_descriptor"]["version_id"],
                file_sha256=request["campaign_descriptor"][
                    "file_sha256"
                ],
            ),
            gpu_spend_approval=Task12OperationSourceCoordinate(
                bucket=request["gpu_spend_approval"]["bucket"],
                key=request["gpu_spend_approval"]["key"],
                version_id=request["gpu_spend_approval"]["version_id"],
                file_sha256=request["gpu_spend_approval"][
                    "file_sha256"
                ],
            ),
        )
        result = coordinate_support_task12_postpublication(
            inputs=materialization_inputs,
            services=Task12PostpublicationServices(
                cloudformation=cloudformation,
                dynamodb=dynamodb,
                s3=s3,
            ),
        )
        args.task12_postpublication_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with args.task12_postpublication_output.open("xb") as handle:
            handle.write(canonical_json_bytes(result) + b"\n")
        return 0
    if args.publish_runtime_revalidation:
        if (
            args.runtime_revalidation_request is None
            or args.runtime_revalidation_output is None
            or any(
                value is not None
                for value in (
                    args.support_stack_id,
                    args.inputs,
                    args.support_template,
                    args.output_dir,
                )
            )
        ):
            raise ValueError(
                "runtime publication mode has a closed argument surface"
            )
        if args.runtime_revalidation_output.exists():
            raise FileExistsError(
                "refusing to overwrite runtime publication evidence"
            )
        if os.environ.get("AWS_PAGER", ""):
            raise ValueError(
                "AWS_PAGER must be empty for runtime publication"
            )
        request = _read_canonical(
            args.runtime_revalidation_request
        )
        if set(request) != {
            "schema_version",
            "record_type",
            "bucket",
            "kms_key_arn",
            "activation_id",
            "generation",
            "source_documents",
            "source_coordinates",
            "semantic_authorities",
            "h1d_live_request",
            "canonical_identity_sha256",
        }:
            raise ValueError(
                "runtime publication request is not closed"
            )
        request_body = dict(request)
        request_identity = request_body.pop(
            "canonical_identity_sha256"
        )
        if (
            request["schema_version"] != 1
            or request["record_type"]
            != "glm52_h1g_runtime_revalidation_publication_request_v1"
            or request_identity != canonical_sha256(request_body)
        ):
            raise ValueError(
                "runtime publication request identity drifted"
            )
        runner = runner_factory(
            profile=args.profile,
            region=args.region,
        )
        _guard_publication_caller(
            profile=args.profile,
            region=args.region,
            runner=runner,
        )
        s3 = getattr(runner, "s3", None)
        if s3 is None:
            raise TypeError("runtime publication runner lacks S3")
        result = publish_runtime_revalidation_bundle(
            s3=s3,
            bucket=request["bucket"],
            kms_key_arn=request["kms_key_arn"],
            activation_id=request["activation_id"],
            generation=request["generation"],
            source_documents=request["source_documents"],
            source_coordinates=request["source_coordinates"],
            semantic_authorities=request["semantic_authorities"],
            h1d_live_request=request["h1d_live_request"],
        )
        args.runtime_revalidation_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with args.runtime_revalidation_output.open("xb") as handle:
            handle.write(canonical_json_bytes(result) + b"\n")
        return 0
    if (
        args.task12_postpublication_request is not None
        or args.task12_postpublication_output is not None
        or
        args.runtime_revalidation_request is not None
        or args.runtime_revalidation_output is not None
        or any(
            value is None
            for value in (
                args.support_stack_id,
                args.inputs,
                args.support_template,
                args.output_dir,
            )
        )
    ):
        raise ValueError(
            "postcreate materialization arguments are incomplete"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.profile != EXPECTED_PROFILE or args.region != EXPECTED_REGION:
        raise ValueError("profile/region guard rejected postcreate materialization")
    if _STACK_ID.fullmatch(args.support_stack_id) is None:
        raise ValueError("support stack ID is not the exact production ARN")
    if os.environ.get("AWS_PAGER", ""):
        raise ValueError("AWS_PAGER must be empty for postcreate materialization")
    runner = runner_factory(profile=args.profile, region=args.region)
    _guard_caller(
        profile=args.profile,
        region=args.region,
        support_stack_id=args.support_stack_id,
        runner=runner,
    )
    inputs = support_build_inputs_from_mapping(_read_canonical(args.inputs))
    template = _read_canonical(args.support_template)
    cloudformation = getattr(runner, "cloudformation", None)
    ec2 = getattr(runner, "ec2", None)
    if cloudformation is None or ec2 is None:
        raise TypeError("postcreate runner lacks exact read clients")
    bundle = coordinate_support_postcreate(
        inputs=inputs,
        template=template,
        support_stack_id=args.support_stack_id,
        services=SupportMaterializationServices(
            cloudformation=cloudformation,
            ec2=ec2,
        ),
    )
    _write_new_artifacts(
        args.output_dir,
        {
            "support-deletion-authority-v1.json": (
                bundle.inputs.support_deletion_inventory
            ),
            "support-materialized-inputs-v1.json": (
                support_inputs_projection(bundle.inputs)
            ),
            "support-postcreate-manifest-v1.json": bundle.manifest,
            "support-retained-augmentation-v1.json": (
                bundle.postcreate_retained_fragment
            ),
            "TASK9_DEPLOYED_IDENTITY.json": (
                bundle.task9_deployed_identity
            ),
        },
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
