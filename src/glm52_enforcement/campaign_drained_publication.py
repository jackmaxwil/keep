"""Retained publication of one authenticated generation campaign-drained marker.

The worker owns only the legacy root objects.  This module runs after the
retained TerminalV2 writer has proved runtime terminality and closed spend.  It
re-reads every immutable S3 input by exact VersionId, authenticates their
ancestry, and conditionally creates the generation terminal marker.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from mlx_vq.quality.glm52_sky_terminal_state import (
    validate_sky_terminal_state,
)

from .canonical import canonical_json_bytes, canonical_sha256
from .records import validate_record
from .s3_keys import campaign_drained_s3_key
from .task10_worker import (
    ACCOUNT_ID,
    GRACEFUL_SCRIPT_NAMES,
    REGION,
    RUN_ID,
    UNIT_NAMES,
    render_worker_units,
    validate_graceful_stop_evidence,
    worker_bootstrap_descriptor_from_mapping,
    worker_unit_hashes,
)

RETAINED_MODELS_BUCKET = "keep-glm52-models-246813579024-us-west-2"
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_VERSION_ID = re.compile(r"(?!null\Z)(?!None\Z)[\x21-\x7e]{1,1024}\Z")
_GENERATION_TEXT = re.compile(r"[0-9]{8}\Z")
_ACTIVATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_TASK9_COPIED_FIELDS = (
    "handoff",
    "binding",
    "final_sky_state",
    "final_ec2_states",
    "request_cardinality",
    "allocations",
    "worker_launch_evidence",
    "worker_launch_liabilities",
    "final_heartbeat_identity",
    "checkpoint_identity",
    "cache_identity",
    "training_identity",
    "evaluation_identity",
    "drain_identity",
    "request_evidence",
    "post_terminal_quiescence_evidence",
    "prior_terminal_v1_identity",
    "outcome",
    "operator_disposition_required",
)
_TASK9_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "campaign_identity_sha256",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "source_family_identity_sha256",
        *_TASK9_COPIED_FIELDS,
        "evidence_created_at",
        "canonical_body_sha256",
    }
)
_DESCRIPTOR_COORDINATE_FIELDS = frozenset(
    {"bucket", "key", "version_id", "file_sha256", "body_sha256"}
)


class CampaignDrainedPublicationError(ValueError):
    """Generation campaign-drained truth is incomplete or ambiguous."""


@dataclass(frozen=True)
class CampaignDrainedPublicationServices:
    """S3 boundary whose SDK retry policy is fixed at one attempt."""

    s3: object
    total_max_attempts: int


@dataclass(frozen=True)
class _ExactObject:
    key: str
    version_id: str
    raw: bytes


def _fail(message: str) -> None:
    raise CampaignDrainedPublicationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _checksum(raw: bytes) -> str:
    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def _version_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or not value.isascii()
        or _VERSION_ID.fullmatch(value) is None
    ):
        _fail(label + " must be one opaque non-null VersionId")
    return value


def _response(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        _fail(operation + " returned no exact response object")
    metadata = response.get("ResponseMetadata")
    retries = (
        metadata.get("RetryAttempts") if type(metadata) is dict else None
    )
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(retries) is not int
        or isinstance(retries, bool)
        or retries != 0
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        _fail(operation + " lacks authenticated zero-retry success")
    return response


def _call(
    client: object,
    method_name: str,
    *,
    operation: str,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    method = getattr(client, method_name, None)
    if not callable(method):
        _fail(operation + " typed client method is absent")
    return _response(method(**dict(request)), operation)


def _inventory_versions(
    *,
    s3: object,
    bucket: str,
    key: str,
    label: str,
) -> tuple[dict[str, object], ...]:
    key_marker: str | None = None
    version_marker: str | None = None
    seen_markers: set[tuple[str, str]] = set()
    exact: list[dict[str, object]] = []
    for _page in range(1024):
        request: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": key,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = _call(
            s3,
            "list_object_versions",
            operation="ListObjectVersions " + label,
            request=request,
        )
        versions = response.get("Versions", [])
        delete_markers = response.get("DeleteMarkers", [])
        truncated = response.get("IsTruncated")
        if (
            type(versions) is not list
            or type(delete_markers) is not list
            or type(truncated) is not bool
            or any(type(item) is not dict for item in versions)
            or any(type(item) is not dict for item in delete_markers)
        ):
            _fail(label + " S3 version history is malformed")
        if any(item.get("Key") == key for item in delete_markers):
            _fail(label + " S3 version history contains a delete marker")
        exact.extend(
            dict(item) for item in versions if item.get("Key") == key
        )
        if not truncated:
            return tuple(exact)
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen_markers
        ):
            _fail(label + " S3 version pagination is incomplete")
        seen_markers.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    _fail(label + " S3 version pagination exceeded its bound")


def _read_body(response: Mapping[str, object], label: str) -> bytes:
    body = response.get("Body")
    read = getattr(body, "read", None)
    try:
        raw = read() if callable(read) else None
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > _MAX_SOURCE_BYTES
    ):
        _fail(label + " exact-version body length is invalid")
    return raw


def _read_exact_version(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: str,
    label: str,
) -> bytes:
    response = _call(
        s3,
        "get_object",
        operation="GetObject " + label,
        request={
            "Bucket": bucket,
            "Key": key,
            "VersionId": version_id,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        },
    )
    raw = _read_body(response, label)
    content_length = response.get("ContentLength")
    checksum = response.get("ChecksumSHA256")
    etag = response.get("ETag")
    if (
        response.get("VersionId") != version_id
        or (
            content_length is not None
            and content_length != len(raw)
        )
        or (
            checksum is not None
            and checksum != _checksum(raw)
        )
        or (
            etag is not None
            and etag
            != '"'
            + hashlib.md5(raw, usedforsecurity=False).hexdigest()
            + '"'
        )
    ):
        _fail(label + " exact-version readback drifted")
    return raw


def _read_singular_source(
    *,
    s3: object,
    bucket: str,
    key: str,
    label: str,
    expected_version_id: str | None = None,
    expected_file_sha256: str | None = None,
) -> _ExactObject:
    versions = _inventory_versions(
        s3=s3, bucket=bucket, key=key, label=label
    )
    if len(versions) != 1:
        _fail(label + " S3 version history is not singular")
    row = versions[0]
    if row.get("IsLatest") is not True:
        _fail(label + " S3 singular version is not latest")
    version_id = _version_id(row.get("VersionId"), label + " VersionId")
    if expected_version_id is not None and version_id != expected_version_id:
        _fail(label + " pinned VersionId is not the singular live version")
    raw = _read_exact_version(
        s3=s3,
        bucket=bucket,
        key=key,
        version_id=version_id,
        label=label,
    )
    if row.get("Size") not in {None, len(raw)}:
        _fail(label + " inventory size drifted")
    if (
        expected_file_sha256 is not None
        and _sha256(raw) != expected_file_sha256
    ):
        _fail(label + " pinned file SHA-256 drifted")
    return _ExactObject(key=key, version_id=version_id, raw=raw)


def _canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignDrainedPublicationError(
            label + " is not UTF-8 JSON"
        ) from exc
    if (
        type(value) is not dict
        or not value
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        _fail(label + " is not one canonical JSON object plus LF")
    return value


def _descriptor_coordinate(
    value: object,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _DESCRIPTOR_COORDINATE_FIELDS
        or value.get("bucket") != RETAINED_MODELS_BUCKET
        or value.get("key") != "task13/production/task10-worker-descriptor.json"
        or type(value.get("file_sha256")) is not str
        or _SHA256.fullmatch(value["file_sha256"]) is None
        or type(value.get("body_sha256")) is not str
        or _SHA256.fullmatch(value["body_sha256"]) is None
    ):
        _fail("Task10 worker descriptor coordinate is invalid")
    _version_id(value.get("version_id"), "Task10 descriptor VersionId")
    return dict(value)


def _validate_scope(
    *,
    activation_id: object,
    activation_ordinal: object,
    generation: object,
    generation_text: object,
) -> tuple[str, int, int, str]:
    if (
        type(activation_id) is not str
        or _ACTIVATION_ID.fullmatch(activation_id) is None
        or type(activation_ordinal) is not int
        or isinstance(activation_ordinal, bool)
        or activation_ordinal <= 0
        or type(generation) is not int
        or isinstance(generation, bool)
        or generation <= 0
        or type(generation_text) is not str
        or _GENERATION_TEXT.fullmatch(generation_text) is None
        or int(generation_text) != generation
    ):
        _fail("authenticated invocation scope is invalid")
    return activation_id, activation_ordinal, generation, generation_text


def _validate_terminal_ancestry(
    *,
    terminal_v2: object,
    task9_evidence: object,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    generation_text: str,
    campaign_identity_sha256: str,
) -> dict[str, object]:
    try:
        terminal = validate_record(
            "glm52_production_terminal_v2", terminal_v2
        )
    except (TypeError, ValueError) as exc:
        raise CampaignDrainedPublicationError(
            "TerminalV2 authority is invalid"
        ) from exc
    if (
        terminal["account_id"] != ACCOUNT_ID
        or terminal["region"] != REGION
        or terminal["run_id"] != RUN_ID
        or terminal["campaign_identity_sha256"]
        != campaign_identity_sha256
        or terminal["activation_id"] != activation_id
        or terminal["activation_ordinal"] != activation_ordinal
        or terminal["generation"] != generation
        or terminal["generation_text"] != generation_text
        or terminal["worker_cardinality"] != "ONE"
        or len(terminal["allocations"]) != 1
        or terminal["outcome"]
        not in {
            "DRAINED_COMPLETED",
            "DRAINED_TRAINING_DEFERRED",
            "DRAINED_RESUMABLE_DEADLINE",
        }
        or terminal["operator_disposition_required"] is not False
    ):
        _fail("TerminalV2 is not one authenticated drained worker")
    if type(task9_evidence) is not dict or set(task9_evidence) != _TASK9_FIELDS:
        _fail("Task9 terminal evidence ancestry schema is incomplete")
    evidence_body = dict(task9_evidence)
    evidence_identity = evidence_body.pop("canonical_body_sha256", None)
    if (
        task9_evidence["schema_version"] != 1
        or task9_evidence["record_type"]
        != "glm52_task9_terminal_evidence_v1"
        or task9_evidence["account_id"] != ACCOUNT_ID
        or task9_evidence["region"] != REGION
        or task9_evidence["run_id"] != RUN_ID
        or task9_evidence["campaign_identity_sha256"]
        != campaign_identity_sha256
        or task9_evidence["activation_id"] != activation_id
        or task9_evidence["activation_ordinal"] != activation_ordinal
        or task9_evidence["generation"] != generation
        or task9_evidence["generation_text"] != generation_text
        or evidence_identity != canonical_sha256(evidence_body)
        or any(
            task9_evidence[field] != terminal[field]
            for field in _TASK9_COPIED_FIELDS
        )
    ):
        _fail("Task9 terminal evidence ancestry drifted")
    return terminal


def _destination_metadata(
    *,
    descriptor: object,
    terminal: Mapping[str, object],
    drained: _ExactObject,
    campaign_ledger: _ExactObject,
    spend_ledger: _ExactObject,
    terminal_evidence: _ExactObject,
    graceful_stop: _ExactObject,
    terminal_state: Mapping[str, object],
) -> dict[str, str]:
    return {
        "glm52-account-id": ACCOUNT_ID,
        "glm52-activation-id": str(terminal["activation_id"]),
        "glm52-body-sha256": _sha256(drained.raw[:-1]),
        "glm52-campaign-ledger-sha256": str(
            terminal_state["campaign_ledger_sha256"]
        ),
        "glm52-campaign-ledger-version-sha256": _sha256(
            campaign_ledger.version_id.encode("ascii")
        ),
        "glm52-file-sha256": _sha256(drained.raw),
        "glm52-generation-text": str(terminal["generation_text"]),
        "glm52-graceful-stop-version-sha256": _sha256(
            graceful_stop.version_id.encode("ascii")
        ),
        "glm52-record-type": "glm52_sky_campaign_drained_v2",
        "glm52-region": REGION,
        "glm52-root-marker-version-sha256": _sha256(
            drained.version_id.encode("ascii")
        ),
        "glm52-run-id": RUN_ID,
        "glm52-spend-ledger-sha256": str(
            terminal_state["gpu_spend_ledger_sha256"]
        ),
        "glm52-spend-ledger-version-sha256": _sha256(
            spend_ledger.version_id.encode("ascii")
        ),
        "glm52-task9-evidence-version-sha256": _sha256(
            terminal_evidence.version_id.encode("ascii")
        ),
        "glm52-terminal-v2-body-sha256": str(
            terminal["canonical_body_sha256"]
        ),
        "glm52-worker-descriptor-body-sha256": str(
            descriptor.descriptor_body_sha256
        ),
    }


def _read_destination_exact(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> None:
    response = _call(
        s3,
        "get_object",
        operation="GetObject generation campaign-drained",
        request={
            "Bucket": bucket,
            "Key": key,
            "VersionId": version_id,
            "ExpectedBucketOwner": ACCOUNT_ID,
            "ChecksumMode": "ENABLED",
        },
    )
    observed = _read_body(response, "generation campaign-drained")
    if (
        observed != raw
        or response.get("ContentLength") != len(raw)
        or response.get("ContentType") != "application/json"
        or response.get("ChecksumType") != "FULL_OBJECT"
        or response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != _checksum(raw)
        or response.get("Metadata") != dict(metadata)
    ):
        _fail("generation campaign-drained exact-version readback drifted")


def _adopt_destination(
    *,
    s3: object,
    bucket: str,
    key: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> str | None:
    versions = _inventory_versions(
        s3=s3,
        bucket=bucket,
        key=key,
        label="generation campaign-drained",
    )
    if not versions:
        return None
    if len(versions) != 1:
        _fail("generation campaign-drained history is not singular")
    row = versions[0]
    if row.get("IsLatest") is not True or row.get("Size") != len(raw):
        _fail("generation campaign-drained durable version is foreign")
    version_id = _version_id(
        row.get("VersionId"), "generation campaign-drained VersionId"
    )
    _read_destination_exact(
        s3=s3,
        bucket=bucket,
        key=key,
        version_id=version_id,
        raw=raw,
        metadata=metadata,
    )
    return version_id


def _publish_once(
    *,
    s3: object,
    bucket: str,
    key: str,
    raw: bytes,
    metadata: Mapping[str, str],
) -> str:
    adopted = _adopt_destination(
        s3=s3,
        bucket=bucket,
        key=key,
        raw=raw,
        metadata=metadata,
    )
    if adopted is not None:
        return adopted
    method = getattr(s3, "put_object", None)
    if not callable(method):
        _fail("PutObject typed client method is absent")
    request = {
        "Bucket": bucket,
        "Key": key,
        "Body": raw,
        "ContentType": "application/json",
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": _checksum(raw),
        "IfNoneMatch": "*",
        "ExpectedBucketOwner": ACCOUNT_ID,
        "Metadata": dict(metadata),
    }
    try:
        raw_response = method(**request)
    except Exception as original_error:
        try:
            durable = _adopt_destination(
                s3=s3,
                bucket=bucket,
                key=key,
                raw=raw,
                metadata=metadata,
            )
        except Exception as adoption_error:  # noqa: BLE001
            raise CampaignDrainedPublicationError(
                "ambiguous generation PutObject has no exact adoption: "
                + str(adoption_error)
            ) from original_error
        if durable is None:
            raise CampaignDrainedPublicationError(
                "ambiguous generation PutObject left zero durable versions"
            ) from original_error
        return durable
    response = _response(raw_response, "PutObject generation campaign-drained")
    version_id = _version_id(
        response.get("VersionId"), "generation PutObject VersionId"
    )
    if response.get("ChecksumSHA256") != _checksum(raw):
        _fail("generation PutObject checksum response drifted")
    durable = _adopt_destination(
        s3=s3,
        bucket=bucket,
        key=key,
        raw=raw,
        metadata=metadata,
    )
    if durable != version_id:
        _fail("generation PutObject is not the singular durable version")
    return version_id


def publish_campaign_drained(
    *,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    generation_text: str,
    terminal_v2: Mapping[str, object],
    task10_worker_descriptor_coordinate: Mapping[str, object],
    services: CampaignDrainedPublicationServices,
) -> dict[str, object]:
    """Publish or exactly adopt one post-TerminalV2 generation marker."""

    if (
        type(services) is not CampaignDrainedPublicationServices
        or services.total_max_attempts != 1
    ):
        _fail("campaign-drained transport is not configured for one attempt")
    (
        activation_id,
        activation_ordinal,
        generation,
        generation_text,
    ) = _validate_scope(
        activation_id=activation_id,
        activation_ordinal=activation_ordinal,
        generation=generation,
        generation_text=generation_text,
    )
    coordinate = _descriptor_coordinate(
        task10_worker_descriptor_coordinate
    )
    bucket = str(coordinate["bucket"])
    descriptor_source = _read_singular_source(
        s3=services.s3,
        bucket=bucket,
        key=str(coordinate["key"]),
        label="Task10 worker descriptor",
        expected_version_id=str(coordinate["version_id"]),
        expected_file_sha256=str(coordinate["file_sha256"]),
    )
    descriptor_value = _canonical_object(
        descriptor_source.raw, "Task10 worker descriptor"
    )
    try:
        descriptor = worker_bootstrap_descriptor_from_mapping(
            descriptor_value
        )
    except ValueError as exc:
        raise CampaignDrainedPublicationError(
            "Task10 worker descriptor authentication failed"
        ) from exc
    if (
        descriptor.account_id != ACCOUNT_ID
        or descriptor.region != REGION
        or descriptor.run_id != RUN_ID
        or descriptor.activation_id != activation_id
        or descriptor.activation_ordinal != activation_ordinal
        or descriptor.generation != generation
        or descriptor.generation_text != generation_text
        or descriptor.descriptor_body_sha256
        != coordinate["body_sha256"]
    ):
        _fail("Task10 worker descriptor invocation ancestry drifted")

    drained = _read_singular_source(
        s3=services.s3,
        bucket=bucket,
        key=f"campaigns/{RUN_ID}/CAMPAIGN_DRAINED.json",
        label="root campaign-drained marker",
    )
    campaign_ledger = _read_singular_source(
        s3=services.s3,
        bucket=bucket,
        key=f"campaigns/{RUN_ID}/ledger/campaign-ledger.jsonl",
        label="campaign ledger",
    )
    spend_ledger = _read_singular_source(
        s3=services.s3,
        bucket=bucket,
        key=f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl",
        label="GPU spend ledger",
    )
    evidence_source = _read_singular_source(
        s3=services.s3,
        bucket=bucket,
        key=(
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{generation_text}/terminal-evidence/"
            "TASK9_TERMINAL_EVIDENCE.json"
        ),
        label="Task9 terminal evidence",
    )
    evidence = _canonical_object(
        evidence_source.raw, "Task9 terminal evidence"
    )
    terminal = _validate_terminal_ancestry(
        terminal_v2=terminal_v2,
        task9_evidence=evidence,
        activation_id=activation_id,
        activation_ordinal=activation_ordinal,
        generation=generation,
        generation_text=generation_text,
        campaign_identity_sha256=descriptor.campaign_identity_sha256,
    )
    allocation = terminal["allocations"][0]
    allocation_ordinal = allocation.get("allocation_ordinal")
    if type(allocation_ordinal) is not int or allocation_ordinal <= 0:
        _fail("TerminalV2 allocation ancestry is invalid")
    graceful_source = _read_singular_source(
        s3=services.s3,
        bucket=bucket,
        key=(
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{generation_text}/allocations/{allocation_ordinal:08d}/"
            "WORKER_GRACEFUL_STOP.json"
        ),
        label="Task10 graceful-stop observation",
    )
    graceful = _canonical_object(
        graceful_source.raw, "Task10 graceful-stop observation"
    )
    try:
        validate_graceful_stop_evidence(
            graceful,
            expected_unit_hashes=worker_unit_hashes(
                dict(render_worker_units())
            ),
            expected_script_hashes={
                name: graceful.get("script_file_sha256", {}).get(name)
                for name in GRACEFUL_SCRIPT_NAMES
            },
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CampaignDrainedPublicationError(
            "Task10 graceful-stop observation ancestry failed"
        ) from exc
    if (
        set(graceful["unit_file_sha256"]) != set(UNIT_NAMES)
        or set(graceful["script_file_sha256"])
        != set(GRACEFUL_SCRIPT_NAMES)
        or graceful["authority"] != "SSM"
        or graceful["campaign_terminal_marker_identity_sha256"]
        != _sha256(drained.raw)
    ):
        _fail("Task10 graceful-stop marker ancestry drifted")

    try:
        terminal_state = validate_sky_terminal_state(
            run_id=RUN_ID,
            execution_deadline=descriptor.execution_deadline,
            gpu_spend_authority_sha256=(
                descriptor.task8_spend_authority_identity_sha256
            ),
            drained_raw=drained.raw,
            campaign_ledger_raw=campaign_ledger.raw,
            spend_ledger_raw=spend_ledger.raw,
        )
    except ValueError as exc:
        raise CampaignDrainedPublicationError(
            "Sky campaign-drained terminal authentication failed: "
            + str(exc)
        ) from exc
    expected_outcome = {
        "DRAINED_COMPLETED": "completed",
        "DRAINED_TRAINING_DEFERRED": "training_deferred",
        "DRAINED_RESUMABLE_DEADLINE": "resumable_deadline",
    }[str(terminal["outcome"])]
    if terminal_state["outcome"] != expected_outcome:
        _fail("TerminalV2 and root campaign-drained outcomes drifted")

    key = campaign_drained_s3_key(
        run_id=RUN_ID, generation=generation
    )
    metadata = _destination_metadata(
        descriptor=descriptor,
        terminal=terminal,
        drained=drained,
        campaign_ledger=campaign_ledger,
        spend_ledger=spend_ledger,
        terminal_evidence=evidence_source,
        graceful_stop=graceful_source,
        terminal_state=terminal_state,
    )
    version_id = _publish_once(
        s3=services.s3,
        bucket=bucket,
        key=key,
        raw=drained.raw,
        metadata=metadata,
    )
    return {
        "schema_version": 1,
        "record_type": "glm52_campaign_drained_publication_v1",
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "generation": generation,
        "generation_text": generation_text,
        "file_sha256": _sha256(drained.raw),
        "body_sha256": _sha256(drained.raw[:-1]),
        "terminal_v2_body_sha256": terminal[
            "canonical_body_sha256"
        ],
    }


__all__ = [
    "RETAINED_MODELS_BUCKET",
    "CampaignDrainedPublicationError",
    "CampaignDrainedPublicationServices",
    "publish_campaign_drained",
]
