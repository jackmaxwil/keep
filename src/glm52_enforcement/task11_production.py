"""Closed production construction for the Task 11 Decision Lambda.

The core closure stays dependency-injected.  This module is the only place
where the Lambda runtime may construct AWS SDK clients.  Construction is
strictly read-before-effect: every immutable coordinate is validated before
``boto3`` is imported or a client is created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import format_datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import base64
import hashlib
import json
import os
import re
import secrets
import time
from types import SimpleNamespace
from typing import Callable, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from . import records as record_contract
from .canonical import canonical_json_bytes, canonical_sha256
from .decision_closure import (
    AUTHORITY_AUDIT_KINDS,
    CLOSURE_STEPS,
    HANDOFF_AUDIT_KIND,
    AdmissionClosureResult,
    DirectDecisionResponse,
    SourcePublication,
    build_admission_closure_result,
    build_audit_evidence,
    build_direct_decision_response,
    build_launch_or_expire_decision,
    build_step_receipt,
    build_transaction_readback,
)
from .dynamodb import (
    DynamoLedgerAdapter,
    ExactCheck,
    ExactPut,
    ExactUpdate,
    LedgerKey,
    TransactionResolution,
    WriteOutcome,
    encode_attribute_value,
    decode_item,
    encode_item,
)
from .h1f_adapter import (
    FreshH1fAuditService,
    H1fAuditRequest,
    H1fAuditResult,
    H1fAuthoritySnapshot,
    audit_current_head,
)
from .live_authority import (
    CallerIdentityObservation,
    CurrentClosureSessionBinding,
    ExpectedStateAuthentication,
    H1dExpectedState,
    H1dLiveAuthorityRequest,
    H1dLiveAuthorityResult,
    H1dLiveServices,
    LiveReadPage,
    LiveReadSpec,
    RuntimeRevalidationSourceCoordinate,
    RuntimeRevalidationSemanticAuthority,
    bind_current_closure_session_to_cutoff,
    build_sky_relay_probe_result,
    h1d_expected_state_from_mapping,
    inspect_h1d_live_authority,
    runtime_revalidation_sources_from_mapping,
    validate_runtime_revalidation_source_documents,
)
from .spend_authority import (
    ReserveListPage,
    SpendAuthorityServices,
    SpendListPage,
    SpendObject,
    inspect_spend_authority,
    spend_authority_request_from_mapping,
)
from .sky_admission import (
    AdmissionResult,
    AttestationResult,
    SkyAdmissionService,
    SkyAttestationService,
)
from .s3_adapter import S3PublicationServices
from .s3_records import build_immutable_json_candidate
from .task11_boundary import (
    Task11BoundaryCoordinate,
    Task11BoundaryDocument,
    load_task11_boundary,
)
from .task11_support_boundary import source_publication_from_payload
from .task11_support_boundary import (
    exact_input_coordinate_from_mapping,
    load_exact_input,
)
from .support_effect_writer_handler import (
    _validate_input as _validate_effect_writer_input,
    effect_write_result_from_payload,
    materialize_effect_builder_arguments,
)
from .task11_effect_writers import (
    EffectWriteResult,
    EffectWriterRequest,
    prepare_effect_candidate,
)
from .task12_correlation import build_sky_post_handoff_record
from .s3_adapter import S3CreateActionRequest
from .fence_executor import parse_fence_execution_result
from .task10_production import production_authority_from_mapping


_SHA = re.compile(r"^[0-9a-f]{64}$")
_VERSION_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-(?:"
    r"attestation|launch-admission|numeric-binding|"
    r"source-gpu-spend|source-submission-intent|"
    r"source-controller-baseline|source-control-plane-readiness|"
    r"source-submission-acquisition|fence-executor|pre-support-fence-executor|"
    r"fence-successor|claim-writer|decision-writer|terminal-v1-writer|"
    r"closure-handoff"
    r"):"
    r"[1-9][0-9]*$"
)
_S3_VERSION_ID = re.compile(r"[A-Za-z0-9._~+/=-]{1,1024}\Z")
_ROLE_ARN = re.compile(
    r"^arn:aws:iam::246813579024:role/"
    r"keep-glm52-[A-Za-z0-9+=,.@_/-]+$"
)
_TASK10_PRODUCTION_AUTHORITY_KEY = (
    "task13/production/task10-production-authority.json"
)


def _load_task10_production_authority(
    *,
    s3: object,
    bucket: object,
) -> object:
    """Load the sole immutable Task 10 production authority version."""

    if type(bucket) is not str or not bucket:
        raise RuntimeError("Task 11 Task 10 authority bucket drifted")
    list_versions = getattr(s3, "list_object_versions", None)
    get_object = getattr(s3, "get_object", None)
    if not callable(list_versions) or not callable(get_object):
        raise TypeError("Task 11 Task 10 authority boundary is absent")
    markers: dict[str, object] = {}
    seen_markers: set[tuple[str, str]] = set()
    versions: list[Mapping[str, object]] = []
    deletes: list[Mapping[str, object]] = []
    for _ in range(100):
        response = list_versions(
            Bucket=bucket,
            Prefix=_TASK10_PRODUCTION_AUTHORITY_KEY,
            MaxKeys=1000,
            ExpectedBucketOwner="246813579024",
            **markers,
        )
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        page_versions = (
            response.get("Versions")
            if type(response) is dict
            else None
        )
        page_deletes = (
            response.get("DeleteMarkers")
            if type(response) is dict
            else None
        )
        truncated = (
            response.get("IsTruncated")
            if type(response) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
            or metadata.get("RetryAttempts") != 0
            or type(page_versions) is not list
            or type(page_deletes) is not list
            or type(truncated) is not bool
        ):
            raise RuntimeError(
                "Task 11 Task 10 authority inventory is unauthenticated"
            )
        if any(type(item) is not dict for item in page_versions + page_deletes):
            raise RuntimeError(
                "Task 11 Task 10 authority inventory drifted"
            )
        versions.extend(page_versions)
        deletes.extend(page_deletes)
        if truncated is False:
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        marker = (next_key, next_version)
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or marker in seen_markers
        ):
            raise RuntimeError(
                "Task 11 Task 10 authority pagination drifted"
            )
        seen_markers.add(marker)
        markers = {
            "KeyMarker": next_key,
            "VersionIdMarker": next_version,
        }
    else:
        raise RuntimeError(
            "Task 11 Task 10 authority pagination exceeded bound"
        )
    if (
        len(versions) != 1
        or deletes
        or versions[0].get("Key") != _TASK10_PRODUCTION_AUTHORITY_KEY
        or versions[0].get("IsLatest") is not True
        or type(versions[0].get("VersionId")) is not str
        or versions[0]["VersionId"] == "null"
        or _S3_VERSION_ID.fullmatch(versions[0]["VersionId"]) is None
        or type(versions[0].get("Size")) is not int
        or versions[0]["Size"] <= 0
    ):
        raise RuntimeError("Task 11 Task 10 authority inventory drifted")
    version_id = versions[0]["VersionId"]
    response = get_object(
        Bucket=bucket,
        Key=_TASK10_PRODUCTION_AUTHORITY_KEY,
        VersionId=version_id,
        ExpectedBucketOwner="246813579024",
        ChecksumMode="ENABLED",
    )
    metadata = (
        response.get("ResponseMetadata")
        if type(response) is dict
        else None
    )
    stream = response.get("Body") if type(response) is dict else None
    read = getattr(stream, "read", None)
    close = getattr(stream, "close", None)
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
        or response.get("VersionId") != version_id
        or response.get("ContentLength") != versions[0]["Size"]
        or response.get("ContentType") != "application/json"
        or response.get("ChecksumType") != "FULL_OBJECT"
        or not callable(read)
        or not callable(close)
    ):
        raise RuntimeError(
            "Task 11 Task 10 authority exact read is unauthenticated"
        )
    try:
        raw = read(2 * 1024 * 1024 + 1)
    finally:
        close()
    checksum = (
        base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        if type(raw) is bytes
        else None
    )
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or len(raw) != versions[0]["Size"]
        or len(raw) > 2 * 1024 * 1024
        or response.get("ChecksumSHA256") != checksum
    ):
        raise RuntimeError("Task 11 Task 10 authority bytes drifted")
    try:
        document = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Task 11 Task 10 authority JSON drifted"
        ) from exc
    if (
        type(document) is not dict
        or canonical_json_bytes(document) + b"\n" != raw
        or response.get("Metadata")
        != {
            "record-type": "glm52_task10_production_authority_v1",
            "canonical-identity-sha256": document.get(
                "canonical_identity_sha256"
            ),
        }
    ):
        raise RuntimeError("Task 11 Task 10 authority metadata drifted")
    try:
        return production_authority_from_mapping(document)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Task 11 Task 10 authority drifted") from exc


def _build_normal_task11_handoff_bindings(
    *,
    run_id: object,
    activation_id: object,
    generation: object,
    action_key: object,
    action: object,
    decision_result: object,
    production_authority: object,
    api_server_identity_sha256: object,
    admission_classification: object,
    admission_request_id: object,
) -> Mapping[str, object]:
    """Cross-close the normal handoff from authenticated production facts."""

    if (
        run_id != "glm52-sky-20260724"
        or type(activation_id) is not str
        or not activation_id
        or type(generation) is not int
        or generation <= 0
        or type(action_key) is not str
        or not action_key
        or type(action) is not dict
        or type(decision_result) is not EffectWriteResult
        or decision_result.writer_kind != "DecisionWriter"
        or decision_result.record_kind != "start-decision"
        or decision_result.direct_response_authenticated is not True
        or _SHA.fullmatch(str(api_server_identity_sha256)) is None
    ):
        raise RuntimeError("Task 11 handoff authority drifted")
    try:
        decision = json.loads(decision_result.raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Task 11 handoff decision bytes drifted") from exc
    identity = decision_result.object_identity
    authority_expected = {
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": run_id,
        "managed_mode": "production",
        "campaign_identity_sha256": action.get(
            "campaign_identity_sha256"
        ),
        "activation_id": activation_id,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "action_key": action_key,
        "action_state": "CONSUMED",
        "current_activation": True,
    }
    action_expected = {
        "record_type": "glm52_production_action",
        "run_id": run_id,
        "activation_id": activation_id,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "action_kind": "SKY_POST",
        "state": "POST_CLASSIFIED",
        "candidate_key": identity.key,
        "candidate_file_sha256": identity.file_sha256,
        "candidate_body_sha256": identity.body_sha256,
        "outcome_class": admission_classification,
        "sky_request_id": admission_request_id,
    }
    decision_expected = {
        "record_type": (
            "glm52_sky_production_generation_start_decision_v1"
        ),
        "run_id": run_id,
        "campaign_identity_sha256": action.get(
            "campaign_identity_sha256"
        ),
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "sky_job_name": getattr(
            production_authority,
            "sky_job_name",
            None,
        ),
        "decision": "launch-once",
        "start_decision_body_sha256": identity.body_sha256,
    }
    if (
        not decision_result.raw.endswith(b"\n")
        or type(decision) is not dict
        or canonical_json_bytes(decision) + b"\n"
        != decision_result.raw
        or any(action.get(name) != value for name, value in action_expected.items())
        or any(
            getattr(production_authority, name, None) != value
            for name, value in authority_expected.items()
        )
        or any(
            decision.get(name) != value
            for name, value in decision_expected.items()
        )
        or type(decision.get("submit_attempt_id")) is not str
    ):
        raise RuntimeError("Task 11 handoff authority drifted")
    values = {
        "run_id": run_id,
        "campaign_identity_sha256": action[
            "campaign_identity_sha256"
        ],
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "submit_attempt_id": decision["submit_attempt_id"],
        "decision_key": identity.key,
        "decision_version_id": identity.version_id,
        "decision_file_sha256": identity.file_sha256,
        "decision_body_sha256": identity.body_sha256,
        "sky_post_action_key": action_key,
        "sky_post_consumed_at": action.get("consumed_at"),
        "sky_post_outcome_class": action.get("outcome_class"),
        "expected_sky_job_name": getattr(
            production_authority,
            "sky_job_name",
            None,
        ),
        "task_yaml_sha256": getattr(
            production_authority,
            "task_yaml_sha256",
            None,
        ),
        "request_body_sha256": getattr(
            production_authority,
            "request_body_sha256",
            None,
        ),
        "api_server_identity_sha256": api_server_identity_sha256,
        "sky_request_id": action.get("sky_request_id"),
        "post_started_at": action.get("post_started_at"),
        "post_completed_or_lost_at": action.get("completed_at"),
        "binding_state": "reconcile-required",
    }
    try:
        validated = build_sky_post_handoff_record(**values)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Task 11 handoff authority drifted") from exc
    validated.pop("handoff_body_sha256")
    return validated


@dataclass(frozen=True)
class Task11ProductionConfig:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    ledger_table_name: str
    campaign_bucket: str
    model_bucket: str
    model_prefix: str
    fence_stack_id: str
    support_stack_id: str
    closure_role_arn: str
    attestation_version_arn: str
    launch_admission_version_arn: str
    numeric_binding_version_arn: str
    source_gpu_spend_version_arn: str
    source_submission_intent_version_arn: str
    source_controller_baseline_version_arn: str
    source_control_plane_readiness_version_arn: str
    source_submission_acquisition_version_arn: str
    fence_executor_version_arn: str
    fence_successor_version_arn: str
    claim_writer_version_arn: str
    decision_writer_version_arn: str
    terminal_v1_writer_version_arn: str
    closure_handoff_version_arn: str
    deployment_identity_sha256: str
    decision_function_version_arn: str


@dataclass(frozen=True)
class Task11AwsClients:
    s3: object
    dynamodb: object
    cloudformation: object
    lambda_client: object
    sts: object
    h1d_clients: Optional[Mapping[str, object]] = None
    credential_expiration: Optional[str] = None
    caller_identity: Optional[Mapping[str, object]] = None
    credential_issue_time: Optional[str] = None
    assume_role_request_id: Optional[str] = None
    source_caller_identity: Optional[Mapping[str, object]] = None
    assumed_role_arn: Optional[str] = None
    assumed_role_id: Optional[str] = None


@dataclass(frozen=True)
class Task11ActionConfig:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    ledger_table_name: str
    campaign_bucket: str
    deployment_identity_sha256: str
    function_version_arn: str


@dataclass(frozen=True)
class RuntimeRevalidationSourceRead:
    coordinate: RuntimeRevalidationSourceCoordinate
    document: Mapping[str, object]
    request_id: str
    response_identity_sha256: str


@dataclass(frozen=True)
class RuntimeAttachmentInventoryEvidence:
    request_ids: tuple[str, ...]
    response_identities: tuple[str, ...]
    family_identities: Mapping[str, str]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class RuntimeRevalidationBundle:
    document: Mapping[str, object]
    expected_state: H1dExpectedState
    expected_state_authentication: ExpectedStateAuthentication
    sources: tuple[RuntimeRevalidationSourceRead, ...]
    semantic_authority: RuntimeRevalidationSemanticAuthority
    current_session: CurrentClosureSessionBinding
    canonical_identity_sha256: str


_RUNTIME_SOURCE_GET_FIELDS = frozenset(
    {
        "Body",
        "VersionId",
        "ContentLength",
        "ChecksumSHA256",
        "ChecksumType",
        "ETag",
        "LastModified",
        "ContentType",
        "Metadata",
        "ServerSideEncryption",
        "SSEKMSKeyId",
        "BucketKeyEnabled",
        "ResponseMetadata",
    }
)
_RUNTIME_SOURCE_METADATA_FIELDS = frozenset(
    {
        "RequestId",
        "HostId",
        "HTTPStatusCode",
        "HTTPHeaders",
        "RetryAttempts",
    }
)


def load_runtime_revalidation_sources(
    *,
    s3: object,
    coordinates: tuple[RuntimeRevalidationSourceCoordinate, ...],
) -> tuple[RuntimeRevalidationSourceRead, ...]:
    """Exact-read every immutable Task 6/7 and runtime source."""

    if (
        type(coordinates) is not tuple
        or len(coordinates) != 7
        or any(
            type(item) is not RuntimeRevalidationSourceCoordinate
            for item in coordinates
        )
    ):
        raise RuntimeError(
            "Task 11 runtime source coordinates are incomplete"
        )
    get_object = getattr(s3, "get_object", None)
    if not callable(get_object):
        raise RuntimeError("Task 11 runtime source reader is absent")
    reads = []
    for coordinate in coordinates:
        response = get_object(
            Bucket=coordinate.bucket,
            Key=coordinate.key,
            VersionId=coordinate.version_id,
            ExpectedBucketOwner="246813579024",
            ChecksumMode="ENABLED",
        )
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        stream = response.get("Body") if type(response) is dict else None
        read = getattr(stream, "read", None)
        if (
            type(response) is not dict
            or not {
                "Body",
                "VersionId",
                "ContentLength",
                "ChecksumSHA256",
                "ETag",
                "ResponseMetadata",
            }.issubset(response)
            or not set(response).issubset(_RUNTIME_SOURCE_GET_FIELDS)
            or response["VersionId"] != coordinate.version_id
            or type(response["ContentLength"]) is not int
            or response["ContentLength"] < 0
            or type(response["ChecksumSHA256"]) is not str
            or not response["ChecksumSHA256"]
            or type(response["ETag"]) is not str
            or not response["ETag"]
            or type(metadata) is not dict
            or not {"HTTPStatusCode", "RequestId"}.issubset(metadata)
            or not set(metadata).issubset(
                _RUNTIME_SOURCE_METADATA_FIELDS
            )
            or type(metadata["HTTPStatusCode"]) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata["RequestId"]) is not str
            or not metadata["RequestId"]
            or not callable(read)
        ):
            raise RuntimeError(
                "Task 11 runtime source GetObject drifted"
            )
        raw = read(8 * 1024 * 1024 + 1)
        if (
            type(raw) is not bytes
            or len(raw) > 8 * 1024 * 1024
            or response["ContentLength"] != len(raw)
            or not raw.endswith(b"\n")
            or raw.endswith(b"\n\n")
            or hashlib.sha256(raw).hexdigest()
            != coordinate.file_sha256
            or response["ChecksumSHA256"]
            != base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii")
        ):
            raise RuntimeError(
                "Task 11 runtime source service evidence drifted"
            )
        try:
            document = json.loads(raw[:-1].decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Task 11 runtime source is invalid JSON"
            ) from exc
        if (
            type(document) is not dict
            or canonical_json_bytes(document) + b"\n" != raw
        ):
            raise RuntimeError(
                "Task 11 runtime source is not canonical"
            )
        identity_field = (
            "canonical_identity_sha256"
            if "canonical_identity_sha256" in document
            else (
                "canonical_body_sha256"
                if "canonical_body_sha256" in document
                else None
            )
        )
        body = dict(document)
        if identity_field is None:
            body_identity = canonical_sha256(body)
        else:
            body_identity = body.pop(identity_field)
            if body_identity != canonical_sha256(body):
                raise RuntimeError(
                    "Task 11 runtime source self identity drifted"
                )
        if body_identity != coordinate.body_sha256:
            raise RuntimeError(
                "Task 11 runtime source body identity drifted"
            )
        response_identity = canonical_sha256(
            {
                "operation": "GetObject",
                "input_kind": coordinate.input_kind,
                "bucket": coordinate.bucket,
                "key": coordinate.key,
                "version_id": coordinate.version_id,
                "content_length": response["ContentLength"],
                "checksum_sha256": response["ChecksumSHA256"],
                "etag": response["ETag"],
                "request_id": metadata["RequestId"],
                "file_sha256": coordinate.file_sha256,
                "body_sha256": coordinate.body_sha256,
            }
        )
        reads.append(
            RuntimeRevalidationSourceRead(
                coordinate=coordinate,
                document=document,
                request_id=metadata["RequestId"],
                response_identity_sha256=response_identity,
            )
        )
    return tuple(reads)


def revalidate_runtime_attachment_specs(
    *,
    specs: tuple[LiveReadSpec, ...],
    clients: Mapping[str, object],
    deadline: datetime,
) -> RuntimeAttachmentInventoryEvidence:
    """Read and exactly match a finite, source-authenticated live inventory."""

    if (
        type(specs) is not tuple
        or not specs
        or any(type(spec) is not LiveReadSpec for spec in specs)
        or len(
            {(spec.family, spec.operation) for spec in specs}
        )
        != len(specs)
        or type(clients) is not dict
        or type(deadline) is not datetime
        or deadline.tzinfo is None
    ):
        raise RuntimeError(
            "Task 11 runtime attachment inventory is not closed"
        )
    reader = Task11H1dReader(clients)
    reader.prepare(specs, deadline=deadline)
    request_ids: list[str] = []
    response_identities: list[str] = []
    family_identities: dict[str, str] = {}
    repeated_families = {
        spec.family
        for spec in specs
        if sum(item.family == spec.family for item in specs) > 1
    }
    for spec in specs:
        page = reader.read_page(spec=spec, continuation_token=None)
        if (
            page.family != spec.family
            or page.operation != spec.operation
            or page.request_token is not None
            or page.page_index != 0
            or page.next_token is not None
            or type(page.service_request_ids) is not tuple
            or not page.service_request_ids
            or type(page.service_response_identities) is not tuple
            or len(page.service_request_ids)
            != len(page.service_response_identities)
        ):
            raise RuntimeError(
                "Task 11 runtime attachment evidence drifted"
            )
        expected = tuple(dict(item) for item in spec.expected_items)
        observed = tuple(dict(item) for item in page.items)
        for label, rows in (("expected", expected), ("observed", observed)):
            identities = []
            for row in rows:
                identity = row.get(spec.identity_field)
                if type(identity) is not str or not identity:
                    raise RuntimeError(
                        "Task 11 runtime attachment identity drifted"
                    )
                identities.append(identity)
            if len(set(identities)) != len(identities):
                raise RuntimeError(
                    f"Task 11 runtime attachment {label} identity repeated"
                )
        expected_sorted = tuple(
            sorted(expected, key=lambda row: row[spec.identity_field])
        )
        observed_sorted = tuple(
            sorted(observed, key=lambda row: row[spec.identity_field])
        )
        if observed_sorted != expected_sorted:
            raise RuntimeError(
                "Task 11 runtime attachment inventory mismatched"
            )
        request_ids.extend(page.service_request_ids)
        response_identities.extend(page.service_response_identities)
        evidence_key = (
            spec.operation
            if spec.family in repeated_families
            else spec.family
        )
        family_identities[evidence_key] = canonical_sha256(
            {
                "family": spec.family,
                **(
                    {"operation": spec.operation}
                    if spec.family in repeated_families
                    else {}
                ),
                "items": observed_sorted,
            }
        )
    if (
        len(set(request_ids)) != len(request_ids)
        or len(set(response_identities)) != len(response_identities)
    ):
        raise RuntimeError(
            "Task 11 runtime attachment service evidence repeated"
        )
    body = {
        "request_ids": tuple(request_ids),
        "response_identities": tuple(response_identities),
        "family_identities": family_identities,
    }
    return RuntimeAttachmentInventoryEvidence(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class Task11AcceptedAdapters:
    """Exact accepted Task 3-10 services used by the production composition."""

    ledger: DynamoLedgerAdapter
    fresh_h1f: FreshH1fAuditService
    clients: Task11AwsClients


class Task11ReadOnlyPhaseServices:
    """One shared entry surface for the six Task 11 read-only phase methods."""

    _METHODS = (
        "prove_preauthorized_batch_template",
        "acquire_cfn_quiescence",
        "revalidate_runtime_attachments_and_seals",
        "warm_clients_and_construct",
        "size_non_authoritative",
        "stable_tls_sky_identity_preflight",
    )
    _MUTATION_METHODS = (
        "assume_role",
        "batch_write_item",
        "create_change_set",
        "delete_object",
        "execute_change_set",
        "invoke",
        "put_item",
        "put_object",
        "start_instances",
        "stop_instances",
        "transact_write_items",
        "update_item",
    )

    def __init__(
        self,
        owner: object,
        *,
        reject_mutation_clients: bool,
    ) -> None:
        self._owner = owner
        if reject_mutation_clients:
            clients = getattr(
                getattr(owner, "_adapters", None),
                "clients",
                None,
            )
            candidates = [
                getattr(clients, name, None)
                for name in (
                    "s3",
                    "cloudformation",
                    "lambda_client",
                    "sts",
                )
            ]
            h1d_clients = getattr(clients, "h1d_clients", None)
            if type(h1d_clients) is dict:
                candidates.extend(h1d_clients.values())
            if (
                getattr(clients, "dynamodb", None) is not None
                or any(
                    callable(getattr(client, method_name, None))
                    for client in candidates
                    if client is not None
                    for method_name in self._MUTATION_METHODS
                )
            ):
                raise RuntimeError(
                    "Task 11 rehearsal received a mutation-capable client"
                )

    def prove_preauthorized_batch_template(
        self, **kwargs: object
    ) -> object:
        return self._owner._read_only_prove_preauthorized_batch_template(
            **kwargs
        )

    def acquire_cfn_quiescence(self, **kwargs: object) -> object:
        return self._owner._read_only_acquire_cfn_quiescence(**kwargs)

    def revalidate_runtime_attachments_and_seals(
        self, **kwargs: object
    ) -> object:
        return self._owner._read_only_revalidate_runtime_attachments_and_seals(
            **kwargs
        )

    def warm_clients_and_construct(self, **kwargs: object) -> object:
        return self._owner._read_only_warm_clients_and_construct(**kwargs)

    def size_non_authoritative(self, **kwargs: object) -> object:
        return self._owner._read_only_size_non_authoritative(**kwargs)

    def stable_tls_sky_identity_preflight(
        self, **kwargs: object
    ) -> object:
        return self._owner._read_only_stable_tls_sky_identity_preflight(
            **kwargs
        )


_SOURCE_METHOD_KINDS = {
    "publish_gpu_spend_snapshot": "GPU_SPEND",
    "publish_submission_intent": "SUBMISSION_INTENT",
    "publish_controller_baseline": "CONTROLLER_BASELINE",
    "publish_control_plane_readiness": "CONTROL_PLANE_READINESS",
    "publish_submission_acquisition": "SUBMISSION_ACQUISITION",
}


class Task11ProductionServices:
    """Concrete, closed service surface consumed by the Task 11 route.

    The object deliberately owns the named route methods.  Raw AWS clients
    are private implementation details and can never be mistaken for the
    closure service itself.
    """

    def __init__(
        self,
        *,
        config: Task11ProductionConfig,
        adapters: Task11AcceptedAdapters,
        boundary: Task11BoundaryDocument,
        event: Mapping[str, object],
        context: object,
        session_binding_validator: Optional[object] = None,
        read_only_rehearsal: bool = False,
    ) -> None:
        self._config = config
        self._adapters = adapters
        self._boundary = boundary
        self._event = dict(event)
        self._session_binding_validator = session_binding_validator
        remaining = getattr(context, "get_remaining_time_in_millis", None)
        if not callable(remaining):
            raise RuntimeError(
                "Task 11 Lambda context lacks remaining-time authority"
            )
        self._remaining = remaining
        self._sequence = 0
        self._attestation_sequence = 0
        self._direct_objects = {}
        self._source_wire_payloads = []
        self._effect_results = {}
        self._effect_wire_payloads = []
        self._prepared_fence_payload = None
        self._executed_fence_result = None
        self._owner_nonce_sha256: Optional[str] = None
        self._decision_nonce_sha256: Optional[str] = None
        action_config = Task11ActionConfig(
            account_id=config.account_id,
            region=config.region,
            run_id=config.run_id,
            activation_id=config.activation_id,
            ledger_table_name=config.ledger_table_name,
            campaign_bucket=config.campaign_bucket,
            deployment_identity_sha256=(
                config.deployment_identity_sha256
            ),
            function_version_arn=config.decision_function_version_arn,
        )
        self._action_service = (
            None
            if read_only_rehearsal
            else Task11DynamoActionService(
                config=action_config,
                ledger=adapters.ledger,
            )
        )
        self._transaction_records = {}
        self._action_key_cache = {}
        self._lambda_invocations = []
        self._read_only_phases = Task11ReadOnlyPhaseServices(
            self,
            reject_mutation_clients=read_only_rehearsal,
        )

    @property
    def read_only_phases(self) -> Task11ReadOnlyPhaseServices:
        return self._read_only_phases

    def _caller_identity(self) -> Mapping[str, object]:
        session_binding_validator = getattr(
            self,
            "_session_binding_validator",
            None,
        )
        if session_binding_validator is not None:
            validate = getattr(
                session_binding_validator,
                "validate_caller_identity",
                None,
            )
            if not callable(validate):
                raise RuntimeError(
                    "Task 11 read-only session validator is incomplete"
                )
            value = validate(
                config=self._config,
                clients=self._adapters.clients,
            )
            if type(value) is not dict:
                raise RuntimeError(
                    "Task 11 read-only caller identity is unauthenticated"
                )
            return value
        value = self._adapters.clients.caller_identity
        expected_role_name = _closure_role_name(
            self._config.activation_id
        )
        expected_prefix = (
            "arn:aws:sts::"
            + self._config.account_id
            + ":assumed-role/"
            + expected_role_name
            + "/h1g-decision-"
        )
        if (
            type(value) is not dict
            or value.get("Account") != self._config.account_id
            or type(value.get("Arn")) is not str
            or not value["Arn"].startswith(expected_prefix)
            or type(value.get("UserId")) is not str
            or not value["UserId"]
            or type(value.get("ResponseMetadata")) is not dict
            or value["ResponseMetadata"].get("HTTPStatusCode") != 200
            or type(
                value["ResponseMetadata"].get("RequestId")
            ) is not str
            or not value["ResponseMetadata"]["RequestId"]
        ):
            raise RuntimeError("Task 11 caller identity is unauthenticated")
        return value

    def _operation_identity(
        self,
        *,
        operation: str,
        request: object,
        transport_identity: str,
    ) -> str:
        self._sequence += 1
        return canonical_sha256(
            {
                "operation": operation,
                "request_identity_sha256": getattr(
                    request,
                    "canonical_identity_sha256",
                    None,
                ),
                "transport_identity": transport_identity,
                "sequence": self._sequence,
                "deployment_identity_sha256": (
                    self._config.deployment_identity_sha256
                ),
            }
        )

    def _authenticated_read_receipt(
        self,
        *,
        step_name: str,
        operation: str,
        evidence: Mapping[str, object],
        nonce_ownership_sha256: Optional[str] = None,
        coherent_readback: bool = False,
    ) -> object:
        self._sequence += 1
        operation_identity = canonical_sha256(
            {
                "operation": operation,
                "authenticated_evidence": dict(evidence),
                "sequence": self._sequence,
                "deployment_identity_sha256": (
                    self._config.deployment_identity_sha256
                ),
            }
        )
        return build_step_receipt(
            step_name=step_name,
            operation_identity_sha256=operation_identity,
            nonce_ownership_sha256=nonce_ownership_sha256,
            coherent_readback=coherent_readback,
        )

    def _describe_stack(self, stack_coordinate: str) -> str:
        value = self._adapters.clients.cloudformation.describe_stacks(
            StackName=stack_coordinate,
        )
        stack = (
            value["Stacks"][0]
            if type(value) is dict
            and type(value.get("Stacks")) is list
            and len(value["Stacks"]) == 1
            and type(value["Stacks"][0]) is dict
            else None
        )
        coordinate_is_arn = stack_coordinate.startswith(
            "arn:aws:cloudformation:"
        )
        if (
            stack is None
            or type(stack.get("StackId")) is not str
            or not stack["StackId"].startswith(
                "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            )
            or (
                coordinate_is_arn
                and stack["StackId"] != stack_coordinate
            )
            or (
                not coordinate_is_arn
                and stack.get("StackName") != stack_coordinate
            )
            or type(value.get("ResponseMetadata")) is not dict
            or value["ResponseMetadata"].get("HTTPStatusCode") != 200
            or type(
                value["ResponseMetadata"].get("RequestId")
            ) is not str
        ):
            raise RuntimeError("Task 11 stack readback is unauthenticated")
        return str(value["ResponseMetadata"]["RequestId"])

    def prove_preauthorized_batch_template(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        return self._read_only_phases.prove_preauthorized_batch_template(
            request=request,
            custody_nonce_sha256=custody_nonce_sha256,
        )

    def _read_only_prove_preauthorized_batch_template(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        coordinate = exact_input_coordinate_from_mapping(
            asdict(self._boundary.inputs[6]),
            expected_kind="FENCE_EXECUTION_REQUEST",
        )
        immutable = load_exact_input(
            s3=self._adapters.clients.s3,
            coordinate=coordinate,
        )
        from .fence_artifacts import (
            FenceSlot,
            parse_fence_transition_request,
        )

        transition_request = parse_fence_transition_request(immutable)
        if (
            transition_request.slot
            is not FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
        ):
            raise RuntimeError(
                "Task 11 preauthorized fence request is not BATCH v2"
            )
        request_value = transition_request.to_dict()
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[0],
            operation="prove_preauthorized_batch_template",
            evidence={
                "custody_nonce_sha256": custody_nonce_sha256,
                "fence_request_identity_sha256": (
                    transition_request.canonical_identity_sha256
                ),
                "manifest_coordinate": request_value[
                    "manifest_coordinate"
                ],
                "selected_entry_identity_sha256": request_value[
                    "selected_entry_identity_sha256"
                ],
                "request_skeleton_sha256": request_value[
                    "request_skeleton_sha256"
                ],
            },
        )

    def acquire_cfn_quiescence(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        return self._read_only_phases.acquire_cfn_quiescence(
            request=request,
            custody_nonce_sha256=custody_nonce_sha256,
        )

    def _read_only_acquire_cfn_quiescence(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        summaries = []
        token = None
        seen = set()
        for _page in range(10):
            arguments = {
                "StackName": self._config.fence_stack_id,
            }
            if token is not None:
                arguments["NextToken"] = token
            value = (
                self._adapters.clients.cloudformation.list_change_sets(
                    **arguments
                )
            )
            metadata = (
                value.get("ResponseMetadata")
                if type(value) is dict
                else None
            )
            page = (
                value.get("Summaries")
                if type(value) is dict
                else None
            )
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 200
                or type(metadata.get("RequestId")) is not str
                or type(page) is not list
            ):
                raise RuntimeError(
                    "Task 11 change-set inventory is unauthenticated"
                )
            summaries.extend(page)
            token = value.get("NextToken")
            if token is None:
                break
            if (
                type(token) is not str
                or not token
                or token in seen
            ):
                raise RuntimeError(
                    "Task 11 change-set pagination is ambiguous"
                )
            seen.add(token)
        else:
            raise RuntimeError("Task 11 change-set inventory is unbounded")
        active = [
            item
            for item in summaries
            if type(item) is not dict
            or item.get("Status")
            not in {"DELETE_COMPLETE", "FAILED"}
        ]
        if active:
            raise RuntimeError("Task 11 CloudFormation is not quiescent")
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[1],
            operation="acquire_cfn_quiescence",
            evidence={
                "custody_nonce_sha256": custody_nonce_sha256,
                "summary_count": len(summaries),
                "page_count": len(seen) + 1,
            },
        )

    def _load_h1d_runtime_bundle(
        self,
        *,
        request: object,
    ) -> RuntimeRevalidationBundle:
        coordinate = exact_input_coordinate_from_mapping(
            asdict(self._boundary.inputs[10]),
            expected_kind="H1D_LIVE_REQUEST",
        )
        document = load_exact_input(
            s3=self._adapters.clients.s3,
            coordinate=coordinate,
        )
        expected_fields = {
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
        body = dict(document)
        identity = body.pop("canonical_identity_sha256", None)
        if (
            set(document) != expected_fields
            or document["schema_version"] != 1
            or document["record_type"]
            != "glm52_task11_h1d_live_input_v1"
            or document["account_id"] != self._config.account_id
            or document["region"] != self._config.region
            or document["run_id"] != self._config.run_id
            or document["activation_id"] != self._config.activation_id
            or document["generation"] != getattr(request, "generation")
            or type(document["expected_state"]) is not dict
            or type(document["expected_state_authentication"]) is not dict
            or type(document["runtime_revalidation_sources"]) is not dict
            or type(document["spend_request_template"]) is not dict
            or type(document["sky_probe_request"]) is not dict
            or _SHA.fullmatch(
                document["sky_probe_admission_identity_sha256"]
            )
            is None
            or identity != canonical_sha256(body)
        ):
            raise RuntimeError("Task 11 H.1d input drifted")
        expected = h1d_expected_state_from_mapping(
            document["expected_state"]
        )
        authentication_value = document[
            "expected_state_authentication"
        ]
        auth_fields = set(
            ExpectedStateAuthentication.__dataclass_fields__
        )
        if (
            set(authentication_value) != auth_fields
            or type(authentication_value.get("direct_read_request_ids"))
            is not list
        ):
            raise RuntimeError("Task 11 H.1d expected-state auth drifted")
        authentication = ExpectedStateAuthentication(
            **{
                **authentication_value,
                "direct_read_request_ids": tuple(
                    authentication_value["direct_read_request_ids"]
                ),
            }
        )
        if (
            expected.activation_id != self._config.activation_id
            or expected.manifest_identity_sha256
            != authentication.task7_postcreate_manifest_identity_sha256
        ):
            raise RuntimeError("Task 11 H.1d expected state was substituted")
        source_coordinates = runtime_revalidation_sources_from_mapping(
            document["runtime_revalidation_sources"],
            activation_id=self._config.activation_id,
            generation=getattr(request, "generation"),
        )
        sources = load_runtime_revalidation_sources(
            s3=self._adapters.clients.s3,
            coordinates=source_coordinates,
        )
        source_by_kind = {
            item.coordinate.input_kind: item for item in sources
        }
        expected_source_identities = {
            "TASK6_MIGRATION_MANIFEST": (
                authentication.task6_manifest_identity_sha256
            ),
            "TASK6_TEMPLATE_INVENTORY": (
                authentication.task6_templates_identity_sha256
            ),
            "TASK7_POSTCREATE_MANIFEST": (
                authentication.task7_postcreate_manifest_identity_sha256
            ),
            "TASK7_SUPPORT_INVENTORY": (
                authentication.task7_inventory_identity_sha256
            ),
        }
        if any(
            source_by_kind[kind].coordinate.body_sha256 != expected_identity
            for kind, expected_identity in expected_source_identities.items()
        ):
            raise RuntimeError(
                "Task 11 Task 6/7 runtime source was substituted"
            )
        semantic_authority = (
            validate_runtime_revalidation_source_documents(
                {
                    kind: source_by_kind[kind].document
                    for kind in (
                        "RUNTIME_CUTOFF_AUTHORITY",
                        "RUNTIME_ATTACHMENT_INVENTORY",
                        "CREDENTIAL_PROBE_INVENTORY",
                    )
                },
                activation_id=self._config.activation_id,
                generation=getattr(request, "generation"),
                task7_inventory_identity_sha256=(
                    authentication.task7_inventory_identity_sha256
                ),
            )
        )
        assumed_role_id = self._adapters.clients.assumed_role_id
        live_role_id = (
            assumed_role_id.split(":", 1)[0]
            if type(assumed_role_id) is str
            else None
        )
        decision_role_arn = (
            "arn:aws:iam::"
            + self._config.account_id
            + ":role/keep-glm52-h1g-support-decision"
        )
        closure_role_id = semantic_authority.role_ids_by_arn.get(
            self._config.closure_role_arn
        )
        session_binding_validator = getattr(
            self,
            "_session_binding_validator",
            None,
        )
        if session_binding_validator is None and (
            semantic_authority.role_ids_by_arn.get(decision_role_arn)
            is None
            or closure_role_id != live_role_id
        ):
            raise RuntimeError(
                "Task 11 runtime cutoff authority drifted"
            )
        if session_binding_validator is None:
            current_session = bind_current_closure_session_to_cutoff(
                activation_id=self._config.activation_id,
                decision_role_arn=decision_role_arn,
                closure_role_arn=self._config.closure_role_arn,
                closure_role_id=closure_role_id,
                cutoff_at=(
                    semantic_authority.runtime_credential_cutoff_at
                ),
                credential_issue_time=(
                    self._adapters.clients.credential_issue_time
                ),
                credential_expiration=(
                    self._adapters.clients.credential_expiration
                ),
                observed_at=_rfc3339(_utc_now()),
                assume_role_request_id=(
                    self._adapters.clients.assume_role_request_id
                ),
                source_caller_identity=(
                    self._adapters.clients.source_caller_identity
                ),
                closure_caller_identity=(
                    self._adapters.clients.caller_identity
                ),
            )
        else:
            bind = getattr(
                session_binding_validator,
                "bind_runtime_session",
                None,
            )
            if not callable(bind):
                raise RuntimeError(
                    "Task 11 read-only session validator is incomplete"
                )
            current_session = bind(
                config=self._config,
                clients=self._adapters.clients,
                semantic_authority=semantic_authority,
            )
            if (
                type(getattr(current_session, "canonical_identity_sha256", None))
                is not str
                or _SHA.fullmatch(
                    current_session.canonical_identity_sha256
                )
                is None
            ):
                raise RuntimeError(
                    "Task 11 read-only session binding is unauthenticated"
                )
        bundle_body = {
            "h1d_input_identity_sha256": identity,
            "expected_state_identity_sha256": (
                expected.canonical_identity_sha256
            ),
            "expected_state_authentication_identity_sha256": (
                authentication.canonical_identity_sha256
            ),
            "source_identities": tuple(
                item.coordinate.canonical_identity_sha256
                for item in sources
            ),
            "source_response_identities": tuple(
                item.response_identity_sha256 for item in sources
            ),
            "current_session_identity_sha256": (
                current_session.canonical_identity_sha256
            ),
            "semantic_authority_identity_sha256": (
                semantic_authority.canonical_identity_sha256
            ),
        }
        return RuntimeRevalidationBundle(
            document=document,
            expected_state=expected,
            expected_state_authentication=authentication,
            sources=sources,
            semantic_authority=semantic_authority,
            current_session=current_session,
            canonical_identity_sha256=canonical_sha256(bundle_body),
        )

    def revalidate_runtime_attachments_and_seals(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        return (
            self._read_only_phases.revalidate_runtime_attachments_and_seals(
                request=request,
                custody_nonce_sha256=custody_nonce_sha256,
            )
        )

    def _read_only_revalidate_runtime_attachments_and_seals(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        runtime = self._load_h1d_runtime_bundle(request=request)
        clients = self._adapters.clients.h1d_clients
        if type(clients) is not dict:
            raise RuntimeError("Task 11 H.1d AWS clients are absent")
        inventory = revalidate_runtime_attachment_specs(
            specs=runtime.expected_state.specs,
            clients=clients,
            deadline=_utc_now() + timedelta(seconds=30),
        )
        stack_transport = self._describe_stack(
            self._config.support_stack_id
        )
        callees = (
            self._config.source_gpu_spend_version_arn,
            self._config.source_submission_intent_version_arn,
            self._config.source_controller_baseline_version_arn,
            self._config.source_control_plane_readiness_version_arn,
            self._config.source_submission_acquisition_version_arn,
            self._config.fence_executor_version_arn,
            self._config.fence_successor_version_arn,
            self._config.claim_writer_version_arn,
            self._config.decision_writer_version_arn,
            self._config.terminal_v1_writer_version_arn,
            self._config.closure_handoff_version_arn,
            self._config.attestation_version_arn,
            self._config.launch_admission_version_arn,
            self._config.numeric_binding_version_arn,
        )
        function_request_ids = []
        for version_arn in callees:
            value = self._adapters.clients.lambda_client.get_function(
                FunctionName=version_arn
            )
            metadata = (
                value.get("ResponseMetadata")
                if type(value) is dict
                else None
            )
            configuration = (
                value.get("Configuration")
                if type(value) is dict
                else None
            )
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 200
                or type(metadata.get("RequestId")) is not str
                or type(configuration) is not dict
                or configuration.get("FunctionArn") != version_arn
            ):
                raise RuntimeError(
                    "Task 11 runtime attachment drifted"
                )
            function_request_ids.append(metadata["RequestId"])
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[2],
            operation="revalidate_runtime_attachments_and_seals",
            evidence={
                "custody_nonce_sha256": custody_nonce_sha256,
                "support_stack_transport_request_id": stack_transport,
                "runtime_revalidation_identity_sha256": (
                    runtime.canonical_identity_sha256
                ),
                "runtime_semantic_authority_identity_sha256": (
                    runtime.semantic_authority.canonical_identity_sha256
                ),
                "runtime_source_request_ids": tuple(
                    item.request_id for item in runtime.sources
                ),
                "runtime_source_response_identities": tuple(
                    item.response_identity_sha256
                    for item in runtime.sources
                ),
                "current_session": asdict(runtime.current_session),
                "runtime_inventory": asdict(inventory),
                "callee_version_arns": callees,
                "function_request_ids": tuple(function_request_ids),
            },
        )

    def warm_clients_and_construct(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        return self._read_only_phases.warm_clients_and_construct(
            request=request,
            custody_nonce_sha256=custody_nonce_sha256,
        )

    def _read_only_warm_clients_and_construct(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        caller = self._caller_identity()
        return self._authenticated_read_receipt(
            step_name="CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION",
            operation="warm_clients_and_construct",
            evidence={
                "custody_nonce_sha256": custody_nonce_sha256,
                "request_identity_sha256": getattr(
                    request,
                    "canonical_identity_sha256",
                    None,
                ),
                "caller_identity": dict(caller),
            },
        )

    def size_non_authoritative(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        return self._read_only_phases.size_non_authoritative(
            request=request,
            custody_nonce_sha256=custody_nonce_sha256,
        )

    def _read_only_size_non_authoritative(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        value = self._adapters.clients.s3.list_object_versions(
            Bucket=self._config.campaign_bucket,
            Prefix="campaigns/" + self._config.run_id + "/",
            MaxKeys=1,
            ExpectedBucketOwner=self._config.account_id,
        )
        if (
            type(value) is not dict
            or type(value.get("ResponseMetadata")) is not dict
            or value["ResponseMetadata"].get("HTTPStatusCode") != 200
        ):
            raise RuntimeError("Task 11 sizing read is unauthenticated")
        return self._authenticated_read_receipt(
            step_name="NON_AUTHORITATIVE_SIZING",
            operation="size_non_authoritative",
            evidence={
                "custody_nonce_sha256": custody_nonce_sha256,
                "request_identity_sha256": getattr(
                    request,
                    "canonical_identity_sha256",
                    None,
                ),
                "transport": value["ResponseMetadata"],
                "version_count": len(value.get("Versions", [])),
                "delete_marker_count": len(
                    value.get("DeleteMarkers", [])
                ),
            },
        )

    def stable_tls_sky_identity_preflight(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        return self._read_only_phases.stable_tls_sky_identity_preflight(
            request=request,
            custody_nonce_sha256=custody_nonce_sha256,
        )

    def _read_only_stable_tls_sky_identity_preflight(
        self,
        *,
        request: object,
        custody_nonce_sha256: str,
    ) -> object:
        value = self._adapters.clients.lambda_client.get_function(
            FunctionName=self._config.attestation_version_arn,
        )
        if (
            type(value) is not dict
            or type(value.get("Configuration")) is not dict
            or value["Configuration"].get("FunctionArn")
            != self._config.attestation_version_arn
            or type(value.get("ResponseMetadata")) is not dict
            or value["ResponseMetadata"].get("HTTPStatusCode") != 200
        ):
            raise RuntimeError("Task 11 attestation version drifted")
        return self._authenticated_read_receipt(
            step_name="STABLE_TLS_SKY_IDENTITY_PREFLIGHT",
            operation="stable_tls_sky_identity_preflight",
            evidence={
                "custody_nonce_sha256": custody_nonce_sha256,
                "attestation_version_arn": (
                    self._config.attestation_version_arn
                ),
                "configuration": value["Configuration"],
                "transport": value["ResponseMetadata"],
            },
        )

    def _invoke_source_version(
        self,
        *,
        version_arn: str,
        method_name: str,
        request: object,
        predecessor_version_id: str,
        prior_publications: tuple[SourcePublication, ...],
        custody_nonce_sha256: str,
    ) -> SourcePublication:
        source_kind = _SOURCE_METHOD_KINDS[method_name]
        index = tuple(_SOURCE_METHOD_KINDS).index(method_name)
        if (
            len(prior_publications) != index
            or len(self._source_wire_payloads) != index
        ):
            raise RuntimeError("Task 11 source order drifted")
        payload = self._invoke_exact(
            version_arn=version_arn,
            payload={
                "schema_version": 1,
                "record_type": "glm52_task11_source_publisher_request_v1",
                "activation_id": self._config.activation_id,
                "generation": getattr(request, "generation", None),
                "source_kind": source_kind,
                "predecessor_version_id": predecessor_version_id,
                "input_coordinate": asdict(self._boundary.inputs[index]),
                "prior_publications": list(self._source_wire_payloads),
                "custody_nonce_sha256": custody_nonce_sha256,
            },
        )
        publication = source_publication_from_payload(
            payload,
            expected_source_kind=source_kind,
            expected_predecessor_version_id=predecessor_version_id,
        )
        self._source_wire_payloads.append(payload)
        return publication

    def publish_gpu_spend_snapshot(self, **kwargs: object) -> SourcePublication:
        return self._invoke_source_version(
            version_arn=self._config.source_gpu_spend_version_arn,
            method_name="publish_gpu_spend_snapshot",
            **kwargs,
        )

    def publish_submission_intent(
        self,
        **kwargs: object,
    ) -> SourcePublication:
        return self._invoke_source_version(
            version_arn=self._config.source_submission_intent_version_arn,
            method_name="publish_submission_intent",
            **kwargs,
        )

    def publish_controller_baseline(
        self,
        **kwargs: object,
    ) -> SourcePublication:
        return self._invoke_source_version(
            version_arn=self._config.source_controller_baseline_version_arn,
            method_name="publish_controller_baseline",
            **kwargs,
        )

    def publish_control_plane_readiness(
        self,
        **kwargs: object,
    ) -> SourcePublication:
        return self._invoke_source_version(
            version_arn=(
                self._config.source_control_plane_readiness_version_arn
            ),
            method_name="publish_control_plane_readiness",
            **kwargs,
        )

    def publish_submission_acquisition(
        self,
        **kwargs: object,
    ) -> SourcePublication:
        return self._invoke_source_version(
            version_arn=(
                self._config.source_submission_acquisition_version_arn
            ),
            method_name="publish_submission_acquisition",
            **kwargs,
        )

    def _invoke_effect_writer(
        self,
        *,
        writer_kind: str,
        version_arn: str,
        boundary_index: int,
        request: object,
        custody_nonce_sha256: str,
        runtime_bindings: Optional[Mapping[str, object]] = None,
        prepared_authority: Optional[Mapping[str, object]] = None,
    ) -> EffectWriteResult:
        if writer_kind in self._effect_results:
            raise RuntimeError("Task 11 effect writer was replayed")
        payload = self._invoke_exact(
            version_arn=version_arn,
            payload={
                "schema_version": 1,
                "record_type": "glm52_task11_effect_writer_request_v1",
                "activation_id": self._config.activation_id,
                "generation": getattr(request, "generation", None),
                "writer_kind": writer_kind,
                "input_coordinate": asdict(
                    self._boundary.inputs[boundary_index]
                ),
                "source_publications": list(
                    self._source_wire_payloads
                ),
                "dependency_results": list(
                    self._effect_wire_payloads
                ),
                "runtime_bindings": dict(runtime_bindings or {}),
                "prepared_authority": (
                    None
                    if prepared_authority is None
                    else dict(prepared_authority)
                ),
                "custody_nonce_sha256": custody_nonce_sha256,
            },
        )
        result = effect_write_result_from_payload(
            payload,
            expected_writer_kind=writer_kind,
        )
        self._effect_results[writer_kind] = result
        self._effect_wire_payloads.append(payload)
        return result

    @staticmethod
    def _effect_receipt(
        *,
        result: EffectWriteResult,
        step_name: str,
        audit_kind: Optional[str] = None,
    ) -> object:
        audits = ()
        if audit_kind is not None:
            audits = (
                build_audit_evidence(
                    kind=audit_kind,
                    audit_identity_sha256=(
                        result.authority_audit_body_sha256
                    ),
                    invocation_identity_sha256=canonical_sha256(
                        {
                            "writer_kind": result.writer_kind,
                            "direct_request_id": result.direct_request_id,
                            "direct_server_date": (
                                result.direct_server_date
                            ),
                            "writer_identity_sha256": (
                                result.canonical_identity_sha256
                            ),
                        }
                    ),
                    closing_revision=result.closing_revision,
                ),
            )
        return build_step_receipt(
            step_name=step_name,
            operation_identity_sha256=result.canonical_identity_sha256,
            audits=audits,
        )

    def _exact_read_effect(self, result: EffectWriteResult) -> None:
        value = self._adapters.clients.s3.get_object(
            Bucket=result.object_identity.bucket,
            Key=result.object_identity.key,
            VersionId=result.object_identity.version_id,
            ExpectedBucketOwner=self._config.account_id,
            ChecksumMode="ENABLED",
        )
        stream = value.get("Body") if type(value) is dict else None
        read = getattr(stream, "read", None)
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or value.get("VersionId")
            != result.object_identity.version_id
            or value.get("ChecksumSHA256")
            != result.object_identity.checksum_sha256_base64
            or not callable(read)
            or read(2 * 1024 * 1024 + 1) != result.raw
        ):
            raise RuntimeError("Task 11 exact effect readback drifted")

    @staticmethod
    def _fence_phase_receipt(
        *,
        operation: str,
        audit_kind: str,
        phase_identity_sha256: str,
        audit_identity_sha256: str,
        closing_revision: int,
    ) -> object:
        audit = build_audit_evidence(
            kind=audit_kind,
            audit_identity_sha256=audit_identity_sha256,
            invocation_identity_sha256=canonical_sha256(
                {
                    "operation": operation,
                    "phase_identity_sha256": phase_identity_sha256,
                    "audit_identity_sha256": audit_identity_sha256,
                    "closing_revision": closing_revision,
                }
            ),
            closing_revision=closing_revision,
        )
        return build_step_receipt(
            step_name=operation.upper(),
            operation_identity_sha256=phase_identity_sha256,
            audits=(audit,),
        )

    def create_batch_successor(self, **kwargs: object) -> object:
        result = self._invoke_effect_writer(
            writer_kind="FenceSuccessor",
            version_arn=self._config.fence_successor_version_arn,
            boundary_index=5,
            request=kwargs["request"],
            custody_nonce_sha256=kwargs["custody_nonce_sha256"],
        )
        return self._effect_receipt(
            result=result,
            step_name="CREATE_BATCH_SUCCESSOR",
            audit_kind="BATCH_SUCCESSOR",
        )

    def invoke_fence_executor_once(self, **kwargs: object) -> object:
        if self._executed_fence_result is not None:
            raise RuntimeError("Task 11 fence executor was replayed")
        input_coordinate = self._boundary.inputs[6]
        request_coordinate = {
            "bucket": input_coordinate.bucket,
            "key": input_coordinate.key,
            "version_id": input_coordinate.version_id,
            "file_sha256": input_coordinate.file_sha256,
            "canonical_identity_sha256": input_coordinate.body_sha256,
        }
        payload = self._invoke_exact(
            version_arn=self._config.fence_executor_version_arn,
            payload={
                "request_coordinate": request_coordinate,
            },
        )
        executed = parse_fence_execution_result(payload)
        if (
            executed.slot.value != "BATCH_FIVE_SOURCE_ACTIVATION"
            or executed.request_identity_sha256
            == executed.prepared_identity_sha256
            or executed.original_template_body_sha256
            != executed.processed_template_body_sha256
            or executed.first_stable_snapshot_identity_sha256
            != executed.second_stable_snapshot_identity_sha256
            or executed.stabilization_first_evidence_sha256
            == executed.stabilization_second_evidence_sha256
            or executed.observed_poststate_stack_role_arn
            != executed.expected_poststate_stack_role_arn
            or not executed.batch_execution_eligible
        ):
            raise RuntimeError(
                "Task 11 fence v2 execution evidence drifted"
            )
        self._executed_fence_result = executed
        return build_step_receipt(
            step_name="INVOKE_FENCE_EXECUTOR_ONCE",
            operation_identity_sha256=canonical_sha256(
                {
                    "phase": "EXECUTE_PREPARED_V2",
                    "request_coordinate": request_coordinate,
                    "result": executed.to_dict(),
                }
            ),
            coherent_readback=True,
        )

    def _policy_readback(
        self,
        *,
        operation: str,
        request: object,
    ) -> object:
        value = self._adapters.clients.s3.get_bucket_policy(
            Bucket=self._config.campaign_bucket,
            ExpectedBucketOwner=self._config.account_id,
        )
        if (
            type(value) is not dict
            or type(value.get("Policy")) is not str
            or type(value.get("ResponseMetadata")) is not dict
            or value["ResponseMetadata"].get("HTTPStatusCode") != 200
        ):
            raise RuntimeError("Task 11 policy readback is unauthenticated")
        try:
            policy = json.loads(value["Policy"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Task 11 policy readback is invalid") from exc
        policy_sha256 = hashlib.sha256(
            canonical_json_bytes(policy)
        ).hexdigest()
        if (
            self._executed_fence_result is None
            or policy_sha256
            != self._executed_fence_result.policy_sha256
        ):
            raise RuntimeError("Task 11 policy readback drifted")
        return self._authenticated_read_receipt(
            step_name=operation.upper(),
            operation=operation,
            evidence={
                "request_identity_sha256": getattr(
                    request,
                    "canonical_identity_sha256",
                    None,
                ),
                "policy_sha256": policy_sha256,
                "transport": value["ResponseMetadata"],
            },
        )

    def policy_deny_readback_one(self, **kwargs: object) -> object:
        return self._policy_readback(
            operation="policy_deny_readback_one",
            request=kwargs["request"],
        )

    def policy_deny_readback_two(self, **kwargs: object) -> object:
        return self._policy_readback(
            operation="policy_deny_readback_two",
            request=kwargs["request"],
        )

    def seal_decision_and_acquire_barrier(
        self,
        **kwargs: object,
    ) -> object:
        index, control = self._adapters.ledger.read_coherent(
            items=(
                (
                    LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        self._config.run_id,
                        "ACTIVATION#"
                        + self._config.activation_id
                        + "#CONTROL",
                    ),
                    "glm52_production_control",
                ),
            )
        )
        if (
            index["current_activation_id"] != self._config.activation_id
            or control["activation_id"] != self._config.activation_id
            or index["campaign_identity_sha256"]
            != control["campaign_identity_sha256"]
            or self._executed_fence_result is None
            or control["phase"] != "OPEN"
            or control["barrier_state"] != "OPEN"
            or control["decision_seal_state"] != "OPEN"
            or type(control["revision"]) is not int
            or control["revision"] <= 0
            or type(control["active_epoch"]) is not int
            or control["active_epoch"] <= 0
            or type(control["active_execution_arn"]) is not str
            or not control["active_execution_arn"]
            or type(control["fence_head_version_id"]) is not str
            or not control["fence_head_version_id"]
        ):
            raise RuntimeError("Task 11 seal authority drifted")
        source_batch = kwargs["source_batch"]
        observed_at = _rfc3339(_utc_now())
        control_after = {
            **control,
            "barrier_state": "ACQUIRED",
            "decision_seal_state": "SEALED",
            "revision": control["revision"] + 1,
            "updated_at": observed_at,
        }
        record_contract.validate_record(
            "glm52_production_control",
            control_after,
        )
        body = {
            "schema_version": 1,
            "record_type": "glm52_task11_decision_seal_v1",
            "run_id": self._config.run_id,
            "activation_id": self._config.activation_id,
            "generation": getattr(kwargs["request"], "generation", None),
            "campaign_identity_sha256": index[
                "campaign_identity_sha256"
            ],
            "epoch": control["active_epoch"],
            "execution_arn": control["active_execution_arn"],
            "control_revision": control_after["revision"],
            "authorized_transition_from_revision": control["revision"],
            "authorized_transition_to_revision": control_after["revision"],
            "barrier_nonce_sha256": control["barrier_nonce_sha256"],
            "barrier_state": control_after["barrier_state"],
            "decision_seal_state": control_after["decision_seal_state"],
            "fence_head_body_sha256": control["fence_head_body_sha256"],
            "fence_head_version_id": control["fence_head_version_id"],
            "fence_execution_identity_sha256": canonical_sha256(
                asdict(self._executed_fence_result)
            ),
            "source_batch_identity_sha256": (
                source_batch.canonical_identity_sha256
            ),
        }
        sealed = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
        physical = {
            "PK": record_contract.ledger_pk(self._config.run_id),
            "SK": (
                "ACTIVATION#"
                + self._config.activation_id
                + "#TASK11_DECISION_SEAL#"
                + f"{body['generation']:08d}"
            ),
            **sealed,
        }
        pk = record_contract.ledger_pk(self._config.run_id)
        control_sk = (
            "ACTIVATION#" + self._config.activation_id + "#CONTROL"
        )
        names = {
            "#revision": "revision",
            "#activation": "activation_id",
            "#campaign": "campaign_identity_sha256",
            "#epoch": "active_epoch",
            "#execution": "active_execution_arn",
            "#head_body": "fence_head_body_sha256",
            "#head_version": "fence_head_version_id",
            "#barrier_nonce": "barrier_nonce_sha256",
            "#barrier_state": "barrier_state",
            "#seal_state": "decision_seal_state",
            "#updated": "updated_at",
        }
        values = {
            ":before_revision": control["revision"],
            ":after_revision": control_after["revision"],
            ":activation": self._config.activation_id,
            ":campaign": index["campaign_identity_sha256"],
            ":epoch": control["active_epoch"],
            ":execution": control["active_execution_arn"],
            ":head_body": control["fence_head_body_sha256"],
            ":head_version": control["fence_head_version_id"],
            ":barrier_nonce": control["barrier_nonce_sha256"],
            ":open": "OPEN",
            ":acquired": "ACQUIRED",
            ":sealed": "SEALED",
            ":updated": observed_at,
        }
        encoded_values = {
            key: encode_attribute_value(value)
            for key, value in values.items()
        }
        transaction = {
            "TransactItems": [
                {
                    "ConditionCheck": {
                        "TableName": self._config.ledger_table_name,
                        "Key": encode_item(
                            {"PK": pk, "SK": "ACTIVATION_INDEX"}
                        ),
                        "ConditionExpression": (
                            "#activation = :activation AND "
                            "#campaign = :campaign AND "
                            "#revision = :index_revision"
                        ),
                        "ExpressionAttributeNames": {
                            "#activation": "current_activation_id",
                            "#campaign": "campaign_identity_sha256",
                            "#revision": "revision",
                        },
                        "ExpressionAttributeValues": {
                            ":activation": encoded_values[":activation"],
                            ":campaign": encoded_values[":campaign"],
                            ":index_revision": encode_attribute_value(
                                index["revision"]
                            ),
                        },
                    }
                },
                {
                    "Update": {
                        "TableName": self._config.ledger_table_name,
                        "Key": encode_item({"PK": pk, "SK": control_sk}),
                        "ConditionExpression": (
                            "#activation = :activation AND "
                            "#campaign = :campaign AND "
                            "#revision = :before_revision AND "
                            "#epoch = :epoch AND "
                            "#execution = :execution AND "
                            "#head_body = :head_body AND "
                            "#head_version = :head_version AND "
                            "#barrier_nonce = :barrier_nonce AND "
                            "#barrier_state = :open AND "
                            "#seal_state = :open"
                        ),
                        "UpdateExpression": (
                            "SET #barrier_state = :acquired, "
                            "#seal_state = :sealed, "
                            "#revision = :after_revision, "
                            "#updated = :updated"
                        ),
                        "ExpressionAttributeNames": names,
                        "ExpressionAttributeValues": encoded_values,
                    }
                },
                {
                    "Put": {
                        "TableName": self._config.ledger_table_name,
                        "Item": encode_item(physical),
                        "ConditionExpression": (
                            "attribute_not_exists(#pk) AND "
                            "attribute_not_exists(#sk)"
                        ),
                        "ExpressionAttributeNames": {
                            "#pk": "PK",
                            "#sk": "SK",
                        },
                    }
                },
            ],
            "ClientRequestToken": canonical_sha256(
                {
                    "domain": "TASK11_DECISION_SEAL_BARRIER_V1",
                    "seal_identity_sha256": sealed[
                        "canonical_identity_sha256"
                    ],
                }
            )[:36],
            "ReturnConsumedCapacity": "NONE",
        }
        write_request_id = None
        try:
            value = self._adapters.clients.dynamodb.transact_write_items(
                **transaction
            )
        except Exception:
            value = None
        if type(value) is dict:
            metadata = value.get("ResponseMetadata")
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 200
                or type(metadata.get("RequestId")) is not str
                or not metadata["RequestId"]
            ):
                raise RuntimeError(
                    "Task 11 seal transaction was unauthenticated"
                )
            write_request_id = metadata["RequestId"]
        read = self._adapters.clients.dynamodb.transact_get_items(
            TransactItems=[
                {
                    "Get": {
                        "TableName": self._config.ledger_table_name,
                        "Key": encode_item(
                            {"PK": pk, "SK": "ACTIVATION_INDEX"}
                        ),
                    }
                },
                {
                    "Get": {
                        "TableName": self._config.ledger_table_name,
                        "Key": encode_item({"PK": pk, "SK": control_sk}),
                    }
                },
                {
                    "Get": {
                        "TableName": self._config.ledger_table_name,
                        "Key": encode_item(
                            {"PK": physical["PK"], "SK": physical["SK"]}
                        ),
                    }
                },
            ],
            ReturnConsumedCapacity="NONE",
        )
        read_metadata = (
            read.get("ResponseMetadata")
            if type(read) is dict
            else None
        )
        responses = read.get("Responses") if type(read) is dict else None
        if (
            type(read_metadata) is not dict
            or read_metadata.get("HTTPStatusCode") != 200
            or type(read_metadata.get("RequestId")) is not str
            or not read_metadata["RequestId"]
            or type(responses) is not list
            or len(responses) != 3
            or any(
                type(item) is not dict or type(item.get("Item")) is not dict
                for item in responses
            )
        ):
            raise RuntimeError("Task 11 seal readback was unavailable")
        index_after = decode_item(responses[0]["Item"])
        stored_control = decode_item(responses[1]["Item"])
        stored_seal = decode_item(responses[2]["Item"])
        if (
            {
                key: value
                for key, value in index_after.items()
                if key not in {"PK", "SK"}
            }
            != index
            or {
                key: value
                for key, value in stored_control.items()
                if key not in {"PK", "SK"}
            }
            != control_after
            or stored_seal != physical
        ):
            raise RuntimeError(
                "Task 11 seal/barrier atomic readback drifted"
            )
        self._seal = sealed
        self._seal_transition_identity_sha256 = canonical_sha256(
            {
                "index": index,
                "control_before": control,
                "control_after": control_after,
                "seal": sealed,
                "transaction_client_request_token": transaction[
                    "ClientRequestToken"
                ],
                "write_request_id": write_request_id,
                "read_request_id": read_metadata["RequestId"],
            }
        )
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[5],
            operation="seal_decision_and_acquire_barrier",
            evidence={
                "seal": sealed,
                "control_after": control_after,
                "transaction_identity_sha256": (
                    self._seal_transition_identity_sha256
                ),
                "write_request_id": write_request_id,
                "get_request_id": read_metadata.get("RequestId"),
            },
        )

    def _invoke_exact(
        self,
        *,
        version_arn: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        qualifier = version_arn.rsplit(":", 1)[-1]
        if (
            _VERSION_ARN.fullmatch(version_arn) is None
            or not qualifier.isdigit()
            or qualifier.startswith("0")
        ):
            raise RuntimeError(
                "Task 11 Lambda target is not a published exact version"
            )
        request_bytes = canonical_json_bytes(payload)
        value = self._adapters.clients.lambda_client.invoke(
            FunctionName=version_arn,
            InvocationType="RequestResponse",
            Payload=request_bytes,
        )
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        stream = value.get("Payload") if type(value) is dict else None
        read = getattr(stream, "read", None)
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
            or value.get("StatusCode") != 200
            or value.get("ExecutedVersion") != qualifier
            or "FunctionError" in value
            or not callable(read)
        ):
            raise RuntimeError("Task 11 exact Lambda invocation failed")
        raw = read(2 * 1024 * 1024 + 1)
        try:
            parsed = json.loads(raw.decode("ascii"))
        except (AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Task 11 exact Lambda response is invalid"
            ) from exc
        if type(parsed) is not dict:
            raise RuntimeError("Task 11 exact Lambda response is not closed")
        self._lambda_invocations.append(
            {
                "version_arn": version_arn,
                "executed_version": qualifier,
                "request_id": metadata["RequestId"],
                "request_identity_sha256": hashlib.sha256(
                    request_bytes
                ).hexdigest(),
                "response_identity_sha256": hashlib.sha256(raw).hexdigest(),
                "invocation_identity_sha256": canonical_sha256(
                    {
                        "version_arn": version_arn,
                        "executed_version": qualifier,
                        "request_id": metadata["RequestId"],
                        "status_code": value["StatusCode"],
                        "request_identity_sha256": hashlib.sha256(
                            request_bytes
                        ).hexdigest(),
                        "response_identity_sha256": hashlib.sha256(
                            raw
                        ).hexdigest(),
                    }
                ),
            }
        )
        return parsed

    @staticmethod
    def _attestation_from_payload(
        value: Mapping[str, object],
    ) -> AttestationResult:
        try:
            result = AttestationResult(
                **{
                    **value,
                    "sky_roles": tuple(value["sky_roles"]),
                }
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Task 11 attestation payload is invalid"
            ) from exc
        return result

    def _attest(self, *, request: object) -> AttestationResult:
        self._attestation_sequence += 1
        value = self._invoke_exact(
            version_arn=self._config.attestation_version_arn,
            payload={
                "schema_version": 1,
                "record_type": "glm52_task11_attestation_request_v1",
                "run_id": self._config.run_id,
                "activation_id": self._config.activation_id,
                "request_identity_sha256": getattr(
                    request,
                    "canonical_identity_sha256",
                    None,
                ),
                "freshness_nonce": secrets.token_hex(32),
                "sequence": self._attestation_sequence,
                "task9_deployed_identity_coordinate": asdict(
                    self._boundary.inputs[13]
                ),
                "task9_deployed_identity_sha256": (
                    self._boundary.inputs[13].body_sha256
                ),
            },
        )
        return self._attestation_from_payload(value)

    def attest_sky_identity_pre_decision(
        self,
        *,
        request: object,
    ) -> AttestationResult:
        return self._attest(request=request)

    def create_or_recover_claim(self, **kwargs: object) -> object:
        result = self._invoke_effect_writer(
            writer_kind="ClaimWriter",
            version_arn=self._config.claim_writer_version_arn,
            boundary_index=7,
            request=kwargs["request"],
            custody_nonce_sha256=kwargs["custody_nonce_sha256"],
        )
        return self._effect_receipt(
            result=result,
            step_name=CLOSURE_STEPS[7],
            audit_kind=AUTHORITY_AUDIT_KINDS[8],
        )

    def remaining_milliseconds(self) -> int:
        value = self._remaining()
        if type(value) is not int or value <= 0:
            raise RuntimeError("Task 11 remaining time is invalid")
        return value

    def model_current_launch_or_expire(self, **kwargs: object) -> object:
        request = kwargs["request"]
        coordinate = exact_input_coordinate_from_mapping(
            asdict(self._boundary.inputs[8]),
            expected_kind="DECISION_CREATE_REQUEST",
        )
        document = _validate_effect_writer_input(
            load_exact_input(
                s3=self._adapters.clients.s3,
                coordinate=coordinate,
            ),
            writer_kind="DecisionWriter",
            activation_id=self._config.activation_id,
            generation=getattr(request, "generation", None),
        )
        modeled_decision = document["builder_arguments"].get("decision")
        if modeled_decision == "launch-once":
            receipt = self._authenticated_read_receipt(
                step_name=CLOSURE_STEPS[8],
                operation="model_current_launch_or_expire",
                evidence={
                    "branch": "LAUNCH",
                    "input_identity_sha256": (
                        document["canonical_identity_sha256"]
                    ),
                },
            )
            return build_launch_or_expire_decision(
                branch="LAUNCH",
                receipt=receipt,
            )
        if modeled_decision != "expire-unstarted":
            raise RuntimeError("Task 11 launch/expire model is not closed")
        decision = self._invoke_effect_writer(
            writer_kind="DecisionWriter",
            version_arn=self._config.decision_writer_version_arn,
            boundary_index=8,
            request=request,
            custody_nonce_sha256=canonical_sha256(
                {
                    "request_identity_sha256": getattr(
                        request,
                        "canonical_identity_sha256",
                        None,
                    ),
                    "branch": "EXPIRE",
                    "decision_input_identity_sha256": document[
                        "canonical_identity_sha256"
                    ],
                }
            ),
        )
        self._exact_read_effect(decision)
        terminal = self._invoke_effect_writer(
            writer_kind="TerminalV1Writer",
            version_arn=self._config.terminal_v1_writer_version_arn,
            boundary_index=9,
            request=request,
            custody_nonce_sha256=canonical_sha256(
                {
                    "request_identity_sha256": getattr(
                        request,
                        "canonical_identity_sha256",
                        None,
                    ),
                    "expire_decision_identity_sha256": (
                        decision.canonical_identity_sha256
                    ),
                }
            ),
        )
        self._exact_read_effect(terminal)
        receipt = self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[8],
            operation="model_current_launch_or_expire",
            evidence={
                "branch": "EXPIRE",
                "decision_writer_identity_sha256": (
                    decision.canonical_identity_sha256
                ),
                "terminal_v1_identity_sha256": (
                    terminal.canonical_identity_sha256
                ),
            },
        )
        return build_launch_or_expire_decision(
            branch="EXPIRE",
            receipt=receipt,
            decision_writer_identity_sha256=(
                decision.canonical_identity_sha256
            ),
            terminal_v1_identity_sha256=(
                terminal.canonical_identity_sha256
            ),
        )

    def create_direct_decision(self, **kwargs: object) -> DirectDecisionResponse:
        request = kwargs["request"]
        custody_nonce_sha256 = kwargs["custody_nonce_sha256"]
        written = self._invoke_effect_writer(
            writer_kind="DecisionWriter",
            version_arn=self._config.decision_writer_version_arn,
            boundary_index=8,
            request=request,
            custody_nonce_sha256=custody_nonce_sha256,
        )
        audit = build_audit_evidence(
            kind=AUTHORITY_AUDIT_KINDS[9],
            audit_identity_sha256=written.authority_audit_body_sha256,
            invocation_identity_sha256=canonical_sha256(
                {
                    "request_id": written.direct_request_id,
                    "server_date": written.direct_server_date,
                    "writer_identity_sha256": (
                        written.canonical_identity_sha256
                    ),
                }
            ),
            closing_revision=written.closing_revision,
        )
        result = build_direct_decision_response(
            status_code=200,
            expected_bucket_owner=self._config.account_id,
            version_id=written.object_identity.version_id,
            etag=written.object_identity.etag,
            checksum_sha256=written.object_identity.file_sha256,
            request_id=written.direct_request_id,
            response_date=written.direct_server_date,
            candidate_identity_sha256=written.candidate_identity_sha256,
            audit=audit,
            custody_nonce_sha256=custody_nonce_sha256,
        )
        if (
            result.candidate_identity_sha256
            != getattr(request, "candidate_identity_sha256")
        ):
            raise RuntimeError(
                "Task 11 direct decision candidate identity drifted"
            )
        self._direct_objects[result.canonical_identity_sha256] = (
            written.object_identity.key,
            written.raw,
        )
        return result

    def paginate_and_exact_read_decision(self, **kwargs: object) -> object:
        request = kwargs["request"]
        direct = kwargs["direct_response"]
        if type(direct) is not DirectDecisionResponse:
            raise RuntimeError("Task 11 direct decision custody was lost")
        stored = self._direct_objects.get(direct.canonical_identity_sha256)
        if stored is None:
            raise RuntimeError("Task 11 direct decision is not invocation-local")
        key, expected_body = stored
        marker = None
        version_marker = None
        seen_tokens = set()
        page_transports = []
        exact_versions = []
        exact_delete_markers = []
        while True:
            list_request = {
                "Bucket": self._config.campaign_bucket,
                "Prefix": key,
                "MaxKeys": 1000,
                "ExpectedBucketOwner": self._config.account_id,
            }
            if marker is not None:
                list_request["KeyMarker"] = marker
                list_request["VersionIdMarker"] = version_marker
            page = self._adapters.clients.s3.list_object_versions(
                **list_request
            )
            page_metadata = (
                page.get("ResponseMetadata")
                if type(page) is dict
                else None
            )
            versions = page.get("Versions") if type(page) is dict else None
            deletes = (
                page.get("DeleteMarkers", [])
                if type(page) is dict
                else None
            )
            if (
                type(page_metadata) is not dict
                or page_metadata.get("HTTPStatusCode") != 200
                or type(page_metadata.get("RequestId")) is not str
                or not page_metadata["RequestId"]
                or type(versions) is not list
                or type(deletes) is not list
                or type(page.get("IsTruncated")) is not bool
            ):
                raise RuntimeError(
                    "Task 11 decision version page is unauthenticated"
                )
            page_transports.append(page_metadata)
            exact_versions.extend(
                item
                for item in versions
                if type(item) is dict and item.get("Key") == key
            )
            exact_delete_markers.extend(
                item
                for item in deletes
                if type(item) is dict and item.get("Key") == key
            )
            if page["IsTruncated"] is False:
                break
            marker = page.get("NextKeyMarker")
            version_marker = page.get("NextVersionIdMarker")
            token = (marker, version_marker)
            if (
                type(marker) is not str
                or not marker
                or type(version_marker) is not str
                or not version_marker
                or token in seen_tokens
            ):
                raise RuntimeError(
                    "Task 11 decision version pagination is ambiguous"
                )
            seen_tokens.add(token)
            if len(seen_tokens) > 100:
                raise RuntimeError(
                    "Task 11 decision version pagination is unbounded"
                )
        if (
            len(exact_versions) != 1
            or exact_versions[0].get("VersionId") != direct.version_id
            or exact_versions[0].get("IsLatest") is not True
            or exact_versions[0].get("Size") != len(expected_body)
            or exact_delete_markers
        ):
            raise RuntimeError(
                "Task 11 decision version inventory drifted"
            )
        head = self._adapters.clients.s3.head_object(
            Bucket=self._config.campaign_bucket,
            Key=key,
            VersionId=direct.version_id,
            ExpectedBucketOwner=self._config.account_id,
            ChecksumMode="ENABLED",
        )
        head_metadata = (
            head.get("ResponseMetadata")
            if type(head) is dict
            else None
        )
        expected_checksum = base64.b64encode(
            hashlib.sha256(expected_body).digest()
        ).decode("ascii")
        if (
            type(head_metadata) is not dict
            or head_metadata.get("HTTPStatusCode") != 200
            or type(head_metadata.get("RequestId")) is not str
            or not head_metadata["RequestId"]
            or head.get("VersionId") != direct.version_id
            or head.get("ContentLength") != len(expected_body)
            or head.get("ChecksumSHA256") != expected_checksum
        ):
            raise RuntimeError("Task 11 exact decision HEAD drifted")
        value = self._adapters.clients.s3.get_object(
            Bucket=self._config.campaign_bucket,
            Key=key,
            VersionId=direct.version_id,
            ExpectedBucketOwner=self._config.account_id,
            ChecksumMode="ENABLED",
        )
        body = value.get("Body") if type(value) is dict else None
        read = getattr(body, "read", None)
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
            or value.get("VersionId") != direct.version_id
            or value.get("ContentLength") != len(expected_body)
            or value.get("ChecksumSHA256") != expected_checksum
            or not callable(read)
            or read(2 * 1024 * 1024 + 1) != expected_body
        ):
            raise RuntimeError("Task 11 exact decision readback drifted")
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[11],
            operation="paginate_and_exact_read_decision",
            evidence={
                "request_identity_sha256": getattr(
                    request,
                    "canonical_identity_sha256",
                ),
                "decision_identity_sha256": (
                    direct.canonical_identity_sha256
                ),
                "key": key,
                "version_id": direct.version_id,
                "file_sha256": hashlib.sha256(expected_body).hexdigest(),
                "version_page_transports": page_transports,
                "head_transport": head_metadata,
                "get_transport": metadata,
            },
        )

    def validate_h1e_modeled_submit_once(self, **kwargs: object) -> object:
        from .glm52_sky_production_generation import (
            GenerationInventoryEntry,
            UnambiguousStartDecisionCreateReceipt,
            VersionedJsonArtifact,
            validate_modeled_submit_once,
        )

        direct = kwargs["direct_response"]
        if (
            type(direct) is not DirectDecisionResponse
            or direct.canonical_identity_sha256 not in self._direct_objects
        ):
            raise RuntimeError("Task 11 H.1e custody was lost")
        claim = self._effect_results.get("ClaimWriter")
        decision = self._effect_results.get("DecisionWriter")
        if (
            type(claim) is not EffectWriteResult
            or type(decision) is not EffectWriteResult
            or decision.object_identity.version_id != direct.version_id
            or type(decision.direct_request_started_at) is not str
            or type(decision.direct_response_received_at) is not str
        ):
            raise RuntimeError("Task 11 H.1e effect custody was lost")

        coordinate = exact_input_coordinate_from_mapping(
            asdict(self._boundary.inputs[7]),
            expected_kind="CLAIM_CREATE_REQUEST",
        )
        document = _validate_effect_writer_input(
            load_exact_input(
                s3=self._adapters.clients.s3,
                coordinate=coordinate,
            ),
            writer_kind="ClaimWriter",
            activation_id=self._config.activation_id,
            generation=getattr(kwargs["request"], "generation", None),
        )
        arguments = materialize_effect_builder_arguments(
            document,
            source_publications=list(self._source_wire_payloads),
            dependency_results=[],
            runtime_bindings={},
        )

        def artifact(value: Mapping[str, object]) -> VersionedJsonArtifact:
            return VersionedJsonArtifact(
                key=value["key"],
                version_id=value["version_id"],
                raw=canonical_json_bytes(value["record"]) + b"\n",
            )

        source_by_kind = {}
        for value in self._source_wire_payloads:
            publication = value["publication"]
            raw = base64.b64decode(value["raw_base64"], validate=True)
            source_by_kind[publication["source_kind"]] = (
                VersionedJsonArtifact(
                    key=publication["key"],
                    version_id=publication["version_id"],
                    raw=raw,
                )
            )
        claim_record = json.loads(claim.raw[:-1].decode("ascii"))
        decision_record = json.loads(decision.raw[:-1].decode("ascii"))
        inventory = []
        for row in arguments["generation_inventory"]:
            inventory.append(
                GenerationInventoryEntry(
                    key=row["key"],
                    raw=(
                        None
                        if row["record"] is None
                        else canonical_json_bytes(row["record"]) + b"\n"
                    ),
                    version_id=row["version_id"],
                    is_latest=row["is_latest"],
                    is_delete_marker=row["is_delete_marker"],
                )
            )
        inventory.extend(
            (
                GenerationInventoryEntry(
                    key=claim.object_identity.key,
                    raw=claim.raw,
                    version_id=claim.object_identity.version_id,
                    is_latest=True,
                    is_delete_marker=False,
                ),
                GenerationInventoryEntry(
                    key=decision.object_identity.key,
                    raw=decision.raw,
                    version_id=decision.object_identity.version_id,
                    is_latest=True,
                    is_delete_marker=False,
                ),
            )
        )
        digest = hashlib.sha256(decision.raw).digest()
        checksum = base64.b64encode(digest).decode("ascii")
        try:
            server = datetime.strptime(
                decision.direct_server_date,
                "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise RuntimeError("Task 11 H.1e server date drifted") from exc
        receipt = UnambiguousStartDecisionCreateReceipt(
            record_kind="generation-start-decision",
            account_id=self._config.account_id,
            region=self._config.region,
            bucket=decision.object_identity.bucket,
            expected_bucket_owner=self._config.account_id,
            operation="PutObject",
            if_none_match="*",
            generation=claim_record["generation"],
            generation_text=claim_record["generation_text"],
            submit_attempt_id=claim_record["submit_attempt_id"],
            key=decision.object_identity.key,
            content_length=len(decision.raw),
            candidate_file_sha256=digest.hex(),
            request_checksum_algorithm="SHA256",
            request_checksum_sha256_base64=checksum,
            immutable_metadata=(
                ("glm52-account-id", self._config.account_id),
                ("glm52-region", self._config.region),
                ("glm52-run-id", self._config.run_id),
                ("glm52-generation", claim_record["generation_text"]),
                (
                    "glm52-submit-attempt-id",
                    claim_record["submit_attempt_id"],
                ),
                (
                    "glm52-record-type",
                    "glm52_sky_production_generation_start_decision_v1",
                ),
                (
                    "glm52-body-sha256",
                    decision_record["start_decision_body_sha256"],
                ),
                ("glm52-file-sha256", digest.hex()),
            ),
            version_id=decision.object_identity.version_id,
            etag=decision.object_identity.etag,
            response_checksum_sha256_base64=checksum,
            aws_request_id=decision.direct_request_id,
            server_date=format_datetime(server, usegmt=True),
            request_started_at=decision.direct_request_started_at,
            response_received_at=decision.direct_response_received_at,
            http_status=200,
            outcome="created",
            source="direct-response",
        )
        now = _utc_now()
        validated = validate_modeled_submit_once(
            generation_claim=VersionedJsonArtifact(
                key=claim.object_identity.key,
                version_id=claim.object_identity.version_id,
                raw=claim.raw,
            ),
            start_decision=VersionedJsonArtifact(
                key=decision.object_identity.key,
                version_id=decision.object_identity.version_id,
                raw=decision.raw,
            ),
            descriptor=artifact(arguments["descriptor"]),
            intent=source_by_kind["SUBMISSION_INTENT"],
            approval=artifact(arguments["approval"]),
            controller_baseline=source_by_kind["CONTROLLER_BASELINE"],
            must_start_control_plane_ready=source_by_kind[
                "CONTROL_PLANE_READINESS"
            ],
            submission_acquisition=source_by_kind[
                "SUBMISSION_ACQUISITION"
            ],
            gpu_spend_snapshot=source_by_kind["GPU_SPEND"],
            start_decision_receipt=receipt,
            post_create_generation_inventory=inventory,
            post_create_audit_server_date=format_datetime(
                now,
                usegmt=True,
            ),
            now=now,
        )
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[12],
            operation="validate_h1e_modeled_submit_once",
            evidence={
                "validation": asdict(validated),
                "claim_identity_sha256": claim.canonical_identity_sha256,
                "decision_identity_sha256": (
                    decision.canonical_identity_sha256
                ),
            },
        )

    def reinspect_task8_live_authority(
        self,
        *,
        request: object,
    ) -> H1dLiveAuthorityResult:
        runtime = self._load_h1d_runtime_bundle(request=request)
        document = runtime.document
        expected = runtime.expected_state
        authentication = runtime.expected_state_authentication
        if len(self._source_wire_payloads) != 5:
            raise RuntimeError("Task 11 H.1d source custody is incomplete")
        gpu_payload = self._source_wire_payloads[0]
        gpu_publication = gpu_payload["publication"]
        try:
            gpu_record = json.loads(
                base64.b64decode(
                    gpu_payload["raw_base64"],
                    validate=True,
                )[:-1].decode("ascii")
            )
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Task 11 H.1d spend source drifted") from exc
        source_values = {
            "key": gpu_publication["key"],
            "version_id": gpu_publication["version_id"],
            "file_sha256": gpu_publication["file_sha256"],
            "body_sha256": gpu_publication["body_sha256"],
            "observed_at": gpu_record["observed_at"],
        }
        live_observed_at = _rfc3339(_utc_now())

        def materialize(value: object) -> object:
            if type(value) is str and value == "$runtime.observed_at":
                return live_observed_at
            if (
                type(value) is str
                and value.startswith("$source.gpu_spend_snapshot.")
            ):
                field = value.rsplit(".", 1)[1]
                if field not in source_values:
                    raise RuntimeError(
                        "Task 11 H.1d spend placeholder drifted"
                    )
                return source_values[field]
            if type(value) is list:
                return [materialize(item) for item in value]
            if type(value) is dict:
                return {
                    key: materialize(item)
                    for key, item in value.items()
                }
            return value

        spend_request = spend_authority_request_from_mapping(
            materialize(document["spend_request_template"])
        )
        version_by_key = {
            spend_request.descriptor_key: (
                spend_request.descriptor_version_id
            ),
            spend_request.approval_key: spend_request.approval_version_id,
            spend_request.latest_key: spend_request.latest_version_id,
            spend_request.snapshot_key: spend_request.snapshot_version_id,
        }
        clients = self._adapters.clients.h1d_clients
        expiration = self._adapters.clients.credential_expiration
        if (
            type(clients) is not dict
            or type(expiration) is not str
            or "ec2" not in clients
        ):
            raise RuntimeError("Task 11 H.1d AWS clients are absent")
        worker_launches = self._adapters.ledger.query_activation_family(
            run_id=self._config.run_id,
            sort_key_prefix=(
                "ACTIVATION#" + self._config.activation_id + "#WORKER_LAUNCH#"
            ),
            record_type="glm52_production_worker_launch",
        )
        liability_settlements = (
            self._adapters.ledger.query_activation_family(
                run_id=self._config.run_id,
                sort_key_prefix=(
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#WORKER_LAUNCH_LIABILITY_SETTLEMENT#"
                ),
                record_type=(
                    "glm52_production_worker_launch_liability_settlement"
                ),
            )
        )
        worker_launch_liabilities = (
            self._adapters.ledger.query_activation_family(
                run_id=self._config.run_id,
                sort_key_prefix=(
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#WORKER_LAUNCH_LIABILITY#"
                ),
                record_type=(
                    "glm52_production_worker_launch_liability"
                ),
            )
        )
        terminal_sources = _read_retained_terminal_v2_sources(
            s3=self._adapters.clients.s3,
            bucket=self._config.campaign_bucket,
            settlements=liability_settlements,
        )
        known_launches = _retained_known_launch_authority(
            worker_launches, activation_id=self._config.activation_id
        )
        settled_launches = _retained_liability_settlement_authority(
            worker_launches,
            worker_launch_liabilities,
            liability_settlements,
            terminal_sources,
            activation_id=self._config.activation_id,
        )
        expected_launches = {
            identity: (tags, token)
            for identity, (state, tags, token) in known_launches.items()
            if state == "ALLOCATION_OPEN"
        }

        def invoke_probe(
            probe_request: Mapping[str, object],
        ) -> AttestationResult:
            self._attestation_sequence += 1
            value = self._invoke_exact(
                version_arn=self._config.attestation_version_arn,
                payload={
                    "schema_version": 1,
                    "record_type": "glm52_task11_attestation_request_v1",
                    "run_id": self._config.run_id,
                    "activation_id": self._config.activation_id,
                    "request_identity_sha256": canonical_sha256(
                        probe_request
                    ),
                    "freshness_nonce": secrets.token_hex(32),
                    "sequence": self._attestation_sequence,
                    "task9_deployed_identity_coordinate": asdict(
                        self._boundary.inputs[13]
                    ),
                    "task9_deployed_identity_sha256": (
                        self._boundary.inputs[13].body_sha256
                    ),
                },
            )
            return self._attestation_from_payload(value)

        sky_probe_request = document["sky_probe_request"]
        h1d_request = H1dLiveAuthorityRequest(
            profile="keep-gpu",
            account_id=self._config.account_id,
            region=self._config.region,
            run_id=self._config.run_id,
            activation_id=self._config.activation_id,
            expected_state_identity_sha256=(
                expected.canonical_identity_sha256
            ),
            spend_request=spend_request,
            sky_probe_request=sky_probe_request,
        )
        spend_token_custody: dict[tuple[str, int, int], str] = {}
        return inspect_h1d_live_authority(
            h1d_request,
            expected,
            H1dLiveServices(
                identity=Task11H1dIdentity(
                    caller_identity=(
                        self._adapters.clients.caller_identity
                    ),
                    credential_expiration=expiration,
                ),
                reader=Task11H1dReader(clients),
                spend=Task11SpendInspector(
                    request=spend_request,
                    services=SpendAuthorityServices(
                        object_store=Task11SpendObjectStore(
                            s3=self._adapters.clients.s3,
                            bucket=self._config.campaign_bucket,
                            version_by_key=version_by_key,
                        ),
                        ec2=Task11SpendEc2(
                            clients["ec2"],
                            activation_id=self._config.activation_id,
                            token_custody=spend_token_custody,
                            expected_launches=expected_launches,
                        ),
                        reserve_reader=Task11ReserveReader(
                            dynamodb=self._adapters.clients.dynamodb,
                            table_name=self._config.ledger_table_name,
                            token_custody=spend_token_custody,
                            activation_id=self._config.activation_id,
                            active_launches=expected_launches,
                            known_launches=known_launches,
                            settled_launches=settled_launches,
                        ),
                    ),
                    token_custody=spend_token_custody,
                    expected_launches=expected_launches,
                ),
                sky_relay_probe=Task11SkyProbe(
                    invoke_attestation=invoke_probe,
                    admission_identity_sha256=document[
                        "sky_probe_admission_identity_sha256"
                    ],
                ),
                clock=_utc_now,
                expected_state_authority=Task11ExpectedStateAuthority(
                    authentication
                ),
            ),
        )

    def reattest_sky_identity(
        self,
        *,
        request: object,
    ) -> AttestationResult:
        return self._attest(request=request)

    def recheck_all_authority(self, **kwargs: object) -> object:
        index, control = self._adapters.ledger.read_coherent(
            items=(
                (
                    LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        self._config.run_id,
                        "ACTIVATION#"
                        + self._config.activation_id
                        + "#CONTROL",
                    ),
                    "glm52_production_control",
                ),
            )
        )
        seal = getattr(self, "_seal", None)
        if type(seal) is not dict:
            raise RuntimeError("Task 11 seal custody is absent")
        seal_sk = (
            "ACTIVATION#"
            + self._config.activation_id
            + "#TASK11_DECISION_SEAL#"
            + f"{getattr(kwargs['request'], 'generation', 0):08d}"
        )
        sealed = self._adapters.clients.dynamodb.get_item(
            TableName=self._config.ledger_table_name,
            Key=encode_item(
                {
                    "PK": record_contract.ledger_pk(
                        self._config.run_id
                    ),
                    "SK": seal_sk,
                }
            ),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        sealed_item = (
            decode_item(sealed["Item"])
            if type(sealed) is dict
            and type(sealed.get("Item")) is dict
            else None
        )
        direct = kwargs["direct_response"]
        stored = self._direct_objects.get(
            direct.canonical_identity_sha256
        )
        if stored is None:
            raise RuntimeError("Task 11 direct decision custody is absent")
        decision_key, expected_raw = stored
        decision = self._adapters.clients.s3.get_object(
            Bucket=self._config.campaign_bucket,
            Key=decision_key,
            VersionId=direct.version_id,
            ExpectedBucketOwner=self._config.account_id,
            ChecksumMode="ENABLED",
        )
        stream = (
            decision.get("Body")
            if type(decision) is dict
            else None
        )
        read = getattr(stream, "read", None)
        if (
            index["current_activation_id"] != self._config.activation_id
            or control["activation_id"] != self._config.activation_id
            or index["campaign_identity_sha256"]
            != control["campaign_identity_sha256"]
            or control["active_epoch"] != seal["epoch"]
            or control["active_execution_arn"] != seal["execution_arn"]
            or control["barrier_nonce_sha256"]
            != seal["barrier_nonce_sha256"]
            or control["barrier_state"] != "ACQUIRED"
            or control["decision_seal_state"] != "SEALED"
            or control["revision"] != seal["control_revision"]
            or control["revision"]
            != seal["authorized_transition_to_revision"]
            or seal["authorized_transition_from_revision"] + 1
            != seal["authorized_transition_to_revision"]
            or control["fence_head_body_sha256"]
            != seal["fence_head_body_sha256"]
            or control["fence_head_version_id"]
            != seal["fence_head_version_id"]
            or type(sealed_item) is not dict
            or {
                key: value
                for key, value in sealed_item.items()
                if key not in {"PK", "SK"}
            }
            != seal
            or type(decision.get("ResponseMetadata"))
            is not dict
            or decision["ResponseMetadata"].get("HTTPStatusCode")
            != 200
            or decision.get("VersionId") != direct.version_id
            or not callable(read)
            or read(2 * 1024 * 1024 + 1) != expected_raw
        ):
            raise RuntimeError("Task 11 fresh authority recheck drifted")
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[15],
            operation="recheck_all_authority",
            evidence={
                "activation_index": index,
                "control": control,
                "seal": seal,
                "decision_transport": decision["ResponseMetadata"],
                "direct_decision_identity_sha256": kwargs[
                    "direct_response"
                ].canonical_identity_sha256,
                "live_h1d_identity_sha256": kwargs[
                    "task8_live"
                ].canonical_identity_sha256,
                "pre_attestation_identity_sha256": kwargs[
                    "pre_decision_attestation"
                ].canonical_identity_sha256,
                "repeated_attestation_identity_sha256": kwargs[
                    "repeated_attestation"
                ].canonical_identity_sha256,
            },
        )

    def _sky_action_request(self, request: object) -> object:
        written = self._effect_results.get("DecisionWriter")
        if type(written) is not EffectWriteResult:
            raise RuntimeError("Task 11 Sky action lacks direct decision")
        candidate = build_immutable_json_candidate(
            record_kind="generation-start-decision",
            bucket=written.object_identity.bucket,
            key=written.object_identity.key,
            raw=written.raw,
            activation_id=self._config.activation_id,
            generation=getattr(request, "generation", None),
        )
        return SimpleNamespace(
            action_key=self._boundary.action_key,
            operation_kind="SKY_POST",
            generation=getattr(request, "generation", None),
            candidate=candidate,
        )

    @staticmethod
    def _transaction_identity(
        value: object,
        *,
        operation: str,
    ) -> str:
        if (
            type(value) is not TransactionResolution
            or value.outcome is not WriteOutcome.EXACT_LIVE_OWNER_COMMIT
            or type(value.records) is not tuple
            or len(value.records) != 3
            or type(value.request_id) is not str
            or not value.request_id
        ):
            raise RuntimeError(
                "Task 11 " + operation + " Task 3 transaction failed"
            )
        return canonical_sha256(
            {
                "operation": operation,
                "outcome": value.outcome.value,
                "records": value.records,
                "request_id": value.request_id,
            }
        )

    def arm_sky_action(self, **kwargs: object) -> object:
        self._owner_nonce_sha256 = kwargs["owner_nonce_sha256"]
        if (
            hashlib.sha256(kwargs["owner_nonce"]).hexdigest()
            != self._owner_nonce_sha256
        ):
            raise RuntimeError("Task 11 arm owner nonce drifted")
        transaction = self._action_service.arm(
            request=self._sky_action_request(kwargs["request"]),
            raw_owner_nonce=kwargs["owner_nonce"],
        )
        identity = self._transaction_identity(
            transaction,
            operation="arm_sky_action",
        )
        self._transaction_records["arm_sky_action"] = transaction
        return build_step_receipt(
            step_name=CLOSURE_STEPS[16],
            operation_identity_sha256=identity,
            nonce_ownership_sha256=self._owner_nonce_sha256,
        )

    def consume_admission_reservation(self, **kwargs: object) -> object:
        if kwargs["owner_nonce_sha256"] != self._owner_nonce_sha256:
            raise RuntimeError("Task 11 arm owner drifted")
        self._decision_nonce_sha256 = kwargs["decision_nonce_sha256"]
        if (
            hashlib.sha256(kwargs["decision_nonce"]).hexdigest()
            != self._decision_nonce_sha256
        ):
            raise RuntimeError("Task 11 decision nonce drifted")
        written = self._effect_results.get("DecisionWriter")
        if type(written) is not EffectWriteResult:
            raise RuntimeError("Task 11 decision audit is absent")
        transaction = self._action_service.consume_audit(
            request=self._sky_action_request(kwargs["request"]),
            authority_audit_body_sha256=(
                written.authority_audit_body_sha256
            ),
            closing_revision=written.closing_revision,
            authorized_revision=written.authorized_revision,
        )
        identity = self._transaction_identity(
            transaction,
            operation="consume_admission_reservation",
        )
        self._transaction_records[
            "consume_admission_reservation"
        ] = transaction
        return build_step_receipt(
            step_name=CLOSURE_STEPS[17],
            operation_identity_sha256=identity,
            nonce_ownership_sha256=self._decision_nonce_sha256,
        )

    def coherent_readback_after_transaction(
        self,
        *,
        request: object,
        transaction_name: str,
        operation_identity_sha256: str,
        custody_nonce_sha256: str,
    ) -> object:
        del request, custody_nonce_sha256
        action_key = self._transaction_action_key(transaction_name)
        try:
            index, control, action = self._adapters.ledger.read_coherent(
                items=(
                    (
                        LedgerKey(
                            self._config.run_id,
                            "ACTIVATION_INDEX",
                        ),
                        "glm52_production_activation_index",
                    ),
                    (
                        LedgerKey(
                            self._config.run_id,
                            "ACTIVATION#"
                            + self._config.activation_id
                            + "#CONTROL",
                        ),
                        "glm52_production_control",
                    ),
                    (
                        LedgerKey(
                            self._config.run_id,
                            action_key,
                        ),
                        "glm52_production_action",
                    ),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Task 11 coherent Task 3 readback failed"
            ) from exc
        coherent = (
            index["current_activation_id"] == self._config.activation_id
            and control["activation_id"] == self._config.activation_id
            and action["activation_id"] == self._config.activation_id
            and index["campaign_identity_sha256"]
            == control["campaign_identity_sha256"]
            == action["campaign_identity_sha256"]
            and control["last_sky_post_action_key"] == action_key
            and action["state"] in {"CONSUMED", "POST_CLASSIFIED"}
        )
        transaction = self._transaction_records.get(transaction_name)
        if type(transaction) is TransactionResolution:
            coherent = coherent and transaction.records == (
                index,
                control,
                action,
            )
        expected_audit = None
        expected_closing = None
        expected_authorized = None
        source_index = {
            name: index
            for index, name in enumerate(_SOURCE_METHOD_KINDS)
        }.get(transaction_name)
        if source_index is not None:
            publication = source_publication_from_payload(
                self._source_wire_payloads[source_index],
                expected_source_kind=tuple(
                    _SOURCE_METHOD_KINDS.values()
                )[source_index],
                expected_predecessor_version_id=(
                    action.get("candidate_version_id")
                    or self._source_wire_payloads[source_index][
                        "publication"
                    ]["predecessor_version_id"]
                ),
            )
            expected_audit = publication.audit.audit_identity_sha256
            expected_closing = publication.audit.closing_revision
            expected_authorized = expected_closing + 1
        effect_kind = {
            "create_batch_successor": "FenceSuccessor",
            "create_or_recover_claim": "ClaimWriter",
            "create_direct_decision": "DecisionWriter",
            "persist_correlation_and_handoff": "ClosureHandoff",
        }.get(transaction_name)
        if effect_kind is not None:
            written = self._effect_results.get(effect_kind)
            if type(written) is EffectWriteResult:
                expected_audit = written.authority_audit_body_sha256
                expected_closing = written.closing_revision
                expected_authorized = written.authorized_revision
        if expected_audit is not None:
            coherent = coherent and (
                action.get("authority_audit_body_sha256")
                == expected_audit
                and action.get("authority_audit_closing_revision")
                == expected_closing
                and action.get("authorized_transition_to_revision")
                == expected_authorized
            )
        return build_transaction_readback(
            transaction_name=transaction_name,
            operation_identity_sha256=operation_identity_sha256,
            index_coherent=coherent,
            control_coherent=coherent,
            action_coherent=coherent,
        )

    def _transaction_action_key(self, transaction_name: str) -> str:
        cached = self._action_key_cache.get(transaction_name)
        if cached is not None:
            return cached
        mapping = {
            "publish_gpu_spend_snapshot": 0,
            "publish_submission_intent": 1,
            "publish_controller_baseline": 2,
            "publish_control_plane_readiness": 3,
            "publish_submission_acquisition": 4,
            "create_batch_successor": 5,
            "authorize_change_set_create": 6,
            "authorize_change_set_execute": 6,
            "create_or_recover_claim": 7,
            "create_direct_decision": 8,
            "persist_correlation_and_handoff": 12,
        }
        if transaction_name in {
            "arm_sky_action",
            "consume_admission_reservation",
            "invoke_launch_admission_version_once",
        }:
            action_key = self._boundary.action_key
        elif transaction_name in mapping:
            coordinate = exact_input_coordinate_from_mapping(
                asdict(self._boundary.inputs[mapping[transaction_name]]),
                expected_kind=self._boundary.inputs[
                    mapping[transaction_name]
                ].input_kind,
            )
            document = load_exact_input(
                s3=self._adapters.clients.s3,
                coordinate=coordinate,
            )
            if transaction_name.startswith("authorize_change_set_"):
                intent_field = (
                    "create_intent"
                    if transaction_name.endswith("_create")
                    else "execute_intent"
                )
                intent = document.get(intent_field)
                action_key = (
                    intent.get("action_key")
                    if type(intent) is dict
                    else None
                )
            else:
                action_key = document.get("action_key")
        else:
            raise RuntimeError(
                "Task 11 transaction action mapping is absent"
            )
        if type(action_key) is not str or not action_key:
            raise RuntimeError("Task 11 transaction action key drifted")
        self._action_key_cache[transaction_name] = action_key
        return action_key

    def coherent_readback_after_transactions(
        self,
        **kwargs: object,
    ) -> object:
        if (
            kwargs["owner_nonce_sha256"] != self._owner_nonce_sha256
            or kwargs["decision_nonce_sha256"]
            != self._decision_nonce_sha256
        ):
            raise RuntimeError("Task 11 coherent readback nonce drifted")
        index, control, action = self._adapters.ledger.read_coherent(
            items=(
                (
                    LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        self._config.run_id,
                        "ACTIVATION#"
                        + self._config.activation_id
                        + "#CONTROL",
                    ),
                    "glm52_production_control",
                ),
                (
                    LedgerKey(
                        self._config.run_id,
                        self._boundary.action_key,
                    ),
                    "glm52_production_action",
                ),
            )
        )
        if (
            action["state"] != "CONSUMED"
            or control["last_sky_post_action_key"]
            != self._boundary.action_key
        ):
            raise RuntimeError("Task 11 final Task 3 readback drifted")
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[18],
            operation="coherent_readback_after_transactions",
            evidence={
                "index": index,
                "control": control,
                "action": action,
            },
            nonce_ownership_sha256=None,
            coherent_readback=True,
        )

    @staticmethod
    def _audit_from_payload(value: Mapping[str, object]) -> object:
        return build_audit_evidence(
            kind=value["kind"],
            audit_identity_sha256=value["audit_identity_sha256"],
            invocation_identity_sha256=(
                value["invocation_identity_sha256"]
            ),
            closing_revision=value["closing_revision"],
        )

    def invoke_launch_admission_version_once(
        self,
        **kwargs: object,
    ) -> AdmissionClosureResult:
        if kwargs["version_arn"] != self._config.launch_admission_version_arn:
            raise RuntimeError("Task 11 admission version drifted")
        direct = kwargs["direct_response"]
        live = kwargs["task8_live"]
        attestation = kwargs["repeated_attestation"]
        seal = getattr(self, "_seal", None)
        barrier_transition = getattr(
            self,
            "_seal_transition_identity_sha256",
            None,
        )
        if (
            type(seal) is not dict
            or _SHA.fullmatch(
                seal.get("canonical_identity_sha256", "")
            )
            is None
            or _SHA.fullmatch(barrier_transition or "") is None
        ):
            raise RuntimeError(
                "Task 11 admission lacks seal/barrier custody"
            )
        payload = self._invoke_exact(
            version_arn=kwargs["version_arn"],
            payload={
                "schema_version": 1,
                "record_type": "glm52_task11_admission_request_v1",
                "run_id": self._config.run_id,
                "activation_id": self._config.activation_id,
                "generation": getattr(kwargs["request"], "generation"),
                "input_coordinate": asdict(self._boundary.inputs[11]),
                "decision_nonce_sha256": (
                    kwargs["decision_nonce_sha256"]
                ),
                "decision_nonce_base64": base64.b64encode(
                    kwargs["decision_nonce"]
                ).decode("ascii"),
                "direct_decision_identity_sha256": (
                    direct.canonical_identity_sha256
                ),
                "live_h1d_identity_sha256": (
                    live.canonical_identity_sha256
                ),
                "attestation_identity_sha256": (
                    attestation.canonical_identity_sha256
                ),
                "decision_seal_identity_sha256": seal[
                    "canonical_identity_sha256"
                ],
                "barrier_transition_identity_sha256": (
                    barrier_transition
                ),
                "task9_deployed_identity_coordinate": asdict(
                    self._boundary.inputs[13]
                ),
                "task9_deployed_identity_sha256": (
                    self._boundary.inputs[13].body_sha256
                ),
            },
        )
        try:
            task9 = AdmissionResult(**payload["task9_result"])
            audit = self._audit_from_payload(payload["post_audit"])
            return build_admission_closure_result(
                version_arn=kwargs["version_arn"],
                decision_nonce_sha256=kwargs["decision_nonce_sha256"],
                internal_steps=tuple(payload["internal_steps"]),
                action_states=tuple(payload["action_states"]),
                post_audit=audit,
                classification=payload["classification"],
                invocation_count=payload["invocation_count"],
                relay_call_count=payload["relay_call_count"],
                retry_count=payload["retry_count"],
                rearm_count=payload["rearm_count"],
                relay_receipt_sha256=payload["relay_receipt_sha256"],
                task9_result=task9,
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Task 11 admission response is invalid"
            ) from exc

    def audit_handoff_authority(self, **kwargs: object) -> object:
        from .task11_relay_runtime import load_task9_deployed_identity

        generation = getattr(kwargs["request"], "generation")
        admission = kwargs["admission"]
        decision_result = self._effect_results.get("DecisionWriter")
        index, control, action = self._adapters.ledger.read_coherent(
            items=(
                (
                    LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        self._config.run_id,
                        "ACTIVATION#"
                        + self._config.activation_id
                        + "#CONTROL",
                    ),
                    "glm52_production_control",
                ),
                (
                    LedgerKey(
                        self._config.run_id,
                        self._boundary.action_key,
                    ),
                    "glm52_production_action",
                ),
            )
        )
        if (
            index["current_activation_id"] != self._config.activation_id
            or control["activation_id"] != self._config.activation_id
            or action["activation_id"] != self._config.activation_id
            or not (
                index["campaign_identity_sha256"]
                == control["campaign_identity_sha256"]
                == action["campaign_identity_sha256"]
            )
            or control["last_sky_post_action_key"]
            != self._boundary.action_key
            or control["last_sky_post_state"] != "POST_CLASSIFIED"
            or action["state"] != "POST_CLASSIFIED"
        ):
            raise RuntimeError(
                "Task 11 classified handoff readback drifted"
            )
        production_authority = _load_task10_production_authority(
            s3=self._adapters.clients.s3,
            bucket=self._config.campaign_bucket,
        )
        task9 = load_task9_deployed_identity(
            s3=self._adapters.clients.s3,
            coordinate=asdict(self._boundary.inputs[13]),
            activation_id=self._config.activation_id,
            expected_body_sha256=self._boundary.inputs[13].body_sha256,
            expected_bucket=self._config.campaign_bucket,
        )
        runtime_bindings = _build_normal_task11_handoff_bindings(
            run_id=self._config.run_id,
            activation_id=self._config.activation_id,
            generation=generation,
            action_key=self._boundary.action_key,
            action=action,
            decision_result=decision_result,
            production_authority=production_authority,
            api_server_identity_sha256=task9.body_sha256,
            admission_classification=admission.classification,
            admission_request_id=admission.task9_result.request_id,
        )
        coordinate = exact_input_coordinate_from_mapping(
            asdict(self._boundary.inputs[12]),
            expected_kind="CORRELATION_HANDOFF_REQUEST",
        )
        document = _validate_effect_writer_input(
            load_exact_input(
                s3=self._adapters.clients.s3,
                coordinate=coordinate,
            ),
            writer_kind="ClosureHandoff",
            activation_id=self._config.activation_id,
            generation=getattr(kwargs["request"], "generation"),
        )
        candidate = prepare_effect_candidate(
            EffectWriterRequest(
                writer_kind="ClosureHandoff",
                activation_id=self._config.activation_id,
                generation=generation,
                action_key=document["action_key"],
                campaign_bucket=document["campaign_bucket"],
                builder_arguments=materialize_effect_builder_arguments(
                    document,
                    source_publications=list(self._source_wire_payloads),
                    dependency_results=list(self._effect_wire_payloads),
                    runtime_bindings=runtime_bindings,
                ),
            ),
        )
        action_request = S3CreateActionRequest(
            authority_domain="ACTIVATION",
            operation_kind="S3_CREATE",
            action_key=document["action_key"],
            candidate=candidate,
            activation_id=self._config.activation_id,
            generation=getattr(kwargs["request"], "generation"),
        )
        raw_owner_nonce = secrets.token_bytes(32)
        armed = self._action_service.arm(
            request=action_request,
            raw_owner_nonce=raw_owner_nonce,
        )
        arm_identity = self._transaction_identity(
            armed,
            operation="audit_handoff_authority.arm",
        )
        audit = self._adapters.fresh_h1f.fresh_audit(
            s3=self._adapters.clients.s3,
            request=H1fAuditRequest(
                operation_kind="S3_CREATE",
                action_key=document["action_key"],
                candidate=candidate,
            ),
        )
        control_rows = tuple(
            row
            for row in armed.records
            if type(row) is dict
            and row.get("record_type") == "glm52_production_control"
        )
        if (
            type(audit) is not H1fAuditResult
            or audit.action_key != document["action_key"]
            or audit.candidate_identity_sha256
            != candidate.candidate_identity_sha256
            or len(control_rows) != 1
            or control_rows[0]["revision"] != audit.closing_revision
        ):
            raise RuntimeError("Task 11 handoff H.1f audit drifted")
        self._handoff_prepared = {
            "runtime_bindings": runtime_bindings,
            "raw_owner_nonce_base64": base64.b64encode(
                raw_owner_nonce
            ).decode("ascii"),
            "audit": asdict(audit),
        }
        closure_audit = build_audit_evidence(
            kind=HANDOFF_AUDIT_KIND,
            audit_identity_sha256=audit.canonical_body_sha256,
            invocation_identity_sha256=arm_identity,
            closing_revision=audit.closing_revision,
        )
        return build_step_receipt(
            step_name=CLOSURE_STEPS[24],
            operation_identity_sha256=canonical_sha256(
                {
                    "operation": "audit_handoff_authority",
                    "action_key": document["action_key"],
                    "candidate_identity_sha256": (
                        candidate.candidate_identity_sha256
                    ),
                    "audit_identity_sha256": (
                        closure_audit.canonical_identity_sha256
                    ),
                }
            ),
            audits=(closure_audit,),
            coherent_readback=True,
        )

    def persist_correlation_and_handoff(self, **kwargs: object) -> object:
        prepared = getattr(self, "_handoff_prepared", None)
        if type(prepared) is not dict:
            raise RuntimeError("Task 11 handoff authority is absent")
        result = self._invoke_effect_writer(
            writer_kind="ClosureHandoff",
            version_arn=self._config.closure_handoff_version_arn,
            boundary_index=12,
            request=kwargs["request"],
            custody_nonce_sha256=canonical_sha256(
                {
                    "admission_identity_sha256": kwargs[
                        "admission"
                    ].canonical_identity_sha256,
                    "handoff_audit_identity_sha256": kwargs[
                        "handoff_audit"
                    ].canonical_identity_sha256,
                }
            ),
            runtime_bindings=prepared["runtime_bindings"],
            prepared_authority={
                "raw_owner_nonce_base64": prepared[
                    "raw_owner_nonce_base64"
                ],
                "audit": prepared["audit"],
            },
        )
        return self._effect_receipt(
            result=result,
            step_name=CLOSURE_STEPS[25],
        )

    def release_to_numeric_binding(self, **kwargs: object) -> object:
        if kwargs["version_arn"] != self._config.numeric_binding_version_arn:
            raise RuntimeError("Task 11 numeric binding version drifted")
        response = self._invoke_exact(
            version_arn=kwargs["version_arn"],
            payload={
                "schema_version": 1,
                "record_type": "glm52_task11_numeric_binding_request_v1",
                "run_id": self._config.run_id,
                "activation_id": self._config.activation_id,
                "admission_identity_sha256": (
                    kwargs["admission"].canonical_identity_sha256
                ),
                "task9_deployed_identity_coordinate": asdict(
                    self._boundary.inputs[13]
                ),
                "task9_deployed_identity_sha256": (
                    self._boundary.inputs[13].body_sha256
                ),
            },
        )
        expected = {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "activation_id",
            "admission_identity_sha256",
            "task9_deployed_identity_sha256",
            "binding_state",
            "numeric_job_id",
            "canonical_identity_sha256",
        }
        body = dict(response)
        identity = body.pop("canonical_identity_sha256", None)
        if (
            set(response) != expected
            or response["schema_version"] != 1
            or response["record_type"]
            != "glm52_task11_numeric_reconciliation_ack_v1"
            or response["account_id"] != self._config.account_id
            or response["region"] != self._config.region
            or response["run_id"] != self._config.run_id
            or response["activation_id"] != self._config.activation_id
            or response["admission_identity_sha256"]
            != kwargs["admission"].canonical_identity_sha256
            or response["task9_deployed_identity_sha256"]
            != self._boundary.inputs[13].body_sha256
            or response["binding_state"] != "RECONCILIATION_REQUIRED"
            or response["numeric_job_id"] is not None
            or identity != canonical_sha256(body)
        ):
            raise RuntimeError(
                "Task 11 numeric reconciliation ack drifted"
            )
        return self._authenticated_read_receipt(
            step_name=CLOSURE_STEPS[26],
            operation="release_to_numeric_binding",
            evidence={
                "request_identity_sha256": getattr(
                    kwargs["request"],
                    "canonical_identity_sha256",
                ),
                "numeric_reconciliation_ack": response,
            },
        )


def _exact(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeError(label + " is absent or invalid")
    return value


def _configuration() -> Task11ProductionConfig:
    value = Task11ProductionConfig(
        account_id=os.environ.get("GLM52_ACCOUNT_ID", ""),
        region=os.environ.get("AWS_REGION", ""),
        run_id=os.environ.get("GLM52_RUN_ID", ""),
        activation_id=os.environ.get("GLM52_ACTIVATION_ID", ""),
        ledger_table_name=os.environ.get(
            "GLM52_LEDGER_TABLE_NAME",
            "",
        ),
        campaign_bucket=os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
        model_bucket=os.environ.get("GLM52_MODEL_BUCKET", ""),
        model_prefix=os.environ.get("GLM52_MODEL_PREFIX", ""),
        fence_stack_id=os.environ.get("GLM52_FENCE_STACK_ID", ""),
        support_stack_id=os.environ.get("GLM52_SUPPORT_STACK_ID", ""),
        closure_role_arn=os.environ.get("GLM52_CLOSURE_ROLE_ARN", ""),
        attestation_version_arn=os.environ.get(
            "GLM52_ATTESTATION_VERSION_ARN",
            "",
        ),
        launch_admission_version_arn=os.environ.get(
            "GLM52_LAUNCH_ADMISSION_VERSION_ARN",
            "",
        ),
        numeric_binding_version_arn=os.environ.get(
            "GLM52_NUMERIC_BINDING_VERSION_ARN",
            "",
        ),
        source_gpu_spend_version_arn=os.environ.get(
            "GLM52_SOURCE_GPU_SPEND_VERSION_ARN",
            "",
        ),
        source_submission_intent_version_arn=os.environ.get(
            "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN",
            "",
        ),
        source_controller_baseline_version_arn=os.environ.get(
            "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN",
            "",
        ),
        source_control_plane_readiness_version_arn=os.environ.get(
            "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN",
            "",
        ),
        source_submission_acquisition_version_arn=os.environ.get(
            "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN",
            "",
        ),
        fence_executor_version_arn=os.environ.get(
            "GLM52_FENCE_EXECUTOR_VERSION_ARN",
            "",
        ),
        fence_successor_version_arn=os.environ.get(
            "GLM52_FENCE_SUCCESSOR_VERSION_ARN",
            "",
        ),
        claim_writer_version_arn=os.environ.get(
            "GLM52_CLAIM_WRITER_VERSION_ARN",
            "",
        ),
        decision_writer_version_arn=os.environ.get(
            "GLM52_DECISION_WRITER_VERSION_ARN",
            "",
        ),
        terminal_v1_writer_version_arn=os.environ.get(
            "GLM52_TERMINAL_V1_WRITER_VERSION_ARN",
            "",
        ),
        closure_handoff_version_arn=os.environ.get(
            "GLM52_CLOSURE_HANDOFF_VERSION_ARN",
            "",
        ),
        deployment_identity_sha256=os.environ.get(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        ),
        decision_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            + os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
            + ":"
            + os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
        ),
    )
    for field in (
        "activation_id",
        "ledger_table_name",
        "campaign_bucket",
        "model_bucket",
        "model_prefix",
        "fence_stack_id",
        "support_stack_id",
    ):
        _exact(getattr(value, field), "Task 11 " + field)
    if (
        value.account_id != "246813579024"
        or value.region != "us-west-2"
        or value.run_id != "glm52-sky-20260724"
        or _ROLE_ARN.fullmatch(value.closure_role_arn) is None
        or _VERSION_ARN.fullmatch(value.attestation_version_arn) is None
        or _VERSION_ARN.fullmatch(
            value.launch_admission_version_arn
        )
        is None
        or _VERSION_ARN.fullmatch(value.numeric_binding_version_arn)
        is None
        or any(
            _VERSION_ARN.fullmatch(version_arn) is None
            for version_arn in (
                value.source_gpu_spend_version_arn,
                value.source_submission_intent_version_arn,
                value.source_controller_baseline_version_arn,
                value.source_control_plane_readiness_version_arn,
                value.source_submission_acquisition_version_arn,
                value.fence_executor_version_arn,
                value.fence_successor_version_arn,
                value.claim_writer_version_arn,
                value.decision_writer_version_arn,
                value.terminal_v1_writer_version_arn,
                value.closure_handoff_version_arn,
            )
        )
        or _SHA.fullmatch(value.deployment_identity_sha256) is None
        or re.fullmatch(
            r"arn:aws:lambda:us-west-2:246813579024:function:"
            r"[A-Za-z0-9-_]{1,64}:[1-9][0-9]*",
            value.decision_function_version_arn,
        )
        is None
    ):
        raise RuntimeError(
            "Task 11 immutable production coordinates are incomplete"
        )
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _candidate_record(candidate: object) -> Mapping[str, object]:
    raw = getattr(candidate, "raw", None)
    if type(raw) is not bytes or not raw.endswith(b"\n"):
        raise RuntimeError("Task 11 candidate bytes are invalid")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Task 11 candidate is invalid JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise RuntimeError("Task 11 candidate is not canonical")
    return value


def _control_after(
    before: Mapping[str, object],
    *,
    action_key: str,
    state: str,
    observed_at: str,
) -> Mapping[str, object]:
    result = dict(before)
    result.update(
        {
            "last_sky_post_action_key": action_key,
            "last_sky_post_state": state,
            "revision": before["revision"] + 1,
            "updated_at": observed_at,
        }
    )
    return record_contract.validate_record(
        "glm52_production_control",
        result,
    )


def _armed_action(
    *,
    config: Task11ActionConfig,
    request: object,
    index: Mapping[str, object],
    control: Mapping[str, object],
    observed_at: str,
) -> Mapping[str, object]:
    candidate = getattr(request, "candidate", None)
    _candidate_record(candidate)
    generation = getattr(request, "generation", None)
    action_key = getattr(request, "action_key", None)
    attempt_match = (
        re.search(r"#([0-9]{8})$", action_key)
        if type(action_key) is str
        else None
    )
    attempt = (
        int(attempt_match.group(1))
        if attempt_match is not None
        else 1
    )
    fields = record_contract.RECORD_FIELDS["glm52_production_action"]
    value = {}
    for field in fields:
        if field == "schema_version":
            value[field] = 1
        elif field == "record_type":
            value[field] = "glm52_production_action"
        elif field == "run_id":
            value[field] = config.run_id
        elif field in record_contract._INT_FIELDS:
            value[field] = 1
        elif field in record_contract._TIMESTAMP_FIELDS:
            value[field] = observed_at
        elif field.endswith("_sha256"):
            value[field] = config.deployment_identity_sha256
        elif field == "generation_text":
            value[field] = f"{generation:08d}"
        else:
            value[field] = "exact"
    value.update(
        {
            "campaign_identity_sha256": index[
                "campaign_identity_sha256"
            ],
            "activation_id": config.activation_id,
            "activation_ordinal": index["current_activation_ordinal"],
            "generation": generation,
            "generation_text": f"{generation:08d}",
            "action_kind": getattr(request, "operation_kind", None),
            "attempt": attempt,
            "candidate_key": getattr(candidate, "key", None),
            "candidate_file_sha256": getattr(
                request,
                "candidate_file_sha256",
                getattr(candidate, "file_sha256", None),
            ),
            "candidate_body_sha256": getattr(
                candidate,
                "body_sha256",
                None,
            ),
            "request_body_sha256": getattr(
                request,
                "request_evidence_sha256",
                getattr(candidate, "file_sha256", None),
            ),
            "owner_epoch": control["active_epoch"],
            "owner_execution_arn": control["active_execution_arn"],
            "armed_by_epoch": control["active_epoch"],
            "armed_by_execution_arn": control["active_execution_arn"],
            "armed_by_state_machine_version_arn": control[
                "active_state_machine_version_arn"
            ],
            "armed_by_function_version_arn": (
                config.function_version_arn
            ),
            "arming_dispatch_identity_sha256": canonical_sha256(
                {
                    "action_key": action_key,
                    "candidate_identity_sha256": getattr(
                        candidate,
                        "candidate_identity_sha256",
                        None,
                    ),
                    "control_revision": control["revision"],
                }
            ),
            "barrier_nonce_sha256": control["barrier_nonce_sha256"],
            "state": "ARMED",
            "armed_at": observed_at,
            "arming_hard_expires_at": _rfc3339(
                datetime.fromisoformat(
                    observed_at.replace("Z", "+00:00")
                )
                + timedelta(seconds=720)
            ),
            "revision": 1,
        }
    )
    rule = record_contract._STATE_RULES[
        "glm52_production_action"
    ]["ARMED"]
    for field in rule["null"]:
        value[field] = None
    for field in rule["nonnull"]:
        if value[field] is None:
            value[field] = (
                config.deployment_identity_sha256
                if field.endswith("_sha256")
                else observed_at
            )
    for field, expected in rule["exact"].items():
        value[field] = expected
    return record_contract.validate_record(
        "glm52_production_action",
        value,
        sk=action_key,
    )


class Task11DynamoActionService:
    """Translate accepted Task 4 action calls to the Task 3 ledger adapter."""

    def __init__(
        self,
        *,
        config: Task11ActionConfig,
        ledger: DynamoLedgerAdapter,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._owner_nonces = {}

    def _read(
        self,
        *,
        action_key: str = "",
    ) -> tuple[Mapping[str, object], ...]:
        rows = [
            (
                LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                "glm52_production_activation_index",
            ),
            (
                LedgerKey(
                    self._config.run_id,
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#CONTROL",
                ),
                "glm52_production_control",
            ),
        ]
        if action_key:
            rows.append(
                (
                    LedgerKey(self._config.run_id, action_key),
                    "glm52_production_action",
                )
            )
        return self._ledger.read_coherent(items=tuple(rows))

    def adopt_owner(
        self,
        *,
        action_key: str,
        raw_owner_nonce: bytes,
    ) -> None:
        """Adopt an invocation-private owner prepared by the decision Lambda."""

        if (
            type(action_key) is not str
            or not action_key
            or type(raw_owner_nonce) is not bytes
            or len(raw_owner_nonce) < 16
        ):
            raise RuntimeError("Task 4 prepared action owner is invalid")
        self._owner_nonces[action_key] = raw_owner_nonce

    def arm(
        self,
        *,
        request: object,
        raw_owner_nonce: Optional[bytes] = None,
    ) -> object:
        action_key = getattr(request, "action_key", None)
        if type(action_key) is not str or not action_key:
            raise RuntimeError("Task 4 action key is invalid")
        action_request = request
        if (
            getattr(request, "candidate", None) is None
            and getattr(request, "selected_successor", None) is not None
        ):
            selected = request.selected_successor
            candidate_body = {
                "key": selected.key,
                "version_id": selected.version_id,
                "body_sha256": selected.body_sha256,
            }
            action_request = SimpleNamespace(
                authority_domain=request.authority_domain,
                operation_kind=request.operation_kind,
                action_key=request.action_key,
                activation_id=request.activation_id,
                generation=request.generation,
                candidate=SimpleNamespace(
                    raw=canonical_json_bytes(candidate_body) + b"\n",
                    key=selected.key,
                    file_sha256=(
                        request.selected_successor_file_sha256
                    ),
                    body_sha256=selected.body_sha256,
                    candidate_identity_sha256=(
                        request.candidate_identity_sha256
                    ),
                ),
                candidate_file_sha256=(
                    request.selected_successor_file_sha256
                ),
                request_evidence_sha256=(
                    request.request_evidence_sha256
                ),
            )
        index, control = self._read()
        observed_at = _rfc3339(_utc_now())
        armed = _armed_action(
            config=self._config,
            request=action_request,
            index=index,
            control=control,
            observed_at=observed_at,
        )
        adopted_nonce = self._owner_nonces.get(action_key)
        raw_nonce = (
            raw_owner_nonce
            if raw_owner_nonce is not None
            else (
                adopted_nonce
                if adopted_nonce is not None
                else secrets.token_bytes(32)
            )
        )
        if type(raw_nonce) is not bytes or len(raw_nonce) < 16:
            raise RuntimeError("Task 4 action owner nonce is invalid")
        result = self._ledger.arm_action(
            index=ExactCheck(
                LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                index,
            ),
            control=ExactUpdate(
                LedgerKey(
                    self._config.run_id,
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#CONTROL",
                ),
                control,
                _control_after(
                    control,
                    action_key=action_key,
                    state="ARMED",
                    observed_at=observed_at,
                ),
            ),
            action=ExactPut(
                LedgerKey(self._config.run_id, action_key),
                armed,
            ),
            domain="ACTIVATION",
            operation_identity_sha256=canonical_sha256(
                {
                    "operation": "ARM",
                    "action_key": action_key,
                    "candidate": getattr(
                        action_request.candidate,
                        "candidate_identity_sha256",
                    ),
                }
            ),
            raw_owner_nonce=raw_nonce,
        )
        self._owner_nonces[action_key] = raw_nonce
        return result

    def consume_audit(
        self,
        *,
        request: object,
        authority_audit_body_sha256: str,
        closing_revision: int,
        authorized_revision: int,
    ) -> object:
        action_key = getattr(request, "action_key", None)
        if (
            type(action_key) is not str
            or action_key not in self._owner_nonces
        ):
            raise RuntimeError("Task 4 action owner is absent")
        index, control, action = self._read(action_key=action_key)
        observed_at = _rfc3339(_utc_now())
        consumed = dict(action)
        consumed.update(
            {
                "state": "CONSUMED",
                "owner_invocation_nonce_sha256": hashlib.sha256(
                    self._owner_nonces[action_key]
                ).hexdigest(),
                "authority_audit_body_sha256": (
                    authority_audit_body_sha256
                ),
                "authority_audit_closing_revision": closing_revision,
                "authorized_transition_from_revision": closing_revision,
                "authorized_transition_to_revision": authorized_revision,
                "consumed_at": observed_at,
                "revision": action["revision"] + 1,
            }
        )
        consumed = record_contract.validate_record(
            "glm52_production_action",
            consumed,
            sk=action_key,
        )
        return self._ledger.consume_action(
            index=ExactCheck(
                LedgerKey(self._config.run_id, "ACTIVATION_INDEX"),
                index,
            ),
            control=ExactUpdate(
                LedgerKey(
                    self._config.run_id,
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#CONTROL",
                ),
                control,
                _control_after(
                    control,
                    action_key=action_key,
                    state="CONSUMED",
                    observed_at=observed_at,
                ),
            ),
            action=ExactUpdate(
                LedgerKey(self._config.run_id, action_key),
                action,
                consumed,
            ),
            domain="ACTIVATION",
            operation_identity_sha256=canonical_sha256(
                {
                    "operation": "CONSUME",
                    "action_key": action_key,
                    "audit": authority_audit_body_sha256,
                }
            ),
            raw_owner_nonce=self._owner_nonces[action_key],
        )


def _canonical_live_value(value: object) -> object:
    if isinstance(value, datetime):
        return _rfc3339(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if type(value) is dict:
        return {
            str(key): _canonical_live_value(item)
            for key, item in value.items()
            if key != "ResponseMetadata"
        }
    if type(value) in {list, tuple}:
        return [_canonical_live_value(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise RuntimeError(
        "Task 11 live response contains a noncanonical value"
    )


def _select_path(value: object, path: object) -> object:
    if type(path) is not list:
        raise RuntimeError("Task 11 live response path is not closed")
    current = value
    for part in path:
        if type(part) is str and type(current) is dict:
            current = current[part]
        elif type(part) is int and type(current) is list:
            current = current[part]
        else:
            raise RuntimeError("Task 11 live response path drifted")
    return current


_H1D_METHODS = {
    "cloudformation": {
        "describe_stacks",
        "list_stack_resources",
        "get_template",
    },
    "lambda": {
        "get_function",
        "get_policy",
        "list_event_source_mappings",
        "list_functions",
    },
    "iam": {
        "get_role",
        "get_instance_profile",
        "get_role_policy",
        "get_policy",
        "get_policy_version",
        "list_attached_role_policies",
        "list_instance_profiles",
        "list_instance_profiles_for_role",
        "list_role_policies",
        "list_roles",
    },
    "eventbridge": {"list_rules", "list_targets_by_rule", "describe_rule"},
    "scheduler": {"list_schedules", "get_schedule"},
    "sqs": {"get_queue_attributes", "list_queues"},
    "sns": {"list_subscriptions_by_topic", "get_subscription_attributes"},
    "ec2": {
        "describe_iam_instance_profile_associations",
        "describe_instances",
    },
    "stepfunctions": {
        "describe_state_machine",
        "list_state_machine_versions",
        "list_state_machines",
    },
    "s3": {
        "get_bucket_versioning",
        "get_bucket_policy",
        "get_bucket_lifecycle_configuration",
        "get_bucket_replication",
        "list_object_versions",
    },
    "dynamodb": {"get_item", "transact_get_items", "query"},
    "ssm": {"describe_instance_information"},
    "cloudwatch": {"describe_alarms", "get_metric_data"},
    "logs": {"describe_log_groups"},
}


class Task11H1dReader:
    """Concurrent, allowlisted AWS reader for all 13 exact live families."""

    def __init__(self, clients: Mapping[str, object]) -> None:
        self._clients = dict(clients)
        self._pages: dict[tuple[str, str], LiveReadPage] = {}

    def _call_pages(
        self,
        *,
        family: str,
        call: Mapping[str, object],
    ) -> tuple[list[Mapping[str, object]], list[str], list[str]]:
        if (
            type(call) is not dict
            or set(call)
            != {
                "method",
                "request",
                "items_path",
                "field_paths",
                "paginate",
            }
            or call["method"] not in _H1D_METHODS[family]
            or type(call["request"]) is not dict
            or type(call["field_paths"]) is not dict
            or type(call["paginate"]) is not bool
        ):
            raise RuntimeError("Task 11 live read call is not allowlisted")
        client = self._clients.get(family)
        method = getattr(client, call["method"], None)
        if not callable(method):
            raise RuntimeError("Task 11 live client method is absent")
        if call["paginate"]:
            paginator_method = getattr(client, "get_paginator", None)
            if not callable(paginator_method):
                raise RuntimeError("Task 11 live paginator is absent")
            responses = paginator_method(call["method"]).paginate(
                **call["request"]
            )
        else:
            responses = (method(**call["request"]),)
        items: list[Mapping[str, object]] = []
        request_ids: list[str] = []
        response_identities: list[str] = []
        seen_page_identities = set()
        seen_markers = set()
        for page_index, response in enumerate(responses):
            if page_index >= 64:
                raise RuntimeError(
                    "Task 11 live pagination exceeded bound"
                )
            metadata = (
                response.get("ResponseMetadata")
                if type(response) is dict
                else None
            )
            request_id = (
                metadata.get("RequestId")
                if type(metadata) is dict
                else None
            )
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 200
                or type(request_id) is not str
                or not request_id
            ):
                raise RuntimeError(
                    "Task 11 live AWS response is unauthenticated"
                )
            if (
                "IsTruncated" in response
                and type(response["IsTruncated"]) is not bool
            ):
                raise RuntimeError(
                    "Task 11 live pagination flag drifted"
                )
            page_identity = canonical_sha256(
                _canonical_live_value(response)
            )
            if page_identity in seen_page_identities:
                raise RuntimeError(
                    "Task 11 live pagination page repeated"
                )
            seen_page_identities.add(page_identity)
            response_identities.append(page_identity)
            marker = None
            for marker_field in (
                "NextToken",
                "NextMarker",
                "NextContinuationToken",
                "Marker",
            ):
                if marker_field in response:
                    marker = (marker_field, response[marker_field])
                    break
            if marker is not None:
                if (
                    type(marker[1]) is not str
                    or not marker[1]
                    or marker in seen_markers
                ):
                    raise RuntimeError(
                        "Task 11 live pagination marker repeated"
                    )
                seen_markers.add(marker)
            request_ids.append(request_id)
            selected = _select_path(response, call["items_path"])
            rows = selected if type(selected) is list else [selected]
            for row in rows:
                projection = {
                    name: _canonical_live_value(
                        _select_path(row, path)
                    )
                    for name, path in call["field_paths"].items()
                }
                items.append(projection)
        if not response_identities:
            raise RuntimeError("Task 11 live pagination returned no page")
        return items, request_ids, response_identities

    def _read_spec(self, spec: LiveReadSpec) -> LiveReadPage:
        parameters = spec.parameters
        if (
            type(parameters) is not dict
            or set(parameters) != {"calls"}
            or type(parameters["calls"]) is not list
            or not parameters["calls"]
        ):
            raise RuntimeError("Task 11 live read parameters drifted")
        items: list[Mapping[str, object]] = []
        request_ids: list[str] = []
        response_identities: list[str] = []
        for call in parameters["calls"]:
            call_items, call_ids, call_responses = self._call_pages(
                family=spec.family,
                call=call,
            )
            items.extend(call_items)
            request_ids.extend(call_ids)
            response_identities.extend(call_responses)
        if not request_ids:
            raise RuntimeError("Task 11 live read made no AWS request")
        return LiveReadPage(
            family=spec.family,
            operation=spec.operation,
            request_token=None,
            page_index=0,
            items=tuple(items),
            next_token=None,
            request_id=request_ids[0],
            service_request_ids=tuple(request_ids),
            service_response_identities=tuple(response_identities),
            observed_at=_rfc3339(_utc_now()),
        )

    def prepare(
        self,
        specs: tuple[LiveReadSpec, ...],
        *,
        deadline: datetime,
    ) -> None:
        remaining = (deadline - _utc_now()).total_seconds()
        if remaining <= 0:
            raise ValueError("Task 11 live read deadline elapsed")
        executor = ThreadPoolExecutor(max_workers=len(specs))
        futures = {}
        try:
            futures = {
                (spec.family, spec.operation): executor.submit(
                    self._read_spec, spec
                )
                for spec in specs
            }
            pages = {}
            for spec_identity, future in futures.items():
                remaining = (deadline - _utc_now()).total_seconds()
                if remaining <= 0:
                    raise FutureTimeout()
                pages[spec_identity] = future.result(timeout=remaining)
            if _utc_now() > deadline:
                raise FutureTimeout()
            self._pages = pages
        except FutureTimeout as exc:
            for future in futures.values():
                future.cancel()
            raise ValueError(
                "Task 11 concurrent live read exceeded deadline"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def read_page(
        self,
        *,
        spec: LiveReadSpec,
        continuation_token: Optional[str],
    ) -> LiveReadPage:
        key = (spec.family, spec.operation)
        if continuation_token is not None or key not in self._pages:
            raise RuntimeError("Task 11 live read page custody drifted")
        return self._pages[key]


class Task11H1dIdentity:
    def __init__(
        self,
        *,
        caller_identity: Mapping[str, object],
        credential_expiration: str,
    ) -> None:
        self._caller_identity = dict(caller_identity)
        self._expiration = credential_expiration

    def get_caller_identity(self) -> CallerIdentityObservation:
        response = self._caller_identity
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
        ):
            raise RuntimeError("Task 11 H.1d STS response drifted")
        return CallerIdentityObservation(
            account_id=response["Account"],
            arn=response["Arn"],
            user_id=response["UserId"],
            credential_expiration=self._expiration,
            request_id=metadata["RequestId"],
            observed_at=_rfc3339(_utc_now()),
        )


class Task11ExpectedStateAuthority:
    def __init__(self, value: ExpectedStateAuthentication) -> None:
        self._value = value

    def authenticate(self, expected: object) -> ExpectedStateAuthentication:
        del expected
        return self._value


class Task11SpendObjectStore:
    _MAX_VERSION_PAGES = 64
    _LIST_RESPONSE_FIELDS = frozenset(
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
    _GET_RESPONSE_FIELDS = frozenset(
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
    _RESPONSE_METADATA_FIELDS = frozenset(
        {
            "RequestId",
            "HostId",
            "HTTPStatusCode",
            "HTTPHeaders",
            "RetryAttempts",
        }
    )

    def __init__(
        self,
        *,
        s3: object,
        bucket: str,
        version_by_key: Mapping[str, str],
    ) -> None:
        self._s3 = s3
        self._bucket = bucket
        self._versions = dict(version_by_key)
        self._pagination: dict[str, dict[str, object]] = {}

    def get_object(self, *, key: str) -> SpendObject:
        version_id = self._versions.get(key)
        if type(version_id) is not str or not version_id:
            raise RuntimeError("Task 11 spend object version is absent")
        response = self._s3.get_object(
            Bucket=self._bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner="246813579024",
            ChecksumMode="ENABLED",
        )
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        stream = response.get("Body") if type(response) is dict else None
        read = getattr(stream, "read", None)
        if (
            type(response) is not dict
            or not {
                "Body",
                "VersionId",
                "ContentLength",
                "ChecksumSHA256",
                "ETag",
                "ResponseMetadata",
            }.issubset(response)
            or not set(response).issubset(self._GET_RESPONSE_FIELDS)
            or type(metadata) is not dict
            or not {
                "RequestId",
                "HTTPStatusCode",
                "RetryAttempts",
            }.issubset(metadata)
            or not set(metadata).issubset(
                self._RESPONSE_METADATA_FIELDS
            )
            or type(metadata["HTTPStatusCode"]) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata["RequestId"]) is not str
            or not metadata["RequestId"]
            or type(metadata["RetryAttempts"]) is not int
            or metadata["RetryAttempts"] != 0
            or response.get("VersionId") != version_id
            or type(response.get("ContentLength")) is not int
            or response["ContentLength"] < 0
            or type(response.get("ChecksumSHA256")) is not str
            or not response["ChecksumSHA256"]
            or type(response.get("ETag")) is not str
            or not response["ETag"]
            or not callable(read)
        ):
            raise RuntimeError("Task 11 spend object read drifted")
        raw = read(2 * 1024 * 1024 + 1)
        if type(raw) is not bytes or len(raw) > 2 * 1024 * 1024:
            raise RuntimeError("Task 11 spend object is unbounded")
        try:
            service_checksum = base64.b64decode(
                response["ChecksumSHA256"],
                validate=True,
            )
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                "Task 11 spend object checksum drifted"
            ) from exc
        raw_digest = hashlib.sha256(raw).digest()
        if (
            response["ContentLength"] != len(raw)
            or len(service_checksum) != 32
            or service_checksum != raw_digest
            or base64.b64encode(service_checksum).decode("ascii")
            != response["ChecksumSHA256"]
        ):
            raise RuntimeError(
                "Task 11 spend object service evidence drifted"
            )
        return SpendObject(
            key=key,
            raw=raw,
            version_id=version_id,
            etag=response["ETag"],
            checksum_sha256=service_checksum.hex(),
            request_id=metadata["RequestId"],
            observed_at=_rfc3339(_utc_now()),
        )

    @staticmethod
    def _token(value: object) -> Optional[str]:
        if value is None:
            return None
        return base64.b64encode(
            canonical_json_bytes(value)
        ).decode("ascii")

    @staticmethod
    def _decode_token(value: Optional[str]) -> Mapping[str, object]:
        if value is None:
            return {}
        try:
            parsed = json.loads(
                base64.b64decode(value, validate=True).decode("ascii")
            )
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Task 11 spend list token drifted") from exc
        if (
            type(parsed) is not dict
            or set(parsed) != {"KeyMarker", "VersionIdMarker"}
            or type(parsed["KeyMarker"]) is not str
            or not parsed["KeyMarker"]
            or type(parsed["VersionIdMarker"]) is not str
            or not parsed["VersionIdMarker"]
            or base64.b64encode(
                canonical_json_bytes(parsed)
            ).decode("ascii")
            != value
        ):
            raise RuntimeError("Task 11 spend list token is not closed")
        return parsed

    def list_namespace(
        self,
        *,
        prefix: str,
        continuation_token: Optional[str],
    ) -> SpendListPage:
        markers = self._decode_token(continuation_token)
        state = self._pagination.get(prefix)
        if continuation_token is None:
            if state is not None:
                raise RuntimeError(
                    "Task 11 spend pagination restarted before completion"
                )
            state = {
                "expected_token": None,
                "seen_markers": set(),
                "seen_versions": set(),
                "seen_keys": set(),
                "pages": 0,
            }
            self._pagination[prefix] = state
        elif (
            state is None
            or state["expected_token"] != continuation_token
        ):
            raise RuntimeError("Task 11 spend list token custody drifted")
        if int(state["pages"]) >= self._MAX_VERSION_PAGES:
            raise RuntimeError(
                "Task 11 spend version pagination exceeded bound"
            )
        request = {
            "Bucket": self._bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": "246813579024",
            **markers,
        }
        response = self._s3.list_object_versions(**request)
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        if (
            type(response) is not dict
            or not {
                "Versions",
                "DeleteMarkers",
                "IsTruncated",
                "Name",
                "Prefix",
                "MaxKeys",
                "ResponseMetadata",
            }.issubset(response)
            or not set(response).issubset(self._LIST_RESPONSE_FIELDS)
            or response["Name"] != self._bucket
            or response["Prefix"] != prefix
            or type(response["MaxKeys"]) is not int
            or response["MaxKeys"] != 1000
            or type(response["IsTruncated"]) is not bool
            or type(response["Versions"]) is not list
            or type(response["DeleteMarkers"]) is not list
            or type(metadata) is not dict
            or not {
                "RequestId",
                "HTTPStatusCode",
                "RetryAttempts",
            }.issubset(metadata)
            or not set(metadata).issubset(
                self._RESPONSE_METADATA_FIELDS
            )
            or type(metadata["HTTPStatusCode"]) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata["RequestId"]) is not str
            or not metadata["RequestId"]
            or type(metadata["RetryAttempts"]) is not int
            or metadata["RetryAttempts"] != 0
        ):
            raise RuntimeError("Task 11 spend namespace read drifted")
        if continuation_token is None:
            if (
                "KeyMarker" in response
                or "VersionIdMarker" in response
            ):
                raise RuntimeError(
                    "Task 11 spend page marker echo drifted"
                )
        elif (
            response.get("KeyMarker") != markers["KeyMarker"]
            or response.get("VersionIdMarker")
            != markers["VersionIdMarker"]
        ):
            raise RuntimeError("Task 11 spend page marker echo drifted")
        if response.get("DeleteMarkers"):
            raise RuntimeError("Task 11 spend namespace is delete-marked")
        versions = []
        for row in response["Versions"]:
            if (
                type(row) is not dict
                or not {
                    "Key",
                    "VersionId",
                    "IsLatest",
                    "Size",
                }.issubset(row)
                or not set(row).issubset(self._VERSION_FIELDS)
                or type(row["Key"]) is not str
                or not row["Key"].startswith(prefix)
                or type(row["VersionId"]) is not str
                or not row["VersionId"]
                or row["IsLatest"] is not True
                or type(row["Size"]) is not int
                or row["Size"] < 0
            ):
                raise RuntimeError(
                    "Task 11 spend version row drifted"
                )
            identity = (row["Key"], row["VersionId"])
            if (
                identity in state["seen_versions"]
                or row["Key"] in state["seen_keys"]
            ):
                raise RuntimeError(
                    "Task 11 spend version row was duplicated"
                )
            state["seen_versions"].add(identity)
            state["seen_keys"].add(row["Key"])
            self._versions[row["Key"]] = row["VersionId"]
            versions.append(row)
        state["pages"] = int(state["pages"]) + 1
        next_value = None
        if response["IsTruncated"] is True:
            next_key = response.get("NextKeyMarker")
            next_version = response.get("NextVersionIdMarker")
            next_marker = (next_key, next_version)
            current_marker = (
                (
                    markers["KeyMarker"],
                    markers["VersionIdMarker"],
                )
                if markers
                else None
            )
            if (
                type(next_key) is not str
                or not next_key
                or type(next_version) is not str
                or not next_version
                or next_marker == current_marker
                or next_marker in state["seen_markers"]
                or int(state["pages"]) >= self._MAX_VERSION_PAGES
            ):
                raise RuntimeError(
                    "Task 11 spend list marker did not progress"
                )
            state["seen_markers"].add(next_marker)
            next_value = {
                "KeyMarker": next_key,
                "VersionIdMarker": next_version,
            }
            state["expected_token"] = self._token(next_value)
        else:
            if (
                "NextKeyMarker" in response
                or "NextVersionIdMarker" in response
            ):
                raise RuntimeError(
                    "Task 11 spend terminal page markers drifted"
                )
            self._pagination.pop(prefix, None)
        return SpendListPage(
            keys=tuple(row["Key"] for row in versions),
            version_ids=tuple(row["VersionId"] for row in versions),
            next_token=(
                state["expected_token"]
                if next_value is not None
                else None
            ),
            request_id=metadata["RequestId"],
            observed_at=_rfc3339(_utc_now()),
        )


def _bind_spend_client_token(
    custody: dict[tuple[str, int, int], object],
    *,
    allocation_identity: tuple[str, int, int],
    client_token: str,
    source: str,
) -> None:
    existing = custody.get(allocation_identity)
    if existing is None:
        existing = {}
        custody[allocation_identity] = existing
    if type(existing) is not dict or (
        source in existing and existing[source] != client_token
    ) or (
        existing and any(token != client_token for token in existing.values())
    ):
        raise RuntimeError(
            "Task 11 spend ClientToken custody drifted"
        )
    for identity, observations in custody.items():
        if (
            identity != allocation_identity
            and type(observations) is dict
            and client_token in observations.values()
        ):
            raise RuntimeError(
                "Task 11 spend ClientToken identity was reused"
            )
    existing[source] = client_token


def _finalize_spend_token_custody(
    custody: Mapping[tuple[str, int, int], object],
    expected: Mapping[tuple[str, int, int], tuple[Mapping[str, str], str]],
) -> None:
    if set(custody) != set(expected):
        raise RuntimeError("Task 11 spend ClientToken custody is incomplete")
    for identity, expected_value in expected.items():
        observed = custody.get(identity)
        if type(observed) is not dict or observed != {"ec2": expected_value[1], "reserve": expected_value[1]}:
            raise RuntimeError("Task 11 spend ClientToken custody is one-sided")


def _retained_known_launch_authority(
    records: tuple[Mapping[str, object], ...], *, activation_id: str
) -> dict[tuple[str, int, int], tuple[str, dict[str, str], str]]:
    """Materialize exact worker-launch authority, including terminal history."""
    expected: dict[
        tuple[str, int, int], tuple[str, dict[str, str], str]
    ] = {}
    for record in records:
        identity = (activation_id, record["generation"], record["allocation_ordinal"])
        state = record.get("state")
        if (
            type(record.get("generation")) is not int
            or type(record.get("allocation_ordinal")) is not int
            or record.get("activation_id") != activation_id
            or identity in expected
            or type(state) is not str
        ):
            raise RuntimeError("Task 11 retained worker-launch authority drifted")
        tags = {
            "Project": "KEEP", "Campaign": "GLM-5.2", "RunId": "glm52-sky-20260724", "Market": "on-demand",
            "campaign-identity-sha256": record["campaign_identity_sha256"],
            "activation-id": activation_id,
            "activation-ordinal-text": "%08d" % record["activation_ordinal"],
            "generation-text": "%08d" % record["generation"],
            "allocation-ordinal-text": "%08d" % record["allocation_ordinal"],
            "action-key": record["sky_action_key"], "sky-request-id": record["sky_request_id"],
            "sky-job-name": record["sky_job_name"], "sky-task-name": "glm52-production",
            "task-yaml-sha256": record["task_yaml_sha256"], "request-body-sha256": record["request_body_sha256"],
        }
        if (
            any(type(value) is not str or not value for value in tags.values())
            or canonical_sha256(tags) != record.get("expected_worker_tags_sha256")
            or type(record.get("ec2_client_token")) is not str
            or _SHA.fullmatch(record["ec2_client_token"]) is None
        ):
            raise RuntimeError("Task 11 retained worker-launch tags drifted")
        expected[identity] = (state, tags, record["ec2_client_token"])
    return expected


def _retained_launch_authority(
    records: tuple[Mapping[str, object], ...], *, activation_id: str
) -> dict[tuple[str, int, int], tuple[dict[str, str], str]]:
    """Return the only retained allocations allowed in live spend evidence."""
    return {
        identity: (tags, token)
        for identity, (state, tags, token) in _retained_known_launch_authority(
            records, activation_id=activation_id
        ).items()
        if state == "ALLOCATION_OPEN"
    }


def _retained_liability_settlement_authority(
    worker_launches: tuple[Mapping[str, object], ...],
    liabilities: tuple[Mapping[str, object], ...],
    settlements: tuple[Mapping[str, object], ...],
    terminal_sources: Mapping[
        tuple[str, str], Mapping[str, object]
    ],
    *,
    activation_id: str,
) -> dict[tuple[str, int, int], tuple[str, str, str, str]]:
    """Authenticate the exact settled reserve releases for retained launches."""
    launches: dict[
        tuple[str, int, int], Mapping[str, object]
    ] = {}
    for launch in worker_launches:
        identity = (
            activation_id,
            launch.get("generation"),
            launch.get("allocation_ordinal"),
        )
        if (
            type(launch) is not dict
            or launch.get("activation_id") != activation_id
            or type(identity[1]) is not int
            or type(identity[2]) is not int
            or identity in launches
            or type(launch.get("canonical_body_sha256")) is not str
            or launch["canonical_body_sha256"]
            != record_contract.canonical_record_identity_unchecked(launch)
            or type(
                launch.get(
                    "gpu_liability_reserve_ledger_identity_sha256"
                )
            )
            is not str
            or _SHA.fullmatch(
                launch["gpu_liability_reserve_ledger_identity_sha256"]
            )
            is None
        ):
            raise RuntimeError(
                "Task 11 retained settlement launch authority drifted"
            )
        launches[identity] = launch

    liability_by_identity: dict[
        tuple[str, int, int], Mapping[str, object]
    ] = {}
    for liability in liabilities:
        identity = (
            activation_id,
            liability.get("generation"),
            liability.get("allocation_ordinal"),
        )
        launch = launches.get(identity)
        if (
            type(liability) is not dict
            or launch is None
            or liability.get("record_type")
            != "glm52_production_worker_launch_liability"
            or liability.get("activation_id") != activation_id
            or liability.get("activation_ordinal")
            != launch.get("activation_ordinal")
            or liability.get("worker_launch_identity_sha256")
            != launch.get("canonical_body_sha256")
            or liability.get("ec2_client_token")
            != launch.get("ec2_client_token")
            or liability.get(
                "gpu_liability_reserve_ledger_identity_sha256"
            )
            != launch.get(
                "gpu_liability_reserve_ledger_identity_sha256"
            )
            or liability.get("canonical_body_sha256")
            != record_contract.canonical_record_identity_unchecked(
                liability
            )
            or identity in liability_by_identity
        ):
            raise RuntimeError(
                "Task 11 retained worker-launch liability drifted"
            )
        liability_by_identity[identity] = liability

    authority: dict[
        tuple[str, int, int], tuple[str, str, str, str]
    ] = {}
    for settlement in settlements:
        identity = (
            activation_id,
            settlement.get("generation"),
            settlement.get("allocation_ordinal"),
        )
        launch = launches.get(identity)
        merged = settlement.get("merged_final_allocations")
        post = settlement.get("post_terminal_allocations")
        kind = settlement.get("settlement_kind")
        release_identity = settlement.get(
            "gpu_liability_reserve_release_identity_sha256"
        )
        settlement_identity = settlement.get("canonical_body_sha256")
        liability = liability_by_identity.get(identity)
        terminal = terminal_sources.get(
            (
                settlement.get("terminal_v2_key"),
                settlement.get("terminal_v2_version_id"),
            )
        )
        terminal_allocations = (
            terminal.get("allocations")
            if type(terminal) is dict
            else None
        )
        terminal_launches = (
            terminal.get("worker_launch_evidence")
            if type(terminal) is dict
            else None
        )
        terminal_liabilities = (
            terminal.get("worker_launch_liabilities")
            if type(terminal) is dict
            else None
        )
        matching_terminal_launches = (
            [
                item
                for item in terminal_launches
                if type(item) is dict
                and item.get("allocation_ordinal") == identity[2]
            ]
            if type(terminal_launches) is list
            else []
        )
        matching_terminal_liabilities = (
            [
                item
                for item in terminal_liabilities
                if type(item) is dict
                and item.get("allocation_ordinal") == identity[2]
            ]
            if type(terminal_liabilities) is list
            else []
        )
        terminal_launch = (
            matching_terminal_launches[0]
            if len(matching_terminal_launches) == 1
            else None
        )
        terminal_liability = (
            matching_terminal_liabilities[0]
            if len(matching_terminal_liabilities) == 1
            else None
        )
        expected_merged = (
            sorted(
                list(terminal_allocations) + list(post),
                key=lambda item: (
                    item.get("allocation_ordinal")
                    if type(item) is dict
                    else -1,
                    item.get("instance_id")
                    if type(item) is dict
                    else "",
                ),
            )
            if type(terminal_allocations) is list
            and type(post) is list
            else None
        )
        if (
            type(settlement) is not dict
            or launch is None
            or settlement.get("record_type")
            != "glm52_production_worker_launch_liability_settlement"
            or settlement.get("account_id") != "246813579024"
            or settlement.get("region") != "us-west-2"
            or settlement.get("run_id") != "glm52-sky-20260724"
            or settlement.get("activation_id") != activation_id
            or settlement.get("activation_ordinal")
            != launch.get("activation_ordinal")
            or settlement.get("campaign_identity_sha256")
            != launch.get("campaign_identity_sha256")
            or settlement.get("worker_launch_identity_sha256")
            != launch.get("canonical_body_sha256")
            or settlement.get(
                "gpu_liability_reserve_ledger_identity_sha256"
            )
            != launch.get(
                "gpu_liability_reserve_ledger_identity_sha256"
            )
            or type(release_identity) is not str
            or _SHA.fullmatch(release_identity) is None
            or type(settlement_identity) is not str
            or _SHA.fullmatch(settlement_identity) is None
            or settlement_identity
            != record_contract.canonical_record_identity_unchecked(
                settlement
            )
            or identity in authority
            or type(merged) is not list
            or type(post) is not list
            or settlement.get("merged_final_allocations_array_sha256")
            != canonical_sha256(merged)
            or settlement.get("post_terminal_allocations_array_sha256")
            != canonical_sha256(post)
            or settlement.get("final_worker_cardinality")
            != str(len(merged))
            or liability is None
            or type(terminal) is not dict
            or terminal.get("activation_id") != activation_id
            or terminal.get("activation_ordinal")
            != launch.get("activation_ordinal")
            or terminal.get("generation") != identity[1]
            or terminal.get("campaign_identity_sha256")
            != launch.get("campaign_identity_sha256")
            or settlement.get("terminal_v2_body_sha256")
            != terminal.get("canonical_body_sha256")
            or settlement.get("terminal_v2_allocations_array_sha256")
            != terminal.get("allocations_array_sha256")
            or terminal.get("allocations_array_sha256")
            != canonical_sha256(terminal_allocations)
            or merged != expected_merged
            or type(terminal_launch) is not dict
            or terminal_launch.get("worker_launch_identity_sha256")
            != launch.get("canonical_body_sha256")
            or terminal_launch.get("ec2_client_token")
            != launch.get("ec2_client_token")
            or type(terminal_liability) is not dict
            or terminal_liability.get("activation_id") != activation_id
            or terminal_liability.get("activation_ordinal")
            != launch.get("activation_ordinal")
            or terminal_liability.get("generation") != identity[1]
            or terminal_liability.get("worker_launch_identity_sha256")
            != launch.get("canonical_body_sha256")
            or terminal_liability.get("ec2_client_token")
            != launch.get("ec2_client_token")
            or settlement.get(
                "worker_launch_liability_identity_sha256"
            )
            != terminal_liability.get(
                "worker_launch_liability_identity_sha256"
            )
            or terminal_launch.get(
                "worker_launch_liability_identity_sha256"
            )
            != terminal_liability.get(
                "worker_launch_liability_identity_sha256"
            )
            or liability.get(
                "gpu_liability_reserve_release_identity_sha256"
            )
            != release_identity
            or liability.get("settlement_identity_sha256")
            != settlement_identity
        ):
            raise RuntimeError(
                "Task 11 retained liability settlement drifted"
            )
        state = launch.get("state")
        no_instance = (
            kind == "NO_INSTANCE_POSITIVE_REJECTION"
            and state == "REJECTED_NO_INSTANCE"
            and not merged
            and not post
            and terminal.get("worker_cardinality") == "ZERO"
            and terminal.get("final_ec2_states") == []
            and terminal_allocations == []
            and terminal_launch.get(
                "instance_terminal_identity_sha256"
            )
            is None
            and terminal_launch.get(
                "spend_allocation_close_identity_sha256"
            )
            is None
            and type(
                terminal_launch.get(
                    "positive_service_rejection_evidence_sha256"
                )
            )
            is str
            and _SHA.fullmatch(
                terminal_launch[
                    "positive_service_rejection_evidence_sha256"
                ]
            )
            is not None
            and liability.get("state")
            == "SETTLED_NO_INSTANCE_REJECTED"
            and settlement.get(
                "terminal_v2_allocations_array_sha256"
            )
            == canonical_sha256([])
            and type(
                settlement.get("service_rejection_evidence_sha256")
            )
            is str
            and _SHA.fullmatch(
                settlement["service_rejection_evidence_sha256"]
            )
            is not None
            and settlement.get(
                "instance_terminal_evidence_array_sha256"
            )
            == canonical_sha256([])
            and settlement.get("spend_close_evidence_array_sha256")
            == canonical_sha256([])
        )
        terminal_evidence: list[str] = []
        spend_close_evidence: list[str] = []
        instance_ids: list[str] = []
        closed_members = bool(merged)
        for member in merged:
            instance_id = (
                member.get("instance_id")
                if type(member) is dict
                else None
            )
            terminal_identity = (
                member.get("instance_terminal_identity_sha256")
                if type(member) is dict
                else None
            )
            spend_close_identity = (
                (
                    member.get(
                        "spend_allocation_close_identity_sha256"
                    )
                    if "spend_allocation_close_identity_sha256" in member
                    else member.get("allocation_close_identity_sha256")
                )
                if type(member) is dict
                else None
            )
            if (
                type(member) is not dict
                or member.get("allocation_ordinal") != identity[2]
                or type(instance_id) is not str
                or re.fullmatch(r"i-[0-9a-f]{17}", instance_id) is None
                or type(terminal_identity) is not str
                or _SHA.fullmatch(terminal_identity) is None
                or type(spend_close_identity) is not str
                or _SHA.fullmatch(spend_close_identity) is None
                or instance_id in instance_ids
            ):
                closed_members = False
                break
            instance_ids.append(instance_id)
            terminal_evidence.append(terminal_identity)
            spend_close_evidence.append(spend_close_identity)
        terminal_evidence.sort()
        spend_close_evidence.sort()
        if (
            settlement.get(
                "instance_terminal_evidence_array_sha256"
            )
            != canonical_sha256(terminal_evidence)
            or settlement.get("spend_close_evidence_array_sha256")
            != canonical_sha256(spend_close_evidence)
        ):
            closed_members = False
        instances_closed = (
            kind == "ALL_INSTANCES_TERMINAL_AND_SPEND_CLOSED"
            and state in {"INSTANCE_TERMINAL", "ALLOCATION_CLOSED"}
            and closed_members
            and any(
                type(item) is dict
                and item.get("allocation_ordinal") == identity[2]
                for item in terminal_allocations
            )
            and type(
                terminal_launch.get(
                    "instance_terminal_identity_sha256"
                )
            )
            is str
            and _SHA.fullmatch(
                terminal_launch[
                    "instance_terminal_identity_sha256"
                ]
            )
            is not None
            and type(
                terminal_launch.get(
                    "spend_allocation_close_identity_sha256"
                )
            )
            is str
            and _SHA.fullmatch(
                terminal_launch[
                    "spend_allocation_close_identity_sha256"
                ]
            )
            is not None
            and terminal_launch.get(
                "positive_service_rejection_evidence_sha256"
            )
            is None
            and liability.get("state") == "SETTLED_INSTANCE_CLOSED"
            and settlement.get("service_rejection_evidence_sha256")
            is None
        )
        token = launch.get("ec2_client_token")
        reserve_identity = launch.get(
            "gpu_liability_reserve_ledger_identity_sha256"
        )
        if (
            not (no_instance or instances_closed)
            or type(token) is not str
            or _SHA.fullmatch(token) is None
            or type(reserve_identity) is not str
        ):
            raise RuntimeError(
                "Task 11 retained liability settlement is not terminal"
            )
        authority[identity] = (
            token,
            reserve_identity,
            release_identity,
            settlement_identity,
        )
    return authority


def _read_retained_terminal_v2_sources(
    *,
    s3: object,
    bucket: str,
    settlements: tuple[Mapping[str, object], ...],
) -> dict[tuple[str, str], dict[str, object]]:
    versions: dict[str, str] = {}
    for settlement in settlements:
        key = settlement.get("terminal_v2_key")
        version_id = settlement.get("terminal_v2_version_id")
        if (
            type(key) is not str
            or not key
            or type(version_id) is not str
            or not version_id
            or (
                key in versions
                and versions[key] != version_id
            )
        ):
            raise RuntimeError(
                "Task 11 retained terminal coordinate drifted"
            )
        versions[key] = version_id
    store = Task11SpendObjectStore(
        s3=s3,
        bucket=bucket,
        version_by_key=versions,
    )
    sources: dict[tuple[str, str], dict[str, object]] = {}
    terminal_fields = frozenset(
        record_contract.RECORD_FIELDS[
            "glm52_production_terminal_v2"
        ]
    )
    for key, version_id in versions.items():
        observed = store.get_object(key=key)
        raw = observed.raw
        if not raw.endswith(b"\n"):
            raise RuntimeError(
                "Task 11 retained terminal bytes are not canonical"
            )
        try:
            terminal = json.loads(raw[:-1].decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Task 11 retained terminal bytes drifted"
            ) from exc
        arrays = (
            "allocations",
            "worker_launch_evidence",
            "worker_launch_liabilities",
            "request_evidence",
        )
        if (
            type(terminal) is not dict
            or set(terminal) != terminal_fields
            or terminal.get("schema_version") != 2
            or terminal.get("record_type")
            != "glm52_production_terminal_v2"
            or terminal.get("account_id") != "246813579024"
            or terminal.get("region") != "us-west-2"
            or terminal.get("run_id") != "glm52-sky-20260724"
            or terminal.get("canonical_body_sha256")
            != record_contract.canonical_record_identity_unchecked(
                terminal
            )
            or raw != canonical_json_bytes(terminal) + b"\n"
            or any(type(terminal.get(field)) is not list for field in arrays)
            or any(
                terminal.get(field + "_array_sha256")
                != canonical_sha256(terminal[field])
                for field in arrays
            )
            or observed.version_id != version_id
        ):
            raise RuntimeError(
                "Task 11 retained terminal object drifted"
            )
        sources[(key, version_id)] = terminal
    return sources


class Task11SpendEc2:
    _MAX_PAGES = 64
    _RESPONSE_FIELDS = frozenset(
        {"Reservations", "NextToken", "ResponseMetadata"}
    )
    _METADATA_FIELDS = frozenset(
        {
            "RequestId",
            "HTTPStatusCode",
            "HTTPHeaders",
            "RetryAttempts",
        }
    )
    _RESERVATION_FIELDS = frozenset(
        {
            "Groups",
            "Instances",
            "OwnerId",
            "RequesterId",
            "ReservationId",
        }
    )
    _INSTANCE_FIELDS = frozenset(
        {
            "AmdSevSnp",
            "Architecture",
            "BlockDeviceMappings",
            "BootMode",
            "ClientToken",
            "CpuOptions",
            "CurrentInstanceBootMode",
            "EbsOptimized",
            "ElasticGpuAssociations",
            "ElasticInferenceAcceleratorAssociations",
            "EnaSupport",
            "EnclaveOptions",
            "HibernationOptions",
            "Hypervisor",
            "IamInstanceProfile",
            "ImageId",
            "InstanceId",
            "InstanceLifecycle",
            "InstanceType",
            "Ipv6Address",
            "KernelId",
            "KeyName",
            "LaunchTime",
            "Licenses",
            "MaintenanceOptions",
            "MetadataOptions",
            "Monitoring",
            "NetworkInterfaces",
            "NetworkPerformanceOptions",
            "Operator",
            "OutpostArn",
            "Placement",
            "Platform",
            "PlatformDetails",
            "PrivateDnsName",
            "PrivateDnsNameOptions",
            "PrivateIpAddress",
            "ProductCodes",
            "PublicDnsName",
            "PublicIpAddress",
            "RamdiskId",
            "RootDeviceName",
            "RootDeviceType",
            "SecurityGroups",
            "SourceDestCheck",
            "SriovNetSupport",
            "State",
            "StateReason",
            "StateTransitionReason",
            "SubnetId",
            "Tags",
            "TpmSupport",
            "UsageOperation",
            "UsageOperationUpdateTime",
            "VirtualizationType",
            "VpcId",
        }
    )
    _REQUIRED_INSTANCE_FIELDS = frozenset(
        {
            "Architecture",
            "BlockDeviceMappings",
            "ClientToken",
            "EbsOptimized",
            "EnaSupport",
            "ImageId",
            "InstanceId",
            "InstanceType",
            "LaunchTime",
            "Placement",
            "RootDeviceName",
            "RootDeviceType",
            "State",
            "Tags",
            "VirtualizationType",
        }
    )
    _PLACEMENT_FIELDS = frozenset(
        {
            "Affinity",
            "AvailabilityZone",
            "AvailabilityZoneId",
            "GroupId",
            "GroupName",
            "HostId",
            "HostResourceGroupArn",
            "PartitionNumber",
            "SpreadDomain",
            "Tenancy",
        }
    )
    _BLOCK_DEVICE_FIELDS = frozenset({"DeviceName", "Ebs"})
    _EBS_FIELDS = frozenset(
        {
            "AssociatedResource",
            "AttachTime",
            "DeleteOnTermination",
            "KmsKeyIdentifier",
            "Status",
            "VolumeId",
        }
    )
    _TAG_FIELDS = frozenset({"Key", "Value"})
    _LAUNCH_TAG_FIELDS = frozenset(
        {
            "Project",
            "Campaign",
            "RunId",
            "Market",
            "campaign-identity-sha256",
            "activation-id",
            "activation-ordinal-text",
            "generation-text",
            "allocation-ordinal-text",
            "action-key",
            "sky-request-id",
            "sky-job-name",
            "sky-task-name",
            "task-yaml-sha256",
            "request-body-sha256",
        }
    )
    _FIXED_LAUNCH_TAGS = {
        "Project": "KEEP",
        "Campaign": "GLM-5.2",
        "RunId": "glm52-sky-20260724",
        "Market": "on-demand",
    }
    _ACTIVATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    _ORDINAL_TEXT = re.compile(r"^[0-9]{8}$")
    _STATE_CODES = {
        "pending": 0,
        "running": 16,
    }

    def __init__(
        self,
        ec2: object,
        *,
        activation_id: str,
        token_custody: Optional[
            dict[tuple[str, int, int], str]
        ] = None,
        expected_launches: Optional[
            Mapping[tuple[str, int, int], tuple[Mapping[str, str], str]]
        ] = None,
    ) -> None:
        if (
            type(activation_id) is not str
            or self._ACTIVATION_ID.fullmatch(activation_id) is None
        ):
            raise RuntimeError(
                "Task 11 spend EC2 activation identity drifted"
            )
        if token_custody is not None and type(token_custody) is not dict:
            raise RuntimeError(
                "Task 11 spend ClientToken custody is malformed"
            )
        self._ec2 = ec2
        self._activation_id = activation_id
        self._token_custody = (
            {} if token_custody is None else token_custody
        )
        self._expected_launches = expected_launches
        self._pagination: dict[str, dict[str, object]] = {}

    @staticmethod
    def _json_safe(value: object) -> object:
        if value is None or type(value) in (str, int, bool, float):
            return value
        if type(value) is datetime:
            if value.tzinfo is None or value.utcoffset() is None:
                raise RuntimeError(
                    "Task 11 spend EC2 timestamp is not timezone-aware"
                )
            return _rfc3339(value)
        if type(value) in (list, tuple):
            return [Task11SpendEc2._json_safe(item) for item in value]
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise RuntimeError(
                    "Task 11 spend EC2 response key drifted"
                )
            return {
                key: Task11SpendEc2._json_safe(item)
                for key, item in value.items()
            }
        raise RuntimeError("Task 11 spend EC2 response type drifted")

    @staticmethod
    def _token(value: Mapping[str, object]) -> str:
        return base64.b64encode(canonical_json_bytes(value)).decode("ascii")

    @classmethod
    def _decode_token(cls, value: str) -> Mapping[str, object]:
        if type(value) is not str or not value or len(value) > 32768:
            raise RuntimeError("Task 11 spend EC2 token drifted")
        try:
            parsed = json.loads(
                base64.b64decode(value, validate=True).decode("ascii")
            )
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Task 11 spend EC2 token drifted") from exc
        if (
            type(parsed) is not dict
            or set(parsed)
            != {
                "page_index",
                "response_identity_sha256",
                "run_id",
                "service_token",
            }
            or type(parsed["page_index"]) is not int
            or parsed["page_index"] < 1
            or type(parsed["response_identity_sha256"]) is not str
            or _SHA.fullmatch(parsed["response_identity_sha256"]) is None
            or parsed["run_id"] != "glm52-sky-20260724"
            or type(parsed["service_token"]) is not str
            or not parsed["service_token"]
            or cls._token(parsed) != value
        ):
            raise RuntimeError("Task 11 spend EC2 token is not closed")
        return parsed

    @classmethod
    def _metadata(cls, response: object) -> Mapping[str, object]:
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        headers = (
            metadata.get("HTTPHeaders")
            if type(metadata) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or set(metadata) - cls._METADATA_FIELDS
            or not {
                "HTTPStatusCode",
                "RequestId",
                "RetryAttempts",
            }.issubset(metadata)
            or type(metadata["HTTPStatusCode"]) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata["RequestId"]) is not str
            or not metadata["RequestId"]
            or type(metadata["RetryAttempts"]) is not int
            or metadata["RetryAttempts"] != 0
            or (
                "HTTPHeaders" in metadata
                and (
                    type(headers) is not dict
                    or any(
                        type(key) is not str or type(item) is not str
                        for key, item in headers.items()
                    )
                )
            )
        ):
            raise RuntimeError("Task 11 spend EC2 metadata drifted")
        return metadata

    @classmethod
    def _instance(
        cls,
        value: object,
        *,
        run_id: str,
        activation_id: str,
        expected_launches: Optional[
            Mapping[tuple[str, int, int], tuple[Mapping[str, str], str]]
        ],
    ) -> tuple[
        Mapping[str, object],
        str,
        tuple[str, int, int],
        str,
    ]:
        if (
            type(value) is not dict
            or not cls._REQUIRED_INSTANCE_FIELDS.issubset(value)
            or set(value) - cls._INSTANCE_FIELDS
        ):
            raise RuntimeError("Task 11 spend EC2 instance row drifted")
        state = value["State"]
        placement = value["Placement"]
        launch_time = value["LaunchTime"]
        tags = value["Tags"]
        mappings = value["BlockDeviceMappings"]
        if (
            type(value["InstanceId"]) is not str
            or re.fullmatch(r"i-[0-9a-f]{17}", value["InstanceId"]) is None
            or value["InstanceType"] != "p5.48xlarge"
            or value.get("InstanceLifecycle") is not None
            or type(value["ClientToken"]) is not str
            or _SHA.fullmatch(value["ClientToken"]) is None
            or type(value["ImageId"]) is not str
            or re.fullmatch(r"ami-[0-9a-f]{17}", value["ImageId"]) is None
            or value["Architecture"] != "x86_64"
            or value["EbsOptimized"] is not True
            or value["EnaSupport"] is not True
            or value["RootDeviceType"] != "ebs"
            or type(value["RootDeviceName"]) is not str
            or not value["RootDeviceName"]
            or value["VirtualizationType"] != "hvm"
            or type(launch_time) is not datetime
            or launch_time.tzinfo is None
            or launch_time.utcoffset() is None
            or type(state) is not dict
            or set(state) != {"Code", "Name"}
            or type(state["Name"]) is not str
            or state["Name"] not in cls._STATE_CODES
            or type(state["Code"]) is not int
            or state["Code"] != cls._STATE_CODES[state["Name"]]
            or type(placement) is not dict
            or "AvailabilityZone" not in placement
            or set(placement) - cls._PLACEMENT_FIELDS
            or type(placement["AvailabilityZone"]) is not str
            or re.fullmatch(
                r"us-west-2[a-f]",
                placement["AvailabilityZone"],
            )
            is None
            or placement.get("Tenancy") != "default"
            or type(tags) is not list
            or type(mappings) is not list
        ):
            raise RuntimeError("Task 11 spend EC2 instance identity drifted")
        tag_map: dict[str, str] = {}
        for tag in tags:
            if (
                type(tag) is not dict
                or set(tag) != cls._TAG_FIELDS
                or type(tag["Key"]) is not str
                or not tag["Key"]
                or type(tag["Value"]) is not str
                or tag["Key"] in tag_map
            ):
                raise RuntimeError("Task 11 spend EC2 tag row drifted")
            tag_map[tag["Key"]] = tag["Value"]
        if (
            set(tag_map) != cls._LAUNCH_TAG_FIELDS
            or any(
                tag_map.get(key) != expected
                for key, expected in cls._FIXED_LAUNCH_TAGS.items()
            )
            or tag_map["RunId"] != run_id
            or tag_map["activation-id"] != activation_id
            or _SHA.fullmatch(
                tag_map["campaign-identity-sha256"]
            )
            is None
            or _SHA.fullmatch(tag_map["task-yaml-sha256"]) is None
            or _SHA.fullmatch(tag_map["request-body-sha256"]) is None
            or cls._ORDINAL_TEXT.fullmatch(
                tag_map["activation-ordinal-text"]
            )
            is None
            or cls._ORDINAL_TEXT.fullmatch(tag_map["generation-text"])
            is None
            or cls._ORDINAL_TEXT.fullmatch(
                tag_map["allocation-ordinal-text"]
            )
            is None
            or int(tag_map["activation-ordinal-text"]) <= 0
            or int(tag_map["generation-text"]) <= 0
            or int(tag_map["allocation-ordinal-text"]) <= 0
            or tag_map["action-key"]
            != (
                "ACTION#"
                + tag_map["activation-ordinal-text"]
                + "#SKY_POST#"
                + tag_map["generation-text"]
            )
            or any(
                type(tag_map[field]) is not str
                or not tag_map[field]
                or not tag_map[field].strip()
                for field in (
                    "sky-request-id",
                    "sky-job-name",
                    "sky-task-name",
                )
            )
        ):
            raise RuntimeError("Task 11 spend EC2 tags drifted")
        root_volume_id: Optional[str] = None
        for mapping in mappings:
            if (
                type(mapping) is not dict
                or set(mapping) != cls._BLOCK_DEVICE_FIELDS
                or type(mapping["DeviceName"]) is not str
                or type(mapping["Ebs"]) is not dict
                or not {
                    "AttachTime",
                    "DeleteOnTermination",
                    "Status",
                    "VolumeId",
                }.issubset(mapping["Ebs"])
                or set(mapping["Ebs"]) - cls._EBS_FIELDS
                or type(mapping["Ebs"]["AttachTime"]) is not datetime
                or mapping["Ebs"]["AttachTime"].tzinfo is None
                or mapping["Ebs"]["AttachTime"].utcoffset() is None
                or mapping["Ebs"]["DeleteOnTermination"] is not True
                or mapping["Ebs"]["Status"] != "attached"
                or type(mapping["Ebs"]["VolumeId"]) is not str
                or re.fullmatch(
                    r"vol-(?:[0-9a-f]{8}|[0-9a-f]{17})",
                    mapping["Ebs"]["VolumeId"],
                )
                is None
            ):
                raise RuntimeError(
                    "Task 11 spend EC2 volume row drifted"
                )
            if mapping["DeviceName"] == value["RootDeviceName"]:
                if root_volume_id is not None:
                    raise RuntimeError(
                        "Task 11 spend EC2 root volume was duplicated"
                    )
                root_volume_id = mapping["Ebs"]["VolumeId"]
        if root_volume_id is None or len(mappings) != 1:
            raise RuntimeError(
                "Task 11 spend EC2 root volume identity drifted"
            )
        allocation_identity = (
            activation_id,
            int(tag_map["generation-text"]),
            int(tag_map["allocation-ordinal-text"]),
        )
        if expected_launches is not None:
            authority = expected_launches.get(allocation_identity)
            if authority is None or tag_map != authority[0] or value["ClientToken"] != authority[1]:
                raise RuntimeError("Task 11 spend EC2 retained launch authority drifted")
        return (
            {
                "InstanceId": value["InstanceId"],
                "InstanceType": value["InstanceType"],
                "InstanceLifecycle": value.get("InstanceLifecycle"),
                "ClientToken": value["ClientToken"],
                "RootVolumeId": root_volume_id,
                "State": state["Name"],
                "LaunchTime": _rfc3339(launch_time),
                "AvailabilityZone": placement["AvailabilityZone"],
                "Tags": tag_map,
            },
            root_volume_id,
            allocation_identity,
            value["ClientToken"],
        )

    def describe_allocation_history(
        self,
        *,
        run_id: str,
        next_token: Optional[str],
    ) -> Mapping[str, object]:
        if run_id != "glm52-sky-20260724":
            raise RuntimeError("Task 11 spend EC2 run identity drifted")
        state = self._pagination.get(run_id)
        service_token = None
        if next_token is None:
            if state is not None:
                raise RuntimeError(
                    "Task 11 spend EC2 pagination restarted"
                )
            state = {
                "expected_token": None,
                "pages": 0,
                "seen_instances": set(),
                "seen_active_launches": set(),
                "seen_service_tokens": set(),
                "seen_volumes": set(),
            }
            self._pagination[run_id] = state
        else:
            parsed = self._decode_token(next_token)
            if (
                state is None
                or state["expected_token"] != next_token
                or parsed["page_index"] != state["pages"]
            ):
                raise RuntimeError(
                    "Task 11 spend EC2 token custody drifted"
                )
            service_token = parsed["service_token"]
        if int(state["pages"]) >= self._MAX_PAGES:
            raise RuntimeError(
                "Task 11 spend EC2 pagination exceeded bound"
            )
        request = {
            "Filters": [
                {
                    "Name": "tag:RunId",
                    "Values": [run_id],
                },
                {
                    "Name": "tag:activation-id",
                    "Values": [self._activation_id],
                },
                {
                    "Name": "instance-state-name",
                    "Values": [
                        "pending",
                        "running",
                    ],
                },
            ],
            "MaxResults": 1000,
        }
        if service_token is not None:
            request["NextToken"] = service_token
        response = self._ec2.describe_instances(**request)
        metadata = self._metadata(response)
        if (
            type(response) is not dict
            or not {"Reservations", "ResponseMetadata"}.issubset(response)
            or set(response) - self._RESPONSE_FIELDS
            or type(response["Reservations"]) is not list
        ):
            raise RuntimeError("Task 11 spend EC2 read drifted")
        instances: list[Mapping[str, object]] = []
        for reservation in response["Reservations"]:
            if (
                type(reservation) is not dict
                or not {
                    "Groups",
                    "Instances",
                    "OwnerId",
                    "ReservationId",
                }.issubset(reservation)
                or set(reservation) - self._RESERVATION_FIELDS
                or reservation["OwnerId"] != "246813579024"
                or type(reservation["ReservationId"]) is not str
                or re.fullmatch(
                    r"r-(?:[0-9a-f]{8}|[0-9a-f]{17})",
                    reservation["ReservationId"],
                )
                is None
                or type(reservation["Groups"]) is not list
                or type(reservation["Instances"]) is not list
                or (
                    "RequesterId" in reservation
                    and (
                        type(reservation["RequesterId"]) is not str
                        or not reservation["RequesterId"]
                    )
                )
            ):
                raise RuntimeError(
                    "Task 11 spend EC2 reservation row drifted"
                )
            for group in reservation["Groups"]:
                if (
                    type(group) is not dict
                    or set(group) != {"GroupId", "GroupName"}
                    or type(group["GroupId"]) is not str
                    or re.fullmatch(
                        r"sg-(?:[0-9a-f]{8}|[0-9a-f]{17})",
                        group["GroupId"],
                    )
                    is None
                    or type(group["GroupName"]) is not str
                ):
                    raise RuntimeError(
                        "Task 11 spend EC2 group row drifted"
                    )
            for item in reservation["Instances"]:
                (
                    instance,
                    volume_id,
                    allocation_identity,
                    client_token,
                ) = self._instance(
                    item,
                    run_id=run_id,
                    activation_id=self._activation_id,
                    expected_launches=self._expected_launches,
                )
                if (
                    instance["InstanceId"] in state["seen_instances"]
                    or volume_id in state["seen_volumes"]
                    or (
                        self._expected_launches is not None
                        and allocation_identity in state["seen_active_launches"]
                    )
                ):
                    raise RuntimeError(
                        "Task 11 spend EC2 row identity was duplicated"
                    )
                state["seen_instances"].add(instance["InstanceId"])
                state["seen_volumes"].add(volume_id)
                state["seen_active_launches"].add(allocation_identity)
                _bind_spend_client_token(
                    self._token_custody,
                    allocation_identity=allocation_identity,
                    client_token=client_token,
                    source="ec2",
                )
                instances.append(instance)
        response_identity = canonical_sha256(self._json_safe(response))
        instances = [
            {
                **instance,
                "RequestId": metadata["RequestId"],
                "ResponseIdentitySha256": response_identity,
            }
            for instance in instances
        ]
        state["pages"] = int(state["pages"]) + 1
        returned_token = response.get("NextToken")
        if returned_token is not None:
            if (
                type(returned_token) is not str
                or not returned_token
                or len(returned_token) > 16384
                or returned_token == service_token
                or returned_token in state["seen_service_tokens"]
                or int(state["pages"]) >= self._MAX_PAGES
            ):
                raise RuntimeError(
                    "Task 11 spend EC2 pagination did not progress"
                )
            state["seen_service_tokens"].add(returned_token)
            state["expected_token"] = self._token(
                {
                    "page_index": state["pages"],
                    "response_identity_sha256": response_identity,
                    "run_id": run_id,
                    "service_token": returned_token,
                }
            )
        else:
            state["expected_token"] = None
            self._pagination.pop(run_id, None)
        return {
            "instances": instances,
            "next_token": state["expected_token"],
            "request_id": metadata["RequestId"],
            "observed_at": _rfc3339(_utc_now()),
        }


class Task11ReserveReader:
    _MAX_PAGES = 64
    _RESPONSE_FIELDS = frozenset(
        {
            "Items",
            "Count",
            "ScannedCount",
            "LastEvaluatedKey",
            "ResponseMetadata",
        }
    )
    _METADATA_FIELDS = Task11SpendEc2._METADATA_FIELDS
    _PK = record_contract.ledger_pk("glm52-sky-20260724")
    _SK_PREFIX = "GPU_LIABILITY_RESERVE#"
    _RESERVE_SK = re.compile(
        r"^GPU_LIABILITY_RESERVE#"
        r"(?P<activation>[A-Za-z0-9][A-Za-z0-9._-]{0,127})#"
        r"(?P<generation>[0-9]{8})#"
        r"(?P<allocation>[0-9]{8})$"
    )
    _RESERVE_ITEM_FIELDS = frozenset(
        {
            "PK",
            "SK",
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "activation_id",
            "generation",
            "generation_text",
            "allocation_ordinal",
            "allocation_ordinal_text",
            "ec2_client_token",
            "request_identity_sha256",
            "ledger_predecessor_identity_sha256",
            "gpu_reserve_seconds",
            "gpu_reserve_cost_usd",
            "root_volume_gib",
            "root_volume_tail_usd_max",
            "residual_liability_approval_identity_sha256",
            "epoch",
            "revision",
            "nonce_owner_identity_sha256",
            "state",
            "observed_at",
            "reserve_key",
            "canonical_body_sha256",
        }
    )
    _TERMINAL_LAUNCH_STATES = frozenset(
        {
            "ABANDONED_NOT_SENT",
            "REJECTED_NO_INSTANCE",
            "INSTANCE_TERMINAL",
            "ALLOCATION_CLOSED",
        }
    )

    def __init__(
        self,
        *,
        dynamodb: object,
        table_name: str,
        token_custody: Optional[
            dict[tuple[str, int, int], str]
        ] = None,
        activation_id: Optional[str] = None,
        active_launches: Optional[
            Mapping[tuple[str, int, int], tuple[Mapping[str, str], str]]
        ] = None,
        known_launches: Optional[
            Mapping[
                tuple[str, int, int],
                tuple[str, Mapping[str, str], str],
            ]
        ] = None,
        settled_launches: Optional[
            Mapping[tuple[str, int, int], tuple[str, str, str, str]]
        ] = None,
    ) -> None:
        if type(table_name) is not str or not table_name:
            raise RuntimeError("Task 11 reserve table identity is absent")
        if token_custody is not None and type(token_custody) is not dict:
            raise RuntimeError(
                "Task 11 reserve ClientToken custody is malformed"
            )
        if activation_id is not None and (
            type(activation_id) is not str
            or Task11SpendEc2._ACTIVATION_ID.fullmatch(activation_id) is None
        ):
            raise RuntimeError("Task 11 reserve activation identity is absent")
        if (active_launches is None) != (known_launches is None):
            raise RuntimeError("Task 11 reserve launch authority is incomplete")
        self._dynamodb = dynamodb
        self._table_name = table_name
        self._token_custody = (
            {} if token_custody is None else token_custody
        )
        self._activation_id = activation_id
        self._active_launches = active_launches
        self._known_launches = known_launches
        self._settled_launches = settled_launches
        self._pagination: Optional[dict[str, object]] = None

    @staticmethod
    def _token(value: Mapping[str, object]) -> str:
        return base64.b64encode(canonical_json_bytes(value)).decode("ascii")

    def _decode_token(self, value: str) -> Mapping[str, object]:
        if type(value) is not str or not value or len(value) > 32768:
            raise RuntimeError("Task 11 reserve token drifted")
        try:
            parsed = json.loads(
                base64.b64decode(value, validate=True).decode("ascii")
            )
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Task 11 reserve token drifted") from exc
        if (
            type(parsed) is not dict
            or set(parsed)
            != {
                "last_evaluated_key",
                "page_index",
                "response_identity_sha256",
                "table_name",
            }
            or parsed["table_name"] != self._table_name
            or type(parsed["page_index"]) is not int
            or parsed["page_index"] < 1
            or type(parsed["response_identity_sha256"]) is not str
            or _SHA.fullmatch(parsed["response_identity_sha256"]) is None
            or not self._valid_key(parsed["last_evaluated_key"])
            or self._token(parsed) != value
        ):
            raise RuntimeError("Task 11 reserve token is not closed")
        return parsed

    @classmethod
    def _valid_key(cls, value: object) -> bool:
        return (
            type(value) is dict
            and set(value) == {"PK", "SK"}
            and type(value["PK"]) is dict
            and set(value["PK"]) == {"S"}
            and value["PK"]["S"] == cls._PK
            and type(value["SK"]) is dict
            and set(value["SK"]) == {"S"}
            and type(value["SK"]["S"]) is str
            and cls._RESERVE_SK.fullmatch(value["SK"]["S"]) is not None
        )

    @classmethod
    def _metadata(cls, response: object) -> Mapping[str, object]:
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        headers = (
            metadata.get("HTTPHeaders")
            if type(metadata) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or set(metadata) - cls._METADATA_FIELDS
            or not {
                "HTTPStatusCode",
                "RequestId",
                "RetryAttempts",
            }.issubset(metadata)
            or type(metadata["HTTPStatusCode"]) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata["RequestId"]) is not str
            or not metadata["RequestId"]
            or type(metadata["RetryAttempts"]) is not int
            or metadata["RetryAttempts"] != 0
            or (
                "HTTPHeaders" in metadata
                and (
                    type(headers) is not dict
                    or any(
                        type(key) is not str or type(item) is not str
                        for key, item in headers.items()
                    )
                )
            )
        ):
            raise RuntimeError("Task 11 reserve metadata drifted")
        return metadata

    def list_reserves(
        self,
        *,
        next_token: Optional[str],
    ) -> ReserveListPage:
        exclusive_start_key = None
        state = self._pagination
        if next_token is None:
            if state is not None:
                raise RuntimeError(
                    "Task 11 reserve pagination restarted"
                )
            state = {
                "expected_token": None,
                "last_sk": None,
                "pages": 0,
                "seen_keys": set(),
                "seen_last_keys": set(),
            }
            self._pagination = state
        else:
            parsed = self._decode_token(next_token)
            if (
                state is None
                or state["expected_token"] != next_token
                or parsed["page_index"] != state["pages"]
            ):
                raise RuntimeError(
                    "Task 11 reserve token custody drifted"
                )
            exclusive_start_key = parsed["last_evaluated_key"]
        if int(state["pages"]) >= self._MAX_PAGES:
            raise RuntimeError(
                "Task 11 reserve pagination exceeded bound"
            )
        request = {
            "TableName": self._table_name,
            "KeyConditionExpression": (
                "#pk = :pk AND begins_with(#sk, :sk_prefix)"
            ),
            "ExpressionAttributeNames": {
                "#pk": "PK",
                "#sk": "SK",
            },
            "ExpressionAttributeValues": {
                ":pk": {"S": self._PK},
                ":sk_prefix": {
                    "S": self._SK_PREFIX
                    + (
                        self._activation_id + "#"
                        if self._activation_id is not None
                        else ""
                    )
                },
            },
            "ConsistentRead": True,
            "Limit": 100,
            "ScanIndexForward": True,
            "ReturnConsumedCapacity": "NONE",
        }
        if exclusive_start_key is not None:
            request["ExclusiveStartKey"] = exclusive_start_key
        response = self._dynamodb.query(**request)
        metadata = self._metadata(response)
        if (
            type(response) is not dict
            or not {
                "Items",
                "Count",
                "ScannedCount",
                "ResponseMetadata",
            }.issubset(response)
            or set(response) - self._RESPONSE_FIELDS
            or type(response["Items"]) is not list
            or type(response["Count"]) is not int
            or response["Count"] != len(response["Items"])
            or type(response["ScannedCount"]) is not int
            or response["ScannedCount"] != response["Count"]
        ):
            raise RuntimeError("Task 11 reserve read drifted")
        records: list[Mapping[str, object]] = []
        last_item_key = None
        for item in response["Items"]:
            try:
                decoded = decode_item(item)
                if encode_item(decoded) != item:
                    raise ValueError(
                        "DynamoDB item was not canonically encoded"
                    )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Task 11 reserve item encoding drifted"
                ) from exc
            pk = decoded.get("PK")
            sk = decoded.get("SK")
            reserve_match = (
                self._RESERVE_SK.fullmatch(sk)
                if type(sk) is str
                else None
            )
            if (
                set(decoded) != self._RESERVE_ITEM_FIELDS
                or pk != self._PK
                or type(sk) is not str
                or not sk
                or reserve_match is None
                or sk in state["seen_keys"]
                or (
                    state["last_sk"] is not None
                    and sk <= state["last_sk"]
                )
            ):
                raise RuntimeError(
                    "Task 11 reserve item identity drifted"
                )
            state["seen_keys"].add(sk)
            state["last_sk"] = sk
            last_item_key = {
                "PK": {"S": self._PK},
                "SK": {"S": sk},
            }
            is_reserve_key = self._RESERVE_SK.fullmatch(sk) is not None
            is_reserve_record = (
                decoded.get("record_type")
                == "glm52_gpu_liability_reserve_v1"
            )
            if not is_reserve_key or not is_reserve_record:
                raise RuntimeError(
                    "Task 11 reserve row type drifted"
                )
            if is_reserve_record:
                activation_id = reserve_match.group("activation")
                generation_text = reserve_match.group("generation")
                allocation_text = reserve_match.group("allocation")
                client_token = decoded.get("ec2_client_token")
                if (
                    decoded.get("reserve_key") != sk
                    or decoded.get("account_id") != "246813579024"
                    or decoded.get("region") != "us-west-2"
                    or decoded.get("run_id") != "glm52-sky-20260724"
                    or decoded.get("activation_id") != activation_id
                    or type(decoded.get("generation")) is not int
                    or decoded["generation"] <= 0
                    or decoded.get("generation_text")
                    != generation_text
                    or decoded["generation"]
                    != int(generation_text)
                    or type(decoded.get("allocation_ordinal")) is not int
                    or decoded["allocation_ordinal"] <= 0
                    or decoded.get("allocation_ordinal_text")
                    != allocation_text
                    or decoded["allocation_ordinal"]
                    != int(allocation_text)
                    or type(client_token) is not str
                    or _SHA.fullmatch(client_token) is None
                ):
                    raise RuntimeError(
                        "Task 11 reserve key binding drifted"
                    )
                identity = (
                    activation_id,
                    decoded["generation"],
                    decoded["allocation_ordinal"],
                )
                if (
                    self._activation_id is not None
                    and activation_id != self._activation_id
                ):
                    raise RuntimeError("Task 11 reserve activation drifted")
                known = (
                    self._known_launches.get(identity)
                    if self._known_launches is not None
                    else None
                )
                if self._known_launches is not None and (
                    known is None or known[2] != client_token
                ):
                    raise RuntimeError(
                        "Task 11 reserve has no retained launch authority"
                    )
                active = (
                    self._active_launches.get(identity)
                    if self._active_launches is not None
                    else None
                )
                settled = (
                    self._settled_launches.get(identity)
                    if self._settled_launches is not None
                    else None
                )
                if active is not None:
                    if active[1] != client_token:
                        raise RuntimeError(
                            "Task 11 reserve active launch token drifted"
                        )
                    _bind_spend_client_token(
                        self._token_custody,
                        allocation_identity=identity,
                        client_token=client_token,
                        source="reserve",
                    )
                elif known is not None:
                    if known[0] not in self._TERMINAL_LAUNCH_STATES:
                        raise RuntimeError(
                            "Task 11 reserve launch is neither active nor terminal"
                        )
                    if settled is not None:
                        if (
                            settled[0] != client_token
                            or settled[1]
                            != decoded.get("canonical_body_sha256")
                        ):
                            raise RuntimeError(
                                "Task 11 settled reserve identity drifted"
                            )
                        continue
                elif self._active_launches is not None:
                    raise RuntimeError(
                        "Task 11 reserve has no active retained launch authority"
                    )
                else:
                    _bind_spend_client_token(
                        self._token_custody,
                        allocation_identity=identity,
                        client_token=client_token,
                        source="reserve",
                    )
                if (
                    active is not None
                    or known is not None
                    or self._active_launches is None
                ):
                    records.append(
                        {
                            key: value
                            for key, value in decoded.items()
                            if key not in {"PK", "SK"}
                        }
                    )
        last_key = response.get("LastEvaluatedKey")
        state["pages"] = int(state["pages"]) + 1
        token = None
        if last_key is not None:
            key_identity = canonical_sha256(last_key)
            if (
                not self._valid_key(last_key)
                or last_key != last_item_key
                or key_identity in state["seen_last_keys"]
                or int(state["pages"]) >= self._MAX_PAGES
            ):
                raise RuntimeError(
                    "Task 11 reserve pagination did not progress"
                )
            state["seen_last_keys"].add(key_identity)
            response_identity = canonical_sha256(response)
            token = self._token(
                {
                    "last_evaluated_key": last_key,
                    "page_index": state["pages"],
                    "response_identity_sha256": response_identity,
                    "table_name": self._table_name,
                }
            )
            state["expected_token"] = token
        else:
            state["expected_token"] = None
            self._pagination = None
        return ReserveListPage(
            records=tuple(records),
            next_token=token,
            request_id=metadata["RequestId"],
            observed_at=_rfc3339(_utc_now()),
        )


class Task11SpendInspector:
    def __init__(
        self,
        *,
        request: object,
        services: SpendAuthorityServices,
        token_custody: Optional[Mapping[tuple[str, int, int], object]] = None,
        expected_launches: Optional[
            Mapping[tuple[str, int, int], tuple[Mapping[str, str], str]]
        ] = None,
    ) -> None:
        self._request = request
        self._services = services
        self._token_custody = token_custody
        self._expected_launches = expected_launches

    def inspect(self, request: object) -> object:
        if request != self._request:
            raise RuntimeError("Task 11 spend request was substituted")
        result = inspect_spend_authority(self._request, self._services)
        if self._token_custody is not None or self._expected_launches is not None:
            if self._token_custody is None or self._expected_launches is None:
                raise RuntimeError("Task 11 spend ClientToken custody is incomplete")
            _finalize_spend_token_custody(self._token_custody, self._expected_launches)
        return result


class Task11SkyProbe:
    def __init__(
        self,
        *,
        invoke_attestation: Callable[[Mapping[str, object]], AttestationResult],
        admission_identity_sha256: str,
    ) -> None:
        self._invoke_attestation = invoke_attestation
        self._admission_identity = admission_identity_sha256

    def inspect(self, request: Mapping[str, object]) -> object:
        attestation = self._invoke_attestation(request)
        return build_sky_relay_probe_result(
            account_id=request["account_id"],
            region=request["region"],
            run_id=request["run_id"],
            request_identity_sha256=canonical_sha256(request),
            direct_response_request_id=(
                attestation.direct_response_request_id
            ),
            tls_peer_certificate_sha256=(
                attestation.tls_peer_certificate_sha256
            ),
            attestation_identity_sha256=(
                attestation.canonical_identity_sha256
            ),
            admission_identity_sha256=self._admission_identity,
            sky_user_identity=attestation.sky_user_identity,
            sky_roles=attestation.sky_roles,
            token_expires_at=attestation.token_expires_at,
            effective_controller_identity_sha256=(
                attestation.effective_controller_identity_sha256
            ),
            observed_at=attestation.observed_at,
        )


def _closure_role_name(activation_id: str) -> str:
    if type(activation_id) is not str or not activation_id.isascii():
        raise RuntimeError("Task 11 activation identity is invalid")
    activation_sha = hashlib.sha256(
        activation_id.encode("ascii")
    ).hexdigest()[:16]
    return "keep-glm52-h1g-closure-session-" + activation_sha


def _closure_role_arn(
    *,
    account_id: str,
    activation_id: str,
) -> str:
    return (
        "arn:aws:iam::"
        + account_id
        + ":role/"
        + _closure_role_name(activation_id)
    )


def _closure_session_name(
    *,
    activation_id: str,
    aws_request_id: str,
) -> str:
    activation_sha = hashlib.sha256(
        activation_id.encode("ascii")
    ).hexdigest()[:16]
    request_sha = hashlib.sha256(
        aws_request_id.encode("ascii")
    ).hexdigest()[:16]
    return "h1g-decision-" + activation_sha + "-" + request_sha


def _authenticated_metadata(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if (
        type(value) is not dict
        or value.get("HTTPStatusCode") != 200
        or type(value.get("RequestId")) is not str
        or not value["RequestId"]
        or value.get("RetryAttempts") != 0
    ):
        raise RuntimeError(label + " metadata is unauthenticated")
    return value


def _aws_clients(
    *,
    config: Task11ProductionConfig,
    context: object,
    session_factory: Optional[Callable[..., object]] = None,
    utc_clock: Callable[[], datetime] = _utc_now,
) -> Task11AwsClients:
    request_id = getattr(context, "aws_request_id", None)
    invoked_function_arn = getattr(
        context,
        "invoked_function_arn",
        None,
    )
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    try:
        parsed_request_id = (
            uuid.UUID(request_id)
            if type(request_id) is str
            else None
        )
    except ValueError:
        parsed_request_id = None
    if (
        parsed_request_id is None
        or str(parsed_request_id) != request_id
    ):
        raise RuntimeError("Task 11 Lambda request ID is invalid")
    if invoked_function_arn != config.decision_function_version_arn:
        raise RuntimeError(
            "Task 11 invoked function is not the exact Decision version"
        )
    if not callable(remaining):
        raise RuntimeError("Task 11 Lambda context is invalid")
    remaining_millis = remaining()
    if (
        type(remaining_millis) is not int
        or remaining_millis <= 0
        or remaining_millis > 840_000
    ):
        raise RuntimeError("Task 11 Lambda remaining time is invalid")
    expected_role_arn = _closure_role_arn(
        account_id=config.account_id,
        activation_id=config.activation_id,
    )
    if config.closure_role_arn != expected_role_arn:
        raise RuntimeError("Task 11 closure role ARN is not exact")
    session_name = _closure_session_name(
        activation_id=config.activation_id,
        aws_request_id=request_id,
    )
    request_started_at = utc_clock()
    if (
        type(request_started_at) is not datetime
        or request_started_at.tzinfo is None
    ):
        raise RuntimeError("Task 11 UTC clock is unauthenticated")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    sdk_config = Config(
        connect_timeout=2,
        read_timeout=5,
        max_pool_connections=14,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session_constructor = (
        boto3.session.Session
        if session_factory is None
        else session_factory
    )
    base_session = session_constructor(region_name=config.region)
    base_sts = base_session.client("sts", config=sdk_config)
    assume_role = getattr(base_sts, "assume_role", None)
    get_base_identity = getattr(
        base_sts,
        "get_caller_identity",
        None,
    )
    if not callable(assume_role) or not callable(get_base_identity):
        raise RuntimeError("Task 11 base STS client is incomplete")
    source_caller_identity = get_base_identity()
    source_metadata = (
        source_caller_identity.get("ResponseMetadata")
        if type(source_caller_identity) is dict
        else None
    )
    _authenticated_metadata(
        source_metadata,
        label="Task 11 source caller identity",
    )
    if (
        type(source_caller_identity) is not dict
        or set(source_caller_identity)
        != {"Account", "Arn", "UserId", "ResponseMetadata"}
        or source_caller_identity.get("Account") != config.account_id
        or type(source_caller_identity.get("Arn")) is not str
        or not source_caller_identity["Arn"].startswith(
            "arn:aws:sts::"
            + config.account_id
            + ":assumed-role/keep-glm52-h1g-support-decision/"
        )
        or type(source_caller_identity.get("UserId")) is not str
        or not source_caller_identity["UserId"]
    ):
        raise RuntimeError(
            "Task 11 source DecisionRole identity is invalid"
        )
    response = assume_role(
        RoleArn=config.closure_role_arn,
        RoleSessionName=session_name,
        DurationSeconds=900,
        ExternalId=config.deployment_identity_sha256,
    )
    response_received_at = utc_clock()
    if (
        type(response_received_at) is not datetime
        or response_received_at.tzinfo is None
        or response_received_at < request_started_at
        or type(response) is not dict
        or set(response)
        != {
            "Credentials",
            "AssumedRoleUser",
            "PackedPolicySize",
            "ResponseMetadata",
        }
    ):
        raise RuntimeError("Task 11 AssumeRole response is unauthenticated")
    _authenticated_metadata(
        response["ResponseMetadata"],
        label="Task 11 AssumeRole response",
    )
    credentials = response["Credentials"]
    assumed_role = response["AssumedRoleUser"]
    if (
        type(credentials) is not dict
        or set(credentials)
        != {
            "AccessKeyId",
            "SecretAccessKey",
            "SessionToken",
            "Expiration",
        }
        or any(
            type(credentials.get(field)) is not str
            or not credentials[field]
            for field in (
                "AccessKeyId",
                "SecretAccessKey",
                "SessionToken",
            )
        )
    ):
        raise RuntimeError("Task 11 assumed credential set is invalid")
    expiration = credentials["Expiration"]
    if (
        type(expiration) is not datetime
        or expiration.tzinfo is None
        or expiration
        < request_started_at + timedelta(seconds=895)
        or expiration
        > response_received_at + timedelta(seconds=905)
        or expiration
        < response_received_at + timedelta(seconds=780)
    ):
        raise RuntimeError(
            "Task 11 assumed credential expiration is invalid"
        )
    expected_assumed_arn = (
        "arn:aws:sts::"
        + config.account_id
        + ":assumed-role/"
        + _closure_role_name(config.activation_id)
        + "/"
        + session_name
    )
    if (
        type(assumed_role) is not dict
        or set(assumed_role) != {"AssumedRoleId", "Arn"}
        or assumed_role.get("Arn") != expected_assumed_arn
        or type(assumed_role.get("AssumedRoleId")) is not str
        or not assumed_role["AssumedRoleId"].endswith(
            ":" + session_name
        )
    ):
        raise RuntimeError("Task 11 assumed role identity is invalid")
    if response["PackedPolicySize"] != 0:
        raise RuntimeError("Task 11 session policy boundary drifted")
    session = session_constructor(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=config.region,
    )
    h1d_clients = {
        "cloudformation": session.client(
            "cloudformation",
            config=sdk_config,
        ),
        "lambda": session.client("lambda", config=sdk_config),
        "iam": session.client("iam", config=sdk_config),
        "eventbridge": session.client("events", config=sdk_config),
        "scheduler": session.client("scheduler", config=sdk_config),
        "sqs": session.client("sqs", config=sdk_config),
        "sns": session.client("sns", config=sdk_config),
        "ec2": session.client("ec2", config=sdk_config),
        "stepfunctions": session.client(
            "stepfunctions", config=sdk_config
        ),
        "s3": session.client("s3", config=sdk_config),
        "dynamodb": session.client("dynamodb", config=sdk_config),
        "ssm": session.client("ssm", config=sdk_config),
        "cloudwatch": session.client("cloudwatch", config=sdk_config),
        "logs": session.client("logs", config=sdk_config),
    }
    target_sts = session.client("sts", config=sdk_config)
    caller_identity = target_sts.get_caller_identity()
    if type(caller_identity) is not dict:
        raise RuntimeError("Task 11 target caller identity is invalid")
    _authenticated_metadata(
        caller_identity.get("ResponseMetadata"),
        label="Task 11 target caller identity",
    )
    if (
        caller_identity.get("Account") != config.account_id
        or caller_identity.get("Arn") != expected_assumed_arn
        or caller_identity.get("UserId")
        != assumed_role["AssumedRoleId"]
    ):
        raise RuntimeError("Task 11 target caller identity is invalid")
    return Task11AwsClients(
        s3=h1d_clients["s3"],
        dynamodb=h1d_clients["dynamodb"],
        cloudformation=h1d_clients["cloudformation"],
        lambda_client=h1d_clients["lambda"],
        sts=target_sts,
        h1d_clients=h1d_clients,
        credential_expiration=_rfc3339(expiration),
        caller_identity=dict(caller_identity),
        credential_issue_time=_rfc3339(response_received_at),
        assume_role_request_id=response["ResponseMetadata"]["RequestId"],
        source_caller_identity=dict(source_caller_identity),
        assumed_role_arn=assumed_role["Arn"],
        assumed_role_id=assumed_role["AssumedRoleId"],
    )


def _source_aws_clients() -> Task11AwsClients:
    """Construct the finite no-retry SDK boundary for a 5-second publisher."""

    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        connect_timeout=1,
        read_timeout=4,
        max_pool_connections=3,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.session.Session(region_name="us-west-2")
    return Task11AwsClients(
        s3=session.client("s3", config=config),
        dynamodb=session.client("dynamodb", config=config),
        cloudformation=None,
        lambda_client=None,
        sts=None,
    )


def build_source_publisher_services(
    *,
    source_name: str,
    generation: int,
) -> object:
    """Construct only the accepted Task 3/4/5 publisher dependencies."""

    from .support_source_publisher_handler import (
        SourcePublisherHandlerServices,
    )

    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    function_version_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        + function_name
        + ":"
        + function_version
    )
    expected_function_name = (
        "keep-glm52-h1g-"
        + source_name.removeprefix("Source")
        .replace("Gpu", "gpu-")
        .replace("Submission", "submission-")
        .replace("Controller", "controller-")
        .replace("ControlPlane", "control-plane-")
        .replace("Acquisition", "acquisition")
        .lower()
    )
    # The explicit environment source name and the Lambda version identity
    # must agree.  CloudFormation supplies both from the same stable resource.
    if (
        source_name not in {
            "SourceGpuSpend",
            "SourceSubmissionIntent",
            "SourceControllerBaseline",
            "SourceControlPlaneReadiness",
            "SourceSubmissionAcquisition",
        }
        or function_name != expected_function_name
        or _VERSION_ARN.fullmatch(function_version_arn) is None
        or type(generation) is not int
        or generation <= 0
    ):
        raise RuntimeError("source publisher Lambda identity drifted")
    config = Task11ActionConfig(
        account_id=os.environ.get("GLM52_ACCOUNT_ID", ""),
        region=os.environ.get("AWS_REGION", ""),
        run_id=os.environ.get("GLM52_RUN_ID", ""),
        activation_id=os.environ.get("GLM52_ACTIVATION_ID", ""),
        ledger_table_name=os.environ.get("GLM52_LEDGER_TABLE_NAME", ""),
        campaign_bucket=os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
        deployment_identity_sha256=os.environ.get(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        ),
        function_version_arn=function_version_arn,
    )
    if (
        config.account_id != "246813579024"
        or config.region != "us-west-2"
        or config.run_id != "glm52-sky-20260724"
        or not config.activation_id
        or not config.ledger_table_name
        or not config.campaign_bucket
        or _SHA.fullmatch(config.deployment_identity_sha256) is None
    ):
        raise RuntimeError("source publisher production config drifted")
    clients = _source_aws_clients()
    ledger = DynamoLedgerAdapter(
        client=clients.dynamodb,
        table_name=config.ledger_table_name,
    )
    actions = Task11DynamoActionService(config=config, ledger=ledger)
    h1f = FreshH1fAuditService(
        authority_reader=_fresh_authority_reader(
            config=config,
            clients=clients,
            generation=generation,
        )
    )
    return SourcePublisherHandlerServices(
        s3=clients.s3,
        publication=S3PublicationServices(
            s3=clients.s3,
            actions=actions,
            h1f=h1f,
        ),
    )


def build_effect_writer_services(
    *,
    writer_kind: str,
    generation: int,
) -> object:
    """Construct the bounded Task 3/4 publication path for one fixed writer."""

    from .support_effect_writer_handler import (
        EffectWriterHandlerServices,
    )

    expected_names = {
        "FenceSuccessor": "keep-glm52-h1g-fence-successor",
        "ClaimWriter": "keep-glm52-h1g-claim-writer",
        "DecisionWriter": "keep-glm52-h1g-decision-writer",
        "TerminalV1Writer": "keep-glm52-h1g-terminal-v1-writer",
        "ClosureHandoff": "keep-glm52-h1g-closure-handoff",
    }
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    if (
        writer_kind not in expected_names
        or function_name != expected_names[writer_kind]
        or not function_version.isdigit()
        or function_version.startswith("0")
        or type(generation) is not int
        or generation <= 0
    ):
        raise RuntimeError("effect writer Lambda identity drifted")
    action_config = Task11ActionConfig(
        account_id=os.environ.get("GLM52_ACCOUNT_ID", ""),
        region=os.environ.get("AWS_REGION", ""),
        run_id=os.environ.get("GLM52_RUN_ID", ""),
        activation_id=os.environ.get("GLM52_ACTIVATION_ID", ""),
        ledger_table_name=os.environ.get("GLM52_LEDGER_TABLE_NAME", ""),
        campaign_bucket=os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
        deployment_identity_sha256=os.environ.get(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        ),
        function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            + function_name
            + ":"
            + function_version
        ),
    )
    if (
        action_config.account_id != "246813579024"
        or action_config.region != "us-west-2"
        or action_config.run_id != "glm52-sky-20260724"
        or not action_config.activation_id
        or not action_config.ledger_table_name
        or not action_config.campaign_bucket
        or _SHA.fullmatch(
            action_config.deployment_identity_sha256
        )
        is None
    ):
        raise RuntimeError("effect writer production config drifted")
    clients = _source_aws_clients()
    ledger = DynamoLedgerAdapter(
        client=clients.dynamodb,
        table_name=action_config.ledger_table_name,
    )
    return EffectWriterHandlerServices(
        s3=clients.s3,
        publication=S3PublicationServices(
            s3=clients.s3,
            actions=Task11DynamoActionService(
                config=action_config,
                ledger=ledger,
            ),
            h1f=FreshH1fAuditService(
                authority_reader=_fresh_authority_reader(
                    config=action_config,
                    clients=clients,
                    generation=generation,
                )
            ),
        ),
    )


def build_attestation_handler_services(
    *,
    task9_deployed_identity_coordinate: object,
    task9_deployed_identity_sha256: str,
) -> object:
    """Construct the nonce-only exact mTLS attestation service."""

    from .support_attestation_handler import AttestationHandlerServices
    from .task11_relay_runtime import (
        DynamoFreshnessNonceRegistry,
        PinnedMtlsRelay,
        RelayAttestor,
        aws_relay_runtime_clients,
        load_task9_deployed_identity,
    )

    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    table_name = os.environ.get("GLM52_LEDGER_TABLE_NAME", "")
    activation_id = os.environ.get("GLM52_ACTIVATION_ID", "")
    if (
        function_name != "keep-glm52-h1g-attestation"
        or not function_version.isdigit()
        or function_version.startswith("0")
        or not table_name
        or not activation_id
        or os.environ.get("GLM52_ACCOUNT_ID") != "246813579024"
        or os.environ.get("AWS_REGION") != "us-west-2"
        or os.environ.get("GLM52_RUN_ID") != "glm52-sky-20260724"
    ):
        raise RuntimeError("attestation production config drifted")
    clients = aws_relay_runtime_clients()
    loaded = load_task9_deployed_identity(
        s3=clients.s3,
        coordinate=task9_deployed_identity_coordinate,
        activation_id=activation_id,
        expected_body_sha256=task9_deployed_identity_sha256,
        expected_bucket=os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
    )
    identity = loaded.sky_identity
    return AttestationHandlerServices(
        attestation=SkyAttestationService(
            identity=identity,
            nonce_registry=DynamoFreshnessNonceRegistry(
                dynamodb=clients.dynamodb,
                table_name=table_name,
                activation_id=activation_id,
            ),
            attestor=RelayAttestor(
                PinnedMtlsRelay(
                    secretsmanager=clients.secretsmanager,
                    purpose="ATTESTATION",
                    identity=identity,
                )
            ),
        ),
        task9_deployed_identity_sha256=loaded.body_sha256,
    )


class _Task11FenceAwsClient:
    """Finite authenticated AWS wire adapter for the one fence invocation."""

    _CREATE_WIRE_NAMES = {
        "StackName": "StackName",
        "ChangeSetName": "ChangeSetName",
        "ChangeSetType": "ChangeSetType",
        "TemplateURL": "TemplateURL",
        "RoleARN": "RoleARN",
        "IncludeNestedStacks": "IncludeNestedStacks",
        "ImportExistingResources": "ImportExistingResources",
        "ClientToken": "ClientToken",
    }

    def __init__(
        self,
        *,
        cloudformation: object,
        s3: object,
        iam: object,
    ) -> None:
        self._cloudformation = cloudformation
        self._s3 = s3
        self._iam = iam
        self._create_wire = {}
        try:
            cloudformation.meta.events.register(
                "before-send.cloudformation.CreateChangeSet",
                self._capture_create_wire,
            )
        except Exception as exc:
            raise RuntimeError(
                "fence CreateChangeSet wire observer is unavailable"
            ) from exc

    @staticmethod
    def _metadata(value: object, operation: str) -> Mapping[str, object]:
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
        ):
            raise RuntimeError(
                f"fence {operation} response is unauthenticated"
            )
        return metadata

    def _capture_create_wire(self, request: object, **_: object) -> None:
        raw = getattr(request, "body", None)
        if type(raw) is str:
            raw = raw.encode("ascii")
        if type(raw) is not bytes:
            raise RuntimeError("CreateChangeSet wire body is absent")
        try:
            fields = parse_qs(
                raw.decode("ascii"),
                keep_blank_values=True,
                strict_parsing=True,
            )
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError(
                "CreateChangeSet wire body is malformed"
            ) from exc
        if any(len(values) != 1 for values in fields.values()):
            raise RuntimeError(
                "CreateChangeSet wire fields are ambiguous"
            )
        one = {key: values[0] for key, values in fields.items()}
        change_set_name = one.get("ChangeSetName")
        if (
            type(change_set_name) is not str
            or not change_set_name
            or change_set_name in self._create_wire
        ):
            raise RuntimeError(
                "CreateChangeSet wire observation is not one-use"
            )
        projected: dict[str, object] = {}
        for sdk_name, wire_name in self._CREATE_WIRE_NAMES.items():
            value = one.get(wire_name)
            if value is None:
                raise RuntimeError(
                    "CreateChangeSet wire omitted an exact field"
                )
            projected[sdk_name] = (
                value == "true"
                if sdk_name
                in {"IncludeNestedStacks", "ImportExistingResources"}
                else value
            )
        resource_members = tuple(
            value
            for key, value in sorted(one.items())
            if key.startswith("ResourceTypes.member.")
        )
        if resource_members != ("AWS::S3::BucketPolicy",):
            raise RuntimeError(
                "CreateChangeSet wire resource type drifted"
            )
        projected["ResourceTypes"] = list(resource_members)
        allowed_wire = {
            "Action",
            "Version",
            *self._CREATE_WIRE_NAMES.values(),
            "ResourceTypes.member.1",
        }
        if set(one) != allowed_wire:
            raise RuntimeError(
                "CreateChangeSet wire contains substituted authority"
            )
        self._create_wire[change_set_name] = {
            "request_sha256": canonical_sha256(projected),
            "present_fields": tuple(sorted(projected)),
            "serialized_parameter_members": tuple(
                key
                for key in sorted(one)
                if key.startswith("Parameters.member.")
            ),
            "serialized_capability_members": tuple(
                key
                for key in sorted(one)
                if key.startswith("Capabilities.member.")
            ),
        }

    def observe_create_change_set_wire(
        self,
        *,
        ChangeSetName: str,
    ) -> Mapping[str, object]:
        value = self._create_wire.pop(ChangeSetName, None)
        if type(value) is not dict:
            raise RuntimeError(
                "CreateChangeSet wire evidence is absent or replayed"
            )
        return value

    def get_template_object(
        self,
        *,
        TemplateURL: str,
        VersionId: str,
    ) -> Mapping[str, object]:
        parsed = urlparse(TemplateURL)
        hostname = parsed.hostname or ""
        suffix = ".s3.us-west-2.amazonaws.com"
        bucket = hostname[: -len(suffix)] if hostname.endswith(suffix) else ""
        query = parse_qs(parsed.query, keep_blank_values=True)
        if (
            not bucket
            or not parsed.path.startswith("/")
            or query != {"versionId": [VersionId]}
        ):
            raise RuntimeError("immutable fence template URL drifted")
        response = self._s3.get_object(
            Bucket=bucket,
            Key=unquote(parsed.path[1:]),
            VersionId=VersionId,
            ExpectedBucketOwner="246813579024",
            ChecksumMode="ENABLED",
        )
        metadata = self._metadata(response, "GetObject")
        body = response.get("Body")
        read = getattr(body, "read", None)
        if (
            response.get("VersionId") != VersionId
            or not callable(read)
        ):
            raise RuntimeError(
                "immutable fence template readback drifted"
            )
        raw = read(256 * 1024 + 1)
        if type(raw) is not bytes or len(raw) > 256 * 1024:
            raise RuntimeError(
                "immutable fence template body is invalid"
            )
        return {
            "Body": raw,
            "VersionId": VersionId,
            "ResponseMetadata": dict(metadata),
        }

    def _describe_until(
        self,
        *,
        change_set_name: str,
        execute: bool,
        deadline: Optional[float] = None,
    ) -> Mapping[str, object]:
        phase_deadline = (
            time.monotonic() + 6.5
            if deadline is None
            else deadline
        )
        while True:
            response = self._cloudformation.describe_change_set(
                ChangeSetName=change_set_name,
                IncludePropertyValues=False,
            )
            self._metadata(response, "DescribeChangeSet")
            status = response.get("Status")
            execution = response.get("ExecutionStatus")
            if (
                status == "CREATE_COMPLETE"
                and (
                    (not execute and execution == "AVAILABLE")
                    or (execute and execution == "EXECUTE_COMPLETE")
                )
            ):
                return response
            if status in {"FAILED", "DELETE_COMPLETE"}:
                raise RuntimeError(
                    "fence change set reached a failed state"
                )
            if time.monotonic() >= phase_deadline:
                raise RuntimeError(
                    "fence change set polling exceeded phase deadline"
                )
            time.sleep(0.2)

    def create_change_set(
        self,
        **request: object,
    ) -> Mapping[str, object]:
        response = self._cloudformation.create_change_set(**request)
        self._metadata(response, "CreateChangeSet")
        change_set_arn = response.get("Id")
        if type(change_set_arn) is not str:
            raise RuntimeError("CreateChangeSet omitted its exact ARN")
        self._describe_until(
            change_set_name=change_set_arn,
            execute=False,
        )
        return response

    def list_change_sets(
        self, **request: object
    ) -> Mapping[str, object]:
        response = self._cloudformation.list_change_sets(**request)
        self._metadata(response, "ListChangeSets")
        return response

    def describe_change_set(
        self,
        **request: object,
    ) -> Mapping[str, object]:
        response = self._cloudformation.describe_change_set(
            **request,
            IncludePropertyValues=False,
        )
        self._metadata(response, "DescribeChangeSet")
        return response

    def get_template(self, **request: object) -> Mapping[str, object]:
        response = self._cloudformation.get_template(**request)
        self._metadata(response, "GetTemplate")
        body = response.get("TemplateBody")
        if type(body) is str:
            try:
                body = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "CloudFormation template body is not JSON"
                ) from exc
            response = {**response, "TemplateBody": body}
        return response

    def describe_stacks(
        self,
        **request: object,
    ) -> Mapping[str, object]:
        response = self._cloudformation.describe_stacks(**request)
        self._metadata(response, "DescribeStacks")
        stacks = response.get("Stacks")
        if type(stacks) is list:
            normalized = []
            for value in stacks:
                if type(value) is not dict:
                    normalized.append(value)
                    continue
                updated = value.get("LastUpdatedTime")
                normalized.append(
                    {
                        **value,
                        "LastUpdatedTime": (
                            _rfc3339(updated)
                            if isinstance(updated, datetime)
                            else updated
                        ),
                    }
                )
            response = {**response, "Stacks": normalized}
        return response

    def execute_change_set(
        self,
        **request: object,
    ) -> Mapping[str, object]:
        response = self._cloudformation.execute_change_set(**request)
        self._metadata(response, "ExecuteChangeSet")
        deadline = time.monotonic() + 6.5
        self._describe_until(
            change_set_name=str(request["ChangeSetName"]),
            execute=True,
            deadline=deadline,
        )
        while True:
            stack = self._cloudformation.describe_stacks(
                StackName=request["StackName"],
            )
            self._metadata(stack, "DescribeStacks")
            values = stack.get("Stacks")
            status = (
                values[0].get("StackStatus")
                if type(values) is list
                and len(values) == 1
                and type(values[0]) is dict
                else None
            )
            if status == "UPDATE_COMPLETE":
                break
            if (
                type(status) is not str
                or status.endswith("_FAILED")
                or "ROLLBACK" in status
            ):
                raise RuntimeError(
                    "fence stack reached a failed update state"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "fence stack polling exceeded phase deadline"
                )
            time.sleep(0.2)
        return response

    def get_bucket_policy(
        self,
        **request: object,
    ) -> Mapping[str, object]:
        response = self._s3.get_bucket_policy(**request)
        self._metadata(response, "GetBucketPolicy")
        return response

    def observe_fence_poststate(
        self, **request: object
    ) -> Mapping[str, object]:
        stack_id = request.get("StackName")

        def observe_pass() -> tuple[
            dict[str, object], str, dict[str, str]
        ]:
            stack_response = self._cloudformation.describe_stacks(
                StackName=stack_id
            )
            stack_metadata = self._metadata(
                stack_response, "DescribeStacks"
            )
            stacks = stack_response.get("Stacks")
            if (
                type(stacks) is not list
                or len(stacks) != 1
                or type(stacks[0]) is not dict
            ):
                raise RuntimeError(
                    "fence poststate stack is not singular"
                )
            stack = stacks[0]
            role_arn = stack.get("RoleARN")
            if type(role_arn) is not str or not role_arn:
                raise RuntimeError("fence poststate stack role is absent")
            role_name = role_arn.rsplit("/", 1)[-1]
            role_response = self._iam.get_role(RoleName=role_name)
            role_metadata = self._metadata(role_response, "GetRole")
            role = role_response.get("Role")
            if type(role) is not dict:
                raise RuntimeError("fence poststate role is absent")

            stack_policy = self._cloudformation.get_stack_policy(
                StackName=stack_id
            )
            stack_policy_metadata = self._metadata(
                stack_policy, "GetStackPolicy"
            )
            policy_body = stack_policy.get("StackPolicyBody")
            if type(policy_body) is str:
                policy_body = json.loads(policy_body)

            original = self._cloudformation.get_template(
                StackName=stack_id,
                TemplateStage="Original",
            )
            original_metadata = self._metadata(
                original, "GetTemplateOriginal"
            )
            processed = self._cloudformation.get_template(
                StackName=stack_id,
                TemplateStage="Processed",
            )
            processed_metadata = self._metadata(
                processed, "GetTemplateProcessed"
            )
            original_body = original.get("TemplateBody")
            processed_body = processed.get("TemplateBody")
            if type(original_body) is str:
                original_body = json.loads(original_body)
            if type(processed_body) is str:
                processed_body = json.loads(processed_body)

            policy_response = self._s3.get_bucket_policy(
                Bucket="keep-glm52-models-246813579024-us-west-2",
                ExpectedBucketOwner="246813579024",
            )
            policy_metadata = self._metadata(
                policy_response, "GetBucketPolicy"
            )
            direct_policy = policy_response.get("Policy")
            if type(direct_policy) is not str:
                raise RuntimeError(
                    "fence poststate direct policy is absent"
                )
            try:
                direct_policy_bytes = direct_policy.encode("ascii")
            except UnicodeEncodeError as exc:
                raise RuntimeError(
                    "fence poststate direct policy is not ASCII"
                ) from exc

            snapshot = {
                "stack_id": stack.get("StackId"),
                "stack_status": stack.get("StackStatus"),
                "stack_role_arn": role_arn,
                "stack_role_id": role.get("RoleId"),
                "termination_protection": (
                    stack.get("EnableTerminationProtection")
                ),
                "stack_policy_sha256": canonical_sha256(policy_body),
                "original_template_body_sha256": canonical_sha256(
                    original_body
                ),
                "processed_template_body_sha256": canonical_sha256(
                    processed_body
                ),
                "direct_policy_sha256": hashlib.sha256(
                    direct_policy_bytes
                ).hexdigest(),
            }
            request_ids = {
                "stack": stack_metadata["RequestId"],
                "role": role_metadata["RequestId"],
                "stack_policy": stack_policy_metadata["RequestId"],
                "original_template": original_metadata["RequestId"],
                "processed_template": processed_metadata["RequestId"],
                "direct_policy": policy_metadata["RequestId"],
            }
            if len(set(request_ids.values())) != len(request_ids):
                raise RuntimeError(
                    "fence poststate observation reused a request ID"
                )
            return snapshot, direct_policy, request_ids

        first_snapshot, first_policy, first_request_ids = observe_pass()
        second_snapshot, second_policy, second_request_ids = observe_pass()
        if not set(first_request_ids.values()).isdisjoint(
            second_request_ids.values()
        ):
            raise RuntimeError(
                "fence poststate observations reused a request ID"
            )
        first_observation = {
            **first_snapshot,
            "direct_policy": first_policy,
        }
        second_observation = {
            **second_snapshot,
            "direct_policy": second_policy,
        }
        if canonical_json_bytes(first_observation) != canonical_json_bytes(
            second_observation
        ):
            raise RuntimeError(
                "fence poststate observations are not exactly equal"
            )

        snapshot_identity = canonical_sha256(first_snapshot)
        return {
            **first_snapshot,
            "direct_policy": first_policy,
            "first_stable_snapshot_identity_sha256": snapshot_identity,
            "second_stable_snapshot_identity_sha256": canonical_sha256(
                second_snapshot
            ),
            "stabilization_first_evidence_sha256": canonical_sha256(
                first_request_ids
            ),
            "stabilization_second_evidence_sha256": canonical_sha256(
                second_request_ids
            ),
            "stabilization_identity_sha256": canonical_sha256(
                {
                    "snapshot_identity_sha256": snapshot_identity,
                    "logical_id": request.get("LogicalResourceId"),
                }
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": second_request_ids["processed_template"],
            },
        }


class _Task11FenceH1fService:
    """Full live fence/IAM/principal/denial stabilization evidence."""

    def __init__(
        self,
        *,
        s3: object,
        iam: object,
        sts: object,
        campaign_bucket: str,
        authority_reader: Callable[[], H1fAuthoritySnapshot],
        assumed_s3_factory: Callable[..., object],
        executor_role_arn: str,
        writer_role_arn: str,
        service_role_arn: str,
        runtime_deadline: float,
        audit_request_ids: Optional[list[str]] = None,
    ) -> None:
        self._s3 = s3
        self._iam = iam
        self._sts = sts
        self._campaign_bucket = campaign_bucket
        self._authority_reader = authority_reader
        self._assumed_s3_factory = assumed_s3_factory
        self._executor_role_arn = executor_role_arn
        self._writer_role_arn = writer_role_arn
        self._service_role_arn = service_role_arn
        self._runtime_deadline = runtime_deadline
        self._audit_request_ids = (
            [] if audit_request_ids is None else audit_request_ids
        )

    def _audit(self, request: object) -> H1fAuditResult:
        return audit_current_head(
            s3=self._s3,
            snapshot=self._authority_reader(),
            operation_kind=request.operation_kind,
            action_key=request.action_key,
            candidate_identity_sha256=(
                request.candidate_identity_sha256
            ),
        )

    def fresh_audit(self, *, request: object) -> H1fAuditResult:
        return self._audit(request)

    @staticmethod
    def _request_id(value: object, operation: str) -> str:
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        request_id = (
            metadata.get("RequestId")
            if type(metadata) is dict
            else None
        )
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(request_id) is not str
            or not request_id
        ):
            raise RuntimeError(operation + " is unauthenticated")
        return request_id

    def _policy_evidence(self) -> tuple[str, str]:
        response = self._s3.get_bucket_policy(
            Bucket=self._campaign_bucket,
            ExpectedBucketOwner="246813579024",
        )
        request_id = self._request_id(
            response,
            "GetBucketPolicy",
        )
        raw = response.get("Policy")
        if type(raw) is not str:
            raise RuntimeError("live fence policy body is absent")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("live fence policy is invalid JSON") from exc
        if type(value) is not dict:
            raise RuntimeError("live fence policy is not an object")
        canonical = canonical_json_bytes(value)
        return hashlib.sha256(canonical).hexdigest(), request_id

    def _selected_sources(
        self,
        *,
        audit: H1fAuditResult,
    ) -> tuple[list[Mapping[str, object]], str]:
        response = self._s3.get_object(
            Bucket=self._campaign_bucket,
            Key=audit.active_head_key,
            VersionId=audit.active_head_version_id,
            ExpectedBucketOwner="246813579024",
            ChecksumMode="ENABLED",
        )
        request_id = self._request_id(
            response,
            "GetObject active H1f head",
        )
        body = response.get("Body")
        read = getattr(body, "read", None)
        if (
            response.get("VersionId")
            != audit.active_head_version_id
            or not callable(read)
        ):
            raise RuntimeError("active H1f head readback drifted")
        raw = read(2 * 1024 * 1024 + 1)
        if (
            type(raw) is not bytes
            or len(raw) > 2 * 1024 * 1024
            or hashlib.sha256(raw).hexdigest()
            != audit.active_head_file_sha256
            or not raw.endswith(b"\n")
        ):
            raise RuntimeError("active H1f head bytes drifted")
        try:
            document = json.loads(raw[:-1].decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("active H1f head JSON drifted") from exc
        sources = (
            document.get("enrolled_sources")
            if type(document) is dict
            else None
        )
        source_families = (
            "gpu-spend-snapshot",
            "production-submission-intent",
            "production-controller-baseline",
            "production-control-plane-readiness",
            "production-submission-acquisition",
        )
        selected = [
            source
            for source in sources
            if type(source) is dict
            and source.get("source_kind") in source_families
        ] if type(sources) is list else []
        if (
            len(selected) != len(source_families)
            or tuple(
                sorted(source["source_kind"] for source in selected)
            )
            != tuple(sorted(source_families))
            or len(
                {
                    source.get("publisher_principal_arn")
                    for source in selected
                }
            )
            != len(source_families)
            or any(
                source.get("publisher_state")
                != "explicitly-denied"
                or type(source.get("key")) is not str
                or not source["key"].startswith(
                    "campaigns/glm52-sky-20260724/"
                )
                or type(source.get("version_id")) is not str
                or not source["version_id"]
                or type(source.get("publisher_principal_id"))
                is not str
                or not source["publisher_principal_id"]
                for source in selected
            )
        ):
            raise RuntimeError(
                "active H1f source/publisher resolution drifted"
            )
        return (
            sorted(selected, key=lambda item: item["source_kind"]),
            request_id,
        )

    def _role_inventory(
        self,
        role_arns: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...], Mapping[str, str]]:
        inventory = []
        request_ids: list[str] = []
        role_ids: dict[str, str] = {}
        for role_arn in sorted(role_arns):
            role_name = role_arn.rsplit("/", 1)[-1]
            role_response = self._iam.get_role(RoleName=role_name)
            request_ids.append(
                self._request_id(role_response, "GetRole")
            )
            role = role_response.get("Role")
            if (
                type(role) is not dict
                or role.get("Arn") != role_arn
                or type(role.get("RoleId")) is not str
                or not role["RoleId"]
                or type(role.get("AssumeRolePolicyDocument")) is not dict
            ):
                raise RuntimeError("live IAM role identity drifted")
            role_ids[role_arn] = role["RoleId"]
            inline_names, inline_ids = self._iam_names(
                method="list_role_policies",
                result_field="PolicyNames",
                RoleName=role_name,
            )
            request_ids.extend(inline_ids)
            inline = {}
            for policy_name in inline_names:
                response = self._iam.get_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name,
                )
                request_ids.append(
                    self._request_id(
                        response,
                        "GetRolePolicy",
                    )
                )
                if (
                    response.get("RoleName") != role_name
                    or response.get("PolicyName") != policy_name
                    or type(response.get("PolicyDocument")) is not dict
                ):
                    raise RuntimeError(
                        "live inline IAM policy drifted"
                    )
                inline[policy_name] = canonical_sha256(
                    response["PolicyDocument"]
                )
            attached, attached_ids = self._iam_names(
                method="list_attached_role_policies",
                result_field="AttachedPolicies",
                RoleName=role_name,
            )
            request_ids.extend(attached_ids)
            managed = []
            for attached_policy in attached:
                if (
                    type(attached_policy) is not dict
                    or type(attached_policy.get("PolicyArn")) is not str
                    or not attached_policy["PolicyArn"]
                ):
                    raise RuntimeError(
                        "live attached IAM policy drifted"
                    )
                policy_response = self._iam.get_policy(
                    PolicyArn=attached_policy["PolicyArn"]
                )
                request_ids.append(
                    self._request_id(policy_response, "GetPolicy")
                )
                policy = policy_response.get("Policy")
                version_id = (
                    policy.get("DefaultVersionId")
                    if type(policy) is dict
                    else None
                )
                if type(version_id) is not str or not version_id:
                    raise RuntimeError(
                        "live managed IAM policy version drifted"
                    )
                version_response = self._iam.get_policy_version(
                    PolicyArn=attached_policy["PolicyArn"],
                    VersionId=version_id,
                )
                request_ids.append(
                    self._request_id(
                        version_response,
                        "GetPolicyVersion",
                    )
                )
                version = version_response.get("PolicyVersion")
                if (
                    type(version) is not dict
                    or type(version.get("Document")) is not dict
                ):
                    raise RuntimeError(
                        "live managed IAM policy document drifted"
                    )
                managed.append(
                    {
                        "policy_arn": attached_policy["PolicyArn"],
                        "version_id": version_id,
                        "document_sha256": canonical_sha256(
                            version["Document"]
                        ),
                    }
                )
            inventory.append(
                {
                    "role_arn": role_arn,
                    "role_id": role["RoleId"],
                    "trust_sha256": canonical_sha256(
                        role["AssumeRolePolicyDocument"]
                    ),
                    "inline_policy_sha256": dict(sorted(inline.items())),
                    "managed_policy_versions": sorted(
                        managed,
                        key=lambda item: item["policy_arn"],
                    ),
                }
            )
        return (
            canonical_sha256(inventory),
            tuple(request_ids),
            role_ids,
        )

    def _iam_names(
        self,
        *,
        method: str,
        result_field: str,
        RoleName: str,
    ) -> tuple[list[object], tuple[str, ...]]:
        operation = getattr(self._iam, method, None)
        if not callable(operation):
            raise RuntimeError("live IAM inventory method is absent")
        marker = None
        seen = set()
        values: list[object] = []
        request_ids = []
        for _page in range(64):
            request = {"RoleName": RoleName}
            if marker is not None:
                request["Marker"] = marker
            response = operation(**request)
            request_ids.append(
                self._request_id(response, method)
            )
            page_values = response.get(result_field)
            if type(page_values) is not list:
                raise RuntimeError("live IAM inventory page drifted")
            values.extend(page_values)
            if response.get("IsTruncated") is False:
                return values, tuple(request_ids)
            marker = response.get("Marker")
            if (
                type(marker) is not str
                or not marker
                or marker in seen
            ):
                raise RuntimeError(
                    "live IAM inventory pagination drifted"
                )
            seen.add(marker)
        raise RuntimeError("live IAM inventory exceeded page bound")

    def _assume_and_probe(
        self,
        source: Mapping[str, object],
    ) -> tuple[
        str,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        role_arn = source["publisher_principal_arn"]
        response = self._sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="h1g-fence-stabilization",
            DurationSeconds=900,
        )
        request_ids = [
            self._request_id(response, "AssumeRole")
        ]
        credentials = response.get("Credentials")
        assumed = response.get("AssumedRoleUser")
        expected_arn = (
            "arn:aws:sts::246813579024:assumed-role/"
            + role_arn.rsplit("/", 1)[-1]
            + "/h1g-fence-stabilization"
        )
        if (
            type(credentials) is not dict
            or any(
                type(credentials.get(field)) is not str
                or not credentials[field]
                for field in (
                    "AccessKeyId",
                    "SecretAccessKey",
                    "SessionToken",
                )
            )
            or type(assumed) is not dict
            or assumed.get("Arn") != expected_arn
            or type(assumed.get("AssumedRoleId")) is not str
            or not assumed["AssumedRoleId"].startswith(
                source["publisher_principal_id"] + ":"
            )
        ):
            raise RuntimeError(
                "stable publisher assumed identity drifted"
            )
        assumed_s3 = self._assumed_s3_factory(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        key = source["key"]
        version_id = source["version_id"]
        operations = (
            (
                "PutObject",
                "put_object",
                {
                    "Bucket": self._campaign_bucket,
                    "Key": key,
                    "Body": b"forbidden-stabilization-probe",
                    "ExpectedBucketOwner": "246813579024",
                },
            ),
            (
                "DeleteObjectVersion",
                "delete_object",
                {
                    "Bucket": self._campaign_bucket,
                    "Key": key,
                    "VersionId": version_id,
                    "ExpectedBucketOwner": "246813579024",
                },
            ),
            (
                "CopyObject",
                "copy_object",
                {
                    "Bucket": self._campaign_bucket,
                    "Key": key,
                    "CopySource": {
                        "Bucket": self._campaign_bucket,
                        "Key": key,
                        "VersionId": version_id,
                    },
                    "ExpectedBucketOwner": "246813579024",
                    "ExpectedSourceBucketOwner": "246813579024",
                },
            ),
            (
                "CreateMultipartUpload",
                "create_multipart_upload",
                {
                    "Bucket": self._campaign_bucket,
                    "Key": key,
                    "ExpectedBucketOwner": "246813579024",
                },
            ),
            (
                "PutObjectTagging",
                "put_object_tagging",
                {
                    "Bucket": self._campaign_bucket,
                    "Key": key,
                    "VersionId": version_id,
                    "Tagging": {"TagSet": []},
                    "ExpectedBucketOwner": "246813579024",
                },
            ),
        )
        error_codes = []
        host_ids = []
        for operation_name, method_name, request in operations:
            method = getattr(assumed_s3, method_name, None)
            if not callable(method):
                raise RuntimeError(
                    "stable publisher probe method is absent"
                )
            try:
                method(**request)
            except Exception as exc:
                error = getattr(exc, "response", None)
            else:
                raise RuntimeError(
                    operation_name + " denial probe succeeded"
                )
            metadata = (
                error.get("ResponseMetadata")
                if type(error) is dict
                else None
            )
            error_value = (
                error.get("Error") if type(error) is dict else None
            )
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 403
                or type(metadata.get("RequestId")) is not str
                or not metadata["RequestId"]
                or type(metadata.get("HostId")) is not str
                or not metadata["HostId"]
                or type(error_value) is not dict
                or error_value.get("Code") != "AccessDenied"
            ):
                raise RuntimeError(
                    operation_name + " denial probe drifted"
                )
            request_ids.append(metadata["RequestId"])
            host_ids.append(metadata["HostId"])
            error_codes.append("AccessDenied")
        return (
            expected_arn,
            tuple(request_ids),
            tuple(host_ids),
            tuple(error_codes),
        )

    def fresh_post_policy_stabilization(
        self,
        *,
        request: object,
        policy_sha256: str,
    ) -> object:
        from .fence_executor import FencePostPolicyProof

        def observation() -> Mapping[str, object]:
            observed = datetime.now(timezone.utc).replace(
                microsecond=0
            )
            self._audit_request_ids.clear()
            audit = self._audit(request)
            audit_request_ids = tuple(self._audit_request_ids)
            self._audit_request_ids.clear()
            if not audit_request_ids:
                raise RuntimeError(
                    "H1f audit service request evidence is absent"
                )
            observed_policy, policy_request_id = (
                self._policy_evidence()
            )
            if observed_policy != policy_sha256:
                raise RuntimeError(
                    "live fence policy read drifted"
                )
            sources, source_request_id = self._selected_sources(
                audit=audit,
            )
            publisher_roles = tuple(
                source["publisher_principal_arn"]
                for source in sources
            )
            role_arns = tuple(
                sorted(
                    {
                        *publisher_roles,
                        self._executor_role_arn,
                        self._writer_role_arn,
                        self._service_role_arn,
                    }
                )
            )
            (
                role_inventory_sha256,
                iam_request_ids,
                role_ids,
            ) = self._role_inventory(role_arns)
            stable_source = sources[0]
            stable_role = stable_source[
                "publisher_principal_arn"
            ]
            stable_principal = stable_source[
                "publisher_principal_id"
            ]
            if role_ids.get(stable_role) != stable_principal:
                raise RuntimeError(
                    "stable publisher IAM identity drifted"
                )
            (
                assumed_arn,
                assumed_request_ids,
                probe_host_ids,
                probe_error_codes,
            ) = self._assume_and_probe(stable_source)
            service_request_ids = (
                *audit_request_ids,
                policy_request_id,
                source_request_id,
                *iam_request_ids,
                *assumed_request_ids,
            )
            self._audit_request_ids.clear()
            stable_body = {
                "active_head_key": audit.active_head_key,
                "active_head_version_id": (
                    audit.active_head_version_id
                ),
                "active_head_file_sha256": (
                    audit.active_head_file_sha256
                ),
                "active_head_body_sha256": (
                    audit.active_head_body_sha256
                ),
                "policy_sha256": observed_policy,
                "iam_role_inventory_sha256": (
                    role_inventory_sha256
                ),
                "stable_publisher_role_arn": stable_role,
                "stable_publisher_principal_id": stable_principal,
                "stable_publisher_assumed_arn": assumed_arn,
                "denial_probe_operations": (
                    "PutObject",
                    "DeleteObjectVersion",
                    "CopyObject",
                    "CreateMultipartUpload",
                    "PutObjectTagging",
                ),
                "denial_probe_error_codes": probe_error_codes,
                "denial_probe_attempt_counts": (1,) * 5,
            }
            stable_identity = canonical_sha256(stable_body)
            evidence_body = {
                **stable_body,
                "observed_at": _rfc3339(observed),
                "service_request_ids": service_request_ids,
                "denial_probe_host_ids": probe_host_ids,
            }
            return {
                "audit": audit,
                "observed_at": observed,
                "stable_identity": stable_identity,
                "evidence_identity": canonical_sha256(
                    evidence_body
                ),
                "iam_role_inventory_sha256": (
                    role_inventory_sha256
                ),
                "stable_role": stable_role,
                "stable_principal": stable_principal,
                "assumed_arn": assumed_arn,
                "service_request_ids": service_request_ids,
                "probe_host_ids": probe_host_ids,
                "probe_error_codes": probe_error_codes,
            }

        first_started = time.monotonic()
        if first_started + 10.75 >= self._runtime_deadline:
            raise RuntimeError(
                "fence stabilization cannot fit frozen runtime"
            )
        first_value = observation()
        target = first_started + 10.0
        while time.monotonic() < target:
            now = time.monotonic()
            if now + 0.75 >= self._runtime_deadline:
                raise RuntimeError(
                    "fence stabilization exhausted runtime reserve"
                )
            time.sleep(
                min(
                    0.2,
                    target - now,
                    self._runtime_deadline - now - 0.75,
                )
            )
        second_value = observation()
        first = first_value["observed_at"]
        second = second_value["observed_at"]
        separation = int((second - first).total_seconds())
        if (
            separation < 10
            or separation > 11
            or first_value["stable_identity"]
            != second_value["stable_identity"]
            or first_value["iam_role_inventory_sha256"]
            != second_value["iam_role_inventory_sha256"]
            or first_value["stable_role"]
            != second_value["stable_role"]
            or first_value["stable_principal"]
            != second_value["stable_principal"]
            or first_value["assumed_arn"]
            != second_value["assumed_arn"]
        ):
            raise RuntimeError(
                "post-policy stabilization sets diverged"
            )
        self._audit_request_ids.clear()
        final_audit = self._audit(request)
        final_audit_request_ids = tuple(self._audit_request_ids)
        self._audit_request_ids.clear()
        if not final_audit_request_ids:
            raise RuntimeError(
                "final H1f audit service evidence is absent"
            )
        if any(
            getattr(second_value["audit"], field)
            != getattr(final_audit, field)
            for field in (
                "active_head_key",
                "active_head_version_id",
                "active_head_file_sha256",
                "active_head_body_sha256",
            )
        ):
            raise RuntimeError("final H1f head drifted")
        service_request_ids = tuple(
            first_value["service_request_ids"]
            + second_value["service_request_ids"]
            + final_audit_request_ids
        )
        probe_host_ids = tuple(
            first_value["probe_host_ids"]
            + second_value["probe_host_ids"]
        )
        if (
            len(set(service_request_ids))
            != len(service_request_ids)
            or len(set(probe_host_ids)) != len(probe_host_ids)
        ):
            raise RuntimeError(
                "post-policy service evidence was reused"
            )
        return FencePostPolicyProof(
            audit=second_value["audit"],
            final_audit=final_audit,
            policy_sha256=policy_sha256,
            observation_count=2,
            separation_seconds=separation,
            unique_zero_child=True,
            first_observed_at=_rfc3339(first),
            second_observed_at=_rfc3339(second),
            first_evidence_sha256=first_value[
                "evidence_identity"
            ],
            second_evidence_sha256=second_value[
                "evidence_identity"
            ],
            first_stable_observation_sha256=first_value[
                "stable_identity"
            ],
            second_stable_observation_sha256=second_value[
                "stable_identity"
            ],
            stable_observation_sha256=first_value[
                "stable_identity"
            ],
            iam_role_inventory_sha256=first_value[
                "iam_role_inventory_sha256"
            ],
            stable_publisher_role_arn=first_value[
                "stable_role"
            ],
            stable_publisher_principal_id=first_value[
                "stable_principal"
            ],
            stable_publisher_assumed_arn=first_value[
                "assumed_arn"
            ],
            denial_probe_operations=(
                (
                    "PutObject",
                    "DeleteObjectVersion",
                    "CopyObject",
                    "CreateMultipartUpload",
                    "PutObjectTagging",
                )
                * 2
            ),
            denial_probe_error_codes=(
                first_value["probe_error_codes"]
                + second_value["probe_error_codes"]
            ),
            denial_probe_attempt_counts=(1,) * 10,
            denial_probe_host_ids=probe_host_ids,
            service_request_ids=service_request_ids,
        )


def build_fence_handler_services(
    *,
    generation: int,
    remaining_time_millis: Optional[int] = None,
) -> object:
    """Construct the coordinate-only Task-6 v2 fence handler boundary."""

    from .fence_executor import DynamoFenceRecordStore, FenceExecutor
    from .fence_artifacts import ExecutorAuthorityClass
    from .support_fence_handler import FenceHandlerServices

    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    table_name = os.environ.get("GLM52_LEDGER_TABLE_NAME", "")
    authority_name = os.environ.get(
        "GLM52_FENCE_EXECUTOR_AUTHORITY_CLASS", ""
    )
    function_version_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        + function_name
        + ":"
        + function_version
    )
    try:
        authority_class = ExecutorAuthorityClass(authority_name)
    except ValueError as exc:
        raise RuntimeError("fence executor authority class drifted") from exc
    if (
        function_name not in {
            "keep-glm52-h1g-fence-executor",
            "keep-glm52-h1g-pre-support-fence-executor",
        }
        or _VERSION_ARN.fullmatch(function_version_arn) is None
        or not table_name
        or type(generation) is not int
        or generation != 1
        or type(remaining_time_millis) is not int
        or not 0 < remaining_time_millis <= 12_000
        or os.environ.get("GLM52_ACCOUNT_ID") != "246813579024"
        or os.environ.get("AWS_REGION") != "us-west-2"
        or os.environ.get("GLM52_RUN_ID") != "glm52-sky-20260724"
    ):
        raise RuntimeError("fence production config drifted")
    sdk = Config(
        connect_timeout=1,
        read_timeout=4,
        max_pool_connections=4,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.session.Session(region_name="us-west-2")
    s3 = session.client("s3", config=sdk)
    dynamodb = session.client("dynamodb", config=sdk)
    cloudformation = session.client("cloudformation", config=sdk)
    iam = session.client("iam", config=sdk)
    live_reader = None
    if authority_class is ExecutorAuthorityClass.SUPPORT_RUNTIME:
        lambda_client = session.client("lambda", config=sdk)

        def live_reader() -> Mapping[str, object]:
            configuration = lambda_client.get_function_configuration(
                FunctionName=function_version_arn
            )
            role_arn = configuration.get("Role")
            role_name = (
                role_arn.rsplit("/", 1)[-1]
                if type(role_arn) is str
                else ""
            )
            role = iam.get_role(RoleName=role_name).get("Role", {})
            if type(role) is not dict:
                raise RuntimeError("support runtime role readback drifted")
            environment = configuration.get("Environment", {})
            variables = (
                environment.get("Variables", {})
                if type(environment) is dict
                else {}
            )
            if type(variables) is not dict:
                raise RuntimeError(
                    "support runtime environment readback drifted"
                )
            inline = iam.list_role_policies(RoleName=role_name)
            inline_names = inline.get("PolicyNames")
            attached = iam.list_attached_role_policies(RoleName=role_name)
            attached_rows = attached.get("AttachedPolicies")
            if (
                type(inline_names) is not list
                or inline.get("IsTruncated") is not False
                or type(attached_rows) is not list
                or attached.get("IsTruncated") is not False
            ):
                raise RuntimeError(
                    "support runtime permission inventory is not bounded"
                )
            permission_inventory = {
                "inline": [
                    {
                        "name": name,
                        "document": iam.get_role_policy(
                            RoleName=role_name,
                            PolicyName=name,
                        ).get("PolicyDocument"),
                    }
                    for name in sorted(inline_names)
                ],
                "attached": [],
            }
            for attached_row in sorted(
                attached_rows,
                key=lambda row: str(row.get("PolicyArn")),
            ):
                policy_arn = attached_row.get("PolicyArn")
                policy = iam.get_policy(
                    PolicyArn=policy_arn
                ).get("Policy", {})
                version_id = (
                    policy.get("DefaultVersionId")
                    if type(policy) is dict
                    else None
                )
                permission_inventory["attached"].append(
                    {
                        "arn": policy_arn,
                        "version_id": version_id,
                        "document": iam.get_policy_version(
                            PolicyArn=policy_arn,
                            VersionId=version_id,
                        ).get("PolicyVersion", {}).get("Document"),
                    }
                )
            support_stack = variables.get("GLM52_SUPPORT_STACK_ID")
            stacks = cloudformation.describe_stacks(
                StackName=support_stack
            ).get("Stacks")
            if (
                type(stacks) is not list
                or len(stacks) != 1
                or type(stacks[0]) is not dict
            ):
                raise RuntimeError("support runtime stack readback drifted")
            stack_id = stacks[0].get("StackId")
            template = cloudformation.get_template(
                StackName=stack_id,
                TemplateStage="Original",
            ).get("TemplateBody")
            if type(template) is str:
                template = json.loads(template)
            try:
                resource_policy = json.loads(
                    lambda_client.get_policy(
                        FunctionName=function_version_arn
                    )["Policy"]
                )
            except lambda_client.exceptions.ResourceNotFoundException:
                resource_policy = {"absent": True}
            mappings: list[Mapping[str, object]] = []
            marker = None
            while True:
                request = {"FunctionName": function_version_arn}
                if marker is not None:
                    request["Marker"] = marker
                response = lambda_client.list_event_source_mappings(
                    **request
                )
                page = response.get("EventSourceMappings")
                if type(page) is not list:
                    raise RuntimeError(
                        "support event-source inventory drifted"
                    )
                mappings.extend(
                    {
                        key: row.get(key)
                        for key in (
                            "UUID",
                            "EventSourceArn",
                            "FunctionArn",
                            "State",
                        )
                    }
                    for row in page
                )
                marker = response.get("NextMarker")
                if marker is None:
                    break
                if type(marker) is not str or len(mappings) > 16:
                    raise RuntimeError(
                        "support event-source inventory is unbounded"
                    )
            try:
                function_url = lambda_client.get_function_url_config(
                    FunctionName=function_version_arn
                )
                function_url_state = {
                    key: function_url.get(key)
                    for key in ("AuthType", "InvokeMode", "FunctionUrl")
                }
            except lambda_client.exceptions.ResourceNotFoundException:
                function_url_state = {"absent": True}
            code_sha = configuration.get("CodeSha256")
            try:
                code_sha256 = base64.b64decode(
                    code_sha, validate=True
                ).hex()
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "support runtime code identity drifted"
                ) from exc
            event_source_sha256 = canonical_sha256(mappings)
            resource_policy_sha256 = canonical_sha256(resource_policy)
            function_url_sha256 = canonical_sha256(function_url_state)
            disabled_profile_sha256 = canonical_sha256(
                {
                    "resource_policy_sha256": resource_policy_sha256,
                    "event_source_state_sha256": event_source_sha256,
                    "function_url_state_sha256": function_url_sha256,
                }
            )
            return {
                "function_version_arn": configuration.get(
                    "FunctionArn"
                ),
                "function_code_sha256": code_sha256,
                "execution_role_arn": role_arn,
                "execution_role_id": role.get("RoleId"),
                "role_trust_policy_sha256": canonical_sha256(
                    role.get("AssumeRolePolicyDocument")
                ),
                "role_permission_policy_sha256": canonical_sha256(
                    permission_inventory
                ),
                "stack_id": stack_id,
                "stack_template_sha256": canonical_sha256(template),
                "resource_policy_sha256": resource_policy_sha256,
                "event_source_state_sha256": event_source_sha256,
                "function_url_state_sha256": function_url_sha256,
                "expected_attachment_identity_sha256": variables.get(
                    "GLM52_EXPECTED_ATTACHMENT_IDENTITY_SHA256"
                ),
                "observed_attachment_identity_sha256": event_source_sha256,
                "disabled_support_profile_sha256": disabled_profile_sha256,
            }

    return FenceHandlerServices(
        s3=s3,
        executor=FenceExecutor(
            client=_Task11FenceAwsClient(
                cloudformation=cloudformation,
                s3=s3,
                iam=iam,
            ),
            record_store=DynamoFenceRecordStore(
                client=dynamodb,
                table_name=table_name,
            ),
        ),
        authority_class=authority_class,
        live_support_runtime_identity=live_reader,
    )


def build_admission_handler_services(
    *,
    generation: int,
    decision_nonce: bytes,
    live_h1d_identity_sha256: str,
    task9_deployed_identity_coordinate: object,
    task9_deployed_identity_sha256: str,
) -> object:
    """Construct the one-wire durable launch-admission reference monitor."""

    from .support_launchadmission_handler import AdmissionHandlerServices
    from .task11_relay_runtime import (
        AdmissionCurrentAuthority,
        AdmissionDynamoStore,
        AdmissionFreshAudit,
        PinnedMtlsRelay,
        RelayAdmissionSender,
        aws_relay_runtime_clients,
        load_task9_deployed_identity,
    )

    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    function_version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "")
    table_name = os.environ.get("GLM52_LEDGER_TABLE_NAME", "")
    campaign_bucket = os.environ.get("GLM52_CAMPAIGN_BUCKET", "")
    activation_id = os.environ.get("GLM52_ACTIVATION_ID", "")
    deployment = os.environ.get(
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
        "",
    )
    function_version_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        + function_name
        + ":"
        + function_version
    )
    if (
        function_name != "keep-glm52-h1g-launch-admission"
        or _VERSION_ARN.fullmatch(function_version_arn) is None
        or not table_name
        or not campaign_bucket
        or not activation_id
        or type(generation) is not int
        or generation <= 0
        or type(decision_nonce) is not bytes
        or len(decision_nonce) < 16
        or _SHA.fullmatch(live_h1d_identity_sha256) is None
        or _SHA.fullmatch(deployment) is None
        or os.environ.get("GLM52_ACCOUNT_ID") != "246813579024"
        or os.environ.get("AWS_REGION") != "us-west-2"
        or os.environ.get("GLM52_RUN_ID") != "glm52-sky-20260724"
    ):
        raise RuntimeError("launch-admission production config drifted")
    clients = aws_relay_runtime_clients()
    loaded_task9 = load_task9_deployed_identity(
        s3=clients.s3,
        coordinate=task9_deployed_identity_coordinate,
        activation_id=activation_id,
        expected_body_sha256=task9_deployed_identity_sha256,
        expected_bucket=campaign_bucket,
    )
    config = Task11ActionConfig(
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id=activation_id,
        ledger_table_name=table_name,
        campaign_bucket=campaign_bucket,
        deployment_identity_sha256=deployment,
        function_version_arn=function_version_arn,
    )
    store = AdmissionDynamoStore(
        dynamodb=clients.dynamodb,
        table_name=table_name,
        activation_id=activation_id,
        generation=generation,
        function_version_arn=function_version_arn,
    )
    fresh = FreshH1fAuditService(
        authority_reader=_fresh_authority_reader(
            config=config,
            clients=Task11AwsClients(
                s3=clients.s3,
                dynamodb=clients.dynamodb,
                cloudformation=None,
                lambda_client=None,
                sts=None,
            ),
            generation=generation,
        )
    )
    sender = RelayAdmissionSender(
        PinnedMtlsRelay(
            secretsmanager=clients.secretsmanager,
            purpose="LAUNCH_ADMISSION",
            identity=loaded_task9.sky_identity,
        )
    )
    admission = SkyAdmissionService(
        store=store,
        fresh_audit=AdmissionFreshAudit(
            store=store,
            s3=clients.s3,
            service=fresh,
        ),
        current_authority=AdmissionCurrentAuthority(
            store=store,
            decision_nonce_sha256=hashlib.sha256(
                decision_nonce
            ).hexdigest(),
            live_h1d_identity_sha256=live_h1d_identity_sha256,
        ),
        relay=sender,
        nonce_source=lambda: secrets.token_bytes(32),
    )

    def evidence() -> Mapping[str, object]:
        value = store.evidence()
        return {
            "post_audit": value["post_audit"],
            "action_states": value["action_states"],
            "relay_call_count": sender.call_count,
        }

    return AdmissionHandlerServices(
        s3=clients.s3,
        admission=admission,
        evidence=evidence,
        task9_deployed_identity_sha256=loaded_task9.body_sha256,
    )


def _fresh_authority_reader(
    *,
    config: Task11ActionConfig,
    clients: Task11AwsClients,
    generation: int,
) -> Callable[[], H1fAuthoritySnapshot]:
    """Build a fresh index/control/S3 head reader for accepted H.1f."""

    ledger = DynamoLedgerAdapter(
        client=clients.dynamodb,
        table_name=config.ledger_table_name,
    )

    def read_current() -> H1fAuthoritySnapshot:
        rows = ledger.read_coherent(
            items=(
                (
                    LedgerKey(config.run_id, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        config.run_id,
                        "ACTIVATION#"
                        + config.activation_id
                        + "#CONTROL",
                    ),
                    "glm52_production_control",
                ),
            )
        )
        if len(rows) != 2:
            raise RuntimeError("live H.1f control readback is incomplete")
        index, control = rows
        if (
            index["current_activation_id"] != config.activation_id
            or control["activation_id"] != config.activation_id
            or index["campaign_identity_sha256"]
            != control["campaign_identity_sha256"]
        ):
            raise RuntimeError("live H.1f activation binding drifted")
        prefix = "campaigns/" + config.run_id + "/authorities/fence/"
        marker = None
        version_marker = None
        matches = []
        seen_tokens = set()
        while True:
            request = {
                "Bucket": config.campaign_bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,
                "ExpectedBucketOwner": config.account_id,
            }
            if marker is not None:
                request["KeyMarker"] = marker
                request["VersionIdMarker"] = version_marker
            response = clients.s3.list_object_versions(**request)
            metadata = (
                response.get("ResponseMetadata")
                if type(response) is dict
                else None
            )
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 200
                or type(metadata.get("RequestId")) is not str
                or not metadata["RequestId"]
                or type(response.get("Versions")) is not list
                or type(response.get("DeleteMarkers", [])) is not list
            ):
                raise RuntimeError(
                    "live H.1f S3 version page is unauthenticated"
                )
            if any(
                item.get("VersionId") == control["fence_head_version_id"]
                for item in response.get("DeleteMarkers", [])
                if type(item) is dict
            ):
                raise RuntimeError("live H.1f head is delete-marked")
            matches.extend(
                item
                for item in response["Versions"]
                if type(item) is dict
                and item.get("VersionId")
                == control["fence_head_version_id"]
            )
            if response.get("IsTruncated") is False:
                break
            marker = response.get("NextKeyMarker")
            version_marker = response.get("NextVersionIdMarker")
            token = (marker, version_marker)
            if (
                type(marker) is not str
                or not marker
                or type(version_marker) is not str
                or not version_marker
                or token in seen_tokens
            ):
                raise RuntimeError(
                    "live H.1f S3 pagination is ambiguous"
                )
            seen_tokens.add(token)
            if len(seen_tokens) > 100:
                raise RuntimeError("live H.1f S3 pagination is unbounded")
        if len(matches) != 1:
            raise RuntimeError("live H.1f head VersionId is not unique")
        key = matches[0].get("Key")
        if type(key) is not str or not key.startswith(prefix):
            raise RuntimeError("live H.1f head key is foreign")
        response = clients.s3.get_object(
            Bucket=config.campaign_bucket,
            Key=key,
            VersionId=control["fence_head_version_id"],
            ExpectedBucketOwner=config.account_id,
            ChecksumMode="ENABLED",
        )
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        body = response.get("Body") if type(response) is dict else None
        read = getattr(body, "read", None)
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
            or response.get("VersionId")
            != control["fence_head_version_id"]
            or not callable(read)
        ):
            raise RuntimeError("live H.1f head read is unauthenticated")
        raw = read(2 * 1024 * 1024 + 1)
        if (
            type(raw) is not bytes
            or not raw.endswith(b"\n")
            or len(raw) > 2 * 1024 * 1024
        ):
            raise RuntimeError("live H.1f head bytes are invalid")
        try:
            parsed = json.loads(raw[:-1].decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("live H.1f head is invalid JSON") from exc
        if (
            type(parsed) is not dict
            or canonical_json_bytes(parsed) + b"\n" != raw
            or parsed.get("run_id") != config.run_id
            or parsed.get("fence_body_sha256")
            != control["fence_head_body_sha256"]
        ):
            raise RuntimeError("live H.1f head body drifted")
        return H1fAuthoritySnapshot(
            authority_domain="ACTIVATION",
            bucket=config.campaign_bucket,
            run_id=config.run_id,
            activation_id=config.activation_id,
            generation=generation,
            epoch=control["active_epoch"],
            execution_arn=control["active_execution_arn"],
            barrier_nonce_sha256=control["barrier_nonce_sha256"],
            closing_revision=control["revision"],
            active_head_key=key,
            active_head_version_id=control["fence_head_version_id"],
            active_head_file_sha256=hashlib.sha256(raw).hexdigest(),
            active_head_body_sha256=control["fence_head_body_sha256"],
        )

    return read_current


def _accepted_adapters(
    *,
    config: Task11ProductionConfig,
    clients: Task11AwsClients,
    generation: int,
) -> Task11AcceptedAdapters:
    ledger = DynamoLedgerAdapter(
        client=clients.dynamodb,
        table_name=config.ledger_table_name,
    )
    action_config = Task11ActionConfig(
        account_id=config.account_id,
        region=config.region,
        run_id=config.run_id,
        activation_id=config.activation_id,
        ledger_table_name=config.ledger_table_name,
        campaign_bucket=config.campaign_bucket,
        deployment_identity_sha256=config.deployment_identity_sha256,
        function_version_arn=config.decision_function_version_arn,
    )
    reader = _fresh_authority_reader(
        config=action_config,
        clients=clients,
        generation=generation,
    )
    return Task11AcceptedAdapters(
        ledger=ledger,
        fresh_h1f=FreshH1fAuditService(authority_reader=reader),
        clients=clients,
    )


def _boundary_coordinate_from_event(
    event: Mapping[str, object],
) -> Task11BoundaryCoordinate:
    closure = event.get("closure_request")
    value = (
        closure.get("task11_boundary")
        if type(closure) is dict
        else None
    )
    if (
        type(value) is not dict
        or set(value)
        != {
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
        }
    ):
        raise RuntimeError("Task 11 boundary coordinate is absent")
    try:
        return Task11BoundaryCoordinate(**value)
    except TypeError as exc:
        raise RuntimeError(
            "Task 11 boundary coordinate is malformed"
        ) from exc


def build_task11_production_services(
    *,
    event: Mapping[str, object],
    context: object,
) -> Task11ProductionServices:
    """Construct only accepted typed adapters for one production invocation."""

    if type(event) is not dict or context is None:
        raise RuntimeError("Task 11 production invocation is not closed")
    config = _configuration()
    clients = _aws_clients(config=config, context=context)
    boundary = load_task11_boundary(
        s3=clients.s3,
        coordinate=_boundary_coordinate_from_event(event),
    )
    closure = event["closure_request"]
    if (
        boundary.activation_id != config.activation_id
        or boundary.activation_id != closure["activation_id"]
        or boundary.generation != closure["generation"]
    ):
        raise RuntimeError("Task 11 boundary scope drifted")
    adapters = _accepted_adapters(
        config=config,
        clients=clients,
        generation=boundary.generation,
    )
    return Task11ProductionServices(
        config=config,
        adapters=adapters,
        boundary=boundary,
        event=event,
        context=context,
    )


__all__ = [
    "RuntimeAttachmentInventoryEvidence",
    "RuntimeRevalidationSourceRead",
    "Task11AcceptedAdapters",
    "Task11ActionConfig",
    "Task11AwsClients",
    "Task11ProductionConfig",
    "Task11ProductionServices",
    "Task11ReadOnlyPhaseServices",
    "build_effect_writer_services",
    "build_admission_handler_services",
    "build_attestation_handler_services",
    "build_source_publisher_services",
    "build_task11_production_services",
    "load_runtime_revalidation_sources",
    "revalidate_runtime_attachment_specs",
]
