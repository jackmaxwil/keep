"""Five fixed one-shot source publishers and their sequential activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Callable, ClassVar, Optional, Tuple

from .s3_adapter import (
    ConditionalCreateResult,
    S3CreateActionRequest,
    S3PublicationServices,
    conditional_create_immutable_json,
)
from .s3_records import (
    S3ObjectIdentity,
    build_immutable_json_candidate,
)
from .source_authorities import (
    VersionedJsonArtifact,
    gpu_spend_snapshot_s3_key,
    production_controller_baseline_s3_key,
    production_must_start_control_plane_ready_s3_key,
    production_submission_acquired_s3_key,
    production_submission_intent_s3_key,
    validate_gpu_spend_snapshot,
    validate_production_controller_baseline,
    validate_production_must_start_control_plane_ready,
    validate_production_submission_acquired,
    validate_production_submission_intent,
)


_ROLE_SEGMENT = r"[A-Za-z0-9+=,.@_-]+"
_ROLE_ARN = re.compile(
    rf"arn:aws:iam::246813579024:role/{_ROLE_SEGMENT}"
    rf"(?:/{_ROLE_SEGMENT})*\Z"
)
_ASSUMED_ROLE_ARN = re.compile(
    rf"arn:aws:sts::246813579024:assumed-role/"
    rf"{_ROLE_SEGMENT}/{_ROLE_SEGMENT}\Z"
)
_CAMPAIGN_BUCKET = re.compile(
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z"
)
_IPV4_FORMATTED_BUCKET = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}\Z")
_RESERVED_BUCKET_PREFIXES = ("xn--", "sthree-", "amzn-s3-demo-")
_RESERVED_BUCKET_SUFFIXES = (
    "-s3alias",
    "--ol-s3",
    ".mrap",
    "--x-s3",
    "--table-s3",
    "-an",
)


class SourcePublicationError(ValueError):
    """A fixed publisher or sequential activation failed closed."""


@dataclass(frozen=True)
class GpuSpendSnapshotPublisherRuntime:
    services: S3PublicationServices
    closure_caller_arn: str
    publisher_role_arn: str
    campaign_bucket: str


@dataclass(frozen=True)
class ProductionSubmissionIntentPublisherRuntime:
    services: S3PublicationServices
    closure_caller_arn: str
    publisher_role_arn: str
    campaign_bucket: str


@dataclass(frozen=True)
class ProductionControllerBaselinePublisherRuntime:
    services: S3PublicationServices
    closure_caller_arn: str
    publisher_role_arn: str
    campaign_bucket: str


@dataclass(frozen=True)
class ProductionControlPlaneReadinessPublisherRuntime:
    services: S3PublicationServices
    closure_caller_arn: str
    publisher_role_arn: str
    campaign_bucket: str


@dataclass(frozen=True)
class ProductionSubmissionAcquisitionPublisherRuntime:
    services: S3PublicationServices
    closure_caller_arn: str
    publisher_role_arn: str
    campaign_bucket: str


@dataclass(frozen=True)
class GpuSpendSnapshotPublicationRequest:
    source_kind: ClassVar[str] = "gpu-spend-snapshot"

    caller_arn: str
    activation_id: str
    generation: int
    action_key: str
    raw: bytes


@dataclass(frozen=True)
class ProductionSubmissionIntentPublicationRequest:
    source_kind: ClassVar[str] = "production-submission-intent"

    caller_arn: str
    activation_id: str
    generation: int
    action_key: str
    raw: bytes


@dataclass(frozen=True)
class ProductionControllerBaselinePublicationRequest:
    source_kind: ClassVar[str] = "production-controller-baseline"

    caller_arn: str
    activation_id: str
    generation: int
    action_key: str
    raw: bytes
    descriptor: VersionedJsonArtifact
    intent: VersionedJsonArtifact


@dataclass(frozen=True)
class ProductionControlPlaneReadinessPublicationRequest:
    source_kind: ClassVar[str] = "production-control-plane-readiness"

    caller_arn: str
    activation_id: str
    generation: int
    action_key: str
    raw: bytes
    descriptor: VersionedJsonArtifact
    intent: VersionedJsonArtifact
    controller_baseline: VersionedJsonArtifact


@dataclass(frozen=True)
class ProductionSubmissionAcquisitionPublicationRequest:
    source_kind: ClassVar[str] = "production-submission-acquisition"

    caller_arn: str
    activation_id: str
    generation: int
    action_key: str
    raw: bytes
    descriptor: VersionedJsonArtifact
    intent: VersionedJsonArtifact
    controller_baseline: VersionedJsonArtifact
    must_start_control_plane_ready: VersionedJsonArtifact
    now: object


@dataclass(frozen=True)
class SourcePublicationResult:
    """The bounded publication identity; it carries no write authority."""

    source_kind: str
    publisher_role_arn: str
    object_identity: S3ObjectIdentity
    candidate_identity_sha256: str
    provenance: str
    authority_audit_body_sha256: str
    closing_revision: int
    authorized_revision: int
    direct_request_id: Optional[str]
    direct_server_date: Optional[str]
    direct_response_authenticated: bool


@dataclass(frozen=True)
class DurableSourceRecord:
    """Same-owner durable recording of one provisional source identity."""

    activation_id: str
    generation: int
    source_kind: str
    publisher_role_arn: str
    object_identity: S3ObjectIdentity
    candidate_identity_sha256: str
    publication_provenance: str
    activation_owner_identity_sha256: str
    authority_audit_body_sha256: str
    closing_revision: int
    authorized_revision: int
    direct_request_id: Optional[str]
    direct_server_date: Optional[str]
    direct_response_authenticated: bool


@dataclass(frozen=True)
class StabilizationProbeSet:
    """One exact policy/readback/publisher-denial observation."""

    observed_at: object
    policy_sha256: str
    readback_sha256: str
    denied_publisher_roles: Tuple[str, ...]


@dataclass(frozen=True)
class BatchSuccessorRequest:
    """All five identities supplied atomically to the precreated successor."""

    activation_id: str
    generation: int
    activation_owner_identity_sha256: str
    expected_campaign_bucket: str
    expected_closure_caller_arn: str
    sources: Tuple[DurableSourceRecord, ...]


@dataclass(frozen=True)
class BatchSuccessorResult:
    """Injected Task 6/8 boundary proof; never policy mutation capability."""

    selected_successor_count: int
    selected_precreated_successor: bool
    family_closing_policy_applied: bool
    activated_sources: Tuple[DurableSourceRecord, ...]
    denied_publisher_roles: Tuple[str, ...]
    probe_sets: Tuple[StabilizationProbeSet, ...]
    fresh_h1f_unique_zero_child_active_head: bool
    fresh_h1f_audit_body_sha256: str
    active_head_body_sha256: str
    active_head_version_id: str


@dataclass(frozen=True)
class SequentialSourceServices:
    gpu_spend_snapshot_runtime: GpuSpendSnapshotPublisherRuntime
    production_submission_intent_runtime: ProductionSubmissionIntentPublisherRuntime
    production_controller_baseline_runtime: ProductionControllerBaselinePublisherRuntime
    production_control_plane_readiness_runtime: (
        ProductionControlPlaneReadinessPublisherRuntime
    )
    production_submission_acquisition_runtime: (
        ProductionSubmissionAcquisitionPublisherRuntime
    )
    durable_recorder: object
    candidate_factory: object
    batch_successor: object


@dataclass(frozen=True)
class SequentialSourceActivationRequest:
    activation_id: str
    generation: int
    activation_owner_identity_sha256: str
    expected_campaign_bucket: str
    expected_closure_caller_arn: str
    gpu_spend_snapshot: GpuSpendSnapshotPublicationRequest


@dataclass(frozen=True)
class SequentialSourceActivationResult:
    durable_sources: Tuple[DurableSourceRecord, ...]
    batch_request: BatchSuccessorRequest
    batch_result: BatchSuccessorResult
    sources_authoritative: bool


def _fail(message: str) -> None:
    raise SourcePublicationError(message)


def _exact_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{field} must be an exact nonempty string")
    return value


def _exact_generation(value: object, *, field: str) -> int:
    if (
        type(value) is not int
        or value < 1
        or value > 99_999_999
    ):
        _fail(f"{field} must be an exact positive eight-digit integer")
    return value


def _is_closure_caller_arn(value: str) -> bool:
    return (
        _ROLE_ARN.fullmatch(value) is not None
        or _ASSUMED_ROLE_ARN.fullmatch(value) is not None
    )


def _validate_campaign_bucket(value: object, *, field: str) -> str:
    bucket = _exact_string(value, field=field)
    if (
        _CAMPAIGN_BUCKET.fullmatch(bucket) is None
        or ".." in bucket
        or ".-" in bucket
        or "-." in bucket
        or _IPV4_FORMATTED_BUCKET.fullmatch(bucket) is not None
        or bucket.startswith(_RESERVED_BUCKET_PREFIXES)
        or bucket.endswith(_RESERVED_BUCKET_SUFFIXES)
    ):
        _fail(f"{field} is not an exact general-purpose S3 bucket name")
    return bucket


def _record(raw: object) -> dict[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("source bytes must end in exactly one LF")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourcePublicationError("source bytes are invalid JSON") from exc
    if type(value) is not dict:
        _fail("source bytes must contain one exact object")
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    if canonical != raw:
        _fail("source bytes are not canonical JSON plus LF")
    return value


def _runtime(
    value: object, *, caller_arn: str, expected_type: type
) -> object:
    if type(value) is not expected_type:
        raise TypeError(f"runtime must be exact {expected_type.__name__}")
    if type(value.services) is not S3PublicationServices:
        _fail("runtime services are not exact Task 4 services")
    expected = _exact_string(
        value.closure_caller_arn, field="closure_caller_arn"
    )
    caller = _exact_string(caller_arn, field="caller_arn")
    if (
        not _is_closure_caller_arn(expected)
        or not _is_closure_caller_arn(caller)
        or caller != expected
    ):
        _fail("caller is not the exact closure principal")
    role = _exact_string(value.publisher_role_arn, field="publisher_role_arn")
    if _ROLE_ARN.fullmatch(role) is None:
        _fail("publisher_role_arn is not a stable IAM role")
    _validate_campaign_bucket(
        value.campaign_bucket,
        field="campaign_bucket",
    )
    return value


def _request_scope(request: object) -> None:
    for field in ("activation_id", "action_key"):
        _exact_string(getattr(request, field), field=field)
    _exact_generation(
        getattr(request, "generation"),
        field="generation",
    )


def _publish_candidate(
    *,
    runtime: object,
    request: object,
    expected_request_type: type,
    expected_runtime_type: type,
    candidate: object,
) -> SourcePublicationResult:
    if type(request) is not expected_request_type:
        raise TypeError(
            f"request must be exact {expected_request_type.__name__}"
        )
    exact_runtime = _runtime(
        runtime,
        caller_arn=request.caller_arn,
        expected_type=expected_runtime_type,
    )
    _request_scope(request)
    source_kind = candidate.record_kind
    try:
        result = conditional_create_immutable_json(
            services=exact_runtime.services,
            request=S3CreateActionRequest(
                authority_domain="ACTIVATION",
                operation_kind="S3_CREATE",
                action_key=request.action_key,
                candidate=candidate,
                activation_id=request.activation_id,
                generation=request.generation,
            ),
        )
    except Exception as exc:
        raise SourcePublicationError(
            f"{source_kind} Task 4 publication failed"
        ) from exc
    if (
        type(result) is not ConditionalCreateResult
        or result.outcome not in {"created-direct", "reconciled-existing"}
        or result.object_identity is None
        or result.provenance
        not in {"direct-response", "all-version-reconciliation"}
    ):
        _fail(f"{source_kind} did not establish one exact object identity")
    if result.outcome == "created-direct":
        if (
            result.provenance != "direct-response"
            or result.direct_response_authenticated is not True
            or type(result.direct_request_id) is not str
            or not result.direct_request_id
            or type(result.direct_server_date) is not str
            or not result.direct_server_date
        ):
            _fail(
                f"{source_kind} lost authenticated direct-response custody"
            )
    elif (
        result.provenance != "all-version-reconciliation"
        or result.direct_response_authenticated is not False
        or result.direct_request_id is not None
        or result.direct_server_date is not None
    ):
        _fail(
            f"{source_kind} invented direct-response custody on adoption"
        )
    identity = result.object_identity
    if (
        identity.bucket != candidate.bucket
        or identity.key != candidate.key
        or identity.file_sha256 != candidate.file_sha256
        or identity.body_sha256 != candidate.body_sha256
    ):
        _fail(f"{source_kind} Task 4 result drifted from candidate")
    return SourcePublicationResult(
        source_kind=source_kind,
        publisher_role_arn=exact_runtime.publisher_role_arn,
        object_identity=identity,
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        provenance=result.provenance,
        authority_audit_body_sha256=(
            result.authority_audit_body_sha256
        ),
        closing_revision=result.closing_revision,
        authorized_revision=result.authorized_revision,
        direct_request_id=result.direct_request_id,
        direct_server_date=result.direct_server_date,
        direct_response_authenticated=(
            result.direct_response_authenticated
        ),
    )


def publish_gpu_spend_snapshot(
    *,
    runtime: GpuSpendSnapshotPublisherRuntime,
    request: GpuSpendSnapshotPublicationRequest,
) -> SourcePublicationResult:
    if type(request) is not GpuSpendSnapshotPublicationRequest:
        raise TypeError(
            "request must be exact GpuSpendSnapshotPublicationRequest"
        )
    try:
        record = validate_gpu_spend_snapshot(_record(request.raw))
        candidate = build_immutable_json_candidate(
            record_kind=GpuSpendSnapshotPublicationRequest.source_kind,
            bucket=runtime.campaign_bucket,
            key=gpu_spend_snapshot_s3_key(
                run_id=str(record["run_id"]),
                snapshot_body_sha256=str(record["snapshot_body_sha256"]),
            ),
            raw=request.raw,
            activation_id=request.activation_id,
            generation=request.generation,
        )
    except (TypeError, ValueError) as exc:
        raise SourcePublicationError(
            "gpu-spend-snapshot candidate did not authenticate"
        ) from exc
    return _publish_candidate(
        runtime=runtime,
        request=request,
        expected_request_type=GpuSpendSnapshotPublicationRequest,
        expected_runtime_type=GpuSpendSnapshotPublisherRuntime,
        candidate=candidate,
    )


def publish_production_submission_intent(
    *,
    runtime: ProductionSubmissionIntentPublisherRuntime,
    request: ProductionSubmissionIntentPublicationRequest,
) -> SourcePublicationResult:
    if type(request) is not ProductionSubmissionIntentPublicationRequest:
        raise TypeError(
            "request must be exact "
            "ProductionSubmissionIntentPublicationRequest"
        )
    try:
        record = validate_production_submission_intent(_record(request.raw))
        if record["bucket"] != runtime.campaign_bucket:
            _fail("production intent destination bucket drifted")
        candidate = build_immutable_json_candidate(
            record_kind=(
                ProductionSubmissionIntentPublicationRequest.source_kind
            ),
            bucket=runtime.campaign_bucket,
            key=production_submission_intent_s3_key(
                run_id=str(record["run_id"]),
                intent_body_sha256=str(record["intent_body_sha256"]),
            ),
            raw=request.raw,
            activation_id=request.activation_id,
            generation=request.generation,
        )
    except (TypeError, ValueError) as exc:
        raise SourcePublicationError(
            "production-submission-intent candidate did not authenticate"
        ) from exc
    return _publish_candidate(
        runtime=runtime,
        request=request,
        expected_request_type=ProductionSubmissionIntentPublicationRequest,
        expected_runtime_type=ProductionSubmissionIntentPublisherRuntime,
        candidate=candidate,
    )


def publish_production_controller_baseline(
    *,
    runtime: ProductionControllerBaselinePublisherRuntime,
    request: ProductionControllerBaselinePublicationRequest,
) -> SourcePublicationResult:
    if type(request) is not ProductionControllerBaselinePublicationRequest:
        raise TypeError(
            "request must be exact "
            "ProductionControllerBaselinePublicationRequest"
        )
    try:
        record = validate_production_controller_baseline(
            _record(request.raw),
            descriptor=request.descriptor,
            intent=request.intent,
        )
        if record["bucket"] != runtime.campaign_bucket:
            _fail("controller baseline destination bucket drifted")
        candidate = build_immutable_json_candidate(
            record_kind=(
                ProductionControllerBaselinePublicationRequest.source_kind
            ),
            bucket=runtime.campaign_bucket,
            key=production_controller_baseline_s3_key(
                run_id=str(record["run_id"]),
                baseline_body_sha256=str(record["baseline_body_sha256"]),
            ),
            raw=request.raw,
            activation_id=request.activation_id,
            generation=request.generation,
        )
    except (TypeError, ValueError) as exc:
        raise SourcePublicationError(
            "production-controller-baseline candidate did not authenticate"
        ) from exc
    return _publish_candidate(
        runtime=runtime,
        request=request,
        expected_request_type=ProductionControllerBaselinePublicationRequest,
        expected_runtime_type=ProductionControllerBaselinePublisherRuntime,
        candidate=candidate,
    )


def publish_production_control_plane_readiness(
    *,
    runtime: ProductionControlPlaneReadinessPublisherRuntime,
    request: ProductionControlPlaneReadinessPublicationRequest,
) -> SourcePublicationResult:
    if type(request) is not ProductionControlPlaneReadinessPublicationRequest:
        raise TypeError(
            "request must be exact "
            "ProductionControlPlaneReadinessPublicationRequest"
        )
    try:
        record = validate_production_must_start_control_plane_ready(
            _record(request.raw),
            descriptor=request.descriptor,
            intent=request.intent,
            controller_baseline=request.controller_baseline,
        )
        if record["bucket"] != runtime.campaign_bucket:
            _fail("control-plane readiness destination bucket drifted")
        candidate = build_immutable_json_candidate(
            record_kind=(
                ProductionControlPlaneReadinessPublicationRequest.source_kind
            ),
            bucket=runtime.campaign_bucket,
            key=production_must_start_control_plane_ready_s3_key(
                run_id=str(record["run_id"]),
                intent_body_sha256=str(record["intent_body_sha256"]),
                control_plane_ready_body_sha256=str(
                    record["control_plane_ready_body_sha256"]
                ),
            ),
            raw=request.raw,
            activation_id=request.activation_id,
            generation=request.generation,
        )
    except (TypeError, ValueError) as exc:
        raise SourcePublicationError(
            "production-control-plane-readiness candidate "
            "did not authenticate"
        ) from exc
    return _publish_candidate(
        runtime=runtime,
        request=request,
        expected_request_type=ProductionControlPlaneReadinessPublicationRequest,
        expected_runtime_type=ProductionControlPlaneReadinessPublisherRuntime,
        candidate=candidate,
    )


def publish_production_submission_acquisition(
    *,
    runtime: ProductionSubmissionAcquisitionPublisherRuntime,
    request: ProductionSubmissionAcquisitionPublicationRequest,
) -> SourcePublicationResult:
    if type(request) is not ProductionSubmissionAcquisitionPublicationRequest:
        raise TypeError(
            "request must be exact "
            "ProductionSubmissionAcquisitionPublicationRequest"
        )
    try:
        record = validate_production_submission_acquired(
            _record(request.raw),
            descriptor=request.descriptor,
            intent=request.intent,
            controller_baseline=request.controller_baseline,
            must_start_control_plane_ready=(
                request.must_start_control_plane_ready
            ),
            now=request.now,
        )
        if record["bucket"] != runtime.campaign_bucket:
            _fail("submission acquisition destination bucket drifted")
        candidate = build_immutable_json_candidate(
            record_kind=(
                ProductionSubmissionAcquisitionPublicationRequest.source_kind
            ),
            bucket=runtime.campaign_bucket,
            key=production_submission_acquired_s3_key(
                run_id=str(record["run_id"]),
                descriptor_file_sha256=str(
                    record["descriptor_file_sha256"]
                ),
            ),
            raw=request.raw,
            activation_id=request.activation_id,
            generation=request.generation,
        )
    except (TypeError, ValueError) as exc:
        raise SourcePublicationError(
            "production-submission-acquisition candidate "
            "did not authenticate"
        ) from exc
    return _publish_candidate(
        runtime=runtime,
        request=request,
        expected_request_type=ProductionSubmissionAcquisitionPublicationRequest,
        expected_runtime_type=ProductionSubmissionAcquisitionPublisherRuntime,
        candidate=candidate,
    )


_SOURCE_ORDER = (
    "gpu-spend-snapshot",
    "production-submission-intent",
    "production-controller-baseline",
    "production-control-plane-readiness",
    "production-submission-acquisition",
)


def _method(service: object, name: str) -> Callable[..., object]:
    try:
        result = getattr(service, name)
    except Exception as exc:
        raise SourcePublicationError(f"service lacks {name}") from exc
    if not callable(result):
        _fail(f"service lacks {name}")
    return result


def _durable_record(
    value: object,
    *,
    request: SequentialSourceActivationRequest,
    publication: SourcePublicationResult,
) -> DurableSourceRecord:
    if type(value) is not DurableSourceRecord:
        _fail("durable recorder did not return an exact source record")
    _exact_generation(value.generation, field="durable generation")
    if (
        value.activation_id != request.activation_id
        or value.generation != request.generation
        or value.source_kind != publication.source_kind
        or value.publisher_role_arn != publication.publisher_role_arn
        or value.object_identity != publication.object_identity
        or value.candidate_identity_sha256
        != publication.candidate_identity_sha256
        or value.publication_provenance != publication.provenance
        or value.activation_owner_identity_sha256
        != request.activation_owner_identity_sha256
        or value.authority_audit_body_sha256
        != publication.authority_audit_body_sha256
        or value.closing_revision != publication.closing_revision
        or value.authorized_revision != publication.authorized_revision
        or value.direct_request_id != publication.direct_request_id
        or value.direct_server_date != publication.direct_server_date
        or value.direct_response_authenticated
        is not publication.direct_response_authenticated
    ):
        _fail("durable source record drifted from publication")
    _digest_text(
        value.activation_owner_identity_sha256,
        field="activation_owner_identity_sha256",
    )
    return value


def _digest_text(value: object, *, field: str) -> str:
    text = _exact_string(value, field=field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail(f"{field} must be exact lowercase SHA-256")
    return text


def _artifact_matches(
    artifact: object, durable: DurableSourceRecord
) -> bool:
    return (
        type(artifact) is VersionedJsonArtifact
        and artifact.key == durable.object_identity.key
        and artifact.version_id == durable.object_identity.version_id
        and hashlib.sha256(artifact.raw).hexdigest()
        == durable.object_identity.file_sha256
    )


def _candidate_predecessors(
    candidate: object,
    *,
    kind: str,
    durable: Tuple[DurableSourceRecord, ...],
) -> None:
    if tuple(item.source_kind for item in durable) != _SOURCE_ORDER[
        : len(durable)
    ]:
        _fail("durable predecessor order is not canonical")
    record = _record(candidate.raw)
    if kind == "production-submission-intent":
        prior = durable[0]
        expected = {
            "gpu_spend_snapshot_key": prior.object_identity.key,
            "gpu_spend_snapshot_file_sha256": (
                prior.object_identity.file_sha256
            ),
            "gpu_spend_snapshot_body_sha256": (
                prior.object_identity.body_sha256
            ),
            "gpu_spend_snapshot_version_id": (
                prior.object_identity.version_id
            ),
        }
        if any(record.get(field) != value for field, value in expected.items()):
            _fail("production intent does not bind the durable spend snapshot")
        return
    artifact_requirements = {
        "production-controller-baseline": (("intent", 1),),
        "production-control-plane-readiness": (
            ("intent", 1),
            ("controller_baseline", 2),
        ),
        "production-submission-acquisition": (
            ("intent", 1),
            ("controller_baseline", 2),
            ("must_start_control_plane_ready", 3),
        ),
    }
    requirements = artifact_requirements.get(kind)
    if requirements is None:
        _fail("candidate source kind is not sequentially registered")
    for field, index in requirements:
        if not _artifact_matches(getattr(candidate, field), durable[index]):
            _fail(f"{kind} does not bind durable predecessor {field}")


def _record_publication(
    *,
    recorder: object,
    activation_request: SequentialSourceActivationRequest,
    publication: SourcePublicationResult,
) -> DurableSourceRecord:
    method = _method(recorder, "record_source")
    try:
        value = method(
            activation_id=activation_request.activation_id,
            generation=activation_request.generation,
            publication=publication,
        )
    except Exception as exc:
        raise SourcePublicationError(
            f"durable recording failed for {publication.source_kind}"
        ) from exc
    return _durable_record(
        value,
        request=activation_request,
        publication=publication,
    )


def _build_next(
    factory: object,
    *,
    method_name: str,
    expected_type: type,
    kind: str,
    durable: Tuple[DurableSourceRecord, ...],
) -> object:
    method = _method(factory, method_name)
    try:
        candidate = method(predecessors=durable)
    except Exception as exc:
        raise SourcePublicationError(f"{kind} candidate construction failed") from exc
    if type(candidate) is not expected_type:
        _fail(f"{kind} factory returned the wrong closed request")
    _candidate_predecessors(candidate, kind=kind, durable=durable)
    return candidate


def _probe_time(value: object, *, field: str) -> datetime:
    if type(value) is not datetime or value.microsecond != 0:
        _fail(f"{field} must be an exact whole-second datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _batch_is_authoritative(
    result: object, *, request: BatchSuccessorRequest
) -> bool:
    if type(result) is not BatchSuccessorResult:
        _fail("batch successor returned the wrong proof type")
    roles = tuple(source.publisher_role_arn for source in request.sources)
    try:
        request_generation = _exact_generation(
            request.generation,
            field="batch request generation",
        )
        for source in request.sources:
            if type(source) is not DurableSourceRecord:
                return False
            _exact_generation(
                source.generation,
                field="batch request source generation",
            )
        for source in result.activated_sources:
            if type(source) is not DurableSourceRecord:
                return False
            _exact_generation(
                source.generation,
                field="activated source generation",
            )
    except (SourcePublicationError, TypeError):
        return False
    if (
        type(result.selected_successor_count) is not int
        or result.selected_successor_count != 1
        or type(result.selected_precreated_successor) is not bool
        or result.selected_precreated_successor is not True
        or type(result.family_closing_policy_applied) is not bool
        or result.family_closing_policy_applied is not True
        or result.activated_sources != request.sources
        or result.denied_publisher_roles != roles
        or len(set(roles)) != 5
        or type(result.fresh_h1f_unique_zero_child_active_head) is not bool
        or result.fresh_h1f_unique_zero_child_active_head is not True
        or any(
            source.activation_owner_identity_sha256
            != request.activation_owner_identity_sha256
            for source in request.sources
        )
        or any(
            source.generation != request_generation
            for source in request.sources
        )
        or any(
            source.generation != request_generation
            for source in result.activated_sources
        )
    ):
        return False
    try:
        _digest_text(
            result.fresh_h1f_audit_body_sha256,
            field="fresh_h1f_audit_body_sha256",
        )
        _digest_text(
            result.active_head_body_sha256,
            field="active_head_body_sha256",
        )
        _exact_string(
            result.active_head_version_id, field="active_head_version_id"
        )
    except SourcePublicationError:
        return False
    if type(result.probe_sets) is not tuple or len(result.probe_sets) != 2:
        return False
    first, second = result.probe_sets
    if (
        type(first) is not StabilizationProbeSet
        or type(second) is not StabilizationProbeSet
    ):
        return False
    try:
        first_time = _probe_time(first.observed_at, field="first probe")
        second_time = _probe_time(second.observed_at, field="second probe")
        for probe in result.probe_sets:
            _digest_text(probe.policy_sha256, field="policy_sha256")
            _digest_text(probe.readback_sha256, field="readback_sha256")
    except SourcePublicationError:
        return False
    return (
        second_time - first_time >= timedelta(seconds=10)
        and first.policy_sha256 == second.policy_sha256
        and first.readback_sha256 == second.readback_sha256
        and first.denied_publisher_roles == roles
        and second.denied_publisher_roles == roles
    )


def _preflight_source_runtimes(
    services: SequentialSourceServices,
    *,
    expected_campaign_bucket: str,
    expected_closure_caller_arn: str,
) -> None:
    campaign_bucket = _validate_campaign_bucket(
        expected_campaign_bucket,
        field="expected_campaign_bucket",
    )
    closure_caller = _exact_string(
        expected_closure_caller_arn,
        field="expected_closure_caller_arn",
    )
    if not _is_closure_caller_arn(closure_caller):
        _fail(
            "expected_closure_caller_arn is not an approved-account principal"
        )
    bindings = (
        (
            services.gpu_spend_snapshot_runtime,
            GpuSpendSnapshotPublisherRuntime,
        ),
        (
            services.production_submission_intent_runtime,
            ProductionSubmissionIntentPublisherRuntime,
        ),
        (
            services.production_controller_baseline_runtime,
            ProductionControllerBaselinePublisherRuntime,
        ),
        (
            services.production_control_plane_readiness_runtime,
            ProductionControlPlaneReadinessPublisherRuntime,
        ),
        (
            services.production_submission_acquisition_runtime,
            ProductionSubmissionAcquisitionPublisherRuntime,
        ),
    )
    roles = []
    for runtime, expected_type in bindings:
        if type(runtime) is not expected_type:
            _fail(
                "source runtime is not its exact source-specific binding"
            )
        if type(runtime.services) is not S3PublicationServices:
            _fail("source runtime services are not exact Task 4 services")
        role = _exact_string(
            runtime.publisher_role_arn, field="publisher_role_arn"
        )
        if _ROLE_ARN.fullmatch(role) is None:
            _fail("publisher_role_arn is not a stable IAM role")
        caller = _exact_string(
            runtime.closure_caller_arn, field="closure_caller_arn"
        )
        if not _is_closure_caller_arn(caller):
            _fail("closure_caller_arn is not an approved-account principal")
        if caller != closure_caller:
            _fail(
                "source runtime does not bind the expected closure caller"
            )
        roles.append(role)
        runtime_bucket = _validate_campaign_bucket(
            runtime.campaign_bucket, field="campaign_bucket"
        )
        if runtime_bucket != campaign_bucket:
            _fail(
                "source runtime does not bind the expected campaign bucket"
            )
    if len(set(roles)) != len(roles):
        _fail("publisher roles must be exact and unique before publication")


def activate_sources_sequentially(
    *,
    services: SequentialSourceServices,
    request: SequentialSourceActivationRequest,
) -> SequentialSourceActivationResult:
    """Publish, durably record, and batch-activate the exact five sources."""

    if type(services) is not SequentialSourceServices:
        raise TypeError("services must be exact SequentialSourceServices")
    if type(request) is not SequentialSourceActivationRequest:
        raise TypeError(
            "request must be exact SequentialSourceActivationRequest"
        )
    _exact_string(request.activation_id, field="activation_id")
    _exact_generation(request.generation, field="generation")
    _digest_text(
        request.activation_owner_identity_sha256,
        field="activation_owner_identity_sha256",
    )
    _preflight_source_runtimes(
        services,
        expected_campaign_bucket=request.expected_campaign_bucket,
        expected_closure_caller_arn=request.expected_closure_caller_arn,
    )
    first_request = request.gpu_spend_snapshot
    if type(first_request) is not GpuSpendSnapshotPublicationRequest:
        _fail("first source request does not bind the activation")
    _exact_generation(first_request.generation, field="first generation")
    if (
        first_request.activation_id != request.activation_id
        or first_request.generation != request.generation
        or first_request.caller_arn
        != request.expected_closure_caller_arn
    ):
        _fail("first source request does not bind the activation")

    durable = []
    first = publish_gpu_spend_snapshot(
        runtime=services.gpu_spend_snapshot_runtime,
        request=first_request,
    )
    durable.append(
        _record_publication(
            recorder=services.durable_recorder,
            activation_request=request,
            publication=first,
        )
    )

    steps = (
        (
            "build_production_submission_intent",
            ProductionSubmissionIntentPublicationRequest,
            "production-submission-intent",
            services.production_submission_intent_runtime,
            publish_production_submission_intent,
        ),
        (
            "build_production_controller_baseline",
            ProductionControllerBaselinePublicationRequest,
            "production-controller-baseline",
            services.production_controller_baseline_runtime,
            publish_production_controller_baseline,
        ),
        (
            "build_production_control_plane_readiness",
            ProductionControlPlaneReadinessPublicationRequest,
            "production-control-plane-readiness",
            services.production_control_plane_readiness_runtime,
            publish_production_control_plane_readiness,
        ),
        (
            "build_production_submission_acquisition",
            ProductionSubmissionAcquisitionPublicationRequest,
            "production-submission-acquisition",
            services.production_submission_acquisition_runtime,
            publish_production_submission_acquisition,
        ),
    )
    for method_name, request_type, kind, runtime, publisher in steps:
        next_request = _build_next(
            services.candidate_factory,
            method_name=method_name,
            expected_type=request_type,
            kind=kind,
            durable=tuple(durable),
        )
        _exact_generation(
            next_request.generation,
            field=f"{kind} generation",
        )
        if (
            next_request.activation_id != request.activation_id
            or next_request.generation != request.generation
            or next_request.caller_arn
            != request.expected_closure_caller_arn
        ):
            _fail(f"{kind} request drifted from activation")
        publication = publisher(runtime=runtime, request=next_request)
        durable.append(
            _record_publication(
                recorder=services.durable_recorder,
                activation_request=request,
                publication=publication,
            )
        )
    sources = tuple(durable)
    if tuple(item.source_kind for item in sources) != _SOURCE_ORDER:
        _fail("source publication order drifted")
    batch_request = BatchSuccessorRequest(
        activation_id=request.activation_id,
        generation=request.generation,
        activation_owner_identity_sha256=(
            request.activation_owner_identity_sha256
        ),
        expected_campaign_bucket=request.expected_campaign_bucket,
        expected_closure_caller_arn=(
            request.expected_closure_caller_arn
        ),
        sources=sources,
    )
    activate = _method(services.batch_successor, "activate")
    try:
        batch_result = activate(request=batch_request)
    except Exception as exc:
        raise SourcePublicationError("batch successor invocation failed") from exc
    authoritative = _batch_is_authoritative(
        batch_result, request=batch_request
    )
    return SequentialSourceActivationResult(
        durable_sources=sources,
        batch_request=batch_request,
        batch_result=batch_result,
        sources_authoritative=authoritative,
    )


__all__ = [
    "BatchSuccessorRequest",
    "BatchSuccessorResult",
    "DurableSourceRecord",
    "GpuSpendSnapshotPublicationRequest",
    "ProductionSubmissionIntentPublicationRequest",
    "ProductionControllerBaselinePublicationRequest",
    "ProductionControlPlaneReadinessPublicationRequest",
    "ProductionSubmissionAcquisitionPublicationRequest",
    "GpuSpendSnapshotPublisherRuntime",
    "ProductionSubmissionIntentPublisherRuntime",
    "ProductionControllerBaselinePublisherRuntime",
    "ProductionControlPlaneReadinessPublisherRuntime",
    "ProductionSubmissionAcquisitionPublisherRuntime",
    "SequentialSourceActivationRequest",
    "SequentialSourceActivationResult",
    "SequentialSourceServices",
    "StabilizationProbeSet",
    "SourcePublicationError",
    "SourcePublicationResult",
    "publish_gpu_spend_snapshot",
    "publish_production_submission_intent",
    "publish_production_controller_baseline",
    "publish_production_control_plane_readiness",
    "publish_production_submission_acquisition",
    "activate_sources_sequentially",
]
