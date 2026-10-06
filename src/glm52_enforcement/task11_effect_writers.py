"""Fixed Task 11 effect writers built on accepted canonical primitives."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .s3_adapter import (
    ConditionalCreateResult,
    S3CreateActionRequest,
    S3PublicationServices,
    conditional_create_immutable_json,
)
from .dynamodb import TransactionResolution, WriteOutcome
from .s3_keys import (
    fence_successor_s3_key,
    production_generation_claim_s3_key,
    production_generation_start_decision_s3_key,
    production_generation_terminal_s3_key,
    sky_post_handoff_s3_key,
)
from .s3_records import S3ObjectIdentity, build_immutable_json_candidate
from .h1f_adapter import (
    H1fAuditResult,
    reconcile_exact_candidate,
    validate_h1f_audit_result,
)
from .task12_correlation import build_sky_post_handoff_record


WRITER_KINDS = (
    "FenceSuccessor",
    "ClaimWriter",
    "DecisionWriter",
    "TerminalV1Writer",
    "ClosureHandoff",
)
_RECORD_KINDS = {
    "FenceSuccessor": "fence-successor",
    "ClaimWriter": "generation-claim",
    "DecisionWriter": "start-decision",
    "TerminalV1Writer": "generation-terminal",
    "ClosureHandoff": "sky-post-handoff",
}


class Task11EffectWriterError(ValueError):
    """A fixed effect builder or conditional create failed closed."""


@dataclass(frozen=True)
class EffectWriterRequest:
    writer_kind: str
    activation_id: str
    generation: int
    action_key: str
    campaign_bucket: str
    builder_arguments: Mapping[str, object]
    preauthorized_audit: H1fAuditResult | None = None


@dataclass(frozen=True)
class EffectWriteResult:
    writer_kind: str
    record_kind: str
    object_identity: S3ObjectIdentity
    candidate_identity_sha256: str
    authority_audit_body_sha256: str
    closing_revision: int
    authorized_revision: int
    direct_request_id: str
    direct_server_date: str
    direct_response_authenticated: bool
    raw: bytes
    canonical_identity_sha256: str
    direct_request_started_at: str | None = None
    direct_response_received_at: str | None = None


def _fail(message: str) -> None:
    raise Task11EffectWriterError(message)


def _artifact(value: object) -> object:
    from .glm52_sky_production_submission import VersionedJsonArtifact

    if type(value) is not dict or set(value) != {
        "key",
        "version_id",
        "record",
    }:
        _fail("effect writer artifact field set drifted")
    if (
        type(value["key"]) is not str
        or not value["key"]
        or type(value["version_id"]) is not str
        or not value["version_id"]
        or type(value["record"]) is not dict
    ):
        _fail("effect writer artifact identity drifted")
    return VersionedJsonArtifact(
        key=value["key"],
        version_id=value["version_id"],
        raw=canonical_json_bytes(value["record"]) + b"\n",
    )


def _generation_inventory(value: object) -> list[object]:
    from .glm52_sky_production_generation import GenerationInventoryEntry

    if type(value) is not list:
        _fail("generation inventory must be a list")
    try:
        return [
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
            for row in value
        ]
    except (KeyError, TypeError) as exc:
        raise Task11EffectWriterError(
            "generation inventory is malformed"
        ) from exc


def _fence_rows(
    value: object,
    *,
    row_type: type,
) -> tuple[object, ...]:
    from .glm52_sky_production_fence import FenceControlArtifact

    if type(value) is not list:
        _fail("fence row set must be a list")
    result = []
    for row in value:
        if type(row) is not dict:
            _fail("fence row is malformed")
        values = dict(row)
        if row_type is FenceControlArtifact:
            record = values.pop("record", None)
            if type(record) is not dict:
                _fail("fence control record is malformed")
            values["raw"] = canonical_json_bytes(record) + b"\n"
        if "metadata" in values:
            metadata = values["metadata"]
            if type(metadata) is not list:
                _fail("fence metadata is malformed")
            values["metadata"] = tuple(tuple(item) for item in metadata)
        try:
            result.append(row_type(**values))
        except TypeError as exc:
            raise Task11EffectWriterError(
                "fence row is malformed"
            ) from exc
    return tuple(result)


def _build_fence_successor(
    arguments: Mapping[str, object],
) -> Mapping[str, object]:
    from .glm52_sky_production_fence import (
        FenceControlArtifact,
        FenceLiveStateIdentity,
        GenerationReservation,
        SourceEnrollment,
        build_fence_successor,
    )

    try:
        return build_fence_successor(
            predecessor_controls=_fence_rows(
                arguments["predecessor_controls"],
                row_type=FenceControlArtifact,
            ),
            descriptor=_artifact(arguments["descriptor"]),
            intent=_artifact(arguments["intent"]),
            approval=_artifact(arguments["approval"]),
            controller_baseline=_artifact(
                arguments["controller_baseline"]
            ),
            must_start_control_plane_ready=_artifact(
                arguments["must_start_control_plane_ready"]
            ),
            submission_acquisition=_artifact(
                arguments["submission_acquisition"]
            ),
            gpu_spend_snapshot=_artifact(
                arguments["gpu_spend_snapshot"]
            ),
            added_sources=_fence_rows(
                arguments["added_sources"],
                row_type=SourceEnrollment,
            ),
            added_generation_reservations=_fence_rows(
                arguments["added_generation_reservations"],
                row_type=GenerationReservation,
            ),
            live_state=FenceLiveStateIdentity(**arguments["live_state"]),
            selected_at=arguments["selected_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Task11EffectWriterError(
            "fence successor canonical builder failed"
        ) from exc


def _build_claim(arguments: Mapping[str, object]) -> Mapping[str, object]:
    from .glm52_sky_production_generation import (
        build_production_generation_claim,
    )

    try:
        return build_production_generation_claim(
            generation=arguments["generation"],
            descriptor=_artifact(arguments["descriptor"]),
            intent=_artifact(arguments["intent"]),
            approval=_artifact(arguments["approval"]),
            controller_baseline=_artifact(
                arguments["controller_baseline"]
            ),
            must_start_control_plane_ready=_artifact(
                arguments["must_start_control_plane_ready"]
            ),
            submission_acquisition=_artifact(
                arguments["submission_acquisition"]
            ),
            gpu_spend_snapshot=_artifact(
                arguments["gpu_spend_snapshot"]
            ),
            generation_inventory=_generation_inventory(
                arguments["generation_inventory"]
            ),
            previous_generation_terminal=(
                None
                if arguments["previous_generation_terminal"] is None
                else _artifact(arguments["previous_generation_terminal"])
            ),
            submit_attempt_id=arguments["submit_attempt_id"],
            claim_created_at=arguments["claim_created_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Task11EffectWriterError(
            "generation claim canonical builder failed"
        ) from exc


def _build_decision(arguments: Mapping[str, object]) -> Mapping[str, object]:
    from .glm52_sky_production_generation import (
        build_production_generation_start_decision,
    )

    try:
        return build_production_generation_start_decision(
            generation_claim=_artifact(arguments["generation_claim"]),
            descriptor=_artifact(arguments["descriptor"]),
            intent=_artifact(arguments["intent"]),
            approval=_artifact(arguments["approval"]),
            controller_baseline=_artifact(
                arguments["controller_baseline"]
            ),
            must_start_control_plane_ready=_artifact(
                arguments["must_start_control_plane_ready"]
            ),
            submission_acquisition=_artifact(
                arguments["submission_acquisition"]
            ),
            decision=arguments["decision"],
            decided_at=arguments["decided_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Task11EffectWriterError(
            "start decision canonical builder failed"
        ) from exc


def _build_terminal(arguments: Mapping[str, object]) -> Mapping[str, object]:
    from .glm52_sky_production_generation import (
        build_production_generation_terminal,
    )

    try:
        return build_production_generation_terminal(
            generation_claim=_artifact(arguments["generation_claim"]),
            start_decision=_artifact(arguments["start_decision"]),
            final_gpu_spend_snapshot=_artifact(
                arguments["final_gpu_spend_snapshot"]
            ),
            terminal_at=arguments["terminal_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Task11EffectWriterError(
            "generation terminal canonical builder failed"
        ) from exc


def _build_handoff(arguments: Mapping[str, object]) -> Mapping[str, object]:
    try:
        return build_sky_post_handoff_record(**dict(arguments))
    except (TypeError, ValueError) as exc:
        raise Task11EffectWriterError(
            "closure handoff canonical fields drifted"
        ) from exc


_BUILDERS = {
    "FenceSuccessor": _build_fence_successor,
    "ClaimWriter": _build_claim,
    "DecisionWriter": _build_decision,
    "TerminalV1Writer": _build_terminal,
    "ClosureHandoff": _build_handoff,
}


def _key(
    writer_kind: str,
    *,
    record: Mapping[str, object],
    generation: int,
) -> str:
    if writer_kind == "FenceSuccessor":
        return fence_successor_s3_key(
            run_id="glm52-sky-20260724",
            predecessor_body_sha256=record[
                "predecessor_control_body_sha256"
            ],
        )
    if writer_kind == "ClaimWriter":
        return production_generation_claim_s3_key(
            run_id="glm52-sky-20260724",
            generation=generation,
        )
    if writer_kind == "DecisionWriter":
        return production_generation_start_decision_s3_key(
            run_id="glm52-sky-20260724",
            generation=generation,
        )
    if writer_kind == "TerminalV1Writer":
        return production_generation_terminal_s3_key(
            run_id="glm52-sky-20260724",
            generation=generation,
        )
    return sky_post_handoff_s3_key(
        run_id="glm52-sky-20260724",
        generation=generation,
    )


def prepare_effect_candidate(
    request: EffectWriterRequest,
) -> object:
    """Build the exact immutable candidate before any Task 4 mutation."""

    if (
        type(request) is not EffectWriterRequest
        or request.writer_kind not in WRITER_KINDS
        or type(request.builder_arguments) is not dict
    ):
        raise TypeError("effect writer request is not exact")
    record = _BUILDERS[request.writer_kind](request.builder_arguments)
    raw = canonical_json_bytes(record) + b"\n"
    return build_immutable_json_candidate(
        record_kind=_RECORD_KINDS[request.writer_kind],
        bucket=request.campaign_bucket,
        key=_key(
            request.writer_kind,
            record=record,
            generation=request.generation,
        ),
        raw=raw,
        activation_id=request.activation_id,
        generation=request.generation,
    )


def _write(
    *,
    expected_kind: str,
    services: S3PublicationServices,
    request: EffectWriterRequest,
) -> EffectWriteResult:
    if (
        type(request) is not EffectWriterRequest
        or request.writer_kind != expected_kind
        or expected_kind not in WRITER_KINDS
        or type(services) is not S3PublicationServices
        or type(request.builder_arguments) is not dict
    ):
        raise TypeError("effect writer request/services are not exact")
    candidate = prepare_effect_candidate(request)
    raw = candidate.raw
    try:
        result = conditional_create_immutable_json(
            services=services,
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
        raise Task11EffectWriterError(
            expected_kind + " Task 4 publication failed"
        ) from exc
    if (
        type(result) is not ConditionalCreateResult
        or result.outcome != "created-direct"
        or result.object_identity is None
        or result.direct_response_authenticated is not True
        or type(result.direct_request_id) is not str
        or type(result.direct_server_date) is not str
        or type(result.direct_request_started_at) is not str
        or type(result.direct_response_received_at) is not str
    ):
        _fail(expected_kind + " lost authenticated direct response custody")
    body = {
        "writer_kind": expected_kind,
        "record_kind": _RECORD_KINDS[expected_kind],
        "object_identity_sha256": (
            result.object_identity.canonical_identity_sha256
        ),
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "authority_audit_body_sha256": (
            result.authority_audit_body_sha256
        ),
        "closing_revision": result.closing_revision,
        "authorized_revision": result.authorized_revision,
        "direct_request_id": result.direct_request_id,
        "direct_server_date": result.direct_server_date,
        "direct_request_started_at": result.direct_request_started_at,
        "direct_response_received_at": (
            result.direct_response_received_at
        ),
    }
    return EffectWriteResult(
        writer_kind=expected_kind,
        record_kind=_RECORD_KINDS[expected_kind],
        object_identity=result.object_identity,
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        authority_audit_body_sha256=result.authority_audit_body_sha256,
        closing_revision=result.closing_revision,
        authorized_revision=result.authorized_revision,
        direct_request_id=result.direct_request_id,
        direct_server_date=result.direct_server_date,
        direct_request_started_at=result.direct_request_started_at,
        direct_response_received_at=result.direct_response_received_at,
        direct_response_authenticated=True,
        raw=raw,
        canonical_identity_sha256=canonical_sha256(body),
    )


def write_fence_successor(
    *, services: S3PublicationServices, request: EffectWriterRequest
) -> EffectWriteResult:
    return _write(
        expected_kind="FenceSuccessor",
        services=services,
        request=request,
    )


def write_generation_claim(
    *, services: S3PublicationServices, request: EffectWriterRequest
) -> EffectWriteResult:
    return _write(
        expected_kind="ClaimWriter",
        services=services,
        request=request,
    )


def write_start_decision(
    *, services: S3PublicationServices, request: EffectWriterRequest
) -> EffectWriteResult:
    return _write(
        expected_kind="DecisionWriter",
        services=services,
        request=request,
    )


def write_terminal_v1(
    *, services: S3PublicationServices, request: EffectWriterRequest
) -> EffectWriteResult:
    return _write(
        expected_kind="TerminalV1Writer",
        services=services,
        request=request,
    )


def write_closure_handoff(
    *, services: S3PublicationServices, request: EffectWriterRequest
) -> EffectWriteResult:
    if request.preauthorized_audit is not None:
        return _write_prepared_handoff(
            services=services,
            request=request,
        )
    return _write(
        expected_kind="ClosureHandoff",
        services=services,
        request=request,
    )


def _write_prepared_handoff(
    *,
    services: S3PublicationServices,
    request: EffectWriterRequest,
) -> EffectWriteResult:
    """Consume the exact decision-prepared H.1f audit, then PUT once."""

    candidate = prepare_effect_candidate(request)
    audit = validate_h1f_audit_result(request.preauthorized_audit)
    if (
        audit.authority_domain != "ACTIVATION"
        or audit.operation_kind != "S3_CREATE"
        or audit.action_key != request.action_key
        or audit.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or audit.activation_id != request.activation_id
        or audit.generation != request.generation
    ):
        _fail("prepared handoff audit does not bind the exact candidate")
    consume = getattr(services.actions, "consume_audit", None)
    if not callable(consume):
        _fail("prepared handoff consume boundary is absent")
    try:
        resolution = consume(
            request=S3CreateActionRequest(
                authority_domain="ACTIVATION",
                operation_kind="S3_CREATE",
                action_key=request.action_key,
                candidate=candidate,
                activation_id=request.activation_id,
                generation=request.generation,
            ),
            authority_audit_body_sha256=audit.canonical_body_sha256,
            closing_revision=audit.closing_revision,
            authorized_revision=audit.expected_authorized_revision,
        )
    except Exception as exc:
        raise Task11EffectWriterError(
            "ClosureHandoff prepared consume failed"
        ) from exc
    if (
        type(resolution) is not TransactionResolution
        or resolution.outcome is not WriteOutcome.EXACT_LIVE_OWNER_COMMIT
        or not resolution.may_issue_external_side_effect
        or len(resolution.records) != 3
    ):
        _fail("prepared handoff consume lost exact owner")
    actions = tuple(
        row
        for row in resolution.records
        if type(row) is dict
        and row.get("record_type") == "glm52_production_action"
    )
    if (
        len(actions) != 1
        or actions[0].get("state") != "CONSUMED"
        or actions[0].get("authority_audit_body_sha256")
        != audit.canonical_body_sha256
        or actions[0].get("authority_audit_closing_revision")
        != audit.closing_revision
    ):
        _fail("prepared handoff consume readback drifted")
    checksum = base64.b64encode(
        hashlib.sha256(candidate.raw).digest()
    ).decode("ascii")
    put = getattr(services.s3, "put_object", None)
    if not callable(put):
        _fail("prepared handoff S3 boundary is absent")
    try:
        response = put(
            Bucket=candidate.bucket,
            Key=candidate.key,
            Body=candidate.raw,
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ContentType="application/json",
            Metadata=dict(candidate.metadata),
            ExpectedBucketOwner="246813579024",
        )
    except Exception as exc:
        raise Task11EffectWriterError(
            "ClosureHandoff direct PUT was ambiguous"
        ) from exc
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
    try:
        server = parsedate_to_datetime(headers["date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Task11EffectWriterError(
            "ClosureHandoff direct response date drifted"
        ) from exc
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(response.get("VersionId")) is not str
        or response["VersionId"] == "null"
        or type(response.get("ETag")) is not str
        or response.get("ChecksumSHA256") != checksum
        or server.tzinfo is None
    ):
        _fail("ClosureHandoff direct response is unauthenticated")
    reconciliation = reconcile_exact_candidate(
        s3=services.s3,
        candidate=candidate,
    )
    identity = reconciliation.object_identity
    if (
        reconciliation.state != "sole-version"
        or identity is None
        or identity.version_id != response["VersionId"]
        or identity.etag != response["ETag"]
        or identity.checksum_sha256_base64 != checksum
    ):
        _fail("ClosureHandoff exact-version reconciliation drifted")
    server_date = server.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    body = {
        "writer_kind": "ClosureHandoff",
        "record_kind": _RECORD_KINDS["ClosureHandoff"],
        "object_identity_sha256": identity.canonical_identity_sha256,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "authority_audit_body_sha256": audit.canonical_body_sha256,
        "closing_revision": audit.closing_revision,
        "authorized_revision": audit.expected_authorized_revision,
        "direct_request_id": metadata["RequestId"],
        "direct_server_date": server_date,
        "direct_request_started_at": server_date,
        "direct_response_received_at": server_date,
    }
    return EffectWriteResult(
        writer_kind="ClosureHandoff",
        record_kind=_RECORD_KINDS["ClosureHandoff"],
        object_identity=identity,
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        authority_audit_body_sha256=audit.canonical_body_sha256,
        closing_revision=audit.closing_revision,
        authorized_revision=audit.expected_authorized_revision,
        direct_request_id=metadata["RequestId"],
        direct_server_date=server_date,
        direct_response_authenticated=True,
        raw=candidate.raw,
        canonical_identity_sha256=canonical_sha256(body),
        direct_request_started_at=server_date,
        direct_response_received_at=server_date,
    )


__all__ = [
    "EffectWriteResult",
    "EffectWriterRequest",
    "Task11EffectWriterError",
    "WRITER_KINDS",
    "prepare_effect_candidate",
    "write_closure_handoff",
    "write_fence_successor",
    "write_generation_claim",
    "write_start_decision",
    "write_terminal_v1",
]
