"""Live Task 12 snapshot/finalization request materialization.

This module never accepts or persists a prepared request for a future
operation.  A request is built only when its complete authority is available
from the exact canonical source families selected by the descriptor and, where
required, an authenticated live AWS observation.

Operations whose declared sources do not contain enough authority fail before
the domain effect.  That failure is intentional: inventing a transition plan,
writer authority, snapshot expectation, orphan baseline, or owner nonce would
turn a descriptor into a second source of truth.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import json
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .records import (
    canonical_record_identity,
    ledger_pk,
    ledger_sk,
    validate_record,
)
from .task12_operations import (
    OPERATION_SOURCE_PRODUCERS,
    validate_operation_request_payload,
)


RUN_ID = "glm52-sky-20260724"
_H1G_MODEL_BUCKET = "keep-glm52-models-246813579024-us-west-2"
_H1G_TERMINAL_KEY = (
    "campaigns/glm52-sky-20260724/submissions/production/generations/"
    "00000001/terminal/PRODUCTION_TERMINAL_V2.json"
)
_H1G_SPEND_LEDGER_KEY = (
    "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER.jsonl"
)

SUPPORTED_OPERATIONS = frozenset(
    {
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
        "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
        "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT",
        "RETAINED_AUDIT_SUPPORT_ORPHANS",
        "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
        "RETAINED_INVOKE_H1G_DRAINED_WRITER",
        "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE",
        "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK",
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
    }
)

_DIRECT_OPERATIONS = frozenset(
    {
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
        "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
        "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT",
        "RETAINED_AUDIT_SUPPORT_ORPHANS",
        "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
        "RETAINED_INVOKE_H1G_DRAINED_WRITER",
        "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE",
        "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK",
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
    }
)

_SOURCE_TYPES: Mapping[str, str] = {
    "finalization_control": "glm52_production_finalization_control",
    "support_finalized": "glm52_production_support_plane_finalized",
    "snapshot_cleanup_control": (
        "glm52_production_snapshot_cleanup_control"
    ),
    "snapshot_cleanup_action": "glm52_production_snapshot_cleanup_action",
    "snapshot_cleanup_transition": (
        "glm52_production_snapshot_cleanup_transition"
    ),
}

class Task12LiveSnapshotFinalizationError(ValueError):
    """A live source, observation, or producer boundary is incomplete."""


def _fail(message: str) -> None:
    raise Task12LiveSnapshotFinalizationError(message)


def _validate_invocation(invocation: object, operation_kind: str) -> None:
    generation = getattr(invocation, "generation", None)
    if (
        getattr(invocation, "operation_kind", None) != operation_kind
        or type(getattr(invocation, "activation_id", None)) is not str
        or not getattr(invocation, "activation_id")
        or type(getattr(invocation, "activation_ordinal", None)) is not int
        or getattr(invocation, "activation_ordinal") < 1
        or type(generation) is not int
        or generation < 1
        or getattr(invocation, "generation_text", None)
        != f"{generation:08d}"
    ):
        _fail(operation_kind + " invocation identity drifted")


def _validate_live_sources(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    _validate_invocation(invocation, operation_kind)
    expected = {
        alias for alias, _producer in OPERATION_SOURCE_PRODUCERS[operation_kind]
    }
    if type(live_sources) is not dict or set(live_sources) != expected:
        _fail(operation_kind + " live source set drifted")

    exact: dict[str, dict[str, object]] = {}
    campaign_identity: str | None = None
    for alias in sorted(expected):
        record_type = _SOURCE_TYPES.get(alias)
        if type(record_type) is not str:
            _fail(alias + " has no canonical source family")
        try:
            record = validate_record(record_type, live_sources[alias])
        except (TypeError, ValueError) as exc:
            raise Task12LiveSnapshotFinalizationError(
                alias + " canonical source drifted"
            ) from exc
        if (
            record.get("activation_id")
            != getattr(invocation, "activation_id")
            or record.get("activation_ordinal")
            != getattr(invocation, "activation_ordinal")
        ):
            _fail(alias + " activation source drifted")
        current_campaign = record.get("campaign_identity_sha256")
        if type(current_campaign) is not str:
            _fail(alias + " campaign identity is absent")
        if campaign_identity is None:
            campaign_identity = current_campaign
        elif current_campaign != campaign_identity:
            _fail(operation_kind + " campaign source identities drifted")
        exact[alias] = record
    return exact


def _metadata(response: object, label: str) -> Mapping[str, object]:
    metadata = response.get("ResponseMetadata") if type(response) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        _fail(label + " response is not authenticated zero-retry")
    return metadata


def _snapshot_capture_and_schedule(
    *,
    control: Mapping[str, object],
    ports: object,
    include_describe_evidence: bool = False,
) -> dict[str, object]:
    from .task12_snapshot_cleanup import (
        SnapshotCapture,
        SnapshotObservation,
        SnapshotSchedule,
    )

    client = getattr(ports, "client", None)
    if not callable(client):
        _fail("snapshot validation has no live EC2 boundary")
    try:
        response = client("ec2").describe_snapshots(
            SnapshotIds=[control["snapshot_id"]]
        )
    except Exception as exc:
        error_response = getattr(exc, "response", None)
        error = (
            error_response.get("Error")
            if type(error_response) is dict
            else None
        )
        metadata = (
            error_response.get("ResponseMetadata")
            if type(error_response) is dict
            else None
        )
        if (
            not include_describe_evidence
            or type(error) is not dict
            or error.get("Code") != "InvalidSnapshot.NotFound"
            or type(error.get("Message")) is not str
            or control["snapshot_id"] not in error["Message"]
            or type(metadata) is not dict
            or metadata.get("HTTPStatusCode") not in {400, 404}
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
            or metadata.get("RetryAttempts") != 0
        ):
            raise Task12LiveSnapshotFinalizationError(
                "snapshot describe failed without exact absence proof"
            ) from exc
        absence_body = {
            "snapshot_id": control["snapshot_id"],
            "snapshot_present": False,
            "error_code": "InvalidSnapshot.NotFound",
            "request_id": metadata["RequestId"],
        }
        return {
            "capture": None,
            "schedule": None,
            "snapshot_present": False,
            "describe_request_id": metadata["RequestId"],
            "describe_response_sha256": canonical_sha256(absence_body),
            "observation_identity_sha256": canonical_sha256(
                absence_body
            ),
        }
    metadata = _metadata(response, "snapshot describe")
    snapshots = response.get("Snapshots")
    if include_describe_evidence and snapshots == []:
        absence_body = {
            "snapshot_id": control["snapshot_id"],
            "snapshot_present": False,
            "request_id": metadata["RequestId"],
        }
        return {
            "capture": None,
            "schedule": None,
            "snapshot_present": False,
            "describe_request_id": metadata["RequestId"],
            "describe_response_sha256": canonical_sha256(absence_body),
            "observation_identity_sha256": canonical_sha256(
                absence_body
            ),
        }
    if (
        type(snapshots) is not list
        or len(snapshots) != 1
        or type(snapshots[0]) is not dict
    ):
        _fail("snapshot describe response is not exactly singular")
    item = snapshots[0]
    started = item.get("StartTime")
    if (
        type(started) is not datetime
        or started.tzinfo is None
        or started.utcoffset() is None
        or started.microsecond != 0
    ):
        _fail("snapshot describe start time is not normalized")
    raw_tags = item.get("Tags", [])
    if (
        type(raw_tags) is not list
        or any(
            type(tag) is not dict
            or set(tag) != {"Key", "Value"}
            or type(tag["Key"]) is not str
            or not tag["Key"]
            or type(tag["Value"]) is not str
            for tag in raw_tags
        )
        or len({tag["Key"] for tag in raw_tags}) != len(raw_tags)
    ):
        _fail("snapshot describe tags are not exact")
    tags = sorted(
        (
            {"Key": tag["Key"], "Value": tag["Value"]}
            for tag in raw_tags
        ),
        key=lambda tag: (str(tag["Key"]), str(tag["Value"])),
    )
    observed_at = started.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    tag_identity = canonical_sha256(tags)
    response_body = {
        "snapshot_id": item.get("SnapshotId"),
        "source_volume_id": item.get("VolumeId"),
        "kms_key_arn": item.get("KmsKeyId"),
        "snapshot_tags_sha256": tag_identity,
        "encrypted": item.get("Encrypted"),
        "state": item.get("State"),
        "observed_at": observed_at,
        "request_id": metadata["RequestId"],
    }
    try:
        observation = SnapshotObservation(
            snapshot_id=item.get("SnapshotId"),
            source_volume_id=item.get("VolumeId"),
            kms_key_arn=item.get("KmsKeyId"),
            snapshot_tags_sha256=tag_identity,
            encrypted=item.get("Encrypted"),
            state=item.get("State"),
            observed_at=observed_at,
            describe_request_id=metadata["RequestId"],
            describe_response_sha256=canonical_sha256(response_body),
        )
        capture = SnapshotCapture.discover(
            observations=(observation,),
            expected_source_volume_id=control["source_volume_id"],
            expected_kms_key_arn=observation.kms_key_arn,
            expected_snapshot_tags_sha256=control["snapshot_tags_sha256"],
        )
        schedule = SnapshotSchedule.validate(
            capture=capture,
            schedule_arn=control["schedule_arn"],
            delete_not_before=control["delete_not_before"],
            schedule_input_sha256=control["schedule_input_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            "snapshot validation live observation drifted"
        ) from exc
    if capture.snapshot_identity_sha256 != control["snapshot_identity_sha256"]:
        _fail("snapshot validation source identity drifted")
    result = {"capture": asdict(capture), "schedule": asdict(schedule)}
    if include_describe_evidence:
        result.update(
            snapshot_present=True,
            describe_request_id=metadata["RequestId"],
            describe_response_sha256=observation.describe_response_sha256,
            observation_identity_sha256=(
                capture.capture_evidence_sha256
            ),
        )
    return result


def _audit_delete_request(
    *, invocation: object, action: Mapping[str, object]
) -> dict[str, object]:
    attempt = action.get("attempt")
    if (
        action.get("state") != "ARMED"
        or type(attempt) is not int
        or attempt < 1
    ):
        _fail("snapshot delete action attempt is not authenticated")
    return {
        "partition_key": ledger_pk(RUN_ID),
        "sort_key": ledger_sk(
            "glm52_production_snapshot_cleanup_action",
            activation_id=getattr(invocation, "activation_id"),
            attempt=attempt,
        ),
        "expected_record_identity_sha256": canonical_record_identity(
            "glm52_production_snapshot_cleanup_action", action
        ),
    }


def _retained_roles(ports: object) -> Mapping[str, object]:
    deployment = getattr(ports, "deployment", None)
    roles = getattr(deployment, "role_coordinates", None)
    if not isinstance(roles, Mapping):
        _fail("retained deployment coordinates are absent")
    return roles


def _retained_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _retained_writer_request(
    *,
    writer_kind: str,
    record_type: str,
    control: Mapping[str, object],
    invocation: object,
    ports: object,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from .task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
        build_retained_writer_candidate,
        validate_retained_writer_authority,
    )

    if control.get("state") not in {"OWNED", "SNAPSHOT_DISPOSITION_RECORDED"}:
        _fail(writer_kind + " finalization control is not write-ready")
    owner_nonce = control.get("owner_invocation_nonce_sha256")
    revision = control.get("revision")
    if (
        type(owner_nonce) is not str
        or len(owner_nonce) != 64
        or type(revision) is not int
        or revision < 1
    ):
        _fail(writer_kind + " finalization owner authority is absent")
    published_at = _retained_now()
    record_body = {
        "schema_version": 1,
        "record_type": record_type,
        "account_id": control["account_id"],
        "region": control["region"],
        "run_id": control["run_id"],
        "campaign_identity_sha256": control[
            "campaign_identity_sha256"
        ],
        "activation_id": control["activation_id"],
        "activation_ordinal": control["activation_ordinal"],
        "writer_function_version_arn": getattr(
            invocation, "invoked_function_version_arn", None
        ),
        "writer_dispatch_identity_sha256": getattr(
            invocation, "dispatch_identity_sha256", None
        ),
        "writer_invocation_nonce_sha256": owner_nonce,
        "published_at": published_at,
    }
    if evidence is not None:
        if writer_kind != "H1GDrained" or type(evidence) is not dict:
            _fail(writer_kind + " retained evidence is not closed")
        record_body.update(evidence)
    record = validate_record(
        record_type,
        {
            **record_body,
            "canonical_body_sha256": canonical_sha256(record_body),
        },
    )
    campaign_bucket = _retained_roles(ports).get("campaign_bucket")
    candidate = build_retained_writer_candidate(
        writer_kind=writer_kind,
        campaign_bucket=campaign_bucket,
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
        authority_domain="FINALIZATION",
        record=record,
    )
    authorized_revision = revision + 1
    action_key = (
        "ACTIVATION#"
        + candidate.activation_id
        + "#FINALIZATION_ACTION#"
        + candidate.action_kind
        + "#"
        + f"{authorized_revision:08d}"
    )
    action_body = {
        "authority_domain": candidate.authority_domain,
        "action_kind": candidate.action_kind,
        "action_key": action_key,
        "candidate_identity_sha256": (
            candidate.candidate_identity_sha256
        ),
        "owner_invocation_nonce_sha256": owner_nonce,
        "state": "CONSUMED",
        "authorized_revision": authorized_revision,
    }
    action = RetainedWriterActionAuthority(
        **action_body,
        action_identity_sha256=canonical_sha256(action_body),
    )
    audit_body = {
        "authority_domain": action.authority_domain,
        "action_kind": action.action_kind,
        "action_key": action.action_key,
        "candidate_identity_sha256": (
            action.candidate_identity_sha256
        ),
        "action_identity_sha256": action.action_identity_sha256,
        "audit_kind": "H1F_GENESIS_TO_ZERO_CHILD",
        "closing_revision": revision,
        "authorized_revision": authorized_revision,
        "current_revision": authorized_revision,
        "observed_at": published_at,
    }
    audit = RetainedWriterAuditAuthority(
        **audit_body,
        canonical_identity_sha256=canonical_sha256(audit_body),
    )
    validate_retained_writer_authority(
        candidate=candidate, action=action, audit=audit
    )
    return {
        "writer_kind": writer_kind,
        "authority_domain": "FINALIZATION",
        "record": record,
        "action": asdict(action),
        "audit": asdict(audit),
    }


def _retained_table_name(ports: object) -> str:
    value = _retained_roles(ports).get("ledger_table_name")
    if type(value) is not str or not value:
        _fail("retained ledger table coordinate is absent")
    return value


def _retained_prior_nonce_capsule(
    *, invocation: object, control: Mapping[str, object]
) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if isinstance(operation_input, Mapping)
        else None
    )
    result = prior.get("result") if isinstance(prior, Mapping) else None
    capsule = (
        result.get("owner_nonce_capsule")
        if isinstance(result, Mapping)
        else None
    )
    context = (
        capsule.get("encryption_context")
        if isinstance(capsule, Mapping)
        else None
    )
    try:
        acquisition_revision = int(
            context.get("control_revision", "")
            if isinstance(context, Mapping)
            else ""
        )
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            "retained finalization nonce revision is not typed"
        ) from exc
    expected = {
        "account_id": str(control.get("account_id")),
        "region": str(control.get("region")),
        "run_id": str(control.get("run_id")),
        "activation_id": str(control.get("activation_id")),
        "authority_domain": "FINALIZATION",
        "owner_execution_arn": str(control.get("owner_execution_arn")),
        "owner_state_machine_version_arn": str(
            control.get("owner_state_machine_version_arn")
        ),
        "owner_attempt": str(control.get("owner_attempt")),
        "barrier_nonce_sha256": str(
            control.get("finalization_barrier_nonce_sha256")
        ),
        "control_revision": str(acquisition_revision),
        "owner_hard_expires_at": str(
            control.get("owner_hard_expires_at")
        ),
    }
    if (
        type(capsule) is not dict
        or type(context) is not dict
        or type(control.get("revision")) is not int
        or acquisition_revision < 1
        or acquisition_revision > control["revision"]
        or context != {key: expected[key] for key in sorted(expected)}
        or capsule.get("nonce_sha256")
        != control.get("owner_invocation_nonce_sha256")
    ):
        _fail("retained finalization nonce continuation is absent or foreign")
    body = dict(capsule)
    identity = body.pop("canonical_body_sha256", None)
    if identity != canonical_sha256(body):
        _fail("retained finalization nonce capsule identity drifted")
    return dict(capsule)


def _retained_decrypt_owner_nonce(
    *, invocation: object, control: Mapping[str, object], ports: object
) -> bytes:
    from .task12_nonce_capsule import decrypt_owner_nonce_capsule

    capsule = _retained_prior_nonce_capsule(
        invocation=invocation, control=control
    )
    context = capsule["encryption_context"]
    expected = {
        "account_id": control["account_id"],
        "region": control["region"],
        "run_id": control["run_id"],
        "activation_id": control["activation_id"],
        "authority_domain": "FINALIZATION",
        "owner_execution_arn": control["owner_execution_arn"],
        "owner_state_machine_version_arn": control[
            "owner_state_machine_version_arn"
        ],
        "owner_attempt": control["owner_attempt"],
        "barrier_nonce_sha256": control[
            "finalization_barrier_nonce_sha256"
        ],
        "control_revision": int(context["control_revision"]),
        "owner_hard_expires_at": control["owner_hard_expires_at"],
    }
    raw = decrypt_owner_nonce_capsule(
        ports=ports,
        capsule=capsule,
        expected_authority=expected,
    )
    if hashlib.sha256(raw).hexdigest() != control.get(
        "owner_invocation_nonce_sha256"
    ):
        _fail("retained finalization nonce plaintext drifted")
    return raw


def _retained_read_record(
    *,
    ports: object,
    activation_id: str,
    record_type: str,
) -> dict[str, object]:
    from .dynamodb import decode_item, encode_item

    sort_key = (
        ledger_sk(record_type)
        if record_type == "glm52_production_activation_index"
        else ledger_sk(record_type, activation_id=activation_id)
    )
    partition_key = ledger_pk(RUN_ID)
    try:
        response = ports.client("dynamodb").get_item(
            TableName=_retained_table_name(ports),
            Key=encode_item({"PK": partition_key, "SK": sort_key}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
    except Exception as exc:
        raise Task12LiveSnapshotFinalizationError(
            "exact retained successor state read failed"
        ) from exc
    _metadata(response, "exact retained successor state read")
    item = response.get("Item")
    if type(item) is not dict:
        _fail("exact retained successor state is absent")
    physical = decode_item(item)
    if (
        physical.pop("PK", None) != partition_key
        or physical.pop("SK", None) != sort_key
    ):
        _fail("exact retained successor coordinate drifted")
    try:
        return validate_record(record_type, physical)
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            "exact retained successor record drifted"
        ) from exc


def _retained_persist_writer_successor(
    *,
    operation_kind: str,
    invocation: object,
    source_control: Mapping[str, object],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> None:
    from .dynamodb import (
        DynamoLedgerAdapter,
        ExactCheck,
        ExactUpdate,
        LedgerKey,
        WriteOutcome,
    )
    from .task12_retained_state import build_finalization_progress_plan
    from .task12_writers import (
        RetainedWriteResult,
        build_retained_writer_candidate,
    )

    if type(domain_result) is not RetainedWriteResult:
        _fail("writer successor requires the exact domain effect")
    writer_kind = request.get("writer_kind")
    record = request.get("record")
    if type(writer_kind) is not str or type(record) is not dict:
        _fail("writer successor request is not exact")
    candidate = build_retained_writer_candidate(
        writer_kind=writer_kind,
        campaign_bucket=_retained_roles(ports).get("campaign_bucket"),
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
        authority_domain="FINALIZATION",
        record=record,
    )
    if (
        domain_result.writer_kind != writer_kind
        or domain_result.candidate_identity_sha256
        != candidate.candidate_identity_sha256
        or domain_result.response_authenticated is not True
    ):
        _fail("writer successor effect is foreign")
    activation_id = getattr(invocation, "activation_id")
    index = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_activation_index",
    )
    control = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_control",
    )
    current = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_finalization_control",
    )
    target = {
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED": (
            "OWNED",
            "SUPPORT_FINALIZED",
            "support_plane_finalized_identity_sha256",
        ),
        "RETAINED_INVOKE_H1G_DRAINED_WRITER": (
            "SNAPSHOT_DISPOSITION_RECORDED",
            "DRAINED_PUBLISHED",
            "h1g_drained_identity_sha256",
        ),
    }.get(operation_kind)
    if target is None:
        _fail("writer successor operation is not closed")
    action = request.get("action")
    audit = request.get("audit")
    if (
        current != source_control
        or current.get("state") != target[0]
        or type(action) is not dict
        or type(audit) is not dict
        or current.get("revision") != audit.get("closing_revision")
        or action.get("authorized_revision") != current.get("revision") + 1
        or audit.get("authorized_revision") != action.get(
            "authorized_revision"
        )
        or current.get("owner_invocation_nonce_sha256")
        != record.get("writer_invocation_nonce_sha256")
    ):
        _fail("writer successor finalization source changed after effect")
    after = dict(current)
    after.update(
        state=target[1],
        revision=current["revision"] + 1,
        updated_at=request["audit"]["observed_at"],
    )
    after[target[2]] = canonical_record_identity(
        record["record_type"], record
    )
    if target[1] == "DRAINED_PUBLISHED":
        after.update(
            owner_attempt=None,
            owner_execution_arn=None,
            owner_state_machine_version_arn=None,
            owner_dispatch_identity_sha256=None,
            owner_invocation_nonce_sha256=None,
            owner_hard_expires_at=None,
        )
    plan = build_finalization_progress_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index")),
            index,
        ),
        control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            control,
        ),
        finalization_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_finalization_control",
                    activation_id=activation_id,
                ),
            ),
            current,
            after,
        ),
    )
    raw_nonce = _retained_decrypt_owner_nonce(
        invocation=invocation, control=current, ports=ports
    )
    resolution = DynamoLedgerAdapter(
        client=ports.client("dynamodb"),
        table_name=_retained_table_name(ports),
    ).commit_finalization_progress(
        plan=plan,
        domain="TASK12_FINALIZATION",
        operation_identity_sha256=canonical_sha256(
            {
                "operation_kind": operation_kind,
                "writer_result_identity_sha256": (
                    domain_result.canonical_identity_sha256
                ),
                "finalization_before": canonical_record_identity(
                    "glm52_production_finalization_control", current
                ),
            }
        ),
        raw_owner_nonce=raw_nonce,
    )
    if resolution.outcome not in {
        WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
        WriteOutcome.EXACT_DURABLE_ADOPTION,
    }:
        _fail("writer effect successor was not durably committed")


def _retained_persist_snapshot_arm_successor(
    *,
    invocation: object,
    source_control: Mapping[str, object],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> None:
    from .dynamodb import (
        DynamoLedgerAdapter,
        ExactCheck,
        ExactUpdate,
        LedgerKey,
        TransactionResolution,
        WriteOutcome,
    )
    from .task12_retained_state import build_finalization_progress_plan

    if type(domain_result) is not dict or set(domain_result) != {
        "transition",
        "schedule",
    }:
        _fail("snapshot arm successor requires the exact domain effect")
    transition = domain_result["transition"]
    schedule_result = domain_result["schedule"]
    plan = request.get("plan")
    cleanup_update = (
        plan.get("cleanup_control") if type(plan) is dict else None
    )
    cleanup_after = (
        cleanup_update.get("after")
        if type(cleanup_update) is dict
        else None
    )
    if (
        type(transition) is not TransactionResolution
        or transition.outcome
        not in {
            WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
            WriteOutcome.EXACT_DURABLE_ADOPTION,
        }
        or type(cleanup_after) is not dict
        or cleanup_after.get("state") != "ARMED"
        or not any(record == cleanup_after for record in transition.records)
        or type(schedule_result) is not dict
        or schedule_result.get("schedule_arn")
        != request.get("schedule", {}).get("schedule_arn")
        or type(schedule_result.get("identity_sha256")) is not str
    ):
        _fail("snapshot arm effect is foreign or incomplete")
    activation_id = getattr(invocation, "activation_id")
    index = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_activation_index",
    )
    control = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_control",
    )
    finalization = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_finalization_control",
    )
    if finalization != source_control:
        _fail("snapshot arm finalization source changed after effect")
    after = dict(finalization)
    after.update(
        state="SNAPSHOT_DISPOSITION_RECORDED",
        snapshot_disposition_identity_sha256=canonical_record_identity(
            "glm52_production_snapshot_cleanup_control", cleanup_after
        ),
        revision=finalization["revision"] + 1,
        updated_at=cleanup_after["updated_at"],
    )
    progress = build_finalization_progress_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index")),
            index,
        ),
        control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            control,
        ),
        finalization_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_finalization_control",
                    activation_id=activation_id,
                ),
            ),
            finalization,
            after,
        ),
    )
    raw_nonce = _retained_decrypt_owner_nonce(
        invocation=invocation, control=finalization, ports=ports
    )
    resolution = DynamoLedgerAdapter(
        client=ports.client("dynamodb"),
        table_name=_retained_table_name(ports),
    ).commit_finalization_progress(
        plan=progress,
        domain="TASK12_FINALIZATION",
        operation_identity_sha256=canonical_sha256(
            {
                "operation_kind": getattr(invocation, "operation_kind"),
                "snapshot_cleanup_identity_sha256": (
                    after["snapshot_disposition_identity_sha256"]
                ),
                "schedule_effect_identity_sha256": schedule_result[
                    "identity_sha256"
                ],
            }
        ),
        raw_owner_nonce=raw_nonce,
    )
    if resolution.outcome not in {
        WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
        WriteOutcome.EXACT_DURABLE_ADOPTION,
    }:
        _fail("snapshot arm successor was not durably committed")


def _retained_snapshot_discovery_request(
    *, invocation: object, ports: object
) -> dict[str, object]:
    from .task12_snapshot_cleanup import SnapshotCapture, SnapshotObservation

    roles = _retained_roles(ports)
    support_stack_id = roles.get("support_stack_id")
    if (
        type(support_stack_id) is not str
        or not support_stack_id.startswith(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-support/"
        )
    ):
        _fail("exact deleted support stack ID authority is absent")
    cloudformation = ports.client("cloudformation")
    rows: list[Mapping[str, object]] = []
    token: str | None = None
    seen: set[str] = set()
    while True:
        request: dict[str, object] = {
            "StackName": support_stack_id
        }
        if token is not None:
            request["NextToken"] = token
        response = cloudformation.list_stack_resources(**request)
        _metadata(response, "support stack resource inventory")
        summaries = response.get("StackResourceSummaries")
        if type(summaries) is not list or any(
            type(item) is not dict for item in summaries
        ):
            _fail("support stack resource inventory is not exact")
        rows.extend(summaries)
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen
        ):
            _fail("support stack resource pagination drifted")
        seen.add(next_token)
        token = next_token
    volumes = [
        row
        for row in rows
        if row.get("LogicalResourceId") == "CombinedHostDataVolume"
        and row.get("ResourceType") == "AWS::EC2::Volume"
    ]
    if (
        len(volumes) != 1
        or type(volumes[0].get("PhysicalResourceId")) is not str
    ):
        _fail("support stack has no exact forensic source volume")
    volume_id = volumes[0]["PhysicalResourceId"]
    kms_key_arn = roles.get("kms_key_id")
    if type(kms_key_arn) is not str or not kms_key_arn.startswith(
        "arn:aws:kms:"
    ):
        _fail("snapshot discovery KMS authority is absent")
    expected_tags = [
        {"Key": "ActivationId", "Value": getattr(invocation, "activation_id")},
        {"Key": "Authority", "Value": "H1g"},
        {"Key": "Campaign", "Value": "GLM-5.2"},
        {"Key": "ManagedBy", "Value": "CloudFormation"},
        {"Key": "Project", "Value": "KEEP"},
        {"Key": "Purpose", "Value": "forensic-data-volume"},
        {"Key": "RunId", "Value": RUN_ID},
    ]
    expected_tags_sha256 = canonical_sha256(expected_tags)
    response = ports.client("ec2").describe_snapshots(
        OwnerIds=["self"],
        Filters=[{"Name": "volume-id", "Values": [volume_id]}],
    )
    metadata = _metadata(response, "recovery snapshot discovery")
    snapshots = response.get("Snapshots")
    if type(snapshots) is not list or any(
        type(item) is not dict for item in snapshots
    ):
        _fail("recovery snapshot discovery response is not exact")
    observations = []
    for item in snapshots:
        started = item.get("StartTime")
        raw_tags = item.get("Tags", [])
        if (
            type(started) is not datetime
            or started.tzinfo is None
            or started.utcoffset() is None
            or started.microsecond != 0
            or type(raw_tags) is not list
            or any(
                type(tag) is not dict
                or set(tag) != {"Key", "Value"}
                or type(tag["Key"]) is not str
                or type(tag["Value"]) is not str
                for tag in raw_tags
            )
        ):
            _fail("recovery snapshot observation is malformed")
        tags = sorted(
            (dict(tag) for tag in raw_tags),
            key=lambda tag: (tag["Key"], tag["Value"]),
        )
        observed_at = started.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        body = {
            "snapshot_id": item.get("SnapshotId"),
            "source_volume_id": item.get("VolumeId"),
            "kms_key_arn": item.get("KmsKeyId"),
            "snapshot_tags_sha256": canonical_sha256(tags),
            "encrypted": item.get("Encrypted"),
            "state": item.get("State"),
            "observed_at": observed_at,
            "request_id": metadata["RequestId"],
        }
        observations.append(
            SnapshotObservation(
                snapshot_id=item.get("SnapshotId"),
                source_volume_id=item.get("VolumeId"),
                kms_key_arn=item.get("KmsKeyId"),
                snapshot_tags_sha256=canonical_sha256(tags),
                encrypted=item.get("Encrypted"),
                state=item.get("State"),
                observed_at=observed_at,
                describe_request_id=metadata["RequestId"],
                describe_response_sha256=canonical_sha256(body),
            )
        )
    SnapshotCapture.discover(
        observations=tuple(observations),
        expected_source_volume_id=volume_id,
        expected_kms_key_arn=kms_key_arn,
        expected_snapshot_tags_sha256=expected_tags_sha256,
    )
    return {
        "observations": [asdict(item) for item in observations],
        "expected_source_volume_id": volume_id,
        "expected_kms_key_arn": kms_key_arn,
        "expected_snapshot_tags_sha256": expected_tags_sha256,
    }


def _retained_snapshot_arm_request(
    *,
    invocation: object,
    sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    from .dynamodb import ExactCheck, ExactPut, ExactUpdate, LedgerKey
    from .task12_retained_state import (
        build_snapshot_cleanup_transition_plan,
    )
    from .task12_snapshot_cleanup import (
        SnapshotCapture,
        SnapshotSchedule,
    )

    capture = _retained_read_snapshot_effect(
        invocation=invocation,
        support=sources["support_finalized"],
        ports=ports,
    )
    if type(capture) is not SnapshotCapture:
        _fail("snapshot arm did not consume the exact discovery effect")
    roles = _retained_roles(ports)
    group = roles.get("snapshot_cleanup_schedule_group_name")
    name = roles.get("snapshot_cleanup_schedule_name")
    if (
        type(group) is not str
        or not group
        or type(name) is not str
        or not name
    ):
        _fail("snapshot cleanup schedule authority is absent")
    schedule = SnapshotSchedule.create(
        capture=capture,
        schedule_arn=(
            "arn:aws:scheduler:"
            + str(sources["finalization_control"]["region"])
            + ":"
            + str(sources["finalization_control"]["account_id"])
            + ":schedule/"
            + group
            + "/"
            + name
        ),
    )
    activation_id = getattr(invocation, "activation_id")
    index = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_activation_index",
    )
    recovery = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_recovery_control",
    )
    cleanup = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_snapshot_cleanup_control",
    )
    finalization = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_finalization_control",
    )
    support = sources["support_finalized"]
    if (
        finalization != sources["finalization_control"]
        or finalization.get("state") != "SUPPORT_FINALIZED"
        or finalization.get("support_plane_finalized_identity_sha256")
        != canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        )
        or recovery.get("state") != "RECOVERY_COMPLETE"
        or cleanup.get("state") != "DORMANT"
    ):
        _fail("snapshot arm current authority is not ready")
    owner_nonce_capsule = _retained_prior_nonce_capsule(
        invocation=invocation, control=finalization
    )
    observed_at = _retained_now()
    cleanup_after = dict(cleanup)
    cleanup_after.update(
        state="ARMED",
        snapshot_id=capture.snapshot_id,
        snapshot_identity_sha256=capture.snapshot_identity_sha256,
        source_volume_id=capture.source_volume_id,
        snapshot_tags_sha256=capture.snapshot_tags_sha256,
        delete_not_before=schedule.delete_not_before,
        schedule_arn=schedule.schedule_arn,
        schedule_input_sha256=schedule.schedule_input_sha256,
        cleanup_barrier_nonce_sha256=canonical_sha256(
            {
                "finalization_identity_sha256": canonical_record_identity(
                    "glm52_production_finalization_control", finalization
                ),
                "snapshot_identity_sha256": (
                    capture.snapshot_identity_sha256
                ),
            }
        ),
        cleanup_lineage_identity_sha256=recovery[
            "cleanup_control_root_identity_sha256"
        ],
        revision=cleanup["revision"] + 1,
        updated_at=observed_at,
    )
    transition_body = {
        "schema_version": 1,
        "record_type": "glm52_production_snapshot_cleanup_transition",
        "account_id": cleanup["account_id"],
        "region": cleanup["region"],
        "run_id": cleanup["run_id"],
        "campaign_identity_sha256": cleanup[
            "campaign_identity_sha256"
        ],
        "activation_id": cleanup["activation_id"],
        "activation_ordinal": cleanup["activation_ordinal"],
        "cleanup_control_root_identity_sha256": recovery[
            "cleanup_control_root_identity_sha256"
        ],
        "from_state": cleanup["state"],
        "to_state": cleanup_after["state"],
        "from_revision": cleanup["revision"],
        "to_revision": cleanup_after["revision"],
        "prior_transition_sha256": recovery[
            "cleanup_transition_chain_head_sha256"
        ],
        "owner_attempt": None,
        "owner_execution_arn": None,
        "owner_state_machine_version_arn": None,
        "owner_dispatch_identity_sha256": None,
        "owner_invocation_nonce_sha256": None,
        "authority_audit_identity_sha256": None,
        "action_identity_sha256": None,
        "transitioned_at": observed_at,
    }
    transition = validate_record(
        "glm52_production_snapshot_cleanup_transition",
        {
            **transition_body,
            "canonical_body_sha256": canonical_sha256(transition_body),
        },
    )
    recovery_after = dict(recovery)
    recovery_after.update(
        cleanup_transition_chain_head_sha256=transition[
            "canonical_body_sha256"
        ],
        cleanup_transition_chain_length=(
            recovery["cleanup_transition_chain_length"] + 1
        ),
        revision=recovery["revision"] + 1,
        updated_at=observed_at,
    )
    plan = build_snapshot_cleanup_transition_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index")),
            index,
        ),
        cleanup_chain=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=activation_id,
                ),
            ),
            recovery,
            recovery_after,
        ),
        authority=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_finalization_control",
                    activation_id=activation_id,
                ),
            ),
            finalization,
        ),
        cleanup_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_control",
                    activation_id=activation_id,
                ),
            ),
            cleanup,
            cleanup_after,
        ),
        cleanup_transition=ExactPut(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_transition",
                    activation_id=activation_id,
                    revision=cleanup_after["revision"],
                ),
            ),
            transition,
        ),
    )
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "capture_evidence_sha256": capture.capture_evidence_sha256,
            "finalization_before": canonical_record_identity(
                "glm52_production_finalization_control", finalization
            ),
            "cleanup_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control", cleanup
            ),
        }
    )
    return {
        "plan": asdict(plan),
        "capture": asdict(capture),
        "schedule": asdict(schedule),
        "domain": "TEARDOWN",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": owner_nonce_capsule,
    }


def _retained_snapshot_effect_key(invocation: object) -> str:
    return (
        "ACTIVATION#"
        + str(getattr(invocation, "activation_id"))
        + "#TASK12_EFFECT#SNAPSHOT_CAPTURE#"
        + str(getattr(invocation, "generation_text"))
    )


def _retained_persist_snapshot_effect(
    *,
    invocation: object,
    support: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> None:
    from .dynamodb import decode_item, encode_item
    from .task12_snapshot_cleanup import SnapshotCapture

    if type(domain_result) is not SnapshotCapture:
        _fail("snapshot successor requires the exact domain effect")
    support_identity = canonical_record_identity(
        "glm52_production_support_plane_finalized", support
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_snapshot_capture_effect_v1",
        "run_id": RUN_ID,
        "activation_id": getattr(invocation, "activation_id"),
        "activation_ordinal": getattr(invocation, "activation_ordinal"),
        "generation": getattr(invocation, "generation"),
        "generation_text": getattr(invocation, "generation_text"),
        "support_plane_finalized_identity_sha256": support_identity,
        "capture": asdict(domain_result),
    }
    effect = {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }
    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": _retained_snapshot_effect_key(invocation),
    }
    client = ports.client("dynamodb")
    try:
        response = client.put_item(
            TableName=_retained_table_name(ports),
            Item=encode_item({**key, **effect}),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
    except Exception:
        response = None
    if type(response) is dict:
        _metadata(response, "snapshot effect successor write")
        return
    readback = client.get_item(
        TableName=_retained_table_name(ports),
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(readback, "snapshot effect successor readback")
    item = readback.get("Item")
    if type(item) is not dict or decode_item(item) != {**key, **effect}:
        _fail("snapshot effect successor adopted foreign bytes")


def _retained_read_snapshot_effect(
    *,
    invocation: object,
    support: Mapping[str, object],
    ports: object,
) -> object:
    from .dynamodb import decode_item, encode_item
    from .task12_snapshot_cleanup import SnapshotCapture

    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": _retained_snapshot_effect_key(invocation),
    }
    response = ports.client("dynamodb").get_item(
        TableName=_retained_table_name(ports),
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "snapshot effect successor read")
    item = response.get("Item")
    if type(item) is not dict:
        _fail("exact snapshot discovery effect is absent")
    effect = decode_item(item)
    if effect.pop("PK", None) != key["PK"] or effect.pop(
        "SK", None
    ) != key["SK"]:
        _fail("snapshot discovery effect coordinate drifted")
    body = dict(effect)
    identity = body.pop("canonical_body_sha256", None)
    if (
        identity != canonical_sha256(body)
        or body.get("record_type")
        != "glm52_task12_snapshot_capture_effect_v1"
        or body.get("run_id") != RUN_ID
        or body.get("activation_id")
        != getattr(invocation, "activation_id")
        or body.get("activation_ordinal")
        != getattr(invocation, "activation_ordinal")
        or body.get("generation") != getattr(invocation, "generation")
        or body.get("generation_text")
        != getattr(invocation, "generation_text")
        or body.get("support_plane_finalized_identity_sha256")
        != canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        )
    ):
        _fail("snapshot discovery effect identity drifted")
    raw_capture = body.get("capture")
    if type(raw_capture) is not dict:
        _fail("snapshot discovery effect capture is absent")
    try:
        capture = SnapshotCapture(**raw_capture)
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            "snapshot discovery effect capture is malformed"
        ) from exc
    immutable_body = {
        "schema_version": 1,
        "snapshot_id": capture.snapshot_id,
        "source_volume_id": capture.source_volume_id,
        "kms_key_arn": capture.kms_key_arn,
        "snapshot_tags_sha256": capture.snapshot_tags_sha256,
        "encrypted": True,
    }
    if capture.snapshot_identity_sha256 != canonical_sha256(immutable_body):
        _fail("snapshot discovery effect immutable identity drifted")
    return capture


def _retained_orphan_effect_key(invocation: object) -> str:
    return (
        "ACTIVATION#"
        + str(getattr(invocation, "activation_id"))
        + "#TASK12_EFFECT#ORPHAN_AUDIT#"
        + str(getattr(invocation, "generation_text"))
    )


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if type(value) is dict:
        return {key: _jsonable(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_jsonable(item) for item in value]
    if type(value) is list:
        return [_jsonable(item) for item in value]
    return value


def _retained_orphan_request(
    *,
    invocation: object,
    ports: object,
) -> Mapping[str, object]:
    """Exact-read activation authority, scan, clean, and build typed audit input."""

    from .task12_orphan_authority import (
        cleanup_activation_orphans,
        read_activation_authority,
    )
    from .task12_resource_scanner import scan_task12_resources

    activation_id = str(getattr(invocation, "activation_id"))
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    authority = read_activation_authority(
        s3=ports.client("s3"),
        dynamodb=ports.client("dynamodb"),
        ledger_table_name=_retained_table_name(ports),
        activation_id=activation_id,
    )
    inventory = scan_task12_resources(
        clients=ports.client,
        authority=authority,
        observed_at=observed_at,
    )
    finalized = cleanup_activation_orphans(
        authority=authority,
        kms=ports.client("kms"),
        resource_inventory=inventory,
        observed_at=observed_at,
    )
    request = _jsonable(finalized["audit_orphans_request"])
    if type(request) is not dict:
        _fail("orphan audit request materialization is malformed")
    return request


def _retained_persist_orphan_effect(
    *,
    invocation: object,
    domain_result: object,
    ports: object,
) -> None:
    from .dynamodb import decode_item, encode_item
    from .task12_orphan_audit import OrphanAuditProof

    if type(domain_result) is not OrphanAuditProof:
        _fail("orphan successor requires the exact domain effect")
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_orphan_audit_effect_v1",
        "run_id": RUN_ID,
        "activation_id": getattr(invocation, "activation_id"),
        "activation_ordinal": getattr(
            invocation, "activation_ordinal"
        ),
        "generation": getattr(invocation, "generation"),
        "generation_text": getattr(invocation, "generation_text"),
        "proof": asdict(domain_result),
    }
    effect = {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }
    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": _retained_orphan_effect_key(invocation),
    }
    client = ports.client("dynamodb")
    try:
        response = client.put_item(
            TableName=_retained_table_name(ports),
            Item=encode_item({**key, **effect}),
            ConditionExpression=(
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            ),
            ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            ReturnConsumedCapacity="NONE",
        )
    except Exception:
        response = None
    if type(response) is dict:
        _metadata(response, "orphan effect successor write")
        return
    readback = client.get_item(
        TableName=_retained_table_name(ports),
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(readback, "orphan effect successor readback")
    item = readback.get("Item")
    if type(item) is not dict or decode_item(item) != {**key, **effect}:
        _fail("orphan effect successor adopted foreign bytes")


def _retained_read_orphan_effect(
    *, invocation: object, ports: object
) -> object:
    from .dynamodb import decode_item, encode_item
    from .task12_orphan_audit import (
        KmsGrantIdentity,
        OrphanAuditProof,
        RetainedResource,
    )

    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": _retained_orphan_effect_key(invocation),
    }
    response = ports.client("dynamodb").get_item(
        TableName=_retained_table_name(ports),
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "orphan effect successor read")
    item = response.get("Item")
    if type(item) is not dict:
        _fail("exact orphan audit effect is absent")
    effect = decode_item(item)
    if effect.pop("PK", None) != key["PK"] or effect.pop(
        "SK", None
    ) != key["SK"]:
        _fail("orphan audit effect coordinate drifted")
    body = dict(effect)
    identity = body.pop("canonical_body_sha256", None)
    if (
        identity != canonical_sha256(body)
        or body.get("record_type")
        != "glm52_task12_orphan_audit_effect_v1"
        or body.get("activation_id")
        != getattr(invocation, "activation_id")
        or body.get("generation") != getattr(invocation, "generation")
    ):
        _fail("orphan audit effect identity drifted")
    proof = body.get("proof")
    if type(proof) is not dict:
        _fail("orphan audit effect proof is absent")
    return OrphanAuditProof(
        retained_resources=tuple(
            RetainedResource(**item)
            for item in proof["retained_resources"]
        ),
        retained_cost_classes=tuple(proof["retained_cost_classes"]),
        orphan_resource_ids=tuple(proof["orphan_resource_ids"]),
        retained_grant_baseline_identity_sha256=proof[
            "retained_grant_baseline_identity_sha256"
        ],
        final_grants=tuple(
            KmsGrantIdentity(**item) for item in proof["final_grants"]
        ),
        final_list_grants_identity_sha256=proof[
            "final_list_grants_identity_sha256"
        ],
        settling_deadline=proof["settling_deadline"],
        observed_at=proof["observed_at"],
        canonical_identity_sha256=proof[
            "canonical_identity_sha256"
        ],
    )


def _retained_read_terminal(
    *, invocation: object, ports: object
) -> tuple[dict[str, object], dict[str, object]]:
    from .dynamodb import decode_item, encode_item

    activation_id = getattr(invocation, "activation_id")
    generation = getattr(invocation, "generation")
    key = {
        "PK": ledger_pk(RUN_ID),
        "SK": ledger_sk(
            "glm52_task12_versioned_writer_control_v1",
            activation_id=activation_id,
            generation=generation,
            writer_kind="TerminalV2",
        ),
    }
    ddb = ports.client("dynamodb").get_item(
        TableName=_retained_table_name(ports),
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(ddb, "TerminalV2 exact version control read")
    item = ddb.get("Item")
    if type(item) is not dict:
        _fail("TerminalV2 exact version control is absent")
    physical = decode_item(item)
    if physical.pop("PK", None) != key["PK"] or physical.pop(
        "SK", None
    ) != key["SK"]:
        _fail("TerminalV2 version control coordinate drifted")
    control = validate_record(
        "glm52_task12_versioned_writer_control_v1", physical
    )
    bucket = control["campaign_bucket"]
    if (
        control.get("activation_id") != activation_id
        or control.get("generation") != generation
        or control.get("generation_text") != f"{generation:08d}"
        or control.get("writer_kind") != "TerminalV2"
        or bucket != _H1G_MODEL_BUCKET
        or bucket != _retained_roles(ports).get("campaign_bucket")
        or control.get("object_key") != _H1G_TERMINAL_KEY
    ):
        _fail("TerminalV2 exact version control authority drifted")
    response = ports.client("s3").get_object(
        Bucket=bucket,
        Key=control["object_key"],
        VersionId=control["object_version_id"],
        ExpectedBucketOwner=control["account_id"],
        ChecksumMode="ENABLED",
    )
    _metadata(response, "TerminalV2 exact object read")
    body = response.get("Body")
    raw = body.read() if body is not None else None
    expected_checksum = (
        base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        if type(raw) is bytes
        else None
    )
    if (
        type(raw) is not bytes
        or response.get("VersionId") != control["object_version_id"]
        or response.get("ChecksumSHA256") != expected_checksum
        or hashlib.sha256(raw).hexdigest() != control["file_sha256"]
    ):
        _fail("TerminalV2 object transport drifted")
    import json

    try:
        value = json.loads(raw.rstrip(b"\n"))
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            "TerminalV2 exact object is not canonical JSON"
        ) from exc
    terminal = validate_record("glm52_production_terminal_v2", value)
    if terminal["canonical_body_sha256"] != control["body_sha256"]:
        _fail("TerminalV2 exact object body drifted")
    return terminal, control


def _retained_read_spend_ledger_head(
    *,
    ports: object,
    expected_head_record_sha256: str,
) -> dict[str, object]:
    """Read one immutable current spend-ledger version and bind its exact bytes."""

    if (
        type(expected_head_record_sha256) is not str
        or len(expected_head_record_sha256) != 64
        or any(
            char not in "0123456789abcdef"
            for char in expected_head_record_sha256
        )
    ):
        _fail("TerminalV2 spend-ledger head authority is absent")
    client = ports.client("s3")
    versions: list[Mapping[str, object]] = []
    latest_delete_markers: list[Mapping[str, object]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    while True:
        request: dict[str, object] = {
            "Bucket": _H1G_MODEL_BUCKET,
            "Prefix": _H1G_SPEND_LEDGER_KEY,
            "ExpectedBucketOwner": "246813579024",
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
        if version_marker is not None:
            request["VersionIdMarker"] = version_marker
        response = client.list_object_versions(**request)
        _metadata(response, "GPU spend ledger exact-version listing")
        listed_versions = response.get("Versions", [])
        delete_markers = response.get("DeleteMarkers", [])
        if type(listed_versions) is not list or type(delete_markers) is not list:
            _fail("GPU spend ledger version inventory is malformed")
        versions.extend(
            item
            for item in listed_versions
            if type(item) is dict
            and item.get("Key") == _H1G_SPEND_LEDGER_KEY
            and item.get("IsLatest") is True
        )
        latest_delete_markers.extend(
            item
            for item in delete_markers
            if type(item) is dict
            and item.get("Key") == _H1G_SPEND_LEDGER_KEY
            and item.get("IsLatest") is True
        )
        if response.get("IsTruncated") is False:
            break
        if response.get("IsTruncated") is not True:
            _fail("GPU spend ledger version inventory truncation drifted")
        key_marker = response.get("NextKeyMarker")
        version_marker = response.get("NextVersionIdMarker")
        if (
            type(key_marker) is not str
            or not key_marker
            or type(version_marker) is not str
            or not version_marker
        ):
            _fail("GPU spend ledger pagination cursor is absent")
    if len(versions) != 1 or latest_delete_markers:
        _fail("GPU spend ledger current immutable version is not singular")
    version_id = versions[0].get("VersionId")
    if (
        type(version_id) is not str
        or not version_id
        or version_id.lower() in {"latest", "$latest", "null"}
    ):
        _fail("GPU spend ledger immutable VersionId is absent")
    response = client.get_object(
        Bucket=_H1G_MODEL_BUCKET,
        Key=_H1G_SPEND_LEDGER_KEY,
        VersionId=version_id,
        ExpectedBucketOwner="246813579024",
        ChecksumMode="ENABLED",
    )
    _metadata(response, "GPU spend ledger exact-version read")
    body = response.get("Body")
    raw = body.read() if callable(getattr(body, "read", None)) else None
    expected_checksum = (
        base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        if type(raw) is bytes
        else None
    )
    if (
        type(raw) is not bytes
        or not raw
        or not raw.endswith(b"\n")
        or response.get("VersionId") != version_id
        or response.get("ChecksumSHA256") != expected_checksum
    ):
        _fail("GPU spend ledger exact-version transport drifted")
    records: list[dict[str, object]] = []
    prior: str | None = None
    active_instance_id: str | None = None
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Task12LiveSnapshotFinalizationError(
                "GPU spend ledger contains malformed JSON"
            ) from exc
        if (
            type(value) is not dict
            or line != canonical_json_bytes(value) + b"\n"
        ):
            _fail("GPU spend ledger is not canonical JSONL")
        record_body = dict(value)
        identity = record_body.pop("record_sha256", None)
        event = record_body.get("event")
        instance_id = record_body.get("instance_id")
        if (
            type(identity) is not str
            or identity != canonical_sha256(record_body)
            or record_body.get("record_type")
            != "glm52_gpu_spend_event_v1"
            or record_body.get("run_id") != RUN_ID
            or type(instance_id) is not str
            or not instance_id
            or event not in {"allocation_started", "allocation_ended"}
            or (
                prior is not None
                and record_body.get("prior_record_sha256") != prior
            )
        ):
            _fail("GPU spend ledger record identity drifted")
        if event == "allocation_started":
            if (
                active_instance_id is not None
                or type(record_body.get("job_id")) is not str
                or not record_body["job_id"]
            ):
                _fail("GPU spend ledger allocation start drifted")
            active_instance_id = instance_id
        elif (
            active_instance_id is None
            or active_instance_id != instance_id
            or "job_id" in record_body
        ):
            _fail("GPU spend ledger allocation end drifted")
        else:
            active_instance_id = None
        records.append(value)
        prior = identity
    if (
        not records
        or active_instance_id is not None
        or prior != expected_head_record_sha256
    ):
        _fail("GPU spend ledger terminal head drifted")
    identity_body = {
        "bucket": _H1G_MODEL_BUCKET,
        "key": _H1G_SPEND_LEDGER_KEY,
        "version_id": version_id,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": canonical_sha256(records),
        "head_record_sha256": prior,
    }
    return {
        **identity_body,
        "canonical_identity_sha256": canonical_sha256(identity_body),
    }


def _retained_support_stack_absent(
    ports: object,
) -> dict[str, object]:
    support_stack_id = _retained_roles(ports).get("support_stack_id")
    if (
        type(support_stack_id) is not str
        or not support_stack_id.startswith(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-support/"
        )
    ):
        _fail("exact deleted support stack ID authority is absent")

    try:
        response = ports.client("cloudformation").describe_stacks(
            StackName=support_stack_id
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        metadata = (
            response.get("ResponseMetadata")
            if type(response) is dict
            else None
        )
        code = (
            response.get("Error", {}).get("Code")
            if type(response) is dict
            else None
        )
        if code != "ValidationError":
            raise
        exact_metadata = _metadata(
            {"ResponseMetadata": metadata}, "support stack absence"
        )
        identity_body = {
            "stack_id": support_stack_id,
            "observation": "ABSENT_VALIDATION_ERROR",
            "request_id": exact_metadata["RequestId"],
        }
        return {
            **identity_body,
            "canonical_identity_sha256": canonical_sha256(identity_body),
        }
    exact_metadata = _metadata(response, "support stack absence")
    stacks = response.get("Stacks")
    if (
        type(stacks) is not list
        or len(stacks) != 1
        or type(stacks[0]) is not dict
        or stacks[0].get("StackId") != support_stack_id
        or type(stacks[0].get("StackStatus")) is not str
    ):
        _fail("support stack absence readback is not exact")
    if stacks[0]["StackStatus"] != "DELETE_COMPLETE":
        _fail("support stack is not deletion-terminal")
    identity_body = {
        "stack_id": support_stack_id,
        "observation": "DELETE_COMPLETE",
        "request_id": exact_metadata["RequestId"],
    }
    return {
        **identity_body,
        "canonical_identity_sha256": canonical_sha256(identity_body),
    }


def _retained_h1g_request(
    *,
    invocation: object,
    sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> dict[str, object]:
    from .task12_orphan_audit import (
        MarkerLastPrerequisites,
        build_h1g_drained_prerequisites,
    )
    from .support_price_card_authority import (
        build_approved_support_price_card,
    )

    activation_id = getattr(invocation, "activation_id")
    finalization = _retained_read_record(
        ports=ports,
        activation_id=activation_id,
        record_type="glm52_production_finalization_control",
    )
    cleanup = sources["snapshot_cleanup_control"]
    support = sources["support_finalized"]
    if (
        finalization.get("state") != "SNAPSHOT_DISPOSITION_RECORDED"
        or cleanup.get("state") != "ARMED"
        or finalization.get("snapshot_disposition_identity_sha256")
        != canonical_record_identity(
            "glm52_production_snapshot_cleanup_control", cleanup
        )
        or finalization.get("support_plane_finalized_identity_sha256")
        != canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        )
    ):
        _fail("H1G marker current finalization lineage is not ready")
    terminal, terminal_control = _retained_read_terminal(
        invocation=invocation, ports=ports
    )
    terminal_identity = canonical_record_identity(
        "glm52_production_terminal_v2", terminal
    )
    allocations = terminal["allocations"]
    liabilities = terminal["worker_launch_liabilities"]
    all_allocations_closed = all(
        type(item) is dict
        and type(item.get("allocation_close_identity_sha256")) is str
        and type(item.get("charged_interval_ended_at")) is str
        for item in allocations
    )
    all_workers_terminal = all(
        type(item) is dict
        and type(item.get("instance_terminal_identity_sha256")) is str
        and type(item.get("terminal_state")) is str
        and bool(item["terminal_state"])
        for item in allocations
    )
    if not all_allocations_closed or not all_workers_terminal:
        _fail("H1G marker TerminalV2 worker closure is incomplete")
    if liabilities and any(
        type(item) is not dict
        or item.get("state") not in {"WATCHING", "LATE_INSTANCE_DRAINING"}
        for item in liabilities
    ):
        _fail("H1G marker liability transfer is incomplete")
    liability_state = (
        "SETTLED" if not liabilities else "RETAINED_TERMINATION_ONLY"
    )
    closure_evidence = (
        terminal.get("drain_identity")
        or terminal.get("post_terminal_quiescence_evidence")
        or terminal.get("terminal_observation_window")
    )
    drain_identity = (
        canonical_sha256(closure_evidence)
        if isinstance(closure_evidence, Mapping)
        else closure_evidence
    )
    spend_head = terminal.get("spend_ledger_head_identity")
    if (
        type(spend_head) is not dict
        or set(spend_head) != {"identity"}
        or type(spend_head.get("identity")) is not str
    ):
        _fail("H1G marker TerminalV2 spend authority is incomplete")
    spend_ledger_identity = _retained_read_spend_ledger_head(
        ports=ports,
        expected_head_record_sha256=spend_head["identity"],
    )
    spend_identity = spend_ledger_identity["head_record_sha256"]
    if (
        terminal_identity != finalization.get("terminal_v2_identity_sha256")
        or type(drain_identity) is not str
        or type(spend_identity) is not str
    ):
        _fail("H1G marker TerminalV2 closure is incomplete")
    orphan_audit = _retained_read_orphan_effect(
        invocation=invocation, ports=ports
    )
    support_stack_deletion = _retained_support_stack_absent(ports)
    finalization_identity = canonical_record_identity(
        "glm52_production_finalization_control", finalization
    )
    cleanup_identity = canonical_record_identity(
        "glm52_production_snapshot_cleanup_control", cleanup
    )
    proof = build_h1g_drained_prerequisites(
        MarkerLastPrerequisites(
            terminal_v2_identity_sha256=terminal_identity,
            finalization_identity_sha256=finalization_identity,
            snapshot_cleanup_control_identity_sha256=cleanup_identity,
            controller_quiesced_identity_sha256=drain_identity,
            spend_ledger_head_identity_sha256=spend_identity,
            orphan_audit=orphan_audit,
            support_stack_absent=True,
            snapshot_cleanup_armed=True,
            all_workers_terminal=all_workers_terminal,
            all_allocations_closed=all_allocations_closed,
            liability_state=liability_state,
            liability_identity_sha256=canonical_sha256(liabilities),
        )
    )
    if proof.marker_write_order != "LAST_CONDITIONAL_CREATE":
        _fail("H1G marker is not last")
    terminal_identity_body = {
        "generation": terminal_control["generation"],
        "generation_text": terminal_control["generation_text"],
        "bucket": terminal_control["campaign_bucket"],
        "key": terminal_control["object_key"],
        "version_id": terminal_control["object_version_id"],
        "file_sha256": terminal_control["file_sha256"],
        "body_sha256": terminal_control["body_sha256"],
    }
    support_identity_body = {
        "finalization_control_identity_sha256": finalization_identity,
        "support_plane_finalized_identity_sha256": canonical_record_identity(
            "glm52_production_support_plane_finalized", support
        ),
    }
    final_grants = [asdict(item) for item in orphan_audit.final_grants]
    kms_equality_body = {
        "baseline_identity_sha256": (
            orphan_audit.retained_grant_baseline_identity_sha256
        ),
        "final_grants_identity_sha256": canonical_sha256(final_grants),
        "final_list_grants_identity_sha256": (
            orphan_audit.final_list_grants_identity_sha256
        ),
    }
    schedule_identity_body = {
        "schedule_arn": cleanup["schedule_arn"],
        "schedule_input_sha256": cleanup["schedule_input_sha256"],
    }
    price_card = dict(build_approved_support_price_card())
    inventory_identity_body = {
        "retained_resources": [
            asdict(item) for item in orphan_audit.retained_resources
        ],
        "retained_cost_classes": list(
            orphan_audit.retained_cost_classes
        ),
        "price_card_identity_sha256": price_card[
            "price_card_identity_sha256"
        ],
        "retained_terms": price_card["retained_terms"],
        "controller_quiescence_identity_sha256": drain_identity,
        "marker_last_prerequisite_identity_sha256": (
            proof.canonical_identity_sha256
        ),
    }
    evidence = {
        "terminal_v2_identity": {
            **terminal_identity_body,
            "canonical_identity_sha256": canonical_sha256(
                terminal_identity_body
            ),
        },
        "support_plane_finalized_identity": {
            **support_identity_body,
            "canonical_identity_sha256": canonical_sha256(
                support_identity_body
            ),
        },
        "support_stack_deletion_identity": support_stack_deletion,
        "orphan_audit_identity": orphan_audit.canonical_identity_sha256,
        "kms_grant_baseline_equality_identity": {
            **kms_equality_body,
            "canonical_identity_sha256": canonical_sha256(
                kms_equality_body
            ),
        },
        "snapshot_cleanup_control_identity": cleanup_identity,
        "snapshot_cleanup_schedule_identity": {
            **schedule_identity_body,
            "canonical_identity_sha256": canonical_sha256(
                schedule_identity_body
            ),
        },
        "worker_launch_liabilities": liabilities,
        "spend_ledger_head_identity": spend_ledger_identity,
        "retained_resource_inventory_identity": {
            **inventory_identity_body,
            "canonical_identity_sha256": canonical_sha256(
                inventory_identity_body
            ),
        },
    }
    return _retained_writer_request(
        writer_kind="H1GDrained",
        record_type="glm52_production_h1g_drained",
        control=finalization,
        invocation=invocation,
        ports=ports,
        evidence=evidence,
    )


def _send_request(
    *,
    control: Mapping[str, object],
    transition: Mapping[str, object],
) -> dict[str, object]:
    if (
        control.get("state") != "DELETE_POSSIBLY_SENT"
        or transition.get("to_state") != control.get("state")
        or transition.get("to_revision") != control.get("revision")
        or transition.get("from_revision") != control.get("revision") - 1
        or transition.get("activation_id") != control.get("activation_id")
        or transition.get("activation_ordinal")
        != control.get("activation_ordinal")
        or transition.get("owner_attempt") != control.get("owner_attempt")
        or transition.get("owner_execution_arn")
        != control.get("owner_execution_arn")
        or transition.get("owner_state_machine_version_arn")
        != control.get("owner_state_machine_version_arn")
        or transition.get("owner_dispatch_identity_sha256")
        != control.get("owner_dispatch_identity_sha256")
        or transition.get("owner_invocation_nonce_sha256")
        != control.get("owner_invocation_nonce_sha256")
        or transition.get("action_identity_sha256")
        != control.get("latest_delete_action_identity_sha256")
        or transition.get("authority_audit_identity_sha256")
        != control.get("latest_cleanup_authority_audit_identity_sha256")
    ):
        _fail("snapshot send causal source lineage drifted")
    snapshot_id = control.get("snapshot_id")
    if type(snapshot_id) is not str or not snapshot_id:
        _fail("snapshot cleanup control has no exact snapshot ID")
    return {"snapshot_id": snapshot_id}


def _snapshot_now(ports: object) -> datetime:
    boundary = getattr(ports, "now_utc", None)
    now = boundary() if callable(boundary) else datetime.now(timezone.utc)
    if (
        type(now) is not datetime
        or now.tzinfo is None
        or now.utcoffset() is None
        or now.microsecond != 0
    ):
        _fail("snapshot runtime clock is not normalized")
    return now.astimezone(timezone.utc)


def _snapshot_read_record(
    *,
    ports: object,
    record_type: str,
    sort_key: str,
) -> dict[str, object]:
    from .dynamodb import decode_item, encode_item

    roles = _retained_roles(ports)
    table_name = roles.get("ledger_table_name")
    client_boundary = getattr(ports, "client", None)
    if (
        type(table_name) is not str
        or not table_name
        or not callable(client_boundary)
    ):
        _fail(record_type + " live ledger boundary is absent")
    response = client_boundary("dynamodb").get_item(
        TableName=table_name,
        Key=encode_item(
            {"PK": ledger_pk(RUN_ID), "SK": sort_key}
        ),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, record_type + " exact read")
    item = response.get("Item") if type(response) is dict else None
    if type(item) is not dict:
        _fail(record_type + " exact record is absent")
    try:
        decoded = decode_item(item)
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            record_type + " exact record is malformed"
        ) from exc
    if (
        type(decoded) is not dict
        or decoded.get("PK") != ledger_pk(RUN_ID)
        or decoded.get("SK") != sort_key
    ):
        _fail(record_type + " exact coordinate drifted")
    body = dict(decoded)
    body.pop("PK")
    body.pop("SK")
    try:
        return validate_record(record_type, body)
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            record_type + " exact record drifted"
        ) from exc


def _snapshot_authorities(
    *,
    invocation: object,
    cleanup_control: Mapping[str, object],
    ports: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    activation_id = getattr(invocation, "activation_id")
    index = _snapshot_read_record(
        ports=ports,
        record_type="glm52_production_activation_index",
        sort_key=ledger_sk("glm52_production_activation_index"),
    )
    authority = _snapshot_read_record(
        ports=ports,
        record_type="glm52_production_control",
        sort_key=ledger_sk(
            "glm52_production_control",
            activation_id=activation_id,
        ),
    )
    chain = _snapshot_read_record(
        ports=ports,
        record_type="glm52_production_recovery_control",
        sort_key=ledger_sk(
            "glm52_production_recovery_control",
            activation_id=activation_id,
        ),
    )
    campaign = cleanup_control.get("campaign_identity_sha256")
    retained_ordinal = getattr(invocation, "activation_ordinal")
    if (
        type(retained_ordinal) is not int
        or index.get("current_activation_ordinal") < retained_ordinal
        or index.get("current_activation_ordinal") > retained_ordinal + 1
        or (
            index.get("current_activation_ordinal") == retained_ordinal
            and index.get("current_activation_id") != activation_id
        )
        or authority.get("activation_id") != activation_id
        or authority.get("activation_ordinal")
        != retained_ordinal
        or authority.get("phase") != "TEARDOWN_SEALED"
        or chain.get("activation_id") != activation_id
        or chain.get("activation_ordinal")
        != retained_ordinal
        or chain.get("state") != "RECOVERY_COMPLETE"
        or any(
            record.get("campaign_identity_sha256") != campaign
            for record in (index, authority, chain)
        )
    ):
        _fail("snapshot transition activation authority drifted")
    return index, authority, chain


def _snapshot_acquire_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    from .dynamodb import ExactCheck, ExactPut, ExactUpdate, LedgerKey
    from .task12_nonce_capsule import generate_owner_nonce_capsule
    from .task12_retained_state import (
        build_snapshot_cleanup_transition_plan,
    )

    if control.get("state") in {
        "OWNED",
        "DELETE_POSSIBLY_SENT",
        "DELETE_RECONCILING",
    }:
        return _snapshot_takeover_request(
            invocation=invocation,
            control=control,
            ports=ports,
        )
    if control.get("state") != "ARMED":
        _fail("snapshot owner acquisition state is invalid")
    now = _snapshot_now(ports)
    observed_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    delete_not_before = control.get("delete_not_before")
    if (
        type(delete_not_before) is not str
        or observed_at < delete_not_before
    ):
        _fail("snapshot owner acquisition is before the cleanup deadline")
    index, authority, chain = _snapshot_authorities(
        invocation=invocation,
        cleanup_control=control,
        ports=ports,
    )
    execution_arn = getattr(invocation, "state_machine_execution_arn", None)
    version_arn = getattr(
        invocation, "caller_state_machine_version_arn", None
    )
    dispatch_identity = getattr(
        invocation, "dispatch_identity_sha256", None
    )
    owner_attempt = 1
    owner_revision = control["revision"] + 1
    expires_at = (now + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    barrier = control.get("cleanup_barrier_nonce_sha256")
    if (
        type(execution_arn) is not str
        or not execution_arn
        or type(version_arn) is not str
        or not version_arn
        or type(dispatch_identity) is not str
        or len(dispatch_identity) != 64
        or type(barrier) is not str
        or len(barrier) != 64
    ):
        _fail("snapshot owner invocation authority is absent")
    capsule_authority = {
        "account_id": control["account_id"],
        "region": control["region"],
        "run_id": control["run_id"],
        "activation_id": control["activation_id"],
        "authority_domain": "SNAPSHOT_CLEANUP",
        "owner_execution_arn": execution_arn,
        "owner_state_machine_version_arn": version_arn,
        "owner_attempt": owner_attempt,
        "barrier_nonce_sha256": barrier,
        "control_revision": owner_revision,
        "owner_hard_expires_at": expires_at,
    }
    capsule, raw_nonce = generate_owner_nonce_capsule(
        ports=ports,
        authority=capsule_authority,
    )
    nonce_sha256 = hashlib.sha256(raw_nonce).hexdigest()
    if capsule.get("nonce_sha256") != nonce_sha256:
        _fail("snapshot owner nonce capsule identity drifted")
    after = {
        **control,
        "state": "OWNED",
        "owner_attempt": owner_attempt,
        "owner_execution_arn": execution_arn,
        "owner_state_machine_version_arn": version_arn,
        "owner_dispatch_identity_sha256": dispatch_identity,
        "owner_invocation_nonce_sha256": nonce_sha256,
        "owner_hard_expires_at": expires_at,
        "revision": owner_revision,
        "updated_at": observed_at,
    }
    transition_body = {
        "schema_version": 1,
        "record_type": "glm52_production_snapshot_cleanup_transition",
        "account_id": control["account_id"],
        "region": control["region"],
        "run_id": control["run_id"],
        "campaign_identity_sha256": control[
            "campaign_identity_sha256"
        ],
        "activation_id": control["activation_id"],
        "activation_ordinal": control["activation_ordinal"],
        "cleanup_control_root_identity_sha256": chain[
            "cleanup_control_root_identity_sha256"
        ],
        "from_state": control["state"],
        "to_state": after["state"],
        "from_revision": control["revision"],
        "to_revision": after["revision"],
        "prior_transition_sha256": chain[
            "cleanup_transition_chain_head_sha256"
        ],
        "owner_attempt": owner_attempt,
        "owner_execution_arn": execution_arn,
        "owner_state_machine_version_arn": version_arn,
        "owner_dispatch_identity_sha256": dispatch_identity,
        "owner_invocation_nonce_sha256": nonce_sha256,
        "authority_audit_identity_sha256": None,
        "action_identity_sha256": None,
        "transitioned_at": observed_at,
    }
    transition = validate_record(
        "glm52_production_snapshot_cleanup_transition",
        {
            **transition_body,
            "canonical_body_sha256": canonical_sha256(transition_body),
        },
    )
    chain_after = {
        **chain,
        "cleanup_transition_chain_head_sha256": transition[
            "canonical_body_sha256"
        ],
        "cleanup_transition_chain_length": (
            chain["cleanup_transition_chain_length"] + 1
        ),
        "revision": chain["revision"] + 1,
        "updated_at": observed_at,
    }
    activation_id = control["activation_id"]
    plan = build_snapshot_cleanup_transition_plan(
        index=ExactCheck(
            LedgerKey(RUN_ID, ledger_sk(
                "glm52_production_activation_index"
            )),
            index,
        ),
        cleanup_chain=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=activation_id,
                ),
            ),
            chain,
            chain_after,
        ),
        authority=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            authority,
        ),
        cleanup_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_control",
                    activation_id=activation_id,
                ),
            ),
            dict(control),
            after,
        ),
        cleanup_transition=ExactPut(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_transition",
                    activation_id=activation_id,
                    revision=after["revision"],
                ),
            ),
            transition,
        ),
    )
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "state_machine_execution_arn": execution_arn,
            "cleanup_control_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "cleanup_transition": transition[
                "canonical_body_sha256"
            ],
        }
    )
    del raw_nonce
    return {
        "plan": asdict(plan),
        "observed_at": observed_at,
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def _snapshot_takeover_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    from .dynamodb import ExactCheck, ExactUpdate, LedgerKey
    from .task12_nonce_capsule import generate_owner_nonce_capsule
    from .task12_retained_state import (
        build_snapshot_cleanup_owner_takeover_plan,
    )

    now = _snapshot_now(ports)
    observed_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    execution_arn = getattr(
        invocation, "state_machine_execution_arn", None
    )
    version_arn = getattr(
        invocation, "caller_state_machine_version_arn", None
    )
    dispatch_identity = getattr(
        invocation, "dispatch_identity_sha256", None
    )
    prior_expiry = control.get("owner_hard_expires_at")
    if (
        control.get("state")
        not in {"OWNED", "DELETE_POSSIBLY_SENT", "DELETE_RECONCILING"}
        or type(prior_expiry) is not str
        or prior_expiry >= observed_at
    ):
        _fail("snapshot takeover cannot replace a live owner")
    if (
        type(execution_arn) is not str
        or not execution_arn
        or execution_arn == control.get("owner_execution_arn")
        or type(version_arn) is not str
        or not version_arn
        or type(dispatch_identity) is not str
        or len(dispatch_identity) != 64
    ):
        _fail("snapshot takeover execution authority is foreign")
    index, authority, _chain = _snapshot_authorities(
        invocation=invocation,
        cleanup_control=control,
        ports=ports,
    )
    owner_attempt = control.get("owner_attempt")
    if type(owner_attempt) is not int or owner_attempt < 1:
        _fail("snapshot takeover owner attempt is invalid")
    owner_attempt += 1
    owner_revision = control["revision"] + 1
    expires_at = (now + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    barrier = control.get("cleanup_barrier_nonce_sha256")
    if type(barrier) is not str or len(barrier) != 64:
        _fail("snapshot takeover barrier authority is absent")
    capsule, raw_nonce = generate_owner_nonce_capsule(
        ports=ports,
        authority={
            "account_id": control["account_id"],
            "region": control["region"],
            "run_id": control["run_id"],
            "activation_id": control["activation_id"],
            "authority_domain": "SNAPSHOT_CLEANUP",
            "owner_execution_arn": execution_arn,
            "owner_state_machine_version_arn": version_arn,
            "owner_attempt": owner_attempt,
            "barrier_nonce_sha256": barrier,
            "control_revision": owner_revision,
            "owner_hard_expires_at": expires_at,
        },
    )
    nonce_sha256 = hashlib.sha256(raw_nonce).hexdigest()
    if capsule.get("nonce_sha256") != nonce_sha256:
        _fail("snapshot takeover nonce capsule identity drifted")
    after = {
        **control,
        "owner_attempt": owner_attempt,
        "owner_execution_arn": execution_arn,
        "owner_state_machine_version_arn": version_arn,
        "owner_dispatch_identity_sha256": dispatch_identity,
        "owner_invocation_nonce_sha256": nonce_sha256,
        "owner_hard_expires_at": expires_at,
        "revision": owner_revision,
        "updated_at": observed_at,
    }
    activation_id = control["activation_id"]
    plan = build_snapshot_cleanup_owner_takeover_plan(
        index=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk("glm52_production_activation_index"),
            ),
            index,
        ),
        authority=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            authority,
        ),
        cleanup_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_control",
                    activation_id=activation_id,
                ),
            ),
            dict(control),
            after,
        ),
    )
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "takeover": True,
            "control_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "control_after": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                after,
            ),
        }
    )
    del raw_nonce
    return {
        "plan": asdict(plan),
        "observed_at": observed_at,
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def _snapshot_prior_capsule(invocation: object) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if type(operation_input) is dict
        else None
    )
    result = prior.get("result") if type(prior) is dict else None
    capsule = (
        result.get("owner_nonce_capsule")
        if type(result) is dict
        else None
    )
    if type(capsule) is not dict:
        _fail("snapshot owner nonce continuation is absent")
    return capsule


def _snapshot_decrypt_nonce(
    *,
    invocation: object,
    control: Mapping[str, object],
    ports: object,
) -> tuple[dict[str, object], bytes]:
    from .task12_nonce_capsule import decrypt_owner_nonce_capsule

    capsule = _snapshot_prior_capsule(invocation)
    context = capsule.get("encryption_context")
    if type(context) is not dict:
        _fail("snapshot owner nonce context is absent")
    required_context = {
        "account_id": control.get("account_id"),
        "region": control.get("region"),
        "run_id": control.get("run_id"),
        "activation_id": control.get("activation_id"),
        "authority_domain": "SNAPSHOT_CLEANUP",
        "owner_execution_arn": control.get("owner_execution_arn"),
        "owner_state_machine_version_arn": control.get(
            "owner_state_machine_version_arn"
        ),
        "owner_attempt": str(control.get("owner_attempt")),
        "barrier_nonce_sha256": control.get(
            "cleanup_barrier_nonce_sha256"
        ),
        "owner_hard_expires_at": control.get("owner_hard_expires_at"),
    }
    if any(
        context.get(key) != str(value)
        for key, value in required_context.items()
    ):
        _fail("snapshot owner nonce context drifted")
    try:
        acquisition_revision = int(context.get("control_revision", ""))
    except (TypeError, ValueError) as exc:
        raise Task12LiveSnapshotFinalizationError(
            "snapshot owner acquisition revision is not typed"
        ) from exc
    if (
        type(control.get("revision")) is not int
        or acquisition_revision < 1
        or acquisition_revision > control["revision"]
        or control.get("owner_execution_arn")
        != getattr(invocation, "state_machine_execution_arn", None)
        or control.get("owner_state_machine_version_arn")
        != getattr(invocation, "caller_state_machine_version_arn", None)
    ):
        _fail("snapshot owner nonce revision or execution drifted")
    expected = {
        "account_id": control["account_id"],
        "region": control["region"],
        "run_id": control["run_id"],
        "activation_id": control["activation_id"],
        "authority_domain": "SNAPSHOT_CLEANUP",
        "owner_execution_arn": control["owner_execution_arn"],
        "owner_state_machine_version_arn": control[
            "owner_state_machine_version_arn"
        ],
        "owner_attempt": control["owner_attempt"],
        "barrier_nonce_sha256": control[
            "cleanup_barrier_nonce_sha256"
        ],
        "control_revision": acquisition_revision,
        "owner_hard_expires_at": control["owner_hard_expires_at"],
    }
    raw_nonce = decrypt_owner_nonce_capsule(
        ports=ports,
        capsule=capsule,
        expected_authority=expected,
    )
    if (
        hashlib.sha256(raw_nonce).hexdigest()
        != control.get("owner_invocation_nonce_sha256")
        or _snapshot_now(ports).strftime("%Y-%m-%dT%H:%M:%SZ")
        > control.get("owner_hard_expires_at", "")
    ):
        _fail("snapshot owner nonce is foreign or expired")
    return capsule, raw_nonce


def _snapshot_action_record(
    *,
    invocation: object,
    control: Mapping[str, object],
    attempt: int,
    observed_at: str,
) -> dict[str, object]:
    candidate_body = {
        "schema_version": 1,
        "snapshot_id": control["snapshot_id"],
        "snapshot_identity_sha256": control[
            "snapshot_identity_sha256"
        ],
        "attempt": attempt,
    }
    request_body = {"SnapshotId": control["snapshot_id"]}
    token_body = {
        "operation": "SNAPSHOT_DELETE_ARM",
        "activation_id": control["activation_id"],
        "attempt": attempt,
        "control_revision": control["revision"],
        "owner_invocation_nonce_sha256": control[
            "owner_invocation_nonce_sha256"
        ],
    }
    return validate_record(
        "glm52_production_snapshot_cleanup_action",
        {
            "schema_version": 1,
            "record_type": "glm52_production_snapshot_cleanup_action",
            "authority_domain": "SNAPSHOT_CLEANUP",
            "account_id": control["account_id"],
            "region": control["region"],
            "run_id": control["run_id"],
            "campaign_identity_sha256": control[
                "campaign_identity_sha256"
            ],
            "activation_id": control["activation_id"],
            "activation_ordinal": control["activation_ordinal"],
            "action_kind": "SNAPSHOT_DELETE",
            "attempt": attempt,
            "candidate_key": (
                "snapshot-delete/"
                + control["activation_id"]
                + "/"
                + f"{attempt:08d}"
            ),
            "candidate_body_sha256": canonical_sha256(candidate_body),
            "request_body_sha256": canonical_sha256(request_body),
            "owner_attempt": control["owner_attempt"],
            "owner_execution_arn": control["owner_execution_arn"],
            "owner_state_machine_version_arn": control[
                "owner_state_machine_version_arn"
            ],
            "owner_dispatch_identity_sha256": control[
                "owner_dispatch_identity_sha256"
            ],
            "owner_invocation_nonce_sha256": control[
                "owner_invocation_nonce_sha256"
            ],
            "owner_hard_expires_at": control[
                "owner_hard_expires_at"
            ],
            "authority_barrier_nonce_sha256": control[
                "cleanup_barrier_nonce_sha256"
            ],
            "authority_audit_body_sha256": None,
            "authority_audit_closing_revision": None,
            "authorized_transition_from_revision": None,
            "authorized_transition_to_revision": None,
            "state": "ARMED",
            "armed_at": observed_at,
            "consumed_at": None,
            "completed_at": None,
            "response_identity_sha256": None,
            "reconciliation_identity_sha256": None,
            "arming_transaction_client_request_token_sha256": (
                canonical_sha256(token_body)
            ),
            "consume_transaction_client_request_token_sha256": None,
            "revision": 1,
            "generation": getattr(invocation, "generation"),
            "generation_text": getattr(invocation, "generation_text"),
            "allocation_ordinal": None,
            "allocation_ordinal_text": None,
            "worker_launch_identity_sha256": None,
            "worker_launch_liability_identity_sha256": None,
        },
    )


def _snapshot_arm_action_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    from .dynamodb import ExactCheck, ExactPut, LedgerKey
    from .task12_retained_state import build_snapshot_delete_action_plan

    if control.get("state") not in {"OWNED", "DELETE_RECONCILING"}:
        _fail("snapshot delete action requires an owned reconcilable control")
    capsule, raw_nonce = _snapshot_decrypt_nonce(
        invocation=invocation,
        control=control,
        ports=ports,
    )
    attempt = control.get("delete_logical_attempt")
    if type(attempt) is not int or attempt < 0 or attempt >= 12:
        _fail("snapshot delete action exceeds the 12-attempt cap")
    attempt += 1
    observed_at = _snapshot_now(ports).strftime("%Y-%m-%dT%H:%M:%SZ")
    action = _snapshot_action_record(
        invocation=invocation,
        control=control,
        attempt=attempt,
        observed_at=observed_at,
    )
    index, _authority, _chain = _snapshot_authorities(
        invocation=invocation,
        cleanup_control=control,
        ports=ports,
    )
    action_sort_key = ledger_sk(
        "glm52_production_snapshot_cleanup_action",
        activation_id=control["activation_id"],
        attempt=attempt,
    )
    plan = build_snapshot_delete_action_plan(
        index=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk("glm52_production_activation_index"),
            ),
            index,
        ),
        control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_control",
                    activation_id=control["activation_id"],
                ),
            ),
            dict(control),
        ),
        action=ExactPut(
            LedgerKey(RUN_ID, action_sort_key),
            action,
        ),
    )
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "control_identity_sha256": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "action_identity_sha256": canonical_record_identity(
                "glm52_production_snapshot_cleanup_action",
                action,
            ),
        }
    )
    if hashlib.sha256(raw_nonce).hexdigest() != action[
        "owner_invocation_nonce_sha256"
    ]:
        _fail("snapshot armed action lost owner nonce binding")
    del raw_nonce
    return {
        "plan": asdict(plan),
        "partition_key": ledger_pk(RUN_ID),
        "sort_key": action_sort_key,
        "expected_record_identity_sha256": canonical_record_identity(
            "glm52_production_snapshot_cleanup_action",
            action,
        ),
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def _snapshot_close_action_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    action: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    from .dynamodb import ExactCheck, ExactUpdate, LedgerKey
    from .task12_retained_state import build_snapshot_delete_action_plan

    if (
        control.get("state") != "DELETE_RECONCILING"
        or action.get("state") != "CONSUMED"
        or action.get("attempt") != control.get("delete_logical_attempt")
    ):
        _fail("snapshot action close current control/action edge drifted")
    capsule, raw_nonce = _snapshot_decrypt_nonce(
        invocation=invocation,
        control=control,
        ports=ports,
    )
    reconciliation = _snapshot_reconciliation_result(invocation)
    observed_at = _snapshot_now(ports).strftime("%Y-%m-%dT%H:%M:%SZ")
    response_identity = control.get("latest_delete_response_sha256")
    reconciliation_identity = control.get(
        "last_describe_response_sha256"
    )
    action_owned_by_current_owner = all(
        action.get(field) == control.get(field)
        for field in (
            "owner_attempt",
            "owner_execution_arn",
            "owner_state_machine_version_arn",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
        )
    )
    if (
        action_owned_by_current_owner
        and type(response_identity) is str
        and len(response_identity) == 64
    ):
        target = "COMPLETED"
        action_after = {
            **action,
            "state": target,
            "completed_at": observed_at,
            "response_identity_sha256": response_identity,
            "revision": action["revision"] + 1,
        }
    else:
        if (
            type(reconciliation_identity) is not str
            or len(reconciliation_identity) != 64
        ):
            _fail("ambiguous snapshot action lacks reconciliation evidence")
        target = "AMBIGUOUS"
        action_after = {
            **action,
            "state": target,
            "completed_at": observed_at,
            "reconciliation_identity_sha256": reconciliation_identity,
            "revision": action["revision"] + 1,
        }
    index, _authority, _chain = _snapshot_authorities(
        invocation=invocation,
        cleanup_control=control,
        ports=ports,
    )
    plan = build_snapshot_delete_action_plan(
        index=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk("glm52_production_activation_index"),
            ),
            index,
        ),
        control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_control",
                    activation_id=control["activation_id"],
                ),
            ),
            dict(control),
        ),
        action=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_action",
                    activation_id=control["activation_id"],
                    attempt=action["attempt"],
                ),
            ),
            dict(action),
            action_after,
        ),
    )
    if hashlib.sha256(raw_nonce).hexdigest() != control[
        "owner_invocation_nonce_sha256"
    ]:
        _fail("snapshot action close owner nonce drifted")
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "control_identity_sha256": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "action_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_action",
                action,
            ),
            "action_after": canonical_record_identity(
                "glm52_production_snapshot_cleanup_action",
                action_after,
            ),
            "reconciliation_result_sha256": canonical_sha256(
                reconciliation
            ),
        }
    )
    del raw_nonce
    return {
        "plan": asdict(plan),
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def _snapshot_arm_next_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    reconciliation = _snapshot_reconciliation_result(invocation)
    if (
        reconciliation["snapshot_present"] is not True
        or control.get("state") != "DELETE_RECONCILING"
        or control.get("delete_logical_attempt", 0) >= 12
    ):
        _fail("snapshot retry requires a present snapshot below the cap")
    armed = _snapshot_arm_action_request(
        invocation=invocation,
        control=control,
        ports=ports,
    )
    return {
        key: armed[key]
        for key in (
            "plan",
            "domain",
            "operation_identity_sha256",
            "owner_nonce_capsule",
        )
    }


def _snapshot_transition_plan(
    *,
    invocation: object,
    control: Mapping[str, object],
    after: Mapping[str, object],
    action_before: Mapping[str, object] | None,
    action_after: Mapping[str, object] | None,
    observed_at: str,
    ports: object,
    transition_action_identity: str | None = None,
) -> object:
    from .dynamodb import ExactCheck, ExactPut, ExactUpdate, LedgerKey
    from .task12_retained_state import (
        build_snapshot_cleanup_transition_plan,
    )

    index, authority, chain = _snapshot_authorities(
        invocation=invocation,
        cleanup_control=control,
        ports=ports,
    )
    action_identity = (
        transition_action_identity
        if action_before is None
        else canonical_record_identity(
            "glm52_production_snapshot_cleanup_action",
            action_before,
        )
    )
    transition_body = {
        "schema_version": 1,
        "record_type": "glm52_production_snapshot_cleanup_transition",
        "account_id": control["account_id"],
        "region": control["region"],
        "run_id": control["run_id"],
        "campaign_identity_sha256": control[
            "campaign_identity_sha256"
        ],
        "activation_id": control["activation_id"],
        "activation_ordinal": control["activation_ordinal"],
        "cleanup_control_root_identity_sha256": chain[
            "cleanup_control_root_identity_sha256"
        ],
        "from_state": control["state"],
        "to_state": after["state"],
        "from_revision": control["revision"],
        "to_revision": after["revision"],
        "prior_transition_sha256": chain[
            "cleanup_transition_chain_head_sha256"
        ],
        "owner_attempt": after.get("owner_attempt"),
        "owner_execution_arn": after.get("owner_execution_arn"),
        "owner_state_machine_version_arn": after.get(
            "owner_state_machine_version_arn"
        ),
        "owner_dispatch_identity_sha256": after.get(
            "owner_dispatch_identity_sha256"
        ),
        "owner_invocation_nonce_sha256": after.get(
            "owner_invocation_nonce_sha256"
        ),
        "authority_audit_identity_sha256": (
            after.get("latest_cleanup_authority_audit_identity_sha256")
        ),
        "action_identity_sha256": action_identity,
        "transitioned_at": observed_at,
    }
    transition = validate_record(
        "glm52_production_snapshot_cleanup_transition",
        {
            **transition_body,
            "canonical_body_sha256": canonical_sha256(transition_body),
        },
    )
    chain_after = {
        **chain,
        "cleanup_transition_chain_head_sha256": transition[
            "canonical_body_sha256"
        ],
        "cleanup_transition_chain_length": (
            chain["cleanup_transition_chain_length"] + 1
        ),
        "revision": chain["revision"] + 1,
        "updated_at": observed_at,
    }
    activation_id = control["activation_id"]
    cleanup_action = (
        None
        if action_before is None or action_after is None
        else ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_action",
                    activation_id=activation_id,
                    attempt=action_before["attempt"],
                ),
            ),
            dict(action_before),
            dict(action_after),
        )
    )
    return build_snapshot_cleanup_transition_plan(
        index=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk("glm52_production_activation_index"),
            ),
            index,
        ),
        cleanup_chain=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=activation_id,
                ),
            ),
            chain,
            chain_after,
        ),
        authority=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=activation_id,
                ),
            ),
            authority,
        ),
        cleanup_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_control",
                    activation_id=activation_id,
                ),
            ),
            dict(control),
            dict(after),
        ),
        cleanup_transition=ExactPut(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_snapshot_cleanup_transition",
                    activation_id=activation_id,
                    revision=after["revision"],
                ),
            ),
            transition,
        ),
        cleanup_action=cleanup_action,
    )


def _snapshot_atomic_stage_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    action: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    if (
        control.get("state") not in {"OWNED", "DELETE_RECONCILING"}
        or action.get("state") != "ARMED"
        or action.get("attempt")
        != control.get("delete_logical_attempt", 0) + 1
    ):
        _fail("snapshot stage current control/action edge drifted")
    capsule, raw_nonce = _snapshot_decrypt_nonce(
        invocation=invocation,
        control=control,
        ports=ports,
    )
    action_identity = canonical_record_identity(
        "glm52_production_snapshot_cleanup_action",
        action,
    )
    now = _snapshot_now(ports)
    observed_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    next_revision = control["revision"] + 1
    action_after = {
        **action,
        "authority_audit_body_sha256": action_identity,
        "authority_audit_closing_revision": control["revision"],
        "authorized_transition_from_revision": control["revision"],
        "authorized_transition_to_revision": next_revision,
        "state": "CONSUMED",
        "consumed_at": observed_at,
        "consume_transaction_client_request_token_sha256": canonical_sha256(
            {
                "operation": "SNAPSHOT_DELETE_CONSUME",
                "activation_id": control["activation_id"],
                "attempt": action["attempt"],
                "control_revision": control["revision"],
            }
        ),
        "revision": action["revision"] + 1,
    }
    action_after = validate_record(
        "glm52_production_snapshot_cleanup_action",
        action_after,
    )
    after = {
        **control,
        "state": "DELETE_POSSIBLY_SENT",
        "cleanup_authority_audit_identities": [
            *control["cleanup_authority_audit_identities"],
            action_identity,
        ],
        "latest_cleanup_authority_audit_identity_sha256": action_identity,
        "delete_action_identities": [
            *control["delete_action_identities"],
            action_identity,
        ],
        "latest_delete_action_identity_sha256": action_identity,
        "delete_logical_attempt": action["attempt"],
        "delete_call_count": control["delete_call_count"] + 1,
        "revision": next_revision,
        "updated_at": observed_at,
    }
    plan = _snapshot_transition_plan(
        invocation=invocation,
        control=control,
        after=after,
        action_before=action,
        action_after=action_after,
        observed_at=observed_at,
        ports=ports,
    )
    if hashlib.sha256(raw_nonce).hexdigest() != control[
        "owner_invocation_nonce_sha256"
    ]:
        _fail("snapshot stage owner nonce drifted")
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "control_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "action_before": action_identity,
            "control_after": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                after,
            ),
        }
    )
    del raw_nonce
    return {
        "plan": asdict(plan),
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def _snapshot_prior_domain_result(invocation: object) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if type(operation_input) is dict
        else None
    )
    result = prior.get("result") if type(prior) is dict else None
    domain = (
        result.get("domain_result")
        if type(result) is dict
        else None
    )
    if type(domain) is not dict:
        _fail("snapshot predecessor domain result is absent")
    return domain


def _snapshot_reconcile_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    action: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    if control.get("state") not in {
        "DELETE_POSSIBLY_SENT",
        "DELETE_RECONCILING",
    }:
        _fail("snapshot describe reconciliation state is not resumable")
    capsule, raw_nonce = _snapshot_decrypt_nonce(
        invocation=invocation,
        control=control,
        ports=ports,
    )
    operation_input = getattr(invocation, "operation_input", None)
    ambiguity = (
        operation_input.get("delete_ambiguity")
        if type(operation_input) is dict
        else None
    )
    prior_domain = _snapshot_prior_domain_result(invocation)
    recovered_owner = (
        set(prior_domain) == {"cleanup_state", "resolution"}
        and prior_domain.get("cleanup_state") == control.get("state")
        and type(prior_domain.get("resolution")) is dict
    )
    if control.get("state") == "DELETE_RECONCILING":
        if ambiguity is not None or not recovered_owner:
            _fail(
                "reconciling snapshot recovery requires authenticated takeover"
            )
        from .dynamodb import ExactCheck, LedgerKey
        from .task12_retained_state import (
            build_snapshot_cleanup_reconcile_read_plan,
        )

        index, authority, _chain = _snapshot_authorities(
            invocation=invocation,
            cleanup_control=control,
            ports=ports,
        )
        plan = build_snapshot_cleanup_reconcile_read_plan(
            index=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk("glm52_production_activation_index"),
                ),
                index,
            ),
            authority=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_control",
                        activation_id=control["activation_id"],
                    ),
                ),
                authority,
            ),
            cleanup_control=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_snapshot_cleanup_control",
                        activation_id=control["activation_id"],
                    ),
                ),
                dict(control),
            ),
            cleanup_action=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_snapshot_cleanup_action",
                        activation_id=control["activation_id"],
                        attempt=control["delete_logical_attempt"],
                    ),
                ),
                dict(action),
            ),
        )
        if hashlib.sha256(raw_nonce).hexdigest() != control[
            "owner_invocation_nonce_sha256"
        ]:
            _fail("snapshot reconciliation owner nonce drifted")
        operation_identity = canonical_sha256(
            {
                "operation_kind": getattr(invocation, "operation_kind"),
                "control_identity_sha256": canonical_record_identity(
                    "glm52_production_snapshot_cleanup_control",
                    control,
                ),
                "action_identity_sha256": canonical_record_identity(
                    "glm52_production_snapshot_cleanup_action",
                    action,
                ),
                "predecessor_evidence_sha256": canonical_sha256(
                    prior_domain
                ),
            }
        )
        del raw_nonce
        return {
            "plan": asdict(plan),
            "domain": "SNAPSHOT_CLEANUP",
            "operation_identity_sha256": operation_identity,
            "owner_nonce_capsule": capsule,
        }
    if ambiguity is None:
        if recovered_owner:
            prior = None
            predecessor_identity = canonical_sha256(prior_domain)
        else:
            prior = prior_domain
            if (
                set(prior) != {"outcome", "request_id", "response_sha256"}
                or prior.get("outcome")
                not in {"ACCEPTED", "DELETION_IN_PROGRESS", "NOT_FOUND"}
                or type(prior.get("request_id")) is not str
                or not prior["request_id"]
                or type(prior.get("response_sha256")) is not str
                or len(prior["response_sha256"]) != 64
            ):
                _fail("snapshot delete predecessor result drifted")
            predecessor_identity = canonical_sha256(prior)
    else:
        if (
            type(ambiguity) is not dict
            or set(ambiguity) != {"Error", "Cause"}
            or any(
                type(ambiguity.get(field)) is not str
                or not ambiguity[field]
                for field in ("Error", "Cause")
            )
        ):
            _fail("snapshot delete ambiguity is malformed")
        prior = None
        predecessor_identity = canonical_sha256(ambiguity)
    live = _snapshot_capture_and_schedule(
        control=control,
        ports=ports,
        include_describe_evidence=True,
    )
    observed_at = _snapshot_now(ports).strftime("%Y-%m-%dT%H:%M:%SZ")
    after = {
        **control,
        "state": "DELETE_RECONCILING",
        "last_describe_request_id": live["describe_request_id"],
        "last_describe_response_sha256": live[
            "describe_response_sha256"
        ],
        "revision": control["revision"] + 1,
        "updated_at": observed_at,
    }
    if prior is not None:
        after["latest_delete_request_id"] = prior["request_id"]
        after["latest_delete_response_sha256"] = prior[
            "response_sha256"
        ]
    plan = _snapshot_transition_plan(
        invocation=invocation,
        control=control,
        after=after,
        action_before=None,
        action_after=None,
        observed_at=observed_at,
        ports=ports,
        transition_action_identity=control[
            "latest_delete_action_identity_sha256"
        ],
    )
    if hashlib.sha256(raw_nonce).hexdigest() != control[
        "owner_invocation_nonce_sha256"
    ]:
        _fail("snapshot reconciliation owner nonce drifted")
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "control_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "observation_identity_sha256": live[
                "observation_identity_sha256"
            ],
            "predecessor_evidence_sha256": predecessor_identity,
            "control_after": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                after,
            ),
        }
    )
    del raw_nonce
    return {
        "plan": asdict(plan),
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def _snapshot_reconciliation_result(
    invocation: object,
) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if type(operation_input) is dict
        else None
    )
    prior_result = (
        prior.get("result") if type(prior) is dict else None
    )
    result = (
        prior_result.get("domain_result")
        if type(prior_result) is dict
        else None
    )
    result_fields = {
        "source_state",
        "target_state",
        "committed",
        "snapshot_present",
        "delete_logical_attempt",
        "resolution",
    }
    if not (
        type(result) is dict
        and set(result) == result_fields
    ):
        retained = (
            operation_input.get("snapshot_reconciliation_result")
            if type(operation_input) is dict
            else None
        )
        if (
            type(retained) is not dict
            or retained.get("schema_version") != 1
            or retained.get("record_type")
            != "glm52_task12_lambda_result_v1"
            or retained.get("operation_kind")
            != "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
            or retained.get("outcome") != "SUCCEEDED"
            or type(retained.get("result")) is not dict
            or type(retained["result"].get("domain_result")) is not dict
        ):
            _fail("snapshot reconciliation predecessor result is absent")
        retained_body = dict(retained)
        retained_identity = retained_body.pop(
            "canonical_body_sha256", None
        )
        if (
            type(retained_identity) is not str
            or retained_identity != canonical_sha256(retained_body)
        ):
            _fail("snapshot retained reconciliation result drifted")
        result = retained["result"]["domain_result"]
    source = result.get("source_state")
    target = result.get("target_state")
    present = result.get("snapshot_present")
    attempt = result.get("delete_logical_attempt")
    committed = result.get("committed")
    resolution = result.get("resolution")
    normal_transition = (
        source == "DELETE_POSSIBLY_SENT"
        and target == "DELETE_RECONCILING"
        and committed is True
        and type(resolution) is dict
    )
    recovered_read = (
        source == "DELETE_RECONCILING"
        and committed is False
        and resolution is None
        and type(attempt) is int
        and (
            (present is False and target == "DELETED")
            or (
                present is True
                and attempt < 12
                and target == "DELETE_RECONCILING"
            )
            or (
                present is True
                and attempt >= 12
                and target == "CLEANUP_INCIDENT"
            )
        )
    )
    if (
        type(present) is not bool
        or type(attempt) is not int
        or attempt < 0
        or (not normal_transition and not recovered_read)
    ):
        _fail("snapshot reconciliation predecessor result drifted")
    return result


def _snapshot_terminal_request(
    *,
    invocation: object,
    control: Mapping[str, object],
    ports: object,
) -> dict[str, object]:
    if control.get("state") != "DELETE_RECONCILING":
        _fail("snapshot terminal evidence requires reconciling control")
    capsule, raw_nonce = _snapshot_decrypt_nonce(
        invocation=invocation,
        control=control,
        ports=ports,
    )
    reconciliation = _snapshot_reconciliation_result(invocation)
    snapshot_present = reconciliation["snapshot_present"]
    attempt = control.get("delete_logical_attempt")
    if snapshot_present and (
        type(attempt) is not int or attempt < 12
    ):
        _fail("present snapshot has not exhausted the bounded retries")
    target = "CLEANUP_INCIDENT" if snapshot_present else "DELETED"
    observed_at = _snapshot_now(ports).strftime("%Y-%m-%dT%H:%M:%SZ")
    terminal_evidence = canonical_sha256(
        {
            "schema_version": 1,
            "activation_id": control["activation_id"],
            "snapshot_identity_sha256": control[
                "snapshot_identity_sha256"
            ],
            "delete_logical_attempt": attempt,
            "delete_call_count": control["delete_call_count"],
            "latest_delete_request_id": control[
                "latest_delete_request_id"
            ],
            "latest_delete_response_sha256": control[
                "latest_delete_response_sha256"
            ],
            "last_describe_request_id": control[
                "last_describe_request_id"
            ],
            "last_describe_response_sha256": control[
                "last_describe_response_sha256"
            ],
            "snapshot_present": snapshot_present,
            "target_state": target,
            "observed_at": observed_at,
        }
    )
    after = {
        **control,
        "state": target,
        "owner_attempt": None,
        "owner_execution_arn": None,
        "owner_state_machine_version_arn": None,
        "owner_dispatch_identity_sha256": None,
        "owner_invocation_nonce_sha256": None,
        "owner_hard_expires_at": None,
        "terminal_evidence_sha256": terminal_evidence,
        "revision": control["revision"] + 1,
        "updated_at": observed_at,
    }
    plan = _snapshot_transition_plan(
        invocation=invocation,
        control=control,
        after=after,
        action_before=None,
        action_after=None,
        observed_at=observed_at,
        ports=ports,
        transition_action_identity=control[
            "latest_delete_action_identity_sha256"
        ],
    )
    if hashlib.sha256(raw_nonce).hexdigest() != control[
        "owner_invocation_nonce_sha256"
    ]:
        _fail("snapshot terminal evidence owner nonce drifted")
    operation_identity = canonical_sha256(
        {
            "operation_kind": getattr(invocation, "operation_kind"),
            "control_before": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                control,
            ),
            "reconciliation_result_sha256": canonical_sha256(
                reconciliation
            ),
            "control_after": canonical_record_identity(
                "glm52_production_snapshot_cleanup_control",
                after,
            ),
        }
    )
    del raw_nonce
    return {
        "plan": asdict(plan),
        "domain": "SNAPSHOT_CLEANUP",
        "operation_identity_sha256": operation_identity,
        "owner_nonce_capsule": capsule,
    }


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> Mapping[str, object] | None:
    """Build a request from current truth, or fail before any domain effect."""

    if operation_kind not in SUPPORTED_OPERATIONS:
        return None
    sources = _validate_live_sources(
        operation_kind=operation_kind,
        invocation=invocation,
        live_sources=live_sources,
    )
    if operation_kind not in _DIRECT_OPERATIONS:
        _fail(operation_kind + " has no direct live request producer")
    if operation_kind == "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED":
        request = _retained_writer_request(
            writer_kind="SupportPlaneFinalized",
            record_type="glm52_production_support_plane_finalized",
            control=sources["finalization_control"],
            invocation=invocation,
            ports=ports,
        )
    elif operation_kind == "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT":
        from .cloudformation_stacks import StackKind, stack_name

        request = {"stack_id": stack_name(StackKind.SUPPORT)}
    elif operation_kind == "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT":
        request = _retained_snapshot_discovery_request(
            invocation=invocation, ports=ports
        )
    elif operation_kind == "RETAINED_AUDIT_SUPPORT_ORPHANS":
        request = _retained_orphan_request(
            invocation=invocation,
            ports=ports,
        )
    elif (
        operation_kind
        == "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE"
    ):
        request = _retained_snapshot_arm_request(
            invocation=invocation, sources=sources, ports=ports
        )
    elif operation_kind == "RETAINED_INVOKE_H1G_DRAINED_WRITER":
        request = _retained_h1g_request(
            invocation=invocation, sources=sources, ports=ports
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE":
        request = _snapshot_capture_and_schedule(
            control=sources["snapshot_cleanup_control"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER":
        request = _snapshot_acquire_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION":
        request = _snapshot_arm_action_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY":
        request = _audit_delete_request(
            invocation=invocation,
            action=sources["snapshot_cleanup_action"],
        )
    elif (
        operation_kind
        == "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT"
    ):
        request = _snapshot_atomic_stage_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            action=sources["snapshot_cleanup_action"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE":
        request = _snapshot_reconcile_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            action=sources["snapshot_cleanup_action"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE":
        request = _snapshot_terminal_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT":
        request = _snapshot_close_action_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            action=sources["snapshot_cleanup_action"],
            ports=ports,
        )
    elif operation_kind == "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT":
        request = _snapshot_arm_next_request(
            invocation=invocation,
            control=sources["snapshot_cleanup_control"],
            ports=ports,
        )
    else:
        request = _send_request(
            control=sources["snapshot_cleanup_control"],
            transition=sources["snapshot_cleanup_transition"],
        )
    return validate_operation_request_payload(operation_kind, request)


def _snapshot_committed_readback(
    *,
    operation_kind: str,
    request: Mapping[str, object],
    domain_result: object,
) -> None:
    from .dynamodb import TransactionResolution, WriteOutcome
    from .task12_snapshot_cleanup import (
        ReconciliationResult,
        SnapshotOwnerAcquisitionResult,
    )

    derived_fields = {
        "arming_transaction_client_request_token_sha256",
        "consume_transaction_client_request_token_sha256",
    }

    def matches(
        expected: Mapping[str, object],
        actual: Mapping[str, object],
    ) -> bool:
        if expected.get("record_type") != actual.get("record_type"):
            return False
        canonical_record_identity(
            str(expected["record_type"]), expected
        )
        canonical_record_identity(str(actual["record_type"]), actual)
        return {
            key: value
            for key, value in expected.items()
            if key not in derived_fields
        } == {
            key: value
            for key, value in actual.items()
            if key not in derived_fields
        }

    if (
        operation_kind
        == "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER"
    ):
        expected_control = request.get("plan", {}).get(
            "cleanup_control", {}
        ).get("after")
        if (
            type(domain_result) is not SnapshotOwnerAcquisitionResult
            or type(expected_control) is not dict
            or domain_result.cleanup_state != expected_control.get("state")
        ):
            _fail(
                "snapshot acquisition lacks authenticated committed readback"
            )
        resolution = domain_result.resolution
    elif operation_kind == "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION":
        if (
            type(domain_result) is not dict
            or set(domain_result) != {"transaction", "live_action"}
            or type(domain_result.get("live_action")) is not dict
        ):
            _fail("snapshot successor lacks authenticated committed readback")
        resolution = domain_result["transaction"]
        expected_action = request.get("plan", {}).get("action", {}).get(
            "item"
        )
        if (
            type(expected_action) is not dict
            or not matches(expected_action, domain_result["live_action"])
        ):
            _fail("snapshot successor armed-action readback drifted")
    elif operation_kind == "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE":
        read_plan = set(request.get("plan", {})) == {
            "index",
            "authority",
            "cleanup_control",
            "cleanup_action",
        }
        if (
            type(domain_result) is not ReconciliationResult
        ):
            _fail("snapshot successor lacks authenticated committed readback")
        if read_plan:
            control = request["plan"]["cleanup_control"]["expected"]
            if (
                domain_result.source_state != "DELETE_RECONCILING"
                or domain_result.committed is not False
                or domain_result.resolution is not None
                or domain_result.delete_logical_attempt
                != control.get("delete_logical_attempt")
            ):
                _fail(
                    "snapshot read-only reconciliation result drifted"
                )
            return
        if (
            domain_result.committed is not True
            or domain_result.resolution is None
        ):
            _fail("snapshot successor lacks authenticated committed readback")
        resolution = domain_result.resolution
    else:
        resolution = domain_result
    if (
        type(resolution) is not TransactionResolution
        or resolution.outcome
        not in {
            WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
            WriteOutcome.EXACT_DURABLE_ADOPTION,
        }
        or type(resolution.records) is not tuple
        or not resolution.records
    ):
        _fail("snapshot successor lacks authenticated committed readback")
    plan = request.get("plan")
    if type(plan) is not dict:
        _fail("snapshot successor request plan is absent")
    expected_records = []
    for member in plan.values():
        if member is None:
            continue
        if type(member) is not dict:
            _fail("snapshot successor plan member is malformed")
        record = member.get(
            "after", member.get("item", member.get("expected"))
        )
        if type(record) is not dict or type(record.get("record_type")) is not str:
            _fail("snapshot successor plan post-record is malformed")
        expected_records.append(record)
    actual_records = []
    for record in resolution.records:
        if type(record) is not dict or type(record.get("record_type")) is not str:
            _fail("snapshot successor readback record is malformed")
        canonical_record_identity(record["record_type"], record)
        actual_records.append(record)
    if not expected_records or any(
        not any(matches(expected, actual) for actual in actual_records)
        for expected in expected_records
    ):
        _fail("snapshot successor authenticated committed readback drifted")


def persist_live_successors(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> bool:
    """Acknowledge canonical effects without writing any future request.

    Only actual domain effects are durably advanced.  No future request or
    projected result is accepted by this boundary.
    """

    if operation_kind not in SUPPORTED_OPERATIONS:
        return False
    if operation_kind == "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED":
        _retained_persist_writer_successor(
            operation_kind=operation_kind,
            invocation=invocation,
            source_control=live_sources["finalization_control"],
            request=request,
            domain_result=domain_result,
            ports=ports,
        )
        return True
    if operation_kind == "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT":
        _retained_persist_snapshot_effect(
            invocation=invocation,
            support=live_sources["support_finalized"],
            domain_result=domain_result,
            ports=ports,
        )
        return True
    if operation_kind == "RETAINED_AUDIT_SUPPORT_ORPHANS":
        _retained_persist_orphan_effect(
            invocation=invocation,
            domain_result=domain_result,
            ports=ports,
        )
        return True
    if operation_kind == "RETAINED_INVOKE_H1G_DRAINED_WRITER":
        current = _retained_read_record(
            ports=ports,
            activation_id=getattr(invocation, "activation_id"),
            record_type="glm52_production_finalization_control",
        )
        _retained_persist_writer_successor(
            operation_kind=operation_kind,
            invocation=invocation,
            source_control=current,
            request=request,
            domain_result=domain_result,
            ports=ports,
        )
        return True
    if (
        operation_kind
        == "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE"
    ):
        _retained_persist_snapshot_arm_successor(
            invocation=invocation,
            source_control=live_sources["finalization_control"],
            request=request,
            domain_result=domain_result,
            ports=ports,
        )
        return True
    if operation_kind in {
        "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
    }:
        _snapshot_committed_readback(
            operation_kind=operation_kind,
            request=request,
            domain_result=domain_result,
        )
        return True
    del invocation, live_sources, request, domain_result, ports
    return True


__all__ = [
    "SUPPORTED_OPERATIONS",
    "Task12LiveSnapshotFinalizationError",
    "materialize_live_request",
    "persist_live_successors",
]
