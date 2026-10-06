from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import format_datetime
import hashlib
import json

import pytest


RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "activation-0001"
ACTION_KEY = (
    f"ACTIVATION#{ACTIVATION_ID}#ACTION#00000001#S3_CREATE#00000001"
)
SHA = "a" * 64
OWNER = "246813579024"
REGION = "us-west-2"
BUCKET = "keep-glm52-production"
NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_every_frozen_source_and_marker_key_is_exact() -> None:
    from glm52_enforcement import s3_keys

    assert s3_keys.gpu_spend_approval_s3_key(
        run_id=RUN_ID, approval_file_sha256=SHA
    ) == f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL-{SHA}.json"
    assert s3_keys.gpu_spend_snapshot_s3_key(
        run_id=RUN_ID, snapshot_body_sha256=SHA
    ) == (
        f"campaigns/{RUN_ID}/spend-snapshots/{SHA}/GPU_SPEND_SNAPSHOT.json"
    )
    assert s3_keys.production_submission_intent_s3_key(
        run_id=RUN_ID, intent_body_sha256=SHA
    ) == (
        f"campaigns/{RUN_ID}/submissions/production/intents/{SHA}/"
        "SKYPILOT_SUBMISSION_INTENT.json"
    )
    assert s3_keys.production_controller_baseline_s3_key(
        run_id=RUN_ID, baseline_body_sha256=SHA
    ) == (
        f"campaigns/{RUN_ID}/production/controller-baselines/{SHA}/"
        "CONTROLLER_BASELINE.json"
    )
    assert s3_keys.production_must_start_control_plane_ready_s3_key(
        run_id=RUN_ID,
        intent_body_sha256=SHA,
        control_plane_ready_body_sha256="b" * 64,
    ) == (
        f"campaigns/{RUN_ID}/monitor/must-start/production/{SHA}/"
        f"control-plane-ready/{'b' * 64}/CONTROL_PLANE_READY.json"
    )
    assert s3_keys.production_submission_acquired_s3_key(
        run_id=RUN_ID, descriptor_file_sha256=SHA
    ) == (
        f"campaigns/{RUN_ID}/submissions/production/acquisitions/{SHA}/"
        "SUBMISSION_ACQUIRED.json"
    )
    assert s3_keys.fence_genesis_s3_key(run_id=RUN_ID) == (
        f"campaigns/{RUN_ID}/authorities/fence/FENCE_GENESIS.json"
    )
    assert s3_keys.fence_successor_s3_key(
        run_id=RUN_ID, predecessor_body_sha256=SHA
    ) == (
        f"campaigns/{RUN_ID}/authorities/fence/successors/{SHA}/"
        "FENCE_SUCCESSOR.json"
    )
    assert s3_keys.sky_post_handoff_s3_key(
        run_id=RUN_ID, generation=1
    ).endswith("/00000001/handoff/SKY_POST_HANDOFF.json")
    assert s3_keys.bootstrap_ready_s3_key(
        run_id=RUN_ID, generation=1, allocation_ordinal=2
    ).endswith("/00000001/allocations/00000002/BOOTSTRAP_READY.json")
    assert s3_keys.worker_graceful_stop_s3_key(
        run_id=RUN_ID, generation=1, allocation_ordinal=2
    ).endswith(
        "/00000001/allocations/00000002/WORKER_GRACEFUL_STOP.json"
    )
    assert s3_keys.campaign_drained_s3_key(
        run_id=RUN_ID, generation=1
    ).endswith("/00000001/terminal/CAMPAIGN_DRAINED.json")
    assert s3_keys.support_plane_finalized_s3_key(
        run_id=RUN_ID, activation_id=ACTIVATION_ID
    ).endswith(
        f"/activations/{ACTIVATION_ID}/finalization/"
        "SUPPORT_PLANE_FINALIZED.json"
    )
    assert s3_keys.h1g_drained_s3_key(
        run_id=RUN_ID, activation_id=ACTIVATION_ID
    ).endswith(
        f"/activations/{ACTIVATION_ID}/finalization/H1G_DRAINED.json"
    )


def test_unsafe_foreign_generic_and_root_marker_keys_are_rejected() -> None:
    from glm52_enforcement import s3_keys

    valid = s3_keys.campaign_drained_s3_key(run_id=RUN_ID, generation=1)
    assert (
        s3_keys.validate_frozen_key(
            key=valid,
            run_id=RUN_ID,
            allowed_families=("campaign-drained",),
        )
        == valid
    )
    invalid_calls = (
        lambda: s3_keys.fence_genesis_s3_key(run_id="../foreign"),
        lambda: s3_keys.fence_successor_s3_key(
            run_id=RUN_ID, predecessor_body_sha256="A" * 64
        ),
        lambda: s3_keys.sky_post_handoff_s3_key(
            run_id=RUN_ID, generation=True
        ),
        lambda: s3_keys.bootstrap_ready_s3_key(
            run_id=RUN_ID, generation=1, allocation_ordinal=0
        ),
        lambda: s3_keys.support_plane_finalized_s3_key(
            run_id=RUN_ID, activation_id="other/activation"
        ),
        lambda: s3_keys.validate_frozen_key(
            key="CAMPAIGN_DRAINED.json",
            run_id=RUN_ID,
            allowed_families=("campaign-drained",),
        ),
        lambda: s3_keys.validate_frozen_key(
            key=valid.replace(RUN_ID, "foreign-run"),
            run_id=RUN_ID,
            allowed_families=("campaign-drained",),
        ),
        lambda: s3_keys.validate_frozen_key(
            key=valid.replace(
                "CAMPAIGN_DRAINED.json", "GENERIC_MARKER.json"
            ),
            run_id=RUN_ID,
            allowed_families=("campaign-drained",),
        ),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()

    assert not hasattr(s3_keys, "generic_marker_s3_key")
    assert not hasattr(s3_keys, "root_marker_s3_key")
    assert not hasattr(s3_keys, "alternate_filename_s3_key")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _candidate():
    from glm52_enforcement.s3_records import build_immutable_json_candidate

    body = {
        "account_id": OWNER,
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "generation_text": "00000001",
        "record_type": "glm52_production_terminal_v2",
        "region": REGION,
        "run_id": RUN_ID,
        "schema_version": 2,
    }
    raw = _canonical(
        {
            **body,
            "canonical_body_sha256": hashlib.sha256(
                _canonical(body)
            ).hexdigest(),
        }
    ) + b"\n"
    return build_immutable_json_candidate(
        record_kind="terminal-v2",
        bucket=BUCKET,
        key=(
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            "00000001/terminal/PRODUCTION_TERMINAL_V2.json"
        ),
        raw=raw,
        activation_id=ACTIVATION_ID,
        generation=1,
    )


def _audit(candidate):
    from glm52_enforcement.h1f_adapter import H1fAuditResult

    provisional = H1fAuditResult(
        authority_domain="ACTIVATION",
        operation_kind="S3_CREATE",
        action_key=ACTION_KEY,
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        activation_id=ACTIVATION_ID,
        generation=1,
        epoch=2,
        execution_arn="arn:aws:states:us-west-2:246813579024:execution:x:y",
        barrier_nonce_sha256="b" * 64,
        closing_revision=7,
        expected_authorized_revision=8,
        active_head_key=f"campaigns/{RUN_ID}/authorities/fence/FENCE_GENESIS.json",
        active_head_version_id="head-version",
        active_head_file_sha256="c" * 64,
        active_head_body_sha256="d" * 64,
        canonical_body_sha256="0" * 64,
    )
    body = {"schema_version": 1}
    body.update(
        {
            key: value
            for key, value in provisional.__dict__.items()
            if key != "canonical_body_sha256"
        }
    )
    return replace(
        provisional,
        canonical_body_sha256=hashlib.sha256(_canonical(body)).hexdigest(),
    )


class H1f:
    def __init__(self, candidate) -> None:
        self.result = _audit(candidate)
        self.calls: list[object] = []

    def fresh_audit(self, *, s3: object, request: object):
        self.calls.append((s3, request))
        return self.result


def _valid_activation_index() -> dict[str, object]:
    from glm52_enforcement.records import validate_record

    value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_activation_index",
        "account_id": OWNER,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": SHA,
        "current_activation_id": ACTIVATION_ID,
        "current_activation_ordinal": 1,
        "prior_activation_id": None,
        "prior_activation_terminal_v2_identity": None,
        "prior_h1g_drained_identity": None,
        "prior_spend_ledger_head_identity": None,
        "snapshot_cleanup_lineage_sha256": SHA,
        "revision": 1,
        "updated_at": "2026-07-28T12:00:00Z",
    }
    assert (
        validate_record("glm52_production_activation_index", value) == value
    )
    return value


def _valid_control(*, audit, revision: int, state: str) -> dict[str, object]:
    from glm52_enforcement.records import validate_record

    value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_control",
        "account_id": OWNER,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": SHA,
        "activation_id": ACTIVATION_ID,
        "activation_ordinal": 1,
        "rollover_identity_sha256": SHA,
        "active_epoch": audit.epoch,
        "active_execution_arn": audit.execution_arn,
        "active_state_machine_version_arn": (
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:glm52:1"
        ),
        "phase": "OPEN",
        "fence_head_body_sha256": audit.active_head_body_sha256,
        "fence_head_version_id": audit.active_head_version_id,
        "barrier_nonce_sha256": audit.barrier_nonce_sha256,
        "barrier_state": "OPEN",
        "decision_seal_state": "OPEN",
        "recovery_seal_state": "OPEN",
        "teardown_seal_state": "OPEN",
        "last_sky_post_generation": 1,
        "last_sky_post_action_key": ACTION_KEY,
        "last_sky_post_state": state,
        "numeric_job_binding_state": "UNBOUND",
        "recovery_control_initial_body_sha256": SHA,
        "finalization_control_initial_body_sha256": SHA,
        "snapshot_cleanup_control_initial_body_sha256": SHA,
        "revision": revision,
        "updated_at": "2026-07-28T12:00:00Z",
    }
    assert validate_record("glm52_production_control", value) == value
    return value


def _valid_action(
    candidate,
    *,
    audit,
    state: str,
) -> dict[str, object]:
    from glm52_enforcement import records as contract

    value: dict[str, object] = {}
    for field in contract.RECORD_FIELDS["glm52_production_action"]:
        if field == "schema_version":
            value[field] = 1
        elif field == "record_type":
            value[field] = "glm52_production_action"
        elif field == "run_id":
            value[field] = RUN_ID
        elif field in contract._INT_FIELDS:
            value[field] = 1
        elif field in contract._TIMESTAMP_FIELDS:
            value[field] = "2026-07-28T12:00:00Z"
        elif field.endswith("_sha256"):
            value[field] = SHA
        elif field == "generation_text":
            value[field] = "00000001"
        else:
            value[field] = "exact"

    value.update(
        {
            "campaign_identity_sha256": SHA,
            "activation_id": ACTIVATION_ID,
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "action_kind": "S3_CREATE",
            "attempt": 1,
            "candidate_key": candidate.key,
            "candidate_file_sha256": candidate.file_sha256,
            "candidate_body_sha256": candidate.body_sha256,
            "request_body_sha256": candidate.file_sha256,
            "owner_epoch": audit.epoch,
            "owner_execution_arn": audit.execution_arn,
            "armed_by_epoch": audit.epoch,
            "armed_by_execution_arn": audit.execution_arn,
            "armed_by_state_machine_version_arn": (
                "arn:aws:states:us-west-2:246813579024:"
                "stateMachine:glm52:1"
            ),
            "barrier_nonce_sha256": audit.barrier_nonce_sha256,
            "state": state,
            "revision": 1 if state == "ARMED" else 2,
        }
    )
    rule = contract._STATE_RULES["glm52_production_action"][state]
    for field in rule["null"]:
        value[field] = None
    for field in rule["nonnull"]:
        if value[field] is None:
            value[field] = (
                SHA if field.endswith("_sha256") else "2026-07-28T12:00:00Z"
            )
    for field, expected in rule["exact"].items():
        value[field] = expected
    if state == "CONSUMED":
        value.update(
            {
                "authority_audit_body_sha256": audit.canonical_body_sha256,
                "authority_audit_closing_revision": audit.closing_revision,
                "authorized_transition_from_revision": audit.closing_revision,
                "authorized_transition_to_revision": (
                    audit.expected_authorized_revision
                ),
                "owner_invocation_nonce_sha256": "f" * 64,
            }
        )
    assert (
        contract.validate_record(
            "glm52_production_action",
            value,
            sk=ACTION_KEY,
        )
        == value
    )
    return value


class Actions:
    def __init__(self, candidate) -> None:
        from glm52_enforcement.dynamodb import (
            TransactionResolution,
            WriteOutcome,
        )

        self.calls: list[tuple[str, dict[str, object]]] = []
        audit = _audit(candidate)
        armed_control = _valid_control(
            audit=audit,
            revision=audit.closing_revision,
            state="ARMED",
        )
        armed_action = _valid_action(
            candidate,
            audit=audit,
            state="ARMED",
        )
        self.arm_result = TransactionResolution(
            WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
            (
                _valid_activation_index(),
                armed_control,
                armed_action,
            ),
            "arm-request",
            None,
            (),
        )
        control = _valid_control(
            audit=audit,
            revision=audit.expected_authorized_revision,
            state="CONSUMED",
        )
        action = _valid_action(
            candidate,
            audit=audit,
            state="CONSUMED",
        )
        self.consume_result = TransactionResolution(
            WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
            (_valid_activation_index(), control, action),
            "consume-request",
            None,
            (),
        )

    def arm(self, **kwargs: object):
        self.calls.append(("arm", dict(kwargs)))
        return self.arm_result

    def consume_audit(self, **kwargs: object):
        self.calls.append(("consume_audit", dict(kwargs)))
        return self.consume_result


class PublicationS3:
    def __init__(self, candidate) -> None:
        self.candidate = candidate
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.stored = False
        self.version_id = "created-version"
        self.etag = '"0123456789abcdef0123456789abcdef"'

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("put_object", dict(kwargs)))
        self.stored = True
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "put-request",
                "HTTPHeaders": {"date": format_datetime(NOW, usegmt=True)},
            },
            "VersionId": self.version_id,
            "ETag": self.etag,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.candidate.raw).digest()
            ).decode("ascii"),
            "RequestTimestamp": NOW,
            "ResponseTimestamp": NOW,
        }

    def list_object_versions(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("list_object_versions", dict(kwargs)))
        versions = []
        if self.stored:
            versions.append(
                {
                    "Key": self.candidate.key,
                    "VersionId": self.version_id,
                    "IsLatest": True,
                    "ETag": self.etag,
                    "Size": len(self.candidate.raw),
                    "LastModified": NOW,
                }
            )
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "list-request",
            },
            "Name": BUCKET,
            "Prefix": self.candidate.key,
            "KeyMarker": "",
            "VersionIdMarker": "",
            "IsTruncated": False,
            "Versions": versions,
            "DeleteMarkers": [],
            "CommonPrefixes": [],
        }

    def _transport(self) -> dict[str, object]:
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "read-request",
            },
            "VersionId": self.version_id,
            "ETag": self.etag,
            "ContentLength": len(self.candidate.raw),
            "LastModified": NOW,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.candidate.raw).digest()
            ).decode("ascii"),
            "ChecksumType": "FULL_OBJECT",
            "ContentType": "application/json",
            "Metadata": dict(self.candidate.metadata),
            "MissingMeta": 0,
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_object", dict(kwargs)))
        class Body:
            def read(inner_self) -> bytes:
                return self.candidate.raw

            def close(inner_self) -> None:
                return None

        return {**self._transport(), "Body": Body()}

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_object", dict(kwargs)))
        return self._transport()


def _request(candidate):
    from glm52_enforcement.s3_adapter import S3CreateActionRequest

    return S3CreateActionRequest(
        authority_domain="ACTIVATION",
        operation_kind="S3_CREATE",
        action_key=ACTION_KEY,
        candidate=candidate,
        activation_id=ACTIVATION_ID,
        generation=1,
    )


def test_action_consume_binds_audit_r_to_coherent_owner_readback_r_plus_one() -> None:
    from glm52_enforcement.s3_adapter import (
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    actions = Actions(candidate)
    from glm52_enforcement.records import validate_record

    arm_index, arm_control, arm_action = actions.arm_result.records
    consume_index, consume_control, consume_action = (
        actions.consume_result.records
    )
    assert "generation" not in arm_control
    assert arm_control["last_sky_post_generation"] == 1
    assert validate_record(
        "glm52_production_activation_index", dict(arm_index)
    ) == dict(arm_index)
    assert validate_record(
        "glm52_production_control", dict(arm_control)
    ) == dict(arm_control)
    assert validate_record(
        "glm52_production_action", dict(arm_action), sk=ACTION_KEY
    ) == dict(arm_action)
    assert validate_record(
        "glm52_production_activation_index", dict(consume_index)
    ) == dict(consume_index)
    assert validate_record(
        "glm52_production_control", dict(consume_control)
    ) == dict(consume_control)
    assert validate_record(
        "glm52_production_action", dict(consume_action), sk=ACTION_KEY
    ) == dict(consume_action)
    s3 = PublicationS3(candidate)
    result = conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=s3,
            actions=actions,
            h1f=H1f(candidate),
        ),
        request=_request(candidate),
    )
    assert result.closing_revision == 7
    assert result.authorized_revision == 8
    assert [name for name, _ in actions.calls] == ["arm", "consume_audit"]
    consume = actions.calls[1][1]
    assert consume["authority_audit_body_sha256"] == _audit(
        candidate
    ).canonical_body_sha256
    assert consume["closing_revision"] == 7
    assert consume["authorized_revision"] == 8


@pytest.mark.parametrize(
    "stage,record_type,field,value",
    [
        (
            "arm",
            "glm52_production_activation_index",
            "campaign_identity_sha256",
            "b" * 64,
        ),
        (
            "arm",
            "glm52_production_activation_index",
            "current_activation_id",
            "activation-foreign",
        ),
        (
            "consume",
            "glm52_production_activation_index",
            "current_activation_ordinal",
            2,
        ),
        ("arm", "glm52_production_control", "last_sky_post_generation", 2),
        ("arm", "glm52_production_control", "campaign_identity_sha256", "b" * 64),
        ("arm", "glm52_production_control", "active_epoch", 3),
        (
            "arm",
            "glm52_production_control",
            "active_execution_arn",
            "foreign-execution",
        ),
        (
            "arm",
            "glm52_production_control",
            "active_state_machine_version_arn",
            "foreign-version",
        ),
        ("arm", "glm52_production_action", "activation_ordinal", 2),
        ("arm", "glm52_production_action", "candidate_key", "foreign-key"),
        ("arm", "glm52_production_action", "candidate_file_sha256", "b" * 64),
        ("arm", "glm52_production_action", "candidate_body_sha256", "b" * 64),
        ("arm", "glm52_production_action", "request_body_sha256", "b" * 64),
        ("arm", "glm52_production_action", "action_kind", "CREATE_CHANGE_SET"),
        ("arm", "glm52_production_action", "owner_epoch", 3),
        ("arm", "glm52_production_action", "armed_by_epoch", 3),
        (
            "arm",
            "glm52_production_action",
            "owner_execution_arn",
            "foreign-execution",
        ),
        (
            "arm",
            "glm52_production_action",
            "armed_by_execution_arn",
            "foreign-execution",
        ),
        (
            "arm",
            "glm52_production_action",
            "armed_by_state_machine_version_arn",
            "foreign-version",
        ),
        ("arm", "glm52_production_action", "barrier_nonce_sha256", "c" * 64),
        (
            "consume",
            "glm52_production_control",
            "fence_head_version_id",
            "foreign-head-version",
        ),
        (
            "consume",
            "glm52_production_control",
            "fence_head_body_sha256",
            "c" * 64,
        ),
        (
            "consume",
            "glm52_production_control",
            "last_sky_post_generation",
            2,
        ),
        (
            "consume",
            "glm52_production_action",
            "candidate_key",
            "foreign-key",
        ),
        (
            "consume",
            "glm52_production_action",
            "candidate_file_sha256",
            "b" * 64,
        ),
        (
            "consume",
            "glm52_production_action",
            "candidate_body_sha256",
            "b" * 64,
        ),
        (
            "consume",
            "glm52_production_action",
            "request_body_sha256",
            "b" * 64,
        ),
        (
            "consume",
            "glm52_production_action",
            "action_kind",
            "CREATE_CHANGE_SET",
        ),
    ],
)
def test_schema_valid_task3_readback_mutants_cannot_authorize_candidate(
    stage: str,
    record_type: str,
    field: str,
    value: object,
) -> None:
    from glm52_enforcement.dynamodb import TransactionResolution
    from glm52_enforcement.records import validate_record
    from glm52_enforcement.s3_adapter import (
        S3CreateError,
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    actions = Actions(candidate)
    original = (
        actions.arm_result if stage == "arm" else actions.consume_result
    )
    records = [dict(record) for record in original.records]
    target_index = next(
        index
        for index, record in enumerate(records)
        if record.get("record_type") == record_type
    )
    records[target_index][field] = value
    assert validate_record(record_type, records[target_index]) == records[
        target_index
    ]
    mutated = TransactionResolution(
        original.outcome,
        tuple(records),
        original.request_id,
        original.error_code,
        original.cancellation_reasons,
    )
    if stage == "arm":
        actions.arm_result = mutated
    else:
        actions.consume_result = mutated

    s3 = PublicationS3(candidate)
    with pytest.raises(S3CreateError):
        conditional_create_immutable_json(
            services=S3PublicationServices(
                s3=s3,
                actions=actions,
                h1f=H1f(candidate),
            ),
            request=_request(candidate),
        )
    assert not [name for name, _ in s3.calls if name == "put_object"]


def test_public_effect_protocol_has_no_cached_audit_or_alternate_consume_bypass() -> None:
    from glm52_enforcement.s3_adapter import (
        S3CreateActionRequest,
        S3CreateError,
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    real_actions = Actions(candidate)

    class AlternateActions:
        arm = real_actions.arm
        consume = real_actions.consume_audit

    s3 = PublicationS3(candidate)
    with pytest.raises(S3CreateError):
        conditional_create_immutable_json(
            services=S3PublicationServices(
                s3=s3, actions=AlternateActions(), h1f=H1f(candidate)
            ),
            request=_request(candidate),
        )
    assert tuple(S3CreateActionRequest.__dataclass_fields__) == (
        "authority_domain",
        "operation_kind",
        "action_key",
        "candidate",
        "activation_id",
        "generation",
    )
    assert not [name for name, _ in s3.calls if name == "put_object"]


@pytest.mark.parametrize(
    "authority_domain",
    ["RECOVERY", "FINALIZATION", "SNAPSHOT_CLEANUP", "OTHER"],
)
def test_public_effect_rejects_relabelled_nonactivation_domain_before_services(
    authority_domain: str,
) -> None:
    from glm52_enforcement.s3_adapter import (
        S3CreateError,
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    actions = Actions(candidate)
    h1f = H1f(candidate)
    s3 = PublicationS3(candidate)
    with pytest.raises(S3CreateError):
        conditional_create_immutable_json(
            services=S3PublicationServices(
                s3=s3,
                actions=actions,
                h1f=h1f,
            ),
            request=replace(
                _request(candidate),
                authority_domain=authority_domain,
            ),
        )
    assert actions.calls == []
    assert h1f.calls == []
    assert s3.calls == []


def test_put_occurs_only_after_fresh_audit_and_coherent_nonce_owned_readback() -> None:
    from glm52_enforcement.s3_adapter import (
        S3CreateError,
        S3PublicationServices,
        conditional_create_immutable_json,
    )
    from glm52_enforcement.dynamodb import (
        TransactionResolution,
        WriteOutcome,
    )

    candidate = _candidate()
    actions = Actions(candidate)
    actions.consume_result = TransactionResolution(
        WriteOutcome.FOREIGN_NONCE,
        actions.consume_result.records,
        "consume-request",
        None,
        (),
    )
    s3 = PublicationS3(candidate)
    with pytest.raises(S3CreateError):
        conditional_create_immutable_json(
            services=S3PublicationServices(
                s3=s3, actions=actions, h1f=H1f(candidate)
            ),
            request=_request(candidate),
        )
    assert not [name for name, _ in s3.calls if name == "put_object"]


def test_put_request_is_one_conditional_single_part_exact_owner_checksum_create() -> None:
    from glm52_enforcement.s3_adapter import (
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    s3 = PublicationS3(candidate)
    conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=s3, actions=Actions(candidate), h1f=H1f(candidate)
        ),
        request=_request(candidate),
    )
    puts = [values for name, values in s3.calls if name == "put_object"]
    assert len(puts) == 1
    assert puts[0] == {
        "Bucket": BUCKET,
        "Key": candidate.key,
        "Body": candidate.raw,
        "IfNoneMatch": "*",
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": base64.b64encode(
            hashlib.sha256(candidate.raw).digest()
        ).decode("ascii"),
        "ContentType": "application/json",
        "Metadata": dict(candidate.metadata),
        "ExpectedBucketOwner": OWNER,
    }


def test_direct_200_still_requires_exact_all_version_reconciliation() -> None:
    from glm52_enforcement.s3_adapter import (
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    class VanishingS3(PublicationS3):
        def list_object_versions(self, **kwargs: object) -> dict[str, object]:
            self.stored = False
            return super().list_object_versions(**kwargs)

    candidate = _candidate()
    result = conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=VanishingS3(candidate),
            actions=Actions(candidate),
            h1f=H1f(candidate),
        ),
        request=_request(candidate),
    )
    assert result.outcome == "absent-after-reconciliation"
    assert result.object_identity is None
    assert result.provenance == "all-version-reconciliation"


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("409"),
        RuntimeError("412"),
        TimeoutError("timeout"),
        ConnectionError("connection loss"),
        None,
    ],
)
def test_409_412_timeout_connection_loss_and_malformed_response_never_retry(
    failure: BaseException | None,
) -> None:
    from glm52_enforcement.s3_adapter import (
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    class AmbiguousS3(PublicationS3):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("put_object", dict(kwargs)))
            self.stored = True
            if failure is None:
                return {"ResponseMetadata": {"HTTPStatusCode": "200"}}
            raise failure

    candidate = _candidate()
    s3 = AmbiguousS3(candidate)
    result = conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=s3, actions=Actions(candidate), h1f=H1f(candidate)
        ),
        request=_request(candidate),
    )
    assert result.outcome == "reconciled-existing"
    assert result.provenance == "all-version-reconciliation"
    assert [name for name, _ in s3.calls].count("put_object") == 1


def test_ambiguous_matching_sole_version_returns_reconciled_provenance_only() -> None:
    from glm52_enforcement.s3_adapter import (
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    class AmbiguousS3(PublicationS3):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("put_object", dict(kwargs)))
            self.stored = True
            raise TimeoutError("response lost")

    candidate = _candidate()
    result = conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=AmbiguousS3(candidate),
            actions=Actions(candidate),
            h1f=H1f(candidate),
        ),
        request=_request(candidate),
    )
    assert result.outcome == "reconciled-existing"
    assert result.object_identity is not None
    assert result.provenance == "all-version-reconciliation"


@pytest.mark.parametrize("case", ["zero", "history", "sibling", "delete", "mismatch"])
def test_ambiguous_zero_history_sibling_delete_marker_or_mismatch_fail_closed(
    case: str,
) -> None:
    from glm52_enforcement.s3_adapter import (
        S3CreateError,
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    class BrokenReadback(PublicationS3):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("put_object", dict(kwargs)))
            self.stored = case != "zero"
            raise TimeoutError("ambiguous")

        def list_object_versions(self, **kwargs: object) -> dict[str, object]:
            response = super().list_object_versions(**kwargs)
            versions = response["Versions"]
            assert isinstance(versions, list)
            if case == "history" and versions:
                versions.append({**versions[0], "VersionId": "history-version"})
            elif case == "sibling" and versions:
                versions.append(
                    {
                        **versions[0],
                        "Key": self.candidate.key + ".sibling",
                        "VersionId": "sibling-version",
                    }
                )
            elif case == "delete":
                response["Versions"] = []
                response["DeleteMarkers"] = [
                    {
                        "Key": self.candidate.key,
                        "VersionId": "delete-version",
                        "IsLatest": True,
                        "LastModified": NOW,
                    }
                ]
            elif case == "mismatch" and versions:
                self.candidate = type(self.candidate)(
                    **{
                        **self.candidate.__dict__,
                        "raw": self.candidate.raw.replace(b'"schema_version":2', b'"schema_version":3'),
                    }
                )
            return response

    candidate = _candidate()
    if case == "zero":
        result = conditional_create_immutable_json(
            services=S3PublicationServices(
                s3=BrokenReadback(candidate),
                actions=Actions(candidate),
                h1f=H1f(candidate),
            ),
            request=_request(candidate),
        )
        assert result.outcome == "absent-after-reconciliation"
        assert result.object_identity is None
    else:
        with pytest.raises(S3CreateError):
            conditional_create_immutable_json(
                services=S3PublicationServices(
                    s3=BrokenReadback(candidate),
                    actions=Actions(candidate),
                    h1f=H1f(candidate),
                ),
                request=_request(candidate),
            )


def test_copy_multipart_delete_tag_replication_and_lifecycle_are_never_called() -> None:
    from glm52_enforcement.s3_adapter import (
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    s3 = PublicationS3(candidate)
    forbidden = (
        "copy_object",
        "create_multipart_upload",
        "upload_part",
        "upload_part_copy",
        "complete_multipart_upload",
        "abort_multipart_upload",
        "delete_object",
        "delete_objects",
        "put_object_tagging",
        "put_bucket_replication",
        "put_bucket_lifecycle_configuration",
        "delete_bucket_lifecycle",
    )
    for name in forbidden:
        setattr(
            s3,
            name,
            lambda **_kwargs: pytest.fail("forbidden S3 API was called"),
        )
    conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=s3, actions=Actions(candidate), h1f=H1f(candidate)
        ),
        request=_request(candidate),
    )
    assert {name for name, _ in s3.calls}.isdisjoint(forbidden)


def test_raw_nonce_audit_or_receipt_is_never_returned_or_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from glm52_enforcement.s3_adapter import (
        ConditionalCreateResult,
        S3PublicationServices,
        conditional_create_immutable_json,
    )

    candidate = _candidate()
    result = conditional_create_immutable_json(
        services=S3PublicationServices(
            s3=PublicationS3(candidate),
            actions=Actions(candidate),
            h1f=H1f(candidate),
        ),
        request=_request(candidate),
    )
    assert tuple(ConditionalCreateResult.__dataclass_fields__) == (
        "outcome",
        "object_identity",
        "authority_audit_body_sha256",
        "closing_revision",
        "authorized_revision",
        "provenance",
        "direct_request_id",
        "direct_server_date",
        "direct_request_started_at",
        "direct_response_received_at",
        "direct_response_authenticated",
    )
    text = repr(result) + caplog.text
    assert "raw_nonce" not in text
    assert "receipt" not in text
    assert "H1fAuditResult" not in text
