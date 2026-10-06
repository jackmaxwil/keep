from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.records import RECORD_FIELDS, validate_record


SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
TS = "2026-07-29T12:00:00Z"
BUCKET = "keep-glm52-246813579024-us-west-2"


def _terminal_v2() -> dict[str, object]:
    record: dict[str, object] = {}
    arrays = {
        "final_ec2_states",
        "allocations",
        "worker_launch_evidence",
        "worker_launch_liabilities",
        "request_evidence",
    }
    nullable = {
        "handoff",
        "binding",
        "final_heartbeat_identity",
        "checkpoint_identity",
        "cache_identity",
        "training_identity",
        "evaluation_identity",
        "drain_identity",
        "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
    }
    for field in RECORD_FIELDS["glm52_production_terminal_v2"]:
        if field == "schema_version":
            record[field] = 2
        elif field == "record_type":
            record[field] = "glm52_production_terminal_v2"
        elif field == "account_id":
            record[field] = "246813579024"
        elif field == "region":
            record[field] = "us-west-2"
        elif field == "run_id":
            record[field] = "glm52-sky-20260724"
        elif field == "activation_id":
            record[field] = "activation-1"
        elif field in {"activation_ordinal", "generation"}:
            record[field] = 1
        elif field == "generation_text":
            record[field] = "00000001"
        elif field in arrays:
            record[field] = []
        elif field in nullable:
            record[field] = None
        elif field.endswith("_array_sha256"):
            record[field] = canonical_sha256([])
        elif field.endswith("_sha256"):
            record[field] = SHA_A
        elif field in {"spend_ledger_head_identity", "terminal_observation_window"}:
            record[field] = {"identity": SHA_A}
        elif field == "remaining_approved_gpu_seconds":
            record[field] = 0
        elif field == "remaining_approved_gpu_usd":
            record[field] = "0.00"
        elif field == "request_cardinality":
            record[field] = "NOT_APPLICABLE"
        elif field == "worker_cardinality":
            record[field] = "ZERO"
        elif field == "outcome":
            record[field] = (
                "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH"
            )
        elif field == "operator_disposition_required":
            record[field] = True
        elif field == "created_at":
            record[field] = TS
        else:
            record[field] = "x"
    record["canonical_body_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in record.items()
            if key != "canonical_body_sha256"
        }
    )
    return validate_record("glm52_production_terminal_v2", record)


def test_versioned_writer_control_binds_only_a_proven_s3_result() -> None:
    from glm52_enforcement.task12_writers import (
        RetainedWriteResult,
        build_retained_writer_candidate,
        build_versioned_writer_control,
    )

    candidate = build_retained_writer_candidate(
        writer_kind="TerminalV2",
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="RECOVERY",
        record=_terminal_v2(),
    )
    result_body = {
        "writer_kind": candidate.writer_kind,
        "outcome": "created-authenticated",
        "coordinate": candidate.coordinate,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "object_version_id": "terminal-version-1",
        "response_request_ids": ("request-1",),
        "response_authenticated": True,
    }
    result = RetainedWriteResult(
        **result_body,
        canonical_identity_sha256=canonical_sha256(result_body),
    )

    control = build_versioned_writer_control(
        candidate=candidate,
        result=result,
        published_at=TS,
    )

    assert control["object_version_id"] == "terminal-version-1"
    assert control["object_key"] == candidate.coordinate.split("/", 3)[3]
    assert control["canonical_body_sha256"] == canonical_sha256(
        {key: value for key, value in control.items() if key != "canonical_body_sha256"}
    )


def _operator_disposition() -> dict[str, object]:
    record: dict[str, object] = {}
    for field in RECORD_FIELDS["glm52_production_operator_disposition"]:
        if field == "schema_version":
            record[field] = 1
        elif field == "record_type":
            record[field] = "glm52_production_operator_disposition"
        elif field == "account_id":
            record[field] = "246813579024"
        elif field == "region":
            record[field] = "us-west-2"
        elif field == "run_id":
            record[field] = "glm52-sky-20260724"
        elif field == "activation_id":
            record[field] = "activation-1"
        elif field == "activation_ordinal":
            record[field] = 1
        elif field.endswith("_sha256"):
            record[field] = SHA_A
        elif field in {"approval_ingested_at", "created_at"}:
            record[field] = TS
        elif field == "terminal_v2_identity":
            record[field] = {
                "key": (
                    "campaigns/glm52-sky-20260724/submissions/production/"
                    "generations/00000001/terminal/"
                    "PRODUCTION_TERMINAL_V2.json"
                ),
                "version_id": "terminal-v2-version",
                "body_sha256": SHA_A,
            }
        elif field == "terminal_v2_outcome":
            record[field] = (
                "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH"
            )
        elif field == "final_worker_cardinality":
            record[field] = "ZERO"
        elif field == "decision":
            record[field] = "ALLOW_NEW_ACTIVATION"
        else:
            record[field] = "x"
    record["canonical_body_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in record.items()
            if key != "canonical_body_sha256"
        }
    )
    return validate_record(
        "glm52_production_operator_disposition",
        record,
    )


def _simple_record(record_type: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": record_type,
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": SHA_A,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "writer_function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12-writer:7"
        ),
        "writer_dispatch_identity_sha256": SHA_A,
        "writer_invocation_nonce_sha256": SHA_B,
        "published_at": TS,
    }
    return {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }


def _identity(**fields: object) -> dict[str, object]:
    return {
        **fields,
        "canonical_identity_sha256": canonical_sha256(fields),
    }


def _h1g_drained() -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_production_h1g_drained",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": SHA_A,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "terminal_v2_identity": _identity(
            generation=1,
            generation_text="00000001",
            bucket="keep-glm52-models-246813579024-us-west-2",
            key=(
                "campaigns/"
                "glm52-sky-20260724/submissions/production/generations/"
                "00000001/terminal/PRODUCTION_TERMINAL_V2.json"
            ),
            version_id="terminal-v2-version-1",
            file_sha256=SHA_A,
            body_sha256=SHA_B,
        ),
        "support_plane_finalized_identity": _identity(
            finalization_control_identity_sha256=SHA_A,
            support_plane_finalized_identity_sha256=SHA_B,
        ),
        "support_stack_deletion_identity": _identity(
            stack_id=(
                "arn:aws:cloudformation:us-west-2:246813579024:stack/"
                "keep-glm52-h1g-support/"
                "12345678-1234-1234-1234-123456789012"
            ),
            observation="ABSENT_VALIDATION_ERROR",
            request_id="support-stack-absence-request-1",
        ),
        "orphan_audit_identity": SHA_A,
        "kms_grant_baseline_equality_identity": _identity(
            baseline_identity_sha256=SHA_A,
            final_grants_identity_sha256=SHA_A,
            final_list_grants_identity_sha256=SHA_B,
        ),
        "snapshot_cleanup_control_identity": SHA_A,
        "snapshot_cleanup_schedule_identity": _identity(
            schedule_arn=(
                "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
                "keep-glm52-h1g-snapshot-cleanup-activation-1"
            ),
            schedule_input_sha256=canonical_sha256({}),
        ),
        "worker_launch_liabilities": [],
        "spend_ledger_head_identity": _identity(
            bucket="keep-glm52-models-246813579024-us-west-2",
            key=(
                "campaigns/glm52-sky-20260724/runtime/"
                "GPU_SPEND_LEDGER.jsonl"
            ),
            version_id="spend-ledger-version-1",
            file_sha256=SHA_A,
            body_sha256=SHA_B,
            head_record_sha256=SHA_A,
        ),
        "retained_resource_inventory_identity": _identity(
            retained_resources=[
                {
                    "resource_type": "LEDGER",
                    "resource_id": "glm52-ledger",
                    "cost_class": "DDB_RETAINED",
                },
                {
                    "resource_type": "KMS_KEY",
                    "resource_id": "key-1",
                    "cost_class": "KMS_RETAINED",
                },
                {
                    "resource_type": "EVIDENCE_BUCKET",
                    "resource_id": "evidence-bucket",
                    "cost_class": "S3_RETAINED",
                },
                {
                    "resource_type": "PRODUCTION_FENCE_STACK",
                    "resource_id": "keep-glm52-production-fence",
                    "cost_class": "CFN_RETAINED",
                },
                {
                    "resource_type": "LIFECYCLE_RESOURCE",
                    "resource_id": "cleanup-schedule",
                    "cost_class": "EVENTBRIDGE_RETAINED",
                },
                {
                    "resource_type": "SOURCE_PUBLISHER_IDENTITY",
                    "resource_id": "source-publisher-role",
                    "cost_class": "IAM_RETAINED",
                },
            ],
            retained_cost_classes=[
                "DDB_RETAINED",
                "KMS_RETAINED",
                "S3_RETAINED",
                "CFN_RETAINED",
                "EVENTBRIDGE_RETAINED",
                "IAM_RETAINED",
            ],
            price_card_identity_sha256=SHA_A,
            retained_terms=[
                {
                    "term": term,
                    "unit": "unit",
                    "quantity": "1.000000",
                    "unit_price_usd": "0.010000",
                    "estimated_usd": "0.01",
                }
                for term in (
                    "retained_s3",
                    "retained_ledger",
                    "retained_kms",
                    "liability_watcher",
                    "snapshot_cleanup",
                    "snapshot_retention",
                )
            ],
            controller_quiescence_identity_sha256=SHA_A,
            marker_last_prerequisite_identity_sha256=SHA_A,
        ),
        "writer_function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-task12-writer:7"
        ),
        "writer_dispatch_identity_sha256": SHA_A,
        "writer_invocation_nonce_sha256": SHA_B,
        "published_at": TS,
    }
    inventory = dict(body["retained_resource_inventory_identity"])
    inventory["marker_last_prerequisite_identity_sha256"] = canonical_sha256(
        {
            "terminal_v2_identity_sha256": body["terminal_v2_identity"][
                "body_sha256"
            ],
            "finalization_identity_sha256": body[
                "support_plane_finalized_identity"
            ]["finalization_control_identity_sha256"],
            "snapshot_cleanup_control_identity_sha256": body[
                "snapshot_cleanup_control_identity"
            ],
            "controller_quiesced_identity_sha256": inventory[
                "controller_quiescence_identity_sha256"
            ],
            "spend_ledger_head_identity_sha256": body[
                "spend_ledger_head_identity"
            ]["head_record_sha256"],
            "orphan_audit_identity_sha256": body["orphan_audit_identity"],
            "liability_state": "SETTLED",
            "liability_identity_sha256": canonical_sha256([]),
            "marker_write_order": "LAST_CONDITIONAL_CREATE",
        }
    )
    inventory_body = dict(inventory)
    inventory_body.pop("canonical_identity_sha256")
    inventory["canonical_identity_sha256"] = canonical_sha256(inventory_body)
    body["retained_resource_inventory_identity"] = inventory
    return {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }


def test_h1g_drained_retains_the_closed_marker_last_prerequisite_envelope() -> None:
    expected_fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "campaign_identity_sha256",
        "activation_id",
        "activation_ordinal",
        "terminal_v2_identity",
        "support_plane_finalized_identity",
        "support_stack_deletion_identity",
        "orphan_audit_identity",
        "kms_grant_baseline_equality_identity",
        "snapshot_cleanup_control_identity",
        "snapshot_cleanup_schedule_identity",
        "worker_launch_liabilities",
        "spend_ledger_head_identity",
        "retained_resource_inventory_identity",
        "writer_function_version_arn",
        "writer_dispatch_identity_sha256",
        "writer_invocation_nonce_sha256",
        "published_at",
        "canonical_body_sha256",
    }
    record = _h1g_drained()

    assert set(RECORD_FIELDS["glm52_production_h1g_drained"]) == expected_fields
    assert validate_record("glm52_production_h1g_drained", record) == record


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["terminal_v2_identity"].update(version_id="latest"),
        lambda value: value["spend_ledger_head_identity"].update(
            bucket="foreign-bucket"
        ),
        lambda value: value["support_stack_deletion_identity"].update(
            observation="STACK_PRESENT"
        ),
        lambda value: value["kms_grant_baseline_equality_identity"].update(
            final_grants_identity_sha256=SHA_B
        ),
        lambda value: value["snapshot_cleanup_schedule_identity"].pop(
            "schedule_input_sha256"
        ),
        lambda value: value["retained_resource_inventory_identity"].update(
            marker_last_prerequisite_identity_sha256=SHA_B
        ),
    ),
)
def test_h1g_drained_rejects_forked_nested_prerequisite_identity(
    mutation: object,
) -> None:
    record = deepcopy(_h1g_drained())
    mutation(record)
    record["canonical_body_sha256"] = canonical_sha256(
        {
            field: value
            for field, value in record.items()
            if field != "canonical_body_sha256"
        }
    )

    with pytest.raises(ValueError):
        validate_record("glm52_production_h1g_drained", record)


@pytest.mark.parametrize(
    "record_type",
    (
        "glm52_production_support_plane_finalized",
        "glm52_production_h1g_drained",
    ),
)
def test_versioned_finalization_writer_records_have_closed_registered_schemas(
    record_type: str,
) -> None:
    from glm52_enforcement.records import RecordValidationError

    record = (
        _h1g_drained()
        if record_type == "glm52_production_h1g_drained"
        else _simple_record(record_type)
    )
    assert validate_record(record_type, record) == record

    for mutation in ("missing", "unknown", "hash", "record_type"):
        foreign = dict(record)
        if mutation == "missing":
            foreign.pop("published_at")
        elif mutation == "unknown":
            foreign["future_payload"] = {}
        elif mutation == "hash":
            foreign["canonical_body_sha256"] = "0" * 64
        else:
            foreign["record_type"] = "glm52_production_unknown"
        with pytest.raises(RecordValidationError):
            validate_record(record_type, foreign)


def _recovery_handoff() -> dict[str, object]:
    body: dict[str, object] = {
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": SHA_A,
        "generation": 1,
        "generation_text": "00000001",
        "submit_attempt_id": "submit-attempt-1",
        "decision_key": "decision-key",
        "decision_version_id": "decision-version",
        "decision_file_sha256": SHA_A,
        "decision_body_sha256": SHA_B,
        "sky_post_action_key": (
            "ACTIVATION#activation-1#ACTION#00000001#SKY_POST#00000001"
        ),
        "sky_post_consumed_at": TS,
        "sky_post_outcome_class": "PROVED_NOT_SENT_OWNER_DIED",
        "expected_sky_job_name": "glm52-sky-20260724",
        "task_yaml_sha256": SHA_A,
        "request_body_sha256": SHA_B,
        "api_server_identity_sha256": SHA_A,
        "sky_request_id": None,
        "post_started_at": TS,
        "post_completed_or_lost_at": TS,
        "binding_state": "reconcile-required",
    }
    return {**body, "handoff_body_sha256": canonical_sha256(body)}


def _authorities(candidate: object) -> tuple[object, object]:
    from glm52_enforcement.task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
    )

    action_key = (
        "ACTIVATION#activation-1#OPERATOR_DISPOSITION"
        if candidate.authority_domain == "OPERATOR_DISPOSITION"
        else (
            "ACTIVATION#activation-1#"
            + candidate.authority_domain
            + "_ACTION#"
            + candidate.action_kind
            + "#00000001"
        )
    )
    action = RetainedWriterActionAuthority(
        authority_domain=candidate.authority_domain,
        action_kind=candidate.action_kind,
        action_key=action_key,
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        action_identity_sha256=SHA_A,
        owner_invocation_nonce_sha256=SHA_B,
        state="CONSUMED",
        authorized_revision=8,
    )
    audit_body = {
        "authority_domain": candidate.authority_domain,
        "action_kind": candidate.action_kind,
        "action_key": action.action_key,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "action_identity_sha256": action.action_identity_sha256,
        "audit_kind": (
            "WRITTEN_APPROVAL_VERSION"
            if candidate.writer_kind == "OperatorDisposition"
            else "H1F_GENESIS_TO_ZERO_CHILD"
        ),
        "closing_revision": 7,
        "authorized_revision": 8,
        "current_revision": 8,
        "observed_at": TS,
    }
    audit = RetainedWriterAuditAuthority(
        **audit_body,
        canonical_identity_sha256=canonical_sha256(audit_body),
    )
    return action, audit


@pytest.mark.parametrize(
    ("writer_kind", "record", "domain", "action_kind", "coordinate"),
    [
        (
            "TerminalV2",
            _terminal_v2,
            "RECOVERY",
            "TERMINAL_V2",
            (
                "s3://"
                + BUCKET
                + "/campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/terminal/"
                "PRODUCTION_TERMINAL_V2.json"
            ),
        ),
        (
            "SupportPlaneFinalized",
            lambda: _simple_record(
                "glm52_production_support_plane_finalized"
            ),
            "FINALIZATION",
            "SUPPORT_PLANE_FINALIZED",
            (
                "s3://"
                + BUCKET
                + "/campaigns/glm52-sky-20260724/submissions/production/"
                "activations/activation-1/finalization/"
                "SUPPORT_PLANE_FINALIZED.json"
            ),
        ),
        (
            "H1GDrained",
            _h1g_drained,
            "FINALIZATION",
            "H1G_DRAINED",
            (
                "s3://"
                + BUCKET
                + "/campaigns/glm52-sky-20260724/submissions/production/"
                "activations/activation-1/finalization/H1G_DRAINED.json"
            ),
        ),
        (
            "RecoveryHandoff",
            _recovery_handoff,
            "RECOVERY",
            "RECOVERY_HANDOFF",
            (
                "s3://"
                + BUCKET
                + "/campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/handoff/SKY_POST_HANDOFF.json"
            ),
        ),
        (
            "OperatorDisposition",
            _operator_disposition,
            "OPERATOR_DISPOSITION",
            "OPERATOR_DISPOSITION",
            (
                "dynamodb://RUN#glm52-sky-20260724/"
                "ACTIVATION#activation-1#OPERATOR_DISPOSITION"
            ),
        ),
    ],
)
def test_fixed_writer_candidates_freeze_domain_coordinate_and_identity(
    writer_kind: str,
    record: object,
    domain: str,
    action_kind: str,
    coordinate: str,
) -> None:
    from glm52_enforcement.task12_writers import (
        build_retained_writer_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind=writer_kind,
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain=domain,
        record=record(),
    )

    assert candidate.authority_domain == domain
    assert candidate.action_kind == action_kind
    assert candidate.coordinate == coordinate
    assert candidate.raw == canonical_json_bytes(record()) + b"\n"
    assert candidate.file_sha256 == hashlib.sha256(candidate.raw).hexdigest()
    assert candidate.candidate_identity_sha256 == canonical_sha256(
        {
            "writer_kind": writer_kind,
            "authority_domain": domain,
            "action_kind": action_kind,
            "coordinate": coordinate,
            "file_sha256": candidate.file_sha256,
            "body_sha256": candidate.body_sha256,
        }
    )


class _Boundary:
    def __init__(self, create: object, reconcile: object = None) -> None:
        self.create_response = create
        self.reconciliation = reconcile
        self.create_calls: list[object] = []
        self.reconcile_calls: list[object] = []

    def conditional_create(
        self,
        *,
        candidate: object,
        action: object,
        audit: object,
    ) -> object:
        self.create_calls.append((candidate, action, audit))
        return self.create_response

    def reconcile_exact(self, *, candidate: object) -> object:
        self.reconcile_calls.append(candidate)
        return self.reconciliation


def _direct_response(candidate: object) -> object:
    from glm52_enforcement.task12_writers import (
        ConditionalCreateResponse,
    )

    return ConditionalCreateResponse(
        classification="CREATED",
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        request_id="request-created-1",
        response_identity_sha256=SHA_A,
        version_id="opaque-version-created-1",
        authenticated=True,
    )


def test_writer_issues_one_conditional_create_and_accepts_authenticated_result() -> None:
    from glm52_enforcement.task12_writers import (
        Task12WriterServices,
        build_retained_writer_candidate,
        write_retained_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind="SupportPlaneFinalized",
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="FINALIZATION",
        record=_simple_record("glm52_production_support_plane_finalized"),
    )
    action, audit = _authorities(candidate)
    boundary = _Boundary(_direct_response(candidate))

    result = write_retained_candidate(
        services=Task12WriterServices(boundary=boundary),
        candidate=candidate,
        action=action,
        audit=audit,
    )

    assert result.outcome == "created-authenticated"
    assert result.object_version_id == "opaque-version-created-1"
    assert result.response_authenticated is True
    assert len(boundary.create_calls) == 1
    assert boundary.reconcile_calls == []


def test_duplicate_or_ambiguous_write_is_reconciliation_only_exact_adoption() -> None:
    from glm52_enforcement.task12_writers import (
        ConditionalCreateResponse,
        ExactCandidateReconciliation,
        Task12WriterServices,
        build_retained_writer_candidate,
        write_retained_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind="H1GDrained",
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="FINALIZATION",
        record=_h1g_drained(),
    )
    action, audit = _authorities(candidate)
    boundary = _Boundary(
        ConditionalCreateResponse(
            classification="PRECONDITION_FAILED",
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            request_id="request-precondition-1",
            response_identity_sha256=SHA_A,
            version_id=None,
            authenticated=True,
        ),
        ExactCandidateReconciliation(
            state="SOLE_EXACT_CANDIDATE",
            candidate_identity_sha256=candidate.candidate_identity_sha256,
            file_sha256=candidate.file_sha256,
            raw=candidate.raw,
            version_id="adopted-version-1",
            request_ids=("list-1", "get-1"),
            authenticated=True,
        ),
    )

    result = write_retained_candidate(
        services=Task12WriterServices(boundary=boundary),
        candidate=candidate,
        action=action,
        audit=audit,
    )

    assert result.outcome == "reconciled-exact-existing"
    assert result.object_version_id == "adopted-version-1"
    assert len(boundary.create_calls) == 1
    assert boundary.reconcile_calls == [candidate]


@pytest.mark.parametrize(
    "mutation",
    [
        "generic-activation",
        "generic-s3-create",
        "stale-audit",
        "foreign-action",
        "coordinate-drift",
        "reconcile-different-bytes",
    ],
)
def test_writer_permanent_mutants_fail_before_adoption(mutation: str) -> None:
    from glm52_enforcement.task12_writers import (
        ConditionalCreateResponse,
        ExactCandidateReconciliation,
        Task12WriterError,
        Task12WriterServices,
        build_retained_writer_candidate,
        write_retained_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind="H1GDrained",
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="FINALIZATION",
        record=_h1g_drained(),
    )
    action, audit = _authorities(candidate)
    reconciliation = ExactCandidateReconciliation(
        state="SOLE_EXACT_CANDIDATE",
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        file_sha256=candidate.file_sha256,
        raw=candidate.raw,
        version_id="adopted-version-1",
        request_ids=("list-1", "get-1"),
        authenticated=True,
    )
    response = ConditionalCreateResponse(
        classification="PRECONDITION_FAILED",
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        request_id="request-precondition-1",
        response_identity_sha256=SHA_A,
        version_id=None,
        authenticated=True,
    )
    if mutation == "generic-activation":
        action = replace(action, authority_domain="ACTIVATION")
    elif mutation == "generic-s3-create":
        action = replace(action, action_kind="S3_CREATE")
    elif mutation == "stale-audit":
        audit = replace(audit, current_revision=9)
    elif mutation == "foreign-action":
        audit = replace(audit, action_identity_sha256=SHA_B)
    elif mutation == "coordinate-drift":
        candidate = replace(
            candidate,
            coordinate=candidate.coordinate + ".foreign",
        )
    elif mutation == "reconcile-different-bytes":
        reconciliation = replace(
            reconciliation,
            raw=candidate.raw + b" ",
        )
    boundary = _Boundary(response, reconciliation)

    with pytest.raises(Task12WriterError):
        write_retained_candidate(
            services=Task12WriterServices(boundary=boundary),
            candidate=candidate,
            action=action,
            audit=audit,
        )
    assert len(boundary.create_calls) <= 1


def test_writer_rejects_untyped_action_and_audit() -> None:
    from glm52_enforcement.task12_writers import (
        Task12WriterError,
        Task12WriterServices,
        build_retained_writer_candidate,
        write_retained_candidate,
    )

    candidate = build_retained_writer_candidate(
        writer_kind="RecoveryHandoff",
        campaign_bucket=BUCKET,
        activation_id="activation-1",
        generation=1,
        authority_domain="RECOVERY",
        record=_recovery_handoff(),
    )
    boundary = _Boundary(_direct_response(candidate))
    with pytest.raises(Task12WriterError, match="typed"):
        write_retained_candidate(
            services=Task12WriterServices(boundary=boundary),
            candidate=candidate,
            action=asdict(_authorities(candidate)[0]),
            audit=asdict(_authorities(candidate)[1]),
        )
    assert boundary.create_calls == []
