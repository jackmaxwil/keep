"""Closed Task 12 operation contracts shared by publication and Lambda dispatch.

The Step Functions definitions publish one authenticated input row per operation.
This module is the single source of truth for the exact payload schema selected by
each operation kind.  Callers must not treat ``operation_kind`` as a label on a
generic handler payload: changing the operation changes the accepted input schema,
result type, and action selected by the runtime adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from collections.abc import Mapping as AbcMapping
from enum import Enum
import types
import hashlib
import json
import re
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

from .records import ledger_sk


_ACTIVATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARN_RE = re.compile(r"^arn:aws:[a-z0-9-]+:[a-z0-9-]*:[0-9]{12}:.+$")
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"


class Task12OperationContractError(ValueError):
    """Raised when a Task 12 operation row violates its closed schema."""


@dataclass(frozen=True)
class OperationSpec:
    operation_kind: str
    handler_kind: str
    record_type: str
    payload_fields: frozenset[str]
    result_kind: str
    action_kind: str
    request_fields: frozenset[str]
    builder_kind: str
    allowed_read_apis: tuple[str, ...]
    coordinate_fields: frozenset[str]


@dataclass(frozen=True)
class OperationExecutionResult:
    """Typed semantic outcome selected by one exact operation contract."""

    result_kind: str
    operation_kind: str
    action_kind: str
    behavior_kind: str
    predecessor_operation_kind: str | None
    predecessor_result_identity_sha256: str | None
    operation_payload_identity_sha256: str
    semantic_result: Mapping[str, Any]
    canonical_body_sha256: str


_REQUEST_FIELDS_BY_ACTION: Mapping[str, tuple[str, ...]] = {
    "reconcile_retained_lifecycle_trigger": (
        "correlation_identity_sha256",
        "runtime_scan",
    ),
    "commit_recovery_seal": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "stop_or_observe_support_execution": (
        "execution_arn",
        "expected_state_machine_version_arn",
        "stop_if_running",
    ),
    "prove_support_execution_terminal_or_sealed_start_incident": (
        "first_runtime_scan",
        "second_runtime_scan",
        "minimum_quiet_seconds",
    ),
    "classify_owner_dead_sky_action": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "create_recovery_handoff_if_required": (
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body",
        "approved_task11_workflow_version_arn",
        "task11_execution_arn",
    ),
    "commit_teardown_seal": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "prove_zero_activation_work": (
        "first_runtime_scan",
        "second_runtime_scan",
        "minimum_quiet_seconds",
    ),
    "commit_teardown_sealed": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "correlate_requests_and_jobs": (
        "partition_key",
        "sort_key",
        "expected_record_identity_sha256",
    ),
    "reconcile_request_and_job_cancellation": ("command", "observation"),
    "quiesce_controller_if_required": (
        "stop",
        "first_runtime_scan",
        "second_runtime_scan",
        "minimum_quiet_seconds",
    ),
    "reconcile_worker_launches_and_transfer_liabilities": (
        "liability_transfer",
    ),
    "reconcile_workers_and_allocations": ("candidate", "action", "audit"),
    "create_or_reconcile_terminal_v2": (
        "writer_kind",
        "authority_domain",
        "record",
        "action",
        "audit",
    ),
    "commit_recovery_complete": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "publish_support_plane_finalized": (
        "writer_kind",
        "authority_domain",
        "record",
        "action",
        "audit",
    ),
    "reconcile_delete_until_absent": ("stack_id",),
    "discover_exact_recovery_snapshot": (
        "observations",
        "expected_source_volume_id",
        "expected_kms_key_arn",
        "expected_snapshot_tags_sha256",
    ),
    "audit_support_orphans": (
        "expected_retained",
        "inventory",
        "retained_grant_baseline",
        "pre_cleanup_grants",
        "direct_grants",
        "service_grants",
        "settling_deadline",
    ),
    "arm_snapshot_cleanup_control_and_schedule": (
        "plan",
        "capture",
        "schedule",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "invoke_h1g_drained_writer": (
        "writer_kind",
        "authority_domain",
        "record",
        "action",
        "audit",
    ),
    "validate_snapshot_cleanup_schedule_and_deadline": ("capture", "schedule"),
    "acquire_or_take_over_snapshot_cleanup_owner": (
        "plan",
        "observed_at",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "arm_snapshot_delete_action": (
        "plan",
        "partition_key",
        "sort_key",
        "expected_record_identity_sha256",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "audit_snapshot_delete_authority": (
        "partition_key",
        "sort_key",
        "expected_record_identity_sha256",
    ),
    "atomic_consume_and_stage_snapshot_delete": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "send_same_snapshot_delete_id_or_read_back": ("snapshot_id",),
    "reconcile_snapshot_describe": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "record_snapshot_cleanup_terminal_evidence": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "close_ambiguous_snapshot_delete_attempt": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
    "arm_next_same_id_snapshot_delete_attempt": (
        "plan",
        "domain",
        "operation_identity_sha256",
        "owner_nonce_capsule",
    ),
}

_INTEGER_REQUEST_FIELDS = frozenset(
    {"minimum_quiet_seconds", "allocation_ordinal"}
)
_BOOLEAN_REQUEST_FIELDS = frozenset({"stop_if_running"})
_ARRAY_REQUEST_FIELDS = frozenset(
    {
        "observations",
        "request_pages",
        "expected_retained",
        "retained_grant_baseline",
        "direct_grants",
        "service_grants",
    }
)
_TEXT_REQUEST_FIELDS = frozenset(
    {
        "activation_id",
        "authority_domain",
        "bucket",
        "domain",
        "expected_action",
        "expected_kms_key_arn",
        "expected_snapshot_tags_sha256",
        "expected_source_volume_id",
        "expected_state_machine_version_arn",
        "execution_arn",
        "file_sha256",
        "key",
        "observed_at",
        "operation_identity_sha256",
        "owner_execution_status",
        "partition_key",
        "settling_deadline",
        "snapshot_id",
        "sort_key",
        "stack_id",
        "task11_execution_arn",
        "terminal_v2_identity_sha256",
        "version_id",
        "approved_task11_workflow_version_arn",
        "writer_kind",
    }
)


def _spec(
    operation_kind: str,
    handler_kind: str,
    _legacy_fields: tuple[str, ...],
    action_kind: str,
) -> OperationSpec:
    operation_slug = operation_kind.lower()
    if action_kind == "stop_or_observe_support_execution":
        allowed_reads = ("states:DescribeExecution",)
    elif action_kind == "classify_owner_dead_sky_action":
        allowed_reads = (
            "dynamodb:GetItem",
            "dynamodb:TransactGetItems",
            "dynamodb:TransactWriteItems",
        )
    elif action_kind == "reconcile_delete_until_absent":
        allowed_reads = ("cloudformation:DescribeStacks",)
    elif action_kind in {
        "discover_exact_recovery_snapshot",
        "reconcile_snapshot_describe",
    }:
        allowed_reads = (
            "dynamodb:GetItem",
            "ec2:DescribeSnapshots",
        )
    elif action_kind == "audit_support_orphans":
        allowed_reads = (
            "cloudformation:DescribeStacks",
            "cloudformation:ListStackResources",
            "cloudwatch:DescribeAlarms",
            "dynamodb:DescribeTable",
            "dynamodb:GetItem",
            "ec2:DescribeAddresses",
            "ec2:DescribeInstances",
            "ec2:DescribeNatGateways",
            "ec2:DescribeNetworkInterfaces",
            "ec2:DescribeRouteTables",
            "ec2:DescribeSecurityGroups",
            "ec2:DescribeSnapshots",
            "ec2:DescribeSubnets",
            "ec2:DescribeVolumes",
            "ec2:DescribeVpcEndpoints",
            "events:ListRules",
            "iam:GetRole",
            "iam:ListInstanceProfiles",
            "iam:ListPolicies",
            "iam:ListRoles",
            "kms:DescribeKey",
            "kms:ListGrants",
            "lambda:GetPolicy",
            "lambda:ListFunctions",
            "lambda:ListVersionsByFunction",
            "logs:DescribeLogGroups",
            "s3:GetBucketVersioning",
            "s3:GetObjectVersion",
            "s3:ListAllMyBuckets",
            "scheduler:GetSchedule",
            "scheduler:ListSchedules",
            "secretsmanager:ListSecrets",
            "sqs:ListQueues",
            "states:DescribeStateMachine",
            "states:ListExecutions",
            "states:ListStateMachineVersions",
            "states:ListStateMachines",
        )
    else:
        allowed_reads = ("dynamodb:GetItem",)
    allowed_reads = tuple(sorted(set(allowed_reads) | {"dynamodb:GetItem"}))
    coordinate_fields = {
        "ledger_partition_key",
        "sources",
    }
    if any(api.startswith("ec2:") for api in allowed_reads):
        coordinate_fields.add("region")
    if "kms:ListGrants" in allowed_reads:
        coordinate_fields.add("kms_key_id")
    return OperationSpec(
        operation_kind=operation_kind,
        handler_kind=handler_kind,
        record_type=f"glm52_task12_{operation_slug}_input_v1",
        payload_fields=frozenset({action_kind + "_request"}),
        result_kind=f"glm52_task12_{operation_slug}_result_v1",
        action_kind=action_kind,
        request_fields=frozenset(_REQUEST_FIELDS_BY_ACTION[action_kind]),
        builder_kind=action_kind.upper() + "_LIVE_BUILDER_V1",
        allowed_read_apis=allowed_reads,
        coordinate_fields=frozenset(coordinate_fields),
    )


_SPECS = (
    _spec(
        "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
        "RETAINED_EXECUTION_OBSERVER",
        ("correlation_identity_sha256", "runtime_scan"),
        "reconcile_retained_lifecycle_trigger",
    ),
    _spec(
        "RETAINED_ACQUIRE_RECOVERY_SEALING",
        "RETAINED_EXECUTION_OBSERVER",
        ("domain", "operation_identity_sha256", "plan", "owner_nonce_capsule"),
        "commit_recovery_seal",
    ),
    _spec(
        "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION",
        "RETAINED_EXECUTION_OBSERVER",
        ("execution_arn", "execution_observation", "stop_command"),
        "stop_or_observe_support_execution",
    ),
    _spec(
        "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT",
        "RETAINED_EXECUTION_OBSERVER",
        ("first_runtime_scan", "minimum_quiet_seconds", "second_runtime_scan"),
        "prove_support_execution_terminal_or_sealed_start_incident",
    ),
    _spec(
        "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION",
        "RETAINED_EXECUTION_OBSERVER",
        ("classification", "runtime_scan", "sky_action"),
        "classify_owner_dead_sky_action",
    ),
    _spec(
        "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
        "RETAINED_EXECUTION_OBSERVER",
        ("audit", "handoff_action", "handoff_record"),
        "create_recovery_handoff_if_required",
    ),
    _spec(
        "RETAINED_ACQUIRE_TEARDOWN_SEALING",
        "RETAINED_EXECUTION_OBSERVER",
        ("domain", "operation_identity_sha256", "plan", "owner_nonce_capsule"),
        "commit_teardown_seal",
    ),
    _spec(
        "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
        "RETAINED_EXECUTION_OBSERVER",
        ("first_runtime_scan", "minimum_quiet_seconds", "second_runtime_scan"),
        "prove_zero_activation_work",
    ),
    _spec(
        "RETAINED_ENTER_TEARDOWN_SEALED",
        "RETAINED_EXECUTION_OBSERVER",
        ("domain", "operation_identity_sha256", "plan", "owner_nonce_capsule"),
        "commit_teardown_sealed",
    ),
    _spec(
        "RETAINED_CORRELATE_REQUESTS_AND_JOBS",
        "RETAINED_OPERATOR_DISPOSITION",
        ("correlation_request", "runtime_scan"),
        "correlate_requests_and_jobs",
    ),
    _spec(
        "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
        "RETAINED_OPERATOR_DISPOSITION",
        ("cancellation_command", "cancellation_observation"),
        "reconcile_request_and_job_cancellation",
    ),
    _spec(
        "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
        "RETAINED_WORKER_DRAIN",
        ("first_runtime_scan", "minimum_quiet_seconds", "second_runtime_scan", "stop_command"),
        "quiesce_controller_if_required",
    ),
    _spec(
        "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        "RETAINED_WORKER_DRAIN",
        ("liability_handoff", "runtime_terminal_proof"),
        "reconcile_worker_launches_and_transfer_liabilities",
    ),
    _spec(
        "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
        "RETAINED_WORKER_DRAIN",
        ("worker_drain_action", "worker_drain_audit", "worker_drain_candidate"),
        "reconcile_workers_and_allocations",
    ),
    _spec(
        "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
        "RETAINED_TERMINAL_V2",
        ("terminal_audit", "terminal_payload", "terminal_write_action"),
        "create_or_reconcile_terminal_v2",
    ),
    _spec(
        "RETAINED_ENTER_RECOVERY_COMPLETE",
        "RETAINED_TERMINAL_V2",
        ("domain", "operation_identity_sha256", "plan", "owner_nonce_capsule"),
        "commit_recovery_complete",
    ),
    _spec(
        "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
        "RETAINED_FINALIZER",
        ("finalization_proof", "finalization_write_action"),
        "publish_support_plane_finalized",
    ),
    _spec(
        "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
        "RETAINED_FINALIZER",
        ("support_delete_result",),
        "reconcile_delete_until_absent",
    ),
    _spec(
        "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT",
        "RETAINED_ORPHAN_AUDIT",
        ("recovery_snapshot_capture", "runtime_scan"),
        "discover_exact_recovery_snapshot",
    ),
    _spec(
        "RETAINED_AUDIT_SUPPORT_ORPHANS",
        "RETAINED_ORPHAN_AUDIT",
        ("orphan_audit_request", "runtime_scan"),
        "audit_support_orphans",
    ),
    _spec(
        "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "cleanup_schedule", "snapshot_identity"),
        "arm_snapshot_cleanup_control_and_schedule",
    ),
    _spec(
        "RETAINED_INVOKE_H1G_DRAINED_WRITER",
        "RETAINED_H1G_DRAINED",
        ("drained_prerequisites", "drained_write_action"),
        "invoke_h1g_drained_writer",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "cleanup_schedule", "observed_at"),
        "validate_snapshot_cleanup_schedule_and_deadline",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "owner_acquisition"),
        "acquire_or_take_over_snapshot_cleanup_owner",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "delete_action"),
        "arm_snapshot_delete_action",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "delete_authority"),
        "audit_snapshot_delete_authority",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "delete_authority", "staged_attempt"),
        "atomic_consume_and_stage_snapshot_delete",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "delete_request", "staged_attempt"),
        "send_same_snapshot_delete_id_or_read_back",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "describe_observation"),
        "reconcile_snapshot_describe",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "terminal_evidence"),
        "record_snapshot_cleanup_terminal_evidence",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("ambiguous_attempt", "cleanup_control", "describe_observation"),
        "close_ambiguous_snapshot_delete_attempt",
    ),
    _spec(
        "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
        "RETAINED_SNAPSHOT_CLEANUP",
        ("cleanup_control", "next_attempt"),
        "arm_next_same_id_snapshot_delete_attempt",
    ),
)


OPERATION_SPECS: Mapping[str, OperationSpec] = {
    spec.operation_kind: spec for spec in _SPECS
}

if len(OPERATION_SPECS) != 32:  # pragma: no cover - import-time invariant
    raise RuntimeError("Task 12 operation registry must contain exactly 32 operations")


RETAINED_OPERATION_SEQUENCE = (
    "RECONCILE_RETAINED_LIFECYCLE_TRIGGER",
    "RETAINED_ACQUIRE_RECOVERY_SEALING",
    "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION",
    "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT",
    "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION",
    "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED",
    "RETAINED_CORRELATE_REQUESTS_AND_JOBS",
    "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION",
    "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED",
    "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
    "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS",
    "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2",
    "RETAINED_ENTER_RECOVERY_COMPLETE",
    "RETAINED_ACQUIRE_TEARDOWN_SEALING",
    "RETAINED_PROVE_ZERO_ACTIVATION_WORK",
    "RETAINED_ENTER_TEARDOWN_SEALED",
    "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED",
    "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
    "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT",
    "RETAINED_AUDIT_SUPPORT_ORPHANS",
    "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
    "RETAINED_INVOKE_H1G_DRAINED_WRITER",
)
SNAPSHOT_OPERATION_SEQUENCE = (
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
)
OPERATION_PREDECESSORS: Mapping[str, str | None] = {
    operation: sequence[index - 1] if index else None
    for sequence in (RETAINED_OPERATION_SEQUENCE, SNAPSHOT_OPERATION_SEQUENCE)
    for index, operation in enumerate(sequence)
}
OPERATION_PREDECESSORS = {
    **OPERATION_PREDECESSORS,
    # RECORD_TERMINAL and CLOSE are alternate Choice branches from the same
    # authenticated RECONCILE result; CLOSE does not follow RECORD_TERMINAL.
    "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT": (
        "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"
    ),
}

# The tuple order above remains the immutable descriptor order.  Step
# Functions Choice/Catch edges are not linear, however, so execution custody
# uses this separately closed set of authenticated predecessor results.
OPERATION_ALLOWED_PREDECESSORS: Mapping[str, frozenset[str]] = {
    operation: (
        frozenset()
        if predecessor is None
        else frozenset({predecessor})
    )
    for operation, predecessor in OPERATION_PREDECESSORS.items()
}
OPERATION_ALLOWED_PREDECESSORS = {
    **OPERATION_ALLOWED_PREDECESSORS,
    "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY": frozenset(
        {
            "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION",
            "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT",
        }
    ),
    "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE": frozenset(
        {
            "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
            "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK",
            "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        }
    ),
    "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE": frozenset(
        {"SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"}
    ),
    "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT": frozenset(
        {"SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"}
    ),
    "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT": frozenset(
        {"SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"}
    ),
}

_CANONICAL_SOURCE_FAMILIES: Mapping[str, Mapping[str, str]] = {
    # Exact sources are fixed records.  Dynamic records are deliberately not
    # given made-up "Task12" sort keys: their epoch/attempt/revision is read
    # from an authenticated canonical control record before the target key is
    # constructed with records.ledger_sk.
    "activation_index": {
        "kind": "DDB_EXACT",
        "record_type": "glm52_production_activation_index",
    },
    "control": {
        "kind": "DDB_EXACT",
        "record_type": "glm52_production_control",
    },
    "execution": {
        "kind": "DDB_DERIVED",
        "record_type": "glm52_production_execution",
        "control_type": "glm52_production_control",
        "identity_field": "epoch",
        "control_field": "active_epoch",
    },
    "recovery_control": {
        "kind": "DDB_EXACT",
        "record_type": "glm52_production_recovery_control",
    },
    "request_job_correlation": {
        "kind": "DDB_EXACT",
        "record_type": "glm52_task12_request_job_correlation_v1",
    },
    "worker_drain_authority": {
        # Worker liabilities are a plural set.  A recovery-control record has
        # no allocation ordinal, so a fabricated singular liability key would
        # silently drain only one worker.  The root control is the exact
        # authority boundary; the runtime builder must enumerate the complete
        # set under that authenticated authority before it can dispatch.
        "kind": "DDB_EXACT",
        "record_type": "glm52_production_recovery_control",
    },
    "terminal_v2": {
        "kind": "S3_VERSIONED_FROM_CONTROL",
        "record_type": "glm52_production_terminal_v2",
        "control_type": "glm52_production_recovery_control",
        "writer_kind": "TerminalV2",
    },
    "finalization_control": {
        "kind": "DDB_EXACT",
        "record_type": "glm52_production_finalization_control",
    },
    "support_finalized": {
        "kind": "S3_VERSIONED_FROM_CONTROL",
        "record_type": "glm52_production_support_plane_finalized",
        "control_type": "glm52_production_finalization_control",
        "writer_kind": "SupportPlaneFinalized",
    },
    "snapshot_cleanup_control": {
        "kind": "DDB_EXACT",
        "record_type": "glm52_production_snapshot_cleanup_control",
    },
    "snapshot_cleanup_action": {
        "kind": "DDB_DERIVED",
        "record_type": "glm52_production_snapshot_cleanup_action",
        "control_type": "glm52_production_snapshot_cleanup_control",
        "identity_field": "attempt",
        "control_field": "delete_logical_attempt",
    },
    "sky_action": {
        "kind": "DDB_KEY_FROM_CONTROL",
        "record_type": "glm52_production_action",
        "control_type": "glm52_production_control",
        "control_field": "last_sky_post_action_key",
    },
    "snapshot_cleanup_transition": {
        "kind": "DDB_DERIVED",
        "record_type": "glm52_production_snapshot_cleanup_transition",
        "control_type": "glm52_production_snapshot_cleanup_control",
        "identity_field": "revision",
        "control_field": "revision",
    },
    "h1g_drained": {
        "kind": "S3_VERSIONED_FROM_CONTROL",
        "record_type": "glm52_production_h1g_drained",
        "control_type": "glm52_production_finalization_control",
        "writer_kind": "H1GDrained",
    },
}

EXTERNAL_PRODUCER = "PREEXISTING_RETAINED_AUTHORITY"

# Each tuple is (source alias, producer).  All non-external producers are real
# earlier operations that must commit/write the canonical family before return.
OPERATION_SOURCE_PRODUCERS: Mapping[
    str, tuple[tuple[str, str], ...]
] = {
    "RECONCILE_RETAINED_LIFECYCLE_TRIGGER": (
        ("activation_index", EXTERNAL_PRODUCER),
        ("control", EXTERNAL_PRODUCER),
        ("execution", EXTERNAL_PRODUCER),
    ),
    "RETAINED_ACQUIRE_RECOVERY_SEALING": (
        ("activation_index", EXTERNAL_PRODUCER),
        ("control", EXTERNAL_PRODUCER),
        ("execution", EXTERNAL_PRODUCER),
        # The root reconciliation conditionally creates/reconciles the first
        # recovery-control record.  The sealing transition is therefore never
        # permitted to begin from three unrelated preexisting reads.
        ("recovery_control", "RECONCILE_RETAINED_LIFECYCLE_TRIGGER"),
    ),
    "RETAINED_STOP_OR_OBSERVE_SUPPORT_EXECUTION": (
        ("execution", EXTERNAL_PRODUCER),
        ("recovery_control", "RETAINED_ACQUIRE_RECOVERY_SEALING"),
    ),
    "RETAINED_PROVE_SUPPORT_EXECUTION_TERMINAL_OR_SEALED_START_INCIDENT": (
        ("execution", EXTERNAL_PRODUCER),
        ("recovery_control", "RETAINED_ACQUIRE_RECOVERY_SEALING"),
    ),
    "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION": (
        ("activation_index", EXTERNAL_PRODUCER),
        ("control", EXTERNAL_PRODUCER),
        ("execution", EXTERNAL_PRODUCER),
        ("recovery_control", "RETAINED_ACQUIRE_RECOVERY_SEALING"),
        ("sky_action", EXTERNAL_PRODUCER),
    ),
    "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED": (
        ("activation_index", EXTERNAL_PRODUCER),
        ("control", EXTERNAL_PRODUCER),
        ("execution", EXTERNAL_PRODUCER),
        ("recovery_control", "RETAINED_ACQUIRE_RECOVERY_SEALING"),
        ("sky_action", "RETAINED_CLASSIFY_OWNER_DEAD_SKY_ACTION"),
    ),
    "RETAINED_CORRELATE_REQUESTS_AND_JOBS": (
        ("recovery_control", "RETAINED_ACQUIRE_RECOVERY_SEALING"),
        ("execution", EXTERNAL_PRODUCER),
    ),
    "RETAINED_REQUEST_AND_JOB_CANCEL_RECONCILIATION": (
        ("request_job_correlation", "RETAINED_CORRELATE_REQUESTS_AND_JOBS"),
    ),
    "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED": (
        ("request_job_correlation", "RETAINED_CORRELATE_REQUESTS_AND_JOBS"),
    ),
    "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES": (
        ("worker_drain_authority", "RETAINED_QUIESCE_CONTROLLER_IF_REQUIRED"),
    ),
    "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS": (
        (
            "worker_drain_authority",
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        ),
    ),
    "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2": (
        ("recovery_control", "RETAINED_ACQUIRE_RECOVERY_SEALING"),
        (
            "worker_drain_authority",
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_TRANSFER_LIABILITIES",
        ),
    ),
    "RETAINED_ENTER_RECOVERY_COMPLETE": (
        ("terminal_v2", "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"),
        ("recovery_control", "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"),
    ),
    "RETAINED_ACQUIRE_TEARDOWN_SEALING": (
        ("recovery_control", "RETAINED_ENTER_RECOVERY_COMPLETE"),
        ("control", "RETAINED_ENTER_RECOVERY_COMPLETE"),
    ),
    "RETAINED_PROVE_ZERO_ACTIVATION_WORK": (
        ("control", "RETAINED_ACQUIRE_TEARDOWN_SEALING"),
    ),
    "RETAINED_ENTER_TEARDOWN_SEALED": (
        ("control", "RETAINED_ACQUIRE_TEARDOWN_SEALING"),
        ("finalization_control", "RETAINED_ACQUIRE_TEARDOWN_SEALING"),
    ),
    "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED": (
        ("finalization_control", "RETAINED_ENTER_TEARDOWN_SEALED"),
    ),
    "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT": (
        ("support_finalized", "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"),
    ),
    "RETAINED_DISCOVER_EXACT_RECOVERY_SNAPSHOT": (
        ("support_finalized", "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"),
    ),
    "RETAINED_AUDIT_SUPPORT_ORPHANS": (
        ("support_finalized", "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"),
    ),
    "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE": (
        ("finalization_control", "RETAINED_ENTER_TEARDOWN_SEALED"),
        ("support_finalized", "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"),
    ),
    "RETAINED_INVOKE_H1G_DRAINED_WRITER": (
        (
            "snapshot_cleanup_control",
            "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
        ),
        ("support_finalized", "RETAINED_PUBLISH_SUPPORT_PLANE_FINALIZED"),
    ),
    "SNAPSHOT_CLEANUP_VALIDATE_SCHEDULE_AND_DEADLINE": (
        (
            "snapshot_cleanup_control",
            "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
        ),
    ),
    "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER": (
        (
            "snapshot_cleanup_control",
            "RETAINED_ARM_SNAPSHOT_CLEANUP_CONTROL_AND_SCHEDULE",
        ),
    ),
    "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION": (
        (
            "snapshot_cleanup_control",
            "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        ),
    ),
    "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY": (
        ("snapshot_cleanup_action", "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION"),
    ),
    "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT": (
        ("snapshot_cleanup_action", "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION"),
        (
            "snapshot_cleanup_control",
            "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        ),
    ),
    "SNAPSHOT_CLEANUP_SEND_SAME_ID_OR_READ_BACK": (
        (
            "snapshot_cleanup_control",
            "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        ),
        (
            "snapshot_cleanup_transition",
            "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
        ),
    ),
    "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE": (
        (
            "snapshot_cleanup_control",
            "SNAPSHOT_CLEANUP_ACQUIRE_OR_TAKE_OVER_OWNER",
        ),
        ("snapshot_cleanup_action", "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION"),
    ),
    "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE": (
        ("snapshot_cleanup_control", "SNAPSHOT_CLEANUP_RECONCILE_DESCRIBE"),
    ),
    "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT": (
        ("snapshot_cleanup_control", "SNAPSHOT_CLEANUP_RECORD_TERMINAL_EVIDENCE"),
        ("snapshot_cleanup_action", "SNAPSHOT_CLEANUP_ARM_DELETE_ACTION"),
    ),
    "SNAPSHOT_CLEANUP_ARM_NEXT_SAME_ID_ATTEMPT": (
        ("snapshot_cleanup_control", "SNAPSHOT_CLEANUP_CLOSE_AMBIGUOUS_ATTEMPT"),
    ),
}


def _produced_sources() -> Mapping[str, frozenset[str]]:
    """Return the canonical families each operation must make available.

    This is intentionally derived from consumers rather than maintained as a
    second hand-written workflow table.  A producer cannot be considered live
    unless some later operation consumes its named canonical family.
    """

    produced: dict[str, set[str]] = {operation: set() for operation in OPERATION_SPECS}
    for bindings in OPERATION_SOURCE_PRODUCERS.values():
        for alias, producer in bindings:
            if producer != EXTERNAL_PRODUCER:
                produced[producer].add(alias)
    return {operation: frozenset(aliases) for operation, aliases in produced.items()}


OPERATION_PRODUCED_SOURCES = _produced_sources()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise Task12OperationContractError("operation decimal is not finite")
        return format(value, "f")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, AbcMapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def build_operation_source_payloads(
    requests: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Serialize and deeply validate the complete 32-operation source graph.

    ``requests`` is keyed by operation kind and contains domain-native typed
    objects (plans, scans, authorities, captures, and proofs).  This is the
    production materializer boundary used before a versioned source object is
    published; it never accepts prehashed operation rows.
    """

    if type(requests) is not dict or set(requests) != set(OPERATION_SPECS):
        raise Task12OperationContractError(
            "operation source request inventory is not the closed 32-state graph"
        )
    payloads: dict[str, dict[str, object]] = {}
    for operation_kind, raw_request in requests.items():
        request = _json_value(raw_request)
        validate_operation_request_payload(operation_kind, request)
        spec = OPERATION_SPECS[operation_kind]
        payloads[operation_kind] = {
            spec.action_kind + "_request": request
        }
    return payloads


def operation_spec(operation_kind: str) -> OperationSpec:
    try:
        return OPERATION_SPECS[operation_kind]
    except KeyError as exc:
        raise Task12OperationContractError(
            f"unknown Task 12 operation_kind: {operation_kind!r}"
        ) from exc


def _exact_source_coordinate(
    *, alias: str, record_type: str, activation_id: str
) -> dict[str, object]:
    identity: dict[str, object] = {}
    if record_type != "glm52_production_activation_index":
        identity["activation_id"] = activation_id
    try:
        sort_key = ledger_sk(record_type, **identity)
    except (TypeError, ValueError) as exc:
        raise Task12OperationContractError(
            alias + " source has no canonical exact ledger coordinate"
        ) from exc
    return {
        "alias": alias,
        "coordinate_kind": "DDB_EXACT",
        "record_type": record_type,
        "sort_key": sort_key,
    }


def _canonical_source_coordinate(
    *,
    alias: str,
    operation_kind: str,
    activation_id: str,
    generation: int,
    campaign_bucket: str,
) -> dict[str, object]:
    """Build one source coordinate with no runtime payload or guessed ordinal."""

    family = _CANONICAL_SOURCE_FAMILIES[alias]
    kind = family["kind"]
    record_type = family["record_type"]
    if kind == "DDB_EXACT":
        return _exact_source_coordinate(
            alias=alias,
            record_type=record_type,
            activation_id=activation_id,
        )
    control_type = family["control_type"]
    control = _exact_source_coordinate(
        alias="control",
        record_type=control_type,
        activation_id=activation_id,
    )
    control.pop("alias")
    if kind == "DDB_DERIVED":
        integer_offset = (
            1
            if alias == "snapshot_cleanup_action"
            and operation_kind
            in {
                "SNAPSHOT_CLEANUP_AUDIT_DELETE_AUTHORITY",
                "SNAPSHOT_CLEANUP_ATOMIC_CONSUME_AND_STAGE_POSSIBLY_SENT",
            }
            else 0
        )
        return {
            "alias": alias,
            "coordinate_kind": "DDB_DERIVED",
            "record_type": record_type,
            "control": control,
            "derive": {
                "identity_field": family["identity_field"],
                "control_field": family["control_field"],
                "integer_offset": integer_offset,
            },
        }
    if kind == "DDB_KEY_FROM_CONTROL":
        return {
            "alias": alias,
            "coordinate_kind": "DDB_KEY_FROM_CONTROL",
            "record_type": record_type,
            "control": control,
            "control_field": family["control_field"],
        }
    if kind == "S3_VERSIONED_FROM_CONTROL":
        writer_kind = family["writer_kind"]
        # Couple reader key derivation to the writer's canonical coordinate
        # builder, while deliberately keeping a future version/body identity
        # out of this static descriptor.
        from .task12_writers import retained_writer_coordinate

        coordinate = retained_writer_coordinate(
            writer_kind=writer_kind,
            campaign_bucket=campaign_bucket,
            activation_id=activation_id,
            generation=generation,
        )
        if not coordinate.startswith("s3://" + campaign_bucket + "/"):
            raise Task12OperationContractError(alias + " writer coordinate drifted")
        try:
            control_sort_key = ledger_sk(
                "glm52_task12_versioned_writer_control_v1",
                activation_id=activation_id,
                generation=generation,
                writer_kind=writer_kind,
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - static invariant
            raise Task12OperationContractError(
                alias + " versioned writer control has no canonical key"
            ) from exc
        return {
            "alias": alias,
            "coordinate_kind": "S3_VERSIONED_FROM_CONTROL",
            "record_type": record_type,
            "writer_kind": writer_kind,
            "campaign_bucket": campaign_bucket,
            "control": {
                "coordinate_kind": "DDB_EXACT",
                "record_type": "glm52_task12_versioned_writer_control_v1",
                "sort_key": control_sort_key,
            },
        }
    raise Task12OperationContractError(alias + " source kind is not closed")


def _expected_source_coordinates(
    *,
    operation_kind: str,
    activation_id: str,
    generation: int,
    campaign_bucket: str,
) -> list[dict[str, object]]:
    try:
        bindings = OPERATION_SOURCE_PRODUCERS[operation_kind]
    except KeyError as exc:  # pragma: no cover - guarded by operation_spec
        raise Task12OperationContractError("operation has no source bindings") from exc
    return [
        _canonical_source_coordinate(
            alias=alias,
            operation_kind=operation_kind,
            activation_id=activation_id,
            generation=generation,
            campaign_bucket=campaign_bucket,
        )
        for alias, _producer in bindings
    ]


def _descriptor_allowed_read_apis(operation_kind: str) -> tuple[str, ...]:
    """Return the action plus descriptor-source read authority, exactly."""

    reads = set(operation_spec(operation_kind).allowed_read_apis)
    for alias, _producer in OPERATION_SOURCE_PRODUCERS[operation_kind]:
        kind = _CANONICAL_SOURCE_FAMILIES[alias]["kind"]
        reads.add("dynamodb:GetItem")
        if kind == "S3_VERSIONED_FROM_CONTROL":
            reads.add("s3:GetObjectVersion")
    return tuple(sorted(reads))


def build_operation_descriptor(
    *,
    operation_kind: str,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    coordinates: Mapping[str, object],
    static_authority_sha256: str,
) -> dict[str, Any]:
    """Build one immutable authority for live operation input materialization."""

    spec = operation_spec(operation_kind)
    if (
        type(activation_id) is not str
        or _ACTIVATION_ID_RE.fullmatch(activation_id) is None
        or type(activation_ordinal) is not int
        or isinstance(activation_ordinal, bool)
        or activation_ordinal < 1
        or type(generation) is not int
        or isinstance(generation, bool)
        or generation < 1
    ):
        raise Task12OperationContractError(
            f"{operation_kind} descriptor activation is invalid"
        )
    if (
        type(coordinates) is not dict
        or set(coordinates) != set(spec.coordinate_fields)
    ):
        raise Task12OperationContractError(
            f"{operation_kind} descriptor coordinates are not exact"
        )
    if _SHA256_RE.fullmatch(static_authority_sha256) is None:
        raise Task12OperationContractError(
            f"{operation_kind} static authority identity is invalid"
        )
    partition_key = coordinates.get("ledger_partition_key")
    sources = coordinates.get("sources")
    if (
        type(partition_key) is not str
        or not partition_key
        or type(sources) is not list
        or not sources
    ):
        raise Task12OperationContractError(
            f"{operation_kind} live ledger sources are not closed"
        )
    s3_buckets = {
        source.get("campaign_bucket")
        for source in sources
        if type(source) is dict
        and source.get("coordinate_kind") == "S3_VERSIONED_FROM_CONTROL"
    }
    if len(s3_buckets) > 1 or any(
        type(bucket) is not str or not bucket for bucket in s3_buckets
    ):
        raise Task12OperationContractError(
            f"{operation_kind} versioned source bucket drifted"
        )
    campaign_bucket = next(iter(s3_buckets), "")
    expected_sources = _expected_source_coordinates(
        operation_kind=operation_kind,
        activation_id=activation_id,
        generation=generation,
        campaign_bucket=campaign_bucket,
    )
    if sources != expected_sources:
        raise Task12OperationContractError(
            f"{operation_kind} typed live sources are not canonical"
        )
    if coordinates.get("region", REGION) != REGION:
        raise Task12OperationContractError(
            f"{operation_kind} descriptor region drifted"
        )
    if "kms_key_id" in spec.coordinate_fields and (
        type(coordinates.get("kms_key_id")) is not str
        or not coordinates["kms_key_id"].startswith(
            f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/"
        )
    ):
        raise Task12OperationContractError(
            f"{operation_kind} descriptor KMS key drifted"
        )
    body = {
        "schema_version": 1,
        "record_type": (
            "glm52_task12_" + operation_kind.lower() + "_descriptor_v1"
        ),
        "handler_kind": spec.handler_kind,
        "operation_kind": operation_kind,
        "activation_id": activation_id,
        "activation_ordinal": activation_ordinal,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "builder_kind": spec.builder_kind,
        "allowed_read_apis": list(_descriptor_allowed_read_apis(operation_kind)),
        "source_coordinates": dict(coordinates),
        "static_authority_sha256": static_authority_sha256,
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def validate_operation_descriptor(value: object) -> dict[str, Any]:
    """Validate one immutable live-source descriptor without reading live state."""

    if type(value) is not dict:
        raise Task12OperationContractError("operation descriptor is not an object")
    operation_kind = value.get("operation_kind")
    spec = operation_spec(operation_kind) if type(operation_kind) is str else None
    if spec is None or set(value) != {
        "schema_version",
        "record_type",
        "handler_kind",
        "operation_kind",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "builder_kind",
        "allowed_read_apis",
        "source_coordinates",
        "static_authority_sha256",
        "canonical_body_sha256",
    }:
        raise Task12OperationContractError(
            "operation descriptor fields are not closed"
        )
    rebuilt = build_operation_descriptor(
        operation_kind=operation_kind,
        activation_id=value["activation_id"],
        activation_ordinal=value["activation_ordinal"],
        generation=value["generation"],
        coordinates=value["source_coordinates"],
        static_authority_sha256=value["static_authority_sha256"],
    )
    if rebuilt != value:
        raise Task12OperationContractError(
            f"{operation_kind} descriptor identity drifted"
        )
    return dict(value)


def build_operation_descriptor_graph(
    *,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    ledger_partition_key: str,
    campaign_bucket: str,
    kms_key_id: str,
    static_authority_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Build all 32 immutable descriptors over operation-scoped live rows.

    Each operation reads only canonical retained record families.  Every source
    names either the preexisting retained activation authority or a real prior
    operation that commits/writes that family before returning.
    """

    descriptors: dict[str, dict[str, Any]] = {}
    for operation_kind, spec in OPERATION_SPECS.items():
        coordinates: dict[str, object] = {
            "ledger_partition_key": ledger_partition_key,
            "sources": _expected_source_coordinates(
                operation_kind=operation_kind,
                activation_id=activation_id,
                generation=generation,
                campaign_bucket=campaign_bucket,
            ),
        }
        if "region" in spec.coordinate_fields:
            coordinates["region"] = REGION
        if "kms_key_id" in spec.coordinate_fields:
            coordinates["kms_key_id"] = kms_key_id
        descriptors[operation_kind] = build_operation_descriptor(
            operation_kind=operation_kind,
            activation_id=activation_id,
            activation_ordinal=activation_ordinal,
            generation=generation,
            coordinates=coordinates,
            static_authority_sha256=static_authority_sha256,
        )
    return descriptors


def validate_operation_source_producers() -> None:
    """Reject any descriptor source without an existing or prior producer."""

    if set(OPERATION_SOURCE_PRODUCERS) != set(OPERATION_SPECS):
        raise Task12OperationContractError(
            "producer/consumer matrix is not the closed 32-state graph"
        )
    order = {
        operation: index
        for index, operation in enumerate(
            RETAINED_OPERATION_SEQUENCE + SNAPSHOT_OPERATION_SEQUENCE
        )
    }
    for consumer, bindings in OPERATION_SOURCE_PRODUCERS.items():
        if not bindings or len({alias for alias, _ in bindings}) != len(bindings):
            raise Task12OperationContractError(
                consumer + " canonical sources are empty or duplicated"
            )
        for alias, producer in bindings:
            if alias not in _CANONICAL_SOURCE_FAMILIES:
                raise Task12OperationContractError(
                    consumer + " source family is not canonical"
                )
            if producer != EXTERNAL_PRODUCER and (
                producer not in order or order[producer] >= order[consumer]
            ):
                raise Task12OperationContractError(
                    consumer + " source lacks a real prior producer"
                )
            if (
                producer != EXTERNAL_PRODUCER
                and alias not in OPERATION_PRODUCED_SOURCES[producer]
            ):
                raise Task12OperationContractError(
                    consumer + " source is not committed by its named producer"
                )
        if OPERATION_PREDECESSORS[consumer] is not None and not any(
            producer != EXTERNAL_PRODUCER for _alias, producer in bindings
        ):
            raise Task12OperationContractError(
                consumer + " has no prior canonical producer"
            )


validate_operation_source_producers()


def validate_operation_request_payload(
    operation_kind: str, payload: object
) -> dict[str, Any]:
    """Validate the nested, operation-specific request before publication."""

    spec = operation_spec(operation_kind)
    if type(payload) is not dict:
        raise Task12OperationContractError(
            f"{operation_kind} request must be one exact object"
        )
    if set(payload) != set(spec.request_fields):
        raise Task12OperationContractError(
            f"{operation_kind} request fields are not exact: expected "
            f"{sorted(spec.request_fields)!r}, got {sorted(payload)!r}"
        )
    for field, value in payload.items():
        if field in _INTEGER_REQUEST_FIELDS:
            valid = type(value) is int and value > 0
        elif field in _BOOLEAN_REQUEST_FIELDS:
            valid = type(value) is bool
        elif field in _ARRAY_REQUEST_FIELDS:
            valid = type(value) is list and bool(value)
        elif field in _TEXT_REQUEST_FIELDS or field.endswith(
            ("_sha256", "_arn", "_id", "_at", "_hex")
        ):
            valid = type(value) is str and bool(value) and value == value.strip()
        else:
            valid = type(value) is dict and bool(value)
        if not valid:
            raise Task12OperationContractError(
                f"{operation_kind} request field {field!r} is not typed"
            )
        if field.endswith("_sha256") and _SHA256_RE.fullmatch(value) is None:
            raise Task12OperationContractError(
                f"{operation_kind} request field {field!r} is not a SHA-256"
            )
        if field.endswith("_arn") and _ARN_RE.fullmatch(value) is None:
            raise Task12OperationContractError(
                f"{operation_kind} request field {field!r} is not an ARN"
            )
        if field.endswith("_hex"):
            try:
                decoded = bytes.fromhex(value)
            except ValueError as exc:
                raise Task12OperationContractError(
                    f"{operation_kind} request field {field!r} is not hex"
                ) from exc
            if len(decoded) != 32:
                raise Task12OperationContractError(
                    f"{operation_kind} request field {field!r} is not 32 bytes"
                )
    _validate_domain_shapes(spec.action_kind, payload)
    try:
        canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise Task12OperationContractError(
            f"{operation_kind} request is not canonical JSON"
        ) from exc
    return dict(payload)


def _shape(annotation: object, value: object) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        for choice in arguments:
            if choice is type(None):
                continue
            try:
                return _shape(choice, value)
            except (Task12OperationContractError, TypeError, ValueError):
                pass
        raise Task12OperationContractError("union request value is not closed")
    if origin in (tuple, list):
        if type(value) not in (list, tuple):
            raise Task12OperationContractError("request sequence is not exact")
        item_type = arguments[0] if arguments else Any
        return tuple(_shape(item_type, item) for item in value)
    if origin in (dict, Mapping, AbcMapping):
        if type(value) is not dict:
            raise Task12OperationContractError("request mapping is not exact")
        return dict(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_shape(annotation, value)
    if annotation is bytes:
        if type(value) is not str:
            raise Task12OperationContractError("request bytes are not hex")
        try:
            return bytes.fromhex(value)
        except ValueError as exc:
            raise Task12OperationContractError("request bytes are not hex") from exc
    if annotation is Decimal:
        if type(value) is not str:
            raise Task12OperationContractError("request decimal is not text")
        return Decimal(value)
    if annotation in (str, int, bool) and type(value) is not annotation:
        raise Task12OperationContractError("request scalar has wrong exact type")
    return value


def _dataclass_shape(cls: type[Any], value: object) -> object:
    if type(value) is not dict or set(value) != {
        field.name for field in fields(cls)
    }:
        raise Task12OperationContractError(
            cls.__name__ + " request fields are not closed"
        )
    hints = get_type_hints(cls)
    try:
        return cls(
            **{
                field.name: _shape(hints.get(field.name, Any), value[field.name])
                for field in fields(cls)
            }
        )
    except Task12OperationContractError:
        raise
    except Exception as exc:
        raise Task12OperationContractError(
            cls.__name__ + " request is invalid"
        ) from exc


def _validate_domain_shapes(
    action_kind: str, payload: Mapping[str, Any]
) -> None:
    from .task12_orphan_audit import (
        DirectGrantAttribution,
        KmsGrantIdentity,
        KmsGrantScan,
        ResourceInventoryScan,
        RetainedResource,
        ServiceGrantAttribution,
    )
    from .task12_retained_state import (
        RecoveryProgressPlan,
        RecoverySealPlan,
        SnapshotCleanupOwnerTakeoverPlan,
        SnapshotCleanupReconcileReadPlan,
        SnapshotDeleteActionPlan,
        SnapshotCleanupTransitionPlan,
        TeardownSealPlan,
    )
    from .task12_runtime import (
        CancellationCommand,
        CancellationObservation,
        ControllerStopEvidence,
        RuntimeScan,
        RuntimeTerminalProof,
    )
    from .task12_snapshot_cleanup import (
        SnapshotCapture,
        SnapshotObservation,
        SnapshotSchedule,
    )
    from .task12_worker_drain import WorkerDrainCandidate
    from .task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
    )

    scalar_shapes: dict[str, type[Any]] = {
        "runtime_scan": RuntimeScan,
        "first_runtime_scan": RuntimeScan,
        "second_runtime_scan": RuntimeScan,
        "stop": ControllerStopEvidence,
        "command": CancellationCommand,
        "observation": CancellationObservation,
        "runtime_terminal": RuntimeTerminalProof,
        "candidate": WorkerDrainCandidate,
        "capture": SnapshotCapture,
        "schedule": SnapshotSchedule,
        "inventory": ResourceInventoryScan,
        "pre_cleanup_grants": KmsGrantScan,
    }
    if action_kind in {
        "commit_recovery_seal",
    }:
        scalar_shapes["plan"] = RecoverySealPlan
    elif action_kind in {
        "commit_recovery_complete",
    }:
        scalar_shapes["plan"] = RecoveryProgressPlan
    elif action_kind in {"commit_teardown_seal", "commit_teardown_sealed"}:
        scalar_shapes["plan"] = TeardownSealPlan
    elif action_kind in {
        "arm_snapshot_delete_action",
        "close_ambiguous_snapshot_delete_attempt",
        "arm_next_same_id_snapshot_delete_attempt",
    }:
        scalar_shapes["plan"] = SnapshotDeleteActionPlan
    elif action_kind == "acquire_or_take_over_snapshot_cleanup_owner":
        raw_plan = payload.get("plan")
        if type(raw_plan) is not dict:
            raise Task12OperationContractError(
                "snapshot owner plan is not one exact object"
            )
        scalar_shapes["plan"] = (
            SnapshotCleanupOwnerTakeoverPlan
            if set(raw_plan)
            == {"index", "authority", "cleanup_control"}
            else SnapshotCleanupTransitionPlan
        )
    elif action_kind == "reconcile_snapshot_describe":
        raw_plan = payload.get("plan")
        if type(raw_plan) is not dict:
            raise Task12OperationContractError(
                "snapshot reconciliation plan is not one exact object"
            )
        scalar_shapes["plan"] = (
            SnapshotCleanupReconcileReadPlan
            if set(raw_plan)
            == {
                "index",
                "authority",
                "cleanup_control",
                "cleanup_action",
            }
            else SnapshotCleanupTransitionPlan
        )
    elif "snapshot" in action_kind and "plan" in payload:
        scalar_shapes["plan"] = SnapshotCleanupTransitionPlan
    if "action" in payload and action_kind not in {
        "audit_snapshot_delete_authority"
    }:
        scalar_shapes["action"] = RetainedWriterActionAuthority
    if "audit" in payload:
        scalar_shapes["audit"] = RetainedWriterAuditAuthority
    for name, cls in scalar_shapes.items():
        if name in payload:
            _dataclass_shape(cls, payload[name])

    array_shapes: dict[str, type[Any]] = {
        "observations": SnapshotObservation,
        "expected_retained": RetainedResource,
        "retained_grant_baseline": KmsGrantIdentity,
        "direct_grants": DirectGrantAttribution,
        "service_grants": ServiceGrantAttribution,
    }
    for name, cls in array_shapes.items():
        if name in payload:
            for item in payload[name]:
                _dataclass_shape(cls, item)


def build_operation_row(
    *,
    operation_kind: str,
    activation_id: str,
    activation_ordinal: int,
    generation: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exact, content-addressed operation input row."""

    spec = operation_spec(operation_kind)
    if not isinstance(activation_id, str) or not _ACTIVATION_ID_RE.fullmatch(
        activation_id
    ):
        raise Task12OperationContractError("activation_id is not closed")
    if (
        isinstance(activation_ordinal, bool)
        or not isinstance(activation_ordinal, int)
        or activation_ordinal < 1
    ):
        raise Task12OperationContractError("activation_ordinal must be a positive int")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise Task12OperationContractError("generation must be a positive int")
    if not isinstance(payload, Mapping):
        raise Task12OperationContractError("payload must be an object")
    actual_fields = frozenset(payload)
    if actual_fields != spec.payload_fields:
        raise Task12OperationContractError(
            "payload fields are not exact for "
            f"{operation_kind}: expected {sorted(spec.payload_fields)!r}, "
            f"got {sorted(actual_fields)!r}"
        )
    request_key = spec.action_kind + "_request"
    validate_operation_request_payload(operation_kind, payload[request_key])

    body: dict[str, Any] = {
        "schema_version": 1,
        "record_type": spec.record_type,
        "handler_kind": spec.handler_kind,
        "operation_kind": operation_kind,
        "activation_id": activation_id,
        "activation_ordinal": activation_ordinal,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        **dict(payload),
    }
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def validate_operation_row(value: object) -> dict[str, Any]:
    """Authenticate and type-check a fully materialized operation row."""

    if type(value) is not dict:
        raise Task12OperationContractError("operation row must be one exact object")
    operation_kind = value.get("operation_kind")
    if type(operation_kind) is not str:
        raise Task12OperationContractError("operation row lacks operation_kind")
    spec = operation_spec(operation_kind)
    common_fields = {
        "schema_version",
        "record_type",
        "handler_kind",
        "operation_kind",
        "activation_id",
        "activation_ordinal",
        "generation",
        "generation_text",
        "canonical_body_sha256",
    }
    if set(value) != common_fields | set(spec.payload_fields):
        raise Task12OperationContractError(
            f"{operation_kind} row fields are not exact"
        )
    request_key = spec.action_kind + "_request"
    validate_operation_request_payload(operation_kind, value[request_key])
    body = dict(value)
    supplied_hash = body.pop("canonical_body_sha256")
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != spec.record_type
        or value.get("handler_kind") != spec.handler_kind
        or type(value.get("activation_id")) is not str
        or _ACTIVATION_ID_RE.fullmatch(value["activation_id"]) is None
        or isinstance(value.get("activation_ordinal"), bool)
        or type(value.get("activation_ordinal")) is not int
        or value["activation_ordinal"] < 1
        or isinstance(value.get("generation"), bool)
        or type(value.get("generation")) is not int
        or value["generation"] < 1
        or value.get("generation_text") != f"{value['generation']:08d}"
        or supplied_hash != canonical_sha256(body)
    ):
        raise Task12OperationContractError(
            f"{operation_kind} row identity drifted"
        )
    return dict(value)


_TRANSITION_ACTIONS = frozenset(
    {
        "commit_recovery_seal",
        "commit_recovery_complete",
        "commit_teardown_seal",
        "commit_teardown_sealed",
        "arm_snapshot_cleanup_control_and_schedule",
        "acquire_or_take_over_snapshot_cleanup_owner",
        "arm_snapshot_delete_action",
        "atomic_consume_and_stage_snapshot_delete",
        "record_snapshot_cleanup_terminal_evidence",
        "close_ambiguous_snapshot_delete_attempt",
        "arm_next_same_id_snapshot_delete_attempt",
    }
)
_WRITE_ACTIONS = frozenset(
    {
        "create_recovery_handoff_if_required",
        "create_or_reconcile_terminal_v2",
        "publish_support_plane_finalized",
        "invoke_h1g_drained_writer",
    }
)
_AUDIT_ACTIONS = frozenset(
    {
        "classify_owner_dead_sky_action",
        "audit_support_orphans",
        "audit_snapshot_delete_authority",
        "discover_exact_recovery_snapshot",
    }
)
_PROOF_ACTIONS = frozenset(
    {
        "reconcile_retained_lifecycle_trigger",
        "prove_support_execution_terminal_or_sealed_start_incident",
        "prove_zero_activation_work",
        "validate_snapshot_cleanup_schedule_and_deadline",
        "reconcile_snapshot_describe",
        "reconcile_delete_until_absent",
    }
)


def _behavior_kind(action_kind: str) -> str:
    if action_kind in _TRANSITION_ACTIONS:
        return "STATE_TRANSITION"
    if action_kind in _WRITE_ACTIONS:
        return "RETAINED_WRITE"
    if action_kind in _AUDIT_ACTIONS:
        return "AUDIT"
    if action_kind in _PROOF_ACTIONS:
        return "PROOF"
    return "EXTERNAL_ACTION"


def _validated_predecessor(
    *,
    operation_kind: str,
    operation_input: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    allowed = OPERATION_ALLOWED_PREDECESSORS[operation_kind]
    prior = operation_input.get("task12_last_result")
    if not allowed:
        if prior is not None:
            raise Task12OperationContractError(
                f"{operation_kind} must not consume a predecessor"
            )
        return None, None
    if (
        type(prior) is not dict
        or prior.get("operation_kind") not in allowed
        or prior.get("outcome") != "SUCCEEDED"
        or type(prior.get("canonical_body_sha256")) is not str
    ):
        raise Task12OperationContractError(
            f"{operation_kind} did not consume its exact predecessor"
        )
    prior_body = dict(prior)
    prior_identity = prior_body.pop("canonical_body_sha256")
    if prior_identity != canonical_sha256(prior_body):
        raise Task12OperationContractError(
            f"{operation_kind} predecessor identity drifted"
        )
    predecessor = prior.get("operation_kind")
    assert isinstance(predecessor, str)
    return predecessor, prior_identity


def execute_operation_semantics(
    *,
    row: object,
    operation_input: Mapping[str, Any],
) -> OperationExecutionResult:
    """Select and execute the operation-specific semantic contract.

    This boundary deliberately does not perform AWS I/O.  It is called after
    the operation's domain transition/action function has produced its closed
    payload, and turns that distinct effect into the typed ResultPath value
    consumed by the next state.
    """

    value = validate_operation_row(row)
    operation_kind = value["operation_kind"]
    spec = OPERATION_SPECS[operation_kind]
    if type(operation_input) is not dict:
        raise Task12OperationContractError("operation_input must be one exact object")
    predecessor, predecessor_identity = _validated_predecessor(
        operation_kind=operation_kind,
        operation_input=operation_input,
    )
    payload = {field: value[field] for field in sorted(spec.payload_fields)}
    payload_identity = canonical_sha256(payload)
    behavior_kind = _behavior_kind(spec.action_kind)
    semantic_result = {
        "effect": spec.action_kind,
        "behavior": behavior_kind,
        "payload_identity_sha256": payload_identity,
    }
    body = {
        "result_kind": spec.result_kind,
        "operation_kind": operation_kind,
        "action_kind": spec.action_kind,
        "behavior_kind": behavior_kind,
        "predecessor_operation_kind": predecessor,
        "predecessor_result_identity_sha256": predecessor_identity,
        "operation_payload_identity_sha256": payload_identity,
        "semantic_result": semantic_result,
    }
    return OperationExecutionResult(
        **body,
        canonical_body_sha256=canonical_sha256(body),
    )


__all__ = [
    "OPERATION_SPECS",
    "OPERATION_ALLOWED_PREDECESSORS",
    "OPERATION_PREDECESSORS",
    "OPERATION_PRODUCED_SOURCES",
    "OPERATION_SOURCE_PRODUCERS",
    "EXTERNAL_PRODUCER",
    "RETAINED_OPERATION_SEQUENCE",
    "SNAPSHOT_OPERATION_SEQUENCE",
    "OperationExecutionResult",
    "OperationSpec",
    "Task12OperationContractError",
    "build_operation_descriptor",
    "build_operation_descriptor_graph",
    "build_operation_row",
    "build_operation_source_payloads",
    "canonical_sha256",
    "execute_operation_semantics",
    "operation_spec",
    "validate_operation_row",
    "validate_operation_request_payload",
    "validate_operation_descriptor",
    "validate_operation_source_producers",
]
