"""Closed, import-light durable record validation for GLM-5.2."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Dict, Mapping, Optional, Tuple

from .canonical import canonical_sha256


class RecordValidationError(ValueError):
    """A durable record violates its closed schema."""


_COMMON = (
    "schema_version", "record_type", "account_id", "region", "run_id",
    "campaign_identity_sha256",
)


def _fields(text: str) -> Tuple[str, ...]:
    return tuple(text.split())


RECORD_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "glm52_production_activation_index": _fields(
        """schema_version record_type account_id region run_id
        campaign_identity_sha256 current_activation_id current_activation_ordinal
        prior_activation_id prior_activation_terminal_v2_identity
        prior_h1g_drained_identity prior_spend_ledger_head_identity
        snapshot_cleanup_lineage_sha256 revision updated_at"""
    ),
    "glm52_production_rollover": _fields(
        """schema_version record_type account_id region run_id
        campaign_identity_sha256 activation_id activation_ordinal
        prior_activation_id prior_activation_ordinal prior_terminal_v2_key
        prior_terminal_v2_version_id prior_terminal_v2_body_sha256
        prior_terminal_v2_outcome prior_allocations_array_sha256
        prior_request_evidence_array_sha256
        prior_worker_launch_evidence_array_sha256
        prior_worker_launch_liabilities_array_sha256
        prior_worker_launch_liability_settlements_array_sha256
        prior_post_terminal_allocations_array_sha256
        prior_merged_final_allocations_array_sha256 prior_final_worker_cardinality
        prior_h1g_drained_key prior_h1g_drained_version_id
        prior_h1g_drained_body_sha256 prior_spend_ledger_head_identity
        prior_snapshot_cleanup_control_identity snapshot_cleanup_lineage
        snapshot_cleanup_lineage_sha256 operator_disposition_required
        operator_disposition_identity index_from_revision index_to_revision
        writer_function_version_arn writer_role_arn writer_caller_identity_sha256
        writer_dispatch_identity_sha256 writer_invocation_nonce_sha256
        transaction_client_request_token_sha256 cloudformation_stack_id
        cloudformation_logical_resource_id cloudformation_request_id
        cloudformation_request_type cloudformation_operation_identity_sha256
        transaction_bytes_sha256 stable_physical_resource_id
        cached_success_response_body cached_success_response_body_sha256
        created_at canonical_body_sha256"""
    ),
    "glm52_production_operator_disposition": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal terminal_v2_identity terminal_v2_outcome
        allocations_array_sha256 request_evidence_array_sha256
        worker_launch_evidence_array_sha256 worker_launch_liabilities_array_sha256
        worker_launch_liability_settlements_array_sha256
        post_terminal_allocations_array_sha256 merged_final_allocations_array_sha256
        final_worker_cardinality decision written_approval_key
        written_approval_version_id written_approval_body_sha256
        approval_principal_identity approval_ingested_at writer_function_version_arn
        writer_dispatch_identity_sha256 writer_invocation_nonce_sha256 created_at
        canonical_body_sha256"""
    ),
    "glm52_production_control": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal rollover_identity_sha256 active_epoch
        active_execution_arn active_state_machine_version_arn phase
        fence_head_body_sha256 fence_head_version_id barrier_nonce_sha256
        barrier_state decision_seal_state recovery_seal_state teardown_seal_state
        last_sky_post_generation last_sky_post_action_key last_sky_post_state
        numeric_job_binding_state recovery_control_initial_body_sha256
        finalization_control_initial_body_sha256
        snapshot_cleanup_control_initial_body_sha256 revision updated_at"""
    ),
    "glm52_production_recovery_control": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal rollover_identity_sha256
        cleanup_control_root_identity_sha256 cleanup_transition_chain_head_sha256
        cleanup_transition_chain_length state owner_attempt owner_execution_arn
        owner_state_machine_version_arn owner_dispatch_identity_sha256
        owner_invocation_nonce_sha256 owner_hard_expires_at
        recovery_barrier_nonce_sha256 support_control_revision_at_seal
        support_execution_identity_sha256 allowed_action_set_sha256
        terminal_v2_identity_sha256 revision updated_at"""
    ),
    "glm52_production_finalization_control": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal rollover_identity_sha256 state owner_attempt
        owner_execution_arn owner_state_machine_version_arn
        owner_dispatch_identity_sha256 owner_invocation_nonce_sha256
        owner_hard_expires_at finalization_barrier_nonce_sha256
        teardown_sealed_control_revision terminal_v2_identity_sha256
        support_plane_finalized_identity_sha256 snapshot_disposition_identity_sha256
        h1g_drained_identity_sha256 revision updated_at"""
    ),
    "glm52_production_snapshot_cleanup_control": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal rollover_identity_sha256 state snapshot_id
        snapshot_identity_sha256 source_volume_id snapshot_tags_sha256
        delete_not_before schedule_arn schedule_input_sha256 owner_attempt
        owner_execution_arn owner_state_machine_version_arn
        owner_dispatch_identity_sha256 owner_invocation_nonce_sha256
        owner_hard_expires_at cleanup_barrier_nonce_sha256
        cleanup_lineage_identity_sha256 cleanup_authority_audit_identities
        latest_cleanup_authority_audit_identity_sha256 delete_action_identities
        latest_delete_action_identity_sha256 delete_logical_attempt
        delete_call_count latest_delete_request_id latest_delete_response_sha256
        last_describe_request_id last_describe_response_sha256
        terminal_evidence_sha256 revision updated_at"""
    ),
    "glm52_production_snapshot_cleanup_transition": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal cleanup_control_root_identity_sha256
        from_state to_state from_revision to_revision prior_transition_sha256
        owner_attempt owner_execution_arn owner_state_machine_version_arn
        owner_dispatch_identity_sha256 owner_invocation_nonce_sha256
        authority_audit_identity_sha256 action_identity_sha256 transitioned_at
        canonical_body_sha256"""
    ),
    "glm52_task12_versioned_writer_control_v1": _fields(
        """schema_version record_type account_id region run_id activation_id
        generation generation_text writer_kind campaign_bucket object_key
        object_version_id file_sha256 body_sha256 candidate_identity_sha256
        writer_result_identity_sha256 published_at canonical_body_sha256"""
    ),
    "glm52_task12_request_job_correlation_v1": _fields(
        """schema_version record_type account_id region run_id
        campaign_identity_sha256 activation_id activation_ordinal generation
        generation_text request_ids job_ids request_states job_states
        request_evidence_identity_sha256 job_evidence_identity_sha256
        correlation_source_identity_sha256 observation_identity_sha256
        observed_at canonical_body_sha256"""
    ),
    "glm52_production_support_plane_finalized": _fields(
        """schema_version record_type account_id region run_id
        campaign_identity_sha256 activation_id activation_ordinal
        writer_function_version_arn writer_dispatch_identity_sha256
        writer_invocation_nonce_sha256 published_at canonical_body_sha256"""
    ),
    "glm52_production_h1g_drained": _fields(
        """schema_version record_type account_id region run_id
        campaign_identity_sha256 activation_id activation_ordinal
        terminal_v2_identity support_plane_finalized_identity
        support_stack_deletion_identity orphan_audit_identity
        kms_grant_baseline_equality_identity snapshot_cleanup_control_identity
        snapshot_cleanup_schedule_identity worker_launch_liabilities
        spend_ledger_head_identity retained_resource_inventory_identity
        writer_function_version_arn writer_dispatch_identity_sha256
        writer_invocation_nonce_sha256 published_at canonical_body_sha256"""
    ),
    "glm52_production_recovery_action": (),
    "glm52_production_finalization_action": (),
    "glm52_production_snapshot_cleanup_action": (),
    "glm52_production_worker_launch_liability_action": (),
    "glm52_production_execution": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal epoch epoch_text expected_execution_arn
        expected_execution_name expected_input_sha256
        expected_state_machine_version_arn allocation_window_open_at
        allocation_window_close_at start_owned_at state
        start_owner_function_version_arn start_owner_dispatch_identity_sha256
        start_owner_invocation_nonce_sha256 start_owner_hard_expires_at
        start_send_stage start_attempted_at direct_start_response_identity
        direct_start_request_id observed_start_date terminal_status
        terminal_observed_at describe_execution_request_id
        describe_execution_response_sha256 terminal_body_sha256
        unresolved_start_incident_body_sha256 owner_control_revision
        owner_rollover_identity_sha256 revision updated_at"""
    ),
    "glm52_production_worker_launch": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal generation generation_text sky_action_key
        sky_request_id sky_job_id sky_job_name task_yaml_sha256 request_body_sha256
        allocation_ordinal allocation_ordinal_text prior_worker_launch_identity_sha256
        prior_instance_terminal_identity_sha256
        prior_spend_allocation_close_identity_sha256 ec2_client_token
        launch_parameters_sha256 expected_worker_tags_sha256
        owner_combined_host_instance_id owner_boot_identity_sha256
        owner_service_identity_sha256 owner_attempt owner_principal_arn
        owner_function_version_arn owner_dispatch_identity_sha256
        owner_invocation_nonce_sha256 owner_hard_expires_at state send_stage
        prepared_at prepared_journal_entry_sha256 owner_nonce_ciphertext_sha256
        ddb_committed_journal_entry_sha256 gpu_liability_reserve_seconds
        gpu_liability_reserve_cost_usd ebs_liability_reserve_cost_usd
        residual_liability_approval_identity_sha256
        gpu_liability_reserve_ledger_identity_sha256 possibly_sent_at
        run_instances_attempt_evidence same_token_completion_count
        direct_run_instances_request_id direct_run_instances_response_sha256
        observed_instance_ids instance_observation_sha256
        spend_allocation_open_identity_sha256 instance_terminal_identity_sha256
        spend_allocation_close_identity_sha256 incident_identity_sha256 revision
        updated_at"""
    ),
    "glm52_production_worker_launch_liability": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal generation allocation_ordinal
        worker_launch_identity_sha256 ec2_client_token launch_parameters_sha256
        expected_worker_tags_sha256 state watch_schedule_arn
        watch_schedule_identity_sha256 scan_interval_seconds
        max_discovery_to_termination_seconds same_token_completion_attempts
        scan_count_current_approval_period scan_period_started_at scan_period_ends_at
        next_scan_at last_scan_started_at last_scan_completed_at
        last_scan_evidence_sha256 watch_started_at watch_not_before watch_not_after
        observed_instance_ids late_instance_drain_identities
        late_instance_termination_action_identities
        late_instance_termination_call_counts post_terminal_allocation_identities
        spend_close_identities retained_control_plane_budget_identity_sha256
        gpu_liability_reserve_seconds gpu_liability_reserve_cost_usd
        ebs_liability_reserve_cost_usd residual_liability_approval_identity_sha256
        gpu_liability_reserve_ledger_identity_sha256
        gpu_liability_reserve_release_identity_sha256 settlement_identity_sha256
        incident_identity_sha256 incident_at owner_attempt owner_execution_arn
        owner_state_machine_version_arn owner_dispatch_identity_sha256
        owner_invocation_nonce_sha256 owner_hard_expires_at revision updated_at"""
    ),
    "glm52_production_post_terminal_allocation": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal generation allocation_ordinal
        ec2_client_token worker_launch_identity_sha256
        worker_launch_liability_identity_sha256 terminal_v2_identity_sha256
        instance_id instance_tags_sha256 discovery_source discovered_at state
        allocation_open_identity_sha256 charged_interval_started_at
        instance_terminal_identity_sha256 charged_interval_ended_at
        spend_allocation_close_identity_sha256 concurrency_window_identity_sha256
        owner_attempt owner_execution_arn owner_state_machine_version_arn
        owner_dispatch_identity_sha256 owner_invocation_nonce_sha256
        owner_hard_expires_at revision updated_at canonical_body_sha256"""
    ),
    "glm52_production_worker_launch_liability_settlement": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal generation allocation_ordinal
        worker_launch_identity_sha256 worker_launch_liability_identity_sha256
        settlement_kind terminal_v2_key terminal_v2_version_id
        terminal_v2_body_sha256 terminal_v2_allocations_array_sha256
        post_terminal_allocations post_terminal_allocations_array_sha256
        merged_final_allocations merged_final_allocations_array_sha256
        final_worker_cardinality service_rejection_evidence_sha256
        instance_terminal_evidence_array_sha256 spend_close_evidence_array_sha256
        gpu_liability_reserve_ledger_identity_sha256
        gpu_liability_reserve_release_identity_sha256
        ebs_liability_reserve_cost_usd residual_liability_approval_identity_sha256
        final_spend_ledger_head_identity_sha256 prior_incident_identity_sha256
        owner_execution_arn owner_state_machine_version_arn
        owner_dispatch_identity_sha256 owner_invocation_nonce_sha256
        transaction_client_request_token_sha256 settled_at canonical_body_sha256"""
    ),
    "glm52_production_action": _fields(
        """schema_version record_type run_id campaign_identity_sha256 activation_id
        activation_ordinal generation generation_text action_kind attempt
        candidate_key candidate_file_sha256 candidate_body_sha256 request_body_sha256
        relay_envelope_sha256 owner_epoch owner_execution_arn armed_by_epoch
        armed_by_execution_arn armed_by_state_machine_version_arn
        armed_by_function_version_arn arming_dispatch_identity_sha256
        armed_by_invocation_nonce_sha256 owner_invocation_nonce_sha256
        post_owner_invocation_nonce_sha256 post_owner_function_version_arn
        post_owner_dispatch_identity_sha256 post_owner_hard_expires_at
        barrier_nonce_sha256 authority_audit_body_sha256
        authority_audit_closing_revision authorized_transition_from_revision
        authorized_transition_to_revision state armed_at arming_hard_expires_at
        consumed_at post_started_at post_authorized_at completed_at abandoned_at
        outcome_class classification_evidence_kind
        classification_evidence_body_sha256 sky_request_id
        response_identity_sha256 abandonment_proof_sha256
        arming_transaction_client_request_token_sha256
        consume_transaction_client_request_token_sha256
        post_start_transaction_client_request_token_sha256
        post_authorization_transaction_client_request_token_sha256
        post_classification_transaction_client_request_token_sha256 revision"""
    ),
    "glm52_production_terminal_v2": _fields(
        """schema_version record_type account_id region run_id campaign_identity_sha256
        activation_id activation_ordinal generation generation_text action_key
        action_identity_sha256 handoff binding final_sky_state final_ec2_states
        request_cardinality worker_cardinality allocations allocations_array_sha256
        worker_launch_evidence worker_launch_evidence_array_sha256
        worker_launch_liabilities worker_launch_liabilities_array_sha256
        spend_ledger_head_identity remaining_approved_gpu_seconds
        remaining_approved_gpu_usd final_heartbeat_identity checkpoint_identity
        cache_identity training_identity evaluation_identity drain_identity
        terminal_observation_window request_evidence request_evidence_array_sha256
        post_terminal_quiescence_evidence prior_terminal_v1_identity outcome
        operator_disposition_required writer_function_version_arn
        writer_dispatch_identity_sha256 writer_invocation_nonce_sha256 created_at
        canonical_body_sha256"""
    ),
}

_RETAINED_ACTION_FIELDS = _fields(
    """schema_version record_type authority_domain account_id region run_id
    campaign_identity_sha256 activation_id activation_ordinal action_kind attempt
    candidate_key candidate_body_sha256 request_body_sha256 owner_attempt
    owner_execution_arn owner_state_machine_version_arn owner_dispatch_identity_sha256
    owner_invocation_nonce_sha256 owner_hard_expires_at authority_barrier_nonce_sha256
    authority_audit_body_sha256 authority_audit_closing_revision
    authorized_transition_from_revision authorized_transition_to_revision state
    armed_at consumed_at completed_at response_identity_sha256
    reconciliation_identity_sha256 arming_transaction_client_request_token_sha256
    consume_transaction_client_request_token_sha256 revision generation
    generation_text allocation_ordinal allocation_ordinal_text
    worker_launch_identity_sha256 worker_launch_liability_identity_sha256"""
)
for _kind in (
    "glm52_production_recovery_action",
    "glm52_production_finalization_action",
    "glm52_production_snapshot_cleanup_action",
    "glm52_production_worker_launch_liability_action",
):
    RECORD_FIELDS[_kind] = _RETAINED_ACTION_FIELDS

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TS = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_ORD = re.compile(r"^[0-9]{8}$")
_INT_FIELDS = frozenset(
    _fields(
        """schema_version current_activation_ordinal activation_ordinal
        prior_activation_ordinal index_from_revision index_to_revision active_epoch
        last_sky_post_generation cleanup_transition_chain_length owner_attempt
        support_control_revision_at_seal teardown_sealed_control_revision revision
        delete_logical_attempt delete_call_count from_revision to_revision
        authority_audit_closing_revision authorized_transition_from_revision
        authorized_transition_to_revision attempt generation allocation_ordinal epoch
        owner_epoch armed_by_epoch owner_control_revision same_token_completion_count
        gpu_liability_reserve_seconds scan_interval_seconds
        max_discovery_to_termination_seconds same_token_completion_attempts
        scan_count_current_approval_period remaining_approved_gpu_seconds"""
    )
)
_ARRAY_FIELDS = frozenset({
    "snapshot_cleanup_lineage",
    "allocations",
    "request_evidence",
    "worker_launch_evidence",
    "worker_launch_liabilities",
    "final_ec2_states",
    "post_terminal_allocations",
    "merged_final_allocations",
    "run_instances_attempt_evidence",
    "cleanup_authority_audit_identities",
    "delete_action_identities",
    "observed_instance_ids",
    "late_instance_drain_identities",
    "late_instance_termination_action_identities",
    "late_instance_termination_call_counts",
    "post_terminal_allocation_identities",
    "spend_close_identities",
    "request_ids",
    "job_ids",
})
_TIMESTAMP_FIELDS = frozenset(
    field
    for fields in RECORD_FIELDS.values()
    for field in fields
    if field.endswith("_at") or field.endswith("_not_before") or field.endswith("_not_after")
)
_POSITIVE_INT_FIELDS = frozenset(
    {
        "current_activation_ordinal", "activation_ordinal", "prior_activation_ordinal",
        "epoch", "generation", "allocation_ordinal", "attempt", "owner_attempt",
    }
)
_PRE_GENESIS_ACTION_KINDS = frozenset()
_OBJECT_FIELDS = frozenset(
    {
        "prior_activation_terminal_v2_identity",
        "prior_h1g_drained_identity",
        "prior_spend_ledger_head_identity",
        "terminal_v2_identity",
        "operator_disposition_identity",
        "prior_snapshot_cleanup_control_identity",
        "cached_success_response_body",
        "direct_start_response_identity",
        "spend_ledger_head_identity",
        "handoff",
        "binding",
        "final_heartbeat_identity",
        "checkpoint_identity",
        "cache_identity",
        "training_identity",
        "evaluation_identity",
        "drain_identity",
        "terminal_observation_window",
        "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
        "support_plane_finalized_identity",
        "support_stack_deletion_identity",
        "kms_grant_baseline_equality_identity",
        "snapshot_cleanup_schedule_identity",
        "retained_resource_inventory_identity",
        "request_states",
        "job_states",
    }
)

_H1G_TERMINAL_BUCKET = "keep-glm52-models-246813579024-us-west-2"
_H1G_TERMINAL_KEY = (
    "campaigns/glm52-sky-20260724/submissions/production/generations/"
    "00000001/terminal/PRODUCTION_TERMINAL_V2.json"
)
_H1G_SPEND_BUCKET = "keep-glm52-models-246813579024-us-west-2"
_H1G_SPEND_KEY = (
    "campaigns/glm52-sky-20260724/runtime/GPU_SPEND_LEDGER.jsonl"
)
_H1G_RETAINED_RESOURCE_TYPES = frozenset(
    {
        "LEDGER",
        "KMS_KEY",
        "EVIDENCE_BUCKET",
        "PRODUCTION_FENCE_STACK",
        "LIFECYCLE_RESOURCE",
        "SOURCE_PUBLISHER_IDENTITY",
    }
)
_H1G_RETAINED_PRICE_TERMS = frozenset(
    {
        "retained_s3",
        "retained_ledger",
        "retained_kms",
        "liability_watcher",
        "snapshot_cleanup",
        "snapshot_retention",
    }
)
_H1G_PRICE_TERM_FIELDS = frozenset(
    {
        "term",
        "unit",
        "quantity",
        "unit_price_usd",
        "estimated_usd",
    }
)


def _validate_h1g_identity(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Dict[str, object]:
    if type(value) is not dict or set(value) != fields | {
        "canonical_identity_sha256"
    }:
        raise RecordValidationError(label + " field set mismatch")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if type(identity) is not str or _SHA.fullmatch(identity) is None:
        raise RecordValidationError(label + " identity must be lowercase SHA-256")
    if identity != canonical_sha256(body):
        raise RecordValidationError(label + " canonical identity mismatch")
    return dict(value)


def _h1g_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise RecordValidationError(label + " must be lowercase SHA-256")
    return value


def _h1g_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or any(ord(char) < 0x21 or ord(char) > 0x7E for char in value)
    ):
        raise RecordValidationError(label + " must be nonempty printable ASCII")
    return value


def _h1g_version(value: object, label: str) -> str:
    exact = _h1g_text(value, label)
    if exact.lower() in {"latest", "$latest", "null"}:
        raise RecordValidationError(label + " must be one immutable VersionId")
    return exact


def _validate_h1g_drained(record: Mapping[str, object]) -> None:
    terminal = _validate_h1g_identity(
        record["terminal_v2_identity"],
        fields=frozenset(
            {
                "generation",
                "generation_text",
                "bucket",
                "key",
                "version_id",
                "file_sha256",
                "body_sha256",
            }
        ),
        label="H1G terminal-v2 identity",
    )
    if (
        terminal["generation"] != 1
        or terminal["generation_text"] != "00000001"
        or terminal["bucket"] != _H1G_TERMINAL_BUCKET
        or terminal["key"] != _H1G_TERMINAL_KEY
    ):
        raise RecordValidationError("H1G terminal-v2 coordinate mismatch")
    _h1g_version(terminal["version_id"], "H1G terminal-v2 VersionId")
    _h1g_sha(terminal["file_sha256"], "H1G terminal-v2 file identity")
    _h1g_sha(terminal["body_sha256"], "H1G terminal-v2 body identity")

    support = _validate_h1g_identity(
        record["support_plane_finalized_identity"],
        fields=frozenset(
            {
                "finalization_control_identity_sha256",
                "support_plane_finalized_identity_sha256",
            }
        ),
        label="H1G support-plane-finalized identity",
    )
    for field in (
        "finalization_control_identity_sha256",
        "support_plane_finalized_identity_sha256",
    ):
        _h1g_sha(support[field], "H1G " + field)

    deletion = _validate_h1g_identity(
        record["support_stack_deletion_identity"],
        fields=frozenset({"stack_id", "observation", "request_id"}),
        label="H1G support-stack deletion identity",
    )
    if (
        not str(deletion["stack_id"]).startswith(
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-support/"
        )
        or deletion["observation"]
        not in {"ABSENT_VALIDATION_ERROR", "DELETE_COMPLETE"}
    ):
        raise RecordValidationError("H1G support-stack deletion proof mismatch")
    _h1g_text(deletion["request_id"], "H1G support-stack request ID")

    for field in (
        "orphan_audit_identity",
        "snapshot_cleanup_control_identity",
    ):
        _h1g_sha(record[field], "H1G " + field)

    grant = _validate_h1g_identity(
        record["kms_grant_baseline_equality_identity"],
        fields=frozenset(
            {
                "baseline_identity_sha256",
                "final_grants_identity_sha256",
                "final_list_grants_identity_sha256",
            }
        ),
        label="H1G KMS baseline equality identity",
    )
    for field in (
        "baseline_identity_sha256",
        "final_grants_identity_sha256",
        "final_list_grants_identity_sha256",
    ):
        _h1g_sha(grant[field], "H1G " + field)
    if grant["baseline_identity_sha256"] != grant["final_grants_identity_sha256"]:
        raise RecordValidationError("H1G KMS grant baseline was not restored")

    schedule = _validate_h1g_identity(
        record["snapshot_cleanup_schedule_identity"],
        fields=frozenset({"schedule_arn", "schedule_input_sha256"}),
        label="H1G snapshot-cleanup schedule identity",
    )
    if not str(schedule["schedule_arn"]).startswith(
        "arn:aws:scheduler:us-west-2:246813579024:schedule/"
    ):
        raise RecordValidationError("H1G snapshot-cleanup schedule ARN mismatch")
    _h1g_sha(
        schedule["schedule_input_sha256"],
        "H1G snapshot-cleanup schedule input",
    )

    spend = _validate_h1g_identity(
        record["spend_ledger_head_identity"],
        fields=frozenset(
            {
                "bucket",
                "key",
                "version_id",
                "file_sha256",
                "body_sha256",
                "head_record_sha256",
            }
        ),
        label="H1G spend-ledger head identity",
    )
    if spend["bucket"] != _H1G_SPEND_BUCKET or spend["key"] != _H1G_SPEND_KEY:
        raise RecordValidationError("H1G spend-ledger coordinate mismatch")
    _h1g_version(spend["version_id"], "H1G spend-ledger VersionId")
    for field in ("file_sha256", "body_sha256", "head_record_sha256"):
        _h1g_sha(spend[field], "H1G spend-ledger " + field)

    liabilities = record["worker_launch_liabilities"]
    _validate_terminal_liabilities(liabilities)
    inventory = _validate_h1g_identity(
        record["retained_resource_inventory_identity"],
        fields=frozenset(
            {
                "retained_resources",
                "retained_cost_classes",
                "price_card_identity_sha256",
                "retained_terms",
                "controller_quiescence_identity_sha256",
                "marker_last_prerequisite_identity_sha256",
            }
        ),
        label="H1G retained-resource inventory identity",
    )
    resources = inventory["retained_resources"]
    cost_classes = inventory["retained_cost_classes"]
    if (
        type(resources) is not list
        or len(resources) != len(_H1G_RETAINED_RESOURCE_TYPES)
        or any(
            type(item) is not dict
            or set(item) != {"resource_type", "resource_id", "cost_class"}
            or type(item["resource_type"]) is not str
            or type(item["resource_id"]) is not str
            or not item["resource_id"]
            or type(item["cost_class"]) is not str
            or not item["cost_class"]
            for item in resources
        )
        or {item["resource_type"] for item in resources}
        != _H1G_RETAINED_RESOURCE_TYPES
        or type(cost_classes) is not list
        or cost_classes != [item["cost_class"] for item in resources]
    ):
        raise RecordValidationError("H1G retained-resource inventory mismatch")
    _h1g_sha(
        inventory["price_card_identity_sha256"],
        "H1G retained price-card identity",
    )
    _h1g_sha(
        inventory["controller_quiescence_identity_sha256"],
        "H1G controller quiescence identity",
    )
    terms = inventory["retained_terms"]
    if (
        type(terms) is not list
        or len(terms) != len(_H1G_RETAINED_PRICE_TERMS)
        or any(
            type(item) is not dict
            or set(item) != _H1G_PRICE_TERM_FIELDS
            or any(type(value) is not str or not value for value in item.values())
            for item in terms
        )
        or {item["term"] for item in terms} != _H1G_RETAINED_PRICE_TERMS
    ):
        raise RecordValidationError("H1G retained price terms mismatch")

    liability_state = (
        "SETTLED" if not liabilities else "RETAINED_TERMINATION_ONLY"
    )
    prerequisite_body = {
        "terminal_v2_identity_sha256": terminal["body_sha256"],
        "finalization_identity_sha256": support[
            "finalization_control_identity_sha256"
        ],
        "snapshot_cleanup_control_identity_sha256": record[
            "snapshot_cleanup_control_identity"
        ],
        "controller_quiesced_identity_sha256": inventory[
            "controller_quiescence_identity_sha256"
        ],
        "spend_ledger_head_identity_sha256": spend["head_record_sha256"],
        "orphan_audit_identity_sha256": record["orphan_audit_identity"],
        "liability_state": liability_state,
        "liability_identity_sha256": canonical_sha256(liabilities),
        "marker_write_order": "LAST_CONDITIONAL_CREATE",
    }
    if inventory["marker_last_prerequisite_identity_sha256"] != canonical_sha256(
        prerequisite_body
    ):
        raise RecordValidationError("H1G marker-last prerequisite identity mismatch")


def validate_record(
    record_type: str,
    value: object,
    *,
    pk: Optional[str] = None,
    sk: Optional[str] = None,
) -> Dict[str, object]:
    """Validate a closed durable-record body."""

    if record_type not in RECORD_FIELDS:
        raise RecordValidationError("unknown record type")
    if type(value) is not dict:
        raise RecordValidationError("record must be a JSON object")
    expected = set(RECORD_FIELDS[record_type])
    actual = set(value)
    if actual != expected:
        raise RecordValidationError(
            "schema mismatch: missing=%r, unknown=%r"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    record = dict(value)
    expected_version = 2 if record_type == "glm52_production_terminal_v2" else 1
    if type(record["schema_version"]) is not int or record["schema_version"] != expected_version:
        raise RecordValidationError("schema_version must be an exact integer")
    if record["record_type"] != record_type or type(record["record_type"]) is not str:
        raise RecordValidationError("record_type mismatch")
    for field, constant in (
        ("account_id", ACCOUNT_ID), ("region", REGION), ("run_id", RUN_ID)
    ):
        if field in record and (
            type(record[field]) is not str or record[field] != constant
        ):
            raise RecordValidationError("%s mismatch" % field)
    for field, item in record.items():
        if field in _ARRAY_FIELDS and type(item) is not list:
            raise RecordValidationError("%s must be an exact array" % field)
        if item is None:
            continue
        if field in _OBJECT_FIELDS and type(item) is not dict:
            raise RecordValidationError("%s must be an exact object" % field)
        if field in _INT_FIELDS and type(item) is not int:
            raise RecordValidationError("%s must be an exact integer" % field)
        if field in _POSITIVE_INT_FIELDS and type(item) is int and item <= 0:
            if not (
                field == "generation"
                and record_type == "glm52_production_action"
                and record.get("action_kind") in _PRE_GENESIS_ACTION_KINDS
                and item == 0
            ):
                raise RecordValidationError("%s must be positive" % field)
        if (
            field.endswith("_sha256")
            or field.endswith("_body_sha256")
        ) and (type(item) is not str or _SHA.fullmatch(item) is None):
            raise RecordValidationError("%s must be lowercase SHA-256" % field)
        if field in _TIMESTAMP_FIELDS and (
            type(item) is not str or _TS.fullmatch(item) is None
        ):
            raise RecordValidationError("%s must be canonical UTC" % field)
        if field in _TIMESTAMP_FIELDS and type(item) is str:
            try:
                parsed = datetime.strptime(item, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError as exc:
                raise RecordValidationError(
                    "%s must be canonical UTC" % field
                ) from exc
            if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != item:
                raise RecordValidationError("%s must be canonical UTC" % field)
        if field.endswith("_text") and field in {
            "generation_text", "allocation_ordinal_text", "epoch_text"
        }:
            if type(item) is not str or _ORD.fullmatch(item) is None:
                raise RecordValidationError("%s must be fixed-width ordinal" % field)
            number_field = field[:-5]
            if record.get(number_field) != int(item):
                raise RecordValidationError("%s ordinal mismatch" % field)
        if (
            field not in _INT_FIELDS
            and field not in _ARRAY_FIELDS
            and field not in _OBJECT_FIELDS
            and field != "operator_disposition_required"
            and type(item) is not str
        ):
            raise RecordValidationError("%s must be an exact string" % field)
    if "canonical_body_sha256" in record:
        expected_hash = canonical_record_identity_unchecked(record)
        if record["canonical_body_sha256"] != expected_hash:
            raise RecordValidationError("canonical body SHA-256 mismatch")
    for array_field in _ARRAY_FIELDS:
        hash_field = array_field + "_array_sha256"
        if array_field in record and hash_field in record:
            if record[hash_field] != canonical_sha256(record[array_field]):
                raise RecordValidationError("%s mismatch" % hash_field)
    if "snapshot_cleanup_lineage" in record and (
        record.get("snapshot_cleanup_lineage_sha256")
        != canonical_sha256(record["snapshot_cleanup_lineage"])
    ):
        raise RecordValidationError("snapshot cleanup lineage SHA-256 mismatch")
    if record_type == "glm52_production_rollover":
        _validate_rollover(record)
    if record_type == "glm52_production_terminal_v2":
        _validate_terminal_v2(record)
    if record_type == "glm52_production_h1g_drained":
        _validate_h1g_drained(record)
    if record_type in (
        "glm52_production_worker_launch",
        "glm52_production_worker_launch_liability",
    ):
        if record["gpu_liability_reserve_seconds"] != 900:
            raise RecordValidationError("GPU reserve seconds must equal 900")
        if record["gpu_liability_reserve_cost_usd"] != "13.76":
            raise RecordValidationError("GPU reserve price must equal 13.76")
        if record["ebs_liability_reserve_cost_usd"] != "0.01":
            raise RecordValidationError("EBS reserve price must equal 0.01")
    if record_type == "glm52_production_worker_launch" and (
        record["generation"] <= 0 or record["allocation_ordinal"] <= 0
    ):
        raise RecordValidationError("worker launch ordinals must be positive")
    if record_type == "glm52_production_worker_launch_liability_settlement":
        if record["ebs_liability_reserve_cost_usd"] != "0.01":
            raise RecordValidationError("EBS reserve price must equal 0.01")
    if "ec2_client_token" in record:
        token = record["ec2_client_token"]
        if (
            type(token) is not str or len(token) != 64
            or any(ord(char) > 127 for char in token)
        ):
            raise RecordValidationError("EC2 ClientToken must be 64 ASCII characters")
    for sorted_field in (
        "observed_instance_ids",
        "final_ec2_states",
        "request_ids",
    ):
        if sorted_field in record and record[sorted_field] is not None:
            items = record[sorted_field]
            if items != sorted(items) or len(items) != len(set(items)):
                raise RecordValidationError("%s must be sorted and duplicate-free" % sorted_field)
    if record_type == "glm52_task12_request_job_correlation_v1":
        request_ids = record["request_ids"]
        job_ids = record["job_ids"]
        if (
            not request_ids
            or any(type(item) is not str or not item for item in request_ids)
            or any(
                type(item) is not str
                or re.fullmatch(r"[1-9][0-9]*", item) is None
                for item in job_ids
            )
            or job_ids != sorted(job_ids, key=int)
            or len(job_ids) != len(set(job_ids))
            or set(record["request_states"]) != set(request_ids)
            or set(record["job_states"]) != set(job_ids)
            or any(
                type(state) is not str or not state
                for state in (
                    list(record["request_states"].values())
                    + list(record["job_states"].values())
                )
            )
        ):
            raise RecordValidationError(
                "request/job correlation set is not exact"
            )
    if "operator_disposition_required" in record and (
        type(record["operator_disposition_required"]) is not bool
    ):
        raise RecordValidationError("operator_disposition_required must be boolean")
    _validate_enums(record_type, record)
    _validate_retained_action(record_type, record)
    _validate_state_nullability(record_type, record)
    for field in (
        "activation_id", "current_activation_id", "prior_activation_id",
        "instance_id",
    ):
        if field in record and record[field] is not None:
            _validate_key_segment(field, record[field])
    if "action_kind" in record:
        action_kind = record["action_kind"]
        if (
            type(action_kind) is not str
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", action_kind) is None
        ):
            raise RecordValidationError("action_kind is not a closed key segment")
    if pk is not None and pk != ledger_pk(record["run_id"]):
        raise RecordValidationError("ledger key partition mismatch")
    if sk is not None and sk != _record_ledger_sk(record_type, record):
        raise RecordValidationError("ledger key sort mismatch")
    return record


def canonical_record_identity_unchecked(value: Mapping[str, object]) -> str:
    body = dict(value)
    body.pop("canonical_body_sha256", None)
    return canonical_sha256(body)


def _validate_rollover(record: Mapping[str, object]) -> None:
    if record["index_to_revision"] != record["index_from_revision"] + 1:
        raise RecordValidationError("rollover index revision must increment")
    prior_scalars = (
        "prior_activation_id", "prior_activation_ordinal",
        "prior_terminal_v2_key", "prior_terminal_v2_version_id",
        "prior_terminal_v2_body_sha256", "prior_terminal_v2_outcome",
        "prior_final_worker_cardinality", "prior_h1g_drained_key",
        "prior_h1g_drained_version_id", "prior_h1g_drained_body_sha256",
        "prior_spend_ledger_head_identity",
        "prior_snapshot_cleanup_control_identity",
    )
    prior_hashes = (
        "prior_allocations_array_sha256",
        "prior_request_evidence_array_sha256",
        "prior_worker_launch_evidence_array_sha256",
        "prior_worker_launch_liabilities_array_sha256",
        "prior_worker_launch_liability_settlements_array_sha256",
        "prior_post_terminal_allocations_array_sha256",
        "prior_merged_final_allocations_array_sha256",
    )
    if record["activation_ordinal"] == 1:
        if any(record[field] is not None for field in prior_scalars):
            raise RecordValidationError("first rollover requires exact-null predecessors")
        if any(record[field] != canonical_sha256([]) for field in prior_hashes):
            raise RecordValidationError("first rollover requires empty predecessor arrays")
        if record["snapshot_cleanup_lineage"] != []:
            raise RecordValidationError("first rollover requires empty cleanup lineage")
        if (
            record["operator_disposition_required"] is not False
            or record["operator_disposition_identity"] is not None
        ):
            raise RecordValidationError("first rollover forbids disposition")
    else:
        if type(record["prior_activation_ordinal"]) is not int or (
            record["prior_activation_ordinal"] + 1 != record["activation_ordinal"]
        ):
            raise RecordValidationError("later rollover ordinal mismatch")
        if any(record[field] is None for field in prior_scalars):
            raise RecordValidationError("later rollover requires predecessor identities")
        lineage = record["snapshot_cleanup_lineage"]
        expected_ordinals = list(range(1, record["prior_activation_ordinal"] + 1))
        if [entry.get("activation_ordinal") for entry in lineage if type(entry) is dict] != expected_ordinals:
            raise RecordValidationError("cleanup lineage must be complete and transitive")
        expected_entry_fields = {
            "activation_ordinal", "cleanup_control_root_identity_sha256",
            "cleanup_transition_chain_head_sha256",
            "cleanup_transition_chain_length", "transition_identities",
        }
        for entry in lineage:
            if type(entry) is not dict or set(entry) != expected_entry_fields:
                raise RecordValidationError("cleanup lineage entry schema mismatch")
            transitions = entry["transition_identities"]
            if type(transitions) is not list or (
                entry["cleanup_transition_chain_length"] != len(transitions)
            ):
                raise RecordValidationError("cleanup lineage chain length mismatch")
            if transitions and entry["cleanup_transition_chain_head_sha256"] != transitions[-1]:
                raise RecordValidationError("cleanup lineage head mismatch")
            for field in (
                "cleanup_control_root_identity_sha256",
                "cleanup_transition_chain_head_sha256",
            ):
                if type(entry[field]) is not str or _SHA.fullmatch(entry[field]) is None:
                    raise RecordValidationError("cleanup lineage SHA-256 malformed")
            if any(type(item) is not str or _SHA.fullmatch(item) is None for item in transitions):
                raise RecordValidationError("cleanup lineage transition malformed")


_OUTCOMES = frozenset(
    """DRAINED_COMPLETED DRAINED_TRAINING_DEFERRED DRAINED_RESUMABLE_DEADLINE
    STORED_DECISION_NO_POST CONSUMED_PROVED_NO_POST KNOWN_REJECTED_NO_JOB
    AMBIGUOUS_PROVED_NO_JOB ACCEPTED_REQUEST_FAILED_NO_JOB
    ACCEPTED_REQUEST_CANCELLED_NO_JOB FAILED_TERMINAL_BEFORE_ALLOCATION
    FAILED_TERMINAL_AFTER_ALLOCATION SUPPORT_EXECUTION_START_UNRESOLVED_INCIDENT
    UNBOUND_LATE_WORKER_INCIDENT MULTIPLE_UNBOUND_WORKERS_INCIDENT
    MULTIPLE_BOUND_WORKERS_INCIDENT WORKER_LAUNCH_UNRESOLVED_NO_INSTANCE_INCIDENT
    WORKER_LAUNCH_UNRESOLVED_WITH_ALLOCATIONS_INCIDENT
    WORKER_LAUNCH_MULTIPLE_INSTANCE_TOKEN_INCIDENT
    QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT
    QUIESCED_UNRESOLVED_REQUEST_WITH_ALLOCATIONS_INCIDENT
    QUIESCED_UNRESOLVED_BOUND_JOB_NO_ALLOCATION_INCIDENT
    QUIESCED_UNRESOLVED_BOUND_JOB_WITH_ALLOCATIONS_INCIDENT
    ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH EXPIRED_UNSTARTED""".split()
)
_NO_ALLOCATION = frozenset(
    """FAILED_TERMINAL_BEFORE_ALLOCATION SUPPORT_EXECUTION_START_UNRESOLVED_INCIDENT
    CONSUMED_PROVED_NO_POST KNOWN_REJECTED_NO_JOB AMBIGUOUS_PROVED_NO_JOB
    ACCEPTED_REQUEST_FAILED_NO_JOB ACCEPTED_REQUEST_CANCELLED_NO_JOB
    WORKER_LAUNCH_UNRESOLVED_NO_INSTANCE_INCIDENT
    QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT
    QUIESCED_UNRESOLVED_BOUND_JOB_NO_ALLOCATION_INCIDENT STORED_DECISION_NO_POST
    ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH EXPIRED_UNSTARTED""".split()
)
_ONE_OR_MORE = frozenset(
    """DRAINED_COMPLETED DRAINED_TRAINING_DEFERRED DRAINED_RESUMABLE_DEADLINE
    FAILED_TERMINAL_AFTER_ALLOCATION UNBOUND_LATE_WORKER_INCIDENT
    WORKER_LAUNCH_UNRESOLVED_WITH_ALLOCATIONS_INCIDENT
    QUIESCED_UNRESOLVED_REQUEST_WITH_ALLOCATIONS_INCIDENT
    QUIESCED_UNRESOLVED_BOUND_JOB_WITH_ALLOCATIONS_INCIDENT""".split()
)
_MULTIPLE = frozenset(
    """MULTIPLE_UNBOUND_WORKERS_INCIDENT MULTIPLE_BOUND_WORKERS_INCIDENT
    WORKER_LAUNCH_MULTIPLE_INSTANCE_TOKEN_INCIDENT""".split()
)
_NOT_APPLICABLE = frozenset(
    """STORED_DECISION_NO_POST CONSUMED_PROVED_NO_POST EXPIRED_UNSTARTED
    SUPPORT_EXECUTION_START_UNRESOLVED_INCIDENT
    ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH""".split()
)
_ZERO_REQUEST = frozenset("KNOWN_REJECTED_NO_JOB AMBIGUOUS_PROVED_NO_JOB".split())
_ONE_REQUEST = frozenset(
    """ACCEPTED_REQUEST_FAILED_NO_JOB ACCEPTED_REQUEST_CANCELLED_NO_JOB
    DRAINED_COMPLETED DRAINED_TRAINING_DEFERRED DRAINED_RESUMABLE_DEADLINE
    FAILED_TERMINAL_BEFORE_ALLOCATION FAILED_TERMINAL_AFTER_ALLOCATION""".split()
)

_TERMINAL_MARKERS = (
    "final_heartbeat_identity", "checkpoint_identity", "cache_identity",
    "training_identity", "evaluation_identity", "drain_identity",
)


def _terminal_rule(
    handoff: str,
    binding: str,
    allocations: str,
    request: Tuple[str, ...],
    disposition: bool,
    *,
    marker_required: Tuple[str, ...] = (),
    marker_null: Tuple[str, ...] = (),
    prior_terminal_v1: str = "null",
) -> Dict[str, object]:
    marker_optional = tuple(
        field
        for field in _TERMINAL_MARKERS
        if field not in marker_required and field not in marker_null
    )
    return {
        "handoff": handoff,
        "binding": binding,
        "allocations": allocations,
        "request": frozenset(request),
        "disposition": disposition,
        "marker_required": marker_required,
        "marker_null": marker_null,
        "marker_optional": marker_optional,
        "prior_terminal_v1": prior_terminal_v1,
    }


_TERMINAL_OUTCOME_RULES: Dict[str, Dict[str, object]] = {
    "DRAINED_COMPLETED": _terminal_rule(
        "nonnull", "nonnull", "nonempty", ("ONE",), False,
        marker_required=("training_identity", "evaluation_identity", "drain_identity"),
    ),
    "DRAINED_TRAINING_DEFERRED": _terminal_rule(
        "nonnull", "nonnull", "nonempty", ("ONE",), False,
        marker_required=("checkpoint_identity", "cache_identity", "drain_identity"),
        marker_null=("training_identity", "evaluation_identity"),
    ),
    "DRAINED_RESUMABLE_DEADLINE": _terminal_rule(
        "nonnull", "nonnull", "nonempty", ("ONE",), False,
        marker_required=("checkpoint_identity", "drain_identity"),
    ),
    "STORED_DECISION_NO_POST": _terminal_rule(
        "null", "null", "zero", ("NOT_APPLICABLE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "CONSUMED_PROVED_NO_POST": _terminal_rule(
        "nonnull", "null", "zero", ("NOT_APPLICABLE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "KNOWN_REJECTED_NO_JOB": _terminal_rule(
        "nonnull", "null", "zero", ("ZERO",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "AMBIGUOUS_PROVED_NO_JOB": _terminal_rule(
        "nonnull", "null", "zero", ("ZERO",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "ACCEPTED_REQUEST_FAILED_NO_JOB": _terminal_rule(
        "nonnull", "null", "zero", ("ONE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "ACCEPTED_REQUEST_CANCELLED_NO_JOB": _terminal_rule(
        "nonnull", "null", "zero", ("ONE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "FAILED_TERMINAL_BEFORE_ALLOCATION": _terminal_rule(
        "nonnull", "nonnull", "zero", ("ONE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "FAILED_TERMINAL_AFTER_ALLOCATION": _terminal_rule(
        "nonnull", "nonnull", "nonempty", ("ONE",), True,
    ),
    "SUPPORT_EXECUTION_START_UNRESOLVED_INCIDENT": _terminal_rule(
        "null", "null", "zero", ("NOT_APPLICABLE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "UNBOUND_LATE_WORKER_INCIDENT": _terminal_rule(
        "nonnull", "null", "one", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "MULTIPLE_UNBOUND_WORKERS_INCIDENT": _terminal_rule(
        "nonnull", "null", "multiple", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "MULTIPLE_BOUND_WORKERS_INCIDENT": _terminal_rule(
        "nonnull", "nonnull", "multiple", ("ONE", "MULTIPLE"), True,
    ),
    "WORKER_LAUNCH_UNRESOLVED_NO_INSTANCE_INCIDENT": _terminal_rule(
        "nonnull", "optional", "zero", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "WORKER_LAUNCH_UNRESOLVED_WITH_ALLOCATIONS_INCIDENT": _terminal_rule(
        "nonnull", "optional", "nonempty", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "WORKER_LAUNCH_MULTIPLE_INSTANCE_TOKEN_INCIDENT": _terminal_rule(
        "nonnull", "optional", "multiple", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT": _terminal_rule(
        "nonnull", "null", "zero", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "QUIESCED_UNRESOLVED_REQUEST_WITH_ALLOCATIONS_INCIDENT": _terminal_rule(
        "nonnull", "null", "nonempty", ("ZERO", "ONE", "MULTIPLE"), True,
    ),
    "QUIESCED_UNRESOLVED_BOUND_JOB_NO_ALLOCATION_INCIDENT": _terminal_rule(
        "nonnull", "nonnull", "zero", ("ONE", "MULTIPLE"), True,
    ),
    "QUIESCED_UNRESOLVED_BOUND_JOB_WITH_ALLOCATIONS_INCIDENT": _terminal_rule(
        "nonnull", "nonnull", "nonempty", ("ONE", "MULTIPLE"), True,
    ),
    "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH": _terminal_rule(
        "null", "null", "zero", ("NOT_APPLICABLE",), True,
        marker_null=_TERMINAL_MARKERS,
    ),
    "EXPIRED_UNSTARTED": _terminal_rule(
        "null", "null", "zero", ("NOT_APPLICABLE",), True,
        marker_null=_TERMINAL_MARKERS, prior_terminal_v1="nonnull",
    ),
}

_ALLOCATION_ENTRY_FIELDS = _fields(
    """allocation_ordinal allocation_ordinal_text instance_id
    instance_tags_sha256 allocation_open_identity_sha256
    allocation_close_identity_sha256 charged_interval_started_at
    charged_interval_ended_at terminal_state instance_terminal_identity_sha256
    concurrency_window_identity_sha256 canonical_entry_sha256"""
)
_OWNER_HISTORY_ENTRY_FIELDS = _fields(
    """owner_attempt owner_principal_arn owner_function_version_arn
    owner_dispatch_identity_sha256 owner_invocation_nonce_sha256
    owner_hard_expires_at canonical_entry_sha256"""
)
_RUN_INSTANCES_ATTEMPT_ENTRY_FIELDS = _fields(
    """attempt attempted_at outcome request_id response_sha256
    transport_error_sha256 canonical_entry_sha256"""
)
_WORKER_LAUNCH_EVIDENCE_ENTRY_FIELDS = _fields(
    """kind allocation_ordinal allocation_ordinal_text ec2_client_token
    launch_parameters_sha256 expected_worker_tags_sha256
    worker_launch_identity_sha256
    prior_worker_launch_identity_sha256 prior_instance_terminal_identity_sha256
    prior_spend_allocation_close_identity_sha256 prepared_journal_entry_sha256
    ddb_committed_journal_entry_sha256 owner_history send_stage prepared_at
    possibly_sent_at run_instances_attempt_evidence direct_run_instances_request_id
    direct_run_instances_response_sha256 cloudtrail_lookup_evidence_sha256
    describe_instances_evidence_sha256 observed_instance_ids
    instance_observation_sha256 spend_allocation_open_identity_sha256
    instance_terminal_identity_sha256 spend_allocation_close_identity_sha256
    incident_identity_sha256 worker_launch_liability_identity_sha256
    abandoned_not_sent_proof_sha256 positive_service_rejection_evidence_sha256
    no_call_in_flight_evidence_sha256 canonical_entry_sha256"""
)
_WORKER_LAUNCH_LIABILITY_ENTRY_FIELDS = _fields(
    """activation_id activation_ordinal generation allocation_ordinal
    allocation_ordinal_text worker_launch_identity_sha256
    worker_launch_liability_identity_sha256 ec2_client_token
    launch_parameters_sha256 expected_worker_tags_sha256 state
    watch_schedule_arn watch_schedule_identity_sha256 watch_not_before
    watch_not_after observed_instance_ids last_scan_evidence_sha256
    incident_identity_sha256 owner_attempt owner_execution_arn
    owner_state_machine_version_arn owner_dispatch_identity_sha256
    owner_invocation_nonce_sha256 owner_hard_expires_at revision
    canonical_entry_sha256"""
)
_REQUEST_EVIDENCE_ENTRY_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "ZERO_MATCH_SCAN": _fields(
        """kind correlation_tuple_sha256 database_lower_bound
        database_upper_bound journal_lower_bound journal_upper_bound
        pagination_complete match_count scan_started_at scan_completed_at
        database_head_identity_sha256 journal_head_identity_sha256
        scheduler_snapshot_identity_sha256 worker_snapshot_identity_sha256
        allocation_snapshot_identity_sha256 canonical_entry_sha256"""
    ),
    "EXACT_REQUEST": _fields(
        """kind request_id correlation_tuple_sha256 stored_request_body_sha256
        stored_arguments_sha256 request_state request_created_at request_updated_at
        scheduler_work_identity_sha256 terminal_observation_identity_sha256
        canonical_entry_sha256"""
    ),
    "MULTIPLE_MATCH": _fields(
        """kind request_ids correlation_tuple_sha256 complete_member_set_sha256
        scan_identity_sha256 canonical_entry_sha256"""
    ),
    "REQUEST_CANCEL": _fields(
        """kind action_key action_identity_sha256 response_identity_sha256
        correlation_identity_sha256 terminal_observation_identity_sha256
        requested_at terminal_observed_at canonical_entry_sha256"""
    ),
    "JOB_CANCEL": _fields(
        """kind action_key action_identity_sha256 response_identity_sha256
        correlation_identity_sha256 terminal_observation_identity_sha256
        requested_at terminal_observed_at canonical_entry_sha256"""
    ),
    "QUIESCENCE_SNAPSHOT": _fields(
        """kind scan_started_at scan_completed_at request_scan_identity_sha256
        job_scan_identity_sha256 worker_scan_identity_sha256
        allocation_scan_identity_sha256 controller_scan_identity_sha256
        canonical_entry_sha256"""
    ),
}


def _validate_nested_timestamp(field: str, value: object) -> None:
    if type(value) is not str or _TS.fullmatch(value) is None:
        raise RecordValidationError(field + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise RecordValidationError(field + " must be canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise RecordValidationError(field + " must be canonical UTC")


def _validate_nested_sha(field: str, value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise RecordValidationError(field + " must be lowercase SHA-256")


def _validate_nonempty_ascii(
    field: str, value: object, *, nullable: bool = False
) -> None:
    if nullable and value is None:
        return
    if (
        type(value) is not str
        or not value
        or any(ord(char) > 127 for char in value)
    ):
        raise RecordValidationError(field + " must be a nonempty ASCII string")


def _validate_nested_entry(
    entry: object, fields: Tuple[str, ...], label: str
) -> Mapping[str, object]:
    if type(entry) is not dict or set(entry) != set(fields):
        raise RecordValidationError(label + " entry schema mismatch")
    expected_hash = canonical_sha256(
        {field: entry[field] for field in fields if field != "canonical_entry_sha256"}
    )
    if entry["canonical_entry_sha256"] != expected_hash:
        raise RecordValidationError(label + " entry canonical hash mismatch")
    return entry


def _validate_ordinal_pair(
    entry: Mapping[str, object], label: str
) -> None:
    ordinal = entry["allocation_ordinal"]
    text = entry["allocation_ordinal_text"]
    if (
        type(ordinal) is not int
        or ordinal <= 0
        or type(text) is not str
        or _ORD.fullmatch(text) is None
        or int(text) != ordinal
    ):
        raise RecordValidationError(label + " ordinal mismatch")


def _validate_terminal_allocations(entries: object) -> None:
    if type(entries) is not list:
        raise RecordValidationError("allocations must be an exact array")
    keys = []
    for raw in entries:
        entry = _validate_nested_entry(
            raw, _ALLOCATION_ENTRY_FIELDS, "allocation"
        )
        _validate_ordinal_pair(entry, "allocation")
        for field in (
            "instance_tags_sha256", "allocation_open_identity_sha256",
            "allocation_close_identity_sha256",
            "instance_terminal_identity_sha256",
            "concurrency_window_identity_sha256",
        ):
            _validate_nested_sha(field, entry[field])
        _validate_nonempty_ascii("allocation instance_id", entry["instance_id"])
        _validate_nonempty_ascii(
            "allocation terminal_state", entry["terminal_state"]
        )
        _validate_nested_timestamp(
            "charged_interval_started_at", entry["charged_interval_started_at"]
        )
        _validate_nested_timestamp(
            "charged_interval_ended_at", entry["charged_interval_ended_at"]
        )
        if entry["charged_interval_started_at"] >= entry["charged_interval_ended_at"]:
            raise RecordValidationError("allocation charged interval is not ordered")
        keys.append((entry["allocation_ordinal"], entry["instance_id"]))
    if keys != sorted(set(keys)):
        raise RecordValidationError("allocations must be sorted and duplicate-free")


def _validate_owner_history(entries: object) -> None:
    if type(entries) is not list or not entries:
        raise RecordValidationError("owner history must be a nonempty array")
    attempts = []
    expiries = []
    for raw in entries:
        entry = _validate_nested_entry(
            raw, _OWNER_HISTORY_ENTRY_FIELDS, "owner history"
        )
        if type(entry["owner_attempt"]) is not int or entry["owner_attempt"] <= 0:
            raise RecordValidationError("owner history attempt is invalid")
        for field in ("owner_principal_arn", "owner_function_version_arn"):
            _validate_nonempty_ascii(field, entry[field])
        for field in (
            "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256"
        ):
            _validate_nested_sha(field, entry[field])
        _validate_nested_timestamp(
            "owner_hard_expires_at", entry["owner_hard_expires_at"]
        )
        attempts.append(entry["owner_attempt"])
        expiries.append(entry["owner_hard_expires_at"])
    if attempts != list(range(1, len(attempts) + 1)):
        raise RecordValidationError("owner history must be complete and ordered")
    if expiries != sorted(expiries):
        raise RecordValidationError(
            "owner history timestamps must be chronologically nondecreasing"
        )


def _validate_run_instances_attempts(entries: object) -> None:
    if type(entries) is not list:
        raise RecordValidationError("RunInstances attempts must be an exact array")
    attempts = []
    attempted_times = []
    allowed = {
        "DIRECT_SUCCESS", "POSITIVE_SERVICE_REJECTION", "TRANSPORT_AMBIGUOUS"
    }
    for raw in entries:
        entry = _validate_nested_entry(
            raw, _RUN_INSTANCES_ATTEMPT_ENTRY_FIELDS, "RunInstances attempt"
        )
        if (
            type(entry["attempt"]) is not int
            or entry["attempt"] <= 0
            or type(entry["outcome"]) is not str
            or entry["outcome"] not in allowed
        ):
            raise RecordValidationError("RunInstances attempt identity is invalid")
        _validate_nested_timestamp("attempted_at", entry["attempted_at"])
        direct_pair = (
            entry["request_id"] is not None,
            entry["response_sha256"] is not None,
        )
        _validate_nonempty_ascii(
            "RunInstances attempt request ID",
            entry["request_id"],
            nullable=True,
        )
        if direct_pair[0] != direct_pair[1]:
            raise RecordValidationError("RunInstances direct attempt is incomplete")
        _validate_nested_sha(
            "response_sha256", entry["response_sha256"], nullable=True
        )
        _validate_nested_sha(
            "transport_error_sha256",
            entry["transport_error_sha256"],
            nullable=True,
        )
        if entry["outcome"] == "TRANSPORT_AMBIGUOUS":
            if any(direct_pair) or entry["transport_error_sha256"] is None:
                raise RecordValidationError("ambiguous attempt evidence mismatch")
        elif not all(direct_pair) or entry["transport_error_sha256"] is not None:
            raise RecordValidationError("direct attempt evidence mismatch")
        attempts.append(entry["attempt"])
        attempted_times.append(entry["attempted_at"])
    if attempts != list(range(1, len(attempts) + 1)):
        raise RecordValidationError("RunInstances attempts must be complete")
    if attempted_times != sorted(attempted_times):
        raise RecordValidationError(
            "RunInstances attempt timestamps must be chronologically nondecreasing"
        )


def _validate_terminal_worker_launch_evidence(entries: object) -> None:
    if type(entries) is not list:
        raise RecordValidationError(
            "worker-launch evidence must be an exact array"
        )
    allowed = {
        "PREPARED_PROVED_NOT_SENT", "POSITIVE_SERVICE_REJECTION_NO_INSTANCE",
        "EXACT_SINGLE_INSTANCE", "MULTIPLE_INSTANCE_TOKEN", "UNRESOLVED_LAUNCH",
    }
    for raw in entries:
        entry = _validate_nested_entry(
            raw, _WORKER_LAUNCH_EVIDENCE_ENTRY_FIELDS, "worker-launch evidence"
        )
        if type(entry["kind"]) is not str or entry["kind"] not in allowed:
            raise RecordValidationError("worker-launch evidence kind is invalid")
        _validate_ordinal_pair(entry, "worker-launch evidence")
        if (
            type(entry["ec2_client_token"]) is not str
            or len(entry["ec2_client_token"]) != 64
            or any(ord(char) > 127 for char in entry["ec2_client_token"])
        ):
            raise RecordValidationError("worker-launch ClientToken is invalid")
        for field in (
            "launch_parameters_sha256", "expected_worker_tags_sha256",
            "worker_launch_identity_sha256", "prepared_journal_entry_sha256",
            "ddb_committed_journal_entry_sha256",
        ):
            _validate_nested_sha(field, entry[field])
        for field in (
            "prior_worker_launch_identity_sha256",
            "prior_instance_terminal_identity_sha256",
            "prior_spend_allocation_close_identity_sha256",
            "direct_run_instances_response_sha256",
            "cloudtrail_lookup_evidence_sha256",
            "describe_instances_evidence_sha256", "instance_observation_sha256",
            "spend_allocation_open_identity_sha256",
            "instance_terminal_identity_sha256",
            "spend_allocation_close_identity_sha256", "incident_identity_sha256",
            "worker_launch_liability_identity_sha256",
            "abandoned_not_sent_proof_sha256",
            "positive_service_rejection_evidence_sha256",
            "no_call_in_flight_evidence_sha256",
        ):
            _validate_nested_sha(field, entry[field], nullable=True)
        _validate_nested_timestamp("prepared_at", entry["prepared_at"])
        if entry["possibly_sent_at"] is not None:
            _validate_nested_timestamp("possibly_sent_at", entry["possibly_sent_at"])
        _validate_owner_history(entry["owner_history"])
        _validate_run_instances_attempts(entry["run_instances_attempt_evidence"])
        observed = entry["observed_instance_ids"]
        if (
            type(observed) is not list
            or any(
                type(item) is not str
                or not item
                or any(ord(char) > 127 for char in item)
                for item in observed
            )
            or observed != sorted(set(observed))
        ):
            raise RecordValidationError("observed instance set is invalid")
        direct_pair = (
            entry["direct_run_instances_request_id"] is not None,
            entry["direct_run_instances_response_sha256"] is not None,
        )
        _validate_nonempty_ascii(
            "worker direct request ID",
            entry["direct_run_instances_request_id"],
            nullable=True,
        )
        if direct_pair[0] != direct_pair[1]:
            raise RecordValidationError(
                "worker-launch direct request evidence is incomplete"
            )
        direct_attempts = [
            attempt
            for attempt in entry["run_instances_attempt_evidence"]
            if attempt["outcome"] in {
                "DIRECT_SUCCESS", "POSITIVE_SERVICE_REJECTION"
            }
        ]
        if (
            all(direct_pair)
            and (
                len(direct_attempts) != 1
                or direct_attempts[0]["request_id"]
                != entry["direct_run_instances_request_id"]
                or direct_attempts[0]["response_sha256"]
                != entry["direct_run_instances_response_sha256"]
            )
        ) or (not any(direct_pair) and direct_attempts):
            raise RecordValidationError(
                "worker-launch direct request correlation mismatch"
            )
        kind = entry["kind"]
        if kind == "PREPARED_PROVED_NOT_SENT":
            valid = (
                entry["send_stage"] == "NOT_SENT"
                and entry["possibly_sent_at"] is None
                and entry["run_instances_attempt_evidence"] == []
                and not any(direct_pair)
                and observed == []
                and entry["abandoned_not_sent_proof_sha256"] is not None
                and entry["worker_launch_liability_identity_sha256"] is None
            )
        else:
            valid = (
                entry["send_stage"] == "POSSIBLY_SENT"
                and entry["possibly_sent_at"] is not None
                and bool(entry["run_instances_attempt_evidence"])
                and entry["cloudtrail_lookup_evidence_sha256"] is not None
                and entry["describe_instances_evidence_sha256"] is not None
                and entry["worker_launch_liability_identity_sha256"] is not None
            )
        if kind == "POSITIVE_SERVICE_REJECTION_NO_INSTANCE":
            valid = valid and (
                observed == []
                and entry["positive_service_rejection_evidence_sha256"] is not None
                and entry["no_call_in_flight_evidence_sha256"] is not None
            )
        elif kind == "EXACT_SINGLE_INSTANCE":
            valid = valid and (
                len(observed) == 1
                and all(
                    entry[field] is not None
                    for field in (
                        "instance_observation_sha256",
                        "spend_allocation_open_identity_sha256",
                        "instance_terminal_identity_sha256",
                        "spend_allocation_close_identity_sha256",
                    )
                )
                and entry["incident_identity_sha256"] is None
            )
        elif kind == "MULTIPLE_INSTANCE_TOKEN":
            valid = valid and (
                len(observed) >= 2
                and entry["incident_identity_sha256"] is not None
                and entry["instance_terminal_identity_sha256"] is not None
                and entry["spend_allocation_close_identity_sha256"] is not None
            )
        elif kind == "UNRESOLVED_LAUNCH":
            valid = valid and entry["incident_identity_sha256"] is not None
        if not valid:
            raise RecordValidationError(
                "worker-launch evidence discriminated union mismatch"
            )


def _validate_terminal_liabilities(entries: object) -> None:
    if type(entries) is not list:
        raise RecordValidationError(
            "worker-launch liabilities must be an exact array"
        )
    for raw in entries:
        entry = _validate_nested_entry(
            raw, _WORKER_LAUNCH_LIABILITY_ENTRY_FIELDS,
            "liability",
        )
        _validate_ordinal_pair(entry, "liability")
        if (
            type(entry["state"]) is not str
            or entry["state"] not in {"WATCHING", "LATE_INSTANCE_DRAINING"}
        ):
            raise RecordValidationError("liability state is invalid")
        for field in (
            "activation_ordinal", "generation", "owner_attempt", "revision"
        ):
            if type(entry[field]) is not int or entry[field] <= 0:
                raise RecordValidationError("liability integer is invalid")
        for field in (
            "activation_id", "watch_schedule_arn", "owner_execution_arn",
            "owner_state_machine_version_arn",
        ):
            _validate_nonempty_ascii(field, entry[field])
        if (
            type(entry["ec2_client_token"]) is not str
            or len(entry["ec2_client_token"]) != 64
            or any(ord(char) > 127 for char in entry["ec2_client_token"])
        ):
            raise RecordValidationError("liability ClientToken is invalid")
        for field in (
            "worker_launch_identity_sha256",
            "worker_launch_liability_identity_sha256",
            "launch_parameters_sha256", "expected_worker_tags_sha256",
            "watch_schedule_identity_sha256", "last_scan_evidence_sha256",
            "owner_dispatch_identity_sha256",
            "owner_invocation_nonce_sha256",
        ):
            _validate_nested_sha(field, entry[field])
        _validate_nested_sha(
            "incident_identity_sha256",
            entry["incident_identity_sha256"],
            nullable=True,
        )
        for field in (
            "watch_not_before", "watch_not_after", "owner_hard_expires_at"
        ):
            _validate_nested_timestamp(field, entry[field])
        if entry["watch_not_before"] >= entry["watch_not_after"]:
            raise RecordValidationError("liability watch interval is not ordered")
        observed = entry["observed_instance_ids"]
        if (
            type(observed) is not list
            or any(
                type(item) is not str
                or not item
                or any(ord(char) > 127 for char in item)
                for item in observed
            )
            or observed != sorted(set(observed))
        ):
            raise RecordValidationError("liability observed instances are invalid")


def _validate_terminal_request_evidence(entries: object) -> None:
    if type(entries) is not list:
        raise RecordValidationError("request evidence must be an exact array")
    for raw in entries:
        if type(raw) is not dict:
            raise RecordValidationError("request evidence kind is invalid")
        kind = raw.get("kind")
        if type(kind) is not str or kind not in _REQUEST_EVIDENCE_ENTRY_FIELDS:
            raise RecordValidationError("request evidence kind is invalid")
        entry = _validate_nested_entry(
            raw, _REQUEST_EVIDENCE_ENTRY_FIELDS[kind],
            "request evidence",
        )
        for field, value in entry.items():
            if field.endswith("_sha256"):
                _validate_nested_sha(field, value)
            if field.endswith("_at"):
                _validate_nested_timestamp(field, value)
        kind = entry["kind"]
        if kind == "ZERO_MATCH_SCAN":
            if (
                entry["pagination_complete"] is not True
                or type(entry["match_count"]) is not int
                or entry["match_count"] != 0
                or entry["scan_started_at"] >= entry["scan_completed_at"]
            ):
                raise RecordValidationError("zero-match proof is incomplete")
            for field in (
                "database_lower_bound", "database_upper_bound",
                "journal_lower_bound", "journal_upper_bound",
            ):
                _validate_nonempty_ascii(field, entry[field])
        elif kind == "EXACT_REQUEST":
            for field in ("request_id", "request_state"):
                _validate_nonempty_ascii(field, entry[field])
            if entry["request_created_at"] > entry["request_updated_at"]:
                raise RecordValidationError("request timestamps are not ordered")
        elif kind == "MULTIPLE_MATCH":
            members = entry["request_ids"]
            if (
                type(members) is not list
                or len(members) < 2
                or any(
                    type(member) is not str
                    or not member
                    or any(ord(char) > 127 for char in member)
                    for member in members
                )
                or members != sorted(set(members))
                or canonical_sha256(members) != entry["complete_member_set_sha256"]
            ):
                raise RecordValidationError("multiple request set is incomplete")
        elif kind in {"REQUEST_CANCEL", "JOB_CANCEL"}:
            _validate_nonempty_ascii("cancel action_key", entry["action_key"])
            if entry["requested_at"] >= entry["terminal_observed_at"]:
                raise RecordValidationError("cancel timestamps are not ordered")
        elif kind == "QUIESCENCE_SNAPSHOT" and (
            entry["scan_started_at"] >= entry["scan_completed_at"]
        ):
            raise RecordValidationError("quiescence scan timestamps are not ordered")
    snapshots = [
        entry for entry in entries
        if type(entry) is dict and entry.get("kind") == "QUIESCENCE_SNAPSHOT"
    ]
    if len(snapshots) == 2 and (
        snapshots[0]["scan_completed_at"] >= snapshots[1]["scan_started_at"]
    ):
        raise RecordValidationError(
            "quiescence snapshots must be chronologically separated"
        )


def _validate_terminal_nested_entries(record: Mapping[str, object]) -> None:
    _validate_terminal_allocations(record["allocations"])
    _validate_terminal_worker_launch_evidence(record["worker_launch_evidence"])
    _validate_terminal_liabilities(record["worker_launch_liabilities"])
    _validate_terminal_request_evidence(record["request_evidence"])
    launches = {
        entry["allocation_ordinal"]: entry
        for entry in record["worker_launch_evidence"]
    }
    for liability in record["worker_launch_liabilities"]:
        launch = launches.get(liability["allocation_ordinal"])
        if (
            liability["activation_id"] != record["activation_id"]
            or liability["activation_ordinal"] != record["activation_ordinal"]
            or liability["generation"] != record["generation"]
            or launch is None
            or any(
                liability[field] != launch[launch_field]
                for field, launch_field in (
                    (
                        "worker_launch_identity_sha256",
                        "worker_launch_identity_sha256",
                    ),
                    ("ec2_client_token", "ec2_client_token"),
                    ("launch_parameters_sha256", "launch_parameters_sha256"),
                    (
                        "expected_worker_tags_sha256",
                        "expected_worker_tags_sha256",
                    ),
                    (
                        "worker_launch_liability_identity_sha256",
                        "worker_launch_liability_identity_sha256",
                    ),
                )
            )
        ):
            raise RecordValidationError(
                "worker-launch liability binding mismatch"
            )


def _validate_terminal_v2(record: Mapping[str, object]) -> None:
    _validate_terminal_nested_entries(record)
    outcome = record["outcome"]
    if outcome not in _OUTCOMES:
        raise RecordValidationError("unknown terminal-v2 outcome")
    rule = _TERMINAL_OUTCOME_RULES[outcome]
    for field in ("handoff", "binding"):
        mode = rule[field]
        if mode == "null" and record[field] is not None:
            raise RecordValidationError("terminal outcome requires null " + field)
        if mode == "nonnull" and record[field] is None:
            raise RecordValidationError("terminal outcome requires " + field)
    count = len(record["allocations"])
    allocation_mode = rule["allocations"]
    if (
        (allocation_mode == "zero" and count != 0)
        or (allocation_mode == "one" and count != 1)
        or (allocation_mode == "nonempty" and count < 1)
        or (allocation_mode == "multiple" and count < 2)
    ):
        raise RecordValidationError("terminal allocation matrix mismatch")
    if record["request_cardinality"] not in rule["request"]:
        raise RecordValidationError("terminal request-cardinality matrix mismatch")
    if record["operator_disposition_required"] is not rule["disposition"]:
        raise RecordValidationError("terminal disposition matrix mismatch")
    for field in rule["marker_required"]:
        if record[field] is None:
            raise RecordValidationError("terminal marker is required: " + field)
    for field in rule["marker_null"]:
        if record[field] is not None:
            raise RecordValidationError("terminal marker must be null: " + field)
    prior_mode = rule["prior_terminal_v1"]
    if (record["prior_terminal_v1_identity"] is None) != (prior_mode == "null"):
        raise RecordValidationError("prior terminal-v1 matrix mismatch")
    needs_post_quiescence = outcome.endswith("_INCIDENT") or outcome.endswith(
        "_NO_JOB"
    )
    if (
        record["post_terminal_quiescence_evidence"] is None
    ) != (not needs_post_quiescence):
        raise RecordValidationError("post-terminal quiescence matrix mismatch")
    allocations = record["allocations"]
    count = len(allocations)
    expected_cardinality = "ZERO" if count == 0 else ("ONE" if count == 1 else "MULTIPLE")
    if record["worker_cardinality"] != expected_cardinality:
        raise RecordValidationError("worker cardinality does not match allocation count")
    if outcome in _NO_ALLOCATION and count != 0:
        raise RecordValidationError("outcome forbids allocations")
    if outcome in _ONE_OR_MORE and count < 1:
        raise RecordValidationError("outcome requires an allocation")
    if outcome in _MULTIPLE and count < 2:
        raise RecordValidationError("outcome requires multiple allocations")
    if outcome in _NOT_APPLICABLE:
        allowed_request = {"NOT_APPLICABLE"}
    elif outcome in _ZERO_REQUEST:
        allowed_request = {"ZERO"}
    elif outcome in _ONE_REQUEST:
        allowed_request = {"ONE"}
    elif outcome in (
        "MULTIPLE_BOUND_WORKERS_INCIDENT",
        "QUIESCED_UNRESOLVED_BOUND_JOB_NO_ALLOCATION_INCIDENT",
        "QUIESCED_UNRESOLVED_BOUND_JOB_WITH_ALLOCATIONS_INCIDENT",
    ):
        allowed_request = {"ONE", "MULTIPLE"}
    else:
        allowed_request = {"ZERO", "ONE", "MULTIPLE"}
    if record["request_cardinality"] not in allowed_request:
        raise RecordValidationError("request cardinality is invalid for outcome")
    evidence = record["request_evidence"]
    kinds = [entry.get("kind") if type(entry) is dict else None for entry in evidence]
    cardinality = record["request_cardinality"]
    if cardinality == "NOT_APPLICABLE" and evidence != []:
        raise RecordValidationError("NOT_APPLICABLE request evidence must be empty")
    if cardinality == "ZERO" and kinds != [
        "ZERO_MATCH_SCAN", "QUIESCENCE_SNAPSHOT", "QUIESCENCE_SNAPSHOT"
    ]:
        raise RecordValidationError("ZERO request evidence construction mismatch")
    if cardinality == "ONE":
        if (
            len(kinds) < 3
            or kinds[0] != "EXACT_REQUEST"
            or kinds[-2:] != ["QUIESCENCE_SNAPSHOT", "QUIESCENCE_SNAPSHOT"]
            or any(kind not in {"REQUEST_CANCEL", "JOB_CANCEL"} for kind in kinds[1:-2])
        ):
            raise RecordValidationError("ONE request evidence construction mismatch")
        cancel_keys = [
            entry.get("action_key") for entry in evidence[1:-2]
        ]
        if cancel_keys != sorted(cancel_keys):
            raise RecordValidationError("ONE request cancel evidence order mismatch")
    if cardinality == "MULTIPLE":
        if not kinds or kinds[0] != "MULTIPLE_MATCH":
            raise RecordValidationError("MULTIPLE request evidence construction mismatch")
        members = evidence[0].get("request_ids") if type(evidence[0]) is dict else None
        if type(members) is not list or members != sorted(set(members)):
            raise RecordValidationError("multiple request identities must be sorted")
        exact_entries = evidence[1 : 1 + len(members)]
        if (
            [entry.get("kind") for entry in exact_entries]
            != ["EXACT_REQUEST"] * len(members)
            or [entry.get("request_id") for entry in exact_entries] != members
        ):
            raise RecordValidationError("multiple request evidence count mismatch")
        suffix = evidence[1 + len(members) :]
        if (
            len(suffix) < 2
            or [entry.get("kind") for entry in suffix[-2:]]
            != ["QUIESCENCE_SNAPSHOT", "QUIESCENCE_SNAPSHOT"]
            or any(
                entry.get("kind") not in {"REQUEST_CANCEL", "JOB_CANCEL"}
                for entry in suffix[:-2]
            )
        ):
            raise RecordValidationError("multiple request evidence order mismatch")
        cancel_keys = [entry.get("action_key") for entry in suffix[:-2]]
        if cancel_keys != sorted(cancel_keys):
            raise RecordValidationError("multiple request cancel evidence order mismatch")
    disposition_required = outcome.endswith("_INCIDENT") or outcome in {
        "STORED_DECISION_NO_POST", "CONSUMED_PROVED_NO_POST",
        "KNOWN_REJECTED_NO_JOB", "AMBIGUOUS_PROVED_NO_JOB",
        "ACCEPTED_REQUEST_FAILED_NO_JOB", "ACCEPTED_REQUEST_CANCELLED_NO_JOB",
        "FAILED_TERMINAL_BEFORE_ALLOCATION", "FAILED_TERMINAL_AFTER_ALLOCATION",
        "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH", "EXPIRED_UNSTARTED",
    }
    if disposition_required and record["operator_disposition_required"] is not True:
        raise RecordValidationError("outcome requires operator disposition")
    no_handoff = {
        "SUPPORT_EXECUTION_START_UNRESOLVED_INCIDENT",
        "STORED_DECISION_NO_POST",
        "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH",
        "EXPIRED_UNSTARTED",
    }
    if outcome in no_handoff and (
        record["handoff"] is not None or record["binding"] is not None
    ):
        raise RecordValidationError("outcome requires null handoff and binding")
    both_bound = {
        "DRAINED_COMPLETED", "DRAINED_TRAINING_DEFERRED",
        "DRAINED_RESUMABLE_DEADLINE", "FAILED_TERMINAL_AFTER_ALLOCATION",
        "FAILED_TERMINAL_BEFORE_ALLOCATION", "MULTIPLE_BOUND_WORKERS_INCIDENT",
        "QUIESCED_UNRESOLVED_BOUND_JOB_NO_ALLOCATION_INCIDENT",
        "QUIESCED_UNRESOLVED_BOUND_JOB_WITH_ALLOCATIONS_INCIDENT",
    }
    handoff_only = {
        "CONSUMED_PROVED_NO_POST", "KNOWN_REJECTED_NO_JOB",
        "AMBIGUOUS_PROVED_NO_JOB", "ACCEPTED_REQUEST_FAILED_NO_JOB",
        "ACCEPTED_REQUEST_CANCELLED_NO_JOB", "UNBOUND_LATE_WORKER_INCIDENT",
        "MULTIPLE_UNBOUND_WORKERS_INCIDENT",
        "QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT",
        "QUIESCED_UNRESOLVED_REQUEST_WITH_ALLOCATIONS_INCIDENT",
    }
    if outcome in both_bound and (
        record["handoff"] is None or record["binding"] is None
    ):
        raise RecordValidationError("outcome requires handoff and binding")
    if outcome in handoff_only and (
        record["handoff"] is None or record["binding"] is not None
    ):
        raise RecordValidationError("outcome requires handoff without binding")
    if outcome == "EXPIRED_UNSTARTED":
        if record["prior_terminal_v1_identity"] is None:
            raise RecordValidationError("expired-unstarted requires terminal-v1")
    elif record["prior_terminal_v1_identity"] is not None:
        raise RecordValidationError("outcome forbids prior terminal-v1")
    if outcome == "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH":
        if (
            record["worker_launch_evidence"] != []
            or record["worker_launch_liabilities"] != []
            or record["operator_disposition_required"] is not True
            or any(
                record[field] is not None
                for field in (
                    "final_heartbeat_identity", "checkpoint_identity",
                    "cache_identity", "training_identity",
                    "evaluation_identity", "drain_identity",
                )
            )
        ):
            raise RecordValidationError("no-launch outcome evidence mismatch")
    if outcome.startswith("WORKER_LAUNCH_UNRESOLVED_") and (
        record["handoff"] is None
    ):
        raise RecordValidationError("unresolved launch requires recovery handoff")
    marker_fields = (
        "final_heartbeat_identity", "checkpoint_identity", "cache_identity",
        "training_identity", "evaluation_identity", "drain_identity",
    )
    marker_shape = {
        "key", "version_id", "body_sha256", "canonical_identity_sha256"
    }
    for field in marker_fields:
        marker = record[field]
        if marker is not None:
            if type(marker) is not dict or set(marker) != marker_shape:
                raise RecordValidationError("terminal marker identity schema mismatch")
            if any(type(marker[name]) is not str or not marker[name] for name in marker):
                raise RecordValidationError("terminal marker identity value mismatch")
            for name in ("body_sha256", "canonical_identity_sha256"):
                if _SHA.fullmatch(marker[name]) is None:
                    raise RecordValidationError("terminal marker SHA-256 malformed")
    if outcome.startswith("DRAINED_") and record["drain_identity"] is None:
        raise RecordValidationError("drained outcome requires drain identity")
    if outcome == "DRAINED_COMPLETED" and (
        record["training_identity"] is None or record["evaluation_identity"] is None
    ):
        raise RecordValidationError("completed outcome requires training and evaluation")
    if outcome == "DRAINED_TRAINING_DEFERRED" and (
        record["cache_identity"] is None
        or record["checkpoint_identity"] is None
        or record["training_identity"] is not None
        or record["evaluation_identity"] is not None
    ):
        raise RecordValidationError("training-deferred marker matrix mismatch")
    if outcome == "DRAINED_RESUMABLE_DEADLINE" and (
        record["checkpoint_identity"] is None
    ):
        raise RecordValidationError("resumable deadline requires checkpoint")
    launch_kinds = {
        "PREPARED_PROVED_NOT_SENT", "POSITIVE_SERVICE_REJECTION_NO_INSTANCE",
        "EXACT_SINGLE_INSTANCE", "MULTIPLE_INSTANCE_TOKEN", "UNRESOLVED_LAUNCH",
    }
    for entry in record["worker_launch_evidence"]:
        if type(entry) is not dict or entry.get("kind") not in launch_kinds:
            raise RecordValidationError("worker-launch evidence kind is invalid")
    if allocations and not record["worker_launch_evidence"]:
        raise RecordValidationError("allocations require worker-launch evidence")
    allocation_keys = []
    for entry in allocations:
        if (
            type(entry) is not dict
            or type(entry.get("allocation_ordinal")) is not int
            or entry["allocation_ordinal"] <= 0
            or type(entry.get("instance_id")) is not str
            or not entry["instance_id"]
        ):
            raise RecordValidationError("allocation identity is invalid")
        allocation_keys.append((entry["allocation_ordinal"], entry["instance_id"]))
    if allocation_keys != sorted(set(allocation_keys)):
        raise RecordValidationError("allocations must be sorted and duplicate-free")
    evidence_ordinals = [
        entry.get("allocation_ordinal") for entry in record["worker_launch_evidence"]
    ]
    if any(type(item) is not int or item <= 0 for item in evidence_ordinals):
        raise RecordValidationError("worker-launch ordinal is invalid")
    if evidence_ordinals != sorted(set(evidence_ordinals)):
        raise RecordValidationError("worker-launch evidence must be ordinal-sorted")
    if evidence_ordinals and evidence_ordinals != list(
        range(1, max(evidence_ordinals) + 1)
    ):
        raise RecordValidationError("worker-launch evidence has an ordinal gap")
    if set(ordinal for ordinal, _ in allocation_keys) - set(evidence_ordinals):
        raise RecordValidationError("allocation lacks worker-launch evidence")
    liability_ordinals = [
        entry.get("allocation_ordinal")
        if type(entry) is dict else None
        for entry in record["worker_launch_liabilities"]
    ]
    if any(type(item) is not int or item <= 0 for item in liability_ordinals):
        raise RecordValidationError("worker-launch liability ordinal is invalid")
    if liability_ordinals != sorted(set(liability_ordinals)):
        raise RecordValidationError("worker-launch liabilities must be ordinal-sorted")
    possibly_sent_ordinals = {
        entry["allocation_ordinal"]
        for entry in record["worker_launch_evidence"]
        if entry["kind"] != "PREPARED_PROVED_NOT_SENT"
    }
    if set(liability_ordinals) != possibly_sent_ordinals:
        raise RecordValidationError("worker-launch liability bijection mismatch")


_RETAINED_ACTION_KINDS = {
    "glm52_production_recovery_action": (
        "RECOVERY", frozenset(
            """REQUEST_CANCEL JOB_CANCEL WORKER_DRAIN TERMINAL_V2_PUBLISH
            RECOVERY_HANDOFF""".split()
        )
    ),
    "glm52_production_finalization_action": (
        "FINALIZATION", frozenset(
            "SUPPORT_DELETE SNAPSHOT_DISPOSITION H1G_DRAINED_PUBLISH".split()
        )
    ),
    "glm52_production_snapshot_cleanup_action": (
        "SNAPSHOT_CLEANUP", frozenset({"SNAPSHOT_DELETE"})
    ),
    "glm52_production_worker_launch_liability_action": (
        "WORKER_LAUNCH_LIABILITY", frozenset(
            """SAME_TOKEN_COMPLETE TERMINATE_LATE_INSTANCE
            POST_TERMINAL_ALLOCATION_DISCOVER POST_TERMINAL_ALLOCATION_OPEN
            POST_TERMINAL_ALLOCATION_CLOSE LIABILITY_SETTLE""".split()
        )
    ),
}


def _validate_retained_action(
    record_type: str, record: Mapping[str, object]
) -> None:
    rule = _RETAINED_ACTION_KINDS.get(record_type)
    if rule is None:
        return
    domain, kinds = rule
    if record["authority_domain"] != domain or record["action_kind"] not in kinds:
        raise RecordValidationError("cross-domain action kind")
    worker = record_type == "glm52_production_worker_launch_liability_action"
    confinement = (
        "allocation_ordinal", "allocation_ordinal_text",
        "worker_launch_identity_sha256",
        "worker_launch_liability_identity_sha256",
    )
    if worker and any(record[field] is None for field in confinement):
        raise RecordValidationError("worker action confinement is incomplete")
    if not worker and any(record[field] is not None for field in confinement):
        raise RecordValidationError("non-worker action has worker confinement")


def _validate_state_nullability(
    record_type: str, record: Mapping[str, object]
) -> None:
    if "state" not in record:
        return
    _validate_declarative_state(record_type, record)
    owner_fields = [
        field for field in (
            "owner_execution_arn", "owner_state_machine_version_arn",
            "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256",
            "owner_hard_expires_at",
        ) if field in record
    ]
    if record_type == "glm52_production_action":
        owner_fields = []
    if owner_fields:
        populated = [record[field] is not None for field in owner_fields]
        if any(populated) != all(populated):
            raise RecordValidationError("owner fields must be all-null or all-non-null")
        if all(populated) != (record.get("owner_attempt") is not None):
            raise RecordValidationError("owner attempt must match owner nullability")
    dormant_evidence = {
        "glm52_production_recovery_control": (
            "recovery_barrier_nonce_sha256", "support_control_revision_at_seal",
            "support_execution_identity_sha256", "allowed_action_set_sha256",
            "terminal_v2_identity_sha256",
        ),
        "glm52_production_finalization_control": (
            "finalization_barrier_nonce_sha256",
            "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
            "support_plane_finalized_identity_sha256",
            "snapshot_disposition_identity_sha256", "h1g_drained_identity_sha256",
        ),
    }
    if record_type in dormant_evidence and record["state"] == "DORMANT":
        if record.get("owner_attempt") is not None:
            raise RecordValidationError("dormant owner attempt must be null")
        if any(record[field] is not None for field in owner_fields):
            raise RecordValidationError("dormant control must be ownerless")
        if any(record[field] is not None for field in dormant_evidence[record_type]):
            raise RecordValidationError("dormant evidence must be null")
    if record_type == "glm52_production_recovery_control":
        if record["state"] != "DORMANT" and any(
            record[field] is None
            for field in (
                "recovery_barrier_nonce_sha256",
                "support_control_revision_at_seal",
                "support_execution_identity_sha256",
                "allowed_action_set_sha256",
            )
        ):
            raise RecordValidationError("owned recovery state requires seal evidence")
        if record["state"] == "OWNED" and record["terminal_v2_identity_sha256"] is not None:
            raise RecordValidationError("OWNED recovery cannot have terminal evidence")
        if record["state"] in {"TERMINAL_V2_PUBLISHED", "RECOVERY_COMPLETE"} and (
            record["terminal_v2_identity_sha256"] is None
        ):
            raise RecordValidationError("recovery state requires terminal evidence")
    if record_type == "glm52_production_finalization_control":
        required = {
            "SUPPORT_FINALIZED": ("support_plane_finalized_identity_sha256",),
            "SNAPSHOT_DISPOSITION_RECORDED": (
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256",
            ),
            "DRAINED_PUBLISHED": (
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256",
                "h1g_drained_identity_sha256",
            ),
        }
        if record["state"] in required and any(
            record[field] is None for field in required[record["state"]]
        ):
            raise RecordValidationError("finalization state requires prior evidence")
        future = {
            "OWNED": (
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256",
                "h1g_drained_identity_sha256",
            ),
            "SUPPORT_FINALIZED": (
                "snapshot_disposition_identity_sha256",
                "h1g_drained_identity_sha256",
            ),
            "SNAPSHOT_DISPOSITION_RECORDED": ("h1g_drained_identity_sha256",),
        }
        if record["state"] in future and any(
            record[field] is not None for field in future[record["state"]]
        ):
            raise RecordValidationError("finalization has premature evidence")
    if record_type == "glm52_production_action" and record["state"] == "ARMED":
        exact_null = (
            "relay_envelope_sha256", "owner_invocation_nonce_sha256",
            "post_owner_invocation_nonce_sha256", "post_owner_function_version_arn",
            "post_owner_dispatch_identity_sha256", "post_owner_hard_expires_at",
            "authority_audit_body_sha256", "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision", "consumed_at", "post_started_at",
            "post_authorized_at", "completed_at", "abandoned_at", "outcome_class",
            "classification_evidence_kind",
            "classification_evidence_body_sha256", "sky_request_id",
            "response_identity_sha256", "abandonment_proof_sha256",
            "consume_transaction_client_request_token_sha256",
            "post_start_transaction_client_request_token_sha256",
            "post_authorization_transaction_client_request_token_sha256",
            "post_classification_transaction_client_request_token_sha256",
        )
        if any(record[field] is not None for field in exact_null):
            raise RecordValidationError("ARMED action has premature evidence")
    if (
        record_type == "glm52_production_action"
        and record["state"] == "POST_CLASSIFIED"
    ):
        post_owner = (
            "post_owner_invocation_nonce_sha256",
            "post_owner_function_version_arn",
            "post_owner_dispatch_identity_sha256",
            "post_owner_hard_expires_at",
            "post_started_at",
        )
        post_authorization = (
            "post_authorized_at",
            "relay_envelope_sha256",
        )
        direct_outcomes = {"ACCEPTED", "KNOWN_REJECTED", "AMBIGUOUS"}
        kind = record["classification_evidence_kind"]
        outcome = record["outcome_class"]
        if kind == "RELAY_RESPONSE":
            valid = (
                outcome in direct_outcomes
                and record["response_identity_sha256"] is not None
                and record["classification_evidence_body_sha256"] is not None
                and all(record[field] is not None for field in post_owner)
                and all(
                    record[field] is not None
                    for field in post_authorization
                )
            )
        elif kind == "OWNER_DEATH_PROOF":
            consumed_preimage = all(
                record[field] is None
                for field in post_owner + post_authorization
            )
            started_preimage = (
                all(record[field] is not None for field in post_owner)
                and all(
                    record[field] is None
                    for field in post_authorization
                )
            )
            authorized_preimage = (
                all(record[field] is not None for field in post_owner)
                and all(
                    record[field] is not None
                    for field in post_authorization
                )
            )
            valid = (
                record["response_identity_sha256"] is None
                and record["sky_request_id"] is None
                and record["classification_evidence_body_sha256"] is not None
                and (
                    (
                        outcome == "PROVED_NOT_SENT_OWNER_DIED"
                        and (consumed_preimage or started_preimage)
                    )
                    or (
                        outcome == "AMBIGUOUS_OWNER_DIED"
                        and authorized_preimage
                    )
                )
            )
        else:
            valid = False
        if not valid:
            raise RecordValidationError(
                "classification evidence discriminated union mismatch"
            )
    if record_type in _RETAINED_ACTION_KINDS and record["state"] == "ARMED":
        exact_null = (
            "authority_audit_body_sha256", "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision", "consumed_at", "completed_at",
            "response_identity_sha256", "reconciliation_identity_sha256",
            "consume_transaction_client_request_token_sha256",
        )
        if any(record[field] is not None for field in exact_null):
            raise RecordValidationError("ARMED retained action has premature evidence")
    if record_type == "glm52_production_execution":
        if (
            record["direct_start_response_identity"] is None
        ) != (record["direct_start_request_id"] is None):
            raise RecordValidationError(
                "direct start evidence must be all-null or all-non-null"
            )
        later = (
            "start_attempted_at", "direct_start_response_identity",
            "direct_start_request_id", "observed_start_date", "terminal_status",
            "terminal_observed_at", "describe_execution_request_id",
            "describe_execution_response_sha256", "terminal_body_sha256",
            "unresolved_start_incident_body_sha256",
        )
        if record["state"] == "START_OWNED":
            if record["start_send_stage"] != "NOT_SENT" or any(
                record[field] is not None for field in later
            ):
                raise RecordValidationError("START_OWNED evidence matrix mismatch")
    if record_type == "glm52_production_worker_launch" and (
        (record["direct_run_instances_request_id"] is None)
        != (record["direct_run_instances_response_sha256"] is None)
    ):
        raise RecordValidationError(
            "direct RunInstances evidence must be all-null or all-non-null"
        )
    if record_type == "glm52_production_worker_launch" and (
        record["state"] != "PREPARED_NOT_SENT"
        and record["state"] != "ABANDONED_NOT_SENT"
    ):
        if (
            record["send_stage"] != "POSSIBLY_SENT"
            or record["possibly_sent_at"] is None
        ):
            raise RecordValidationError("possibly-sent state lacks send evidence")


_STATE_ENUMS = {
    "glm52_production_recovery_control": frozenset(
        "DORMANT OWNED TERMINAL_V2_PUBLISHED RECOVERY_COMPLETE".split()
    ),
    "glm52_production_finalization_control": frozenset(
        "DORMANT OWNED SUPPORT_FINALIZED SNAPSHOT_DISPOSITION_RECORDED DRAINED_PUBLISHED".split()
    ),
    "glm52_production_snapshot_cleanup_control": frozenset(
        """DORMANT ARMED OWNED DELETE_POSSIBLY_SENT DELETE_RECONCILING DELETED
        ALREADY_ABSENT CLEANUP_INCIDENT""".split()
    ),
    "glm52_production_execution": frozenset(
        """START_OWNED START_POSSIBLY_SENT RUNNING SUCCEEDED FAILED TIMED_OUT
        ABORTED START_POSSIBLY_SENT_UNRESOLVED_INCIDENT
        ABANDONED_PROVED_NOT_STARTED""".split()
    ),
    "glm52_production_worker_launch": frozenset(
        """PREPARED_NOT_SENT POSSIBLY_SENT INSTANCE_OBSERVED REJECTED_NO_INSTANCE
        MULTIPLE_INSTANCE_TOKEN_INCIDENT UNRESOLVED_LAUNCH_INCIDENT
        ALLOCATION_OPEN INSTANCE_TERMINAL ALLOCATION_CLOSED
        ABANDONED_NOT_SENT""".split()
    ),
    "glm52_production_worker_launch_liability": frozenset(
        """UNOWNED_NOT_ACTIONABLE WATCHING SAME_TOKEN_COMPLETION
        REJECTION_PROVED_AWAITING_TERMINAL_V2 SETTLED_NO_INSTANCE_REJECTED
        LATE_INSTANCE_DRAINING SETTLED_INSTANCE_CLOSED LIABILITY_INCIDENT""".split()
    ),
    "glm52_production_post_terminal_allocation": frozenset(
        "DISCOVERED ALLOCATION_OPEN INSTANCE_TERMINAL ALLOCATION_CLOSED".split()
    ),
    "glm52_production_action": frozenset(
        "ARMED CONSUMED COMPLETED AMBIGUOUS ABANDONED POST_STARTED POST_AUTHORIZED POST_CLASSIFIED".split()
    ),
}
for _action_type in _RETAINED_ACTION_KINDS:
    _STATE_ENUMS[_action_type] = frozenset(
        "ARMED CONSUMED COMPLETED AMBIGUOUS ABANDONED".split()
    )


def _validate_enums(record_type: str, record: Mapping[str, object]) -> None:
    if record_type in _STATE_ENUMS and record["state"] not in _STATE_ENUMS[record_type]:
        raise RecordValidationError("state enum is invalid")
    if record_type == "glm52_production_action":
        post_states = {"POST_STARTED", "POST_AUTHORIZED", "POST_CLASSIFIED"}
        if (
            record["state"] in post_states
            and record["action_kind"] != "SKY_POST"
        ):
            raise RecordValidationError("post state requires SKY_POST action")
        if (
            record["action_kind"] == "SKY_POST"
            and record["state"] in {"COMPLETED", "AMBIGUOUS"}
        ):
            raise RecordValidationError(
                "SKY_POST action cannot use non-post terminal state"
            )
    if record_type == "glm52_production_control" and record["phase"] not in {
        "OPEN", "RECOVERY_SEALING", "RECOVERY_COMPLETE",
        "TEARDOWN_SEALING", "TEARDOWN_SEALED",
    }:
        raise RecordValidationError("phase enum is invalid")
    if record_type == "glm52_production_operator_disposition" and record["decision"] not in {
        "ALLOW_NEW_ACTIVATION", "DENY_NEW_ACTIVATION"
    }:
        raise RecordValidationError("decision enum is invalid")
    if record_type == "glm52_production_worker_launch_liability_settlement" and (
        record["settlement_kind"] not in {
            "NO_INSTANCE_POSITIVE_REJECTION",
            "ALL_INSTANCES_TERMINAL_AND_SPEND_CLOSED",
        }
    ):
        raise RecordValidationError("settlement kind is invalid")


_STANDARD_OWNER = (
    "owner_execution_arn", "owner_state_machine_version_arn",
    "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256",
    "owner_hard_expires_at",
)
_WORKER_OWNER = (
    "owner_principal_arn", "owner_function_version_arn",
    "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256",
    "owner_hard_expires_at",
)
_RETAINED_ACTION_FUTURE = (
    "authority_audit_body_sha256", "authority_audit_closing_revision",
    "authorized_transition_from_revision", "authorized_transition_to_revision",
    "consumed_at", "completed_at", "response_identity_sha256",
    "reconciliation_identity_sha256",
    "consume_transaction_client_request_token_sha256",
)
_EXECUTION_AFTER_START = (
    "start_attempted_at", "direct_start_response_identity",
    "direct_start_request_id", "observed_start_date", "terminal_status",
    "terminal_observed_at", "describe_execution_request_id",
    "describe_execution_response_sha256", "terminal_body_sha256",
    "unresolved_start_incident_body_sha256",
)
_WORKER_AFTER_PREPARED = (
    "gpu_liability_reserve_ledger_identity_sha256", "possibly_sent_at",
    "direct_run_instances_request_id", "direct_run_instances_response_sha256",
    "instance_observation_sha256", "spend_allocation_open_identity_sha256",
    "instance_terminal_identity_sha256", "spend_allocation_close_identity_sha256",
    "incident_identity_sha256",
)
_ACTION_ARMED_FUTURE = (
    "relay_envelope_sha256", "owner_invocation_nonce_sha256",
    "post_owner_invocation_nonce_sha256", "post_owner_function_version_arn",
    "post_owner_dispatch_identity_sha256", "post_owner_hard_expires_at",
    "authority_audit_body_sha256", "authority_audit_closing_revision",
    "authorized_transition_from_revision", "authorized_transition_to_revision",
    "consumed_at", "post_started_at", "post_authorized_at", "completed_at",
    "abandoned_at", "outcome_class", "classification_evidence_kind",
    "classification_evidence_body_sha256", "sky_request_id",
    "response_identity_sha256", "abandonment_proof_sha256",
    "consume_transaction_client_request_token_sha256",
    "post_start_transaction_client_request_token_sha256",
    "post_authorization_transaction_client_request_token_sha256",
    "post_classification_transaction_client_request_token_sha256",
)


def _state_rule(
    *,
    owner: str = "unchanged",
    null: Tuple[str, ...] = (),
    nonnull: Tuple[str, ...] = (),
    nonempty: Tuple[str, ...] = (),
    exact: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "owner": owner,
        "null": null,
        "nonnull": nonnull,
        "nonempty": nonempty,
        "exact": {} if exact is None else dict(exact),
    }


_SNAPSHOT_BINDING = (
    "snapshot_id", "snapshot_identity_sha256", "source_volume_id",
    "snapshot_tags_sha256", "delete_not_before", "schedule_arn",
    "schedule_input_sha256", "cleanup_barrier_nonce_sha256",
    "cleanup_lineage_identity_sha256",
)
_SNAPSHOT_LATER = (
    "latest_cleanup_authority_audit_identity_sha256",
    "latest_delete_action_identity_sha256", "latest_delete_request_id",
    "latest_delete_response_sha256", "last_describe_request_id",
    "last_describe_response_sha256", "terminal_evidence_sha256",
)
_LIABILITY_WATCH = (
    "watch_started_at", "watch_not_before", "watch_not_after", "next_scan_at",
)
_LIABILITY_SETTLEMENT = (
    "gpu_liability_reserve_release_identity_sha256", "settlement_identity_sha256",
)
_POST_ALLOCATION_EVIDENCE = (
    "allocation_open_identity_sha256", "charged_interval_started_at",
    "instance_terminal_identity_sha256", "charged_interval_ended_at",
    "spend_allocation_close_identity_sha256", "concurrency_window_identity_sha256",
)

_STATE_RULES: Mapping[str, Mapping[str, Mapping[str, object]]] = {
    "glm52_production_recovery_control": {
        "DORMANT": _state_rule(
            owner="null",
            null=(
                "recovery_barrier_nonce_sha256",
                "support_control_revision_at_seal",
                "support_execution_identity_sha256", "allowed_action_set_sha256",
                "terminal_v2_identity_sha256",
            ),
        ),
        "OWNED": _state_rule(
            owner="nonnull",
            nonnull=(
                "recovery_barrier_nonce_sha256",
                "support_control_revision_at_seal",
                "support_execution_identity_sha256", "allowed_action_set_sha256",
            ),
            null=("terminal_v2_identity_sha256",),
        ),
        "TERMINAL_V2_PUBLISHED": _state_rule(
            owner="nonnull",
            nonnull=(
                "recovery_barrier_nonce_sha256",
                "support_control_revision_at_seal",
                "support_execution_identity_sha256", "allowed_action_set_sha256",
                "terminal_v2_identity_sha256",
            ),
        ),
        "RECOVERY_COMPLETE": _state_rule(
            owner="null",
            nonnull=(
                "recovery_barrier_nonce_sha256",
                "support_control_revision_at_seal",
                "support_execution_identity_sha256", "allowed_action_set_sha256",
                "terminal_v2_identity_sha256",
            ),
        ),
    },
    "glm52_production_finalization_control": {
        "DORMANT": _state_rule(
            owner="null",
            null=(
                "finalization_barrier_nonce_sha256",
                "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256", "h1g_drained_identity_sha256",
            ),
        ),
        "OWNED": _state_rule(
            owner="nonnull",
            nonnull=(
                "finalization_barrier_nonce_sha256",
                "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
            ),
            null=(
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256", "h1g_drained_identity_sha256",
            ),
        ),
        "SUPPORT_FINALIZED": _state_rule(
            owner="nonnull",
            nonnull=(
                "finalization_barrier_nonce_sha256",
                "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
                "support_plane_finalized_identity_sha256",
            ),
            null=("snapshot_disposition_identity_sha256", "h1g_drained_identity_sha256"),
        ),
        "SNAPSHOT_DISPOSITION_RECORDED": _state_rule(
            owner="nonnull",
            nonnull=(
                "finalization_barrier_nonce_sha256",
                "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256",
            ),
            null=("h1g_drained_identity_sha256",),
        ),
        "DRAINED_PUBLISHED": _state_rule(
            owner="null",
            nonnull=(
                "finalization_barrier_nonce_sha256",
                "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256", "h1g_drained_identity_sha256",
            ),
        ),
    },
    "glm52_production_snapshot_cleanup_control": {
        "DORMANT": _state_rule(
            owner="null", null=_SNAPSHOT_BINDING + _SNAPSHOT_LATER,
            exact={
                "cleanup_authority_audit_identities": [],
                "delete_action_identities": [],
                "delete_logical_attempt": 0,
                "delete_call_count": 0,
            },
        ),
        "ARMED": _state_rule(
            owner="null", nonnull=_SNAPSHOT_BINDING, null=_SNAPSHOT_LATER,
            exact={
                "cleanup_authority_audit_identities": [],
                "delete_action_identities": [],
                "delete_logical_attempt": 0,
                "delete_call_count": 0,
            },
        ),
        "OWNED": _state_rule(
            owner="nonnull", nonnull=_SNAPSHOT_BINDING, null=_SNAPSHOT_LATER,
            exact={
                "cleanup_authority_audit_identities": [],
                "delete_action_identities": [],
                "delete_logical_attempt": 0,
                "delete_call_count": 0,
            },
        ),
        "DELETE_POSSIBLY_SENT": _state_rule(
            owner="nonnull", nonnull=_SNAPSHOT_BINDING + (
                "latest_cleanup_authority_audit_identity_sha256",
                "latest_delete_action_identity_sha256",
            ),
            nonempty=(
                "cleanup_authority_audit_identities", "delete_action_identities"
            ),
            null=(
                "latest_delete_request_id", "latest_delete_response_sha256",
                "last_describe_request_id", "last_describe_response_sha256",
                "terminal_evidence_sha256",
            ),
        ),
        "DELETE_RECONCILING": _state_rule(
            owner="nonnull", nonnull=_SNAPSHOT_BINDING + (
                "latest_cleanup_authority_audit_identity_sha256",
                "latest_delete_action_identity_sha256",
            ),
            nonempty=(
                "cleanup_authority_audit_identities", "delete_action_identities"
            ),
            null=("terminal_evidence_sha256",),
        ),
        "DELETED": _state_rule(
            owner="null", nonnull=_SNAPSHOT_BINDING + ("terminal_evidence_sha256",)
        ),
        "ALREADY_ABSENT": _state_rule(
            owner="null", nonnull=_SNAPSHOT_BINDING + ("terminal_evidence_sha256",)
        ),
        "CLEANUP_INCIDENT": _state_rule(
            owner="null", nonnull=_SNAPSHOT_BINDING + ("terminal_evidence_sha256",)
        ),
    },
    "glm52_production_execution": {
        "START_OWNED": _state_rule(
            null=_EXECUTION_AFTER_START, exact={"start_send_stage": "NOT_SENT"}
        ),
        "START_POSSIBLY_SENT": _state_rule(
            nonnull=("start_attempted_at",),
            null=(
                "observed_start_date", "terminal_status", "terminal_observed_at",
                "describe_execution_request_id",
                "describe_execution_response_sha256", "terminal_body_sha256",
                "unresolved_start_incident_body_sha256",
            ),
            exact={"start_send_stage": "POSSIBLY_SENT"},
        ),
        "RUNNING": _state_rule(
            nonnull=("start_attempted_at", "observed_start_date"),
            null=(
                "terminal_status", "terminal_observed_at",
                "describe_execution_request_id",
                "describe_execution_response_sha256", "terminal_body_sha256",
                "unresolved_start_incident_body_sha256",
            ),
            exact={"start_send_stage": "POSSIBLY_SENT"},
        ),
        **{
            state: _state_rule(
                nonnull=(
                    "start_attempted_at", "observed_start_date", "terminal_status",
                    "terminal_observed_at", "describe_execution_request_id",
                    "describe_execution_response_sha256", "terminal_body_sha256",
                ),
                null=("unresolved_start_incident_body_sha256",),
                exact={"start_send_stage": "POSSIBLY_SENT", "terminal_status": state},
            )
            for state in ("SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED")
        },
        "START_POSSIBLY_SENT_UNRESOLVED_INCIDENT": _state_rule(
            nonnull=("start_attempted_at", "unresolved_start_incident_body_sha256"),
            null=(
                "observed_start_date", "terminal_status", "terminal_observed_at",
                "describe_execution_request_id",
                "describe_execution_response_sha256", "terminal_body_sha256",
            ),
            exact={"start_send_stage": "POSSIBLY_SENT"},
        ),
        "ABANDONED_PROVED_NOT_STARTED": _state_rule(
            null=_EXECUTION_AFTER_START, exact={"start_send_stage": "NOT_SENT"}
        ),
    },
    "glm52_production_worker_launch": {
        "PREPARED_NOT_SENT": _state_rule(
            null=_WORKER_AFTER_PREPARED,
            exact={
                "send_stage": "NOT_SENT", "observed_instance_ids": [],
                "run_instances_attempt_evidence": [],
                "same_token_completion_count": 0,
            },
        ),
        "POSSIBLY_SENT": _state_rule(
            nonnull=(
                "gpu_liability_reserve_ledger_identity_sha256", "possibly_sent_at"
            ),
            null=(
                "instance_observation_sha256",
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256", "incident_identity_sha256",
            ),
            exact={"send_stage": "POSSIBLY_SENT", "observed_instance_ids": []},
        ),
        "INSTANCE_OBSERVED": _state_rule(
            nonnull=("possibly_sent_at", "instance_observation_sha256"),
            nonempty=("observed_instance_ids",),
            null=(
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256", "incident_identity_sha256",
            ),
            exact={"send_stage": "POSSIBLY_SENT"},
        ),
        "ALLOCATION_OPEN": _state_rule(
            nonnull=(
                "possibly_sent_at", "instance_observation_sha256",
                "spend_allocation_open_identity_sha256",
            ),
            nonempty=("observed_instance_ids",),
            null=(
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256", "incident_identity_sha256",
            ),
            exact={"send_stage": "POSSIBLY_SENT"},
        ),
        "INSTANCE_TERMINAL": _state_rule(
            nonnull=(
                "possibly_sent_at", "instance_observation_sha256",
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
            ),
            nonempty=("observed_instance_ids",),
            null=("spend_allocation_close_identity_sha256", "incident_identity_sha256"),
            exact={"send_stage": "POSSIBLY_SENT"},
        ),
        "ALLOCATION_CLOSED": _state_rule(
            nonnull=(
                "possibly_sent_at", "instance_observation_sha256",
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256",
            ),
            nonempty=("observed_instance_ids",),
            null=("incident_identity_sha256",),
            exact={"send_stage": "POSSIBLY_SENT"},
        ),
        "REJECTED_NO_INSTANCE": _state_rule(
            nonnull=("possibly_sent_at",),
            nonempty=("run_instances_attempt_evidence",),
            null=(
                "instance_observation_sha256",
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256", "incident_identity_sha256",
            ),
            exact={"send_stage": "POSSIBLY_SENT", "observed_instance_ids": []},
        ),
        "MULTIPLE_INSTANCE_TOKEN_INCIDENT": _state_rule(
            nonnull=(
                "possibly_sent_at", "instance_observation_sha256",
                "incident_identity_sha256",
            ),
            nonempty=("observed_instance_ids",),
            null=(
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256",
            ),
            exact={"send_stage": "POSSIBLY_SENT"},
        ),
        "UNRESOLVED_LAUNCH_INCIDENT": _state_rule(
            nonnull=("possibly_sent_at", "incident_identity_sha256"),
            null=(
                "instance_observation_sha256",
                "spend_allocation_open_identity_sha256",
                "instance_terminal_identity_sha256",
                "spend_allocation_close_identity_sha256",
            ),
            exact={
                "send_stage": "POSSIBLY_SENT", "observed_instance_ids": [],
            },
        ),
        "ABANDONED_NOT_SENT": _state_rule(
            null=_WORKER_AFTER_PREPARED,
            exact={
                "send_stage": "NOT_SENT", "observed_instance_ids": [],
                "run_instances_attempt_evidence": [],
                "same_token_completion_count": 0,
            },
        ),
    },
    "glm52_production_worker_launch_liability": {
        "UNOWNED_NOT_ACTIONABLE": _state_rule(
            owner="null", null=_LIABILITY_WATCH + _LIABILITY_SETTLEMENT + (
                "last_scan_started_at", "last_scan_completed_at",
                "last_scan_evidence_sha256", "incident_identity_sha256",
                "incident_at",
            ),
            exact={
                "observed_instance_ids": [],
                "late_instance_drain_identities": [],
                "late_instance_termination_action_identities": [],
                "late_instance_termination_call_counts": [],
                "post_terminal_allocation_identities": [],
                "spend_close_identities": [],
            },
        ),
        "WATCHING": _state_rule(
            owner="nonnull", nonnull=_LIABILITY_WATCH,
            null=_LIABILITY_SETTLEMENT + (
                "incident_identity_sha256", "incident_at",
            ),
        ),
        "SAME_TOKEN_COMPLETION": _state_rule(
            owner="nonnull", nonnull=_LIABILITY_WATCH,
            null=_LIABILITY_SETTLEMENT + (
                "incident_identity_sha256", "incident_at",
            ),
        ),
        "REJECTION_PROVED_AWAITING_TERMINAL_V2": _state_rule(
            owner="nonnull", nonnull=_LIABILITY_WATCH,
            null=_LIABILITY_SETTLEMENT + (
                "incident_identity_sha256", "incident_at",
            ),
            exact={"observed_instance_ids": []},
        ),
        "LATE_INSTANCE_DRAINING": _state_rule(
            owner="nonnull", nonnull=_LIABILITY_WATCH,
            nonempty=("observed_instance_ids", "late_instance_drain_identities"),
        ),
        "LIABILITY_INCIDENT": _state_rule(
            owner="nonnull", nonnull=_LIABILITY_WATCH + (
                "incident_identity_sha256", "incident_at",
            ),
            null=_LIABILITY_SETTLEMENT,
        ),
        "SETTLED_NO_INSTANCE_REJECTED": _state_rule(
            owner="null", nonnull=_LIABILITY_SETTLEMENT,
            exact={"observed_instance_ids": []},
        ),
        "SETTLED_INSTANCE_CLOSED": _state_rule(
            owner="null", nonnull=_LIABILITY_SETTLEMENT,
            nonempty=("observed_instance_ids", "spend_close_identities"),
        ),
    },
    "glm52_production_post_terminal_allocation": {
        "DISCOVERED": _state_rule(
            owner="nonnull", null=_POST_ALLOCATION_EVIDENCE
        ),
        "ALLOCATION_OPEN": _state_rule(
            owner="nonnull",
            nonnull=(
                "allocation_open_identity_sha256", "charged_interval_started_at"
            ),
            null=_POST_ALLOCATION_EVIDENCE[2:],
        ),
        "INSTANCE_TERMINAL": _state_rule(
            owner="nonnull", nonnull=_POST_ALLOCATION_EVIDENCE[:4],
            null=_POST_ALLOCATION_EVIDENCE[4:],
        ),
        "ALLOCATION_CLOSED": _state_rule(
            owner="null", nonnull=_POST_ALLOCATION_EVIDENCE
        ),
    },
}

for _retained_kind in _RETAINED_ACTION_KINDS:
    _STATE_RULES[_retained_kind] = {
        "ARMED": _state_rule(
            owner="nonnull", null=_RETAINED_ACTION_FUTURE,
            nonnull=("armed_at", "arming_transaction_client_request_token_sha256"),
        ),
        "CONSUMED": _state_rule(
            owner="nonnull",
            nonnull=(
                "authority_audit_body_sha256",
                "authority_audit_closing_revision",
                "authorized_transition_from_revision",
                "authorized_transition_to_revision", "consumed_at",
                "consume_transaction_client_request_token_sha256",
            ),
            null=(
                "completed_at", "response_identity_sha256",
                "reconciliation_identity_sha256",
            ),
        ),
        "COMPLETED": _state_rule(
            owner="nonnull",
            nonnull=(
                "authority_audit_body_sha256",
                "authority_audit_closing_revision",
                "authorized_transition_from_revision",
                "authorized_transition_to_revision", "consumed_at",
                "consume_transaction_client_request_token_sha256",
                "completed_at", "response_identity_sha256",
            ),
            null=("reconciliation_identity_sha256",),
        ),
        "AMBIGUOUS": _state_rule(
            owner="nonnull",
            nonnull=(
                "authority_audit_body_sha256",
                "authority_audit_closing_revision",
                "authorized_transition_from_revision",
                "authorized_transition_to_revision", "consumed_at",
                "consume_transaction_client_request_token_sha256",
                "completed_at", "reconciliation_identity_sha256",
            ),
            null=("response_identity_sha256",),
        ),
        "ABANDONED": _state_rule(
            owner="nonnull",
            null=(
                "authority_audit_body_sha256",
                "authority_audit_closing_revision",
                "authorized_transition_from_revision",
                "authorized_transition_to_revision", "consumed_at",
                "response_identity_sha256",
                "consume_transaction_client_request_token_sha256",
            ),
            nonnull=("completed_at", "reconciliation_identity_sha256"),
        ),
    }

_STATE_RULES["glm52_production_action"] = {
    "ARMED": _state_rule(null=_ACTION_ARMED_FUTURE),
    "CONSUMED": _state_rule(
        nonnull=(
            "owner_invocation_nonce_sha256",
            "authority_audit_body_sha256",
            "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision",
            "consumed_at",
        ),
        null=(
            "post_owner_invocation_nonce_sha256", "post_owner_function_version_arn",
            "post_owner_dispatch_identity_sha256", "post_owner_hard_expires_at",
            "post_started_at", "post_authorized_at", "relay_envelope_sha256",
            "completed_at", "abandoned_at",
            "outcome_class", "classification_evidence_kind",
            "classification_evidence_body_sha256", "sky_request_id",
            "response_identity_sha256", "abandonment_proof_sha256",
        ),
    ),
    "POST_STARTED": _state_rule(
        nonnull=(
            "owner_invocation_nonce_sha256", "consumed_at",
            "post_owner_invocation_nonce_sha256", "post_owner_function_version_arn",
            "post_owner_dispatch_identity_sha256", "post_owner_hard_expires_at",
            "post_started_at", "authority_audit_body_sha256",
            "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision",
        ),
        null=(
            "relay_envelope_sha256",
            "post_authorized_at",
            "completed_at", "outcome_class", "classification_evidence_kind",
            "classification_evidence_body_sha256", "sky_request_id",
            "response_identity_sha256", "abandoned_at",
            "abandonment_proof_sha256",
        ),
    ),
    "POST_AUTHORIZED": _state_rule(
        nonnull=(
            "owner_invocation_nonce_sha256", "consumed_at",
            "post_owner_invocation_nonce_sha256", "post_owner_function_version_arn",
            "post_owner_dispatch_identity_sha256", "post_owner_hard_expires_at",
            "post_started_at", "post_authorized_at", "relay_envelope_sha256",
            "authority_audit_body_sha256", "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision",
        ),
        null=(
            "completed_at", "outcome_class", "classification_evidence_kind",
            "classification_evidence_body_sha256", "sky_request_id",
            "response_identity_sha256", "abandoned_at",
            "abandonment_proof_sha256",
        ),
    ),
    "POST_CLASSIFIED": _state_rule(
        nonnull=(
            "owner_invocation_nonce_sha256", "consumed_at",
            "authority_audit_body_sha256",
            "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision",
            "completed_at", "outcome_class",
            "classification_evidence_kind",
            "classification_evidence_body_sha256",
        ),
        null=("abandoned_at", "abandonment_proof_sha256"),
    ),
    "COMPLETED": _state_rule(
        nonnull=(
            "owner_invocation_nonce_sha256", "consumed_at", "completed_at",
            "response_identity_sha256",
        ),
        null=(
            "post_started_at", "post_authorized_at", "abandoned_at",
            "abandonment_proof_sha256",
        ),
    ),
    "AMBIGUOUS": _state_rule(
        nonnull=("owner_invocation_nonce_sha256", "consumed_at", "completed_at"),
        null=(
            "post_started_at", "post_authorized_at", "abandoned_at",
            "response_identity_sha256", "abandonment_proof_sha256",
        ),
    ),
    "ABANDONED": _state_rule(
        null=(
            "consumed_at", "post_started_at", "post_authorized_at", "completed_at",
            "response_identity_sha256",
        ),
        nonnull=("abandoned_at", "abandonment_proof_sha256"),
    ),
}


def _validate_declarative_state(
    record_type: str, record: Mapping[str, object]
) -> None:
    rules = _STATE_RULES.get(record_type)
    if rules is None:
        return
    rule = rules[record["state"]]
    if rule["owner"] != "unchanged":
        owner_fields = (
            _WORKER_OWNER
            if record_type == "glm52_production_worker_launch"
            else _STANDARD_OWNER
        )
        expected_nonnull = rule["owner"] == "nonnull"
        if any((record.get(field) is not None) != expected_nonnull for field in owner_fields):
            raise RecordValidationError("state owner matrix mismatch")
        if (record.get("owner_attempt") is not None) != expected_nonnull:
            raise RecordValidationError("state owner-attempt matrix mismatch")
    for field in rule["null"]:
        if record[field] is not None:
            raise RecordValidationError("state requires null field: " + field)
    for field in rule["nonnull"]:
        if record[field] is None:
            raise RecordValidationError("state requires evidence field: " + field)
    for field in rule["nonempty"]:
        if type(record[field]) is not list or not record[field]:
            raise RecordValidationError("state requires nonempty array: " + field)
    for field, expected in rule["exact"].items():
        if record[field] != expected:
            raise RecordValidationError("state requires exact field: " + field)


def canonical_record_identity(
    record_type: str, value: Mapping[str, object]
) -> str:
    """Return the validated record's canonical self identity."""

    record = validate_record(record_type, value)
    body = dict(record)
    body.pop("canonical_body_sha256", None)
    return canonical_sha256(body)


def ledger_pk(run_id: str) -> str:
    if run_id != RUN_ID:
        raise RecordValidationError("invalid run_id")
    return "RUN#" + run_id


def ledger_sk(record_type: str, **identity: object) -> str:
    if record_type == "glm52_production_activation_index" and not identity:
        return "ACTIVATION_INDEX"
    activation_id = identity.get("activation_id")
    if type(activation_id) is not str or not activation_id:
        raise RecordValidationError("activation_id is required")
    _validate_key_segment("activation_id", activation_id)

    def ordinal(name: str, *, allow_zero: bool = False) -> str:
        value = identity.get(name)
        minimum = 0 if allow_zero else 1
        if type(value) is not int or value < minimum or value > 99_999_999:
            raise RecordValidationError("%s must be a valid ordinal" % name)
        return "%08d" % value

    base = "ACTIVATION#" + activation_id
    simple = {
        "glm52_production_control": "CONTROL",
        "glm52_production_rollover": "ROLLOVER",
        "glm52_production_operator_disposition": "OPERATOR_DISPOSITION",
        "glm52_production_recovery_control": "RECOVERY_CONTROL",
        "glm52_production_finalization_control": "FINALIZATION_CONTROL",
        "glm52_production_snapshot_cleanup_control": "SNAPSHOT_CLEANUP_CONTROL",
        "glm52_task12_request_job_correlation_v1": (
            "TASK12_REQUEST_JOB_CORRELATION"
        ),
    }
    if record_type in simple and set(identity) == {"activation_id"}:
        return base + "#" + simple[record_type]
    if record_type == "glm52_production_execution" and set(identity) == {
        "activation_id", "epoch"
    }:
        return base + "#EXECUTION#" + ordinal("epoch")
    if record_type in (
        "glm52_production_worker_launch",
        "glm52_production_worker_launch_liability",
        "glm52_production_worker_launch_liability_settlement",
    ) and set(identity) == {"activation_id", "allocation_ordinal"}:
        family = record_type.removeprefix("glm52_production_").upper()
        return base + "#" + family + "#" + ordinal("allocation_ordinal")
    if record_type == "glm52_production_post_terminal_allocation" and set(
        identity
    ) == {"activation_id", "allocation_ordinal", "instance_id"}:
        instance_id = identity["instance_id"]
        if type(instance_id) is not str or not instance_id:
            raise RecordValidationError("instance_id is required")
        _validate_key_segment("instance_id", instance_id)
        return (
            base + "#POST_TERMINAL_ALLOCATION#"
            + ordinal("allocation_ordinal") + "#" + instance_id
        )
    if record_type == "glm52_production_action" and set(identity) == {
        "activation_id", "generation", "action_kind", "attempt"
    }:
        action_kind = identity["action_kind"]
        if type(action_kind) is not str or not action_kind:
            raise RecordValidationError("action_kind is required")
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", action_kind) is None:
            raise RecordValidationError("action_kind is not a closed key segment")
        generation_text = ordinal(
            "generation",
            allow_zero=action_kind in _PRE_GENESIS_ACTION_KINDS,
        )
        return (
            base + "#ACTION#" + generation_text + "#"
            + action_kind + "#" + ordinal("attempt")
        )
    retained = {
        "glm52_production_recovery_action": "RECOVERY_ACTION",
        "glm52_production_finalization_action": "FINALIZATION_ACTION",
    }
    if record_type in retained and set(identity) == {
        "activation_id", "action_kind", "attempt"
    }:
        action_kind = identity["action_kind"]
        if type(action_kind) is not str or not action_kind:
            raise RecordValidationError("action_kind is required")
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", action_kind) is None:
            raise RecordValidationError("action_kind is not a closed key segment")
        return base + "#" + retained[record_type] + "#" + action_kind + "#" + ordinal("attempt")
    if record_type == "glm52_production_snapshot_cleanup_action" and set(
        identity
    ) == {"activation_id", "attempt"}:
        return base + "#SNAPSHOT_CLEANUP_ACTION#" + ordinal("attempt")
    if record_type == "glm52_production_worker_launch_liability_action" and set(
        identity
    ) == {"activation_id", "allocation_ordinal", "action_kind", "attempt"}:
        action_kind = identity["action_kind"]
        if type(action_kind) is not str or not action_kind:
            raise RecordValidationError("action_kind is required")
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", action_kind) is None:
            raise RecordValidationError("action_kind is not a closed key segment")
        return (
            base + "#WORKER_LAUNCH_LIABILITY_ACTION#"
            + ordinal("allocation_ordinal") + "#" + action_kind + "#"
            + ordinal("attempt")
        )
    if record_type == "glm52_production_snapshot_cleanup_transition" and set(
        identity
    ) == {"activation_id", "revision"}:
        return base + "#SNAPSHOT_CLEANUP_TRANSITION#" + ordinal("revision")
    if record_type == "glm52_task12_versioned_writer_control_v1" and set(
        identity
    ) == {"activation_id", "generation", "writer_kind"}:
        writer_kind = identity["writer_kind"]
        if type(writer_kind) is not str or writer_kind not in {
            "TerminalV2",
            "SupportPlaneFinalized",
            "H1GDrained",
            "RecoveryHandoff",
        }:
            raise RecordValidationError("writer_kind is not closed")
        return (
            base + "#TASK12_VERSIONED_WRITER_CONTROL#" + ordinal("generation")
            + "#" + writer_kind.upper()
        )
    raise RecordValidationError("unsupported ledger identity")


def _validate_key_segment(name: str, value: object) -> None:
    if (
        type(value) is not str
        or not value
        or "#" in value
        or any(ord(char) < 0x21 or ord(char) > 0x7E for char in value)
    ):
        raise RecordValidationError(name + " is not a safe ledger-key segment")


def _record_ledger_sk(
    record_type: str, record: Mapping[str, object]
) -> str:
    if record_type == "glm52_production_activation_index":
        return ledger_sk(record_type)
    args: Dict[str, object] = {"activation_id": record["activation_id"]}
    if record_type == "glm52_production_execution":
        args["epoch"] = record["epoch"]
    elif record_type in (
        "glm52_production_worker_launch",
        "glm52_production_worker_launch_liability",
        "glm52_production_worker_launch_liability_settlement",
    ):
        args["allocation_ordinal"] = record["allocation_ordinal"]
    elif record_type == "glm52_production_post_terminal_allocation":
        args.update(
            allocation_ordinal=record["allocation_ordinal"],
            instance_id=record["instance_id"],
        )
    elif record_type == "glm52_production_action":
        args.update(
            generation=record["generation"],
            action_kind=record["action_kind"],
            attempt=record["attempt"],
        )
    elif record_type in (
        "glm52_production_recovery_action",
        "glm52_production_finalization_action",
    ):
        args.update(action_kind=record["action_kind"], attempt=record["attempt"])
    elif record_type == "glm52_production_snapshot_cleanup_action":
        args["attempt"] = record["attempt"]
    elif record_type == "glm52_production_worker_launch_liability_action":
        args.update(
            allocation_ordinal=record["allocation_ordinal"],
            action_kind=record["action_kind"],
            attempt=record["attempt"],
        )
    elif record_type == "glm52_production_snapshot_cleanup_transition":
        args["revision"] = record["to_revision"]
    elif record_type == "glm52_task12_versioned_writer_control_v1":
        args.update(
            generation=record["generation"],
            writer_kind=record["writer_kind"],
        )
    return ledger_sk(record_type, **args)
