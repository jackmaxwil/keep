from __future__ import annotations

import importlib
from copy import deepcopy

import pytest

from glm52_enforcement.canonical import canonical_sha256


RECORD_TYPES = (
    "glm52_production_activation_index",
    "glm52_production_rollover",
    "glm52_production_operator_disposition",
    "glm52_production_control",
    "glm52_production_recovery_control",
    "glm52_production_finalization_control",
    "glm52_production_snapshot_cleanup_control",
    "glm52_production_snapshot_cleanup_transition",
    "glm52_task12_versioned_writer_control_v1",
    "glm52_task12_request_job_correlation_v1",
    "glm52_production_support_plane_finalized",
    "glm52_production_h1g_drained",
    "glm52_production_recovery_action",
    "glm52_production_finalization_action",
    "glm52_production_snapshot_cleanup_action",
    "glm52_production_worker_launch_liability_action",
    "glm52_production_execution",
    "glm52_production_worker_launch",
    "glm52_production_worker_launch_liability",
    "glm52_production_post_terminal_allocation",
    "glm52_production_worker_launch_liability_settlement",
    "glm52_production_action",
    "glm52_production_terminal_v2",
)

SHA = "a" * 64
TS = "2026-07-28T12:00:00Z"


def _activation_index(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_activation_index",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": SHA,
        "current_activation_id": "activation-1",
        "current_activation_ordinal": 1,
        "prior_activation_id": None,
        "prior_activation_terminal_v2_identity": None,
        "prior_h1g_drained_identity": None,
        "prior_spend_ledger_head_identity": None,
        "snapshot_cleanup_lineage_sha256": SHA,
        "revision": 1,
        "updated_at": TS,
    }
    value.update(overrides)
    return value


def _closed_record(record_type: str, **overrides: object) -> dict[str, object]:
    records = importlib.import_module("glm52_enforcement.records")
    if record_type == "glm52_production_h1g_drained":
        from test_glm52_task12_writers import _h1g_drained

        value = deepcopy(_h1g_drained())
        value.update(overrides)
        body = dict(value)
        body.pop("canonical_body_sha256")
        value["canonical_body_sha256"] = canonical_sha256(body)
        return value
    arrays = {
        "snapshot_cleanup_lineage", "cleanup_authority_audit_identities",
        "delete_action_identities", "run_instances_attempt_evidence",
        "observed_instance_ids", "late_instance_drain_identities",
        "late_instance_termination_action_identities",
        "late_instance_termination_call_counts",
        "post_terminal_allocation_identities", "spend_close_identities",
        "post_terminal_allocations", "merged_final_allocations",
        "final_ec2_states", "allocations", "worker_launch_evidence",
        "worker_launch_liabilities", "request_evidence",
    }
    integers = {
        "schema_version", "current_activation_ordinal", "activation_ordinal",
        "prior_activation_ordinal", "index_from_revision", "index_to_revision",
        "active_epoch", "last_sky_post_generation",
        "cleanup_transition_chain_length", "owner_attempt",
        "support_control_revision_at_seal", "teardown_sealed_control_revision",
        "revision", "delete_logical_attempt", "delete_call_count",
        "from_revision", "to_revision", "authority_audit_closing_revision",
        "authorized_transition_from_revision", "authorized_transition_to_revision",
        "attempt", "generation", "allocation_ordinal", "epoch", "owner_epoch",
        "armed_by_epoch", "owner_control_revision", "same_token_completion_count",
        "gpu_liability_reserve_seconds", "scan_interval_seconds",
        "max_discovery_to_termination_seconds", "same_token_completion_attempts",
        "scan_count_current_approval_period", "remaining_approved_gpu_seconds",
    }
    booleans = {"operator_disposition_required"}
    objects = {
        "prior_activation_terminal_v2_identity", "prior_h1g_drained_identity",
        "prior_spend_ledger_head_identity", "terminal_v2_identity",
        "operator_disposition_identity", "prior_snapshot_cleanup_control_identity",
        "cached_success_response_body", "direct_start_response_identity",
        "spend_ledger_head_identity", "handoff", "binding",
        "final_heartbeat_identity", "checkpoint_identity", "cache_identity",
        "training_identity", "evaluation_identity", "drain_identity",
        "terminal_observation_window", "post_terminal_quiescence_evidence",
        "prior_terminal_v1_identity",
    }
    value: dict[str, object] = {}
    for field in records.RECORD_FIELDS[record_type]:
        if field == "schema_version":
            value[field] = 2 if record_type.endswith("terminal_v2") else 1
        elif field == "record_type":
            value[field] = record_type
        elif field == "account_id":
            value[field] = "246813579024"
        elif field == "region":
            value[field] = "us-west-2"
        elif field == "run_id":
            value[field] = "glm52-sky-20260724"
        elif field in arrays:
            value[field] = []
        elif field in objects:
            value[field] = {"identity": SHA}
        elif field in integers:
            value[field] = 1
        elif field in booleans:
            value[field] = False
        elif field.endswith("_array_sha256"):
            value[field] = canonical_sha256(
                value.get(field.removesuffix("_array_sha256"), [])
            )
        elif field.endswith("_sha256"):
            value[field] = SHA
        elif field.endswith("_at") or field.endswith("_not_before") or field.endswith("_not_after"):
            value[field] = TS
        elif field in {"generation_text", "allocation_ordinal_text", "epoch_text"}:
            value[field] = "00000001"
        elif field == "ec2_client_token":
            value[field] = "t" * 64
        else:
            value[field] = "x"
    value.update(overrides)
    retained_defaults = {
        "glm52_production_recovery_action": ("RECOVERY", "REQUEST_CANCEL"),
        "glm52_production_finalization_action": ("FINALIZATION", "SUPPORT_DELETE"),
        "glm52_production_snapshot_cleanup_action": (
            "SNAPSHOT_CLEANUP", "SNAPSHOT_DELETE"
        ),
        "glm52_production_worker_launch_liability_action": (
            "WORKER_LAUNCH_LIABILITY", "SAME_TOKEN_COMPLETE"
        ),
    }
    if record_type in retained_defaults:
        domain, action_kind = retained_defaults[record_type]
        value["authority_domain"] = domain
        value["action_kind"] = action_kind
        if record_type != "glm52_production_worker_launch_liability_action":
            for field in (
                "allocation_ordinal", "allocation_ordinal_text",
                "worker_launch_identity_sha256",
                "worker_launch_liability_identity_sha256",
            ):
                value[field] = None
    if record_type == "glm52_production_action":
        value["action_kind"] = "S3_CREATE"
    if record_type == "glm52_task12_versioned_writer_control_v1":
        value.update(
            generation=1,
            generation_text="00000001",
            writer_kind="TerminalV2",
            campaign_bucket=(
                "keep-glm52-models-246813579024-us-west-2"
            ),
            object_key="terminal/PRODUCTION_TERMINAL_V2.json",
            object_version_id="version-1",
        )
    if record_type == "glm52_task12_request_job_correlation_v1":
        value.update(
            generation=1,
            generation_text="00000001",
            request_ids=["request-1"],
            job_ids=["1"],
            request_states={"request-1": "SUCCEEDED"},
            job_states={"1": "RUNNING"},
        )
    if record_type in {
        "glm52_production_worker_launch",
        "glm52_production_worker_launch_liability",
    }:
        value["gpu_liability_reserve_seconds"] = 900
        value["gpu_liability_reserve_cost_usd"] = "13.76"
        value["ebs_liability_reserve_cost_usd"] = "0.01"
    if record_type == "glm52_production_worker_launch_liability_settlement":
        value["ebs_liability_reserve_cost_usd"] = "0.01"
    if record_type == "glm52_production_rollover":
        value.update(
            activation_ordinal=1,
            index_from_revision=0,
            index_to_revision=1,
            snapshot_cleanup_lineage=[],
            snapshot_cleanup_lineage_sha256=canonical_sha256([]),
            operator_disposition_required=False,
        )
        for field in (
            "prior_activation_id", "prior_activation_ordinal",
            "prior_terminal_v2_key", "prior_terminal_v2_version_id",
            "prior_terminal_v2_body_sha256", "prior_terminal_v2_outcome",
            "prior_final_worker_cardinality", "prior_h1g_drained_key",
            "prior_h1g_drained_version_id", "prior_h1g_drained_body_sha256",
            "prior_spend_ledger_head_identity",
            "prior_snapshot_cleanup_control_identity",
            "operator_disposition_identity",
        ):
            value[field] = None
    if record_type == "glm52_production_terminal_v2":
        value.update(
            outcome="ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH",
            request_cardinality="NOT_APPLICABLE",
            worker_cardinality="ZERO",
            allocations=[],
            allocations_array_sha256=canonical_sha256([]),
            worker_launch_evidence=[],
            worker_launch_evidence_array_sha256=canonical_sha256([]),
            worker_launch_liabilities=[],
            worker_launch_liabilities_array_sha256=canonical_sha256([]),
            request_evidence=[],
            request_evidence_array_sha256=canonical_sha256([]),
            operator_disposition_required=True,
        )
        for field in (
            "handoff", "binding", "final_heartbeat_identity", "checkpoint_identity",
            "cache_identity", "training_identity", "evaluation_identity",
            "drain_identity", "post_terminal_quiescence_evidence",
            "prior_terminal_v1_identity",
        ):
            value[field] = None
    default_states = {
        "glm52_production_recovery_control": "OWNED",
        "glm52_production_finalization_control": "OWNED",
        "glm52_production_snapshot_cleanup_control": "OWNED",
        "glm52_production_recovery_action": "ARMED",
        "glm52_production_finalization_action": "ARMED",
        "glm52_production_snapshot_cleanup_action": "ARMED",
        "glm52_production_worker_launch_liability_action": "ARMED",
        "glm52_production_execution": "START_OWNED",
        "glm52_production_worker_launch": "PREPARED_NOT_SENT",
        "glm52_production_worker_launch_liability": "WATCHING",
        "glm52_production_post_terminal_allocation": "DISCOVERED",
        "glm52_production_action": "ARMED",
    }
    if record_type in default_states:
        value["state"] = default_states[record_type]
    if record_type == "glm52_production_control":
        value["phase"] = "OPEN"
    if record_type == "glm52_production_operator_disposition":
        value["decision"] = "ALLOW_NEW_ACTIVATION"
    if record_type == "glm52_production_worker_launch_liability_settlement":
        value["settlement_kind"] = "NO_INSTANCE_POSITIVE_REJECTION"
    value.update(overrides)
    if record_type in retained_defaults and value["state"] == "ARMED":
        for field in (
            "authority_audit_body_sha256", "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision", "consumed_at", "completed_at",
            "response_identity_sha256", "reconciliation_identity_sha256",
            "consume_transaction_client_request_token_sha256",
        ):
            if field not in overrides:
                value[field] = None
    if record_type == "glm52_production_action" and value["state"] == "ARMED":
        for field in (
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
        ):
            if field not in overrides:
                value[field] = None
    if record_type == "glm52_production_execution" and value["state"] == "START_OWNED":
        value["start_send_stage"] = "NOT_SENT"
        for field in (
            "start_attempted_at", "direct_start_response_identity",
            "direct_start_request_id", "observed_start_date", "terminal_status",
            "terminal_observed_at", "describe_execution_request_id",
            "describe_execution_response_sha256", "terminal_body_sha256",
            "unresolved_start_incident_body_sha256",
        ):
            if field not in overrides:
                value[field] = None
    if (
        record_type == "glm52_production_worker_launch"
        and value["state"] == "PREPARED_NOT_SENT"
    ):
        value["send_stage"] = "NOT_SENT"
        value["observed_instance_ids"] = []
        value["run_instances_attempt_evidence"] = []
        value["same_token_completion_count"] = 0
        for field in (
            "gpu_liability_reserve_ledger_identity_sha256", "possibly_sent_at",
            "direct_run_instances_request_id",
            "direct_run_instances_response_sha256",
            "instance_observation_sha256",
            "spend_allocation_open_identity_sha256",
            "instance_terminal_identity_sha256",
            "spend_allocation_close_identity_sha256", "incident_identity_sha256",
        ):
            if field not in overrides:
                value[field] = None
    if record_type == "glm52_production_recovery_control":
        if value["state"] == "OWNED" and "terminal_v2_identity_sha256" not in overrides:
            value["terminal_v2_identity_sha256"] = None
    if record_type == "glm52_production_finalization_control":
        reached = {
            "OWNED": 0,
            "SUPPORT_FINALIZED": 1,
            "SNAPSHOT_DISPOSITION_RECORDED": 2,
            "DRAINED_PUBLISHED": 3,
        }.get(value["state"], 0)
        for index, field in enumerate(
            (
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256",
                "h1g_drained_identity_sha256",
            ),
            start=1,
        ):
            if index > reached and field not in overrides:
                value[field] = None
    if record_type == "glm52_production_rollover" and value["activation_ordinal"] > 1:
        prior_defaults = {
            "prior_activation_id": "activation-prior",
            "prior_activation_ordinal": value["activation_ordinal"] - 1,
            "prior_terminal_v2_key": "terminal/key",
            "prior_terminal_v2_version_id": "version",
            "prior_terminal_v2_body_sha256": SHA,
            "prior_terminal_v2_outcome": "DRAINED_COMPLETED",
            "prior_final_worker_cardinality": "ZERO",
            "prior_h1g_drained_key": "drained/key",
            "prior_h1g_drained_version_id": "version",
            "prior_h1g_drained_body_sha256": SHA,
            "prior_spend_ledger_head_identity": {"identity": SHA},
            "prior_snapshot_cleanup_control_identity": {"identity": SHA},
        }
        for field, item in prior_defaults.items():
            if field not in overrides:
                value[field] = item
    if record_type in records._STATE_RULES:
        rule = records._STATE_RULES[record_type][value["state"]]
        for field in rule["null"]:
            if field not in overrides:
                value[field] = None
        for field in rule["nonnull"]:
            if field not in overrides and value[field] is None:
                if field.endswith("_sha256"):
                    value[field] = "b" * 64
                elif field in records._TIMESTAMP_FIELDS:
                    value[field] = "2026-07-28T13:00:00Z"
                elif field in records._INT_FIELDS:
                    value[field] = 1
                else:
                    value[field] = "evidence"
        for field in rule["nonempty"]:
            if field not in overrides and not value[field]:
                value[field] = (
                    ["i-1"]
                    if field == "observed_instance_ids"
                    else ["evidence"]
                )
        for field, item in rule["exact"].items():
            if field not in overrides:
                value[field] = deepcopy(item)
    if "canonical_body_sha256" in value:
        body = dict(value)
        body.pop("canonical_body_sha256")
        value["canonical_body_sha256"] = canonical_sha256(body)
    return value


def _rehash(value: dict[str, object]) -> dict[str, object]:
    if "canonical_body_sha256" in value:
        body = dict(value)
        body.pop("canonical_body_sha256")
        value["canonical_body_sha256"] = canonical_sha256(body)
    return value


def test_every_closed_record_rejects_missing_unknown_and_type_confused_fields() -> None:
    try:
        records = importlib.import_module("glm52_enforcement.records")
    except ModuleNotFoundError:
        pytest.fail("glm52_enforcement.records is not implemented")

    assert tuple(records.RECORD_FIELDS) == RECORD_TYPES
    for record_type in RECORD_TYPES:
        candidate = _closed_record(record_type)
        assert records.validate_record(record_type, candidate) == candidate
        missing = dict(candidate)
        missing.pop(next(iter(records.RECORD_FIELDS[record_type])))
        with pytest.raises(records.RecordValidationError, match="schema mismatch"):
            records.validate_record(record_type, missing)
        unknown = dict(candidate)
        unknown["unknown"] = None
        with pytest.raises(records.RecordValidationError, match="schema mismatch"):
            records.validate_record(record_type, unknown)


def test_ledger_key_grammar_binds_body_identity_and_fixed_width_ordinals() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    assert records.ledger_pk("glm52-sky-20260724") == "RUN#glm52-sky-20260724"
    assert records.ledger_sk("glm52_production_activation_index") == "ACTIVATION_INDEX"
    assert records.ledger_sk(
        "glm52_production_execution", activation_id="act-1", epoch=3
    ) == "ACTIVATION#act-1#EXECUTION#00000003"
    assert records.ledger_sk(
        "glm52_production_worker_launch",
        activation_id="act-1",
        allocation_ordinal=2,
    ) == "ACTIVATION#act-1#WORKER_LAUNCH#00000002"
    with pytest.raises(records.RecordValidationError):
        records.ledger_sk(
            "glm52_production_execution",
            activation_id="act-1",
            epoch_text="3",
        )
    activation = _activation_index()
    assert records.validate_record(
        "glm52_production_activation_index",
        activation,
        pk="RUN#glm52-sky-20260724",
        sk="ACTIVATION_INDEX",
    ) == activation
    with pytest.raises(records.RecordValidationError, match="ledger key"):
        records.validate_record(
            "glm52_production_activation_index",
            activation,
            pk="RUN#foreign",
            sk="ACTIVATION_INDEX",
        )


def test_bool_never_satisfies_integer_fields() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    with pytest.raises(records.RecordValidationError, match="integer"):
        records.validate_record(
            "glm52_production_activation_index",
            _activation_index(revision=True),
        )


def test_canonical_self_hash_excludes_only_the_self_hash() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _closed_record("glm52_production_operator_disposition")
    expected = candidate["canonical_body_sha256"]
    assert records.canonical_record_identity(
        "glm52_production_operator_disposition", candidate
    ) == expected
    candidate["decision"] = "DENY_NEW_ACTIVATION"
    with pytest.raises(records.RecordValidationError, match="canonical body"):
        records.validate_record("glm52_production_operator_disposition", candidate)


def test_array_and_lineage_hashes_are_recomputed() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _closed_record("glm52_production_rollover")
    candidate["snapshot_cleanup_lineage"] = [{"activation_ordinal": 1}]
    body = dict(candidate)
    body.pop("canonical_body_sha256")
    candidate["canonical_body_sha256"] = canonical_sha256(body)
    with pytest.raises(records.RecordValidationError, match="lineage"):
        records.validate_record("glm52_production_rollover", candidate)


def test_rollover_is_one_exact_six_key_authority_transaction() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    null_prior = {
        name: None
        for name in (
            "prior_activation_id", "prior_activation_ordinal",
            "prior_terminal_v2_key", "prior_terminal_v2_version_id",
            "prior_terminal_v2_body_sha256", "prior_terminal_v2_outcome",
            "prior_final_worker_cardinality", "prior_h1g_drained_key",
            "prior_h1g_drained_version_id", "prior_h1g_drained_body_sha256",
            "prior_spend_ledger_head_identity",
            "prior_snapshot_cleanup_control_identity",
            "operator_disposition_identity",
        )
    }
    candidate = _closed_record(
        "glm52_production_rollover",
        activation_ordinal=1,
        index_from_revision=0,
        index_to_revision=1,
        snapshot_cleanup_lineage=[],
        snapshot_cleanup_lineage_sha256=canonical_sha256([]),
        operator_disposition_required=False,
        **null_prior,
    )
    assert records.validate_record("glm52_production_rollover", candidate) == candidate
    candidate["prior_activation_id"] = "unexpected"
    body = dict(candidate)
    body.pop("canonical_body_sha256")
    candidate["canonical_body_sha256"] = canonical_sha256(body)
    with pytest.raises(records.RecordValidationError, match="first rollover"):
        records.validate_record("glm52_production_rollover", candidate)


def _no_launch_terminal(**overrides: object) -> dict[str, object]:
    null_markers = {
        name: None
        for name in (
            "handoff", "binding", "final_heartbeat_identity", "checkpoint_identity",
            "cache_identity", "training_identity", "evaluation_identity",
            "drain_identity", "post_terminal_quiescence_evidence",
            "prior_terminal_v1_identity",
        )
    }
    values = {
        "outcome": "ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH",
        "request_cardinality": "NOT_APPLICABLE",
        "worker_cardinality": "ZERO",
        "allocations": [],
        "allocations_array_sha256": canonical_sha256([]),
        "worker_launch_evidence": [],
        "worker_launch_evidence_array_sha256": canonical_sha256([]),
        "worker_launch_liabilities": [],
        "worker_launch_liabilities_array_sha256": canonical_sha256([]),
        "request_evidence": [],
        "request_evidence_array_sha256": canonical_sha256([]),
        "operator_disposition_required": True,
        **null_markers,
        **overrides,
    }
    return _closed_record("glm52_production_terminal_v2", **values)


def test_no_launch_deployment_failure_is_a_closed_terminal_outcome() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _no_launch_terminal()
    assert records.validate_record("glm52_production_terminal_v2", candidate) == candidate
    candidate["allocations"] = [{"instance_id": "i-late"}]
    candidate["allocations_array_sha256"] = canonical_sha256(candidate["allocations"])
    body = dict(candidate)
    body.pop("canonical_body_sha256")
    candidate["canonical_body_sha256"] = canonical_sha256(body)
    with pytest.raises(records.RecordValidationError, match="allocation"):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_v2_request_and_worker_cardinalities_are_orthogonal() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    allocation = _allocation_entry(1)
    request_evidence = _request_evidence("ZERO")
    launch = _worker_launch_entry(1)
    liability = _worker_launch_liability_entry(1)
    candidate = _no_launch_terminal(
        outcome="UNBOUND_LATE_WORKER_INCIDENT",
        handoff={"identity": SHA},
        allocations=[allocation],
        allocations_array_sha256=canonical_sha256([allocation]),
        worker_launch_evidence=[launch],
        worker_launch_evidence_array_sha256=canonical_sha256([launch]),
        worker_launch_liabilities=[liability],
        worker_launch_liabilities_array_sha256=canonical_sha256([liability]),
        worker_cardinality="ONE",
        request_cardinality="ZERO",
        request_evidence=request_evidence,
        request_evidence_array_sha256=canonical_sha256(request_evidence),
        post_terminal_quiescence_evidence={"identity": SHA},
    )
    assert records.validate_record("glm52_production_terminal_v2", candidate) == candidate
    candidate["worker_cardinality"] = "ZERO"
    body = dict(candidate)
    body.pop("canonical_body_sha256")
    candidate["canonical_body_sha256"] = canonical_sha256(body)
    with pytest.raises(records.RecordValidationError, match="worker cardinality"):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_v2_drain_marker_matrix_is_state_specific() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    marker = {
        "key": "marker.json",
        "version_id": "version-1",
        "body_sha256": SHA,
        "canonical_identity_sha256": "b" * 64,
    }
    allocation = _allocation_entry(1)
    launch = _worker_launch_entry(1)
    liability = _worker_launch_liability_entry(1)
    request = _request_evidence("ONE")
    candidate = _no_launch_terminal(
        outcome="DRAINED_COMPLETED",
        handoff={"identity": SHA},
        binding={"identity": SHA},
        allocations=[allocation],
        allocations_array_sha256=canonical_sha256([allocation]),
        worker_cardinality="ONE",
        worker_launch_evidence=[launch],
        worker_launch_evidence_array_sha256=canonical_sha256([launch]),
        worker_launch_liabilities=[liability],
        worker_launch_liabilities_array_sha256=canonical_sha256([liability]),
        request_cardinality="ONE",
        request_evidence=request,
        request_evidence_array_sha256=canonical_sha256(request),
        training_identity=marker,
        evaluation_identity=marker,
        drain_identity=marker,
        operator_disposition_required=False,
    )
    assert records.validate_record("glm52_production_terminal_v2", candidate) == candidate
    candidate["drain_identity"] = None
    body = dict(candidate)
    body.pop("canonical_body_sha256")
    candidate["canonical_body_sha256"] = canonical_sha256(body)
    with pytest.raises(records.RecordValidationError, match="drain"):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_13_77_gpu_reserve_mutant_is_rejected() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _closed_record(
        "glm52_production_worker_launch",
        state="PREPARED_NOT_SENT",
        send_stage="NOT_SENT",
    )
    assert records.validate_record("glm52_production_worker_launch", candidate) == candidate
    candidate["gpu_liability_reserve_cost_usd"] = "13.77"
    with pytest.raises(records.RecordValidationError, match="13.76"):
        records.validate_record("glm52_production_worker_launch", candidate)


@pytest.mark.parametrize(
    ("record_type", "state"),
    [
        ("glm52_production_recovery_control", "OWNED"),
        ("glm52_production_recovery_control", "TERMINAL_V2_PUBLISHED"),
        ("glm52_production_finalization_control", "OWNED"),
        ("glm52_production_finalization_control", "SUPPORT_FINALIZED"),
        ("glm52_production_finalization_control", "SNAPSHOT_DISPOSITION_RECORDED"),
        ("glm52_production_snapshot_cleanup_control", "OWNED"),
        ("glm52_production_snapshot_cleanup_control", "DELETE_POSSIBLY_SENT"),
        ("glm52_production_snapshot_cleanup_control", "DELETE_RECONCILING"),
        ("glm52_production_worker_launch", "PREPARED_NOT_SENT"),
        ("glm52_production_worker_launch", "POSSIBLY_SENT"),
        ("glm52_production_worker_launch", "INSTANCE_OBSERVED"),
        ("glm52_production_worker_launch", "ALLOCATION_OPEN"),
        ("glm52_production_worker_launch", "INSTANCE_TERMINAL"),
        ("glm52_production_worker_launch_liability", "WATCHING"),
        ("glm52_production_worker_launch_liability", "SAME_TOKEN_COMPLETION"),
        (
            "glm52_production_worker_launch_liability",
            "REJECTION_PROVED_AWAITING_TERMINAL_V2",
        ),
        ("glm52_production_worker_launch_liability", "LATE_INSTANCE_DRAINING"),
        ("glm52_production_worker_launch_liability", "LIABILITY_INCIDENT"),
        ("glm52_production_post_terminal_allocation", "DISCOVERED"),
        ("glm52_production_post_terminal_allocation", "ALLOCATION_OPEN"),
        ("glm52_production_post_terminal_allocation", "INSTANCE_TERMINAL"),
    ],
)
def test_owner_takeover_from_every_nonterminal_state_has_no_ownerless_gap(
    record_type: str, state: str
) -> None:
    records = importlib.import_module("glm52_enforcement.records")
    transitions = importlib.import_module("glm52_enforcement.transitions")
    before = _state_fixture(records, record_type, state)
    before["revision"] = 4
    before = _rehash(before)
    after = deepcopy(before)
    after.update(owner_attempt=2, owner_dispatch_identity_sha256="b" * 64,
                 owner_invocation_nonce_sha256="c" * 64,
                 owner_hard_expires_at="2026-07-28T13:00:00Z",
                 revision=5, updated_at="2026-07-28T12:30:00Z")
    if record_type == "glm52_production_worker_launch":
        after.update(
            owner_principal_arn="arn:new-owner",
            owner_function_version_arn="arn:new-function",
        )
        owner_identity_field = "owner_principal_arn"
    else:
        after.update(
            owner_execution_arn="arn:new-owner",
            owner_state_machine_version_arn="arn:new-machine",
        )
        owner_identity_field = "owner_execution_arn"
    if "canonical_body_sha256" in after:
        body = dict(after)
        body.pop("canonical_body_sha256")
        after["canonical_body_sha256"] = canonical_sha256(body)
    assert transitions.validate_owner_takeover(record_type, before, after) == after
    ownerless = deepcopy(after)
    ownerless[owner_identity_field] = None
    with pytest.raises(transitions.TransitionValidationError, match="owner"):
        transitions.validate_owner_takeover(record_type, before, ownerless)


def test_stale_or_nonincrementing_revision_is_rejected() -> None:
    transitions = importlib.import_module("glm52_enforcement.transitions")
    before = _closed_record(
        "glm52_production_recovery_control", state="OWNED", revision=3
    )
    after = deepcopy(before)
    after.update(
        state="TERMINAL_V2_PUBLISHED",
        terminal_v2_identity_sha256="b" * 64,
        revision=3,
    )
    with pytest.raises(transitions.TransitionValidationError, match="revision"):
        transitions.validate_transition(
            "glm52_production_recovery_control", before, after
        )


def test_prior_evidence_cannot_be_cleared_on_progress_or_takeover() -> None:
    transitions = importlib.import_module("glm52_enforcement.transitions")
    before = _closed_record(
        "glm52_production_finalization_control",
        state="SUPPORT_FINALIZED",
        support_plane_finalized_identity_sha256=SHA,
        revision=2,
    )
    after = deepcopy(before)
    after.update(
        state="SNAPSHOT_DISPOSITION_RECORDED",
        support_plane_finalized_identity_sha256=None,
        revision=3,
    )
    with pytest.raises(transitions.TransitionValidationError, match="evidence"):
        transitions.validate_transition(
            "glm52_production_finalization_control", before, after
        )


@pytest.mark.parametrize(
    ("record_type", "dormant_evidence"),
    [
        (
            "glm52_production_recovery_control",
            (
                "recovery_barrier_nonce_sha256", "support_control_revision_at_seal",
                "support_execution_identity_sha256", "allowed_action_set_sha256",
                "terminal_v2_identity_sha256",
            ),
        ),
        (
            "glm52_production_finalization_control",
            (
                "finalization_barrier_nonce_sha256",
                "teardown_sealed_control_revision", "terminal_v2_identity_sha256",
                "support_plane_finalized_identity_sha256",
                "snapshot_disposition_identity_sha256", "h1g_drained_identity_sha256",
            ),
        ),
    ],
)
def test_state_specific_nullability_rejects_partially_owned_dormant_controls(
    record_type: str, dormant_evidence: tuple[str, ...]
) -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _closed_record(record_type, state="DORMANT", owner_attempt=None)
    for field in (
        "owner_execution_arn", "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256",
        "owner_hard_expires_at", *dormant_evidence,
    ):
        candidate[field] = None
    assert records.validate_record(record_type, candidate) == candidate
    candidate["owner_execution_arn"] = "arn:partial"
    with pytest.raises(records.RecordValidationError, match="owner"):
        records.validate_record(record_type, candidate)


def test_three_activation_snapshot_cleanup_lineage_is_transitive() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    lineage = [
        {
            "activation_ordinal": ordinal,
            "cleanup_control_root_identity_sha256": chr(96 + ordinal) * 64,
            "cleanup_transition_chain_head_sha256": str(ordinal) * 64,
            "cleanup_transition_chain_length": ordinal,
            "transition_identities": [str(index) * 64 for index in range(1, ordinal + 1)],
        }
        for ordinal in (1, 2, 3)
    ]
    candidate = _closed_record(
        "glm52_production_rollover",
        activation_ordinal=4,
        prior_activation_ordinal=3,
        snapshot_cleanup_lineage=lineage,
        snapshot_cleanup_lineage_sha256=canonical_sha256(lineage),
        operator_disposition_required=True,
    )
    assert records.validate_record("glm52_production_rollover", candidate) == candidate
    candidate["snapshot_cleanup_lineage"] = [lineage[0], lineage[2]]
    candidate["snapshot_cleanup_lineage_sha256"] = canonical_sha256(
        candidate["snapshot_cleanup_lineage"]
    )
    body = dict(candidate)
    body.pop("canonical_body_sha256")
    candidate["canonical_body_sha256"] = canonical_sha256(body)
    with pytest.raises(records.RecordValidationError, match="transitive"):
        records.validate_record("glm52_production_rollover", candidate)


def test_every_declared_state_edge_passes_and_every_other_edge_fails() -> None:
    try:
        transitions = importlib.import_module("glm52_enforcement.transitions")
    except ModuleNotFoundError:
        pytest.fail("glm52_enforcement.transitions is not implemented")
    expected = {
        "glm52_production_control": {
            ("OPEN", "RECOVERY_SEALING"),
            ("RECOVERY_SEALING", "RECOVERY_COMPLETE"),
            ("RECOVERY_COMPLETE", "TEARDOWN_SEALING"),
            ("TEARDOWN_SEALING", "TEARDOWN_SEALED"),
        },
        "glm52_production_recovery_control": {
            ("DORMANT", "OWNED"),
            ("OWNED", "TERMINAL_V2_PUBLISHED"),
            ("TERMINAL_V2_PUBLISHED", "RECOVERY_COMPLETE"),
        },
        "glm52_production_finalization_control": {
            ("DORMANT", "OWNED"),
            ("OWNED", "SUPPORT_FINALIZED"),
            ("SUPPORT_FINALIZED", "SNAPSHOT_DISPOSITION_RECORDED"),
            ("SNAPSHOT_DISPOSITION_RECORDED", "DRAINED_PUBLISHED"),
        },
        "glm52_production_snapshot_cleanup_control": {
            ("DORMANT", "ARMED"),
            ("ARMED", "OWNED"),
            ("OWNED", "DELETE_POSSIBLY_SENT"),
            ("OWNED", "ALREADY_ABSENT"),
            ("DELETE_POSSIBLY_SENT", "DELETE_RECONCILING"),
            ("DELETE_RECONCILING", "DELETE_POSSIBLY_SENT"),
            ("DELETE_RECONCILING", "DELETED"),
            ("DELETE_RECONCILING", "ALREADY_ABSENT"),
            ("DELETE_RECONCILING", "CLEANUP_INCIDENT"),
        },
        "glm52_production_execution": {
            ("START_OWNED", "START_POSSIBLY_SENT"),
            ("START_OWNED", "ABANDONED_PROVED_NOT_STARTED"),
            ("START_POSSIBLY_SENT", "RUNNING"),
            ("START_POSSIBLY_SENT", "START_POSSIBLY_SENT_UNRESOLVED_INCIDENT"),
            ("RUNNING", "SUCCEEDED"), ("RUNNING", "FAILED"),
            ("RUNNING", "TIMED_OUT"), ("RUNNING", "ABORTED"),
        },
        "glm52_production_worker_launch": {
            ("PREPARED_NOT_SENT", "POSSIBLY_SENT"),
            ("PREPARED_NOT_SENT", "ABANDONED_NOT_SENT"),
            ("POSSIBLY_SENT", "INSTANCE_OBSERVED"),
            ("POSSIBLY_SENT", "REJECTED_NO_INSTANCE"),
            ("POSSIBLY_SENT", "MULTIPLE_INSTANCE_TOKEN_INCIDENT"),
            ("POSSIBLY_SENT", "UNRESOLVED_LAUNCH_INCIDENT"),
            ("INSTANCE_OBSERVED", "ALLOCATION_OPEN"),
            ("ALLOCATION_OPEN", "INSTANCE_TERMINAL"),
            ("INSTANCE_TERMINAL", "ALLOCATION_CLOSED"),
        },
        "glm52_production_worker_launch_liability": {
            ("UNOWNED_NOT_ACTIONABLE", "WATCHING"),
            ("WATCHING", "SAME_TOKEN_COMPLETION"),
            ("SAME_TOKEN_COMPLETION", "WATCHING"),
            (
                "SAME_TOKEN_COMPLETION",
                "REJECTION_PROVED_AWAITING_TERMINAL_V2",
            ),
            ("WATCHING", "REJECTION_PROVED_AWAITING_TERMINAL_V2"),
            (
                "REJECTION_PROVED_AWAITING_TERMINAL_V2",
                "SETTLED_NO_INSTANCE_REJECTED",
            ),
            ("WATCHING", "LATE_INSTANCE_DRAINING"),
            ("LIABILITY_INCIDENT", "LATE_INSTANCE_DRAINING"),
            ("LATE_INSTANCE_DRAINING", "WATCHING"),
            ("LATE_INSTANCE_DRAINING", "LIABILITY_INCIDENT"),
            ("WATCHING", "SETTLED_INSTANCE_CLOSED"),
            ("WATCHING", "LIABILITY_INCIDENT"),
            ("SAME_TOKEN_COMPLETION", "LIABILITY_INCIDENT"),
            (
                "LIABILITY_INCIDENT",
                "REJECTION_PROVED_AWAITING_TERMINAL_V2",
            ),
            ("LIABILITY_INCIDENT", "SETTLED_INSTANCE_CLOSED"),
        },
        "glm52_production_post_terminal_allocation": {
            ("DISCOVERED", "ALLOCATION_OPEN"),
            ("ALLOCATION_OPEN", "INSTANCE_TERMINAL"),
            ("INSTANCE_TERMINAL", "ALLOCATION_CLOSED"),
        },
    }
    action_edges = {
        ("ARMED", "CONSUMED"), ("CONSUMED", "COMPLETED"),
        ("CONSUMED", "AMBIGUOUS"), ("ARMED", "ABANDONED"),
    }
    for kind in (
        "glm52_production_recovery_action",
        "glm52_production_finalization_action",
        "glm52_production_snapshot_cleanup_action",
        "glm52_production_worker_launch_liability_action",
    ):
        expected[kind] = action_edges
    expected["glm52_production_action"] = action_edges | {
        ("CONSUMED", "POST_STARTED"),
        ("CONSUMED", "POST_CLASSIFIED"),
        ("POST_STARTED", "POST_AUTHORIZED"),
        ("POST_STARTED", "POST_CLASSIFIED"),
        ("POST_AUTHORIZED", "POST_CLASSIFIED"),
    }
    for record_type, edges in expected.items():
        assert transitions.TRANSITIONS[record_type] == frozenset(edges)
        states = {state for edge in edges for state in edge}
        for state in states:
            assert transitions.allowed_targets(record_type, state) == frozenset(
                target for source, target in edges if source == state
            )


def test_no_launch_terminal_rejects_nonnull_drain() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    marker = {
        "key": "marker.json",
        "version_id": "version-1",
        "body_sha256": SHA,
        "canonical_identity_sha256": "b" * 64,
    }

    no_launch_nonnull_drain = _no_launch_terminal(drain_identity=marker)
    with pytest.raises(records.RecordValidationError):
        records.validate_record(
            "glm52_production_terminal_v2", no_launch_nonnull_drain
        )


def test_unresolved_no_instance_requires_handoff() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    zero_evidence = [
        {"kind": "ZERO_MATCH_SCAN"},
        {"kind": "QUIESCENCE_SNAPSHOT"},
        {"kind": "QUIESCENCE_SNAPSHOT"},
    ]
    unresolved_no_handoff = _no_launch_terminal(
        outcome="WORKER_LAUNCH_UNRESOLVED_NO_INSTANCE_INCIDENT",
        handoff=None,
        binding=None,
        request_cardinality="ZERO",
        request_evidence=zero_evidence,
        request_evidence_array_sha256=canonical_sha256(zero_evidence),
        worker_launch_evidence=[
            {"kind": "UNRESOLVED_LAUNCH", "allocation_ordinal": 1}
        ],
        worker_launch_evidence_array_sha256=canonical_sha256(
            [{"kind": "UNRESOLVED_LAUNCH", "allocation_ordinal": 1}]
        ),
        worker_launch_liabilities=[
            {"allocation_ordinal": 1, "state": "WATCHING"}
        ],
        worker_launch_liabilities_array_sha256=canonical_sha256(
            [{"allocation_ordinal": 1, "state": "WATCHING"}]
        ),
    )
    with pytest.raises(records.RecordValidationError):
        records.validate_record(
            "glm52_production_terminal_v2", unresolved_no_handoff
        )


def test_one_request_requires_quiescence_evidence() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    one_request_without_quiescence = _no_launch_terminal(
        outcome="ACCEPTED_REQUEST_FAILED_NO_JOB",
        handoff={"identity": SHA},
        request_cardinality="ONE",
        request_evidence=[{"kind": "EXACT_REQUEST"}],
        request_evidence_array_sha256=canonical_sha256(
            [{"kind": "EXACT_REQUEST"}]
        ),
    )
    with pytest.raises(records.RecordValidationError):
        records.validate_record(
            "glm52_production_terminal_v2", one_request_without_quiescence
        )


def test_worker_launch_rejects_generation_zero() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    worker_generation_zero = _closed_record(
        "glm52_production_worker_launch",
        generation=0,
        generation_text="00000000",
    )
    with pytest.raises(records.RecordValidationError):
        records.validate_record(
            "glm52_production_worker_launch", worker_generation_zero
        )


def test_possibly_sent_requires_send_evidence() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    possibly_sent_without_send_evidence = _closed_record(
        "glm52_production_worker_launch",
        state="POSSIBLY_SENT",
        send_stage="NOT_SENT",
        possibly_sent_at=None,
        run_instances_attempt_evidence=[],
    )
    with pytest.raises(records.RecordValidationError):
        records.validate_record(
            "glm52_production_worker_launch",
            possibly_sent_without_send_evidence,
        )


def test_transition_rejects_changed_expected_execution_arn() -> None:
    transitions = importlib.import_module("glm52_enforcement.transitions")
    before = _closed_record(
        "glm52_production_execution", state="START_OWNED", revision=4
    )
    after = deepcopy(before)
    after.update(
        state="START_POSSIBLY_SENT",
        expected_execution_arn="arn:changed",
        start_send_stage="POSSIBLY_SENT",
        start_attempted_at=TS,
        revision=5,
        updated_at="2026-07-28T12:30:00Z",
    )
    with pytest.raises(transitions.TransitionValidationError):
        transitions.validate_transition(
            "glm52_production_execution", before, after
        )


def _state_fixture(
    records: object, record_type: str, state: str
) -> dict[str, object]:
    value = _closed_record(record_type, state=state)
    if (
        record_type == "glm52_production_action"
        and state in {"POST_STARTED", "POST_AUTHORIZED", "POST_CLASSIFIED"}
    ):
        value["action_kind"] = "SKY_POST"
    rule = records._STATE_RULES[record_type][state]
    owner_fields = (
        records._WORKER_OWNER
        if record_type == "glm52_production_worker_launch"
        else records._STANDARD_OWNER
    )
    if rule["owner"] in {"null", "nonnull"}:
        owner_value = rule["owner"] == "nonnull"
        for field in owner_fields:
            value[field] = (
                ("b" * 64 if field.endswith("_sha256") else (
                    "2026-07-28T13:00:00Z"
                    if field.endswith("_at")
                    else "arn:owner"
                ))
                if owner_value
                else None
            )
        value["owner_attempt"] = 1 if owner_value else None
    for field in rule["null"]:
        value[field] = None
    for field in rule["nonnull"]:
        if value[field] is None:
            if field.endswith("_sha256"):
                value[field] = "b" * 64
            elif field in records._TIMESTAMP_FIELDS:
                value[field] = "2026-07-28T13:00:00Z"
            elif field in {
                "authority_audit_closing_revision",
                "authorized_transition_from_revision",
                "authorized_transition_to_revision",
            }:
                value[field] = 1
            else:
                value[field] = "evidence"
    for field in rule["nonempty"]:
        if field == "observed_instance_ids":
            value[field] = ["i-1", "i-2"]
        elif field == "late_instance_termination_call_counts":
            value[field] = [1]
        else:
            value[field] = ["evidence"]
    for field, exact in rule["exact"].items():
        value[field] = deepcopy(exact)
    if (
        record_type == "glm52_production_action"
        and state == "POST_CLASSIFIED"
    ):
        value.update(
            outcome_class="ACCEPTED",
            classification_evidence_kind="RELAY_RESPONSE",
            classification_evidence_body_sha256=SHA,
            response_identity_sha256=SHA,
            sky_request_id="sky-request-1",
        )
    return _rehash(value)


def _transition_fixture(
    records: object,
    record_type: str,
    source: str,
    target: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if record_type == "glm52_production_control":
        before = _closed_record(record_type, phase=source, revision=4)
        after = deepcopy(before)
        after.update(
            phase=target,
            revision=5,
            updated_at="2026-07-28T13:00:00Z",
        )
        return before, _rehash(after)
    before = _state_fixture(records, record_type, source)
    before["revision"] = 4
    before = _rehash(before)
    after = deepcopy(before)
    rule = records._STATE_RULES[record_type][target]
    owner_fields = (
        records._WORKER_OWNER
        if record_type == "glm52_production_worker_launch"
        else records._STANDARD_OWNER
    )
    if rule["owner"] in {"null", "nonnull"}:
        has_owner = rule["owner"] == "nonnull"
        source_owner = records._STATE_RULES[record_type][source]["owner"]
        if source_owner != rule["owner"]:
            for field in owner_fields:
                after[field] = (
                    (
                        "c" * 64
                        if field.endswith("_sha256")
                        else (
                            "2026-07-28T14:00:00Z"
                            if field.endswith("_at")
                            else "arn:transition-owner"
                        )
                    )
                    if has_owner
                    else None
                )
            after["owner_attempt"] = 1 if has_owner else None
    for field in rule["null"]:
        after[field] = None
    for field in rule["nonnull"]:
        if after[field] is None:
            if field.endswith("_sha256"):
                after[field] = "c" * 64
            elif field in records._TIMESTAMP_FIELDS:
                after[field] = "2026-07-28T14:00:00Z"
            elif field in records._INT_FIELDS:
                after[field] = 2
            else:
                after[field] = "transition-evidence"
    for field in rule["nonempty"]:
        if not after[field]:
            after[field] = (
                ["i-1"] if field == "observed_instance_ids" else ["evidence"]
            )
    for field, exact in rule["exact"].items():
        after[field] = deepcopy(exact)
    after.update(state=target, revision=5)
    if "updated_at" in after:
        after["updated_at"] = "2026-07-28T13:00:00Z"
    if record_type == "glm52_production_action":
        action_kind = (
            "SKY_POST"
            if (source, target)
            in {
                ("CONSUMED", "POST_STARTED"),
                ("CONSUMED", "POST_CLASSIFIED"),
                ("POST_STARTED", "POST_AUTHORIZED"),
                ("POST_STARTED", "POST_CLASSIFIED"),
                ("POST_AUTHORIZED", "POST_CLASSIFIED"),
            }
            else "S3_CREATE"
        )
        before["action_kind"] = action_kind
        after["action_kind"] = action_kind
        if target == "POST_CLASSIFIED":
            if source in {"CONSUMED", "POST_STARTED"}:
                after.update(
                    outcome_class="PROVED_NOT_SENT_OWNER_DIED",
                    classification_evidence_kind="OWNER_DEATH_PROOF",
                    classification_evidence_body_sha256=SHA,
                    response_identity_sha256=None,
                    sky_request_id=None,
                )
            else:
                after.update(
                    outcome_class="ACCEPTED",
                    classification_evidence_kind="RELAY_RESPONSE",
                    classification_evidence_body_sha256=SHA,
                    response_identity_sha256=SHA,
                    sky_request_id="sky-request-1",
                )
        before = _rehash(before)
    return before, _rehash(after)


def test_every_state_has_exhaustive_null_owner_and_evidence_matrix() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    transitions = importlib.import_module("glm52_enforcement.transitions")
    expected_record_types = {
        "glm52_production_recovery_control",
        "glm52_production_finalization_control",
        "glm52_production_snapshot_cleanup_control",
        "glm52_production_recovery_action",
        "glm52_production_finalization_action",
        "glm52_production_snapshot_cleanup_action",
        "glm52_production_worker_launch_liability_action",
        "glm52_production_execution",
        "glm52_production_worker_launch",
        "glm52_production_worker_launch_liability",
        "glm52_production_post_terminal_allocation",
        "glm52_production_action",
    }
    assert set(records._STATE_RULES) == expected_record_types
    for record_type, state_rules in records._STATE_RULES.items():
        declared_states = {
            state for edge in transitions.TRANSITIONS[record_type] for state in edge
        }
        assert set(state_rules) == declared_states
        for state, rule in state_rules.items():
            candidate = _state_fixture(records, record_type, state)
            assert records.validate_record(record_type, candidate) == candidate
            for field in rule["null"]:
                mutant = deepcopy(candidate)
                mutant[field] = (
                    "b" * 64 if field.endswith("_sha256") else (
                        1 if field.endswith("_revision") else "premature"
                    )
                )
                _rehash(mutant)
                with pytest.raises(records.RecordValidationError):
                    records.validate_record(record_type, mutant)
            for field in rule["nonnull"]:
                mutant = deepcopy(candidate)
                mutant[field] = None
                _rehash(mutant)
                with pytest.raises(records.RecordValidationError):
                    records.validate_record(record_type, mutant)
            for field in rule["nonempty"]:
                mutant = deepcopy(candidate)
                mutant[field] = []
                _rehash(mutant)
                with pytest.raises(records.RecordValidationError):
                    records.validate_record(record_type, mutant)
            for field, exact in rule["exact"].items():
                mutant = deepcopy(candidate)
                mutant[field] = (
                    exact + 1 if type(exact) is int else (
                        ["mutant"] if type(exact) is list else "mutant"
                    )
                )
                _rehash(mutant)
                with pytest.raises(records.RecordValidationError):
                    records.validate_record(record_type, mutant)
            if rule["owner"] == "nonnull":
                mutant = deepcopy(candidate)
                owner_field = (
                    records._WORKER_OWNER[0]
                    if record_type == "glm52_production_worker_launch"
                    else records._STANDARD_OWNER[0]
                )
                mutant[owner_field] = None
                _rehash(mutant)
                with pytest.raises(records.RecordValidationError):
                    records.validate_record(record_type, mutant)


def test_validate_transition_accepts_every_edge_rejects_every_other_pair_and_mutant() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    transitions = importlib.import_module("glm52_enforcement.transitions")
    assert set(transitions.EDGE_WRITABLE_FIELDS) == set(transitions.TRANSITIONS)
    assert set(transitions.EDGE_IMMUTABLE_FIELDS) == set(transitions.TRANSITIONS)
    for record_type, edges in transitions.TRANSITIONS.items():
        assert set(transitions.EDGE_WRITABLE_FIELDS[record_type]) == set(edges)
        assert set(transitions.EDGE_IMMUTABLE_FIELDS[record_type]) == set(edges)
        for edge in edges:
            before, after = _transition_fixture(
                records, record_type, edge[0], edge[1]
            )
            assert transitions.validate_transition(
                record_type, before, after
            ) == after
            immutable = transitions.EDGE_IMMUTABLE_FIELDS[record_type][edge]
            mutable_identity = next(
                (
                    field
                    for field in sorted(immutable)
                    if field in before
                    and type(before[field]) is str
                    and field not in {"record_type"}
                ),
                None,
            )
            if mutable_identity is not None:
                mutant = deepcopy(after)
                mutant[mutable_identity] = (
                    "b" * 64
                    if mutable_identity.endswith("_sha256")
                    else "mutated"
                )
                _rehash(mutant)
                with pytest.raises(transitions.TransitionValidationError):
                    transitions.validate_transition(
                        record_type, before, mutant
                    )
        states = {state for edge in edges for state in edge}
        for source in states:
            for target in states:
                if (source, target) in edges:
                    continue
                before, after = _transition_fixture(
                    records, record_type, source, target
                )
                with pytest.raises(transitions.TransitionValidationError):
                    transitions.validate_transition(
                        record_type, before, after
                    )


def test_transition_append_only_and_action_domain_policies_are_enforced() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    transitions = importlib.import_module("glm52_enforcement.transitions")
    before, after = _transition_fixture(
        records,
        "glm52_production_snapshot_cleanup_control",
        "DELETE_RECONCILING",
        "DELETE_POSSIBLY_SENT",
    )
    before["cleanup_authority_audit_identities"] = ["first"]
    after["cleanup_authority_audit_identities"] = ["replacement"]
    with pytest.raises(
        transitions.TransitionValidationError, match="append-only"
    ):
        transitions.validate_transition(
            "glm52_production_snapshot_cleanup_control", before, after
        )
    for edge, action_kind in (
        (("CONSUMED", "POST_STARTED"), "S3_CREATE"),
        (("CONSUMED", "COMPLETED"), "SKY_POST"),
    ):
        before, after = _transition_fixture(
            records, "glm52_production_action", edge[0], edge[1]
        )
        before["action_kind"] = action_kind
        after["action_kind"] = action_kind
        with pytest.raises(
            transitions.TransitionValidationError, match="SKY_POST"
        ):
            transitions.validate_transition(
                "glm52_production_action", before, after
            )
    for edge in (("ARMED", "CONSUMED"), ("ARMED", "ABANDONED")):
        for action_kind in ("S3_CREATE", "SKY_POST"):
            before, after = _transition_fixture(
                records, "glm52_production_action", edge[0], edge[1]
            )
            before["action_kind"] = action_kind
            after["action_kind"] = action_kind
            assert transitions.validate_transition(
                "glm52_production_action", before, after
            ) == after


def _nested_entry(**values: object) -> dict[str, object]:
    entry = dict(values)
    entry["canonical_entry_sha256"] = canonical_sha256(entry)
    return entry


def _rehash_entry(entry: dict[str, object]) -> None:
    entry["canonical_entry_sha256"] = canonical_sha256(
        {
            field: value
            for field, value in entry.items()
            if field != "canonical_entry_sha256"
        }
    )


def _allocation_entry(ordinal: int) -> dict[str, object]:
    return _nested_entry(
        allocation_ordinal=ordinal,
        allocation_ordinal_text=f"{ordinal:08d}",
        instance_id=f"i-{ordinal}",
        instance_tags_sha256=SHA,
        allocation_open_identity_sha256=SHA,
        allocation_close_identity_sha256=SHA,
        charged_interval_started_at="2026-07-28T12:00:00Z",
        charged_interval_ended_at="2026-07-28T12:10:00Z",
        terminal_state="terminated",
        instance_terminal_identity_sha256=SHA,
        concurrency_window_identity_sha256=SHA,
    )


def _owner_history_entry(
    attempt: int = 1,
    hard_expires_at: str = "2026-07-28T12:30:00Z",
) -> dict[str, object]:
    return _nested_entry(
        owner_attempt=attempt,
        owner_principal_arn="arn:owner",
        owner_function_version_arn="arn:function:1",
        owner_dispatch_identity_sha256=SHA,
        owner_invocation_nonce_sha256=SHA,
        owner_hard_expires_at=hard_expires_at,
    )


def _run_instances_attempt(
    outcome: str = "DIRECT_SUCCESS",
    attempt: int = 1,
    attempted_at: str = "2026-07-28T12:01:00Z",
) -> dict[str, object]:
    ambiguous = outcome == "TRANSPORT_AMBIGUOUS"
    return _nested_entry(
        attempt=attempt,
        attempted_at=attempted_at,
        outcome=outcome,
        request_id=None if ambiguous else "ec2-request-1",
        response_sha256=None if ambiguous else SHA,
        transport_error_sha256=SHA if ambiguous else None,
    )


def _worker_launch_entry(
    ordinal: int,
    kind: str = "EXACT_SINGLE_INSTANCE",
    *,
    observed_instance_ids=None,
) -> dict[str, object]:
    sent = kind != "PREPARED_PROVED_NOT_SENT"
    ambiguous = kind == "UNRESOLVED_LAUNCH"
    multiple = kind == "MULTIPLE_INSTANCE_TOKEN"
    rejection = kind == "POSITIVE_SERVICE_REJECTION_NO_INSTANCE"
    if observed_instance_ids is None:
        if multiple:
            observed_instance_ids = ["i-1", "i-2"]
        elif kind == "EXACT_SINGLE_INSTANCE":
            observed_instance_ids = [f"i-{ordinal}"]
        else:
            observed_instance_ids = []
    attempt_outcome = (
        "TRANSPORT_AMBIGUOUS"
        if ambiguous
        else (
            "POSITIVE_SERVICE_REJECTION"
            if rejection
            else "DIRECT_SUCCESS"
        )
    )
    has_closed_instance = kind in {
        "EXACT_SINGLE_INSTANCE", "MULTIPLE_INSTANCE_TOKEN"
    }
    direct = sent and not ambiguous
    return _nested_entry(
        kind=kind,
        allocation_ordinal=ordinal,
        allocation_ordinal_text=f"{ordinal:08d}",
        ec2_client_token="t" * 64,
        launch_parameters_sha256=SHA,
        expected_worker_tags_sha256=SHA,
        worker_launch_identity_sha256=SHA,
        prior_worker_launch_identity_sha256=None if ordinal == 1 else SHA,
        prior_instance_terminal_identity_sha256=None if ordinal == 1 else SHA,
        prior_spend_allocation_close_identity_sha256=(
            None if ordinal == 1 else SHA
        ),
        prepared_journal_entry_sha256=SHA,
        ddb_committed_journal_entry_sha256=SHA,
        owner_history=[_owner_history_entry()],
        send_stage="POSSIBLY_SENT" if sent else "NOT_SENT",
        prepared_at="2026-07-28T12:00:00Z",
        possibly_sent_at="2026-07-28T12:01:00Z" if sent else None,
        run_instances_attempt_evidence=(
            [_run_instances_attempt(attempt_outcome)] if sent else []
        ),
        direct_run_instances_request_id="ec2-request-1" if direct else None,
        direct_run_instances_response_sha256=SHA if direct else None,
        cloudtrail_lookup_evidence_sha256=SHA if sent else None,
        describe_instances_evidence_sha256=SHA if sent else None,
        observed_instance_ids=observed_instance_ids,
        instance_observation_sha256=SHA if has_closed_instance else None,
        spend_allocation_open_identity_sha256=(
            SHA if has_closed_instance else None
        ),
        instance_terminal_identity_sha256=(
            SHA if has_closed_instance else None
        ),
        spend_allocation_close_identity_sha256=(
            SHA if has_closed_instance else None
        ),
        incident_identity_sha256=(
            SHA if kind in {"MULTIPLE_INSTANCE_TOKEN", "UNRESOLVED_LAUNCH"}
            else None
        ),
        worker_launch_liability_identity_sha256=SHA if sent else None,
        abandoned_not_sent_proof_sha256=SHA if not sent else None,
        positive_service_rejection_evidence_sha256=SHA if rejection else None,
        no_call_in_flight_evidence_sha256=SHA if rejection else None,
    )


def _worker_launch_liability_entry(
    ordinal: int, state: str = "WATCHING"
) -> dict[str, object]:
    return _nested_entry(
        activation_id="x",
        activation_ordinal=1,
        generation=1,
        allocation_ordinal=ordinal,
        allocation_ordinal_text=f"{ordinal:08d}",
        worker_launch_identity_sha256=SHA,
        worker_launch_liability_identity_sha256=SHA,
        ec2_client_token="t" * 64,
        launch_parameters_sha256=SHA,
        expected_worker_tags_sha256=SHA,
        state=state,
        watch_schedule_arn="arn:schedule",
        watch_schedule_identity_sha256=SHA,
        watch_not_before="2026-07-28T12:01:00Z",
        watch_not_after="2026-08-27T12:01:00Z",
        observed_instance_ids=(
            [f"i-{ordinal}"] if state == "LATE_INSTANCE_DRAINING" else []
        ),
        last_scan_evidence_sha256=SHA,
        incident_identity_sha256=None,
        owner_attempt=1,
        owner_execution_arn="arn:execution",
        owner_state_machine_version_arn="arn:state-machine:1",
        owner_dispatch_identity_sha256=SHA,
        owner_invocation_nonce_sha256=SHA,
        owner_hard_expires_at="2026-07-28T12:30:00Z",
        revision=1,
    )


def _zero_match_entry() -> dict[str, object]:
    return _nested_entry(
        kind="ZERO_MATCH_SCAN",
        correlation_tuple_sha256=SHA,
        database_lower_bound="db-0001",
        database_upper_bound="db-0002",
        journal_lower_bound="journal-0001",
        journal_upper_bound="journal-0002",
        pagination_complete=True,
        match_count=0,
        scan_started_at="2026-07-28T12:00:00Z",
        scan_completed_at="2026-07-28T12:01:00Z",
        database_head_identity_sha256=SHA,
        journal_head_identity_sha256=SHA,
        scheduler_snapshot_identity_sha256=SHA,
        worker_snapshot_identity_sha256=SHA,
        allocation_snapshot_identity_sha256=SHA,
    )


def _exact_request_entry(request_id: str) -> dict[str, object]:
    return _nested_entry(
        kind="EXACT_REQUEST",
        request_id=request_id,
        correlation_tuple_sha256=SHA,
        stored_request_body_sha256=SHA,
        stored_arguments_sha256=SHA,
        request_state="CANCELLED",
        request_created_at="2026-07-28T11:50:00Z",
        request_updated_at="2026-07-28T11:59:00Z",
        scheduler_work_identity_sha256=SHA,
        terminal_observation_identity_sha256=SHA,
    )


def _multiple_match_entry(request_ids: list[str]) -> dict[str, object]:
    return _nested_entry(
        kind="MULTIPLE_MATCH",
        request_ids=request_ids,
        correlation_tuple_sha256=SHA,
        complete_member_set_sha256=canonical_sha256(request_ids),
        scan_identity_sha256=SHA,
    )


def _cancel_entry(kind: str, action_key: str) -> dict[str, object]:
    return _nested_entry(
        kind=kind,
        action_key=action_key,
        action_identity_sha256=SHA,
        response_identity_sha256=SHA,
        correlation_identity_sha256=SHA,
        terminal_observation_identity_sha256=SHA,
        requested_at="2026-07-28T12:00:00Z",
        terminal_observed_at="2026-07-28T12:01:00Z",
    )


def _quiescence_entry(index: int) -> dict[str, object]:
    started = "2026-07-28T12:02:00Z" if index == 1 else "2026-07-28T12:04:00Z"
    completed = "2026-07-28T12:03:00Z" if index == 1 else "2026-07-28T12:05:00Z"
    return _nested_entry(
        kind="QUIESCENCE_SNAPSHOT",
        scan_started_at=started,
        scan_completed_at=completed,
        request_scan_identity_sha256=SHA,
        job_scan_identity_sha256=SHA,
        worker_scan_identity_sha256=SHA,
        allocation_scan_identity_sha256=SHA,
        controller_scan_identity_sha256=SHA,
    )


def _request_evidence(cardinality: str) -> list[dict[str, object]]:
    if cardinality == "NOT_APPLICABLE":
        return []
    if cardinality == "ZERO":
        return [_zero_match_entry(), _quiescence_entry(1), _quiescence_entry(2)]
    if cardinality == "ONE":
        return [
            _exact_request_entry("request-1"),
            _quiescence_entry(1),
            _quiescence_entry(2),
        ]
    request_ids = ["request-1", "request-2"]
    return [
        _multiple_match_entry(request_ids),
        *[_exact_request_entry(request_id) for request_id in request_ids],
        _quiescence_entry(1),
        _quiescence_entry(2),
    ]


def _terminal_for_outcome(
    records: object,
    outcome: str,
    request_cardinality=None,
    optional_binding: bool = False,
) -> dict[str, object]:
    rule = records._TERMINAL_OUTCOME_RULES[outcome]
    marker = {
        "key": "marker.json",
        "version_id": "version-1",
        "body_sha256": SHA,
        "canonical_identity_sha256": "b" * 64,
    }
    candidate = _no_launch_terminal(outcome=outcome)
    candidate["handoff"] = None if rule["handoff"] == "null" else {"identity": SHA}
    candidate["binding"] = (
        (
            {"identity": SHA}
            if rule["binding"] == "optional" and optional_binding
            else None
        )
        if rule["binding"] in {"null", "optional"}
        else {"identity": SHA}
    )
    allocation_count = {
        "zero": 0, "one": 1, "nonempty": 1, "multiple": 2
    }[rule["allocations"]]
    allocations = [
        _allocation_entry(ordinal)
        for ordinal in range(1, allocation_count + 1)
    ]
    candidate["allocations"] = allocations
    candidate["allocations_array_sha256"] = canonical_sha256(allocations)
    candidate["worker_cardinality"] = (
        "ZERO" if allocation_count == 0 else (
            "ONE" if allocation_count == 1 else "MULTIPLE"
        )
    )
    launches = [
        _worker_launch_entry(ordinal)
        for ordinal in range(1, allocation_count + 1)
    ]
    liabilities = [
        _worker_launch_liability_entry(ordinal)
        for ordinal in range(1, allocation_count + 1)
    ]
    if outcome == "WORKER_LAUNCH_UNRESOLVED_NO_INSTANCE_INCIDENT":
        launches = [_worker_launch_entry(1, "UNRESOLVED_LAUNCH")]
        liabilities = [_worker_launch_liability_entry(1)]
    candidate["worker_launch_evidence"] = launches
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(launches)
    candidate["worker_launch_liabilities"] = liabilities
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        liabilities
    )
    cardinality = request_cardinality or next(
        value for value in ("NOT_APPLICABLE", "ZERO", "ONE", "MULTIPLE")
        if value in rule["request"]
    )
    candidate["request_cardinality"] = cardinality
    request_evidence = _request_evidence(cardinality)
    candidate["request_evidence"] = request_evidence
    candidate["request_evidence_array_sha256"] = canonical_sha256(request_evidence)
    candidate["operator_disposition_required"] = rule["disposition"]
    for field in records._TERMINAL_MARKERS:
        candidate[field] = None
    for field in rule["marker_required"]:
        candidate[field] = deepcopy(marker)
    candidate["prior_terminal_v1_identity"] = (
        deepcopy(marker) if rule["prior_terminal_v1"] == "nonnull" else None
    )
    candidate["post_terminal_quiescence_evidence"] = (
        {"identity": SHA}
        if outcome.endswith("_INCIDENT") or outcome.endswith("_NO_JOB")
        else None
    )
    return _rehash(candidate)


def test_every_terminal_v2_outcome_has_an_exhaustive_declarative_matrix() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    assert set(records._TERMINAL_OUTCOME_RULES) == set(records._OUTCOMES)
    for outcome, rule in records._TERMINAL_OUTCOME_RULES.items():
        assert (
            set(rule["marker_required"])
            | set(rule["marker_null"])
            | set(rule["marker_optional"])
        ) == set(records._TERMINAL_MARKERS)
        assert not (
            set(rule["marker_required"]) & set(rule["marker_null"])
            or set(rule["marker_required"]) & set(rule["marker_optional"])
            or set(rule["marker_null"]) & set(rule["marker_optional"])
        )
        candidate = _terminal_for_outcome(records, outcome)
        assert records.validate_record(
            "glm52_production_terminal_v2", candidate
        ) == candidate
        for cardinality in rule["request"]:
            request_candidate = _terminal_for_outcome(
                records, outcome, cardinality
            )
            assert records.validate_record(
                "glm52_production_terminal_v2", request_candidate
            ) == request_candidate
        if rule["binding"] == "optional":
            optional_binding_candidate = _terminal_for_outcome(
                records, outcome, optional_binding=True
            )
            assert records.validate_record(
                "glm52_production_terminal_v2", optional_binding_candidate
            ) == optional_binding_candidate
        for field in rule["marker_optional"]:
            optional_marker_candidate = deepcopy(candidate)
            optional_marker_candidate[field] = {
                "key": "marker.json", "version_id": "v",
                "body_sha256": SHA, "canonical_identity_sha256": "b" * 64,
            }
            _rehash(optional_marker_candidate)
            assert records.validate_record(
                "glm52_production_terminal_v2", optional_marker_candidate
            ) == optional_marker_candidate
        for field in ("handoff", "binding"):
            if rule[field] == "optional":
                continue
            mutant = deepcopy(candidate)
            mutant[field] = (
                {"identity": "b" * 64} if mutant[field] is None else None
            )
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(
                    "glm52_production_terminal_v2", mutant
                )
        disallowed_request = next(
            (
                value
                for value in ("NOT_APPLICABLE", "ZERO", "ONE", "MULTIPLE")
                if value not in rule["request"]
            ),
            None,
        )
        if disallowed_request is not None:
            mutant = deepcopy(candidate)
            mutant["request_cardinality"] = disallowed_request
            mutant["request_evidence"] = _request_evidence(disallowed_request)
            mutant["request_evidence_array_sha256"] = canonical_sha256(
                mutant["request_evidence"]
            )
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(
                    "glm52_production_terminal_v2", mutant
                )
        mutant = deepcopy(candidate)
        if rule["allocations"] == "zero":
            mutant["allocations"] = [
                {"allocation_ordinal": 1, "instance_id": "i-1"}
            ]
        else:
            mutant["allocations"] = []
        mutant["allocations_array_sha256"] = canonical_sha256(
            mutant["allocations"]
        )
        _rehash(mutant)
        with pytest.raises(records.RecordValidationError):
            records.validate_record("glm52_production_terminal_v2", mutant)
        mutant = deepcopy(candidate)
        mutant["prior_terminal_v1_identity"] = (
            None
            if candidate["prior_terminal_v1_identity"] is not None
            else {
                "key": "marker.json", "version_id": "v",
                "body_sha256": SHA, "canonical_identity_sha256": "b" * 64,
            }
        )
        _rehash(mutant)
        with pytest.raises(records.RecordValidationError):
            records.validate_record("glm52_production_terminal_v2", mutant)
        mutant = deepcopy(candidate)
        mutant["post_terminal_quiescence_evidence"] = (
            None
            if candidate["post_terminal_quiescence_evidence"] is not None
            else {"identity": SHA}
        )
        _rehash(mutant)
        with pytest.raises(records.RecordValidationError):
            records.validate_record("glm52_production_terminal_v2", mutant)
        mutant = deepcopy(candidate)
        mutant["operator_disposition_required"] = not rule["disposition"]
        _rehash(mutant)
        with pytest.raises(records.RecordValidationError):
            records.validate_record("glm52_production_terminal_v2", mutant)
        for field in rule["marker_required"]:
            mutant = deepcopy(candidate)
            mutant[field] = None
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(
                    "glm52_production_terminal_v2", mutant
                )
        for field in rule["marker_null"]:
            mutant = deepcopy(candidate)
            mutant[field] = {
                "key": "marker.json", "version_id": "v",
                "body_sha256": SHA, "canonical_identity_sha256": "b" * 64,
            }
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(
                    "glm52_production_terminal_v2", mutant
                )


def test_terminal_nested_field_sets_are_literal_and_frozen() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    assert records._ALLOCATION_ENTRY_FIELDS == (
        "allocation_ordinal", "allocation_ordinal_text", "instance_id",
        "instance_tags_sha256", "allocation_open_identity_sha256",
        "allocation_close_identity_sha256", "charged_interval_started_at",
        "charged_interval_ended_at", "terminal_state",
        "instance_terminal_identity_sha256",
        "concurrency_window_identity_sha256", "canonical_entry_sha256",
    )
    assert records._WORKER_LAUNCH_EVIDENCE_ENTRY_FIELDS == (
        "kind", "allocation_ordinal", "allocation_ordinal_text",
        "ec2_client_token", "launch_parameters_sha256",
        "expected_worker_tags_sha256", "worker_launch_identity_sha256",
        "prior_worker_launch_identity_sha256",
        "prior_instance_terminal_identity_sha256",
        "prior_spend_allocation_close_identity_sha256",
        "prepared_journal_entry_sha256",
        "ddb_committed_journal_entry_sha256", "owner_history", "send_stage",
        "prepared_at", "possibly_sent_at", "run_instances_attempt_evidence",
        "direct_run_instances_request_id",
        "direct_run_instances_response_sha256",
        "cloudtrail_lookup_evidence_sha256",
        "describe_instances_evidence_sha256", "observed_instance_ids",
        "instance_observation_sha256",
        "spend_allocation_open_identity_sha256",
        "instance_terminal_identity_sha256",
        "spend_allocation_close_identity_sha256", "incident_identity_sha256",
        "worker_launch_liability_identity_sha256",
        "abandoned_not_sent_proof_sha256",
        "positive_service_rejection_evidence_sha256",
        "no_call_in_flight_evidence_sha256", "canonical_entry_sha256",
    )
    assert records._OWNER_HISTORY_ENTRY_FIELDS == (
        "owner_attempt", "owner_principal_arn", "owner_function_version_arn",
        "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256",
        "owner_hard_expires_at", "canonical_entry_sha256",
    )
    assert records._RUN_INSTANCES_ATTEMPT_ENTRY_FIELDS == (
        "attempt", "attempted_at", "outcome", "request_id",
        "response_sha256", "transport_error_sha256",
        "canonical_entry_sha256",
    )
    assert records._WORKER_LAUNCH_LIABILITY_ENTRY_FIELDS == (
        "activation_id", "activation_ordinal", "generation",
        "allocation_ordinal", "allocation_ordinal_text",
        "worker_launch_identity_sha256",
        "worker_launch_liability_identity_sha256", "ec2_client_token",
        "launch_parameters_sha256", "expected_worker_tags_sha256", "state",
        "watch_schedule_arn", "watch_schedule_identity_sha256",
        "watch_not_before", "watch_not_after", "observed_instance_ids",
        "last_scan_evidence_sha256", "incident_identity_sha256",
        "owner_attempt", "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256", "owner_invocation_nonce_sha256",
        "owner_hard_expires_at", "revision", "canonical_entry_sha256",
    )
    assert records._REQUEST_EVIDENCE_ENTRY_FIELDS == {
        "ZERO_MATCH_SCAN": (
            "kind", "correlation_tuple_sha256", "database_lower_bound",
            "database_upper_bound", "journal_lower_bound",
            "journal_upper_bound", "pagination_complete", "match_count",
            "scan_started_at", "scan_completed_at",
            "database_head_identity_sha256",
            "journal_head_identity_sha256",
            "scheduler_snapshot_identity_sha256",
            "worker_snapshot_identity_sha256",
            "allocation_snapshot_identity_sha256",
            "canonical_entry_sha256",
        ),
        "EXACT_REQUEST": (
            "kind", "request_id", "correlation_tuple_sha256",
            "stored_request_body_sha256", "stored_arguments_sha256",
            "request_state", "request_created_at", "request_updated_at",
            "scheduler_work_identity_sha256",
            "terminal_observation_identity_sha256",
            "canonical_entry_sha256",
        ),
        "MULTIPLE_MATCH": (
            "kind", "request_ids", "correlation_tuple_sha256",
            "complete_member_set_sha256", "scan_identity_sha256",
            "canonical_entry_sha256",
        ),
        "REQUEST_CANCEL": (
            "kind", "action_key", "action_identity_sha256",
            "response_identity_sha256", "correlation_identity_sha256",
            "terminal_observation_identity_sha256", "requested_at",
            "terminal_observed_at", "canonical_entry_sha256",
        ),
        "JOB_CANCEL": (
            "kind", "action_key", "action_identity_sha256",
            "response_identity_sha256", "correlation_identity_sha256",
            "terminal_observation_identity_sha256", "requested_at",
            "terminal_observed_at", "canonical_entry_sha256",
        ),
        "QUIESCENCE_SNAPSHOT": (
            "kind", "scan_started_at", "scan_completed_at",
            "request_scan_identity_sha256", "job_scan_identity_sha256",
            "worker_scan_identity_sha256",
            "allocation_scan_identity_sha256",
            "controller_scan_identity_sha256", "canonical_entry_sha256",
        ),
    }


def test_terminal_allocation_entries_are_closed_hashed_and_time_ordered() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    assert records.validate_record(
        "glm52_production_terminal_v2", candidate
    ) == candidate
    for mutation in ("unknown", "missing", "type", "hash", "time"):
        mutant = deepcopy(candidate)
        entry = mutant["allocations"][0]
        if mutation == "unknown":
            entry["forged"] = True
        elif mutation == "missing":
            entry.pop("allocation_open_identity_sha256")
        elif mutation == "type":
            entry["allocation_ordinal"] = True
        elif mutation == "hash":
            entry["canonical_entry_sha256"] = "b" * 64
        else:
            entry["charged_interval_ended_at"] = (
                entry["charged_interval_started_at"]
            )
            entry["canonical_entry_sha256"] = canonical_sha256(
                {
                    field: value
                    for field, value in entry.items()
                    if field != "canonical_entry_sha256"
                }
            )
        mutant["allocations_array_sha256"] = canonical_sha256(
            mutant["allocations"]
        )
        _rehash(mutant)
        with pytest.raises(records.RecordValidationError):
            records.validate_record("glm52_production_terminal_v2", mutant)


def test_every_worker_launch_evidence_kind_is_closed_complete_and_hashed() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    cases = []

    prepared = _terminal_for_outcome(records, "CONSUMED_PROVED_NO_POST")
    prepared_launch = _worker_launch_entry(1, "PREPARED_PROVED_NOT_SENT")
    prepared["worker_launch_evidence"] = [prepared_launch]
    prepared["worker_launch_evidence_array_sha256"] = canonical_sha256(
        [prepared_launch]
    )
    _rehash(prepared)
    cases.append(prepared)

    rejected = _terminal_for_outcome(records, "KNOWN_REJECTED_NO_JOB")
    rejected_launch = _worker_launch_entry(
        1, "POSITIVE_SERVICE_REJECTION_NO_INSTANCE"
    )
    rejected_liability = _worker_launch_liability_entry(1)
    rejected["worker_launch_evidence"] = [rejected_launch]
    rejected["worker_launch_evidence_array_sha256"] = canonical_sha256(
        [rejected_launch]
    )
    rejected["worker_launch_liabilities"] = [rejected_liability]
    rejected["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        [rejected_liability]
    )
    _rehash(rejected)
    cases.append(rejected)

    cases.append(
        _terminal_for_outcome(records, "UNBOUND_LATE_WORKER_INCIDENT")
    )

    multiple = _terminal_for_outcome(
        records, "WORKER_LAUNCH_MULTIPLE_INSTANCE_TOKEN_INCIDENT"
    )
    observed = ["i-1", "i-2"]
    multiple_launches = [
        _worker_launch_entry(
            ordinal, "MULTIPLE_INSTANCE_TOKEN",
            observed_instance_ids=observed,
        )
        for ordinal in (1, 2)
    ]
    multiple["worker_launch_evidence"] = multiple_launches
    multiple["worker_launch_evidence_array_sha256"] = canonical_sha256(
        multiple_launches
    )
    _rehash(multiple)
    cases.append(multiple)

    cases.append(
        _terminal_for_outcome(
            records, "WORKER_LAUNCH_UNRESOLVED_NO_INSTANCE_INCIDENT"
        )
    )

    assert {
        candidate["worker_launch_evidence"][0]["kind"]
        for candidate in cases
    } == {
        "PREPARED_PROVED_NOT_SENT",
        "POSITIVE_SERVICE_REJECTION_NO_INSTANCE",
        "EXACT_SINGLE_INSTANCE", "MULTIPLE_INSTANCE_TOKEN",
        "UNRESOLVED_LAUNCH",
    }
    for candidate in cases:
        assert records.validate_record(
            "glm52_production_terminal_v2", candidate
        ) == candidate
        for mutation in ("unknown", "missing", "type", "hash"):
            mutant = deepcopy(candidate)
            entry = mutant["worker_launch_evidence"][0]
            if mutation == "unknown":
                entry["forged"] = True
            elif mutation == "missing":
                entry.pop("launch_parameters_sha256")
            elif mutation == "type":
                entry["allocation_ordinal"] = True
            else:
                entry["canonical_entry_sha256"] = "b" * 64
            mutant["worker_launch_evidence_array_sha256"] = canonical_sha256(
                mutant["worker_launch_evidence"]
            )
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(
                    "glm52_production_terminal_v2", mutant
                )


def test_terminal_liability_entries_are_closed_complete_hashed_and_bijective() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    for state in ("WATCHING", "LATE_INSTANCE_DRAINING"):
        candidate = _terminal_for_outcome(
            records, "UNBOUND_LATE_WORKER_INCIDENT"
        )
        liability = _worker_launch_liability_entry(1, state)
        candidate["worker_launch_liabilities"] = [liability]
        candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
            [liability]
        )
        _rehash(candidate)
        assert records.validate_record(
            "glm52_production_terminal_v2", candidate
        ) == candidate
        for mutation in ("unknown", "missing", "type", "hash", "state"):
            mutant = deepcopy(candidate)
            entry = mutant["worker_launch_liabilities"][0]
            if mutation == "unknown":
                entry["forged"] = True
            elif mutation == "missing":
                entry.pop("worker_launch_identity_sha256")
            elif mutation == "type":
                entry["allocation_ordinal"] = True
            elif mutation == "hash":
                entry["canonical_entry_sha256"] = "b" * 64
            else:
                entry["state"] = "UNRELATED"
                entry["canonical_entry_sha256"] = canonical_sha256(
                    {
                        field: value
                        for field, value in entry.items()
                        if field != "canonical_entry_sha256"
                    }
                )
            mutant["worker_launch_liabilities_array_sha256"] = canonical_sha256(
                mutant["worker_launch_liabilities"]
            )
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(
                    "glm52_production_terminal_v2", mutant
                )


def test_every_request_evidence_kind_is_closed_complete_hashed_and_ordered() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidates = [
        _terminal_for_outcome(records, "KNOWN_REJECTED_NO_JOB"),
        _terminal_for_outcome(records, "ACCEPTED_REQUEST_FAILED_NO_JOB"),
        _terminal_for_outcome(
            records, "QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT",
            "MULTIPLE",
        ),
    ]
    cancel_candidate = _terminal_for_outcome(
        records, "ACCEPTED_REQUEST_CANCELLED_NO_JOB"
    )
    cancel_evidence = [
        _exact_request_entry("request-1"),
        _cancel_entry("REQUEST_CANCEL", "a-request-cancel"),
        _cancel_entry("JOB_CANCEL", "b-job-cancel"),
        _quiescence_entry(1), _quiescence_entry(2),
    ]
    cancel_candidate["request_evidence"] = cancel_evidence
    cancel_candidate["request_evidence_array_sha256"] = canonical_sha256(
        cancel_evidence
    )
    _rehash(cancel_candidate)
    candidates.append(cancel_candidate)

    assert {
        entry["kind"]
        for candidate in candidates
        for entry in candidate["request_evidence"]
    } == set(records._REQUEST_EVIDENCE_ENTRY_FIELDS)
    for candidate in candidates:
        assert records.validate_record(
            "glm52_production_terminal_v2", candidate
        ) == candidate
        for index, entry in enumerate(candidate["request_evidence"]):
            for mutation in ("unknown", "missing", "type", "hash"):
                mutant = deepcopy(candidate)
                changed = mutant["request_evidence"][index]
                if mutation == "unknown":
                    changed["forged"] = True
                elif mutation == "missing":
                    removable = next(
                        field for field in changed
                        if field not in {"kind", "canonical_entry_sha256"}
                    )
                    changed.pop(removable)
                elif mutation == "type":
                    changed["kind"] = 1
                else:
                    changed["canonical_entry_sha256"] = "b" * 64
                mutant["request_evidence_array_sha256"] = canonical_sha256(
                    mutant["request_evidence"]
                )
                _rehash(mutant)
                with pytest.raises(records.RecordValidationError):
                    records.validate_record(
                        "glm52_production_terminal_v2", mutant
                    )


def test_terminal_nested_timestamps_and_quiescence_are_strictly_ordered() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "ACCEPTED_REQUEST_FAILED_NO_JOB"
    )
    second = candidate["request_evidence"][-1]
    second["scan_started_at"] = "2026-07-28T12:03:00Z"
    second["canonical_entry_sha256"] = canonical_sha256(
        {
            field: value for field, value in second.items()
            if field != "canonical_entry_sha256"
        }
    )
    candidate["request_evidence_array_sha256"] = canonical_sha256(
        candidate["request_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="chronologically separated"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_every_nested_primitive_rejects_type_confusion_after_all_hashes_recomputed() -> None:
    records = importlib.import_module("glm52_enforcement.records")

    def confused(value: object) -> object:
        if type(value) is int:
            return True
        if type(value) is bool:
            return 1
        if type(value) is list:
            return {}
        return 7

    def assert_rejected(
        candidate: dict[str, object],
        array_field: str,
        entry_index: int,
        field: str,
        nested_field=None,
    ) -> None:
        mutant = deepcopy(candidate)
        entry = mutant[array_field][entry_index]
        if nested_field is None:
            entry[field] = confused(entry[field])
        else:
            nested = entry[field][0]
            nested[nested_field] = confused(nested[nested_field])
            _rehash_entry(nested)
        _rehash_entry(entry)
        mutant[array_field + "_array_sha256"] = canonical_sha256(
            mutant[array_field]
        )
        _rehash(mutant)
        with pytest.raises(records.RecordValidationError):
            records.validate_record("glm52_production_terminal_v2", mutant)

    allocated = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    for field in records._ALLOCATION_ENTRY_FIELDS:
        if field != "canonical_entry_sha256":
            assert_rejected(allocated, "allocations", 0, field)

    for field in records._WORKER_LAUNCH_EVIDENCE_ENTRY_FIELDS:
        if field != "canonical_entry_sha256":
            assert_rejected(
                allocated, "worker_launch_evidence", 0, field
            )
    for field in records._OWNER_HISTORY_ENTRY_FIELDS:
        if field != "canonical_entry_sha256":
            assert_rejected(
                allocated,
                "worker_launch_evidence",
                0,
                "owner_history",
                field,
            )
    for field in records._RUN_INSTANCES_ATTEMPT_ENTRY_FIELDS:
        if field != "canonical_entry_sha256":
            assert_rejected(
                allocated,
                "worker_launch_evidence",
                0,
                "run_instances_attempt_evidence",
                field,
            )

    for field in records._WORKER_LAUNCH_LIABILITY_ENTRY_FIELDS:
        if field != "canonical_entry_sha256":
            assert_rejected(
                allocated, "worker_launch_liabilities", 0, field
            )

    request_candidates = {
        "ZERO_MATCH_SCAN": _terminal_for_outcome(
            records, "KNOWN_REJECTED_NO_JOB"
        ),
        "EXACT_REQUEST": _terminal_for_outcome(
            records, "ACCEPTED_REQUEST_FAILED_NO_JOB"
        ),
        "MULTIPLE_MATCH": _terminal_for_outcome(
            records,
            "QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT",
            "MULTIPLE",
        ),
    }
    cancel_candidate = _terminal_for_outcome(
        records, "ACCEPTED_REQUEST_CANCELLED_NO_JOB"
    )
    cancel_evidence = [
        _exact_request_entry("request-1"),
        _cancel_entry("REQUEST_CANCEL", "a-request-cancel"),
        _cancel_entry("JOB_CANCEL", "b-job-cancel"),
        _quiescence_entry(1), _quiescence_entry(2),
    ]
    cancel_candidate["request_evidence"] = cancel_evidence
    cancel_candidate["request_evidence_array_sha256"] = canonical_sha256(
        cancel_evidence
    )
    _rehash(cancel_candidate)
    request_candidates["REQUEST_CANCEL"] = cancel_candidate
    request_candidates["JOB_CANCEL"] = cancel_candidate
    request_candidates["QUIESCENCE_SNAPSHOT"] = cancel_candidate
    for kind, fields in records._REQUEST_EVIDENCE_ENTRY_FIELDS.items():
        candidate = request_candidates[kind]
        index = next(
            index
            for index, entry in enumerate(candidate["request_evidence"])
            if entry["kind"] == kind
        )
        for field in fields:
            if field != "canonical_entry_sha256":
                assert_rejected(
                    candidate, "request_evidence", index, field
                )


def test_all_ordinals_are_positive_and_ledger_segments_are_closed() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    ordinal_fields = {
        "current_activation_ordinal", "activation_ordinal", "epoch",
        "generation", "allocation_ordinal", "attempt", "owner_attempt",
    }
    for record_type, fields in records.RECORD_FIELDS.items():
        candidate = _closed_record(record_type)
        for field in ordinal_fields & set(fields):
            if candidate[field] is None:
                continue
            mutant = deepcopy(candidate)
            mutant[field] = 0
            text_field = field + "_text"
            if text_field in mutant:
                mutant[text_field] = "00000000"
            _rehash(mutant)
            with pytest.raises(records.RecordValidationError):
                records.validate_record(record_type, mutant)
    with pytest.raises(records.RecordValidationError):
        records.ledger_sk(
            "glm52_production_execution",
            activation_id="activation#foreign",
            epoch=1,
        )
    with pytest.raises(records.RecordValidationError):
        records.ledger_sk(
            "glm52_production_action",
            activation_id="activation-1",
            generation=1,
            action_kind="SKY_POST#FOREIGN",
            attempt=1,
        )
    with pytest.raises(records.RecordValidationError):
        records.ledger_sk(
            "glm52_production_post_terminal_allocation",
            activation_id="activation-1",
            allocation_ordinal=1,
            instance_id="i-1#FOREIGN",
        )


def test_timestamp_validation_rejects_impossible_calendar_dates() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _activation_index(updated_at="2026-02-30T12:00:00Z")
    with pytest.raises(records.RecordValidationError, match="UTC"):
        records.validate_record("glm52_production_activation_index", candidate)


def test_execution_direct_start_evidence_is_all_or_none() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _state_fixture(
        records, "glm52_production_execution", "START_POSSIBLY_SENT"
    )
    candidate["direct_start_response_identity"] = {"identity": SHA}
    candidate["direct_start_request_id"] = None
    with pytest.raises(
        records.RecordValidationError, match="direct start evidence"
    ):
        records.validate_record("glm52_production_execution", candidate)


def test_worker_launch_direct_run_instances_evidence_is_all_or_none() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _state_fixture(
        records, "glm52_production_worker_launch", "POSSIBLY_SENT"
    )
    candidate["direct_run_instances_request_id"] = "request-1"
    candidate["direct_run_instances_response_sha256"] = None
    with pytest.raises(
        records.RecordValidationError, match="direct RunInstances evidence"
    ):
        records.validate_record("glm52_production_worker_launch", candidate)


def test_action_classification_evidence_union_is_closed() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    direct = _state_fixture(
        records, "glm52_production_action", "POST_CLASSIFIED"
    )
    direct.update(
        outcome_class="ACCEPTED",
        classification_evidence_kind="RELAY_RESPONSE",
        classification_evidence_body_sha256=SHA,
        response_identity_sha256=None,
    )
    with pytest.raises(
        records.RecordValidationError, match="classification evidence"
    ):
        records.validate_record("glm52_production_action", direct)

    owner_death = _state_fixture(
        records, "glm52_production_action", "POST_CLASSIFIED"
    )
    owner_death.update(
        outcome_class="AMBIGUOUS_OWNER_DIED",
        classification_evidence_kind="OWNER_DEATH_PROOF",
        classification_evidence_body_sha256=SHA,
        response_identity_sha256=SHA,
        sky_request_id=None,
    )
    with pytest.raises(
        records.RecordValidationError, match="classification evidence"
    ):
        records.validate_record("glm52_production_action", owner_death)


@pytest.mark.parametrize(
    ("source", "outcome"),
    (
        ("CONSUMED", "PROVED_NOT_SENT_OWNER_DIED"),
        ("POST_STARTED", "PROVED_NOT_SENT_OWNER_DIED"),
        ("POST_AUTHORIZED", "AMBIGUOUS_OWNER_DIED"),
    ),
)
def test_owner_death_sky_action_edges_have_closed_evidence_matrices(
    source: str, outcome: str
) -> None:
    records = importlib.import_module("glm52_enforcement.records")
    transitions = importlib.import_module("glm52_enforcement.transitions")
    before, after = _transition_fixture(
        records, "glm52_production_action", source, "POST_CLASSIFIED"
    )
    after.update(
        outcome_class=outcome,
        classification_evidence_kind="OWNER_DEATH_PROOF",
        classification_evidence_body_sha256=SHA,
        response_identity_sha256=None,
        sky_request_id=None,
    )
    after = _rehash(after)
    assert transitions.validate_transition(
        "glm52_production_action", before, after
    ) == after

    if source == "CONSUMED":
        assert after["post_owner_invocation_nonce_sha256"] is None
        assert after["post_started_at"] is None
        assert after["post_authorized_at"] is None
    elif source == "POST_STARTED":
        assert after["post_owner_invocation_nonce_sha256"] is not None
        assert after["post_started_at"] is not None
        assert after["post_authorized_at"] is None
    else:
        assert after["post_authorized_at"] is not None
        assert after["relay_envelope_sha256"] is not None


def test_recovery_handoff_is_a_predeclared_recovery_action_kind() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _state_fixture(
        records, "glm52_production_recovery_action", "ARMED"
    )
    candidate["action_kind"] = "RECOVERY_HANDOFF"
    candidate = _rehash(candidate)
    assert records.validate_record(
        "glm52_production_recovery_action", candidate
    ) == candidate


def test_ledger_sk_rejects_generation_zero_for_unlisted_action_kinds() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    for action_kind in ("BOGUS", "SKY_POST"):
        with pytest.raises(
            records.RecordValidationError, match="generation"
        ):
            records.ledger_sk(
                "glm52_production_action",
                activation_id="activation-1",
                generation=0,
                action_kind=action_kind,
                attempt=1,
            )


def test_terminal_v2_rejects_minimal_allocation_entry() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    allocation = {"allocation_ordinal": 1, "instance_id": "i-1"}
    candidate["allocations"] = [allocation]
    candidate["allocations_array_sha256"] = canonical_sha256([allocation])
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="allocation entry schema"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_v2_rejects_forged_liability_entry() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = {
        "allocation_ordinal": 1, "state": "UNRELATED", "forged": True
    }
    candidate["worker_launch_liabilities"] = [liability]
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        [liability]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="liability entry schema"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_run_instances_attempt_request_id_requires_nonempty_ascii_string() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    attempt = launch["run_instances_attempt_evidence"][0]
    attempt["request_id"] = 7
    _rehash_entry(attempt)
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="request ID"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_direct_run_instances_request_id_requires_nonempty_ascii_string() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["direct_run_instances_request_id"] = 7
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="direct request ID"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_client_token_rejects_non_ascii_bytes() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["ec2_client_token"] = ("t" * 63) + "é"
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="ClientToken"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_identity_is_byte_bound_to_worker_launch() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["ec2_client_token"] = "u" * 64
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="liability binding"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_owner_history_timestamps_are_chronologically_nondecreasing() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["owner_history"] = [
        _owner_history_entry(1, "2026-07-28T12:30:00Z"),
        _owner_history_entry(2, "2026-07-28T12:20:00Z"),
    ]
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="owner history timestamps"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_run_instances_attempt_timestamps_are_chronologically_nondecreasing() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["run_instances_attempt_evidence"] = [
        _run_instances_attempt(
            attempt=1, attempted_at="2026-07-28T12:02:00Z"
        ),
        _run_instances_attempt(
            attempt=2, attempted_at="2026-07-28T12:01:00Z"
        ),
    ]
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="attempt timestamps"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_activation_id_binds_outer_terminal() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["activation_id"] = "different-activation"
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="liability binding"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_activation_ordinal_binds_outer_terminal() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["activation_ordinal"] = 2
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="liability binding"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_generation_binds_outer_terminal() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["generation"] = 2
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="liability binding"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_direct_request_id_binds_unique_direct_attempt() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["direct_run_instances_request_id"] = "different-request"
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="direct request correlation"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_direct_request_history_rejects_contradictory_attempts() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["run_instances_attempt_evidence"].append(
        _run_instances_attempt(
            "POSITIVE_SERVICE_REJECTION",
            attempt=2,
            attempted_at="2026-07-28T12:02:00Z",
        )
    )
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="direct request correlation"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_run_instances_attempt_outcome_rejects_list_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    attempt = launch["run_instances_attempt_evidence"][0]
    attempt["outcome"] = []
    _rehash_entry(attempt)
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_run_instances_attempt_outcome_rejects_dict_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    attempt = launch["run_instances_attempt_evidence"][0]
    attempt["outcome"] = {}
    _rehash_entry(attempt)
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_launch_kind_rejects_list_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["kind"] = []
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_launch_kind_rejects_dict_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["kind"] = {}
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_state_rejects_list_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["state"] = []
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_state_rejects_dict_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["state"] = {}
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_request_evidence_kind_rejects_list_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT", request_cardinality="ONE"
    )
    request = candidate["request_evidence"][0]
    request["kind"] = []
    _rehash_entry(request)
    candidate["request_evidence_array_sha256"] = canonical_sha256(
        candidate["request_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_request_evidence_kind_rejects_dict_type_confusion() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT", request_cardinality="ONE"
    )
    request = candidate["request_evidence"][0]
    request["kind"] = {}
    _rehash_entry(request)
    candidate["request_evidence_array_sha256"] = canonical_sha256(
        candidate["request_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(records.RecordValidationError):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_terminal_liability_worker_identity_binds_same_ordinal_launch() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    liability = candidate["worker_launch_liabilities"][0]
    liability["worker_launch_identity_sha256"] = "b" * 64
    _rehash_entry(liability)
    candidate["worker_launch_liabilities_array_sha256"] = canonical_sha256(
        candidate["worker_launch_liabilities"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="liability binding"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)


def test_worker_direct_response_sha_binds_unique_direct_attempt() -> None:
    records = importlib.import_module("glm52_enforcement.records")
    candidate = _terminal_for_outcome(
        records, "UNBOUND_LATE_WORKER_INCIDENT"
    )
    launch = candidate["worker_launch_evidence"][0]
    launch["direct_run_instances_response_sha256"] = "b" * 64
    _rehash_entry(launch)
    candidate["worker_launch_evidence_array_sha256"] = canonical_sha256(
        candidate["worker_launch_evidence"]
    )
    _rehash(candidate)
    with pytest.raises(
        records.RecordValidationError, match="direct request correlation"
    ):
        records.validate_record("glm52_production_terminal_v2", candidate)
