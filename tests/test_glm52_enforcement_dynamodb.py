from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from glm52_enforcement import records as record_contract
from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import (
    DynamoLedgerAdapter,
    LedgerKey,
    ExactCheck,
    ExactPut,
    ExactUpdate,
    WriteOutcome,
    decode_attribute_value,
    decode_item,
    encode_attribute_value,
    encode_item,
)


SHA = "a" * 64
TS = "2026-07-28T12:00:00Z"
RUN_ID = "glm52-sky-20260724"


class SimulatedClientError(Exception):
    def __init__(
        self,
        code: str,
        *,
        request_id: str = "write-error-request",
        cancellation_reasons: tuple[dict[str, object], ...] = (),
    ) -> None:
        super().__init__("simulated DynamoDB client error")
        self.response = {
            "Error": {"Code": code, "Message": "simulated"},
            "ResponseMetadata": {
                "HTTPStatusCode": 400,
                "RequestId": request_id,
            },
            "CancellationReasons": list(deepcopy(cancellation_reasons)),
        }


class AwsShapedSimulator:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.query_pages: list[dict[str, object]] = []
        self.transact_get_fault: BaseException | None = None
        self.before_read_hook: object = None
        self.pre_write_fault: BaseException | None = None
        self.post_commit_fault: BaseException | None = None
        self.malformed_write_response = False

    def install(
        self, sort_key: str, record: dict[str, object], *, run_id: str = RUN_ID
    ) -> None:
        physical = {"PK": "RUN#" + run_id, "SK": sort_key, **deepcopy(record)}
        self.items[("RUN#" + run_id, sort_key)] = encode_item(physical)

    def get_item(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_item", deepcopy(kwargs)))
        key = decode_item(kwargs["Key"])
        response: dict[str, object] = {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-request",
            }
        }
        item = self.items.get((key["PK"], key["SK"]))
        if item is not None:
            response["Item"] = deepcopy(item)
        return response

    def transact_get_items(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("transact_get_items", deepcopy(kwargs)))
        if callable(self.before_read_hook):
            self.before_read_hook()
        if self.transact_get_fault is not None:
            raise self.transact_get_fault
        responses: list[dict[str, object]] = []
        for request in kwargs["TransactItems"]:
            key = decode_item(request["Get"]["Key"])
            item = self.items.get((key["PK"], key["SK"]))
            responses.append({} if item is None else {"Item": deepcopy(item)})
        return {
            "Responses": responses,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "read-request",
            },
        }

    def query(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("query", deepcopy(kwargs)))
        if not self.query_pages:
            raise AssertionError("unexpected query")
        return deepcopy(self.query_pages.pop(0))

    def transact_write_items(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("transact_write_items", deepcopy(kwargs)))
        if self.pre_write_fault is not None:
            raise self.pre_write_fault
        candidate = deepcopy(self.items)
        try:
            for entry in kwargs["TransactItems"]:
                if set(entry) == {"Put"}:
                    request = entry["Put"]
                    item = decode_item(request["Item"])
                    key = (item["PK"], item["SK"])
                    self._require_condition(candidate.get(key), request)
                    candidate[key] = deepcopy(request["Item"])
                elif set(entry) == {"Update"}:
                    request = entry["Update"]
                    decoded_key = decode_item(request["Key"])
                    key = (decoded_key["PK"], decoded_key["SK"])
                    current = candidate.get(key)
                    self._require_condition(current, request)
                    physical = (
                        {"PK": decoded_key["PK"], "SK": decoded_key["SK"]}
                        if current is None
                        else decode_item(current)
                    )
                    names = request["ExpressionAttributeNames"]
                    values = request["ExpressionAttributeValues"]
                    for assignment in request["UpdateExpression"][4:].split(", "):
                        name, placeholder = assignment.split(" = ")
                        physical[names[name]] = decode_attribute_value(
                            values[placeholder]
                        )
                    candidate[key] = encode_item(physical)
                elif set(entry) == {"ConditionCheck"}:
                    request = entry["ConditionCheck"]
                    decoded_key = decode_item(request["Key"])
                    key = (decoded_key["PK"], decoded_key["SK"])
                    self._require_condition(candidate.get(key), request)
                else:
                    raise AssertionError("unknown transaction entry")
        except SimulatedClientError:
            raise
        self.items = candidate
        if self.post_commit_fault is not None:
            raise self.post_commit_fault
        if self.malformed_write_response:
            return {"malformed": True}
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "write-request",
            }
        }

    @staticmethod
    def _require_condition(
        current: dict[str, object] | None, request: dict[str, object]
    ) -> None:
        expression = request["ConditionExpression"]
        names = request["ExpressionAttributeNames"]
        values = request.get("ExpressionAttributeValues", {})
        decoded = None if current is None else decode_item(current)
        for clause in expression.split(" AND "):
            if clause.startswith("attribute_not_exists("):
                name = clause[len("attribute_not_exists(") : -1]
                if decoded is not None and names[name] in decoded:
                    raise SimulatedClientError(
                        "TransactionCanceledException",
                        cancellation_reasons=({"Code": "ConditionalCheckFailed"},),
                    )
            else:
                name, placeholder = clause.split(" = ")
                if decoded is None or decoded.get(names[name]) != (
                    decode_attribute_value(values[placeholder])
                ):
                    raise SimulatedClientError(
                        "TransactionCanceledException",
                        cancellation_reasons=({"Code": "ConditionalCheckFailed"},),
                    )


def _activation_index(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_activation_index",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": RUN_ID,
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


def _control(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_control",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": RUN_ID,
        "campaign_identity_sha256": SHA,
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "rollover_identity_sha256": SHA,
        "active_epoch": 1,
        "active_execution_arn": "execution-arn",
        "active_state_machine_version_arn": "state-machine-version-arn",
        "phase": "OPEN",
        "fence_head_body_sha256": SHA,
        "fence_head_version_id": "fence-version",
        "barrier_nonce_sha256": SHA,
        "barrier_state": "OPEN",
        "decision_seal_state": "OPEN",
        "recovery_seal_state": "OPEN",
        "teardown_seal_state": "OPEN",
        "last_sky_post_generation": 1,
        "last_sky_post_action_key": "action-key",
        "last_sky_post_state": "NONE",
        "numeric_job_binding_state": "UNBOUND",
        "recovery_control_initial_body_sha256": SHA,
        "finalization_control_initial_body_sha256": SHA,
        "snapshot_cleanup_control_initial_body_sha256": SHA,
        "revision": 1,
        "updated_at": TS,
    }
    value.update(overrides)
    return value


def _closed_record(record_type: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {}
    for field in record_contract.RECORD_FIELDS[record_type]:
        if field == "schema_version":
            value[field] = 2 if record_type.endswith("terminal_v2") else 1
        elif field == "record_type":
            value[field] = record_type
        elif field == "account_id":
            value[field] = "246813579024"
        elif field == "region":
            value[field] = "us-west-2"
        elif field == "run_id":
            value[field] = RUN_ID
        elif field in record_contract._ARRAY_FIELDS:
            value[field] = []
        elif field in record_contract._OBJECT_FIELDS:
            value[field] = {"identity": SHA}
        elif field in record_contract._INT_FIELDS:
            value[field] = 1
        elif field == "operator_disposition_required":
            value[field] = False
        elif field.endswith("_array_sha256"):
            value[field] = canonical_sha256(
                value.get(field.removesuffix("_array_sha256"), [])
            )
        elif field.endswith("_sha256"):
            value[field] = SHA
        elif field in record_contract._TIMESTAMP_FIELDS:
            value[field] = TS
        elif field in {"generation_text", "allocation_ordinal_text", "epoch_text"}:
            value[field] = "00000001"
        elif field == "ec2_client_token":
            value[field] = "t" * 64
        else:
            value[field] = "x"
    retained_defaults = {
        "glm52_production_recovery_action": ("RECOVERY", "REQUEST_CANCEL"),
        "glm52_production_finalization_action": (
            "FINALIZATION",
            "SUPPORT_PLANE_FINALIZED",
        ),
        "glm52_production_snapshot_cleanup_action": (
            "SNAPSHOT_CLEANUP",
            "SNAPSHOT_DELETE",
        ),
        "glm52_production_worker_launch_liability_action": (
            "WORKER_LAUNCH_LIABILITY",
            "SAME_TOKEN_COMPLETE",
        ),
    }
    if record_type in retained_defaults:
        value["authority_domain"], value["action_kind"] = retained_defaults[
            record_type
        ]
        if record_type != "glm52_production_worker_launch_liability_action":
            for field in (
                "allocation_ordinal",
                "allocation_ordinal_text",
                "worker_launch_identity_sha256",
                "worker_launch_liability_identity_sha256",
            ):
                value[field] = None
    default_states = {
        "glm52_production_recovery_control": "DORMANT",
        "glm52_production_finalization_control": "DORMANT",
        "glm52_production_snapshot_cleanup_control": "DORMANT",
        "glm52_production_recovery_action": "ARMED",
        "glm52_production_finalization_action": "ARMED",
        "glm52_production_snapshot_cleanup_action": "ARMED",
        "glm52_production_worker_launch_liability_action": "ARMED",
        "glm52_production_execution": "START_OWNED",
        "glm52_production_worker_launch": "PREPARED_NOT_SENT",
        "glm52_production_worker_launch_liability": "UNOWNED_NOT_ACTIONABLE",
        "glm52_production_post_terminal_allocation": "DISCOVERED",
        "glm52_production_action": "ARMED",
    }
    if record_type in default_states:
        value["state"] = default_states[record_type]
    if record_type == "glm52_production_control":
        value["phase"] = "OPEN"
    if record_type == "glm52_production_worker_launch_liability_settlement":
        value["settlement_kind"] = "NO_INSTANCE_POSITIVE_REJECTION"
        value["ebs_liability_reserve_cost_usd"] = "0.01"
    if record_type in {
        "glm52_production_worker_launch",
        "glm52_production_worker_launch_liability",
    }:
        value.update(
            gpu_liability_reserve_seconds=900,
            gpu_liability_reserve_cost_usd="13.76",
            ebs_liability_reserve_cost_usd="0.01",
        )
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
            "prior_activation_id",
            "prior_activation_ordinal",
            "prior_terminal_v2_key",
            "prior_terminal_v2_version_id",
            "prior_terminal_v2_body_sha256",
            "prior_terminal_v2_outcome",
            "prior_final_worker_cardinality",
            "prior_h1g_drained_key",
            "prior_h1g_drained_version_id",
            "prior_h1g_drained_body_sha256",
            "prior_spend_ledger_head_identity",
            "prior_snapshot_cleanup_control_identity",
            "operator_disposition_identity",
        ):
            value[field] = None
        for field in (
            "prior_allocations_array_sha256",
            "prior_request_evidence_array_sha256",
            "prior_worker_launch_evidence_array_sha256",
            "prior_worker_launch_liabilities_array_sha256",
            "prior_worker_launch_liability_settlements_array_sha256",
            "prior_post_terminal_allocations_array_sha256",
            "prior_merged_final_allocations_array_sha256",
        ):
            value[field] = canonical_sha256([])
    value.update(overrides)
    if record_type == "glm52_production_action" and value["state"] == "ARMED":
        for field in record_contract._ACTION_ARMED_FUTURE:
            if field not in overrides:
                value[field] = None
    if record_type in retained_defaults and value["state"] == "ARMED":
        for field in record_contract._RETAINED_ACTION_FUTURE:
            if field not in overrides:
                value[field] = None
    if (
        record_type == "glm52_production_execution"
        and value["state"] == "START_OWNED"
    ):
        value["start_send_stage"] = "NOT_SENT"
        for field in record_contract._EXECUTION_AFTER_START:
            if field not in overrides:
                value[field] = None
    if (
        record_type == "glm52_production_worker_launch"
        and value["state"] == "PREPARED_NOT_SENT"
    ):
        value["send_stage"] = "NOT_SENT"
        value["same_token_completion_count"] = 0
        for field in record_contract._WORKER_AFTER_PREPARED:
            if field not in overrides:
                value[field] = None
    if record_type in record_contract._STATE_RULES:
        rule = record_contract._STATE_RULES[record_type][value["state"]]
        owner_fields = (
            record_contract._WORKER_OWNER
            if record_type == "glm52_production_worker_launch"
            else record_contract._STANDARD_OWNER
        )
        if rule["owner"] == "null":
            for field in owner_fields:
                if field not in overrides:
                    value[field] = None
            if "owner_attempt" not in overrides:
                value["owner_attempt"] = None
        for field in rule["null"]:
            if field not in overrides:
                value[field] = None
        for field in rule["nonnull"]:
            if field not in overrides and value[field] is None:
                value[field] = "b" * 64 if field.endswith("_sha256") else "evidence"
        for field in rule["nonempty"]:
            if field not in overrides and not value[field]:
                value[field] = ["i-1"]
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


def _rollover_arguments(
    raw_nonce: bytes, operation_identity: str = "b" * 64
) -> dict[str, object]:
    activation_id = "activation-1"
    return {
        "index": ExactUpdate(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
            {},
            _activation_index(
                snapshot_cleanup_lineage_sha256=canonical_sha256([])
            ),
        ),
        "rollover": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#ROLLOVER"),
            _closed_record(
                "glm52_production_rollover",
                activation_id=activation_id,
                writer_invocation_nonce_sha256=hashlib.sha256(
                    raw_nonce
                ).hexdigest(),
                cloudformation_operation_identity_sha256=operation_identity,
            ),
        ),
        "control": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#CONTROL"),
            _closed_record(
                "glm52_production_control", activation_id=activation_id
            ),
        ),
        "recovery_control": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#RECOVERY_CONTROL"),
            _closed_record(
                "glm52_production_recovery_control",
                activation_id=activation_id,
            ),
        ),
        "finalization_control": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-1#FINALIZATION_CONTROL"),
            _closed_record(
                "glm52_production_finalization_control",
                activation_id=activation_id,
            ),
        ),
        "snapshot_cleanup_control": ExactPut(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL",
            ),
            _closed_record(
                "glm52_production_snapshot_cleanup_control",
                activation_id=activation_id,
            ),
        ),
        "cleanup_heads": (),
        "domain": "ACTIVATION_ROLLOVER",
        "operation_identity_sha256": operation_identity,
        "raw_owner_nonce": raw_nonce,
    }


def _later_rollover_arguments(
    raw_nonce: bytes, operation_identity: str = "b" * 64
) -> dict[str, object]:
    activation_id = "activation-2"
    lineage = [
        {
            "activation_ordinal": 1,
            "cleanup_control_root_identity_sha256": "1" * 64,
            "cleanup_transition_chain_head_sha256": "2" * 64,
            "cleanup_transition_chain_length": 1,
            "transition_identities": ["2" * 64],
        }
    ]
    terminal_body_sha256 = "3" * 64
    drained_body_sha256 = "4" * 64
    terminal_identity = {
        "key": "terminal-v2-key",
        "version_id": "terminal-v2-version",
        "body_sha256": terminal_body_sha256,
    }
    drained_identity = {
        "key": "h1g-drained-key",
        "version_id": "h1g-drained-version",
        "body_sha256": drained_body_sha256,
    }
    spend_identity = {"identity": "5" * 64}
    index_before = _activation_index(
        snapshot_cleanup_lineage_sha256=canonical_sha256([]),
    )
    index_after = _activation_index(
        current_activation_id=activation_id,
        current_activation_ordinal=2,
        prior_activation_id="activation-1",
        prior_activation_terminal_v2_identity=terminal_identity,
        prior_h1g_drained_identity=drained_identity,
        prior_spend_ledger_head_identity=spend_identity,
        snapshot_cleanup_lineage_sha256=canonical_sha256(lineage),
        revision=2,
    )
    return {
        "index": ExactUpdate(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
            index_before,
            index_after,
        ),
        "rollover": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-2#ROLLOVER"),
            _closed_record(
                "glm52_production_rollover",
                activation_id=activation_id,
                activation_ordinal=2,
                prior_activation_id="activation-1",
                prior_activation_ordinal=1,
                prior_terminal_v2_key="terminal-v2-key",
                prior_terminal_v2_version_id="terminal-v2-version",
                prior_terminal_v2_body_sha256=terminal_body_sha256,
                prior_terminal_v2_outcome="DRAINED_COMPLETED",
                prior_final_worker_cardinality="ZERO",
                prior_h1g_drained_key="h1g-drained-key",
                prior_h1g_drained_version_id="h1g-drained-version",
                prior_h1g_drained_body_sha256=drained_body_sha256,
                prior_spend_ledger_head_identity=spend_identity,
                prior_snapshot_cleanup_control_identity={
                    "identity": "1" * 64
                },
                snapshot_cleanup_lineage=lineage,
                snapshot_cleanup_lineage_sha256=canonical_sha256(lineage),
                index_from_revision=1,
                index_to_revision=2,
                writer_invocation_nonce_sha256=hashlib.sha256(
                    raw_nonce
                ).hexdigest(),
                cloudformation_operation_identity_sha256=operation_identity,
            ),
        ),
        "control": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-2#CONTROL"),
            _closed_record(
                "glm52_production_control",
                activation_id=activation_id,
                activation_ordinal=2,
            ),
        ),
        "recovery_control": ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-2#RECOVERY_CONTROL"),
            _closed_record(
                "glm52_production_recovery_control",
                activation_id=activation_id,
                activation_ordinal=2,
            ),
        ),
        "finalization_control": ExactPut(
            LedgerKey(
                RUN_ID, "ACTIVATION#activation-2#FINALIZATION_CONTROL"
            ),
            _closed_record(
                "glm52_production_finalization_control",
                activation_id=activation_id,
                activation_ordinal=2,
            ),
        ),
        "snapshot_cleanup_control": ExactPut(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-2#SNAPSHOT_CLEANUP_CONTROL",
            ),
            _closed_record(
                "glm52_production_snapshot_cleanup_control",
                activation_id=activation_id,
                activation_ordinal=2,
            ),
        ),
        "cleanup_heads": (),
        "domain": "ACTIVATION_ROLLOVER",
        "operation_identity_sha256": operation_identity,
        "raw_owner_nonce": raw_nonce,
    }


def _apply_later_rollover_mutant(
    arguments: dict[str, object], mutant: str
) -> None:
    index = arguments["index"]
    rollover = arguments["rollover"]
    assert type(index) is ExactUpdate
    assert type(rollover) is ExactPut
    before = dict(index.before)
    after = dict(index.after)
    rollover_item = dict(rollover.item)
    if mutant == "before_run":
        before["run_id"] = "foreign-run"
    elif mutant == "before_campaign":
        before["campaign_identity_sha256"] = "c" * 64
    elif mutant == "before_activation":
        before["current_activation_id"] = "activation-foreign"
    elif mutant == "before_ordinal":
        before["current_activation_ordinal"] = 9
    elif mutant == "before_revision":
        before["revision"] = 9
    elif mutant == "after_run":
        after["run_id"] = "foreign-run"
    elif mutant == "after_campaign":
        after["campaign_identity_sha256"] = "c" * 64
    elif mutant == "after_activation":
        after["current_activation_id"] = "activation-foreign"
    elif mutant == "after_ordinal":
        after["current_activation_ordinal"] = 9
    elif mutant == "after_prior_activation":
        after["prior_activation_id"] = "activation-foreign"
    elif mutant == "after_terminal":
        after["prior_activation_terminal_v2_identity"] = {
            **after["prior_activation_terminal_v2_identity"],
            "body_sha256": "c" * 64,
        }
    elif mutant == "after_drained":
        after["prior_h1g_drained_identity"] = {
            **after["prior_h1g_drained_identity"],
            "body_sha256": "c" * 64,
        }
    elif mutant == "after_spend":
        after["prior_spend_ledger_head_identity"] = {
            "identity": "c" * 64
        }
    elif mutant == "after_lineage":
        after["snapshot_cleanup_lineage_sha256"] = "c" * 64
    elif mutant in {"rollover_foreign_lineage", "rollover_missing_lineage"}:
        lineage = (
            []
            if mutant == "rollover_missing_lineage"
            else [
                {
                    **rollover_item["snapshot_cleanup_lineage"][0],
                    "cleanup_control_root_identity_sha256": "c" * 64,
                }
            ]
        )
        rollover_item["snapshot_cleanup_lineage"] = lineage
        rollover_item["snapshot_cleanup_lineage_sha256"] = canonical_sha256(
            lineage
        )
    else:
        raise AssertionError("unhandled later rollover mutant")
    arguments["index"] = ExactUpdate(index.key, before, after)
    arguments["rollover"] = ExactPut(
        rollover.key, _rehash(rollover_item)
    )


def _liability_action_arguments(
    raw_nonce: bytes,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    owner = {
        "owner_attempt": 1,
        "owner_execution_arn": "liability-owner-execution",
        "owner_state_machine_version_arn": "liability-owner-version",
        "owner_dispatch_identity_sha256": "d" * 64,
        "owner_invocation_nonce_sha256": nonce_sha,
        "owner_hard_expires_at": TS,
    }
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    liability_before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        **owner,
    )
    liability_after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        revision=2,
        **owner,
    )
    liability_identity = canonical_sha256(liability_before)
    action_kind = "POST_TERMINAL_ALLOCATION_DISCOVER"
    action_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY_ACTION#"
        "00000001#POST_TERMINAL_ALLOCATION_DISCOVER#00000001"
    )
    action_identity = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "allocation_ordinal": 1,
        "allocation_ordinal_text": "00000001",
        "worker_launch_identity_sha256": liability_before[
            "worker_launch_identity_sha256"
        ],
        "worker_launch_liability_identity_sha256": liability_identity,
        "action_kind": action_kind,
        **owner,
    }
    action_before = _closed_record(
        "glm52_production_worker_launch_liability_action",
        **action_identity,
    )
    action_after = _closed_record(
        "glm52_production_worker_launch_liability_action",
        **action_identity,
        state="CONSUMED",
        authority_audit_body_sha256="e" * 64,
        authority_audit_closing_revision=1,
        authorized_transition_from_revision=1,
        authorized_transition_to_revision=2,
        revision=2,
    )
    allocation_key = (
        "ACTIVATION#activation-1#POST_TERMINAL_ALLOCATION#"
        "00000001#i-123"
    )
    allocation = _closed_record(
        "glm52_production_post_terminal_allocation",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        allocation_ordinal=1,
        worker_launch_identity_sha256=liability_before[
            "worker_launch_identity_sha256"
        ],
        worker_launch_liability_identity_sha256=liability_identity,
        instance_id="i-123",
        **owner,
    )
    return (
        {
            "action_name": action_kind,
            "index": ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                _activation_index(),
            ),
            "liability": ExactUpdate(
                LedgerKey(RUN_ID, liability_key),
                liability_before,
                liability_after,
            ),
            "action": ExactUpdate(
                LedgerKey(RUN_ID, action_key),
                action_before,
                action_after,
            ),
            "post_terminal_allocation": ExactPut(
                LedgerKey(RUN_ID, allocation_key),
                allocation,
            ),
            "domain": "LIABILITY_ACTION",
            "operation_identity_sha256": "7" * 64,
            "raw_owner_nonce": raw_nonce,
        },
        {
            "index": _activation_index(),
            "liability_before": liability_before,
            "action_before": action_before,
        },
    )


def _settlement_arguments(
    raw_nonce: bytes,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    owner = {
        "owner_attempt": 1,
        "owner_execution_arn": "liability-owner-execution",
        "owner_state_machine_version_arn": "liability-owner-version",
        "owner_dispatch_identity_sha256": "d" * 64,
        "owner_invocation_nonce_sha256": nonce_sha,
        "owner_hard_expires_at": TS,
    }
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    liability_before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="REJECTION_PROVED_AWAITING_TERMINAL_V2",
        **owner,
    )
    liability_after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="SETTLED_NO_INSTANCE_REJECTED",
        revision=2,
    )
    liability_identity = canonical_sha256(liability_before)
    action_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY_ACTION#"
        "00000001#LIABILITY_SETTLE#00000001"
    )
    action_identity = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "allocation_ordinal": 1,
        "allocation_ordinal_text": "00000001",
        "worker_launch_identity_sha256": liability_before[
            "worker_launch_identity_sha256"
        ],
        "worker_launch_liability_identity_sha256": liability_identity,
        "action_kind": "LIABILITY_SETTLE",
        **owner,
    }
    action_before = _closed_record(
        "glm52_production_worker_launch_liability_action",
        **action_identity,
    )
    action_after = _closed_record(
        "glm52_production_worker_launch_liability_action",
        **action_identity,
        state="CONSUMED",
        authority_audit_body_sha256="e" * 64,
        authority_audit_closing_revision=1,
        authorized_transition_from_revision=1,
        authorized_transition_to_revision=2,
        revision=2,
    )
    settlement_key = (
        "ACTIVATION#activation-1#"
        "WORKER_LAUNCH_LIABILITY_SETTLEMENT#00000001"
    )
    settlement = _closed_record(
        "glm52_production_worker_launch_liability_settlement",
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        allocation_ordinal=1,
        worker_launch_identity_sha256=liability_before[
            "worker_launch_identity_sha256"
        ],
        worker_launch_liability_identity_sha256=liability_identity,
        **{
            field: value
            for field, value in owner.items()
            if field not in {"owner_attempt", "owner_hard_expires_at"}
        },
    )
    return (
        {
            "index": ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                _activation_index(),
            ),
            "liability_action": ExactUpdate(
                LedgerKey(RUN_ID, action_key),
                action_before,
                action_after,
            ),
            "liability": ExactUpdate(
                LedgerKey(RUN_ID, liability_key),
                liability_before,
                liability_after,
            ),
            "settlement": ExactPut(
                LedgerKey(RUN_ID, settlement_key), settlement
            ),
            "domain": "LIABILITY_SETTLEMENT",
            "operation_identity_sha256": "8" * 64,
            "raw_owner_nonce": raw_nonce,
        },
        {
            "index": _activation_index(),
            "action_before": action_before,
            "liability_before": liability_before,
        },
    )


def _consume_action_arguments(
    raw_nonce: bytes,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    action_key = (
        "ACTIVATION#activation-1#ACTION#00000001#S3_CREATE#00000001"
    )
    control_before = _control(
        last_sky_post_action_key=action_key,
        last_sky_post_state="ARMED",
    )
    control_after = _control(
        last_sky_post_action_key=action_key,
        last_sky_post_state="CONSUMED",
        revision=2,
    )
    action_identity = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "action_kind": "S3_CREATE",
        "owner_epoch": 1,
        "owner_execution_arn": "execution-arn",
        "armed_by_epoch": 1,
        "armed_by_execution_arn": "execution-arn",
        "armed_by_state_machine_version_arn": "state-machine-version-arn",
        "barrier_nonce_sha256": SHA,
    }
    action_before = _closed_record(
        "glm52_production_action",
        **action_identity,
    )
    action_after = _closed_record(
        "glm52_production_action",
        **action_identity,
        state="CONSUMED",
        authority_audit_body_sha256="b" * 64,
        authority_audit_closing_revision=1,
        authorized_transition_from_revision=1,
        authorized_transition_to_revision=2,
        revision=2,
    )
    return (
        {
            "index": ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                _activation_index(),
            ),
            "control": ExactUpdate(
                LedgerKey(RUN_ID, "ACTIVATION#activation-1#CONTROL"),
                control_before,
                control_after,
            ),
            "action": ExactUpdate(
                LedgerKey(RUN_ID, action_key),
                action_before,
                action_after,
            ),
            "domain": "ACTION_CONSUME",
            "operation_identity_sha256": "f" * 64,
            "raw_owner_nonce": raw_nonce,
        },
        {
            "index": _activation_index(),
            "control_before": control_before,
            "action_before": action_before,
        },
    )


@pytest.mark.parametrize(
    "mutant",
    [
        "campaign",
        "generation",
        "action_key",
        "barrier",
        "owner_epoch",
        "owner_execution",
        "armed_epoch",
        "armed_execution",
        "armed_version",
        "audit_closing_revision",
        "audit_from_revision",
        "audit_to_revision",
        "control_active_epoch",
        "control_active_execution",
        "control_active_version",
        "control_action_key_changed",
    ],
)
def test_action_consume_rejects_cross_record_same_class_mutants(
    mutant: str,
) -> None:
    raw_nonce = b"C" * 32
    arguments, _ = _consume_action_arguments(raw_nonce)
    control = arguments["control"]
    action = arguments["action"]
    assert type(control) is ExactUpdate
    assert type(action) is ExactUpdate
    control_before = dict(control.before)
    control_after = dict(control.after)
    action_before = dict(action.before)
    action_after = dict(action.after)

    if mutant == "campaign":
        action_before["campaign_identity_sha256"] = "c" * 64
        action_after["campaign_identity_sha256"] = "c" * 64
    elif mutant == "generation":
        action_before["generation"] = 2
        action_before["generation_text"] = "00000002"
        action_after["generation"] = 2
        action_after["generation_text"] = "00000002"
    elif mutant == "action_key":
        control_before["last_sky_post_action_key"] = "other-action-key"
    elif mutant == "barrier":
        action_before["barrier_nonce_sha256"] = "c" * 64
        action_after["barrier_nonce_sha256"] = "c" * 64
    elif mutant == "owner_epoch":
        action_before["owner_epoch"] = 2
        action_after["owner_epoch"] = 2
    elif mutant == "owner_execution":
        action_before["owner_execution_arn"] = "foreign-execution"
        action_after["owner_execution_arn"] = "foreign-execution"
    elif mutant == "armed_epoch":
        action_before["armed_by_epoch"] = 2
        action_after["armed_by_epoch"] = 2
    elif mutant == "armed_execution":
        action_before["armed_by_execution_arn"] = "foreign-execution"
        action_after["armed_by_execution_arn"] = "foreign-execution"
    elif mutant == "armed_version":
        action_before["armed_by_state_machine_version_arn"] = "foreign-version"
        action_after["armed_by_state_machine_version_arn"] = "foreign-version"
    elif mutant == "audit_closing_revision":
        action_after["authority_audit_closing_revision"] = 2
    elif mutant == "audit_from_revision":
        action_after["authorized_transition_from_revision"] = 2
    elif mutant == "audit_to_revision":
        action_after["authorized_transition_to_revision"] = 3
    elif mutant == "control_active_epoch":
        control_after["active_epoch"] = 2
        action_after["owner_epoch"] = 2
    elif mutant == "control_active_execution":
        control_after["active_execution_arn"] = "other-execution"
        action_after["owner_execution_arn"] = "other-execution"
    elif mutant == "control_active_version":
        control_after["active_state_machine_version_arn"] = "other-version"
        action_after["armed_by_state_machine_version_arn"] = "other-version"
    elif mutant == "control_action_key_changed":
        control_before["last_sky_post_action_key"] = "prior-action-key"
    else:
        raise AssertionError("unhandled mutant")

    arguments["control"] = ExactUpdate(
        control.key, control_before, control_after
    )
    arguments["action"] = ExactUpdate(
        action.key, action_before, action_after
    )
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", arguments["index"].expected)
    client.install(control.key.sort_key, control_before)
    client.install(action.key.sort_key, action_before)

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).consume_action(**arguments)

    assert not client.calls


@pytest.mark.parametrize(
    "mutant",
    [
        "campaign",
        "liability_campaign",
        "generation",
        "generation_text",
        "action_allocation_ordinal",
        "worker_launch_identity",
        "liability_identity",
        "owner_execution",
        "owner_version",
        "owner_dispatch",
        "owner_nonce",
        "owner_attempt",
        "owner_hard_expiry",
        "audit_closing_revision",
        "audit_from_revision",
        "audit_to_revision",
        "liability_to_revision",
        "allocation_campaign",
        "allocation_activation",
        "allocation_generation",
        "allocation_ordinal",
        "allocation_worker_launch_identity",
        "allocation_liability_identity",
        "allocation_owner_execution",
        "allocation_owner_version",
        "allocation_owner_dispatch",
        "allocation_owner_nonce",
        "allocation_owner_attempt",
        "allocation_owner_hard_expiry",
    ],
)
def test_liability_action_rejects_cross_resource_same_class_mutants(
    mutant: str,
) -> None:
    raw_nonce = b"L" * 32
    arguments, _ = _liability_action_arguments(raw_nonce)
    liability = arguments["liability"]
    action = arguments["action"]
    allocation = arguments["post_terminal_allocation"]
    assert type(liability) is ExactUpdate
    assert type(action) is ExactUpdate
    assert type(allocation) is ExactPut
    liability_before = dict(liability.before)
    liability_after = dict(liability.after)
    action_before = dict(action.before)
    action_after = dict(action.after)
    allocation_item = dict(allocation.item)

    if mutant == "campaign":
        action_before["campaign_identity_sha256"] = "c" * 64
        action_after["campaign_identity_sha256"] = "c" * 64
    elif mutant == "liability_campaign":
        liability_before["campaign_identity_sha256"] = "c" * 64
        liability_after["campaign_identity_sha256"] = "c" * 64
        action_before["campaign_identity_sha256"] = "c" * 64
        action_after["campaign_identity_sha256"] = "c" * 64
        allocation_item["campaign_identity_sha256"] = "c" * 64
    elif mutant == "generation":
        action_before["generation"] = 2
        action_before["generation_text"] = "00000002"
        action_after["generation"] = 2
        action_after["generation_text"] = "00000002"
    elif mutant == "generation_text":
        action_before["generation_text"] = "00000002"
        action_after["generation_text"] = "00000002"
    elif mutant == "action_allocation_ordinal":
        action_before["allocation_ordinal"] = 2
        action_before["allocation_ordinal_text"] = "00000002"
        action_after["allocation_ordinal"] = 2
        action_after["allocation_ordinal_text"] = "00000002"
    elif mutant == "worker_launch_identity":
        action_before["worker_launch_identity_sha256"] = "c" * 64
        action_after["worker_launch_identity_sha256"] = "c" * 64
    elif mutant == "liability_identity":
        action_before["worker_launch_liability_identity_sha256"] = "c" * 64
        action_after["worker_launch_liability_identity_sha256"] = "c" * 64
    elif mutant == "owner_execution":
        action_before["owner_execution_arn"] = "foreign-execution"
        action_after["owner_execution_arn"] = "foreign-execution"
    elif mutant == "owner_version":
        action_before["owner_state_machine_version_arn"] = "foreign-version"
        action_after["owner_state_machine_version_arn"] = "foreign-version"
    elif mutant == "owner_dispatch":
        action_before["owner_dispatch_identity_sha256"] = "c" * 64
        action_after["owner_dispatch_identity_sha256"] = "c" * 64
    elif mutant == "owner_nonce":
        action_before["owner_invocation_nonce_sha256"] = "c" * 64
        action_after["owner_invocation_nonce_sha256"] = "c" * 64
    elif mutant == "owner_attempt":
        action_before["owner_attempt"] = 2
        action_after["owner_attempt"] = 2
    elif mutant == "owner_hard_expiry":
        action_before["owner_hard_expires_at"] = (
            "2026-07-28T13:00:00Z"
        )
        action_after["owner_hard_expires_at"] = (
            "2026-07-28T13:00:00Z"
        )
    elif mutant == "audit_closing_revision":
        action_after["authority_audit_closing_revision"] = 2
    elif mutant == "audit_from_revision":
        action_after["authorized_transition_from_revision"] = 2
    elif mutant == "audit_to_revision":
        action_after["authorized_transition_to_revision"] = 3
    elif mutant == "liability_to_revision":
        liability_after["revision"] = 3
    elif mutant == "allocation_campaign":
        allocation_item["campaign_identity_sha256"] = "c" * 64
    elif mutant == "allocation_activation":
        allocation_item["activation_id"] = "activation-2"
    elif mutant == "allocation_generation":
        allocation_item["generation"] = 2
    elif mutant == "allocation_ordinal":
        allocation_item["allocation_ordinal"] = 2
    elif mutant == "allocation_worker_launch_identity":
        allocation_item["worker_launch_identity_sha256"] = "c" * 64
    elif mutant == "allocation_liability_identity":
        allocation_item["worker_launch_liability_identity_sha256"] = "c" * 64
    elif mutant == "allocation_owner_execution":
        allocation_item["owner_execution_arn"] = "foreign-execution"
    elif mutant == "allocation_owner_version":
        allocation_item["owner_state_machine_version_arn"] = "foreign-version"
    elif mutant == "allocation_owner_dispatch":
        allocation_item["owner_dispatch_identity_sha256"] = "c" * 64
    elif mutant == "allocation_owner_nonce":
        allocation_item["owner_invocation_nonce_sha256"] = "c" * 64
    elif mutant == "allocation_owner_attempt":
        allocation_item["owner_attempt"] = 2
    elif mutant == "allocation_owner_hard_expiry":
        allocation_item["owner_hard_expires_at"] = (
            "2026-07-28T13:00:00Z"
        )
    else:
        raise AssertionError("unhandled mutant")

    arguments["liability"] = ExactUpdate(
        liability.key, liability_before, liability_after
    )
    arguments["action"] = ExactUpdate(
        action.key, action_before, action_after
    )
    arguments["post_terminal_allocation"] = ExactPut(
        allocation.key, _rehash(allocation_item)
    )
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", arguments["index"].expected)
    client.install(liability.key.sort_key, liability_before)
    client.install(action.key.sort_key, action_before)

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).consume_liability_action(**arguments)

    assert not client.calls


@pytest.mark.parametrize(
    "mutant",
    [
        "campaign",
        "liability_campaign",
        "generation",
        "generation_text",
        "allocation_ordinal",
        "worker_launch_identity",
        "liability_identity",
        "owner_execution",
        "owner_version",
        "owner_dispatch",
        "owner_nonce",
        "owner_attempt",
        "owner_hard_expiry",
        "audit_closing_revision",
        "audit_from_revision",
        "audit_to_revision",
        "liability_to_revision",
        "settlement_campaign",
        "settlement_activation",
        "settlement_generation",
        "settlement_ordinal",
        "settlement_worker_launch_identity",
        "settlement_liability_identity",
        "settlement_owner_execution",
        "settlement_owner_version",
        "settlement_owner_dispatch",
        "settlement_owner_nonce",
    ],
)
def test_liability_settlement_rejects_cross_resource_same_class_mutants(
    mutant: str,
) -> None:
    raw_nonce = b"S" * 32
    arguments, _ = _settlement_arguments(raw_nonce)
    liability = arguments["liability"]
    action = arguments["liability_action"]
    settlement = arguments["settlement"]
    assert type(liability) is ExactUpdate
    assert type(action) is ExactUpdate
    assert type(settlement) is ExactPut
    liability_before = dict(liability.before)
    liability_after = dict(liability.after)
    action_before = dict(action.before)
    action_after = dict(action.after)
    settlement_item = dict(settlement.item)

    if mutant == "campaign":
        action_before["campaign_identity_sha256"] = "c" * 64
        action_after["campaign_identity_sha256"] = "c" * 64
    elif mutant == "liability_campaign":
        liability_before["campaign_identity_sha256"] = "c" * 64
        liability_after["campaign_identity_sha256"] = "c" * 64
        action_before["campaign_identity_sha256"] = "c" * 64
        action_after["campaign_identity_sha256"] = "c" * 64
        settlement_item["campaign_identity_sha256"] = "c" * 64
    elif mutant == "generation":
        action_before["generation"] = 2
        action_before["generation_text"] = "00000002"
        action_after["generation"] = 2
        action_after["generation_text"] = "00000002"
    elif mutant == "generation_text":
        action_before["generation_text"] = "00000002"
        action_after["generation_text"] = "00000002"
    elif mutant == "allocation_ordinal":
        action_before["allocation_ordinal"] = 2
        action_before["allocation_ordinal_text"] = "00000002"
        action_after["allocation_ordinal"] = 2
        action_after["allocation_ordinal_text"] = "00000002"
    elif mutant == "worker_launch_identity":
        action_before["worker_launch_identity_sha256"] = "c" * 64
        action_after["worker_launch_identity_sha256"] = "c" * 64
    elif mutant == "liability_identity":
        action_before["worker_launch_liability_identity_sha256"] = "c" * 64
        action_after["worker_launch_liability_identity_sha256"] = "c" * 64
    elif mutant == "owner_execution":
        action_before["owner_execution_arn"] = "foreign-execution"
        action_after["owner_execution_arn"] = "foreign-execution"
    elif mutant == "owner_version":
        action_before["owner_state_machine_version_arn"] = "foreign-version"
        action_after["owner_state_machine_version_arn"] = "foreign-version"
    elif mutant == "owner_dispatch":
        action_before["owner_dispatch_identity_sha256"] = "c" * 64
        action_after["owner_dispatch_identity_sha256"] = "c" * 64
    elif mutant == "owner_nonce":
        action_before["owner_invocation_nonce_sha256"] = "c" * 64
        action_after["owner_invocation_nonce_sha256"] = "c" * 64
    elif mutant == "owner_attempt":
        action_before["owner_attempt"] = 2
        action_after["owner_attempt"] = 2
    elif mutant == "owner_hard_expiry":
        action_before["owner_hard_expires_at"] = (
            "2026-07-28T13:00:00Z"
        )
        action_after["owner_hard_expires_at"] = (
            "2026-07-28T13:00:00Z"
        )
    elif mutant == "audit_closing_revision":
        action_after["authority_audit_closing_revision"] = 2
    elif mutant == "audit_from_revision":
        action_after["authorized_transition_from_revision"] = 2
    elif mutant == "audit_to_revision":
        action_after["authorized_transition_to_revision"] = 3
    elif mutant == "liability_to_revision":
        liability_after["revision"] = 3
    elif mutant == "settlement_campaign":
        settlement_item["campaign_identity_sha256"] = "c" * 64
    elif mutant == "settlement_activation":
        settlement_item["activation_id"] = "activation-2"
    elif mutant == "settlement_generation":
        settlement_item["generation"] = 2
    elif mutant == "settlement_ordinal":
        settlement_item["allocation_ordinal"] = 2
    elif mutant == "settlement_worker_launch_identity":
        settlement_item["worker_launch_identity_sha256"] = "c" * 64
    elif mutant == "settlement_liability_identity":
        settlement_item["worker_launch_liability_identity_sha256"] = "c" * 64
    elif mutant == "settlement_owner_execution":
        settlement_item["owner_execution_arn"] = "foreign-execution"
    elif mutant == "settlement_owner_version":
        settlement_item["owner_state_machine_version_arn"] = "foreign-version"
    elif mutant == "settlement_owner_dispatch":
        settlement_item["owner_dispatch_identity_sha256"] = "c" * 64
    elif mutant == "settlement_owner_nonce":
        settlement_item["owner_invocation_nonce_sha256"] = "c" * 64
    else:
        raise AssertionError("unhandled mutant")

    arguments["liability"] = ExactUpdate(
        liability.key, liability_before, liability_after
    )
    arguments["liability_action"] = ExactUpdate(
        action.key, action_before, action_after
    )
    arguments["settlement"] = ExactPut(
        settlement.key, _rehash(settlement_item)
    )
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", arguments["index"].expected)
    client.install(liability.key.sort_key, liability_before)
    client.install(action.key.sort_key, action_before)

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).settle_worker_launch_liability(**arguments)

    assert not client.calls


def test_liability_settlement_installs_final_settlement_identity() -> None:
    raw_nonce = b"I" * 32
    arguments, installed = _settlement_arguments(raw_nonce)
    liability = arguments["liability"]
    assert type(liability) is ExactUpdate
    input_identity = liability.after["settlement_identity_sha256"]
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", installed["index"])
    client.install(
        arguments["liability_action"].key.sort_key,
        installed["action_before"],
    )
    client.install(
        arguments["liability"].key.sort_key,
        installed["liability_before"],
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).settle_worker_launch_liability(**arguments)

    settlement_identity = result.records[3]["canonical_body_sha256"]
    assert input_identity != settlement_identity
    assert result.records[2]["settlement_identity_sha256"] == (
        settlement_identity
    )


def test_liability_settlement_uses_exact_frozen_sentinel_preimage() -> None:
    raw_nonce = b"J" * 32
    arguments, installed = _settlement_arguments(raw_nonce)
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", installed["index"])
    client.install(
        arguments["liability_action"].key.sort_key,
        installed["action_before"],
    )
    client.install(
        arguments["liability"].key.sort_key,
        installed["liability_before"],
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).settle_worker_launch_liability(**arguments)

    def normalized(
        record: dict[str, object],
        *,
        derived_settlement_identity: bool = False,
    ) -> dict[str, object]:
        value = {
            key: deepcopy(item)
            for key, item in record.items()
            if key != "canonical_body_sha256"
            and key != "transaction_bytes_sha256"
            and not key.endswith(
                "transaction_client_request_token_sha256"
            )
        }
        if derived_settlement_identity:
            value["settlement_identity_sha256"] = (
                "__DERIVED_SETTLEMENT_CANONICAL_BODY_SHA256__"
            )
        return value

    action = arguments["liability_action"]
    liability = arguments["liability"]
    settlement = arguments["settlement"]
    ordered_plan = [
        {
            "kind": "ConditionCheck",
            "key": {
                "run_id": arguments["index"].key.run_id,
                "sort_key": arguments["index"].key.sort_key,
            },
            "condition": "EXACT_EXPECTED",
            "expected": normalized(dict(result.records[0])),
        },
        {
            "kind": "Update",
            "key": {
                "run_id": action.key.run_id,
                "sort_key": action.key.sort_key,
            },
            "condition": "EXACT_PRESTATE",
            "before": normalized(dict(action.before)),
            "after": normalized(dict(result.records[1])),
        },
        {
            "kind": "Update",
            "key": {
                "run_id": liability.key.run_id,
                "sort_key": liability.key.sort_key,
            },
            "condition": "EXACT_PRESTATE",
            "before": normalized(dict(liability.before)),
            "after": normalized(
                dict(result.records[2]),
                derived_settlement_identity=True,
            ),
        },
        {
            "kind": "Put",
            "key": {
                "run_id": settlement.key.run_id,
                "sort_key": settlement.key.sort_key,
            },
            "condition": "ABSENT_PK_AND_SK",
            "item": normalized(dict(result.records[3])),
        },
    ]
    transaction_identity = canonical_sha256(
        {
            "schema_version": 1,
            "domain": "LIABILITY_SETTLEMENT",
            "operation_identity_sha256": "8" * 64,
            "ordered_logical_write_plan": ordered_plan,
        }
    )
    digest = hashlib.sha256()
    for item in (
        b"glm52-ddb-crt-v1",
        b"LIABILITY_SETTLEMENT",
        ("8" * 64).encode("ascii"),
        transaction_identity.encode("ascii"),
    ):
        digest.update(item)
        digest.update(b"\0")
    digest.update(raw_nonce)
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert write["ClientRequestToken"] == "h1g-" + digest.hexdigest()[:32]


def test_rollover_uses_frozen_sentinel_preimage_and_final_control_identity() -> None:
    raw_nonce = b"R" * 32
    arguments = _rollover_arguments(raw_nonce)
    input_control_identity = arguments["control"].item[
        "rollover_identity_sha256"
    ]
    client = AwsShapedSimulator()

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**arguments)

    rollover_identity = result.records[1]["canonical_body_sha256"]
    assert input_control_identity != rollover_identity
    assert all(
        record["rollover_identity_sha256"] == rollover_identity
        for record in result.records[2:6]
    )
    for record in result.records[1:6]:
        if "canonical_body_sha256" not in record:
            continue
        body = dict(record)
        expected = body.pop("canonical_body_sha256")
        assert expected == canonical_sha256(body)

    normalized_records: list[dict[str, object]] = []
    for position, record in enumerate(result.records):
        normalized = {
            key: deepcopy(value)
            for key, value in record.items()
            if key != "canonical_body_sha256"
            and key != "transaction_bytes_sha256"
            and not key.endswith(
                "transaction_client_request_token_sha256"
            )
        }
        if 2 <= position <= 5:
            normalized["rollover_identity_sha256"] = (
                "__DERIVED_ROLLOVER_CANONICAL_BODY_SHA256__"
            )
        normalized_records.append(normalized)
    ordered_plan = [
        {
            "kind": "Update",
            "key": {
                "run_id": arguments["index"].key.run_id,
                "sort_key": arguments["index"].key.sort_key,
            },
            "condition": "ABSENT_PK_AND_SK",
            "before": {},
            "after": normalized_records[0],
        }
    ]
    for plan, record in zip(
        (
            arguments["rollover"],
            arguments["control"],
            arguments["recovery_control"],
            arguments["finalization_control"],
            arguments["snapshot_cleanup_control"],
        ),
        normalized_records[1:],
    ):
        ordered_plan.append(
            {
                "kind": "Put",
                "key": {
                    "run_id": plan.key.run_id,
                    "sort_key": plan.key.sort_key,
                },
                "condition": "ABSENT_PK_AND_SK",
                "item": record,
            }
        )
    expected_preimage = {
        "schema_version": 1,
        "domain": "ACTIVATION_ROLLOVER",
        "operation_identity_sha256": "b" * 64,
        "ordered_logical_write_plan": ordered_plan,
    }
    assert result.records[1]["transaction_bytes_sha256"] == (
        canonical_sha256(expected_preimage)
    )


def test_later_rollover_rejects_foreign_prior_index_activation_before_sdk() -> None:
    arguments = _later_rollover_arguments(b"P" * 32)
    index = arguments["index"]
    assert type(index) is ExactUpdate
    arguments["index"] = ExactUpdate(
        index.key,
        {
            **index.before,
            "current_activation_id": "activation-foreign",
        },
        index.after,
    )
    client = AwsShapedSimulator()

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**arguments)

    assert not client.calls


def test_later_rollover_accepts_exact_frozen_predecessor_projection() -> None:
    arguments = _later_rollover_arguments(b"Q" * 32)
    index = arguments["index"]
    assert type(index) is ExactUpdate
    client = AwsShapedSimulator()
    client.install(index.key.sort_key, index.before)

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**arguments)

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records[0] == index.after


def test_first_rollover_rejects_nonempty_index_prestate_before_sdk() -> None:
    arguments = _rollover_arguments(b"E" * 32)
    index = arguments["index"]
    assert type(index) is ExactUpdate
    arguments["index"] = ExactUpdate(
        index.key, _activation_index(), index.after
    )
    client = AwsShapedSimulator()

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**arguments)

    assert not client.calls


@pytest.mark.parametrize(
    "mutant",
    [
        "before_run",
        "before_campaign",
        "before_activation",
        "before_ordinal",
        "before_revision",
        "after_run",
        "after_campaign",
        "after_activation",
        "after_ordinal",
        "after_prior_activation",
        "after_terminal",
        "after_drained",
        "after_spend",
        "after_lineage",
        "rollover_foreign_lineage",
        "rollover_missing_lineage",
    ],
)
def test_later_rollover_rejects_every_predecessor_projection_mutant(
    mutant: str,
) -> None:
    arguments = _later_rollover_arguments(b"G" * 32)
    _apply_later_rollover_mutant(arguments, mutant)
    client = AwsShapedSimulator()

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**arguments)

    assert not client.calls


@pytest.mark.parametrize(
    "mutant",
    [
        "after_lineage",
        "rollover_foreign_lineage",
        "rollover_missing_lineage",
    ],
)
def test_later_rollover_duplicate_rejects_foreign_or_missing_lineage(
    mutant: str,
) -> None:
    raw_nonce = b"H" * 32
    operation_identity = "b" * 64
    live_arguments = _later_rollover_arguments(
        raw_nonce, operation_identity
    )
    live_index = live_arguments["index"]
    assert type(live_index) is ExactUpdate
    live_client = AwsShapedSimulator()
    live_client.install(live_index.key.sort_key, live_index.before)
    live = DynamoLedgerAdapter(
        client=live_client, table_name="ledger-table"
    ).commit_rollover(**live_arguments)
    arguments = _later_rollover_arguments(raw_nonce, operation_identity)
    arguments.update(
        index=ExactUpdate(
            arguments["index"].key,
            arguments["index"].before,
            dict(live.records[0]),
        ),
        rollover=ExactPut(
            arguments["rollover"].key, dict(live.records[1])
        ),
        control=ExactPut(
            arguments["control"].key, dict(live.records[2])
        ),
        recovery_control=ExactPut(
            arguments["recovery_control"].key, dict(live.records[3])
        ),
        finalization_control=ExactPut(
            arguments["finalization_control"].key,
            dict(live.records[4]),
        ),
        snapshot_cleanup_control=ExactPut(
            arguments["snapshot_cleanup_control"].key,
            dict(live.records[5]),
        ),
        raw_owner_nonce=None,
        duplicate_operation_identity_sha256=operation_identity,
    )
    _apply_later_rollover_mutant(arguments, mutant)
    duplicate_client = AwsShapedSimulator()
    for plan in (
        arguments["index"],
        arguments["rollover"],
        arguments["control"],
        arguments["recovery_control"],
        arguments["finalization_control"],
        arguments["snapshot_cleanup_control"],
    ):
        record = (
            plan.after if type(plan) is ExactUpdate else plan.item
        )
        duplicate_client.install(
            plan.key.sort_key, dict(record)
        )

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=duplicate_client, table_name="ledger-table"
        ).commit_rollover(**arguments)

    assert not duplicate_client.calls


@pytest.mark.parametrize(
    "mutant",
    [
        "foreign_rollover_activation",
        "foreign_rollover_campaign",
        "foreign_rollover_ordinal",
        "foreign_control_activation",
        "foreign_control_ordinal",
        "foreign_recovery_activation",
        "foreign_finalization_activation",
        "foreign_cleanup_activation",
        "index_revision",
        "rollover_from_revision",
        "rollover_to_revision",
    ],
)
def test_rollover_rejects_cross_record_revision_and_activation_mutants(
    mutant: str,
) -> None:
    arguments = _rollover_arguments(b"F" * 32)
    if mutant == "foreign_rollover_activation":
        plan = arguments["rollover"]
        item = _rehash({**plan.item, "activation_id": "activation-2"})
        arguments["rollover"] = ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-2#ROLLOVER"),
            item,
        )
    elif mutant == "foreign_rollover_campaign":
        plan = arguments["rollover"]
        arguments["rollover"] = ExactPut(
            plan.key,
            _rehash(
                {**plan.item, "campaign_identity_sha256": "c" * 64}
            ),
        )
    elif mutant == "foreign_rollover_ordinal":
        plan = arguments["rollover"]
        arguments["rollover"] = ExactPut(
            plan.key,
            _rehash({**plan.item, "activation_ordinal": 2}),
        )
    elif mutant == "foreign_control_activation":
        plan = arguments["control"]
        item = dict(plan.item)
        item["activation_id"] = "activation-2"
        arguments["control"] = ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-2#CONTROL"),
            item,
        )
    elif mutant == "foreign_control_ordinal":
        plan = arguments["control"]
        arguments["control"] = ExactPut(
            plan.key, {**plan.item, "activation_ordinal": 2}
        )
    elif mutant == "foreign_recovery_activation":
        plan = arguments["recovery_control"]
        arguments["recovery_control"] = ExactPut(
            LedgerKey(RUN_ID, "ACTIVATION#activation-2#RECOVERY_CONTROL"),
            {**plan.item, "activation_id": "activation-2"},
        )
    elif mutant == "foreign_finalization_activation":
        plan = arguments["finalization_control"]
        arguments["finalization_control"] = ExactPut(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-2#FINALIZATION_CONTROL",
            ),
            {**plan.item, "activation_id": "activation-2"},
        )
    elif mutant == "foreign_cleanup_activation":
        plan = arguments["snapshot_cleanup_control"]
        arguments["snapshot_cleanup_control"] = ExactPut(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-2#SNAPSHOT_CLEANUP_CONTROL",
            ),
            {**plan.item, "activation_id": "activation-2"},
        )
    elif mutant == "index_revision":
        plan = arguments["index"]
        arguments["index"] = ExactUpdate(
            plan.key, plan.before, _activation_index(revision=2)
        )
    elif mutant == "rollover_from_revision":
        plan = arguments["rollover"]
        arguments["rollover"] = ExactPut(
            plan.key,
            _rehash({**plan.item, "index_from_revision": 1}),
        )
    elif mutant == "rollover_to_revision":
        plan = arguments["rollover"]
        arguments["rollover"] = ExactPut(
            plan.key,
            _rehash({**plan.item, "index_to_revision": 2}),
        )
    else:
        raise AssertionError("unhandled mutant")
    client = AwsShapedSimulator()

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**arguments)

    assert not client.calls


@pytest.mark.parametrize("control_position", [2, 3, 4, 5])
def test_rollover_duplicate_revalidates_final_control_identity(
    control_position: int,
) -> None:
    raw_nonce = b"D" * 32
    operation_identity = "b" * 64
    live = DynamoLedgerAdapter(
        client=AwsShapedSimulator(), table_name="ledger-table"
    ).commit_rollover(
        **_rollover_arguments(raw_nonce, operation_identity)
    )
    records = [dict(record) for record in live.records]
    rollover_identity = records[1]["canonical_body_sha256"]
    for position in range(2, 6):
        records[position]["rollover_identity_sha256"] = rollover_identity
        records[position] = _rehash(records[position])
    records[control_position]["rollover_identity_sha256"] = "c" * 64
    records[control_position] = _rehash(records[control_position])
    keys = [
        "ACTIVATION_INDEX",
        "ACTIVATION#activation-1#ROLLOVER",
        "ACTIVATION#activation-1#CONTROL",
        "ACTIVATION#activation-1#RECOVERY_CONTROL",
        "ACTIVATION#activation-1#FINALIZATION_CONTROL",
        "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL",
    ]
    client = AwsShapedSimulator()
    for key, record in zip(keys, records):
        client.install(key, record)
    arguments = _rollover_arguments(raw_nonce, operation_identity)
    arguments.update(
        index=ExactUpdate(arguments["index"].key, {}, records[0]),
        rollover=ExactPut(arguments["rollover"].key, records[1]),
        control=ExactPut(arguments["control"].key, records[2]),
        recovery_control=ExactPut(
            arguments["recovery_control"].key, records[3]
        ),
        finalization_control=ExactPut(
            arguments["finalization_control"].key, records[4]
        ),
        snapshot_cleanup_control=ExactPut(
            arguments["snapshot_cleanup_control"].key, records[5]
        ),
        raw_owner_nonce=None,
        duplicate_operation_identity_sha256=operation_identity,
    )

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**arguments)

    assert not client.calls


def test_attribute_value_codec_round_trips_closed_records_without_type_coercion() -> None:
    record = _activation_index()

    encoded = encode_item(record)

    assert encoded["schema_version"] == {"N": "1"}
    assert decode_item(encoded) == record
    assert decode_attribute_value(encode_attribute_value([1, "x", True, None])) == [
        1,
        "x",
        True,
        None,
    ]
    assert record == _activation_index()


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        Decimal("1.00"),
        {1},
        b"secret",
        {1: "not-a-string-key"},
        type("IntegerSubclass", (int,), {})(1),
        object(),
    ],
)
def test_attribute_value_codec_rejects_bool_as_integer_float_set_and_unknown_type(
    value: object,
) -> None:
    with pytest.raises(TypeError):
        encode_attribute_value(value)

    assert encode_attribute_value(True) == {"BOOL": True}
    with pytest.raises(ValueError):
        decode_attribute_value({"N": True})


def test_consistent_item_read_uses_consistent_read_and_no_projection() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")

    record = adapter.read_consistent_item(
        key=LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
        record_type="glm52_production_activation_index",
    )

    assert record == _activation_index()
    assert client.calls == [
        (
            "get_item",
            {
                "TableName": "ledger-table",
                "Key": {
                    "PK": {"S": "RUN#" + RUN_ID},
                    "SK": {"S": "ACTIVATION_INDEX"},
                },
                "ConsistentRead": True,
                "ReturnConsumedCapacity": "NONE",
            },
        )
    ]


def test_transact_get_preserves_exact_key_order_and_requires_every_item() -> None:
    client = AwsShapedSimulator()
    control_key = "ACTIVATION#activation-1#CONTROL"
    client.install(control_key, _control())
    client.install("ACTIVATION_INDEX", _activation_index())
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")

    records = adapter.read_coherent(
        items=(
            (
                LedgerKey(RUN_ID, control_key),
                "glm52_production_control",
            ),
            (
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                "glm52_production_activation_index",
            ),
        )
    )

    assert records == (_control(), _activation_index())
    call_name, request = client.calls[-1]
    assert call_name == "transact_get_items"
    assert request == {
        "TransactItems": [
            {
                "Get": {
                    "TableName": "ledger-table",
                    "Key": {
                        "PK": {"S": "RUN#" + RUN_ID},
                        "SK": {"S": control_key},
                    },
                }
            },
            {
                "Get": {
                    "TableName": "ledger-table",
                    "Key": {
                        "PK": {"S": "RUN#" + RUN_ID},
                        "SK": {"S": "ACTIVATION_INDEX"},
                    },
                }
            },
        ],
        "ReturnConsumedCapacity": "NONE",
    }
    del client.items[("RUN#" + RUN_ID, "ACTIVATION_INDEX")]
    with pytest.raises(Exception, match="required coherent item"):
        adapter.read_coherent(
            items=(
                (
                    LedgerKey(RUN_ID, control_key),
                    "glm52_production_control",
                ),
                (
                    LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                    "glm52_production_activation_index",
                ),
            )
        )


def test_rollover_request_is_index_update_five_absent_puts_and_lineage_checks() -> None:
    client = AwsShapedSimulator()
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")
    activation_id = "activation-1"
    raw_nonce = b"n" * 32
    index = ExactUpdate(
        LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
        {},
        _activation_index(
            snapshot_cleanup_lineage_sha256=canonical_sha256([])
        ),
    )
    rollover = ExactPut(
        LedgerKey(RUN_ID, "ACTIVATION#activation-1#ROLLOVER"),
        _closed_record(
            "glm52_production_rollover",
            activation_id=activation_id,
            writer_invocation_nonce_sha256=hashlib.sha256(raw_nonce).hexdigest(),
        ),
    )
    control = ExactPut(
        LedgerKey(RUN_ID, "ACTIVATION#activation-1#CONTROL"),
        _closed_record("glm52_production_control", activation_id=activation_id),
    )
    recovery = ExactPut(
        LedgerKey(RUN_ID, "ACTIVATION#activation-1#RECOVERY_CONTROL"),
        _closed_record(
            "glm52_production_recovery_control", activation_id=activation_id
        ),
    )
    finalization = ExactPut(
        LedgerKey(RUN_ID, "ACTIVATION#activation-1#FINALIZATION_CONTROL"),
        _closed_record(
            "glm52_production_finalization_control", activation_id=activation_id
        ),
    )
    cleanup = ExactPut(
        LedgerKey(RUN_ID, "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL"),
        _closed_record(
            "glm52_production_snapshot_cleanup_control",
            activation_id=activation_id,
        ),
    )

    resolution = adapter.commit_rollover(
        index=index,
        rollover=rollover,
        control=control,
        recovery_control=recovery,
        finalization_control=finalization,
        snapshot_cleanup_control=cleanup,
        cleanup_heads=(),
        domain="ACTIVATION_ROLLOVER",
        operation_identity_sha256="b" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert resolution.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert resolution.may_issue_external_side_effect is True
    write_calls = [
        request for name, request in client.calls if name == "transact_write_items"
    ]
    assert len(write_calls) == 1
    request = write_calls[0]
    assert len(request["ClientRequestToken"]) == 36
    assert request["ClientRequestToken"].startswith("h1g-")
    assert request["ReturnConsumedCapacity"] == "NONE"
    assert request["ReturnItemCollectionMetrics"] == "NONE"
    assert [next(iter(item)) for item in request["TransactItems"]] == [
        "Update",
        "Put",
        "Put",
        "Put",
        "Put",
        "Put",
    ]
    for put in request["TransactItems"][1:]:
        assert put["Put"]["ConditionExpression"] == (
            "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
        )
    read_calls = [
        request for name, request in client.calls if name == "transact_get_items"
    ]
    assert len(read_calls) == 1
    assert [
        decode_item(entry["Get"]["Key"])["SK"]
        for entry in read_calls[0]["TransactItems"]
    ] == [
        "ACTIVATION_INDEX",
        "ACTIVATION#activation-1#ROLLOVER",
        "ACTIVATION#activation-1#CONTROL",
        "ACTIVATION#activation-1#RECOVERY_CONTROL",
        "ACTIVATION#activation-1#FINALIZATION_CONTROL",
        "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL",
    ]


def test_execution_allocation_atomically_updates_control_and_puts_start_owned_execution() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    control_key = "ACTIVATION#activation-1#CONTROL"
    before_control = _control()
    client.install(control_key, before_control)
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")
    raw_nonce = b"e" * 32
    after_control = _control(
        active_epoch=2,
        active_execution_arn="execution-arn-2",
        active_state_machine_version_arn="state-machine-version-arn-2",
        revision=2,
    )
    execution_key = "ACTIVATION#activation-1#EXECUTION#00000002"
    execution = _closed_record(
        "glm52_production_execution",
        activation_id="activation-1",
        epoch=2,
        epoch_text="00000002",
        expected_execution_arn="execution-arn-2",
        expected_state_machine_version_arn="state-machine-version-arn-2",
        owner_control_revision=2,
    )

    result = adapter.allocate_execution_epoch(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        control=ExactUpdate(
            LedgerKey(RUN_ID, control_key), before_control, after_control
        ),
        execution=ExactPut(
            LedgerKey(RUN_ID, execution_key), execution
        ),
        domain="EXECUTION_ALLOCATION",
        operation_identity_sha256="c" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "Update",
        "Put",
    ]
    assert result.records[2]["state"] == "START_OWNED"
    assert result.records[2]["start_owner_invocation_nonce_sha256"] == (
        hashlib.sha256(raw_nonce).hexdigest()
    )


def test_owner_takeover_changes_only_owner_attempt_expiry_revision_and_updated_at() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    owner_key = "ACTIVATION#activation-1#RECOVERY_CONTROL"
    before = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
        state="OWNED",
    )
    client.install(owner_key, before)
    after = dict(before)
    after.update(
        owner_attempt=2,
        owner_execution_arn="new-owner-execution",
        owner_state_machine_version_arn="new-owner-version",
        owner_dispatch_identity_sha256="b" * 64,
        owner_invocation_nonce_sha256="c" * 64,
        owner_hard_expires_at="2026-07-28T13:00:00Z",
        revision=2,
        updated_at="2026-07-28T12:00:01Z",
    )
    raw_nonce = b"o" * 32

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_owner_takeover(
        guards=(
            ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
                _activation_index(),
            ),
        ),
        owner=ExactUpdate(LedgerKey(RUN_ID, owner_key), before, after),
        domain="RECOVERY_OWNER_TAKEOVER",
        operation_identity_sha256="d" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records[-1]["owner_attempt"] == 2
    assert result.records[-1]["owner_invocation_nonce_sha256"] == (
        hashlib.sha256(raw_nonce).hexdigest()
    )


def test_action_arm_is_control_update_plus_absent_action_put() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    control_key = "ACTIVATION#activation-1#CONTROL"
    action_key = (
        "ACTIVATION#activation-1#ACTION#00000001#S3_CREATE#00000001"
    )
    before_control = _control()
    client.install(control_key, before_control)
    raw_nonce = b"a" * 32

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).arm_action(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        control=ExactUpdate(
            LedgerKey(RUN_ID, control_key),
            before_control,
            _control(
                last_sky_post_action_key=action_key,
                last_sky_post_state="ARMED",
                revision=2,
            ),
        ),
        action=ExactPut(
            LedgerKey(RUN_ID, action_key),
            _closed_record(
                "glm52_production_action",
                activation_id="activation-1",
                action_kind="S3_CREATE",
            ),
        ),
        domain="ACTION_ARM",
        operation_identity_sha256="e" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "Update",
        "Put",
    ]


def test_action_consume_is_index_check_plus_exactly_two_conditional_updates() -> None:
    arguments, installed = _consume_action_arguments(b"c" * 32)
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", installed["index"])
    client.install(arguments["control"].key.sort_key, installed["control_before"])
    client.install(arguments["action"].key.sort_key, installed["action_before"])

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).consume_action(**arguments)

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "Update",
        "Update",
    ]


def test_action_consume_request_contains_no_control_condition_check() -> None:
    arguments, installed = _consume_action_arguments(b"d" * 32)
    arguments["operation_identity_sha256"] = "1" * 64
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", installed["index"])
    client.install(arguments["control"].key.sort_key, installed["control_before"])
    client.install(arguments["action"].key.sort_key, installed["action_before"])
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")

    adapter.consume_action(**arguments)

    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    checks = [
        decode_item(item["ConditionCheck"]["Key"])["SK"]
        for item in write["TransactItems"]
        if "ConditionCheck" in item
    ]
    assert checks == ["ACTIVATION_INDEX"]


def test_worker_launch_prepared_put_uses_absent_condition_and_consistent_readback() -> None:
    client = AwsShapedSimulator()
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")
    launch_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    )

    result = adapter.put_worker_launch_prepared(
        worker_launch=ExactPut(
            LedgerKey(RUN_ID, launch_key),
            _closed_record(
                "glm52_production_worker_launch",
                activation_id="activation-1",
            ),
        ),
        domain="WORKER_LAUNCH_PREPARED",
        operation_identity_sha256="2" * 64,
        raw_owner_nonce=b"w" * 32,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert [
        name for name, _ in client.calls
    ] == ["transact_write_items", "get_item"]
    put = client.calls[0][1]["TransactItems"][0]["Put"]
    assert put["ConditionExpression"] == (
        "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
    )


def test_worker_launch_committed_wal_binding_is_a_closed_owned_revision() -> None:
    client = AwsShapedSimulator()
    launch_key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    raw_nonce = b"w" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        owner_invocation_nonce_sha256=nonce_sha,
        ddb_committed_journal_entry_sha256=None,
    )
    after = deepcopy(before)
    after.update(
        ddb_committed_journal_entry_sha256="d" * 64,
        revision=before["revision"] + 1,
        updated_at="2026-07-28T12:00:01Z",
    )
    client.install(launch_key, before)

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).bind_worker_launch_committed_wal(
        worker_launch=ExactUpdate(
            LedgerKey(RUN_ID, launch_key), before, after
        ),
        domain="WORKER_LAUNCH_WAL_BIND",
        operation_identity_sha256="2" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records == (after,)
    assert [name for name, _ in client.calls] == [
        "transact_write_items",
        "get_item",
    ]


def test_liability_evidence_revision_uses_current_index_and_owner_nonce() -> None:
    client = AwsShapedSimulator()
    index = _activation_index()
    client.install("ACTIVATION_INDEX", index)
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    raw_nonce = b"l" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    after = deepcopy(before)
    after.update(
        scan_count_current_approval_period=(
            before["scan_count_current_approval_period"] + 1
        ),
        last_scan_started_at="2026-07-28T12:00:01Z",
        last_scan_completed_at="2026-07-28T12:00:02Z",
        last_scan_evidence_sha256="e" * 64,
        next_scan_at="2026-07-28T12:01:00Z",
        revision=before["revision"] + 1,
        updated_at="2026-07-28T12:00:02Z",
    )
    client.install(liability_key, before)

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).record_liability_evidence(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
        ),
        liability=ExactUpdate(
            LedgerKey(RUN_ID, liability_key), before, after
        ),
        domain="LIABILITY_SCAN_EVIDENCE",
        operation_identity_sha256="3" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records == (index, after)
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "Update",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("same_token_completion_attempts", 6),
        ("observed_instance_ids", ["i-0123456789abcdef0"]),
        ("late_instance_drain_identities", ["a" * 64]),
        ("late_instance_termination_action_identities", ["b" * 64]),
        ("late_instance_termination_call_counts", [6]),
        ("post_terminal_allocation_identities", ["c" * 64]),
        ("spend_close_identities", ["d" * 64]),
        ("gpu_liability_reserve_release_identity_sha256", "e" * 64),
        ("settlement_identity_sha256", "f" * 64),
        ("ec2_client_token", "z" * 64),
        ("launch_parameters_sha256", "9" * 64),
        ("owner_execution_arn", "foreign-owner"),
    ],
)
def test_liability_scan_evidence_rejects_sensitive_or_foreign_deltas_before_sdk(
    field: str,
    value: object,
) -> None:
    client = AwsShapedSimulator()
    index = _activation_index()
    client.install("ACTIVATION_INDEX", index)
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    raw_nonce = b"l" * 32
    before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        owner_invocation_nonce_sha256=hashlib.sha256(raw_nonce).hexdigest(),
    )
    after = deepcopy(before)
    after.update(
        {
            field: value,
            "scan_count_current_approval_period": (
                before["scan_count_current_approval_period"] + 1
            ),
            "last_scan_started_at": "2026-07-28T12:00:01Z",
            "last_scan_completed_at": "2026-07-28T12:00:02Z",
            "last_scan_evidence_sha256": "e" * 64,
            "next_scan_at": "2026-07-28T12:01:00Z",
            "revision": before["revision"] + 1,
            "updated_at": "2026-07-28T12:00:02Z",
        }
    )
    with pytest.raises((TypeError, ValueError)):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).record_liability_evidence(
            index=ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
            ),
            liability=ExactUpdate(
                LedgerKey(RUN_ID, liability_key), before, after
            ),
            domain="LIABILITY_SCAN_EVIDENCE",
            operation_identity_sha256="3" * 64,
            raw_owner_nonce=raw_nonce,
        )
    assert not client.calls


def test_activation_family_query_exhausts_pages_with_exact_validated_records() -> None:
    client = AwsShapedSimulator()
    first_key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    second_key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000002"
    first = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
    )
    second = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        allocation_ordinal=2,
        allocation_ordinal_text="00000002",
    )
    last_key = encode_item({"PK": "RUN#" + RUN_ID, "SK": first_key})
    client.query_pages = [
        {
            "Items": [
                encode_item(
                    {"PK": "RUN#" + RUN_ID, "SK": first_key, **first}
                )
            ],
            "LastEvaluatedKey": last_key,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "query-1",
            },
        },
        {
            "Items": [
                encode_item(
                    {"PK": "RUN#" + RUN_ID, "SK": second_key, **second}
                )
            ],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "query-2",
            },
        },
    ]

    records = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).query_activation_family(
        run_id=RUN_ID,
        sort_key_prefix="ACTIVATION#activation-1#WORKER_LAUNCH#",
        record_type="glm52_production_worker_launch",
    )

    assert records == (first, second)
    queries = [
        request for name, request in client.calls if name == "query"
    ]
    assert len(queries) == 2
    assert "ExclusiveStartKey" not in queries[0]
    assert queries[1]["ExclusiveStartKey"] == last_key
    assert all(request["ConsistentRead"] is True for request in queries)


@pytest.mark.parametrize(
    "prefix",
    [
        "ACTIVATION##WORKER_LAUNCH#",
        "ACTIVATION#activation-1#FOREIGN#WORKER_LAUNCH#",
        "ACTIVATION#activation-1#WORKER_LAUNCH#SUFFIX#",
        "ACTIVATION#activation#foreign#WORKER_LAUNCH#",
    ],
)
def test_activation_family_query_rejects_nonexact_prefix_grammar_before_sdk(
    prefix: str,
) -> None:
    client = AwsShapedSimulator()
    with pytest.raises(ValueError, match="closed"):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).query_activation_family(
            run_id=RUN_ID,
            sort_key_prefix=prefix,
            record_type="glm52_production_worker_launch",
        )
    assert not client.calls


@pytest.mark.parametrize(
    ("record_type", "family"),
    (
        ("glm52_production_worker_launch", "WORKER_LAUNCH"),
        (
            "glm52_production_worker_launch_liability",
            "WORKER_LAUNCH_LIABILITY",
        ),
        (
            "glm52_production_worker_launch_liability_action",
            "WORKER_LAUNCH_LIABILITY_ACTION",
        ),
        (
            "glm52_production_worker_launch_liability_settlement",
            "WORKER_LAUNCH_LIABILITY_SETTLEMENT",
        ),
        (
            "glm52_production_post_terminal_allocation",
            "POST_TERMINAL_ALLOCATION",
        ),
    ),
)
def test_activation_family_query_exact_record_family_cross_product(
    record_type: str,
    family: str,
) -> None:
    families = (
        "WORKER_LAUNCH",
        "WORKER_LAUNCH_LIABILITY",
        "WORKER_LAUNCH_LIABILITY_ACTION",
        "WORKER_LAUNCH_LIABILITY_SETTLEMENT",
        "POST_TERMINAL_ALLOCATION",
    )
    for candidate in families:
        client = AwsShapedSimulator()
        prefix = f"ACTIVATION#activation-1#{candidate}#"
        adapter = DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        )
        if candidate == family:
            client.query_pages = [
                {
                    "Items": [],
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "query-empty",
                    },
                }
            ]
            assert adapter.query_activation_family(
                run_id=RUN_ID,
                sort_key_prefix=prefix,
                record_type=record_type,
            ) == ()
            assert [name for name, _ in client.calls] == ["query"]
        else:
            with pytest.raises(ValueError, match="closed"):
                adapter.query_activation_family(
                    run_id=RUN_ID,
                    sort_key_prefix=prefix,
                    record_type=record_type,
                )
            assert not client.calls


def test_worker_completion_result_is_one_closed_three_record_transaction() -> None:
    client = AwsShapedSimulator()
    index = _activation_index()
    client.install("ACTIVATION_INDEX", index)
    raw_nonce = b"c" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    owner = {
        "owner_attempt": 1,
        "owner_execution_arn": "liability-owner-execution",
        "owner_state_machine_version_arn": "liability-owner-version",
        "owner_dispatch_identity_sha256": "d" * 64,
        "owner_invocation_nonce_sha256": nonce_sha,
        "owner_hard_expires_at": TS,
    }
    launch_key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    launch_before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="POSSIBLY_SENT",
        revision=2,
        owner_invocation_nonce_sha256=nonce_sha,
        same_token_completion_count=1,
    )
    launch_after = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="INSTANCE_OBSERVED",
        revision=3,
        owner_invocation_nonce_sha256=nonce_sha,
        same_token_completion_count=1,
        run_instances_attempt_evidence=["e" * 64],
        observed_instance_ids=["i-0123456789abcdef0"],
        instance_observation_sha256="f" * 64,
    )
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    liability_before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="SAME_TOKEN_COMPLETION",
        **owner,
    )
    liability_after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        revision=2,
        **owner,
    )
    action_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY_ACTION#"
        "00000001#SAME_TOKEN_COMPLETE#00000001"
    )
    action_identity = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "allocation_ordinal": 1,
        "allocation_ordinal_text": "00000001",
        "worker_launch_identity_sha256": liability_before[
            "worker_launch_identity_sha256"
        ],
        "worker_launch_liability_identity_sha256": canonical_sha256(
            liability_before
        ),
        "action_kind": "SAME_TOKEN_COMPLETE",
        **owner,
    }
    action_before = _closed_record(
        "glm52_production_worker_launch_liability_action",
        state="CONSUMED",
        revision=2,
        **action_identity,
    )
    action_after = _closed_record(
        "glm52_production_worker_launch_liability_action",
        state="COMPLETED",
        revision=3,
        response_identity_sha256="e" * 64,
        **action_identity,
    )
    client.install(launch_key, launch_before)
    client.install(liability_key, liability_before)
    client.install(action_key, action_before)

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).complete_worker_launch_attempt(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
        ),
        worker_launch=ExactUpdate(
            LedgerKey(RUN_ID, launch_key), launch_before, launch_after
        ),
        liability=ExactUpdate(
            LedgerKey(RUN_ID, liability_key),
            liability_before,
            liability_after,
        ),
        liability_action=ExactUpdate(
            LedgerKey(RUN_ID, action_key), action_before, action_after
        ),
        classification="DIRECT_SUCCESS",
        domain="WORKER_LAUNCH_ATTEMPT_RESULT",
        operation_identity_sha256="4" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records == (
        index,
        launch_after,
        liability_after,
        action_after,
    )


def _unrelated_completion_mutant(
    raw_nonce: bytes,
) -> dict[str, object]:
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    owner = {
        "owner_attempt": 1,
        "owner_execution_arn": "liability-owner-execution",
        "owner_state_machine_version_arn": "liability-owner-version",
        "owner_dispatch_identity_sha256": "d" * 64,
        "owner_invocation_nonce_sha256": nonce_sha,
        "owner_hard_expires_at": TS,
    }
    launch_before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="POSSIBLY_SENT",
        revision=2,
        owner_invocation_nonce_sha256=nonce_sha,
        same_token_completion_count=1,
    )
    launch_after = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="INSTANCE_OBSERVED",
        revision=3,
        owner_invocation_nonce_sha256=nonce_sha,
        same_token_completion_count=1,
        run_instances_attempt_evidence=["e" * 64],
        observed_instance_ids=["i-0123456789abcdef0"],
        instance_observation_sha256="f" * 64,
    )
    liability_before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="SAME_TOKEN_COMPLETION",
        **owner,
    )
    liability_after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        revision=2,
        spend_close_identities=["c" * 64],
        **owner,
    )
    action_identity = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "allocation_ordinal": 1,
        "allocation_ordinal_text": "00000001",
        "worker_launch_identity_sha256": liability_before[
            "worker_launch_identity_sha256"
        ],
        "worker_launch_liability_identity_sha256": canonical_sha256(
            liability_before
        ),
        "action_kind": "SAME_TOKEN_COMPLETE",
        **owner,
    }
    action_before = _closed_record(
        "glm52_production_worker_launch_liability_action",
        state="CONSUMED",
        revision=2,
        **action_identity,
    )
    action_after = _closed_record(
        "glm52_production_worker_launch_liability_action",
        state="COMPLETED",
        revision=3,
        response_identity_sha256="e" * 64,
        **action_identity,
    )
    return {
        "index": ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        "worker_launch": ExactUpdate(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#WORKER_LAUNCH#00000001",
            ),
            launch_before,
            launch_after,
        ),
        "liability": ExactUpdate(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001",
            ),
            liability_before,
            liability_after,
        ),
        "liability_action": ExactUpdate(
            LedgerKey(
                RUN_ID,
                "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY_ACTION#"
                "00000001#SAME_TOKEN_COMPLETE#00000001",
            ),
            action_before,
            action_after,
        ),
        "classification": "DIRECT_SUCCESS",
        "domain": "WORKER_LAUNCH_ATTEMPT_RESULT",
        "operation_identity_sha256": "4" * 64,
        "raw_owner_nonce": raw_nonce,
    }


def test_worker_completion_rejects_unrelated_liability_delta_before_sdk() -> None:
    client = AwsShapedSimulator()
    with pytest.raises((TypeError, ValueError)):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).complete_worker_launch_attempt(
            **_unrelated_completion_mutant(b"c" * 32)
        )
    assert not client.calls


def test_liability_incident_is_a_closed_owned_state_transition() -> None:
    client = AwsShapedSimulator()
    index = _activation_index()
    client.install("ACTIVATION_INDEX", index)
    raw_nonce = b"i" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    key = "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        owner_attempt=1,
        owner_execution_arn="owner-execution",
        owner_state_machine_version_arn="owner-version",
        owner_dispatch_identity_sha256="d" * 64,
        owner_invocation_nonce_sha256=nonce_sha,
        owner_hard_expires_at=TS,
    )
    after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="LIABILITY_INCIDENT",
        revision=2,
        incident_identity_sha256="e" * 64,
        incident_at=TS,
        owner_attempt=1,
        owner_execution_arn="owner-execution",
        owner_state_machine_version_arn="owner-version",
        owner_dispatch_identity_sha256="d" * 64,
        owner_invocation_nonce_sha256=nonce_sha,
        owner_hard_expires_at=TS,
    )
    client.install(key, before)

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).record_liability_incident(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
        ),
        liability=ExactUpdate(LedgerKey(RUN_ID, key), before, after),
        domain="LIABILITY_INCIDENT",
        operation_identity_sha256="5" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records == (index, after)


def test_liability_incident_rejects_unrelated_evidence_delta_before_sdk() -> None:
    client = AwsShapedSimulator()
    raw_nonce = b"i" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        owner_attempt=1,
        owner_execution_arn="owner-execution",
        owner_state_machine_version_arn="owner-version",
        owner_dispatch_identity_sha256="d" * 64,
        owner_invocation_nonce_sha256=nonce_sha,
        owner_hard_expires_at=TS,
    )
    after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="LIABILITY_INCIDENT",
        revision=2,
        incident_identity_sha256="e" * 64,
        incident_at=TS,
        spend_close_identities=["f" * 64],
        owner_attempt=1,
        owner_execution_arn="owner-execution",
        owner_state_machine_version_arn="owner-version",
        owner_dispatch_identity_sha256="d" * 64,
        owner_invocation_nonce_sha256=nonce_sha,
        owner_hard_expires_at=TS,
    )
    with pytest.raises((TypeError, ValueError)):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).record_liability_incident(
            index=ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
            ),
            liability=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#"
                    "WORKER_LAUNCH_LIABILITY#00000001",
                ),
                before,
                after,
            ),
            domain="LIABILITY_INCIDENT",
            operation_identity_sha256="5" * 64,
            raw_owner_nonce=raw_nonce,
        )
    assert not client.calls


def test_worker_reconciliation_is_a_closed_owned_state_transition() -> None:
    client = AwsShapedSimulator()
    index = _activation_index()
    client.install("ACTIVATION_INDEX", index)
    raw_nonce = b"r" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="INSTANCE_OBSERVED",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    after = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="ALLOCATION_OPEN",
        revision=2,
        owner_invocation_nonce_sha256=nonce_sha,
        spend_allocation_open_identity_sha256="e" * 64,
    )
    client.install(key, before)

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).record_worker_launch_reconciliation(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
        ),
        worker_launch=ExactUpdate(
            LedgerKey(RUN_ID, key), before, after
        ),
        domain="WORKER_LAUNCH_RECONCILIATION",
        operation_identity_sha256="6" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records == (index, after)


def test_worker_reconciliation_rejects_unrelated_attempt_delta_before_sdk() -> None:
    client = AwsShapedSimulator()
    raw_nonce = b"r" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="INSTANCE_OBSERVED",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    after = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="ALLOCATION_OPEN",
        revision=2,
        owner_invocation_nonce_sha256=nonce_sha,
        spend_allocation_open_identity_sha256="e" * 64,
        run_instances_attempt_evidence=["f" * 64],
    )
    with pytest.raises((TypeError, ValueError)):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).record_worker_launch_reconciliation(
            index=ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
            ),
            worker_launch=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#WORKER_LAUNCH#00000001",
                ),
                before,
                after,
            ),
            domain="WORKER_LAUNCH_RECONCILIATION",
            operation_identity_sha256="6" * 64,
            raw_owner_nonce=raw_nonce,
        )
    assert not client.calls


def _task9_narrow_write_case(
    api: str,
) -> tuple[str, dict[str, object]]:
    raw_nonce = (api[0].encode("ascii") * 32)[:32]
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    index = _activation_index()
    if api == "wal":
        before = _closed_record(
            "glm52_production_worker_launch",
            activation_id="activation-1",
            owner_invocation_nonce_sha256=nonce_sha,
            ddb_committed_journal_entry_sha256=None,
        )
        after = deepcopy(before)
        after.update(
            ddb_committed_journal_entry_sha256="d" * 64,
            revision=before["revision"] + 1,
            updated_at="2026-07-28T12:00:01Z",
        )
        return "bind_worker_launch_committed_wal", {
            "worker_launch": ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#WORKER_LAUNCH#00000001",
                ),
                before,
                after,
            ),
            "domain": "WORKER_LAUNCH_WAL_BIND",
            "operation_identity_sha256": "2" * 64,
            "raw_owner_nonce": raw_nonce,
        }
    if api == "scan":
        before = _closed_record(
            "glm52_production_worker_launch_liability",
            activation_id="activation-1",
            state="WATCHING",
            owner_invocation_nonce_sha256=nonce_sha,
        )
        after = deepcopy(before)
        after.update(
            scan_count_current_approval_period=(
                before["scan_count_current_approval_period"] + 1
            ),
            last_scan_started_at="2026-07-28T12:00:01Z",
            last_scan_completed_at="2026-07-28T12:00:02Z",
            last_scan_evidence_sha256="e" * 64,
            next_scan_at="2026-07-28T12:01:00Z",
            revision=before["revision"] + 1,
            updated_at="2026-07-28T12:00:02Z",
        )
        return "record_liability_evidence", {
            "index": ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
            ),
            "liability": ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#"
                    "WORKER_LAUNCH_LIABILITY#00000001",
                ),
                before,
                after,
            ),
            "domain": "LIABILITY_SCAN_EVIDENCE",
            "operation_identity_sha256": "3" * 64,
            "raw_owner_nonce": raw_nonce,
        }
    if api == "completion":
        arguments = _unrelated_completion_mutant(raw_nonce)
        liability = arguments["liability"]
        assert type(liability) is ExactUpdate
        liability_after = deepcopy(liability.after)
        liability_after["spend_close_identities"] = []
        arguments["liability"] = ExactUpdate(
            liability.key,
            liability.before,
            _rehash(liability_after),
        )
        return "complete_worker_launch_attempt", arguments
    if api == "incident":
        owner = {
            "owner_attempt": 1,
            "owner_execution_arn": "owner-execution",
            "owner_state_machine_version_arn": "owner-version",
            "owner_dispatch_identity_sha256": "d" * 64,
            "owner_invocation_nonce_sha256": nonce_sha,
            "owner_hard_expires_at": TS,
        }
        before = _closed_record(
            "glm52_production_worker_launch_liability",
            activation_id="activation-1",
            state="WATCHING",
            **owner,
        )
        after = _closed_record(
            "glm52_production_worker_launch_liability",
            activation_id="activation-1",
            state="LIABILITY_INCIDENT",
            revision=2,
            incident_identity_sha256="e" * 64,
            incident_at=TS,
            **owner,
        )
        return "record_liability_incident", {
            "index": ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
            ),
            "liability": ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#"
                    "WORKER_LAUNCH_LIABILITY#00000001",
                ),
                before,
                after,
            ),
            "domain": "LIABILITY_INCIDENT",
            "operation_identity_sha256": "5" * 64,
            "raw_owner_nonce": raw_nonce,
        }
    if api == "reconciliation":
        before = _closed_record(
            "glm52_production_worker_launch",
            activation_id="activation-1",
            state="INSTANCE_OBSERVED",
            owner_invocation_nonce_sha256=nonce_sha,
        )
        after = _closed_record(
            "glm52_production_worker_launch",
            activation_id="activation-1",
            state="ALLOCATION_OPEN",
            revision=2,
            owner_invocation_nonce_sha256=nonce_sha,
            spend_allocation_open_identity_sha256="e" * 64,
        )
        return "record_worker_launch_reconciliation", {
            "index": ExactCheck(
                LedgerKey(RUN_ID, "ACTIVATION_INDEX"), index
            ),
            "worker_launch": ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    "ACTIVATION#activation-1#WORKER_LAUNCH#00000001",
                ),
                before,
                after,
            ),
            "domain": "WORKER_LAUNCH_RECONCILIATION",
            "operation_identity_sha256": "6" * 64,
            "raw_owner_nonce": raw_nonce,
        }
    raise AssertionError("unknown Task 9 narrow write API")


def _install_task9_narrow_case(
    client: AwsShapedSimulator,
    arguments: dict[str, object],
    *,
    post: bool,
) -> None:
    for value in arguments.values():
        if type(value) is ExactCheck:
            client.install(value.key.sort_key, dict(value.expected))
        elif type(value) is ExactUpdate:
            record = value.after if post else value.before
            client.install(value.key.sort_key, dict(record))


@pytest.mark.parametrize(
    "api",
    ("wal", "scan", "completion", "incident", "reconciliation"),
)
def test_task9_narrow_writes_reject_foreign_owner_before_sdk(api: str) -> None:
    method_name, arguments = _task9_narrow_write_case(api)
    arguments["raw_owner_nonce"] = b"x" * 32
    client = AwsShapedSimulator()
    with pytest.raises(ValueError, match="owner|nonce"):
        getattr(
            DynamoLedgerAdapter(
                client=client, table_name="ledger-table"
            ),
            method_name,
        )(**arguments)
    assert not client.calls


@pytest.mark.parametrize(
    "api",
    ("wal", "scan", "completion", "incident", "reconciliation"),
)
def test_task9_narrow_writes_do_not_adopt_conditional_failure(
    api: str,
) -> None:
    method_name, arguments = _task9_narrow_write_case(api)
    client = AwsShapedSimulator()
    client.pre_write_fault = SimulatedClientError(
        "TransactionCanceledException",
        cancellation_reasons=({"Code": "ConditionalCheckFailed"},),
    )
    result = getattr(
        DynamoLedgerAdapter(client=client, table_name="ledger-table"),
        method_name,
    )(**arguments)
    assert result.outcome is WriteOutcome.CONDITION_REJECTED
    assert [name for name, _ in client.calls] == [
        "transact_write_items"
    ]


@pytest.mark.parametrize(
    "api",
    ("wal", "scan", "completion", "incident", "reconciliation"),
)
def test_task9_narrow_writes_adopt_only_exact_live_owner_after_lost_response(
    api: str,
) -> None:
    method_name, arguments = _task9_narrow_write_case(api)
    client = AwsShapedSimulator()
    _install_task9_narrow_case(client, arguments, post=True)
    client.pre_write_fault = TimeoutError("opaque response loss")
    result = getattr(
        DynamoLedgerAdapter(client=client, table_name="ledger-table"),
        method_name,
    )(**arguments)
    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert [name for name, _ in client.calls] in (
        ["transact_write_items", "get_item"],
        ["transact_write_items", "transact_get_items"],
    )


def _different_unrelated_value(field: str, value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is list:
        if field == "late_instance_termination_call_counts":
            return [*value, 6]
        if field == "observed_instance_ids":
            return [*value, "i-0123456789abcdef0"]
        return [*value, "f" * 64]
    if type(value) is dict:
        return {"identity": "f" * 64}
    if value is None:
        return "f" * 64 if field.endswith("_sha256") else "foreign"
    if type(value) is str:
        if field.endswith("_sha256"):
            return ("e" if value != "e" * 64 else "f") * 64
        if field.endswith("_at"):
            return "2026-07-28T12:00:59Z"
        return value + "-foreign"
    raise AssertionError("unsupported closed-record field type")


@pytest.mark.parametrize(
    ("api", "plan_name", "allowed"),
    (
        (
            "wal",
            "worker_launch",
            {
                "ddb_committed_journal_entry_sha256",
                "revision",
                "updated_at",
            },
        ),
        (
            "scan",
            "liability",
            {
                "scan_count_current_approval_period",
                "scan_period_started_at",
                "scan_period_ends_at",
                "next_scan_at",
                "last_scan_started_at",
                "last_scan_completed_at",
                "last_scan_evidence_sha256",
                "revision",
                "updated_at",
            },
        ),
        (
            "completion",
            "worker_launch",
            {
                "state",
                "run_instances_attempt_evidence",
                "observed_instance_ids",
                "instance_observation_sha256",
                "revision",
                "updated_at",
            },
        ),
        (
            "completion",
            "liability",
            {"state", "revision", "updated_at"},
        ),
        (
            "completion",
            "liability_action",
            {
                "state",
                "completed_at",
                "response_identity_sha256",
                "revision",
                "updated_at",
            },
        ),
        (
            "incident",
            "liability",
            {
                "state",
                "incident_identity_sha256",
                "incident_at",
                "revision",
                "updated_at",
            },
        ),
        (
            "reconciliation",
            "worker_launch",
            {
                "state",
                "spend_allocation_open_identity_sha256",
                "revision",
                "updated_at",
            },
        ),
    ),
)
def test_task9_narrow_write_unrelated_field_matrix(
    api: str,
    plan_name: str,
    allowed: set[str],
) -> None:
    _, baseline_arguments = _task9_narrow_write_case(api)
    baseline = baseline_arguments[plan_name]
    assert type(baseline) is ExactUpdate
    unrelated = sorted(set(baseline.after) - allowed)
    assert unrelated
    for field in unrelated:
        method_name, arguments = _task9_narrow_write_case(api)
        plan = arguments[plan_name]
        assert type(plan) is ExactUpdate
        mutant = deepcopy(plan.after)
        mutant[field] = _different_unrelated_value(
            field, mutant[field]
        )
        arguments[plan_name] = ExactUpdate(
            plan.key,
            plan.before,
            _rehash(mutant),
        )
        client = AwsShapedSimulator()
        with pytest.raises((TypeError, ValueError)):
            getattr(
                DynamoLedgerAdapter(
                    client=client, table_name="ledger-table"
                ),
                method_name,
            )(**arguments)
        assert not client.calls, (api, plan_name, field)


def test_worker_possible_send_updates_launch_and_creates_unowned_liability_atomically() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    recovery_key = "ACTIVATION#activation-1#RECOVERY_CONTROL"
    finalization_key = "ACTIVATION#activation-1#FINALIZATION_CONTROL"
    recovery = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
    )
    finalization = _closed_record(
        "glm52_production_finalization_control",
        activation_id="activation-1",
    )
    client.install(recovery_key, recovery)
    client.install(finalization_key, finalization)
    launch_key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    raw_nonce = b"p" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    launch_before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    client.install(launch_key, launch_before)
    launch_after = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="POSSIBLY_SENT",
        revision=2,
        owner_invocation_nonce_sha256=nonce_sha,
    )
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    liability = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_worker_launch_possibly_sent(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        reserve_checks=(
            ExactCheck(LedgerKey(RUN_ID, recovery_key), recovery),
            ExactCheck(LedgerKey(RUN_ID, finalization_key), finalization),
        ),
        worker_launch=ExactUpdate(
            LedgerKey(RUN_ID, launch_key), launch_before, launch_after
        ),
        liability=ExactPut(
            LedgerKey(RUN_ID, liability_key), liability
        ),
        domain="WORKER_POSSIBLY_SENT",
        operation_identity_sha256="3" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "ConditionCheck",
        "ConditionCheck",
        "Update",
        "Put",
    ]
    assert result.records[1]["state"] == "POSSIBLY_SENT"
    assert result.records[2]["state"] == "UNOWNED_NOT_ACTIONABLE"


def test_worker_possible_send_binds_exact_900_second_13_76_reserve_and_approval() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    guard_key = "ACTIVATION#activation-1#RECOVERY_CONTROL"
    guard = _closed_record(
        "glm52_production_recovery_control",
        activation_id="activation-1",
    )
    client.install(guard_key, guard)
    launch_key = "ACTIVATION#activation-1#WORKER_LAUNCH#00000001"
    approval = "9" * 64
    reserve = "8" * 64
    raw_nonce = b"r" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    launch_before = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        residual_liability_approval_identity_sha256=approval,
        owner_invocation_nonce_sha256=nonce_sha,
    )
    client.install(launch_key, launch_before)
    launch_after = _closed_record(
        "glm52_production_worker_launch",
        activation_id="activation-1",
        state="POSSIBLY_SENT",
        revision=2,
        residual_liability_approval_identity_sha256=approval,
        gpu_liability_reserve_ledger_identity_sha256=reserve,
        owner_invocation_nonce_sha256=nonce_sha,
    )
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    liability = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        residual_liability_approval_identity_sha256=approval,
        gpu_liability_reserve_ledger_identity_sha256=reserve,
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_worker_launch_possibly_sent(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        reserve_checks=(
            ExactCheck(LedgerKey(RUN_ID, guard_key), guard),
        ),
        worker_launch=ExactUpdate(
            LedgerKey(RUN_ID, launch_key), launch_before, launch_after
        ),
        liability=ExactPut(
            LedgerKey(RUN_ID, liability_key), liability
        ),
        domain="WORKER_POSSIBLY_SENT",
        operation_identity_sha256="4" * 64,
        raw_owner_nonce=raw_nonce,
    )

    for record in result.records[1:]:
        assert record["gpu_liability_reserve_seconds"] == 900
        assert record["gpu_liability_reserve_cost_usd"] == "13.76"
        assert record["residual_liability_approval_identity_sha256"] == approval


def test_liability_first_owner_acquisition_installs_fresh_nonce_before_sender_call() -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
    )
    client.install(liability_key, before)
    after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        revision=2,
    )
    raw_nonce = b"l" * 32

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).acquire_liability_owner(
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        liability=ExactUpdate(
            LedgerKey(RUN_ID, liability_key), before, after
        ),
        domain="LIABILITY_OWNER_ACQUIRE",
        operation_identity_sha256="5" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.records[1]["state"] == "WATCHING"
    assert result.records[1]["owner_invocation_nonce_sha256"] == (
        hashlib.sha256(raw_nonce).hexdigest()
    )
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_write_items"
        ]
    ) == 1


@pytest.mark.parametrize(
    "stale_name",
    [
        "SAME_TOKEN_RUN_INSTANCES",
        "LATE_INSTANCE_TERMINATE",
        "POST_TERMINAL_ALLOCATION_CREATE",
        "SETTLE",
        "",
    ],
)
def test_liability_action_dispatch_rejects_stale_candidate_action_kind_names(
    stale_name: str,
) -> None:
    adapter = DynamoLedgerAdapter(
        client=AwsShapedSimulator(), table_name="ledger-table"
    )
    with pytest.raises(ValueError, match="stale|unknown"):
        adapter.consume_liability_action(
            action_name=stale_name,
            index=None,
            liability=None,
            action=None,
            post_terminal_allocation=None,
            domain="LIABILITY_ACTION",
            operation_identity_sha256="6" * 64,
            raw_owner_nonce=b"x" * 32,
        )


def test_post_terminal_discover_atomically_consumes_action_and_puts_allocation() -> None:
    arguments, installed = _liability_action_arguments(b"i" * 32)
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", installed["index"])
    client.install(
        arguments["liability"].key.sort_key,
        installed["liability_before"],
    )
    client.install(arguments["action"].key.sort_key, installed["action_before"])

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).consume_liability_action(**arguments)

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "Update",
        "Update",
        "Put",
    ]
    assert result.records[-1]["state"] == "DISCOVERED"


def test_liability_settlement_atomically_consumes_updates_and_puts_marker() -> None:
    arguments, installed = _settlement_arguments(b"s" * 32)
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", installed["index"])
    client.install(
        arguments["liability_action"].key.sort_key,
        installed["action_before"],
    )
    client.install(
        arguments["liability"].key.sort_key,
        installed["liability_before"],
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).settle_worker_launch_liability(**arguments)

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "ConditionCheck",
        "Update",
        "Update",
        "Put",
    ]
    assert result.records[1]["state"] == "CONSUMED"
    assert result.records[2]["state"] == "SETTLED_NO_INSTANCE_REJECTED"


def test_snapshot_delete_stage_atomically_consumes_sets_and_puts_transition() -> None:
    client = AwsShapedSimulator()
    raw_nonce = b"z" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    control_key = "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL"
    control_before = _closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id="activation-1",
        state="OWNED",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    control_after = _closed_record(
        "glm52_production_snapshot_cleanup_control",
        activation_id="activation-1",
        state="DELETE_POSSIBLY_SENT",
        owner_invocation_nonce_sha256=nonce_sha,
        revision=2,
    )
    client.install(control_key, control_before)
    action_key = "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_ACTION#00000001"
    action_before = _closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id="activation-1",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    action_after = _closed_record(
        "glm52_production_snapshot_cleanup_action",
        activation_id="activation-1",
        owner_invocation_nonce_sha256=nonce_sha,
        state="CONSUMED",
        revision=2,
    )
    client.install(action_key, action_before)
    transition_key = (
        "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_TRANSITION#00000002"
    )
    transition = _closed_record(
        "glm52_production_snapshot_cleanup_transition",
        activation_id="activation-1",
        from_state="OWNED",
        to_state="DELETE_POSSIBLY_SENT",
        from_revision=1,
        to_revision=2,
        owner_invocation_nonce_sha256=nonce_sha,
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).stage_snapshot_delete(
        cleanup_control=ExactUpdate(
            LedgerKey(RUN_ID, control_key), control_before, control_after
        ),
        cleanup_action=ExactUpdate(
            LedgerKey(RUN_ID, action_key), action_before, action_after
        ),
        cleanup_transition=ExactPut(
            LedgerKey(RUN_ID, transition_key), transition
        ),
        domain="SNAPSHOT_DELETE_STAGE",
        operation_identity_sha256="9" * 64,
        raw_owner_nonce=raw_nonce,
    )

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == [
        "Update",
        "Update",
        "Put",
    ]
    assert result.records[0]["state"] == "DELETE_POSSIBLY_SENT"
    assert result.records[1]["state"] == "CONSUMED"


def test_rollover_success_still_requires_six_record_coherent_readback() -> None:
    client = AwsShapedSimulator()
    client.transact_get_fault = TimeoutError("read unavailable")

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"q" * 32))

    assert result.outcome is WriteOutcome.READBACK_UNAVAILABLE
    assert result.may_issue_external_side_effect is False
    assert [name for name, _ in client.calls] == [
        "transact_write_items",
        "transact_get_items",
    ]


def test_rollover_timeout_after_commit_is_accepted_only_by_live_nonce_owner() -> None:
    client = AwsShapedSimulator()
    client.post_commit_fault = TimeoutError("write response lost")
    raw_nonce = b"v" * 32

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(raw_nonce))

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert result.error_code == "TimeoutError"
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_write_items"
        ]
    ) == 1


@pytest.mark.parametrize("mode", ["before", "partial", "foreign"])
def test_rollover_timeout_before_commit_partial_and_foreign_nonce_readback_fail_closed(
    mode: str,
) -> None:
    client = AwsShapedSimulator()
    raw_nonce = b"y" * 32
    if mode == "before":
        client.pre_write_fault = TimeoutError("write not committed")
    elif mode == "partial":
        client.post_commit_fault = TimeoutError("write response lost")

        def remove_one() -> None:
            client.items.pop(
                (
                    "RUN#" + RUN_ID,
                    "ACTIVATION#activation-1#FINALIZATION_CONTROL",
                )
            )

        client.before_read_hook = remove_one
    else:
        client.post_commit_fault = TimeoutError("write response lost")

        def replace_nonce() -> None:
            key = (
                "RUN#" + RUN_ID,
                "ACTIVATION#activation-1#ROLLOVER",
            )
            physical = decode_item(client.items[key])
            physical["writer_invocation_nonce_sha256"] = "f" * 64
            logical = dict(physical)
            logical.pop("PK")
            logical.pop("SK")
            logical = _rehash(logical)
            client.items[key] = encode_item(
                {
                    "PK": physical["PK"],
                    "SK": physical["SK"],
                    **logical,
                }
            )

        client.before_read_hook = replace_nonce

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(raw_nonce))

    expected = {
        "before": WriteOutcome.PRESTATE_OR_ABSENT,
        "partial": WriteOutcome.MISMATCHED_COMMIT,
        "foreign": WriteOutcome.FOREIGN_NONCE,
    }[mode]
    assert result.outcome is expected
    assert result.may_issue_external_side_effect is False
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_write_items"
        ]
    ) == 1


def test_rollover_duplicate_delivery_adopts_exact_durable_records_without_write() -> None:
    raw_nonce = b"u" * 32
    operation_identity = "b" * 64
    live_client = AwsShapedSimulator()
    live = DynamoLedgerAdapter(
        client=live_client, table_name="ledger-table"
    ).commit_rollover(
        **_rollover_arguments(raw_nonce, operation_identity)
    )
    duplicate_client = AwsShapedSimulator()
    keys = [
        "ACTIVATION_INDEX",
        "ACTIVATION#activation-1#ROLLOVER",
        "ACTIVATION#activation-1#CONTROL",
        "ACTIVATION#activation-1#RECOVERY_CONTROL",
        "ACTIVATION#activation-1#FINALIZATION_CONTROL",
        "ACTIVATION#activation-1#SNAPSHOT_CLEANUP_CONTROL",
    ]
    for key, record in zip(keys, live.records):
        duplicate_client.install(key, dict(record))
    plans = _rollover_arguments(raw_nonce, operation_identity)
    plans.update(
        index=ExactUpdate(
            plans["index"].key, {}, dict(live.records[0])
        ),
        rollover=ExactPut(plans["rollover"].key, dict(live.records[1])),
        control=ExactPut(plans["control"].key, dict(live.records[2])),
        recovery_control=ExactPut(
            plans["recovery_control"].key, dict(live.records[3])
        ),
        finalization_control=ExactPut(
            plans["finalization_control"].key, dict(live.records[4])
        ),
        snapshot_cleanup_control=ExactPut(
            plans["snapshot_cleanup_control"].key, dict(live.records[5])
        ),
        raw_owner_nonce=None,
        duplicate_operation_identity_sha256=operation_identity,
    )

    adopted = DynamoLedgerAdapter(
        client=duplicate_client, table_name="ledger-table"
    ).commit_rollover(**plans)

    assert adopted.outcome is WriteOutcome.EXACT_DURABLE_ADOPTION
    assert adopted.may_issue_external_side_effect is False
    assert [name for name, _ in duplicate_client.calls] == [
        "transact_get_items"
    ]


def test_liability_settlement_duplicate_adopts_terminal_pair_without_write() -> None:
    raw_nonce = b"t" * 32
    live_plans, installed = _settlement_arguments(raw_nonce)
    live_client = AwsShapedSimulator()
    live_client.install("ACTIVATION_INDEX", installed["index"])
    live_client.install(
        live_plans["liability_action"].key.sort_key,
        installed["action_before"],
    )
    live_client.install(
        live_plans["liability"].key.sort_key,
        installed["liability_before"],
    )
    live = DynamoLedgerAdapter(
        client=live_client, table_name="ledger-table"
    ).settle_worker_launch_liability(**live_plans)
    duplicate_client = AwsShapedSimulator()
    for plan, record in zip(
        (
            live_plans["index"],
            live_plans["liability_action"],
            live_plans["liability"],
            live_plans["settlement"],
        ),
        live.records,
    ):
        duplicate_client.install(plan.key.sort_key, dict(record))
    duplicate_plans, duplicate_installed = _settlement_arguments(raw_nonce)
    duplicate_plans.update(
        index=ExactCheck(
            duplicate_plans["index"].key, dict(live.records[0])
        ),
        liability_action=ExactUpdate(
            duplicate_plans["liability_action"].key,
            duplicate_installed["action_before"],
            dict(live.records[1]),
        ),
        liability=ExactUpdate(
            duplicate_plans["liability"].key,
            duplicate_installed["liability_before"],
            dict(live.records[2]),
        ),
        settlement=ExactPut(
            duplicate_plans["settlement"].key,
            dict(live.records[3]),
        ),
        raw_owner_nonce=None,
        duplicate_settlement_identity_sha256=live.records[3][
            "canonical_body_sha256"
        ],
    )

    adopted = DynamoLedgerAdapter(
        client=duplicate_client, table_name="ledger-table"
    ).settle_worker_launch_liability(**duplicate_plans)

    assert adopted.outcome is WriteOutcome.EXACT_DURABLE_ADOPTION
    assert adopted.may_issue_external_side_effect is False
    assert [name for name, _ in duplicate_client.calls] == [
        "transact_get_items"
    ]


@pytest.mark.parametrize(
    "mutant", ["liability_identity", "rehashed_settlement"]
)
def test_liability_settlement_duplicate_rejects_mismatched_final_binding(
    mutant: str,
) -> None:
    raw_nonce = b"N" * 32
    live_plans, installed = _settlement_arguments(raw_nonce)
    live_client = AwsShapedSimulator()
    live_client.install("ACTIVATION_INDEX", installed["index"])
    live_client.install(
        live_plans["liability_action"].key.sort_key,
        installed["action_before"],
    )
    live_client.install(
        live_plans["liability"].key.sort_key,
        installed["liability_before"],
    )
    live = DynamoLedgerAdapter(
        client=live_client, table_name="ledger-table"
    ).settle_worker_launch_liability(**live_plans)
    records = [dict(record) for record in live.records]
    if mutant == "liability_identity":
        records[2]["settlement_identity_sha256"] = "c" * 64
    elif mutant == "rehashed_settlement":
        records[3]["prior_incident_identity_sha256"] = "c" * 64
        records[3] = _rehash(records[3])
    else:
        raise AssertionError("unhandled settlement duplicate mutant")
    duplicate_plans, duplicate_installed = _settlement_arguments(raw_nonce)
    duplicate_plans.update(
        index=ExactCheck(
            duplicate_plans["index"].key, records[0]
        ),
        liability_action=ExactUpdate(
            duplicate_plans["liability_action"].key,
            duplicate_installed["action_before"],
            records[1],
        ),
        liability=ExactUpdate(
            duplicate_plans["liability"].key,
            duplicate_installed["liability_before"],
            records[2],
        ),
        settlement=ExactPut(
            duplicate_plans["settlement"].key, records[3]
        ),
        raw_owner_nonce=None,
        duplicate_settlement_identity_sha256=records[3][
            "canonical_body_sha256"
        ],
    )
    duplicate_client = AwsShapedSimulator()
    for plan, record in zip(
        (
            duplicate_plans["index"],
            duplicate_plans["liability_action"],
            duplicate_plans["liability"],
            duplicate_plans["settlement"],
        ),
        records,
    ):
        duplicate_client.install(plan.key.sort_key, record)

    with pytest.raises(ValueError):
        DynamoLedgerAdapter(
            client=duplicate_client, table_name="ledger-table"
        ).settle_worker_launch_liability(**duplicate_plans)

    assert not duplicate_client.calls


def test_client_request_token_is_exactly_36_characters_and_changes_with_inputs() -> None:
    tokens = []
    transaction_hashes = []
    for raw_nonce, operation_identity in (
        (b"1" * 32, "a" * 64),
        (b"2" * 32, "a" * 64),
        (b"1" * 32, "b" * 64),
    ):
        client = AwsShapedSimulator()
        result = DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(
            **_rollover_arguments(raw_nonce, operation_identity)
        )
        write = next(
            request
            for name, request in client.calls
            if name == "transact_write_items"
        )
        tokens.append(write["ClientRequestToken"])
        transaction_hashes.append(
            result.records[1]["transaction_bytes_sha256"]
        )
        exact_digest = hashlib.sha256(
            b"glm52-ddb-crt-v1"
            + b"\0"
            + b"ACTIVATION_ROLLOVER"
            + b"\0"
            + operation_identity.encode("ascii")
            + b"\0"
            + result.records[1]["transaction_bytes_sha256"].encode("ascii")
            + b"\0"
            + raw_nonce
        ).hexdigest()
        assert write["ClientRequestToken"] == "h1g-" + exact_digest[:32]
        assert result.records[1][
            "transaction_client_request_token_sha256"
        ] == hashlib.sha256(
            write["ClientRequestToken"].encode("ascii")
        ).hexdigest()
        assert raw_nonce not in tuple(
            value
            for value in write.values()
            if type(value) is bytes
        )

    assert all(
        len(token) == 36
        and token.startswith("h1g-")
        and all(char in "0123456789abcdef" for char in token[4:])
        for token in tokens
    )
    assert len(set(tokens)) == 3
    assert len(set(transaction_hashes)) == 3


def test_same_token_different_transaction_is_terminal_parameter_mismatch() -> None:
    client = AwsShapedSimulator()
    client.pre_write_fault = SimulatedClientError(
        "IdempotentParameterMismatchException"
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"m" * 32))

    assert result.outcome is WriteOutcome.MISMATCHED_COMMIT
    assert result.error_code == "IdempotentParameterMismatchException"
    assert [name for name, _ in client.calls] == ["transact_write_items"]


@pytest.mark.parametrize(
    "mode",
    ["success", "timeout", "connection_loss", "malformed_response"],
)
def test_transaction_success_timeout_connection_loss_and_malformed_response_never_retry(
    mode: str,
) -> None:
    client = AwsShapedSimulator()
    if mode == "timeout":
        client.post_commit_fault = TimeoutError("response lost")
    elif mode == "connection_loss":
        client.post_commit_fault = ConnectionResetError("connection reset")
    elif mode == "malformed_response":
        client.malformed_write_response = True

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"k" * 32))

    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_write_items"
        ]
    ) == 1
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_get_items"
        ]
    ) == 1


def test_transaction_read_failure_after_sdk_success_grants_no_authority() -> None:
    client = AwsShapedSimulator()
    client.transact_get_fault = ConnectionError("read path closed")

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"j" * 32))

    assert result.outcome is WriteOutcome.READBACK_UNAVAILABLE
    assert result.may_issue_external_side_effect is False


def test_ambiguous_consumption_mutant_issues_zero_external_calls() -> None:
    client = AwsShapedSimulator()
    client.post_commit_fault = TimeoutError("write response lost")
    raw_nonce = b"h" * 32

    def install_foreign_owner() -> None:
        key = (
            "RUN#" + RUN_ID,
            "ACTIVATION#activation-1#ROLLOVER",
        )
        physical = decode_item(client.items[key])
        logical = dict(physical)
        logical.pop("PK")
        logical.pop("SK")
        logical["writer_invocation_nonce_sha256"] = "0" * 64
        logical = _rehash(logical)
        client.items[key] = encode_item(
            {"PK": physical["PK"], "SK": physical["SK"], **logical}
        )

    client.before_read_hook = install_foreign_owner
    external_calls: list[str] = []
    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(raw_nonce))
    if result.may_issue_external_side_effect:
        external_calls.append("protected-effect")

    assert result.outcome is WriteOutcome.FOREIGN_NONCE
    assert external_calls == []


def test_importing_dynamodb_adapter_imports_no_boto3_botocore_mlx_numpy_or_mlx_vq() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import glm52_enforcement.dynamodb
for name in ("boto3", "botocore", "mlx", "numpy", "mlx_vq"):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_field_in_committed_readback_is_mismatched_not_unavailable() -> None:
    client = AwsShapedSimulator()
    client.post_commit_fault = TimeoutError("response lost")

    def add_unknown_field() -> None:
        key = ("RUN#" + RUN_ID, "ACTIVATION_INDEX")
        client.items[key]["unknown"] = {"S": "forbidden"}

    client.before_read_hook = add_unknown_field
    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"g" * 32))

    assert result.outcome is WriteOutcome.MISMATCHED_COMMIT
    assert result.may_issue_external_side_effect is False


def test_transaction_rejects_condition_check_on_an_item_it_also_writes() -> None:
    client = AwsShapedSimulator()
    plans = _rollover_arguments(b"f" * 32)
    cleanup_put = plans["snapshot_cleanup_control"]
    plans["cleanup_heads"] = (
        ExactCheck(cleanup_put.key, dict(cleanup_put.item)),
    )

    with pytest.raises(ValueError, match="duplicate|same ledger key"):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**plans)

    assert client.calls == []


def test_coherent_read_rejects_more_than_100_items_before_sdk_call() -> None:
    client = AwsShapedSimulator()
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")
    items = tuple(
        (
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
            "glm52_production_activation_index",
        )
        for _ in range(101)
    )

    with pytest.raises(ValueError, match="100"):
        adapter.read_coherent(items=items)

    assert client.calls == []


def test_rollover_rejects_more_than_100_write_items_before_sdk_call() -> None:
    client = AwsShapedSimulator()
    plans = _rollover_arguments(b"c" * 32)
    check = ExactCheck(
        LedgerKey(RUN_ID, "ACTIVATION#activation-1#RECOVERY_CONTROL"),
        plans["recovery_control"].item,
    )
    plans["cleanup_heads"] = tuple(check for _ in range(95))

    with pytest.raises(ValueError, match="limit"):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**plans)

    assert client.calls == []


@pytest.mark.parametrize(
    ("code", "expected", "reads"),
    [
        ("TransactionCanceledException", WriteOutcome.CONDITION_REJECTED, 0),
        (
            "IdempotentParameterMismatchException",
            WriteOutcome.MISMATCHED_COMMIT,
            0,
        ),
        ("ValidationException", WriteOutcome.MISMATCHED_COMMIT, 0),
        ("AccessDeniedException", WriteOutcome.MISMATCHED_COMMIT, 0),
        ("ResourceNotFoundException", WriteOutcome.MISMATCHED_COMMIT, 0),
        ("InternalServerError", WriteOutcome.PRESTATE_OR_ABSENT, 1),
        ("TransactionInProgressException", WriteOutcome.PRESTATE_OR_ABSENT, 1),
    ],
)
def test_write_error_classification_matrix_is_no_retry_and_fail_closed(
    code: str, expected: WriteOutcome, reads: int
) -> None:
    client = AwsShapedSimulator()
    client.pre_write_fault = SimulatedClientError(
        code,
        cancellation_reasons=(
            {"Code": "ConditionalCheckFailed", "Message": "exact"},
        ),
    )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"e" * 32))

    assert result.outcome is expected
    assert result.error_code == code
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_write_items"
        ]
    ) == 1
    assert len(
        [
            request
            for name, request in client.calls
            if name == "transact_get_items"
        ]
    ) == reads
    if code == "TransactionCanceledException":
        assert result.cancellation_reasons == (
            {"Code": "ConditionalCheckFailed", "Message": "exact"},
        )


@pytest.mark.parametrize(
    "raw_nonce",
    [b"", b"x" * 31, b"x" * 33, bytearray(b"x" * 32), "x" * 32],
)
def test_raw_owner_nonce_requires_exact_32_byte_value(raw_nonce: object) -> None:
    client = AwsShapedSimulator()
    plans = _rollover_arguments(b"x" * 32)
    plans["raw_owner_nonce"] = raw_nonce
    with pytest.raises(ValueError, match="32 bytes"):
        DynamoLedgerAdapter(
            client=client, table_name="ledger-table"
        ).commit_rollover(**plans)
    assert client.calls == []


def test_raw_owner_nonce_never_reaches_request_result_or_exception_text() -> None:
    raw_nonce = bytes(range(32))
    client = AwsShapedSimulator()
    client.post_commit_fault = TimeoutError("opaque response loss")
    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(raw_nonce))

    request_repr = repr(client.calls)
    result_repr = repr(result)
    assert repr(raw_nonce) not in request_repr
    assert raw_nonce.hex() not in request_repr
    assert repr(raw_nonce) not in result_repr
    assert raw_nonce.hex() not in result_repr
    assert result.records[1]["writer_invocation_nonce_sha256"] == (
        hashlib.sha256(raw_nonce).hexdigest()
    )


def test_update_wire_shape_uses_only_set_and_lexically_stable_placeholders() -> None:
    client = AwsShapedSimulator()
    DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).commit_rollover(**_rollover_arguments(b"d" * 32))
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    update = write["TransactItems"][0]["Update"]

    assert update["UpdateExpression"].startswith("SET ")
    assert " REMOVE " not in update["UpdateExpression"]
    assert " ADD " not in update["UpdateExpression"]
    assert " DELETE " not in update["UpdateExpression"]
    fields = [
        update["ExpressionAttributeNames"][
            assignment.split(" = ")[0]
        ]
        for assignment in update["UpdateExpression"][4:].split(", ")
    ]
    assert fields == sorted(fields)


@pytest.mark.parametrize(
    "metadata",
    [
        {"HTTPStatusCode": True, "RequestId": "request"},
        {"HTTPStatusCode": 200, "RequestId": ""},
        {"HTTPStatusCode": 500, "RequestId": "request"},
        {},
    ],
)
def test_consistent_read_rejects_malformed_response_metadata(
    metadata: dict[str, object],
) -> None:
    class MalformedReadSimulator(AwsShapedSimulator):
        def get_item(self, **kwargs: object) -> dict[str, object]:
            response = super().get_item(**kwargs)
            response["ResponseMetadata"] = deepcopy(metadata)
            return response

    client = MalformedReadSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    adapter = DynamoLedgerAdapter(client=client, table_name="ledger-table")

    with pytest.raises(Exception, match="response|status|request ID|metadata"):
        adapter.read_consistent_item(
            key=LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
            record_type="glm52_production_activation_index",
        )


@pytest.mark.parametrize(
    "action_name",
    [
        "SAME_TOKEN_COMPLETE",
        "TERMINATE_LATE_INSTANCE",
        "POST_TERMINAL_ALLOCATION_OPEN",
        "POST_TERMINAL_ALLOCATION_CLOSE",
    ],
)
def test_liability_dispatcher_derives_each_closed_write_set(
    action_name: str,
) -> None:
    client = AwsShapedSimulator()
    client.install("ACTIVATION_INDEX", _activation_index())
    raw_nonce = b"b" * 32
    nonce_sha = hashlib.sha256(raw_nonce).hexdigest()
    liability_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY#00000001"
    )
    liability_before = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state="WATCHING",
        owner_invocation_nonce_sha256=nonce_sha,
    )
    liability_state = {
        "SAME_TOKEN_COMPLETE": "SAME_TOKEN_COMPLETION",
        "TERMINATE_LATE_INSTANCE": "LATE_INSTANCE_DRAINING",
    }.get(action_name, "WATCHING")
    liability_after = _closed_record(
        "glm52_production_worker_launch_liability",
        activation_id="activation-1",
        state=liability_state,
        owner_invocation_nonce_sha256=nonce_sha,
        revision=2,
    )
    liability_identity = canonical_sha256(liability_before)
    action_resource = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": liability_before["generation"],
        "generation_text": "00000001",
        "allocation_ordinal": liability_before["allocation_ordinal"],
        "allocation_ordinal_text": "00000001",
        "worker_launch_identity_sha256": liability_before[
            "worker_launch_identity_sha256"
        ],
        "worker_launch_liability_identity_sha256": liability_identity,
        "owner_execution_arn": liability_before["owner_execution_arn"],
        "owner_state_machine_version_arn": liability_before[
            "owner_state_machine_version_arn"
        ],
        "owner_dispatch_identity_sha256": liability_before[
            "owner_dispatch_identity_sha256"
        ],
        "owner_invocation_nonce_sha256": nonce_sha,
    }
    client.install(liability_key, liability_before)
    action_key = (
        "ACTIVATION#activation-1#WORKER_LAUNCH_LIABILITY_ACTION#"
        f"00000001#{action_name}#00000001"
    )
    action_before = _closed_record(
        "glm52_production_worker_launch_liability_action",
        **action_resource,
        action_kind=action_name,
    )
    action_after = _closed_record(
        "glm52_production_worker_launch_liability_action",
        **action_resource,
        action_kind=action_name,
        state="CONSUMED",
        authority_audit_body_sha256="e" * 64,
        authority_audit_closing_revision=1,
        authorized_transition_from_revision=1,
        authorized_transition_to_revision=2,
        revision=2,
    )
    client.install(action_key, action_before)
    allocation: object = None
    if action_name.startswith("POST_TERMINAL"):
        allocation_key = (
            "ACTIVATION#activation-1#POST_TERMINAL_ALLOCATION#"
            "00000001#i-123"
        )
        before_state = (
            "DISCOVERED"
            if action_name.endswith("_OPEN")
            else "INSTANCE_TERMINAL"
        )
        after_state = (
            "ALLOCATION_OPEN"
            if action_name.endswith("_OPEN")
            else "ALLOCATION_CLOSED"
        )
        allocation_before = _closed_record(
            "glm52_production_post_terminal_allocation",
            activation_id="activation-1",
            activation_ordinal=1,
            generation=liability_before["generation"],
            allocation_ordinal=liability_before["allocation_ordinal"],
            worker_launch_identity_sha256=liability_before[
                "worker_launch_identity_sha256"
            ],
            worker_launch_liability_identity_sha256=liability_identity,
            instance_id="i-123",
            state=before_state,
            owner_execution_arn=liability_before["owner_execution_arn"],
            owner_state_machine_version_arn=liability_before[
                "owner_state_machine_version_arn"
            ],
            owner_dispatch_identity_sha256=liability_before[
                "owner_dispatch_identity_sha256"
            ],
            owner_invocation_nonce_sha256=nonce_sha,
        )
        allocation_after = _closed_record(
            "glm52_production_post_terminal_allocation",
            activation_id="activation-1",
            activation_ordinal=1,
            generation=liability_before["generation"],
            allocation_ordinal=liability_before["allocation_ordinal"],
            worker_launch_identity_sha256=liability_before[
                "worker_launch_identity_sha256"
            ],
            worker_launch_liability_identity_sha256=liability_identity,
            instance_id="i-123",
            state=after_state,
            owner_execution_arn=(
                None
                if after_state == "ALLOCATION_CLOSED"
                else liability_before["owner_execution_arn"]
            ),
            owner_state_machine_version_arn=(
                None
                if after_state == "ALLOCATION_CLOSED"
                else liability_before["owner_state_machine_version_arn"]
            ),
            owner_dispatch_identity_sha256=(
                None
                if after_state == "ALLOCATION_CLOSED"
                else liability_before["owner_dispatch_identity_sha256"]
            ),
            owner_invocation_nonce_sha256=(
                None if after_state == "ALLOCATION_CLOSED" else nonce_sha
            ),
            revision=2,
        )
        client.install(allocation_key, allocation_before)
        allocation = ExactUpdate(
            LedgerKey(RUN_ID, allocation_key),
            allocation_before,
            allocation_after,
        )

    result = DynamoLedgerAdapter(
        client=client, table_name="ledger-table"
    ).consume_liability_action(
        action_name=action_name,
        index=ExactCheck(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"), _activation_index()
        ),
        liability=ExactUpdate(
            LedgerKey(RUN_ID, liability_key),
            liability_before,
            liability_after,
        ),
        action=ExactUpdate(
            LedgerKey(RUN_ID, action_key), action_before, action_after
        ),
        post_terminal_allocation=allocation,
        domain="LIABILITY_ACTION",
        operation_identity_sha256="a" * 64,
        raw_owner_nonce=raw_nonce,
    )

    expected_kinds = ["ConditionCheck", "Update", "Update"]
    if allocation is not None:
        expected_kinds.append("Update")
    write = next(
        request
        for name, request in client.calls
        if name == "transact_write_items"
    )
    assert [next(iter(item)) for item in write["TransactItems"]] == expected_kinds
    assert result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT
