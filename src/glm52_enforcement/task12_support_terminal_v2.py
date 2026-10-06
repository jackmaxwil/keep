"""Activation-domain support ingress for the retained TerminalV2 writer.

The published TerminalV2 Lambda has two closed modes.  The retained mode is
owned by the retained lifecycle workflow.  This module is the second mode:
an exact SupportDeadline version supplies its complete current predecessor,
the writer authenticates the still-running support execution and durable
drain successor, assembles TerminalV2 itself, then conditionally creates and
read-backs the immutable object plus its version-control successor.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import decode_item, encode_item
from .records import RECORD_FIELDS, ledger_pk, ledger_sk, validate_record
from .task12_operations import validate_operation_descriptor
from .task12_writers import (
    RetainedWriterActionAuthority,
    RetainedWriterAuditAuthority,
    RetainedWriteResult,
    build_retained_writer_candidate,
    build_versioned_writer_control,
    validate_retained_writer_authority,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LAMBDA_VERSION = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:"
    r"function:keep-glm52-h1g-support-deadline:[1-9][0-9]*\Z"
)
_TERMINAL_VERSION = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:"
    r"function:keep-glm52-h1g-terminal-v2-writer:[1-9][0-9]*\Z"
)
_STATE_VERSION = re.compile(
    r"arn:aws:states:us-west-2:246813579024:"
    r"stateMachine:keep-glm52-h1g-support:[1-9][0-9]*\Z"
)
_EXECUTION = re.compile(
    r"arn:aws:states:us-west-2:246813579024:"
    r"execution:[A-Za-z0-9_-]+:[A-Za-z0-9._:-]+\Z"
)
_REQUEST_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "activation_id",
    "generation",
    "generation_text",
    "caller_function_version_arn",
    "caller_state_machine_version_arn",
    "caller_execution_arn",
    "support_state_identity_sha256",
    "support_state",
    "canonical_identity_sha256",
}


class SupportTerminalV2Error(ValueError):
    """The activation-domain TerminalV2 edge lost exact authority."""


@dataclass(frozen=True)
class SupportTerminalV2Config:
    account_id: str
    run_id: str
    activation_id: str
    ledger_table_name: str
    campaign_bucket: str


@dataclass(frozen=True)
class SupportTerminalV2Clients:
    s3: object
    dynamodb: object
    stepfunctions: object


def _meta(value: object, label: str) -> Mapping[str, object]:
    metadata = value.get("ResponseMetadata") if type(value) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(metadata.get("RetryAttempts")) is not int
        or metadata["RetryAttempts"] != 0
    ):
        raise SupportTerminalV2Error(
            label + " is unauthenticated or not zero-retry"
        )
    return metadata


def _self_hash(value: Mapping[str, object], field: str) -> None:
    identity = value.get(field)
    body = dict(value)
    body.pop(field, None)
    if (
        type(identity) is not str
        or _SHA.fullmatch(identity) is None
        or identity != canonical_sha256(body)
    ):
        raise SupportTerminalV2Error(field + " drifted")


def _parse(event: object) -> Mapping[str, object]:
    if type(event) is not dict or set(event) != _REQUEST_FIELDS:
        raise SupportTerminalV2Error("support TerminalV2 request is not closed")
    _self_hash(event, "canonical_identity_sha256")
    state = event["support_state"]
    if (
        event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task12_support_terminal_v2_request_v1"
        or event["run_id"] != "glm52-sky-20260724"
        or type(event["generation"]) is not int
        or event["generation"] < 1
        or event["generation_text"] != f"{event['generation']:08d}"
        or _LAMBDA_VERSION.fullmatch(
            str(event["caller_function_version_arn"])
        )
        is None
        or _STATE_VERSION.fullmatch(
            str(event["caller_state_machine_version_arn"])
        )
        is None
        or _EXECUTION.fullmatch(str(event["caller_execution_arn"])) is None
        or type(state) is not dict
        or state.get("canonical_identity_sha256")
        != event["support_state_identity_sha256"]
    ):
        raise SupportTerminalV2Error(
            "support TerminalV2 request identity drifted"
        )
    _self_hash(state, "canonical_identity_sha256")
    return event


def _entry(kind: str, **values: object) -> dict[str, object]:
    body = {"kind": kind, **values}
    return {**body, "canonical_entry_sha256": canonical_sha256(body)}


def _effect(state: Mapping[str, object], kind: str) -> Mapping[str, object]:
    effects = state.get("effects")
    matches = [
        item
        for item in effects
        if type(item) is dict and item.get("effect_kind") == kind
    ] if type(effects) is list else []
    if len(matches) != 1:
        raise SupportTerminalV2Error(kind + " effect is not singular")
    return matches[0]


def _request_evidence(
    request: Mapping[str, object],
) -> list[dict[str, object]]:
    state = request["support_state"]
    handoff = state["handoff"]
    cardinality = state["request_cardinality"]
    correlation_sha = canonical_sha256(state["correlation_tuple"])
    close = _effect(state, "ALLOCATION_CLOSED")["evidence"]
    observations = close.get("terminal_observations")
    if (
        type(observations) is not list
        or len(observations) != 2
        or observations[0]["scan_completed_at"]
        >= observations[1]["scan_started_at"]
    ):
        raise SupportTerminalV2Error(
            "terminal quiescence observations are incomplete"
        )
    result: list[dict[str, object]] = []
    correlation = state.get("correlation")
    matches = (
        correlation.get("matches")
        if type(correlation) is dict
        else []
    )
    if cardinality == "ZERO":
        proof = state.get("zero_request_proof")
        if type(proof) is not dict:
            raise SupportTerminalV2Error("zero request proof is absent")
        result.append(
            _entry(
                "ZERO_MATCH_SCAN",
                correlation_tuple_sha256=correlation_sha,
                database_lower_bound=proof["first_observed_at"],
                database_upper_bound=proof["second_observed_at"],
                journal_lower_bound=proof["first_observed_at"],
                journal_upper_bound=proof["second_observed_at"],
                pagination_complete=True,
                match_count=0,
                scan_started_at=proof["first_observed_at"],
                scan_completed_at=proof["second_observed_at"],
                database_head_identity_sha256=proof[
                    "canonical_identity_sha256"
                ],
                journal_head_identity_sha256=proof[
                    "canonical_identity_sha256"
                ],
                scheduler_snapshot_identity_sha256=proof[
                    "canonical_identity_sha256"
                ],
                worker_snapshot_identity_sha256=proof[
                    "canonical_identity_sha256"
                ],
                allocation_snapshot_identity_sha256=proof[
                    "canonical_identity_sha256"
                ],
            )
        )
    elif cardinality == "MULTIPLE":
        ids = sorted(item["request_id"] for item in matches)
        result.append(
            _entry(
                "MULTIPLE_MATCH",
                request_ids=ids,
                correlation_tuple_sha256=correlation_sha,
                complete_member_set_sha256=canonical_sha256(ids),
                scan_identity_sha256=state[
                    "canonical_identity_sha256"
                ],
            )
        )
    for item in matches:
        body = item["body"]
        result.append(
            _entry(
                "EXACT_REQUEST",
                request_id=item["request_id"],
                correlation_tuple_sha256=correlation_sha,
                stored_request_body_sha256=handoff["body"][
                    "request_body_sha256"
                ],
                stored_arguments_sha256=canonical_sha256(body),
                request_state=item["state"],
                request_created_at=body["created_at"],
                request_updated_at=body["updated_at"],
                scheduler_work_identity_sha256=body[
                    "evidence_identity_sha256"
                ],
                terminal_observation_identity_sha256=observations[0][
                    "identity_sha256"
                ],
            )
        )
    cancel_kinds = {
        "REQUEST_CANCEL": "REQUEST_CANCEL",
        "JOB_CANCEL": "JOB_CANCEL",
    }
    for effect in state["effects"]:
        kind = cancel_kinds.get(effect.get("effect_kind"))
        if kind is None:
            continue
        result.append(
            _entry(
                kind,
                action_key=effect["evidence"]["action_key"],
                action_identity_sha256=effect[
                    "canonical_identity_sha256"
                ],
                response_identity_sha256=effect["evidence"][
                    "relay_result_identity_sha256"
                ],
                correlation_identity_sha256=correlation_sha,
                terminal_observation_identity_sha256=observations[0][
                    "identity_sha256"
                ],
                requested_at=effect["evidence"]["requested_at"],
                terminal_observed_at=observations[0][
                    "scan_completed_at"
                ],
            )
        )
    result.extend(
        _entry(
            "QUIESCENCE_SNAPSHOT",
            scan_started_at=item["scan_started_at"],
            scan_completed_at=item["scan_completed_at"],
            request_scan_identity_sha256=item["identity_sha256"],
            job_scan_identity_sha256=item["identity_sha256"],
            worker_scan_identity_sha256=item["identity_sha256"],
            allocation_scan_identity_sha256=item["identity_sha256"],
            controller_scan_identity_sha256=item["identity_sha256"],
        )
        for item in observations
    )
    return result


def _terminal_record(
    request: Mapping[str, object],
    *,
    invoked_function_arn: str,
    activation_ordinal: int,
    spend: object,
    intervals: tuple[Mapping[str, object], ...],
    families: Mapping[str, tuple[dict[str, object], ...]],
    terminal_evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    state = request["support_state"]
    handoff = state["handoff"]
    cardinality = state["request_cardinality"]
    observations = _effect(
        state, "ALLOCATION_CLOSED"
    )["evidence"]["terminal_observations"]
    created_at = observations[-1]["scan_completed_at"]
    bound = state.get("binding") is not None
    support_request_evidence = _request_evidence(request)
    launches = families["glm52_production_worker_launch"]
    liabilities = families[
        "glm52_production_worker_launch_liability"
    ]
    worker_ids = set(
        _effect(state, "WORKER_TERMINAL")["evidence"].get(
            "instances", []
        )
    )
    if terminal_evidence is None:
        if launches or liabilities or intervals or worker_ids:
            raise SupportTerminalV2Error(
                "zero-allocation TerminalV2 has live allocation evidence"
            )
        outcome = (
            "QUIESCED_UNRESOLVED_BOUND_JOB_NO_ALLOCATION_INCIDENT"
            if bound
            else "QUIESCED_UNRESOLVED_REQUEST_NO_ALLOCATION_INCIDENT"
        )
        final_sky_state = "QUIESCED"
        final_ec2_states: list[object] = []
        allocations: list[object] = []
        launch_evidence: list[object] = []
        liability_evidence: list[object] = []
        final_fields = {
            "final_heartbeat_identity": None,
            "checkpoint_identity": None,
            "cache_identity": None,
            "training_identity": None,
            "evaluation_identity": None,
            "drain_identity": None,
            "post_terminal_quiescence_evidence": {
                "identity": canonical_sha256(observations)
            },
            "prior_terminal_v1_identity": None,
            "operator_disposition_required": True,
        }
    else:
        if (
            terminal_evidence.get("handoff") != handoff
            or terminal_evidence.get("binding") != state.get("binding")
            or terminal_evidence.get("request_evidence")
            != support_request_evidence
        ):
            raise SupportTerminalV2Error(
                "Task9 terminal evidence drifted from support state"
            )
        final_sky_state = str(
            terminal_evidence["final_sky_state"]
        )
        final_ec2_states = list(
            terminal_evidence["final_ec2_states"]
        )
        allocations = list(terminal_evidence["allocations"])
        launch_evidence = list(
            terminal_evidence["worker_launch_evidence"]
        )
        liability_evidence = list(
            terminal_evidence["worker_launch_liabilities"]
        )
        allocation_instances = {
            item.get("instance_id")
            for item in allocations
            if type(item) is dict
        }
        interval_instances = {
            item.get("instance_id") for item in intervals
        }
        if (
            allocation_instances != interval_instances
            or not worker_ids.issubset(allocation_instances)
        ):
            raise SupportTerminalV2Error(
                "support allocation evidence drifted from live spend"
            )
        outcome = str(terminal_evidence["outcome"])
        final_fields = {
            field: terminal_evidence[field]
            for field in (
                "final_heartbeat_identity",
                "checkpoint_identity",
                "cache_identity",
                "training_identity",
                "evaluation_identity",
                "drain_identity",
                "post_terminal_quiescence_evidence",
                "prior_terminal_v1_identity",
                "operator_disposition_required",
            )
        }
    arrays: dict[str, list[object]] = {
        "final_ec2_states": final_ec2_states,
        "allocations": allocations,
        "worker_launch_evidence": launch_evidence,
        "worker_launch_liabilities": liability_evidence,
        "request_evidence": support_request_evidence,
    }
    record: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_production_terminal_v2",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": handoff["body"][
            "campaign_identity_sha256"
        ],
        "activation_id": request["activation_id"],
        "activation_ordinal": activation_ordinal,
        "generation": request["generation"],
        "generation_text": request["generation_text"],
        "action_key": (
            f"ACTIVATION#{request['activation_id']}#ACTIVATION_ACTION#"
            f"TERMINAL_V2#{request['generation_text']}"
        ),
        "action_identity_sha256": canonical_sha256(
            {
                "support_state_identity_sha256": request[
                    "support_state_identity_sha256"
                ],
                "outcome": outcome,
            }
        ),
        "handoff": handoff,
        "binding": state.get("binding"),
        "final_sky_state": final_sky_state,
        **arrays,
        "request_cardinality": cardinality,
        "worker_cardinality": (
            "ZERO"
            if not allocations
            else ("ONE" if len(allocations) == 1 else "MULTIPLE")
        ),
        "spend_ledger_head_identity": {
            "identity": spend.ledger_head_identity_sha256
        },
        "remaining_approved_gpu_seconds": spend.remaining_gpu_seconds,
        "remaining_approved_gpu_usd": spend.remaining_gpu_cost_usd,
        **final_fields,
        "terminal_observation_window": {
            "first": observations[0]["identity_sha256"],
            "second": observations[1]["identity_sha256"],
            "identity": canonical_sha256(observations),
        },
        "outcome": outcome,
        "writer_function_version_arn": invoked_function_arn,
        "writer_dispatch_identity_sha256": request[
            "support_state_identity_sha256"
        ],
        "writer_invocation_nonce_sha256": canonical_sha256(
            {
                "caller": request["caller_function_version_arn"],
                "state": request["support_state_identity_sha256"],
            }
        ),
        "created_at": created_at,
    }
    for name, value in arrays.items():
        if name != "final_ec2_states":
            record[name + "_array_sha256"] = canonical_sha256(value)
    record["canonical_body_sha256"] = canonical_sha256(record)
    if set(record) != set(
        RECORD_FIELDS["glm52_production_terminal_v2"]
    ):
        raise SupportTerminalV2Error("TerminalV2 field set drifted")
    return validate_record("glm52_production_terminal_v2", record)


class AwsSupportTerminalV2Services:
    def __init__(
        self,
        *,
        config: SupportTerminalV2Config,
        clients: SupportTerminalV2Clients,
    ) -> None:
        self.config = config
        self.clients = clients

    def _read(self, sk: str) -> tuple[Mapping[str, object], str]:
        response = self.clients.dynamodb.get_item(
            TableName=self.config.ledger_table_name,
            Key=encode_item({"PK": ledger_pk(self.config.run_id), "SK": sk}),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        request_id = str(_meta(response, "TerminalV2 authority read")["RequestId"])
        item = response.get("Item")
        if type(item) is not dict:
            raise SupportTerminalV2Error("TerminalV2 authority is absent")
        value = decode_item(item)
        value.pop("PK", None)
        value.pop("SK", None)
        return value, request_id

    def _deployment(
        self,
        request: Mapping[str, object],
        *,
        invoked_function_arn: str,
    ) -> tuple[Mapping[str, object], list[str]]:
        caller_version = str(
            request["caller_state_machine_version_arn"]
        )
        caller_base = caller_version.rsplit(":", 1)[0]
        deployment_sk = (
            "TASK12_LAMBDA_DEPLOYMENT#"
            + invoked_function_arn
            + "#CALLER#"
            + caller_base
            + "#ACTIVATION#"
            + self.config.activation_id
            + "#GENERATION#"
            + str(request["generation_text"])
        )
        deployment, deployment_request = self._read(deployment_sk)
        deployment_body = dict(deployment)
        deployment_identity = deployment_body.pop(
            "canonical_body_sha256", None
        )
        roles = deployment.get("role_coordinates")
        expected_role_fields = {
            "authority",
            "campaign_bucket",
            "spend_runtime_prefix",
            "campaign_descriptor_key",
            "campaign_descriptor_version_id",
            "campaign_descriptor_file_sha256",
            "gpu_spend_approval_key",
            "gpu_spend_approval_version_id",
            "gpu_spend_approval_file_sha256",
            "kms_key_id",
            "ledger_table_name",
            "numeric_binding_version_arn",
            "task9_deployed_identity_coordinate",
            "task9_deployed_identity_sha256",
            "task10_worker_descriptor_coordinate",
            "terminal_evidence_prefix",
            "support_observer_version_arn",
        }
        if (
            set(deployment)
            != {
                "schema_version",
                "record_type",
                "account_id",
                "region",
                "handler_kind",
                "mode",
                "activation_id",
                "activation_ordinal",
                "generation",
                "generation_text",
                "function_name",
                "function_version",
                "invoked_function_version_arn",
                "caller_state_machine_version_arn",
                "role_coordinates",
                "canonical_body_sha256",
            }
            or deployment_identity
            != canonical_sha256(deployment_body)
            or deployment.get("schema_version") != 1
            or deployment.get("record_type")
            != "glm52_task12_lambda_deployment_v1"
            or deployment.get("account_id") != self.config.account_id
            or deployment.get("region") != "us-west-2"
            or deployment.get("handler_kind")
            != "RETAINED_TERMINAL_V2"
            or deployment.get("mode") != "SUPPORT_OBSERVER"
            or deployment.get("activation_id")
            != self.config.activation_id
            or deployment.get("generation") != request["generation"]
            or deployment.get("generation_text")
            != request["generation_text"]
            or deployment.get("invoked_function_version_arn")
            != invoked_function_arn
            or deployment.get("caller_state_machine_version_arn")
            != caller_version
            or type(roles) is not dict
            or set(roles) != expected_role_fields
            or roles.get("support_observer_version_arn")
            != request["caller_function_version_arn"]
            or roles.get("ledger_table_name")
            != self.config.ledger_table_name
            or roles.get("campaign_bucket")
            != self.config.campaign_bucket
        ):
            raise SupportTerminalV2Error(
                "support deployment authority is foreign"
            )
        authority_coordinate = roles["authority"]
        if (
            type(authority_coordinate) is not dict
            or set(authority_coordinate)
            != {"bucket", "key", "version_id", "file_sha256"}
        ):
            raise SupportTerminalV2Error(
                "support invocation authority coordinate drifted"
            )
        response = self.clients.s3.get_object(
            Bucket=authority_coordinate["bucket"],
            Key=authority_coordinate["key"],
            VersionId=authority_coordinate["version_id"],
            ExpectedBucketOwner=self.config.account_id,
        )
        authority_request = str(
            _meta(response, "support invocation authority")["RequestId"]
        )
        body = response.get("Body")
        raw = body.read() if callable(getattr(body, "read", None)) else body
        if (
            type(raw) is not bytes
            or hashlib.sha256(raw).hexdigest()
            != authority_coordinate["file_sha256"]
        ):
            raise SupportTerminalV2Error(
                "support invocation authority transport drifted"
            )
        import json

        try:
            authority = json.loads(raw.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SupportTerminalV2Error(
                "support invocation authority is not canonical JSON"
            ) from exc
        authority_body = dict(authority)
        authority_identity = authority_body.pop(
            "canonical_body_sha256", None
        )
        operations = authority.get("operations")
        operation = "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        if (
            type(authority) is not dict
            or canonical_json_bytes(authority) != raw
            or authority_identity != canonical_sha256(authority_body)
            or authority.get("handler_kind")
            != "RETAINED_TERMINAL_V2"
            or authority.get("activation_id")
            != self.config.activation_id
            or authority.get("activation_ordinal")
            != deployment["activation_ordinal"]
            or authority.get("generation") != request["generation"]
            or authority.get("generation_text")
            != request["generation_text"]
            or type(operations) is not dict
            or set(operations) != {operation}
        ):
            raise SupportTerminalV2Error(
                "support invocation authority is foreign"
            )
        entry = operations[operation]
        if (
            type(entry) is not dict
            or set(entry)
            != {"state_sort_key", "state_body_sha256"}
        ):
            raise SupportTerminalV2Error(
                "support operation descriptor coordinate drifted"
            )
        descriptor, descriptor_request = self._read(
            str(entry["state_sort_key"])
        )
        try:
            validated = validate_operation_descriptor(descriptor)
        except ValueError as exc:
            raise SupportTerminalV2Error(
                "support operation descriptor is invalid"
            ) from exc
        if (
            validated.get("canonical_body_sha256")
            != entry["state_body_sha256"]
            or validated.get("operation_kind") != operation
            or validated.get("handler_kind")
            != "RETAINED_TERMINAL_V2"
        ):
            raise SupportTerminalV2Error(
                "support operation descriptor identity drifted"
            )
        return deployment, [
            deployment_request,
            authority_request,
            descriptor_request,
        ]

    def _live_terminal_authority(
        self,
        request: Mapping[str, object],
        deployment: Mapping[str, object],
    ) -> tuple[
        object,
        tuple[Mapping[str, object], ...],
        Mapping[str, tuple[dict[str, object], ...]],
        Mapping[str, object] | None,
    ]:
        from .task12_live_drain_effects import (
            _families,
            _read_terminal_evidence,
        )
        from .task12_live_runtime import PROVE_TERMINAL, _spend_authority

        roles = deployment["role_coordinates"]

        class _Ports:
            def __init__(self, services: AwsSupportTerminalV2Services):
                self.deployment = SimpleNamespace(
                    role_coordinates=roles
                )
                self._services = services

            def client(self, service: str) -> object:
                clients = self._services.clients
                if service == "s3":
                    return clients.s3
                if service == "dynamodb":
                    return clients.dynamodb
                raise SupportTerminalV2Error(
                    "support live authority requested foreign client"
                )

        ports = _Ports(self)
        invocation = SimpleNamespace(
            activation_id=self.config.activation_id,
            activation_ordinal=deployment["activation_ordinal"],
            generation=request["generation"],
            generation_text=request["generation_text"],
            dispatch_identity_sha256=request[
                "support_state_identity_sha256"
            ],
            state_machine_execution_arn=request[
                "caller_execution_arn"
            ],
            caller_state_machine_version_arn=request[
                "caller_state_machine_version_arn"
            ],
            operation_input={},
            operation_kind=(
                "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
            ),
        )
        campaign_identity = str(
            request["support_state"]["handoff"]["body"][
                "campaign_identity_sha256"
            ]
        )
        families = _families(
            ports=ports,
            invocation=invocation,
            campaign_identity_sha256=campaign_identity,
        )
        observations = _effect(
            request["support_state"], "ALLOCATION_CLOSED"
        )["evidence"]["terminal_observations"]
        spend, intervals, _chain, _request_ids = _spend_authority(
            ports=ports,
            invocation=SimpleNamespace(
                activation_id=self.config.activation_id,
                activation_ordinal=deployment["activation_ordinal"],
                generation=request["generation"],
                generation_text=request["generation_text"],
                operation_kind=PROVE_TERMINAL,
            ),
            observed_at=observations[-1]["scan_completed_at"],
            campaign_identity_sha256=campaign_identity,
            families=families,
        )
        if (
            getattr(spend, "state", None) != "CLOSED"
            or any(item.get("state") != "CLOSED" for item in intervals)
        ):
            raise SupportTerminalV2Error(
                "live spend is not terminally closed"
            )
        terminal_evidence = (
            _read_terminal_evidence(
                ports=ports,
                invocation=invocation,
                recovery={
                    "campaign_identity_sha256": campaign_identity
                },
                families=families,
            )
            if (
                families["glm52_production_worker_launch"]
                or families[
                    "glm52_production_worker_launch_liability"
                ]
            )
            else None
        )
        return spend, intervals, families, terminal_evidence

    def _activation_writer_authority(
        self,
        *,
        candidate: object,
        request: Mapping[str, object],
        observed_at: str,
    ) -> tuple[
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
        str,
    ]:
        action_key = (
            "ACTIVATION#"
            + self.config.activation_id
            + "#ACTIVATION_ACTION#TERMINAL_V2#"
            + str(request["generation_text"])
        )
        action_identity = canonical_sha256(
            {
                "action_key": action_key,
                "candidate_identity_sha256": candidate.candidate_identity_sha256,
                "support_state_identity_sha256": request[
                    "support_state_identity_sha256"
                ],
            }
        )
        action = RetainedWriterActionAuthority(
            authority_domain="ACTIVATION",
            action_kind="TERMINAL_V2",
            action_key=action_key,
            candidate_identity_sha256=(
                candidate.candidate_identity_sha256
            ),
            action_identity_sha256=action_identity,
            owner_invocation_nonce_sha256=request[
                "support_state_identity_sha256"
            ],
            state="CONSUMED",
            authorized_revision=1,
        )
        audit_body = {
            "authority_domain": "ACTIVATION",
            "action_kind": "TERMINAL_V2",
            "action_key": action_key,
            "candidate_identity_sha256": (
                candidate.candidate_identity_sha256
            ),
            "action_identity_sha256": action_identity,
            "audit_kind": "H1F_GENESIS_TO_ZERO_CHILD",
            "closing_revision": 0,
            "authorized_revision": 1,
            "current_revision": 1,
            "observed_at": observed_at,
        }
        audit = RetainedWriterAuditAuthority(
            **audit_body,
            canonical_identity_sha256=canonical_sha256(audit_body),
        )
        validate_retained_writer_authority(
            candidate=candidate,
            action=action,
            audit=audit,
        )
        authority_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_terminal_writer_authority_v1"
            ),
            "activation_id": self.config.activation_id,
            "generation": request["generation"],
            "generation_text": request["generation_text"],
            "support_state_identity_sha256": request[
                "support_state_identity_sha256"
            ],
            "action": {
                field: getattr(action, field)
                for field in action.__dataclass_fields__
            },
            "audit": {
                field: getattr(audit, field)
                for field in audit.__dataclass_fields__
            },
        }
        authority = {
            **authority_body,
            "canonical_identity_sha256": canonical_sha256(
                authority_body
            ),
        }
        key = {
            "PK": ledger_pk(self.config.run_id),
            "SK": action_key,
        }
        item = {**key, **authority}
        try:
            response = self.clients.dynamodb.put_item(
                TableName=self.config.ledger_table_name,
                Item=encode_item(item),
                ConditionExpression=(
                    "attribute_not_exists(#pk) AND "
                    "attribute_not_exists(#sk)"
                ),
                ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
                ReturnConsumedCapacity="NONE",
            )
            request_id = str(
                _meta(
                    response, "support TerminalV2 action authority"
                )["RequestId"]
            )
        except Exception:  # noqa: BLE001 - ambiguous write requires readback
            response = self.clients.dynamodb.get_item(
                TableName=self.config.ledger_table_name,
                Key=encode_item(key),
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
            request_id = str(
                _meta(
                    response,
                    "support TerminalV2 action authority readback",
                )["RequestId"]
            )
            stored = response.get("Item")
            if type(stored) is not dict or decode_item(stored) != item:
                raise SupportTerminalV2Error(
                    "support TerminalV2 action authority adopted foreign bytes"
                )
        return action, audit, request_id

    def execute(
        self, request: Mapping[str, object], *, invoked_function_arn: str
    ) -> Mapping[str, object]:
        deployment, deployment_requests = self._deployment(
            request,
            invoked_function_arn=invoked_function_arn,
        )
        support_workflow_version_arn = str(
            deployment["caller_state_machine_version_arn"]
        )
        execution = self.clients.stepfunctions.describe_execution(
            executionArn=request["caller_execution_arn"]
        )
        _meta(execution, "support execution observation")
        if (
            execution.get("status") != "RUNNING"
            or execution.get("stateMachineVersionArn")
            != support_workflow_version_arn
        ):
            raise SupportTerminalV2Error(
                "support execution is not the live authorized caller"
            )
        index, index_request = self._read(
            ledger_sk("glm52_production_activation_index")
        )
        phase, phase_request = self._read(
            "ACTIVATION#"
            + self.config.activation_id
            + "#SUPPORT_DRAIN"
        )
        if (
            index.get("current_activation_id")
            != self.config.activation_id
            or phase.get("phase") != "ALLOCATION_CLOSED"
            or phase.get("support_state_identity_sha256")
            != request["support_state_identity_sha256"]
        ):
            raise SupportTerminalV2Error(
                "support drain successor is not current"
            )
        (
            spend,
            intervals,
            families,
            terminal_evidence,
        ) = self._live_terminal_authority(request, deployment)
        record = _terminal_record(
            request,
            invoked_function_arn=invoked_function_arn,
            activation_ordinal=index["current_activation_ordinal"],
            spend=spend,
            intervals=intervals,
            families=families,
            terminal_evidence=terminal_evidence,
        )
        candidate = build_retained_writer_candidate(
            writer_kind="TerminalV2",
            campaign_bucket=self.config.campaign_bucket,
            activation_id=self.config.activation_id,
            generation=request["generation"],
            authority_domain="ACTIVATION",
            record=record,
        )
        _action, _audit, action_request = (
            self._activation_writer_authority(
                candidate=candidate,
                request=request,
                observed_at=record["created_at"],
            )
        )
        key = candidate.coordinate.split("/", 3)[3]
        checksum = base64.b64encode(
            hashlib.sha256(candidate.raw).digest()
        ).decode("ascii")
        try:
            response = self.clients.s3.put_object(
                Bucket=self.config.campaign_bucket,
                Key=key,
                Body=candidate.raw,
                IfNoneMatch="*",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=checksum,
                ContentType="application/json",
                ExpectedBucketOwner=self.config.account_id,
            )
            request_id = str(
                _meta(response, "TerminalV2 create")["RequestId"]
            )
            version_id = response.get("VersionId")
            if type(version_id) is not str or not version_id:
                raise SupportTerminalV2Error(
                    "TerminalV2 create returned no object version"
                )
            write_request_ids = [request_id]
        except Exception:  # noqa: BLE001 - ambiguous write requires readback
            listed = self.clients.s3.list_object_versions(
                Bucket=self.config.campaign_bucket,
                Prefix=key,
                ExpectedBucketOwner=self.config.account_id,
            )
            list_request = str(
                _meta(
                    listed, "TerminalV2 ambiguous version listing"
                )["RequestId"]
            )
            if listed.get("IsTruncated") is not False:
                raise SupportTerminalV2Error(
                    "TerminalV2 reconciliation is truncated"
                )
            exact: list[tuple[str, str]] = []
            for item in listed.get("Versions", []):
                if item.get("Key") != key:
                    continue
                version = item.get("VersionId")
                if type(version) is not str or not version:
                    continue
                readback = self.clients.s3.get_object(
                    Bucket=self.config.campaign_bucket,
                    Key=key,
                    VersionId=version,
                    ExpectedBucketOwner=self.config.account_id,
                )
                read_request = str(
                    _meta(
                        readback, "TerminalV2 ambiguous exact read"
                    )["RequestId"]
                )
                stream = readback.get("Body")
                raw = (
                    stream.read()
                    if callable(getattr(stream, "read", None))
                    else stream
                )
                if raw == candidate.raw:
                    exact.append((version, read_request))
            if len(exact) != 1:
                raise SupportTerminalV2Error(
                    "TerminalV2 ambiguous write was not adopted exactly"
                )
            version_id, read_request = exact[0]
            request_id = list_request
            write_request_ids = [list_request, read_request]
        result_body = {
            "writer_kind": "TerminalV2",
            "outcome": "created-authenticated",
            "coordinate": candidate.coordinate,
            "candidate_identity_sha256": (
                candidate.candidate_identity_sha256
            ),
            "object_version_id": version_id,
            "response_request_ids": tuple(write_request_ids),
            "response_authenticated": True,
        }
        write = RetainedWriteResult(
            **result_body,
            canonical_identity_sha256=canonical_sha256(result_body),
        )
        control = build_versioned_writer_control(
            candidate=candidate,
            result=write,
            published_at=record["created_at"],
        )
        control_key = {
            "PK": ledger_pk(self.config.run_id),
            "SK": ledger_sk(
                "glm52_task12_versioned_writer_control_v1",
                activation_id=self.config.activation_id,
                generation=request["generation"],
                writer_kind="TerminalV2",
            ),
        }
        control_item = {**control_key, **control}
        try:
            ddb = self.clients.dynamodb.put_item(
                TableName=self.config.ledger_table_name,
                Item=encode_item(control_item),
                ConditionExpression=(
                    "attribute_not_exists(#pk) AND "
                    "attribute_not_exists(#sk)"
                ),
                ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
                ReturnConsumedCapacity="NONE",
            )
            control_request = str(
                _meta(ddb, "TerminalV2 version control")["RequestId"]
            )
        except Exception:  # noqa: BLE001 - ambiguous write requires readback
            readback = self.clients.dynamodb.get_item(
                TableName=self.config.ledger_table_name,
                Key=encode_item(control_key),
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
            control_request = str(
                _meta(
                    readback, "TerminalV2 version control readback"
                )["RequestId"]
            )
            item = readback.get("Item")
            if type(item) is not dict or decode_item(item) != control_item:
                raise SupportTerminalV2Error(
                    "TerminalV2 version control adopted foreign bytes"
                )
        body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_terminal_v2_result_v1"
            ),
            "outcome": "SUCCEEDED",
            "activation_id": self.config.activation_id,
            "generation": request["generation"],
            "caller_function_version_arn": request[
                "caller_function_version_arn"
            ],
            "caller_state_machine_version_arn": request[
                "caller_state_machine_version_arn"
            ],
            "caller_execution_arn": request["caller_execution_arn"],
            "support_state_identity_sha256": request[
                "support_state_identity_sha256"
            ],
            "coordinate": key,
            "object_version_id": version_id,
            "body_identity_sha256": record["canonical_body_sha256"],
            "file_sha256": candidate.file_sha256,
            "write_request_ids": [
                *deployment_requests,
                index_request,
                phase_request,
                action_request,
                *write_request_ids,
                control_request,
            ],
        }
        return {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }


def build_aws_support_terminal_v2_services(
    *,
    environment: Mapping[str, str] | None = None,
    clients: SupportTerminalV2Clients | None = None,
) -> AwsSupportTerminalV2Services:
    """Build the exact zero-retry AWS boundary for TerminalV2."""

    env = os.environ if environment is None else environment
    config = SupportTerminalV2Config(
        account_id=env["GLM52_ACCOUNT_ID"],
        run_id=env["GLM52_RUN_ID"],
        activation_id=env["GLM52_ACTIVATION_ID"],
        ledger_table_name=env[
            "GLM52_TASK12_DEPLOYMENT_TABLE_NAME"
        ],
        campaign_bucket=env["GLM52_CAMPAIGN_BUCKET"],
    )
    if (
        config.account_id != "246813579024"
        or config.run_id != "glm52-sky-20260724"
        or not config.activation_id
        or not config.ledger_table_name
        or not config.campaign_bucket.endswith(
            "-246813579024-us-west-2"
        )
    ):
        raise SupportTerminalV2Error(
            "TerminalV2 AWS coordinates are not exact"
        )
    if clients is None:
        import boto3
        from botocore.config import Config

        client_config = Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        )
        clients = SupportTerminalV2Clients(
            s3=boto3.client(
                "s3",
                region_name="us-west-2",
                config=client_config,
            ),
            dynamodb=boto3.client(
                "dynamodb",
                region_name="us-west-2",
                config=client_config,
            ),
            stepfunctions=boto3.client(
                "stepfunctions",
                region_name="us-west-2",
                config=client_config,
            ),
        )
    if type(clients) is not SupportTerminalV2Clients:
        raise TypeError("TerminalV2 AWS clients are not exact")
    return AwsSupportTerminalV2Services(
        config=config,
        clients=clients,
    )


def main(
    event: object,
    context: object,
    *,
    services: AwsSupportTerminalV2Services | None = None,
) -> Mapping[str, object]:
    request = _parse(event)
    invoked = getattr(context, "invoked_function_arn", None)
    if (
        type(invoked) is not str
        or _TERMINAL_VERSION.fullmatch(invoked) is None
    ):
        raise SupportTerminalV2Error(
            "TerminalV2 invoked version is not exact"
        )
    if services is None:
        services = build_aws_support_terminal_v2_services()
    return services.execute(request, invoked_function_arn=invoked)


__all__ = [
    "AwsSupportTerminalV2Services",
    "SupportTerminalV2Clients",
    "SupportTerminalV2Config",
    "SupportTerminalV2Error",
    "build_aws_support_terminal_v2_services",
    "main",
]
