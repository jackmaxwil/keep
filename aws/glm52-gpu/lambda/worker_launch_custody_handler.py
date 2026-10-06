"""Production-capable, dependency-injected Task 9 effect adapters.

The module performs no SDK or network work at import time.  Runtime packaging
injects one-attempt clients and the exact ledger-backed authority readers.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import ssl
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Tuple

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import (
    DynamoLedgerAdapter,
    LedgerKey,
    TransactionResolution,
    WriteOutcome,
)
from glm52_enforcement.launch_custody import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    AmbiguousRunInstances,
    AttemptPermit,
    LaunchContext,
    LaunchCustodyError,
    LaunchParameters,
    LaunchResult,
    LiabilitySettlementCoordinator,
    LiabilityWatcher,
    PositiveRunInstancesRejection,
    SameTokenCompleter,
    TerminationPermit,
    WatchInput,
    WorkerLaunchIntent,
    build_deterministic_client_token,
    validate_launch_parameters,
)


LAUNCH_RELAY_HOST = "glm52-launch-relay.internal"
LAUNCH_RELAY_PORT = 18444
LAUNCH_RELAY_PATH = "/jobs/launch"
ALLOWED_EFFECT_OPERATIONS = (
    "sts:GetCallerIdentity",
    "ec2:RunInstances",
    "ec2:TerminateInstances",
)


class Task3LaunchCustodyStore:
    """Production Task 9 store over the closed Task 3 DynamoDB adapter.

    The injected authority may construct only the named, frozen Task 3 plans
    requested below.  It never receives a generic transaction method, and the
    Task 3 adapter revalidates every record, transition, nonce, activation
    binding, and exact readback before this store accepts a result.
    """

    def __init__(
        self,
        *,
        ledger: DynamoLedgerAdapter,
        ledger_plan_authority: object,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(ledger, DynamoLedgerAdapter):
            raise LaunchCustodyError("Task 3 ledger adapter is not production")
        if not callable(clock):
            raise LaunchCustodyError("Task 9 store clock is absent")
        self._ledger = ledger
        self._authority = ledger_plan_authority
        self._clock = clock
        self._attempt_context = {}
        self._settlement_coordinator: Optional[
            LiabilitySettlementCoordinator
        ] = None

    @staticmethod
    def _worker_key(activation_id: str, ordinal: int) -> LedgerKey:
        if (
            type(activation_id) is not str
            or not activation_id
            or type(ordinal) is not int
            or ordinal <= 0
        ):
            raise LaunchCustodyError("worker launch key is invalid")
        return LedgerKey(
            RUN_ID,
            "ACTIVATION#"
            + activation_id
            + "#WORKER_LAUNCH#%08d" % ordinal,
        )

    @staticmethod
    def _liability_key(activation_id: str, ordinal: int) -> LedgerKey:
        if (
            type(activation_id) is not str
            or not activation_id
            or type(ordinal) is not int
            or ordinal <= 0
        ):
            raise LaunchCustodyError("worker liability key is invalid")
        return LedgerKey(
            RUN_ID,
            "ACTIVATION#"
            + activation_id
            + "#WORKER_LAUNCH_LIABILITY#%08d" % ordinal,
        )

    @staticmethod
    def _settlement_key(activation_id: str, ordinal: int) -> LedgerKey:
        if (
            type(activation_id) is not str
            or not activation_id
            or type(ordinal) is not int
            or ordinal <= 0
        ):
            raise LaunchCustodyError("worker settlement key is invalid")
        return LedgerKey(
            RUN_ID,
            "ACTIVATION#"
            + activation_id
            + "#WORKER_LAUNCH_LIABILITY_SETTLEMENT#%08d" % ordinal,
        )

    def _authority_call(
        self, name: str, *args: object, **kwargs: object
    ) -> object:
        method = getattr(self._authority, name, None)
        if not callable(method):
            raise LaunchCustodyError(
                "production ledger authority lacks " + name
            )
        return method(*args, **kwargs)

    def _closed_plan(
        self,
        name: str,
        expected_keys: set,
        *args: object,
        **kwargs: object,
    ) -> Mapping[str, object]:
        value = self._authority_call(name, *args, **kwargs)
        if type(value) is not dict or set(value) != expected_keys:
            raise LaunchCustodyError(name + " plan is not closed")
        return value

    @staticmethod
    def _live(
        result: object, label: str
    ) -> Tuple[Mapping[str, object], ...]:
        if (
            not isinstance(result, TransactionResolution)
            or result.outcome is not WriteOutcome.EXACT_LIVE_OWNER_COMMIT
            or type(result.records) is not tuple
            or not result.records
        ):
            raise LaunchCustodyError(label + " was not an exact live commit")
        return result.records

    def _required_read(
        self, key: LedgerKey, record_type: str
    ) -> Mapping[str, object]:
        record = self._ledger.read_consistent_item(
            key=key, record_type=record_type
        )
        if record is None:
            raise LaunchCustodyError(
                "required " + record_type + " record is absent"
            )
        return record

    def _query(
        self, activation_id: str, family: str, record_type: str
    ) -> Tuple[Mapping[str, object], ...]:
        return self._ledger.query_activation_family(
            run_id=RUN_ID,
            sort_key_prefix=(
                "ACTIVATION#" + activation_id + "#" + family + "#"
            ),
            record_type=record_type,
        )

    def _prior_settlement(
        self, activation_id: str, ordinal: int
    ) -> Optional[Mapping[str, object]]:
        if ordinal == 1:
            return None
        return self._required_read(
            self._settlement_key(activation_id, ordinal - 1),
            "glm52_production_worker_launch_liability_settlement",
        )

    def _intent(
        self, record: Mapping[str, object]
    ) -> WorkerLaunchIntent:
        activation_id = record.get("activation_id")
        ordinal = record.get("allocation_ordinal")
        if type(activation_id) is not str or type(ordinal) is not int:
            raise LaunchCustodyError("worker launch coordinates are invalid")
        prior_settlement = self._prior_settlement(
            activation_id, ordinal
        )
        value = self._authority_call(
            "project_worker_launch",
            record,
            prior_settlement,
        )
        if not isinstance(value, WorkerLaunchIntent):
            raise LaunchCustodyError("worker launch projection is not typed")
        validate_launch_parameters(
            value.launch_parameters, value.launch_parameters.authority
        )
        expected_token = build_deterministic_client_token(
            value.launch_parameters.authority,
            value.allocation_ordinal,
            value.launch_parameters,
        )
        bindings = {
            "account_id": value.account_id,
            "region": value.region,
            "run_id": value.run_id,
            "campaign_identity_sha256": value.campaign_identity_sha256,
            "activation_id": value.activation_id,
            "activation_ordinal": value.activation_ordinal,
            "generation": value.generation,
            "sky_action_key": value.action_key,
            "allocation_ordinal": value.allocation_ordinal,
            "ec2_client_token": value.ec2_client_token,
            "launch_parameters_sha256": value.launch_parameters_sha256,
            "expected_worker_tags_sha256": (
                value.expected_worker_tags_sha256
            ),
            "prior_worker_launch_identity_sha256": (
                value.prior_worker_launch_identity_sha256
            ),
            "prior_instance_terminal_identity_sha256": (
                value.prior_instance_terminal_identity_sha256
            ),
            "prior_spend_allocation_close_identity_sha256": (
                value.prior_spend_allocation_close_identity_sha256
            ),
            "owner_invocation_nonce_sha256": (
                value.owner_invocation_nonce_sha256
            ),
            "owner_nonce_ciphertext_sha256": (
                value.owner_nonce_ciphertext_sha256
            ),
            "prepared_at": value.prepared_at,
            "prepared_journal_entry_sha256": (
                value.prepared_journal_entry_sha256
            ),
            "ddb_committed_journal_entry_sha256": (
                value.ddb_committed_journal_entry_sha256
            ),
            "state": value.state,
            "send_stage": value.send_stage,
            "possibly_sent_at": value.possibly_sent_at,
            "gpu_liability_reserve_ledger_identity_sha256": (
                value.gpu_liability_reserve_ledger_identity_sha256
            ),
        }
        prior_identity = (
            None
            if prior_settlement is None
            else prior_settlement.get("canonical_body_sha256")
        )
        if (
            any(record.get(field) != item for field, item in bindings.items())
            or value.launch_parameters.canonical_identity_sha256
            != record.get("launch_parameters_sha256")
            or expected_token != record.get("ec2_client_token")
            or value.prior_liability_settlement_identity_sha256
            != prior_identity
            or type(value.canonical_identity_sha256) is not str
            or len(value.canonical_identity_sha256) != 64
        ):
            raise LaunchCustodyError("worker launch projection drifted")
        return value

    def allocate_context(
        self, authority: object
    ) -> LaunchContext:
        activation_id = getattr(authority, "activation_id", None)
        if type(activation_id) is not str or not activation_id:
            raise LaunchCustodyError("launch allocation authority is invalid")
        launches = self._query(
            activation_id,
            "WORKER_LAUNCH",
            "glm52_production_worker_launch",
        )
        liabilities = self._query(
            activation_id,
            "WORKER_LAUNCH_LIABILITY",
            "glm52_production_worker_launch_liability",
        )
        settlements = self._query(
            activation_id,
            "WORKER_LAUNCH_LIABILITY_SETTLEMENT",
            "glm52_production_worker_launch_liability_settlement",
        )
        unsettled = [
            item
            for item in liabilities
            if item.get("state")
            not in {
                "SETTLED_NO_INSTANCE_REJECTED",
                "SETTLED_INSTANCE_CLOSED",
            }
        ]
        if not launches:
            ordinal = 1
            prior = (None, None, None, None)
        else:
            latest = max(
                launches, key=lambda item: item["allocation_ordinal"]
            )
            if latest["state"] == "PREPARED_NOT_SENT":
                ordinal = latest["allocation_ordinal"]
                prior = (
                    latest["prior_worker_launch_identity_sha256"],
                    latest["prior_instance_terminal_identity_sha256"],
                    latest[
                        "prior_spend_allocation_close_identity_sha256"
                    ],
                    (
                        None
                        if ordinal == 1
                        else self._prior_settlement(
                            activation_id, ordinal
                        )["canonical_body_sha256"]
                    ),
                )
            else:
                ordinal = latest["allocation_ordinal"] + 1
                matching = [
                    item
                    for item in settlements
                    if item["allocation_ordinal"]
                    == latest["allocation_ordinal"]
                ]
                prior = (
                    canonical_sha256(latest),
                    latest["instance_terminal_identity_sha256"],
                    latest["spend_allocation_close_identity_sha256"],
                    (
                        matching[0]["canonical_body_sha256"]
                        if len(matching) == 1
                        else None
                    ),
                )
        return LaunchContext(
            allocation_ordinal=ordinal,
            prior_worker_launch_identity_sha256=prior[0],
            prior_instance_terminal_identity_sha256=prior[1],
            prior_spend_allocation_close_identity_sha256=prior[2],
            prior_liability_settlement_identity_sha256=prior[3],
            no_unsettled_liability=not unsettled,
        )

    def put_prepared(
        self, intent: WorkerLaunchIntent, raw_owner_nonce: bytes
    ) -> WorkerLaunchIntent:
        plan = self._closed_plan(
            "plan_put_prepared",
            {
                "worker_launch",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            intent,
            raw_owner_nonce,
        )
        records = self._live(
            self._ledger.put_worker_launch_prepared(**plan),
            "prepared worker launch",
        )
        if len(records) != 1:
            raise LaunchCustodyError("prepared worker readback count drifted")
        return self._intent(records[0])

    def exact_read_launch(
        self, activation_id: str, allocation_ordinal: int
    ) -> WorkerLaunchIntent:
        return self._intent(
            self._required_read(
                self._worker_key(activation_id, allocation_ordinal),
                "glm52_production_worker_launch",
            )
        )

    def bind_committed_wal(
        self,
        intent: WorkerLaunchIntent,
        committed_sha256: str,
        raw_owner_nonce: bytes,
    ) -> WorkerLaunchIntent:
        plan = self._closed_plan(
            "plan_bind_committed_wal",
            {
                "worker_launch",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            intent,
            committed_sha256,
            raw_owner_nonce,
        )
        records = self._live(
            self._ledger.bind_worker_launch_committed_wal(**plan),
            "committed WAL binding",
        )
        if len(records) != 1:
            raise LaunchCustodyError("committed WAL readback count drifted")
        return self._intent(records[0])

    def commit_possibly_sent(
        self,
        intent: WorkerLaunchIntent,
        reserve: Mapping[str, object],
        raw_owner_nonce: bytes,
    ) -> Tuple[WorkerLaunchIntent, Mapping[str, object]]:
        plan = self._closed_plan(
            "plan_commit_possibly_sent",
            {
                "index",
                "reserve_checks",
                "worker_launch",
                "liability",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            intent,
            reserve,
            raw_owner_nonce,
        )
        records = self._live(
            self._ledger.commit_worker_launch_possibly_sent(**plan),
            "possibly-sent worker launch",
        )
        if len(records) != 3:
            raise LaunchCustodyError("possibly-sent readback count drifted")
        return self._intent(records[1]), records[2]

    def wait_watching(
        self, activation_id: str, allocation_ordinal: int
    ) -> Mapping[str, object]:
        return self._required_read(
            self._liability_key(activation_id, allocation_ordinal),
            "glm52_production_worker_launch_liability",
        )

    def signal_ready(
        self, activation_id: str, allocation_ordinal: int
    ) -> None:
        liability = self.wait_watching(
            activation_id, allocation_ordinal
        )
        result = self._authority_call("signal_watcher_ready", liability)
        if result is not None:
            raise LaunchCustodyError("watcher ready signal widened")

    def poll_result(
        self, activation_id: str, allocation_ordinal: int
    ) -> LaunchResult:
        record = self._required_read(
            self._worker_key(activation_id, allocation_ordinal),
            "glm52_production_worker_launch",
        )
        states = {
            "INSTANCE_OBSERVED": "OBSERVED",
            "ALLOCATION_OPEN": "OBSERVED",
            "INSTANCE_TERMINAL": "OBSERVED",
            "ALLOCATION_CLOSED": "OBSERVED",
            "REJECTED_NO_INSTANCE": "POSITIVE_REJECTION",
            "MULTIPLE_INSTANCE_TOKEN_INCIDENT": "INCIDENT",
            "UNRESOLVED_LAUNCH_INCIDENT": "AMBIGUOUS",
        }
        classification = states.get(record["state"])
        if classification is None:
            raise LaunchCustodyError("launch result is not yet terminal/observed")
        return LaunchResult(
            state=record["state"],
            instance_ids=tuple(record["observed_instance_ids"]),
            classification=classification,
        )

    def read_completion_authority(
        self, activation_id: str, allocation_ordinal: int
    ) -> Mapping[str, object]:
        worker, liability = self._ledger.read_coherent(
            items=(
                (
                    self._worker_key(activation_id, allocation_ordinal),
                    "glm52_production_worker_launch",
                ),
                (
                    self._liability_key(activation_id, allocation_ordinal),
                    "glm52_production_worker_launch_liability",
                ),
            )
        )
        parameters = self._authority_call(
            "exact_launch_parameters", worker
        )
        if not isinstance(parameters, LaunchParameters):
            raise LaunchCustodyError("exact launch parameters are absent")
        validate_launch_parameters(parameters, parameters.authority)
        current_owner = self._authority_call(
            "liability_owner_is_current", liability, self._clock()
        )
        if type(current_owner) is not bool:
            raise LaunchCustodyError("liability owner result is not exact")
        if (
            worker["ec2_client_token"]
            != build_deterministic_client_token(
                parameters.authority,
                allocation_ordinal,
                parameters,
            )
            or worker["launch_parameters_sha256"]
            != parameters.canonical_identity_sha256
            or liability["ec2_client_token"] != worker["ec2_client_token"]
        ):
            raise LaunchCustodyError("completion launch bytes drifted")
        return {
            "activation_id": activation_id,
            "allocation_ordinal": allocation_ordinal,
            "state": liability["state"],
            "same_token_completion_attempts": liability[
                "same_token_completion_attempts"
            ],
            "possibly_sent_at": worker["possibly_sent_at"],
            "ec2_client_token": worker["ec2_client_token"],
            "launch_parameters": parameters,
            "current_owner": current_owner,
            "reserve_held": (
                liability[
                    "gpu_liability_reserve_ledger_identity_sha256"
                ]
                is not None
                and liability[
                    "gpu_liability_reserve_release_identity_sha256"
                ]
                is None
            ),
        }

    def consume_attempt(
        self, activation_id: str, allocation_ordinal: int
    ) -> AttemptPermit:
        plan = self._closed_plan(
            "plan_consume_same_token_attempt",
            {
                "action_name",
                "index",
                "liability",
                "action",
                "post_terminal_allocation",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            activation_id,
            allocation_ordinal,
        )
        records = self._live(
            self._ledger.consume_liability_action(**plan),
            "same-token action consumption",
        )
        if len(records) != 3:
            raise LaunchCustodyError("same-token readback count drifted")
        liability, action = records[1], records[2]
        identity = canonical_sha256(action)
        self._attempt_context[identity] = {
            "activation_id": activation_id,
            "allocation_ordinal": allocation_ordinal,
            "raw_owner_nonce": plan["raw_owner_nonce"],
            "liability": liability,
            "action": action,
        }
        return AttemptPermit(
            may_send=True,
            attempt=action["attempt"],
            action_identity_sha256=identity,
        )

    def _record_attempt(
        self,
        permit: AttemptPermit,
        classification: str,
        evidence: object,
    ) -> None:
        context = self._attempt_context.pop(
            permit.action_identity_sha256, None
        )
        if type(context) is not dict:
            raise LaunchCustodyError("consumed attempt context is absent")
        plan = self._closed_plan(
            "plan_complete_worker_launch_attempt",
            {
                "index",
                "worker_launch",
                "liability",
                "liability_action",
                "classification",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            context,
            permit,
            classification,
            evidence,
        )
        records = self._live(
            self._ledger.complete_worker_launch_attempt(**plan),
            "worker launch attempt result",
        )
        if len(records) != 4:
            raise LaunchCustodyError("attempt result readback count drifted")

    def record_direct(
        self, permit: AttemptPermit, response: Mapping[str, object]
    ) -> None:
        self._record_attempt(permit, "DIRECT_SUCCESS", response)

    def record_ambiguous(
        self, permit: AttemptPermit, evidence_identity_sha256: str
    ) -> None:
        self._record_attempt(
            permit, "AMBIGUOUS", evidence_identity_sha256
        )

    def record_rejection(
        self, permit: AttemptPermit, evidence_identity_sha256: str
    ) -> None:
        self._record_attempt(
            permit, "POSITIVE_REJECTION", evidence_identity_sha256
        )

    def _watch_authority(
        self, liability: Mapping[str, object]
    ) -> Mapping[str, object]:
        worker = self._required_read(
            self._worker_key(
                liability["activation_id"],
                liability["allocation_ordinal"],
            ),
            "glm52_production_worker_launch",
        )
        work = self._authority_call(
            "exact_work_authority", worker, liability
        )
        if (
            type(work) is not dict
            or set(work)
            != {"work_authorized", "authorized_instance_id"}
        ):
            raise LaunchCustodyError("work authority is not closed")
        return {
            "activation_id": liability["activation_id"],
            "allocation_ordinal": liability["allocation_ordinal"],
            "ec2_client_token": liability["ec2_client_token"],
            "expected_worker_tags_sha256": liability[
                "expected_worker_tags_sha256"
            ],
            "state": liability["state"],
            "watch_started_at": liability["watch_started_at"],
            "next_scan_at": liability["next_scan_at"],
            "current_owner": True,
            "settlement_identity_sha256": liability[
                "settlement_identity_sha256"
            ],
            "work_authorized": work["work_authorized"],
            "authorized_instance_id": work["authorized_instance_id"],
        }

    def acquire_or_takeover(
        self, now: datetime, *, deadline: datetime
    ) -> Mapping[str, object]:
        index = self._required_read(
            LedgerKey(RUN_ID, "ACTIVATION_INDEX"),
            "glm52_production_activation_index",
        )
        activation_id = index["current_activation_id"]
        liabilities = self._query(
            activation_id,
            "WORKER_LAUNCH_LIABILITY",
            "glm52_production_worker_launch_liability",
        )
        candidates = [
            item
            for item in liabilities
            if item["state"]
            not in {
                "UNOWNED_NOT_ACTIONABLE",
                "SETTLED_NO_INSTANCE_REJECTED",
                "SETTLED_INSTANCE_CLOSED",
            }
            or item["state"] == "UNOWNED_NOT_ACTIONABLE"
        ]
        selected = self._authority_call(
            "select_watch_liability", index, tuple(candidates), now, deadline
        )
        if type(selected) is not dict or selected not in liabilities:
            raise LaunchCustodyError("watch liability selection is foreign")
        if selected["state"] == "UNOWNED_NOT_ACTIONABLE":
            plan = self._closed_plan(
                "plan_acquire_liability",
                {
                    "index",
                    "liability",
                    "domain",
                    "operation_identity_sha256",
                    "raw_owner_nonce",
                },
                index,
                selected,
                now,
                deadline,
            )
            records = self._live(
                self._ledger.acquire_liability_owner(**plan),
                "liability owner acquisition",
            )
            selected = records[1]
        else:
            current = self._authority_call(
                "liability_owner_is_current", selected, now
            )
            if type(current) is not bool:
                raise LaunchCustodyError("liability owner result is not exact")
            if not current:
                plan = self._closed_plan(
                    "plan_takeover_liability",
                    {
                        "guards",
                        "owner",
                        "domain",
                        "operation_identity_sha256",
                        "raw_owner_nonce",
                    },
                    index,
                    selected,
                    now,
                    deadline,
                )
                records = self._live(
                    self._ledger.commit_owner_takeover(**plan),
                    "liability owner takeover",
                )
                selected = records[-1]
        return self._watch_authority(selected)

    def record_scan(
        self,
        authority: Mapping[str, object],
        evidence: Mapping[str, object],
        next_scan_at: str,
        *,
        deadline: datetime,
    ) -> None:
        plan = self._closed_plan(
            "plan_record_scan",
            {
                "index",
                "liability",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            authority,
            evidence,
            next_scan_at,
            deadline,
        )
        self._live(
            self._ledger.record_liability_evidence(**plan),
            "liability scan evidence",
        )

    def record_incident(
        self,
        authority: Mapping[str, object],
        incident_kind: str,
        *,
        deadline: datetime,
    ) -> None:
        plan = self._closed_plan(
            "plan_record_incident",
            {
                "index",
                "liability",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            authority,
            incident_kind,
            deadline,
        )
        self._live(
            self._ledger.record_liability_incident(**plan),
            "liability incident",
        )

    def open_late_allocation(
        self,
        authority: Mapping[str, object],
        instance: Mapping[str, object],
        terminal_v2_published: bool,
        *,
        deadline: datetime,
    ) -> None:
        plan = self._closed_plan(
            "plan_open_late_allocation",
            {
                "action_name",
                "index",
                "liability",
                "action",
                "post_terminal_allocation",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            authority,
            instance,
            terminal_v2_published,
            deadline,
        )
        self._live(
            self._ledger.consume_liability_action(**plan),
            "late allocation discovery",
        )

    def consume_termination(
        self,
        authority: Mapping[str, object],
        instance_id: str,
        *,
        deadline: datetime,
    ) -> TerminationPermit:
        plan = self._closed_plan(
            "plan_consume_termination",
            {
                "action_name",
                "index",
                "liability",
                "action",
                "post_terminal_allocation",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
            },
            authority,
            instance_id,
            deadline,
        )
        records = self._live(
            self._ledger.consume_liability_action(**plan),
            "termination action consumption",
        )
        liability, action = records[1], records[2]
        return TerminationPermit(
            may_terminate=True,
            instance_id=instance_id,
            call_count=action["attempt"],
            window_started_at=liability["scan_period_started_at"],
            action_identity_sha256=canonical_sha256(action),
        )

    def reconcile_instance(
        self,
        authority: Mapping[str, object],
        instance_id: str,
        *,
        deadline: datetime,
    ) -> None:
        plans = self._authority_call(
            "plan_reconcile_instance",
            authority,
            instance_id,
            deadline,
        )
        if type(plans) is not dict or set(plans) != {
            "worker_launch",
            "liability_evidence",
            "post_terminal_action",
        }:
            raise LaunchCustodyError("instance reconciliation is not closed")
        worker_plan = plans["worker_launch"]
        if worker_plan is not None:
            if type(worker_plan) is not dict:
                raise LaunchCustodyError("worker reconciliation plan drifted")
            self._live(
                self._ledger.record_worker_launch_reconciliation(
                    **worker_plan
                ),
                "worker reconciliation",
            )
        evidence_plan = plans["liability_evidence"]
        if evidence_plan is not None:
            if type(evidence_plan) is not dict:
                raise LaunchCustodyError("liability evidence plan drifted")
            self._live(
                self._ledger.record_liability_evidence(**evidence_plan),
                "liability reconciliation",
            )
        action_plan = plans["post_terminal_action"]
        if action_plan is not None:
            if type(action_plan) is not dict:
                raise LaunchCustodyError("allocation action plan drifted")
            self._live(
                self._ledger.consume_liability_action(**action_plan),
                "post-terminal reconciliation",
            )

    def install_settlement_coordinator(
        self, coordinator: LiabilitySettlementCoordinator
    ) -> None:
        if (
            not isinstance(coordinator, LiabilitySettlementCoordinator)
            or self._settlement_coordinator is not None
        ):
            raise LaunchCustodyError("settlement coordinator install drifted")
        self._settlement_coordinator = coordinator

    def settle_if_complete(
        self,
        authority: Mapping[str, object],
        *,
        deadline: datetime,
    ) -> Optional[Mapping[str, object]]:
        view = self.read_complete_view(
            authority["activation_id"], authority["allocation_ordinal"]
        )
        ready = self._authority_call(
            "settlement_is_ready", authority, view, deadline
        )
        if type(ready) is not bool:
            raise LaunchCustodyError("settlement readiness is not exact")
        if not ready:
            return None
        if self._settlement_coordinator is None:
            raise LaunchCustodyError("settlement coordinator is absent")
        return self._settlement_coordinator.settle(
            authority["activation_id"], authority["allocation_ordinal"]
        )

    def read_complete_view(
        self, activation_id: str, allocation_ordinal: int
    ) -> Mapping[str, object]:
        read_set = self._authority_call(
            "complete_view_read_set", activation_id, allocation_ordinal
        )
        if (
            type(read_set) is not tuple
            or not read_set
            or any(
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], LedgerKey)
                or type(item[1]) is not str
                for item in read_set
            )
        ):
            raise LaunchCustodyError("complete-view read set is not closed")
        records = self._ledger.read_coherent(items=read_set)
        post = self._query(
            activation_id,
            "POST_TERMINAL_ALLOCATION",
            "glm52_production_post_terminal_allocation",
        )
        view = self._authority_call(
            "project_complete_view",
            records,
            post,
            activation_id,
            allocation_ordinal,
        )
        if type(view) is not dict:
            raise LaunchCustodyError("complete-view projection is malformed")
        return view

    def transact_settle_once(
        self, settlement: Mapping[str, object]
    ) -> object:
        plan = self._closed_plan(
            "plan_settlement",
            {
                "index",
                "liability_action",
                "liability",
                "settlement",
                "domain",
                "operation_identity_sha256",
                "raw_owner_nonce",
                "duplicate_settlement_identity_sha256",
            },
            settlement,
        )
        result = self._ledger.settle_worker_launch_liability(**plan)
        if result.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT:
            if len(result.records) != 4 or result.records[-1] != settlement:
                raise LaunchCustodyError("settlement live readback drifted")
            return settlement
        if result.outcome is WriteOutcome.READBACK_UNAVAILABLE:
            return {"classification": "AMBIGUOUS"}
        if (
            result.outcome is WriteOutcome.EXACT_DURABLE_ADOPTION
            and len(result.records) == 4
            and result.records[-1] == settlement
        ):
            return settlement
        raise LaunchCustodyError("settlement transaction was not exact")

    def coherent_read_settlement(
        self, key: str
    ) -> Optional[Mapping[str, object]]:
        prefix = "WORKER_LAUNCH_LIABILITY_SETTLEMENT#"
        if type(key) is not str or not key.startswith(prefix):
            raise LaunchCustodyError("settlement read key is invalid")
        tail = key[len(prefix) :]
        activation_id, separator, ordinal_text = tail.rpartition("#")
        if (
            not separator
            or len(ordinal_text) != 8
            or not ordinal_text.isdigit()
        ):
            raise LaunchCustodyError("settlement read key is malformed")
        return self._ledger.read_consistent_item(
            key=self._settlement_key(
                activation_id, int(ordinal_text)
            ),
            record_type=(
                "glm52_production_worker_launch_liability_settlement"
            ),
        )


class PublishedFunctionIdentityGuard:
    """Authenticate a qualified Lambda ARN and its exact resource policy."""

    def __init__(
        self,
        *,
        expected_published_version_arn: str,
        expected_resource_policy_identity_sha256: str,
        identity_reader: object,
    ) -> None:
        if (
            type(expected_published_version_arn) is not str
            or re.fullmatch(
                (
                    r"arn:aws:lambda:us-west-2:246813579024:"
                    r"function:[A-Za-z0-9_-]+:[1-9][0-9]*"
                ),
                expected_published_version_arn,
            )
            is None
            or type(expected_resource_policy_identity_sha256) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                expected_resource_policy_identity_sha256,
            )
            is None
        ):
            raise LaunchCustodyError("published function identity is invalid")
        self._arn = expected_published_version_arn
        self._policy = expected_resource_policy_identity_sha256
        self._reader = identity_reader

    def prove(self, context: object) -> None:
        invoked = getattr(context, "invoked_function_arn", None)
        read = getattr(self._reader, "exact_read", None)
        if invoked != self._arn or not callable(read):
            raise LaunchCustodyError("published function invocation is absent")
        identity = read()
        if identity != {
            "published_version_arn": self._arn,
            "resource_policy_identity_sha256": self._policy,
        }:
            raise LaunchCustodyError(
                "published function resource policy is foreign"
            )


class Task9EffectDispatcher:
    """Qualified Lambda handler core with closed, non-authoritative inputs."""

    def __init__(
        self,
        *,
        function_identity_guard: object,
        same_token_completer: object,
        liability_watcher: object,
    ) -> None:
        self._identity = function_identity_guard
        self._completer = same_token_completer
        self._watcher = liability_watcher

    def handle(
        self, event: Mapping[str, object], context: object
    ) -> Mapping[str, object]:
        if type(event) is not dict or type(event.get("mode")) is not str:
            raise LaunchCustodyError("effect dispatch event is not closed")
        mode = event["mode"]
        if mode == "SAME_TOKEN_COMPLETE":
            if set(event) != {
                "mode",
                "activation_id",
                "allocation_ordinal",
            }:
                raise LaunchCustodyError("same-token dispatch is not closed")
            input_value = {
                "activation_id": event["activation_id"],
                "allocation_ordinal": event["allocation_ordinal"],
            }
            prove = getattr(self._identity, "prove", None)
            complete = getattr(self._completer, "complete", None)
            if not callable(prove) or not callable(complete):
                raise LaunchCustodyError("same-token production route is absent")
            prove(context)
            result = complete(input_value)
            return {
                "classification": result.classification,
                "attempt": result.attempt,
                "instance_ids": list(result.instance_ids),
                "evidence_identity_sha256": (
                    result.evidence_identity_sha256
                ),
            }
        if mode == "LIABILITY_WATCH":
            if set(event) not in (
                {"mode"},
                {"mode", "event_accelerator"},
            ):
                raise LaunchCustodyError("watch dispatch is not closed")
            prove = getattr(self._identity, "prove", None)
            run = getattr(self._watcher, "run", None)
            if not callable(prove) or not callable(run):
                raise LaunchCustodyError("watch production route is absent")
            prove(context)
            result = run(
                WatchInput(event_accelerator=event.get("event_accelerator"))
            )
            return {
                "settled": result.settled,
                "instance_ids": list(result.instance_ids),
                "next_scan_at": result.next_scan_at,
                "incident": result.incident,
            }
        raise LaunchCustodyError("effect dispatch mode is not closed")


def _request_id(response: object, label: str) -> str:
    if (
        type(response) is not dict
        or type(response.get("ResponseMetadata")) is not dict
        or response["ResponseMetadata"].get("HTTPStatusCode") != 200
        or type(response["ResponseMetadata"].get("RequestId")) is not str
        or not response["ResponseMetadata"]["RequestId"]
    ):
        raise LaunchCustodyError(label + " response metadata is invalid")
    return str(response["ResponseMetadata"]["RequestId"])


def _classify_ec2_exception(exc: Exception) -> None:
    response = getattr(exc, "response", None)
    if type(response) is dict:
        metadata = response.get("ResponseMetadata")
        error = response.get("Error")
        status = (
            metadata.get("HTTPStatusCode")
            if type(metadata) is dict
            else None
        )
        code = error.get("Code") if type(error) is dict else None
        if (
            type(status) is int
            and 400 <= status < 500
            and type(code) is str
            and bool(code)
        ):
            raise PositiveRunInstancesRejection(code) from exc
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout)):
        raise AmbiguousRunInstances(type(exc).__name__) from exc
    raise AmbiguousRunInstances(type(exc).__name__) from exc


class AwsAccountRegionGuard:
    """STS must be the first operation before every effect."""

    def __init__(self, *, sts: object, region: str) -> None:
        self._sts = sts
        self._region = region

    def prove(self) -> str:
        if self._region != REGION:
            raise LaunchCustodyError("AWS region guard failed")
        method = getattr(self._sts, "get_caller_identity", None)
        if not callable(method):
            raise LaunchCustodyError("STS identity boundary is absent")
        response = method()
        _request_id(response, "STS")
        if response.get("Account") != ACCOUNT_ID:
            raise LaunchCustodyError("AWS account guard failed")
        return str(response["ResponseMetadata"]["RequestId"])


class ZeroRetryEc2RunInstances:
    """One exact EC2 call after STS, with no botocore retry."""

    total_max_attempts = 1

    def __init__(
        self,
        *,
        guard: AwsAccountRegionGuard,
        ec2: object,
    ) -> None:
        self._guard = guard
        self._ec2 = ec2
        config = getattr(getattr(ec2, "meta", None), "config", None)
        retries = getattr(config, "retries", None)
        if type(retries) is not dict or retries.get("total_max_attempts") != 1:
            raise LaunchCustodyError("EC2 client is not configured for one attempt")

    def run_instances(self, **request: object) -> Mapping[str, object]:
        self._guard.prove()
        method = getattr(self._ec2, "run_instances", None)
        if not callable(method):
            raise LaunchCustodyError("EC2 completion effect is absent")
        try:
            response = method(**request)
        except Exception as exc:
            _classify_ec2_exception(exc)
            raise AssertionError("unreachable")
        _request_id(response, "RunInstances")
        return response


class ZeroRetryExactInstanceTerminator:
    """Termination-only retained adapter with exact-instance self-read."""

    total_max_attempts = 1

    def __init__(
        self,
        *,
        guard: AwsAccountRegionGuard,
        ec2: object,
        instance_authority: object,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._guard = guard
        self._ec2 = ec2
        self._authority = instance_authority
        self._clock = clock
        config = getattr(getattr(ec2, "meta", None), "config", None)
        retries = getattr(config, "retries", None)
        if type(retries) is not dict or retries.get("total_max_attempts") != 1:
            raise LaunchCustodyError("EC2 client is not configured for one attempt")

    def terminate(
        self,
        instance_id: str,
        *,
        deadline: datetime,
    ) -> Mapping[str, str]:
        if (
            not isinstance(deadline, datetime)
            or deadline.tzinfo is None
            or self._clock().astimezone(timezone.utc)
            > deadline.astimezone(timezone.utc)
        ):
            raise LaunchCustodyError("termination deadline elapsed before STS")
        self._guard.prove()
        exact_read = getattr(self._authority, "exact_read", None)
        if not callable(exact_read) or exact_read(instance_id) != {
            "instance_id": instance_id,
            "termination_authorized": True,
        }:
            raise LaunchCustodyError("exact-instance termination is unauthorized")
        method = getattr(self._ec2, "terminate_instances", None)
        if not callable(method):
            raise LaunchCustodyError("EC2 termination effect is absent")
        if self._clock().astimezone(timezone.utc) > deadline.astimezone(
            timezone.utc
        ):
            raise LaunchCustodyError("termination deadline elapsed before EC2")
        try:
            response = method(InstanceIds=[instance_id])
        except Exception as exc:
            raise AmbiguousRunInstances(type(exc).__name__) from exc
        request_id = _request_id(response, "TerminateInstances")
        return {
            "request_id": request_id,
            "response_sha256": hashlib.sha256(
                json.dumps(
                    response,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }


class FixedMtlsLaunchRelay:
    """One fixed-host, fixed-port, fixed-path mTLS launch call."""

    def __init__(
        self,
        *,
        connection_factory: Callable[..., object],
        tls_context: ssl.SSLContext,
        expected_peer_certificate_sha256: str,
    ) -> None:
        if (
            type(expected_peer_certificate_sha256) is not str
            or len(expected_peer_certificate_sha256) != 64
        ):
            raise LaunchCustodyError("launch relay peer identity is invalid")
        self._factory = connection_factory
        self._context = tls_context
        self._peer = expected_peer_certificate_sha256

    def send(self, body: bytes) -> Mapping[str, object]:
        if type(body) is not bytes or not body:
            raise LaunchCustodyError("launch relay body is invalid")
        connection = self._factory(
            LAUNCH_RELAY_HOST,
            LAUNCH_RELAY_PORT,
            timeout=12,
            context=self._context,
        )
        try:
            connection.connect()
            certificate = connection.sock.getpeercert(binary_form=True)
            if (
                type(certificate) is not bytes
                or hashlib.sha256(certificate).hexdigest() != self._peer
            ):
                raise LaunchCustodyError(
                    "launch relay peer certificate mismatched"
                )
            connection.request(
                "POST",
                LAUNCH_RELAY_PATH,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = connection.getresponse()
            response_body = response.read(1_048_577)
            if len(response_body) > 1_048_576:
                raise LaunchCustodyError("launch relay response is oversized")
            request_id = response.getheader("x-request-id")
            if type(request_id) is not str or not request_id:
                raise LaunchCustodyError(
                    "launch relay response lacks request identity"
                )
            return {
                "status_code": int(response.status),
                "request_id": request_id,
                "response_sha256": hashlib.sha256(response_body).hexdigest(),
            }
        except LaunchCustodyError:
            raise
        except (TimeoutError, ConnectionError, socket.timeout, OSError) as exc:
            raise TimeoutError("launch relay transport is ambiguous") from exc
        finally:
            connection.close()


def build_one_attempt_clients(
    *, boto3_module: object, botocore_config_type: object
) -> Mapping[str, object]:
    """Build exact-region clients; called only by Lambda runtime bootstrap."""

    config = botocore_config_type(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=15,
    )
    client = getattr(boto3_module, "client", None)
    if not callable(client):
        raise LaunchCustodyError("boto3 client factory is absent")
    return {
        "sts": client("sts", region_name=REGION, config=config),
        "ec2": client("ec2", region_name=REGION, config=config),
        "dynamodb": client("dynamodb", region_name=REGION, config=config),
    }


def build_production_task9_dispatcher(
    *,
    boto3_module: object,
    botocore_config_type: object,
    table_name: str,
    ledger_plan_authority: object,
    expected_published_version_arn: str,
    expected_resource_policy_identity_sha256: str,
    function_identity_reader: object,
    discovery: object,
    instance_authority: object,
    clock: Callable[[], datetime],
) -> Task9EffectDispatcher:
    """Compose the executable published-function route over Task 3."""

    clients = build_one_attempt_clients(
        boto3_module=boto3_module,
        botocore_config_type=botocore_config_type,
    )
    guard = AwsAccountRegionGuard(sts=clients["sts"], region=REGION)
    store = Task3LaunchCustodyStore(
        ledger=DynamoLedgerAdapter(
            client=clients["dynamodb"], table_name=table_name
        ),
        ledger_plan_authority=ledger_plan_authority,
        clock=clock,
    )
    settlement = LiabilitySettlementCoordinator(store=store, clock=clock)
    store.install_settlement_coordinator(settlement)
    completer = SameTokenCompleter(
        store=store,
        ec2=ZeroRetryEc2RunInstances(
            guard=guard,
            ec2=clients["ec2"],
        ),
        clock=clock,
    )
    watcher = LiabilityWatcher(
        store=store,
        discovery=discovery,
        terminator=ZeroRetryExactInstanceTerminator(
            guard=guard,
            ec2=clients["ec2"],
            instance_authority=instance_authority,
            clock=clock,
        ),
        clock=clock,
    )
    identity = PublishedFunctionIdentityGuard(
        expected_published_version_arn=expected_published_version_arn,
        expected_resource_policy_identity_sha256=(
            expected_resource_policy_identity_sha256
        ),
        identity_reader=function_identity_reader,
    )
    return Task9EffectDispatcher(
        function_identity_guard=identity,
        same_token_completer=completer,
        liability_watcher=watcher,
    )


__all__ = [
    "ALLOWED_EFFECT_OPERATIONS",
    "AwsAccountRegionGuard",
    "FixedMtlsLaunchRelay",
    "LAUNCH_RELAY_HOST",
    "LAUNCH_RELAY_PATH",
    "LAUNCH_RELAY_PORT",
    "PublishedFunctionIdentityGuard",
    "Task3LaunchCustodyStore",
    "Task9EffectDispatcher",
    "ZeroRetryEc2RunInstances",
    "ZeroRetryExactInstanceTerminator",
    "build_one_attempt_clients",
    "build_production_task9_dispatcher",
]
