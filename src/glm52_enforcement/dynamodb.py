"""Import-light, no-retry DynamoDB transport boundary for GLM-5.2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import re
from typing import Dict, List, Mapping, Optional, Tuple

from .canonical import canonical_sha256
from .records import ledger_pk, validate_record
from .transitions import validate_owner_takeover, validate_transition


_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class LedgerKey:
    run_id: str
    sort_key: str


@dataclass(frozen=True)
class ExactUpdate:
    key: LedgerKey
    before: Mapping[str, object]
    after: Mapping[str, object]


@dataclass(frozen=True)
class ExactPut:
    key: LedgerKey
    item: Mapping[str, object]


@dataclass(frozen=True)
class ExactCheck:
    key: LedgerKey
    expected: Mapping[str, object]


class WriteOutcome(Enum):
    EXACT_LIVE_OWNER_COMMIT = "EXACT_LIVE_OWNER_COMMIT"
    EXACT_DURABLE_ADOPTION = "EXACT_DURABLE_ADOPTION"
    PRESTATE_OR_ABSENT = "PRESTATE_OR_ABSENT"
    CONDITION_REJECTED = "CONDITION_REJECTED"
    FOREIGN_NONCE = "FOREIGN_NONCE"
    MISMATCHED_COMMIT = "MISMATCHED_COMMIT"
    READBACK_UNAVAILABLE = "READBACK_UNAVAILABLE"


@dataclass(frozen=True)
class TransactionResolution:
    outcome: WriteOutcome
    records: Tuple[Mapping[str, object], ...]
    request_id: Optional[str]
    error_code: Optional[str]
    cancellation_reasons: Tuple[Mapping[str, object], ...]

    @property
    def may_issue_external_side_effect(self) -> bool:
        return self.outcome is WriteOutcome.EXACT_LIVE_OWNER_COMMIT


class LedgerReadError(RuntimeError):
    """A DynamoDB read did not prove a complete closed record."""


class DynamoLedgerAdapter:
    """Closed one-shot DynamoDB ledger adapter."""

    def __init__(self, *, client: object, table_name: str) -> None:
        if type(table_name) is not str or not table_name:
            raise ValueError("table_name must be a nonempty exact string")
        self._client = client
        self._table_name = table_name

    def read_consistent_item(
        self, *, key: LedgerKey, record_type: str
    ) -> Optional[Mapping[str, object]]:
        response = self._client.get_item(
            TableName=self._table_name,
            Key=_encoded_key(key),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        if type(response) is not dict:
            raise LedgerReadError("malformed DynamoDB read response")
        _require_success_metadata(response)
        if "Item" not in response:
            return None
        return _decode_validated_physical_item(
            response["Item"], key=key, record_type=record_type
        )

    def read_coherent(
        self, *, items: Tuple[Tuple[LedgerKey, str], ...]
    ) -> Tuple[Mapping[str, object], ...]:
        if type(items) is not tuple or not items:
            raise ValueError("coherent read requires a nonempty exact tuple")
        if len(items) > 100:
            raise ValueError("coherent read exceeds DynamoDB 100-item limit")
        requests = []
        for item in items:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[1]) is not str
            ):
                raise TypeError("coherent read item is malformed")
            requests.append(
                {
                    "Get": {
                        "TableName": self._table_name,
                        "Key": _encoded_key(item[0]),
                    }
                }
            )
        try:
            response = self._client.transact_get_items(
                TransactItems=requests,
                ReturnConsumedCapacity="NONE",
            )
        except Exception as exc:
            raise LedgerReadError("coherent DynamoDB read failed") from exc
        _require_success_metadata(response)
        responses = response.get("Responses")
        if type(responses) is not list or len(responses) != len(items):
            raise LedgerReadError("coherent read response count mismatch")
        records = []
        for response_item, requested in zip(responses, items):
            if type(response_item) is not dict or set(response_item) != {"Item"}:
                raise LedgerReadError("required coherent item is unavailable")
            records.append(
                _decode_validated_physical_item(
                    response_item["Item"],
                    key=requested[0],
                    record_type=requested[1],
                )
            )
        return tuple(records)

    def query_activation_family(
        self,
        *,
        run_id: str,
        sort_key_prefix: str,
        record_type: str,
    ) -> Tuple[Mapping[str, object], ...]:
        record_families = {
            "glm52_production_worker_launch": "WORKER_LAUNCH",
            "glm52_production_worker_launch_liability": (
                "WORKER_LAUNCH_LIABILITY"
            ),
            "glm52_production_worker_launch_liability_action": (
                "WORKER_LAUNCH_LIABILITY_ACTION"
            ),
            "glm52_production_worker_launch_liability_settlement": (
                "WORKER_LAUNCH_LIABILITY_SETTLEMENT"
            ),
            "glm52_production_post_terminal_allocation": (
                "POST_TERMINAL_ALLOCATION"
            ),
        }
        expected_family = record_families.get(record_type)
        prefix_match = (
            None
            if type(sort_key_prefix) is not str
            else re.fullmatch(
                r"ACTIVATION#([^#]+)#([A-Z0-9_]+)#",
                sort_key_prefix,
            )
        )
        if (
            type(run_id) is not str
            or not run_id
            or prefix_match is None
            or expected_family is None
            or prefix_match.group(2) != expected_family
        ):
            raise ValueError("activation-family query is not closed")
        exclusive_start_key: Optional[Mapping[str, object]] = None
        seen_page_keys = set()
        seen_record_keys = set()
        records: List[Mapping[str, object]] = []
        while True:
            request: Dict[str, object] = {
                "TableName": self._table_name,
                "KeyConditionExpression": (
                    "#pk = :pk AND begins_with(#sk, :prefix)"
                ),
                "ExpressionAttributeNames": {
                    "#pk": "PK",
                    "#sk": "SK",
                },
                "ExpressionAttributeValues": {
                    ":pk": encode_attribute_value(ledger_pk(run_id)),
                    ":prefix": encode_attribute_value(sort_key_prefix),
                },
                "ConsistentRead": True,
                "ScanIndexForward": True,
                "ReturnConsumedCapacity": "NONE",
            }
            if exclusive_start_key is not None:
                request["ExclusiveStartKey"] = dict(exclusive_start_key)
            try:
                response = self._client.query(**request)
            except Exception as exc:
                raise LedgerReadError(
                    "activation-family DynamoDB query failed"
                ) from exc
            _require_success_metadata(response)
            items = response.get("Items")
            if type(items) is not list:
                raise LedgerReadError(
                    "activation-family query items are malformed"
                )
            for physical_item in items:
                decoded = decode_item(physical_item)
                physical_pk = decoded.get("PK")
                physical_sk = decoded.get("SK")
                if (
                    physical_pk != ledger_pk(run_id)
                    or type(physical_sk) is not str
                    or not physical_sk.startswith(sort_key_prefix)
                    or physical_sk in seen_record_keys
                ):
                    raise LedgerReadError(
                        "activation-family query returned foreign records"
                    )
                seen_record_keys.add(physical_sk)
                records.append(
                    _decode_validated_physical_item(
                        physical_item,
                        key=LedgerKey(run_id, physical_sk),
                        record_type=record_type,
                    )
                )
            next_key = response.get("LastEvaluatedKey")
            if next_key is None:
                break
            if type(next_key) is not dict:
                raise LedgerReadError(
                    "activation-family continuation key is malformed"
                )
            decoded_next_key = decode_item(next_key)
            if set(decoded_next_key) != {"PK", "SK"}:
                raise LedgerReadError(
                    "activation-family continuation key is not closed"
                )
            page_identity = canonical_sha256(decoded_next_key)
            if (
                decoded_next_key["PK"] != ledger_pk(run_id)
                or type(decoded_next_key["SK"]) is not str
                or not decoded_next_key["SK"].startswith(sort_key_prefix)
                or page_identity in seen_page_keys
            ):
                raise LedgerReadError(
                    "activation-family continuation key drifted or cycled"
                )
            seen_page_keys.add(page_identity)
            exclusive_start_key = next_key
        return tuple(records)

    def commit_rollover(
        self,
        *,
        index: ExactUpdate,
        rollover: ExactPut,
        control: ExactPut,
        recovery_control: ExactPut,
        finalization_control: ExactPut,
        snapshot_cleanup_control: ExactPut,
        cleanup_heads: Tuple[ExactCheck, ...],
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: Optional[bytes],
        duplicate_operation_identity_sha256: Optional[str] = None,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (
            index,
            rollover,
            control,
            recovery_control,
            finalization_control,
            snapshot_cleanup_control,
        ) + cleanup_heads
        if len(plans) > 100:
            raise ValueError("rollover exceeds DynamoDB transaction item limit")
        _require_exact_plan_type(index, ExactUpdate)
        for item in (
            rollover,
            control,
            recovery_control,
            finalization_control,
            snapshot_cleanup_control,
        ):
            _require_exact_plan_type(item, ExactPut)
        if type(cleanup_heads) is not tuple:
            raise TypeError("cleanup_heads must be an exact tuple")
        for item in cleanup_heads:
            _require_exact_plan_type(item, ExactCheck)
        expected_types = (
            "glm52_production_activation_index",
            "glm52_production_rollover",
            "glm52_production_control",
            "glm52_production_recovery_control",
            "glm52_production_finalization_control",
            "glm52_production_snapshot_cleanup_control",
        )
        for plan, record_type in zip(plans[:6], expected_types):
            if _plan_record_type(plan) != record_type:
                raise ValueError("rollover write-set record type mismatch")
        for check in cleanup_heads:
            if (
                _plan_record_type(check)
                != "glm52_production_snapshot_cleanup_control"
            ):
                raise ValueError("rollover cleanup check family mismatch")
        _validate_rollover_binding(
            index,
            rollover,
            (
                control,
                recovery_control,
                finalization_control,
                snapshot_cleanup_control,
            ),
            require_final_identity=(
                duplicate_operation_identity_sha256 is not None
            ),
        )
        if duplicate_operation_identity_sha256 is not None:
            if raw_owner_nonce is not None:
                raise ValueError("durable adoption forbids a raw owner nonce")
            duplicate_identity = _require_sha256(
                "duplicate_operation_identity_sha256",
                duplicate_operation_identity_sha256,
            )
            if (
                duplicate_identity != operation_identity_sha256
                or rollover.item.get(
                    "cloudformation_operation_identity_sha256"
                )
                != duplicate_identity
            ):
                raise ValueError("rollover duplicate identity mismatch")
            _validate_plans(plans)
            return self._adopt_durable(plans[:6])
        if raw_owner_nonce is None:
            raise ValueError("live rollover requires a raw owner nonce")
        materialized, client_request_token = _materialize_transaction(
            plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((1, "writer_invocation_nonce_sha256"),),
            token_targets=((1, "transaction_client_request_token_sha256"),),
            transaction_hash_targets=((1, "transaction_bytes_sha256"),),
            rollover_binding=(1, (2, 3, 4, 5)),
        )
        _validate_plans(materialized)
        return self._execute_transaction(
            plans=materialized,
            readback_plans=materialized[:6],
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((1, "writer_invocation_nonce_sha256"),),
            allow_durable_adoption=False,
            client_request_token=client_request_token,
        )

    def allocate_execution_epoch(
        self,
        *,
        index: ExactCheck,
        control: ExactUpdate,
        execution: ExactPut,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, control, execution)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_control",
                "glm52_production_execution",
            ),
            (ExactCheck, ExactUpdate, ExactPut),
        )
        _validate_control_revision_update(control)
        if execution.item.get("state") != "START_OWNED":
            raise ValueError("execution allocation must create START_OWNED")
        if (
            control.after.get("active_epoch") != execution.item.get("epoch")
            or control.after.get("active_execution_arn")
            != execution.item.get("expected_execution_arn")
            or control.after.get("active_state_machine_version_arn")
            != execution.item.get("expected_state_machine_version_arn")
            or execution.item.get("owner_control_revision")
            != control.after.get("revision")
        ):
            raise ValueError("execution allocation binding mismatch")
        _require_activation_binding(index.expected, control.after, execution.item)
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((2, "start_owner_invocation_nonce_sha256"),),
            token_targets=(),
        )

    def commit_owner_takeover(
        self,
        *,
        guards: Tuple[ExactCheck, ...],
        owner: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        if type(guards) is not tuple or not guards:
            raise ValueError("owner takeover requires guard checks")
        for guard in guards:
            _require_exact_plan_type(guard, ExactCheck)
        _require_exact_plan_type(owner, ExactUpdate)
        if guards[0].expected.get("record_type") != (
            "glm52_production_activation_index"
        ):
            raise ValueError("owner takeover first guard must be activation index")
        owner_type = _plan_record_type(owner)
        allowed = {
            "glm52_production_recovery_control",
            "glm52_production_finalization_control",
            "glm52_production_snapshot_cleanup_control",
            "glm52_production_worker_launch",
            "glm52_production_worker_launch_liability",
            "glm52_production_post_terminal_allocation",
        }
        if owner_type not in allowed:
            raise ValueError("owner takeover target family is forbidden")
        validate_owner_takeover(owner_type, owner.before, owner.after)
        plans = guards + (owner,)
        owner_index = len(guards)
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((owner_index, "owner_invocation_nonce_sha256"),),
            token_targets=(),
        )

    def commit_recovery_seal(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import RecoverySealPlan

        if type(plan) is not RecoverySealPlan:
            raise TypeError("recovery seal requires a typed exact plan")
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def commit_recovery_progress(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import RecoveryProgressPlan

        if type(plan) is not RecoveryProgressPlan:
            raise TypeError("recovery progress requires a typed exact plan")
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def commit_recovery_handoff_arm(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_live_recovery_handoff import RecoveryHandoffArmPlan

        if type(plan) is not RecoveryHandoffArmPlan or domain != "RECOVERY":
            raise TypeError("recovery handoff arm requires an exact plan")
        _validate_recovery_handoff_transaction(plan, stage="ARM")
        _require_nonce_ownership(
            plan.recovery_control.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plan.transaction_plans,
            readback_plans=plan.transaction_plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(
                (2, "owner_invocation_nonce_sha256"),
                (3, "owner_invocation_nonce_sha256"),
            ),
            token_targets=((3, "arming_transaction_client_request_token_sha256"),),
        )

    def commit_recovery_handoff_consume(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_live_recovery_handoff import RecoveryHandoffConsumePlan

        if type(plan) is not RecoveryHandoffConsumePlan or domain != "RECOVERY":
            raise TypeError("recovery handoff consume requires an exact plan")
        _validate_recovery_handoff_transaction(plan, stage="CONSUME")
        for record in (
            plan.recovery_control.before,
            plan.action.before,
        ):
            _require_nonce_ownership(
                record,
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
        return self._prepare_and_execute(
            plans=plan.transaction_plans,
            readback_plans=plan.transaction_plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(
                (2, "owner_invocation_nonce_sha256"),
                (3, "owner_invocation_nonce_sha256"),
            ),
            token_targets=((3, "consume_transaction_client_request_token_sha256"),),
        )

    def commit_recovery_handoff_complete(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_live_recovery_handoff import RecoveryHandoffCompletePlan

        if type(plan) is not RecoveryHandoffCompletePlan or domain != "RECOVERY":
            raise TypeError("recovery handoff completion requires an exact plan")
        _validate_recovery_handoff_transaction(plan, stage="COMPLETE")
        for record in (
            plan.recovery_control.before,
            plan.action.before,
        ):
            _require_nonce_ownership(
                record,
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
        return self._prepare_and_execute(
            plans=plan.transaction_plans,
            readback_plans=plan.transaction_plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(
                (2, "owner_invocation_nonce_sha256"),
                (3, "owner_invocation_nonce_sha256"),
            ),
            token_targets=(),
        )

    def commit_teardown_seal(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import TeardownSealPlan

        if type(plan) is not TeardownSealPlan:
            raise TypeError("teardown seal requires a typed exact plan")
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def commit_finalization_progress(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import FinalizationProgressPlan

        if type(plan) is not FinalizationProgressPlan:
            raise TypeError(
                "finalization progress requires a typed exact plan"
            )
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def commit_retained_owner_takeover(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import RetainedOwnerTakeoverPlan

        if type(plan) is not RetainedOwnerTakeoverPlan:
            raise TypeError("retained takeover requires a typed exact plan")
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def commit_snapshot_cleanup_transition(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import SnapshotCleanupTransitionPlan

        if type(plan) is not SnapshotCleanupTransitionPlan:
            raise TypeError(
                "snapshot cleanup transition requires a typed exact plan"
            )
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def arm_snapshot_delete_action(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import SnapshotDeleteActionPlan

        if (
            type(plan) is not SnapshotDeleteActionPlan
            or type(plan.action) is not ExactPut
        ):
            raise TypeError(
                "snapshot action arm requires an exact action-only put plan"
            )
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def close_snapshot_delete_action(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import SnapshotDeleteActionPlan

        if (
            type(plan) is not SnapshotDeleteActionPlan
            or type(plan.action) is not ExactUpdate
        ):
            raise TypeError(
                "snapshot action close requires an exact action-only update plan"
            )
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def take_over_snapshot_cleanup_owner(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import (
            SnapshotCleanupOwnerTakeoverPlan,
        )

        if type(plan) is not SnapshotCleanupOwnerTakeoverPlan:
            raise TypeError(
                "snapshot takeover requires an exact owner takeover plan"
            )
        return self._commit_retained_state_plan(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def _commit_retained_state_plan(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        from .task12_retained_state import (
            FinalizationProgressPlan,
            RecoveryProgressPlan,
            RecoverySealPlan,
            RetainedOwnerTakeoverPlan,
            SnapshotCleanupOwnerTakeoverPlan,
            SnapshotDeleteActionPlan,
            SnapshotCleanupTransitionPlan,
            TeardownSealPlan,
        )

        plans = plan.transaction_plans
        nonce_targets: Tuple[Tuple[int, str], ...]
        token_targets: Tuple[Tuple[int, str], ...] = ()
        allow_durable_adoption = False
        if type(plan) is RecoverySealPlan:
            nonce_targets = ((3, "owner_invocation_nonce_sha256"),)
        elif type(plan) is RecoveryProgressPlan:
            recovery = plan.recovery_control
            _require_nonce_ownership(
                recovery.before,
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
            if recovery.after["state"] == "RECOVERY_COMPLETE":
                nonce_targets = ()
                allow_durable_adoption = True
            else:
                nonce_targets = ((2, "owner_invocation_nonce_sha256"),)
        elif type(plan) is TeardownSealPlan:
            if type(plan.finalization_control) is ExactUpdate:
                nonce_targets = ((3, "owner_invocation_nonce_sha256"),)
            else:
                nonce_targets = ()
                allow_durable_adoption = True
        elif type(plan) is FinalizationProgressPlan:
            finalization = plan.finalization_control
            _require_nonce_ownership(
                finalization.before,
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
            if finalization.after["state"] == "DRAINED_PUBLISHED":
                nonce_targets = ()
                allow_durable_adoption = True
            else:
                nonce_targets = ((2, "owner_invocation_nonce_sha256"),)
        elif type(plan) is RetainedOwnerTakeoverPlan:
            nonce_targets = (
                (len(plans) - 1, "owner_invocation_nonce_sha256"),
            )
        elif type(plan) is SnapshotCleanupTransitionPlan:
            cleanup = plan.cleanup_control
            if cleanup.before["state"] == "DORMANT":
                nonce_targets = (
                    (2, "owner_invocation_nonce_sha256"),
                )
            elif cleanup.after["state"] == "OWNED":
                nonce_targets = ((3, "owner_invocation_nonce_sha256"),)
            else:
                _require_nonce_ownership(
                    cleanup.before,
                    "owner_invocation_nonce_sha256",
                    raw_owner_nonce,
                )
                if cleanup.after["state"] in {
                    "DELETED",
                    "ALREADY_ABSENT",
                    "CLEANUP_INCIDENT",
                }:
                    nonce_targets = ()
                    allow_durable_adoption = True
                else:
                    nonce_targets = (
                        (3, "owner_invocation_nonce_sha256"),
                    )
            if plan.cleanup_action is not None:
                action_index = 4
                if type(plan.cleanup_action) is ExactUpdate:
                    _require_nonce_ownership(
                        plan.cleanup_action.before,
                        "owner_invocation_nonce_sha256",
                        raw_owner_nonce,
                    )
                nonce_targets = nonce_targets + (
                    (action_index, "owner_invocation_nonce_sha256"),
                )
                if type(plan.cleanup_action) is ExactUpdate:
                    token_targets = (
                        (
                            action_index,
                            "consume_transaction_client_request_token_sha256",
                        ),
                    )
        elif type(plan) is SnapshotCleanupOwnerTakeoverPlan:
            nonce_targets = (
                (2, "owner_invocation_nonce_sha256"),
            )
        elif type(plan) is SnapshotDeleteActionPlan:
            _require_nonce_ownership(
                plan.control.expected,
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
            action_owned_by_current_nonce = (
                type(plan.action) is not ExactUpdate
                or plan.action.before["owner_invocation_nonce_sha256"]
                == hashlib.sha256(raw_owner_nonce).hexdigest()
            )
            if (
                type(plan.action) is ExactUpdate
                and action_owned_by_current_nonce
            ):
                _require_nonce_ownership(
                    plan.action.before,
                    "owner_invocation_nonce_sha256",
                    raw_owner_nonce,
                )
            nonce_targets = (
                ((2, "owner_invocation_nonce_sha256"),)
                if action_owned_by_current_nonce
                else ()
            )
            allow_durable_adoption = True
        else:
            raise TypeError("unsupported Task 12 retained-state plan")
        materialized, client_request_token = _materialize_transaction(
            plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=nonce_targets,
            token_targets=token_targets,
            transaction_hash_targets=(),
        )
        _validate_plans(materialized)
        return self._execute_transaction(
            plans=materialized,
            readback_plans=materialized,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=nonce_targets,
            allow_durable_adoption=allow_durable_adoption,
            client_request_token=client_request_token,
        )

    def arm_action(
        self,
        *,
        index: ExactCheck,
        control: ExactUpdate,
        action: ExactPut,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, control, action)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_control",
                "glm52_production_action",
            ),
            (ExactCheck, ExactUpdate, ExactPut),
        )
        _validate_control_revision_update(control)
        if action.item.get("state") != "ARMED":
            raise ValueError("action arm must create ARMED")
        _require_activation_binding(index.expected, control.after, action.item)
        if control.after.get("last_sky_post_action_key") != action.key.sort_key:
            raise ValueError("control does not bind the armed action key")
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((2, "armed_by_invocation_nonce_sha256"),),
            token_targets=(
                (2, "arming_transaction_client_request_token_sha256"),
            ),
        )

    def consume_action(
        self,
        *,
        index: ExactCheck,
        control: ExactUpdate,
        action: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, control, action)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_control",
                "glm52_production_action",
            ),
            (ExactCheck, ExactUpdate, ExactUpdate),
        )
        _validate_control_revision_update(control)
        validate_transition(
            "glm52_production_action", action.before, action.after
        )
        if (
            action.before.get("state") != "ARMED"
            or action.after.get("state") != "CONSUMED"
        ):
            raise ValueError("action consume requires ARMED to CONSUMED")
        _validate_action_consumption(index, control, action)
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((2, "owner_invocation_nonce_sha256"),),
            token_targets=(
                (2, "consume_transaction_client_request_token_sha256"),
            ),
        )

    def classify_owner_dead_sky_action(
        self,
        *,
        plan: object,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        """Commit or exactly adopt the retained no-launch Sky classification."""

        from .task12_live_owner_death import (
            OwnerDeathClassificationPlan,
        )

        if type(plan) is not OwnerDeathClassificationPlan:
            raise TypeError(
                "owner-death classification requires one typed exact plan"
            )
        plans = plan.transaction_plans
        if (
            len(plans) != 5
            or type(plan.index) is not ExactCheck
            or type(plan.recovery_control) is not ExactCheck
            or type(plan.execution) is not ExactCheck
            or type(plan.control) not in (ExactCheck, ExactUpdate)
            or type(plan.action) not in (ExactCheck, ExactUpdate)
            or type(plan.control) is not type(plan.action)
        ):
            raise TypeError("owner-death plan shape is not closed")
        expected_types = (
            "glm52_production_activation_index",
            "glm52_production_control",
            "glm52_production_recovery_control",
            "glm52_production_execution",
            "glm52_production_action",
        )
        if tuple(_plan_record_type(item) for item in plans) != expected_types:
            raise ValueError("owner-death plan record families drifted")
        recovery = plan.recovery_control.expected
        execution = plan.execution.expected
        action_before = (
            plan.action.expected
            if type(plan.action) is ExactCheck
            else plan.action.before
        )
        control_before = (
            plan.control.expected
            if type(plan.control) is ExactCheck
            else plan.control.before
        )
        _require_nonce_ownership(
            recovery,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        _require_activation_binding(
            plan.index.expected,
            control_before,
            recovery,
            execution,
            action_before,
        )
        if (
            recovery.get("state") != "OWNED"
            or execution.get("state")
            not in {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
            or execution.get("terminal_status") != execution.get("state")
            or control_before.get("active_epoch") != execution.get("epoch")
            or control_before.get("active_execution_arn")
            != execution.get("expected_execution_arn")
            or control_before.get("active_state_machine_version_arn")
            != execution.get("expected_state_machine_version_arn")
            or control_before.get("last_sky_post_action_key")
            != plan.action.key.sort_key
            or control_before.get("last_sky_post_generation")
            != action_before.get("generation")
            or control_before.get("last_sky_post_state")
            != action_before.get("state")
            or action_before.get("action_kind") != "SKY_POST"
        ):
            raise ValueError("owner-death authority tuple drifted")
        if type(plan.action) is ExactCheck:
            if (
                action_before.get("state") != "POST_CLASSIFIED"
                or control_before.get("last_sky_post_state")
                != "POST_CLASSIFIED"
            ):
                raise ValueError(
                    "owner-death durable adoption is not classified"
                )
            _validate_plans(plans)
            return self._adopt_durable(plans)

        assert type(plan.control) is ExactUpdate
        _validate_control_revision_update(plan.control)
        validate_transition(
            "glm52_production_action",
            plan.action.before,
            plan.action.after,
        )
        target_state = plan.action.after.get("state")
        if (
            target_state not in {"ABANDONED", "POST_CLASSIFIED"}
            or plan.control.after.get("last_sky_post_state") != target_state
            or plan.control.after.get("last_sky_post_action_key")
            != plan.action.key.sort_key
            or plan.control.after.get("last_sky_post_generation")
            != plan.action.after.get("generation")
        ):
            raise ValueError("owner-death target state drifted")
        _require_activation_binding(
            plan.index.expected,
            plan.control.after,
            plan.action.after,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((2, "owner_invocation_nonce_sha256"),),
            token_targets=(
                (
                    (4, "post_classification_transaction_client_request_token_sha256"),
                )
                if target_state == "POST_CLASSIFIED"
                else ()
            ),
        )

    def put_worker_launch_prepared(
        self,
        *,
        worker_launch: ExactPut,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (worker_launch,)
        _require_plan_types(
            plans,
            ("glm52_production_worker_launch",),
            (ExactPut,),
        )
        if worker_launch.item.get("state") != "PREPARED_NOT_SENT":
            raise ValueError("worker launch must be PREPARED_NOT_SENT")
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((0, "owner_invocation_nonce_sha256"),),
            token_targets=(),
            consistent_single_readback=True,
        )

    def bind_worker_launch_committed_wal(
        self,
        *,
        worker_launch: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (worker_launch,)
        _require_plan_types(
            plans,
            ("glm52_production_worker_launch",),
            (ExactUpdate,),
        )
        validate_record(
            "glm52_production_worker_launch", worker_launch.before
        )
        validate_record(
            "glm52_production_worker_launch", worker_launch.after
        )
        changed = {
            field
            for field in worker_launch.before
            if worker_launch.before[field] != worker_launch.after[field]
        }
        if (
            domain != "WORKER_LAUNCH_WAL_BIND"
            or worker_launch.before.get("state") != "PREPARED_NOT_SENT"
            or worker_launch.after.get("state") != "PREPARED_NOT_SENT"
            or worker_launch.before.get(
                "ddb_committed_journal_entry_sha256"
            )
            is not None
            or worker_launch.after.get(
                "ddb_committed_journal_entry_sha256"
            )
            is None
            or worker_launch.after.get("revision")
            != worker_launch.before.get("revision") + 1
            or not {
                "ddb_committed_journal_entry_sha256",
                "revision",
            }.issubset(changed)
            or not changed
            <= {
                "ddb_committed_journal_entry_sha256",
                "revision",
                "updated_at",
            }
        ):
            raise ValueError("worker WAL binding revision is invalid")
        _require_nonce_ownership(
            worker_launch.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((0, "owner_invocation_nonce_sha256"),),
            token_targets=(),
            consistent_single_readback=True,
        )

    def record_liability_evidence(
        self,
        *,
        index: ExactCheck,
        liability: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, liability)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch_liability",
            ),
            (ExactCheck, ExactUpdate),
        )
        if domain not in {
            "LIABILITY_SCAN_EVIDENCE",
            "LIABILITY_RECONCILIATION_EVIDENCE",
        }:
            raise ValueError("liability evidence domain is not closed")
        _validate_liability_revision_only(liability)
        _validate_liability_scan_revision(liability)
        _require_activation_binding(index.expected, liability.after)
        _require_nonce_ownership(
            liability.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((1, "owner_invocation_nonce_sha256"),),
            token_targets=(),
        )

    def complete_worker_launch_attempt(
        self,
        *,
        index: ExactCheck,
        worker_launch: ExactUpdate,
        liability: ExactUpdate,
        liability_action: ExactUpdate,
        classification: str,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (
            index,
            worker_launch,
            liability,
            liability_action,
        )
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch",
                "glm52_production_worker_launch_liability",
                "glm52_production_worker_launch_liability_action",
            ),
            (ExactCheck, ExactUpdate, ExactUpdate, ExactUpdate),
        )
        if domain != "WORKER_LAUNCH_ATTEMPT_RESULT":
            raise ValueError("worker launch attempt result domain drifted")
        expected_edges = {
            "DIRECT_SUCCESS": (
                ("POSSIBLY_SENT", "INSTANCE_OBSERVED"),
                ("SAME_TOKEN_COMPLETION", "WATCHING"),
                ("CONSUMED", "COMPLETED"),
            ),
            "POSITIVE_REJECTION": (
                ("POSSIBLY_SENT", "REJECTED_NO_INSTANCE"),
                (
                    "SAME_TOKEN_COMPLETION",
                    "REJECTION_PROVED_AWAITING_TERMINAL_V2",
                ),
                ("CONSUMED", "COMPLETED"),
            ),
            "AMBIGUOUS": (
                ("POSSIBLY_SENT", "POSSIBLY_SENT"),
                ("SAME_TOKEN_COMPLETION", "WATCHING"),
                ("CONSUMED", "AMBIGUOUS"),
            ),
        }
        edges = expected_edges.get(classification)
        actual_edges = (
            (
                worker_launch.before.get("state"),
                worker_launch.after.get("state"),
            ),
            (
                liability.before.get("state"),
                liability.after.get("state"),
            ),
            (
                liability_action.before.get("state"),
                liability_action.after.get("state"),
            ),
        )
        if edges is None or actual_edges != edges:
            raise ValueError("worker launch attempt classification drifted")
        if classification == "AMBIGUOUS":
            _validate_worker_attempt_revision_only(worker_launch)
        else:
            validate_transition(
                "glm52_production_worker_launch",
                worker_launch.before,
                worker_launch.after,
            )
            _validate_worker_completion_revision(
                worker_launch,
                classification=classification,
            )
        validate_transition(
            "glm52_production_worker_launch_liability",
            liability.before,
            liability.after,
        )
        _require_exact_changed_fields(
            liability,
            allowed={"state", "revision", "updated_at"},
            required={"state", "revision"},
            label="worker completion liability",
        )
        validate_transition(
            "glm52_production_worker_launch_liability_action",
            liability_action.before,
            liability_action.after,
        )
        _validate_worker_completion_action_revision(
            liability_action,
            classification=classification,
        )
        if liability_action.after.get("action_kind") != "SAME_TOKEN_COMPLETE":
            raise ValueError("worker completion action kind drifted")
        _require_activation_binding(
            index.expected,
            worker_launch.before,
            worker_launch.after,
            liability.before,
            liability.after,
            liability_action.before,
            liability_action.after,
        )
        _require_same_identity(
            liability.before,
            liability_action.before,
            (
                "run_id",
                "campaign_identity_sha256",
                "activation_id",
                "activation_ordinal",
                "generation",
                "allocation_ordinal",
                "worker_launch_identity_sha256",
            ),
        )
        _require_same_identity(
            liability_action.before,
            liability_action.after,
            (
                "run_id",
                "campaign_identity_sha256",
                "activation_id",
                "activation_ordinal",
                "generation",
                "allocation_ordinal",
                "worker_launch_identity_sha256",
                "worker_launch_liability_identity_sha256",
                "action_kind",
            ),
        )
        for record in (
            worker_launch.before,
            worker_launch.after,
            liability.before,
            liability.after,
        ):
            _require_same_identity(
                worker_launch.before,
                record,
                (
                    "run_id",
                    "campaign_identity_sha256",
                    "activation_id",
                    "activation_ordinal",
                    "generation",
                    "allocation_ordinal",
                    "ec2_client_token",
                    "launch_parameters_sha256",
                    "expected_worker_tags_sha256",
                ),
            )
        if (
            worker_launch.after.get("same_token_completion_count")
            != liability.after.get("same_token_completion_attempts")
            or liability_action.after.get(
                "worker_launch_identity_sha256"
            )
            != liability.after.get("worker_launch_identity_sha256")
        ):
            raise ValueError("worker completion counter/identity drifted")
        for record in (
            worker_launch.before,
            liability.before,
            liability_action.before,
        ):
            _require_nonce_ownership(
                record,
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(),
            token_targets=(),
        )

    def record_liability_incident(
        self,
        *,
        index: ExactCheck,
        liability: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, liability)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch_liability",
            ),
            (ExactCheck, ExactUpdate),
        )
        if domain != "LIABILITY_INCIDENT":
            raise ValueError("liability incident domain drifted")
        validate_transition(
            "glm52_production_worker_launch_liability",
            liability.before,
            liability.after,
        )
        _require_exact_changed_fields(
            liability,
            allowed={
                "state",
                "incident_identity_sha256",
                "incident_at",
                "revision",
                "updated_at",
            },
            required={
                "state",
                "incident_identity_sha256",
                "incident_at",
                "revision",
            },
            label="liability incident",
        )
        if (
            liability.after.get("state") != "LIABILITY_INCIDENT"
            or liability.after.get("incident_identity_sha256") is None
            or liability.after.get("incident_at") is None
        ):
            raise ValueError("liability incident transition drifted")
        _require_activation_binding(index.expected, liability.after)
        _require_nonce_ownership(
            liability.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(),
            token_targets=(),
        )

    def record_worker_launch_reconciliation(
        self,
        *,
        index: ExactCheck,
        worker_launch: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, worker_launch)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch",
            ),
            (ExactCheck, ExactUpdate),
        )
        if domain != "WORKER_LAUNCH_RECONCILIATION":
            raise ValueError("worker reconciliation domain drifted")
        validate_transition(
            "glm52_production_worker_launch",
            worker_launch.before,
            worker_launch.after,
        )
        allowed_edges = {
            ("POSSIBLY_SENT", "INSTANCE_OBSERVED"),
            ("POSSIBLY_SENT", "MULTIPLE_INSTANCE_TOKEN_INCIDENT"),
            ("POSSIBLY_SENT", "UNRESOLVED_LAUNCH_INCIDENT"),
            ("INSTANCE_OBSERVED", "ALLOCATION_OPEN"),
            ("ALLOCATION_OPEN", "INSTANCE_TERMINAL"),
            ("INSTANCE_TERMINAL", "ALLOCATION_CLOSED"),
        }
        if (
            worker_launch.before.get("state"),
            worker_launch.after.get("state"),
        ) not in allowed_edges:
            raise ValueError("worker reconciliation edge is forbidden")
        _validate_worker_reconciliation_revision(worker_launch)
        _require_activation_binding(index.expected, worker_launch.after)
        _require_nonce_ownership(
            worker_launch.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(),
            token_targets=(),
        )

    def commit_worker_launch_possibly_sent(
        self,
        *,
        index: ExactCheck,
        reserve_checks: Tuple[ExactCheck, ...],
        worker_launch: ExactUpdate,
        liability: ExactPut,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        if type(reserve_checks) is not tuple or not reserve_checks:
            raise ValueError("worker send requires reserve checks")
        plans: Tuple[object, ...] = (
            (index,) + reserve_checks + (worker_launch, liability)
        )
        _require_exact_plan_type(index, ExactCheck)
        for check in reserve_checks:
            _require_exact_plan_type(check, ExactCheck)
        _require_exact_plan_type(worker_launch, ExactUpdate)
        _require_exact_plan_type(liability, ExactPut)
        if _plan_record_type(index) != "glm52_production_activation_index":
            raise ValueError("worker send index family mismatch")
        if _plan_record_type(worker_launch) != "glm52_production_worker_launch":
            raise ValueError("worker send launch family mismatch")
        if _plan_record_type(liability) != (
            "glm52_production_worker_launch_liability"
        ):
            raise ValueError("worker send liability family mismatch")
        validate_transition(
            "glm52_production_worker_launch",
            worker_launch.before,
            worker_launch.after,
        )
        if (
            worker_launch.before.get("state") != "PREPARED_NOT_SENT"
            or worker_launch.after.get("state") != "POSSIBLY_SENT"
            or liability.item.get("state") != "UNOWNED_NOT_ACTIONABLE"
        ):
            raise ValueError("worker send state set mismatch")
        _require_nonce_ownership(
            worker_launch.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        for field in (
            "activation_id",
            "activation_ordinal",
            "generation",
            "allocation_ordinal",
            "ec2_client_token",
            "launch_parameters_sha256",
            "expected_worker_tags_sha256",
            "gpu_liability_reserve_seconds",
            "gpu_liability_reserve_cost_usd",
            "ebs_liability_reserve_cost_usd",
            "residual_liability_approval_identity_sha256",
            "gpu_liability_reserve_ledger_identity_sha256",
        ):
            if worker_launch.after.get(field) != liability.item.get(field):
                raise ValueError("worker send launch/liability binding mismatch")
        _require_activation_binding(
            index.expected, worker_launch.after, liability.item
        )
        launch_index = 1 + len(reserve_checks)
        liability_index = launch_index + 1
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=(index, plans[launch_index], plans[liability_index]),
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(
                (launch_index, "owner_invocation_nonce_sha256"),
            ),
            token_targets=(),
        )

    def acquire_liability_owner(
        self,
        *,
        index: ExactCheck,
        liability: ExactUpdate,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (index, liability)
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch_liability",
            ),
            (ExactCheck, ExactUpdate),
        )
        validate_transition(
            "glm52_production_worker_launch_liability",
            liability.before,
            liability.after,
        )
        if (
            liability.before.get("state") != "UNOWNED_NOT_ACTIONABLE"
            or liability.after.get("state") != "WATCHING"
        ):
            raise ValueError("liability acquisition state mismatch")
        _require_activation_binding(index.expected, liability.after)
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((1, "owner_invocation_nonce_sha256"),),
            token_targets=(),
        )

    def consume_liability_action(
        self,
        *,
        action_name: str,
        index: ExactCheck,
        liability: ExactUpdate,
        action: ExactUpdate,
        post_terminal_allocation: Optional[object],
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        allowed = {
            "SAME_TOKEN_COMPLETE": None,
            "TERMINATE_LATE_INSTANCE": None,
            "POST_TERMINAL_ALLOCATION_DISCOVER": ExactPut,
            "POST_TERMINAL_ALLOCATION_OPEN": ExactUpdate,
            "POST_TERMINAL_ALLOCATION_CLOSE": ExactUpdate,
        }
        if action_name == "LIABILITY_SETTLE":
            raise ValueError("LIABILITY_SETTLE requires settlement method")
        if action_name not in allowed:
            raise ValueError("stale or unknown liability action name")
        required_allocation_type = allowed[action_name]
        if required_allocation_type is None:
            if post_terminal_allocation is not None:
                raise ValueError("liability action forbids allocation write")
            tail: Tuple[object, ...] = ()
        else:
            if type(post_terminal_allocation) is not required_allocation_type:
                raise ValueError("liability action allocation shape mismatch")
            if _plan_record_type(post_terminal_allocation) != (
                "glm52_production_post_terminal_allocation"
            ):
                raise ValueError("liability action allocation family mismatch")
            _require_nonce_ownership(
                (
                    post_terminal_allocation.before
                    if type(post_terminal_allocation) is ExactUpdate
                    else post_terminal_allocation.item
                ),
                "owner_invocation_nonce_sha256",
                raw_owner_nonce,
            )
            tail = (post_terminal_allocation,)
        plans = (index, liability, action) + tail
        _require_plan_types(
            plans[:3],
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch_liability",
                "glm52_production_worker_launch_liability_action",
            ),
            (ExactCheck, ExactUpdate, ExactUpdate),
        )
        if action.after.get("action_kind") != action_name:
            raise ValueError("liability action name does not match record")
        _require_activation_binding(
            index.expected, liability.after, action.after
        )
        _require_nonce_ownership(
            action.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        _require_nonce_ownership(
            liability.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        validate_transition(
            "glm52_production_worker_launch_liability_action",
            action.before,
            action.after,
        )
        if action_name == "SAME_TOKEN_COMPLETE":
            validate_transition(
                "glm52_production_worker_launch_liability",
                liability.before,
                liability.after,
            )
            if (
                liability.before.get("state") != "WATCHING"
                or liability.after.get("state") != "SAME_TOKEN_COMPLETION"
            ):
                raise ValueError("same-token liability state mismatch")
        elif action_name == "TERMINATE_LATE_INSTANCE":
            validate_transition(
                "glm52_production_worker_launch_liability",
                liability.before,
                liability.after,
            )
            if liability.after.get("state") != "LATE_INSTANCE_DRAINING":
                raise ValueError("late-instance liability state mismatch")
        else:
            _validate_liability_revision_only(liability)
        if type(post_terminal_allocation) is ExactPut and (
            post_terminal_allocation.item.get("state") != "DISCOVERED"
        ):
            raise ValueError("post-terminal discover must create DISCOVERED")
        if type(post_terminal_allocation) is ExactUpdate:
            validate_transition(
                "glm52_production_post_terminal_allocation",
                post_terminal_allocation.before,
                post_terminal_allocation.after,
            )
            target_state = (
                "ALLOCATION_OPEN"
                if action_name == "POST_TERMINAL_ALLOCATION_OPEN"
                else "ALLOCATION_CLOSED"
            )
            if post_terminal_allocation.after.get("state") != target_state:
                raise ValueError("post-terminal allocation target mismatch")
        allocation_identity = (
            ()
            if post_terminal_allocation is None
            else (
                (
                    _plan_record(post_terminal_allocation),
                    (
                        post_terminal_allocation.before
                        if type(post_terminal_allocation) is ExactUpdate
                        else post_terminal_allocation.item
                    ),
                ),
            )
        )
        _validate_liability_action_binding(
            index,
            liability,
            action,
            related_records=allocation_identity,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((2, "owner_invocation_nonce_sha256"),),
            token_targets=(
                (2, "consume_transaction_client_request_token_sha256"),
            ),
        )

    def stage_snapshot_delete(
        self,
        *,
        cleanup_control: ExactUpdate,
        cleanup_action: ExactUpdate,
        cleanup_transition: ExactPut,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (
            cleanup_control,
            cleanup_action,
            cleanup_transition,
        )
        _require_plan_types(
            plans,
            (
                "glm52_production_snapshot_cleanup_control",
                "glm52_production_snapshot_cleanup_action",
                "glm52_production_snapshot_cleanup_transition",
            ),
            (ExactUpdate, ExactUpdate, ExactPut),
        )
        validate_transition(
            "glm52_production_snapshot_cleanup_control",
            cleanup_control.before,
            cleanup_control.after,
        )
        validate_transition(
            "glm52_production_snapshot_cleanup_action",
            cleanup_action.before,
            cleanup_action.after,
        )
        if (
            cleanup_control.after.get("state") != "DELETE_POSSIBLY_SENT"
            or cleanup_action.after.get("state") != "CONSUMED"
        ):
            raise ValueError("snapshot delete staging state mismatch")
        transition = cleanup_transition.item
        if (
            transition.get("from_state") != cleanup_control.before.get("state")
            or transition.get("to_state") != cleanup_control.after.get("state")
            or transition.get("from_revision")
            != cleanup_control.before.get("revision")
            or transition.get("to_revision")
            != cleanup_control.after.get("revision")
        ):
            raise ValueError("snapshot cleanup transition binding mismatch")
        _require_same_identity(
            cleanup_control.after,
            cleanup_action.after,
            ("run_id", "activation_id", "activation_ordinal"),
        )
        _require_same_identity(
            cleanup_control.after,
            transition,
            ("run_id", "activation_id", "activation_ordinal"),
        )
        _require_nonce_ownership(
            cleanup_control.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        _require_nonce_ownership(
            cleanup_action.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=((1, "owner_invocation_nonce_sha256"),),
            token_targets=(
                (1, "consume_transaction_client_request_token_sha256"),
            ),
        )

    def settle_worker_launch_liability(
        self,
        *,
        index: ExactCheck,
        liability_action: ExactUpdate,
        liability: ExactUpdate,
        settlement: ExactPut,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: Optional[bytes],
        duplicate_settlement_identity_sha256: Optional[str] = None,
    ) -> TransactionResolution:
        plans: Tuple[object, ...] = (
            index,
            liability_action,
            liability,
            settlement,
        )
        _require_plan_types(
            plans,
            (
                "glm52_production_activation_index",
                "glm52_production_worker_launch_liability_action",
                "glm52_production_worker_launch_liability",
                "glm52_production_worker_launch_liability_settlement",
            ),
            (ExactCheck, ExactUpdate, ExactUpdate, ExactPut),
        )
        validate_transition(
            "glm52_production_worker_launch_liability_action",
            liability_action.before,
            liability_action.after,
        )
        validate_transition(
            "glm52_production_worker_launch_liability",
            liability.before,
            liability.after,
        )
        if (
            liability_action.after.get("state") != "CONSUMED"
            or liability.after.get("state")
            not in {
                "SETTLED_NO_INSTANCE_REJECTED",
                "SETTLED_INSTANCE_CLOSED",
            }
        ):
            raise ValueError("liability settlement state mismatch")
        if liability_action.after.get("action_kind") != "LIABILITY_SETTLE":
            raise ValueError("settlement action kind mismatch")
        _require_activation_binding(
            index.expected,
            liability_action.after,
            liability.after,
            settlement.item,
        )
        _require_same_identity(
            liability.after,
            settlement.item,
            (
                "generation",
                "allocation_ordinal",
                "worker_launch_identity_sha256",
            ),
        )
        _validate_liability_action_binding(
            index,
            liability,
            liability_action,
            related_records=((settlement.item, settlement.item),),
        )
        if duplicate_settlement_identity_sha256 is not None:
            if raw_owner_nonce is not None:
                raise ValueError("durable adoption forbids a raw owner nonce")
            duplicate_identity = _require_sha256(
                "duplicate_settlement_identity_sha256",
                duplicate_settlement_identity_sha256,
            )
            if settlement.item.get("canonical_body_sha256") != duplicate_identity:
                raise ValueError("settlement duplicate identity mismatch")
            _validate_settlement_final_binding(liability, settlement)
            _validate_plans(plans)
            return self._adopt_durable(plans)
        if raw_owner_nonce is None:
            raise ValueError("live settlement requires a raw owner nonce")
        _require_nonce_ownership(
            liability_action.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        _require_nonce_ownership(
            liability.before,
            "owner_invocation_nonce_sha256",
            raw_owner_nonce,
        )
        return self._prepare_and_execute(
            plans=plans,
            readback_plans=plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=(
                (1, "owner_invocation_nonce_sha256"),
                (3, "owner_invocation_nonce_sha256"),
            ),
            token_targets=(
                (1, "consume_transaction_client_request_token_sha256"),
                (3, "transaction_client_request_token_sha256"),
            ),
            settlement_binding=(2, 3),
        )

    def _adopt_durable(
        self, readback_plans: Tuple[object, ...]
    ) -> TransactionResolution:
        readback, request_id, read_error = self._read_optional(readback_plans)
        if read_error:
            return TransactionResolution(
                WriteOutcome.READBACK_UNAVAILABLE,
                (),
                request_id,
                None,
                (),
            )
        post = tuple(_plan_post_record(plan) for plan in readback_plans)
        present = tuple(item for item in readback if item is not None)
        outcome = (
            WriteOutcome.EXACT_DURABLE_ADOPTION
            if readback == post
            else WriteOutcome.MISMATCHED_COMMIT
        )
        return TransactionResolution(
            outcome,
            present,
            request_id,
            None,
            (),
        )

    def _prepare_and_execute(
        self,
        *,
        plans: Tuple[object, ...],
        readback_plans: Tuple[object, ...],
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
        nonce_targets: Tuple[Tuple[int, str], ...],
        token_targets: Tuple[Tuple[int, str], ...],
        transaction_hash_targets: Tuple[Tuple[int, str], ...] = (),
        settlement_binding: Optional[Tuple[int, int]] = None,
        consistent_single_readback: bool = False,
    ) -> TransactionResolution:
        if len(plans) > 100:
            raise ValueError("DynamoDB transaction exceeds 100 items")
        materialized, client_request_token = _materialize_transaction(
            plans,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=nonce_targets,
            token_targets=token_targets,
            transaction_hash_targets=transaction_hash_targets,
            settlement_binding=settlement_binding,
        )
        _validate_plans(materialized)
        materialized_readbacks = tuple(
            _find_materialized_plan(plans, materialized, plan)
            for plan in readback_plans
        )
        return self._execute_transaction(
            plans=materialized,
            readback_plans=materialized_readbacks,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=tuple(
                (
                    _identity_index(
                        materialized_readbacks, materialized[index]
                    ),
                    field,
                )
                for index, field in nonce_targets
                if any(
                    item is materialized[index]
                    for item in materialized_readbacks
                )
            ),
            allow_durable_adoption=False,
            client_request_token=client_request_token,
            consistent_single_readback=consistent_single_readback,
        )

    def _execute_transaction(
        self,
        *,
        plans: Tuple[object, ...],
        readback_plans: Tuple[object, ...],
        raw_owner_nonce: bytes,
        nonce_targets: Tuple[Tuple[int, str], ...],
        allow_durable_adoption: bool,
        client_request_token: str,
        consistent_single_readback: bool = False,
    ) -> TransactionResolution:
        request_id: Optional[str] = None
        error_code: Optional[str] = None
        cancellation_reasons: Tuple[Mapping[str, object], ...] = ()
        try:
            response = self._client.transact_write_items(
                TransactItems=[_wire_item(plan, self._table_name) for plan in plans],
                ClientRequestToken=client_request_token,
                ReturnConsumedCapacity="NONE",
                ReturnItemCollectionMetrics="NONE",
            )
            try:
                _, request_id = _require_success_metadata(response)
            except LedgerReadError:
                error_code = "MalformedResponse"
        except Exception as exc:
            error_code, request_id, cancellation_reasons = _exception_details(exc)
            if error_code == "TransactionCanceledException":
                return TransactionResolution(
                    WriteOutcome.CONDITION_REJECTED,
                    (),
                    request_id,
                    error_code,
                    cancellation_reasons,
                )
            if error_code in {
                "IdempotentParameterMismatchException",
                "ValidationException",
                "AccessDeniedException",
                "ResourceNotFoundException",
            }:
                return TransactionResolution(
                    WriteOutcome.MISMATCHED_COMMIT,
                    (),
                    request_id,
                    error_code,
                    cancellation_reasons,
                )
        if consistent_single_readback:
            readback, read_request_id, read_error = (
                self._read_optional_consistent(readback_plans)
            )
        else:
            readback, read_request_id, read_error = self._read_optional(
                readback_plans
            )
        if read_error:
            return TransactionResolution(
                WriteOutcome.READBACK_UNAVAILABLE,
                (),
                request_id or read_request_id,
                error_code,
                cancellation_reasons,
            )
        present = tuple(
            item for item in readback if item is not None
        )
        post = tuple(_plan_post_record(plan) for plan in readback_plans)
        if readback == post:
            if allow_durable_adoption:
                outcome = WriteOutcome.EXACT_DURABLE_ADOPTION
            elif _readback_has_live_nonce(
                readback,
                raw_owner_nonce=raw_owner_nonce,
                nonce_targets=nonce_targets,
            ):
                outcome = WriteOutcome.EXACT_LIVE_OWNER_COMMIT
            else:
                outcome = WriteOutcome.FOREIGN_NONCE
            return TransactionResolution(
                outcome,
                present,
                request_id or read_request_id,
                error_code,
                cancellation_reasons,
            )
        if _readback_is_foreign_nonce(
            readback,
            post=post,
            raw_owner_nonce=raw_owner_nonce,
            nonce_targets=nonce_targets,
        ):
            outcome = WriteOutcome.FOREIGN_NONCE
        elif readback == tuple(
            _plan_pre_record(plan) for plan in readback_plans
        ):
            outcome = WriteOutcome.PRESTATE_OR_ABSENT
        else:
            outcome = WriteOutcome.MISMATCHED_COMMIT
        return TransactionResolution(
            outcome,
            present,
            request_id or read_request_id,
            error_code,
            cancellation_reasons,
        )

    def _read_optional_consistent(
        self, plans: Tuple[object, ...]
    ) -> Tuple[
        Tuple[Optional[Mapping[str, object]], ...],
        Optional[str],
        bool,
    ]:
        if len(plans) != 1:
            return (), None, True
        plan = plans[0]
        try:
            response = self._client.get_item(
                TableName=self._table_name,
                Key=_encoded_key(plan.key),
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
            _, request_id = _require_success_metadata(response)
        except Exception:
            return (), None, True
        if "Item" not in response:
            return (None,), request_id, False
        try:
            record = _decode_validated_physical_item(
                response["Item"],
                key=plan.key,
                record_type=_plan_record_type(plan),
            )
        except LedgerReadError:
            return (), request_id, False
        return (record,), request_id, False

    def _read_optional(
        self, plans: Tuple[object, ...]
    ) -> Tuple[
        Tuple[Optional[Mapping[str, object]], ...],
        Optional[str],
        bool,
    ]:
        requests = []
        for plan in plans:
            requests.append(
                {
                    "Get": {
                        "TableName": self._table_name,
                        "Key": _encoded_key(plan.key),
                    }
                }
            )
        try:
            response = self._client.transact_get_items(
                TransactItems=requests,
                ReturnConsumedCapacity="NONE",
            )
            _, request_id = _require_success_metadata(response)
        except Exception:
            return (), None, True
        responses = response.get("Responses")
        if type(responses) is not list or len(responses) != len(plans):
            return (), request_id, True
        result: List[Optional[Mapping[str, object]]] = []
        for response_item, plan in zip(responses, plans):
            if type(response_item) is not dict:
                return (), request_id, True
            if not response_item:
                result.append(None)
                continue
            if set(response_item) != {"Item"}:
                return (), request_id, True
            try:
                result.append(
                    _decode_validated_physical_item(
                        response_item["Item"],
                        key=plan.key,
                        record_type=_plan_record_type(plan),
                    )
                )
            except LedgerReadError:
                return (), request_id, False
        return tuple(result), request_id, False


def _encoded_key(key: LedgerKey) -> Dict[str, Dict[str, object]]:
    if type(key) is not LedgerKey:
        raise TypeError("key must be an exact LedgerKey")
    if type(key.run_id) is not str or type(key.sort_key) is not str:
        raise TypeError("ledger key fields must be exact strings")
    if not key.sort_key:
        raise ValueError("sort key must be nonempty")
    return encode_item({"PK": ledger_pk(key.run_id), "SK": key.sort_key})


def _decode_validated_physical_item(
    value: object, *, key: LedgerKey, record_type: str
) -> Mapping[str, object]:
    physical = decode_item(value)
    if "PK" not in physical or "SK" not in physical:
        raise LedgerReadError("physical ledger keys are missing")
    pk = physical.pop("PK")
    sk = physical.pop("SK")
    if (
        type(pk) is not str
        or type(sk) is not str
        or pk != ledger_pk(key.run_id)
        or sk != key.sort_key
    ):
        raise LedgerReadError("physical ledger key mismatch")
    try:
        return validate_record(record_type, physical, pk=pk, sk=sk)
    except ValueError as exc:
        raise LedgerReadError("closed ledger record validation failed") from exc


def _require_success_metadata(response: object) -> Tuple[int, str]:
    if type(response) is not dict:
        raise LedgerReadError("malformed DynamoDB response")
    metadata = response.get("ResponseMetadata")
    if type(metadata) is not dict:
        raise LedgerReadError("DynamoDB response metadata is missing")
    status = metadata.get("HTTPStatusCode")
    request_id = metadata.get("RequestId")
    if type(status) is not int or status != 200:
        raise LedgerReadError("DynamoDB response status is not exact 200")
    if type(request_id) is not str or not request_id:
        raise LedgerReadError("DynamoDB request ID is missing")
    return status, request_id


def _require_exact_plan_type(plan: object, expected: object) -> None:
    if type(plan) is not expected:
        raise TypeError("transaction plan value has the wrong exact type")


def _require_plan_types(
    plans: Tuple[object, ...],
    record_types: Tuple[str, ...],
    plan_types: Tuple[object, ...],
) -> None:
    if len(plans) != len(record_types) or len(plans) != len(plan_types):
        raise ValueError("closed transaction shape mismatch")
    for plan, record_type, plan_type in zip(
        plans, record_types, plan_types
    ):
        _require_exact_plan_type(plan, plan_type)
        if _plan_record_type(plan) != record_type:
            raise ValueError("closed transaction record family mismatch")


def _validate_recovery_handoff_transaction(
    plan: object,
    *,
    stage: str,
) -> None:
    expected_shapes = {
        "ARM": (
            (
                "glm52_production_activation_index",
                "glm52_production_control",
                "glm52_production_recovery_control",
                "glm52_production_recovery_action",
            ),
            (ExactCheck, ExactCheck, ExactUpdate, ExactPut),
        ),
        "CONSUME": (
            (
                "glm52_production_activation_index",
                "glm52_production_control",
                "glm52_production_recovery_control",
                "glm52_production_recovery_action",
            ),
            (ExactCheck, ExactCheck, ExactUpdate, ExactUpdate),
        ),
        "COMPLETE": (
            (
                "glm52_production_activation_index",
                "glm52_production_control",
                "glm52_production_recovery_control",
                "glm52_production_recovery_action",
                "glm52_task12_versioned_writer_control_v1",
            ),
            (ExactCheck, ExactCheck, ExactUpdate, ExactUpdate, ExactPut),
        ),
    }
    if stage not in expected_shapes:
        raise ValueError("recovery handoff transaction stage drifted")
    records, types = expected_shapes[stage]
    plans = plan.transaction_plans
    _require_plan_types(plans, records, types)
    index = plan.index.expected
    control = plan.control.expected
    recovery_before = plan.recovery_control.before
    recovery_after = plan.recovery_control.after
    action_before = (
        None if stage == "ARM" else plan.action.before
    )
    action_after = (
        plan.action.item if stage == "ARM" else plan.action.after
    )
    validate_record("glm52_production_activation_index", index)
    validate_record("glm52_production_control", control)
    validate_record(
        "glm52_production_recovery_control", recovery_before
    )
    validate_record(
        "glm52_production_recovery_control", recovery_after
    )
    validate_record("glm52_production_recovery_action", action_after)
    if action_before is not None:
        validate_record(
            "glm52_production_recovery_action", action_before
        )
        validate_transition(
            "glm52_production_recovery_action",
            action_before,
            action_after,
        )
    _require_activation_binding(
        index,
        control,
        recovery_before,
        recovery_after,
        action_after,
    )
    if action_before is not None:
        _require_activation_binding(index, action_before)
    if (
        control.get("phase") != "RECOVERY_SEALING"
        or recovery_before.get("state") != "OWNED"
        or recovery_after.get("state") != "OWNED"
        or recovery_after.get("revision")
        != recovery_before.get("revision") + 1
    ):
        raise ValueError("recovery handoff control/revision binding drifted")
    _require_exact_changed_fields(
        plan.recovery_control,
        allowed={"revision", "updated_at", "canonical_body_sha256"},
        required={"revision", "updated_at"},
        label="recovery handoff control",
    )
    owner_fields = (
        "campaign_identity_sha256",
        "activation_id",
        "activation_ordinal",
        "owner_attempt",
        "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256",
        "owner_invocation_nonce_sha256",
        "owner_hard_expires_at",
    )
    _require_same_identity(recovery_before, action_after, owner_fields)
    if action_before is not None:
        mutable_action_fields = {
            "state",
            "authority_audit_body_sha256",
            "authority_audit_closing_revision",
            "authorized_transition_from_revision",
            "authorized_transition_to_revision",
            "consumed_at",
            "completed_at",
            "response_identity_sha256",
            "reconciliation_identity_sha256",
            "consume_transaction_client_request_token_sha256",
            "revision",
        }
        _require_same_identity(
            action_before,
            action_after,
            tuple(
                field
                for field in action_before
                if field not in mutable_action_fields
            ),
        )
    if (
        action_after.get("authority_domain") != "RECOVERY"
        or action_after.get("action_kind") != "RECOVERY_HANDOFF"
        or action_after.get("attempt") != 1
        or action_after.get("authority_barrier_nonce_sha256")
        != recovery_before.get("recovery_barrier_nonce_sha256")
    ):
        raise ValueError("recovery handoff action ownership drifted")
    if stage == "ARM":
        if (
            action_after.get("state") != "ARMED"
            or action_after.get("revision") != 1
            or recovery_after.get("updated_at")
            != action_after.get("armed_at")
        ):
            raise ValueError("recovery handoff ARM binding drifted")
    elif stage == "CONSUME":
        if (
            action_before.get("state") != "ARMED"
            or action_after.get("state") != "CONSUMED"
            or action_after.get("authority_audit_closing_revision")
            != recovery_before.get("revision")
            or action_after.get("authorized_transition_from_revision")
            != recovery_before.get("revision")
            or action_after.get("authorized_transition_to_revision")
            != recovery_after.get("revision")
            or recovery_after.get("updated_at")
            != action_after.get("consumed_at")
        ):
            raise ValueError("recovery handoff consume binding drifted")
    else:
        writer = plan.writer_control.item
        validate_record(
            "glm52_task12_versioned_writer_control_v1", writer
        )
        if (
            action_before.get("state") != "CONSUMED"
            or action_after.get("state") != "COMPLETED"
            or recovery_before.get("revision")
            != action_before.get("authorized_transition_to_revision")
            or recovery_after.get("updated_at")
            != action_after.get("completed_at")
            or writer.get("writer_kind") != "RecoveryHandoff"
            or writer.get("object_key")
            != action_after.get("candidate_key")
            or writer.get("body_sha256")
            != action_after.get("candidate_body_sha256")
            or writer.get("candidate_identity_sha256")
            != action_after.get("request_body_sha256")
            or writer.get("writer_result_identity_sha256")
            != action_after.get("response_identity_sha256")
            or writer.get("published_at")
            != action_after.get("consumed_at")
        ):
            raise ValueError("recovery handoff completion binding drifted")


def _validate_control_revision_update(update: ExactUpdate) -> None:
    validate_record("glm52_production_control", update.before)
    validate_record("glm52_production_control", update.after)
    if update.after["revision"] != update.before["revision"] + 1:
        raise ValueError("control revision must increment exactly once")
    immutable = set(update.before) - {
        "active_epoch",
        "active_execution_arn",
        "active_state_machine_version_arn",
        "last_sky_post_action_key",
        "last_sky_post_state",
        "revision",
        "updated_at",
    }
    if any(update.before[field] != update.after[field] for field in immutable):
        raise ValueError("control update changed a forbidden field")


def _validate_rollover_binding(
    index: ExactUpdate,
    rollover: ExactPut,
    controls: Tuple[ExactPut, ExactPut, ExactPut, ExactPut],
    *,
    require_final_identity: bool,
) -> None:
    activation_ordinal = rollover.item.get("activation_ordinal")
    if type(activation_ordinal) is not int:
        raise ValueError("rollover activation ordinal binding mismatch")
    if activation_ordinal == 1:
        if index.before != {}:
            raise ValueError("first rollover requires exact empty index prestate")
    elif not index.before:
        raise ValueError("later rollover requires an exact prior index")
    before_revision = 0 if not index.before else index.before.get("revision")
    after_revision = index.after.get("revision")
    if (
        type(before_revision) is not int
        or after_revision != before_revision + 1
        or rollover.item.get("index_from_revision") != before_revision
        or rollover.item.get("index_to_revision") != after_revision
    ):
        raise ValueError("rollover index revision binding mismatch")
    run_id = rollover.item.get("run_id")
    campaign_identity = rollover.item.get("campaign_identity_sha256")
    if index.before and (
        index.before.get("run_id") != run_id
        or index.before.get("campaign_identity_sha256")
        != campaign_identity
        or index.before.get("current_activation_id")
        != rollover.item.get("prior_activation_id")
        or index.before.get("current_activation_ordinal")
        != rollover.item.get("prior_activation_ordinal")
    ):
        raise ValueError("rollover prior index binding mismatch")
    terminal_identity = _frozen_rollover_predecessor_identity(
        rollover.item,
        key_field="prior_terminal_v2_key",
        version_field="prior_terminal_v2_version_id",
        body_field="prior_terminal_v2_body_sha256",
    )
    drained_identity = _frozen_rollover_predecessor_identity(
        rollover.item,
        key_field="prior_h1g_drained_key",
        version_field="prior_h1g_drained_version_id",
        body_field="prior_h1g_drained_body_sha256",
    )
    expected_after = {
        "run_id": run_id,
        "campaign_identity_sha256": campaign_identity,
        "current_activation_id": rollover.item.get("activation_id"),
        "current_activation_ordinal": activation_ordinal,
        "prior_activation_id": rollover.item.get("prior_activation_id"),
        "prior_activation_terminal_v2_identity": terminal_identity,
        "prior_h1g_drained_identity": drained_identity,
        "prior_spend_ledger_head_identity": rollover.item.get(
            "prior_spend_ledger_head_identity"
        ),
        "snapshot_cleanup_lineage_sha256": rollover.item.get(
            "snapshot_cleanup_lineage_sha256"
        ),
    }
    if any(
        index.after.get(field) != expected
        for field, expected in expected_after.items()
    ):
        raise ValueError("rollover final index predecessor binding mismatch")
    if index.before and any(
        index.before.get(field) != index.after.get(field)
        for field in ("run_id", "campaign_identity_sha256")
    ):
        raise ValueError("rollover index changed stable campaign identity")
    activation_id = index.after.get("current_activation_id")
    activation_ordinal = index.after.get("current_activation_ordinal")
    reference = {
        "run_id": index.after.get("run_id"),
        "campaign_identity_sha256": index.after.get(
            "campaign_identity_sha256"
        ),
        "activation_id": activation_id,
        "activation_ordinal": activation_ordinal,
    }
    for record in (rollover.item,) + tuple(plan.item for plan in controls):
        if any(record.get(field) != value for field, value in reference.items()):
            raise ValueError("rollover activation binding mismatch")
    if require_final_identity:
        rollover_identity = rollover.item.get("canonical_body_sha256")
        if type(rollover_identity) is not str or any(
            control.item.get("rollover_identity_sha256")
            != rollover_identity
            for control in controls
        ):
            raise ValueError("rollover final control identity mismatch")


def _frozen_rollover_predecessor_identity(
    rollover: Mapping[str, object],
    *,
    key_field: str,
    version_field: str,
    body_field: str,
) -> Optional[Mapping[str, object]]:
    values = (
        rollover.get(key_field),
        rollover.get(version_field),
        rollover.get(body_field),
    )
    if values == (None, None, None):
        return None
    return {
        "key": values[0],
        "version_id": values[1],
        "body_sha256": values[2],
    }


def _validate_action_consumption(
    index: ExactCheck,
    control: ExactUpdate,
    action: ExactUpdate,
) -> None:
    _require_activation_binding(
        index.expected,
        control.before,
        control.after,
        action.before,
        action.after,
    )
    _require_same_identity(
        index.expected,
        control.before,
        ("run_id", "campaign_identity_sha256"),
    )
    for record in (control.after, action.before, action.after):
        _require_same_identity(
            control.before,
            record,
            (
                "run_id",
                "campaign_identity_sha256",
                "activation_id",
                "activation_ordinal",
            ),
        )
    changed_control = {
        field
        for field in control.before
        if control.before[field] != control.after[field]
    }
    if not changed_control <= {
        "last_sky_post_state",
        "revision",
        "updated_at",
    }:
        raise ValueError("action consume changed protected CONTROL fields")
    if (
        control.before.get("last_sky_post_state") != "ARMED"
        or control.after.get("last_sky_post_state") != "CONSUMED"
        or control.before.get("last_sky_post_action_key")
        != action.key.sort_key
        or control.after.get("last_sky_post_action_key")
        != action.key.sort_key
        or control.before.get("last_sky_post_generation")
        != action.before.get("generation")
        or control.after.get("last_sky_post_generation")
        != action.after.get("generation")
    ):
        raise ValueError("action consume eligibility binding mismatch")
    active_bindings = (
        ("active_epoch", "owner_epoch"),
        ("active_epoch", "armed_by_epoch"),
        ("active_execution_arn", "owner_execution_arn"),
        ("active_execution_arn", "armed_by_execution_arn"),
        (
            "active_state_machine_version_arn",
            "armed_by_state_machine_version_arn",
        ),
        ("barrier_nonce_sha256", "barrier_nonce_sha256"),
    )
    for control_field, action_field in active_bindings:
        expected = control.before.get(control_field)
        if (
            action.before.get(action_field) != expected
            or action.after.get(action_field) != expected
        ):
            raise ValueError("action consume authority binding mismatch")
    before_revision = control.before.get("revision")
    after_revision = control.after.get("revision")
    if (
        action.after.get("authority_audit_closing_revision")
        != before_revision
        or action.after.get("authorized_transition_from_revision")
        != before_revision
        or action.after.get("authorized_transition_to_revision")
        != after_revision
    ):
        raise ValueError("action consume audit revision binding mismatch")


def _validate_liability_revision_only(update: ExactUpdate) -> None:
    validate_record(
        "glm52_production_worker_launch_liability", update.before
    )
    validate_record(
        "glm52_production_worker_launch_liability", update.after
    )
    if (
        update.after.get("state") != update.before.get("state")
        or update.after.get("revision") != update.before.get("revision") + 1
    ):
        raise ValueError("liability evidence update state/revision mismatch")
    mutable = {
        "same_token_completion_attempts",
        "scan_count_current_approval_period",
        "scan_period_started_at",
        "scan_period_ends_at",
        "next_scan_at",
        "last_scan_started_at",
        "last_scan_completed_at",
        "last_scan_evidence_sha256",
        "observed_instance_ids",
        "late_instance_drain_identities",
        "late_instance_termination_action_identities",
        "late_instance_termination_call_counts",
        "post_terminal_allocation_identities",
        "spend_close_identities",
        "revision",
        "updated_at",
    }
    changed = {
        field
        for field in update.before
        if update.before[field] != update.after[field]
    }
    if not changed <= mutable:
        raise ValueError("liability evidence update changed immutable fields")


def _validate_liability_scan_revision(update: ExactUpdate) -> None:
    changed = {
        field
        for field in update.before
        if update.before[field] != update.after[field]
    }
    scan_fields = {
        "scan_count_current_approval_period",
        "scan_period_started_at",
        "scan_period_ends_at",
        "next_scan_at",
        "last_scan_started_at",
        "last_scan_completed_at",
        "last_scan_evidence_sha256",
    }
    if (
        not changed & scan_fields
        or not changed <= scan_fields | {"revision", "updated_at"}
    ):
        raise ValueError(
            "liability scan changed action, spend, allocation, or identity fields"
        )


def _validate_worker_attempt_revision_only(update: ExactUpdate) -> None:
    validate_record("glm52_production_worker_launch", update.before)
    validate_record("glm52_production_worker_launch", update.after)
    if (
        update.after.get("state") != "POSSIBLY_SENT"
        or update.before.get("state") != "POSSIBLY_SENT"
        or update.after.get("revision") != update.before.get("revision") + 1
    ):
        raise ValueError("worker attempt evidence state/revision mismatch")
    changed = {
        field
        for field in update.before
        if update.before[field] != update.after[field]
    }
    if (
        not {
            "run_instances_attempt_evidence",
            "same_token_completion_count",
            "revision",
        }.issubset(changed)
        or not changed
        <= {
            "run_instances_attempt_evidence",
            "same_token_completion_count",
            "revision",
            "updated_at",
        }
        or update.after["run_instances_attempt_evidence"][
            : len(update.before["run_instances_attempt_evidence"])
        ]
        != update.before["run_instances_attempt_evidence"]
        or len(update.after["run_instances_attempt_evidence"])
        != len(update.before["run_instances_attempt_evidence"]) + 1
        or update.after["same_token_completion_count"]
        != update.before["same_token_completion_count"] + 1
    ):
        raise ValueError("worker attempt evidence revision drifted")


def _require_exact_changed_fields(
    update: ExactUpdate,
    *,
    allowed: set,
    required: set,
    label: str,
) -> None:
    changed = {
        field
        for field in update.before
        if update.before[field] != update.after[field]
    }
    if not required <= changed or not changed <= allowed:
        raise ValueError(label + " changed unrelated durable fields")


def _validate_worker_completion_revision(
    update: ExactUpdate,
    *,
    classification: str,
) -> None:
    if classification == "DIRECT_SUCCESS":
        allowed = {
            "state",
            "run_instances_attempt_evidence",
            "observed_instance_ids",
            "instance_observation_sha256",
            "revision",
            "updated_at",
        }
        required = allowed - {"updated_at"}
        if (
            len(update.after["observed_instance_ids"]) != 1
            or update.after["run_instances_attempt_evidence"][
                : len(update.before["run_instances_attempt_evidence"])
            ]
            != update.before["run_instances_attempt_evidence"]
            or len(update.after["run_instances_attempt_evidence"])
            != len(update.before["run_instances_attempt_evidence"]) + 1
        ):
            raise ValueError("direct worker completion evidence drifted")
    elif classification == "POSITIVE_REJECTION":
        allowed = {
            "state",
            "run_instances_attempt_evidence",
            "revision",
            "updated_at",
        }
        required = allowed - {"updated_at"}
        if (
            update.after["run_instances_attempt_evidence"][
                : len(update.before["run_instances_attempt_evidence"])
            ]
            != update.before["run_instances_attempt_evidence"]
            or len(update.after["run_instances_attempt_evidence"])
            != len(update.before["run_instances_attempt_evidence"]) + 1
        ):
            raise ValueError("rejected worker completion evidence drifted")
    else:
        raise ValueError("worker completion classification drifted")
    _require_exact_changed_fields(
        update,
        allowed=allowed,
        required=required,
        label="worker completion",
    )


def _validate_worker_completion_action_revision(
    update: ExactUpdate,
    *,
    classification: str,
) -> None:
    if classification in {"DIRECT_SUCCESS", "POSITIVE_REJECTION"}:
        evidence_field = "response_identity_sha256"
    elif classification == "AMBIGUOUS":
        evidence_field = "reconciliation_identity_sha256"
    else:
        raise ValueError("worker completion classification drifted")
    _require_exact_changed_fields(
        update,
        allowed={
            "state",
            "completed_at",
            evidence_field,
            "revision",
            "updated_at",
        },
        required={
            "state",
            "completed_at",
            evidence_field,
            "revision",
        },
        label="worker completion action",
    )


def _validate_worker_reconciliation_revision(
    update: ExactUpdate,
) -> None:
    edge = (update.before.get("state"), update.after.get("state"))
    edge_fields = {
        ("POSSIBLY_SENT", "INSTANCE_OBSERVED"): {
            "observed_instance_ids",
            "instance_observation_sha256",
        },
        ("POSSIBLY_SENT", "MULTIPLE_INSTANCE_TOKEN_INCIDENT"): {
            "observed_instance_ids",
            "instance_observation_sha256",
            "incident_identity_sha256",
        },
        ("POSSIBLY_SENT", "UNRESOLVED_LAUNCH_INCIDENT"): {
            "incident_identity_sha256",
        },
        ("INSTANCE_OBSERVED", "ALLOCATION_OPEN"): {
            "spend_allocation_open_identity_sha256",
        },
        ("ALLOCATION_OPEN", "INSTANCE_TERMINAL"): {
            "instance_terminal_identity_sha256",
        },
        ("INSTANCE_TERMINAL", "ALLOCATION_CLOSED"): {
            "spend_allocation_close_identity_sha256",
        },
    }
    operation_fields = edge_fields.get(edge)
    if operation_fields is None:
        raise ValueError("worker reconciliation edge is forbidden")
    _require_exact_changed_fields(
        update,
        allowed=operation_fields
        | {"state", "revision", "updated_at"},
        required=operation_fields | {"state", "revision"},
        label="worker reconciliation",
    )


def _validate_liability_action_binding(
    index: ExactCheck,
    liability: ExactUpdate,
    action: ExactUpdate,
    *,
    related_records: Tuple[
        Tuple[Mapping[str, object], Mapping[str, object]], ...
    ],
) -> None:
    _require_activation_binding(
        index.expected,
        liability.before,
        liability.after,
        action.before,
        action.after,
        *(identity for identity, _ in related_records),
    )
    _require_same_identity(
        index.expected,
        liability.before,
        ("run_id", "campaign_identity_sha256"),
    )
    identity_fields = (
        "run_id",
        "campaign_identity_sha256",
        "activation_id",
        "activation_ordinal",
    )
    for record in (liability.after, action.before, action.after):
        _require_same_identity(liability.before, record, identity_fields)
    before_revision = liability.before.get("revision")
    after_revision = liability.after.get("revision")
    if (
        type(before_revision) is not int
        or after_revision != before_revision + 1
        or action.before.get("state") != "ARMED"
        or action.after.get("state") != "CONSUMED"
        or action.after.get("authority_audit_closing_revision")
        != before_revision
        or action.after.get("authorized_transition_from_revision")
        != before_revision
        or action.after.get("authorized_transition_to_revision")
        != after_revision
    ):
        raise ValueError("liability action revision binding mismatch")
    generation = liability.before.get("generation")
    allocation_ordinal = liability.before.get("allocation_ordinal")
    if (
        type(generation) is not int
        or type(allocation_ordinal) is not int
        or liability.after.get("generation") != generation
        or liability.after.get("allocation_ordinal") != allocation_ordinal
    ):
        raise ValueError("liability resource ordinal binding mismatch")
    resource_bindings = {
        "generation": generation,
        "generation_text": "%08d" % generation,
        "allocation_ordinal": allocation_ordinal,
        "allocation_ordinal_text": "%08d" % allocation_ordinal,
        "worker_launch_identity_sha256": liability.before.get(
            "worker_launch_identity_sha256"
        ),
        "worker_launch_liability_identity_sha256": canonical_sha256(
            liability.before
        ),
    }
    owner_fields = (
        "owner_attempt",
        "owner_execution_arn",
        "owner_state_machine_version_arn",
        "owner_dispatch_identity_sha256",
        "owner_invocation_nonce_sha256",
        "owner_hard_expires_at",
    )
    for record in (action.before, action.after):
        if any(
            record.get(field) != expected
            for field, expected in resource_bindings.items()
        ) or any(
            record.get(field) != liability.before.get(field)
            for field in owner_fields
        ):
            raise ValueError("liability action resource binding mismatch")
    related_bindings = {
        field: value
        for field, value in resource_bindings.items()
        if not field.endswith("_text")
    }
    for identity, owner in related_records:
        _require_same_identity(liability.before, identity, identity_fields)
        common_owner_fields = tuple(
            field
            for field in owner_fields
            if field in liability.before and field in owner
        )
        if any(
            identity.get(field) != expected
            for field, expected in related_bindings.items()
        ) or any(
            owner.get(field) != liability.before.get(field)
            for field in common_owner_fields
        ):
            raise ValueError("liability related-record binding mismatch")


def _validate_settlement_final_binding(
    liability: ExactUpdate,
    settlement: ExactPut,
) -> None:
    settlement_identity = settlement.item.get("canonical_body_sha256")
    if (
        type(settlement_identity) is not str
        or liability.after.get("settlement_identity_sha256")
        != settlement_identity
    ):
        raise ValueError("terminal liability settlement identity mismatch")


def _require_same_identity(
    first: Mapping[str, object],
    second: Mapping[str, object],
    fields: Tuple[str, ...],
) -> None:
    if any(first.get(field) != second.get(field) for field in fields):
        raise ValueError("closed record identity binding mismatch")


def _require_activation_binding(
    index: Mapping[str, object],
    *records: Mapping[str, object]
) -> None:
    for record in records:
        if (
            record.get("run_id") != index.get("run_id")
            or record.get("activation_id")
            != index.get("current_activation_id")
            or record.get("activation_ordinal")
            != index.get("current_activation_ordinal")
        ):
            raise ValueError("activation index binding mismatch")


def _find_materialized_plan(
    originals: Tuple[object, ...],
    materialized: Tuple[object, ...],
    selected: object,
) -> object:
    for index, original in enumerate(originals):
        if original is selected:
            return materialized[index]
    raise ValueError("readback plan is outside the transaction")


def _identity_index(items: Tuple[object, ...], selected: object) -> int:
    for index, item in enumerate(items):
        if item is selected:
            return index
    raise ValueError("selected plan is outside the readback set")


def _plan_record(plan: object) -> Mapping[str, object]:
    if type(plan) is ExactUpdate:
        return plan.after
    if type(plan) is ExactPut:
        return plan.item
    if type(plan) is ExactCheck:
        return plan.expected
    raise TypeError("unsupported transaction plan type")


def _plan_record_type(plan: object) -> str:
    record = _plan_record(plan)
    record_type = record.get("record_type")
    if type(record_type) is not str:
        raise ValueError("transaction record_type is malformed")
    return record_type


def _copy_record(record: Mapping[str, object]) -> Dict[str, object]:
    if type(record) is not dict:
        raise TypeError("transaction records must be exact dictionaries")
    return {
        key: _copy_json(item)
        for key, item in record.items()
    }


def _copy_json(value: object) -> object:
    if type(value) is dict:
        return {key: _copy_json(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_json(item) for item in value]
    if type(value) is tuple:
        return tuple(_copy_json(item) for item in value)
    return value


def _replace_plan_record(plan: object, record: Mapping[str, object]) -> object:
    if type(plan) is ExactUpdate:
        return ExactUpdate(plan.key, _copy_record(plan.before), record)
    if type(plan) is ExactPut:
        return ExactPut(plan.key, record)
    if type(plan) is ExactCheck:
        return ExactCheck(plan.key, record)
    raise TypeError("unsupported transaction plan type")


def _normalized_record_for_preimage(
    record: Mapping[str, object],
    *,
    derived_rollover_identity: bool = False,
    derived_settlement_identity: bool = False,
) -> Dict[str, object]:
    normalized = {
        key: _copy_json(value)
        for key, value in record.items()
        if key != "canonical_body_sha256"
        and key != "transaction_bytes_sha256"
        and not key.endswith("transaction_client_request_token_sha256")
    }
    if derived_rollover_identity:
        if "rollover_identity_sha256" not in normalized:
            raise ValueError("rollover control lacks derived identity field")
        normalized["rollover_identity_sha256"] = (
            "__DERIVED_ROLLOVER_CANONICAL_BODY_SHA256__"
        )
    if derived_settlement_identity:
        if "settlement_identity_sha256" not in normalized:
            raise ValueError("liability lacks derived settlement identity field")
        normalized["settlement_identity_sha256"] = (
            "__DERIVED_SETTLEMENT_CANONICAL_BODY_SHA256__"
        )
    return normalized


def _logical_plan_entry(
    plan: object,
    *,
    derived_rollover_identity: bool = False,
    derived_settlement_after: bool = False,
) -> Mapping[str, object]:
    key = {"run_id": plan.key.run_id, "sort_key": plan.key.sort_key}
    if type(plan) is ExactUpdate:
        return {
            "kind": "Update",
            "key": key,
            "condition": (
                "ABSENT_PK_AND_SK" if not plan.before else "EXACT_PRESTATE"
            ),
            "before": _normalized_record_for_preimage(
                plan.before,
                derived_rollover_identity=derived_rollover_identity,
            ),
            "after": _normalized_record_for_preimage(
                plan.after,
                derived_rollover_identity=derived_rollover_identity,
                derived_settlement_identity=derived_settlement_after,
            ),
        }
    if derived_settlement_after:
        raise ValueError("derived settlement identity requires an Update")
    if type(plan) is ExactPut:
        return {
            "kind": "Put",
            "key": key,
            "condition": "ABSENT_PK_AND_SK",
            "item": _normalized_record_for_preimage(
                plan.item,
                derived_rollover_identity=derived_rollover_identity,
            ),
        }
    if type(plan) is ExactCheck:
        return {
            "kind": "ConditionCheck",
            "key": key,
            "condition": "EXACT_EXPECTED",
            "expected": _normalized_record_for_preimage(
                plan.expected,
                derived_rollover_identity=derived_rollover_identity,
            ),
        }
    raise TypeError("unsupported transaction plan type")


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(name + " must be lowercase SHA-256")
    return value


def _require_raw_nonce(raw_owner_nonce: object) -> bytes:
    if type(raw_owner_nonce) is not bytes or len(raw_owner_nonce) != 32:
        raise ValueError("raw owner nonce must be exactly 32 bytes")
    return raw_owner_nonce


def _require_nonce_ownership(
    record: Mapping[str, object],
    field: str,
    raw_owner_nonce: bytes,
) -> None:
    expected = hashlib.sha256(
        _require_raw_nonce(raw_owner_nonce)
    ).hexdigest()
    actual = record.get(field)
    if type(actual) is not str or not hmac.compare_digest(actual, expected):
        raise ValueError("raw nonce does not own the durable record")


def _materialize_transaction(
    plans: Tuple[object, ...],
    *,
    domain: str,
    operation_identity_sha256: str,
    raw_owner_nonce: bytes,
    nonce_targets: Tuple[Tuple[int, str], ...],
    token_targets: Tuple[Tuple[int, str], ...],
    transaction_hash_targets: Tuple[Tuple[int, str], ...],
    rollover_binding: Optional[
        Tuple[int, Tuple[int, int, int, int]]
    ] = None,
    settlement_binding: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[object, ...], str]:
    if type(domain) is not str or not domain:
        raise ValueError("domain must be a nonempty exact string")
    operation_identity_sha256 = _require_sha256(
        "operation_identity_sha256", operation_identity_sha256
    )
    raw_owner_nonce = _require_raw_nonce(raw_owner_nonce)
    materialized = [
        _replace_plan_record(plan, _copy_record(_plan_record(plan)))
        for plan in plans
    ]
    nonce_sha256 = hashlib.sha256(raw_owner_nonce).hexdigest()
    for index, field in nonce_targets:
        record = _copy_record(_plan_record(materialized[index]))
        if field not in record:
            raise ValueError("nonce target field is absent")
        record[field] = nonce_sha256
        materialized[index] = _replace_plan_record(materialized[index], record)
    preimage = {
        "schema_version": 1,
        "domain": domain,
        "operation_identity_sha256": operation_identity_sha256,
        "ordered_logical_write_plan": [
            _logical_plan_entry(
                plan,
                derived_rollover_identity=(
                    rollover_binding is not None
                    and index in rollover_binding[1]
                ),
                derived_settlement_after=(
                    settlement_binding is not None
                    and index == settlement_binding[0]
                ),
            )
            for index, plan in enumerate(materialized)
        ],
    }
    transaction_bytes_sha256 = canonical_sha256(preimage)
    digest = hashlib.sha256()
    for item in (
        b"glm52-ddb-crt-v1",
        domain.encode("utf-8"),
        operation_identity_sha256.encode("ascii"),
        transaction_bytes_sha256.encode("ascii"),
    ):
        digest.update(item)
        digest.update(b"\0")
    digest.update(raw_owner_nonce)
    client_request_token = "h1g-" + digest.hexdigest()[:32]
    token_sha256 = hashlib.sha256(
        client_request_token.encode("ascii")
    ).hexdigest()
    for index, field in transaction_hash_targets:
        record = _copy_record(_plan_record(materialized[index]))
        if field not in record:
            raise ValueError("transaction hash target field is absent")
        record[field] = transaction_bytes_sha256
        materialized[index] = _replace_plan_record(materialized[index], record)
    for index, field in token_targets:
        record = _copy_record(_plan_record(materialized[index]))
        if field not in record:
            raise ValueError("transaction token target field is absent")
        record[field] = token_sha256
        materialized[index] = _replace_plan_record(materialized[index], record)
    if rollover_binding is not None:
        rollover_index, control_indices = rollover_binding
        rollover_plan = materialized[rollover_index]
        rollover_record = _copy_record(_plan_record(rollover_plan))
        if "canonical_body_sha256" not in rollover_record:
            raise ValueError("rollover record lacks canonical self-hash")
        rollover_body = dict(rollover_record)
        rollover_body.pop("canonical_body_sha256")
        rollover_identity = canonical_sha256(rollover_body)
        rollover_record["canonical_body_sha256"] = rollover_identity
        materialized[rollover_index] = _replace_plan_record(
            rollover_plan, rollover_record
        )
        for control_index in control_indices:
            control_plan = materialized[control_index]
            control_record = _copy_record(_plan_record(control_plan))
            if "rollover_identity_sha256" not in control_record:
                raise ValueError("rollover control lacks derived identity field")
            control_record["rollover_identity_sha256"] = rollover_identity
            materialized[control_index] = _replace_plan_record(
                control_plan, control_record
            )
    if settlement_binding is not None:
        liability_index, settlement_index = settlement_binding
        liability_plan = materialized[liability_index]
        settlement_plan = materialized[settlement_index]
        if type(liability_plan) is not ExactUpdate:
            raise ValueError("settlement liability binding requires an Update")
        if type(settlement_plan) is not ExactPut:
            raise ValueError("settlement identity binding requires a Put")
        settlement_record = _copy_record(
            _plan_record(settlement_plan)
        )
        if "canonical_body_sha256" not in settlement_record:
            raise ValueError("settlement record lacks canonical self-hash")
        settlement_body = dict(settlement_record)
        settlement_body.pop("canonical_body_sha256")
        settlement_identity = canonical_sha256(settlement_body)
        settlement_record["canonical_body_sha256"] = settlement_identity
        materialized[settlement_index] = _replace_plan_record(
            settlement_plan, settlement_record
        )
        liability_record = _copy_record(_plan_record(liability_plan))
        if "settlement_identity_sha256" not in liability_record:
            raise ValueError("liability lacks settlement identity field")
        liability_record["settlement_identity_sha256"] = settlement_identity
        materialized[liability_index] = _replace_plan_record(
            liability_plan, liability_record
        )
        _validate_settlement_final_binding(
            materialized[liability_index],
            materialized[settlement_index],
        )
    for index, plan in enumerate(materialized):
        if type(plan) not in (ExactUpdate, ExactPut):
            continue
        record = _copy_record(_plan_record(plan))
        if "canonical_body_sha256" in record:
            body = dict(record)
            body.pop("canonical_body_sha256")
            record["canonical_body_sha256"] = canonical_sha256(body)
            materialized[index] = _replace_plan_record(plan, record)
    return tuple(materialized), client_request_token


def _validate_plans(plans: Tuple[object, ...]) -> None:
    keys = [(plan.key.run_id, plan.key.sort_key) for plan in plans]
    if len(keys) != len(set(keys)):
        raise ValueError("transaction repeats the same ledger key")
    for plan in plans:
        record_type = _plan_record_type(plan)
        pk = ledger_pk(plan.key.run_id)
        if type(plan) is ExactUpdate:
            if plan.before:
                validate_record(
                    record_type,
                    plan.before,
                    pk=pk,
                    sk=plan.key.sort_key,
                )
            validate_record(
                record_type,
                plan.after,
                pk=pk,
                sk=plan.key.sort_key,
            )
        elif type(plan) is ExactPut:
            validate_record(
                record_type,
                plan.item,
                pk=pk,
                sk=plan.key.sort_key,
            )
        elif type(plan) is ExactCheck:
            validate_record(
                record_type,
                plan.expected,
                pk=pk,
                sk=plan.key.sort_key,
            )
        else:
            raise TypeError("unsupported transaction plan type")


def _names_for_fields(fields: Tuple[str, ...]) -> Dict[str, str]:
    return {
        "#n%03d" % index: field
        for index, field in enumerate(sorted(fields))
    }


def _name_by_field(names: Mapping[str, str]) -> Mapping[str, str]:
    return {field: name for name, field in names.items()}


def _wire_item(plan: object, table_name: str) -> Mapping[str, object]:
    if type(plan) is ExactPut:
        physical = {
            "PK": ledger_pk(plan.key.run_id),
            "SK": plan.key.sort_key,
            **_copy_record(plan.item),
        }
        return {
            "Put": {
                "TableName": table_name,
                "Item": encode_item(physical),
                "ConditionExpression": (
                    "attribute_not_exists(#pk) AND "
                    "attribute_not_exists(#sk)"
                ),
                "ExpressionAttributeNames": {"#pk": "PK", "#sk": "SK"},
            }
        }
    if type(plan) is ExactUpdate:
        before_fields = tuple(plan.before)
        after_fields = tuple(plan.after)
        names = _names_for_fields(tuple(set(before_fields) | set(after_fields)))
        by_field = _name_by_field(names)
        values: Dict[str, object] = {}
        if before_fields:
            condition_parts = []
            for index, field in enumerate(sorted(before_fields)):
                placeholder = ":b%03d" % index
                values[placeholder] = encode_attribute_value(plan.before[field])
                condition_parts.append(
                    by_field[field] + " = " + placeholder
                )
            condition = " AND ".join(condition_parts)
        else:
            names = dict(names)
            names["#pk"] = "PK"
            names["#sk"] = "SK"
            condition = (
                "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
            )
        updates = []
        for index, field in enumerate(sorted(after_fields)):
            placeholder = ":a%03d" % index
            values[placeholder] = encode_attribute_value(plan.after[field])
            updates.append(by_field[field] + " = " + placeholder)
        return {
            "Update": {
                "TableName": table_name,
                "Key": _encoded_key(plan.key),
                "ConditionExpression": condition,
                "UpdateExpression": "SET " + ", ".join(updates),
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values,
            }
        }
    if type(plan) is ExactCheck:
        fields = tuple(plan.expected)
        names = _names_for_fields(fields)
        by_field = _name_by_field(names)
        values = {}
        clauses = []
        for index, field in enumerate(sorted(fields)):
            placeholder = ":c%03d" % index
            values[placeholder] = encode_attribute_value(plan.expected[field])
            clauses.append(by_field[field] + " = " + placeholder)
        return {
            "ConditionCheck": {
                "TableName": table_name,
                "Key": _encoded_key(plan.key),
                "ConditionExpression": " AND ".join(clauses),
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values,
            }
        }
    raise TypeError("unsupported transaction plan type")


def _plan_post_record(plan: object) -> Optional[Mapping[str, object]]:
    if type(plan) is ExactUpdate:
        return plan.after
    if type(plan) is ExactPut:
        return plan.item
    if type(plan) is ExactCheck:
        return plan.expected
    raise TypeError("unsupported transaction plan type")


def _plan_pre_record(plan: object) -> Optional[Mapping[str, object]]:
    if type(plan) is ExactUpdate:
        return plan.before if plan.before else None
    if type(plan) is ExactPut:
        return None
    if type(plan) is ExactCheck:
        return plan.expected
    raise TypeError("unsupported transaction plan type")


def _readback_has_live_nonce(
    readback: Tuple[Optional[Mapping[str, object]], ...],
    *,
    raw_owner_nonce: bytes,
    nonce_targets: Tuple[Tuple[int, str], ...],
) -> bool:
    nonce_sha256 = hashlib.sha256(_require_raw_nonce(raw_owner_nonce)).hexdigest()
    for index, field in nonce_targets:
        record = readback[index]
        if record is None:
            return False
        actual = record.get(field)
        if type(actual) is not str or not hmac.compare_digest(
            actual, nonce_sha256
        ):
            return False
    return True


def _readback_is_foreign_nonce(
    readback: Tuple[Optional[Mapping[str, object]], ...],
    *,
    post: Tuple[Optional[Mapping[str, object]], ...],
    raw_owner_nonce: bytes,
    nonce_targets: Tuple[Tuple[int, str], ...],
) -> bool:
    if not nonce_targets or len(readback) != len(post):
        return False
    expected_nonce = hashlib.sha256(
        _require_raw_nonce(raw_owner_nonce)
    ).hexdigest()
    adjusted = [
        None if record is None else _copy_record(record)
        for record in readback
    ]
    foreign = False
    for index, field in nonce_targets:
        record = adjusted[index]
        if record is None or type(record.get(field)) is not str:
            return False
        actual = record[field]
        if not hmac.compare_digest(actual, expected_nonce):
            foreign = True
        expected_record = post[index]
        if expected_record is None:
            return False
        record[field] = expected_record[field]
        if "canonical_body_sha256" in record:
            body = dict(record)
            body.pop("canonical_body_sha256")
            record["canonical_body_sha256"] = canonical_sha256(body)
    return foreign and tuple(adjusted) == post


def _exception_details(
    exc: Exception,
) -> Tuple[
    str,
    Optional[str],
    Tuple[Mapping[str, object], ...],
]:
    error_code = type(exc).__name__
    request_id: Optional[str] = None
    reasons: Tuple[Mapping[str, object], ...] = ()
    response = getattr(exc, "response", None)
    if type(response) is dict:
        error = response.get("Error")
        if type(error) is dict and type(error.get("Code")) is str:
            error_code = error["Code"]
        metadata = response.get("ResponseMetadata")
        if type(metadata) is dict and type(metadata.get("RequestId")) is str:
            request_id = metadata["RequestId"] or None
        raw_reasons = response.get("CancellationReasons")
        if type(raw_reasons) is list and all(
            type(item) is dict for item in raw_reasons
        ):
            reasons = tuple(_copy_record(item) for item in raw_reasons)
    return error_code, request_id, reasons


def encode_attribute_value(value: object) -> Dict[str, object]:
    """Encode one closed JSON-domain value as a DynamoDB AttributeValue."""

    if type(value) is str:
        return {"S": value}
    if type(value) is bool:
        return {"BOOL": value}
    if type(value) is int:
        return {"N": str(value)}
    if value is None:
        return {"NULL": True}
    if type(value) in (list, tuple):
        return {"L": [encode_attribute_value(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("DynamoDB map keys must be exact strings")
        return {
            "M": {
                key: encode_attribute_value(item)
                for key, item in value.items()
            }
        }
    raise TypeError("unsupported closed AttributeValue type")


def decode_attribute_value(value: object) -> object:
    """Decode one closed DynamoDB AttributeValue without coercion."""

    if type(value) is not dict or len(value) != 1:
        raise ValueError("AttributeValue must contain exactly one type tag")
    tag, item = next(iter(value.items()))
    if tag == "S" and type(item) is str:
        return item
    if tag == "N" and type(item) is str and _INTEGER.fullmatch(item):
        return int(item)
    if tag == "BOOL" and type(item) is bool:
        return item
    if tag == "NULL" and item is True:
        return None
    if tag == "L" and type(item) is list:
        return [decode_attribute_value(member) for member in item]
    if tag == "M" and type(item) is dict:
        if any(type(key) is not str for key in item):
            raise ValueError("DynamoDB map keys must be exact strings")
        return {
            key: decode_attribute_value(member)
            for key, member in item.items()
        }
    raise ValueError("unsupported or malformed AttributeValue")


def encode_item(value: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Encode a logical or physical item with exact string field names."""

    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("item must be an exact string-keyed mapping")
    return {key: encode_attribute_value(item) for key, item in value.items()}


def decode_item(value: object) -> Dict[str, object]:
    """Decode an exact string-keyed DynamoDB item."""

    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError("item must be an exact string-keyed mapping")
    return {key: decode_attribute_value(item) for key, item in value.items()}
