"""Pure Task 12 forensic snapshot cleanup and rollover-lineage contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Mapping, Optional, Protocol, Tuple

from .canonical import canonical_sha256
from .dynamodb import TransactionResolution
from .records import canonical_record_identity, validate_record
from .task12_retained_state import SnapshotCleanupTransitionPlan


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT_ID = re.compile(r"snap-[0-9a-f]{17}\Z")
_VOLUME_ID = re.compile(r"vol-[0-9a-f]{17}\Z")
_TERMINAL_STATES = frozenset(
    {"DELETED", "ALREADY_ABSENT", "CLEANUP_INCIDENT"}
)
_MAX_DELETE_ATTEMPTS = 12


def _require_nonempty_ascii(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or any(ord(character) > 127 for character in value)
    ):
        raise ValueError(name + " must be a nonempty ASCII string")
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _parse_utc(name: str, value: object) -> datetime:
    text = _require_nonempty_ascii(name, value)
    if not text.endswith("Z"):
        raise ValueError(name + " must be normalized UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(name + " must be RFC3339 UTC") from exc
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise ValueError(name + " must use whole-second UTC")
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class SnapshotObservation:
    snapshot_id: str
    source_volume_id: str
    kms_key_arn: str
    snapshot_tags_sha256: str
    encrypted: bool
    state: str
    observed_at: str
    describe_request_id: str
    describe_response_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.snapshot_id) is not str
            or _SNAPSHOT_ID.fullmatch(self.snapshot_id) is None
        ):
            raise ValueError("snapshot ID is not exact")
        if (
            type(self.source_volume_id) is not str
            or _VOLUME_ID.fullmatch(self.source_volume_id) is None
        ):
            raise ValueError("source volume ID is not exact")
        kms_key_arn = _require_nonempty_ascii(
            "KMS key ARN", self.kms_key_arn
        )
        if not kms_key_arn.startswith("arn:aws:kms:"):
            raise ValueError("KMS key ARN is outside the closed namespace")
        _require_sha256(
            "snapshot_tags_sha256", self.snapshot_tags_sha256
        )
        if type(self.encrypted) is not bool:
            raise TypeError("encrypted must be an exact boolean")
        if self.state not in {"pending", "completed", "error"}:
            raise ValueError("snapshot state is outside the closed enum")
        _parse_utc("observed_at", self.observed_at)
        _require_nonempty_ascii(
            "describe request ID", self.describe_request_id
        )
        _require_sha256(
            "describe_response_sha256",
            self.describe_response_sha256,
        )

    @property
    def immutable_identity_body(self) -> Mapping[str, object]:
        return {
            "schema_version": 1,
            "snapshot_id": self.snapshot_id,
            "source_volume_id": self.source_volume_id,
            "kms_key_arn": self.kms_key_arn,
            "snapshot_tags_sha256": self.snapshot_tags_sha256,
            "encrypted": self.encrypted,
        }

    @property
    def snapshot_identity_sha256(self) -> str:
        return canonical_sha256(self.immutable_identity_body)

    @property
    def capture_evidence_sha256(self) -> str:
        return canonical_sha256(
            {
                **self.immutable_identity_body,
                "state": self.state,
                "observed_at": self.observed_at,
                "describe_request_id": self.describe_request_id,
                "describe_response_sha256": self.describe_response_sha256,
            }
        )


@dataclass(frozen=True)
class SnapshotCapture:
    snapshot_id: str
    source_volume_id: str
    kms_key_arn: str
    snapshot_tags_sha256: str
    observed_at: str
    snapshot_identity_sha256: str
    capture_evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.snapshot_id) is not str
            or _SNAPSHOT_ID.fullmatch(self.snapshot_id) is None
        ):
            raise ValueError("captured snapshot ID is invalid")
        if (
            type(self.source_volume_id) is not str
            or _VOLUME_ID.fullmatch(self.source_volume_id) is None
        ):
            raise ValueError("captured source volume ID is invalid")
        _require_nonempty_ascii("captured KMS key ARN", self.kms_key_arn)
        _require_sha256(
            "captured snapshot tags", self.snapshot_tags_sha256
        )
        _parse_utc("captured observed_at", self.observed_at)
        _require_sha256(
            "captured snapshot identity", self.snapshot_identity_sha256
        )
        _require_sha256(
            "captured evidence identity", self.capture_evidence_sha256
        )

    @classmethod
    def discover(
        cls,
        *,
        observations: Tuple[SnapshotObservation, ...],
        expected_source_volume_id: str,
        expected_kms_key_arn: str,
        expected_snapshot_tags_sha256: str,
    ) -> "SnapshotCapture":
        if type(observations) is not tuple:
            raise TypeError("snapshot observations must be an exact tuple")
        if (
            len(observations) != 1
            or type(observations[0]) is not SnapshotObservation
        ):
            raise ValueError("snapshot discovery must resolve exactly one")
        observation = observations[0]
        if (
            observation.source_volume_id != expected_source_volume_id
            or observation.kms_key_arn != expected_kms_key_arn
            or observation.snapshot_tags_sha256
            != expected_snapshot_tags_sha256
            or observation.encrypted is not True
            or observation.state != "completed"
        ):
            raise ValueError("snapshot discovery identity mismatch")
        return cls(
            snapshot_id=observation.snapshot_id,
            source_volume_id=observation.source_volume_id,
            kms_key_arn=observation.kms_key_arn,
            snapshot_tags_sha256=observation.snapshot_tags_sha256,
            observed_at=observation.observed_at,
            snapshot_identity_sha256=(
                observation.snapshot_identity_sha256
            ),
            capture_evidence_sha256=(
                observation.capture_evidence_sha256
            ),
        )


@dataclass(frozen=True)
class SnapshotSchedule:
    snapshot_identity_sha256: str
    schedule_arn: str
    delete_not_before: str
    schedule_input_sha256: str
    schedule_identity_sha256: str

    @classmethod
    def create(
        cls, *, capture: SnapshotCapture, schedule_arn: str
    ) -> "SnapshotSchedule":
        delete_not_before = _utc_text(
            _parse_utc("capture observed_at", capture.observed_at)
            + timedelta(days=7)
        )
        return cls.validate(
            capture=capture,
            schedule_arn=schedule_arn,
            delete_not_before=delete_not_before,
            schedule_input_sha256=canonical_sha256({}),
        )

    @classmethod
    def validate(
        cls,
        *,
        capture: SnapshotCapture,
        schedule_arn: str,
        delete_not_before: str,
        schedule_input_sha256: str,
    ) -> "SnapshotSchedule":
        if type(capture) is not SnapshotCapture:
            raise TypeError("capture must be an exact SnapshotCapture")
        expected_deadline = _parse_utc(
            "capture observed_at", capture.observed_at
        ) + timedelta(days=7)
        if (
            _parse_utc("delete_not_before", delete_not_before)
            != expected_deadline
        ):
            raise ValueError("snapshot cleanup deadline is not seven days")
        if schedule_input_sha256 != canonical_sha256({}):
            raise ValueError("snapshot cleanup schedule must be parameterless")
        snapshot_identity = _require_sha256(
            "snapshot_identity_sha256",
            capture.snapshot_identity_sha256,
        )
        exact_schedule_arn = _require_nonempty_ascii(
            "schedule ARN", schedule_arn
        )
        body = {
            "schema_version": 1,
            "snapshot_identity_sha256": snapshot_identity,
            "schedule_arn": exact_schedule_arn,
            "delete_not_before": delete_not_before,
            "schedule_input_sha256": schedule_input_sha256,
        }
        return cls(
            snapshot_identity_sha256=snapshot_identity,
            schedule_arn=exact_schedule_arn,
            delete_not_before=delete_not_before,
            schedule_input_sha256=schedule_input_sha256,
            schedule_identity_sha256=canonical_sha256(body),
        )


@dataclass(frozen=True, init=False)
class CleanupTransitionProof:
    identity_sha256: str
    activation_ordinal: int
    cleanup_control_root_identity_sha256: str
    prior_transition_sha256: str

    @classmethod
    def from_record(
        cls, record: Mapping[str, object]
    ) -> "CleanupTransitionProof":
        validated = validate_record(
            "glm52_production_snapshot_cleanup_transition", record
        )
        identity = canonical_record_identity(
            "glm52_production_snapshot_cleanup_transition", validated
        )
        if validated["canonical_body_sha256"] != identity:
            raise ValueError("cleanup transition self-identity mismatch")
        proof = object.__new__(cls)
        object.__setattr__(proof, "identity_sha256", identity)
        object.__setattr__(
            proof, "activation_ordinal", validated["activation_ordinal"]
        )
        object.__setattr__(
            proof,
            "cleanup_control_root_identity_sha256",
            validated["cleanup_control_root_identity_sha256"],
        )
        object.__setattr__(
            proof,
            "prior_transition_sha256",
            validated["prior_transition_sha256"],
        )
        return proof


@dataclass(frozen=True)
class VerifiedCleanupLineage:
    lineage_identity_sha256: str
    recorded_activation_ordinals: Tuple[int, ...]
    live_chain_heads: Tuple[Tuple[int, str], ...]


class SnapshotCleanupAdapter(Protocol):
    def commit_snapshot_cleanup_transition(
        self,
        *,
        plan: SnapshotCleanupTransitionPlan,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        ...


class SnapshotDeleteClient(Protocol):
    def delete_snapshot(self, *, snapshot_id: str) -> Mapping[str, object]:
        ...

    def describe_snapshot(
        self, *, snapshot_id: str
    ) -> Optional[SnapshotObservation]:
        ...


@dataclass(frozen=True)
class DeleteAttemptResult:
    outcome: str
    resolution: TransactionResolution
    request_id: Optional[str]
    response_sha256: Optional[str]


@dataclass(frozen=True)
class SnapshotOwnerAcquisitionResult:
    """Authenticated committed control state selected after acquisition."""

    cleanup_state: str
    resolution: TransactionResolution

    def __post_init__(self) -> None:
        if self.cleanup_state not in {
            "OWNED",
            "DELETE_POSSIBLY_SENT",
            "DELETE_RECONCILING",
        }:
            raise ValueError("snapshot acquisition state is not resumable")
        if type(self.resolution) is not TransactionResolution:
            raise TypeError(
                "snapshot acquisition result requires an exact transaction"
            )


@dataclass(frozen=True)
class ReconciliationResult:
    source_state: str
    target_state: str
    committed: bool
    snapshot_present: bool
    delete_logical_attempt: int
    resolution: Optional[TransactionResolution]


_LINEAGE_FIELDS = frozenset(
    {
        "activation_ordinal",
        "cleanup_control_root_identity_sha256",
        "cleanup_transition_chain_head_sha256",
        "cleanup_transition_chain_length",
        "transition_identities",
    }
)


def _normalize_lineage(
    value: object, *, expected_ordinals: Tuple[int, ...]
) -> Tuple[Mapping[str, object], ...]:
    if type(value) is not tuple or len(value) != len(expected_ordinals):
        raise ValueError("cleanup lineage cardinality is not transitive")
    normalized = []
    for raw, expected_ordinal in zip(value, expected_ordinals):
        if type(raw) is not dict or set(raw) != _LINEAGE_FIELDS:
            raise ValueError("cleanup lineage entry schema mismatch")
        if raw["activation_ordinal"] != expected_ordinal:
            raise ValueError("cleanup lineage ordinal is incomplete")
        root = _require_sha256(
            "cleanup root",
            raw["cleanup_control_root_identity_sha256"],
        )
        head = _require_sha256(
            "cleanup head",
            raw["cleanup_transition_chain_head_sha256"],
        )
        transitions = raw["transition_identities"]
        if type(transitions) is not tuple:
            raise TypeError("transition identities must be an exact tuple")
        for identity in transitions:
            _require_sha256("transition identity", identity)
        if (
            len(set(transitions)) != len(transitions)
            or root in transitions
            or raw["cleanup_transition_chain_length"] != len(transitions)
            or head != (root if not transitions else transitions[-1])
        ):
            raise ValueError("cleanup lineage head or length mismatch")
        normalized.append(
            {
                "activation_ordinal": expected_ordinal,
                "cleanup_control_root_identity_sha256": root,
                "cleanup_transition_chain_head_sha256": head,
                "cleanup_transition_chain_length": len(transitions),
                "transition_identities": tuple(transitions),
            }
        )
    return tuple(normalized)


def _json_lineage(
    lineage: Tuple[Mapping[str, object], ...]
) -> list[Mapping[str, object]]:
    return [
        {
            **entry,
            "transition_identities": list(entry["transition_identities"]),
        }
        for entry in lineage
    ]


def verify_three_activation_lineage(
    *,
    second_activation_lineage: Tuple[Mapping[str, object], ...],
    third_activation_lineage: Tuple[Mapping[str, object], ...],
    third_lineage_sha256: str,
    live_transition_chains: Mapping[
        int, Tuple[CleanupTransitionProof, ...]
    ],
) -> VerifiedCleanupLineage:
    """Prove activation-three lineage plus monotonic live descendants."""
    second = _normalize_lineage(
        second_activation_lineage, expected_ordinals=(1,)
    )
    third = _normalize_lineage(
        third_activation_lineage, expected_ordinals=(1, 2)
    )
    expected_hash = canonical_sha256(_json_lineage(third))
    if third_lineage_sha256 != expected_hash:
        raise ValueError("third-activation cleanup lineage hash mismatch")
    prior_first = second[0]
    current_first = third[0]
    prior_transitions = prior_first["transition_identities"]
    current_transitions = current_first["transition_identities"]
    if (
        prior_first["cleanup_control_root_identity_sha256"]
        != current_first["cleanup_control_root_identity_sha256"]
        or current_transitions[: len(prior_transitions)]
        != prior_transitions
    ):
        raise ValueError("cleanup lineage forked after rollover")
    recorded_identities = [
        identity
        for entry in third
        for identity in entry["transition_identities"]
    ]
    if len(set(recorded_identities)) != len(recorded_identities):
        raise ValueError("cleanup lineage reuses a transition identity")
    if (
        type(live_transition_chains) is not dict
        or set(live_transition_chains) != {1, 2}
    ):
        raise ValueError("live cleanup lineage membership is incomplete")
    heads = []
    for entry in third:
        ordinal = entry["activation_ordinal"]
        chain = live_transition_chains[ordinal]
        if type(chain) is not tuple:
            raise TypeError("live transition chain must be an exact tuple")
        root = entry["cleanup_control_root_identity_sha256"]
        prior = root
        identities = []
        for proof in chain:
            if type(proof) is not CleanupTransitionProof:
                raise TypeError("live cleanup transition proof is not exact")
            if (
                proof.activation_ordinal != ordinal
                or proof.cleanup_control_root_identity_sha256 != root
                or proof.prior_transition_sha256 != prior
            ):
                raise ValueError("live cleanup transition chain has a gap")
            identities.append(proof.identity_sha256)
            prior = proof.identity_sha256
        if len(set(identities)) != len(identities):
            raise ValueError("live cleanup transition chain contains a cycle")
        recorded = entry["transition_identities"]
        if tuple(identities[: len(recorded)]) != recorded:
            raise ValueError("live cleanup chain is not a recorded descendant")
        heads.append((ordinal, prior))
    return VerifiedCleanupLineage(
        lineage_identity_sha256=expected_hash,
        recorded_activation_ordinals=(1, 2),
        live_chain_heads=tuple(heads),
    )


class SnapshotCleanupCoordinator:
    """Pure orchestration over injected retained-state and snapshot ports."""

    def __init__(
        self,
        *,
        adapter: SnapshotCleanupAdapter,
        snapshot_client: SnapshotDeleteClient,
    ) -> None:
        self._adapter = adapter
        self._snapshot_client = snapshot_client

    @staticmethod
    def _records(
        plan: SnapshotCleanupTransitionPlan,
    ) -> Tuple[Mapping[str, object], Mapping[str, object]]:
        if type(plan) is not SnapshotCleanupTransitionPlan:
            raise TypeError(
                "cleanup operation requires a typed retained-state plan"
            )
        before = plan.cleanup_control.before
        after = plan.cleanup_control.after
        if type(before) is not dict or type(after) is not dict:
            raise TypeError("cleanup control plans require exact dictionaries")
        return before, after

    @staticmethod
    def _validate_binding(
        before: Mapping[str, object],
        after: Mapping[str, object],
    ) -> None:
        fields = (
            "snapshot_id",
            "snapshot_identity_sha256",
            "source_volume_id",
            "snapshot_tags_sha256",
            "delete_not_before",
            "schedule_arn",
            "schedule_input_sha256",
        )
        if any(
            before.get(field) != after.get(field) for field in fields
        ):
            raise ValueError("snapshot binding changed across cleanup edge")
        if (
            type(after.get("snapshot_id")) is not str
            or _SNAPSHOT_ID.fullmatch(after["snapshot_id"]) is None
        ):
            raise ValueError("snapshot binding has an invalid snapshot ID")
        _require_sha256(
            "snapshot_identity_sha256",
            after.get("snapshot_identity_sha256"),
        )
        _parse_utc("delete_not_before", after.get("delete_not_before"))

    def _commit(
        self,
        *,
        plan: SnapshotCleanupTransitionPlan,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        _require_nonempty_ascii("cleanup domain", domain)
        _require_sha256(
            "operation_identity_sha256", operation_identity_sha256
        )
        if type(raw_owner_nonce) is not bytes or len(raw_owner_nonce) != 32:
            raise ValueError("raw owner nonce must be exactly 32 bytes")
        result = self._adapter.commit_snapshot_cleanup_transition(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )
        if type(result) is not TransactionResolution:
            raise TypeError("cleanup adapter returned a malformed resolution")
        return result

    def arm(
        self,
        *,
        plan: SnapshotCleanupTransitionPlan,
        capture: SnapshotCapture,
        schedule: SnapshotSchedule,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        before, after = self._records(plan)
        if (
            before.get("state") != "DORMANT"
            or after.get("state") != "ARMED"
        ):
            raise ValueError("snapshot arm requires DORMANT to ARMED")
        validated_schedule = SnapshotSchedule.validate(
            capture=capture,
            schedule_arn=schedule.schedule_arn,
            delete_not_before=schedule.delete_not_before,
            schedule_input_sha256=schedule.schedule_input_sha256,
        )
        if validated_schedule != schedule:
            raise ValueError("snapshot schedule is not canonical")
        expected = {
            "snapshot_id": capture.snapshot_id,
            "snapshot_identity_sha256": capture.snapshot_identity_sha256,
            "source_volume_id": capture.source_volume_id,
            "snapshot_tags_sha256": capture.snapshot_tags_sha256,
            "delete_not_before": schedule.delete_not_before,
            "schedule_arn": schedule.schedule_arn,
            "schedule_input_sha256": schedule.schedule_input_sha256,
        }
        if any(after.get(field) != value for field, value in expected.items()):
            raise ValueError("armed cleanup does not bind exact capture")
        if (
            schedule.snapshot_identity_sha256
            != capture.snapshot_identity_sha256
        ):
            raise ValueError("schedule does not bind captured snapshot")
        return self._commit(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def acquire(
        self,
        *,
        plan: SnapshotCleanupTransitionPlan,
        observed_at: str,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> TransactionResolution:
        before, after = self._records(plan)
        if (
            before.get("state") != "ARMED"
            or after.get("state") != "OWNED"
        ):
            raise ValueError("snapshot owner acquisition requires ARMED")
        self._validate_binding(before, after)
        if _parse_utc("observed_at", observed_at) < _parse_utc(
            "delete_not_before", after["delete_not_before"]
        ):
            raise ValueError("snapshot owner acquisition is before deadline")
        return self._commit(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )

    def delete_once(
        self,
        *,
        plan: SnapshotCleanupTransitionPlan,
        observed_at: str,
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
    ) -> DeleteAttemptResult:
        before, after = self._records(plan)
        if before.get("state") == "DELETE_POSSIBLY_SENT":
            raise ValueError(
                "ambiguous delete requires readback before another call"
            )
        if (
            before.get("state") not in {"OWNED", "DELETE_RECONCILING"}
            or after.get("state") != "DELETE_POSSIBLY_SENT"
        ):
            raise ValueError("snapshot delete staging edge is not closed")
        self._validate_binding(before, after)
        if _parse_utc("observed_at", observed_at) < _parse_utc(
            "delete_not_before", after["delete_not_before"]
        ):
            raise ValueError("snapshot delete is before deadline")
        prior_attempt = before.get("delete_logical_attempt")
        next_attempt = after.get("delete_logical_attempt")
        prior_calls = before.get("delete_call_count")
        next_calls = after.get("delete_call_count")
        if (
            type(prior_attempt) is not int
            or type(next_attempt) is not int
            or type(prior_calls) is not int
            or type(next_calls) is not int
            or next_attempt != prior_attempt + 1
            or next_calls != prior_calls + 1
            or next_attempt > _MAX_DELETE_ATTEMPTS
            or next_calls > _MAX_DELETE_ATTEMPTS
        ):
            raise ValueError("snapshot delete exceeds the 12-attempt cap")
        resolution = self._commit(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )
        if not resolution.may_issue_external_side_effect:
            return DeleteAttemptResult(
                "NOT_COMMITTED", resolution, None, None
            )
        try:
            response = self._snapshot_client.delete_snapshot(
                snapshot_id=after["snapshot_id"]
            )
        except Exception:
            return DeleteAttemptResult(
                "AMBIGUOUS", resolution, None, None
            )
        if type(response) is not dict or set(response) != {
            "outcome",
            "request_id",
            "response_sha256",
        }:
            raise ValueError("delete response is not closed")
        outcome = response["outcome"]
        if outcome not in {
            "ACCEPTED",
            "DELETION_IN_PROGRESS",
            "NOT_FOUND",
        }:
            raise ValueError("delete response outcome is not closed")
        request_id = _require_nonempty_ascii(
            "delete request ID", response["request_id"]
        )
        response_sha256 = _require_sha256(
            "delete response SHA-256", response["response_sha256"]
        )
        return DeleteAttemptResult(
            outcome, resolution, request_id, response_sha256
        )

    def reconcile_readback(
        self,
        *,
        plan: Optional[SnapshotCleanupTransitionPlan],
        domain: str,
        operation_identity_sha256: str,
        raw_owner_nonce: bytes,
        current_control: Optional[Mapping[str, object]] = None,
    ) -> ReconciliationResult:
        if plan is None:
            if type(current_control) is not dict:
                raise TypeError(
                    "readback without a plan requires exact current control"
                )
            before = current_control
        else:
            before, after = self._records(plan)
            self._validate_binding(before, after)
        state = before.get("state")
        if state in _TERMINAL_STATES:
            raise ValueError("terminal snapshot cleanup cannot reconcile")
        snapshot_id = before.get("snapshot_id")
        if (
            type(snapshot_id) is not str
            or _SNAPSHOT_ID.fullmatch(snapshot_id) is None
        ):
            raise ValueError("cleanup readback snapshot binding is invalid")
        observed = self._snapshot_client.describe_snapshot(
            snapshot_id=snapshot_id
        )
        if observed is not None:
            if type(observed) is not SnapshotObservation:
                raise TypeError("snapshot readback observation is malformed")
            if (
                observed.snapshot_id != snapshot_id
                or observed.source_volume_id
                != before.get("source_volume_id")
                or observed.snapshot_tags_sha256
                != before.get("snapshot_tags_sha256")
                or observed.snapshot_identity_sha256
                != before.get("snapshot_identity_sha256")
            ):
                raise ValueError("readback returned a foreign snapshot")
        attempt = before.get("delete_logical_attempt")
        if type(attempt) is not int or attempt < 0:
            raise ValueError("cleanup logical attempt is invalid")
        if observed is None:
            targets = {
                "OWNED": "ALREADY_ABSENT",
                "DELETE_POSSIBLY_SENT": "DELETE_RECONCILING",
                "DELETE_RECONCILING": "DELETED",
            }
            target = targets.get(state)
        elif state == "DELETE_POSSIBLY_SENT":
            target = "DELETE_RECONCILING"
        elif state == "DELETE_RECONCILING" and (
            attempt >= _MAX_DELETE_ATTEMPTS
        ):
            target = "CLEANUP_INCIDENT"
        else:
            target = state
        if target is None:
            raise ValueError("cleanup readback state is not reconcilable")
        if plan is None:
            return ReconciliationResult(
                state,
                target,
                False,
                observed is not None,
                attempt,
                None,
            )
        if target == state:
            raise ValueError(
                "present snapshot needs a new closed delete attempt"
            )
        if after.get("state") != target:
            raise ValueError("cleanup readback target state mismatch")
        resolution = self._commit(
            plan=plan,
            domain=domain,
            operation_identity_sha256=operation_identity_sha256,
            raw_owner_nonce=raw_owner_nonce,
        )
        return ReconciliationResult(
            state,
            target,
            True,
            observed is not None,
            attempt,
            resolution,
        )
