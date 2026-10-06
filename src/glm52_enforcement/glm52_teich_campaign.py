"""Enforcement-native campaign state and Capacity Block deadline policy."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping

_HEX64 = re.compile(r"[0-9a-f]{64}")


class CampaignPhase(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    CUDA_GATE = "CUDA_GATE"
    TEACHER = "TEACHER"
    CACHE_AUDIT = "CACHE_AUDIT"
    TRAINING_PREFLIGHT = "TRAINING_PREFLIGHT"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    DRAINED = "DRAINED"


_PHASES = tuple(CampaignPhase)


class CampaignTransitionError(ValueError):
    pass


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_identities(values: Mapping[str, str], *, label: str) -> None:
    if any(not isinstance(key, str) or not key for key in values):
        raise CampaignTransitionError(f"{label} identity names must be non-empty strings")
    for name, value in values.items():
        if _HEX64.fullmatch(value) is None:
            raise CampaignTransitionError(f"{label} identity {name!r} must be SHA-256")


@dataclass(frozen=True)
class CampaignRecord:
    phase: CampaignPhase
    input_identities: dict[str, str]
    output_identities: dict[str, str]
    timestamp: str
    reservation_deadline: str
    prior_record_sha256: str
    record_sha256: str


class CampaignLedger:
    """Append-only local ledger designed to be mirrored to S3 after each append."""

    def __init__(self, path: str | Path, *, run_id: str, reservation_deadline: str):
        self.path = Path(path)
        self.run_id = run_id
        self.reservation_deadline = reservation_deadline
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("campaign run_id must be a path-safe identifier")
        try:
            deadline = datetime.fromisoformat(reservation_deadline.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("reservation_deadline must be ISO-8601") from error
        if deadline.tzinfo is None:
            raise ValueError("reservation_deadline must be timezone-aware")
        self.records = self._read_and_validate()

    @property
    def _genesis(self) -> str:
        return _sha256(
            {
                "record_type": "glm52_teich_campaign_ledger_genesis_v1",
                "run_id": self.run_id,
                "reservation_deadline": self.reservation_deadline,
            }
        )

    @property
    def current_phase(self) -> CampaignPhase | None:
        return self.records[-1].phase if self.records else None

    def _read_and_validate(self) -> list[CampaignRecord]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise CampaignTransitionError("campaign ledger must be a regular non-symlink file")
        records: list[CampaignRecord] = []
        previous_sha = self._genesis
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line:
                raise CampaignTransitionError(f"campaign ledger line {line_number} is empty")
            try:
                value = json.loads(line)
            except Exception as error:
                raise CampaignTransitionError(
                    f"campaign ledger line {line_number} is malformed: {error}"
                ) from error
            if not isinstance(value, dict):
                raise CampaignTransitionError("campaign ledger records must be objects")
            body = dict(value)
            record_sha = body.pop("record_sha256", None)
            if record_sha != _sha256(body):
                raise CampaignTransitionError("campaign ledger record SHA-256 mismatch")
            if body.get("record_type") != "glm52_teich_campaign_transition_v1":
                raise CampaignTransitionError("campaign ledger record schema mismatch")
            if body.get("run_id") != self.run_id or body.get(
                "reservation_deadline"
            ) != self.reservation_deadline:
                raise CampaignTransitionError("campaign ledger contains a foreign run")
            if body.get("prior_record_sha256") != previous_sha:
                raise CampaignTransitionError("campaign ledger prior-record hash mismatch")
            try:
                phase = CampaignPhase(str(body.get("phase")))
            except ValueError as error:
                raise CampaignTransitionError("campaign ledger phase is invalid") from error
            expected_phase = _PHASES[len(records)] if len(records) < len(_PHASES) else None
            if phase != expected_phase:
                raise CampaignTransitionError("campaign ledger phases are not contiguous")
            inputs = body.get("input_identities")
            outputs = body.get("output_identities")
            if not isinstance(inputs, dict) or not isinstance(outputs, dict):
                raise CampaignTransitionError("campaign ledger identities must be objects")
            _validate_identities(inputs, label="input")
            _validate_identities(outputs, label="output")
            record = CampaignRecord(
                phase=phase,
                input_identities=dict(inputs),
                output_identities=dict(outputs),
                timestamp=str(body.get("timestamp")),
                reservation_deadline=self.reservation_deadline,
                prior_record_sha256=previous_sha,
                record_sha256=str(record_sha),
            )
            records.append(record)
            previous_sha = record.record_sha256
        return records

    def transition(
        self,
        phase: CampaignPhase,
        *,
        input_identities: Mapping[str, str],
        output_identities: Mapping[str, str],
        timestamp: datetime | None = None,
    ) -> CampaignRecord:
        expected = _PHASES[len(self.records)] if len(self.records) < len(_PHASES) else None
        if phase != expected:
            raise CampaignTransitionError(
                f"campaign next phase must be {expected.value if expected else 'none'}"
            )
        _validate_identities(input_identities, label="input")
        _validate_identities(output_identities, label="output")
        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("campaign transition timestamp must be timezone-aware")
        previous_sha = self.records[-1].record_sha256 if self.records else self._genesis
        body: dict[str, object] = {
            "record_type": "glm52_teich_campaign_transition_v1",
            "run_id": self.run_id,
            "phase": phase.value,
            "input_identities": dict(sorted(input_identities.items())),
            "output_identities": dict(sorted(output_identities.items())),
            "timestamp": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "reservation_deadline": self.reservation_deadline,
            "prior_record_sha256": previous_sha,
        }
        record_sha = _sha256(body)
        encoded = _canonical_bytes({**body, "record_sha256": record_sha}) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY,
            0o600,
        )
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short campaign ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.records = self._read_and_validate()
        return self.records[-1]


@dataclass(frozen=True)
class TrainingPreflightDecision:
    start_training: bool
    reason: str
    projected_runtime_seconds: float
    safety_factored_runtime_seconds: float
    seconds_until_compute_deadline: float
    safety_factor: float = 1.5


def training_preflight_decision(
    *,
    peak_gpu_gib: float,
    sample_steps: int,
    sample_seconds: float,
    remaining_steps: int,
    now: datetime,
    capacity_block_end: datetime | None = None,
    execution_deadline: datetime | None = None,
) -> TrainingPreflightDecision:
    if sample_steps != 20:
        raise ValueError("training preflight timing sample must contain exactly 20 steps")
    if sample_seconds <= 0 or remaining_steps < 0:
        raise ValueError("training preflight timing values are invalid")
    deadlines = [
        value for value in (capacity_block_end, execution_deadline) if value is not None
    ]
    if len(deadlines) != 1:
        raise ValueError(
            "training preflight requires exactly one compute-window deadline"
        )
    deadline = deadlines[0]
    if now.tzinfo is None or deadline.tzinfo is None:
        raise ValueError("training preflight times must be timezone-aware")
    projected = sample_seconds / sample_steps * remaining_steps
    factored = projected * 1.5
    compute_deadline = deadline - timedelta(minutes=60)
    available = (compute_deadline - now).total_seconds()
    if peak_gpu_gib >= 70.0:
        reason = "peak_gpu_memory_gate"
        accepted = False
    elif factored > available:
        reason = "runtime_deadline_gate"
        accepted = False
    else:
        reason = "accepted"
        accepted = True
    return TrainingPreflightDecision(
        start_training=accepted,
        reason=reason,
        projected_runtime_seconds=projected,
        safety_factored_runtime_seconds=factored,
        seconds_until_compute_deadline=available,
    )


def deadline_action(*, now: datetime, capacity_block_end: datetime) -> str:
    if now.tzinfo is None or capacity_block_end.tzinfo is None:
        raise ValueError("deadline times must be timezone-aware")
    remaining = capacity_block_end - now
    if remaining <= timedelta(minutes=30):
        return "AWS_FORCED_WINDOW"
    if remaining <= timedelta(minutes=32):
        return "TERMINATE_IF_DRAINED"
    if remaining <= timedelta(minutes=35):
        return "FINAL_SYNC"
    if remaining <= timedelta(minutes=50):
        return "SIGTERM_WORKERS"
    if remaining <= timedelta(minutes=60):
        return "STOP_ASSIGNING"
    return "RUN"


__all__ = [
    "CampaignLedger",
    "CampaignPhase",
    "CampaignRecord",
    "CampaignTransitionError",
    "TrainingPreflightDecision",
    "deadline_action",
    "training_preflight_decision",
]
