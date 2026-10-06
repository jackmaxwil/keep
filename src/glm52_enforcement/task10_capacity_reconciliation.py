"""Durably publish the sole Task 10 internal-capacity outcome."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .task10_durable_s3 import publish_or_adopt_exact


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
ZONES = ["us-west-2" + value for value in "abcdef"]


class CapacityReconciliationError(ValueError):
    pass


@dataclass(frozen=True)
class CapacityReconciliation:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    generation: int
    generation_text: str
    classification: str
    execution_arn: str
    capacity_outcomes: list[dict[str, object]]
    instance_id: str | None
    availability_zone: str | None
    same_token_identity_sha256: str
    reserve_identity_sha256: str
    spend_authority_identity_sha256: str
    action_identity_sha256: str
    task9_custody_identity_sha256: str
    canonical_identity_sha256: str


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise CapacityReconciliationError(label + " must be a SHA-256")
    return value


def _body(value: CapacityReconciliation) -> dict[str, object]:
    return {name: getattr(value, name) for name in CapacityReconciliation.__dataclass_fields__ if name != "canonical_identity_sha256"}


def validate_capacity_reconciliation(value: object) -> CapacityReconciliation:
    if not isinstance(value, CapacityReconciliation):
        raise CapacityReconciliationError("capacity reconciliation must be typed")
    if (
        value.schema_version != 1 or value.record_type != "glm52_task10_capacity_reconciliation_v1"
        or value.account_id != ACCOUNT_ID or value.region != REGION or value.run_id != RUN_ID
        or not value.activation_id or value.generation < 1 or value.generation_text != "%08d" % value.generation
        or value.classification not in {"RUNNING", "WORKER_ALLOCATED", "CAPACITY_EXHAUSTED", "FAILED"}
        or not value.execution_arn
    ):
        raise CapacityReconciliationError("capacity reconciliation scope drifted")
    outcomes = value.capacity_outcomes
    if type(outcomes) is not list or len(outcomes) > 6:
        raise CapacityReconciliationError("capacity outcome count drifted")
    allocated = []
    for index, outcome in enumerate(outcomes, 1):
        if type(outcome) is not dict or set(outcome) != {"attempt", "availability_zone", "outcome"} or outcome["attempt"] != index or outcome["availability_zone"] != ZONES[index - 1] or outcome["outcome"] not in {"CAPACITY_REJECTED", "WORKER_ALLOCATED"}:
            raise CapacityReconciliationError("capacity outcomes are not ordered")
        if outcome["outcome"] == "WORKER_ALLOCATED":
            allocated.append(outcome)
    if value.classification == "WORKER_ALLOCATED":
        if len(allocated) != 1 or outcomes[-1] != allocated[0] or not value.instance_id or value.availability_zone != allocated[0]["availability_zone"]:
            raise CapacityReconciliationError("worker allocation outcome drifted")
    elif value.classification == "CAPACITY_EXHAUSTED":
        if len(outcomes) != 6 or allocated or value.instance_id is not None or value.availability_zone is not None:
            raise CapacityReconciliationError("capacity exhaustion outcome drifted")
    elif allocated or value.instance_id is not None or value.availability_zone is not None:
        raise CapacityReconciliationError("nonterminal outcome drifted")
    for field in ("same_token_identity_sha256", "reserve_identity_sha256", "spend_authority_identity_sha256", "action_identity_sha256", "task9_custody_identity_sha256"):
        _sha(getattr(value, field), field)
    if value.canonical_identity_sha256 != canonical_sha256(_body(value)):
        raise CapacityReconciliationError("capacity reconciliation self-hash drifted")
    return value


def build_capacity_reconciliation(**values: object) -> CapacityReconciliation:
    provisional = CapacityReconciliation(**{**values, "canonical_identity_sha256": ""})
    body = _body(provisional)
    return validate_capacity_reconciliation(
        CapacityReconciliation(
            **{
                **body,
                "canonical_identity_sha256": canonical_sha256(body),
            }
        )
    )


def outcome_key(value: CapacityReconciliation) -> str:
    value = validate_capacity_reconciliation(value)
    return f"campaigns/{RUN_ID}/submissions/production/generations/{value.generation_text}/workflow/LAUNCH_OUTCOME.json"


def publish_capacity_reconciliation(client: object, *, value: CapacityReconciliation) -> Mapping[str, object]:
    value = validate_capacity_reconciliation(value)
    raw = canonical_json_bytes(_body(value) | {"canonical_identity_sha256": value.canonical_identity_sha256}) + b"\n"
    version_id = publish_or_adopt_exact(client, bucket=BUCKET, key=outcome_key(value), raw=raw, metadata={"record-type": value.record_type, "canonical-identity-sha256": value.canonical_identity_sha256})
    return {"bucket": BUCKET, "key": outcome_key(value), "version_id": version_id, "file_sha256": hashlib.sha256(raw).hexdigest(), "body_sha256": value.canonical_identity_sha256}
