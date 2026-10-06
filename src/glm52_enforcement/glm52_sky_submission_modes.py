"""Enforcement-native registry for managed Sky submission records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ManagedMode = Literal["cache-seed", "qualification", "production"]

RecordKind = Literal[
    "submission-intent",
    "submission-accepted",
    "submission-acquisition",
    "controller-baseline",
    "control-plane-ready",
    "dynamic-job-binding",
    "worker-start-latch",
    "worker-start-accepted",
    "launch-claim",
    "generation-claim",
    "generation-start-decision",
    "generation-terminal",
]

AddressKind = Literal[
    "content-addressed-body",
    "descriptor-file-singleton",
    "intent-singleton",
    "intent-control-body",
    "intent-instance-latch",
    "generation-claim-singleton",
    "generation-start-decision-singleton",
    "generation-terminal-singleton",
]


@dataclass(frozen=True)
class RecordContract:
    """One immutable managed-mode record discriminant."""

    schema_version: int
    record_type: str
    digest_field: str
    address_kind: AddressKind


class SubmissionModeContractError(ValueError):
    """Raised when a managed-mode registry boundary is not exact."""


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ROWS: tuple[
    tuple[ManagedMode, RecordKind, int, str, str, AddressKind],
    ...,
] = (
    (
        "qualification",
        "submission-intent",
        2,
        "glm52_sky_submission_intent_v2",
        "intent_body_sha256",
        "content-addressed-body",
    ),
    (
        "qualification",
        "submission-accepted",
        2,
        "glm52_sky_submission_accepted_v2",
        "accepted_body_sha256",
        "content-addressed-body",
    ),
    (
        "qualification",
        "submission-acquisition",
        1,
        "glm52_sky_submission_acquired_v1",
        "acquisition_body_sha256",
        "descriptor-file-singleton",
    ),
    (
        "qualification",
        "controller-baseline",
        1,
        "glm52_controller_baseline_v1",
        "baseline_body_sha256",
        "content-addressed-body",
    ),
    (
        "qualification",
        "control-plane-ready",
        1,
        "glm52_must_start_control_plane_ready_v1",
        "control_plane_ready_body_sha256",
        "intent-control-body",
    ),
    (
        "qualification",
        "dynamic-job-binding",
        2,
        "glm52_sky_must_start_job_binding_v2",
        "job_binding_body_sha256",
        "intent-singleton",
    ),
    (
        "qualification",
        "worker-start-latch",
        2,
        "glm52_sky_worker_start_latch_v2",
        "worker_latch_body_sha256",
        "intent-instance-latch",
    ),
    (
        "qualification",
        "worker-start-accepted",
        2,
        "glm52_sky_worker_start_accepted_v2",
        "worker_acceptance_body_sha256",
        "intent-instance-latch",
    ),
    (
        "cache-seed",
        "submission-intent",
        1,
        "glm52_sky_cache_seed_submission_intent_v1",
        "intent_body_sha256",
        "content-addressed-body",
    ),
    (
        "cache-seed",
        "launch-claim",
        1,
        "glm52_sky_cache_seed_launch_claim_v1",
        "launch_claim_body_sha256",
        "descriptor-file-singleton",
    ),
    (
        "production",
        "submission-intent",
        1,
        "glm52_sky_production_submission_intent_v1",
        "intent_body_sha256",
        "content-addressed-body",
    ),
    (
        "production",
        "controller-baseline",
        1,
        "glm52_production_controller_baseline_v1",
        "baseline_body_sha256",
        "content-addressed-body",
    ),
    (
        "production",
        "control-plane-ready",
        1,
        "glm52_production_must_start_control_plane_ready_v1",
        "control_plane_ready_body_sha256",
        "intent-control-body",
    ),
    (
        "production",
        "submission-acquisition",
        1,
        "glm52_sky_production_submission_acquired_v1",
        "acquisition_body_sha256",
        "descriptor-file-singleton",
    ),
    (
        "production",
        "generation-claim",
        1,
        "glm52_sky_production_generation_claim_v1",
        "generation_claim_body_sha256",
        "generation-claim-singleton",
    ),
    (
        "production",
        "generation-start-decision",
        1,
        "glm52_sky_production_generation_start_decision_v1",
        "start_decision_body_sha256",
        "generation-start-decision-singleton",
    ),
    (
        "production",
        "generation-terminal",
        1,
        "glm52_sky_production_generation_terminal_v1",
        "generation_terminal_body_sha256",
        "generation-terminal-singleton",
    ),
)


def _exact_string(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise SubmissionModeContractError(f"{field} must be an exact string")
    return value


def record_contract(
    *,
    managed_mode: ManagedMode,
    record_kind: RecordKind,
) -> RecordContract:
    """Return the exact registered contract or fail closed."""

    mode = _exact_string(managed_mode, field="managed_mode")
    kind = _exact_string(record_kind, field="record_kind")
    for (
        registered_mode,
        registered_kind,
        schema_version,
        record_type,
        digest_field,
        address_kind,
    ) in _ROWS:
        if mode == registered_mode and kind == registered_kind:
            return RecordContract(
                schema_version=schema_version,
                record_type=record_type,
                digest_field=digest_field,
                address_kind=address_kind,
            )
    raise SubmissionModeContractError(
        f"unsupported managed submission record: {mode}/{kind}"
    )


def expected_sky_job_name(
    *,
    run_id: str,
    managed_mode: ManagedMode,
) -> str:
    """Return the exact Sky job name for one closed managed mode."""

    exact_run_id = _exact_string(run_id, field="run_id")
    mode = _exact_string(managed_mode, field="managed_mode")
    if _RUN_ID.fullmatch(exact_run_id) is None:
        raise SubmissionModeContractError("run_id is invalid")
    if mode == "production":
        return exact_run_id
    if mode == "qualification":
        return f"{exact_run_id}-qualification"
    if mode == "cache-seed":
        return f"{exact_run_id}-cache-seed"
    raise SubmissionModeContractError("managed_mode is unsupported")


def require_opaque_version_id(
    value: object,
    *,
    field: str,
) -> str:
    """Require an opaque, nonempty printable-ASCII S3 VersionId token."""

    label = _exact_string(field, field="field")
    if type(value) is not str:
        raise SubmissionModeContractError(f"{label} must be an exact string")
    if (
        not value
        or value == "null"
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise SubmissionModeContractError(
            f"{label} must be a nonempty printable-ASCII opaque VersionId"
        )
    return value


__all__ = [
    "AddressKind",
    "ManagedMode",
    "RecordContract",
    "RecordKind",
    "SubmissionModeContractError",
    "expected_sky_job_name",
    "record_contract",
    "require_opaque_version_id",
]
