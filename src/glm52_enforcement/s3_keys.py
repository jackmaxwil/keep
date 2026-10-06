"""Exact import-light S3 coordinates for GLM-5.2 immutable JSON records."""

from __future__ import annotations

import re


_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ACTIVATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _segment(name: str, value: object, *, activation: bool = False) -> str:
    pattern = _ACTIVATION if activation else _SEGMENT
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is not an exact ASCII-safe path segment")
    if value in {".", ".."}:
        raise ValueError(f"{name} is not an exact ASCII-safe path segment")
    return value


def _sha(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be exact lowercase SHA-256")
    return value


def _ordinal(name: str, value: object, *, positive: bool = True) -> str:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum or value > 99999999:
        raise ValueError(f"{name} must be an exact eight-digit integer")
    return f"{value:08d}"


def _production_generation_prefix(*, run_id: str, generation: int) -> str:
    run = _segment("run_id", run_id)
    text = _ordinal("generation", generation)
    return (
        f"campaigns/{run}/submissions/production/generations/{text}"
    )


def gpu_spend_approval_s3_key(
    *, run_id: str, approval_file_sha256: str
) -> str:
    run = _segment("run_id", run_id)
    digest = _sha("approval_file_sha256", approval_file_sha256)
    return f"campaigns/{run}/authorities/GPU_SPEND_APPROVAL-{digest}.json"


def gpu_spend_snapshot_s3_key(
    *, run_id: str, snapshot_body_sha256: str
) -> str:
    run = _segment("run_id", run_id)
    digest = _sha("snapshot_body_sha256", snapshot_body_sha256)
    return f"campaigns/{run}/spend-snapshots/{digest}/GPU_SPEND_SNAPSHOT.json"


def production_submission_intent_s3_key(
    *, run_id: str, intent_body_sha256: str
) -> str:
    run = _segment("run_id", run_id)
    digest = _sha("intent_body_sha256", intent_body_sha256)
    return (
        f"campaigns/{run}/submissions/production/intents/{digest}/"
        "SKYPILOT_SUBMISSION_INTENT.json"
    )


def production_controller_baseline_s3_key(
    *, run_id: str, baseline_body_sha256: str
) -> str:
    run = _segment("run_id", run_id)
    digest = _sha("baseline_body_sha256", baseline_body_sha256)
    return (
        f"campaigns/{run}/production/controller-baselines/{digest}/"
        "CONTROLLER_BASELINE.json"
    )


def production_must_start_control_plane_ready_s3_key(
    *,
    run_id: str,
    intent_body_sha256: str,
    control_plane_ready_body_sha256: str,
) -> str:
    run = _segment("run_id", run_id)
    intent = _sha("intent_body_sha256", intent_body_sha256)
    ready = _sha(
        "control_plane_ready_body_sha256",
        control_plane_ready_body_sha256,
    )
    return (
        f"campaigns/{run}/monitor/must-start/production/{intent}/"
        f"control-plane-ready/{ready}/CONTROL_PLANE_READY.json"
    )


def production_submission_acquired_s3_key(
    *, run_id: str, descriptor_file_sha256: str
) -> str:
    run = _segment("run_id", run_id)
    digest = _sha("descriptor_file_sha256", descriptor_file_sha256)
    return (
        f"campaigns/{run}/submissions/production/acquisitions/{digest}/"
        "SUBMISSION_ACQUIRED.json"
    )


def fence_genesis_s3_key(*, run_id: str) -> str:
    run = _segment("run_id", run_id)
    return f"campaigns/{run}/authorities/fence/FENCE_GENESIS.json"


def fence_successor_s3_key(
    *, run_id: str, predecessor_body_sha256: str
) -> str:
    run = _segment("run_id", run_id)
    digest = _sha("predecessor_body_sha256", predecessor_body_sha256)
    return (
        f"campaigns/{run}/authorities/fence/successors/{digest}/"
        "FENCE_SUCCESSOR.json"
    )


def production_generation_claim_s3_key(
    *, run_id: str, generation: int
) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/GENERATION_CLAIM.json"
    )


def production_generation_start_decision_s3_key(
    *, run_id: str, generation: int
) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/START_DECISION.json"
    )


def production_generation_terminal_s3_key(
    *, run_id: str, generation: int
) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/GENERATION_TERMINAL.json"
    )


def production_terminal_v2_s3_key(
    *, run_id: str, generation: int
) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/terminal/PRODUCTION_TERMINAL_V2.json"
    )


def sky_post_handoff_s3_key(*, run_id: str, generation: int) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/handoff/SKY_POST_HANDOFF.json"
    )


def sky_request_correlated_s3_key(
    *, run_id: str, generation: int
) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/requests/SKY_REQUEST_CORRELATED.json"
    )


def sky_job_bound_s3_key(*, run_id: str, generation: int) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/bindings/SKY_JOB_BOUND.json"
    )


def bootstrap_ready_s3_key(
    *, run_id: str, generation: int, allocation_ordinal: int
) -> str:
    allocation = _ordinal("allocation_ordinal", allocation_ordinal)
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + f"/allocations/{allocation}/BOOTSTRAP_READY.json"
    )


def worker_graceful_stop_s3_key(
    *, run_id: str, generation: int, allocation_ordinal: int
) -> str:
    allocation = _ordinal("allocation_ordinal", allocation_ordinal)
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + f"/allocations/{allocation}/WORKER_GRACEFUL_STOP.json"
    )


def campaign_drained_s3_key(*, run_id: str, generation: int) -> str:
    return (
        _production_generation_prefix(run_id=run_id, generation=generation)
        + "/terminal/CAMPAIGN_DRAINED.json"
    )


def _activation_finalization_prefix(
    *, run_id: str, activation_id: str
) -> str:
    run = _segment("run_id", run_id)
    activation = _segment(
        "activation_id", activation_id, activation=True
    )
    return (
        f"campaigns/{run}/submissions/production/activations/"
        f"{activation}/finalization"
    )


def support_plane_finalized_s3_key(
    *, run_id: str, activation_id: str
) -> str:
    return (
        _activation_finalization_prefix(
            run_id=run_id, activation_id=activation_id
        )
        + "/SUPPORT_PLANE_FINALIZED.json"
    )


def h1g_drained_s3_key(*, run_id: str, activation_id: str) -> str:
    return (
        _activation_finalization_prefix(
            run_id=run_id, activation_id=activation_id
        )
        + "/H1G_DRAINED.json"
    )


def snapshot_cleanup_authority_audit_s3_key(
    *,
    run_id: str,
    activation_ordinal: int,
    delete_logical_attempt: int,
) -> str:
    run = _segment("run_id", run_id)
    activation = _ordinal("activation_ordinal", activation_ordinal)
    attempt = _ordinal(
        "delete_logical_attempt", delete_logical_attempt
    )
    return (
        f"campaigns/{run}/h1g/snapshot-cleanup/{activation}/"
        f"audits/{attempt}.json"
    )


def validate_frozen_key(
    *,
    key: str,
    run_id: str,
    allowed_families: tuple[str, ...],
) -> str:
    run = _segment("run_id", run_id)
    if type(key) is not str or not key:
        raise ValueError("key must be an exact nonempty string")
    if type(allowed_families) is not tuple or not allowed_families:
        raise ValueError("allowed_families must be a nonempty exact tuple")
    prefix = re.escape(f"campaigns/{run}")
    sha = r"[0-9a-f]{64}"
    segment = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}"
    eight = r"[0-9]{8}"
    patterns = {
        "gpu-spend-approval": rf"{prefix}/authorities/GPU_SPEND_APPROVAL-{sha}[.]json",
        "gpu-spend-snapshot": rf"{prefix}/spend-snapshots/{sha}/GPU_SPEND_SNAPSHOT[.]json",
        "production-submission-intent": rf"{prefix}/submissions/production/intents/{sha}/SKYPILOT_SUBMISSION_INTENT[.]json",
        "production-controller-baseline": rf"{prefix}/production/controller-baselines/{sha}/CONTROLLER_BASELINE[.]json",
        "production-control-plane-readiness": rf"{prefix}/monitor/must-start/production/{sha}/control-plane-ready/{sha}/CONTROL_PLANE_READY[.]json",
        "production-submission-acquisition": rf"{prefix}/submissions/production/acquisitions/{sha}/SUBMISSION_ACQUIRED[.]json",
        "fence-genesis": rf"{prefix}/authorities/fence/FENCE_GENESIS[.]json",
        "fence-successor": rf"{prefix}/authorities/fence/successors/{sha}/FENCE_SUCCESSOR[.]json",
        "generation-claim": rf"{prefix}/submissions/production/generations/{eight}/GENERATION_CLAIM[.]json",
        "start-decision": rf"{prefix}/submissions/production/generations/{eight}/START_DECISION[.]json",
        "generation-terminal": rf"{prefix}/submissions/production/generations/{eight}/GENERATION_TERMINAL[.]json",
        "production-terminal-v2": rf"{prefix}/submissions/production/generations/{eight}/terminal/PRODUCTION_TERMINAL_V2[.]json",
        "sky-post-handoff": rf"{prefix}/submissions/production/generations/{eight}/handoff/SKY_POST_HANDOFF[.]json",
        "sky-request-correlated": rf"{prefix}/submissions/production/generations/{eight}/requests/SKY_REQUEST_CORRELATED[.]json",
        "sky-job-bound": rf"{prefix}/submissions/production/generations/{eight}/bindings/SKY_JOB_BOUND[.]json",
        "bootstrap-ready": rf"{prefix}/submissions/production/generations/{eight}/allocations/{eight}/BOOTSTRAP_READY[.]json",
        "worker-graceful-stop": rf"{prefix}/submissions/production/generations/{eight}/allocations/{eight}/WORKER_GRACEFUL_STOP[.]json",
        "campaign-drained": rf"{prefix}/submissions/production/generations/{eight}/terminal/CAMPAIGN_DRAINED[.]json",
        "support-plane-finalized": rf"{prefix}/submissions/production/activations/{segment}/finalization/SUPPORT_PLANE_FINALIZED[.]json",
        "h1g-drained": rf"{prefix}/submissions/production/activations/{segment}/finalization/H1G_DRAINED[.]json",
        "snapshot-cleanup-authority-audit": rf"{prefix}/h1g/snapshot-cleanup/{eight}/audits/{eight}[.]json",
    }
    for family in allowed_families:
        if type(family) is not str or family not in patterns:
            raise ValueError("allowed family is unknown")
    if not any(re.fullmatch(patterns[family], key) for family in allowed_families):
        raise ValueError("key is not an exact allowed frozen coordinate")
    return key


__all__ = [
    "bootstrap_ready_s3_key",
    "campaign_drained_s3_key",
    "fence_genesis_s3_key",
    "fence_successor_s3_key",
    "gpu_spend_approval_s3_key",
    "gpu_spend_snapshot_s3_key",
    "h1g_drained_s3_key",
    "production_controller_baseline_s3_key",
    "production_generation_claim_s3_key",
    "production_generation_start_decision_s3_key",
    "production_generation_terminal_s3_key",
    "production_must_start_control_plane_ready_s3_key",
    "production_submission_acquired_s3_key",
    "production_submission_intent_s3_key",
    "production_terminal_v2_s3_key",
    "sky_job_bound_s3_key",
    "sky_post_handoff_s3_key",
    "sky_request_correlated_s3_key",
    "snapshot_cleanup_authority_audit_s3_key",
    "support_plane_finalized_s3_key",
    "validate_frozen_key",
    "worker_graceful_stop_s3_key",
]
