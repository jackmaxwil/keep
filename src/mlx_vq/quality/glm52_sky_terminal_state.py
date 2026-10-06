"""Authenticate a drained SkyPilot campaign before worker teardown."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_RUNTIME_SECONDS,
    APPROVED_HOURLY_COST_USD,
    GpuSpendLedger,
    SkyCampaignLedger,
)
from mlx_vq.quality.glm52_teich_campaign import CampaignPhase


_DRAINED_FIELDS = {
    "record_type",
    "run_id",
    "outcome",
    "last_completed_phase",
    "drain_reason",
    "prior_record_sha256",
    "teacher_cache_ready_sha256",
    "automatic_model_promotion",
    "model_uploaded",
    "execution_deadline",
    "gpu_allocation_sha256",
}


def validate_sky_terminal_state(
    *,
    run_id: str,
    execution_deadline: str,
    gpu_spend_authority_sha256: str,
    drained_raw: bytes,
    campaign_ledger_raw: bytes,
    spend_ledger_raw: bytes,
) -> dict[str, object]:
    """Require a closed spend allocation and ledger-bound drained marker."""

    try:
        drained = json.loads(drained_raw)
    except json.JSONDecodeError as error:
        raise ValueError("drained marker is invalid JSON") from error
    if not isinstance(drained, dict) or set(drained) != _DRAINED_FIELDS:
        raise ValueError("Sky drained marker schema mismatch")
    if (
        drained.get("record_type") != "glm52_sky_campaign_drained_v2"
        or drained.get("run_id") != run_id
        or drained.get("execution_deadline") != execution_deadline
        or drained.get("automatic_model_promotion") is not False
        or drained.get("model_uploaded") is not False
    ):
        raise ValueError("Sky drained marker authority mismatch")
    outcome = drained.get("outcome")
    last_phase = drained.get("last_completed_phase")
    drain_reason = drained.get("drain_reason")
    ready_sha = drained.get("teacher_cache_ready_sha256")
    allowed_phases = {phase.value for phase in CampaignPhase if phase is not CampaignPhase.DRAINED}
    if last_phase not in allowed_phases:
        raise ValueError("Sky drained marker last completed phase is invalid")
    if outcome == "completed":
        if (
            last_phase != CampaignPhase.EVALUATION.value
            or drain_reason != "campaign_complete"
            or not isinstance(ready_sha, str)
            or len(ready_sha) != 64
        ):
            raise ValueError("Sky completed drain authority mismatch")
    elif outcome == "training_deferred":
        if (
            last_phase != CampaignPhase.EVALUATION.value
            or drain_reason != "training_deferred"
            or not isinstance(ready_sha, str)
            or len(ready_sha) != 64
        ):
            raise ValueError("Sky deferred drain authority mismatch")
    elif outcome == "resumable_deadline":
        if drain_reason not in {
            "execution_window_prestop",
            "operator_graceful_stop",
        } or (
            ready_sha is not None
            and (not isinstance(ready_sha, str) or len(ready_sha) != 64)
        ):
            raise ValueError("Sky resumable drain authority mismatch")
    else:
        raise ValueError("Sky drained marker outcome is invalid")

    try:
        first_spend = json.loads(spend_ledger_raw.splitlines()[0])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("GPU spend ledger is empty or malformed") from error
    if not isinstance(first_spend, dict):
        raise ValueError("GPU spend ledger first record is malformed")
    approval_sha256 = first_spend.get("approval_sha256")
    if not isinstance(approval_sha256, str):
        raise ValueError("GPU spend ledger approval authority is missing")

    with tempfile.TemporaryDirectory(prefix="glm52-terminal-") as directory:
        root = Path(directory)
        spend_path = root / "GPU_SPEND_LEDGER.jsonl"
        campaign_path = root / "campaign-ledger.jsonl"
        spend_path.write_bytes(spend_ledger_raw)
        campaign_path.write_bytes(campaign_ledger_raw)
        spend = GpuSpendLedger(
            spend_path,
            run_id=run_id,
            approval_sha256=approval_sha256,
            approved_gpu_runtime_seconds=APPROVED_GPU_RUNTIME_SECONDS,
            approved_gpu_cost_usd=APPROVED_GPU_COST_USD,
            hourly_cost_usd=APPROVED_HOURLY_COST_USD,
        )
        if spend.genesis_sha256 != gpu_spend_authority_sha256:
            raise ValueError("GPU spend ledger authority mismatch")
        if not spend.records:
            raise ValueError("GPU spend ledger is empty")
        last_spend = spend.records[-1]
        if last_spend.get("event") != "allocation_ended":
            raise ValueError("GPU spend allocation remains open")

        campaign = SkyCampaignLedger(
            campaign_path,
            run_id=run_id,
            execution_deadline=execution_deadline,
            gpu_spend_authority_sha256=gpu_spend_authority_sha256,
        )
        if campaign.current_phase is not CampaignPhase.DRAINED:
            raise ValueError("campaign ledger is not in DRAINED phase")
        record = campaign.records[-1]
        drained_sha256 = hashlib.sha256(drained_raw).hexdigest()
        if record.output_identities.get("drained") != drained_sha256:
            raise ValueError("drained marker SHA-256 is not bound by campaign ledger")
        if drained.get("prior_record_sha256") != record.prior_record_sha256:
            raise ValueError("drained marker prior-record authority mismatch")
        if drained.get("gpu_allocation_sha256") != record.gpu_allocation_sha256:
            raise ValueError("drained marker GPU allocation authority mismatch")
    return {
        "record_type": "glm52_sky_terminal_verification_v1",
        "run_id": run_id,
        "phase": "DRAINED",
        "outcome": outcome,
        "last_completed_phase": last_phase,
        "terminal_state_authenticated": True,
        "drained_sha256": hashlib.sha256(drained_raw).hexdigest(),
        "campaign_ledger_sha256": hashlib.sha256(campaign_ledger_raw).hexdigest(),
        "gpu_spend_ledger_sha256": hashlib.sha256(spend_ledger_raw).hexdigest(),
    }
