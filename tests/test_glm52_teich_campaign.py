"""Campaign ledger, preflight gate, and deadline action tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlx_vq.quality.glm52_teich_campaign import (
    CampaignLedger,
    CampaignPhase,
    CampaignTransitionError,
    deadline_action,
    training_preflight_decision,
)


def _sha(value: str) -> str:
    return value * 64


def test_append_only_campaign_ledger_restarts_from_authenticated_artifacts(tmp_path) -> None:
    ledger = CampaignLedger(
        tmp_path / "campaign-ledger.jsonl",
        run_id="campaign-20260717",
        reservation_deadline="2026-07-19T11:30:00Z",
    )
    bootstrap = ledger.transition(
        CampaignPhase.BOOTSTRAP,
        input_identities={"descriptor": _sha("1")},
        output_identities={"environment": _sha("2")},
    )
    cuda = ledger.transition(
        CampaignPhase.CUDA_GATE,
        input_identities={"environment": _sha("2")},
        output_identities={"cuda_gate": _sha("3")},
    )

    restarted = CampaignLedger(
        tmp_path / "campaign-ledger.jsonl",
        run_id="campaign-20260717",
        reservation_deadline="2026-07-19T11:30:00Z",
    )
    assert restarted.current_phase == CampaignPhase.CUDA_GATE
    assert restarted.records[-1].prior_record_sha256 == bootstrap.record_sha256
    assert restarted.records[-1].record_sha256 == cuda.record_sha256
    with pytest.raises(CampaignTransitionError, match="next phase"):
        restarted.transition(
            CampaignPhase.TRAINING,
            input_identities={},
            output_identities={},
        )


def test_ledger_rejects_tampering_or_foreign_run(tmp_path) -> None:
    path = tmp_path / "campaign-ledger.jsonl"
    CampaignLedger(
        path,
        run_id="campaign-a",
        reservation_deadline="2026-07-19T11:30:00Z",
    ).transition(CampaignPhase.BOOTSTRAP, input_identities={}, output_identities={})
    path.write_bytes(path.read_bytes().replace(b"BOOTSTRAP", b"BOOTSTRAX"))
    with pytest.raises(CampaignTransitionError, match="SHA-256"):
        CampaignLedger(
            path,
            run_id="campaign-a",
            reservation_deadline="2026-07-19T11:30:00Z",
        )


def test_training_preflight_requires_memory_and_safety_factored_runtime() -> None:
    now = datetime(2026, 7, 18, 0, 0, tzinfo=timezone.utc)
    end = now + timedelta(hours=12)
    accepted = training_preflight_decision(
        peak_gpu_gib=69.9,
        sample_steps=20,
        sample_seconds=100,
        remaining_steps=5_000,
        now=now,
        capacity_block_end=end,
    )
    assert accepted.start_training
    assert accepted.safety_factor == 1.5

    memory = training_preflight_decision(
        peak_gpu_gib=70.0,
        sample_steps=20,
        sample_seconds=100,
        remaining_steps=5_000,
        now=now,
        capacity_block_end=end,
    )
    assert not memory.start_training and memory.reason == "peak_gpu_memory_gate"
    timing = training_preflight_decision(
        peak_gpu_gib=60.0,
        sample_steps=20,
        sample_seconds=600,
        remaining_steps=5_000,
        now=now,
        capacity_block_end=now + timedelta(hours=3),
    )
    assert not timing.start_training and timing.reason == "runtime_deadline_gate"


def test_deadline_actions_match_capacity_block_drain_protocol() -> None:
    end = datetime(2026, 7, 19, 11, 30, tzinfo=timezone.utc)
    assert deadline_action(now=end - timedelta(minutes=61), capacity_block_end=end) == "RUN"
    assert deadline_action(now=end - timedelta(minutes=60), capacity_block_end=end) == "STOP_ASSIGNING"
    assert deadline_action(now=end - timedelta(minutes=50), capacity_block_end=end) == "SIGTERM_WORKERS"
    assert deadline_action(now=end - timedelta(minutes=35), capacity_block_end=end) == "FINAL_SYNC"
    assert deadline_action(now=end - timedelta(minutes=32), capacity_block_end=end) == "TERMINATE_IF_DRAINED"
    assert deadline_action(now=end - timedelta(minutes=30), capacity_block_end=end) == "AWS_FORCED_WINDOW"
