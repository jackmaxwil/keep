"""Strict SkyPilot campaign, approval, account, and spend-ledger tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_ACCOUNT_ID,
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_HOURS,
    APPROVED_HOURLY_COST_USD,
    GPU_APPROVAL_REQUEST,
    GpuSpendLedger,
    SkyCampaignLedger,
    SkyCampaignValidationError,
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
    execution_window_action,
    require_approved_aws_identity,
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_teich_campaign import (
    CampaignPhase,
    training_preflight_decision,
)


UTC = timezone.utc
INGESTED_AT = datetime(2026, 7, 23, 18, 5, tzinfo=UTC)
MUST_START_BY = datetime(2026, 7, 24, 6, 5, tzinfo=UTC)


def _sha(character: str) -> str:
    return character * 64


def _approval() -> dict[str, object]:
    return build_gpu_spend_approval(
        ingested_at=INGESTED_AT,
        slack_permalink=None,
    )


def _descriptor(
    approval: dict[str, object],
    *,
    qualification_cache_prefix: str | None = None,
    qualification_cache_manifest_sha256: str | None = None,
) -> dict[str, object]:
    cache_sha = qualification_cache_manifest_sha256 or _sha("9")
    cache_prefix = qualification_cache_prefix or (
        "qualification-cache/seeds/glm52-sky-20260723/"
        f"{cache_sha}/"
    )
    return build_sky_campaign_descriptor(
        run_id="glm52-sky-20260723",
        must_start_by=MUST_START_BY,
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-us-west-2-246813579024",
        jobs_bucket="keep-glm52-us-west-2-246813579024",
        repo_tar_key="campaigns/glm52-sky-20260723/repository/repo.tar.gz",
        repo_tar_sha256=_sha("1"),
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260723/submissions/first/campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260723/authorities/GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=str(approval["approval_body_sha256"]),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": _sha("2"),
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": _sha("3"),
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": _sha("4"),
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": _sha("5"),
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": _sha("6"),
            "training_config_key": (
                "campaigns/glm52-sky-20260723/authorities/training.json"
            ),
            "training_config_sha256": _sha("7"),
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260723/inventories/"
                f"artifact-inventory-{_sha('8')}.json"
            ),
            "artifact_inventory_sha256": _sha("8"),
            "qualification_cache_prefix": cache_prefix,
            "qualification_cache_manifest_sha256": cache_sha,
        },
    )


def test_gpu_approval_binds_kon_response_to_exact_cost_envelope() -> None:
    approval = validate_gpu_spend_approval(_approval())

    assert approval["approved_by"] == "Alex Approver"
    assert approval["approval_request"] == GPU_APPROVAL_REQUEST
    assert approval["approval_response"] == "yep"
    assert approval["approval_displayed_time"] == "5:59 PM"
    assert approval["approved_hourly_usd"] == APPROVED_HOURLY_COST_USD
    assert approval["approved_gpu_hours"] == APPROVED_GPU_HOURS
    assert approval["approved_gpu_cost_usd"] == APPROVED_GPU_COST_USD
    assert approval["includes_qualification"] is True
    assert approval["includes_recovery_instances"] is True


def test_gpu_approval_rejects_cost_or_unknown_field_drift() -> None:
    approval = _approval()
    with pytest.raises(SkyCampaignValidationError, match="body SHA-256"):
        validate_gpu_spend_approval({**approval, "approved_gpu_hours": 48})

    with pytest.raises(SkyCampaignValidationError, match="schema mismatch"):
        validate_gpu_spend_approval({**approval, "capacity_block": True})


def test_sky_descriptor_is_strict_and_capacity_block_fields_are_unrepresentable() -> None:
    approval = _approval()
    descriptor = validate_sky_campaign_descriptor(_descriptor(approval))

    assert descriptor["record_type"] == "glm52_sky_campaign_descriptor_v2"
    assert descriptor["account_id"] == APPROVED_ACCOUNT_ID
    assert descriptor["provider"] == "aws"
    assert descriptor["region"] == "us-west-2"
    assert descriptor["instance_type"] == "p5.48xlarge"
    assert descriptor["instance_count"] == 1
    assert descriptor["use_spot"] is False
    assert descriptor["approved_gpu_runtime_seconds"] == 86_400
    assert len(str(descriptor["campaign_identity_sha256"])) == 64
    assert "capacity_reservation_id" not in descriptor

    with pytest.raises(SkyCampaignValidationError, match="schema mismatch"):
        validate_sky_campaign_descriptor(
            {**descriptor, "capacity_reservation_id": "cr-123"}
        )


def test_qualification_cache_prefix_is_bound_to_manifest_identity() -> None:
    approval = _approval()
    placeholder = _descriptor(
        approval,
        qualification_cache_prefix="qualification-cache/",
        qualification_cache_manifest_sha256="0" * 64,
    )
    assert (
        placeholder["artifacts"]["qualification_cache_manifest_sha256"]
        == "0" * 64
    )

    with pytest.raises(
        SkyCampaignValidationError,
        match="qualification cache prefix",
    ):
        _descriptor(
            approval,
            qualification_cache_prefix="qualification-cache/",
            qualification_cache_manifest_sha256="9" * 64,
        )


def test_campaign_identity_survives_a_new_submission_window() -> None:
    approval = _approval()
    first = _descriptor(approval)
    second = build_sky_campaign_descriptor(
        run_id=str(first["run_id"]),
        must_start_by=MUST_START_BY + timedelta(hours=12),
        controller_identity=str(first["controller_identity"]),
        worker_identity=str(first["worker_identity"]),
        vpc_name=str(first["vpc_name"]),
        image_id=str(first["image_id"]),
        bucket=str(first["bucket"]),
        jobs_bucket=str(first["jobs_bucket"]),
        repo_tar_key=str(first["repo_tar_key"]),
        repo_tar_sha256=str(first["repo_tar_sha256"]),
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260723/submissions/second.json"
        ),
        approval_key=str(first["approval_key"]),
        approval_sha256=str(first["approval_sha256"]),
        artifacts=first["artifacts"],  # type: ignore[arg-type]
    )
    assert second["campaign_identity_sha256"] == first["campaign_identity_sha256"]
    assert second["descriptor_body_sha256"] != first["descriptor_body_sha256"]

    tampered = {**first, "campaign_identity_sha256": _sha("f")}
    tampered.pop("descriptor_body_sha256")
    with pytest.raises(SkyCampaignValidationError, match="campaign identity"):
        validate_sky_campaign_descriptor(tampered, verify_body_sha=False)


def test_sky_descriptor_rejects_wrong_account_spot_region_or_price() -> None:
    descriptor = _descriptor(_approval())
    changes = (
        ({"account_id": "135792468013"}, "approved AWS account"),
        ({"use_spot": True}, "on-demand"),
        ({"region": "us-east-1"}, "us-west-2"),
        ({"max_hourly_cost_usd": 60.0}, "hourly"),
    )
    for update, message in changes:
        changed = {**descriptor, **update}
        changed.pop("descriptor_body_sha256")
        with pytest.raises(SkyCampaignValidationError, match=message):
            validate_sky_campaign_descriptor(changed, verify_body_sha=False)


def test_account_guard_refuses_default_wrong_account() -> None:
    accepted = require_approved_aws_identity(
        {
            "Account": APPROVED_ACCOUNT_ID,
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/"
                "AWSReservedSSO_AdministratorAccess/operator@example.com"
            ),
        }
    )
    assert accepted["Account"] == APPROVED_ACCOUNT_ID

    with pytest.raises(SkyCampaignValidationError, match="refusing AWS operation"):
        require_approved_aws_identity(
            {
                "Account": "135792468013",
                "Arn": "arn:aws:sts::135792468013:assumed-role/wrong/user",
            }
        )


def test_spend_ledger_is_cumulative_across_replacement_instances(tmp_path) -> None:
    ledger = GpuSpendLedger(
        tmp_path / "GPU_SPEND_LEDGER.jsonl",
        run_id="glm52-sky-20260723",
        approval_sha256=_sha("a"),
        approved_gpu_runtime_seconds=86_400,
        approved_gpu_cost_usd=1_320.96,
        hourly_cost_usd=55.04,
    )
    first_start = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    ledger.start_allocation(
        job_id="sky-job-1",
        instance_id="i-first",
        launched_at=first_start,
    )
    ledger.end_allocation(
        instance_id="i-first",
        ended_at=first_start + timedelta(hours=2),
    )
    second_start = first_start + timedelta(hours=3)
    ledger.start_allocation(
        job_id="sky-job-1",
        instance_id="i-replacement",
        launched_at=second_start,
    )

    assert ledger.consumed_gpu_seconds(now=second_start + timedelta(hours=1)) == 10_800
    assert ledger.remaining_gpu_seconds(now=second_start + timedelta(hours=1)) == 75_600
    assert ledger.estimated_gpu_cost_usd(now=second_start + timedelta(hours=1)) == 165.12

    restarted = GpuSpendLedger(
        ledger.path,
        run_id="glm52-sky-20260723",
        approval_sha256=_sha("a"),
        approved_gpu_runtime_seconds=86_400,
        approved_gpu_cost_usd=1_320.96,
        hourly_cost_usd=55.04,
    )
    assert restarted.remaining_gpu_seconds(
        now=second_start + timedelta(hours=1)
    ) == 75_600


def test_spend_ledger_rejects_parallel_allocations_and_tampering(tmp_path) -> None:
    path = tmp_path / "GPU_SPEND_LEDGER.jsonl"
    ledger = GpuSpendLedger(
        path,
        run_id="glm52-sky-20260723",
        approval_sha256=_sha("a"),
        approved_gpu_runtime_seconds=86_400,
        approved_gpu_cost_usd=1_320.96,
        hourly_cost_usd=55.04,
    )
    launched_at = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    ledger.start_allocation(
        job_id="sky-job-1",
        instance_id="i-first",
        launched_at=launched_at,
    )
    with pytest.raises(SkyCampaignValidationError, match="active allocation"):
        ledger.start_allocation(
            job_id="sky-job-1",
            instance_id="i-parallel",
            launched_at=launched_at,
        )

    path.write_bytes(path.read_bytes().replace(b"i-first", b"i-other"))
    with pytest.raises(SkyCampaignValidationError, match="SHA-256"):
        GpuSpendLedger(
            path,
            run_id="glm52-sky-20260723",
            approval_sha256=_sha("a"),
            approved_gpu_runtime_seconds=86_400,
            approved_gpu_cost_usd=1_320.96,
            hourly_cost_usd=55.04,
        )


def test_execution_window_matches_graceful_budget_drain_offsets() -> None:
    deadline = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)

    assert execution_window_action(
        now=deadline - timedelta(minutes=61), execution_deadline=deadline
    ) == "RUN"
    assert execution_window_action(
        now=deadline - timedelta(minutes=60), execution_deadline=deadline
    ) == "STOP_ASSIGNING"
    assert execution_window_action(
        now=deadline - timedelta(minutes=50), execution_deadline=deadline
    ) == "SIGTERM_WORKERS"
    assert execution_window_action(
        now=deadline - timedelta(minutes=35), execution_deadline=deadline
    ) == "FINAL_SYNC"
    assert execution_window_action(
        now=deadline - timedelta(minutes=30), execution_deadline=deadline
    ) == "REQUIRE_DRAINED"
    assert execution_window_action(
        now=deadline, execution_deadline=deadline
    ) == "CANCEL"


def test_sky_phase_ledger_binds_execution_deadline_and_gpu_allocation(tmp_path) -> None:
    path = tmp_path / "campaign-ledger-v2.jsonl"
    ledger = SkyCampaignLedger(
        path,
        run_id="glm52-sky-20260723",
        execution_deadline="2026-07-25T08:00:00Z",
        gpu_spend_authority_sha256=_sha("a"),
    )
    bootstrap = ledger.transition(
        CampaignPhase.BOOTSTRAP,
        input_identities={"descriptor": _sha("1")},
        output_identities={"environment": _sha("2")},
        gpu_allocation_sha256=_sha("b"),
        timestamp=datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
    )
    ledger.transition(
        CampaignPhase.CUDA_GATE,
        input_identities={"environment": _sha("2")},
        output_identities={"cuda_gate": _sha("3")},
        gpu_allocation_sha256=_sha("c"),
        timestamp=datetime(2026, 7, 24, 8, 1, tzinfo=UTC),
    )

    restarted = SkyCampaignLedger(
        path,
        run_id="glm52-sky-20260723",
        execution_deadline="2026-07-25T08:00:00Z",
        gpu_spend_authority_sha256=_sha("a"),
    )
    assert restarted.current_phase == CampaignPhase.CUDA_GATE
    assert restarted.records[-1].prior_record_sha256 == bootstrap.record_sha256
    assert restarted.records[-1].execution_deadline == "2026-07-25T08:00:00Z"
    assert restarted.records[0].gpu_allocation_sha256 == _sha("b")
    assert restarted.records[-1].gpu_allocation_sha256 == _sha("c")

    with pytest.raises(SkyCampaignValidationError, match="foreign"):
        SkyCampaignLedger(
            path,
            run_id="glm52-sky-20260723",
            execution_deadline="2026-07-25T08:00:00Z",
            gpu_spend_authority_sha256=_sha("d"),
        )


def test_sky_phase_ledger_accepts_new_deadline_only_for_replacement_allocation(
    tmp_path,
) -> None:
    path = tmp_path / "campaign-ledger-v2.jsonl"
    first = SkyCampaignLedger(
        path,
        run_id="glm52-sky-20260723",
        execution_deadline="2026-07-25T08:00:00Z",
        gpu_spend_authority_sha256=_sha("a"),
    )
    first.transition(
        CampaignPhase.BOOTSTRAP,
        input_identities={"descriptor": _sha("1")},
        output_identities={"environment": _sha("2")},
        gpu_allocation_sha256=_sha("b"),
        timestamp=datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
    )
    replacement = SkyCampaignLedger(
        path,
        run_id="glm52-sky-20260723",
        execution_deadline="2026-07-25T09:00:00Z",
        gpu_spend_authority_sha256=_sha("a"),
    )
    replacement.transition(
        CampaignPhase.CUDA_GATE,
        input_identities={"environment": _sha("2")},
        output_identities={"cuda_gate": _sha("3")},
        gpu_allocation_sha256=_sha("c"),
        timestamp=datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
    )
    resumed = SkyCampaignLedger(
        path,
        run_id="glm52-sky-20260723",
        execution_deadline="2026-07-25T09:00:00Z",
        gpu_spend_authority_sha256=_sha("a"),
    )
    assert [record.execution_deadline for record in resumed.records] == [
        "2026-07-25T08:00:00Z",
        "2026-07-25T09:00:00Z",
    ]
    assert [record.gpu_allocation_sha256 for record in resumed.records] == [
        _sha("b"),
        _sha("c"),
    ]


def test_sky_phase_ledger_can_terminally_drain_from_an_incomplete_phase(
    tmp_path,
) -> None:
    ledger = SkyCampaignLedger(
        tmp_path / "campaign-ledger.jsonl",
        run_id="glm52-sky-early-drain",
        execution_deadline="2026-07-25T08:00:00Z",
        gpu_spend_authority_sha256=_sha("a"),
    )
    for phase in (CampaignPhase.BOOTSTRAP, CampaignPhase.CUDA_GATE):
        ledger.transition(
            phase,
            input_identities={"input": _sha("1")},
            output_identities={"output": _sha("2")},
            gpu_allocation_sha256=_sha("b"),
            timestamp=datetime(2026, 7, 24, 8, len(ledger.records), tzinfo=UTC),
        )
    ledger.transition(
        CampaignPhase.DRAINED,
        input_identities={"prestop": ledger.records[-1].record_sha256},
        output_identities={"drained": _sha("3")},
        gpu_allocation_sha256=_sha("b"),
        timestamp=datetime(2026, 7, 24, 9, 0, tzinfo=UTC),
    )
    assert ledger.current_phase is CampaignPhase.DRAINED
    with pytest.raises(SkyCampaignValidationError, match="next phase"):
        ledger.transition(
            CampaignPhase.TEACHER,
            input_identities={"input": _sha("1")},
            output_identities={"output": _sha("2")},
            gpu_allocation_sha256=_sha("b"),
        )


def test_training_preflight_accepts_generic_execution_deadline() -> None:
    now = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    decision = training_preflight_decision(
        peak_gpu_gib=60.0,
        sample_steps=20,
        sample_seconds=100.0,
        remaining_steps=1_000,
        now=now,
        execution_deadline=now + timedelta(hours=4),
    )
    assert decision.start_training is True

    with pytest.raises(ValueError, match="exactly one"):
        training_preflight_decision(
            peak_gpu_gib=60.0,
            sample_steps=20,
            sample_seconds=100.0,
            remaining_steps=1_000,
            now=now,
            capacity_block_end=now + timedelta(hours=4),
            execution_deadline=now + timedelta(hours=4),
        )
