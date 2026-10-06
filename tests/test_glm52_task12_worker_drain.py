from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import hashlib

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.spend_authority import (
    AllocationInterval,
    GpuLiabilityReserveResult,
    LiabilitySettlementEvidence,
    canonical_decimal_json_bytes,
)
from glm52_enforcement.task10_worker import (
    GRACEFUL_SCRIPT_NAMES,
    GRACEFUL_STOP_DOCUMENT,
    UNIT_NAMES,
    build_graceful_stop_evidence,
    build_worker_bootstrap_descriptor,
    build_worker_instance_observation,
)


TS = "2026-07-29T12:00:00Z"
SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
INSTANCE_ID = "i-0123456789abcdef0"


def _allocation(
    *,
    instance_id: str = INSTANCE_ID,
) -> AllocationInterval:
    return AllocationInterval(
        instance_id=instance_id,
        job_id="1",
        started_at="2026-07-29T11:00:00Z",
        ended_at=None,
        charged_seconds=3600,
        charged_cost_usd=Decimal("55.04"),
        state="OPEN",
    )


def _reserve() -> GpuLiabilityReserveResult:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_liability_reserve_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-1",
        "generation": 1,
        "generation_text": "00000001",
        "allocation_ordinal": 1,
        "allocation_ordinal_text": "00000001",
        "ec2_client_token": SHA_A,
        "request_identity_sha256": SHA_A,
        "ledger_predecessor_identity_sha256": SHA_B,
        "gpu_reserve_seconds": 900,
        "gpu_reserve_cost_usd": Decimal("13.76"),
        "root_volume_gib": 300,
        "root_volume_tail_usd_max": Decimal("0.01"),
        "residual_liability_approval_identity_sha256": SHA_A,
        "epoch": 1,
        "revision": 7,
        "nonce_owner_identity_sha256": SHA_B,
        "state": "HELD",
        "observed_at": TS,
        "reserve_key": (
            "GPU_LIABILITY_RESERVE#activation-1#00000001#00000001"
        ),
    }
    identity = hashlib.sha256(
        canonical_decimal_json_bytes(body)
    ).hexdigest()
    record = {**body, "canonical_body_sha256": identity}
    return GpuLiabilityReserveResult(
        disposition="CREATED",
        reserve_key=str(body["reserve_key"]),
        reserve_identity_sha256=identity,
        gpu_reserve_seconds=900,
        gpu_reserve_cost_usd=Decimal("13.76"),
        root_volume_gib=300,
        root_volume_tail_usd_max=Decimal("0.01"),
        remaining_gpu_seconds=85_500,
        remaining_gpu_cost_usd=Decimal("1307.20"),
        record=record,
    )


def _runtime(
    *,
    allocation: AllocationInterval | None = None,
) -> tuple[object, object]:
    allocation = _allocation() if allocation is None else allocation
    allocation_identity = canonical_sha256(
        {
            **asdict(allocation),
            "charged_cost_usd": format(
                allocation.charged_cost_usd,
                "f",
            ),
        }
    )
    descriptor = build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        campaign_identity_sha256=SHA_A,
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_job_name="glm52-sky-20260724",
        execution_deadline="2026-07-31T12:00:00Z",
        gpu_allocation_sha256=allocation_identity,
        base_descriptor_s3_uri=(
            "s3://keep-glm52-246813579024-us-west-2/"
            "campaigns/glm52-sky-20260724/descriptor.json"
        ),
        base_descriptor_version_id="descriptor-version-opaque",
        base_descriptor_file_sha256=SHA_A,
        base_descriptor_body_sha256=SHA_B,
        archive_identity_sha256=SHA_A,
        repository_archive_version_id="archive-version-opaque",
        approval_identity_sha256=SHA_A,
        approval_version_id="approval-version-opaque",
        intent_identity_sha256=SHA_B,
        intent_version_id="intent-version-opaque",
        task8_live_h1d_identity_sha256=SHA_A,
        task8_spend_authority_identity_sha256=SHA_B,
        task9_launch_identity_sha256=SHA_A,
        task9_admission_identity_sha256=SHA_B,
        task9_custody_identity_sha256=SHA_A,
    )
    observation = build_worker_instance_observation(
        schema_version=1,
        record_type="glm52_task10_worker_instance_observation_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        campaign_identity_sha256=descriptor.campaign_identity_sha256,
        activation_id=descriptor.activation_id,
        activation_ordinal=descriptor.activation_ordinal,
        activation_ordinal_text="00000001",
        generation=descriptor.generation,
        generation_text=descriptor.generation_text,
        allocation_ordinal=1,
        allocation_ordinal_text="00000001",
        instance_id=allocation.instance_id,
        action_key=descriptor.action_key,
        sky_job_name=descriptor.sky_job_name,
        task_yaml_sha256=SHA_A,
        request_body_sha256=SHA_B,
        task9_launch_identity_sha256=(
            descriptor.task9_launch_identity_sha256
        ),
        task9_custody_identity_sha256=(
            descriptor.task9_custody_identity_sha256
        ),
    )
    return descriptor, observation


def _hashes() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {name: hashlib.sha256(name.encode()).hexdigest() for name in UNIT_NAMES},
        {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in GRACEFUL_SCRIPT_NAMES
        },
    )


def _candidate(
    *,
    allocation: AllocationInterval | None = None,
    reserve: GpuLiabilityReserveResult | None = None,
) -> object:
    from glm52_enforcement.task12_worker_drain import (
        prepare_worker_drain_candidate,
    )

    allocation = _allocation() if allocation is None else allocation
    descriptor, observation = _runtime(allocation=allocation)
    unit_hashes, script_hashes = _hashes()
    return prepare_worker_drain_candidate(
        allocation=allocation,
        liability_reserve=_reserve() if reserve is None else reserve,
        worker_descriptor=descriptor,
        worker_observation=observation,
        expected_unit_hashes=unit_hashes,
        expected_script_hashes=script_hashes,
        document_version="7",
    )


def _authority(candidate: object) -> object:
    from glm52_enforcement.task12_worker_drain import (
        build_task12_worker_drain_authority,
    )
    from glm52_enforcement.task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
    )

    action = RetainedWriterActionAuthority(
        authority_domain="RECOVERY",
        action_kind="WORKER_DRAIN_SIGNAL",
        action_key=(
            "ACTIVATION#activation-1#RECOVERY_ACTION#"
            "WORKER_DRAIN_SIGNAL#00000001"
        ),
        candidate_identity_sha256=candidate.candidate_identity_sha256,
        action_identity_sha256=SHA_A,
        owner_invocation_nonce_sha256=SHA_B,
        state="CONSUMED",
        authorized_revision=8,
    )
    audit_body = {
        "authority_domain": action.authority_domain,
        "action_kind": action.action_kind,
        "action_key": action.action_key,
        "candidate_identity_sha256": (
            action.candidate_identity_sha256
        ),
        "action_identity_sha256": action.action_identity_sha256,
        "audit_kind": "H1F_GENESIS_TO_ZERO_CHILD",
        "closing_revision": 7,
        "authorized_revision": 8,
        "current_revision": 8,
        "observed_at": TS,
    }
    audit = RetainedWriterAuditAuthority(
        **audit_body,
        canonical_identity_sha256=canonical_sha256(audit_body),
    )
    return build_task12_worker_drain_authority(
        candidate=candidate,
        action=action,
        audit=audit,
    )


def test_worker_drain_authority_cross_binds_task9_task10_and_task12() -> None:
    candidate = _candidate()
    authority = _authority(candidate)

    assert authority.candidate == candidate
    assert authority.action.action_kind == "WORKER_DRAIN_SIGNAL"
    assert authority.audit.authorized_revision == 8
    assert authority.worker_drain_role_arn == (
        "arn:aws:iam::246813579024:"
        "role/keep-glm52-h1g-worker-drain-signal"
    )
    assert authority.authority_identity_sha256 == canonical_sha256(
        {
            "candidate_identity_sha256": (
                candidate.candidate_identity_sha256
            ),
            "action_identity_sha256": (
                authority.action.action_identity_sha256
            ),
            "audit_identity_sha256": (
                authority.audit.canonical_identity_sha256
            ),
        }
    )


class _Ssm:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def send_command(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "ssm-request-1",
                "RetryAttempts": 0,
            },
            "Command": {
                "CommandId": (
                    "12345678-1234-1234-1234-123456789abc"
                ),
                "DocumentName": GRACEFUL_STOP_DOCUMENT,
                "DocumentVersion": "7",
                "InstanceIds": [INSTANCE_ID],
                "Status": "Pending",
            },
        }


def test_ssm_acceptance_is_one_parameterless_call_and_not_drain_completion() -> None:
    from glm52_enforcement.task12_worker_drain import (
        WorkerDrainServices,
        dispatch_worker_drain,
    )

    authority = _authority(_candidate())
    ssm = _Ssm()
    dispatch = dispatch_worker_drain(
        services=WorkerDrainServices(ssm=ssm),
        authority=authority,
    )

    assert len(ssm.calls) == 1
    assert ssm.calls[0] == {
        "DocumentName": GRACEFUL_STOP_DOCUMENT,
        "DocumentVersion": "7",
        "InstanceIds": [INSTANCE_ID],
    }
    assert dispatch.state == "SSM_ACCEPTED_NOT_DRAINED"
    assert dispatch.drain_complete is False
    assert dispatch.liability_settled is False


def _graceful(command_id: str) -> dict[str, object]:
    unit_hashes, script_hashes = _hashes()
    return build_graceful_stop_evidence(
        unit_hashes=unit_hashes,
        script_hashes=script_hashes,
        active_state="inactive",
        sub_state="dead",
        result="success",
        exec_main_code=1,
        exec_main_status=0,
        control_group_pids=[],
        stop_file_identity_sha256=SHA_A,
        checkpoint_identity_sha256=SHA_B,
        latest_marker_identity_sha256=SHA_A,
        campaign_terminal_marker_identity_sha256=SHA_B,
        ssm_command_id=command_id,
        authority="SSM",
    )


def _settlement(
    reserve: GpuLiabilityReserveResult,
    *,
    reserve_identity: str | None = None,
) -> LiabilitySettlementEvidence:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_liability_settlement_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "reserve_identity_sha256": (
            reserve.reserve_identity_sha256
            if reserve_identity is None
            else reserve_identity
        ),
        "liability_identity_sha256": reserve.reserve_identity_sha256,
        "allocation_close_identity_sha256": SHA_A,
        "instance_terminal_identity_sha256": SHA_B,
        "charged_gpu_seconds": 900,
        "charged_gpu_cost_usd": Decimal("13.76"),
        "refundable_gpu_seconds": 0,
        "refundable_gpu_cost_usd": Decimal("0.00"),
        "settled_at": TS,
    }
    identity = hashlib.sha256(
        canonical_decimal_json_bytes(body)
    ).hexdigest()
    return LiabilitySettlementEvidence(
        **body,
        canonical_body_sha256=identity,
    )


def test_drain_completes_only_after_graceful_observation_and_settlement() -> None:
    from glm52_enforcement.task12_worker_drain import (
        WorkerDrainServices,
        complete_worker_drain,
        dispatch_worker_drain,
    )

    candidate = _candidate()
    authority = _authority(candidate)
    dispatch = dispatch_worker_drain(
        services=WorkerDrainServices(ssm=_Ssm()),
        authority=authority,
    )
    completion = complete_worker_drain(
        authority=authority,
        dispatch=dispatch,
        graceful_stop_observation=_graceful(dispatch.command_id),
        liability_settlement=_settlement(
            candidate.liability_reserve
        ),
    )

    assert completion.state == "DRAINED_AND_LIABILITY_SETTLED"
    assert completion.drain_complete is True
    assert completion.liability_settled is True
    assert completion.command_id == dispatch.command_id


@pytest.mark.parametrize(
    "mutation",
    [
        "allocation-instance",
        "allocation-identity",
        "liability-reserve",
        "generic-action",
        "stale-audit",
        "command-mismatch",
        "active-service",
        "missing-settlement",
        "foreign-settlement",
    ],
)
def test_worker_drain_permanent_mutants_cannot_claim_completion(
    mutation: str,
) -> None:
    from glm52_enforcement.task12_worker_drain import (
        Task12WorkerDrainError,
        WorkerDrainServices,
        build_task12_worker_drain_authority,
        complete_worker_drain,
        dispatch_worker_drain,
        prepare_worker_drain_candidate,
    )
    from glm52_enforcement.task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
    )

    candidate = _candidate()
    authority = _authority(candidate)
    if mutation == "allocation-instance":
        descriptor, observation = _runtime()
        unit_hashes, script_hashes = _hashes()
        with pytest.raises(Task12WorkerDrainError):
            prepare_worker_drain_candidate(
                allocation=_allocation(
                    instance_id="i-0fedcba9876543210"
                ),
                liability_reserve=_reserve(),
                worker_descriptor=descriptor,
                worker_observation=observation,
                expected_unit_hashes=unit_hashes,
                expected_script_hashes=script_hashes,
                document_version="7",
            )
        return
    if mutation == "allocation-identity":
        with pytest.raises(Task12WorkerDrainError):
            build_task12_worker_drain_authority(
                candidate=replace(
                    candidate,
                    allocation_identity_sha256=SHA_B,
                ),
                action=authority.action,
                audit=authority.audit,
            )
        return
    if mutation == "liability-reserve":
        foreign = replace(_reserve(), reserve_identity_sha256=SHA_B)
        with pytest.raises(Task12WorkerDrainError):
            _candidate(reserve=foreign)
        return
    if mutation in {"generic-action", "stale-audit"}:
        action: RetainedWriterActionAuthority = authority.action
        audit: RetainedWriterAuditAuthority = authority.audit
        if mutation == "generic-action":
            action = replace(
                action,
                authority_domain="ACTIVATION",
                action_kind="S3_CREATE",
            )
        else:
            audit = replace(audit, current_revision=9)
        with pytest.raises(Task12WorkerDrainError):
            build_task12_worker_drain_authority(
                candidate=candidate,
                action=action,
                audit=audit,
            )
        return
    dispatch = dispatch_worker_drain(
        services=WorkerDrainServices(ssm=_Ssm()),
        authority=authority,
    )
    graceful: object = _graceful(dispatch.command_id)
    settlement: object = _settlement(candidate.liability_reserve)
    if mutation == "command-mismatch":
        graceful = _graceful(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        )
    elif mutation == "active-service":
        graceful = {**graceful, "ActiveState": "active"}
        body = dict(graceful)
        body.pop("graceful_stop_body_sha256")
        graceful["graceful_stop_body_sha256"] = canonical_sha256(body)
    elif mutation == "missing-settlement":
        settlement = None
    elif mutation == "foreign-settlement":
        settlement = _settlement(
            candidate.liability_reserve,
            reserve_identity=SHA_B,
        )
    with pytest.raises(Task12WorkerDrainError):
        complete_worker_drain(
            authority=authority,
            dispatch=dispatch,
            graceful_stop_observation=graceful,
            liability_settlement=settlement,
        )
