"""Strict SkyPilot submission intent and acceptance lifecycle tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    build_must_start_controller_observation,
    build_must_start_job_binding,
)
from mlx_vq.quality.glm52_sky_must_start_dynamic import (
    build_dynamic_v2_job_binding,
    dynamic_v2_canonical_bytes,
    dynamic_v2_job_binding_s3_key,
)

RUN_ID = "glm52-sky-20260723"
INTENT_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
MUST_START_BY = datetime(2026, 7, 26, 18, 0, tzinfo=UTC)
CONTROLLER_SUBMITTED_AT = datetime(2026, 7, 26, 12, 0, 5, tzinfo=UTC)
CONTROLLER_OBSERVED_AT = datetime(2026, 7, 26, 12, 0, 7, tzinfo=UTC)
ACCEPTED_AT = datetime(2026, 7, 26, 12, 0, 10, tzinfo=UTC)
BASELINE_OBSERVED_AT = datetime(2026, 7, 26, 12, 0, 1, tzinfo=UTC)
ACQUIRED_AT = datetime(2026, 7, 26, 12, 0, 3, tzinfo=UTC)
BINDING_BOUND_AT = datetime(2026, 7, 26, 12, 0, 6, tzinfo=UTC)
CONTROLLER_IDENTITY = "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
CONTROLLER_INSTANCE_ID = "i-0511af4e31aa5406a"
CONTROLLER_INSTANCE_TYPE = "c6a.xlarge"
CONTROLLER_PROFILE_ARN = (
    "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
)
CONTROLLER_CLUSTER_NAME = "sky-jobs-controller-9d9f31a9-9d9f31a9"


def _module() -> Any:
    return importlib.import_module("mlx_vq.quality.glm52_sky_submission_lifecycle")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha(body: dict[str, object]) -> str:
    return _sha(_canonical(body))


def _self_hashed(body: dict[str, object], digest_field: str) -> dict[str, object]:
    return {**body, digest_field: _body_sha(body)}


def _rehash(value: dict[str, object], digest_field: str) -> dict[str, object]:
    body = dict(value)
    body.pop(digest_field)
    return _self_hashed(body, digest_field)


def _descriptor() -> dict[str, object]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
        slack_permalink=None,
    )
    approval_raw = _canonical(approval) + b"\n"
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=MUST_START_BY,
        controller_identity=CONTROLLER_IDENTITY,
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-us-west-2-246813579024",
        jobs_bucket="keep-glm52-us-west-2-246813579024",
        repo_tar_key=(f"campaigns/{RUN_ID}/repository/repo.tar.gz"),
        repo_tar_sha256="1" * 64,
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/qualification/campaign-descriptor-v2.json"
        ),
        approval_key=(f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"),
        approval_sha256=_sha(approval_raw),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": "2" * 64,
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": "3" * 64,
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": "4" * 64,
            "frozen_prompt_pack_key": "quality/frozen-66.json",
            "frozen_prompt_pack_sha256": "1" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
            "training_config_key": (f"campaigns/{RUN_ID}/authorities/training.json"),
            "training_config_sha256": "7" * 64,
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'8' * 64}.json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": "qualification-cache/",
            "qualification_cache_manifest_sha256": "0" * 64,
        },
    )


def _cache_seed_acceptance(
    descriptor: dict[str, object],
) -> dict[str, object]:
    descriptor_file_sha256 = _sha(_canonical(descriptor) + b"\n")
    submission_body_sha256 = "c" * 64
    submission_key = (
        f"campaigns/{RUN_ID}/monitor/submission-locks/"
        f"{descriptor_file_sha256}-cache-seed.json"
    )
    job_binding = build_must_start_job_binding(
        run_id=RUN_ID,
        managed_mode="cache-seed",
        account_id=str(descriptor["account_id"]),
        region=str(descriptor["region"]),
        bucket=str(descriptor["bucket"]),
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=submission_body_sha256,
        sky_job_name=f"{RUN_ID}-cache-seed",
        must_start_by=MUST_START_BY,
        descriptor_key=str(descriptor["campaign_descriptor_key"]),
        descriptor_file_sha256=descriptor_file_sha256,
        submission_key=submission_key,
        submission_submitted_at="2026-07-26T07:50:00Z",
        target_job_id=3,
        workspace="default",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        observation_body_sha256="e" * 64,
        bound_at="2026-07-26T07:55:00Z",
    )
    cache_prefix = f"qualification-cache/seeds/{RUN_ID}/{'9' * 64}/"
    manifest_key = f"{cache_prefix}glm52-teacher-signal-cache-v3-manifest.json"
    teacher_ready_key = f"{cache_prefix}TEACHER_CACHE_READY.json"
    prompt_pack_key = f"{cache_prefix}prompt-pack.json"
    shard_key = f"{cache_prefix}teacher_signal/row-000000.safetensors"
    inventory: list[dict[str, object]] = sorted(
        [
            {
                "key": teacher_ready_key,
                "size": 1024,
                "sha256": "8" * 64,
                "checksum_type": "FULL_OBJECT",
                "etag": '"11111111111111111111111111111111"',
                "version_id": None,
            },
            {
                "key": manifest_key,
                "size": 4096,
                "sha256": "9" * 64,
                "checksum_type": "FULL_OBJECT",
                "etag": '"22222222222222222222222222222222"',
                "version_id": None,
            },
            {
                "key": prompt_pack_key,
                "size": 2048,
                "sha256": "7" * 64,
                "checksum_type": "FULL_OBJECT",
                "etag": '"33333333333333333333333333333333"',
                "version_id": None,
            },
            {
                "key": shard_key,
                "size": 8192,
                "sha256": "6" * 64,
                "checksum_type": "FULL_OBJECT",
                "etag": '"44444444444444444444444444444444"',
                "version_id": None,
            },
        ],
        key=lambda item: str(item["key"]),
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_qualification_cache_seed_accepted_v1",
        "seed_authority": {
            "account_id": descriptor["account_id"],
            "region": descriptor["region"],
            "bucket": descriptor["bucket"],
            "run_id": RUN_ID,
            "managed_mode": "cache-seed",
            "target_job_id": 3,
            "sky_job_name": f"{RUN_ID}-cache-seed",
            "descriptor_key": descriptor["campaign_descriptor_key"],
            "descriptor_file_sha256": descriptor_file_sha256,
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "repo_tar_key": descriptor["repo_tar_key"],
            "repo_tar_sha256": descriptor["repo_tar_sha256"],
            "approval_key": descriptor["approval_key"],
            "approval_sha256": descriptor["approval_sha256"],
            "submission_key": submission_key,
            "submission_file_sha256": "b" * 64,
            "submission_body_sha256": submission_body_sha256,
            "submission_alias_key": (
                f"campaigns/{RUN_ID}/monitor/QUALIFICATION_CACHE_SEED_SUBMITTED.json"
            ),
            "job_binding_key": (
                f"campaigns/{RUN_ID}/monitor/must-start/cache-seed/"
                f"{submission_body_sha256}/JOB_BINDING.json"
            ),
            "job_binding_file_sha256": _sha(_canonical(job_binding)),
            "job_binding_body_sha256": job_binding["job_binding_body_sha256"],
            "job_binding": job_binding,
            "seed_ready_key": (
                f"campaigns/{RUN_ID}/qualification-cache-seed/"
                "QUALIFICATION_CACHE_SEED_READY.json"
            ),
        },
        "cache_object_inventory": inventory,
        "cache_object_inventory_sha256": _body_sha(inventory),
        "cache_audit": {
            "legacy_seed_ready_file_sha256": "5" * 64,
            "legacy_seed_ready_body_sha256": "4" * 64,
            "cache_prefix": cache_prefix,
            "manifest_key": manifest_key,
            "manifest_file_sha256": "9" * 64,
            "manifest_body_sha256": "3" * 64,
            "teacher_cache_ready_key": teacher_ready_key,
            "teacher_cache_ready_file_sha256": "8" * 64,
            "teacher_cache_ready_body_sha256": "2" * 64,
            "prompt_pack_key": prompt_pack_key,
            "prompt_pack_file_sha256": "7" * 64,
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_file_sha256": "4" * 64,
            "frozen_prompt_pack_key": "quality/frozen-66.json",
            "frozen_prompt_pack_file_sha256": "1" * 64,
            "prompt_id": "teich_claude_agent-a3622521df8a9137d",
            "session_count": 1,
            "supervised_position_count": 64,
            "top_k": 2048,
            "hidden_size": 6144,
            "semantic_audit_pass": True,
        },
        "spend_closure": {
            "allocation_key": (
                f"campaigns/{RUN_ID}/runtime/GPU_RUNTIME_ALLOCATION.json"
            ),
            "allocation_file_sha256": "0" * 64,
            "allocation_body_sha256": "1" * 64,
            "allocation_start_record_sha256": "2" * 64,
            "spend_authority_sha256": "3" * 64,
            "ledger_key": (f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"),
            "ledger_file_sha256": "4" * 64,
            "ledger_tip_record_sha256": "d" * 64,
            "ledger_tip_event": "allocation_ended",
            "spend_status_key": (f"campaigns/{RUN_ID}/runtime/GPU_SPEND_STATUS.json"),
            "spend_status_file_sha256": "5" * 64,
            "job_status_key": (f"campaigns/{RUN_ID}/monitor/SKY_JOB_STATUS.json"),
            "job_status_file_sha256": "6" * 64,
            "job_status_body_sha256": "7" * 64,
            "job_status": "SUCCEEDED",
            "instance_id": "i-00000000000000003",
            "sky_job_name": f"{RUN_ID}-cache-seed",
            "submission_submitted_at": "2026-07-26T07:50:00Z",
            "launched_at": "2026-07-26T08:00:00Z",
            "must_start_by": "2026-07-26T18:00:00Z",
            "ended_at": "2026-07-26T10:00:00Z",
            "job_observed_at": "2026-07-26T11:15:00Z",
            "consumed_gpu_seconds": 10_800,
            "remaining_gpu_seconds": 75_600,
            "estimated_gpu_cost_usd": 165.12,
        },
        "accepted_at": "2026-07-26T11:30:00Z",
    }
    return _self_hashed(body, "acceptance_body_sha256")


def _spend_snapshot(
    descriptor: dict[str, object],
) -> dict[str, object]:
    descriptor_file_sha256 = _sha(_canonical(descriptor) + b"\n")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": "e" * 64,
        "gpu_spend_ledger_latest_sha256": "f" * 64,
        "gpu_spend_ledger_latest_body_sha256": "0" * 64,
        "gpu_spend_ledger_genesis_sha256": "a" * 64,
        "gpu_spend_ledger_record_count": 4,
        "gpu_spend_ledger_tip_record_sha256": "d" * 64,
        "gpu_spend_ledger_file_sha256": "b" * 64,
        "ec2_allocation_history_sha256": "c" * 64,
        "ec2_allocation_instance_ids": [
            "i-00000000000000001",
            "i-00000000000000002",
        ],
        "observed_at": "2026-07-26T11:50:00Z",
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "hourly_cost_usd": 55.04,
        "consumed_gpu_seconds": 10_800,
        "remaining_gpu_seconds": 75_600,
        "consumed_gpu_cost_usd": 165.12,
        "remaining_gpu_cost_usd": 1_155.84,
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "open_allocation_count": 0,
    }
    return _self_hashed(body, "snapshot_body_sha256")


def _post_seed_descriptor(
    seed_descriptor: dict[str, object],
    acceptance: dict[str, object],
) -> dict[str, object]:
    artifacts = dict(seed_descriptor["artifacts"])
    cache_audit = acceptance["cache_audit"]
    assert isinstance(cache_audit, dict)
    artifacts.update(
        {
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'a' * 64}.json"
            ),
            "artifact_inventory_sha256": "a" * 64,
            "qualification_cache_prefix": cache_audit["cache_prefix"],
            "qualification_cache_manifest_sha256": cache_audit["manifest_file_sha256"],
        }
    )
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=MUST_START_BY,
        controller_identity=str(seed_descriptor["controller_identity"]),
        worker_identity=str(seed_descriptor["worker_identity"]),
        vpc_name=str(seed_descriptor["vpc_name"]),
        image_id=str(seed_descriptor["image_id"]),
        bucket=str(seed_descriptor["bucket"]),
        jobs_bucket=str(seed_descriptor["jobs_bucket"]),
        repo_tar_key=str(seed_descriptor["repo_tar_key"]),
        repo_tar_sha256=str(seed_descriptor["repo_tar_sha256"]),
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/post-seed/campaign-descriptor-v2.json"
        ),
        approval_key=str(seed_descriptor["approval_key"]),
        approval_sha256=str(seed_descriptor["approval_sha256"]),
        artifacts=artifacts,
    )


def _staged_readiness(
    descriptor: dict[str, object],
    *,
    descriptor_file_sha256: str,
) -> tuple[dict[str, object], str, str]:
    key = (
        str(descriptor["campaign_descriptor_key"]).removesuffix(
            "campaign-descriptor-v2.json"
        )
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    manifest_body_sha256 = "b" * 64
    manifest_key = (
        str(descriptor["campaign_descriptor_key"]).removesuffix(
            "campaign-descriptor-v2.json"
        )
        + f"bundle-manifests/{manifest_body_sha256}/bundle-manifest-v1.json"
    )
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": RUN_ID,
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": manifest_key,
        "bundle_manifest_file_sha256": "d" * 64,
        "bundle_manifest_body_sha256": manifest_body_sha256,
        "bundle_manifest_version_id": "version-bundle-manifest",
        "staged_object_version_ids": {
            "repository_tar": "version-repository",
            "approval": "version-approval",
            "training_config": "version-training-config",
            "watchdog": "version-watchdog",
            "artifact_inventory": "version-inventory",
            "artifact_audit": "version-audit",
            "descriptor": "version-descriptor",
        },
        "artifact_audit_key": (
            f"campaigns/{RUN_ID}/audits/artifact-audit-{'c' * 64}.json"
        ),
        "artifact_audit_sha256": "c" * 64,
        "staged_at": "2026-07-26T11:35:00Z",
    }
    value = _self_hashed(body, "ready_body_sha256")
    return value, key, _sha(_canonical(value) + b"\n")


def _rehearsal_evidence(
    descriptor: dict[str, object],
    *,
    descriptor_file_sha256: str,
    staged_readiness: dict[str, object],
    staged_readiness_key: str,
    staged_readiness_file_sha256: str,
) -> tuple[dict[str, object], str, str]:
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_rehearsal_v2",
        "status": "passed_before_cuda_h100_boundary",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "bootstrap_receipt_file_sha256": "0" * 64,
        "staged_readiness_key": staged_readiness_key,
        "staged_readiness_file_sha256": staged_readiness_file_sha256,
        "staged_readiness_body_sha256": staged_readiness["ready_body_sha256"],
        "artifact_inventory_key": descriptor["artifacts"]["artifact_inventory_key"],
        "artifact_inventory_file_sha256": descriptor["artifacts"][
            "artifact_inventory_sha256"
        ],
        "artifact_inventory_body_sha256": "1" * 64,
        "artifact_audit_key": staged_readiness["artifact_audit_key"],
        "artifact_audit_file_sha256": staged_readiness["artifact_audit_sha256"],
        "extracted_repo_path": "/tmp/glm52-lifecycle-rehearsal/repo",
        "production_repo_path": "/opt/keep-campaign/repo",
        "production_resume_root": "/mnt/nvme/glm52-campaign",
        "skypilot_task_path": (
            "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ),
        "skypilot_task_file_sha256": "2" * 64,
        "skypilot_config_file_sha256": "3" * 64,
        "skypilot_validation_file_sha256": "4" * 64,
        "skypilot_version": "0.13.0",
        "skypilot_task_name": descriptor["task_name"],
        "completed_at": "2026-07-26T11:50:00Z",
    }
    value = _self_hashed(body, "rehearsal_body_sha256")
    body_sha256 = str(value["rehearsal_body_sha256"])
    key = (
        f"campaigns/{RUN_ID}/qualification/rehearsals/{body_sha256}/"
        "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )
    return value, key, _sha(_canonical(value) + b"\n")


def _qualification_submission_readiness(
    *,
    seed_descriptor: dict[str, object],
    descriptor: dict[str, object],
    acceptance: dict[str, object],
    snapshot: dict[str, object],
) -> dict[str, object]:
    ready_module = importlib.import_module(
        "mlx_vq.quality.glm52_qualification_submission_ready"
    )
    descriptor_file_sha256 = _sha(_canonical(descriptor) + b"\n")
    seed_descriptor_raw = _canonical(seed_descriptor) + b"\n"
    seed_descriptor_file_sha256 = _sha(seed_descriptor_raw)
    staged, staged_key, staged_file_sha256 = _staged_readiness(
        descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
    )
    rehearsal, rehearsal_key, rehearsal_file_sha256 = _rehearsal_evidence(
        descriptor,
        descriptor_file_sha256=descriptor_file_sha256,
        staged_readiness=staged,
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=staged_file_sha256,
    )
    acceptance_body_sha256 = str(acceptance["acceptance_body_sha256"])
    acceptance_key = (
        f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
        f"{acceptance_body_sha256}/QUALIFICATION_CACHE_SEED_ACCEPTED.json"
    )
    acceptance_file_sha256 = _sha(_canonical(acceptance) + b"\n")
    snapshot_body_sha256 = str(snapshot["snapshot_body_sha256"])
    snapshot_key = (
        f"campaigns/{RUN_ID}/spend-snapshots/{snapshot_body_sha256}/"
        "GPU_SPEND_SNAPSHOT.json"
    )
    snapshot_file_sha256 = _sha(_canonical(snapshot) + b"\n")
    ready = ready_module.build_qualification_submission_ready(
        descriptor=descriptor,
        seed_descriptor_raw=seed_descriptor_raw,
        seed_descriptor_file_sha256=seed_descriptor_file_sha256,
        staged_readiness=staged,
        cache_seed_acceptance=acceptance,
        gpu_spend_snapshot=snapshot,
        rehearsal_evidence=rehearsal,
        descriptor_file_sha256=descriptor_file_sha256,
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=staged_file_sha256,
        cache_seed_acceptance_key=acceptance_key,
        cache_seed_acceptance_file_sha256=acceptance_file_sha256,
        gpu_spend_snapshot_key=snapshot_key,
        gpu_spend_snapshot_sha256=snapshot_file_sha256,
        rehearsal_evidence_key=rehearsal_key,
        rehearsal_evidence_sha256=rehearsal_file_sha256,
        built_at="2026-07-26T11:55:00Z",
    )
    return {
        "staged_readiness": staged,
        "rehearsal_evidence": rehearsal,
        "qualification_submission_ready": ready,
        "qualification_submission_ready_key": (
            ready_module.qualification_submission_ready_s3_key(ready)
        ),
        "qualification_submission_ready_sha256": _sha(_canonical(ready) + b"\n"),
        "qualification_submission_ready_body_sha256": ready["readiness_body_sha256"],
    }


def _intent_inputs() -> dict[str, object]:
    seed_descriptor = _descriptor()
    acceptance = _cache_seed_acceptance(seed_descriptor)
    descriptor = _post_seed_descriptor(seed_descriptor, acceptance)
    snapshot = _spend_snapshot(descriptor)
    acceptance_body_sha256 = str(acceptance["acceptance_body_sha256"])
    snapshot_body_sha256 = str(snapshot["snapshot_body_sha256"])
    readiness = _qualification_submission_readiness(
        seed_descriptor=seed_descriptor,
        descriptor=descriptor,
        acceptance=acceptance,
        snapshot=snapshot,
    )
    return {
        "descriptor": descriptor,
        "managed_mode": "qualification",
        "descriptor_file_sha256": _sha(_canonical(descriptor) + b"\n"),
        "cache_seed_acceptance": acceptance,
        "cache_seed_acceptance_key": (
            f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
            f"{acceptance_body_sha256}/"
            "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        "cache_seed_acceptance_file_sha256": _sha(_canonical(acceptance) + b"\n"),
        "gpu_spend_snapshot": snapshot,
        "gpu_spend_snapshot_key": (
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot_body_sha256}/GPU_SPEND_SNAPSHOT.json"
        ),
        "gpu_spend_snapshot_sha256": _sha(_canonical(snapshot) + b"\n"),
        **readiness,
        "sky_job_name": f"{RUN_ID}-qualification",
        "intent_at": INTENT_AT,
    }


def _build_intent() -> tuple[Any, dict[str, object], dict[str, object]]:
    module = _module()
    inputs = _intent_inputs()
    intent = module.build_submission_intent(**inputs)
    baseline, acquisition = _dynamic_authorities(intent, inputs)
    inputs["controller_baseline"] = baseline
    inputs["submission_acquisition"] = acquisition
    return module, inputs, intent


def _readiness_objects(inputs: dict[str, object]) -> dict[str, object]:
    return {
        "staged_readiness": inputs["staged_readiness"],
        "rehearsal_evidence": inputs["rehearsal_evidence"],
        "qualification_submission_ready": inputs["qualification_submission_ready"],
    }


def _accepted_authorities(inputs: dict[str, object]) -> dict[str, object]:
    return {
        **_readiness_objects(inputs),
        "controller_baseline": inputs["controller_baseline"],
        "submission_acquisition": inputs["submission_acquisition"],
    }


def _dynamic_authorities(
    intent: dict[str, object],
    inputs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    descriptor = inputs["descriptor"]
    assert isinstance(descriptor, dict)
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/qualification/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    baseline_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_controller_baseline_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": descriptor["bucket"],
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "intent_key": intent_key,
        "intent_file_sha256": _sha(_canonical(intent) + b"\n"),
        "intent_body_sha256": intent["intent_body_sha256"],
        "sky_job_name": intent["sky_job_name"],
        "workspace": "default",
        "controller_instance_id": CONTROLLER_INSTANCE_ID,
        "controller_instance_type": CONTROLLER_INSTANCE_TYPE,
        "controller_profile_arn": CONTROLLER_PROFILE_ARN,
        "controller_cluster_name": CONTROLLER_CLUSTER_NAME,
        "ssm_ping_status": "Online",
        "exact_name_history": [],
        "active_exact_name_job_ids": [],
        "active_tagged_p5_instance_ids": [],
        "observed_at": BASELINE_OBSERVED_AT.isoformat().replace("+00:00", "Z"),
    }
    baseline = _self_hashed(baseline_body, "baseline_body_sha256")
    baseline_key = (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )
    control_ready_body_sha256 = "a" * 64
    acquisition_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_submission_acquired_v1",
        "account_id": intent["account_id"],
        "region": intent["region"],
        "run_id": intent["run_id"],
        "managed_mode": intent["managed_mode"],
        "descriptor_key": intent["descriptor_key"],
        "descriptor_file_sha256": intent["descriptor_file_sha256"],
        "descriptor_body_sha256": intent["descriptor_body_sha256"],
        "qualification_submission_ready_key": intent[
            "qualification_submission_ready_key"
        ],
        "qualification_submission_ready_sha256": intent[
            "qualification_submission_ready_sha256"
        ],
        "qualification_submission_ready_body_sha256": intent[
            "qualification_submission_ready_body_sha256"
        ],
        "intent_key": intent_key,
        "intent_file_sha256": _sha(_canonical(intent) + b"\n"),
        "intent_body_sha256": intent["intent_body_sha256"],
        "controller_baseline_key": baseline_key,
        "controller_baseline_file_sha256": _sha(_canonical(baseline) + b"\n"),
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        "must_start_control_plane_ready_key": (
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/control-plane-ready/"
            f"{control_ready_body_sha256}/CONTROL_PLANE_READY.json"
        ),
        "must_start_control_plane_ready_file_sha256": "b" * 64,
        "must_start_control_plane_ready_body_sha256": control_ready_body_sha256,
        "sky_job_name": intent["sky_job_name"],
        "must_start_by": intent["must_start_by"],
        "acquired_at": ACQUIRED_AT.isoformat().replace("+00:00", "Z"),
    }
    acquisition = _self_hashed(acquisition_body, "acquisition_body_sha256")
    return baseline, acquisition


def _controller_job(
    intent: dict[str, object],
    *,
    sky_job_id: object = 17,
    sky_job_name: object | None = None,
    workspace: object = "default",
    controller_identity: object = CONTROLLER_IDENTITY,
    controller_status: object = "PENDING",
    controller_submitted_at: object = CONTROLLER_SUBMITTED_AT,
) -> dict[str, object]:
    return {
        "sky_job_id": sky_job_id,
        "sky_job_name": (
            intent["sky_job_name"] if sky_job_name is None else sky_job_name
        ),
        "workspace": workspace,
        "controller_identity": controller_identity,
        "controller_submitted_at": (
            controller_submitted_at.isoformat().replace("+00:00", "Z")
            if isinstance(controller_submitted_at, datetime)
            else controller_submitted_at
        ),
        "controller_status": controller_status,
    }


def _controller_observation(
    intent: dict[str, object],
    inputs: dict[str, object],
) -> dict[str, object]:
    descriptor = inputs["descriptor"]
    assert isinstance(descriptor, dict)
    return build_must_start_controller_observation(
        run_id=RUN_ID,
        managed_mode="qualification",
        account_id="246813579024",
        region="us-west-2",
        bucket=str(descriptor["bucket"]),
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(intent["intent_body_sha256"]),
        sky_job_name=str(intent["sky_job_name"]),
        must_start_by=str(intent["must_start_by"]),
        target_job_id=17,
        workspace="default",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        status="PENDING",
        schedule_state="LAUNCHING",
        submitted_at=CONTROLLER_SUBMITTED_AT,
        start_at=None,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at=CONTROLLER_OBSERVED_AT,
    )


def _controller_receipt(
    intent: dict[str, object],
    inputs: dict[str, object],
    *,
    observation_status: str = "PENDING",
    binding_status: str = "PENDING",
    binding_bound_at: datetime = BINDING_BOUND_AT,
) -> dict[str, object]:
    observation = _controller_observation(intent, inputs)
    if observation_status != "PENDING":
        observation = _rehash(
            {**observation, "status": observation_status},
            "observation_body_sha256",
        )
    body_sha256 = str(observation["observation_body_sha256"])
    prefix = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}"
    )
    descriptor = inputs["descriptor"]
    baseline = inputs["controller_baseline"]
    acquisition = inputs["submission_acquisition"]
    assert isinstance(descriptor, dict)
    assert isinstance(baseline, dict)
    assert isinstance(acquisition, dict)
    binding_row = _controller_job(
        intent,
        controller_status=binding_status,
    )
    job_binding = build_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=str(descriptor["controller_identity"]),
        current_exact_name_history=[
            *baseline["exact_name_history"],
            binding_row,
        ],
        bound_at=binding_bound_at,
        now=CONTROLLER_OBSERVED_AT,
    )
    return {
        "controller_observation": observation,
        "controller_observation_key": f"{prefix}/observations/{body_sha256}.json",
        "controller_observation_file_sha256": _sha(_canonical(observation)),
        "job_binding": job_binding,
        "job_binding_key": dynamic_v2_job_binding_s3_key(intent=intent),
        "job_binding_file_sha256": _sha(dynamic_v2_canonical_bytes(job_binding)),
    }


def _must_start_controller_receipt(
    intent: dict[str, object],
    inputs: dict[str, object],
) -> dict[str, object]:
    return _controller_receipt(intent, inputs)


def _legacy_qualification_binding(
    intent: dict[str, object],
    inputs: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    descriptor = inputs["descriptor"]
    assert isinstance(descriptor, dict)
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/qualification/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    return build_must_start_job_binding(
        run_id=RUN_ID,
        managed_mode="qualification",
        account_id="246813579024",
        region="us-west-2",
        bucket=str(descriptor["bucket"]),
        descriptor_body_sha256=str(intent["descriptor_body_sha256"]),
        submission_body_sha256=str(intent["intent_body_sha256"]),
        sky_job_name=str(intent["sky_job_name"]),
        must_start_by=str(intent["must_start_by"]),
        descriptor_key=str(intent["descriptor_key"]),
        descriptor_file_sha256=str(intent["descriptor_file_sha256"]),
        submission_key=intent_key,
        submission_submitted_at=str(intent["intent_at"]),
        target_job_id=17,
        workspace="default",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        observation_body_sha256=str(observation["observation_body_sha256"]),
        bound_at=BINDING_BOUND_AT,
    )


def _build_accepted() -> tuple[
    Any,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    module, inputs, intent = _build_intent()
    controller_receipt = _controller_receipt(intent, inputs)
    accepted = module.build_submission_accepted(
        intent=intent,
        descriptor=inputs["descriptor"],
        cache_seed_acceptance=inputs["cache_seed_acceptance"],
        gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
        **_accepted_authorities(inputs),
        controller_job=_controller_job(intent),
        **controller_receipt,
    )
    return module, inputs, intent, controller_receipt, accepted


def _decision(
    module: Any,
    inputs: dict[str, object],
    *,
    intent: dict[str, object] | None,
    accepted: dict[str, object] | None = None,
    accepted_controller_observation: dict[str, object] | None = None,
    controller_jobs: object = (),
    now: datetime = INTENT_AT,
    include_accepted_ancestry: bool = True,
    accepted_ancestry_overrides: dict[str, object | None] | None = None,
) -> Any:
    controller_baseline = (
        inputs.get("controller_baseline") if include_accepted_ancestry else None
    )
    submission_acquisition = (
        inputs.get("submission_acquisition") if include_accepted_ancestry else None
    )
    if accepted_ancestry_overrides is not None:
        controller_baseline = accepted_ancestry_overrides.get(
            "controller_baseline",
            controller_baseline,
        )
        submission_acquisition = accepted_ancestry_overrides.get(
            "submission_acquisition",
            submission_acquisition,
        )
    return module.decide_submission_lifecycle(
        intent=intent,
        accepted=accepted,
        descriptor=inputs["descriptor"],
        cache_seed_acceptance=inputs["cache_seed_acceptance"],
        gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
        staged_readiness=inputs["staged_readiness"],
        rehearsal_evidence=inputs["rehearsal_evidence"],
        qualification_submission_ready=inputs["qualification_submission_ready"],
        qualification_submission_ready_key=inputs["qualification_submission_ready_key"],
        qualification_submission_ready_sha256=inputs[
            "qualification_submission_ready_sha256"
        ],
        controller_baseline=controller_baseline,
        submission_acquisition=submission_acquisition,
        accepted_controller_observation=(accepted_controller_observation),
        controller_jobs=controller_jobs,
        now=now,
    )


def test_qualification_intent_is_exact_canonical_and_self_hashed() -> None:
    module, inputs, intent = _build_intent()

    assert set(intent) == {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "campaign_identity_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "repo_tar_sha256",
        "cache_seed_acceptance_key",
        "cache_seed_acceptance_file_sha256",
        "cache_seed_acceptance_body_sha256",
        "gpu_spend_snapshot_key",
        "gpu_spend_snapshot_sha256",
        "gpu_spend_snapshot_body_sha256",
        "gpu_spend_ledger_tip_record_sha256",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
        "open_allocation_count",
        "qualification_submission_ready_key",
        "qualification_submission_ready_sha256",
        "qualification_submission_ready_body_sha256",
        "sky_job_name",
        "must_start_by",
        "intent_at",
        "intent_body_sha256",
    }
    assert intent["schema_version"] == 2
    assert intent["record_type"] == "glm52_sky_submission_intent_v2"
    assert intent["account_id"] == "246813579024"
    assert intent["region"] == "us-west-2"
    assert intent["managed_mode"] == "qualification"
    assert intent["sky_job_name"] == f"{RUN_ID}-qualification"
    assert intent["remaining_gpu_seconds"] == 75_600
    assert intent["qualification_allowance_seconds"] == 14_400
    body = dict(intent)
    digest = body.pop("intent_body_sha256")
    assert digest == _body_sha(body)
    assert (
        module.validate_submission_intent(
            intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
        )
        == intent
    )


def test_intent_binds_validated_post_seed_readiness_bridge() -> None:
    module, inputs, intent = _build_intent()
    ready = inputs["qualification_submission_ready"]
    descriptor = inputs["descriptor"]
    assert isinstance(ready, dict)
    assert isinstance(descriptor, dict)

    assert (
        ready["seed_campaign_identity_sha256"] != descriptor["campaign_identity_sha256"]
    )
    assert ready["campaign_identity_sha256"] == descriptor["campaign_identity_sha256"]
    assert (
        intent["qualification_submission_ready_body_sha256"]
        == ready["readiness_body_sha256"]
    )
    assert (
        module.validate_submission_intent(
            intent,
            descriptor=descriptor,
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
        )
        == intent
    )


def test_intent_rejects_readiness_older_than_five_minutes() -> None:
    module = _module()
    inputs = _intent_inputs()
    inputs["intent_at"] = "2026-07-26T12:00:01Z"

    with pytest.raises(
        module.SubmissionLifecycleValidationError,
        match=r"readiness.*(?:5|five) minutes",
    ):
        module.build_submission_intent(**inputs)


def test_intent_accepts_readiness_exactly_five_minutes_old() -> None:
    module = _module()
    inputs = _intent_inputs()

    intent = module.build_submission_intent(**inputs)

    assert intent["intent_at"] == "2026-07-26T12:00:00Z"
    assert inputs["qualification_submission_ready"]["built_at"] == (
        "2026-07-26T11:55:00Z"
    )


def test_accepted_record_binds_one_exact_controller_job_and_intent() -> None:
    module, inputs, intent = _build_intent()
    controller_receipt = _controller_receipt(intent, inputs)

    accepted = module.build_submission_accepted(
        intent=intent,
        descriptor=inputs["descriptor"],
        cache_seed_acceptance=inputs["cache_seed_acceptance"],
        gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
        **_accepted_authorities(inputs),
        controller_job=_controller_job(intent),
        **controller_receipt,
    )

    assert set(accepted) == {
        "schema_version",
        "record_type",
        "intent_body_sha256",
        "sky_job_id",
        "sky_job_name",
        "workspace",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "controller_observed_at",
        "controller_observation_key",
        "controller_observation_file_sha256",
        "controller_observation_body_sha256",
        "controller_observation",
        "job_binding_key",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "job_binding",
        "accepted_at",
        "accepted_body_sha256",
    }
    assert accepted["schema_version"] == 2
    assert accepted["record_type"] == "glm52_sky_submission_accepted_v2"
    assert accepted["intent_body_sha256"] == intent["intent_body_sha256"]
    assert accepted["sky_job_id"] == 17
    assert accepted["workspace"] == "default"
    assert accepted["controller_status"] == "PENDING"
    assert accepted["controller_identity"] == CONTROLLER_IDENTITY
    assert accepted["controller_instance_id"] == CONTROLLER_INSTANCE_ID
    assert accepted["controller_profile_arn"] == CONTROLLER_PROFILE_ARN
    assert accepted["controller_cluster_name"] == CONTROLLER_CLUSTER_NAME
    assert (
        accepted["controller_observation_body_sha256"]
        == controller_receipt["controller_observation"]["observation_body_sha256"]
    )
    assert (
        accepted["controller_observation"]
        == controller_receipt["controller_observation"]
    )
    assert accepted["job_binding"] == controller_receipt["job_binding"]
    assert (
        accepted["job_binding_body_sha256"]
        == controller_receipt["job_binding"]["job_binding_body_sha256"]
    )
    assert accepted["job_binding"]["record_type"] == (
        "glm52_sky_must_start_job_binding_v2"
    )
    assert accepted["job_binding_key"].endswith("/DYNAMIC_JOB_BINDING.json")
    assert accepted["accepted_at"] == "2026-07-26T12:00:07Z"
    body = dict(accepted)
    digest = body.pop("accepted_body_sha256")
    assert digest == _body_sha(body)
    assert (
        module.validate_submission_accepted(
            accepted,
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
        )
        == accepted
    )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("account_id", "135792468013"),
        ("region", "us-east-1"),
        ("run_id", "foreign-run"),
        ("descriptor_body_sha256", "f" * 64),
        ("submission_body_sha256", "e" * 64),
        ("target_job_id", True),
        ("target_job_id", 0),
        ("workspace", "other"),
        (
            "controller_profile_arn",
            "arn:aws:iam::246813579024:instance-profile/foreign-controller",
        ),
        ("controller_cluster_name", "foreign-controller"),
        ("status", "RUNNING"),
        ("status", "UNKNOWN"),
        ("submitted_at", "2026-07-26T12:00:06Z"),
        ("submitted_at", "2026-07-26T11:59:59Z"),
        ("observed_at", "2026-07-26T11:59:59Z"),
    ],
)
def test_foreign_controller_observation_receipts_are_rejected(
    field: str,
    foreign: object,
) -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    observation = {
        **receipt["controller_observation"],
        field: foreign,
    }
    observation = _rehash(observation, "observation_body_sha256")
    body_sha256 = str(observation["observation_body_sha256"])
    receipt.update(
        controller_observation=observation,
        controller_observation_key=(
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/observations/{body_sha256}.json"
        ),
        controller_observation_file_sha256=_sha(_canonical(observation)),
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("target_job_id", 18),
        ("status", "RUNNING"),
        ("submitted_at", "2026-07-26T12:00:06Z"),
    ],
)
def test_acceptance_cross_binds_the_reconciled_job_and_controller_receipt(
    field: str,
    foreign: object,
) -> None:
    module, inputs, intent = _build_intent()
    reconciled_job = _controller_job(intent)
    receipt = _controller_receipt(intent, inputs)
    foreign_observation = _rehash(
        {
            **receipt["controller_observation"],
            field: foreign,
        },
        "observation_body_sha256",
    )
    foreign_body_sha256 = str(foreign_observation["observation_body_sha256"])
    binding_changes: dict[str, object] = {
        "observation_body_sha256": foreign_body_sha256,
    }
    if field == "target_job_id":
        binding_changes["target_job_id"] = foreign
    foreign_binding = _rehash(
        {
            **receipt["job_binding"],
            **binding_changes,
        },
        "job_binding_body_sha256",
    )
    foreign_receipt = {
        **receipt,
        "controller_observation": foreign_observation,
        "controller_observation_key": (
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/observations/"
            f"{foreign_body_sha256}.json"
        ),
        "controller_observation_file_sha256": _sha(_canonical(foreign_observation)),
        "job_binding": foreign_binding,
        "job_binding_file_sha256": _sha(_canonical(foreign_binding)),
    }

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=reconciled_job,
            **foreign_receipt,
        )


def test_acceptance_reuses_exact_must_start_observation_key_and_bytes() -> None:
    module, inputs, intent = _build_intent()
    receipt = _must_start_controller_receipt(intent, inputs)

    accepted = module.build_submission_accepted(
        intent=intent,
        descriptor=inputs["descriptor"],
        cache_seed_acceptance=inputs["cache_seed_acceptance"],
        gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
        **_accepted_authorities(inputs),
        controller_job=_controller_job(intent),
        **receipt,
    )

    assert (
        accepted["controller_observation_key"] == receipt["controller_observation_key"]
    )
    assert (
        accepted["controller_observation_file_sha256"]
        == receipt["controller_observation_file_sha256"]
    )


def test_accepted_body_and_key_are_deterministic_for_exact_receipts() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    common = {
        "intent": intent,
        "descriptor": inputs["descriptor"],
        "cache_seed_acceptance": inputs["cache_seed_acceptance"],
        "gpu_spend_snapshot": inputs["gpu_spend_snapshot"],
        **_accepted_authorities(inputs),
        "controller_job": _controller_job(intent),
        **receipt,
    }

    first = module.build_submission_accepted(**common)
    retry = module.build_submission_accepted(**common)

    assert retry == first
    assert first["accepted_body_sha256"] == (
        "5693ad178830321395ac2c06b48df535aff07b5bdd01eee2200bfa89962fa834"
    )
    assert first["job_binding_file_sha256"] == (
        "6a70219aff859f922e950384d20ac279b2addec24f35d9658c437116f74656dd"
    )
    accepted_key = module.submission_accepted_s3_key(
        run_id=RUN_ID,
        managed_mode="qualification",
        accepted_body_sha256=str(first["accepted_body_sha256"]),
    )
    assert accepted_key == (
        "campaigns/glm52-sky-20260723/submissions/qualification/accepted/"
        "5693ad178830321395ac2c06b48df535aff07b5bdd01eee2200bfa89962fa834/"
        "SKYPILOT_SUBMISSION_ACCEPTED.json"
    )
    assert accepted_key == module.submission_accepted_s3_key(
        run_id=RUN_ID,
        managed_mode="qualification",
        accepted_body_sha256=str(retry["accepted_body_sha256"]),
    )


def test_self_hashed_legacy_binding_and_legacy_key_are_rejected() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    legacy = _legacy_qualification_binding(
        intent,
        inputs,
        receipt["controller_observation"],
    )
    receipt.update(
        job_binding=legacy,
        job_binding_key=(
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/JOB_BINDING.json"
        ),
        job_binding_file_sha256=_sha(_canonical(legacy)),
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


@pytest.mark.parametrize("mixed_field", ["target_job_id", "submission_body_sha256"])
def test_dynamic_binding_rejects_mixed_legacy_fields(mixed_field: str) -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    mixed = _rehash(
        {
            **receipt["job_binding"],
            mixed_field: 17 if mixed_field == "target_job_id" else "e" * 64,
        },
        "job_binding_body_sha256",
    )
    receipt.update(
        job_binding=mixed,
        job_binding_file_sha256=_sha(_canonical(mixed)),
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


def test_accepted_schema_rejects_a_parallel_legacy_binding() -> None:
    module, inputs, intent, receipt, accepted = _build_accepted()
    legacy = _legacy_qualification_binding(
        intent,
        inputs,
        receipt["controller_observation"],
    )
    forged = _rehash(
        {**accepted, "legacy_job_binding": legacy},
        "accepted_body_sha256",
    )

    with pytest.raises(module.SubmissionLifecycleValidationError, match="schema"):
        module.validate_submission_accepted(
            forged,
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
        )


def test_foreign_rehashed_controller_baseline_ancestry_is_rejected() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    baseline = _rehash(
        {
            **inputs["controller_baseline"],
            "controller_cluster_name": "foreign-controller-cluster",
        },
        "baseline_body_sha256",
    )
    baseline_key = (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )
    acquisition = _rehash(
        {
            **inputs["submission_acquisition"],
            "controller_baseline_key": baseline_key,
            "controller_baseline_file_sha256": _sha(_canonical(baseline) + b"\n"),
            "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        },
        "acquisition_body_sha256",
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
            controller_baseline=baseline,
            submission_acquisition=acquisition,
            controller_job=_controller_job(intent),
            **receipt,
        )


def test_foreign_rehashed_submission_acquisition_ancestry_is_rejected() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    acquisition = _rehash(
        {
            **inputs["submission_acquisition"],
            "acquired_at": "2026-07-26T12:00:04Z",
        },
        "acquisition_body_sha256",
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
            controller_baseline=inputs["controller_baseline"],
            submission_acquisition=acquisition,
            controller_job=_controller_job(intent),
            **receipt,
        )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("intent_body_sha256", "e" * 64),
        ("sky_job_id", 18),
        ("sky_job_id", True),
        ("workspace", "other"),
        ("controller_instance_id", "i-00000000000000004"),
        ("controller_baseline_body_sha256", "f" * 64),
        ("intent_key", f"campaigns/{RUN_ID}/submissions/foreign.json"),
        ("controller_submitted_at", "2026-07-26T12:00:06Z"),
        ("bound_at", "2026-07-26T12:00:08Z"),
    ],
)
def test_foreign_job_binding_receipts_are_rejected(
    field: str,
    foreign: object,
) -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    job_binding = _rehash(
        {
            **receipt["job_binding"],
            field: foreign,
        },
        "job_binding_body_sha256",
    )
    receipt.update(
        job_binding=job_binding,
        job_binding_file_sha256=_sha(_canonical(job_binding)),
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("controller_observation_key", "campaigns/foreign/key.json"),
        ("controller_observation_file_sha256", "f" * 64),
        ("job_binding_key", "campaigns/foreign/JOB_BINDING.json"),
        ("job_binding_file_sha256", "f" * 64),
    ],
)
def test_controller_receipt_keys_and_file_hashes_are_exact(
    field: str,
    foreign: object,
) -> None:
    module, inputs, intent = _build_intent()
    receipt = {
        **_controller_receipt(intent, inputs),
        field: foreign,
    }

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


@pytest.mark.parametrize(
    ("field", "record_field"),
    [
        ("controller_observation_file_sha256", "controller_observation"),
        ("job_binding_file_sha256", "job_binding"),
    ],
)
def test_controller_receipt_files_require_actual_no_newline_bytes(
    field: str,
    record_field: str,
) -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    actual_sha256 = str(receipt[field])
    newline_sha256 = _sha(_canonical(receipt[record_field]) + b"\n")
    assert actual_sha256 != newline_sha256
    receipt[field] = newline_sha256

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


def test_binding_cannot_follow_controller_observation() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    binding = _rehash(
        {
            **receipt["job_binding"],
            "bound_at": "2026-07-26T12:00:08Z",
        },
        "job_binding_body_sha256",
    )
    receipt.update(
        job_binding=binding,
        job_binding_file_sha256=_sha(_canonical(binding)),
    )

    with pytest.raises(
        module.SubmissionLifecycleValidationError,
        match="observation",
    ):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


def test_binding_may_precede_observation_with_legal_status_progression() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(
        intent,
        inputs,
        observation_status="STARTING",
        binding_status="PENDING",
    )

    accepted = module.build_submission_accepted(
        intent=intent,
        descriptor=inputs["descriptor"],
        cache_seed_acceptance=inputs["cache_seed_acceptance"],
        gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
        **_accepted_authorities(inputs),
        controller_job=_controller_job(intent, controller_status="STARTING"),
        **receipt,
    )

    assert accepted["job_binding"]["bound_at"] == "2026-07-26T12:00:06Z"
    assert accepted["controller_observed_at"] == "2026-07-26T12:00:07Z"
    assert accepted["controller_status"] == "STARTING"
    assert accepted["accepted_at"] == "2026-07-26T12:00:07Z"


def test_binding_status_cannot_regress_at_controller_observation() -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(
        intent,
        inputs,
        observation_status="PENDING",
        binding_status="STARTING",
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


@pytest.mark.parametrize("tamper", ["hash", "unknown"])
def test_controller_observation_hash_or_schema_tamper_is_rejected(
    tamper: str,
) -> None:
    module, inputs, intent = _build_intent()
    receipt = _controller_receipt(intent, inputs)
    observation = dict(receipt["controller_observation"])
    if tamper == "hash":
        observation["status"] = "RUNNING"
    else:
        observation["unknown"] = True
    receipt["controller_observation"] = observation

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_accepted(
            intent=intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_accepted_authorities(inputs),
            controller_job=_controller_job(intent),
            **receipt,
        )


def test_absent_intent_authorizes_only_intent_creation() -> None:
    module = _module()
    inputs = _intent_inputs()

    decision = _decision(module, inputs, intent=None)

    assert decision.action == "create-intent"
    assert decision.controller_job is None
    assert "launch" not in decision.action


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("descriptor", None),
        ("descriptor", 1),
        ("descriptor", True),
        ("descriptor", []),
        ("descriptor", {}),
        ("gpu_spend_snapshot", None),
        ("gpu_spend_snapshot", 1),
        ("gpu_spend_snapshot", True),
        ("gpu_spend_snapshot", []),
        ("gpu_spend_snapshot", {}),
        ("qualification_submission_ready", None),
        ("qualification_submission_ready", 1),
        ("qualification_submission_ready", True),
        ("qualification_submission_ready", []),
        ("qualification_submission_ready", {}),
    ],
)
def test_non_object_or_malformed_submission_source_decision_fails_closed(
    field: str,
    malformed: object,
) -> None:
    module = _module()
    inputs = _intent_inputs()
    inputs[field] = malformed

    decision = _decision(module, inputs, intent=None)

    assert decision.action == "fail-closed"
    assert decision.controller_job is None
    assert "source authority" in decision.reason


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("controller_observation", None),
        ("controller_observation", 1),
        ("controller_observation", True),
        ("controller_observation", []),
        ("controller_observation", {}),
        ("job_binding", None),
        ("job_binding", 1),
        ("job_binding", True),
        ("job_binding", []),
        ("job_binding", {}),
    ],
)
def test_non_object_or_malformed_embedded_receipt_decision_fails_closed(
    field: str,
    malformed: object,
) -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()
    forged = _rehash(
        {
            **accepted,
            field: malformed,
        },
        "accepted_body_sha256",
    )

    decision = _decision(
        module,
        inputs,
        intent=intent,
        accepted=forged,
        now=ACCEPTED_AT,
    )

    assert decision.action == "fail-closed"
    assert decision.controller_job is None
    assert "accepted evidence is invalid" in decision.reason


def test_launch_failure_intent_only_reconciles_without_relaunch() -> None:
    module, inputs, intent = _build_intent()

    first = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(),
        now=INTENT_AT,
    )
    duplicate_retry = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(),
        now=INTENT_AT,
    )

    assert first.action == "reconcile-wait"
    assert duplicate_retry == first
    assert first.controller_job is None
    assert "launch" not in first.action


def test_one_exact_controller_job_is_accepted() -> None:
    module, inputs, intent = _build_intent()
    controller_job = _controller_job(intent)

    decision = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(controller_job,),
        now=ACCEPTED_AT,
    )

    assert decision.action == "accept"
    assert decision.controller_job == controller_job


def test_accepted_retry_is_idempotently_complete() -> None:
    module, inputs, intent, controller_receipt, accepted = _build_accepted()

    first = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        accepted_controller_observation=controller_receipt["controller_observation"],
        now=ACCEPTED_AT,
    )
    duplicate_retry = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        accepted_controller_observation=controller_receipt["controller_observation"],
        now=ACCEPTED_AT,
    )
    observed_retry = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        accepted_controller_observation=controller_receipt["controller_observation"],
        controller_jobs=(_controller_job(intent),),
        now=ACCEPTED_AT,
    )

    assert first.action == "idempotent-complete"
    assert duplicate_retry == first
    assert observed_retry == first


def test_accepted_retry_remains_valid_after_must_start_deadline() -> None:
    module, inputs, intent, controller_receipt, accepted = _build_accepted()

    decision = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        accepted_controller_observation=controller_receipt["controller_observation"],
        now=MUST_START_BY + timedelta(hours=1),
    )

    assert decision.action == "idempotent-complete"


def test_accepted_retry_requires_exact_baseline_and_acquisition_ancestry() -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()

    decision = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        now=MUST_START_BY + timedelta(hours=1),
        include_accepted_ancestry=False,
    )

    assert decision.action == "fail-closed"
    assert "ancestry" in decision.reason


@pytest.mark.parametrize(
    "ancestry_case",
    [
        "missing-baseline",
        "missing-acquisition",
        "foreign-baseline",
        "foreign-acquisition",
    ],
)
def test_accepted_retry_rejects_partial_or_foreign_ancestry(
    ancestry_case: str,
) -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()
    overrides: dict[str, object | None] = {}
    if ancestry_case == "missing-baseline":
        overrides["controller_baseline"] = None
    elif ancestry_case == "missing-acquisition":
        overrides["submission_acquisition"] = None
    elif ancestry_case == "foreign-baseline":
        overrides["controller_baseline"] = _rehash(
            {
                **inputs["controller_baseline"],
                "controller_cluster_name": "foreign-controller-cluster",
            },
            "baseline_body_sha256",
        )
    else:
        overrides["submission_acquisition"] = _rehash(
            {
                **inputs["submission_acquisition"],
                "acquired_at": "2026-07-26T12:00:04Z",
            },
            "acquisition_body_sha256",
        )

    decision = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        now=MUST_START_BY + timedelta(hours=1),
        accepted_ancestry_overrides=overrides,
    )

    assert decision.action == "fail-closed"


@pytest.mark.parametrize(
    "controller_jobs",
    [
        (
            {
                "sky_job_id": 18,
                "sky_job_name": f"{RUN_ID}-qualification",
                "workspace": "default",
                "controller_identity": CONTROLLER_IDENTITY,
                "controller_submitted_at": "2026-07-26T12:00:05Z",
                "controller_status": "PENDING",
            },
        ),
        (
            {
                "sky_job_id": 17,
                "sky_job_name": f"{RUN_ID}-qualification",
                "workspace": "default",
                "controller_identity": CONTROLLER_IDENTITY,
                "controller_submitted_at": "2026-07-26T12:00:05Z",
                "controller_status": "PENDING",
            },
            {
                "sky_job_id": 18,
                "sky_job_name": f"{RUN_ID}-qualification",
                "workspace": "default",
                "controller_identity": CONTROLLER_IDENTITY,
                "controller_submitted_at": "2026-07-26T12:00:06Z",
                "controller_status": "PENDING",
            },
        ),
    ],
)
def test_accepted_retry_with_foreign_or_ambiguous_history_fails_closed(
    controller_jobs: tuple[dict[str, object], ...],
) -> None:
    module, inputs, intent, controller_receipt, accepted = _build_accepted()

    decision = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        accepted_controller_observation=controller_receipt["controller_observation"],
        controller_jobs=controller_jobs,
        now=ACCEPTED_AT,
    )

    assert decision.action == "fail-closed"


def test_accepted_retry_is_self_contained_without_external_receipt() -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()

    decision = _decision(
        module,
        inputs,
        intent=intent,
        accepted=accepted,
        accepted_controller_observation=None,
        now=ACCEPTED_AT,
    )

    assert decision.action == "idempotent-complete"


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("sky_job_id", 0),
        ("sky_job_id", True),
        ("sky_job_id", 17.0),
        ("sky_job_name", "foreign-qualification"),
        ("workspace", "other"),
        (
            "controller_identity",
            "arn:aws:iam::246813579024:role/foreign-controller",
        ),
        ("controller_status", "UNKNOWN"),
        ("controller_submitted_at", "2026-07-26T11:59:59Z"),
        ("controller_submitted_at", "2026-07-26T18:00:00Z"),
    ],
)
def test_foreign_or_type_confused_controller_jobs_fail_closed(
    field: str,
    foreign: object,
) -> None:
    module, inputs, intent = _build_intent()
    controller_job = {
        **_controller_job(intent),
        field: foreign,
    }

    decision = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(controller_job,),
        now=ACCEPTED_AT,
    )

    assert decision.action == "fail-closed"
    assert decision.controller_job is None


def test_ambiguous_controller_history_fails_closed() -> None:
    module, inputs, intent = _build_intent()

    decision = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(
            _controller_job(intent, sky_job_id=17),
            _controller_job(intent, sky_job_id=18),
        ),
        now=ACCEPTED_AT,
    )

    assert decision.action == "fail-closed"
    assert "ambiguous" in decision.reason


def test_intent_only_after_deadline_fails_closed() -> None:
    module, inputs, intent = _build_intent()

    decision = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(),
        now=MUST_START_BY,
    )

    assert decision.action == "fail-closed"
    assert "expired" in decision.reason or "deadline" in decision.reason


def test_budget_drift_after_intent_fails_closed() -> None:
    module, inputs, intent = _build_intent()
    snapshot = dict(inputs["gpu_spend_snapshot"])
    snapshot.update(
        consumed_gpu_seconds=14_400,
        remaining_gpu_seconds=72_000,
        consumed_gpu_cost_usd=220.16,
        remaining_gpu_cost_usd=1_100.80,
    )
    inputs["gpu_spend_snapshot"] = _rehash(snapshot, "snapshot_body_sha256")

    decision = _decision(
        module,
        inputs,
        intent=intent,
        controller_jobs=(),
        now=ACCEPTED_AT,
    )

    assert decision.action == "fail-closed"
    assert "snapshot" in decision.reason or "intent" in decision.reason


@pytest.mark.parametrize(
    ("kind", "field", "foreign"),
    [
        ("intent", "schema_version", True),
        ("intent", "schema_version", 1),
        ("intent", "record_type", "glm52_skypilot_submission_v1"),
        ("intent", "remaining_gpu_seconds", True),
        ("intent", "intent_at", "2026-07-26T12:00:00+00:00"),
        ("accepted", "schema_version", True),
        ("accepted", "schema_version", 1),
        ("accepted", "record_type", "glm52_skypilot_submission_v1"),
        ("accepted", "sky_job_id", True),
        ("accepted", "sky_job_id", 18),
        ("accepted", "sky_job_name", "foreign-qualification"),
        ("accepted", "workspace", "other"),
        (
            "accepted",
            "controller_identity",
            "arn:aws:iam::246813579024:role/foreign-controller",
        ),
        ("accepted", "controller_instance_id", "i-00000000000000004"),
        (
            "accepted",
            "controller_profile_arn",
            "arn:aws:iam::246813579024:instance-profile/foreign-controller",
        ),
        (
            "accepted",
            "controller_cluster_name",
            "sky-jobs-controller-foreign",
        ),
        ("accepted", "controller_status", "UNKNOWN"),
        (
            "accepted",
            "controller_observation_body_sha256",
            "f" * 64,
        ),
        ("accepted", "accepted_at", "2026-07-26T12:00:10+00:00"),
    ],
)
def test_rehashed_type_legacy_or_authority_confusion_is_rejected(
    kind: str,
    field: str,
    foreign: object,
) -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()
    if kind == "intent":
        forged = _rehash(
            {**intent, field: foreign},
            "intent_body_sha256",
        )
        with pytest.raises(module.SubmissionLifecycleValidationError):
            module.validate_submission_intent(
                forged,
                descriptor=inputs["descriptor"],
                cache_seed_acceptance=inputs["cache_seed_acceptance"],
                gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
                **_readiness_objects(inputs),
            )
    else:
        forged = _rehash(
            {**accepted, field: foreign},
            "accepted_body_sha256",
        )
        with pytest.raises(module.SubmissionLifecycleValidationError):
            module.validate_submission_accepted(
                forged,
                intent=intent,
                descriptor=inputs["descriptor"],
                cache_seed_acceptance=inputs["cache_seed_acceptance"],
                gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
                **_accepted_authorities(inputs),
            )


@pytest.mark.parametrize("kind", ["intent", "accepted"])
def test_unknown_fields_are_rejected(kind: str) -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()
    if kind == "intent":
        with pytest.raises(module.SubmissionLifecycleValidationError, match="schema"):
            module.validate_submission_intent(
                {**intent, "unknown": True},
                descriptor=inputs["descriptor"],
                cache_seed_acceptance=inputs["cache_seed_acceptance"],
                gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
                **_readiness_objects(inputs),
            )
    else:
        with pytest.raises(module.SubmissionLifecycleValidationError, match="schema"):
            module.validate_submission_accepted(
                {**accepted, "unknown": True},
                intent=intent,
                descriptor=inputs["descriptor"],
                cache_seed_acceptance=inputs["cache_seed_acceptance"],
                gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
                **_accepted_authorities(inputs),
            )


@pytest.mark.parametrize("kind", ["intent", "accepted"])
def test_hash_tamper_is_rejected(kind: str) -> None:
    module, inputs, intent, _controller_receipt_value, accepted = _build_accepted()
    if kind == "intent":
        with pytest.raises(module.SubmissionLifecycleValidationError, match="SHA-256"):
            module.validate_submission_intent(
                {**intent, "sky_job_name": "tampered"},
                descriptor=inputs["descriptor"],
                cache_seed_acceptance=inputs["cache_seed_acceptance"],
                gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
                **_readiness_objects(inputs),
            )
    else:
        with pytest.raises(module.SubmissionLifecycleValidationError, match="SHA-256"):
            module.validate_submission_accepted(
                {**accepted, "controller_status": "RUNNING"},
                intent=intent,
                descriptor=inputs["descriptor"],
                cache_seed_acceptance=inputs["cache_seed_acceptance"],
                gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
                **_accepted_authorities(inputs),
            )


@pytest.mark.parametrize(
    "field",
    [
        "cache_seed_acceptance_file_sha256",
        "gpu_spend_snapshot_sha256",
        "qualification_submission_ready_sha256",
    ],
)
def test_rehashed_source_file_sha_lies_are_rejected(field: str) -> None:
    module, inputs, intent = _build_intent()
    forged = _rehash({**intent, field: "f" * 64}, "intent_body_sha256")

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.validate_submission_intent(
            forged,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
        )


def test_coherently_rehashed_readiness_lie_is_rejected() -> None:
    module, inputs, intent = _build_intent()
    forged_ready = _rehash(
        {
            **inputs["qualification_submission_ready"],
            "sky_job_name": f"{RUN_ID}-production",
        },
        "readiness_body_sha256",
    )
    forged_body_sha256 = str(forged_ready["readiness_body_sha256"])
    forged = _rehash(
        {
            **intent,
            "qualification_submission_ready_key": (
                f"campaigns/{RUN_ID}/qualification/submission-ready/"
                f"{forged_body_sha256}/QUALIFICATION_SUBMISSION_READY.json"
            ),
            "qualification_submission_ready_sha256": _sha(
                _canonical(forged_ready) + b"\n"
            ),
            "qualification_submission_ready_body_sha256": (forged_body_sha256),
        },
        "intent_body_sha256",
    )
    inputs["qualification_submission_ready"] = forged_ready

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.validate_submission_intent(
            forged,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
        )


@pytest.mark.parametrize("source", ["staged", "rehearsal"])
def test_semantically_foreign_readiness_sources_are_rejected(source: str) -> None:
    module, inputs, intent = _build_intent()
    if source == "staged":
        staged = _rehash(
            {
                **inputs["staged_readiness"],
                "run_id": "foreign-run",
            },
            "ready_body_sha256",
        )
        inputs["staged_readiness"] = staged
    else:
        rehearsal = _rehash(
            {
                **inputs["rehearsal_evidence"],
                "descriptor_file_sha256": "f" * 64,
            },
            "rehearsal_body_sha256",
        )
        inputs["rehearsal_evidence"] = rehearsal

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.validate_submission_intent(
            intent,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=inputs["gpu_spend_snapshot"],
            **_readiness_objects(inputs),
        )


def test_coordinated_descriptor_file_sha_lie_is_rejected() -> None:
    module, inputs, intent = _build_intent()
    forged_descriptor_file_sha256 = "f" * 64
    snapshot = _rehash(
        {
            **inputs["gpu_spend_snapshot"],
            "descriptor_sha256": forged_descriptor_file_sha256,
        },
        "snapshot_body_sha256",
    )
    forged_snapshot_body_sha256 = str(snapshot["snapshot_body_sha256"])
    forged_snapshot_file_sha256 = _sha(_canonical(snapshot) + b"\n")
    forged = _rehash(
        {
            **intent,
            "descriptor_file_sha256": forged_descriptor_file_sha256,
            "gpu_spend_snapshot_key": (
                f"campaigns/{RUN_ID}/spend-snapshots/"
                f"{forged_snapshot_body_sha256}/GPU_SPEND_SNAPSHOT.json"
            ),
            "gpu_spend_snapshot_sha256": forged_snapshot_file_sha256,
            "gpu_spend_snapshot_body_sha256": forged_snapshot_body_sha256,
        },
        "intent_body_sha256",
    )

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.validate_submission_intent(
            forged,
            descriptor=inputs["descriptor"],
            cache_seed_acceptance=inputs["cache_seed_acceptance"],
            gpu_spend_snapshot=snapshot,
            **_readiness_objects(inputs),
        )


def test_stale_intent_and_zero_qualification_allowance_are_rejected() -> None:
    module = _module()
    stale = _intent_inputs()
    stale["intent_at"] = MUST_START_BY
    with pytest.raises(module.SubmissionLifecycleValidationError, match="stale"):
        module.build_submission_intent(**stale)

    exhausted = _intent_inputs()
    snapshot = dict(exhausted["gpu_spend_snapshot"])
    snapshot.update(
        consumed_gpu_seconds=86_400,
        remaining_gpu_seconds=0,
        consumed_gpu_cost_usd=1_320.96,
        remaining_gpu_cost_usd=0.0,
        qualification_allowance_seconds=0,
        qualification_allowance_cost_usd=0.0,
    )
    exhausted_snapshot = _rehash(snapshot, "snapshot_body_sha256")
    exhausted["gpu_spend_snapshot"] = exhausted_snapshot
    exhausted["gpu_spend_snapshot_key"] = (
        f"campaigns/{RUN_ID}/spend-snapshots/"
        f"{exhausted_snapshot['snapshot_body_sha256']}/"
        "GPU_SPEND_SNAPSHOT.json"
    )
    exhausted["gpu_spend_snapshot_sha256"] = _sha(
        _canonical(exhausted_snapshot) + b"\n"
    )
    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_intent(**exhausted)


def test_every_mode_has_one_exact_sky_job_name() -> None:
    module = _module()

    assert module.expected_sky_job_name(RUN_ID, "production") == RUN_ID
    assert (
        module.expected_sky_job_name(RUN_ID, "qualification")
        == f"{RUN_ID}-qualification"
    )
    assert module.expected_sky_job_name(RUN_ID, "cache-seed") == f"{RUN_ID}-cache-seed"
    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.expected_sky_job_name(RUN_ID, "training")

    inputs = _intent_inputs()
    inputs["sky_job_name"] = RUN_ID
    with pytest.raises(
        module.SubmissionLifecycleValidationError,
        match="managed mode",
    ):
        module.build_submission_intent(**inputs)


def test_lifecycle_artifact_keys_are_exact_and_content_addressed() -> None:
    module = _module()

    assert module.submission_intent_s3_key(
        run_id=RUN_ID,
        managed_mode="qualification",
        intent_body_sha256="a" * 64,
    ) == (
        f"campaigns/{RUN_ID}/submissions/qualification/intents/"
        f"{'a' * 64}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    assert module.submission_accepted_s3_key(
        run_id=RUN_ID,
        managed_mode="qualification",
        accepted_body_sha256="b" * 64,
    ) == (
        f"campaigns/{RUN_ID}/submissions/qualification/accepted/"
        f"{'b' * 64}/SKYPILOT_SUBMISSION_ACCEPTED.json"
    )
    for managed_mode in ("production", "qualification", "cache-seed"):
        assert f"/{managed_mode}/" in module.submission_intent_s3_key(
            run_id=RUN_ID,
            managed_mode=managed_mode,
            intent_body_sha256="c" * 64,
        )
    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.submission_intent_s3_key(
            run_id=RUN_ID,
            managed_mode="foreign",
            intent_body_sha256="c" * 64,
        )
    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.submission_accepted_s3_key(
            run_id=RUN_ID,
            managed_mode="qualification",
            accepted_body_sha256="NOT-A-SHA",
        )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("account_id", "135792468013"),
        ("provider", "gcp"),
        ("region", "us-east-1"),
        ("instance_type", "p4d.24xlarge"),
        ("instance_count", 2),
        ("instance_count", True),
        ("use_spot", True),
    ],
)
def test_descriptor_must_remain_one_on_demand_p5_in_us_west_2(
    field: str,
    foreign: object,
) -> None:
    module = _module()
    inputs = _intent_inputs()
    descriptor = dict(inputs["descriptor"])
    descriptor[field] = foreign
    body = dict(descriptor)
    body.pop("descriptor_body_sha256")
    descriptor["descriptor_body_sha256"] = _body_sha(body)
    inputs["descriptor"] = descriptor

    with pytest.raises(module.SubmissionLifecycleValidationError):
        module.build_submission_intent(**inputs)
