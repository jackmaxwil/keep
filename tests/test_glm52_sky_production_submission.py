"""Pure production-submission intent and exact-source authentication tests."""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pytest

from mlx_vq.quality.glm52_h100_qualification import (
    build_h100_resume_ready,
    validate_h100_resume_ready,
    validate_h100_runtime_allocation,
)
from mlx_vq.quality.glm52_qualification_cache_seed import (
    accepted_s3_key,
    validate_qualification_cache_seed_accepted,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_sky_campaign_descriptor,
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_production_submission import (
    ProductionSubmissionError,
    VersionedJsonArtifact,
    build_production_submission_intent,
    canonical_file_bytes,
    production_submission_intent_file_bytes,
    production_submission_intent_file_sha256,
    production_submission_intent_s3_key,
    validate_production_submission_intent,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
SEED_ACCEPTED_AT = datetime(2026, 7, 26, 12, 1, tzinfo=timezone.utc)
STAGED_AT = datetime(2026, 7, 26, 12, 2, tzinfo=timezone.utc)
REHEARSAL_AT = datetime(2026, 7, 26, 12, 4, tzinfo=timezone.utc)
FIRST_LAUNCHED_AT = datetime(2026, 7, 26, 12, 4, tzinfo=timezone.utc)
FIRST_OBSERVED_AT = datetime(2026, 7, 26, 12, 5, tzinfo=timezone.utc)
REPLACEMENT_LAUNCHED_AT = datetime(2026, 7, 26, 12, 6, tzinfo=timezone.utc)
REPLACEMENT_OBSERVED_AT = datetime(2026, 7, 26, 12, 7, tzinfo=timezone.utc)
H100_COMPLETED_AT = datetime(2026, 7, 26, 12, 9, tzinfo=timezone.utc)
SNAPSHOT_AT = datetime(2026, 7, 26, 12, 10, tzinfo=timezone.utc)
INTENT_AT = datetime(2026, 7, 26, 12, 10, 30, tzinfo=timezone.utc)
MUST_START_BY = datetime(2026, 7, 26, 18, tzinfo=timezone.utc)
SOURCE_INSTANCE_ID = "i-11111111111111111"
REPLACEMENT_INSTANCE_ID = "i-22222222222222222"
EXTRA_CLOSED_INSTANCE_ID = "i-33333333333333333"

INTENT_FIELDS = {
    "schema_version",
    "record_type",
    "managed_mode",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "sky_job_name",
    "must_start_by",
    "intent_at",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "repo_tar_key",
    "repo_tar_sha256",
    "approval_key",
    "approval_file_sha256",
    "approval_body_sha256",
    "approval_version_id",
    "staged_readiness_key",
    "staged_readiness_file_sha256",
    "staged_readiness_body_sha256",
    "staged_readiness_version_id",
    "staged_at",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "rehearsal_evidence_key",
    "rehearsal_evidence_file_sha256",
    "rehearsal_evidence_body_sha256",
    "rehearsal_evidence_version_id",
    "rehearsal_completed_at",
    "cache_seed_acceptance_key",
    "cache_seed_acceptance_file_sha256",
    "cache_seed_acceptance_body_sha256",
    "cache_seed_acceptance_version_id",
    "qualification_cache_prefix",
    "qualification_cache_manifest_sha256",
    "cache_seed_instance_id",
    "cache_seed_accepted_at",
    "h100_resume_ready_key",
    "h100_resume_ready_file_sha256",
    "h100_resume_ready_body_sha256",
    "h100_resume_ready_version_id",
    "h100_first_instance_id",
    "h100_replacement_instance_id",
    "h100_completed_at",
    "first_h100_allocation_key",
    "first_h100_allocation_file_sha256",
    "first_h100_allocation_body_sha256",
    "first_h100_allocation_record_sha256",
    "first_h100_allocation_version_id",
    "replacement_h100_allocation_key",
    "replacement_h100_allocation_file_sha256",
    "replacement_h100_allocation_body_sha256",
    "replacement_h100_allocation_record_sha256",
    "replacement_h100_allocation_version_id",
    "gpu_spend_snapshot_key",
    "gpu_spend_snapshot_file_sha256",
    "gpu_spend_snapshot_body_sha256",
    "gpu_spend_snapshot_version_id",
    "gpu_spend_ledger_tip_record_sha256",
    "gpu_spend_ledger_genesis_sha256",
    "ec2_allocation_history_sha256",
    "closed_instance_ids",
    "open_allocation_count",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "spend_snapshot_observed_at",
    "intent_body_sha256",
}
EXPECTED_INTENT_BODY_SHA256 = (
    "36adb903adfdc3fe3790c417dd1da7f3cee2a7d00cbc60933c0989bcbc1503a5"
)
EXPECTED_INTENT_FILE_SHA256 = (
    "7b743070e75ce427136860ae313a6f17d6cc1bc4fedf023eda833feaa76f81ab"
)
EXPECTED_INTENT_FILE_SIZE = 6075


class DictSubclass(dict[str, object]):
    pass


class ListSubclass(list[object]):
    pass


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


class BytesSubclass(bytes):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hashed(
    body: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    return {**body, digest_field: _sha(_canonical(body))}


def _rehash(
    value: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(digest_field)
    return _self_hashed(body, digest_field)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _money_for_seconds(seconds: int, hourly: float) -> float:
    return float(
        (Decimal(seconds) * Decimal(str(hourly)) / Decimal(3600)).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def _load_cache_fixture_module() -> Any:
    path = ROOT / "tests/test_glm52_qualification_cache_seed.py"
    specification = importlib.util.spec_from_file_location(
        "_task3ph1c_cache_seed_fixture",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _allocation(
    *,
    instance_id: str,
    launched_at: datetime,
    observed_at: datetime,
    approval_file_sha256: str,
    spend_genesis_sha256: str,
    record_sha256: str,
    ledger_sha256: str,
    remaining_gpu_seconds: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "record_type": "glm52_gpu_runtime_allocation_v1",
        "run_id": RUN_ID,
        "job_id": f"{RUN_ID}-qualification",
        "instance_id": instance_id,
        "launched_at": _iso(launched_at),
        "observed_at": _iso(observed_at),
        "execution_deadline": _iso(
            observed_at + timedelta(seconds=remaining_gpu_seconds)
        ),
        "approval_sha256": approval_file_sha256,
        "gpu_spend_authority_sha256": spend_genesis_sha256,
        "gpu_spend_record_sha256": record_sha256,
        "gpu_spend_ledger_sha256": ledger_sha256,
        "remaining_gpu_seconds": remaining_gpu_seconds,
        "estimated_gpu_cost_usd": 1.25,
    }
    allocation = _self_hashed(body, "allocation_body_sha256")
    assert validate_h100_runtime_allocation(allocation) == allocation
    return allocation


def _stabilize_cache_acceptance(
    value: dict[str, object],
) -> dict[str, object]:
    acceptance = copy.deepcopy(value)
    audit: Any = acceptance["cache_audit"]
    manifest_file_sha = "a1" * 32
    ready_file_sha = "b2" * 32
    prompt_file_sha = "c3" * 32
    shard_file_sha = "d4" * 32
    prefix = f"qualification-cache/seeds/{RUN_ID}/{manifest_file_sha}/"
    audit.update(
        {
            "cache_prefix": prefix,
            "legacy_seed_ready_file_sha256": "07" * 32,
            "legacy_seed_ready_body_sha256": "18" * 32,
            "manifest_key": (f"{prefix}glm52-teacher-signal-cache-v3-manifest.json"),
            "manifest_file_sha256": manifest_file_sha,
            "manifest_body_sha256": "e5" * 32,
            "teacher_cache_ready_key": f"{prefix}TEACHER_CACHE_READY.json",
            "teacher_cache_ready_file_sha256": ready_file_sha,
            "teacher_cache_ready_body_sha256": "f6" * 32,
            "prompt_pack_key": f"{prefix}prompt-pack.json",
            "prompt_pack_file_sha256": prompt_file_sha,
        }
    )
    rows = [
        {
            "key": audit["teacher_cache_ready_key"],
            "size": 517,
            "sha256": ready_file_sha,
            "checksum_type": "FULL_OBJECT",
            "etag": f'"{ready_file_sha[:32]}"',
            "version_id": "fixture-version",
        },
        {
            "key": audit["manifest_key"],
            "size": 2476,
            "sha256": manifest_file_sha,
            "checksum_type": "FULL_OBJECT",
            "etag": f'"{manifest_file_sha[:32]}"',
            "version_id": "fixture-version",
        },
        {
            "key": audit["prompt_pack_key"],
            "size": 391,
            "sha256": prompt_file_sha,
            "checksum_type": "FULL_OBJECT",
            "etag": f'"{prompt_file_sha[:32]}"',
            "version_id": "fixture-version",
        },
        {
            "key": f"{prefix}teacher_signal/row-00000-fixture.safetensors",
            "size": 25870,
            "sha256": shard_file_sha,
            "checksum_type": "FULL_OBJECT",
            "etag": f'"{shard_file_sha[:32]}"',
            "version_id": "fixture-version",
        },
    ]
    rows.sort(key=lambda row: str(row["key"]))
    acceptance["cache_object_inventory"] = rows
    acceptance["cache_object_inventory_sha256"] = _sha(_canonical(rows))
    return _rehash(acceptance, "acceptance_body_sha256")


@pytest.fixture(scope="module")
def source_chain(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    cache_tests = _load_cache_fixture_module()
    client, pins, acceptance, _keys = cache_tests._accepted_fixture(
        tmp_path_factory.mktemp("task3ph1c-cache-seed")
    )
    acceptance = _stabilize_cache_acceptance(acceptance)
    seed_descriptor_raw = client.objects[pins.descriptor_key]
    seed_descriptor = json.loads(seed_descriptor_raw)
    approval_key = str(acceptance["seed_authority"]["approval_key"])
    approval_raw = client.objects[approval_key]
    approval = json.loads(approval_raw)
    assert validate_sky_campaign_descriptor(seed_descriptor) == seed_descriptor
    assert validate_gpu_spend_approval(approval) == approval
    assert validate_qualification_cache_seed_accepted(acceptance) == acceptance
    assert acceptance["accepted_at"] == _iso(SEED_ACCEPTED_AT)

    artifacts = copy.deepcopy(seed_descriptor["artifacts"])
    artifacts.update(
        {
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'a' * 64}.json"
            ),
            "artifact_inventory_sha256": "a" * 64,
            "qualification_cache_prefix": acceptance["cache_audit"]["cache_prefix"],
            "qualification_cache_manifest_sha256": acceptance["cache_audit"][
                "manifest_file_sha256"
            ],
        }
    )
    descriptor_key = (
        f"campaigns/{RUN_ID}/submissions/post-seed/campaign-descriptor-v2.json"
    )
    descriptor = build_sky_campaign_descriptor(
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
        campaign_descriptor_key=descriptor_key,
        approval_key=approval_key,
        approval_sha256=_sha(approval_raw),
        artifacts=artifacts,
    )
    descriptor_raw = _json_bytes(descriptor)
    descriptor_file_sha256 = _sha(descriptor_raw)

    manifest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_campaign_bundle_v1",
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "files": [
            {
                "local_name": "repository.tar.gz",
                "key": descriptor["repo_tar_key"],
                "role": "repository_tar",
                "stage_order": 10,
                "size": 101,
                "sha256": descriptor["repo_tar_sha256"],
            },
            {
                "local_name": "approval.json",
                "key": descriptor["approval_key"],
                "role": "approval",
                "stage_order": 20,
                "size": len(approval_raw),
                "sha256": _sha(approval_raw),
            },
            {
                "local_name": "training.json",
                "key": descriptor["artifacts"]["training_config_key"],
                "role": "training_config",
                "stage_order": 30,
                "size": 303,
                "sha256": descriptor["artifacts"]["training_config_sha256"],
            },
            {
                "local_name": "watchdog.py",
                "key": f"campaigns/{RUN_ID}/authorities/watchdog.py",
                "role": "watchdog",
                "stage_order": 40,
                "size": 404,
                "sha256": "9" * 64,
            },
            {
                "local_name": "artifact-inventory.json",
                "key": descriptor["artifacts"]["artifact_inventory_key"],
                "role": "artifact_inventory",
                "stage_order": 50,
                "size": 505,
                "sha256": descriptor["artifacts"]["artifact_inventory_sha256"],
            },
            {
                "local_name": "campaign-descriptor-v2.json",
                "key": descriptor_key,
                "role": "descriptor",
                "stage_order": 100,
                "size": len(descriptor_raw),
                "sha256": descriptor_file_sha256,
            },
        ],
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    }
    manifest = _self_hashed(manifest_body, "bundle_manifest_body_sha256")
    manifest_raw = _json_bytes(manifest)
    manifest_key = (
        f"campaigns/{RUN_ID}/submissions/post-seed/bundle-manifests/"
        f"{manifest['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )

    staged_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": RUN_ID,
        "descriptor_key": descriptor_key,
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": manifest_key,
        "bundle_manifest_file_sha256": _sha(manifest_raw),
        "bundle_manifest_body_sha256": manifest["bundle_manifest_body_sha256"],
        "bundle_manifest_version_id": "manifest-version-1",
        "staged_object_version_ids": {
            "repository_tar": "repository-version-1",
            "approval": "approval-version-1",
            "training_config": "training-version-1",
            "watchdog": "watchdog-version-1",
            "artifact_inventory": "inventory-version-1",
            "artifact_audit": "audit-version-1",
            "descriptor": "descriptor-version-1",
        },
        "artifact_audit_key": (
            f"campaigns/{RUN_ID}/audits/artifact-audit-{'c' * 64}.json"
        ),
        "artifact_audit_sha256": "c" * 64,
        "staged_at": _iso(STAGED_AT),
    }
    staged = _self_hashed(staged_body, "ready_body_sha256")
    staged_raw = _json_bytes(staged)
    staged_key = (
        f"campaigns/{RUN_ID}/submissions/post-seed/STAGED_CONTROL_PLANE_READY.json"
    )

    rehearsal_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_rehearsal_v2",
        "status": "passed_before_cuda_h100_boundary",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "bootstrap_receipt_file_sha256": "0" * 64,
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": _sha(staged_raw),
        "staged_readiness_body_sha256": staged["ready_body_sha256"],
        "artifact_inventory_key": descriptor["artifacts"]["artifact_inventory_key"],
        "artifact_inventory_file_sha256": descriptor["artifacts"][
            "artifact_inventory_sha256"
        ],
        "artifact_inventory_body_sha256": "1" * 64,
        "artifact_audit_key": staged["artifact_audit_key"],
        "artifact_audit_file_sha256": staged["artifact_audit_sha256"],
        "extracted_repo_path": "/tmp/task3ph1c-rehearsal/repo",
        "production_repo_path": "/opt/keep-campaign/repo",
        "production_resume_root": "/mnt/nvme/glm52-campaign",
        "skypilot_task_path": (
            "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ),
        "skypilot_task_file_sha256": "2" * 64,
        "skypilot_config_file_sha256": "3" * 64,
        "skypilot_validation_file_sha256": "4" * 64,
        "skypilot_version": "0.13.0",
        "skypilot_task_name": "glm52-campaign",
        "completed_at": _iso(REHEARSAL_AT),
    }
    rehearsal = _self_hashed(rehearsal_body, "rehearsal_body_sha256")
    rehearsal_raw = _json_bytes(rehearsal)
    rehearsal_key = (
        f"campaigns/{RUN_ID}/qualification/rehearsals/"
        f"{rehearsal['rehearsal_body_sha256']}/"
        "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    )

    seed_instance_id = str(acceptance["spend_closure"]["instance_id"])
    spend_genesis_sha256 = "f" * 64
    first_record_sha256 = "4" * 64
    replacement_record_sha256 = "5" * 64
    first_allocation = _allocation(
        instance_id=SOURCE_INSTANCE_ID,
        launched_at=FIRST_LAUNCHED_AT,
        observed_at=FIRST_OBSERVED_AT,
        approval_file_sha256=_sha(approval_raw),
        spend_genesis_sha256=spend_genesis_sha256,
        record_sha256=first_record_sha256,
        ledger_sha256="6" * 64,
        remaining_gpu_seconds=80_000,
    )
    replacement_allocation = _allocation(
        instance_id=REPLACEMENT_INSTANCE_ID,
        launched_at=REPLACEMENT_LAUNCHED_AT,
        observed_at=REPLACEMENT_OBSERVED_AT,
        approval_file_sha256=_sha(approval_raw),
        spend_genesis_sha256=spend_genesis_sha256,
        record_sha256=replacement_record_sha256,
        ledger_sha256="7" * 64,
        remaining_gpu_seconds=79_000,
    )
    first_allocation_raw = _json_bytes(first_allocation)
    replacement_allocation_raw = _json_bytes(replacement_allocation)

    h100_ready = build_h100_resume_ready(
        run_id=RUN_ID,
        campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        repo_tar_sha256=str(descriptor["repo_tar_sha256"]),
        qualification_cache_manifest_sha256=str(
            descriptor["artifacts"]["qualification_cache_manifest_sha256"]
        ),
        first_instance_id=SOURCE_INSTANCE_ID,
        replacement_instance_id=REPLACEMENT_INSTANCE_ID,
        first_allocation_record_sha256=first_record_sha256,
        replacement_allocation_record_sha256=replacement_record_sha256,
        source_checkpoint_marker_sha256="8" * 64,
        parity_report_sha256="9" * 64,
        training_smoke_sha256="a" * 64,
        resumed_capture_sha256="b" * 64,
        peak_gpu_gib=69.5,
        completed_at=H100_COMPLETED_AT,
    )
    assert validate_h100_resume_ready(h100_ready) == h100_ready
    h100_ready_raw = _json_bytes(h100_ready)

    spend = acceptance["spend_closure"]
    consumed_seconds = int(spend["consumed_gpu_seconds"])
    remaining_seconds = 86_400 - consumed_seconds
    consumed_cost = _money_for_seconds(consumed_seconds, 55.04)
    remaining_cost = float(
        (Decimal("1320.96") - Decimal(str(consumed_cost))).quantize(Decimal("0.01"))
    )
    closed_ids = [
        seed_instance_id,
        SOURCE_INSTANCE_ID,
        EXTRA_CLOSED_INSTANCE_ID,
        REPLACEMENT_INSTANCE_ID,
    ]
    snapshot_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "approval_sha256": _sha(approval_raw),
        "approval_body_sha256": approval["approval_body_sha256"],
        "gpu_spend_ledger_latest_sha256": "d" * 64,
        "gpu_spend_ledger_latest_body_sha256": "e" * 64,
        "gpu_spend_ledger_genesis_sha256": spend_genesis_sha256,
        "gpu_spend_ledger_record_count": len(closed_ids) * 2,
        "gpu_spend_ledger_tip_record_sha256": "1" * 64,
        "gpu_spend_ledger_file_sha256": "2" * 64,
        "ec2_allocation_history_sha256": "3" * 64,
        "ec2_allocation_instance_ids": closed_ids,
        "observed_at": _iso(SNAPSHOT_AT),
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "hourly_cost_usd": 55.04,
        "consumed_gpu_seconds": consumed_seconds,
        "remaining_gpu_seconds": remaining_seconds,
        "consumed_gpu_cost_usd": consumed_cost,
        "remaining_gpu_cost_usd": remaining_cost,
        "qualification_allowance_seconds": min(14_400, remaining_seconds),
        "qualification_allowance_cost_usd": _money_for_seconds(
            min(14_400, remaining_seconds),
            55.04,
        ),
        "open_allocation_count": 0,
    }
    snapshot = _self_hashed(snapshot_body, "snapshot_body_sha256")
    snapshot_raw = _json_bytes(snapshot)

    acceptance_raw = _json_bytes(acceptance)
    acceptance_key = accepted_s3_key(
        run_id=RUN_ID,
        acceptance_body_sha256=str(acceptance["acceptance_body_sha256"]),
    )
    snapshot_key = (
        f"campaigns/{RUN_ID}/spend-snapshots/"
        f"{snapshot['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
    )
    allocation_key = f"campaigns/{RUN_ID}/runtime/GPU_RUNTIME_ALLOCATION.json"
    artifacts_by_name = {
        "descriptor": VersionedJsonArtifact(
            key=descriptor_key,
            raw=descriptor_raw,
            version_id="descriptor-version-1",
        ),
        "approval": VersionedJsonArtifact(
            key=approval_key,
            raw=approval_raw,
            version_id="approval-version-1",
        ),
        "staged_readiness": VersionedJsonArtifact(
            key=staged_key,
            raw=staged_raw,
            version_id="staged-version-1",
        ),
        "bundle_manifest": VersionedJsonArtifact(
            key=manifest_key,
            raw=manifest_raw,
            version_id="manifest-version-1",
        ),
        "rehearsal_evidence": VersionedJsonArtifact(
            key=rehearsal_key,
            raw=rehearsal_raw,
            version_id="rehearsal-version-1",
        ),
        "cache_seed_acceptance": VersionedJsonArtifact(
            key=acceptance_key,
            raw=acceptance_raw,
            version_id="cache-acceptance-version-1",
        ),
        "h100_resume_ready": VersionedJsonArtifact(
            key=f"campaigns/{RUN_ID}/qualification/H100_RESUME_READY.json",
            raw=h100_ready_raw,
            version_id="h100-ready-version-1",
        ),
        "first_h100_runtime_allocation": VersionedJsonArtifact(
            key=allocation_key,
            raw=first_allocation_raw,
            version_id="allocation-version-source",
        ),
        "replacement_h100_runtime_allocation": VersionedJsonArtifact(
            key=allocation_key,
            raw=replacement_allocation_raw,
            version_id="allocation-version-replacement",
        ),
        "closed_gpu_spend_snapshot": VersionedJsonArtifact(
            key=snapshot_key,
            raw=snapshot_raw,
            version_id="snapshot-version-1",
        ),
    }
    return {
        "artifacts": artifacts_by_name,
        "records": {
            "descriptor": descriptor,
            "approval": approval,
            "staged_readiness": staged,
            "bundle_manifest": manifest,
            "rehearsal_evidence": rehearsal,
            "cache_seed_acceptance": acceptance,
            "h100_resume_ready": h100_ready,
            "first_h100_runtime_allocation": first_allocation,
            "replacement_h100_runtime_allocation": replacement_allocation,
            "closed_gpu_spend_snapshot": snapshot,
        },
        "seed_descriptor": seed_descriptor,
        "closed_ids": closed_ids,
        "intent_at": INTENT_AT,
    }


def _kwargs(chain: dict[str, object]) -> dict[str, object]:
    return {
        **copy.deepcopy(chain["artifacts"]),
        "intent_at": chain["intent_at"],
    }


def _build(chain: dict[str, object], **overrides: object) -> dict[str, object]:
    kwargs = _kwargs(chain)
    kwargs.update(overrides)
    return build_production_submission_intent(**kwargs)  # type: ignore[arg-type]


def _artifact_with_record(
    artifact: VersionedJsonArtifact,
    record: dict[str, object],
) -> VersionedJsonArtifact:
    return replace(artifact, raw=_json_bytes(record))


def _rehash_source_record(
    name: str,
    record: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    if name == "descriptor":
        identity = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "campaign_identity_sha256",
                "descriptor_body_sha256",
                "must_start_by",
                "campaign_descriptor_key",
            }
        }
        record["campaign_identity_sha256"] = _sha(_canonical(identity))
    return _rehash(record, digest_field)


def _artifact_for_source_record(
    chain: dict[str, object],
    name: str,
    record: dict[str, object],
    digest_field: str,
) -> VersionedJsonArtifact:
    artifact = _artifact_with_record(chain["artifacts"][name], record)
    digest = record[digest_field]
    if name == "bundle_manifest":
        artifact = replace(
            artifact,
            key=(
                f"campaigns/{RUN_ID}/submissions/post-seed/bundle-manifests/"
                f"{digest}/bundle-manifest-v1.json"
            ),
        )
    elif name == "rehearsal_evidence":
        artifact = replace(
            artifact,
            key=(
                f"campaigns/{RUN_ID}/qualification/rehearsals/{digest}/"
                "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
            ),
        )
    elif name == "cache_seed_acceptance":
        artifact = replace(
            artifact,
            key=(
                f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/{digest}/"
                "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
            ),
        )
    elif name == "closed_gpu_spend_snapshot":
        artifact = replace(
            artifact,
            key=(
                f"campaigns/{RUN_ID}/spend-snapshots/{digest}/GPU_SPEND_SNAPSHOT.json"
            ),
        )
    return artifact


def _mutated_record(
    chain: dict[str, object],
    name: str,
    *,
    digest_field: str,
    path: tuple[str | int, ...],
    value: object,
) -> VersionedJsonArtifact:
    record = copy.deepcopy(chain["records"][name])
    target: Any = record
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    record = _rehash_source_record(name, record, digest_field)
    return _artifact_for_source_record(chain, name, record, digest_field)


def _put_rehashed_source_record(
    chain: dict[str, object],
    name: str,
    record: dict[str, object],
    digest_field: str,
) -> None:
    rehashed = _rehash_source_record(name, record, digest_field)
    chain["records"][name] = rehashed
    chain["artifacts"][name] = _artifact_for_source_record(
        chain,
        name,
        rehashed,
        digest_field,
    )


def _cases_with_expected_errors(
    cases: list[Any],
    messages_by_id: dict[str, str],
) -> list[Any]:
    case_ids = {case.id for case in cases}
    assert None not in case_ids
    assert case_ids == set(messages_by_id)
    return [
        pytest.param(
            *case.values,
            f"^{re.escape(messages_by_id[case.id])}$",
            id=case.id,
            marks=case.marks,
        )
        for case in cases
    ]


def _chain_with_must_start_by(
    source_chain: dict[str, object],
    must_start_by: datetime,
) -> dict[str, object]:
    chain = copy.deepcopy(source_chain)

    descriptor = copy.deepcopy(chain["records"]["descriptor"])
    descriptor["must_start_by"] = _iso(must_start_by)
    _put_rehashed_source_record(
        chain,
        "descriptor",
        descriptor,
        "descriptor_body_sha256",
    )
    descriptor = chain["records"]["descriptor"]
    descriptor_artifact = chain["artifacts"]["descriptor"]

    manifest = copy.deepcopy(chain["records"]["bundle_manifest"])
    descriptor_row = next(
        row for row in manifest["files"] if row["role"] == "descriptor"
    )
    descriptor_row.update(
        {
            "key": descriptor_artifact.key,
            "size": len(descriptor_artifact.raw),
            "sha256": _sha(descriptor_artifact.raw),
        }
    )
    manifest["descriptor_body_sha256"] = descriptor["descriptor_body_sha256"]
    _put_rehashed_source_record(
        chain,
        "bundle_manifest",
        manifest,
        "bundle_manifest_body_sha256",
    )
    manifest = chain["records"]["bundle_manifest"]
    manifest_artifact = chain["artifacts"]["bundle_manifest"]

    staged = copy.deepcopy(chain["records"]["staged_readiness"])
    staged.update(
        {
            "descriptor_key": descriptor_artifact.key,
            "descriptor_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "bundle_manifest_key": manifest_artifact.key,
            "bundle_manifest_file_sha256": _sha(manifest_artifact.raw),
            "bundle_manifest_body_sha256": manifest["bundle_manifest_body_sha256"],
        }
    )
    _put_rehashed_source_record(
        chain,
        "staged_readiness",
        staged,
        "ready_body_sha256",
    )
    staged = chain["records"]["staged_readiness"]
    staged_artifact = chain["artifacts"]["staged_readiness"]

    rehearsal = copy.deepcopy(chain["records"]["rehearsal_evidence"])
    rehearsal.update(
        {
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "descriptor_key": descriptor_artifact.key,
            "descriptor_file_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "staged_readiness_key": staged_artifact.key,
            "staged_readiness_file_sha256": _sha(staged_artifact.raw),
            "staged_readiness_body_sha256": staged["ready_body_sha256"],
        }
    )
    _put_rehashed_source_record(
        chain,
        "rehearsal_evidence",
        rehearsal,
        "rehearsal_body_sha256",
    )

    snapshot = copy.deepcopy(chain["records"]["closed_gpu_spend_snapshot"])
    snapshot.update(
        {
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "descriptor_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        }
    )
    _put_rehashed_source_record(
        chain,
        "closed_gpu_spend_snapshot",
        snapshot,
        "snapshot_body_sha256",
    )
    return chain


def _chain_with_reapproved_numeric_field(
    source_chain: dict[str, object],
    *,
    field: str,
    value: object,
) -> dict[str, object]:
    chain = copy.deepcopy(source_chain)

    approval = copy.deepcopy(chain["records"]["approval"])
    approval[field] = value
    _put_rehashed_source_record(
        chain,
        "approval",
        approval,
        "approval_body_sha256",
    )
    approval = chain["records"]["approval"]
    approval_artifact = chain["artifacts"]["approval"]
    approval_file_sha = _sha(approval_artifact.raw)

    descriptor = copy.deepcopy(chain["records"]["descriptor"])
    descriptor["approval_sha256"] = approval_file_sha
    _put_rehashed_source_record(
        chain,
        "descriptor",
        descriptor,
        "descriptor_body_sha256",
    )
    descriptor = chain["records"]["descriptor"]
    descriptor_artifact = chain["artifacts"]["descriptor"]
    descriptor_file_sha = _sha(descriptor_artifact.raw)

    manifest = copy.deepcopy(chain["records"]["bundle_manifest"])
    approval_row = next(row for row in manifest["files"] if row["role"] == "approval")
    approval_row.update(
        {
            "size": len(approval_artifact.raw),
            "sha256": approval_file_sha,
        }
    )
    descriptor_row = next(
        row for row in manifest["files"] if row["role"] == "descriptor"
    )
    descriptor_row.update(
        {
            "size": len(descriptor_artifact.raw),
            "sha256": descriptor_file_sha,
        }
    )
    manifest["descriptor_body_sha256"] = descriptor["descriptor_body_sha256"]
    _put_rehashed_source_record(
        chain,
        "bundle_manifest",
        manifest,
        "bundle_manifest_body_sha256",
    )
    manifest = chain["records"]["bundle_manifest"]
    manifest_artifact = chain["artifacts"]["bundle_manifest"]

    staged = copy.deepcopy(chain["records"]["staged_readiness"])
    staged.update(
        {
            "descriptor_sha256": descriptor_file_sha,
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "bundle_manifest_key": manifest_artifact.key,
            "bundle_manifest_file_sha256": _sha(manifest_artifact.raw),
            "bundle_manifest_body_sha256": manifest["bundle_manifest_body_sha256"],
        }
    )
    _put_rehashed_source_record(
        chain,
        "staged_readiness",
        staged,
        "ready_body_sha256",
    )
    staged = chain["records"]["staged_readiness"]
    staged_artifact = chain["artifacts"]["staged_readiness"]

    rehearsal = copy.deepcopy(chain["records"]["rehearsal_evidence"])
    rehearsal.update(
        {
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "descriptor_file_sha256": descriptor_file_sha,
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "staged_readiness_file_sha256": _sha(staged_artifact.raw),
            "staged_readiness_body_sha256": staged["ready_body_sha256"],
        }
    )
    _put_rehashed_source_record(
        chain,
        "rehearsal_evidence",
        rehearsal,
        "rehearsal_body_sha256",
    )

    cache = copy.deepcopy(chain["records"]["cache_seed_acceptance"])
    cache["seed_authority"]["approval_sha256"] = approval_file_sha
    _put_rehashed_source_record(
        chain,
        "cache_seed_acceptance",
        cache,
        "acceptance_body_sha256",
    )

    h100 = copy.deepcopy(chain["records"]["h100_resume_ready"])
    h100["campaign_identity_sha256"] = descriptor["campaign_identity_sha256"]
    _put_rehashed_source_record(
        chain,
        "h100_resume_ready",
        h100,
        "ready_body_sha256",
    )

    for allocation_name in (
        "first_h100_runtime_allocation",
        "replacement_h100_runtime_allocation",
    ):
        allocation = copy.deepcopy(chain["records"][allocation_name])
        allocation["approval_sha256"] = approval_file_sha
        _put_rehashed_source_record(
            chain,
            allocation_name,
            allocation,
            "allocation_body_sha256",
        )

    snapshot = copy.deepcopy(chain["records"]["closed_gpu_spend_snapshot"])
    snapshot.update(
        {
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "descriptor_sha256": descriptor_file_sha,
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "approval_sha256": approval_file_sha,
            "approval_body_sha256": approval["approval_body_sha256"],
        }
    )
    _put_rehashed_source_record(
        chain,
        "closed_gpu_spend_snapshot",
        snapshot,
        "snapshot_body_sha256",
    )
    return chain


def _independent_expected_intent(
    chain: dict[str, object],
) -> dict[str, object]:
    records: Any = chain["records"]
    artifacts: Any = chain["artifacts"]
    descriptor = records["descriptor"]
    approval = records["approval"]
    staged = records["staged_readiness"]
    manifest = records["bundle_manifest"]
    rehearsal = records["rehearsal_evidence"]
    cache = records["cache_seed_acceptance"]
    h100 = records["h100_resume_ready"]
    first = records["first_h100_runtime_allocation"]
    replacement_allocation = records["replacement_h100_runtime_allocation"]
    snapshot = records["closed_gpu_spend_snapshot"]
    descriptor_artifact = artifacts["descriptor"]
    approval_artifact = artifacts["approval"]
    staged_artifact = artifacts["staged_readiness"]
    manifest_artifact = artifacts["bundle_manifest"]
    rehearsal_artifact = artifacts["rehearsal_evidence"]
    cache_artifact = artifacts["cache_seed_acceptance"]
    h100_artifact = artifacts["h100_resume_ready"]
    first_artifact = artifacts["first_h100_runtime_allocation"]
    replacement_artifact = artifacts["replacement_h100_runtime_allocation"]
    snapshot_artifact = artifacts["closed_gpu_spend_snapshot"]
    expected: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_production_submission_intent_v1",
        "managed_mode": "production",
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "sky_job_name": RUN_ID,
        "must_start_by": _iso(MUST_START_BY),
        "intent_at": _iso(INTENT_AT),
        "descriptor_key": descriptor_artifact.key,
        "descriptor_file_sha256": _sha(descriptor_artifact.raw),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "descriptor_version_id": descriptor_artifact.version_id,
        "repo_tar_key": descriptor["repo_tar_key"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "approval_key": approval_artifact.key,
        "approval_file_sha256": _sha(approval_artifact.raw),
        "approval_body_sha256": approval["approval_body_sha256"],
        "approval_version_id": approval_artifact.version_id,
        "staged_readiness_key": staged_artifact.key,
        "staged_readiness_file_sha256": _sha(staged_artifact.raw),
        "staged_readiness_body_sha256": staged["ready_body_sha256"],
        "staged_readiness_version_id": staged_artifact.version_id,
        "staged_at": staged["staged_at"],
        "bundle_manifest_key": manifest_artifact.key,
        "bundle_manifest_file_sha256": _sha(manifest_artifact.raw),
        "bundle_manifest_body_sha256": manifest["bundle_manifest_body_sha256"],
        "bundle_manifest_version_id": manifest_artifact.version_id,
        "rehearsal_evidence_key": rehearsal_artifact.key,
        "rehearsal_evidence_file_sha256": _sha(rehearsal_artifact.raw),
        "rehearsal_evidence_body_sha256": rehearsal["rehearsal_body_sha256"],
        "rehearsal_evidence_version_id": rehearsal_artifact.version_id,
        "rehearsal_completed_at": rehearsal["completed_at"],
        "cache_seed_acceptance_key": cache_artifact.key,
        "cache_seed_acceptance_file_sha256": _sha(cache_artifact.raw),
        "cache_seed_acceptance_body_sha256": cache["acceptance_body_sha256"],
        "cache_seed_acceptance_version_id": cache_artifact.version_id,
        "qualification_cache_prefix": descriptor["artifacts"][
            "qualification_cache_prefix"
        ],
        "qualification_cache_manifest_sha256": descriptor["artifacts"][
            "qualification_cache_manifest_sha256"
        ],
        "cache_seed_instance_id": cache["spend_closure"]["instance_id"],
        "cache_seed_accepted_at": cache["accepted_at"],
        "h100_resume_ready_key": h100_artifact.key,
        "h100_resume_ready_file_sha256": _sha(h100_artifact.raw),
        "h100_resume_ready_body_sha256": h100["ready_body_sha256"],
        "h100_resume_ready_version_id": h100_artifact.version_id,
        "h100_first_instance_id": h100["first_instance_id"],
        "h100_replacement_instance_id": h100["replacement_instance_id"],
        "h100_completed_at": h100["completed_at"],
        "first_h100_allocation_key": first_artifact.key,
        "first_h100_allocation_file_sha256": _sha(first_artifact.raw),
        "first_h100_allocation_body_sha256": first["allocation_body_sha256"],
        "first_h100_allocation_record_sha256": first["gpu_spend_record_sha256"],
        "first_h100_allocation_version_id": first_artifact.version_id,
        "replacement_h100_allocation_key": replacement_artifact.key,
        "replacement_h100_allocation_file_sha256": _sha(replacement_artifact.raw),
        "replacement_h100_allocation_body_sha256": replacement_allocation[
            "allocation_body_sha256"
        ],
        "replacement_h100_allocation_record_sha256": replacement_allocation[
            "gpu_spend_record_sha256"
        ],
        "replacement_h100_allocation_version_id": replacement_artifact.version_id,
        "gpu_spend_snapshot_key": snapshot_artifact.key,
        "gpu_spend_snapshot_file_sha256": _sha(snapshot_artifact.raw),
        "gpu_spend_snapshot_body_sha256": snapshot["snapshot_body_sha256"],
        "gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
        "gpu_spend_ledger_tip_record_sha256": snapshot[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "gpu_spend_ledger_genesis_sha256": snapshot["gpu_spend_ledger_genesis_sha256"],
        "ec2_allocation_history_sha256": snapshot["ec2_allocation_history_sha256"],
        "closed_instance_ids": copy.deepcopy(snapshot["ec2_allocation_instance_ids"]),
        "open_allocation_count": snapshot["open_allocation_count"],
        "approved_gpu_runtime_seconds": snapshot["approved_gpu_runtime_seconds"],
        "approved_gpu_cost_usd": snapshot["approved_gpu_cost_usd"],
        "hourly_cost_usd": snapshot["hourly_cost_usd"],
        "consumed_gpu_seconds": snapshot["consumed_gpu_seconds"],
        "consumed_gpu_cost_usd": snapshot["consumed_gpu_cost_usd"],
        "remaining_gpu_seconds": snapshot["remaining_gpu_seconds"],
        "remaining_gpu_cost_usd": snapshot["remaining_gpu_cost_usd"],
        "spend_snapshot_observed_at": snapshot["observed_at"],
        "intent_body_sha256": EXPECTED_INTENT_BODY_SHA256,
    }
    assert set(expected) == INTENT_FIELDS
    return expected


def test_public_interface_is_exact() -> None:
    import mlx_vq.quality.glm52_sky_production_submission as module

    assert module.__all__ == [
        "ProductionSubmissionError",
        "VersionedJsonArtifact",
        "build_production_submission_intent",
        "canonical_file_bytes",
        "production_submission_intent_file_bytes",
        "production_submission_intent_file_sha256",
        "production_submission_intent_s3_key",
        "validate_production_submission_intent",
    ]


def test_import_fallback_does_not_mask_transitive_named_module_error(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "glm52_gpu_spend_snapshot.py").write_text(
        "raise ModuleNotFoundError("
        '"transitive dependency is absent", '
        'name="glm52_h100_qualification")\n',
        encoding="utf-8",
    )
    module_path = ROOT / "src/mlx_vq/quality/glm52_sky_production_submission.py"
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('probe', {str(module_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "try:\n"
        "    spec.loader.exec_module(module)\n"
        "except ModuleNotFoundError as error:\n"
        "    raise SystemExit(0 if error.name == 'glm52_h100_qualification' else 2)\n"
        "raise SystemExit(3)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(fake_modules), str(ROOT / "src")])
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


def test_realistic_post_seed_cross_node_chain_builds_exact_intent(
    source_chain: dict[str, object],
) -> None:
    intent = _build(source_chain)
    expected = _independent_expected_intent(source_chain)
    expected_body = dict(expected)
    expected_body.pop("intent_body_sha256")
    expected_file = _json_bytes(expected)

    assert _sha(_canonical(expected_body)) == EXPECTED_INTENT_BODY_SHA256
    assert _sha(expected_file) == EXPECTED_INTENT_FILE_SHA256
    assert len(expected_file) == EXPECTED_INTENT_FILE_SIZE
    assert intent == expected
    assert validate_production_submission_intent(intent) == intent
    assert production_submission_intent_file_bytes(intent) == expected_file
    assert expected_file.endswith(b"\n") and not expected_file.endswith(b"\n\n")
    assert (
        production_submission_intent_file_sha256(intent) == EXPECTED_INTENT_FILE_SHA256
    )
    assert production_submission_intent_s3_key(
        run_id=RUN_ID,
        intent_body_sha256=EXPECTED_INTENT_BODY_SHA256,
    ) == (
        f"campaigns/{RUN_ID}/submissions/production/intents/"
        f"{EXPECTED_INTENT_BODY_SHA256}/SKYPILOT_SUBMISSION_INTENT.json"
    )


def test_intent_is_structural_only_and_preserves_transitive_attestations(
    source_chain: dict[str, object],
) -> None:
    intent = _build(source_chain)
    forbidden_fragments = {
        "automatic",
        "deployment_ready",
        "launch_ready",
        "live_ready",
        "source_authenticated",
        "version_current",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
    }
    assert forbidden_fragments.isdisjoint(intent)
    assert intent["closed_instance_ids"] == source_chain["closed_ids"]
    assert (
        source_chain["seed_descriptor"]["campaign_identity_sha256"]
        != intent["campaign_identity_sha256"]
    )


def test_builder_performs_no_file_clock_network_or_process_io(
    source_chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("I/O attempted")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert _build(source_chain)["managed_mode"] == "production"


@pytest.mark.parametrize(
    "value",
    [
        {"text": "\u2603"},
        {"nested": [None, True, 1, 1.5, "ok"]},
    ],
)
def test_canonical_file_bytes_are_sorted_compact_ascii_with_one_lf(
    value: dict[str, object],
) -> None:
    raw = canonical_file_bytes(value)
    assert raw == _canonical(value) + b"\n"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert all(byte < 128 for byte in raw)
    if value == {"text": "\u2603"}:
        assert b"\\u2603" in raw


@pytest.mark.parametrize(
    "value",
    [
        {"tuple": (1,)},
        {"set": {1}},
        {"bytes": b"x"},
        {1: "non-string-key"},
        DictSubclass({"x": 1}),
        {"list": ListSubclass([1])},
        {"string": StringSubclass("x")},
        {"integer": IntSubclass(1)},
        {"nan": math.nan},
        {"positive_infinity": math.inf},
        {"negative_infinity": -math.inf},
    ],
)
def test_canonical_file_bytes_reject_non_exact_or_nonfinite_json_types(
    value: object,
) -> None:
    with pytest.raises(ProductionSubmissionError):
        canonical_file_bytes(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw_transform",
    [
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: raw[:-1] + b"\r\n",
        lambda raw: b"\xef\xbb\xbf" + raw,
        lambda raw: b" " + raw,
        lambda raw: raw[:-1] + b" \n",
        lambda raw: raw.replace(b"{", b"{ ", 1),
        lambda raw: raw.replace(b":", b": ", 1),
    ],
)
def test_every_noncanonical_raw_envelope_spelling_fails(
    source_chain: dict[str, object],
    raw_transform: Any,
) -> None:
    descriptor = source_chain["artifacts"]["descriptor"]
    changed = replace(descriptor, raw=raw_transform(descriptor.raw))
    with pytest.raises(ProductionSubmissionError):
        _build(source_chain, descriptor=changed)


@pytest.mark.parametrize(
    "raw",
    [
        lambda value: BytesSubclass(value),
        lambda value: bytearray(value),
        lambda value: memoryview(value),
        lambda value: value.decode("utf-8"),
    ],
)
def test_source_raw_requires_exact_bytes(
    source_chain: dict[str, object],
    raw: Any,
) -> None:
    descriptor = source_chain["artifacts"]["descriptor"]
    changed = replace(descriptor, raw=raw(descriptor.raw))
    with pytest.raises(ProductionSubmissionError, match="exact bytes"):
        _build(source_chain, descriptor=changed)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}\n',
        b"\xff\n",
        b'{"unterminated":\n',
        b"[]\n",
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":-Infinity}\n',
        b'{"text":"\xe2\x98\x83"}\n',
    ],
)
def test_malformed_duplicate_nonobject_nonfinite_or_literal_unicode_raw_fails(
    source_chain: dict[str, object],
    raw: bytes,
) -> None:
    descriptor = replace(source_chain["artifacts"]["descriptor"], raw=raw)
    with pytest.raises(ProductionSubmissionError):
        _build(source_chain, descriptor=descriptor)


@pytest.mark.parametrize("wildcard", ["*", "?", "[", "]"])
def test_each_wildcard_source_key_fails(
    source_chain: dict[str, object],
    wildcard: str,
) -> None:
    artifact = source_chain["artifacts"]["bundle_manifest"]
    changed = replace(artifact, key=f"campaigns/{RUN_ID}/{wildcard}/manifest.json")
    with pytest.raises(ProductionSubmissionError):
        _build(source_chain, bundle_manifest=changed)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "/leading",
        "trailing/",
        "double//segment",
        "dot/./segment",
        "dotdot/../segment",
        "has space/key",
        "backslash\\key",
        "caf\u00e9/key",
    ],
)
def test_all_other_unsafe_source_key_shapes_fail(
    source_chain: dict[str, object],
    key: str,
) -> None:
    changed = replace(source_chain["artifacts"]["descriptor"], key=key)
    with pytest.raises(ProductionSubmissionError):
        _build(source_chain, descriptor=changed)


@pytest.mark.parametrize(
    "version_id",
    ["", "null", "has space", "\t", "caf\u00e9", StringSubclass("v1")],
)
def test_invalid_source_version_ids_fail(
    source_chain: dict[str, object],
    version_id: object,
) -> None:
    changed = replace(
        source_chain["artifacts"]["descriptor"],
        version_id=version_id,
    )
    with pytest.raises(ProductionSubmissionError):
        _build(source_chain, descriptor=changed)


@pytest.mark.parametrize(
    ("name", "digest_field", "path", "value", "error_pattern"),
    _cases_with_expected_errors(
        [
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("schema_version",),
                True,
                id="staged-schema-bool",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("schema_version",),
                1.0,
                id="manifest-schema-float",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 0, "stage_order"),
                10.0,
                id="manifest-stage-order-float",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 0, "size"),
                True,
                id="manifest-size-bool",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("schema_version",),
                True,
                id="rehearsal-schema-bool",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("training_sample_steps",),
                True,
                id="h100-steps-bool",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("training_sample_steps",),
                2.0,
                id="h100-steps-float",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("peak_gpu_gib",),
                69,
                id="h100-peak-int",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("peak_gpu_gib",),
                True,
                id="h100-peak-bool",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("remaining_gpu_seconds",),
                80_000.0,
                id="allocation-remaining-float",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("estimated_gpu_cost_usd",),
                1,
                id="allocation-money-int",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("estimated_gpu_cost_usd",),
                True,
                id="allocation-money-bool",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("open_allocation_count",),
                False,
                id="snapshot-open-bool",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("remaining_gpu_seconds",),
                85_800.0,
                id="snapshot-remaining-float",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("hourly_cost_usd",),
                True,
                id="snapshot-money-bool",
            ),
        ],
        {
            "staged-schema-bool": "staged readiness schema mismatch",
            "manifest-schema-float": "bundle manifest schema mismatch",
            "manifest-stage-order-float": (
                "bundle manifest row 0 stage_order must be an exact integer"
            ),
            "manifest-size-bool": (
                "bundle manifest row 0 size must be an exact integer"
            ),
            "rehearsal-schema-bool": "rehearsal schema or status mismatch",
            "h100-steps-bool": (
                "H100 qualification must run exactly two training steps"
            ),
            "h100-steps-float": "H100 readiness source chain mismatch",
            "h100-peak-int": "H100 readiness source chain mismatch",
            "h100-peak-bool": "H100 readiness source chain mismatch",
            "allocation-remaining-float": (
                "H100 qualification allocation remaining_gpu_seconds is invalid"
            ),
            "allocation-money-int": (
                "first estimated_gpu_cost_usd must be an exact finite float"
            ),
            "allocation-money-bool": (
                "H100 qualification allocation estimated cost is invalid"
            ),
            "snapshot-open-bool": ("open_allocation_count must be an integer >= 0"),
            "snapshot-remaining-float": (
                "remaining_gpu_seconds must be an integer >= 0"
            ),
            "snapshot-money-bool": "hourly_cost_usd must be a finite number",
        },
    ),
)
def test_source_type_confusion_fails_closed(
    source_chain: dict[str, object],
    name: str,
    digest_field: str,
    path: tuple[str | int, ...],
    value: object,
    error_pattern: str,
) -> None:
    artifact = _mutated_record(
        source_chain,
        name,
        digest_field=digest_field,
        path=path,
        value=value,
    )
    with pytest.raises(
        ProductionSubmissionError,
        match=error_pattern,
    ):
        _build(source_chain, **{name: artifact})


@pytest.mark.parametrize(
    ("path", "value", "error_pattern"),
    _cases_with_expected_errors(
        [
            pytest.param(("account_id",), "000000000000", id="descriptor-account"),
            pytest.param(("provider",), "gcp", id="descriptor-provider"),
            pytest.param(("region",), "us-east-1", id="descriptor-region"),
            pytest.param(
                ("bucket",),
                "other-valid-bucket",
                id="descriptor-bucket",
            ),
            pytest.param(("use_spot",), True, id="descriptor-use-spot"),
            pytest.param(
                ("instance_type",),
                "p4d.24xlarge",
                id="descriptor-instance",
            ),
            pytest.param(("instance_count",), 2, id="descriptor-count"),
            pytest.param(
                ("max_hourly_cost_usd",),
                55.05,
                id="descriptor-price",
            ),
            pytest.param(
                ("approved_gpu_runtime_seconds",),
                86_399,
                id="descriptor-runtime",
            ),
            pytest.param(
                ("approved_gpu_cost_usd",),
                1_320.95,
                id="descriptor-cost",
            ),
            pytest.param(("task_name",), "other-task", id="descriptor-job-task"),
            pytest.param(
                ("skypilot_version",),
                "0.12.0",
                id="descriptor-skypilot-version",
            ),
            pytest.param(
                ("approval_key",),
                f"campaigns/{RUN_ID}/authorities/other-approval.json",
                id="descriptor-approval-key",
            ),
            pytest.param(
                ("approval_sha256",),
                "f" * 64,
                id="descriptor-approval-sha",
            ),
            pytest.param(
                ("repo_tar_key",),
                f"campaigns/{RUN_ID}/repository/other.tar.gz",
                id="descriptor-repository-key",
            ),
            pytest.param(
                ("repo_tar_sha256",),
                "e" * 64,
                id="descriptor-repository-sha",
            ),
            pytest.param(
                ("artifacts", "qualification_cache_prefix"),
                f"qualification-cache/seeds/{RUN_ID}/{'d' * 64}/",
                id="descriptor-cache-prefix",
            ),
            pytest.param(
                ("artifacts", "qualification_cache_manifest_sha256"),
                "d" * 64,
                id="descriptor-cache-manifest",
            ),
            pytest.param(
                ("artifacts", "training_config_key"),
                f"campaigns/{RUN_ID}/training/other.json",
                id="descriptor-training-config-key",
            ),
            pytest.param(
                ("artifacts", "training_config_sha256"),
                "d" * 64,
                id="descriptor-training-config-sha",
            ),
            pytest.param(
                ("artifacts", "artifact_inventory_key"),
                f"campaigns/{RUN_ID}/inventories/other.json",
                id="descriptor-inventory-key",
            ),
            pytest.param(
                ("artifacts", "artifact_inventory_sha256"),
                "d" * 64,
                id="descriptor-inventory-sha",
            ),
        ],
        {
            "descriptor-account": (
                "SkyPilot campaign must use the approved AWS account"
            ),
            "descriptor-provider": "SkyPilot campaign provider must be aws",
            "descriptor-region": ("SkyPilot campaign region must be us-west-2"),
            "descriptor-bucket": (
                "SkyPilot jobs bucket must match the regional campaign bucket"
            ),
            "descriptor-use-spot": ("SkyPilot campaign must remain on-demand"),
            "descriptor-instance": (
                "SkyPilot campaign permits one p5.48xlarge instance"
            ),
            "descriptor-count": ("SkyPilot campaign permits one p5.48xlarge instance"),
            "descriptor-price": (
                "SkyPilot campaign hourly price exceeds approved authority"
            ),
            "descriptor-runtime": (
                "SkyPilot campaign GPU budget does not match approved authority"
            ),
            "descriptor-cost": (
                "SkyPilot campaign GPU budget does not match approved authority"
            ),
            "descriptor-job-task": "SkyPilot task name is invalid",
            "descriptor-skypilot-version": ("SkyPilot version is not pinned to 0.13.0"),
            "descriptor-approval-key": ("descriptor/approval authority mismatch"),
            "descriptor-approval-sha": ("descriptor/approval authority mismatch"),
            "descriptor-repository-key": "staging source chain mismatch",
            "descriptor-repository-sha": "staging source chain mismatch",
            "descriptor-cache-prefix": (
                "SkyPilot qualification cache prefix is not bound to its manifest"
            ),
            "descriptor-cache-manifest": (
                "SkyPilot qualification cache prefix is not bound to its manifest"
            ),
            "descriptor-training-config-key": (
                "SkyPilot campaign artifact key is outside the approved inventory"
            ),
            "descriptor-training-config-sha": ("staging source chain mismatch"),
            "descriptor-inventory-key": (
                "SkyPilot campaign artifact key is outside the approved inventory"
            ),
            "descriptor-inventory-sha": (
                "SkyPilot campaign artifact key is outside the approved inventory"
            ),
        },
    ),
)
def test_rehashed_descriptor_identity_drift_fails(
    source_chain: dict[str, object],
    path: tuple[str, ...],
    value: object,
    error_pattern: str,
) -> None:
    changed = _mutated_record(
        source_chain,
        "descriptor",
        digest_field="descriptor_body_sha256",
        path=path,
        value=value,
    )
    with pytest.raises(
        ProductionSubmissionError,
        match=error_pattern,
    ):
        _build(source_chain, descriptor=changed)


@pytest.mark.parametrize(
    ("name", "digest_field", "path", "value", "error_pattern"),
    _cases_with_expected_errors(
        [
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("approved_by",),
                "Other Approver",
                id="approval-identity",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("approved_hourly_usd",),
                55.05,
                id="approval-price",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("approved_gpu_hours",),
                23,
                id="approval-hours",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("approved_gpu_cost_usd",),
                1_320.95,
                id="approval-cost",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("region",),
                "us-east-1",
                id="approval-region",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("instance_type",),
                "p4d.24xlarge",
                id="approval-instance",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("includes_qualification",),
                False,
                id="approval-qualification-boolean",
            ),
            pytest.param(
                "approval",
                "approval_body_sha256",
                ("includes_recovery_instances",),
                False,
                id="approval-recovery-boolean",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("descriptor_key",),
                f"campaigns/{RUN_ID}/submissions/post-seed/other-descriptor.json",
                id="staged-descriptor-key",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("descriptor_sha256",),
                "0" * 64,
                id="staged-descriptor-file-sha",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("descriptor_body_sha256",),
                "0" * 64,
                id="staged-descriptor-body-sha",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("bundle_manifest_key",),
                f"campaigns/{RUN_ID}/submissions/post-seed/other-manifest.json",
                id="staged-manifest-key",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("bundle_manifest_file_sha256",),
                "0" * 64,
                id="staged-manifest-file-sha",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("bundle_manifest_body_sha256",),
                "0" * 64,
                id="staged-manifest-body-sha",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("bundle_manifest_version_id",),
                "different-version",
                id="staged-manifest-version-id",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("campaign_identity_sha256",),
                "0" * 64,
                id="staged-campaign",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("staged_object_version_ids", "descriptor"),
                "different-version",
                id="staged-descriptor-version-id",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("staged_object_version_ids", "approval"),
                "different-version",
                id="staged-approval-version-id",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("staged_object_version_ids", "repository_tar"),
                "bad version",
                id="staged-repository-version-id",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("artifact_audit_key",),
                f"campaigns/{RUN_ID}/audits/other.json",
                id="staged-audit-key",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("artifact_audit_sha256",),
                "0" * 64,
                id="staged-audit-sha",
            ),
            pytest.param(
                "staged_readiness",
                "ready_body_sha256",
                ("staged_at",),
                _iso(REHEARSAL_AT + timedelta(microseconds=1)),
                id="staged-time",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 0, "stage_order"),
                11,
                id="manifest-stage-order",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 0, "role"),
                "other",
                id="manifest-role-inventory",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 0, "key"),
                f"campaigns/{RUN_ID}/repository/other.tar.gz",
                id="manifest-file-key",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 5, "sha256"),
                "0" * 64,
                id="manifest-file-sha",
            ),
            pytest.param(
                "bundle_manifest",
                "bundle_manifest_body_sha256",
                ("files", 0, "size"),
                102,
                id="manifest-file-size",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("status",),
                "failed",
                id="rehearsal-status",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("descriptor_key",),
                f"campaigns/{RUN_ID}/submissions/post-seed/other.json",
                id="rehearsal-descriptor-key",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("descriptor_file_sha256",),
                "0" * 64,
                id="rehearsal-descriptor-file-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("descriptor_body_sha256",),
                "0" * 64,
                id="rehearsal-descriptor-body-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("staged_readiness_key",),
                f"campaigns/{RUN_ID}/submissions/post-seed/other-ready.json",
                id="rehearsal-readiness-key",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("staged_readiness_file_sha256",),
                "0" * 64,
                id="rehearsal-readiness-file-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("staged_readiness_body_sha256",),
                "0" * 64,
                id="rehearsal-readiness-body-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("artifact_inventory_key",),
                f"campaigns/{RUN_ID}/inventories/other.json",
                id="rehearsal-inventory-key",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("artifact_inventory_file_sha256",),
                "0" * 64,
                id="rehearsal-inventory-file-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("artifact_inventory_body_sha256",),
                "not-a-digest",
                id="rehearsal-inventory-body-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("artifact_audit_key",),
                f"campaigns/{RUN_ID}/audits/other.json",
                id="rehearsal-audit-key",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("artifact_audit_file_sha256",),
                "0" * 64,
                id="rehearsal-audit-file-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("repo_tar_sha256",),
                "0" * 64,
                id="rehearsal-repository",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("production_repo_path",),
                "/opt/other",
                id="rehearsal-production-repo-path",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("production_resume_root",),
                "/mnt/other",
                id="rehearsal-resume-path",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("skypilot_task_path",),
                "/opt/other/task.yaml",
                id="rehearsal-task-path",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("skypilot_task_file_sha256",),
                "not-a-digest",
                id="rehearsal-task-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("skypilot_config_file_sha256",),
                "not-a-digest",
                id="rehearsal-config-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("skypilot_validation_file_sha256",),
                "not-a-digest",
                id="rehearsal-validation-sha",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("skypilot_version",),
                "0.12.0",
                id="rehearsal-version",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("skypilot_task_name",),
                "other-task",
                id="rehearsal-task-name",
            ),
            pytest.param(
                "rehearsal_evidence",
                "rehearsal_body_sha256",
                ("completed_at",),
                _iso(H100_COMPLETED_AT + timedelta(microseconds=1)),
                id="rehearsal-time",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("seed_authority", "run_id"),
                "glm52-sky-20260725",
                id="cache-run",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("seed_authority", "account_id"),
                "000000000000",
                id="cache-account",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("seed_authority", "region"),
                "us-east-1",
                id="cache-region",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("seed_authority", "bucket"),
                "other-valid-bucket",
                id="cache-bucket",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("seed_authority", "repo_tar_sha256"),
                "0" * 64,
                id="cache-repository",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("seed_authority", "approval_sha256"),
                "0" * 64,
                id="cache-approval",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("cache_audit", "cache_prefix"),
                f"qualification-cache/seeds/{RUN_ID}/{'0' * 64}/",
                id="cache-prefix",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("cache_audit", "manifest_file_sha256"),
                "0" * 64,
                id="cache-manifest",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("cache_audit", "semantic_audit_pass"),
                False,
                id="cache-audit",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("cache_audit", "supervised_position_count"),
                0,
                id="cache-supervised-count",
            ),
            pytest.param(
                "cache_seed_acceptance",
                "acceptance_body_sha256",
                ("spend_closure", "instance_id"),
                SOURCE_INSTANCE_ID,
                id="cache-seed-instance",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("run_id",),
                "glm52-sky-20260725",
                id="h100-run",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("campaign_identity_sha256",),
                "0" * 64,
                id="h100-campaign",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("repo_tar_sha256",),
                "0" * 64,
                id="h100-repository",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("qualification_cache_manifest_sha256",),
                "0" * 64,
                id="h100-cache",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("first_instance_id",),
                REPLACEMENT_INSTANCE_ID,
                id="h100-source-identity",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("replacement_instance_id",),
                SOURCE_INSTANCE_ID,
                id="h100-replacement-identity",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("first_allocation_record_sha256",),
                "0" * 64,
                id="h100-source-record-sha",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("replacement_allocation_record_sha256",),
                "0" * 64,
                id="h100-replacement-record-sha",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("training_sample_steps",),
                3,
                id="h100-two-step",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("cross_node_resume",),
                False,
                id="h100-cross-node",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("peak_gpu_gib",),
                70.0,
                id="h100-peak-memory",
            ),
            pytest.param(
                "h100_resume_ready",
                "ready_body_sha256",
                ("completed_at",),
                _iso(SNAPSHOT_AT + timedelta(microseconds=1)),
                id="h100-completion-time",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("run_id",),
                "glm52-sky-20260725",
                id="first-allocation-run",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("job_id",),
                RUN_ID,
                id="first-allocation-job",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("instance_id",),
                EXTRA_CLOSED_INSTANCE_ID,
                id="first-allocation-instance",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("approval_sha256",),
                "0" * 64,
                id="first-allocation-approval",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("gpu_spend_authority_sha256",),
                "0" * 64,
                id="first-allocation-spend-authority",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("gpu_spend_record_sha256",),
                "0" * 64,
                id="first-allocation-spend-record",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("gpu_spend_ledger_sha256",),
                "not-a-digest",
                id="first-allocation-ledger-digest",
            ),
            pytest.param(
                "first_h100_runtime_allocation",
                "allocation_body_sha256",
                ("remaining_gpu_seconds",),
                80_001,
                id="first-allocation-remaining-relation",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("run_id",),
                "glm52-sky-20260725",
                id="replacement-allocation-run",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("job_id",),
                RUN_ID,
                id="replacement-allocation-job",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("instance_id",),
                EXTRA_CLOSED_INSTANCE_ID,
                id="replacement-allocation-instance",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("approval_sha256",),
                "0" * 64,
                id="replacement-allocation-approval",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("gpu_spend_authority_sha256",),
                "0" * 64,
                id="replacement-allocation-spend-authority",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("gpu_spend_record_sha256",),
                "0" * 64,
                id="replacement-allocation-spend-record",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("gpu_spend_ledger_sha256",),
                "not-a-digest",
                id="replacement-allocation-ledger-digest",
            ),
            pytest.param(
                "replacement_h100_runtime_allocation",
                "allocation_body_sha256",
                ("remaining_gpu_seconds",),
                79_001,
                id="replacement-allocation-remaining-relation",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("descriptor_sha256",),
                "0" * 64,
                id="snapshot-descriptor-file-sha",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("descriptor_body_sha256",),
                "0" * 64,
                id="snapshot-descriptor-body-sha",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("approval_sha256",),
                "0" * 64,
                id="snapshot-approval-file-sha",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("approval_body_sha256",),
                "0" * 64,
                id="snapshot-approval-body-sha",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("open_allocation_count",),
                1,
                id="snapshot-open-count",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("campaign_identity_sha256",),
                "0" * 64,
                id="snapshot-campaign",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("hourly_cost_usd",),
                55.05,
                id="snapshot-rate",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("gpu_spend_ledger_latest_sha256",),
                "not-a-digest",
                id="snapshot-ledger-latest",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("gpu_spend_ledger_latest_body_sha256",),
                "not-a-digest",
                id="snapshot-ledger-latest-body",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("gpu_spend_ledger_genesis_sha256",),
                "0" * 64,
                id="snapshot-ledger-genesis",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("gpu_spend_ledger_tip_record_sha256",),
                "not-a-digest",
                id="snapshot-ledger-tip",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("gpu_spend_ledger_file_sha256",),
                "not-a-digest",
                id="snapshot-ledger-file",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("ec2_allocation_history_sha256",),
                "not-a-digest",
                id="snapshot-ec2-history",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("ec2_allocation_instance_ids", 0),
                EXTRA_CLOSED_INSTANCE_ID,
                id="snapshot-closed-inventory",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("remaining_gpu_seconds",),
                3_600,
                id="snapshot-remaining-time",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("remaining_gpu_cost_usd",),
                55.04,
                id="snapshot-remaining-cost",
            ),
            pytest.param(
                "closed_gpu_spend_snapshot",
                "snapshot_body_sha256",
                ("observed_at",),
                _iso(INTENT_AT + timedelta(microseconds=1)),
                id="snapshot-observation-time",
            ),
        ],
        {
            "approval-identity": (
                "GPU spend approval approved_by does not match approved authority"
            ),
            "approval-price": (
                "GPU spend approval approved_hourly_usd does not match approved authority"
            ),
            "approval-hours": (
                "GPU spend approval approved_gpu_hours does not match approved authority"
            ),
            "approval-cost": (
                "GPU spend approval approved_gpu_cost_usd does not match approved authority"
            ),
            "approval-region": (
                "GPU spend approval region does not match approved authority"
            ),
            "approval-instance": (
                "GPU spend approval instance_type does not match approved authority"
            ),
            "approval-qualification-boolean": (
                "GPU spend approval includes_qualification does not match approved authority"
            ),
            "approval-recovery-boolean": (
                "GPU spend approval includes_recovery_instances does not match approved authority"
            ),
            "staged-descriptor-key": "staging source chain mismatch",
            "staged-descriptor-file-sha": "staging source chain mismatch",
            "staged-descriptor-body-sha": "staging source chain mismatch",
            "staged-manifest-key": "staging source chain mismatch",
            "staged-manifest-file-sha": "staging source chain mismatch",
            "staged-manifest-body-sha": "staging source chain mismatch",
            "staged-manifest-version-id": "staging source chain mismatch",
            "staged-campaign": "staging source chain mismatch",
            "staged-descriptor-version-id": "staging source chain mismatch",
            "staged-approval-version-id": "staging source chain mismatch",
            "staged-repository-version-id": (
                "repository_tar VersionId must be a nonempty printable-ASCII "
                "opaque VersionId"
            ),
            "staged-audit-key": "rehearsal source chain mismatch",
            "staged-audit-sha": "rehearsal source chain mismatch",
            "staged-time": "staged_at must be canonical whole-second UTC",
            "manifest-stage-order": (
                "bundle manifest roles or stage order are not exact"
            ),
            "manifest-role-inventory": (
                "bundle manifest roles or stage order are not exact"
            ),
            "manifest-file-key": "staging source chain mismatch",
            "manifest-file-sha": "staging source chain mismatch",
            "manifest-file-size": "staging source chain mismatch",
            "rehearsal-status": "rehearsal schema or status mismatch",
            "rehearsal-descriptor-key": "rehearsal source chain mismatch",
            "rehearsal-descriptor-file-sha": ("rehearsal source chain mismatch"),
            "rehearsal-descriptor-body-sha": ("rehearsal source chain mismatch"),
            "rehearsal-readiness-key": "rehearsal source chain mismatch",
            "rehearsal-readiness-file-sha": ("rehearsal source chain mismatch"),
            "rehearsal-readiness-body-sha": ("rehearsal source chain mismatch"),
            "rehearsal-inventory-key": "rehearsal source chain mismatch",
            "rehearsal-inventory-file-sha": ("rehearsal source chain mismatch"),
            "rehearsal-inventory-body-sha": (
                "artifact_inventory_body_sha256 must be lowercase SHA-256"
            ),
            "rehearsal-audit-key": "rehearsal source chain mismatch",
            "rehearsal-audit-file-sha": "rehearsal source chain mismatch",
            "rehearsal-repository": "rehearsal source chain mismatch",
            "rehearsal-production-repo-path": (
                "rehearsal fixed production contract drifted"
            ),
            "rehearsal-resume-path": ("rehearsal fixed production contract drifted"),
            "rehearsal-task-path": ("rehearsal fixed production contract drifted"),
            "rehearsal-task-sha": (
                "skypilot_task_file_sha256 must be lowercase SHA-256"
            ),
            "rehearsal-config-sha": (
                "skypilot_config_file_sha256 must be lowercase SHA-256"
            ),
            "rehearsal-validation-sha": (
                "skypilot_validation_file_sha256 must be lowercase SHA-256"
            ),
            "rehearsal-version": ("rehearsal fixed production contract drifted"),
            "rehearsal-task-name": ("rehearsal fixed production contract drifted"),
            "rehearsal-time": "production source chronology mismatch",
            "cache-run": "seed authority identity mismatch",
            "cache-account": "seed authority identity mismatch",
            "cache-region": "seed authority identity mismatch",
            "cache-bucket": "seed authority JOB_BINDING.json bucket mismatch",
            "cache-repository": "cache acceptance source chain mismatch",
            "cache-approval": "cache acceptance source chain mismatch",
            "cache-prefix": "cache audit prefix is not content-addressed",
            "cache-manifest": "cache audit prefix is not content-addressed",
            "cache-audit": (
                "cache audit prompt/session/top_k/hidden_size authority mismatch"
            ),
            "cache-supervised-count": (
                "cache audit prompt/session/top_k/hidden_size authority mismatch"
            ),
            "cache-seed-instance": "H100 readiness source chain mismatch",
            "h100-run": "H100 readiness source chain mismatch",
            "h100-campaign": "H100 readiness source chain mismatch",
            "h100-repository": "H100 readiness source chain mismatch",
            "h100-cache": "H100 readiness source chain mismatch",
            "h100-source-identity": (
                "H100 qualification instance identities must be distinct"
            ),
            "h100-replacement-identity": (
                "H100 qualification instance identities must be distinct"
            ),
            "h100-source-record-sha": ("runtime allocation source chain mismatch"),
            "h100-replacement-record-sha": ("runtime allocation source chain mismatch"),
            "h100-two-step": ("H100 qualification must run exactly two training steps"),
            "h100-cross-node": ("H100 qualification must prove cross-node resume"),
            "h100-peak-memory": ("H100 qualification peak must remain below 70 GiB"),
            "h100-completion-time": "production source chronology mismatch",
            "first-allocation-run": "runtime allocation source chain mismatch",
            "first-allocation-job": "runtime allocation source chain mismatch",
            "first-allocation-instance": ("runtime allocation source chain mismatch"),
            "first-allocation-approval": ("runtime allocation source chain mismatch"),
            "first-allocation-spend-authority": ("closed spend source chain mismatch"),
            "first-allocation-spend-record": (
                "runtime allocation source chain mismatch"
            ),
            "first-allocation-ledger-digest": (
                "H100 qualification runtime allocation "
                "gpu_spend_ledger_sha256 is invalid"
            ),
            "first-allocation-remaining-relation": (
                "H100 qualification allocation deadline does not match remaining time"
            ),
            "replacement-allocation-run": ("runtime allocation source chain mismatch"),
            "replacement-allocation-job": ("runtime allocation source chain mismatch"),
            "replacement-allocation-instance": (
                "runtime allocation source chain mismatch"
            ),
            "replacement-allocation-approval": (
                "runtime allocation source chain mismatch"
            ),
            "replacement-allocation-spend-authority": (
                "closed spend source chain mismatch"
            ),
            "replacement-allocation-spend-record": (
                "runtime allocation source chain mismatch"
            ),
            "replacement-allocation-ledger-digest": (
                "H100 qualification runtime allocation "
                "gpu_spend_ledger_sha256 is invalid"
            ),
            "replacement-allocation-remaining-relation": (
                "H100 qualification allocation deadline does not match remaining time"
            ),
            "snapshot-descriptor-file-sha": ("closed spend source chain mismatch"),
            "snapshot-descriptor-body-sha": ("closed spend source chain mismatch"),
            "snapshot-approval-file-sha": ("closed spend source chain mismatch"),
            "snapshot-approval-body-sha": ("closed spend source chain mismatch"),
            "snapshot-open-count": (
                "GPU spend snapshot runtime accounting is inconsistent"
            ),
            "snapshot-campaign": "closed spend source chain mismatch",
            "snapshot-rate": ("GPU spend snapshot dollar accounting is inconsistent"),
            "snapshot-ledger-latest": (
                "gpu_spend_ledger_latest_sha256 must be a lowercase SHA-256"
            ),
            "snapshot-ledger-latest-body": (
                "gpu_spend_ledger_latest_body_sha256 must be a lowercase SHA-256"
            ),
            "snapshot-ledger-genesis": "closed spend source chain mismatch",
            "snapshot-ledger-tip": (
                "gpu_spend_ledger_tip_record_sha256 must be a lowercase SHA-256"
            ),
            "snapshot-ledger-file": (
                "gpu_spend_ledger_file_sha256 must be a lowercase SHA-256"
            ),
            "snapshot-ec2-history": (
                "ec2_allocation_history_sha256 must be a lowercase SHA-256"
            ),
            "snapshot-closed-inventory": (
                "GPU spend snapshot EC2 allocation inventory is inconsistent"
            ),
            "snapshot-remaining-time": (
                "GPU spend snapshot runtime accounting is inconsistent"
            ),
            "snapshot-remaining-cost": (
                "GPU spend snapshot dollar accounting is inconsistent"
            ),
            "snapshot-observation-time": ("production source chronology mismatch"),
        },
    ),
)
def test_rehashed_source_semantic_or_cross_binding_drift_fails(
    source_chain: dict[str, object],
    name: str,
    digest_field: str,
    path: tuple[str | int, ...],
    value: object,
    error_pattern: str,
) -> None:
    changed = _mutated_record(
        source_chain,
        name,
        digest_field=digest_field,
        path=path,
        value=value,
    )
    with pytest.raises(
        ProductionSubmissionError,
        match=error_pattern,
    ):
        _build(source_chain, **{name: changed})


def test_allocation_singleton_key_requires_distinct_exact_version_ids(
    source_chain: dict[str, object],
) -> None:
    first = source_chain["artifacts"]["first_h100_runtime_allocation"]
    replacement_allocation = source_chain["artifacts"][
        "replacement_h100_runtime_allocation"
    ]
    with pytest.raises(
        ProductionSubmissionError,
        match="runtime allocation source chain mismatch",
    ):
        _build(
            source_chain,
            replacement_h100_runtime_allocation=replace(
                replacement_allocation,
                version_id=first.version_id,
            ),
        )
    with pytest.raises(
        ProductionSubmissionError,
        match="runtime allocation source chain mismatch",
    ):
        _build(
            source_chain,
            replacement_h100_runtime_allocation=replace(
                replacement_allocation,
                key=(
                    f"campaigns/{RUN_ID}/runtime/"
                    f"{replacement_allocation.version_id}/"
                    "GPU_RUNTIME_ALLOCATION.json"
                ),
            ),
        )


@pytest.mark.parametrize(
    ("name", "wrong_key", "error_pattern"),
    [
        (
            "descriptor",
            f"campaigns/{RUN_ID}/submissions/other/campaign-descriptor-v2.json",
            "^descriptor envelope key mismatch$",
        ),
        (
            "approval",
            f"campaigns/{RUN_ID}/authorities/OTHER_APPROVAL.json",
            "^descriptor/approval authority mismatch$",
        ),
        (
            "staged_readiness",
            f"campaigns/{RUN_ID}/submissions/other/STAGED_CONTROL_PLANE_READY.json",
            "^staging source chain mismatch$",
        ),
        (
            "bundle_manifest",
            f"campaigns/{RUN_ID}/submissions/post-seed/other-manifest.json",
            "^staging source chain mismatch$",
        ),
        (
            "rehearsal_evidence",
            f"campaigns/{RUN_ID}/qualification/rehearsals/other/rehearsal.json",
            "^rehearsal source chain mismatch$",
        ),
        (
            "cache_seed_acceptance",
            f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/other.json",
            "^cache acceptance source chain mismatch$",
        ),
        (
            "h100_resume_ready",
            f"campaigns/{RUN_ID}/qualification/OTHER_H100_READY.json",
            "^H100 readiness source chain mismatch$",
        ),
        (
            "first_h100_runtime_allocation",
            f"campaigns/{RUN_ID}/runtime/FIRST_ALLOCATION.json",
            "^runtime allocation source chain mismatch$",
        ),
        (
            "replacement_h100_runtime_allocation",
            f"campaigns/{RUN_ID}/runtime/REPLACEMENT_ALLOCATION.json",
            "^runtime allocation source chain mismatch$",
        ),
        (
            "closed_gpu_spend_snapshot",
            f"campaigns/{RUN_ID}/spend-snapshots/other/snapshot.json",
            "^closed spend source chain mismatch$",
        ),
    ],
    ids=[
        "descriptor-key",
        "approval-key",
        "staged-key",
        "manifest-key",
        "rehearsal-key",
        "cache-key",
        "h100-key",
        "first-allocation-key",
        "replacement-allocation-key",
        "snapshot-key",
    ],
)
def test_every_source_envelope_key_drift_fails(
    source_chain: dict[str, object],
    name: str,
    wrong_key: str,
    error_pattern: str,
) -> None:
    artifact = replace(source_chain["artifacts"][name], key=wrong_key)
    with pytest.raises(
        ProductionSubmissionError,
        match=error_pattern,
    ):
        _build(source_chain, **{name: artifact})


@pytest.mark.parametrize(
    "name",
    [
        "descriptor",
        "approval",
        "staged_readiness",
        "bundle_manifest",
        "rehearsal_evidence",
        "cache_seed_acceptance",
        "h100_resume_ready",
        "first_h100_runtime_allocation",
        "replacement_h100_runtime_allocation",
        "closed_gpu_spend_snapshot",
    ],
)
def test_every_source_envelope_version_id_is_lexically_validated(
    source_chain: dict[str, object],
    name: str,
) -> None:
    artifact = replace(source_chain["artifacts"][name], version_id="bad version")
    with pytest.raises(ProductionSubmissionError, match="VersionId"):
        _build(source_chain, **{name: artifact})


@pytest.mark.parametrize(
    "role",
    [
        "repository_tar",
        "approval",
        "training_config",
        "watchdog",
        "artifact_inventory",
        "artifact_audit",
        "descriptor",
    ],
)
def test_every_staged_object_version_id_is_lexically_validated(
    source_chain: dict[str, object],
    role: str,
) -> None:
    artifact = _mutated_record(
        source_chain,
        "staged_readiness",
        digest_field="ready_body_sha256",
        path=("staged_object_version_ids", role),
        value="bad version",
    )
    with pytest.raises(ProductionSubmissionError, match="VersionId"):
        _build(source_chain, staged_readiness=artifact)


@pytest.mark.parametrize(
    ("name", "digest_field"),
    [
        ("descriptor", "descriptor_body_sha256"),
        ("approval", "approval_body_sha256"),
        ("staged_readiness", "ready_body_sha256"),
        ("bundle_manifest", "bundle_manifest_body_sha256"),
        ("rehearsal_evidence", "rehearsal_body_sha256"),
        ("cache_seed_acceptance", "acceptance_body_sha256"),
        ("h100_resume_ready", "ready_body_sha256"),
        ("first_h100_runtime_allocation", "allocation_body_sha256"),
        ("replacement_h100_runtime_allocation", "allocation_body_sha256"),
        ("closed_gpu_spend_snapshot", "snapshot_body_sha256"),
    ],
)
def test_every_source_body_digest_drift_fails_its_self_hash(
    source_chain: dict[str, object],
    name: str,
    digest_field: str,
) -> None:
    record = copy.deepcopy(source_chain["records"][name])
    record[digest_field] = "0" * 64
    artifact = _artifact_for_source_record(
        source_chain,
        name,
        record,
        digest_field,
    )
    with pytest.raises(
        ProductionSubmissionError,
        match=r"(body SHA-256|body sha256|body digest)",
    ):
        _build(source_chain, **{name: artifact})


@pytest.mark.parametrize(
    ("name", "digest_field", "schema_version"),
    [
        ("descriptor", "descriptor_body_sha256", 2.0),
        ("approval", "approval_body_sha256", True),
        ("approval", "approval_body_sha256", 1.0),
        ("cache_seed_acceptance", "acceptance_body_sha256", True),
        ("cache_seed_acceptance", "acceptance_body_sha256", 1.0),
        ("h100_resume_ready", "ready_body_sha256", True),
        ("h100_resume_ready", "ready_body_sha256", 1.0),
        ("closed_gpu_spend_snapshot", "snapshot_body_sha256", True),
        ("closed_gpu_spend_snapshot", "snapshot_body_sha256", 1.0),
    ],
)
def test_builder_rejects_upstream_schema_equality_type_confusion(
    source_chain: dict[str, object],
    name: str,
    digest_field: str,
    schema_version: object,
) -> None:
    record = copy.deepcopy(source_chain["records"][name])
    record["schema_version"] = schema_version
    record = _rehash_source_record(name, record, digest_field)
    artifact = _artifact_for_source_record(
        source_chain,
        name,
        record,
        digest_field,
    )
    with pytest.raises(ProductionSubmissionError, match="schema"):
        _build(source_chain, **{name: artifact})


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        pytest.param(
            "approved_gpu_hours",
            True,
            "^GPU spend approval approved_gpu_hours does not match approved authority$",
            id="hours-bool",
        ),
        pytest.param(
            "approved_gpu_hours",
            24.0,
            "^descriptor/approval authority mismatch$",
            id="hours-float",
        ),
        pytest.param(
            "approved_hourly_usd",
            55,
            "^GPU spend approval approved_hourly_usd does not match approved authority$",
            id="hourly-int",
        ),
        pytest.param(
            "approved_hourly_usd",
            True,
            "^GPU spend approval approved_hourly_usd does not match approved authority$",
            id="hourly-bool",
        ),
        pytest.param(
            "approved_gpu_cost_usd",
            1320,
            (
                "^GPU spend approval approved_gpu_cost_usd "
                "does not match approved authority$"
            ),
            id="cost-int",
        ),
        pytest.param(
            "approved_gpu_cost_usd",
            True,
            (
                "^GPU spend approval approved_gpu_cost_usd "
                "does not match approved authority$"
            ),
            id="cost-bool",
        ),
    ],
)
def test_builder_rejects_approval_numeric_type_confusion(
    source_chain: dict[str, object],
    field: str,
    value: object,
    error_pattern: str,
) -> None:
    chain = _chain_with_reapproved_numeric_field(
        source_chain,
        field=field,
        value=value,
    )
    with pytest.raises(ProductionSubmissionError, match=error_pattern):
        _build(chain)


def test_rehearsal_extracted_path_rejects_nul_after_rehash_and_readdress(
    source_chain: dict[str, object],
) -> None:
    record = copy.deepcopy(source_chain["records"]["rehearsal_evidence"])
    record["extracted_repo_path"] = "/tmp/\x00evil"
    record = _rehash(record, "rehearsal_body_sha256")
    artifact = _artifact_with_record(
        source_chain["artifacts"]["rehearsal_evidence"],
        record,
    )
    artifact = replace(
        artifact,
        key=(
            f"campaigns/{RUN_ID}/qualification/rehearsals/"
            f"{record['rehearsal_body_sha256']}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
    )
    with pytest.raises(
        ProductionSubmissionError,
        match="extracted_repo_path is not an exact path",
    ):
        _build(source_chain, rehearsal_evidence=artifact)


def test_every_closed_instance_id_requires_full_ec2_syntax(
    source_chain: dict[str, object],
) -> None:
    snapshot = copy.deepcopy(source_chain["records"]["closed_gpu_spend_snapshot"])
    snapshot["ec2_allocation_instance_ids"][2] = "i-"
    snapshot = _rehash(snapshot, "snapshot_body_sha256")
    artifact = _artifact_with_record(
        source_chain["artifacts"]["closed_gpu_spend_snapshot"],
        snapshot,
    )
    artifact = replace(
        artifact,
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
    )
    with pytest.raises(
        ProductionSubmissionError,
        match="closed instance inventory",
    ):
        _build(source_chain, closed_gpu_spend_snapshot=artifact)

    intent = copy.deepcopy(_build(source_chain))
    intent["closed_instance_ids"][2] = "i-"
    intent = _rehash(intent, "intent_body_sha256")
    with pytest.raises(
        ProductionSubmissionError,
        match="closed inventory",
    ):
        validate_production_submission_intent(intent)


@pytest.mark.parametrize(
    ("remaining_seconds", "remaining_cost", "accepted"),
    [
        (3_600, 55.05, False),
        (3_601, 55.06, True),
        (3_601, 55.04, False),
    ],
)
def test_closed_spend_thresholds_are_strict(
    source_chain: dict[str, object],
    remaining_seconds: int,
    remaining_cost: float,
    accepted: bool,
) -> None:
    snapshot = copy.deepcopy(source_chain["records"]["closed_gpu_spend_snapshot"])
    snapshot["remaining_gpu_seconds"] = remaining_seconds
    snapshot["consumed_gpu_seconds"] = 86_400 - remaining_seconds
    snapshot["remaining_gpu_cost_usd"] = remaining_cost
    snapshot["consumed_gpu_cost_usd"] = float(
        (Decimal("1320.96") - Decimal(str(remaining_cost))).quantize(Decimal("0.01"))
    )
    snapshot["qualification_allowance_seconds"] = min(14_400, remaining_seconds)
    snapshot["qualification_allowance_cost_usd"] = _money_for_seconds(
        min(14_400, remaining_seconds),
        55.04,
    )
    snapshot = _rehash(snapshot, "snapshot_body_sha256")
    artifact = _artifact_with_record(
        source_chain["artifacts"]["closed_gpu_spend_snapshot"],
        snapshot,
    )
    artifact = replace(
        artifact,
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
    )
    if accepted:
        assert _build(source_chain, closed_gpu_spend_snapshot=artifact)
    else:
        with pytest.raises(ProductionSubmissionError):
            _build(source_chain, closed_gpu_spend_snapshot=artifact)


def test_seed_source_replacement_inventory_is_pairwise_closed_and_ordered(
    source_chain: dict[str, object],
) -> None:
    snapshot = copy.deepcopy(source_chain["records"]["closed_gpu_spend_snapshot"])
    seed_id = source_chain["records"]["cache_seed_acceptance"]["spend_closure"][
        "instance_id"
    ]
    for missing in (seed_id, SOURCE_INSTANCE_ID, REPLACEMENT_INSTANCE_ID):
        changed = copy.deepcopy(snapshot)
        changed["ec2_allocation_instance_ids"].remove(missing)
        changed["gpu_spend_ledger_record_count"] = (
            len(changed["ec2_allocation_instance_ids"]) * 2
        )
        changed = _rehash(changed, "snapshot_body_sha256")
        artifact = _artifact_for_source_record(
            source_chain,
            "closed_gpu_spend_snapshot",
            changed,
            "snapshot_body_sha256",
        )
        with pytest.raises(
            ProductionSubmissionError,
            match="closed instance inventory",
        ):
            _build(source_chain, closed_gpu_spend_snapshot=artifact)

    duplicate = copy.deepcopy(snapshot)
    duplicate["ec2_allocation_instance_ids"][2] = SOURCE_INSTANCE_ID
    duplicate = _rehash(duplicate, "snapshot_body_sha256")
    duplicate_artifact = _artifact_for_source_record(
        source_chain,
        "closed_gpu_spend_snapshot",
        duplicate,
        "snapshot_body_sha256",
    )
    with pytest.raises(ProductionSubmissionError):
        _build(
            source_chain,
            closed_gpu_spend_snapshot=duplicate_artifact,
        )


@pytest.mark.parametrize(
    ("name", "field", "timestamp", "error_pattern"),
    [
        pytest.param(
            "first_h100_runtime_allocation",
            "launched_at",
            REHEARSAL_AT - timedelta(microseconds=1),
            "^runtime allocation chronology mismatch$",
            id="first-launch-before-rehearsal",
        ),
        pytest.param(
            "first_h100_runtime_allocation",
            "observed_at",
            REPLACEMENT_OBSERVED_AT + timedelta(microseconds=1),
            "^runtime allocation chronology mismatch$",
            id="first-observation-after-replacement-observation",
        ),
        pytest.param(
            "replacement_h100_runtime_allocation",
            "launched_at",
            FIRST_OBSERVED_AT - timedelta(microseconds=1),
            "^runtime allocation chronology mismatch$",
            id="replacement-launch-before-first-observation",
        ),
        pytest.param(
            "replacement_h100_runtime_allocation",
            "observed_at",
            H100_COMPLETED_AT + timedelta(microseconds=1),
            "^runtime allocation chronology mismatch$",
            id="replacement-observation-after-h100",
        ),
    ],
)
def test_allocation_rehearsal_replacement_h100_chronology_is_closed(
    source_chain: dict[str, object],
    name: str,
    field: str,
    timestamp: datetime,
    error_pattern: str,
) -> None:
    record = copy.deepcopy(source_chain["records"][name])
    record[field] = _iso(timestamp)
    if field == "observed_at":
        record["execution_deadline"] = _iso(
            timestamp + timedelta(seconds=int(record["remaining_gpu_seconds"]))
        )
    record = _rehash(record, "allocation_body_sha256")
    artifact = _artifact_with_record(source_chain["artifacts"][name], record)
    with pytest.raises(ProductionSubmissionError, match=error_pattern):
        _build(source_chain, **{name: artifact})


@pytest.mark.parametrize(
    ("name", "digest_field", "field", "timestamp"),
    [
        pytest.param(
            "cache_seed_acceptance",
            "acceptance_body_sha256",
            "accepted_at",
            STAGED_AT + timedelta(microseconds=1),
            id="cache-after-staged",
        ),
        pytest.param(
            "staged_readiness",
            "ready_body_sha256",
            "staged_at",
            REHEARSAL_AT + timedelta(seconds=1),
            id="staged-after-rehearsal",
        ),
        pytest.param(
            "rehearsal_evidence",
            "rehearsal_body_sha256",
            "completed_at",
            H100_COMPLETED_AT + timedelta(microseconds=1),
            id="rehearsal-after-h100",
        ),
        pytest.param(
            "h100_resume_ready",
            "ready_body_sha256",
            "completed_at",
            SNAPSHOT_AT + timedelta(microseconds=1),
            id="h100-after-snapshot",
        ),
        pytest.param(
            "closed_gpu_spend_snapshot",
            "snapshot_body_sha256",
            "observed_at",
            INTENT_AT + timedelta(microseconds=1),
            id="snapshot-after-intent",
        ),
    ],
)
def test_every_main_source_chronology_edge_rejects_reverse_order(
    source_chain: dict[str, object],
    name: str,
    digest_field: str,
    field: str,
    timestamp: datetime,
) -> None:
    if name == "staged_readiness":
        chain = copy.deepcopy(source_chain)
        staged = copy.deepcopy(chain["records"]["staged_readiness"])
        staged[field] = _iso(timestamp)
        _put_rehashed_source_record(
            chain,
            "staged_readiness",
            staged,
            digest_field,
        )
        rehearsal = copy.deepcopy(chain["records"]["rehearsal_evidence"])
        staged_artifact = chain["artifacts"]["staged_readiness"]
        rehearsal["staged_readiness_file_sha256"] = _sha(staged_artifact.raw)
        rehearsal["staged_readiness_body_sha256"] = chain["records"][
            "staged_readiness"
        ]["ready_body_sha256"]
        _put_rehashed_source_record(
            chain,
            "rehearsal_evidence",
            rehearsal,
            "rehearsal_body_sha256",
        )
        with pytest.raises(ProductionSubmissionError, match="chronology"):
            _build(chain)
        return

    artifact = _mutated_record(
        source_chain,
        name,
        digest_field=digest_field,
        path=(field,),
        value=_iso(timestamp),
    )
    with pytest.raises(ProductionSubmissionError, match="chronology"):
        _build(source_chain, **{name: artifact})


def test_snapshot_age_and_must_start_boundaries(
    source_chain: dict[str, object],
) -> None:
    assert _build(source_chain, intent_at=SNAPSHOT_AT + timedelta(seconds=60))
    with pytest.raises(
        ProductionSubmissionError,
        match="^production intent time window mismatch$",
    ):
        _build(
            source_chain,
            intent_at=SNAPSHOT_AT + timedelta(seconds=60, microseconds=1),
        )
    with pytest.raises(
        ProductionSubmissionError,
        match="^production source chronology mismatch$",
    ):
        _build(source_chain, intent_at=MUST_START_BY)
    with pytest.raises(
        ProductionSubmissionError,
        match="^intent_at must be canonical UTC Z text$",
    ):
        _build(source_chain, intent_at=StringSubclass(_iso(INTENT_AT)))


def test_builder_chain_accepts_exact_12_hour_submission_window_and_rejects_one_microsecond_more(
    source_chain: dict[str, object],
) -> None:
    exact = _chain_with_must_start_by(
        source_chain,
        INTENT_AT + timedelta(hours=12),
    )
    assert _build(exact)["must_start_by"] == _iso(INTENT_AT + timedelta(hours=12))

    late = _chain_with_must_start_by(
        source_chain,
        INTENT_AT + timedelta(hours=12, microseconds=1),
    )
    with pytest.raises(
        ProductionSubmissionError,
        match="^production intent time window mismatch$",
    ):
        _build(late)


def test_standalone_and_file_helpers_accept_exact_12_hour_window_and_reject_one_microsecond_more(
    source_chain: dict[str, object],
) -> None:
    exact = copy.deepcopy(_build(source_chain))
    exact["must_start_by"] = _iso(INTENT_AT + timedelta(hours=12))
    exact = _rehash(exact, "intent_body_sha256")
    assert validate_production_submission_intent(exact) == exact
    assert production_submission_intent_file_bytes(exact) == (_canonical(exact) + b"\n")
    assert production_submission_intent_file_sha256(exact) == _sha(
        _canonical(exact) + b"\n"
    )

    late = copy.deepcopy(exact)
    late["must_start_by"] = _iso(INTENT_AT + timedelta(hours=12, microseconds=1))
    late = _rehash(late, "intent_body_sha256")
    for operation in (
        validate_production_submission_intent,
        production_submission_intent_file_bytes,
        production_submission_intent_file_sha256,
    ):
        with pytest.raises(
            ProductionSubmissionError,
            match="^production intent timestamp chain mismatch$",
        ):
            operation(late)


def test_equal_timestamps_are_accepted_for_every_non_strict_edge(
    source_chain: dict[str, object],
) -> None:
    chain = copy.deepcopy(source_chain)
    equal_at = REHEARSAL_AT

    cache = copy.deepcopy(chain["records"]["cache_seed_acceptance"])
    cache["accepted_at"] = _iso(equal_at)
    _put_rehashed_source_record(
        chain,
        "cache_seed_acceptance",
        cache,
        "acceptance_body_sha256",
    )

    staged = copy.deepcopy(chain["records"]["staged_readiness"])
    staged["staged_at"] = _iso(equal_at)
    _put_rehashed_source_record(
        chain,
        "staged_readiness",
        staged,
        "ready_body_sha256",
    )

    rehearsal = copy.deepcopy(chain["records"]["rehearsal_evidence"])
    staged_artifact = chain["artifacts"]["staged_readiness"]
    rehearsal["staged_readiness_file_sha256"] = _sha(staged_artifact.raw)
    rehearsal["staged_readiness_body_sha256"] = chain["records"]["staged_readiness"][
        "ready_body_sha256"
    ]
    rehearsal["completed_at"] = _iso(equal_at)
    _put_rehashed_source_record(
        chain,
        "rehearsal_evidence",
        rehearsal,
        "rehearsal_body_sha256",
    )

    first = copy.deepcopy(chain["records"]["first_h100_runtime_allocation"])
    first["launched_at"] = _iso(equal_at)
    first["observed_at"] = _iso(equal_at)
    first["execution_deadline"] = _iso(
        equal_at + timedelta(seconds=int(first["remaining_gpu_seconds"]))
    )
    _put_rehashed_source_record(
        chain,
        "first_h100_runtime_allocation",
        first,
        "allocation_body_sha256",
    )

    replacement_allocation = copy.deepcopy(
        chain["records"]["replacement_h100_runtime_allocation"]
    )
    replacement_allocation["launched_at"] = _iso(equal_at)
    replacement_allocation["observed_at"] = _iso(equal_at)
    replacement_allocation["execution_deadline"] = _iso(
        equal_at
        + timedelta(seconds=int(replacement_allocation["remaining_gpu_seconds"]))
    )
    _put_rehashed_source_record(
        chain,
        "replacement_h100_runtime_allocation",
        replacement_allocation,
        "allocation_body_sha256",
    )

    h100 = copy.deepcopy(chain["records"]["h100_resume_ready"])
    h100["completed_at"] = _iso(equal_at)
    _put_rehashed_source_record(
        chain,
        "h100_resume_ready",
        h100,
        "ready_body_sha256",
    )

    snapshot = copy.deepcopy(chain["records"]["closed_gpu_spend_snapshot"])
    snapshot["observed_at"] = _iso(equal_at)
    _put_rehashed_source_record(
        chain,
        "closed_gpu_spend_snapshot",
        snapshot,
        "snapshot_body_sha256",
    )

    assert _build(chain, intent_at=equal_at)


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        (
            "schema_version",
            True,
            "^production intent discriminant mismatch$",
        ),
        (
            "open_allocation_count",
            False,
            "^production intent is not spend-closed$",
        ),
        (
            "remaining_gpu_seconds",
            3_601.0,
            "^remaining_gpu_seconds must be an exact integer$",
        ),
        (
            "approved_gpu_cost_usd",
            1320,
            "^approved_gpu_cost_usd must be an exact finite float$",
        ),
        (
            "hourly_cost_usd",
            True,
            "^hourly_cost_usd must be an exact finite float$",
        ),
        (
            "closed_instance_ids",
            tuple(),
            r"^production intent\.closed_instance_ids is not an exact JSON value$",
        ),
    ],
)
def test_standalone_validator_rejects_type_confusion(
    source_chain: dict[str, object],
    field: str,
    value: object,
    error_pattern: str,
) -> None:
    intent = _build(source_chain)
    changed = {**intent, field: value}
    changed = _rehash(changed, "intent_body_sha256")
    with pytest.raises(ProductionSubmissionError, match=error_pattern):
        validate_production_submission_intent(changed)


@pytest.mark.parametrize("bucket", ["Bad_Bucket", "bad\x00bucket"])
def test_standalone_validator_rejects_bad_bucket_grammar(
    source_chain: dict[str, object],
    bucket: str,
) -> None:
    changed = copy.deepcopy(_build(source_chain))
    changed["bucket"] = bucket
    changed = _rehash(changed, "intent_body_sha256")
    with pytest.raises(ProductionSubmissionError, match="bucket"):
        validate_production_submission_intent(changed)


@pytest.mark.parametrize(
    ("mutation", "rehash", "error_pattern"),
    _cases_with_expected_errors(
        [
            pytest.param(
                lambda value: value.pop("bucket"),
                False,
                id="missing-field",
            ),
            pytest.param(
                lambda value: value.update({"unknown": True}),
                False,
                id="unknown-field",
            ),
            pytest.param(
                lambda value: value.update({"schema_version": 2}),
                True,
                id="wrong-schema",
            ),
            pytest.param(
                lambda value: value.update({"record_type": "wrong_record"}),
                True,
                id="wrong-record",
            ),
            pytest.param(
                lambda value: value.update({"managed_mode": "qualification"}),
                True,
                id="wrong-mode",
            ),
            pytest.param(
                lambda value: value.update({"sky_job_name": f"{RUN_ID}-production"}),
                True,
                id="wrong-job",
            ),
            pytest.param(
                lambda value: value.update({"intent_body_sha256": "0" * 64}),
                False,
                id="bad-self-hash",
            ),
            pytest.param(
                lambda value: value.update(
                    {"gpu_spend_snapshot_key": f"campaigns/{RUN_ID}/other.json"}
                ),
                True,
                id="bad-embedded-key",
            ),
            pytest.param(
                lambda value: value.update({"descriptor_version_id": "bad version"}),
                True,
                id="bad-version-id",
            ),
            pytest.param(
                lambda value: value.update(
                    {
                        "first_h100_allocation_version_id": "same-version",
                        "replacement_h100_allocation_version_id": "same-version",
                    }
                ),
                True,
                id="duplicate-allocation-version",
            ),
            pytest.param(
                lambda value: value.update(
                    {"closed_instance_ids": [SOURCE_INSTANCE_ID, SOURCE_INSTANCE_ID]}
                ),
                True,
                id="duplicate-instance",
            ),
            pytest.param(
                lambda value: value.update({"remaining_gpu_seconds": 3_600}),
                True,
                id="remaining-time-threshold",
            ),
            pytest.param(
                lambda value: value.update({"remaining_gpu_cost_usd": 55.04}),
                True,
                id="remaining-cost-threshold",
            ),
            pytest.param(
                lambda value: value.update({"staged_at": "2026-07-26T12:00:00Z"}),
                True,
                id="invalid-time-order",
            ),
            pytest.param(
                lambda value: value.update({"account_id": "000000000000"}),
                True,
                id="wrong-account",
            ),
            pytest.param(
                lambda value: value.update({"region": "us-east-1"}),
                True,
                id="wrong-region",
            ),
        ],
        {
            "missing-field": "production intent schema mismatch",
            "unknown-field": "production intent schema mismatch",
            "wrong-schema": "production intent discriminant mismatch",
            "wrong-record": "production intent discriminant mismatch",
            "wrong-mode": "production intent discriminant mismatch",
            "wrong-job": "production intent Sky job name mismatch",
            "bad-self-hash": "production intent body SHA-256 mismatch",
            "bad-embedded-key": "production intent embedded key mismatch",
            "bad-version-id": (
                "descriptor_version_id must be a nonempty printable-ASCII "
                "opaque VersionId"
            ),
            "duplicate-allocation-version": (
                "production allocation identities are not distinct"
            ),
            "duplicate-instance": "production intent closed inventory mismatch",
            "remaining-time-threshold": ("production intent runtime budget mismatch"),
            "remaining-cost-threshold": ("production intent dollar budget mismatch"),
            "invalid-time-order": "production intent timestamp chain mismatch",
            "wrong-account": "production intent cloud identity mismatch",
            "wrong-region": "production intent cloud identity mismatch",
        },
    ),
)
def test_standalone_validator_repeats_embedded_invariants(
    source_chain: dict[str, object],
    mutation: Any,
    rehash: bool,
    error_pattern: str,
) -> None:
    changed = copy.deepcopy(_build(source_chain))
    mutation(changed)
    if rehash:
        changed = _rehash(changed, "intent_body_sha256")
    with pytest.raises(ProductionSubmissionError, match=error_pattern):
        validate_production_submission_intent(changed)
    with pytest.raises(ProductionSubmissionError, match=error_pattern):
        production_submission_intent_file_bytes(changed)
    with pytest.raises(ProductionSubmissionError, match=error_pattern):
        production_submission_intent_file_sha256(changed)


@pytest.mark.parametrize(
    ("run_id", "digest"),
    [
        ("", "a" * 64),
        ("bad/run", "a" * 64),
        (StringSubclass("run"), "a" * 64),
        ("run", "A" * 64),
        ("run", "a" * 63),
        ("run", StringSubclass("a" * 64)),
    ],
)
def test_key_helper_rejects_unsafe_run_or_noncanonical_digest(
    run_id: object,
    digest: object,
) -> None:
    with pytest.raises(ProductionSubmissionError):
        production_submission_intent_s3_key(
            run_id=run_id,  # type: ignore[arg-type]
            intent_body_sha256=digest,  # type: ignore[arg-type]
        )


def test_all_public_failures_translate_to_production_error(
    source_chain: dict[str, object],
) -> None:
    with pytest.raises(ProductionSubmissionError):
        build_production_submission_intent(
            **{
                **_kwargs(source_chain),
                "descriptor": object(),
            }
        )
    with pytest.raises(ProductionSubmissionError):
        validate_production_submission_intent(None)  # type: ignore[arg-type]
    with pytest.raises(ProductionSubmissionError):
        canonical_file_bytes(None)  # type: ignore[arg-type]
    with pytest.raises(ProductionSubmissionError):
        production_submission_intent_file_bytes(None)  # type: ignore[arg-type]
    with pytest.raises(ProductionSubmissionError):
        production_submission_intent_file_sha256(None)  # type: ignore[arg-type]
