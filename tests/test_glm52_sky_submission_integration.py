"""Qualification submission I/O orchestration and crash-safety contracts."""

from __future__ import annotations

import base64
import datetime as datetime_module
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

UTC = timezone.utc
if not hasattr(datetime_module, "UTC"):
    datetime_module.UTC = UTC

from mlx_vq.quality.glm52_gpu_spend_snapshot import (  # noqa: E402
    build_gpu_spend_snapshot,
)
from mlx_vq.quality.glm52_qualification_submission_ready import (  # noqa: E402
    build_qualification_submission_ready,
    gpu_spend_snapshot_s3_key,
    qualification_submission_ready_s3_key,
    rehearsal_evidence_s3_key,
    validate_qualification_submission_ready,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    GpuSpendLedger,
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (  # noqa: E402
    build_must_start_job_binding,
)
from mlx_vq.quality.glm52_sky_worker_start_v2_readiness import (  # noqa: E402
    WorkerStartV2SourceSnapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.py"
SHELL = REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh"
CFN_TEMPLATE = REPO_ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
LIFECYCLE_TEST = REPO_ROOT / "tests/test_glm52_sky_submission_lifecycle.py"
STAGING_TEST = REPO_ROOT / "tests/test_glm52_sky_staging.py"

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260723"
BUCKET = "keep-glm52-us-west-2-246813579024"
NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
MUST_START_BY = datetime(2026, 7, 26, 18, 0, tzinfo=UTC)
CONTROLLER_ROLE_ARN = "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
CONTROLLER_PROFILE_ARN = (
    "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
)
CONTROLLER_INSTANCE_ID = "i-0511af4e31aa5406a"
CONTROLLER_INSTANCE_TYPE = "c6a.xlarge"
CONTROLLER_CLUSTER_NAME = "sky-jobs-controller-9d9f31a9"
SKY_JOB_NAME = f"{RUN_ID}-qualification"
SKY_TASK_RELATIVE_PATH = "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
SEALED_MOUNT_RELATIVE_PATHS = (
    "aws/glm52-gpu/skypilot/publish_worker_start_v2.py",
    "aws/glm52-gpu/skypilot/run_h100_qualification.sh",
    "src/glm52_enforcement/glm52_h100_qualification.py",
    "src/mlx_vq/quality/glm52_h100_qualification.py",
    "src/mlx_vq/quality/glm52_sky_campaign.py",
    "src/glm52_enforcement/glm52_sky_campaign.py",
    "src/mlx_vq/quality/glm52_sky_must_start.py",
    "src/glm52_enforcement/glm52_sky_must_start.py",
    "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py",
    "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
    "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py",
)
TRAINING_CONFIG_RAW = b'{"batch_size":1}\n'
WATCHDOG_RAW = b"#!/bin/sh\nexit 0\n"


def _module() -> ModuleType:
    if not SCRIPT.exists():
        pytest.fail("qualification submission orchestrator is not implemented")
    specification = importlib.util.spec_from_file_location(
        "_glm52_submission_integration_under_test",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _lifecycle_fixture_module() -> ModuleType:
    """Reuse the frozen lifecycle's complex accepted-cache fixture factory."""

    name = "_glm52_submission_lifecycle_fixture"
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, LIFECYCLE_TEST)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _staging_fixture_module() -> ModuleType:
    """Reuse the controlled in-memory stager boundary for round-trip proof."""

    name = "_glm52_sky_staging_fixture"
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, STAGING_TEST)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _self_hashed(
    body: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    return {**body, digest_field: _sha(_canonical(body))}


def _rehash(
    value: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    body = dict(value)
    body.pop(digest_field, None)
    return _self_hashed(body, digest_field)


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path


def _worker_start_v2_source_snapshot() -> WorkerStartV2SourceSnapshot:
    return WorkerStartV2SourceSnapshot(
        publisher_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
        ).read_bytes(),
        h100_qualification_worker_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/run_h100_qualification.sh"
        ).read_bytes(),
        h100_qualification_native_raw=(
            REPO_ROOT / "src/glm52_enforcement/glm52_h100_qualification.py"
        ).read_bytes(),
        h100_qualification_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_h100_qualification.py"
        ).read_bytes(),
        campaign_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
        ).read_bytes(),
        campaign_policy_native_raw=(
            REPO_ROOT / "src/glm52_enforcement/glm52_sky_campaign.py"
        ).read_bytes(),
        must_start_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start.py"
        ).read_bytes(),
        must_start_policy_native_raw=(
            REPO_ROOT / "src/glm52_enforcement/glm52_sky_must_start.py"
        ).read_bytes(),
        dynamic_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"
        ).read_bytes(),
        worker_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py"
        ).read_bytes(),
        coordinator_raw=(
            REPO_ROOT / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"
        ).read_bytes(),
        sky_task_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ).read_bytes(),
        bootstrap_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/bootstrap_campaign.sh"
        ).read_bytes(),
        managed_entrypoint_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/run_managed_campaign.sh"
        ).read_bytes(),
    )


def _temporary_worker_source_tree(root: Path) -> Path:
    source_root = root / "worker-source-root"
    for relative_path in (
        *SEALED_MOUNT_RELATIVE_PATHS,
        SKY_TASK_RELATIVE_PATH,
        "aws/glm52-gpu/skypilot/bootstrap_campaign.sh",
        "aws/glm52-gpu/skypilot/run_managed_campaign.sh",
    ):
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative_path).read_bytes())
    return source_root


def _deployment_identity() -> dict[str, object]:
    return {
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-sky-control-plane/"
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ),
        "template_sha256": "d" * 64,
        "lambda_function_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:keep-glm52-sky-must-start"
        ),
        "lambda_code_sha256": "e" * 64,
        "iam_policy_sha256": "f" * 64,
        "reconciliation_rule_arn": (
            "arn:aws:events:us-west-2:246813579024:rule/keep-glm52-sky-reconcile"
        ),
        "deadline_schedule_arn": (
            "arn:aws:scheduler:us-west-2:246813579024:"
            "schedule/default/keep-glm52-sky-deadline"
        ),
        "dlq_arn": ("arn:aws:sqs:us-west-2:246813579024:keep-glm52-sky-must-start-dlq"),
    }


def _approval() -> dict[str, object]:
    return build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
        slack_permalink=None,
    )


def _seed_descriptor(
    *,
    approval_raw: bytes,
    repo_raw: bytes,
) -> dict[str, object]:
    repo_sha256 = _sha(repo_raw)
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=MUST_START_BY,
        controller_identity=CONTROLLER_ROLE_ARN,
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=(f"campaigns/{RUN_ID}/repository/keep-{repo_sha256}.tar.gz"),
        repo_tar_sha256=repo_sha256,
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/cache-seed/campaign-descriptor-v2.json"
        ),
        approval_key=f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
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
            "training_config_key": f"campaigns/{RUN_ID}/authorities/training.json",
            "training_config_sha256": _sha(TRAINING_CONFIG_RAW),
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'8' * 64}.json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": "qualification-cache/",
            "qualification_cache_manifest_sha256": "0" * 64,
        },
    )


def _post_seed_descriptor(
    *,
    seed_descriptor: dict[str, object],
    acceptance: dict[str, object],
    inventory_raw: bytes,
) -> dict[str, object]:
    cache_audit = acceptance["cache_audit"]
    assert isinstance(cache_audit, dict)
    artifacts = dict(seed_descriptor["artifacts"])
    inventory_sha = _sha(inventory_raw)
    artifacts.update(
        {
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/"
                f"artifact-inventory-{inventory_sha}.json"
            ),
            "artifact_inventory_sha256": inventory_sha,
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
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=str(seed_descriptor["repo_tar_key"]),
        repo_tar_sha256=str(seed_descriptor["repo_tar_sha256"]),
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/qualification/campaign-descriptor-v2.json"
        ),
        approval_key=str(seed_descriptor["approval_key"]),
        approval_sha256=str(seed_descriptor["approval_sha256"]),
        artifacts=artifacts,
    )


def _staged_readiness(
    *,
    descriptor: dict[str, object],
    descriptor_raw: bytes,
    audit_raw: bytes,
    manifest: dict[str, object],
    manifest_raw: bytes,
    manifest_key: str,
    version_ids: dict[str, str],
) -> tuple[dict[str, object], str]:
    descriptor_key = str(descriptor["campaign_descriptor_key"])
    key = (
        descriptor_key.removesuffix("campaign-descriptor-v2.json")
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": RUN_ID,
        "descriptor_key": descriptor_key,
        "descriptor_sha256": _sha(descriptor_raw),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": manifest_key,
        "bundle_manifest_file_sha256": _sha(manifest_raw),
        "bundle_manifest_body_sha256": manifest["bundle_manifest_body_sha256"],
        "bundle_manifest_version_id": version_ids["bundle_manifest"],
        "staged_object_version_ids": {
            role: version_ids[role]
            for role in (
                "repository_tar",
                "approval",
                "training_config",
                "watchdog",
                "artifact_inventory",
                "artifact_audit",
                "descriptor",
            )
        },
        "artifact_audit_key": (
            f"campaigns/{RUN_ID}/audits/artifact-audit-{_sha(audit_raw)}.json"
        ),
        "artifact_audit_sha256": _sha(audit_raw),
        "staged_at": "2026-07-26T11:35:00Z",
    }
    return _self_hashed(body, "ready_body_sha256"), key


def _rehearsal(
    *,
    descriptor: dict[str, object],
    descriptor_raw: bytes,
    staged: dict[str, object],
    staged_key: str,
    task_raw: bytes,
    config_raw: bytes,
    validator_raw: bytes,
) -> tuple[dict[str, object], str]:
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_rehearsal_v2",
        "status": "passed_before_cuda_h100_boundary",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": _sha(descriptor_raw),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "bootstrap_receipt_file_sha256": "0" * 64,
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": _sha(_file_bytes(staged)),
        "staged_readiness_body_sha256": staged["ready_body_sha256"],
        "artifact_inventory_key": descriptor["artifacts"]["artifact_inventory_key"],
        "artifact_inventory_file_sha256": descriptor["artifacts"][
            "artifact_inventory_sha256"
        ],
        "artifact_inventory_body_sha256": "1" * 64,
        "artifact_audit_key": staged["artifact_audit_key"],
        "artifact_audit_file_sha256": staged["artifact_audit_sha256"],
        "extracted_repo_path": "/tmp/glm52-integration-rehearsal/repo",
        "production_repo_path": "/opt/keep-campaign/repo",
        "production_resume_root": "/mnt/nvme/glm52-campaign",
        "skypilot_task_path": (
            "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ),
        "skypilot_task_file_sha256": _sha(task_raw),
        "skypilot_config_file_sha256": _sha(config_raw),
        "skypilot_validation_file_sha256": _sha(validator_raw),
        "skypilot_version": "0.13.0",
        "skypilot_task_name": descriptor["task_name"],
        "completed_at": "2026-07-26T11:50:00Z",
    }
    value = _self_hashed(body, "rehearsal_body_sha256")
    key = rehearsal_evidence_s3_key(
        run_id=RUN_ID,
        rehearsal_body_sha256=str(value["rehearsal_body_sha256"]),
    )
    return value, key


def _spend_authorities(
    tmp_path: Path,
    *,
    descriptor_raw: bytes,
    approval_raw: bytes,
) -> tuple[
    dict[str, object],
    dict[str, bytes],
    dict[str, object],
    dict[str, object],
]:
    allocations = (
        (
            "i-00000000000000001",
            datetime(2026, 7, 26, 8, 0, tzinfo=UTC),
            datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
        ),
        (
            "i-00000000000000002",
            datetime(2026, 7, 26, 10, 30, tzinfo=UTC),
            datetime(2026, 7, 26, 11, 30, tzinfo=UTC),
        ),
    )
    ledger = GpuSpendLedger(
        tmp_path / "spend-ledger.jsonl",
        run_id=RUN_ID,
        approval_sha256=_sha(approval_raw),
        approved_gpu_runtime_seconds=86_400,
        approved_gpu_cost_usd=1_320.96,
        hourly_cost_usd=55.04,
    )
    records: dict[str, bytes] = {}
    record_keys: list[str] = []
    for index, (instance_id, launched_at, ended_at) in enumerate(allocations):
        ledger.start_allocation(
            job_id=f"sky-job-{index + 1}",
            instance_id=instance_id,
            launched_at=launched_at,
        )
        ledger.end_allocation(
            instance_id=instance_id,
            ended_at=ended_at,
        )
    for index, record in enumerate(ledger.records):
        name = f"{index:06d}-{record['event']}-{record['record_sha256']}.json"
        record_keys.append(name)
        records[name] = _file_bytes(record)
    latest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_ledger_latest_v1",
        "run_id": RUN_ID,
        "gpu_spend_authority_sha256": ledger.genesis_sha256,
        "record_count": len(record_keys),
        "record_keys": record_keys,
        "latest_record_sha256": ledger.records[-1]["record_sha256"],
        "ledger_sha256": _sha(b"".join(records[name] for name in record_keys)),
    }
    latest = _self_hashed(latest_body, "latest_body_sha256")
    history = {
        "Reservations": [
            {
                "OwnerId": ACCOUNT_ID,
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "InstanceType": "p5.48xlarge",
                        "LaunchTime": _iso(launched_at),
                        "State": {"Name": "terminated"},
                        "StateTransitionReason": (
                            "User initiated "
                            f"({ended_at.strftime('%Y-%m-%d %H:%M:%S')} GMT)"
                        ),
                        "Placement": {"AvailabilityZone": "us-west-2a"},
                        "Tags": _tags(
                            dict(
                                sorted(
                                    {
                                        "project": "keep-glm52",
                                        "owner": "jack.mazac",
                                        "model": "glm-5.2",
                                        "campaign-run-id": RUN_ID,
                                        "cost-allocation": "glm52-sky-campaign",
                                    }.items()
                                )
                            )
                        ),
                    }
                    for instance_id, launched_at, ended_at in allocations
                ],
            }
        ]
    }
    latest_raw = _file_bytes(latest)
    history_raw = _file_bytes(history)
    snapshot = build_gpu_spend_snapshot(
        descriptor_raw=descriptor_raw,
        expected_descriptor_sha256=_sha(descriptor_raw),
        approval_raw=approval_raw,
        expected_approval_sha256=_sha(approval_raw),
        latest_raw=latest_raw,
        expected_latest_sha256=_sha(latest_raw),
        immutable_record_raw_by_key=records,
        ec2_allocation_history_raw=history_raw,
        expected_ec2_allocation_history_sha256=_sha(history_raw),
        observed_at="2026-07-26T11:52:00Z",
    )
    return latest, records, history, snapshot


def _spend_latest() -> dict[str, object]:
    """Legacy malformed tip used only by corruption tests."""

    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_ledger_latest_v1",
        "run_id": RUN_ID,
        "gpu_spend_authority_sha256": "3" * 64,
        "record_count": 4,
        "record_keys": [
            f"campaigns/{RUN_ID}/runtime/spend-ledger/records/{index}.json"
            for index in range(4)
        ],
        "latest_record_sha256": "d" * 64,
        "ledger_sha256": "b" * 64,
    }
    return _self_hashed(body, "latest_body_sha256")


@dataclass
class Fixture:
    paths: Any
    values: dict[str, object]
    remote: dict[tuple[str, str], bytes]
    versions: dict[tuple[str, str, str], bytes]


def _fixture(tmp_path: Path) -> Fixture:
    module = _module()
    lifecycle_fixture = _lifecycle_fixture_module()
    approval = _approval()
    approval_raw = _file_bytes(approval)
    repo_raw = b"exact repository tar fixture\n"
    seed_descriptor = _seed_descriptor(
        approval_raw=approval_raw,
        repo_raw=repo_raw,
    )
    acceptance = lifecycle_fixture._cache_seed_acceptance(seed_descriptor)

    inventory_object = {
        "key": f"campaigns/{RUN_ID}/source/model-config.json",
        "size": 2,
        "sha256": _sha(b"{}\n"),
        "kind": "source_model",
        "safetensors": False,
        "run_scope": RUN_ID,
    }
    inventory_body = {
        "schema_version": 1,
        "record_type": "glm52_s3_artifact_inventory_v1",
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "objects": [inventory_object],
    }
    inventory = _self_hashed(inventory_body, "inventory_body_sha256")
    inventory_raw = _file_bytes(inventory)
    descriptor = _post_seed_descriptor(
        seed_descriptor=seed_descriptor,
        acceptance=acceptance,
        inventory_raw=inventory_raw,
    )
    descriptor_raw = _file_bytes(descriptor)
    audit = {
        "schema_version": 1,
        "record_type": "glm52_s3_artifact_audit_v1",
        "audit_pass": True,
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "inventory_body_sha256": inventory["inventory_body_sha256"],
        "object_count": 1,
        "object_bytes": 2,
        "safetensors_object_count": 0,
        "safetensors_tensor_count": 0,
    }
    audit_raw = _file_bytes(audit)
    files = [
        {
            "local_name": "keep-repository.tar.gz",
            "key": descriptor["repo_tar_key"],
            "role": "repository_tar",
            "stage_order": 10,
            "size": len(repo_raw),
            "sha256": _sha(repo_raw),
        },
        {
            "local_name": "GPU_SPEND_APPROVAL.json",
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
            "size": len(TRAINING_CONFIG_RAW),
            "sha256": _sha(TRAINING_CONFIG_RAW),
        },
        {
            "local_name": "watchdog.sh",
            "key": f"campaigns/{RUN_ID}/authorities/watchdog.sh",
            "role": "watchdog",
            "stage_order": 40,
            "size": len(WATCHDOG_RAW),
            "sha256": _sha(WATCHDOG_RAW),
        },
        {
            "local_name": "artifact-inventory.json",
            "key": descriptor["artifacts"]["artifact_inventory_key"],
            "role": "artifact_inventory",
            "stage_order": 50,
            "size": len(inventory_raw),
            "sha256": _sha(inventory_raw),
        },
        {
            "local_name": "campaign-descriptor-v2.json",
            "key": descriptor["campaign_descriptor_key"],
            "role": "descriptor",
            "stage_order": 60,
            "size": len(descriptor_raw),
            "sha256": _sha(descriptor_raw),
        },
    ]
    manifest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_campaign_bundle_v1",
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "files": files,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    }
    manifest = _self_hashed(manifest_body, "bundle_manifest_body_sha256")
    manifest_raw = _file_bytes(manifest)
    submission_prefix = str(descriptor["campaign_descriptor_key"]).removesuffix(
        "campaign-descriptor-v2.json"
    )
    manifest_key = (
        f"{submission_prefix}bundle-manifests/"
        f"{manifest['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    version_ids = {
        role: f"version-{role}"
        for role in (
            "repository_tar",
            "approval",
            "training_config",
            "watchdog",
            "artifact_inventory",
            "artifact_audit",
            "bundle_manifest",
            "descriptor",
        )
    }
    staged, staged_key = _staged_readiness(
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        audit_raw=audit_raw,
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_key=manifest_key,
        version_ids=version_ids,
    )
    staged_raw = _file_bytes(staged)

    latest, spend_records, ec2_history, snapshot = _spend_authorities(
        tmp_path,
        descriptor_raw=descriptor_raw,
        approval_raw=approval_raw,
    )
    spend_closure = dict(acceptance["spend_closure"])
    spend_closure["ledger_tip_record_sha256"] = latest["latest_record_sha256"]
    acceptance["spend_closure"] = spend_closure
    acceptance = _rehash(acceptance, "acceptance_body_sha256")
    latest_raw = _file_bytes(latest)

    task_raw = (REPO_ROOT / "aws/glm52-gpu/skypilot/glm52-campaign.yaml").read_bytes()
    config_raw = b"aws:\n  use_internal_ips: true\n"
    validator_raw = (
        REPO_ROOT / "aws/glm52-gpu/scripts/validate_skypilot_control_plane.py"
    ).read_bytes()
    rehearsal, rehearsal_key = _rehearsal(
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        staged=staged,
        staged_key=staged_key,
        task_raw=task_raw,
        config_raw=config_raw,
        validator_raw=validator_raw,
    )
    acceptance_key = (
        f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
        f"{acceptance['acceptance_body_sha256']}/"
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
    )
    snapshot_key = gpu_spend_snapshot_s3_key(
        run_id=RUN_ID,
        snapshot_body_sha256=str(snapshot["snapshot_body_sha256"]),
    )
    readiness = build_qualification_submission_ready(
        descriptor=descriptor,
        seed_descriptor_raw=_file_bytes(seed_descriptor),
        seed_descriptor_file_sha256=_sha(_file_bytes(seed_descriptor)),
        staged_readiness=staged,
        cache_seed_acceptance=acceptance,
        gpu_spend_snapshot=snapshot,
        rehearsal_evidence=rehearsal,
        descriptor_file_sha256=_sha(descriptor_raw),
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=_sha(staged_raw),
        cache_seed_acceptance_key=acceptance_key,
        cache_seed_acceptance_file_sha256=_sha(_file_bytes(acceptance)),
        gpu_spend_snapshot_key=snapshot_key,
        gpu_spend_snapshot_sha256=_sha(_file_bytes(snapshot)),
        rehearsal_evidence_key=rehearsal_key,
        rehearsal_evidence_sha256=_sha(_file_bytes(rehearsal)),
        built_at="2026-07-26T11:56:00Z",
    )
    validate_qualification_submission_ready(
        readiness,
        descriptor=descriptor,
        staged_readiness=staged,
        cache_seed_acceptance=acceptance,
        gpu_spend_snapshot=snapshot,
        rehearsal_evidence=rehearsal,
        descriptor_file_sha256=_sha(descriptor_raw),
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=_sha(staged_raw),
        cache_seed_acceptance_key=acceptance_key,
        cache_seed_acceptance_file_sha256=_sha(_file_bytes(acceptance)),
        gpu_spend_snapshot_key=snapshot_key,
        gpu_spend_snapshot_sha256=_sha(_file_bytes(snapshot)),
        rehearsal_evidence_key=rehearsal_key,
        rehearsal_evidence_sha256=_sha(_file_bytes(rehearsal)),
        now=NOW,
    )

    paths = module.LocalAuthorityPaths(
        descriptor=_write(tmp_path / "descriptor.json", descriptor_raw),
        seed_descriptor=_write(
            tmp_path / "seed-descriptor.json",
            _file_bytes(seed_descriptor),
        ),
        approval=_write(tmp_path / "approval.json", approval_raw),
        staged_ready=_write(tmp_path / "staged-ready.json", staged_raw),
        rehearsal_evidence=_write(
            tmp_path / "rehearsal.json",
            _file_bytes(rehearsal),
        ),
        cache_seed_accepted=_write(
            tmp_path / "cache-accepted.json",
            _file_bytes(acceptance),
        ),
        gpu_spend_snapshot=_write(
            tmp_path / "spend-snapshot.json",
            _file_bytes(snapshot),
        ),
        qualification_ready=_write(
            tmp_path / "qualification-ready.json",
            _file_bytes(readiness),
        ),
        task=_write(tmp_path / "glm52-campaign.yaml", task_raw),
        config=_write(tmp_path / "skypilot-config.yaml", config_raw),
    )
    remote = {
        (BUCKET, str(descriptor["campaign_descriptor_key"])): descriptor_raw,
        (BUCKET, str(descriptor["approval_key"])): approval_raw,
        (BUCKET, str(descriptor["repo_tar_key"])): repo_raw,
        (BUCKET, str(descriptor["artifacts"]["training_config_key"])): (
            TRAINING_CONFIG_RAW
        ),
        (BUCKET, f"campaigns/{RUN_ID}/authorities/watchdog.sh"): WATCHDOG_RAW,
        (BUCKET, str(descriptor["artifacts"]["artifact_inventory_key"])): (
            inventory_raw
        ),
        (BUCKET, str(staged["artifact_audit_key"])): audit_raw,
        (BUCKET, manifest_key): manifest_raw,
        (BUCKET, staged_key): staged_raw,
        (
            BUCKET,
            f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json",
        ): latest_raw,
        **{
            (
                BUCKET,
                f"campaigns/{RUN_ID}/runtime/spend-ledger/records/{name}",
            ): raw
            for name, raw in spend_records.items()
        },
    }
    versioned_raw = {
        "repository_tar": (
            str(descriptor["repo_tar_key"]),
            repo_raw,
        ),
        "approval": (str(descriptor["approval_key"]), approval_raw),
        "training_config": (
            str(descriptor["artifacts"]["training_config_key"]),
            TRAINING_CONFIG_RAW,
        ),
        "watchdog": (
            f"campaigns/{RUN_ID}/authorities/watchdog.sh",
            WATCHDOG_RAW,
        ),
        "artifact_inventory": (
            str(descriptor["artifacts"]["artifact_inventory_key"]),
            inventory_raw,
        ),
        "artifact_audit": (str(staged["artifact_audit_key"]), audit_raw),
        "bundle_manifest": (manifest_key, manifest_raw),
        "descriptor": (
            str(descriptor["campaign_descriptor_key"]),
            descriptor_raw,
        ),
    }
    versions = {
        (BUCKET, key, version_ids[role]): raw
        for role, (key, raw) in versioned_raw.items()
    }
    return Fixture(
        paths=paths,
        values={
            "approval": approval,
            "seed_descriptor": seed_descriptor,
            "descriptor": descriptor,
            "staged": staged,
            "staged_key": staged_key,
            "manifest": manifest,
            "manifest_key": manifest_key,
            "version_ids": version_ids,
            "acceptance": acceptance,
            "acceptance_key": acceptance_key,
            "snapshot": snapshot,
            "snapshot_key": snapshot_key,
            "rehearsal": rehearsal,
            "rehearsal_key": rehearsal_key,
            "readiness": readiness,
            "readiness_key": qualification_submission_ready_s3_key(readiness),
            "deployment_identity": _deployment_identity(),
            "ec2_history": ec2_history,
            "spend_record_names": tuple(spend_records),
        },
        remote=remote,
        versions=versions,
    )


@dataclass
class CacheSeedFixture:
    paths: Any
    values: dict[str, object]
    remote: dict[tuple[str, str], bytes]
    versions: dict[tuple[str, str, str], bytes]


def _cache_seed_fixture(tmp_path: Path) -> CacheSeedFixture:
    """Build a disjoint pre-seed authority with no post-seed artifacts."""

    module = _module()
    approval = _approval()
    approval_raw = _file_bytes(approval)
    repo_raw = b"exact cache-seed repository tar fixture\n"
    inventory = _self_hashed(
        {
            "schema_version": 1,
            "record_type": "glm52_s3_artifact_inventory_v1",
            "run_id": RUN_ID,
            "bucket": BUCKET,
            "objects": [
                {
                    "key": f"campaigns/{RUN_ID}/source/model-config.json",
                    "size": 2,
                    "sha256": _sha(b"{}\n"),
                    "kind": "source_model",
                    "safetensors": False,
                    "run_scope": RUN_ID,
                }
            ],
        },
        "inventory_body_sha256",
    )
    inventory_raw = _file_bytes(inventory)
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=MUST_START_BY,
        controller_identity=CONTROLLER_ROLE_ARN,
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=f"campaigns/{RUN_ID}/repository/keep-{_sha(repo_raw)}.tar.gz",
        repo_tar_sha256=_sha(repo_raw),
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/cache-seed/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
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
            "training_config_key": f"campaigns/{RUN_ID}/authorities/training.json",
            "training_config_sha256": _sha(TRAINING_CONFIG_RAW),
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/"
                f"artifact-inventory-{_sha(inventory_raw)}.json"
            ),
            "artifact_inventory_sha256": _sha(inventory_raw),
            "qualification_cache_prefix": "qualification-cache/",
            "qualification_cache_manifest_sha256": "0" * 64,
        },
    )
    descriptor_raw = _file_bytes(descriptor)
    audit = {
        "schema_version": 1,
        "record_type": "glm52_s3_artifact_audit_v1",
        "audit_pass": True,
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "inventory_body_sha256": inventory["inventory_body_sha256"],
        "object_count": 1,
        "object_bytes": 2,
        "safetensors_object_count": 0,
        "safetensors_tensor_count": 0,
    }
    audit_raw = _file_bytes(audit)
    files = [
        {
            "local_name": "keep-repository.tar.gz",
            "key": descriptor["repo_tar_key"],
            "role": "repository_tar",
            "stage_order": 10,
            "size": len(repo_raw),
            "sha256": _sha(repo_raw),
        },
        {
            "local_name": "GPU_SPEND_APPROVAL.json",
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
            "size": len(TRAINING_CONFIG_RAW),
            "sha256": _sha(TRAINING_CONFIG_RAW),
        },
        {
            "local_name": "watchdog.sh",
            "key": f"campaigns/{RUN_ID}/authorities/watchdog.sh",
            "role": "watchdog",
            "stage_order": 40,
            "size": len(WATCHDOG_RAW),
            "sha256": _sha(WATCHDOG_RAW),
        },
        {
            "local_name": "artifact-inventory.json",
            "key": descriptor["artifacts"]["artifact_inventory_key"],
            "role": "artifact_inventory",
            "stage_order": 50,
            "size": len(inventory_raw),
            "sha256": _sha(inventory_raw),
        },
        {
            "local_name": "campaign-descriptor-v2.json",
            "key": descriptor["campaign_descriptor_key"],
            "role": "descriptor",
            "stage_order": 60,
            "size": len(descriptor_raw),
            "sha256": _sha(descriptor_raw),
        },
    ]
    manifest = _self_hashed(
        {
            "schema_version": 1,
            "record_type": "glm52_sky_campaign_bundle_v1",
            "run_id": RUN_ID,
            "bucket": BUCKET,
            "files": files,
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        },
        "bundle_manifest_body_sha256",
    )
    manifest_raw = _file_bytes(manifest)
    manifest_key = (
        f"campaigns/{RUN_ID}/submissions/cache-seed/bundle-manifests/"
        f"{manifest['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    version_ids = {
        role: f"version-{role}"
        for role in (
            "repository_tar",
            "approval",
            "training_config",
            "watchdog",
            "artifact_inventory",
            "artifact_audit",
            "bundle_manifest",
            "descriptor",
        )
    }
    staged, staged_key = _staged_readiness(
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        audit_raw=audit_raw,
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_key=manifest_key,
        version_ids=version_ids,
    )
    staged_raw = _file_bytes(staged)
    task_raw = (
        REPO_ROOT / "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
    ).read_bytes()
    config_raw = b"aws:\n  use_internal_ips: true\n"
    validator_raw = (
        REPO_ROOT / "aws/glm52-gpu/scripts/validate_skypilot_control_plane.py"
    ).read_bytes()
    rehearsal, rehearsal_key = _rehearsal(
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        staged=staged,
        staged_key=staged_key,
        task_raw=task_raw,
        config_raw=config_raw,
        validator_raw=validator_raw,
    )
    paths = module.CacheSeedLocalAuthorityPaths(
        descriptor=_write(tmp_path / "cache-descriptor.json", descriptor_raw),
        approval=_write(tmp_path / "cache-approval.json", approval_raw),
        staged_ready=_write(tmp_path / "cache-staged.json", staged_raw),
        rehearsal_evidence=_write(
            tmp_path / "cache-rehearsal.json",
            _file_bytes(rehearsal),
        ),
        task=_write(tmp_path / "cache-task.yaml", task_raw),
        config=_write(tmp_path / "cache-config.yaml", config_raw),
    )
    remote = {
        (BUCKET, str(descriptor["campaign_descriptor_key"])): descriptor_raw,
        (BUCKET, str(descriptor["approval_key"])): approval_raw,
        (BUCKET, str(descriptor["repo_tar_key"])): repo_raw,
        (BUCKET, str(descriptor["artifacts"]["training_config_key"])): (
            TRAINING_CONFIG_RAW
        ),
        (BUCKET, f"campaigns/{RUN_ID}/authorities/watchdog.sh"): WATCHDOG_RAW,
        (BUCKET, str(descriptor["artifacts"]["artifact_inventory_key"])): (
            inventory_raw
        ),
        (BUCKET, str(staged["artifact_audit_key"])): audit_raw,
        (BUCKET, manifest_key): manifest_raw,
        (BUCKET, staged_key): staged_raw,
        (BUCKET, rehearsal_key): _file_bytes(rehearsal),
    }
    versioned_raw = {
        "repository_tar": (str(descriptor["repo_tar_key"]), repo_raw),
        "approval": (str(descriptor["approval_key"]), approval_raw),
        "training_config": (
            str(descriptor["artifacts"]["training_config_key"]),
            TRAINING_CONFIG_RAW,
        ),
        "watchdog": (
            f"campaigns/{RUN_ID}/authorities/watchdog.sh",
            WATCHDOG_RAW,
        ),
        "artifact_inventory": (
            str(descriptor["artifacts"]["artifact_inventory_key"]),
            inventory_raw,
        ),
        "artifact_audit": (str(staged["artifact_audit_key"]), audit_raw),
        "bundle_manifest": (manifest_key, manifest_raw),
        "descriptor": (
            str(descriptor["campaign_descriptor_key"]),
            descriptor_raw,
        ),
    }
    versions = {
        (BUCKET, key, version_ids[role]): raw
        for role, (key, raw) in versioned_raw.items()
    }
    versions[(BUCKET, staged_key, "version-staged-ready")] = staged_raw
    return CacheSeedFixture(
        paths=paths,
        values={
            "approval": approval,
            "descriptor": descriptor,
            "descriptor_raw": descriptor_raw,
            "staged": staged,
            "staged_key": staged_key,
            "manifest": manifest,
            "manifest_key": manifest_key,
            "rehearsal": rehearsal,
            "rehearsal_key": rehearsal_key,
            "version_ids": version_ids,
        },
        remote=remote,
        versions=versions,
    )


class FakeAwsError(Exception):
    def __init__(self, code: str, status: int, message: str = "fake AWS error"):
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class ProcessCrash(BaseException):
    """Injected process death that ordinary error handling must not absorb."""


class FakeBody:
    def __init__(self, raw: bytes):
        self.raw = raw

    def read(self) -> bytes:
        return self.raw


class FakeSts:
    def __init__(self, events: list[str]):
        self.events = events

    def get_caller_identity(self) -> dict[str, str]:
        self.events.append("sts:get-caller-identity")
        return {
            "Account": ACCOUNT_ID,
            "Arn": f"arn:aws:iam::{ACCOUNT_ID}:role/test",
            "UserId": "test",
        }


class FakeS3:
    def __init__(
        self,
        objects: dict[tuple[str, str], bytes],
        events: list[str],
        versions: dict[tuple[str, str, str], bytes] | None = None,
    ):
        self.objects = dict(objects)
        self.versions = {} if versions is None else dict(versions)
        self.events = events
        self.version_response_overrides: dict[tuple[str, str], object] = {}
        self.put_modes: dict[str, str] = {}
        self.crash_contains: dict[str, BaseException] = {}
        self.after_put: dict[str, Any] = {}
        self.after_list: dict[str, Any] = {}
        self.after_get: dict[str, Any] = {}
        self.page_size: int | None = None

    def get_object(
        self,
        *,
        Bucket: str,
        Key: str,
        VersionId: str | None = None,
    ) -> dict[str, object]:
        if VersionId is None:
            self.events.append(f"s3:get:{Key}")
        else:
            self.events.append(f"s3:get-version:{Key}:{VersionId}")
        try:
            raw = (
                self.objects[(Bucket, Key)]
                if VersionId is None
                else self.versions[(Bucket, Key, VersionId)]
            )
        except KeyError as error:
            raise FakeAwsError("NoSuchKey", 404) from error
        response = {
            "Body": FakeBody(raw),
            "ContentLength": len(raw),
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(_sha(raw))).decode(
                "ascii"
            ),
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
        if VersionId is not None:
            response["VersionId"] = self.version_response_overrides.get(
                (Key, VersionId),
                VersionId,
            )
        hook = self.after_get.get(Key)
        if hook is not None:
            hook()
        return response

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        ContinuationToken: str | None = None,
    ) -> dict[str, object]:
        self.events.append(f"s3:list:{Prefix}")
        keys = sorted(
            key
            for candidate_bucket, key in self.objects
            if candidate_bucket == Bucket and key.startswith(Prefix)
        )
        offset = 0 if ContinuationToken is None else int(ContinuationToken)
        limit = len(keys) if self.page_size is None else self.page_size
        page = keys[offset : offset + limit]
        next_offset = offset + len(page)
        truncated = next_offset < len(keys)
        response = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": truncated,
            "KeyCount": len(page),
            "ResponseMetadata": {"HTTPStatusCode": 200},
            **({"NextContinuationToken": str(next_offset)} if truncated else {}),
        }
        hook = self.after_list.get(Prefix)
        if hook is not None:
            hook()
        return response

    def put_object(self, **kwargs: object) -> dict[str, object]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        body = kwargs["Body"]
        assert isinstance(body, bytes)
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["ChecksumAlgorithm"] == "SHA256"
        assert kwargs["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(body).digest()
        ).decode("ascii")
        metadata = kwargs["Metadata"]
        assert isinstance(metadata, dict)
        assert metadata["glm52-run-id"] == RUN_ID
        self.events.append(f"s3:put:{key}")
        mode = self.put_modes.get(key)
        if (bucket, key) in self.objects or mode == "precondition":
            raise FakeAwsError("PreconditionFailed", 412)
        self.objects[(bucket, key)] = body
        hook = self.after_put.get(key)
        if hook is not None:
            hook()
        for fragment, crash in self.crash_contains.items():
            if fragment in key:
                raise crash
        if mode == "race":
            raise FakeAwsError("PreconditionFailed", 412)
        if mode == "lost":
            raise TimeoutError("lost conditional-put response")
        if mode == "malformed":
            return {}
        return {"ETag": '"fake"', "ResponseMetadata": {"HTTPStatusCode": 200}}


class CacheSeedS3(FakeS3):
    def __init__(
        self,
        objects: dict[tuple[str, str], bytes],
        events: list[str],
        versions: dict[tuple[str, str, str], bytes],
    ):
        super().__init__(objects, events, versions)
        self.claim_versions: list[tuple[str, str, bytes, bool]] = []
        self.claim_put_calls: list[dict[str, object]] = []
        self.claim_put_mode = "normal"
        self.version_page_size: int | None = None
        self.delete_markers: list[dict[str, object]] = []
        self.claim_get_mode: str | None = None
        self.current_claim_version_override: str | None = None
        self.explode_get_keys: set[str] = set()
        self.version_list_mode: str | None = None
        self.version_list_calls = 0
        self.after_version_list: dict[int, Any] = {}
        self.claim_get_calls = 0
        self.after_claim_get: dict[int, Any] = {}
        self.absence_code = "NoSuchKey"
        self.absence_status = 404

    def get_object(
        self,
        *,
        Bucket: str,
        Key: str,
        VersionId: str | None = None,
    ) -> dict[str, object]:
        if Key in self.explode_get_keys:
            self.events.append(f"s3:exploding-reference-get:{Key}")
            raise ProcessCrash(f"forged claim reached referenced GET: {Key}")
        claim = next(
            (
                (version_id, raw)
                for key, version_id, raw, _is_latest in self.claim_versions
                if key == Key and (VersionId is None or version_id == VersionId)
            ),
            None,
        )
        if claim is None or "/launch-claims/" not in Key:
            if "/launch-claims/" in Key:
                self.events.append(
                    f"s3:get-claim:{Key}:{VersionId or 'current'}"
                )
                raise FakeAwsError(self.absence_code, self.absence_status)
            return super().get_object(
                Bucket=Bucket,
                Key=Key,
                VersionId=VersionId,
            )
        version_id, raw = claim
        self.claim_get_calls += 1
        self.events.append(
            f"s3:get-claim:{Key}:{VersionId if VersionId is not None else 'current'}"
        )
        response: dict[str, object] = {
            "Body": FakeBody(raw),
            "ContentLength": len(raw),
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(raw).digest()).decode(
                "ascii"
            ),
            "VersionId": version_id,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
        if VersionId is None and self.current_claim_version_override is not None:
            response["VersionId"] = self.current_claim_version_override
        if self.claim_get_mode == "missing-version":
            response.pop("VersionId")
        elif self.claim_get_mode == "wrong-version":
            response["VersionId"] = "wrong-claim-version"
        elif self.claim_get_mode == "missing-status":
            response["ResponseMetadata"] = {}
        elif self.claim_get_mode == "non200":
            response["ResponseMetadata"] = {"HTTPStatusCode": 206}
        hook = self.after_claim_get.get(self.claim_get_calls)
        if hook is not None:
            hook()
        return response

    def list_object_versions(
        self,
        *,
        Bucket: str,
        Prefix: str,
        KeyMarker: str | None = None,
        VersionIdMarker: str | None = None,
    ) -> dict[str, object]:
        del Bucket
        self.version_list_calls += 1
        self.events.append(
            "s3:list-versions:"
            f"{Prefix}:{KeyMarker or '-'}:{VersionIdMarker or '-'}"
        )
        rows = [
            (key, version_id, raw, is_latest)
            for key, version_id, raw, is_latest in self.claim_versions
            if key.startswith(Prefix)
        ]
        offset = 0 if KeyMarker is None else int(KeyMarker)
        limit = len(rows) if self.version_page_size is None else self.version_page_size
        page = rows[offset : offset + limit]
        next_offset = offset + len(page)
        truncated = next_offset < len(rows)
        response: dict[str, object] = {
            "Versions": [
                {
                    "Key": key,
                    "VersionId": version_id,
                    "IsLatest": is_latest,
                    "Size": len(raw),
                }
                for key, version_id, raw, is_latest in page
            ],
            "DeleteMarkers": deepcopy(self.delete_markers),
            "IsTruncated": truncated,
            "ResponseMetadata": {"HTTPStatusCode": 200},
            **(
                {
                    "NextKeyMarker": str(next_offset),
                    "NextVersionIdMarker": f"marker-{next_offset}",
                }
                if truncated
                else {}
            ),
        }
        if truncated and self.version_list_mode == "partial-key-only":
            response.pop("NextVersionIdMarker", None)
        elif truncated and self.version_list_mode == "partial-version-only":
            response.pop("NextKeyMarker", None)
        elif truncated and self.version_list_mode == "missing-pair":
            response.pop("NextKeyMarker", None)
            response.pop("NextVersionIdMarker", None)
        elif (
            truncated
            and self.version_list_mode == "repeated-pair"
            and KeyMarker is not None
        ):
            response["NextKeyMarker"] = KeyMarker
            response["NextVersionIdMarker"] = VersionIdMarker
        elif not truncated and self.version_list_mode == "foreign-terminal-pair":
            response["NextKeyMarker"] = "foreign-terminal-key"
            response["NextVersionIdMarker"] = "foreign-terminal-version"
        hook = self.after_version_list.get(self.version_list_calls)
        if hook is not None:
            hook()
        return response

    def put_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        if "/launch-claims/" not in key:
            return super().put_object(**kwargs)
        self.claim_put_calls.append(dict(kwargs))
        self.events.append(f"s3:claim-put:{key}")
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["ChecksumAlgorithm"] == "SHA256"
        body = kwargs["Body"]
        assert isinstance(body, bytes)
        assert kwargs["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(body).digest()
        ).decode("ascii")
        if self.claim_put_mode == "precondition":
            raise FakeAwsError("PreconditionFailed", 412)
        if self.claim_put_mode == "conflict":
            raise FakeAwsError("Conflict", 409)
        if self.claim_put_mode == "server":
            raise FakeAwsError("InternalError", 500)
        if self.claim_put_mode == "lost":
            raise TimeoutError("lost claim PUT response")
        if self.claim_put_mode == "malformed":
            return {}
        if self.claim_put_mode == "nonmapping-absent":
            return []  # type: ignore[return-value]
        version_id = "claim-version-1"
        self.claim_versions = [(key, version_id, body, True)]
        if self.claim_put_mode == "nonmapping-after-write":
            return []  # type: ignore[return-value]
        if self.claim_put_mode == "nonmapping-foreign":
            self.claim_versions = [(key, version_id, b"{}\n", True)]
            return []  # type: ignore[return-value]
        if self.claim_put_mode == "lost-after-write":
            raise TimeoutError("lost claim PUT response after write")
        if self.claim_put_mode == "mismatched-version":
            return {
                "VersionId": "claim-version-2",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        return {
            "VersionId": version_id,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


def test_get_exact_with_version_uses_exact_boto_args_and_matching_response() -> None:
    module = _module()
    calls: list[dict[str, object]] = []
    raw = b"exact versioned object\n"

    class ExactS3:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {
                "Body": FakeBody(raw),
                "ContentLength": len(raw),
                "ChecksumSHA256": base64.b64encode(
                    hashlib.sha256(raw).digest()
                ).decode("ascii"),
                "VersionId": "version-1",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

    assert module._get_exact(
        ExactS3(),
        bucket="bucket",
        key="campaigns/run/object.json",
        version_id="version-1",
    ) == raw
    assert calls == [
        {
            "Bucket": "bucket",
            "Key": "campaigns/run/object.json",
            "VersionId": "version-1",
        }
    ]


@pytest.mark.parametrize(
    ("response_version", "message"),
    (
        ("missing", "VersionId"),
        ("", "VersionId"),
        (None, "VersionId"),
        ("null", "VersionId"),
        ("wrong-version", "VersionId"),
    ),
)
def test_get_exact_with_version_rejects_missing_or_wrong_response_version(
    response_version: object,
    message: str,
) -> None:
    module = _module()
    raw = b"exact versioned object\n"

    class ExactS3:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["VersionId"] == "version-1"
            response: dict[str, object] = {
                "Body": FakeBody(raw),
                "ContentLength": len(raw),
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
            if response_version != "missing":
                response["VersionId"] = response_version
            return response

    with pytest.raises(module.SubmissionIntegrationError, match=message):
        module._get_exact(
            ExactS3(),
            bucket="bucket",
            key="campaigns/run/object.json",
            version_id="version-1",
        )


def test_get_exact_with_version_preserves_missing_key_translation() -> None:
    module = _module()

    class MissingS3:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["VersionId"] == "version-1"
            raise FakeAwsError("NoSuchKey", 404)

    assert (
        module._get_exact(
            MissingS3(),
            bucket="bucket",
            key="campaigns/run/object.json",
            version_id="version-1",
        )
        is None
    )


@pytest.mark.parametrize("corruption", ["service-error", "length", "checksum"])
def test_get_exact_with_version_preserves_error_length_and_checksum_failures(
    corruption: str,
) -> None:
    module = _module()
    raw = b"versioned body\n"

    class CorruptS3:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["VersionId"] == "version-1"
            if corruption == "service-error":
                raise FakeAwsError("AccessDenied", 403)
            return {
                "Body": FakeBody(raw),
                "ContentLength": (
                    len(raw) + 1 if corruption == "length" else len(raw)
                ),
                "ChecksumSHA256": (
                    base64.b64encode(b"x" * 32).decode("ascii")
                    if corruption == "checksum"
                    else base64.b64encode(hashlib.sha256(raw).digest()).decode(
                        "ascii"
                    )
                ),
                "VersionId": "version-1",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

    with pytest.raises(module.SubmissionIntegrationError):
        module._get_exact(
            CorruptS3(),
            bucket="bucket",
            key="campaigns/run/object.json",
            version_id="version-1",
        )


def _tags(values: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": key, "Value": value} for key, value in values.items()]


class FakeEc2:
    def __init__(
        self,
        events: list[str],
        *,
        allocation_history: dict[str, object],
    ):
        self.events = events
        self.active_p5_ids: list[str] = []
        self.active_p5_states: list[str] = []
        self.allocation_history = deepcopy(allocation_history)
        self.requests: list[dict[str, object]] = []

    def describe_instances(
        self,
        *,
        Filters: list[dict[str, object]] | None = None,
        InstanceIds: list[str] | None = None,
        NextToken: str | None = None,
    ) -> dict[str, object]:
        self.requests.append(
            {
                "Filters": deepcopy(Filters),
                "InstanceIds": deepcopy(InstanceIds),
                "NextToken": NextToken,
            }
        )
        assert NextToken is None
        if InstanceIds is not None:
            assert InstanceIds == [CONTROLLER_INSTANCE_ID]
            kind = "controller"
        else:
            names = {str(item["Name"]) for item in Filters or []}
            kind = (
                "spend-history"
                if "instance-type" in names and "instance-state-name" not in names
                else "active-p5"
                if "instance-type" in names
                or "tag:cost-allocation" in names
                and "tag:ray-cluster-name" not in names
                else "controller"
            )
        self.events.append(f"ec2:describe:{kind}")
        if kind == "spend-history":
            return deepcopy(self.allocation_history)
        if kind == "active-p5":
            instances = [
                {
                    "InstanceId": instance_id,
                    "InstanceType": "p5.48xlarge",
                    "State": {"Name": state},
                    "Tags": _tags(
                        {
                            "project": "keep-glm52",
                            "owner": "jack.mazac",
                            "model": "glm-5.2",
                            "campaign-run-id": RUN_ID,
                            "cost-allocation": "glm52-sky-campaign",
                        }
                    ),
                }
                for instance_id, state in (
                    [
                        (instance_id, "running")
                        for instance_id in self.active_p5_ids
                    ]
                    + [
                        (f"i-active-{index}", state)
                        for index, state in enumerate(self.active_p5_states)
                    ]
                )
            ]
        else:
            instances = [
                {
                    "InstanceId": CONTROLLER_INSTANCE_ID,
                    "InstanceType": CONTROLLER_INSTANCE_TYPE,
                    "State": {"Name": "running"},
                    "IamInstanceProfile": {"Arn": CONTROLLER_PROFILE_ARN},
                    "Placement": {"AvailabilityZone": "us-west-2a"},
                    "Tags": _tags(
                        {
                            "project": "keep-glm52",
                            "owner": "jack.mazac",
                            "model": "glm-5.2",
                            "campaign-run-id": RUN_ID,
                            "cost-allocation": "glm52-sky-campaign",
                            "ray-cluster-name": CONTROLLER_CLUSTER_NAME,
                        }
                    ),
                }
            ]
        return {
            "Reservations": [{"Instances": instances}] if instances else [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


class FakeClock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _history_row(
    *,
    job_id: int,
    status: str,
    submitted_at: datetime,
    schedule_state: str = "ALIVE",
    start_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "sky_job_id": job_id,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": _iso(submitted_at),
        "controller_status": status,
        "controller_identity": CONTROLLER_ROLE_ARN,
        "schedule_state": schedule_state,
        "start_at": None if start_at is None else _iso(start_at),
        "worker_cluster_name": None,
        "recovery_count": 0,
    }


class FakeSsm:
    def __init__(self, events: list[str], clock: FakeClock):
        self.events = events
        self.clock = clock
        self.rows: list[dict[str, object]] = []

    def describe_instance_information(
        self,
        *,
        Filters: list[dict[str, object]],
    ) -> dict[str, object]:
        assert Filters == [{"Key": "InstanceIds", "Values": [CONTROLLER_INSTANCE_ID]}]
        self.events.append("ssm:describe-controller")
        return {
            "InstanceInformationList": [
                {
                    "InstanceId": CONTROLLER_INSTANCE_ID,
                    "PingStatus": "Online",
                }
            ]
        }

    def send_command(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["InstanceIds"] == [CONTROLLER_INSTANCE_ID]
        assert kwargs["DocumentName"] == "AWS-RunShellScript"
        self.events.append("ssm:send-observe")
        return {"Command": {"CommandId": "command-1"}}

    def get_command_invocation(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "CommandId": "command-1",
            "InstanceId": CONTROLLER_INSTANCE_ID,
        }
        self.events.append("ssm:get-observe")
        result = {
            "schema_version": 1,
            "record_type": "glm52_sky_controller_history_v1",
            "sky_job_name": SKY_JOB_NAME,
            "workspace": "default",
            "rows": self.rows,
            "observed_at": _iso(self.clock()),
        }
        return {
            "Status": "Success",
            "StandardOutputContent": (
                "GLM52_SUBMISSION_HISTORY=" + _canonical(result).decode("ascii") + "\n"
            ),
            "StandardErrorContent": "",
        }


class FakeCloudFormation:
    def __init__(self, events: list[str], clock: FakeClock):
        self.events = events
        self.clock = clock
        self.active = True
        self.identity = _deployment_identity()
        self.requests: list[dict[str, str]] = []

    def inspect_dynamic_must_start(
        self,
        *,
        run_id: str,
        managed_mode: str,
        intent_body_sha256: str,
        controller_baseline_body_sha256: str,
    ) -> dict[str, object] | None:
        assert run_id == RUN_ID
        assert managed_mode == "qualification"
        assert len(intent_body_sha256) == 64
        assert len(controller_baseline_body_sha256) == 64
        self.requests.append(
            {
                "run_id": run_id,
                "managed_mode": managed_mode,
                "intent_body_sha256": intent_body_sha256,
                "controller_baseline_body_sha256": (controller_baseline_body_sha256),
            }
        )
        self.events.append("cfn:inspect-must-start")
        if not self.active:
            return None
        return {
            "reviewed_deployment_identity": deepcopy(self.identity),
            "observed_deployment_identity": deepcopy(self.identity),
            "reconciliation_rule_state": "ENABLED",
            "deadline_schedule_state": "ENABLED",
            "coordinator_mode": "dynamic-job-binding-active",
            "observed_at": _iso(self.clock()),
        }


class FakeSky:
    def __init__(
        self,
        events: list[str],
        clock: FakeClock,
        ssm: FakeSsm,
    ):
        self.events = events
        self.clock = clock
        self.ssm = ssm
        self.launches: list[list[str]] = []
        self.launch_cwds: list[Path | None] = []
        self.launch_snapshots: list[dict[str, object]] = []
        self.launch_result: object = 0
        self.add_controller_row = True
        self.extra_controller_rows = 0
        self.row_overrides: dict[str, object] = {}

    def validate_control_plane(self, *, task: Path, config: Path) -> dict[str, object]:
        self.events.append("sky:validate")
        return {
            "skypilot_version": "0.13.0",
            "task_file_sha256": _sha(task.read_bytes()),
            "config_file_sha256": _sha(config.read_bytes()),
        }

    def launch(self, argv: list[str], *, cwd: Path | None = None) -> object:
        config_path = Path(argv[argv.index("--config") + 1])
        task_path = Path(argv[-1])
        mount_root = Path.cwd() if cwd is None else cwd
        observed_paths = {
            "config": config_path,
            "task": task_path,
            **{
                relative_path: mount_root / relative_path
                for relative_path in SEALED_MOUNT_RELATIVE_PATHS
            },
        }
        observed_raw = {
            name: path.read_bytes() for name, path in observed_paths.items()
        }
        observed_stats = {name: os.lstat(path) for name, path in observed_paths.items()}
        self.events.append("sky:launch")
        self.launches.append(list(argv))
        self.launch_cwds.append(cwd)
        self.launch_snapshots.append(
            {
                "paths": observed_paths,
                "raw": observed_raw,
                "modes": {
                    name: stat.S_IMODE(value.st_mode)
                    for name, value in observed_stats.items()
                },
                "links": {
                    name: value.st_nlink for name, value in observed_stats.items()
                },
                "regular": {
                    name: stat.S_ISREG(value.st_mode)
                    for name, value in observed_stats.items()
                },
            }
        )
        self.clock.advance(2)
        if self.add_controller_row:
            rows = [
                {
                    **_history_row(
                        job_id=17 + index,
                        status="PENDING",
                        submitted_at=self.clock(),
                        schedule_state="ALIVE",
                    ),
                    **self.row_overrides,
                }
                for index in range(self.extra_controller_rows + 1)
            ]
            self.ssm.rows.extend(rows)
        if isinstance(self.launch_result, BaseException):
            raise self.launch_result
        return self.launch_result


class CacheSeedCloudFormation:
    def __init__(self, events: list[str], clock: FakeClock):
        self.events = events
        self.clock = clock
        self.active = True
        self.overrides: dict[str, object] = {}
        self.missing_fields: set[str] = set()
        self.requests: list[dict[str, object]] = []
        self.request_hooks: dict[int, Any] = {}

    def inspect_cache_seed_launch_capability(
        self,
        *,
        run_id: str,
        managed_mode: str,
        descriptor_file_sha256: str,
        descriptor_body_sha256: str,
        staged_readiness_version_id: str,
    ) -> dict[str, object] | None:
        request = {
            "run_id": run_id,
            "managed_mode": managed_mode,
            "descriptor_file_sha256": descriptor_file_sha256,
            "descriptor_body_sha256": descriptor_body_sha256,
            "staged_readiness_version_id": staged_readiness_version_id,
        }
        self.requests.append(request)
        self.events.append("cfn:inspect-cache-seed")
        hook = self.request_hooks.get(len(self.requests))
        if hook is not None:
            hook()
        if not self.active:
            return None
        prefix = (
            f"campaigns/{run_id}/submissions/cache-seed/launch-claims/"
            f"{descriptor_file_sha256}/"
        )
        capability = {
            **request,
            "worker_runtime_mode": "cache-seed-v1",
            "launch_route_state": "ENABLED",
            "bucket": BUCKET,
            "launch_claim_prefix": prefix,
            "claim_bucket_versioning_state": "ENABLED",
            "claim_list_bucket_versions_state": "ENABLED",
            "claim_conditional_put_enforcement_state": "ENFORCED",
            "claim_lifecycle_protection_state": "ENFORCED",
            "claim_delete_object_deny_state": "ENFORCED",
            "claim_delete_object_version_deny_state": "ENFORCED",
            "observed_at": _iso(self.clock()),
            **self.overrides,
        }
        for field in self.missing_fields:
            capability.pop(field, None)
        return capability


class CacheSeedSky:
    def __init__(self, events: list[str]):
        self.events = events
        self.launches: list[list[str]] = []
        self.launch_cwds: list[Path] = []
        self.launch_snapshots: list[dict[str, bytes]] = []

    def validate_control_plane(self, *, task: Path, config: Path) -> dict[str, object]:
        self.events.append("sky:validate-cache-seed")
        return {
            "skypilot_version": "0.13.0",
            "task_file_sha256": _sha(task.read_bytes()),
            "config_file_sha256": _sha(config.read_bytes()),
        }

    def launch(self, argv: list[str], *, cwd: Path) -> int:
        self.events.append("sky:launch-cache-seed")
        self.launches.append(list(argv))
        self.launch_cwds.append(cwd)
        config_path = Path(argv[argv.index("--config") + 1])
        task_path = Path(argv[-1])
        assert config_path.is_file()
        assert task_path.is_file()
        assert stat.S_IMODE(config_path.stat().st_mode) in {0o400, 0o600}
        assert stat.S_IMODE(task_path.stat().st_mode) in {0o400, 0o600}
        self.launch_snapshots.append(
            {
                "task": task_path.read_bytes(),
                "config": config_path.read_bytes(),
                **{
                    relative_path: (cwd / relative_path).read_bytes()
                    for relative_path in SEALED_MOUNT_RELATIVE_PATHS
                },
            }
        )
        return 0


@dataclass
class ServiceFixture:
    services: Any
    events: list[str]
    s3: FakeS3
    ec2: FakeEc2
    ssm: FakeSsm
    cfn: FakeCloudFormation
    sky: FakeSky
    clock: FakeClock


@dataclass
class CacheSeedServiceFixture:
    services: Any
    events: list[str]
    s3: CacheSeedS3
    ec2: FakeEc2
    cfn: CacheSeedCloudFormation
    sky: CacheSeedSky
    clock: FakeClock


def _services(fixture: Fixture) -> ServiceFixture:
    module = _module()
    events: list[str] = []
    clock = FakeClock()
    s3 = FakeS3(fixture.remote, events, fixture.versions)
    ec2_history = fixture.values["ec2_history"]
    assert isinstance(ec2_history, dict)
    ec2 = FakeEc2(events, allocation_history=ec2_history)
    ssm = FakeSsm(events, clock)
    cfn = FakeCloudFormation(events, clock)
    sky = FakeSky(events, clock, ssm)
    return ServiceFixture(
        services=module.SubmissionServices(
            sts=FakeSts(events),
            s3=s3,
            ec2=ec2,
            ssm=ssm,
            cloudformation=cfn,
            sky=sky,
            clock=clock,
            sleep=lambda _seconds: None,
        ),
        events=events,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        cfn=cfn,
        sky=sky,
        clock=clock,
    )


def _cache_seed_services(fixture: CacheSeedFixture) -> CacheSeedServiceFixture:
    module = _module()
    events: list[str] = []
    clock = FakeClock()
    s3 = CacheSeedS3(fixture.remote, events, fixture.versions)
    ec2 = FakeEc2(
        events,
        allocation_history={
            "Reservations": [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
    )
    cfn = CacheSeedCloudFormation(events, clock)
    sky = CacheSeedSky(events)
    return CacheSeedServiceFixture(
        services=module.SubmissionServices(
            sts=FakeSts(events),
            s3=s3,
            ec2=ec2,
            ssm=None,
            cloudformation=cfn,
            sky=sky,
            clock=clock,
            sleep=lambda _seconds: None,
            claim_put_s3=s3,
        ),
        events=events,
        s3=s3,
        ec2=ec2,
        cfn=cfn,
        sky=sky,
        clock=clock,
    )


def _request(
    fixture: Fixture,
    *,
    action: str,
    output_handoff: Path | None = None,
    intent_s3_uri: str | None = None,
    must_start_ready_s3_uri: str | None = None,
    acquisition_s3_uri: str | None = None,
) -> Any:
    module = _module()
    return module.QualificationRequest(
        action=action,
        profile=PROFILE,
        local=fixture.paths,
        output_handoff=output_handoff,
        intent_s3_uri=intent_s3_uri,
        must_start_ready_s3_uri=must_start_ready_s3_uri,
        acquisition_s3_uri=acquisition_s3_uri,
        sky_bin="/opt/keep/skypilot-0.13.0/bin/sky",
    )


def _cache_seed_request(
    fixture: CacheSeedFixture,
    *,
    action: str,
    waive_launch_claim_protection: bool = False,
) -> Any:
    module = _module()
    return module.CacheSeedRequest(
        action=action,
        profile=PROFILE,
        local=fixture.paths,
        staged_readiness_version_id="version-staged-ready",
        sky_bin="/opt/keep/skypilot-0.13.0/bin/sky",
        waive_launch_claim_protection=waive_launch_claim_protection,
    )


def test_cache_seed_validate_only_collects_exact_live_observations_without_mutation(
    tmp_path: Path,
) -> None:
    """Catches missing staged-version auth, reordered evidence, or mutation."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "cache-seed-valid"
    latest = f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json"
    legacy = f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"
    records = f"campaigns/{RUN_ID}/runtime/spend-ledger/records/"
    live_events = [
        event
        for event in services.events
        if event
        in {
            f"s3:get:{latest}",
            f"s3:get:{legacy}",
            f"s3:list:{records}",
            "ec2:describe:spend-history",
            "ec2:describe:active-p5",
        }
    ]
    assert live_events == [
        f"s3:get:{latest}",
        f"s3:get:{legacy}",
        f"s3:list:{records}",
        f"s3:get:{latest}",
        f"s3:get:{legacy}",
        "ec2:describe:spend-history",
        "ec2:describe:active-p5",
    ]
    assert any(
        event
        == f"s3:get-version:{fixture.values['staged_key']}:version-staged-ready"
        for event in services.events
    )
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not any(event.startswith("s3:claim-put:") for event in services.events)
    assert "cfn:inspect-cache-seed" not in services.events
    assert "sky:launch-cache-seed" not in services.events
    assert [request["Filters"] for request in services.ec2.requests] == [
        [
            {"Name": "instance-type", "Values": ["p5.48xlarge"]},
            {"Name": "tag:campaign-run-id", "Values": [RUN_ID]},
            {
                "Name": "tag:cost-allocation",
                "Values": ["glm52-sky-campaign"],
            },
            {"Name": "tag:model", "Values": ["glm-5.2"]},
            {"Name": "tag:owner", "Values": ["jack.mazac"]},
            {"Name": "tag:project", "Values": ["keep-glm52"]},
        ],
        [
            {"Name": "instance-type", "Values": ["p5.48xlarge"]},
            {"Name": "tag:campaign-run-id", "Values": [RUN_ID]},
            {
                "Name": "tag:cost-allocation",
                "Values": ["glm52-sky-campaign"],
            },
            {"Name": "tag:model", "Values": ["glm-5.2"]},
            {"Name": "tag:owner", "Values": ["jack.mazac"]},
            {"Name": "tag:project", "Values": ["keep-glm52"]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "shutting-down", "stopping"],
            },
        ],
    ]


def test_cache_seed_default_capability_is_deployment_required_before_live_work(
    tmp_path: Path,
) -> None:
    """Catches real/default acquisition collecting or publishing before deploy."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.active = False

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 78
    assert outcome.status == "deployment-required"
    assert any(event.startswith("s3:list-versions:") for event in services.events)
    assert "cfn:inspect-cache-seed" in services.events
    assert not any(
        "/runtime/GPU_SPEND_LEDGER" in event for event in services.events
    )
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


def test_cache_seed_explicit_waiver_keeps_claim_and_launches_without_capability(
    tmp_path: Path,
) -> None:
    """Catches a waiver that still blocks or bypasses the one-shot claim."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.active = False

    outcome = module.run_cache_seed(
        _cache_seed_request(
            fixture,
            action="acquire-and-launch",
            waive_launch_claim_protection=True,
        ),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "cache-seed-submitted"
    assert "cfn:inspect-cache-seed" not in services.events
    assert len(services.s3.claim_put_calls) == 1
    assert len(services.sky.launches) == 1


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("managed_mode", "qualification"),
        ("worker_runtime_mode", "qualification-v1"),
        ("launch_route_state", "DISABLED"),
        ("claim_bucket_versioning_state", "SUSPENDED"),
        ("claim_list_bucket_versions_state", "ROLE_LOCAL"),
        ("claim_conditional_put_enforcement_state", "IMPLICIT"),
        ("claim_lifecycle_protection_state", "UNPROVEN"),
        ("claim_delete_object_deny_state", "ROLE_LOCAL"),
        ("claim_delete_object_version_deny_state", "IMPLICIT"),
        ("bucket", "foreign-bucket"),
        ("launch_claim_prefix", "campaigns/foreign/"),
        ("staged_readiness_version_id", "newer-version"),
    ],
)
def test_cache_seed_rejects_each_capability_drift_before_publication(
    tmp_path: Path,
    field: str,
    wrong: object,
) -> None:
    """Catches any non-universal or identity-drifted deployment capability."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.overrides[field] = wrong

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not any(
        "/runtime/GPU_SPEND_LEDGER" in event for event in services.events
    )
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    "missing_field",
    [
        "run_id",
        "managed_mode",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "staged_readiness_version_id",
        "worker_runtime_mode",
        "launch_route_state",
        "bucket",
        "launch_claim_prefix",
        "claim_bucket_versioning_state",
        "claim_list_bucket_versions_state",
        "claim_conditional_put_enforcement_state",
        "claim_lifecycle_protection_state",
        "claim_delete_object_deny_state",
        "claim_delete_object_version_deny_state",
        "observed_at",
    ],
)
def test_cache_seed_rejects_each_missing_capability_field_before_publication(
    tmp_path: Path,
    missing_field: str,
) -> None:
    """Catches defaulting any omitted deployment-capability authority."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.missing_fields.add(missing_field)

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    "unproven_route",
    ["MULTIPART_ROUTE_UNPROVEN", "COPY_ROUTE_UNPROVEN"],
)
def test_conditional_put_capability_separately_requires_multipart_and_copy_denial(
    tmp_path: Path,
    unproven_route: str,
) -> None:
    """Catches treating If-None-Match as proof against multipart or copy writes."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.overrides["claim_conditional_put_enforcement_state"] = (
        unproven_route
    )

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    ("seconds", "expected_reason"),
    [
        (60, "stale"),
        (-1, "future"),
    ],
)
def test_cache_seed_capability_rejects_exact_expiry_and_future_observation(
    tmp_path: Path,
    seconds: int,
    expected_reason: str,
) -> None:
    """Catches inclusive expiry or a future-dated deployment capability."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.overrides["observed_at"] = _iso(
        services.clock() - timedelta(seconds=seconds)
    )

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert expected_reason in str(outcome.detail["reason"])
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


def test_cache_seed_success_owns_one_versioned_claim_and_launches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches retrying claim PUT, weak readback, or launch-before-final-gate."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    original_validate = module.validate_gpu_launch_allowance

    def recording_validate(*args: object, **kwargs: object) -> object:
        services.events.append("allowance:validate")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "validate_gpu_launch_allowance",
        recording_validate,
    )

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "cache-seed-submitted"
    assert len(services.s3.claim_put_calls) == 1
    assert len(services.sky.launches) == 1
    assert sum(event.startswith("s3:list-versions:") for event in services.events) >= 3
    claim_reads = [
        event for event in services.events if event.startswith("s3:get-claim:")
    ]
    assert any(event.endswith(":claim-version-1") for event in claim_reads)
    assert any(event.endswith(":current") for event in claim_reads)
    assert services.events[-2:] == [
        "allowance:validate",
        "sky:launch-cache-seed",
    ]
    argv = services.sky.launches[0]
    assert argv[:3] == ["jobs", "launch", "--detach-run"]
    assert "--async" not in argv
    assert "--no-use-spot" in argv
    assert argv.count("--config") == 1
    assert "--image-id" in argv
    assert "GLM52_MANAGED_MODE=cache-seed" in argv
    assert not any("AWS_" in item or "credential" in item.lower() for item in argv)


def test_preexisting_valid_claim_is_reconciliation_only_without_fresh_work(
    tmp_path: Path,
) -> None:
    """Catches restart replay of a descriptor whose durable claim already won."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    first = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )
    assert first.exit_code == 0
    launch_count = len(services.sky.launches)
    put_count = len(services.s3.claim_put_calls)
    event_count = len(services.events)

    second = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert second.exit_code == 75
    assert second.status == "launch-claim-reconciliation-only"
    assert len(services.sky.launches) == launch_count
    assert len(services.s3.claim_put_calls) == put_count
    retry_events = services.events[event_count:]
    assert not any(
        event.startswith("s3:list:campaigns/")
        and "/runtime/spend-ledger/records/" in event
        for event in retry_events
    )
    assert "cfn:inspect-cache-seed" not in retry_events


@pytest.mark.parametrize("forgery", ["bad-self-hash", "foreign-derived-key"])
def test_forged_preexisting_claim_is_rejected_before_any_referenced_get(
    tmp_path: Path,
    forgery: str,
) -> None:
    """Catches fetching claim references before claim bytes and keys authenticate."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    first = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )
    assert first.exit_code == 0
    key, version_id, raw, is_latest = services.s3.claim_versions[0]
    claim = json.loads(raw)
    if forgery == "bad-self-hash":
        claim["launch_claim_body_sha256"] = "f" * 64
    else:
        claim["submission_intent_key"] = (
            f"campaigns/{RUN_ID}/submissions/cache-seed/intents/"
            f"{'e' * 64}/SKYPILOT_SUBMISSION_INTENT.json"
        )
        claim = _rehash(claim, "launch_claim_body_sha256")
    forged_raw = _file_bytes(claim)
    services.s3.claim_versions = [(key, version_id, forged_raw, is_latest)]
    referenced_keys = {
        str(claim[field])
        for field in (
            "submission_intent_key",
            "launch_allowance_key",
            "s3_spend_ledger_absence_observation_key",
            "ec2_tagged_p5_zero_inventory_observation_key",
        )
    }
    services.s3.explode_get_keys = referenced_keys
    services.events.clear()

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not any(
        event.startswith("s3:exploding-reference-get:")
        for event in services.events
    )
    assert len(services.s3.claim_put_calls) == 1
    assert len(services.sky.launches) == 1


@pytest.mark.parametrize(
    ("mode", "expected_exit"),
    [
        ("lost-after-write", 75),
        ("precondition", 70),
        ("conflict", 70),
        ("server", 70),
        ("lost", 70),
        ("malformed", 70),
        ("mismatched-version", 70),
    ],
)
def test_claim_put_ambiguity_never_retries_or_launches(
    tmp_path: Path,
    mode: str,
    expected_exit: int,
) -> None:
    """Catches a second wire PUT or launch after any ambiguous claim result."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.s3.claim_put_mode = mode

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == expected_exit
    if expected_exit == 75:
        assert outcome.status == "launch-claim-reconciliation-only"
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


@pytest.mark.parametrize(
    ("mode", "expected_exit"),
    [
        ("nonmapping-after-write", 75),
        ("nonmapping-absent", 70),
        ("nonmapping-foreign", 70),
    ],
)
def test_nonmapping_claim_put_response_always_inspects_once_and_never_launches(
    tmp_path: Path,
    mode: str,
    expected_exit: int,
) -> None:
    """Catches skipping reconciliation when claim PUT returns a non-mapping."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.s3.claim_put_mode = mode

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == expected_exit
    if expected_exit == 75:
        assert outcome.status == "launch-claim-reconciliation-only"
    assert len(services.s3.claim_put_calls) == 1
    history_indexes = [
        index
        for index, event in enumerate(services.events)
        if event.startswith("s3:list-versions:")
    ]
    assert len(history_indexes) == 2
    claim_put_index = services.events.index(
        next(event for event in services.events if event.startswith("s3:claim-put:"))
    )
    assert history_indexes[0] < claim_put_index < history_indexes[1]
    assert not services.sky.launches


@pytest.mark.parametrize(
    "corruption",
    [
        "delete-marker",
        "sibling",
        "multiple",
        "null-version",
        "non-latest",
        "absence-alias",
    ],
)
def test_complete_claim_history_corruption_fails_before_capability(
    tmp_path: Path,
    corruption: str,
) -> None:
    """Catches hidden, ambiguous, or non-versioned historical claim state."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    descriptor_raw = fixture.values["descriptor_raw"]
    assert isinstance(descriptor_raw, bytes)
    descriptor_sha = _sha(descriptor_raw)
    key = (
        f"campaigns/{RUN_ID}/submissions/cache-seed/launch-claims/"
        f"{descriptor_sha}/LAUNCH_CLAIM.json"
    )
    if corruption == "delete-marker":
        services.s3.delete_markers = [
            {"Key": key, "VersionId": "deleted-version", "IsLatest": True}
        ]
    elif corruption == "sibling":
        services.s3.claim_versions = [
            (key + ".foreign", "claim-version-1", b"{}\n", True)
        ]
    elif corruption == "multiple":
        services.s3.claim_versions = [
            (key, "claim-version-1", b"{}\n", False),
            (key, "claim-version-2", b"{}\n", True),
        ]
        services.s3.version_page_size = 1
    elif corruption == "null-version":
        services.s3.claim_versions = [(key, "null", b"{}\n", True)]
    elif corruption == "non-latest":
        services.s3.claim_versions = [
            (key, "claim-version-1", b"{}\n", False)
        ]
    else:
        services.s3.absence_code = "NotFound"
        services.s3.absence_status = 404

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "cfn:inspect-cache-seed" not in services.events
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    "marker_corruption",
    [
        "partial-key-only",
        "partial-version-only",
        "missing-pair",
        "repeated-pair",
        "foreign-terminal-pair",
    ],
)
def test_claim_version_history_rejects_every_marker_pair_corruption(
    tmp_path: Path,
    marker_corruption: str,
) -> None:
    """Catches accepting partial, missing, repeated, or terminal-foreign markers."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    descriptor_raw = fixture.values["descriptor_raw"]
    assert isinstance(descriptor_raw, bytes)
    key = (
        f"campaigns/{RUN_ID}/submissions/cache-seed/launch-claims/"
        f"{_sha(descriptor_raw)}/LAUNCH_CLAIM.json"
    )
    row_count = 3 if marker_corruption == "repeated-pair" else 2
    services.s3.claim_versions = [
        (key, f"claim-version-{index}", b"{}\n", index == row_count - 1)
        for index in range(row_count)
    ]
    services.s3.version_page_size = 1
    services.s3.version_list_mode = marker_corruption

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    "mode",
    ["missing-version", "wrong-version", "missing-status", "non200"],
)
def test_claim_exact_and_current_readback_requires_status_and_version(
    tmp_path: Path,
    mode: str,
) -> None:
    """Catches claim ownership from weak exact/current GET responses."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.s3.claim_get_mode = mode

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


def test_current_claim_get_version_must_equal_sole_listed_winner(
    tmp_path: Path,
) -> None:
    """Catches accepting an identical-body current version not listed as winner."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.s3.current_claim_version_override = "identical-body-foreign-current"

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert len(services.s3.claim_put_calls) == 1
    assert any(event.endswith(":claim-version-1") for event in services.events)
    assert any(event.endswith(":current") for event in services.events)
    assert not services.sky.launches


@pytest.mark.parametrize("mutation", ["task", "config"])
def test_post_validation_task_or_config_mutation_cannot_change_cache_seed_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Catches reopening unauthenticated task/config bytes after claim ownership."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    original = {
        "task": fixture.paths.task.read_bytes(),
        "config": fixture.paths.config.read_bytes(),
    }
    target = fixture.paths.task if mutation == "task" else fixture.paths.config
    attacker_raw = f"attacker-controlled-cache-seed-{mutation}\n".encode()
    real_inspect = module._inspect_cache_seed_claim_history
    inspections = 0

    def mutate_after_ownership(*args: object, **kwargs: object) -> object:
        nonlocal inspections
        result = real_inspect(*args, **kwargs)
        inspections += 1
        if inspections == 2:
            target.write_bytes(attacker_raw)
        return result

    monkeypatch.setattr(
        module,
        "_inspect_cache_seed_claim_history",
        mutate_after_ownership,
    )

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert target.read_bytes() == attacker_raw
    if outcome.exit_code == 0:
        assert services.sky.launch_snapshots == [original]
    else:
        assert outcome.exit_code == 70
        assert not services.sky.launches


@pytest.mark.parametrize("mutation", SEALED_MOUNT_RELATIVE_PATHS)
def test_post_validation_mount_source_mutation_blocks_cache_seed_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Catches cache-seed launch from an unauthenticated file-mount source."""

    module = _module()
    source_root = _temporary_worker_source_tree(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", source_root)
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    target = source_root / mutation
    attacker_raw = f"attacker-controlled-cache-seed-mount:{mutation}\n".encode()
    real_inspect_claim = module._inspect_cache_seed_claim_history
    real_read_sources = module._read_worker_start_v2_source_snapshot
    ownership_verified = False
    mutated = False

    def record_verified_ownership(*args: object, **kwargs: object) -> object:
        nonlocal ownership_verified
        result = real_inspect_claim(*args, **kwargs)
        if result is not None:
            ownership_verified = True
        return result

    def mutate_at_post_ownership_source_read(
        *args: object, **kwargs: object
    ) -> object:
        nonlocal mutated
        if ownership_verified and not mutated:
            target.write_bytes(attacker_raw)
            mutated = True
        return real_read_sources(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_inspect_cache_seed_claim_history",
        record_verified_ownership,
    )
    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        mutate_at_post_ownership_source_read,
    )

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert mutated
    assert target.read_bytes() == attacker_raw
    assert outcome.exit_code == 70
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


def test_collection_completion_at_exact_expiry_cannot_relabel_start_time(
    tmp_path: Path,
) -> None:
    """Catches choosing a fresh timestamp after a slow live collection."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    records = f"campaigns/{RUN_ID}/runtime/spend-ledger/records/"
    services.s3.after_list[records] = lambda: services.clock.advance(60)

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


def test_exact_expiry_during_final_capability_recheck_fails_before_launch(
    tmp_path: Path,
) -> None:
    """Catches final authority reads replacing the last allowance freshness gate."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.cfn.request_hooks[2] = lambda: services.clock.advance(60)

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


@pytest.mark.parametrize(
    "final_recheck",
    [
        "version-history",
        "claim-current",
        "intent",
        "staged-chain",
        "capability",
    ],
)
def test_exact_expiry_during_each_final_recheck_reaches_last_allowance_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_recheck: str,
) -> None:
    """Catches any final authority recheck bypassing exclusive allowance expiry."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    original_validate = module.validate_gpu_launch_allowance
    validation_times: list[datetime] = []

    def recording_validate(*args: object, **kwargs: object) -> object:
        now = kwargs.get("now")
        if isinstance(now, datetime):
            validation_times.append(now)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "validate_gpu_launch_allowance",
        recording_validate,
    )
    if final_recheck == "version-history":
        services.s3.after_version_list[3] = lambda: services.clock.advance(60)
    elif final_recheck == "claim-current":
        services.s3.after_claim_get[4] = lambda: services.clock.advance(60)
    elif final_recheck == "intent":
        intent_reads = 0

        def expire_on_final_intent_read() -> None:
            nonlocal intent_reads
            intent_reads += 1
            if intent_reads == 2:
                services.clock.advance(60)

        # The body-addressed intent key is learned from the claim PUT body.
        real_put = services.s3.put_object

        def register_intent_hook(**kwargs: object) -> dict[str, object]:
            response = real_put(**kwargs)
            key = str(kwargs["Key"])
            if key.endswith("/SKYPILOT_SUBMISSION_INTENT.json"):
                services.s3.after_get[key] = expire_on_final_intent_read
            return response

        monkeypatch.setattr(services.s3, "put_object", register_intent_hook)
    elif final_recheck == "staged-chain":
        staged_reads = 0
        staged_key = str(fixture.values["staged_key"])

        def expire_on_final_staged_read() -> None:
            nonlocal staged_reads
            staged_reads += 1
            if staged_reads == 2:
                services.clock.advance(60)

        services.s3.after_get[staged_key] = expire_on_final_staged_read
    else:
        services.cfn.request_hooks[2] = lambda: services.clock.advance(60)

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert validation_times == [NOW + timedelta(seconds=60)]
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


@pytest.mark.parametrize(
    "final_recheck",
    [
        "version-history",
        "claim-current",
        "intent",
        "staged-chain",
        "capability",
    ],
)
def test_drift_during_each_final_authority_recheck_blocks_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_recheck: str,
) -> None:
    """Catches a final authority recheck accepting post-ownership drift."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    if final_recheck == "version-history":

        def add_foreign_claim_version() -> None:
            claim_key, _version_id, claim_raw, _is_latest = (
                services.s3.claim_versions[0]
            )
            services.s3.claim_versions.append(
                (claim_key, "foreign-claim-version", claim_raw, False)
            )

        services.s3.after_claim_get[2] = add_foreign_claim_version
    elif final_recheck == "claim-current":
        services.s3.after_claim_get[2] = lambda: setattr(
            services.s3,
            "current_claim_version_override",
            "foreign-current-version",
        )
    elif final_recheck == "intent":
        real_put = services.s3.put_object

        def register_intent_drift(**kwargs: object) -> dict[str, object]:
            response = real_put(**kwargs)
            key = str(kwargs["Key"])
            if key.endswith("/SKYPILOT_SUBMISSION_INTENT.json"):

                def drift_after_readback() -> None:
                    services.s3.objects[(BUCKET, key)] = b"foreign-intent\n"

                services.s3.after_get[key] = drift_after_readback
            return response

        monkeypatch.setattr(services.s3, "put_object", register_intent_drift)
    elif final_recheck == "staged-chain":
        original = module._cache_seed_remote_staged_chain
        staged_chain_calls = 0

        def drift_on_final_staged_chain(
            *args: object,
            **kwargs: object,
        ) -> object:
            nonlocal staged_chain_calls
            staged_chain_calls += 1
            if staged_chain_calls == 2:
                raise module.SubmissionIntegrationError(
                    "injected final staged-chain drift"
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(
            module,
            "_cache_seed_remote_staged_chain",
            drift_on_final_staged_chain,
        )
    else:
        services.cfn.request_hooks[2] = lambda: services.cfn.overrides.update(
            {"descriptor_body_sha256": "f" * 64}
        )

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


def test_final_staged_chain_recheck_blocks_drift_after_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches launching without a post-claim exact staged-chain check."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    original = module._cache_seed_remote_staged_chain
    calls = 0

    def drifting_chain(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise module.SubmissionIntegrationError(
                "injected post-claim staged VersionId drift"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_cache_seed_remote_staged_chain", drifting_chain)

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="acquire-and-launch"),
        services=services.services,
    )

    assert calls == 2
    assert outcome.exit_code == 70
    assert len(services.s3.claim_put_calls) == 1
    assert not services.sky.launches


@pytest.mark.parametrize(
    "live_corruption",
    ["latest-exists", "record-exists", "historical-stopped", "active-running"],
)
def test_live_observation_nonzero_state_blocks_before_publication(
    tmp_path: Path,
    live_corruption: str,
) -> None:
    """Catches singleton, immutable-record, or tagged-P5 evidence being ignored."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    if live_corruption == "latest-exists":
        services.s3.objects[
            (
                BUCKET,
                f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json",
            )
        ] = b"{}\n"
    elif live_corruption == "record-exists":
        services.s3.objects[
            (
                BUCKET,
                f"campaigns/{RUN_ID}/runtime/spend-ledger/records/foreign.json",
            )
        ] = b"{}\n"
    elif live_corruption == "historical-stopped":
        services.ec2.allocation_history = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-stopped",
                            "InstanceType": "p5.48xlarge",
                            "State": {"Name": "stopped"},
                        }
                    ]
                }
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
    else:
        services.ec2.active_p5_ids = ["i-running"]

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    "active_state",
    ["pending", "running", "shutting-down", "stopping"],
)
def test_every_required_active_ec2_state_blocks_cache_seed_admission(
    tmp_path: Path,
    active_state: str,
) -> None:
    """Catches omitting one active EC2 state from zero-inventory enforcement."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    services.ec2.active_p5_states = [active_state]

    outcome = module.run_cache_seed(
        _cache_seed_request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert services.ec2.requests[1]["Filters"][-1] == {
        "Name": "instance-state-name",
        "Values": ["pending", "running", "shutting-down", "stopping"],
    }
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


@pytest.mark.parametrize(
    "drift",
    [
        "readiness-v1",
        "readiness-null-request-version",
        "readiness-wrong-response-version",
        "manifest-version-id",
        "manifest-wrong-response-version",
        "missing-upstream-version-id",
        "upstream-repository-bytes",
    ],
)
def test_cache_seed_rejects_each_staged_manifest_or_upstream_version_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    """Catches latest substitution or incomplete exact-VersionId authentication."""

    module = _module()
    fixture = _cache_seed_fixture(tmp_path)
    services = _cache_seed_services(fixture)
    request = _cache_seed_request(fixture, action="validate-only")
    staged_key = str(fixture.values["staged_key"])
    manifest_key = str(fixture.values["manifest_key"])
    version_ids = fixture.values["version_ids"]
    assert isinstance(version_ids, dict)
    if drift in {
        "readiness-v1",
        "manifest-version-id",
        "missing-upstream-version-id",
    }:
        staged = deepcopy(fixture.values["staged"])
        assert isinstance(staged, dict)
        if drift == "readiness-v1":
            staged["schema_version"] = 1
            staged["record_type"] = "glm52_staged_control_plane_ready_v1"
        elif drift == "manifest-version-id":
            staged["bundle_manifest_version_id"] = "foreign-manifest-version"
        else:
            staged_versions = dict(staged["staged_object_version_ids"])
            staged_versions.pop("repository_tar")
            staged["staged_object_version_ids"] = staged_versions
        staged = _rehash(staged, "ready_body_sha256")
        staged_raw = _file_bytes(staged)
        fixture.paths.staged_ready.write_bytes(staged_raw)
        services.s3.objects[(BUCKET, staged_key)] = staged_raw
        services.s3.versions[
            (BUCKET, staged_key, "version-staged-ready")
        ] = staged_raw
    elif drift == "readiness-null-request-version":
        request = replace(request, staged_readiness_version_id="null")
    elif drift == "readiness-wrong-response-version":
        services.s3.version_response_overrides[
            (staged_key, "version-staged-ready")
        ] = "foreign-readiness-version"
    elif drift == "manifest-wrong-response-version":
        services.s3.version_response_overrides[
            (manifest_key, str(version_ids["bundle_manifest"]))
        ] = "foreign-manifest-version"
    else:
        descriptor = fixture.values["descriptor"]
        assert isinstance(descriptor, dict)
        services.s3.versions[
            (
                BUCKET,
                str(descriptor["repo_tar_key"]),
                str(version_ids["repository_tar"]),
            )
        ] = b"foreign pinned repository bytes\n"

    outcome = module.run_cache_seed(
        request,
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert not any(event.startswith("s3:put:") for event in services.events)
    assert not services.s3.claim_put_calls
    assert not services.sky.launches


def _prepare(
    fixture: Fixture,
    services: ServiceFixture,
    tmp_path: Path,
) -> dict[str, object]:
    module = _module()
    handoff_path = tmp_path / "submission-handoff.json"
    outcome = module.run_qualification(
        _request(
            fixture,
            action="prepare-intent",
            output_handoff=handoff_path,
        ),
        services=services.services,
    )
    assert outcome.exit_code == 0
    return json.loads(handoff_path.read_bytes())


def _acquisition_key(fixture: Fixture) -> str:
    module = _module()
    descriptor_raw = fixture.paths.descriptor.read_bytes()
    return module.submission_acquired_s3_key(
        run_id=RUN_ID,
        managed_mode="qualification",
        descriptor_file_sha256=_sha(descriptor_raw),
    )


def _acquire_request(fixture: Fixture, handoff: dict[str, object]) -> Any:
    return _request(
        fixture,
        action="acquire-and-launch",
        intent_s3_uri=str(handoff["intent_s3_uri"]),
        must_start_ready_s3_uri=str(handoff["must_start_ready_s3_uri"]),
    )


def _remote_record_by_suffix(
    services: ServiceFixture,
    suffix: str,
) -> tuple[str, dict[str, object]]:
    matches = [
        (key, json.loads(raw))
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith(suffix)
    ]
    assert len(matches) == 1
    return matches[0]


def _build_coordinator_binding(
    services: ServiceFixture,
    *,
    bound_at: datetime,
) -> tuple[str, dict[str, object]]:
    module = _module()
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    _baseline_key, baseline = _remote_record_by_suffix(
        services,
        "/CONTROLLER_BASELINE.json",
    )
    _acquisition_key_value, acquisition = _remote_record_by_suffix(
        services,
        "/SUBMISSION_ACQUIRED.json",
    )
    row = services.ssm.rows[-1]
    current_row = {
        "sky_job_id": row["sky_job_id"],
        "sky_job_name": row["sky_job_name"],
        "workspace": row["workspace"],
        "controller_submitted_at": row["controller_submitted_at"],
        "controller_status": row["controller_status"],
        "controller_identity": row["controller_identity"],
    }
    baseline_history = baseline["exact_name_history"]
    assert isinstance(baseline_history, list)
    binding = module.build_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_ROLE_ARN,
        current_exact_name_history=[
            *(dict(item) for item in baseline_history),
            current_row,
        ],
        bound_at=bound_at,
        now=services.clock(),
    )
    return module.dynamic_v2_job_binding_s3_key(intent=intent), binding


def _legacy_binding_for_remote(
    services: ServiceFixture,
) -> tuple[str, dict[str, object]]:
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    observations = [
        (key, json.loads(raw))
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and "/observations/" in key
    ]
    assert len(observations) == 1
    _observation_key_value, observation = observations[0]
    descriptor_raw = services.s3.objects[(BUCKET, str(intent["descriptor_key"]))]
    descriptor = json.loads(descriptor_raw)
    prefix = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}"
    )
    legacy = build_must_start_job_binding(
        run_id=RUN_ID,
        managed_mode="qualification",
        account_id=ACCOUNT_ID,
        region=REGION,
        bucket=BUCKET,
        descriptor_body_sha256=str(intent["descriptor_body_sha256"]),
        submission_body_sha256=str(intent["intent_body_sha256"]),
        sky_job_name=SKY_JOB_NAME,
        must_start_by=str(intent["must_start_by"]),
        descriptor_key=str(intent["descriptor_key"]),
        descriptor_file_sha256=str(intent["descriptor_file_sha256"]),
        submission_key=(
            f"campaigns/{RUN_ID}/submissions/qualification/intents/"
            f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
        ),
        submission_submitted_at=str(intent["intent_at"]),
        target_job_id=int(observation["target_job_id"]),
        workspace="default",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        observation_body_sha256=str(observation["observation_body_sha256"]),
        bound_at=str(observation["observed_at"]),
    )
    assert descriptor["controller_identity"] == CONTROLLER_ROLE_ARN
    return prefix + "/JOB_BINDING.json", legacy


def test_validate_only_performs_all_reads_without_mutation_or_launch(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "validated"
    assert services.events[0] == "sts:get-caller-identity"
    assert "sky:validate" in services.events
    assert "ec2:describe:controller" in services.events
    assert "ec2:describe:active-p5" in services.events
    assert "ssm:send-observe" in services.events
    assert "cfn:inspect-must-start" in services.events
    assert not [event for event in services.events if event.startswith("s3:put:")]
    assert "sky:launch" not in services.events


def test_local_authority_authenticates_remote_manifest_and_every_pinned_version(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 0
    version_ids = fixture.values["version_ids"]
    manifest_key = fixture.values["manifest_key"]
    staged = fixture.values["staged"]
    manifest = fixture.values["manifest"]
    assert isinstance(version_ids, dict)
    assert isinstance(manifest_key, str)
    assert isinstance(staged, dict)
    assert isinstance(manifest, dict)
    expected = [
        f"s3:get-version:{manifest_key}:{version_ids['bundle_manifest']}",
        *[
            (
                f"s3:get-version:{item['key']}:"
                f"{version_ids[str(item['role'])]}"
            )
            for item in sorted(
                manifest["files"],
                key=lambda candidate: int(candidate["stage_order"]),
            )
        ],
        (
            f"s3:get-version:{staged['artifact_audit_key']}:"
            f"{version_ids['artifact_audit']}"
        ),
    ]
    indices = [services.events.index(event) for event in expected]
    assert indices == sorted(indices)
    assert "sky:launch" not in services.events
    assert not [event for event in services.events if event.startswith("s3:put:")]


def test_microsecond_staging_clock_round_trips_through_qualification_and_submitter(
    tmp_path: Path,
) -> None:
    submitter = _module()
    staging = _staging_fixture_module()
    fixture = _fixture(tmp_path)
    descriptor = fixture.values["descriptor"]
    manifest = fixture.values["manifest"]
    acceptance = fixture.values["acceptance"]
    snapshot = fixture.values["snapshot"]
    acceptance_key = fixture.values["acceptance_key"]
    snapshot_key = fixture.values["snapshot_key"]
    assert isinstance(descriptor, dict)
    assert isinstance(manifest, dict)
    assert isinstance(acceptance, dict)
    assert isinstance(snapshot, dict)
    assert isinstance(acceptance_key, str)
    assert isinstance(snapshot_key, str)
    bundle = tmp_path / "stager-round-trip"
    bundle.mkdir()
    manifest_files = manifest["files"]
    assert isinstance(manifest_files, list)
    for item in manifest_files:
        assert isinstance(item, dict)
        (bundle / str(item["local_name"])).write_bytes(
            fixture.remote[(BUCKET, str(item["key"]))]
        )
    (bundle / "bundle-manifest-v1.json").write_bytes(_file_bytes(manifest))
    staged_s3 = staging._FakeVersionedS3()
    clock = staging._ClockRecorder(
        datetime(2026, 7, 26, 11, 35, 0, 123456, tzinfo=UTC)
    )

    stager = staging._load_bundle_stager()
    first_receipt = stager._stage_bundle(
        bundle=bundle,
        s3=staged_s3,
        audit_output=bundle / "artifact-audit-v1.json",
        run_audit=staging._AuditRecorder(),
        clock=clock,
    )
    replay_audit = staging._AuditRecorder()
    replay_clock = staging._ClockRecorder(
        datetime(2026, 7, 26, 11, 40, 0, 654321, tzinfo=UTC)
    )
    replay_receipt = stager._stage_bundle(
        bundle=bundle,
        s3=staged_s3,
        audit_output=bundle / "unused-replay-audit.json",
        run_audit=replay_audit,
        clock=replay_clock,
    )
    assert {
        key: value for key, value in replay_receipt.items() if key != "staged"
    } == {
        key: value for key, value in first_receipt.items() if key != "staged"
    }
    assert set(replay_receipt["staged"].values()) == {"reused"}
    assert replay_audit.calls == 0
    assert replay_clock.calls == 0

    staged_key = (
        str(descriptor["campaign_descriptor_key"]).removesuffix(
            "campaign-descriptor-v2.json"
        )
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    staged_raw = staged_s3.objects[staged_key][-1][1]
    staged = json.loads(staged_raw)
    assert staged["staged_at"] == "2026-07-26T11:35:00Z"
    descriptor_raw = fixture.paths.descriptor.read_bytes()
    rehearsal, rehearsal_key = _rehearsal(
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
        staged=staged,
        staged_key=staged_key,
        task_raw=fixture.paths.task.read_bytes(),
        config_raw=fixture.paths.config.read_bytes(),
        validator_raw=(
            REPO_ROOT / "aws/glm52-gpu/scripts/validate_skypilot_control_plane.py"
        ).read_bytes(),
    )
    readiness = build_qualification_submission_ready(
        descriptor=descriptor,
        seed_descriptor_raw=fixture.paths.seed_descriptor.read_bytes(),
        seed_descriptor_file_sha256=_sha(
            fixture.paths.seed_descriptor.read_bytes()
        ),
        staged_readiness=staged,
        cache_seed_acceptance=acceptance,
        gpu_spend_snapshot=snapshot,
        rehearsal_evidence=rehearsal,
        descriptor_file_sha256=_sha(descriptor_raw),
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=_sha(staged_raw),
        cache_seed_acceptance_key=acceptance_key,
        cache_seed_acceptance_file_sha256=_sha(_file_bytes(acceptance)),
        gpu_spend_snapshot_key=snapshot_key,
        gpu_spend_snapshot_sha256=_sha(_file_bytes(snapshot)),
        rehearsal_evidence_key=rehearsal_key,
        rehearsal_evidence_sha256=_sha(_file_bytes(rehearsal)),
        built_at="2026-07-26T11:56:00Z",
    )
    validate_qualification_submission_ready(
        readiness,
        descriptor=descriptor,
        staged_readiness=staged,
        cache_seed_acceptance=acceptance,
        gpu_spend_snapshot=snapshot,
        rehearsal_evidence=rehearsal,
        descriptor_file_sha256=_sha(descriptor_raw),
        staged_readiness_key=staged_key,
        staged_readiness_file_sha256=_sha(staged_raw),
        cache_seed_acceptance_key=acceptance_key,
        cache_seed_acceptance_file_sha256=_sha(_file_bytes(acceptance)),
        gpu_spend_snapshot_key=snapshot_key,
        gpu_spend_snapshot_sha256=_sha(_file_bytes(snapshot)),
        rehearsal_evidence_key=rehearsal_key,
        rehearsal_evidence_sha256=_sha(_file_bytes(rehearsal)),
        now=NOW,
    )
    submitter._validate_remote_staged_readiness(
        submitter._JsonArtifact(staged_key, staged_raw, staged),
        descriptor=submitter._JsonArtifact(
            str(descriptor["campaign_descriptor_key"]),
            descriptor_raw,
            descriptor,
        ),
    )


def test_submitter_rejects_fractional_remote_staged_at(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    descriptor = fixture.values["descriptor"]
    staged = deepcopy(fixture.values["staged"])
    staged_key = fixture.values["staged_key"]
    assert isinstance(descriptor, dict)
    assert isinstance(staged, dict)
    assert isinstance(staged_key, str)
    staged["staged_at"] = "2026-07-26T11:35:00.123456Z"
    staged = _rehash(staged, "ready_body_sha256")
    staged_raw = _file_bytes(staged)

    with pytest.raises(
        module.SubmissionIntegrationError,
        match="whole-second",
    ):
        module._validate_remote_staged_readiness(
            module._JsonArtifact(staged_key, staged_raw, staged),
            descriptor=module._JsonArtifact(
                str(descriptor["campaign_descriptor_key"]),
                fixture.paths.descriptor.read_bytes(),
                descriptor,
            ),
        )


def test_remote_chain_ignores_newer_current_manifest_and_uses_pinned_version(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    manifest_key = fixture.values["manifest_key"]
    version_ids = fixture.values["version_ids"]
    assert isinstance(manifest_key, str)
    assert isinstance(version_ids, dict)
    services.s3.objects[(BUCKET, manifest_key)] = b'{"foreign":"latest"}\n'

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert (
        f"s3:get-version:{manifest_key}:{version_ids['bundle_manifest']}"
        in services.events
    )
    assert f"s3:get:{manifest_key}" not in services.events


@pytest.mark.parametrize(
    "unsafe",
    (
        "campaigns/run/watchdog\npayload.zip",
        "campaigns/run/watchdog\tpayload.zip",
        "campaigns/run/watchdog\x00payload.zip",
        "campaigns/run/watchdog\x7fpayload.zip",
        r"campaigns/run/watchdog\payload.zip",
    ),
)
def test_submitter_rejects_non_printable_or_backslash_remote_watchdog_key(
    tmp_path: Path,
    unsafe: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    descriptor = fixture.values["descriptor"]
    manifest = deepcopy(fixture.values["manifest"])
    staged = dict(fixture.values["staged"])
    assert isinstance(descriptor, dict)
    assert isinstance(manifest, dict)
    files = manifest["files"]
    assert isinstance(files, list)
    watchdog = next(item for item in files if item["role"] == "watchdog")
    watchdog["key"] = unsafe
    manifest = _rehash(manifest, "bundle_manifest_body_sha256")
    manifest_raw = _file_bytes(manifest)
    staged.update(
        {
            "bundle_manifest_key": (
                str(descriptor["campaign_descriptor_key"]).removesuffix(
                    "campaign-descriptor-v2.json"
                )
                + "bundle-manifests/"
                + str(manifest["bundle_manifest_body_sha256"])
                + "/bundle-manifest-v1.json"
            ),
            "bundle_manifest_file_sha256": _sha(manifest_raw),
            "bundle_manifest_body_sha256": manifest[
                "bundle_manifest_body_sha256"
            ],
        }
    )

    with pytest.raises(module.SubmissionIntegrationError, match="safe|ASCII"):
        module._validate_remote_bundle_manifest(
            manifest_raw,
            ready=staged,
            descriptor=module._JsonArtifact(
                str(descriptor["campaign_descriptor_key"]),
                fixture.paths.descriptor.read_bytes(),
                descriptor,
            ),
        )


def test_remote_chain_rejects_historical_v1_readiness_before_sky_or_mutation(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    staged_key = fixture.values["staged_key"]
    descriptor = fixture.values["descriptor"]
    staged = fixture.values["staged"]
    assert isinstance(staged_key, str)
    assert isinstance(descriptor, dict)
    assert isinstance(staged, dict)
    historical = {
        "schema_version": 1,
        "record_type": "glm52_staged_control_plane_ready_v1",
        "run_id": RUN_ID,
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_sha256": fixture.paths.descriptor.read_bytes().hex()[:64],
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "bundle_manifest_body_sha256": staged["bundle_manifest_body_sha256"],
        "artifact_audit_key": staged["artifact_audit_key"],
        "artifact_audit_sha256": staged["artifact_audit_sha256"],
        "staged_at": staged["staged_at"],
    }
    services.s3.objects[(BUCKET, staged_key)] = _file_bytes(
        _self_hashed(historical, "ready_body_sha256")
    )

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:validate" not in services.events
    assert "sky:launch" not in services.events
    assert not [event for event in services.events if event.startswith("s3:put:")]


@pytest.mark.parametrize("returned_version", ["missing", "", "null", "wrong-version"])
def test_remote_chain_rejects_missing_or_wrong_manifest_response_version(
    tmp_path: Path,
    returned_version: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    manifest_key = fixture.values["manifest_key"]
    version_ids = fixture.values["version_ids"]
    assert isinstance(manifest_key, str)
    assert isinstance(version_ids, dict)
    pinned = str(version_ids["bundle_manifest"])
    if returned_version == "missing":
        services.s3.version_response_overrides[(manifest_key, pinned)] = None
    else:
        services.s3.version_response_overrides[
            (manifest_key, pinned)
        ] = returned_version

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:validate" not in services.events
    assert "sky:launch" not in services.events


def test_remote_chain_rejects_pinned_staged_object_drift_before_launch(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    descriptor = fixture.values["descriptor"]
    version_ids = fixture.values["version_ids"]
    assert isinstance(descriptor, dict)
    assert isinstance(version_ids, dict)
    repo_key = str(descriptor["repo_tar_key"])
    services.s3.versions[
        (BUCKET, repo_key, str(version_ids["repository_tar"]))
    ] = b"foreign pinned repository\n"

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:validate" not in services.events
    assert "sky:launch" not in services.events
    assert not [event for event in services.events if event.startswith("s3:put:")]


@pytest.mark.parametrize("mutation", ["missing", "undeclared"])
def test_live_spend_reconstruction_requires_every_immutable_record_and_exact_history(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    record_prefix = f"campaigns/{RUN_ID}/runtime/spend-ledger/records/"
    names = fixture.values["spend_record_names"]
    assert isinstance(names, tuple)
    if mutation == "missing":
        del services.s3.objects[(BUCKET, record_prefix + str(names[0]))]
    else:
        services.s3.objects[
            (
                BUCKET,
                record_prefix + "999999-allocation_ended-" + "f" * 64 + ".json",
            )
        ] = b"{}\n"

    outcome = module.run_qualification(
        _request(fixture, action="validate-only"),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:launch" not in services.events
    assert not [event for event in services.events if event.startswith("s3:put:")]


def test_pinned_sky_history_projects_fractional_and_transitional_runtime_rows(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    fractional = NOW.replace(microsecond=123456)
    services.ssm.rows = [
        _history_row(
            job_id=3,
            status="SUBMITTED",
            submitted_at=fractional,
        ),
        _history_row(
            job_id=4,
            status="WINDING_DOWN",
            submitted_at=fractional,
            start_at=fractional,
        ),
    ]

    rows, _observed_at = module._observe_history(
        services.services,
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        sky_job_name=SKY_JOB_NAME,
        controller_identity=CONTROLLER_ROLE_ARN,
    )

    assert [row["controller_status"] for row in rows] == [
        "SUBMITTED",
        "WINDING_DOWN",
    ]
    assert rows[0]["controller_submitted_at"] == _iso(fractional)
    assert rows[1]["start_at"] == _iso(fractional)


def test_pinned_sky_pending_row_without_submitted_at_is_reconcile_pending(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.sky.add_controller_row = False
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 75
    services.ssm.rows.append(
        {
            **_history_row(
                job_id=17,
                status="PENDING",
                submitted_at=NOW,
            ),
            "controller_submitted_at": None,
        }
    )

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 75
    assert (
        "sky:launch" not in services.events[services.events.index("sky:launch") + 1 :]
    )


def test_prepare_intent_conditionally_publishes_ordered_inert_authorities(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff_path = tmp_path / "submission-handoff.json"

    outcome = module.run_qualification(
        _request(
            fixture,
            action="prepare-intent",
            output_handoff=handoff_path,
        ),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "intent-prepared"
    handoff = json.loads(handoff_path.read_bytes())
    assert handoff["record_type"] == "glm52_sky_submission_handoff_v1"
    assert handoff["requires_separate_active_deployment"] is True
    assert handoff["intent_s3_uri"].endswith("/SKYPILOT_SUBMISSION_INTENT.json")
    assert handoff["must_start_ready_s3_uri"].endswith("/CONTROL_PLANE_READY.json")
    put_keys = [
        event.removeprefix("s3:put:")
        for event in services.events
        if event.startswith("s3:put:")
    ]
    assert [key.rsplit("/", 1)[-1] for key in put_keys] == [
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json",
        "GPU_SPEND_SNAPSHOT.json",
        "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json",
        "QUALIFICATION_SUBMISSION_READY.json",
        "SKYPILOT_SUBMISSION_INTENT.json",
        "CONTROLLER_BASELINE.json",
        "CONTROL_PLANE_READY.json",
    ]
    assert "sky:launch" not in services.events
    assert not any("H100_QUALIFICATION_SUBMITTED" in key for key in put_keys)
    assert not any("SKY_JOB_STATUS" in key for key in put_keys)
    assert not any("submission-locks" in key for key in put_keys)


def test_prepare_publishes_resumable_baseline_handoff_before_deployment_exists(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff_path = tmp_path / "deployment-handoff.json"
    services.cfn.active = False

    first = module.run_qualification(
        _request(
            fixture,
            action="prepare-intent",
            output_handoff=handoff_path,
        ),
        services=services.services,
    )

    assert first.exit_code == 78
    deployment_handoff = json.loads(handoff_path.read_bytes())
    assert (
        deployment_handoff["record_type"]
        == "glm52_sky_submission_deployment_handoff_v1"
    )
    assert deployment_handoff["controller_baseline_s3_uri"].endswith(
        "/CONTROLLER_BASELINE.json"
    )
    assert "must_start_ready_s3_uri" not in deployment_handoff
    baseline_puts = [
        event
        for event in services.events
        if event.startswith("s3:put:") and event.endswith("/CONTROLLER_BASELINE.json")
    ]
    assert len(baseline_puts) == 1
    first_request = dict(services.cfn.requests[-1])

    services.cfn.active = True
    second = module.run_qualification(
        _request(
            fixture,
            action="prepare-intent",
            output_handoff=handoff_path,
        ),
        services=services.services,
    )

    assert second.exit_code == 0
    ready_handoff = json.loads(handoff_path.read_bytes())
    assert ready_handoff["record_type"] == "glm52_sky_submission_handoff_v1"
    assert ready_handoff["intent_s3_uri"] == deployment_handoff["intent_s3_uri"]
    assert (
        ready_handoff["controller_baseline_s3_uri"]
        == deployment_handoff["controller_baseline_s3_uri"]
    )
    assert services.cfn.requests[-1] == first_request


def test_acquisition_uses_reviewed_baseline_without_republishing_live_authority(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    reviewed_request = dict(services.cfn.requests[-1])
    services.clock.advance(1)
    services.sky.add_controller_row = False
    services.events.clear()

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 75
    assert services.cfn.requests[-1] == reviewed_request
    assert not any(
        event.endswith(("/CONTROLLER_BASELINE.json", "/CONTROL_PLANE_READY.json"))
        for event in services.events
        if event.startswith("s3:put:")
    )


def test_only_successful_selector_put_launches_once_and_accepts_exact_job(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.events.clear()

    outcome = module.run_qualification(
        _request(
            fixture,
            action="acquire-and-launch",
            intent_s3_uri=str(handoff["intent_s3_uri"]),
            must_start_ready_s3_uri=str(handoff["must_start_ready_s3_uri"]),
        ),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"
    assert services.events.count("sky:launch") == 1
    assert len(services.sky.launches) == 1
    launch = services.sky.launches[0]
    descriptor = fixture.values["descriptor"]
    assert isinstance(descriptor, dict)
    private_config = launch[launch.index("--config") + 1]
    private_task = launch[-1]
    expected_launch = [
        "jobs",
        "launch",
        "--async",
        "--config",
        private_config,
        "-y",
        "-n",
        SKY_JOB_NAME,
        "--no-use-spot",
        "--image-id",
        "ami-0123456789abcdef0",
        "--env",
        f"GLM52_CAMPAIGN_DESCRIPTOR_S3_URI={handoff['descriptor_s3_uri']}",
        "--env",
        (f"GLM52_CAMPAIGN_DESCRIPTOR_SHA256={handoff['descriptor_file_sha256']}"),
        "--env",
        (f"GLM52_GPU_SPEND_APPROVAL_S3_URI=s3://{BUCKET}/{descriptor['approval_key']}"),
        "--env",
        (
            "GLM52_GPU_SPEND_APPROVAL_SHA256="
            f"{_sha(fixture.paths.approval.read_bytes())}"
        ),
        "--env",
        f"GLM52_SKY_JOB_NAME={SKY_JOB_NAME}",
        "--env",
        "GLM52_MANAGED_MODE=qualification",
        "--env",
        f"GLM52_SKY_SUBMISSION_INTENT_S3_URI={handoff['intent_s3_uri']}",
        "--env",
        (f"GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256={handoff['intent_file_sha256']}"),
        "--env",
        (f"GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256={handoff['intent_body_sha256']}"),
        private_task,
    ]
    assert launch == expected_launch
    assert descriptor["region"] == REGION
    assert launch.count("--no-use-spot") == 1
    assert "--use-spot" not in launch
    assert launch[launch.index("--image-id") + 1] == descriptor["image_id"]
    assert private_config != str(fixture.paths.config)
    assert private_task != str(fixture.paths.task)
    assert services.sky.launch_cwds[0] is not None
    private_repo = services.sky.launch_cwds[0]
    assert private_repo is not None
    assert Path(private_task) == private_repo / SKY_TASK_RELATIVE_PATH
    assert Path(private_config).is_absolute()
    snapshot = services.sky.launch_snapshots[0]
    raw = snapshot["raw"]
    modes = snapshot["modes"]
    links = snapshot["links"]
    regular = snapshot["regular"]
    assert isinstance(raw, dict)
    assert isinstance(modes, dict)
    assert isinstance(links, dict)
    assert isinstance(regular, dict)
    assert raw["task"] == fixture.paths.task.read_bytes()
    assert raw["config"] == fixture.paths.config.read_bytes()
    assert all(
        raw[relative_path] == (REPO_ROOT / relative_path).read_bytes()
        for relative_path in SEALED_MOUNT_RELATIVE_PATHS
    )
    assert set(modes.values()) == {0o400}
    assert set(links.values()) == {1}
    assert set(regular.values()) == {True}
    assert not private_repo.parent.exists()
    new_intent_keys = (
        "GLM52_SKY_SUBMISSION_INTENT_S3_URI",
        "GLM52_SKY_SUBMISSION_INTENT_FILE_SHA256",
        "GLM52_SKY_SUBMISSION_INTENT_BODY_SHA256",
    )
    assert all(
        sum(value.startswith(f"{key}=") for value in launch) == 1
        for key in new_intent_keys
    )
    old_intent_keys = (
        "GLM52_IMMUTABLE_SUBMISSION_S3_URI",
        "GLM52_IMMUTABLE_SUBMISSION_FILE_SHA256",
        "GLM52_IMMUTABLE_SUBMISSION_BODY_SHA256",
    )
    assert not any(old_key in value for old_key in old_intent_keys for value in launch)
    acquisition_put = next(
        index
        for index, event in enumerate(services.events)
        if event.endswith("/SUBMISSION_ACQUIRED.json") and event.startswith("s3:put:")
    )
    launch_index = services.events.index("sky:launch")
    manifest_key = fixture.values["manifest_key"]
    version_ids = fixture.values["version_ids"]
    assert isinstance(manifest_key, str)
    assert isinstance(version_ids, dict)
    selected_manifest_read = max(
        index
        for index, event in enumerate(services.events)
        if event
        == f"s3:get-version:{manifest_key}:{version_ids['bundle_manifest']}"
    )
    observation_put = next(
        index
        for index, event in enumerate(services.events)
        if "/observations/" in event and event.startswith("s3:put:")
    )
    binding_put = next(
        index
        for index, event in enumerate(services.events)
        if event.endswith("/DYNAMIC_JOB_BINDING.json") and event.startswith("s3:put:")
    )
    accepted_put = next(
        index
        for index, event in enumerate(services.events)
        if event.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        and event.startswith("s3:put:")
    )
    assert (
        acquisition_put
        < selected_manifest_read
        < launch_index
        < observation_put
        < binding_put
        < accepted_put
    )
    acquisition_key = _acquisition_key(fixture)
    acquisition_raw = services.s3.objects[(BUCKET, acquisition_key)]
    assert acquisition_raw.endswith(b"\n")
    observation_raw = next(
        raw
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and "/observations/" in key
    )
    binding_raw = next(
        raw
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/DYNAMIC_JOB_BINDING.json")
    )
    assert not observation_raw.endswith(b"\n")
    assert not binding_raw.endswith(b"\n")
    binding = json.loads(binding_raw)
    assert binding["record_type"] == "glm52_sky_must_start_job_binding_v2"
    accepted = next(
        json.loads(raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
    )
    assert accepted["job_binding"] == binding
    assert accepted["job_binding_key"].endswith("/DYNAMIC_JOB_BINDING.json")
    assert accepted["job_binding_file_sha256"] == _sha(binding_raw)
    assert accepted["job_binding_body_sha256"] == binding["job_binding_body_sha256"]
    legacy_key = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{accepted['intent_body_sha256']}/JOB_BINDING.json"
    )
    assert f"s3:get:{legacy_key}" not in services.events
    assert f"s3:put:{legacy_key}" not in services.events
    latest_event = f"s3:get:campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json"
    last_latest_before_acquisition = max(
        index
        for index, event in enumerate(services.events[:acquisition_put])
        if event == latest_event
    )
    assert (
        f"s3:list:campaigns/{RUN_ID}/runtime/spend-ledger/records/"
        in services.events[last_latest_before_acquisition + 1 : acquisition_put]
    )
    assert (
        "ec2:describe:spend-history"
        in services.events[last_latest_before_acquisition + 1 : acquisition_put]
    )
    assert services.events[acquisition_put - 2 : acquisition_put] == [
        f"s3:get:{acquisition_key}",
        f"s3:list:{acquisition_key.rsplit('/', 1)[0]}/",
    ]


def test_selected_retry_reauthenticates_pinned_chain_before_reconciliation(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 0
    descriptor = fixture.values["descriptor"]
    version_ids = fixture.values["version_ids"]
    assert isinstance(descriptor, dict)
    assert isinstance(version_ids, dict)
    services.events.clear()
    services.s3.versions[
        (
            BUCKET,
            str(descriptor["repo_tar_key"]),
            str(version_ids["repository_tar"]),
        )
    ] = b"foreign selected repository\n"

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert "sky:launch" not in services.events
    assert "ssm:send-observe" not in services.events


def test_reconcile_reauthenticates_pinned_chain_before_controller_reads(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 0
    staged = fixture.values["staged"]
    version_ids = fixture.values["version_ids"]
    assert isinstance(staged, dict)
    assert isinstance(version_ids, dict)
    services.events.clear()
    services.s3.versions[
        (
            BUCKET,
            str(staged["artifact_audit_key"]),
            str(version_ids["artifact_audit"]),
        )
    ] = b'{"foreign":"audit"}\n'

    outcome = module.run_qualification(
        _request(
            fixture,
            action="reconcile",
            acquisition_s3_uri=f"s3://{BUCKET}/{_acquisition_key(fixture)}",
        ),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:launch" not in services.events
    assert "ssm:send-observe" not in services.events


def test_conditional_create_builds_gate_one_and_fresh_validates_gate_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    exact = _worker_start_v2_source_snapshot()
    reads: list[Path] = []

    def read_sources(*, task: Path) -> WorkerStartV2SourceSnapshot:
        reads.append(task)
        return exact

    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        read_sources,
        raising=False,
    )
    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"
    assert reads == [fixture.paths.task, fixture.paths.task]
    assert services.events.count("sky:launch") == 1
    assert not any(
        "/worker-start-v2-readiness/" in key
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


@pytest.mark.parametrize(
    "mutation",
    ["task", "config", *SEALED_MOUNT_RELATIVE_PATHS],
)
def test_post_gate_two_original_mutations_cannot_change_bytes_consumed_by_sky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    source_root = _temporary_worker_source_tree(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", source_root)
    monkeypatch.chdir(source_root)
    authenticated = {
        "task": fixture.paths.task.read_bytes(),
        "config": fixture.paths.config.read_bytes(),
        **{
            relative_path: (source_root / relative_path).read_bytes()
            for relative_path in SEALED_MOUNT_RELATIVE_PATHS
        },
    }
    original_path = (
        fixture.paths.task
        if mutation == "task"
        else fixture.paths.config
        if mutation == "config"
        else source_root / mutation
    )
    attacker_raw = f"attacker-controlled:{mutation}\n".encode()
    if mutation == "config":
        real_stable_read = module._read_stable_source
        config_reads = 0

        def mutate_after_config_read(path: Path, *, label: str) -> bytes:
            nonlocal config_reads
            raw = real_stable_read(path, label=label)
            if path == fixture.paths.config:
                config_reads += 1
                if config_reads == 2:
                    original_path.write_bytes(attacker_raw)
            return raw

        monkeypatch.setattr(
            module,
            "_read_stable_source",
            mutate_after_config_read,
        )
    else:
        real_source_read = module._read_worker_start_v2_source_snapshot
        source_reads = 0

        def mutate_after_source_read(*, task: Path) -> WorkerStartV2SourceSnapshot:
            nonlocal source_reads
            snapshot = real_source_read(task=task)
            source_reads += 1
            if source_reads == 2:
                original_path.write_bytes(attacker_raw)
            return snapshot

        monkeypatch.setattr(
            module,
            "_read_worker_start_v2_source_snapshot",
            mutate_after_source_read,
        )
    real_build_bundle = module._build_sealed_launch_bundle
    bundle_builds = 0

    def build_after_mutation(*args: object, **kwargs: object) -> object:
        nonlocal bundle_builds
        assert original_path.read_bytes() == attacker_raw
        result = real_build_bundle(*args, **kwargs)
        bundle_builds += 1
        return result

    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        build_after_mutation,
    )

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert bundle_builds == 1
    assert original_path.read_bytes() == attacker_raw
    assert services.sky.launch_cwds[0] is not None
    observed = services.sky.launch_snapshots[0]["raw"]
    assert isinstance(observed, dict)
    assert observed == authenticated


def test_gate_one_source_drift_prevents_selector_put_and_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    exact = _worker_start_v2_source_snapshot()
    drifted = replace(exact, sky_task_raw=exact.sky_task_raw + b"\n")
    reads = 0

    def read_sources(*, task: Path) -> WorkerStartV2SourceSnapshot:
        nonlocal reads
        assert task == fixture.paths.task
        reads += 1
        return drifted

    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        read_sources,
        raising=False,
    )
    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert reads == 1
    assert (BUCKET, _acquisition_key(fixture)) not in services.s3.objects
    assert "sky:launch" not in services.events


@pytest.mark.parametrize("task_mode", ["missing", "symlink", "hardlink", "different"])
def test_source_snapshot_reader_rejects_unsafe_or_foreign_task_files(
    tmp_path: Path,
    task_mode: str,
) -> None:
    module = _module()
    repository_task = REPO_ROOT / "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
    task = tmp_path / "task.yaml"
    if task_mode == "symlink":
        task.symlink_to(repository_task)
    elif task_mode == "hardlink":
        backing = tmp_path / "task-backing.yaml"
        backing.write_bytes(repository_task.read_bytes())
        os.link(backing, task)
    elif task_mode == "different":
        task.write_bytes(b"name: foreign-task\n")

    with pytest.raises(module.SubmissionIntegrationError):
        module._read_worker_start_v2_source_snapshot(task=task)


def test_source_drift_between_gates_leaves_selector_but_never_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    exact = _worker_start_v2_source_snapshot()
    reads = 0

    def read_sources(*, task: Path) -> WorkerStartV2SourceSnapshot:
        nonlocal reads
        assert task == fixture.paths.task
        reads += 1
        if reads == 1:
            return exact
        return replace(
            exact, managed_entrypoint_raw=exact.managed_entrypoint_raw + b"\n"
        )

    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        read_sources,
        raising=False,
    )
    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "source" in str(outcome.detail["reason"]).lower()
    assert reads == 2
    assert (BUCKET, _acquisition_key(fixture)) in services.s3.objects
    assert "sky:launch" not in services.events


@pytest.mark.parametrize(
    ("drift_read", "selector_expected"),
    [(1, False), (2, True)],
    ids=["gate-one", "gate-two"],
)
def test_config_drift_at_either_gate_fails_closed_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_read: int,
    selector_expected: bool,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    real_read = module._read_stable_source
    config_reads = 0

    def read_with_drift(path: Path, *, label: str) -> bytes:
        nonlocal config_reads
        raw = real_read(path, label=label)
        if path == fixture.paths.config:
            config_reads += 1
            if config_reads == drift_read:
                return raw + b"# attacker drift\n"
        return raw

    monkeypatch.setattr(module, "_read_stable_source", read_with_drift)

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert config_reads == drift_read
    assert ((BUCKET, _acquisition_key(fixture)) in services.s3.objects) is (
        selector_expected
    )
    assert "sky:launch" not in services.events


def test_stored_selector_and_racing_selector_never_invoke_gate_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    exact = _worker_start_v2_source_snapshot()
    reads = 0

    def read_sources(*, task: Path) -> WorkerStartV2SourceSnapshot:
        nonlocal reads
        assert task == fixture.paths.task
        reads += 1
        return exact

    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        read_sources,
        raising=False,
    )

    def forbidden_sealed_launch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            "non-winner constructed or materialized sealed launch data"
        )

    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        forbidden_sealed_launch,
    )
    monkeypatch.setattr(
        module,
        "_materialize_sealed_launch",
        forbidden_sealed_launch,
    )
    services.s3.put_modes[_acquisition_key(fixture)] = "race"
    raced = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert raced.exit_code == 75
    assert reads == 1
    assert "sky:launch" not in services.events

    def forbidden_read(*, task: Path) -> WorkerStartV2SourceSnapshot:
        raise AssertionError(f"stored selector read source gate: {task}")

    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        forbidden_read,
    )
    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 75
    assert "sky:launch" not in services.events


def test_racing_selector_never_materializes_private_launch_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.put_modes[_acquisition_key(fixture)] = "race"

    def forbidden_materialization(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("race loser materialized sealed launch data")

    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        forbidden_materialization,
    )
    monkeypatch.setattr(
        module,
        "_materialize_sealed_launch",
        forbidden_materialization,
        raising=False,
    )

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 75
    assert "sky:launch" not in services.events


def test_materialized_readback_failure_is_not_classified_as_a_sky_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)

    real_verify = module._verify_materialized_launch
    verification_count = 0

    def fail_final_readback(*args: object, **kwargs: object) -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            raise module.SubmissionIntegrationError("sealed readback failed")
        real_verify(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_verify_materialized_launch",
        fail_final_readback,
        raising=False,
    )

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert verification_count == 2
    assert "sealed readback failed" in str(outcome.detail["reason"])
    assert (BUCKET, _acquisition_key(fixture)) in services.s3.objects
    assert "sky:launch" not in services.events


@pytest.mark.parametrize(
    "tamper",
    ["content", "mode", "hardlink", "symlink", "unexpected"],
)
def test_private_materialization_rejects_tampering_and_unexpected_members(
    tmp_path: Path,
    tamper: str,
) -> None:
    module = _module()
    sources = _worker_start_v2_source_snapshot()
    config_raw = b"aws:\n  use_internal_ips: true\n"
    bundle = module._build_sealed_launch_bundle(
        source_snapshot=sources,
        config_raw=config_raw,
    )

    with module._materialize_sealed_launch(bundle) as materialized:
        if tamper == "content":
            materialized.config_path.chmod(0o600)
            materialized.config_path.write_bytes(b"attacker\n")
        elif tamper == "mode":
            materialized.config_path.chmod(0o600)
        elif tamper == "hardlink":
            os.link(
                materialized.config_path,
                materialized.root / "second-link",
            )
        elif tamper == "symlink":
            outside = tmp_path / "outside-task.yaml"
            outside.write_bytes(sources.sky_task_raw)
            materialized.task_path.unlink()
            materialized.task_path.symlink_to(outside)
        else:
            unexpected = materialized.repository_root / "unexpected.txt"
            unexpected.write_bytes(b"unexpected")

        with pytest.raises(module.SubmissionIntegrationError):
            module._verify_materialized_launch(
                materialized=materialized,
                bundle=bundle,
            )


def test_private_materialization_has_exact_inventory_and_safe_directory_modes() -> None:
    module = _module()
    sources = _worker_start_v2_source_snapshot()
    config_raw = b"aws:\n  use_internal_ips: true\n"
    bundle = module._build_sealed_launch_bundle(
        source_snapshot=sources,
        config_raw=config_raw,
    )

    with module._materialize_sealed_launch(bundle) as materialized:
        assert stat.S_IMODE(os.lstat(materialized.root).st_mode) == 0o700
        files: set[str] = set()
        directories: set[str] = set()
        for path in materialized.root.rglob("*"):
            relative = path.relative_to(materialized.root).as_posix()
            identity = os.lstat(path)
            if stat.S_ISDIR(identity.st_mode):
                directories.add(relative)
                assert stat.S_IMODE(identity.st_mode) == 0o700
            else:
                files.add(relative)
                assert stat.S_ISREG(identity.st_mode)
                assert stat.S_IMODE(identity.st_mode) == 0o400
                assert identity.st_nlink == 1
        expected_files = {
            module._SEALED_CONFIG_RELATIVE_PATH,
            f"repo/{SKY_TASK_RELATIVE_PATH}",
            *[f"repo/{path}" for path in SEALED_MOUNT_RELATIVE_PATHS],
        }
        assert files == expected_files
        expected_directories: set[str] = set()
        for file_path in expected_files:
            parent = Path(file_path).parent
            while parent != Path("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        assert directories == expected_directories
        assert all(
            not (os.lstat(materialized.root / path).st_mode & 0o077)
            for path in directories
        )
        assert materialized.repository_root == materialized.root / "repo"
        assert materialized.task_path == (
            materialized.repository_root / SKY_TASK_RELATIVE_PATH
        )


def test_private_file_write_uses_exact_safe_syscall_order_and_complete_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    path = tmp_path / "sealed"
    raw = b"abcdef"
    events: list[str] = []
    descriptors: list[int] = []
    real_open = module.os.open
    real_write = module.os.write
    real_fsync = module.os.fsync
    real_close = module.os.close
    real_fchmod = module.os.fchmod
    real_chmod = module.os.chmod

    def tracked_open(
        opened_path: object,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        assert Path(opened_path) == path
        assert flags & module.os.O_CREAT
        assert flags & module.os.O_EXCL
        assert flags & module.os.O_NOFOLLOW
        events.append("open")
        descriptor = real_open(opened_path, flags, mode)
        descriptors.append(descriptor)
        return descriptor

    def short_write(descriptor: int, value: object) -> int:
        assert descriptor == descriptors[0]
        events.append("write")
        view = memoryview(value)
        return real_write(descriptor, view[:2])

    def tracked_fsync(descriptor: int) -> None:
        assert descriptor == descriptors[0]
        events.append("fsync")
        real_fsync(descriptor)

    def tracked_close(descriptor: int) -> None:
        assert descriptor == descriptors[0]
        events.append("close")
        real_close(descriptor)

    def forbidden_fchmod(descriptor: int, mode: int) -> None:
        _ = descriptor, mode
        events.append("fchmod")
        real_fchmod(descriptor, mode)

    def tracked_chmod(
        target: object,
        mode: int,
        *args: object,
        **kwargs: object,
    ) -> None:
        assert Path(target) == path
        assert mode == 0o400
        assert kwargs.get("follow_symlinks") is False
        events.append("chmod")
        real_chmod(target, mode, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", tracked_open)
    monkeypatch.setattr(module.os, "write", short_write)
    monkeypatch.setattr(module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(module.os, "close", tracked_close)
    monkeypatch.setattr(module.os, "fchmod", forbidden_fchmod)
    monkeypatch.setattr(module.os, "chmod", tracked_chmod)

    module._write_private_file(path, raw)

    assert events == [
        "open",
        "write",
        "write",
        "write",
        "fsync",
        "close",
        "chmod",
    ]
    assert path.read_bytes() == raw
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o400


@pytest.mark.parametrize("existing_kind", ["regular", "symlink"])
def test_private_file_write_refuses_existing_or_symlink_destinations(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    module = _module()
    path = tmp_path / "sealed"
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    if existing_kind == "regular":
        path.write_bytes(b"existing")
    else:
        path.symlink_to(outside)

    with pytest.raises(module.SubmissionIntegrationError):
        module._write_private_file(path, b"authenticated")

    assert outside.read_bytes() == b"outside"
    if existing_kind == "regular":
        assert path.read_bytes() == b"existing"
    else:
        assert path.is_symlink()


def test_private_file_write_requires_nofollow_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.delattr(module.os, "O_NOFOLLOW")

    with pytest.raises(
        module.SubmissionIntegrationError,
        match="no-follow",
    ):
        module._write_private_file(tmp_path / "sealed", b"authenticated")


@pytest.mark.parametrize("failure_stage", ["fsync", "close", "chmod"])
def test_private_file_write_fails_closed_and_closes_once_on_sync_or_mode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    module = _module()
    path = tmp_path / "sealed"
    close_calls = 0
    real_fsync = module.os.fsync
    real_close = module.os.close
    real_chmod = module.os.chmod

    def injected_fsync(descriptor: int) -> None:
        if failure_stage == "fsync":
            raise OSError("injected fsync failure")
        real_fsync(descriptor)

    def injected_close(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        if failure_stage == "close":
            raise OSError("injected close failure")

    def injected_chmod(
        target: object,
        mode: int,
        *args: object,
        **kwargs: object,
    ) -> None:
        if failure_stage == "chmod":
            raise OSError("injected chmod failure")
        real_chmod(target, mode, *args, **kwargs)

    monkeypatch.setattr(module.os, "fsync", injected_fsync)
    monkeypatch.setattr(module.os, "close", injected_close)
    monkeypatch.setattr(module.os, "chmod", injected_chmod)

    with pytest.raises(module.SubmissionIntegrationError):
        module._write_private_file(path, b"authenticated")

    assert close_calls == 1


def test_private_materialization_rejects_caller_supplied_mount_paths() -> None:
    module = _module()
    sources = _worker_start_v2_source_snapshot()
    bundle = module._build_sealed_launch_bundle(
        source_snapshot=sources,
        config_raw=b"aws:\n  use_internal_ips: true\n",
    )
    foreign_mounts = (
        ("../../attacker.py", bundle.mount_sources[0][1]),
        *bundle.mount_sources[1:],
    )
    foreign = module._SealedLaunchBundle(
        task_raw=bundle.task_raw,
        config_raw=bundle.config_raw,
        mount_sources=foreign_mounts,
    )

    with pytest.raises(module.SubmissionIntegrationError):
        with module._materialize_sealed_launch(foreign):
            pytest.fail("foreign mount path was materialized")


@pytest.mark.parametrize(
    "launch_result",
    [
        0,
        19,
        RuntimeError("Sky CLI failed"),
        TimeoutError("lost Sky CLI response"),
        ProcessCrash("process died during Sky launch"),
    ],
    ids=["zero", "nonzero", "failure", "timeout", "process-crash"],
)
def test_cleanup_failure_cannot_replace_launch_outcome_or_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_result: object,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.sky.launch_result = launch_result
    real_temporary_directory = module.tempfile.TemporaryDirectory
    cleanup_calls = 0

    class CleanupFailure:
        def __init__(self, *args: object, **kwargs: object):
            self._inner = real_temporary_directory(*args, **kwargs)
            self.name = self._inner.name

        def cleanup(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            self._inner.cleanup()
            raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        module.tempfile,
        "TemporaryDirectory",
        CleanupFailure,
    )

    if isinstance(launch_result, ProcessCrash):
        with pytest.raises(ProcessCrash):
            module.run_qualification(
                _acquire_request(fixture, handoff),
                services=services.services,
            )
    else:
        outcome = module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
        assert outcome.exit_code == 0
        assert outcome.status == "accepted"

    assert cleanup_calls == 1
    assert services.events.count("sky:launch") == 1


def test_cleanup_failure_uses_a_best_effort_private_tree_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    temporary = module.tempfile.TemporaryDirectory(
        prefix="glm52-sky-sealed-launch-fallback-"
    )
    root = Path(temporary.name)
    (root / "sealed").write_bytes(b"authenticated")

    class CleanupFailure:
        name = temporary.name

        @staticmethod
        def cleanup() -> None:
            raise OSError("injected cleanup failure before removal")

    try:
        module._cleanup_sealed_launch(CleanupFailure())
        assert not root.exists()
    finally:
        temporary.cleanup()


def test_owned_submission_sources_do_not_require_datetime_utc() -> None:
    forbidden = "from datetime import " + "UTC"
    test_source = Path(__file__).read_text()
    compatibility_assignment = "datetime_module." + "UTC = UTC"
    assert forbidden not in SCRIPT.read_text()
    assert forbidden not in test_source
    assert ("strict" + "=True") not in SCRIPT.read_text()
    assert compatibility_assignment in test_source
    assert test_source.index(compatibility_assignment) < test_source.index(
        "from mlx_vq."
    )


def test_submitter_imports_and_renders_help_on_system_python_39() -> None:
    python = Path("/usr/bin/python3")
    if not python.is_file():
        pytest.skip("system Python is unavailable")
    version = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0 or version.stdout.strip() != "3.9":
        pytest.skip("system Python is not Python 3.9")

    dependency_stub_bootstrap = """
import importlib.abc
import importlib.util
import runpy
import sys
import types

class DependencyValue:
    def __call__(self, *args, **kwargs):
        return None

class StubModule(types.ModuleType):
    def __getattr__(self, name):
        value = DependencyValue()
        setattr(self, name, value)
        return value

class StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []

class StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "mlx_vq" or fullname.startswith("mlx_vq."):
            return importlib.util.spec_from_loader(
                fullname,
                StubLoader(),
                is_package=True,
            )
        return None

script = sys.argv[1]
sys.meta_path.insert(0, StubFinder())
sys.argv = [script, "--help"]
runpy.run_path(script, run_name="__main__")
"""
    result = subprocess.run(
        [str(python), "-c", dependency_stub_bootstrap, str(SCRIPT)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "qualification" in result.stdout


def test_nonlaunch_commands_do_not_invoke_source_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    real_build_sealed_launch_bundle = module._build_sealed_launch_bundle
    real_materialize_sealed_launch = module._materialize_sealed_launch

    def forbidden_read(*, task: Path) -> WorkerStartV2SourceSnapshot:
        raise AssertionError(f"nonlaunch command read source gate: {task}")

    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        forbidden_read,
        raising=False,
    )

    def forbidden_sealed_launch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("nonlaunch path constructed sealed launch data")

    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        forbidden_sealed_launch,
    )
    monkeypatch.setattr(
        module,
        "_materialize_sealed_launch",
        forbidden_sealed_launch,
    )
    validation_root = tmp_path / "validate"
    validation_root.mkdir()
    validation_fixture = _fixture(validation_root)
    validation_services = _services(validation_fixture)
    validated = module.run_qualification(
        _request(validation_fixture, action="validate-only"),
        services=validation_services.services,
    )
    assert validated.exit_code == 0

    prepare_root = tmp_path / "prepare"
    prepare_root.mkdir()
    prepare_fixture = _fixture(prepare_root)
    prepare_services = _services(prepare_fixture)
    prepared = module.run_qualification(
        _request(
            prepare_fixture,
            action="prepare-intent",
            output_handoff=prepare_root / "handoff.json",
        ),
        services=prepare_services.services,
    )
    assert prepared.exit_code == 0

    reconcile_root = tmp_path / "reconcile"
    reconcile_root.mkdir()
    reconcile_fixture = _fixture(reconcile_root)
    reconcile_services = _services(reconcile_fixture)
    handoff = _prepare(reconcile_fixture, reconcile_services, reconcile_root)
    exact = _worker_start_v2_source_snapshot()
    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        lambda *, task: exact,
    )
    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        real_build_sealed_launch_bundle,
    )
    monkeypatch.setattr(
        module,
        "_materialize_sealed_launch",
        real_materialize_sealed_launch,
    )
    acquired = module.run_qualification(
        _acquire_request(reconcile_fixture, handoff),
        services=reconcile_services.services,
    )
    assert acquired.exit_code == 0
    monkeypatch.setattr(
        module,
        "_read_worker_start_v2_source_snapshot",
        forbidden_read,
    )
    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        forbidden_sealed_launch,
    )
    monkeypatch.setattr(
        module,
        "_materialize_sealed_launch",
        forbidden_sealed_launch,
    )
    acquisition_key = _acquisition_key(reconcile_fixture)
    reconciled = module.run_qualification(
        module.QualificationRequest(
            action="reconcile",
            profile=PROFILE,
            descriptor=reconcile_fixture.paths.descriptor,
            acquisition_s3_uri=f"s3://{BUCKET}/{acquisition_key}",
        ),
        services=reconcile_services.services,
    )
    assert reconciled.exit_code == 0


@pytest.mark.parametrize(
    "decision_action",
    ["fail-closed", "reconcile-only"],
)
def test_nonlaunch_acquisition_decisions_never_construct_or_materialize_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision_action: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)

    @dataclass(frozen=True)
    class ForcedDecision:
        action: str
        reason: str

    def forbidden_sealed_launch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(
            f"{decision_action} constructed or materialized sealed launch data"
        )

    monkeypatch.setattr(
        module,
        "decide_submission_acquisition",
        lambda **_kwargs: ForcedDecision(
            action=decision_action,
            reason=f"forced {decision_action}",
        ),
    )
    monkeypatch.setattr(
        module,
        "_build_sealed_launch_bundle",
        forbidden_sealed_launch,
    )
    monkeypatch.setattr(
        module,
        "_materialize_sealed_launch",
        forbidden_sealed_launch,
    )

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:launch" not in services.events


@pytest.mark.parametrize("put_mode", ["lost", "malformed"])
def test_ambiguous_dynamic_binding_put_rereads_and_authenticates_the_winner(
    tmp_path: Path,
    put_mode: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    binding_key = module.dynamic_v2_job_binding_s3_key(intent=intent)
    services.s3.put_modes[binding_key] = put_mode

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"
    assert services.events.count(f"s3:put:{binding_key}") == 1
    assert services.events.count("sky:launch") == 1


def test_conditional_binding_race_accepts_a_different_authenticated_winner(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    binding_key = module.dynamic_v2_job_binding_s3_key(intent=intent)
    services.s3.put_modes[binding_key] = "race"
    winner: dict[str, object] = {}

    def replace_candidate() -> None:
        key, value = _build_coordinator_binding(
            services,
            bound_at=NOW + timedelta(seconds=1),
        )
        assert key == binding_key
        winner.update(value)
        services.s3.objects[(BUCKET, binding_key)] = _canonical(value)

    services.s3.after_put[binding_key] = replace_candidate

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert services.events.count(f"s3:put:{binding_key}") == 1
    accepted = next(
        json.loads(raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
    )
    assert accepted["job_binding"] == winner
    assert accepted["accepted_at"] == accepted["controller_observed_at"]


def test_conditional_binding_race_rejects_a_foreign_winner(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    binding_key = module.dynamic_v2_job_binding_s3_key(intent=intent)
    services.s3.put_modes[binding_key] = "race"

    def replace_candidate() -> None:
        _key, value = _build_coordinator_binding(
            services,
            bound_at=NOW + timedelta(seconds=1),
        )
        foreign = _rehash(
            {**value, "acquisition_body_sha256": "f" * 64},
            "job_binding_body_sha256",
        )
        services.s3.objects[(BUCKET, binding_key)] = _canonical(foreign)

    services.s3.after_put[binding_key] = replace_candidate

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert services.events.count(f"s3:put:{binding_key}") == 1
    assert not any(
        key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


@pytest.mark.parametrize("put_mode", ["race", "lost"])
def test_ambiguous_selector_put_is_reconcile_only_and_never_launches(
    tmp_path: Path,
    put_mode: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.events.clear()
    services.s3.put_modes[_acquisition_key(fixture)] = put_mode

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 75
    assert outcome.status == "reconcile-pending"
    assert "sky:launch" not in services.events
    assert services.sky.launches == []


@pytest.mark.parametrize(
    "launch_result",
    [0, 19, RuntimeError("Sky CLI failed"), TimeoutError("lost Sky CLI response")],
    ids=["zero", "nonzero", "failure", "timeout"],
)
def test_every_ordinary_sky_result_uses_the_same_exact_reconciler(
    tmp_path: Path,
    launch_result: object,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.sky.launch_result = launch_result

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"
    assert services.events.count("sky:launch") == 1
    assert outcome.detail["sky_job_id"] == 17
    private_repo = services.sky.launch_cwds[0]
    assert private_repo is not None
    assert not private_repo.parent.exists()


def test_zero_new_controller_rows_is_pending_and_never_relaunches(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.sky.add_controller_row = False

    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    second = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert first.exit_code == second.exit_code == 75
    assert services.events.count("sky:launch") == 1


@pytest.mark.parametrize(
    ("configure", "reason"),
    [
        (
            lambda service: setattr(service.sky, "extra_controller_rows", 1),
            "multiple",
        ),
        (
            lambda service: service.sky.row_overrides.update({"workspace": "foreign"}),
            "foreign",
        ),
    ],
)
def test_ambiguous_or_foreign_controller_rows_fail_closed(
    tmp_path: Path,
    configure: Any,
    reason: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    configure(services)

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert reason in str(outcome.detail["reason"]).lower()
    assert services.events.count("sky:launch") == 1


@pytest.mark.parametrize(
    "gate",
    ["active-p5", "active-job", "inactive-control-plane", "stale-reviewed"],
)
def test_live_gate_regressions_fail_before_selector_and_launch(
    tmp_path: Path,
    gate: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.events.clear()
    if gate == "active-p5":
        services.ec2.active_p5_ids.append("i-0123456789abcdef0")
    elif gate == "active-job":
        services.ssm.rows.append(
            _history_row(
                job_id=9,
                status="RUNNING",
                submitted_at=NOW,
            )
        )
    elif gate == "inactive-control-plane":
        services.cfn.active = False
    else:
        services.clock.advance(61)

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code in {70, 78}
    assert "sky:launch" not in services.events
    assert not any(
        event.endswith("/SUBMISSION_ACQUIRED.json") and event.startswith("s3:put:")
        for event in services.events
    )


@pytest.mark.parametrize(
    "crash_fragment",
    [
        "/observations/",
        "/DYNAMIC_JOB_BINDING.json",
        "/SKYPILOT_SUBMISSION_ACCEPTED.json",
    ],
)
def test_receipt_publication_crashes_converge_without_second_launch(
    tmp_path: Path,
    crash_fragment: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains[crash_fragment] = ProcessCrash(crash_fragment)

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )

    services.s3.crash_contains.clear()
    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"
    assert services.events.count("sky:launch") == 1


def test_deadline_without_a_dynamic_binding_never_publishes_acceptance(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/observations/"] = ProcessCrash("observation")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    services.clock.value = MUST_START_BY

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert services.events.count("sky:launch") == 1
    assert not any(
        key.endswith("/DYNAMIC_JOB_BINDING.json")
        or key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


def test_existing_observation_survives_live_controller_status_advance(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/observations/"] = ProcessCrash("observation")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    observation_key = next(
        key
        for bucket, key in services.s3.objects
        if bucket == BUCKET and "/observations/" in key
    )
    observation_raw = services.s3.objects[(BUCKET, observation_key)]
    observation_put_count = services.events.count(f"s3:put:{observation_key}")
    new_row = services.ssm.rows[-1]
    assert new_row["sky_job_id"] == 17
    new_row["controller_status"] = "STARTING"
    new_row["start_at"] = _iso(NOW + timedelta(seconds=1))
    new_row["worker_cluster_name"] = "sky-glm52-qualification-worker"

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 0
    assert retry.status == "accepted"
    assert services.events.count("sky:launch") == 1
    assert services.s3.objects[(BUCKET, observation_key)] == observation_raw
    assert services.events.count(f"s3:put:{observation_key}") == observation_put_count
    accepted = next(
        json.loads(raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
    )
    assert accepted["controller_observation"] == json.loads(observation_raw)
    assert accepted["job_binding"]["controller_status"] == "PENDING"


def test_coordinator_pending_binding_precedes_later_starting_observation(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.sky.launch_result = ProcessCrash("after controller submission")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )

    binding_key, binding = _build_coordinator_binding(
        services,
        bound_at=services.clock(),
    )
    binding_raw = _canonical(binding)
    services.s3.objects[(BUCKET, binding_key)] = binding_raw
    row = services.ssm.rows[-1]
    row["controller_status"] = "STARTING"
    row["start_at"] = _iso(services.clock())
    row["worker_cluster_name"] = "sky-glm52-qualification-worker"
    services.clock.advance(1)

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 0
    assert retry.status == "accepted"
    assert services.events.count("sky:launch") == 1
    assert services.s3.objects[(BUCKET, binding_key)] == binding_raw
    assert f"s3:put:{binding_key}" not in services.events
    accepted = next(
        json.loads(raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
    )
    assert accepted["job_binding"]["controller_status"] == "PENDING"
    assert accepted["controller_status"] == "STARTING"
    assert accepted["accepted_at"] == accepted["controller_observed_at"]


def test_fractional_controller_timestamp_is_preserved_in_exact_receipts(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    fractional = (NOW + timedelta(seconds=1)).replace(microsecond=654321)
    services.sky.row_overrides = {
        "controller_status": "RUNNING",
        "controller_submitted_at": _iso(fractional),
        "start_at": _iso(fractional),
    }

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    observation = next(
        json.loads(raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and "/observations/" in key
    )
    assert observation["submitted_at"] == _iso(fractional)
    assert observation["start_at"] == _iso(fractional)


def test_foreign_self_hashed_observation_stops_before_binding(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/observations/"] = ProcessCrash("observation")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    old_key = next(
        key
        for bucket, key in services.s3.objects
        if bucket == BUCKET and "/observations/" in key
    )
    old = json.loads(services.s3.objects.pop((BUCKET, old_key)))
    intent = next(
        json.loads(raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/SKYPILOT_SUBMISSION_INTENT.json")
    )
    foreign_run = "foreign-run"
    foreign = module.build_must_start_controller_observation(
        run_id=foreign_run,
        managed_mode="qualification",
        account_id=ACCOUNT_ID,
        region=REGION,
        bucket=BUCKET,
        descriptor_body_sha256=str(intent["descriptor_body_sha256"]),
        submission_body_sha256=str(intent["intent_body_sha256"]),
        sky_job_name=f"{foreign_run}-qualification",
        must_start_by=str(intent["must_start_by"]),
        target_job_id=int(old["target_job_id"]),
        workspace="default",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        status=str(old["status"]),
        schedule_state=str(old["schedule_state"]),
        submitted_at=str(old["submitted_at"]),
        start_at=old["start_at"],
        worker_cluster_name=old["worker_cluster_name"],
        recovery_count=int(old["recovery_count"]),
        observed_at=str(old["observed_at"]),
    )
    foreign_key = (
        old_key.rsplit("/", 1)[0]
        + "/"
        + str(foreign["observation_body_sha256"])
        + ".json"
    )
    services.s3.objects[(BUCKET, foreign_key)] = module.must_start_canonical_bytes(
        foreign
    )

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert not any(
        key.endswith("/JOB_BINDING.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert services.events.count("sky:launch") == 1


def test_same_intent_observation_cannot_fabricate_mutable_controller_state(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/observations/"] = ProcessCrash("observation")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    old_key = next(
        key
        for bucket, key in services.s3.objects
        if bucket == BUCKET and "/observations/" in key
    )
    old = json.loads(services.s3.objects.pop((BUCKET, old_key)))
    fabricated = module.build_must_start_controller_observation(
        run_id=str(old["run_id"]),
        managed_mode=str(old["managed_mode"]),
        account_id=str(old["account_id"]),
        region=str(old["region"]),
        bucket=str(old["bucket"]),
        descriptor_body_sha256=str(old["descriptor_body_sha256"]),
        submission_body_sha256=str(old["submission_body_sha256"]),
        sky_job_name=str(old["sky_job_name"]),
        must_start_by=str(old["must_start_by"]),
        target_job_id=int(old["target_job_id"]),
        workspace=str(old["workspace"]),
        controller_instance_id=str(old["controller_instance_id"]),
        controller_instance_type=str(old["controller_instance_type"]),
        controller_profile_arn=str(old["controller_profile_arn"]),
        controller_cluster_name=str(old["controller_cluster_name"]),
        status="SUCCEEDED",
        schedule_state="DONE",
        submitted_at=str(old["submitted_at"]),
        start_at=str(old["submitted_at"]),
        worker_cluster_name="fabricated-worker-cluster",
        recovery_count=99,
        observed_at=str(old["observed_at"]),
    )
    fabricated_key = (
        old_key.rsplit("/", 1)[0]
        + "/"
        + str(fabricated["observation_body_sha256"])
        + ".json"
    )
    services.s3.objects[(BUCKET, fabricated_key)] = module.must_start_canonical_bytes(
        fabricated
    )

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert not any(
        key.endswith("/JOB_BINDING.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert services.events.count("sky:launch") == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "bad-hash",
        "foreign-intent",
        "foreign-baseline",
        "foreign-acquisition",
        "wrong-job",
        "identity-drift",
        "coordinate-drift",
        "v1-at-dynamic-key",
    ],
)
def test_foreign_or_malformed_binding_stops_before_accepted(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/DYNAMIC_JOB_BINDING.json"] = ProcessCrash("binding")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    binding_key = next(
        key
        for bucket, key in services.s3.objects
        if bucket == BUCKET and key.endswith("/DYNAMIC_JOB_BINDING.json")
    )
    binding = json.loads(services.s3.objects[(BUCKET, binding_key)])
    if mutation == "v1-at-dynamic-key":
        _legacy_key, foreign = _legacy_binding_for_remote(services)
    else:
        field, value = {
            "bad-hash": ("controller_status", "RUNNING"),
            "foreign-intent": ("intent_body_sha256", "f" * 64),
            "foreign-baseline": ("controller_baseline_body_sha256", "f" * 64),
            "foreign-acquisition": ("acquisition_body_sha256", "f" * 64),
            "wrong-job": ("sky_job_id", 18),
            "identity-drift": (
                "controller_identity",
                "arn:aws:iam::246813579024:role/foreign-controller",
            ),
            "coordinate-drift": (
                "controller_instance_id",
                "i-0123456789abcdef0",
            ),
        }[mutation]
        foreign = {**binding, field: value}
        if mutation != "bad-hash":
            foreign = _rehash(foreign, "job_binding_body_sha256")
    services.s3.objects[(BUCKET, binding_key)] = _canonical(foreign)

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert not any(
        key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert services.events.count("sky:launch") == 1


def test_legacy_v1_only_binding_fails_closed_without_dynamic_fallback(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/observations/"] = ProcessCrash("observation")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    legacy_key, legacy = _legacy_binding_for_remote(services)
    services.s3.objects[(BUCKET, legacy_key)] = _canonical(legacy)

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert f"s3:get:{legacy_key}" not in services.events
    assert not any(
        key.endswith("/DYNAMIC_JOB_BINDING.json")
        or key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


def test_simultaneous_legacy_and_dynamic_bindings_fail_before_acceptance(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/DYNAMIC_JOB_BINDING.json"] = ProcessCrash("binding")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    services.s3.crash_contains.clear()
    legacy_key, legacy = _legacy_binding_for_remote(services)
    services.s3.objects[(BUCKET, legacy_key)] = _canonical(legacy)

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert f"s3:get:{legacy_key}" not in services.events
    assert not any(
        key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


def test_legacy_binding_race_after_dynamic_audit_fails_before_acceptance(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    must_start_prefix = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}/"
    )

    def inject_legacy_after_list_snapshot() -> None:
        dynamic_exists = any(
            bucket == BUCKET and key.endswith("/DYNAMIC_JOB_BINDING.json")
            for bucket, key in services.s3.objects
        )
        if dynamic_exists:
            legacy_key, legacy = _legacy_binding_for_remote(services)
            services.s3.objects.setdefault(
                (BUCKET, legacy_key),
                _canonical(legacy),
            )

    services.s3.after_list[must_start_prefix] = inject_legacy_after_list_snapshot

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert any(
        key.endswith("/DYNAMIC_JOB_BINDING.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert any(
        key.endswith("/JOB_BINDING.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert not any(
        key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


def test_legacy_binding_race_after_final_pre_put_audit_fails_operationally(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    must_start_prefix = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}/"
    )
    list_count = 0

    def inject_legacy_after_final_pre_put_list_snapshot() -> None:
        nonlocal list_count
        list_count += 1
        if list_count == 4:
            legacy_key, legacy = _legacy_binding_for_remote(services)
            services.s3.objects[(BUCKET, legacy_key)] = _canonical(legacy)

    services.s3.after_list[must_start_prefix] = (
        inject_legacy_after_final_pre_put_list_snapshot
    )

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert list_count >= 5
    assert outcome.exit_code == 70
    assert any(
        key.endswith("/DYNAMIC_JOB_BINDING.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert any(
        key.endswith("/JOB_BINDING.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )
    assert any(
        key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


def test_accepted_replay_refuses_a_late_parallel_legacy_binding(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 0
    legacy_key, legacy = _legacy_binding_for_remote(services)
    services.s3.objects[(BUCKET, legacy_key)] = _canonical(legacy)
    services.events.clear()

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert "sky:launch" not in services.events
    assert f"s3:get:{legacy_key}" not in services.events


def test_accepted_replay_refuses_legacy_inserted_after_initial_audit(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 0
    _intent_key, intent = _remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    must_start_prefix = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}/"
    )
    legacy_key, legacy = _legacy_binding_for_remote(services)

    def inject_legacy_after_replay_audit_snapshot() -> None:
        services.s3.objects[(BUCKET, legacy_key)] = _canonical(legacy)

    services.s3.after_list[must_start_prefix] = (
        inject_legacy_after_replay_audit_snapshot
    )
    services.events.clear()

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert "sky:launch" not in services.events
    assert f"s3:get:{legacy_key}" not in services.events
    assert any(
        key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
        for bucket, key in services.s3.objects
        if bucket == BUCKET
    )


def test_selector_only_process_crash_retries_through_standalone_reconcile(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.sky.launch_result = ProcessCrash("after launch invocation")

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    private_repo = services.sky.launch_cwds[0]
    assert private_repo is not None
    assert not private_repo.parent.exists()

    acquisition_key = _acquisition_key(fixture)
    outcome = module.run_qualification(
        module.QualificationRequest(
            action="reconcile",
            profile=PROFILE,
            descriptor=fixture.paths.descriptor,
            acquisition_s3_uri=f"s3://{BUCKET}/{acquisition_key}",
        ),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"
    assert services.events.count("sky:launch") == 1


def test_multiple_accepted_prefix_members_fail_closed_on_retry(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 0
    accepted_entries = [
        (key, raw)
        for (bucket, key), raw in services.s3.objects.items()
        if bucket == BUCKET and key.endswith("/SKYPILOT_SUBMISSION_ACCEPTED.json")
    ]
    assert len(accepted_entries) == 1
    _, accepted_raw = accepted_entries[0]
    duplicate_key = (
        f"campaigns/{RUN_ID}/submissions/qualification/accepted/"
        f"{'0' * 64}/SKYPILOT_SUBMISSION_ACCEPTED.json"
    )
    services.s3.objects[(BUCKET, duplicate_key)] = accepted_raw

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 70
    assert "multiple accepted" in str(retry.detail["reason"])
    assert services.events.count("sky:launch") == 1


def test_authenticated_accepted_retry_does_not_depend_on_live_controller(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    first = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert first.exit_code == 0
    services.ssm.rows = []
    services.clock.value = MUST_START_BY + timedelta(seconds=1)
    services.events.clear()

    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert retry.exit_code == 0
    assert retry.status == "accepted"
    assert "ssm:send-observe" not in services.events
    assert "sky:launch" not in services.events


def test_two_content_addressed_intents_race_on_one_stable_selector(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    services.clock.value = NOW - timedelta(seconds=1)
    first_handoff = _prepare(fixture, services, tmp_path)
    services.clock.advance(1)
    second_handoff_path = tmp_path / "second-handoff.json"
    second = module.run_qualification(
        _request(
            fixture,
            action="prepare-intent",
            output_handoff=second_handoff_path,
        ),
        services=services.services,
    )
    assert second.exit_code == 0
    second_handoff = json.loads(second_handoff_path.read_bytes())
    assert first_handoff["intent_s3_uri"] != second_handoff["intent_s3_uri"]
    services.sky.add_controller_row = False

    first = module.run_qualification(
        _acquire_request(fixture, first_handoff),
        services=services.services,
    )
    selected_raw = services.s3.objects[(BUCKET, _acquisition_key(fixture))]
    selected = json.loads(selected_raw)
    second = module.run_qualification(
        _acquire_request(fixture, second_handoff),
        services=services.services,
    )

    assert first.exit_code == second.exit_code == 75
    assert (
        selected["intent_key"]
        == str(first_handoff["intent_s3_uri"]).split(
            f"s3://{BUCKET}/",
            1,
        )[1]
    )
    assert services.events.count("sky:launch") == 1


@pytest.mark.parametrize(
    "regression",
    ["latest-spend", "active-p5", "inactive-control-plane", "deadline"],
)
def test_post_acquisition_live_regression_fails_before_sky(
    tmp_path: Path,
    regression: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    selector_key = _acquisition_key(fixture)
    latest_key = f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json"

    def regress() -> None:
        if regression == "latest-spend":
            services.s3.objects[(BUCKET, latest_key)] = b"{}\n"
        elif regression == "active-p5":
            services.ec2.active_p5_ids.append("i-0123456789abcdef0")
        elif regression == "inactive-control-plane":
            services.cfn.active = False
        else:
            services.clock.value = MUST_START_BY

    services.s3.after_put[selector_key] = regress
    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 70
    assert "sky:launch" not in services.events
    assert (BUCKET, selector_key) in services.s3.objects


@pytest.mark.parametrize(
    "crash_fragment",
    [
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json",
        "QUALIFICATION_SUBMISSION_READY.json",
        "SKYPILOT_SUBMISSION_INTENT.json",
        "CONTROLLER_BASELINE.json",
        "CONTROL_PLANE_READY.json",
    ],
)
def test_pre_acquisition_publication_crashes_remain_inert_and_retryable(
    tmp_path: Path,
    crash_fragment: str,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff_path = tmp_path / "handoff.json"
    services.s3.crash_contains[crash_fragment] = ProcessCrash(crash_fragment)

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _request(
                fixture,
                action="prepare-intent",
                output_handoff=handoff_path,
            ),
            services=services.services,
        )
    assert "sky:launch" not in services.events
    assert (BUCKET, _acquisition_key(fixture)) not in services.s3.objects

    services.s3.crash_contains.clear()
    retry = module.run_qualification(
        _request(
            fixture,
            action="prepare-intent",
            output_handoff=handoff_path,
        ),
        services=services.services,
    )
    assert retry.exit_code == 0
    assert retry.status == "intent-prepared"
    assert "sky:launch" not in services.events


def test_crash_after_acquisition_commit_never_launches_on_retry(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    services.s3.crash_contains["/SUBMISSION_ACQUIRED.json"] = ProcessCrash(
        "selector committed"
    )

    with pytest.raises(ProcessCrash):
        module.run_qualification(
            _acquire_request(fixture, handoff),
            services=services.services,
        )
    assert "sky:launch" not in services.events

    services.s3.crash_contains.clear()
    retry = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )
    assert retry.exit_code == 75
    assert "sky:launch" not in services.events


def test_complete_paginated_lists_are_used_for_selector_and_receipts(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    services.s3.page_size = 1
    handoff = _prepare(fixture, services, tmp_path)

    outcome = module.run_qualification(
        _acquire_request(fixture, handoff),
        services=services.services,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "accepted"


def _shell_harness(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scripts = tmp_path / "aws/glm52-gpu/scripts"
    scripts.mkdir(parents=True)
    shell = scripts / "submit_sky_campaign.sh"
    shell.write_bytes(SHELL.read_bytes())
    shell.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    for name in (
        "assert_rnd_aws_account.sh",
        "assert_sns_email_confirmed.sh",
    ):
        helper = scripts / name
        helper.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' '[\"{name}\"]' >> \"$FAKE_SHELL_LOG\"\n"
        )
        helper.chmod(0o755)
    submitter = scripts / "submit_sky_campaign.py"
    submitter.write_text("raise SystemExit('real Python submitter was invoked')\n")
    fake_sky = tmp_path / "sky"
    fake_sky.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '[\"sky\",\"--version\"]' >> \"$FAKE_SHELL_LOG\"\n"
        "printf '%s\\n' 'sky, version 0.13.0'\n"
    )
    fake_sky.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
with Path(os.environ["FAKE_SHELL_LOG"]).open("a") as handle:
    handle.write(json.dumps(["python", *sys.argv[1:]]) + "\\n")
"""
    )
    fake_python.chmod(0o755)
    return shell, log, fake_sky, fake_python


@pytest.mark.parametrize(
    "args",
    (
        [],
        ["--dry-run"],
        ["--cache-seed"],
        ["--unknown"],
        ["--qualification"],
    ),
)
def test_shell_hard_disables_every_non_qualification_invocation_before_side_effect(
    tmp_path: Path,
    args: list[str],
) -> None:
    shell, log, _sky, _python = _shell_harness(tmp_path)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SKY_BIN", "KEEP_PYTHON"}
    }
    environment["FAKE_SHELL_LOG"] = str(log)

    result = subprocess.run(
        ["bash", str(shell), *args],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 64
    assert "usage:" in result.stderr
    assert not log.exists()


def test_shell_routes_valid_qualification_with_exact_full_argv(
    tmp_path: Path,
) -> None:
    shell, log, sky, python = _shell_harness(tmp_path)
    args = [
        "--qualification",
        "prepare-intent",
        "--descriptor",
        "/exact/descriptor.json",
        "--flag=value with spaces",
    ]

    result = subprocess.run(
        ["bash", str(shell), *args],
        env={
            **os.environ,
            "FAKE_SHELL_LOG": str(log),
            "SKY_BIN": str(sky),
            "KEEP_PYTHON": str(python),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        ["assert_rnd_aws_account.sh"],
        ["assert_sns_email_confirmed.sh"],
        ["sky", "--version"],
        [
            "python",
            str(shell.with_name("submit_sky_campaign.py")),
            "qualification",
            *args[1:],
        ],
    ]


def test_real_sky_launch_has_a_bounded_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sky = bin_dir / "sky"
    _write_fake_sky_entrypoint(sky)
    task = tmp_path / "task.yaml"
    task.write_text("name: exact\n")
    config = tmp_path / "config.yaml"
    config.write_text("allowed_clouds: [aws]\n")
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        command = args[0]
        if command == [str(sky), "--version"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="skypilot, version 0.13.0\n",
                stderr="",
            )
        if isinstance(command, list) and command[0] == str(
            sky.with_name("python")
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "record_type": (
                            "glm52_skypilot_parser_validation_v1"
                        ),
                        "skypilot_version": "0.13.0",
                        "status": "passed",
                    }
                ),
                stderr="",
            )
        captured["command"] = args[0]
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    client = module._SubprocessSky(str(sky))
    client.validate_control_plane(task=task, config=config)

    with pytest.raises(subprocess.TimeoutExpired):
        client.launch(
            ["jobs", "launch"],
            cwd=tmp_path,
        )

    assert captured["timeout"] == module.SKY_LAUNCH_TIMEOUT_SECONDS
    assert captured["cwd"] == tmp_path
    assert captured["command"] == [str(sky), "jobs", "launch"]


def _write_fake_sky_entrypoint(sky: Path) -> None:
    python = sky.with_name("python")
    python.write_text("#!/usr/bin/env bash\nexit 0\n")
    python.chmod(0o755)
    sky.write_text(
        f"#!{python}\n"
        "import sys\n"
        "from sky.cli import cli\n"
        "if __name__ == \"__main__\":\n"
        "    sys.exit(cli())\n"
    )
    sky.chmod(0o755)


@pytest.mark.parametrize(
    "reported",
    (
        "skypilot, version 0.12.0",
        "forged skypilot 0.13.0",
        "skypilot, version 0.13.0-malicious",
    ),
)
def test_python_sky_rejects_nonexact_version_before_parser_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported: str,
) -> None:
    """Break caught: substring version output admitted a foreign executable."""

    module = _module()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sky = bin_dir / "sky"
    _write_fake_sky_entrypoint(sky)
    task = tmp_path / "task.yaml"
    task.write_text("name: exact\n")
    config = tmp_path / "config.yaml"
    config.write_text("allowed_clouds: [aws]\n")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=reported + "\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(module.SubmissionIntegrationError, match="SkyPilot"):
        module._SubprocessSky(str(sky)).validate_control_plane(
            task=task,
            config=config,
        )
    assert calls == [[str(sky), "--version"]]


def test_python_sky_rejects_non_sky_path_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: Python formed a launch command from an absent Sky path."""

    module = _module()
    task = tmp_path / "task.yaml"
    task.write_text("name: exact\n")
    config = tmp_path / "config.yaml"
    config.write_text("allowed_clouds: [aws]\n")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    with pytest.raises(module.SubmissionIntegrationError, match="executable"):
        module._SubprocessSky(
            str(tmp_path / "missing-sky")
        ).validate_control_plane(task=task, config=config)
    assert calls == []


def test_python_sky_reports_observed_exact_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: Python hard-coded a version it never observed."""

    module = _module()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sky = bin_dir / "sky"
    _write_fake_sky_entrypoint(sky)
    task = tmp_path / "task.yaml"
    task.write_text("name: exact\n")
    config = tmp_path / "config.yaml"
    config.write_text("allowed_clouds: [aws]\n")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        if command == [str(sky), "--version"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="skypilot, version 0.13.0\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "record_type": "glm52_skypilot_parser_validation_v1",
                    "skypilot_version": "0.13.0",
                    "status": "passed",
                }
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._SubprocessSky(str(sky)).validate_control_plane(
        task=task,
        config=config,
    )

    assert result["skypilot_version"] == "0.13.0"
    assert calls[0] == [str(sky), "--version"]
    assert calls[1][0] == str(bin_dir / "python")


def test_python_sky_rejects_exact_version_spoof_from_non_sky_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: exact version text alone did not prove a Sky entry point."""

    module = _module()
    sky = tmp_path / "sky"
    sky.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'skypilot, version 0.13.0'\n"
    )
    sky.chmod(0o755)
    task = tmp_path / "task.yaml"
    task.write_text("name: exact\n")
    config = tmp_path / "config.yaml"
    config.write_text("allowed_clouds: [aws]\n")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    with pytest.raises(module.SubmissionIntegrationError, match="entry point"):
        module._SubprocessSky(str(sky)).validate_control_plane(
            task=task,
            config=config,
        )
    assert calls == []


def test_python_sky_launch_requires_prior_exact_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: callers could execute an unvalidated Sky binary directly."""

    module = _module()
    sky = tmp_path / "sky"
    sky.write_text("#!/usr/bin/env bash\n")
    sky.chmod(0o755)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    with pytest.raises(module.SubmissionIntegrationError, match="validated"):
        module._SubprocessSky(str(sky)).launch(
            ["jobs", "launch"],
            cwd=tmp_path,
        )
    assert calls == []


def test_production_subprocess_sky_receives_only_sealed_paths_and_private_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    services = _services(fixture)
    handoff = _prepare(fixture, services, tmp_path)
    request = _acquire_request(fixture, handoff)
    local = module._load_local(request, services.services)
    intent, _baseline, _control_ready = module._load_prepared_artifacts(
        request,
        services.services,
        local=local,
    )
    original_arguments = module._launch_arguments(
        request=request,
        local=local,
        intent=intent,
    )
    source_snapshot = _worker_start_v2_source_snapshot()
    bundle = module._build_sealed_launch_bundle(
        source_snapshot=source_snapshot,
        config_raw=fixture.paths.config.read_bytes(),
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sky = bin_dir / "sky"
    _write_fake_sky_entrypoint(sky)
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        command = list(args[0])
        if command == [str(sky), "--version"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="skypilot, version 0.13.0\n",
                stderr="",
            )
        if command[0] == str(sky.with_name("python")):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "record_type": (
                            "glm52_skypilot_parser_validation_v1"
                        ),
                        "skypilot_version": "0.13.0",
                        "status": "passed",
                    }
                ),
                stderr="",
            )
        cwd = Path(kwargs["cwd"])
        config = Path(command[command.index("--config") + 1])
        task = Path(command[-1])
        captured.update(
            {
                "command": command,
                "cwd": cwd,
                "config_raw": config.read_bytes(),
                "task_raw": task.read_bytes(),
                "mount_raw": {
                    path: (cwd / path).read_bytes()
                    for path in SEALED_MOUNT_RELATIVE_PATHS
                },
            }
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with module._materialize_sealed_launch(bundle) as materialized:
        client = module._SubprocessSky(str(sky))
        client.validate_control_plane(
            task=materialized.task_path,
            config=materialized.config_path,
        )
        sealed_arguments = module._sealed_launch_arguments(
            arguments=original_arguments,
            local=local,
            materialized=materialized,
        )
        client.launch(
            sealed_arguments,
            cwd=materialized.repository_root,
        )
        assert captured["cwd"] == materialized.repository_root
        assert captured["command"] == [str(sky), *sealed_arguments]
        assert str(fixture.paths.config) not in sealed_arguments
        assert str(fixture.paths.task) not in sealed_arguments
        assert captured["config_raw"] == bundle.config_raw
        assert captured["task_raw"] == bundle.task_raw
        assert captured["mount_raw"] == dict(bundle.mount_sources)

    for index, (original, sealed) in enumerate(
        zip(original_arguments, sealed_arguments)
    ):
        if index not in {
            original_arguments.index("--config") + 1,
            len(original_arguments) - 1,
        }:
            assert sealed == original


def test_cli_service_construction_failure_uses_structured_exit_70(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("{}\n")

    def broken_factory(**_kwargs: object) -> object:
        raise RuntimeError("profile construction failed")

    exit_code = module.main(
        [
            "qualification",
            "reconcile",
            "--profile",
            PROFILE,
            "--descriptor",
            str(descriptor),
        ],
        services_factory=broken_factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 70
    assert json.loads(captured.err) == {
        "exit_code": 70,
        "reason": "profile construction failed",
        "status": "fail-closed",
    }


def test_cli_usage_errors_use_exit_64() -> None:
    module = _module()
    with pytest.raises(SystemExit) as error:
        module.main(["qualification"])
    assert error.value.code == 64


def _cache_seed_cli_argv(action: str = "validate-only") -> list[str]:
    return [
        "cache-seed",
        action,
        "--profile",
        PROFILE,
        "--descriptor",
        "/pins/descriptor.json",
        "--approval",
        "/pins/approval.json",
        "--staged-ready",
        "/pins/staged-ready.json",
        "--staged-ready-version-id",
        "version-staged-ready",
        "--rehearsal-evidence",
        "/pins/rehearsal.json",
        "--task",
        "/pins/task.yaml",
        "--config",
        "/pins/config.yaml",
        "--sky-bin",
        "/opt/keep/skypilot-0.13.0/bin/sky",
    ]


@pytest.mark.parametrize("action", ["validate-only", "acquire-and-launch"])
def test_cache_seed_cli_accepts_only_exact_actions_and_constructs_services(
    action: str,
) -> None:
    """Catches the cache-seed mode or either supported action being absent."""

    module = _module()
    factory_calls: list[dict[str, object]] = []

    def services_factory(**kwargs: object) -> object:
        factory_calls.append(dict(kwargs))
        return module.SubmissionServices(
            sts=FakeSts([]),
            s3=None,
            ec2=None,
            ssm=None,
            cloudformation=None,
            sky=None,
            clock=lambda: NOW,
            sleep=lambda _seconds: None,
        )

    assert module.main(
        _cache_seed_cli_argv(action),
        services_factory=services_factory,
    ) == 64
    assert factory_calls == [
        {
            "profile": PROFILE,
            "sky_bin": "/opt/keep/skypilot-0.13.0/bin/sky",
        }
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["production", "validate-only"],
        ["cache-seed", "prepare-intent"],
        ["cache-seed", "reconcile"],
        ["qualification", "cache-seed"],
    ],
)
def test_cache_seed_cli_rejects_unsupported_mode_actions_before_services(
    argv: list[str],
) -> None:
    """Catches production or cross-mode actions reaching service construction."""

    module = _module()

    def forbidden_factory(**_kwargs: object) -> object:
        pytest.fail("usage rejection must precede services_factory")

    with pytest.raises(SystemExit) as error:
        module.main(argv, services_factory=forbidden_factory)
    assert error.value.code == 64


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--seed-descriptor", "/pins/seed.json"),
        ("--cache-seed-accepted", "/pins/accepted.json"),
        ("--gpu-spend-snapshot", "/pins/spend.json"),
        ("--qualification-ready", "/pins/ready.json"),
        ("--output-handoff", "/pins/handoff.json"),
        ("--intent-s3-uri", "s3://bucket/intent"),
        ("--must-start-ready-s3-uri", "s3://bucket/ready"),
        ("--acquisition-s3-uri", "s3://bucket/acquisition"),
    ],
)
def test_cache_seed_cli_rejects_each_qualification_flag_before_services(
    flag: str,
    value: str,
) -> None:
    """Catches qualification-only authority pins leaking into cache seed."""

    module = _module()

    def forbidden_factory(**_kwargs: object) -> object:
        pytest.fail("forbidden flag must be rejected before services_factory")

    with pytest.raises(SystemExit) as error:
        module.main(
            [*_cache_seed_cli_argv(), flag, value],
            services_factory=forbidden_factory,
        )
    assert error.value.code == 64


@pytest.mark.parametrize(
    "replacement",
    [None, "", "null"],
)
def test_cache_seed_cli_requires_exact_nonnull_staged_version_before_services(
    replacement: str | None,
) -> None:
    """Catches absent, empty, or null staged VersionIds before AWS clients."""

    module = _module()
    argv = _cache_seed_cli_argv()
    index = argv.index("--staged-ready-version-id")
    if replacement is None:
        del argv[index : index + 2]
    else:
        argv[index + 1] = replacement

    def forbidden_factory(**_kwargs: object) -> object:
        pytest.fail("VersionId rejection must precede services_factory")

    with pytest.raises(SystemExit) as error:
        module.main(argv, services_factory=forbidden_factory)
    assert error.value.code == 64


def test_qualification_rejects_cache_seed_version_flag_before_services() -> None:
    """Catches cache-seed-only VersionIds changing qualification arguments."""

    module = _module()

    def forbidden_factory(**_kwargs: object) -> object:
        pytest.fail("cross-mode VersionId must be rejected before services_factory")

    with pytest.raises(SystemExit) as error:
        module.main(
            [
                "qualification",
                "validate-only",
                "--profile",
                PROFILE,
                "--descriptor",
                "/pins/descriptor.json",
                "--staged-ready-version-id",
                "version-staged-ready",
            ],
            services_factory=forbidden_factory,
        )
    assert error.value.code == 64


def test_real_cloudformation_adapter_rejects_legacy_outputs_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class LegacyClient:
        def describe_stacks(self, **_kwargs: object) -> object:
            pytest.fail("legacy stack must not be inspected without a v2 map")

    monkeypatch.delenv("GLM52_SKY_DYNAMIC_OUTPUT_MAP_JSON", raising=False)
    inspector = module._BotoCloudFormationInspector(LegacyClient())
    assert (
        inspector.inspect_dynamic_must_start(
            run_id=RUN_ID,
            managed_mode="qualification",
            intent_body_sha256="1" * 64,
            controller_baseline_body_sha256="2" * 64,
        )
        is None
    )
    assert (
        inspector.inspect_cache_seed_launch_capability(
            run_id=RUN_ID,
            managed_mode="cache-seed",
            descriptor_file_sha256="3" * 64,
            descriptor_body_sha256="4" * 64,
            staged_readiness_version_id="version-staged-ready",
        )
        is None
    )


def test_default_services_builds_dedicated_single_attempt_claim_client(
) -> None:
    """Catches claim PUT sharing the ordinary retry-enabled S3 client."""

    script = """
import importlib.util
import json
from pathlib import Path
import sys
import boto3
from botocore.config import Config

calls = []
class Client:
    def __init__(self, service):
        self.service = service

class Session:
    def client(self, service, **kwargs):
        calls.append((service, kwargs))
        return Client(service)
boto3.Session = lambda **kwargs: Session()
path = Path("aws/glm52-gpu/scripts/submit_sky_campaign.py").resolve()
spec = importlib.util.spec_from_file_location("_claim_client_probe", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
services = module._default_services(
    profile="keep-gpu",
    sky_bin="/opt/keep/skypilot-0.13.0/bin/sky",
)
s3_calls = [kwargs for service, kwargs in calls if service == "s3"]
assert len(s3_calls) == 2
assert s3_calls[0] == {}
config = s3_calls[1]["config"]
assert isinstance(config, Config)
assert config.retries["mode"] == "standard"
assert config.retries["total_max_attempts"] == 1
assert services.claim_put_s3 is not services.s3
assert services.ssm is services.cloudformation.ssm_client
assert services.cloudformation.client.service == "cloudformation"
assert services.cloudformation.lambda_client.service == "lambda"
assert services.cloudformation.iam_client.service == "iam"
assert services.cloudformation.events_client.service == "events"
assert services.cloudformation.scheduler_client.service == "scheduler"
assert services.cloudformation.sqs_client.service == "sqs"
assert services.cloudformation.logs_client.service == "logs"
assert services.cloudformation.ssm_client.service == "ssm"
assert [service for service, _kwargs in calls].count("ssm") == 1
print(json.dumps(config.retries, sort_keys=True))
"""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--with",
            "boto3",
            "python",
            "-c",
            script,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "mode": "standard",
        "total_max_attempts": 1,
    }


def test_unfrozen_cloudformation_output_mapping_cannot_authorize_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class OutputOnlyClient:
        def describe_stacks(self, **_kwargs: object) -> object:
            pytest.fail("unfrozen outputs must not be treated as live readiness")

    mapping = {
        field: f"Output{index}"
        for index, field in enumerate(
            sorted(module._BotoCloudFormationInspector._MAPPING_FIELDS)
        )
    }
    monkeypatch.setenv(
        "GLM52_SKY_DYNAMIC_OUTPUT_MAP_JSON",
        json.dumps(mapping),
    )
    inspector = module._BotoCloudFormationInspector(OutputOnlyClient())

    assert (
        inspector.inspect_dynamic_must_start(
            run_id=RUN_ID,
            managed_mode="qualification",
            intent_body_sha256="1" * 64,
            controller_baseline_body_sha256="2" * 64,
        )
        is None
    )


def test_ordinary_aws_transport_failure_maps_to_fail_closed_exit() -> None:
    module = _module()

    class UnreachableSts:
        def get_caller_identity(self) -> object:
            raise TimeoutError("STS response lost")

    services = module.SubmissionServices(
        sts=UnreachableSts(),
        s3=None,
        ec2=None,
        ssm=None,
        cloudformation=None,
        sky=None,
        clock=lambda: NOW,
        sleep=lambda _seconds: None,
    )
    outcome = module.run_qualification(
        module.QualificationRequest(
            action="validate-only",
            profile=PROFILE,
        ),
        services=services,
    )
    assert outcome.exit_code == 70
    assert outcome.status == "fail-closed"
    assert outcome.detail["reason"] == "STS response lost"


class _DynamicServiceClient:
    def __init__(self, **responses: object):
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __getattr__(self, name: str) -> object:
        if name not in self.responses:
            raise AttributeError(name)

        def call(**kwargs: object) -> object:
            self.calls.append((name, deepcopy(kwargs)))
            return deepcopy(self.responses[name])

        return call


class _DynamicCloudFormationClient:
    def __init__(
        self,
        *,
        stack: dict[str, object],
        template_body: str,
        resources: dict[str, dict[str, object]],
    ):
        self.stack = stack
        self.template_body = template_body
        self.resources = resources
        self.calls: list[tuple[str, dict[str, object]]] = []

    def describe_stacks(self, **kwargs: object) -> object:
        self.calls.append(("describe_stacks", deepcopy(kwargs)))
        return {"Stacks": [deepcopy(self.stack)]}

    def get_template(self, **kwargs: object) -> object:
        self.calls.append(("get_template", deepcopy(kwargs)))
        return {"TemplateBody": self.template_body}

    def describe_stack_resource(self, **kwargs: object) -> object:
        self.calls.append(("describe_stack_resource", deepcopy(kwargs)))
        logical_id = kwargs["LogicalResourceId"]
        return {"StackResourceDetail": deepcopy(self.resources[logical_id])}


def _dynamic_inspector_fixture(module: ModuleType) -> dict[str, Any]:
    stack_name = "keep-glm52-gpu"
    stack_id = (
        f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/{stack_name}/"
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    project_tag = "keep-glm52"
    code_sha256 = "e" * 64
    baseline_sha256 = "2" * 64
    intent_sha256 = "1" * 64
    parameters = {
        "ProjectTag": project_tag,
        "OwnerTag": "jack.mazac",
        "EnableSkyPilotSupport": "true",
        "EnableSkyMustStartObserve": "true",
        "EnableSkyMustStartCancel": "true",
        "SkyCampaignRunId": RUN_ID,
        "SkyMustStartManagedMode": "qualification",
        "SkyMustStartSubmissionBodySha256": intent_sha256,
        "SkyMustStartControllerBaselineBodySha256": baseline_sha256,
        "SkyMustStartFoundationRevision": "versioned-code-v1",
        "SkyMustStartCancelCodeSha256": code_sha256,
        "SkyMustStartCancelCodeVersionId": "version-code",
        "SkyMustStartActivationJobBindingSha256": "a" * 64,
        "SkyMustStartActivationObservationSha256": "b" * 64,
        "SkyMustStartDescriptorRelativeKey": (
            "submissions/qualification/campaign-descriptor-v2.json"
        ),
        "SkyMustStartDescriptorFileSha256": "c" * 64,
        "SkyMustStartTargetJobId": "42",
        "SkyMustStartJobName": f"{RUN_ID}-qualification",
        "SkyMustStartBy": "2026-07-26T18:00:00Z",
        "SkyMustStartPrimaryWakeAt": "2026-07-26T17:59:00Z",
        "SkyMustStartWorkspace": "default",
        "SkyMustStartSubmissionSubmittedAt": "2026-07-26T12:00:00Z",
        "SkyMustStartControllerInstanceId": CONTROLLER_INSTANCE_ID,
        "SkyMustStartControllerInstanceType": CONTROLLER_INSTANCE_TYPE,
        "SkyMustStartControllerProfileArn": CONTROLLER_PROFILE_ARN,
        "SkyMustStartControllerClusterName": CONTROLLER_CLUSTER_NAME,
        "SkyMustStartStartingGraceSeconds": "20",
    }
    function_name = f"{project_tag}-sky-must-start-cancel"
    rule_name = function_name
    schedule_name = f"{project_tag}-sky-must-start-deadline"
    dlq_name = f"{project_tag}-sky-must-start-cancel-dlq"
    bucket_name = BUCKET
    alert_topic_arn = (
        f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:{project_tag}-campaign-alerts"
    )
    function_arn = (
        f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{function_name}"
    )
    rule_arn = f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{rule_name}"
    schedule_arn = (
        f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:"
        f"schedule/default/{schedule_name}"
    )
    dlq_url = f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT_ID}/{dlq_name}"
    dlq_arn = f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:{dlq_name}"
    cancel_role_name = f"{stack_name}-SkyMustStartCancelRole-ABCDEF"
    scheduler_role_name = f"{stack_name}-SkyMustStartSchedulerRole-ABCDEF"
    cancel_role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{cancel_role_name}"
    scheduler_role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{scheduler_role_name}"
    log_group_name = f"/aws/lambda/{function_name}"
    physical = {
        "ModelBucket": bucket_name,
        "CampaignAlertTopic": alert_topic_arn,
        "SkyMustStartCancelDeadLetterQueue": dlq_url,
        "SkyMustStartCancelRole": cancel_role_name,
        "SkyMustStartCancelFunction": function_name,
        "SkyMustStartCancelLogGroup": log_group_name,
        "SkyMustStartCancelDeadLetterQueuePolicy": (
            f"{stack_name}-SkyMustStartCancelDeadLetterQueuePolicy-ABCDEF"
        ),
        "SkyMustStartCancelRule": rule_name,
        "SkyMustStartSchedulerRole": scheduler_role_name,
        "SkyMustStartDeadlineSchedule": schedule_arn,
    }
    resource_details = {
        logical_id: {
            "StackId": stack_id,
            "LogicalResourceId": logical_id,
            "PhysicalResourceId": physical[logical_id],
            "ResourceType": resource_type,
            "ResourceStatus": "CREATE_COMPLETE",
        }
        for logical_id, resource_type in module._BotoCloudFormationInspector._RESOURCE_TYPES.items()
    }
    stack = {
        "StackId": stack_id,
        "StackName": stack_name,
        "StackStatus": "CREATE_COMPLETE",
        "Parameters": [
            {"ParameterKey": key, "ParameterValue": value}
            for key, value in parameters.items()
        ],
        "Outputs": [
            {"OutputKey": "LegacyMustStartReady", "OutputValue": "true"}
        ],
    }
    template_body = CFN_TEMPLATE.read_text()
    parsed = module._BotoCloudFormationInspector._load_template_body(
        template_body,
        stage="test",
    )
    refs = {
        **parameters,
        "AWS::AccountId": ACCOUNT_ID,
        "AWS::Partition": "aws",
        "AWS::Region": REGION,
        "ModelBucket": bucket_name,
        "CampaignAlertTopic": alert_topic_arn,
        "SkyMustStartCancelDeadLetterQueue": dlq_url,
        "SkyMustStartCancelFunction": function_name,
    }
    get_atts = {
        ("ModelBucket", "Arn"): f"arn:aws:s3:::{bucket_name}",
        ("SkyMustStartCancelDeadLetterQueue", "Arn"): dlq_arn,
        ("SkyMustStartCancelFunction", "Arn"): function_arn,
        ("SkyMustStartCancelRole", "Arn"): cancel_role_arn,
        ("SkyMustStartCancelRule", "Arn"): rule_arn,
        ("SkyMustStartSchedulerRole", "Arn"): scheduler_role_arn,
    }
    resources = parsed["Resources"]
    reviewed_policy = module._BotoCloudFormationInspector._resolve_intrinsics(
        resources["SkyMustStartCancelRole"]["Properties"]["Policies"][0][
            "PolicyDocument"
        ],
        refs=refs,
        get_atts=get_atts,
    )
    queue_policy = module._BotoCloudFormationInspector._resolve_intrinsics(
        resources["SkyMustStartCancelDeadLetterQueuePolicy"]["Properties"][
            "PolicyDocument"
        ],
        refs=refs,
        get_atts=get_atts,
    )
    environment = {
        "EXPECTED_ACCOUNT_ID": ACCOUNT_ID,
        "CAMPAIGN_BUCKET": bucket_name,
        "CAMPAIGN_DESCRIPTOR_KEY": (
            f"campaigns/{RUN_ID}/"
            f"{parameters['SkyMustStartDescriptorRelativeKey']}"
        ),
        "IMMUTABLE_SUBMISSION_KEY": (
            f"campaigns/{RUN_ID}/monitor/submission-locks/"
            f"{parameters['SkyMustStartDescriptorFileSha256']}-qualification.json"
        ),
        "SUBMISSION_BODY_SHA256": intent_sha256,
        "EXPECTED_CONTROLLER_BASELINE_BODY_SHA256": baseline_sha256,
        "EXPECTED_TARGET_JOB_ID": "42",
        "OBSERVE_ONLY": "false",
        "EXPECTED_WORKSPACE": "default",
        "EXPECTED_DESCRIPTOR_FILE_SHA256": "c" * 64,
        "EXPECTED_SUBMISSION_SUBMITTED_AT": "2026-07-26T12:00:00Z",
        "EXPECTED_CONTROLLER_INSTANCE_ID": CONTROLLER_INSTANCE_ID,
        "EXPECTED_CONTROLLER_INSTANCE_TYPE": CONTROLLER_INSTANCE_TYPE,
        "EXPECTED_CONTROLLER_PROFILE_ARN": CONTROLLER_PROFILE_ARN,
        "EXPECTED_CONTROLLER_CLUSTER_NAME": CONTROLLER_CLUSTER_NAME,
        "STARTING_GRACE_SECONDS": "20",
        "PRIMARY_WAKE_MAX_WAIT_SECONDS": "120",
        "RECONCILIATION_RULE_NAME": rule_name,
        "EXPECTED_ACTIVATION_JOB_BINDING_SHA256": "a" * 64,
        "EXPECTED_ACTIVATION_OBSERVATION_SHA256": "b" * 64,
        "CAMPAIGN_RUN_ID": RUN_ID,
        "MANAGED_MODE": "qualification",
        "SKY_JOB_NAME": f"{RUN_ID}-qualification",
        "MUST_START_BY": "2026-07-26T18:00:00Z",
        "ALERT_TOPIC_ARN": alert_topic_arn,
    }
    rule_input = json.dumps(
        {
            "campaign_run_id": RUN_ID,
            "managed_mode": "qualification",
            "sky_job_name": f"{RUN_ID}-qualification",
            "must_start_by": "2026-07-26T18:00:00Z",
            "trigger": "reconcile",
        },
        separators=(",", ":"),
    )
    deadline_input = rule_input.replace("reconcile", "primary-deadline")
    cloudformation = _DynamicCloudFormationClient(
        stack=stack,
        template_body=template_body,
        resources=resource_details,
    )
    lambda_client = _DynamicServiceClient(
        get_function={
            "Configuration": {
                "FunctionName": function_name,
                "FunctionArn": function_arn,
                "Runtime": "python3.13",
                "Role": cancel_role_arn,
                "Handler": "handler.lambda_handler",
                "CodeSha256": base64.b64encode(
                    bytes.fromhex(code_sha256)
                ).decode(),
                "Timeout": 300,
                "MemorySize": 256,
                "PackageType": "Zip",
                "State": "Active",
                "LastUpdateStatus": "Successful",
                "DeadLetterConfig": {"TargetArn": dlq_arn},
                "Environment": {"Variables": environment},
            },
            "Code": {"RepositoryType": "S3"},
        },
        get_function_concurrency={"ReservedConcurrentExecutions": 1},
    )
    iam = _DynamicServiceClient(
        get_role_policy={
            "RoleName": cancel_role_name,
            "PolicyName": "sky-must-start-cancel-only",
            "PolicyDocument": reviewed_policy,
        }
    )
    events = _DynamicServiceClient(
        describe_rule={
            "Name": rule_name,
            "Arn": rule_arn,
            "State": "ENABLED",
            "ScheduleExpression": "rate(1 minute)",
            "Description": (
                f"KEEP GLM52 must-start cancellation for {RUN_ID} qualification"
            ),
        },
        list_targets_by_rule={
            "Targets": [
                {
                    "Id": "sky-must-start-cancel",
                    "Arn": function_arn,
                    "Input": rule_input,
                    "RetryPolicy": {
                        "MaximumEventAgeInSeconds": 300,
                        "MaximumRetryAttempts": 2,
                    },
                    "DeadLetterConfig": {"Arn": dlq_arn},
                }
            ]
        },
    )
    scheduler = _DynamicServiceClient(
        get_schedule={
            "Name": schedule_name,
            "GroupName": "default",
            "Arn": schedule_arn,
            "State": "ENABLED",
            "ActionAfterCompletion": "DELETE",
            "ScheduleExpression": "at(2026-07-26T17:59:00)",
            "ScheduleExpressionTimezone": "UTC",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "Target": {
                "Arn": function_arn,
                "RoleArn": scheduler_role_arn,
                "Input": deadline_input,
                "RetryPolicy": {
                    "MaximumEventAgeInSeconds": 300,
                    "MaximumRetryAttempts": 2,
                },
                "DeadLetterConfig": {"Arn": dlq_arn},
            },
            "Description": (
                f"Exact must-start deadline for {RUN_ID} qualification"
            ),
        }
    )
    sqs = _DynamicServiceClient(
        get_queue_attributes={
            "Attributes": {
                "QueueArn": dlq_arn,
                "MessageRetentionPeriod": "1209600",
                "SqsManagedSseEnabled": "true",
                "Policy": json.dumps(queue_policy),
            }
        }
    )
    logs = _DynamicServiceClient(
        describe_log_groups={
            "logGroups": [
                {
                    "logGroupName": log_group_name,
                    "retentionInDays": 14,
                }
            ]
        }
    )
    ssm = _DynamicServiceClient(
        describe_instance_information={
            "InstanceInformationList": [
                {
                    "InstanceId": CONTROLLER_INSTANCE_ID,
                    "PingStatus": "Online",
                }
            ]
        }
    )
    inspector = module._BotoCloudFormationInspector(
        cloudformation,
        lambda_client=lambda_client,
        iam_client=iam,
        events_client=events,
        scheduler_client=scheduler,
        sqs_client=sqs,
        logs_client=logs,
        ssm_client=ssm,
        clock=lambda: NOW,
    )
    return {
        "inspector": inspector,
        "cloudformation": cloudformation,
        "lambda": lambda_client,
        "iam": iam,
        "events": events,
        "scheduler": scheduler,
        "sqs": sqs,
        "logs": logs,
        "ssm": ssm,
        "stack_id": stack_id,
        "resource_details": resource_details,
        "intent_sha256": intent_sha256,
        "baseline_sha256": baseline_sha256,
    }


def _inspect_dynamic(fixture: dict[str, Any], **overrides: object) -> object:
    request = {
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "intent_body_sha256": fixture["intent_sha256"],
        "controller_baseline_body_sha256": fixture["baseline_sha256"],
    }
    request.update(overrides)
    return fixture["inspector"].inspect_dynamic_must_start(**request)


def _set_stack_parameter(
    fixture: dict[str, Any],
    name: str,
    value: str,
) -> None:
    for parameter in fixture["cloudformation"].stack["Parameters"]:
        if parameter["ParameterKey"] == name:
            parameter["ParameterValue"] = value
            return
    raise AssertionError(name)


def test_dynamic_must_start_inspector_returns_exact_live_identity_and_calls() -> None:
    module = _module()
    fixture = _dynamic_inspector_fixture(module)

    result = _inspect_dynamic(fixture)

    assert set(result) == {
        "reviewed_deployment_identity",
        "observed_deployment_identity",
        "reconciliation_rule_state",
        "deadline_schedule_state",
        "coordinator_mode",
        "observed_at",
    }
    assert result["reviewed_deployment_identity"] == result[
        "observed_deployment_identity"
    ]
    assert set(result["reviewed_deployment_identity"]) == {
        "stack_id",
        "template_sha256",
        "lambda_function_arn",
        "lambda_code_sha256",
        "iam_policy_sha256",
        "reconciliation_rule_arn",
        "deadline_schedule_arn",
        "dlq_arn",
    }
    assert result["reconciliation_rule_state"] == "ENABLED"
    assert result["deadline_schedule_state"] == "ENABLED"
    assert result["coordinator_mode"] == "dynamic-job-binding-active"
    assert result["observed_at"] == _iso(NOW)
    stack_id = fixture["stack_id"]
    assert fixture["cloudformation"].calls == [
        ("describe_stacks", {"StackName": "keep-glm52-gpu"}),
        ("get_template", {"StackName": stack_id, "TemplateStage": "Original"}),
        ("get_template", {"StackName": stack_id, "TemplateStage": "Processed"}),
        *[
            (
                "describe_stack_resource",
                {"StackName": stack_id, "LogicalResourceId": logical_id},
            )
            for logical_id in module._BotoCloudFormationInspector._RESOURCE_TYPES
        ],
    ]
    assert fixture["lambda"].calls == [
        ("get_function", {"FunctionName": "keep-glm52-sky-must-start-cancel"}),
        (
            "get_function_concurrency",
            {"FunctionName": "keep-glm52-sky-must-start-cancel"},
        ),
    ]
    assert fixture["iam"].calls == [
        (
            "get_role_policy",
            {
                "RoleName": (
                    "keep-glm52-gpu-SkyMustStartCancelRole-ABCDEF"
                ),
                "PolicyName": "sky-must-start-cancel-only",
            },
        )
    ]
    assert fixture["events"].calls == [
        ("describe_rule", {"Name": "keep-glm52-sky-must-start-cancel"}),
        (
            "list_targets_by_rule",
            {"Rule": "keep-glm52-sky-must-start-cancel", "Limit": 100},
        ),
    ]
    assert fixture["scheduler"].calls == [
        (
            "get_schedule",
            {
                "Name": "keep-glm52-sky-must-start-deadline",
                "GroupName": "default",
            },
        )
    ]
    assert fixture["sqs"].calls == [
        (
            "get_queue_attributes",
            {
                "QueueUrl": (
                    "https://sqs.us-west-2.amazonaws.com/246813579024/"
                    "keep-glm52-sky-must-start-cancel-dlq"
                ),
                "AttributeNames": ["All"],
            },
        )
    ]
    assert fixture["logs"].calls == [
        (
            "describe_log_groups",
            {
                "logGroupNamePrefix": (
                    "/aws/lambda/keep-glm52-sky-must-start-cancel"
                ),
                "limit": 1,
            },
        )
    ]
    assert fixture["ssm"].calls == [
        (
            "describe_instance_information",
            {
                "Filters": [
                    {
                        "Key": "InstanceIds",
                        "Values": [CONTROLLER_INSTANCE_ID],
                    }
                ],
                "MaxResults": 5,
            },
        )
    ]


@pytest.mark.parametrize(
    ("request_field", "parameter_name", "foreign_value"),
    [
        ("run_id", "SkyCampaignRunId", "foreign-run"),
        ("managed_mode", "SkyMustStartManagedMode", "production"),
        ("intent_body_sha256", "SkyMustStartSubmissionBodySha256", "3" * 64),
        (
            "controller_baseline_body_sha256",
            "SkyMustStartControllerBaselineBodySha256",
            "4" * 64,
        ),
    ],
)
def test_dynamic_must_start_inspector_rejects_each_stack_input_mismatch(
    request_field: str,
    parameter_name: str,
    foreign_value: str,
) -> None:
    module = _module()
    fixture = _dynamic_inspector_fixture(module)
    _set_stack_parameter(fixture, parameter_name, foreign_value)

    assert _inspect_dynamic(fixture) is None


@pytest.mark.parametrize(
    "mismatch",
    [
        "stack-status",
        "stack-region",
        "stack-account",
        "resource-status",
        "lambda-runtime",
        "lambda-concurrency",
        "iam-policy",
        "rule-state",
        "rule-target",
        "schedule-auto-delete",
        "queue-policy",
        "log-retention",
        "ssm-instance",
    ],
)
def test_dynamic_must_start_inspector_rejects_live_resource_drift(
    mismatch: str,
) -> None:
    module = _module()
    fixture = _dynamic_inspector_fixture(module)
    if mismatch == "stack-status":
        fixture["cloudformation"].stack["StackStatus"] = "UPDATE_IN_PROGRESS"
    elif mismatch == "stack-region":
        fixture["cloudformation"].stack["StackId"] = fixture["stack_id"].replace(
            ":us-west-2:",
            ":us-east-1:",
        )
    elif mismatch == "stack-account":
        fixture["cloudformation"].stack["StackId"] = fixture["stack_id"].replace(
            f":{ACCOUNT_ID}:",
            ":000000000000:",
        )
    elif mismatch == "resource-status":
        fixture["resource_details"]["SkyMustStartCancelFunction"][
            "ResourceStatus"
        ] = "UPDATE_IN_PROGRESS"
    elif mismatch == "lambda-runtime":
        fixture["lambda"].responses["get_function"]["Configuration"][
            "Runtime"
        ] = "python3.12"
    elif mismatch == "lambda-concurrency":
        fixture["lambda"].responses["get_function_concurrency"][
            "ReservedConcurrentExecutions"
        ] = 2
    elif mismatch == "iam-policy":
        fixture["iam"].responses["get_role_policy"]["PolicyDocument"][
            "Statement"
        ][0]["Effect"] = "Deny"
    elif mismatch == "rule-state":
        fixture["events"].responses["describe_rule"]["State"] = "DISABLED"
    elif mismatch == "rule-target":
        fixture["events"].responses["list_targets_by_rule"]["Targets"][0][
            "Id"
        ] = "foreign"
    elif mismatch == "schedule-auto-delete":
        fixture["scheduler"].responses["get_schedule"][
            "ActionAfterCompletion"
        ] = "NONE"
    elif mismatch == "queue-policy":
        fixture["sqs"].responses["get_queue_attributes"]["Attributes"][
            "Policy"
        ] = '{"Version":"2012-10-17","Statement":[]}'
    elif mismatch == "log-retention":
        fixture["logs"].responses["describe_log_groups"]["logGroups"][0][
            "retentionInDays"
        ] = 30
    elif mismatch == "ssm-instance":
        fixture["ssm"].responses["describe_instance_information"][
            "InstanceInformationList"
        ][0]["InstanceId"] = "i-00000000000000000"
    else:
        raise AssertionError(mismatch)

    assert _inspect_dynamic(fixture) is None


def test_dynamic_must_start_inspector_rejects_unreviewed_coherent_template() -> (
    None
):
    module = _module()
    fixture = _dynamic_inspector_fixture(module)
    fixture["cloudformation"].template_body += (
        "\nMetadata:\n  UnreviewedReplacement: true\n"
    )

    assert _inspect_dynamic(fixture) is None


def test_dynamic_must_start_inspector_reviewed_template_pin_matches_live_template() -> (
    None
):
    """Guard against silent fail-closed drift: any edit to the live CFN
    template that is not accompanied by a matching update to
    ``_REVIEWED_TEMPLATE_CANONICAL_SHA256`` makes
    ``inspect_dynamic_must_start`` return ``None`` for every caller, with no
    direct error pointing at the actual cause. This test fails loudly, by
    name, instead.
    """
    module = _module()
    inspector = module._BotoCloudFormationInspector
    loaded = inspector._load_template_body(
        CFN_TEMPLATE.read_text(),
        stage="test",
    )
    assert inspector._canonical_sha256(loaded) == (
        inspector._REVIEWED_TEMPLATE_CANONICAL_SHA256
    ), (
        "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml changed without refreshing "
        "_BotoCloudFormationInspector._REVIEWED_TEMPLATE_CANONICAL_SHA256 in "
        "aws/glm52-gpu/scripts/submit_sky_campaign.py"
    )


def test_dynamic_must_start_inspector_classifies_absent_sdk_resource() -> None:
    from botocore.exceptions import ClientError

    module = _module()
    fixture = _dynamic_inspector_fixture(module)

    def missing_function(**_kwargs: object) -> object:
        raise ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "function disappeared",
                }
            },
            "GetFunction",
        )

    fixture["lambda"].get_function = missing_function

    assert _inspect_dynamic(fixture) is None


def test_dynamic_must_start_inspector_rejects_reviewed_observed_identity_drift() -> (
    None
):
    module = _module()
    fixture = _dynamic_inspector_fixture(module)
    fixture["lambda"].responses["get_function"]["Configuration"][
        "CodeSha256"
    ] = base64.b64encode(bytes.fromhex("f" * 64)).decode()

    assert _inspect_dynamic(fixture) is None


def test_dynamic_must_start_inspector_rejects_malformed_service_response() -> None:
    module = _module()
    fixture = _dynamic_inspector_fixture(module)
    fixture["lambda"].responses["get_function"] = []

    with pytest.raises(
        module.SubmissionIntegrationError,
        match="Lambda get-function response is malformed",
    ):
        _inspect_dynamic(fixture)


def test_dynamic_must_start_inspector_rejects_unreviewed_role_intrinsic() -> None:
    module = _module()
    fixture = _dynamic_inspector_fixture(module)
    fixture["cloudformation"].template_body = fixture[
        "cloudformation"
    ].template_body.replace(
        "ssm:resourceTag/project: !Ref ProjectTag",
        'ssm:resourceTag/project: !Join ["", [keep, glm52]]',
        1,
    )

    assert _inspect_dynamic(fixture) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_id": ""},
        {"managed_mode": "observe"},
        {"intent_body_sha256": "A" * 64},
        {"controller_baseline_body_sha256": "2" * 63},
    ],
)
def test_dynamic_must_start_inspector_rejects_invalid_requested_inputs_before_aws(
    overrides: dict[str, object],
) -> None:
    module = _module()
    fixture = _dynamic_inspector_fixture(module)

    assert _inspect_dynamic(fixture, **overrides) is None
    assert fixture["cloudformation"].calls == []
