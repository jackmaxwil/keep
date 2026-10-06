"""Strict post-seed qualification submission-readiness authority tests."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pytest

from mlx_vq.quality.glm52_gpu_spend_snapshot import (
    validate_gpu_spend_snapshot,
)
from mlx_vq.quality.glm52_qualification_cache_seed import (
    accepted_s3_key,
    validate_qualification_cache_seed_accepted,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_sky_campaign_descriptor,
    validate_sky_campaign_descriptor,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"
MODULE_NAME = "mlx_vq.quality.glm52_qualification_submission_ready"
RUN_ID = "glm52-sky-20260724"
STAGED_AT = datetime(2026, 7, 26, 12, 2, tzinfo=UTC)
SNAPSHOT_AT = datetime(2026, 7, 26, 12, 3, tzinfo=UTC)
REHEARSED_AT = datetime(2026, 7, 26, 12, 4, tzinfo=UTC)
BUILT_AT = datetime(2026, 7, 26, 12, 5, tzinfo=UTC)
MUST_START_BY = datetime(2026, 7, 26, 18, 0, tzinfo=UTC)

READY_FIELDS = {
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
    "seed_descriptor_key",
    "seed_descriptor_file_sha256",
    "seed_descriptor_body_sha256",
    "seed_campaign_identity_sha256",
    "seed_repo_tar_sha256",
    "seed_descriptor",
    "staged_readiness_key",
    "staged_readiness_file_sha256",
    "staged_readiness_body_sha256",
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
    "rehearsal_evidence_key",
    "rehearsal_evidence_sha256",
    "rehearsal_evidence_body_sha256",
    "sky_task_name",
    "sky_job_name",
    "must_start_by",
    "allowed_gpu_seconds",
    "allowed_gpu_cost_usd",
    "built_at",
    "readiness_body_sha256",
}
STAGED_READY_V2_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "descriptor_key",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "staged_object_version_ids",
    "artifact_audit_key",
    "artifact_audit_sha256",
    "staged_at",
    "ready_body_sha256",
}
STAGED_OBJECT_VERSION_ROLES = {
    "repository_tar",
    "approval",
    "training_config",
    "watchdog",
    "artifact_inventory",
    "artifact_audit",
    "descriptor",
}


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
    body = dict(value)
    body.pop(digest_field)
    return _self_hashed(body, digest_field)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _money_for_seconds(seconds: int, hourly: float) -> float:
    return float(
        (Decimal(seconds) * Decimal(str(hourly)) / Decimal(3600)).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def _module() -> Any:
    return importlib.import_module(MODULE_NAME)


def _load_cache_test_module() -> Any:
    path = ROOT / "tests/test_glm52_qualification_cache_seed.py"
    specification = importlib.util.spec_from_file_location(
        "_task3e_cache_seed_fixture",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seed_authority(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    cache_tests = _load_cache_test_module()
    client, pins, acceptance, _keys = cache_tests._accepted_fixture(
        tmp_path_factory.mktemp("task3e-cache-seed")
    )
    seed_descriptor_raw = client.objects[pins.descriptor_key]
    seed_descriptor = json.loads(seed_descriptor_raw)
    approval_key = str(acceptance["seed_authority"]["approval_key"])
    approval = json.loads(client.objects[approval_key])
    assert validate_sky_campaign_descriptor(seed_descriptor) == seed_descriptor
    assert validate_qualification_cache_seed_accepted(acceptance) == acceptance
    return {
        "seed_descriptor": seed_descriptor,
        "seed_descriptor_raw": seed_descriptor_raw,
        "cache_seed_acceptance": acceptance,
        "approval": approval,
    }


def _staged_readiness(
    descriptor: dict[str, object],
    descriptor_file_sha256: str,
) -> dict[str, object]:
    manifest_body_sha = "b" * 64
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": RUN_ID,
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": (
            f"campaigns/{RUN_ID}/submissions/post-seed/bundle-manifests/"
            f"{manifest_body_sha}/bundle-manifest-v1.json"
        ),
        "bundle_manifest_file_sha256": "d" * 64,
        "bundle_manifest_body_sha256": manifest_body_sha,
        "bundle_manifest_version_id": "manifest-version-1",
        "staged_object_version_ids": {
            role: f"{role}-version-1"
            for role in STAGED_OBJECT_VERSION_ROLES
        },
        "artifact_audit_key": (
            f"campaigns/{RUN_ID}/audits/artifact-audit-{'c' * 64}.json"
        ),
        "artifact_audit_sha256": "c" * 64,
        "staged_at": _iso(STAGED_AT),
    }
    return _self_hashed(body, "ready_body_sha256")


def _gpu_spend_snapshot(
    descriptor: dict[str, object],
    descriptor_file_sha256: str,
    acceptance: dict[str, object],
    approval: dict[str, object],
) -> dict[str, object]:
    spend = acceptance["spend_closure"]
    consumed_seconds = int(spend["consumed_gpu_seconds"])
    remaining_seconds = 86_400 - consumed_seconds
    hourly_cost = float(descriptor["max_hourly_cost_usd"])
    consumed_cost = _money_for_seconds(consumed_seconds, hourly_cost)
    remaining_cost = float(
        (
            Decimal(str(descriptor["approved_gpu_cost_usd"]))
            - Decimal(str(consumed_cost))
        ).quantize(Decimal("0.01"))
    )
    allowance_seconds = min(14_400, remaining_seconds)
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": approval["approval_body_sha256"],
        "gpu_spend_ledger_latest_sha256": "d" * 64,
        "gpu_spend_ledger_latest_body_sha256": "e" * 64,
        "gpu_spend_ledger_genesis_sha256": "f" * 64,
        "gpu_spend_ledger_record_count": 2,
        "gpu_spend_ledger_tip_record_sha256": (spend["ledger_tip_record_sha256"]),
        "gpu_spend_ledger_file_sha256": spend["ledger_file_sha256"],
        "ec2_allocation_history_sha256": "a" * 64,
        "ec2_allocation_instance_ids": [spend["instance_id"]],
        "observed_at": _iso(SNAPSHOT_AT),
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "hourly_cost_usd": hourly_cost,
        "consumed_gpu_seconds": consumed_seconds,
        "remaining_gpu_seconds": remaining_seconds,
        "consumed_gpu_cost_usd": consumed_cost,
        "remaining_gpu_cost_usd": remaining_cost,
        "qualification_allowance_seconds": allowance_seconds,
        "qualification_allowance_cost_usd": _money_for_seconds(
            allowance_seconds,
            hourly_cost,
        ),
        "open_allocation_count": 0,
    }
    snapshot = _self_hashed(body, "snapshot_body_sha256")
    assert validate_gpu_spend_snapshot(snapshot) == snapshot
    return snapshot


def _rehearsal_evidence(
    descriptor: dict[str, object],
    descriptor_file_sha256: str,
    staged_readiness: dict[str, object],
    staged_readiness_key: str,
    staged_readiness_file_sha256: str,
) -> dict[str, object]:
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
        "extracted_repo_path": "/tmp/glm52-task3e-rehearsal/repo",
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
        "completed_at": _iso(REHEARSED_AT),
    }
    return _self_hashed(body, "rehearsal_body_sha256")


def _authority_inputs(
    seed: dict[str, object],
    *,
    post_artifact_overrides: dict[str, str] | None = None,
    post_repo_tar_key: str | None = None,
    post_repo_tar_sha256: str | None = None,
    must_start_by: datetime = MUST_START_BY,
) -> dict[str, object]:
    seed_descriptor = copy.deepcopy(seed["seed_descriptor"])
    acceptance = copy.deepcopy(seed["cache_seed_acceptance"])
    approval = copy.deepcopy(seed["approval"])
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
    if post_artifact_overrides:
        artifacts.update(post_artifact_overrides)
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=must_start_by,
        controller_identity=str(seed_descriptor["controller_identity"]),
        worker_identity=str(seed_descriptor["worker_identity"]),
        vpc_name=str(seed_descriptor["vpc_name"]),
        image_id=str(seed_descriptor["image_id"]),
        bucket=str(seed_descriptor["bucket"]),
        jobs_bucket=str(seed_descriptor["jobs_bucket"]),
        repo_tar_key=post_repo_tar_key or str(seed_descriptor["repo_tar_key"]),
        repo_tar_sha256=(
            post_repo_tar_sha256 or str(seed_descriptor["repo_tar_sha256"])
        ),
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/post-seed/campaign-descriptor-v2.json"
        ),
        approval_key=str(seed_descriptor["approval_key"]),
        approval_sha256=str(seed_descriptor["approval_sha256"]),
        artifacts=artifacts,
    )
    descriptor_raw = _json_bytes(descriptor)
    descriptor_file_sha256 = _sha(descriptor_raw)
    staged = _staged_readiness(descriptor, descriptor_file_sha256)
    staged_key = (
        str(descriptor["campaign_descriptor_key"]).removesuffix(
            "campaign-descriptor-v2.json"
        )
        + "STAGED_CONTROL_PLANE_READY.json"
    )
    staged_file_sha256 = _sha(_json_bytes(staged))
    snapshot = _gpu_spend_snapshot(
        descriptor,
        descriptor_file_sha256,
        acceptance,
        approval,
    )
    rehearsal = _rehearsal_evidence(
        descriptor,
        descriptor_file_sha256,
        staged,
        staged_key,
        staged_file_sha256,
    )
    cache_body_sha = str(acceptance["acceptance_body_sha256"])
    snapshot_body_sha = str(snapshot["snapshot_body_sha256"])
    rehearsal_body_sha = str(rehearsal["rehearsal_body_sha256"])
    return {
        "descriptor": descriptor,
        "descriptor_raw": descriptor_raw,
        "descriptor_file_sha256": descriptor_file_sha256,
        "seed_descriptor_raw": seed["seed_descriptor_raw"],
        "seed_descriptor_file_sha256": _sha(seed["seed_descriptor_raw"]),
        "staged_readiness": staged,
        "staged_readiness_key": staged_key,
        "staged_readiness_file_sha256": staged_file_sha256,
        "cache_seed_acceptance": acceptance,
        "cache_seed_acceptance_key": accepted_s3_key(
            run_id=RUN_ID,
            acceptance_body_sha256=cache_body_sha,
        ),
        "cache_seed_acceptance_file_sha256": _sha(_json_bytes(acceptance)),
        "gpu_spend_snapshot": snapshot,
        "gpu_spend_snapshot_key": (
            f"campaigns/{RUN_ID}/spend-snapshots/{snapshot_body_sha}/"
            "GPU_SPEND_SNAPSHOT.json"
        ),
        "gpu_spend_snapshot_sha256": _sha(_json_bytes(snapshot)),
        "rehearsal_evidence": rehearsal,
        "rehearsal_evidence_key": (
            f"campaigns/{RUN_ID}/qualification/rehearsals/"
            f"{rehearsal_body_sha}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "rehearsal_evidence_sha256": _sha(_json_bytes(rehearsal)),
        "built_at": BUILT_AT,
    }


def _rebind_ready_to_post_descriptor(
    ready: dict[str, object],
    authorities: dict[str, object],
) -> dict[str, object]:
    descriptor = authorities["descriptor"]
    staged = authorities["staged_readiness"]
    snapshot = authorities["gpu_spend_snapshot"]
    rehearsal = authorities["rehearsal_evidence"]
    assert isinstance(descriptor, dict)
    assert isinstance(staged, dict)
    assert isinstance(snapshot, dict)
    assert isinstance(rehearsal, dict)
    ready.update(
        {
            "descriptor_file_sha256": authorities["descriptor_file_sha256"],
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "repo_tar_sha256": descriptor["repo_tar_sha256"],
            "staged_readiness_file_sha256": authorities["staged_readiness_file_sha256"],
            "staged_readiness_body_sha256": staged["ready_body_sha256"],
            "gpu_spend_snapshot_key": authorities["gpu_spend_snapshot_key"],
            "gpu_spend_snapshot_sha256": authorities["gpu_spend_snapshot_sha256"],
            "gpu_spend_snapshot_body_sha256": snapshot["snapshot_body_sha256"],
            "rehearsal_evidence_key": authorities["rehearsal_evidence_key"],
            "rehearsal_evidence_sha256": authorities["rehearsal_evidence_sha256"],
            "rehearsal_evidence_body_sha256": rehearsal["rehearsal_body_sha256"],
        }
    )
    return _rehash(ready, "readiness_body_sha256")


def _builder_kwargs(authorities: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in authorities.items()
        if key
        not in {
            "descriptor_raw",
        }
    }


def _validator_kwargs(
    authorities: dict[str, object],
    *,
    now: datetime | str | None = BUILT_AT,
) -> dict[str, object]:
    names = {
        "descriptor",
        "staged_readiness",
        "cache_seed_acceptance",
        "gpu_spend_snapshot",
        "rehearsal_evidence",
        "descriptor_file_sha256",
        "staged_readiness_key",
        "staged_readiness_file_sha256",
        "cache_seed_acceptance_key",
        "cache_seed_acceptance_file_sha256",
        "gpu_spend_snapshot_key",
        "gpu_spend_snapshot_sha256",
        "rehearsal_evidence_key",
        "rehearsal_evidence_sha256",
    }
    return {
        **{name: authorities[name] for name in names},
        "now": now,
    }


def test_builder_rejects_historical_staged_readiness_v1(
    seed_authority: dict[str, object],
) -> None:
    module = _module()
    valid_inputs = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(
        **_builder_kwargs(valid_inputs)
    )
    inputs = copy.deepcopy(valid_inputs)
    staged = copy.deepcopy(inputs["staged_readiness"])
    for field in (
        "bundle_manifest_key",
        "bundle_manifest_file_sha256",
        "bundle_manifest_version_id",
        "staged_object_version_ids",
    ):
        staged.pop(field)
    staged["schema_version"] = 1
    staged["record_type"] = "glm52_staged_control_plane_ready_v1"
    inputs["staged_readiness"] = _rehash(staged, "ready_body_sha256")

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="staged readiness schema",
    ):
        module.build_qualification_submission_ready(**_builder_kwargs(inputs))
    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="staged readiness schema",
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(inputs),
        )


@pytest.mark.parametrize(
    ("mutation", "invalid_value"),
    (
        ("missing_field", "unused"),
        ("extra_field", "unused"),
        ("mixed_v1", "unused"),
        ("unsafe_manifest_key", "unused"),
        ("bad_manifest_file_sha", "unused"),
        ("bad_manifest_body_sha", "unused"),
        ("missing_manifest_version", "unused"),
        ("empty_manifest_version", ""),
        ("null_manifest_version", None),
        ("literal_null_manifest_version", "null"),
        ("missing_version_role", "unused"),
        ("extra_version_role", "unused"),
        ("empty_role_version", ""),
        ("null_role_version", None),
        ("literal_null_role_version", "null"),
        ("non_string_role_version", 7),
        ("fractional_staged_at", "unused"),
    ),
)
def test_builder_and_validator_reject_invalid_staged_readiness_v2_contract(
    seed_authority: dict[str, object],
    mutation: str,
    invalid_value: object,
) -> None:
    module = _module()
    valid_inputs = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(
        **_builder_kwargs(valid_inputs)
    )
    inputs = copy.deepcopy(valid_inputs)
    staged = inputs["staged_readiness"]
    assert isinstance(staged, dict)
    if mutation == "missing_field":
        staged.pop("bundle_manifest_key")
    elif mutation == "extra_field":
        staged["unexpected"] = "forbidden"
    elif mutation == "mixed_v1":
        staged["schema_version"] = 1
        staged["record_type"] = "glm52_staged_control_plane_ready_v1"
    elif mutation == "unsafe_manifest_key":
        staged["bundle_manifest_key"] = "../bundle-manifest-v1.json"
    elif mutation == "bad_manifest_file_sha":
        staged["bundle_manifest_file_sha256"] = "D" * 64
    elif mutation == "bad_manifest_body_sha":
        staged["bundle_manifest_body_sha256"] = "B" * 64
    elif mutation == "missing_manifest_version":
        staged.pop("bundle_manifest_version_id")
    elif mutation.endswith("manifest_version"):
        staged["bundle_manifest_version_id"] = invalid_value
    elif mutation == "fractional_staged_at":
        staged["staged_at"] = "2026-07-26T12:00:00.123456Z"
    else:
        versions = dict(staged["staged_object_version_ids"])
        if mutation == "missing_version_role":
            versions.pop("repository_tar")
        elif mutation == "extra_version_role":
            versions["unexpected"] = "foreign-version"
        else:
            versions["repository_tar"] = invalid_value
        staged["staged_object_version_ids"] = versions
    inputs["staged_readiness"] = _rehash(staged, "ready_body_sha256")

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="staged readiness",
    ):
        module.build_qualification_submission_ready(**_builder_kwargs(inputs))
    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="staged readiness",
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(inputs),
        )


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
def test_nested_consumer_safe_key_rejects_non_printable_or_backslash(
    unsafe: str,
) -> None:
    module = _module()

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="safe exact key",
    ):
        module._safe_key(unsafe, field="staged readiness watchdog key")


def test_builds_exact_post_seed_authority_and_public_key_contracts(
    seed_authority: dict[str, object],
) -> None:
    module = _module()
    inputs = _authority_inputs(seed_authority)

    ready = module.build_qualification_submission_ready(**_builder_kwargs(inputs))

    assert set(inputs["staged_readiness"]) == STAGED_READY_V2_FIELDS
    assert set(
        inputs["staged_readiness"]["staged_object_version_ids"]
    ) == STAGED_OBJECT_VERSION_ROLES
    assert set(ready) == READY_FIELDS
    assert ready["schema_version"] == 1
    assert ready["record_type"] == "glm52_qualification_submission_ready_v1"
    assert ready["seed_descriptor"] == seed_authority["seed_descriptor"]
    assert (
        inputs["descriptor"]["repo_tar_key"]
        == seed_authority["seed_descriptor"]["repo_tar_key"]
    )
    assert ready["repo_tar_sha256"] == ready["seed_repo_tar_sha256"]
    assert ready["allowed_gpu_seconds"] == 14_400
    assert ready["allowed_gpu_cost_usd"] == 220.16
    assert (
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(inputs),
        )
        == ready
    )
    assert (
        module.gpu_spend_snapshot_s3_key(
            run_id=RUN_ID,
            snapshot_body_sha256=ready["gpu_spend_snapshot_body_sha256"],
        )
        == inputs["gpu_spend_snapshot_key"]
    )
    assert (
        module.rehearsal_evidence_s3_key(
            run_id=RUN_ID,
            rehearsal_body_sha256=ready["rehearsal_evidence_body_sha256"],
        )
        == inputs["rehearsal_evidence_key"]
    )
    assert module.qualification_submission_ready_s3_key(ready) == (
        f"campaigns/{RUN_ID}/qualification/submission-ready/"
        f"{ready['readiness_body_sha256']}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )


@pytest.mark.parametrize(
    ("run_id", "digest"),
    (
        ("../foreign", "a" * 64),
        (RUN_ID, "../" + "a" * 64),
        (RUN_ID, "A" * 64),
        (RUN_ID, "a" * 63),
    ),
)
def test_public_key_helpers_reject_alternate_or_unsafe_shapes(
    run_id: str,
    digest: str,
) -> None:
    module = _module()

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="run_id|SHA-256",
    ):
        module.gpu_spend_snapshot_s3_key(
            run_id=run_id,
            snapshot_body_sha256=digest,
        )


def test_builder_rejects_seed_to_post_science_artifact_drift(
    seed_authority: dict[str, object],
) -> None:
    module = _module()
    inputs = _authority_inputs(
        seed_authority,
        post_artifact_overrides={"source_snapshot_sha256": "f" * 64},
    )

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="science artifact",
    ):
        module.build_qualification_submission_ready(**_builder_kwargs(inputs))


@pytest.mark.parametrize(
    ("field", "override", "message"),
    (
        (
            "repo_tar_key",
            f"campaigns/{RUN_ID}/repository/foreign-repo.tar.gz",
            "seed-to-post descriptor authority drift: repo_tar_key",
        ),
        (
            "repo_tar_sha256",
            "f" * 64,
            "seed-to-post descriptor authority drift: repo_tar_sha256",
        ),
    ),
)
def test_builder_rejects_seed_to_post_repository_drift(
    seed_authority: dict[str, object],
    field: str,
    override: str,
    message: str,
) -> None:
    module = _module()
    inputs = _authority_inputs(
        seed_authority,
        **{f"post_{field}": override},
    )

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match=message,
    ):
        module.build_qualification_submission_ready(**_builder_kwargs(inputs))


@pytest.mark.parametrize(
    ("field", "override", "message"),
    (
        (
            "repo_tar_key",
            f"campaigns/{RUN_ID}/repository/foreign-repo.tar.gz",
            "seed-to-post descriptor authority drift: repo_tar_key",
        ),
        (
            "repo_tar_sha256",
            "f" * 64,
            "seed-to-post descriptor authority drift: repo_tar_sha256",
        ),
    ),
)
def test_validator_rejects_rehashed_seed_to_post_repository_drift(
    seed_authority: dict[str, object],
    field: str,
    override: str,
    message: str,
) -> None:
    module = _module()
    original = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(**_builder_kwargs(original))
    drifted = _authority_inputs(
        seed_authority,
        **{f"post_{field}": override},
    )
    ready = _rebind_ready_to_post_descriptor(ready, drifted)

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match=message,
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(drifted),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("unknown", "schema"),
        ("bool_allowance", "allowed_gpu_seconds"),
        ("foreign_job", "sky_job_name"),
        ("early_build", "timestamp"),
        ("nan_cost", "finite"),
    ),
)
def test_validator_rejects_rehashed_semantic_lies(
    seed_authority: dict[str, object],
    mutation: str,
    message: str,
) -> None:
    module = _module()
    inputs = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(**_builder_kwargs(inputs))
    if mutation == "unknown":
        ready["unknown"] = True
    elif mutation == "bool_allowance":
        ready["allowed_gpu_seconds"] = True
        ready = _rehash(ready, "readiness_body_sha256")
    elif mutation == "foreign_job":
        ready["sky_job_name"] = f"{RUN_ID}-production"
        ready = _rehash(ready, "readiness_body_sha256")
    elif mutation == "early_build":
        ready["built_at"] = _iso(REHEARSED_AT - timedelta(seconds=1))
        ready = _rehash(ready, "readiness_body_sha256")
    else:
        ready["allowed_gpu_cost_usd"] = float("nan")

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match=message,
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(inputs),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("cache_key", "cache-seed acceptance key"),
        ("snapshot_tip", "ledger tip"),
        ("rehearsal_descriptor", "rehearsal.*descriptor"),
        ("staged_run", "staged readiness.*run"),
        ("staged_audit_key", "artifact audit key"),
        ("descriptor_file_pin", "descriptor file"),
    ),
)
def test_validator_rejects_foreign_or_drifted_upstream_authority(
    seed_authority: dict[str, object],
    mutation: str,
    message: str,
) -> None:
    module = _module()
    inputs = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(**_builder_kwargs(inputs))
    if mutation == "cache_key":
        inputs["cache_seed_acceptance_key"] = (
            f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
            f"{'f' * 64}/QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        )
    elif mutation == "snapshot_tip":
        snapshot = copy.deepcopy(inputs["gpu_spend_snapshot"])
        snapshot["gpu_spend_ledger_tip_record_sha256"] = "f" * 64
        snapshot = _rehash(snapshot, "snapshot_body_sha256")
        inputs["gpu_spend_snapshot"] = snapshot
        inputs["gpu_spend_snapshot_sha256"] = _sha(_json_bytes(snapshot))
        inputs["gpu_spend_snapshot_key"] = (
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        )
    elif mutation == "rehearsal_descriptor":
        rehearsal = copy.deepcopy(inputs["rehearsal_evidence"])
        rehearsal["descriptor_file_sha256"] = "f" * 64
        rehearsal = _rehash(rehearsal, "rehearsal_body_sha256")
        inputs["rehearsal_evidence"] = rehearsal
        inputs["rehearsal_evidence_sha256"] = _sha(_json_bytes(rehearsal))
        inputs["rehearsal_evidence_key"] = (
            f"campaigns/{RUN_ID}/qualification/rehearsals/"
            f"{rehearsal['rehearsal_body_sha256']}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        )
    elif mutation == "staged_run":
        staged = copy.deepcopy(inputs["staged_readiness"])
        staged["run_id"] = "foreign-run"
        staged = _rehash(staged, "ready_body_sha256")
        inputs["staged_readiness"] = staged
        inputs["staged_readiness_file_sha256"] = _sha(_json_bytes(staged))
    elif mutation == "staged_audit_key":
        staged = copy.deepcopy(inputs["staged_readiness"])
        staged["artifact_audit_key"] = (
            f"campaigns/{RUN_ID}/audits/./"
            f"artifact-audit-{staged['artifact_audit_sha256']}.json"
        )
        staged = _rehash(staged, "ready_body_sha256")
        inputs["staged_readiness"] = staged
        inputs["staged_readiness_file_sha256"] = _sha(_json_bytes(staged))
    else:
        inputs["descriptor_file_sha256"] = "f" * 64

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match=message,
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(inputs),
        )


def test_builder_and_validator_reject_expired_or_overlong_windows(
    seed_authority: dict[str, object],
) -> None:
    module = _module()
    inputs = _authority_inputs(
        seed_authority,
        must_start_by=BUILT_AT + timedelta(hours=12, seconds=1),
    )
    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="12 hours",
    ):
        module.build_qualification_submission_ready(**_builder_kwargs(inputs))

    inputs = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(**_builder_kwargs(inputs))
    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="expired",
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(inputs, now=MUST_START_BY),
        )


def test_builder_and_validator_reject_authority_older_than_five_minutes(
    seed_authority: dict[str, object],
) -> None:
    module = _module()
    inputs = _authority_inputs(seed_authority)
    inputs["built_at"] = SNAPSHOT_AT + timedelta(minutes=5)
    module.build_qualification_submission_ready(**_builder_kwargs(inputs))

    inputs = _authority_inputs(seed_authority)
    inputs["built_at"] = SNAPSHOT_AT + timedelta(minutes=5, seconds=1)

    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="5 minutes",
    ):
        module.build_qualification_submission_ready(**_builder_kwargs(inputs))

    inputs = _authority_inputs(seed_authority)
    ready = module.build_qualification_submission_ready(**_builder_kwargs(inputs))
    assert (
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(
                inputs,
                now=BUILT_AT + timedelta(minutes=5),
            ),
        )
        == ready
    )
    with pytest.raises(
        module.QualificationSubmissionReadyError,
        match="5 minutes",
    ):
        module.validate_qualification_submission_ready(
            ready,
            **_validator_kwargs(
                inputs,
                now=BUILT_AT + timedelta(minutes=5, seconds=1),
            ),
        )


def _write_authority_files(
    tmp_path: Path,
    authorities: dict[str, object],
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in (
        "descriptor",
        "staged_readiness",
        "cache_seed_acceptance",
        "gpu_spend_snapshot",
        "rehearsal_evidence",
    ):
        path = tmp_path / f"{name}.json"
        raw = (
            authorities["descriptor_raw"]
            if name == "descriptor"
            else _json_bytes(authorities[name])
        )
        assert isinstance(raw, bytes)
        path.write_bytes(raw)
        paths[name] = path
    seed_path = tmp_path / "seed_descriptor.json"
    seed_path.write_bytes(authorities["seed_descriptor_raw"])
    paths["seed_descriptor"] = seed_path
    return paths


def _run_cli(
    tmp_path: Path,
    authorities: dict[str, object],
) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, Path]]:
    paths = _write_authority_files(tmp_path, authorities)
    output = tmp_path / "output/QUALIFICATION_SUBMISSION_READY.json"
    command = [
        sys.executable,
        str(SCRIPTS / "build_qualification_submission_ready.py"),
        "--descriptor",
        str(paths["descriptor"]),
        "--descriptor-file-sha256",
        str(authorities["descriptor_file_sha256"]),
        "--seed-descriptor",
        str(paths["seed_descriptor"]),
        "--seed-descriptor-file-sha256",
        str(authorities["seed_descriptor_file_sha256"]),
        "--staged-readiness",
        str(paths["staged_readiness"]),
        "--staged-readiness-key",
        str(authorities["staged_readiness_key"]),
        "--staged-readiness-file-sha256",
        str(authorities["staged_readiness_file_sha256"]),
        "--cache-seed-acceptance",
        str(paths["cache_seed_acceptance"]),
        "--cache-seed-acceptance-key",
        str(authorities["cache_seed_acceptance_key"]),
        "--cache-seed-acceptance-file-sha256",
        str(authorities["cache_seed_acceptance_file_sha256"]),
        "--gpu-spend-snapshot",
        str(paths["gpu_spend_snapshot"]),
        "--gpu-spend-snapshot-key",
        str(authorities["gpu_spend_snapshot_key"]),
        "--gpu-spend-snapshot-sha256",
        str(authorities["gpu_spend_snapshot_sha256"]),
        "--rehearsal-evidence",
        str(paths["rehearsal_evidence"]),
        "--rehearsal-evidence-key",
        str(authorities["rehearsal_evidence_key"]),
        "--rehearsal-evidence-sha256",
        str(authorities["rehearsal_evidence_sha256"]),
        "--built-at",
        _iso(BUILT_AT),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output, paths


def test_cli_writes_canonical_atomic_record_and_exact_receipt(
    seed_authority: dict[str, object],
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _authority_inputs(seed_authority)

    result, output, _paths = _run_cli(tmp_path, inputs)

    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_bytes())
    assert output.read_bytes() == _json_bytes(value)
    assert (
        module.validate_qualification_submission_ready(
            value,
            **_validator_kwargs(inputs),
        )
        == value
    )
    receipt = json.loads(result.stdout)
    assert receipt == {
        "output": str(output.resolve()),
        "qualification_submission_ready_key": (
            module.qualification_submission_ready_s3_key(value)
        ),
        "qualification_submission_ready_sha256": _sha(output.read_bytes()),
        "readiness_body_sha256": value["readiness_body_sha256"],
    }
    assert not list(output.parent.glob(f".{output.name}.*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong_sha", "descriptor.*SHA-256"),
        ("noncanonical_descriptor", "descriptor.*canonical"),
        ("noncanonical_seed", "seed descriptor.*canonical"),
    ),
)
def test_cli_rejects_wrong_or_noncanonical_exact_bytes_without_output(
    seed_authority: dict[str, object],
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    inputs = _authority_inputs(seed_authority)
    if mutation == "wrong_sha":
        inputs["descriptor_file_sha256"] = "f" * 64
    elif mutation == "noncanonical_descriptor":
        descriptor = json.loads(inputs["descriptor_raw"])
        noncanonical = json.dumps(descriptor, indent=2).encode() + b"\n"
        inputs["descriptor_raw"] = noncanonical
        inputs["descriptor_file_sha256"] = _sha(noncanonical)
    else:
        seed = json.loads(inputs["seed_descriptor_raw"])
        noncanonical = json.dumps(seed, indent=2).encode() + b"\n"
        inputs["seed_descriptor_raw"] = noncanonical
        inputs["seed_descriptor_file_sha256"] = _sha(noncanonical)
    result, output, _paths = _run_cli(tmp_path, inputs)

    assert result.returncode != 0
    assert re.search(message, result.stderr, re.IGNORECASE)
    assert not output.exists()


def test_rehearsal_wrapper_delegates_v2_validation_and_rejects_v1(
    seed_authority: dict[str, object],
    tmp_path: Path,
) -> None:
    inputs = _authority_inputs(seed_authority)
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(_json_bytes(inputs["rehearsal_evidence"]))

    valid = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "rehearse_sky_control_plane.sh"),
            "--validate-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stderr
    assert inputs["rehearsal_evidence"]["rehearsal_body_sha256"] in valid.stdout

    legacy = copy.deepcopy(inputs["rehearsal_evidence"])
    legacy["schema_version"] = 1
    legacy["record_type"] = "glm52_sky_control_plane_rehearsal_v1"
    legacy = _rehash(legacy, "rehearsal_body_sha256")
    evidence.write_bytes(_json_bytes(legacy))
    rejected = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "rehearse_sky_control_plane.sh"),
            "--validate-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "schema_version|record_type" not in rejected.stdout
    assert "rehearsal evidence" in rejected.stderr


def test_rehearsal_wrapper_rejects_default_profile_before_aws(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws-called"
    aws = fake_bin / "aws"
    aws.write_text(f"#!/usr/bin/env bash\ntouch {aws_log!s}\nexit 97\n")
    aws.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "rehearse_sky_control_plane.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "AWS_PROFILE": "default",
            "DESCRIPTOR_URI": (
                f"s3://fixture/campaigns/{RUN_ID}/submissions/post-seed/"
                "campaign-descriptor-v2.json"
            ),
            "EXPECTED_DESCRIPTOR_SHA256": "a" * 64,
            "REPO_TAR_URI": (f"s3://fixture/campaigns/{RUN_ID}/repository/repo.tar.gz"),
            "EXPECTED_REPO_TAR_SHA256": "b" * 64,
            "REHEARSAL_EVIDENCE_OUTPUT": str(tmp_path / "evidence.json"),
        },
    )

    assert result.returncode != 0
    assert "AWS_PROFILE must be exactly keep-gpu" in result.stderr
    assert not aws_log.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        (
            {
                "DESCRIPTOR_URI": (
                    f"s3://fixture/campaigns/{RUN_ID}/submissions/../post-seed/"
                    "campaign-descriptor-v2.json"
                )
            },
            "traversal",
        ),
        (
            {"EXPECTED_DESCRIPTOR_SHA256": "A" * 64},
            "lowercase SHA-256",
        ),
    ),
)
def test_rehearsal_wrapper_rejects_unsafe_pins_before_aws(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_log = tmp_path / "aws-called"
    aws = fake_bin / "aws"
    aws.write_text(f"#!/usr/bin/env bash\ntouch {aws_log!s}\nexit 97\n")
    aws.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_PROFILE": "keep-gpu",
        "DESCRIPTOR_URI": (
            f"s3://fixture/campaigns/{RUN_ID}/submissions/post-seed/"
            "campaign-descriptor-v2.json"
        ),
        "EXPECTED_DESCRIPTOR_SHA256": "a" * 64,
        "REPO_TAR_URI": (f"s3://fixture/campaigns/{RUN_ID}/repository/repo.tar.gz"),
        "EXPECTED_REPO_TAR_SHA256": "b" * 64,
        "REHEARSAL_EVIDENCE_OUTPUT": str(
            tmp_path / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        **override,
    }

    result = subprocess.run(
        ["bash", str(SCRIPTS / "rehearse_sky_control_plane.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not aws_log.exists()
