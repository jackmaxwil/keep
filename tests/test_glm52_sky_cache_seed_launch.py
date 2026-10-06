"""Pure cache-seed submission intent and one-winner launch-claim tests."""

# ruff: noqa: UP017

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

from mlx_vq.quality.glm52_gpu_launch_allowance import build_gpu_launch_allowance
from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)

UTC = timezone.utc
RUN_ID = "glm52-sky-20260726"
BUCKET = "keep-glm52-us-west-2-246813579024"
OBSERVED_AT = "2026-07-26T17:45:00Z"
STAGED_VERSION_ID = "version-staged-ready"
EXPECTED_INTENT_BODY_SHA256 = (
    "451273e8a495c9fe22e83fdcea95eda411801881fe1b43c396841a7dd8b8473c"
)
EXPECTED_INTENT_FILE_SHA256 = (
    "c8399e9a7cc00d791fe5ead84362a859294bacf0bf1c0ff41028989dbc18fdcf"
)
EXPECTED_CLAIM_BODY_SHA256 = (
    "d63b3cd65303bed3db1bc979040c3fac8c8bd8ab6f6363afc4741194faf564a3"
)
EXPECTED_CLAIM_FILE_SHA256 = (
    "29be41c8071cafa9d0c0cca89745f6ddf07d7b23e573b72400f5aa2e7d23a8f8"
)


def _module() -> ModuleType:
    return importlib.import_module("mlx_vq.quality.glm52_sky_cache_seed_launch")


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


def _self_hashed(
    value: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    return {**value, digest_field: _sha(_canonical(value))}


def _rehash(
    value: dict[str, object],
    digest_field: str,
) -> dict[str, object]:
    body = dict(value)
    body.pop(digest_field, None)
    return _self_hashed(body, digest_field)


def _s3_observation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_s3_spend_ledger_absence_observation_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "bucket": BUCKET,
        "run_id": RUN_ID,
        "latest_marker_key": (
            f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json"
        ),
        "latest_marker_reads": [
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "before-records-list",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
        ],
        "legacy_ledger_key": f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl",
        "legacy_ledger_reads": [
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "before-records-list",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
        ],
        "spend_records_prefix": (f"campaigns/{RUN_ID}/runtime/spend-ledger/records/"),
        "spend_records_list_pages": [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        "observed_at": OBSERVED_AT,
    }


def _ec2_filters() -> list[dict[str, object]]:
    return [
        {"name": "instance-type", "values": ["p5.48xlarge"]},
        {"name": "tag:campaign-run-id", "values": [RUN_ID]},
        {
            "name": "tag:cost-allocation",
            "values": ["glm52-sky-campaign"],
        },
        {"name": "tag:model", "values": ["glm-5.2"]},
        {"name": "tag:owner", "values": ["jack.mazac"]},
        {"name": "tag:project", "values": ["keep-glm52"]},
    ]


def _ec2_observation() -> dict[str, object]:
    filters = _ec2_filters()
    return {
        "schema_version": 1,
        "record_type": "glm52_ec2_tagged_p5_zero_inventory_observation_v1",
        "account_id": "246813579024",
        "caller_arn": (
            "arn:aws:sts::246813579024:"
            "assumed-role/AWSReservedSSO_AdministratorAccess_abcd/jack.mazac"
        ),
        "region": "us-west-2",
        "run_id": RUN_ID,
        "history_request": {"filters": filters},
        "history_pages": [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": [],
                "next_token": None,
            }
        ],
        "active_request": {
            "filters": [
                *filters,
                {
                    "name": "instance-state-name",
                    "values": [
                        "pending",
                        "running",
                        "shutting-down",
                        "stopping",
                    ],
                },
            ]
        },
        "active_pages": [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": [],
                "next_token": None,
            }
        ],
        "observed_at": OBSERVED_AT,
    }


def _authorities() -> dict[str, object]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
        slack_permalink=None,
    )
    approval_raw = _file_bytes(approval)
    repository_raw = b"exact cache-seed repository tar fixture\n"
    training_raw = b'{"training":"authority"}\n'
    watchdog_raw = b"#!/bin/sh\nexit 0\n"
    inventory_raw = b'{"inventory":"authority"}\n'
    audit_raw = b'{"audit":"authority"}\n'
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=datetime(2026, 7, 26, 18, 45, tzinfo=UTC),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=f"campaigns/{RUN_ID}/repository/keep-cache-seed.tar.gz",
        repo_tar_sha256=_sha(repository_raw),
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
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": "5" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
            "training_config_key": f"campaigns/{RUN_ID}/authorities/training.json",
            "training_config_sha256": _sha(training_raw),
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
    files = [
        {
            "local_name": "keep-repository.tar.gz",
            "key": descriptor["repo_tar_key"],
            "role": "repository_tar",
            "stage_order": 10,
            "size": len(repository_raw),
            "sha256": _sha(repository_raw),
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
            "size": len(training_raw),
            "sha256": _sha(training_raw),
        },
        {
            "local_name": "watchdog.sh",
            "key": f"campaigns/{RUN_ID}/authorities/watchdog.sh",
            "role": "watchdog",
            "stage_order": 40,
            "size": len(watchdog_raw),
            "sha256": _sha(watchdog_raw),
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
    versions = {
        role: f"version-{role}"
        for role in (
            "repository_tar",
            "approval",
            "training_config",
            "watchdog",
            "artifact_inventory",
            "artifact_audit",
            "descriptor",
        )
    }
    staged = _self_hashed(
        {
            "schema_version": 2,
            "record_type": "glm52_staged_control_plane_ready_v2",
            "run_id": RUN_ID,
            "descriptor_key": descriptor["campaign_descriptor_key"],
            "descriptor_sha256": _sha(descriptor_raw),
            "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
            "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
            "bundle_manifest_key": manifest_key,
            "bundle_manifest_file_sha256": _sha(manifest_raw),
            "bundle_manifest_body_sha256": manifest["bundle_manifest_body_sha256"],
            "bundle_manifest_version_id": "version-bundle_manifest",
            "staged_object_version_ids": versions,
            "artifact_audit_key": (
                f"campaigns/{RUN_ID}/audits/artifact-audit-{_sha(audit_raw)}.json"
            ),
            "artifact_audit_sha256": _sha(audit_raw),
            "staged_at": "2026-07-26T17:00:00Z",
        },
        "ready_body_sha256",
    )
    staged_raw = _file_bytes(staged)
    staged_key = (
        f"campaigns/{RUN_ID}/submissions/cache-seed/STAGED_CONTROL_PLANE_READY.json"
    )
    rehearsal = _self_hashed(
        {
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
            "staged_readiness_file_sha256": _sha(staged_raw),
            "staged_readiness_body_sha256": staged["ready_body_sha256"],
            "artifact_inventory_key": descriptor["artifacts"]["artifact_inventory_key"],
            "artifact_inventory_file_sha256": descriptor["artifacts"][
                "artifact_inventory_sha256"
            ],
            "artifact_inventory_body_sha256": "1" * 64,
            "artifact_audit_key": staged["artifact_audit_key"],
            "artifact_audit_file_sha256": staged["artifact_audit_sha256"],
            "extracted_repo_path": "/tmp/glm52-cache-seed-rehearsal/repo",
            "production_repo_path": "/opt/keep-campaign/repo",
            "production_resume_root": "/mnt/nvme/glm52-campaign",
            "skypilot_task_path": (
                "/opt/keep-campaign/repo/aws/glm52-gpu/skypilot/glm52-campaign.yaml"
            ),
            "skypilot_task_file_sha256": "a" * 64,
            "skypilot_config_file_sha256": "b" * 64,
            "skypilot_validation_file_sha256": "c" * 64,
            "skypilot_version": "0.13.0",
            "skypilot_task_name": "glm52-campaign",
            "completed_at": "2026-07-26T17:30:00Z",
        },
        "rehearsal_body_sha256",
    )
    rehearsal_raw = _file_bytes(rehearsal)
    s3_raw = _file_bytes(_s3_observation())
    ec2_raw = _file_bytes(_ec2_observation())
    allowance = build_gpu_launch_allowance(
        descriptor_raw=descriptor_raw,
        expected_descriptor_sha256=_sha(descriptor_raw),
        approval_raw=approval_raw,
        expected_approval_sha256=_sha(approval_raw),
        s3_spend_ledger_absence_observation_raw=s3_raw,
        expected_s3_spend_ledger_absence_observation_sha256=_sha(s3_raw),
        ec2_tagged_p5_zero_inventory_observation_raw=ec2_raw,
        expected_ec2_tagged_p5_zero_inventory_observation_sha256=_sha(ec2_raw),
        observed_at=OBSERVED_AT,
    )
    return {
        "descriptor": descriptor,
        "descriptor_file_sha256": _sha(descriptor_raw),
        "approval": approval,
        "approval_file_sha256": _sha(approval_raw),
        "staged_readiness": staged,
        "staged_readiness_file_sha256": _sha(staged_raw),
        "staged_readiness_version_id": STAGED_VERSION_ID,
        "bundle_manifest": manifest,
        "bundle_manifest_file_sha256": _sha(manifest_raw),
        "rehearsal_evidence": rehearsal,
        "rehearsal_evidence_file_sha256": _sha(rehearsal_raw),
        "s3_spend_ledger_absence_observation_file_sha256": _sha(s3_raw),
        "ec2_tagged_p5_zero_inventory_observation_file_sha256": _sha(ec2_raw),
        "launch_allowance": allowance,
        "launch_allowance_file_sha256": _sha(_file_bytes(allowance)),
    }


def _intent_arguments(authorities: dict[str, object]) -> dict[str, object]:
    return {
        **authorities,
        "intent_at": OBSERVED_AT,
    }


def _claim_arguments(
    authorities: dict[str, object],
    intent: dict[str, object],
) -> dict[str, object]:
    return {
        "descriptor": authorities["descriptor"],
        "descriptor_file_sha256": authorities["descriptor_file_sha256"],
        "intent": intent,
        "intent_file_sha256": _sha(_file_bytes(intent)),
        "launch_allowance": authorities["launch_allowance"],
        "launch_allowance_file_sha256": authorities["launch_allowance_file_sha256"],
        "claimed_at": OBSERVED_AT,
    }


def test_feature_module_exists_with_exact_public_surface() -> None:
    """Catches a missing cache-seed authority module or exported API drift."""

    module = _module()

    assert module.__all__ == [
        "CLAIM_DIGEST_FIELD",
        "CLAIM_RECORD_TYPE",
        "CLAIM_SCHEMA_VERSION",
        "INTENT_DIGEST_FIELD",
        "INTENT_RECORD_TYPE",
        "INTENT_SCHEMA_VERSION",
        "MANAGED_MODE",
        "MAX_LIVE_AGE_SECONDS",
        "CacheSeedLaunchError",
        "build_cache_seed_launch_claim",
        "build_cache_seed_submission_intent",
        "cache_seed_launch_claim_s3_key",
        "cache_seed_submission_intent_s3_key",
        "canonical_file_bytes",
        "ec2_tagged_p5_zero_inventory_observation_s3_key",
        "s3_spend_ledger_absence_observation_s3_key",
        "validate_cache_seed_launch_claim",
        "validate_cache_seed_submission_intent",
    ]


def test_canonical_file_bytes_and_all_key_grammars_are_exact() -> None:
    """Catches noncanonical JSON or any foreign/cache-seed key coordinate."""

    module = _module()
    assert module.canonical_file_bytes({"z": 1, "ascii": "ok"}) == (
        b'{"ascii":"ok","z":1}\n'
    )
    assert module.s3_spend_ledger_absence_observation_s3_key(
        run_id=RUN_ID,
        observation_file_sha256="1" * 64,
    ) == (
        f"campaigns/{RUN_ID}/submissions/cache-seed/launch-observations/"
        f"s3-spend-ledger-absence/{'1' * 64}/"
        "S3_SPEND_LEDGER_ABSENCE_OBSERVATION.json"
    )
    assert module.ec2_tagged_p5_zero_inventory_observation_s3_key(
        run_id=RUN_ID,
        observation_file_sha256="2" * 64,
    ) == (
        f"campaigns/{RUN_ID}/submissions/cache-seed/launch-observations/"
        f"ec2-tagged-p5-zero-inventory/{'2' * 64}/"
        "EC2_TAGGED_P5_ZERO_INVENTORY_OBSERVATION.json"
    )
    assert module.cache_seed_submission_intent_s3_key(
        run_id=RUN_ID,
        intent_body_sha256="3" * 64,
    ) == (
        f"campaigns/{RUN_ID}/submissions/cache-seed/intents/"
        f"{'3' * 64}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    assert module.cache_seed_launch_claim_s3_key(
        run_id=RUN_ID,
        descriptor_file_sha256="4" * 64,
    ) == (
        f"campaigns/{RUN_ID}/submissions/cache-seed/launch-claims/"
        f"{'4' * 64}/LAUNCH_CLAIM.json"
    )


def test_intent_and_claim_build_validate_and_derive_stable_keys() -> None:
    """Catches missing exact bindings, wrong hashes, or allowance-keyed claims."""

    module = _module()
    authorities = _authorities()
    intent = module.build_cache_seed_submission_intent(**_intent_arguments(authorities))
    assert intent["intent_body_sha256"] == EXPECTED_INTENT_BODY_SHA256
    assert _sha(_file_bytes(intent)) == EXPECTED_INTENT_FILE_SHA256
    assert module.cache_seed_submission_intent_s3_key(
        run_id=RUN_ID,
        intent_body_sha256=EXPECTED_INTENT_BODY_SHA256,
    ) == (
        "campaigns/glm52-sky-20260726/submissions/cache-seed/intents/"
        "451273e8a495c9fe22e83fdcea95eda411801881fe1b43c396841a7dd8b8473c/"
        "SKYPILOT_SUBMISSION_INTENT.json"
    )
    assert (
        module.validate_cache_seed_submission_intent(
            intent,
            **authorities,
        )
        == intent
    )
    assert intent["managed_mode"] == "cache-seed"
    assert intent["intent_at"] == OBSERVED_AT
    assert intent["remaining_gpu_seconds"] == 86_400
    assert intent["remaining_gpu_cost_usd"] == 1320.96
    assert intent["cache_seed_allowance_seconds"] == 21_600
    assert intent["cache_seed_allowance_cost_usd"] == 330.24
    assert (
        _sha(_canonical({k: v for k, v in intent.items() if k != "intent_body_sha256"}))
        == intent["intent_body_sha256"]
    )
    claim = module.build_cache_seed_launch_claim(
        **_claim_arguments(authorities, intent)
    )
    assert claim["launch_claim_body_sha256"] == EXPECTED_CLAIM_BODY_SHA256
    assert _sha(_file_bytes(claim)) == EXPECTED_CLAIM_FILE_SHA256
    assert module.cache_seed_launch_claim_s3_key(
        run_id=RUN_ID,
        descriptor_file_sha256=str(authorities["descriptor_file_sha256"]),
    ) == (
        "campaigns/glm52-sky-20260726/submissions/cache-seed/launch-claims/"
        "70510655843ff2b37028a95ec625ad7e4d1a537eeb9c2110a3294a2e0bcb8fed/"
        "LAUNCH_CLAIM.json"
    )
    assert (
        module.validate_cache_seed_launch_claim(
            claim,
            **{
                key: value
                for key, value in _claim_arguments(authorities, intent).items()
                if key != "claimed_at"
            },
        )
        == claim
    )
    assert claim["submission_intent_file_sha256"] == _sha(_file_bytes(intent))
    assert module.cache_seed_launch_claim_s3_key(
        run_id=RUN_ID,
        descriptor_file_sha256=str(authorities["descriptor_file_sha256"]),
    ).endswith(f"/{authorities['descriptor_file_sha256']}/LAUNCH_CLAIM.json")


def test_claim_key_is_stable_across_allowances_but_descriptor_addressed() -> None:
    """Catches a short-lived allowance-addressed or run-only claim key."""

    module = _module()
    authorities = _authorities()
    first = module.cache_seed_launch_claim_s3_key(
        run_id=RUN_ID,
        descriptor_file_sha256=str(authorities["descriptor_file_sha256"]),
    )
    second = module.cache_seed_launch_claim_s3_key(
        run_id=RUN_ID,
        descriptor_file_sha256=str(authorities["descriptor_file_sha256"]),
    )
    rewindowed = module.cache_seed_launch_claim_s3_key(
        run_id=RUN_ID,
        descriptor_file_sha256="f" * 64,
    )
    assert first == second
    assert rewindowed != first


@pytest.mark.parametrize(
    "value",
    [
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": object()},
        ["not", "a", "mapping"],
    ],
)
def test_canonical_file_bytes_rejects_nonfinite_nonjson_and_nonmapping(
    value: object,
) -> None:
    """Catches permissive JSON serialization at the durable byte boundary."""

    module = _module()
    with pytest.raises(module.CacheSeedLaunchError):
        module.canonical_file_bytes(value)


class _ForeignDict(dict[str, object]):
    pass


class _ForeignList(list[object]):
    pass


class _ForeignInt(int):
    pass


class _ForeignString(str):
    pass


@pytest.mark.parametrize(
    "value",
    [
        {1: "x"},
        {"nested": {2: "x"}},
        {"value": (1, 2)},
        {"value": {1, 2}},
        {"value": b"bytes"},
        {"value": _ForeignList([1])},
        {"value": _ForeignInt(1)},
        {"value": _ForeignString("x")},
        _ForeignDict({"value": 1}),
    ],
)
def test_canonical_file_bytes_rejects_json_coercion_and_subclass_confusion(
    value: object,
) -> None:
    """Catches coercing foreign Python types into collision-prone JSON values."""

    module = _module()
    with pytest.raises(module.CacheSeedLaunchError):
        module.canonical_file_bytes(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("managed_mode", "qualification"),
        ("remaining_gpu_seconds", 86_400.0),
        ("remaining_gpu_cost_usd", 1320),
        ("staged_readiness_version_id", "null"),
        ("open_allocation_count", False),
    ],
)
def test_coherently_rehashed_intent_type_and_mode_drift_fails(
    field: str,
    value: object,
) -> None:
    """Catches semantically foreign intent values despite a coherent self-hash."""

    module = _module()
    authorities = _authorities()
    intent = module.build_cache_seed_submission_intent(**_intent_arguments(authorities))
    intent[field] = value
    intent = _rehash(intent, "intent_body_sha256")
    with pytest.raises(module.CacheSeedLaunchError):
        module.validate_cache_seed_submission_intent(intent, **authorities)


@pytest.mark.parametrize(
    "foreign_field",
    [
        "cache_seed_acceptance_key",
        "gpu_spend_snapshot_key",
        "qualification_ready_key",
        "h100_ready_key",
        "acquisition_key",
        "production_key",
    ],
)
def test_mixed_lifecycle_fields_are_rejected(foreign_field: str) -> None:
    """Catches mixed cache-seed/qualification/production record schemas."""

    module = _module()
    authorities = _authorities()
    intent = module.build_cache_seed_submission_intent(**_intent_arguments(authorities))
    intent[foreign_field] = "foreign"
    intent = _rehash(intent, "intent_body_sha256")
    with pytest.raises(module.CacheSeedLaunchError):
        module.validate_cache_seed_submission_intent(intent, **authorities)


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("account_id", "111111111111"),
        ("region", "us-east-1"),
        ("sky_job_name", f"{RUN_ID}-qualification"),
        (
            "descriptor_key",
            f"campaigns/{RUN_ID}/submissions/foreign/campaign-descriptor-v2.json",
        ),
        ("descriptor_file_sha256", "f" * 64),
        ("descriptor_body_sha256", "e" * 64),
        ("campaign_identity_sha256", "d" * 64),
        (
            "approval_key",
            f"campaigns/{RUN_ID}/authorities/FOREIGN_GPU_SPEND_APPROVAL.json",
        ),
        ("approval_sha256", "c" * 64),
        ("approval_body_sha256", "b" * 64),
        (
            "repo_tar_key",
            f"campaigns/{RUN_ID}/repository/foreign-cache-seed.tar.gz",
        ),
        ("repo_tar_sha256", "a" * 64),
        (
            "staged_readiness_key",
            f"campaigns/{RUN_ID}/submissions/cache-seed/FOREIGN_READY.json",
        ),
        (
            "bundle_manifest_key",
            (
                f"campaigns/{RUN_ID}/submissions/cache-seed/bundle-manifests/"
                f"{'7' * 64}/bundle-manifest-v1.json"
            ),
        ),
        ("bundle_manifest_body_sha256", "7" * 64),
        (
            "rehearsal_evidence_key",
            (
                f"campaigns/{RUN_ID}/qualification/rehearsals/"
                f"{'5' * 64}/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
            ),
        ),
        ("rehearsal_evidence_body_sha256", "5" * 64),
        (
            "s3_spend_ledger_absence_observation_key",
            (
                f"campaigns/{RUN_ID}/submissions/cache-seed/launch-observations/"
                f"s3-spend-ledger-absence/{'3' * 64}/"
                "S3_SPEND_LEDGER_ABSENCE_OBSERVATION.json"
            ),
        ),
        ("s3_spend_ledger_absence_observation_sha256", "3" * 64),
        (
            "ec2_tagged_p5_zero_inventory_observation_key",
            (
                f"campaigns/{RUN_ID}/submissions/cache-seed/launch-observations/"
                f"ec2-tagged-p5-zero-inventory/{'2' * 64}/"
                "EC2_TAGGED_P5_ZERO_INVENTORY_OBSERVATION.json"
            ),
        ),
        ("ec2_tagged_p5_zero_inventory_observation_sha256", "2" * 64),
        (
            "launch_allowance_key",
            (
                f"campaigns/{RUN_ID}/launch-allowances/{'1' * 64}/"
                "GPU_LAUNCH_ALLOWANCE.json"
            ),
        ),
        ("launch_allowance_file_sha256", "1" * 64),
        ("launch_allowance_body_sha256", "0" * 64),
    ],
)
def test_claim_builder_rejects_every_coherently_rehashed_foreign_intent_binding(
    field: str,
    foreign: object,
) -> None:
    """Catches copying a safe-looking foreign intent coordinate into a claim."""

    module = _module()
    authorities = _authorities()
    intent = module.build_cache_seed_submission_intent(**_intent_arguments(authorities))
    intent[field] = foreign
    intent = _rehash(intent, "intent_body_sha256")
    arguments = _claim_arguments(authorities, intent)
    arguments["intent_file_sha256"] = _sha(_file_bytes(intent))

    with pytest.raises(module.CacheSeedLaunchError):
        module.build_cache_seed_launch_claim(**arguments)


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("staged_readiness_file_sha256", "9" * 64),
        ("staged_readiness_body_sha256", "8" * 64),
        ("staged_readiness_version_id", "foreign-staged-version"),
        ("bundle_manifest_file_sha256", "6" * 64),
        ("bundle_manifest_version_id", "foreign-manifest-version"),
        ("rehearsal_evidence_file_sha256", "4" * 64),
    ],
)
def test_intent_validator_rejects_source_authenticated_file_or_version_drift(
    field: str,
    foreign: object,
) -> None:
    """Catches rehashing staged/manifest/rehearsal pins away from source files."""

    module = _module()
    authorities = _authorities()
    intent = module.build_cache_seed_submission_intent(**_intent_arguments(authorities))
    intent[field] = foreign
    intent = _rehash(intent, "intent_body_sha256")

    with pytest.raises(module.CacheSeedLaunchError):
        module.validate_cache_seed_submission_intent(intent, **authorities)


@pytest.mark.parametrize(
    ("claimed_at", "accepted"),
    [
        (OBSERVED_AT, True),
        ("2026-07-26T17:45:59Z", True),
        ("2026-07-26T17:46:00Z", False),
        ("2026-07-26T17:44:59Z", False),
    ],
)
def test_claim_time_window_is_inclusive_only_at_observation(
    claimed_at: str,
    accepted: bool,
) -> None:
    """Catches inclusive expiry or claims before the observed allowance."""

    module = _module()
    authorities = _authorities()
    intent = module.build_cache_seed_submission_intent(**_intent_arguments(authorities))
    arguments = _claim_arguments(authorities, intent)
    arguments["claimed_at"] = claimed_at
    if accepted:
        assert module.build_cache_seed_launch_claim(**arguments)["claimed_at"] == (
            claimed_at
        )
    else:
        with pytest.raises(module.CacheSeedLaunchError):
            module.build_cache_seed_launch_claim(**arguments)


def test_actual_python_396_build_validate_and_key_round_trip(
    tmp_path: Path,
) -> None:
    """Catches Python 3.9 or flat-source incompatibility beyond import-only proof."""

    authorities = _authorities()
    payload = {
        "descriptor": authorities["descriptor"],
        "approval": authorities["approval"],
        "staged_readiness": authorities["staged_readiness"],
        "bundle_manifest": authorities["bundle_manifest"],
        "rehearsal_evidence": authorities["rehearsal_evidence"],
        "s3_observation": _s3_observation(),
        "ec2_observation": _ec2_observation(),
        "staged_readiness_version_id": STAGED_VERSION_ID,
    }
    fixture = tmp_path / "python39-cache-seed-authorities.json"
    fixture.write_bytes(_file_bytes(payload))
    quality = Path(__file__).parents[1] / "src/mlx_vq/quality"
    script = r"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from glm52_gpu_launch_allowance import build_gpu_launch_allowance
from glm52_sky_cache_seed_launch import (
    build_cache_seed_launch_claim,
    build_cache_seed_submission_intent,
    cache_seed_launch_claim_s3_key,
    cache_seed_submission_intent_s3_key,
    canonical_file_bytes,
    validate_cache_seed_launch_claim,
    validate_cache_seed_submission_intent,
)

data = json.loads(Path(sys.argv[2]).read_bytes())
descriptor = data["descriptor"]
approval = data["approval"]
staged = data["staged_readiness"]
manifest = data["bundle_manifest"]
rehearsal = data["rehearsal_evidence"]
s3_raw = canonical_file_bytes(data["s3_observation"])
ec2_raw = canonical_file_bytes(data["ec2_observation"])
descriptor_raw = canonical_file_bytes(descriptor)
approval_raw = canonical_file_bytes(approval)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
allowance = build_gpu_launch_allowance(
    descriptor_raw=descriptor_raw,
    expected_descriptor_sha256=sha(descriptor_raw),
    approval_raw=approval_raw,
    expected_approval_sha256=sha(approval_raw),
    s3_spend_ledger_absence_observation_raw=s3_raw,
    expected_s3_spend_ledger_absence_observation_sha256=sha(s3_raw),
    ec2_tagged_p5_zero_inventory_observation_raw=ec2_raw,
    expected_ec2_tagged_p5_zero_inventory_observation_sha256=sha(ec2_raw),
    observed_at="2026-07-26T17:45:00Z",
)
allowance_raw = canonical_file_bytes(allowance)
kwargs = dict(
    descriptor=descriptor,
    descriptor_file_sha256=sha(descriptor_raw),
    approval=approval,
    approval_file_sha256=sha(approval_raw),
    staged_readiness=staged,
    staged_readiness_file_sha256=sha(canonical_file_bytes(staged)),
    staged_readiness_version_id=data["staged_readiness_version_id"],
    bundle_manifest=manifest,
    bundle_manifest_file_sha256=sha(canonical_file_bytes(manifest)),
    rehearsal_evidence=rehearsal,
    rehearsal_evidence_file_sha256=sha(canonical_file_bytes(rehearsal)),
    s3_spend_ledger_absence_observation_file_sha256=sha(s3_raw),
    ec2_tagged_p5_zero_inventory_observation_file_sha256=sha(ec2_raw),
    launch_allowance=allowance,
    launch_allowance_file_sha256=sha(allowance_raw),
)
intent = build_cache_seed_submission_intent(
    intent_at="2026-07-26T17:45:00Z",
    **kwargs
)
validate_cache_seed_submission_intent(intent, **kwargs)
intent_raw = canonical_file_bytes(intent)
claim_kwargs = dict(
    descriptor=descriptor,
    descriptor_file_sha256=sha(descriptor_raw),
    intent=intent,
    intent_file_sha256=sha(intent_raw),
    launch_allowance=allowance,
    launch_allowance_file_sha256=sha(allowance_raw),
)
claim = build_cache_seed_launch_claim(
    claimed_at="2026-07-26T17:45:00Z",
    **claim_kwargs
)
validate_cache_seed_launch_claim(claim, **claim_kwargs)
print(json.dumps({
    "python": ".".join(str(part) for part in sys.version_info[:3]),
    "allowance_body_sha256": allowance["allowance_body_sha256"],
    "intent_body_sha256": intent["intent_body_sha256"],
    "intent_key": cache_seed_submission_intent_s3_key(
        run_id=descriptor["run_id"],
        intent_body_sha256=intent["intent_body_sha256"],
    ),
    "claim_body_sha256": claim["launch_claim_body_sha256"],
    "claim_key": cache_seed_launch_claim_s3_key(
        run_id=descriptor["run_id"],
        descriptor_file_sha256=sha(descriptor_raw),
    ),
}, sort_keys=True))
"""
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script, str(quality), str(fixture)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output == {
        "python": "3.9.6",
        "allowance_body_sha256": (
            "64e238305c48734a3da3348edf7ffed7644ce7b93fdf600a520093865133c658"
        ),
        "intent_body_sha256": EXPECTED_INTENT_BODY_SHA256,
        "intent_key": (
            "campaigns/glm52-sky-20260726/submissions/cache-seed/intents/"
            "451273e8a495c9fe22e83fdcea95eda411801881fe1b43c396841a7dd8b8473c/"
            "SKYPILOT_SUBMISSION_INTENT.json"
        ),
        "claim_body_sha256": EXPECTED_CLAIM_BODY_SHA256,
        "claim_key": (
            "campaigns/glm52-sky-20260726/submissions/cache-seed/launch-claims/"
            "70510655843ff2b37028a95ec625ad7e4d1a537eeb9c2110a3294a2e0bcb8fed/"
            "LAUNCH_CLAIM.json"
        ),
    }
