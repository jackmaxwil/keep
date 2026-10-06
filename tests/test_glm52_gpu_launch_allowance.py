"""Pure pre-seed GPU launch allowance authority tests."""

# ruff: noqa: UP017

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)

UTC = timezone.utc
RUN_ID = "glm52-sky-20260726"
OBSERVED_AT = "2026-07-26T17:45:00Z"
MUST_START_BY = datetime(2026, 7, 26, 18, 45, tzinfo=UTC)
BUCKET = "keep-glm52-us-west-2-246813579024"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_file(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha_raw(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _authorities(
    *,
    submission_id: str = "first",
    must_start_by: datetime = MUST_START_BY,
    qualification_cache_prefix: str = "qualification-cache/",
    qualification_cache_manifest_sha256: str = "0" * 64,
) -> tuple[bytes, bytes]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
        slack_permalink=None,
    )
    approval_raw = _canonical_file(approval)
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=must_start_by,
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=f"campaigns/{RUN_ID}/repository/repo.tar.gz",
        repo_tar_sha256="1" * 64,
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/{submission_id}/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
        approval_sha256=_sha_raw(approval_raw),
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
            "training_config_key": (f"campaigns/{RUN_ID}/authorities/training.json"),
            "training_config_sha256": "7" * 64,
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'8' * 64}.json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": qualification_cache_prefix,
            "qualification_cache_manifest_sha256": (
                qualification_cache_manifest_sha256
            ),
        },
    )
    return _canonical_file(descriptor), approval_raw


def _history_filters() -> list[dict[str, object]]:
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
        "legacy_ledger_key": (f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"),
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


def _ec2_observation() -> dict[str, object]:
    history_filters = _history_filters()
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
        "history_request": {"filters": history_filters},
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
                *history_filters,
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


def _inputs() -> dict[str, object]:
    descriptor_raw, approval_raw = _authorities()
    s3_raw = _canonical_file(_s3_observation())
    ec2_raw = _canonical_file(_ec2_observation())
    return {
        "descriptor_raw": descriptor_raw,
        "expected_descriptor_sha256": _sha_raw(descriptor_raw),
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha_raw(approval_raw),
        "s3_spend_ledger_absence_observation_raw": s3_raw,
        "expected_s3_spend_ledger_absence_observation_sha256": _sha_raw(s3_raw),
        "ec2_tagged_p5_zero_inventory_observation_raw": ec2_raw,
        "expected_ec2_tagged_p5_zero_inventory_observation_sha256": _sha_raw(ec2_raw),
        "observed_at": OBSERVED_AT,
    }


def _module() -> object:
    return importlib.import_module("mlx_vq.quality.glm52_gpu_launch_allowance")


def _rehash_descriptor(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("descriptor_body_sha256", None)
    identity = {
        name: item
        for name, item in body.items()
        if name
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    body["campaign_identity_sha256"] = _sha_raw(_canonical(identity))
    return {**body, "descriptor_body_sha256": _sha_raw(_canonical(body))}


def _rehash_approval(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("approval_body_sha256", None)
    return {**body, "approval_body_sha256": _sha_raw(_canonical(body))}


def _replace_descriptor(
    inputs: dict[str, object],
    value: dict[str, object],
) -> dict[str, object]:
    raw = _canonical_file(value)
    return {
        **inputs,
        "descriptor_raw": raw,
        "expected_descriptor_sha256": _sha_raw(raw),
    }


def _replace_approval_and_rebind_descriptor(
    inputs: dict[str, object],
    approval: dict[str, object],
) -> dict[str, object]:
    approval_raw = _canonical_file(approval)
    descriptor = json.loads(inputs["descriptor_raw"])
    descriptor["approval_sha256"] = _sha_raw(approval_raw)
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))
    return {
        **inputs,
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha_raw(approval_raw),
    }


def _replace_raw(
    inputs: dict[str, object],
    *,
    raw_field: str,
    expected_field: str,
    raw: bytes,
) -> dict[str, object]:
    return {
        **inputs,
        raw_field: raw,
        expected_field: _sha_raw(raw),
    }


def _json_copy(value: object) -> object:
    return json.loads(_canonical(value))


def _replace_s3_observation(
    inputs: dict[str, object],
    value: dict[str, object],
) -> dict[str, object]:
    raw = _canonical_file(value)
    return {
        **inputs,
        "s3_spend_ledger_absence_observation_raw": raw,
        "expected_s3_spend_ledger_absence_observation_sha256": _sha_raw(raw),
    }


def _replace_ec2_observation(
    inputs: dict[str, object],
    value: dict[str, object],
) -> dict[str, object]:
    raw = _canonical_file(value)
    return {
        **inputs,
        "ec2_tagged_p5_zero_inventory_observation_raw": raw,
        "expected_ec2_tagged_p5_zero_inventory_observation_sha256": _sha_raw(raw),
    }


def _valid_allowance() -> dict[str, object]:
    return _module().build_gpu_launch_allowance(**_inputs())


def _rehash_allowance(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("allowance_body_sha256", None)
    return {**body, "allowance_body_sha256": _sha_raw(_canonical(body))}


def test_builds_exact_deterministic_first_allocation_allowance() -> None:
    module = _module()
    inputs = _inputs()

    first = module.build_gpu_launch_allowance(**inputs)
    second = module.build_gpu_launch_allowance(**inputs)

    assert first == second
    assert set(first) == {
        "schema_version",
        "record_type",
        "run_id",
        "managed_mode",
        "sky_job_name",
        "account_id",
        "region",
        "instance_type",
        "instance_count",
        "use_spot",
        "bucket",
        "campaign_identity_sha256",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "descriptor_key",
        "approval_sha256",
        "approval_body_sha256",
        "approval_key",
        "observed_at",
        "must_start_by",
        "not_after",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_seconds",
        "remaining_gpu_seconds",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "cache_seed_max_seconds",
        "cache_seed_max_cost_usd",
        "allowed_gpu_seconds",
        "allowed_gpu_cost_usd",
        "open_allocation_count",
        "spend_ledger_latest_marker_key",
        "spend_ledger_legacy_key",
        "spend_ledger_records_prefix",
        "s3_spend_ledger_absence_observation_sha256",
        "ec2_tagged_p5_zero_inventory_observation_sha256",
        "ec2_tagged_p5_history_instance_ids",
        "ec2_active_tagged_p5_instance_ids",
        "allowance_body_sha256",
    }
    assert first["schema_version"] == 1
    assert first["record_type"] == "glm52_gpu_launch_allowance_v1"
    assert first["run_id"] == RUN_ID
    assert first["managed_mode"] == "cache-seed"
    assert first["sky_job_name"] == f"{RUN_ID}-cache-seed"
    assert first["account_id"] == "246813579024"
    assert first["region"] == "us-west-2"
    assert first["instance_type"] == "p5.48xlarge"
    assert first["instance_count"] == 1
    assert first["use_spot"] is False
    assert first["approved_gpu_runtime_seconds"] == 86_400
    assert first["approved_gpu_cost_usd"] == 1_320.96
    assert first["hourly_cost_usd"] == 55.04
    assert first["consumed_gpu_seconds"] == 0
    assert first["remaining_gpu_seconds"] == 86_400
    assert first["consumed_gpu_cost_usd"] == 0.0
    assert first["remaining_gpu_cost_usd"] == 1_320.96
    assert first["cache_seed_max_seconds"] == 21_600
    assert first["cache_seed_max_cost_usd"] == 330.24
    assert first["allowed_gpu_seconds"] == 21_600
    assert first["allowed_gpu_cost_usd"] == 330.24
    assert first["open_allocation_count"] == 0
    assert first["spend_ledger_legacy_key"] == (
        f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"
    )
    assert first["observed_at"] == OBSERVED_AT
    assert first["must_start_by"] == "2026-07-26T18:45:00Z"
    assert first["not_after"] == "2026-07-26T17:46:00Z"
    assert first["campaign_identity_sha256"] == (
        "be8fd8cd053ae57d74c6e1f1e52845f803025a9fb1d781a8c570cd720e4e1b61"
    )
    assert first["descriptor_body_sha256"] == (
        "6d2013d05f3ee3c5a808be3058c9655fd61b2cd21c4e2e0a8251a4513fb23b57"
    )
    assert first["approval_body_sha256"] == (
        "a146058e769112467d0273ff2758f1f76700ee61899ac83bb1f7c54a8ab8e378"
    )
    assert first["descriptor_sha256"] == inputs["expected_descriptor_sha256"]
    assert first["approval_sha256"] == inputs["expected_approval_sha256"]
    assert (
        first["s3_spend_ledger_absence_observation_sha256"]
        == inputs["expected_s3_spend_ledger_absence_observation_sha256"]
    )
    assert (
        first["ec2_tagged_p5_zero_inventory_observation_sha256"]
        == inputs["expected_ec2_tagged_p5_zero_inventory_observation_sha256"]
    )
    assert first["ec2_tagged_p5_history_instance_ids"] == []
    assert first["ec2_active_tagged_p5_instance_ids"] == []
    assert first["allowance_body_sha256"] == (
        "61af17504eb49b8386ee7632f305729b56feb811c4c6f9a01d22cdd59261bd26"
    )
    assert _canonical(first) == (
        b'{"account_id":"246813579024","allowance_body_sha256":'
        b'"61af17504eb49b8386ee7632f305729b56feb811c4c6f9a01d22cdd59261bd26",'
        b'"allowed_gpu_cost_usd":330.24,"allowed_gpu_seconds":21600,'
        b'"approval_body_sha256":'
        b'"a146058e769112467d0273ff2758f1f76700ee61899ac83bb1f7c54a8ab8e378",'
        b'"approval_key":"campaigns/glm52-sky-20260726/authorities/'
        b'GPU_SPEND_APPROVAL.json","approval_sha256":'
        b'"9e49121eb3462c25ac652f68988c1510372151e43abc2bcb9c107d2713ea531e",'
        b'"approved_gpu_cost_usd":1320.96,"approved_gpu_runtime_seconds":86400,'
        b'"bucket":"keep-glm52-us-west-2-246813579024",'
        b'"cache_seed_max_cost_usd":330.24,"cache_seed_max_seconds":21600,'
        b'"campaign_identity_sha256":'
        b'"be8fd8cd053ae57d74c6e1f1e52845f803025a9fb1d781a8c570cd720e4e1b61",'
        b'"consumed_gpu_cost_usd":0.0,"consumed_gpu_seconds":0,'
        b'"descriptor_body_sha256":'
        b'"6d2013d05f3ee3c5a808be3058c9655fd61b2cd21c4e2e0a8251a4513fb23b57",'
        b'"descriptor_key":"campaigns/glm52-sky-20260726/submissions/first/'
        b'campaign-descriptor-v2.json","descriptor_sha256":'
        b'"93060cd3e815d11492398e9e0871cc57f420fe69d82e7696a72bc09c68bcc57f",'
        b'"ec2_active_tagged_p5_instance_ids":[],'
        b'"ec2_tagged_p5_history_instance_ids":[],'
        b'"ec2_tagged_p5_zero_inventory_observation_sha256":'
        b'"40789da4114c1feb0942de90876e91dab080fdd94af519a3b0ff517a201c8248",'
        b'"hourly_cost_usd":55.04,"instance_count":1,'
        b'"instance_type":"p5.48xlarge","managed_mode":"cache-seed",'
        b'"must_start_by":"2026-07-26T18:45:00Z",'
        b'"not_after":"2026-07-26T17:46:00Z","observed_at":'
        b'"2026-07-26T17:45:00Z","open_allocation_count":0,'
        b'"record_type":"glm52_gpu_launch_allowance_v1",'
        b'"region":"us-west-2","remaining_gpu_cost_usd":1320.96,'
        b'"remaining_gpu_seconds":86400,"run_id":"glm52-sky-20260726",'
        b'"s3_spend_ledger_absence_observation_sha256":'
        b'"d2a853391a9b4b4450658c1ff1ab1aa7b95490d1bd6a2bb899a09b5dc169d7b4",'
        b'"schema_version":1,"sky_job_name":"glm52-sky-20260726-cache-seed",'
        b'"spend_ledger_latest_marker_key":"campaigns/glm52-sky-20260726/'
        b'runtime/GPU_SPEND_LEDGER_LATEST.json",'
        b'"spend_ledger_legacy_key":"campaigns/glm52-sky-20260726/'
        b'runtime/GPU_SPEND_LEDGER.jsonl",'
        b'"spend_ledger_records_prefix":"campaigns/glm52-sky-20260726/runtime/'
        b'spend-ledger/records/","use_spot":false}'
    )
    assert module.gpu_launch_allowance_s3_key(
        run_id=RUN_ID,
        allowance_body_sha256=first["allowance_body_sha256"],
    ) == (
        "campaigns/glm52-sky-20260726/submissions/cache-seed/"
        "launch-allowances/"
        "61af17504eb49b8386ee7632f305729b56feb811c4c6f9a01d22cdd59261bd26/"
        "GLM52_GPU_LAUNCH_ALLOWANCE.json"
    )


@pytest.mark.parametrize(
    ("raw_field", "expected_field"),
    [
        ("descriptor_raw", "expected_descriptor_sha256"),
        ("approval_raw", "expected_approval_sha256"),
        (
            "s3_spend_ledger_absence_observation_raw",
            "expected_s3_spend_ledger_absence_observation_sha256",
        ),
        (
            "ec2_tagged_p5_zero_inventory_observation_raw",
            "expected_ec2_tagged_p5_zero_inventory_observation_sha256",
        ),
    ],
)
def test_rejects_each_exact_raw_hash_mismatch(
    raw_field: str,
    expected_field: str,
) -> None:
    inputs = _inputs()
    inputs[expected_field] = "f" * 64

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="exact-byte SHA-256 mismatch",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("raw_field", "expected_field"),
    [
        ("descriptor_raw", "expected_descriptor_sha256"),
        ("approval_raw", "expected_approval_sha256"),
        (
            "s3_spend_ledger_absence_observation_raw",
            "expected_s3_spend_ledger_absence_observation_sha256",
        ),
        (
            "ec2_tagged_p5_zero_inventory_observation_raw",
            "expected_ec2_tagged_p5_zero_inventory_observation_sha256",
        ),
    ],
)
def test_rejects_non_bytes_for_every_raw_authority(
    raw_field: str,
    expected_field: str,
) -> None:
    inputs = _inputs()
    inputs[raw_field] = bytearray(inputs[raw_field])

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="bytes must be bytes",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "expected_digest",
    [
        "A" * 64,
        "a" * 63,
        "g" * 64,
        7,
        True,
    ],
)
def test_rejects_noncanonical_expected_raw_digest(
    expected_digest: object,
) -> None:
    inputs = _inputs()
    inputs["expected_descriptor_sha256"] = expected_digest

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="lowercase SHA-256",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"schema_version":2}', "canonical"),
        (b'{"schema_version":2}\n\n', "canonical"),
        (b'{ "schema_version": 2 }\n', "canonical"),
        (
            b'{"schema_version":2,"schema_version":2}\n',
            "duplicate",
        ),
        (b"[]\n", "object"),
        (b'{"value":NaN}\n', "nonfinite"),
        (b"\xff\n", "malformed JSON"),
        (b'{"unterminated":\n', "malformed JSON"),
    ],
)
def test_rejects_malformed_or_noncanonical_raw_json(
    raw: bytes,
    message: str,
) -> None:
    inputs = _replace_raw(
        _inputs(),
        raw_field="descriptor_raw",
        expected_field="expected_descriptor_sha256",
        raw=raw,
    )

    with pytest.raises(_module().GpuLaunchAllowanceError, match=message):
        _module().build_gpu_launch_allowance(**inputs)


def test_descriptor_must_bind_approval_file_hash_not_body_hash() -> None:
    inputs = _inputs()
    descriptor = json.loads(inputs["descriptor_raw"])
    approval = json.loads(inputs["approval_raw"])
    descriptor["approval_sha256"] = approval["approval_body_sha256"]
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="exact GPU spend approval bytes",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_rejects_float_descriptor_schema_version_after_coherent_rehash() -> None:
    inputs = _inputs()
    descriptor = json.loads(inputs["descriptor_raw"])
    descriptor["schema_version"] = 2.0
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1.0),
        ("includes_qualification", 1),
        ("includes_recovery_instances", 1),
    ],
)
def test_rejects_nonexact_approval_schema_or_boolean_types(
    field: str,
    value: object,
) -> None:
    inputs = _inputs()
    approval = json.loads(inputs["approval_raw"])
    approval[field] = value
    approval = _rehash_approval(approval)
    inputs = _replace_approval_and_rebind_descriptor(inputs, approval)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "135792468013"),
        ("region", "us-east-1"),
        ("instance_type", "p4d.24xlarge"),
        ("instance_count", 2),
        ("instance_count", True),
        ("max_hourly_cost_usd", 55.05),
        ("max_hourly_cost_usd", 55),
        ("use_spot", True),
    ],
)
def test_rejects_descriptor_campaign_authority_drift(
    field: str,
    value: object,
) -> None:
    inputs = _inputs()
    descriptor = json.loads(inputs["descriptor_raw"])
    descriptor[field] = value
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("extra_field", "value"),
    [
        ("capacity_block_id", "cr-0123456789"),
        ("capacity_reservation_id", "cr-0123456789"),
        ("schema_v1_compatibility", True),
    ],
)
def test_rejects_unknown_mixed_or_capacity_block_descriptor_fields(
    extra_field: str,
    value: object,
) -> None:
    inputs = _inputs()
    descriptor = json.loads(inputs["descriptor_raw"])
    descriptor[extra_field] = value
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="schema mismatch",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_rejects_real_post_seed_cache_manifest() -> None:
    cache_sha = "9" * 64
    descriptor_raw, approval_raw = _authorities(
        qualification_cache_prefix=(f"qualification-cache/seeds/{RUN_ID}/{cache_sha}/"),
        qualification_cache_manifest_sha256=cache_sha,
    )
    inputs = _inputs()
    inputs.update(
        {
            "descriptor_raw": descriptor_raw,
            "expected_descriptor_sha256": _sha_raw(descriptor_raw),
            "approval_raw": approval_raw,
            "expected_approval_sha256": _sha_raw(approval_raw),
        }
    )

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="zero-cache",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "submission_id",
    [
        "first",
        "seed",
        "test",
        "cache-seed",
        "20260726T174500Z-a1b2c3",
    ],
)
def test_accepts_safe_immutable_submission_attempt_ids(
    submission_id: str,
) -> None:
    descriptor_raw, approval_raw = _authorities(submission_id=submission_id)
    inputs = _inputs()
    inputs.update(
        {
            "descriptor_raw": descriptor_raw,
            "expected_descriptor_sha256": _sha_raw(descriptor_raw),
            "approval_raw": approval_raw,
            "expected_approval_sha256": _sha_raw(approval_raw),
        }
    )

    allowance = _module().build_gpu_launch_allowance(**inputs)

    assert allowance["managed_mode"] == "cache-seed"
    assert allowance["descriptor_key"] == (
        f"campaigns/{RUN_ID}/submissions/{submission_id}/campaign-descriptor-v2.json"
    )


@pytest.mark.parametrize(
    "submission_id",
    [
        "",
        "nested/attempt",
        ".",
        "..",
        "has space",
        "has\tcontrol",
        "back\\slash",
        "wild*card",
        "wild?card",
        "wild[card]",
        "/leading",
        "trailing/",
    ],
)
def test_rejects_unsafe_immutable_submission_attempt_ids(
    submission_id: str,
) -> None:
    descriptor_raw, approval_raw = _authorities(submission_id=submission_id)
    inputs = _inputs()
    inputs.update(
        {
            "descriptor_raw": descriptor_raw,
            "expected_descriptor_sha256": _sha_raw(descriptor_raw),
            "approval_raw": approval_raw,
            "expected_approval_sha256": _sha_raw(approval_raw),
        }
    )

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


def test_rejects_foreign_run_descriptor_coordinate() -> None:
    inputs = _inputs()
    descriptor = json.loads(inputs["descriptor_raw"])
    descriptor["campaign_descriptor_key"] = (
        "campaigns/glm52-sky-foreign/submissions/first/campaign-descriptor-v2.json"
    )
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "bucket",
    [
        "ab",
        "a" * 64,
        "HasUppercase",
        "bad..dots",
        "192.168.10.1",
        "xn--reserved",
        "sthree-reserved",
        "amzn-s3-demo-reserved",
        "bucket-s3alias",
        "bucket--ol-s3",
        "bucket.mrap",
        "bucket--x-s3",
        "bucket--table-s3",
    ],
)
def test_rejects_invalid_s3_bucket_grammar(bucket: str) -> None:
    inputs = _inputs()
    descriptor = json.loads(inputs["descriptor_raw"])
    descriptor["bucket"] = bucket
    descriptor["jobs_bucket"] = bucket
    inputs = _replace_descriptor(inputs, _rehash_descriptor(descriptor))

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="bucket",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_rewindowed_attempts_keep_campaign_identity_but_change_allowance() -> None:
    first = _inputs()
    first_allowance = _module().build_gpu_launch_allowance(**first)
    descriptor_raw, approval_raw = _authorities(
        submission_id="20260726T184500Z-a1b2c3",
        must_start_by=datetime(2026, 7, 26, 19, 45, tzinfo=UTC),
    )
    second = {
        **first,
        "descriptor_raw": descriptor_raw,
        "expected_descriptor_sha256": _sha_raw(descriptor_raw),
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha_raw(approval_raw),
    }

    second_allowance = _module().build_gpu_launch_allowance(**second)

    assert (
        first_allowance["campaign_identity_sha256"]
        == (second_allowance["campaign_identity_sha256"])
    )
    assert first_allowance["must_start_by"] != second_allowance["must_start_by"]
    assert first_allowance["descriptor_key"] != second_allowance["descriptor_key"]
    assert (
        first_allowance["descriptor_sha256"] != (second_allowance["descriptor_sha256"])
    )
    assert (
        first_allowance["allowance_body_sha256"]
        != (second_allowance["allowance_body_sha256"])
    )
    assert _module().gpu_launch_allowance_s3_key(
        run_id=RUN_ID,
        allowance_body_sha256=first_allowance["allowance_body_sha256"],
    ) != _module().gpu_launch_allowance_s3_key(
        run_id=RUN_ID,
        allowance_body_sha256=second_allowance["allowance_body_sha256"],
    )


def test_accepts_complete_empty_multi_page_s3_record_listing() -> None:
    observation = _s3_observation()
    observation["spend_records_list_pages"] = [
        {
            "page_index": 0,
            "request_continuation_token": None,
            "http_status_code": 200,
            "key_count": 0,
            "keys": [],
            "is_truncated": True,
            "next_continuation_token": "token-1",
        },
        {
            "page_index": 1,
            "request_continuation_token": "token-1",
            "http_status_code": 200,
            "key_count": 0,
            "keys": [],
            "is_truncated": False,
            "next_continuation_token": None,
        },
    ]
    inputs = _replace_s3_observation(_inputs(), observation)

    allowance = _module().build_gpu_launch_allowance(**inputs)

    assert allowance["spend_ledger_latest_marker_key"] == (
        f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json"
    )
    assert allowance["spend_ledger_legacy_key"] == (
        f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl"
    )
    assert allowance["spend_ledger_records_prefix"] == (
        f"campaigns/{RUN_ID}/runtime/spend-ledger/records/"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("record_type", "glm52_s3_spend_ledger_observation_v1"),
        ("account_id", "135792468013"),
        ("region", "us-east-1"),
        ("bucket", "foreign-valid-bucket"),
        ("run_id", "glm52-sky-foreign"),
        (
            "latest_marker_key",
            ("campaigns/glm52-sky-foreign/runtime/GPU_SPEND_LEDGER_LATEST.json"),
        ),
        (
            "legacy_ledger_key",
            "campaigns/glm52-sky-foreign/runtime/GPU_SPEND_LEDGER.jsonl",
        ),
        (
            "spend_records_prefix",
            "campaigns/glm52-sky-foreign/runtime/spend-ledger/records/",
        ),
    ],
)
def test_rejects_s3_observation_identity_or_coordinate_drift(
    field: str,
    value: object,
) -> None:
    observation = _s3_observation()
    observation[field] = value
    inputs = _replace_s3_observation(_inputs(), observation)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_rejects_s3_observation_schema_drift(change: str) -> None:
    observation = _s3_observation()
    if change == "missing":
        observation.pop("legacy_ledger_reads")
    else:
        observation["objects_absent"] = True
    inputs = _replace_s3_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="S3.*schema",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "reads_field",
    ["latest_marker_reads", "legacy_ledger_reads"],
)
@pytest.mark.parametrize(
    "replacement",
    [
        [],
        [
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "before-records-list",
            }
        ],
        [
            {
                "error_code": "NoSuchKey",
                "http_status_code": 200,
                "phase": "before-records-list",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
        ],
        [
            {
                "error_code": "AccessDenied",
                "http_status_code": 404,
                "phase": "before-records-list",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
        ],
        [
            {
                "error_code": "NoSuchKey",
                "http_status_code": True,
                "phase": "before-records-list",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
        ],
        [
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "before-records-list",
            },
        ],
        [
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "before-records-list",
                "body": "",
            },
            {
                "error_code": "NoSuchKey",
                "http_status_code": 404,
                "phase": "after-records-list",
            },
        ],
    ],
)
def test_rejects_inexact_singleton_absence_reads(
    reads_field: str,
    replacement: list[dict[str, object]],
) -> None:
    observation = _s3_observation()
    observation[reads_field] = replacement
    inputs = _replace_s3_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="absence reads",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "pages",
    [
        [],
        [
            {
                "page_index": 1,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": True,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": "foreign-token",
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 201,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": True,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 1,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": "not-a-list",
                "is_truncated": False,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": 0,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": True,
                "next_continuation_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": "unexpected-token",
            }
        ],
        [
            {
                "page_index": 0,
                "request_continuation_token": None,
                "http_status_code": 200,
                "key_count": 0,
                "keys": [],
                "is_truncated": False,
                "next_continuation_token": None,
                "unknown": True,
            }
        ],
    ],
)
def test_rejects_malformed_s3_record_listing_pages(
    pages: list[dict[str, object]],
) -> None:
    observation = _s3_observation()
    observation["spend_records_list_pages"] = pages
    inputs = _replace_s3_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="record listing",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_rejects_incomplete_s3_pagination_token_chain() -> None:
    observation = _s3_observation()
    observation["spend_records_list_pages"] = [
        {
            "page_index": 0,
            "request_continuation_token": None,
            "http_status_code": 200,
            "key_count": 0,
            "keys": [],
            "is_truncated": True,
            "next_continuation_token": "token-1",
        },
        {
            "page_index": 1,
            "request_continuation_token": "different-token",
            "http_status_code": 200,
            "key_count": 0,
            "keys": [],
            "is_truncated": False,
            "next_continuation_token": None,
        },
    ]
    inputs = _replace_s3_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="pagination",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "keys",
    [
        [
            (
                f"campaigns/{RUN_ID}/runtime/spend-ledger/records/"
                "000001-allocation_started.json"
            )
        ],
        ["campaigns/glm52-sky-foreign/runtime/spend-ledger/records/a.json"],
        [
            f"campaigns/{RUN_ID}/runtime/spend-ledger/records/b.json",
            f"campaigns/{RUN_ID}/runtime/spend-ledger/records/a.json",
        ],
        [
            f"campaigns/{RUN_ID}/runtime/spend-ledger/records/a.json",
            f"campaigns/{RUN_ID}/runtime/spend-ledger/records/a.json",
        ],
    ],
)
def test_rejects_any_nonempty_or_malformed_s3_record_inventory(
    keys: list[str],
) -> None:
    observation = _s3_observation()
    page = observation["spend_records_list_pages"][0]
    page["key_count"] = len(keys)
    page["keys"] = keys
    inputs = _replace_s3_observation(_inputs(), observation)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "caller_arn",
    [
        "arn:aws:iam::246813579024:role/keep-glm52-read-authority",
        (
            "arn:aws:sts::246813579024:assumed-role/"
            "AWSReservedSSO_AdministratorAccess_abcd/jack.mazac"
        ),
    ],
)
def test_accepts_approved_account_role_callers_and_complete_ec2_pages(
    caller_arn: str,
) -> None:
    observation = _ec2_observation()
    observation["caller_arn"] = caller_arn
    pages = [
        {
            "page_index": 0,
            "request_next_token": None,
            "http_status_code": 200,
            "reservations": [],
            "next_token": "token-1",
        },
        {
            "page_index": 1,
            "request_next_token": "token-1",
            "http_status_code": 200,
            "reservations": [],
            "next_token": None,
        },
    ]
    observation["history_pages"] = pages
    observation["active_pages"] = _json_copy(pages)
    inputs = _replace_ec2_observation(_inputs(), observation)

    allowance = _module().build_gpu_launch_allowance(**inputs)

    assert allowance["ec2_tagged_p5_history_instance_ids"] == []
    assert allowance["ec2_active_tagged_p5_instance_ids"] == []


@pytest.mark.parametrize(
    "caller_arn",
    [
        "arn:aws:iam::246813579024:role/foo/",
        "arn:aws:iam::246813579024:role//",
        "arn:aws:iam::246813579024:role/a//",
    ],
)
def test_rejects_iam_role_arns_without_a_final_role_name(
    caller_arn: str,
) -> None:
    observation = _ec2_observation()
    observation["caller_arn"] = caller_arn
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("record_type", "glm52_ec2_inventory_observation_v1"),
        ("account_id", "135792468013"),
        (
            "caller_arn",
            "arn:aws:sts::135792468013:assumed-role/Admin/jack",
        ),
        ("caller_arn", "arn:aws:iam::246813579024:user/jack.mazac"),
        ("caller_arn", "not-an-arn"),
        ("region", "us-east-1"),
        ("run_id", "glm52-sky-foreign"),
    ],
)
def test_rejects_ec2_observation_identity_drift(
    field: str,
    value: object,
) -> None:
    observation = _ec2_observation()
    observation[field] = value
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_rejects_ec2_observation_schema_drift(change: str) -> None:
    observation = _ec2_observation()
    if change == "missing":
        observation.pop("active_pages")
    else:
        observation["instance_ids"] = []
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="EC2.*schema",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    ("request_field", "mutation"),
    [
        ("history_request", "missing-filter"),
        ("history_request", "extra-filter"),
        ("history_request", "reordered"),
        ("history_request", "foreign-run"),
        ("history_request", "wrong-type"),
        ("history_request", "extra-request-field"),
        ("active_request", "missing-filter"),
        ("active_request", "extra-filter"),
        ("active_request", "reordered"),
        ("active_request", "foreign-run"),
        ("active_request", "wrong-type"),
        ("active_request", "extra-request-field"),
        ("active_request", "state-order"),
        ("active_request", "state-extra"),
    ],
)
def test_rejects_inexact_ec2_filter_requests(
    request_field: str,
    mutation: str,
) -> None:
    observation = _ec2_observation()
    request = observation[request_field]
    filters = request["filters"]
    if mutation == "missing-filter":
        filters.pop(0)
    elif mutation == "extra-filter":
        filters.append({"name": "tag:extra", "values": ["no"]})
    elif mutation == "reordered":
        filters[0], filters[1] = filters[1], filters[0]
    elif mutation == "foreign-run":
        filters[1]["values"] = ["glm52-sky-foreign"]
    elif mutation == "wrong-type":
        filters[0]["values"] = "p5.48xlarge"
    elif mutation == "extra-request-field":
        request["dry_run"] = True
    elif mutation == "state-order":
        filters[-1]["values"] = [
            "running",
            "pending",
            "shutting-down",
            "stopping",
        ]
    elif mutation == "state-extra":
        filters[-1]["values"].append("stopped")
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="EC2.*filter",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize("pages_field", ["history_pages", "active_pages"])
@pytest.mark.parametrize(
    "pages",
    [
        [],
        [
            {
                "page_index": 1,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": [],
                "next_token": None,
            }
        ],
        [
            {
                "page_index": True,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": [],
                "next_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_next_token": "foreign-token",
                "http_status_code": 200,
                "reservations": [],
                "next_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": True,
                "reservations": [],
                "next_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": 201,
                "reservations": [],
                "next_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": {},
                "next_token": None,
            }
        ],
        [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": [],
                "next_token": "unexpected-terminal-token",
            }
        ],
        [
            {
                "page_index": 0,
                "request_next_token": None,
                "http_status_code": 200,
                "reservations": [],
                "next_token": None,
                "unknown": True,
            }
        ],
    ],
)
def test_rejects_malformed_ec2_inventory_pages(
    pages_field: str,
    pages: list[dict[str, object]],
) -> None:
    observation = _ec2_observation()
    observation[pages_field] = pages
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="EC2.*(pages|token)",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize("pages_field", ["history_pages", "active_pages"])
def test_rejects_any_ec2_reservation(pages_field: str) -> None:
    observation = _ec2_observation()
    observation[pages_field][0]["reservations"] = [
        {"Instances": [{"InstanceId": "i-0123456789abcdef0"}]}
    ]
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="reservations must be empty",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize("pages_field", ["history_pages", "active_pages"])
def test_rejects_incomplete_or_repeated_ec2_token_chain(
    pages_field: str,
) -> None:
    observation = _ec2_observation()
    observation[pages_field] = [
        {
            "page_index": 0,
            "request_next_token": None,
            "http_status_code": 200,
            "reservations": [],
            "next_token": "token-1",
        },
        {
            "page_index": 1,
            "request_next_token": "token-1",
            "http_status_code": 200,
            "reservations": [],
            "next_token": "token-1",
        },
        {
            "page_index": 2,
            "request_next_token": "token-1",
            "http_status_code": 200,
            "reservations": [],
            "next_token": None,
        },
    ]
    inputs = _replace_ec2_observation(_inputs(), observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="EC2.*token",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_accepts_aware_datetime_observation_and_normalizes_to_utc() -> None:
    inputs = _inputs()
    inputs["observed_at"] = datetime(
        2026,
        7,
        26,
        10,
        45,
        tzinfo=timezone(timedelta(hours=-7)),
    )

    allowance = _module().build_gpu_launch_allowance(**inputs)

    assert allowance["observed_at"] == OBSERVED_AT
    assert allowance["not_after"] == "2026-07-26T17:46:00Z"


def test_near_deadline_caps_not_after_at_must_start_by() -> None:
    descriptor_raw, approval_raw = _authorities(
        must_start_by=datetime(2026, 7, 26, 17, 45, 30, tzinfo=UTC),
    )
    inputs = {
        **_inputs(),
        "descriptor_raw": descriptor_raw,
        "expected_descriptor_sha256": _sha_raw(descriptor_raw),
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha_raw(approval_raw),
    }

    allowance = _module().build_gpu_launch_allowance(**inputs)

    assert allowance["must_start_by"] == "2026-07-26T17:45:30Z"
    assert allowance["not_after"] == "2026-07-26T17:45:30Z"


@pytest.mark.parametrize(
    "observation_field",
    ["s3", "ec2"],
)
def test_rejects_disagreeing_raw_observation_time(
    observation_field: str,
) -> None:
    inputs = _inputs()
    if observation_field == "s3":
        observation = _s3_observation()
        observation["observed_at"] = "2026-07-26T17:45:01Z"
        inputs = _replace_s3_observation(inputs, observation)
    else:
        observation = _ec2_observation()
        observation["observed_at"] = "2026-07-26T17:45:01Z"
        inputs = _replace_ec2_observation(inputs, observation)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="observation time",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "observed_at",
    [
        "2026-07-26T17:45:00.000000Z",
        "2026-07-26T17:45:00+00:00",
        "2026-07-26 17:45:00Z",
        datetime(2026, 7, 26, 17, 45),  # noqa: DTZ001
        1_721,
    ],
)
def test_rejects_noncanonical_builder_observation_time(
    observed_at: object,
) -> None:
    inputs = _inputs()
    inputs["observed_at"] = observed_at

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="observed_at",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_rejects_fractional_raw_observation_times() -> None:
    s3_observation = _s3_observation()
    s3_observation["observed_at"] = "2026-07-26T17:45:00.500000Z"
    ec2_observation = _ec2_observation()
    ec2_observation["observed_at"] = "2026-07-26T17:45:00.500000Z"
    inputs = _replace_s3_observation(_inputs(), s3_observation)
    inputs = _replace_ec2_observation(inputs, ec2_observation)
    inputs["observed_at"] = "2026-07-26T17:45:00.500000Z"

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="whole-second",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_rejects_fractional_descriptor_deadline() -> None:
    descriptor_raw, approval_raw = _authorities(
        must_start_by=datetime(
            2026,
            7,
            26,
            18,
            45,
            0,
            500_000,
            tzinfo=UTC,
        ),
    )
    inputs = {
        **_inputs(),
        "descriptor_raw": descriptor_raw,
        "expected_descriptor_sha256": _sha_raw(descriptor_raw),
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha_raw(approval_raw),
    }

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="must_start_by",
    ):
        _module().build_gpu_launch_allowance(**inputs)


@pytest.mark.parametrize(
    "must_start_by",
    [
        datetime(2026, 7, 26, 17, 45, tzinfo=UTC),
        datetime(2026, 7, 26, 17, 44, 59, tzinfo=UTC),
        datetime(2026, 7, 27, 5, 45, 1, tzinfo=UTC),
    ],
)
def test_rejects_expired_or_overlong_descriptor_window(
    must_start_by: datetime,
) -> None:
    descriptor_raw, approval_raw = _authorities(
        must_start_by=must_start_by,
    )
    inputs = {
        **_inputs(),
        "descriptor_raw": descriptor_raw,
        "expected_descriptor_sha256": _sha_raw(descriptor_raw),
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha_raw(approval_raw),
    }

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="must_start",
    ):
        _module().build_gpu_launch_allowance(**inputs)


def test_validator_accepts_exact_allowance_without_consulting_clock() -> None:
    allowance = _valid_allowance()

    assert _module().validate_gpu_launch_allowance(allowance) == allowance
    assert (
        _module().validate_gpu_launch_allowance(
            allowance,
            now=OBSERVED_AT,
        )
        == allowance
    )
    assert (
        _module().validate_gpu_launch_allowance(
            allowance,
            now="2026-07-26T17:45:59Z",
        )
        == allowance
    )


def test_validator_rejects_unrehashed_mutation() -> None:
    allowance = _valid_allowance()
    allowance["managed_mode"] = "qualification"

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="body SHA-256",
    ):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize("change", ["missing", "extra", "snapshot-mixed"])
def test_validator_rejects_schema_drift(change: str) -> None:
    allowance = _valid_allowance()
    if change == "missing":
        allowance.pop("descriptor_key")
    elif change == "extra":
        allowance["submission_id"] = "first"
    else:
        allowance["gpu_spend_ledger_record_count"] = 0
    allowance = _rehash_allowance(allowance)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="schema mismatch",
    ):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("record_type", "glm52_gpu_spend_snapshot_v1"),
        ("managed_mode", "qualification"),
        ("sky_job_name", "foreign-cache-seed"),
        ("account_id", "135792468013"),
        ("region", "us-east-1"),
        ("instance_type", "p4d.24xlarge"),
        ("instance_count", 2),
        ("use_spot", True),
        ("approved_gpu_runtime_seconds", 86_399),
        ("approved_gpu_cost_usd", 1_320.95),
        ("hourly_cost_usd", 55.05),
        ("consumed_gpu_seconds", 1),
        ("remaining_gpu_seconds", 86_399),
        ("consumed_gpu_cost_usd", 0.01),
        ("remaining_gpu_cost_usd", 1_320.95),
        ("cache_seed_max_seconds", 21_599),
        ("cache_seed_max_cost_usd", 330.23),
        ("allowed_gpu_seconds", 21_599),
        ("allowed_gpu_cost_usd", 330.23),
        ("open_allocation_count", 1),
        ("ec2_tagged_p5_history_instance_ids", ["i-0123456789abcdef0"]),
        ("ec2_active_tagged_p5_instance_ids", ["i-0123456789abcdef0"]),
    ],
)
def test_validator_rejects_coherently_rehashed_fixed_or_math_drift(
    field: str,
    value: object,
) -> None:
    allowance = _valid_allowance()
    allowance[field] = value
    allowance = _rehash_allowance(allowance)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("instance_count", True),
        ("consumed_gpu_seconds", False),
        ("allowed_gpu_seconds", "21600"),
        ("approved_gpu_cost_usd", 1320),
        ("hourly_cost_usd", "55.04"),
        ("consumed_gpu_cost_usd", 0),
        ("remaining_gpu_cost_usd", Decimal("1320.96")),
        ("allowed_gpu_cost_usd", float("nan")),
        ("cache_seed_max_cost_usd", float("inf")),
    ],
)
def test_validator_rejects_wrong_numeric_types_or_nonfinite_values(
    field: str,
    value: object,
) -> None:
    allowance = _valid_allowance()
    allowance[field] = value
    try:
        allowance = _rehash_allowance(allowance)
    except (TypeError, ValueError):
        allowance["allowance_body_sha256"] = "f" * 64

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize(
    "field",
    [
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "cache_seed_max_cost_usd",
        "allowed_gpu_cost_usd",
    ],
)
def test_validator_rejects_negative_zero_money(field: str) -> None:
    allowance = _valid_allowance()
    allowance[field] = -0.0
    allowance = _rehash_allowance(allowance)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="negative zero|money",
    ):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize(
    "field",
    [
        "campaign_identity_sha256",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "s3_spend_ledger_absence_observation_sha256",
        "ec2_tagged_p5_zero_inventory_observation_sha256",
    ],
)
def test_validator_rejects_malformed_external_digest(field: str) -> None:
    allowance = _valid_allowance()
    allowance[field] = "A" * 64
    allowance = _rehash_allowance(allowance)

    with pytest.raises(
        _module().GpuLaunchAllowanceError,
        match="lowercase SHA-256",
    ):
        _module().validate_gpu_launch_allowance(allowance)


def test_validator_allows_coherently_rehashed_external_pins() -> None:
    allowance = _valid_allowance()
    allowance["descriptor_sha256"] = "a" * 64
    allowance["approval_sha256"] = "b" * 64
    allowance["s3_spend_ledger_absence_observation_sha256"] = "c" * 64
    allowance["ec2_tagged_p5_zero_inventory_observation_sha256"] = "d" * 64
    allowance = _rehash_allowance(allowance)

    assert _module().validate_gpu_launch_allowance(allowance) == allowance


def test_validator_allows_different_safe_attempt_coordinate() -> None:
    allowance = _valid_allowance()
    allowance["descriptor_key"] = (
        f"campaigns/{RUN_ID}/submissions/seed/campaign-descriptor-v2.json"
    )
    allowance = _rehash_allowance(allowance)

    assert _module().validate_gpu_launch_allowance(allowance) == allowance


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "descriptor_key",
            (
                "campaigns/glm52-sky-foreign/submissions/first/"
                "campaign-descriptor-v2.json"
            ),
        ),
        (
            "descriptor_key",
            (
                f"campaigns/{RUN_ID}/submissions/nested/attempt/"
                "campaign-descriptor-v2.json"
            ),
        ),
        (
            "approval_key",
            ("campaigns/glm52-sky-foreign/authorities/GPU_SPEND_APPROVAL.json"),
        ),
        (
            "approval_key",
            f"campaigns/{RUN_ID}/authorities/bad*key.json",
        ),
        (
            "spend_ledger_latest_marker_key",
            f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER.jsonl",
        ),
        (
            "spend_ledger_legacy_key",
            f"campaigns/{RUN_ID}/runtime/GPU_SPEND_LEDGER_LATEST.json",
        ),
        (
            "spend_ledger_records_prefix",
            f"campaigns/{RUN_ID}/runtime/spend-ledger/records",
        ),
        ("bucket", "bad..bucket"),
        ("run_id", "../unsafe"),
    ],
)
def test_validator_rejects_unsafe_or_cross_run_coordinates(
    field: str,
    value: object,
) -> None:
    allowance = _valid_allowance()
    allowance[field] = value
    allowance = _rehash_allowance(allowance)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_at", "2026-07-26T17:45:00.000000Z"),
        ("must_start_by", "2026-07-26T17:45:00Z"),
        ("must_start_by", "2026-07-27T05:45:01Z"),
        ("not_after", "2026-07-26T17:46:01Z"),
        ("not_after", "2026-07-26T17:45:59Z"),
    ],
)
def test_validator_rejects_invalid_relative_time_math(
    field: str,
    value: str,
) -> None:
    allowance = _valid_allowance()
    allowance[field] = value
    allowance = _rehash_allowance(allowance)

    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().validate_gpu_launch_allowance(allowance)


@pytest.mark.parametrize(
    "now",
    [
        "2026-07-26T17:44:59Z",
        "2026-07-26T17:46:00Z",
        "2026-07-26T17:46:01Z",
        "2026-07-26T17:45:00.500000Z",
        datetime(2026, 7, 26, 17, 45),  # noqa: DTZ001
    ],
)
def test_validator_rejects_now_outside_half_open_freshness_window(
    now: object,
) -> None:
    with pytest.raises(_module().GpuLaunchAllowanceError):
        _module().validate_gpu_launch_allowance(
            _valid_allowance(),
            now=now,
        )


def test_content_addressed_key_is_exact_and_strict() -> None:
    digest = "a" * 64

    assert _module().gpu_launch_allowance_s3_key(
        run_id=RUN_ID,
        allowance_body_sha256=digest,
    ) == (
        f"campaigns/{RUN_ID}/submissions/cache-seed/launch-allowances/"
        f"{digest}/GLM52_GPU_LAUNCH_ALLOWANCE.json"
    )

    for run_id, body_sha in [
        ("../unsafe", digest),
        ("", digest),
        (RUN_ID, "A" * 64),
        (RUN_ID, "a" * 63),
    ]:
        with pytest.raises(_module().GpuLaunchAllowanceError):
            _module().gpu_launch_allowance_s3_key(
                run_id=run_id,
                allowance_body_sha256=body_sha,
            )


def test_public_exports_are_exact() -> None:
    assert set(_module().__all__) == {
        "CACHE_SEED_MAX_COST_USD",
        "CACHE_SEED_MAX_SECONDS",
        "DIGEST_FIELD",
        "GpuLaunchAllowanceError",
        "MANAGED_MODE",
        "MAX_ALLOWANCE_AGE_SECONDS",
        "MAX_MUST_START_WINDOW_SECONDS",
        "RECORD_TYPE",
        "SCHEMA_VERSION",
        "build_gpu_launch_allowance",
        "gpu_launch_allowance_s3_key",
        "validate_gpu_launch_allowance",
    }


def test_post_seed_snapshot_v1_rejects_zero_record_chain() -> None:
    from mlx_vq.quality.glm52_gpu_spend_snapshot import (
        GpuSpendSnapshotError,
        validate_gpu_spend_snapshot,
    )

    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": "1" * 64,
        "descriptor_sha256": "2" * 64,
        "descriptor_body_sha256": "3" * 64,
        "approval_sha256": "4" * 64,
        "approval_body_sha256": "5" * 64,
        "gpu_spend_ledger_latest_sha256": "6" * 64,
        "gpu_spend_ledger_latest_body_sha256": "7" * 64,
        "gpu_spend_ledger_genesis_sha256": "8" * 64,
        "gpu_spend_ledger_record_count": 0,
        "gpu_spend_ledger_tip_record_sha256": "9" * 64,
        "gpu_spend_ledger_file_sha256": "a" * 64,
        "ec2_allocation_history_sha256": "b" * 64,
        "ec2_allocation_instance_ids": [],
        "observed_at": OBSERVED_AT,
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "hourly_cost_usd": 55.04,
        "consumed_gpu_seconds": 0,
        "remaining_gpu_seconds": 86_400,
        "consumed_gpu_cost_usd": 0.0,
        "remaining_gpu_cost_usd": 1_320.96,
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "open_allocation_count": 0,
    }
    snapshot = {
        **body,
        "snapshot_body_sha256": _sha_raw(_canonical(body)),
    }

    with pytest.raises(
        GpuSpendSnapshotError,
        match="integer >= 2",
    ):
        validate_gpu_spend_snapshot(snapshot)
