from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from mlx_vq.quality.glm52_campaign_watchdog import (
    build_skypilot_job_status,
    build_skypilot_submission_marker,
)
from mlx_vq.quality.glm52_qualification_cache_seed import (
    QualificationCacheSeedPins,
    authenticate_qualification_cache_seed,
    validate_qualification_cache_seed_accepted,
)
from mlx_vq.quality.glm52_sky_campaign import (
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_RUNTIME_SECONDS,
    APPROVED_HOURLY_COST_USD,
    GpuSpendLedger,
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import build_must_start_job_binding
from mlx_vq.quality.glm52_teich_training_cache import (
    MANIFEST_FILENAME,
    READY_FILENAME,
    finalize_glm52_teich_teacher_cache,
)

RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
INSTANCE_ID = "i-0123456789abcdef0"
PROMPT_ID = "teich_claude_agent-a3622521df8a9137d"
JOB_NAME = f"{RUN_ID}-cache-seed"
NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _json_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FakeAwsError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.head_overrides: dict[str, dict[str, object]] = {}
        self.errors: dict[tuple[str, str], str] = {}
        self.multipart_uploads: list[str] = []
        self.replace_after_get: dict[str, bytes] = {}
        self.get_counts: dict[str, int] = {}
        self.reads: list[tuple[str, str]] = []
        self.mutations: list[tuple[str, str]] = []

    def _maybe_error(self, operation: str, key: str) -> None:
        code = self.errors.get((operation, key))
        if code is not None:
            raise FakeAwsError(code)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == BUCKET
        self.reads.append(("get_object", Key))
        self._maybe_error("get_object", Key)
        if Key not in self.objects:
            raise FakeAwsError("NoSuchKey")
        raw = self.objects[Key]
        self.get_counts[Key] = self.get_counts.get(Key, 0) + 1
        if self.get_counts[Key] == 1 and Key in self.replace_after_get:
            self.objects[Key] = self.replace_after_get[Key]
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "ETag": f'"{_sha(raw)[:32]}"',
            "VersionId": "fixture-version",
        }

    def head_object(
        self,
        *,
        Bucket: str,
        Key: str,
        ChecksumMode: str,
    ) -> dict[str, object]:
        assert Bucket == BUCKET
        assert ChecksumMode == "ENABLED"
        self.reads.append(("head_object", Key))
        self._maybe_error("head_object", Key)
        if Key not in self.objects:
            raise FakeAwsError("404")
        raw = self.objects[Key]
        response: dict[str, object] = {
            "ContentLength": len(raw),
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(raw).digest()).decode(),
            "ChecksumType": "FULL_OBJECT",
            "ETag": f'"{_sha(raw)[:32]}"',
            "VersionId": "fixture-version",
            "Metadata": {"glm52-run-id": RUN_ID},
        }
        response.update(self.head_overrides.get(Key, {}))
        return response

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        ContinuationToken: str | None = None,
    ) -> dict[str, object]:
        assert Bucket == BUCKET
        assert ContinuationToken is None
        self.reads.append(("list_objects_v2", Prefix))
        self._maybe_error("list_objects_v2", Prefix)
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        return {
            "IsTruncated": False,
            "KeyCount": len(keys),
            "Contents": [{"Key": key, "Size": len(self.objects[key])} for key in keys],
        }

    def list_multipart_uploads(
        self,
        *,
        Bucket: str,
        Prefix: str,
        KeyMarker: str | None = None,
        UploadIdMarker: str | None = None,
    ) -> dict[str, object]:
        assert Bucket == BUCKET
        assert KeyMarker is None
        assert UploadIdMarker is None
        self.reads.append(("list_multipart_uploads", Prefix))
        self._maybe_error("list_multipart_uploads", Prefix)
        return {
            "IsTruncated": False,
            "Uploads": [
                {"Key": key, "UploadId": "upload"}
                for key in self.multipart_uploads
                if key.startswith(Prefix)
            ],
        }

    def put_object(self, **kwargs: object) -> None:
        self.mutations.append(("put_object", str(kwargs.get("Key"))))
        raise AssertionError("acceptance must be read-only")


def _write_json(path: Path, value: object) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _cache_fixture(
    root: Path,
    *,
    target_token_id: int = 7,
    foreign_fields: bool = False,
) -> tuple[Path, Path, dict[str, bytes]]:
    prompt_row: dict[str, object] = {
        "prompt_id": PROMPT_ID,
        "positions": [0],
        "target_token_ids": [target_token_id],
        "token_ids_sha256": "a" * 64,
        "provenance": {
            "provider": "claude",
            "source_session_id": "session-a3622521df8a9137d",
        },
    }
    if foreign_fields:
        prompt_row["foreign_row"] = "self-consistent but unauthorized"
    prompt_pack = {
        "schema_version": 1,
        "record_type": "qualification_prompt_pack_fixture",
        "prompt_row_count": 1,
        "supervised_tokens": 1,
        "prompt_rows": [prompt_row],
    }
    if foreign_fields:
        prompt_pack["foreign_top_level"] = True
    frozen_pack = {
        "schema_version": 1,
        "record_type": "frozen_prompt_pack_fixture",
        "prompt_rows": [],
    }
    prompt_path = root / "prompt-pack.json"
    frozen_path = root / "frozen-66.json"
    _write_json(prompt_path, prompt_pack)
    _write_json(frozen_path, frozen_pack)
    captures = root / "captures"
    captures.mkdir()
    np.savez(
        captures / f"{PROMPT_ID}.npz",
        positions=np.asarray([0], dtype=np.int32),
        target_token_ids=np.asarray([target_token_id], dtype=np.int32),
        topk_logit_ids=np.arange(2048, dtype=np.int32).reshape(1, 2048),
        topk_logit_values=np.zeros((1, 2048), dtype=np.float16),
        logsumexp=np.asarray([0.0], dtype=np.float32),
        tail_mass=np.asarray([0.5], dtype=np.float16),
        layer_77_hidden_probe=np.zeros((1, 6144), dtype=np.float16),
        router_top8_expert_ids=np.zeros((8, 1, 8), dtype=np.int32),
        router_top8_normalized_weights=np.full((8, 1, 8), 0.125, dtype=np.float16),
    )
    cache_dir = root / "cache"
    finalize_glm52_teich_teacher_cache(
        capture_dir=captures,
        prompt_pack_path=prompt_path,
        frozen_prompt_pack_path=frozen_path,
        cache_dir=cache_dir,
        expected_session_count=1,
        expected_supervised_positions=1,
        top_k=2048,
        hidden_size=6144,
    )
    files = {
        path.relative_to(cache_dir).as_posix(): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    }
    files["prompt-pack.json"] = prompt_path.read_bytes()
    return cache_dir, frozen_path, files


def _descriptor_artifacts(
    *,
    frozen_sha: str,
    frozen_key: str,
    teich_sha: str,
    teich_key: str,
) -> dict[str, str]:
    return {
        "source_snapshot_prefix": "source-snapshot/",
        "source_snapshot_sha256": "1" * 64,
        "non_vq_prefix": "non-vq-package/",
        "non_vq_package_sha256": "2" * 64,
        "teich_pack_key": teich_key,
        "teich_pack_sha256": teich_sha,
        "frozen_prompt_pack_key": frozen_key,
        "frozen_prompt_pack_sha256": frozen_sha,
        "training_baseline_prefix": "training-baseline/",
        "training_baseline_sha256": "4" * 64,
        "training_config_key": f"campaigns/{RUN_ID}/authorities/training.json",
        "training_config_sha256": "5" * 64,
        "artifact_inventory_key": (
            f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'6' * 64}.json"
        ),
        "artifact_inventory_sha256": "6" * 64,
        "qualification_cache_prefix": "qualification-cache/",
        "qualification_cache_manifest_sha256": "0" * 64,
    }


def _accepted_fixture(
    tmp_path: Path,
) -> tuple[
    FakeS3,
    QualificationCacheSeedPins,
    dict[str, object],
    dict[str, str],
]:
    cache_dir, frozen_path, cache_files = _cache_fixture(tmp_path)
    manifest_raw = (cache_dir / MANIFEST_FILENAME).read_bytes()
    ready_raw = (cache_dir / READY_FILENAME).read_bytes()
    manifest_sha = _sha(manifest_raw)
    prefix = f"qualification-cache/seeds/{RUN_ID}/{manifest_sha}/"
    frozen_key = "quality/frozen-66.json"
    frozen_raw = frozen_path.read_bytes()
    teich_key = "teich-pack/glm52-coding-agent.json"
    teich_raw = cache_files["prompt-pack.json"]

    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 15, 1, 0, tzinfo=timezone.utc),
        slack_permalink=None,
    )
    approval_raw = _json_bytes(approval)
    approval_sha = _sha(approval_raw)
    repo_raw = b"exact immutable seed repository tar fixture\n"
    repo_sha = _sha(repo_raw)
    descriptor_key = f"campaigns/{RUN_ID}/submissions/seed/campaign-descriptor-v2.json"
    approval_key = f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"
    repo_key = f"campaigns/{RUN_ID}/repository/keep-{repo_sha}.tar.gz"
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=NOW + timedelta(hours=2),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-skypilot-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=repo_key,
        repo_tar_sha256=repo_sha,
        campaign_descriptor_key=descriptor_key,
        approval_key=approval_key,
        approval_sha256=approval_sha,
        artifacts=_descriptor_artifacts(
            frozen_sha=_sha(frozen_raw),
            frozen_key=frozen_key,
            teich_sha=_sha(teich_raw),
            teich_key=teich_key,
        ),
    )
    descriptor_raw = _json_bytes(descriptor)
    descriptor_file_sha = _sha(descriptor_raw)
    submission = build_skypilot_submission_marker(
        run_id=RUN_ID,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submitted_at=NOW - timedelta(hours=1),
        must_start_by=NOW + timedelta(hours=2),
        sky_job_name=JOB_NAME,
    )
    submission_raw = _json_bytes(submission)
    submission_key = (
        f"campaigns/{RUN_ID}/monitor/submission-locks/"
        f"{descriptor_file_sha}-cache-seed.json"
    )
    submission_alias_key = (
        f"campaigns/{RUN_ID}/monitor/QUALIFICATION_CACHE_SEED_SUBMITTED.json"
    )
    job_binding = build_must_start_job_binding(
        run_id=RUN_ID,
        managed_mode="cache-seed",
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(submission["submission_body_sha256"]),
        sky_job_name=JOB_NAME,
        must_start_by=NOW + timedelta(hours=2),
        descriptor_key=descriptor_key,
        descriptor_file_sha256=descriptor_file_sha,
        submission_key=submission_key,
        submission_submitted_at=NOW - timedelta(hours=1),
        target_job_id=3,
        workspace="default",
        controller_instance_id="i-0fedcba9876543210",
        controller_instance_type="c6a.xlarge",
        controller_profile_arn=(
            "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
        ),
        controller_cluster_name="sky-jobs-controller-fixture",
        observation_body_sha256="b" * 64,
        bound_at=NOW - timedelta(minutes=30),
    )
    job_binding_raw = _canonical(job_binding)
    job_binding_key = (
        f"campaigns/{RUN_ID}/monitor/must-start/cache-seed/"
        f"{submission['submission_body_sha256']}/JOB_BINDING.json"
    )

    ledger_path = tmp_path / "GPU_SPEND_LEDGER.jsonl"
    ledger = GpuSpendLedger(
        ledger_path,
        run_id=RUN_ID,
        approval_sha256=approval_sha,
        approved_gpu_runtime_seconds=APPROVED_GPU_RUNTIME_SECONDS,
        approved_gpu_cost_usd=APPROVED_GPU_COST_USD,
        hourly_cost_usd=APPROVED_HOURLY_COST_USD,
    )
    launched_at = NOW - timedelta(minutes=10)
    observed_at = launched_at
    started = ledger.start_allocation(
        job_id=JOB_NAME,
        instance_id=INSTANCE_ID,
        launched_at=launched_at,
    )
    start_ledger_sha = _sha(ledger_path.read_bytes())
    remaining_at_start = int(ledger.remaining_gpu_seconds(now=observed_at))
    allocation_body: dict[str, object] = {
        "record_type": "glm52_gpu_runtime_allocation_v1",
        "run_id": RUN_ID,
        "job_id": JOB_NAME,
        "instance_id": INSTANCE_ID,
        "launched_at": launched_at.isoformat().replace("+00:00", "Z"),
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "execution_deadline": (observed_at + timedelta(seconds=remaining_at_start))
        .isoformat()
        .replace("+00:00", "Z"),
        "approval_sha256": approval_sha,
        "gpu_spend_authority_sha256": ledger.genesis_sha256,
        "gpu_spend_record_sha256": started["record_sha256"],
        "gpu_spend_ledger_sha256": start_ledger_sha,
        "remaining_gpu_seconds": remaining_at_start,
        "estimated_gpu_cost_usd": ledger.estimated_gpu_cost_usd(now=observed_at),
    }
    allocation = {
        **allocation_body,
        "allocation_body_sha256": _sha(_canonical(allocation_body)),
    }
    ended_at = NOW
    ended = ledger.end_allocation(
        instance_id=INSTANCE_ID,
        ended_at=ended_at,
    )
    ledger_raw = ledger_path.read_bytes()
    status = {
        "record_type": "glm52_gpu_spend_status_v1",
        "run_id": RUN_ID,
        "instance_id": INSTANCE_ID,
        "ended_at": ended_at.isoformat().replace("+00:00", "Z"),
        "gpu_spend_record_sha256": ended["record_sha256"],
        "gpu_spend_ledger_sha256": _sha(ledger_raw),
        "consumed_gpu_seconds": int(ledger.consumed_gpu_seconds(now=ended_at)),
        "remaining_gpu_seconds": int(ledger.remaining_gpu_seconds(now=ended_at)),
        "estimated_gpu_cost_usd": ledger.estimated_gpu_cost_usd(now=ended_at),
    }
    job_status = build_skypilot_job_status(
        run_id=RUN_ID,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        sky_job_name=JOB_NAME,
        status="SUCCEEDED",
        instance_id=INSTANCE_ID,
        observed_at=NOW + timedelta(seconds=1),
    )
    marker_body = {
        "schema_version": 1,
        "record_type": "glm52_qualification_cache_seed_ready_v1",
        "run_id": RUN_ID,
        "qualification_cache_prefix": prefix,
        "qualification_cache_manifest_sha256": manifest_sha,
        "teacher_cache_ready_sha256": _sha(ready_raw),
    }
    marker = {
        **marker_body,
        "ready_body_sha256": _sha(_canonical(marker_body)),
    }
    marker_key = (
        f"campaigns/{RUN_ID}/qualification-cache-seed/"
        "QUALIFICATION_CACHE_SEED_READY.json"
    )
    runtime_prefix = f"campaigns/{RUN_ID}/runtime"
    objects = {
        descriptor_key: descriptor_raw,
        approval_key: approval_raw,
        repo_key: repo_raw,
        frozen_key: frozen_raw,
        teich_key: teich_raw,
        submission_key: submission_raw,
        submission_alias_key: submission_raw,
        job_binding_key: job_binding_raw,
        marker_key: _json_bytes(marker),
        f"{runtime_prefix}/GPU_RUNTIME_ALLOCATION.json": _json_bytes(allocation),
        f"{runtime_prefix}/GPU_SPEND_LEDGER.jsonl": ledger_raw,
        f"{runtime_prefix}/GPU_SPEND_STATUS.json": _json_bytes(status),
        f"campaigns/{RUN_ID}/monitor/SKY_JOB_STATUS.json": _json_bytes(job_status),
    }
    for relative, raw in cache_files.items():
        objects[f"{prefix}{relative}"] = raw
    pins = QualificationCacheSeedPins(
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        run_id=RUN_ID,
        managed_mode="cache-seed",
        target_job_id=3,
        descriptor_key=descriptor_key,
        descriptor_file_sha256=descriptor_file_sha,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        campaign_identity_sha256=str(descriptor["campaign_identity_sha256"]),
        repo_tar_sha256=repo_sha,
        approval_sha256=approval_sha,
        submission_key=submission_key,
        submission_file_sha256=_sha(submission_raw),
        submission_body_sha256=str(submission["submission_body_sha256"]),
        ready_key=marker_key,
    )
    keys = {
        "prefix": prefix,
        "manifest": f"{prefix}{MANIFEST_FILENAME}",
        "ready": f"{prefix}{READY_FILENAME}",
        "submission_alias": submission_alias_key,
        "job_binding": job_binding_key,
        "teich": teich_key,
        "ledger": f"{runtime_prefix}/GPU_SPEND_LEDGER.jsonl",
        "allocation": f"{runtime_prefix}/GPU_RUNTIME_ALLOCATION.json",
        "spend_status": f"{runtime_prefix}/GPU_SPEND_STATUS.json",
    }
    client = FakeS3(objects)
    accepted = authenticate_qualification_cache_seed(
        s3_client=client,
        pins=pins,
        work_dir=tmp_path / "isolated-acceptance",
        accepted_at=NOW + timedelta(minutes=1),
    )
    return client, pins, accepted, keys


def test_acceptance_binds_external_authorities_spend_tip_and_one_row_cache(
    tmp_path: Path,
) -> None:
    client, pins, accepted, keys = _accepted_fixture(tmp_path)
    job_binding = json.loads(client.objects[keys["job_binding"]])

    assert accepted == validate_qualification_cache_seed_accepted(accepted)
    assert accepted["seed_authority"] == {
        "account_id": pins.account_id,
        "region": pins.region,
        "bucket": pins.bucket,
        "run_id": RUN_ID,
        "managed_mode": "cache-seed",
        "target_job_id": 3,
        "sky_job_name": JOB_NAME,
        "descriptor_key": pins.descriptor_key,
        "descriptor_file_sha256": pins.descriptor_file_sha256,
        "descriptor_body_sha256": pins.descriptor_body_sha256,
        "campaign_identity_sha256": pins.campaign_identity_sha256,
        "repo_tar_key": (
            f"campaigns/{RUN_ID}/repository/keep-{pins.repo_tar_sha256}.tar.gz"
        ),
        "repo_tar_sha256": pins.repo_tar_sha256,
        "approval_key": (f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"),
        "approval_sha256": pins.approval_sha256,
        "submission_key": pins.submission_key,
        "submission_file_sha256": pins.submission_file_sha256,
        "submission_body_sha256": pins.submission_body_sha256,
        "submission_alias_key": keys["submission_alias"],
        "job_binding_key": keys["job_binding"],
        "job_binding": job_binding,
        "job_binding_file_sha256": _sha(client.objects[keys["job_binding"]]),
        "job_binding_body_sha256": job_binding["job_binding_body_sha256"],
        "seed_ready_key": pins.ready_key,
    }
    assert accepted["cache_audit"]["cache_prefix"] == keys["prefix"]
    assert accepted["cache_audit"]["session_count"] == 1
    assert accepted["cache_audit"]["prompt_id"] == PROMPT_ID
    assert accepted["cache_audit"]["top_k"] == 2048
    assert accepted["cache_audit"]["hidden_size"] == 6144
    assert accepted["cache_audit"]["semantic_audit_pass"] is True
    assert accepted["cache_audit"]["teich_pack_key"] == keys["teich"]
    assert accepted["cache_audit"]["teich_pack_file_sha256"] == _sha(
        client.objects[keys["teich"]]
    )
    inventory = accepted["cache_object_inventory"]
    assert [item["key"] for item in inventory] == sorted(
        key for key in client.objects if key.startswith(keys["prefix"])
    )
    assert len(inventory) == 4
    assert all(item["checksum_type"] == "FULL_OBJECT" for item in inventory)
    assert accepted["spend_closure"]["instance_id"] == INSTANCE_ID
    assert accepted["spend_closure"]["sky_job_name"] == JOB_NAME
    assert accepted["spend_closure"]["ledger_tip_event"] == "allocation_ended"
    assert accepted["spend_closure"]["submission_submitted_at"] == (
        NOW - timedelta(hours=1)
    ).isoformat().replace("+00:00", "Z")
    assert accepted["spend_closure"]["must_start_by"] == (
        NOW + timedelta(hours=2)
    ).isoformat().replace("+00:00", "Z")
    assert accepted["spend_closure"]["job_observed_at"] == (
        NOW + timedelta(seconds=1)
    ).isoformat().replace("+00:00", "Z")
    assert client.mutations == []


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("extra_object", "exact cache object inventory"),
        ("missing_object", "exact cache object inventory"),
        ("missing_checksum", "FULL_OBJECT SHA-256"),
        ("multipart_etag", "multipart"),
        ("active_multipart", "unfinished multipart"),
        ("alias_drift", "mutable submission alias drift"),
        ("foreign_marker", "seed ready schema"),
        ("open_allocation", "ledger tip"),
        ("foreign_metadata", "foreign-run"),
        ("extra_metadata", "exact S3 metadata"),
        ("unknown_ledger", "ledger record schema"),
        ("late_alias_drift", "mutable submission alias drift"),
        ("late_cache_drift", "changed during acceptance"),
    ],
)
def test_acceptance_rejects_foreign_drifted_or_incomplete_artifacts(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    client, pins, _accepted, keys = _accepted_fixture(tmp_path / "baseline")
    work = tmp_path / mutation
    if mutation == "extra_object":
        client.objects[f"{keys['prefix']}foreign.bin"] = b"foreign"
    elif mutation == "missing_object":
        del client.objects[keys["ready"]]
    elif mutation == "missing_checksum":
        client.head_overrides[keys["manifest"]] = {"ChecksumSHA256": None}
    elif mutation == "multipart_etag":
        client.head_overrides[keys["manifest"]] = {"ETag": '"etag-2"'}
    elif mutation == "active_multipart":
        client.multipart_uploads.append(f"{keys['prefix']}partial")
    elif mutation == "alias_drift":
        client.objects[keys["submission_alias"]] = b"{}\n"
    elif mutation == "foreign_marker":
        marker = json.loads(client.objects[pins.ready_key])
        marker["foreign"] = True
        body = dict(marker)
        body.pop("ready_body_sha256")
        marker["ready_body_sha256"] = _sha(_canonical(body))
        client.objects[pins.ready_key] = _json_bytes(marker)
    elif mutation == "open_allocation":
        lines = client.objects[keys["ledger"]].splitlines(keepends=True)
        client.objects[keys["ledger"]] = lines[0]
    elif mutation == "foreign_metadata":
        client.head_overrides[keys["manifest"]] = {
            "Metadata": {"glm52-run-id": "foreign-run"}
        }
    elif mutation == "extra_metadata":
        client.head_overrides[keys["manifest"]] = {
            "Metadata": {
                "glm52-run-id": RUN_ID,
                "foreign": "self-consistent but unauthorized",
            }
        }
    elif mutation == "unknown_ledger":
        lines = [
            json.loads(line) for line in client.objects[keys["ledger"]].splitlines()
        ]
        tip = lines[-1]
        tip["foreign"] = True
        tip_body = dict(tip)
        tip_body.pop("record_sha256")
        tip["record_sha256"] = _sha(_canonical(tip_body))
        ledger_raw = b"".join(_json_bytes(line) for line in lines)
        client.objects[keys["ledger"]] = ledger_raw
        status = json.loads(client.objects[keys["spend_status"]])
        status["gpu_spend_record_sha256"] = tip["record_sha256"]
        status["gpu_spend_ledger_sha256"] = _sha(ledger_raw)
        client.objects[keys["spend_status"]] = _json_bytes(status)
    elif mutation == "late_alias_drift":
        client.get_counts.clear()
        client.replace_after_get[keys["submission_alias"]] = b"{}\n"
    elif mutation == "late_cache_drift":
        client.get_counts.clear()
        client.replace_after_get[keys["manifest"]] = (
            client.objects[keys["manifest"]] + b" "
        )
    with pytest.raises(ValueError, match=error):
        authenticate_qualification_cache_seed(
            s3_client=client,
            pins=pins,
            work_dir=work,
            accepted_at=NOW + timedelta(minutes=1),
        )
    assert client.mutations == []


def test_acceptance_rejects_non404_transport_errors_and_bad_external_pins(
    tmp_path: Path,
) -> None:
    client, pins, _accepted, _keys = _accepted_fixture(tmp_path / "baseline")
    client.errors[("get_object", pins.descriptor_key)] = "AccessDenied"
    with pytest.raises(ValueError, match="AccessDenied"):
        authenticate_qualification_cache_seed(
            s3_client=client,
            pins=pins,
            work_dir=tmp_path / "denied",
            accepted_at=NOW,
        )

    client.errors.clear()
    with pytest.raises(ValueError, match="descriptor file"):
        authenticate_qualification_cache_seed(
            s3_client=client,
            pins=replace(pins, descriptor_file_sha256="f" * 64),
            work_dir=tmp_path / "wrong-pin",
            accepted_at=NOW,
        )


def test_acceptance_rejects_self_consistent_cache_not_derived_from_teich_authority(
    tmp_path: Path,
) -> None:
    client, pins, _accepted, keys = _accepted_fixture(tmp_path / "baseline")
    forged_dir, _frozen_path, forged_files = _cache_fixture(
        tmp_path / "forged",
        target_token_id=999,
        foreign_fields=True,
    )
    forged_manifest_raw = (forged_dir / MANIFEST_FILENAME).read_bytes()
    forged_manifest_sha = _sha(forged_manifest_raw)
    forged_ready_raw = (forged_dir / READY_FILENAME).read_bytes()
    forged_prefix = f"qualification-cache/seeds/{RUN_ID}/{forged_manifest_sha}/"
    for key in tuple(client.objects):
        if key.startswith(keys["prefix"]):
            del client.objects[key]
    for relative, raw in forged_files.items():
        client.objects[f"{forged_prefix}{relative}"] = raw

    marker_body = {
        "schema_version": 1,
        "record_type": "glm52_qualification_cache_seed_ready_v1",
        "run_id": RUN_ID,
        "qualification_cache_prefix": forged_prefix,
        "qualification_cache_manifest_sha256": forged_manifest_sha,
        "teacher_cache_ready_sha256": _sha(forged_ready_raw),
    }
    client.objects[pins.ready_key] = _json_bytes(
        {
            **marker_body,
            "ready_body_sha256": _sha(_canonical(marker_body)),
        }
    )

    with pytest.raises(ValueError, match="Teich prompt authority"):
        authenticate_qualification_cache_seed(
            s3_client=client,
            pins=pins,
            work_dir=tmp_path / "forged-acceptance",
            accepted_at=NOW + timedelta(minutes=1),
        )


def test_acceptance_rejects_positive_job_id_not_bound_by_must_start_authority(
    tmp_path: Path,
) -> None:
    client, pins, _accepted, _keys = _accepted_fixture(tmp_path / "baseline")
    with pytest.raises(ValueError, match="JOB_BINDING.json target_job_id"):
        authenticate_qualification_cache_seed(
            s3_client=client,
            pins=replace(pins, target_job_id=999),
            work_dir=tmp_path / "wrong-job-id",
            accepted_at=NOW + timedelta(minutes=1),
        )


def test_acceptance_rejects_acceptance_time_before_authenticated_closure(
    tmp_path: Path,
) -> None:
    client, pins, _accepted, _keys = _accepted_fixture(tmp_path / "baseline")
    with pytest.raises(ValueError, match="timestamp chain"):
        authenticate_qualification_cache_seed(
            s3_client=client,
            pins=pins,
            work_dir=tmp_path / "early-acceptance",
            accepted_at=NOW - timedelta(days=1),
        )


def test_accepted_record_rejects_unknown_fields_and_rehashed_semantic_lies(
    tmp_path: Path,
) -> None:
    _client, _pins, accepted, _keys = _accepted_fixture(tmp_path)
    accepted["unknown"] = True
    with pytest.raises(ValueError, match="schema"):
        validate_qualification_cache_seed_accepted(accepted)

    accepted.pop("unknown")
    accepted["cache_audit"]["top_k"] = 1024
    body = dict(accepted)
    body.pop("acceptance_body_sha256")
    accepted["acceptance_body_sha256"] = _sha(_canonical(body))
    with pytest.raises(ValueError, match="top_k"):
        validate_qualification_cache_seed_accepted(accepted)


def test_accepted_record_rejects_rehashed_target_lie_with_unchanged_job_binding(
    tmp_path: Path,
) -> None:
    _client, _pins, accepted, _keys = _accepted_fixture(tmp_path)
    authority = accepted["seed_authority"]
    assert isinstance(authority, dict)
    authority["target_job_id"] = 999
    body = dict(accepted)
    body.pop("acceptance_body_sha256")
    accepted["acceptance_body_sha256"] = _sha(_canonical(body))

    with pytest.raises(ValueError, match="JOB_BINDING.json target_job_id"):
        validate_qualification_cache_seed_accepted(accepted)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launched_at", (NOW - timedelta(hours=2)).isoformat().replace("+00:00", "Z")),
        (
            "must_start_by",
            (NOW - timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
        ),
        (
            "job_observed_at",
            (NOW + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        ),
    ],
)
def test_accepted_record_rejects_rehashed_timestamp_chain_lies(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    _client, _pins, accepted, _keys = _accepted_fixture(tmp_path)
    spend = accepted["spend_closure"]
    assert isinstance(spend, dict)
    spend.update(
        {
            "submission_submitted_at": (NOW - timedelta(hours=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "must_start_by": (NOW + timedelta(hours=2))
            .isoformat()
            .replace("+00:00", "Z"),
            "job_observed_at": (NOW + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
        }
    )
    spend[field] = value
    body = dict(accepted)
    body.pop("acceptance_body_sha256")
    accepted["acceptance_body_sha256"] = _sha(_canonical(body))
    with pytest.raises(ValueError, match="timestamp chain"):
        validate_qualification_cache_seed_accepted(accepted)


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/authenticate_qualification_cache_seed.py"
    )
    spec = importlib.util.spec_from_file_location("_cache_seed_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_requires_exact_profile_guards_sts_before_s3_and_writes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, pins, _accepted, _keys = _accepted_fixture(tmp_path / "baseline")
    module = _load_cli_module()
    events: list[str] = []

    class FixedDateTime:
        @staticmethod
        def now(_timezone: object) -> datetime:
            return NOW + timedelta(minutes=1)

    monkeypatch.setattr(module, "datetime", FixedDateTime)

    class Sts:
        def get_caller_identity(self) -> dict[str, str]:
            events.append("sts-read")
            return {
                "Account": "246813579024",
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/keep-glm52-operator/test"
                ),
            }

    class Session:
        def client(self, name: str):
            events.append(f"client:{name}")
            if name == "sts":
                return Sts()
            assert name == "s3"
            assert "sts-read" in events
            return client

    output = tmp_path / "accepted.json"
    argv = [
        "--profile",
        "keep-gpu",
        "--region",
        pins.region,
        "--bucket",
        pins.bucket,
        "--run-id",
        pins.run_id,
        "--managed-mode",
        pins.managed_mode,
        "--job-id",
        str(pins.target_job_id),
        "--seed-descriptor-key",
        pins.descriptor_key,
        "--seed-descriptor-file-sha256",
        pins.descriptor_file_sha256,
        "--seed-descriptor-body-sha256",
        pins.descriptor_body_sha256,
        "--campaign-identity-sha256",
        pins.campaign_identity_sha256,
        "--repo-tar-sha256",
        pins.repo_tar_sha256,
        "--approval-sha256",
        pins.approval_sha256,
        "--submission-key",
        pins.submission_key,
        "--submission-file-sha256",
        pins.submission_file_sha256,
        "--submission-body-sha256",
        pins.submission_body_sha256,
        "--seed-ready-key",
        pins.ready_key,
        "--output",
        str(output),
    ]
    assert module.main(argv, session_factory=lambda **_kwargs: Session()) == 0
    assert output.read_bytes() == _json_bytes(json.loads(output.read_bytes()))
    assert validate_qualification_cache_seed_accepted(json.loads(output.read_bytes()))
    assert events[:3] == ["client:sts", "sts-read", "client:s3"]
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    with pytest.raises(SystemExit):
        module.main(
            [*argv[:1], "default", *argv[2:]],
            session_factory=lambda **_kwargs: Session(),
        )
