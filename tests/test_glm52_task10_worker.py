"""Task 10 mount-free task, worker, deadline, and graceful-stop contracts."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task10_worker import (
    CAMPAIGN_MAIN_ARGV,
    CAMPAIGN_UNIT,
    DEADLINE_DROPIN_PATH,
    DEADLINE_TIMER,
    GRACEFUL_STOP_DOCUMENT,
    JOBS_LAUNCH_BODY_FIELDS,
    MountFreeTaskInputs,
    RUN_ID,
    Task10WorkerError,
    UNIT_NAMES,
    WORKER_DRAIN_ROLE_ARN,
    bootstrap_systemd_sequence,
    build_bootstrap_ready,
    build_closed_wire_request,
    build_deadline_state,
    build_graceful_stop_evidence,
    build_jobs_launch_body,
    build_retained_ssm_graceful_stop_handoff,
    build_task12_worker_drain_authority,
    build_worker_bootstrap_descriptor,
    build_worker_instance_observation,
    build_worker_runtime_authority,
    evaluate_deadline,
    render_deadline_timer_dropin,
    render_graceful_stop_document,
    render_mount_free_task,
    render_worker_drain_iam,
    render_worker_units,
    simulate_systemd_stop,
    task_environment,
    validate_closed_wire_request,
    validate_deadline_state,
    validate_graceful_stop_evidence,
    validate_jobs_launch_body,
    validate_worker_units,
    worker_instance_observation_from_mapping,
    worker_unit_hashes,
)
from mlx_vq.quality.glm52_sky_campaign import build_sky_campaign_descriptor
from mlx_vq.quality.glm52_sky_campaign import SkyCampaignLedger
from mlx_vq.quality.glm52_teich_campaign import CampaignPhase


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[1]
DEADLINE = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
SIGNING_KEY = b"task10-deadline-signing-key-value"


def _inputs() -> MountFreeTaskInputs:
    return MountFreeTaskInputs(
        job_name=RUN_ID,
        descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/descriptor.json"
        ),
        descriptor_version_id="descriptor-version-opaque-1",
        descriptor_file_sha256="1" * 64,
        approval_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/authorities/approval.json"
        ),
        approval_version_id="approval-version-opaque-1",
        approval_file_sha256="2" * 64,
        intent_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/submissions/production/intent.json"
        ),
        intent_version_id="intent-version-opaque-1",
        intent_file_sha256="3" * 64,
        intent_body_sha256="4" * 64,
        repository_archive_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/repository/repo.tar.gz"
        ),
        repository_archive_version_id="archive-version-opaque-1",
        repository_archive_file_sha256="5" * 64,
    )


def _state():
    return build_deadline_state(
        descriptor_file_sha256="1" * 64,
        instance_id="i-0123456789abcdef0",
        allocation_ordinal=1,
        execution_deadline=DEADLINE,
        signing_key=SIGNING_KEY,
    )


def _ready_readback(state=None) -> dict[str, object]:
    state = state or _state()
    return {
        "campaign_active_state": "active",
        "campaign_sub_state": "running",
        "campaign_type": "notify",
        "campaign_notify_access": "main",
        "campaign_fragment_path": (
            "/etc/systemd/system/keep-glm52-campaign.service"
        ),
        "campaign_main_pid": 4242,
        "campaign_exec_main_pid": 4242,
        "campaign_main_argv": list(CAMPAIGN_MAIN_ARGV),
        "deadline_timer_active_state": "active",
        "deadline_timer_sub_state": "waiting",
        "deadline_timer_enabled_state": "enabled",
        "deadline_timer_next_elapse_usec_realtime": (
            state.stop_assignment_at
        ),
        "deadline_timer_dropin_path": DEADLINE_DROPIN_PATH,
        "deadline_timer_dropin_sha256": hashlib.sha256(
            render_deadline_timer_dropin(state)
        ).hexdigest(),
        "deadline_timer_on_calendar": [
            state.stop_assignment_at,
            state.graceful_stop_at,
        ],
    }


def _accepted_descriptor() -> dict[str, object]:
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=datetime(2026, 7, 31, 6, 5, tzinfo=UTC),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"
        ),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-us-west-2-246813579024",
        jobs_bucket="keep-glm52-us-west-2-246813579024",
        repo_tar_key=(
            "campaigns/glm52-sky-20260724/repository/repo.tar.gz"
        ),
        repo_tar_sha256=_digest("accepted-repository-archive"),
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260724/authorities/"
            "GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=_digest("accepted-gpu-spend-approval"),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": _digest("source-snapshot"),
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": _digest("non-vq-package"),
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": _digest("teich-pack"),
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": _digest("frozen-prompt-pack"),
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": _digest("training-baseline"),
            "training_config_key": (
                "campaigns/glm52-sky-20260724/authorities/training.json"
            ),
            "training_config_sha256": _digest("training-config"),
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260724/inventories/"
                f"artifact-inventory-{_digest('artifact-inventory')}.json"
            ),
            "artifact_inventory_sha256": _digest("artifact-inventory"),
            "qualification_cache_prefix": (
                "qualification-cache/seeds/glm52-sky-20260724/"
                f"{_digest('qualification-cache')}/"
            ),
            "qualification_cache_manifest_sha256": _digest(
                "qualification-cache"
            ),
        },
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_script(filename: str, module_name: str):
    module_path = REPO_ROOT / "aws/glm52-gpu/scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ExactS3:
    def __init__(self) -> None:
        self.objects: dict[
            tuple[str, str, str],
            dict[str, object],
        ] = {}
        self.put_calls: list[dict[str, object]] = []
        self.head_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self._version = 0

    def seed(
        self,
        *,
        bucket: str,
        key: str,
        version_id: str,
        raw: bytes,
        metadata: dict[str, str],
    ) -> None:
        self.objects[(bucket, key, version_id)] = {
            "raw": raw,
            "metadata": dict(metadata),
            "checksum": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
        }

    def _versions(self, bucket: str, key: str) -> list[str]:
        return [
            version
            for stored_bucket, stored_key, version in self.objects
            if stored_bucket == bucket and stored_key == key
        ]

    def put_object(self, **values: object) -> dict[str, object]:
        self.put_calls.append(values)
        bucket = str(values["Bucket"])
        key = str(values["Key"])
        if self._versions(bucket, key):
            raise RuntimeError("PreconditionFailed")
        self._version += 1
        version_id = f"published-version-{self._version}"
        self.seed(
            bucket=bucket,
            key=key,
            version_id=version_id,
            raw=bytes(values["Body"]),
            metadata=dict(values["Metadata"]),
        )
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "put-exact-s3",
                "RetryAttempts": 0,
            },
            "VersionId": version_id,
        }

    def head_object(self, **values: object) -> dict[str, object]:
        self.head_calls.append(values)
        versions = self._versions(
            str(values["Bucket"]),
            str(values["Key"]),
        )
        if not versions:
            raise RuntimeError("NotFound")
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "head-exact-s3",
                "RetryAttempts": 0,
            },
            "VersionId": versions[-1],
        }

    def get_object(self, **values: object) -> dict[str, object]:
        self.get_calls.append(values)
        stored = self.objects[
            (
                str(values["Bucket"]),
                str(values["Key"]),
                str(values["VersionId"]),
            )
        ]
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-exact-s3",
                "RetryAttempts": 0,
            },
            "VersionId": values["VersionId"],
            "ChecksumSHA256": stored["checksum"],
            "Metadata": stored["metadata"],
            "Body": io.BytesIO(bytes(stored["raw"])),
        }


def _runtime_bundle(tmp_path: Path) -> SimpleNamespace:
    accepted = _accepted_descriptor()
    accepted_raw = canonical_json_bytes(accepted) + b"\n"
    wrapper = build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        campaign_identity_sha256=str(
            accepted["campaign_identity_sha256"]
        ),
        activation_id="activation-0001",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_job_name=RUN_ID,
        execution_deadline="2026-07-31T12:00:00Z",
        gpu_allocation_sha256=_digest("gpu-allocation"),
        base_descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/submissions/production/"
            "campaign-descriptor-v2.json"
        ),
        base_descriptor_version_id="accepted-h1c-version-opaque",
        base_descriptor_file_sha256=hashlib.sha256(
            accepted_raw
        ).hexdigest(),
        base_descriptor_body_sha256=str(
            accepted["descriptor_body_sha256"]
        ),
        archive_identity_sha256=str(accepted["repo_tar_sha256"]),
        repository_archive_version_id="archive-version-opaque-1",
        approval_identity_sha256=_digest("approval"),
        approval_version_id="approval-version-opaque-1",
        intent_identity_sha256=_digest("intent"),
        intent_version_id="intent-version-opaque-1",
        task8_live_h1d_identity_sha256=_digest("task8-live"),
        task8_spend_authority_identity_sha256=_digest("task8-spend"),
        task9_launch_identity_sha256=_digest("task9-launch"),
        task9_admission_identity_sha256=_digest("task9-admission"),
        task9_custody_identity_sha256=_digest("task9-custody"),
    )
    wrapper_raw = canonical_json_bytes(asdict(wrapper)) + b"\n"
    wrapper_sha = hashlib.sha256(wrapper_raw).hexdigest()
    inputs = replace(
        _inputs(),
        descriptor_file_sha256=wrapper_sha,
        repository_archive_file_sha256=wrapper.archive_identity_sha256,
        approval_file_sha256=wrapper.approval_identity_sha256,
        intent_body_sha256=wrapper.intent_identity_sha256,
    )
    task_yaml = render_mount_free_task(inputs)
    request_body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=inputs.job_name,
    )
    observation = build_worker_instance_observation(
        schema_version=1,
        record_type="glm52_task10_worker_instance_observation_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        campaign_identity_sha256=wrapper.campaign_identity_sha256,
        activation_id=wrapper.activation_id,
        activation_ordinal=wrapper.activation_ordinal,
        activation_ordinal_text="00000001",
        generation=wrapper.generation,
        generation_text=wrapper.generation_text,
        allocation_ordinal=1,
        allocation_ordinal_text="00000001",
        instance_id="i-0123456789abcdef0",
        action_key=wrapper.action_key,
        sky_job_name=wrapper.sky_job_name,
        task_yaml_sha256=hashlib.sha256(task_yaml.encode()).hexdigest(),
        request_body_sha256=canonical_sha256(request_body),
        task9_launch_identity_sha256=(
            wrapper.task9_launch_identity_sha256
        ),
        task9_custody_identity_sha256=(
            wrapper.task9_custody_identity_sha256
        ),
    )
    signing_key = b"r" * 32
    state = build_deadline_state(
        descriptor_file_sha256=wrapper_sha,
        instance_id=observation.instance_id,
        allocation_ordinal=observation.allocation_ordinal,
        execution_deadline=DEADLINE,
        signing_key=signing_key,
    )
    campaign_path = tmp_path / "campaign.json"
    wrapper_path = tmp_path / "worker-bootstrap.json"
    observation_path = tmp_path / "worker-observation.json"
    state_path = tmp_path / "deadline-state.json"
    key_path = tmp_path / "deadline-state.key"
    campaign_path.write_bytes(accepted_raw)
    wrapper_path.write_bytes(wrapper_raw)
    observation_path.write_bytes(
        canonical_json_bytes(asdict(observation)) + b"\n"
    )
    state_path.write_bytes(canonical_json_bytes(asdict(state)) + b"\n")
    key_path.write_bytes(signing_key)
    return SimpleNamespace(
        accepted=accepted,
        campaign_path=campaign_path,
        wrapper=wrapper,
        wrapper_path=wrapper_path,
        observation=observation,
        observation_path=observation_path,
        state=state,
        state_path=state_path,
        key_path=key_path,
        root=tmp_path / "campaign-root",
    )


def test_mount_free_task_is_exact_one_node_aws_on_demand_and_archive_only() -> None:
    inputs = _inputs()
    raw = render_mount_free_task(inputs)
    task = yaml.safe_load(raw)

    assert task["name"] == inputs.job_name
    assert task["num_nodes"] == 1
    assert task["api_server_access"] is False
    assert task["resources"] == {
        "infra": "aws/us-west-2",
        "instance_type": "p5.48xlarge",
        "use_spot": False,
        "max_hourly_cost": 55.04,
        "job_recovery": {
            "strategy": "FAILOVER",
            "max_restarts_on_errors": 0,
        },
    }
    assert task["envs"] == task_environment(inputs)
    assert "file_mounts" not in task
    assert "workdir" not in task
    assert "sky jobs launch" not in raw
    assert task["setup"].count("s3api get-object") == 2
    assert "--version-id" in task["setup"]
    assert "/opt/keep-campaign/repo" in task["setup"]
    assert "bootstrap_production_campaign.sh" in task["setup"]


def test_mount_free_task_rejects_noncanonical_production_job_name() -> None:
    with pytest.raises(Task10WorkerError, match="job name"):
        render_mount_free_task(
            replace(_inputs(), job_name=f"{RUN_ID}-production")
        )


def test_fix2_mount_free_setup_passes_every_closed_task_env_across_sudo() -> None:
    inputs = _inputs()
    setup = yaml.safe_load(render_mount_free_task(inputs))["setup"]
    root_handoff = setup.split("sudo env \\\n", 1)[1]

    for name in task_environment(inputs):
        assert f'{name}="${name}"' in root_handoff
    assert "sudo -E" not in setup
    assert (
        "sudo /opt/keep-campaign/repo/aws/glm52-gpu/skypilot/"
        "bootstrap_production_campaign.sh"
    ) not in setup


def test_fix3_runtime_authority_cross_binds_exact_frozen_sky_job_name(
    tmp_path: Path,
) -> None:
    bundle = _runtime_bundle(tmp_path)

    accepted = build_worker_runtime_authority(
        bundle.wrapper,
        bundle.observation,
    )
    assert accepted.observation.sky_job_name == bundle.wrapper.sky_job_name
    assert accepted.observation.sky_job_name == RUN_ID

    for foreign_job_name in (
        "foreign-production-job",
        RUN_ID.upper(),
    ):
        mutant = asdict(bundle.observation)
        mutant["sky_job_name"] = foreign_job_name
        mutant.pop("observation_body_sha256")
        mutant["observation_body_sha256"] = canonical_sha256(mutant)
        with pytest.raises(Task10WorkerError):
            build_worker_runtime_authority(
                bundle.wrapper,
                replace(bundle.observation, **mutant),
            )

    missing = asdict(bundle.observation)
    missing.pop("sky_job_name")
    with pytest.raises(Task10WorkerError, match="schema"):
        worker_instance_observation_from_mapping(missing)


def test_real_wrapper_breaks_hash_cycle_and_initializes_from_accepted_h1c(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = _accepted_descriptor()
    accepted_raw = canonical_json_bytes(accepted) + b"\n"
    archive_sha = str(accepted["repo_tar_sha256"])
    approval_sha = _digest("production-approval-file")
    intent_file_sha = _digest("production-intent-file")
    intent_body_sha = _digest("production-intent-body")
    wrapper = build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        campaign_identity_sha256=str(
            accepted["campaign_identity_sha256"]
        ),
        activation_id="activation-0001",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_job_name=RUN_ID,
        execution_deadline="2026-07-31T12:00:00Z",
        gpu_allocation_sha256=_digest("gpu-allocation"),
        base_descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/submissions/production/"
            "campaign-descriptor-v2.json"
        ),
        base_descriptor_version_id="accepted-h1c-version-opaque",
        base_descriptor_file_sha256=hashlib.sha256(
            accepted_raw
        ).hexdigest(),
        base_descriptor_body_sha256=str(
            accepted["descriptor_body_sha256"]
        ),
        archive_identity_sha256=archive_sha,
        repository_archive_version_id="archive-version-opaque",
        approval_identity_sha256=approval_sha,
        approval_version_id="approval-version-opaque",
        intent_identity_sha256=intent_body_sha,
        intent_version_id="intent-version-opaque",
        task8_live_h1d_identity_sha256=_digest("task8-live-h1d"),
        task8_spend_authority_identity_sha256=_digest(
            "task8-spend-authority"
        ),
        task9_launch_identity_sha256=_digest("task9-launch"),
        task9_admission_identity_sha256=_digest("task9-admission"),
        task9_custody_identity_sha256=_digest("task9-custody"),
    )
    wrapper_mapping = asdict(wrapper)
    assert "task_yaml_sha256" not in wrapper_mapping
    assert "request_body_sha256" not in wrapper_mapping
    wrapper_raw = canonical_json_bytes(wrapper_mapping) + b"\n"
    wrapper_file_sha = hashlib.sha256(wrapper_raw).hexdigest()

    inputs = MountFreeTaskInputs(
        job_name=wrapper.sky_job_name,
        descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/runtime/"
            "worker-bootstrap.json"
        ),
        descriptor_version_id="worker-wrapper-version-opaque",
        descriptor_file_sha256=wrapper_file_sha,
        approval_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/authorities/approval.json"
        ),
        approval_version_id="approval-version-opaque",
        approval_file_sha256=approval_sha,
        intent_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/submissions/production/"
            "intent.json"
        ),
        intent_version_id="intent-version-opaque",
        intent_file_sha256=intent_file_sha,
        intent_body_sha256=intent_body_sha,
        repository_archive_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/repository/repo.tar.gz"
        ),
        repository_archive_version_id="archive-version-opaque",
        repository_archive_file_sha256=archive_sha,
    )
    task_yaml = render_mount_free_task(inputs)
    request_body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=inputs.job_name,
    )
    observation = build_worker_instance_observation(
        schema_version=1,
        record_type="glm52_task10_worker_instance_observation_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        campaign_identity_sha256=wrapper.campaign_identity_sha256,
        activation_id=wrapper.activation_id,
        activation_ordinal=wrapper.activation_ordinal,
        activation_ordinal_text="00000001",
        generation=wrapper.generation,
        generation_text=wrapper.generation_text,
        allocation_ordinal=1,
        allocation_ordinal_text="00000001",
        instance_id="i-0123456789abcdef0",
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_job_name=wrapper.sky_job_name,
        task_yaml_sha256=hashlib.sha256(task_yaml.encode()).hexdigest(),
        request_body_sha256=canonical_sha256(request_body),
        task9_launch_identity_sha256=(
            wrapper.task9_launch_identity_sha256
        ),
        task9_custody_identity_sha256=(
            wrapper.task9_custody_identity_sha256
        ),
    )
    runtime = build_worker_runtime_authority(wrapper, observation)
    assert runtime.descriptor.base_descriptor_file_sha256 == hashlib.sha256(
        accepted_raw
    ).hexdigest()

    module_path = (
        REPO_ROOT / "aws/glm52-gpu/scripts/glm52_deadline_guard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task10_deadline_guard_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    descriptor_path = tmp_path / "worker-bootstrap.json"
    observation_path = tmp_path / "worker-observation.json"
    state_path = tmp_path / "deadline-state.json"
    key_path = tmp_path / "deadline-state.key"
    dropin_path = tmp_path / "10-immutable-deadline.conf"
    descriptor_path.write_bytes(wrapper_raw)
    observation_path.write_bytes(
        canonical_json_bytes(asdict(observation)) + b"\n"
    )
    monkeypatch.setattr(module, "DESCRIPTOR", descriptor_path)
    monkeypatch.setattr(module, "OBSERVATION", observation_path)
    monkeypatch.setattr(module, "STATE", state_path)
    monkeypatch.setattr(module, "KEY", key_path)
    monkeypatch.setattr(module, "DROPIN", dropin_path)
    monkeypatch.setenv("GLM52_DESCRIPTOR_FILE_SHA256", wrapper_file_sha)

    module._initialize()

    state = json.loads(state_path.read_bytes())
    assert state["descriptor_file_sha256"] == wrapper_file_sha
    assert state["instance_id"] == observation.instance_id
    assert state["allocation_ordinal"] == observation.allocation_ordinal
    assert state["execution_deadline"] == wrapper.execution_deadline
    assert len(key_path.read_bytes()) == 32
    assert b"OnCalendar=2026-07-31T11:00:00Z" in dropin_path.read_bytes()


def test_fixed_runtime_entrypoint_derives_authority_before_real_controller_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    module = _load_script(
        "run_task10_production_campaign.py",
        "task10_production_runtime_test",
    )
    campaign_runner = _load_script(
        "../../../benchmarks/run_glm52_campaign.py",
        "task10_real_campaign_controller_test",
    )
    monkeypatch.setattr(module, "CAMPAIGN", bundle.campaign_path)
    monkeypatch.setattr(module, "WRAPPER", bundle.wrapper_path)
    monkeypatch.setattr(module, "OBSERVATION", bundle.observation_path)
    monkeypatch.setattr(module, "STATE", bundle.state_path)
    monkeypatch.setattr(module, "KEY", bundle.key_path)
    monkeypatch.setattr(module, "CAMPAIGN_ROOT", bundle.root)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", REPO_ROOT)
    monkeypatch.setattr(
        module,
        "BENCHMARK",
        REPO_ROOT / "benchmarks/run_glm52_campaign.py",
    )
    for name in (
        "GLM52_EXECUTION_DEADLINE",
        "GLM52_GPU_ALLOCATION_SHA256",
        "GLM52_GPU_SPEND_AUTHORITY_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "GLM52_CAMPAIGN_STOP_FILE",
        str(tmp_path / "run/STOP"),
    )
    observed: dict[str, object] = {}

    def execute(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        assert executable == "/usr/bin/python3"
        assert arguments == [
            "/usr/bin/python3",
            str(REPO_ROOT / "benchmarks/run_glm52_campaign.py"),
            "--descriptor",
            str(bundle.campaign_path),
            "--root",
            str(bundle.root),
            "--repo-root",
            str(REPO_ROOT),
        ]
        saved = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(environment)
            controller = campaign_runner.CampaignController(
                descriptor_path=bundle.campaign_path,
                root=bundle.root,
                repo_root=REPO_ROOT,
            )
        finally:
            os.environ.clear()
            os.environ.update(saved)
        observed["runtime"] = controller.runtime

    assert module.main([], execve=execute) == 0
    runtime = observed["runtime"]
    assert runtime.execution_deadline == DEADLINE
    assert (
        runtime.gpu_allocation_sha256
        == bundle.wrapper.gpu_allocation_sha256
    )
    assert (
        runtime.gpu_spend_authority_sha256
        == bundle.wrapper.task8_spend_authority_identity_sha256
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-allocation",
        "gpu-allocation",
        "deadline",
        "spend",
        "wrapper",
    ],
)
def test_fixed_runtime_entrypoint_rejects_mutated_authority_before_campaign_work(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    module = _load_script(
        "run_task10_production_campaign.py",
        "task10_production_runtime_mutant_test_" + mutation.replace("-", "_"),
    )
    monkeypatch.setattr(module, "CAMPAIGN", bundle.campaign_path)
    monkeypatch.setattr(module, "WRAPPER", bundle.wrapper_path)
    monkeypatch.setattr(module, "OBSERVATION", bundle.observation_path)
    monkeypatch.setattr(module, "STATE", bundle.state_path)
    monkeypatch.setattr(module, "KEY", bundle.key_path)
    monkeypatch.setattr(module, "CAMPAIGN_ROOT", bundle.root)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", REPO_ROOT)
    monkeypatch.setattr(
        module,
        "BENCHMARK",
        REPO_ROOT / "benchmarks/run_glm52_campaign.py",
    )
    if mutation == "wrapper":
        bundle.wrapper_path.write_bytes(b"{}\n")
    else:
        values = asdict(bundle.wrapper)
        values.pop("descriptor_body_sha256")
        if mutation == "missing-allocation":
            values.pop("gpu_allocation_sha256")
            mutant_raw = canonical_json_bytes(values) + b"\n"
        else:
            field, value = {
                "gpu-allocation": (
                    "gpu_allocation_sha256",
                    _digest("foreign-gpu-allocation"),
                ),
                "deadline": (
                    "execution_deadline",
                    "2026-07-31T11:59:59Z",
                ),
                "spend": (
                    "task8_spend_authority_identity_sha256",
                    _digest("foreign-spend"),
                ),
            }[mutation]
            values[field] = value
            mutant = build_worker_bootstrap_descriptor(**values)
            mutant_raw = canonical_json_bytes(asdict(mutant)) + b"\n"
        bundle.wrapper_path.write_bytes(mutant_raw)
    campaign_calls: list[object] = []

    with pytest.raises(SystemExit) as failure:
        module.main(
            [],
            execve=lambda *_arguments: campaign_calls.append(object()),
        )

    assert failure.value.code == 70
    assert campaign_calls == []


def test_worker_materializer_recomputes_task_and_body_tags_from_closed_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preliminary = _inputs()
    wrapper = build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        campaign_identity_sha256=_digest("campaign"),
        activation_id="activation-0001",
        activation_ordinal=1,
        generation=1,
        generation_text="00000001",
        action_key="ACTION#00000001#SKY_POST#00000001",
        sky_job_name=preliminary.job_name,
        execution_deadline="2026-07-31T12:00:00Z",
        gpu_allocation_sha256=_digest("gpu-allocation"),
        base_descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/base.json"
        ),
        base_descriptor_version_id="accepted-h1c-version",
        base_descriptor_file_sha256=_digest("accepted-h1c-file"),
        base_descriptor_body_sha256=_digest("accepted-h1c-body"),
        archive_identity_sha256=(
            preliminary.repository_archive_file_sha256
        ),
        repository_archive_version_id=(
            preliminary.repository_archive_version_id
        ),
        approval_identity_sha256=preliminary.approval_file_sha256,
        approval_version_id=preliminary.approval_version_id,
        intent_identity_sha256=preliminary.intent_body_sha256,
        intent_version_id=preliminary.intent_version_id,
        task8_live_h1d_identity_sha256=_digest("task8-live"),
        task8_spend_authority_identity_sha256=_digest("task8-spend"),
        task9_launch_identity_sha256=_digest("task9-launch"),
        task9_admission_identity_sha256=_digest("task9-admission"),
        task9_custody_identity_sha256=_digest("task9-custody"),
    )
    wrapper_raw = canonical_json_bytes(asdict(wrapper)) + b"\n"
    inputs = replace(
        preliminary,
        descriptor_file_sha256=hashlib.sha256(wrapper_raw).hexdigest(),
    )
    task_yaml = render_mount_free_task(inputs)
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=inputs.job_name,
    )
    task_sha = hashlib.sha256(task_yaml.encode()).hexdigest()
    body_sha = canonical_sha256(body)
    descriptor_path = tmp_path / "worker-bootstrap.json"
    output_path = tmp_path / "worker-observation.json"
    descriptor_path.write_bytes(wrapper_raw)
    for name, value in task_environment(inputs).items():
        monkeypatch.setenv(name, value)

    module_path = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/materialize_task10_worker_observation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task10_worker_materializer_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DESCRIPTOR", descriptor_path)
    monkeypatch.setattr(module, "OUTPUT", output_path)
    monkeypatch.setattr(
        module,
        "_self_identity",
        lambda: (
            "i-0123456789abcdef0",
            "246813579024",
            "us-west-2",
        ),
    )
    tags = {
        "RunId": RUN_ID,
        "campaign-identity-sha256": wrapper.campaign_identity_sha256,
        "activation-id": wrapper.activation_id,
        "activation-ordinal-text": "00000001",
        "generation-text": wrapper.generation_text,
        "allocation-ordinal-text": "00000001",
        "action-key": "ACTION#00000001#SKY_POST#00000001",
        "sky-job-name": wrapper.sky_job_name,
        "task-yaml-sha256": task_sha,
        "request-body-sha256": body_sha,
    }
    monkeypatch.setattr(
        module,
        "_instance",
        lambda _instance_id: {
            "Tags": [
                {"Key": key, "Value": value}
                for key, value in tags.items()
            ]
        },
    )

    assert module.main([]) == 0
    observation = json.loads(output_path.read_bytes())
    assert observation["task_yaml_sha256"] == task_sha
    assert observation["request_body_sha256"] == body_sha

    output_path.unlink()
    tags["task-yaml-sha256"] = _digest("foreign-task")
    with pytest.raises(SystemExit) as failure:
        module.main([])
    assert failure.value.code == 70
    assert not output_path.exists()


def test_worker_authenticator_delegates_to_production_intent_validator_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/authenticate_task10_worker_inputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task10_worker_authenticator_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    intent = {
        "record_type": "glm52_sky_production_submission_intent_v1",
        "intent_body_sha256": _digest("production-intent-body"),
    }
    raw = canonical_json_bytes(intent) + b"\n"
    path = tmp_path / "intent.json"
    path.write_bytes(raw)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "validate_production_submission_intent",
        lambda value: calls.append(value) or value,
    )
    monkeypatch.setattr(
        module,
        "production_submission_intent_file_bytes",
        lambda _value: raw,
    )
    monkeypatch.setenv(
        "GLM52_SUBMISSION_INTENT_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setenv(
        "GLM52_SUBMISSION_INTENT_BODY_SHA256",
        str(intent["intent_body_sha256"]),
    )

    assert module._authenticate_intent(path) == intent
    assert calls == [intent]

    monkeypatch.setenv(
        "GLM52_SUBMISSION_INTENT_BODY_SHA256",
        _digest("foreign-intent-body"),
    )
    with pytest.raises(ValueError, match="body identity"):
        module._authenticate_intent(path)


def test_production_sigterm_enters_status75_checkpoint_sync_and_drain_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = REPO_ROOT / "benchmarks/run_glm52_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "task10_campaign_sigterm_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller = object.__new__(module.CampaignController)
    controller.kind = module.RuntimeCampaignKind.SKYPILOT
    controller.stop_file = tmp_path / "run/STOP"
    controller.stop_file.parent.mkdir(parents=True)
    controller.ledger_path = tmp_path / "ledger.jsonl"
    controller.ledger_path.write_text("{}\n")
    controller.execution_deadline = datetime.now(UTC) + timedelta(hours=4)
    controller._production_sigterm_received = False
    controller._stop_monitor = module.threading.Event()
    events: list[object] = []
    installed: list[object] = []
    previous = object()
    monkeypatch.setenv("GLM52_MANAGED_MODE", "production")
    monkeypatch.setattr(
        module.signal,
        "getsignal",
        lambda _signum: previous,
    )
    monkeypatch.setattr(
        module.signal,
        "signal",
        lambda _signum, handler: installed.append(handler),
    )
    controller.restore = lambda: events.append("restore")
    controller._publish_heartbeat = lambda: events.append("heartbeat")
    controller._deadline_monitor = lambda: None

    def bootstrap() -> None:
        events.append("bootstrap")
        handler = installed[0]
        assert callable(handler)
        handler(module.signal.SIGTERM, None)

    controller.bootstrap = bootstrap
    controller.sync_all = lambda *, final=False: events.append(
        ("checkpoint-sync", final)
    )
    controller.ledger = lambda: type(
        "Ledger",
        (),
        {"current_phase": module.CampaignPhase.BOOTSTRAP},
    )()
    controller.drain = lambda **values: events.append(("drain", values))

    assert controller.run() == 0
    assert controller.stop_file.is_file()
    assert ("checkpoint-sync", True) in events
    drain = next(item for item in events if item[0] == "drain")
    assert drain[1] == {
        "outcome": "resumable_deadline",
        "drain_reason": "operator_graceful_stop",
    }
    assert events.index(("checkpoint-sync", True)) < events.index(drain)
    assert installed[-1] is previous


def test_jobs_launch_body_has_the_exact_pinned_013_field_set_and_nulls() -> None:
    task_yaml = render_mount_free_task(_inputs())
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=_inputs().job_name,
    )

    assert set(body) == JOBS_LAUNCH_BODY_FIELDS
    assert body == {
        "env_vars": {},
        "entrypoint": "",
        "entrypoint_command": "",
        "using_remote_api_server": True,
        "override_skypilot_config": {},
        "override_skypilot_config_path": None,
        "file_mounts_blob_id": None,
        "client_api_version": None,
        "task": task_yaml,
        "name": _inputs().job_name,
        "pool": None,
        "num_jobs": None,
    }


def test_pinned_skypilot_013_parser_accepts_mount_free_production_task(
    tmp_path: Path,
) -> None:
    pinned_python = Path(
        "/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/python"
    )
    task = tmp_path / "task10-production.yaml"
    task.write_text(render_mount_free_task(_inputs()))
    result = subprocess.run(
        [
            str(pinned_python),
            str(
                REPO_ROOT
                / "aws/glm52-gpu/scripts/validate_skypilot_control_plane.py"
            ),
            "--task",
            str(task),
            "--config",
            str(
                REPO_ROOT
                / "aws/glm52-gpu/skypilot/skypilot-config.yaml.in"
            ),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "SKYPILOT_DISABLE_USAGE_COLLECTION": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "instance_type": "p5.48xlarge",
        "record_type": "glm52_skypilot_parser_validation_v1",
        "region": "us-west-2",
        "skypilot_version": "0.13.0",
        "status": "passed",
        "task": _inputs().job_name,
    }


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("env_vars", {"CALLER": "authority"}),
        ("entrypoint", "caller"),
        ("using_remote_api_server", False),
        ("file_mounts_blob_id", "blob"),
        ("client_api_version", 56),
        ("pool", "pool"),
        ("num_jobs", 1),
        ("name", "foreign"),
    ],
)
def test_jobs_launch_body_rejects_wrong_env_value_and_nullability_mutants(
    mutation: str,
    value: object,
) -> None:
    task_yaml = render_mount_free_task(_inputs())
    body = dict(
        build_jobs_launch_body(
            task_yaml=task_yaml,
            job_name=_inputs().job_name,
        )
    )
    body[mutation] = value

    with pytest.raises(Task10WorkerError):
        validate_jobs_launch_body(
            body,
            expected_task_yaml=task_yaml,
            expected_job_name=_inputs().job_name,
        )


def test_jobs_launch_body_rejects_extra_and_missing_fields() -> None:
    task_yaml = render_mount_free_task(_inputs())
    body = dict(
        build_jobs_launch_body(
            task_yaml=task_yaml,
            job_name=_inputs().job_name,
        )
    )
    for mutant in ({**body, "extra": None}, {k: v for k, v in body.items() if k != "pool"}):
        with pytest.raises(Task10WorkerError, match="field set"):
            validate_jobs_launch_body(
                mutant,
                expected_task_yaml=task_yaml,
                expected_job_name=_inputs().job_name,
            )


def test_wire_request_closes_endpoint_headers_bearer_user_task_and_envelope() -> None:
    task_yaml = render_mount_free_task(_inputs())
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=_inputs().job_name,
    )
    request = build_closed_wire_request(
        body=body,
        task_yaml=task_yaml,
        job_name=_inputs().job_name,
        bearer_token="sky_exact-service-token",
        service_account_user_id="glm52-service-account",
        relay_envelope_sha256="6" * 64,
        client_certificate_sha256="7" * 64,
    )

    assert request.endpoint == "/jobs/launch"
    assert request.headers == {
        "Authorization": "Bearer sky_exact-service-token",
        "Content-Type": "application/json",
        "X-SkyPilot-API-Version": "56",
        "X-SkyPilot-Version": "0.13.0",
    }
    assert request.service_account_role == "user"
    validate_closed_wire_request(
        request,
        expected_body=body,
        expected_task_yaml=task_yaml,
        expected_job_name=_inputs().job_name,
        expected_user_id="glm52-service-account",
        expected_relay_envelope_sha256="6" * 64,
        expected_client_certificate_sha256="7" * 64,
    )

    with pytest.raises(Task10WorkerError):
        validate_closed_wire_request(
            replace(request, endpoint="/jobs/queue"),
            expected_body=body,
            expected_task_yaml=task_yaml,
            expected_job_name=_inputs().job_name,
            expected_user_id="glm52-service-account",
            expected_relay_envelope_sha256="6" * 64,
            expected_client_certificate_sha256="7" * 64,
        )


def test_exact_three_worker_units_freeze_stop_and_path_contract() -> None:
    units = render_worker_units()
    validate_worker_units(units)

    assert tuple(units) == UNIT_NAMES
    campaign = units[CAMPAIGN_UNIT].decode()
    assert "Restart=no" in campaign
    assert "KillMode=control-group" in campaign
    assert "TimeoutStopSec=1200" in campaign
    assert "SendSIGKILL=no" in campaign
    assert "WorkingDirectory=/opt/keep-campaign/repo" in campaign
    assert "/etc/keep-glm52/campaign.json" in campaign
    assert "/var/lib/keep-glm52/deadline-state.json" in campaign
    assert "/mnt/nvme/glm52-campaign/runtime/heavy-job.lock" in campaign
    assert "EnvironmentFile=" not in campaign
    assert "Persistent=true" in units[DEADLINE_TIMER].decode()

    with pytest.raises(Task10WorkerError, match="exactly three"):
        validate_worker_units({**units, "caller.service": b"[Service]\n"})


def test_fix2_campaign_start_is_synchronous_notify_readiness() -> None:
    campaign = render_worker_units()[CAMPAIGN_UNIT].decode()
    benchmark = (
        REPO_ROOT / "benchmarks/run_glm52_campaign.py"
    ).read_text()

    assert "Type=notify" in campaign
    assert "NotifyAccess=main" in campaign
    assert "TimeoutStartSec=300" in campaign
    assert "Type=simple" not in campaign
    assert "_notify_systemd_ready" in benchmark
    assert benchmark.index("controller = CampaignController(") < benchmark.rindex(
        "_notify_systemd_ready()"
    )


def test_fix2_benchmark_notifies_only_after_real_controller_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script(
        "../../../benchmarks/run_glm52_campaign.py",
        "task10_fix2_notify_order",
    )
    events: list[str] = []

    class Controller:
        def __init__(self, **_values: object) -> None:
            events.append("constructed")

        def run(self) -> int:
            events.append("run")
            return 0

    monkeypatch.setattr(runner, "CampaignController", Controller)
    monkeypatch.setattr(
        runner,
        "_notify_systemd_ready",
        lambda: events.append("ready"),
    )
    argv = [
        "--descriptor",
        str(tmp_path / "campaign.json"),
        "--root",
        str(tmp_path / "root"),
        "--repo-root",
        str(REPO_ROOT),
    ]

    assert runner.main(argv) == 0
    assert events == ["constructed", "ready", "run"]

    class FailedController:
        def __init__(self, **_values: object) -> None:
            raise ValueError("controller construction failed")

    events.clear()
    monkeypatch.setattr(runner, "CampaignController", FailedController)
    with pytest.raises(ValueError, match="controller construction"):
        runner.main(argv)
    assert events == []


def test_signed_deadline_and_oncalendar_dropin_are_wall_clock_derived() -> None:
    state = _state()
    validate_deadline_state(
        state,
        descriptor_file_sha256="1" * 64,
        signing_key=SIGNING_KEY,
    )
    dropin = render_deadline_timer_dropin(state).decode()

    assert "OnCalendar=2026-07-31T11:00:00Z" in dropin
    assert "OnCalendar=2026-07-31T11:10:00Z" in dropin
    assert "Persistent=true" in dropin

    with pytest.raises(Task10WorkerError):
        validate_deadline_state(
            replace(state, state_signature_sha256="0" * 64),
            descriptor_file_sha256="1" * 64,
            signing_key=SIGNING_KEY,
        )
    with pytest.raises(Task10WorkerError):
        validate_deadline_state(
            state,
            descriptor_file_sha256="9" * 64,
            signing_key=SIGNING_KEY,
        )


@pytest.mark.parametrize(
    ("minutes_before", "phase", "allow_start", "stop", "commands"),
    [
        (61, "BEFORE_T_MINUS_60", True, False, ()),
        (60, "T_MINUS_60", False, True, ()),
        (50, "T_MINUS_50", False, True, (("systemctl", "stop", CAMPAIGN_UNIT),)),
        (30, "T_MINUS_30", False, True, (("systemctl", "stop", CAMPAIGN_UNIT),)),
    ],
)
def test_reboot_reconstruction_enforces_every_deadline_edge(
    minutes_before: int,
    phase: str,
    allow_start: bool,
    stop: bool,
    commands: tuple[tuple[str, ...], ...],
) -> None:
    decision = evaluate_deadline(
        now=DEADLINE - timedelta(minutes=minutes_before),
        state=_state(),
        descriptor_file_sha256="1" * 64,
        signing_key=SIGNING_KEY,
        stop_marker_present=False,
    )

    assert decision.phase == phase
    assert decision.allow_campaign_start is allow_start
    assert decision.recreate_stop_marker is stop
    assert decision.systemctl_commands == commands
    assert decision.allow_restart is False
    assert decision.allow_force_kill is False


def test_premature_stop_and_missing_corrupt_persistent_state_fail_closed() -> None:
    with pytest.raises(Task10WorkerError, match="premature"):
        evaluate_deadline(
            now=DEADLINE - timedelta(minutes=61),
            state=_state(),
            descriptor_file_sha256="1" * 64,
            signing_key=SIGNING_KEY,
            stop_marker_present=True,
        )
    with pytest.raises(Task10WorkerError, match="absent"):
        evaluate_deadline(
            now=DEADLINE,
            state=None,  # type: ignore[arg-type]
            descriptor_file_sha256="1" * 64,
            signing_key=SIGNING_KEY,
            stop_marker_present=False,
        )


@pytest.mark.parametrize(
    ("completed_edge", "expected_phase", "expected_commands"),
    [
        ("T_MINUS_60", "T_MINUS_60", ()),
        (
            "T_MINUS_50",
            "T_MINUS_50",
            (("systemctl", "stop", CAMPAIGN_UNIT),),
        ),
    ],
)
def test_fix2_signed_deadline_progress_never_regresses_after_clock_rollback(
    completed_edge: str,
    expected_phase: str,
    expected_commands: tuple[tuple[str, ...], ...],
) -> None:
    state = build_deadline_state(
        descriptor_file_sha256="1" * 64,
        instance_id="i-0123456789abcdef0",
        allocation_ordinal=1,
        execution_deadline=DEADLINE,
        signing_key=SIGNING_KEY,
        last_completed_edge=completed_edge,
    )

    decision = evaluate_deadline(
        now=DEADLINE - timedelta(minutes=61),
        state=state,
        descriptor_file_sha256="1" * 64,
        signing_key=SIGNING_KEY,
        stop_marker_present=False,
    )

    assert decision.phase == expected_phase
    assert decision.allow_campaign_start is False
    assert decision.allow_new_assignments is False
    assert decision.recreate_stop_marker is True
    assert decision.persist_edge is None
    assert decision.systemctl_commands == expected_commands
    assert decision.allow_restart is False
    assert decision.allow_force_kill is False


def test_child_survives_full_timeout_without_systemd_sigkill() -> None:
    observation = simulate_systemd_stop(
        control_group_pids=[100, 101],
        exited_after_sigterm=[100],
        elapsed_seconds=1200,
    )

    assert observation.term_signals == (100, 101)
    assert observation.kill_signals == ()
    assert observation.remaining_control_group == (101,)
    assert observation.active_state == "deactivating"
    assert observation.result == "timeout"
    assert observation.exec_main_status == 15


def test_bootstrap_sequence_and_marker_last_ledger_binding() -> None:
    units = render_worker_units()
    hashes = worker_unit_hashes(units)
    state = _state()
    readback = _ready_readback(state)

    assert bootstrap_systemd_sequence() == (
        ("systemctl", "daemon-reload"),
        ("systemctl", "enable", DEADLINE_TIMER),
        ("systemctl", "start", DEADLINE_TIMER),
        ("systemctl", "start", CAMPAIGN_UNIT),
        ("systemctl", "is-enabled", DEADLINE_TIMER),
        ("systemctl", "show", DEADLINE_TIMER),
        ("systemctl", "show", CAMPAIGN_UNIT),
    )
    ready = build_bootstrap_ready(
        unit_hashes=hashes,
        deadline_state=state,
        systemd_readback=readback,
        descriptor_file_sha256="1" * 64,
        descriptor_version_id="descriptor-version-1",
        archive_file_sha256="5" * 64,
        archive_version_id="archive-version-1",
        instance_id="i-0123456789abcdef0",
        allocation_ordinal=1,
        marker_publish_sequence=9,
    )
    assert ready["unit_file_sha256"] == hashes
    assert ready["bootstrap_body_sha256"] == canonical_sha256(
        {key: value for key, value in ready.items() if key != "bootstrap_body_sha256"}
    )

    with pytest.raises(Task10WorkerError, match="marker-last"):
        build_bootstrap_ready(
            unit_hashes=hashes,
            deadline_state=state,
            systemd_readback={**readback, "deadline_timer_active_state": "failed"},
            descriptor_file_sha256="1" * 64,
            descriptor_version_id="descriptor-version-1",
            archive_file_sha256="5" * 64,
            archive_version_id="archive-version-1",
            instance_id="i-0123456789abcdef0",
            allocation_ordinal=1,
            marker_publish_sequence=9,
        )
    with pytest.raises(Task10WorkerError, match="marker-last"):
        build_bootstrap_ready(
            unit_hashes=hashes,
            deadline_state=state,
            systemd_readback={
                **readback,
                "deadline_timer_next_elapse_usec_realtime": (
                    state.graceful_stop_at
                ),
            },
            descriptor_file_sha256="1" * 64,
            descriptor_version_id="descriptor-version-1",
            archive_file_sha256="5" * 64,
            archive_version_id="archive-version-1",
            instance_id="i-0123456789abcdef0",
            allocation_ordinal=1,
            marker_publish_sequence=9,
        )


@pytest.mark.parametrize(
    "mutation",
    ["corrupt-signature", "foreign-wrapper", "foreign-h1c"],
)
def test_bootstrap_publisher_authenticates_deadline_state_and_wrapper_before_publish(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    module = _load_script(
        "publish_task10_bootstrap_ready.py",
        "task10_bootstrap_publisher_test_" + mutation.replace("-", "_"),
    )
    installed = tmp_path / "systemd"
    production = tmp_path / "production-units"
    installed.mkdir()
    production.mkdir()
    for name, raw in render_worker_units().items():
        (installed / name).write_bytes(raw)
        (production / name).write_bytes(raw)
    marker = tmp_path / "BOOTSTRAP_READY.json"
    monkeypatch.setattr(module, "SYSTEMD", installed)
    monkeypatch.setattr(module, "PRODUCTION_UNITS", production)
    monkeypatch.setattr(module, "MARKER", marker)
    monkeypatch.setattr(module, "STATE", bundle.state_path)
    monkeypatch.setattr(module, "DESCRIPTOR", bundle.campaign_path)
    monkeypatch.setattr(module, "WORKER_DESCRIPTOR", bundle.wrapper_path)
    monkeypatch.setattr(module, "KEY", bundle.key_path, raising=False)
    monkeypatch.setattr(
        module,
        "_readback",
        lambda _state: _ready_readback(_state),
    )
    monkeypatch.setenv(
        "GLM52_REPOSITORY_ARCHIVE_FILE_SHA256",
        bundle.wrapper.archive_identity_sha256,
    )
    monkeypatch.setenv(
        "GLM52_REPOSITORY_ARCHIVE_VERSION_ID",
        "archive-version-opaque",
    )
    if mutation == "corrupt-signature":
        state = asdict(bundle.state)
        state["state_signature_sha256"] = "0" * 64
        bundle.state_path.write_bytes(canonical_json_bytes(state) + b"\n")
    elif mutation == "foreign-wrapper":
        values = asdict(bundle.wrapper)
        values.pop("descriptor_body_sha256")
        values["gpu_allocation_sha256"] = _digest(
            "foreign-gpu-allocation"
        )
        foreign = build_worker_bootstrap_descriptor(**values)
        bundle.wrapper_path.write_bytes(
            canonical_json_bytes(asdict(foreign)) + b"\n"
        )
    else:
        bundle.campaign_path.write_bytes(
            canonical_json_bytes(
                {
                    "bucket": "foreign-bucket",
                    "campaign_identity_sha256": "f" * 64,
                }
            )
            + b"\n"
        )
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_publish",
        lambda **values: published.append(values),
    )

    with pytest.raises(SystemExit) as failure:
        module.main([])

    assert failure.value.code == 70
    assert published == []
    assert not marker.exists()


def test_fix2_bootstrap_waits_for_durable_hash_linked_ledger_receipt() -> None:
    bootstrap = (
        REPO_ROOT
        / "aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh"
    ).read_text()
    controller = (
        REPO_ROOT / "benchmarks/run_glm52_campaign.py"
    ).read_text()
    publisher = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/"
        "publish_task10_bootstrap_ledger_receipt.py"
    )
    waiter = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/wait_task10_bootstrap_ledger.py"
    )

    assert publisher.is_file()
    assert waiter.is_file()
    assert "publish_task10_bootstrap_ledger_receipt.py" in controller
    assert "wait_task10_bootstrap_ledger.py" in bootstrap
    assert bootstrap.index("publish_task10_bootstrap_ready.py") < bootstrap.index(
        "wait_task10_bootstrap_ledger.py"
    )


def test_fix2_bootstrap_ledger_publishes_and_waits_for_two_exact_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    bundle.root.mkdir()
    marker_path = bundle.root / "BOOTSTRAP_READY.json"
    ledger_path = bundle.root / "ledger/campaign-ledger.jsonl"
    completion_path = (
        bundle.root / "runtime/BOOTSTRAP_LEDGER_DURABLE.json"
    )
    marker = build_bootstrap_ready(
        unit_hashes=worker_unit_hashes(render_worker_units()),
        deadline_state=bundle.state,
        systemd_readback=_ready_readback(bundle.state),
        descriptor_file_sha256=hashlib.sha256(
            bundle.campaign_path.read_bytes()
        ).hexdigest(),
        descriptor_version_id=(
            bundle.wrapper.base_descriptor_version_id
        ),
        archive_file_sha256=bundle.wrapper.archive_identity_sha256,
        archive_version_id=(
            bundle.wrapper.repository_archive_version_id
        ),
        instance_id=bundle.observation.instance_id,
        allocation_ordinal=bundle.observation.allocation_ordinal,
        marker_publish_sequence=9,
    )
    marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")
    ledger = SkyCampaignLedger(
        ledger_path,
        run_id=RUN_ID,
        execution_deadline=bundle.wrapper.execution_deadline,
        gpu_spend_authority_sha256=(
            bundle.wrapper.task8_spend_authority_identity_sha256
        ),
    )
    ledger.transition(
        CampaignPhase.BOOTSTRAP,
        input_identities={
            "descriptor": bundle.wrapper.campaign_identity_sha256,
            "repo_tar": bundle.wrapper.archive_identity_sha256,
        },
        output_identities={
            "bootstrap_report": hashlib.sha256(
                marker_path.read_bytes()
            ).hexdigest()
        },
        gpu_allocation_sha256=bundle.wrapper.gpu_allocation_sha256,
        timestamp=datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
    )

    class S3:
        def __init__(self) -> None:
            self.objects: dict[tuple[str, str, str], dict[str, object]] = {}
            self.puts: list[dict[str, object]] = []

        def put_object(self, **values: object) -> dict[str, object]:
            self.puts.append(values)
            version = f"version-{len(self.puts)}"
            raw = bytes(values["Body"])
            self.objects[
                (str(values["Bucket"]), str(values["Key"]), version)
            ] = {
                "raw": raw,
                "metadata": dict(values["Metadata"]),
                "checksum": values["ChecksumSHA256"],
            }
            return {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "put-bootstrap-ledger",
                    "RetryAttempts": 0,
                },
                "VersionId": version,
            }

        def head_object(self, **_values: object) -> dict[str, object]:
            raise AssertionError("fresh publication must not adopt")

        def get_object(self, **values: object) -> dict[str, object]:
            stored = self.objects[
                (
                    str(values["Bucket"]),
                    str(values["Key"]),
                    str(values["VersionId"]),
                )
            ]
            return {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "get-bootstrap-ledger",
                    "RetryAttempts": 0,
                },
                "VersionId": values["VersionId"],
                "ChecksumSHA256": stored["checksum"],
                "Metadata": stored["metadata"],
                "Body": io.BytesIO(bytes(stored["raw"])),
            }

    boundary = S3()
    publisher = _load_script(
        "publish_task10_bootstrap_ledger_receipt.py",
        "task10_fix2_bootstrap_ledger_publisher",
    )
    waiter = _load_script(
        "wait_task10_bootstrap_ledger.py",
        "task10_fix2_bootstrap_ledger_waiter",
    )
    for module in (publisher, waiter):
        monkeypatch.setattr(module, "MARKER", marker_path)
        monkeypatch.setattr(module, "LEDGER", ledger_path)
        monkeypatch.setattr(module, "COMPLETION", completion_path)
        monkeypatch.setattr(module, "WRAPPER", bundle.wrapper_path)
        monkeypatch.setattr(module, "CAMPAIGN", bundle.campaign_path)

    assert publisher.main([], client=boundary) == 0
    assert len(boundary.puts) == 2
    assert "BOOTSTRAP_LEDGER_RECORD-" in str(boundary.puts[0]["Key"])
    assert str(boundary.puts[1]["Key"]).endswith(
        "/BOOTSTRAP_LEDGER.json"
    )
    assert waiter.main(
        [],
        client=boundary,
        service_active=lambda: True,
    ) == 0


def test_fix2_controller_death_between_marker_and_ledger_fails_waiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiter = _load_script(
        "wait_task10_bootstrap_ledger.py",
        "task10_fix2_dead_controller_waiter",
    )
    monkeypatch.setattr(
        waiter,
        "COMPLETION",
        tmp_path / "BOOTSTRAP_LEDGER_DURABLE.json",
    )

    with pytest.raises(SystemExit) as failure:
        waiter.main(
            [],
            client=object(),
            service_active=lambda: False,
        )

    assert failure.value.code == 70


def test_fix2_every_initial_s3_get_is_exact_owner_checksum_and_one_attempt() -> None:
    task = yaml.safe_load(render_mount_free_task(_inputs()))
    setup = task["setup"]
    bootstrap = (
        REPO_ROOT
        / "aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh"
    ).read_text()
    downloader = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/download_task10_runtime_inputs.py"
    ).read_text()
    environment = task_environment(_inputs())

    assert environment["GLM52_APPROVAL_VERSION_ID"]
    assert environment["GLM52_SUBMISSION_INTENT_VERSION_ID"]
    assert setup.count("--expected-bucket-owner 246813579024") == 2
    assert setup.count("--checksum-mode ENABLED") == 2
    assert setup.count("AWS_MAX_ATTEMPTS=1") == 2
    assert setup.count("--version-id") == 2
    assert "aws s3 cp" not in bootstrap
    assert "download_task10_runtime_inputs.py" in bootstrap
    assert 'ExpectedBucketOwner=ACCOUNT_ID' in downloader
    assert 'ChecksumMode="ENABLED"' in downloader
    assert '"total_max_attempts": 1' in downloader
    assert "GLM52_APPROVAL_VERSION_ID" in downloader
    assert "GLM52_SUBMISSION_INTENT_VERSION_ID" in downloader


def test_fix2_approval_and_intent_gets_use_exact_version_owner_and_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "download_task10_runtime_inputs.py",
        "task10_fix2_runtime_downloads",
    )
    approval = b"approval\n"
    intent = b"intent\n"
    environment = dict(task_environment(_inputs()))
    environment["GLM52_APPROVAL_FILE_SHA256"] = hashlib.sha256(
        approval
    ).hexdigest()
    environment["GLM52_SUBMISSION_INTENT_FILE_SHA256"] = hashlib.sha256(
        intent
    ).hexdigest()
    calls: list[dict[str, object]] = []

    class Client:
        def get_object(self, **values: object) -> dict[str, object]:
            calls.append(values)
            version_id = str(values["VersionId"])
            raw = (
                approval
                if version_id == environment["GLM52_APPROVAL_VERSION_ID"]
                else intent
            )
            return {
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "VersionId": version_id,
                "ChecksumSHA256": base64.b64encode(
                    hashlib.sha256(raw).digest()
                ).decode("ascii"),
                "Body": io.BytesIO(raw),
            }

    monkeypatch.setattr(module, "APPROVAL", tmp_path / "approval.json")
    monkeypatch.setattr(module, "INTENT", tmp_path / "intent.json")

    assert module.main([], client=Client(), environment=environment) == 0
    assert [call["VersionId"] for call in calls] == [
        environment["GLM52_APPROVAL_VERSION_ID"],
        environment["GLM52_SUBMISSION_INTENT_VERSION_ID"],
    ]
    assert all(
        call["ExpectedBucketOwner"] == "246813579024"
        and call["ChecksumMode"] == "ENABLED"
        for call in calls
    )
    assert module.APPROVAL.read_bytes() == approval
    assert module.INTENT.read_bytes() == intent


def _graceful_hashes() -> tuple[dict[str, str], dict[str, str]]:
    return (
        worker_unit_hashes(render_worker_units()),
        {
            "collect_task10_ssm_graceful_stop.py": _digest(
                "ssm-graceful-collector"
            ),
            "glm52_deadline_guard.py": _digest("deadline-guard"),
            "materialize_task10_graceful_stop.py": _digest(
                "graceful-materializer"
            ),
            "run_campaign.sh": _digest("campaign-runner"),
            "run_task10_production_campaign.py": _digest(
                "production-runtime"
            ),
        },
    )


def _local_graceful_evidence() -> dict[str, object]:
    unit_hashes, script_hashes = _graceful_hashes()
    return dict(
        build_graceful_stop_evidence(
            unit_hashes=unit_hashes,
            script_hashes=script_hashes,
            active_state="inactive",
            sub_state="dead",
            result="success",
            exec_main_code=1,
            exec_main_status=0,
            control_group_pids=[],
            stop_file_identity_sha256=_digest("stop"),
            checkpoint_identity_sha256=_digest("checkpoint"),
            latest_marker_identity_sha256=_digest("ledger-latest"),
            campaign_terminal_marker_identity_sha256=_digest(
                "campaign-drained"
            ),
            ssm_command_id=None,
            authority="LOCAL_TIMER",
        )
    )


def _task12_handoff(
    bundle: SimpleNamespace,
    *,
    command_id: str = "12345678-1234-1234-1234-123456789abc",
    document_version: str = "7",
    action_identity: str | None = None,
) -> tuple[object, bytes, dict[str, str]]:
    unit_hashes, script_hashes = _graceful_hashes()
    authority = build_task12_worker_drain_authority(
        worker_descriptor=bundle.wrapper,
        schema_version=1,
        record_type=(
            "glm52_task12_retained_worker_drain_authority_v1"
        ),
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        campaign_identity_sha256=(
            bundle.wrapper.campaign_identity_sha256
        ),
        activation_id=bundle.wrapper.activation_id,
        activation_ordinal=bundle.wrapper.activation_ordinal,
        generation=bundle.wrapper.generation,
        generation_text=bundle.wrapper.generation_text,
        allocation_ordinal=bundle.observation.allocation_ordinal,
        allocation_ordinal_text=(
            bundle.observation.allocation_ordinal_text
        ),
        instance_id=bundle.observation.instance_id,
        command_id=command_id,
        document_name=GRACEFUL_STOP_DOCUMENT,
        document_version=document_version,
        worker_drain_role_arn=WORKER_DRAIN_ROLE_ARN,
        worker_descriptor_body_sha256=(
            bundle.wrapper.descriptor_body_sha256
        ),
        worker_observation=asdict(bundle.observation),
        expected_unit_hashes=unit_hashes,
        expected_script_hashes=script_hashes,
        task9_launch_identity_sha256=(
            bundle.wrapper.task9_launch_identity_sha256
        ),
        task9_custody_identity_sha256=(
            bundle.wrapper.task9_custody_identity_sha256
        ),
        task12_action_identity_sha256=(
            action_identity or _digest("task12-worker-drain-action")
        ),
        task12_audit_identity_sha256=_digest(
            "task12-worker-drain-audit"
        ),
    )
    authority_raw = canonical_json_bytes(asdict(authority)) + b"\n"
    authority_key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{bundle.wrapper.generation:08d}/allocations/"
        f"{bundle.observation.allocation_ordinal:08d}/"
        "TASK12_WORKER_DRAIN_AUTHORITY.json"
    )
    handoff = build_retained_ssm_graceful_stop_handoff(
        schema_version=1,
        record_type=(
            "glm52_task10_retained_ssm_graceful_stop_handoff_v1"
        ),
        account_id="246813579024",
        region="us-west-2",
        run_id=RUN_ID,
        command_id=command_id,
        document_name=GRACEFUL_STOP_DOCUMENT,
        document_version=document_version,
        instance_id=bundle.observation.instance_id,
        allocation_ordinal=bundle.observation.allocation_ordinal,
        worker_descriptor=asdict(bundle.wrapper),
        campaign_descriptor=bundle.accepted,
        task12_authority_key=authority_key,
        task12_authority_version_id="task12-authority-version-opaque",
        task12_authority_file_sha256=hashlib.sha256(
            authority_raw
        ).hexdigest(),
        task12_authority=asdict(authority),
    )
    metadata = {
        "glm52-account-id": "246813579024",
        "glm52-activation-id": bundle.wrapper.activation_id,
        "glm52-body-sha256": authority.authority_body_sha256,
        "glm52-file-sha256": hashlib.sha256(
            authority_raw
        ).hexdigest(),
        "glm52-generation-text": bundle.wrapper.generation_text,
        "glm52-record-kind": "TASK12_WORKER_DRAIN_AUTHORITY",
        "glm52-region": "us-west-2",
        "glm52-run-id": RUN_ID,
    }
    return handoff, authority_raw, metadata


def test_parameterless_ssm_document_and_exact_role_tag_policy() -> None:
    document = render_graceful_stop_document()
    policy = render_worker_drain_iam(activation_id="activation-0001")

    assert document["description"] == "KeepGlm52GracefulStopV1"
    assert document["parameters"] == {}
    raw = str(document)
    assert "{{" not in raw
    assert "systemctl stop keep-glm52-campaign.service" in raw
    allow = policy["Statement"][0]
    assert allow["Action"] == "ssm:SendCommand"
    assert allow["Resource"][0].endswith(
        ":document/KeepGlm52GracefulStopV1"
    )
    assert allow["Condition"]["StringEquals"] == {
        "ssm:resourceTag/RunId": RUN_ID,
        "ssm:resourceTag/activation-id": "activation-0001",
    }
    assert "ssm:StartSession" in policy["Statement"][1]["Action"]


def test_graceful_stop_requires_checkpoint_service_and_empty_cgroup_evidence() -> None:
    units = worker_unit_hashes(render_worker_units())
    scripts = {
        "collect_task10_ssm_graceful_stop.py": hashlib.sha256(
            b"collector"
        ).hexdigest(),
        "glm52_deadline_guard.py": hashlib.sha256(b"guard").hexdigest(),
        "materialize_task10_graceful_stop.py": hashlib.sha256(
            b"materializer"
        ).hexdigest(),
        "run_campaign.sh": hashlib.sha256(b"run").hexdigest(),
        "run_task10_production_campaign.py": hashlib.sha256(
            b"runtime"
        ).hexdigest(),
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_worker_graceful_stop_v1",
        "run_id": RUN_ID,
        "unit_file_sha256": dict(units),
        "script_file_sha256": scripts,
        "ActiveState": "inactive",
        "SubState": "dead",
        "Result": "success",
        "ExecMainCode": 1,
        "ExecMainStatus": 0,
        "control_group_pids": [],
        "stop_file_identity_sha256": "6" * 64,
        "checkpoint_identity_sha256": "7" * 64,
        "latest_marker_identity_sha256": "8" * 64,
        "campaign_terminal_marker_identity_sha256": None,
        "ssm_command_id": None,
        "authority": "LOCAL_TIMER",
    }
    evidence = {
        **body,
        "graceful_stop_body_sha256": canonical_sha256(body),
    }
    validate_graceful_stop_evidence(
        evidence,
        expected_unit_hashes=units,
        expected_script_hashes=scripts,
    )
    assert build_graceful_stop_evidence(
        unit_hashes=units,
        script_hashes=scripts,
        active_state="inactive",
        sub_state="dead",
        result="success",
        exec_main_code=1,
        exec_main_status=0,
        control_group_pids=[],
        stop_file_identity_sha256="6" * 64,
        checkpoint_identity_sha256="7" * 64,
        latest_marker_identity_sha256="8" * 64,
        campaign_terminal_marker_identity_sha256=None,
        ssm_command_id=None,
        authority="LOCAL_TIMER",
    ) == evidence

    for field, value in (
        ("checkpoint_identity_sha256", None),
        ("ActiveState", "deactivating"),
        ("control_group_pids", [101]),
        ("ssm_command_id", "caller"),
    ):
        mutant = {**evidence, field: value}
        mutant_body = dict(mutant)
        mutant_body.pop("graceful_stop_body_sha256")
        mutant["graceful_stop_body_sha256"] = canonical_sha256(mutant_body)
        with pytest.raises(Task10WorkerError):
            validate_graceful_stop_evidence(
                mutant,
                expected_unit_hashes=units,
                expected_script_hashes=scripts,
            )


def test_local_timer_materializes_canonical_graceful_stop_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/materialize_task10_graceful_stop.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task10_graceful_stop_materializer_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stop = tmp_path / "STOP"
    checkpoint = tmp_path / "checkpoint-latest.json"
    latest = tmp_path / "ledger-latest.json"
    terminal = tmp_path / "CAMPAIGN_DRAINED.json"
    output = tmp_path / "WORKER_GRACEFUL_STOP.json"
    for path, raw in (
        (stop, b"stop\n"),
        (checkpoint, b"checkpoint\n"),
        (latest, b"latest\n"),
        (terminal, b"terminal\n"),
    ):
        path.write_bytes(raw)
    unit_hashes = {
        name: _digest(name)
        for name in UNIT_NAMES
    }
    script_hashes = {
        "collect_task10_ssm_graceful_stop.py": _digest(
            "ssm-graceful-collector"
        ),
        "glm52_deadline_guard.py": _digest("deadline-guard"),
        "materialize_task10_graceful_stop.py": _digest(
            "graceful-materializer"
        ),
        "run_campaign.sh": _digest("campaign-runner"),
        "run_task10_production_campaign.py": _digest(
            "production-runtime"
        ),
    }
    monkeypatch.setattr(module, "STOP", stop)
    monkeypatch.setattr(module, "CHECKPOINTS", (checkpoint,))
    monkeypatch.setattr(module, "LATEST", latest)
    monkeypatch.setattr(module, "TERMINAL", terminal)
    monkeypatch.setattr(module, "OUTPUT", output)
    monkeypatch.setattr(
        module,
        "_show",
        lambda: {
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
            "ControlGroup": "/system.slice/keep-glm52-campaign.service",
        },
    )
    monkeypatch.setattr(
        module,
        "_control_group_pids",
        lambda _readback: [],
    )
    monkeypatch.setattr(
        module,
        "_hashes",
        lambda: (unit_hashes, script_hashes),
    )
    monkeypatch.setattr(module, "_runtime_authority", lambda: None)
    publications: list[tuple[dict[str, object], bytes]] = []
    monkeypatch.setattr(
        module,
        "_publish_or_adopt",
        lambda evidence, raw: publications.append((evidence, raw)),
    )

    assert module.main([]) == 0
    raw = output.read_bytes()
    evidence = json.loads(raw)
    assert raw == canonical_json_bytes(evidence) + b"\n"
    assert evidence["checkpoint_identity_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    validate_graceful_stop_evidence(
        evidence,
        expected_unit_hashes=unit_hashes,
        expected_script_hashes=script_hashes,
    )
    assert publications == [(evidence, raw)]


def test_fix2_local_graceful_stop_publishes_and_exactly_adopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    local = _load_script(
        "materialize_task10_graceful_stop.py",
        "task10_fix2_local_graceful_publisher",
    )
    for name, value in (
        ("WRAPPER", bundle.wrapper_path),
        ("CAMPAIGN", bundle.campaign_path),
        ("OBSERVATION", bundle.observation_path),
        ("STATE", bundle.state_path),
        ("KEY", bundle.key_path),
    ):
        monkeypatch.setattr(local, name, value)
    evidence = _local_graceful_evidence()
    raw = canonical_json_bytes(evidence) + b"\n"
    boundary = _ExactS3()

    version_id = local._publish_or_adopt(
        evidence,
        raw,
        client=boundary,
    )
    assert version_id == "published-version-1"
    assert len(boundary.put_calls) == 1
    publication = boundary.put_calls[0]
    assert str(publication["Key"]).endswith(
        "/generations/00000001/allocations/00000001/"
        "WORKER_GRACEFUL_STOP.json"
    )
    assert publication["ExpectedBucketOwner"] == "246813579024"
    assert publication["IfNoneMatch"] == "*"
    assert local._publish_or_adopt(
        evidence,
        raw,
        client=boundary,
    ) == version_id
    assert len(boundary.put_calls) == 2
    assert len(boundary.head_calls) == 1
    assert all(
        call["ExpectedBucketOwner"] == "246813579024"
        and call["ChecksumMode"] == "ENABLED"
        and call["VersionId"] == version_id
        for call in boundary.get_calls
    )


@pytest.mark.parametrize(
    "missing_evidence",
    ["checkpoint", "systemd", "cgroup"],
)
def test_fix2_invalid_local_graceful_evidence_has_zero_puts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_evidence: str,
) -> None:
    local = _load_script(
        "materialize_task10_graceful_stop.py",
        "task10_fix2_invalid_local_" + missing_evidence,
    )
    stop = tmp_path / "STOP"
    checkpoint = tmp_path / "checkpoint.json"
    latest = tmp_path / "latest.json"
    for path in (stop, checkpoint, latest):
        path.write_bytes(path.name.encode("utf-8"))
    if missing_evidence == "checkpoint":
        checkpoint.unlink()
    monkeypatch.setattr(local, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(local, "STOP", stop)
    monkeypatch.setattr(local, "CHECKPOINTS", (checkpoint,))
    monkeypatch.setattr(local, "LATEST", latest)
    monkeypatch.setattr(local, "TERMINAL", tmp_path / "terminal.json")
    monkeypatch.setattr(local, "_runtime_authority", lambda: None)
    monkeypatch.setattr(local, "_hashes", _graceful_hashes)
    monkeypatch.setattr(
        local,
        "_show",
        lambda: {
            "ActiveState": (
                "active" if missing_evidence == "systemd" else "inactive"
            ),
            "SubState": (
                "running" if missing_evidence == "systemd" else "dead"
            ),
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
            "ControlGroup": (
                "/system.slice/keep-glm52-campaign.service"
            ),
        },
    )
    monkeypatch.setattr(
        local,
        "_control_group_pids",
        lambda _readback: (
            [4242] if missing_evidence == "cgroup" else []
        ),
    )
    puts: list[object] = []
    monkeypatch.setattr(
        local,
        "_publish_or_adopt",
        lambda *_args, **_kwargs: puts.append(object()),
    )

    with pytest.raises(SystemExit) as failure:
        local.main([])
    assert failure.value.code == 70
    assert puts == []


def test_fix2_retained_ssm_observer_requires_task12_task9_and_document_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    observer = _load_script(
        "observe_task10_ssm_graceful_stop.py",
        "task10_fix2_retained_ssm_observer",
    )
    handoff, authority_raw, authority_metadata = _task12_handoff(bundle)
    handoff_path = tmp_path / "retained-handoff.json"
    completion = tmp_path / "retained-completion.json"
    handoff_path.write_bytes(
        canonical_json_bytes(asdict(handoff)) + b"\n"
    )
    monkeypatch.setattr(observer, "HANDOFF", handoff_path)
    monkeypatch.setattr(observer, "COMPLETION", completion)
    boundary = _ExactS3()
    boundary.seed(
        bucket=str(bundle.accepted["bucket"]),
        key=handoff.task12_authority_key,
        version_id=handoff.task12_authority_version_id,
        raw=authority_raw,
        metadata=authority_metadata,
    )
    local_observation = _local_graceful_evidence()

    class Ssm:
        def get_command_invocation(
            self,
            **values: object,
        ) -> dict[str, object]:
            assert values == {
                "CommandId": handoff.command_id,
                "InstanceId": handoff.instance_id,
            }
            return {
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "CommandId": handoff.command_id,
                "InstanceId": handoff.instance_id,
                "DocumentName": handoff.document_name,
                "DocumentVersion": handoff.document_version,
                "Status": "Success",
                "StatusDetails": "Success",
                "ResponseCode": 0,
                "StandardOutputContent": (
                    canonical_json_bytes(local_observation).decode("utf-8")
                    + "\n"
                ),
                "StandardErrorContent": "",
            }

    assert observer.main(
        [],
        ssm_client=Ssm(),
        s3_client=boundary,
    ) == 0
    assert len(boundary.put_calls) == 1
    publication = boundary.put_calls[0]
    assert str(publication["Key"]).endswith(
        "/generations/00000001/allocations/00000001/"
        "WORKER_GRACEFUL_STOP.json"
    )
    published = json.loads(bytes(publication["Body"]))
    assert published["authority"] == "SSM"
    assert published["ssm_command_id"] == handoff.command_id
    assert completion.is_file()
    authority_get = boundary.get_calls[0]
    assert authority_get == {
        "Bucket": str(bundle.accepted["bucket"]),
        "Key": handoff.task12_authority_key,
        "VersionId": handoff.task12_authority_version_id,
        "ExpectedBucketOwner": "246813579024",
        "ChecksumMode": "ENABLED",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "unpublished_task12_authority",
        "document_version",
        "artifact_hash",
    ],
)
def test_fix2_retained_ssm_mutants_have_zero_graceful_puts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    observer = _load_script(
        "observe_task10_ssm_graceful_stop.py",
        "task10_fix2_retained_ssm_mutant_" + mutation,
    )
    handoff, authority_raw, authority_metadata = _task12_handoff(bundle)
    handoff_path = tmp_path / "retained-handoff.json"
    completion = tmp_path / "retained-completion.json"
    if mutation == "unpublished_task12_authority":
        handoff, _, _ = _task12_handoff(
            bundle,
            action_identity=_digest("forged-task12-action"),
        )
    handoff_path.write_bytes(
        canonical_json_bytes(asdict(handoff)) + b"\n"
    )
    monkeypatch.setattr(observer, "HANDOFF", handoff_path)
    monkeypatch.setattr(observer, "COMPLETION", completion)
    boundary = _ExactS3()
    boundary.seed(
        bucket=str(bundle.accepted["bucket"]),
        key=handoff.task12_authority_key,
        version_id=handoff.task12_authority_version_id,
        raw=authority_raw,
        metadata=authority_metadata,
    )
    local_observation = _local_graceful_evidence()
    if mutation == "artifact_hash":
        local_observation["script_file_sha256"] = {
            **dict(local_observation["script_file_sha256"]),
            "run_campaign.sh": _digest("foreign-campaign-runner"),
        }
        body = dict(local_observation)
        body.pop("graceful_stop_body_sha256")
        local_observation["graceful_stop_body_sha256"] = canonical_sha256(
            body
        )

    class Ssm:
        def get_command_invocation(
            self,
            **_values: object,
        ) -> dict[str, object]:
            return {
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "CommandId": handoff.command_id,
                "InstanceId": handoff.instance_id,
                "DocumentName": handoff.document_name,
                "DocumentVersion": (
                    "8"
                    if mutation == "document_version"
                    else handoff.document_version
                ),
                "Status": "Success",
                "StatusDetails": "Success",
                "ResponseCode": 0,
                "StandardOutputContent": (
                    canonical_json_bytes(local_observation).decode("utf-8")
                    + "\n"
                ),
                "StandardErrorContent": "",
            }

    with pytest.raises(SystemExit) as failure:
        observer.main(
            [],
            ssm_client=Ssm(),
            s3_client=boundary,
        )
    assert failure.value.code == 70
    assert boundary.put_calls == []
    assert not completion.exists()


def test_fix3_retained_ssm_foreign_sky_job_has_zero_graceful_puts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _runtime_bundle(tmp_path)
    observer = _load_script(
        "observe_task10_ssm_graceful_stop.py",
        "task10_fix3_retained_ssm_foreign_sky_job",
    )
    handoff, authority_raw, authority_metadata = _task12_handoff(bundle)

    authority = json.loads(authority_raw)
    worker_observation = dict(authority["worker_observation"])
    worker_observation["sky_job_name"] = "foreign-production-job"
    worker_observation_body = dict(worker_observation)
    worker_observation_body.pop("observation_body_sha256")
    worker_observation["observation_body_sha256"] = canonical_sha256(
        worker_observation_body
    )
    authority["worker_observation"] = worker_observation
    authority_body = dict(authority)
    authority_body.pop("authority_body_sha256")
    authority["authority_body_sha256"] = canonical_sha256(authority_body)
    foreign_authority_raw = canonical_json_bytes(authority) + b"\n"

    foreign_handoff = asdict(handoff)
    foreign_handoff["task12_authority"] = authority
    foreign_handoff["task12_authority_file_sha256"] = hashlib.sha256(
        foreign_authority_raw
    ).hexdigest()
    handoff_body = dict(foreign_handoff)
    handoff_body.pop("handoff_body_sha256")
    foreign_handoff["handoff_body_sha256"] = canonical_sha256(handoff_body)

    handoff_path = tmp_path / "retained-handoff.json"
    completion = tmp_path / "retained-completion.json"
    handoff_path.write_bytes(canonical_json_bytes(foreign_handoff) + b"\n")
    monkeypatch.setattr(observer, "HANDOFF", handoff_path)
    monkeypatch.setattr(observer, "COMPLETION", completion)

    boundary = _ExactS3()
    boundary.seed(
        bucket=str(bundle.accepted["bucket"]),
        key=handoff.task12_authority_key,
        version_id=handoff.task12_authority_version_id,
        raw=foreign_authority_raw,
        metadata={
            **authority_metadata,
            "glm52-body-sha256": authority["authority_body_sha256"],
            "glm52-file-sha256": hashlib.sha256(
                foreign_authority_raw
            ).hexdigest(),
        },
    )

    with pytest.raises(SystemExit) as failure:
        observer.main(
            [],
            ssm_client=object(),
            s3_client=boundary,
        )
    assert failure.value.code == 70
    assert boundary.get_calls == []
    assert boundary.put_calls == []
    assert not completion.exists()


def test_checked_in_units_and_parameterless_document_equal_rendered_contracts() -> None:
    production = REPO_ROOT / "aws/glm52-gpu/skypilot/production"
    checked_units = {
        name: (production / name).read_bytes()
        for name in UNIT_NAMES
    }
    assert checked_units == render_worker_units()
    document = json.loads(
        (
            REPO_ROOT
            / "aws/glm52-gpu/cloudformation/KeepGlm52GracefulStopV1.json"
        ).read_bytes()
    )
    assert document == render_graceful_stop_document()
    policy = json.loads(
        (
            REPO_ROOT
            / "aws/glm52-gpu/cloudformation/"
            "worker-drain-signal-policy.json.in"
        ).read_bytes()
    )
    assert policy == render_worker_drain_iam(
        activation_id="__ACTIVATION_ID__"
    )


def test_task10_worker_scripts_are_executable_and_syntax_clean() -> None:
    scripts = (
        REPO_ROOT / "aws/glm52-gpu/scripts/glm52_deadline_guard.py",
        REPO_ROOT / "aws/glm52-gpu/scripts/publish_task10_bootstrap_ready.py",
        REPO_ROOT / "aws/glm52-gpu/scripts/build_task10_production_task.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/authenticate_task10_worker_inputs.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/install_task10_worker_descriptors.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/download_task10_runtime_inputs.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/materialize_task10_worker_observation.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/materialize_task10_graceful_stop.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/collect_task10_ssm_graceful_stop.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/observe_task10_ssm_graceful_stop.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/publish_task10_bootstrap_ledger_receipt.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/run_task10_production_campaign.py",
        REPO_ROOT
        / "aws/glm52-gpu/scripts/wait_task10_bootstrap_ledger.py",
        REPO_ROOT / "aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh",
    )
    for script in scripts:
        assert script.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        [
            "bash",
            "-n",
            str(
                REPO_ROOT
                / "aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_run_campaign_rejects_every_caller_selected_path() -> None:
    source = (
        REPO_ROOT / "aws/glm52-gpu/scripts/run_campaign.sh"
    ).read_text()
    assert "refusing caller-selected campaign arguments" in source
    assert "DESCRIPTOR=/etc/keep-glm52/campaign.json" in source
    assert "DEADLINE_STATE=/var/lib/keep-glm52/deadline-state.json" in source
    assert 'exec 9>"$ROOT/runtime/heavy-job.lock"' in source
    assert (
        'exec /usr/bin/python3 "$KEEP_REPO_DIR/aws/glm52-gpu/scripts/'
        'run_task10_production_campaign.py"'
    ) in source
    assert "benchmarks/run_glm52_campaign.py" not in source
    deadline = (
        REPO_ROOT / "aws/glm52-gpu/scripts/glm52_deadline_guard.py"
    ).read_text()
    assert "materialize_task10_graceful_stop.py" in deadline
