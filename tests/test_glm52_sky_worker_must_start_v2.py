"""Pure dynamic-v2 worker-start authority tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from mlx_vq.quality.glm52_sky_must_start import (
    build_must_start_controller_observation,
)
from mlx_vq.quality.glm52_sky_must_start_dynamic import (
    build_dynamic_v2_job_binding,
    dynamic_v2_canonical_bytes,
    dynamic_v2_job_binding_s3_key,
)

RUN_ID = "glm52-sky-20260726"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
CONTROLLER_ROLE = "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
CONTROLLER_PROFILE = (
    "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
)
WORKER_ROLE = "arn:aws:iam::246813579024:role/keep-glm52-skypilot-worker"
CONTROLLER_INSTANCE_ID = "i-0511af4e31aa5406a"
CONTROLLER_INSTANCE_TYPE = "c6a.xlarge"
CONTROLLER_CLUSTER = "sky-jobs-controller-9d9f31a9"
WORKER_INSTANCE_ID = "i-0123456789abcdef0"
RECOVERY_INSTANCE_ID = "i-1123456789abcdef0"
SECOND_RECOVERY_INSTANCE_ID = "i-2123456789abcdef0"
AMI_ID = "ami-0123456789abcdef0"
SKY_JOB_ID = 17
SKY_JOB_NAME = f"{RUN_ID}-qualification"
WORKER_CLUSTER = "sky-glm52-worker-a"

INTENT_AT = datetime(2026, 7, 26, 12, 0, 0, tzinfo=UTC)
BASELINE_AT = INTENT_AT + timedelta(seconds=1)
ACQUIRED_AT = INTENT_AT + timedelta(seconds=2)
SUBMITTED_AT = INTENT_AT + timedelta(seconds=3)
BOUND_AT = INTENT_AT + timedelta(seconds=4)
SUBMISSION_OBSERVED_AT = INTENT_AT + timedelta(seconds=5)
PENDING_AT = INTENT_AT + timedelta(seconds=6)
ENTRYPOINT_AT = INTENT_AT + timedelta(seconds=7)
START_AT = INTENT_AT + timedelta(seconds=8)
WORKER_OBSERVED_AT = INTENT_AT + timedelta(seconds=9)
WORKER_ACCEPTED_AT = WORKER_OBSERVED_AT
MUST_START_BY = INTENT_AT + timedelta(minutes=10)


def _module() -> Any:
    try:
        return importlib.import_module("mlx_vq.quality.glm52_sky_worker_must_start_v2")
    except ModuleNotFoundError:
        pytest.fail("dynamic-v2 worker-start authority module is not implemented")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


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
    body = {name: item for name, item in value.items() if name != digest_field}
    return _self_hashed(body, digest_field)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _descriptor() -> dict[str, object]:
    artifacts = {
        "source_snapshot_prefix": "source-snapshot/",
        "source_snapshot_sha256": "1" * 64,
        "non_vq_prefix": "non-vq/",
        "non_vq_package_sha256": "2" * 64,
        "teich_pack_key": "teich/pack.json",
        "teich_pack_sha256": "3" * 64,
        "frozen_prompt_pack_key": "prompts/frozen.json",
        "frozen_prompt_pack_sha256": "4" * 64,
        "training_baseline_prefix": "training/baseline/",
        "training_baseline_sha256": "5" * 64,
        "training_config_key": "training/config.json",
        "training_config_sha256": "6" * 64,
        "artifact_inventory_key": "inventory/model.json",
        "artifact_inventory_sha256": "7" * 64,
        "qualification_cache_prefix": "qualification/cache/",
        "qualification_cache_manifest_sha256": "8" * 64,
    }
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_campaign_descriptor_v2",
        "run_id": RUN_ID,
        "campaign_identity_sha256": "",
        "account_id": ACCOUNT_ID,
        "provider": "aws",
        "region": REGION,
        "instance_type": "p5.48xlarge",
        "instance_count": 1,
        "use_spot": False,
        "max_hourly_cost_usd": 55.04,
        "approved_gpu_runtime_seconds": 86400,
        "approved_gpu_cost_usd": 1320.96,
        "must_start_by": _iso(MUST_START_BY),
        "skypilot_version": "0.13.0",
        "task_name": "glm52-campaign",
        "controller_identity": CONTROLLER_ROLE,
        "worker_identity": WORKER_ROLE,
        "vpc_name": "keep-glm52-gpu-vpc",
        "image_id": AMI_ID,
        "bucket": BUCKET,
        "jobs_bucket": BUCKET,
        "repo_tar_key": f"campaigns/{RUN_ID}/repository/repo.tar",
        "repo_tar_sha256": "9" * 64,
        "campaign_descriptor_key": (
            f"campaigns/{RUN_ID}/submissions/qualification/descriptor.json"
        ),
        "approval_key": f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
        "approval_sha256": "a" * 64,
        "artifacts": artifacts,
    }
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
    body["campaign_identity_sha256"] = _sha(_canonical(identity))
    return _self_hashed(body, "descriptor_body_sha256")


def _rebuild_descriptor(
    descriptor: dict[str, object],
    **changes: object,
) -> dict[str, object]:
    body = {
        name: item
        for name, item in descriptor.items()
        if name != "descriptor_body_sha256"
    }
    body.update(changes)
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
    body["campaign_identity_sha256"] = _sha(_canonical(identity))
    return _self_hashed(body, "descriptor_body_sha256")


def _intent(descriptor: dict[str, object]) -> dict[str, object]:
    descriptor_file_sha = _sha(_canonical(descriptor) + b"\n")
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_submission_intent_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": "b" * 64,
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "cache_seed_acceptance_key": (
            f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
            f"{'c' * 64}/QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        "cache_seed_acceptance_file_sha256": "d" * 64,
        "cache_seed_acceptance_body_sha256": "c" * 64,
        "gpu_spend_snapshot_key": (
            f"campaigns/{RUN_ID}/spend-snapshots/{'e' * 64}/GPU_SPEND_SNAPSHOT.json"
        ),
        "gpu_spend_snapshot_sha256": "f" * 64,
        "gpu_spend_snapshot_body_sha256": "e" * 64,
        "gpu_spend_ledger_tip_record_sha256": "1" * 64,
        "remaining_gpu_seconds": 86400,
        "remaining_gpu_cost_usd": 1320.96,
        "qualification_allowance_seconds": 3600,
        "qualification_allowance_cost_usd": 55.04,
        "open_allocation_count": 0,
        "qualification_submission_ready_key": (
            f"campaigns/{RUN_ID}/qualification/submission-ready/"
            f"{'2' * 64}/QUALIFICATION_SUBMISSION_READY.json"
        ),
        "qualification_submission_ready_sha256": "3" * 64,
        "qualification_submission_ready_body_sha256": "2" * 64,
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(MUST_START_BY),
        "intent_at": _iso(INTENT_AT),
    }
    return _self_hashed(body, "intent_body_sha256")


def _intent_key(intent: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/submissions/qualification/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )


def _baseline(
    intent: dict[str, object],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_controller_baseline_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "bucket": BUCKET,
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "intent_key": _intent_key(intent),
        "intent_file_sha256": _sha(_canonical(intent) + b"\n"),
        "intent_body_sha256": intent["intent_body_sha256"],
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_instance_id": CONTROLLER_INSTANCE_ID,
        "controller_instance_type": CONTROLLER_INSTANCE_TYPE,
        "controller_profile_arn": CONTROLLER_PROFILE,
        "controller_cluster_name": CONTROLLER_CLUSTER,
        "ssm_ping_status": "Online",
        "exact_name_history": [],
        "active_exact_name_job_ids": [],
        "active_tagged_p5_instance_ids": [],
        "observed_at": _iso(BASELINE_AT),
    }
    return _self_hashed(body, "baseline_body_sha256")


def _baseline_key(baseline: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )


def _acquisition(
    intent: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_submission_acquired_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": "qualification",
        "descriptor_key": intent["descriptor_key"],
        "descriptor_file_sha256": intent["descriptor_file_sha256"],
        "descriptor_body_sha256": intent["descriptor_body_sha256"],
        "qualification_submission_ready_key": (
            intent["qualification_submission_ready_key"]
        ),
        "qualification_submission_ready_sha256": (
            intent["qualification_submission_ready_sha256"]
        ),
        "qualification_submission_ready_body_sha256": (
            intent["qualification_submission_ready_body_sha256"]
        ),
        "intent_key": _intent_key(intent),
        "intent_file_sha256": _sha(_canonical(intent) + b"\n"),
        "intent_body_sha256": intent["intent_body_sha256"],
        "controller_baseline_key": _baseline_key(baseline),
        "controller_baseline_file_sha256": _sha(_canonical(baseline) + b"\n"),
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        "must_start_control_plane_ready_key": (
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/control-plane-ready/"
            f"{'4' * 64}/CONTROL_PLANE_READY.json"
        ),
        "must_start_control_plane_ready_file_sha256": "5" * 64,
        "must_start_control_plane_ready_body_sha256": "4" * 64,
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": intent["must_start_by"],
        "acquired_at": _iso(ACQUIRED_AT),
    }
    return _self_hashed(body, "acquisition_body_sha256")


def _acquisition_key(intent: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/submissions/qualification/acquisitions/"
        f"{intent['descriptor_file_sha256']}/SUBMISSION_ACQUIRED.json"
    )


def _controller_row(
    *,
    status: str = "PENDING",
) -> dict[str, object]:
    return {
        "sky_job_id": SKY_JOB_ID,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": _iso(SUBMITTED_AT),
        "controller_status": status,
        "controller_identity": CONTROLLER_ROLE,
    }


def _binding(
    descriptor: dict[str, object],
    intent: dict[str, object],
    baseline: dict[str, object],
    acquisition: dict[str, object],
) -> dict[str, object]:
    return build_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=str(descriptor["controller_identity"]),
        current_exact_name_history=[_controller_row()],
        bound_at=BOUND_AT,
        now=SUBMISSION_OBSERVED_AT,
    )


def _observation(
    descriptor: dict[str, object],
    intent: dict[str, object],
    *,
    status: str,
    start_at: datetime | None,
    worker_cluster_name: str | None,
    recovery_count: int,
    observed_at: datetime,
) -> dict[str, object]:
    return build_must_start_controller_observation(
        run_id=RUN_ID,
        managed_mode="qualification",
        account_id=ACCOUNT_ID,
        region=REGION,
        bucket=BUCKET,
        descriptor_body_sha256=str(descriptor["descriptor_body_sha256"]),
        submission_body_sha256=str(intent["intent_body_sha256"]),
        sky_job_name=SKY_JOB_NAME,
        must_start_by=str(intent["must_start_by"]),
        target_job_id=SKY_JOB_ID,
        workspace="default",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE,
        controller_cluster_name=CONTROLLER_CLUSTER,
        status=status,
        schedule_state="ALIVE",
        submitted_at=SUBMITTED_AT,
        start_at=start_at,
        worker_cluster_name=worker_cluster_name,
        recovery_count=recovery_count,
        observed_at=observed_at,
    )


def _submission_accepted(
    intent: dict[str, object],
    binding: dict[str, object],
    submission_observation: dict[str, object],
) -> dict[str, object]:
    observation_key = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}/observations/"
        f"{submission_observation['observation_body_sha256']}.json"
    )
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_submission_accepted_v2",
        "intent_body_sha256": intent["intent_body_sha256"],
        "sky_job_id": SKY_JOB_ID,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": _iso(SUBMITTED_AT),
        "controller_status": "PENDING",
        "controller_identity": CONTROLLER_ROLE,
        "controller_instance_id": CONTROLLER_INSTANCE_ID,
        "controller_instance_type": CONTROLLER_INSTANCE_TYPE,
        "controller_profile_arn": CONTROLLER_PROFILE,
        "controller_cluster_name": CONTROLLER_CLUSTER,
        "controller_observed_at": _iso(SUBMISSION_OBSERVED_AT),
        "controller_observation_key": observation_key,
        "controller_observation_file_sha256": _sha(_canonical(submission_observation)),
        "controller_observation_body_sha256": (
            submission_observation["observation_body_sha256"]
        ),
        "controller_observation": submission_observation,
        "job_binding_key": dynamic_v2_job_binding_s3_key(intent=intent),
        "job_binding_file_sha256": _sha(dynamic_v2_canonical_bytes(binding)),
        "job_binding_body_sha256": binding["job_binding_body_sha256"],
        "job_binding": binding,
        "accepted_at": _iso(
            max(
                datetime.fromisoformat(str(binding["bound_at"]).replace("Z", "+00:00")),
                datetime.fromisoformat(
                    str(submission_observation["observed_at"]).replace(
                        "Z",
                        "+00:00",
                    )
                ),
            )
        ),
    }
    return _self_hashed(body, "accepted_body_sha256")


def _submission_accepted_key(accepted: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/submissions/qualification/accepted/"
        f"{accepted['accepted_body_sha256']}/SKYPILOT_SUBMISSION_ACCEPTED.json"
    )


def _worker_observation_key(
    intent: dict[str, object],
    observation: dict[str, object],
    instance_id: str,
) -> str:
    return (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{intent['intent_body_sha256']}/worker-controller-observations/"
        f"{instance_id}/{observation['observation_body_sha256']}.json"
    )


def _worker_instance(
    *,
    instance_id: str = WORKER_INSTANCE_ID,
    cluster_name: str = WORKER_CLUSTER,
    launch_time: datetime = PENDING_AT,
    observed_at: datetime = WORKER_OBSERVED_AT,
) -> dict[str, object]:
    return {
        "worker_cluster_name": cluster_name,
        "instance_id": instance_id,
        "worker_instance_type": "p5.48xlarge",
        "worker_image_id": AMI_ID,
        "worker_instance_lifecycle": None,
        "worker_instance_state": "running",
        "worker_instance_launch_time": _iso(launch_time),
        "worker_ray_cluster_name": cluster_name,
        "worker_skypilot_cluster_name": cluster_name,
        "worker_campaign_tags": {
            "project": "keep-glm52",
            "owner": "jack.mazac",
            "model": "glm-5.2",
            "campaign-run-id": RUN_ID,
            "cost-allocation": "glm52-sky-campaign",
        },
        "ec2_observed_at": _iso(observed_at),
    }


def _upstream() -> dict[str, object]:
    descriptor = _descriptor()
    intent = _intent(descriptor)
    baseline = _baseline(intent)
    acquisition = _acquisition(intent, baseline)
    binding = _binding(descriptor, intent, baseline, acquisition)
    submission_observation = _observation(
        descriptor,
        intent,
        status="PENDING",
        start_at=None,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at=SUBMISSION_OBSERVED_AT,
    )
    accepted = _submission_accepted(intent, binding, submission_observation)
    return {
        "descriptor": descriptor,
        "intent": intent,
        "baseline": baseline,
        "acquisition": acquisition,
        "binding": binding,
        "submission_observation": submission_observation,
        "submission_accepted": accepted,
    }


def _latch(
    module: Any,
    upstream: dict[str, object],
    *,
    instance_id: str = WORKER_INSTANCE_ID,
    job_id: object = str(SKY_JOB_ID),
    pending_at: datetime = PENDING_AT,
    entrypoint_at: datetime = ENTRYPOINT_AT,
) -> dict[str, object]:
    return module.build_worker_start_latch_v2(
        descriptor=upstream["descriptor"],
        intent=upstream["intent"],
        controller_injected_sky_job_id=job_id,
        instance_id=instance_id,
        instance_type="p5.48xlarge",
        image_id=AMI_ID,
        worker_role_arn=WORKER_ROLE,
        instance_identity_document_sha256="6" * 64,
        ec2_pending_time=pending_at,
        entrypoint_observed_at=entrypoint_at,
    )


def _acceptance_kwargs(
    module: Any,
    upstream: dict[str, object],
    *,
    latch: dict[str, object] | None = None,
    instance_id: str = WORKER_INSTANCE_ID,
    worker_cluster: str = WORKER_CLUSTER,
    worker_observation: dict[str, object] | None = None,
    recovery_count: int = 0,
    last_modified: datetime = ENTRYPOINT_AT,
    accepted_at: datetime | str | None = None,
    now: datetime | str | None = None,
    prior_acceptances: list[dict[str, object]] | None = None,
    prior_latches: list[dict[str, object]] | None = None,
    prior_observations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    intent = upstream["intent"]
    baseline = upstream["baseline"]
    acquisition = upstream["acquisition"]
    binding = upstream["binding"]
    accepted = upstream["submission_accepted"]
    assert isinstance(intent, dict)
    assert isinstance(baseline, dict)
    assert isinstance(acquisition, dict)
    assert isinstance(binding, dict)
    assert isinstance(accepted, dict)
    if latch is None:
        latch = _latch(module, upstream, instance_id=instance_id)
    if worker_observation is None:
        worker_observation = _observation(
            upstream["descriptor"],
            intent,
            status="STARTING" if recovery_count == 0 else "RECOVERING",
            start_at=START_AT,
            worker_cluster_name=worker_cluster,
            recovery_count=recovery_count,
            observed_at=WORKER_OBSERVED_AT,
        )
    worker_observed_at = datetime.fromisoformat(
        str(worker_observation["observed_at"]).replace("Z", "+00:00")
    )
    worker_launch_time = datetime.fromisoformat(
        str(latch["ec2_pending_time"]).replace("Z", "+00:00")
    )
    deterministic_accepted_at = max(last_modified, worker_observed_at)
    accepted_at = deterministic_accepted_at if accepted_at is None else accepted_at
    now = deterministic_accepted_at if now is None else now
    return {
        "descriptor": upstream["descriptor"],
        "intent": intent,
        "controller_baseline": baseline,
        "controller_baseline_key": _baseline_key(baseline),
        "controller_baseline_file_sha256": _sha(_canonical(baseline) + b"\n"),
        "submission_acquisition": acquisition,
        "submission_acquisition_key": _acquisition_key(intent),
        "submission_acquisition_file_sha256": _sha(_canonical(acquisition) + b"\n"),
        "submission_accepted": accepted,
        "submission_accepted_key": _submission_accepted_key(accepted),
        "submission_accepted_file_sha256": _sha(_canonical(accepted) + b"\n"),
        "job_binding": binding,
        "job_binding_key": dynamic_v2_job_binding_s3_key(intent=intent),
        "job_binding_file_sha256": _sha(dynamic_v2_canonical_bytes(binding)),
        "submission_controller_observation": upstream["submission_observation"],
        "submission_controller_observation_key": accepted["controller_observation_key"],
        "submission_controller_observation_file_sha256": accepted[
            "controller_observation_file_sha256"
        ],
        "worker_controller_observation": worker_observation,
        "worker_controller_observation_key": _worker_observation_key(
            intent, worker_observation, instance_id
        ),
        "worker_controller_observation_file_sha256": _sha(
            _canonical(worker_observation)
        ),
        "worker_latch": latch,
        "worker_latch_key": module.worker_start_latch_v2_s3_key(worker_latch=latch),
        "worker_latch_file_sha256": _sha(module.worker_v2_canonical_bytes(latch)),
        "worker_latch_version_id": "v1-worker-latch",
        "worker_latch_etag": '"0123456789abcdef0123456789abcdef"',
        "worker_latch_last_modified": last_modified,
        "worker_instance": _worker_instance(
            instance_id=instance_id,
            cluster_name=worker_cluster,
            launch_time=worker_launch_time,
            observed_at=worker_observed_at,
        ),
        "active_campaign_p5_instance_ids": [instance_id],
        "prior_worker_acceptance_chain": (
            [] if prior_acceptances is None else prior_acceptances
        ),
        "prior_worker_latch_chain": [] if prior_latches is None else prior_latches,
        "prior_worker_controller_observation_chain": (
            [] if prior_observations is None else prior_observations
        ),
        "accepted_at": accepted_at,
        "now": now,
    }


def _latch_evidence(
    module: Any,
    latch: dict[str, object],
    *,
    last_modified: datetime,
) -> dict[str, object]:
    return {
        "worker_latch": latch,
        "worker_latch_key": module.worker_start_latch_v2_s3_key(worker_latch=latch),
        "worker_latch_file_sha256": _sha(module.worker_v2_canonical_bytes(latch)),
        "worker_latch_version_id": "v1-worker-latch",
        "worker_latch_etag": '"0123456789abcdef0123456789abcdef"',
        "worker_latch_last_modified": _iso(last_modified),
    }


def _observation_evidence(
    upstream: dict[str, object],
    observation: dict[str, object],
    instance_id: str,
) -> dict[str, object]:
    intent = upstream["intent"]
    assert isinstance(intent, dict)
    return {
        "worker_controller_observation": observation,
        "worker_controller_observation_key": _worker_observation_key(
            intent, observation, instance_id
        ),
        "worker_controller_observation_file_sha256": _sha(_canonical(observation)),
        "worker_controller_observation_body_sha256": observation[
            "observation_body_sha256"
        ],
    }


def test_latch_exact_round_trip_and_key() -> None:
    module = _module()
    upstream = _upstream()
    latch = _latch(module, upstream)
    expected_fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "sky_job_name",
        "must_start_by",
        "repo_tar_sha256",
        "controller_injected_sky_job_id",
        "instance_id",
        "instance_type",
        "image_id",
        "worker_role_arn",
        "instance_identity_document_sha256",
        "ec2_pending_time",
        "entrypoint_observed_at",
        "worker_latch_body_sha256",
    }
    assert set(latch) == expected_fields
    body = dict(latch)
    digest = body.pop("worker_latch_body_sha256")
    assert digest == _sha(_canonical(body))
    assert module.worker_v2_canonical_bytes(latch) == _canonical(latch)
    assert not module.worker_v2_canonical_bytes(latch).endswith(b"\n")
    assert (
        module.validate_worker_start_latch_v2(
            latch,
            descriptor=upstream["descriptor"],
            intent=upstream["intent"],
        )
        == latch
    )
    assert module.worker_start_latch_v2_s3_key(worker_latch=latch) == (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{latch['intent_body_sha256']}/worker-latches/{WORKER_INSTANCE_ID}/"
        f"{digest}.json"
    )


@pytest.mark.parametrize(
    "job_id",
    [None, True, 0, -1, "0", "-1", "17.0", "seventeen", ""],
)
def test_latch_rejects_invalid_controller_injected_job_id(job_id: object) -> None:
    module = _module()
    with pytest.raises(module.WorkerStartValidationError):
        _latch(module, _upstream(), job_id=job_id)


def test_latch_translates_huge_decimal_job_id_to_validation_error() -> None:
    module = _module()
    with pytest.raises(module.WorkerStartValidationError):
        _latch(module, _upstream(), job_id="9" * 5000)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_count", True),
        ("instance_count", 1.0),
        ("use_spot", 0),
        ("use_spot", 0.0),
    ],
)
def test_latch_requires_exact_descriptor_scalar_types(
    field: str,
    value: object,
) -> None:
    module = _module()
    descriptor = _rebuild_descriptor(_descriptor(), **{field: value})
    intent = _intent(descriptor)
    with pytest.raises(module.WorkerStartValidationError):
        _latch(
            module,
            {
                "descriptor": descriptor,
                "intent": intent,
            },
        )


@pytest.mark.parametrize(
    ("target", "field", "foreign"),
    [
        ("descriptor", "account_id", "135792468013"),
        ("descriptor", "region", "us-east-1"),
        ("descriptor", "instance_type", "p4d.24xlarge"),
        ("descriptor", "image_id", "ami-1123456789abcdef0"),
        ("descriptor", "worker_identity", CONTROLLER_ROLE),
        ("intent", "repo_tar_sha256", "0" * 64),
        ("intent", "sky_job_name", "foreign-qualification"),
        ("intent", "must_start_by", "2026-07-26T12:11:00Z"),
    ],
)
def test_latch_rejects_foreign_descriptor_or_intent(
    target: str,
    field: str,
    foreign: object,
) -> None:
    module = _module()
    upstream = _upstream()
    record = deepcopy(upstream[target])
    assert isinstance(record, dict)
    digest_field = (
        "descriptor_body_sha256" if target == "descriptor" else "intent_body_sha256"
    )
    record[field] = foreign
    upstream[target] = _rehash(record, digest_field)
    with pytest.raises(module.WorkerStartValidationError):
        _latch(module, upstream)


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("instance_id", "i-short"),
        ("instance_type", "p4d.24xlarge"),
        ("image_id", "ami-1123456789abcdef0"),
        ("worker_role_arn", CONTROLLER_ROLE),
    ],
)
def test_latch_rejects_foreign_worker_identity(
    field: str,
    foreign: object,
) -> None:
    module = _module()
    upstream = _upstream()
    kwargs = {
        "descriptor": upstream["descriptor"],
        "intent": upstream["intent"],
        "controller_injected_sky_job_id": "17",
        "instance_id": WORKER_INSTANCE_ID,
        "instance_type": "p5.48xlarge",
        "image_id": AMI_ID,
        "worker_role_arn": WORKER_ROLE,
        "instance_identity_document_sha256": "6" * 64,
        "ec2_pending_time": PENDING_AT,
        "entrypoint_observed_at": ENTRYPOINT_AT,
    }
    kwargs[field] = foreign
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_latch_v2(**kwargs)


def test_latch_rejects_pending_after_entrypoint() -> None:
    module = _module()
    with pytest.raises(module.WorkerStartValidationError):
        _latch(
            module,
            _upstream(),
            pending_at=ENTRYPOINT_AT + timedelta(microseconds=1),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ec2_pending_time", "2026-07-26T05:00:06-07:00"),
        ("entrypoint_observed_at", "2026-07-26T05:00:07-07:00"),
    ],
)
def test_latch_builder_rejects_noncanonical_timestamp_strings(
    field: str,
    value: object,
) -> None:
    module = _module()
    upstream = _upstream()
    kwargs: dict[str, object] = {
        "descriptor": upstream["descriptor"],
        "intent": upstream["intent"],
        "controller_injected_sky_job_id": "17",
        "instance_id": WORKER_INSTANCE_ID,
        "instance_type": "p5.48xlarge",
        "image_id": AMI_ID,
        "worker_role_arn": WORKER_ROLE,
        "instance_identity_document_sha256": "6" * 64,
        "ec2_pending_time": PENDING_AT,
        "entrypoint_observed_at": ENTRYPOINT_AT,
    }
    kwargs[field] = value
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_latch_v2(**kwargs)


def test_latch_builder_normalizes_aware_datetime_objects() -> None:
    module = _module()
    upstream = _upstream()
    pacific = timezone(timedelta(hours=-7))
    latch = module.build_worker_start_latch_v2(
        descriptor=upstream["descriptor"],
        intent=upstream["intent"],
        controller_injected_sky_job_id="17",
        instance_id=WORKER_INSTANCE_ID,
        instance_type="p5.48xlarge",
        image_id=AMI_ID,
        worker_role_arn=WORKER_ROLE,
        instance_identity_document_sha256="6" * 64,
        ec2_pending_time=PENDING_AT.astimezone(pacific),
        entrypoint_observed_at=ENTRYPOINT_AT.astimezone(pacific),
    )
    assert latch["ec2_pending_time"] == _iso(PENDING_AT)
    assert latch["entrypoint_observed_at"] == _iso(ENTRYPOINT_AT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("instance_id"),
        lambda value: value.__setitem__("legacy_submission_body_sha256", "1" * 64),
        lambda value: value.__setitem__("schema_version", 1),
        lambda value: value.__setitem__(
            "record_type", "glm52_sky_worker_start_latch_v1"
        ),
        lambda value: value.__setitem__(
            "ec2_pending_time", "2026-07-26T05:00:06-07:00"
        ),
        lambda value: value.__setitem__("entrypoint_observed_at", float("nan")),
        lambda value: value.__setitem__("worker_latch_body_sha256", "0" * 64),
    ],
)
def test_latch_rejects_schema_time_nonfinite_or_digest_drift(mutation: Any) -> None:
    module = _module()
    upstream = _upstream()
    latch = _latch(module, upstream)
    mutation(latch)
    with pytest.raises(module.WorkerStartValidationError):
        module.validate_worker_start_latch_v2(
            latch,
            descriptor=upstream["descriptor"],
            intent=upstream["intent"],
        )


def test_latch_key_rejects_a_coherently_rehashed_invalid_intent_digest() -> None:
    module = _module()
    latch = _latch(module, _upstream())
    latch["intent_body_sha256"] = "not-a-sha"
    latch = _rehash(latch, "worker_latch_body_sha256")
    with pytest.raises(module.WorkerStartValidationError):
        module.worker_start_latch_v2_s3_key(worker_latch=latch)


def test_initial_acceptance_exact_round_trip_and_separate_observations() -> None:
    module = _module()
    upstream = _upstream()
    kwargs = _acceptance_kwargs(module, upstream)
    accepted = module.build_worker_start_accepted_v2(**kwargs)
    assert upstream["submission_observation"]["status"] == "PENDING"
    assert upstream["submission_observation"]["worker_cluster_name"] is None
    assert kwargs["worker_controller_observation"]["status"] == "STARTING"
    assert kwargs["worker_controller_observation"]["worker_cluster_name"] == (
        WORKER_CLUSTER
    )
    expected_fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "intent_key",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_key",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "submission_acquisition_key",
        "submission_acquisition_file_sha256",
        "submission_acquisition_body_sha256",
        "submission_accepted_key",
        "submission_accepted_file_sha256",
        "submission_accepted_body_sha256",
        "job_binding_key",
        "job_binding_file_sha256",
        "job_binding_body_sha256",
        "worker_controller_observation_key",
        "worker_controller_observation_file_sha256",
        "worker_controller_observation_body_sha256",
        "sky_job_id",
        "sky_job_name",
        "worker_cluster_name",
        "recovery_count",
        "worker_instance_type",
        "worker_image_id",
        "worker_instance_lifecycle",
        "worker_instance_state",
        "worker_instance_launch_time",
        "worker_ray_cluster_name",
        "worker_skypilot_cluster_name",
        "worker_campaign_tags",
        "active_campaign_p5_instance_ids",
        "ec2_observed_at",
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_body_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
        "instance_id",
        "acceptance_kind",
        "prior_worker_acceptance_key",
        "prior_worker_acceptance_file_sha256",
        "prior_worker_acceptance_body_sha256",
        "accepted_at",
        "worker_acceptance_body_sha256",
    }
    assert set(accepted) == expected_fields
    body = dict(accepted)
    digest = body.pop("worker_acceptance_body_sha256")
    assert digest == _sha(_canonical(body))
    assert accepted["acceptance_kind"] == "initial"
    assert accepted["prior_worker_acceptance_key"] is None
    assert accepted["prior_worker_acceptance_file_sha256"] is None
    assert accepted["prior_worker_acceptance_body_sha256"] is None
    assert accepted["accepted_at"] == _iso(WORKER_OBSERVED_AT)
    assert (
        module.validate_worker_start_accepted_v2(
            accepted,
            **kwargs,
        )
        == accepted
    )
    assert module.worker_start_accepted_v2_s3_key(worker_start_accepted=accepted) == (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{accepted['intent_body_sha256']}/worker-acceptances/"
        f"{WORKER_INSTANCE_ID}/{accepted['worker_latch_body_sha256']}/"
        "WORKER_START_ACCEPTED.json"
    )
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=accepted,
        ).action
        == "idempotent-complete"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controller_instance_id", "i-3123456789abcdef0"),
        ("controller_instance_type", "c6a.large"),
        ("controller_cluster_name", "sky-jobs-controller-foreign"),
    ],
)
def test_submission_observation_controller_coordinates_match_baseline(
    field: str,
    value: str,
) -> None:
    module = _module()
    upstream = _upstream()
    descriptor = upstream["descriptor"]
    intent = upstream["intent"]
    submission_observation = upstream["submission_observation"]
    assert isinstance(descriptor, dict)
    assert isinstance(intent, dict)
    assert isinstance(submission_observation, dict)
    baseline = deepcopy(upstream["baseline"])
    assert isinstance(baseline, dict)
    baseline[field] = value
    baseline = _rehash(baseline, "baseline_body_sha256")
    acquisition = _acquisition(intent, baseline)
    binding = _binding(descriptor, intent, baseline, acquisition)
    upstream.update(
        {
            "baseline": baseline,
            "acquisition": acquisition,
            "binding": binding,
            "submission_accepted": _submission_accepted(
                intent,
                binding,
                submission_observation,
            ),
        }
    )
    decision = module.decide_worker_start_acceptance_v2(
        **_acceptance_kwargs(module, upstream),
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_binding_cannot_follow_submission_observation() -> None:
    module = _module()
    upstream = _upstream()
    descriptor = upstream["descriptor"]
    intent = upstream["intent"]
    baseline = upstream["baseline"]
    acquisition = upstream["acquisition"]
    submission_observation = upstream["submission_observation"]
    assert isinstance(descriptor, dict)
    assert isinstance(intent, dict)
    assert isinstance(baseline, dict)
    assert isinstance(acquisition, dict)
    assert isinstance(submission_observation, dict)
    late_bound_at = SUBMISSION_OBSERVED_AT + timedelta(microseconds=1)
    binding = build_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=str(descriptor["controller_identity"]),
        current_exact_name_history=[_controller_row()],
        bound_at=late_bound_at,
        now=late_bound_at,
    )
    upstream["binding"] = binding
    upstream["submission_accepted"] = _submission_accepted(
        intent,
        binding,
        submission_observation,
    )
    decision = module.decide_worker_start_acceptance_v2(
        **_acceptance_kwargs(module, upstream),
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_malformed_baseline_history_uses_worker_validation_error() -> None:
    module = _module()
    upstream = _upstream()
    intent = upstream["intent"]
    submission_observation = upstream["submission_observation"]
    assert isinstance(intent, dict)
    assert isinstance(submission_observation, dict)
    baseline = deepcopy(upstream["baseline"])
    acquisition = deepcopy(upstream["acquisition"])
    binding = deepcopy(upstream["binding"])
    assert isinstance(baseline, dict)
    assert isinstance(acquisition, dict)
    assert isinstance(binding, dict)
    baseline["exact_name_history"] = [None]
    baseline = _rehash(baseline, "baseline_body_sha256")
    acquisition.update(
        {
            "controller_baseline_key": _baseline_key(baseline),
            "controller_baseline_file_sha256": _sha(_canonical(baseline) + b"\n"),
            "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        }
    )
    acquisition = _rehash(acquisition, "acquisition_body_sha256")
    binding.update(
        {
            "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
            "acquisition_body_sha256": acquisition["acquisition_body_sha256"],
        }
    )
    binding = _rehash(binding, "job_binding_body_sha256")
    upstream.update(
        {
            "baseline": baseline,
            "acquisition": acquisition,
            "binding": binding,
            "submission_accepted": _submission_accepted(
                intent,
                binding,
                submission_observation,
            ),
        }
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(
            **_acceptance_kwargs(module, upstream),
        )


def test_realized_worker_observation_cannot_use_submitter_prefix() -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    observation = kwargs["worker_controller_observation"]
    assert isinstance(observation, dict)
    kwargs["worker_controller_observation_key"] = (
        f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
        f"{kwargs['intent']['intent_body_sha256']}/observations/"
        f"{observation['observation_body_sha256']}.json"
    )
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_worker_waits_only_for_valid_unrealized_controller_authority() -> None:
    module = _module()
    upstream = _upstream()
    unrealized = _observation(
        upstream["descriptor"],
        upstream["intent"],
        status="PENDING",
        start_at=None,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at=WORKER_OBSERVED_AT,
    )
    kwargs = _acceptance_kwargs(
        module,
        upstream,
        worker_observation=unrealized,
    )
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "wait-for-authority"
    assert decision.reason


def test_build_translates_unrealized_worker_to_public_validation_error() -> None:
    module = _module()
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(
            **_unrealized_wait_kwargs(module),
        )


def test_validate_translates_unrealized_worker_to_public_validation_error() -> None:
    module = _module()
    accepted = module.build_worker_start_accepted_v2(
        **_acceptance_kwargs(module, _upstream()),
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.validate_worker_start_accepted_v2(
            accepted,
            **_unrealized_wait_kwargs(module),
        )


def _unrealized_wait_kwargs(module: Any) -> dict[str, object]:
    upstream = _upstream()
    unrealized = _observation(
        upstream["descriptor"],
        upstream["intent"],
        status="PENDING",
        start_at=None,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at=WORKER_OBSERVED_AT,
    )
    return _acceptance_kwargs(
        module,
        upstream,
        worker_observation=unrealized,
    )


def test_wait_rejects_forged_worker_observation_key() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["worker_controller_observation_key"] = "forged"
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_forged_worker_observation_file_hash() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["worker_controller_observation_file_sha256"] = "0" * 64
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_malformed_now() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["now"] = "2026-07-26T05:00:09-07:00"
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_nondeterministic_accepted_at() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["accepted_at"] = WORKER_ACCEPTED_AT + timedelta(microseconds=1)
    kwargs["now"] = WORKER_ACCEPTED_AT + timedelta(seconds=1)
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_invalid_prior_chain() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["prior_worker_acceptance_chain"] = [{}]
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_malformed_ec2_mapping() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["worker_instance"].pop("instance_id")
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_duplicate_active_campaign_p5() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    kwargs["active_campaign_p5_instance_ids"] = [
        WORKER_INSTANCE_ID,
        RECOVERY_INSTANCE_ID,
    ]
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_wait_rejects_malformed_existing_acceptance() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance={},
        ).action
        == "fail-closed"
    )


def test_wait_rejects_valid_existing_acceptance_without_realized_candidate() -> None:
    module = _module()
    existing = module.build_worker_start_accepted_v2(
        **_acceptance_kwargs(module, _upstream()),
    )
    kwargs = _unrealized_wait_kwargs(module)
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=existing,
        ).action
        == "fail-closed"
    )


def test_wait_does_not_join_ec2_cluster_fields_to_missing_controller_cluster() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    snapshot = kwargs["worker_instance"]
    assert isinstance(snapshot, dict)
    snapshot["worker_cluster_name"] = "ec2-cluster-already-tagged"
    snapshot["worker_ray_cluster_name"] = "ec2-cluster-already-tagged"
    snapshot["worker_skypilot_cluster_name"] = "ec2-cluster-already-tagged"
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "wait-for-authority"
    )


def test_wait_rejects_internally_inconsistent_ec2_cluster_names() -> None:
    module = _module()
    kwargs = _unrealized_wait_kwargs(module)
    snapshot = kwargs["worker_instance"]
    assert isinstance(snapshot, dict)
    snapshot["worker_ray_cluster_name"] = "foreign-ray-cluster"
    assert (
        module.decide_worker_start_acceptance_v2(
            **kwargs,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_acceptance_requires_exact_evidence_derived_timestamp() -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    kwargs["accepted_at"] = WORKER_ACCEPTED_AT + timedelta(microseconds=1)
    kwargs["now"] = WORKER_ACCEPTED_AT + timedelta(seconds=1)
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


@pytest.mark.parametrize("field", ["accepted_at", "now"])
def test_acceptance_rejects_noncanonical_explicit_timestamp_strings(
    field: str,
) -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    kwargs[field] = "2026-07-26T05:00:09-07:00"
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


def test_independent_same_evidence_candidates_are_byte_identical() -> None:
    module = _module()
    first_kwargs = _acceptance_kwargs(module, _upstream())
    second_kwargs = deepcopy(first_kwargs)
    second_kwargs["accepted_at"] = _iso(WORKER_ACCEPTED_AT)
    second_kwargs["now"] = WORKER_ACCEPTED_AT + timedelta(seconds=30)
    first = module.build_worker_start_accepted_v2(**first_kwargs)
    second = module.build_worker_start_accepted_v2(**second_kwargs)
    assert first == second
    assert module.worker_v2_canonical_bytes(first) == module.worker_v2_canonical_bytes(
        second
    )
    assert (
        module.decide_worker_start_acceptance_v2(
            **second_kwargs,
            existing_worker_acceptance=first,
        ).action
        == "idempotent-complete"
    )


def test_realized_worker_observation_cannot_predate_submission_observation() -> None:
    module = _module()
    upstream = _upstream()
    pending = SUBMITTED_AT + timedelta(milliseconds=100)
    entrypoint = SUBMITTED_AT + timedelta(milliseconds=200)
    current_observation = _observation(
        upstream["descriptor"],
        upstream["intent"],
        status="STARTING",
        start_at=SUBMITTED_AT + timedelta(milliseconds=300),
        worker_cluster_name=WORKER_CLUSTER,
        recovery_count=0,
        observed_at=BOUND_AT,
    )
    latch = _latch(
        module,
        upstream,
        pending_at=pending,
        entrypoint_at=entrypoint,
    )
    kwargs = _acceptance_kwargs(
        module,
        upstream,
        latch=latch,
        worker_observation=current_observation,
        last_modified=entrypoint,
    )
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_initial_boundary_times_equal_deadline_pass() -> None:
    module = _module()
    upstream = _upstream()
    latch = _latch(
        module,
        upstream,
        pending_at=MUST_START_BY,
        entrypoint_at=MUST_START_BY,
    )
    worker_observation = _observation(
        upstream["descriptor"],
        upstream["intent"],
        status="STARTING",
        start_at=MUST_START_BY,
        worker_cluster_name=WORKER_CLUSTER,
        recovery_count=0,
        observed_at=MUST_START_BY + timedelta(seconds=1),
    )
    kwargs = _acceptance_kwargs(
        module,
        upstream,
        latch=latch,
        worker_observation=worker_observation,
        last_modified=MUST_START_BY,
        accepted_at=MUST_START_BY + timedelta(seconds=1),
        now=MUST_START_BY + timedelta(seconds=2),
    )
    assert module.build_worker_start_accepted_v2(**kwargs)["acceptance_kind"] == (
        "initial"
    )


@pytest.mark.parametrize("late_field", ["pending", "entrypoint", "last_modified"])
def test_initial_rejects_each_late_latch_boundary(late_field: str) -> None:
    module = _module()
    upstream = _upstream()
    pending = MUST_START_BY
    entrypoint = MUST_START_BY
    last_modified = MUST_START_BY
    if late_field == "pending":
        pending += timedelta(microseconds=1)
        entrypoint = pending
    elif late_field == "entrypoint":
        entrypoint += timedelta(microseconds=1)
    else:
        last_modified += timedelta(microseconds=1)
    if last_modified < entrypoint:
        last_modified = entrypoint
    observed_at = max(pending, entrypoint, last_modified) + timedelta(microseconds=1)
    latch = _latch(
        module,
        upstream,
        pending_at=pending,
        entrypoint_at=entrypoint,
    )
    worker_observation = _observation(
        upstream["descriptor"],
        upstream["intent"],
        status="STARTING",
        start_at=MUST_START_BY,
        worker_cluster_name=WORKER_CLUSTER,
        recovery_count=0,
        observed_at=observed_at,
    )
    kwargs = _acceptance_kwargs(
        module,
        upstream,
        latch=latch,
        worker_observation=worker_observation,
        last_modified=last_modified,
        accepted_at=observed_at,
        now=observed_at,
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda kwargs: kwargs["worker_latch"].__setitem__(
            "controller_injected_sky_job_id", 18
        ),
        lambda kwargs: kwargs["job_binding"].__setitem__("sky_job_id", 18),
        lambda kwargs: kwargs["submission_accepted"].__setitem__("sky_job_id", 18),
        lambda kwargs: kwargs["worker_controller_observation"].__setitem__(
            "target_job_id", 18
        ),
    ],
)
def test_acceptance_rejects_every_job_id_mismatch(mutator: Any) -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    mutator(kwargs)
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "instance_id", RECOVERY_INSTANCE_ID
        ),
        lambda kwargs: kwargs.__setitem__(
            "active_campaign_p5_instance_ids",
            [WORKER_INSTANCE_ID, RECOVERY_INSTANCE_ID],
        ),
        lambda kwargs: kwargs.__setitem__(
            "active_campaign_p5_instance_ids",
            [RECOVERY_INSTANCE_ID, WORKER_INSTANCE_ID],
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_instance_lifecycle", "spot"
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_instance_type", "p4d.24xlarge"
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_image_id", "ami-1123456789abcdef0"
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_instance_state", "stopped"
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_ray_cluster_name", "foreign"
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_skypilot_cluster_name", "foreign"
        ),
        lambda kwargs: kwargs["worker_instance"]["worker_campaign_tags"].pop("owner"),
        lambda kwargs: kwargs["worker_instance"]["worker_campaign_tags"].__setitem__(
            "campaign-run-id", "foreign"
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "worker_instance_launch_time",
            _iso(WORKER_OBSERVED_AT + timedelta(seconds=1)),
        ),
        lambda kwargs: kwargs["worker_instance"].__setitem__(
            "ec2_observed_at", _iso(WORKER_ACCEPTED_AT - timedelta(seconds=61))
        ),
    ],
)
def test_initial_rejects_foreign_or_stale_ec2_evidence(mutator: Any) -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    mutator(kwargs)
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("worker_latch_version_id", ""),
        ("worker_latch_etag", "0123456789abcdef0123456789abcdef"),
        ("worker_latch_last_modified", "2026-07-26T05:00:07-07:00"),
        ("worker_latch_file_sha256", "0" * 64),
    ],
)
def test_initial_rejects_malformed_latch_transport(
    field: str,
    foreign: object,
) -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    kwargs[field] = foreign
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


@pytest.mark.parametrize(
    ("target", "field", "foreign"),
    [
        ("job_binding", "record_type", "glm52_sky_must_start_job_binding_v1"),
        (
            "submission_accepted",
            "record_type",
            "glm52_skypilot_submission_accepted_v1",
        ),
        ("worker_latch", "record_type", "glm52_sky_worker_start_latch_v1"),
    ],
)
def test_acceptance_rejects_legacy_or_mixed_inputs(
    target: str,
    field: str,
    foreign: object,
) -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    kwargs[target][field] = foreign
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_initial_rejects_prior_chain() -> None:
    module = _module()
    upstream = _upstream()
    initial_kwargs = _acceptance_kwargs(module, upstream)
    initial = module.build_worker_start_accepted_v2(**initial_kwargs)
    kwargs = _acceptance_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


def _initial_chain(
    module: Any,
    upstream: dict[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    kwargs = _acceptance_kwargs(module, upstream)
    accepted = module.build_worker_start_accepted_v2(**kwargs)
    latch = kwargs["worker_latch"]
    observation = kwargs["worker_controller_observation"]
    assert isinstance(latch, dict)
    assert isinstance(observation, dict)
    return (
        accepted,
        _latch_evidence(module, latch, last_modified=ENTRYPOINT_AT),
        _observation_evidence(upstream, observation, WORKER_INSTANCE_ID),
        kwargs,
    )


def _recovery_kwargs(
    module: Any,
    upstream: dict[str, object],
    *,
    prior_acceptances: list[dict[str, object]],
    prior_latches: list[dict[str, object]],
    prior_observations: list[dict[str, object]],
    instance_id: str = RECOVERY_INSTANCE_ID,
    cluster_name: str = "sky-glm52-worker-b",
    recovery_count: int = 1,
    pending_at: datetime | None = None,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    pending = pending_at or MUST_START_BY + timedelta(minutes=1)
    observed = observed_at or pending + timedelta(seconds=3)
    latch = _latch(
        module,
        upstream,
        instance_id=instance_id,
        pending_at=pending,
        entrypoint_at=pending + timedelta(seconds=1),
    )
    worker_observation = _observation(
        upstream["descriptor"],
        upstream["intent"],
        status="RECOVERING",
        start_at=pending + timedelta(seconds=2),
        worker_cluster_name=cluster_name,
        recovery_count=recovery_count,
        observed_at=observed,
    )
    kwargs = _acceptance_kwargs(
        module,
        upstream,
        latch=latch,
        instance_id=instance_id,
        worker_cluster=cluster_name,
        worker_observation=worker_observation,
        recovery_count=recovery_count,
        last_modified=pending + timedelta(seconds=1),
        accepted_at=observed,
        now=observed + timedelta(seconds=1),
        prior_acceptances=prior_acceptances,
        prior_latches=prior_latches,
        prior_observations=prior_observations,
    )
    kwargs["worker_instance"] = _worker_instance(
        instance_id=instance_id,
        cluster_name=cluster_name,
        launch_time=pending,
        observed_at=observed,
    )
    return kwargs


def test_managed_recovery_passes_after_deadline_and_links_initial() -> None:
    module = _module()
    upstream = _upstream()
    initial, latch_evidence, observation_evidence, _ = _initial_chain(module, upstream)
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[latch_evidence],
        prior_observations=[observation_evidence],
    )
    accepted = module.build_worker_start_accepted_v2(**kwargs)
    assert accepted["acceptance_kind"] == "managed-recovery"
    assert accepted["instance_id"] == RECOVERY_INSTANCE_ID
    assert accepted["recovery_count"] == 1
    assert accepted["prior_worker_acceptance_key"] == (
        module.worker_start_accepted_v2_s3_key(worker_start_accepted=initial)
    )
    assert accepted["prior_worker_acceptance_file_sha256"] == _sha(
        module.worker_v2_canonical_bytes(initial)
    )
    assert (
        accepted["prior_worker_acceptance_body_sha256"]
        == initial["worker_acceptance_body_sha256"]
    )
    assert module.validate_worker_start_accepted_v2(accepted, **kwargs) == accepted


def test_prior_acceptance_replays_launch_pending_observed_chronology() -> None:
    module = _module()
    upstream = _upstream()
    initial, latch_evidence, observation_evidence, _ = _initial_chain(module, upstream)
    initial["worker_instance_launch_time"] = _iso(
        PENDING_AT + timedelta(microseconds=1)
    )
    initial = _rehash(initial, "worker_acceptance_body_sha256")
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[latch_evidence],
        prior_observations=[observation_evidence],
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


def test_prior_acceptance_rejects_ec2_observation_before_signed_pending() -> None:
    module = _module()
    upstream = _upstream()
    initial, latch_evidence, observation_evidence, _ = _initial_chain(module, upstream)
    initial["worker_instance_launch_time"] = _iso(
        PENDING_AT - timedelta(microseconds=2)
    )
    initial["ec2_observed_at"] = _iso(PENDING_AT - timedelta(microseconds=1))
    initial = _rehash(initial, "worker_acceptance_body_sha256")
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[latch_evidence],
        prior_observations=[observation_evidence],
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


def test_prior_acceptance_replays_exact_deterministic_timestamp() -> None:
    module = _module()
    upstream = _upstream()
    initial, latch_evidence, observation_evidence, _ = _initial_chain(module, upstream)
    initial["accepted_at"] = _iso(WORKER_ACCEPTED_AT + timedelta(microseconds=1))
    initial = _rehash(initial, "worker_acceptance_body_sha256")
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[latch_evidence],
        prior_observations=[observation_evidence],
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


@pytest.mark.parametrize("prior_offset_seconds", [0, 2])
def test_prior_acceptance_must_strictly_precede_current_and_now(
    prior_offset_seconds: int,
) -> None:
    module = _module()
    upstream = _upstream()
    initial, latch_evidence, observation_evidence, _ = _initial_chain(module, upstream)
    current_observed = WORKER_OBSERVED_AT + timedelta(seconds=7)
    prior_accepted = current_observed + timedelta(seconds=prior_offset_seconds)
    latch_evidence["worker_latch_last_modified"] = _iso(prior_accepted)
    initial["worker_latch_last_modified"] = _iso(prior_accepted)
    initial["accepted_at"] = _iso(prior_accepted)
    initial = _rehash(initial, "worker_acceptance_body_sha256")
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[latch_evidence],
        prior_observations=[observation_evidence],
        pending_at=WORKER_OBSERVED_AT + timedelta(seconds=4),
        observed_at=current_observed,
    )
    with pytest.raises(module.WorkerStartValidationError):
        module.build_worker_start_accepted_v2(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda kwargs: kwargs.__setitem__("prior_worker_acceptance_chain", []),
        lambda kwargs: kwargs.__setitem__("prior_worker_latch_chain", []),
        lambda kwargs: kwargs.__setitem__(
            "prior_worker_controller_observation_chain", []
        ),
        lambda kwargs: kwargs["worker_latch"].__setitem__(
            "instance_id", WORKER_INSTANCE_ID
        ),
        lambda kwargs: kwargs["worker_controller_observation"].__setitem__(
            "recovery_count", 0
        ),
        lambda kwargs: kwargs["worker_controller_observation"].__setitem__(
            "target_job_id", 18
        ),
        lambda kwargs: kwargs["prior_worker_acceptance_chain"][0].__setitem__(
            "worker_acceptance_body_sha256", "0" * 64
        ),
        lambda kwargs: kwargs["prior_worker_latch_chain"][0].__setitem__(
            "worker_latch_last_modified", _iso(MUST_START_BY + timedelta(seconds=1))
        ),
        lambda kwargs: kwargs["prior_worker_controller_observation_chain"][
            0
        ].__setitem__(
            "worker_controller_observation_body_sha256",
            "0" * 64,
        ),
    ],
)
def test_managed_recovery_fails_closed_on_broken_chain(mutation: Any) -> None:
    module = _module()
    upstream = _upstream()
    initial, latch_evidence, observation_evidence, _ = _initial_chain(module, upstream)
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[latch_evidence],
        prior_observations=[observation_evidence],
    )
    mutation(kwargs)
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_second_recovery_links_immediate_predecessor_and_rejects_reorder() -> None:
    module = _module()
    upstream = _upstream()
    initial, initial_latch, initial_observation, _ = _initial_chain(module, upstream)
    first_kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[initial_latch],
        prior_observations=[initial_observation],
    )
    first = module.build_worker_start_accepted_v2(**first_kwargs)
    first_latch = _latch_evidence(
        module,
        first_kwargs["worker_latch"],
        last_modified=datetime.fromisoformat(
            str(first_kwargs["worker_latch_last_modified"]).replace("Z", "+00:00")
        ),
    )
    first_observation = _observation_evidence(
        upstream,
        first_kwargs["worker_controller_observation"],
        RECOVERY_INSTANCE_ID,
    )
    second_kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial, first],
        prior_latches=[initial_latch, first_latch],
        prior_observations=[initial_observation, first_observation],
        instance_id=SECOND_RECOVERY_INSTANCE_ID,
        cluster_name="sky-glm52-worker-c",
        recovery_count=2,
        pending_at=MUST_START_BY + timedelta(minutes=2),
        observed_at=MUST_START_BY + timedelta(minutes=2, seconds=3),
    )
    second = module.build_worker_start_accepted_v2(**second_kwargs)
    assert (
        second["prior_worker_acceptance_body_sha256"]
        == first["worker_acceptance_body_sha256"]
    )
    reordered = deepcopy(second_kwargs)
    reordered["prior_worker_acceptance_chain"] = [first, initial]
    assert (
        module.decide_worker_start_acceptance_v2(
            **reordered,
            existing_worker_acceptance=None,
        ).action
        == "fail-closed"
    )


def test_managed_recovery_observation_must_follow_predecessor_observation() -> None:
    module = _module()
    upstream = _upstream()
    initial, initial_latch, initial_observation, _ = _initial_chain(module, upstream)
    early_pending = START_AT - timedelta(seconds=2)
    early_observed = WORKER_OBSERVED_AT - timedelta(microseconds=1)
    kwargs = _recovery_kwargs(
        module,
        upstream,
        prior_acceptances=[initial],
        prior_latches=[initial_latch],
        prior_observations=[initial_observation],
        pending_at=early_pending,
        observed_at=early_observed,
    )
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=None,
    )
    assert decision.action == "fail-closed"


def test_acceptance_key_rejects_a_coherently_rehashed_invalid_intent_digest() -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    accepted = module.build_worker_start_accepted_v2(**kwargs)
    accepted["intent_body_sha256"] = "not-a-sha"
    accepted = _rehash(accepted, "worker_acceptance_body_sha256")
    with pytest.raises(module.WorkerStartValidationError):
        module.worker_start_accepted_v2_s3_key(worker_start_accepted=accepted)


def test_same_key_different_valid_winner_fails_closed() -> None:
    module = _module()
    kwargs = _acceptance_kwargs(module, _upstream())
    accepted = module.build_worker_start_accepted_v2(**kwargs)
    different = _rehash(
        {
            **accepted,
            "worker_instance_state": "pending",
        },
        "worker_acceptance_body_sha256",
    )
    decision = module.decide_worker_start_acceptance_v2(
        **kwargs,
        existing_worker_acceptance=different,
    )
    assert decision.action == "fail-closed"
