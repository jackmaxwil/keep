"""Strict short-lived SkyPilot submission authority contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest

from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)

RUN_ID = "glm52-sky-20260726"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
MODE = "qualification"
SKY_JOB_NAME = f"{RUN_ID}-qualification"
BUCKET = "keep-glm52-us-west-2-246813579024"
CONTROLLER_ROLE_ARN = "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
CONTROLLER_INSTANCE_ID = "i-0511af4e31aa5406a"
CONTROLLER_INSTANCE_TYPE = "c6a.xlarge"
CONTROLLER_PROFILE_ARN = (
    "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
)
CONTROLLER_CLUSTER_NAME = "sky-jobs-controller-9d9f31a9"
READINESS_BUILT_AT = datetime(2026, 7, 26, 11, 59, 30, tzinfo=UTC)
INTENT_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
BASELINE_OBSERVED_AT = datetime(2026, 7, 26, 12, 0, 20, tzinfo=UTC)
CONTROL_PLANE_OBSERVED_AT = datetime(2026, 7, 26, 12, 0, 25, tzinfo=UTC)
ACQUIRED_AT = datetime(2026, 7, 26, 12, 0, 30, tzinfo=UTC)
MUST_START_BY = datetime(2026, 7, 26, 13, 0, tzinfo=UTC)

BASELINE_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "sky_job_name",
    "workspace",
    "controller_instance_id",
    "controller_instance_type",
    "controller_profile_arn",
    "controller_cluster_name",
    "ssm_ping_status",
    "exact_name_history",
    "active_exact_name_job_ids",
    "active_tagged_p5_instance_ids",
    "observed_at",
    "baseline_body_sha256",
}
MUST_START_READY_FIELDS = {
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
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "stack_id",
    "template_sha256",
    "lambda_function_arn",
    "lambda_code_sha256",
    "iam_policy_sha256",
    "reconciliation_rule_arn",
    "reconciliation_rule_state",
    "deadline_schedule_arn",
    "deadline_schedule_state",
    "dlq_arn",
    "coordinator_mode",
    "observed_at",
    "control_plane_ready_body_sha256",
}
DEPLOYMENT_IDENTITY = {
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
        "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
        "keep-glm52-sky-deadline"
    ),
    "dlq_arn": ("arn:aws:sqs:us-west-2:246813579024:keep-glm52-sky-must-start-dlq"),
}


def _module() -> Any:
    try:
        return importlib.import_module(
            "mlx_vq.quality.glm52_sky_submission_live_authority"
        )
    except ModuleNotFoundError:
        pytest.fail("submission live-authority contract is not implemented")


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


def _h(character: str) -> str:
    return character * 64


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


def _descriptor() -> dict[str, object]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
        slack_permalink=None,
    )
    approval_file_sha = _sha(_canonical(approval) + b"\n")
    return build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=MUST_START_BY,
        controller_identity=CONTROLLER_ROLE_ARN,
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=f"campaigns/{RUN_ID}/repository/repo.tar.gz",
        repo_tar_sha256=_h("1"),
        campaign_descriptor_key=(
            f"campaigns/{RUN_ID}/submissions/qualification/campaign-descriptor-v2.json"
        ),
        approval_key=f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
        approval_sha256=approval_file_sha,
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": _h("2"),
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": _h("3"),
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": _h("4"),
            "frozen_prompt_pack_key": "quality/frozen-66.json",
            "frozen_prompt_pack_sha256": _h("5"),
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": _h("6"),
            "training_config_key": (f"campaigns/{RUN_ID}/authorities/training.json"),
            "training_config_sha256": _h("7"),
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{_h('8')}.json"
            ),
            "artifact_inventory_sha256": _h("8"),
            "qualification_cache_prefix": "qualification-cache/",
            "qualification_cache_manifest_sha256": _h("0"),
        },
    )


def _readiness(descriptor: dict[str, object]) -> dict[str, object]:
    descriptor_file_sha = _sha(_canonical(descriptor) + b"\n")
    cache_body_sha = _h("9")
    spend_body_sha = _h("a")
    rehearsal_body_sha = _h("b")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_qualification_submission_ready_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": descriptor_file_sha,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": _h("c"),
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "seed_descriptor_key": (
            f"campaigns/{RUN_ID}/submissions/cache-seed/campaign-descriptor-v2.json"
        ),
        "seed_descriptor_file_sha256": _h("d"),
        "seed_descriptor_body_sha256": _h("e"),
        "seed_campaign_identity_sha256": _h("f"),
        "seed_repo_tar_sha256": _h("1"),
        "seed_descriptor": {
            "schema_version": 2,
            "record_type": "glm52_sky_campaign_descriptor_v2",
            "run_id": f"{RUN_ID}-seed",
        },
        "staged_readiness_key": (
            f"campaigns/{RUN_ID}/staged/STAGED_CONTROL_PLANE_READY.json"
        ),
        "staged_readiness_file_sha256": _h("2"),
        "staged_readiness_body_sha256": _h("3"),
        "cache_seed_acceptance_key": (
            f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
            f"{cache_body_sha}/QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        "cache_seed_acceptance_file_sha256": _h("4"),
        "cache_seed_acceptance_body_sha256": cache_body_sha,
        "gpu_spend_snapshot_key": (
            f"campaigns/{RUN_ID}/spend-snapshots/{spend_body_sha}/"
            "GPU_SPEND_SNAPSHOT.json"
        ),
        "gpu_spend_snapshot_sha256": _h("5"),
        "gpu_spend_snapshot_body_sha256": spend_body_sha,
        "gpu_spend_ledger_tip_record_sha256": _h("6"),
        "remaining_gpu_seconds": 82_800,
        "remaining_gpu_cost_usd": 1265.92,
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "rehearsal_evidence_key": (
            f"campaigns/{RUN_ID}/qualification/rehearsals/"
            f"{rehearsal_body_sha}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "rehearsal_evidence_sha256": _h("7"),
        "rehearsal_evidence_body_sha256": rehearsal_body_sha,
        "sky_task_name": "glm52-sky-campaign",
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(MUST_START_BY),
        "allowed_gpu_seconds": 14_400,
        "allowed_gpu_cost_usd": 220.16,
        "built_at": _iso(READINESS_BUILT_AT),
    }
    return _self_hashed(body, "readiness_body_sha256")


def _intent(
    descriptor: dict[str, object],
    readiness: dict[str, object],
) -> dict[str, object]:
    readiness_key = (
        f"campaigns/{RUN_ID}/qualification/submission-ready/"
        f"{readiness['readiness_body_sha256']}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_submission_intent_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": _sha(_canonical(descriptor) + b"\n"),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
        "approval_body_sha256": readiness["approval_body_sha256"],
        "repo_tar_sha256": descriptor["repo_tar_sha256"],
        "cache_seed_acceptance_key": readiness["cache_seed_acceptance_key"],
        "cache_seed_acceptance_file_sha256": readiness[
            "cache_seed_acceptance_file_sha256"
        ],
        "cache_seed_acceptance_body_sha256": readiness[
            "cache_seed_acceptance_body_sha256"
        ],
        "gpu_spend_snapshot_key": readiness["gpu_spend_snapshot_key"],
        "gpu_spend_snapshot_sha256": readiness["gpu_spend_snapshot_sha256"],
        "gpu_spend_snapshot_body_sha256": readiness["gpu_spend_snapshot_body_sha256"],
        "gpu_spend_ledger_tip_record_sha256": readiness[
            "gpu_spend_ledger_tip_record_sha256"
        ],
        "remaining_gpu_seconds": readiness["remaining_gpu_seconds"],
        "remaining_gpu_cost_usd": readiness["remaining_gpu_cost_usd"],
        "qualification_allowance_seconds": readiness["qualification_allowance_seconds"],
        "qualification_allowance_cost_usd": readiness[
            "qualification_allowance_cost_usd"
        ],
        "open_allocation_count": 0,
        "qualification_submission_ready_key": readiness_key,
        "qualification_submission_ready_sha256": _sha(_canonical(readiness) + b"\n"),
        "qualification_submission_ready_body_sha256": readiness[
            "readiness_body_sha256"
        ],
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(MUST_START_BY),
        "intent_at": _iso(INTENT_AT),
    }
    return _self_hashed(body, "intent_body_sha256")


def _sources() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    descriptor = _descriptor()
    readiness = _readiness(descriptor)
    return descriptor, readiness, _intent(descriptor, readiness)


def _readiness_key(readiness: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/qualification/submission-ready/"
        f"{readiness['readiness_body_sha256']}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )


def _intent_key(intent: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/submissions/{MODE}/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )


def _baseline_key(baseline: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )


def _must_start_ready_key(ready: dict[str, object]) -> str:
    return (
        f"campaigns/{RUN_ID}/monitor/must-start/{MODE}/"
        f"{ready['intent_body_sha256']}/control-plane-ready/"
        f"{ready['control_plane_ready_body_sha256']}/"
        "CONTROL_PLANE_READY.json"
    )


def _baseline_history() -> list[dict[str, object]]:
    return [
        {
            "sky_job_id": 3,
            "sky_job_name": SKY_JOB_NAME,
            "workspace": "default",
            "controller_submitted_at": "2026-07-25T10:00:00Z",
            "controller_status": "SUCCEEDED",
            "controller_identity": CONTROLLER_ROLE_ARN,
        }
    ]


def _build_baseline(
    *,
    sources: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]
    | None = None,
    exact_name_history: list[dict[str, object]] | None = None,
    active_exact_name_job_ids: list[int] | None = None,
    active_tagged_p5_instance_ids: list[str] | None = None,
    observed_at: datetime | str = BASELINE_OBSERVED_AT,
    controller_profile_arn: str = CONTROLLER_PROFILE_ARN,
    descriptor_key: str | None = None,
    descriptor_raw: bytes | None = None,
    qualification_submission_ready_key: str | None = None,
    qualification_submission_ready_raw: bytes | None = None,
    intent_key: str | None = None,
    intent_raw: bytes | None = None,
) -> dict[str, object]:
    descriptor, readiness, intent = sources or _sources()
    history = _baseline_history() if exact_name_history is None else exact_name_history
    return _module().build_controller_baseline(
        descriptor=descriptor,
        descriptor_key=(
            str(descriptor["campaign_descriptor_key"])
            if descriptor_key is None
            else descriptor_key
        ),
        descriptor_raw=(
            _canonical(descriptor) + b"\n" if descriptor_raw is None else descriptor_raw
        ),
        qualification_submission_ready=readiness,
        qualification_submission_ready_key=(
            _readiness_key(readiness)
            if qualification_submission_ready_key is None
            else qualification_submission_ready_key
        ),
        qualification_submission_ready_raw=(
            _canonical(readiness) + b"\n"
            if qualification_submission_ready_raw is None
            else qualification_submission_ready_raw
        ),
        intent=intent,
        intent_key=_intent_key(intent) if intent_key is None else intent_key,
        intent_raw=(_canonical(intent) + b"\n" if intent_raw is None else intent_raw),
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=controller_profile_arn,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        ssm_ping_status="Online",
        exact_name_history=history,
        active_exact_name_job_ids=(
            [] if active_exact_name_job_ids is None else active_exact_name_job_ids
        ),
        active_tagged_p5_instance_ids=(
            []
            if active_tagged_p5_instance_ids is None
            else active_tagged_p5_instance_ids
        ),
        observed_at=observed_at,
    )


def _validate_baseline(
    value: object,
    *,
    sources: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]
    | None = None,
    acquired_at: datetime | str = ACQUIRED_AT,
    exact_name_history: list[dict[str, object]] | None = None,
    observed_at: datetime | str = BASELINE_OBSERVED_AT,
    controller_baseline_key: str | None = None,
    controller_baseline_raw: bytes | None = None,
    qualification_submission_ready_raw: bytes | None = None,
) -> dict[str, object]:
    descriptor, readiness, intent = sources or _sources()
    baseline = value if isinstance(value, dict) else {}
    return _module().validate_controller_baseline(
        value,
        descriptor=descriptor,
        descriptor_key=str(descriptor["campaign_descriptor_key"]),
        descriptor_raw=_canonical(descriptor) + b"\n",
        qualification_submission_ready=readiness,
        qualification_submission_ready_key=_readiness_key(readiness),
        qualification_submission_ready_raw=(
            _canonical(readiness) + b"\n"
            if qualification_submission_ready_raw is None
            else qualification_submission_ready_raw
        ),
        intent=intent,
        intent_key=_intent_key(intent),
        intent_raw=_canonical(intent) + b"\n",
        controller_baseline_key=(
            _baseline_key(baseline)
            if controller_baseline_key is None
            else controller_baseline_key
        ),
        controller_baseline_raw=(
            _canonical(baseline) + b"\n"
            if controller_baseline_raw is None
            else controller_baseline_raw
        ),
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        ssm_ping_status="Online",
        exact_name_history=(
            _baseline_history() if exact_name_history is None else exact_name_history
        ),
        active_exact_name_job_ids=[],
        active_tagged_p5_instance_ids=[],
        observed_at=observed_at,
        acquired_at=acquired_at,
    )


def test_builds_exact_fresh_controller_baseline_and_canonical_file() -> None:
    module = _module()
    baseline = _build_baseline()

    assert set(baseline) == BASELINE_FIELDS
    assert baseline["schema_version"] == 1
    assert baseline["record_type"] == "glm52_controller_baseline_v1"
    body = dict(baseline)
    digest = body.pop("baseline_body_sha256")
    assert digest == _sha(_canonical(body))
    assert _validate_baseline(baseline) == baseline
    assert module.controller_baseline_s3_key(
        run_id=RUN_ID,
        baseline_body_sha256=digest,
    ) == (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/{digest}/"
        "CONTROLLER_BASELINE.json"
    )
    assert module.controller_baseline_s3_key(baseline) == (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/{digest}/"
        "CONTROLLER_BASELINE.json"
    )
    assert module.controller_baseline_file_bytes(baseline) == (
        _canonical(baseline) + b"\n"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("region"), "schema mismatch"),
        (
            lambda value: value.__setitem__("owner_token", "not-authority"),
            "schema mismatch",
        ),
        (
            lambda value: value.__setitem__(
                "record_type",
                "glm52_controller_baseline_v2",
            ),
            "schema mismatch",
        ),
        (
            lambda value: value.__setitem__("schema_version", True),
            "schema mismatch",
        ),
    ],
)
def test_rejects_missing_unknown_mixed_and_boolean_baseline_schema(
    mutation: Any,
    match: str,
) -> None:
    baseline = _build_baseline()
    mutation(baseline)
    baseline = _rehash(baseline, "baseline_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate_baseline(baseline)


def test_rejects_baseline_body_tampering_without_rehash() -> None:
    baseline = _build_baseline()
    baseline["observed_at"] = "2026-07-26T12:00:21Z"

    with pytest.raises(ValueError, match="body SHA-256 mismatch"):
        _validate_baseline(baseline)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["exact_name_history"][0].__setitem__(
                "controller_status",
                "FAILED",
            ),
            "authority drift",
        ),
        (
            lambda value: value.__setitem__(
                "observed_at",
                "2026-07-26T12:00:21Z",
            ),
            "authority drift",
        ),
    ],
)
def test_rejects_coherently_rehashed_captured_baseline_fact_drift(
    mutate: Any,
    match: str,
) -> None:
    baseline = deepcopy(_build_baseline())
    mutate(baseline)
    baseline = _rehash(baseline, "baseline_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate_baseline(baseline)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("controller_instance_id", "i-00000000000000001", "authority drift"),
        ("controller_instance_type", "m5.large", "authority drift"),
        (
            "controller_profile_arn",
            ("arn:aws:iam::246813579024:instance-profile/foreign-controller"),
            "authority drift",
        ),
        ("controller_cluster_name", "foreign-controller", "authority drift"),
        ("ssm_ping_status", "ConnectionLost", "Online"),
    ],
)
def test_rejects_coherently_rehashed_controller_identity_drift(
    field: str,
    value: object,
    match: str,
) -> None:
    baseline = _build_baseline()
    baseline[field] = value
    baseline = _rehash(baseline, "baseline_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate_baseline(baseline)


def test_rejects_foreign_controller_profile_at_build_time() -> None:
    with pytest.raises(ValueError, match="descriptor controller identity"):
        _build_baseline(
            controller_profile_arn=(
                "arn:aws:iam::246813579024:instance-profile/foreign-controller"
            )
        )


@pytest.mark.parametrize(
    ("history", "match"),
    [
        (
            [
                {
                    "sky_job_id": True,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": "SUCCEEDED",
                    "controller_identity": CONTROLLER_ROLE_ARN,
                }
            ],
            "sky_job_id",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": "foreign-job",
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": "SUCCEEDED",
                    "controller_identity": CONTROLLER_ROLE_ARN,
                }
            ],
            "sky_job_name",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "other",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": "SUCCEEDED",
                    "controller_identity": CONTROLLER_ROLE_ARN,
                }
            ],
            "workspace",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": "SUCCEEDED",
                    "controller_identity": (
                        "arn:aws:iam::246813579024:role/foreign-controller"
                    ),
                }
            ],
            "controller_identity",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00+00:00",
                    "controller_status": "SUCCEEDED",
                    "controller_identity": CONTROLLER_ROLE_ARN,
                }
            ],
            "canonical UTC",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": "MYSTERY",
                    "controller_identity": CONTROLLER_ROLE_ARN,
                }
            ],
            "controller_status",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": [],
                    "controller_identity": CONTROLLER_ROLE_ARN,
                }
            ],
            "controller_status",
        ),
        (
            [
                {
                    "sky_job_id": 3,
                    "sky_job_name": SKY_JOB_NAME,
                    "workspace": "default",
                    "controller_submitted_at": "2026-07-25T10:00:00Z",
                    "controller_status": "SUCCEEDED",
                    "controller_identity": CONTROLLER_ROLE_ARN,
                    "legacy": True,
                }
            ],
            "schema mismatch",
        ),
    ],
)
def test_rejects_malformed_or_foreign_exact_name_history(
    history: list[dict[str, object]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _build_baseline(exact_name_history=history)


def test_rejects_ambiguous_duplicate_or_unsorted_controller_history() -> None:
    first = {
        "sky_job_id": 3,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": "2026-07-25T10:00:00Z",
        "controller_status": "SUCCEEDED",
        "controller_identity": CONTROLLER_ROLE_ARN,
    }
    duplicate = {**first, "controller_status": "FAILED"}
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        _build_baseline(exact_name_history=[first, duplicate])

    later = {**first, "sky_job_id": 4}
    with pytest.raises(ValueError, match="canonical sorted order"):
        _build_baseline(exact_name_history=[later, first])


def test_rejects_history_row_observed_before_its_submission() -> None:
    future_row = {
        "sky_job_id": 3,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": "2026-07-26T12:00:21Z",
        "controller_status": "SUCCEEDED",
        "controller_identity": CONTROLLER_ROLE_ARN,
    }

    with pytest.raises(ValueError, match="after baseline observation"):
        _build_baseline(exact_name_history=[future_row])


def test_rejects_active_exact_name_job_from_history_and_explicit_set() -> None:
    running = {
        "sky_job_id": 17,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": "2026-07-26T12:00:10Z",
        "controller_status": "RUNNING",
        "controller_identity": CONTROLLER_ROLE_ARN,
    }
    with pytest.raises(ValueError, match="do not match exact_name_history"):
        _build_baseline(exact_name_history=[running])
    with pytest.raises(ValueError, match="active exact-name job"):
        _build_baseline(
            exact_name_history=[running],
            active_exact_name_job_ids=[17],
        )


def test_rejects_active_or_malformed_campaign_p5_set() -> None:
    with pytest.raises(ValueError, match="active tagged P5"):
        _build_baseline(active_tagged_p5_instance_ids=["i-0611af4e31aa5406a"])
    with pytest.raises(ValueError, match="sorted and unique"):
        _build_baseline(
            active_tagged_p5_instance_ids=[
                "i-0611af4e31aa5406a",
                "i-0611af4e31aa5406a",
            ]
        )


def test_baseline_accepts_exact_sixty_second_age_and_rejects_older() -> None:
    baseline = _build_baseline()
    exact_boundary = BASELINE_OBSERVED_AT.replace(
        minute=BASELINE_OBSERVED_AT.minute + 1
    )
    assert _validate_baseline(baseline, acquired_at=exact_boundary) == baseline

    with pytest.raises(ValueError, match="older than 60 seconds"):
        _validate_baseline(
            baseline,
            acquired_at=exact_boundary.replace(second=21),
        )


def test_rejects_baseline_after_acquisition_or_deadline() -> None:
    baseline = _build_baseline()
    with pytest.raises(ValueError, match="after acquisition"):
        _validate_baseline(
            baseline,
            acquired_at="2026-07-26T12:00:19Z",
        )
    with pytest.raises(ValueError, match="at or after must_start_by"):
        _build_baseline(observed_at=MUST_START_BY)


def test_rejects_coherently_rehashed_foreign_intent_and_readiness() -> None:
    descriptor, readiness, intent = _sources()
    intent["descriptor_key"] = (
        f"campaigns/{RUN_ID}/submissions/qualification/../foreign.json"
    )
    intent = _rehash(intent, "intent_body_sha256")
    with pytest.raises(ValueError, match="path traversal"):
        _validate_baseline(
            _build_baseline(),
            sources=(descriptor, readiness, intent),
        )


@pytest.mark.parametrize(
    ("field", "wrong_key", "match"),
    [
        (
            "cache_seed_acceptance_key",
            f"campaigns/{RUN_ID}/qualification-cache-seed/not-addressed.json",
            "cache_seed_acceptance_key",
        ),
        (
            "gpu_spend_snapshot_key",
            f"campaigns/{RUN_ID}/spend-snapshots/not-addressed.json",
            "gpu_spend_snapshot_key",
        ),
        (
            "rehearsal_evidence_key",
            f"campaigns/{RUN_ID}/qualification/not-content-addressed.json",
            "rehearsal_evidence_key",
        ),
    ],
)
def test_rejects_same_run_noncanonical_upstream_authority_key(
    field: str,
    wrong_key: str,
    match: str,
) -> None:
    descriptor, readiness, intent = _sources()
    readiness[field] = wrong_key
    readiness = _rehash(readiness, "readiness_body_sha256")
    if field in intent:
        intent[field] = wrong_key
    intent["qualification_submission_ready_key"] = (
        f"campaigns/{RUN_ID}/qualification/submission-ready/"
        f"{readiness['readiness_body_sha256']}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )
    intent["qualification_submission_ready_sha256"] = _sha(
        _canonical(readiness) + b"\n"
    )
    intent["qualification_submission_ready_body_sha256"] = readiness[
        "readiness_body_sha256"
    ]
    intent = _rehash(intent, "intent_body_sha256")

    with pytest.raises(ValueError, match=match):
        _build_baseline(sources=(descriptor, readiness, intent))


def test_rejects_noncanonical_actual_source_bytes_before_build() -> None:
    descriptor, readiness, intent = _sources()
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/{MODE}/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )

    with pytest.raises(ValueError, match="descriptor bytes are not exact"):
        _module().build_controller_baseline(
            descriptor=descriptor,
            descriptor_key=str(descriptor["campaign_descriptor_key"]),
            descriptor_raw=_canonical(descriptor) + b"\n\n",
            qualification_submission_ready=readiness,
            qualification_submission_ready_key=str(
                intent["qualification_submission_ready_key"]
            ),
            qualification_submission_ready_raw=_canonical(readiness) + b"\n",
            intent=intent,
            intent_key=intent_key,
            intent_raw=_canonical(intent) + b"\n",
            controller_instance_id=CONTROLLER_INSTANCE_ID,
            controller_instance_type=CONTROLLER_INSTANCE_TYPE,
            controller_profile_arn=CONTROLLER_PROFILE_ARN,
            controller_cluster_name=CONTROLLER_CLUSTER_NAME,
            ssm_ping_status="Online",
            exact_name_history=_baseline_history(),
            active_exact_name_job_ids=[],
            active_tagged_p5_instance_ids=[],
            observed_at=BASELINE_OBSERVED_AT,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "descriptor_key": (
                    f"campaigns/{RUN_ID}/submissions/qualification/"
                    "wrong-descriptor.json"
                )
            },
            "descriptor key",
        ),
        (
            {
                "qualification_submission_ready_key": (
                    f"campaigns/{RUN_ID}/qualification/submission-ready/"
                    f"{_h('a')}/QUALIFICATION_SUBMISSION_READY.json"
                )
            },
            "qualification submission readiness key",
        ),
        (
            {"qualification_submission_ready_raw": b"{}\r\n"},
            "qualification submission readiness bytes",
        ),
        (
            {
                "intent_key": (
                    f"campaigns/{RUN_ID}/submissions/{MODE}/intents/"
                    f"{_h('b')}/SKYPILOT_SUBMISSION_INTENT.json"
                )
            },
            "submission intent key",
        ),
        (
            {"intent_raw": b"{}\n\n"},
            "submission intent bytes",
        ),
    ],
)
def test_rejects_wrong_actual_source_key_or_raw_bytes(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _build_baseline(**overrides)  # type: ignore[arg-type]


def test_rejects_wrong_actual_baseline_key_or_raw_bytes() -> None:
    baseline = _build_baseline()
    with pytest.raises(ValueError, match="controller baseline key"):
        _validate_baseline(
            baseline,
            controller_baseline_key=(
                f"campaigns/{RUN_ID}/qualification/controller-baselines/"
                f"{_h('0')}/CONTROLLER_BASELINE.json"
            ),
        )
    with pytest.raises(ValueError, match="controller baseline bytes"):
        _validate_baseline(
            baseline,
            controller_baseline_raw=_canonical(baseline),
        )


def _build_must_start_ready(
    *,
    baseline: dict[str, object] | None = None,
    deployment_identity: dict[str, object] | None = None,
    reconciliation_rule_state: str = "ENABLED",
    deadline_schedule_state: str = "ENABLED",
    coordinator_mode: str = "dynamic-job-binding-active",
    observed_at: datetime | str = CONTROL_PLANE_OBSERVED_AT,
    baseline_observed_at: datetime | str = BASELINE_OBSERVED_AT,
) -> dict[str, object]:
    descriptor, readiness, intent = _sources()
    baseline_value = baseline or _build_baseline()
    return _module().build_must_start_control_plane_ready(
        descriptor=descriptor,
        descriptor_key=str(descriptor["campaign_descriptor_key"]),
        descriptor_raw=_canonical(descriptor) + b"\n",
        qualification_submission_ready=readiness,
        qualification_submission_ready_key=_readiness_key(readiness),
        qualification_submission_ready_raw=_canonical(readiness) + b"\n",
        intent=intent,
        intent_key=_intent_key(intent),
        intent_raw=_canonical(intent) + b"\n",
        controller_baseline=baseline_value,
        controller_baseline_key=_baseline_key(baseline_value),
        controller_baseline_raw=_canonical(baseline_value) + b"\n",
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        baseline_ssm_ping_status="Online",
        baseline_exact_name_history=_baseline_history(),
        baseline_active_exact_name_job_ids=[],
        baseline_active_tagged_p5_instance_ids=[],
        baseline_observed_at=baseline_observed_at,
        reviewed_deployment_identity=DEPLOYMENT_IDENTITY,
        observed_deployment_identity=(
            DEPLOYMENT_IDENTITY if deployment_identity is None else deployment_identity
        ),
        reconciliation_rule_state=reconciliation_rule_state,
        deadline_schedule_state=deadline_schedule_state,
        coordinator_mode=coordinator_mode,
        observed_at=observed_at,
    )


def _validate_must_start_ready(
    value: object,
    *,
    baseline: dict[str, object] | None = None,
    deployment_identity: dict[str, object] | None = None,
    acquired_at: datetime | str = ACQUIRED_AT,
    baseline_observed_at: datetime | str = BASELINE_OBSERVED_AT,
    observed_at: datetime | str = CONTROL_PLANE_OBSERVED_AT,
    controller_baseline_key: str | None = None,
    controller_baseline_raw: bytes | None = None,
    must_start_control_plane_ready_key: str | None = None,
    must_start_control_plane_ready_raw: bytes | None = None,
) -> dict[str, object]:
    descriptor, readiness, intent = _sources()
    baseline_value = baseline or _build_baseline()
    ready = value if isinstance(value, dict) else {}
    return _module().validate_must_start_control_plane_ready(
        value,
        descriptor=descriptor,
        descriptor_key=str(descriptor["campaign_descriptor_key"]),
        descriptor_raw=_canonical(descriptor) + b"\n",
        qualification_submission_ready=readiness,
        qualification_submission_ready_key=_readiness_key(readiness),
        qualification_submission_ready_raw=_canonical(readiness) + b"\n",
        intent=intent,
        intent_key=_intent_key(intent),
        intent_raw=_canonical(intent) + b"\n",
        controller_baseline=baseline_value,
        controller_baseline_key=(
            _baseline_key(baseline_value)
            if controller_baseline_key is None
            else controller_baseline_key
        ),
        controller_baseline_raw=(
            _canonical(baseline_value) + b"\n"
            if controller_baseline_raw is None
            else controller_baseline_raw
        ),
        must_start_control_plane_ready_key=(
            _must_start_ready_key(ready)
            if must_start_control_plane_ready_key is None
            else must_start_control_plane_ready_key
        ),
        must_start_control_plane_ready_raw=(
            _canonical(ready) + b"\n"
            if must_start_control_plane_ready_raw is None
            else must_start_control_plane_ready_raw
        ),
        controller_instance_id=CONTROLLER_INSTANCE_ID,
        controller_instance_type=CONTROLLER_INSTANCE_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE_ARN,
        controller_cluster_name=CONTROLLER_CLUSTER_NAME,
        baseline_ssm_ping_status="Online",
        baseline_exact_name_history=_baseline_history(),
        baseline_active_exact_name_job_ids=[],
        baseline_active_tagged_p5_instance_ids=[],
        baseline_observed_at=baseline_observed_at,
        reviewed_deployment_identity=DEPLOYMENT_IDENTITY,
        observed_deployment_identity=(
            DEPLOYMENT_IDENTITY if deployment_identity is None else deployment_identity
        ),
        reconciliation_rule_state="ENABLED",
        deadline_schedule_state="ENABLED",
        coordinator_mode="dynamic-job-binding-active",
        observed_at=observed_at,
        acquired_at=acquired_at,
    )


def test_builds_exact_intent_scoped_must_start_readiness_and_file() -> None:
    module = _module()
    baseline = _build_baseline()
    ready = _build_must_start_ready(baseline=baseline)

    assert set(ready) == MUST_START_READY_FIELDS
    assert ready["schema_version"] == 1
    assert ready["record_type"] == "glm52_must_start_control_plane_ready_v1"
    body = dict(ready)
    digest = body.pop("control_plane_ready_body_sha256")
    assert digest == _sha(_canonical(body))
    assert _validate_must_start_ready(ready, baseline=baseline) == ready
    expected_key = (
        f"campaigns/{RUN_ID}/monitor/must-start/{MODE}/"
        f"{ready['intent_body_sha256']}/control-plane-ready/{digest}/"
        "CONTROL_PLANE_READY.json"
    )
    assert (
        module.must_start_control_plane_ready_s3_key(
            run_id=RUN_ID,
            managed_mode=MODE,
            intent_body_sha256=str(ready["intent_body_sha256"]),
            control_plane_ready_body_sha256=digest,
        )
        == expected_key
    )
    assert module.must_start_control_plane_ready_s3_key(ready) == expected_key
    assert module.must_start_control_plane_ready_file_bytes(ready) == (
        _canonical(ready) + b"\n"
    )


def test_rejects_wrong_actual_ready_or_baseline_artifact_bytes() -> None:
    baseline = _build_baseline()
    ready = _build_must_start_ready(baseline=baseline)
    with pytest.raises(ValueError, match="controller baseline bytes"):
        _validate_must_start_ready(
            ready,
            baseline=baseline,
            controller_baseline_raw=_canonical(baseline),
        )
    with pytest.raises(ValueError, match="must-start control-plane readiness key"):
        _validate_must_start_ready(
            ready,
            baseline=baseline,
            must_start_control_plane_ready_key=(
                f"campaigns/{RUN_ID}/monitor/must-start/{MODE}/"
                f"{ready['intent_body_sha256']}/control-plane-ready/"
                f"{_h('0')}/CONTROL_PLANE_READY.json"
            ),
        )
    with pytest.raises(
        ValueError,
        match="must-start control-plane readiness bytes",
    ):
        _validate_must_start_ready(
            ready,
            baseline=baseline,
            must_start_control_plane_ready_raw=_canonical(ready) + b"\r\n",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("region"), "schema mismatch"),
        (
            lambda value: value.__setitem__("legacy_target_job_id", 3),
            "schema mismatch",
        ),
        (
            lambda value: value.__setitem__(
                "record_type",
                "glm52_must_start_control_plane_ready_v2",
            ),
            "schema mismatch",
        ),
        (
            lambda value: value.__setitem__("schema_version", True),
            "schema mismatch",
        ),
    ],
)
def test_rejects_missing_unknown_mixed_and_boolean_readiness_schema(
    mutation: Any,
    match: str,
) -> None:
    ready = _build_must_start_ready()
    mutation(ready)
    ready = _rehash(ready, "control_plane_ready_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate_must_start_ready(ready)


def test_rejects_readiness_body_tampering_without_rehash() -> None:
    ready = _build_must_start_ready()
    ready["lambda_code_sha256"] = _h("0")

    with pytest.raises(ValueError, match="body SHA-256 mismatch"):
        _validate_must_start_ready(ready)


def test_rejects_coherently_rehashed_control_plane_observation_drift() -> None:
    ready = _build_must_start_ready()
    ready["observed_at"] = "2026-07-26T12:00:26Z"
    ready = _rehash(ready, "control_plane_ready_body_sha256")

    with pytest.raises(ValueError, match="authority drift"):
        _validate_must_start_ready(ready)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("template_sha256", _h("0"), "authority drift"),
        ("lambda_code_sha256", _h("1"), "authority drift"),
        ("iam_policy_sha256", _h("2"), "authority drift"),
        (
            "lambda_function_arn",
            ("arn:aws:lambda:us-west-2:246813579024:function:foreign-function"),
            "authority drift",
        ),
        (
            "reconciliation_rule_arn",
            ("arn:aws:events:us-west-2:246813579024:rule/foreign-reconciliation"),
            "authority drift",
        ),
        (
            "deadline_schedule_arn",
            (
                "arn:aws:scheduler:us-west-2:246813579024:"
                "schedule/default/foreign-deadline"
            ),
            "authority drift",
        ),
        (
            "dlq_arn",
            ("arn:aws:sqs:us-west-2:246813579024:foreign-must-start-dlq"),
            "authority drift",
        ),
    ],
)
def test_rejects_coherently_rehashed_wrong_code_config_role_or_resource(
    field: str,
    value: object,
    match: str,
) -> None:
    ready = _build_must_start_ready()
    ready[field] = value
    ready = _rehash(ready, "control_plane_ready_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate_must_start_ready(ready)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "stack_id",
            ("arn:aws:cloudformation:us-west-2:246813579024:stack/no-stack-id"),
            "stack_id",
        ),
        (
            "lambda_function_arn",
            (
                "arn:aws:lambda:us-east-1:246813579024:function:"
                "keep-glm52-sky-must-start"
            ),
            "foreign",
        ),
        (
            "reconciliation_rule_arn",
            ("arn:aws:events:us-west-2:135792468013:rule/keep-glm52-sky-reconcile"),
            "foreign",
        ),
        (
            "deadline_schedule_arn",
            ("arn:aws:scheduler:us-west-2:246813579024:schedule/../foreign"),
            "invalid",
        ),
        (
            "dlq_arn",
            ("arn:aws:sqs:us-west-2:246813579024:foreign/queue"),
            "dlq_arn",
        ),
        ("lambda_code_sha256", _h("A"), "canonical lowercase"),
    ],
)
def test_rejects_malformed_or_foreign_deployment_identity(
    field: str,
    value: object,
    match: str,
) -> None:
    deployment = dict(DEPLOYMENT_IDENTITY)
    deployment[field] = value

    with pytest.raises(ValueError, match=match):
        _build_must_start_ready(deployment_identity=deployment)


def test_rejects_missing_or_unknown_deployment_identity_fields() -> None:
    missing = dict(DEPLOYMENT_IDENTITY)
    missing.pop("iam_policy_sha256")
    with pytest.raises(ValueError, match="schema mismatch"):
        _build_must_start_ready(deployment_identity=missing)

    unknown = {**DEPLOYMENT_IDENTITY, "lambda_environment": {}}
    with pytest.raises(ValueError, match="schema mismatch"):
        _build_must_start_ready(deployment_identity=unknown)


def test_rejects_unrelated_same_account_deployment_as_observation() -> None:
    unrelated = {
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "unrelated-control-plane/"
            "11111111-2222-3333-4444-555555555555"
        ),
        "template_sha256": _h("1"),
        "lambda_function_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:unrelated-function"
        ),
        "lambda_code_sha256": _h("2"),
        "iam_policy_sha256": _h("3"),
        "reconciliation_rule_arn": (
            "arn:aws:events:us-west-2:246813579024:rule/unrelated-rule"
        ),
        "deadline_schedule_arn": (
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
            "unrelated-schedule"
        ),
        "dlq_arn": ("arn:aws:sqs:us-west-2:246813579024:unrelated-dead-letter"),
    }

    with pytest.raises(ValueError, match="reviewed deployment identity"):
        _build_must_start_ready(deployment_identity=unrelated)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "coordinator_mode",
            "legacy-static-job-id",
            "coordinator_mode",
        ),
        ("reconciliation_rule_state", "DISABLED", "must be ENABLED"),
        ("deadline_schedule_state", "DISABLED", "must be ENABLED"),
    ],
)
def test_rejects_inactive_or_legacy_control_plane(
    field: str,
    value: str,
    match: str,
) -> None:
    arguments: dict[str, str] = {field: value}
    with pytest.raises(ValueError, match=match):
        _build_must_start_ready(**arguments)  # type: ignore[arg-type]


def test_rejects_readiness_before_baseline_or_at_deadline() -> None:
    with pytest.raises(ValueError, match="after acquisition|predates"):
        _build_must_start_ready(
            observed_at="2026-07-26T12:00:19Z",
        )
    with pytest.raises(ValueError, match="at or after must_start_by"):
        _build_must_start_ready(observed_at=MUST_START_BY)


def test_readiness_accepts_exact_sixty_second_age_and_rejects_older() -> None:
    baseline = _build_baseline(observed_at=CONTROL_PLANE_OBSERVED_AT)
    ready = _build_must_start_ready(
        baseline=baseline,
        baseline_observed_at=CONTROL_PLANE_OBSERVED_AT,
    )
    exact_boundary = CONTROL_PLANE_OBSERVED_AT.replace(
        minute=CONTROL_PLANE_OBSERVED_AT.minute + 1
    )
    assert (
        _validate_must_start_ready(
            ready,
            baseline=baseline,
            baseline_observed_at=CONTROL_PLANE_OBSERVED_AT,
            acquired_at=exact_boundary,
        )
        == ready
    )
    with pytest.raises(ValueError, match="older than 60 seconds"):
        _validate_must_start_ready(
            ready,
            baseline=baseline,
            baseline_observed_at=CONTROL_PLANE_OBSERVED_AT,
            acquired_at=exact_boundary.replace(second=26),
        )


def test_rejects_readiness_after_acquisition_or_acquisition_at_deadline() -> None:
    ready = _build_must_start_ready()
    with pytest.raises(ValueError, match="after acquisition"):
        _validate_must_start_ready(
            ready,
            acquired_at="2026-07-26T12:00:24Z",
        )
    with pytest.raises(ValueError, match="at or after must_start_by"):
        _validate_must_start_ready(ready, acquired_at=MUST_START_BY)


def test_rejects_coherently_rehashed_baseline_and_intent_binding_drift() -> None:
    baseline = _build_baseline()
    ready = _build_must_start_ready(baseline=baseline)

    drifted_baseline = deepcopy(baseline)
    drifted_baseline["controller_cluster_name"] = "foreign-controller"
    drifted_baseline = _rehash(
        drifted_baseline,
        "baseline_body_sha256",
    )
    with pytest.raises(ValueError, match="baseline"):
        _validate_must_start_ready(
            ready,
            baseline=drifted_baseline,
        )

    ready["intent_file_sha256"] = _h("0")
    ready = _rehash(ready, "control_plane_ready_body_sha256")
    with pytest.raises(ValueError, match="authority drift"):
        _validate_must_start_ready(ready, baseline=baseline)


def test_rejects_noncanonical_time_hash_key_and_nonfinite_source() -> None:
    ready = _build_must_start_ready()
    ready["observed_at"] = "2026-07-26T12:00:25+00:00"
    ready = _rehash(ready, "control_plane_ready_body_sha256")
    with pytest.raises(ValueError, match="canonical UTC"):
        _validate_must_start_ready(ready)

    module = _module()
    with pytest.raises(ValueError, match="canonical lowercase"):
        module.must_start_control_plane_ready_s3_key(
            run_id=RUN_ID,
            managed_mode=MODE,
            intent_body_sha256=_h("A"),
            control_plane_ready_body_sha256=_h("b"),
        )
    with pytest.raises(ValueError, match="run_id"):
        module.controller_baseline_s3_key(
            run_id="../foreign",
            baseline_body_sha256=_h("b"),
        )

    descriptor, readiness, intent = _sources()
    readiness["remaining_gpu_cost_usd"] = float("nan")
    readiness["readiness_body_sha256"] = _h("0")
    with pytest.raises(ValueError, match="finite JSON"):
        _validate_baseline(
            _build_baseline(),
            sources=(descriptor, readiness, intent),
            qualification_submission_ready_raw=b"{}\n",
        )

    descriptor, readiness, intent = _sources()
    readiness["remaining_gpu_seconds"] = 82_799
    readiness = _rehash(readiness, "readiness_body_sha256")
    with pytest.raises(ValueError, match="drift"):
        _validate_baseline(
            _build_baseline(),
            sources=(descriptor, readiness, intent),
        )
