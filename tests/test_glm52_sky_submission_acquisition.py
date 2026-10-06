"""Strict singleton SkyPilot submission-acquisition selector tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

RUN_ID = "glm52-sky-20260726"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
MODE = "qualification"
SKY_JOB_NAME = f"{RUN_ID}-qualification"
READINESS_BUILT_AT = datetime(2026, 7, 26, 11, 59, 30, tzinfo=UTC)
INTENT_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
BASELINE_OBSERVED_AT = datetime(2026, 7, 26, 12, 0, 20, tzinfo=UTC)
CONTROL_PLANE_OBSERVED_AT = datetime(2026, 7, 26, 12, 0, 25, tzinfo=UTC)
ACQUIRED_AT = datetime(2026, 7, 26, 12, 0, 30, tzinfo=UTC)
MUST_START_BY = datetime(2026, 7, 26, 13, 0, tzinfo=UTC)

ACQUISITION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "managed_mode",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "qualification_submission_ready_key",
    "qualification_submission_ready_sha256",
    "qualification_submission_ready_body_sha256",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "must_start_control_plane_ready_key",
    "must_start_control_plane_ready_file_sha256",
    "must_start_control_plane_ready_body_sha256",
    "sky_job_name",
    "must_start_by",
    "acquired_at",
    "acquisition_body_sha256",
}


def _module() -> Any:
    try:
        return importlib.import_module(
            "mlx_vq.quality.glm52_sky_submission_acquisition"
        )
    except ModuleNotFoundError:
        pytest.fail("submission-acquisition contract is not implemented")


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


def _file_sha(value: object) -> str:
    return _sha(_canonical(value) + b"\n")


def _record_key(label: str, value: dict[str, object]) -> str:
    if label == "readiness":
        return (
            f"campaigns/{RUN_ID}/qualification/submission-ready/"
            f"{value['readiness_body_sha256']}/"
            "QUALIFICATION_SUBMISSION_READY.json"
        )
    if label == "intent":
        return (
            f"campaigns/{RUN_ID}/submissions/{MODE}/intents/"
            f"{value['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
        )
    if label == "baseline":
        return (
            f"campaigns/{RUN_ID}/qualification/controller-baselines/"
            f"{value['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
        )
    if label == "control_plane":
        return (
            f"campaigns/{RUN_ID}/monitor/must-start/{MODE}/"
            f"{value['intent_body_sha256']}/control-plane-ready/"
            f"{value['control_plane_ready_body_sha256']}/"
            "CONTROL_PLANE_READY.json"
        )
    if label == "acquisition":
        return (
            f"campaigns/{RUN_ID}/submissions/{MODE}/acquisitions/"
            f"{value['descriptor_file_sha256']}/SUBMISSION_ACQUIRED.json"
        )
    raise AssertionError(f"unknown record label: {label}")


def _artifact(
    label: str,
    value: dict[str, object],
    *,
    key: str | None = None,
    raw: bytes | None = None,
    file_sha256: str | None = None,
) -> object:
    raw_value = _canonical(value) + b"\n" if raw is None else raw
    return _module().ImmutableJsonArtifact(
        key=_record_key(label, value) if key is None else key,
        raw=raw_value,
        file_sha256=_sha(raw_value) if file_sha256 is None else file_sha256,
    )


def _source_artifacts(
    records: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        label: _artifact(label, records[label])
        for label in ("readiness", "intent", "baseline", "control_plane")
    }


def _repin_control_plane_baseline(
    records: dict[str, dict[str, object]],
) -> None:
    baseline = records["baseline"]
    control_plane = records["control_plane"]
    control_plane["controller_baseline_key"] = (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )
    control_plane["controller_baseline_file_sha256"] = _file_sha(baseline)
    control_plane["controller_baseline_body_sha256"] = baseline["baseline_body_sha256"]
    records["control_plane"] = _rehash(
        control_plane,
        "control_plane_ready_body_sha256",
    )


def _repin_live_authorities_to_intent(
    records: dict[str, dict[str, object]],
) -> None:
    intent = records["intent"]
    intent_key = _record_key("intent", intent)
    baseline = records["baseline"]
    baseline["intent_key"] = intent_key
    baseline["intent_file_sha256"] = _file_sha(intent)
    baseline["intent_body_sha256"] = intent["intent_body_sha256"]
    records["baseline"] = _rehash(baseline, "baseline_body_sha256")

    control_plane = records["control_plane"]
    control_plane["intent_key"] = intent_key
    control_plane["intent_file_sha256"] = _file_sha(intent)
    control_plane["intent_body_sha256"] = intent["intent_body_sha256"]
    _repin_control_plane_baseline(records)


def _records() -> dict[str, dict[str, object]]:
    descriptor_key = (
        f"campaigns/{RUN_ID}/submissions/qualification/campaign-descriptor-v2.json"
    )
    descriptor_file_sha256 = _h("a")
    descriptor_body_sha256 = _h("b")
    campaign_identity_sha256 = _h("c")
    approval_sha256 = _h("d")
    approval_body_sha256 = _h("e")
    repo_tar_sha256 = _h("f")
    cache_acceptance_body_sha256 = _h("1")
    cache_acceptance_key = (
        f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/"
        f"{cache_acceptance_body_sha256}/"
        "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
    )
    spend_body_sha256 = _h("2")
    spend_key = (
        f"campaigns/{RUN_ID}/spend-snapshots/{spend_body_sha256}/"
        "GPU_SPEND_SNAPSHOT.json"
    )

    readiness_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_qualification_submission_ready_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_body_sha256,
        "campaign_identity_sha256": campaign_identity_sha256,
        "approval_sha256": approval_sha256,
        "approval_body_sha256": approval_body_sha256,
        "repo_tar_sha256": repo_tar_sha256,
        "seed_descriptor_key": (
            f"campaigns/{RUN_ID}/submissions/cache-seed/campaign-descriptor-v2.json"
        ),
        "seed_descriptor_file_sha256": _h("3"),
        "seed_descriptor_body_sha256": _h("4"),
        "seed_campaign_identity_sha256": _h("5"),
        "seed_repo_tar_sha256": _h("6"),
        "seed_descriptor": {
            "schema_version": 2,
            "record_type": "glm52_sky_campaign_v2",
            "run_id": RUN_ID,
            "managed_mode": "cache-seed",
        },
        "staged_readiness_key": (
            f"campaigns/{RUN_ID}/staged/STAGED_CONTROL_PLANE_READY.json"
        ),
        "staged_readiness_file_sha256": _h("7"),
        "staged_readiness_body_sha256": _h("8"),
        "cache_seed_acceptance_key": cache_acceptance_key,
        "cache_seed_acceptance_file_sha256": _h("9"),
        "cache_seed_acceptance_body_sha256": cache_acceptance_body_sha256,
        "gpu_spend_snapshot_key": spend_key,
        "gpu_spend_snapshot_sha256": _h("0"),
        "gpu_spend_snapshot_body_sha256": spend_body_sha256,
        "gpu_spend_ledger_tip_record_sha256": _h("a"),
        "remaining_gpu_seconds": 82_800,
        "remaining_gpu_cost_usd": 1265.92,
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "rehearsal_evidence_key": (
            f"campaigns/{RUN_ID}/qualification/rehearsals/{_h('b')}/"
            "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "rehearsal_evidence_sha256": _h("c"),
        "rehearsal_evidence_body_sha256": _h("b"),
        "sky_task_name": "glm52-sky-campaign",
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(MUST_START_BY),
        "allowed_gpu_seconds": 14_400,
        "allowed_gpu_cost_usd": 220.16,
        "built_at": _iso(READINESS_BUILT_AT),
    }
    readiness = _self_hashed(readiness_body, "readiness_body_sha256")
    readiness_key = (
        f"campaigns/{RUN_ID}/qualification/submission-ready/"
        f"{readiness['readiness_body_sha256']}/"
        "QUALIFICATION_SUBMISSION_READY.json"
    )

    intent_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_submission_intent_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_body_sha256,
        "campaign_identity_sha256": campaign_identity_sha256,
        "approval_sha256": approval_sha256,
        "approval_body_sha256": approval_body_sha256,
        "repo_tar_sha256": repo_tar_sha256,
        "cache_seed_acceptance_key": cache_acceptance_key,
        "cache_seed_acceptance_file_sha256": _h("9"),
        "cache_seed_acceptance_body_sha256": cache_acceptance_body_sha256,
        "gpu_spend_snapshot_key": spend_key,
        "gpu_spend_snapshot_sha256": _h("0"),
        "gpu_spend_snapshot_body_sha256": spend_body_sha256,
        "gpu_spend_ledger_tip_record_sha256": _h("a"),
        "remaining_gpu_seconds": 82_800,
        "remaining_gpu_cost_usd": 1265.92,
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "open_allocation_count": 0,
        "qualification_submission_ready_key": readiness_key,
        "qualification_submission_ready_sha256": _file_sha(readiness),
        "qualification_submission_ready_body_sha256": readiness[
            "readiness_body_sha256"
        ],
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(MUST_START_BY),
        "intent_at": _iso(INTENT_AT),
    }
    intent = _self_hashed(intent_body, "intent_body_sha256")
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/{MODE}/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )

    baseline_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_controller_baseline_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "bucket": "keep-glm52-us-west-2-246813579024",
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "intent_key": intent_key,
        "intent_file_sha256": _file_sha(intent),
        "intent_body_sha256": intent["intent_body_sha256"],
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_instance_id": "i-0511af4e31aa5406a",
        "controller_instance_type": "c6a.xlarge",
        "controller_profile_arn": (
            "arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller"
        ),
        "controller_cluster_name": "sky-jobs-controller-9d9f31a9",
        "ssm_ping_status": "Online",
        "exact_name_history": [],
        "active_exact_name_job_ids": [],
        "active_tagged_p5_instance_ids": [],
        "observed_at": _iso(BASELINE_OBSERVED_AT),
    }
    baseline = _self_hashed(baseline_body, "baseline_body_sha256")
    baseline_key = (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )

    control_plane_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_must_start_control_plane_ready_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "bucket": "keep-glm52-us-west-2-246813579024",
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": descriptor_key,
        "descriptor_file_sha256": descriptor_file_sha256,
        "descriptor_body_sha256": descriptor_body_sha256,
        "intent_key": intent_key,
        "intent_file_sha256": _file_sha(intent),
        "intent_body_sha256": intent["intent_body_sha256"],
        "controller_baseline_key": baseline_key,
        "controller_baseline_file_sha256": _file_sha(baseline),
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-sky-control-plane/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ),
        "template_sha256": _h("d"),
        "lambda_function_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:keep-glm52-sky-must-start"
        ),
        "lambda_code_sha256": _h("e"),
        "iam_policy_sha256": _h("f"),
        "reconciliation_rule_arn": (
            "arn:aws:events:us-west-2:246813579024:rule/keep-glm52-sky-reconcile"
        ),
        "reconciliation_rule_state": "ENABLED",
        "deadline_schedule_arn": (
            "arn:aws:scheduler:us-west-2:246813579024:schedule/default/"
            "keep-glm52-sky-deadline"
        ),
        "deadline_schedule_state": "ENABLED",
        "dlq_arn": ("arn:aws:sqs:us-west-2:246813579024:keep-glm52-sky-must-start-dlq"),
        "coordinator_mode": "dynamic-job-binding-active",
        "observed_at": _iso(CONTROL_PLANE_OBSERVED_AT),
    }
    control_plane = _self_hashed(
        control_plane_body,
        "control_plane_ready_body_sha256",
    )
    return {
        "readiness": readiness,
        "intent": intent,
        "baseline": baseline,
        "control_plane": control_plane,
    }


def _build(
    records: dict[str, dict[str, object]] | None = None,
    *,
    acquired_at: datetime | str = ACQUIRED_AT,
) -> dict[str, object]:
    values = records or _records()
    artifacts = _source_artifacts(values)
    return _module().build_submission_acquired(
        intent=artifacts["intent"],
        qualification_submission_ready=artifacts["readiness"],
        controller_baseline=artifacts["baseline"],
        must_start_control_plane_ready=artifacts["control_plane"],
        acquired_at=acquired_at,
    )


def _validate(
    acquisition: dict[str, object],
    records: dict[str, dict[str, object]] | None = None,
    *,
    now: datetime | str = ACQUIRED_AT,
    acquisition_artifact: object | None = None,
) -> dict[str, object]:
    values = records or _records()
    artifacts = _source_artifacts(values)
    return _module().validate_submission_acquired(
        (
            _artifact("acquisition", acquisition)
            if acquisition_artifact is None
            else acquisition_artifact
        ),
        intent=artifacts["intent"],
        qualification_submission_ready=artifacts["readiness"],
        controller_baseline=artifacts["baseline"],
        must_start_control_plane_ready=artifacts["control_plane"],
        now=now,
    )


def _decision(
    candidate: dict[str, object],
    *,
    selectors: object,
    records: dict[str, dict[str, object]] | None = None,
    selected_records: dict[str, dict[str, object]] | None = None,
    now: datetime | str = ACQUIRED_AT,
    read_result_override: object | None = None,
) -> object:
    values = records or _records()
    artifacts = _source_artifacts(values)
    candidate_artifact = _artifact("acquisition", candidate)
    expected_key = _record_key("acquisition", candidate)
    prefix = expected_key.rsplit("/", 1)[0] + "/"
    if read_result_override is not None:
        read_result: object = read_result_override
    elif isinstance(selectors, (list, tuple)):
        selector_artifacts = [
            _artifact("acquisition", selector) for selector in selectors
        ]
        read_result = _module().SubmissionAcquisitionReadResult(
            requested_key=expected_key,
            result="exact-404" if not selector_artifacts else "found",
            http_status=404 if not selector_artifacts else 200,
            artifact=None if not selector_artifacts else selector_artifacts[0],
            listed_prefix=prefix,
            prefix_keys=tuple(
                artifact.key  # type: ignore[attr-defined]
                for artifact in selector_artifacts
            ),
            listing_complete=True,
            observed_at=_iso(
                now if isinstance(now, datetime) else datetime.fromisoformat(now)
            ),
        )
    else:
        read_result = selectors
    selected = (
        _source_artifacts(selected_records) if selected_records is not None else None
    )
    return _module().decide_submission_acquisition(
        candidate=candidate_artifact,
        read_result=read_result,
        intent=artifacts["intent"],
        qualification_submission_ready=artifacts["readiness"],
        controller_baseline=artifacts["baseline"],
        must_start_control_plane_ready=artifacts["control_plane"],
        selected_intent=None if selected is None else selected["intent"],
        selected_qualification_submission_ready=(
            None if selected is None else selected["readiness"]
        ),
        selected_controller_baseline=(
            None if selected is None else selected["baseline"]
        ),
        selected_must_start_control_plane_ready=(
            None if selected is None else selected["control_plane"]
        ),
        now=now,
    )


def test_builds_exact_deterministic_selector_and_descriptor_addressed_key() -> None:
    module = _module()
    records = _records()

    first = _build(records)
    second = _build(records)

    assert first == second
    assert set(first) == ACQUISITION_FIELDS
    assert first["schema_version"] == 1
    assert first["record_type"] == "glm52_sky_submission_acquired_v1"
    body = dict(first)
    digest = body.pop("acquisition_body_sha256")
    assert digest == _sha(_canonical(body))
    assert _validate(first, records) == first
    assert module.submission_acquired_s3_key(
        run_id=RUN_ID,
        managed_mode=MODE,
        descriptor_file_sha256=_h("a"),
    ) == (
        f"campaigns/{RUN_ID}/submissions/qualification/acquisitions/"
        f"{_h('a')}/SUBMISSION_ACQUIRED.json"
    )
    assert _canonical(first).endswith(b"}")
    assert not _canonical(first).endswith(b"\n")
    assert module.submission_acquired_file_bytes(first) == _canonical(first) + b"\n"
    assert module.submission_acquired_file_sha256(first) == _file_sha(first)


@pytest.mark.parametrize("variant", ["wrong-key", "no-lf", "pretty", "bad-file-sha"])
def test_validator_authenticates_actual_selector_key_raw_bytes_and_file_digest(
    variant: str,
) -> None:
    acquisition = _build()
    artifact = _artifact("acquisition", acquisition)
    if variant == "wrong-key":
        artifact = _artifact(
            "acquisition",
            acquisition,
            key=(
                f"campaigns/{RUN_ID}/submissions/{MODE}/acquisitions/"
                f"{_h('b')}/SUBMISSION_ACQUIRED.json"
            ),
        )
    elif variant == "no-lf":
        raw = _canonical(acquisition)
        artifact = _artifact(
            "acquisition",
            acquisition,
            raw=raw,
            file_sha256=_sha(raw),
        )
    elif variant == "pretty":
        raw = json.dumps(acquisition, indent=2, sort_keys=True).encode() + b"\n"
        artifact = _artifact(
            "acquisition",
            acquisition,
            raw=raw,
            file_sha256=_sha(raw),
        )
    else:
        artifact = _artifact(
            "acquisition",
            acquisition,
            file_sha256=_h("f"),
        )

    with pytest.raises(ValueError, match="key|canonical|file SHA-256"):
        _validate(acquisition, acquisition_artifact=artifact)


def test_builder_rejects_normalized_source_mapping_without_exact_lf_bytes() -> None:
    records = _records()
    artifacts = _source_artifacts(records)
    raw = _canonical(records["intent"])
    artifacts["intent"] = _artifact(
        "intent",
        records["intent"],
        raw=raw,
        file_sha256=_sha(raw),
    )

    with pytest.raises(ValueError, match="canonical.*newline"):
        _module().build_submission_acquired(
            intent=artifacts["intent"],
            qualification_submission_ready=artifacts["readiness"],
            controller_baseline=artifacts["baseline"],
            must_start_control_plane_ready=artifacts["control_plane"],
            acquired_at=ACQUIRED_AT,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("region"), "schema mismatch"),
        (
            lambda value: value.__setitem__("owner_token", "not-stored"),
            "schema mismatch",
        ),
        (
            lambda value: value.__setitem__(
                "record_type",
                "glm52_skypilot_submission_v1",
            ),
            "schema mismatch",
        ),
        (lambda value: value.__setitem__("schema_version", True), "schema mismatch"),
    ],
)
def test_rejects_missing_unknown_mixed_and_boolean_schema_fields(
    mutation: object,
    match: str,
) -> None:
    acquisition = _build()
    mutation(acquisition)  # type: ignore[operator]
    acquisition = _rehash(acquisition, "acquisition_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate(acquisition)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("descriptor_file_sha256", "A" * 64, "lowercase SHA-256"),
        (
            "acquired_at",
            "2026-07-26T12:00:30+00:00",
            "canonical UTC",
        ),
        ("managed_mode", "production", "qualification"),
    ],
)
def test_rejects_noncanonical_hash_time_and_mixed_mode(
    field: str,
    replacement: object,
    match: str,
) -> None:
    acquisition = _build()
    acquisition[field] = replacement
    acquisition = _rehash(acquisition, "acquisition_body_sha256")

    with pytest.raises(ValueError, match=match):
        _validate(acquisition)


@pytest.mark.parametrize(
    ("run_id", "managed_mode", "descriptor_sha"),
    [
        ("../foreign", MODE, _h("a")),
        (RUN_ID, "../qualification", _h("a")),
        (RUN_ID, MODE, "A" * 64),
    ],
)
def test_key_helper_rejects_traversal_and_noncanonical_components(
    run_id: str,
    managed_mode: str,
    descriptor_sha: str,
) -> None:
    with pytest.raises(ValueError):
        _module().submission_acquired_s3_key(
            run_id=run_id,
            managed_mode=managed_mode,
            descriptor_file_sha256=descriptor_sha,
        )


def test_rejects_descriptor_key_traversal_even_when_sources_are_rehashed() -> None:
    records = _records()
    records["intent"]["descriptor_key"] = (
        f"campaigns/{RUN_ID}/submissions/../foreign.json"
    )
    records["intent"] = _rehash(records["intent"], "intent_body_sha256")

    with pytest.raises(ValueError, match="descriptor_key"):
        _build(records)


@pytest.mark.parametrize(
    ("readiness_field", "intent_field"),
    [
        ("descriptor_body_sha256", "descriptor_body_sha256"),
        ("gpu_spend_snapshot_key", "gpu_spend_snapshot_key"),
        ("gpu_spend_snapshot_sha256", "gpu_spend_snapshot_sha256"),
        ("gpu_spend_snapshot_body_sha256", "gpu_spend_snapshot_body_sha256"),
        (
            "gpu_spend_ledger_tip_record_sha256",
            "gpu_spend_ledger_tip_record_sha256",
        ),
        ("sky_job_name", "sky_job_name"),
        ("must_start_by", "must_start_by"),
    ],
)
def test_rejects_readiness_intent_and_transitive_spend_drift(
    readiness_field: str,
    intent_field: str,
) -> None:
    records = _records()
    original = records["readiness"][readiness_field]
    if readiness_field == "must_start_by":
        records["readiness"][readiness_field] = _iso(
            MUST_START_BY - timedelta(minutes=1)
        )
    else:
        records["readiness"][readiness_field] = (
            _h("f") if isinstance(original, str) and len(original) == 64 else "foreign"
        )
    records["readiness"] = _rehash(
        records["readiness"],
        "readiness_body_sha256",
    )
    assert records["readiness"][readiness_field] != records["intent"][intent_field]

    with pytest.raises(ValueError, match="drift|mismatch|exact"):
        _build(records)


def test_rejects_intent_readiness_file_hash_lie() -> None:
    records = _records()
    records["intent"]["qualification_submission_ready_sha256"] = _h("f")
    records["intent"] = _rehash(records["intent"], "intent_body_sha256")

    with pytest.raises(ValueError, match="readiness file SHA-256"):
        _build(records)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("active_exact_name_job_ids", [17], "active exact-name job"),
        (
            "active_tagged_p5_instance_ids",
            ["i-0611af4e31aa5406a"],
            "active tagged P5",
        ),
    ],
)
def test_rejects_nonempty_controller_baseline(
    field: str,
    value: object,
    match: str,
) -> None:
    records = _records()
    records["baseline"][field] = value
    records["baseline"] = _rehash(records["baseline"], "baseline_body_sha256")

    with pytest.raises(ValueError, match=match):
        _build(records)


def test_rejects_stale_controller_baseline() -> None:
    records = _records()

    with pytest.raises(ValueError, match="older than 60 seconds"):
        _build(
            records,
            acquired_at=BASELINE_OBSERVED_AT + timedelta(seconds=61),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("coordinator_mode", "legacy-static-job-id", "coordinator_mode"),
        ("reconciliation_rule_state", "DISABLED", "must be ENABLED"),
        ("deadline_schedule_state", "DISABLED", "must be ENABLED"),
    ],
)
def test_rejects_inactive_or_legacy_must_start_control_plane(
    field: str,
    value: object,
    match: str,
) -> None:
    records = _records()
    records["control_plane"][field] = value
    records["control_plane"] = _rehash(
        records["control_plane"],
        "control_plane_ready_body_sha256",
    )

    with pytest.raises(ValueError, match=match):
        _build(records)


def test_rejects_boolean_controller_history_job_id() -> None:
    records = _records()
    records["baseline"]["exact_name_history"] = [
        {
            "sky_job_id": True,
            "sky_job_name": SKY_JOB_NAME,
            "workspace": "default",
            "controller_submitted_at": "2026-07-25T10:00:00Z",
            "controller_status": "SUCCEEDED",
            "controller_identity": (
                "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
            ),
        }
    ]
    records["baseline"] = _rehash(records["baseline"], "baseline_body_sha256")

    with pytest.raises(ValueError, match="sky_job_id"):
        _build(records)


def test_rejects_active_history_row_hidden_by_empty_active_id_list() -> None:
    records = _records()
    records["baseline"]["exact_name_history"] = [
        {
            "sky_job_id": 17,
            "sky_job_name": SKY_JOB_NAME,
            "workspace": "default",
            "controller_submitted_at": "2026-07-26T11:58:00Z",
            "controller_status": "RUNNING",
            "controller_identity": (
                "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
            ),
        }
    ]
    records["baseline"] = _rehash(records["baseline"], "baseline_body_sha256")
    _repin_control_plane_baseline(records)

    with pytest.raises(ValueError, match="active exact-name"):
        _build(records)


def test_rejects_source_and_selector_hash_drift() -> None:
    records = _records()
    acquisition = _build(records)
    records["baseline"]["controller_cluster_name"] = "foreign-controller"
    records["baseline"] = _rehash(records["baseline"], "baseline_body_sha256")

    with pytest.raises(ValueError, match="baseline"):
        _validate(acquisition, records)


def test_rejects_live_authority_bucket_drift() -> None:
    records = _records()
    records["control_plane"]["bucket"] = "foreign-valid-bucket"
    records["control_plane"] = _rehash(
        records["control_plane"],
        "control_plane_ready_body_sha256",
    )

    with pytest.raises(ValueError, match="bucket"):
        _build(records)


def test_rejects_acquisition_at_or_after_deadline_and_expired_validation() -> None:
    with pytest.raises(ValueError, match="before must_start_by"):
        _build(acquired_at=MUST_START_BY)

    acquisition = _build()
    with pytest.raises(ValueError, match="expired"):
        _validate(acquisition, now=MUST_START_BY)


def test_absent_selector_requests_conditional_create_without_launch_authority() -> None:
    candidate = _build()

    decision = _decision(candidate, selectors=[])

    assert decision.state == "absent"
    assert decision.action == "conditional-create"
    assert decision.selector is None


def test_identical_stored_winner_is_idempotent_reconciliation_only() -> None:
    candidate = _build()

    first = _decision(candidate, selectors=[deepcopy(candidate)])
    second = _decision(candidate, selectors=(deepcopy(candidate),))

    assert first == second
    assert first.state == "winner"
    assert first.action == "reconcile-only"
    assert first.selector == candidate
    assert "never launch" in first.reason


def test_different_valid_selector_at_same_key_is_the_reconciliation_winner() -> None:
    candidate = _build()
    competing = dict(candidate)
    competing["acquired_at"] = _iso(ACQUIRED_AT + timedelta(seconds=1))
    competing = _rehash(competing, "acquisition_body_sha256")

    decision = _decision(
        candidate,
        selectors=[competing],
        now=ACQUIRED_AT + timedelta(seconds=2),
    )

    assert decision.state == "winner"
    assert decision.action == "reconcile-only"
    assert decision.selector == competing


def test_different_intent_winner_authenticates_from_selected_sources() -> None:
    candidate = _build()
    selected_records = deepcopy(_records())
    selected_records["intent"]["intent_at"] = _iso(INTENT_AT + timedelta(seconds=1))
    selected_records["intent"] = _rehash(
        selected_records["intent"],
        "intent_body_sha256",
    )
    _repin_live_authorities_to_intent(selected_records)
    selected = _build(
        selected_records,
        acquired_at=ACQUIRED_AT + timedelta(seconds=1),
    )

    without_selected_sources = _decision(
        candidate,
        selectors=[selected],
        now=ACQUIRED_AT + timedelta(seconds=2),
    )
    with_selected_sources = _decision(
        candidate,
        selectors=[selected],
        selected_records=selected_records,
        now=ACQUIRED_AT + timedelta(seconds=2),
    )

    assert without_selected_sources.state == "conflict"
    assert without_selected_sources.action == "fail-closed"
    assert with_selected_sources.state == "winner"
    assert with_selected_sources.action == "reconcile-only"
    assert with_selected_sources.selector == selected


def test_multiple_selectors_fail_closed_even_when_byte_identical() -> None:
    candidate = _build()

    decision = _decision(
        candidate,
        selectors=[deepcopy(candidate), deepcopy(candidate)],
    )

    assert decision.state == "conflict"
    assert decision.action == "fail-closed"
    assert "multiple" in decision.reason


def test_exact_404_candidate_rechecks_baseline_freshness_at_put_decision_time() -> None:
    candidate = _build()

    decision = _decision(
        candidate,
        selectors=[],
        now=BASELINE_OBSERVED_AT + timedelta(seconds=61),
    )

    assert decision.state == "conflict"
    assert decision.action == "fail-closed"
    assert "fresh" in decision.reason


def test_stored_winner_remains_reconciliation_only_after_baseline_ages() -> None:
    candidate = _build()

    decision = _decision(
        candidate,
        selectors=[deepcopy(candidate)],
        now=BASELINE_OBSERVED_AT + timedelta(seconds=61),
    )

    assert decision.state == "winner"
    assert decision.action == "reconcile-only"


def test_expired_decision_is_distinct_and_never_reconciles_or_acquires() -> None:
    candidate = _build()

    decision = _decision(
        candidate,
        selectors=[],
        now=MUST_START_BY,
    )

    assert decision.state == "expired"
    assert decision.action == "fail-closed"
    assert decision.selector is None


@pytest.mark.parametrize(
    "variant",
    [
        "non-404",
        "incomplete-list",
        "unexpected-prefix-member",
        "wrong-requested-key",
        "stale-read",
    ],
)
def test_read_provenance_must_be_exact_and_prefix_complete(variant: str) -> None:
    candidate = _build()
    module = _module()
    key = _record_key("acquisition", candidate)
    prefix = key.rsplit("/", 1)[0] + "/"
    values: dict[str, object] = {
        "requested_key": key,
        "result": "exact-404",
        "http_status": 404,
        "artifact": None,
        "listed_prefix": prefix,
        "prefix_keys": (),
        "listing_complete": True,
        "observed_at": _iso(ACQUIRED_AT),
    }
    if variant == "non-404":
        values["http_status"] = 500
    elif variant == "incomplete-list":
        values["listing_complete"] = False
    elif variant == "unexpected-prefix-member":
        values["prefix_keys"] = (f"{prefix}FOREIGN.json",)
    elif variant == "wrong-requested-key":
        values["requested_key"] = f"{prefix}FOREIGN.json"
    else:
        values["observed_at"] = _iso(ACQUIRED_AT - timedelta(seconds=1))
    read_result = module.SubmissionAcquisitionReadResult(**values)

    decision = _decision(
        candidate,
        selectors=[],
        read_result_override=read_result,
    )

    assert decision.state == "conflict"
    assert decision.action == "fail-closed"
    assert "read" in decision.reason


def test_non_read_result_input_fails_closed() -> None:
    candidate = _build()

    decision = _decision(candidate, selectors={"selector": candidate})

    assert decision.state == "conflict"
    assert decision.action == "fail-closed"
    assert "read" in decision.reason
