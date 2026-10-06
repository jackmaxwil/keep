"""Pure dynamic-v2 numeric Sky job binding decisions."""

from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

RUN_ID = "glm52-sky-20260726"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
MODE = "qualification"
SKY_JOB_NAME = "glm52-sky-20260726-qualification"
DESCRIPTOR_KEY = (
    "campaigns/glm52-sky-20260726/submissions/qualification/campaign-descriptor-v2.json"
)
CONTROLLER_IDENTITY = "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
INTENT_AT = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
DEADLINE = datetime(2026, 7, 26, 13, 0, tzinfo=UTC)


def _module() -> Any:
    try:
        return importlib.import_module("mlx_vq.quality.glm52_sky_must_start_dynamic")
    except ModuleNotFoundError:
        pytest.fail("dynamic-v2 resolver behavior is not implemented")


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    ).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _hash_record(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: _sha(body)}


def _h(character: str) -> str:
    return character * 64


def _authorities() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Hand-built authenticated dynamic-v2 authorities for one exact run."""

    intent_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_submission_intent_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": DESCRIPTOR_KEY,
        "descriptor_file_sha256": _h("a"),
        "descriptor_body_sha256": _h("b"),
        "campaign_identity_sha256": _h("c"),
        "approval_sha256": _h("d"),
        "approval_body_sha256": _h("e"),
        "repo_tar_sha256": _h("f"),
        "cache_seed_acceptance_key": (
            f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/{_h('2')}/"
            "QUALIFICATION_CACHE_SEED_ACCEPTED.json"
        ),
        "cache_seed_acceptance_file_sha256": _h("1"),
        "cache_seed_acceptance_body_sha256": _h("2"),
        "gpu_spend_snapshot_key": "/".join(
            (
                f"campaigns/{RUN_ID}/spend-snapshots/{_h('4')}",
                "GPU_SPEND_SNAPSHOT.json",
            )
        ),
        "gpu_spend_snapshot_sha256": _h("3"),
        "gpu_spend_snapshot_body_sha256": _h("4"),
        "gpu_spend_ledger_tip_record_sha256": _h("5"),
        "remaining_gpu_seconds": 82800,
        "remaining_gpu_cost_usd": 1265.92,
        "qualification_allowance_seconds": 14400,
        "qualification_allowance_cost_usd": 220.16,
        "open_allocation_count": 0,
        "qualification_submission_ready_key": (
            f"campaigns/{RUN_ID}/qualification/submission-ready/{_h('7')}/"
            "QUALIFICATION_SUBMISSION_READY.json"
        ),
        "qualification_submission_ready_sha256": _h("6"),
        "qualification_submission_ready_body_sha256": _h("7"),
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(DEADLINE),
        "intent_at": _iso(INTENT_AT),
    }
    intent = _hash_record(intent_body, "intent_body_sha256")
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
        "controller_profile_arn": "/".join(
            (
                "arn:aws:iam::246813579024:instance-profile",
                "keep-glm52-skypilot-controller",
            )
        ),
        "controller_cluster_name": "sky-jobs-controller-9d9f31a9",
        "ssm_ping_status": "Online",
        "exact_name_history": [],
        "active_exact_name_job_ids": [],
        "active_tagged_p5_instance_ids": [],
        "observed_at": _iso(INTENT_AT + timedelta(seconds=20)),
    }
    baseline = _hash_record(baseline_body, "baseline_body_sha256")
    baseline_key = (
        f"campaigns/{RUN_ID}/qualification/controller-baselines/"
        f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
    )

    acquisition_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_submission_acquired_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": DESCRIPTOR_KEY,
        "descriptor_file_sha256": _h("a"),
        "descriptor_body_sha256": _h("b"),
        "qualification_submission_ready_key": intent_body[
            "qualification_submission_ready_key"
        ],
        "qualification_submission_ready_sha256": _h("6"),
        "qualification_submission_ready_body_sha256": _h("7"),
        "intent_key": intent_key,
        "intent_file_sha256": _file_sha(intent),
        "intent_body_sha256": intent["intent_body_sha256"],
        "controller_baseline_key": baseline_key,
        "controller_baseline_file_sha256": _file_sha(baseline),
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        "must_start_control_plane_ready_key": (
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/control-plane-ready/{_h('a')}/"
            "CONTROL_PLANE_READY.json"
        ),
        "must_start_control_plane_ready_file_sha256": _h("0"),
        "must_start_control_plane_ready_body_sha256": _h("a"),
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(DEADLINE),
        "acquired_at": _iso(INTENT_AT + timedelta(seconds=30)),
    }
    return intent, baseline, _hash_record(acquisition_body, "acquisition_body_sha256")


def _new_row(job_id: int = 41) -> dict[str, object]:
    return {
        "sky_job_id": job_id,
        "sky_job_name": SKY_JOB_NAME,
        "workspace": "default",
        "controller_submitted_at": _iso(INTENT_AT + timedelta(seconds=31)),
        "controller_status": "PENDING",
        "controller_identity": (
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
    }


def _binding(
    intent: dict[str, object],
    baseline: dict[str, object],
    acquisition: dict[str, object],
    row: dict[str, object],
) -> dict[str, object]:
    """A hand-derived v2 JOB_BINDING.json for the exact observed row."""

    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_must_start_job_binding_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "managed_mode": MODE,
        "descriptor_key": DESCRIPTOR_KEY,
        "descriptor_file_sha256": _h("a"),
        "descriptor_body_sha256": _h("b"),
        "intent_key": acquisition["intent_key"],
        "intent_body_sha256": intent["intent_body_sha256"],
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
        "acquisition_body_sha256": acquisition["acquisition_body_sha256"],
        "sky_job_name": SKY_JOB_NAME,
        "must_start_by": _iso(DEADLINE),
        "workspace": baseline["workspace"],
        "controller_instance_id": baseline["controller_instance_id"],
        "controller_instance_type": baseline["controller_instance_type"],
        "controller_profile_arn": baseline["controller_profile_arn"],
        "controller_cluster_name": baseline["controller_cluster_name"],
        "sky_job_id": row["sky_job_id"],
        "controller_submitted_at": row["controller_submitted_at"],
        "controller_status": row["controller_status"],
        "controller_identity": row["controller_identity"],
        "bound_at": _iso(INTENT_AT + timedelta(minutes=2)),
    }
    return _hash_record(body, "job_binding_body_sha256")


def _rehash_binding(binding: dict[str, object]) -> dict[str, object]:
    return _hash_record(
        {
            key: value
            for key, value in binding.items()
            if key != "job_binding_body_sha256"
        },
        "job_binding_body_sha256",
    )


def _production_authorities() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    """Rebind the literal authorities to production without readiness policy."""

    intent, baseline, acquisition = _authorities()
    intent["managed_mode"] = "production"
    intent["sky_job_name"] = RUN_ID
    intent = _hash_record(
        {key: value for key, value in intent.items() if key != "intent_body_sha256"},
        "intent_body_sha256",
    )
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/production/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    baseline["managed_mode"] = "production"
    baseline["sky_job_name"] = RUN_ID
    baseline["intent_key"] = intent_key
    baseline["intent_body_sha256"] = intent["intent_body_sha256"]
    baseline = _hash_record(
        {
            key: value
            for key, value in baseline.items()
            if key != "baseline_body_sha256"
        },
        "baseline_body_sha256",
    )
    for field, value in {
        "managed_mode": "production",
        "sky_job_name": RUN_ID,
        "intent_key": intent_key,
        "intent_body_sha256": intent["intent_body_sha256"],
        "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
    }.items():
        acquisition[field] = value
    acquisition = _hash_record(
        {
            key: value
            for key, value in acquisition.items()
            if key != "acquisition_body_sha256"
        },
        "acquisition_body_sha256",
    )
    return intent, baseline, acquisition


def _repin_authority_chain(
    intent: dict[str, object],
    baseline: dict[str, object],
    acquisition: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Rehash and cross-pin a deliberately mutated authority chain."""

    intent = _hash_record(
        {key: value for key, value in intent.items() if key != "intent_body_sha256"},
        "intent_body_sha256",
    )
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/{intent['managed_mode']}/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    baseline["intent_key"] = intent_key
    baseline["intent_body_sha256"] = intent["intent_body_sha256"]
    baseline = _hash_record(
        {
            key: value
            for key, value in baseline.items()
            if key != "baseline_body_sha256"
        },
        "baseline_body_sha256",
    )
    acquisition["intent_key"] = intent_key
    acquisition["intent_body_sha256"] = intent["intent_body_sha256"]
    acquisition["controller_baseline_body_sha256"] = baseline["baseline_body_sha256"]
    acquisition = _hash_record(
        {
            key: value
            for key, value in acquisition.items()
            if key != "acquisition_body_sha256"
        },
        "acquisition_body_sha256",
    )
    return intent, baseline, acquisition


def _repin_strict_authority_chain(
    intent: dict[str, object],
    baseline: dict[str, object],
    acquisition: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Rehash all exact file/body/key pins after a deliberate intent mutation."""

    intent = _hash_record(
        {key: value for key, value in intent.items() if key != "intent_body_sha256"},
        "intent_body_sha256",
    )
    intent_key = (
        f"campaigns/{RUN_ID}/submissions/{intent['managed_mode']}/intents/"
        f"{intent['intent_body_sha256']}/SKYPILOT_SUBMISSION_INTENT.json"
    )
    baseline["intent_key"] = intent_key
    baseline["intent_file_sha256"] = _file_sha(intent)
    baseline["intent_body_sha256"] = intent["intent_body_sha256"]
    baseline = _hash_record(
        {
            key: value
            for key, value in baseline.items()
            if key != "baseline_body_sha256"
        },
        "baseline_body_sha256",
    )
    acquisition.update(
        {
            "descriptor_key": intent["descriptor_key"],
            "descriptor_file_sha256": intent["descriptor_file_sha256"],
            "descriptor_body_sha256": intent["descriptor_body_sha256"],
            "qualification_submission_ready_key": intent[
                "qualification_submission_ready_key"
            ],
            "qualification_submission_ready_sha256": intent[
                "qualification_submission_ready_sha256"
            ],
            "qualification_submission_ready_body_sha256": intent[
                "qualification_submission_ready_body_sha256"
            ],
            "intent_key": intent_key,
            "intent_file_sha256": _file_sha(intent),
            "intent_body_sha256": intent["intent_body_sha256"],
            "controller_baseline_key": (
                f"campaigns/{RUN_ID}/qualification/controller-baselines/"
                f"{baseline['baseline_body_sha256']}/CONTROLLER_BASELINE.json"
            ),
            "controller_baseline_file_sha256": _file_sha(baseline),
            "controller_baseline_body_sha256": baseline["baseline_body_sha256"],
            "must_start_control_plane_ready_key": (
                f"campaigns/{RUN_ID}/monitor/must-start/{intent['managed_mode']}/"
                f"{intent['intent_body_sha256']}/control-plane-ready/"
                f"{acquisition['must_start_control_plane_ready_body_sha256']}/"
                "CONTROL_PLANE_READY.json"
            ),
        }
    )
    acquisition = _hash_record(
        {
            key: value
            for key, value in acquisition.items()
            if key != "acquisition_body_sha256"
        },
        "acquisition_body_sha256",
    )
    return intent, baseline, acquisition


def _build_binding(
    *,
    intent: dict[str, object] | None = None,
    baseline: dict[str, object] | None = None,
    acquisition: dict[str, object] | None = None,
    history: list[dict[str, object]] | None = None,
    bound_at: datetime = INTENT_AT + timedelta(minutes=2),
    now: datetime = INTENT_AT + timedelta(minutes=3),
) -> dict[str, object]:
    module = _module()
    exact_intent, exact_baseline, exact_acquisition = _authorities()
    return module.build_dynamic_v2_job_binding(
        intent=exact_intent if intent is None else intent,
        controller_baseline=exact_baseline if baseline is None else baseline,
        acquisition=exact_acquisition if acquisition is None else acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()] if history is None else history,
        bound_at=bound_at,
        now=now,
    )


def test_public_canonical_bytes_are_exact_ascii_json_without_a_newline() -> None:
    """Catches a public encoder that changes the dynamic-v2 hash byte contract."""

    module = _module()

    assert (
        module.dynamic_v2_canonical_bytes({"z": [1, True, None], "a": "é"})
        == b'{"a":"\\u00e9","z":[1,true,null]}'
    )
    assert not module.dynamic_v2_canonical_bytes({"value": 1}).endswith(b"\n")


@pytest.mark.parametrize("value", [{"value": float("nan")}, {"value": {1, 2}}])
def test_public_canonical_bytes_reject_noncanonicalizable_values(
    value: object,
) -> None:
    """Catches leaking raw JSON errors or hashing non-finite dynamic-v2 input."""

    module = _module()

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.dynamic_v2_canonical_bytes(value)


def test_public_canonical_bytes_translate_deep_recursion_to_validation_error() -> None:
    """Catches leaking RecursionError for a deeply nested noncanonical value."""

    module = _module()
    value: list[object] = []
    cursor = value
    for _ in range(10_000):
        child: list[object] = []
        cursor.append(child)
        cursor = child

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.dynamic_v2_canonical_bytes(value)


def test_dynamic_v2_binding_key_authenticates_the_exact_qualification_intent() -> None:
    """Catches key construction from unauthenticated or non-v2 intent fields."""

    module = _module()
    intent, _, _ = _authorities()

    assert module.dynamic_v2_job_binding_s3_key(intent=intent) == (
        "campaigns/glm52-sky-20260726/monitor/must-start/qualification/"
        "783e03494a666c8c3ed3fdb941277b9b589cadb6628edb88720f7a37b6b9cbef/"
        "DYNAMIC_JOB_BINDING.json"
    )


@pytest.mark.parametrize("invalid_kind", ["production", "drift", "unknown", "v1"])
def test_dynamic_v2_binding_key_rejects_nonqualification_or_drifted_intent(
    invalid_kind: str,
) -> None:
    """Catches deriving a dynamic key from foreign, mixed, or unauthenticated intent."""

    module = _module()
    intent, _, _ = _authorities()
    if invalid_kind == "production":
        intent, _, _ = _production_authorities()
    elif invalid_kind == "drift":
        intent["run_id"] = "foreign-run"
    elif invalid_kind == "unknown":
        intent["legacy_job_binding_key"] = "JOB_BINDING.json"
    else:
        intent["schema_version"] = 1
        intent["record_type"] = "glm52_sky_submission_intent_v1"
        intent["submission_body_sha256"] = intent.pop("intent_body_sha256")

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.dynamic_v2_job_binding_s3_key(intent=intent)


def test_key_helper_translates_timestamp_utc_overflow_to_validation_error() -> None:
    """Catches leaking OverflowError from intent timestamp UTC normalization."""

    module = _module()
    intent, _, _ = _authorities()
    intent["intent_at"] = "0001-01-01T00:00:00+01:00"
    intent = _hash_record(
        {key: value for key, value in intent.items() if key != "intent_body_sha256"},
        "intent_body_sha256",
    )

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.dynamic_v2_job_binding_s3_key(intent=intent)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("descriptor_key", None),
        (
            "cache_seed_acceptance_key",
            (
                f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/{_h('2')}/"
                "FOREIGN.json"
            ),
        ),
        (
            "gpu_spend_snapshot_key",
            f"campaigns/{RUN_ID}/spend-snapshots/{_h('9')}/GPU_SPEND_SNAPSHOT.json",
        ),
        (
            "qualification_submission_ready_key",
            (
                f"campaigns/{RUN_ID}/qualification/submission-ready/{_h('9')}/"
                "QUALIFICATION_SUBMISSION_READY.json"
            ),
        ),
    ],
)
def test_key_helper_rejects_coherently_rehashed_malformed_intent_keys(
    field: str,
    malformed: object,
) -> None:
    """Catches authenticating malformed or non-derivable intent key authority."""

    module = _module()
    intent, _, _ = _authorities()
    intent[field] = malformed
    intent = _hash_record(
        {key: value for key, value in intent.items() if key != "intent_body_sha256"},
        "intent_body_sha256",
    )

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.dynamic_v2_job_binding_s3_key(intent=intent)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("descriptor_key", None),
        (
            "cache_seed_acceptance_key",
            (
                f"campaigns/{RUN_ID}/qualification-cache-seed/accepted/{_h('2')}/"
                "FOREIGN.json"
            ),
        ),
        (
            "gpu_spend_snapshot_key",
            f"campaigns/{RUN_ID}/spend-snapshots/{_h('9')}/GPU_SPEND_SNAPSHOT.json",
        ),
        (
            "qualification_submission_ready_key",
            (
                f"campaigns/{RUN_ID}/qualification/submission-ready/{_h('9')}/"
                "QUALIFICATION_SUBMISSION_READY.json"
            ),
        ),
    ],
)
def test_builder_rejects_coherently_rehashed_malformed_intent_keys(
    field: str,
    malformed: object,
) -> None:
    """Catches building from coherently rehashed malformed intent key authority."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    intent[field] = malformed
    intent, baseline, acquisition = _repin_strict_authority_chain(
        intent, baseline, acquisition
    )

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.build_dynamic_v2_job_binding(
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=CONTROLLER_IDENTITY,
            current_exact_name_history=[_new_row()],
            bound_at=INTENT_AT + timedelta(minutes=2),
            now=INTENT_AT + timedelta(minutes=3),
        )


def test_builder_rejects_coherently_rehashed_foreign_readiness_acquisition() -> None:
    """Catches trusting acquisition readiness pins that drift from the intent."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    acquisition.update(
        {
            "qualification_submission_ready_key": (
                f"campaigns/{RUN_ID}/qualification/submission-ready/{_h('8')}/"
                "QUALIFICATION_SUBMISSION_READY.json"
            ),
            "qualification_submission_ready_sha256": _h("9"),
            "qualification_submission_ready_body_sha256": _h("8"),
        }
    )
    acquisition = _hash_record(
        {
            key: value
            for key, value in acquisition.items()
            if key != "acquisition_body_sha256"
        },
        "acquisition_body_sha256",
    )

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.build_dynamic_v2_job_binding(
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=CONTROLLER_IDENTITY,
            current_exact_name_history=[_new_row()],
            bound_at=INTENT_AT + timedelta(minutes=2),
            now=INTENT_AT + timedelta(minutes=3),
        )


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("workspace", "foreign"),
        ("bucket", "192.168.10.1"),
        ("controller_instance_id", "i-1234"),
        ("controller_instance_type", "C6A.XLARGE"),
        (
            "controller_profile_arn",
            "arn:aws:iam::246813579024:instance-profile/foreign-controller",
        ),
        ("controller_cluster_name", "sky..jobs-controller"),
    ],
)
def test_builder_rejects_fully_repinned_malformed_controller_coordinates(
    field: str,
    malformed: object,
) -> None:
    """Catches accepting invalid baseline coordinates after exact ancestry repinning."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    baseline[field] = malformed
    intent, baseline, acquisition = _repin_strict_authority_chain(
        intent, baseline, acquisition
    )
    row = _new_row()
    if field == "workspace":
        row["workspace"] = malformed

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.build_dynamic_v2_job_binding(
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=CONTROLLER_IDENTITY,
            current_exact_name_history=[row],
            bound_at=INTENT_AT + timedelta(minutes=2),
            now=INTENT_AT + timedelta(minutes=3),
        )


def test_builder_returns_the_exact_frozen_dynamic_v2_binding() -> None:
    """Catches omitted ancestry, controller, row, time, or digest binding fields."""

    intent, baseline, acquisition = _authorities()
    expected = _binding(intent, baseline, acquisition, _new_row())

    built = _build_binding(
        intent=intent,
        baseline=baseline,
        acquisition=acquisition,
    )

    assert built == expected
    assert set(built) == {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "managed_mode",
        "descriptor_key",
        "descriptor_file_sha256",
        "descriptor_body_sha256",
        "intent_key",
        "intent_body_sha256",
        "controller_baseline_body_sha256",
        "acquisition_body_sha256",
        "sky_job_name",
        "must_start_by",
        "workspace",
        "controller_instance_id",
        "controller_instance_type",
        "controller_profile_arn",
        "controller_cluster_name",
        "sky_job_id",
        "controller_submitted_at",
        "controller_status",
        "controller_identity",
        "bound_at",
        "job_binding_body_sha256",
    }
    assert built["job_binding_body_sha256"] == (
        "646826a58c87c57cdd61dcbca0ce44e5863e0f8e7f1ae61182d5152f966424ea"
    )
    module_bytes = _module().dynamic_v2_canonical_bytes(
        {key: value for key, value in built.items() if key != "job_binding_body_sha256"}
    )
    assert hashlib.sha256(module_bytes).hexdigest() == built["job_binding_body_sha256"]
    assert not module_bytes.endswith(b"\n")


@pytest.mark.parametrize(
    ("acquisition_present", "history"),
    [
        (False, [_new_row()]),
        (True, []),
        (True, [_new_row(41), _new_row(42)]),
        (
            True,
            [
                {
                    **_new_row(),
                    "controller_submitted_at": _iso(DEADLINE),
                }
            ],
        ),
    ],
)
def test_builder_refuses_every_non_bind_exact_job_resolver_action(
    acquisition_present: bool,
    history: list[dict[str, object]],
) -> None:
    """Catches publishing a binding for waiting, ambiguous, or fail-closed decisions."""

    module = _module()
    intent, baseline, acquisition = _authorities()

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.build_dynamic_v2_job_binding(
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition if acquisition_present else None,
            descriptor_controller_identity=CONTROLLER_IDENTITY,
            current_exact_name_history=history,
            bound_at=INTENT_AT + timedelta(minutes=2),
            now=INTENT_AT + timedelta(minutes=3),
        )


@pytest.mark.parametrize(
    "bound_at",
    [
        INTENT_AT + timedelta(seconds=29),
        INTENT_AT + timedelta(minutes=4),
        datetime(2026, 7, 26, 12, 2),
    ],
)
def test_builder_rejects_preacquisition_future_or_naive_bound_time(
    bound_at: datetime,
) -> None:
    """Catches a builder that emits an unauthenticated binding timestamp."""

    module = _module()

    with pytest.raises(module.DynamicJobBindingValidationError):
        _build_binding(bound_at=bound_at)


def test_builder_translates_explicit_bound_at_utc_overflow_to_validation_error() -> (
    None
):
    """Catches leaking OverflowError while canonicalizing explicit bound_at."""

    module = _module()
    overflowing = datetime(
        1,
        1,
        1,
        tzinfo=timezone(timedelta(hours=1)),
    )

    with pytest.raises(module.DynamicJobBindingValidationError):
        _build_binding(bound_at=overflowing)


def test_validator_accepts_the_exact_built_record_and_returns_a_plain_copy() -> None:
    """Catches validation that cannot immediately authenticate builder output."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _build_binding(
        intent=intent,
        baseline=baseline,
        acquisition=acquisition,
    )

    authenticated = module.validate_dynamic_v2_job_binding(
        binding,
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[{**_new_row(), "controller_status": "RUNNING"}],
        now=INTENT_AT + timedelta(minutes=3),
    )

    assert authenticated == binding
    assert authenticated is not binding


@pytest.mark.parametrize("invalid_kind", ["unknown", "v1", "bad-digest"])
def test_validator_rejects_unknown_v1_or_bad_digest_binding(
    invalid_kind: str,
) -> None:
    """Catches accepting legacy, mixed, unknown-field, or unhashed binding data."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _binding(intent, baseline, acquisition, _new_row())
    if invalid_kind == "unknown":
        binding["unknown"] = True
    elif invalid_kind == "v1":
        binding["schema_version"] = 1
        binding["record_type"] = "glm52_sky_must_start_job_binding_v1"
        binding["target_job_id"] = binding.pop("sky_job_id")
        binding = _rehash_binding(binding)
    else:
        binding["job_binding_body_sha256"] = _h("0")

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.validate_dynamic_v2_job_binding(
            binding,
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=CONTROLLER_IDENTITY,
            current_exact_name_history=[_new_row()],
            now=INTENT_AT + timedelta(minutes=3),
        )


@pytest.mark.parametrize(
    "drift_kind",
    [
        "intent",
        "baseline",
        "acquisition",
        "descriptor-identity",
        "controller-coordinate",
        "numeric-job",
        "submitted-at",
        "row-identity",
        "extra-suffix-row",
        "status-regression",
    ],
)
def test_validator_rejects_foreign_authority_or_controller_row_drift(
    drift_kind: str,
) -> None:
    """Catches reconciling a binding after its authority, target, or row drifts."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _binding(intent, baseline, acquisition, _new_row())
    history = [_new_row()]
    descriptor_identity = CONTROLLER_IDENTITY
    if drift_kind == "intent":
        intent["approval_body_sha256"] = _h("9")
        intent, baseline, acquisition = _repin_strict_authority_chain(
            intent, baseline, acquisition
        )
    elif drift_kind == "baseline":
        baseline["bucket"] = "keep-glm52-foreign"
        intent, baseline, acquisition = _repin_strict_authority_chain(
            intent, baseline, acquisition
        )
    elif drift_kind == "acquisition":
        acquisition["must_start_control_plane_ready_body_sha256"] = _h("8")
        acquisition["must_start_control_plane_ready_key"] = (
            f"campaigns/{RUN_ID}/monitor/must-start/qualification/"
            f"{intent['intent_body_sha256']}/control-plane-ready/{_h('8')}/"
            "CONTROL_PLANE_READY.json"
        )
        acquisition = _hash_record(
            {
                key: value
                for key, value in acquisition.items()
                if key != "acquisition_body_sha256"
            },
            "acquisition_body_sha256",
        )
    elif drift_kind == "descriptor-identity":
        descriptor_identity = (
            "arn:aws:iam::246813579024:role/descriptor-pinned-authority"
        )
    elif drift_kind == "controller-coordinate":
        binding["controller_cluster_name"] = "sky-jobs-controller-foreign"
        binding = _rehash_binding(binding)
    elif drift_kind == "numeric-job":
        binding["sky_job_id"] = 42
        binding = _rehash_binding(binding)
    elif drift_kind == "submitted-at":
        history[0]["controller_submitted_at"] = _iso(INTENT_AT + timedelta(seconds=32))
    elif drift_kind == "row-identity":
        binding["controller_identity"] = (
            "arn:aws:iam::246813579024:role/descriptor-pinned-authority"
        )
        binding = _rehash_binding(binding)
    elif drift_kind == "extra-suffix-row":
        history.append(_new_row(42))
    else:
        binding["controller_status"] = "RUNNING"
        binding = _rehash_binding(binding)

    if drift_kind in {"intent", "baseline", "acquisition"}:
        alternate = _build_binding(
            intent=intent,
            baseline=baseline,
            acquisition=acquisition,
        )
        assert (
            module.validate_dynamic_v2_job_binding(
                alternate,
                intent=intent,
                controller_baseline=baseline,
                acquisition=acquisition,
                descriptor_controller_identity=descriptor_identity,
                current_exact_name_history=history,
                now=INTENT_AT + timedelta(minutes=3),
            )
            == alternate
        )

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.validate_dynamic_v2_job_binding(
            binding,
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=descriptor_identity,
            current_exact_name_history=history,
            now=INTENT_AT + timedelta(minutes=3),
        )


@pytest.mark.parametrize(
    "bound_at",
    [
        "2026-07-26T12:02:00+00:00",
        _iso(INTENT_AT + timedelta(seconds=29)),
        _iso(INTENT_AT + timedelta(minutes=4)),
    ],
)
def test_validator_rejects_noncanonical_preacquisition_or_future_bound_time(
    bound_at: str,
) -> None:
    """Catches authenticating a binding outside the acquired-bound-now order."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _binding(intent, baseline, acquisition, _new_row())
    binding["bound_at"] = bound_at
    binding = _rehash_binding(binding)

    with pytest.raises(module.DynamicJobBindingValidationError):
        module.validate_dynamic_v2_job_binding(
            binding,
            intent=intent,
            controller_baseline=baseline,
            acquisition=acquisition,
            descriptor_controller_identity=CONTROLLER_IDENTITY,
            current_exact_name_history=[_new_row()],
            now=INTENT_AT + timedelta(minutes=3),
        )


def test_resolver_binds_the_one_new_in_window_exact_controller_row() -> None:
    """Catches a resolver that cannot bind the one controller-created numeric job."""

    module = _module()
    intent, baseline, acquisition = _authorities()

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "bind-exact-job"
    assert decision.sky_job_id == 41
    assert decision.job_row == _new_row()


def test_resolver_requires_exact_descriptor_pinned_controller_identity() -> None:
    """Catches deriving controller role authority from an instance-profile ARN."""

    module = _module()
    intent, baseline, acquisition = _authorities()

    exact = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )
    profile_derived_imposter = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=(
            "arn:aws:iam::246813579024:role/descriptor-pinned-authority"
        ),
        current_exact_name_history=[_new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert exact.action == "bind-exact-job"
    assert profile_derived_imposter.action == "fail-closed"


def test_resolver_waits_inertly_without_an_acquisition_selector() -> None:
    """Catches a resolver that invents a numeric target before acquisition."""

    module = _module()
    intent, baseline, _ = _authorities()

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=None,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "wait-for-acquisition"
    assert decision.sky_job_id is None


def test_resolver_fails_closed_for_production_until_acquisition_is_generalized() -> (
    None
):
    """Catches accepting production through a qualification-only acquisition."""

    module = _module()
    intent, baseline, acquisition = _production_authorities()
    row = _new_row()
    row["sky_job_name"] = RUN_ID

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[row],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"
    assert decision.sky_job_id is None


def test_resolver_fails_closed_for_cache_seed_in_dynamic_v2_position() -> None:
    """Catches accepting cache-seed as a generalized acquisition mode."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    cache_seed_name = f"{RUN_ID}-cache-seed"
    intent["managed_mode"] = "cache-seed"
    intent["sky_job_name"] = cache_seed_name
    baseline["managed_mode"] = "cache-seed"
    baseline["sky_job_name"] = cache_seed_name
    acquisition["managed_mode"] = "cache-seed"
    acquisition["sky_job_name"] = cache_seed_name
    intent, baseline, acquisition = _repin_authority_chain(
        intent, baseline, acquisition
    )
    row = {**_new_row(), "sky_job_name": cache_seed_name}

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[row],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"
    assert decision.sky_job_id is None


def test_resolver_waits_when_acquisition_has_no_new_exact_name_row() -> None:
    """Catches a resolver that binds an ID when the controller has made none."""

    module = _module()
    intent, baseline, acquisition = _authorities()

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "wait-for-one-new-job"
    assert decision.sky_job_id is None


@pytest.mark.parametrize(
    "history",
    [
        [_new_row(41), _new_row(42)],
        [{**_new_row(), "sky_job_id": True}],
        [
            {
                **_new_row(),
                "controller_submitted_at": _iso(INTENT_AT - timedelta(seconds=1)),
            }
        ],
        [{**_new_row(), "controller_submitted_at": _iso(DEADLINE)}],
    ],
)
def test_resolver_fails_closed_for_ambiguous_or_out_of_window_candidate(
    history: list[dict[str, object]],
) -> None:
    """Catches binding multiple, boolean, pre-intent, or deadline-time rows."""

    module = _module()
    intent, baseline, acquisition = _authorities()

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=history,
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"
    assert decision.sky_job_id is None


def test_resolver_reconciles_only_the_authenticated_stored_v2_binding() -> None:
    """Catches retargeting a stored numeric binding to a later controller row."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _binding(intent, baseline, acquisition, _new_row())

    reconciled = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=binding,
        now=INTENT_AT + timedelta(minutes=3),
    )
    retargeted = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row(42)],
        stored_binding=binding,
        now=INTENT_AT + timedelta(minutes=3),
    )

    assert reconciled.action == "reconcile-bound-job"
    assert reconciled.sky_job_id == 41
    assert retargeted.action == "fail-closed"


def test_stored_binding_reconciliation_rejects_an_extra_post_baseline_row() -> None:
    """Catches reconciling a bound ID while another new job is also present."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _binding(intent, baseline, acquisition, _new_row(41))

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row(41), _new_row(42)],
        stored_binding=binding,
        now=INTENT_AT + timedelta(minutes=3),
    )

    assert decision.action == "fail-closed"
    assert decision.sky_job_id is None


@pytest.mark.parametrize(
    ("submitted_at", "bound_at", "now", "expected_action"),
    [
        (
            INTENT_AT - timedelta(microseconds=1),
            INTENT_AT + timedelta(minutes=2),
            INTENT_AT + timedelta(minutes=3),
            "fail-closed",
        ),
        (
            INTENT_AT,
            INTENT_AT + timedelta(minutes=2),
            INTENT_AT + timedelta(minutes=3),
            "reconcile-bound-job",
        ),
        (
            DEADLINE - timedelta(seconds=1),
            DEADLINE - timedelta(seconds=1),
            DEADLINE - timedelta(microseconds=1),
            "reconcile-bound-job",
        ),
        (
            DEADLINE,
            DEADLINE,
            DEADLINE,
            "fail-closed",
        ),
    ],
)
def test_stored_binding_reconciliation_applies_candidate_submission_window(
    submitted_at: datetime,
    bound_at: datetime,
    now: datetime,
    expected_action: str,
) -> None:
    """Catches reconciling a bound row outside the intent/deadline window."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    row = {
        **_new_row(),
        "controller_submitted_at": _iso(submitted_at),
    }
    binding = _binding(intent, baseline, acquisition, row)
    binding["bound_at"] = _iso(bound_at)
    binding = _hash_record(
        {
            key: value
            for key, value in binding.items()
            if key != "job_binding_body_sha256"
        },
        "job_binding_body_sha256",
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[row],
        stored_binding=binding,
        now=now,
    )

    assert decision.action == expected_action


@pytest.mark.parametrize("current_status", ["STARTING", "RUNNING"])
def test_stored_binding_reconciliation_allows_normal_controller_status_progression(
    current_status: str,
) -> None:
    """Catches treating mutable controller status as immutable binding identity."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    pending_row = _new_row()
    binding = _binding(intent, baseline, acquisition, pending_row)
    current_row = {**pending_row, "controller_status": current_status}

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[current_row],
        stored_binding=binding,
        now=INTENT_AT + timedelta(minutes=3),
    )

    assert decision.action == "reconcile-bound-job"
    assert decision.sky_job_id == 41
    assert decision.job_row == current_row


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("remaining_gpu_seconds", True),
        ("remaining_gpu_seconds", "82800"),
        ("remaining_gpu_cost_usd", True),
        ("remaining_gpu_cost_usd", "1265.92"),
        ("qualification_allowance_seconds", False),
        ("qualification_allowance_cost_usd", "220.16"),
        ("open_allocation_count", True),
    ],
)
def test_resolver_rejects_boolean_or_nonnumeric_spend_authority(
    field: str,
    invalid: object,
) -> None:
    """Catches accepting coherently rehashed malformed spend authority."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    intent[field] = invalid
    intent, baseline, acquisition = _repin_authority_chain(
        intent, baseline, acquisition
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"


@pytest.mark.parametrize(
    ("field", "active"),
    [
        ("active_exact_name_job_ids", [41]),
        ("active_tagged_p5_instance_ids", ["i-0aaaaaaaaaaaaaaaa"]),
    ],
)
def test_resolver_rejects_nonempty_baseline_active_resource_sets(
    field: str,
    active: list[object],
) -> None:
    """Catches binding while the authenticated baseline already has active work."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    baseline[field] = active
    intent, baseline, acquisition = _repin_authority_chain(
        intent, baseline, acquisition
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"


def test_resolver_rejects_active_baseline_history_hidden_by_empty_summary() -> None:
    """Catches trusting a forged empty active-ID summary over exact history."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    active_prior = _new_row(11)
    baseline["exact_name_history"] = [active_prior]
    baseline["active_exact_name_job_ids"] = []
    intent, baseline, acquisition = _repin_authority_chain(
        intent, baseline, acquisition
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[active_prior, _new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"


@pytest.mark.parametrize(
    "invalid_order",
    ["baseline-before-intent", "row-before-intent", "row-after-baseline"],
)
def test_resolver_rejects_invalid_intent_baseline_history_time_order(
    invalid_order: str,
) -> None:
    """Catches accepting baseline facts outside their authenticated time order."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    current_history = [_new_row()]
    if invalid_order == "baseline-before-intent":
        baseline["observed_at"] = _iso(INTENT_AT - timedelta(microseconds=1))
    else:
        prior = {
            **_new_row(11),
            "controller_status": "FAILED",
            "controller_submitted_at": _iso(
                INTENT_AT - timedelta(microseconds=1)
                if invalid_order == "row-before-intent"
                else INTENT_AT + timedelta(seconds=21)
            ),
        }
        baseline["exact_name_history"] = [prior]
        current_history = [prior, _new_row()]
    intent, baseline, acquisition = _repin_authority_chain(
        intent, baseline, acquisition
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=current_history,
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"


def test_resolver_rejects_baseline_older_than_sixty_seconds_at_acquisition() -> None:
    """Catches accepting an acquisition against a stale controller baseline."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    acquisition["acquired_at"] = _iso(INTENT_AT + timedelta(seconds=81))
    intent, baseline, acquisition = _repin_authority_chain(
        intent, baseline, acquisition
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=2),
    )

    assert decision.action == "fail-closed"


@pytest.mark.parametrize(
    "bound_at",
    [
        INTENT_AT + timedelta(seconds=29),
        INTENT_AT + timedelta(minutes=4),
    ],
)
def test_stored_binding_enforces_acquired_bound_now_time_order(
    bound_at: datetime,
) -> None:
    """Catches reconciling a binding before acquisition or after observation."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    binding = _binding(intent, baseline, acquisition, _new_row())
    binding["bound_at"] = _iso(bound_at)
    binding = _hash_record(
        {
            key: value
            for key, value in binding.items()
            if key != "job_binding_body_sha256"
        },
        "job_binding_body_sha256",
    )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[_new_row()],
        stored_binding=binding,
        now=INTENT_AT + timedelta(minutes=3),
    )

    assert decision.action == "fail-closed"


@pytest.mark.parametrize("future_source", ["controller-row", "controller-baseline"])
def test_resolver_rejects_future_controller_observations(
    future_source: str,
) -> None:
    """Catches authorizing controller facts that had not happened by now."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    row = _new_row()
    if future_source == "controller-row":
        row["controller_submitted_at"] = _iso(INTENT_AT + timedelta(minutes=2))
    else:
        baseline["observed_at"] = _iso(INTENT_AT + timedelta(minutes=2))
        intent, baseline, acquisition = _repin_authority_chain(
            intent, baseline, acquisition
        )

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[row],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"


def test_resolver_fails_closed_when_the_authenticated_baseline_prefix_changes() -> None:
    """Catches accepting a current history that rewrites baseline facts."""

    module = _module()
    intent, baseline, acquisition = _authorities()
    prior = _new_row(11)
    baseline["exact_name_history"] = [prior]
    baseline["baseline_body_sha256"] = _sha(
        {key: value for key, value in baseline.items() if key != "baseline_body_sha256"}
    )
    changed = deepcopy(prior)
    changed["controller_status"] = "RUNNING"

    decision = module.resolve_dynamic_v2_job_binding(
        intent=intent,
        controller_baseline=baseline,
        acquisition=acquisition,
        descriptor_controller_identity=CONTROLLER_IDENTITY,
        current_exact_name_history=[changed, _new_row(41)],
        stored_binding=None,
        now=INTENT_AT + timedelta(minutes=1),
    )

    assert decision.action == "fail-closed"
