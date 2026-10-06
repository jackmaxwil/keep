from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

import pytest


RUN_ID = "glm52-sky-20260724"
SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _rehash(
    value: dict[str, object], digest_field: str
) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(digest_field, None)
    return {
        **body,
        digest_field: hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _snapshot() -> dict[str, object]:
    consumed_seconds = 10_800
    hourly = Decimal("55.04")
    consumed_cost = (
        Decimal(consumed_seconds) * hourly / Decimal(3600)
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": SHA,
        "descriptor_sha256": "b" * 64,
        "descriptor_body_sha256": "c" * 64,
        "approval_sha256": "d" * 64,
        "approval_body_sha256": "e" * 64,
        "gpu_spend_ledger_latest_sha256": "f" * 64,
        "gpu_spend_ledger_latest_body_sha256": "1" * 64,
        "gpu_spend_ledger_genesis_sha256": "2" * 64,
        "gpu_spend_ledger_record_count": 4,
        "gpu_spend_ledger_tip_record_sha256": "3" * 64,
        "gpu_spend_ledger_file_sha256": "4" * 64,
        "ec2_allocation_history_sha256": "5" * 64,
        "ec2_allocation_instance_ids": [
            "i-00000000000000001",
            "i-00000000000000002",
        ],
        "observed_at": "2026-07-28T12:00:00Z",
        "approved_gpu_runtime_seconds": 86_400,
        "approved_gpu_cost_usd": 1_320.96,
        "hourly_cost_usd": 55.04,
        "consumed_gpu_seconds": consumed_seconds,
        "remaining_gpu_seconds": 86_400 - consumed_seconds,
        "consumed_gpu_cost_usd": float(consumed_cost),
        "remaining_gpu_cost_usd": float(
            Decimal("1320.96") - consumed_cost
        ),
        "qualification_allowance_seconds": 14_400,
        "qualification_allowance_cost_usd": 220.16,
        "open_allocation_count": 0,
    }
    return _rehash(body, "snapshot_body_sha256")


def _load_test_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "tests" / filename
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _accepted_chain(tmp_path_factory: pytest.TempPathFactory):
    production_tests = _load_test_module(
        "_task5_production_fixtures",
        "test_glm52_sky_production_submission.py",
    )
    production_sources = production_tests.source_chain.__wrapped__(
        tmp_path_factory
    )
    snapshot = production_sources["records"]["closed_gpu_spend_snapshot"]

    prelaunch_tests = _load_test_module(
        "_task5_prelaunch_fixtures",
        "test_glm52_sky_production_acquisition.py",
    )
    authorities = prelaunch_tests.source_authorities.__wrapped__(
        tmp_path_factory
    )
    accepted = prelaunch_tests._module()
    chain = prelaunch_tests._full_chain(accepted, authorities)
    return production_tests, prelaunch_tests, accepted, authorities, chain, snapshot


def test_each_ported_source_validator_matches_accepted_bytes_and_rejects_mutants(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from glm52_enforcement.source_authorities import (
        VersionedJsonArtifact,
        validate_gpu_spend_snapshot,
        validate_production_controller_baseline,
        validate_production_must_start_control_plane_ready,
        validate_production_submission_acquired,
        validate_production_submission_intent,
    )
    from mlx_vq.quality.glm52_gpu_spend_snapshot import (
        validate_gpu_spend_snapshot as accepted_validate,
    )

    (
        production_tests,
        prelaunch_tests,
        accepted,
        authorities,
        chain,
        snapshot,
    ) = _accepted_chain(tmp_path_factory)

    def local_artifact(value):
        return VersionedJsonArtifact(
            key=value.key,
            raw=value.raw,
            version_id=value.version_id,
        )

    records = (
        (
            snapshot,
            accepted_validate,
            validate_gpu_spend_snapshot,
            {},
        ),
        (
            authorities["intent_record"],
            production_tests.validate_production_submission_intent,
            validate_production_submission_intent,
            {},
        ),
        (
            chain["baseline"],
            accepted.validate_production_controller_baseline,
            validate_production_controller_baseline,
            {
                "descriptor": local_artifact(authorities["descriptor"]),
                "intent": local_artifact(authorities["intent"]),
            },
        ),
        (
            chain["ready"],
            accepted.validate_production_must_start_control_plane_ready,
            validate_production_must_start_control_plane_ready,
            {
                "descriptor": local_artifact(authorities["descriptor"]),
                "intent": local_artifact(authorities["intent"]),
                "controller_baseline": local_artifact(
                    chain["baseline_artifact"]
                ),
            },
        ),
        (
            chain["acquisition"],
            accepted.validate_production_submission_acquired,
            validate_production_submission_acquired,
            {
                "descriptor": local_artifact(authorities["descriptor"]),
                "intent": local_artifact(authorities["intent"]),
                "controller_baseline": local_artifact(
                    chain["baseline_artifact"]
                ),
                "must_start_control_plane_ready": local_artifact(
                    chain["ready_artifact"]
                ),
                "now": chain["acquired_at"],
            },
        ),
    )
    for value, accepted_validator, ported_validator, local_kwargs in records:
        accepted_kwargs = {}
        for name, artifact in local_kwargs.items():
            if name == "now":
                accepted_kwargs[name] = artifact
            else:
                accepted_kwargs[name] = getattr(
                    authorities["descriptor"].__class__,
                    "__call__",
                    None,
                )
                accepted_kwargs[name] = (
                    authorities["descriptor"].__class__(
                        key=artifact.key,
                        raw=artifact.raw,
                        version_id=artifact.version_id,
                    )
                )
        assert accepted_validator(value, **accepted_kwargs) == value
        assert ported_validator(value, **local_kwargs) == value
        assert _canonical(ported_validator(value, **local_kwargs)) == _canonical(
            accepted_validator(value, **accepted_kwargs)
        )

        unknown = {**value, "unknown": None}
        with pytest.raises(ValueError):
            accepted_validator(unknown, **accepted_kwargs)
        with pytest.raises(ValueError):
            ported_validator(unknown, **local_kwargs)

        digest_field = next(
            field
            for field in (
                "snapshot_body_sha256",
                "intent_body_sha256",
                "baseline_body_sha256",
                "control_plane_ready_body_sha256",
                "acquisition_body_sha256",
            )
            if field in value
        )
        mutant_field = {
            "snapshot_body_sha256": ("open_allocation_count", False),
            "intent_body_sha256": ("open_allocation_count", False),
            "baseline_body_sha256": ("active_exact_name_job_ids", False),
            "control_plane_ready_body_sha256": (
                "activation_capabilities",
                False,
            ),
            "acquisition_body_sha256": ("acquired_at", False),
        }[digest_field]
        wrong_type = _rehash(
            {**value, mutant_field[0]: mutant_field[1]}, digest_field
        )
        with pytest.raises(ValueError):
            accepted_validator(wrong_type, **accepted_kwargs)
        with pytest.raises(ValueError):
            ported_validator(wrong_type, **local_kwargs)


def test_each_source_key_is_derived_from_its_closed_record_and_never_caller_input(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from glm52_enforcement import source_authorities

    (
        _production_tests,
        _prelaunch_tests,
        _accepted,
        authorities,
        chain,
        snapshot,
    ) = _accepted_chain(tmp_path_factory)
    cases = (
        (
            source_authorities.gpu_spend_snapshot_s3_key,
            {
                "run_id": snapshot["run_id"],
                "snapshot_body_sha256": snapshot["snapshot_body_sha256"],
            },
            (
                f"campaigns/{snapshot['run_id']}/spend-snapshots/"
                f"{snapshot['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
            ),
        ),
        (
            source_authorities.production_submission_intent_s3_key,
            {
                "run_id": authorities["intent_record"]["run_id"],
                "intent_body_sha256": authorities["intent_record"][
                    "intent_body_sha256"
                ],
            },
            authorities["intent"].key,
        ),
        (
            source_authorities.production_controller_baseline_s3_key,
            {
                "run_id": chain["baseline"]["run_id"],
                "baseline_body_sha256": chain["baseline"][
                    "baseline_body_sha256"
                ],
            },
            chain["baseline_artifact"].key,
        ),
        (
            source_authorities.production_must_start_control_plane_ready_s3_key,
            {
                "run_id": chain["ready"]["run_id"],
                "intent_body_sha256": chain["ready"]["intent_body_sha256"],
                "control_plane_ready_body_sha256": chain["ready"][
                    "control_plane_ready_body_sha256"
                ],
            },
            chain["ready_artifact"].key,
        ),
        (
            source_authorities.production_submission_acquired_s3_key,
            {
                "run_id": chain["acquisition"]["run_id"],
                "descriptor_file_sha256": chain["acquisition"][
                    "descriptor_file_sha256"
                ],
            },
            (
                f"campaigns/{chain['acquisition']['run_id']}/submissions/"
                "production/acquisitions/"
                f"{chain['acquisition']['descriptor_file_sha256']}/"
                "SUBMISSION_ACQUIRED.json"
            ),
        ),
    )
    for key_function, kwargs, expected in cases:
        assert key_function(**kwargs) == expected
        with pytest.raises(TypeError):
            key_function(**kwargs, key="caller/alternate.json")
