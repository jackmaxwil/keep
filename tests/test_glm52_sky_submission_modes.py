"""Closed managed-submission mode registry tests."""

from __future__ import annotations

import json
import inspect
import shutil
import subprocess
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

import mlx_vq.quality.glm52_sky_submission_modes as modes

EXPECTED_ROWS = {
    ("qualification", "submission-intent"): (
        2,
        "glm52_sky_submission_intent_v2",
        "intent_body_sha256",
        "content-addressed-body",
    ),
    ("qualification", "submission-accepted"): (
        2,
        "glm52_sky_submission_accepted_v2",
        "accepted_body_sha256",
        "content-addressed-body",
    ),
    ("qualification", "submission-acquisition"): (
        1,
        "glm52_sky_submission_acquired_v1",
        "acquisition_body_sha256",
        "descriptor-file-singleton",
    ),
    ("qualification", "controller-baseline"): (
        1,
        "glm52_controller_baseline_v1",
        "baseline_body_sha256",
        "content-addressed-body",
    ),
    ("qualification", "control-plane-ready"): (
        1,
        "glm52_must_start_control_plane_ready_v1",
        "control_plane_ready_body_sha256",
        "intent-control-body",
    ),
    ("qualification", "dynamic-job-binding"): (
        2,
        "glm52_sky_must_start_job_binding_v2",
        "job_binding_body_sha256",
        "intent-singleton",
    ),
    ("qualification", "worker-start-latch"): (
        2,
        "glm52_sky_worker_start_latch_v2",
        "worker_latch_body_sha256",
        "intent-instance-latch",
    ),
    ("qualification", "worker-start-accepted"): (
        2,
        "glm52_sky_worker_start_accepted_v2",
        "worker_acceptance_body_sha256",
        "intent-instance-latch",
    ),
    ("cache-seed", "submission-intent"): (
        1,
        "glm52_sky_cache_seed_submission_intent_v1",
        "intent_body_sha256",
        "content-addressed-body",
    ),
    ("cache-seed", "launch-claim"): (
        1,
        "glm52_sky_cache_seed_launch_claim_v1",
        "launch_claim_body_sha256",
        "descriptor-file-singleton",
    ),
    ("production", "submission-intent"): (
        1,
        "glm52_sky_production_submission_intent_v1",
        "intent_body_sha256",
        "content-addressed-body",
    ),
    ("production", "controller-baseline"): (
        1,
        "glm52_production_controller_baseline_v1",
        "baseline_body_sha256",
        "content-addressed-body",
    ),
    ("production", "control-plane-ready"): (
        1,
        "glm52_production_must_start_control_plane_ready_v1",
        "control_plane_ready_body_sha256",
        "intent-control-body",
    ),
    ("production", "submission-acquisition"): (
        1,
        "glm52_sky_production_submission_acquired_v1",
        "acquisition_body_sha256",
        "descriptor-file-singleton",
    ),
    ("production", "generation-claim"): (
        1,
        "glm52_sky_production_generation_claim_v1",
        "generation_claim_body_sha256",
        "generation-claim-singleton",
    ),
    ("production", "generation-start-decision"): (
        1,
        "glm52_sky_production_generation_start_decision_v1",
        "start_decision_body_sha256",
        "generation-start-decision-singleton",
    ),
    ("production", "generation-terminal"): (
        1,
        "glm52_sky_production_generation_terminal_v1",
        "generation_terminal_body_sha256",
        "generation-terminal-singleton",
    ),
}


class StringSubclass(str):
    """A coercible string that must not cross exact-string boundaries."""


class Coercible:
    def __str__(self) -> str:
        return "qualification"


def _row(contract: modes.RecordContract) -> tuple[int, str, str, str]:
    return (
        contract.schema_version,
        contract.record_type,
        contract.digest_field,
        contract.address_kind,
    )


def test_mutant_production_public_interface_or_registry_literal_rows_drift() -> None:
    assert modes.__all__ == [
        "AddressKind",
        "ManagedMode",
        "RecordContract",
        "RecordKind",
        "SubmissionModeContractError",
        "expected_sky_job_name",
        "record_contract",
        "require_opaque_version_id",
    ]
    assert {
        pair: _row(modes.record_contract(managed_mode=pair[0], record_kind=pair[1]))
        for pair in EXPECTED_ROWS
    } == EXPECTED_ROWS
    assert len(EXPECTED_ROWS) == 17
    assert len(modes._ROWS) == 17
    assert len({(row[0], row[1]) for row in modes._ROWS}) == 17
    assert get_args(modes.RecordKind) == (
        "submission-intent",
        "submission-accepted",
        "submission-acquisition",
        "controller-baseline",
        "control-plane-ready",
        "dynamic-job-binding",
        "worker-start-latch",
        "worker-start-accepted",
        "launch-claim",
        "generation-claim",
        "generation-start-decision",
        "generation-terminal",
    )
    assert get_args(modes.AddressKind) == (
        "content-addressed-body",
        "descriptor-file-singleton",
        "intent-singleton",
        "intent-control-body",
        "intent-instance-latch",
        "generation-claim-singleton",
        "generation-start-decision-singleton",
        "generation-terminal-singleton",
    )
    assert modes._ROWS == tuple(
        (
            mode,
            kind,
            schema,
            record_type,
            digest_field,
            address_kind,
        )
        for (mode, kind), (
            schema,
            record_type,
            digest_field,
            address_kind,
        ) in EXPECTED_ROWS.items()
    )
    assert modes._ROWS[-3:] == (
        (
            "production",
            "generation-claim",
            1,
            "glm52_sky_production_generation_claim_v1",
            "generation_claim_body_sha256",
            "generation-claim-singleton",
        ),
        (
            "production",
            "generation-start-decision",
            1,
            "glm52_sky_production_generation_start_decision_v1",
            "start_decision_body_sha256",
            "generation-start-decision-singleton",
        ),
        (
            "production",
            "generation-terminal",
            1,
            "glm52_sky_production_generation_terminal_v1",
            "generation_terminal_body_sha256",
            "generation-terminal-singleton",
        ),
    )


def test_mutant_production_contract_aliases_or_noninjective_triples() -> None:
    contracts = [
        modes.record_contract(managed_mode=mode, record_kind=kind)
        for mode, kind in EXPECTED_ROWS
    ]
    assert len(
        {
            (
                contract.schema_version,
                contract.record_type,
                contract.digest_field,
            )
            for contract in contracts
        }
    ) == len(contracts)
    first = contracts[0]
    second = modes.record_contract(
        managed_mode="qualification",
        record_kind="submission-intent",
    )
    assert first == second
    assert first is not second
    with pytest.raises((AttributeError, TypeError)):
        first.record_type = "changed"  # type: ignore[misc]


def test_mutant_production_extension_rewrites_frozen_legacy_discriminants() -> None:
    from mlx_vq.quality import glm52_sky_cache_seed_launch as cache_seed
    from mlx_vq.quality import glm52_sky_submission_acquisition as acquisition
    from mlx_vq.quality import glm52_sky_submission_lifecycle as lifecycle
    from mlx_vq.quality import glm52_sky_submission_live_authority as live

    expected_from_frozen = {
        ("qualification", "submission-intent"): (
            lifecycle.INTENT_SCHEMA_VERSION,
            lifecycle.INTENT_RECORD_TYPE,
            lifecycle.INTENT_DIGEST_FIELD,
        ),
        ("qualification", "submission-accepted"): (
            lifecycle.ACCEPTED_SCHEMA_VERSION,
            lifecycle.ACCEPTED_RECORD_TYPE,
            lifecycle.ACCEPTED_DIGEST_FIELD,
        ),
        ("qualification", "submission-acquisition"): (
            acquisition.ACQUISITION_SCHEMA_VERSION,
            acquisition.ACQUISITION_RECORD_TYPE,
            acquisition.ACQUISITION_DIGEST_FIELD,
        ),
        ("qualification", "controller-baseline"): (
            live.CONTROLLER_BASELINE_SCHEMA_VERSION,
            live.CONTROLLER_BASELINE_RECORD_TYPE,
            live.CONTROLLER_BASELINE_DIGEST_FIELD,
        ),
        ("qualification", "control-plane-ready"): (
            live.MUST_START_CONTROL_PLANE_READY_SCHEMA_VERSION,
            live.MUST_START_CONTROL_PLANE_READY_RECORD_TYPE,
            live.MUST_START_CONTROL_PLANE_READY_DIGEST_FIELD,
        ),
        ("cache-seed", "submission-intent"): (
            cache_seed.INTENT_SCHEMA_VERSION,
            cache_seed.INTENT_RECORD_TYPE,
            cache_seed.INTENT_DIGEST_FIELD,
        ),
        ("cache-seed", "launch-claim"): (
            cache_seed.CLAIM_SCHEMA_VERSION,
            cache_seed.CLAIM_RECORD_TYPE,
            cache_seed.CLAIM_DIGEST_FIELD,
        ),
    }
    for pair, expected in expected_from_frozen.items():
        contract = modes.record_contract(
            managed_mode=pair[0],
            record_kind=pair[1],
        )
        assert (
            contract.schema_version,
            contract.record_type,
            contract.digest_field,
        ) == expected


@pytest.mark.parametrize(
    ("managed_mode", "record_kind"),
    [
        ("cache-seed", "submission-accepted"),
        ("cache-seed", "submission-acquisition"),
        ("cache-seed", "controller-baseline"),
        ("cache-seed", "control-plane-ready"),
        ("cache-seed", "dynamic-job-binding"),
        ("cache-seed", "worker-start-latch"),
        ("cache-seed", "worker-start-accepted"),
        ("production", "submission-accepted"),
        ("production", "dynamic-job-binding"),
        ("production", "worker-start-latch"),
        ("production", "worker-start-accepted"),
        ("qualification", "generation-claim"),
        ("qualification", "generation-start-decision"),
        ("qualification", "generation-terminal"),
        ("cache-seed", "generation-claim"),
        ("cache-seed", "generation-start-decision"),
        ("cache-seed", "generation-terminal"),
    ],
)
def test_mutant_production_registry_accepts_cross_mode_or_fallback_records(
    managed_mode: str,
    record_kind: str,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.record_contract(
            managed_mode=managed_mode,  # type: ignore[arg-type]
            record_kind=record_kind,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("managed_mode", "record_kind"),
    [
        ("production", "launch-claim"),
        ("cache-seed", "submission-acquisition"),
        ("qualification", "launch-claim"),
        ("unknown", "submission-intent"),
        ("", "submission-intent"),
        ("production", "unknown"),
    ],
)
def test_mutant_production_registry_accepts_an_unknown_pair(
    managed_mode: str,
    record_kind: str,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.record_contract(
            managed_mode=managed_mode,  # type: ignore[arg-type]
            record_kind=record_kind,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [StringSubclass("production"), Coercible(), 1, None])
def test_mutant_production_registry_coerces_or_accepts_string_subclasses(
    value: object,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.record_contract(
            managed_mode=value,  # type: ignore[arg-type]
            record_kind="submission-intent",
        )
    with pytest.raises(modes.SubmissionModeContractError):
        modes.record_contract(
            managed_mode="production",
            record_kind=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("managed_mode", "expected"),
    [
        ("production", "run.A_1"),
        ("qualification", "run.A_1-qualification"),
        ("cache-seed", "run.A_1-cache-seed"),
    ],
)
def test_mutant_production_job_name_grammar_drifts(
    managed_mode: str,
    expected: str,
) -> None:
    assert (
        modes.expected_sky_job_name(
            run_id="run.A_1",
            managed_mode=managed_mode,  # type: ignore[arg-type]
        )
        == expected
    )


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "-leading",
        ".leading",
        "_leading",
        "has space",
        "slash/value",
        "star*",
        "question?",
        "bracket[",
        "bracket]",
        "x" * 129,
        StringSubclass("run"),
        Coercible(),
        1,
    ],
)
def test_mutant_production_job_name_accepts_unsafe_or_nonexact_run_id(
    run_id: object,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.expected_sky_job_name(
            run_id=run_id,  # type: ignore[arg-type]
            managed_mode="production",
        )


@pytest.mark.parametrize(
    "managed_mode", ["Production", "", StringSubclass("production"), 1]
)
def test_mutant_production_job_name_accepts_unknown_or_nonexact_mode(
    managed_mode: object,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.expected_sky_job_name(
            run_id="run",
            managed_mode=managed_mode,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", ["v1", "!", "~", "opaque/version+token=="])
def test_mutant_production_version_id_rejects_valid_visible_ascii(
    value: str,
) -> None:
    assert modes.require_opaque_version_id(value, field="version_id") == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "null",
        " ",
        "\t",
        "\n",
        "has space",
        "\x1f",
        "\x7f",
        "caf\u00e9",
        StringSubclass("v1"),
        Coercible(),
        1,
        None,
    ],
)
def test_mutant_production_version_id_accepts_invalid_or_nonexact_values(
    value: object,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.require_opaque_version_id(value, field="version_id")


@pytest.mark.parametrize("field", [StringSubclass("version_id"), Coercible(), 1])
def test_mutant_production_version_id_field_name_accepts_nonexact_string(
    field: object,
) -> None:
    with pytest.raises(modes.SubmissionModeContractError):
        modes.require_opaque_version_id("v1", field=field)  # type: ignore[arg-type]


def test_mutant_production_registry_exposes_generic_or_launch_capability() -> None:
    forbidden = {
        "build_record",
        "canonical_file_bytes",
        "format_s3_key",
        "launch",
        "launch_ready",
        "readiness",
        "registry",
        "serialize",
        "validate_record",
    }
    assert forbidden.isdisjoint(modes.__all__)
    assert all(not hasattr(modes, name) for name in forbidden)


def test_mutant_production_registry_breaks_python39_or_imports_nonstdlib(
    tmp_path: Path,
) -> None:
    isolated = tmp_path / "glm52_sky_submission_modes.py"
    shutil.copyfile(Path(modes.__file__), isolated)
    code = """
import importlib.util
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("isolated_modes", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
contract = module.record_contract(
    managed_mode="production",
    record_kind="submission-intent",
)
assert contract.record_type == "glm52_sky_production_submission_intent_v1"
assert module.expected_sky_job_name(
    run_id="glm52-run",
    managed_mode="qualification",
) == "glm52-run-qualification"
assert module.require_opaque_version_id(
    "opaque-token",
    field="version_id",
) == "opaque-token"
try:
    module.record_contract(
        managed_mode="production",
        record_kind="launch-claim",
    )
except module.SubmissionModeContractError:
    pass
else:
    raise AssertionError("unsupported pair accepted")
print(json.dumps({"python": sys.version_info[:2], "ok": True}))
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-c", code, str(isolated)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert tuple(result["python"]) == (3, 9)


def test_mutant_submission_modes_public_typing_or_error_contract_drifts() -> None:
    assert get_args(modes.ManagedMode) == (
        "cache-seed",
        "qualification",
        "production",
    )
    assert get_type_hints(modes.RecordContract) == {
        "schema_version": int,
        "record_type": str,
        "digest_field": str,
        "address_kind": modes.AddressKind,
    }
    assert modes.SubmissionModeContractError.__bases__ == (ValueError,)
    assert inspect.signature(modes.record_contract) == inspect.Signature(
        parameters=[
            inspect.Parameter(
                "managed_mode",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="ManagedMode",
            ),
            inspect.Parameter(
                "record_kind",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="RecordKind",
            ),
        ],
        return_annotation="RecordContract",
    )
    assert inspect.signature(modes.expected_sky_job_name) == inspect.Signature(
        parameters=[
            inspect.Parameter(
                "run_id",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="str",
            ),
            inspect.Parameter(
                "managed_mode",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="ManagedMode",
            ),
        ],
        return_annotation="str",
    )
    assert inspect.signature(modes.require_opaque_version_id) == inspect.Signature(
        parameters=[
            inspect.Parameter(
                "value",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="object",
            ),
            inspect.Parameter(
                "field",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="str",
            ),
        ],
        return_annotation="str",
    )
    assert get_type_hints(modes.record_contract) == {
        "managed_mode": modes.ManagedMode,
        "record_kind": modes.RecordKind,
        "return": modes.RecordContract,
    }
    assert get_type_hints(modes.expected_sky_job_name) == {
        "run_id": str,
        "managed_mode": modes.ManagedMode,
        "return": str,
    }
    assert get_type_hints(modes.require_opaque_version_id) == {
        "value": object,
        "field": str,
        "return": str,
    }
