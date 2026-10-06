from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "aws/glm52-gpu/scripts/build_glm52_task13_staged_deployment_request.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("task13_v2_invocation_guard", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invocation(tmp_path: Path) -> dict[str, object]:
    output = (tmp_path / "out").resolve()
    output.mkdir()
    names = (
        "retained_bootstrap_runtime_deployment_path",
        "bridge_seed_publication_path",
        "retained_fence_runtime_deployment_path",
        "bridge_seed_path",
        "migration_operations_1_to_6_path",
        "bootstrap_fence_publication_path",
        "prepare_execution_path",
        "support_input_materialization_request_path",
        "disabled_support_deployment_path",
        "operation_7_path",
        "support_runtime_identity_path",
        "no_launch_evidence_path",
    )
    unsigned: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_staged_deployment_request_build_v2",
        "activation_id": "glm52-v2-amber-quartz",
        "output_directory": str(output),
        "staged_infrastructure_evidence_output": str(output / "evidence.json"),
        "journal_path": str(output / "journal.jsonl"),
        "retained_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-gpu/11111111-1111-1111-1111-111111111111"
        ),
        "request_output_path": str(output / "request.json"),
    }
    for name in names:
        unsigned[name] = str((tmp_path / (name + ".json")).resolve())
    return {**unsigned, "canonical_identity_sha256": canonical_sha256(unsigned)}


def test_canonical_invocation_boundary_accepts_only_exact_v2(tmp_path: Path) -> None:
    builder = _load_builder()
    exact = _invocation(tmp_path)
    guarded = builder._guard_invocation(exact)
    assert guarded == exact

    for mutant in (
        {**exact, "schema_version": 1},
        {**exact, "record_type": "glm52_task13_staged_deployment_request_build_v1"},
        {**exact, "change_set_arn": "arn:aws:cloudformation:us-west-2:1:changeSet/x"},
    ):
        with pytest.raises(builder.RequestBuildError):
            builder._guard_invocation(mutant)


def test_post_parse_mutation_is_not_rehashed_or_adopted(tmp_path: Path) -> None:
    builder = _load_builder()
    exact = _invocation(tmp_path)
    exact["activation_id"] = "glm52-v2-mutated"
    with pytest.raises(builder.RequestBuildError, match="identity drifted"):
        builder._guard_invocation(exact)


def test_operation_inputs_are_twelve_distinct_canonical_files(tmp_path: Path) -> None:
    builder = _load_builder()
    exact = _invocation(tmp_path)
    exact["operation_7_path"] = exact["prepare_execution_path"]
    unsigned = dict(exact)
    unsigned.pop("canonical_identity_sha256")
    exact["canonical_identity_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(builder.RequestBuildError, match="distinct"):
        builder._guard_invocation(exact)
