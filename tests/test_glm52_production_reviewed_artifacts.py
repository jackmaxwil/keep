from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.task13_live_inputs import (
    Task13LiveInputError,
    build_production_reviewed_artifacts,
)
from glm52_enforcement.task13_transport_gates import SEMANTIC_GATE_PINS


BUCKET = "keep-glm52-models-246813579024-us-west-2"
ACTIVATION = "approved-20260728"
PREQUALIFICATION_KEYS = {
    "TASK11_REVIEW_APPROVAL": "reviews/task11/approval.json",
    "TASK12_REVIEW_APPROVAL": "reviews/task12/approval.json",
    "RETAINED_FOUNDATION_TEMPLATE": "task13/templates/retained-foundation.yaml",
    "RETAINED_PRE_SUPPORT_TEMPLATE": "task13/templates/retained-pre-support.yaml",
    "RETAINED_TEMPLATE": "task13/templates/retained.yaml",
    "FENCE_TEMPLATE": "task13/migration/fence-transfer.json",
    "SUPPORT_TEMPLATE": "task13/templates/support-disabled.yaml",
    "SUPPORT_INPUTS": "task13/inputs/support-build-inputs.json",
    "BOOTSTRAP_TEMPLATE": "task13/templates/container-bootstrap-v1.json",
    "QUALIFICATION_CACHE_SEED_INPUT": (
        f"task13/activations/{ACTIVATION}/"
        "qualification/cache-seed-input.json"
    ),
    "H100_QUALIFICATION_INPUT": (
        f"task13/activations/{ACTIVATION}/qualification/h100-input.json"
    ),
    "T01_T25_GATE": "task13/gates/t01-t25.json",
    "TRANSPORT_22_MUTANT_GATE": "task13/gates/transport-22-mutants.json",
    "REPOSITORY_ARCHIVE": (
        f"task13/activations/{ACTIVATION}/archive/repo-tar.json"
    ),
    "ACCEPTED_BASELINE": "task13/inputs/accepted-baseline.json",
    "PROMPT_PACK": "task13/inputs/prompt-pack.json",
    "TRAINING_CONFIGURATION": "task13/inputs/training-configuration.json",
    "GPU_SPEND_APPROVAL": "task13/approvals/gpu-spend.json",
    "SUPPORT_APPROVAL": "task13/approvals/support-plane.json",
    "RESIDUAL_LIABILITY_APPROVAL": (
        "task13/approvals/residual-liability.json"
    ),
    "PRODUCTION_DESCRIPTOR": (
        f"task13/activations/{ACTIVATION}/inputs/campaign-descriptor-v2.json"
    ),
}
ADDITION_KEYS = {
    "CLEAN_REHEARSAL": (
        "task13/gates/clean-rehearsal/"
        f"{ACTIVATION}/{'c' * 64}.json"
    ),
    "TASK10_PRODUCTION_AUTHORITY": (
        "task13/production/task10-production-authority.json"
    ),
    "TASK10_WORKER_DESCRIPTOR": (
        "task13/production/task10-worker-descriptor.json"
    ),
    "TASK10_TASK_INPUTS": "task13/production/task10-task-inputs.json",
}
PARAMETER_BY_KIND = {
    "CLEAN_REHEARSAL": "clean_rehearsal_coordinate",
    "TASK10_PRODUCTION_AUTHORITY": "task10_production_authority",
    "TASK10_WORKER_DESCRIPTOR": "task10_worker_descriptor",
    "TASK10_TASK_INPUTS": "task10_task_inputs",
}


def _coordinate(kind: str, key: str) -> dict[str, object]:
    semantic = SEMANTIC_GATE_PINS.get(kind)
    return {
        "artifact_kind": kind,
        "bucket": BUCKET,
        "key": key,
        "version_id": "3Lg" + kind.title().replace("_", "") + "Version",
        "file_sha256": (
            semantic["file_sha256"]
            if semantic is not None
            else hashlib.sha256((kind + "-file").encode()).hexdigest()
        ),
        "body_sha256": (
            semantic["body_sha256"]
            if semantic is not None
            else hashlib.sha256(kind.encode()).hexdigest()
        ),
    }


def _predecessor() -> list[dict[str, object]]:
    return sorted(
        [_coordinate(kind, key) for kind, key in PREQUALIFICATION_KEYS.items()],
        key=lambda row: str(row["artifact_kind"]),
    )


def _additions() -> dict[str, dict[str, object]]:
    return {
        kind: _coordinate(kind, key) for kind, key in ADDITION_KEYS.items()
    }


def _builder_arguments() -> dict[str, object]:
    additions = _additions()
    return {
        "prequalification_reviewed_artifacts": _predecessor(),
        **{
            parameter: additions[kind]
            for kind, parameter in PARAMETER_BY_KIND.items()
        },
    }


def _load_cli() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/build_glm52_production_reviewed_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "glm52_production_reviewed_artifacts_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _cli_arguments(tmp_path: Path, output: Path) -> list[str]:
    values = _builder_arguments()
    flags = (
        (
            "--prequalification-reviewed-artifacts",
            "prequalification_reviewed_artifacts",
        ),
        ("--clean-rehearsal-coordinate", "clean_rehearsal_coordinate"),
        ("--task10-production-authority", "task10_production_authority"),
        ("--task10-worker-descriptor", "task10_worker_descriptor"),
        ("--task10-task-inputs", "task10_task_inputs"),
    )
    arguments: list[str] = []
    for flag, name in flags:
        path = (tmp_path / (name + ".json")).resolve()
        _write_canonical(path, values[name])
        arguments.extend((flag, str(path)))
    arguments.extend(("--output", str(output)))
    return arguments


def test_builder_closes_exact_additive_21_to_25_lineage() -> None:
    arguments = _builder_arguments()
    predecessor = arguments["prequalification_reviewed_artifacts"]
    assert isinstance(predecessor, list)
    predecessor_bytes = {
        row["artifact_kind"]: canonical_json_bytes(row) for row in predecessor
    }

    result = build_production_reviewed_artifacts(**arguments)

    assert len(result) == 25
    assert [row["artifact_kind"] for row in result] == sorted(
        row["artifact_kind"] for row in result
    )
    result_by_kind = {row["artifact_kind"]: row for row in result}
    assert set(result_by_kind) - set(predecessor_bytes) == set(ADDITION_KEYS)
    assert {
        kind: canonical_json_bytes(result_by_kind[kind])
        for kind in predecessor_bytes
    } == predecessor_bytes


@pytest.mark.parametrize("kind", tuple(ADDITION_KEYS))
def test_builder_rejects_each_wrong_added_kind(kind: str) -> None:
    arguments = _builder_arguments()
    other_kind = next(candidate for candidate in ADDITION_KEYS if candidate != kind)
    arguments[PARAMETER_BY_KIND[kind]] = _additions()[other_kind]

    with pytest.raises(Task13LiveInputError, match="kind"):
        build_production_reviewed_artifacts(**arguments)


@pytest.mark.parametrize("kind", tuple(ADDITION_KEYS))
def test_builder_rejects_each_missing_addition(kind: str) -> None:
    arguments = _builder_arguments()
    arguments[PARAMETER_BY_KIND[kind]] = None

    with pytest.raises((Task13LiveInputError, ValueError)):
        build_production_reviewed_artifacts(**arguments)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "extra", "wrong-kind", "coordinate", "order"),
)
def test_builder_rejects_predecessor_drift(mutation: str) -> None:
    arguments = _builder_arguments()
    predecessor = json.loads(
        canonical_json_bytes(arguments["prequalification_reviewed_artifacts"])
    )
    if mutation == "missing":
        predecessor.pop()
    elif mutation == "duplicate":
        predecessor[-1] = predecessor[0]
        predecessor.sort(key=lambda row: row["artifact_kind"])
    elif mutation == "extra":
        predecessor.append(_additions()["CLEAN_REHEARSAL"])
        predecessor.sort(key=lambda row: row["artifact_kind"])
    elif mutation == "wrong-kind":
        predecessor[-1] = _additions()["TASK10_TASK_INPUTS"]
        predecessor.sort(key=lambda row: row["artifact_kind"])
    elif mutation == "coordinate":
        next(
            row
            for row in predecessor
            if row["artifact_kind"] == "ACCEPTED_BASELINE"
        )["key"] = "task13/inputs/prompt-pack.json"
    else:
        predecessor.reverse()
    arguments["prequalification_reviewed_artifacts"] = predecessor

    with pytest.raises((Task13LiveInputError, ValueError)):
        build_production_reviewed_artifacts(**arguments)


def test_cli_writes_canonical_private_create_only_output(tmp_path: Path) -> None:
    module = _load_cli()
    output = (tmp_path / "production-reviewed.json").resolve()
    arguments = _cli_arguments(tmp_path, output)

    assert module.main(arguments) == 0

    raw = output.read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw)) + b"\n"
    assert len(json.loads(raw)) == 25
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_cli_preserves_preexisting_output(tmp_path: Path) -> None:
    module = _load_cli()
    output = (tmp_path / "production-reviewed.json").resolve()
    output.write_bytes(b"preexisting authority\n")
    arguments = _cli_arguments(tmp_path, output)

    assert module.main(arguments) == 64
    assert output.read_bytes() == b"preexisting authority\n"


def test_cli_rejects_noncanonical_input_without_output(tmp_path: Path) -> None:
    module = _load_cli()
    output = (tmp_path / "production-reviewed.json").resolve()
    arguments = _cli_arguments(tmp_path, output)
    predecessor_path = Path(arguments[1])
    value = json.loads(predecessor_path.read_bytes())
    predecessor_path.write_bytes(json.dumps(value).encode() + b"\n")

    assert module.main(arguments) == 64
    assert not output.exists()


def test_writer_completes_short_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_cli()
    output = (tmp_path / "production-reviewed.json").resolve()
    value = build_production_reviewed_artifacts(**_builder_arguments())
    real_write = module.os.write
    calls = 0

    def short_write(descriptor: int, remaining: memoryview) -> int:
        nonlocal calls
        calls += 1
        return real_write(descriptor, remaining[:7])

    monkeypatch.setattr(module.os, "write", short_write)
    module._write_new(output, value)

    assert calls > 1
    assert output.read_bytes() == canonical_json_bytes(value) + b"\n"


@pytest.mark.parametrize("failure", ("zero-write", "fsync", "close"))
def test_writer_removes_its_output_on_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _load_cli()
    output = (tmp_path / "production-reviewed.json").resolve()
    value = build_production_reviewed_artifacts(**_builder_arguments())
    real_close = module.os.close

    if failure == "zero-write":
        monkeypatch.setattr(module.os, "write", lambda _descriptor, _raw: 0)
    elif failure == "fsync":
        def fail_fsync(_descriptor: int) -> None:
            raise OSError("synthetic fsync failure")

        monkeypatch.setattr(module.os, "fsync", fail_fsync)
    else:
        def fail_close(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError("synthetic close failure")

        monkeypatch.setattr(module.os, "close", fail_close)

    with pytest.raises(OSError):
        module._write_new(output, value)

    assert not os.path.lexists(output)
