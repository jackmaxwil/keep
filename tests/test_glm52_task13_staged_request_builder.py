from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task13_production_operations import _guard_production_request
from glm52_enforcement.task13_staged_deployment import StagedDeploymentRequest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/build_glm52_task13_staged_deployment_request.py"
SHA_A = "a" * 64
RETAINED_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/keep-glm52-gpu/"
    "11111111-1111-1111-1111-111111111111"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("task13_v2_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _support_request() -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_support_input_materialization_request_v2",
        "activation_id": "glm52-v2-amber-quartz",
        "bootstrap_manifest_coordinate": {
            "bucket": "keep-glm52-models-246813579024-us-west-2",
            "key": (
                "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
                "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
            ),
            "version_id": "manifest-v1",
            "file_sha256": SHA_A,
            "canonical_identity_sha256": SHA_A,
        },
        "prepare_entry_identity_sha256": SHA_A,
        "host_user_data": "#!/bin/sh\ntrue",
        "host_boot_identity_sha256": SHA_A,
        "cryptography_layer_arn": (
            "arn:aws:lambda:us-west-2:246813579024:layer/h1g-crypto:7"
        ),
        "cryptography_layer_sha256": SHA_A,
        "lambda_code_bucket": "keep-glm52-code-246813579024-us-west-2",
        "lambda_code_key": "task13/code/support.zip",
        "lambda_code_version_id": "code-v1",
        "lambda_code_sha256": SHA_A,
        "price_card_identity_sha256": SHA_A,
        "activation_started_at": "2026-07-31T00:00:00Z",
        "runtime_credential_cutoff_at": "2026-07-31T01:00:00Z",
    }


def _invocation(tmp_path: Path) -> dict[str, object]:
    from test_glm52_task13_production_operations import (
        _request as _production_request,
    )

    output = tmp_path / "output"
    output.mkdir()
    production = _production_request(output)
    fields = {
        "retained_bootstrap_runtime_deployment_path": production[
            "retained_bootstrap_runtime_deployment"
        ],
        "bridge_seed_publication_path": production["bridge_seed_publication"],
        "retained_fence_runtime_deployment_path": production[
            "retained_fence_runtime_deployment"
        ],
        "bridge_seed_path": {"record_type": "seed"},
        "migration_operations_1_to_6_path": {"record_type": "checkpoint-request"},
        "bootstrap_fence_publication_path": production[
            "bootstrap_fence_publication"
        ],
        "prepare_execution_path": {"record_type": "prepare-request"},
        "support_input_materialization_request_path": _support_request(),
        "disabled_support_deployment_path": {"record_type": "disabled-request"},
        "operation_7_path": {"record_type": "operation-7-request"},
        "support_runtime_identity_path": {"record_type": "runtime-request"},
        "no_launch_evidence_path": {"record_type": "no-launch-request"},
    }
    paths: dict[str, str] = {}
    for name, body in fields.items():
        path = tmp_path / (name + ".json")
        _write(path, body)
        paths[name] = str(path.resolve())
    unsigned: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_staged_deployment_request_build_v2",
        "activation_id": "glm52-v2-amber-quartz",
        "output_directory": str(output.resolve()),
        "staged_infrastructure_evidence_output": str(
            (output / "staged-infrastructure-evidence-v2.json").resolve()
        ),
        "journal_path": str((output / "staged-v2.jsonl").resolve()),
        "retained_stack_id": RETAINED_STACK_ID,
        **paths,
        "request_output_path": str((output / "request-v2.json").resolve()),
    }
    return {**unsigned, "canonical_identity_sha256": canonical_sha256(unsigned)}


def test_builder_emits_one_canonical_v2_boundary(tmp_path: Path) -> None:
    builder = _load_builder()
    invocation = _invocation(tmp_path)
    request, request_path, evidence_path, journal_path = builder.build_request(
        invocation
    )
    guarded = _guard_production_request(request["production_request"])
    staged = StagedDeploymentRequest(
        schema_version=request["schema_version"],
        record_type=request["record_type"],
        activation_id=request["activation_id"],
        journal_path=journal_path,
        production_request=guarded,
    )
    assert staged.schema_version == 2
    assert request_path.name == "request-v2.json"
    assert evidence_path.name == "staged-infrastructure-evidence-v2.json"
    assert "fence_template_inventory" not in canonical_json_bytes(request).decode()
    assert "task11_writer_bindings" not in canonical_json_bytes(request).decode()


def test_builder_rejects_v1_or_caller_copied_support_fields(tmp_path: Path) -> None:
    builder = _load_builder()
    invocation = _invocation(tmp_path)
    support_path = Path(invocation["support_input_materialization_request_path"])
    legacy = _support_request()
    legacy["schema_version"] = 1
    legacy["record_type"] = "glm52_task13_support_input_materialization_request_v1"
    _write(support_path, legacy)
    with pytest.raises(builder.RequestBuildError, match="v2"):
        builder.build_request(invocation)

    _write(support_path, {**_support_request(), "fence_template_inventory": []})
    with pytest.raises(builder.RequestBuildError, match="v2"):
        builder.build_request(invocation)


def test_builder_refuses_noncanonical_or_existing_output(tmp_path: Path) -> None:
    builder = _load_builder()
    invocation = _invocation(tmp_path)
    request_path = Path(invocation["request_output_path"])
    request_path.write_text("sentinel\n")
    with pytest.raises(builder.RequestBuildError, match="new exact"):
        builder.build_request(invocation)

    request_path.unlink()
    invocation["activation_id"] = "mutated"
    with pytest.raises(builder.RequestBuildError, match="identity drifted"):
        builder.build_request(invocation)
