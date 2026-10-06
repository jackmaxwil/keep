from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/build_glm52_h1g_bootstrap_requests.py"
SHA_A = "a" * 64


def _load_builder():
    spec = importlib.util.spec_from_file_location("h1g_bootstrap_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> str:
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return str(path.resolve())


def _coordinate(key: str, index: int) -> dict[str, object]:
    return {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": key,
        "version_id": f"version-{index}",
        "file_sha256": f"{index + 1:064x}",
        "canonical_identity_sha256": f"{index + 11:064x}",
    }


def _invocation(tmp_path: Path) -> tuple[dict[str, object], object]:
    from test_glm52_fence_bootstrap_publication import (
        _checkpoint,
        _policy_input_projection,
        _renderer_inputs,
        _request,
    )
    from test_glm52_task13_production_operations import _request as production_request

    renderer_inputs = _renderer_inputs()
    checkpoint = _checkpoint(renderer_inputs)
    bootstrap_request = _request(renderer_inputs, checkpoint)
    bootstrap_authority = dict(bootstrap_request)
    bootstrap_authority["record_type"] = (
        "glm52_fence_bootstrap_publication_authority_v2"
    )
    bootstrap_authority.pop("materializer_function_version_arn")
    bootstrap_authority.pop("checkpoint_identity_sha256")
    seed_authority = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bridge_seed_publication_authority_v2",
        "activation_id": "glm52-v2-amber-quartz",
        "generation": 1,
        "expected_live_preseed_policy_sha256": "c" * 64,
        "renderer_input": _policy_input_projection(renderer_inputs[0]),
    }
    production = production_request(tmp_path)
    runtime_authority = dict(production["retained_fence_runtime_deployment"])
    sources = tmp_path / "sources"
    outputs = tmp_path / "outputs"
    sources.mkdir(parents=True)
    outputs.mkdir()
    source_paths = {
        "retained_bootstrap_inputs_path": _write(
            sources / "bootstrap-inputs.json",
            production["retained_bootstrap_runtime_deployment"],
        ),
        "bridge_seed_authority_path": _write(
            sources / "seed-authority.json", seed_authority
        ),
        "bootstrap_publication_authority_path": _write(
            sources / "bootstrap-authority.json", bootstrap_authority
        ),
        "retained_runtime_authority_path": _write(
            sources / "runtime-authority.json", runtime_authority
        ),
    }
    output_paths = {
        "retained_bootstrap_inputs_output": str(
            (outputs / "retained-bootstrap-inputs.json").resolve()
        ),
        "bridge_seed_request_output": str(
            (outputs / "bridge-seed-request.json").resolve()
        ),
        "bootstrap_publication_request_output": str(
            (outputs / "bootstrap-publication-request.json").resolve()
        ),
        "retained_runtime_inputs_output": str(
            (outputs / "retained-runtime-inputs.json").resolve()
        ),
    }
    unsigned: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bootstrap_request_materialization_v2",
        **source_paths,
        **output_paths,
    }
    return {
        **unsigned,
        "canonical_identity_sha256": canonical_sha256(unsigned),
    }, checkpoint


def test_materializer_emits_only_pre_effect_authorities(tmp_path: Path) -> None:
    from glm52_enforcement.support_plane import (
        retained_fence_bootstrap_inputs_from_mapping,
    )

    builder = _load_builder()
    invocation, _checkpoint = _invocation(tmp_path)
    built = builder.build_requests(invocation)

    bootstrap_inputs = retained_fence_bootstrap_inputs_from_mapping(
        built["retained_bootstrap_inputs"]
    )
    seed = built["bridge_seed_publication"]
    publication = built["bootstrap_fence_publication"]
    runtime = built["retained_fence_runtime_deployment"]
    assert bootstrap_inputs.activation_id == seed["activation_id"]
    assert "materializer_function_version_arn" not in seed
    assert "materializer_function_version_arn" not in publication
    assert "checkpoint_identity_sha256" not in publication
    assert "bootstrap_manifest_coordinate" not in runtime
    assert "fence_stack_id" not in runtime
    for path in built["output_paths"]:
        raw = Path(path).read_bytes()
        assert raw == canonical_json_bytes(json.loads(raw)) + b"\n"


def test_materializer_refuses_drift_and_existing_outputs(tmp_path: Path) -> None:
    builder = _load_builder()
    invocation, _checkpoint = _invocation(tmp_path)
    seed_path = Path(invocation["bridge_seed_authority_path"])
    seed = json.loads(seed_path.read_bytes())
    seed["activation_id"] = "foreign-activation"
    _write(seed_path, seed)
    with pytest.raises(builder.BootstrapRequestMaterializationError, match="identity"):
        builder.build_requests(invocation)

    invocation, _checkpoint = _invocation(tmp_path / "second")
    target = Path(invocation["bridge_seed_request_output"])
    target.write_text("sentinel\n", encoding="ascii")
    with pytest.raises(builder.BootstrapRequestMaterializationError, match="new exact"):
        builder.build_requests(invocation)
