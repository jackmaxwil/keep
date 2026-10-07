from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "probe_glm52_synthetic_generation.py"
)


def _api():
    assert SCRIPT_PATH.is_file(), "the durable GLM52 synthetic generation CLI is missing"
    spec = importlib.util.spec_from_file_location(
        "probe_glm52_synthetic_generation_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_runs_repeatable_tiny_whole_model_generate_step() -> None:
    api = _api()

    payload = api.probe_glm52_synthetic_generation()

    assert payload["record_type"] == "glm52_synthetic_generation_probe"
    assert payload["probe_status"] == "glm52_synthetic_generation_probe_ready"
    assert payload["probe_pass"] is True
    assert payload["whole_model_scope"] == "tiny_fixture"
    assert payload["synthetic_runtime_contract"] is True
    assert payload["production_generation_proven"] is False
    assert payload["production_artifact_used"] is False
    assert payload["tokenizer_used"] is False
    assert payload["prompt_token_ids"] == [1, 2, 3]
    assert payload["prompt_token_count"] == 3
    assert payload["requested_generated_tokens"] == 2
    assert payload["greedy_sampling"] is True

    assert payload["sparse_layer_count"] == 1
    assert payload["bound_sparse_layer_count"] == 1
    assert payload["unbound_sparse_layer_count"] == 0
    assert payload["dense_routed_weight_present"] is False

    assert payload["direct_forward"]["logits_shape"] == [1, 3, 32]
    assert payload["direct_forward"]["logits_finite"] is True
    assert payload["direct_forward"]["full_cache_offsets"] == [3, 3]
    assert payload["direct_forward"]["shared_cache_offsets"] == [3]

    generation = payload["generate_step"]
    assert generation["generated_token_count"] == 2
    assert len(generation["generated_token_ids"]) == 2
    assert generation["generated_ids_within_vocab"] is True
    assert generation["logprobs_shape"] == [2, 32]
    assert generation["logprobs_finite"] is True
    assert generation["cache_offsets_after_yields"] == [
        {
            "yield_index": 1,
            "full_layer": [4, 4],
            "shared_layer": [4],
        },
        {
            "yield_index": 2,
            "full_layer": [5, 5],
            "shared_layer": [5],
        },
    ]
    assert generation["final_full_cache_offsets"] == [5, 5]
    assert generation["final_shared_cache_offsets"] == [5]

    repeat = payload["repeatability"]
    assert repeat["repeatability_pass"] is True
    assert repeat["generated_token_ids"] == generation["generated_token_ids"]
    assert repeat["logprobs_sha256"] == generation["logprobs_sha256"]
    assert repeat["final_full_cache_offsets"] == [5, 5]
    assert repeat["final_shared_cache_offsets"] == [5]


def test_main_writes_atomic_success_json(tmp_path: Path) -> None:
    api = _api()
    output_path = tmp_path / "generation.json"

    exit_code = api.main(["--output-json", str(output_path)])

    payload = json.loads(output_path.read_text())
    assert exit_code == 0
    assert payload["probe_pass"] is True
    assert payload["synthetic_runtime_contract"] is True
    assert payload["production_generation_proven"] is False
    assert list(tmp_path.glob(".generation.json.*.tmp")) == []


def test_main_writes_atomic_failure_json(tmp_path: Path) -> None:
    api = _api()
    output_path = tmp_path / "failure.json"

    def fail_probe():
        raise api.GLM52SyntheticGenerationProbeError(
            "generate_step",
            "injected deterministic failure",
        )

    api.probe_glm52_synthetic_generation = fail_probe
    exit_code = api.main(["--output-json", str(output_path)])

    payload = json.loads(output_path.read_text())
    assert exit_code == 1
    assert payload["probe_status"] == "glm52_synthetic_generation_probe_failed"
    assert payload["probe_pass"] is False
    assert payload["failure_stage"] == "generate_step"
    assert payload["input_error"] == {
        "type": "GLM52SyntheticGenerationProbeError",
        "message": "injected deterministic failure",
    }
    assert payload["whole_model_scope"] == "tiny_fixture"
    assert payload["synthetic_runtime_contract"] is False
    assert payload["production_generation_proven"] is False
    assert payload["production_artifact_used"] is False
    assert payload["tokenizer_used"] is False
    assert list(tmp_path.glob(".failure.json.*.tmp")) == []
