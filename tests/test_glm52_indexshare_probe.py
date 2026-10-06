from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "probe_glm52_indexshare_runtime.py"
)
MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PROFILE_NAME = "glm52-reap-504b-v2"
PINNED_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--0xSero--glm-5.2-reap-504B-v2"
    / "snapshots"
    / REVISION
)
INDEXER_SUFFIXES = (
    "k_norm.bias",
    "k_norm.weight",
    "weights_proj.weight",
    "wk.weight",
    "wq_b.weight",
)


def _api():
    assert SCRIPT_PATH.is_file(), "the durable GLM52 IndexShare probe CLI is missing"
    spec = importlib.util.spec_from_file_location(
        "probe_glm52_indexshare_runtime_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_schedule() -> list[str]:
    return [
        "full" if (max(layer - 3 + 1, 0) % 4) == 0 else "shared"
        for layer in range(78)
    ]


def _production_config() -> dict[str, object]:
    return {
        "model_type": "glm_moe_dsa",
        "vocab_size": 154_880,
        "hidden_size": 6_144,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2_048,
        "intermediate_size": 12_288,
        "moe_intermediate_size": 2_048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "n_shared_experts": 1,
        "n_routed_experts": 168,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "q_lora_rank": 2_048,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "qk_nope_head_dim": 192,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 8,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 3,
        "max_position_embeddings": 1_048_576,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 8_000_000, "rope_type": "default"},
        "attention_bias": False,
        "indexer_types": _production_schedule(),
        "index_topk_pattern": None,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "mlp_layer_types": ["dense", "dense", "dense"] + ["sparse"] * 75,
        "rope_interleave": True,
        "indexer_rope_interleave": True,
    }


def _production_index() -> dict[str, object]:
    full_layers = [
        layer
        for layer, indexer_type in enumerate(_production_schedule())
        if indexer_type == "full"
    ]
    names = [
        f"model.layers.{layer}.self_attn.indexer.{suffix}"
        for layer in [*full_layers, 78]
        for suffix in INDEXER_SUFFIXES
    ]
    return {
        "metadata": {"total_size": 1},
        "weight_map": {name: "fixture.safetensors" for name in names},
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config_path.write_text(json.dumps(_production_config(), sort_keys=True) + "\n")
    index_path.write_text(json.dumps(_production_index(), sort_keys=True) + "\n")
    return config_path, index_path


def _fixture_contract(api, config_path: Path, index_path: Path):
    return api.GLM52IndexShareIdentityContract(
        model_id=MODEL_ID,
        revision=REVISION,
        profile_name=PROFILE_NAME,
        config_sha256=_sha256(config_path),
        index_sha256=_sha256(index_path),
    )


def _probe_kwargs(config_path: Path, index_path: Path) -> dict[str, object]:
    return {
        "config_path": config_path,
        "index_path": index_path,
        "model_id": MODEL_ID,
        "revision": REVISION,
    }


def test_probe_runs_static_and_synthetic_indexshare_contracts(tmp_path: Path) -> None:
    api = _api()
    config_path, index_path = _write_fixture(tmp_path)
    contract = _fixture_contract(api, config_path, index_path)

    payload = api.probe_glm52_indexshare_runtime(
        **_probe_kwargs(config_path, index_path),
        identity_contract=contract,
    )

    assert payload["record_type"] == "glm52_indexshare_runtime_probe"
    assert payload["probe_status"] == "glm52_indexshare_runtime_probe_ready"
    assert payload["probe_pass"] is True
    assert payload["production_identity_contract"] is False
    assert payload["profile_validation_pass"] is True
    assert payload["static_contract_pass"] is True
    assert payload["production_static_contract"] is False
    assert payload["synthetic_runtime_contract"] is True
    assert payload["full_layer_count"] == 21
    assert payload["shared_layer_count"] == 57
    assert payload["main_indexer_tensor_count"] == 105
    assert payload["mtp_indexer_tensor_count"] == 5
    assert payload["synthetic_runtime"]["index_topk"] == 2
    assert payload["synthetic_runtime"]["prefill_tokens"] == 3
    assert payload["synthetic_runtime"]["prefill_topk_shape"] == [1, 1, 3, 2]
    assert payload["synthetic_runtime"]["decode_topk_shape"] == [1, 1, 1, 2]
    assert payload["synthetic_runtime"]["prefill_topk_reuse_exact"] is True
    assert payload["synthetic_runtime"]["decode_topk_reuse_exact"] is True
    assert payload["synthetic_runtime"]["prefill_full_cache_offsets"] == [3, 3]
    assert payload["synthetic_runtime"]["prefill_shared_cache_offsets"] == [3]
    assert payload["synthetic_runtime"]["decode_full_cache_offsets"] == [4, 4]
    assert payload["synthetic_runtime"]["decode_shared_cache_offsets"] == [4]
    assert payload["synthetic_runtime"]["prefill_output_finite"] is True
    assert payload["synthetic_runtime"]["decode_output_finite"] is True
    assert payload["synthetic_runtime"]["dense_routed_weight_present"] is False
    guard = payload["shared_missing_topk_guard"]
    assert guard["guard_pass"] is True
    assert guard["error_type"] == "ValueError"
    assert guard["cache_offset_before"] == 0
    assert guard["cache_offset_after"] == 0
    assert "requires top-k indices from a previous full layer" in guard["message"]
    assert payload["production_long_context_proven"] is False
    assert payload["whole_model_runtime_proven"] is False
    assert payload["full_model_bind_proven"] is False
    assert payload["generation_proven"] is False


@pytest.mark.skipif(
    not (PINNED_SNAPSHOT / "config.json").is_file()
    or not (PINNED_SNAPSHOT / "model.safetensors.index.json").is_file(),
    reason="pinned GLM52 config/index are not resident",
)
def test_probe_accepts_resident_pinned_production_contract() -> None:
    api = _api()

    payload = api.probe_glm52_indexshare_runtime(
        config_path=PINNED_SNAPSHOT / "config.json",
        index_path=PINNED_SNAPSHOT / "model.safetensors.index.json",
        model_id=MODEL_ID,
        revision=REVISION,
    )

    assert payload["probe_pass"] is True
    assert payload["production_identity_contract"] is True
    assert payload["production_static_contract"] is True
    assert payload["synthetic_runtime_contract"] is True
    assert payload["production_long_context_proven"] is False


def test_main_writes_atomic_failure_json_for_config_identity_drift(
    tmp_path: Path,
) -> None:
    api = _api()
    config_path, index_path = _write_fixture(tmp_path)
    contract = _fixture_contract(api, config_path, index_path)
    config_path.write_text(config_path.read_text() + "\n")
    output_path = tmp_path / "failure.json"

    exit_code = api.main(
        [
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(output_path),
        ],
        identity_contract=contract,
    )

    failure = json.loads(output_path.read_text())
    assert exit_code == 1
    assert failure["probe_status"] == "glm52_indexshare_runtime_probe_failed"
    assert failure["probe_pass"] is False
    assert failure["failure_stage"] == "immutable_file_identity"
    assert failure["production_static_contract"] is False
    assert failure["synthetic_runtime_contract"] is False
    assert failure["input_error"]["type"] == "GLM52IndexShareProbeError"
    assert "config SHA-256" in failure["input_error"]["message"]
    assert list(tmp_path.glob(".failure.json.*.tmp")) == []


def test_main_writes_failure_json_for_static_indexer_inventory_drift(
    tmp_path: Path,
) -> None:
    api = _api()
    config_path, index_path = _write_fixture(tmp_path)
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.layers.74.self_attn.indexer.wq_b.weight"]
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
    contract = _fixture_contract(api, config_path, index_path)
    output_path = tmp_path / "static-failure.json"

    exit_code = api.main(
        [
            "--config-path",
            str(config_path),
            "--index-path",
            str(index_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(output_path),
        ],
        identity_contract=contract,
    )

    failure = json.loads(output_path.read_text())
    assert exit_code == 1
    assert failure["failure_stage"] == "production_static_contract"
    assert failure["static_contract_pass"] is False
    assert "missing_indexer_tensors" in failure["input_error"]["message"]
