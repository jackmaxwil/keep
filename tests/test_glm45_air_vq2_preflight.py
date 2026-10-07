from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from ramp.benchmark.vq2_preflight import run_vq2_preflight


def test_vq2_preflight_reports_kernel_status_and_artifact_readiness() -> None:
    report = run_vq2_preflight()

    assert report["encode_e8p_rtn"] is True
    assert report["qmv_uint16"] is True
    assert report["qmm_uint16"] is True
    assert report["gather_uint16"] is True
    assert report["switch_uint16"] is True
    assert report["artifact_rtn_code_bits_16"] is True
    assert report["artifact_ready"] is True
    assert report["real_source_smoke"] == "skipped"


def test_vq2_preflight_can_smoke_one_real_source_tensor(tmp_path) -> None:
    config = {
        "model_type": "glm4_moe",
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
    }
    arrays = {}
    weight_map = {}
    for expert in range(2):
        for projection, shape in {
            "gate_proj": (8, 16),
            "up_proj": (8, 16),
            "down_proj": (16, 8),
        }.items():
            name = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            arrays[name] = mx.array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 128)
            weight_map[name] = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), arrays)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map}))

    report = run_vq2_preflight(
        source_dir=tmp_path,
        index_path=index_path,
        config=config,
        model_id="tiny-glm4",
    )

    smoke = report["real_source_smoke"]
    assert smoke["ok"] is True
    assert smoke["scope"] == "single_real_source_tensor_no_artifact_written"
    assert smoke["estimate_scope"] == "encoder_only_linear_extrapolation_from_single_tensor"
    assert smoke["source_tensor"] == "model.layers.1.mlp.experts.0.gate_proj.weight"
    assert smoke["source_shape"] == [8, 16]
    assert smoke["code_bits"] == 16
    assert smoke["group_size"] == 16
    assert smoke["vectors"] == 16
    assert smoke["codes_dtype"] == "uint16"
    assert smoke["codes_shape"] == [8, 2]
    assert smoke["scales_shape"] == [8, 1]
    assert smoke["planned_vq_groups"] == 3
    assert smoke["planned_vq_total_bytes"] > 0
    assert smoke["linear_estimated_full_routed_encoder_seconds"] is not None
    assert report["artifact_ready"] is True
