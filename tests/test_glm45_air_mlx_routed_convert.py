from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from keep.convert.mlx_routed_quant import convert_mlx_routed_quant_layer
from keep.convert.stream_convert import load_safetensors_index
from ramp.models.glm45_air_vq_adapter import load_glm45_air_mlx_quantized_switch_glu


def _write_tiny_expert_source(path: Path) -> Path:
    rng = np.random.default_rng(8812)
    arrays = {}
    weight_map = {}
    shard_name = "model-00001-of-00001.safetensors"
    for expert in range(4):
        for projection, shape in {
            "gate_proj": (32, 64),
            "up_proj": (32, 64),
            "down_proj": (64, 32),
        }.items():
            key = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            arrays[key] = mx.array(rng.normal(scale=0.04, size=shape).astype(np.float32)).astype(mx.bfloat16)
            weight_map[key] = shard_name
    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return index_path


def test_convert_mlx_routed_quant_layer_streams_selected_experts(tmp_path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    index_path = _write_tiny_expert_source(source_dir)

    records = convert_mlx_routed_quant_layer(
        source_dir=source_dir,
        index=load_safetensors_index(index_path),
        output_dir=output_dir,
        layer=1,
        num_experts=4,
        group_size=32,
        bits=2,
        mode="affine",
    )
    switch = load_glm45_air_mlx_quantized_switch_glu(output_dir, layer=1)

    assert [record["projection"] for record in records] == ["gate_proj", "up_proj", "down_proj"]
    assert records[0]["source_tensors_read"] == 4
    assert records[0]["bits"] == 2
    assert records[0]["group_size"] == 32
    assert switch.gate_proj.bits == 2
    assert switch.gate_proj.group_size == 32
