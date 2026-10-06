from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.glm4_moe import ModelArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mlx_vq.models.glm45_air_vq_adapter import (
    GLM45AirMLXQuantizedSwitchGLU,
    GLM45AirVQModel,
    bind_glm45_air_mlx_quantized_routed_experts,
    has_dense_glm45_air_routed_expert_parameters,
    has_unbound_glm45_air_vq_experts,
    load_glm45_air_mlx_quantized_switch_glu,
)


def _tiny_args() -> ModelArgs:
    return ModelArgs(
        model_type="glm4_moe",
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        max_position_embeddings=32,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        num_attention_heads=4,
        n_group=1,
        head_dim=16,
        topk_group=1,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        num_hidden_layers=2,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        use_qk_norm=False,
        tie_word_embeddings=False,
        attention_bias=False,
        partial_rotary_factor=0.25,
    )


def _write_mlx_quant_switch_artifacts(path: Path, *, layer: int = 1) -> None:
    rng = np.random.default_rng(2402)
    path.mkdir(parents=True, exist_ok=True)
    switch = GLM45AirMLXQuantizedSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.04, size=(4, 32, 64)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.04, size=(4, 32, 64)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.04, size=(4, 64, 32)).astype(np.float32)),
        group_size=32,
        bits=2,
        mode="affine",
    )
    prefix = f"model.layers.{layer}.mlp.switch_mlp"
    for projection, linear in {
        "gate_proj": switch.gate_proj,
        "up_proj": switch.up_proj,
        "down_proj": switch.down_proj,
    }.items():
        arrays = {
            f"{prefix}.{projection}.weight": linear.weight,
            f"{prefix}.{projection}.scales": linear.scales,
        }
        biases = linear.get("biases")
        if biases is not None:
            arrays[f"{prefix}.{projection}.biases"] = biases
        mx.save_safetensors(
            str(path / f"layer-{layer:05d}-{projection}.safetensors"),
            arrays,
            metadata={
                "mlx_quantization_config": json.dumps(
                    {"group_size": 32, "bits": 2, "mode": "affine"}
                )
            },
        )


def test_mlx_quantized_switch_glu_from_weights_runs_selected_experts() -> None:
    rng = np.random.default_rng(911)
    switch = GLM45AirMLXQuantizedSwitchGLU.from_weights(
        gate_weight=mx.array(rng.normal(scale=0.04, size=(4, 32, 64)).astype(np.float32)),
        up_weight=mx.array(rng.normal(scale=0.04, size=(4, 32, 64)).astype(np.float32)),
        down_weight=mx.array(rng.normal(scale=0.04, size=(4, 64, 32)).astype(np.float32)),
        group_size=32,
        bits=2,
        mode="affine",
    )
    x = mx.array(rng.normal(size=(3, 64)).astype(np.float32))
    indices = mx.array(np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int32))

    actual = switch(x, indices)
    mx.eval(actual)

    assert isinstance(switch.gate_proj, QuantizedSwitchLinear)
    assert switch.input_dims == 64
    assert switch.hidden_dims == 32
    assert switch.num_experts == 4
    assert actual.shape == (3, 2, 64)
    assert bool(mx.all(mx.isfinite(actual)).item())


def test_load_and_bind_mlx_quantized_routed_air_switches(tmp_path) -> None:
    _write_mlx_quant_switch_artifacts(tmp_path, layer=1)
    switch = load_glm45_air_mlx_quantized_switch_glu(tmp_path, layer=1)
    model = GLM45AirVQModel(_tiny_args())

    bound = bind_glm45_air_mlx_quantized_routed_experts(model, tmp_path)
    logits = model(mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(logits)

    assert switch.gate_proj.bits == 2
    assert bound == (1,)
    assert not has_unbound_glm45_air_vq_experts(model)
    assert not has_dense_glm45_air_routed_expert_parameters(model)
    assert logits.shape == (1, 2, 32)
    assert bool(mx.all(mx.isfinite(logits)).item())
