from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from keep.convert.stream_convert import load_safetensors_index
from ramp.models import glm52_vq_adapter
from ramp.models.glm52_vq_adapter import (
    GLM52VQModel,
    GLM52VQModelArgs,
    Glm52VQDecoderLayer,
    Glm52VQMoE,
    bind_glm52_non_vq_weights,
    bind_glm52_vq_experts,
    bind_glm52_vq_experts_from_paths,
    collect_glm52_non_vq_weights,
    dense_glm52_routed_parameter_names,
    glm52_vq_args_from_config,
    has_unbound_vq_experts,
)
from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from ramp.models.profiles import ModelProfile


def _tiny_args() -> GLM52VQModelArgs:
    return GLM52VQModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        v_head_dim=4,
        qk_nope_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=["full", "full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse", "sparse"],
    )


def _tiny_profile(*, experts: int = 2) -> ModelProfile:
    return ModelProfile(
        name=f"glm52-layer-bind-fixture-{experts}",
        hf_model_id="fixture/glm52-layer-bind",
        revision="fixture-revision",
        architecture="glm_moe_dsa",
        num_layers=3,
        num_sparse_layers=2,
        first_sparse_layer=1,
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=experts,
        experts_per_tok=1,
        vocab_size=32,
        shared_experts=1,
        converter="glm52_vq_groups",
        fused_gate_up=False,
        default_code_bits=8,
        group_size_policy={"gate_proj": 8, "up_proj": 8, "down_proj": 8},
        default_engine="vq_e1_routed_nax_e8",
    )


def _write_tiny_vq_switch_artifacts(path: Path, layers: tuple[int, ...]) -> None:
    rng = np.random.default_rng(5202)
    path.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        gate = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
        up = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
        down = mx.array(rng.normal(scale=0.04, size=(2, 16, 8)).astype(np.float32))
        switch = QuantizedVQSwitchGLU.from_weights(
            gate_weight=gate,
            up_weight=up,
            down_weight=down,
            group_size=8,
        )
        prefix = f"model.layers.{layer}.mlp.switch_mlp"
        for projection, linear in {
            "gate_proj": switch.gate_proj,
            "up_proj": switch.up_proj,
            "down_proj": switch.down_proj,
        }.items():
            mx.save_safetensors(
                str(path / f"layer-{layer:05d}-{projection}.safetensors"),
                {
                    f"{prefix}.{projection}.codes": linear.codes,
                    f"{prefix}.{projection}.scales": linear.scales,
                    "model.vq_codebook.e8": linear.codebook,
                },
            )


def _kv_b_from_split_weights(embed_q: mx.array, unembed_out: mx.array) -> mx.array:
    nope = embed_q.swapaxes(-1, -2)
    combined = mx.concatenate([nope, unembed_out], axis=1)
    return combined.reshape(-1, combined.shape[-1])


_MODEL_OPT_ROUTED_SUFFIXES = (
    "down_proj.input_scale",
    "down_proj.weight",
    "down_proj.weight_scale",
    "down_proj.weight_scale_2",
    "gate_proj.input_scale",
    "gate_proj.weight",
    "gate_proj.weight_scale",
    "gate_proj.weight_scale_2",
    "up_proj.input_scale",
    "up_proj.weight",
    "up_proj.weight_scale",
    "up_proj.weight_scale_2",
)


def _write_tiny_decoder_layer_source_checkpoint(
    path: Path,
    layer: Glm52VQDecoderLayer,
    *,
    omit_local: tuple[str, ...] = (),
    unexpected_local: tuple[str, ...] = (),
    include_routed_companions: bool = True,
) -> tuple[Path, dict[str, mx.array], tuple[str, ...]]:
    layer_idx = layer.layer_idx
    layer_prefix = f"model.layers.{layer_idx}"
    params = dict(tree_flatten(layer.parameters()))
    expected: dict[str, mx.array] = {}
    for ordinal, (name, value) in enumerate(sorted(params.items()), start=1):
        expected[name] = mx.full(
            value.shape,
            ordinal / 32.0,
            dtype=mx.bfloat16,
        )

    embed_q_name = "self_attn.embed_q.weight"
    unembed_out_name = "self_attn.unembed_out.weight"
    arrays: dict[str, mx.array] = {
        f"{layer_prefix}.{name}": value
        for name, value in expected.items()
        if name not in {embed_q_name, unembed_out_name, *omit_local}
    }
    if embed_q_name not in omit_local and unembed_out_name not in omit_local:
        arrays[f"{layer_prefix}.self_attn.kv_b_proj.weight"] = _kv_b_from_split_weights(
            expected[embed_q_name],
            expected[unembed_out_name],
        )

    # A real layer-local bind runs against the full source index. These entries
    # must be out of scope rather than treated as selected-layer surprises.
    arrays["model.layers.0.input_layernorm.weight"] = mx.zeros(
        (16,), dtype=mx.bfloat16
    )
    arrays["model.norm.weight"] = mx.zeros((16,), dtype=mx.bfloat16)

    routed_names: list[str] = []
    if include_routed_companions:
        for suffix in _MODEL_OPT_ROUTED_SUFFIXES:
            name = f"{layer_prefix}.mlp.experts.0.{suffix}"
            routed_names.append(name)
            arrays[name] = mx.zeros((1,), dtype=mx.bfloat16)
    for name in unexpected_local:
        arrays[f"{layer_prefix}.{name}"] = mx.zeros((1,), dtype=mx.bfloat16)

    path.mkdir(parents=True, exist_ok=True)
    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in arrays},
            },
            sort_keys=True,
        )
        + "\n"
    )
    return index_path, expected, tuple(sorted(routed_names))


def _write_tiny_non_vq_source_checkpoint(path: Path, model: GLM52VQModel) -> Path:
    params = dict(tree_flatten(model.parameters()))
    arrays = {}
    weight_map = {}
    shard_name = "model-00001-of-00001.safetensors"
    skip_keys = set()

    for layer_idx in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.self_attn"
        embed_q_key = f"{prefix}.embed_q.weight"
        unembed_out_key = f"{prefix}.unembed_out.weight"
        skip_keys.update({embed_q_key, unembed_out_key})
        kv_b_key = f"{prefix}.kv_b_proj.weight"
        arrays[kv_b_key] = _kv_b_from_split_weights(
            params[embed_q_key],
            params[unembed_out_key],
        ).astype(mx.bfloat16)
        weight_map[kv_b_key] = shard_name

    for key, value in params.items():
        if key in skip_keys or ".mlp.switch_mlp." in key:
            continue
        arrays[key] = value.astype(mx.bfloat16) if value.dtype in (mx.float32, mx.float16) else value
        weight_map[key] = shard_name

    for layer_idx in (1, 2):
        for expert in range(model.args.n_routed_experts):
            for projection, shape in {
                "gate_proj": (8, 16),
                "up_proj": (8, 16),
                "down_proj": (16, 8),
            }.items():
                key = f"model.layers.{layer_idx}.mlp.experts.{expert}.{projection}.weight"
                arrays[key] = mx.zeros(shape, dtype=mx.bfloat16)
                weight_map[key] = shard_name

    mtp_key = "model.layers.3.mlp.experts.0.gate_proj.weight"
    arrays[mtp_key] = mx.zeros((8, 16), dtype=mx.bfloat16)
    weight_map[mtp_key] = shard_name

    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return index_path


def test_glm52_vq_model_uses_indexshare_cache_schedule_without_dense_switch_weights() -> None:
    model = GLM52VQModel(_tiny_args())

    assert model.model.layers[0].self_attn.indexer is not None
    assert model.model.layers[1].self_attn.indexer is not None
    assert model.model.layers[2].self_attn.indexer is None
    assert len(model.make_cache()[0].caches) == 2
    assert len(model.make_cache()[2].caches) == 1
    assert isinstance(model.model.layers[1].mlp, Glm52VQMoE)
    assert model.model.layers[1].mlp.switch_mlp is None

    params = dict(tree_flatten(model.parameters()))
    assert not any(".mlp.switch_mlp." in key and key.endswith(".weight") for key in params)


def test_glm52_vq_args_from_config_keeps_indexshare_schedule() -> None:
    args = _tiny_args()
    rebuilt = glm52_vq_args_from_config(dict(args.__dict__))

    assert rebuilt.indexer_types == ["full", "full", "shared"]
    assert rebuilt.mlp_layer_types == ["dense", "sparse", "sparse"]


def test_glm52_vq_model_binds_artifact_switches_and_runs_tiny_forward(tmp_path) -> None:
    model = GLM52VQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))

    assert has_unbound_vq_experts(model)
    bound = bind_glm52_vq_experts(model, tmp_path)

    assert bound == (1, 2)
    assert not has_unbound_vq_experts(model)
    assert isinstance(model.model.layers[1].mlp.switch_mlp, QuantizedVQSwitchGLU)

    logits = model(mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(logits)

    assert logits.shape == (1, 2, 32)
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_glm52_strict_bulk_vq_bind_uses_profile_and_exact_sparse_layer_set(
    tmp_path: Path,
) -> None:
    model = GLM52VQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))

    bound = bind_glm52_vq_experts(
        model,
        tmp_path,
        profile=_tiny_profile(),
        strict=True,
    )

    assert bound == (1, 2)
    assert has_unbound_vq_experts(model) is False
    assert dense_glm52_routed_parameter_names(model) == ()


def test_glm52_strict_vq_bind_accepts_explicit_authenticated_paths(
    tmp_path: Path,
) -> None:
    model = GLM52VQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    descriptors = {
        path.name: os.open(path, os.O_RDONLY)
        for path in tmp_path.glob("layer-*.safetensors")
    }
    paths = {
        filename: Path(f"/dev/fd/{descriptor}")
        for filename, descriptor in descriptors.items()
    }
    try:
        bound = bind_glm52_vq_experts_from_paths(
            model,
            paths,
            profile=_tiny_profile(),
            strict=True,
        )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    assert bound == (1, 2)
    assert has_unbound_vq_experts(model) is False


@pytest.mark.parametrize(
    "kwargs",
    ({}, {"profile": _tiny_profile()}, {"strict": True}),
)
def test_glm52_explicit_path_bind_requires_strict_profiled_mode(
    kwargs: dict[str, object],
) -> None:
    model = GLM52VQModel(_tiny_args())

    with pytest.raises(ValueError, match="explicit.*strict.*profile"):
        bind_glm52_vq_experts_from_paths(model, {}, **kwargs)


def test_glm52_base_bind_rejects_non_strict_explicit_paths() -> None:
    model = GLM52VQModel(_tiny_args())

    with pytest.raises(ValueError, match="explicit.*strict.*profile"):
        bind_glm52_vq_experts(
            model,
            Path("."),
            artifact_paths={},
            profile=_tiny_profile(),
            strict=False,
        )


@pytest.mark.parametrize("layers", [(0,), (1, 3)])
def test_glm52_strict_bulk_vq_bind_rejects_dense_or_out_of_range_layers(
    tmp_path: Path,
    layers: tuple[int, ...],
) -> None:
    model = GLM52VQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))

    with pytest.raises(ValueError, match="requested sparse layers"):
        bind_glm52_vq_experts(
            model,
            tmp_path,
            layers=layers,
            profile=_tiny_profile(),
            strict=True,
        )


def test_collect_glm52_non_vq_weights_splits_kv_b_and_skips_vq_and_mtp(tmp_path) -> None:
    source_model = GLM52VQModel(_tiny_args())
    index_path = _write_tiny_non_vq_source_checkpoint(tmp_path, source_model)
    target_model = GLM52VQModel(_tiny_args())

    weights, report = collect_glm52_non_vq_weights(
        target_model,
        tmp_path,
        load_safetensors_index(index_path),
    )

    loaded = {key for key, _ in weights}
    assert "model.layers.0.self_attn.embed_q.weight" in loaded
    assert "model.layers.0.self_attn.unembed_out.weight" in loaded
    assert not any(".kv_b_proj." in key for key in loaded)
    assert not any(".mlp.experts." in key for key in loaded)
    assert report.missing_model_parameters == ()
    assert len(report.transformed_kv_b_tensors) == 3
    assert len(report.skipped_routed_expert_tensors) == 12
    assert report.skipped_mtp_tensors == ("model.layers.3.mlp.experts.0.gate_proj.weight",)
    assert dict(weights)["model.layers.0.self_attn.embed_q.weight"].dtype == mx.bfloat16


def test_bind_glm52_non_vq_weights_loads_strict_tiny_source(tmp_path) -> None:
    source_model = GLM52VQModel(_tiny_args())
    index_path = _write_tiny_non_vq_source_checkpoint(tmp_path, source_model)
    target_model = GLM52VQModel(_tiny_args())

    report = bind_glm52_non_vq_weights(
        target_model,
        tmp_path,
        load_safetensors_index(index_path),
    )
    mx.eval(target_model.parameters())

    params = dict(tree_flatten(target_model.parameters()))
    assert report.missing_model_parameters == ()
    assert params["model.layers.0.self_attn.q_b_proj.weight"].dtype == mx.bfloat16
    expected_embed_q = source_model.model.layers[0].self_attn.embed_q.weight.astype(mx.bfloat16).astype(mx.float32)
    np.testing.assert_allclose(
        np.asarray(params["model.layers.0.self_attn.embed_q.weight"].astype(mx.float32)),
        np.asarray(expected_embed_q),
    )


def test_strict_non_vq_bind_rejects_unmatched_package_tensor_before_loading(
    tmp_path: Path,
) -> None:
    source_model = GLM52VQModel(_tiny_args())
    index_path = _write_tiny_non_vq_source_checkpoint(tmp_path, source_model)
    raw_index = json.loads(index_path.read_text())
    raw_index["weight_map"]["model.unexpected.weight"] = (
        "model-00001-of-00001.safetensors"
    )
    index_path.write_text(json.dumps(raw_index))
    target_model = GLM52VQModel(_tiny_args())

    with pytest.raises(ValueError, match="Unexpected .* non-VQ source tensors"):
        bind_glm52_non_vq_weights(
            target_model,
            tmp_path,
            load_safetensors_index(index_path),
            strict=True,
        )


def test_non_vq_target_contract_map_does_not_retain_initializer_arrays() -> None:
    contracts = glm52_vq_adapter._target_non_vq_contracts(
        GLM52VQModel(_tiny_args())
    )

    assert contracts
    assert all(isinstance(contract.shape, tuple) for contract in contracts.values())
    assert not any(isinstance(contract, mx.array) for contract in contracts.values())


def test_decoder_layer_non_vq_bind_excludes_every_modelopt_routed_companion(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx=1)
    index_path, _expected, routed_names = _write_tiny_decoder_layer_source_checkpoint(
        tmp_path,
        layer,
    )

    report = glm52_vq_adapter.bind_glm52_decoder_layer_non_vq_weights(
        layer,
        tmp_path,
        load_safetensors_index(index_path),
        model_args=args,
        strict=True,
    )

    assert report.skipped_routed_expert_tensors == routed_names
    assert all(".mlp.experts." in name for name in routed_names)
    assert len(routed_names) == 12


def test_decoder_layer_non_vq_bind_is_strict_and_splits_kv_b(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx=1)
    index_path, expected, _routed_names = _write_tiny_decoder_layer_source_checkpoint(
        tmp_path,
        layer,
        include_routed_companions=False,
    )

    report = glm52_vq_adapter.bind_glm52_decoder_layer_non_vq_weights(
        layer,
        tmp_path,
        load_safetensors_index(index_path),
        model_args=args,
        strict=True,
    )
    mx.eval(layer.parameters())

    params = dict(tree_flatten(layer.parameters()))
    assert report.missing_model_parameters == ()
    assert report.skipped_unmatched_tensors == ()
    assert report.transformed_kv_b_tensors == (
        "model.layers.1.self_attn.kv_b_proj.weight",
    )
    for name in (
        "input_layernorm.weight",
        "self_attn.embed_q.weight",
        "self_attn.unembed_out.weight",
    ):
        np.testing.assert_array_equal(
            np.asarray(params[name].astype(mx.float32)),
            np.asarray(expected[name].astype(mx.float32)),
        )


@pytest.mark.parametrize(
    ("omit_local", "unexpected_local", "expected_message"),
    [
        (
            ("input_layernorm.weight",),
            (),
            "GLM52 decoder layer 1 strict non-VQ bind failed: "
            "missing_model_parameters=['model.layers.1.input_layernorm.weight']; "
            "unexpected_source_tensors=[]",
        ),
        (
            (),
            ("self_attn.unexpected.weight",),
            "GLM52 decoder layer 1 strict non-VQ bind failed: "
            "missing_model_parameters=[]; "
            "unexpected_source_tensors=['model.layers.1.self_attn.unexpected.weight']",
        ),
    ],
    ids=("missing", "unexpected"),
)
def test_decoder_layer_non_vq_strict_failure_is_exact_and_non_mutating(
    tmp_path: Path,
    omit_local: tuple[str, ...],
    unexpected_local: tuple[str, ...],
    expected_message: str,
) -> None:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx=1)
    index_path, _expected, _routed_names = _write_tiny_decoder_layer_source_checkpoint(
        tmp_path,
        layer,
        omit_local=omit_local,
        unexpected_local=unexpected_local,
    )
    before = np.asarray(layer.post_attention_layernorm.weight).copy()

    with pytest.raises(ValueError) as error:
        glm52_vq_adapter.bind_glm52_decoder_layer_non_vq_weights(
            layer,
            tmp_path,
            load_safetensors_index(index_path),
            model_args=args,
            strict=True,
        )

    assert str(error.value) == expected_message
    np.testing.assert_array_equal(
        np.asarray(layer.post_attention_layernorm.weight),
        before,
    )


def test_decoder_layer_vq_bind_loads_all_three_profile_sized_projections(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx=1)
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1,))
    expected = glm52_vq_adapter.load_glm52_vq_switch_glu(tmp_path, layer=1)

    glm52_vq_adapter.bind_glm52_decoder_layer_vq_experts(
        layer,
        tmp_path,
        profile=_tiny_profile(experts=2),
    )

    switch = layer.mlp.switch_mlp
    assert isinstance(switch, QuantizedVQSwitchGLU)
    assert switch.num_experts == 2
    for projection in ("gate_proj", "up_proj", "down_proj"):
        actual_linear = getattr(switch, projection)
        expected_linear = getattr(expected, projection)
        mx.eval(actual_linear.codes, expected_linear.codes)
        np.testing.assert_array_equal(
            np.asarray(actual_linear.codes),
            np.asarray(expected_linear.codes),
        )


def test_decoder_layer_vq_bind_rejects_profile_expert_count_before_mutation(
    tmp_path: Path,
) -> None:
    layer = Glm52VQDecoderLayer(_tiny_args(), layer_idx=1)
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1,))

    with pytest.raises(ValueError, match="profile.*expert|expert.*profile"):
        glm52_vq_adapter.bind_glm52_decoder_layer_vq_experts(
            layer,
            tmp_path,
            profile=_tiny_profile(experts=3),
        )

    assert layer.mlp.switch_mlp is None


def test_decoder_layer_combined_binding_runs_finite_shared_layer_forward(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx=2)
    source_dir = tmp_path / "source"
    artifact_dir = tmp_path / "artifact"
    index_path, _expected, _routed_names = _write_tiny_decoder_layer_source_checkpoint(
        source_dir,
        layer,
        include_routed_companions=False,
    )
    _write_tiny_vq_switch_artifacts(artifact_dir, layers=(2,))

    report = glm52_vq_adapter.bind_glm52_decoder_layer_non_vq_weights(
        layer,
        source_dir,
        load_safetensors_index(index_path),
        model_args=args,
        strict=True,
    )
    glm52_vq_adapter.bind_glm52_decoder_layer_vq_experts(
        layer,
        artifact_dir,
        profile=_tiny_profile(experts=2),
    )
    output, topk = layer(mx.ones((1, 1, args.hidden_size), dtype=mx.bfloat16))
    mx.eval(output)

    assert report.missing_model_parameters == ()
    assert report.skipped_unmatched_tensors == ()
    assert output.shape == (1, 1, args.hidden_size)
    assert bool(mx.all(mx.isfinite(output)).item())
    assert topk is None


def test_decoder_layer_non_vq_bind_ignores_already_bound_local_vq_parameters(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    layer = Glm52VQDecoderLayer(args, layer_idx=2)
    source_dir = tmp_path / "source"
    artifact_dir = tmp_path / "artifact"
    index_path, _expected, _routed_names = _write_tiny_decoder_layer_source_checkpoint(
        source_dir,
        layer,
        include_routed_companions=False,
    )
    _write_tiny_vq_switch_artifacts(artifact_dir, layers=(2,))

    glm52_vq_adapter.bind_glm52_decoder_layer_vq_experts(
        layer,
        artifact_dir,
        profile=_tiny_profile(experts=2),
    )
    report = glm52_vq_adapter.bind_glm52_decoder_layer_non_vq_weights(
        layer,
        source_dir,
        load_safetensors_index(index_path),
        model_args=args,
        strict=True,
    )

    assert report.missing_model_parameters == ()
    assert report.skipped_unmatched_tensors == ()
