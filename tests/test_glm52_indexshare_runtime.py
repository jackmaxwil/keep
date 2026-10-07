from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList, KVCache

from ramp.models import glm52_vq_adapter
from ramp.models.glm52_vq_adapter import (
    GLM52VQModel,
    GLM52VQModelArgs,
    Glm52VQMoE,
)
from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU


_PINNED_REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
_PINNED_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--0xSero--glm-5.2-reap-504B-v2"
    / "snapshots"
    / _PINNED_REVISION
)
_INDEXER_SUFFIXES = (
    "k_norm.bias",
    "k_norm.weight",
    "weights_proj.weight",
    "wk.weight",
    "wq_b.weight",
)


def _tiny_indexshare_args() -> GLM52VQModelArgs:
    return GLM52VQModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
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
        indexer_types=["full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )


def _production_schedule() -> list[str]:
    return [
        "full" if layer < 3 or (layer >= 6 and (layer - 2) % 4 == 0) else "shared"
        for layer in range(78)
    ]


def _production_static_fixture() -> tuple[dict[str, object], dict[str, object]]:
    schedule = _production_schedule()
    config: dict[str, object] = {
        "num_hidden_layers": 78,
        "indexer_types": schedule,
        "index_topk_pattern": None,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "rope_interleave": True,
        "indexer_rope_interleave": True,
    }
    full_layers = [layer for layer, kind in enumerate(schedule) if kind == "full"]
    names = [
        f"model.layers.{layer}.self_attn.indexer.{suffix}"
        for layer in [*full_layers, 78]
        for suffix in _INDEXER_SUFFIXES
    ]
    index = {"metadata": {}, "weight_map": {name: "fixture.safetensors" for name in names}}
    return config, index


def test_synthetic_full_to_shared_chain_propagates_topk_and_advances_caches() -> None:
    args = _tiny_indexshare_args()
    model = GLM52VQModel(args)
    full_layer, shared_layer = model.layers
    assert isinstance(shared_layer.mlp, Glm52VQMoE)
    shared_layer.mlp.bind_switch_mlp(
        QuantizedVQSwitchGLU.from_weights(
            gate_weight=mx.full((2, 8, 16), 0.03125, dtype=mx.float32),
            up_weight=mx.full((2, 8, 16), -0.015625, dtype=mx.float32),
            down_weight=mx.full((2, 16, 8), 0.0078125, dtype=mx.float32),
            group_size=8,
        )
    )
    caches = model.make_cache()

    assert isinstance(caches[0], CacheList)
    assert isinstance(caches[1], CacheList)
    assert len(caches[0].caches) == 2
    assert len(caches[1].caches) == 1
    assert all(isinstance(cache, KVCache) for cache in caches[0].caches)
    assert all(isinstance(cache, KVCache) for cache in caches[1].caches)

    prefill = mx.arange(3 * args.hidden_size, dtype=mx.float32).reshape(
        1, 3, args.hidden_size
    ) / 64.0
    prefill_mask = create_attention_mask(
        prefill,
        caches[0][0],
        return_array=True,
    )
    full_output, full_topk = full_layer(prefill, prefill_mask, caches[0], None)
    shared_output, shared_topk = shared_layer(
        full_output,
        prefill_mask,
        caches[1],
        full_topk,
    )
    mx.eval(shared_output, full_topk, shared_topk)

    assert full_topk is not None
    assert full_topk.shape == (1, 1, 3, 2)
    np.testing.assert_array_equal(np.asarray(shared_topk), np.asarray(full_topk))
    assert all(cache.offset == 3 for cache in caches[0].caches)
    assert caches[1][0].offset == 3
    assert bool(mx.all(mx.isfinite(shared_output)).item())

    decode = mx.full((1, 1, args.hidden_size), 0.125, dtype=mx.float32)
    decode_mask = create_attention_mask(
        decode,
        caches[0][0],
        return_array=True,
    )
    full_decode, full_decode_topk = full_layer(
        decode,
        decode_mask,
        caches[0],
        None,
    )
    shared_decode, shared_decode_topk = shared_layer(
        full_decode,
        decode_mask,
        caches[1],
        full_decode_topk,
    )
    mx.eval(shared_decode, full_decode_topk, shared_decode_topk)

    assert full_decode_topk is not None
    assert full_decode_topk.shape == (1, 1, 1, 2)
    np.testing.assert_array_equal(
        np.asarray(shared_decode_topk),
        np.asarray(full_decode_topk),
    )
    assert all(cache.offset == 4 for cache in caches[0].caches)
    assert caches[1][0].offset == 4
    assert bool(mx.all(mx.isfinite(shared_decode)).item())

    parameter_names = {name for name, _value in tree_flatten(model.parameters())}
    assert not any(".mlp.experts." in name for name in parameter_names)
    assert not any(
        ".mlp.switch_mlp." in name and name.endswith(".weight")
        for name in parameter_names
    )


def test_production_static_contract_is_exact_and_excludes_layer78_mtp() -> None:
    config, index = _production_static_fixture()

    report = glm52_vq_adapter.audit_glm52_indexshare_static_contract(config, index)

    assert report.audit_pass
    assert report.full_layers == (
        0,
        1,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        42,
        46,
        50,
        54,
        58,
        62,
        66,
        70,
        74,
    )
    assert len(report.full_layers) == 21
    assert len(report.shared_layers) == 57
    assert report.main_indexer_tensor_count == 105
    assert len(report.mtp_indexer_tensors) == 5
    assert all(name.startswith("model.layers.78.") for name in report.mtp_indexer_tensors)
    assert report.production_long_context_proven is False


@pytest.mark.skipif(
    not (_PINNED_SNAPSHOT / "config.json").is_file()
    or not (_PINNED_SNAPSHOT / "model.safetensors.index.json").is_file(),
    reason="pinned GLM52 source config/index are not resident",
)
def test_resident_pinned_production_config_and_index_match_static_contract() -> None:
    config = json.loads((_PINNED_SNAPSHOT / "config.json").read_text())
    index = json.loads(
        (_PINNED_SNAPSHOT / "model.safetensors.index.json").read_text()
    )

    report = glm52_vq_adapter.audit_glm52_indexshare_static_contract(config, index)

    assert report.audit_pass
    assert len(report.full_layers) == 21
    assert len(report.shared_layers) == 57
    assert report.main_indexer_tensor_count == 105
    assert len(report.mtp_indexer_tensors) == 5


def test_static_contract_fails_loud_on_missing_full_indexer_tensor() -> None:
    config, index = _production_static_fixture()
    del index["weight_map"]["model.layers.74.self_attn.indexer.wq_b.weight"]

    with pytest.raises(
        ValueError,
        match=r"missing_indexer_tensors=.*model\.layers\.74\.self_attn\.indexer\.wq_b\.weight",
    ):
        glm52_vq_adapter.audit_glm52_indexshare_static_contract(config, index)


def test_static_contract_fails_loud_on_indexer_tensor_in_shared_layer() -> None:
    config, index = _production_static_fixture()
    unexpected = "model.layers.3.self_attn.indexer.wq_b.weight"
    index["weight_map"][unexpected] = "fixture.safetensors"

    with pytest.raises(
        ValueError,
        match=r"unexpected_indexer_tensors=.*model\.layers\.3\.self_attn\.indexer\.wq_b\.weight",
    ):
        glm52_vq_adapter.audit_glm52_indexshare_static_contract(config, index)


@pytest.mark.parametrize("field", ["rope_interleave", "indexer_rope_interleave"])
def test_runtime_args_reject_unsupported_noninterleaved_rope(field: str) -> None:
    config = dict(_tiny_indexshare_args().__dict__)
    config[field] = False

    with pytest.raises(ValueError, match=rf"{field} must be true"):
        glm52_vq_adapter.glm52_vq_args_from_config(config)


def test_runtime_args_reject_shared_schedule_without_prior_full_indexer() -> None:
    config = dict(_tiny_indexshare_args().__dict__)
    config["indexer_types"] = ["shared", "full"]

    with pytest.raises(ValueError, match="indexer_types must start with 'full'"):
        glm52_vq_adapter.glm52_vq_args_from_config(config)


def test_shared_layer_fails_loud_without_topk_after_sparse_threshold() -> None:
    args = _tiny_indexshare_args()
    model = GLM52VQModel(args)
    shared_attention = model.layers[1].self_attn
    shared_cache = model.make_cache()[1]
    hidden = mx.ones((1, args.index_topk + 1, args.hidden_size), dtype=mx.float32)

    with pytest.raises(
        ValueError,
        match="shared IndexShare layer 1 requires top-k indices from a previous full layer",
    ):
        shared_attention(hidden, cache=shared_cache, prev_topk_indices=None)

    assert shared_cache[0].offset == 0
