from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_METAL_IMPORT = subprocess.run(
    [sys.executable, "-c", "import mlx.nn"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
if _METAL_IMPORT.returncode != 0:
    pytest.skip(
        "GLM52 adapter integration tests require Metal; NumPy loss oracles run separately",
        allow_module_level=True,
    )

import mlx.core as mx

from benchmarks import finetune_glm52_low_rank as finetune_glm52
from mlx_vq.quality import glm52_adapter_training as adapter_training
from mlx_vq.io.authenticated_artifacts import AuthenticatedFile
from mlx_vq.models.glm4_moe_adapter import (
    GLM4MoeRoutingConfig,
    GLM4MoEGate,
    QuantizedVQSwitchGLU,
)
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear
from mlx_vq.quality.glm52_adapter_training import (
    AdapterTrainingConfig,
    PreparedGLM52TeacherRow,
    ValidatedGLM52TrainingBaseline,
    _StopGradientIndexer,
    attest_glm52_projection_inventory,
    bind_glm52_low_rank_adapters,
    glm52_adapter_loss,
    glm52_feature_space_linear_cka_loss,
    glm52_router_distillation_loss,
    glm52_topk_kl_loss,
    load_authenticated_glm52_adapter,
    prepare_glm52_teacher_rows,
    route_local_projection,
    run_low_rank_training,
    save_authenticated_glm52_adapter,
    selected_glm52_logits_selected_layer,
    validate_tuning_row,
)
from mlx_vq.quality.mlx_surrogate import RouteLocalSwitchLinearSurrogate


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class _Identity:
    def __call__(self, x):
        return x


class _Embedding:
    def __init__(self, vocab: int, hidden: int):
        values = np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden)
        self.weight = mx.array((values % 11 - 5) / 20.0)

    def __call__(self, tokens):
        return self.weight[tokens]


class _Attention:
    def __init__(self, *, shared: bool):
        self.shared = shared

    def __call__(self, x, mask, cache, prev_topk_indices):
        del mask, cache
        if self.shared:
            if prev_topk_indices is None:
                raise AssertionError("shared IndexShare attention did not receive prior indices")
            next_topk = prev_topk_indices
        else:
            next_topk = mx.zeros((x.shape[0], 1, x.shape[1], 1), dtype=mx.int32)
        return mx.zeros_like(x), next_topk


class _ComputedSparseIndexer:
    def __call__(self, x):
        return mx.argmax(x[..., 0], axis=1)[:, None]


class _ComputedSparseAttention:
    def __init__(self):
        self.indexer = _ComputedSparseIndexer()

    def __call__(self, x, mask, cache, prev_topk_indices):
        del mask, cache, prev_topk_indices
        topk_indices = self.indexer(x)
        sparse_mask = mx.zeros(x.shape[:2], dtype=mx.bool_)
        sparse_mask = mx.put_along_axis(
            sparse_mask,
            topk_indices,
            mx.array(True),
            axis=-1,
        )
        attention = mx.where(sparse_mask[..., None], x, mx.zeros_like(x))
        return attention, topk_indices


class _Gate:
    def __init__(self, *, hidden: int = 8, experts: int = 2):
        self.gate = GLM4MoEGate(
            GLM4MoeRoutingConfig(
                hidden_size=hidden,
                moe_intermediate_size=hidden,
                n_routed_experts=experts,
                num_experts_per_tok=1,
                norm_topk_prob=False,
                n_group=1,
                topk_group=1,
                routed_scaling_factor=1.0,
            ),
            weight=mx.array(
                np.arange(experts * hidden, dtype=np.float32).reshape(experts, hidden) / 32.0
            ),
        )

    def __call__(self, x):
        return self.gate(x)


class _MoE:
    def __init__(self, switch_mlp):
        self.switch_mlp = switch_mlp
        self.gate = _Gate(experts=switch_mlp.num_experts)

    def get(self, name):
        assert name == "shared_experts"
        return None


class _DenseMLP:
    def __call__(self, x):
        return mx.zeros_like(x)


class _DecoderLayer:
    def __init__(self, *, mlp, shared_attention: bool):
        self.input_layernorm = _Identity()
        self.post_attention_layernorm = _Identity()
        self.self_attn = _Attention(shared=shared_attention)
        self.mlp = mlp

    def __call__(self, h, mask, cache, prev_topk_indices):
        attention, next_topk = self.self_attn(
            self.input_layernorm(h), mask, cache, prev_topk_indices
        )
        h = h + attention
        return h + self.mlp(self.post_attention_layernorm(h)), next_topk


class _LMHead:
    def __init__(self, hidden: int, vocab: int):
        values = np.arange(hidden * vocab, dtype=np.float32).reshape(hidden, vocab)
        self.weight = mx.array((values % 7 - 3) / 13.0)

    def __call__(self, x):
        return x @ self.weight


class _TinyModel:
    def __init__(
        self,
        switch_mlp,
        *,
        second_switch_mlp=None,
        computed_second_attention: bool = False,
        vocab: int = 7,
        hidden: int = 8,
    ):
        layers = [
            _DecoderLayer(mlp=_MoE(switch_mlp), shared_attention=False),
            _DecoderLayer(
                mlp=_DenseMLP() if second_switch_mlp is None else _MoE(second_switch_mlp),
                shared_attention=True,
            ),
        ]
        if computed_second_attention:
            layers[1].self_attn = _ComputedSparseAttention()
        self.model = SimpleNamespace(
            embed_tokens=_Embedding(vocab, hidden), layers=layers, norm=_Identity()
        )
        self.layers = layers
        self.lm_head = _LMHead(hidden, vocab)


def _projection(
    input_dims: int,
    output_dims: int,
    experts: int = 2,
    *,
    use_gather_vqmm: bool = False,
):
    return QuantizedVQSwitchLinear(
        input_dims=input_dims,
        output_dims=output_dims,
        num_experts=experts,
        codes=mx.zeros((experts, output_dims, input_dims // 8), dtype=mx.uint8),
        scales=mx.ones((experts, output_dims, input_dims // 8), dtype=mx.float32),
        group_size=8,
        code_bits=8,
        use_gather_vqmm=use_gather_vqmm,
    )


def _switch(experts: int = 2, *, use_gather_vqmm: bool = False):
    return QuantizedVQSwitchGLU(
        gate_proj=_projection(8, 8, experts, use_gather_vqmm=use_gather_vqmm),
        up_proj=_projection(8, 8, experts, use_gather_vqmm=use_gather_vqmm),
        down_proj=_projection(8, 8, experts, use_gather_vqmm=use_gather_vqmm),
    )


def _row(split: str = "selection", tuning_eligible: bool = True):
    return PreparedGLM52TeacherRow(
        row_index=0,
        prompt_id="synthetic-000",
        split=split,
        tuning_eligible=tuning_eligible,
        input_token_ids=(1, 2, 3),
        positions=(0, 1),
        target_token_ids=(2, 3),
        teacher_logits=mx.array(
            [[-0.2, 0.1, 1.4, 0.0, -0.4, 0.3, -0.1], [0.0, -0.3, 0.2, 1.1, 0.1, -0.2, 0.4]],
            dtype=mx.float32,
        ),
        teacher_manifest_body_sha256="a" * 64,
        teacher_shard_sha256="b" * 64,
        non_release_waiver=True,
    )


def _projection_map(switch_mlp):
    return {(0, name): getattr(switch_mlp, name) for name in PROJECTIONS}


def test_distillation_config_and_teacher_signal_contract_validation():
    with pytest.raises(ValueError, match="2048 and 8192"):
        AdapterTrainingConfig(
            layers=(0,), projections=("gate_proj",), rank=2, steps=1, topk=1024
        )
    with pytest.raises(ValueError, match="ordered unique"):
        AdapterTrainingConfig(
            layers=(0,),
            projections=("gate_proj",),
            rank=2,
            steps=1,
            cka_enabled=True,
            cka_layers=(2, 1),
        )
    with pytest.raises(ValueError, match="non-negative"):
        AdapterTrainingConfig(
            layers=(0,),
            projections=("gate_proj",),
            rank=2,
            steps=1,
            router_kl_weight=-0.1,
        )

    row = replace(
        _row(),
        topk_logit_ids=mx.array([[2, 1], [3, 6]], dtype=mx.int32),
        topk_logit_values=mx.array([[1.5, 0.2], [1.0, 0.3]], dtype=mx.float16),
        tail_mass=mx.array([0.1, 0.2], dtype=mx.float16),
        teacher_probe_hidden_states={
            0: mx.zeros((2, 8), dtype=mx.float16),
        },
        teacher_router_topk_ids={
            0: mx.array([[0, 1], [1, 0]], dtype=mx.int32),
        },
        teacher_router_topk_weights={
            0: mx.array([[0.75, 0.25], [0.6, 0.4]], dtype=mx.float16),
        },
    )
    validate_tuning_row(row)


def test_topk_full_vocab_zero_tail_matches_dense_loss_bytes():
    teacher = mx.array(
        [[-1.25, 0.5, 2.0, -0.75], [0.125, -0.5, 0.75, 1.5]],
        dtype=mx.float16,
    )
    student = mx.array(
        [[0.25, -0.125, 1.25, -1.0], [-0.75, 0.5, 1.0, 0.25]],
        dtype=mx.float32,
    )
    ids = mx.broadcast_to(mx.arange(4, dtype=mx.int32)[None, :], (2, 4))
    topk_loss = glm52_topk_kl_loss(
        student, ids, teacher, mx.zeros((2,), dtype=mx.float16)
    )
    dense_row = replace(_row(), teacher_logits=teacher.astype(mx.float32))
    dense_loss = glm52_adapter_loss(
        student,
        dense_row,
        target_nll_weight=0.0,
        teacher_top1_margin_weight=0.0,
        teacher_top1_margin=0.0,
        teacher_top1_competitor_token_ids=None,
        teacher_top1_include_hardest_competitor=True,
        tail_kld_weight=0.0,
    )
    mx.eval(topk_loss, dense_loss)
    assert np.asarray(topk_loss).tobytes() == np.asarray(dense_loss).tobytes()


def test_cka_matches_gram_oracle_is_invariant_and_stops_teacher_gradient():
    teacher_np = np.array(
        [[1.0, 2.0, -1.0], [0.0, -1.0, 2.0], [2.0, 0.5, 1.0], [-1.0, 1.5, 0.0]],
        dtype=np.float32,
    )
    student_np = np.array(
        [[0.5, -1.0, 2.0], [1.5, 0.0, -0.5], [-1.0, 2.0, 0.5], [2.0, 1.0, 1.5]],
        dtype=np.float32,
    )
    teacher = mx.array(teacher_np)
    student = mx.array(student_np)
    loss = glm52_feature_space_linear_cka_loss(teacher, student)
    tc = teacher_np - teacher_np.mean(axis=0, keepdims=True)
    sc = student_np - student_np.mean(axis=0, keepdims=True)
    kt = tc @ tc.T
    ks = sc @ sc.T
    gram_similarity = np.sum(kt * ks) / np.sqrt(np.sum(kt * kt) * np.sum(ks * ks))
    mx.eval(loss)
    np.testing.assert_allclose(float(loss.item()), 1.0 - gram_similarity, rtol=1e-5, atol=1e-6)
    assert float(glm52_feature_space_linear_cka_loss(teacher, teacher).item()) == pytest.approx(
        0.0, abs=1e-6
    )
    np.testing.assert_allclose(
        float(glm52_feature_space_linear_cka_loss(teacher, student * 3.0).item()),
        float(loss.item()),
        rtol=1e-5,
        atol=1e-6,
    )

    def teacher_loss(value):
        return glm52_feature_space_linear_cka_loss(value, student)

    _, teacher_grad = mx.value_and_grad(teacher_loss)(teacher)
    mx.eval(teacher_grad)
    assert bool(mx.all(teacher_grad == 0).item())


def test_router_loss_seeded_mc_is_byte_identical_and_all_experts_receive_gradient():
    logits = mx.array(
        [[0.4, -0.2, 1.0, 0.1, -0.5], [-0.1, 0.8, 0.25, -0.4, 0.5]],
        dtype=mx.float32,
    )
    ids = mx.array([[2, 0], [1, 4]], dtype=mx.int32)
    weights = mx.array([[0.75, 0.25], [0.6, 0.4]], dtype=mx.float16)

    def explored(value):
        return glm52_router_distillation_loss(
            value,
            ids,
            weights,
            entropy_beta=0.2,
            mc_expert_explore=True,
            seed=20260712,
        )

    loss_a, grad_a = mx.value_and_grad(explored)(logits)
    loss_b, grad_b = mx.value_and_grad(explored)(logits)
    mx.eval(loss_a, loss_b, grad_a, grad_b)
    assert np.asarray(loss_a).tobytes() == np.asarray(loss_b).tobytes()
    assert np.asarray(grad_a).tobytes() == np.asarray(grad_b).tobytes()
    assert bool(mx.all(mx.any(grad_a != 0, axis=0)).item())


def _peak_memory() -> int:
    getter = getattr(mx, "get_peak_memory", None)
    if callable(getter):
        return int(getter())
    metal = getattr(mx, "metal", None)
    getter = getattr(metal, "get_peak_memory", None)
    return int(getter()) if callable(getter) else -1


def _reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()
        return
    metal = getattr(mx, "metal", None)
    reset = getattr(metal, "reset_peak_memory", None)
    if callable(reset):
        reset()


class _MxEmbedding:
    def __init__(self, vocab: int, hidden: int):
        values = mx.arange(vocab * hidden, dtype=mx.float32).reshape((vocab, hidden))
        self.weight = (values % 13 - 6) / 17.0

    def __call__(self, tokens):
        return self.weight[tokens]


class _CountingAttention:
    def __init__(self, layer_index: int):
        self.layer_index = layer_index
        self.call_count = 0

    def __call__(self, x, mask, cache, prev_topk_indices):
        del mask, cache
        self.call_count += 1
        if prev_topk_indices is None:
            route_signal = mx.zeros((*x.shape[:2], 1), dtype=mx.float32)
        else:
            route_signal = prev_topk_indices.astype(mx.float32)
        attention = x * (0.02 * (self.layer_index + 1)) + route_signal * 0.01
        next_topk = mx.full(
            (*x.shape[:2], 1),
            self.layer_index % 2,
            dtype=mx.int32,
        )
        return attention, next_topk


class _ScaledMLP:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


class _MxGate:
    def __call__(self, x):
        indices = (mx.argmax(x, axis=-1) % 2).astype(mx.int32)[..., None]
        return indices, mx.ones(indices.shape, dtype=mx.float32)


class _MxMoE:
    def __init__(self, switch_mlp):
        self.switch_mlp = switch_mlp
        self.gate = _MxGate()

    def get(self, name):
        assert name == "shared_experts"
        return None


def _mx_projection(hidden: int = 8, experts: int = 2):
    return QuantizedVQSwitchLinear(
        input_dims=hidden,
        output_dims=hidden,
        num_experts=experts,
        codes=mx.zeros((experts, hidden, hidden // 8), dtype=mx.uint8),
        scales=mx.ones((experts, hidden, hidden // 8), dtype=mx.float32),
        group_size=8,
        code_bits=8,
        use_gather_vqmm=False,
    )


def _mx_switch(hidden: int = 8, experts: int = 2):
    return QuantizedVQSwitchGLU(
        gate_proj=_mx_projection(hidden, experts),
        up_proj=_mx_projection(hidden, experts),
        down_proj=_mx_projection(hidden, experts),
    )


class _PrefixCacheToyModel:
    def __init__(self, *, vocab: int = 7, hidden: int = 8):
        self.switches = (_mx_switch(hidden), _mx_switch(hidden))
        self.model = SimpleNamespace(
            embed_tokens=_MxEmbedding(vocab, hidden),
            layers=[
                _DecoderLayer(mlp=_ScaledMLP(0.03), shared_attention=False),
                _DecoderLayer(mlp=_ScaledMLP(-0.02), shared_attention=False),
                _DecoderLayer(mlp=_MxMoE(self.switches[0]), shared_attention=False),
                _DecoderLayer(mlp=_MxMoE(self.switches[1]), shared_attention=False),
            ],
            norm=_Identity(),
        )
        for layer_index, layer in enumerate(self.model.layers):
            layer.self_attn = _CountingAttention(layer_index)
        self.layers = self.model.layers
        self.lm_head = _LMHead(hidden, vocab)


def _prefix_cache_row(
    row_index: int,
    *,
    token_offset: int = 0,
    input_token_ids: tuple[int, ...] | None = None,
):
    tokens = input_token_ids or (1 + token_offset, 2 + token_offset, 3 + token_offset)
    return PreparedGLM52TeacherRow(
        row_index=row_index,
        prompt_id=f"prefix-cache-{row_index}",
        split="selection",
        tuning_eligible=True,
        input_token_ids=tokens,
        positions=(0, 1),
        target_token_ids=(2 + token_offset, 3 + token_offset),
        teacher_logits=mx.zeros((2, 7), dtype=mx.float32),
        teacher_manifest_body_sha256="a" * 64,
        teacher_shard_sha256="b" * 64,
        non_release_waiver=True,
    )


def _top_projection_map(model, projections=("gate_proj",)):
    return {
        (layer_index, projection): getattr(model.switches[layer_index - 2], projection)
        for layer_index in (2, 3)
        for projection in projections
    }


def _bind_toy_adapter_params(model, params):
    for layer_index in (2, 3):
        projection = model.switches[layer_index - 2].gate_proj
        projection.set_continuous_sidecar(
            low_rank_left=params[f"{layer_index}.left"],
            low_rank_right=params[f"{layer_index}.right"],
        )


def test_prefix_cached_logits_match_full_forward_reference():
    model = _PrefixCacheToyModel()
    row = _prefix_cache_row(0)
    layers = frozenset((2, 3))
    full_logits = adapter_training._selected_logits_for_layers(
        model,
        row,
        layers=layers,
        surrogate_projections=frozenset(("gate_proj",)),
        output_chunk_size=4,
        surrogate_downstream=True,
    )
    boundary = adapter_training._frozen_prefix_boundary(model, row, layers=layers)
    cached_logits = adapter_training._selected_logits_from_prefix(
        model,
        row,
        boundary,
        layers=layers,
        output_chunk_size=4,
    )
    mx.eval(full_logits, cached_logits)

    assert bool(mx.allclose(cached_logits, full_logits, atol=1e-7, rtol=1e-7).item())


def test_batched_prefix_boundaries_match_mixed_length_single_row_boundaries():
    rows = [
        _prefix_cache_row(20, input_token_ids=(1, 2)),
        _prefix_cache_row(21, input_token_ids=(2, 3, 4)),
        _prefix_cache_row(22, input_token_ids=(3, 4, 5, 6)),
    ]
    layers = frozenset((2, 3))

    single_model = _PrefixCacheToyModel()
    single = [
        adapter_training._frozen_prefix_boundary(single_model, row, layers=layers)
        for row in rows
    ]
    batched_model = _PrefixCacheToyModel()
    batched = adapter_training._batched_frozen_prefix_boundaries(
        batched_model,
        rows,
        layers=layers,
    )

    assert len(batched) == len(single)
    for (batched_h, batched_prev, batched_mask), (single_h, single_prev, single_mask) in zip(
        batched, single
    ):
        assert bool(mx.allclose(batched_h, single_h, atol=1e-5).item())
        assert bool(mx.array_equal(batched_prev, single_prev).item())
        assert bool(mx.array_equal(batched_mask, single_mask).item())

    # Lengths 2 and 3 share one low-waste bucket; length 4 starts a second.
    # Every frozen prefix layer therefore runs once per bucket, not once per row.
    assert batched_model.model.layers[0].self_attn.call_count == 2
    assert batched_model.model.layers[1].self_attn.call_count == 2


def test_training_precomputes_ram_boundaries_in_one_batched_pass(monkeypatch):
    rows = [_prefix_cache_row(10), _prefix_cache_row(11, token_offset=1)]
    config = AdapterTrainingConfig(
        layers=(2, 3),
        projections=("gate_proj",),
        rank=2,
        steps=5,
        learning_rate=0.01,
        low_rank_init_scale=0.01,
        surrogate_output_chunk_size=4,
    )

    # RAM-cache mode explicitly (independent of ambient env).
    monkeypatch.delenv("GLM52_TRAIN_BOUNDARY_DISK", raising=False)
    # Capacity settings do not reintroduce per-row prefix forwards: RAM mode
    # precomputes every row together and keeps the per-row boundary artifacts.
    monkeypatch.setenv("GLM52_TRAIN_BOUNDARY_CACHE", str(len(rows)))
    cached_model = _PrefixCacheToyModel()
    run_low_rank_training(
        model=cached_model,
        rows=rows,
        projection_map=_top_projection_map(cached_model),
        config=config,
    )
    assert cached_model.model.layers[0].self_attn.call_count == 1
    assert cached_model.model.layers[1].self_attn.call_count == 1

    # The legacy capacity knob likewise cannot cause per-step recomputation.
    monkeypatch.setenv("GLM52_TRAIN_BOUNDARY_CACHE", "1")
    recompute_model = _PrefixCacheToyModel()
    run_low_rank_training(
        model=recompute_model,
        rows=rows,
        projection_map=_top_projection_map(recompute_model),
        config=config,
    )
    assert recompute_model.model.layers[0].self_attn.call_count == 1
    assert recompute_model.model.layers[1].self_attn.call_count == 1


def test_disk_boundary_mode_matches_ram_and_computes_once(monkeypatch):
    rows = [_prefix_cache_row(10), _prefix_cache_row(11, token_offset=1)]
    config = AdapterTrainingConfig(
        layers=(2, 3),
        projections=("gate_proj",),
        rank=2,
        steps=5,
        learning_rate=0.01,
        low_rank_init_scale=0.01,
        surrogate_output_chunk_size=4,
    )

    monkeypatch.delenv("GLM52_TRAIN_BOUNDARY_DISK", raising=False)
    monkeypatch.setenv("GLM52_TRAIN_BOUNDARY_CACHE", str(len(rows)))
    ram_model = _PrefixCacheToyModel()
    ram = run_low_rank_training(
        model=ram_model,
        rows=rows,
        projection_map=_top_projection_map(ram_model),
        config=config,
    )

    monkeypatch.setenv("GLM52_TRAIN_BOUNDARY_DISK", "1")
    disk_model = _PrefixCacheToyModel()
    disk = run_low_rank_training(
        model=disk_model,
        rows=rows,
        projection_map=_top_projection_map(disk_model),
        config=config,
    )

    # Disk mode persists each boundary after one batched prefix pass and
    # yields identical trained parameters to the in-RAM path.
    assert disk_model.model.layers[0].self_attn.call_count == 1
    assert set(disk.params) == set(ram.params)
    for key in ram.params:
        assert bool(mx.allclose(disk.params[key], ram.params[key], atol=1e-6).item())


def test_disk_boundary_mode_persists_each_length_bucket_before_the_next(monkeypatch):
    rows = [
        _prefix_cache_row(20, input_token_ids=(1, 2)),
        _prefix_cache_row(21, input_token_ids=(2, 3, 4)),
        _prefix_cache_row(22, input_token_ids=(3, 4, 5, 6)),
    ]
    config = AdapterTrainingConfig(
        layers=(2, 3),
        projections=("gate_proj",),
        rank=2,
        steps=3,
        learning_rate=0.01,
        low_rank_init_scale=0.01,
        surrogate_output_chunk_size=4,
    )
    model = _PrefixCacheToyModel()
    saved_after_prefix_calls = []
    original_save = adapter_training._save_boundary

    def recording_save(path, boundary):
        saved_after_prefix_calls.append(model.model.layers[0].self_attn.call_count)
        original_save(path, boundary)

    monkeypatch.setenv("GLM52_TRAIN_BOUNDARY_DISK", "1")
    monkeypatch.setattr(adapter_training, "_save_boundary", recording_save)
    run_low_rank_training(
        model=model,
        rows=rows,
        projection_map=_top_projection_map(model),
        config=config,
    )

    # Rows of lengths 2/3 are saved immediately after bucket one; length 4 is
    # saved only after bucket two has run. No all-boundaries RAM accumulation.
    assert saved_after_prefix_calls == [1, 1, 2]


def test_prefix_cached_gradients_match_full_forward_reference():
    model = _PrefixCacheToyModel()
    row = _prefix_cache_row(0)
    layers = frozenset((2, 3))
    boundary = adapter_training._frozen_prefix_boundary(model, row, layers=layers)
    params = {
        f"{layer_index}.left": mx.full((2, 8, 2), 0.02, dtype=mx.float32)
        for layer_index in (2, 3)
    }
    params.update(
        {
            f"{layer_index}.right": mx.full((2, 2, 8), -0.03, dtype=mx.float32)
            for layer_index in (2, 3)
        }
    )

    def full_loss(current):
        _bind_toy_adapter_params(model, current)
        logits = adapter_training._selected_logits_for_layers(
            model,
            row,
            layers=layers,
            surrogate_projections=frozenset(("gate_proj",)),
            output_chunk_size=4,
            surrogate_downstream=True,
        )
        return mx.sum(logits * logits)

    def cached_loss(current):
        _bind_toy_adapter_params(model, current)
        logits = adapter_training._selected_logits_from_prefix(
            model,
            row,
            boundary,
            layers=layers,
            output_chunk_size=4,
        )
        return mx.sum(logits * logits)

    _, full_grads = mx.value_and_grad(full_loss)(params)
    _, cached_grads = mx.value_and_grad(cached_loss)(params)
    mx.eval(*full_grads.values(), *cached_grads.values())

    assert set(cached_grads) == set(full_grads)
    for key in full_grads:
        assert bool(mx.allclose(cached_grads[key], full_grads[key], atol=1e-7, rtol=1e-7).item())


def test_stop_gradient_indexer_detaches_array_result():
    indexer = _StopGradientIndexer(lambda x: x * 2)
    x = mx.array([1.0, 2.0], dtype=mx.float32)

    assert isinstance(indexer(x), mx.array)

    def loss(x):
        return mx.sum(indexer(x))

    _, grad = mx.value_and_grad(loss)(x)
    mx.eval(grad)

    assert bool(mx.all(grad == 0).item())


def test_stop_gradient_indexer_preserves_none_result():
    indexer = _StopGradientIndexer(lambda x: None)

    assert indexer(mx.array([1.0], dtype=mx.float32)) is None


def test_stop_gradient_indexer_preserves_tuple_and_detaches_array_results():
    indexer = _StopGradientIndexer(lambda x: (x, None, x * 2))
    x = mx.array([1.0, 2.0], dtype=mx.float32)
    result = indexer(x)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], mx.array)
    assert result[1] is None
    assert isinstance(result[2], mx.array)

    def loss(x):
        result = indexer(x)
        return mx.sum(result[0]) + mx.sum(result[2])

    _, grad = mx.value_and_grad(loss)(x)
    mx.eval(grad)

    assert bool(mx.all(grad == 0).item())


def test_synthetic_metal_training_and_authenticated_round_trip(tmp_path):
    _reset_peak_memory()
    switch_mlp = _switch()
    model = _TinyModel(switch_mlp)
    row = _row()
    config = AdapterTrainingConfig(
        layers=(0,),
        projections=PROJECTIONS,
        rank=2,
        steps=2,
        learning_rate=0.05,
        low_rank_init_scale=0.1,
        grad_clip_norm=1.0,
        target_nll_weight=0.2,
        teacher_top1_margin_weight=0.1,
        teacher_top1_margin=0.25,
        tail_kld_weight=0.05,
        seed=20260711,
        surrogate_output_chunk_size=4,
    )

    result = run_low_rank_training(
        model=model,
        rows=[row],
        projection_map=_projection_map(switch_mlp),
        config=config,
    )
    mx.eval(result.losses, *result.params.values(), *result.gradient_norms.values())

    assert result.losses.shape == (2,)
    assert bool(mx.all(mx.isfinite(result.losses)).item())
    assert all(bool(mx.isfinite(value).item()) for value in result.gradient_norms.values())
    assert any(
        bool(mx.any(value != 0).item())
        for key, value in result.params.items()
        if key.endswith("low_rank_left")
    )
    for projection in PROJECTIONS:
        assert result.params[f"0.{projection}.low_rank_left"].shape == (2, 8, 2)
        assert result.params[f"0.{projection}.low_rank_right"].shape == (2, 2, 8)

    trained_logits = selected_glm52_logits_selected_layer(
        model,
        row,
        layer=0,
        surrogate_projections=frozenset(PROJECTIONS),
        output_chunk_size=4,
    )
    mx.eval(trained_logits)
    assert trained_logits.shape == (2, 7)

    output_dir = tmp_path / "adapter"
    saved = save_authenticated_glm52_adapter(
        output_dir,
        params=result.params,
        projection_map=_projection_map(switch_mlp),
        parent_candidate_identity_sha256="c" * 64,
        teacher_manifest_body_sha256=row.teacher_manifest_body_sha256,
        training_config=config,
        selection_rows=[row],
        release_eligible=False,
        projection_inventory_attestation=attest_glm52_projection_inventory(
            _projection_map(switch_mlp)
        ),
    )
    saved_manifest = json.loads(
        (output_dir / "glm52-low-rank-adapter-manifest.json").read_text(encoding="utf-8")
    )
    expected_inventory = attest_glm52_projection_inventory(_projection_map(switch_mlp))
    assert saved_manifest["routed_projection_inventory"] == {
        "count": expected_inventory.projection_count,
        "metadata_sha256": expected_inventory.projection_metadata_sha256,
    }
    assert {
        key: saved_manifest["training_config"][key]
        for key in (
            "topk",
            "cka_enabled",
            "cka_layers",
            "router_kl_weight",
            "router_entropy_beta",
            "mc_expert_explore",
            "seed",
        )
    } == {
        "topk": None,
        "cka_enabled": False,
        "cka_layers": [],
        "router_kl_weight": 0.0,
        "router_entropy_beta": 0.0,
        "mc_expert_explore": False,
        "seed": 20260711,
    }
    loaded = load_authenticated_glm52_adapter(
        output_dir,
        expected_parent_candidate_identity_sha256="c" * 64,
        expected_teacher_manifest_body_sha256=row.teacher_manifest_body_sha256,
        expected_manifest_body_sha256=saved.manifest_body_sha256,
        expected_candidate_identity_sha256=saved.candidate_identity_sha256,
        expected_num_experts=2,
    )
    assert all(value.dtype == mx.float32 for value in loaded.params.values())

    fresh_switch = _switch()
    bind_glm52_low_rank_adapters(_projection_map(fresh_switch), loaded)
    fresh_logits = selected_glm52_logits_selected_layer(
        _TinyModel(fresh_switch),
        row,
        layer=0,
        surrogate_projections=frozenset(PROJECTIONS),
        output_chunk_size=4,
    )
    mx.eval(fresh_logits)
    assert bool(mx.allclose(fresh_logits, trained_logits, atol=1e-6, rtol=1e-6).item())

    peak = _peak_memory()
    print(f"SMOKE_PEAK_MEMORY_BYTES={peak}")
    assert peak < 4_000_000_000


def test_training_with_computed_route_indices_keeps_only_adapters_trainable():
    switch_mlp = _switch()
    model = _TinyModel(switch_mlp, computed_second_attention=True)
    projection_map = {(0, "gate_proj"): switch_mlp.gate_proj}
    config = AdapterTrainingConfig(
        layers=(0,),
        projections=("gate_proj",),
        rank=4,
        steps=2,
        learning_rate=0.05,
        low_rank_init_scale=0.1,
        seed=20260711,
        surrogate_output_chunk_size=4,
    )

    result = run_low_rank_training(
        model=model,
        rows=[_row()],
        projection_map=projection_map,
        config=config,
    )
    mx.eval(result.losses, *result.params.values(), *result.gradient_norms.values())

    assert set(result.params) == {
        "0.gate_proj.low_rank_left",
        "0.gate_proj.low_rank_right",
    }
    assert bool(mx.all(mx.isfinite(result.losses)).item())
    assert all(bool(mx.isfinite(value).item()) for value in result.gradient_norms.values())
    assert bool(
        mx.any(result.params["0.gate_proj.low_rank_left"] != 0).item()
    )


def test_new_distillation_features_disabled_preserve_training_and_sidecar_bytes(tmp_path):
    base_switch = _switch()
    explicit_switch = _switch()
    common = dict(
        layers=(0,),
        projections=("gate_proj",),
        rank=2,
        steps=2,
        learning_rate=0.05,
        low_rank_init_scale=0.1,
        seed=20260711,
        surrogate_output_chunk_size=4,
    )
    base_config = AdapterTrainingConfig(**common)
    explicit_off_config = AdapterTrainingConfig(
        **common,
        topk=None,
        cka_enabled=False,
        cka_layers=(),
        router_kl_weight=0.0,
        router_entropy_beta=0.0,
        mc_expert_explore=False,
    )
    base = run_low_rank_training(
        model=_TinyModel(base_switch),
        rows=[_row()],
        projection_map={(0, "gate_proj"): base_switch.gate_proj},
        config=base_config,
    )
    explicit = run_low_rank_training(
        model=_TinyModel(explicit_switch),
        rows=[_row()],
        projection_map={(0, "gate_proj"): explicit_switch.gate_proj},
        config=explicit_off_config,
    )
    mx.eval(
        base.losses,
        explicit.losses,
        *base.params.values(),
        *explicit.params.values(),
        *base.gradient_norms.values(),
        *explicit.gradient_norms.values(),
    )
    assert np.asarray(base.losses).tobytes() == np.asarray(explicit.losses).tobytes()
    for key in base.params:
        assert np.asarray(base.params[key]).tobytes() == np.asarray(explicit.params[key]).tobytes()
        assert np.asarray(base.gradient_norms[key]).tobytes() == np.asarray(
            explicit.gradient_norms[key]
        ).tobytes()

    for name, result, switch, config in (
        ("base", base, base_switch, base_config),
        ("explicit", explicit, explicit_switch, explicit_off_config),
    ):
        projection_map = {(0, "gate_proj"): switch.gate_proj}
        save_authenticated_glm52_adapter(
            tmp_path / name,
            params=result.params,
            projection_map=projection_map,
            parent_candidate_identity_sha256="c" * 64,
            teacher_manifest_body_sha256="a" * 64,
            training_config=config,
            selection_rows=[_row()],
            release_eligible=False,
            projection_inventory_attestation=attest_glm52_projection_inventory(projection_map),
        )
    assert (
        tmp_path / "base/sidecars/layer-00000-gate_proj.safetensors"
    ).read_bytes() == (
        tmp_path / "explicit/sidecars/layer-00000-gate_proj.safetensors"
    ).read_bytes()


def test_route_local_surrogate_matches_decoded_reference():
    projection = _projection(8, 8)
    x = mx.array([[0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2, 0.1]], dtype=mx.float32)
    indices = mx.array([[1]], dtype=mx.int32)
    expected = projection(x, indices)
    actual = RouteLocalSwitchLinearSurrogate.from_layer(projection, output_chunk_size=4)(x, indices)
    via_helper = route_local_projection(projection, x, indices, output_chunk_size=4)
    mx.eval(expected, actual, via_helper)
    assert bool(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
    assert bool(mx.allclose(via_helper, expected, atol=1e-5, rtol=1e-5).item())


def test_adapter_tamper_missing_pair_and_expert_mismatch_are_rejected(tmp_path):
    switch_mlp = _switch()
    config = AdapterTrainingConfig(layers=(0,), projections=("gate_proj",), rank=2, steps=1)
    params = {
        "0.gate_proj.low_rank_left": mx.zeros((2, 8, 2), dtype=mx.float32),
        "0.gate_proj.low_rank_right": mx.ones((2, 2, 8), dtype=mx.float32),
    }
    output_dir = tmp_path / "adapter"
    saved = save_authenticated_glm52_adapter(
        output_dir,
        params=params,
        projection_map={(0, "gate_proj"): switch_mlp.gate_proj},
        parent_candidate_identity_sha256="c" * 64,
        teacher_manifest_body_sha256="a" * 64,
        training_config=config,
        selection_rows=[_row()],
        release_eligible=False,
        projection_inventory_attestation=attest_glm52_projection_inventory(
            {(0, "gate_proj"): switch_mlp.gate_proj}
        ),
    )
    sidecar = next((output_dir / "sidecars").glob("*.safetensors"))
    raw = bytearray(sidecar.read_bytes())
    raw[-1] ^= 0x01
    sidecar.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256"):
        load_authenticated_glm52_adapter(
            output_dir,
            expected_parent_candidate_identity_sha256="c" * 64,
            expected_teacher_manifest_body_sha256="a" * 64,
            expected_manifest_body_sha256=saved.manifest_body_sha256,
            expected_candidate_identity_sha256=saved.candidate_identity_sha256,
            expected_num_experts=2,
        )

    with pytest.raises(ValueError, match="both low_rank_left and low_rank_right"):
        save_authenticated_glm52_adapter(
            tmp_path / "missing-pair",
            params={"0.gate_proj.low_rank_left": mx.zeros((2, 8, 2))},
            projection_map={(0, "gate_proj"): switch_mlp.gate_proj},
            parent_candidate_identity_sha256="c" * 64,
            teacher_manifest_body_sha256="a" * 64,
            training_config=config,
            selection_rows=[_row()],
            release_eligible=False,
            projection_inventory_attestation=attest_glm52_projection_inventory(
                {(0, "gate_proj"): switch_mlp.gate_proj}
            ),
        )

    switch_128 = _switch(experts=128)
    params_128 = {
        "0.gate_proj.low_rank_left": mx.zeros((128, 8, 2), dtype=mx.float32),
        "0.gate_proj.low_rank_right": mx.ones((128, 2, 8), dtype=mx.float32),
    }
    mismatch_dir = tmp_path / "mismatch"
    mismatch = save_authenticated_glm52_adapter(
        mismatch_dir,
        params=params_128,
        projection_map={(0, "gate_proj"): switch_128.gate_proj},
        parent_candidate_identity_sha256="d" * 64,
        teacher_manifest_body_sha256="a" * 64,
        training_config=config,
        selection_rows=[_row()],
        release_eligible=False,
        projection_inventory_attestation=attest_glm52_projection_inventory(
            {(0, "gate_proj"): switch_128.gate_proj}
        ),
    )
    with pytest.raises(ValueError, match="expert count 128 does not match expected 168"):
        load_authenticated_glm52_adapter(
            mismatch_dir,
            expected_parent_candidate_identity_sha256="d" * 64,
            expected_teacher_manifest_body_sha256="a" * 64,
            expected_manifest_body_sha256=mismatch.manifest_body_sha256,
            expected_candidate_identity_sha256=mismatch.candidate_identity_sha256,
            expected_num_experts=168,
        )


def test_adapter_sidecar_replacement_after_authentication_loads_original_bytes(
    tmp_path,
):
    switch_mlp = _switch()
    projection_map = {(0, "gate_proj"): switch_mlp.gate_proj}
    config = AdapterTrainingConfig(layers=(0,), projections=("gate_proj",), rank=2, steps=1)
    original_params = {
        "0.gate_proj.low_rank_left": mx.zeros((2, 8, 2), dtype=mx.float32),
        "0.gate_proj.low_rank_right": mx.ones((2, 2, 8), dtype=mx.float32),
    }
    output_dir = tmp_path / "adapter"
    saved = save_authenticated_glm52_adapter(
        output_dir,
        params=original_params,
        projection_map=projection_map,
        parent_candidate_identity_sha256="c" * 64,
        teacher_manifest_body_sha256="a" * 64,
        training_config=config,
        selection_rows=[_row()],
        release_eligible=False,
        projection_inventory_attestation=attest_glm52_projection_inventory(projection_map),
    )
    replacement = tmp_path / "replacement.safetensors"
    mx.save_safetensors(
        str(replacement),
        {
            "low_rank_left": mx.full((2, 8, 2), 9.0, dtype=mx.float32),
            "low_rank_right": mx.full((2, 2, 8), 7.0, dtype=mx.float32),
        },
        metadata={"format": "glm52_low_rank_adapter_v1"},
    )
    open_count = 0

    def swapping_opener(path, *, label):
        nonlocal open_count
        open_count += 1
        authenticated = AuthenticatedFile.open(path, label=label)
        os.replace(replacement, path)
        return authenticated

    loaded = load_authenticated_glm52_adapter(
        output_dir,
        expected_parent_candidate_identity_sha256="c" * 64,
        expected_teacher_manifest_body_sha256="a" * 64,
        expected_manifest_body_sha256=saved.manifest_body_sha256,
        expected_candidate_identity_sha256=saved.candidate_identity_sha256,
        expected_num_experts=2,
        _sidecar_opener=swapping_opener,
    )
    mx.eval(*loaded.params.values())
    assert open_count == 1
    assert bool(mx.all(loaded.params["0.gate_proj.low_rank_left"] == 0).item())
    assert bool(mx.all(loaded.params["0.gate_proj.low_rank_right"] == 1).item())


@pytest.mark.parametrize("split", ["report", "holdout"])
def test_non_selection_rows_are_rejected_for_tuning(split):
    with pytest.raises(ValueError, match="selection-only"):
        validate_tuning_row(_row(split=split, tuning_eligible=False))


def test_cli_requires_typed_validated_baseline_and_derives_identity_from_it():
    model = _TinyModel(_switch())
    args = SimpleNamespace()
    with pytest.raises(TypeError, match="ValidatedGLM52TrainingBaseline"):
        finetune_glm52._load_validated_baseline(lambda _args: (model, "c" * 64), args)

    baseline = ValidatedGLM52TrainingBaseline(
        model=model,
        candidate_identity_sha256="d" * 64,
    )
    loaded = finetune_glm52._load_validated_baseline(lambda _args: baseline, args)
    assert loaded is baseline
    assert loaded.candidate_identity_sha256 == "d" * 64


def test_save_attests_pretraining_projection_inventory_and_rejects_drift(tmp_path):
    before_switch = _switch(experts=2)
    before_map = {(0, "gate_proj"): before_switch.gate_proj}
    attestation = attest_glm52_projection_inventory(before_map)
    after_switch = _switch(experts=3)
    after_map = {(0, "gate_proj"): after_switch.gate_proj}
    config = AdapterTrainingConfig(layers=(0,), projections=("gate_proj",), rank=2, steps=1)
    params = {
        "0.gate_proj.low_rank_left": mx.zeros((3, 8, 2), dtype=mx.float32),
        "0.gate_proj.low_rank_right": mx.ones((3, 2, 8), dtype=mx.float32),
    }

    with pytest.raises(ValueError, match="projection inventory changed"):
        save_authenticated_glm52_adapter(
            tmp_path / "drifted",
            params=params,
            projection_map=after_map,
            parent_candidate_identity_sha256="c" * 64,
            teacher_manifest_body_sha256="a" * 64,
            training_config=config,
            selection_rows=[_row()],
            release_eligible=False,
            projection_inventory_attestation=attestation,
        )


def test_teacher_cache_shaped_selection_loader_authenticates_full_logits(tmp_path):
    cache_dir = tmp_path / "cache"
    logits_dir = cache_dir / "teacher_logits"
    logits_dir.mkdir(parents=True)
    prompt_pack = tmp_path / "prompt-pack.json"
    prompt_payload = {
        "prompt_rows": [
            {
                "prompt_id": "synthetic-000",
                "split": "selection",
                "tuning_eligible": True,
                "encoded_token_ids": [1, 2, 3],
            },
            {
                "prompt_id": "synthetic-report",
                "split": "report",
                "tuning_eligible": False,
                "encoded_token_ids": [1, 4],
            },
        ]
    }
    prompt_pack.write_text(json.dumps(prompt_payload, sort_keys=True), encoding="utf-8")
    shard = logits_dir / "synthetic-000.safetensors"
    mx.save_safetensors(str(shard), {"logits": _row().teacher_logits})
    shard_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_teacher_cache",
        "release_eligible": False,
        "prompt_pack_sha256": hashlib.sha256(prompt_pack.read_bytes()).hexdigest(),
        "shards": [
            {
                "prompt_id": "synthetic-000",
                "split": "selection",
                "tuning_eligible": True,
                "relative_path": "teacher_logits/synthetic-000.safetensors",
                "tensor_name": "logits",
                "dtype": "F32",
                "shape": [2, 7],
                "file_sha256": shard_sha,
            }
        ],
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_body_sha256"] = hashlib.sha256(body).hexdigest()
    (cache_dir / "glm52-teacher-cache-fp32-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    rows = prepare_glm52_teacher_rows(
        teacher_cache_dir=cache_dir,
        prompt_pack_path=prompt_pack,
        split="selection",
        max_rows=None,
        max_positions=None,
        test_only_synthetic=True,
        allow_non_release_teacher_cache=True,
    )
    assert len(rows) == 1
    assert rows[0].positions == (0, 1)
    assert rows[0].target_token_ids == (2, 3)
    assert rows[0].teacher_logits.shape == (2, 7)

    with pytest.raises(ValueError, match="selection-only"):
        prepare_glm52_teacher_rows(
            teacher_cache_dir=cache_dir,
            prompt_pack_path=prompt_pack,
            split="report",  # type: ignore[arg-type]
            max_rows=None,
            max_positions=None,
            test_only_synthetic=True,
            allow_non_release_teacher_cache=True,
        )


def test_teacher_cache_production_load_cannot_downgrade_to_fixture_schema(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prompt_pack = tmp_path / "prompt-pack.json"
    prompt_pack.write_text(json.dumps({"prompt_rows": []}), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_teacher_cache",
        "release_eligible": True,
        "producer": {},
        "source_evidence": {},
        "non_vq_package": {},
        # Deliberately omit prompt_authority: this used to select fixture checks.
        "shards": [],
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(body).hexdigest()
    manifest["manifest_body_sha256"] = identity
    (cache_dir / "glm52-teacher-cache-fp32-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="production schema"):
        prepare_glm52_teacher_rows(
            teacher_cache_dir=cache_dir,
            prompt_pack_path=prompt_pack,
            split="selection",
            max_rows=None,
            max_positions=None,
            expected_manifest_body_sha256=identity,
        )


def test_synthetic_teacher_cache_mode_forces_non_release(tmp_path):
    cache_dir = tmp_path / "cache"
    logits_dir = cache_dir / "teacher_logits"
    logits_dir.mkdir(parents=True)
    prompt_pack = tmp_path / "prompt-pack.json"
    prompt_pack.write_text(
        json.dumps(
            {
                "prompt_rows": [
                    {
                        "prompt_id": "synthetic-000",
                        "split": "selection",
                        "tuning_eligible": True,
                        "encoded_token_ids": [1, 2, 3],
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    shard = logits_dir / "synthetic-000.safetensors"
    mx.save_safetensors(str(shard), {"logits": _row().teacher_logits})
    manifest = {
        "schema_version": 1,
        "record_type": "glm52_teacher_cache",
        "release_eligible": True,
        "prompt_pack_sha256": hashlib.sha256(prompt_pack.read_bytes()).hexdigest(),
        "shards": [
            {
                "prompt_id": "synthetic-000",
                "split": "selection",
                "tuning_eligible": True,
                "relative_path": "teacher_logits/synthetic-000.safetensors",
                "tensor_name": "logits",
                "dtype": "F32",
                "shape": [2, 7],
                "file_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_body_sha256"] = hashlib.sha256(body).hexdigest()
    (cache_dir / "glm52-teacher-cache-fp32-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    rows = prepare_glm52_teacher_rows(
        teacher_cache_dir=cache_dir,
        prompt_pack_path=prompt_pack,
        split="selection",
        max_rows=None,
        max_positions=None,
        test_only_synthetic=True,
        allow_non_release_teacher_cache=True,
    )
    assert rows[0].non_release_waiver is True


def test_prepare_rows_consumes_rich_topk_cka_router_teacher_contract(tmp_path):
    cache_dir = tmp_path / "teacher-cache"
    signals_dir = cache_dir / "teacher_signals"
    signals_dir.mkdir(parents=True)
    prompt_pack = tmp_path / "prompt-pack.json"
    prompt_pack.write_text(
        json.dumps(
            {
                "prompt_rows": [
                    {
                        "prompt_id": "synthetic-rich",
                        "split": "selection",
                        "tuning_eligible": True,
                        "encoded_token_ids": [1, 2, 3],
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    shard = signals_dir / "synthetic-rich.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            "topk_ids": mx.array([[2, 1], [3, 0]], dtype=mx.int32),
            "topk_values": mx.array([[1.5, 0.2], [1.0, 0.3]], dtype=mx.float16),
            "tail_mass": mx.array([0.1, 0.2], dtype=mx.float16),
            "cka.0": mx.arange(16, dtype=mx.float16).reshape((2, 8)),
            "router_ids.0": mx.array([[0, 1], [1, 0]], dtype=mx.int32),
            "router_weights.0": mx.array(
                [[0.75, 0.25], [0.6, 0.4]], dtype=mx.float16
            ),
        },
    )
    manifest = {
        "schema_version": 2,
        "record_type": "glm52_teacher_signal_cache",
        "release_eligible": False,
        "prompt_pack_sha256": hashlib.sha256(prompt_pack.read_bytes()).hexdigest(),
        "shards": [
            {
                "prompt_id": "synthetic-rich",
                "split": "selection",
                "tuning_eligible": True,
                "relative_path": "teacher_signals/synthetic-rich.safetensors",
                "file_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "topk": {
                    "k": 2,
                    "vocab_size": 7,
                    "ids_tensor": "topk_ids",
                    "values_tensor": "topk_values",
                    "tail_mass_tensor": "tail_mass",
                },
                "cka_probes": [
                    {
                        "layer": 0,
                        "tensor_name": "cka.0",
                        "dtype": "F16",
                        "shape": [2, 8],
                    }
                ],
                "router_targets": [
                    {
                        "layer": 0,
                        "ids_tensor": "router_ids.0",
                        "weights_tensor": "router_weights.0",
                        "weights_dtype": "F16",
                    }
                ],
            }
        ],
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_body_sha256"] = hashlib.sha256(body).hexdigest()
    (cache_dir / "glm52-teacher-cache-fp32-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    rows = prepare_glm52_teacher_rows(
        teacher_cache_dir=cache_dir,
        prompt_pack_path=prompt_pack,
        split="selection",
        max_rows=None,
        max_positions=None,
        allow_non_release_teacher_cache=True,
        test_only_synthetic=True,
    )
    row = rows[0]
    assert row.teacher_logits is None
    assert row.topk_logit_ids.shape == (2, 2)
    assert row.topk_logit_values.dtype == mx.float16
    assert row.tail_mass.dtype == mx.float16
    assert set(row.teacher_probe_hidden_states) == {0}
    assert set(row.teacher_router_topk_ids) == {0}
    assert set(row.teacher_router_topk_weights) == {0}
