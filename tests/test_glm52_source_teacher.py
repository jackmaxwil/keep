from __future__ import annotations

import importlib
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "ramp" / "models" / "glm52_source_teacher.py"
)


def _api() -> Any:
    assert MODULE_PATH.is_file(), "the GLM52 source-teacher forward runner is missing"
    return importlib.import_module("ramp.models.glm52_source_teacher")


def _expert_weights() -> dict[int, dict[str, np.ndarray]]:
    rng = np.random.default_rng(5202)
    return {
        expert: {
            "gate_proj": np.ascontiguousarray(
                rng.normal(scale=0.3, size=(6, 8)).astype(np.float32)
            ),
            "up_proj": np.ascontiguousarray(
                rng.normal(scale=0.2, size=(6, 8)).astype(np.float32)
            ),
            "down_proj": np.ascontiguousarray(
                rng.normal(scale=0.25, size=(8, 6)).astype(np.float32)
            ),
        }
        for expert in range(4)
    }


def _layer_expert_weights() -> dict[int, dict[int, dict[str, np.ndarray]]]:
    return {0: _expert_weights(), 1: _expert_weights()}


def _tiny_model() -> Any:
    import mlx.core as mx
    from mlx.utils import tree_map_with_path

    from ramp.models.glm52_vq_adapter import (
        GLM52VQModel,
        glm52_vq_args_from_config,
    )

    config = {
        "model_type": "glm_moe_dsa",
        "vocab_size": 16,
        "hidden_size": 8,
        "index_head_dim": 2,
        "index_n_heads": 2,
        "index_topk": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "routed_scaling_factor": 1.0,
        "kv_lora_rank": 2,
        "q_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "qk_nope_head_dim": 2,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 0,
        "max_position_embeddings": 16,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False,
        "indexer_types": ["full", "full"],
        "index_topk_pattern": None,
        "index_topk_freq": 1,
        "index_skip_topk_offset": 0,
        "mlp_layer_types": ["sparse", "sparse"],
        "rope_interleave": True,
        "indexer_rope_interleave": True,
    }
    mx.random.seed(5202)
    model = GLM52VQModel(glm52_vq_args_from_config(config))

    def cast_parameter(path: str, value: Any) -> Any:
        if model.cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(cast_parameter, model.parameters()))
    mx.eval(model.parameters())
    return model


def _resolver_factory(
    weights: dict[int, dict[int, dict[str, np.ndarray]]],
) -> Any:
    return lambda layer_index: lambda expert_index: weights[layer_index][expert_index]


def _identities(api: Any) -> Any:
    return api.CheckpointIdentities(
        prompt_sha256="1" * 64,
        source_sha256="2" * 64,
        profile_sha256="3" * 64,
        config_sha256="4" * 64,
        policy_sha256="5" * 64,
        precision_sha256="6" * 64,
    )


def _dense_expert(hidden: Any, weights: dict[str, np.ndarray]) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    gate = hidden @ mx.array(weights["gate_proj"]).astype(mx.bfloat16).T
    up = hidden @ mx.array(weights["up_proj"]).astype(mx.bfloat16).T
    activated = nn.silu(gate) * up
    return activated @ mx.array(weights["down_proj"]).astype(mx.bfloat16).T


def _bf16_bytes(value: Any) -> bytes:
    import mlx.core as mx

    return np.array(value.view(mx.uint16)).tobytes()


def test_selected_expert_streaming_matches_dense_reference_bit_for_bit() -> None:
    api = _api()
    import mlx.core as mx
    weights = _expert_weights()
    hidden = mx.array(
        np.arange(5 * 8, dtype=np.float32).reshape(5, 8) / np.float32(19.0)
    ).astype(mx.bfloat16)
    expected_indices = mx.array(
        [[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]],
        dtype=mx.int32,
    )
    expected_scores = mx.array(
        [[0.75, 0.25], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2], [0.51, 0.49]],
        dtype=mx.float32,
    )
    resolved: list[int] = []

    def gate(_: mx.array) -> tuple[mx.array, mx.array]:
        return expected_indices, expected_scores

    def resolver(expert: int) -> dict[str, np.ndarray]:
        resolved.append(expert)
        return weights[expert]

    shared = lambda x: mx.full(x.shape, 0.125, dtype=mx.bfloat16)
    actual = api.stream_selected_expert_moe(
        hidden,
        gate=gate,
        expert_weight_resolver=resolver,
        shared_expert=shared,
    )

    all_experts = mx.stack(
        [_dense_expert(hidden, weights[expert]) for expert in range(4)],
        axis=1,
    )
    selected = mx.take_along_axis(
        all_experts,
        mx.broadcast_to(
            expected_indices[..., None],
            expected_indices.shape + (hidden.shape[-1],),
        ),
        axis=1,
    )
    expected = (
        (selected * expected_scores[..., None]).sum(axis=-2).astype(mx.bfloat16)
        + shared(hidden)
    )
    mx.eval(actual.output, expected, actual.expert_indices, actual.route_scores)

    assert _bf16_bytes(actual.output) == _bf16_bytes(expected)
    np.testing.assert_array_equal(
        np.array(actual.expert_indices), np.array(expected_indices)
    )
    np.testing.assert_array_equal(
        np.array(actual.route_scores), np.array(expected_scores)
    )
    assert sorted(resolved) == [0, 1, 2]
    assert actual.valid_assignment_count == 10
    assert actual.padded_assignment_count == 0


def test_decode_pipeline_preserves_sorted_expert_consumption_and_bounds_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    import mlx.core as mx

    weights = _expert_weights()
    hidden = mx.ones((4, 8), dtype=mx.bfloat16)
    indices = mx.array([[3], [0], [2], [1]], dtype=mx.int32)
    scores = mx.ones((4, 1), dtype=mx.float32)
    completed: list[int] = []

    def resolver(expert: int) -> dict[str, Any]:
        def projection(name: str) -> Any:
            def decode() -> np.ndarray:
                time.sleep(0.01)
                if name == "down_proj":
                    completed.append(expert)
                return weights[expert][name]

            return decode

        return {
            name: projection(name)
            for name in ("gate_proj", "up_proj", "down_proj")
        }

    monkeypatch.setattr(api, "_EXPERT_DECODE_PIPELINE_ENABLED", True)
    streamed = api.stream_selected_expert_moe(
        hidden,
        gate=lambda _: (indices, scores),
        expert_weight_resolver=resolver,
        num_routed_experts=4,
    )
    mx.eval(streamed.output)

    assert sorted(completed) == [0, 1, 2, 3]
    assert streamed.decode_workers == 2
    assert streamed.decode_queue_depth == 2
    assert 2 <= streamed.observed_peak_decoded_experts <= 3


def test_pipeline_on_is_byte_identical_to_pipeline_off_for_tiny_teacher_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    cache_api = importlib.import_module("keep.quality.glm52_teacher_cache")
    prompts = ([1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12])
    prompt_rows = tuple(
        cache_api.GLM52TeacherCachePrompt(
            prompt_id=f"tiny-{index:03d}",
            split="report",
            domain="synthetic",
            tuning_eligible=False,
            encoded_token_ids=tuple(token_ids),
        )
        for index, token_ids in enumerate(prompts)
    )

    def run(enabled: bool, root: Path) -> tuple[tuple[bytes, ...], dict[str, bytes]]:
        monkeypatch.setattr(api, "_EXPERT_DECODE_PIPELINE_ENABLED", enabled)
        logits, _ = api.run_glm52_source_teacher(
            _tiny_model(),
            prompts,
            expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
            expected_num_layers=2,
            checkpoint_dir=root / "checkpoints",
            checkpoint_identities=_identities(api),
        )
        cache_root = root / "cache"
        ledger_path = root / "shard-ledger.jsonl"
        for prompt, value in zip(prompt_rows, logits, strict=True):
            cache_api.write_glm52_teacher_cache_shard(
                cache_root,
                ledger_path=ledger_path,
                prompt=prompt,
                logits=value,
                vocab_size=16,
                producer_phase_ids=("tiny-byte-identity",),
                bound_identity_sha256="7" * 64,
            )
        durable = {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.name.endswith(".lock")
        }
        return tuple(value.tobytes() for value in logits), durable

    off_logits, off_files = run(False, tmp_path / "off")
    on_logits, on_files = run(True, tmp_path / "on")

    assert on_logits == off_logits
    assert on_files == off_files


def test_bf16_projection_materializes_weight_before_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    import mlx.core as mx

    hidden = mx.ones((2, 8), dtype=mx.bfloat16)
    decoded = np.ascontiguousarray(
        np.arange(6 * 8, dtype=np.float32).reshape(6, 8) / np.float32(31.0)
    )
    evaluated_shapes: list[tuple[int, ...]] = []
    original_eval = mx.eval

    def recording_eval(*values: Any) -> None:
        evaluated_shapes.extend(
            tuple(int(size) for size in value.shape) for value in values
        )
        original_eval(*values)

    monkeypatch.setattr(api.mx, "eval", recording_eval)
    output = api._bf16_projection(
        hidden,
        decoded,
        projection="gate_proj",
        decoded_dtypes=[],
        matmul_dtypes=[],
    )

    assert output.shape == (2, 6)
    assert evaluated_shapes == [(6, 8), (2, 6)]


def test_multi_expert_streaming_peak_memory_is_near_one_expert_working_set() -> None:
    api = _api()
    import mlx.core as mx

    get_active_memory = getattr(mx, "get_active_memory", None)
    get_peak_memory = getattr(mx, "get_peak_memory", None)
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if not all((get_active_memory, get_peak_memory, reset_peak_memory)):
        pytest.skip("MLX peak-memory counters are unavailable")

    expert_count = 12
    hidden_size = 512
    intermediate_size = 256
    rng = np.random.default_rng(5217)
    weights = {
        expert: {
            "gate_proj": np.ascontiguousarray(
                rng.normal(scale=0.02, size=(intermediate_size, hidden_size)).astype(
                    np.float32
                )
            ),
            "up_proj": np.ascontiguousarray(
                rng.normal(scale=0.02, size=(intermediate_size, hidden_size)).astype(
                    np.float32
                )
            ),
            "down_proj": np.ascontiguousarray(
                rng.normal(scale=0.02, size=(hidden_size, intermediate_size)).astype(
                    np.float32
                )
            ),
        }
        for expert in range(expert_count)
    }
    hidden = mx.array(
        rng.normal(size=(expert_count, hidden_size)).astype(np.float32)
    ).astype(mx.bfloat16)
    indices = mx.arange(expert_count, dtype=mx.int32)[:, None]
    scores = mx.ones((expert_count, 1), dtype=mx.float32)
    mx.eval(hidden, indices, scores)
    mx.clear_cache()
    baseline_active = int(get_active_memory())
    reset_peak_memory()

    streamed = api.stream_selected_expert_moe(
        hidden,
        gate=lambda _: (indices, scores),
        expert_weight_resolver=lambda expert: weights[expert],
        num_routed_experts=expert_count,
    )
    mx.eval(streamed.output)
    peak_delta = max(0, int(get_peak_memory()) - baseline_active)

    one_expert_source_bytes = sum(value.nbytes for value in weights[0].values())
    assert peak_delta <= 4 * one_expert_source_bytes


def test_batched_right_padded_forward_matches_individual_prompts() -> None:
    api = _api()
    model = _tiny_model()
    weights = _layer_expert_weights()
    prompts = ([1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12])

    batched_logits, route_trace = api.run_glm52_source_teacher(
        model,
        prompts,
        expert_resolver_for_layer=_resolver_factory(weights),
        expected_num_layers=2,
        capture_route_trace=False,
    )
    individual_logits = []
    for prompt in prompts:
        logits, _ = api.run_glm52_source_teacher(
            model,
            [prompt],
            expert_resolver_for_layer=_resolver_factory(weights),
            expected_num_layers=2,
            capture_route_trace=False,
        )
        individual_logits.append(logits[0])

    assert route_trace is None
    assert [row.shape for row in batched_logits] == [(4, 16), (2, 16), (3, 16)]
    for batched, individual in zip(batched_logits, individual_logits, strict=True):
        np.testing.assert_array_equal(batched, individual)


def test_causal_mask_receives_per_row_pad_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()

    observed: list[np.ndarray] = []
    original = api.create_causal_mask

    def recording_mask(length: int, **kwargs: Any) -> Any:
        observed.append(np.array(kwargs["right_padding"]))
        return original(length, **kwargs)

    monkeypatch.setattr(api, "create_causal_mask", recording_mask)
    api.run_glm52_source_teacher(
        _tiny_model(),
        ([1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]),
        expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
        expected_num_layers=2,
    )

    np.testing.assert_array_equal(observed, [np.array([0, 2, 1], dtype=np.int32)])


def test_padded_rows_never_enter_router_and_assignment_count_is_exact() -> None:
    api = _api()
    model = _tiny_model()
    weights = _layer_expert_weights()
    router_token_counts: list[int] = []

    for layer in model.layers:
        original_gate = layer.mlp.gate

        def recording_gate(hidden: Any, *, original: Any = original_gate) -> Any:
            router_token_counts.append(int(hidden.shape[0]))
            return original(hidden)

        layer.mlp.gate = recording_gate

    logits, trace = api.run_glm52_source_teacher(
        model,
        ([1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]),
        expert_resolver_for_layer=_resolver_factory(weights),
        expected_num_layers=2,
        capture_route_trace=True,
    )

    assert len(logits) == 3
    assert router_token_counts == [9, 9]
    assert trace is not None
    assert [layer.valid_assignment_count for layer in trace] == [18, 18]
    assert [layer.padded_assignment_count for layer in trace] == [0, 0]
    assert all(layer.expert_ids.shape == (9, 2) for layer in trace)
    assert all(layer.scores.shape == (9, 2) for layer in trace)


def test_precision_contract_is_explicit_at_every_source_teacher_boundary() -> None:
    api = _api()
    import mlx.core as mx

    weights = _expert_weights()
    hidden = mx.ones((3, 8), dtype=mx.bfloat16)
    indices = mx.array([[0, 1], [1, 2], [2, 0]], dtype=mx.int32)
    scores = mx.full((3, 2), 0.5, dtype=mx.float32)
    streamed = api.stream_selected_expert_moe(
        hidden,
        gate=lambda _: (indices, scores),
        expert_weight_resolver=lambda expert: weights[expert],
    )
    logits, trace = api.run_glm52_source_teacher(
        _tiny_model(),
        ([1, 2, 3], [4, 5]),
        expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
        expected_num_layers=2,
        capture_route_trace=True,
    )

    assert streamed.precision.hidden_dtype == "mlx.core.bfloat16"
    assert streamed.precision.route_output_dtype == "mlx.core.bfloat16"
    assert streamed.precision.route_score_dtype == "mlx.core.float32"
    assert set(streamed.precision.decoded_weight_dtypes) == {"float32"}
    assert set(streamed.precision.matmul_weight_dtypes) == {"mlx.core.bfloat16"}
    assert all(row.dtype == np.float32 and row.flags.c_contiguous for row in logits)
    assert trace is not None
    assert all(layer.scores.dtype == np.float32 for layer in trace)


def test_streaming_rejects_non_bf16_activated_gate_up_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    import mlx.core as mx

    hidden = mx.ones((2, 8), dtype=mx.bfloat16)
    indices = mx.zeros((2, 1), dtype=mx.int32)
    scores = mx.ones((2, 1), dtype=mx.float32)
    decoded: list[str] = []
    weights = _expert_weights()[0]

    def resolver(_expert: int) -> dict[str, Any]:
        return {
            projection: (
                lambda projection=projection: (
                    decoded.append(projection) or weights[projection]
                )
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
        }

    monkeypatch.setattr(
        api.nn,
        "silu",
        lambda value: value.astype(mx.float32),
    )

    with pytest.raises(
        ValueError,
        match="activated gate/up product must use bfloat16",
    ):
        api.stream_selected_expert_moe(
            hidden,
            gate=lambda _: (indices, scores),
            expert_weight_resolver=resolver,
        )

    assert decoded == ["gate_proj", "up_proj"]
    assert not any(
        thread.name.startswith("glm52-expert-decode")
        for thread in threading.enumerate()
    )


def test_forward_rejects_fp32_lm_head_before_expert_decode() -> None:
    api = _api()
    import mlx.core as mx

    model = _tiny_model()
    model.lm_head.weight = model.lm_head.weight.astype(mx.float32)
    resolved: list[int] = []

    def resolver_for_layer(_layer: int) -> Any:
        def resolve(expert: int) -> dict[str, np.ndarray]:
            resolved.append(expert)
            return _expert_weights()[expert]

        return resolve

    with pytest.raises(
        ValueError,
        match=r"lm_head\.weight=mlx\.core\.float32, expected=mlx\.core\.bfloat16",
    ):
        api.run_glm52_source_teacher(
            model,
            ([1, 2, 3],),
            expert_resolver_for_layer=resolver_for_layer,
            expected_num_layers=2,
        )

    assert resolved == []


def test_forward_rejects_mtp_bearing_79_layer_surface_before_compute() -> None:
    api = _api()
    model = _tiny_model()
    model.args.num_hidden_layers = 79
    model.model.layers.extend([model.model.layers[-1]] * 77)
    model.model.end_idx = 79
    compute_calls: list[str] = []

    def forbidden_embed(_tokens: Any) -> Any:
        compute_calls.append("embed")
        raise AssertionError("embedding must not run for a 79-layer surface")

    def forbidden_resolver(_layer: int) -> Any:
        compute_calls.append("expert_decode")
        raise AssertionError("expert decode must not run for a 79-layer surface")

    model.model.embed_tokens = forbidden_embed

    with pytest.raises(
        ValueError,
        match=r"num_hidden_layers=79, expected_num_layers=78",
    ):
        api.run_glm52_source_teacher(
            model,
            ([1, 2, 3],),
            expert_resolver_for_layer=forbidden_resolver,
            expected_num_layers=78,
        )

    assert compute_calls == []


def test_interrupted_checkpoint_resume_is_byte_identical(tmp_path: Path) -> None:
    api = _api()
    model = _tiny_model()
    weights = _layer_expert_weights()
    prompts = ([1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12])

    uninterrupted, _ = api.run_glm52_source_teacher(
        model,
        prompts,
        expert_resolver_for_layer=_resolver_factory(weights),
        expected_num_layers=2,
        checkpoint_dir=tmp_path / "uninterrupted" / "checkpoints",
        checkpoint_identities=_identities(api),
    )
    resumed_dir = tmp_path / "resumed" / "checkpoints"
    with pytest.raises(api.SourceTeacherInterrupted, match="layer 0"):
        api.run_glm52_source_teacher(
            model,
            prompts,
            expert_resolver_for_layer=_resolver_factory(weights),
            expected_num_layers=2,
            checkpoint_dir=resumed_dir,
            checkpoint_identities=_identities(api),
            interrupt_after_layer=0,
        )
    resumed, _ = api.run_glm52_source_teacher(
        model,
        prompts,
        expert_resolver_for_layer=_resolver_factory(weights),
        expected_num_layers=2,
        checkpoint_dir=resumed_dir,
        checkpoint_identities=_identities(api),
    )

    assert sorted(path.name for path in resumed_dir.iterdir()) == [
        "checkpoint-ledger.jsonl",
        "layer-00000.safetensors",
        "layer-00001.safetensors",
    ]
    for expected, actual in zip(uninterrupted, resumed, strict=True):
        assert expected.tobytes() == actual.tobytes()


def test_orphan_checkpoint_temp_is_non_authoritative(tmp_path: Path) -> None:
    api = _api()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    orphan = checkpoint_dir / ".layer-00000.orphan.tmp.safetensors"
    orphan.write_bytes(b"incomplete")

    logits, _ = api.run_glm52_source_teacher(
        _tiny_model(),
        ([1, 2, 3], [4, 5]),
        expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
        expected_num_layers=2,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identities=_identities(api),
    )

    assert len(logits) == 2
    assert orphan.read_bytes() == b"incomplete"
    assert (checkpoint_dir / "layer-00000.safetensors").is_file()
    assert (checkpoint_dir / "layer-00001.safetensors").is_file()


def test_tampered_final_checkpoint_fails_loudly(tmp_path: Path) -> None:
    api = _api()
    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(api.SourceTeacherInterrupted):
        api.run_glm52_source_teacher(
            _tiny_model(),
            ([1, 2, 3], [4, 5]),
            expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
            expected_num_layers=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_identities=_identities(api),
            interrupt_after_layer=0,
        )
    checkpoint = checkpoint_dir / "layer-00000.safetensors"
    raw = bytearray(checkpoint.read_bytes())
    raw[-1] ^= 0x01
    checkpoint.write_bytes(raw)

    with pytest.raises(api.CheckpointValidationError, match="hidden SHA-256"):
        api.run_glm52_source_teacher(
            _tiny_model(),
            ([1, 2, 3], [4, 5]),
            expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
            expected_num_layers=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_identities=_identities(api),
        )


def test_resume_rejects_checkpoint_replaced_between_validation_and_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(api.SourceTeacherInterrupted, match="layer 0"):
        api.run_glm52_source_teacher(
            _tiny_model(),
            ([1, 2, 3], [4, 5]),
            expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
            expected_num_layers=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_identities=_identities(api),
            interrupt_after_layer=0,
        )

    original_reopen = api.read_safetensors_tensor_mlx
    reopened: list[str] = []

    def replace_then_reopen(path: Path, tensor_name: str) -> Any:
        reopened.append(Path(path).name)
        replacement = bytearray(Path(path).read_bytes())
        replacement[-1] ^= 0x01
        Path(path).write_bytes(replacement)
        return original_reopen(path, tensor_name)

    monkeypatch.setattr(api, "read_safetensors_tensor_mlx", replace_then_reopen)

    with pytest.raises(
        api.CheckpointValidationError,
        match="changed between validation and load",
    ):
        api.run_glm52_source_teacher(
            _tiny_model(),
            ([1, 2, 3], [4, 5]),
            expert_resolver_for_layer=_resolver_factory(_layer_expert_weights()),
            expected_num_layers=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_identities=_identities(api),
        )

    assert reopened == ["layer-00000.safetensors"]


def test_route_rank_reduction_differs_from_expert_major_on_adversarial_values() -> None:
    api = _api()
    import mlx.core as mx

    route_outputs = mx.array([[[1e20], [1.0], [-1e20], [1.0]]]).astype(
        mx.bfloat16
    )
    route_scores = mx.ones((1, 4), dtype=mx.float32)
    route_rank = api.reduce_route_outputs_route_rank(route_outputs, route_scores)
    expert_order = mx.array([0, 2, 1, 3], dtype=mx.int32)
    expert_major = api.reduce_route_outputs_route_rank(
        mx.take(route_outputs, expert_order, axis=1),
        mx.take(route_scores, expert_order, axis=1),
    )
    mx.eval(route_rank, expert_major)

    assert _bf16_bytes(route_rank) != _bf16_bytes(expert_major)


def test_modelopt_resolver_is_lazy_per_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _api()
    weight_map: dict[str, str] = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.1.mlp.experts.2.{projection}"
        weight_map[f"{base}.weight"] = "weights.safetensors"
        weight_map[f"{base}.weight_scale"] = "scales.safetensors"
        weight_map[f"{base}.weight_scale_2"] = "globals.safetensors"
    decoded: list[str] = []

    def fake_read(_root: Path, bundle: Any) -> np.ndarray:
        decoded.append(bundle.weight_name)
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(api, "read_modelopt_nvfp4_weight", fake_read)
    resolver = api.make_modelopt_nvfp4_expert_weight_resolver(
        "/source",
        weight_map,
        layer_index=1,
    )
    projections = resolver(2)

    assert decoded == []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        value = projections[projection]
        assert callable(value)
        assert value().dtype == np.float32
        assert decoded[-1].endswith(f".{projection}.weight")


def test_scatter_valid_rows_handles_row_counts_past_int32_flatten_limit() -> None:
    # Regression test for a real crash: put_along_axis(..., axis=None) needs
    # to flatten the destination to a single 1-D view, and MLX's internal
    # flatten uses a signed 32-bit size -- it silently wraps for arrays with
    # more than ~2.15B total elements (measured: a 65,487-token real session,
    # 8 experts-per-token, hidden_size 6144 hit this in
    # stream_selected_expert_moe's identical scatter pattern, crashing with
    # "Cannot reshape array of size 3218767872 into shape (-1076199424)" --
    # exactly total_size - 2**32). hidden_size is kept tiny here so only the
    # row count needs to cross the threshold, keeping the test's memory
    # footprint to ~4.3GB (unavoidable: proving the fix requires actually
    # allocating a buffer past the limit) instead of scaling with hidden_size
    # too.
    import mlx.core as mx

    api = _api()
    hidden_size = 8
    total_rows = (2**31 // hidden_size) + 1024  # just past the flatten limit
    assert total_rows * hidden_size > 2**31

    valid_flat_indices = np.array([0, 5, total_rows - 1], dtype=np.int64)
    valid_output = mx.arange(len(valid_flat_indices) * hidden_size, dtype=mx.float32)
    valid_output = valid_output.reshape(len(valid_flat_indices), hidden_size).astype(
        mx.bfloat16
    )

    result = api._scatter_valid_rows(
        valid_output,
        valid_flat_indices=valid_flat_indices,
        batch_size=1,
        sequence_length=total_rows,
        hidden_size=hidden_size,
    )
    mx.eval(result)

    assert result.shape == (1, total_rows, hidden_size)
    flat = result.reshape(total_rows, hidden_size)
    for row_position, source_row in enumerate(valid_flat_indices.tolist()):
        got = np.array(flat[source_row].astype(mx.float32))
        want = np.array(valid_output[row_position].astype(mx.float32))
        np.testing.assert_array_equal(got, want)
    # a row never listed in valid_flat_indices must remain zero
    untouched = np.array(flat[total_rows // 2].astype(mx.float32))
    assert np.all(untouched == 0)
