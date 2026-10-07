import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step

from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from mlx_vq.models.glm52_vq_adapter import (
    GLM52VQModel,
    GLM52VQModelArgs,
    Glm52VQMoE,
)


PROBE_RECORD_TYPE = "glm52_synthetic_generation_probe"
PROBE_READY_STATUS = "glm52_synthetic_generation_probe_ready"
PROBE_FAILED_STATUS = "glm52_synthetic_generation_probe_failed"
WHOLE_MODEL_SCOPE = "tiny_fixture"
PROMPT_TOKEN_IDS = (1, 2, 3)
REQUESTED_GENERATED_TOKENS = 2
MODEL_SEED = 5203


class GLM52SyntheticGenerationProbeError(ValueError):
    """A staged failure from the bounded synthetic generation probe."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


def _base_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": PROBE_RECORD_TYPE,
        "probe_status": PROBE_FAILED_STATUS,
        "probe_pass": False,
        "whole_model_scope": WHOLE_MODEL_SCOPE,
        "synthetic_runtime_contract": False,
        "production_generation_proven": False,
        "production_artifact_used": False,
        "tokenizer_used": False,
        "model_seed": MODEL_SEED,
        "prompt_token_ids": list(PROMPT_TOKEN_IDS),
        "prompt_token_count": len(PROMPT_TOKEN_IDS),
        "requested_generated_tokens": REQUESTED_GENERATED_TOKENS,
        "greedy_sampling": True,
        "explicit_prompt_cache": True,
        "sparse_layer_count": 0,
        "bound_sparse_layer_count": 0,
        "unbound_sparse_layer_count": 0,
        "dense_routed_weight_present": False,
        "direct_forward": {},
        "generate_step": {},
        "repeatability": {},
    }


def _tiny_args() -> GLM52VQModelArgs:
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
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=["full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )


def _in_memory_switch() -> QuantizedVQSwitchGLU:
    return QuantizedVQSwitchGLU.from_weights(
        gate_weight=mx.full((2, 8, 16), 0.03125, dtype=mx.float32),
        up_weight=mx.full((2, 8, 16), -0.015625, dtype=mx.float32),
        down_weight=mx.full((2, 16, 8), 0.0078125, dtype=mx.float32),
        group_size=8,
    )


def _build_model() -> tuple[GLM52VQModel, dict[str, int | bool]]:
    mx.random.seed(MODEL_SEED)
    model = GLM52VQModel(_tiny_args())
    sparse_layers = [
        layer for layer in model.layers if isinstance(layer.mlp, Glm52VQMoE)
    ]
    for layer in sparse_layers:
        layer.mlp.bind_switch_mlp(_in_memory_switch())

    bound_sparse_layers = sum(
        layer.mlp.switch_mlp is not None for layer in sparse_layers
    )
    parameter_names = {name for name, _value in tree_flatten(model.parameters())}
    dense_routed_weight_present = any(
        ".mlp.experts." in name
        or (
            ".mlp.switch_mlp." in name
            and name.endswith(".weight")
        )
        for name in parameter_names
    )
    return model, {
        "sparse_layer_count": len(sparse_layers),
        "bound_sparse_layer_count": bound_sparse_layers,
        "unbound_sparse_layer_count": len(sparse_layers) - bound_sparse_layers,
        "dense_routed_weight_present": dense_routed_weight_present,
    }


def _cache_offsets(cache_list: Any) -> list[int]:
    return [int(cache.offset) for cache in cache_list.caches]


def _cache_snapshot(prompt_cache: list[Any]) -> dict[str, list[int]]:
    return {
        "full_layer": _cache_offsets(prompt_cache[0]),
        "shared_layer": _cache_offsets(prompt_cache[1]),
    }


def _greedy_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _logprobs_sha256(logprobs: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(logprobs)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _run_generate_step(
    model: GLM52VQModel,
    prompt: mx.array,
) -> tuple[dict[str, Any], np.ndarray]:
    prompt_cache = model.make_cache()
    token_ids: list[int] = []
    rows: list[np.ndarray] = []
    cache_offsets_after_yields: list[dict[str, Any]] = []
    generator = generate_step(
        prompt,
        model,
        max_tokens=REQUESTED_GENERATED_TOKENS,
        sampler=_greedy_sampler,
        prompt_cache=prompt_cache,
    )
    for yield_index, (token_id, logprobs) in enumerate(generator, start=1):
        mx.eval(logprobs)
        token_ids.append(int(token_id))
        rows.append(np.asarray(logprobs).copy())
        snapshot = _cache_snapshot(prompt_cache)
        cache_offsets_after_yields.append(
            {
                "yield_index": yield_index,
                "full_layer": snapshot["full_layer"],
                "shared_layer": snapshot["shared_layer"],
            }
        )

    if rows:
        stacked_logprobs = np.stack(rows, axis=0)
    else:
        stacked_logprobs = np.empty((0, model.args.vocab_size), dtype=np.float32)
    final_snapshot = _cache_snapshot(prompt_cache)
    record = {
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "generated_ids_within_vocab": all(
            0 <= token_id < model.args.vocab_size for token_id in token_ids
        ),
        "logprobs_shape": list(stacked_logprobs.shape),
        "logprobs_dtype": str(stacked_logprobs.dtype),
        "logprobs_finite": bool(np.isfinite(stacked_logprobs).all()),
        "logprobs_sha256": _logprobs_sha256(stacked_logprobs),
        "cache_offsets_after_yields": cache_offsets_after_yields,
        "final_full_cache_offsets": final_snapshot["full_layer"],
        "final_shared_cache_offsets": final_snapshot["shared_layer"],
    }
    return record, stacked_logprobs


def probe_glm52_synthetic_generation() -> dict[str, Any]:
    payload = _base_payload()
    model, binding = _build_model()
    payload.update(binding)
    if (
        binding["sparse_layer_count"] != 1
        or binding["bound_sparse_layer_count"] != 1
        or binding["unbound_sparse_layer_count"] != 0
        or binding["dense_routed_weight_present"] is not False
    ):
        raise GLM52SyntheticGenerationProbeError(
            "synthetic_binding",
            f"tiny routed binding contract failed: {binding}",
        )

    prompt = mx.array(PROMPT_TOKEN_IDS, dtype=mx.int32)
    direct_cache = model.make_cache()
    direct_logits = model(prompt[None], cache=direct_cache)
    mx.eval(direct_logits)
    direct_snapshot = _cache_snapshot(direct_cache)
    direct_forward = {
        "logits_shape": list(direct_logits.shape),
        "logits_finite": bool(mx.all(mx.isfinite(direct_logits)).item()),
        "full_cache_offsets": direct_snapshot["full_layer"],
        "shared_cache_offsets": direct_snapshot["shared_layer"],
    }
    payload["direct_forward"] = direct_forward
    if direct_forward != {
        "logits_shape": [1, 3, model.args.vocab_size],
        "logits_finite": True,
        "full_cache_offsets": [3, 3],
        "shared_cache_offsets": [3],
    }:
        raise GLM52SyntheticGenerationProbeError(
            "direct_forward",
            f"tiny whole-model direct forward contract failed: {direct_forward}",
        )

    first_generation, first_logprobs = _run_generate_step(model, prompt)
    second_generation, second_logprobs = _run_generate_step(model, prompt)
    payload["generate_step"] = first_generation
    repeatability_pass = bool(
        first_generation["generated_token_ids"]
        == second_generation["generated_token_ids"]
        and first_generation["logprobs_sha256"]
        == second_generation["logprobs_sha256"]
        and np.array_equal(first_logprobs, second_logprobs)
        and first_generation["cache_offsets_after_yields"]
        == second_generation["cache_offsets_after_yields"]
    )
    payload["repeatability"] = {
        "repeatability_pass": repeatability_pass,
        "generated_token_ids": second_generation["generated_token_ids"],
        "logprobs_sha256": second_generation["logprobs_sha256"],
        "final_full_cache_offsets": second_generation[
            "final_full_cache_offsets"
        ],
        "final_shared_cache_offsets": second_generation[
            "final_shared_cache_offsets"
        ],
    }

    expected_yield_offsets = [
        {"yield_index": 1, "full_layer": [4, 4], "shared_layer": [4]},
        {"yield_index": 2, "full_layer": [5, 5], "shared_layer": [5]},
    ]
    generation_checks = {
        "generated_token_count": (
            first_generation["generated_token_count"]
            == REQUESTED_GENERATED_TOKENS
        ),
        "generated_ids_within_vocab": first_generation[
            "generated_ids_within_vocab"
        ],
        "logprobs_shape": first_generation["logprobs_shape"]
        == [REQUESTED_GENERATED_TOKENS, model.args.vocab_size],
        "logprobs_finite": first_generation["logprobs_finite"],
        "cache_offsets_after_yields": (
            first_generation["cache_offsets_after_yields"]
            == expected_yield_offsets
        ),
        "final_full_cache_offsets": (
            first_generation["final_full_cache_offsets"] == [5, 5]
        ),
        "final_shared_cache_offsets": (
            first_generation["final_shared_cache_offsets"] == [5]
        ),
        "repeatability": repeatability_pass,
    }
    failed_checks = [name for name, passed in generation_checks.items() if not passed]
    if failed_checks:
        raise GLM52SyntheticGenerationProbeError(
            "generate_step",
            f"tiny whole-model generate_step checks failed: {failed_checks}",
        )

    payload.update(
        {
            "probe_status": PROBE_READY_STATUS,
            "probe_pass": True,
            "synthetic_runtime_contract": True,
        }
    )
    return payload


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = _absolute(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded, deterministic generate_step compatibility proof on "
            "an in-memory tiny GLM-5.2 fixture. This is not production generation."
        )
    )
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = probe_glm52_synthetic_generation()
    except Exception as error:  # Preserve durable failure evidence for every stage.
        payload = _base_payload()
        payload["failure_stage"] = (
            error.stage
            if isinstance(error, GLM52SyntheticGenerationProbeError)
            else "unexpected_error"
        )
        payload["input_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    _write_json_atomic(args.output_json, payload)
    return 0 if payload.get("probe_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
