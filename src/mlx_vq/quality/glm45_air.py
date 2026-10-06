from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.benchmark.glm45_air import (
    append_jsonl,
    load_mlx_quantized_routed_air,
    load_resident_air,
)
from mlx_vq.benchmark.metrics import collect_metric_snapshot, collect_vm_stat_counts, reset_mlx_peak_memory
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.models.glm45_air_vq_adapter import (
    has_dense_glm45_air_routed_expert_parameters,
    has_unbound_glm45_air_vq_experts,
)
from mlx_vq.quality.prompts import QualityPrompt, get_quality_prompts


def _softmax_summary(values: np.ndarray, *, top_k: int) -> dict[str, Any]:
    shifted = values.astype(np.float64) - float(np.max(values))
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    top_count = min(top_k, probs.shape[0])
    top_indices = np.argpartition(-probs, top_count - 1)[:top_count]
    top_indices = top_indices[np.argsort(-probs[top_indices])]
    entropy = float(-(probs * np.log(probs + 1e-300)).sum())
    return {
        "finite_logits": bool(np.isfinite(values).all()),
        "entropy": entropy,
        "max_probability": float(probs[top_indices[0]]),
        "top_token_ids": [int(idx) for idx in top_indices],
        "top_probabilities": [float(probs[idx]) for idx in top_indices],
    }


def compute_topk_summary(logits: mx.array, *, top_k: int = 5) -> dict[str, Any]:
    last_logits = np.array(logits[0, -1].astype(mx.float32), copy=False)
    return _softmax_summary(last_logits, top_k=top_k)


def compute_nll_metrics(
    logits: mx.array,
    token_ids: list[int],
    *,
    max_tokens: int = 128,
) -> dict[str, float | int | None]:
    if len(token_ids) < 2:
        return {"nll": None, "perplexity": None, "nll_token_count": 0}
    token_count = min(len(token_ids) - 1, max_tokens, logits.shape[1] - 1)
    if token_count <= 0:
        return {"nll": None, "perplexity": None, "nll_token_count": 0}
    logits_slice = logits[0, :token_count].astype(mx.float32)
    targets = mx.array(token_ids[1 : token_count + 1], dtype=mx.int32)
    log_probs = logits_slice - mx.logsumexp(logits_slice, axis=-1, keepdims=True)
    target_log_probs = mx.take_along_axis(log_probs, targets[:, None], axis=-1).reshape((-1,))
    nll = -mx.mean(target_log_probs)
    mx.eval(nll)
    nll_value = float(nll.item())
    return {
        "nll": nll_value,
        "perplexity": float(np.exp(nll_value)),
        "nll_token_count": int(token_count),
    }


def build_quality_record(
    *,
    model_id: str,
    revision: str,
    artifact_dir: str,
    prompt: QualityPrompt,
    input_token_ids: list[int],
    generated_token_ids: list[int],
    generated_text: str,
    prefill_seconds: float,
    decode_seconds: float,
    metrics: dict[str, int | None],
    topk_summary: dict[str, Any],
    nll_metrics: dict[str, float | int | None],
    dense_expert_params: bool,
    unbound_vq_experts: bool,
    generation_settings: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "artifact_dir": artifact_dir,
        "prompt_id": prompt.prompt_id,
        "prompt_text": prompt.text,
        "context_tokens": len(input_token_ids),
        "max_new_tokens": prompt.max_new_tokens,
        "input_token_ids": input_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            len(generated_token_ids) / decode_seconds if decode_seconds > 0 else None
        ),
        "dense_expert_params": dense_expert_params,
        "unbound_vq_experts": unbound_vq_experts,
        "generation_settings": generation_settings or {"method": "greedy", "temperature": 0.0},
    }
    record.update(metrics)
    record.update(topk_summary)
    record.update(nll_metrics)
    if extra:
        record.update(extra)
    json.dumps(record, sort_keys=True)
    return record


def _prompt_tokens(tokenizer, prompt: QualityPrompt) -> list[int]:
    tokens = list(tokenizer.encode(prompt.text, add_special_tokens=False))
    if prompt.context_tokens is None:
        return tokens
    if not tokens:
        raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to zero tokens")
    expanded = list(tokens)
    while len(expanded) < prompt.context_tokens:
        expanded.extend(tokens)
    return expanded[: prompt.context_tokens]


def _argmax_token(logits: mx.array) -> int:
    values = np.array(logits.astype(mx.float32), copy=False)
    return int(np.argmax(values))


def run_quality_suite(
    *,
    prompt_ids: set[str] | None = None,
    append_path: str | Path | None = None,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | None = None,
    config_path: str | None = None,
    index_path: str | None = None,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-vq",
    engine: str = "vq_e1_routed",
    top_k: int = 5,
    nll_max_tokens: int = 128,
) -> list[dict[str, Any]]:
    prompts = [
        prompt for prompt in get_quality_prompts()
        if prompt_ids is None or prompt.prompt_id in prompt_ids
    ]
    if not prompts:
        raise ValueError("no quality prompts selected")

    if engine in {"vq_e1_routed", "vq_e1_routed_vq_metal", "vq_e1_routed_nax_e8", "vq_e1_routed_nax_e8p"}:
        prefill_engine = {
            "vq_e1_routed": "auto",
            "vq_e1_routed_vq_metal": "vq_metal",
            "vq_e1_routed_nax_e8": "nax_e8",
            "vq_e1_routed_nax_e8p": "nax_e8p",
        }[engine]
        model, tokenizer, _, _ = load_resident_air(
            model_id=model_id,
            revision=revision,
            source_dir=source_dir,
            config_path=config_path,
            index_path=index_path,
            artifact_dir=artifact_dir,
            prefill_engine=prefill_engine,
        )
        engine_extra = {
            "engine": {
                "vq_e1_routed": "vq_resident",
                "vq_e1_routed_vq_metal": "vq_resident_vq_metal",
                "vq_e1_routed_nax_e8": "vq_resident_nax_e8",
                "vq_e1_routed_nax_e8p": "vq_resident_nax_e8p",
            }[engine],
            "quant_family": "vq_e8",
            "quantized_scope": "routed_experts",
            "prefill_engine": prefill_engine,
        }
    elif engine == "mlx_q2_routed_g128":
        model, tokenizer, _, _ = load_mlx_quantized_routed_air(
            model_id=model_id,
            revision=revision,
            source_dir=source_dir,
            config_path=config_path,
            index_path=index_path,
            artifact_dir=artifact_dir,
        )
        engine_extra = {
            "engine": "mlx_quant_routed_only",
            "quant_family": "mlx_affine",
            "quantized_scope": "routed_experts",
        }
    else:
        raise ValueError(f"unknown quality engine {engine!r}")
    records = []
    for prompt in prompts:
        token_ids = _prompt_tokens(tokenizer, prompt)
        before_vm = collect_vm_stat_counts()
        reset_mlx_peak_memory()
        cache = model.make_cache()
        prefill_start = time.perf_counter()
        logits = model(mx.array([token_ids], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        prefill_seconds = time.perf_counter() - prefill_start
        topk_summary = compute_topk_summary(logits, top_k=top_k)
        nll_metrics = compute_nll_metrics(logits, token_ids, max_tokens=nll_max_tokens)

        generated: list[int] = []
        decode_seconds = 0.0
        next_token = _argmax_token(logits[0, -1])
        eos_tokens = set(getattr(tokenizer, "eos_token_ids", []) or [])
        for _ in range(prompt.max_new_tokens):
            generated.append(next_token)
            if next_token in eos_tokens:
                break
            decode_start = time.perf_counter()
            logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            decode_seconds += time.perf_counter() - decode_start
            next_token = _argmax_token(logits[0, -1])

        metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
        generated_text = tokenizer.decode(generated)
        record = build_quality_record(
            model_id=model_id,
            revision=revision,
            artifact_dir=str(artifact_dir),
            prompt=prompt,
            input_token_ids=token_ids,
            generated_token_ids=generated,
            generated_text=generated_text,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            metrics=metrics,
            topk_summary=topk_summary,
            nll_metrics=nll_metrics,
            dense_expert_params=has_dense_glm45_air_routed_expert_parameters(model),
            unbound_vq_experts=has_unbound_glm45_air_vq_experts(model),
            extra=engine_extra,
        )
        records.append(record)
        if append_path is not None:
            append_jsonl(append_path, record)
    return records
