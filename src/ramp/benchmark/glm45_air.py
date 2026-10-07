from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from ramp.benchmark.metrics import (
    MemoryPhaseTracer,
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
    wait_for_memory_quiet,
)
from keep.vq.e8 import e8_1bit_packed
from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from keep.convert.stream_convert import load_safetensors_index
from ramp.models.glm45_air_vq_adapter import (
    GLM45AirVQModel,
    bind_glm45_air_mlx_quantized_routed_experts,
    bind_glm45_air_non_expert_precision_policy,
    bind_glm45_air_non_expert_weights,
    bind_glm45_air_vq_experts,
    glm45_air_args_from_config,
    glm45_air_non_expert_dtype_report,
    has_dense_glm45_air_routed_expert_parameters,
    has_unbound_glm45_air_vq_experts,
    set_glm45_air_prefill_engine,
    set_glm45_air_switch_gather_vqmm,
)
from ramp.nn.switch_linear import QuantizedVQSwitchLinear
from ramp.ops.vq_switch import gather_vqmm, vq_switch_qmv

Scenario = Literal[
    "short_decode",
    "decode_128",
    "prefill_1k",
    "prefill_4k",
    "prefill",
    "moe_kernel",
    "sparse_residual_micro",
]

CAPITAL_FRANCE_PROMPT = "The capital of France is"
SYNTHETIC_CONTEXT_TEXT = (
    " GLM-4.5-Air resident VQ benchmark context. "
    "This deterministic sentence is repeated to fill the requested token window."
)


class ComponentTimer:
    def __init__(self, *, eval_outputs: bool = True):
        self.eval_outputs = eval_outputs
        self._records: dict[str, dict[str, float | int | str]] = {}

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            record = self._records.setdefault(
                name,
                {
                    "name": name,
                    "calls": 0,
                    "total_seconds": 0.0,
                    "seconds_per_call": 0.0,
                },
            )
            record["calls"] = int(record["calls"]) + 1
            record["total_seconds"] = float(record["total_seconds"]) + elapsed
            record["seconds_per_call"] = float(record["total_seconds"]) / int(record["calls"])

    def to_records(self) -> list[dict[str, float | int | str]]:
        return [dict(record) for record in self._records.values()]


def _component_kind(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "moe":
        return ".".join(parts[2:])
    if len(parts) >= 3 and parts[0] == "layer":
        return ".".join(parts[2:])
    return name


def summarize_component_timings(
    records: list[dict[str, float | int | str]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for record in records:
        kind = _component_kind(str(record["name"]))
        calls = int(record.get("calls", 0))
        total_seconds = float(record.get("total_seconds", 0.0))
        grouped = summary.setdefault(
            kind,
            {
                "calls": 0,
                "sections": 0,
                "total_seconds": 0.0,
                "seconds_per_call": 0.0,
            },
        )
        grouped["calls"] = int(grouped["calls"]) + calls
        grouped["sections"] = int(grouped["sections"]) + 1
        grouped["total_seconds"] = float(grouped["total_seconds"]) + total_seconds
        grouped["seconds_per_call"] = (
            float(grouped["total_seconds"]) / int(grouped["calls"])
            if int(grouped["calls"]) > 0
            else 0.0
        )
    return {name: summary[name] for name in sorted(summary)}


def scenario_defaults(scenario: Scenario) -> dict[str, int | str | None]:
    defaults: dict[str, dict[str, int | str | None]] = {
        "short_decode": {
            "prompt_id": "capital_france",
            "context_tokens": None,
            "max_new_tokens": 8,
        },
        "decode_128": {
            "prompt_id": "capital_france_decode_128",
            "context_tokens": None,
            "max_new_tokens": 128,
        },
        "prefill_1k": {
            "prompt_id": "synthetic_repeat_1k",
            "context_tokens": 1024,
            "max_new_tokens": 1,
        },
        "prefill_4k": {
            "prompt_id": "synthetic_repeat_4k",
            "context_tokens": 4096,
            "max_new_tokens": 1,
        },
        "prefill": {
            "prompt_id": "synthetic_repeat",
            "context_tokens": 4096,
            "max_new_tokens": 1,
        },
        "moe_kernel": {
            "prompt_id": "synthetic_moe",
            "context_tokens": 8,
            "max_new_tokens": 0,
        },
        "sparse_residual_micro": {
            "prompt_id": "synthetic_sparse_residual",
            "context_tokens": 8,
            "max_new_tokens": 0,
        },
    }
    if scenario not in defaults:
        raise ValueError(f"unknown scenario {scenario!r}")
    return dict(defaults[scenario])


def validate_context_gate(
    *,
    context_tokens: int,
    allow_larger_context: bool,
    max_mlx_peak_bytes: int | None,
    max_rss_bytes: int | None,
    max_prefill_seconds: float | None,
) -> None:
    if context_tokens <= 4096:
        return
    if not allow_larger_context:
        raise ValueError(
            f"context_tokens={context_tokens} is larger than the 4K baseline; "
            "pass --allow-larger-context and explicit caps before running it"
        )
    if max_mlx_peak_bytes is None or max_rss_bytes is None or max_prefill_seconds is None:
        raise ValueError(
            "larger-context execution requires all larger-context caps: "
            "--max-mlx-peak-bytes, --max-rss-bytes, and --max-prefill-seconds"
        )


def _require_numeric_record_value(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if value is None:
        raise ValueError(f"{key} is unavailable; cannot enforce larger-context cap")
    return float(value)


def enforce_context_caps(
    record: dict[str, Any],
    *,
    max_mlx_peak_bytes: int | None,
    max_rss_bytes: int | None,
    max_prefill_seconds: float | None,
) -> None:
    if int(record.get("context_tokens", 0)) <= 4096:
        return
    if max_mlx_peak_bytes is not None:
        actual = _require_numeric_record_value(record, "mlx_peak_bytes")
        if actual > max_mlx_peak_bytes:
            raise ValueError(f"mlx_peak_bytes {actual} exceeded cap {max_mlx_peak_bytes}")
    if max_rss_bytes is not None:
        actual = _require_numeric_record_value(record, "rss_bytes")
        if actual > max_rss_bytes:
            raise ValueError(f"rss_bytes {actual} exceeded cap {max_rss_bytes}")
    if max_prefill_seconds is not None:
        actual = _require_numeric_record_value(record, "prefill_seconds")
        if actual > max_prefill_seconds:
            raise ValueError(f"prefill_seconds {actual} exceeded cap {max_prefill_seconds}")


def build_benchmark_record(
    *,
    model_id: str,
    artifact_dir: str,
    scenario: str,
    prompt_id: str,
    context_tokens: int,
    max_new_tokens: int,
    elapsed_seconds: float,
    prefill_seconds: float,
    decode_seconds: float,
    generated_text: str,
    metrics: dict[str, int | None],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": model_id,
        "artifact_dir": artifact_dir,
        "scenario": scenario,
        "prompt_id": prompt_id,
        "context_tokens": context_tokens,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": (
            context_tokens / prefill_seconds if prefill_seconds > 0 else None
        ),
        "decode_tokens_per_second": (
            max_new_tokens / decode_seconds if decode_seconds > 0 else None
        ),
        "generated_text": generated_text,
    }
    record.update(metrics)
    if extra:
        record.update(extra)
    return record


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def format_markdown_summary(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Scenario: `{record['scenario']}`",
            f"- Context/max new tokens: `{record['context_tokens']}` / `{record['max_new_tokens']}`",
            f"- Prefill/decode seconds: `{record['prefill_seconds']}` / `{record['decode_seconds']}`",
            f"- Prefill/decode tok/s: `{record['prefill_tokens_per_second']}` / `{record['decode_tokens_per_second']}`",
            f"- MLX active bytes: `{record.get('mlx_active_bytes')}`",
            f"- MLX peak bytes: `{record.get('mlx_peak_bytes')}`",
            f"- MLX cache bytes: `{record.get('mlx_cache_bytes')}`",
            f"- RSS bytes: `{record.get('rss_bytes')}`",
            "- Pageouts/swapouts delta: "
            f"`{record.get('pageouts_delta')}` / `{record.get('swapouts_delta')}`",
            f"- Generated text: `{record.get('generated_text', '')}`",
        ]
    )


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict[str, Any]:
    if config_path is not None:
        return json.loads(Path(config_path).read_text())
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text())
    except Exception:
        return fetch_hf_config(model_id, revision=revision)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "model.safetensors.index.json",
            ],
            local_files_only=True,
        )
    )


def load_tokenizer_for_source(source_dir: str | Path):
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    return load_tokenizer(Path(source_dir))


def load_resident_air(
    *,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | None = None,
    config_path: str | None = None,
    index_path: str | None = None,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-vq",
    use_gather_switch: bool = True,
    prefill_engine: Literal["auto", "vq_metal", "nax_e8", "nax_e8p"] = "auto",
):
    source_root = _resolve_source_dir(model_id=model_id, revision=revision, source_dir=source_dir)
    config = _load_config(model_id=model_id, revision=revision, config_path=config_path)
    index = load_safetensors_index(
        Path(index_path) if index_path is not None else source_root / "model.safetensors.index.json"
    )

    model = GLM45AirVQModel(glm45_air_args_from_config(config))
    non_expert_report = bind_glm45_air_non_expert_precision_policy(
        model, artifact_dir, source_root, index
    )
    bound_layers = bind_glm45_air_vq_experts(model, artifact_dir)
    set_glm45_air_switch_gather_vqmm(model, use_gather_switch)
    set_glm45_air_prefill_engine(model, prefill_engine)
    mx.eval(model.parameters())

    if has_unbound_glm45_air_vq_experts(model):
        raise RuntimeError("resident Air model has unbound VQ experts")
    if has_dense_glm45_air_routed_expert_parameters(model):
        raise RuntimeError("resident Air model materialized dense routed expert parameters")

    tokenizer = load_tokenizer_for_source(source_root)
    return model, tokenizer, non_expert_report, bound_layers


def load_mlx_quantized_routed_air(
    *,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | None = None,
    config_path: str | None = None,
    index_path: str | None = None,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-mlx-q2-routed-g128",
):
    source_root = _resolve_source_dir(model_id=model_id, revision=revision, source_dir=source_dir)
    config = _load_config(model_id=model_id, revision=revision, config_path=config_path)
    index = load_safetensors_index(
        Path(index_path) if index_path is not None else source_root / "model.safetensors.index.json"
    )

    model = GLM45AirVQModel(glm45_air_args_from_config(config))
    non_expert_report = bind_glm45_air_non_expert_weights(model, source_root, index)
    bound_layers = bind_glm45_air_mlx_quantized_routed_experts(model, artifact_dir)
    mx.eval(model.parameters())

    if has_unbound_glm45_air_vq_experts(model):
        raise RuntimeError("MLX routed Air model has unbound quantized experts")
    if has_dense_glm45_air_routed_expert_parameters(model):
        raise RuntimeError("MLX routed Air model materialized dense routed expert parameters")

    tokenizer = load_tokenizer_for_source(source_root)
    return model, tokenizer, non_expert_report, bound_layers


def _prompt_tokens(tokenizer, *, prompt: str, context_tokens: int | None) -> list[int]:
    tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    if context_tokens is None:
        return tokens
    filler = list(tokenizer.encode(SYNTHETIC_CONTEXT_TEXT, add_special_tokens=False))
    if not filler:
        raise ValueError("synthetic context filler tokenized to zero tokens")
    expanded = list(tokens)
    while len(expanded) < context_tokens:
        expanded.extend(filler)
    return expanded[:context_tokens]


def _argmax_token(logits: mx.array) -> int:
    values = np.array(logits.astype(mx.float32), copy=False)
    return int(np.argmax(values))


def _prefill_with_optional_chunks(
    model: Any,
    tokens: list[int],
    *,
    cache: Any,
    prefill_chunk_size: int | None,
    phase_trace: MemoryPhaseTracer,
) -> tuple[mx.array, int]:
    if prefill_chunk_size is None or int(prefill_chunk_size) <= 0 or int(prefill_chunk_size) >= len(tokens):
        logits = model(mx.array([tokens], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        return logits, 1
    chunk_size = int(prefill_chunk_size)
    logits = None
    chunk_count = 0
    for start in range(0, len(tokens), chunk_size):
        chunk_count += 1
        chunk = tokens[start : start + chunk_size]
        logits = model(mx.array([chunk], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        phase_trace.mark(f"after_prefill_chunk_{chunk_count}")
    if logits is None:
        raise RuntimeError("chunked prefill produced no logits")
    return logits, chunk_count


def run_generation_benchmark(
    *,
    scenario: Scenario,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | None = None,
    config_path: str | None = None,
    index_path: str | None = None,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-vq",
    prompt: str = CAPITAL_FRANCE_PROMPT,
    prompt_id: str | None = None,
    context_tokens: int | None = None,
    max_new_tokens: int | None = None,
    profile_components: bool = False,
    use_gather_switch: bool = True,
    prefill_engine: Literal["auto", "vq_metal", "nax_e8", "nax_e8p"] = "auto",
    prefill_memory_quiet_window_seconds: float = 0.0,
    prefill_memory_quiet_max_attempts: int = 1,
    memory_phase_trace: bool = False,
    prefill_chunk_size: int | None = None,
) -> dict[str, Any]:
    defaults = scenario_defaults(scenario)
    resolved_context = context_tokens
    if resolved_context is None:
        default_context = defaults["context_tokens"]
        resolved_context = None if default_context is None else int(default_context)
    resolved_max_new = max_new_tokens if max_new_tokens is not None else int(defaults["max_new_tokens"] or 0)
    resolved_prompt_id = prompt_id or str(defaults["prompt_id"])

    phase_trace = MemoryPhaseTracer(enabled=memory_phase_trace)
    phase_trace.mark("before_load")
    model, tokenizer, non_expert_report, bound_layers = load_resident_air(
        model_id=model_id,
        revision=revision,
        source_dir=source_dir,
        config_path=config_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
        use_gather_switch=use_gather_switch,
        prefill_engine=prefill_engine,
    )
    phase_trace.mark("after_load")
    component_timer = ComponentTimer(eval_outputs=True) if profile_components else None
    if component_timer is not None:
        model.set_profile_recorder(component_timer)
    tokens = _prompt_tokens(tokenizer, prompt=prompt, context_tokens=resolved_context)
    if not tokens:
        raise ValueError("benchmark prompt produced zero tokens")
    phase_trace.mark("after_tokenize")

    prefill_memory_quiet = wait_for_memory_quiet(
        window_seconds=float(prefill_memory_quiet_window_seconds or 0.0),
        max_attempts=int(prefill_memory_quiet_max_attempts or 1),
    )
    phase_trace.mark("after_prefill_quiet")
    before_vm = collect_vm_stat_counts()
    phase_trace.mark("before_measured")
    reset_mlx_peak_memory()
    cache = model.make_cache()
    phase_trace.mark("after_cache")
    start = time.perf_counter()
    prefill_start = time.perf_counter()
    logits, prefill_chunk_count = _prefill_with_optional_chunks(
        model,
        tokens,
        cache=cache,
        prefill_chunk_size=prefill_chunk_size,
        phase_trace=phase_trace,
    )
    prefill_seconds = time.perf_counter() - prefill_start
    phase_trace.mark("after_prefill")

    generated: list[int] = []
    decode_seconds = 0.0
    if resolved_max_new > 0:
        next_token = _argmax_token(logits[0, -1])
        generated.append(next_token)
        eos_tokens = set(getattr(tokenizer, "eos_token_ids", []) or [])
        for _ in range(resolved_max_new - 1):
            if next_token in eos_tokens:
                break
            decode_start = time.perf_counter()
            logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            decode_seconds += time.perf_counter() - decode_start
            next_token = _argmax_token(logits[0, -1])
            generated.append(next_token)
    phase_trace.mark("after_decode")
    elapsed = time.perf_counter() - start
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    phase_trace.mark("after_metrics")
    generated_text = tokenizer.decode(tokens + generated) if scenario == "short_decode" else tokenizer.decode(generated)
    dtype_report = glm45_air_non_expert_dtype_report(model)
    extra: dict[str, Any] = {
        "bound_vq_layers": len(bound_layers),
        "non_expert_loaded_count": non_expert_report.loaded_count,
        "dense_expert_params": has_dense_glm45_air_routed_expert_parameters(model),
        "unbound_vq_experts": has_unbound_glm45_air_vq_experts(model),
        "use_gather_switch": use_gather_switch,
        "prefill_engine": prefill_engine,
        "prefill_chunk_size": prefill_chunk_size,
        "prefill_chunk_count": prefill_chunk_count,
        "non_expert_dtype_report": dtype_report,
        "non_expert_dtype_verified": bool(dtype_report["verified"]),
    }
    if phase_trace.enabled:
        extra["memory_phase_trace"] = phase_trace.records
    if prefill_memory_quiet.get("enabled"):
        extra["prefill_memory_quiet"] = prefill_memory_quiet
    if component_timer is not None:
        component_timings = component_timer.to_records()
        extra["component_timings"] = component_timings
        extra["component_summary"] = summarize_component_timings(component_timings)
        extra["profile_components"] = True
        extra["timing_instrumented"] = True
        extra["timing_instrumentation_note"] = (
            "Component profiling forces mx.eval inside timed sections; use component timings for attribution, "
            "not the top-level timing fields for parity gates."
        )
    return build_benchmark_record(
        model_id=model_id,
        artifact_dir=str(artifact_dir),
        scenario=scenario,
        prompt_id=resolved_prompt_id,
        context_tokens=len(tokens),
        max_new_tokens=resolved_max_new,
        elapsed_seconds=elapsed,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        generated_text=generated_text,
        metrics=metrics,
        extra=extra,
    )


def run_mlx_quantized_routed_generation_benchmark(
    *,
    scenario: Scenario,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | None = None,
    config_path: str | None = None,
    index_path: str | None = None,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-mlx-q2-routed-g128",
    prompt: str = CAPITAL_FRANCE_PROMPT,
    prompt_id: str | None = None,
    context_tokens: int | None = None,
    max_new_tokens: int | None = None,
    profile_components: bool = False,
    prefill_memory_quiet_window_seconds: float = 0.0,
    prefill_memory_quiet_max_attempts: int = 1,
    memory_phase_trace: bool = False,
    prefill_chunk_size: int | None = None,
) -> dict[str, Any]:
    defaults = scenario_defaults(scenario)
    resolved_context = context_tokens
    if resolved_context is None:
        default_context = defaults["context_tokens"]
        resolved_context = None if default_context is None else int(default_context)
    resolved_max_new = max_new_tokens if max_new_tokens is not None else int(defaults["max_new_tokens"] or 0)
    resolved_prompt_id = prompt_id or str(defaults["prompt_id"])

    phase_trace = MemoryPhaseTracer(enabled=memory_phase_trace)
    phase_trace.mark("before_load")
    model, tokenizer, non_expert_report, bound_layers = load_mlx_quantized_routed_air(
        model_id=model_id,
        revision=revision,
        source_dir=source_dir,
        config_path=config_path,
        index_path=index_path,
        artifact_dir=artifact_dir,
    )
    phase_trace.mark("after_load")
    component_timer = ComponentTimer(eval_outputs=True) if profile_components else None
    if component_timer is not None:
        model.set_profile_recorder(component_timer)
    tokens = _prompt_tokens(tokenizer, prompt=prompt, context_tokens=resolved_context)
    if not tokens:
        raise ValueError("benchmark prompt produced zero tokens")
    phase_trace.mark("after_tokenize")

    prefill_memory_quiet = wait_for_memory_quiet(
        window_seconds=float(prefill_memory_quiet_window_seconds or 0.0),
        max_attempts=int(prefill_memory_quiet_max_attempts or 1),
    )
    phase_trace.mark("after_prefill_quiet")
    before_vm = collect_vm_stat_counts()
    phase_trace.mark("before_measured")
    reset_mlx_peak_memory()
    cache = model.make_cache()
    phase_trace.mark("after_cache")
    start = time.perf_counter()
    prefill_start = time.perf_counter()
    logits, prefill_chunk_count = _prefill_with_optional_chunks(
        model,
        tokens,
        cache=cache,
        prefill_chunk_size=prefill_chunk_size,
        phase_trace=phase_trace,
    )
    prefill_seconds = time.perf_counter() - prefill_start
    phase_trace.mark("after_prefill")

    generated: list[int] = []
    decode_seconds = 0.0
    if resolved_max_new > 0:
        next_token = _argmax_token(logits[0, -1])
        generated.append(next_token)
        eos_tokens = set(getattr(tokenizer, "eos_token_ids", []) or [])
        for _ in range(resolved_max_new - 1):
            if next_token in eos_tokens:
                break
            decode_start = time.perf_counter()
            logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            decode_seconds += time.perf_counter() - decode_start
            next_token = _argmax_token(logits[0, -1])
            generated.append(next_token)
    phase_trace.mark("after_decode")
    elapsed = time.perf_counter() - start
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    phase_trace.mark("after_metrics")
    generated_text = tokenizer.decode(tokens + generated) if scenario == "short_decode" else tokenizer.decode(generated)
    dtype_report = glm45_air_non_expert_dtype_report(model)
    extra: dict[str, Any] = {
        "bound_mlx_quant_layers": len(bound_layers),
        "non_expert_loaded_count": non_expert_report.loaded_count,
        "dense_expert_params": has_dense_glm45_air_routed_expert_parameters(model),
        "unbound_vq_experts": has_unbound_glm45_air_vq_experts(model),
        "prefill_chunk_size": prefill_chunk_size,
        "prefill_chunk_count": prefill_chunk_count,
        "non_expert_dtype_report": dtype_report,
        "non_expert_dtype_verified": bool(dtype_report["verified"]),
    }
    if phase_trace.enabled:
        extra["memory_phase_trace"] = phase_trace.records
    if prefill_memory_quiet.get("enabled"):
        extra["prefill_memory_quiet"] = prefill_memory_quiet
    if component_timer is not None:
        component_timings = component_timer.to_records()
        extra["component_timings"] = component_timings
        extra["component_summary"] = summarize_component_timings(component_timings)
        extra["profile_components"] = True
        extra["timing_instrumented"] = True
        extra["timing_instrumentation_note"] = (
            "Component profiling forces mx.eval inside timed sections; use component timings for attribution, "
            "not the top-level timing fields for parity gates."
        )
    return build_benchmark_record(
        model_id=model_id,
        artifact_dir=str(artifact_dir),
        scenario=scenario,
        prompt_id=resolved_prompt_id,
        context_tokens=len(tokens),
        max_new_tokens=resolved_max_new,
        elapsed_seconds=elapsed,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        generated_text=generated_text,
        metrics=metrics,
        extra=extra,
    )


def run_moe_kernel_benchmark(
    *,
    model_id: str = GLM45_AIR_MODEL_ID,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-vq",
    tokens: int = 8,
    top_k: int = 8,
    experts: int = 128,
    input_dims: int = 4096,
    output_dims: int = 1408,
    group_size: int = 512,
    iterations: int = 3,
    warmup: int = 1,
    seed: int = 20260624,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    codes = mx.array(
        rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    )
    scales = mx.array(
        rng.uniform(0.01, 0.05, size=(experts, output_dims, input_dims // group_size)).astype(np.float16)
    )
    x = mx.array(rng.normal(size=(tokens, input_dims)).astype(np.float16))
    rhs = mx.array(
        (np.arange(tokens * top_k, dtype=np.int32).reshape(tokens, top_k) % min(experts, max(top_k, 4)))
    )
    codebook = mx.array(e8_1bit_packed())
    mx.eval(codes, scales, x, rhs, codebook)

    def fused():
        return gather_vqmm(
            x,
            codes,
            scales,
            codebook,
            rhs,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            code_bits=8,
            implementation="metal",
        )

    def scalar():
        return vq_switch_qmv(
            x,
            codes,
            scales,
            codebook,
            rhs,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            code_bits=8,
            implementation="metal",
        )

    for _ in range(warmup):
        mx.eval(fused())
    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(fused())
    elapsed = time.perf_counter() - start
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)

    scalar_seconds = None
    if tokens * top_k <= 128:
        scalar_start = time.perf_counter()
        mx.eval(scalar())
        scalar_seconds = time.perf_counter() - scalar_start

    return build_benchmark_record(
        model_id=model_id,
        artifact_dir=str(artifact_dir),
        scenario="moe_kernel",
        prompt_id="synthetic_moe",
        context_tokens=tokens,
        max_new_tokens=0,
        elapsed_seconds=elapsed,
        prefill_seconds=elapsed,
        decode_seconds=0.0,
        generated_text="",
        metrics=metrics,
        extra={
            "tokens": tokens,
            "top_k": top_k,
            "experts": experts,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "group_size": group_size,
            "iterations": iterations,
            "fused_ms_per_iter": elapsed * 1000.0 / iterations,
            "scalar_one_iter_seconds": scalar_seconds,
        },
    )


def run_sparse_residual_microbenchmark(
    *,
    model_id: str = GLM45_AIR_MODEL_ID,
    artifact_dir: str | Path = "artifacts/glm-4.5-air-vq",
    tokens: int = 8,
    top_k: int = 8,
    experts: int = 128,
    input_dims: int = 4096,
    output_dims: int = 1408,
    group_size: int = 512,
    sparse_rows: int = 16,
    iterations: int = 10,
    warmup: int = 2,
    seed: int = 20260629,
) -> dict[str, Any]:
    if tokens <= 0 or top_k <= 0 or experts <= 0:
        raise ValueError("tokens, top_k, and experts must be positive")
    if input_dims <= 0 or output_dims <= 0 or group_size <= 0:
        raise ValueError("input_dims, output_dims, and group_size must be positive")
    if sparse_rows <= 0:
        raise ValueError("sparse_rows must be positive")
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative")

    rng = np.random.default_rng(seed)
    codes = mx.array(
        rng.integers(0, 256, size=(experts, output_dims, input_dims // 8), dtype=np.uint8)
    )
    scales = mx.array(
        rng.uniform(0.01, 0.05, size=(experts, output_dims, input_dims // group_size)).astype(np.float16)
    )
    codebook = mx.array(e8_1bit_packed())
    no_residual_layer = QuantizedVQSwitchLinear(
        input_dims=input_dims,
        output_dims=output_dims,
        num_experts=experts,
        codes=codes,
        scales=scales,
        codebook=codebook,
        group_size=group_size,
    )
    residual_layer = QuantizedVQSwitchLinear(
        input_dims=input_dims,
        output_dims=output_dims,
        num_experts=experts,
        codes=codes,
        scales=scales,
        codebook=codebook,
        group_size=group_size,
    )
    active_experts = (
        np.arange(tokens * top_k, dtype=np.int32).reshape(tokens, top_k)
        % min(experts, max(top_k, 1))
    )
    expert_indices = np.resize(active_experts.reshape(-1), sparse_rows).astype(np.int32)
    output_indices = (
        np.arange(sparse_rows, dtype=np.int32) * max(1, output_dims // sparse_rows)
    ) % output_dims
    residual_values = rng.normal(scale=0.001, size=(sparse_rows, input_dims)).astype(np.float32)
    residual_layer.set_sparse_residual_rows(
        expert_indices=mx.array(expert_indices),
        output_indices=mx.array(output_indices),
        values=mx.array(residual_values),
    )

    x = mx.array(rng.normal(size=(tokens, input_dims)).astype(np.float16))
    indices = mx.array(active_experts)
    y = mx.array(rng.normal(size=(tokens, top_k, output_dims)).astype(np.float16))
    mx.eval(codes, scales, x, indices, y, residual_layer.sparse_residual_values)

    def noop():
        return no_residual_layer._apply_sparse_residual_rows(y, x, indices)

    def residual():
        return residual_layer._apply_sparse_residual_rows(y, x, indices)

    for _ in range(warmup):
        mx.eval(noop(), residual())
    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    noop_start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(noop())
    noop_elapsed = time.perf_counter() - noop_start
    residual_start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(residual())
    residual_elapsed = time.perf_counter() - residual_start
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)

    noop_ms = noop_elapsed * 1000.0 / iterations
    residual_ms = residual_elapsed * 1000.0 / iterations
    routed_rows = tokens * top_k
    return build_benchmark_record(
        model_id=model_id,
        artifact_dir=str(artifact_dir),
        scenario="sparse_residual_micro",
        prompt_id="synthetic_sparse_residual",
        context_tokens=tokens,
        max_new_tokens=0,
        elapsed_seconds=residual_elapsed,
        prefill_seconds=residual_elapsed,
        decode_seconds=0.0,
        generated_text="",
        metrics=metrics,
        extra={
            "tokens": tokens,
            "top_k": top_k,
            "routed_rows": routed_rows,
            "experts": experts,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "group_size": group_size,
            "sparse_rows": sparse_rows,
            "iterations": iterations,
            "warmup": warmup,
            "noop_ms_per_iter": noop_ms,
            "residual_ms_per_iter": residual_ms,
            "residual_over_noop_ratio": residual_ms / noop_ms if noop_ms > 0 else None,
            "flat_x_by_residual_values_shape": [routed_rows, input_dims, sparse_rows],
            "row_score_by_output_selector_shape": [routed_rows, sparse_rows, output_dims],
        },
    )
