from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import hf_hub_download

from ramp.benchmark.metrics import collect_vm_stat_counts, wait_for_memory_quiet
from ramp.benchmark.glm45_air import (
    append_jsonl,
    run_generation_benchmark,
    run_mlx_quantized_routed_generation_benchmark,
    scenario_defaults,
)
from ramp.benchmark.quant_compare import (
    add_comparison_metadata,
    aggregate_repetition_records,
    audit_vq_artifact_prefill_compatibility,
    artifact_quantized_parameter_bytes,
    glm45_air_routed_expert_weight_count,
    has_memory_pressure,
    mlx_affine_bit_support,
    official_conversion_memory_preflight,
    validate_mlx_routed_group_size,
)
from ramp.benchmark.vq2_preflight import run_vq2_preflight
from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID


ENGINE_CONFIG = {
    "vq_e1_routed": {
        "engine": "vq_resident",
        "quant_family": "vq_e8",
        "quantized_scope": "routed_experts",
        "artifact_dir": "artifacts/glm-4.5-air-vq",
        "vq_code_bits": 8,
        "vq_group_size": None,
        "mlx_bits": None,
        "mlx_group_size": None,
        "mlx_mode": None,
        "non_expert_dtype_policy": "source_bfloat16",
        "prefill_engine": "auto",
    },
    "vq_e1_routed_vq_metal": {
        "engine": "vq_resident_vq_metal",
        "quant_family": "vq_e8",
        "quantized_scope": "routed_experts",
        "artifact_dir": "artifacts/glm-4.5-air-vq",
        "vq_code_bits": 8,
        "vq_group_size": None,
        "mlx_bits": None,
        "mlx_group_size": None,
        "mlx_mode": None,
        "non_expert_dtype_policy": "source_bfloat16",
        "prefill_engine": "vq_metal",
    },
    "vq_e1_routed_nax_e8": {
        "engine": "vq_resident_nax_e8",
        "quant_family": "vq_e8",
        "quantized_scope": "routed_experts",
        "artifact_dir": "artifacts/glm-4.5-air-vq",
        "vq_code_bits": 8,
        "vq_group_size": None,
        "mlx_bits": None,
        "mlx_group_size": None,
        "mlx_mode": None,
        "non_expert_dtype_policy": "source_bfloat16",
        "prefill_engine": "nax_e8",
    },
    "vq_e1_routed_nax_e8p": {
        "engine": "vq_resident_nax_e8p",
        "quant_family": "vq_e8",
        "quantized_scope": "routed_experts",
        "artifact_dir": "artifacts/glm-4.5-air-vq",
        "vq_code_bits": 16,
        "vq_group_size": None,
        "mlx_bits": None,
        "mlx_group_size": None,
        "mlx_mode": None,
        "non_expert_dtype_policy": "source_bfloat16",
        "prefill_engine": "nax_e8p",
    },
    "mlx_q2_routed_g128": {
        "engine": "mlx_quant_routed_only",
        "quant_family": "mlx_affine",
        "quantized_scope": "routed_experts",
        "artifact_dir": "artifacts/glm-4.5-air-mlx-q2-routed-g128",
        "vq_code_bits": None,
        "vq_group_size": None,
        "mlx_bits": 2,
        "mlx_group_size": 128,
        "mlx_mode": "affine",
        "non_expert_dtype_policy": "source_bfloat16",
        "prefill_engine": None,
    },
}


class MemoryQuietPreflightError(RuntimeError):
    def __init__(self, preflight: dict[str, Any]):
        super().__init__("memory quiet preflight failed")
        self.preflight = preflight


PUBLICATION_ENGINES = (
    "vq_e1_routed",
    "vq_e1_routed_nax_e8",
    "vq_e1_routed_nax_e8p",
    "vq_e1_routed_vq_metal",
    "mlx_q2_routed_g128",
)
PUBLICATION_SCENARIOS = ("decode_128", "prefill_1k", "prefill_4k")


def _load_config(path: str | None, *, model_id: str, revision: str) -> dict[str, Any]:
    if path is not None:
        return json.loads(Path(path).read_text())
    config_path = hf_hub_download(model_id, "config.json", revision=revision, local_files_only=True)
    return json.loads(Path(config_path).read_text())


def _resolve_index_path(path: str | None, *, model_id: str, revision: str) -> Path:
    if path is not None:
        return Path(path)
    return Path(
        hf_hub_download(
            model_id,
            "model.safetensors.index.json",
            revision=revision,
            local_files_only=True,
        )
    )


def _system_memory_bytes() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        return 0


def _scenario_timing_key(scenario: str) -> str:
    if scenario in {"short_decode", "decode_128"}:
        return "decode_seconds"
    return "prefill_seconds"


def _single_int_from_counts(counts: dict[str, int]) -> int | None:
    if len(counts) != 1:
        return None
    return int(next(iter(counts)))


def _vq_metadata_from_prefill_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "vq_code_bits": _single_int_from_counts(dict(audit.get("code_bits_counts") or {})),
        "vq_group_size": _single_int_from_counts(dict(audit.get("group_size_counts") or {})),
        "prefill_compatibility": {
            "nax_e8_compatible_layer_count": int(audit.get("nax_e8_compatible_layer_count") or 0),
            "nax_e8p_compatible_layer_count": int(audit.get("nax_e8p_compatible_layer_count") or 0),
            "nax_fast_compatible_layer_count": int(audit.get("nax_fast_compatible_layer_count") or 0),
            "metal_fallback_layer_count": int(audit.get("metal_fallback_layer_count") or 0),
            "auto_prefill_all_layers_nax_e8_compatible": bool(
                audit.get("auto_prefill_all_layers_nax_e8_compatible")
            ),
            "auto_prefill_all_layers_nax_e8p_compatible": bool(
                audit.get("auto_prefill_all_layers_nax_e8p_compatible")
            ),
            "auto_prefill_all_layers_nax_fast_compatible": bool(
                audit.get("auto_prefill_all_layers_nax_fast_compatible")
            ),
            "pinned_nax_e8_will_fail": bool(audit.get("pinned_nax_e8_will_fail")),
            "pinned_nax_e8p_will_fail": bool(audit.get("pinned_nax_e8p_will_fail")),
            "code_bits_counts": dict(audit.get("code_bits_counts") or {}),
            "group_size_counts": dict(audit.get("group_size_counts") or {}),
            "continuous_sidecar_count": int(audit.get("continuous_sidecar_count") or 0),
            "symlink_projection_count": int(audit.get("symlink_projection_count") or 0),
        },
    }


def _gb_to_bytes(
    value: float | None, *, name: str, allow_zero: bool = False
) -> int | None:
    if value is None:
        return None
    if float(value) < 0 or (float(value) == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")
    return int(float(value) * (1024**3))


def _apply_mlx_memory_policy(args: argparse.Namespace) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "cache_limit_bytes": None,
        "memory_limit_bytes": None,
        "wired_limit_bytes": None,
        "previous_cache_limit_bytes": None,
        "previous_memory_limit_bytes": None,
        "previous_wired_limit_bytes": None,
        "clear_cache_before_run": bool(getattr(args, "mlx_clear_cache_before_run", False)),
        "cache_bytes_before_clear": None,
        "cache_bytes_after_clear": None,
    }
    cache_limit = _gb_to_bytes(
        getattr(args, "mlx_cache_limit_gb", None),
        name="--mlx-cache-limit-gb",
        allow_zero=True,
    )
    memory_limit = _gb_to_bytes(getattr(args, "mlx_memory_limit_gb", None), name="--mlx-memory-limit-gb")
    wired_limit = _gb_to_bytes(getattr(args, "mlx_wired_limit_gb", None), name="--mlx-wired-limit-gb")
    if cache_limit is not None:
        policy["cache_limit_bytes"] = cache_limit
        policy["previous_cache_limit_bytes"] = int(mx.set_cache_limit(cache_limit))
    if memory_limit is not None:
        policy["memory_limit_bytes"] = memory_limit
        policy["previous_memory_limit_bytes"] = int(mx.set_memory_limit(memory_limit))
    if wired_limit is not None:
        policy["wired_limit_bytes"] = wired_limit
        policy["previous_wired_limit_bytes"] = int(mx.set_wired_limit(wired_limit))
    if policy["clear_cache_before_run"]:
        policy["cache_bytes_before_clear"] = int(mx.get_cache_memory())
        mx.clear_cache()
        policy["cache_bytes_after_clear"] = int(mx.get_cache_memory())
    policy["enabled"] = any(
        value is not None
        for value in (
            policy["cache_limit_bytes"],
            policy["memory_limit_bytes"],
            policy["wired_limit_bytes"],
        )
    ) or policy["clear_cache_before_run"]
    return policy


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    mlx_memory_policy = _apply_mlx_memory_policy(args)
    defaults = scenario_defaults(args.scenario)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(defaults["max_new_tokens"] or 0)
    context_tokens = args.context_tokens
    if context_tokens is None:
        default_context = defaults["context_tokens"]
        context_tokens = None if default_context is None else int(default_context)

    engine_config = dict(ENGINE_CONFIG[args.engine])
    artifact_dir = args.artifact_dir or str(engine_config["artifact_dir"])
    if str(engine_config["quant_family"]) == "vq_e8":
        record = run_generation_benchmark(
            scenario=args.scenario,
            model_id=args.model_id,
            revision=args.revision,
            source_dir=args.source_dir,
            config_path=args.config_path,
            index_path=args.index_path,
            artifact_dir=artifact_dir,
            prompt=args.prompt,
            context_tokens=context_tokens,
            max_new_tokens=max_new_tokens,
            profile_components=args.profile_components,
            prefill_engine=str(engine_config["prefill_engine"]),
            prefill_memory_quiet_window_seconds=args.prefill_memory_quiet_window_seconds,
            prefill_memory_quiet_max_attempts=args.prefill_memory_quiet_max_attempts,
            memory_phase_trace=args.memory_phase_trace,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        non_expert_dtype_verified = bool(record.get("non_expert_dtype_verified"))
    elif args.engine == "mlx_q2_routed_g128":
        record = run_mlx_quantized_routed_generation_benchmark(
            scenario=args.scenario,
            model_id=args.model_id,
            revision=args.revision,
            source_dir=args.source_dir,
            config_path=args.config_path,
            index_path=args.index_path,
            artifact_dir=artifact_dir,
            prompt=args.prompt,
            context_tokens=context_tokens,
            max_new_tokens=max_new_tokens,
            profile_components=args.profile_components,
            prefill_memory_quiet_window_seconds=args.prefill_memory_quiet_window_seconds,
            prefill_memory_quiet_max_attempts=args.prefill_memory_quiet_max_attempts,
            memory_phase_trace=args.memory_phase_trace,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        non_expert_dtype_verified = bool(record.get("non_expert_dtype_verified"))
    else:
        raise ValueError(f"unsupported engine {args.engine!r}")

    config = _load_config(args.config_path, model_id=args.model_id, revision=args.revision)
    counts = artifact_quantized_parameter_bytes(artifact_dir)
    vq_metadata = {
        "vq_code_bits": engine_config["vq_code_bits"],
        "vq_group_size": engine_config["vq_group_size"],
        "prefill_compatibility": None,
    }
    if str(engine_config["quant_family"]) == "vq_e8":
        vq_metadata = _vq_metadata_from_prefill_audit(
            audit_vq_artifact_prefill_compatibility(artifact_dir)
        )
    decorated = add_comparison_metadata(
        record,
        engine=str(engine_config["engine"]),
        quant_family=str(engine_config["quant_family"]),
        quantized_scope=str(engine_config["quantized_scope"]),
        artifact_dir=artifact_dir,
        code_bytes=counts["code_bytes"],
        scale_bytes=counts["scale_bytes"],
        bias_bytes=counts["bias_bytes"],
        quantized_weights=glm45_air_routed_expert_weight_count(config),
        mlx_bits=engine_config["mlx_bits"],
        mlx_group_size=engine_config["mlx_group_size"],
        mlx_mode=engine_config["mlx_mode"],
        vq_code_bits=vq_metadata["vq_code_bits"],
        vq_group_size=vq_metadata["vq_group_size"],
        non_expert_dtype_policy=str(engine_config["non_expert_dtype_policy"]),
        non_expert_dtype_verified=non_expert_dtype_verified,
        run_index=args.run_index,
        repetition_count=args.repetitions,
    )
    if str(engine_config["quant_family"]) == "vq_e8":
        decorated["vq_group_size_policy"] = "gate_up_512_down_352"
        decorated["prefill_compatibility"] = vq_metadata["prefill_compatibility"]
    if mlx_memory_policy["enabled"]:
        decorated["mlx_memory_policy"] = mlx_memory_policy
    return decorated


def _worker_command(args: argparse.Namespace, *, run_index: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--engine",
        args.engine,
        "--scenario",
        args.scenario,
        "--model-id",
        args.model_id,
        "--revision",
        args.revision,
        "--prompt",
        args.prompt,
        "--repetitions",
        str(args.repetitions),
        "--run-index",
        str(run_index),
    ]
    for flag, value in [
        ("--artifact-dir", args.artifact_dir),
        ("--source-dir", args.source_dir),
        ("--config-path", args.config_path),
        ("--index-path", args.index_path),
        ("--context-tokens", args.context_tokens),
        ("--max-new-tokens", args.max_new_tokens),
        ("--mlx-cache-limit-gb", getattr(args, "mlx_cache_limit_gb", None)),
        ("--mlx-memory-limit-gb", getattr(args, "mlx_memory_limit_gb", None)),
        ("--mlx-wired-limit-gb", getattr(args, "mlx_wired_limit_gb", None)),
        (
            "--prefill-memory-quiet-window-seconds",
            getattr(args, "prefill_memory_quiet_window_seconds", None),
        ),
        (
            "--prefill-memory-quiet-max-attempts",
            getattr(args, "prefill_memory_quiet_max_attempts", None),
        ),
        ("--prefill-chunk-size", getattr(args, "prefill_chunk_size", None)),
    ]:
        if value is not None:
            command.extend([flag, str(value)])
    if args.profile_components:
        command.append("--profile-components")
    if getattr(args, "memory_phase_trace", False):
        command.append("--memory-phase-trace")
    if getattr(args, "mlx_clear_cache_before_run", False):
        command.append("--mlx-clear-cache-before-run")
    return command


def _run_repetition(args: argparse.Namespace, *, run_index: int) -> dict[str, Any]:
    command = _worker_command(args, run_index=run_index)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def _vm_stat_diagnostic(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, Any]:
    if before is None or after is None:
        return {
            "available": False,
            "before": before,
            "after": after,
            "deltas": None,
        }
    common_keys = sorted(set(before) & set(after))
    return {
        "available": True,
        "before": before,
        "after": after,
        "deltas": {key: int(after[key]) - int(before[key]) for key in common_keys},
    }


def _artifact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in {"records", "warmup_records"}}


def _wait_for_memory_quiet(*, window_seconds: float, max_attempts: int) -> dict[str, Any]:
    return dict(wait_for_memory_quiet(window_seconds=window_seconds, max_attempts=max_attempts))


def _memory_quiet_failure_payload(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": "memory_quiet_preflight_failed",
        "reason": (
            "Host pageout/swapout counters did not stay flat for the requested "
            "quiet window; rerun after memory pressure settles."
        ),
        "memory_quiet_preflight": preflight,
    }


def _run_repetition_series(
    args: argparse.Namespace,
    *,
    matrix_record_type: str | None = None,
) -> dict[str, Any]:
    records = []
    warmup_records = []
    start_run_index = int(getattr(args, "run_index", 1) or 1)
    warmup_repetitions = max(0, int(getattr(args, "warmup_repetitions", 0) or 0))
    clean_repetition_max_attempts = max(1, int(getattr(args, "clean_repetition_max_attempts", 1) or 1))
    parent_vm_stat_diagnostics = bool(getattr(args, "parent_vm_stat_diagnostics", False))

    def run_attempt(run_index: int) -> dict[str, Any]:
        quiet_preflight = _wait_for_memory_quiet(
            window_seconds=float(getattr(args, "memory_quiet_window_seconds", 0.0) or 0.0),
            max_attempts=int(getattr(args, "memory_quiet_max_attempts", 1) or 1),
        )
        if bool(getattr(args, "require_memory_quiet_preflight", False)) and not quiet_preflight.get("quiet"):
            raise MemoryQuietPreflightError(quiet_preflight)
        parent_vm_before = collect_vm_stat_counts() if parent_vm_stat_diagnostics else None
        record = _run_repetition(args, run_index=run_index)
        parent_vm_after = collect_vm_stat_counts() if parent_vm_stat_diagnostics else None
        if parent_vm_stat_diagnostics:
            record["parent_vm_stat"] = _vm_stat_diagnostic(parent_vm_before, parent_vm_after)
        if quiet_preflight.get("enabled"):
            record["memory_quiet_preflight"] = quiet_preflight
        record["engine_name"] = args.engine
        record["scenario"] = args.scenario
        return record

    def run_once(run_index: int) -> dict[str, Any]:
        discarded_memory_pressure_attempts = 0
        last_record: dict[str, Any] | None = None
        previous_parent_vm_after = None
        for attempt_index in range(1, clean_repetition_max_attempts + 1):
            record = run_attempt(run_index)
            if (
                parent_vm_stat_diagnostics
                and previous_parent_vm_after is not None
                and isinstance(record.get("parent_vm_stat"), dict)
            ):
                record["memory_pressure_retry_after"] = _vm_stat_diagnostic(
                    previous_parent_vm_after,
                    record["parent_vm_stat"].get("after"),
                )
            if not has_memory_pressure(record):
                if attempt_index > 1:
                    record["clean_repetition_attempt_count"] = attempt_index
                    record["discarded_memory_pressure_attempts"] = discarded_memory_pressure_attempts
                    if parent_vm_stat_diagnostics:
                        record["memory_pressure_retry_replaced_dirty_record"] = True
                return record
            record["invalid_memory_pressure"] = True
            discarded_memory_pressure_attempts += 1
            last_record = record
            if parent_vm_stat_diagnostics and isinstance(record.get("parent_vm_stat"), dict):
                previous_parent_vm_after = record["parent_vm_stat"].get("after")
        if last_record is None:
            raise RuntimeError("no benchmark repetition attempts were run")
        if discarded_memory_pressure_attempts > 1:
            last_record["clean_repetition_attempt_count"] = clean_repetition_max_attempts
            last_record["discarded_memory_pressure_attempts"] = discarded_memory_pressure_attempts - 1
        if parent_vm_stat_diagnostics:
            last_record["memory_pressure_retry_replaced_dirty_record"] = False
        return last_record

    warmup_start_run_index = start_run_index - warmup_repetitions
    for warmup_index, run_index in enumerate(
        range(warmup_start_run_index, start_run_index),
        start=1,
    ):
        record = run_once(run_index)
        record["benchmark_phase"] = "warmup"
        record["warmup_index"] = warmup_index
        warmup_records.append(record)

    for run_index in range(start_run_index, start_run_index + int(args.repetitions)):
        record = run_once(run_index)
        record["benchmark_phase"] = "measured"
        if matrix_record_type is not None:
            record["record_type"] = matrix_record_type
            record["publication_matrix"] = True
        records.append(record)
        if args.append_jsonl:
            append_jsonl(args.append_jsonl, record)

    summary = aggregate_repetition_records(
        records,
        timing_key=_scenario_timing_key(args.scenario),
        max_relative_spread=float(getattr(args, "timing_stability_max_relative_spread", 0.5)),
    )
    summary.update(
        {
            "engine_name": args.engine,
            "scenario": args.scenario,
            "records": records,
            "warmup_repetition_count": warmup_repetitions,
            "warmup_records": warmup_records,
        }
    )
    return summary


def _run_publication_matrix(args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    engines = tuple(args.publication_engine or PUBLICATION_ENGINES)
    scenarios = tuple(args.publication_scenario or PUBLICATION_SCENARIOS)
    for engine in engines:
        for scenario in scenarios:
            cell_args = argparse.Namespace(**vars(args))
            cell_args.engine = engine
            cell_args.scenario = scenario
            summary = _run_repetition_series(cell_args, matrix_record_type="publication_repetition")
            summary.update(
                {
                    "record_type": "publication_summary",
                    "publication_matrix": True,
                    "accepted_clean_rows": int(summary["valid_repetition_count"]) >= int(args.repetitions),
                }
            )
            summaries.append(summary)
            if args.append_jsonl:
                append_jsonl(args.append_jsonl, _artifact_summary(summary))
    return {
        "record_type": "publication_matrix_summary",
        "engines": list(engines),
        "scenarios": list(scenarios),
        "repetitions": int(args.repetitions),
        "warmup_repetitions": max(0, int(getattr(args, "warmup_repetitions", 0) or 0)),
        "summaries": [_artifact_summary(summary) for summary in summaries],
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config_path, model_id=args.model_id, revision=args.revision)
    index_path = _resolve_index_path(args.index_path, model_id=args.model_id, revision=args.revision)
    metadata = json.loads(index_path.read_text()).get("metadata", {})
    dense_total = int(metadata.get("total_size", 0))
    memory_bytes = _system_memory_bytes()
    official_memory = official_conversion_memory_preflight(
        dense_index_total_bytes=dense_total,
        system_memory_bytes=memory_bytes,
        safety_factor=args.official_conversion_safety_factor,
    )
    return {
        "model_id": args.model_id,
        "mlx_affine_bit_support": mlx_affine_bit_support(bits=(1, 2, 3, 4), group_size=64),
        "routed_group_size_g128": validate_mlx_routed_group_size(128),
        "routed_group_size_g256": validate_mlx_routed_group_size(256),
        "official_conversion_memory_preflight": {
            "mlx_q2_routed_only": official_memory,
            "mlx_q2_full": official_memory,
        },
        "vq2_preflight": (
            run_vq2_preflight(
                source_dir=args.source_dir,
                index_path=index_path,
                config=config,
                model_id=args.model_id,
            )
            if args.include_vq2_preflight
            else "skipped"
        ),
        "engines": ENGINE_CONFIG,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare GLM-4.5-Air resident VQ with normal MLX low-bit routed quantization."
    )
    parser.add_argument("--engine", choices=sorted(ENGINE_CONFIG), default="vq_e1_routed")
    parser.add_argument(
        "--scenario",
        choices=["short_decode", "decode_128", "prefill_1k", "prefill_4k", "prefill"],
        default="short_decode",
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=0,
        help="Run this many fresh-process warmup repetitions before measured rows; warmups are not appended to evidence JSONL.",
    )
    parser.add_argument(
        "--timing-stability-max-relative-spread",
        type=float,
        default=0.5,
        help="Maximum (max-min)/median timing spread accepted for clean repetition summaries.",
    )
    parser.add_argument(
        "--clean-repetition-max-attempts",
        type=int,
        default=1,
        help="For each measured row, retry fresh subprocess attempts until a clean memory-pressure-free row is captured or this bound is exhausted.",
    )
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--append-jsonl")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--audit-prefill-compatibility",
        action="store_true",
        help="Inspect the VQ artifact metadata and report whether auto prefill can use the NAX E8 fast path.",
    )
    parser.add_argument(
        "--publication-matrix",
        action="store_true",
        help="Run the publication engine/scenario matrix with fresh subprocess repetitions.",
    )
    parser.add_argument(
        "--publication-engine",
        action="append",
        choices=sorted(ENGINE_CONFIG),
        help="Limit --publication-matrix to one or more engines; repeat for multiple engines.",
    )
    parser.add_argument(
        "--publication-scenario",
        action="append",
        choices=["decode_128", "prefill_1k", "prefill_4k"],
        help="Limit --publication-matrix to one or more publication scenarios; repeat for multiple.",
    )
    parser.add_argument("--include-vq2-preflight", action="store_true")
    parser.add_argument("--official-conversion-safety-factor", type=float, default=2.0)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument(
        "--memory-phase-trace",
        action="store_true",
        help="Attach phase-level pageout/swapout counters around load, tokenization, cache creation, prefill, and metrics collection.",
    )
    parser.add_argument(
        "--memory-quiet-window-seconds",
        type=float,
        default=0.0,
        help="Wait for pageout/swapout counters to remain flat for this many seconds before each worker.",
    )
    parser.add_argument(
        "--memory-quiet-max-attempts",
        type=int,
        default=1,
        help="Maximum quiet-window samples to try before launching a worker anyway.",
    )
    parser.add_argument(
        "--prefill-memory-quiet-window-seconds",
        type=float,
        default=0.0,
        help="Inside each worker, wait for this many quiet seconds after model load and before measured prefill.",
    )
    parser.add_argument(
        "--prefill-memory-quiet-max-attempts",
        type=int,
        default=1,
        help="Maximum child-side quiet-window samples to try before measured prefill.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help="Diagnostic: split measured prompt prefill into sequential cache updates of this many tokens.",
    )
    parser.add_argument(
        "--memory-quiet-preflight",
        action="store_true",
        help="Only run the host pageout/swapout quiet-window check and exit without loading the model.",
    )
    parser.add_argument(
        "--require-memory-quiet-preflight",
        action="store_true",
        help="Fail before launching a worker if the quiet-window check is unavailable or not quiet.",
    )
    parser.add_argument(
        "--parent-vm-stat-diagnostics",
        action="store_true",
        help="Attach parent-process vm_stat before/after snapshots and deltas to each repetition row.",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        help="Set MLX cache limit in GiB inside each worker before loading/running the model.",
    )
    parser.add_argument(
        "--mlx-memory-limit-gb",
        type=float,
        help="Set MLX memory limit in GiB inside each worker before loading/running the model.",
    )
    parser.add_argument(
        "--mlx-wired-limit-gb",
        type=float,
        help="Set MLX wired-memory limit in GiB inside each worker before loading/running the model.",
    )
    parser.add_argument(
        "--mlx-clear-cache-before-run",
        action="store_true",
        help="Call mx.clear_cache() inside each worker before loading/running the model.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.require_memory_quiet_preflight and float(args.memory_quiet_window_seconds or 0.0) <= 0:
        parser.error("--require-memory-quiet-preflight requires --memory-quiet-window-seconds > 0")

    if args.memory_quiet_preflight:
        result = _wait_for_memory_quiet(
            window_seconds=float(args.memory_quiet_window_seconds or 0.0),
            max_attempts=int(args.memory_quiet_max_attempts or 1),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("quiet"):
            raise SystemExit(21)
        return

    if args.preflight:
        print(json.dumps(_preflight(args), indent=2, sort_keys=True))
        return

    if args.audit_prefill_compatibility:
        engine_config = dict(ENGINE_CONFIG[args.engine])
        artifact_dir = args.artifact_dir or str(engine_config["artifact_dir"])
        print(json.dumps(audit_vq_artifact_prefill_compatibility(artifact_dir), indent=2, sort_keys=True))
        return

    if args.worker:
        print(json.dumps(_run_worker(args), sort_keys=True))
        return

    try:
        if args.publication_matrix:
            print(json.dumps(_run_publication_matrix(args), indent=2, sort_keys=True))
            return

        summary = _run_repetition_series(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except MemoryQuietPreflightError as error:
        payload = _memory_quiet_failure_payload(error.preflight)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(21) from error


if __name__ == "__main__":
    main()
