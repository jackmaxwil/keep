from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from mlx_vq.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.quality.prompts import (
    QualityPrompt,
    get_quality_prompt_by_id,
    get_quality_prompt_ids,
    get_quality_prompt_set_names,
    get_quality_prompts,
    quality_prompt_metadata,
)
from mlx_vq.quality.teacher_cache import build_teacher_cache_payload

BYTES_PER_GIB = 1024**3
SOURCE_MEMORY_GUARD_EXIT_CODE = 23


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


def _limit_bytes(value_gb: float) -> int:
    if value_gb < 0:
        raise ValueError("MLX memory limits must be non-negative")
    return int(value_gb * BYTES_PER_GIB)


def configure_mlx_memory_limits(
    *,
    mlx_cache_limit_gb: float | None,
    mlx_wired_limit_gb: float | None,
) -> dict[str, int]:
    record: dict[str, int] = {}
    if mlx_cache_limit_gb is not None:
        cache_limit_bytes = _limit_bytes(mlx_cache_limit_gb)
        previous = mx.set_cache_limit(cache_limit_bytes)
        record["mlx_cache_limit_bytes"] = cache_limit_bytes
        record["previous_mlx_cache_limit_bytes"] = int(previous)
    if mlx_wired_limit_gb is not None:
        wired_limit_bytes = _limit_bytes(mlx_wired_limit_gb)
        previous = mx.set_wired_limit(wired_limit_bytes)
        record["mlx_wired_limit_bytes"] = wired_limit_bytes
        record["previous_mlx_wired_limit_bytes"] = int(previous)
    return record


def _local_physical_memory_bytes() -> int:
    output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
    return int(output.strip())


def _source_index_size_record(model_path: Path) -> dict[str, int] | None:
    if not model_path.is_dir():
        return None
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        return None
    total_size = metadata.get("total_size")
    if not isinstance(total_size, int):
        return None
    weight_map = index.get("weight_map")
    present_shard_bytes = 0
    if isinstance(weight_map, dict):
        for shard in sorted({str(item) for item in weight_map.values()}):
            shard_path = model_path / shard
            if shard_path.is_file():
                present_shard_bytes += shard_path.stat().st_size
    return {
        "source_index_total_size": total_size,
        "present_shard_bytes": present_shard_bytes,
        "guard_source_bytes": max(total_size, present_shard_bytes),
    }


def source_memory_guard_record(
    *,
    model_path: str,
    source_memory_guard_ratio: float | None,
    physical_memory_bytes: int | None = None,
) -> dict[str, object]:
    if source_memory_guard_ratio is None:
        return {"source_memory_guard_enabled": False}
    if source_memory_guard_ratio <= 0:
        raise ValueError("--source-memory-guard-ratio must be positive")

    source_size_record = _source_index_size_record(Path(model_path))
    if source_size_record is None:
        return {
            "source_memory_guard_enabled": True,
            "source_memory_guard_checked": False,
            "source_memory_guard_pass": True,
            "decision": "single_host_source_memory_guard_unchecked",
            "model_path": model_path,
            "source_memory_guard_ratio": source_memory_guard_ratio,
        }

    if physical_memory_bytes is None:
        physical_memory_bytes = _local_physical_memory_bytes()
    source_bytes = source_size_record["guard_source_bytes"]
    ratio = float(source_bytes / physical_memory_bytes) if physical_memory_bytes else None
    guard_pass = bool(ratio is not None and ratio <= source_memory_guard_ratio)
    return {
        "source_memory_guard_enabled": True,
        "source_memory_guard_checked": True,
        "source_memory_guard_pass": guard_pass,
        "decision": (
            "single_host_source_memory_guard_pass"
            if guard_pass
            else "single_host_source_memory_guard_blocked"
        ),
        "model_path": model_path,
        "source_memory_guard_ratio": source_memory_guard_ratio,
        **source_size_record,
        "physical_memory_bytes": physical_memory_bytes,
        "expected_source_to_physical_memory_ratio": ratio,
    }


def enforce_source_memory_guard(
    *,
    model_path: str,
    source_memory_guard_ratio: float | None,
) -> dict[str, object]:
    record = source_memory_guard_record(
        model_path=model_path,
        source_memory_guard_ratio=source_memory_guard_ratio,
    )
    if record.get("source_memory_guard_pass") is False:
        print(json.dumps(record, indent=2, sort_keys=True))
        raise SystemExit(SOURCE_MEMORY_GUARD_EXIT_CODE)
    return record


def main() -> None:
    prompt_ids = list(get_quality_prompt_ids(prompt_set=None))
    parser = argparse.ArgumentParser(
        description="Export an off-box GLM-4.5-Air high-bit teacher-logit cache."
    )
    parser.add_argument(
        "--model-path",
        default=GLM45_AIR_MODEL_ID,
        help="High-bit teacher path or HF repo. Use a BF16/source or trusted Q8 model.",
    )
    parser.add_argument("--revision")
    parser.add_argument(
        "--teacher-kind",
        choices=["bf16_source", "q8", "other_high_bit"],
        default="bf16_source",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt-set",
        choices=get_quality_prompt_set_names(),
        default="base",
        help="Named prompt battery to export when --prompt-id is not supplied.",
    )
    parser.add_argument("--prompt-id", action="append", choices=prompt_ids)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument(
        "--no-full-logits",
        action="store_true",
        help="Store top-k/target tensors only. This disables exact KLD in the consumer.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Forwarded to mlx_lm.load(). Useful only when the target runtime supports it.",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        help="Set MLX cache limit before loading the single-host teacher.",
    )
    parser.add_argument(
        "--mlx-wired-limit-gb",
        type=float,
        help="Set MLX wired limit before loading the single-host teacher.",
    )
    parser.add_argument(
        "--source-memory-guard-ratio",
        type=float,
        help=(
            "For local safetensors-index sources, exit before load if indexed "
            "source bytes exceed this multiple of physical memory."
        ),
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be positive")
    if args.source_memory_guard_ratio is not None and args.source_memory_guard_ratio <= 0:
        parser.error("--source-memory-guard-ratio must be positive")
    try:
        memory_limits = configure_mlx_memory_limits(
            mlx_cache_limit_gb=args.mlx_cache_limit_gb,
            mlx_wired_limit_gb=args.mlx_wired_limit_gb,
        )
    except ValueError as exc:
        parser.error(str(exc))
    guard_record = enforce_source_memory_guard(
        model_path=args.model_path,
        source_memory_guard_ratio=args.source_memory_guard_ratio,
    )

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "teacher_logits"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    prompts = (
        [get_quality_prompt_by_id(prompt_id) for prompt_id in args.prompt_id]
        if args.prompt_id
        else list(get_quality_prompts(prompt_set=args.prompt_set))
    )
    if not prompts:
        raise ValueError("no quality prompts selected")

    model, tokenizer = load(args.model_path, revision=args.revision, lazy=args.lazy)
    rows = []
    for prompt in prompts:
        token_ids = _prompt_tokens(tokenizer, prompt)
        before_vm = collect_vm_stat_counts()
        reset_mlx_peak_memory()
        start = time.perf_counter()
        logits = model(mx.array([token_ids], dtype=mx.int32))
        mx.eval(logits)
        elapsed_seconds = time.perf_counter() - start
        metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
        shard_relative = f"teacher_logits/{prompt.prompt_id}.safetensors"
        row, tensors = build_teacher_cache_payload(
            logits=logits,
            input_token_ids=token_ids,
            prompt_id=prompt.prompt_id,
            model_id=args.model_path,
            revision=args.revision or "main",
            teacher_kind=args.teacher_kind,
            shard_path=shard_relative,
            top_k=args.top_k,
            max_positions=args.max_positions,
            save_full_logits=not args.no_full_logits,
        )
        row.update(
            {
                "elapsed_seconds": elapsed_seconds,
                **quality_prompt_metadata(prompt, prompt_set=args.prompt_set),
                **metrics,
            }
        )
        mx.save_safetensors(str(output_dir / shard_relative), tensors)
        with metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        rows.append(row)

    print(
        json.dumps(
            {
                "metadata_jsonl": str(metadata_path),
                "output_dir": str(output_dir),
                "record_count": len(rows),
                "prompt_set": args.prompt_set,
                "prompt_ids": [row["prompt_id"] for row in rows],
                "full_logits_available": not args.no_full_logits,
                "top_k": args.top_k,
                "max_positions": args.max_positions,
                "memory_limits": memory_limits,
                "source_memory_guard": guard_record,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
