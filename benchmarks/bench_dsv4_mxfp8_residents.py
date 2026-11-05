"""Kill test: keep DeepSeek-V4 FP8 residents as their source bytes (mxfp8).

Two checks in one fresh process:

1. **Exactness.** Each chosen resident, byte-copied into MLX ``mxfp8`` by
   :func:`keep.convert.fp8_block.repack_fp8_block_as_mxfp8`, must dequantize
   bit-identically to :func:`keep.convert.fp8_block.dequantize_fp8_block`,
   which is what RAMP binds in bf16 today.
2. **Decode-shape speed (M=1).** Time ``quantized_matmul(mode="mxfp8")``
   against the bf16 matmul RAMP runs today and an affine-8 g64 baseline (the
   roadmap's S1 alternative).

Verdict, from the ranked plan: **KILL** if any bit differs, if mxfp8 is less
than 1.5x the bf16 speed, or if it is more than 10% slower than affine-8.
Speed is judged on the summed per-call medians across the chosen tensors, the
way one decode step reads them.

Inputs: a resident package or the source checkpoint (``--checkpoint``, a
directory with ``model.safetensors.index.json``), or ``--synthetic`` FP8
tensors at the real DeepSeek-V4-Flash shapes, which needs no download.
Weights are cycled across ``--copies`` buffers so each timed call reads DRAM,
not a cache. Speed only means something on an Apple GPU; on a CPU device the
script still runs the exactness half and marks the speed rows invalid.

Example (M4 Max, no model download needed)::

    uv run python benchmarks/bench_dsv4_mxfp8_residents.py --synthetic \\
        --output-json artifacts/benchmarks/dsv4-mxfp8-residents-m4max.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from keep.convert.fp8_block import dequantize_fp8_block, repack_fp8_block_as_mxfp8
from keep.io.source_safetensors import read_safetensors_file_header

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
RECORD_TYPE = "dsv4_mxfp8_resident_kill_test_v1"
FP8_BLOCK = (128, 128)

#: DeepSeek-V4-Flash-0731 resident shapes (out, in) as they ship, from the
#: model defaults: hidden 4096, 64 heads x 512, q_lora_rank 1024, o_groups 8 x
#: o_lora_rank 1024, shared expert width 2048. ``wo_a`` ships flat and runs as
#: an 8-group MultiLinear; see :data:`GROUPED_TENSORS`.
V4_FLASH_RESIDENT_SHAPES: dict[str, tuple[int, int]] = {
    "attn.wq_a": (1024, 4096),
    "attn.wq_b": (32768, 1024),
    "attn.wkv": (512, 4096),
    "attn.wo_a": (8192, 4096),
    "attn.wo_b": (4096, 8192),
    "ffn.shared_experts.w1": (2048, 4096),
    "ffn.shared_experts.w2": (4096, 2048),
    "ffn.shared_experts.w3": (2048, 4096),
}

#: Tensors timed as a grouped (batched) matmul, the way RAMP calls them:
#: ``wo_a`` is ``(o_groups * o_lora_rank, in)`` on disk and
#: ``(o_groups, o_lora_rank, in)`` at runtime, fed one row per group.
GROUPED_TENSORS: dict[str, int] = {"attn.wo_a": 8}

MIN_SPEEDUP_OVER_BF16 = 1.5
MAX_SLOWDOWN_VS_AFFINE8 = 1.10
_FORBIDDEN_ENV = ("GLM_MLX_WIRED_LIMIT_GB", "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB")


def synthetic_fp8_pair(
    rng: np.random.Generator, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Random finite e4m3 codes plus ue8m0 block exponents at ``shape``."""

    rows, cols = shape
    codes = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    codes = np.where((codes & 0x7F) == 0x7F, 0x7E, codes).astype(np.uint8)
    blocks = (-(-rows // FP8_BLOCK[0]), -(-cols // FP8_BLOCK[1]))
    scales = rng.integers(115, 126, size=blocks, dtype=np.uint8)
    return codes, scales


class _CheckpointReader:
    """Raw-byte reader over a safetensors index; one memmap per shard."""

    def __init__(self, root: Path) -> None:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.root = root
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._shards: dict[str, tuple[np.memmap, int, Mapping[str, Any]]] = {}

    def raw(self, name: str) -> tuple[np.ndarray, str, tuple[int, ...]]:
        shard = self.weight_map[name]
        if shard not in self._shards:
            path = self.root / shard
            header = read_safetensors_file_header(path)
            self._shards[shard] = (
                np.memmap(path, dtype=np.uint8, mode="r"),
                header.header_size,
                header.tensors,
            )
        mm, header_size, tensors = self._shards[shard]
        tensor = tensors[name]
        start, end = tensor.data_offsets
        base = 8 + header_size
        return np.array(mm[base + start : base + end]), tensor.dtype, tuple(tensor.shape)

    def fp8_pair(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        codes, code_dtype, shape = self.raw(f"{prefix}.weight")
        scales, scale_dtype, scale_shape = self.raw(f"{prefix}.scale")
        if code_dtype != "F8_E4M3" or scale_dtype != "F8_E8M0":
            raise ValueError(
                f"{prefix} is {code_dtype}/{scale_dtype}, not an F8_E4M3 resident"
            )
        return codes.reshape(shape), scales.reshape(scale_shape)


def check_exactness(codes: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    """Compare the mxfp8 byte copy against today's bf16 decode, bit for bit."""

    packed, group_scales = repack_fp8_block_as_mxfp8(codes, scales, block_size=FP8_BLOCK)
    reference = mx.array(dequantize_fp8_block(codes, scales, block_size=FP8_BLOCK)).astype(
        mx.bfloat16
    )
    restored = mx.dequantize(
        mx.array(packed),
        mx.array(group_scales),
        group_size=32,
        bits=8,
        mode="mxfp8",
        dtype=mx.bfloat16,
    )
    mismatches = int(mx.sum(restored.view(mx.uint16) != reference.view(mx.uint16)).item())
    return {
        "elements": int(codes.size),
        "bit_mismatches": mismatches,
        "bf16_bytes": int(codes.size) * 2,
        "mxfp8_bytes": int(packed.nbytes + group_scales.nbytes),
    }


def _median_call_ms(
    fn: Callable[[int], mx.array], *, iterations: int, warmup: int
) -> float:
    for i in range(warmup):
        mx.eval(fn(i))
    started = time.perf_counter()
    for i in range(iterations):
        mx.eval(fn(i))
    return (time.perf_counter() - started) * 1_000.0 / iterations


def time_variants(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    copies: int,
    iterations: int,
    warmup: int,
    rounds: int,
    seed: int,
    groups: int = 1,
) -> dict[str, Any]:
    """Per-call M=1 latency for bf16, mxfp8 and affine-8 g64, interleaved.

    ``groups > 1`` times the grouped form: rows split into ``groups`` equal
    blocks, each multiplied by its own activation row, as ``wo_a`` runs.
    """

    rows, cols = codes.shape
    if groups < 1 or rows % groups:
        raise ValueError(f"{rows} rows do not split into {groups} groups")
    packed, group_scales = repack_fp8_block_as_mxfp8(codes, scales, block_size=FP8_BLOCK)
    dense = mx.array(dequantize_fp8_block(codes, scales, block_size=FP8_BLOCK)).astype(
        mx.bfloat16
    )
    if groups > 1:
        packed = packed.reshape(groups, rows // groups, -1)
        group_scales = group_scales.reshape(groups, rows // groups, -1)
        dense = dense.reshape(groups, rows // groups, cols)
    affine_w, affine_s, affine_b = mx.quantize(dense, group_size=64, bits=8)
    # Distinct buffers per copy so consecutive calls cannot hit a cache.
    bf16_copies = [dense + mx.zeros_like(dense) for _ in range(copies)]
    packed_copies = [mx.array(packed) for _ in range(copies)]
    scale_copies = [mx.array(group_scales) for _ in range(copies)]
    affine_copies = [
        (affine_w + mx.zeros_like(affine_w), affine_s + 0, affine_b + 0) for _ in range(copies)
    ]
    mx.eval(bf16_copies, packed_copies, scale_copies, affine_copies)
    x_shape = (1, groups, 1, cols) if groups > 1 else (1, cols)
    x = mx.random.normal(x_shape, key=mx.random.key(seed)).astype(mx.bfloat16)

    def bf16(i: int) -> mx.array:
        return x @ bf16_copies[i % copies].swapaxes(-1, -2)

    def mxfp8(i: int) -> mx.array:
        j = i % copies
        return mx.quantized_matmul(
            x, packed_copies[j], scale_copies[j], transpose=True, group_size=32, bits=8, mode="mxfp8"
        )

    def affine8(i: int) -> mx.array:
        w, s, b = affine_copies[i % copies]
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=8)

    variants = {"bf16": bf16, "mxfp8": mxfp8, "affine8_g64": affine8}
    samples: dict[str, list[float]] = {name: [] for name in variants}
    order = list(variants)
    for round_index in range(rounds):
        # Rotate the order so no variant always runs first or last.
        rotated = order[round_index % len(order) :] + order[: round_index % len(order)]
        for name in rotated:
            samples[name].append(
                _median_call_ms(variants[name], iterations=iterations, warmup=warmup)
            )
    weight_bytes = {
        "bf16": rows * cols * 2,
        "mxfp8": int(packed.nbytes + group_scales.nbytes),
        "affine8_g64": int(affine_w.nbytes + affine_s.nbytes + affine_b.nbytes),
    }
    out: dict[str, Any] = {}
    for name, values in samples.items():
        median_ms = statistics.median(values)
        out[name] = {
            "median_ms_per_call": median_ms,
            "round_ms": values,
            "weight_bytes": weight_bytes[name],
            "effective_gb_per_s": weight_bytes[name] / (median_ms / 1_000.0) / 1e9,
        }
    del bf16_copies, packed_copies, scale_copies, affine_copies
    return out


def verdict(
    exactness: Mapping[str, Mapping[str, Any]],
    timings: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
    *,
    speed_valid: bool,
) -> dict[str, Any]:
    """Apply the kill criteria. Exactness alone can kill; speed needs a GPU."""

    mismatches = sum(int(row["bit_mismatches"]) for row in exactness.values())
    result: dict[str, Any] = {
        "bit_mismatches": mismatches,
        "exact": mismatches == 0,
        "speed_valid": bool(speed_valid and timings),
    }
    reasons: list[str] = []
    if mismatches:
        reasons.append(f"{mismatches} dequantized weights differ from the bf16 decode")
    if timings:
        totals = {
            variant: sum(float(rows[variant]["median_ms_per_call"]) for rows in timings.values())
            for variant in ("bf16", "mxfp8", "affine8_g64")
        }
        speedup = totals["bf16"] / totals["mxfp8"]
        vs_affine = totals["mxfp8"] / totals["affine8_g64"]
        result.update(
            {
                "summed_median_ms": totals,
                "mxfp8_speedup_over_bf16": speedup,
                "mxfp8_time_over_affine8": vs_affine,
            }
        )
        if speed_valid:
            if speedup < MIN_SPEEDUP_OVER_BF16:
                reasons.append(
                    f"mxfp8 is {speedup:.3f}x bf16, below the {MIN_SPEEDUP_OVER_BF16}x bar"
                )
            if vs_affine > MAX_SLOWDOWN_VS_AFFINE8:
                reasons.append(
                    f"mxfp8 takes {vs_affine:.3f}x affine-8's time, above {MAX_SLOWDOWN_VS_AFFINE8}x"
                )
    if reasons:
        result["verdict"] = "KILL"
    elif result["speed_valid"]:
        result["verdict"] = "PASS"
    else:
        result["verdict"] = "EXACT_ONLY"  # exactness passed; rerun on an Apple GPU for speed
    result["reasons"] = reasons
    return result


def run(
    *,
    tensors: Sequence[str],
    layer: int,
    checkpoint: Path | None,
    synthetic_shapes: Mapping[str, tuple[int, int]] | None,
    copies: int,
    iterations: int,
    warmup: int,
    rounds: int,
    seed: int,
    exactness_only: bool,
) -> dict[str, Any]:
    if (checkpoint is None) == (synthetic_shapes is None):
        raise ValueError("pass exactly one of checkpoint or synthetic_shapes")
    rng = np.random.default_rng(seed)
    reader = _CheckpointReader(checkpoint) if checkpoint is not None else None
    on_gpu = mx.default_device() == mx.gpu
    exactness: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, Any]] = {}
    shapes: dict[str, list[int]] = {}
    for tensor in tensors:
        if reader is not None:
            codes, scales = reader.fp8_pair(f"layers.{layer}.{tensor}")
        else:
            codes, scales = synthetic_fp8_pair(rng, synthetic_shapes[tensor])
        shapes[tensor] = list(codes.shape)
        exactness[tensor] = check_exactness(codes, scales)
        if not exactness_only:
            timings[tensor] = time_variants(
                codes,
                scales,
                copies=copies,
                iterations=iterations,
                warmup=warmup,
                rounds=rounds,
                seed=seed,
                groups=GROUPED_TENSORS.get(tensor, 1),
            )
        mx.clear_cache()
    return {
        "record_type": RECORD_TYPE,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "synthetic" if reader is None else str(checkpoint),
        "layer": None if reader is None else layer,
        "shapes": shapes,
        "device": str(mx.default_device()),
        "device_info": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in mx.device_info().items()}
        if on_gpu
        else {},
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "settings": {
            "copies": copies,
            "iterations": iterations,
            "warmup": warmup,
            "rounds": rounds,
            "seed": seed,
        },
        "exactness": exactness,
        "timings": timings,
        "verdict": verdict(exactness, timings or None, speed_valid=on_gpu),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="resident package or source checkpoint dir")
    source.add_argument("--synthetic", action="store_true", help="random FP8 at V4-Flash shapes")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--tensors", default=",".join(V4_FLASH_RESIDENT_SHAPES))
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--exactness-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true", help="skip the heavy-job lock")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    forbidden = [name for name in _FORBIDDEN_ENV if name in os.environ]
    if forbidden:
        parser.error("unset custom MLX wired limits first: " + ", ".join(forbidden))
    tensors = [name.strip() for name in args.tensors.split(",") if name.strip()]
    unknown = [name for name in tensors if args.synthetic and name not in V4_FLASH_RESIDENT_SHAPES]
    if unknown:
        parser.error("no synthetic shape for: " + ", ".join(unknown))

    def _go() -> dict[str, Any]:
        return run(
            tensors=tensors,
            layer=args.layer,
            checkpoint=args.checkpoint,
            synthetic_shapes=V4_FLASH_RESIDENT_SHAPES if args.synthetic else None,
            copies=args.copies,
            iterations=args.iterations,
            warmup=args.warmup,
            rounds=args.rounds,
            seed=args.seed,
            exactness_only=args.exactness_only,
        )

    if args.no_lock:
        result = _go()
    else:
        from keep.quality.dsv4_teacher_agreement import heavy_job_lock

        with heavy_job_lock(HEAVY_JOB_LOCK, holder="bench_dsv4_mxfp8_residents"):
            result = _go()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0 if result["verdict"]["verdict"] != "KILL" else 1


if __name__ == "__main__":
    sys.exit(main())
