from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8_1bit_packed
from mlx_vq.ops.vq_switch import gather_vqmm, vq_switch_qmv


def _time_ms(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(fn())
    end = time.perf_counter()
    return (end - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused gather_vqmm against scalar per-route qmv.")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--in-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=1408)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--codebook-duplication", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.in_dim % 8 != 0 or args.in_dim % args.group_size != 0:
        raise SystemExit("--in-dim must be divisible by 8 and --group-size")

    codes = mx.array(
        rng.integers(
            0,
            256,
            size=(args.experts, args.out_dim, args.in_dim // 8),
            dtype=np.uint8,
        )
    )
    scales = mx.array(
        rng.uniform(
            0.01,
            0.05,
            size=(args.experts, args.out_dim, args.in_dim // args.group_size),
        ).astype(np.float16)
    )
    x = mx.array(rng.normal(size=(args.tokens, args.in_dim)).astype(np.float16))
    hot_experts = min(args.experts, max(args.top_k, 4))
    rhs_np = np.arange(args.tokens * args.top_k, dtype=np.int32).reshape(args.tokens, args.top_k) % hot_experts
    rhs = mx.array(rhs_np)
    route_x = mx.array(rng.normal(size=(args.tokens * args.top_k, args.in_dim)).astype(np.float16))
    flat_rhs = mx.array(rhs_np.reshape(-1))
    flat_lhs = mx.arange(args.tokens * args.top_k, dtype=mx.int32)
    codebook = mx.array(e8_1bit_packed())
    mx.eval(codes, scales, x, rhs, route_x, flat_rhs, flat_lhs, codebook)

    scalar_ms = _time_ms(
        lambda: vq_switch_qmv(
            x,
            codes,
            scales,
            codebook,
            rhs,
            input_dims=args.in_dim,
            output_dims=args.out_dim,
            group_size=args.group_size,
            code_bits=8,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fused_ms = _time_ms(
        lambda: gather_vqmm(
            x,
            codes,
            scales,
            codebook,
            rhs,
            input_dims=args.in_dim,
            output_dims=args.out_dim,
            group_size=args.group_size,
            code_bits=8,
            implementation="metal",
            codebook_duplication=args.codebook_duplication,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    sorted_fused_ms = _time_ms(
        lambda: gather_vqmm(
            x,
            codes,
            scales,
            codebook,
            rhs,
            input_dims=args.in_dim,
            output_dims=args.out_dim,
            group_size=args.group_size,
            code_bits=8,
            implementation="metal",
            sorted_indices=True,
            codebook_duplication=args.codebook_duplication,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    explicit_lhs_scalar_ms = _time_ms(
        lambda: vq_switch_qmv(
            route_x,
            codes,
            scales,
            codebook,
            flat_rhs,
            input_dims=args.in_dim,
            output_dims=args.out_dim,
            group_size=args.group_size,
            code_bits=8,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    explicit_lhs_fused_ms = _time_ms(
        lambda: gather_vqmm(
            route_x,
            codes,
            scales,
            codebook,
            flat_rhs,
            lhs_indices=flat_lhs,
            input_dims=args.in_dim,
            output_dims=args.out_dim,
            group_size=args.group_size,
            code_bits=8,
            implementation="metal",
            codebook_duplication=args.codebook_duplication,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "top_k": args.top_k,
                "experts": args.experts,
                "in_dim": args.in_dim,
                "out_dim": args.out_dim,
                "group_size": args.group_size,
                "codebook_duplication": args.codebook_duplication,
                "iterations": args.iterations,
                "scalar_ms": scalar_ms,
                "fused_ms": fused_ms,
                "sorted_fused_ms": sorted_fused_ms,
                "speedup": scalar_ms / fused_ms if fused_ms > 0 else None,
                "explicit_lhs_scalar_ms": explicit_lhs_scalar_ms,
                "explicit_lhs_fused_ms": explicit_lhs_fused_ms,
                "explicit_lhs_speedup": (
                    explicit_lhs_scalar_ms / explicit_lhs_fused_ms if explicit_lhs_fused_ms > 0 else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
