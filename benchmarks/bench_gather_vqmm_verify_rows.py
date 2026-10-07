"""Benchmark VQ verify kernels against the M=1 decode kernel looped M times.

MTP verify runs M=2-8 rows through the same expert-gathered projections that
decode runs at M=1. Without a wide-M kernel a verify pass costs M decode-shaped
launches. This benchmark measures the fused wide-M variants against that
baseline on the DeepSeek-V4-Flash routed-expert shapes with synthetic weights.

Two timings are reported per row because they answer different questions:

``latency_ms``
    One call per ``mx.eval``. This is the real per-projection verify step cost
    and includes MLX dispatch, which is ~0.2 ms on this machine and therefore
    compresses every ratio at these small shapes.
``amortized_ms``
    ``--repeat`` independent calls enqueued in one ``mx.eval``, divided by
    ``--repeat``. This isolates GPU cost, which is what the kernel change
    actually moves.

``latency_ms`` is the median over ``--iterations`` samples. ``amortized_ms`` is
the **minimum** over ``max(5, --iterations // 2)`` samples after
``--amortized-warmup`` warmups — the minimum being the standard statistic for
kernel microbenchmarks, since this machine shows up to ~1.8x run-to-run spread
from background load and the minimum is the only sample not contaminated by it.

Each row also carries ``bytes_moved``, ``code_bytes``, and per-variant ``_gbps``
and ``_code_gbps``, computed from the shapes, dtypes, and distinct-expert count.
The achieved-bandwidth argument that drives the next-increment ranking in
``docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md`` is therefore reproducible from
this script rather than asserted in prose.

Variants are timed both as raw kernels (descriptors prebuilt, validation off) and
through the ``gather_vqmm_verify_rows`` op, so graph-construction overhead in the
op layer is attributable rather than silently folded into the kernel result.

``--include-dispatch-flip`` adds the Wave 4 increment 2 comparison: the old
``route_strategy="auto"`` rule (``legacy_auto_strategy``) against the current
one, in both the token layout and the per-route layout the down projection uses.
Run it across M=1 and M=32 as well as the verify widths -- those two are the
controls, where old and new resolve to the same strategy and the ratio should
read ~1.0. A number far from 1.0 there is measuring machine noise, not the flip,
and calibrates how much of the in-window ratio to believe.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from keep.vq.e8 import e8_1bit_packed
from ramp.kernels.gather_vqmm import (
    gather_vqmm_m1_kernel_unchecked,
    gather_vqmm_verify_mrows_kernel,
    m1_rows_per_threadgroup,
)
from ramp.ops.vq_switch import (
    _flat_route_descriptors,
    _token_route_lhs,
    gather_vqmm,
    gather_vqmm_verify_rows,
)

# DeepSeek-V4-Flash routed experts, measured 2026-08-11: gate/up are
# K=4096 -> N=2048 and down expands N=2048 -> 4096, 256 experts, top-6.
V4_SHAPES = (
    ("gate", 4096, 2048),
    ("up", 4096, 2048),
    ("down", 2048, 4096),
)


def legacy_auto_strategy(route_count: int, *, sorted_indices: bool = False) -> str:
    """The ``route_strategy="auto"`` rule as it stood before Wave 4 increment 2.

    Increment 2 inserted a ``per_route_decoded`` arm between the sorted_tiled
    test and the ``direct`` fallback, so post-change ``"auto"`` can no longer be
    used to measure the pre-change decision. Naming the old rule here lets one
    process time old-vs-new honestly at every M, including the widths where the
    two agree and the measurement should therefore read ~1.0x.

    Note this is not the same thing as always passing ``"direct"``: at
    ``route_count >= 128`` the old rule already chose ``sorted_tiled``, so a
    ``"direct"`` stand-in would manufacture a fake regression at M >= 22.
    """

    return "sorted_tiled" if (sorted_indices or route_count >= 128) else "direct"


def _samples_ms(fn, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _latency_ms(fn, *, warmup: int, iterations: int) -> float:
    return statistics.median(_samples_ms(fn, warmup=warmup, iterations=iterations))


def _amortized_ms(fn, *, repeat: int, warmup: int, iterations: int) -> float:
    samples = _samples_ms(
        lambda: [fn() for _ in range(repeat)], warmup=warmup, iterations=iterations
    )
    return min(samples) / repeat


def _route_indices(
    rng: np.random.Generator,
    *,
    pattern: str,
    verify_rows: int,
    top_k: int,
    experts: int,
) -> np.ndarray:
    """Verify-row routing patterns, in increasing same-expert overlap."""

    if pattern == "uniform":
        return rng.integers(0, experts, size=(verify_rows, top_k), dtype=np.int32)
    if pattern == "skewed":
        # Draft tokens biased toward a hot slice of the expert table.
        hot = max(top_k, experts // 16)
        return rng.integers(0, hot, size=(verify_rows, top_k), dtype=np.int32)
    if pattern == "duplicate":
        # Every verify row routes to the same experts: the maximum-overlap case
        # expert grouping is built for.
        base = rng.choice(experts, size=(top_k,), replace=False).astype(np.int32)
        return np.tile(base, (verify_rows, 1))
    raise ValueError(f"unknown routing pattern {pattern!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark wide-M VQ verify kernels against M x (M=1) decode launches."
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--verify-rows", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["uniform", "skewed", "duplicate"],
        choices=["uniform", "skewed", "duplicate"],
    )
    parser.add_argument("--repeat", type=int, default=32, help="calls per eval for the amortized timing")
    parser.add_argument("--iterations", type=int, default=15,
                        help="latency samples; the amortized timing takes max(5, iterations // 2)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--amortized-warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--include-current-dispatch", action="store_true",
                        help="also time gather_vqmm(route_strategy='auto'), the pre-wave-4 path")
    parser.add_argument(
        "--include-dispatch-flip", action="store_true",
        help="time the increment-2 dispatch flip: old auto rule vs new auto, in both "
             "the token layout and the per-route (down-projection) layout",
    )
    parser.add_argument(
        "--skip-verify-variants", action="store_true",
        help="drop the increment-1 verify kernel/op variants and keep only the looped "
             "M=1 baseline, for a lean dispatch-flip run",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="write the final JSON document here as well as to stdout, so a committed "
             "artifact does not have to be recovered from interleaved progress lines",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    codebook = mx.array(e8_1bit_packed())
    mx.eval(codebook)
    rows: list[dict[str, object]] = []

    for proj, in_dim, out_dim in V4_SHAPES:
        if in_dim % args.group_size != 0:
            raise SystemExit(f"--group-size must divide in_dim {in_dim}")
        codes = mx.array(
            rng.integers(0, 256, size=(args.experts, out_dim, in_dim // 8), dtype=np.uint8)
        )
        scales = mx.array(
            rng.uniform(0.01, 0.05, size=(args.experts, out_dim, in_dim // args.group_size)).astype(
                np.float16
            )
        )
        mx.eval(codes, scales)

        for verify_rows in args.verify_rows:
            for pattern in args.patterns:
                rhs_np = _route_indices(
                    rng,
                    pattern=pattern,
                    verify_rows=verify_rows,
                    top_k=args.top_k,
                    experts=args.experts,
                )
                x = mx.array(rng.normal(size=(verify_rows, in_dim)).astype(np.float16))
                rhs = mx.array(rhs_np)
                # Pre-materialize the per-row slices so the looped baseline is
                # not charged for slicing it would not do in a real verify loop.
                x_rows = [mx.array(np.array(x[i : i + 1])) for i in range(verify_rows)]
                rhs_rows = [mx.array(rhs_np[i : i + 1]) for i in range(verify_rows)]
                mx.eval(x, rhs, *x_rows, *rhs_rows)

                # Both sides take the row packing from the helper the M=1 decode
                # path uses, so baseline and variant stay comparable if it is
                # ever tuned off 32.
                packing = m1_rows_per_threadgroup(in_dim, out_dim)

                def looped_m1() -> list[mx.array]:
                    return [
                        gather_vqmm_m1_kernel_unchecked(
                            x_rows[i],
                            codes,
                            scales,
                            codebook,
                            rhs_rows[i],
                            output_dims=out_dim,
                            code_bits=8,
                            rows_per_threadgroup=packing,
                            use_threadgroup_codebook=False,
                            use_decoded_codebook=True,
                        )
                        for i in range(verify_rows)
                    ]

                def verify_call(grouping: str) -> mx.array:
                    return gather_vqmm_verify_rows(
                        x,
                        codes,
                        scales,
                        codebook,
                        rhs,
                        input_dims=in_dim,
                        output_dims=out_dim,
                        group_size=args.group_size,
                        grouping=grouping,
                    )

                # Raw-kernel form of the flat variant: descriptors prebuilt and
                # validation off, so the number is the kernel and nothing else.
                route_count = verify_rows * args.top_k
                flat_rhs = rhs.reshape((-1,))
                flat_lhs = _token_route_lhs(route_count, args.top_k)
                flat_offsets, flat_counts = _flat_route_descriptors(route_count)
                mx.eval(flat_rhs, flat_lhs, flat_offsets, flat_counts)

                def verify_flat_kernel() -> mx.array:
                    return gather_vqmm_verify_mrows_kernel(
                        x,
                        codes,
                        scales,
                        codebook,
                        flat_lhs,
                        flat_rhs,
                        flat_offsets,
                        flat_counts,
                        route_count=route_count,
                        input_dims=in_dim,
                        output_dims=out_dim,
                        group_size=args.group_size,
                        m_rows=1,
                        rows_per_threadgroup=packing,
                        validate=False,
                    )

                timings = {"baseline_looped_m1": looped_m1}
                if not args.skip_verify_variants:
                    timings.update(
                        {
                            "verify_flat_kernel": verify_flat_kernel,
                            "verify_flat_op": lambda: verify_call("flat"),
                            "verify_expert_op": lambda: verify_call("expert"),
                        }
                    )
                row_dispatch_meta: dict[str, object] = {}
                if args.include_dispatch_flip:
                    legacy = legacy_auto_strategy(route_count)

                    # Per-route activation layout: one activation row per route,
                    # which is what the verify down projection passes. x is
                    # [routes, in_dim] and lhs_indices is the identity map.
                    x_per_route = mx.array(
                        rng.normal(size=(route_count, in_dim)).astype(np.float16)
                    )
                    per_route_lhs = mx.arange(route_count, dtype=mx.int32)
                    mx.eval(x_per_route, per_route_lhs)

                    def token_dispatch(strategy: str) -> mx.array:
                        return gather_vqmm(
                            x, codes, scales, codebook, rhs,
                            input_dims=in_dim, output_dims=out_dim,
                            group_size=args.group_size, code_bits=8,
                            route_strategy=strategy,
                        )

                    def per_route_dispatch(strategy: str) -> mx.array:
                        return gather_vqmm(
                            x_per_route, codes, scales, codebook, flat_rhs,
                            lhs_indices=per_route_lhs,
                            input_dims=in_dim, output_dims=out_dim,
                            group_size=args.group_size, code_bits=8,
                            route_strategy=strategy,
                        )

                    timings["old_dispatch"] = lambda s=legacy: token_dispatch(s)
                    timings["new_dispatch"] = lambda: token_dispatch("auto")
                    timings["old_dispatch_per_route"] = lambda s=legacy: per_route_dispatch(s)
                    timings["new_dispatch_per_route"] = lambda: per_route_dispatch("auto")
                    # Reported per layout, because they differ at M=1: the token
                    # layout has an M=1 fast path inside gather_vqmm and the
                    # per-route layout does not, so a single verify row flips in
                    # the per-route layout while staying put in the token layout.
                    row_dispatch_meta = {
                        "legacy_auto_strategy": legacy,
                        "new_auto_strategy_token": (
                            "sorted_tiled" if route_count >= 128
                            else "m1_fast_path" if verify_rows == 1
                            else "per_route_decoded"
                        ),
                        "new_auto_strategy_per_route": (
                            "sorted_tiled" if route_count >= 128 else "per_route_decoded"
                        ),
                    }
                if args.include_current_dispatch:
                    timings["current_dispatch_auto"] = lambda: gather_vqmm(
                        x,
                        codes,
                        scales,
                        codebook,
                        rhs,
                        input_dims=in_dim,
                        output_dims=out_dim,
                        group_size=args.group_size,
                        code_bits=8,
                    )

                # Bytes the kernel must move, so the reported GB/s is derivable
                # from committed code rather than asserted in prose. Codes
                # dominate: one uint8 per 8D codeword per output row, read once
                # per *distinct* expert (a second route on the same expert hits
                # cache). Scales and activations are included for completeness.
                distinct_experts = int(len(set(rhs_np.reshape(-1).tolist())))
                code_bytes = distinct_experts * out_dim * (in_dim // 8)
                scale_bytes = distinct_experts * out_dim * (in_dim // args.group_size) * 2
                activation_bytes = verify_rows * in_dim * 2
                output_bytes = verify_rows * args.top_k * out_dim * 2
                bytes_moved = code_bytes + scale_bytes + activation_bytes + output_bytes

                row: dict[str, object] = {
                    "proj": proj,
                    "K": in_dim,
                    "N": out_dim,
                    "verify_rows": verify_rows,
                    "routes": verify_rows * args.top_k,
                    "pattern": pattern,
                    "distinct_experts": distinct_experts,
                    "code_bytes": code_bytes,
                    "bytes_moved": bytes_moved,
                    **row_dispatch_meta,
                }
                for label, fn in timings.items():
                    row[f"{label}_latency_ms"] = round(
                        _latency_ms(fn, warmup=args.warmup, iterations=args.iterations), 4
                    )
                    amortized_ms = _amortized_ms(
                        fn,
                        repeat=args.repeat,
                        warmup=args.amortized_warmup,
                        iterations=max(5, args.iterations // 2),
                    )
                    row[f"{label}_amortized_ms"] = round(amortized_ms, 4)
                    row[f"{label}_gbps"] = round(bytes_moved / 1e9 / (amortized_ms / 1e3), 1)
                    row[f"{label}_code_gbps"] = round(
                        code_bytes / 1e9 / (amortized_ms / 1e3), 1
                    )
                base_lat = row["baseline_looped_m1_latency_ms"]
                base_amo = row["baseline_looped_m1_amortized_ms"]
                for label in timings:
                    if label == "baseline_looped_m1":
                        continue
                    row[f"{label}_latency_speedup"] = round(base_lat / row[f"{label}_latency_ms"], 3)
                    row[f"{label}_amortized_speedup"] = round(
                        base_amo / row[f"{label}_amortized_ms"], 3
                    )
                # The increment-2 headline: old auto rule / new auto, per layout.
                # Expected ~1.0 at M=1 (both take the M=1 fast path) and at
                # M >= 22 (both take sorted_tiled); the win lives in between.
                for layout in ("", "_per_route"):
                    old_key = f"old_dispatch{layout}_amortized_ms"
                    new_key = f"new_dispatch{layout}_amortized_ms"
                    if old_key in row and new_key in row:
                        row[f"dispatch_flip{layout}_amortized_speedup"] = round(
                            row[old_key] / row[new_key], 3
                        )
                        row[f"dispatch_flip{layout}_latency_speedup"] = round(
                            row[f"old_dispatch{layout}_latency_ms"]
                            / row[f"new_dispatch{layout}_latency_ms"],
                            3,
                        )
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

        del codes, scales

    document = json.dumps(
        {
            "device": str(mx.device_info().get("architecture", "unknown")),
            "experts": args.experts,
            "top_k": args.top_k,
            "group_size": args.group_size,
            "repeat": args.repeat,
            "iterations": args.iterations,
            "amortized_warmup": args.amortized_warmup,
            "seed": args.seed,
            "rows": rows,
        },
        indent=2,
        sort_keys=True,
    )
    print(document)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(document + "\n")


if __name__ == "__main__":
    main()
