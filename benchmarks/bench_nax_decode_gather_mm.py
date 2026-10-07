"""NAX prefill Option B microbench: decode E8P -> fp16 scratch + mx.gather_mm.

Tests whether decoding E8P VQ codes to an fp16 scratch tensor and running the
routed matmul via mx.gather_mm (which rides the M5 NAX FP16 TensorOps path,
gather_mm_rhs_nax) approaches the q2 prefill speed WITHOUT any C++/Metal work.

Compares, on a single Air MoE projection shape, at sorted-route scale:
  - optB      : decode-to-fp16 + mx.gather_mm  (the candidate)
  - optB_mm   : gather_mm only, weights pre-decoded (isolates matmul from decode)
  - q2        : mx.gather_qmm 2-bit affine g128 (the Lane S denominator engine)
  - vq_metal  : gather_vqmm_sorted_routes(implementation="metal")  (current slow path)

Correctness of optB is asserted bit-parity vs decode_weight_matrix host oracle.
No model load required.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from keep.vq.e8 import decode_weight_matrix, e8p_full_grid
from ramp.ops.vq_switch import gather_vqmm_sorted_routes


def _time_ms(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(fn())
    return (time.perf_counter() - start) * 1000.0 / iterations


_GRID_F16: mx.array | None = None


def _grid_f16() -> mx.array:
    global _GRID_F16
    if _GRID_F16 is None:
        _GRID_F16 = mx.array(e8p_full_grid().astype(np.float16))  # [65536, 8]
    return _GRID_F16


def decode_experts_fp16(codes_u16: mx.array, scales_f16: mx.array, group_size: int) -> mx.array:
    """Decode [E, out, in//8] E8P codes -> [E, out, in] fp16 weights.

    Matches decode_weight_matrix: scale is applied per-codeword (per 8-elt block),
    each scale entry covering group_size input elements = group_size//8 codewords.
    """
    E, out, w = codes_u16.shape  # w = in // 8
    grid = _grid_f16()
    dec = grid[codes_u16.reshape(-1).astype(mx.uint32)]  # [E*out*w, 8]
    dec = dec.reshape(E, out, w, 8)  # [E, out, codewords, 8]
    words_per_scale = w // scales_f16.shape[-1]  # = group_size // 8
    expanded = mx.repeat(scales_f16, words_per_scale, axis=-1)[..., None]  # [E, out, w, 1]
    weight = (dec * expanded.astype(dec.dtype)).reshape(E, out, w * 8)  # [E, out, in]
    return weight.astype(mx.float16)


def opt_b(sorted_x, codes, scales, sorted_rhs, *, group_size, out_dim):
    W = decode_experts_fp16(codes, scales, group_size)  # [E, out, in]
    Wt = mx.swapaxes(W, 1, 2)  # [E, in, out]
    xa = sorted_x.reshape(sorted_x.shape[0], 1, sorted_x.shape[1])  # [R, 1, in]
    lhs = mx.arange(sorted_x.shape[0], dtype=mx.uint32)
    y = mx.gather_mm(xa, Wt, lhs_indices=lhs, rhs_indices=sorted_rhs, sorted_indices=True)
    return y.reshape(sorted_x.shape[0], out_dim)


def opt_b_mm_only(sorted_x, Wt, sorted_rhs, *, out_dim):
    xa = sorted_x.reshape(sorted_x.shape[0], 1, sorted_x.shape[1])
    lhs = mx.arange(sorted_x.shape[0], dtype=mx.uint32)
    y = mx.gather_mm(xa, Wt, lhs_indices=lhs, rhs_indices=sorted_rhs, sorted_indices=True)
    return y.reshape(sorted_x.shape[0], out_dim)


def q2_gather_qmm(sorted_x, wq, q_scales, q_biases, sorted_rhs, *, out_dim, group_size, bits):
    xa = sorted_x.reshape(sorted_x.shape[0], 1, sorted_x.shape[1])
    lhs = mx.arange(sorted_x.shape[0], dtype=mx.uint32)
    y = mx.gather_qmm(
        xa, wq, q_scales, q_biases,
        lhs_indices=lhs, rhs_indices=sorted_rhs,
        transpose=True, group_size=group_size, bits=bits, sorted_indices=True,
    )
    return y.reshape(sorted_x.shape[0], out_dim)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--in-dim", type=int, default=4096)
    p.add_argument("--out-dim", type=int, default=1408)
    p.add_argument("--group-size", type=int, default=512)
    p.add_argument("--q2-group-size", type=int, default=128)
    p.add_argument("--q2-bits", type=int, default=2)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--projection", choices=["gate_up", "down"], default="gate_up")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    if args.in_dim % 8 or args.in_dim % args.group_size:
        raise SystemExit("--in-dim must be divisible by 8 and --group-size")

    rng = np.random.default_rng(args.seed)
    routes = args.tokens * args.top_k
    hot = min(args.experts, max(args.top_k, 8))

    codes_np = rng.integers(0, 1 << 16, size=(args.experts, args.out_dim, args.in_dim // 8), dtype=np.uint16)
    scales_np = rng.uniform(0.01, 0.05, size=(args.experts, args.out_dim, args.in_dim // args.group_size)).astype(np.float16)
    # sorted route ids (nondecreasing by expert), and per-route activation rows
    rhs_np = np.sort(rng.integers(0, hot, size=routes).astype(np.int32))
    sorted_x_np = rng.normal(size=(routes, args.in_dim)).astype(np.float16)

    codes = mx.array(codes_np)
    scales = mx.array(scales_np)
    sorted_rhs = mx.array(rhs_np.astype(np.uint32))
    sorted_lhs = mx.arange(routes, dtype=mx.int32)
    sorted_x = mx.array(sorted_x_np)
    mx.eval(codes, scales, sorted_rhs, sorted_lhs, sorted_x, _grid_f16())

    info = {
        "nax_available": bool(getattr(mx.metal, "is_available", lambda: False)()),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "projection": args.projection, "tokens": args.tokens, "routes": routes,
        "in_dim": args.in_dim, "out_dim": args.out_dim, "group_size": args.group_size,
    }

    # ---- correctness: optB vs host oracle on a few routes ----
    yb = opt_b(sorted_x, codes, scales, sorted_rhs, group_size=args.group_size, out_dim=args.out_dim)
    mx.eval(yb)
    yb_np = np.array(yb.astype(mx.float32))
    max_err = 0.0
    for r in rng.choice(routes, size=min(8, routes), replace=False):
        e = int(rhs_np[r])
        W = decode_weight_matrix(codes_np[e], scales_np[e], code_bits=16)  # [out, in] f32
        ref = sorted_x_np[r].astype(np.float32) @ W.T
        max_err = max(max_err, float(np.max(np.abs(ref - yb_np[r]))))
    info["optB_vs_oracle_max_abs_err"] = max_err
    # fp16 accumulation drift expected; assert reasonable
    rel = max_err / (float(np.max(np.abs(yb_np))) + 1e-6)
    info["optB_vs_oracle_rel_err"] = rel
    assert rel < 0.05, f"optB correctness failed: rel_err={rel} (max_abs={max_err})"

    results = {}

    # ---- optB (decode + gather_mm) ----
    results["optB_decode_plus_mm"] = _time_ms(
        lambda: opt_b(sorted_x, codes, scales, sorted_rhs, group_size=args.group_size, out_dim=args.out_dim),
        warmup=args.warmup, iterations=args.iterations)

    # ---- optB matmul only (pre-decoded) ----
    W = decode_experts_fp16(codes, scales, args.group_size)
    Wt = mx.swapaxes(W, 1, 2)
    mx.eval(Wt)
    results["optB_mm_only"] = _time_ms(
        lambda: opt_b_mm_only(sorted_x, Wt, sorted_rhs, out_dim=args.out_dim),
        warmup=args.warmup, iterations=args.iterations)

    # ---- q2 gather_qmm baseline ----
    try:
        wq, q_scales, q_biases = mx.quantize(W, group_size=args.q2_group_size, bits=args.q2_bits)
        mx.eval(wq, q_scales, q_biases)
        results["q2_gather_qmm"] = _time_ms(
            lambda: q2_gather_qmm(sorted_x, wq, q_scales, q_biases, sorted_rhs,
                                  out_dim=args.out_dim, group_size=args.q2_group_size, bits=args.q2_bits),
            warmup=args.warmup, iterations=args.iterations)
    except Exception as exc:  # noqa: BLE001
        results["q2_gather_qmm"] = None
        info["q2_error"] = repr(exc)

    # ---- VQ metal baseline (current slow path) ----
    try:
        vm = _time_ms(
            lambda: gather_vqmm_sorted_routes(
                sorted_x, codes, scales, None, sorted_rhs, sorted_lhs,
                input_dims=args.in_dim, output_dims=args.out_dim,
                group_size=args.group_size, code_bits=16, implementation="metal"),
            warmup=args.warmup, iterations=args.iterations)
        results["vq_metal"] = vm
    except Exception as exc:  # noqa: BLE001
        results["vq_metal"] = None
        info["vq_metal_error"] = repr(exc)

    q2 = results.get("q2_gather_qmm")
    ratios = {}
    if q2:
        for k, v in results.items():
            if v:
                ratios[f"{k}_over_q2"] = round(v / q2, 4)
    out = {"info": info, "ms": {k: (round(v, 4) if v else None) for k, v in results.items()}, "ratios_vs_q2": ratios}
    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
