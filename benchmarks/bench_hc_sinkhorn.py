"""Kill test: the one-dispatch hyper-connection sinkhorn against the compiled path.

Times the model's compiled ``_hc_split_sinkhorn_ops`` against the logits half
plus :func:`ramp.kernels.hc_sinkhorn.fused_hc_sinkhorn`, at decode shape (one
row) and a prefill chunk, and reports milliseconds per DeepSeek-V4-Flash
decode step (86 calls: two hyper-connections in each of 43 blocks).

KILL if the fused path is less than 3x faster at decode shape, or if it
differs from the compiled path by more than 2e-6 relative. If it passes, set
``RAMP_HC_SINKHORN=metal`` for the decode headline.

    uv run python benchmarks/bench_hc_sinkhorn.py --output-json artifacts/benchmarks/hc-sinkhorn-m4max.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np

from ramp.kernels.hc_sinkhorn import fused_hc_sinkhorn
from ramp.models.deepseek_v4_flash_adapter import _hc_split_logits_ops, _hc_split_sinkhorn_ops

CALLS_PER_DECODE_STEP = 86
MIN_DECODE_SPEEDUP = 3.0
MAX_RELATIVE_DIFF = 2e-6


def _per_call_ms(fn, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(fn())
    return (time.perf_counter() - started) * 1e3 / iterations


def measure(rows: int, *, hc: int, iters: int, eps: float, iterations: int, warmup: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mix = (2 + hc) * hc
    mixes = mx.array(rng.normal(size=(1, rows, mix)).astype(np.float32) * 3)
    scale = mx.array(rng.uniform(0.5, 2.0, size=(3,)).astype(np.float32))
    base = mx.array(rng.normal(size=(mix,)).astype(np.float32))

    def compiled():
        return _hc_split_sinkhorn_ops(mixes, scale, base, hc, iters, eps)

    def fused():
        pre, post, logits = _hc_split_logits_ops(mixes, scale, base, hc, eps)
        return pre, post, fused_hc_sinkhorn(logits, iters=iters, eps=eps)

    want, got = np.array(compiled()[2]), np.array(fused()[2])
    rel = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)))
    t_compiled = _per_call_ms(compiled, iterations, warmup)
    t_fused = _per_call_ms(fused, iterations, warmup)
    return {
        "rows": rows,
        "compiled_ms_per_call": t_compiled,
        "fused_ms_per_call": t_fused,
        "speedup": t_compiled / t_fused,
        "compiled_ms_per_decode_step": t_compiled * CALLS_PER_DECODE_STEP,
        "fused_ms_per_decode_step": t_fused * CALLS_PER_DECODE_STEP,
        "max_relative_diff": rel,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default="1,1024", help="matrices per call (1 = decode)")
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    on_gpu = mx.default_device() == mx.gpu
    rows = [
        measure(
            int(r),
            hc=args.hc_mult,
            iters=args.iters,
            eps=args.eps,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )
        for r in args.rows.split(",")
    ]
    decode = next((row for row in rows if row["rows"] == 1), rows[0])
    reasons = []
    if any(row["max_relative_diff"] > MAX_RELATIVE_DIFF for row in rows):
        reasons.append(f"fused output differs by more than {MAX_RELATIVE_DIFF} relative")
    if decode["speedup"] < MIN_DECODE_SPEEDUP:
        reasons.append(f"decode-shape speedup {decode['speedup']:.2f}x is under {MIN_DECODE_SPEEDUP}x")
    result = {
        "record_type": "ramp_hc_sinkhorn_kill_test_v1",
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "rows": rows,
        "verdict": "KILL" if reasons else ("PASS" if on_gpu else "NOT_ON_GPU"),
        "reasons": reasons,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 1 if result["verdict"] == "KILL" else 0


if __name__ == "__main__":
    sys.exit(main())
