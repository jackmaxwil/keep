from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np

from keep.vq.e8 import e8_1bit_packed
from ramp.kernels.vq_qmv import vq_qmv


@dataclass(frozen=True)
class BandwidthResult:
    name: str
    iterations: int
    total_seconds: float
    ms_per_iter: float
    bytes_per_iter: int
    gb_per_second: float


def _time_loop(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(fn())
    return time.perf_counter() - start


def measure_elementwise_bandwidth(mb: int, *, warmup: int, iterations: int) -> BandwidthResult:
    count = (mb * 1024 * 1024) // np.dtype(np.float32).itemsize
    a = mx.ones((count,), dtype=mx.float32)
    b = mx.ones((count,), dtype=mx.float32)
    elapsed = _time_loop(lambda: a + b, warmup=warmup, iterations=iterations)
    bytes_per_iter = count * np.dtype(np.float32).itemsize * 3
    return BandwidthResult(
        name="mlx_elementwise_add_read_read_write",
        iterations=iterations,
        total_seconds=elapsed,
        ms_per_iter=elapsed * 1000 / iterations,
        bytes_per_iter=bytes_per_iter,
        gb_per_second=(bytes_per_iter * iterations) / elapsed / 1e9,
    )


def measure_qmv(
    *,
    in_dim: int,
    out_dim: int,
    group_size: int,
    warmup: int,
    iterations: int,
    seed: int,
    codebook_duplication: int,
) -> BandwidthResult:
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(in_dim,)).astype(np.float32))
    codes_np = rng.integers(0, 256, size=(out_dim, in_dim // 8), dtype=np.uint8)
    scales_np = rng.uniform(0.9, 1.1, size=(out_dim, in_dim // group_size)).astype(np.float32)
    codes = mx.array(codes_np)
    scales = mx.array(scales_np)
    codebook = mx.array(e8_1bit_packed())

    def run():
        return vq_qmv(
            x,
            codes,
            scales,
            codebook,
            in_dim=in_dim,
            out_dim=out_dim,
            group_size=group_size,
            code_bits=8,
            codebook_duplication=codebook_duplication,
        )

    elapsed = _time_loop(run, warmup=warmup, iterations=iterations)

    code_bytes = codes_np.nbytes
    scale_bytes = scales_np.nbytes
    output_bytes = out_dim * np.dtype(np.float32).itemsize
    x_reread_bytes = out_dim * in_dim * np.dtype(np.float32).itemsize
    bytes_per_iter = code_bytes + scale_bytes + output_bytes + x_reread_bytes
    return BandwidthResult(
        name=f"vq_qmv_metal_kernel_read_estimate_dup{codebook_duplication}",
        iterations=iterations,
        total_seconds=elapsed,
        ms_per_iter=elapsed * 1000 / iterations,
        bytes_per_iter=bytes_per_iter,
        gb_per_second=(bytes_per_iter * iterations) / elapsed / 1e9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX VQ qmv kernel.")
    parser.add_argument("--in-dim", type=int, default=6144)
    parser.add_argument("--out-dim", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--bandwidth-mb", type=int, default=64)
    parser.add_argument("--duplications", default="1,4,8")
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    baseline = measure_elementwise_bandwidth(args.bandwidth_mb, warmup=args.warmup, iterations=args.iterations)
    duplications = [int(value) for value in args.duplications.split(",") if value]
    qmv = [
        measure_qmv(
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            group_size=args.group_size,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
            codebook_duplication=duplication,
        )
        for duplication in duplications
    ]
    print(json.dumps({"baseline": asdict(baseline), "qmv": [asdict(result) for result in qmv]}, indent=2))


if __name__ == "__main__":
    main()
