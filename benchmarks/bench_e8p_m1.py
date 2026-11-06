"""Kill test: single-token E8P (16-bit codes) on a GPU without the NAX kernels.

Times one routed FFN at decode shape (gate, up and down for top-6 experts) two
ways in the same process:

- ``production``: today's dispatch (the scalar kernel on M1-M4; NAX on M5);
- ``e8p_m1``: ``RAMP_E8P_M1=1``, the M=1 E8P kernels, swept over row packing
  and threadgroup codebook staging.

Effective bandwidth is the expert bytes one FFN reads (6 experts x 3
projections of 2-bit codes plus their scales) over the median time per FFN.
Each timed sample chains ``--ffns-per-sync`` dependent FFNs (default 43, one
per V4-Flash layer) before one ``mx.eval``, as one decode step does; syncing
after every FFN measured mostly CPU-GPU round trips (v1 rows). Verdict
for the best ``e8p_m1`` configuration: **PASS** at >= 200 GB/s, **TUNE** at
100-200 (tune; decoding each codeword once was tried and is slower),
**KILL** under 100 (serve on the M5 instead). Outputs of the two paths are
compared on every configuration.

Synthetic codes at DeepSeek-V4-Flash shapes need no download; ``--artifact-dir``
uses one real layer of the shipped artifact instead::

    RAMP_E8P_M1=0 uv run python benchmarks/bench_e8p_m1.py --output-json artifacts/benchmarks/e8p-m1-m4max.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np

from keep.vq.e8 import e8p_packed_abs_grid
from ramp.kernels import e8p_m1
from ramp.nn.switch_linear import QuantizedVQSwitchLinear

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PASS_GB_S = 200.0
KILL_GB_S = 100.0


def synthetic_projection(*, experts: int, out: int, inp: int, group_size: int, seed: int) -> QuantizedVQSwitchLinear:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 1 << 16, size=(experts, out, inp // 8), dtype=np.uint16)
    scales = rng.uniform(0.01, 0.05, size=(experts, out, inp // group_size)).astype(np.float16)
    return QuantizedVQSwitchLinear(
        input_dims=inp,
        output_dims=out,
        num_experts=experts,
        codes=mx.array(codes),
        scales=mx.array(scales),
        codebook=mx.array(e8p_packed_abs_grid()),
        group_size=group_size,
        code_bits=16,
    )


def projection_bytes_per_expert(layer: QuantizedVQSwitchLinear) -> int:
    per_expert = layer.codes.nbytes // layer.num_experts + layer.scales.nbytes // layer.num_experts
    return int(per_expert)


def _ffn(layers, x, routes, intermediate):
    gate = layers["gate_proj"](x, routes)
    up = layers["up_proj"](x, routes)
    act = (gate * mx.sigmoid(gate) * up).astype(x.dtype)  # [1, k, I]
    return layers["down_proj"](act.reshape(1, routes.shape[-1], intermediate), routes)


def _chain(layers, x, routes_seq, intermediate):
    """Dependent FFNs, each feeding the next like consecutive decoder layers."""

    for routes in routes_seq:
        y = _ffn(layers, x, routes, intermediate)  # [1, k, hidden]
        x = (x + y.sum(axis=1) * 1e-3).astype(x.dtype)
    return x


def _median_ms(layers, x, routes_list, intermediate, iterations, warmup, ffns_per_sync) -> float:
    """Median ms per FFN, with ``ffns_per_sync`` chained FFNs per ``mx.eval``."""

    def sample(i):
        seq = [routes_list[(i * ffns_per_sync + j) % len(routes_list)] for j in range(ffns_per_sync)]
        started = time.perf_counter()
        mx.eval(_chain(layers, x, seq, intermediate))
        return (time.perf_counter() - started) * 1e3 / ffns_per_sync

    for i in range(warmup):
        sample(i)
    return statistics.median(sample(i) for i in range(iterations))


def run(
    layers,
    *,
    top_k: int,
    iterations: int,
    warmup: int,
    seed: int,
    sweep: Sequence[tuple[int, bool]],
    ffns_per_sync: int = 43,
) -> dict:
    rng = np.random.default_rng(seed)
    experts = layers["gate_proj"].num_experts
    hidden = layers["gate_proj"].input_dims
    intermediate = layers["gate_proj"].output_dims
    x = mx.array(rng.normal(size=(1, hidden)).astype(np.float32)).astype(mx.float16)
    # Fresh routes per call so consecutive calls do not reuse cached experts.
    routes_list = [
        mx.array(np.sort(rng.choice(experts, size=top_k, replace=False))[None].astype(np.uint32))
        for _ in range(64)
    ]
    ffn_bytes = top_k * sum(projection_bytes_per_expert(layers[p]) for p in PROJECTIONS)

    os.environ[e8p_m1.E8P_M1_ENV] = "0"
    production_ms = _median_ms(layers, x, routes_list, intermediate, iterations, warmup, ffns_per_sync)
    reference = [np.array(_ffn(layers, x, r, intermediate).astype(mx.float32)) for r in routes_list[:4]]

    configs = []
    os.environ[e8p_m1.E8P_M1_ENV] = "1"
    original_token, original_route = e8p_m1.e8p_m1_token_layout, e8p_m1.e8p_m1_per_route
    import ramp.nn.switch_linear as switch_linear

    try:
        for rows_per_tg, tg_codebook in sweep:
            switch_linear.e8p_m1_token_layout = (
                lambda *a, _r=rows_per_tg, _c=tg_codebook, **k: original_token(
                    *a, rows_per_threadgroup=_r, use_threadgroup_codebook=_c, **k
                )
            )
            switch_linear.e8p_m1_per_route = (
                lambda *a, _r=rows_per_tg, _c=tg_codebook, **k: original_route(
                    *a, rows_per_threadgroup=_r, use_threadgroup_codebook=_c, **k
                )
            )
            got = [np.array(_ffn(layers, x, r, intermediate).astype(mx.float32)) for r in routes_list[:4]]
            diff = max(
                float(np.max(np.abs(g - w)) / max(float(np.max(np.abs(w))), 1e-12))
                for g, w in zip(got, reference)
            )
            ms = _median_ms(layers, x, routes_list, intermediate, iterations, warmup, ffns_per_sync)
            configs.append(
                {
                    "rows_per_threadgroup": rows_per_tg,
                    "threadgroup_codebook": tg_codebook,
                    "median_ms_per_ffn": ms,
                    "effective_gb_per_s": ffn_bytes / (ms / 1e3) / 1e9,
                    "speedup_over_production": production_ms / ms,
                    "max_relative_diff_vs_production": diff,
                }
            )
    finally:
        switch_linear.e8p_m1_token_layout = original_token
        switch_linear.e8p_m1_per_route = original_route
        os.environ.pop(e8p_m1.E8P_M1_ENV, None)

    best = max(configs, key=lambda c: c["effective_gb_per_s"])
    agrees = all(c["max_relative_diff_vs_production"] <= 1e-2 for c in configs)
    gb_s = best["effective_gb_per_s"]
    verdict = "KILL" if not agrees or gb_s < KILL_GB_S else ("PASS" if gb_s >= PASS_GB_S else "TUNE")
    return {
        "ffn_bytes": ffn_bytes,
        "ffns_per_sync": ffns_per_sync,
        "production_median_ms_per_ffn": production_ms,
        "production_effective_gb_per_s": ffn_bytes / (production_ms / 1e3) / 1e9,
        "configs": configs,
        "best": best,
        "outputs_agree": agrees,
        "verdict": verdict if mx.default_device() == mx.gpu else "NOT_ON_GPU",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", type=Path, help="real E8P artifact; default synthetic")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--experts", type=int, default=64, help="synthetic experts per projection")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50, help="timed samples, each --ffns-per-sync FFNs")
    parser.add_argument("--ffns-per-sync", type=int, default=43, help="chained FFNs per mx.eval (V4-Flash has 43 layers)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--sweep", default="8:1,16:1,32:1,64:1,32:0", help="rows_per_threadgroup:tg_codebook,...")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    if args.artifact_dir is not None:
        from keep.io.load import load_quantized_vq_switch_linear

        layers = {
            p: load_quantized_vq_switch_linear(
                args.artifact_dir.expanduser() / f"layer-{args.layer:05d}-{p}.safetensors",
                f"model.layers.{args.layer}.ffn.switch_mlp.{p}",
            )
            for p in PROJECTIONS
        }
        if any(layer.code_bits != 16 for layer in layers.values()):
            parser.error("the artifact layer is not 16-bit E8P")
    else:
        layers = {
            "gate_proj": synthetic_projection(experts=args.experts, out=args.intermediate, inp=args.hidden, group_size=args.group_size, seed=1),
            "up_proj": synthetic_projection(experts=args.experts, out=args.intermediate, inp=args.hidden, group_size=args.group_size, seed=2),
            "down_proj": synthetic_projection(experts=args.experts, out=args.hidden, inp=args.intermediate, group_size=args.group_size, seed=3),
        }
    sweep = []
    for item in args.sweep.split(","):
        rows, codebook = item.split(":")
        sweep.append((int(rows), codebook.strip() == "1"))
    result = {
        "record_type": "ramp_e8p_m1_kill_test_v2",
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "source": "synthetic" if args.artifact_dir is None else str(args.artifact_dir),
        **run(
            layers,
            top_k=args.top_k,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
            sweep=sweep,
            ffns_per_sync=args.ffns_per_sync,
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 1 if result["verdict"] == "KILL" else 0


if __name__ == "__main__":
    sys.exit(main())
