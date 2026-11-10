"""K2: one decode-step MoE at V4.1 shape, by routed-expert format, in one process.

- oMLX's fused decode kernels (``decode_moe.fused_decode_moe``: router, top-k routed experts and
  the shared expert in three dispatches) with routed experts as MXFP4 (the checkpoint's own
  format) and as oMLX's affine codes at 2 and 3 bits, group 64. Random weights.
- This repo's E8P lattice kernel (``ramp.kernels.e8p_m1``; routed experts only, no router and
  no shared expert) on real 2-bit codes of one layer.

Each sample chains ``--chain`` dependent MoEs before one ``mx.eval``, as a decode step does.
Running each at top-k 6 and top-k 1 separates the cost per routed expert from the fixed cost.

    PYTHONPATH=src:benchmarks ~/Development/_refs/omlx/.venv/bin/python \
        scripts/v41_build/bench_expert_formats.py --config ~/Models/v41-reap50/config.json \
        --e8p ~/Models/v41-reap-run/quant/e8p-podimp/layer-20.npz
"""

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

EXPERTS = 64


def median_ms(step, chain, iterations=30, warmup=10):
    def sample():
        started = time.perf_counter()
        step(chain)
        return (time.perf_counter() - started) * 1e3 / chain

    for _ in range(warmup):
        sample()
    return statistics.median(sample() for _ in range(iterations))


def omlx_moe(config, mode, bits, group, rng):
    from omlx.patches.deepseek_v41.language import MoE
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    def projection(out, inp, lead, mode, bits, group):
        weight = mx.array(rng.integers(0, 1 << 32, size=(*lead, out, inp * bits // 32), dtype=np.uint32))
        if mode == "affine":
            shape = (*lead, out, inp // group)
            return QuantizedProjection(weight, mx.full(shape, 0.01, mx.bfloat16), bits, mode,
                                       biases=mx.full(shape, -0.005 * (2**bits - 1), mx.bfloat16), group_size=group)
        return QuantizedProjection(weight, mx.full((*lead, out, inp // 32), 118, mx.uint8), bits, mode)  # 2^-9

    moe = MoE(config)
    dim, inter = config.dim, config.moe_inter_dim
    for name, out, inp in (("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)):
        setattr(moe.experts, name, projection(out, inp, (EXPERTS,), mode, bits, group))
        setattr(moe.shared_experts, name, projection(out, inp, (), "mxfp8", 8, 32))
    moe.gate.weight = mx.array(rng.normal(size=(EXPERTS, dim)).astype(np.float32)).astype(mx.bfloat16)
    moe.gate.bias = mx.zeros((EXPERTS,))
    arrays = [moe.gate.weight, moe.gate.bias]
    nbytes = 0
    for name in ("w1", "w3", "w2"):
        p, s = getattr(moe.experts, name), getattr(moe.shared_experts, name)
        fields = [p.weight, p.scales] + ([p.biases] if "biases" in p else [])
        arrays += fields + [s.weight, s.scales]
        nbytes += sum(a.nbytes for a in fields) // EXPERTS
    mx.eval(arrays)
    return moe, nbytes


def bench_omlx(config, mode, bits, group, top_k, chain, seed=0):
    from omlx.patches.deepseek_v41 import decode_moe

    rng = np.random.default_rng(seed)
    c = replace(config, n_routed_experts=EXPERTS, n_activated_experts=top_k)
    moe, expert_bytes = omlx_moe(c, mode, bits, group, rng)
    x0 = mx.array(rng.normal(size=(1, 1, c.dim)).astype(np.float32)).astype(mx.bfloat16)
    assert decode_moe.supported(moe, x0, c), (mode, bits, group)
    state = {"x": x0}

    def step(n):
        x = state["x"]
        for _ in range(n):
            out = decode_moe.fused_decode_moe(moe, x, c)[0].astype(mx.float32)
            x = (out / (mx.max(mx.abs(out)) + 1e-6) * 3).astype(mx.bfloat16)  # a new input, so new routes
        mx.eval(x)
        state["x"] = x

    ms = median_ms(step, chain)
    assert bool(mx.all(mx.isfinite(state["x"].astype(mx.float32)))), "non-finite activations"
    return ms, expert_bytes


def bench_e8p(path, top_k, chain, sweep=((4, 1), (8, 1), (4, 4), (2, 4))):
    from bench_e8p_m1 import PROJECTIONS, projection_bytes_per_expert, run
    from keep.vq.e8 import e8p_packed_abs_grid
    from ramp.nn.switch_linear import QuantizedVQSwitchLinear

    z, layers = np.load(path), {}
    for name, key in zip(PROJECTIONS, ("w1", "w3", "w2")):
        codes, scales = z[f"{key}_codes"][:EXPERTS], z[f"{key}_scales"][:EXPERTS]
        layers[name] = QuantizedVQSwitchLinear(
            input_dims=codes.shape[2] * 8, output_dims=codes.shape[1], num_experts=EXPERTS,
            codes=mx.array(codes), scales=mx.array(scales), codebook=mx.array(e8p_packed_abs_grid()),
            group_size=codes.shape[2] * 8 // scales.shape[2], code_bits=16)
    mx.eval([layer.parameters() for layer in layers.values()])
    result = run(layers, top_k=top_k, iterations=30, warmup=10, seed=0, sweep=sweep, ffns_per_sync=chain)
    assert result["outputs_agree"], result["configs"]
    best = result["best"]
    return (best["median_ms_per_ffn"], sum(projection_bytes_per_expert(layers[p]) for p in PROJECTIONS),
            (best["rows_per_simdgroup"], best["simdgroups"]), result["production_median_ms_per_ffn"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--e8p", type=Path, required=True)
    p.add_argument("--chain", type=int, default=40)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    from omlx.patches.deepseek_v41.config import ModelConfig

    config = ModelConfig.from_dict(json.loads(args.config.expanduser().read_text()))
    rows = []
    for label, mode, bits, group in (("omlx mxfp4", "mxfp4", 4, 32), ("omlx affine 2-bit g64", "affine", 2, 64),
                                     ("omlx affine 3-bit g64", "affine", 3, 64)):
        (ms6, nbytes), (ms1, _) = (bench_omlx(config, mode, bits, group, k, args.chain) for k in (6, 1))
        rows.append({"format": label, "ms_top6": ms6, "ms_top1": ms1, "expert_mb": nbytes / 1e6})
        mx.clear_cache()
    (ms6, nbytes, best, production), (ms1, *_) = (bench_e8p(args.e8p.expanduser(), k, args.chain) for k in (6, 1))
    rows.append({"format": "e8p lattice 2-bit (routed experts only)", "ms_top6": ms6, "ms_top1": ms1,
                 "expert_mb": nbytes / 1e6, "rows_per_simdgroup,simdgroups": best, "generic_kernel_ms_top6": production})
    for row in rows:
        row["ms_per_expert"] = (row["ms_top6"] - row["ms_top1"]) / 5
        row["ms_fixed"] = row["ms_top1"] - row["ms_per_expert"]
        row["expert_gb_per_s"] = row["expert_mb"] / 1e3 / (row["ms_per_expert"] / 1e3)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
    if args.out:
        args.out.write_text(json.dumps({"mlx": mx.__version__, "chain": args.chain, "experts": EXPERTS, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
