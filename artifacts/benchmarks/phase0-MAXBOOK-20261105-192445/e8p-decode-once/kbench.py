import time, statistics, sys
import numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from benchmarks.bench_e8p_m1 import synthetic_projection
from ramp.kernels import e8p_m1
from ramp.kernels.gather_vqmm import gather_vqmm_m1_kernel_unchecked

L = synthetic_projection(experts=64, out=2048, inp=4096, group_size=512, seed=1)
rng = np.random.default_rng(0)
x = mx.array(rng.normal(size=(1, 4096)).astype(np.float16))
routes = [mx.array(np.sort(rng.choice(64, 6, replace=False))[None].astype(np.uint32)) for _ in range(64)]
nbytes = 6 * (L.codes.nbytes + L.scales.nbytes) // 64
sc = L.scales

def old(r, rows, tg):
    return gather_vqmm_m1_kernel_unchecked(x, L.codes, sc, L.codebook, r, output_dims=2048, code_bits=16,
        rows_per_threadgroup=rows, use_threadgroup_codebook=tg, use_decoded_codebook=False)
def new(r, rows, tg):
    return e8p_m1.e8p_m1_token_layout(x, L.codes, sc, L.codebook, r, output_dims=2048,
        rows_per_threadgroup=rows, use_threadgroup_codebook=tg)
def prod(r, rows, tg):
    return L(x, r)

def once(fn, rows, tg):
    a = time.perf_counter(); mx.eval([fn(r, rows, tg) for r in routes * 4]); return (time.perf_counter() - a) / (len(routes) * 4)

cases = [("prod", prod, 0)] + [(n, f, r) for r in (4, 8, 16, 32) for n, f in (("old", old), ("new", new))]
for n, f, r in cases:
    for _ in range(3): once(f, r, True)
best = {c[0] + str(c[2]): 1e9 for c in cases}
for _ in range(20):
    for n, f, r in cases:
        best[n + str(r)] = min(best[n + str(r)], once(f, r, True))
ref = np.array(prod(routes[0], 0, 0).astype(mx.float32))
for n, f, r in cases:
    ms = best[n + str(r)] * 1e3
    d = np.max(np.abs(np.array(f(routes[0], r, True).astype(mx.float32)) - ref)) / np.max(np.abs(ref))
    print("%s rows=%2d %.4f ms %.1f GB/s diff %.1e" % (n, r, ms, nbytes / (ms / 1e3) / 1e9, d))
