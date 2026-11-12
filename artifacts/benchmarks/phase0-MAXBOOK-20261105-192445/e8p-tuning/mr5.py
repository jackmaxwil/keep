import sys, time, itertools
import numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from benchmarks.bench_e8p_m1 import synthetic_projection
from ramp.kernels import e8p_m1
from ramp.kernels.gather_vqmm import _qmv_output_dtype
S = sys.argv[1]
K = mx.fast.metal_kernel(name="e8p_mr", input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
                         output_names=["out"], source=open(S + "/e8p_mr5.metal").read())
def new(x, L, r, per_route, R, SG, C, tg):
    routes = r.size; out_dim = L.codes.shape[1]
    (o,) = K(inputs=[x, L.codes, L.scales, L.codebook, r.reshape(1, routes)],
             template=[("ROWS_PER_SG", R), ("SIMDGROUPS", SG), ("CHUNK", C), ("TG_CODEBOOK", tg),
                       ("PER_ROUTE", per_route), ("OUT_T", _qmv_output_dtype(x.dtype))],
             grid=(32 * SG, (out_dim + R * SG - 1) // (R * SG), routes), threadgroup=(32 * SG, 1, 1),
             output_shapes=[(1, routes, out_dim)], output_dtypes=[_qmv_output_dtype(x.dtype)])
    return o
def old(x, L, r, per_route):
    f = e8p_m1.e8p_m1_per_route if per_route else e8p_m1.e8p_m1_token_layout
    return f(x, L.codes, L.scales, L.codebook, r, output_dims=L.codes.shape[1], rows_per_threadgroup=16)
rng = np.random.default_rng(0)
gate = synthetic_projection(experts=64, out=2048, inp=4096, group_size=512, seed=1)
down = synthetic_projection(experts=64, out=4096, inp=2048, group_size=512, seed=3)
routes = [mx.array(np.sort(rng.choice(64, 6, replace=False))[None].astype(np.uint32)) for _ in range(64)]
xg = mx.array(rng.normal(size=(1, 4096)).astype(np.float16))
xd = mx.array(rng.normal(size=(6, 2048)).astype(np.float16))
def once(fn):
    a = time.perf_counter(); mx.eval([fn(r) for r in routes * 4]); return (time.perf_counter() - a) / 256
cases = {}
for name, L, x, pr in (("gate", gate, xg, 0), ("down", down, xd, 1)):
    nbytes = 6 * (L.codes.nbytes + L.scales.nbytes) // 64
    cases[(name, "old")] = (lambda r, L=L, x=x, pr=pr: old(x, L, r, pr), nbytes, L, x, pr)
    for R, SG, C, tg in itertools.product((1, 2, 4, 8), (1, 2, 4), (4, 8), (1, 0)):
        cases[(name, f"R{R} SG{SG} C{C} tg{tg}")] = (lambda r, L=L, x=x, pr=pr, a=(R, SG, C, tg): new(x, L, r, pr, *a), nbytes, L, x, pr)
for k, (fn, *_ ) in cases.items():
    for _ in range(2): once(fn)
best = {k: 1e9 for k in cases}
for _ in range(8):
    for k, (fn, *_ ) in cases.items(): best[k] = min(best[k], once(fn))
for name in ("gate", "down"):
    ref = np.array(cases[(name, "old")][0](routes[0]).astype(mx.float32)).reshape(-1)
    rows = sorted(((k[1], best[k]) for k in cases if k[0] == name), key=lambda t: t[1])
    for cfg, t in rows[:8] + [r for r in rows if r[0] == "old"]:
        got = np.array(cases[(name, cfg)][0](routes[0]).astype(mx.float32)).reshape(-1)
        d = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        print(f"{name} {cfg:18s} {t*1e3:.4f} ms {cases[(name, cfg)][1] / t / 1e9:6.1f} GB/s diff {d:.1e}")
