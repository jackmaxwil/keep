import sys, importlib.util, time
import numpy as np, mlx.core as mx
S = sys.argv[1]; sys.argv = [sys.argv[0], S]
def load(name):
    src = open(f"{S}/{name}.py").read()
    src = src[: src.index("rng = np.random.default_rng(0)")]  # defs only
    ns = {}; exec(src, ns); return ns
A, B = load("mr"), load("mr5")
from benchmarks.bench_e8p_m1 import synthetic_projection
rng = np.random.default_rng(0)
gate = synthetic_projection(experts=64, out=2048, inp=4096, group_size=512, seed=1)
down = synthetic_projection(experts=64, out=4096, inp=2048, group_size=512, seed=3)
routes = [mx.array(np.sort(rng.choice(64, 6, replace=False))[None].astype(np.uint32)) for _ in range(64)]
xg = mx.array(rng.normal(size=(1, 4096)).astype(np.float16)); xd = mx.array(rng.normal(size=(6, 2048)).astype(np.float16))
cfgA = [(4, 1, 8, 1), (8, 1, 8, 0)]
cfgB = [(4, 1, 8, 1), (8, 1, 8, 0), (4, 4, 8, 1), (2, 1, 8, 1), (8, 2, 8, 1), (4, 1, 4, 1)]
def once(fn):
    a = time.perf_counter(); mx.eval([fn(r) for r in routes * 4]); return (time.perf_counter() - a) / 256
for name, L, x, pr in (("gate", gate, xg, 0), ("down", down, xd, 1)):
    nb = 6 * (L.codes.nbytes + L.scales.nbytes) // 64
    cases = {"old": lambda r: A["old"](x, L, r, pr)}
    for c in cfgA: cases[f"arith {c}"] = lambda r, c=c: A["new"](x, L, r, pr, *c)
    for c in cfgB: cases[f"half2 {c}"] = lambda r, c=c: B["new"](x, L, r, pr, *c)
    for f in cases.values(): once(f); once(f)
    best = {k: 1e9 for k in cases}
    for _ in range(25):
        for k, f in cases.items(): best[k] = min(best[k], once(f))
    for k in sorted(best, key=best.get):
        print(f"{name} {k:22s} {best[k]*1e3:.4f} ms {nb / best[k] / 1e9:6.1f} GB/s  x{best['old'] / best[k]:.2f}")
