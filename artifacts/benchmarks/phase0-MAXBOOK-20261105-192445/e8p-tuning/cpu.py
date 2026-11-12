import sys, time
import numpy as np, mlx.core as mx
S = sys.argv[1]
src = open(f"{S}/mr.py").read(); src = src[: src.index("rng = np.random.default_rng(0)")]
ns = {}; sys.argv = [sys.argv[0], S]; exec(src, ns)
from benchmarks.bench_e8p_m1 import synthetic_projection
rng = np.random.default_rng(0)
gate = synthetic_projection(experts=64, out=2048, inp=4096, group_size=512, seed=1)
routes = [mx.array(np.sort(rng.choice(64, 6, replace=False))[None].astype(np.uint32)) for _ in range(64)]
x = mx.array(rng.normal(size=(1, 4096)).astype(np.float16))
one = [mx.array(np.array([[i]], dtype=np.uint32)) for i in range(64)]
f = lambda r: ns["new"](x, gate, r, 0, 4, 1, 8, 1)
for _ in range(3): mx.eval([f(r) for r in routes * 4])
for label, rs in (("6 routes", routes), ("1 route ", one)):
    best_build = best_total = 1e9
    for _ in range(20):
        a = time.perf_counter(); outs = [f(r) for r in rs * 4]; b = time.perf_counter(); mx.eval(outs); c = time.perf_counter()
        best_build = min(best_build, (b - a) / 256); best_total = min(best_total, (c - a) / 256)
    print(f"{label}: build {best_build*1e6:.1f} us/call, build+eval {best_total*1e6:.1f} us/call")
# matmul build overhead for comparison
w = mx.random.normal((1560, 4096)).astype(mx.bfloat16); xb = x.astype(mx.bfloat16); mx.eval(w, xb)
a = time.perf_counter(); outs = [xb @ w.T for _ in range(256)]; b = time.perf_counter(); mx.eval(outs); c = time.perf_counter()
print(f"bf16 matmul: build {(b-a)/256*1e6:.1f} us/call, build+eval {(c-a)/256*1e6:.1f} us/call")
