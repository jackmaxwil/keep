import os, sys, time
import numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from benchmarks.bench_e8p_m1 import synthetic_projection, _ffn, projection_bytes_per_expert, PROJECTIONS
from ramp.kernels import e8p_m1
layers = {
    "gate_proj": synthetic_projection(experts=64, out=2048, inp=4096, group_size=512, seed=1),
    "up_proj": synthetic_projection(experts=64, out=2048, inp=4096, group_size=512, seed=2),
    "down_proj": synthetic_projection(experts=64, out=4096, inp=2048, group_size=512, seed=3),
}
rng = np.random.default_rng(0)
routes = [mx.array(np.sort(rng.choice(64, 6, replace=False))[None].astype(np.uint32)) for _ in range(64)]
ffn_bytes = 6 * sum(projection_bytes_per_expert(layers[p]) for p in PROJECTIONS)
x0 = mx.array(rng.normal(size=(1, 4096)).astype(np.float16))

def chain(n):
    x = x0
    for i in range(n):  # dependent, like consecutive layers in one decode step
        y = _ffn(layers, x, routes[i % 64], 2048)
        x = (x + y.sum(axis=0, keepdims=True) * 1e-3).astype(mx.float16)
    return x

for flag in ("0", "1"):
    os.environ[e8p_m1.E8P_M1_ENV] = flag
    for n in (1, 43):
        for _ in range(3): mx.eval(chain(n))
        best = min((lambda a: (mx.eval(chain(n)), time.perf_counter() - a)[1])(time.perf_counter()) for _ in range(30))
        ms = best / n * 1e3
        print(f"E8P_M1={flag} ffns/eval={n:2d} {ms:.4f} ms/ffn {ffn_bytes / (ms / 1e3) / 1e9:.1f} GB/s")
