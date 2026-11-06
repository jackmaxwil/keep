import os, sys, time, statistics, warnings
warnings.filterwarnings("ignore")
import numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from benchmarks.bench_e8p_m1 import synthetic_projection
from ramp.kernels import e8p_m1
from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU
E = 256
glu = QuantizedVQSwitchGLU(
    gate_proj=synthetic_projection(experts=E, out=2048, inp=4096, group_size=512, seed=1),
    up_proj=synthetic_projection(experts=E, out=2048, inp=4096, group_size=512, seed=2),
    down_proj=synthetic_projection(experts=E, out=4096, inp=2048, group_size=512, seed=3),
)
per_expert = sum((p.codes.nbytes + p.scales.nbytes) // E for p in (glu.gate_proj, glu.up_proj, glu.down_proj))
rng = np.random.default_rng(0)
def routes(m):
    return [mx.array(np.stack([np.sort(rng.choice(E, 6, replace=False)) for _ in range(m)])[None].astype(np.uint32)) for _ in range(32)]
def chain(x, rs):
    for r in rs:
        y = glu(x, r)  # [1, m, 6, hidden]
        x = (x + y.sum(axis=2) * 1e-3).astype(x.dtype)
    return x
def ms_per_ffn(m, rs, n=43, samples=12):
    x = mx.array(rng.normal(size=(1, m, 4096)).astype(np.float16))
    def one(i):
        seq = [rs[(i * n + j) % len(rs)] for j in range(n)]
        t = time.perf_counter(); mx.eval(chain(x, seq)); return (time.perf_counter() - t) * 1e3 / n
    one(0); one(1)
    return statistics.median(one(i) for i in range(samples))
for m in (1, 2, 4, 8, 11, 16, 21, 32):
    rs = routes(m)
    distinct = statistics.mean(len(set(np.array(r).reshape(-1).tolist())) for r in rs)
    out = {}
    for flag in ("0", "1"):
        os.environ[e8p_m1.E8P_M1_ENV] = flag
        out[flag] = ms_per_ffn(m, rs)
    x = mx.array(rng.normal(size=(1, m, 4096)).astype(np.float16))
    os.environ[e8p_m1.E8P_M1_ENV] = "0"; a = np.array(glu(x, rs[0]).astype(mx.float32))
    os.environ[e8p_m1.E8P_M1_ENV] = "1"; b = np.array(glu(x, rs[0]).astype(mx.float32))
    diff = np.max(np.abs(a - b)) / np.max(np.abs(a))
    gbs = distinct * per_expert / (out["1"] / 1e3) / 1e9
    print(f"M={m:2d} distinct experts {distinct:5.1f}  old {out['0']:.3f} ms  new {out['1']:.3f} ms  x{out['0']/out['1']:.2f}  {gbs:.0f} GB/s distinct  diff {diff:.1e}")
