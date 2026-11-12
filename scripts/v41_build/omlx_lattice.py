"""The no-wait expert cache (``omlx_routed.py``) with 2-bit lattice experts, under stock oMLX.

The routed experts are this repo's E8P codes (2.04 bits per weight, 9.0 MB an expert against
18.8 MB at MXFP4), so about twice as many stay resident. oMLX loads a small model
(``omlx_compact.py make --experts 6``); after the load each layer's routed experts are replaced:

- Storage. A slot is one row of six stacked arrays (codes and scales of w1, w3, w2). The bytes
  come straight out of the quantizer's ``layer-NN.npz`` files, which are stored uncompressed, so
  an install is a positional read into the slot's rows, as in the MXFP4 cache.
- Router. It keeps every row of the pool and scores all of it once per token (oMLX's gate
  kernel). One small kernel then takes two top-k's of those scores: among the resident experts
  (the bias of every other expert is -1e30), which is what runs, and in the whole pool, which
  is the demand the cache policy reads later. Installing or evicting an expert changes its bias
  and its entry in the expert-to-slot table; the router rows never move.
- Decode step. Router, then this repo's ``e8p_m1`` kernels for the routed experts, then oMLX's
  own kernels for the shared expert, which also add the routed sum.
- Prefill. oMLX's unfused MoE, with the routed experts on this repo's sorted-route kernels.

    OMLX_COMPACT=DIR/compact-map.json OMLX_LATTICE=CODES_DIR python omlx_lattice.py serve ...
    python omlx_lattice.py check --config CONFIG.json --codes CODES_DIR

The pool can be wider than the loaded checkpoint. ``OMLX_LATTICE`` is then a JSON file::

    {"codes": [[CODES_DIR, IDS.json], ...], "router": "router.safetensors", "mass": "reap-stats.npz"}

``IDS.json`` (``{"survivors": {layer: [...]}}``) gives the pool number of each expert of its
directory, in file order; the first directory holds the loaded checkpoint's experts. ``router`` has
every router row of the pool and ``mass`` (``reap__gate_norm_sum``) orders the starting residents.

Environment: ``OMLX_LATTICE_SLOTS`` (resident experts per layer; default ``auto``, from this
machine's RAM, see ``auto_slots``), ``OMLX_LATTICE_BUDGET_GIB`` (the footprint ``auto`` aims at,
default 70% of RAM) and the policy settings of ``omlx_routed.py``. ``check`` compares one layer's decode step with dense matrices
decoded on the host, the prefill path with the decode path, and slots after installs with their
stored bytes, then times the decode-step MoE.
"""

import json
import os
import struct
import sys
import time
import zipfile
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omlx.patches.deepseek_v41 import decode_moe as DM
from omlx.patches.deepseek_v41 import language, loading
from omlx.patches.deepseek_v41.activation import quantize_swiglu_activation
from omlx.patches.deepseek_v41.quantization import quantize_activation
from omlx_compact import patch as compact_patch
from omlx_routed import OFF, Cache, Store

from keep.vq.e8 import decode_weight_matrix, e8p_packed_abs_grid
from ramp.kernels.e8p_m1 import e8p_m1_per_route, e8p_m1_token_layout
from ramp.nn.switch_linear import QuantizedVQSwitchLinear

TENSORS = [(proj, field) for proj in ("w1", "w3", "w2") for field in ("codes", "scales")]

# One threadgroup per token. Top-k among the residents (slots and routing weights, as oMLX's
# up kernel computes them) and top-k in the whole pool (expert numbers and raw scores).
_TOPK = r"""
    threadgroup int chosen[8];
    const uint tid = thread_index_in_threadgroup, b = threadgroup_position_in_grid.x;
    const device float* sc = scores + b * NE;
    topk_select<NE, TOPK>(sc, bias, chosen, tid);
    if (tid == 0) {
        float wsum = 0.0f;
        for (int j = 0; j < TOPK; ++j) wsum += sc[chosen[j]];
        for (int j = 0; j < TOPK; ++j) {
            slot[b * TOPK + j] = slot_of[chosen[j]];
            weight[b * TOPK + j] = sc[chosen[j]] / (wsum + 1e-20f) * route_scale[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    topk_select<NE, TOPK>(sc, pool_bias, chosen, tid);
    if (tid == 0)
        for (int j = 0; j < TOPK; ++j) {
            want[b * TOPK + j] = uint(chosen[j]);
            want_score[b * TOPK + j] = sc[chosen[j]];
        }
"""


@cache
def _topk_kernel():
    return mx.fast.metal_kernel(
        name="v41_lattice_topk", input_names=["scores", "bias", "pool_bias", "slot_of", "route_scale"],
        output_names=["slot", "weight", "want", "want_score"], source=_TOPK, header=DM._HEADER)


@cache
def _unused(dtype):
    """Stands in for the routed MXFP4 inputs of oMLX's up kernel, which runs no routed column."""
    a = mx.zeros((8,), dtype)
    mx.eval(a)
    return a


class Codes:
    """Where each expert's codes and scales sit inside the ``layer-NN.npz`` files of ``roots``.
    ``where[layer][expert]`` is (directory, position in its file); without it, (0, expert)."""

    def __init__(self, roots: list[Path], layers: int, where=None):
        self.files, self.where = [[] for _ in range(layers)], where  # [layer][directory]
        for i, root in ((i, root) for i in range(layers) for root in roots):
            path = root / f"layer-{i:02d}.npz"
            fd, members = os.open(path, os.O_RDONLY), {}
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    assert info.compress_type == zipfile.ZIP_STORED, f"{path}: {info.filename} is compressed"
                    name, extra = struct.unpack("<HH", os.pread(fd, 4, info.header_offset + 26))
                    with z.open(info) as f:
                        major, _ = np.lib.format.read_magic(f)
                        read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                        shape, fortran, dtype = read(f)
                        assert not fortran, info.filename
                        start = info.header_offset + 30 + name + extra + f.tell()
                    members[info.filename.removesuffix(".npy")] = (start, shape, dtype)
            self.files[i].append((fd, members))

    def slab(self, layer: int, name: str, expert: int) -> tuple[int, int, int]:
        d, expert = self.where[layer][expert] if self.where else (0, expert)
        fd, members = self.files[layer][d]
        start, shape, dtype = members[name]
        size = int(np.prod(shape[1:])) * dtype.itemsize
        return fd, start + expert * size, size

    def read(self, layer: int, name: str, experts, pool=None) -> np.ndarray:
        _, shape, dtype = self.files[layer][0][1][name]
        host = np.empty((len(experts), *shape[1:]), dtype)
        jobs = [(*self.slab(layer, name, e), host[s]) for s, e in enumerate(experts)]
        for done in [pool.submit(Store.pread_into, *job) for job in jobs] if pool else jobs:
            done.result() if pool else Store.pread_into(*done)
        return host


class Experts(nn.Module):
    """oMLX's routed ``Expert`` for the unfused path; ``indices`` are slots."""

    quantizes_input = True

    def __init__(self, limit):
        super().__init__()
        self._limit = limit

    def __call__(self, x, indices, weights, sorted_indices=False, *, input_quantized=False):
        # sorted: x [routes, 1, dim], indices [routes]; otherwise x [..., 1, 1, dim], indices [..., k]
        lead = indices.shape if sorted_indices else indices.shape[:-1]
        rows = x.reshape(*lead, x.shape[-1])
        gate = self.w1(rows, indices, sorted_indices=sorted_indices)
        up = self.w3(rows, indices, sorted_indices=sorted_indices)
        h = quantize_swiglu_activation(gate, up, weights, x.dtype, self._limit)
        return self.w2(h, indices, sorted_indices=sorted_indices)[..., None, :].astype(x.dtype)


class Layer:
    def __init__(self, index, moe, experts, weight, bias, order, slots):
        assert weight.dtype == mx.bfloat16, weight.dtype  # the gate kernel reads bf16 rows
        self.index, self.gate, self.experts = index, moe.gate, experts
        self.full_weight, self.full_bias = weight.astype(mx.float32), bias.astype(mx.float32)
        mx.eval(self.full_weight, self.full_bias)
        self.pool_bias = np.array(self.full_bias)
        resident = [int(e) for e in order[:slots]]  # hottest first
        self.slot_expert = np.array(resident, dtype=np.int64)  # -1: unrouted
        self.slot_of = {e: s for s, e in enumerate(resident)}
        self.last_use = -1.0 - np.arange(slots, dtype=np.float64)  # colder starting experts go first
        self.cooling: list[tuple[int, int]] = []
        self.free: list[int] = []
        self.reading: dict[int, tuple[int, list]] = {}
        masked, slot_map = np.full(self.pool_bias.size, OFF, np.float32), np.zeros(self.pool_bias.size, np.uint32)
        masked[resident], slot_map[resident] = self.pool_bias[resident], np.arange(slots)
        self.slot_map = mx.array(slot_map)  # expert -> slot, read only for experts whose bias is on
        moe.gate.weight, moe.gate.bias, moe.experts = weight, mx.array(masked), experts
        self.views = {}
        for proj, field in TENSORS:
            lin = getattr(experts, proj)
            self.views[proj, field] = (lin, lin[field], np.asarray(memoryview(lin[field])))
        assert self.views_current()

    def views_current(self) -> bool:
        """The host views still alias the arrays the kernels read."""
        return all(lin[field] is owner and np.frombuffer(memoryview(owner), np.uint8).ctypes.data == view.ctypes.data
                   for (_, field), (lin, owner, view) in self.views.items())


class LatticeCache(Cache):
    def __init__(self, model, config, roots: list[Path], order, full, slots: int, where=None):
        super().__init__(config, mx)
        blocks = model.language_model.layers
        self.codes = Codes(roots, len(blocks), where)
        self.codebook = mx.array(e8p_packed_abs_grid())
        layers = []
        for i, blk in enumerate(blocks):
            experts = Experts(config.swiglu_limit)
            for proj in ("w1", "w3", "w2"):
                codes, scales = (mx.array(self.codes.read(i, f"{proj}_{field}", order[i][:slots], self.pool))
                                 for field in ("codes", "scales"))
                mx.eval(codes, scales)
                setattr(experts, proj, QuantizedVQSwitchLinear(
                    input_dims=codes.shape[2] * 8, output_dims=codes.shape[1], num_experts=slots, codes=codes,
                    scales=scales, codebook=self.codebook, group_size=codes.shape[2] * 8 // scales.shape[2],
                    code_bits=16))
            layers.append(Layer(i, blk.ffn, experts, full[i, "weight"], full[i, "bias"], order[i], slots))
        self.pool_config = replace(config, n_routed_experts=layers[0].pool_bias.size)
        self.ready(layers)

    def slabs(self, layer, expert, slot):
        for (proj, field), (_, _, view) in layer.views.items():
            yield (*self.codes.slab(layer.index, f"{proj}_{field}", expert), view[slot])

    def route(self, layer, expert, slot):
        layer.slot_map[expert] = slot
        layer.gate.bias[expert] = float(layer.pool_bias[expert])

    def unroute(self, layer, expert, slot):
        layer.gate.bias[expert] = OFF


def decode(cache, layer, moe, x, config, mma, stock_gate):
    """One MoE for the decode step. x: [tokens, dim] bf16. Returns ([tokens, dim] bf16, slots, weights)."""
    if layer.index == 0:
        cache.begin()
    nb, dim, inter, k = x.shape[0], config.dim, config.moe_inter_dim, config.n_activated_experts
    scores, xq = stock_gate(moe, x, cache.pool_config)  # every expert of the pool; act_quant(x)
    slot, weight, want, want_score = _topk_kernel()(
        inputs=[scores, moe.gate.bias, layer.full_bias, layer.slot_map, DM._const((config.route_scale,))],
        template=[("NE", cache.pool_config.n_routed_experts), ("TOPK", k)],
        grid=(nb * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(nb * k,)] * 4, output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.float32])
    cache.pending.setdefault(cache.step, []).append((want, want_score))  # as evaluated: reading them later costs no eval
    e, s = layer.experts, moe.shared_experts
    # 16-bit activations: the lattice kernel is 1.7x slower on fp32 rows, and act_quant values are exact in bf16.
    rows, slots = xq.reshape(nb, dim).astype(mx.bfloat16), slot.reshape(nb, k)
    gate = e8p_m1_token_layout(rows, e.w1.codes, e.w1.scales, cache.codebook, slots, output_dims=inter)
    up = e8p_m1_token_layout(rows, e.w3.codes, e.w3.scales, cache.codebook, slots, output_dims=inter)
    h = quantize_swiglu_activation(gate, up, weight, mx.bfloat16, config.swiglu_limit)
    down = e8p_m1_per_route(h.reshape(nb * k, inter), e.w2.codes, e.w2.scales, cache.codebook, slot, output_dims=dim)
    routed = down.astype(mx.float32).reshape(nb, k, dim).sum(1).reshape(-1)  # bf16 per expert, fp32 sum
    # The shared expert on oMLX's kernels: its up kernel with no routed column (TOPK 0), then its
    # shared down kernel, which adds the routed sum and rounds to bf16.
    shared = tuple(DM.weight_format(getattr(s, n)) for n in ("w1", "w3", "w2"))
    up_k, _, down_s = DM._kernels_nb(mma and all(f[0] in ("mxfp8", "affine") for f in shared), (DM.MXFP4,) * 3 + shared)
    no_w, no_s, no_b = _unused(mx.uint32), _unused(mx.uint8), _unused(mx.float32)
    hqs = up_k(
        inputs=[no_b, no_b, no_b, DM._const((config.swiglu_limit,)), xq, no_w, no_s, no_b, no_w, no_s, no_b,
                s.w1.weight, s.w1.scales, DM._biases(s.w1), s.w3.weight, s.w3.scales, DM._biases(s.w3)],
        template=[("NE", 1), ("TOPK", 0), ("DIM", dim), ("INTER", inter), ("NB", nb)],
        grid=(256, inter // 32, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1,), (nb * inter,), (1,), (1,)],
        output_dtypes=[mx.float32, mx.float32, mx.int32, mx.float32])[1]
    out = down_s(
        inputs=[hqs, routed, s.w2.weight, s.w2.scales, DM._biases(s.w2)],
        template=[("DIM", dim), ("INTER", inter), ("NB", nb)],
        grid=(dim // 8 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(nb * dim,)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(nb, dim), slots, weight.reshape(nb, k)


def hook(state: dict) -> None:
    """Send the layers of ``state["cache"]`` through the lattice paths; every other MoE is oMLX's."""
    stock = SimpleNamespace(one=DM.fused_decode_moe, many=DM.fused_decode_moe_nb, supported=DM.supported,
                            gate=DM._gate, gate_call=language.Gate.__call__)

    def find(gate):
        cache = state.get("cache")
        return cache, cache.by_gate.get(id(gate)) if cache else None

    def supported(moe, x, config):
        layer = find(moe.gate)[1]
        return stock.supported(moe, x, config) if layer is None else x.size == config.dim and x.dtype == mx.bfloat16

    def one(moe, x, config):
        cache, layer = find(moe.gate)
        if layer is None:
            return stock.one(moe, x, config)
        out, slots, weights = decode(cache, layer, moe, x.reshape(1, -1), config, False, stock.gate)
        return out.reshape(x.shape), slots, weights

    def many(moe, x, config, mma=None):
        cache, layer = find(moe.gate)
        if layer is None:
            return stock.many(moe, x, config, mma)
        return decode(cache, layer, moe, x, config, DM.use_mma(x.shape[0]) if mma is None else mma, stock.gate)

    def gate_plain(self, x, image_mask):  # prefill and every unfused path
        cache, layer = find(self)
        if layer is None:
            return stock.gate_call(self, x, image_mask)
        if layer.index == 0:
            cache.begin()
        index, weights = stock.gate_call(self, x, image_mask)  # pool numbers of resident experts
        return cache.observe(self, x, layer.slot_map[index]), weights

    DM.fused_decode_moe, DM.fused_decode_moe_nb, DM.supported = one, many, supported
    language.Gate.__call__ = gate_plain


def wide_pool(path: Path, loaded: dict):
    """(directories, order, router rows, where) of the pool file ``path``. ``loaded`` is the
    router of the loaded checkpoint, whose rows must be the first directory's."""
    spec = json.loads(path.read_text())
    ids = [json.loads(Path(f).expanduser().read_text())["survivors"] for _, f in spec["codes"]]
    where = [{e: (d, j) for d, per in enumerate(ids) for j, e in enumerate(per[str(i)])} for i in range(len(ids[0]))]
    mass = np.load(Path(spec["mass"]).expanduser())["reap__gate_norm_sum"]
    assert all(sorted(w) == list(range(mass.shape[1])) for w in where), "the directories do not cover the pool once"
    router = mx.load(str(Path(spec["router"]).expanduser()))
    full = {(i, f): router[f"layers.{i}.ffn.gate.{f}"] for i in range(len(where)) for f in ("weight", "bias")}
    for (i, f), rows in loaded.items():
        if f in ("weight", "bias"):
            assert mx.array_equal(full[i, f][mx.array(ids[0][str(i)])], rows), f"layer {i}: router {f} differs"
    order = [np.argsort(-row, kind="stable").tolist() for row in mass]
    return [Path(d).expanduser() for d, _ in spec["codes"]], order, full, where


FLOOR = 120  # residents per layer below which sampled code came out misspelled (build plan, C2)


def auto_slots(ram_bytes: int, drafter: bool, budget_gib: float | None = None) -> int:
    """Resident experts per layer for a footprint of ``budget_gib`` (default: 70% of RAM).

    Measured on an M4 Max, 128 GB (2026-11-12): the server's footprint peaks near 20 GiB plus
    0.336 GiB per resident, and 8 GiB more with the drafter (87 GiB at 200 residents, 88 GiB at
    180 with the drafter). Other chips are unmeasured; the budget is the knob."""
    budget = 0.7 * ram_bytes / 2**30 if budget_gib is None else budget_gib
    return int((budget - 20 - 8 * drafter) / 0.336)


def drafter_on(argv: list[str]) -> bool:
    """Does ``--base-path``'s model_settings.json switch the drafter on for any model?"""
    if "--base-path" not in argv:
        return False
    settings = Path(argv[argv.index("--base-path") + 1]).expanduser() / "model_settings.json"
    return settings.exists() and any(m.get("mtp_enabled") for m in json.loads(settings.read_text())["models"].values())


def patch(spec: dict, root: Path, slots: int) -> None:
    full, state = {}, {}
    compact_patch(spec, full)
    load = loading.load

    def load_lattice(*args, **kwargs):
        model, processor = load(*args, **kwargs)
        config = model.language_model.layers[0].ffn.gate._config
        if root.is_file():
            roots, order, wide, where = wide_pool(root, full)
            state["cache"] = LatticeCache(model, config, roots, order, wide, min(slots, len(order[0])), where)
        else:
            state["cache"] = LatticeCache(model, config, [root], spec["order"], full, min(slots, len(spec["order"][0])))
        full.clear()
        return model, processor

    loading.load = load_lattice
    hook(state)


def check(config_path: Path, root: Path, slots: int = 32, chain: int = 40) -> None:
    from bench_expert_formats import median_ms, omlx_moe
    from omlx.patches.deepseek_v41.config import ModelConfig

    rng = np.random.default_rng(0)
    c = replace(ModelConfig.from_dict(json.loads(config_path.read_text())), n_routed_experts=64)
    moes = [omlx_moe(c, "mxfp4", 4, 32, rng)[0] for _ in range(2)]
    for moe in moes:  # shared experts: random MXFP8 bytes below the codes for inf and NaN
        for name in ("w1", "w3", "w2"):
            shape = getattr(moe.shared_experts, name).weight.shape
            codes = rng.integers(0, 0x60, size=(*shape, 4), dtype=np.uint8) | (rng.integers(0, 2, size=(*shape, 4), dtype=np.uint8) << 7)
            getattr(moe.shared_experts, name).weight = mx.array(codes.view(np.uint32).reshape(shape))
    pool = 192
    full = {}
    for i in range(2):
        full[i, "weight"] = mx.array(rng.normal(size=(pool, c.dim)).astype(np.float32)).astype(mx.bfloat16)
        full[i, "bias"] = mx.array(rng.normal(size=(pool,)).astype(np.float32) * 0.1)
    os.environ.setdefault("OMLX_ROUTED_TAU", "inf")
    model = SimpleNamespace(language_model=SimpleNamespace(layers=[SimpleNamespace(ffn=m) for m in moes]))
    roots, where = [root], None
    if root.is_file():  # a pool file: both directories, the first two layers
        roots, _, _, where = wide_pool(root, {})
        pool = len(where[0])
        for i in range(2):
            full[i, "weight"] = mx.array(rng.normal(size=(pool, c.dim)).astype(np.float32)).astype(mx.bfloat16)
            full[i, "bias"] = mx.array(rng.normal(size=(pool,)).astype(np.float32) * 0.1)
    state = {"cache": LatticeCache(model, c, roots, [rng.permutation(pool).tolist() for _ in moes], full, slots, where)}
    cache = state["cache"]
    hook(state)

    def token():
        return mx.array(rng.normal(size=(c.dim,)).astype(np.float32)).astype(mx.bfloat16)

    def dense(layer, proj, expert):
        return decode_weight_matrix(*(cache.codes.read(layer, f"{proj}_{f}", [expert])[0] for f in ("codes", "scales")),
                                    code_bits=16)

    def host(a):
        return np.array(a.astype(mx.float32))

    def reference(i, x, slots_, weights):
        """The same MoE from matrices decoded on the host (and oMLX's shared expert, unfused),
        rounded to bf16 where oMLX's kernels round."""
        xq, limit, total = host(quantize_activation(x.reshape(1, -1)))[0], c.swiglu_limit, 0.0
        for slot, weight in zip(slots_, weights):
            expert = int(cache.layers[i].slot_expert[slot])
            g, u = bf16(dense(i, "w1", expert) @ xq), bf16(dense(i, "w3", expert) @ xq)
            if limit:
                g, u = np.minimum(g, limit), np.clip(u, -limit, limit)
            h = bf16(g / (1 + np.exp(-g)) * u * weight)
            total = total + bf16(dense(i, "w2", expert) @ host(quantize_activation(mx.array(h[None])))[0])
        shared = moes[i].shared_experts(quantize_activation(x.reshape(1, 1, -1)), input_quantized=True)
        return total + host(shared).reshape(-1)

    def bf16(a):
        return host(mx.array(a.astype(np.float32)).astype(mx.bfloat16))

    def gap(a, b):
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    def against_reference(i):
        x = token()
        out, slots_, weights = DM.fused_decode_moe(moes[i], x.reshape(1, 1, -1), c)
        return gap(host(out).reshape(-1), reference(i, x, np.array(slots_).reshape(-1), np.array(weights).reshape(-1)))

    worst = max(against_reference(i) for i in (0, 1) for _ in range(2))
    print(f"decode step against host-decoded matrices: worst relative gap {worst:.4f}")
    assert worst < 2e-2

    x = mx.stack([token() for _ in range(40)])[None]  # [1, 40, dim]: the sorted-route prefill path
    decoded = host(DM.fused_decode_moe_nb(moes[0], x[0, :8], c)[0])
    worst = max(gap(host(moes[0](x[:, :t], None))[0, :8][: min(t, 8)], decoded[: min(t, 8)]) for t in (40, 4))
    print(f"prefill path against decode path: worst relative gap {worst:.4f}")
    assert worst < 2e-2

    cache.tau = 0.0  # admit everything demanded
    for step in range(24):
        for moe in moes:
            mx.eval(DM.fused_decode_moe(moe, token().reshape(1, 1, -1), c)[0])
        time.sleep(0.05)
    print("after 24 tokens at tau 0:", {k: cache.stats[k] for k in ("wanted", "admitted", "installed", "evicted", "no_slot")})
    time.sleep(0.5)
    assert cache.stats["installed"] > 0 and cache.mismatched() == 0
    cache.tau = float("inf")
    worst = max(against_reference(i) for i in (0, 1) for _ in range(3))
    print(f"after installs, against host-decoded matrices: worst relative gap {worst:.4f}; no slot differs from its bytes")
    assert worst < 2e-2

    state_x = {"x": token().reshape(1, 1, -1)}

    def step(n):
        x = state_x["x"]
        for j in range(n):
            out = DM.fused_decode_moe(moes[j % 2], x, c)[0].astype(mx.float32)
            x = (out / (mx.max(mx.abs(out)) + 1e-6) * 3).astype(mx.bfloat16)
        mx.eval(x)
        state_x["x"] = x

    print(f"decode-step MoE (router over {pool}, {c.n_activated_experts} lattice experts, shared expert): "
          f"{median_ms(step, chain):.3f} ms")


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--codes", type=Path, required=True)
        a = p.parse_args(sys.argv[2:])
        check(a.config.expanduser(), a.codes.expanduser())
    else:
        assert auto_slots(128 << 30, True) == 183 and auto_slots(128 << 30, False, 87) == 199
        slots = os.environ.get("OMLX_LATTICE_SLOTS", "auto")
        if slots == "auto":
            budget = os.environ.get("OMLX_LATTICE_BUDGET_GIB")
            slots = auto_slots(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), drafter_on(sys.argv),
                               float(budget) if budget else None)
        slots = int(slots)
        print(f"lattice cache: {slots} resident experts per layer" + (
            f"; below {FLOOR}, generated code was unreliable in our tests" if slots < FLOOR else ""), file=sys.stderr)
        assert slots >= 6, "not enough memory for the router's top-6"
        patch(json.loads(Path(os.environ["OMLX_COMPACT"]).read_text()), Path(os.environ["OMLX_LATTICE"]).expanduser(), slots)
        from omlx.cli import main

        sys.exit(main())
