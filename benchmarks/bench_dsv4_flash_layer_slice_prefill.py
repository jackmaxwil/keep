"""Prefill throughput of a real DeepSeek-V4-Flash backbone slice on Metal.

Why a slice: the released checkpoint is 163 GB and the routed experts alone are
147 GB, so the whole model cannot be resident on a 128 GB machine. Layers 0-3 of
the real ``compress_ratios`` schedule happen to contain **one of every attention
variant plus both router kinds**, and all 20 real ratio-4 layers pool
identically (L/4) as do all 20 ratio-128 layers (L/128) -- so timing one of each
is enough to weight a 43-layer extrapolation. That extrapolation lives in
``benchmarks/analyze_dsv4_flash_teacher_wallclock.py``.

| slice layer | ratio | attention | router |
| ----------- | ----- | --------- | ------ |
| 0, 1        | 0     | LocalAttention             | hash   |
| 2           | 4     | SparseCompressedAttention  | hash   |
| 3           | 128   | CompressedAttention        | scored |

Weights are the real ones and are held **exactly**: residents are FP8 e4m3
decoded against their ue8m0 128x128 block scales into bf16, which is bit-exact
(4 significant bits into 8, power-of-two scale), and routed experts are never
decoded at all -- the shipped FP4 code bytes are reinterpreted as ``uint32`` and
handed to ``mx.gather_qmm(mode="mxfp4", group_size=32, bits=4)``, the same
reinterpret omlx does. That is 3.42 GB/layer against 12.9 GB/layer dequantised.

Layers past ``--expert-layers`` reuse an already-loaded expert stack: identical
shapes, identical kernel work, a fraction of the memory. This changes the
numeric output of those layers and never their timing, and only timing is
claimed from this script.

Measurement protocol: three passes per configuration. Pass 1 is discarded (it
pays Metal kernel specialisation for every new shape and reads ~3x high), pass 2
takes per-layer timings behind ``mx.eval`` barriers, pass 3 takes the
unbarriered wall-clock. Once warm the two agree within a few percent.

Usage::

    uv run --group dev python benchmarks/bench_dsv4_flash_layer_slice_prefill.py \\
        --lengths 8192 45056 --batches 1 --out /tmp/slice.json

    uv run --group dev python benchmarks/bench_dsv4_flash_layer_slice_prefill.py \\
        --ssd-read --out /tmp/ssd.json

The heavy-job lock at the repo root is taken for the whole run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList
from mlx_lm.models.switch_layers import SwitchGLU

from mlx_vq.io.source_safetensors import read_safetensors_file_header
from ramp.models.deepseek_v4_flash_adapter import (
    DeepseekV4FlashVQModel,
    LimitedSwiGLU,
    bind_deepseek_v4_flash_non_vq_weights,
    deepseek_v4_flash_args_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
DEFAULT_CHECKPOINT = Path.home() / "models/DeepSeek-V4-Flash-0731"
DEFAULT_PACK = Path.home() / "models/teich/dsv4-coding-agent-v1-20260811.json"

SLICE_LAYERS = 4
#: checkpoint projection name -> module name. w1=gate, w2=down, w3=up.
PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
#: macOS fcntl: bypass the unified buffer cache on this descriptor.
F_NOCACHE = 48
READ_CHUNK = 8 << 20


# ---------------------------------------------------------------------------
# SSD read
# ---------------------------------------------------------------------------


def _read_whole_file(path: Path, *, nocache: bool) -> tuple[int, float]:
    total = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        start = time.perf_counter()
        while True:
            buf = os.read(fd, READ_CHUNK)
            if not buf:
                break
            total += len(buf)
        elapsed = time.perf_counter() - start
    finally:
        os.close(fd)
    return total, elapsed


def measure_ssd_read(checkpoint: Path, sustained_bytes: int) -> dict:
    """Cold (F_NOCACHE) and warm read rates off the real shards."""

    shards = sorted(checkpoint.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards under {checkpoint}")
    sizes = sorted(p.stat().st_size for p in shards)

    cold_bytes, cold_s = _read_whole_file(shards[7], nocache=True)
    _read_whole_file(shards[8], nocache=False)
    warm_bytes, warm_s = _read_whole_file(shards[8], nocache=False)

    total = 0
    used: list[str] = []
    start = time.perf_counter()
    for shard in shards[12:]:
        got, _ = _read_whole_file(shard, nocache=True)
        total += got
        used.append(shard.name)
        if total >= sustained_bytes:
            break
    sustained_s = time.perf_counter() - start

    return {
        "shard_count": len(shards),
        "shard_bytes_median": sizes[len(sizes) // 2],
        "shard_bytes_total": sum(sizes),
        "single_shard_cold": {
            "bytes": cold_bytes, "seconds": cold_s,
            "gb_per_s": cold_bytes / cold_s / 1e9,
        },
        "single_shard_warm": {
            "bytes": warm_bytes, "seconds": warm_s,
            "gb_per_s": warm_bytes / warm_s / 1e9,
        },
        "sustained_cold": {
            "shards": len(used), "first": used[0], "last": used[-1],
            "bytes": total, "seconds": sustained_s,
            "gb_per_s": total / sustained_s / 1e9,
        },
    }


# ---------------------------------------------------------------------------
# Routed experts, straight off the checkpoint bytes
# ---------------------------------------------------------------------------


class MXFP4SwitchLinear(nn.Module):
    """``gather_qmm`` over the release's own FP4 bytes -- no dequantisation."""

    def __init__(self, weight_u32: mx.array, scales_u8: mx.array):
        super().__init__()
        self.weight = weight_u32
        self.scales = scales_u8
        self.group_size = 32
        self.bits = 4
        self.mode = "mxfp4"

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_qmm(
            x, self["weight"], self["scales"], None,
            rhs_indices=indices, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
            sorted_indices=sorted_indices,
        )


class DenseSwitchLinear(nn.Module):
    """Dequantised bf16 comparison path."""

    def __init__(self, weight: mx.array):
        super().__init__()
        self.weight = weight

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_mm(
            x, self["weight"].swapaxes(-1, -2),
            rhs_indices=indices, sorted_indices=sorted_indices,
        )


class PreloadedSwitchGLU(SwitchGLU):
    """``SwitchGLU.__call__`` verbatim, without allocating random dense weights.

    ``SwitchGLU.__init__`` builds three ``SwitchLinear`` layers from
    ``mx.random.uniform``; at this model's shapes that is 8.6 GB of noise per
    projection, thrown away a line later.
    """

    def __init__(self, gate_proj, up_proj, down_proj, activation):
        nn.Module.__init__(self)
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation


class ShardReader:
    """Memory-mapped raw-byte reader, one mmap per shard."""

    def __init__(self, checkpoint: Path, weight_map: dict[str, str]):
        self.checkpoint = checkpoint
        self.weight_map = weight_map
        self._maps: dict[str, tuple[np.memmap, int, dict]] = {}

    def _shard(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._maps:
            path = self.checkpoint / shard
            header = read_safetensors_file_header(path)
            self._maps[shard] = (
                np.memmap(path, dtype=np.uint8, mode="r"),
                header.header_size,
                header.tensors,
            )
        return self._maps[shard]

    def raw(self, name: str):
        mm, header_size, tensors = self._shard(name)
        tensor = tensors[name]
        start, end = tensor.data_offsets
        base = 8 + header_size
        return np.asarray(mm[base + start : base + end]), tensor.dtype, tensor.shape

    def close(self) -> None:
        self._maps.clear()


def load_layer_experts(reader, layer, num_experts, swiglu_limit, *, mode):
    started = time.perf_counter()
    codes: dict[str, list[mx.array]] = {p: [] for p in PROJECTIONS}
    scales: dict[str, list[mx.array]] = {p: [] for p in PROJECTIONS}
    source_bytes = 0
    for expert in range(num_experts):
        for projection in PROJECTIONS:
            base = f"layers.{layer}.ffn.experts.{expert}.{projection}"
            raw_w, w_dtype, w_shape = reader.raw(f"{base}.weight")
            raw_s, s_dtype, s_shape = reader.raw(f"{base}.scale")
            if w_dtype != "I8" or s_dtype != "F8_E8M0":
                raise ValueError(f"{base}: unexpected dtypes {w_dtype}/{s_dtype}")
            source_bytes += raw_w.nbytes + raw_s.nbytes
            codes[projection].append(mx.array(raw_w).reshape(w_shape).view(mx.uint32))
            scales[projection].append(mx.array(raw_s).reshape(s_shape))
    packed = {p: mx.stack(codes[p]) for p in PROJECTIONS}
    packed_scales = {p: mx.stack(scales[p]) for p in PROJECTIONS}
    mx.eval(list(packed.values()), list(packed_scales.values()))

    if mode == "mxfp4":
        linears = {
            PROJECTIONS[p]: MXFP4SwitchLinear(packed[p], packed_scales[p])
            for p in PROJECTIONS
        }
    else:
        linears = {
            PROJECTIONS[p]: DenseSwitchLinear(
                mx.dequantize(
                    packed[p], packed_scales[p],
                    group_size=32, bits=4, mode="mxfp4",
                ).astype(mx.bfloat16)
            )
            for p in PROJECTIONS
        }
        mx.eval([lin.weight for lin in linears.values()])
    resident_bytes = sum(
        sum(v.nbytes for v in lin.parameters().values()) for lin in linears.values()
    )

    glu = PreloadedSwitchGLU(
        gate_proj=linears["gate_proj"],
        up_proj=linears["up_proj"],
        down_proj=linears["down_proj"],
        activation=LimitedSwiGLU(swiglu_limit),
    )
    return glu, {
        "layer": layer,
        "seconds": time.perf_counter() - started,
        "source_bytes": source_bytes,
        "resident_bytes": resident_bytes,
    }


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def build_slice(checkpoint: Path, expert_mode: str, expert_layers: int):
    config = json.loads((checkpoint / "config.json").read_text())
    args = replace(
        deepseek_v4_flash_args_from_config(config),
        num_hidden_layers=SLICE_LAYERS,
        # __post_init__ truncates to num_hidden_layers, giving the real
        # schedule's first four entries.
        compress_ratios=list(config["compress_ratios"]),
    )
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    started = time.perf_counter()
    # layers=None on a 4-layer model binds embed/norm/head plus layers 0-3;
    # layers 4-42 fall out as out-of-scope.
    report = bind_deepseek_v4_flash_non_vq_weights(
        model, checkpoint, layers=None, include_mtp=False, strict=False,
        compute_dtype=mx.bfloat16,
    )
    mx.eval(model.parameters())
    resident_seconds = time.perf_counter() - started

    reader = ShardReader(checkpoint, _weight_map(checkpoint))
    stacks, expert_stats = [], []
    for layer in range(expert_layers):
        glu, stats = load_layer_experts(
            reader, layer, args.n_routed_experts, args.swiglu_limit, mode=expert_mode
        )
        stacks.append(glu)
        expert_stats.append(stats)
    reader.close()

    for index, layer in enumerate(model.model.layers):
        layer.ffn.switch_mlp = stacks[index % len(stacks)]

    return args, model, {
        "resident_bind_seconds": resident_seconds,
        "resident_bound": report.bound_count,
        "resident_fp8_decoded": len(report.fp8_block_decoded_tensors),
        "experts": expert_stats,
    }


def _weight_map(checkpoint: Path) -> dict[str, str]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return {str(k): str(v) for k, v in index["weight_map"].items()}


def timed_prefill(model, args, ids: mx.array, chunk: int, *, per_layer: bool) -> dict:
    cache = model.make_cache()
    layer_seconds = [0.0] * len(model.model.layers)
    head_seconds = embed_seconds = 0.0
    started = time.perf_counter()
    logits = None

    for start in range(0, ids.shape[1], chunk):
        piece = ids[:, start : start + chunk]
        t0 = time.perf_counter()
        h = model.model.embed_tokens(piece)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :], (h.shape[0], h.shape[1], args.hc_mult, h.shape[2])
            )
        )
        first = cache[0]
        mask = create_attention_mask(
            h[:, :, 0, :],
            first[0] if isinstance(first, CacheList) else first,
            window_size=args.sliding_window,
            return_array=True,
        )
        if per_layer:
            mx.eval(h, mask)
            embed_seconds += time.perf_counter() - t0

        for index, (layer, layer_cache) in enumerate(zip(model.model.layers, cache)):
            t1 = time.perf_counter()
            h = layer(h, mask, layer_cache, piece)
            if per_layer:
                mx.eval(h)
                layer_seconds[index] += time.perf_counter() - t1

        t2 = time.perf_counter()
        logits = model.lm_head(model.model.norm(model.model.hc_head(h)))
        mx.eval(logits)
        if per_layer:
            head_seconds += time.perf_counter() - t2

    return {
        "total_seconds": time.perf_counter() - started,
        "layer_seconds": layer_seconds,
        "head_seconds": head_seconds,
        "embed_seconds": embed_seconds,
        "logits_finite": bool(mx.all(mx.isfinite(logits))),
        "logits_shape": list(logits.shape),
    }


def _prompt_ids(rows, length: int, batch: int) -> tuple[mx.array, int]:
    """Real teich ids. Returns the array and the true token count.

    ``length`` is a request, not a promise: the longest real session is 77,075
    tokens, so asking for more yields fewer, and the caller must report what it
    actually ran rather than what it asked for.
    """

    pool = [np.asarray(r["encoded_token_ids"], dtype=np.int64) for r in rows[-batch:]]
    trimmed = [p[:length] for p in pool]
    width = min(len(p) for p in trimmed)
    return mx.array(np.stack([p[:width] for p in trimmed])), width * batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--expert-mode", default="mxfp4", choices=("mxfp4", "bf16"))
    ap.add_argument("--expert-layers", type=int, default=2)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--ssd-read", action="store_true",
                    help="measure sustained SSD read off the shards and exit")
    ap.add_argument("--sustained-bytes", type=int, default=24_000_000_000)
    ap.add_argument("--out", type=Path, required=True)
    opts = ap.parse_args()

    lock = HEAVY_JOB_LOCK.open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if opts.ssd_read:
            payload = measure_ssd_read(opts.checkpoint, opts.sustained_bytes)
            print(json.dumps(payload, indent=2))
            opts.out.write_text(json.dumps(payload, indent=2))
            return

        mx.reset_peak_memory()
        args, model, build = build_slice(
            opts.checkpoint, opts.expert_mode, opts.expert_layers
        )
        build["peak_gb_after_build"] = mx.get_peak_memory() / 1e9
        build["active_gb_after_build"] = mx.get_active_memory() / 1e9

        rows = sorted(
            json.loads(opts.pack.read_text())["prompt_rows"],
            key=lambda r: r["token_count"],
        )
        results = []
        for length in opts.lengths:
            for batch in opts.batches:
                ids, tokens = _prompt_ids(rows, length, batch)
                timed_prefill(model, args, ids, opts.chunk, per_layer=False)
                mx.reset_peak_memory()
                record = timed_prefill(model, args, ids, opts.chunk, per_layer=True)
                record["peak_gb"] = mx.get_peak_memory() / 1e9
                unbarriered = timed_prefill(
                    model, args, ids, opts.chunk, per_layer=False
                )
                record.update(
                    requested_length=length,
                    length=ids.shape[1],
                    batch=batch,
                    chunk=opts.chunk,
                    tokens=tokens,
                    total_seconds_unbarriered=unbarriered["total_seconds"],
                    tokens_per_s=tokens / unbarriered["total_seconds"],
                )
                results.append(record)
                print(json.dumps(record), flush=True)

        opts.out.write_text(json.dumps({
            "expert_mode": opts.expert_mode,
            "expert_layers": opts.expert_layers,
            "checkpoint": str(opts.checkpoint),
            "slice_compress_ratios": list(args.compress_ratios),
            "build": build,
            "results": results,
        }, indent=2))
        print("wrote", opts.out)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
