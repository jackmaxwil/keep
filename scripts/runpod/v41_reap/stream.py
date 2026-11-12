"""DeepSeek-V4.1-Flash forward on one GPU, one layer at a time.

Design: ``docs/deepseek-v4-flash/2026-11-06-v41-reap-single-gpu-design.md``.

The reference ``Block`` modules run unchanged. Only the order of work changes:
for a chunk of records, each layer's weights are loaded once (straight from the
HF shards, no ``convert.py``), every record runs through that layer's attention,
then the MoE runs over whole groups of records at once. Between layers each
record keeps three things:

- its hyper-connection residual;
- its ``pre_mix``;
- its slice of ``shared_attn``, the module-level slot the reference uses to hand
  compressed KV, index keys, top-k indices and candidate blocks down the stack.

Engram rows are gathered from the shard files on the host and never put on the
GPU. ``tests/test_runpod_v41_stream.py`` checks the whole thing against
``Transformer.forward`` on a toy model.
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

INDEX = "model.safetensors.index.json"
SHARED_FIELDS = ("compress_kv", "index_k", "topk_idxs", "candidates")
ST_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
}


def import_reference(ref_dir: Path, kernel: Any | None = None) -> Any:
    """Import the reference ``model`` module, optionally with a stand-in ``kernel``."""

    if kernel is not None:
        sys.modules["kernel"] = kernel
    path = str(ref_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("model")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def reference_name(name: str) -> str:
    """HF tensor name -> reference parameter name (``convert.py`` at --model-parallel 1)."""

    name = name.removeprefix("model.")
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    return name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(size)), 8 + size


class HFCheckpoint:
    """The HF safetensors checkpoint, read one layer at a time."""

    def __init__(self, root: Path):
        self.root = root
        weight_map = json.loads((root / INDEX).read_text())["weight_map"]
        self.shard_of = {reference_name(n): (n, shard) for n, shard in weight_map.items()}

    def load(self, prefix: str, skip: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        """Every tensor under ``prefix``, keyed relative to it, with convert.py's fixups."""

        from safetensors import safe_open

        wanted: dict[str, list[tuple[str, str]]] = {}
        for ref_name, (hf_name, shard) in self.shard_of.items():
            if ref_name.startswith(prefix) and not any(s in ref_name for s in skip):
                wanted.setdefault(shard, []).append((ref_name, hf_name))
        out: dict[str, torch.Tensor] = {}
        for shard, names in wanted.items():
            with safe_open(str(self.root / shard), framework="pt", device="cpu") as handle:
                for ref_name, hf_name in names:
                    out[ref_name[len(prefix):]] = handle.get_tensor(hf_name)
        return fix_tensors(out)


def fix_tensors(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``convert.py``'s per-tensor changes: wo_a dequantized to bf16, FP4 experts viewed as fp4."""

    for name in [n for n in sd if n.endswith("wo_a.weight")]:
        scale_name = name.replace("weight", "scale")
        if scale_name in sd:
            weight, scale = sd[name], sd.pop(scale_name)
            ob, ib = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
            assert (ob, ib) in ((32, 32), (128, 128)), (name, weight.shape, scale.shape)
            full = weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * scale[:, None, :, None].float()
            sd[name] = full.flatten(2, 3).flatten(0, 1).bfloat16()
    for name, tensor in sd.items():
        if "experts" in name and tensor.dtype == torch.int8:
            sd[name] = tensor.view(torch.float4_e2m1fn_x2)
    return sd


# ---------------------------------------------------------------------------
# Engram rows on the host
# ---------------------------------------------------------------------------


class EngramRows:
    """One layer's n-gram table as host byte arrays (np.memmap of the shard, or plain arrays)."""

    def __init__(self, weight: np.ndarray, scale: np.ndarray, scale_dtype: torch.dtype, block_size: int):
        self.weight, self.scale = weight, scale  # uint8 [rows, dim], uint8 [rows, scale bytes]
        self.scale_dtype, self.block_size = scale_dtype, block_size

    @classmethod
    def from_checkpoint(cls, ckpt: HFCheckpoint, layer: int, block_size: int) -> EngramRows:
        arrays = {}
        for leaf in ("weight", "scale"):
            hf_name, shard = ckpt.shard_of[f"layers.{layer}.engram.embed.{leaf}"]
            header, base = read_header(ckpt.root / shard)
            info = header[hf_name]
            start, end = info["data_offsets"]
            rows = info["shape"][0]
            arrays[leaf] = (np.memmap(ckpt.root / shard, dtype=np.uint8, mode="r", offset=base + start,
                                      shape=(rows, (end - start) // rows)), info["dtype"])
        assert arrays["weight"][1] == "F8_E4M3", arrays["weight"][1]
        return cls(arrays["weight"][0], arrays["scale"][0], ST_DTYPES[arrays["scale"][1]], block_size)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """``ParallelEngramEmbedding.forward`` at world_size 1, reading rows from the host."""

        flat = indices.reshape(-1).cpu().numpy()
        unique, inverse = np.unique(flat, return_inverse=True)  # sorted: near-sequential reads
        w = torch.from_numpy(np.ascontiguousarray(self.weight[unique]))[torch.from_numpy(inverse)]
        s = torch.from_numpy(np.ascontiguousarray(self.scale[unique]))[torch.from_numpy(inverse)]
        w = w.to(indices.device).view(torch.float8_e4m3fn).view(*indices.shape, -1)
        s = s.to(indices.device).view(self.scale_dtype).view(*indices.shape, -1)
        values = w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


class HostTableEmbedding(torch.nn.Module):
    """Stand-in for ``ParallelEngramEmbedding``: same forward, no table on the device."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim = num_embeddings, dim
        self.rows: EngramRows | None = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.rows is not None, "engram rows not attached"
        return self.rows.lookup(indices)


# ---------------------------------------------------------------------------
# shared_attn per record
# ---------------------------------------------------------------------------


def pack_bits(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    n = mask.shape[-1]
    padded = torch.nn.functional.pad(mask.to(torch.uint8), (0, -n % 8)).unflatten(-1, (-1, 8))
    shifts = torch.arange(8, dtype=torch.uint8, device=mask.device)
    return (padded << shifts).sum(dim=-1, dtype=torch.uint8), n


def unpack_bits(packed: tuple[torch.Tensor, int]) -> torch.Tensor:
    bits, n = packed
    shifts = torch.arange(8, dtype=torch.uint8, device=bits.device)
    return ((bits.unsqueeze(-1) >> shifts) & 1).bool().flatten(-2)[..., :n]


# ---------------------------------------------------------------------------
# The streamer
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One record in one stream (the teacher, or a masked-router student)."""

    tokens: torch.Tensor  # [1, s] long
    group: Any  # items sharing a group go through the MoE together
    h: torch.Tensor | None = None
    pre_mix: torch.Tensor | None = None
    hashes: torch.Tensor | None = None
    shared: dict[str, Any] = field(default_factory=dict)
    restored: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)


class LayerStreamer:
    def __init__(
        self,
        ref: Any,
        args: Any,
        load_layer: Callable[[str], dict[str, torch.Tensor]],
        engram_rows: dict[int, EngramRows],
        hash_state: Any,
        device: str,
    ):
        self.ref, self.args, self.device = ref, args, device
        self.load_layer, self.engram_rows, self.hash_state = load_layer, engram_rows, hash_state
        ref.world_size, ref.rank = 1, 0
        ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        self.layout = ref.EngramLayout.from_args(args)
        self.hc = args.hc_mult
        with torch.device(device):
            self.embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
            self.norm = ref.RMSNorm(args.dim, args.norm_eps)
            self.head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        for prefix, module in (("embed.", self.embed), ("norm.", self.norm), ("head.", self.head)):
            module.load_state_dict(load_layer(prefix), strict=True)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def build_block(self, layer: int, state: dict[str, torch.Tensor]) -> Any:
        ref = self.ref
        real = ref.ParallelEngramEmbedding
        ref.ParallelEngramEmbedding = HostTableEmbedding
        try:
            with torch.device(self.device):
                block = ref.Block(layer, self.args, self.layout)
        finally:
            ref.ParallelEngramEmbedding = real
        block.load_state_dict(state, strict=True)
        if block.engram is not None:
            block.engram.embed.rows = self.engram_rows[layer]
        return block

    # -- per-record shared_attn ------------------------------------------------

    def restore(self, item: Item) -> None:
        shared = self.ref.shared_attn
        item.restored = {}
        for name in SHARED_FIELDS:
            value = item.shared.get(name)
            if name == "candidates" and value is not None:
                value = unpack_bits(value)
            item.restored[name] = value
            setattr(shared, name, value)

    def snapshot(self, item: Item, block: Any) -> None:
        shared = self.ref.shared_attn
        own = {id(b) for b in block.buffers()}
        for name in SHARED_FIELDS:
            value = getattr(shared, name)
            if value is None or value is item.restored.get(name):
                continue  # unchanged by this layer: keep the stored copy
            if id(value) in own:  # a view of this block's cache: the next record overwrites it
                value = value.clone()
            item.shared[name] = pack_bits(value) if name == "candidates" else value
        for name in SHARED_FIELDS:
            setattr(shared, name, None)

    # -- the forward -------------------------------------------------------------

    def start(self, items: list[Item]) -> None:
        for item in items:
            h = self.embed(item.tokens)
            item.h = h.unsqueeze(2).repeat(1, 1, self.hc, 1)
            item.pre_mix = self.ref.make_identity_pre_mix(item.h, self.hc)
            if self.hash_state is not None:
                item.hashes = self.hash_state(item.tokens, 0)

    def run_layer(self, block: Any, items: list[Item], before_moe: Callable[[int, Any, Any], None] | None) -> None:
        for item in items:  # attention: one record at a time, whole sequence
            if block.engram is not None:
                item.h = block.engram(item.h, item.hashes[:, :, block.engram.layer_hash_index, :], None)
            self.restore(item)
            residual = item.h
            attn_pre, attn_post, attn_comb = block.hc_mixes(item.h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
            x = block.attn_norm(block.hc_pre(item.h, item.pre_mix))
            x = block.attn(x, 0)
            item.h = block.hc_post(x, residual, attn_post, attn_comb)
            item.scratch["attn_pre"] = attn_pre
            self.snapshot(item, block)

        groups: dict[Any, list[Item]] = {}
        for item in items:
            groups.setdefault(item.group, []).append(item)
        for key, members in groups.items():  # FFN: the whole group through the MoE at once
            xs = []
            for item in members:
                ffn_pre, ffn_post, ffn_comb = block.hc_mixes(item.h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
                xs.append(block.ffn_norm(block.hc_pre(item.h, item.scratch.pop("attn_pre"))))
                item.scratch["ffn"] = (ffn_pre, ffn_post, ffn_comb)
            if before_moe is not None:
                before_moe(block.layer_id, block, key)
            y = block.ffn(torch.cat(xs, dim=1) if len(xs) > 1 else xs[0], None)
            for item, part in zip(members, y.split([x.size(1) for x in xs], dim=1)):
                ffn_pre, ffn_post, ffn_comb = item.scratch.pop("ffn")
                item.h = block.hc_post(part, item.h, ffn_post, ffn_comb)
                item.pre_mix = ffn_pre

    def forward(self, items: list[Item], before_moe: Callable[[int, Any, Any], None] | None = None) -> None:
        """Run ``items`` through every backbone layer; leaves each ``item.h`` / ``item.pre_mix``."""

        n_layers = int(self.args.n_layers)
        skip = (".engram.embed.",)
        pending = self._pool.submit(self.load_layer, "layers.0.", skip)
        self.start(items)
        for layer in range(n_layers):
            state = pending.result()
            if layer + 1 < n_layers:
                pending = self._pool.submit(self.load_layer, f"layers.{layer + 1}.", skip)
            block = self.build_block(layer, state)
            del state
            self.run_layer(block, items, before_moe)
            del block
        for item in items:
            item.shared.clear()

    def final_hidden(self, item: Item) -> torch.Tensor:
        """``self.norm(layer.hc_pre(h, pre_mix))``: the head's input, [1, s, dim]."""

        return self.norm(self.ref.Block.hc_pre(None, item.h, item.pre_mix))
