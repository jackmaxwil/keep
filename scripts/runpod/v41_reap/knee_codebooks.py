"""Where the knee of the code lies: output-blind error of 1, 1.5 and 2-bit codebooks on real experts.

Each row of a projection is cut into groups of ``--group`` inputs with one scale each and into
vectors of eight weights, each rounded to the nearest codebook point; scales and codes alternate
for a few rounds (no importance weighting, so the numbers compare codebooks, not fits). Codebooks:

- ``e8-1bit``   KEEP's 256-point E8 codebook (1 bit per weight)
- ``e8p-12``    the 4,096 E8P points of smallest norm (1.5 bits)
- ``lloyd-12``  4,096 points trained by k-means on other experts' vectors: about the best any
                1.5-bit codebook of dimension eight can do on this data
- ``e8p-16``    the full E8P grid (2 bits)

    python knee_codebooks.py --hf-path SLIM_CHECKPOINT [--layers 0 20 39] [--experts 0 1]

``--experts`` and ``--train-expert`` count through the experts the slim checkpoint still holds at 4 bits
(``compact-map.json``'s ``hot`` list, six per layer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT.parents[2] / "src"))

from keep_quant_torch import PROJ_SPACE, decode_fp4

from keep.vq.e8 import e8_1bit_grid, e8p_full_grid


def load_expert(root: Path, layer: int, expert: int, device) -> dict:
    """The 4-bit weights of one expert, read from whichever shard stores them under its own id."""
    from safetensors import safe_open

    want = {f"layers.{layer}.ffn.experts.{expert}.{proj}.{part}" for proj in PROJ_SPACE for part in ("weight", "scale")}
    got = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, "pt") as handle:
            for name in want & set(handle.keys()):
                got[name] = handle.get_tensor(name)
        if len(got) == len(want):
            break
    prefix = f"layers.{layer}.ffn.experts.{expert}."
    return {proj: decode_fp4(got[f"{prefix}{proj}.weight"], got[f"{prefix}{proj}.scale"], device) for proj in PROJ_SPACE}


def nearest(x: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    """Index of the nearest ``book`` row for each row of ``x`` ([n, 8])."""
    out, half = [], (book * book).sum(1) / 2
    for part in x.split(1 << 15):
        out.append((part @ book.T - half).argmax(1))
    return torch.cat(out)


def fit(w: torch.Tensor, book: torch.Tensor, group: int, rounds: int = 4) -> float:
    """Relative squared error of ``w`` ([out, in]) under one scale per (row, group) and ``book``."""
    v = w.reshape(w.shape[0], -1, group // 8, 8)  # [out, groups, vectors, 8]
    scale = v.pow(2).mean((-1, -2), keepdim=True).sqrt() / book.pow(2).sum(1).mean().div(8).sqrt()
    for _ in range(rounds):
        c = book[nearest((v / scale).reshape(-1, 8), book)].reshape(v.shape)
        scale = (v * c).sum((-1, -2), keepdim=True) / c.pow(2).sum((-1, -2), keepdim=True).clamp_min(1e-12)
    return float(((v - scale * c) ** 2).sum() / (v ** 2).sum())


def lloyd(x: torch.Tensor, k: int, rounds: int = 12) -> torch.Tensor:
    book = x[torch.randperm(x.shape[0], device=x.device)[:k]].clone()
    for _ in range(rounds):
        idx = nearest(x, book)
        sums = torch.zeros_like(book).index_add_(0, idx, x)
        count = torch.zeros(k, device=x.device).index_add_(0, idx, torch.ones(x.shape[0], device=x.device))
        book = torch.where(count[:, None] > 0, sums / count[:, None].clamp_min(1), book)
    return book


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", required=True, type=Path)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 20, 39])
    p.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    p.add_argument("--train-expert", type=int, default=5, help="the expert whose vectors train lloyd-12")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    dev, ckpt = torch.device(args.device), args.hf_path
    held = json.loads((args.hf_path / "compact-map.json").read_text())["hot"]
    torch.manual_seed(0)
    full = torch.from_numpy(e8p_full_grid().astype(np.float32)).to(dev)
    books = {"e8-1bit": torch.from_numpy(e8_1bit_grid().astype(np.float32)).to(dev),
             "e8p-12": full[full.pow(2).sum(1).argsort(stable=True)[:4096]], "lloyd-12": None, "e8p-16": full}
    print("layer expert proj | " + " | ".join(books))
    totals = {name: [] for name in books}
    for layer in args.layers:
        train = load_expert(ckpt, layer, held[layer][args.train_expert], dev)
        vecs = [w.float().reshape(w.shape[0], -1, (512 if w.shape[1] % 512 == 0 else 256) // 8, 8) for w in train.values()]
        vecs = torch.cat([(v / v.pow(2).mean((-1, -2), keepdim=True).sqrt()).reshape(-1, 8) for v in vecs])
        books["lloyd-12"] = lloyd(vecs[torch.randperm(vecs.shape[0], device=dev)[: 1 << 20]], 4096)
        for expert in args.experts:
            for proj, w in load_expert(ckpt, layer, held[layer][expert], dev).items():
                w = w.float()
                errs = {name: fit(w, book, 512 if w.shape[1] % 512 == 0 else 256) for name, book in books.items()}
                for name, e in errs.items():
                    totals[name].append(e)
                print(f"{layer:5d} {expert:6d} {proj:>4} | " + " | ".join(f"{e:.4f}" for e in errs.values()), flush=True)
    print("mean              | " + " | ".join(f"{np.mean(v):.4f}" for v in totals.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
