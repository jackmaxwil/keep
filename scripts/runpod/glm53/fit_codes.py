#!/usr/bin/env python3
"""2-bit E8P codes and 1-bit residuals for GLM-5.3-Flash's routed experts, on torch.

The second model's twin of ``v41_reap/keep_quant_torch.py fit``: the same search (``Search``) and
layer loop (``fit_layers``), with GLM's FP8 experts decoded exactly by ``residency.Checkpoint`` (FP8
codes times a float32 multiplier per 128 x 128 block). gate/up/down are w1/w3/w2; gate and up read
the hidden importance (4096), down the GLU importance (2048), both from ``residency.py
--importance`` (``imp-glm.npz``). Every input dimension divides by 512, so every group is 512.
Output: ``layer-XX.npz`` per routing layer (3 to 44) in ``keep_quant``'s format, with the experts'
ids in the file's ``experts`` row.

    # 2-bit codes, all 288 experts; one process per GPU
    fit_codes.py fit --ckpt DIR --importance imp-glm.npz --out codes2 --layers 3,7,11
    # 1-bit residual of those codes for the 96 experts of largest calibration mass per layer
    fit_codes.py fit --ckpt DIR --importance imp-glm.npz --out resid1 --residual-of codes2 --code-bits 8 --top 96
    # the torch fit against KEEP's numpy fit (random weights, or rows of one real expert with --ckpt)
    fit_codes.py check [--ckpt DIR [--importance imp-glm.npz]]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parent / "v41_reap"))

from keep_quant_torch import PROJ_SPACE, Search, fit_layers

PROJ = {"w1": "gate", "w3": "up", "w2": "down"}


def loader(ckpt_dir: Path, device):
    from residency import PREFIX, Checkpoint

    ckpt = Checkpoint(ckpt_dir)

    def load(layer: int, expert: int) -> dict[str, torch.Tensor]:
        stem = f"{PREFIX}layers.{layer}.mlp.experts.{expert}."
        return {w: ckpt.read(f"{stem}{p}_proj.weight", device).float() for w, p in PROJ.items()}

    return load


def fit(args) -> None:
    device = torch.device(args.device)
    with np.load(args.importance) as z:
        imp, rows, mass = {s: z[s] for s in ("hidden", "down")}, z["layers"].tolist(), z["mass"]
    layers = rows if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    # experts in file order: all, or the --top heaviest by calibration mass (ties by id)
    experts = {layer: (np.argsort(-mass[rows.index(layer)], kind="stable")[: args.top] if args.top
                       else np.arange(mass.shape[1])) for layer in layers}
    fit_layers(Search(device), loader(args.ckpt, device), layers, {layer: e.tolist() for layer, e in experts.items()},
               {layer: {s: a[rows.index(layer)][experts[layer]] for s, a in imp.items()} for layer in layers},
               args.out, args.iterations, args.code_bits, args.residual_of)


def check(args) -> None:
    """Rows of a projection fitted by ``Search`` and by KEEP's numpy fit (2-bit, then the 1-bit residual)."""
    from keep_quant import decode_codes

    from keep.convert.dsv4_vq_pilot import fit_projection, reconstruct_quantized

    search, rng = Search("cpu"), np.random.default_rng(0)
    if args.ckpt:
        imp = {"hidden": np.ones(4096), "down": np.ones(2048)}  # without an importance file, uniform
        if args.importance:
            with np.load(args.importance) as z:
                row = z["layers"].tolist().index(args.layer)
                imp = {s: z[s][row, args.expert] for s in ("hidden", "down")}
        weights = loader(args.ckpt, "cpu")(args.layer, args.expert)
        weights = {proj: w[: args.rows].numpy() for proj, w in weights.items()}
    else:  # random weights at GLM's shapes, heavy-tailed importance
        weights = {"w1": rng.standard_normal((args.rows, 4096)), "w3": rng.standard_normal((args.rows, 4096)),
                   "w2": rng.standard_normal((args.rows, 2048))}
        imp = {"hidden": rng.gamma(0.5, size=4096), "down": rng.gamma(0.5, size=2048)}
    for proj, space in PROJ_SPACE.items():
        w, d = weights[proj].astype(np.float32), imp[space].astype(np.float32)
        base = 0.0
        for bits in (16, 8):
            target = w - base
            codes, scales, wrel = search.fit(torch.from_numpy(target), torch.from_numpy(d), 512, args.iterations, bits)
            q, metrics, _ = fit_projection(target, d, group_size=512, code_bits=bits, iterations=args.iterations,
                                           backend="numpy")
            same = float((codes == np.asarray(q.codes)).mean())
            decoded = search.decode(codes, scales, bits).numpy()
            want = decode_codes({f"{proj}_codes": codes[None], f"{proj}_scales": scales[None], "code_bits": bits}, proj, 0)
            assert np.array_equal(decoded, want), "torch decode differs from keep_quant.decode_codes"
            print(f"{proj} {bits}-bit: codes equal {same:.4f}, weighted error {wrel:.6f} "
                  f"(numpy {metrics.weighted_relative_mse:.6f})")
            assert same > 0.99 and abs(wrel - metrics.weighted_relative_mse) < 1e-3 * metrics.weighted_relative_mse
            base = base + reconstruct_quantized(q)
    print("check ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "check"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", type=Path)
        s.add_argument("--importance", type=Path)
        s.add_argument("--out", type=Path)
        s.add_argument("--layers", default="all", help="'all' (3 to 44) or comma-separated layer ids")
        s.add_argument("--iterations", type=int, default=8)
        s.add_argument("--code-bits", type=int, default=16, choices=(8, 16))
        s.add_argument("--residual-of", type=Path)
        s.add_argument("--top", type=int, default=0, help="only the N experts of largest calibration mass per layer")
        s.add_argument("--layer", type=int, default=3)
        s.add_argument("--expert", type=int, default=0)
        s.add_argument("--rows", type=int, default=64)
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    with torch.no_grad():
        {"fit": fit, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
