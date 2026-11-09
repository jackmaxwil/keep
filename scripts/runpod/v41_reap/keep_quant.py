"""KEEP E8P quantization of a local REAP checkpoint's routed experts, one layer at a time.

The source is the checkpoint's FP4 experts, decoded exactly. Each projection is fitted with
KEEP's importance-aware E8P fit (``fit_projection``: 16-bit codes, weighted least-squares group
scales), with a per-input-channel importance diagonal:

- ``--importance pod``: the full model's calibration imatrix from the Runpod run
  (``calibration/sessions/cal-*.npz``, summed), indexed by each survivor's original expert id.
  This is what KEEP would use without re-calibrating after the prune.
- ``--importance PATH``: an imatrix npz measured on the pruned model itself (``local_calib.py``),
  already in the checkpoint's local expert order.

gate/up (w1/w3, input 5120) use group size 512 (2.031 bpw); down (w2, input 2304, which 512
does not divide) uses 256 (2.0625 bpw). ``--code-bits 8`` uses KEEP's 1-bit E8 codebook at the
same group sizes instead (1.031 / 1.0625 bpw). ``--residual-of DIR`` fits the residual of another
run's codes (``W - decode(DIR)``) instead of ``W``: a nested code, whose base is usable alone and
base + residual is the higher-precision tier; ``total_wrel_mse`` scores the sum against ``W``. Output: one ``layer-XX.npz`` per layer with
``{w1,w2,w3}_codes`` (uint16, or uint8 for 1-bit codes) ``[E, out, in/8]`` and ``{w1,w2,w3}_scales`` fp16 ``[E, out, in/g]``,
plus ``metrics.jsonl``. Finished layers are skipped, so the run can resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parents[2] / "src"))

PROJ_SPACE = {"w1": "hidden", "w3": "hidden", "w2": "down"}


def pod_importance(stats: Path) -> dict[str, np.ndarray]:
    """Summed ``importance_sum__{hidden,down}`` [L, 384, in] over the calibration sessions."""

    total: dict[str, np.ndarray] = {}
    for path in sorted((stats / "calibration" / "sessions").glob("cal-*.npz")):
        with np.load(path) as z:
            for space in ("hidden", "down"):
                a = z[f"importance_sum__{space}"].astype(np.float64)
                total[space] = total.get(space, 0) + a
    assert total, "no calibration sessions found"
    return total


def decode_fp4(weight, scale) -> np.ndarray:
    import torch_kernel

    w = torch_kernel.decode_e2m1(weight)  # [out, in] fp32
    return (w.unflatten(-1, (-1, 32)) * scale.float()[..., None]).flatten(-2).numpy()


def decode_codes(z, proj: str, e: int) -> np.ndarray:
    """One expert's projection from a keep_quant layer file, as ``local_eval.expert_override`` decodes it."""

    from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

    grid = (e8p_full_grid() if int(z["code_bits"]) == 16 else e8_1bit_grid()).astype(np.float32)
    vec = grid[z[f"{proj}_codes"][e].astype(np.int64)]  # [out, in/8, 8]
    scales = z[f"{proj}_scales"][e].astype(np.float32)
    vec = vec * np.repeat(scales, vec.shape[1] // scales.shape[-1], axis=-1)[..., None]
    return vec.reshape(vec.shape[0], -1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", required=True, type=Path)
    p.add_argument("--checkpoint-survivors", required=True, type=Path)
    p.add_argument("--stats", type=Path, help="local REAP-stats copy (for --importance pod)")
    p.add_argument("--importance", required=True, help="'pod' or a local imatrix npz")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", default="all", help="'all' or comma-separated layer ids")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--backend", default="metal", help="E8P search backend: metal or numpy")
    p.add_argument("--code-bits", type=int, default=16, choices=(8, 16),
                   help="16: E8P, 2 bits/weight + scales; 8: E8 1-bit codebook, 1 bit/weight + scales")
    p.add_argument("--experts", type=int, default=0, help="debug: fit only the first N experts per layer")
    p.add_argument("--residual-of", type=Path, help="fit W minus this keep_quant run's decoded codes (nested tier)")
    args = p.parse_args(argv)

    import stream
    from keep.convert.dsv4_vq_pilot import fit_projection, projection_error, reconstruct_quantized

    survivors = json.loads(args.checkpoint_survivors.read_text())["survivors"]
    ckpt = stream.HFCheckpoint(args.hf_path)
    n_layers = len(survivors)
    layers = range(n_layers) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if args.importance == "pod":
        full = pod_importance(args.stats)
        imp = {space: np.stack([full[space][l][sorted(survivors[str(l)])] for l in range(n_layers)]) for space in full}
    else:
        with np.load(args.importance) as z:
            imp = {space: z[f"importance__{space}"].astype(np.float64) for space in ("hidden", "down")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args) | {"hf_path": str(args.hf_path), "out": str(args.out),
                                                                     "checkpoint_survivors": str(args.checkpoint_survivors),
                                                                     "stats": str(args.stats)}, indent=2, default=str))

    for layer in layers:
        dest = args.out / f"layer-{layer:02d}.npz"
        if dest.exists():
            continue
        t0 = time.time()
        sd = ckpt.load(f"layers.{layer}.ffn.experts.")
        n_exp = len(survivors[str(layer)]) if not args.experts else args.experts
        arrays: dict[str, list[np.ndarray]] = {}
        errs = {k: [] for k in PROJ_SPACE}
        total = {k: [] for k in PROJ_SPACE}
        base = np.load(args.residual_of / f"layer-{layer:02d}.npz") if args.residual_of else None
        for e in range(n_exp):
            for proj, space in PROJ_SPACE.items():
                w = decode_fp4(sd[f"{e}.{proj}.weight"], sd[f"{e}.{proj}.scale"])
                group = 512 if w.shape[1] % 512 == 0 else 256
                d = imp[space][layer, e].astype(np.float32)
                b = decode_codes(base, proj, e) if base is not None else 0.0
                q, metrics, _ = fit_projection(w - b, d, group_size=group,
                                               code_bits=args.code_bits, iterations=args.iterations, backend=args.backend)
                if base is not None:
                    total[proj].append(projection_error(w, b + reconstruct_quantized(q), d).weighted_relative_mse)
                # 1-bit E8 codes are one byte; E8P codes are two
                arrays.setdefault(f"{proj}_codes", []).append(np.asarray(q.codes, dtype=np.uint8 if args.code_bits == 8 else np.uint16))
                arrays.setdefault(f"{proj}_scales", []).append(np.asarray(q.scales, dtype=np.float16))
                errs[proj].append(metrics.weighted_relative_mse if hasattr(metrics, "weighted_relative_mse") else float("nan"))
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, code_bits=np.asarray(args.code_bits), **{k: np.stack(v) for k, v in arrays.items()})
        tmp.rename(dest)
        row = {"layer": layer, "experts": n_exp, "seconds": round(time.time() - t0, 1),
               **{f"{k}_wrel_mse": float(np.mean(v)) for k, v in errs.items()},
               **{f"{k}_total_wrel_mse": float(np.mean(v)) for k, v in total.items() if v}}
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
