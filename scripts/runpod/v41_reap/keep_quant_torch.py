"""``keep_quant.py``'s 2-bit E8P fit on torch, for the experts a REAP prune removed.

``keep_quant.py`` fits the survivors of a pruned checkpoint with KEEP's Metal search, on a Mac.
This fits the other experts from the full checkpoint on a CUDA pod, so the cache's pool can be
all 384. The fit is the same alternation as ``quantize_weight_importance_aware`` with its
default policy (weighted nearest E8P codeword under the per-input-channel importance, then the
exact weighted least-squares group scale; 16-bit codes; group 512 for w1/w3, 256 for w2), in
torch. Rows are independent, so ``check`` can compare a few rows against a ``keep_quant`` file.

    # Mac: the importance of the removed experts, one small file to upload
    keep_quant_torch.py importance --stats STATS --survivors survivors-192.json --out imp-pruned.npz
    # Mac: the torch fit against stored codes of the survivors (fitted by the Metal search)
    keep_quant_torch.py check --hf-path PRUNED_CKPT --survivors survivors-192.json --stats STATS --codes QUANT_DIR
    # pod: fit, one process per GPU (--layers splits the work); finished layers are skipped
    keep_quant_torch.py fit --hf-path FULL_CKPT --importance imp-pruned.npz --out OUT [--layers 0,1,...]
    # pod: every router row of the full checkpoint
    keep_quant_torch.py router --hf-path FULL_CKPT --out router-384.npz

``fit`` writes ``layer-XX.npz`` in ``keep_quant``'s format (experts in the order of
``imp-pruned.npz``'s ``experts`` row, which ``experts.json`` repeats) and ``metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parents[2] / "src"))

from keep.vq.e8 import E8P_SHUFFLE_MAP, e8p_abs_grid, e8p_full_grid

PROJ_SPACE = {"w1": "hidden", "w3": "hidden", "w2": "down"}


def _parity(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    return (x ^ (x >> 1)) & 1


class Search:
    """Weighted nearest E8P codeword; ``e8._encode_e8p_rtn_per_row_diagonal_hessian`` in torch."""

    def __init__(self, device, vectors_per_chunk: int = 1 << 17):
        base = e8p_abs_grid().astype(np.float32)  # signed base rows [256, 8]
        self.abs_rows = torch.from_numpy(np.abs(base)).to(device)
        self.bit = torch.from_numpy(E8P_SHUFFLE_MAP.astype(np.int32)).to(device)  # dim -> sign bit
        self.base_signs = ((torch.from_numpy(base < 0).to(device).int() << self.bit).sum(-1)).int()
        self.grid = torch.from_numpy(e8p_full_grid().astype(np.float32)).to(device)  # [65536, 8]
        self.vectors_per_chunk = vectors_per_chunk

    def assign(self, vectors: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """vectors [rows, P, 8] (already divided by their scales), w [P, 8] -> codes [rows, P] int32."""
        abs_norm_w = w @ (self.abs_rows * self.abs_rows).T  # [P, 256]
        step = max(1, self.vectors_per_chunk // vectors.shape[1])
        return torch.cat([self._assign(vectors[i:i + step], w, abs_norm_w) for i in range(0, vectors.shape[0], step)])

    def _assign(self, chunk, w, abs_norm_w):
        best_dist = torch.full(chunk.shape[:2], torch.inf, device=chunk.device)
        best_code = torch.zeros(chunk.shape[:2], dtype=torch.int32, device=chunk.device)
        for parity, shift in ((0, 0.25), (1, -0.25)):
            target = chunk - shift
            weighted = target.abs() * w  # [R, P, 8]
            dist = (target.abs() * weighted).sum(-1, keepdim=True) + abs_norm_w - 2.0 * (weighted @ self.abs_rows.T)
            signs = ((target < 0).int() << self.bit).sum(-1, dtype=torch.int32)[..., None] ^ self.base_signs  # [R, P, 256]
            odd = _parity(signs).bool()
            # An odd sign count is fixed by flipping the dimension that costs least.
            dist = torch.where(odd, dist + (4 * weighted[..., None, :] * self.abs_rows).amin(-1), dist)
            row = dist.argmin(-1, keepdim=True)
            row_dist, signs, odd = dist.gather(-1, row)[..., 0], signs.gather(-1, row)[..., 0], odd.gather(-1, row)[..., 0]
            row = row[..., 0]
            penalty = 4 * weighted * self.abs_rows[row]  # [R, P, 8]
            flipped = signs[..., None] ^ (1 << self.bit)
            tied = penalty == penalty.amin(-1, keepdim=True)
            flip = torch.where(tied, flipped, 65535).argmin(-1)  # ties: the lowest resulting code
            signs = torch.where(odd, signs ^ (1 << self.bit[flip]), signs)
            code = (row.int() << 8) | (signs ^ parity)
            take = (row_dist < best_dist) | ((row_dist == best_dist) & (code < best_code))
            best_dist, best_code = torch.where(take, row_dist, best_dist), torch.where(take, code, best_code)
        return best_code

    def fit(self, weight: torch.Tensor, importance: torch.Tensor, group: int, iterations: int = 8):
        """weight [out, in] fp32, importance [in] -> (codes uint16 [out, in/8], scales fp16 [out, in/group],
        weighted relative squared error of the stored code)."""
        out_dim, in_dim = weight.shape
        grouped = weight.reshape(out_dim, in_dim // group, group // 8, 8)
        h = importance.reshape(in_dim // group, group // 8, 8)
        w = importance.reshape(-1, 8)
        scales = grouped.abs().amax((2, 3)) / self.grid.abs().max()
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))

        def assign():
            return self.assign((grouped / scales[:, :, None, None]).reshape(out_dim, -1, 8), w).reshape(grouped.shape[:3])

        for _ in range(iterations):
            decoded = self.grid[assign().long()]
            numerator = (h * grouped * decoded).double().sum((2, 3))
            denominator = (h * decoded * decoded).double().sum((2, 3))
            ok = (denominator > 0) & (numerator > 0)
            scales = torch.where(ok, numerator / denominator.clamp_min(1e-300), scales.double()).float()
        codes = assign()
        stored = scales.half()
        error = self.grid[codes.long()] * stored.float()[:, :, None, None] - grouped
        wrel = float((h * error * error).double().sum() / (h * grouped * grouped).double().sum())
        return codes.reshape(out_dim, in_dim // 8).cpu().numpy().astype(np.uint16), stored.cpu().numpy(), wrel


def decode_fp4(weight: torch.Tensor, scale: torch.Tensor, device) -> torch.Tensor:
    import torch_kernel

    # The packed dtypes leave the CPU as bytes and floats; not every backend has them.
    w = torch_kernel.decode_e2m1(weight.contiguous().view(torch.uint8).to(device))  # [out, in] fp32
    return (w.unflatten(-1, (-1, 32)) * scale.float().to(device)[..., None]).flatten(-2)


def load_expert(ckpt, layer: int, expert: int, device) -> dict[str, torch.Tensor]:
    sd = ckpt.load(f"layers.{layer}.ffn.experts.{expert}.")
    return {proj: decode_fp4(sd[f"{proj}.weight"], sd[f"{proj}.scale"], device) for proj in PROJ_SPACE}


def importance(args) -> None:
    from keep_quant import pod_importance

    # Without --survivors: every expert, in order.
    survivors = json.loads(args.survivors.read_text())["survivors"] if args.survivors else {}
    full = pod_importance(args.stats)  # {space: [L, 384, in]}
    n_layers, n_experts = full["hidden"].shape[:2]
    removed = np.array([sorted(set(range(n_experts)) - set(survivors.get(str(layer), ()))) for layer in range(n_layers)])
    out, replaced = {"experts": removed}, 0
    for space, total in full.items():
        picked = np.stack([total[layer][removed[layer]] for layer in range(n_layers)])
        # The search needs some weight in every group of eight inputs. An expert that calibration
        # never routed has none: it gets its layer's mean over all experts.
        dead = ~(picked.reshape(*picked.shape[:2], -1, 8) > 0).any(-1).all(-1)  # [L, E]
        picked[dead] = np.broadcast_to(total.mean(1)[:, None, :], picked.shape)[dead]
        replaced += int(dead.sum())
        assert (picked.reshape(*picked.shape[:2], -1, 8) > 0).any(-1).all() and np.isfinite(picked).all(), space
        out[space] = picked.astype(np.float32)
    np.savez(args.out, **out)
    print(f"{args.out}: {removed.shape[1]} experts in each of {n_layers} layers; "
          f"{replaced} (expert, space) importances replaced by the layer mean")


def check(args) -> None:
    """A few rows of some surviving experts, fitted here, against the stored Metal-search codes."""
    import stream
    from keep_quant import pod_importance

    device = torch.device(args.device)
    survivors = json.loads(args.survivors.read_text())["survivors"]
    full, ckpt, search = pod_importance(args.stats), stream.HFCheckpoint(args.hf_path), Search(device)
    grid = e8p_full_grid()
    for layer, expert in ((0, 3), (20, 0), (39, 191)):
        z = np.load(args.codes / f"layer-{layer:02d}.npz")
        weights = load_expert(ckpt, layer, expert, device)
        original = sorted(survivors[str(layer)])[expert]
        for proj, space in PROJ_SPACE.items():
            w = weights[proj][: args.rows]
            d = torch.from_numpy(full[space][layer][original].astype(np.float32)).to(device)
            started = time.time()
            codes, _, wrel = search.fit(w, d, 512 if w.shape[1] % 512 == 0 else 256)
            seconds = time.time() - started
            ref_codes, ref_scales = z[f"{proj}_codes"][expert, : args.rows], z[f"{proj}_scales"][expert, : args.rows]
            group = w.shape[1] // ref_scales.shape[1]
            ref = (grid[ref_codes].reshape(args.rows, -1, group // 8, 8) * ref_scales.astype(np.float32)[:, :, None, None])
            ref = torch.from_numpy(ref.reshape(args.rows, -1)).to(device)
            ref_wrel = float((d * (ref - w) ** 2).double().sum() / (d * w * w).double().sum())
            same = float((codes == ref_codes).mean())
            print(f"layer {layer} expert {expert} {proj}: codes equal {same:.4f}, weighted error {wrel:.6f} "
                  f"(stored {ref_wrel:.6f}), {seconds:.2f} s for {args.rows} rows")
            assert same > 0.99 and abs(wrel - ref_wrel) < 1e-3 * ref_wrel, (same, wrel, ref_wrel)


def fit(args) -> None:
    import stream

    device = torch.device(args.device)
    with np.load(args.importance) as z:
        imp = {"hidden": z["hidden"], "down": z["down"]}
        experts = z["experts"]
    ckpt, search = stream.HFCheckpoint(args.hf_path), Search(device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "experts.json").write_text(json.dumps({"experts": experts.tolist(), "source": str(args.hf_path)}))
    layers = range(experts.shape[0]) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    for layer in layers:
        dest = args.out / f"layer-{layer:02d}.npz"
        if dest.exists():
            continue
        started = time.time()
        arrays: dict[str, list[np.ndarray]] = {}
        errors: dict[str, list[float]] = {proj: [] for proj in PROJ_SPACE}
        for slot, expert in enumerate(experts[layer][: args.experts or None].tolist()):
            weights = load_expert(ckpt, layer, expert, device)
            for proj, space in PROJ_SPACE.items():
                w = weights[proj]
                codes, scales, wrel = search.fit(w, torch.from_numpy(imp[space][layer, slot]).to(device),
                                                 512 if w.shape[1] % 512 == 0 else 256, args.iterations)
                arrays.setdefault(f"{proj}_codes", []).append(codes)
                arrays.setdefault(f"{proj}_scales", []).append(scales)
                errors[proj].append(wrel)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, code_bits=np.asarray(16), **{k: np.stack(v) for k, v in arrays.items()})
        tmp.rename(dest)
        row = {"layer": layer, "experts": len(errors["w1"]), "seconds": round(time.time() - started, 1),
               **{f"{proj}_wrel_mse": float(np.mean(v)) for proj, v in errors.items()}}
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def router(args) -> None:
    import stream

    ckpt, rows = stream.HFCheckpoint(args.hf_path), {}
    for layer in range(args.n_layers):
        for name, tensor in ckpt.load(f"layers.{layer}.ffn.gate.").items():
            rows.setdefault(name, []).append(tensor.float().numpy())  # bf16 widens to fp32 without loss
    np.savez(args.out, **{name: np.stack(v) for name, v in rows.items()})
    print({name: np.stack(v).shape for name, v in rows.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("importance", "check", "fit", "router"):
        s = sub.add_parser(name)
        s.add_argument("--hf-path", type=Path)
        s.add_argument("--survivors", type=Path)
        s.add_argument("--stats", type=Path)
        s.add_argument("--codes", type=Path)
        s.add_argument("--importance", type=Path)
        s.add_argument("--out", type=Path)
        s.add_argument("--layers", default="all")
        s.add_argument("--iterations", type=int, default=8)
        s.add_argument("--experts", type=int, default=0, help="debug: only the first N experts of a layer")
        s.add_argument("--rows", type=int, default=64)
        s.add_argument("--n-layers", type=int, default=40)
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = p.parse_args()
    with torch.no_grad():
        {"importance": importance, "check": check, "fit": fit, "router": router}[args.command](args)


if __name__ == "__main__":
    main()
