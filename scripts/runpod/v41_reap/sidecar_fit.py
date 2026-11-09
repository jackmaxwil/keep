"""Step 3: layer-local residual sidecars for a pruned, KEEP-quantized V4.1 student, in one pass.

The student is a survivors set (base expert ids) whose experts run from KEEP codes as bf16 GEMMs.
The teacher is the same layer with every expert the checkpoint holds, at its own FP4 weights. In
one layer-streamed pass over the student's own hidden states, each layer's MoE computes both:

    r = y_teacher(x) - y_student(x)          routed experts only; the shared expert cancels

and fits a reduced-rank linear map x~ = [x, 1] -> r by ridge regression on the fit tokens:

    W = (X~'X~ + lam * mean(diag) * I)^-1 X~'R,  then the best rank-k W V V' (V: top eigvecs of W'X~'X~W)

The rank-k sidecar is added to the student's MoE output before the next layer runs, so every
later layer is fit on inputs that already carry the earlier sidecars (sequential, like GPTQ).
Held-out records score each layer: the share of teacher output energy the residual holds,
before and after the sidecar.

On the pod the checkpoint is the full 384-expert source; on the Mac, REAP-50 stands in as a
proxy teacher for smoke tests. Student codes come from ``pack-codes`` (only the student's
experts, keyed by base id), so the pod never needs the full code directory:

    python sidecar_fit.py pack-codes --codes ~/Models/v41-reap-run/quant/e8p-podimp \\
        --checkpoint-survivors ~/Models/v41-reap-run/stats/survivors-192.json \\
        --student ~/Models/v41-reap-run/alloc/sel/prunedmass-96.json --out ~/Models/v41-reap-run/sidecar/codes

    python sidecar_fit.py fit --hf-path HF --pack PACK --records RECORDS.json \\
        --student SURVIVORS.json --codes CODES_DIR --rank 256 --out OUT

``RECORDS.json`` is a list of ``session#index`` names; every ``--holdout-every``-th one is held out.
Writes ``OUT/layer-LL.npz`` (U [dim+1, k], Vt [k, dim], fp32; optional X~'X~ and X~'R with
``--save-stats``) and ``OUT/metrics.jsonl``.
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


def local_ids(hf_path: Path, ckpt_survivors: Path | None, n_experts: int) -> dict[int, list[int]]:
    """Per layer, the base expert id of each local expert slot in this checkpoint."""

    if not (hf_path / "keep-prune.json").exists():
        return {layer: list(range(n_experts)) for layer in range(64)}
    assert ckpt_survivors is not None, "a pruned checkpoint needs --checkpoint-survivors"
    return {int(k): sorted(v) for k, v in json.loads(ckpt_survivors.read_text())["survivors"].items()}


def pack_codes(args: argparse.Namespace) -> int:
    have = {int(k): sorted(v) for k, v in json.loads(args.checkpoint_survivors.read_text())["survivors"].items()}
    want = {int(k): sorted(v) for k, v in json.loads(args.student.read_text())["survivors"].items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for layer in sorted(want):
        local = [have[layer].index(e) for e in want[layer]]
        with np.load(args.codes / f"layer-{layer:02d}.npz") as z:
            out = {"ids": np.asarray(want[layer], dtype=np.int32),
                   "code_bits": z["code_bits"] if "code_bits" in z.files else np.asarray(16)}
            for proj in ("w1", "w2", "w3"):
                out[f"{proj}_codes"] = z[f"{proj}_codes"][local]
                out[f"{proj}_scales"] = z[f"{proj}_scales"][local]
        np.savez(args.out / f"layer-{layer:02d}.npz", **out)
    print(json.dumps({"layers": len(want), "experts": int(np.mean([len(v) for v in want.values()]))}))
    return 0


class LayerCodes:
    """One ``pack-codes`` layer file, held as codes; ``[base id]`` decodes that expert to bf16.

    Decoding on use keeps only one expert's bf16 weights alive at a time (all 96 of a layer, twice
    over with the prefetch, cost ~13 GB of host RAM and tripped the Mac's 80 GB guard)."""

    def __init__(self, path: Path, torch, grids):
        self.torch = torch
        with np.load(path) as z:
            self.grid = grids[int(z["code_bits"])]
            self.row = {e: i for i, e in enumerate(z["ids"].tolist())}
            self.arrays = {k: z[k] for k in z.files if k.endswith(("_codes", "_scales"))}

    def __getitem__(self, e: int) -> dict[str, object]:
        torch, i, out = self.torch, self.row[e], {}
        with torch.device("cpu"):
            for proj in ("w1", "w2", "w3"):
                vec = self.grid[torch.as_tensor(self.arrays[f"{proj}_codes"][i].astype(np.int64))]
                scales = torch.as_tensor(self.arrays[f"{proj}_scales"][i].astype(np.float32))
                vec = vec * scales.repeat_interleave(vec.shape[1] // scales.shape[-1], dim=-1)[..., None]
                out[proj] = vec.flatten(-2).bfloat16()
        return out


def reduced_rank(A, B, lam: float, rank: int):
    """Ridge W = (A + reg)^-1 B, cut to the rank-k map that keeps the most fitted output energy."""

    import torch

    n = A.shape[0]
    reg = lam * A.diagonal()[:-1].mean()
    W = torch.linalg.solve(A + reg * torch.eye(n, dtype=A.dtype, device=A.device), B)  # [n, d]
    M = W.T @ A @ W
    _, V = torch.linalg.eigh((M + M.T) / 2)
    V = V[:, -rank:].flip(-1)  # largest first, so U[:, :j] @ Vt[:j] is the best rank-j map
    return W @ V, V.T  # U [n, k], Vt [k, d]


def fit(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    import collect
    import stream
    from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

    cuda = args.device == "cuda"
    if not cuda:
        import torch_kernel

        torch_kernel.exact_quant = True
        ref = stream.import_reference(args.hf_path / "inference", kernel=torch_kernel)
    else:
        ref = stream.import_reference(args.hf_path / "inference")
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    model_args = ref.ModelArgs(**json.loads((args.hf_path / "inference" / "config.json").read_text()))
    model_args.max_batch_size = 1
    dim, n_layers = int(model_args.dim), int(model_args.n_layers)

    sessions = collect.load_sessions(args.pack)
    names = json.loads(args.records.read_text())
    records = []
    for name in names:
        session, _, idx = name.partition("#")
        r = sessions[session][int(idx or 0)]
        records.append(dict(r, token_ids=r["token_ids"][: args.max_tokens]))
    model_args.max_seq_len = max(len(r["token_ids"]) for r in records)
    holdout = torch.cat([torch.full((len(r["token_ids"]),), i % args.holdout_every == args.holdout_every - 1)
                         for i, r in enumerate(records)]).to(args.device)

    ckpt = stream.HFCheckpoint(args.hf_path)
    layout = ref.EngramLayout.from_args(model_args)
    with torch.device(args.device):
        hash_state = ref.NgramHashState(model_args, layout, AutoTokenizer.from_pretrained(args.hf_path))
    engram_rows = {l: stream.EngramRows.from_checkpoint(ckpt, l, ref.fp8_block_size) for l in model_args.engram_layer_ids}
    streamer = stream.LayerStreamer(ref, model_args, ckpt.load, engram_rows, hash_state, args.device)

    ids = local_ids(args.hf_path, args.checkpoint_survivors, int(model_args.n_routed_experts))
    student = {int(k): v for k, v in json.loads(args.student.read_text())["survivors"].items()}
    grids = {16: torch.as_tensor(e8p_full_grid().astype(np.float32), device="cpu"),
             8: torch.as_tensor(e8_1bit_grid().astype(np.float32), device="cpu")}
    decoded: dict[int, dict] = {}
    load_layer = streamer.load_layer

    def load_with_codes(prefix, skip=()):  # prefetch thread: the next layer's student weights too
        sd = load_layer(prefix, skip)
        if prefix.startswith("layers."):
            layer = int(prefix.split(".")[1])
            decoded[layer] = LayerCodes(args.codes / f"layer-{layer:02d}.npz", torch, grids)
        return sd

    streamer.load_layer = load_with_codes
    args.out.mkdir(parents=True, exist_ok=True)
    # Resume: a layer with a metrics row keeps its saved sidecar; only the student is run through it.
    done = {json.loads(line)["layer"] for line in (args.out / "metrics.jsonl").read_text().splitlines()} \
        if (args.out / "metrics.jsonl").exists() else set()
    metrics = (args.out / "metrics.jsonl").open("a")

    def expert(x, w, w1, w2, w3, limit):
        gate, up = F.linear(x, w1).float(), F.linear(x, w3).float()
        if limit > 0:
            up, gate = up.clamp(-limit, limit), gate.clamp(max=limit)
        return F.linear((w * (F.silu(gate) * up)).to(x.dtype), w2)

    def moe_forward(self, x, image_mask=None):
        t0 = time.time()
        shape, layer = x.size(), self.layer_id
        x = x.view(-1, dim)
        local = ids[layer]
        # This layer's Engram lookups are done. Unmapping the table drops the pages they touched
        # (tens of GB at 150k tokens) from RSS; left mapped, layers 1 and 14 together passed 80 GB.
        engram_rows.pop(layer, None)
        n = x.size(0)

        def blocks():  # [x, 1] in row blocks, so no full fp32 copy of x is ever resident
            for s in range(0, n, args.block_rows):
                e = min(n, s + args.block_rows)
                yield s, e, torch.cat([x[s:e].float(), torch.ones(e - s, 1, device=x.device)], dim=1)

        full_bias = self.gate.bias.data.clone()
        if layer not in done:
            weights, indices = self.gate(x, None)  # teacher: every expert in the checkpoint
            y_t = torch.zeros_like(x, dtype=torch.float32)
            for i in torch.unique(indices).tolist():
                idx, top = torch.where(indices == i)
                y_t[idx] += self.experts[i](x[idx], weights[idx, top, None]).float()
        keep = set(student[layer])
        self.gate.bias.data[torch.as_tensor([e not in keep for e in local], device=x.device)] = float("-inf")
        weights, indices = self.gate(x, None)  # student: masked router, KEEP-decoded experts
        self.gate.bias.data.copy_(full_bias)
        weights_of = decoded.pop(layer)
        y_s = torch.zeros_like(x, dtype=torch.float32)
        limit = self.experts[0].swiglu_limit if self.experts[0] is not None else 0.0
        for i in torch.unique(indices).tolist():
            idx, top = torch.where(indices == i)
            w = {p: t.to(x.device) for p, t in weights_of[local[i]].items()}
            y_s[idx] += expert(x[idx], weights[idx, top, None], w["w1"], w["w2"], w["w3"], limit).float()

        if layer in done:
            if args.apply:
                with np.load(args.out / f"layer-{layer:02d}.npz") as z:
                    U, Vt = torch.as_tensor(z["U"], device=x.device), torch.as_tensor(z["Vt"], device=x.device)
                for s, e, xt in blocks():
                    y_s[s:e] += (xt @ U) @ Vt
            print(json.dumps({"layer": layer, "resumed": True, "seconds": round(time.time() - t0, 1)}), flush=True)
            y_s += self.shared_experts(x).float()
            return y_s.type_as(x).view(shape)

        energy = {s: float(y_t[m].pow(2).sum()) for s, m in (("fit", ~holdout), ("holdout", holdout))}
        r = y_t.sub_(y_s)  # in place: y_t is not needed past its energy
        del y_t
        A = torch.zeros(dim + 1, dim + 1, dtype=torch.float64, device=x.device)
        B = torch.zeros(dim + 1, dim, dtype=torch.float64, device=x.device)

        for s, e, xt in blocks():
            m = ~holdout[s:e]
            A += (xt[m].T @ xt[m]).double()
            B += (xt[m].T @ r[s:e][m]).double()
        U, Vt = reduced_rank(A, B, args.ridge, args.rank)
        U, Vt = U.float(), Vt.float()
        before = {"fit": 0.0, "holdout": 0.0}
        after = {"fit": 0.0, "holdout": 0.0}
        for s, e, xt in blocks():
            corr = (xt @ U) @ Vt
            for split, m in (("fit", ~holdout[s:e]), ("holdout", holdout[s:e])):
                before[split] += float(r[s:e][m].pow(2).sum())
                after[split] += float((r[s:e][m] - corr[m]).pow(2).sum())
            if args.apply:
                y_s[s:e] += corr
        del r
        row = {"layer": layer, "tokens_fit": int((~holdout).sum()), "tokens_holdout": int(holdout.sum()),
               "seconds": round(time.time() - t0, 1)}
        for split in ("fit", "holdout"):
            row[f"{split}_res_before"] = before[split] / energy[split]
            row[f"{split}_res_after"] = after[split] / energy[split]
        out = {"U": U.cpu().numpy(), "Vt": Vt.cpu().numpy(), "rank": np.asarray(args.rank), "ridge": np.asarray(args.ridge)}
        if args.save_stats:
            out |= {"XtX": A.float().cpu().numpy(), "XtR": B.float().cpu().numpy()}
        np.savez(args.out / f"layer-{layer:02d}.npz", **out)
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        print(json.dumps(row), flush=True)
        y_s += self.shared_experts(x).float()
        return y_s.type_as(x).view(shape)

    ref.MoE.forward = moe_forward
    items = [stream.Item(tokens=torch.as_tensor([r["token_ids"]], dtype=torch.long, device=args.device), group="student")
             for r in records]
    t0 = time.time()
    with torch.inference_mode():
        streamer.forward(items)
    summary = {"seconds": round(time.time() - t0, 1), "tokens": int(holdout.numel()), "layers": n_layers}
    if cuda:
        summary["max_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    print(json.dumps(summary), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("pack-codes")
    pc.add_argument("--codes", required=True, type=Path)
    pc.add_argument("--checkpoint-survivors", required=True, type=Path)
    pc.add_argument("--student", required=True, type=Path)
    pc.add_argument("--out", required=True, type=Path)
    f = sub.add_parser("fit")
    f.add_argument("--hf-path", required=True, type=Path)
    f.add_argument("--checkpoint-survivors", type=Path, help="only for a pruned checkpoint (the Mac proxy)")
    f.add_argument("--pack", required=True, type=Path)
    f.add_argument("--records", required=True, type=Path)
    f.add_argument("--max-tokens", type=int, default=1024)
    f.add_argument("--holdout-every", type=int, default=10)
    f.add_argument("--student", required=True, type=Path)
    f.add_argument("--codes", required=True, type=Path)
    f.add_argument("--rank", type=int, default=256)
    f.add_argument("--ridge", type=float, default=1e-3)
    f.add_argument("--no-apply", dest="apply", action="store_false", help="fit only; do not feed sidecars forward")
    f.add_argument("--save-stats", action="store_true")
    f.add_argument("--block-rows", type=int, default=32768, help="rows per [x, 1] block in the fit")
    f.add_argument("--device", default="cuda")
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--max-rss-gb", type=float, default=80.0)
    args = p.parse_args(argv)
    if args.cmd == "pack-codes":
        return pack_codes(args)
    from local_parity import rss_guard

    rss_guard(args.max_rss_gb)
    return fit(args)


def _check() -> None:
    import torch

    torch.manual_seed(0)
    x = torch.randn(4000, 16, dtype=torch.float64)
    xt = torch.cat([x, torch.ones(4000, 1, dtype=torch.float64)], 1)
    true = torch.randn(17, 3, dtype=torch.float64) @ torch.randn(3, 16, dtype=torch.float64)  # rank 3
    r = xt @ true
    U, Vt = reduced_rank(xt.T @ xt, xt.T @ r, 1e-9, 3)
    assert torch.allclose(U @ Vt, true, atol=1e-6), (U @ Vt - true).abs().max()
    U3, Vt3 = U, Vt
    U, Vt = reduced_rank(xt.T @ xt, xt.T @ r, 1e-9, 1)
    assert ((xt @ U @ Vt - r).pow(2).sum() / r.pow(2).sum()) < 1  # rank 1 keeps part of it
    assert torch.allclose(U3[:, :1] @ Vt3[:1], U @ Vt, atol=1e-6)  # stored largest first: truncation works


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _check()
        print("ok")
        sys.exit(0)
    sys.exit(main())
