"""Plain and error-feedback 2-bit E8P codes for every expert of the source checkpoint, in one pass per GPU.

Calibration records go through the full model a layer at a time (``stream.LayerStreamer``, the reference's
code). At a layer this process owns (``--shard I/N``: the layers ``l % N == I``) the MoE's input rows are kept,
and each expert is fitted twice from the checkpoint's own FP4 weights:

  plain     ``keep_quant_torch.Search.fit``: the weighted nearest E8P point under the diagonal of the second
            moments of the rows the full model routed to the expert (their sum of squares, the importance
            ``keep_quant`` uses), with least-squares group scales.
  feedback  the same scales; codes by ``ldlq.ldlq_codes`` through the whole second-moment matrix of those rows
            (w1 and w3 through the layer input's, w2 through the GLU activation's).

No layer's moments are stored: an expert's matrices are built from its rows, used and dropped.

The feedback fit's damping is chosen per layer and input space on rows the fit did not see. For ``--holdout``
experts, spread over the layer's routed counts, the records are split in two; codes fitted on one half's
moments at each ``--damp`` are scored on the other half's, as output error ``tr(E H E^T) / tr(W H W^T)``. The
best damping is used for the layer if it beats the plain fit there; otherwise that space's feedback codes are
the plain ones. Every expert's final fit uses all rows. An expert with fewer than ``--min-rows`` rows is fitted
plain under the mean diagonal of the layer's other experts, in both outputs.

    fit_source.py                      # self-check of the feedback path on synthetic inputs, no checkpoint
    CUDA_VISIBLE_DEVICES=I fit_source.py --hf-path CKPT --pack v41-pack.jsonl --out OUT --shard I/N

OUT/plain and OUT/feedback hold ``layer-XX.npz`` in ``keep_quant``'s format (all experts, in order);
OUT/imp/layer-XX.npz the diagonals and row counts; OUT/metrics-I.jsonl one row per layer with the held-out
errors and the damping chosen.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parents[2] / "src"))

import ldlq
from keep_quant_torch import PROJ_SPACE, Search, load_expert

SPACES = {"hidden": ("w1", "w3"), "down": ("w2",)}


def group_of(w: torch.Tensor) -> int:
    return 512 if w.shape[1] % 512 == 0 else 256


@contextlib.contextmanager
def fp32():
    """The reference runs with bf16 as torch's default dtype; the fits are written for fp32."""
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(before)


def fast_assign(search: Search):
    """``ldlq.nearest`` by the structured E8P search with unit weights: (vectors [n, 8], book) -> book rows."""
    ones = torch.ones(1, 8, dtype=torch.float32, device=search.grid.device)
    return lambda v, book: search.assign(v[:, None, :].float(), ones)[:, 0].long()


def stored(scales: np.ndarray, device) -> torch.Tensor:
    s = torch.from_numpy(scales.astype(np.float32)).to(device)
    return torch.where(s > 0, s, torch.ones_like(s))


def calibration(pack: Path, budget: int, cap: int) -> list[list[int]]:
    """Token ids of calibration records only: one record of each session in turn, cut to ``cap`` tokens,
    until ``budget`` tokens are in."""
    import collect

    sessions = [records for records in collect.load_sessions(pack).values() if records[0]["split"] == "calibration"]
    out: list[list[int]] = []
    for i in range(max(map(len, sessions))):
        for records in sessions:
            if i < len(records) and sum(map(len, out)) < budget:
                assert records[i]["split"] == "calibration"
                out.append(records[i]["token_ids"][:cap])
    return out


def moments(x: torch.Tensor, expert, chunk: int = 8192) -> dict[str, torch.Tensor]:
    """Sums of ``r r^T`` over an expert's routed rows ``x`` ([n, dim], the model's dtype): ``hidden`` for the rows
    themselves, ``down`` for the GLU activations the reference's expert computes from them (float64)."""
    import collect

    out: dict[str, torch.Tensor] = {}
    for part in x.split(chunk):
        act = collect.expert_forward_with_activation(
            F, torch, expert, part, torch.ones(part.shape[0], 1, dtype=torch.float32, device=part.device))[1]
        for space, rows in (("hidden", part.float()), ("down", act.float())):
            out[space] = out.get(space, 0) + (rows.T @ rows).double()
    return out


def heldout(search: Search, book, assign, weights, fit_h, score_h, damps) -> dict[str, dict]:
    """Per space: the output error on ``score_h`` of the plain fit and of the feedback fit at each damping, all
    fitted on ``fit_h``."""
    out = {}
    for space, projs in SPACES.items():
        w = torch.cat([weights[p] for p in projs])
        codes, scales, _ = search.fit(w, fit_h[space].diagonal().float(), group_of(w))
        row = {"plain": ldlq.output_error(w, search.decode(codes, scales), score_h[space])}
        for damp in damps:
            w_hat, _ = ldlq.ldlq_codes(w, fit_h[space], book, group_of(w), damp, stored(scales, w.device), assign)
            row[damp] = ldlq.output_error(w, w_hat, score_h[space])
        out[space] = row
    return out


def fit_expert(search: Search, book, assign, weights, diag, h, damp) -> dict[str, tuple]:
    """``{proj: (plain codes, scales, weighted error, feedback codes)}``. ``diag[space]`` weights the plain fit;
    a space with ``damp[space]`` set and ``h`` given gets feedback codes, the others repeat the plain ones."""
    out = {}
    for proj, space in PROJ_SPACE.items():
        codes, scales, wrel = search.fit(weights[proj], diag[space], group_of(weights[proj]))
        out[proj] = [codes, scales, wrel, codes]
    for space, projs in SPACES.items():
        if h is None or damp[space] is None:
            continue
        w = torch.cat([weights[p] for p in projs])
        s = torch.cat([stored(out[p][1], w.device) for p in projs])
        _, fed = ldlq.ldlq_codes(w, h[space], book, group_of(w), damp[space], s, assign)
        for p, part in zip(projs, fed.split(weights[projs[0]].shape[0])):
            out[p][3] = part.cpu().numpy().astype(np.uint16)
    return out


def fit_layer(layer: int, block, x: torch.Tensor, second_half: torch.Tensor, load, search: Search, args, out: Path) -> dict:
    """Fit every expert of ``block`` from its routed rows of ``x`` ([tokens, dim]); write the layer's files."""
    started = time.time()
    gate, experts = block.ffn.gate, block.ffn.experts
    n = args.experts or len(experts)
    routes = torch.cat([gate(part, None)[1] for part in x.split(1 << 15)])
    rows = [torch.where((routes == e).any(1))[0] for e in range(n)]
    counts = np.array([len(r) for r in rows])
    live = [e for e in range(n) if counts[e] >= args.min_rows]
    assert live, f"layer {layer}: no expert has {args.min_rows} rows"
    book, assign = search.grid, fast_assign(search)

    # Damping, on rows the fit does not see.
    both = [e for e in sorted(live, key=lambda e: -counts[e])
            if min(int(second_half[rows[e]].sum()), int((~second_half[rows[e]]).sum())) >= args.min_rows]
    sample = [both[i] for i in sorted(set(np.linspace(0, len(both) - 1, min(args.holdout, len(both))).astype(int).tolist()))] if both else []
    errors: dict[str, dict] = {space: {} for space in SPACES}
    for e in sample:
        b = second_half[rows[e]]
        fit_h, score_h = moments(x[rows[e][~b]], experts[e]), moments(x[rows[e][b]], experts[e])
        with fp32():
            for space, row in heldout(search, book, assign, load(layer, e), fit_h, score_h, args.damp).items():
                for k, v in row.items():
                    errors[space].setdefault(k, []).append(v)
    means = {space: {str(k): float(np.mean(v)) for k, v in row.items()} for space, row in errors.items()}
    damp: dict[str, float | None] = {}
    for space, row in means.items():
        best = min(args.damp, key=lambda d: row[str(d)]) if row else None
        damp[space] = best if best is not None and row[str(best)] < row["plain"] else None

    arrays: dict[str, dict[str, list]] = {"plain": {}, "feedback": {}}
    wrel: dict[str, list[float]] = {proj: [] for proj in PROJ_SPACE}
    diags: dict[str, torch.Tensor] = {}
    fitted: dict[int, dict] = {}
    for e in live:
        h = moments(x[rows[e]], experts[e])
        for space in SPACES:
            zeros = torch.zeros(n, h[space].shape[0], dtype=torch.float32, device=x.device)
            diags.setdefault(space, zeros)[e] = h[space].diagonal().float()
        with fp32():
            fitted[e] = fit_expert(search, book, assign, load(layer, e), {s: diags[s][e] for s in SPACES}, h, damp)
        del h
    mean = {space: d[live].mean(0) for space, d in diags.items()}
    for e in range(n):
        if e not in fitted:  # too few rows: the layer's mean diagonal, plain codes in both outputs
            for space in SPACES:
                diags[space][e] = mean[space]
            with fp32():
                fitted[e] = fit_expert(search, book, assign, load(layer, e), mean, None, damp)
        for proj, (codes, scales, error, fed) in fitted.pop(e).items():
            for kind, c in (("plain", codes), ("feedback", fed)):
                arrays[kind].setdefault(f"{proj}_codes", []).append(c)
                arrays[kind].setdefault(f"{proj}_scales", []).append(scales)
            wrel[proj].append(error)

    for kind, a in arrays.items():
        dest = out / kind / f"layer-{layer:02d}.npz"
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, code_bits=np.asarray(16), experts=np.arange(n), **{k: np.stack(v) for k, v in a.items()})
        tmp.rename(dest)
    np.savez(out / "imp" / f"layer-{layer:02d}.npz", rows=counts, **{space: d.cpu().numpy() for space, d in diags.items()})
    return {"layer": layer, "experts": n, "tokens": int(x.shape[0]), "seconds": round(time.time() - started, 1),
            "few_rows": n - len(live), "holdout_experts": len(sample), "heldout": means, "damp": damp,
            **{f"{proj}_wrel_mse": float(np.mean(v)) for proj, v in wrel.items()}}


def demo() -> None:
    torch.manual_seed(0)
    search = Search(torch.device("cpu"))
    book, assign = search.grid, fast_assign(search)
    v = torch.randn(4096, 8) * 0.8
    gap = (v - book[assign(v, book)]).pow(2).sum(1) - (v - book[ldlq.nearest(v, book)]).pow(2).sum(1)
    assert gap.abs().max() < 1e-5, f"the structured search is not the nearest point: {gap.abs().max()}"
    n, rows = 512, 24
    basis = torch.linalg.qr(torch.randn(n, n))[0]
    h = (basis * torch.arange(1, n + 1).float().pow(-1.0)) @ basis.T
    weights = {"w1": torch.randn(rows, n), "w3": torch.randn(rows, n), "w2": torch.randn(rows, n)}
    hs = {"hidden": h.double(), "down": h.double()}
    diag = {space: m.diagonal().float() for space, m in hs.items()}
    fit = fit_expert(search, book, assign, weights, diag, hs, {"hidden": 0.01, "down": None})
    for proj, (codes, scales, _, fed) in fit.items():
        plain, feedback = (ldlq.output_error(weights[proj], search.decode(c, scales), h) for c in (codes, fed))
        print(f"{proj}: plain {plain:.4f}, feedback {feedback:.4f}")
        assert (feedback < 0.6 * plain) if proj != "w2" else (fed is codes), (proj, plain, feedback)
    row = heldout(search, book, assign, weights, hs, hs, [0.01, 1.0])["hidden"]
    assert row[0.01] < row["plain"], row
    print("ok")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", required=True, type=Path)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--shard", default="0/1", help="I/N: this process fits the layers l % N == I")
    p.add_argument("--layers", help="these layers instead of a shard's (comma separated)")
    p.add_argument("--cal-tokens", type=int, default=300_000)
    p.add_argument("--record-tokens", type=int, default=4096, help="each calibration record is cut to its first tokens")
    p.add_argument("--damp", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--holdout", type=int, default=16, help="experts per layer whose rows choose the damping")
    p.add_argument("--min-rows", type=int, default=16)
    p.add_argument("--experts", type=int, default=0, help="debug: only the first N experts of a layer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--torch-kernels", action="store_true", help="on cuda, the torch stand-ins instead of the reference's kernels")
    args = p.parse_args(argv)

    import collect
    import stream
    import torch_kernel
    from transformers import AutoTokenizer

    dev = args.device
    ref_dir = args.hf_path / "inference"
    torch_kernel.exact_quant = True
    ref = stream.import_reference(ref_dir, kernel=torch_kernel if dev == "cpu" or args.torch_kernels else None)
    torch.set_default_dtype(torch.bfloat16)
    if dev != "cpu":
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.set_default_device(dev)
    torch.manual_seed(33377335)
    model_args = ref.ModelArgs(**json.loads((ref_dir / "config.json").read_text()))
    model_args.max_batch_size = 1
    records = calibration(args.pack, args.cal_tokens, args.record_tokens)
    model_args.max_seq_len = max(map(len, records))
    n_layers = int(model_args.n_layers)
    shard, n_shards = map(int, args.shard.split("/"))
    mine = [int(v) for v in args.layers.split(",")] if args.layers else [l for l in range(n_layers) if l % n_shards == shard]
    for kind in ("plain", "feedback", "imp"):
        (args.out / kind).mkdir(parents=True, exist_ok=True)
    todo = [l for l in mine if not all((args.out / k / f"layer-{l:02d}.npz").exists() for k in ("plain", "feedback", "imp"))]
    print(f"{len(records)} calibration records, {sum(map(len, records))} tokens; layers to fit: {todo}", flush=True)
    if not todo:
        return

    ckpt = stream.HFCheckpoint(args.hf_path)
    layout = ref.EngramLayout.from_args(model_args)
    with torch.device(dev):
        hash_state = ref.NgramHashState(model_args, layout, AutoTokenizer.from_pretrained(args.hf_path))
    engram_rows = {l: stream.EngramRows.from_checkpoint(ckpt, l, ref.fp8_block_size) for l in model_args.engram_layer_ids}
    streamer = stream.LayerStreamer(ref, model_args, ckpt.load, engram_rows, hash_state, dev)
    collect.install_hooks(streamer, top_k=2048, rows_per_slice=256)
    n_experts = int(model_args.n_routed_experts)
    for kind in ("plain", "feedback"):
        (args.out / kind / "experts.json").write_text(json.dumps(
            {"experts": [list(range(args.experts or n_experts))] * n_layers, "source": str(args.hf_path)}))
    with fp32():
        search = Search(torch.device(dev))
    items = [stream.Item(tokens=torch.as_tensor([ids], dtype=torch.long, device=dev), group=i) for i, ids in enumerate(records)]
    second_half = torch.cat([torch.full((len(ids),), i % 2 == 1, dtype=torch.bool, device=dev) for i, ids in enumerate(records)])

    def load(layer: int, expert: int) -> dict[str, torch.Tensor]:
        return load_expert(ckpt, layer, expert, dev)

    with torch.inference_mode():  # stream.LayerStreamer.forward, stopping after the last layer to fit
        skip = (".engram.embed.", *streamer.absent)
        pending = streamer._pool.submit(streamer.load_layer, "layers.0.", skip)
        streamer.start(items)
        for layer in range(max(todo) + 1):
            state = pending.result()
            if layer < max(todo):
                pending = streamer._pool.submit(streamer.load_layer, f"layers.{layer + 1}.", skip)
            block = streamer.build_block(layer, state)
            del state
            seen: list[torch.Tensor] = []
            if layer in todo:
                block.ffn.register_forward_pre_hook(lambda _, inputs, seen=seen: seen.append(inputs[0].reshape(-1, inputs[0].shape[-1])))
            streamer.run_layer(block, items, None)
            if layer in todo:
                row = fit_layer(layer, block, torch.cat(seen), second_half, load, search, args, args.out)
                if dev != "cpu":
                    row["max_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                with (args.out / f"metrics-{shard}.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            del block


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 1 else demo()
