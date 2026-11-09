"""Score router-masked variants of a local REAP checkpoint against the pod's teacher logits.

Every variant keeps a subset of the checkpoint's experts in each layer (any count per layer,
so non-uniform budgets work) by setting the dropped experts' gate bias to -inf, which is what
``collect.py --survivors`` did on the pod and equals physically slicing them out. All variants
go through one layer-streamed pass on CPU, through DeepSeek's reference code with the
``torch_kernel`` stand-ins.

Sessions are cut to their first ``--max-tokens`` tokens. The model is causal, so the logits at
the kept positions are the ones the pod computed on the whole record; the teacher's top-K rows
for those positions are compared directly.

    python local_eval.py --hf-path ~/Models/v41-reap50 \\
        --checkpoint-survivors ~/Models/v41-reap-run/stats/survivors-192.json \\
        --pack ~/Models/v41-reap-run/pack/v41-pack.jsonl --stats ~/Models/v41-reap-run/stats \\
        --sessions-file ~/Models/v41-reap-run/mac-eval-sessions.json --max-tokens 1024 \\
        --variant reap50=all --variant reap75=~/Models/v41-reap-run/stats/survivors-96.json \\
        --out ~/Models/v41-reap-run/evals/baseline

A variant spec is ``all`` (the checkpoint as is) or a survivors JSON in the base model's expert
ids (``{"survivors": {"<layer>": [ids...]}}``), each layer a subset of the checkpoint's.

``--expert-override`` swaps every routed expert's weights for bf16 ones before the pass:
``fp4-bf16`` decodes the checkpoint's FP4 weights exactly (the control), a directory of
``keep_quant.py`` layer files decodes KEEP's codes (E8P or 1-bit E8), and
``HOT_DIR@SET.json+REST_DIR`` takes the experts in a survivors set from one directory and the
rest from another (a per-expert precision tier). Overridden experts run as plain bf16
GEMMs without FP8 activation quantization, as KEEP's runtime does.

``--capture-imatrix OUT.npz`` measures KEEP's importance diagonal on this checkpoint instead of
scoring: one variant, any sessions (``name`` or ``name#record``, calibration records allowed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parents[2] / "src"))

from local_parity import rss_guard  # noqa: E402


def local_masks(spec: str, ckpt_survivors: dict[str, list[int]]) -> dict[int, np.ndarray] | None:
    """Per layer, a bool array over the checkpoint's local experts: True = dropped."""

    if spec == "all":
        return None
    wanted = json.loads(Path(spec).expanduser().read_text())["survivors"]
    masks = {}
    for layer, have in ckpt_survivors.items():
        have = sorted(have)
        keep = set(wanted[layer])
        assert keep <= set(have), f"layer {layer}: variant keeps experts the checkpoint dropped"
        masks[int(layer)] = np.asarray([e not in keep for e in have])
    return masks


def expert_override(spec: str, n_experts: int, ckpt_survivors: dict):
    """``layer -> {(expert, proj): bf16 [out, in]}`` from FP4 (``fp4-bf16``) or E8P codes."""

    import torch

    import torch_kernel

    if spec == "fp4-bf16":
        def from_state(layer: int, sd: dict) -> dict:
            out = {}
            for e in range(n_experts):
                for proj in ("w1", "w2", "w3"):
                    w = torch_kernel.decode_e2m1(sd[f"ffn.experts.{e}.{proj}.weight"])
                    s = sd[f"ffn.experts.{e}.{proj}.scale"].float()
                    out[(e, proj)] = (w.unflatten(-1, (-1, 32)) * s[..., None]).flatten(-2).bfloat16()
            return out
        return from_state

    from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

    grids = {16: torch.as_tensor(e8p_full_grid().astype(np.float32)), 8: torch.as_tensor(e8_1bit_grid().astype(np.float32))}

    def from_dir(root: Path):
        def from_codes(layer: int, sd: dict, only: set[int] | None = None) -> dict:
            out = {}
            with np.load(root / f"layer-{layer:02d}.npz") as z:
                grid = grids[int(z["code_bits"]) if "code_bits" in z.files else 16]
                for proj in ("w1", "w2", "w3"):
                    codes = torch.as_tensor(z[f"{proj}_codes"].astype(np.int64))  # [E, out, in/8]
                    scales = torch.as_tensor(z[f"{proj}_scales"].astype(np.float32))  # [E, out, in/g]
                    assert codes.shape[0] == n_experts, (layer, proj, codes.shape)
                    for e in range(n_experts):
                        if only is not None and e not in only:
                            continue
                        vec = grid[codes[e]]  # [out, in/8, 8]
                        words = vec.shape[1] // scales.shape[-1]
                        vec = vec * scales[e].repeat_interleave(words, dim=-1)[..., None]
                        out[(e, proj)] = vec.flatten(-2).bfloat16()
            return out
        return from_codes

    # "HOT_DIR@HOT_SET.json+REST_DIR": the hot set's experts from one source, the rest from another
    if "@" in spec:
        hot_spec, _, rest = spec.partition("@")
        hot_set_path, _, rest_spec = rest.partition("+")
        hot = from_dir(Path(hot_spec).expanduser())
        other = from_dir(Path(rest_spec).expanduser())
        hot_ids = json.loads(Path(hot_set_path).expanduser().read_text())["survivors"]

        def mixed(layer: int, sd: dict) -> dict:
            have = sorted(ckpt_survivors[str(layer)])
            is_hot = {i for i, e in enumerate(have) if e in set(hot_ids[str(layer)])}
            out = hot(layer, sd, is_hot)
            out.update(other(layer, sd, set(range(n_experts)) - is_hot))  # decode each expert once
            return out
        return mixed

    return from_dir(Path(spec).expanduser())


def residual_cache(ind: np.ndarray, wts: np.ndarray, k: int, tau: float, bounds: list[int],
                   warm: np.ndarray | None, counts: np.ndarray, policy: str = "lru",
                   trace: list | None = None) -> np.ndarray:
    """Which (token, slot) pairs may run base + residual, under an LRU of ``k`` resident residuals.

    ``ind``/``wts``: [T, top_k] routed experts and gate weights, the records of ``bounds`` back to
    back. A slot runs high precision only if its expert's residual is already resident. After each
    token, every slot with gate >= ``tau`` admits (or refreshes) its residual: an async SSD fetch
    that serves the next token on, so decode never stalls. Each record starts from ``warm`` (or
    empty). ``counts[e, 4]`` accumulates admissions, i.e. residual fetches; ``trace`` collects
    them as (record, token in record, expert).

    ``policy="oracle"`` is the upper bound for any cache of ``k``: each record's hot set is its own
    top ``k`` experts by routed gate weight, known in advance and never changed (no admissions)."""

    from collections import OrderedDict

    hi = np.zeros(ind.shape, dtype=bool)
    start = 0
    for rec, n in enumerate(bounds):
        if policy == "oracle":
            mass = np.bincount(ind[start:start + n].ravel(), weights=wts[start:start + n].ravel(), minlength=counts.shape[0])
            hot = np.argsort(-mass, kind="stable")[:k]
            hi[start:start + n] = np.isin(ind[start:start + n], hot[mass[hot] > 0])
            start += n
            continue
        lru = OrderedDict((int(e), None) for e in (warm if warm is not None else []))
        for t in range(start, start + n):
            row = ind[t].tolist()
            hi[t] = [e in lru for e in row]
            for s, e in enumerate(row):
                if wts[t, s] >= tau:
                    if e in lru:
                        lru.move_to_end(e)
                    else:
                        counts[e, 4] += 1
                        if trace is not None:
                            trace.append((rec, t - start, e))
                        lru[e] = None
                        if len(lru) > k:
                            lru.popitem(last=False)
        start += n
    return hi


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", required=True, type=Path)
    p.add_argument("--checkpoint-survivors", required=True, type=Path, help="the survivors JSON the checkpoint was exported from")
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--stats", required=True, type=Path)
    p.add_argument("--sessions-file", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--variant", action="append", required=True, metavar="NAME=SPEC")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-rss-gb", type=float, default=64.0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--expert-override", help="'fp4-bf16' or a keep_quant.py output directory")
    p.add_argument("--capture-imatrix", type=Path, help="measure the importance diagonal instead of scoring")
    p.add_argument("--dynamic", help="per-token tiers. BASE_DIR+RESIDUAL_DIR: base alone or base + nested residual. "
                   "BASE_DIR|HI_DIR: base alone or a separate high-precision copy (a hot expert then holds both)")
    p.add_argument("--hi-proj", action="append", default=[], metavar="NAME=w2|w1w3",
                   help="nested only, per variant: the high tier adds the residual for these projections only")
    p.add_argument("--cache-policy", action="append", default=[], metavar="NAME=lru|oracle",
                   help="with --cache-k: lru (default) or oracle (each record's own top-K, the upper bound)")
    p.add_argument("--trace", action="append", default=[], metavar="NAME",
                   help="with --cache-k: write this variant's residual admissions (layer, record, token, expert)")
    p.add_argument("--tau", action="append", default=[], metavar="NAME=VALUE",
                   help="with --dynamic, per variant: a slot runs base + residual when its gate weight >= VALUE")
    p.add_argument("--tau-drop", action="append", default=[], metavar="NAME=VALUE",
                   help="with --dynamic, per variant: a slot with gate weight < VALUE is skipped (rung 0 per token)")
    p.add_argument("--cache-k", action="append", default=[], metavar="NAME=K",
                   help="with --dynamic: hi tier only from an LRU of K resident residuals per layer, admitted at gate >= tau")
    p.add_argument("--cache-warm", type=Path, help="imatrix npz (reap__gate_norm_sum): warm-start each cache with the top-K")
    p.add_argument("--sidecar", type=Path, help="a sidecar_fit.py output directory, added to every variant's MoE output")
    p.add_argument("--sidecar-rank", type=int, help="use only the top-k sidecar components (stored largest first)")
    args = p.parse_args(argv)
    rss_guard(args.max_rss_gb)

    import torch
    from transformers import AutoTokenizer

    import collect
    import stream
    import torch_kernel

    prune = json.loads((args.hf_path / "keep-prune.json").read_text())
    sha = hashlib.sha256(args.checkpoint_survivors.read_bytes()).hexdigest()
    assert sha == prune["survivors_sha256"], "--checkpoint-survivors is not what this checkpoint was exported from"
    ckpt_survivors = json.loads(args.checkpoint_survivors.read_text())["survivors"]
    variants = {}
    for spec in args.variant:
        name, _, path = spec.partition("=")
        masks = local_masks(path, ckpt_survivors)
        variants[name] = None if masks is None else {l: torch.as_tensor(m) for l, m in masks.items()}

    torch_kernel.exact_quant = True
    torch.set_num_threads(args.threads)
    ref_dir = args.hf_path / "inference"
    ref = stream.import_reference(ref_dir, kernel=torch_kernel)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    model_args = ref.ModelArgs(**json.loads((ref_dir / "config.json").read_text()))
    model_args.max_batch_size = 1

    sessions = collect.load_sessions(args.pack)
    names = json.loads(args.sessions_file.read_text())
    records = {}
    for name in names:
        session, _, idx = name.partition("#")
        recs = sessions[session]
        if not idx:
            assert len(recs) == 1, f"{session} has {len(recs)} records; name one as {session}#N"
        record = recs[int(idx or 0)]
        record = dict(record, token_ids=record["token_ids"][: args.max_tokens])
        records[name] = record
    model_args.max_seq_len = max(len(r["token_ids"]) for r in records.values())

    ckpt = stream.HFCheckpoint(args.hf_path)
    layout = ref.EngramLayout.from_args(model_args)
    hash_state = ref.NgramHashState(model_args, layout, AutoTokenizer.from_pretrained(args.hf_path))
    engram_rows = {l: stream.EngramRows.from_checkpoint(ckpt, l, ref.fp8_block_size) for l in model_args.engram_layer_ids}
    streamer = stream.LayerStreamer(ref, model_args, ckpt.load, engram_rows, hash_state, "cpu")
    state = collect.install_hooks(streamer, top_k=2048, rows_per_slice=256)
    n_local = len(next(iter(ckpt_survivors.values())))

    dyn_counts: dict[str, np.ndarray] = {}
    if args.dynamic:  # per-token precision: each (token, expert) slot picks its tier from its gate weight
        import torch.nn.functional as F

        from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

        copy_mode = "|" in args.dynamic  # the high tier is its own copy, not base + residual
        base_dir, _, res_dir = args.dynamic.partition("|" if copy_mode else "+")
        base_dir, res_dir = Path(base_dir).expanduser(), Path(res_dir).expanduser()
        hi_proj = {name: {"w2": ("w2",), "w1w3": ("w1", "w3")}[v] for name, _, v in (s.partition("=") for s in args.hi_proj)}
        assert not (copy_mode and hi_proj), "--hi-proj needs nested codes"
        policies = {name: v for name, _, v in (s.partition("=") for s in args.cache_policy)}
        traces: dict[str, list] = {name: [] for name in args.trace}
        taus = {name: float(t) for name, _, t in (s.partition("=") for s in args.tau)}
        tau_drop = {name: float(t) for name, _, t in (s.partition("=") for s in args.tau_drop)}
        assert set(taus) == set(variants), "--tau needs one NAME=VALUE per --variant"
        np_grids = {16: e8p_full_grid().astype(np.float32), 8: e8_1bit_grid().astype(np.float32)}
        cache: dict = {"layer": None}
        n_layers_ = int(model_args.n_layers)
        cache_k = {name: int(k) for name, _, k in (s.partition("=") for s in args.cache_k)}
        warm = None
        if args.cache_warm:  # warm start: each layer's top-K experts by routed mass (the static hot set)
            warm = np.argsort(-np.load(args.cache_warm)["reap__gate_norm_sum"], axis=1, kind="stable")
        for v in variants:  # [layer, expert, (lo slots, hi slots, lo mass, hi mass, admissions, dropped slots, dropped mass)]
            dyn_counts[v] = np.zeros((n_layers_, n_local, 7))

        def simulate_cache(layer, ind, wts, k, tau, counts):
            v, found = state["variant"], []
            hi = residual_cache(ind, wts, k, tau, state["bounds"], None if warm is None else warm[layer, :k], counts,
                                policy=policies.get(v, "lru"), trace=found if v in traces else None)
            if v in traces:
                traces[v].extend((layer, *row) for row in found)
            return hi

        def tier_weights(layer: int, e: int) -> tuple[dict, dict]:
            """(base, base + residual) bf16 weights of local expert ``e``, decoded once per layer."""
            if cache["layer"] != layer:
                cache.clear()
                cache["layer"] = layer
                cache["z"] = [dict(np.load(d / f"layer-{layer:02d}.npz")) for d in (base_dir, res_dir)]
            if e not in cache:
                tiers = []
                for z in cache["z"]:
                    grid = np_grids[int(z.get("code_bits", 16))]  # the first E8P runs wrote no code_bits
                    tier = {}
                    for proj in ("w1", "w2", "w3"):
                        vec = grid[z[f"{proj}_codes"][e].astype(np.int64)]
                        s = z[f"{proj}_scales"][e].astype(np.float32)
                        vec = vec * np.repeat(s, vec.shape[1] // s.shape[-1], axis=-1)[..., None]
                        tier[proj] = vec.reshape(vec.shape[0], -1)
                    tiers.append(tier)
                lo = {p: torch.as_tensor(w).bfloat16() for p, w in tiers[0].items()}
                hi = {p: torch.as_tensor(tiers[1][p] if copy_mode else tiers[0][p] + tiers[1][p]).bfloat16() for p in tiers[0]}
                cache[e] = (lo, hi)
            return cache[e]

        def run_expert(x, w, tier, limit):
            gate, up = F.linear(x, tier["w1"]).float(), F.linear(x, tier["w3"]).float()
            if limit > 0:
                up, gate = up.clamp(-limit, limit), gate.clamp(max=limit)
            return F.linear((w * (F.silu(gate) * up)).to(x.dtype), tier["w2"])

        def dynamic_forward(self, x, image_mask=None):
            shape, layer = x.size(), self.layer_id
            x = x.view(-1, self.dim)
            weights, indices = self.gate(x, None)
            tau, counts = taus[state["variant"]], dyn_counts[state["variant"]][layer]
            y = torch.zeros_like(x, dtype=torch.float32)
            limit = self.shared_experts.swiglu_limit
            k = cache_k.get(state["variant"])
            hi_all = None if k is None else torch.as_tensor(
                simulate_cache(layer, indices.numpy(), weights.float().numpy(), k, tau, counts))
            for i in torch.unique(indices).tolist():
                idx, top = torch.where(indices == i)
                w = weights[idx, top]
                hi_mask = w >= tau if hi_all is None else hi_all[idx, top]
                drop = w < tau_drop.get(state["variant"], 0.0)  # rung 0 per token: the slot is skipped
                counts[i, 5] += int(drop.sum())
                counts[i, 6] += float(w[drop].sum())
                hi_mask = hi_mask & ~drop
                lo_w, hi_w = tier_weights(layer, i)
                only = hi_proj.get(state["variant"])
                if only is not None:  # a finer rung: the residual of some projections only
                    hi_w = {p: hi_w[p] if p in only else lo_w[p] for p in lo_w}
                for mask, tier, col in ((~hi_mask & ~drop, lo_w, 0), (hi_mask, hi_w, 1)):
                    if mask.any():
                        y[idx[mask]] += run_expert(x[idx[mask]], w[mask, None], tier, limit).float()
                        counts[i, col] += int(mask.sum())
                        counts[i, col + 2] += float(w[mask].sum())
            y += self.shared_experts(x).float()
            return y.type_as(x).view(shape)

        ref.MoE.forward = dynamic_forward

    if args.sidecar:  # sidecar_fit.py output: y += ([x, 1] @ U) @ Vt on every MoE call
        sidecars = {}
        for layer in range(int(model_args.n_layers)):
            if not (args.sidecar / f"layer-{layer:02d}.npz").exists():
                continue  # a fit still in progress: the layers it has are valid without the rest
            with np.load(args.sidecar / f"layer-{layer:02d}.npz") as z:
                U, Vt = z["U"][:, : args.sidecar_rank or None], z["Vt"][: args.sidecar_rank or None]
                sidecars[layer] = (torch.as_tensor(U), torch.as_tensor(Vt))
        moe_forward = ref.MoE.forward

        def with_sidecar(self, x, image_mask=None):
            y = moe_forward(self, x, image_mask)
            if self.layer_id not in sidecars:
                return y
            U, Vt = sidecars[self.layer_id]
            xf = x.reshape(-1, x.size(-1)).float()
            corr = (xf @ U[:-1] + U[-1]) @ Vt
            return (y.reshape(-1, y.size(-1)).float() + corr).type_as(y).view(y.shape)

        ref.MoE.forward = with_sidecar

    if args.expert_override:
        decode = expert_override(args.expert_override, n_local, ckpt_survivors)
        decoded: dict[int, dict] = {}
        load_layer, build_block = streamer.load_layer, streamer.build_block

        def load_with_override(prefix, skip=()):  # runs in the prefetch thread
            sd = load_layer(prefix, skip)
            if prefix.startswith("layers."):
                layer = int(prefix.split(".")[1])
                decoded[layer] = decode(layer, sd)
            return sd

        def build_with_override(layer, sd):
            block = build_block(layer, sd)
            for (e, proj), w in decoded.pop(layer).items():
                lin = getattr(block.ffn.experts[e], proj)
                lin.weight = torch.nn.Parameter(w, requires_grad=False)
                if hasattr(lin, "scale"):
                    del lin.scale
            return block

        streamer.load_layer, streamer.build_block = load_with_override, build_with_override

    capture = None
    if args.capture_imatrix:
        assert len(variants) == 1, "--capture-imatrix takes one variant"
        capture = collect.SessionStats(torch=torch, n_layers=int(model_args.n_layers), n_experts=n_local,
                                       dims={"hidden": int(model_args.dim), "down": int(model_args.moe_inter_dim)}, device="cpu")

    state["bounds"] = [len(r["token_ids"]) for r in records.values()]  # one MoE call holds a variant's records in order
    items = {
        (v, n): stream.Item(tokens=torch.as_tensor([r["token_ids"]], dtype=torch.long), group=(v, None))
        for v in variants
        for n, r in records.items()
    }

    def before_moe(layer: int, block, key) -> None:
        bias = block.ffn.gate.bias
        if not hasattr(block, "full_bias"):
            block.full_bias = bias.data.clone()
        bias.data.copy_(block.full_bias)
        masks = variants[key[0]]
        if masks is not None:
            bias.data[masks[layer]] = float("-inf")
        state["stats"] = capture
        state["variant"] = key[0]

    t0 = time.time()
    with torch.inference_mode():
        streamer.forward(list(items.values()), before_moe)
    forward_s = time.time() - t0
    state["stats"] = None

    if capture is not None:
        arrays = capture.arrays(list(range(int(model_args.n_layers))))
        args.capture_imatrix.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.capture_imatrix,
                 importance__hidden=arrays["importance_sum__hidden"], importance__down=arrays["importance_sum__down"],
                 route_count=arrays["route_count__hidden"], reap__gate_norm_sum=arrays["reap__gate_norm_sum"],
                 reap__route_count=arrays["reap__route_count"],
                 sessions=np.asarray(names), tokens=np.asarray(sum(len(r["token_ids"]) for r in records.values())))
        print(json.dumps({"forward_seconds": round(forward_s, 1), "tokens": int(sum(len(r["token_ids"]) for r in records.values())),
                          "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)}), flush=True)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"forward_seconds": round(forward_s, 1), "max_tokens": args.max_tokens, "variants": {}}
    for name, record in records.items():
        teacher = np.load(args.stats / "eval" / "sessions" / f"{name}.npz")
        positions = collect.target_positions(record)
        rows = slice(0, positions.size)
        assert np.array_equal(positions, teacher["positions"][rows]), name
        targets = np.asarray(record["token_ids"], dtype=np.int64)[positions + 1]
        teacher_nll = (teacher["logsumexp"][rows] - teacher["target_logit"][rows]).astype(np.float64)
        state["positions"] = torch.as_tensor(positions)
        state["targets"] = torch.as_tensor(targets)
        state["teacher"] = {
            "ids": torch.as_tensor(teacher["topk_logit_ids"][rows].astype(np.int64)),
            "logprob": torch.as_tensor(teacher["topk_logit_values"][rows].astype(np.float32) - teacher["logsumexp"][rows, None]),
            "targets": state["targets"],
        }
        for v in variants:
            state["captured"] = {k: [] for k in collect.CAPTURE_KEYS}
            with torch.inference_mode():
                streamer.head(streamer.final_hidden(items[(v, name)]))
            cap = {k: np.concatenate(x) for k, x in state["captured"].items() if x}
            out = {
                "positions": positions.astype(np.int32),
                "kl": cap["kl_on_teacher_topk"],
                "nll_delta": (cap["student_nll"] - teacher_nll).astype(np.float32),
                "top1": cap["top1_agree"],
            }
            (args.out / v).mkdir(exist_ok=True)
            np.savez(args.out / v / f"{name}.npz", **out)
            s = summary["variants"].setdefault(v, {})
            s[name] = {"domain": record["domain"], "positions": int(positions.size),
                       **{k: float(np.mean(out[k])) for k in ("kl", "nll_delta", "top1")}}
        state["teacher"] = state["positions"] = state["targets"] = None

    for v, per in summary["variants"].items():
        doms: dict[str, list] = {}
        for row in per.values():
            doms.setdefault(row["domain"], []).append(row)
        by_dom = {d: {k: float(np.average([r[k] for r in rs], weights=[r["positions"] for r in rs])) for k in ("kl", "nll_delta", "top1")}
                  for d, rs in sorted(doms.items())}
        per["_domains"] = by_dom
        per["_domain_mean"] = {k: float(np.mean([x[k] for x in by_dom.values()])) for k in ("kl", "nll_delta", "top1")}
        print(v, json.dumps(per["_domain_mean"]), flush=True)
    for v, c in dyn_counts.items():  # per-token precision: how often (and with how much gate mass) the hi tier ran
        np.save(args.out / v / "tier-counts.npy", c)
        tot = c.sum(axis=(0, 1))
        if v in traces:
            t = np.asarray(traces[v], dtype=np.int32).reshape(-1, 4)
            np.savez(args.out / v / "admissions.npz", layer=t[:, 0], record=t[:, 1], token=t[:, 2], expert=t[:, 3],
                     records=np.asarray(list(records)), tokens=np.asarray(state["bounds"]))
        kept = n_local if variants[v] is None else float(np.mean([int((~m).sum()) for m in variants[v].values()]))
        # hi_units: memory (and bytes fetched) the high tier adds per expert, in units of one 1-bit expert
        hi_units = 2.0 if copy_mode else len(hi_proj.get(v, ("w1", "w2", "w3"))) / 3
        summary["variants"][v]["_tiers"] = {"tau": taus[v], "cache_k": cache_k.get(v),
                                            "policy": policies.get(v, "lru") if v in cache_k else None,
                                            "copy_mode": copy_mode, "hi_units": hi_units, "n_bases": kept,
                                            "tau_drop": tau_drop.get(v, 0.0),
                                            "hi_slot_share": float(tot[1] / (tot[:2].sum() + tot[5])),
                                            "drop_slot_share": float(tot[5] / (tot[:2].sum() + tot[5])),
                                            "hi_mass_share": float(tot[3] / (tot[2:4].sum() + tot[6])),
                                            "drop_mass_share": float(tot[6] / (tot[2:4].sum() + tot[6])),
                                            "admissions_per_token_layer": float(tot[4] / ((tot[:2].sum() + tot[5]) / int(model_args.n_activated_experts)) / int(model_args.n_layers))}
        print(v, json.dumps(summary["variants"][v]["_tiers"]), flush=True)
    summary["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("forward_seconds", "peak_rss_gb")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
