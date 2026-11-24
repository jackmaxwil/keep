#!/usr/bin/env python3
"""Routed residency against static resident sets on GLM-5.3-Flash, at the model's own precision.

The second model of the paper's case study. No codes are fitted: every expert runs from the
checkpoint's FP8 weights, so the comparison isolates residency (which experts are in memory)
from precision. The reference model (transformers v5.16.1, ``glm5_next``) runs one decoder layer
at a time on one GPU: a layer is built, loaded from the shards, applied to every stream, and
freed. A stream is one (variant, session) pair and carries the layer's output to the next layer.

Per layer, calibration sessions run first through the full model and give each expert's routed
gate mass; the K heaviest are a variant's starting residents. Evaluation sessions then run
through the full model (the teacher) and through each variant, in which the router's choices are
re-routed among resident experts and the cache policy of ``v41_reap/local_eval.py``
(``residual_cache``) admits and evicts. A static set is the same policy with a threshold nothing
reaches. The threshold is on the model's routing weights, which sum to its routing scale (2.5 here,
1.5 in the first model); a policy that starts with ``share`` reads it as a share of the token's weight
instead, which is how the runs of 2026-11-20 applied it. Scores are the KL from the teacher's next-token distribution, per session.

Coded variants (``NAME=K:TAU[:POLICY]:codes[:hot=H[:fp8]]``) run every expert from its 2-bit codes
(``--codes``, ``fit_codes.py``) instead of FP8, except a hot set: the H experts of largest calibration
mass, at base + 1-bit residual (``--residual``; among the experts that have one) or with ``fp8`` at
the checkpoint's own weights. The teacher stays at FP8. ``--importance`` (calibration only, no
``--eval`` needed) saves the per-expert importance ``fit_codes.py`` reads: squared inputs and squared
GLU outputs of the rows the full model routes to each expert.

    python residency.py --ckpt DIR --cal cal.json --eval eval.json --out OUT \\
        --variant s108=108:1e9 --variant k108=108:0.3
    python residency.py --ckpt DIR --cal cal.json --importance imp-glm.npz --out OUT
    python residency.py --ckpt DIR --cal cal.json --eval eval.json --out OUT --codes codes2 --residual resid1 \\
        --variant c108=108:0.3:codes --variant c76h64=76:0.3:codes:hot=64
    python residency.py --selftest

Session files map a name to ``{"domain": ..., "text": ...}``; text is tokenized with the
checkpoint's ``tokenizer.json`` and cut to ``--max-tokens``.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT.parent / "v41_reap"))

from keep_quant_torch import Search, fill_dead
from local_eval import hot_set, residual_cache

PREFIX = "model.language_model."
EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
STATE: dict = {"variant": None, "mass": None, "layer": 0, "warm": None, "counts": {}, "imp": None, "coded": None}


class Checkpoint:
    """Tensors of a safetensors checkpoint by name, decoded to floats on a device."""

    def __init__(self, root: Path):
        self.where: dict[str, tuple] = {}
        for shard in sorted(root.glob("*.safetensors")):
            with shard.open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            for name, info in header.items():
                if name != "__metadata__":
                    start, end = info["data_offsets"]
                    self.where[name] = (shard, 8 + size + start, end - start, info["dtype"], info["shape"])

    def read(self, name: str, device) -> torch.Tensor:
        shard, offset, size, dtype, shape = self.where[name]
        raw = torch.from_numpy(np.fromfile(shard, dtype=np.uint8, count=size, offset=offset))
        if dtype == "F8_E4M3":  # FP8 codes times a float32 multiplier per 128 x 128 block
            scale = self.read(name + "_scale_inv", device).repeat_interleave(128, 0).repeat_interleave(128, 1)
            codes = raw.view(torch.float8_e4m3fn).reshape(shape)
            try:
                value = codes.to(device).float()
            except (RuntimeError, NotImplementedError):  # a backend without FP8 casts
                value = codes.float().to(device)
            return value * scale[: shape[0], : shape[1]]
        return raw.view({"BF16": torch.bfloat16, "F32": torch.float32}[dtype]).reshape(shape).to(device)

    def layer_state(self, layer: int, device, dtype) -> dict[str, torch.Tensor]:
        """One decoder layer under the reference module's names (its conversion mapping)."""
        stem = f"{PREFIX}layers.{layer}."
        state, qkv, experts = {}, {}, {}
        for name in [n for n in self.where if n.startswith(stem) and not n.endswith("weight_scale_inv")]:
            value, key = self.read(name, device).to(dtype), name.removeprefix(stem)
            if m := EXPERT.match(key):
                experts[int(m[1]), m[2]] = value
                continue
            key = re.sub(r"self_attn\.(f_a_proj|f_b_proj|dt_bias|A_log)", r"self_attn.forget_gate.\1", key)
            key = re.sub(r"hc_(attn|ffn)_(fn|base|scale)", r"\1_hc.\2", key)
            if m := re.match(r"self_attn\.([qkv])_conv1d\.weight$", key):
                qkv[m[1]] = value
                continue
            state[key] = value
        if qkv:
            state["self_attn.conv1d.weight"] = torch.cat([qkv[p] for p in "qkv"])
        if experts:
            count = 1 + max(e for e, _ in experts)
            state["mlp.experts.gate_up_proj"] = torch.stack(
                [torch.cat([experts.pop((e, "gate")), experts.pop((e, "up"))]) for e in range(count)])
            state["mlp.experts.down_proj"] = torch.stack([experts.pop((e, "down")) for e in range(count)])
        return state


class Codes:
    """A layer's experts from ``fit_codes.py`` files: every expert at 2 bits, and base + 1-bit residual
    for the experts the residual run covers, as the reference module's bf16 stacks."""

    def __init__(self, base: Path, residual: Path | None, device, dtype):
        self.base, self.residual, self.dtype, self.search = base, residual, dtype, Search(device)

    def residual_experts(self, layer: int) -> list[int]:
        if self.residual is None:
            return []
        with np.load(self.residual / f"layer-{layer:02d}.npz") as z:
            return z["experts"].tolist()

    def _decode(self, z: dict, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = {p: self.search.decode(z[f"{p}_codes"][slot], z[f"{p}_scales"][slot], int(z["code_bits"])) for p in ("w1", "w3", "w2")}
        return torch.cat([w["w1"], w["w3"]]), w["w2"]  # gate_up as layer_state stacks it

    def layer(self, layer: int, hot: set[int]) -> dict:
        """``{"base": (gate_up [E, 2I, H], down [E, H, I]), "hot": {expert: (gate_up, down)}}``."""
        name = f"layer-{layer:02d}.npz"
        z = dict(np.load(self.base / name))  # each key read once
        assert z["experts"].tolist() == list(range(len(z["experts"]))), "2-bit codes must cover every expert in order"
        stacks = None
        for e in range(len(z["experts"])):
            gate_up, down = self._decode(z, e)
            if stacks is None:
                stacks = [torch.empty((len(z["experts"]), *t.shape), dtype=self.dtype, device=t.device) for t in (gate_up, down)]
            stacks[0][e], stacks[1][e] = gate_up, down
        out = {"base": tuple(stacks), "hot": {}}
        if hot:
            r = dict(np.load(self.residual / name))
            slot = {int(e): s for s, e in enumerate(r["experts"])}
            for e in sorted(hot):  # the residual was fitted against the base's float decode
                (gate_up, down), (res_gate_up, res_down) = self._decode(z, e), self._decode(r, slot[e])
                out["hot"][e] = ((gate_up + res_gate_up).to(self.dtype), (down + res_down).to(self.dtype))
        return out


def experts_forward(experts, x, index, weight, weights_of) -> torch.Tensor:
    """The reference experts' eager loop with each expert's ``(gate_up, down)`` from ``weights_of(e)``."""
    final = torch.zeros_like(x)
    for e in index.unique().tolist():
        pos, tok = torch.where(index.T == e)
        gate_up, down = weights_of(e)
        h = experts._apply_gate(F.linear(x[tok], gate_up))
        final.index_add_(0, tok, (F.linear(h, down) * weight[tok, pos, None]).to(final.dtype))
    return final


def record_importance(experts, x, index, imp: dict) -> None:
    """Add the squared inputs and squared GLU outputs of each expert's routed rows to ``imp[space][e]``."""
    for e in index.unique().tolist():
        rows = x[(index == e).any(-1)]
        imp["hidden"][e] += rows.double().square().sum(0)
        imp["down"][e] += experts._apply_gate(F.linear(rows, experts.gate_up_proj[e])).double().square().sum(0)


_full_moe = M.Glm5NextTextMoE.forward


def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """The reference MoE with demand recorded and, for a variant, re-routed among resident experts."""
    spec, layer = STATE["variant"], STATE["layer"]
    if spec is None and STATE["mass"] is None:
        return _full_moe(self, hidden_states)
    gate = self.gate
    assert gate.num_group == 1 and gate.norm_topk_prob
    x = hidden_states.view(-1, hidden_states.shape[-1])
    scores = F.linear(x.float(), gate.weight.float()).sigmoid()
    choice = scores + gate.e_score_correction_bias

    def pick(allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index = allowed.topk(gate.top_k, dim=-1).indices
        weight = scores.gather(1, index)
        return index, weight / (weight.sum(-1, keepdim=True) + 1e-20)

    index, weight = pick(choice)  # the demand: what the full model would run
    if STATE["mass"] is not None:
        STATE["mass"][layer] += torch.bincount(index.reshape(-1), weight.reshape(-1).double(), minlength=gate.num_experts).cpu().numpy()
    if spec is not None:
        k, tau, policy, _ = spec
        resident = np.zeros((x.shape[0], gate.num_experts), dtype=bool)
        counts = STATE["counts"].setdefault(STATE["name"], np.zeros((gate.num_experts, 5)))
        # tau applies to the routing weights the model uses (they sum to routed_scaling_factor), as in the
        # serving code and the first model's simulator; ``share`` variants give it as a share of a token's weight.
        scale = 1.0 if policy.startswith("share") else gate.routed_scaling_factor
        policy = policy.removeprefix("share").lstrip("+") or "lru"
        residual_cache(index.cpu().numpy(), (weight.double() * scale).cpu().numpy(), k, tau, [x.shape[0]], STATE["warm"][layer][:k],
                       counts, policy=policy, resident=resident)
        resident = torch.as_tensor(resident, device=x.device)
        STATE["served"][STATE["name"]] += np.array([float((weight * resident.gather(1, index)).sum()), float(weight.sum())])
        index, weight = pick(choice.masked_fill(~resident, float("-inf")))
    native = self.experts
    if spec is not None and spec[3] is not None:  # coded: 2-bit unless hot
        hot, fp8 = STATE["hot"][STATE["name"]]
        base, better = STATE["coded"]["base"], STATE["coded"]["hot"]

        def weights_of(e):
            if e in hot:
                return (native.gate_up_proj[e], native.down_proj[e]) if fp8 else better[e]
            return base[0][e], base[1][e]

        routed = experts_forward(native, x, index, weight * gate.routed_scaling_factor, weights_of)
    else:
        if spec is None and STATE["imp"] is not None:  # the calibration group
            record_importance(native, x, index, {space: a[layer] for space, a in STATE["imp"].items()})
        routed = native(x, index, weight * gate.routed_scaling_factor)
    return routed.view(hidden_states.shape) + self.shared_experts(hidden_states)


def stream_forward(config, embed, layers, streams: list[dict], calibration: list[dict], log=print, prepare=None) -> None:
    """Run every stream through every layer. ``layers(i)`` returns layer ``i`` as a module; each stream is
    ``{"ids", "variant", "name"}`` and gains ``"h"``, the hidden streams leaving the last layer.
    ``prepare(i, layer)`` runs between the calibration group and the other streams."""
    for stream in (*calibration, *streams):
        emb = embed(stream["ids"])
        mask = M.create_recurrent_attention_mask(config=config, inputs_embeds=emb, attention_mask=None, past_key_values=None)
        stream["mask"] = (torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device) if mask is None else mask).bool()
        stream["pos"] = torch.arange(emb.shape[1], device=emb.device).unsqueeze(0)
        stream["h"] = emb.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
        stream["topk"] = None
    for i in range(config.num_hidden_layers):
        started = time.time()
        layer, STATE["layer"] = layers(i), i
        loaded = time.time() - started
        with torch.inference_mode():
            for group, mass in ((calibration, STATE["mass"]), (streams, None)):
                if group is streams and prepare is not None:
                    prepare(i, layer)
                saved, STATE["mass"] = STATE["mass"], mass
                for stream in group:
                    STATE["variant"], STATE["name"] = stream.get("variant"), stream.get("name")
                    stream["h"], stream["topk"] = layer(
                        stream["h"], attention_mask=stream["mask"], position_ids=stream["pos"], position_embeddings=None,
                        input_ids=stream["ids"], past_key_values=None, prev_topk_indices=stream["topk"])
                STATE["mass"] = saved
                if group is calibration and STATE["mass"] is not None and isinstance(layer.mlp, M.Glm5NextTextMoE):
                    STATE["warm"][i] = np.argsort(-STATE["mass"][i], kind="stable")
        del layer
        STATE["coded"] = None
        log(f"layer {i}: load {loaded:.1f} s, run {time.time() - started - loaded:.1f} s")


def score(config, norm, head, streams: list[dict]) -> dict:
    """KL from the teacher (the stream with no variant) to each variant, per session."""
    by_session: dict[str, dict] = {}
    for stream in streams:
        by_session.setdefault(stream["session"], {})[stream.get("name") or "_teacher"] = stream
    out: dict = {}
    with torch.inference_mode():
        for session, group in by_session.items():
            def logp(stream):
                return F.log_softmax(F.linear(norm(stream["h"].mean(dim=2)).float(), head.float())[0], dim=-1)

            teacher, ids = logp(group["_teacher"]), group["_teacher"]["ids"][0]
            nll = -teacher[:-1].gather(1, ids[1:, None]).mean()
            out.setdefault("_teacher", {})[session] = {"nll": float(nll), "tokens": int(ids.numel()),
                                                       "domain": group["_teacher"]["domain"]}
            for name, stream in group.items():
                if name == "_teacher":
                    continue
                student = logp(stream)
                out.setdefault(name, {})[session] = {
                    "kl": float((teacher.exp() * (teacher - student)).sum(-1).mean()),
                    "nll_delta": float(-student[:-1].gather(1, ids[1:, None]).mean() - nll),
                    "top1": float((teacher.argmax(-1) == student.argmax(-1)).float().mean()),
                    "domain": stream["domain"]}
    return out


def parse_variant(spec: str) -> tuple[str, tuple]:
    """``NAME=K:TAU[:POLICY][:codes[:hot=H[:fp8]]]`` -> (name, (K, TAU, POLICY, None or (H, fp8)))."""
    name, _, rest = spec.partition("=")
    rest, codes, options = rest.partition(":codes")
    k, tau, *policy = rest.split(":", 2)
    coded = None
    if codes:
        options = options.removeprefix(":").split(":") if options else []
        assert all(o == "fp8" or o.startswith("hot=") for o in options), spec
        coded = (next((int(o[4:]) for o in options if o.startswith("hot=")), 0), "fp8" in options)
    return name, (int(k), float(tau), policy[0] if policy else "lru", coded)


def make_streams(sessions: dict, variants: dict, device) -> list[dict]:
    return [{"session": s, "domain": d["domain"], "ids": torch.as_tensor([d["ids"]], device=device), "variant": spec, "name": name}
            for s, d in sessions.items() for name, spec in ((None, None), *variants.items())]


def run(config, embed, layers, norm, head, cal: dict, evals: dict, variants: dict, device, log=print,
        codes: Codes | None = None, importance: bool = False) -> dict:
    """With ``importance``, ``STATE["imp"]`` holds the calibration group's importance afterwards."""
    n = config.n_routed_experts
    STATE.update(mass=np.zeros((config.num_hidden_layers, n)), warm={}, counts={}, served={v: np.zeros(2) for v in variants},
                 imp=None, coded=None, hot={})
    if importance:
        STATE["imp"] = {space: torch.zeros((config.num_hidden_layers, n, dim), dtype=torch.float64, device=device)
                        for space, dim in (("hidden", config.hidden_size), ("down", config.moe_intermediate_size))}
    calibration = [{"ids": torch.as_tensor([d["ids"]], device=device)} for d in cal.values()]
    streams = make_streams(evals, variants, device)
    coded = {name: spec[3] for name, spec in variants.items() if spec[3] is not None}
    assert codes is not None or not coded, "coded variants need --codes"

    def prepare(i: int, layer) -> None:
        if not coded or not isinstance(layer.mlp, M.Glm5NextTextMoE):
            return
        have = codes.residual_experts(i)
        STATE["hot"] = {name: (hot_set(STATE["warm"][i], h, range(n) if fp8 else have), fp8) for name, (h, fp8) in coded.items()}
        assert all(len(hot) == coded[name][0] for name, (hot, _) in STATE["hot"].items()), "too few experts with residuals"
        STATE["coded"] = codes.layer(i, set().union(*(hot for hot, fp8 in STATE["hot"].values() if not fp8)))

    M.Glm5NextTextMoE.forward = moe_forward
    try:
        stream_forward(config, embed, layers, streams, calibration, log, prepare)
    finally:
        M.Glm5NextTextMoE.forward = _full_moe
    summary = score(config, norm, head, streams)
    tokens = sum(len(d["ids"]) for d in evals.values())
    moe_layers = sum(t == "sparse" for t in config.mlp_layer_types[: config.num_hidden_layers])
    for name, (k, tau, policy, coded_spec) in variants.items():
        summary[name]["_policy"] = {"residents": k, "tau": tau, "policy": policy,
                                    "codes": None if coded_spec is None else {"hot": coded_spec[0], "hot_fp8": coded_spec[1]},
                                    "gate_weight_resident": float(STATE["served"][name][0] / STATE["served"][name][1]),
                                    "admissions_per_token_layer": float(STATE["counts"][name][:, 4].sum() / tokens / moe_layers)}
    summary["_mass"] = STATE["mass"].tolist()
    return summary


def save_importance(path: Path, config) -> int:
    """``STATE["imp"]`` of the routing layers as ``fit_codes.py`` reads it; returns how many (expert, space)
    importances were dead and replaced by their layer's mean (``keep_quant_torch.fill_dead``)."""
    rows = [i for i, t in enumerate(config.mlp_layer_types[: config.num_hidden_layers]) if t == "sparse"]
    n = config.n_routed_experts
    out, replaced = {"layers": np.asarray(rows), "mass": STATE["mass"][rows], "experts": np.tile(np.arange(n), (len(rows), 1))}, 0
    for space, a in STATE["imp"].items():
        total = a[rows].cpu().numpy()
        picked = total.copy()
        replaced += fill_dead(picked, total)
        out[space] = picked.astype(np.float32)
    np.savez(path, **out)
    return replaced


def selftest() -> None:
    sys.path.insert(0, str(KIT.parents[1]))
    from make_glm53_reference_fixtures import TINY, randomize

    torch.manual_seed(0)
    config = Glm5NextTextConfig(**TINY)
    config._attn_implementation = "eager"
    model = M.Glm5NextTextModel(config).float().eval()
    randomize(model, torch.Generator().manual_seed(1))
    head = torch.randn(config.vocab_size, config.hidden_size)
    ids = torch.randint(1, config.vocab_size, (3, 40))
    texts = {f"s{i}": {"domain": "test", "ids": row.tolist()} for i, row in enumerate(ids)}
    n = config.n_routed_experts
    variants = dict(map(parse_variant, [f"all={n}:0", f"static={n // 2}:1e9", f"routed={n // 2}:0.3"]))
    out = run(config, model.embed_tokens, lambda i: model.layers[i], model.norm, head, texts, texts, variants, "cpu", log=lambda *_: None)
    with torch.inference_mode():  # the streamed teacher is the monolithic model
        want = model(input_ids=ids[:1]).last_hidden_state
    STATE.update(mass=None, variant=None)
    stream = {"ids": ids[:1]}
    stream_forward(config, model.embed_tokens, lambda i: model.layers[i], [stream], [], log=lambda *_: None)
    assert torch.equal(model.norm(stream["h"].mean(dim=2)), want), "streamed forward differs from the model's own"
    kl = {v: np.mean([s["kl"] for k, s in out[v].items() if not k.startswith("_")]) for v in variants}
    assert kl["all"] < 1e-9, kl  # every expert resident: the full model
    assert kl["static"] > 1e-4 and kl["routed"] > 1e-4, kl
    assert out["all"]["_policy"]["gate_weight_resident"] == 1.0
    assert out["static"]["_policy"]["admissions_per_token_layer"] == 0 < out["routed"]["_policy"]["admissions_per_token_layer"]
    print("selftest ok:", {k: round(float(v), 5) for k, v in kl.items()},
          {v: round(out[v]["_policy"]["gate_weight_resident"], 3) for v in variants})

    import tempfile

    class Exact:  # "codes" that are the checkpoint's own weights
        def residual_experts(self, i):
            return []

        def layer(self, i, hot):
            return {"base": (model.layers[i].mlp.experts.gate_up_proj, model.layers[i].mlp.experts.down_proj), "hot": {}}

    def kls(out):
        return {v: float(np.mean([s["kl"] for k, s in out[v].items() if not k.startswith("_")])) for v in out if not v.startswith("_")}

    layers = lambda i: model.layers[i]
    exact = run(config, model.embed_tokens, layers, model.norm, head, texts, texts, dict([parse_variant(f"exact={n}:0:codes")]),
                "cpu", log=lambda *_: None, codes=Exact(), importance=True)
    # the coded path with exact weights is the full model (to float32 rounding: the reference experts may run another kernel)
    assert kls(exact)["exact"] < 1e-7, kls(exact)
    moe = [i for i, t in enumerate(config.mlp_layer_types) if t == "sparse"]
    routed = STATE["mass"][moe] > 0
    assert ((STATE["imp"]["hidden"][moe].sum(-1) > 0).numpy() == routed).all() and not STATE["imp"]["down"][0].any()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_importance(tmp / "imp.npz", config)
        with np.load(tmp / "imp.npz") as z:
            assert z["layers"].tolist() == moe and z["hidden"].shape == (len(moe), n, config.hidden_size) and (z["hidden"] > 0).all()
        # 2-bit codes, then 1-bit residuals of half the experts, one group per row (the tiny widths do not divide 512)
        search = Search("cpu")
        for i in moe:
            native = model.layers[i].mlp.experts
            gates, ups = native.gate_up_proj.detach().chunk(2, dim=1)
            weights, base = {"w1": gates, "w3": ups, "w2": native.down_proj.detach()}, None
            for name, bits, experts in (("codes", 16, range(n)), ("resid", 8, range(0, n, 2))):
                arrays = {}
                for e in experts:
                    for proj, w in weights.items():
                        target = w[e] - (search.decode(base[f"{proj}_codes"][e], base[f"{proj}_scales"][e]) if bits == 8 else 0)
                        codes, scales, _ = search.fit(target.float(), torch.ones(w.shape[2]), w.shape[2], 4, bits)
                        arrays.setdefault(f"{proj}_codes", []).append(codes)
                        arrays.setdefault(f"{proj}_scales", []).append(scales)
                arrays = {k: np.stack(v) for k, v in arrays.items()}
                (tmp / name).mkdir(exist_ok=True)
                np.savez(tmp / name / f"layer-{i:02d}.npz", code_bits=np.asarray(bits), experts=np.asarray(list(experts)), **arrays)
                base = arrays
        specs = [f"c={n}:0:codes", f"ch={n}:0:codes:hot={n // 2}", f"cf={n}:0:codes:hot={n}:fp8", f"cr={n // 2}:0.3:codes:hot=2"]
        coded = run(config, model.embed_tokens, layers, model.norm, head, texts, texts, dict(map(parse_variant, specs)), "cpu",
                    log=lambda *_: None, codes=Codes(tmp / "codes", tmp / "resid", "cpu", torch.float32))
    kl = kls(coded)
    assert kl["cf"] < 1e-7 and 1e-4 < kl["ch"] < kl["c"] and kl["cr"] > 1e-4, kl  # residuals help, fp8 hot is exact
    assert coded["ch"]["_policy"]["codes"] == {"hot": n // 2, "hot_fp8": False}
    print("selftest codes ok:", {k: round(v, 5) for k, v in kl.items()})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--cal", type=Path)
    p.add_argument("--eval", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--variant", action="append", default=[], metavar="NAME=K:TAU[:POLICY][:codes[:hot=H[:fp8]]]")
    p.add_argument("--importance", type=Path, help="save the calibration group's per-expert importance here")
    p.add_argument("--codes", type=Path, help="fit_codes.py 2-bit codes, every expert (for coded variants)")
    p.add_argument("--residual", type=Path, help="fit_codes.py 1-bit residuals of --codes (for hot sets)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--cal-tokens", type=int, default=512)
    p.add_argument("--layers", type=int, help="only the first N layers (a smoke run)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return 0
    from tokenizers import Tokenizer

    device, dtype = torch.device(args.device), torch.bfloat16
    text = dict(json.loads((args.ckpt / "config.json").read_text())["text_config"])
    text["num_nextn_predict_layers"] = 0
    config = Glm5NextTextConfig(**text)
    config._attn_implementation = "eager"
    if args.layers:
        config.num_hidden_layers = args.layers
    ckpt, tokenizer = Checkpoint(args.ckpt), Tokenizer.from_file(str(args.ckpt / "tokenizer.json"))

    def sessions(path: Path, limit: int) -> dict:
        return {name: {"domain": d["domain"], "ids": tokenizer.encode(d["text"]).ids[:limit]}
                for name, d in json.loads(path.read_text()).items()}

    embed_weight = ckpt.read(PREFIX + "embed_tokens.weight", device).to(dtype)
    head_name = next(n for n in ckpt.where if n.endswith("lm_head.weight"))
    head = ckpt.read(head_name, device).to(dtype)
    norm = M.Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps).to(device, dtype)
    norm.load_state_dict({"weight": ckpt.read(PREFIX + "norm.weight", device).to(dtype)})

    def layers(i: int):
        layer = M.Glm5NextTextDecoderLayer(config, i).to(device, dtype).eval()
        layer.load_state_dict(ckpt.layer_state(i, device, dtype), strict=True, assign=True)
        return layer

    variants = dict(map(parse_variant, args.variant))
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary = run(config, lambda ids: F.embedding(ids, embed_weight), layers, norm, head,
                  sessions(args.cal, args.cal_tokens), sessions(args.eval, args.max_tokens) if args.eval else {}, variants,
                  device, log=lambda *a: print(*a, flush=True), importance=args.importance is not None,
                  codes=Codes(args.codes, args.residual, device, dtype) if args.codes else None)
    np.save(args.out / "mass.npy", np.asarray(summary.pop("_mass")))
    if args.importance:
        print(f"{args.importance}: {save_importance(args.importance, config)} (expert, space) importances replaced by the layer mean")
    summary["_run"] = {"seconds": round(time.time() - started, 1), "layers": config.num_hidden_layers,
                       "max_tokens": args.max_tokens, "head": head_name, "variants": args.variant}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    teacher = [s["nll"] for s in summary.get("_teacher", {}).values()]
    if teacher:
        print(f"teacher mean NLL {np.mean(teacher):.3f} over {len(teacher)} sessions")
    for name in variants:
        kl = [s["kl"] for k, s in summary[name].items() if not k.startswith("_")]
        print(name, f"KL {np.mean(kl):.4f}", json.dumps(summary[name]["_policy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
