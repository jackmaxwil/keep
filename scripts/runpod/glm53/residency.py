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

    python residency.py --ckpt DIR --cal cal.json --eval eval.json --out OUT \\
        --variant s108=108:1e9 --variant k108=108:0.3
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

from local_eval import residual_cache

PREFIX = "model.language_model."
EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
STATE: dict = {"variant": None, "mass": None, "layer": 0, "warm": None, "counts": {}}


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
        k, tau, policy = spec
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
    routed = self.experts(x, index, weight * gate.routed_scaling_factor).view(hidden_states.shape)
    return routed + self.shared_experts(hidden_states)


def stream_forward(config, embed, layers, streams: list[dict], calibration: list[dict], log=print) -> None:
    """Run every stream through every layer. ``layers(i)`` returns layer ``i`` as a module; each stream is
    ``{"ids", "variant", "name"}`` and gains ``"h"``, the hidden streams leaving the last layer."""
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


def parse_variant(spec: str) -> tuple[str, tuple[int, float, str]]:
    name, _, rest = spec.partition("=")
    k, tau, *policy = rest.split(":", 2)
    return name, (int(k), float(tau), policy[0] if policy else "lru")


def make_streams(sessions: dict, variants: dict, device) -> list[dict]:
    return [{"session": s, "domain": d["domain"], "ids": torch.as_tensor([d["ids"]], device=device), "variant": spec, "name": name}
            for s, d in sessions.items() for name, spec in ((None, None), *variants.items())]


def run(config, embed, layers, norm, head, cal: dict, evals: dict, variants: dict, device, log=print) -> dict:
    n = config.n_routed_experts
    STATE.update(mass=np.zeros((config.num_hidden_layers, n)), warm={}, counts={}, served={v: np.zeros(2) for v in variants})
    calibration = [{"ids": torch.as_tensor([d["ids"]], device=device)} for d in cal.values()]
    streams = make_streams(evals, variants, device)
    M.Glm5NextTextMoE.forward = moe_forward
    try:
        stream_forward(config, embed, layers, streams, calibration, log)
    finally:
        M.Glm5NextTextMoE.forward = _full_moe
    summary = score(config, norm, head, streams)
    tokens = sum(len(d["ids"]) for d in evals.values())
    moe_layers = sum(t == "sparse" for t in config.mlp_layer_types[: config.num_hidden_layers])
    for name, (k, tau, policy) in variants.items():
        summary[name]["_policy"] = {"residents": k, "tau": tau, "policy": policy,
                                    "gate_weight_resident": float(STATE["served"][name][0] / STATE["served"][name][1]),
                                    "admissions_per_token_layer": float(STATE["counts"][name][:, 4].sum() / tokens / moe_layers)}
    summary["_mass"] = STATE["mass"].tolist()
    return summary


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--cal", type=Path)
    p.add_argument("--eval", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--variant", action="append", default=[], metavar="NAME=K:TAU[:POLICY]")
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
                  sessions(args.cal, args.cal_tokens), sessions(args.eval, args.max_tokens), variants, device,
                  log=lambda *a: print(*a, flush=True))
    np.save(args.out / "mass.npy", np.asarray(summary.pop("_mass")))
    summary["_run"] = {"seconds": round(time.time() - started, 1), "layers": config.num_hidden_layers,
                       "max_tokens": args.max_tokens, "head": head_name, "variants": args.variant}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    teacher = [s["nll"] for s in summary["_teacher"].values()]
    print(f"teacher mean NLL {np.mean(teacher):.3f} over {len(teacher)} sessions")
    for name in variants:
        kl = [s["kl"] for k, s in summary[name].items() if not k.startswith("_")]
        print(name, f"KL {np.mean(kl):.4f}", json.dumps(summary[name]["_policy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
