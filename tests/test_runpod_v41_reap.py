"""CPU checks for the RunPod V4.1 REAP kit (scripts/runpod/v41_reap).

The MoE below copies the reference ``inference/model.py`` (pinned revision)
Gate / Expert / MoE with world_size = 1 and nn.Linear in place of the FP8/FP4
Linear. The patched forward must give the same output and the statistics must
match a direct float64 computation and pass KEEP's own validators.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "scripts" / "runpod" / "v41_reap"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"v41_{name}", KIT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- collector


def _reference(torch):
    nn, F = torch.nn, torch.nn.functional
    ref = types.ModuleType("ref")
    ref.world_size = 1

    class Gate(nn.Module):
        def __init__(self, n, dim, topk):
            super().__init__()
            self.topk, self.route_scale = topk, 1.5
            self.weight = nn.Parameter(torch.randn(n, dim) * 0.3)
            self.bias = nn.Parameter(torch.randn(n) * 0.1)

        def forward(self, x, image_mask=None):
            scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
            indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
            weights = scores.gather(1, indices)
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
            weights *= self.route_scale
            return weights, indices

    class Expert(nn.Module):
        def __init__(self, dim, inter, swiglu_limit=7.0):
            super().__init__()
            self.w1 = nn.Linear(dim, inter, bias=False)
            self.w2 = nn.Linear(inter, dim, bias=False)
            self.w3 = nn.Linear(dim, inter, bias=False)
            self.swiglu_limit = swiglu_limit

        def forward(self, x, weights=None):
            dtype = x.dtype
            gate = self.w1(x).float()
            up = self.w3(x).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            x = F.silu(gate) * up
            if weights is not None:
                x = weights * x
            return self.w2(x.to(dtype))

    class MoE(nn.Module):
        def __init__(self, layer_id, dim=16, inter=12, n=8, topk=2):
            super().__init__()
            self.layer_id, self.dim, self.n_routed_experts = layer_id, dim, n
            self.experts_start_idx, self.experts_end_idx = 0, n
            self.gate = Gate(n, dim, topk)
            self.experts = nn.ModuleList([Expert(dim, inter) for _ in range(n)])
            self.shared_experts = Expert(dim, inter)

        def forward(self, x, image_mask=None):
            shape = x.size()
            x = x.view(-1, self.dim)
            weights, indices = self.gate(x)
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                idx, top = torch.where(indices == i)
                y[idx] += self.experts[i](x[idx], weights[idx, top, None])
            y += self.shared_experts(x)
            return y.type_as(x).view(shape)

    ref.MoE = MoE
    return ref


def test_patched_moe_matches_reference_and_direct_stats():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    collect = _load("collect")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    ref = _reference(torch)
    layers = [ref.MoE(i) for i in range(2)] + [ref.MoE(2)]  # layer 2 plays an mtp block
    xs = [torch.randn(1, 37, 16) for _ in layers]
    expected = [m(x) for m, x in zip(layers, xs)]

    original = ref.MoE.forward
    state = {"rank": 0}
    stats = collect.SessionStats(torch=torch, n_layers=2, n_experts=8, dims={"hidden": 16, "down": 12}, device="cpu")
    state["stats"] = stats
    ref.MoE.forward = collect.make_moe_forward(ref, torch, F, None, state)
    try:
        got = [m(x) for m, x in zip(layers, xs)]
    finally:
        ref.MoE.forward = original
    for a, b in zip(expected, got):
        assert torch.equal(a, b)

    arrays = stats.arrays([0, 1])
    for layer, (m, x) in enumerate(zip(layers[:2], xs)):
        flat = x.view(-1, 16)
        w, idx = m.gate(flat)
        count = np.zeros(8)
        gnorm = np.zeros(8)
        fmax = np.zeros(8)
        hid = np.zeros((8, 16))
        down = np.zeros((8, 12))
        for row in range(flat.shape[0]):
            for slot in range(idx.shape[1]):
                e, g = int(idx[row, slot]), float(w[row, slot])
                ex = m.experts[e]
                h = F.silu(ex.w1(flat[row]).clamp(max=7.0)) * ex.w3(flat[row]).clamp(-7.0, 7.0)
                out = ex.w2(g * h).double()
                count[e] += 1
                gnorm[e] += float(out.norm())
                fmax[e] = max(fmax[e], float(out.norm()) / g)
                hid[e] += flat[row].double().square().numpy()
                down[e] += h.double().square().detach().numpy()
        np.testing.assert_array_equal(arrays["reap__route_count"][layer], count)
        np.testing.assert_allclose(arrays["reap__gate_norm_sum"][layer], gnorm, rtol=1e-5)
        np.testing.assert_allclose(arrays["reap__norm_max"][layer], fmax, rtol=1e-5)
        np.testing.assert_allclose(arrays["importance_sum__hidden"][layer], hid, rtol=1e-5)
        np.testing.assert_allclose(arrays["importance_sum__down"][layer], down, rtol=1e-4, atol=1e-6)
    assert arrays["routes__expert_ids"].shape == (37, 2, 2)
    assert arrays["routes__expert_ids"].dtype == np.int16

    from keep.quality.dsv4_teacher_runner import validate_dsv4_calibration_capture
    from keep.quality.reap import ReapStats

    validate_dsv4_calibration_capture(arrays, layers=[0, 1], num_experts=8)
    assert ReapStats.from_arrays(arrays).routed_slots.tolist() == [74, 74]
    torch.set_grad_enabled(True)


def test_head_capture_matches_full_logits_schema():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    collect = _load("collect")
    from keep.quality.dsv4_teacher_runner import validate_dsv4_logits_capture

    head = types.SimpleNamespace(weight=torch.randn(50, 8))
    head.forward = lambda x, full_logits=False: F.linear(x[:, -1].float(), head.weight)
    positions = np.array([2, 3, 7])
    targets = torch.as_tensor([5, 9, 11])
    state = {"rank": 0, "positions": torch.as_tensor(positions), "targets": targets, "captured": {k: [] for k in collect.CAPTURE_KEYS}}
    head.forward = collect.make_head_forward(head, torch, F, None, state, top_k=10, rows_per_slice=2)
    x = torch.randn(1, 9, 8)
    head.forward(x)
    cap = {k: np.concatenate(v) for k, v in state["captured"].items() if v}
    cap["positions"] = positions.astype(np.int32)
    cap["target_token_ids"] = np.zeros(3, dtype=np.int32)
    validate_dsv4_logits_capture(cap, top_k=10, supervised=3)
    full = F.linear(x[0, positions], head.weight)
    np.testing.assert_allclose(cap["logsumexp"], torch.logsumexp(full, -1).numpy(), rtol=1e-6)
    assert (cap["topk_logit_ids"][:, 0] == full.argmax(-1).numpy()).all()
    np.testing.assert_allclose(cap["target_logit"], full.gather(1, targets[:, None])[:, 0].numpy(), rtol=1e-6)

    # Student pass scored against that teacher: the same head gives KL 0 and full agreement.
    lse = torch.as_tensor(cap["logsumexp"])
    state["captured"] = {k: [] for k in collect.CAPTURE_KEYS}
    state["teacher"] = {
        "ids": torch.as_tensor(cap["topk_logit_ids"].astype(np.int64)),
        "logprob": torch.as_tensor(cap["topk_logit_values"].astype(np.float32)) - lse[:, None],
        "targets": targets,
    }
    head.forward(x)
    student = {k: np.concatenate(v) for k, v in state["captured"].items() if v}
    assert set(student) == {"kl_on_teacher_topk", "student_nll", "top1_agree"}
    np.testing.assert_allclose(student["kl_on_teacher_topk"], 0.0, atol=2e-3)
    assert student["top1_agree"].all()
    np.testing.assert_allclose(student["student_nll"], (lse - torch.as_tensor(cap["target_logit"])).numpy(), rtol=1e-5)


def test_pack_rules(tmp_path):
    collect = _load("collect")
    pack = tmp_path / "pack.jsonl"
    rows = [
        {"id": "a", "session": "s1", "split": "calibration", "domain": "rust", "token_ids": [1, 2, 3], "target_spans": []},
        {"id": "b", "session": "e1", "split": "eval", "domain": "bash", "token_ids": [1, 2, 3, 4, 5], "target_spans": [[0, 2], [3, 5]]},
    ]
    pack.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sessions = collect.load_sessions(pack)
    assert list(sessions) == ["s1", "e1"]
    # targets 1 and 3, 4 (0 has no predecessor); logits are read one position earlier
    assert collect.target_positions(rows[1]).tolist() == [0, 2, 3]
    rows.append({**rows[1], "id": "c"})
    pack.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="exactly one sample"):
        collect.load_sessions(pack)


# --------------------------------------------------------------------------- pruner


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _read_safetensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    header.pop("__metadata__", None)
    base = 8 + size
    return {n: (i["dtype"], i["shape"], data[base + i["data_offsets"][0] : base + i["data_offsets"][1]]) for n, i in header.items()}


def test_prune_copies_bytes_renumbers_and_slices(tmp_path):
    prune = _load("prune")
    rng = np.random.default_rng(0)
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "inference").mkdir(parents=True)
    shards: dict[str, dict] = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for layer in range(2):
        shard = shards[f"model-0000{layer + 1}-of-00002.safetensors"]
        shard[f"layers.{layer}.ffn.gate.weight"] = ("BF16", [6, 4], rng.bytes(6 * 4 * 2))
        shard[f"layers.{layer}.ffn.gate.bias"] = ("F32", [6], rng.bytes(24))
        shard[f"layers.{layer}.ffn.gate.bias_vl"] = ("F32", [6], rng.bytes(24))
        shard[f"layers.{layer}.attn.wq.weight"] = ("F8_E4M3", [4, 4], rng.bytes(16))
        for e in range(6):
            shard[f"layers.{layer}.ffn.experts.{e}.w1.weight"] = ("I8", [3, 2], rng.bytes(6))  # packed fp4 stand-in
            shard[f"layers.{layer}.ffn.experts.{e}.w1.scale"] = ("F8_E8M0", [3, 1], rng.bytes(3))
    shards["model-00002-of-00002.safetensors"]["mtp.0.ffn.experts.0.w1.weight"] = ("I8", [3, 2], rng.bytes(6))
    shards["model-00002-of-00002.safetensors"]["mtp.0.ffn.gate.weight"] = ("BF16", [2, 4], rng.bytes(16))
    weight_map = {}
    for name, tensors in shards.items():
        _write_safetensors(src / name, tensors)
        weight_map.update({t: name for t in tensors})
    (src / prune.INDEX).write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps({"text_config": {"n_routed_experts": 6, "dspark_n_routed_experts": 2}}))
    (src / "inference" / "config.json").write_text(json.dumps({"n_routed_experts": 6, "dspark_n_routed_experts": 2}))
    (src / "tokenizer.json").write_text("{}")
    survivors = tmp_path / "survivors.json"
    survivors.write_text(json.dumps({"record_type": "keep_reap_survivors_v1", "survivors": {"0": [1, 4, 5], "1": [0, 2, 3]}}))

    assert prune.main(["--src", str(src), "--survivors", str(survivors), "--out", str(out)]) == 0

    before = {**_read_safetensors(src / "model-00001-of-00002.safetensors"), **_read_safetensors(src / "model-00002-of-00002.safetensors")}
    after = {**_read_safetensors(out / "model-00001-of-00002.safetensors"), **_read_safetensors(out / "model-00002-of-00002.safetensors")}
    assert after["layers.0.ffn.experts.1.w1.weight"] == before["layers.0.ffn.experts.4.w1.weight"]
    assert after["layers.0.ffn.experts.2.w1.scale"] == before["layers.0.ffn.experts.5.w1.scale"]
    assert "layers.0.ffn.experts.3.w1.weight" not in after
    gate = before["layers.1.ffn.gate.weight"][2]
    assert after["layers.1.ffn.gate.weight"][1] == [3, 4]
    assert after["layers.1.ffn.gate.weight"][2] == gate[0:8] + gate[16:24] + gate[24:32]
    assert after["layers.0.ffn.gate.bias_vl"][2] == before["layers.0.ffn.gate.bias_vl"][2][4:8] + before["layers.0.ffn.gate.bias_vl"][2][16:24]
    assert after["layers.0.attn.wq.weight"] == before["layers.0.attn.wq.weight"]
    assert after["mtp.0.ffn.gate.weight"] == before["mtp.0.ffn.gate.weight"]
    index = json.loads((out / prune.INDEX).read_text())
    assert set(index["weight_map"]) == set(after)
    assert json.loads((out / "config.json").read_text())["text_config"] == {"n_routed_experts": 3, "dspark_n_routed_experts": 2}
    assert json.loads((out / "inference" / "config.json").read_text())["n_routed_experts"] == 3
    sums = (out / "SHA256SUMS").read_text()
    assert "tokenizer.json" in sums and "keep-prune.json" in sums


# --------------------------------------------------------------------------- pack builder


def test_english_filter_rejects_cjk_and_other_scripts():
    pack = _load("build_pack")
    assert pack.is_english("fix the TLS handshake; naïve café résumé")
    assert not pack.is_english("fix: Form组件配置")
    assert not pack.is_english("Привет мир, hello there")
    assert not pack.is_english("リリース notes")


def test_toucan_rows_become_openai_tool_calls():
    pack = _load("build_pack")
    row = {
        "subset_name": "multi-turn",
        "tools": json.dumps([{"type": "function", "function": {"name": "srv-find", "parameters": {}}}]),
        "messages": json.dumps([
            {"role": "user", "content": "find x"},
            {"role": "assistant", "content": "Looking."},
            {"role": "tool_call", "content": "{'name': 'srv-find', 'arguments': '{\"q\": \"x\"}'}"},
            {"role": "tool_response", "content": "found"},
            {"role": "assistant", "content": "Done."},
        ]),
    }
    kind, messages = pack.adapt_toucan(row)
    assert kind == "chat" and messages[0]["tools"][0]["function"]["name"] == "srv-find"
    assert messages[2]["tool_calls"][0]["function"] == {"name": "srv-find", "arguments": '{"q": "x"}'}
    assert messages[3] == {"role": "tool", "content": "found", "tool_call_id": "call_1"}
    assert pack.adapt_toucan({**row, "subset_name": "single-turn-original"}) is None


def test_build_keeps_groups_disjoint_meets_budgets_and_sessions():
    pack = _load("build_pack")
    rows = [{"repo": f"r{i % 40}", "text": "word " * (20 + i % 7)} for i in range(4000)]
    rows.append({"repo": "zh", "text": "中文 text"})
    source = {"name": "fake", "dataset": "fake", "load": {}, "adapter": "text", "domain": "d", "group": "repo",
              "budget": {"d": (3000, 400)}}  # fmt: skip
    render = lambda kind, text: [0] + [7] * len(text.split())
    records, report = pack.build([source], render, eos=1, max_tokens=64, pack_tokens=0, session_tokens=500,
                                 revisions={}, rows=lambda s, r: iter(rows))  # fmt: skip
    cal = [r for r in records if r["split"] == "calibration"]
    ev = [r for r in records if r["split"] == "eval"]
    assert {g for r in cal for g in r["groups"]}.isdisjoint({g for r in ev for g in r["groups"]})
    assert report["tokens"]["d"]["calibration"] >= 3000 and report["tokens"]["d"]["eval"] >= 400
    sizes = Counter()
    for r in cal:
        sizes[r["session"]] += len(r["token_ids"])
    assert max(sizes.values()) <= 500
    assert len({r["session"] for r in ev}) == len(ev) and all(r["target_spans"] == [[1, len(r["token_ids"])]] for r in ev)
    assert report["sources"]["fake"].get("non_english", 0) == 0  # the CJK row sits after the budgets are met
    rows.insert(0, rows.pop())
    _, report = pack.build([source], render, eos=1, max_tokens=64, pack_tokens=0, session_tokens=500,
                           revisions={}, rows=lambda s, r: iter(rows))  # fmt: skip
    assert report["sources"]["fake"]["non_english"] == 1
