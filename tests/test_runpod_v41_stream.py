"""The layer-streamed driver against DeepSeek's own Transformer.forward, on CPU at toy size.

The real ``model.py`` and ``engram.py`` (tests/fixtures/v41_reference) run with
plain-torch kernels. The toy model keeps every feature that makes streaming
subtle:

- hyper-connections;
- compressed KV shared from source layers;
- an indexer shared across layers;
- a candidate-block source and a layer consuming it;
- an Engram layer;
- a MoE.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
REF_DIR = REPO / "tests" / "fixtures" / "v41_reference"
KIT = REPO / "scripts" / "runpod" / "v41_reap"
sys.path.insert(0, str(KIT))

VOCAB = 96


def _install_stubs() -> None:
    import torch_kernel

    sys.modules["kernel"] = torch_kernel
    image = types.ModuleType("image_processor")
    image.IMAGE = image.IMAGE_END = image.IMAGE_NEW_LINE = image.IMAGE_START = None
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = None
    sys.modules.setdefault("image_processor", image)
    sys.modules.setdefault("vision", vision)
    try:
        import sympy
    except ImportError:  # engram only needs isprime
        sympy = types.ModuleType("sympy")
        sympy.isprime = lambda n: n > 1 and all(n % p for p in range(2, int(n**0.5) + 1))
        sys.modules["sympy"] = sympy


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    _install_stubs()
    import stream

    ref = stream.import_reference(REF_DIR)
    import engram

    engram.build_compressed_token_map = lambda tokenizer: (list(range(VOCAB)), VOCAB)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    base = dict(
        max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None, vocab_size=VOCAB,
        dim=64, moe_inter_dim=32, n_layers=6, n_mtp_layers=0, n_heads=4, n_routed_experts=8,
        n_activated_experts=2, route_scale=1.5, swiglu_limit=10.0, q_lora_rank=32, head_dim=32,
        rope_head_dim=8, norm_eps=1e-6, o_groups=2, o_lora_rank=16, window_size=8,
        compress_ratios=(0, 2, 2, 1, 1, 0), kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
        index_n_heads=2, index_head_dim=16, index_topk=4, candidate_source_layer=3,
        candidate_topk_blocks=2, candidate_block_size=2, original_seq_len=64,
        engram_layer_ids=(2,), engram_max_ngram_size=3, engram_vocab_size=40, engram_n_heads=2,
        engram_head_dim=32, engram_pad_id=2, engram_compressed_vocab_size=VOCAB,
    )
    layout = ref.EngramLayout.from_args(types.SimpleNamespace(**{**base, "engram_num_embeddings": (0,)}))
    rows = int(sum(p for per_ngram in layout.primes[0] for p in per_ngram))
    args = ref.ModelArgs(**base, engram_num_embeddings=(rows,))
    model = ref.Transformer(args, tokenizer=None)

    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            shape = p.shape
            if p.dtype == torch.float8_e4m3fn:
                p.copy_((torch.randn(shape, generator=gen) * 0.5).to(p.dtype))
            elif p.dtype == torch.float8_e8m0fnu:
                p.copy_(torch.full(shape, 0.5).to(p.dtype))
            elif "norm" in name or name.endswith(("q_weight", "k_weight")):
                p.copy_(1 + 0.1 * torch.randn(shape, generator=gen))
            elif "hc_" in name and "scale" in name:
                p.copy_(0.5 + 0.1 * torch.randn(shape, generator=gen))
            else:
                fan_in = shape[-1] if len(shape) > 1 else 1
                p.copy_(torch.randn(shape, generator=gen) / max(fan_in, 1) ** 0.5)

    from safetensors.torch import save_file

    ckpt = tmp_path_factory.mktemp("hf")
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    names = sorted(sd)
    half = len(names) // 2
    shards = {"model-00001.safetensors": names[:half], "model-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: sd[k] for k in keys}, str(ckpt / shard))
        weight_map.update({k: shard for k in keys})
    (ckpt / stream.INDEX).write_text(json.dumps({"weight_map": weight_map}))

    records = [torch.randint(0, VOCAB, (1, n), generator=gen) for n in (37, 23, 57)]
    return types.SimpleNamespace(ref=ref, stream=stream, args=args, model=model, ckpt=ckpt, records=records)


def _reference_hidden(toy, tokens):
    captured = {}
    handle = toy.model.norm.register_forward_hook(lambda m, i, o: captured.setdefault("h", o))
    try:
        with torch.inference_mode():
            toy.model.forward(tokens, 0)
    finally:
        handle.remove()
    return captured["h"]


def _streamer(toy):
    stream = toy.stream
    ckpt = stream.HFCheckpoint(toy.ckpt)
    rows = {layer: stream.EngramRows.from_checkpoint(ckpt, layer, toy.ref.fp8_block_size) for layer in toy.args.engram_layer_ids}
    return stream.LayerStreamer(toy.ref, toy.args, ckpt.load, rows, toy.model.engram_hash, "cpu")


@pytest.mark.parametrize("grouping", ["one-moe-batch", "moe-per-record"])
def test_streamed_forward_matches_reference(toy, grouping):
    reference = [_reference_hidden(toy, t) for t in toy.records]
    streamer = _streamer(toy)
    items = [
        toy.stream.Item(tokens=t, group=0 if grouping == "one-moe-batch" else i) for i, t in enumerate(toy.records)
    ]
    with torch.inference_mode():
        streamer.forward(items)
        streamed = [streamer.final_hidden(item) for item in items]
    for want, got in zip(reference, streamed):
        assert got.shape == want.shape
        torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)


def test_record_alone_equals_record_in_chunk(toy):
    """The shared_attn save/restore: a record's output does not depend on its chunk-mates."""

    streamer = _streamer(toy)
    alone = [toy.stream.Item(tokens=toy.records[2], group=0)]
    crowd = [toy.stream.Item(tokens=t, group=i) for i, t in enumerate(toy.records)]
    with torch.inference_mode():
        streamer.forward(alone)
        streamer.forward(crowd)
        torch.testing.assert_close(streamer.final_hidden(crowd[2]), streamer.final_hidden(alone[0]), rtol=0, atol=0)


def test_pack_bits_round_trip(toy):
    mask = torch.rand(3, 5, 13) > 0.5
    assert torch.equal(toy.stream.unpack_bits(toy.stream.pack_bits(mask)), mask)


# --------------------------------------------------------------------------- the two sweeps (collect.py)

LAYERS = list(range(6))
DIMS = {"hidden": 64, "down": 32}


def _record(tokens, name, split):
    ids = tokens[0].tolist()
    return {"id": f"{name}-{len(ids)}", "session": name, "split": split, "domain": "code", "token_ids": ids, "target_spans": [[3, len(ids)]]}


def _reference_stats(toy, collect, records, dims):
    stats = collect.SessionStats(torch=torch, n_layers=6, n_experts=8, dims=dims, device="cpu")
    original = toy.ref.MoE.forward
    toy.ref.MoE.forward = collect.make_moe_forward(toy.ref, torch, torch.nn.functional, None, {"rank": 0, "stats": stats})
    try:
        hidden = [_reference_hidden(toy, torch.as_tensor([r["token_ids"]])) for r in records]
    finally:
        toy.ref.MoE.forward = original
    return stats.arrays(LAYERS), hidden


def _sweep(toy, collect, monkeypatch, chunk, split, students=None):
    streamer = _streamer(toy)
    monkeypatch.setattr(toy.ref.MoE, "forward", toy.ref.MoE.forward)  # install_hooks patches it; undo at teardown
    state = collect.install_hooks(streamer, top_k=16, rows_per_slice=7)
    return collect.run_chunk(streamer, state, chunk, split=split, students=students or {}, dims=DIMS, device="cpu"), streamer


def _assert_stats_match(got, want):
    assert set(got) == set(want)
    for key, value in want.items():
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(got[key], value, err_msg=key)
        else:
            np.testing.assert_allclose(got[key], value, rtol=1e-6, atol=1e-12, err_msg=key)


def test_calibration_sweep_matches_reference(toy, monkeypatch):
    import collect

    r0, r1, r2 = toy.records
    chunk = [("cal-0", [_record(r0, "cal-0", "calibration"), _record(r1, "cal-0", "calibration")]),
             ("cal-1", [_record(r2, "cal-1", "calibration")])]
    want = {name: _reference_stats(toy, collect, records, DIMS)[0] for name, records in chunk}
    results, _ = _sweep(toy, collect, monkeypatch, chunk, "calibration")
    assert set(results) == {("teacher", "cal-0"), ("teacher", "cal-1")}
    for name, _ in chunk:
        _assert_stats_match(results[("teacher", name)], want[name])
    assert results[("teacher", "cal-0")]["routes__expert_ids"].shape == (60, 6, 2)


def test_eval_sweep_teacher_and_students_match_reference(toy, monkeypatch):
    import collect

    F = torch.nn.functional
    chunk = [(f"eval-{i}", [_record(t, f"eval-{i}", "eval")]) for i, t in enumerate(toy.records[:2])]
    gen = torch.Generator().manual_seed(7)
    half = {"survivors": {str(layer): sorted(torch.randperm(8, generator=gen)[:4].tolist()) for layer in LAYERS}}
    students = {"8": collect.survivor_masks({"survivors": {str(layer): list(range(8)) for layer in LAYERS}}, 8),
                "4": collect.survivor_masks(half, 8)}

    want = {name: _reference_stats(toy, collect, records, {}) for name, records in chunk}
    with torch.no_grad():
        full = {layer: toy.model.layers[layer].ffn.gate.bias.clone() for layer in LAYERS}
        for layer, mask in students["4"].items():
            toy.model.layers[layer].ffn.gate.bias[mask] = float("-inf")
        masked = {name: _reference_hidden(toy, torch.as_tensor([records[0]["token_ids"]])) for name, records in chunk}
        for layer, bias in full.items():
            toy.model.layers[layer].ffn.gate.bias.copy_(bias)

    results, _ = _sweep(toy, collect, monkeypatch, chunk, "eval", students)
    assert set(results) == {(s, name) for s in ("teacher", "8", "4") for name, _ in chunk}
    weight = toy.model.head.weight.detach()
    for name, records in chunk:
        stats, (hidden,) = want[name]
        teacher = results[("teacher", name)]
        _assert_stats_match({k: v for k, v in teacher.items() if k in stats}, stats)
        assert not any(k.startswith("importance_sum") for k in teacher)

        positions = collect.target_positions(records[0])
        targets = torch.as_tensor(records[0]["token_ids"])[positions + 1]
        logits = F.linear(hidden[0, positions].float(), weight)
        np.testing.assert_allclose(teacher["logsumexp"], torch.logsumexp(logits, -1).numpy(), rtol=1e-6)
        np.testing.assert_allclose(teacher["target_logit"], logits.gather(1, targets[:, None])[:, 0].numpy(), rtol=1e-6)
        np.testing.assert_array_equal(teacher["positions"], positions)

        everyone = results[("8", name)]
        np.testing.assert_allclose(everyone["kl_on_teacher_topk"], 0.0, atol=2e-3)
        assert everyone["top1_agree"].all()

        pruned = F.linear(masked[name][0, positions].float(), weight)
        nll = torch.logsumexp(pruned, -1) - pruned.gather(1, targets[:, None])[:, 0]
        np.testing.assert_allclose(results[("4", name)]["student_nll"], nll.numpy(), rtol=1e-5)
        assert not np.allclose(results[("4", name)]["student_nll"], everyone["student_nll"])


def test_chunk_sessions_keeps_sessions_whole():
    import collect

    sessions = [(n, [{"token_ids": [0] * size}]) for n, size in (("a", 5), ("b", 5), ("c", 20), ("d", 1))]
    chunks = collect.chunk_sessions(sessions, budget=24, streams=2)
    assert [[n for n, _ in c] for c in chunks] == [["a", "b"], ["c"], ["d"]]
