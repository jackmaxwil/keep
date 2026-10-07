"""DeepSeek-V4-Flash layer-sequential streaming teacher runner.

Everything here runs on a synthetic tiny model and a synthetic tiny pack -- no
163 GB checkpoint, no 90 MB pack. The three properties that matter and are easy
to get silently wrong:

1. **Layer-major == chunk-major, bit-exactly.** The runner inverts the loop
   nesting (layer outer, chunk inner) so a session reads each layer's experts
   once instead of once per chunk. That is only safe if it computes the same
   function, so it is asserted against the chunk-major reference rather than
   argued for in a comment.
2. **Collecting activation statistics does not change the forward.** The
   calibration path re-expresses the block body far enough to see the routed MoE
   input, and the statistics tap sits inside a ``SwitchGLU`` subclass. Both are
   asserted equal to the untouched module.
3. **A completed session file is either whole or absent.** ``kill -9`` during a
   write must not leave a half-npz that the resumability check would then have
   to be clever about.

Tests that need the real shards or the real pack are marked ``checkpoint`` /
``pack`` and skip when absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

from benchmarks.produce_dsv4_teacher_cache import _refuse_wired_limit_env
from keep.quality.dsv4_teacher_runner import (
    DSV4_PACK_RECORD_TYPE,
    DSV4_TEACHER_MODES,
    IMATRIX_SPACES,
    Dsv4RunHeartbeat,
    Dsv4Session,
    Dsv4StreamStats,
    InMemoryExpertProvider,
    MtpDrafterUnavailable,
    _atomic_savez,
    _ImatrixAccumulator,
    _MXFP4SwitchLinear,
    _StatsSwitchGLU,
    build_dsv4_expert_span_index,
    finalize_dsv4_calibration_imatrix,
    layer_major_prefill,
    load_dsv4_teich_pack,
    run_dsv4_teacher_production,
    session_output_path,
    validate_dsv4_calibration_capture,
    validate_dsv4_logits_capture,
    validate_dsv4_mtp_targets_capture,
)
from ramp.models.deepseek_v4_flash_adapter import (
    DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS,
    DeepseekV4FlashVQModel,
    DeepseekV4FlashVQModelArgs,
    LimitedSwiGLU,
    PoolingCache,
    SparseCompressedAttention,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = ROOT / "tests/data/deepseek-v4-flash-0731-config.json"
CHECKPOINT = Path.home() / "models/DeepSeek-V4-Flash-0731"
REAL_PACK = Path.home() / "models/teich/dsv4-coding-agent-v1-20260811.json"

requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"DeepSeek-V4-Flash-0731 shards not present at {CHECKPOINT}",
)
requires_pack = pytest.mark.skipif(
    not REAL_PACK.is_file(), reason=f"retokenized teich pack not present at {REAL_PACK}"
)


# ---------------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------------


def _tiny_args(**overrides) -> DeepseekV4FlashVQModelArgs:
    """A 4-layer model whose ``compress_ratios`` hits every attention variant.

    ``[0, 4, 128, 4]`` gives one ``LocalAttention``, two
    ``SparseCompressedAttention`` and one ``CompressedAttention``, and
    ``num_hash_layers=2`` puts hash routing on layers 0-1 and scored routing on
    2-3 -- so both ``MoEGate`` branches run in every forward under test.
    """

    base = {
        "model_type": "deepseek_v4",
        "vocab_size": 512,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 2,
        "compress_ratios": [0, 4, 128, 4],
        "sliding_window": 8,
        "head_dim": 32,
        "q_lora_rank": 32,
        "o_lora_rank": 32,
        "o_groups": 2,
        "qk_rope_head_dim": 8,
        "index_head_dim": 8,
        "index_n_heads": 4,
        "index_topk": 4,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "swiglu_limit": 10.0,
    }
    base.update(overrides)
    return DeepseekV4FlashVQModelArgs(**base)


def _randomise(model: DeepseekV4FlashVQModel, *, seed: int = 7) -> None:
    """Replace every zero-initialised parameter with real values.

    The skeleton defaults leave ``gate.weight``, ``tid2eid``, ``attn_sink``, the
    hyper-connection mixers and the compressor's ``ape`` at zero, which collapses
    routing onto expert 0 and makes the hyper-connection mixes constant. A
    forward over those proves much less than one over real values -- the same
    reason increment 2's forward gate randomised first.
    """

    from mlx.utils import tree_flatten, tree_unflatten

    mx.random.seed(seed)
    updated = []
    for name, value in tree_flatten(model.parameters()):
        if value.dtype in (mx.int64, mx.int32, mx.uint32):
            updated.append(
                (name, mx.random.randint(0, 8, shape=value.shape).astype(value.dtype))
            )
        else:
            updated.append(
                (
                    name,
                    (
                        0.05 * mx.random.normal(shape=value.shape, dtype=mx.float32)
                    ).astype(value.dtype),
                )
            )
    model.update(tree_unflatten(updated))
    mx.eval(model.parameters())


def _tiny_experts(args: DeepseekV4FlashVQModelArgs, *, seed: int = 11):
    """One mxfp4 expert stack per layer, quantised from random bf16 weights."""

    mx.random.seed(seed)
    modules = {}
    for layer in range(args.num_hidden_layers):
        linears = {}
        for name, (out_dims, in_dims) in {
            "gate_proj": (args.moe_intermediate_size, args.hidden_size),
            "up_proj": (args.moe_intermediate_size, args.hidden_size),
            "down_proj": (args.hidden_size, args.moe_intermediate_size),
        }.items():
            dense = 0.1 * mx.random.normal(
                shape=(args.n_routed_experts, out_dims, in_dims)
            )
            codes, scales = mx.quantize(dense, group_size=32, bits=4, mode="mxfp4")
            linears[name] = _MXFP4SwitchLinear(codes, scales)
        modules[layer] = _StatsSwitchGLU(
            gate_proj=linears["gate_proj"],
            up_proj=linears["up_proj"],
            down_proj=linears["down_proj"],
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
    mx.eval([module.parameters() for module in modules.values()])
    return modules


def _tiny_model(args=None, *, seed: int = 7):
    args = args or _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    _randomise(model, seed=seed)
    return args, model


def _mtp_args(**overrides) -> DeepseekV4FlashVQModelArgs:
    """``_tiny_args`` plus a two-stage DSpark drafter over layers [2, 3]."""

    base = {
        # Four backbone ratios then a drafter tail of zeros, like the released
        # 46-for-43 list.
        "compress_ratios": [0, 4, 128, 4, 0, 0],
        "dspark_block_size": 4,
        "dspark_noise_token_id": 5,
        "dspark_target_layer_ids": [2, 3],
        "dspark_markov_rank": 8,
    }
    base.update(overrides)
    return _tiny_args(**base)


def _tiny_drafter_model(args=None, *, seed: int = 7):
    """A model whose drafter can actually run: dense fp32-SwiGLU routed experts.

    The drafter's routed experts stay resident (it is re-entered once per
    supervised position), so they are bound here rather than handed to the
    streaming provider, which is exactly what ``load_dsv4_streaming_workload(
    with_mtp=True)`` does against the real checkpoint.
    """

    from mlx_lm.models.switch_layers import SwitchGLU

    args = args or _mtp_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=True)
    _randomise(model, seed=seed)
    mx.random.seed(seed + 1)
    for block in model.mtp_drafter.blocks:
        switch = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit, fp32=True),
        )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(switch, projection)
            linear.weight = 0.1 * mx.random.normal(shape=linear.weight.shape)
        block.ffn.switch_mlp = switch
    mx.eval(model.parameters())
    return args, model


def _make_cache(model):
    args = model.args
    caches = []
    for layer in model.model.layers:
        ratio = layer.attn.compress_ratio
        if ratio == 0:
            caches.append(RotatingKVCache(max_size=args.sliding_window))
        elif isinstance(layer.attn, SparseCompressedAttention):
            caches.append(
                CacheList(
                    RotatingKVCache(max_size=args.sliding_window),
                    PoolingCache(ratio),
                    PoolingCache(ratio),
                )
            )
        else:
            caches.append(
                CacheList(
                    RotatingKVCache(max_size=args.sliding_window), PoolingCache(ratio)
                )
            )
    return caches


def _chunk_major_prefill(model, experts, token_ids, chunk):
    """The reference: chunk outer, layer inner -- what the bench measured."""

    args = model.args
    for index, layer in enumerate(model.model.layers):
        layer.ffn.switch_mlp = experts[index]
        layer.ffn.switch_mlp.down_hook = None
    cache = _make_cache(model)
    outputs = []
    for start in range(0, token_ids.shape[1], chunk):
        piece = token_ids[:, start : start + chunk]
        h = model.model.embed_tokens(piece)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :], (h.shape[0], h.shape[1], args.hc_mult, h.shape[2])
            )
        )
        first = cache[0]
        mask = create_attention_mask(
            h[:, :, 0, :],
            first[0] if isinstance(first, CacheList) else first,
            window_size=args.sliding_window,
            return_array=True,
        )
        for layer, layer_cache in zip(model.model.layers, cache):
            h = layer(h, mask, layer_cache, piece)
        mx.eval(h)
        outputs.append(h)
    for layer in model.model.layers:
        layer.ffn.switch_mlp = None
    return outputs


def _pack_payload(sessions, *, record_type: str = DSV4_PACK_RECORD_TYPE) -> dict:
    rows = []
    for prompt_id, split, ids, positions in sessions:
        canonical = json.dumps(
            list(ids), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        rows.append(
            {
                "prompt_id": prompt_id,
                "campaign_split": split,
                "encoded_token_ids": list(ids),
                "token_count": len(ids),
                "token_ids_sha256": hashlib.sha256(canonical).hexdigest(),
                "positions": list(positions),
                "target_token_ids": [int(ids[p + 1]) for p in positions],
            }
        )
    return {
        "record_type": record_type,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "prompt_row_count": len(rows),
        "prompt_rows": rows,
    }


def _write_tiny_pack(path: Path, *, vocab: int = 512, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    sessions = []
    for index, length in enumerate((48, 96)):
        ids = rng.integers(0, vocab, size=length).tolist()
        positions = list(range(length // 2, length - 1, 3))
        sessions.append((f"tiny_session_{index}", "calibration", ids, positions))
    path.write_text(json.dumps(_pack_payload(sessions)))
    return path


def _tiny_loader(args, model, experts, *, identity_overrides=None):
    def loader(*, checkpoint_dir=None, io_threads=8, nocache=True, stats=None, **_kw):
        provider = InMemoryExpertProvider(experts, stats=stats)
        identities = {
            "checkpoint_dir": str(checkpoint_dir),
            "synthetic": True,
            "num_hidden_layers": args.num_hidden_layers,
            "n_routed_experts": args.n_routed_experts,
            "expert_mode": "native_mxfp4",
            **(identity_overrides or {}),
        }
        return model, provider, identities

    return loader


def _rewrite_as_legacy_generation(out: Path) -> str:
    manifest_path = out / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    current = manifest["generation_config"]
    keys = (
        "mode",
        "prefill_chunk_tokens",
        "batch_size",
        "top_k",
        "lm_head_slice",
        "io_threads",
        "nocache",
        "compress_outputs",
        "imatrix_spaces",
        "mtp_draft_width",
        "dspark_target_layer_ids",
        "expert_mode",
        "resident_dtype",
        "num_hidden_layers",
        "teacher_window_tokens",
    )
    legacy = {
        "record_type": "dsv4_teacher_generation_config_v1",
        **{key: current[key] for key in keys},
    }
    generation_sha = hashlib.sha256(
        json.dumps(
            legacy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    manifest["generation_config"] = legacy
    manifest["generation_config_sha256"] = generation_sha
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in (out / "sessions").glob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            capture = {name: archive[name] for name in archive.files}
        capture["generation_config_sha256"] = np.asarray(generation_sha)
        np.savez(path, **capture)
    return generation_sha


# ---------------------------------------------------------------------------
# Pack loading
# ---------------------------------------------------------------------------


class TestPackLoading:
    def test_loads_and_orders_shortest_first(self, tmp_path):
        pack = _write_tiny_pack(tmp_path / "pack.json")
        sessions = load_dsv4_teich_pack(pack)
        assert [s.prompt_id for s in sessions] == ["tiny_session_0", "tiny_session_1"]
        assert [s.token_count for s in sessions] == [48, 96]
        assert all(isinstance(s, Dsv4Session) for s in sessions)
        assert sessions[0].campaign_split == "calibration"

    def test_filters_by_split_and_prompt_id(self, tmp_path):
        payload = _pack_payload(
            [
                ("a", "calibration", list(range(20)), [10, 12]),
                ("b", "holdout", list(range(20)), [10, 12]),
            ]
        )
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))
        assert [
            s.prompt_id for s in load_dsv4_teich_pack(pack, splits=["holdout"])
        ] == ["b"]
        assert [s.prompt_id for s in load_dsv4_teich_pack(pack, prompt_ids=["a"])] == [
            "a"
        ]
        with pytest.raises(ValueError, match="unknown prompt ids"):
            load_dsv4_teich_pack(pack, prompt_ids=["nope"])

    def test_does_not_validate_an_unselected_holdout_row(self, tmp_path):
        payload = _pack_payload(
            [
                ("train", "mtp-train", list(range(20)), [10, 12]),
                ("sealed", "holdout", list(range(20)), [10, 12]),
            ]
        )
        payload["prompt_rows"][1]["token_ids_sha256"] = "sealed-not-read"
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))

        sessions = load_dsv4_teich_pack(pack, splits=["mtp-train"])

        assert [session.prompt_id for session in sessions] == ["train"]

    def test_rejects_the_wrong_model_id(self, tmp_path):
        payload = _pack_payload([("a", "mtp-train", list(range(20)), [10, 12])])
        payload["model_id"] = "other/model"
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="model_id"):
            load_dsv4_teich_pack(pack)

    def test_rejects_wrong_record_type(self, tmp_path):
        pack = tmp_path / "pack.json"
        pack.write_text(
            json.dumps(
                _pack_payload(
                    [("a", "calibration", list(range(20)), [5])], record_type="other"
                )
            )
        )
        with pytest.raises(ValueError, match="record_type"):
            load_dsv4_teich_pack(pack)

    def test_rejects_misaligned_targets(self, tmp_path):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5, 8])])
        payload["prompt_rows"][0]["target_token_ids"] = [99, 99]
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=r"target_token_ids do not equal"):
            load_dsv4_teich_pack(pack)

    def test_rejects_bad_token_sha(self, tmp_path):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5])])
        payload["prompt_rows"][0]["token_ids_sha256"] = "0" * 64
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="token_ids_sha256 mismatch"):
            load_dsv4_teich_pack(pack)

    @pytest.mark.parametrize("bad", ("1", True, 1.5))
    def test_rejects_coercible_token_ids(self, tmp_path, bad):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5])])
        row = payload["prompt_rows"][0]
        row["encoded_token_ids"][1] = bad
        row["token_ids_sha256"] = hashlib.sha256(
            json.dumps(
                row["encoded_token_ids"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="encoded_token_ids.*JSON integers"):
            load_dsv4_teich_pack(pack)

    @pytest.mark.parametrize(
        ("field", "bad"),
        (
            ("positions", ["5"]),
            ("positions", [5.5]),
            ("target_token_ids", ["6"]),
            ("target_token_ids", [6.5]),
        ),
    )
    def test_rejects_coercible_supervision_integers(self, tmp_path, field, bad):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5])])
        payload["prompt_rows"][0][field] = bad
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match=f"{field}.*JSON integers"):
            load_dsv4_teich_pack(pack)

    @pytest.mark.parametrize(
        ("field", "bad"), (("prompt_row_count", True), ("token_count", 20.0))
    )
    def test_rejects_non_integer_counts(self, tmp_path, field, bad):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5])])
        if field == "prompt_row_count":
            payload[field] = bad
        else:
            payload["prompt_rows"][0][field] = bad
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match=f"{field}.*JSON integer"):
            load_dsv4_teich_pack(pack)

    @pytest.mark.parametrize("bad", (-1, 129_280))
    def test_rejects_token_ids_outside_the_pinned_vocabulary(self, tmp_path, bad):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5])])
        row = payload["prompt_rows"][0]
        row["encoded_token_ids"][1] = bad
        row["token_ids_sha256"] = hashlib.sha256(
            json.dumps(
                row["encoded_token_ids"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="encoded_token_ids.*vocabulary"):
            load_dsv4_teich_pack(pack)

    def test_rejects_position_at_last_token(self, tmp_path):
        payload = _pack_payload([("a", "calibration", list(range(20)), [5])])
        payload["prompt_rows"][0]["positions"] = [19]
        payload["prompt_rows"][0]["target_token_ids"] = [0]
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="out of predictor range"):
            load_dsv4_teich_pack(pack)

    def test_rejects_session_past_the_teacher_window(self, tmp_path):
        payload = _pack_payload([("a", "calibration", list(range(40)), [5])])
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="teacher window"):
            load_dsv4_teich_pack(pack, teacher_window=20)

    def test_rejects_duplicate_prompt_ids(self, tmp_path):
        payload = _pack_payload(
            [
                ("a", "calibration", list(range(20)), [5]),
                ("a", "calibration", list(range(20)), [6]),
            ]
        )
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="unique"):
            load_dsv4_teich_pack(pack)

    @requires_pack
    def test_real_pack_calibration_split_matches_the_manifest(self):
        sessions = load_dsv4_teich_pack(REAL_PACK, splits=["calibration"])
        manifest = json.loads(
            (ROOT / "recipes/dsv4_teich_split_manifest_v1_20260811.json").read_text()
        )["splits"]["calibration"]
        assert len(sessions) == manifest["sessions"] == 40
        assert sum(s.token_count for s in sessions) == manifest["v4_raw_tokens"]
        assert (
            sum(s.supervised_count for s in sessions)
            == manifest["v4_supervised_tokens"]
        )
        assert (
            max(s.token_count for s in sessions)
            <= DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS
        )


# ---------------------------------------------------------------------------
# The property the whole design rests on
# ---------------------------------------------------------------------------


class TestLayerMajorEquivalence:
    @pytest.mark.parametrize("chunk", (8, 16))
    def test_layer_major_is_bit_identical_to_chunk_major(self, chunk):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        ids = mx.array(np.random.default_rng(5).integers(0, args.vocab_size, (1, 48)))

        reference = _chunk_major_prefill(model, experts, ids, chunk)
        streamed = layer_major_prefill(
            model,
            ids,
            chunk=chunk,
            expert_provider=InMemoryExpertProvider(experts),
            release_after_layer=False,
        )

        assert len(streamed) == len(reference)
        for got, want in zip(streamed, reference):
            assert got.shape == want.shape
            assert mx.array_equal(got, want), "layer-major diverged from chunk-major"

    def test_prefetch_runs_one_layer_ahead(self):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        provider = InMemoryExpertProvider(experts)
        ids = mx.array(np.random.default_rng(5).integers(0, args.vocab_size, (1, 32)))
        layer_major_prefill(model, ids, chunk=16, expert_provider=provider)
        # Layer 0 is fetched up front, then each layer prefetches its successor.
        assert provider.prefetched == list(range(args.num_hidden_layers))

    def test_experts_are_unbound_after_each_layer(self):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        ids = mx.array(np.random.default_rng(5).integers(0, args.vocab_size, (1, 32)))
        layer_major_prefill(
            model, ids, chunk=16, expert_provider=InMemoryExpertProvider(experts)
        )
        assert all(layer.ffn.switch_mlp is None for layer in model.model.layers)


class TestStatsCollectionIsTransparent:
    def test_stats_switch_glu_matches_upstream(self):
        """The subclass computes ``SwitchGLU.__call__``, hook or no hook."""

        args = _tiny_args()
        mx.random.seed(2)
        upstream = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        tapped = _StatsSwitchGLU(
            gate_proj=upstream.gate_proj,
            up_proj=upstream.up_proj,
            down_proj=upstream.down_proj,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        for tokens in (4, 64):  # straddle SwitchGLU's do_sort >= 64 threshold
            x = mx.random.normal(shape=(1, tokens, args.hidden_size))
            indices = mx.random.randint(
                0, args.n_routed_experts, shape=(1, tokens, args.num_experts_per_tok)
            )
            want = upstream(x, indices)
            tapped.down_hook = None
            assert mx.array_equal(tapped(x, indices), want)

            seen: list = []
            tapped.down_hook = lambda act, ids, order, _seen=seen: _seen.append(
                (act, ids, order)
            )
            assert mx.array_equal(tapped(x, indices), want)
            assert len(seen) == 1
            activated, expert_ids, order = seen[0]
            assert activated.shape[-1] == args.moe_intermediate_size
            assert int(expert_ids.size) == int(indices.size)
            expected_sorted = tokens * args.num_experts_per_tok >= 64
            assert (order is not None) == expected_sorted

    def test_reported_permutation_aligns_ids_with_the_unsorted_routes(self):
        args = _tiny_args()
        mx.random.seed(4)
        upstream = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        tapped = _StatsSwitchGLU(
            gate_proj=upstream.gate_proj,
            up_proj=upstream.up_proj,
            down_proj=upstream.down_proj,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        tokens = 64
        x = mx.random.normal(shape=(1, tokens, args.hidden_size))
        indices = mx.random.randint(
            0, args.n_routed_experts, shape=(1, tokens, args.num_experts_per_tok)
        )
        seen: list = []
        tapped.down_hook = lambda act, ids, order: seen.append((act, ids, order))
        tapped(x, indices)
        _activated, expert_ids, order = seen[0]
        flat = np.asarray(indices).reshape(-1)
        assert order is not None
        assert np.array_equal(
            np.asarray(expert_ids).reshape(-1), flat[np.asarray(order)]
        )

    @pytest.mark.parametrize("chunk", (8, 16))
    def test_collecting_stats_does_not_change_the_forward(self, chunk):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        ids = mx.array(np.random.default_rng(5).integers(0, args.vocab_size, (1, 48)))

        plain = layer_major_prefill(
            model,
            ids,
            chunk=chunk,
            expert_provider=InMemoryExpertProvider(experts),
            release_after_layer=False,
        )
        accumulator = _ImatrixAccumulator(
            layers=range(args.num_hidden_layers),
            num_experts=args.n_routed_experts,
            dims={"hidden": args.hidden_size, "down": args.moe_intermediate_size},
        )
        collected = layer_major_prefill(
            model,
            ids,
            chunk=chunk,
            expert_provider=InMemoryExpertProvider(experts),
            accumulator=accumulator,
            collect_spaces=IMATRIX_SPACES,
            release_after_layer=False,
        )
        for got, want in zip(collected, plain):
            assert mx.array_equal(got, want), "the stats path changed the forward"


class TestImatrixAccumulation:
    def test_matches_the_host_reference(self):
        """The device scatter-add equals ``keep.quality.imatrix``'s NumPy.

        Tolerance is fp32-grade (2e-6), not merely "close": the one-hot matmul
        formulation this replaced agreed to only 5.9e-4 and one-sidedly, which
        is the whole reason the accumulator scatters instead.
        """

        from keep.quality.dsv4_teacher_runner import routed_slot_columns
        from keep.quality.imatrix import accumulate_routed_projection_imatrix

        rng = np.random.default_rng(9)
        tokens, top_k, dims, experts = 40, 3, 16, 6
        rows = rng.normal(size=(tokens, dims)).astype(np.float32)
        indices = rng.integers(0, experts, size=(tokens, top_k))
        scores = rng.uniform(0.1, 1.0, size=(tokens, top_k)).astype(np.float32)

        accumulator = _ImatrixAccumulator(
            layers=[0], num_experts=experts, dims={"hidden": dims}
        )
        rows_mx = mx.array(rows)
        accumulator.add_chunk(
            layer=0,
            space="hidden",
            squared_rows=rows_mx * rows_mx,
            slot_columns=routed_slot_columns(
                mx.array(indices), mx.array(scores), per_row=False
            ),
        )
        arrays = accumulator.as_arrays()

        reference = accumulate_routed_projection_imatrix(
            layer=0,
            projection="gate_proj",
            inputs=rows,
            route_indices=indices,
            router_scores=scores,
            num_experts=experts,
        )
        for entry in reference:
            np.testing.assert_allclose(
                arrays["importance_sum__hidden"][0, entry.expert],
                entry.importance_sum,
                rtol=2e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                arrays["affinity_weighted_importance__hidden"][0, entry.expert],
                entry.affinity_weighted_importance,
                rtol=2e-6,
                atol=1e-6,
            )
            assert arrays["route_count__hidden"][0, entry.expert] == entry.route_count
            assert arrays["total_route_count__hidden"][0] == entry.total_route_count

    def test_down_space_columns_are_one_row_per_routed_slot(self):
        from keep.quality.dsv4_teacher_runner import routed_slot_columns

        indices = mx.array(np.arange(12).reshape(4, 3) % 5)
        scores = mx.array(np.linspace(0.1, 1.0, 12).reshape(4, 3).astype(np.float32))
        wide = routed_slot_columns(indices, scores, per_row=False)
        assert len(wide) == 3
        assert all(int(ids.size) == 4 for ids, _ in wide)
        flat = routed_slot_columns(
            indices.reshape(-1), scores.reshape(-1), per_row=True
        )
        assert len(flat) == 1
        assert int(flat[0][0].size) == 12


# ---------------------------------------------------------------------------
# Modes, schema, resumability
# ---------------------------------------------------------------------------


class TestRunModes:
    def test_mtp_targets_needs_a_model_with_a_drafter(self, tmp_path):
        """No longer "not implemented" -- but still refused without a drafter."""

        assert "mtp-targets" in DSV4_TEACHER_MODES
        args, model = _tiny_model()  # with_mtp=False
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        with pytest.raises(MtpDrafterUnavailable, match="with_mtp=True"):
            run_dsv4_teacher_production(
                mode="mtp-targets",
                pack_path=pack,
                out_dir=tmp_path / "out",
                chunk=16,
                workload_loader=_tiny_loader(args, model, experts),
            )

    def test_unknown_mode_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mode must be one of"):
            run_dsv4_teacher_production(
                mode="hidden-states", pack_path=tmp_path / "x.json", out_dir=tmp_path
            )

    def test_calibration_writes_the_expected_schema(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"

        summary = run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=_tiny_loader(args, model, experts),
        )
        assert summary["sessions_produced"] == 2
        assert summary["sessions_skipped"] == 0
        assert summary["prefill_chunk_tokens"] == 16
        assert summary["tokens_per_s"] > 0

        manifest = json.loads((out / "run-manifest.json").read_text())
        assert manifest["record_type"] == "dsv4_teacher_capture_run"
        assert manifest["generation_config"]["prefill_chunk_tokens"] == 16
        assert manifest["generation_config"]["batch_size"] == 1
        assert manifest["generation_config"]["expert_mode"] == "native_mxfp4"

        path = session_output_path(out, "tiny_session_0")
        with np.load(path, allow_pickle=False) as archive:
            capture = {name: archive[name] for name in archive.files}
        validate_dsv4_calibration_capture(
            capture,
            layers=range(args.num_hidden_layers),
            num_experts=args.n_routed_experts,
        )
        assert int(capture["prefill_chunk_tokens"]) == 16
        assert str(capture["record_type"]) == "dsv4_teacher_calibration_v1"
        for space, dim in (
            ("hidden", args.hidden_size),
            ("down", args.moe_intermediate_size),
        ):
            block = capture[f"importance_sum__{space}"]
            assert block.shape == (args.num_hidden_layers, args.n_routed_experts, dim)
            assert np.all(np.isfinite(block))
            assert block.sum() > 0, f"{space} statistics are all zero"
            counts = capture[f"route_count__{space}"]
            assert counts.sum() == capture[f"total_route_count__{space}"].sum()

    def test_logits_writes_the_expected_schema(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"

        run_dsv4_teacher_production(
            mode="logits",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            top_k=32,
            lm_head_slice=8,
            workload_loader=_tiny_loader(args, model, experts),
        )
        sessions = load_dsv4_teich_pack(pack)
        for session in sessions:
            path = session_output_path(out, session.prompt_id)
            with np.load(path, allow_pickle=False) as archive:
                capture = {name: archive[name] for name in archive.files}
            validate_dsv4_logits_capture(
                capture, top_k=32, supervised=session.supervised_count
            )
            assert np.array_equal(
                capture["positions"], session.positions.astype(np.int32)
            )
            assert np.array_equal(
                capture["target_token_ids"], session.target_token_ids.astype(np.int32)
            )
            assert int(capture["prefill_chunk_tokens"]) == 16

    def test_mtp_targets_writes_the_expected_schema(self, tmp_path):
        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"

        summary = run_dsv4_teacher_production(
            mode="mtp-targets",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            top_k=16,
            workload_loader=_tiny_loader(args, model, experts),
        )
        assert summary["sessions_produced"] == 2

        manifest = json.loads((out / "run-manifest.json").read_text())
        config = manifest["generation_config"]
        assert config["mode"] == "mtp-targets"
        assert config["mtp_draft_width"] == args.dspark_block_size
        assert config["dspark_target_layer_ids"] == [2, 3]
        assert config["top_k"] == 16
        assert config["lm_head_slice"] == 2048
        assert config["io_threads"] == 8
        assert config["nocache"] is True
        assert config["compress_outputs"] is False
        # The chunk size and the config sha travel with the data, as in every
        # other mode -- chunked prefill is not bit-identical across chunks.
        assert config["prefill_chunk_tokens"] == 16

        width = args.dspark_block_size
        for session in load_dsv4_teich_pack(pack):
            path = session_output_path(out, session.prompt_id)
            with np.load(path, allow_pickle=False) as archive:
                capture = {name: archive[name] for name in archive.files}
            validate_dsv4_mtp_targets_capture(
                capture,
                top_k=16,
                supervised=session.supervised_count,
                width=width,
                hidden_size=args.hidden_size,
            )
            assert str(capture["record_type"]) == "dsv4_teacher_mtp_targets_v1"
            assert int(capture["prefill_chunk_tokens"]) == 16
            assert (
                str(capture["generation_config_sha256"])
                == (manifest["generation_config_sha256"])
            )
            assert np.array_equal(
                capture["positions"], session.positions.astype(np.int32)
            )
            # Slot 0 is the same supervision the logits mode carries...
            assert np.array_equal(
                capture["mtp_target_token_ids"][:, 0],
                session.target_token_ids.astype(np.int32),
            )
            # ...and the later slots are the tokens that actually follow.
            ids = session.token_ids.astype(np.int64)
            for row, position in enumerate(session.positions.tolist()):
                for slot in range(width):
                    wanted = position + 1 + slot
                    if wanted < ids.size:
                        assert capture["mtp_target_valid"][row, slot]
                        assert capture["mtp_target_token_ids"][row, slot] == ids[wanted]
                    else:
                        assert not capture["mtp_target_valid"][row, slot]
                        assert capture["mtp_target_token_ids"][row, slot] == -1

    def test_mtp_targets_capture_matches_the_reference_calibration_forward(
        self, tmp_path
    ):
        """The reference is *the reference*, sliced by nobody but itself.

        The runner appends context incrementally as it walks ascending
        supervised positions -- one pass over the session rather than one per
        position. Two separate things have to hold for that to be sound, and
        this asserts both against
        ``DeepseekV4FlashVQModel.dspark_calibration_forward``, which is a 1:1
        transcription of ``deepseek_v4_model.py:540-554``:

        * the incremental ring lands where a from-scratch run would, and
        * it lands at the *right* offset -- context through ``p-1``, anchor at
          ``p``.

        The earlier version of this test hand-sliced ``fused[:, :position + 1]``
        into ``dspark_forward``, which baked the runner's own off-by-one into
        the "reference" and could not see the seam. Handing the reference a
        whole prefix and letting *it* do the slicing is what gives this teeth.
        """

        from keep.quality.dsv4_teacher_runner import (
            _fuse_dspark_taps,
            layer_major_prefill,
            mtp_target_capture,
        )

        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        session = load_dsv4_teich_pack(pack)[0]
        ids = mx.array(session.token_ids.astype(np.int64)[None, :])

        taps: dict[int, list] = {}
        layer_major_prefill(
            model,
            ids,
            chunk=16,
            expert_provider=InMemoryExpertProvider(experts),
            dspark_taps=taps,
        )
        assert sorted(taps) == [2, 3]
        capture = mtp_target_capture(model, taps, session, top_k=16)

        fused = _fuse_dspark_taps(model, taps)
        width = args.dspark_block_size
        for row, position in enumerate(session.positions.tolist()):
            # The whole prefix, ids and taps together. dspark_calibration_forward
            # decides what the context is and what the anchor is.
            reference, hidden = model.dspark_calibration_forward(
                fused[:, : position + 1],
                ids[:, : position + 1],
                draft_length=width,
            )
            mx.eval(reference, hidden)
            order = np.array(mx.argsort(-reference[0].astype(mx.float32), axis=1))
            assert np.array_equal(
                capture["mtp_topk_logit_ids"][row], order[:, :16].astype(np.int32)
            )
            assert np.array_equal(
                capture["mtp_final_hidden"][row],
                np.array(hidden[0].astype(mx.float16)),
            )

    def test_mtp_targets_context_excludes_the_anchors_own_tap(self, tmp_path):
        """The seam, asserted directly on the ring offset rather than inferred.

        After drafting at supervised position ``p`` the context ring must hold
        exactly ``p`` committed tokens -- taps ``0 .. p-1``. One more would mean
        the drafter saw the backbone hidden state of the token it is being asked
        to condition on, which it never has at inference (``dspark.py:204-209``:
        ``take_primed`` refuses unless the ring is one token behind the target
        cache), and would shift every draft slot's RoPE by one.

        Also asserted numerically: a run that *does* commit the anchor's tap
        produces different logits, so the offset above is load-bearing and not
        cosmetic.
        """

        from keep.quality.dsv4_teacher_runner import (
            _fuse_dspark_taps,
            layer_major_prefill,
            mtp_target_capture,
        )

        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        session = load_dsv4_teich_pack(pack)[0]
        ids = mx.array(session.token_ids.astype(np.int64)[None, :])

        taps: dict[int, list] = {}
        layer_major_prefill(
            model,
            ids,
            chunk=16,
            expert_provider=InMemoryExpertProvider(experts),
            dspark_taps=taps,
        )
        fused = _fuse_dspark_taps(model, taps)
        last = int(session.positions[-1])

        correct = mtp_target_capture(model, taps, session, top_k=16)

        # The ring the reference convention produces for the last position.
        reference_cache = model.make_mtp_cache()
        model.dspark_append_context(fused[:, :last], reference_cache)
        assert [entry.offset for entry in reference_cache] == [last] * len(
            reference_cache
        )

        # The off-by-one the review caught: one extra committed token.
        skewed_cache = model.make_mtp_cache()
        model.dspark_append_context(fused[:, : last + 1], skewed_cache)
        assert [entry.offset for entry in skewed_cache] == [last + 1] * len(
            skewed_cache
        )

        anchor = ids[:, last : last + 1]
        good, _ = model.dspark_draft_block(
            anchor, reference_cache, draft_length=args.dspark_block_size
        )
        skewed, _ = model.dspark_draft_block(
            anchor, skewed_cache, draft_length=args.dspark_block_size
        )
        mx.eval(good, skewed)

        # The capture took the correct branch...
        order = np.array(mx.argsort(-good[0].astype(mx.float32), axis=1))
        assert np.array_equal(
            correct["mtp_topk_logit_ids"][-1], order[:, :16].astype(np.int32)
        )
        # ...and the two branches are genuinely different functions, so that
        # equality is a statement about the seam and not a tautology.
        assert float(mx.max(mx.abs(good - skewed))) > 1e-4

    def test_mtp_targets_hidden_and_logits_are_a_usable_recovery_pair(self, tmp_path):
        """``mtp_final_hidden`` must be the head's *input*, not its output.

        Recovery training regresses ``norm`` + ``lm_head`` from the stored
        hidden onto the stored logits, so pushing the stored hidden through the
        real head has to reproduce the stored top-K. Storing the wrong hidden
        (the pre-``hc_head`` 4D stream, say) would still be finite and the right
        shape -- and useless.
        """

        from keep.quality.dsv4_teacher_runner import (
            layer_major_prefill,
            mtp_target_capture,
        )

        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        session = load_dsv4_teich_pack(pack)[0]
        ids = mx.array(session.token_ids.astype(np.int64)[None, :])

        taps: dict[int, list] = {}
        layer_major_prefill(
            model,
            ids,
            chunk=16,
            expert_provider=InMemoryExpertProvider(experts),
            dspark_taps=taps,
        )
        capture = mtp_target_capture(model, taps, session, top_k=16)

        stored = mx.array(capture["mtp_final_hidden"][0].astype(np.float32))
        head = model.mtp_drafter.blocks[-1]
        replayed = np.array(model.lm_head(head.norm(stored)).astype(mx.float32))

        # Asserted on the logit *values* at the stored ids, not on rank order:
        # the stored hidden is fp16 and the stored logits came from the fp32
        # one, so two entries a hair apart can legitimately swap places deep in
        # the top-K. Values are the recovery-training signal; their order past
        # the first few is not.
        gathered = np.take_along_axis(
            replayed, capture["mtp_topk_logit_ids"][0].astype(np.int64), axis=1
        )
        stored_values = capture["mtp_topk_logit_values"][0].astype(np.float32)
        scale = np.maximum(np.abs(stored_values), 1.0)
        replay_gap = float(np.max(np.abs(gathered - stored_values)))
        assert np.all(np.abs(gathered - stored_values) < 2e-2 * scale)
        # The argmax has a real margin, so it must survive exactly.
        assert np.array_equal(
            replayed.argmax(axis=1).astype(np.int32),
            capture["mtp_topk_logit_ids"][0][:, 0],
        )
        # Teeth: a *different* hidden does not reproduce these logits. Compared
        # against the replay gap rather than an absolute epsilon, because this
        # fixture's logits are O(0.05) and any constant would be arbitrary.
        wrong = np.array(model.lm_head(head.norm(stored[::-1])).astype(mx.float32))
        wrong_gathered = np.take_along_axis(
            wrong, capture["mtp_topk_logit_ids"][0].astype(np.int64), axis=1
        )
        wrong_gap = float(np.max(np.abs(wrong_gathered - stored_values)))
        assert wrong_gap > 10 * replay_gap, (wrong_gap, replay_gap)

    def test_mtp_draft_width_is_capped_and_scales_the_artifact(self, tmp_path):
        from keep.quality.dsv4_teacher_runner import (
            layer_major_prefill,
            mtp_target_capture,
        )

        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        session = load_dsv4_teich_pack(pack)[0]
        ids = mx.array(session.token_ids.astype(np.int64)[None, :])

        taps: dict[int, list] = {}
        layer_major_prefill(
            model,
            ids,
            chunk=16,
            expert_provider=InMemoryExpertProvider(experts),
            dspark_taps=taps,
        )
        narrow = mtp_target_capture(model, taps, session, top_k=8, draft_width=2)
        assert narrow["mtp_topk_logit_ids"].shape[1] == 2
        assert int(narrow["mtp_draft_width"]) == 2

        # Asking past dspark_block_size is capped, not honoured.
        capped = mtp_target_capture(model, taps, session, top_k=8, draft_width=99)
        assert int(capped["mtp_draft_width"]) == args.dspark_block_size

    def test_mtp_targets_resumes_only_after_re_validating_the_file(self, tmp_path):
        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)

        first = run_dsv4_teacher_production(
            mode="mtp-targets",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            top_k=16,
            workload_loader=loader,
        )
        assert first["sessions_produced"] == 2

        again = run_dsv4_teacher_production(
            mode="mtp-targets",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            top_k=16,
            workload_loader=loader,
        )
        assert again["sessions_produced"] == 0
        assert again["sessions_skipped"] == 2

        # A file written at a different chunk is refused, not silently trusted.
        with pytest.raises(ValueError, match="prefill chunk"):
            run_dsv4_teacher_production(
                mode="mtp-targets",
                pack_path=pack,
                out_dir=out,
                chunk=32,
                top_k=16,
                workload_loader=loader,
            )

        # I/O policy is part of the production identity too: a resumed run may
        # not relabel existing outputs as if they were generated under it.
        with pytest.raises(ValueError, match="generation identity"):
            run_dsv4_teacher_production(
                mode="mtp-targets",
                pack_path=pack,
                out_dir=out,
                chunk=16,
                top_k=16,
                nocache=False,
                workload_loader=loader,
            )

    def test_dspark_tap_capture_is_opt_in_in_the_layer_major_prefill(self, tmp_path):
        """The default prefill returns exactly what it returned before."""

        from keep.quality.dsv4_teacher_runner import layer_major_prefill

        args, model = _tiny_drafter_model()
        experts = _tiny_experts(args)
        ids = mx.array(
            np.random.default_rng(2).integers(0, args.vocab_size, size=(1, 48))
        )

        plain = layer_major_prefill(
            model, ids, chunk=16, expert_provider=InMemoryExpertProvider(experts)
        )
        taps: dict[int, list] = {}
        tapped = layer_major_prefill(
            model,
            ids,
            chunk=16,
            expert_provider=InMemoryExpertProvider(experts),
            dspark_taps=taps,
        )
        mx.eval(plain, tapped)

        assert len(plain) == len(tapped) == 3
        for left, right in zip(plain, tapped):
            assert bool(mx.array_equal(left, right))
        # Taps only when asked for, and only for the target layers.
        assert sorted(taps) == list(args.dspark_target_layer_ids)
        assert all(len(chunks) == 3 for chunks in taps.values())

    def test_tap_capture_without_target_layer_ids_is_refused(self, tmp_path):
        from keep.quality.dsv4_teacher_runner import layer_major_prefill

        args, model = _tiny_model()  # no dspark_target_layer_ids
        experts = _tiny_experts(args)
        ids = mx.array(np.zeros((1, 16), dtype=np.int64))
        with pytest.raises(ValueError, match="dspark_target_layer_ids"):
            layer_major_prefill(
                model,
                ids,
                chunk=16,
                expert_provider=InMemoryExpertProvider(experts),
                dspark_taps={},
            )

    def test_logits_match_a_direct_lm_head_projection(self, tmp_path):
        """The sliced supervised-position head equals projecting the whole run."""

        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        session = load_dsv4_teich_pack(pack)[0]
        ids = mx.array(session.token_ids.astype(np.int64)[None, :])

        hidden = layer_major_prefill(
            model,
            ids,
            chunk=16,
            expert_provider=InMemoryExpertProvider(experts),
            release_after_layer=False,
        )
        whole = mx.concatenate(hidden, axis=1)
        logits = model.lm_head(model.model.norm(model.model.hc_head(whole)))[0]
        want = np.asarray(
            mx.take(logits, mx.array(session.positions.astype(np.int32)), axis=0)
        )

        from keep.quality.dsv4_teacher_runner import _supervised_logit_capture

        capture = _supervised_logit_capture(
            model, hidden, session, chunk=16, top_k=8, lm_head_slice=4
        )
        for row in range(want.shape[0]):
            top = np.argsort(-want[row])[:8]
            assert set(capture["topk_logit_ids"][row].tolist()) == set(top.tolist())
            np.testing.assert_allclose(
                capture["topk_logit_values"][row].astype(np.float32),
                want[row][capture["topk_logit_ids"][row]],
                rtol=2e-2,
                atol=2e-2,
            )


class TestResumability:
    def test_generation_identity_binds_authenticated_source(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        authority = {
            "source_model_id": "example/model",
            "source_revision": "revision-a",
            "config_sha256": "a" * 64,
            "index_sha256": "b" * 64,
            "download_completion_sha256": "c" * 64,
        }

        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=tmp_path / "source-a",
            chunk=16,
            workload_loader=_tiny_loader(
                args, model, experts, identity_overrides=authority
            ),
        )
        authority["source_revision"] = "revision-b"
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=tmp_path / "source-b",
            chunk=16,
            workload_loader=_tiny_loader(
                args, model, experts, identity_overrides=authority
            ),
        )

        first = json.loads((tmp_path / "source-a/run-manifest.json").read_text())
        second = json.loads((tmp_path / "source-b/run-manifest.json").read_text())
        assert first["generation_config_sha256"] != second["generation_config_sha256"]

    def test_generation_identity_binds_pack_and_selection(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        loader = _tiny_loader(args, model, experts)

        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=tmp_path / "all",
            chunk=16,
            workload_loader=loader,
        )
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=tmp_path / "one",
            chunk=16,
            max_sessions=1,
            workload_loader=loader,
        )
        changed_pack = tmp_path / "changed-pack.json"
        payload = json.loads(pack.read_text())
        payload["unused_metadata"] = "changes the authenticated pack bytes"
        changed_pack.write_text(json.dumps(payload))
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=changed_pack,
            out_dir=tmp_path / "changed-pack",
            chunk=16,
            workload_loader=loader,
        )

        all_sha = json.loads((tmp_path / "all/run-manifest.json").read_text())[
            "generation_config_sha256"
        ]
        one_sha = json.loads((tmp_path / "one/run-manifest.json").read_text())[
            "generation_config_sha256"
        ]
        changed_pack_sha = json.loads(
            (tmp_path / "changed-pack/run-manifest.json").read_text()
        )["generation_config_sha256"]
        assert all_sha != one_sha
        assert all_sha != changed_pack_sha

    def test_existing_manifest_is_validated_before_replacement(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        manifest_path = out / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["pack_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="existing run manifest"):
            run_dsv4_teacher_production(
                mode="calibration",
                pack_path=pack,
                out_dir=out,
                chunk=16,
                workload_loader=loader,
            )
        assert manifest_path.read_bytes() == before

    def test_complete_legacy_cache_is_authenticated_without_manifest_rewrite(
        self, tmp_path
    ):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        _rewrite_as_legacy_generation(out)
        manifest_path = out / "run-manifest.json"
        before = (manifest_path.read_bytes(), manifest_path.stat().st_ino)

        result = run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )

        assert result["sessions_produced"] == 0
        assert result["sessions_skipped"] == 2
        assert (manifest_path.read_bytes(), manifest_path.stat().st_ino) == before

    def test_partial_legacy_cache_is_not_mixed_with_bound_generation(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        _rewrite_as_legacy_generation(out)
        next((out / "sessions").glob("*.npz")).unlink()

        with pytest.raises(ValueError, match="partial legacy cache"):
            run_dsv4_teacher_production(
                mode="calibration",
                pack_path=pack,
                out_dir=out,
                chunk=16,
                workload_loader=loader,
            )

    def test_second_run_skips_validated_sessions(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)

        first = run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        assert first["sessions_produced"] == 2
        stamps = {
            path.name: path.stat().st_mtime_ns
            for path in (out / "sessions").glob("*.npz")
        }

        second = run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        assert second["sessions_produced"] == 0
        assert second["sessions_skipped"] == 2
        assert {
            path.name: path.stat().st_mtime_ns
            for path in (out / "sessions").glob("*.npz")
        } == stamps

    def test_selection_extension_is_refused_not_mixed(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)

        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            max_sessions=1,
            workload_loader=loader,
        )
        assert len(list((out / "sessions").glob("*.npz"))) == 1

        with pytest.raises(ValueError, match="session_count disagrees"):
            run_dsv4_teacher_production(
                mode="calibration",
                pack_path=pack,
                out_dir=out,
                chunk=16,
                workload_loader=loader,
            )

    def test_a_different_prefill_chunk_is_refused_not_mixed(self, tmp_path):
        """The pinned finding, enforced: chunk sizes cannot be mixed in a cache."""

        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)

        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        with pytest.raises(ValueError, match="not bit-identical across chunk sizes"):
            run_dsv4_teacher_production(
                mode="calibration",
                pack_path=pack,
                out_dir=out,
                chunk=32,
                workload_loader=loader,
            )

    def test_a_truncated_session_file_fails_rather_than_being_skipped(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        victim = session_output_path(out, "tiny_session_0")
        raw = victim.read_bytes()
        victim.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ValueError, match="malformed"):
            run_dsv4_teacher_production(
                mode="calibration",
                pack_path=pack,
                out_dir=out,
                chunk=16,
                workload_loader=loader,
            )

    def test_a_session_file_from_another_prompt_is_refused(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        loader = _tiny_loader(args, model, experts)
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=loader,
        )
        first = session_output_path(out, "tiny_session_0")
        second = session_output_path(out, "tiny_session_1")
        first.write_bytes(second.read_bytes())
        with pytest.raises(ValueError, match="prompt_id drifted"):
            run_dsv4_teacher_production(
                mode="calibration",
                pack_path=pack,
                out_dir=out,
                chunk=16,
                workload_loader=loader,
            )


class TestAtomicWrite:
    def test_writes_at_the_exact_path_and_loads_back(self, tmp_path):
        target = tmp_path / "session.npz"
        _atomic_savez(target, positions=np.arange(4, dtype=np.int32))
        assert target.is_file()
        assert not (tmp_path / "session.tmp.npz").exists()
        with np.load(target, allow_pickle=False) as archive:
            assert np.array_equal(archive["positions"], np.arange(4, dtype=np.int32))

    def test_overwrites_an_existing_target(self, tmp_path):
        target = tmp_path / "session.npz"
        _atomic_savez(target, positions=np.arange(2, dtype=np.int32))
        _atomic_savez(target, positions=np.arange(5, dtype=np.int32))
        with np.load(target, allow_pickle=False) as archive:
            assert archive["positions"].size == 5

    def test_survives_sigkill_mid_write(self, tmp_path):
        """A ``kill -9`` during the write leaves the old file, never a half one.

        The child writes a large array through ``_atomic_savez`` and signals the
        parent once bytes are on disk; the parent kills it with ``SIGKILL`` while
        the temp file is still growing. The published path must still hold the
        previous, complete archive.
        """

        target = tmp_path / "session.npz"
        _atomic_savez(target, positions=np.arange(3, dtype=np.int32))
        good = target.read_bytes()
        ready = tmp_path / "ready"

        script = textwrap.dedent(
            f"""
            import sys, time
            from pathlib import Path
            sys.path.insert(0, {str(ROOT / "src")!r})
            import numpy as np
            from keep.quality.dsv4_teacher_runner import _atomic_savez

            target = Path({str(target)!r})
            ready = Path({str(ready)!r})
            tmp = target.with_name(target.stem + ".tmp.npz")

            import threading
            def announce():
                while not tmp.exists() or tmp.stat().st_size < 4_000_000:
                    time.sleep(0.005)
                ready.touch()
            threading.Thread(target=announce, daemon=True).start()

            # Big enough that the write is still in flight when the parent fires.
            _atomic_savez(target, positions=np.zeros(120_000_000, dtype=np.int32))
            """
        )
        script_path = tmp_path / "writer.py"
        script_path.write_text(script)
        child = subprocess.Popen([sys.executable, str(script_path)])
        try:
            deadline = time.monotonic() + 60
            while not ready.exists() and child.poll() is None:
                if time.monotonic() > deadline:
                    pytest.skip("child never reached the write; machine too slow")
                time.sleep(0.002)
            os.kill(child.pid, signal.SIGKILL)
        finally:
            child.wait(timeout=30)

        if child.returncode == 0:
            pytest.skip("child finished the 480 MB write before the kill landed")
        assert child.returncode == -signal.SIGKILL
        assert target.read_bytes() == good, "the published file was corrupted"
        with np.load(target, allow_pickle=False) as archive:
            assert archive["positions"].size == 3
        assert (
            not (tmp_path / "session.tmp.npz").exists()
            or (tmp_path / "session.tmp.npz").stat().st_size >= 0
        )  # a killed writer may leave the temp; never the target


class TestFinalize:
    def test_merges_sessions_into_imatrix_sidecars(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=_tiny_loader(args, model, experts),
        )
        report = finalize_dsv4_calibration_imatrix(out)
        assert report["session_count"] == 2
        expected = args.num_hidden_layers * 3 * args.n_routed_experts
        assert report["entry_count"] == expected

        from keep.quality.imatrix import load_projection_imatrix_manifest

        manifest = load_projection_imatrix_manifest(report["manifest_path"])
        assert manifest["entry_count"] == expected
        sidecar = Path(report["sidecar_dir"]) / manifest["entries"][0]["path"]
        assert sidecar.is_file()
        loaded = mx.load(str(sidecar))
        assert "importance_sum" in loaded
        assert "affinity_weighted_importance" in loaded

    def test_gate_and_up_share_the_hidden_space_vector(self, tmp_path):
        args, model = _tiny_model()
        experts = _tiny_experts(args)
        pack = _write_tiny_pack(tmp_path / "pack.json", vocab=args.vocab_size)
        out = tmp_path / "out"
        run_dsv4_teacher_production(
            mode="calibration",
            pack_path=pack,
            out_dir=out,
            chunk=16,
            workload_loader=_tiny_loader(args, model, experts),
        )
        report = finalize_dsv4_calibration_imatrix(out)
        root = Path(report["sidecar_dir"]) / "imatrix"
        gate = mx.load(str(root / "layer-00000-gate_proj-expert-00000.safetensors"))
        up = mx.load(str(root / "layer-00000-up_proj-expert-00000.safetensors"))
        down = mx.load(str(root / "layer-00000-down_proj-expert-00000.safetensors"))
        assert mx.array_equal(gate["importance_sum"], up["importance_sum"])
        assert down["importance_sum"].shape == (args.moe_intermediate_size,)
        assert gate["importance_sum"].shape == (args.hidden_size,)


class TestStreamStats:
    def test_device_rate_uses_busy_time_not_the_overlapped_span(self):
        """The span is not a throughput; only ``io_busy_seconds`` is.

        Once the prefetch overlaps a layer's read with the previous layer's
        compute, the submit-to-complete span grows with *compute* -- so bytes
        divided by the span falls as the machine gets busier, which reads as a
        slow disk and is the opposite of the truth.
        """

        stats = Dsv4StreamStats(
            layer_reads=43,
            bytes_read=147_170_000_000,
            read_span_seconds=190.0,  # mostly the previous layers' compute
            io_busy_seconds=8 * 16.0,  # 16 s of wall across 8 threads
            read_wait_seconds=0.6,
            convert_seconds=13.3,
            io_threads=8,
        )
        payload = stats.as_dict()
        assert payload["io_gb_per_s"] == pytest.approx(147.17 / 16.0, rel=1e-3)
        assert payload["read_span_seconds"] == 190.0
        assert payload["read_wait_seconds"] == 0.6
        assert "read_gb_per_s" not in payload, "the span-derived rate must not ship"

    def test_zero_io_is_reported_as_absent_not_as_a_division(self):
        assert Dsv4StreamStats().as_dict()["io_gb_per_s"] is None


class TestHeartbeat:
    def test_status_file_and_progress_log(self, tmp_path):
        heartbeat = Dsv4RunHeartbeat(
            tmp_path / "status.json",
            mode="calibration",
            chunk=1024,
            total_sessions=2,
            total_tokens=100,
            total_supervised=20,
        )
        heartbeat.update(phase="planned")
        assert json.loads((tmp_path / "status.json").read_text())["phase"] == "planned"

        heartbeat.sessions_done = 1
        heartbeat.tokens_done = 50
        heartbeat.compute_seconds = 5.0
        heartbeat.session_rates.append(10.0)
        heartbeat.log(phase="session-complete", session="a")
        heartbeat.log(phase="complete")
        lines = (tmp_path / "status.log").read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["record_type"] == "dsv4_teacher_run_status_v1"
        assert first["tokens_per_s"] == 10.0
        assert first["prefill_chunk_tokens"] == 1024
        assert first["eta_s"] == 5.0
        assert json.loads(lines[1])["phase"] == "complete"


@pytest.mark.parametrize(
    "name", ("GLM_MLX_WIRED_LIMIT_GB", "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB")
)
def test_cli_refuses_repo_wired_limit_variables(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(SystemExit, match="wired-limit"):
        _refuse_wired_limit_env()


class TestValidators:
    def _good_logits(self, positions=4, top_k=3):
        values = np.tile(
            np.linspace(5.0, 1.0, top_k, dtype=np.float32), (positions, 1)
        ).astype(np.float16)
        return {
            "positions": np.arange(positions, dtype=np.int32),
            "target_token_ids": np.arange(positions, dtype=np.int32),
            "topk_logit_ids": np.tile(np.arange(top_k, dtype=np.int32), (positions, 1)),
            "topk_logit_values": values,
            "logsumexp": np.full(positions, 6.0, dtype=np.float32),
            "tail_mass": np.full(positions, 0.1, dtype=np.float16),
        }

    def test_accepts_a_valid_capture(self):
        validate_dsv4_logits_capture(self._good_logits(), top_k=3, supervised=4)

    @pytest.mark.parametrize(
        "mutate,message",
        (
            (
                lambda c: c.__setitem__("tail_mass", np.full(4, 1.5, np.float16)),
                "tail_mass",
            ),
            (
                lambda c: c.__setitem__("logsumexp", np.full(4, np.inf, np.float32)),
                "finite",
            ),
            (
                lambda c: c.__setitem__(
                    "topk_logit_values", np.zeros((4, 3), dtype=np.float32)
                ),
                "dtype",
            ),
            (
                lambda c: c.__setitem__("positions", np.array([3, 2, 1, 0], np.int32)),
                "strictly increasing",
            ),
            (
                lambda c: c.__setitem__(
                    "topk_logit_values",
                    np.tile(np.array([1.0, 5.0, 3.0], np.float16), (4, 1)),
                ),
                "descending",
            ),
            (
                lambda c: c.__setitem__("logsumexp", np.full(4, 0.0, np.float32)),
                "at least the top logit",
            ),
            (
                lambda c: c.__setitem__(
                    "topk_logit_ids", np.tile(np.array([1, 1, 2], np.int32), (4, 1))
                ),
                "distinct",
            ),
        ),
    )
    def test_rejects_drift(self, mutate, message):
        capture = self._good_logits()
        mutate(capture)
        with pytest.raises(ValueError, match=message):
            validate_dsv4_logits_capture(capture, top_k=3, supervised=4)

    def test_calibration_validator_catches_a_broken_count(self):
        arrays = {
            "layers": np.array([0, 1], dtype=np.int32),
            "num_experts": np.array(2, dtype=np.int32),
            "importance_sum__hidden": np.ones((2, 2, 4), dtype=np.float32),
            "affinity_weighted_importance__hidden": np.ones(
                (2, 2, 4), dtype=np.float32
            ),
            "route_count__hidden": np.array([[3, 3], [3, 3]], dtype=np.int64),
            "affinity_score_sum__hidden": np.ones((2, 2), dtype=np.float64),
            "total_route_count__hidden": np.array([6, 5], dtype=np.int64),
        }
        with pytest.raises(ValueError, match="must sum to"):
            validate_dsv4_calibration_capture(arrays, layers=[0, 1], num_experts=2)


# ---------------------------------------------------------------------------
# Real checkpoint: the streaming plan, off the shard headers only
# ---------------------------------------------------------------------------


@requires_checkpoint
class TestRealCheckpointSpans:
    def test_every_layer_is_one_shard_and_two_sequential_runs(self):
        spans = build_dsv4_expert_span_index(CHECKPOINT)
        config = json.loads(FROZEN_CONFIG.read_text())
        assert len(spans) == config["num_hidden_layers"] == 43
        for layer, span in spans.items():
            assert span.num_experts == config["n_routed_experts"] == 256
            assert len(span.reads) == 256 * 3 * 2
            # Scales then weights: two runs, and the gap between them is the
            # layer's own resident tensors, which a streaming pass never wants.
            assert len(span.runs) == 2, f"layer {layer} coalesced into {len(span.runs)}"
            assert 3.40e9 < span.tensor_bytes < 3.45e9
            assert span.tensor_bytes <= span.span_bytes

    def test_total_expert_bytes_match_the_measured_budget(self):
        spans = build_dsv4_expert_span_index(CHECKPOINT)
        total = sum(span.tensor_bytes for span in spans.values())
        # Increment 2 §3.7: routed experts, backbone = 147.17 GB.
        assert 146.5e9 < total < 148.0e9

    def test_reads_land_the_bytes_the_header_promises(self):
        """One layer, actually read, checked against ``safetensors``' own view."""

        from safetensors import safe_open

        from keep.quality.dsv4_teacher_runner import (
            ShardStreamingExpertProvider,
            _LayerBuffers,
            _pread_exact,
        )

        spans = build_dsv4_expert_span_index(CHECKPOINT, layers=[17])
        span = spans[17]
        buffers = _LayerBuffers(span)
        provider = ShardStreamingExpertProvider(
            CHECKPOINT, spans, swiglu_limit=10.0, io_threads=4
        )
        try:
            fd = provider._fd(span.shard)
            wanted = {("w1", "weight", 0), ("w2", "scale", 255), ("w3", "weight", 128)}
            for projection, kind, expert, offset, length in span.reads:
                if (projection, kind, expert) not in wanted:
                    continue
                count, busy = _pread_exact(
                    fd, buffers.destination(projection, kind, expert), offset, length
                )
                assert count == length
                assert busy >= 0.0
            # Two independent references, because one would only prove
            # ``preadv`` agrees with itself: ``safetensors`` for the I8 code
            # bytes (numpy has no ``float8_e8m0fnu``, so the library cannot
            # view the scales at all), and a plain buffered read at the header's
            # own offsets for both.
            shard_path = CHECKPOINT / span.shard
            with safe_open(str(shard_path), framework="np") as handle:
                for projection, kind, expert in sorted(wanted):
                    if kind != "weight":
                        continue
                    name = f"layers.17.ffn.experts.{expert}.{projection}.{kind}"
                    reference = handle.get_tensor(name)
                    got = buffers.destination(projection, kind, expert)
                    assert got.nbytes == reference.nbytes
                    assert got.tobytes() == reference.tobytes(), name
            with shard_path.open("rb") as handle:
                for projection, kind, expert, offset, length in span.reads:
                    if (projection, kind, expert) not in wanted:
                        continue
                    handle.seek(offset)
                    reference = handle.read(length)
                    got = buffers.destination(projection, kind, expert)
                    assert got.tobytes() == reference, (projection, kind, expert)
        finally:
            provider.close()
