"""DeepSeek-V4-Flash VQ adapter: args, VQ bind, resident bind, MTP, registry.

The frozen config at ``tests/data/deepseek-v4-flash-0731-config.json`` is a
byte-for-byte copy of ``config.json`` from
``deepseek-ai/DeepSeek-V4-Flash-0731`` revision
``7872f01b1d1fe23eabc4c98b48bffcef5a386062``, taken from the local snapshot at
``~/models/DeepSeek-V4-Flash-0731/`` on 2026-08-11.
sha256 ``6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023``.
It is committed (1.9 kB) so the args contract is testable without the 163 GB of
shards; JSON has no comment syntax, which is why the provenance lives here.

Tests that need the shards are marked ``checkpoint`` and skip when the snapshot
is absent.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList

from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from keep.quality.dsv4_mtp_runtime import (
    generate_dsv4_autoregressive,
    generate_dsv4_speculative,
)
from ramp.models.deepseek_v4_flash_adapter import (
    DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS,
    CompressedAttention,
    DeepseekV4FlashMTPBlock,
    DeepseekV4FlashRotatingKVCache,
    DeepseekV4FlashVQModel,
    DeepseekV4FlashVQModelArgs,
    DeepseekV4FlashVQMoE,
    DSparkAttention,
    DSparkContextCache,
    LimitedSwiGLU,
    LocalAttention,
    PoolingCache,
    SparseCompressedAttention,
    bind_deepseek_v4_flash_mtp_vq_experts,
    bind_deepseek_v4_flash_non_vq_weights,
    bind_deepseek_v4_flash_vq_experts,
    deepseek_v4_flash_args_from_config,
    deepseek_v4_flash_pooling_undo_window,
    dense_deepseek_v4_flash_routed_parameter_names,
    has_unbound_deepseek_v4_flash_mtp_experts,
    has_unbound_deepseek_v4_flash_vq_experts,
    v4_attention_factory,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = ROOT / "recipes/dsv4_teich_split_manifest_v1_20260811.json"
FROZEN_CONFIG = ROOT / "tests/data/deepseek-v4-flash-0731-config.json"
FROZEN_CONFIG_SHA256 = (
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
)
CHECKPOINT = Path.home() / "models/DeepSeek-V4-Flash-0731"

requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"DeepSeek-V4-Flash-0731 shards not present at {CHECKPOINT}",
)


def _real_config() -> dict:
    return json.loads(FROZEN_CONFIG.read_text())


# ---------------------------------------------------------------------------
# Args from the real config
# ---------------------------------------------------------------------------


def test_public_surface_is_complete():
    import ramp.models.deepseek_v4_flash_adapter as adapter

    assert [name for name in adapter.__all__ if not hasattr(adapter, name)] == []


def test_frozen_config_matches_pinned_revision_hash():
    digest = hashlib.sha256(FROZEN_CONFIG.read_bytes()).hexdigest()
    assert digest == FROZEN_CONFIG_SHA256


def test_args_from_real_config_carry_the_measured_structure():
    args = deepseek_v4_flash_args_from_config(_real_config())

    assert args.model_type == "deepseek_v4"
    assert args.num_hidden_layers == 43
    assert args.hidden_size == 4096
    assert args.vocab_size == 129280
    assert args.n_routed_experts == 256
    assert args.n_shared_experts == 1
    assert args.num_experts_per_tok == 6
    assert args.moe_intermediate_size == 2048
    assert args.num_hash_layers == 3
    assert args.head_dim == 512
    assert args.o_groups == 8
    assert args.o_lora_rank == 1024
    assert args.q_lora_rank == 1024
    assert args.index_topk == 512
    assert args.index_n_heads == 64
    assert args.index_head_dim == 128
    assert args.scoring_func == "sqrtsoftplus"
    assert args.swiglu_limit == 10.0
    assert args.sliding_window == 128
    assert args.hc_mult == 4


def test_args_ignore_loader_metadata_keys_in_the_real_config():
    """The released config has 52 keys; several are not model structure."""

    config = _real_config()
    assert len(config) == 52
    for key in ("architectures", "quantization_config", "expert_dtype", "bos_token_id"):
        assert key in config
    # Constructing must not raise on them.
    deepseek_v4_flash_args_from_config(config)


def test_compress_ratios_truncate_to_hidden_layers_and_keep_the_mtp_tail():
    config = _real_config()
    assert len(config["compress_ratios"]) == 46

    args = deepseek_v4_flash_args_from_config(config)
    assert len(args.compress_ratios) == 43
    assert args.compress_ratios[:4] == [0, 0, 4, 128]
    # The three trailing entries describe mtp.{0,1,2}: all uncompressed.
    assert args.mtp_compress_ratios == [0, 0, 0]


def test_dspark_fields_settle_the_drafter_depth():
    args = deepseek_v4_flash_args_from_config(_real_config())
    assert args.dspark_target_layer_ids == [40, 41, 42]
    assert args.dspark_block_size == 5
    assert args.dspark_markov_rank == 256
    # ``num_nextn_predict_layers`` is a legacy compat field, not the depth.
    assert args.num_nextn_predict_layers == 1
    assert args.num_mtp_blocks == 3


def test_unsupported_compress_ratio_is_rejected():
    with pytest.raises(ValueError, match="compress ratios"):
        DeepseekV4FlashVQModelArgs(num_hidden_layers=3, compress_ratios=[0, 7, 128])


def test_short_compress_ratios_are_rejected():
    with pytest.raises(ValueError, match="one entry per hidden layer"):
        DeepseekV4FlashVQModelArgs(num_hidden_layers=4, compress_ratios=[0, 4])


# ---------------------------------------------------------------------------
# Teacher window
# ---------------------------------------------------------------------------


def test_teacher_window_covers_the_longest_session_in_the_split_manifest():
    """81,920 is a decision; this is the measurement it has to survive.

    A window under ``v4_session_tokens_max`` silently truncates the longest
    sessions, and the tokens that fall off are supervised positions whose
    teacher logits would simply never exist -- a hole in the distillation target
    that no downstream gate looks for.
    """

    totals = json.loads(SPLIT_MANIFEST.read_text())["totals"]

    assert totals["sessions_included"] == 257
    assert totals["v4_raw_tokens"] == 11_714_468
    assert DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS >= totals["v4_session_tokens_max"]
    assert totals["v4_session_tokens_max"] == 77_075
    # 65,536 is the obvious power-of-two choice and it is not enough: 31 of the
    # 257 sessions are longer.
    assert totals["sessions_over_65536_v4"] == 31


def test_teacher_window_is_a_whole_number_of_pooling_windows():
    """No partial ``PoolingCache`` remainder can straddle a session boundary."""

    args = deepseek_v4_flash_args_from_config(_real_config())
    for ratio in sorted(set(args.compress_ratios) | {args.sliding_window}):
        if ratio:
            assert DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS % ratio == 0
    assert DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS <= args.max_position_embeddings


# ---------------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------------

# Routed experts are FP4 with group-32 scales, so the tiny input dims must be
# whole multiples of 32 for the drafter's dense decode path to be exercisable.
_TINY_HIDDEN = 64
_TINY_MOE = 32
_TINY_EXPERTS = 4
_TINY_VOCAB = 32


def _tiny_args(**overrides) -> DeepseekV4FlashVQModelArgs:
    kwargs = dict(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        intermediate_size=_TINY_MOE,
        moe_intermediate_size=_TINY_MOE,
        num_hidden_layers=2,
        num_attention_heads=2,
        n_shared_experts=1,
        n_routed_experts=_TINY_EXPERTS,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        num_experts_per_tok=2,
        head_dim=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        num_hash_layers=1,
        hc_mult=4,
        max_position_embeddings=64,
        compress_ratios=[0, 4, 0],
        dspark_block_size=5,
        dspark_target_layer_ids=[1],
        dspark_markov_rank=4,
    )
    kwargs.update(overrides)
    return DeepseekV4FlashVQModelArgs(**kwargs)


def _write_tiny_vq_artifacts(path: Path, args, layers) -> None:
    rng = np.random.default_rng(4731)
    path.mkdir(parents=True, exist_ok=True)
    experts = args.n_routed_experts
    for layer in layers:
        gate = mx.array(
            rng.normal(scale=0.04, size=(experts, args.moe_intermediate_size, args.hidden_size)).astype(np.float32)
        )
        up = mx.array(
            rng.normal(scale=0.04, size=(experts, args.moe_intermediate_size, args.hidden_size)).astype(np.float32)
        )
        down = mx.array(
            rng.normal(scale=0.04, size=(experts, args.hidden_size, args.moe_intermediate_size)).astype(np.float32)
        )
        switch = QuantizedVQSwitchGLU.from_weights(
            gate_weight=gate,
            up_weight=up,
            down_weight=down,
            group_size=8,
        )
        prefix = f"model.layers.{layer}.ffn.switch_mlp"
        for projection, linear in (
            ("gate_proj", switch.gate_proj),
            ("up_proj", switch.up_proj),
            ("down_proj", switch.down_proj),
        ):
            mx.save_safetensors(
                str(path / f"layer-{layer:05d}-{projection}.safetensors"),
                {
                    f"{prefix}.{projection}.codes": linear.codes,
                    f"{prefix}.{projection}.scales": linear.scales,
                    "model.vq_codebook.e8": linear.codebook,
                },
            )


def _write_tiny_mtp_vq_artifacts(path: Path, args, stages) -> None:
    rng = np.random.default_rng(4732)
    path.mkdir(parents=True, exist_ok=True)
    experts = args.n_routed_experts
    for stage in stages:
        gate = mx.array(
            rng.normal(
                scale=0.04,
                size=(experts, args.moe_intermediate_size, args.hidden_size),
            ).astype(np.float32)
        )
        up = mx.array(
            rng.normal(
                scale=0.04,
                size=(experts, args.moe_intermediate_size, args.hidden_size),
            ).astype(np.float32)
        )
        down = mx.array(
            rng.normal(
                scale=0.04,
                size=(experts, args.hidden_size, args.moe_intermediate_size),
            ).astype(np.float32)
        )
        switch = QuantizedVQSwitchGLU.from_weights(
            gate_weight=gate,
            up_weight=up,
            down_weight=down,
            group_size=8,
        )
        prefix = f"mtp_drafter.blocks.{stage}.ffn.switch_mlp"
        for projection, linear in (
            ("gate_proj", switch.gate_proj),
            ("up_proj", switch.up_proj),
            ("down_proj", switch.down_proj),
        ):
            mx.save_safetensors(
                str(path / f"mtp-{stage:05d}-{projection}.safetensors"),
                {
                    f"{prefix}.{projection}.codes": linear.codes,
                    f"{prefix}.{projection}.scales": linear.scales,
                    "model.vq_codebook.e8": linear.codebook,
                },
            )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_attention_variant_follows_compress_ratio():
    model = DeepseekV4FlashVQModel(
        _tiny_args(num_hidden_layers=3, compress_ratios=[0, 4, 128]),
        with_mtp=False,
    )
    assert isinstance(model.model.layers[0].attn, LocalAttention)
    assert isinstance(model.model.layers[1].attn, SparseCompressedAttention)
    assert isinstance(model.model.layers[2].attn, CompressedAttention)
    # Only ratio-4 layers carry an indexer.
    assert model.model.layers[1].attn.indexer is not None
    assert not hasattr(model.model.layers[0].attn, "compressor")
    assert not hasattr(model.model.layers[2].attn, "indexer")


def test_hash_layers_carry_tid2eid_and_scored_layers_carry_the_bias():
    model = DeepseekV4FlashVQModel(
        _tiny_args(num_hidden_layers=3, num_hash_layers=2, compress_ratios=[0, 0, 0]),
        with_mtp=False,
    )
    names = {key for key, _ in tree_flatten(model.parameters())}
    for layer in (0, 1):
        assert f"model.layers.{layer}.ffn.gate.tid2eid" in names
        assert f"model.layers.{layer}.ffn.gate.e_score_correction_bias" not in names
    assert "model.layers.2.ffn.gate.tid2eid" not in names
    assert "model.layers.2.ffn.gate.e_score_correction_bias" in names


def test_tid2eid_is_native_int64_and_never_a_float():
    model = DeepseekV4FlashVQModel(_tiny_args(), with_mtp=False)
    assert model.model.layers[0].ffn.gate.tid2eid.dtype == mx.int64
    # ``cast_predicate`` must keep it out of any dtype sweep.
    assert model.cast_predicate("model.layers.0.ffn.gate.tid2eid") is False


def test_mtp_drafter_is_built_and_shaped_like_a_full_moe_block():
    args = _tiny_args(dspark_target_layer_ids=[0, 1, 2])
    model = DeepseekV4FlashVQModel(args)
    assert model.mtp_drafter is not None
    assert len(model.mtp_drafter.blocks) == 3

    for stage, block in enumerate(model.mtp_drafter.blocks):
        assert isinstance(block, DeepseekV4FlashMTPBlock)
        assert isinstance(block.ffn, DeepseekV4FlashVQMoE)
        # Drafter routers are always scored (they ship ``gate.bias``).
        assert block.ffn.gate.hash is False
        assert isinstance(block.attn, LocalAttention)
        assert hasattr(block, "main_proj") is (stage == 0)
        assert hasattr(block, "markov_head") is (stage == 2)
        assert hasattr(block, "confidence_head") is (stage == 2)


def test_markov_head_w1_is_an_embedding_and_w2_a_linear():
    """Both tensors are [vocab, rank]; only the module type says gather vs matmul.

    Reference: ``deepseek_v4_dspark.py:293`` -- ``markov_w1`` is an Embedding
    (indexed by token id), ``markov_w2`` a Linear to vocabulary logits.
    """

    args = _tiny_args(dspark_target_layer_ids=[0, 1, 2])
    model = DeepseekV4FlashVQModel(args)
    head = model.mtp_drafter.blocks[2].markov_head

    assert isinstance(head.markov_w1, nn.Embedding)
    assert isinstance(head.markov_w2, nn.Linear)
    assert not isinstance(head.markov_w1, nn.Linear)

    # Same stored shape either way -- which is why the type matters.
    assert head.markov_w1.weight.shape == (args.vocab_size, args.dspark_markov_rank)
    assert head.markov_w2.weight.shape == (args.vocab_size, args.dspark_markov_rank)

    # Gather semantics: calling it with token ids returns those rows verbatim,
    # which a Linear of the same weight could not do (it would matmul).
    head.markov_w1.weight = mx.arange(
        args.vocab_size * args.dspark_markov_rank, dtype=mx.float32
    ).reshape(args.vocab_size, args.dspark_markov_rank)
    token_ids = mx.array([3, 0, 7], dtype=mx.int32)
    gathered = head.markov_w1(token_ids)
    mx.eval(gathered)
    assert gathered.shape == (3, args.dspark_markov_rank)
    assert bool(mx.array_equal(gathered, head.markov_w1.weight[token_ids]))


def test_main_proj_width_is_derived_from_dspark_target_layer_ids():
    """Reference: ``hidden_size * len(dspark_target_layer_ids)`` (dspark.py:319).

    3 * hidden is a coincidence of this release targeting three layers, not the
    rule -- a 2-target config must get a narrower projection.
    """

    three = _tiny_args(dspark_target_layer_ids=[0, 1, 2])
    assert three.main_proj_in_features == 3 * three.hidden_size
    model_three = DeepseekV4FlashVQModel(three)
    assert model_three.mtp_drafter.blocks[0].main_proj.weight.shape == (
        three.hidden_size,
        3 * three.hidden_size,
    )

    two = _tiny_args(dspark_target_layer_ids=[0, 1], compress_ratios=[0, 4, 0, 0])
    assert two.main_proj_in_features == 2 * two.hidden_size
    model_two = DeepseekV4FlashVQModel(two)
    assert len(model_two.mtp_drafter.blocks) == 2
    assert model_two.mtp_drafter.blocks[0].main_proj.weight.shape == (
        two.hidden_size,
        2 * two.hidden_size,
    )


def test_real_config_main_proj_width_matches_the_shipped_tensor():
    args = deepseek_v4_flash_args_from_config(_real_config())
    # The released mtp.0.main_proj.weight is [4096, 12288].
    assert args.main_proj_in_features == 12288
    assert args.main_proj_in_features == args.hidden_size * len(
        args.dspark_target_layer_ids
    )


def test_attention_factory_honours_an_explicit_compress_ratio():
    """A drafter stage is indexed 0..N-1 but its ratio lives past the backbone.

    Accepting ``compress_ratio`` and then re-reading
    ``config.compress_ratios[layer_idx]`` would hand a drafter stage the ratio
    of backbone layer 0/1/2 while looking correct.
    """

    args = _tiny_args(num_hidden_layers=3, compress_ratios=[0, 4, 128])

    # Backbone layer 1 is ratio 4; ask for 128 at the same index.
    override = v4_attention_factory(args, 1, compress_ratio=128)
    assert isinstance(override, CompressedAttention)
    assert override.compress_ratio == 128
    assert override.compressor.compress_ratio == 128

    # And ratio 4 at an index whose backbone ratio is 128.
    override = v4_attention_factory(args, 2, compress_ratio=4)
    assert isinstance(override, SparseCompressedAttention)
    assert override.compress_ratio == 4
    assert override.compressor.compress_ratio == 4
    assert override.indexer.compressor.compress_ratio == 4

    # No override still reads the backbone schedule.
    assert v4_attention_factory(args, 1).compress_ratio == 4


def test_mtp_stage_attention_does_not_inherit_backbone_ratios():
    args = _tiny_args(
        num_hidden_layers=3,
        compress_ratios=[0, 4, 128, 0, 0, 0],
        dspark_target_layer_ids=[0, 1, 2],
    )
    model = DeepseekV4FlashVQModel(args)
    assert args.mtp_compress_ratios == [0, 0, 0]
    for block in model.mtp_drafter.blocks:
        assert isinstance(block.attn, LocalAttention)
        assert block.attn.compress_ratio == 0


def test_no_dense_routed_parameters_exist_before_any_bind():
    model = DeepseekV4FlashVQModel(_tiny_args(), with_mtp=False)
    assert dense_deepseek_v4_flash_routed_parameter_names(model) == ()
    assert has_unbound_deepseek_v4_flash_vq_experts(model) is True


def test_running_an_unbound_moe_layer_fails_loudly():
    model = DeepseekV4FlashVQModel(_tiny_args(), with_mtp=False)
    with pytest.raises(RuntimeError, match="must be bound"):
        model.model.layers[0].ffn(mx.zeros((1, 2, _TINY_HIDDEN)), mx.zeros((1, 2), mx.int32))


# ---------------------------------------------------------------------------
# Synthetic VQ bind
# ---------------------------------------------------------------------------


def test_synthetic_vq_bind_leaves_zero_unbound_and_zero_dense_routed(tmp_path):
    args = _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, range(args.num_hidden_layers))

    bound = bind_deepseek_v4_flash_vq_experts(model, artifact_dir, strict=True)

    assert bound == tuple(range(args.num_hidden_layers))
    assert has_unbound_deepseek_v4_flash_vq_experts(model) is False
    assert dense_deepseek_v4_flash_routed_parameter_names(model) == ()
    for layer in model.model.layers:
        assert isinstance(layer.ffn.switch_mlp, QuantizedVQSwitchGLU)
        assert layer.ffn.switch_mlp.num_experts == args.n_routed_experts
        assert layer.ffn.switch_mlp.input_dims == args.hidden_size
        assert layer.ffn.switch_mlp.hidden_dims == args.moe_intermediate_size


def test_bound_switch_glu_honours_the_projection_call_contract(tmp_path):
    args = _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, range(args.num_hidden_layers))
    bind_deepseek_v4_flash_vq_experts(model, artifact_dir)

    switch = model.model.layers[0].ffn.switch_mlp
    x = mx.random.normal((3, args.hidden_size))
    indices = mx.array([[0, 1], [2, 3], [1, 2]], dtype=mx.uint32)
    y = switch(x, indices)
    mx.eval(y)
    assert y.shape == (3, args.num_experts_per_tok, args.hidden_size)
    assert bool(mx.all(mx.isfinite(y)))


def test_bound_routed_experts_carry_the_swiglu_limit_clamp(tmp_path):
    """Upstream builds the routed SwitchGLU with LimitedSwiGLU(swiglu_limit).

    Dropping it is the archetypal silent failure: plain SwiGLU has the same
    shapes and finite output, and differs only where an activation exceeds the
    limit.
    """

    args = _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, range(args.num_hidden_layers))
    bind_deepseek_v4_flash_vq_experts(model, artifact_dir)

    for layer in model.model.layers:
        activation = layer.ffn.switch_mlp.get("activation")
        assert isinstance(activation, LimitedSwiGLU)
        assert activation.limit == args.swiglu_limit == 10.0


def test_binding_a_switch_mlp_without_the_clamp_is_refused(tmp_path):
    args = _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, (0,))

    from ramp.models.deepseek_v4_flash_adapter import (
        load_deepseek_v4_flash_vq_switch_glu,
    )

    unclamped = load_deepseek_v4_flash_vq_switch_glu(
        artifact_dir, 0, swiglu_limit=args.swiglu_limit
    )
    unclamped.activation = None
    with pytest.raises(ValueError, match="LimitedSwiGLU"):
        model.model.layers[0].ffn.bind_switch_mlp(unclamped)

    wrong_limit = load_deepseek_v4_flash_vq_switch_glu(
        artifact_dir, 0, swiglu_limit=args.swiglu_limit
    )
    wrong_limit.activation = LimitedSwiGLU(4.0)
    with pytest.raises(ValueError, match="swiglu_limit must match"):
        model.model.layers[0].ffn.bind_switch_mlp(wrong_limit)


def test_swiglu_limit_changes_the_routed_output_and_matches_a_reference():
    """Numerical proof that the clamp is not decoration.

    The projections here are high precision (no VQ error to hide behind), and
    the inputs are scaled so gate and up land well outside +-10, which is where
    LimitedSwiGLU and plain SwiGLU disagree.
    """

    from ramp.nn.switch_linear import HighPrecisionSwitchLinear

    experts, in_dim, hidden = 2, 8, 4
    rng = np.random.default_rng(1010)

    def linear(out_dim: int, inp_dim: int) -> HighPrecisionSwitchLinear:
        weight = mx.array(
            rng.normal(scale=1.5, size=(experts, out_dim, inp_dim)).astype(np.float32)
        )
        return HighPrecisionSwitchLinear(weight=weight)

    gate_proj, up_proj, down_proj = linear(hidden, in_dim), linear(hidden, in_dim), linear(in_dim, hidden)
    limit = 10.0

    clamped = QuantizedVQSwitchGLU(
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        activation=LimitedSwiGLU(limit),
    )
    plain = QuantizedVQSwitchGLU(
        gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj
    )

    x = mx.array(rng.normal(scale=6.0, size=(3, in_dim)).astype(np.float32))
    indices = mx.array([[0], [1], [0]], dtype=mx.uint32)

    clamped_out = clamped(x, indices)
    plain_out = plain(x, indices)
    mx.eval(clamped_out, plain_out)

    # The clamp must actually bite on this input.
    raw_gate = gate_proj(x, indices)
    raw_up = up_proj(x, indices)
    mx.eval(raw_gate, raw_up)
    assert float(mx.max(raw_gate)) > limit
    assert float(mx.max(mx.abs(raw_up))) > limit
    assert not bool(mx.allclose(clamped_out, plain_out))

    # Reference: clamp gate above, clip up to +-limit, then silu(gate) * up.
    ref_gate = mx.minimum(raw_gate, limit)
    ref_up = mx.clip(raw_up, -limit, limit)
    reference = down_proj(nn.silu(ref_gate) * ref_up, indices)
    mx.eval(reference)
    assert bool(mx.allclose(clamped_out, reference, atol=1e-5))

    # And the default (no activation) is still exactly the historical SwiGLU.
    plain_reference = down_proj(nn.silu(raw_gate) * raw_up, indices)
    mx.eval(plain_reference)
    assert bool(mx.allclose(plain_out, plain_reference, atol=1e-5))


def test_default_activation_hook_is_bit_identical_to_plain_swiglu():
    """The hook must be invisible to GLM-4.5-Air / GLM-5.2 / Qwen.

    They construct ``QuantizedVQSwitchGLU`` without an activation, so the
    default branch has to be the same expression, not merely a close one.
    """

    from ramp.nn.switch_linear import HighPrecisionSwitchLinear

    experts, in_dim, hidden = 2, 8, 4
    rng = np.random.default_rng(2020)

    def linear(out_dim: int, inp_dim: int) -> HighPrecisionSwitchLinear:
        weight = mx.array(
            rng.normal(scale=1.5, size=(experts, out_dim, inp_dim)).astype(np.float32)
        )
        return HighPrecisionSwitchLinear(weight=weight)

    gate_proj, up_proj, down_proj = linear(hidden, in_dim), linear(hidden, in_dim), linear(in_dim, hidden)
    switch = QuantizedVQSwitchGLU(
        gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj
    )
    assert switch.get("activation") is None

    x = mx.array(rng.normal(scale=6.0, size=(3, in_dim)).astype(np.float32))
    indices = mx.array([[0], [1], [0]], dtype=mx.uint32)
    got = switch(x, indices)
    expected = down_proj(
        nn.silu(gate_proj(x, indices)) * up_proj(x, indices), indices
    )
    mx.eval(got, expected)
    assert bool(mx.array_equal(got, expected))


def test_partial_vq_bind_is_reported_and_strict_mode_refuses_it(tmp_path):
    args = _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, (0,))

    assert bind_deepseek_v4_flash_vq_experts(model, artifact_dir, layers=(0,)) == (0,)
    assert has_unbound_deepseek_v4_flash_vq_experts(model) is True

    fresh = DeepseekV4FlashVQModel(args, with_mtp=False)
    with pytest.raises(ValueError, match="exact runtime sparse-layer set"):
        bind_deepseek_v4_flash_vq_experts(fresh, artifact_dir, layers=(0,), strict=True)


def test_vq_bind_rejects_layers_outside_the_backbone(tmp_path):
    args = _tiny_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    with pytest.raises(ValueError, match="outside the backbone"):
        bind_deepseek_v4_flash_vq_experts(model, tmp_path, layers=(99,))


def test_vq_bind_rejects_an_artifact_with_the_wrong_expert_count(tmp_path):
    args = _tiny_args()
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, (0, 1))

    wider = DeepseekV4FlashVQModel(
        _tiny_args(n_routed_experts=_TINY_EXPERTS * 2), with_mtp=False
    )
    with pytest.raises(ValueError, match="expert count"):
        bind_deepseek_v4_flash_vq_experts(wider, artifact_dir)


def test_mtp_vq_bind_is_exact_and_strict(tmp_path):
    args = _tiny_args(dspark_target_layer_ids=[0, 1, 2])
    model = DeepseekV4FlashVQModel(args)
    artifact_dir = tmp_path / "vq"
    _write_tiny_mtp_vq_artifacts(artifact_dir, args, range(args.num_mtp_blocks))

    assert has_unbound_deepseek_v4_flash_mtp_experts(model) is True
    bound = bind_deepseek_v4_flash_mtp_vq_experts(
        model, artifact_dir, strict=True
    )

    assert bound == tuple(range(args.num_mtp_blocks))
    assert has_unbound_deepseek_v4_flash_mtp_experts(model) is False
    assert dense_deepseek_v4_flash_routed_parameter_names(
        model, include_mtp=True
    ) == ()
    assert all(
        isinstance(block.ffn.switch_mlp, QuantizedVQSwitchGLU)
        for block in model.mtp_drafter.blocks
    )


def test_mtp_vq_bind_fails_closed_on_partial_duplicate_or_absent_drafter(tmp_path):
    args = _tiny_args(dspark_target_layer_ids=[0, 1, 2])
    model = DeepseekV4FlashVQModel(args)
    with pytest.raises(ValueError, match="exact MTP stage set"):
        bind_deepseek_v4_flash_mtp_vq_experts(
            model, tmp_path, stages=(0, 1), strict=True
        )
    with pytest.raises(ValueError, match="duplicates"):
        bind_deepseek_v4_flash_mtp_vq_experts(
            model, tmp_path, stages=(0, 0)
        )
    with pytest.raises(ValueError, match="without an MTP drafter"):
        bind_deepseek_v4_flash_mtp_vq_experts(
            DeepseekV4FlashVQModel(args, with_mtp=False), tmp_path
        )


# ---------------------------------------------------------------------------
# Tiny-config forward
# ---------------------------------------------------------------------------
#
# Increment 1 bound weights and asserted structure; the ``__call__`` bodies of
# the attention variants, the compressor, the indexer and the hyper-connections
# were vendored but never executed. These tests are the does-it-run-and-stay-
# finite gate over the *stripped* code paths (the native ``wsdpa`` /
# ``glm_moe_dsa`` kernels and the ``exact_*`` decode branch are gone, so the MLX
# fallbacks below are the only implementation KEEP has). They compare against no
# external reference: logit parity against upstream waits until a reference
# implementation of this architecture is runnable.

# Yarn scaling is on the compressed-attention and compressor RoPEs in the real
# config; keeping it here means the correction-dim branch runs rather than the
# ``default`` one.
_FORWARD_ROPE_SCALING = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 4,
    "original_max_position_embeddings": 256,
    "type": "yarn",
}

# 128 is the largest compress ratio, so the prompt must exceed it for the
# ratio-128 layer to pool anything at all; 160 also pools 40 windows at ratio 4,
# comfortably past ``index_topk`` so the indexer's sparse branch is the one
# taken rather than the "few enough pooled tokens to just concatenate" branch.
_FORWARD_PROMPT = 160


def _forward_args(**overrides) -> DeepseekV4FlashVQModelArgs:
    """A 4-layer config that reaches every backbone code path at once.

    ``compress_ratios=[0, 4, 128, 4]`` instantiates all three attention
    variants, and ``num_hash_layers=2`` puts hash routing on layers 0-1 and
    scored routing on layers 2-3, so both ``MoEGate`` branches run in the same
    forward.
    """

    kwargs = {
        "vocab_size": 64,
        "num_hidden_layers": 4,
        "compress_ratios": [0, 4, 128, 4],
        "num_hash_layers": 2,
        "n_routed_experts": 8,
        "sliding_window": 16,
        "max_position_embeddings": 4096,
        "rope_scaling": _FORWARD_ROPE_SCALING,
        "hc_sinkhorn_iters": 3,
        "dspark_target_layer_ids": [],
    }
    kwargs.update(overrides)
    return _tiny_args(**kwargs)


def _randomize(model: DeepseekV4FlashVQModel, seed: int = 20260811) -> None:
    """Give every non-VQ parameter a non-degenerate value.

    Skeleton defaults leave ``gate.weight``, ``tid2eid``, ``attn_sink``, the
    hyper-connection mixers and the compressor's ``ape`` at zero, which collapses
    routing onto expert 0 and makes the hyper-connection mixes constant -- a
    forward that runs on those proves much less than one that runs on real
    values.
    """

    rng = np.random.default_rng(seed)
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if ".switch_mlp." in key:
            continue  # VQ codes/scales/codebook arrive from the artifact
        if key.endswith("tid2eid"):
            table = rng.integers(0, model.args.n_routed_experts, size=value.shape)
            updates.append((key, mx.array(table.astype(np.int64))))
            continue
        if not mx.issubdtype(value.dtype, mx.floating):
            continue
        draw = rng.normal(scale=0.05 if value.ndim >= 2 else 0.02, size=value.shape)
        if key.endswith("norm.weight"):
            draw = 1.0 + 0.02 * draw  # RMSNorm gains live around 1
        updates.append((key, mx.array(draw.astype(np.float32)).astype(value.dtype)))
    model.update(tree_unflatten(updates))
    mx.eval(model.parameters())


def _forward_model(tmp_path, **overrides):
    args = _forward_args(**overrides)
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    artifact_dir = tmp_path / f"vq-{abs(hash(tuple(args.compress_ratios))) % 10_000}"
    _write_tiny_vq_artifacts(artifact_dir, args, range(args.num_hidden_layers))
    bind_deepseek_v4_flash_vq_experts(model, artifact_dir, strict=True)
    _randomize(model)
    return args, model


def _prompt(args, length=_FORWARD_PROMPT, seed=3) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, args.vocab_size, size=(1, length)))


def test_tiny_forward_covers_all_three_attention_variants_and_both_gates(tmp_path):
    args, model = _forward_model(tmp_path)

    assert [type(layer.attn).__name__ for layer in model.model.layers] == [
        "LocalAttention",
        "SparseCompressedAttention",
        "CompressedAttention",
        "SparseCompressedAttention",
    ]
    assert [layer.ffn.gate.hash for layer in model.model.layers] == [
        True,
        True,
        False,
        False,
    ]

    logits = model(_prompt(args), cache=model.make_cache())
    mx.eval(logits)

    assert logits.shape == (1, _FORWARD_PROMPT, args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)))
    # Not a constant field: a forward that returns the same number everywhere
    # would also be finite.
    assert float(mx.std(logits)) > 1e-4


def test_forward_responds_to_its_input(tmp_path):
    args, model = _forward_model(tmp_path)
    a = model(_prompt(args, seed=3), cache=model.make_cache())
    b = model(_prompt(args, seed=99), cache=model.make_cache())
    mx.eval(a, b)

    assert bool(mx.all(mx.isfinite(a)))
    assert bool(mx.all(mx.isfinite(b)))
    assert float(mx.max(mx.abs(a - b))) > float(mx.std(a))


def test_prefill_then_decode_advances_every_cache(tmp_path):
    args, model = _forward_model(tmp_path)
    cache = model.make_cache()

    prefill = model(_prompt(args), cache=cache)
    mx.eval(prefill)

    def offsets(entry):
        return [c.offset for c in entry] if isinstance(entry, CacheList) else [entry.offset]

    # Local KV offsets count tokens; pooling offsets count completed windows.
    assert offsets(cache[0]) == [_FORWARD_PROMPT]
    assert offsets(cache[1]) == [_FORWARD_PROMPT, _FORWARD_PROMPT // 4, _FORWARD_PROMPT // 4]
    assert offsets(cache[2]) == [_FORWARD_PROMPT, _FORWARD_PROMPT // 128]
    assert offsets(cache[3]) == [_FORWARD_PROMPT, _FORWARD_PROMPT // 4, _FORWARD_PROMPT // 4]
    # The ratio-128 layer carries the 32-token tail that did not complete a
    # window; the ratio-4 layers divide evenly and carry nothing.
    assert cache[2][1].remainder == _FORWARD_PROMPT % 128
    assert cache[1][1].remainder == 0

    step = model(mx.array([[7]]), cache=cache)
    mx.eval(step)

    assert step.shape == (1, 1, args.vocab_size)
    assert bool(mx.all(mx.isfinite(step)))
    assert offsets(cache[0]) == [_FORWARD_PROMPT + 1]
    assert offsets(cache[1])[0] == _FORWARD_PROMPT + 1
    # One decode token does not complete a new window at either ratio.
    assert offsets(cache[1])[1] == _FORWARD_PROMPT // 4
    assert offsets(cache[2])[1] == _FORWARD_PROMPT // 128
    assert cache[2][1].remainder == (_FORWARD_PROMPT % 128) + 1

    # Enough decode steps to complete another ratio-4 window and to rotate the
    # sliding-window KV cache past its max size.
    for token in range(8):
        step = model(mx.array([[token % args.vocab_size]]), cache=cache)
    mx.eval(step)
    assert bool(mx.all(mx.isfinite(step)))
    assert offsets(cache[0]) == [_FORWARD_PROMPT + 9]
    assert offsets(cache[1])[1] == (_FORWARD_PROMPT + 8) // 4
    assert cache[0].keys.shape[2] <= args.sliding_window + _FORWARD_PROMPT


def test_chunked_prefill_matches_a_single_shot_prefill(tmp_path):
    """Chunking is not optional at scale, so it must not change the answer.

    A single-shot prefill materialises an ``(L, L)`` attention mask and an
    ``(L, L)`` score matrix; at a 45 k-token session that is quadratic in both
    time and memory even though ``sliding_window`` makes all but 128 columns
    masked-out. Every real run therefore feeds the prompt in chunks, which takes
    a different path through ``RotatingKVCache`` (concat-then-trim rather than
    keep-everything) and through ``PoolingCache.accumulate_windows`` (partial
    windows buffered across calls rather than one exact division).

    Agreement is *close*, not bit-exact, and deliberately asserted that way. The
    two paths reduce over the same values in a different order, and the
    disagreement compounds with depth: measured on this fixture, one local layer
    with no windowing disagrees by 2.5e-6 relative, one windowed layer by 5.1e-5,
    and four layers by 1.8e-4. It is not routing flipping (the same gap survives
    ``n_routed_experts=1``). The consequence for the campaign is that a teacher
    logit cache is only reproducible against a *pinned* chunk size.
    """

    args, model = _forward_model(tmp_path)
    ids = _prompt(args)

    single = model(ids, cache=model.make_cache())
    chunked_cache = model.make_cache()
    chunk = 32
    for start in range(0, ids.shape[1], chunk):
        chunked = model(ids[:, start : start + chunk], cache=chunked_cache)
    unrelated = model(_prompt(args, seed=99), cache=model.make_cache())
    mx.eval(single, chunked, unrelated)

    scale = float(mx.max(mx.abs(single[:, -1])))
    gap = float(mx.max(mx.abs(chunked[:, -1] - single[:, -1])))
    assert bool(mx.all(mx.isfinite(chunked)))
    assert gap < 5e-3 * scale
    # Teeth: a cache that dropped or misordered keys would land in the same
    # neighbourhood as a completely different prompt, not three orders of
    # magnitude below it.
    unrelated_gap = float(mx.max(mx.abs(unrelated[:, -1] - single[:, -1])))
    assert gap < 0.01 * unrelated_gap


def test_forward_without_a_cache_drops_the_pooled_causal_mask(tmp_path):
    """``cache=None`` is not an equivalent shortcut once anything is pooled.

    The pooled visibility mask comes from ``PoolingCache.make_mask``, so with no
    cache a query can attend pooled windows built from tokens that come after it
    (``pmask = pool_cache.make_mask(...) if pool_cache is not None else None``,
    vendored 1:1 from upstream). Local attention is unaffected -- its mask comes
    from ``create_attention_mask``. Pinned here because "it ran and was finite"
    is exactly how this would otherwise be discovered.
    """

    args, local_only = _forward_model(tmp_path, compress_ratios=[0, 0, 0, 0])
    ids = _prompt(args)
    cached = local_only(ids, cache=local_only.make_cache())
    uncached = local_only(ids, cache=None)
    mx.eval(cached, uncached)
    assert float(mx.max(mx.abs(cached - uncached))) == 0.0

    _, pooled = _forward_model(tmp_path)
    pooled_cached = pooled(ids, cache=pooled.make_cache())
    pooled_uncached = pooled(ids, cache=None)
    mx.eval(pooled_cached, pooled_uncached)
    assert bool(mx.all(mx.isfinite(pooled_uncached)))
    assert float(mx.max(mx.abs(pooled_cached - pooled_uncached))) > 0.0


def test_bfloat16_forward_keeps_the_model_output_logits_in_float32(tmp_path):
    """BF16 weights feed the intentional F32 logit/output stability boundary.

    The hyper-connection accumulators are deliberately F32 and the shared LM
    head receives their F32 collapse, so both prompt and decode logits are F32.
    This explicitly rebaselines the old BF16-logit assertion to the unchanged
    reference arithmetic; casting the output down would be a production change,
    not a stronger BF16 test.
    """

    args, model = _forward_model(tmp_path)
    model.update(
        tree_unflatten(
            [
                (key, value.astype(mx.bfloat16))
                if model.cast_predicate(key) and mx.issubdtype(value.dtype, mx.floating)
                else (key, value)
                for key, value in tree_flatten(model.parameters())
            ]
        )
    )
    mx.eval(model.parameters())

    # ``cast_predicate`` holds back the fp32 accumulators upstream keeps in fp32.
    kept = {
        key
        for key, value in tree_flatten(model.parameters())
        if value.dtype == mx.float32
    }
    assert any(".attn_hc." in key for key in kept)
    assert any("attn_sink" in key for key in kept)
    assert any(
        value.dtype == mx.bfloat16
        for key, value in tree_flatten(model.parameters())
        if model.cast_predicate(key) and mx.issubdtype(value.dtype, mx.floating)
    )

    cache = model.make_cache()
    logits = model(_prompt(args), cache=cache)
    step = model(mx.array([[3]]), cache=cache)
    mx.eval(logits, step)

    assert logits.dtype == mx.float32
    assert step.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(logits)))
    assert bool(mx.all(mx.isfinite(step)))


def test_forward_runs_a_batch(tmp_path):
    args, model = _forward_model(tmp_path)
    rng = np.random.default_rng(17)
    ids = mx.array(rng.integers(0, args.vocab_size, size=(3, _FORWARD_PROMPT)))

    logits = model(ids, cache=model.make_cache())
    mx.eval(logits)

    assert logits.shape == (3, _FORWARD_PROMPT, args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)))
    # Independent rows: nothing leaks across the batch dimension.
    solo = model(ids[:1], cache=model.make_cache())
    mx.eval(solo)
    assert float(mx.max(mx.abs(solo - logits[:1]))) < 1e-4


# ---------------------------------------------------------------------------
# Synthetic non-VQ bind
# ---------------------------------------------------------------------------

_FP8_BLOCK = 128


def _save_raw_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Write a safetensors shard with dtypes numpy/MLX cannot express.

    ``mx.save_safetensors`` has no ``F8_E4M3``/``F8_E8M0`` writer, and those are
    exactly the dtypes the DeepSeek-V4 release ships, so the fixture emits the
    container by hand. Exercising the adapter's "opaque dtype means raw code
    bytes" reader is the whole point of this fixture.
    """

    header: dict[str, object] = {}
    offset = 0
    blobs: list[bytes] = []
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        for blob in blobs:
            handle.write(blob)


def _mutate_shard_header(path: Path, overrides: dict[str, dict]) -> None:
    """Rewrite tensor descriptors in place to simulate a malformed checkpoint.

    The payload is untouched, so the corruption is exactly the kind the binder's
    guards exist to catch: right bytes, wrong declaration.
    """

    shard = path / "model-00001-of-00001.safetensors"
    with shard.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        payload = handle.read()
    for name, patch in overrides.items():
        header[name].update(patch)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    with shard.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        handle.write(payload)


def _drop_from_index(path: Path, names: tuple[str, ...]) -> None:
    """Remove entries from the safetensors index without touching the shard."""

    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in names:
        index["weight_map"].pop(name, None)
    index_path.write_text(json.dumps(index))


def _bf16_bytes(values: np.ndarray) -> bytes:
    """Truncate float32 to bfloat16 words (round-to-zero is fine for fixtures)."""

    words = (values.astype("<f4").view("<u4") >> 16).astype("<u2")
    return words.tobytes()


def _fp8_block_pair(rng, rows: int, cols: int):
    """Random e4m3 code bytes plus a matching ue8m0 128x128 block scale.

    0x7F/0xFF are e4m3 NaN and 0xFF is ue8m0 NaN; a fixture that emitted them
    would make the finiteness assertions vacuous, so they are steered away.
    """

    codes = rng.integers(0, 256, size=(rows, cols)).astype(np.uint8)
    codes = np.where((codes & 0x7F) == 0x7F, 0x7E, codes)
    n_blocks = (-(-rows // _FP8_BLOCK), -(-cols // _FP8_BLOCK))
    scales = rng.integers(120, 134, size=n_blocks).astype(np.uint8)
    return codes, scales


def _write_synthetic_checkpoint(
    path: Path,
    args,
    *,
    with_mtp: bool = False,
) -> dict[str, str]:
    """One shard carrying the release's real per-tensor dtype layout."""

    rng = np.random.default_rng(99)
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def add_fp8(name: str, rows: int, cols: int) -> None:
        codes, scales = _fp8_block_pair(rng, rows, cols)
        tensors[f"{name}.weight"] = ("F8_E4M3", codes.shape, codes.tobytes())
        tensors[f"{name}.scale"] = ("F8_E8M0", scales.shape, scales.tobytes())

    def add_bf16(name: str, shape) -> None:
        values = rng.normal(size=shape)
        tensors[name] = ("BF16", tuple(shape), _bf16_bytes(values))

    def add_f32(name: str, shape) -> None:
        values = rng.normal(size=shape).astype("<f4")
        tensors[name] = ("F32", tuple(shape), values.tobytes())

    n_heads = args.num_attention_heads
    hidden = args.hidden_size
    head_dim = args.head_dim
    hc_mix = (2 + args.hc_mult) * args.hc_mult
    shared_dim = args.moe_intermediate_size * args.n_shared_experts

    def add_block(prefix: str, ratio: int, *, hash_routing: bool) -> None:
        add_fp8(f"{prefix}.attn.wq_a", args.q_lora_rank, hidden)
        add_fp8(f"{prefix}.attn.wq_b", n_heads * head_dim, args.q_lora_rank)
        add_fp8(f"{prefix}.attn.wkv", head_dim, hidden)
        # ``wo_a`` ships flattened: (o_groups * o_lora_rank, n_heads*head_dim /
        # o_groups), which the binder folds back into MultiLinear's 3-D layout.
        add_fp8(
            f"{prefix}.attn.wo_a",
            args.o_groups * args.o_lora_rank,
            n_heads * head_dim // args.o_groups,
        )
        add_fp8(f"{prefix}.attn.wo_b", hidden, args.o_groups * args.o_lora_rank)
        add_bf16(f"{prefix}.attn.q_norm.weight", (args.q_lora_rank,))
        add_bf16(f"{prefix}.attn.kv_norm.weight", (head_dim,))
        add_f32(f"{prefix}.attn.attn_sink", (n_heads,))
        add_bf16(f"{prefix}.attn_norm.weight", (hidden,))
        add_bf16(f"{prefix}.ffn_norm.weight", (hidden,))
        for group in ("attn", "ffn"):
            add_f32(f"{prefix}.hc_{group}_fn", (hc_mix, args.hc_mult * hidden))
            add_f32(f"{prefix}.hc_{group}_base", (hc_mix,))
            add_f32(f"{prefix}.hc_{group}_scale", (3,))
        add_bf16(f"{prefix}.ffn.gate.weight", (args.n_routed_experts, hidden))
        if hash_routing:
            table = rng.integers(
                0,
                args.n_routed_experts,
                size=(args.vocab_size, args.num_experts_per_tok),
            ).astype("<i8")
            tensors[f"{prefix}.ffn.gate.tid2eid"] = ("I64", table.shape, table.tobytes())
        else:
            add_f32(f"{prefix}.ffn.gate.bias", (args.n_routed_experts,))
        add_fp8(f"{prefix}.ffn.shared_experts.w1", shared_dim, hidden)
        add_fp8(f"{prefix}.ffn.shared_experts.w2", hidden, shared_dim)
        add_fp8(f"{prefix}.ffn.shared_experts.w3", shared_dim, hidden)

        if ratio != 0:
            out_dim = head_dim * (2 if ratio == 4 else 1)
            add_bf16(f"{prefix}.attn.compressor.wkv.weight", (out_dim, hidden))
            add_bf16(f"{prefix}.attn.compressor.wgate.weight", (out_dim, hidden))
            add_f32(f"{prefix}.attn.compressor.ape", (ratio, out_dim))
            add_bf16(f"{prefix}.attn.compressor.norm.weight", (head_dim,))
            if ratio == 4:
                index_out = args.index_head_dim * 2
                add_bf16(
                    f"{prefix}.attn.indexer.compressor.wkv.weight",
                    (index_out, hidden),
                )
                add_bf16(
                    f"{prefix}.attn.indexer.compressor.wgate.weight",
                    (index_out, hidden),
                )
                add_f32(f"{prefix}.attn.indexer.compressor.ape", (ratio, index_out))
                add_bf16(
                    f"{prefix}.attn.indexer.compressor.norm.weight",
                    (args.index_head_dim,),
                )
                add_bf16(
                    f"{prefix}.attn.indexer.weights_proj.weight",
                    (args.index_n_heads, hidden),
                )
                add_fp8(
                    f"{prefix}.attn.indexer.wq_b",
                    args.index_n_heads * args.index_head_dim,
                    args.q_lora_rank,
                )

        # Routed experts: FP4 packed two-per-byte in I8 with E8M0 group-32
        # scales. Present so the binder has something to *skip* (backbone) and
        # something to dense-decode (drafter).
        for expert in range(args.n_routed_experts):
            for projection, (out_dim, in_dim) in (
                ("w1", (args.moe_intermediate_size, hidden)),
                ("w2", (hidden, args.moe_intermediate_size)),
                ("w3", (args.moe_intermediate_size, hidden)),
            ):
                base = f"{prefix}.ffn.experts.{expert}.{projection}"
                packed = rng.integers(0, 256, size=(out_dim, in_dim // 2)).astype(
                    np.uint8
                )
                group_scales = rng.integers(
                    120, 134, size=(out_dim, in_dim // 32)
                ).astype(np.uint8)
                tensors[f"{base}.weight"] = ("I8", packed.shape, packed.tobytes())
                tensors[f"{base}.scale"] = (
                    "F8_E8M0",
                    group_scales.shape,
                    group_scales.tobytes(),
                )

    add_bf16("embed.weight", (args.vocab_size, hidden))
    add_bf16("head.weight", (args.vocab_size, hidden))
    add_bf16("norm.weight", (hidden,))
    add_f32("hc_head_fn", (args.hc_mult, args.hc_mult * hidden))
    add_f32("hc_head_base", (args.hc_mult,))
    add_f32("hc_head_scale", (1,))

    for layer in range(args.num_hidden_layers):
        add_block(
            f"layers.{layer}",
            args.compress_ratios[layer],
            hash_routing=layer < args.num_hash_layers,
        )

    if with_mtp:
        stages = args.num_mtp_blocks
        ratios = list(args.mtp_compress_ratios) + [0] * stages
        for stage in range(stages):
            # Drafter routers always ship ``gate.bias``, never ``tid2eid``.
            add_block(f"mtp.{stage}", ratios[stage], hash_routing=False)
            if stage == 0:
                add_bf16(f"mtp.{stage}.main_norm.weight", (hidden,))
                add_fp8(f"mtp.{stage}.main_proj", hidden, 3 * hidden)
            if stage == stages - 1:
                add_bf16(f"mtp.{stage}.norm.weight", (hidden,))
                add_f32(f"mtp.{stage}.hc_head_fn", (args.hc_mult, args.hc_mult * hidden))
                add_f32(f"mtp.{stage}.hc_head_base", (args.hc_mult,))
                add_f32(f"mtp.{stage}.hc_head_scale", (1,))
                add_bf16(
                    f"mtp.{stage}.markov_head.markov_w1.weight",
                    (args.vocab_size, args.dspark_markov_rank),
                )
                add_bf16(
                    f"mtp.{stage}.markov_head.markov_w2.weight",
                    (args.vocab_size, args.dspark_markov_rank),
                )
                add_bf16(
                    f"mtp.{stage}.confidence_head.proj.weight",
                    (1, hidden + args.dspark_markov_rank),
                )

    shard = "model-00001-of-00001.safetensors"
    _save_raw_safetensors(path / shard, tensors)
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {name: shard for name in tensors},
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return index["weight_map"]


def test_synthetic_non_vq_bind_reports_and_preserves_dtypes(tmp_path):
    args = _tiny_args(num_hidden_layers=2, num_hash_layers=1, compress_ratios=[0, 4])
    _write_synthetic_checkpoint(tmp_path, args)
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, tmp_path, include_mtp=False, strict=True
    )

    assert report.missing_model_parameters == ()
    assert report.skipped_unmatched_tensors == ()
    assert report.mtp_bound_count == 0
    assert report.bound_count == report.backbone_bound_count
    # Every FP8 resident went through the block decoder.
    assert any(name.endswith(".attn.wq_b.weight") for name in report.fp8_block_decoded_tensors)
    assert any(
        name.endswith(".ffn.shared_experts.w1.weight")
        for name in report.fp8_block_decoded_tensors
    )
    # Routed experts skipped, never bound.
    assert len(report.skipped_routed_expert_tensors) == (
        args.num_hidden_layers * args.n_routed_experts * 3
    )
    assert dense_deepseek_v4_flash_routed_parameter_names(model) == ()
    # tid2eid kept its native int64.
    assert report.preserved_integer_tensors == ("layers.0.ffn.gate.tid2eid",)
    assert model.model.layers[0].ffn.gate.tid2eid.dtype == mx.int64
    # Decoded residents are finite and in the requested compute dtype.
    wq_b = model.model.layers[0].attn.wq_b.weight
    assert wq_b.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(wq_b)))


def test_non_vq_bind_maps_shared_experts_and_hyper_connections(tmp_path):
    args = _tiny_args(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
    _write_synthetic_checkpoint(tmp_path, args)
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, tmp_path, include_mtp=False, strict=True
    )
    bound = set(report.bound_backbone_parameters)

    # w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj
    assert "model.layers.0.ffn.shared_experts.gate_proj.weight" in bound
    assert "model.layers.0.ffn.shared_experts.down_proj.weight" in bound
    assert "model.layers.0.ffn.shared_experts.up_proj.weight" in bound
    # hc_attn_* -> attn_hc.*, hc_ffn_* -> ffn_hc.*
    assert "model.layers.0.attn_hc.fn" in bound
    assert "model.layers.0.ffn_hc.scale" in bound
    # gate.bias -> gate.e_score_correction_bias on a scored layer
    assert "model.layers.0.ffn.gate.e_score_correction_bias" in bound
    # Top-level renames.
    assert "model.embed_tokens.weight" in bound
    assert "lm_head.weight" in bound
    assert "model.hc_head.fn" in bound


def test_non_vq_bind_reshapes_wo_a_into_the_multilinear_layout(tmp_path):
    args = _tiny_args(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
    _write_synthetic_checkpoint(tmp_path, args)
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, tmp_path, include_mtp=False, strict=True
    )

    assert "layers.0.attn.wo_a.weight" in report.reshaped_tensors
    assert model.model.layers[0].attn.wo_a.weight.shape == (
        args.o_groups,
        args.o_lora_rank,
        args.num_attention_heads * args.head_dim // args.o_groups,
    )


def test_reshape_is_restricted_to_the_wo_a_allowlist(tmp_path):
    """A numel-only reshape is a layout reinterpretation, so it is allowlisted.

    ``wo_a`` is the one tensor whose checkpoint layout legitimately differs from
    its module layout. Anywhere else a shape mismatch means the name mapping is
    wrong, and reshaping would scramble the weight rather than report it.
    """

    args = _tiny_args(num_hidden_layers=2, num_hash_layers=1, compress_ratios=[0, 4])
    _write_synthetic_checkpoint(tmp_path, args)
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, tmp_path, include_mtp=False, strict=True
    )

    assert report.reshaped_tensors == (
        "layers.0.attn.wo_a.weight",
        "layers.1.attn.wo_a.weight",
    )


def test_a_non_allowlisted_shape_mismatch_is_refused_not_reshaped(tmp_path):
    """Same element count, wrong axes, not wo_a: must raise."""

    args = _tiny_args(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
    _write_synthetic_checkpoint(tmp_path, args)
    _mutate_shard_header(
        tmp_path,
        # gate.weight is (n_routed_experts, hidden); transposing the declared
        # shape keeps the byte count identical.
        {"layers.0.ffn.gate.weight": {"shape": [args.hidden_size, args.n_routed_experts]}},
    )
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    with pytest.raises(ValueError, match="only attn.wo_a.weight may be reshaped"):
        bind_deepseek_v4_flash_non_vq_weights(model, tmp_path, include_mtp=False)


def test_fp8_codes_without_a_scale_companion_are_refused(tmp_path):
    """Undecoded FP8 bytes have the right shape and finite values.

    Binding them raw would look like a successful load and be wrong by whatever
    the block exponents were, so the missing companion must be fatal -- the
    mirror of the existing "scale on non-FP8 codes" guard.
    """

    args = _tiny_args(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
    _write_synthetic_checkpoint(tmp_path, args)
    _drop_from_index(tmp_path, ("layers.0.attn.wq_a.scale",))
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    with pytest.raises(ValueError, match="no .scale companion"):
        bind_deepseek_v4_flash_non_vq_weights(model, tmp_path, include_mtp=False)


def test_a_bare_block_scale_is_never_bound_as_a_value(tmp_path):
    from ramp.models.deepseek_v4_flash_adapter import _decode_source_tensor

    with pytest.raises(ValueError, match="only meaningful alongside"):
        _decode_source_tensor(
            "F8_E8M0", b"\x7f" * 4, (2, 2), None, compute_dtype=mx.bfloat16
        )


def test_layer_scoped_non_vq_bind_touches_only_that_layer(tmp_path):
    args = _tiny_args(num_hidden_layers=2, num_hash_layers=1, compress_ratios=[0, 0])
    _write_synthetic_checkpoint(tmp_path, args)
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, tmp_path, layers=(1,), include_mtp=False
    )

    assert all(
        name.startswith("model.layers.1.") for name in report.bound_backbone_parameters
    )
    assert report.bound_count > 0
    assert any(
        name.startswith("layers.0.") for name in report.skipped_out_of_scope_tensors
    )
    # Layer-scoping must not silently drag in the embedding or the head.
    assert "embed.weight" in report.skipped_out_of_scope_tensors


# ---------------------------------------------------------------------------
# MTP drafter binding
# ---------------------------------------------------------------------------


def _mtp_args(**overrides):
    return _tiny_args(
        num_hidden_layers=1,
        num_hash_layers=1,
        compress_ratios=[0, 0, 0, 0],
        dspark_target_layer_ids=[0, 0, 0],
        **overrides,
    )


def test_mtp_residents_bind_into_the_drafter_and_count_separately(tmp_path):
    args = _mtp_args()
    _write_synthetic_checkpoint(tmp_path, args, with_mtp=True)
    model = DeepseekV4FlashVQModel(args)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, tmp_path, include_mtp=True, strict=True
    )

    assert report.missing_model_parameters == ()
    assert report.skipped_unmatched_tensors == ()
    assert report.mtp_bound_count > 0
    assert report.backbone_bound_count > 0
    assert report.bound_count == report.mtp_bound_count + report.backbone_bound_count
    assert all(
        name.startswith("mtp_drafter.blocks.")
        for name in report.bound_mtp_parameters
    )
    assert all(
        not name.startswith("mtp_drafter.")
        for name in report.bound_backbone_parameters
    )

    bound = set(report.bound_mtp_parameters)
    # A full MoE block per stage, not a thin head.
    assert "mtp_drafter.blocks.0.attn.wq_b.weight" in bound
    assert "mtp_drafter.blocks.0.ffn.shared_experts.gate_proj.weight" in bound
    assert "mtp_drafter.blocks.0.attn_hc.fn" in bound
    assert "mtp_drafter.blocks.0.ffn.gate.e_score_correction_bias" in bound
    # Stage 0 fusion, last stage tail.
    assert "mtp_drafter.blocks.0.main_proj.weight" in bound
    assert "mtp_drafter.blocks.0.main_norm.weight" in bound
    assert "mtp_drafter.blocks.2.norm.weight" in bound
    assert "mtp_drafter.blocks.2.hc_head.fn" in bound
    assert "mtp_drafter.blocks.2.markov_head.markov_w1.weight" in bound
    assert "mtp_drafter.blocks.2.confidence_head.proj.weight" in bound

    # Drafter routed experts are counted apart from the backbone's.
    per_block = args.n_routed_experts * 3
    assert len(report.skipped_mtp_routed_expert_tensors) == per_block * 3
    assert len(report.skipped_routed_expert_tensors) == per_block
    assert all(
        name.startswith("mtp.") for name in report.skipped_mtp_routed_expert_tensors
    )


def test_mtp_can_be_deliberately_excluded_and_is_reported_as_skipped(tmp_path):
    args = _mtp_args()
    _write_synthetic_checkpoint(tmp_path, args, with_mtp=True)
    model = DeepseekV4FlashVQModel(args)

    report = bind_deepseek_v4_flash_non_vq_weights(model, tmp_path, include_mtp=False)

    assert report.mtp_bound_count == 0
    assert report.bound_mtp_parameters == ()
    assert report.skipped_mtp_tensors
    assert all(name.startswith("mtp.") for name in report.skipped_mtp_tensors)
    # Excluding the drafter leaves its parameters unbound; the report says so
    # instead of pretending the model is complete.
    assert any(
        name.startswith("mtp_drafter.")
        for name in report.missing_model_parameters
    )


def test_mtp_routed_experts_bind_densely_from_fp4(tmp_path):
    from ramp.models.deepseek_v4_flash_adapter import (
        bind_deepseek_v4_flash_mtp_dense_experts,
        has_unbound_deepseek_v4_flash_mtp_experts,
    )

    args = _mtp_args()
    _write_synthetic_checkpoint(tmp_path, args, with_mtp=True)
    model = DeepseekV4FlashVQModel(args)

    assert has_unbound_deepseek_v4_flash_mtp_experts(model) is True
    bound = bind_deepseek_v4_flash_mtp_dense_experts(model, tmp_path, stages=(0,))

    assert bound == {0: args.n_routed_experts}
    switch = model.mtp_drafter.blocks[0].ffn.switch_mlp
    assert switch is not None
    # w1 -> gate, w2 -> down, w3 -> up.
    assert switch.gate_proj.weight.shape == (
        args.n_routed_experts,
        args.moe_intermediate_size,
        args.hidden_size,
    )
    assert switch.down_proj.weight.shape == (
        args.n_routed_experts,
        args.hidden_size,
        args.moe_intermediate_size,
    )
    assert switch.up_proj.weight.shape == switch.gate_proj.weight.shape
    mx.eval(switch.gate_proj.weight)
    assert bool(mx.all(mx.isfinite(switch.gate_proj.weight)))

    # Stages 1 and 2 stay unbound; the drafter audit reports that honestly.
    assert has_unbound_deepseek_v4_flash_mtp_experts(model) is True
    # Upstream's drafter builds its SwitchGLU with LimitedSwiGLU too.
    assert isinstance(switch.activation, LimitedSwiGLU)
    assert switch.activation.limit == args.swiglu_limit
    # A dense drafter stack *is* dense routed storage -- the backbone audit must
    # not be fooled into calling it clean when asked to include the drafter.
    assert dense_deepseek_v4_flash_routed_parameter_names(model) == ()
    assert dense_deepseek_v4_flash_routed_parameter_names(model, include_mtp=True) != ()


def test_drafter_fp4_read_refuses_a_mismatched_declared_dtype(tmp_path):
    """Reading code bytes is a reinterpret, so the declared dtype must agree.

    Handing FP8 bytes to the FP4 decoder (or vice versa) yields a full-size,
    finite, entirely wrong expert.
    """

    from ramp.models.deepseek_v4_flash_adapter import (
        bind_deepseek_v4_flash_mtp_dense_experts,
    )

    args = _mtp_args()
    _write_synthetic_checkpoint(tmp_path, args, with_mtp=True)
    _mutate_shard_header(
        tmp_path,
        {"mtp.0.ffn.experts.0.w1.weight": {"dtype": "F8_E4M3"}},
    )
    model = DeepseekV4FlashVQModel(args)

    with pytest.raises(ValueError, match="declared F8_E4M3, expected I8"):
        bind_deepseek_v4_flash_mtp_dense_experts(model, tmp_path, stages=(0,))


def test_drafter_fp4_read_refuses_a_mismatched_scale_dtype(tmp_path):
    from ramp.models.deepseek_v4_flash_adapter import (
        bind_deepseek_v4_flash_mtp_dense_experts,
    )

    args = _mtp_args()
    _write_synthetic_checkpoint(tmp_path, args, with_mtp=True)
    _mutate_shard_header(
        tmp_path,
        {"mtp.0.ffn.experts.0.w1.scale": {"dtype": "I8"}},
    )
    model = DeepseekV4FlashVQModel(args)

    with pytest.raises(ValueError, match="declared I8, expected F8_E8M0"):
        bind_deepseek_v4_flash_mtp_dense_experts(model, tmp_path, stages=(0,))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_the_real_deepseek_v4_flash_symbols():
    from ramp.models import registry

    resolved = registry.resolve_family("deepseek_v4_flash")

    assert resolved.model_args is DeepseekV4FlashVQModelArgs
    assert resolved.bind_vq_experts is bind_deepseek_v4_flash_vq_experts
    assert resolved.bind_non_vq_weights is bind_deepseek_v4_flash_non_vq_weights
    assert resolved.has_unbound is has_unbound_deepseek_v4_flash_vq_experts
    for func in (
        resolved.bind_vq_experts,
        resolved.bind_non_vq_weights,
        resolved.has_unbound,
    ):
        assert callable(func)


def test_registry_model_args_build_from_the_real_config():
    from ramp.models import registry

    resolved = registry.resolve_family("deepseek_v4_flash")
    config = _real_config()
    args = resolved.model_args(
        **{
            name: config[name]
            for name in (
                "num_hidden_layers",
                "hidden_size",
                "n_routed_experts",
                "compress_ratios",
            )
        }
    )
    assert args.num_hidden_layers == 43
    assert len(args.compress_ratios) == 43


# ---------------------------------------------------------------------------
# Real checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
@requires_checkpoint
def test_real_config_on_disk_matches_the_frozen_copy():
    live = json.loads((CHECKPOINT / "config.json").read_text())
    assert live == _real_config()


@pytest.mark.checkpoint
@requires_checkpoint
def test_bind_layer_zero_residents_from_the_real_checkpoint():
    """Layer 0 only: no embedding, no head, no experts.

    Measured on this machine: ~0.3 s and under 1 GB peak, because the bind
    evaluates only the tensors it just decoded and never forces the rest of the
    lazily-initialised skeleton.
    """

    args = deepseek_v4_flash_args_from_config(_real_config())
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    report = bind_deepseek_v4_flash_non_vq_weights(
        model, CHECKPOINT, layers=(0,), include_mtp=False
    )

    # 29 layer-0 non-expert tensors, 8 of which are ``.scale`` companions
    # consumed with their ``.weight``.
    assert report.bound_count == 21
    assert len(report.fp8_block_decoded_tensors) == 8
    assert report.skipped_unmatched_tensors == ()

    bound = set(report.bound_backbone_parameters)
    assert "model.layers.0.attn.wq_b.weight" in bound
    assert "model.layers.0.ffn.shared_experts.gate_proj.weight" in bound
    assert "model.layers.0.ffn.gate.tid2eid" in bound
    assert report.preserved_integer_tensors == ("layers.0.ffn.gate.tid2eid",)

    attn = model.model.layers[0].attn
    assert attn.wq_a.weight.shape == (args.q_lora_rank, args.hidden_size)
    assert attn.wq_b.weight.shape == (
        args.num_attention_heads * args.head_dim,
        args.q_lora_rank,
    )
    assert attn.wkv.weight.shape == (args.head_dim, args.hidden_size)
    assert attn.wo_a.weight.shape == (
        args.o_groups,
        args.o_lora_rank,
        args.num_attention_heads * args.head_dim // args.o_groups,
    )
    assert attn.wo_b.weight.shape == (
        args.hidden_size,
        args.o_groups * args.o_lora_rank,
    )
    shared = model.model.layers[0].ffn.shared_experts
    assert shared.gate_proj.weight.shape == (
        args.moe_intermediate_size,
        args.hidden_size,
    )
    assert shared.down_proj.weight.shape == (
        args.hidden_size,
        args.moe_intermediate_size,
    )

    gate = model.model.layers[0].ffn.gate
    assert gate.tid2eid.dtype == mx.int64
    assert gate.tid2eid.shape == (args.vocab_size, args.num_experts_per_tok)
    assert int(mx.max(gate.tid2eid)) < args.n_routed_experts
    assert int(mx.min(gate.tid2eid)) >= 0

    for tensor in (
        attn.wq_a.weight,
        attn.wq_b.weight,
        attn.wkv.weight,
        attn.wo_a.weight,
        attn.wo_b.weight,
        shared.gate_proj.weight,
        shared.down_proj.weight,
        shared.up_proj.weight,
    ):
        mx.eval(tensor)
        assert bool(mx.all(mx.isfinite(tensor))), "FP8 block decode produced non-finite"
        assert float(mx.max(mx.abs(tensor))) > 0.0


def test_full_skeleton_from_the_real_config_has_the_measured_structure():
    """Structural parity on the real 43-layer config.

    Not ``checkpoint``-marked: it needs the frozen config only. MLX arrays are
    lazy graph nodes until evaluated, so building the whole skeleton (backbone
    plus drafter) costs milliseconds and about 100 MB -- nothing here forces a
    single parameter, which is precisely the property being asserted.
    """

    args = deepseek_v4_flash_args_from_config(_real_config())
    model = DeepseekV4FlashVQModel(args)

    assert len(model.model.layers) == 43
    assert model.mtp_drafter is not None
    assert len(model.mtp_drafter.blocks) == 3

    ratios = [layer.attn.compress_ratio for layer in model.model.layers]
    assert ratios == args.compress_ratios
    assert ratios.count(0) == 2  # layers 0 and 1
    indexed = [
        i
        for i, layer in enumerate(model.model.layers)
        if isinstance(layer.attn, SparseCompressedAttention)
    ]
    assert indexed == [i for i, r in enumerate(ratios) if r == 4]

    hash_layers = [
        i for i, layer in enumerate(model.model.layers) if layer.ffn.gate.hash
    ]
    assert hash_layers == [0, 1, 2]

    assert has_unbound_deepseek_v4_flash_vq_experts(model) is True
    assert dense_deepseek_v4_flash_routed_parameter_names(model, include_mtp=True) == ()


# Measured on revision 7872f01b. Written as an arithmetic identity rather than
# three magic numbers so a structural change fails loudly instead of being
# papered over by editing a constant.
_CHECKPOINT_TENSOR_COUNT = 72_317
# 46 MoE blocks (43 backbone + 3 drafter) x 256 experts x 3 projections,
# each contributing a .weight and a .scale.
_ROUTED_EXPERT_TENSOR_COUNT = 46 * 256 * 3 * 2
# Block-scale companions of the FP8 residents, consumed with their .weight.
_RESIDENT_SCALE_COUNT = 390
_MAPPED_TENSOR_COUNT = (
    _CHECKPOINT_TENSOR_COUNT - _ROUTED_EXPERT_TENSOR_COUNT - _RESIDENT_SCALE_COUNT
)


@pytest.mark.checkpoint
@requires_checkpoint
def test_checkpoint_tensors_and_model_parameters_are_a_bijection():
    """Every source tensor is mapped or deliberately skipped, and vice versa.

    A one-directional check would pass while half the model stayed unbound, and
    a check without totals would pass if the walk silently visited nothing. So
    this asserts the full partition of all 72,317 index entries, that the mapped
    side covers every model parameter exactly once, and that no two source
    tensors claim the same parameter.
    """

    from ramp.models.deepseek_v4_flash_adapter import (
        _checkpoint_layer_index,
        _checkpoint_mtp_stage,
        _is_routed_expert_tensor,
        _map_checkpoint_name,
        deepseek_v4_flash_checkpoint_tensor_names,
    )

    args = deepseek_v4_flash_args_from_config(_real_config())
    model = DeepseekV4FlashVQModel(args)
    targets = {key for key, _ in tree_flatten(model.parameters())}

    names = deepseek_v4_flash_checkpoint_tensor_names(CHECKPOINT)
    assert len(names) == _CHECKPOINT_TENSOR_COUNT

    routed: list[str] = []
    scales: list[str] = []
    unmapped: list[str] = []
    claimed: dict[str, str] = {}
    for name in names:
        # Use the production predicate, not a hardcoded substring: if the skip
        # rule and the test's idea of it ever diverge, that is the bug.
        if _is_routed_expert_tensor(name):
            routed.append(name)
            continue
        if name.endswith(".scale"):
            scales.append(name)
            continue
        target = _map_checkpoint_name(name)
        if target is None or target not in targets:
            unmapped.append(name)
            continue
        assert target not in claimed, (
            f"{name} and {claimed[target]} both map to {target}"
        )
        claimed[target] = name

    assert unmapped == [], unmapped
    assert len(routed) == _ROUTED_EXPERT_TENSOR_COUNT
    assert len(scales) == _RESIDENT_SCALE_COUNT
    assert len(claimed) == _MAPPED_TENSOR_COUNT == 1_271
    # The walk actually did something.
    assert len(claimed) > 0
    # Partition: nothing counted twice, nothing missed.
    assert len(routed) + len(scales) + len(claimed) == len(names)

    # Reverse direction: no model parameter was left without a source tensor.
    assert sorted(targets - set(claimed)) == []
    assert len(targets) == _MAPPED_TENSOR_COUNT

    # Routed experts really are the VQ path, and every drafter stage is present.
    assert all(_is_routed_expert_tensor(name) for name in routed)
    assert sum(1 for name in routed if name.startswith("mtp.")) == 3 * 256 * 3 * 2

    # And the drafter stages resolve where we say they do.
    assert _checkpoint_mtp_stage("mtp.2.markov_head.markov_w1.weight") == 2
    assert _checkpoint_layer_index("layers.42.attn.wkv.weight") == 42


# ---------------------------------------------------------------------------
# DSpark drafter forward (increment 3)
# ---------------------------------------------------------------------------
#
# Increment 1 bound the ``mtp.{0,1,2}`` weights and increment 2 ran the backbone;
# neither ever executed the drafter. These tests are the does-it-run gate plus
# the three things binding and shapes cannot express: the drafter's SwiGLU is
# float32 where the backbone's is not, its attention is non-causal *inside the
# draft block*, and ``main_proj`` consumes exactly the ``dspark_target_layer_ids``
# taps in that list's order.

_DRAFT_PROMPT = 40


def _drafter_args(**overrides) -> DeepseekV4FlashVQModelArgs:
    """A 4-layer backbone plus a 3-stage drafter targeting layers [1, 2, 3]."""

    kwargs = {
        "vocab_size": 64,
        "num_hidden_layers": 4,
        # Four backbone ratios (all three attention variants) then three
        # drafter-tail zeros, exactly like the released 46-for-43 list.
        "compress_ratios": [0, 4, 128, 4, 0, 0, 0],
        "num_hash_layers": 2,
        "n_routed_experts": 8,
        "sliding_window": 16,
        "max_position_embeddings": 4096,
        "rope_scaling": _FORWARD_ROPE_SCALING,
        "hc_sinkhorn_iters": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 3,
        "dspark_target_layer_ids": [1, 2, 3],
        "dspark_markov_rank": 4,
    }
    kwargs.update(overrides)
    return _tiny_args(**kwargs)


def _dense_switch_glu(args, rng, *, fp32: bool):
    from mlx_lm.models.switch_layers import SwitchGLU

    switch = SwitchGLU(
        args.hidden_size,
        args.moe_intermediate_size,
        args.n_routed_experts,
        activation=LimitedSwiGLU(args.swiglu_limit, fp32=fp32),
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        linear = getattr(switch, projection)
        linear.weight = mx.array(
            rng.normal(scale=0.1, size=linear.weight.shape).astype(np.float32)
        )
    return switch


def _drafter_model(*, seed: int = 20260813, drafter_fp32: bool = True, **overrides):
    """A whole model whose backbone *and* drafter can actually run.

    Routed experts are dense here rather than VQ: the drafter's experts have no
    VQ artifact (Wave 5 decides whether they get one), and the point of these
    tests is the drafter's own arithmetic, not the routed codec.
    """

    args = _drafter_args(**overrides)
    model = DeepseekV4FlashVQModel(args)
    _randomize(model, seed=seed)
    rng = np.random.default_rng(seed)
    for layer in model.model.layers:
        layer.ffn.switch_mlp = _dense_switch_glu(args, rng, fp32=False)
    for block in model.mtp_drafter.blocks:
        if not drafter_fp32:
            block.ffn.fp32_swiglu = False
            block.ffn.shared_experts.fp32_swiglu = False
        block.ffn.switch_mlp = _dense_switch_glu(args, rng, fp32=drafter_fp32)
    mx.eval(model.parameters())
    return args, model


def _backbone_taps(model, ids):
    logits, taps = model(ids, cache=model.make_cache(), return_dspark_hidden=True)
    mx.eval(logits, taps)
    return logits, taps


def test_drafter_forward_runs_and_stays_finite():
    args, model = _drafter_model()
    ids = _prompt(args, length=_DRAFT_PROMPT)
    _logits, taps = _backbone_taps(model, ids)

    assert taps.shape == (
        1,
        _DRAFT_PROMPT,
        args.hidden_size * len(args.dspark_target_layer_ids),
    )

    cache = model.make_mtp_cache()
    assert len(cache) == len(model.mtp_drafter.blocks)

    draft_logits, head_hidden = model.dspark_forward(taps[:, :-1], ids[:, -1:], cache)
    mx.eval(draft_logits, head_hidden)

    width = args.dspark_block_size
    assert draft_logits.shape == (1, width, args.vocab_size)
    assert head_hidden.shape == (1, width, args.hidden_size)
    assert bool(mx.all(mx.isfinite(draft_logits)))
    assert bool(mx.all(mx.isfinite(head_hidden)))
    # Not a constant field, and not the same proposal at every slot.
    assert float(mx.std(draft_logits)) > 1e-4
    assert float(mx.max(mx.abs(draft_logits[:, 0] - draft_logits[:, -1]))) > 1e-4

    # Only committed context lands in the ring; the draft block never does.
    assert [entry.offset for entry in cache] == [_DRAFT_PROMPT - 1] * len(cache)

    # The Markov head and the confidence head are reachable and finite.
    bias, embedding = model.dspark_markov(ids[:, -1:])
    mx.eval(bias, embedding)
    assert bias.shape == (1, 1, args.vocab_size)
    assert embedding.shape == (1, 1, args.dspark_markov_rank)
    confidence = model.dspark_confidence(
        head_hidden,
        mx.broadcast_to(embedding, (1, width, args.dspark_markov_rank)),
    )
    mx.eval(confidence)
    assert confidence.shape == (1, width)
    assert bool(mx.all(mx.isfinite(confidence)))


def test_draft_length_shrinks_the_block_and_query_width_only_the_query():
    """Two different narrowings, and they are not the same narrowing.

    ``draft_length`` builds a *shorter block*: fewer slots, therefore fewer
    keys, therefore a different (not truncated) answer -- which is exactly what
    the reference does (``deepseek_v4_model.py:495-516``: the noise block is
    ``width`` wide, so its ``output_width=width`` is a no-op).

    ``output_width`` on the stage narrows only the *query* and the MoE that
    follows it, leaving the whole block as keys, so its rows must come out
    bit-identical to the corresponding rows of the full-width stage. Both halves
    matter: if the drafter had truncated the block before attention, the second
    assertion would fail, and if it had never narrowed at all the first would.
    """

    args, model = _drafter_model()
    ids = _prompt(args, length=_DRAFT_PROMPT)
    _logits, taps = _backbone_taps(model, ids)

    narrow, _ = model.dspark_forward(
        taps[:, :-1], ids[:, -1:], model.make_mtp_cache(), draft_length=2
    )
    full, _ = model.dspark_forward(taps[:, :-1], ids[:, -1:], model.make_mtp_cache())
    mx.eval(narrow, full)

    assert narrow.shape == (1, 2, args.vocab_size)
    assert full.shape == (1, args.dspark_block_size, args.vocab_size)
    # Fewer keys is a different function, not a prefix of the wider one.
    assert not bool(mx.array_equal(narrow, full[:, :2]))
    # ...but a closely related one: nowhere near an unrelated proposal.
    gap = float(mx.max(mx.abs(narrow - full[:, :2])))
    assert 0 < gap < float(mx.std(full))

    # Now the query-width narrowing, at the stage that owns it.
    rng = np.random.default_rng(71)
    block = mx.array(
        rng.normal(
            size=(1, args.dspark_block_size, args.hc_mult, args.hidden_size)
        ).astype(np.float32)
    )
    block_ids = mx.array(
        rng.integers(0, args.vocab_size, size=(1, args.dspark_block_size))
    )
    stage = model.mtp_drafter.blocks[-1]
    wide = stage(block, block_ids, DSparkContextCache(args.sliding_window))
    thin = stage(
        block, block_ids, DSparkContextCache(args.sliding_window), output_width=2
    )
    mx.eval(wide, thin)
    assert wide.shape[1] == args.dspark_block_size
    assert thin.shape[1] == 2
    assert bool(mx.array_equal(thin, wide[:, :2]))


def test_drafter_swiglu_runs_in_float32_and_the_backbone_does_not(monkeypatch):
    """The reference sets ``fp32=True`` / ``fp32_swiglu=True`` on the drafter only.

    Asserted twice, because either alone is weak: the dtype *at the activation*
    (a recorder wrapped round the module-level ``_limited_swiglu``), and that
    the fp32 path's answer differs from a bf16 path's by far more than a re-run
    of the same path differs from itself.
    """

    import ramp.models.deepseek_v4_flash_adapter as adapter

    seen: list[str] = []
    real = adapter._limited_swiglu

    def recorder(gate, up, limit):
        seen.append(f"{gate.dtype}|{up.dtype}")
        return real(gate, up, limit)

    monkeypatch.setattr(adapter, "_limited_swiglu", recorder)

    args, model = _drafter_model()
    ids = _prompt(args, length=_DRAFT_PROMPT)

    seen.clear()
    _logits, taps = _backbone_taps(model, ids)
    assert seen, "the backbone never reached a SwiGLU"

    seen.clear()
    draft, _ = model.dspark_forward(taps[:, :-1], ids[:, -1:], model.make_mtp_cache())
    mx.eval(draft)
    assert seen, "the drafter never reached a SwiGLU"
    # Every drafter SwiGLU -- routed and shared, all three stages -- is float32.
    assert set(seen) == {f"{mx.float32}|{mx.float32}"}

    monkeypatch.undo()

    # Numerics, not just dtypes: run the same drafter in bfloat16 with and
    # without the fp32 promotion and show the answers separate. In float32
    # parameters the promotion is a no-op, which is why this half has to cast.
    def to_bf16(model):
        model.update(
            tree_unflatten(
                [
                    (key, value.astype(mx.bfloat16))
                    for key, value in tree_flatten(model.parameters())
                    if mx.issubdtype(value.dtype, mx.floating)
                ]
            )
        )
        mx.eval(model.parameters())

    _a32, fp32_model = _drafter_model(drafter_fp32=True)
    _a16, bf16_model = _drafter_model(drafter_fp32=False)
    to_bf16(fp32_model)
    to_bf16(bf16_model)

    _l, taps32 = _backbone_taps(fp32_model, ids)
    context = taps32.astype(mx.bfloat16)[:, :-1]
    fp32_draft, _ = fp32_model.dspark_forward(
        context, ids[:, -1:], fp32_model.make_mtp_cache()
    )
    bf16_draft, _ = bf16_model.dspark_forward(
        context, ids[:, -1:], bf16_model.make_mtp_cache()
    )
    rerun, _ = fp32_model.dspark_forward(
        context, ids[:, -1:], fp32_model.make_mtp_cache()
    )
    mx.eval(fp32_draft, bf16_draft, rerun)

    fp32_draft = fp32_draft.astype(mx.float32)
    bf16_draft = bf16_draft.astype(mx.float32)
    rerun = rerun.astype(mx.float32)

    # Same path twice is exactly reproducible -- so the gap below is the fp32
    # promotion and nothing else.
    assert bool(mx.array_equal(fp32_draft, rerun))
    scale = float(mx.max(mx.abs(fp32_draft)))
    gap = float(mx.max(mx.abs(fp32_draft - bf16_draft)))
    assert gap > 1e-3 * scale, (gap, scale)


def test_drafter_refuses_a_routed_module_whose_activation_is_not_fp32():
    """An fp32/bf16 activation mismatch binds, runs, and is silently wrong."""

    args = _drafter_args()
    model = DeepseekV4FlashVQModel(args)
    rng = np.random.default_rng(5)
    gate = mx.array(
        rng.normal(
            scale=0.04,
            size=(args.n_routed_experts, args.moe_intermediate_size, args.hidden_size),
        ).astype(np.float32)
    )
    down = mx.array(
        rng.normal(
            scale=0.04,
            size=(args.n_routed_experts, args.hidden_size, args.moe_intermediate_size),
        ).astype(np.float32)
    )
    switch = QuantizedVQSwitchGLU.from_weights(
        gate_weight=gate, up_weight=gate, down_weight=down, group_size=8
    )

    switch.activation = LimitedSwiGLU(args.swiglu_limit, fp32=False)
    with pytest.raises(ValueError, match="fp32"):
        model.mtp_drafter.blocks[0].ffn.bind_switch_mlp(switch)

    switch.activation = LimitedSwiGLU(args.swiglu_limit, fp32=True)
    model.mtp_drafter.blocks[0].ffn.bind_switch_mlp(switch)
    assert model.mtp_drafter.blocks[0].ffn.switch_mlp is switch

    # And the reverse: a backbone layer refuses the fp32 one.
    with pytest.raises(ValueError, match="fp32"):
        model.model.layers[0].ffn.bind_switch_mlp(switch)


def test_dspark_attention_is_non_causal_inside_the_draft_block():
    """Slot 0 of the block must see slot ``W-1``. Under a causal mask it cannot.

    Asserted behaviourally rather than by reading ``mask=None`` off the call:
    perturb only the *last* block position's input and check the *first* query
    position's output moves. The same perturbation then goes through the
    backbone's causal ``LocalAttention`` -- where position 0 must not move at
    all -- so the test discriminates instead of merely passing.
    """

    args, model = _drafter_model()
    attn = model.mtp_drafter.blocks[0].attn
    assert isinstance(attn, DSparkAttention)

    rng = np.random.default_rng(17)
    width = args.dspark_block_size
    block = mx.array(rng.normal(size=(1, width, args.hidden_size)).astype(np.float32))
    bump = mx.array(np.full((1, 1, args.hidden_size), 3.0, np.float32))
    bumped = mx.concatenate([block[:, :-1], block[:, -1:] + bump], axis=1)

    base = attn(block, DSparkContextCache(args.sliding_window))
    moved = attn(bumped, DSparkContextCache(args.sliding_window))
    mx.eval(base, moved)

    first_position_delta = float(mx.max(mx.abs(moved[:, 0] - base[:, 0])))
    assert first_position_delta > 1e-4, first_position_delta
    # Every position moves, not only the perturbed one: the block is a single
    # all-to-all attention.
    assert float(mx.min(mx.max(mx.abs(moved - base), axis=-1))) > 1e-4

    # Control: the backbone's causal attention over the identical inputs leaves
    # position 0 bit-identical...
    causal = model.model.layers[0].attn
    assert isinstance(causal, LocalAttention)
    assert not isinstance(causal, DSparkAttention)
    mask = create_attention_mask(
        block, None, window_size=args.sliding_window, return_array=True
    )
    causal_base = causal(block, mask=mask, cache=None)
    causal_moved = causal(bumped, mask=mask, cache=None)
    mx.eval(causal_base, causal_moved)
    assert bool(mx.array_equal(causal_base[:, 0], causal_moved[:, 0]))
    # ...while its last position does move, so the control is not vacuous.
    assert float(mx.max(mx.abs(causal_moved[:, -1] - causal_base[:, -1]))) > 1e-4


def test_calibration_forward_excludes_the_anchors_own_tap():
    """The context seam, pinned where the convention lives.

    ``dspark_calibration_forward`` (reference: ``deepseek_v4_model.py:540-554``)
    takes a whole prefix and slices it itself: context ``0 .. p-1``, anchor at
    ``p``. Committing the anchor's own tap too would leave the ring one token
    long, shift every draft slot's RoPE by one, and condition the drafter on a
    backbone hidden state it cannot have at inference -- ``dspark.py:204-209``
    is the same invariant from the runtime side (``take_primed`` refuses unless
    the ring is exactly one behind the target cache).
    """

    args, model = _drafter_model()
    ids = _prompt(args, length=_DRAFT_PROMPT)
    _logits, taps = _backbone_taps(model, ids)
    last = _DRAFT_PROMPT - 1

    reference, hidden = model.dspark_calibration_forward(taps, ids)
    mx.eval(reference, hidden)
    assert reference.shape == (1, args.dspark_block_size, args.vocab_size)

    # Equal to the hand-built correct seam...
    correct_cache = model.make_mtp_cache()
    model.dspark_append_context(taps[:, :last], correct_cache)
    assert [entry.offset for entry in correct_cache] == [last] * len(correct_cache)
    correct, _ = model.dspark_draft_block(
        ids[:, last:], correct_cache, draft_length=args.dspark_block_size
    )
    mx.eval(correct)
    assert bool(mx.array_equal(reference, correct))

    # ...and NOT equal to the off-by-one, which also runs and stays finite.
    skewed_cache = model.make_mtp_cache()
    model.dspark_append_context(taps, skewed_cache)
    assert [entry.offset for entry in skewed_cache] == [_DRAFT_PROMPT] * len(
        skewed_cache
    )
    skewed, _ = model.dspark_draft_block(
        ids[:, last:], skewed_cache, draft_length=args.dspark_block_size
    )
    mx.eval(skewed)
    assert bool(mx.all(mx.isfinite(skewed)))
    assert not bool(mx.array_equal(reference, skewed))
    assert float(mx.max(mx.abs(reference - skewed))) > 1e-4

    # Mismatched taps and ids are a caller bug, not something to slice around.
    with pytest.raises(ValueError, match="one target tap per input token"):
        model.dspark_calibration_forward(taps[:, :-1], ids)


def test_dspark_context_is_the_committed_past_and_the_block_attends_it():
    """The ring holds committed context only, and the draft actually reads it."""

    args, model = _drafter_model()
    ids = _prompt(args, length=_DRAFT_PROMPT)
    _logits, taps = _backbone_taps(model, ids)

    empty = model.make_mtp_cache()
    without, _ = model.dspark_draft_block(ids[:, -1:], empty)

    primed = model.make_mtp_cache()
    model.dspark_append_context(taps[:, :-1], primed)
    with_context, _ = model.dspark_draft_block(ids[:, -1:], primed)
    mx.eval(without, with_context)

    assert [entry.offset for entry in empty] == [0] * len(empty)
    assert all(entry.keys is None for entry in empty)
    # The ring is capped at the sliding window and holds committed tokens only.
    assert [entry.offset for entry in primed] == [_DRAFT_PROMPT - 1] * len(primed)
    assert all(entry.keys.shape[2] == args.sliding_window for entry in primed)

    gap = float(mx.max(mx.abs(with_context - without)))
    assert gap > 1e-3 * float(mx.max(mx.abs(with_context))), gap

    # A non-contiguous append is refused rather than silently mis-positioned.
    with pytest.raises(ValueError, match="not contiguous"):
        model.dspark_append_context(taps[:, :2], primed, start_offset=0)

    # ...and so is a context of the wrong fused width.
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        model.dspark_append_context(taps[..., : args.hidden_size], model.make_mtp_cache())


def test_main_proj_consumes_exactly_the_target_layers_in_order():
    """The taps are layers ``dspark_target_layer_ids``, concatenated in *that* order.

    Checked against independently recomputed per-layer hidden states, and
    against the *wrong* layers, so a capture that grabbed layers ``[0, 1]`` or
    silently sorted the ids would fail rather than merely produce the right
    shape.
    """

    import ramp.models.deepseek_v4_flash_adapter as adapter

    # Deliberately out of ascending order and skipping layer 2.
    args, model = _drafter_model(dspark_target_layer_ids=[3, 1])
    assert args.main_proj_in_features == 2 * args.hidden_size
    assert model.mtp_drafter.blocks[0].main_proj.weight.shape == (
        args.hidden_size,
        2 * args.hidden_size,
    )
    assert len(model.mtp_drafter.blocks) == 2

    ids = _prompt(args, length=_DRAFT_PROMPT)
    _logits, taps = _backbone_taps(model, ids)
    assert taps.shape == (1, _DRAFT_PROMPT, 2 * args.hidden_size)

    # Recompute every layer's output independently.
    cache = model.make_cache()
    h = model.model.embed_tokens(ids)
    h = mx.contiguous(
        mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], args.hc_mult, h.shape[2]))
    )
    first = cache[0]
    mask = create_attention_mask(
        h[:, :, 0, :],
        first[0] if isinstance(first, CacheList) else first,
        window_size=args.sliding_window,
        return_array=True,
    )
    per_layer = []
    for layer, layer_cache in zip(model.model.layers, cache):
        h = layer(h, mask, layer_cache, ids)
        per_layer.append(adapter._dspark_tap(h))
    mx.eval(per_layer)

    hidden = args.hidden_size
    assert bool(mx.array_equal(taps[..., :hidden], per_layer[3]))
    assert bool(mx.array_equal(taps[..., hidden:], per_layer[1]))
    # Teeth: it is not the ascending order, and not any other layer.
    assert not bool(mx.array_equal(taps[..., :hidden], per_layer[1]))
    assert not bool(mx.array_equal(taps[..., hidden:], per_layer[3]))
    for index in (0, 2):
        assert not bool(mx.array_equal(taps[..., :hidden], per_layer[index]))
        assert not bool(mx.array_equal(taps[..., hidden:], per_layer[index]))


def test_capture_refuses_targets_outside_the_backbone():
    args, model = _drafter_model(dspark_target_layer_ids=[1, 2])
    model.args.dspark_target_layer_ids = [1, 99]
    with pytest.raises(ValueError, match="outside the backbone"):
        model(
            _prompt(args, length=8),
            cache=model.make_cache(),
            return_dspark_hidden=True,
        )


def test_real_config_targets_backbone_layers_40_41_42():
    args = deepseek_v4_flash_args_from_config(_real_config())
    assert args.dspark_target_layer_ids == [40, 41, 42]
    assert all(0 <= i < args.num_hidden_layers for i in args.dspark_target_layer_ids)
    # The three targeted layers are the last three of the 43-layer backbone.
    assert args.dspark_target_layer_ids == list(
        range(args.num_hidden_layers - 3, args.num_hidden_layers)
    )
    assert args.main_proj_in_features == 3 * args.hidden_size == 12288
    assert args.num_mtp_blocks == 3

    # And the drafter built from it really does fuse three taps.
    model = DeepseekV4FlashVQModel(args)
    assert model.mtp_drafter.blocks[0].main_proj.weight.shape == (4096, 12288)
    assert len(model.mtp_drafter.blocks) == 3


def test_dspark_hidden_capture_is_opt_in_and_the_default_path_is_unchanged(monkeypatch):
    """Opting in must not be the price of the default forward staying honest.

    Three claims: the default returns a bare array (not a tuple), its logits are
    *bit-identical* to the opted-in path's, and the tap is not computed at all
    when the flag is off -- counted at the single call site rather than inferred
    from a memory number a busy machine can move.
    """

    import ramp.models.deepseek_v4_flash_adapter as adapter

    calls: list[tuple[int, ...]] = []
    real_tap = adapter._dspark_tap

    def counting_tap(h):
        calls.append(tuple(h.shape))
        return real_tap(h)

    monkeypatch.setattr(adapter, "_dspark_tap", counting_tap)

    args, model = _drafter_model()
    ids = _prompt(args, length=_DRAFT_PROMPT)

    calls.clear()
    plain = model(ids, cache=model.make_cache())
    mx.eval(plain)
    assert isinstance(plain, mx.array)
    assert calls == [], "the default forward computed DSpark taps"

    calls.clear()
    captured, tapped = model(ids, cache=model.make_cache(), return_dspark_hidden=True)
    mx.eval(captured, tapped)
    assert len(calls) == len(args.dspark_target_layer_ids)
    assert calls == [(1, _DRAFT_PROMPT, args.hc_mult, args.hidden_size)] * len(calls)

    # Same function, to the bit.
    assert bool(mx.array_equal(plain, captured))
    # ``mx.get_peak_memory()`` is deliberately *not* the instrument here: MLX's
    # allocator reuses freed blocks across runs, so the second forward of a pair
    # can peak lower than the first whatever it computes (measured: the plain
    # run peaked 3.01 MB, the capture run 2.94 MB, in that order). The call
    # counter above is the exact statement -- zero taps computed when the flag
    # is off -- and it cannot be moved by allocator history.

    # The backbone-level default is the same story.
    calls.clear()
    hidden_only = model.model(ids, cache=model.make_cache())
    mx.eval(hidden_only)
    assert isinstance(hidden_only, mx.array)
    assert calls == []


def test_capture_without_target_layer_ids_is_refused():
    args, model = _drafter_model(dspark_target_layer_ids=[1])
    model.args.dspark_target_layer_ids = []
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        model(
            _prompt(args, length=8),
            cache=model.make_cache(),
            return_dspark_hidden=True,
        )


def test_a_model_without_a_drafter_refuses_drafter_calls():
    args = _drafter_args()
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    assert model.mtp_drafter is None
    with pytest.raises(RuntimeError, match="with_mtp=False"):
        model.make_mtp_cache()


def test_drafter_stage_refuses_a_compressed_attention_ratio():
    """The released drafter tail is all zeros; a non-zero ratio is a config bug."""

    args = _drafter_args()
    with pytest.raises(ValueError, match="non-causal block"):
        DeepseekV4FlashMTPBlock(args, 0, compress_ratio=4, is_entry=True, is_exit=True)


# ---------------------------------------------------------------------------
# PoolingCache rollback (the increment-1 deferral, closed)
# ---------------------------------------------------------------------------


def _pooling_state(cache: PoolingCache):
    """Everything a rejected draft has to see restored, materialised."""

    def snapshot(value):
        if value is None:
            return None
        mx.eval(value)
        return np.array(value.astype(mx.float32))

    return {
        "remainder": cache.remainder,
        "pool_len": cache._pool_len,
        "buf_kv": snapshot(
            cache.buf_kv[:, : cache.remainder] if cache.remainder else None
        ),
        "buf_gate": snapshot(
            cache.buf_gate[:, : cache.remainder] if cache.remainder else None
        ),
        "pooled": snapshot(cache.pooled),
        "prev_win_kv": snapshot(cache.prev_win_kv),
        "prev_win_gate": snapshot(cache.prev_win_gate),
    }


def _assert_same_pooling_state(left, right):
    assert left["remainder"] == right["remainder"]
    assert left["pool_len"] == right["pool_len"]
    for key in ("buf_kv", "buf_gate", "pooled", "prev_win_kv", "prev_win_gate"):
        a, b = left[key], right[key]
        assert (a is None) == (b is None), key
        if a is not None:
            assert np.array_equal(a, b), key


_POOL_KV_DIM = 4
_POOL_GATE_DIM = 3


def _push_window(cache: PoolingCache, kv, gate, offset):
    """One update through the compressor's real call sequence."""

    windows_kv, windows_gate, _base = cache.accumulate_windows(kv, gate, offset)
    if windows_kv.shape[1] > 0:
        shaped_kv = windows_kv.reshape(1, -1, cache.ratio, _POOL_KV_DIM)
        shaped_gate = windows_gate.reshape(1, -1, cache.ratio, _POOL_GATE_DIM)
        cache.update_and_fetch(shaped_kv.mean(axis=2))
        cache.store_prev(shaped_kv, shaped_gate, 0)


def _feed(cache: PoolingCache, rng, count, *, offset):
    kv = mx.array(rng.normal(size=(1, count, _POOL_KV_DIM)).astype(np.float32))
    gate = mx.array(rng.normal(size=(1, count, _POOL_GATE_DIM)).astype(np.float32))
    _push_window(cache, kv, gate, offset)


def test_pooling_cache_reports_trimmable_only_when_it_can_actually_undo():
    rng = np.random.default_rng(11)
    cache = PoolingCache(4)

    # Nothing pooled yet: trivially trimmable.
    assert cache.is_trimmable() is True

    for step in range(6):
        _feed(cache, rng, 1, offset=step)
    assert cache.pooled is not None
    # Two tokens sit in the remainder buffer, so the last one is free to drop.
    assert cache.remainder == 2
    assert cache.is_trimmable() is True

    # A prompt-sized update stashes no undo record. 30 + the 2 already buffered
    # divides evenly by the ratio, so nothing is left in the remainder either --
    # the one state where the answer has to be an honest "no".
    _feed(cache, rng, 30, offset=6)
    assert cache.remainder == 0
    assert cache._undo is None
    assert cache.is_trimmable() is False
    with pytest.raises(ValueError, match="cannot trim 1"):
        cache.trim(1)

    # ...and the next decode token is trimmable again.
    _feed(cache, rng, 1, offset=36)
    assert cache.is_trimmable() is True


def test_pooling_cache_trim_restores_exact_prior_state():
    """A rejected draft must put the cache back bit-for-bit.

    Run to the hard case on purpose: the trimmed token is the one that
    *completed* a pooling window, so its ``ratio`` predecessors were already
    consumed out of the remainder buffer, compressed and appended. Nothing short
    of the undo log can put that back, which is why increment 1 refused to
    return ``True`` here without one.
    """

    rng = np.random.default_rng(23)
    cache = PoolingCache(4)
    for step in range(7):  # one completed window, three in the remainder
        _feed(cache, rng, 1, offset=step)
    assert cache.remainder == 3
    before = _pooling_state(cache)
    pooled_before = before["pool_len"]

    # The eighth token completes the second window.
    _feed(cache, rng, 1, offset=7)
    assert cache.remainder == 0
    assert cache._pool_len == pooled_before + 1
    assert cache.is_trimmable() is True

    assert cache.trim(1) == 1
    _assert_same_pooling_state(_pooling_state(cache), before)

    # The undo record is one update deep and is consumed by the trim; the next
    # trim succeeds only because three tokens are back in the remainder buffer.
    assert cache._undo is None
    assert cache.trim(1) == 1
    assert cache.remainder == 2


def test_pooling_cache_trim_then_replay_reaches_the_clean_state():
    """Undo then redo with a different token: no residue from the rejected one."""

    rng = np.random.default_rng(97)
    ratio = 4
    rejected = PoolingCache(ratio)
    clean = PoolingCache(ratio)

    accepted = [
        (
            mx.array(rng.normal(size=(1, 1, _POOL_KV_DIM)).astype(np.float32)),
            mx.array(rng.normal(size=(1, 1, _POOL_GATE_DIM)).astype(np.float32)),
        )
        for _ in range(9)
    ]

    for index, (kv, gate) in enumerate(accepted[:7]):
        _push_window(rejected, kv, gate, index)
        _push_window(clean, kv, gate, index)

    # The rejected timeline takes a wrong eighth token, then rolls it back.
    wrong_kv = mx.array(rng.normal(size=(1, 1, _POOL_KV_DIM)).astype(np.float32))
    wrong_gate = mx.array(rng.normal(size=(1, 1, _POOL_GATE_DIM)).astype(np.float32))
    _push_window(rejected, wrong_kv, wrong_gate, 7)
    assert rejected.trim(1) == 1

    # Both now take the accepted eighth and ninth tokens.
    for index, (kv, gate) in enumerate(accepted[7:], start=7):
        _push_window(rejected, kv, gate, index)
        _push_window(clean, kv, gate, index)

    _assert_same_pooling_state(_pooling_state(rejected), _pooling_state(clean))


def test_pooling_cache_raises_rather_than_half_trimming():
    """A trim it cannot cover must be loud, because ``CacheList`` hides it.

    ``CacheList.trim`` returns only its *last* sub-cache's result. A Wave-5
    block rollback trims a ``CacheList(RotatingKVCache, PoolingCache)`` by
    ``block_size - (accepted + 1)``, i.e. usually ``n > 1``: if this cache
    quietly returned 0 while ``RotatingKVCache`` dropped ``n``, the caller would
    see ``n``, believe both caches moved, and carry two different timelines
    forward with nothing to signal it. ``is_trimmable()`` cannot help -- it
    takes no ``n``.
    """

    rng = np.random.default_rng(41)
    cache = PoolingCache(4)
    for step in range(8):
        _feed(cache, rng, 1, offset=step)
    assert cache.remainder == 0
    before = _pooling_state(cache)

    # The undo record covers one update of one token; two is beyond it.
    assert cache.is_trimmable() is True  # ...for n == 1, which is all it claims
    with pytest.raises(ValueError, match="cannot trim 2"):
        cache.trim(2)
    _assert_same_pooling_state(_pooling_state(cache), before)
    # Refusing did not consume the record.
    assert cache.trim(1) == 1

    # A no-op trim is still a no-op, not a raise.
    assert cache.trim(0) == 0


def test_pooling_undo_window_is_derived_from_the_checkpoints_block_size():
    """``max_undo_update`` follows ``dspark_block_size``, not a magic constant.

    A verify step forwards the whole proposed block plus the anchor in one
    call, so the widest update a rollback ever has to undo is
    ``dspark_block_size + 1``. The vendored code hardcoded 8, which merely
    happens to exceed 6 on this release.
    """

    args = _drafter_args()  # dspark_block_size = 5
    assert deepseek_v4_flash_pooling_undo_window(args) == 6
    assert deepseek_v4_flash_pooling_undo_window(_drafter_args(dspark_block_size=11)) == 12
    # A config with no DSpark drafter promises decode only.
    assert deepseek_v4_flash_pooling_undo_window(_tiny_args(dspark_block_size=0)) == 1

    real = deepseek_v4_flash_args_from_config(_real_config())
    assert deepseek_v4_flash_pooling_undo_window(real) == real.dspark_block_size + 1

    # make_cache threads it through to every pooled cache...
    model = DeepseekV4FlashVQModel(args, with_mtp=False)
    pooled = [
        entry
        for cache in model.make_cache()
        if isinstance(cache, CacheList)
        for entry in cache
        if isinstance(entry, PoolingCache)
    ]
    assert pooled
    assert {entry.max_undo_update for entry in pooled} == {6}

    # ...and a bare cache promises only what a decode step needs.
    assert PoolingCache(4).max_undo_update == 1


def test_a_block_sized_update_is_undoable_only_with_the_derived_window():
    """The window is load-bearing: it decides whether a block can roll back."""

    rng = np.random.default_rng(53)
    ratio = 4
    block = 5  # dspark_block_size; a verify update is block + 1 tokens

    def run(cache):
        for step in range(7):
            _feed(cache, rng, 1, offset=step)
        _feed(cache, rng, block + 1, offset=7)
        return cache

    wide = run(PoolingCache(ratio, max_undo_update=block + 1))
    assert wide._undo is not None
    assert wide.trim(block) == block

    narrow = run(PoolingCache(ratio))  # decode-only default
    assert narrow._undo is None
    with pytest.raises(ValueError, match="cannot trim"):
        narrow.trim(block)


def test_rotated_target_cache_rolls_back_and_replays_accepted_prefix_exactly():
    """Wide verification must leave the same physical ring as M=1 decode."""

    rng = np.random.default_rng(20260819)
    max_size = 8
    wide = DeepseekV4FlashRotatingKVCache(max_size=max_size, max_undo_update=6)
    clean = DeepseekV4FlashRotatingKVCache(max_size=max_size, max_undo_update=6)

    def rows(count):
        keys = mx.array(rng.normal(size=(1, 1, count, 4)).astype(np.float32))
        values = mx.zeros((1, 1, count, 0), dtype=mx.float32)
        return keys, values

    prefix_k, prefix_v = rows(13)
    for idx in range(prefix_k.shape[2]):
        for cache in (wide, clean):
            cache.update_and_fetch(
                prefix_k[..., idx : idx + 1, :],
                prefix_v[..., idx : idx + 1, :],
            )
    assert wide.offset > max_size

    verify_k, verify_v = rows(6)
    wide.update_and_fetch(verify_k, verify_v)
    # Keep the confirmed anchor plus two accepted drafts; reject three drafts.
    assert wide.trim(3) == 3
    for idx in range(3):
        clean.update_and_fetch(
            verify_k[..., idx : idx + 1, :],
            verify_v[..., idx : idx + 1, :],
        )

    mx.eval(wide.keys, clean.keys)
    assert wide.meta_state == clean.meta_state
    assert bool(mx.array_equal(wide.keys, clean.keys))


def test_make_cache_uses_verify_rollback_cache_for_every_local_ring():
    model = DeepseekV4FlashVQModel(_drafter_args(), with_mtp=False)
    local = [entry[0] if isinstance(entry, CacheList) else entry for entry in model.make_cache()]
    assert len(local) == model.args.num_hidden_layers
    assert all(isinstance(entry, DeepseekV4FlashRotatingKVCache) for entry in local)
    assert {entry.max_undo_update for entry in local} == {
        model.args.dspark_block_size + 1
    }


def test_greedy_speculative_runtime_matches_ar_and_counts_real_verify_calls(tmp_path):
    args = _drafter_args(dspark_block_size=3)
    model = DeepseekV4FlashVQModel(args)
    artifact_dir = tmp_path / "vq"
    _write_tiny_vq_artifacts(artifact_dir, args, range(args.num_hidden_layers))
    _write_tiny_mtp_vq_artifacts(
        artifact_dir, args, range(args.num_mtp_blocks)
    )
    bind_deepseek_v4_flash_vq_experts(model, artifact_dir, strict=True)
    bind_deepseek_v4_flash_mtp_vq_experts(model, artifact_dir, strict=True)
    _randomize(model)

    prompt = _prompt(args, length=20, seed=819)
    baseline = generate_dsv4_autoregressive(model, prompt, max_new_tokens=9)
    speculative = generate_dsv4_speculative(model, prompt, max_new_tokens=9)

    assert speculative.token_ids == baseline.token_ids
    assert speculative.stats.emitted_tokens == 9
    assert speculative.stats.drafted_tokens > 0
    assert speculative.stats.drafted_tokens == (
        speculative.stats.accepted_tokens + speculative.stats.rejected_tokens
    )
    assert speculative.stats.verify_passes > 0
    assert speculative.stats.verify_kernel_calls > 0
    assert speculative.stats.acceptance_rate == pytest.approx(
        speculative.stats.accepted_tokens / speculative.stats.drafted_tokens
    )
    assert speculative.stats.accepted_per_verify_pass == pytest.approx(
        speculative.stats.accepted_tokens / speculative.stats.verify_passes
    )
    assert speculative.stats.emitted_per_verify_pass > 0
