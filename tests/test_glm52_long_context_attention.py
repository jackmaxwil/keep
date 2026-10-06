"""Correctness tests for the query-chunked, gather-based DSA attention.

Verifies ``chunked_index_share_attention`` (memory-bounded) produces output
numerically equivalent to the dense reference (``Glm52IndexShareAttention.__call__``)
for sequences small enough that both can run, across several chunk sizes
including non-divisors of the sequence length (boundary handling) and a
degenerate single-chunk case.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.base import create_causal_mask

from mlx_vq.models.glm52_long_context_attention import (
    chunked_index_share_attention,
    dense_or_chunked_attention,
)
from mlx_vq.models.glm52_vq_adapter import GLM52VQModel, GLM52VQModelArgs


def _tiny_args(*, index_topk: int) -> GLM52VQModelArgs:
    return GLM52VQModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=3,
        index_topk=index_topk,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=5,
        num_key_value_heads=5,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=6,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        v_head_dim=4,
        qk_nope_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=["full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )


def _random_input(seq_len: int, hidden_size: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.3, size=(1, seq_len, hidden_size)).astype(np.float32)
    return mx.array(data)


class TestChunkedIndexShareAttentionParity:
    @pytest.mark.parametrize("query_chunk_size", [3, 5, 8, 32])
    def test_matches_dense_reference_output(self, query_chunk_size: int) -> None:
        seq_len = 24
        index_topk = 4
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer = model.layers[0].self_attn

        x = _random_input(seq_len, args.hidden_size, seed=7)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        dense_output, dense_topk = full_layer(x, mask, None, None)
        mx.eval(dense_output, dense_topk)

        chunked_output, chunked_topk = chunked_index_share_attention(
            full_layer, x, mask, None, query_chunk_size=query_chunk_size
        )
        mx.eval(chunked_output, chunked_topk)

        assert dense_topk is not None
        assert dense_topk.shape == chunked_topk.shape

        # Chunk size does not affect the noise floor (verified: identical max
        # abs diff for chunk_size=3 vs a single degenerate full-length chunk),
        # confirming this is inherent floating-point non-associativity between
        # gathered small-K matmuls and dense-then-masked matmuls -- not a
        # chunking-boundary bug. atol reflects the measured noise floor
        # (~6e-4 worst case across seeds/lengths) with headroom.
        np_dense = np.asarray(dense_output)
        np_chunked = np.asarray(chunked_output)
        np.testing.assert_allclose(np_chunked, np_dense, atol=1.5e-3, rtol=5e-2)

    def test_selected_topk_sets_match_per_query(self) -> None:
        # Exact index order can differ under argpartition tie-breaks; verify
        # the *set* of selected key positions per query is identical instead
        # of requiring array-level equality.
        seq_len = 20
        index_topk = 5
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer = model.layers[0].self_attn

        x = _random_input(seq_len, args.hidden_size, seed=11)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        _, dense_topk = full_layer(x, mask, None, None)
        _, chunked_topk = chunked_index_share_attention(
            full_layer, x, mask, None, query_chunk_size=6
        )
        mx.eval(dense_topk, chunked_topk)

        dense_np = np.asarray(dense_topk)[0, 0]  # (L, topk)
        chunked_np = np.asarray(chunked_topk)[0, 0]
        assert dense_np.shape == chunked_np.shape
        for row in range(dense_np.shape[0]):
            assert set(dense_np[row].tolist()) == set(chunked_np[row].tolist()), (
                f"row {row}: dense={sorted(dense_np[row].tolist())} "
                f"chunked={sorted(chunked_np[row].tolist())}"
            )

    def test_raises_when_sequence_too_short_to_need_indexing(self) -> None:
        # k.shape[2] <= index_topk -> indexer returns None; chunked path
        # currently requires long-context indexing to be active (the dense
        # reference is fine and fast at this size, so callers should just use
        # the dense path directly when it fits).
        seq_len = 3
        index_topk = 8
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer = model.layers[0].self_attn
        x = _random_input(seq_len, args.hidden_size, seed=3)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        with pytest.raises(ValueError, match="topk_indices"):
            chunked_index_share_attention(full_layer, x, mask, None, query_chunk_size=2)

    def test_shared_layer_reuses_prev_topk_indices(self) -> None:
        # A "shared" layer (indexer=None) must reuse prev_topk_indices from a
        # prior full layer rather than compute its own.
        seq_len = 16
        index_topk = 4
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer, shared_layer = model.layers[0].self_attn, model.layers[1].self_attn

        x = _random_input(seq_len, args.hidden_size, seed=5)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        dense_full_out, dense_topk = full_layer(x, mask, None, None)
        dense_shared_out, dense_shared_topk = shared_layer(dense_full_out, mask, None, dense_topk)
        mx.eval(dense_shared_out, dense_shared_topk)

        chunked_full_out, chunked_topk = chunked_index_share_attention(
            full_layer, x, mask, None, query_chunk_size=6
        )
        chunked_shared_out, chunked_shared_topk = chunked_index_share_attention(
            shared_layer, chunked_full_out, mask, chunked_topk, query_chunk_size=6
        )
        mx.eval(chunked_shared_out, chunked_shared_topk)

        # Two chained layers: noise from layer 0's chunked output propagates
        # into layer 1's input, so the floor is looser than the single-layer
        # case above.
        np.testing.assert_allclose(
            np.asarray(chunked_shared_out), np.asarray(dense_shared_out), atol=3e-3, rtol=5e-2
        )


class TestDenseOrChunkedAttentionDispatch:
    def test_short_sequence_full_layer_uses_dense_path_exactly(self) -> None:
        # seq_len <= index_topk -> indexer returns None -> must dispatch to
        # the byte-identical dense call, not chunked (which would itself
        # raise since topk_indices would be None).
        index_topk = 8
        seq_len = 6  # < index_topk
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer = model.layers[0].self_attn
        x = _random_input(seq_len, args.hidden_size, seed=1)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        dense_output, dense_topk = full_layer(x, mask, None, None)
        mx.eval(dense_output, dense_topk)

        dispatched_output, dispatched_topk = dense_or_chunked_attention(
            full_layer, x, mask, None, query_chunk_size=4
        )
        mx.eval(dispatched_output, dispatched_topk)

        assert dispatched_topk is None
        np.testing.assert_array_equal(np.asarray(dispatched_output), np.asarray(dense_output))

    def test_boundary_seq_len_equals_index_topk_uses_dense_path(self) -> None:
        # seq_len == index_topk: the reference's own check is
        # `k.shape[2] <= index_topk` (returns None) -- must match that exact
        # boundary, not an off-by-one.
        index_topk = 8
        seq_len = index_topk
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer = model.layers[0].self_attn
        x = _random_input(seq_len, args.hidden_size, seed=2)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        dense_output, dense_topk = full_layer(x, mask, None, None)
        mx.eval(dense_output, dense_topk)
        dispatched_output, dispatched_topk = dense_or_chunked_attention(
            full_layer, x, mask, None, query_chunk_size=4
        )
        mx.eval(dispatched_output, dispatched_topk)

        assert dense_topk is None  # confirms this IS the boundary case
        assert dispatched_topk is None
        np.testing.assert_array_equal(np.asarray(dispatched_output), np.asarray(dense_output))

    def test_long_sequence_full_layer_uses_chunked_path(self) -> None:
        index_topk = 4
        seq_len = 24  # > index_topk
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer = model.layers[0].self_attn
        x = _random_input(seq_len, args.hidden_size, seed=3)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        direct_output, direct_topk = chunked_index_share_attention(
            full_layer, x, mask, None, query_chunk_size=6
        )
        mx.eval(direct_output, direct_topk)

        dispatched_output, dispatched_topk = dense_or_chunked_attention(
            full_layer, x, mask, None, query_chunk_size=6
        )
        mx.eval(dispatched_output, dispatched_topk)

        assert dispatched_topk is not None
        np.testing.assert_array_equal(np.asarray(dispatched_topk), np.asarray(direct_topk))
        np.testing.assert_array_equal(np.asarray(dispatched_output), np.asarray(direct_output))

    def test_shared_layer_short_sequence_falls_back_to_dense_without_prior_topk(self) -> None:
        index_topk = 8
        seq_len = 5  # short enough that no indexing is needed at all
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        shared_layer = model.layers[1].self_attn
        x = _random_input(seq_len, args.hidden_size, seed=4)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        dense_output, dense_topk = shared_layer(x, mask, None, None)
        mx.eval(dense_output, dense_topk)
        dispatched_output, dispatched_topk = dense_or_chunked_attention(
            shared_layer, x, mask, None, query_chunk_size=4
        )
        mx.eval(dispatched_output, dispatched_topk)

        assert dispatched_topk is None
        np.testing.assert_array_equal(np.asarray(dispatched_output), np.asarray(dense_output))

    def test_shared_layer_long_sequence_with_prior_topk_uses_chunked_path(self) -> None:
        index_topk = 4
        seq_len = 20
        args = _tiny_args(index_topk=index_topk)
        model = GLM52VQModel(args)
        full_layer, shared_layer = model.layers[0].self_attn, model.layers[1].self_attn
        x = _random_input(seq_len, args.hidden_size, seed=5)
        mask = create_causal_mask(seq_len, right_padding=mx.array([0]))

        full_out, prior_topk = dense_or_chunked_attention(
            full_layer, x, mask, None, query_chunk_size=6
        )
        mx.eval(full_out, prior_topk)
        assert prior_topk is not None  # sanity: this run actually engages indexing

        direct_output, direct_topk = chunked_index_share_attention(
            shared_layer, full_out, mask, prior_topk, query_chunk_size=6
        )
        mx.eval(direct_output, direct_topk)

        dispatched_output, dispatched_topk = dense_or_chunked_attention(
            shared_layer, full_out, mask, prior_topk, query_chunk_size=6
        )
        mx.eval(dispatched_output, dispatched_topk)

        np.testing.assert_array_equal(np.asarray(dispatched_topk), np.asarray(direct_topk))
        np.testing.assert_array_equal(np.asarray(dispatched_output), np.asarray(direct_output))
