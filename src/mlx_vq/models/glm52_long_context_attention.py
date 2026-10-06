"""Memory-bounded replacement for GLM-5.2's DSA/IndexShare attention forward.

The upstream mlx-lm DeepSeek-V3.2 attention (which ``Glm52IndexShareAttention``
subclasses) computes two full ``O(L^2)`` dense tensors regardless of how
sparse the eventual top-k selection ends up being:

1. The ``Indexer``'s own relevance-ranking scores — needed to *choose* the
   top-k positions, and inherently ``O(L^2)`` (ranking requires looking at
   every candidate; this cannot be made sub-quadratic without a different
   indexing scheme, and this module does not try).
2. The main attention's RoPE positional bias (``pe_scores``), computed
   densely against the *full* key sequence and only masked down to the
   selected top-k *after* the matmul — so the mask never reduces the cost of
   producing it.

For a ~64K-token session this materializes hundreds of GB and aborts
(measured: 274 GB for a single 65,487-token session's Indexer scores alone,
against Metal's ~87 GB buffer ceiling; the subsequent dense ``pe_scores``
would be even larger, using 64 heads vs the indexer's 32).

This module processes queries in bounded-size chunks. For each chunk it:

- computes the Indexer's ranking scores only against this chunk's queries
  (bounded: ``chunk_size * L_kv``, not ``L^2``) to get that chunk's top-k
  key indices;
- **gathers** the key/value vectors (and causal-validity mask) at just those
  top-k positions per query, via ``mx.take_along_axis`` — verified
  output-size-bounded (``chunk_size * top_k * dim``), not the naive
  ``chunk_size * L_kv * dim`` a broadcast-then-gather might suggest;
- computes ``pe_scores`` and calls SDPA only over the gathered top-k set, so
  the expensive final attention step is ``O(chunk_size * top_k)``, not
  ``O(chunk_size * L_kv)``.

Only supports the single-shot prefill path (``cache=None``), matching the
teacher-gen producer's usage — no incremental-decode KV-cache handling.
Numerically equivalent to the dense reference up to floating-point
reduction-order effects (see
``tests/test_glm52_long_context_attention.py``).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def chunked_indexer_topk(
    indexer,
    x: mx.array,
    qr: mx.array,
    mask: Optional[mx.array],
    *,
    query_chunk_size: int,
) -> Optional[mx.array]:
    """Query-chunked replacement for ``Indexer.__call__`` (prefill only, no cache).

    Returns the same ``(B, 1, L, index_topk)`` int32 index tensor the dense
    reference would produce (mod argpartition tie-break order among exactly
    equal scores, which does not affect which *set* of top-k is chosen for
    the non-degenerate float scores produced by a real model).
    """
    b, s, _ = x.shape
    # Linear-cost prefix: per-token projections, safe to compute for the full
    # sequence (no O(L^2) term appears until the score matmul below).
    q_full = indexer.wq_b(qr)
    q_full = q_full.reshape(b, s, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
    q_full = indexer.rope(q_full, offset=0)
    k_full = indexer.wk(x)
    k_full = indexer.k_norm(k_full)
    k_full = mx.reshape(k_full, (b, 1, s, indexer.head_dim))
    k_full = indexer.rope(k_full, offset=0)
    if k_full.shape[2] <= indexer.index_topk:
        return None
    weights_full = indexer.weights_proj(x) * (indexer.n_heads**-0.5 * indexer.softmax_scale)

    chunks = []
    for start, end in _chunk_ranges(s, query_chunk_size):
        q_chunk = q_full[:, :, start:end, :]
        w_chunk = weights_full[:, start:end, :].swapaxes(-1, -2)[..., None]
        scores = q_chunk @ k_full.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        scores = scores * w_chunk
        scores = scores.sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask[:, :, start:end, :], scores, -float("inf"))
        topk_chunk = mx.argpartition(scores, kth=-indexer.index_topk, axis=-1)[
            ..., -indexer.index_topk :
        ]
        mx.eval(topk_chunk)
        mx.clear_cache()
        chunks.append(topk_chunk)
    return mx.concatenate(chunks, axis=2)


def _gather_along_keys(source: mx.array, indices: mx.array) -> mx.array:
    """Gather ``source[..., idx, :]`` per-query, broadcasting to match indices.

    ``source``: (B, 1, L_kv, D). ``indices``: (B, 1, chunk, K). Returns
    (B, 1, chunk, K, D). Verified output-size-bounded, not
    ``chunk * L_kv``-bounded, in ``glm52_long_context_attention`` bring-up.
    """
    b, _, l_kv, d = source.shape
    chunk = indices.shape[2]
    src_b = mx.broadcast_to(mx.expand_dims(source, 2), (b, 1, chunk, l_kv, d))
    idx_b = mx.broadcast_to(mx.expand_dims(indices, -1), indices.shape + (d,))
    return mx.take_along_axis(src_b, idx_b, axis=3)


def chunked_index_share_attention(
    attn,
    x: mx.array,
    mask: Optional[mx.array],
    prev_topk_indices: Optional[mx.array],
    *,
    query_chunk_size: int,
) -> tuple[mx.array, Optional[mx.array]]:
    """Query-chunked, gather-based replacement for ``Glm52IndexShareAttention.__call__``.

    Prefill-only (no KV cache). Produces output numerically equivalent to
    the dense reference for the same inputs.
    """
    B, L, _ = x.shape

    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    q = attn.q_b_proj(qr)
    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)
    compressed_kv = attn.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe_full = mx.split(compressed_kv, [attn.kv_lora_rank], axis=-1)
    k_pe_full = k_pe_full.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent_full = attn.kv_a_layernorm(compressed_kv)

    q_pe = attn.rope(q_pe, offset=0)
    k_pe_full = attn.rope(k_pe_full, offset=0)
    kv_latent_full = mx.expand_dims(kv_latent_full, axis=1)

    if attn.indexer is not None:
        topk_indices = chunked_indexer_topk(
            attn.indexer, x, qr, mask, query_chunk_size=query_chunk_size
        )
    else:
        topk_indices = prev_topk_indices

    if topk_indices is None:
        raise ValueError(
            "chunked_index_share_attention requires a non-None topk_indices "
            "(sequence must exceed index_topk to need chunking at all)"
        )

    outputs = []
    for start, end in _chunk_ranges(L, query_chunk_size):
        idx_chunk = topk_indices[:, :, start:end, :]
        # _gather_along_keys already returns (B, 1, chunk, K, D) -- the "1" is
        # the head-broadcast placeholder (shared key across heads, as
        # upstream), broadcasting naturally against the (B, heads, chunk, ...)
        # query tensors below. No further axis manipulation needed.
        k_pe_g = _gather_along_keys(k_pe_full, idx_chunk)  # (B, 1, chunk, K, rope_dim)
        kv_latent_g = _gather_along_keys(kv_latent_full, idx_chunk)  # (B, 1, chunk, K, kv_lora_rank)

        mask_slice = (
            mask[:, :, start:end, :]
            if mask is not None
            else mx.ones((B, 1, end - start, k_pe_full.shape[2]), dtype=mx.bool_)
        )
        valid = mx.take_along_axis(mask_slice, idx_chunk, axis=-1)  # (B, 1, chunk, K)

        q_pe_chunk = q_pe[:, :, start:end, :]  # (B, heads, chunk, rope_dim)
        # pe_scores: (B, heads, chunk, K) via a genuine batched matmul (one
        # (1, rope_dim) @ (rope_dim, K) matrix product per query, batched over
        # B/heads/chunk) -- not elementwise-multiply-then-sum, so floating
        # point reduction order matches the reference's own matmul-based path.
        k_pe_gT = k_pe_g.swapaxes(-1, -2)  # (B, 1, chunk, rope_dim, K)
        pe_scores = (q_pe_chunk[..., None, :] * attn.scale) @ k_pe_gT
        pe_scores = pe_scores[..., 0, :]  # (B, heads, chunk, K)
        pe_scores = mx.where(
            valid, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
        )

        q_nope_chunk = q_nope[:, :, start:end, :]
        # kv_latent_g is (B,1,chunk,K,kv_lora_rank); embed_q/unembed_out expect
        # (B,1,N,kv_lora_rank)-style input per query -- fold chunk*K into one
        # axis, apply per-head projection, then unfold.
        chunk_n = end - start
        K = idx_chunk.shape[-1]
        kv_flat = kv_latent_g.reshape(B, 1, chunk_n * K, -1)
        k_flat = attn.embed_q(kv_flat, transpose=False)  # (B, heads, chunk*K, qk_nope_head_dim)
        v_flat = attn.unembed_out(kv_flat)  # (B, heads, chunk*K, v_head_dim)
        heads = k_flat.shape[1]
        k_g = k_flat.reshape(B, heads, chunk_n, K, -1)
        v_g = v_flat.reshape(B, heads, chunk_n, K, -1)

        # Per-query gathered attention: scores over just K keys via matmul,
        # softmax, weighted sum -- mirrors scaled_dot_product_attention's own
        # computation, just batched over an extra per-query "chunk" axis.
        content_scores = (q_nope_chunk[..., None, :] * attn.scale) @ k_g.swapaxes(-1, -2)
        content_scores = content_scores[..., 0, :]  # (B, heads, chunk, K)
        combined = content_scores + pe_scores
        combined = combined - mx.max(combined, axis=-1, keepdims=True)
        weights = mx.softmax(combined, axis=-1, precise=True)
        chunk_out = weights[..., None, :] @ v_g  # (B, heads, chunk, 1, v_head_dim)
        chunk_out = chunk_out[..., 0, :]  # (B, heads, chunk, v_head_dim)
        mx.eval(chunk_out)
        mx.clear_cache()
        outputs.append(chunk_out)

    output = mx.concatenate(outputs, axis=2)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output), topk_indices


def dense_or_chunked_attention(
    attn,
    x: mx.array,
    mask: Optional[mx.array],
    prev_topk_indices: Optional[mx.array],
    *,
    query_chunk_size: int,
) -> tuple[mx.array, Optional[mx.array]]:
    """Production dispatch point: dense reference for short sequences (no
    behavior change from the already-validated path), memory-bounded chunked
    path only for sequences long enough to actually need it.

    The "needs indexing" test mirrors ``Indexer.__call__``'s own internal
    check (``k.shape[2] <= index_topk`` -> returns ``None``) exactly, so this
    dispatch never disagrees with the reference about which layers engage
    DSA. Short sequences (e.g. the 66-prompt eval pack, all far under
    ``index_topk``) take the byte-identical original code path.
    """
    seq_len = x.shape[1]
    if attn.indexer is not None:
        needs_indexing = seq_len > attn.indexer.index_topk
    else:
        needs_indexing = prev_topk_indices is not None

    if not needs_indexing:
        return attn(x, mask, None, prev_topk_indices)
    return chunked_index_share_attention(
        attn, x, mask, prev_topk_indices, query_chunk_size=query_chunk_size
    )
