"""KEEP VQ adapter for ``deepseek-ai/DeepSeek-V4-Flash-0731``.

Provenance
==========

``mlx-lm`` 0.31.3 (the version pinned in ``uv.lock``) ships no
``mlx_lm.models.deepseek_v4`` module -- ``deepseek``, ``deepseek_v2``,
``deepseek_v3`` and ``deepseek_v32`` only -- so there is nothing upstream to
subclass. The model structure below is therefore **vendored** from

    ``.repos/omlx/omlx/patches/deepseek_v4/deepseek_v4_model.py``
    ``.repos/omlx/omlx/patches/deepseek_v4/hyper_connection.py``
    ``.repos/omlx/omlx/patches/deepseek_v4/cache_extras.py``

which are 1:1 copies of mlx-lm PR 1192 (``Blaizzy/mlx-lm`` branch
``pc/add-deepseekv4flash-model``, HEAD
``5c10538136b9038b9626c134612b08afc18d697a``, 2026-05-01), Apache 2.0,
"Copyright (c) 2026 Apple Inc.".

What was kept
-------------
DSpark attention (``LocalAttention`` / ``CompressedAttention`` /
``SparseCompressedAttention`` + ``Compressor`` + ``Indexer``), hyper-connections
(``HyperConnection`` / ``HyperHead`` / :func:`hc_expand`), :class:`LimitedSwiGLU`,
the hash/score ``MoEGate``, the YaRN ``DeepseekV4RoPE``, ``PoolingCache``, and
the full MTP (DSpark drafter) structure at ``mtp.{0,1,2}``.

What was stripped
-----------------
Everything server- or accelerator-specific that the KEEP runtime does not own:
the ``decode_consistency`` verify-arming globals and ``exact_*`` / ``verify_qmv``
paths, the ``wsdpa`` and ``glm_moe_dsa`` native-kernel dispatches (the vendored
MLX fallbacks are kept as the only path), continuous-batching
``BatchPoolingCache`` and the omlx cache handlers, ``PipelineMixin`` /
``mx.distributed`` sharding, and ``make_quantization_config`` (KEEP's routed
experts are VQ, not mxfp4). ``PoolingCache`` keeps only what forward needs: its
MTP-rollback undo log and prompt-cache trimming hooks are gone.

.. warning::

   **The ``self.dspark`` exact-attention decode branch must come back with the
   Wave-5 MTP verify runtime.** Upstream's three attention classes each take an
   ``exact_attention(q, [kv], scale, sinks)`` path whenever
   ``self.dspark and B == 1 and L == 1`` -- gated on ``_is_dspark_model(config)``
   (true for this checkpoint: ``dspark_block_size=5`` and non-empty
   ``dspark_target_layer_ids``), **not** on the verify-armed flag. Only the
   verify-armed ``exact_*`` machinery was legitimately strippable here. Decode
   on this build therefore takes ``scaled_dot_product_attention`` where upstream
   takes a different reduction order, so single-token decode is not
   bit-identical to upstream even before any KEEP quantization. That is
   tolerable for increment 1 (no decode is executed) and *not* tolerable for
   speculative verify, where the drafter and the target must agree exactly.

Divergences from the vendored source
------------------------------------
1. ``DeepseekV4MoE.switch_mlp`` is not a dense ``SwitchGLU``. Routed experts are
   the tensors KEEP compresses, so :class:`DeepseekV4FlashVQMoE` starts with
   ``switch_mlp = None`` and takes a
   :class:`~ramp.models.glm4_moe_adapter.QuantizedVQSwitchGLU` from
   :func:`bind_deepseek_v4_flash_vq_experts`. This is the same shape as the
   GLM-5.2 adapter and is what makes "zero dense routed parameters" true by
   construction rather than by audit.
2. ``Model.sanitize`` dropped every ``mtp.`` tensor. Here the drafter is loaded
   *intentionally* into :attr:`DeepseekV4FlashVQModel.mtp_drafter` (campaign
   pillars 3 and 4 need it), and the bind report counts drafter tensors
   separately from backbone tensors.
3. ``MoEGate.tid2eid`` keeps the checkpoint's native ``int64`` instead of being
   cast to ``int32``. It is a token->expert routing table, never a numeric
   weight, so it is never quantized and never re-typed (T6 review note).

Weight naming
-------------
The released checkpoint is unprefixed and uses DeepSeek's own names; this module
tree mirrors mlx-lm's (``model.`` backbone + ``lm_head``). The mapping applied by
:func:`bind_deepseek_v4_flash_non_vq_weights` is the vendored ``sanitize``
mapping, made explicit so it can run shard-by-shard off the safetensors index:

======================================  =====================================
checkpoint                              module
======================================  =====================================
``embed.weight``                        ``model.embed_tokens.weight``
``norm.weight``                         ``model.norm.weight``
``head.weight``                         ``lm_head.weight``
``hc_head_{fn,base,scale}``             ``model.hc_head.{fn,base,scale}``
``layers.N.*``                          ``model.layers.N.*``
``layers.N.hc_attn_X``                  ``model.layers.N.attn_hc.X``
``layers.N.hc_ffn_X``                   ``model.layers.N.ffn_hc.X``
``layers.N.ffn.gate.bias``              ``...ffn.gate.e_score_correction_bias``
``layers.N.attn.wo_a.weight``           reshaped ``(o_groups, o_lora_rank, -1)``
``layers.N.ffn.shared_experts.w1``      ``...shared_experts.gate_proj``
``layers.N.ffn.shared_experts.w2``      ``...shared_experts.down_proj``
``layers.N.ffn.shared_experts.w3``      ``...shared_experts.up_proj``
``layers.N.ffn.experts.E.w{1,2,3}``     routed -- VQ, never bound here
``mtp.K.*``                             ``mtp_drafter.blocks.K.*`` (same rules)
======================================  =====================================

``w1``=gate, ``w2``=down, ``w3``=up. This mapping is **verified only for the
shared experts**, whose renamed tensors are bound and shape-checked here against
the module tree. For the *routed* experts it is the intended contract, not yet a
verified one: no VQ materializer for this family exists (``converter:
deepseek_v4_vq_groups`` in the profile is a registered string with no
implementation behind it), so nothing has yet produced a routed artifact whose
projection naming could be checked. Wave 5 must confirm it against real fitted
artifacts before any quality number is trusted.

gate and up are separate tensors in the release, which is why
``models/deepseek-v4-flash-0731.yaml`` sets ``fused_gate_up: false``.

Source storage formats (measured at revision ``7872f01b``):

* residents (attention, shared experts, MTP ``main_proj``): ``F8_E4M3`` codes
  with a ``F8_E8M0`` 128x128 block scale -> :func:`keep.convert.fp8_block.dequantize_fp8_block`
* norms, gate weights, embeddings, ``markov_head``, ``confidence_head``: ``BF16``
* ``attn_sink``, hyper-connection scalars, ``compressor.ape``: ``F32``
* ``ffn.gate.tid2eid``: ``I64``
* routed experts: FP4 two-per-byte in ``I8`` with ``F8_E8M0`` group-32 scales ->
  :func:`keep.convert.fp4_expert.dequantize_fp4_expert` (VQ source, or the MTP
  dense path in :func:`bind_deepseek_v4_flash_mtp_dense_experts`)
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import CacheList, RotatingKVCache, _BaseCache
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

from keep.convert.fp4_expert import dequantize_fp4_expert
from keep.convert.fp8_block import dequantize_fp8_block
from keep.io.load import load_quantized_vq_switch_linear
from keep.io.source_safetensors import read_safetensors_file_header
from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU

__all__ = [
    "DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS",
    "CompressedAttention",
    "DSparkAttention",
    "DSparkContextCache",
    "DeepseekV4FlashBlock",
    "DeepseekV4FlashMTPBlock",
    "DeepseekV4FlashMTPDrafter",
    "DeepseekV4FlashNonVQBindReport",
    "DeepseekV4FlashRotatingKVCache",
    "DeepseekV4FlashVQModel",
    "DeepseekV4FlashVQModelArgs",
    "DeepseekV4FlashVQMoE",
    "LimitedSwiGLU",
    "LocalAttention",
    "PoolingCache",
    "SparseCompressedAttention",
    "bind_deepseek_v4_flash_mtp_dense_experts",
    "bind_deepseek_v4_flash_mtp_vq_experts",
    "bind_deepseek_v4_flash_non_vq_weights",
    "bind_deepseek_v4_flash_vq_experts",
    "deepseek_v4_flash_args_from_config",
    "deepseek_v4_flash_pooling_undo_window",
    "deepseek_v4_flash_checkpoint_tensor_names",
    "dense_deepseek_v4_flash_routed_parameter_names",
    "has_unbound_deepseek_v4_flash_mtp_experts",
    "has_unbound_deepseek_v4_flash_vq_experts",
    "load_deepseek_v4_flash_vq_switch_glu",
    "v4_attention_factory",
]


# ---------------------------------------------------------------------------
# Teacher window
# ---------------------------------------------------------------------------

#: Context window every DeepSeek-V4-Flash teacher pass runs at, in tokens.
#:
#: **Decided, not derived** -- 81,920 = 640 x 128, i.e. a whole number of
#: ratio-128 pooling windows (and therefore also of the ratio-4 windows and the
#: 128-token sliding window), which keeps ``PoolingCache`` from carrying a
#: partial-window remainder across a session boundary.
#:
#: It has to cover the longest session in the corpus or that session is silently
#: truncated and its supervised tail never gets a teacher logit.
#: ``recipes/dsv4_teich_split_manifest_v1_20260811.json`` measures the V4
#: re-tokenisation at ``v4_session_tokens_max = 77,075`` (median 48,166; 31 of
#: 257 sessions exceed 65,536, which is why 65,536 was not enough). The margin is
#: 4,845 tokens, ~6%.
#:
#: The released config's ``max_position_embeddings`` is 1,048,576 and the YaRN
#: ``original_max_position_embeddings`` is 65,536, so this window sits inside the
#: scaled regime the checkpoint was trained for and well inside the positional
#: range the RoPE supports.
DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS = 81_920


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4FlashVQModelArgs(BaseModelArgs):
    """Vendored PR-1192 ``ModelArgs``; validated against the real config.json.

    ``compress_ratios`` truncation is load-bearing: the released config ships 46
    entries for 43 hidden layers because the trailing three describe the
    ``mtp.{0,1,2}`` drafter blocks (all ``0``, i.e. uncompressed local
    attention). ``__post_init__`` truncates to ``num_hidden_layers``, which is
    exactly the upstream behaviour -- do not "fix" it.
    """

    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: list[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    # DeepSeek-V4-Flash-0731 embeds a DSpark drafter in ``mtp.0..N``. The legacy
    # ``num_nextn_predict_layers`` field is still set for compatibility, so
    # these fields are the architecture discriminator.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: list[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    n_mtp_layers: int = 0
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"

    def __post_init__(self) -> None:
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        # The drafter ratios live past ``num_hidden_layers``; keep them so the
        # MTP blocks can be built from the same list, then truncate the
        # backbone view exactly as upstream does.
        self.mtp_compress_ratios = list(self.compress_ratios[self.num_hidden_layers :])
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")
        if self.num_hash_layers > self.num_hidden_layers:
            raise ValueError(
                "`num_hash_layers` cannot exceed `num_hidden_layers`: "
                f"{self.num_hash_layers} > {self.num_hidden_layers}"
            )

    @property
    def num_mtp_blocks(self) -> int:
        """Drafter depth, measured from the checkpoint rather than assumed.

        ``num_nextn_predict_layers`` is 1 on this release but the checkpoint
        carries three full MoE blocks at ``mtp.{0,1,2}``; the honest signal is
        ``dspark_target_layer_ids`` (``[40, 41, 42]``).
        """

        if self.dspark_target_layer_ids:
            return len(self.dspark_target_layer_ids)
        return max(int(self.n_mtp_layers), 0)

    @property
    def main_proj_in_features(self) -> int:
        """Width of the drafter's stage-0 fusion projection.

        ``mtp.0.main_proj`` fuses the backbone hidden states captured at
        ``dspark_target_layer_ids``, so its input is
        ``hidden_size * len(dspark_target_layer_ids)`` -- reference
        ``deepseek_v4_dspark.py:319``. That happens to be ``3 * hidden_size`` on
        this release only because three layers are targeted; a config targeting
        two layers must get a narrower projection, not the same constant.
        """

        return self.hidden_size * max(len(self.dspark_target_layer_ids), 1)


def deepseek_v4_flash_args_from_config(
    config: Mapping[str, Any],
) -> DeepseekV4FlashVQModelArgs:
    """Build model args from a raw ``config.json`` mapping.

    Unknown keys are ignored (the released config carries 52 keys, several of
    which -- ``architectures``, ``quantization_config``, ``expert_dtype``,
    tokenizer ids -- are loader metadata rather than model structure).
    """

    field_names = {f.name for f in fields(DeepseekV4FlashVQModelArgs)}
    kwargs = {name: config[name] for name in field_names if name in config}
    return DeepseekV4FlashVQModelArgs(**kwargs)


# ---------------------------------------------------------------------------
# Pooling cache (vendored, trimmed to what forward needs)
# ---------------------------------------------------------------------------


#: Longest update a bare :class:`PoolingCache` stashes an undo record for.
#:
#: 1 -- decode only. A cache built without a config cannot know how wide a
#: verify window will be, and the honest default is to promise only what a
#: single decode step needs. :func:`deepseek_v4_flash_pooling_undo_window`
#: derives the real bound from the config, and :meth:`
#: DeepseekV4FlashVQModel.make_cache` passes it. Prompt chunks (1,024 tokens
#: here) must never be stashed: pinning a chunk's projections for the whole
#: prefill to serve a rollback nothing asks for is pure cost.
#:
#: The vendored value was a bare ``8`` (``cache_extras.py:122``). 8 happens to
#: exceed ``dspark_block_size + 1 == 6`` on this release, which is the kind of
#: coincidence that stops being true the day a checkpoint ships a wider block.
_POOLING_UNDO_DEFAULT_MAX_UPDATE = 1


def deepseek_v4_flash_pooling_undo_window(
    config: DeepseekV4FlashVQModelArgs,
) -> int:
    """Longest single cache update a DSpark verify cycle can produce.

    A verify step forwards the whole proposed block plus the anchor through the
    target model in one call, so the widest update is
    ``dspark_block_size + 1``. A non-DSpark config (``dspark_block_size == 0``)
    gets 1: decode only.
    """

    return max(int(getattr(config, "dspark_block_size", 0) or 0), 0) + 1


class PoolingCache(_BaseCache):
    """Pooled (compressed) KV tokens plus a partial-window remainder buffer.

    Vendored from PR 1192 ``mlx_lm/models/cache.py``. The prompt-cache
    ``state``/``meta_state`` hooks stay stripped (KEEP does not reuse V4 prompt
    caches), but the **MTP-rollback undo log is implemented**: a rejected DSpark
    draft has to put this cache back exactly where it was, and
    :meth:`is_trimmable` returning a bare ``True`` without one would silently
    corrupt the pooled timeline.

    How the rollback works. A trim of ``n`` tokens is free while those tokens
    are still sitting in the partial-window remainder buffer -- nothing was
    pooled, so dropping them is an integer subtraction. The hard case is the
    token that *completed* a window: its ``ratio`` predecessors were consumed
    out of ``buf_kv``, compressed, and appended to the pooled run, and the
    remainder buffer no longer holds them. :meth:`accumulate_windows` therefore
    stashes, before it mutates anything, the pre-update remainder rows, the
    pre-update pooled length and overlap windows, and this update's raw
    ``(kv, gate)`` inputs. :meth:`trim` replays that prefix minus ``n`` tokens.

    Two deliberate divergences from ``cache_extras.py``:

    * The stash records the pre-update pooled **length**, not the pooled array.
      ``trim`` only ever reads ``pooled_prev.shape[1]`` upstream, and
      ``self.pooled`` is a lazy view of ``_pool_buf`` that later in-place
      appends could alias.
    * There is no ``_undo_chain``. Upstream chains consecutive decode updates
      into one undo record, gated on omlx's ``cache_rollback`` arming flags,
      which were stripped with the rest of the verify machinery. Without them
      the honest contract is one update deep: ``trim`` past the last update
      returns 0 rather than guessing.
    """

    def __init__(
        self,
        ratio: int,
        *,
        max_undo_update: int = _POOLING_UNDO_DEFAULT_MAX_UPDATE,
    ):
        self.ratio = ratio
        self.max_undo_update = max(int(max_undo_update), 1)
        self.buf_kv: mx.array | None = None
        self.buf_gate: mx.array | None = None
        self.remainder = 0
        self._pool_buf: mx.array | None = None
        self._pool_len = 0
        # Overlap (ratio == 4) compressors need the previous completed window.
        self.prev_win_kv: mx.array | None = None
        self.prev_win_gate: mx.array | None = None
        # One-update undo record; see the class docstring.
        self._undo: tuple[Any, ...] | None = None

    @property
    def pooled(self) -> mx.array | None:
        if self._pool_buf is None:
            return None
        return self._pool_buf[:, : self._pool_len]

    @pooled.setter
    def pooled(self, value: mx.array | None) -> None:
        if value is None:
            self._pool_buf = None
            self._pool_len = 0
        else:
            self._pool_buf = value
            self._pool_len = value.shape[1]

    @property
    def offset(self) -> int:
        return self._pool_len

    def _grow_pool(self, needed: int) -> None:
        old = self._pool_buf
        capacity = max(needed, 2 * old.shape[1])
        new = mx.zeros((old.shape[0], capacity, old.shape[2]), dtype=old.dtype)
        new[:, : self._pool_len] = old[:, : self._pool_len]
        self._pool_buf = new

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, self.ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, self.ratio, D2), dtype=gate.dtype)

        # Stash the undo record *before* any mutation, so the buffer slices
        # below reference the pre-update array node.
        if L <= self.max_undo_update:
            self._undo = (
                self.buf_kv[:, : self.remainder] if self.remainder > 0 else None,
                self.buf_gate[:, : self.remainder] if self.remainder > 0 else None,
                self.remainder,
                self._pool_len,
                kv,
                gate,
                self.prev_win_kv,
                self.prev_win_gate,
            )
        else:
            self._undo = None

        if L > 1:  # prompt
            total = L + self.remainder
            usable = (total // self.ratio) * self.ratio
            new_remainder = total % self.ratio

            if usable > 0:
                r_kv = mx.concatenate(
                    [
                        self.buf_kv[:, : self.remainder],
                        kv[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_gate = mx.concatenate(
                    [
                        self.buf_gate[:, : self.remainder],
                        gate[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_base = offset - self.remainder
                self.remainder = 0
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            if new_remainder > 0:
                self.buf_kv[:, self.remainder : new_remainder] = kv[:, -new_remainder:]
                self.buf_gate[:, self.remainder : new_remainder] = gate[
                    :, -new_remainder:
                ]
            self.remainder = new_remainder
            return r_kv, r_gate, r_base

        # decode
        self.buf_kv[:, self.remainder : self.remainder + 1] = kv
        self.buf_gate[:, self.remainder : self.remainder + 1] = gate
        self.remainder = (self.remainder + 1) % self.ratio

        if self.remainder == 0:
            return self.buf_kv, self.buf_gate, offset - self.ratio + 1
        return (
            mx.zeros((B, 0, D1), dtype=kv.dtype),
            mx.zeros((B, 0, D2), dtype=gate.dtype),
            0,
        )

    def update_and_fetch(self, px: mx.array) -> mx.array:
        if px.shape[1] == 0:
            if self._pool_buf is None:
                return mx.zeros((px.shape[0], 0, px.shape[-1]), dtype=px.dtype)
            return self.pooled

        n = px.shape[1]
        if self._pool_buf is None:
            self._pool_buf = px
            self._pool_len = n
        else:
            if self._pool_len + n > self._pool_buf.shape[1]:
                self._grow_pool(self._pool_len + n)
            self._pool_buf[:, self._pool_len : self._pool_len + n] = px
            self._pool_len += n
        return self.pooled

    def make_mask(self, L: int = 1, offset: int = 0):
        """Query at ``offset + j`` may attend pooled token ``i`` iff
        ``i < (offset + j) // ratio``. ``None`` means "everything visible"."""

        if self.pooled is None or L == 1:
            return None
        pool_idx = mx.arange(self.pooled.shape[1])
        query_idx = mx.arange(offset + 1, offset + L + 1)
        return pool_idx < query_idx[:, None] // self.ratio

    def prev_for_prepend(self):
        """Previous completed window for the Compressor to prepend, or
        ``(None, None)`` when no overlap carry is available."""

        if self.prev_win_kv is None:
            return None, None
        return self.prev_win_kv, self.prev_win_gate

    def store_prev(self, kv: mx.array, gate: mx.array, dropped: int) -> None:
        """Roll the prev window after a compression step.

        ``dropped`` is unused for the single-sequence cache; it is kept for
        signature parity with the vendored ``BatchPoolingCache``.
        """

        self.prev_win_kv = kv[:, -1:]
        self.prev_win_gate = gate[:, -1:]

    def is_trimmable(self) -> bool:
        """Can the last token be undone exactly?

        ``True`` in two situations and no others: the pooled run is still empty
        (nothing to unpick), or the last token is recoverable -- either it never
        left the remainder buffer, or the one-update undo log still holds the
        inputs that pooled it. This is a real capability claim now, not the
        blanket ``False`` increment 1 shipped; ``trim`` below is what makes it
        true, and ``test_pooling_cache_trim_restores_exact_prior_state`` proves
        the restored state is bit-identical rather than merely plausible.
        """

        if self.pooled is None or self.remainder >= 1:
            return True
        return self._can_undo(1)

    def _can_undo(self, n: int) -> bool:
        undo = self._undo
        if undo is None:
            return False
        return undo[4].shape[1] - n >= 0

    def trim(self, n: int) -> int:
        """Drop the last ``n`` tokens; return how many were dropped.

        **Raises** when the undo record cannot cover the request, rather than
        returning 0. That is a deliberate divergence from ``cache_extras.py``
        and from mlx-lm's usual "return what you managed" convention, and the
        reason is ``CacheList.trim``: it returns only its *last* sub-cache's
        result, so a caller trimming a `CacheList(RotatingKVCache,
        PoolingCache)` by ``n`` would get ``RotatingKVCache``'s ``n`` back
        while this cache had silently dropped 0. The two caches would then
        describe different timelines with nothing to signal it -- which is the
        exact corruption ``is_trimmable() -> False`` was standing in for.

        A Wave-5 block rollback trims by ``block_size - (accepted + 1)``, so
        ``n > 1`` is the normal case there, not an exotic one. Loud beats
        silent. The ``n == 1`` consumer contract is unchanged.
        """

        n = int(n)
        if n <= 0:
            return 0
        if n <= self.remainder:
            self.remainder -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            covered = 0 if self._undo is None else int(self._undo[4].shape[1])
            raise ValueError(
                f"PoolingCache cannot trim {n}: the remainder holds "
                f"{self.remainder} and the one-update undo log covers "
                f"{covered} more (max_undo_update={self.max_undo_update}). "
                "Trimming partially would desync this cache from the local KV "
                "cache; CacheList.trim would not surface it."
            )
        buf_kv, buf_gate, _rem_prev, pooled_prev, kv, gate, prev_kv, prev_gate = self._undo
        self._undo = None

        keep = kv.shape[1] - n
        prefix_kv = kv[:, :keep]
        prefix_gate = gate[:, :keep]
        if buf_kv is not None:
            prefix_kv = mx.concatenate([buf_kv, prefix_kv], axis=1)
            prefix_gate = mx.concatenate([buf_gate, prefix_gate], axis=1)

        completed = prefix_kv.shape[1] // self.ratio
        if completed == 0:
            # Restore the pre-update logical length. The appended rows were only
            # ever written at ``[pooled_prev, ...)``, so shortening the logical
            # length discards exactly them.
            if pooled_prev == 0:
                self._pool_buf = None
                self._pool_len = 0
            else:
                self._pool_len = pooled_prev
            self.prev_win_kv = prev_kv
            self.prev_win_gate = prev_gate
        else:
            # The update already compressed these windows; keep their exact rows
            # rather than recompressing during rollback.
            self._pool_len = pooled_prev + completed
            end = completed * self.ratio
            self.prev_win_kv = prefix_kv[:, end - self.ratio : end, :][:, None]
            self.prev_win_gate = prefix_gate[:, end - self.ratio : end, :][:, None]

        used = completed * self.ratio
        remainder_kv = prefix_kv[:, used:]
        remainder_gate = prefix_gate[:, used:]
        self.remainder = remainder_kv.shape[1]
        if self.remainder:
            self.buf_kv[:, : self.remainder] = remainder_kv
            self.buf_gate[:, : self.remainder] = remainder_gate
        return n

    def size(self) -> int:
        return self._pool_len

    def empty(self) -> bool:
        return self._pool_buf is None and self.remainder == 0

    def clear_undo(self) -> None:
        """Discard a completed verify window's rollback record."""

        self._undo = None

    @property
    def nbytes(self) -> int:
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self._pool_buf is not None:
            total += self._pool_buf.nbytes
        if self.prev_win_kv is not None:
            total += self.prev_win_kv.nbytes + self.prev_win_gate.nbytes
        return total


class DeepseekV4FlashRotatingKVCache(RotatingKVCache):
    """Rotating target cache with one exact wide-verify undo record.

    Stock ``RotatingKVCache`` cannot trim after it has wrapped because the
    evicted slots are gone. A DSpark target verify writes ``anchor + drafts``
    with the functional multi-row path, so retaining the pre-write array
    references is sufficient. Rollback restores that snapshot and replays the
    accepted prefix one token at a time, preserving the physical ring layout
    ordinary autoregressive decode would have produced.
    """

    def __init__(self, max_size: int, *, max_undo_update: int):
        super().__init__(max_size=max_size)
        self.max_undo_update = max(int(max_undo_update), 2)
        self._verify_undo: tuple[Any, ...] | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        steps = int(keys.shape[2])
        if 1 < steps <= self.max_undo_update:
            # Multi-row updates rebind rather than mutate these arrays.
            self._verify_undo = (
                self.keys,
                self.values,
                self.offset,
                self._idx,
                keys,
                values,
            )
        else:
            self._verify_undo = None
        return super().update_and_fetch(keys, values)

    def can_trim(self, n: int) -> bool:
        n = int(n)
        if n <= 0:
            return True
        undo = self._verify_undo
        return bool(
            (undo is not None and n <= int(undo[4].shape[2])) or super().is_trimmable()
        )

    def is_trimmable(self) -> bool:
        return self.can_trim(1)

    def trim(self, n: int) -> int:
        n = int(n)
        if n <= 0:
            return 0
        undo = self._verify_undo
        self._verify_undo = None
        if undo is None:
            if not super().is_trimmable():
                raise ValueError(
                    f"rotated DeepSeek-V4 cache cannot trim {n} without a "
                    "wide-verify undo record"
                )
            return super().trim(n)

        keys_before, values_before, offset_before, idx_before, keys, values = undo
        keep = int(keys.shape[2]) - n
        if keep < 0:
            self._verify_undo = undo
            raise ValueError(
                f"DeepSeek-V4 verify cache cannot trim {n}; last update has "
                f"only {int(keys.shape[2])} rows"
            )
        self.keys = keys_before
        self.values = values_before
        self.offset = offset_before
        self._idx = idx_before
        for idx in range(keep):
            super().update_and_fetch(
                keys[..., idx : idx + 1, :],
                values[..., idx : idx + 1, :],
            )
        return n

    def clear_undo(self) -> None:
        self._verify_undo = None


# ---------------------------------------------------------------------------
# Hyper-connections (vendored, pure-MLX path only)
# ---------------------------------------------------------------------------


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


class HyperConnection(nn.Module):
    def __init__(self, config: DeepseekV4FlashVQModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T
        pre, post, comb = _hc_split_sinkhorn_ops(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb


@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)


def hc_expand(x, residual, post, comb):
    return _hc_expand_op(x, residual, post, comb)


class HyperHead(nn.Module):
    def __init__(self, config: DeepseekV4FlashVQModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)


# ---------------------------------------------------------------------------
# Routing / activation (vendored)
# ---------------------------------------------------------------------------


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _hash_expert_select(
    input_ids: mx.array,
    logits: mx.array,
    tid2eid: mx.array,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    inds = tid2eid[input_ids]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    """SwiGLU with DeepSeek-V4's ``swiglu_limit`` clamp (10.0 on this release)."""

    def __init__(self, limit: float, *, fp32: bool = False):
        super().__init__()
        self.limit = limit
        self.fp32 = fp32

    def __call__(self, x, gate):
        if not self.fp32:
            return _limited_swiglu(gate, x, self.limit)
        dtype = x.dtype
        return _limited_swiglu(
            gate.astype(mx.float32),
            x.astype(mx.float32),
            self.limit,
        ).astype(dtype)


class MoEGate(nn.Module):
    """Hash routing on the first ``num_hash_layers``, scored routing after.

    ``tid2eid`` is a ``[vocab_size, top_k]`` token -> expert table and stays in
    the checkpoint's native ``int64``: it is an index table, never a numeric
    weight, so it is never quantized and never re-typed.
    """

    def __init__(self, config: DeepseekV4FlashVQModelArgs, layer_idx: int, *, hash_routing: bool | None = None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = (
            layer_idx < config.num_hash_layers if hash_routing is None else hash_routing
        )
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int64)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = x @ self.weight.T
        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            return _hash_expert_select(
                input_ids,
                logits,
                self.tid2eid,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )
        return _expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.scoring_func,
        )


class DeepseekV4FlashMLP(nn.Module):
    """Dense SwiGLU MLP -- shared experts, and the MTP blocks' shared experts.

    ``fp32_swiglu`` mirrors the vendored flag (``deepseek_v4_model.py:827,843``).
    The backbone leaves it ``False``; every DSpark drafter stage sets it ``True``
    (``deepseek_v4_dspark.py:313``), so the drafter's shared-expert GLU is
    evaluated in float32 and cast back, while the backbone's runs at the
    activation dtype. It changes numbers, not shapes, which is precisely why it
    is easy to drop and expensive to have dropped.
    """

    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
        *,
        fp32_swiglu: bool = False,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit
        self.fp32_swiglu = bool(fp32_swiglu)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.fp32_swiglu:
            hidden = _limited_swiglu(
                gate.astype(mx.float32),
                up.astype(mx.float32),
                self.swiglu_limit,
            ).astype(x.dtype)
        else:
            hidden = _limited_swiglu(gate, up, self.swiglu_limit)
        return self.down_proj(hidden)


# ---------------------------------------------------------------------------
# RoPE / compressor / indexer / attention (vendored)
# ---------------------------------------------------------------------------


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max = scaling_config["original_max_position_embeddings"]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(original_max / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache: dict[tuple[int, bool], mx.array] = {}

    def _get_freqs(self, head_dim: int, inverse: bool) -> mx.array:
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(self, x: mx.array, offset: Any = 0, inverse: bool = False) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


def _apply_score_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape
    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, N - S))
    return mx.concatenate([mask, pool_mask], axis=-1)


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, ape, head_dim):
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


@mx.compile
def _overlap_compress_kv(kv, gate, ape, head_dim):
    B, L, R, D = kv.shape
    gate = gate + ape.astype(gate.dtype)

    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)

    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)

    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


@partial(mx.compile, shapeless=True)
def _indexer_head_reduce(scores, weights, scale):
    return (mx.maximum(scores, 0) * scale * weights).sum(axis=1)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    return (
        mx.exp(logits_a - log_normalizer),
        mx.exp(logits_b - log_normalizer),
    )


# mlx kernels index with int32, so a tensor at or past 2**31 elements has its
# tail silently zeroed. Stay a factor of 2 below that.
_INDEXER_POOL_TILE = 16384
_INDEXER_MAX_ELEMS = 2**30


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        compress_ratio: int,
        head_dim: int,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )

    def project(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.wkv(x), self.wgate(x)

    def consume(
        self,
        kv: mx.array,
        gate: mx.array,
        pool_cache: Optional[PoolingCache],
        offset,
    ) -> mx.array:
        B, _, _ = kv.shape
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=kv.dtype)
        else:
            compress_func = (
                _overlap_compress_kv if self.overlap else _simple_compress_kv
            )
            kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))

            # Overlap (ratio == 4) pools each window from its own lane-B plus
            # the previous window's lane-A; prepend the carried window so the
            # shift sees a real predecessor.
            prev_kv = prev_gate = None
            if self.overlap and pool_cache is not None:
                prev_kv, prev_gate = pool_cache.prev_for_prepend()
            if prev_kv is not None:
                kv = mx.concatenate([prev_kv, kv], axis=1)
                gate = mx.concatenate([prev_gate, gate], axis=1)
                new_pooled = compress_func(kv, gate, self.ape, self.head_dim)[:, 1:]
            else:
                new_pooled = compress_func(kv, gate, self.ape, self.head_dim)

            if self.overlap and pool_cache is not None:
                pool_cache.store_prev(kv, gate, dropped=1 if prev_kv is not None else 0)

            new_pooled = self.norm(new_pooled)
            new_pooled = self.rope(new_pooled[:, None], offset=pool_base).squeeze(1)

        if pool_cache is not None:
            new_pooled = pool_cache.update_and_fetch(new_pooled)
        return new_pooled

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset,
    ) -> mx.array:
        return self.consume(*self.project(x), pool_cache, offset)


def _stable_topk_indices(scores: mx.array, k: int) -> mx.array:
    """Top-k with a deterministic position tie-break and sorted output."""

    partition = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    selected = mx.take_along_axis(scores, partition, axis=-1)
    threshold = mx.min(selected, axis=-1, keepdims=True)

    size = scores.shape[-1]
    positions = mx.arange(size, dtype=mx.uint32)
    region = mx.where(
        scores > threshold,
        0,
        mx.where(scores == threshold, 1, 2),
    ).astype(mx.uint32)
    keys = region * size + positions
    indices = mx.argpartition(keys, kth=k - 1, axis=-1)[..., :k]
    return mx.sort(indices, axis=-1)


class Indexer(nn.Module):
    def __init__(self, config: DeepseekV4FlashVQModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        position_rope: DeepseekV4RoPE,
        pool_cache: Optional[PoolingCache],
        offset,
        compressor_projection: Optional[tuple[mx.array, mx.array]] = None,
        projected_q: Optional[mx.array] = None,
        projected_weights: Optional[mx.array] = None,
    ):
        B, L, _ = x.shape
        if compressor_projection is None:
            pooled = self.compressor(x, pool_cache, offset)
        else:
            pooled = self.compressor.consume(*compressor_projection, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None

        if projected_q is None:
            q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
            q = q.transpose(0, 2, 1, 3)
            q = position_rope(q, offset)
        else:
            q = projected_q

        pmask = pool_cache.make_mask(L, offset) if pool_cache is not None else None
        k = min(self.index_topk, pooled.shape[1])

        weights = (
            self.weights_proj(x) if projected_weights is None else projected_weights
        ).astype(mx.float32) * (self.n_heads**-0.5)
        weights = weights.swapaxes(-1, -2)[..., None]  # (B, H, L, 1)
        qf = q.astype(mx.float32)
        kf = pooled[:, None].swapaxes(-1, -2).astype(mx.float32)  # (B, 1, Dh, P)
        n_pool = pooled.shape[1]
        n_elems = qf.shape[0] * qf.shape[1] * qf.shape[2] * n_pool
        if n_elems < _INDEXER_MAX_ELEMS:
            scores = _indexer_head_reduce(qf @ kf, weights, self.scale)
        else:
            per_pool = max(1, qf.shape[0] * qf.shape[1] * qf.shape[2])
            tile = max(1024, min(_INDEXER_POOL_TILE, _INDEXER_MAX_ELEMS // per_pool))
            scores = mx.concatenate(
                [
                    _indexer_head_reduce(qf @ kf[..., s : s + tile], weights, self.scale)
                    for s in range(0, n_pool, tile)
                ],
                axis=-1,
            )
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None],
                scores,
                mx.finfo(scores.dtype).min,
            )
        return _stable_topk_indices(scores, k)


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    B, H, L, D = q.shape
    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )
    pooled_sq = pooled.squeeze(1)

    q_scaled = q * scale
    local_scores = _apply_score_mask(q_scaled @ local_kv.swapaxes(-1, -2), local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = (q_bl @ pooled_sq.swapaxes(-1, -2)).transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )
    out = local_weights @ local_kv
    pooled_out = pooled_weights.transpose(0, 2, 1, 3) @ pooled_sq
    return (out + pooled_out.transpose(0, 2, 1, 3)).astype(q.dtype)


def _project_attention_output(attn: nn.Module, out: mx.array, offset: Any) -> mx.array:
    out = attn.rope(out, offset, inverse=True)

    def prepare(row: mx.array) -> mx.array:
        batch, _, length, _ = row.shape
        row = row.reshape(batch, attn.o_groups, -1, length, attn.head_dim)
        return row.transpose(0, 1, 3, 2, 4).flatten(-2)

    def finish(row: mx.array) -> mx.array:
        return row.transpose(0, 2, 1, 3).flatten(-2)

    return attn.wo_b(finish(attn.wo_a(prepare(out))))


class _V4AttentionBase(nn.Module):
    """Shared projection surface of the three DSpark attention variants."""

    def __init__(self, config: DeepseekV4FlashVQModelArgs, layer_idx: int, compress_ratio: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

    def _q_kv(self, x: mx.array, offset):
        B, L, _ = x.shape
        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = self.rope(q.transpose(0, 2, 1, 3), offset)
        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        return q_residual, q, kv


class LocalAttention(_V4AttentionBase):
    """DeepSeek-V4 attention with no KV compression (``compress_ratio == 0``)."""

    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        layer_idx: int,
        *,
        compress_ratio: int = 0,
    ):
        if compress_ratio != 0:
            raise ValueError(
                f"LocalAttention is the compress_ratio==0 variant, got {compress_ratio}"
            )
        super().__init__(config, layer_idx, 0)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        _, q, kv = self._q_kv(x, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        out = scaled_dot_product_attention(
            q, kv, kv, cache=cache, scale=self.scale, mask=mask, sinks=sinks
        )
        return _project_attention_output(self, out, offset)


class CompressedAttention(_V4AttentionBase):
    """DeepSeek-V4 attention with pooled KV compression (``compress_ratio == 128``)."""

    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        layer_idx: int,
        *,
        compress_ratio: int | None = None,
    ):
        super().__init__(
            config,
            layer_idx,
            _resolve_compress_ratio(config, layer_idx, compress_ratio),
        )
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

    def __call__(self, x, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        _, q, kv = self._q_kv(x, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        pooled = self.compressor(x, pool_cache, offset)
        pooled_mask = None
        if pooled.shape[1] > 0:
            pooled_mask = (
                pool_cache.make_mask(L, offset) if pool_cache is not None else None
            )
            kv = mx.concatenate([kv, pooled[:, None]], axis=2)
        mask = _extend_mask(mask, pooled_mask, kv.shape[2])
        out = scaled_dot_product_attention(
            q, kv, kv, cache=local_cache, scale=self.scale, mask=mask, sinks=sinks
        )
        return _project_attention_output(self, out, offset)


class SparseCompressedAttention(_V4AttentionBase):
    """DeepSeek-V4 attention with an indexer over pooled KV (``compress_ratio == 4``)."""

    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        layer_idx: int,
        *,
        compress_ratio: int | None = None,
    ):
        super().__init__(
            config,
            layer_idx,
            _resolve_compress_ratio(config, layer_idx, compress_ratio),
        )
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        self.indexer = Indexer(config, self.compress_ratio)

    def __call__(self, x, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        comp_cache = cache[1] if cache is not None else None
        idx_cache = cache[2] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        q_residual, q, kv = self._q_kv(x, offset)
        sinks = self.attn_sink.astype(q.dtype)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        pooled = self.compressor(x, comp_cache, offset)
        pmask = comp_cache.make_mask(L, offset) if comp_cache is not None else None
        if 0 < pooled.shape[1] <= self.indexer.index_topk:
            index_pooled = self.indexer.compressor(x, idx_cache, offset)
            if index_pooled.shape[1] != pooled.shape[1]:
                raise RuntimeError(
                    "DeepSeek-V4 attention/indexer pooling caches diverged"
                )
            topk = mx.broadcast_to(
                mx.arange(pooled.shape[1], dtype=mx.uint32)[None, None],
                (B, L, pooled.shape[1]),
            )
        else:
            topk = self.indexer(x, q_residual, self.rope, idx_cache, offset)

        sparse_mask = None
        if pmask is not None and topk is not None:
            sparse_mask = mx.take_along_axis(
                pmask[None] if pmask.ndim == 2 else pmask, topk, axis=2
            )[:, None]

        if pooled.shape[1] == 0:
            out = scaled_dot_product_attention(
                q, kv, kv, cache=local_cache, scale=self.scale, mask=mask, sinks=sinks
            )
        elif pooled.shape[1] <= self.indexer.index_topk:
            full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
            out = scaled_dot_product_attention(
                q,
                full_kv,
                full_kv,
                cache=local_cache,
                scale=self.scale,
                mask=_extend_mask(mask, pmask, full_kv.shape[2]),
                sinks=sinks,
            )
        else:
            out = _sparse_pooled_attention(
                q, kv, pooled, topk, mask, sparse_mask, self.scale, sinks
            )
        return _project_attention_output(self, out, offset)


def _resolve_compress_ratio(
    config: DeepseekV4FlashVQModelArgs,
    layer_idx: int,
    compress_ratio: int | None,
) -> int:
    """Explicit ratio wins; otherwise index the backbone schedule.

    The MTP stages are indexed 0..N-1 but their ratios live *past*
    ``num_hidden_layers`` in the released ``compress_ratios``. Falling back to
    ``config.compress_ratios[stage]`` for a drafter block would hand it the
    ratio of backbone layer 0/1/2 -- a silent mismatch, since a drafter stage
    accepting an explicit ratio and then ignoring it is indistinguishable from
    one that was configured correctly.
    """

    if compress_ratio is not None:
        return int(compress_ratio)
    return config.compress_ratios[layer_idx]


def v4_attention_factory(
    config: DeepseekV4FlashVQModelArgs,
    layer_idx: int,
    *,
    compress_ratio: int | None = None,
) -> nn.Module:
    ratio = _resolve_compress_ratio(config, layer_idx, compress_ratio)
    if ratio == 0:
        return LocalAttention(config, layer_idx, compress_ratio=0)
    if ratio == 128:
        return CompressedAttention(config, layer_idx, compress_ratio=ratio)
    return SparseCompressedAttention(config, layer_idx, compress_ratio=ratio)


# ---------------------------------------------------------------------------
# MoE blocks
# ---------------------------------------------------------------------------


class DeepseekV4FlashVQMoE(nn.Module):
    """Routed MoE whose experts arrive as a ``QuantizedVQSwitchGLU``.

    ``switch_mlp`` starts as ``None``. Nothing dense is ever allocated for the
    routed experts, so "zero dense routed parameters" holds by construction and
    :func:`dense_deepseek_v4_flash_routed_parameter_names` is an audit of that
    invariant, not the mechanism enforcing it.
    """

    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        layer_idx: int,
        *,
        hash_routing: bool | None = None,
        fp32_swiglu: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts_per_tok = config.num_experts_per_tok
        # Drafter stages run their SwiGLU in float32 on *both* the routed and the
        # shared path (``deepseek_v4_dspark.py:309-313``); the backbone runs
        # neither in float32. ``bind_switch_mlp`` refuses a routed module whose
        # activation disagrees with this flag.
        self.fp32_swiglu = bool(fp32_swiglu)
        self.switch_mlp: QuantizedVQSwitchGLU | SwitchGLU | None = None
        self.gate = MoEGate(config, layer_idx, hash_routing=hash_routing)
        self.shared_experts = DeepseekV4FlashMLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            swiglu_limit=config.swiglu_limit,
            fp32_swiglu=fp32_swiglu,
        )

    def bind_switch_mlp(self, switch_mlp: QuantizedVQSwitchGLU) -> None:
        if switch_mlp.input_dims != self.config.hidden_size:
            raise ValueError("switch_mlp input dimension must match config.hidden_size")
        if switch_mlp.hidden_dims != self.config.moe_intermediate_size:
            raise ValueError(
                "switch_mlp hidden dimension must match config.moe_intermediate_size"
            )
        if switch_mlp.num_experts != self.config.n_routed_experts:
            raise ValueError(
                "switch_mlp expert count must match config.n_routed_experts"
            )
        # A routed module without the clamp computes plain SwiGLU: same shapes,
        # finite output, silently wrong wherever an activation exceeds the
        # limit. Refuse it here rather than let it reach a quality run.
        activation = switch_mlp.get("activation")
        if not isinstance(activation, LimitedSwiGLU):
            raise ValueError(
                "DeepSeek-V4 routed experts require a LimitedSwiGLU activation "
                f"(swiglu_limit={self.config.swiglu_limit}); got "
                f"{type(activation).__name__}"
            )
        if activation.limit != self.config.swiglu_limit:
            raise ValueError(
                "switch_mlp swiglu_limit must match config.swiglu_limit: "
                f"{activation.limit} != {self.config.swiglu_limit}"
            )
        # Same failure mode as the missing clamp: an fp32/bf16 mismatch binds,
        # runs, and stays finite while quietly computing a different function.
        if bool(getattr(activation, "fp32", False)) != self.fp32_swiglu:
            raise ValueError(
                "switch_mlp activation fp32 flag must match this block's: "
                f"expected fp32={self.fp32_swiglu}, got "
                f"fp32={bool(getattr(activation, 'fp32', False))}"
            )
        self.switch_mlp = switch_mlp

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None) -> mx.array:
        if self.switch_mlp is None:
            raise RuntimeError(
                "routed experts must be bound before running a DeepSeek-V4 MoE layer"
            )
        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        return y + self.shared_experts(x)


class DeepseekV4FlashBlock(nn.Module):
    def __init__(self, config: DeepseekV4FlashVQModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = v4_attention_factory(config, layer_idx)
        self.ffn = DeepseekV4FlashVQMoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, h, mask, cache, input_ids) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


# ---------------------------------------------------------------------------
# MTP (DSpark drafter)
# ---------------------------------------------------------------------------


class DSparkContextCache:
    """Physical-ring context K/V owned by one DSpark drafter stage.

    Vendored from ``deepseek_v4_dspark.py:45-110``. Draft-block K/V is
    deliberately *never* committed here: only the projected hidden states of
    target tokens that survived verification are appended, so a rejected draft
    needs no rollback of this cache at all -- it always sits on the committed
    timeline. (That is a different question from
    :meth:`PoolingCache.is_trimmable`, which is about the *backbone's* pooled
    cache.)

    Once the ring is full, slots follow the checkpoint reference's physical
    ``absolute_position % max_size`` order and DSpark attends them in that order
    rather than rotating them back to chronological order. ``_chronological`` /
    ``_physical`` exist only to splice a new append into that ordering.
    """

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        if self.max_size <= 0:
            raise ValueError("DSparkContextCache needs a positive max_size")
        self.offset = 0
        self.keys: mx.array | None = None

    def _chronological(self, value: mx.array) -> mx.array:
        """Temporarily expose a full physical ring oldest-to-newest."""

        if value.shape[2] < self.max_size or self.offset <= self.max_size:
            return value
        cutoff = self.offset % self.max_size
        if cutoff == 0:
            return value
        return mx.concatenate([value[:, :, cutoff:], value[:, :, :cutoff]], axis=2)

    def _physical(self, value: mx.array, offset: int) -> mx.array:
        """Place chronological entries back into absolute-position ring slots."""

        if value.shape[2] < self.max_size:
            return value
        cutoff = offset % self.max_size
        if cutoff == 0:
            return value
        split = self.max_size - cutoff
        return mx.concatenate([value[:, :, split:], value[:, :, :split]], axis=2)

    def append(self, keys: mx.array, *, start_offset: int | None = None) -> None:
        length = int(keys.shape[2])
        if start_offset is not None:
            start = int(start_offset)
            if self.keys is None:
                self.offset = start
            elif self.offset != start:
                raise ValueError(
                    "DSpark context is not contiguous: "
                    f"cache={self.offset}, append={start}"
                )
        if self.keys is None:
            next_keys = keys
        else:
            next_keys = mx.concatenate([self._chronological(self.keys), keys], axis=2)
        next_offset = self.offset + length
        if next_keys.shape[2] > self.max_size:
            next_keys = next_keys[:, :, -self.max_size :]
        self.keys = self._physical(next_keys, next_offset)
        self.offset = next_offset

    def size(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])


class DSparkAttention(LocalAttention):
    """Non-causal block attention over committed context plus the draft block.

    Vendored from ``deepseek_v4_dspark.py:224-287``. This **replaces**
    :meth:`LocalAttention.__call__`; it does not extend it. Three differences
    make reuse impossible rather than merely inconvenient:

    * **No mask at all.** ``mask=None`` is passed to
      ``scaled_dot_product_attention`` on purpose: DSpark proposes a whole block
      in parallel, so draft slot ``j`` attends every other slot of the same
      block, including slots ``> j``. Under the backbone's causal mask that is
      exactly what is forbidden. The committed context is entirely in the past,
      so non-causality is confined to the block.
    * **Its own cache.** Keys come from :class:`DSparkContextCache` (a physical
      ring of committed context) concatenated with this block's own K/V, not
      from ``RotatingKVCache.update_and_fetch`` -- the block's K/V is never
      committed.
    * **A narrower query than key.** ``query_width`` lets the last stage score
      only the ``width`` positions that will actually be proposed while still
      keying against the full block.
    """

    def append_context(
        self,
        main_x: mx.array,
        cache: DSparkContextCache,
        *,
        start_offset: int | None = None,
    ) -> None:
        """Project committed target hidden states into this stage's ring."""

        offset = cache.offset if start_offset is None else int(start_offset)
        batch, length, _ = main_x.shape
        kv = self.kv_norm(self.wkv(main_x)).reshape(batch, 1, length, self.head_dim)
        kv = self.rope(kv, offset)
        cache.append(kv, start_offset=start_offset)

    def __call__(  # type: ignore[override]
        self,
        x: mx.array,
        cache: DSparkContextCache,
        query_width: int | None = None,
    ) -> mx.array:
        batch, block_width, _ = x.shape
        query_width = min(int(query_width or block_width), block_width)
        offset = cache.offset

        q = self.wq_b(self.q_norm(self.wq_a(x[:, :query_width])))
        q = q.reshape(batch, query_width, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = self.rope(q.transpose(0, 2, 1, 3), offset)

        block_kv = self.kv_norm(self.wkv(x)).reshape(
            batch, 1, block_width, self.head_dim
        )
        block_kv = self.rope(block_kv, offset)
        if cache.keys is None:
            kv = block_kv
        else:
            kv = mx.concatenate([cache.keys, block_kv], axis=2)

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        return _project_attention_output(self, out, offset)


class DeepseekV4FlashMarkovHead(nn.Module):
    """DSpark Markov head: a token *lookup* plus a vocabulary projection.

    Both checkpoint tensors are ``[vocab_size, dspark_markov_rank]``, so shape
    alone cannot tell the two roles apart -- which is exactly why this is easy to
    get silently wrong. The reference
    (``.repos/omlx/omlx/patches/mlx_lm_mtp/deepseek_v4_dspark.py:293``) makes
    ``markov_w1`` an :class:`mlx.nn.Embedding` (gather by token id) and only
    ``markov_w2`` an :class:`mlx.nn.Linear` (matmul to vocabulary logits).
    Modelling ``markov_w1`` as a Linear would bind the identical bytes and then
    compute a matmul where a gather belongs.
    """

    def __init__(self, config: DeepseekV4FlashVQModelArgs):
        super().__init__()
        self.markov_w1 = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.markov_w2 = nn.Linear(
            config.dspark_markov_rank, config.vocab_size, bias=False
        )


class DeepseekV4FlashConfidenceHead(nn.Module):
    """``confidence_head.proj``: ``[1, hidden_size + dspark_markov_rank]``."""

    def __init__(self, config: DeepseekV4FlashVQModelArgs):
        super().__init__()
        self.proj = nn.Linear(
            config.hidden_size + config.dspark_markov_rank, 1, bias=False
        )


class DeepseekV4FlashMTPBlock(nn.Module):
    """One DSpark drafter stage -- a full MoE block, not a thin head.

    Every stage carries its own attention, hyper-connections, router (scored,
    never hash), 256 routed FP4 experts and FP8 shared experts. Stage 0 adds the
    ``main_proj``/``main_norm`` fusion of the backbone hidden states captured at
    ``dspark_target_layer_ids`` (``[40, 41, 42]`` here); the last stage adds the
    output ``norm``, its own ``hc_head``, the Markov head and the
    ``confidence_head``. ``compress_ratios`` entries past ``num_hidden_layers``
    are all ``0`` on this release, so every stage uses local attention.

    Both increment-2 requirements are implemented here, from the drafter
    reference (``deepseek_v4_dspark.py:302-352``):

    * ``DSparkBlock`` sets ``fp32=True`` on the routed ``LimitedSwiGLU`` and
      ``fp32_swiglu = True`` on the shared experts (:309-313). The drafter runs
      its SwiGLU in float32 while the backbone does not. It is *enforced*, not
      merely set: ``ffn.bind_switch_mlp`` refuses a routed module whose
      activation is not fp32, because that mismatch binds, runs, stays finite,
      and only shows up in the acceptance rate.
    * ``DSparkAttention`` (:224+) is a **non-causal block attention** over
      committed context plus the draft block, with its own
      :class:`DSparkContextCache` and a ``query_width`` narrower than the block.
      It replaces the causal ``LocalAttention`` forward rather than wrapping it.
    """

    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        stage: int,
        *,
        compress_ratio: int = 0,
        is_entry: bool = False,
        is_exit: bool = False,
        main_proj_in_features: int | None = None,
    ):
        super().__init__()
        self.stage = stage
        self.is_entry = is_entry
        self.is_exit = is_exit
        ratio = _resolve_compress_ratio(config, stage, compress_ratio)
        if ratio != 0:
            raise ValueError(
                "DSpark drafter stages are local (non-causal block) attention "
                "only; the released compress_ratios drafter tail is all zero. "
                f"Stage {stage} was given compress_ratio={ratio}."
            )
        self.attn = DSparkAttention(config, stage, compress_ratio=0)
        # Drafter routers ship ``gate.bias`` (never ``tid2eid``): scored routing
        # on every stage, regardless of ``num_hash_layers``.
        self.ffn = DeepseekV4FlashVQMoE(
            config, stage, hash_routing=False, fp32_swiglu=True
        )
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        if is_entry:
            # Reference: ``hidden_size * len(dspark_target_layer_ids)``
            # (deepseek_v4_dspark.py:319). On this release that is 4096 * 3 =
            # 12288, which matches the shipped ``mtp.0.main_proj`` -- but the
            # width is the *number of captured backbone layers*, not a constant
            # 3, so it is derived rather than hardcoded.
            in_features = (
                main_proj_in_features
                if main_proj_in_features is not None
                else config.main_proj_in_features
            )
            self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.main_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        if is_exit:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hc_head = HyperHead(config)
            self.markov_head = DeepseekV4FlashMarkovHead(config)
            self.confidence_head = DeepseekV4FlashConfidenceHead(config)

    def __call__(
        self,
        hidden: mx.array,
        input_ids: mx.array,
        cache: DSparkContextCache,
        output_width: int | None = None,
    ) -> mx.array:
        """One drafter stage over a 4D hyper-stream draft block.

        ``hidden`` is ``[B, block, hc_mult, hidden_size]``; the return has the
        same rank, narrowed to ``output_width`` positions when one is given.
        The narrowing happens *after* attention (so the dropped slots still act
        as keys) and *before* the MoE, which is where the reference puts it
        (``deepseek_v4_dspark.py:328-352``) and is the only reason a
        ``query_width`` narrower than the block saves anything.
        """

        residual = hidden
        x, post, comb = self.attn_hc(hidden)
        x = self.attn(self.attn_norm(x), cache, query_width=output_width)
        if output_width is not None and output_width < hidden.shape[1]:
            residual = residual[:, :output_width]
            post = post[:, :output_width]
            comb = comb[:, :output_width]
            input_ids = input_ids[:, :output_width]
        hidden = hc_expand(x, residual, post, comb)

        residual = hidden
        x, post, comb = self.ffn_hc(hidden)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class DeepseekV4FlashMTPDrafter(nn.Module):
    """Container for ``mtp.{0..N-1}``, bound intentionally rather than skipped.

    The stage loop lives here; the whole-model entry points
    (:meth:`DeepseekV4FlashVQModel.dspark_forward` and friends) wrap it because
    they also need ``embed_tokens`` and the shared ``lm_head``.
    """

    def __init__(self, config: DeepseekV4FlashVQModelArgs, num_blocks: int):
        super().__init__()
        self.config = config
        ratios = list(getattr(config, "mtp_compress_ratios", []) or [])
        ratios += [0] * max(num_blocks - len(ratios), 0)
        self.blocks = [
            DeepseekV4FlashMTPBlock(
                config,
                stage,
                compress_ratio=ratios[stage],
                is_entry=stage == 0,
                is_exit=stage == num_blocks - 1,
            )
            for stage in range(num_blocks)
        ]

    def __len__(self) -> int:
        return len(self.blocks)

    def make_context_cache(self) -> list[DSparkContextCache]:
        """One committed-context ring per stage, sized like the local window."""

        return [
            DSparkContextCache(self.config.sliding_window) for _ in self.blocks
        ]

    def append_context(
        self,
        main_hidden: mx.array,
        cache: Sequence[DSparkContextCache],
        *,
        start_offset: int | None = None,
    ) -> mx.array:
        """Fuse the backbone taps once, then key them into every stage's ring.

        ``main_hidden`` is ``[B, L, hidden_size * len(dspark_target_layer_ids)]``
        -- the concatenated per-layer taps produced by
        :meth:`DeepseekV4FlashBackbone.__call__` with ``return_dspark_hidden``.
        Stage 0 owns ``main_proj``/``main_norm``; the projected result is shared
        by all stages (``deepseek_v4_model.py:455-473``), so it is computed once.
        """

        if len(cache) != len(self.blocks):
            raise ValueError(
                f"DSpark context cache count {len(cache)} != {len(self.blocks)} stages"
            )
        entry = self.blocks[0]
        expected = entry.main_proj.weight.shape[1]
        if main_hidden.shape[-1] != expected:
            raise ValueError(
                "DSpark context width must be hidden_size * "
                f"len(dspark_target_layer_ids) = {expected}, got "
                f"{main_hidden.shape[-1]}"
            )
        main_x = entry.main_norm(entry.main_proj(main_hidden))
        for block, stage_cache in zip(self.blocks, cache):
            block.attn.append_context(main_x, stage_cache, start_offset=start_offset)
        return main_x

    def __call__(
        self,
        hidden: mx.array,
        draft_ids: mx.array,
        cache: Sequence[DSparkContextCache],
        *,
        width: int,
    ) -> mx.array:
        """Run every stage over the draft block; return the last stage's hidden.

        Only the final stage narrows to ``width``: the earlier stages need the
        whole block so its later slots can still serve as keys.
        """

        last = len(self.blocks) - 1
        for stage_idx, (block, stage_cache) in enumerate(zip(self.blocks, cache)):
            hidden = block(
                hidden,
                draft_ids,
                stage_cache,
                output_width=width if stage_idx == last else None,
            )
        return hidden


# ---------------------------------------------------------------------------
# Backbone / model
# ---------------------------------------------------------------------------


def _dspark_tap(h: mx.array) -> mx.array:
    """Collapse one backbone layer's hyper-stream into a DSpark target tap.

    ``h`` is ``[B, L, hc_mult, hidden]``; the drafter consumes the mean over the
    hyper-connection streams (``deepseek_v4_model.py:236``), not the ``hc_head``
    collapse the LM head uses. A module-level function so the opt-in capture has
    exactly one call site and tests can count it.
    """

    return h.mean(axis=2)


class DeepseekV4FlashBackbone(nn.Module):
    def __init__(self, config: DeepseekV4FlashVQModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4FlashBlock(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        *,
        return_dspark_hidden: bool = False,
    ):
        """Run the backbone. **Always pass a cache from :meth:`make_cache`.**

        ``cache=None`` runs and produces finite logits, but it is not the same
        function: the pooled-KV visibility mask comes from
        ``PoolingCache.make_mask``, so with no cache every query attends every
        pooled window -- including windows built from tokens that come after it.
        Local attention is unaffected (its mask comes from
        ``create_attention_mask``), so an all-``compress_ratio==0`` config gives
        bit-identical results either way and a compressed one does not. This is
        vendored upstream behaviour, not a strip regression, and it is pinned by
        ``test_forward_without_a_cache_drops_the_pooled_causal_mask``.

        ``return_dspark_hidden`` is **opt-in and off by default**. When set, the
        return becomes ``(out, taps)`` where ``taps`` is the concatenation of
        :func:`_dspark_tap` at each ``config.dspark_target_layer_ids`` layer, in
        that list's order -- the ``[B, L, hidden * n_targets]`` input
        ``mtp.0.main_proj`` expects. The default path allocates nothing extra
        and returns a bare array: every existing caller (including the teacher
        runner's layer-major prefill) keeps its return shape and its cost.
        """

        h = self.embed_tokens(inputs)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
            )
        )
        if cache is None:
            cache = [None] * len(self.layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )
        if not return_dspark_hidden:
            for layer, layer_cache in zip(self.layers, cache):
                h = layer(h, mask, layer_cache, inputs)
            return self.norm(self.hc_head(h))

        target_ids = list(self.args.dspark_target_layer_ids)
        if not target_ids:
            raise ValueError(
                "return_dspark_hidden requires config.dspark_target_layer_ids"
            )
        out_of_range = [i for i in target_ids if not 0 <= i < len(self.layers)]
        if out_of_range:
            raise ValueError(
                f"dspark_target_layer_ids outside the backbone: {out_of_range}"
            )
        wanted = set(target_ids)
        taps: dict[int, mx.array] = {}
        for layer_idx, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, layer_cache, inputs)
            if layer_idx in wanted:
                taps[layer_idx] = _dspark_tap(h)
        if len(taps) != len(wanted):
            raise RuntimeError(
                f"DSpark target tap mismatch: captured={len(taps)}, "
                f"expected={len(wanted)}"
            )
        # Concatenated in ``dspark_target_layer_ids`` order, not sorted order:
        # ``main_proj``'s columns were trained against that order.
        fused = mx.concatenate([taps[i] for i in target_ids], axis=-1)
        return self.norm(self.hc_head(h)), fused


class DeepseekV4FlashVQModel(nn.Module):
    def __init__(
        self,
        config: DeepseekV4FlashVQModelArgs,
        *,
        with_mtp: bool = True,
    ):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4FlashBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_drafter: DeepseekV4FlashMTPDrafter | None = None
        if with_mtp and config.num_mtp_blocks > 0:
            self.mtp_drafter = DeepseekV4FlashMTPDrafter(
                config, config.num_mtp_blocks
            )

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(key: str) -> bool:
            return not (
                "attn_sink" in key
                or "e_score_correction_bias" in key
                or "tid2eid" in key
                or ".attn_hc." in key
                or ".ffn_hc." in key
                or ".hc_head." in key
            )

        return predicate

    def make_cache(self):
        # The undo window is derived from this checkpoint's own block size, not
        # a constant that happens to be big enough today.
        undo = deepseek_v4_flash_pooling_undo_window(self.args)
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            local = DeepseekV4FlashRotatingKVCache(
                max_size=self.args.sliding_window,
                max_undo_update=undo,
            )
            if ratio == 0:
                caches.append(local)
            elif isinstance(layer.attn, SparseCompressedAttention):
                caches.append(
                    CacheList(
                        local,
                        PoolingCache(ratio, max_undo_update=undo),
                        PoolingCache(ratio, max_undo_update=undo),
                    )
                )
            else:
                caches.append(
                    CacheList(
                        local,
                        PoolingCache(ratio, max_undo_update=undo),
                    )
                )
        return caches

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        *,
        return_dspark_hidden: bool = False,
    ):
        if not return_dspark_hidden:
            return self.lm_head(self.model(inputs, cache))
        hidden, taps = self.model(inputs, cache, return_dspark_hidden=True)
        return self.lm_head(hidden), taps

    # -- DSpark drafter -----------------------------------------------------

    def _drafter(self) -> DeepseekV4FlashMTPDrafter:
        if self.mtp_drafter is None:
            raise RuntimeError(
                "this model was built with_mtp=False; there is no DSpark drafter"
            )
        return self.mtp_drafter

    def make_mtp_cache(self) -> list[DSparkContextCache]:
        """One :class:`DSparkContextCache` per drafter stage."""

        return self._drafter().make_context_cache()

    def dspark_append_context(
        self,
        main_hidden: mx.array,
        cache: Sequence[DSparkContextCache],
        *,
        start_offset: int | None = None,
    ) -> mx.array:
        """Commit backbone taps into the drafter's context rings."""

        return self._drafter().append_context(
            main_hidden, cache, start_offset=start_offset
        )

    def dspark_draft_block(
        self,
        anchor_ids: mx.array,
        cache: Sequence[DSparkContextCache],
        *,
        draft_length: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Propose one block from already-committed context.

        Returns ``(logits, head_hidden)``, both narrowed to the draft width:
        ``logits`` is ``[B, width, vocab]`` through the shared ``lm_head``, and
        ``head_hidden`` is ``[B, width, hidden_size]`` -- the last stage's
        ``hc_head`` output, i.e. the exact input ``norm`` + ``lm_head`` consume.
        That is what MTP-head recovery training regresses from, which is why it
        is returned rather than recomputed.

        The block's first slot carries ``anchor_ids`` (the newest
        target-confirmed token); the rest carry the checkpoint's
        ``dspark_noise_token_id``. Reference: ``deepseek_v4_model.py:475-530``.
        """

        drafter = self._drafter()
        args = self.args
        max_width = int(args.dspark_block_size)
        if max_width <= 0:
            raise ValueError(
                "dspark_block_size must be positive to run the DSpark drafter"
            )
        width = int(draft_length or max_width)
        width = max(1, min(width, max_width))

        anchor_ids = anchor_ids.reshape(anchor_ids.shape[0], -1)[:, -1:]
        noise = mx.full(
            (anchor_ids.shape[0], width),
            int(args.dspark_noise_token_id),
            dtype=anchor_ids.dtype,
        )
        draft_ids = mx.concatenate([anchor_ids, noise[:, 1:]], axis=1)

        hidden = self.model.embed_tokens(draft_ids)
        hidden = mx.contiguous(
            mx.broadcast_to(
                hidden[:, :, None, :],
                (hidden.shape[0], hidden.shape[1], args.hc_mult, hidden.shape[-1]),
            )
        )
        hidden = drafter(hidden, draft_ids, cache, width=width)

        final = drafter.blocks[-1]
        head_hidden = final.hc_head(hidden)
        logits = self.lm_head(final.norm(head_hidden))
        return logits[:, :width], head_hidden[:, :width]

    def dspark_forward(
        self,
        main_hidden: mx.array,
        anchor_ids: mx.array,
        cache: Sequence[DSparkContextCache] | None = None,
        *,
        draft_length: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Commit ``main_hidden`` as context, then propose one block.

        The reference's single entry point (``deepseek_v4_model.py:475``). Split
        into append + draft here because the teacher runner walks supervised
        positions forward and appends one token at a time between drafts, and
        re-appending the whole prefix per position would be quadratic.
        """

        if cache is None:
            cache = self.make_mtp_cache()
        self.dspark_append_context(main_hidden, cache)
        return self.dspark_draft_block(
            anchor_ids, cache, draft_length=draft_length
        )

    def dspark_calibration_forward(
        self,
        target_hiddens: mx.array,
        input_ids: mx.array,
        *,
        draft_length: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Draft the continuation of a token prefix. **This fixes the seam.**

        Vendored 1:1 from ``deepseek_v4_model.py:540-554``, and it exists here
        precisely because the off-by-one is invisible without it. Given a prefix
        of ``L`` tokens whose last token sits at absolute position ``p``:

        * the committed context is ``target_hiddens[:, :-1]`` -- taps for
          ``0 .. p-1``, **excluding the anchor's own tap**;
        * the anchor is ``input_ids[:, -1:]`` -- the token at ``p``;
        * so the ring ends at offset ``p``, the draft block's RoPE starts at
          ``p``, and slot 0 holds token ``p`` and predicts ``p+1``.

        Committing the anchor's own tap instead (context through ``p``
        inclusive) still runs, still produces finite logits, and is wrong twice
        over: every slot's RoPE position is shifted by one relative to the
        Wave-5 verify runtime, and the drafter is conditioned on a backbone
        hidden state it cannot have at inference time. ``dspark.py:204-209``
        (``take_primed`` refusing unless the ring is exactly one token behind
        the target cache) is the same invariant stated from the runtime side.

        Anything producing MTP training targets should call **this**, not
        :meth:`dspark_forward` with a hand-sliced context.
        """

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if int(input_ids.shape[1]) != int(target_hiddens.shape[1]):
            raise ValueError(
                "dspark_calibration_forward needs one target tap per input "
                f"token: {target_hiddens.shape[1]} taps for "
                f"{input_ids.shape[1]} ids"
            )
        width = min(
            int(self.args.dspark_block_size),
            max(1, int(input_ids.shape[1]) - 1),
        )
        if draft_length is not None:
            width = max(1, min(int(draft_length), width))
        return self.dspark_forward(
            target_hiddens[:, :-1],
            input_ids[:, -1:],
            self.make_mtp_cache(),
            draft_length=width,
        )

    def dspark_markov(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Return DSpark's previous-token logit bias and its rank-R embedding.

        ``markov_w1`` is an Embedding (gather by token id), ``markov_w2`` a
        Linear to vocabulary logits -- see
        :class:`DeepseekV4FlashMarkovHead`.
        """

        head = self._drafter().blocks[-1].markov_head
        embedding = head.markov_w1(token_ids)
        return head.markov_w2(embedding), embedding

    def dspark_confidence(
        self, head_hidden: mx.array, markov_embedding: mx.array
    ) -> mx.array:
        """Score the drafter's own confidence in a proposed slot.

        ``confidence_head.proj`` is ``[1, hidden_size + dspark_markov_rank]``,
        which fixes the input as the drafter's final hidden concatenated with
        the Markov embedding, and the output as one scalar per position.

        .. warning::
           **The reference constructs this head but never calls it**
           (``deepseek_v4_dspark.py:296-300, 326`` -- the only hit for
           "confidence" in the whole omlx tree). The wiring above is forced by
           the shapes; the *squashing* is not. This returns the raw score, so a
           caller that wants a probability must decide on and pin the
           activation itself. Do not treat this number as a calibrated
           acceptance probability until a reference call site exists.
        """

        joined = mx.concatenate(
            [head_hidden, markov_embedding.astype(head_hidden.dtype)], axis=-1
        )
        return self._drafter().blocks[-1].confidence_head.proj(joined)[..., 0]


# ---------------------------------------------------------------------------
# VQ expert binding
# ---------------------------------------------------------------------------


def load_deepseek_v4_flash_vq_switch_glu(
    artifact_dir: str | Path,
    layer: int,
    *,
    swiglu_limit: float,
    prefix: str | None = None,
    file_stem: str | None = None,
    fp32_swiglu: bool = False,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> QuantizedVQSwitchGLU:
    """Load one layer's three VQ projections into a ``QuantizedVQSwitchGLU``.

    Artifact file names follow the family convention
    ``layer-{layer:05d}-{gate,up,down}_proj.safetensors``; the tensor prefix
    inside each file is ``model.layers.{layer}.ffn.switch_mlp.{projection}``.
    That convention is inherited from the GLM-5.2 materializer -- the V4 one
    does not exist yet, so it is an expectation this loader states, not a
    contract anything has produced.

    ``swiglu_limit`` is not optional. Upstream builds the routed ``SwitchGLU``
    with ``activation=LimitedSwiGLU(config.swiglu_limit)``
    (``deepseek_v4_model.py:858-864``), which clamps ``gate <= limit`` and
    ``up`` into ``[-limit, +limit]`` *before* the SiLU. Plain
    ``silu(gate) * up`` is a different function wherever an activation exceeds
    10.0, and it fails silently -- the shapes match and the output is finite.
    Callers must therefore pass the model args so the clamp travels with the
    binding.
    """

    tensor_prefix = prefix or f"model.layers.{layer}.ffn.switch_mlp"
    artifact_stem = file_stem or f"layer-{layer:05d}"
    artifact_root = Path(artifact_dir)

    def path_for(projection: str) -> Path:
        filename = f"{artifact_stem}-{projection}.safetensors"
        if artifact_paths is None:
            return artifact_root / filename
        try:
            return Path(artifact_paths[filename])
        except KeyError as error:
            raise ValueError(
                f"authenticated DeepSeek-V4 artifact map is missing {filename}"
            ) from error

    return QuantizedVQSwitchGLU(
        gate_proj=load_quantized_vq_switch_linear(
            path_for("gate_proj"), f"{tensor_prefix}.gate_proj"
        ),
        up_proj=load_quantized_vq_switch_linear(
            path_for("up_proj"), f"{tensor_prefix}.up_proj"
        ),
        down_proj=load_quantized_vq_switch_linear(
            path_for("down_proj"), f"{tensor_prefix}.down_proj"
        ),
        activation=LimitedSwiGLU(swiglu_limit, fp32=fp32_swiglu),
    )


def bind_deepseek_v4_flash_vq_experts(
    model: DeepseekV4FlashVQModel,
    artifact_dir: str | Path,
    *,
    layers: Sequence[int] | None = None,
    strict: bool = False,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> tuple[int, ...]:
    """Swap each routed-expert module for a ``QuantizedVQSwitchGLU``.

    Every V4-Flash layer is sparse (there is no ``first_k_dense_replace``), so
    the expected set is simply ``range(num_hidden_layers)``. Each bound module
    carries ``LimitedSwiGLU(args.swiglu_limit)`` so the routed activation clamp
    matches upstream instead of degrading to plain SwiGLU.
    """

    expected = tuple(range(model.args.num_hidden_layers))
    if layers is None:
        requested = expected
    else:
        requested = tuple(int(layer) for layer in layers)
        if len(set(requested)) != len(requested):
            raise ValueError("requested layers contain duplicates")
        unknown = [layer for layer in requested if layer not in expected]
        if unknown:
            raise ValueError(f"requested layers outside the backbone: {unknown}")
    if strict and requested != expected:
        raise ValueError(
            "strict binding requires the exact runtime sparse-layer set: "
            f"requested={requested} expected={expected}"
        )

    target = set(requested)
    bound: list[int] = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in target:
            continue
        if not isinstance(layer.ffn, DeepseekV4FlashVQMoE):
            raise ValueError(f"DeepSeek-V4 layer {layer_idx} is not a routed MoE layer")
        layer.ffn.bind_switch_mlp(
            load_deepseek_v4_flash_vq_switch_glu(
                artifact_dir,
                layer_idx,
                swiglu_limit=model.args.swiglu_limit,
                artifact_paths=artifact_paths,
            )
        )
        bound.append(layer_idx)
    if strict and tuple(bound) != expected:
        raise ValueError("bound layers do not match the exact runtime layer set")
    return tuple(bound)


def bind_deepseek_v4_flash_mtp_vq_experts(
    model: DeepseekV4FlashVQModel,
    artifact_dir: str | Path,
    *,
    stages: Sequence[int] | None = None,
    strict: bool = False,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> tuple[int, ...]:
    """Bind the VQ-routed experts for every embedded DSpark stage.

    MTP files use ``mtp-{stage:05d}-{projection}.safetensors`` and tensor
    prefix ``mtp_drafter.blocks.{stage}.ffn.switch_mlp``. The drafter's routed
    SwiGLU is evaluated in float32, matching its resident/shared path.
    """

    if model.mtp_drafter is None:
        raise ValueError("model was built without an MTP drafter")
    expected = tuple(range(len(model.mtp_drafter.blocks)))
    requested = expected if stages is None else tuple(int(stage) for stage in stages)
    if len(set(requested)) != len(requested):
        raise ValueError("requested MTP stages contain duplicates")
    unknown = [stage for stage in requested if stage not in expected]
    if unknown:
        raise ValueError(f"requested stages outside the MTP drafter: {unknown}")
    if strict and requested != expected:
        raise ValueError(
            "strict binding requires the exact MTP stage set: "
            f"requested={requested} expected={expected}"
        )

    bound: list[int] = []
    for stage in requested:
        block = model.mtp_drafter.blocks[stage]
        if not isinstance(block.ffn, DeepseekV4FlashVQMoE):
            raise ValueError(f"DeepSeek-V4 MTP stage {stage} is not a routed MoE")
        block.ffn.bind_switch_mlp(
            load_deepseek_v4_flash_vq_switch_glu(
                artifact_dir,
                stage,
                swiglu_limit=model.args.swiglu_limit,
                prefix=f"mtp_drafter.blocks.{stage}.ffn.switch_mlp",
                file_stem=f"mtp-{stage:05d}",
                fp32_swiglu=True,
                artifact_paths=artifact_paths,
            )
        )
        bound.append(stage)
    if strict and tuple(bound) != expected:
        raise ValueError("bound stages do not match the exact MTP stage set")
    return tuple(bound)


def has_unbound_deepseek_v4_flash_vq_experts(model: DeepseekV4FlashVQModel) -> bool:
    """True while any backbone routed-expert module is still unbound."""

    return any(
        isinstance(layer.ffn, DeepseekV4FlashVQMoE) and layer.ffn.switch_mlp is None
        for layer in model.model.layers
    )


def has_unbound_deepseek_v4_flash_mtp_experts(model: DeepseekV4FlashVQModel) -> bool:
    """True while any drafter stage's routed experts are still unbound."""

    if model.mtp_drafter is None:
        return False
    return any(block.ffn.switch_mlp is None for block in model.mtp_drafter.blocks)


def dense_deepseek_v4_flash_routed_parameter_names(
    model: DeepseekV4FlashVQModel,
    *,
    include_mtp: bool = False,
) -> tuple[str, ...]:
    """Routed-expert parameters still stored densely.

    A bound VQ layer exposes ``codes``/``scales``/``codebook`` only, so a
    non-empty result means a dense ``SwitchLinear.weight`` (or an unsanitized
    per-expert tensor) survived somewhere it should not have.
    """

    prefixes = ("model.layers.",) if not include_mtp else ("model.layers.", "mtp_drafter.")
    return tuple(
        sorted(
            key
            for key, _value in tree_flatten(model.parameters())
            if key.startswith(prefixes)
            and (
                ".ffn.experts." in key
                or (".ffn.switch_mlp." in key and key.endswith(".weight"))
            )
        )
    )


# ---------------------------------------------------------------------------
# Non-VQ (resident) weight binding
# ---------------------------------------------------------------------------

_FP8_CODE_DTYPE = "F8_E4M3"
_FP8_SCALE_DTYPE = "F8_E8M0"
_FP8_BLOCK = (128, 128)
# Routed experts ship FP4 packed two-per-byte; safetensors has no FP4 dtype, so
# the release declares the container ``I8``. It is a byte bag, not int8 numbers.
_FP4_CODE_DTYPE = "I8"

# The one tensor whose checkpoint layout legitimately differs from its module
# layout: ``attn.wo_a`` ships flat as ``(o_groups * o_lora_rank, ·)`` and the
# module holds MultiLinear's ``(o_groups, o_lora_rank, ·)``. Everything else must
# arrive already-shaped; see the reshape guard in the binder.
_RESHAPE_ALLOWED_RE = re.compile(r"\.attn\.wo_a\.weight$")
_RESHAPE_ALLOWED_DESCRIPTION = "attn.wo_a.weight"

_SHARED_EXPERT_RENAME = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_HC_RENAME = {
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
    "hc_head_fn": "hc_head.fn",
    "hc_head_base": "hc_head.base",
    "hc_head_scale": "hc_head.scale",
}
_TOP_LEVEL_RENAME = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
    "hc_head_fn": "model.hc_head.fn",
    "hc_head_base": "model.hc_head.base",
    "hc_head_scale": "model.hc_head.scale",
}


@dataclass(frozen=True)
class DeepseekV4FlashNonVQBindReport:
    """What :func:`bind_deepseek_v4_flash_non_vq_weights` did, tensor by tensor."""

    bound_backbone_parameters: tuple[str, ...] = ()
    bound_mtp_parameters: tuple[str, ...] = ()
    fp8_block_decoded_tensors: tuple[str, ...] = ()
    direct_tensors: tuple[str, ...] = ()
    preserved_integer_tensors: tuple[str, ...] = ()
    reshaped_tensors: tuple[str, ...] = ()
    skipped_routed_expert_tensors: tuple[str, ...] = ()
    skipped_mtp_routed_expert_tensors: tuple[str, ...] = ()
    skipped_mtp_tensors: tuple[str, ...] = ()
    skipped_out_of_scope_tensors: tuple[str, ...] = ()
    skipped_unmatched_tensors: tuple[str, ...] = ()
    missing_model_parameters: tuple[str, ...] = ()

    @property
    def bound_count(self) -> int:
        return len(self.bound_backbone_parameters) + len(self.bound_mtp_parameters)

    @property
    def backbone_bound_count(self) -> int:
        return len(self.bound_backbone_parameters)

    @property
    def mtp_bound_count(self) -> int:
        return len(self.bound_mtp_parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "bound_backbone_parameters": list(self.bound_backbone_parameters),
            "bound_mtp_parameters": list(self.bound_mtp_parameters),
            "bound_count": self.bound_count,
            "backbone_bound_count": self.backbone_bound_count,
            "mtp_bound_count": self.mtp_bound_count,
            "fp8_block_decoded_tensors": list(self.fp8_block_decoded_tensors),
            "direct_tensors": list(self.direct_tensors),
            "preserved_integer_tensors": list(self.preserved_integer_tensors),
            "reshaped_tensors": list(self.reshaped_tensors),
            "skipped_routed_expert_tensors": list(self.skipped_routed_expert_tensors),
            "skipped_mtp_routed_expert_tensors": list(
                self.skipped_mtp_routed_expert_tensors
            ),
            "skipped_mtp_tensors": list(self.skipped_mtp_tensors),
            "skipped_out_of_scope_tensors": list(self.skipped_out_of_scope_tensors),
            "skipped_unmatched_tensors": list(self.skipped_unmatched_tensors),
            "missing_model_parameters": list(self.missing_model_parameters),
        }


def _load_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"{index_path} has no usable weight_map")
        return {str(name): str(shard) for name, shard in weight_map.items()}

    weight_map: dict[str, str] = {}
    shards = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{checkpoint_dir} contains no safetensors shards")
    for shard in shards:
        for name in read_safetensors_file_header(shard).tensors:
            weight_map[name] = shard.name
    return weight_map


def _checkpoint_layer_index(name: str) -> int | None:
    if not name.startswith("layers."):
        return None
    parts = name.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _checkpoint_mtp_stage(name: str) -> int | None:
    if not name.startswith("mtp."):
        return None
    parts = name.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _is_routed_expert_tensor(name: str) -> bool:
    return ".ffn.experts." in name


def _map_block_suffix(suffix: str) -> str | None:
    """Map an intra-block checkpoint suffix onto the module-tree suffix."""

    if suffix in _HC_RENAME:
        return _HC_RENAME[suffix]
    if suffix == "ffn.gate.bias":
        return "ffn.gate.e_score_correction_bias"
    if suffix.startswith("ffn.shared_experts."):
        parts = suffix.split(".")
        # ffn.shared_experts.<w1|w2|w3>.<weight>
        if len(parts) >= 4 and parts[2] in _SHARED_EXPERT_RENAME:
            parts[2] = _SHARED_EXPERT_RENAME[parts[2]]
            return ".".join(parts)
        return None
    return suffix


def _map_checkpoint_name(name: str) -> str | None:
    """Checkpoint tensor name -> module parameter name (``None`` = unmapped)."""

    if name in _TOP_LEVEL_RENAME:
        return _TOP_LEVEL_RENAME[name]

    layer = _checkpoint_layer_index(name)
    if layer is not None:
        suffix = _map_block_suffix(name.split(".", 2)[2])
        return None if suffix is None else f"model.layers.{layer}.{suffix}"

    stage = _checkpoint_mtp_stage(name)
    if stage is not None:
        suffix = _map_block_suffix(name.split(".", 2)[2])
        return None if suffix is None else f"mtp_drafter.blocks.{stage}.{suffix}"

    return None


def _read_raw(header_size: int, tensor, handle) -> bytes:
    start, end = tensor.data_offsets
    handle.seek(8 + header_size + start)
    return handle.read(end - start)


def _as_numpy(dtype: str, raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").copy()
        return words.reshape(shape)
    numpy_dtype = {
        "F32": "<f4",
        "F16": "<f2",
        "F64": "<f8",
        "I64": "<i8",
        "I32": "<i4",
        "I16": "<i2",
        "I8": "i1",
        "U8": "u1",
        "BOOL": "?",
    }.get(dtype)
    if numpy_dtype is None:
        # F8_E4M3 / F8_E8M0 and anything else opaque: raw code bytes. Reading
        # them as uint8 is a reinterpret, never a conversion -- which is exactly
        # what keep.convert.fp8_block wants.
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    return np.frombuffer(raw, dtype=numpy_dtype).reshape(shape).copy()


def _bf16_words_to_mx(words: np.ndarray) -> mx.array:
    return mx.array(words).view(mx.bfloat16)


def _decode_source_tensor(
    dtype: str,
    raw: bytes,
    shape: tuple[int, ...],
    scale: tuple[str, bytes, tuple[int, ...]] | None,
    *,
    compute_dtype: mx.Dtype,
) -> tuple[mx.array, str]:
    """Return ``(value, kind)`` where kind is ``fp8_block``/``direct``/``integer``."""

    if scale is not None:
        scale_dtype, scale_raw, scale_shape = scale
        if dtype != _FP8_CODE_DTYPE:
            raise ValueError(
                f"a .scale companion requires {_FP8_CODE_DTYPE} codes, got {dtype}"
            )
        if scale_dtype != _FP8_SCALE_DTYPE:
            raise ValueError(
                f"block scales must be {_FP8_SCALE_DTYPE}, got {scale_dtype}"
            )
        codes = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        scales = np.frombuffer(scale_raw, dtype=np.uint8).reshape(scale_shape).copy()
        decoded = dequantize_fp8_block(codes, scales, block_size=_FP8_BLOCK)
        return mx.array(decoded).astype(compute_dtype), "fp8_block"

    if dtype == _FP8_CODE_DTYPE:
        # The reverse direction (a ``.scale`` on non-FP8 codes) already raises.
        # Without this, an FP8 weight whose ``.scale`` is missing from the index
        # would bind as raw uint8 code bytes -- right shape, finite values, and
        # off by whatever the block exponents were.
        raise ValueError(
            f"{_FP8_CODE_DTYPE} codes have no .scale companion in the index; "
            "refusing to bind undecoded FP8 bytes as if they were values"
        )
    if dtype == _FP8_SCALE_DTYPE:
        raise ValueError(
            f"{_FP8_SCALE_DTYPE} block scales are only meaningful alongside "
            f"their {_FP8_CODE_DTYPE} weight; refusing to bind one as a value"
        )

    if dtype == "BF16":
        return _bf16_words_to_mx(_as_numpy(dtype, raw, shape)), "direct"
    values = _as_numpy(dtype, raw, shape)
    if np.issubdtype(values.dtype, np.integer) and values.dtype != np.uint8:
        # Routing tables (``tid2eid``) keep their native width: never quantized,
        # never re-typed.
        return mx.array(values), "integer"
    return mx.array(values), "direct"


def bind_deepseek_v4_flash_non_vq_weights(
    model: DeepseekV4FlashVQModel,
    checkpoint_dir: str | Path,
    *,
    layers: Sequence[int] | None = None,
    include_mtp: bool = True,
    strict: bool = True,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> DeepseekV4FlashNonVQBindReport:
    """Bind every non-routed tensor, shard by shard, off the safetensors index.

    Resident ``F8_E4M3`` weights are decoded against their ``F8_E8M0`` 128x128
    block scales via :func:`keep.convert.fp8_block.dequantize_fp8_block`; ``BF16``
    and ``F32`` tensors load directly; ``I64`` routing tables keep their native
    dtype. Routed experts are skipped -- they are the VQ path.

    ``layers`` restricts the backbone layers touched (a layer-0 smoke test does
    not need 163 GB of I/O). ``strict`` is only meaningful for a full bind:
    it is forced off when ``layers`` or ``include_mtp=False`` deliberately leaves
    parameters unbound.
    """

    checkpoint_path = Path(checkpoint_dir)
    weight_map = _load_weight_map(checkpoint_path)
    partial_scope = layers is not None or not include_mtp
    if strict and partial_scope:
        strict = False

    layer_filter = None if layers is None else {int(layer) for layer in layers}

    target_params = {
        key: value for key, value in tree_flatten(model.parameters())
    }

    bound_backbone: list[str] = []
    bound_mtp: list[str] = []
    fp8_decoded: list[str] = []
    direct: list[str] = []
    integers: list[str] = []
    reshaped: list[str] = []
    skipped_experts: list[str] = []
    skipped_mtp_experts: list[str] = []
    skipped_mtp: list[str] = []
    skipped_scope: list[str] = []
    unmatched: list[str] = []

    # Plan first: decide every tensor's fate before opening a single shard.
    plan: list[tuple[str, str, str | None]] = []  # (source, target, scale_source)
    for name in sorted(weight_map):
        if name.endswith(".scale"):
            continue  # consumed with its .weight
        stage = _checkpoint_mtp_stage(name)
        if stage is not None:
            if _is_routed_expert_tensor(name):
                skipped_mtp_experts.append(name)
                continue
            if not include_mtp:
                skipped_mtp.append(name)
                continue
            if model.mtp_drafter is None or stage >= len(model.mtp_drafter.blocks):
                skipped_scope.append(name)
                continue
        else:
            if _is_routed_expert_tensor(name):
                skipped_experts.append(name)
                continue
            layer = _checkpoint_layer_index(name)
            if layer is not None:
                if layer >= model.args.num_hidden_layers:
                    skipped_scope.append(name)
                    continue
                if layer_filter is not None and layer not in layer_filter:
                    skipped_scope.append(name)
                    continue
            elif layer_filter is not None:
                # A layer-scoped bind must not silently pull in the 2 GB
                # embedding or the head.
                skipped_scope.append(name)
                continue

        target = _map_checkpoint_name(name)
        if target is None or target not in target_params:
            unmatched.append(name)
            continue
        scale_name = f"{name.rsplit('.', 1)[0]}.scale"
        plan.append((name, target, scale_name if scale_name in weight_map else None))

    if strict and unmatched:
        raise ValueError(
            f"unexpected {len(unmatched)} source tensors with no model parameter: "
            + ", ".join(unmatched[:20])
        )

    # Group by shard so each file is opened (and its header parsed) once.
    by_shard: dict[str, list[tuple[str, str, str | None]]] = {}
    for source, target, scale_name in plan:
        by_shard.setdefault(weight_map[source], []).append((source, target, scale_name))

    for shard_name in sorted(by_shard):
        shard_path = checkpoint_path / shard_name
        file_header = read_safetensors_file_header(shard_path)
        headers = file_header.tensors
        batch: list[tuple[str, mx.array]] = []
        with shard_path.open("rb") as handle:
            for source, target, scale_name in by_shard[shard_name]:
                tensor = headers[source]
                raw = _read_raw(file_header.header_size, tensor, handle)
                scale_payload = None
                if scale_name is not None:
                    scale_shard = weight_map[scale_name]
                    if scale_shard == shard_name:
                        scale_tensor = headers[scale_name]
                        scale_raw = _read_raw(
                            file_header.header_size, scale_tensor, handle
                        )
                    else:
                        other = checkpoint_path / scale_shard
                        other_header = read_safetensors_file_header(other)
                        scale_tensor = other_header.tensors[scale_name]
                        with other.open("rb") as other_handle:
                            scale_raw = _read_raw(
                                other_header.header_size,
                                scale_tensor,
                                other_handle,
                            )
                    scale_payload = (
                        scale_tensor.dtype,
                        scale_raw,
                        scale_tensor.shape,
                    )

                value, kind = _decode_source_tensor(
                    tensor.dtype,
                    raw,
                    tensor.shape,
                    scale_payload,
                    compute_dtype=compute_dtype,
                )
                expected = target_params[target]
                if tuple(value.shape) != tuple(expected.shape):
                    # A numel-only reshape is a reinterpretation of the tensor's
                    # axis layout, so it is allowed exactly where the layout
                    # difference is known and intended -- ``wo_a``, which ships
                    # flat and lives in MultiLinear's 3-D form. Anywhere else a
                    # shape mismatch means the name mapping is wrong, and
                    # quietly reshaping would scramble the weight instead of
                    # saying so.
                    if not _RESHAPE_ALLOWED_RE.search(source):
                        raise ValueError(
                            f"{source} decodes to shape {tuple(value.shape)} but "
                            f"{target} expects {tuple(expected.shape)}; only "
                            f"{_RESHAPE_ALLOWED_DESCRIPTION} may be reshaped"
                        )
                    if int(np.prod(value.shape)) != int(np.prod(expected.shape)):
                        raise ValueError(
                            f"{source} decodes to shape {tuple(value.shape)} "
                            f"({int(np.prod(value.shape))} elements) but "
                            f"{target} expects {tuple(expected.shape)} "
                            f"({int(np.prod(expected.shape))} elements)"
                        )
                    value = value.reshape(expected.shape)
                    reshaped.append(source)

                batch.append((target, value))
                if kind == "fp8_block":
                    fp8_decoded.append(source)
                elif kind == "integer":
                    integers.append(source)
                else:
                    direct.append(source)
                if target.startswith("mtp_drafter."):
                    bound_mtp.append(target)
                else:
                    bound_backbone.append(target)
        if batch:
            model.load_weights(batch, strict=False)
            # Evaluate only what this shard produced. ``mx.eval(model.parameters())``
            # would also force every *unbound* lazily-initialised parameter,
            # materialising the whole skeleton -- hundreds of GB on the real
            # config -- for a bind that touched one layer.
            mx.eval([value for _, value in batch])

    bound_all = set(bound_backbone) | set(bound_mtp)
    missing = tuple(sorted(set(target_params) - bound_all))
    report = DeepseekV4FlashNonVQBindReport(
        bound_backbone_parameters=tuple(sorted(bound_backbone)),
        bound_mtp_parameters=tuple(sorted(bound_mtp)),
        fp8_block_decoded_tensors=tuple(sorted(fp8_decoded)),
        direct_tensors=tuple(sorted(direct)),
        preserved_integer_tensors=tuple(sorted(integers)),
        reshaped_tensors=tuple(sorted(reshaped)),
        skipped_routed_expert_tensors=tuple(sorted(skipped_experts)),
        skipped_mtp_routed_expert_tensors=tuple(sorted(skipped_mtp_experts)),
        skipped_mtp_tensors=tuple(sorted(skipped_mtp)),
        skipped_out_of_scope_tensors=tuple(sorted(skipped_scope)),
        skipped_unmatched_tensors=tuple(sorted(unmatched)),
        missing_model_parameters=missing,
    )
    if strict and missing:
        raise ValueError(
            f"missing {len(missing)} non-VQ parameters: " + ", ".join(missing[:20])
        )
    return report


# ---------------------------------------------------------------------------
# MTP routed experts (dense FP4-decoded -- increment 1 path)
# ---------------------------------------------------------------------------


def bind_deepseek_v4_flash_mtp_dense_experts(
    model: DeepseekV4FlashVQModel,
    checkpoint_dir: str | Path,
    *,
    stages: Sequence[int] | None = None,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> dict[int, int]:
    """Decode the drafter's FP4 routed experts into dense ``SwitchGLU`` stacks.

    The drafter is not VQ-compressed yet (Wave 5 decides whether it should be),
    so its routed experts are decoded with
    :func:`keep.convert.fp4_expert.dequantize_fp4_expert` and stacked into
    mlx-lm's dense ``SwitchGLU``. ``w1``=gate, ``w2``=down, ``w3``=up.

    Returns ``{stage: expert_count}``. Each stage is ~13 GB in bf16, so callers
    should bind only the stages they need.
    """

    if model.mtp_drafter is None:
        raise ValueError("model was built without an MTP drafter")

    checkpoint_path = Path(checkpoint_dir)
    weight_map = _load_weight_map(checkpoint_path)
    config = model.args
    requested = (
        tuple(range(len(model.mtp_drafter.blocks)))
        if stages is None
        else tuple(int(stage) for stage in stages)
    )

    bound: dict[int, int] = {}
    for stage in requested:
        block = model.mtp_drafter.blocks[stage]
        # ``fp32=True``: the drafter's routed GLU runs in float32
        # (``deepseek_v4_dspark.py:309-313``). ``bind_switch_mlp`` enforces it,
        # but this path assigns ``switch_mlp`` directly, so it must be right
        # here too -- and the assertion below is what keeps the two in step.
        switch = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit, fp32=True),
        )
        if not block.ffn.fp32_swiglu:
            raise RuntimeError(
                "drafter stage is not marked fp32_swiglu; the dense bind and "
                "the block disagree about the routed activation"
            )
        stacked: dict[str, list[mx.array]] = {"w1": [], "w2": [], "w3": []}
        for expert in range(config.n_routed_experts):
            for projection in ("w1", "w2", "w3"):
                base = f"mtp.{stage}.ffn.experts.{expert}.{projection}"
                codes = _read_named_tensor_bytes(
                    checkpoint_path,
                    weight_map,
                    f"{base}.weight",
                    expected_dtype=_FP4_CODE_DTYPE,
                )
                scales = _read_named_tensor_bytes(
                    checkpoint_path,
                    weight_map,
                    f"{base}.scale",
                    expected_dtype=_FP8_SCALE_DTYPE,
                )
                decoded = dequantize_fp4_expert(codes, scales)
                stacked[projection].append(mx.array(decoded).astype(compute_dtype))
        switch.gate_proj.weight = mx.stack(stacked["w1"])
        switch.down_proj.weight = mx.stack(stacked["w2"])
        switch.up_proj.weight = mx.stack(stacked["w3"])
        block.ffn.switch_mlp = switch
        bound[stage] = config.n_routed_experts
    return bound


def _read_named_tensor_bytes(
    checkpoint_dir: Path,
    weight_map: Mapping[str, str],
    name: str,
    *,
    expected_dtype: str,
) -> np.ndarray:
    """Read one tensor's payload as raw uint8 code bytes.

    ``expected_dtype`` is mandatory because this is a *reinterpret*, not a
    conversion: handing FP4 codes to a decoder that was promised FP8 (or vice
    versa) produces a full-size, finite, entirely wrong weight. Checking the
    declared dtype first turns that into a load-time error.
    """

    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"{name!r} is missing from the safetensors index")
    shard_path = checkpoint_dir / shard
    file_header = read_safetensors_file_header(shard_path)
    tensor = file_header.tensors[name]
    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name} is declared {tensor.dtype}, expected {expected_dtype}; "
            "refusing to reinterpret bytes across encodings"
        )
    with shard_path.open("rb") as handle:
        raw = _read_raw(file_header.header_size, tensor, handle)
    return np.frombuffer(raw, dtype=np.uint8).reshape(tensor.shape).copy()


def deepseek_v4_flash_checkpoint_tensor_names(
    checkpoint_dir: str | Path,
) -> tuple[str, ...]:
    """Every tensor name in the checkpoint index (headless, no payload reads)."""

    return tuple(sorted(_load_weight_map(Path(checkpoint_dir))))


def iter_deepseek_v4_flash_layer_sources(
    checkpoint_dir: str | Path,
    layer: int,
) -> Iterable[str]:
    prefix = f"layers.{layer}."
    for name in deepseek_v4_flash_checkpoint_tensor_names(checkpoint_dir):
        if name.startswith(prefix):
            yield name
