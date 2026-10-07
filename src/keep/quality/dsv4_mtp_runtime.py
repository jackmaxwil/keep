"""Greedy DeepSeek-V4 DSpark generation on KEEP's existing model rails."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import CacheList

from ramp.ops.vq_switch import trace_gather_vqmm_calls
from ramp.models.deepseek_v4_flash_adapter import (
    DeepseekV4FlashRotatingKVCache,
    DeepseekV4FlashVQModel,
    PoolingCache,
    has_unbound_deepseek_v4_flash_mtp_experts,
    has_unbound_deepseek_v4_flash_vq_experts,
)


@dataclass(frozen=True)
class Dsv4AutoregressiveResult:
    token_ids: tuple[int, ...]


@dataclass
class Dsv4SpeculativeStats:
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    verify_passes: int = 0
    emitted_tokens: int = 0
    verify_kernel_calls: int = 0
    verify_kernel_dispatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return (
            self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0
        )

    @property
    def accepted_per_verify_pass(self) -> float:
        return self.accepted_tokens / self.verify_passes if self.verify_passes else 0.0

    @property
    def emitted_per_verify_pass(self) -> float:
        return self.emitted_tokens / self.verify_passes if self.verify_passes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "verify_passes": self.verify_passes,
            "emitted_tokens": self.emitted_tokens,
            "acceptance_rate": self.acceptance_rate,
            "accepted_per_verify_pass": self.accepted_per_verify_pass,
            "emitted_per_verify_pass": self.emitted_per_verify_pass,
            "verify_kernel_calls": self.verify_kernel_calls,
            "verify_kernel_dispatches": self.verify_kernel_dispatches,
        }


@dataclass(frozen=True)
class Dsv4SpeculativeResult:
    token_ids: tuple[int, ...]
    stats: Dsv4SpeculativeStats


def _validate_request(
    model: DeepseekV4FlashVQModel,
    prompt: mx.array,
    max_new_tokens: int,
    *,
    require_mtp: bool,
) -> None:
    if prompt.ndim != 2 or int(prompt.shape[0]) != 1 or int(prompt.shape[1]) < 1:
        raise ValueError("DSV4 generation requires one non-empty [1, L] prompt")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if has_unbound_deepseek_v4_flash_vq_experts(model):
        raise ValueError("all DSV4 backbone VQ experts must be bound")
    if require_mtp and has_unbound_deepseek_v4_flash_mtp_experts(model):
        raise ValueError("all DSV4 MTP VQ experts must be bound")


def _token_tuple(tokens: Sequence[mx.array]) -> tuple[int, ...]:
    if not tokens:
        return ()
    return tuple(int(value) for value in mx.concatenate(tokens).tolist())


def generate_dsv4_autoregressive(
    model: DeepseekV4FlashVQModel,
    prompt: mx.array,
    *,
    max_new_tokens: int,
) -> Dsv4AutoregressiveResult:
    """Generate the greedy compressed-model baseline."""

    _validate_request(model, prompt, max_new_tokens, require_mtp=False)
    cache = model.make_cache()
    logits = model(prompt, cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    mx.eval(token)
    output = [token]
    while len(output) < int(max_new_tokens):
        logits = model(token[:, None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(token)
        output.append(token)
    return Dsv4AutoregressiveResult(token_ids=_token_tuple(output))


def _cache_entries(cache: Any):
    if isinstance(cache, CacheList):
        for entry in cache.caches:
            yield from _cache_entries(entry)
    else:
        yield cache


def _can_trim(entry: Any, n: int) -> bool:
    if isinstance(entry, PoolingCache):
        return n <= int(entry.remainder) or entry._can_undo(n)
    if isinstance(entry, DeepseekV4FlashRotatingKVCache):
        return entry.can_trim(n)
    checker = getattr(entry, "is_trimmable", None)
    return bool(callable(checker) and checker())


def _rollback_target_cache(cache: Sequence[Any], rejected: int) -> None:
    rejected = int(rejected)
    entries = [entry for layer in cache for entry in _cache_entries(layer)]
    if rejected == 0:
        for entry in entries:
            clear = getattr(entry, "clear_undo", None)
            if callable(clear):
                clear()
        return
    refused = [
        type(entry).__name__ for entry in entries if not _can_trim(entry, rejected)
    ]
    if refused:
        raise RuntimeError(
            f"DSV4 verify rollback of {rejected} rows is unsupported by {refused}"
        )
    for entry in entries:
        trimmed = int(entry.trim(rejected))
        if trimmed != rejected:
            raise RuntimeError(
                f"{type(entry).__name__} trimmed {trimmed}, expected {rejected}"
            )


def _draft_tokens(
    model: DeepseekV4FlashVQModel,
    hidden_rows: mx.array,
    committed: mx.array,
    mtp_cache: Sequence[Any],
    depth: int,
) -> mx.array:
    depth = int(depth)
    if depth == 0:
        model.dspark_append_context(hidden_rows, mtp_cache)
        return mx.zeros((0,), dtype=mx.uint32)
    anchor = committed[-1:].reshape(1, 1)
    logits, _ = model.dspark_forward(
        hidden_rows,
        anchor,
        mtp_cache,
        draft_length=depth,
    )
    previous = anchor.reshape(1)
    drafts: list[mx.array] = []
    for index in range(depth):
        bias, _ = model.dspark_markov(previous)
        token = mx.argmax(logits[:, index, :] + bias, axis=-1).astype(mx.uint32)
        drafts.append(token)
        previous = token
    result = mx.concatenate(drafts)
    mx.eval(result)
    return result


def generate_dsv4_speculative(
    model: DeepseekV4FlashVQModel,
    prompt: mx.array,
    *,
    max_new_tokens: int,
    require_verify_kernel_calls: bool = True,
) -> Dsv4SpeculativeResult:
    """Generate greedily with DSpark draft, one wide target verify, and rollback."""

    _validate_request(model, prompt, max_new_tokens, require_mtp=True)
    target_cache = model.make_cache()
    logits, taps = model(prompt, cache=target_cache, return_dspark_hidden=True)
    next_main = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    mx.eval(next_main, taps)

    output = [next_main]
    stats = Dsv4SpeculativeStats(emitted_tokens=1)
    if max_new_tokens == 1:
        return Dsv4SpeculativeResult(_token_tuple(output), stats)

    mtp_cache = model.make_mtp_cache()
    remaining = int(max_new_tokens) - 1
    depth = min(int(model.args.dspark_block_size), max(0, remaining - 1))
    drafts = _draft_tokens(model, taps[:, -1:], next_main, mtp_cache, depth)

    while len(output) < int(max_new_tokens):
        k = int(drafts.shape[0])
        verify_ids = mx.concatenate([next_main, drafts])[None, :]
        with trace_gather_vqmm_calls() as dispatches:
            verify_logits, verify_taps = model(
                verify_ids,
                cache=target_cache,
                return_dspark_hidden=True,
            )
            mx.eval(verify_logits, verify_taps)

        wide_dispatches = [
            record for record in dispatches if int(record["token_rows"]) > 1
        ]
        if k > 0 and require_verify_kernel_calls and not wide_dispatches:
            raise RuntimeError(
                "DSV4 target verification made zero wide-M VQ kernel calls"
            )
        stats.verify_kernel_dispatches.extend(wide_dispatches)
        stats.verify_kernel_calls += len(wide_dispatches)

        targets = mx.argmax(verify_logits[0], axis=-1).astype(mx.uint32)
        if k:
            matches = (targets[:k] == drafts).astype(mx.int32)
            accepted = int(mx.sum(mx.cumprod(matches)).item())
        else:
            accepted = 0
        final = targets[accepted : accepted + 1]
        committed = mx.concatenate([drafts[:accepted], final])
        mx.eval(committed)

        rejected = k - accepted
        _rollback_target_cache(target_cache, rejected)
        stats.drafted_tokens += k
        stats.accepted_tokens += accepted
        stats.rejected_tokens += rejected
        stats.verify_passes += 1

        for token in [*list(drafts[:accepted]), final]:
            output.append(token.reshape(1))
        stats.emitted_tokens = len(output)
        next_main = final
        if len(output) >= int(max_new_tokens):
            break

        remaining = int(max_new_tokens) - len(output)
        next_depth = min(int(model.args.dspark_block_size), max(0, remaining - 1))
        drafts = _draft_tokens(
            model,
            verify_taps[:, : accepted + 1],
            committed,
            mtp_cache,
            next_depth,
        )

    if require_verify_kernel_calls and stats.verify_kernel_calls == 0:
        raise RuntimeError("DSV4 speculative run made zero wide-M VQ kernel calls")
    return Dsv4SpeculativeResult(_token_tuple(output), stats)


__all__ = [
    "Dsv4AutoregressiveResult",
    "Dsv4SpeculativeResult",
    "Dsv4SpeculativeStats",
    "generate_dsv4_autoregressive",
    "generate_dsv4_speculative",
]
