from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from mlx_vq.codebook.e8 import e8_1bit_grid, e8p_full_grid
from mlx_vq.quant.rtn import (
    nearest_e8_codes_diagonal_hessian,
    nearest_e8p_codes_diagonal_hessian,
)


DEFAULT_MULTIPLIERS: tuple[float, ...] = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1)


def _validated_multipliers(multipliers: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(multipliers, tuple) or not multipliers:
        raise ValueError("multipliers must be a nonempty tuple")
    values = tuple(float(multiplier) for multiplier in multipliers)
    if not np.isfinite(values).all() or any(multiplier <= 0 for multiplier in values):
        raise ValueError("multipliers must be finite and strictly positive")
    if len(set(values)) != len(values):
        raise ValueError("multipliers must not contain duplicates")
    if 1.0 not in values:
        raise ValueError("multipliers must include 1.0")
    return values


def canonical_multiplier_string(multipliers: tuple[float, ...]) -> str:
    """Serialize an ordered EBSS multiplier policy without locale dependence."""

    values = _validated_multipliers(multipliers)
    return ",".join(repr(multiplier) for multiplier in values)


def search_group_scales(
    weight: np.ndarray,
    diagonal: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int,
    multipliers: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | str]]:
    """Choose max-abs scale multipliers under a diagonal-Hessian objective.

    Candidate encoding is vectorized over every output row and group for each
    multiplier. Equal objectives prefer the multiplier closest to one, then
    the smaller multiplier, independently of caller ordering.

    ``codebook`` must exactly equal the repository's canonical decode table for
    ``code_bits`` (including row ordering): E8 for 8 bits or E8P for 16 bits.
    Noncanonical tables are rejected because the fast encoders emit indices
    into those canonical tables.
    """

    values = np.asarray(weight, dtype=np.float32)
    hessian = np.asarray(diagonal, dtype=np.float32)
    table = np.asarray(codebook, dtype=np.float32)
    candidates = _validated_multipliers(multipliers)
    if values.ndim != 2:
        raise ValueError("weight must have shape [out, in]")
    if not np.isfinite(values).all():
        raise ValueError("weight must be finite")
    if hessian.shape != (values.shape[1],):
        raise ValueError("diagonal must have shape [in]")
    if not np.isfinite(hessian).all() or np.any(hessian < 0):
        raise ValueError("diagonal must be finite and non-negative")
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if group_size <= 0 or group_size % 8 or values.shape[1] % group_size:
        raise ValueError(
            "group_size must be positive, divisible by 8, and divide the input dimension"
        )
    expected_rows = 1 << code_bits
    if table.shape != (expected_rows, 8):
        raise ValueError(f"codebook must have shape ({expected_rows}, 8)")
    if not np.isfinite(table).all():
        raise ValueError("codebook must be finite")
    canonical_table = e8_1bit_grid() if code_bits == 8 else e8p_full_grid()
    if not np.array_equal(table, canonical_table):
        raise ValueError(f"codebook must exactly match the canonical {code_bits}-bit table")
    divisor = np.max(np.abs(table)).astype(np.float32)
    if divisor <= 0:
        raise ValueError("codebook scale divisor must be positive")

    out_dim, in_dim = values.shape
    group_count = in_dim // group_size
    words_per_group = group_size // 8
    grouped = values.reshape(out_dim, group_count, words_per_group, 8)
    grouped_hessian = hessian.reshape(group_count, words_per_group, 8)
    if np.any(np.sum(grouped_hessian, axis=-1) <= 0):
        raise ValueError("each 8-coordinate diagonal block must contain positive weight")

    baseline_scales = np.max(np.abs(grouped), axis=(2, 3)) / divisor
    baseline_scales = np.where(
        baseline_scales > 0, baseline_scales, np.float32(1.0)
    ).astype(np.float32)
    flat_diagonal = np.broadcast_to(
        grouped_hessian[None, ...], grouped.shape
    ).reshape(-1, 8)
    code_dtype = np.dtype(np.uint8 if code_bits == 8 else np.uint16)
    best_error = np.full((out_dim, group_count), np.inf, dtype=np.float64)
    best_scales = np.empty_like(baseline_scales)
    best_codes = np.empty(
        (out_dim, group_count, words_per_group), dtype=code_dtype
    )
    chosen = np.empty((out_dim, group_count), dtype=np.float32)
    baseline_error = np.empty((out_dim, group_count), dtype=np.float64)

    for multiplier in sorted(candidates, key=lambda value: (abs(value - 1.0), value)):
        candidate_scales = baseline_scales * np.float32(multiplier)
        normalized = grouped / candidate_scales[:, :, None, None]
        flat_values = normalized.reshape(-1, 8)
        if code_bits == 8:
            candidate_codes = nearest_e8_codes_diagonal_hessian(
                flat_values,
                flat_diagonal,
                backend="numpy",
                index_dtype=np.uint8,
            )
        else:
            candidate_codes = nearest_e8p_codes_diagonal_hessian(
                flat_values, flat_diagonal, backend="numpy"
            )
        candidate_codes = candidate_codes.reshape(
            out_dim, group_count, words_per_group
        ).astype(code_dtype, copy=False)
        decoded = table[candidate_codes]
        residual = grouped - candidate_scales[:, :, None, None] * decoded
        error = np.sum(
            grouped_hessian[None, ...] * residual * residual,
            axis=(2, 3),
            dtype=np.float64,
        )
        if multiplier == 1.0:
            baseline_error[...] = error
        improved = error < best_error
        best_error[improved] = error[improved]
        best_scales[improved] = candidate_scales[improved]
        best_codes[improved] = candidate_codes[improved]
        chosen[improved] = np.float32(multiplier)

    stats: dict[str, np.ndarray | str] = {
        "chosen_multipliers": chosen,
        "baseline_objective": baseline_error,
        "selected_objective": best_error,
        "multiplier_policy": canonical_multiplier_string(candidates),
    }
    return best_scales.astype(np.float32, copy=False), best_codes.reshape(
        out_dim, in_dim // 8
    ), stats


@dataclass
class _PromptRouteUsage:
    prompt_id: str
    layers: dict[int, list[int]] = field(default_factory=dict)
    route_count: int = 0
    record_count: int = 0


def build_route_record(
    *,
    prompt_id: str,
    layer: int,
    route_indices: Sequence[Sequence[int]],
    num_experts: int,
    router_scores: Sequence[Sequence[float]] | None = None,
    context_tokens: int | None = None,
) -> dict[str, object]:
    if not prompt_id:
        raise ValueError("prompt_id must not be empty")
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if not route_indices:
        raise ValueError("route_indices must not be empty")
    if router_scores is not None and len(router_scores) != len(route_indices):
        raise ValueError("router_scores must have one row per route_indices row")

    counts = [0] * num_experts
    score_sums = [0.0] * num_experts
    top_k: int | None = None
    route_count = 0
    for token_index, row in enumerate(route_indices):
        if top_k is None:
            top_k = len(row)
            if top_k == 0:
                raise ValueError("route_indices rows must not be empty")
        elif len(row) != top_k:
            raise ValueError("route_indices rows must have a consistent top_k")
        score_row = router_scores[token_index] if router_scores is not None else None
        if score_row is not None and len(score_row) != len(row):
            raise ValueError("router_scores rows must match route_indices rows")
        for route_rank, expert_value in enumerate(row):
            expert = int(expert_value)
            if expert < 0 or expert >= num_experts:
                raise ValueError(f"expert {expert} outside [0, {num_experts})")
            counts[expert] += 1
            if score_row is not None:
                score_sums[expert] += float(score_row[route_rank])
            route_count += 1

    record: dict[str, object] = {
        "schema_version": 1,
        "record_type": "air_route_record",
        "prompt_id": prompt_id,
        "layer": int(layer),
        "num_experts": int(num_experts),
        "top_k": int(top_k or 0),
        "route_count": int(route_count),
        "expert_counts": {str(expert): int(count) for expert, count in enumerate(counts)},
    }
    if router_scores is not None:
        record["expert_router_score_sums"] = {
            str(expert): float(score) for expert, score in enumerate(score_sums)
        }
    if context_tokens is not None:
        record["context_tokens"] = int(context_tokens)
    return record


def _parse_expert_counts(record: dict[str, Any]) -> tuple[str, int, int, list[int]]:
    prompt_id = record.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError(f"route record has invalid prompt_id: {prompt_id!r}")
    layer = record.get("layer")
    if not isinstance(layer, int):
        raise ValueError(f"route record {prompt_id!r} has invalid layer: {layer!r}")
    num_experts = record.get("num_experts")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError(f"route record {prompt_id!r} layer {layer} has invalid num_experts: {num_experts!r}")
    expert_counts = record.get("expert_counts")
    if not isinstance(expert_counts, dict):
        raise ValueError(f"route record {prompt_id!r} layer {layer} is missing expert_counts")

    counts = [0] * num_experts
    for expert_text, count_value in expert_counts.items():
        expert = int(expert_text)
        if expert < 0 or expert >= num_experts:
            raise ValueError(
                f"route record {prompt_id!r} layer {layer} has expert {expert} outside num_experts={num_experts}"
            )
        if not isinstance(count_value, int) or count_value < 0:
            raise ValueError(
                f"route record {prompt_id!r} layer {layer} has invalid count for expert {expert}: {count_value!r}"
            )
        counts[expert] += count_value
    return prompt_id, layer, num_experts, counts


def _usage_by_prompt(records: Iterable[dict[str, Any]]) -> dict[str, _PromptRouteUsage]:
    usages: dict[str, _PromptRouteUsage] = {}
    expected_experts_by_layer: dict[int, int] = {}
    for record in records:
        prompt_id, layer, num_experts, counts = _parse_expert_counts(record)
        expected = expected_experts_by_layer.setdefault(layer, num_experts)
        if expected != num_experts:
            raise ValueError(f"layer {layer} mixes num_experts values {expected} and {num_experts}")
        usage = usages.setdefault(prompt_id, _PromptRouteUsage(prompt_id=prompt_id))
        layer_counts = usage.layers.setdefault(layer, [0] * num_experts)
        for expert, count in enumerate(counts):
            layer_counts[expert] += count
        usage.route_count += sum(counts)
        usage.record_count += 1
    if not usages:
        raise ValueError("at least one route record is required")
    return usages


def _combine_layer_counts(usages: Iterable[_PromptRouteUsage]) -> dict[int, list[int]]:
    combined: dict[int, list[int]] = {}
    for usage in usages:
        for layer, counts in usage.layers.items():
            layer_counts = combined.setdefault(layer, [0] * len(counts))
            if len(layer_counts) != len(counts):
                raise ValueError(f"layer {layer} mixes expert-count widths")
            for expert, count in enumerate(counts):
                layer_counts[expert] += count
    return combined


def _layer_imbalance(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    frequencies = [count / total for count in counts]
    uniform = 1.0 / len(frequencies)
    return math.sqrt(sum((frequency - uniform) ** 2 for frequency in frequencies) / len(frequencies))


def _imbalance(layer_counts: dict[int, list[int]]) -> float:
    if not layer_counts:
        return 0.0
    return sum(_layer_imbalance(counts) for counts in layer_counts.values()) / len(layer_counts)


def _layer_summary(counts: list[int]) -> dict[str, object]:
    total = sum(counts)
    frequencies = {
        str(expert): (float(count / total) if total else 0.0)
        for expert, count in enumerate(counts)
    }
    return {
        "route_count": int(total),
        "selected_expert_count": sum(1 for count in counts if count > 0),
        "cold_expert_count": sum(1 for count in counts if count == 0),
        "expert_counts": {str(expert): int(count) for expert, count in enumerate(counts)},
        "expert_frequencies": frequencies,
        "expert_frequency_stddev": _layer_imbalance(counts),
    }


def _layer_summaries(layer_counts: dict[int, list[int]]) -> dict[str, dict[str, object]]:
    return {str(layer): _layer_summary(layer_counts[layer]) for layer in sorted(layer_counts)}


def build_ebss_prompt_selection(
    records: Iterable[dict[str, Any]],
    *,
    max_prompts: int,
) -> dict[str, object]:
    if max_prompts <= 0:
        raise ValueError("max_prompts must be positive")

    usages_by_prompt = _usage_by_prompt(records)
    selected: list[_PromptRouteUsage] = []
    remaining = dict(sorted(usages_by_prompt.items()))
    target_count = min(max_prompts, len(remaining))
    source_prefix = list(usages_by_prompt.values())[:target_count]

    while len(selected) < target_count:
        best_prompt_id: str | None = None
        best_score: tuple[float, int, str] | None = None
        for prompt_id, usage in remaining.items():
            candidate_counts = _combine_layer_counts([*selected, usage])
            score = (_imbalance(candidate_counts), -usage.route_count, prompt_id)
            if best_score is None or score < best_score:
                best_score = score
                best_prompt_id = prompt_id
        if best_prompt_id is None:
            break
        selected.append(remaining.pop(best_prompt_id))

    full_pool_counts = _combine_layer_counts(usages_by_prompt.values())
    source_prefix_counts = _combine_layer_counts(source_prefix)
    selected_counts = _combine_layer_counts(selected)
    full_pool_imbalance = _imbalance(full_pool_counts)
    source_prefix_imbalance = _imbalance(source_prefix_counts)
    selected_imbalance = _imbalance(selected_counts)
    selected_prompt_ids = [usage.prompt_id for usage in selected]

    return {
        "record_type": "air_ebss_rebalanced_prompt_selection",
        "selection_mode": "ebss_rebalanced_existing_pool",
        "source_record_count": sum(usage.record_count for usage in usages_by_prompt.values()),
        "source_prompt_count": len(usages_by_prompt),
        "max_prompts": int(max_prompts),
        "selected_prompt_count": len(selected_prompt_ids),
        "selected_prompt_ids": selected_prompt_ids,
        "full_pool_imbalance": full_pool_imbalance,
        "source_prefix_imbalance": source_prefix_imbalance,
        "selected_imbalance": selected_imbalance,
        "imbalance_reduction": full_pool_imbalance - selected_imbalance,
        "source_prefix_imbalance_reduction": source_prefix_imbalance - selected_imbalance,
        "full_pool_layer_summaries": _layer_summaries(full_pool_counts),
        "source_prefix_layer_summaries": _layer_summaries(source_prefix_counts),
        "selected_layer_summaries": _layer_summaries(selected_counts),
        "notes": [
            "Selection is deterministic and uses existing calibration expert_counts only.",
            "No prompt generation, protected seed mutation, quantization, model execution, or evaluation is performed.",
        ],
    }
