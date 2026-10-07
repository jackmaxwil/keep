from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask

from ramp.benchmark.glm45_air import append_jsonl, load_resident_air
from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    link_seed_artifact_groups,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from keep.io.logit_bias import copy_declared_logit_bias_sidecar
from keep.io.source_safetensors import read_safetensors_tensor_mlx
from ramp.models.glm45_air_vq_adapter import GLM45AirVQMoE
from keep.quality.mlx_surrogate import (
    RouteLocalSwitchLinearSurrogate,
    switch_linear_layer_sidecar,
)
from keep.quality.teacher_cache import (
    read_teacher_cache_rows,
    validate_teacher_cache_metadata,
)

SUPPORTED_TRAINABLE_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class PreparedTeacherRow:
    row_index: int
    prompt_id: str
    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    teacher_logits: mx.array


@dataclass(frozen=True)
class FinalLayerPrefixCache:
    row: PreparedTeacherRow
    hidden: mx.array
    attention_mask: mx.array | None
    teacher_log_probs: mx.array
    teacher_probs: mx.array


def _parse_row_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("row indices must be non-negative")
        indices.append(index)
    if not indices:
        raise ValueError("row index list must not be empty")
    return tuple(indices)


def _parse_position_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("position indices must be non-negative")
        indices.append(index)
    if not indices:
        raise ValueError("position index list must not be empty")
    return tuple(indices)


def _parse_token_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    token_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        token_id = int(item)
        if token_id < 0:
            raise ValueError("token ids must be non-negative")
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError("token id list must not be empty")
    return tuple(token_ids)


def _parse_competitor_token_ids(value: str | None) -> tuple[tuple[int, ...], ...] | None:
    if value is None or not value.strip():
        return None
    groups: list[tuple[int, ...]] = []
    for group_text in value.split(","):
        group_text = group_text.strip()
        if not group_text:
            continue
        group: list[int] = []
        for item in group_text.split("+"):
            item = item.strip()
            if not item:
                continue
            token_id = int(item)
            if token_id < 0:
                raise ValueError("competitor token ids must be non-negative")
            group.append(token_id)
        if not group:
            raise ValueError("competitor token id group must not be empty")
        groups.append(tuple(group))
    if not groups:
        raise ValueError("competitor token id list must not be empty")
    return tuple(groups)


def _normalize_competitor_token_groups(
    competitor_token_ids: tuple[int, ...] | tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if competitor_token_ids is None:
        return None
    if not competitor_token_ids:
        raise ValueError("competitor token id list must not be empty")
    first = competitor_token_ids[0]
    if isinstance(first, tuple):
        groups = competitor_token_ids  # type: ignore[assignment]
    else:
        groups = tuple((int(token_id),) for token_id in competitor_token_ids)  # type: ignore[arg-type]
    if not groups:
        raise ValueError("competitor token id list must not be empty")
    for group in groups:
        if not group:
            raise ValueError("competitor token id group must not be empty")
    return groups


def _serialize_competitor_token_groups(
    competitor_token_ids: tuple[tuple[int, ...], ...] | None,
) -> list[int] | list[list[int]] | None:
    if competitor_token_ids is None:
        return None
    if all(len(group) == 1 for group in competitor_token_ids):
        return [int(group[0]) for group in competitor_token_ids]
    return [[int(token_id) for token_id in group] for group in competitor_token_ids]


def _select_rows(
    rows: list[dict],
    *,
    max_rows: int | None,
    row_indices: tuple[int, ...] | None,
) -> list[tuple[int, dict]]:
    if row_indices is not None:
        selected = []
        for row_index in row_indices:
            if row_index >= len(rows):
                raise IndexError(f"row index {row_index} is out of range for {len(rows)} rows")
            selected.append((row_index, rows[row_index]))
        return selected
    selected_rows = rows[:max_rows] if max_rows is not None else rows
    return list(enumerate(selected_rows))


def _interleave_rows(
    first: list[PreparedTeacherRow],
    second: list[PreparedTeacherRow],
) -> list[PreparedTeacherRow]:
    interleaved: list[PreparedTeacherRow] = []
    max_len = max(len(first), len(second))
    for idx in range(max_len):
        if idx < len(first):
            interleaved.append(first[idx])
        if idx < len(second):
            interleaved.append(second[idx])
    return interleaved


def _resolve_cache_shard(cache_root: Path, row: dict) -> Path:
    shard = row.get("logit_shard")
    if not isinstance(shard, str) or not shard:
        raise ValueError(f"teacher row {row.get('prompt_id')} is missing logit_shard")
    shard_path = Path(shard)
    if shard_path.is_absolute():
        raise ValueError(f"logit_shard must be cache-relative, got {shard!r}")
    root = cache_root.resolve(strict=False)
    candidate = (root / shard_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"logit_shard escapes cache root: {shard!r}") from error
    return candidate


def _prepare_teacher_rows(
    *,
    teacher_jsonl: Path,
    cache_root: Path,
    max_rows: int | None,
    max_positions: int | None,
    row_indices: tuple[int, ...] | None = None,
) -> list[PreparedTeacherRow]:
    rows = read_teacher_cache_rows(teacher_jsonl)
    prepared: list[PreparedTeacherRow] = []
    for row_index, row in _select_rows(rows, max_rows=max_rows, row_indices=row_indices):
        if row.get("full_logits_available") is not True:
            raise ValueError(f"teacher row {row_index} does not have full logits")
        input_ids = row.get("input_token_ids")
        positions = row.get("positions")
        target_ids = row.get("target_token_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError(f"teacher row {row_index} must contain input_token_ids")
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"teacher row {row_index} must contain positions")
        if not isinstance(target_ids, list) or not target_ids:
            raise ValueError(f"teacher row {row_index} must contain target_token_ids")
        if len(positions) != len(target_ids):
            raise ValueError(f"teacher row {row_index} positions/targets length mismatch")
        position_count = len(positions)
        if max_positions is not None:
            position_count = min(position_count, max_positions)
        if position_count <= 0:
            raise ValueError(f"teacher row {row_index} has no selected positions")
        tensor_name = str(row.get("logit_tensor", "logits"))
        logits = read_safetensors_tensor_mlx(_resolve_cache_shard(cache_root, row), tensor_name)
        logits = logits[:position_count].astype(mx.float32)
        prepared.append(
            PreparedTeacherRow(
                row_index=row_index,
                prompt_id=str(row.get("prompt_id", row_index)),
                input_token_ids=tuple(int(value) for value in input_ids),
                positions=tuple(int(value) for value in positions[:position_count]),
                target_token_ids=tuple(int(value) for value in target_ids[:position_count]),
                teacher_logits=logits,
            )
        )
    if not prepared:
        raise ValueError("no teacher rows selected")
    return prepared


def _require_clean_full_logit_cache(
    *,
    teacher_jsonl: Path,
    cache_root: Path,
    min_top_k: int,
    allow_dirty_cache: bool,
) -> dict:
    validation = validate_teacher_cache_metadata(
        teacher_jsonl,
        cache_root=cache_root,
        min_top_k=min_top_k,
        check_values=False,
    )
    if not validation["ok"]:
        raise ValueError(f"teacher cache validation failed for {teacher_jsonl}")
    if validation["full_logits_row_count"] != validation["row_count"]:
        raise ValueError(f"Rung 2 requires full-logit teacher rows: {teacher_jsonl}")
    if not allow_dirty_cache and not validation["all_memory_clean"]:
        raise ValueError(f"teacher cache must be memory-clean: {teacher_jsonl}")
    return validation


def _target_projection(model, *, layer: int, projection: str):
    layer_module = model.model.layers[layer]
    if not isinstance(layer_module.mlp, GLM45AirVQMoE):
        raise ValueError(f"layer {layer} is not a sparse GLM-4.5-Air MoE layer")
    if layer_module.mlp.switch_mlp is None:
        raise RuntimeError(f"layer {layer} VQ switch_mlp is not bound")
    if projection not in SUPPORTED_TRAINABLE_PROJECTIONS:
        raise ValueError("projection must be gate_proj, up_proj, or down_proj")
    return getattr(layer_module.mlp.switch_mlp, projection)


def _resolve_trainable_projections(
    projection: str,
    projections: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = tuple(projections or (projection,))
    if not selected:
        raise ValueError("at least one projection must be selected")
    invalid = [item for item in selected if item not in SUPPORTED_TRAINABLE_PROJECTIONS]
    if invalid:
        raise ValueError(f"unsupported projection(s): {invalid}")
    duplicates = sorted({item for item in selected if selected.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate projection(s): {duplicates}")
    return selected


def _selected_logits(model, row: PreparedTeacherRow) -> mx.array:
    tokens = mx.array([list(row.input_token_ids)], dtype=mx.int32)
    logits = model(tokens)
    return logits[0, mx.array(row.positions, dtype=mx.int32), :].astype(mx.float32)


def _call_switch_mlp_with_route_local_surrogates(
    switch_mlp,
    x: mx.array,
    indices: mx.array,
    *,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    gate = _call_projection_with_optional_route_local_surrogate(
        switch_mlp.gate_proj,
        "gate_proj",
        x,
        indices,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )
    up = _call_projection_with_optional_route_local_surrogate(
        switch_mlp.up_proj,
        "up_proj",
        x,
        indices,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )
    hidden = nn.silu(gate) * up
    return _call_projection_with_optional_route_local_surrogate(
        switch_mlp.down_proj,
        "down_proj",
        hidden,
        indices,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )


def _call_projection_with_optional_route_local_surrogate(
    projection,
    projection_name: str,
    x: mx.array,
    indices: mx.array,
    *,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    if projection_name not in surrogate_projections:
        return projection(x, indices)
    sidecar = switch_linear_layer_sidecar(projection)
    return RouteLocalSwitchLinearSurrogate.from_layer(
        projection,
        sidecar=sidecar,
        output_chunk_size=output_chunk_size,
    )(x, indices)


def _splice_selected_sequence_positions(h: mx.array, positions: tuple[int, ...], selected_h: mx.array) -> mx.array:
    if h.ndim != 3 or h.shape[0] != 1:
        raise ValueError(f"expected hidden state shape [1, seq, hidden], got {h.shape}")
    if selected_h.ndim != 2:
        raise ValueError(f"expected selected hidden shape [positions, hidden], got {selected_h.shape}")
    if len(positions) != int(selected_h.shape[0]):
        raise ValueError("positions length must match selected hidden row count")
    seq_len = int(h.shape[1])
    hidden = int(h.shape[2])
    for position in positions:
        if position < 0 or position >= seq_len:
            raise ValueError(f"position {position} is out of range for sequence length {seq_len}")
    position_ids = mx.array(list(positions), dtype=mx.int32)
    seq_ids = mx.arange(seq_len, dtype=mx.int32)
    mask = position_ids[:, None] == seq_ids[None, :]
    replacement = mx.sum(
        mx.where(mask[:, :, None], selected_h[:, None, :], mx.zeros((1, 1, hidden), dtype=selected_h.dtype)),
        axis=0,
    )
    active = mx.sum(mask.astype(mx.int32), axis=0) > 0
    spliced = mx.where(active[:, None], replacement, h[0])
    return spliced[None, :, :]


def _call_vq_moe_with_route_local_surrogates(
    moe,
    x: mx.array,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    if not isinstance(moe, GLM45AirVQMoE):
        raise ValueError(f"layer {layer} is not a sparse GLM-4.5-Air MoE layer")
    if moe.switch_mlp is None:
        raise RuntimeError(f"layer {layer} VQ switch_mlp is not bound")
    inds, scores = moe.route(x)
    y = _call_switch_mlp_with_route_local_surrogates(
        moe.switch_mlp,
        x,
        inds,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    shared_experts = moe.get("shared_experts")
    if shared_experts is not None:
        y = y + shared_experts(x)
    return y


def _call_decoder_layer_with_route_local_surrogates(
    layer_module,
    h: mx.array,
    mask: mx.array,
    layer_cache,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, layer_cache)
    h = h + attention
    mlp_input = layer_module.post_attention_layernorm(h)
    if isinstance(layer_module.mlp, GLM45AirVQMoE):
        y = _call_vq_moe_with_route_local_surrogates(
            layer_module.mlp,
            mlp_input,
            layer=layer,
            surrogate_projections=surrogate_projections,
            output_chunk_size=output_chunk_size,
        )
    else:
        y = layer_module.mlp(mlp_input)
    return h + y


def _selected_layer_moe_output(
    layer_module,
    selected_h: mx.array,
    selected_moe_input: mx.array,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    if isinstance(layer_module.mlp, GLM45AirVQMoE):
        y = _call_vq_moe_with_route_local_surrogates(
            layer_module.mlp,
            selected_moe_input,
            layer=layer,
            surrogate_projections=surrogate_projections,
            output_chunk_size=output_chunk_size,
        )
    else:
        if surrogate_projections:
            raise ValueError("surrogate projections require a sparse GLM-4.5-Air MoE layer")
        y = layer_module.mlp(selected_moe_input)
    return selected_h + y


def _selected_logits_selected_layer(
    model,
    row: PreparedTeacherRow,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    if layer < 0 or layer >= len(model.model.layers):
        raise ValueError(f"--layer must be between 0 and {len(model.model.layers) - 1}, got {layer}")
    tokens = mx.array([list(row.input_token_ids)], dtype=mx.int32)
    h = model.model.embed_tokens(tokens)
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    for layer_idx, layer_module in enumerate(model.model.layers[:layer]):
        h = layer_module(h, mask, cache[layer_idx])
    h = mx.stop_gradient(h)

    layer_module = model.model.layers[layer]
    attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer])
    h = mx.stop_gradient(h + attention)
    moe_input = mx.stop_gradient(layer_module.post_attention_layernorm(h))
    positions = mx.array(row.positions, dtype=mx.int32)
    selected_h = mx.take(h[0], positions, axis=0)
    selected_moe_input = mx.take(moe_input[0], positions, axis=0)
    selected_out = _selected_layer_moe_output(
        layer_module,
        selected_h,
        selected_moe_input,
        layer=layer,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )
    h = _splice_selected_sequence_positions(h, row.positions, selected_out)
    for layer_idx, layer_module in enumerate(model.model.layers[layer + 1 :], start=layer + 1):
        h = _call_decoder_layer_with_route_local_surrogates(
            layer_module,
            h,
            mask,
            cache[layer_idx],
            layer=layer_idx,
            surrogate_projections=surrogate_projections,
            output_chunk_size=output_chunk_size,
        )
    out = model.model.norm(h)
    selected_out = mx.take(out[0], positions, axis=0)
    return model.lm_head(selected_out).astype(mx.float32)


def _selected_logits_final_layer(
    model,
    row: PreparedTeacherRow,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    if layer != len(model.model.layers) - 1:
        raise ValueError("--loss-scope final_layer_selected requires --layer to be the final decoder layer")
    tokens = mx.array([list(row.input_token_ids)], dtype=mx.int32)
    h = model.model.embed_tokens(tokens)
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    for layer_idx, layer_module in enumerate(model.model.layers[:layer]):
        h = layer_module(h, mask, cache[layer_idx])
    h = mx.stop_gradient(h)

    layer_module = model.model.layers[layer]
    attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer])
    h = mx.stop_gradient(h + attention)
    moe_input = mx.stop_gradient(layer_module.post_attention_layernorm(h))
    positions = mx.array(row.positions, dtype=mx.int32)
    selected_h = mx.take(h[0], positions, axis=0)
    selected_moe_input = mx.take(moe_input[0], positions, axis=0)

    out = model.model.norm(
        _selected_layer_moe_output(
            layer_module,
            selected_h,
            selected_moe_input,
            layer=layer,
            surrogate_projections=surrogate_projections,
            output_chunk_size=output_chunk_size,
        )
    )
    return model.lm_head(out).astype(mx.float32)


def _teacher_distribution(row: PreparedTeacherRow) -> tuple[mx.array, mx.array]:
    teacher_logits = row.teacher_logits.astype(mx.float32)
    teacher_log_probs = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)
    return teacher_log_probs, teacher_probs


def _cache_final_layer_training_row(
    model,
    row: PreparedTeacherRow,
    *,
    layer: int,
) -> FinalLayerPrefixCache:
    if layer != len(model.model.layers) - 1:
        raise ValueError("--cache-training-prefix requires --layer to be the final decoder layer")
    tokens = mx.array([list(row.input_token_ids)], dtype=mx.int32)
    h = model.model.embed_tokens(tokens)
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])
    for layer_idx, layer_module in enumerate(model.model.layers[:layer]):
        h = layer_module(h, mask, cache[layer_idx])
    h = mx.stop_gradient(h)
    teacher_log_probs, teacher_probs = _teacher_distribution(row)
    eval_args = [h, teacher_log_probs, teacher_probs]
    if mask is not None:
        eval_args.append(mask)
    mx.eval(*eval_args)
    return FinalLayerPrefixCache(
        row=row,
        hidden=h,
        attention_mask=mask,
        teacher_log_probs=teacher_log_probs,
        teacher_probs=teacher_probs,
    )


def _selected_logits_final_layer_from_prefix(
    model,
    prefix_cache: FinalLayerPrefixCache,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    if layer != len(model.model.layers) - 1:
        raise ValueError("--loss-scope final_layer_selected requires --layer to be the final decoder layer")
    h = prefix_cache.hidden
    layer_module = model.model.layers[layer]
    attention = layer_module.self_attn(layer_module.input_layernorm(h), prefix_cache.attention_mask, None)
    h = mx.stop_gradient(h + attention)
    moe_input = mx.stop_gradient(layer_module.post_attention_layernorm(h))
    positions = mx.array(prefix_cache.row.positions, dtype=mx.int32)
    selected_h = mx.take(h[0], positions, axis=0)
    selected_moe_input = mx.take(moe_input[0], positions, axis=0)

    out = model.model.norm(
        _selected_layer_moe_output(
            layer_module,
            selected_h,
            selected_moe_input,
            layer=layer,
            surrogate_projections=surrogate_projections,
            output_chunk_size=output_chunk_size,
        )
    )
    return model.lm_head(out).astype(mx.float32)


def _kl_loss(
    model,
    row: PreparedTeacherRow,
    *,
    target_nll_weight: float = 0.0,
    teacher_top1_margin_weight: float = 0.0,
    teacher_top1_margin: float = 0.0,
    teacher_top1_competitor_token_ids: tuple[tuple[int, ...], ...] | None = None,
    teacher_top1_include_hardest_competitor: bool = False,
    tail_kld_weight: float = 0.0,
    aux_loss_position_indices: tuple[int, ...] | None = None,
    loss_scope: Literal["full_model", "final_layer_selected", "selected_layer"] = "full_model",
    layer: int = 45,
    surrogate_projections: frozenset[str] = frozenset(),
    surrogate_output_chunk_size: int = 256,
    final_layer_prefix_cache: FinalLayerPrefixCache | None = None,
    teacher_log_probs: mx.array | None = None,
    teacher_probs: mx.array | None = None,
) -> mx.array:
    teacher_logits = row.teacher_logits.astype(mx.float32)
    if teacher_log_probs is None or teacher_probs is None:
        teacher_log_probs, teacher_probs = _teacher_distribution(row)
    if loss_scope == "full_model":
        if final_layer_prefix_cache is not None:
            raise ValueError("prefix cache is only supported for final_layer_selected loss")
        vq_logits = _selected_logits(model, row)
    elif loss_scope == "final_layer_selected":
        if final_layer_prefix_cache is None:
            vq_logits = _selected_logits_final_layer(
                model,
                row,
                layer=layer,
                surrogate_projections=surrogate_projections,
                output_chunk_size=surrogate_output_chunk_size,
            )
        else:
            if final_layer_prefix_cache.row is not row:
                raise ValueError("prefix cache row does not match loss row")
            vq_logits = _selected_logits_final_layer_from_prefix(
                model,
                final_layer_prefix_cache,
                layer=layer,
                surrogate_projections=surrogate_projections,
                output_chunk_size=surrogate_output_chunk_size,
            )
    elif loss_scope == "selected_layer":
        if final_layer_prefix_cache is not None:
            raise ValueError("prefix cache is only supported for final_layer_selected loss")
        vq_logits = _selected_logits_selected_layer(
            model,
            row,
            layer=layer,
            surrogate_projections=surrogate_projections,
            output_chunk_size=surrogate_output_chunk_size,
        )
    else:
        raise ValueError(f"unsupported loss_scope {loss_scope!r}")
    vq_log_probs = vq_logits - mx.logsumexp(vq_logits, axis=-1, keepdims=True)
    loss = mx.mean(mx.sum(teacher_probs * (teacher_log_probs - vq_log_probs), axis=-1))
    if target_nll_weight > 0.0:
        targets = mx.array(row.target_token_ids, dtype=mx.int32)
        target_log_probs = mx.take_along_axis(vq_log_probs, targets[:, None], axis=-1).reshape((-1,))
        loss = loss + float(target_nll_weight) * (-mx.mean(target_log_probs))
    aux_vq_logits, aux_teacher_logits = _filter_logits_by_position_indices(
        vq_logits,
        teacher_logits,
        aux_loss_position_indices,
    )
    if teacher_top1_margin_weight > 0.0:
        loss = loss + float(teacher_top1_margin_weight) * _teacher_top1_margin_loss(
            aux_vq_logits,
            aux_teacher_logits,
            margin=teacher_top1_margin,
            competitor_token_ids=teacher_top1_competitor_token_ids,
            include_hardest_competitor=teacher_top1_include_hardest_competitor,
        )
    if tail_kld_weight > 0.0:
        aux_vq_log_probs = aux_vq_logits - mx.logsumexp(aux_vq_logits, axis=-1, keepdims=True)
        aux_teacher_log_probs, aux_teacher_probs = _filter_logits_by_position_indices(
            teacher_log_probs,
            teacher_probs,
            aux_loss_position_indices,
        )
        loss = loss + float(tail_kld_weight) * _max_token_kld_loss(
            aux_vq_log_probs,
            aux_teacher_log_probs,
            aux_teacher_probs,
        )
    return loss


def _max_token_kld_loss(vq_log_probs: mx.array, teacher_log_probs: mx.array, teacher_probs: mx.array) -> mx.array:
    if vq_log_probs.shape != teacher_log_probs.shape or teacher_probs.shape != teacher_log_probs.shape:
        raise ValueError("vq_log_probs, teacher_log_probs, and teacher_probs must have matching shapes")
    if vq_log_probs.ndim != 2:
        raise ValueError(f"expected log-probs shape [positions, vocab], got {vq_log_probs.shape}")
    token_klds = mx.sum(teacher_probs * (teacher_log_probs - vq_log_probs), axis=-1)
    return mx.max(token_klds)


def _filter_logits_by_position_indices(
    vq_logits: mx.array,
    teacher_logits: mx.array,
    position_indices: tuple[int, ...] | None,
) -> tuple[mx.array, mx.array]:
    if position_indices is None:
        return vq_logits, teacher_logits
    if vq_logits.shape != teacher_logits.shape:
        raise ValueError(f"vq_logits shape {vq_logits.shape} must match teacher_logits shape {teacher_logits.shape}")
    if vq_logits.ndim != 2:
        raise ValueError(f"expected logits shape [positions, vocab], got {vq_logits.shape}")
    position_count = int(vq_logits.shape[0])
    for index in position_indices:
        if index >= position_count:
            raise ValueError(f"position index {index} is out of range for {position_count} selected positions")
    positions = mx.array(list(position_indices), dtype=mx.int32)
    return mx.take(vq_logits, positions, axis=0), mx.take(teacher_logits, positions, axis=0)


def _teacher_top1_margin_loss(
    vq_logits: mx.array,
    teacher_logits: mx.array,
    *,
    margin: float,
    competitor_token_ids: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
    include_hardest_competitor: bool = False,
) -> mx.array:
    if vq_logits.shape != teacher_logits.shape:
        raise ValueError(f"vq_logits shape {vq_logits.shape} must match teacher_logits shape {teacher_logits.shape}")
    if vq_logits.ndim != 2:
        raise ValueError(f"expected logits shape [positions, vocab], got {vq_logits.shape}")
    teacher_top1 = mx.argmax(teacher_logits.astype(mx.float32), axis=-1).astype(mx.int32)
    teacher_vq_logits = mx.take_along_axis(vq_logits.astype(mx.float32), teacher_top1[:, None], axis=-1).reshape((-1,))
    loss_terms = []
    if competitor_token_ids is None or include_hardest_competitor:
        vocab = vq_logits.shape[-1]
        vocab_ids = mx.arange(vocab, dtype=mx.int32)[None, :]
        is_teacher_top1 = vocab_ids == teacher_top1[:, None]
        competitor_logits = mx.where(is_teacher_top1, mx.array(-1.0e30, dtype=mx.float32), vq_logits.astype(mx.float32))
        competitor_vq_logits = mx.max(competitor_logits, axis=-1)
        violation = competitor_vq_logits - teacher_vq_logits + float(margin)
        loss_terms.append(mx.mean(mx.maximum(violation, mx.array(0.0, dtype=mx.float32))))
    competitor_groups = _normalize_competitor_token_groups(competitor_token_ids)
    if competitor_groups is not None:
        position_count = int(vq_logits.shape[0])
        vocab = int(vq_logits.shape[-1])
        if len(competitor_groups) != position_count:
            raise ValueError(
                "competitor token id group count must match the number of positions after auxiliary filtering"
            )
        explicit_violations = []
        vq_float = vq_logits.astype(mx.float32)
        for position_index, group in enumerate(competitor_groups):
            for token_id in group:
                if token_id < 0 or token_id >= vocab:
                    raise ValueError(f"competitor token id {token_id} is out of range for vocab size {vocab}")
                violation = vq_float[position_index, int(token_id)] - teacher_vq_logits[position_index] + float(margin)
                explicit_violations.append(mx.maximum(violation, mx.array(0.0, dtype=mx.float32)))
        loss_terms.append(mx.sum(mx.stack(explicit_violations)))
    loss = loss_terms[0]
    for term in loss_terms[1:]:
        loss = loss + term
    return loss


def _grad_norm(grad: mx.array) -> mx.array:
    return mx.sqrt(mx.sum(grad.astype(mx.float32) * grad.astype(mx.float32)))


def _initial_trainable_param(
    projection,
    *,
    trainable: str,
    low_rank: int,
    low_rank_init_scale: float,
    projection_name: str,
    reinitialize_existing_sidecar: bool = False,
) -> tuple[dict[str, mx.array], dict[str, bool]]:
    if trainable == "scale_delta":
        existing = projection.get("continuous_scale_delta")
        if existing is not None and not reinitialize_existing_sidecar:
            return {"scale_delta": existing.astype(mx.float32)}, {"scale_delta": True}
        return {"scale_delta": mx.zeros(projection.scales.shape, dtype=mx.float32)}, {"scale_delta": False}
    if trainable == "output_bias":
        existing = projection.get("continuous_output_bias")
        if existing is not None and not reinitialize_existing_sidecar:
            return {"output_bias": existing.astype(mx.float32)}, {"output_bias": True}
        return {
            "output_bias": mx.zeros((projection.num_experts, projection.output_dims), dtype=mx.float32)
        }, {"output_bias": False}
    if trainable != "low_rank_residual":
        raise ValueError(f"unsupported trainable {trainable!r}")
    if low_rank <= 0:
        raise ValueError("low_rank must be positive for low_rank_residual")
    existing_left = projection.get("continuous_low_rank_left")
    existing_right = projection.get("continuous_low_rank_right")
    if existing_left is not None and existing_right is not None and not reinitialize_existing_sidecar:
        return {
            "low_rank_left": existing_left.astype(mx.float32),
            "low_rank_right": existing_right.astype(mx.float32),
        }, {"low_rank_left": True, "low_rank_right": True}
    seed = 20260701 + sum(ord(ch) for ch in projection_name)
    rng = np.random.default_rng(seed)
    right = rng.normal(
        scale=float(low_rank_init_scale),
        size=(projection.num_experts, low_rank, projection.input_dims),
    ).astype(np.float32)
    return {
        "low_rank_left": mx.zeros((projection.num_experts, projection.output_dims, low_rank), dtype=mx.float32),
        "low_rank_right": mx.array(right),
    }, {"low_rank_left": False, "low_rank_right": False}


def _initial_trainable_params(
    projections: dict[str, object],
    *,
    trainable: str,
    low_rank: int,
    low_rank_init_scale: float,
    reinitialize_existing_sidecars: bool = False,
) -> tuple[dict[str, mx.array], dict[str, bool]]:
    params: dict[str, mx.array] = {}
    initialized: dict[str, bool] = {}
    for projection_name, projection in projections.items():
        projection_params, projection_initialized = _initial_trainable_param(
            projection,
            trainable=trainable,
            low_rank=low_rank,
            low_rank_init_scale=low_rank_init_scale,
            projection_name=projection_name,
            reinitialize_existing_sidecar=reinitialize_existing_sidecars,
        )
        for param_name, param in projection_params.items():
            key = f"{projection_name}.{param_name}" if trainable == "low_rank_residual" else projection_name
            params[key] = param
            initialized[key] = projection_initialized[param_name]
    return params, initialized


def _assign_trainable_params(
    projections: dict[str, object],
    params: dict[str, mx.array],
    *,
    trainable: str,
) -> None:
    if trainable == "low_rank_residual":
        for projection_name, projection in projections.items():
            left = params[f"{projection_name}.low_rank_left"]
            right = params[f"{projection_name}.low_rank_right"]
            projection.set_continuous_sidecar(
                scale_delta=projection.get("continuous_scale_delta"),
                output_bias=projection.get("continuous_output_bias"),
                low_rank_left=left,
                low_rank_right=right,
            )
        return
    for projection_name, value in params.items():
        projection = projections[projection_name]
        if trainable == "scale_delta":
            projection.set_continuous_sidecar(scale_delta=value, output_bias=None)
        else:
            projection.set_continuous_sidecar(scale_delta=None, output_bias=value)


def _apply_update(
    param: mx.array,
    grad: mx.array,
    *,
    learning_rate: float,
    grad_clip_norm: float | None,
) -> tuple[mx.array, mx.array]:
    norm = _grad_norm(grad)
    clipped_grad = grad
    if grad_clip_norm is not None and grad_clip_norm > 0:
        scale = mx.minimum(mx.array(1.0, dtype=mx.float32), mx.array(grad_clip_norm, dtype=mx.float32) / (norm + 1e-12))
        clipped_grad = grad * scale
    updated = param - float(learning_rate) * clipped_grad
    return updated, norm


def _apply_updates(
    params: dict[str, mx.array],
    grads: dict[str, mx.array],
    *,
    learning_rate: float,
    grad_clip_norm: float | None,
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    updated: dict[str, mx.array] = {}
    norms: dict[str, mx.array] = {}
    for projection_name, param in params.items():
        updated_param, norm = _apply_update(
            param,
            grads[projection_name],
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
        )
        updated[projection_name] = updated_param
        norms[projection_name] = norm
    return updated, norms


def _maybe_clear_cache() -> None:
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rung 2 MLX-only continuous sidecar finetune for GLM-4.5-Air VQ."
    )
    parser.add_argument("--selection-teacher-jsonl", required=True)
    parser.add_argument("--selection-cache-root", required=True)
    parser.add_argument("--validation-teacher-jsonl", required=True)
    parser.add_argument("--validation-cache-root", required=True)
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument(
        "--prefill-engine",
        choices=["auto", "vq_metal", "nax_e8", "nax_e8p"],
        default="auto",
        help="Resident prefill engine to use during KL training.",
    )
    parser.add_argument(
        "--disable-gather-switch",
        action="store_true",
        help="Use the scalar VQ path for training probes when native gather kernels lack VJP support.",
    )
    parser.add_argument("--layer", type=int, default=41)
    parser.add_argument(
        "--projection",
        choices=["gate_proj", "up_proj", "down_proj"],
        default="down_proj",
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=["gate_proj", "up_proj", "down_proj"],
        help="Train multiple projection sidecars jointly. Overrides --projection when supplied.",
    )
    parser.add_argument(
        "--trainable",
        choices=["scale_delta", "output_bias", "low_rank_residual"],
        default="output_bias",
    )
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument("--low-rank-init-scale", type=float, default=1.0e-3)
    parser.add_argument(
        "--reinitialize-existing-sidecars",
        action="store_true",
        help=(
            "Ignore existing sidecars for the selected trainable parameters. "
            "Use this for honest down-rank probes from artifacts that already declare low-rank sidecars."
        ),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--target-nll-weight", type=float, default=0.0)
    parser.add_argument("--teacher-top1-margin-weight", type=float, default=0.0)
    parser.add_argument("--teacher-top1-margin", type=float, default=0.0)
    parser.add_argument(
        "--teacher-top1-competitor-token-ids",
        help=(
            "VQ competitor token ids aligned to auxiliary loss positions. Commas separate "
            "positions; plus signs add multiple competitors for the same position, e.g. "
            "'220+82,17'. When provided, the top1 margin loss compares each teacher top1 "
            "token against the matching explicit competitor group instead of the dynamic "
            "hardest competitor."
        ),
    )
    parser.add_argument(
        "--teacher-top1-include-hardest-competitor",
        action="store_true",
        help=(
            "When explicit competitor token ids are provided, also include the dynamic hardest-competitor "
            "top1 margin term instead of replacing it."
        ),
    )
    parser.add_argument(
        "--tail-kld-weight",
        type=float,
        default=0.0,
        help="Add a max token-level KL term over the selected training positions to directly target p999 tails.",
    )
    parser.add_argument(
        "--aux-loss-position-indices",
        help=(
            "Comma-separated zero-based selected-token indices for auxiliary top1-margin and tail-KLD losses. "
            "Base KL and target NLL still use all selected positions."
        ),
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--loss-scope",
        choices=["full_model", "final_layer_selected", "selected_layer"],
        default="full_model",
        help=(
            "Use full-model KL, a final-layer selected-token KL path, or a selected-layer "
            "path that splices the target layer update back through downstream layers."
        ),
    )
    parser.add_argument(
        "--surrogate-projections",
        nargs="*",
        choices=["gate_proj", "up_proj", "down_proj"],
        default=[],
        help="Projection calls to replace with chunked route-local differentiable surrogates during loss.",
    )
    parser.add_argument("--surrogate-output-chunk-size", type=int, default=256)
    parser.add_argument(
        "--train-cache",
        choices=["selection", "validation", "both"],
        default="selection",
        help="Teacher cache split used for KL updates. 'validation' is the report cache in this workflow.",
    )
    parser.add_argument("--max-train-rows", type=int, default=8)
    parser.add_argument(
        "--train-row-indices",
        help=(
            "Comma-separated zero-based teacher row indices to train on instead of the split prefix. "
            "With --train-cache both, the same indices are selected from each cache before interleaving."
        ),
    )
    parser.add_argument("--max-positions", type=int, default=16)
    parser.add_argument("--min-top-k", type=int, default=128)
    parser.add_argument("--allow-dirty-cache", action="store_true")
    parser.add_argument(
        "--cache-training-prefix",
        dest="cache_training_prefix",
        action="store_true",
        default=True,
        help=(
            "For final-layer selected-token training, cache frozen decoder layers before the target layer "
            "once per training row and reuse that prefix across optimizer steps."
        ),
    )
    parser.add_argument(
        "--no-cache-training-prefix",
        dest="cache_training_prefix",
        action="store_false",
        help="Disable final-layer prefix caching and use the uncached reference path.",
    )
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--append-jsonl", required=True)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.teacher_top1_margin_weight < 0.0:
        parser.error("--teacher-top1-margin-weight must be non-negative")
    if args.teacher_top1_margin < 0.0:
        parser.error("--teacher-top1-margin must be non-negative")
    if args.tail_kld_weight < 0.0:
        parser.error("--tail-kld-weight must be non-negative")
    if args.max_train_rows is not None and args.max_train_rows <= 0:
        parser.error("--max-train-rows must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be positive")
    if args.surrogate_output_chunk_size <= 0:
        parser.error("--surrogate-output-chunk-size must be positive")
    if args.low_rank <= 0:
        parser.error("--low-rank must be positive")
    if args.low_rank_init_scale <= 0.0:
        parser.error("--low-rank-init-scale must be positive")
    surrogate_projections = frozenset(args.surrogate_projections)
    if surrogate_projections and args.loss_scope not in {"final_layer_selected", "selected_layer"}:
        parser.error("--surrogate-projections currently requires a selected-token loss scope")
    try:
        trainable_projections = _resolve_trainable_projections(args.projection, args.projections)
    except ValueError as error:
        parser.error(str(error))
    try:
        train_row_indices = _parse_row_indices(args.train_row_indices)
    except ValueError as error:
        parser.error(str(error))
    try:
        aux_loss_position_indices = _parse_position_indices(args.aux_loss_position_indices)
    except ValueError as error:
        parser.error(str(error))
    try:
        teacher_top1_competitor_token_ids = _parse_competitor_token_ids(args.teacher_top1_competitor_token_ids)
    except ValueError as error:
        parser.error(str(error))

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.allow_existing:
        parser.error(f"--output-dir already exists: {output_dir}")

    selection_jsonl = Path(args.selection_teacher_jsonl)
    selection_root = Path(args.selection_cache_root)
    validation_jsonl = Path(args.validation_teacher_jsonl)
    validation_root = Path(args.validation_cache_root)
    seed_artifact_dir = Path(args.seed_artifact_dir)

    selection_validation = _require_clean_full_logit_cache(
        teacher_jsonl=selection_jsonl,
        cache_root=selection_root,
        min_top_k=args.min_top_k,
        allow_dirty_cache=args.allow_dirty_cache,
    )
    validation_validation = _require_clean_full_logit_cache(
        teacher_jsonl=validation_jsonl,
        cache_root=validation_root,
        min_top_k=args.min_top_k,
        allow_dirty_cache=args.allow_dirty_cache,
    )
    selection_train_rows = _prepare_teacher_rows(
        teacher_jsonl=selection_jsonl,
        cache_root=selection_root,
        max_rows=args.max_train_rows,
        max_positions=args.max_positions,
        row_indices=train_row_indices if args.train_cache in {"selection", "both"} else None,
    )
    validation_train_rows = None
    if args.train_cache in {"validation", "both"}:
        validation_train_rows = _prepare_teacher_rows(
            teacher_jsonl=validation_jsonl,
            cache_root=validation_root,
            max_rows=args.max_train_rows,
            max_positions=args.max_positions,
            row_indices=train_row_indices if args.train_cache in {"validation", "both"} else None,
        )
    if args.train_cache == "selection":
        train_rows = selection_train_rows
    elif args.train_cache == "validation":
        assert validation_train_rows is not None
        train_rows = validation_train_rows
    else:
        assert validation_train_rows is not None
        train_rows = _interleave_rows(selection_train_rows, validation_train_rows)

    start_record = {
        "schema_version": 1,
        "record_type": "air_vq_continuous_finetune_start",
        "model_id": args.model_id,
        "revision": args.revision,
        "seed_artifact_dir": str(seed_artifact_dir),
        "output_dir": str(output_dir),
        "selection_teacher_jsonl": str(selection_jsonl),
        "selection_cache_root": str(selection_root),
        "validation_teacher_jsonl": str(validation_jsonl),
        "validation_cache_root": str(validation_root),
        "selection_cache_rows": selection_validation["row_count"],
        "validation_cache_rows": validation_validation["row_count"],
        "train_row_count": len(train_rows),
        "train_cache": args.train_cache,
        "train_row_indices": None if train_row_indices is None else list(train_row_indices),
        "trainable": args.trainable,
        "low_rank": int(args.low_rank),
        "low_rank_init_scale": float(args.low_rank_init_scale),
        "reinitialize_existing_sidecars": bool(args.reinitialize_existing_sidecars),
        "layer": int(args.layer),
        "projection": trainable_projections[0] if len(trainable_projections) == 1 else None,
        "projections": list(trainable_projections),
        "prefill_engine": args.prefill_engine,
        "use_gather_switch": not args.disable_gather_switch,
        "loss_scope": args.loss_scope,
        "cache_training_prefix": bool(args.cache_training_prefix),
        "surrogate_projections": sorted(surrogate_projections),
        "surrogate_output_chunk_size": int(args.surrogate_output_chunk_size),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "teacher_top1_margin_weight": float(args.teacher_top1_margin_weight),
        "teacher_top1_margin": float(args.teacher_top1_margin),
        "teacher_top1_competitor_token_ids": None
        if teacher_top1_competitor_token_ids is None
        else _serialize_competitor_token_groups(teacher_top1_competitor_token_ids),
        "teacher_top1_include_hardest_competitor": bool(args.teacher_top1_include_hardest_competitor),
        "tail_kld_weight": float(args.tail_kld_weight),
        "aux_loss_position_indices": None
        if aux_loss_position_indices is None
        else list(aux_loss_position_indices),
        "max_positions": args.max_positions,
    }
    append_jsonl(args.append_jsonl, start_record)

    model, _, _, _ = load_resident_air(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=seed_artifact_dir,
        use_gather_switch=not args.disable_gather_switch,
        prefill_engine=args.prefill_engine,
    )
    projections = {
        projection_name: _target_projection(model, layer=args.layer, projection=projection_name)
        for projection_name in trainable_projections
    }
    params, initialized_from_existing_sidecar = _initial_trainable_params(
        projections,
        trainable=args.trainable,
        low_rank=args.low_rank,
        low_rank_init_scale=args.low_rank_init_scale,
        reinitialize_existing_sidecars=args.reinitialize_existing_sidecars,
    )
    use_prefix_cache = bool(args.cache_training_prefix and args.loss_scope == "final_layer_selected")
    prefix_cached_rows: list[FinalLayerPrefixCache] | None = None
    if use_prefix_cache:
        if args.layer != len(model.model.layers) - 1:
            raise ValueError("--cache-training-prefix requires --layer to be the final decoder layer")
        prefix_cached_rows = [
            _cache_final_layer_training_row(model, row, layer=args.layer)
            for row in train_rows
        ]

    def loss_fn(
        values: dict[str, mx.array],
        row: PreparedTeacherRow,
        prefix_cache: FinalLayerPrefixCache | None,
    ) -> mx.array:
        _assign_trainable_params(projections, values, trainable=args.trainable)
        return _kl_loss(
            model,
            row,
            target_nll_weight=args.target_nll_weight,
            teacher_top1_margin_weight=args.teacher_top1_margin_weight,
            teacher_top1_margin=args.teacher_top1_margin,
            teacher_top1_competitor_token_ids=teacher_top1_competitor_token_ids,
            teacher_top1_include_hardest_competitor=args.teacher_top1_include_hardest_competitor,
            tail_kld_weight=args.tail_kld_weight,
            aux_loss_position_indices=aux_loss_position_indices,
            loss_scope=args.loss_scope,
            layer=args.layer,
            surrogate_projections=surrogate_projections,
            surrogate_output_chunk_size=args.surrogate_output_chunk_size,
            final_layer_prefix_cache=prefix_cache,
            teacher_log_probs=None if prefix_cache is None else prefix_cache.teacher_log_probs,
            teacher_probs=None if prefix_cache is None else prefix_cache.teacher_probs,
        )

    last_loss = None
    for step in range(args.steps):
        row_index = step % len(train_rows)
        row = train_rows[row_index]
        prefix_cache = None if prefix_cached_rows is None else prefix_cached_rows[row_index]
        started = time.perf_counter()
        try:
            loss_value, grads = mx.value_and_grad(lambda values: loss_fn(values, row, prefix_cache))(params)
            params, norms = _apply_updates(
                params,
                grads,
                learning_rate=args.learning_rate,
                grad_clip_norm=args.grad_clip_norm,
            )
            _assign_trainable_params(projections, params, trainable=args.trainable)
            mx.eval(loss_value, params, norms)
            elapsed = time.perf_counter() - started
            last_loss = float(loss_value.item())
            grad_norms = {
                projection_name: float(norm.item())
                for projection_name, norm in norms.items()
            }
            step_record = {
                "schema_version": 1,
                "record_type": "air_vq_continuous_finetune_step",
                "step": int(step),
                "prompt_id": row.prompt_id,
                "row_index": int(row.row_index),
                "target_token_count": int(len(row.positions)),
                "trainable": args.trainable,
                "projections": list(trainable_projections),
                "loss": last_loss,
                "grad_norm": max(grad_norms.values()),
                "grad_norms": grad_norms,
                "elapsed_seconds": elapsed,
                "cache_training_prefix": bool(prefix_cache is not None),
            }
            append_jsonl(args.append_jsonl, step_record)
        except Exception as error:
            failure = {
                "schema_version": 1,
                "record_type": "air_vq_continuous_finetune_failure",
                "step": int(step),
                "prompt_id": row.prompt_id,
                "row_index": int(row.row_index),
                "trainable": args.trainable,
                "projections": list(trainable_projections),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            append_jsonl(args.append_jsonl, failure)
            print(json.dumps(failure, indent=2, sort_keys=True))
            raise SystemExit(2) from error
        finally:
            _maybe_clear_cache()

    linked_count = link_seed_artifact_groups(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
    )
    preserved_logit_bias = copy_declared_logit_bias_sidecar(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
    )
    preserved_sidecars = copy_declared_continuous_sidecars(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
        exclude={(int(args.layer), projection_name) for projection_name in trainable_projections},
    )
    sidecars = []
    for projection_name in trainable_projections:
        if args.trainable == "scale_delta":
            param = params[projection_name]
            sidecar = write_continuous_sidecar(
                output_dir=output_dir,
                layer=args.layer,
                projection=projection_name,
                scale_delta=param,
            )
        elif args.trainable == "output_bias":
            param = params[projection_name]
            sidecar = write_continuous_sidecar(
                output_dir=output_dir,
                layer=args.layer,
                projection=projection_name,
                output_bias=param,
            )
        else:
            projection = projections[projection_name]
            sidecar = write_continuous_sidecar(
                output_dir=output_dir,
                layer=args.layer,
                projection=projection_name,
                scale_delta=projection.get("continuous_scale_delta"),
                output_bias=projection.get("continuous_output_bias"),
                low_rank_left=params[f"{projection_name}.low_rank_left"],
                low_rank_right=params[f"{projection_name}.low_rank_right"],
            )
        sidecars.append(sidecar)
    run_manifest = {
        "kind": "rung2_continuous_finetune_probe",
        "trainable": args.trainable,
        "low_rank": int(args.low_rank),
        "low_rank_init_scale": float(args.low_rank_init_scale),
        "reinitialize_existing_sidecars": bool(args.reinitialize_existing_sidecars),
        "initialized_from_existing_sidecar": initialized_from_existing_sidecar,
        "layer": int(args.layer),
        "projection": trainable_projections[0] if len(trainable_projections) == 1 else None,
        "projections": list(trainable_projections),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "target_nll_weight": float(args.target_nll_weight),
        "teacher_top1_margin_weight": float(args.teacher_top1_margin_weight),
        "teacher_top1_margin": float(args.teacher_top1_margin),
        "teacher_top1_competitor_token_ids": None
        if teacher_top1_competitor_token_ids is None
        else _serialize_competitor_token_groups(teacher_top1_competitor_token_ids),
        "teacher_top1_include_hardest_competitor": bool(args.teacher_top1_include_hardest_competitor),
        "tail_kld_weight": float(args.tail_kld_weight),
        "aux_loss_position_indices": None
        if aux_loss_position_indices is None
        else list(aux_loss_position_indices),
        "max_train_rows": args.max_train_rows,
        "max_positions": args.max_positions,
        "train_row_indices": None if train_row_indices is None else list(train_row_indices),
        "train_cache": args.train_cache,
        "prefill_engine": args.prefill_engine,
        "use_gather_switch": not args.disable_gather_switch,
        "loss_scope": args.loss_scope,
        "cache_training_prefix": bool(args.cache_training_prefix),
        "prefix_cache_effective": bool(use_prefix_cache),
        "surrogate_projections": sorted(surrogate_projections),
        "surrogate_output_chunk_size": int(args.surrogate_output_chunk_size),
        "final_train_loss": last_loss,
    }
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed_artifact_dir,
        output_dir=output_dir,
        sidecars=[*preserved_sidecars, *sidecars],
        run_manifest=run_manifest,
    )
    summary = {
        "schema_version": 1,
        "record_type": "air_vq_continuous_finetune_summary",
        "ok": True,
        "output_dir": str(output_dir),
        "linked_group_count": int(linked_count),
        "preserved_sidecar_count": int(len(preserved_sidecars)),
        "preserved_logit_bias_sidecar": preserved_logit_bias is not None,
        "sidecar": sidecars[0] if len(sidecars) == 1 else None,
        "sidecars": sidecars,
        "run": run_manifest,
    }
    append_jsonl(args.append_jsonl, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
