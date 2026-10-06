from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import (
    CODEWORD_DIM,
    E8P_SHUFFLE_MAP,
    decode_e8_1bit,
    decode_e8p,
    decode_weight_matrix,
    e8p_abs_grid,
)
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config, validate_glm45_air
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.models.glm4_moe_adapter import GLM4MoEGate
from mlx_vq.validate.glm45_air_vq import (
    ArrayComparisonMetrics,
    _array_metrics,
    _expert_weight_name,
    _load_switch_glu,
    _read_named_tensor,
    _routing_config,
)


@dataclass(frozen=True)
class HessianReassignStats:
    group_size: int
    code_bits: int
    row_count: int
    codewords_per_row: int
    code_count: int
    changed_code_count: int
    changed_code_fraction: float
    current_hessian_weighted_error: float
    hessian_weighted_error: float
    hessian_weighted_error_ratio: float
    current_weight_rel_l2: float
    hessian_weight_rel_l2: float


@dataclass(frozen=True)
class HessianReassignedWeight:
    codes: np.ndarray
    weight: np.ndarray
    stats: HessianReassignStats


@dataclass(frozen=True)
class ProjectionHessianProbeResult:
    projection: str
    selected_experts: tuple[int, ...]
    route_count: int
    current_output: ArrayComparisonMetrics
    hessian_output: ArrayComparisonMetrics
    hessian_vs_current_output: ArrayComparisonMetrics
    current_hessian_weighted_error: float
    hessian_weighted_error: float
    hessian_weighted_error_ratio: float
    changed_code_count: int
    code_count: int
    changed_code_fraction: float
    expert_stats: dict[str, HessianReassignStats]

    def to_dict(self) -> dict[str, object]:
        return {
            "projection": self.projection,
            "selected_experts": list(self.selected_experts),
            "route_count": self.route_count,
            "current_output": asdict(self.current_output),
            "hessian_output": asdict(self.hessian_output),
            "hessian_vs_current_output": asdict(self.hessian_vs_current_output),
            "current_hessian_weighted_error": self.current_hessian_weighted_error,
            "hessian_weighted_error": self.hessian_weighted_error,
            "hessian_weighted_error_ratio": self.hessian_weighted_error_ratio,
            "changed_code_count": self.changed_code_count,
            "code_count": self.code_count,
            "changed_code_fraction": self.changed_code_fraction,
            "expert_stats": {name: asdict(stats) for name, stats in self.expert_stats.items()},
        }


@dataclass(frozen=True)
class GLM45AirHessianProbeResult:
    model_id: str
    layer: int
    tokens: int
    input_source: str
    prompt_text: str | None
    prompt_token_ids: tuple[int, ...]
    input_state_indices: tuple[int, ...]
    hidden_size: int
    moe_intermediate_size: int
    top_k: int
    selected_experts: tuple[int, ...]
    routes: int
    sampled_output_rows: tuple[int, ...]
    source_tensors_read: int
    peak_source_tensor_bytes: int
    source_dir: str
    artifact_dir: str
    routed_indices: tuple[tuple[int, ...], ...]
    router_scores: tuple[tuple[float, ...], ...]
    hessian: dict[str, object]
    projection_results: dict[str, ProjectionHessianProbeResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "tokens": self.tokens,
            "input_source": self.input_source,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "input_state_indices": list(self.input_state_indices),
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "top_k": self.top_k,
            "selected_experts": list(self.selected_experts),
            "routes": self.routes,
            "sampled_output_rows": list(self.sampled_output_rows),
            "source_tensors_read": self.source_tensors_read,
            "peak_source_tensor_bytes": self.peak_source_tensor_bytes,
            "source_dir": self.source_dir,
            "artifact_dir": self.artifact_dir,
            "routed_indices": [list(row) for row in self.routed_indices],
            "router_scores": [list(row) for row in self.router_scores],
            "hessian": self.hessian,
            "projection_results": {
                projection: result.to_dict() for projection, result in self.projection_results.items()
            },
            "probe": {
                "kind": "diagonal_hessian_code_reassignment_decision_probe",
                "candidate_artifact_mutated": False,
                "acceptance_gate": False,
                "notes": (
                    "This is a sampled diagonal-Hessian proxy over frozen scales and selected output rows. "
                    "It is not full BlockLDLQ and does not create or accept a candidate artifact."
                ),
            },
        }


def _safe_ratio(numerator: float, denominator: float, *, eps: float = 1e-12) -> float:
    if abs(denominator) <= eps:
        return 1.0 if abs(numerator) <= eps else float("inf")
    return float(numerator / denominator)


def _select_or_random_input_states(
    *,
    summary_hidden_size: int,
    tokens: int,
    seed: int,
    input_scale: float,
    input_states: np.ndarray | None,
    input_source: str | None,
    input_state_indices: tuple[int, ...] | None,
) -> tuple[np.ndarray, int, str, tuple[int, ...]]:
    if input_states is None:
        rng = np.random.default_rng(seed)
        x = rng.normal(loc=0.0, scale=input_scale, size=(tokens, summary_hidden_size)).astype(np.float32)
        return x, tokens, input_source or "random_normal", tuple(range(tokens))

    x = np.asarray(input_states, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"input_states must have shape [tokens, hidden], found {x.shape}")
    if x.shape[0] <= 0:
        raise ValueError("input_states must include at least one token row")
    if x.shape[1] != summary_hidden_size:
        raise ValueError(f"input_states hidden dimension must be {summary_hidden_size}, found {x.shape[1]}")
    resolved_indices = input_state_indices or tuple(range(x.shape[0]))
    if len(resolved_indices) != x.shape[0]:
        raise ValueError(
            "input_state_indices length must match input_states token rows, "
            f"found {len(resolved_indices)} and {x.shape[0]}"
        )
    return x, int(x.shape[0]), input_source or "provided", tuple(int(value) for value in resolved_indices)


def _sample_output_rows(output_dim: int, max_output_rows: int) -> np.ndarray:
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if max_output_rows <= 0:
        raise ValueError("max_output_rows must be positive")
    if max_output_rows >= output_dim:
        return np.arange(output_dim, dtype=np.int64)
    return np.linspace(0, output_dim - 1, num=max_output_rows, dtype=np.int64)


def _decode_from_full_codebook(codes: np.ndarray, scales: np.ndarray, codebook: np.ndarray, *, group_size: int) -> np.ndarray:
    code_array = np.asarray(codes)
    scale_array = np.asarray(scales, dtype=np.float32)
    table = np.asarray(codebook, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != CODEWORD_DIM:
        raise ValueError(f"codebook must have shape [N, {CODEWORD_DIM}], found {table.shape}")
    if code_array.size and int(np.max(code_array)) >= table.shape[0]:
        raise ValueError("codes reference entries outside the supplied codebook")
    codewords = code_array.shape[-1]
    words_per_scale = group_size // CODEWORD_DIM
    if codewords % words_per_scale != 0:
        raise ValueError("codeword count must be divisible by group_size / 8")
    if scale_array.shape[-1] != codewords // words_per_scale:
        raise ValueError("scale count does not match codeword groups")
    expanded_scales = np.repeat(scale_array, words_per_scale, axis=-1)[..., None]
    decoded = table[code_array.astype(np.int64)] * expanded_scales
    return decoded.reshape((*decoded.shape[:-2], decoded.shape[-2] * CODEWORD_DIM)).astype(np.float32)


def _hessian_error(weight: np.ndarray, source: np.ndarray, hessian_diag: np.ndarray) -> float:
    diff = np.asarray(weight, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    hessian = np.asarray(hessian_diag, dtype=np.float64)
    return float(np.sum(diff * diff * hessian[None, :]))


def _rel_l2(weight: np.ndarray, source: np.ndarray) -> float:
    diff = np.asarray(weight, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    source_norm = np.linalg.norm(np.asarray(source, dtype=np.float64).reshape(-1))
    return float(np.linalg.norm(diff.reshape(-1)) / max(float(source_norm), 1e-12))


def _sign_parity(signs: np.ndarray) -> np.ndarray:
    parity = np.zeros_like(signs, dtype=np.uint32)
    for bit in range(CODEWORD_DIM):
        parity ^= (signs >> np.uint32(bit)) & np.uint32(1)
    return parity


def _weighted_encode_e8p(
    vectors: np.ndarray,
    hessian_vectors: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(hessian_vectors, dtype=np.float32)
    if values.shape != weights.shape or values.shape[-1] != CODEWORD_DIM:
        raise ValueError(
            "vectors and hessian_vectors must have matching trailing dimension "
            f"{CODEWORD_DIM}; found {values.shape} and {weights.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat = values.reshape(-1, CODEWORD_DIM)
    flat_weights = weights.reshape(-1, CODEWORD_DIM)
    base_rows = e8p_abs_grid().astype(np.float32)
    abs_rows = np.abs(base_rows)
    abs_rows_sq = abs_rows * abs_rows
    packed_dim_by_out_dim = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs = np.zeros(abs_rows.shape[0], dtype=np.uint16)
    base_negative = base_rows < 0
    for out_dim, packed_dim in enumerate(packed_dim_by_out_dim):
        base_signs |= base_negative[:, out_dim].astype(np.uint16) << packed_dim

    best_codes = np.zeros(flat.shape[0], dtype=np.uint16)
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        chunk_weights = flat_weights[start : start + chunk.shape[0]]
        local_best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        local_best_codes = np.zeros(chunk.shape[0], dtype=np.uint16)

        for parity, shift in ((0, np.float32(0.25)), (1, np.float32(-0.25))):
            target = chunk - shift
            abs_target = np.abs(target)
            distances = (
                np.sum(chunk_weights * abs_target * abs_target, axis=1)[:, None]
                + chunk_weights @ abs_rows_sq.T
                - 2.0 * ((chunk_weights * abs_target) @ abs_rows.T)
            )

            negative = target < 0
            target_signs = np.zeros(chunk.shape[0], dtype=np.uint16)
            for out_dim, packed_dim in enumerate(packed_dim_by_out_dim):
                target_signs |= negative[:, out_dim].astype(np.uint16) << packed_dim

            effective_signs = (target_signs[:, None] ^ base_signs[None, :]).astype(np.uint16)
            sign_parity = _sign_parity(effective_signs.astype(np.uint32)).astype(bool)
            flip_dim = None
            if np.any(sign_parity):
                flip_penalty = 4.0 * chunk_weights[:, None, :] * abs_target[:, None, :] * abs_rows[None, :, :]
                flip_dim = np.argmin(flip_penalty, axis=2).astype(np.uint8)
                correction = np.min(flip_penalty, axis=2)
                distances = np.where(sign_parity, distances + correction, distances)

            abs_idx = np.argmin(distances, axis=1).astype(np.uint16)
            row_dist = distances[np.arange(chunk.shape[0]), abs_idx]
            candidate_signs = effective_signs[np.arange(chunk.shape[0]), abs_idx].copy()
            if flip_dim is not None and np.any(sign_parity):
                selected_flip_dim = flip_dim[np.arange(chunk.shape[0]), abs_idx]
                flip_masks = (np.uint16(1) << packed_dim_by_out_dim[selected_flip_dim]).astype(np.uint16)
                selected_needs_flip = sign_parity[np.arange(chunk.shape[0]), abs_idx]
                candidate_signs = np.where(
                    selected_needs_flip,
                    candidate_signs ^ flip_masks,
                    candidate_signs,
                ).astype(np.uint16)

            stored_signs = (candidate_signs ^ np.uint16(parity)).astype(np.uint16)
            codes = ((abs_idx.astype(np.uint16) << np.uint16(8)) | stored_signs).astype(np.uint16)
            improved = row_dist < local_best_dist
            local_best_dist[improved] = row_dist[improved]
            local_best_codes[improved] = codes[improved]

        best_codes[start : start + chunk.shape[0]] = local_best_codes
    return best_codes.reshape(values.shape[:-1])


def hessian_weighted_reassign_codes(
    source_weight: np.ndarray,
    current_codes: np.ndarray,
    scales: np.ndarray,
    hessian_diag: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int = 8,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None = None,
    codeword_chunk_size: int = 4096,
) -> HessianReassignedWeight:
    """Reassign 8D codewords under a diagonal-Hessian weighted error proxy.

    The probe freezes artifact scales and searches the matching E8-family table
    per 8D codeword. It intentionally stays a diagonal proxy rather than a full
    BlockLDLQ implementation.
    """

    if code_bits not in (8, 16):
        raise ValueError("hessian reassignment supports code_bits 8 or 16")
    if group_size <= 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if codeword_chunk_size <= 0:
        raise ValueError("codeword_chunk_size must be positive")

    source = np.asarray(source_weight, dtype=np.float32)
    codes = np.asarray(current_codes)
    scale_array = np.asarray(scales, dtype=np.float32)
    hessian = np.asarray(hessian_diag, dtype=np.float32)
    codebook_array = np.asarray(codebook)

    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    if source.shape[1] % CODEWORD_DIM != 0:
        raise ValueError("source_weight input dimension must be divisible by 8")
    if codes.ndim != 2:
        raise ValueError(f"current_codes must be 2D [out, codewords], found {codes.shape}")
    if scale_array.ndim != 2:
        raise ValueError(f"scales must be 2D [out, groups], found {scale_array.shape}")
    if code_bits == 8:
        table = (
            decode_e8_1bit(np.arange(256, dtype=np.uint8), codebook_array)
            if codebook_array.ndim == 1
            else codebook_array.astype(np.float32)
        )
        if table.ndim != 2 or table.shape[1] != CODEWORD_DIM or table.shape[0] > 256:
            raise ValueError(f"8-bit reassignment codebook must have shape [N<=256, {CODEWORD_DIM}], found {table.shape}")
    else:
        table = None
        if codebook_array.shape != (256,):
            raise ValueError(f"16-bit E8P reassignment codebook must be packed shape [256], found {codebook_array.shape}")
    if hessian.shape != (source.shape[1],):
        raise ValueError(f"hessian_diag must have shape ({source.shape[1]},), found {hessian.shape}")
    if not np.isfinite(hessian).all() or np.any(hessian < 0.0):
        raise ValueError("hessian_diag must be finite and non-negative")
    if codes.shape[0] != source.shape[0] or scale_array.shape[0] != source.shape[0]:
        raise ValueError("source_weight, current_codes, and scales must have matching output rows")
    if codes.shape[1] != source.shape[1] // CODEWORD_DIM:
        raise ValueError("current_codes codeword count does not match source input dimension")
    if source.shape[1] % group_size != 0:
        raise ValueError("source input dimension must be divisible by group_size")
    if scale_array.shape[1] != source.shape[1] // group_size:
        raise ValueError("scale group count does not match source input dimension")

    if row_indices is None:
        selected_rows = np.arange(source.shape[0], dtype=np.int64)
    else:
        selected_rows = np.asarray(row_indices, dtype=np.int64)
    if selected_rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if selected_rows.size and (int(np.min(selected_rows)) < 0 or int(np.max(selected_rows)) >= source.shape[0]):
        raise ValueError("row_indices contain rows outside source_weight")

    selected_source = source[selected_rows]
    selected_codes = codes[selected_rows]
    selected_scales = scale_array[selected_rows]
    current_weight = (
        _decode_from_full_codebook(selected_codes, selected_scales, table, group_size=group_size)
        if code_bits == 8 and table is not None
        else decode_weight_matrix(
            selected_codes,
            selected_scales,
            code_bits=16,
            codebook=codebook_array,
        ).astype(np.float32, copy=False)
    )

    row_count = selected_source.shape[0]
    codewords = selected_codes.shape[1]
    words_per_scale = group_size // CODEWORD_DIM
    expanded_scales = np.repeat(selected_scales, words_per_scale, axis=-1)
    source_vectors = selected_source.reshape(row_count, codewords, CODEWORD_DIM)
    hessian_vectors = hessian.reshape(codewords, CODEWORD_DIM)

    flat_source = source_vectors.reshape(-1, CODEWORD_DIM)
    flat_scales = expanded_scales.reshape(-1).astype(np.float32)
    flat_hessian = np.broadcast_to(hessian_vectors[None, :, :], (row_count, codewords, CODEWORD_DIM)).reshape(
        -1, CODEWORD_DIM
    )
    if code_bits == 8:
        assert table is not None
        flat_new_codes = np.empty(flat_source.shape[0], dtype=np.uint8)
        table64 = table.astype(np.float64)
        for start in range(0, flat_source.shape[0], codeword_chunk_size):
            end = min(start + codeword_chunk_size, flat_source.shape[0])
            target = flat_source[start:end].astype(np.float64)
            scale = flat_scales[start:end].astype(np.float64)
            hess = flat_hessian[start:end].astype(np.float64)
            candidates = table64[None, :, :] * scale[:, None, None]
            diff = target[:, None, :] - candidates
            distances = np.sum(diff * diff * hess[:, None, :], axis=2)
            flat_new_codes[start:end] = np.argmin(distances, axis=1).astype(np.uint8)
    else:
        safe_scales = np.where(flat_scales > 0.0, flat_scales, np.float32(1.0)).astype(np.float32)
        normalized_source = (flat_source / safe_scales[:, None]).astype(np.float32)
        flat_new_codes = _weighted_encode_e8p(
            normalized_source,
            flat_hessian.astype(np.float32, copy=False),
            chunk_size=codeword_chunk_size,
        )

    new_codes = flat_new_codes.reshape(selected_codes.shape)
    new_weight = (
        _decode_from_full_codebook(new_codes, selected_scales, table, group_size=group_size)
        if code_bits == 8 and table is not None
        else decode_weight_matrix(
            new_codes,
            selected_scales,
            code_bits=16,
            codebook=codebook_array,
        ).astype(np.float32, copy=False)
    )
    current_error = _hessian_error(current_weight, selected_source, hessian)
    new_error = _hessian_error(new_weight, selected_source, hessian)
    changed = int(np.count_nonzero(new_codes != selected_codes))
    code_count = int(new_codes.size)
    stats = HessianReassignStats(
        group_size=int(group_size),
        code_bits=int(code_bits),
        row_count=int(row_count),
        codewords_per_row=int(codewords),
        code_count=code_count,
        changed_code_count=changed,
        changed_code_fraction=float(changed / code_count) if code_count else 0.0,
        current_hessian_weighted_error=current_error,
        hessian_weighted_error=new_error,
        hessian_weighted_error_ratio=_safe_ratio(new_error, current_error),
        current_weight_rel_l2=_rel_l2(current_weight, selected_source),
        hessian_weight_rel_l2=_rel_l2(new_weight, selected_source),
    )
    return HessianReassignedWeight(codes=new_codes, weight=new_weight, stats=stats)


def _select_experts(indices: np.ndarray, max_experts: int) -> tuple[int, ...]:
    if max_experts <= 0:
        raise ValueError("max_experts must be positive")
    unique, counts = np.unique(indices.reshape(-1), return_counts=True)
    ranked = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-int(item[1]), int(item[0])))
    return tuple(int(expert) for expert, _ in ranked[:max_experts])


def _full_8bit_table_from_layer(layer) -> np.ndarray:
    if layer.code_bits != 8:
        raise ValueError(
            "diagonal Hessian code reassignment currently supports code_bits=8 only; "
            f"projection has code_bits={layer.code_bits}"
        )
    return decode_e8_1bit(np.arange(256, dtype=np.uint8), np.asarray(layer.codebook))


def _full_codebook_table_from_layer(layer) -> np.ndarray:
    if layer.code_bits == 8:
        return decode_e8_1bit(np.arange(256, dtype=np.uint8), np.asarray(layer.codebook))
    if layer.code_bits == 16:
        return decode_e8p(np.arange(1 << 16, dtype=np.uint16), np.asarray(layer.codebook))
    raise ValueError(f"unsupported code_bits={layer.code_bits}; expected 8 or 16")


def _evaluate_projection(
    *,
    projection_name: str,
    projection_layer,
    source_dir: Path,
    index,
    layer: int,
    x: np.ndarray,
    indices: np.ndarray,
    selected_experts: tuple[int, ...],
    sampled_rows: np.ndarray,
    hessian_diag: np.ndarray,
) -> tuple[ProjectionHessianProbeResult, int, int]:
    full_codebook = _full_8bit_table_from_layer(projection_layer)
    source_outputs: list[np.ndarray] = []
    current_outputs: list[np.ndarray] = []
    hessian_outputs: list[np.ndarray] = []
    expert_stats: dict[str, HessianReassignStats] = {}
    source_tensors_read = 0
    peak_source_bytes = 0
    current_error = 0.0
    hessian_error = 0.0
    changed_codes = 0
    total_codes = 0
    route_count = 0

    for expert in selected_experts:
        route_positions = np.argwhere(indices == expert)
        if route_positions.size == 0:
            continue

        source_weight = _read_named_tensor(source_dir, index, _expert_weight_name(layer, expert, projection_name)).astype(
            np.float32,
            copy=False,
        )
        source_tensors_read += 1
        peak_source_bytes = max(peak_source_bytes, int(source_weight.nbytes))

        current_codes = np.asarray(projection_layer.codes[expert])
        current_scales = np.asarray(projection_layer.scales[expert])
        current_weight = decode_weight_matrix(
            current_codes[sampled_rows],
            current_scales[sampled_rows],
            code_bits=projection_layer.code_bits,
            codebook=np.asarray(projection_layer.codebook),
        ).astype(np.float32, copy=False)
        reassigned = hessian_weighted_reassign_codes(
            source_weight,
            current_codes,
            current_scales,
            hessian_diag,
            codebook=full_codebook,
            group_size=projection_layer.group_size,
            code_bits=projection_layer.code_bits,
            row_indices=sampled_rows,
        )
        expert_stats[f"expert_{expert}"] = reassigned.stats
        current_error += reassigned.stats.current_hessian_weighted_error
        hessian_error += reassigned.stats.hessian_weighted_error
        changed_codes += reassigned.stats.changed_code_count
        total_codes += reassigned.stats.code_count

        source_rows = source_weight[sampled_rows]
        for token_idx, _route_idx in route_positions:
            token = x[token_idx]
            source_outputs.append(source_rows @ token)
            current_outputs.append(current_weight @ token)
            hessian_outputs.append(reassigned.weight @ token)
            route_count += 1

    if route_count == 0:
        shape = (0, int(sampled_rows.size))
        source_array = np.empty(shape, dtype=np.float32)
        current_array = np.empty(shape, dtype=np.float32)
        hessian_array = np.empty(shape, dtype=np.float32)
    else:
        source_array = np.stack(source_outputs).astype(np.float32, copy=False)
        current_array = np.stack(current_outputs).astype(np.float32, copy=False)
        hessian_array = np.stack(hessian_outputs).astype(np.float32, copy=False)

    result = ProjectionHessianProbeResult(
        projection=projection_name,
        selected_experts=selected_experts,
        route_count=route_count,
        current_output=_array_metrics(current_array, source_array),
        hessian_output=_array_metrics(hessian_array, source_array),
        hessian_vs_current_output=_array_metrics(hessian_array, current_array),
        current_hessian_weighted_error=current_error,
        hessian_weighted_error=hessian_error,
        hessian_weighted_error_ratio=_safe_ratio(hessian_error, current_error),
        changed_code_count=int(changed_codes),
        code_count=int(total_codes),
        changed_code_fraction=float(changed_codes / total_codes) if total_codes else 0.0,
        expert_stats=expert_stats,
    )
    return result, source_tensors_read, peak_source_bytes


def evaluate_glm45_air_hessian_probe(
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    index_path: str | Path,
    artifact_dir: str | Path,
    model_id: str = GLM45_AIR_MODEL_ID,
    layer: int = 1,
    tokens: int = 2,
    seed: int = 20260624,
    input_scale: float = 1.0,
    input_states: np.ndarray | None = None,
    input_source: str | None = None,
    prompt_text: str | None = None,
    prompt_token_ids: tuple[int, ...] = (),
    input_state_indices: tuple[int, ...] | None = None,
    projections: tuple[str, ...] = ("gate_proj", "up_proj"),
    max_experts: int = 8,
    max_output_rows: int = 128,
    strict_config: bool = True,
) -> GLM45AirHessianProbeResult:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if not projections:
        raise ValueError("at least one projection is required")

    normalized_projections = tuple(str(projection) for projection in projections)
    unsupported = sorted(set(normalized_projections) - {"gate_proj", "up_proj"})
    if unsupported:
        raise ValueError(
            "diagonal Hessian probe currently supports gate_proj/up_proj only; "
            f"unsupported projections: {', '.join(unsupported)}"
        )

    summary = summarize_config(config, model_id=model_id)
    if strict_config:
        failed_checks = validate_glm45_air(summary)
        if failed_checks:
            raise ValueError(f"config does not match GLM-4.5-Air: {', '.join(failed_checks)}")
    if summary.model_type != "glm4_moe":
        raise ValueError(f"expected model_type='glm4_moe', found {summary.model_type!r}")
    if layer < summary.dense_layers or layer >= summary.num_hidden_layers:
        raise ValueError(
            f"layer {layer} is not a sparse MoE layer; sparse range is "
            f"[{summary.dense_layers}, {summary.num_hidden_layers})"
        )

    source_root = Path(source_dir)
    artifact_root = Path(artifact_dir)
    index = load_safetensors_index(index_path)
    routing = _routing_config(config, model_id=model_id)

    gate_weight_name = f"model.layers.{layer}.mlp.gate.weight"
    correction_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    gate_weight = _read_named_tensor(source_root, index, gate_weight_name)
    source_tensors_read = 1
    peak_source_bytes = int(gate_weight.nbytes)
    if correction_name in index.weight_map:
        correction = _read_named_tensor(source_root, index, correction_name)
        source_tensors_read += 1
        peak_source_bytes = max(peak_source_bytes, int(correction.nbytes))
    else:
        correction = np.zeros((summary.n_routed_experts,), dtype=np.float32)

    x, tokens, resolved_input_source, resolved_state_indices = _select_or_random_input_states(
        summary_hidden_size=summary.hidden_size,
        tokens=tokens,
        seed=seed,
        input_scale=input_scale,
        input_states=input_states,
        input_source=input_source,
        input_state_indices=input_state_indices,
    )

    gate = GLM4MoEGate(
        routing,
        weight=mx.array(gate_weight),
        e_score_correction_bias=mx.array(correction),
    )
    indices_mx, scores_mx = gate(mx.array(x))
    mx.eval(indices_mx, scores_mx)
    indices = np.asarray(indices_mx, dtype=np.int64)
    scores = np.asarray(scores_mx.astype(mx.float32), dtype=np.float32)
    if indices.ndim != 2 or indices.shape[0] != tokens:
        raise ValueError(f"router indices must have shape [tokens, top_k], found {indices.shape}")

    selected_experts = _select_experts(indices, max_experts)
    selected_routes = int(np.count_nonzero(np.isin(indices, np.asarray(selected_experts, dtype=np.int64))))
    switch_glu = _load_switch_glu(artifact_root, layer)
    sampled_rows = _sample_output_rows(switch_glu.hidden_dims, max_output_rows)
    hessian_diag = np.mean(x.astype(np.float64) * x.astype(np.float64), axis=0).astype(np.float32)

    projection_layers = {
        "gate_proj": switch_glu.gate_proj,
        "up_proj": switch_glu.up_proj,
    }
    projection_results: dict[str, ProjectionHessianProbeResult] = {}
    for projection_name in normalized_projections:
        projection_result, projection_reads, projection_peak = _evaluate_projection(
            projection_name=projection_name,
            projection_layer=projection_layers[projection_name],
            source_dir=source_root,
            index=index,
            layer=layer,
            x=x,
            indices=indices,
            selected_experts=selected_experts,
            sampled_rows=sampled_rows,
            hessian_diag=hessian_diag,
        )
        projection_results[projection_name] = projection_result
        source_tensors_read += projection_reads
        peak_source_bytes = max(peak_source_bytes, projection_peak)

    hessian_summary: dict[str, object] = {
        "proxy": "mean_squared_moe_input_diagonal",
        "sample_count": int(x.shape[0]),
        "dimension": int(hessian_diag.shape[0]),
        "diag_min": float(np.min(hessian_diag)) if hessian_diag.size else 0.0,
        "diag_max": float(np.max(hessian_diag)) if hessian_diag.size else 0.0,
        "diag_mean": float(np.mean(hessian_diag)) if hessian_diag.size else 0.0,
        "diag_std": float(np.std(hessian_diag)) if hessian_diag.size else 0.0,
    }

    return GLM45AirHessianProbeResult(
        model_id=model_id,
        layer=layer,
        tokens=tokens,
        input_source=resolved_input_source,
        prompt_text=prompt_text,
        prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
        input_state_indices=tuple(int(index) for index in resolved_state_indices),
        hidden_size=summary.hidden_size,
        moe_intermediate_size=summary.moe_intermediate_size,
        top_k=int(indices.shape[1]),
        selected_experts=selected_experts,
        routes=selected_routes,
        sampled_output_rows=tuple(int(value) for value in sampled_rows),
        source_tensors_read=source_tensors_read,
        peak_source_tensor_bytes=peak_source_bytes,
        source_dir=str(source_root),
        artifact_dir=str(artifact_root),
        routed_indices=tuple(tuple(int(value) for value in row) for row in indices),
        router_scores=tuple(tuple(float(value) for value in row) for row in scores),
        hessian=hessian_summary,
        projection_results=projection_results,
    )
