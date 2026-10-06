from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mlx_vq.io.load import load_quantized_vq_switch_linear
from mlx_vq.kernels.e8p_rhs_layout import pack_e8p_rhs_tiles

ArtifactProjection = Literal["gate_proj", "up_proj", "down_proj"]

_ALL_PROJECTIONS: tuple[ArtifactProjection, ...] = ("gate_proj", "up_proj", "down_proj")


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return _round_float(float(np.percentile(np.asarray(values, dtype=np.float64), percentile)))


def _reuse_recommendation(
    *,
    global_reuse_factor: float,
    median_tile_unique_ratio: float | None,
) -> str:
    if median_tile_unique_ratio is None:
        return "insufficient_tile_samples"
    if median_tile_unique_ratio >= 0.90 or global_reuse_factor <= 1.25:
        return "low_code_reuse_do_not_prioritize_codebook_cache"
    if median_tile_unique_ratio <= 0.50 or global_reuse_factor >= 4.0:
        return "code_reuse_promising_for_codebook_cache_probe"
    return "mixed_code_reuse_requires_kernel_microbench"


def _tile_reuse_summary(
    *,
    code_tiles: np.ndarray,
    output_dims: int,
    bn: int,
    codewords_per_bk: int,
) -> dict[str, Any]:
    unique_counts: list[float] = []
    unique_ratios: list[float] = []
    reuse_factors: list[float] = []
    experts, n_tiles, k_blocks = code_tiles.shape[:3]
    for expert in range(experts):
        for n_tile in range(n_tiles):
            out_start = n_tile * bn
            out_count = min(bn, output_dims - out_start)
            if out_count <= 0:
                continue
            for k_block in range(k_blocks):
                tile = code_tiles[expert, n_tile, k_block, :out_count, :codewords_per_bk]
                total = int(tile.size)
                if total <= 0:
                    continue
                unique = int(np.unique(tile).size)
                unique_counts.append(float(unique))
                unique_ratios.append(float(unique / total))
                reuse_factors.append(float(total / unique) if unique else 0.0)
    return {
        "tile_count": len(unique_counts),
        "median_unique_codes": _percentile(unique_counts, 50),
        "p95_unique_codes": _percentile(unique_counts, 95),
        "median_unique_ratio": _percentile(unique_ratios, 50),
        "p95_unique_ratio": _percentile(unique_ratios, 95),
        "median_reuse_factor": _percentile(reuse_factors, 50),
        "p95_reuse_factor": _percentile(reuse_factors, 95),
    }


def _parity8(values: np.ndarray) -> np.ndarray:
    signs = np.asarray(values, dtype=np.uint16) & np.uint16(0xFF)
    parity = np.zeros(signs.shape, dtype=np.uint16)
    for bit in range(8):
        parity ^= (signs >> np.uint16(bit)) & np.uint16(1)
    return parity


def _factor_values(codes: np.ndarray, factor: str) -> np.ndarray:
    codes_u16 = np.asarray(codes, dtype=np.uint16)
    if factor == "sign_byte":
        return codes_u16 & np.uint16(0xFF)
    if factor == "sign_low_nibble":
        return codes_u16 & np.uint16(0x0F)
    if factor == "sign_high_nibble":
        return (codes_u16 >> np.uint16(4)) & np.uint16(0x0F)
    if factor == "abs_index":
        return codes_u16 >> np.uint16(8)
    if factor == "parity":
        return _parity8(codes_u16)
    raise ValueError(f"unsupported E8P code factor {factor!r}")


def _factor_tile_summary(
    *,
    code_tiles: np.ndarray,
    output_dims: int,
    bn: int,
    codewords_per_bk: int,
    factor: str,
) -> dict[str, Any]:
    unique_counts: list[float] = []
    unique_ratios: list[float] = []
    reuse_factors: list[float] = []
    experts, n_tiles, k_blocks = code_tiles.shape[:3]
    for expert in range(experts):
        for n_tile in range(n_tiles):
            out_start = n_tile * bn
            out_count = min(bn, output_dims - out_start)
            if out_count <= 0:
                continue
            for k_block in range(k_blocks):
                tile = code_tiles[expert, n_tile, k_block, :out_count, :codewords_per_bk]
                factor_tile = _factor_values(tile, factor)
                total = int(factor_tile.size)
                if total <= 0:
                    continue
                unique = int(np.unique(factor_tile).size)
                unique_counts.append(float(unique))
                unique_ratios.append(float(unique / total))
                reuse_factors.append(float(total / unique) if unique else 0.0)
    return {
        "tile_count": len(unique_counts),
        "median_unique_values": _percentile(unique_counts, 50),
        "p95_unique_values": _percentile(unique_counts, 95),
        "median_unique_ratio": _percentile(unique_ratios, 50),
        "p95_unique_ratio": _percentile(unique_ratios, 95),
        "median_reuse_factor": _percentile(reuse_factors, 50),
        "p95_reuse_factor": _percentile(reuse_factors, 95),
    }


def _factor_codeword_position_summary(
    *,
    code_tiles: np.ndarray,
    output_dims: int,
    bn: int,
    codewords_per_bk: int,
    factor: str,
) -> dict[str, Any]:
    unique_counts: list[float] = []
    unique_ratios: list[float] = []
    reuse_factors: list[float] = []
    experts, n_tiles, k_blocks = code_tiles.shape[:3]
    for expert in range(experts):
        for n_tile in range(n_tiles):
            out_start = n_tile * bn
            out_count = min(bn, output_dims - out_start)
            if out_count <= 0:
                continue
            for k_block in range(k_blocks):
                tile = code_tiles[expert, n_tile, k_block, :out_count, :codewords_per_bk]
                factor_tile = _factor_values(tile, factor)
                for codeword_position in range(codewords_per_bk):
                    values = factor_tile[:, codeword_position]
                    total = int(values.size)
                    if total <= 0:
                        continue
                    unique = int(np.unique(values).size)
                    unique_counts.append(float(unique))
                    unique_ratios.append(float(unique / total))
                    reuse_factors.append(float(total / unique) if unique else 0.0)
    return {
        "position_count": len(unique_counts),
        "median_unique_values_per_position": _percentile(unique_counts, 50),
        "p95_unique_values_per_position": _percentile(unique_counts, 95),
        "median_unique_ratio_per_position": _percentile(unique_ratios, 50),
        "p95_unique_ratio_per_position": _percentile(unique_ratios, 95),
        "median_reuse_factor_per_position": _percentile(reuse_factors, 50),
        "p95_reuse_factor_per_position": _percentile(reuse_factors, 95),
    }


def _factor_expert_kblock_summary(
    *,
    codes: np.ndarray,
    codewords_per_bk: int,
    factor: str,
) -> dict[str, Any]:
    unique_counts: list[float] = []
    unique_ratios: list[float] = []
    reuse_factors: list[float] = []
    codes_np = np.asarray(codes, dtype=np.uint16)
    experts, _output_dims, codewords = codes_np.shape
    for expert in range(experts):
        for k_start in range(0, codewords, codewords_per_bk):
            k_stop = min(k_start + codewords_per_bk, codewords)
            block = codes_np[expert, :, k_start:k_stop]
            factor_block = _factor_values(block, factor)
            total = int(factor_block.size)
            if total <= 0:
                continue
            unique = int(np.unique(factor_block).size)
            unique_counts.append(float(unique))
            unique_ratios.append(float(unique / total))
            reuse_factors.append(float(total / unique) if unique else 0.0)
    return {
        "expert_kblock_count": len(unique_counts),
        "median_unique_values_per_expert_kblock": _percentile(unique_counts, 50),
        "p95_unique_values_per_expert_kblock": _percentile(unique_counts, 95),
        "median_unique_ratio_per_expert_kblock": _percentile(unique_ratios, 50),
        "p95_unique_ratio_per_expert_kblock": _percentile(unique_ratios, 95),
        "median_reuse_factor_per_expert_kblock": _percentile(reuse_factors, 50),
        "p95_reuse_factor_per_expert_kblock": _percentile(reuse_factors, 95),
    }


def _byte_factor_summary(
    *,
    codes: np.ndarray,
    code_tiles: np.ndarray,
    output_dims: int,
    bn: int,
    codewords_per_bk: int,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    total = int(np.asarray(codes).size)
    for factor in ("sign_byte", "abs_index", "parity"):
        values = _factor_values(codes, factor)
        unique = int(np.unique(values).size)
        summaries[factor] = {
            "global_unique_values": unique,
            "global_unique_ratio": _round_float(unique / total) if total else None,
            "global_reuse_factor": _round_float(total / unique) if unique else None,
            "tile_summary": _factor_tile_summary(
                code_tiles=code_tiles,
                output_dims=output_dims,
                bn=bn,
                codewords_per_bk=codewords_per_bk,
                factor=factor,
            ),
        }
    return summaries


def _sign_nibble_factor_summary(
    *,
    codes: np.ndarray,
    code_tiles: np.ndarray,
    output_dims: int,
    bn: int,
    codewords_per_bk: int,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    total = int(np.asarray(codes).size)
    for factor in ("sign_low_nibble", "sign_high_nibble", "abs_index", "parity"):
        values = _factor_values(codes, factor)
        unique = int(np.unique(values).size)
        summaries[factor] = {
            "global_unique_values": unique,
            "global_unique_ratio": _round_float(unique / total) if total else None,
            "global_reuse_factor": _round_float(total / unique) if unique else None,
            "tile_summary": _factor_tile_summary(
                code_tiles=code_tiles,
                output_dims=output_dims,
                bn=bn,
                codewords_per_bk=codewords_per_bk,
                factor=factor,
            ),
        }
    return summaries


def _codeword_position_factor_summary(
    *,
    codes: np.ndarray,
    code_tiles: np.ndarray,
    output_dims: int,
    bn: int,
    codewords_per_bk: int,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    total = int(np.asarray(codes).size)
    for factor in ("sign_low_nibble", "sign_high_nibble", "sign_byte", "abs_index", "parity"):
        values = _factor_values(codes, factor)
        unique = int(np.unique(values).size)
        summaries[factor] = {
            "global_unique_values": unique,
            "global_unique_ratio": _round_float(unique / total) if total else None,
            "global_reuse_factor": _round_float(total / unique) if unique else None,
            "position_summary": _factor_codeword_position_summary(
                code_tiles=code_tiles,
                output_dims=output_dims,
                bn=bn,
                codewords_per_bk=codewords_per_bk,
                factor=factor,
            ),
        }
    return summaries


def _expert_kblock_factor_summary(
    *,
    codes: np.ndarray,
    codewords_per_bk: int,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    total = int(np.asarray(codes).size)
    for factor in ("sign_low_nibble", "sign_high_nibble", "sign_byte", "abs_index", "parity"):
        values = _factor_values(codes, factor)
        unique = int(np.unique(values).size)
        summaries[factor] = {
            "global_unique_values": unique,
            "global_unique_ratio": _round_float(unique / total) if total else None,
            "global_reuse_factor": _round_float(total / unique) if unique else None,
            "expert_kblock_summary": _factor_expert_kblock_summary(
                codes=codes,
                codewords_per_bk=codewords_per_bk,
                factor=factor,
            ),
        }
    return summaries


def _factor_recommendation(factors: dict[str, dict[str, Any]]) -> str:
    sign_ratio = factors["sign_byte"]["tile_summary"]["median_unique_ratio"]
    abs_ratio = factors["abs_index"]["tile_summary"]["median_unique_ratio"]
    if sign_ratio is None or abs_ratio is None:
        return "insufficient_tile_samples"
    if float(sign_ratio) <= 0.60 and float(abs_ratio) <= 0.60:
        return "split_byte_layout_worth_microbench"
    if float(abs_ratio) <= 0.60:
        return "abs_index_layout_worth_microbench"
    return "byte_factor_reuse_too_low_for_layout_probe"


def _sign_nibble_recommendation(factors: dict[str, dict[str, Any]]) -> str:
    low_ratio = factors["sign_low_nibble"]["tile_summary"]["median_unique_ratio"]
    high_ratio = factors["sign_high_nibble"]["tile_summary"]["median_unique_ratio"]
    abs_ratio = factors["abs_index"]["tile_summary"]["median_unique_ratio"]
    if low_ratio is None or high_ratio is None or abs_ratio is None:
        return "insufficient_tile_samples"
    if float(low_ratio) <= 0.05 and float(high_ratio) <= 0.05 and float(abs_ratio) <= 0.60:
        return "sign_nibble_abs_index_layout_worth_microbench"
    if float(low_ratio) <= 0.05 and float(high_ratio) <= 0.05:
        return "sign_nibble_layout_worth_microbench"
    return "sign_nibble_reuse_too_low_for_layout_probe"


def _codeword_position_recommendation(factors: dict[str, dict[str, Any]]) -> str:
    low_ratio = factors["sign_low_nibble"]["position_summary"][
        "median_unique_ratio_per_position"
    ]
    high_ratio = factors["sign_high_nibble"]["position_summary"][
        "median_unique_ratio_per_position"
    ]
    abs_ratio = factors["abs_index"]["position_summary"]["median_unique_ratio_per_position"]
    if low_ratio is None or high_ratio is None or abs_ratio is None:
        return "insufficient_codeword_position_samples"
    if float(low_ratio) <= 0.25 and float(high_ratio) <= 0.25 and float(abs_ratio) <= 0.25:
        return "codeword_position_sign_nibble_abs_index_layout_worth_microbench"
    if float(abs_ratio) <= 0.25:
        return "codeword_position_abs_index_layout_worth_microbench"
    return "codeword_position_reuse_too_low_for_layout_probe"


def _expert_kblock_recommendation(factors: dict[str, dict[str, Any]]) -> str:
    sign_ratio = factors["sign_byte"]["expert_kblock_summary"][
        "median_unique_ratio_per_expert_kblock"
    ]
    abs_ratio = factors["abs_index"]["expert_kblock_summary"][
        "median_unique_ratio_per_expert_kblock"
    ]
    if sign_ratio is None or abs_ratio is None:
        return "insufficient_expert_kblock_samples"
    if float(sign_ratio) <= 0.05 and float(abs_ratio) <= 0.05:
        return "expert_kblock_factor_reuse_worth_kernel_family_probe"
    if float(abs_ratio) <= 0.05:
        return "expert_kblock_abs_index_reuse_worth_kernel_family_probe"
    return "expert_kblock_reuse_too_low_for_layout_probe"


def _expert_reuse_rows(codes: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expert in range(codes.shape[0]):
        expert_codes = codes[expert]
        total = int(expert_codes.size)
        unique = int(np.unique(expert_codes).size)
        rows.append(
            {
                "expert": int(expert),
                "total_codewords": total,
                "unique_codes": unique,
                "unique_ratio": _round_float(unique / total) if total else None,
                "reuse_factor": _round_float(total / unique) if unique else None,
            }
        )
    return rows


def analyze_projection_code_reuse(
    *,
    projection: ArtifactProjection,
    codes: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    bn: int = 64,
    bk: int = 64,
) -> dict[str, Any]:
    codes_np = np.asarray(codes)
    scales_np = np.asarray(scales)
    packed = pack_e8p_rhs_tiles(codes_np, scales_np, group_size=group_size, bn=bn, bk=bk)
    total_codewords = int(codes_np.size)
    unique_codes = int(np.unique(codes_np).size)
    unique_ratio = unique_codes / total_codewords if total_codewords else 0.0
    reuse_factor = total_codewords / unique_codes if unique_codes else 0.0
    tile_summary = _tile_reuse_summary(
        code_tiles=packed.code_tiles,
        output_dims=packed.layout.output_dims,
        bn=packed.layout.bn,
        codewords_per_bk=packed.layout.codewords_per_bk,
    )
    recommendation = _reuse_recommendation(
        global_reuse_factor=reuse_factor,
        median_tile_unique_ratio=tile_summary["median_unique_ratio"],
    )
    byte_factor_summary = _byte_factor_summary(
        codes=codes_np,
        code_tiles=packed.code_tiles,
        output_dims=packed.layout.output_dims,
        bn=packed.layout.bn,
        codewords_per_bk=packed.layout.codewords_per_bk,
    )
    sign_nibble_factor_summary = _sign_nibble_factor_summary(
        codes=codes_np,
        code_tiles=packed.code_tiles,
        output_dims=packed.layout.output_dims,
        bn=packed.layout.bn,
        codewords_per_bk=packed.layout.codewords_per_bk,
    )
    codeword_position_factor_summary = _codeword_position_factor_summary(
        codes=codes_np,
        code_tiles=packed.code_tiles,
        output_dims=packed.layout.output_dims,
        bn=packed.layout.bn,
        codewords_per_bk=packed.layout.codewords_per_bk,
    )
    expert_kblock_factor_summary = _expert_kblock_factor_summary(
        codes=codes_np,
        codewords_per_bk=packed.layout.codewords_per_bk,
    )
    return {
        "projection": projection,
        "layout": asdict(packed.layout),
        "total_codewords": total_codewords,
        "unique_codes": unique_codes,
        "global_unique_ratio": _round_float(unique_ratio),
        "global_reuse_factor": _round_float(reuse_factor),
        "tile_summary": tile_summary,
        "byte_factor_summary": byte_factor_summary,
        "sign_nibble_factor_summary": sign_nibble_factor_summary,
        "codeword_position_factor_summary": codeword_position_factor_summary,
        "expert_kblock_factor_summary": expert_kblock_factor_summary,
        "expert_summaries": _expert_reuse_rows(codes_np),
        "recommendation": recommendation,
        "factor_recommendation": _factor_recommendation(byte_factor_summary),
        "sign_nibble_recommendation": _sign_nibble_recommendation(sign_nibble_factor_summary),
        "codeword_position_recommendation": _codeword_position_recommendation(
            codeword_position_factor_summary
        ),
        "expert_kblock_recommendation": _expert_kblock_recommendation(
            expert_kblock_factor_summary
        ),
    }


def _load_projection(
    *,
    artifact_dir: Path,
    layer_index: int,
    projection: ArtifactProjection,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    shard = artifact_dir / f"layer-{layer_index:05d}-{projection}.safetensors"
    prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
    layer = load_quantized_vq_switch_linear(shard, prefix)
    if layer.code_bits != 16:
        raise ValueError(f"{shard} is code_bits={layer.code_bits}; E8P code reuse analysis requires 16-bit codes")
    metadata = {
        "artifact_shard": str(shard),
        "artifact_prefix": prefix,
        "input_dims": int(layer.input_dims),
        "output_dims": int(layer.output_dims),
        "num_experts": int(layer.num_experts),
        "group_size": int(layer.group_size),
        "code_bits": int(layer.code_bits),
    }
    return np.array(layer.codes), np.array(layer.scales), int(layer.group_size), metadata


def _overall_recommendation(rows: list[dict[str, Any]]) -> str:
    recommendations = {str(row["recommendation"]) for row in rows}
    if not recommendations:
        return "no_projection_rows"
    if recommendations == {"low_code_reuse_do_not_prioritize_codebook_cache"}:
        return "low_code_reuse_do_not_prioritize_codebook_cache"
    if "code_reuse_promising_for_codebook_cache_probe" in recommendations:
        return "code_reuse_promising_for_codebook_cache_probe"
    return "mixed_code_reuse_requires_kernel_microbench"


def _variant_rows(benchmark_analysis: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    return [
        row
        for row in benchmark_analysis.get("comparisons", [])
        if row.get("candidate_variant") == variant
    ]


def _matching_variant_row(
    *,
    rows: list[dict[str, Any]],
    tokens: int,
    projection: str,
) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("tokens", -1)) == tokens and str(row.get("projection")) == projection:
            return row
    return None


def _observed_variant(
    row: dict[str, Any],
    *,
    steel_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed = {
        "variant": row["candidate_variant"],
        "tokens": int(row["tokens"]),
        "projection": str(row["projection"]),
        "ms_per_iter": row["candidate_ms_per_iter"],
        "ratio_to_q2": row["ratio_to_q2"],
    }
    if steel_row is not None:
        observed["steel_ms_per_iter"] = steel_row["candidate_ms_per_iter"]
        observed["ratio_to_steel"] = _round_float(
            float(row["candidate_ms_per_iter"]) / float(steel_row["candidate_ms_per_iter"])
        )
    return observed


def _best_factor_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for factor in ("sign_byte", "abs_index", "parity"):
        candidates: list[dict[str, Any]] = []
        for row in rows:
            factor_summary = row.get("byte_factor_summary", {}).get(factor, {})
            tile_summary = factor_summary.get("tile_summary", {})
            if tile_summary.get("median_unique_ratio") is None:
                continue
            candidates.append(
                {
                    "projection": row.get("projection"),
                    "median_unique_ratio": tile_summary.get("median_unique_ratio"),
                    "median_reuse_factor": tile_summary.get("median_reuse_factor"),
                }
            )
        if candidates:
            best[factor] = min(candidates, key=lambda item: float(item["median_unique_ratio"]))
    return best


def _best_sign_nibble_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for factor in ("sign_low_nibble", "sign_high_nibble", "abs_index", "parity"):
        candidates: list[dict[str, Any]] = []
        for row in rows:
            factor_summary = row.get("sign_nibble_factor_summary", {}).get(factor, {})
            tile_summary = factor_summary.get("tile_summary", {})
            if tile_summary.get("median_unique_ratio") is None:
                continue
            candidates.append(
                {
                    "projection": row.get("projection"),
                    "median_unique_ratio": tile_summary.get("median_unique_ratio"),
                    "median_reuse_factor": tile_summary.get("median_reuse_factor"),
                }
            )
        if candidates:
            best[factor] = min(candidates, key=lambda item: float(item["median_unique_ratio"]))
    return best


def _best_codeword_position_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for factor in ("sign_low_nibble", "sign_high_nibble", "sign_byte", "abs_index", "parity"):
        candidates: list[dict[str, Any]] = []
        for row in rows:
            factor_summary = row.get("codeword_position_factor_summary", {}).get(factor, {})
            position_summary = factor_summary.get("position_summary", {})
            ratio = position_summary.get("median_unique_ratio_per_position")
            if ratio is None:
                continue
            candidates.append(
                {
                    "projection": row.get("projection"),
                    "median_unique_ratio_per_position": ratio,
                    "median_reuse_factor_per_position": position_summary.get(
                        "median_reuse_factor_per_position"
                    ),
                }
            )
        if candidates:
            best[factor] = min(
                candidates,
                key=lambda item: float(item["median_unique_ratio_per_position"]),
            )
    return best


def _best_expert_kblock_reuse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for factor in ("sign_low_nibble", "sign_high_nibble", "sign_byte", "abs_index", "parity"):
        candidates: list[dict[str, Any]] = []
        for row in rows:
            factor_summary = row.get("expert_kblock_factor_summary", {}).get(factor, {})
            expert_kblock_summary = factor_summary.get("expert_kblock_summary", {})
            ratio = expert_kblock_summary.get("median_unique_ratio_per_expert_kblock")
            if ratio is None:
                continue
            candidates.append(
                {
                    "projection": row.get("projection"),
                    "median_unique_ratio_per_expert_kblock": ratio,
                    "median_reuse_factor_per_expert_kblock": expert_kblock_summary.get(
                        "median_reuse_factor_per_expert_kblock"
                    ),
                    "median_unique_values_per_expert_kblock": expert_kblock_summary.get(
                        "median_unique_values_per_expert_kblock"
                    ),
                }
            )
        if candidates:
            best[factor] = min(
                candidates,
                key=lambda item: float(item["median_unique_ratio_per_expert_kblock"]),
            )
    return best


def _sign_nibble_schedule_requirements(reuse_summary: dict[str, Any]) -> dict[str, Any]:
    rows = list(reuse_summary.get("projections", []))
    recommendations = {str(row.get("sign_nibble_recommendation")) for row in rows}
    signal_present = "sign_nibble_abs_index_layout_worth_microbench" in recommendations
    return {
        "decision": (
            "microbench_sign_nibble_abs_index_layout"
            if signal_present
            else "sign_nibble_layout_not_established"
        ),
        "factor_signal": (
            "sign_nibble_abs_index_reuse_present"
            if signal_present
            else "sign_nibble_abs_index_reuse_not_established"
        ),
        "best_sign_nibble_reuse": _best_sign_nibble_reuse(rows),
        "required_layout_features": (
            [
                "split_sign_byte_into_low_high_nibble_masks",
                "keep_abs_index_byte_factor",
                "preserve_compressed_rhs_storage",
                "preserve_codeword_scale_slots",
            ]
            if signal_present
            else []
        ),
        "required_kernel_features": (
            [
                "decode_nibble_sign_masks_without_per_fragment_sign_byte_reconstruction",
                "reuse_abs_index_rows_or_sign_masks_outside_staged_b",
                "avoid_lane_local_barrier_fragment_buffer",
                "avoid_decoded_b_threadgroup_staging",
            ]
            if signal_present
            else []
        ),
        "rejected_next_steps": [
            "whole_sign_byte_factor_reuse_unchanged",
            "lane_local_shared_n_factor_reuse",
            "current_non_staged_shared_decode_factor_reuse",
            "same_staged_b_factor_reuse_tiled",
            "same_staged_b_split_byte_tiled",
            "same_staged_b_packed_rhs_tiled",
        ],
        "next_kernel_requirement": (
            "Prototype a sign-nibble/abs-index RHS layout or source-structure guardrail "
            "before another shared-n/shared-decode/staged-B timing run."
            if signal_present
            else "Establish sign-nibble factor locality before selecting this layout."
        ),
    }


def _codeword_position_schedule_requirements(reuse_summary: dict[str, Any]) -> dict[str, Any]:
    rows = list(reuse_summary.get("projections", []))
    recommendations = {str(row.get("codeword_position_recommendation")) for row in rows}
    signal_present = bool(
        recommendations
        & {
            "codeword_position_sign_nibble_abs_index_layout_worth_microbench",
            "codeword_position_abs_index_layout_worth_microbench",
        }
    )
    return {
        "decision": (
            "microbench_codeword_position_factor_layout"
            if signal_present
            else "codeword_position_layout_not_established"
        ),
        "factor_signal": (
            "codeword_position_factor_reuse_present"
            if signal_present
            else "codeword_position_factor_reuse_not_established"
        ),
        "best_codeword_position_reuse": _best_codeword_position_reuse(rows),
        "required_layout_features": (
            [
                "group_rhs_by_codeword_slot_inside_bk64",
                "reuse_abs_index_rows_across_output_columns",
                "reuse_sign_masks_across_output_columns",
                "preserve_compressed_rhs_storage",
                "preserve_codeword_scale_slots",
            ]
            if signal_present
            else []
        ),
        "required_kernel_features": (
            [
                "decode_factor_rows_at_codeword_position_scope",
                "avoid_per_fragment_sign_plane_abs_scale_reconstruction",
                "avoid_per_fragment_nibble_abs_scale_reconstruction",
                "avoid_decoded_b_threadgroup_staging",
                "avoid_lane_local_barrier_fragment_buffer",
            ]
            if signal_present
            else []
        ),
        "rejected_next_steps": [
            "per_fragment_sign_plane_tensorops_schedule",
            "per_fragment_nibble_tensorops_schedule",
            "per_fragment_micro_lut_tensorops_schedule",
            "lane_local_shared_n_factor_reuse",
            "current_non_staged_shared_decode_factor_reuse",
            "same_staged_b_factor_reuse_tiled",
            "same_staged_b_split_byte_tiled",
            "same_staged_b_packed_rhs_tiled",
        ],
        "next_kernel_requirement": (
            "Prototype only a layout/kernel that reuses factor decode at fixed "
            "codeword positions across output columns before filling TensorOps B fragments."
            if signal_present
            else "Establish codeword-position factor locality before selecting this layout."
        ),
    }


def _expert_kblock_schedule_requirements(reuse_summary: dict[str, Any]) -> dict[str, Any]:
    rows = list(reuse_summary.get("projections", []))
    recommendations = {str(row.get("expert_kblock_recommendation")) for row in rows}
    signal_present = bool(
        recommendations
        & {
            "expert_kblock_factor_reuse_worth_kernel_family_probe",
            "expert_kblock_abs_index_reuse_worth_kernel_family_probe",
        }
    )
    return {
        "decision": (
            "probe_expert_kblock_factor_decode_reuse"
            if signal_present
            else "expert_kblock_layout_not_established"
        ),
        "factor_signal": (
            "expert_kblock_factor_reuse_present"
            if signal_present
            else "expert_kblock_factor_reuse_not_established"
        ),
        "best_expert_kblock_reuse": _best_expert_kblock_reuse(rows),
        "required_layout_features": (
            [
                "preserve_compressed_rhs_storage",
                "preserve_codeword_scale_slots",
                "keep_factor_maps_addressable_by_expert_kblock",
                "avoid_fixed_codeword_position_dependency",
            ]
            if signal_present
            else []
        ),
        "required_kernel_features": (
            [
                "reuse_factor_decode_across_output_tiles_per_expert_kblock",
                "avoid_per_fragment_factor_reconstruction",
                "avoid_lane_local_barrier_fragment_buffer",
                "account_for_output_column_scale_groups",
            ]
            if signal_present
            else []
        ),
        "rejected_next_steps": [
            "fixed_codeword_position_output_column_layout",
            "per_fragment_sign_plane_tensorops_schedule",
            "per_fragment_nibble_tensorops_schedule",
            "per_fragment_micro_lut_tensorops_schedule",
            "lane_local_shared_n_factor_reuse",
            "current_non_staged_shared_decode_factor_reuse",
            "same_staged_b_factor_reuse_tiled",
            "same_staged_b_split_byte_tiled",
            "same_staged_b_packed_rhs_tiled",
        ],
        "next_kernel_requirement": (
            "Prototype only a kernel family that reuses E8P sign/abs factor decode across "
            "output tiles within the same expert and K block; do not depend on fixed "
            "codeword-position abs-index locality."
            if signal_present
            else "Establish expert/K-block factor locality before selecting this scope."
        ),
    }


def _split_byte_schedule_requirements(
    *,
    reuse_summary: dict[str, Any],
    benchmark_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(reuse_summary.get("projections", []))
    factor_recommendations = {str(row.get("factor_recommendation")) for row in rows}
    requirements: dict[str, Any] = {
        "factor_signal": (
            "split_byte_factor_reuse_present"
            if "split_byte_layout_worth_microbench" in factor_recommendations
            else "split_byte_factor_reuse_not_established"
        ),
        "best_factor_reuse": _best_factor_reuse(rows),
        "required_schedule_features": [],
        "rejected_next_steps": [
            "whole_code_cache",
            "direct_reduce_scalar",
            "inline_b_per_simdgroup_decode",
            "dense_predecoded_fp16_rhs",
            "simple_steel_tile_size_variant",
        ],
    }
    if benchmark_analysis is None:
        requirements.update(
            {
                "decision": "microbench_split_byte_layout_before_kernel_family_claim",
                "reason": "split-byte factor locality exists, but no split-byte timing verdict was supplied",
                "required_schedule_features": [
                    "benchmark_split_byte_tiled_against_steel_q2",
                    "preserve_compressed_rhs_storage",
                ],
            }
        )
        return requirements

    split_rows = _variant_rows(benchmark_analysis, "nax_e8p_split_byte_rhs_sorted_tiled_raw")
    steel_rows = _variant_rows(benchmark_analysis, "nax_e8p_fp16_sorted_steel_raw")
    packed_rows = _variant_rows(benchmark_analysis, "nax_e8p_packed_rhs_sorted_tiled_raw")
    factor_native_rows = _variant_rows(
        benchmark_analysis,
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_native_raw",
    )
    factor_tiled_rows = _variant_rows(
        benchmark_analysis,
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_raw",
    )
    factor_shared_decode_rows = _variant_rows(
        benchmark_analysis,
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_raw",
    )
    factor_shared_n_rows = _variant_rows(
        benchmark_analysis,
        "nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_raw",
    )
    failed_targets = not bool(benchmark_analysis.get("all_parity_pass")) or not bool(
        benchmark_analysis.get("all_lane_s_pass")
    )
    if factor_shared_n_rows:
        shared_n_row = min(factor_shared_n_rows, key=lambda row: float(row["ratio_to_q2"]))
        tokens = int(shared_n_row["tokens"])
        projection = str(shared_n_row["projection"])
        steel_row = _matching_variant_row(rows=steel_rows, tokens=tokens, projection=projection)
        native_row = _matching_variant_row(
            rows=factor_native_rows,
            tokens=tokens,
            projection=projection,
        )
        tiled_row = _matching_variant_row(
            rows=factor_tiled_rows,
            tokens=tokens,
            projection=projection,
        )
        shared_decode_row = _matching_variant_row(
            rows=factor_shared_decode_rows,
            tokens=tokens,
            projection=projection,
        )
        observed_shared_n = _observed_variant(shared_n_row, steel_row=steel_row)
        observed_shared_decode = (
            _observed_variant(shared_decode_row, steel_row=steel_row)
            if shared_decode_row is not None
            else None
        )
        observed_native = (
            _observed_variant(native_row, steel_row=steel_row) if native_row is not None else None
        )
        observed_tiled = (
            _observed_variant(tiled_row, steel_row=steel_row) if tiled_row is not None else None
        )
        if shared_decode_row is not None:
            observed_shared_n["shared_decode_ms_per_iter"] = shared_decode_row[
                "candidate_ms_per_iter"
            ]
            observed_shared_n["ratio_to_shared_decode"] = _round_float(
                float(shared_n_row["candidate_ms_per_iter"])
                / float(shared_decode_row["candidate_ms_per_iter"])
            )
        if native_row is not None:
            observed_shared_n["factor_reuse_native_ms_per_iter"] = native_row[
                "candidate_ms_per_iter"
            ]
            observed_shared_n["ratio_to_factor_reuse_native"] = _round_float(
                float(shared_n_row["candidate_ms_per_iter"])
                / float(native_row["candidate_ms_per_iter"])
            )
        if tiled_row is not None:
            observed_shared_n["factor_reuse_tiled_ms_per_iter"] = tiled_row[
                "candidate_ms_per_iter"
            ]
            observed_shared_n["ratio_to_factor_reuse_tiled"] = _round_float(
                float(shared_n_row["candidate_ms_per_iter"])
                / float(tiled_row["candidate_ms_per_iter"])
            )
        slower_than_steel = float(observed_shared_n.get("ratio_to_steel", 0.0)) > 1.0
        slower_than_native = float(
            observed_shared_n.get("ratio_to_factor_reuse_native", 0.0)
        ) > 1.0
        slower_than_tiled = float(observed_shared_n.get("ratio_to_factor_reuse_tiled", 0.0)) > 1.0
        if slower_than_steel and slower_than_native and slower_than_tiled and failed_targets:
            requirements.update(
                {
                    "decision": "reject_shared_n_factor_reuse",
                    "reason": (
                        "lane-local shared-n factor decode improves the rejected shared-decode "
                        "implementation, but the benchmarked path still loses to Steel, "
                        "scalar factor-reuse, staged factor-reuse, and q2"
                    ),
                    "observed_factor_reuse_shared_n": observed_shared_n,
                    "observed_factor_reuse_shared_decode": observed_shared_decode,
                    "observed_factor_reuse_native": observed_native,
                    "observed_factor_reuse_tiled": observed_tiled,
                    "required_schedule_features": [
                        "avoid_lane_local_barrier_fragment_buffer",
                        "broader_expert_output_factor_decode_reuse",
                        "avoid_per_fragment_factor_reconstruction",
                        "preserve_compressed_rhs_storage",
                        "change_rhs_layout_or_kernel_family",
                    ],
                    "rejected_next_steps": [
                        "lane_local_shared_n_factor_reuse",
                        "air_shape_sweep_of_current_shared_n_factor_reuse",
                        "current_non_staged_shared_decode_factor_reuse",
                        "air_shape_sweep_of_current_shared_decode_factor_reuse",
                        "scalar_direct_factor_reuse_traversal",
                        "same_staged_b_factor_reuse_tiled",
                        "same_staged_b_split_byte_tiled",
                        "whole_code_cache",
                        "direct_reduce_scalar",
                        "inline_b_per_simdgroup_decode",
                        "dense_predecoded_fp16_rhs",
                        "simple_steel_tile_size_variant",
                    ],
                    "next_kernel_requirement": (
                        "Do not benchmark shared-n or shared-decode unchanged. Prototype a "
                        "schedule that reuses split-byte factor decode without lane-local "
                        "fragment buffers/barriers, or switch RHS layout/kernel family again."
                    ),
                }
            )
            return requirements
    if factor_shared_decode_rows:
        shared_row = min(factor_shared_decode_rows, key=lambda row: float(row["ratio_to_q2"]))
        tokens = int(shared_row["tokens"])
        projection = str(shared_row["projection"])
        steel_row = _matching_variant_row(rows=steel_rows, tokens=tokens, projection=projection)
        native_row = _matching_variant_row(
            rows=factor_native_rows,
            tokens=tokens,
            projection=projection,
        )
        tiled_row = _matching_variant_row(
            rows=factor_tiled_rows,
            tokens=tokens,
            projection=projection,
        )
        observed_shared = _observed_variant(shared_row, steel_row=steel_row)
        observed_native = (
            _observed_variant(native_row, steel_row=steel_row) if native_row is not None else None
        )
        observed_tiled = (
            _observed_variant(tiled_row, steel_row=steel_row) if tiled_row is not None else None
        )
        if native_row is not None:
            observed_shared["factor_reuse_native_ms_per_iter"] = native_row[
                "candidate_ms_per_iter"
            ]
            observed_shared["ratio_to_factor_reuse_native"] = _round_float(
                float(shared_row["candidate_ms_per_iter"])
                / float(native_row["candidate_ms_per_iter"])
            )
        if tiled_row is not None:
            observed_shared["factor_reuse_tiled_ms_per_iter"] = tiled_row[
                "candidate_ms_per_iter"
            ]
            observed_shared["ratio_to_factor_reuse_tiled"] = _round_float(
                float(shared_row["candidate_ms_per_iter"])
                / float(tiled_row["candidate_ms_per_iter"])
            )
        slower_than_steel = float(observed_shared.get("ratio_to_steel", 0.0)) > 1.0
        if slower_than_steel and failed_targets:
            requirements.update(
                {
                    "decision": "reject_current_shared_decode_factor_reuse",
                    "reason": (
                        "the benchmarked non-staged TensorOps shared-decode path avoids "
                        "decoded-B threadgroup staging, but reconstructs split-byte factors "
                        "inside B fragments expensively enough to lose to Steel, scalar "
                        "factor-reuse, staged factor-reuse, and q2"
                    ),
                    "observed_factor_reuse_shared_decode": observed_shared,
                    "observed_factor_reuse_native": observed_native,
                    "observed_factor_reuse_tiled": observed_tiled,
                    "required_schedule_features": [
                        "broader_expert_output_factor_decode_reuse",
                        "avoid_per_fragment_factor_reconstruction",
                        "preserve_compressed_rhs_storage",
                        "change_rhs_layout_or_kernel_family",
                    ],
                    "rejected_next_steps": [
                        "current_non_staged_shared_decode_factor_reuse",
                        "air_shape_sweep_of_current_shared_decode_factor_reuse",
                        "scalar_direct_factor_reuse_traversal",
                        "same_staged_b_factor_reuse_tiled",
                        "same_staged_b_split_byte_tiled",
                        "whole_code_cache",
                        "direct_reduce_scalar",
                        "inline_b_per_simdgroup_decode",
                        "dense_predecoded_fp16_rhs",
                        "simple_steel_tile_size_variant",
                    ],
                    "next_kernel_requirement": (
                        "Do not benchmark the current shared-decode path unchanged. "
                        "Prototype only a schedule that reuses split-byte factor decode "
                        "across a broader expert/output scope, or switch RHS layout/kernel "
                        "family again."
                    ),
                }
            )
            return requirements
    if factor_native_rows:
        native_row = min(factor_native_rows, key=lambda row: float(row["ratio_to_q2"]))
        tokens = int(native_row["tokens"])
        projection = str(native_row["projection"])
        steel_row = _matching_variant_row(rows=steel_rows, tokens=tokens, projection=projection)
        tiled_row = _matching_variant_row(rows=factor_tiled_rows, tokens=tokens, projection=projection)
        observed_native = _observed_variant(native_row, steel_row=steel_row)
        observed_tiled = (
            _observed_variant(tiled_row, steel_row=steel_row) if tiled_row is not None else None
        )
        if observed_tiled is not None:
            observed_native["factor_reuse_tiled_ms_per_iter"] = observed_tiled["ms_per_iter"]
            observed_native["ratio_to_factor_reuse_tiled"] = _round_float(
                float(native_row["candidate_ms_per_iter"])
                / float(tiled_row["candidate_ms_per_iter"])
            )
        slower_than_steel = float(observed_native.get("ratio_to_steel", 0.0)) > 1.0
        if slower_than_steel and failed_targets:
            rejected = [
                "scalar_direct_factor_reuse_traversal",
                "same_staged_b_factor_reuse_tiled",
                "air_shape_sweep_of_scalar_factor_reuse",
                "air_shape_sweep_of_staged_b_factor_reuse",
                "same_staged_b_split_byte_tiled",
                "whole_code_cache",
                "direct_reduce_scalar",
                "inline_b_per_simdgroup_decode",
                "dense_predecoded_fp16_rhs",
                "simple_steel_tile_size_variant",
            ]
            requirements.update(
                {
                    "decision": "reject_scalar_and_staged_factor_reuse",
                    "reason": (
                        "factor-reuse split-byte locality is real, and removing decoded-B "
                        "threadgroup staging beats the staged factor-reuse control, but the "
                        "native scalar/direct traversal is still slower than Steel and far "
                        "from q2"
                    ),
                    "observed_factor_reuse_native": observed_native,
                    "observed_factor_reuse_tiled": observed_tiled,
                    "required_schedule_features": [
                        "non_staged_tensorops_shared_decode",
                        "no_scalar_direct_route_output_traversal",
                        "reuse_abs_index_rows_or_sign_masks_outside_staged_b",
                        "avoid_per_threadgroup_decoded_b_tile",
                        "preserve_compressed_rhs_storage",
                    ],
                    "rejected_next_steps": rejected,
                    "next_kernel_requirement": (
                        "Build a non-staged TensorOps schedule that shares split-byte "
                        "factor decode across routes/output tiles without scalar direct "
                        "route-output traversal or decoded-B threadgroup Ws staging."
                    ),
                }
            )
            return requirements
    if not split_rows:
        requirements.update(
            {
                "decision": "benchmark_split_byte_tiled_before_schedule_decision",
                "reason": "benchmark analysis did not include nax_e8p_split_byte_rhs_sorted_tiled_raw",
                "required_schedule_features": [
                    "benchmark_split_byte_tiled_against_steel_q2",
                    "preserve_compressed_rhs_storage",
                ],
            }
        )
        return requirements

    split_row = min(split_rows, key=lambda row: float(row["ratio_to_q2"]))
    tokens = int(split_row["tokens"])
    projection = str(split_row["projection"])
    steel_row = _matching_variant_row(rows=steel_rows, tokens=tokens, projection=projection)
    packed_row = _matching_variant_row(rows=packed_rows, tokens=tokens, projection=projection)
    observed = {
        "variant": split_row["candidate_variant"],
        "tokens": tokens,
        "projection": projection,
        "ms_per_iter": split_row["candidate_ms_per_iter"],
        "ratio_to_q2": split_row["ratio_to_q2"],
    }
    if steel_row is not None:
        observed["steel_ms_per_iter"] = steel_row["candidate_ms_per_iter"]
        observed["ratio_to_steel"] = _round_float(
            float(split_row["candidate_ms_per_iter"]) / float(steel_row["candidate_ms_per_iter"])
        )
    if packed_row is not None:
        observed["packed_tiled_ms_per_iter"] = packed_row["candidate_ms_per_iter"]
        observed["ratio_to_packed_tiled"] = _round_float(
            float(split_row["candidate_ms_per_iter"]) / float(packed_row["candidate_ms_per_iter"])
        )

    slower_than_steel = float(observed.get("ratio_to_steel", 0.0)) > 1.0
    slower_than_packed = float(observed.get("ratio_to_packed_tiled", 0.0)) > 1.0
    if slower_than_steel and slower_than_packed and failed_targets:
        decision = "reject_staged_b_split_byte_tiled"
        required_features = [
            "abs_index_row_reuse_outside_staged_b",
            "sign_mask_reuse_outside_staged_b",
            "avoid_per_threadgroup_decoded_b_tile",
            "preserve_compressed_rhs_storage",
        ]
        rejected = [
            "same_staged_b_split_byte_tiled",
            "air_shape_sweep_of_unchanged_split_byte_staged_b",
            "whole_code_cache",
            "direct_reduce_scalar",
            "inline_b_per_simdgroup_decode",
            "dense_predecoded_fp16_rhs",
            "simple_steel_tile_size_variant",
        ]
        reason = (
            "split-byte factors are locally reusable, but the benchmarked staged-B split-byte "
            "tile is slower than both Steel and whole-code packed tiled while still missing q2"
        )
    else:
        decision = "split_byte_tiled_requires_air_shape_followup"
        required_features = [
            "air_shape_benchmark_against_steel_q2",
            "preserve_compressed_rhs_storage",
        ]
        rejected = requirements["rejected_next_steps"]
        reason = "split-byte benchmark verdict is incomplete or not clearly slower than current baselines"
    requirements.update(
        {
            "decision": decision,
            "reason": reason,
            "observed_split_byte_tiled": observed,
            "required_schedule_features": required_features,
            "rejected_next_steps": rejected,
        }
    )
    return requirements


def analyze_artifact_code_reuse(
    *,
    artifact_dir: str | Path,
    layer_index: int,
    projections: list[ArtifactProjection],
    bn: int = 64,
    bk: int = 64,
    benchmark_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    rows: list[dict[str, Any]] = []
    for projection in projections:
        codes, scales, group_size, metadata = _load_projection(
            artifact_dir=artifact_root,
            layer_index=layer_index,
            projection=projection,
        )
        row = analyze_projection_code_reuse(
            projection=projection,
            codes=codes,
            scales=scales,
            group_size=group_size,
            bn=bn,
            bk=bk,
        )
        row.update(metadata)
        rows.append(row)
    summary = {
        "schema_version": 1,
        "record_type": "glm45_air_e8p_code_reuse_analysis",
        "artifact_dir": str(artifact_root),
        "layer_index": int(layer_index),
        "bn": int(bn),
        "bk": int(bk),
        "projection_count": len(rows),
        "overall_recommendation": _overall_recommendation(rows),
        "projections": rows,
    }
    if any(row.get("factor_recommendation") == "split_byte_layout_worth_microbench" for row in rows):
        summary["split_byte_schedule_requirements"] = _split_byte_schedule_requirements(
            reuse_summary=summary,
            benchmark_analysis=benchmark_analysis,
        )
    if any(
        row.get("sign_nibble_recommendation")
        == "sign_nibble_abs_index_layout_worth_microbench"
        for row in rows
    ):
        summary["sign_nibble_schedule_requirements"] = _sign_nibble_schedule_requirements(
            summary
        )
    summary["codeword_position_schedule_requirements"] = (
        _codeword_position_schedule_requirements(summary)
    )
    summary["expert_kblock_schedule_requirements"] = _expert_kblock_schedule_requirements(summary)
    return summary


def _parse_projection_args(values: list[str]) -> list[ArtifactProjection]:
    if any(value == "all" for value in values):
        return list(_ALL_PROJECTIONS)
    projections: list[ArtifactProjection] = []
    for value in values:
        if value not in _ALL_PROJECTIONS:
            raise ValueError(f"unsupported projection {value!r}")
        projections.append(value)  # type: ignore[arg-type]
    return projections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GLM-4.5-Air E8P packed RHS code reuse for Track B kernel-layout decisions."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--projection",
        action="append",
        default=["all"],
        help="Projection to analyze: gate_proj, up_proj, down_proj, or all. May be repeated.",
    )
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument(
        "--benchmark-analysis-json",
        help="Optional benchmark-analysis JSON used to turn split-byte locality into next-schedule requirements.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()
    benchmark_analysis = None
    if args.benchmark_analysis_json:
        benchmark_analysis = json.loads(Path(args.benchmark_analysis_json).read_text(encoding="utf-8"))
    summary = analyze_artifact_code_reuse(
        artifact_dir=args.artifact_dir,
        layer_index=args.layer,
        projections=_parse_projection_args(args.projection),
        bn=args.bn,
        bk=args.bk,
        benchmark_analysis=benchmark_analysis,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.append_jsonl:
        append_path = Path(args.append_jsonl)
        append_path.parent.mkdir(parents=True, exist_ok=True)
        with append_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
