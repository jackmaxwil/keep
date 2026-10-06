from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import CODEWORD_DIM


@dataclass(frozen=True)
class LDLFactor:
    l: np.ndarray
    d: np.ndarray
    regularization: float


@dataclass(frozen=True)
class BlockLDLQStats:
    group_size: int
    code_bits: int
    row_count: int
    codewords_per_row: int
    code_count: int
    changed_code_count: int
    changed_code_fraction: float
    current_hin_weighted_error: float
    hin_weighted_error: float
    hin_weighted_error_ratio: float
    sweeps: int


@dataclass(frozen=True)
class BlockLDLQReassignedWeight:
    codes: np.ndarray
    weight: np.ndarray
    stats: BlockLDLQStats


@dataclass(frozen=True)
class FullBlockLDLQStats:
    group_size: int
    code_bits: int
    row_count: int
    codewords_per_row: int
    code_count: int
    changed_code_count: int
    changed_code_fraction: float
    current_kronecker_weighted_error: float
    kronecker_weighted_error: float
    kronecker_weighted_error_ratio: float
    sweeps: int


@dataclass(frozen=True)
class FullBlockLDLQReassignedWeight:
    codes: np.ndarray
    weight: np.ndarray
    stats: FullBlockLDLQStats


@dataclass(frozen=True)
class FullYAQAHessianEntry:
    path: str
    layer: int
    projection: str
    expert: int
    sample_count: int
    h_in_dim: int
    h_out_dim: int


def activation_covariance(samples: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Return an input-side covariance / Hessian proxy from activation samples."""

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D [sample, dim], found {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("samples must contain at least one row")
    if not np.isfinite(x).all():
        raise ValueError("samples must be finite")
    h = x.T @ x
    if normalize:
        h = h / float(x.shape[0])
    return h.astype(np.float32)


def yaqa_sketch_a_hessian_factors(
    input_samples: np.ndarray,
    output_grad_samples: np.ndarray,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return YAQA Sketch-A Kronecker factors from input and output-gradient rows.

    Sketch A uses the independent-token Fisher approximation
    ``E[x^T x]`` and ``E[(dL/dy)^T (dL/dy)]`` for a linear layer. The
    output-side factor must come from an error/gradient signal, not from the
    layer output activations.
    """

    inputs = np.asarray(input_samples, dtype=np.float32)
    output_grads = np.asarray(output_grad_samples, dtype=np.float32)
    if inputs.ndim != 2:
        raise ValueError(f"input_samples must be 2D [sample, input_dim], found {inputs.shape}")
    if output_grads.ndim != 2:
        raise ValueError(
            f"output_grad_samples must be 2D [sample, output_dim], found {output_grads.shape}"
        )
    if inputs.shape[0] != output_grads.shape[0]:
        raise ValueError(
            "input_samples and output_grad_samples must have the same number of samples, "
            f"found {inputs.shape[0]} and {output_grads.shape[0]}"
        )
    if inputs.shape[0] == 0:
        raise ValueError("samples must contain at least one row")
    return (
        activation_covariance(inputs, normalize=normalize),
        activation_covariance(output_grads, normalize=normalize),
    )


def regularize_hessian(
    hessian: np.ndarray,
    *,
    damping: float = 0.01,
    min_diagonal: float = 1.0e-6,
) -> tuple[np.ndarray, float]:
    """Symmetrize and diagonally damp a Hessian-like matrix."""

    h = np.asarray(hessian, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"hessian must be square, found {h.shape}")
    if h.shape[0] == 0:
        raise ValueError("hessian must not be empty")
    if not np.isfinite(h).all():
        raise ValueError("hessian must be finite")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if min_diagonal <= 0.0:
        raise ValueError("min_diagonal must be positive")
    h = 0.5 * (h + h.T)
    mean_diag = float(np.mean(np.diag(h)))
    regularization = max(float(damping * mean_diag), float(min_diagonal))
    h = h.copy()
    h[np.diag_indices_from(h)] += regularization
    return h.astype(np.float32), regularization


def ldl_factor(hessian: np.ndarray, *, jitter: float = 1.0e-8) -> LDLFactor:
    """Compute a dense no-pivot LDL^T factorization for small/regularized SPD matrices."""

    a = np.asarray(hessian, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hessian must be square, found {a.shape}")
    n = a.shape[0]
    l = np.eye(n, dtype=np.float64)
    d = np.zeros((n,), dtype=np.float64)
    for j in range(n):
        accum = np.sum((l[j, :j] ** 2) * d[:j])
        diag = float(a[j, j] - accum)
        if diag <= jitter:
            diag = jitter
        d[j] = diag
        for i in range(j + 1, n):
            numerator = a[i, j] - np.sum(l[i, :j] * l[j, :j] * d[:j])
            l[i, j] = numerator / diag
    return LDLFactor(l=l.astype(np.float32), d=d.astype(np.float32), regularization=0.0)


def reconstruct_ldl(factor: LDLFactor) -> np.ndarray:
    l = np.asarray(factor.l, dtype=np.float64)
    d = np.asarray(factor.d, dtype=np.float64)
    if l.ndim != 2 or l.shape[0] != l.shape[1]:
        raise ValueError("LDL L factor must be square")
    if d.shape != (l.shape[0],):
        raise ValueError("LDL D factor shape must match L")
    return (l * d[None, :]) @ l.T


def save_hessian_factors(
    path: str | Path,
    *,
    h_in: np.ndarray,
    h_out: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    damping: float = 0.01,
) -> dict[str, Any]:
    """Save regularized H_in/H_out matrices and LDL factors to a safetensors file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    h_in_regularized, h_in_regularization = regularize_hessian(h_in, damping=damping)
    h_in_factor = ldl_factor(h_in_regularized)
    arrays: dict[str, mx.array] = {
        "h_in": mx.array(h_in_regularized.astype(np.float32)),
        "h_in_ldl_l": mx.array(h_in_factor.l),
        "h_in_ldl_d": mx.array(h_in_factor.d),
    }
    payload: dict[str, Any] = dict(metadata or {})
    payload.update(
        {
            "schema_version": 1,
            "method": payload.get("method", "blockldlq_hin_only"),
            "h_in_dim": int(h_in_regularized.shape[0]),
            "h_in_regularization": float(h_in_regularization),
            "h_out_available": h_out is not None,
        }
    )
    if h_out is not None:
        h_out_regularized, h_out_regularization = regularize_hessian(h_out, damping=damping)
        h_out_factor = ldl_factor(h_out_regularized)
        arrays["h_out"] = mx.array(h_out_regularized.astype(np.float32))
        arrays["h_out_ldl_l"] = mx.array(h_out_factor.l)
        arrays["h_out_ldl_d"] = mx.array(h_out_factor.d)
        payload["h_out_dim"] = int(h_out_regularized.shape[0])
        payload["h_out_regularization"] = float(h_out_regularization)
    mx.save_safetensors(str(target), arrays, metadata={key: str(value) for key, value in payload.items()})
    return payload


def load_hessian_file(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays = mx.load(str(path))
    # MLX does not expose safetensors metadata on load; callers use manifest
    # metadata for durable fields and this helper for numeric tensors.
    return {name: np.asarray(value) for name, value in arrays.items()}, {}


def _truthy_manifest_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _require_square_finite_matrix(arrays: dict[str, np.ndarray], name: str, *, path: Path) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"{path} is missing required tensor {name!r}")
    value = np.asarray(arrays[name], dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{path} tensor {name!r} must be square, found {value.shape}")
    if value.shape[0] == 0:
        raise ValueError(f"{path} tensor {name!r} must not be empty")
    if not np.isfinite(value).all():
        raise ValueError(f"{path} tensor {name!r} must be finite")
    return value


def _require_ldl_factor(arrays: dict[str, np.ndarray], prefix: str, *, dim: int, path: Path) -> None:
    l_name = f"{prefix}_ldl_l"
    d_name = f"{prefix}_ldl_d"
    for name in (l_name, d_name):
        if name not in arrays:
            raise ValueError(f"{path} is missing required tensor {name!r}")
        if not np.isfinite(np.asarray(arrays[name], dtype=np.float32)).all():
            raise ValueError(f"{path} tensor {name!r} must be finite")
    l_value = np.asarray(arrays[l_name])
    d_value = np.asarray(arrays[d_name])
    if l_value.shape != (dim, dim):
        raise ValueError(f"{path} tensor {l_name!r} must have shape ({dim}, {dim}), found {l_value.shape}")
    if d_value.shape != (dim,):
        raise ValueError(f"{path} tensor {d_name!r} must have shape ({dim},), found {d_value.shape}")


def require_full_yaqa_hessian_manifest(
    hessian_dir: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate that a Hessian manifest is eligible for full YAQA/BlockLDLQ.

    The existing single-host fallback is intentionally named
    ``blockldlq_hin_only``. Full YAQA must be stronger than a manifest label:
    every selected file must contain both H_in and H_out matrices plus their LDL
    factors before a materializer can claim output-error-aware calibration.
    """

    root = Path(hessian_dir)
    payload = (
        json.loads((root / "kronecker-hessian-manifest.json").read_text(encoding="utf-8"))
        if manifest is None
        else manifest
    )
    method = payload.get("method", {})
    if method.get("kind") != "full_yaqa":
        raise ValueError(f"hessian manifest method.kind must be 'full_yaqa', found {method.get('kind')!r}")
    if not _truthy_manifest_value(method.get("full_yaqa")):
        raise ValueError("hessian manifest method.full_yaqa must be true")
    if not _truthy_manifest_value(method.get("h_out_collected")):
        raise ValueError("hessian manifest method.h_out_collected must be true")

    selected_entries = list(payload.get("entries", []) if entries is None else entries)
    if not selected_entries:
        raise ValueError("hessian manifest must contain at least one full YAQA entry")

    summaries: list[FullYAQAHessianEntry] = []
    for entry in selected_entries:
        relative_path = str(entry.get("path", ""))
        if not relative_path:
            raise ValueError("hessian manifest entry is missing path")
        file_path = root / relative_path
        arrays, _metadata = load_hessian_file(file_path)
        h_in = _require_square_finite_matrix(arrays, "h_in", path=file_path)
        h_out = _require_square_finite_matrix(arrays, "h_out", path=file_path)
        _require_ldl_factor(arrays, "h_in", dim=int(h_in.shape[0]), path=file_path)
        _require_ldl_factor(arrays, "h_out", dim=int(h_out.shape[0]), path=file_path)

        if "h_in_dim" in entry and int(entry["h_in_dim"]) != int(h_in.shape[0]):
            raise ValueError(f"{file_path} h_in_dim does not match manifest entry")
        if "h_out_dim" in entry and int(entry["h_out_dim"]) != int(h_out.shape[0]):
            raise ValueError(f"{file_path} h_out_dim does not match manifest entry")
        if str(entry.get("hout_status", "")).startswith("not_collected"):
            raise ValueError(f"{file_path} manifest entry records hout_status={entry['hout_status']!r}")

        summaries.append(
            FullYAQAHessianEntry(
                path=relative_path,
                layer=int(entry["layer"]),
                projection=str(entry["projection"]),
                expert=int(entry["expert"]),
                sample_count=int(entry.get("sample_count", 0)),
                h_in_dim=int(h_in.shape[0]),
                h_out_dim=int(h_out.shape[0]),
            )
        )

    return {
        "schema_version": 1,
        "record_type": "full_yaqa_hessian_manifest_audit",
        "ok": True,
        "method": dict(method),
        "entry_count": len(summaries),
        "entries": [asdict(summary) for summary in summaries],
    }


def _decode_from_full_codebook(
    codes: np.ndarray,
    scales: np.ndarray,
    codebook: np.ndarray,
    *,
    group_size: int,
) -> np.ndarray:
    code_array = np.asarray(codes)
    scale_array = np.asarray(scales, dtype=np.float32)
    table = np.asarray(codebook, dtype=np.float32)
    if code_array.ndim != 2:
        raise ValueError("codes must be 2D [out, codewords]")
    if scale_array.ndim != 2:
        raise ValueError("scales must be 2D [out, groups]")
    if table.ndim != 2 or table.shape[1] != CODEWORD_DIM:
        raise ValueError(f"codebook must have shape [N, {CODEWORD_DIM}]")
    words_per_scale = group_size // CODEWORD_DIM
    if words_per_scale <= 0:
        raise ValueError("group_size must be at least 8")
    if code_array.shape[-1] % words_per_scale != 0:
        raise ValueError("codeword count must be divisible by group_size / 8")
    if scale_array.shape[-1] != code_array.shape[-1] // words_per_scale:
        raise ValueError("scale count does not match codeword groups")
    expanded_scales = np.repeat(scale_array, words_per_scale, axis=-1)[..., None]
    decoded = table[code_array.astype(np.int64)] * expanded_scales
    return decoded.reshape((code_array.shape[0], code_array.shape[1] * CODEWORD_DIM)).astype(np.float32)


def _validate_codebook_search_table(table: np.ndarray, *, code_bits: int) -> None:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    max_entries = 1 << int(code_bits)
    if table.ndim != 2 or table.shape[1] != CODEWORD_DIM or table.shape[0] == 0 or table.shape[0] > max_entries:
        raise ValueError(
            f"codebook must have shape [1..{max_entries}, {CODEWORD_DIM}] for code_bits={code_bits}, "
            f"found {table.shape}"
        )


def hin_weighted_error(weight: np.ndarray, source: np.ndarray, h_in: np.ndarray) -> float:
    diff = np.asarray(weight, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    h = np.asarray(h_in, dtype=np.float64)
    return float(np.einsum("oi,ij,oj->", diff, h, diff))


def kronecker_weighted_error(
    weight: np.ndarray,
    source: np.ndarray,
    h_in: np.ndarray,
    h_out: np.ndarray,
) -> float:
    diff = np.asarray(weight, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    h_i = np.asarray(h_in, dtype=np.float64)
    h_o = np.asarray(h_out, dtype=np.float64)
    if diff.ndim != 2:
        raise ValueError(f"weight/source diff must be 2D [out, in], found {diff.shape}")
    if h_i.shape != (diff.shape[1], diff.shape[1]):
        raise ValueError(f"h_in must have shape ({diff.shape[1]}, {diff.shape[1]}), found {h_i.shape}")
    if h_o.shape != (diff.shape[0], diff.shape[0]):
        raise ValueError(f"h_out must have shape ({diff.shape[0]}, {diff.shape[0]}), found {h_o.shape}")
    if not np.isfinite(diff).all() or not np.isfinite(h_i).all() or not np.isfinite(h_o).all():
        raise ValueError("weight, source, h_in, and h_out must be finite")
    return float(np.einsum("oi,op,pj,ij->", diff, h_o, diff, h_i))


def _selected_output_hessian(
    h_out: np.ndarray,
    *,
    selected_rows: np.ndarray,
    output_dim: int,
) -> np.ndarray:
    h = np.asarray(h_out, dtype=np.float64)
    selected_count = int(selected_rows.size)
    if h.shape == (output_dim, output_dim):
        return h[np.ix_(selected_rows, selected_rows)]
    if h.shape == (selected_count, selected_count):
        return h
    raise ValueError(
        "h_out must be either full output-dimension square or selected-row square; "
        f"found {h.shape}, expected ({output_dim}, {output_dim}) or ({selected_count}, {selected_count})"
    )


def _blockldlq_hin_only_reassign_codes_reference(
    source_weight: np.ndarray,
    current_codes: np.ndarray,
    scales: np.ndarray,
    h_in: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int = 8,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None = None,
    sweeps: int = 1,
) -> BlockLDLQReassignedWeight:
    """Scalar reference for the H_in-only BlockLDLQ coordinate updates.

    This is the explicit single-host fallback named by the plan:
    ``blockldlq_hin_only``. It uses input-side curvature and fixed scales, but
    no output-side Hessian and no RHT/YAQA recovery training.
    """

    if group_size <= 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")
    source = np.asarray(source_weight, dtype=np.float32)
    codes = np.asarray(current_codes)
    scale_array = np.asarray(scales, dtype=np.float32)
    h = np.asarray(h_in, dtype=np.float64)
    table = np.asarray(codebook, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    if source.shape[1] % CODEWORD_DIM != 0:
        raise ValueError("source input dimension must be divisible by 8")
    if h.shape != (source.shape[1], source.shape[1]):
        raise ValueError(f"h_in must have shape ({source.shape[1]}, {source.shape[1]}), found {h.shape}")
    if codes.shape != (source.shape[0], source.shape[1] // CODEWORD_DIM):
        raise ValueError("codes shape does not match source weight")
    if scale_array.shape != (source.shape[0], source.shape[1] // group_size):
        raise ValueError("scales shape does not match source weight")
    _validate_codebook_search_table(table, code_bits=code_bits)
    if not np.isfinite(h).all():
        raise ValueError("h_in must be finite")

    if row_indices is None:
        selected_rows = np.arange(source.shape[0], dtype=np.int64)
    else:
        selected_rows = np.asarray(row_indices, dtype=np.int64)
    if selected_rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if selected_rows.size and (int(np.min(selected_rows)) < 0 or int(np.max(selected_rows)) >= source.shape[0]):
        raise ValueError("row_indices contain rows outside source_weight")

    selected_source = source[selected_rows]
    selected_codes = codes[selected_rows].copy()
    selected_scales = scale_array[selected_rows]
    current_weight = _decode_from_full_codebook(
        selected_codes,
        selected_scales,
        table,
        group_size=group_size,
    )
    new_weight = current_weight.copy()
    new_codes = selected_codes.copy()

    codewords = new_codes.shape[1]
    words_per_scale = group_size // CODEWORD_DIM
    table64 = table.astype(np.float64)
    h_blocks = [
        h[start : start + CODEWORD_DIM, start : start + CODEWORD_DIM]
        for start in range(0, source.shape[1], CODEWORD_DIM)
    ]

    for row in range(new_codes.shape[0]):
        residual = new_weight[row].astype(np.float64) - selected_source[row].astype(np.float64)
        gradient = h @ residual
        for _sweep in range(sweeps):
            for codeword in range(codewords):
                dim_start = codeword * CODEWORD_DIM
                dim_end = dim_start + CODEWORD_DIM
                scale = float(selected_scales[row, codeword // words_per_scale])
                candidates = table64 * scale
                current_block = new_weight[row, dim_start:dim_end].astype(np.float64)
                deltas = candidates - current_block[None, :]
                gradient_block = gradient[dim_start:dim_end]
                h_block = h_blocks[codeword]
                delta_objective = 2.0 * (deltas @ gradient_block) + np.einsum(
                    "bi,ij,bj->b",
                    deltas,
                    h_block,
                    deltas,
                )
                best_code = int(np.argmin(delta_objective))
                if best_code == int(new_codes[row, codeword]):
                    continue
                delta = deltas[best_code]
                new_codes[row, codeword] = best_code
                new_weight[row, dim_start:dim_end] = candidates[best_code].astype(np.float32)
                residual[dim_start:dim_end] += delta
                gradient += h[:, dim_start:dim_end] @ delta

    current_error = hin_weighted_error(current_weight, selected_source, h)
    new_error = hin_weighted_error(new_weight, selected_source, h)
    changed = int(np.count_nonzero(new_codes != selected_codes))
    code_count = int(new_codes.size)
    stats = BlockLDLQStats(
        group_size=int(group_size),
        code_bits=int(code_bits),
        row_count=int(new_codes.shape[0]),
        codewords_per_row=int(codewords),
        code_count=code_count,
        changed_code_count=changed,
        changed_code_fraction=float(changed / code_count) if code_count else 0.0,
        current_hin_weighted_error=current_error,
        hin_weighted_error=new_error,
        hin_weighted_error_ratio=float(new_error / current_error) if abs(current_error) > 1.0e-12 else 1.0,
        sweeps=int(sweeps),
    )
    return BlockLDLQReassignedWeight(codes=new_codes, weight=new_weight, stats=stats)


def blockldlq_hin_only_reassign_codes(
    source_weight: np.ndarray,
    current_codes: np.ndarray,
    scales: np.ndarray,
    h_in: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int = 8,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None = None,
    sweeps: int = 1,
) -> BlockLDLQReassignedWeight:
    """Reassign 8D codewords with batched H_in coordinate updates.

    Rows are independent and processed in bounded chunks. Within each chunk,
    codewords and sweeps remain sequential while each codeword step searches
    all selected rows with batched float64 NumPy operations.
    """

    if group_size <= 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")
    source = np.asarray(source_weight, dtype=np.float32)
    codes = np.asarray(current_codes)
    scale_array = np.asarray(scales, dtype=np.float32)
    h = np.asarray(h_in, dtype=np.float64)
    table = np.asarray(codebook, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    if source.shape[1] % CODEWORD_DIM != 0:
        raise ValueError("source input dimension must be divisible by 8")
    if h.shape != (source.shape[1], source.shape[1]):
        raise ValueError(f"h_in must have shape ({source.shape[1]}, {source.shape[1]}), found {h.shape}")
    if codes.shape != (source.shape[0], source.shape[1] // CODEWORD_DIM):
        raise ValueError("codes shape does not match source weight")
    if scale_array.shape != (source.shape[0], source.shape[1] // group_size):
        raise ValueError("scales shape does not match source weight")
    _validate_codebook_search_table(table, code_bits=code_bits)
    if not np.isfinite(h).all():
        raise ValueError("h_in must be finite")

    if row_indices is None:
        selected_rows = np.arange(source.shape[0], dtype=np.int64)
    else:
        selected_rows = np.asarray(row_indices, dtype=np.int64)
    if selected_rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if selected_rows.size and (int(np.min(selected_rows)) < 0 or int(np.max(selected_rows)) >= source.shape[0]):
        raise ValueError("row_indices contain rows outside source_weight")

    selected_source = source[selected_rows]
    selected_codes = codes[selected_rows].copy()
    selected_scales = scale_array[selected_rows]
    current_weight = _decode_from_full_codebook(
        selected_codes,
        selected_scales,
        table,
        group_size=group_size,
    )
    new_weight = current_weight.copy()
    new_codes = selected_codes.copy()

    codewords = new_codes.shape[1]
    words_per_scale = group_size // CODEWORD_DIM
    table64 = table.astype(np.float64)
    h_blocks = [
        h[start : start + CODEWORD_DIM, start : start + CODEWORD_DIM]
        for start in range(0, source.shape[1], CODEWORD_DIM)
    ]

    row_chunk_size = max(1, min(512, (512 * 256) // table64.shape[0]))
    for row_start in range(0, new_codes.shape[0], row_chunk_size):
        row_end = min(row_start + row_chunk_size, new_codes.shape[0])
        chunk_slice = slice(row_start, row_end)
        chunk_weight = new_weight[chunk_slice]
        chunk_codes = new_codes[chunk_slice]
        chunk_scales = selected_scales[chunk_slice]
        residual = chunk_weight.astype(np.float64) - selected_source[chunk_slice].astype(np.float64)
        gradient = residual @ h
        chunk_rows = np.arange(row_end - row_start)
        for _sweep in range(sweeps):
            candidates = None
            for codeword in range(codewords):
                dim_start = codeword * CODEWORD_DIM
                dim_end = dim_start + CODEWORD_DIM
                if codeword % words_per_scale == 0:
                    scale = chunk_scales[:, codeword // words_per_scale].astype(np.float64)
                    candidates = table64[None, :, :] * scale[:, None, None]
                assert candidates is not None
                current_block = chunk_weight[:, dim_start:dim_end].astype(np.float64)
                deltas = candidates - current_block[:, None, :]
                gradient_block = gradient[:, dim_start:dim_end]
                h_block = h_blocks[codeword]
                flat_deltas = deltas.reshape((-1, CODEWORD_DIM))
                delta_objective = 2.0 * np.einsum(
                    "rbi,ri->rb",
                    deltas,
                    gradient_block,
                ) + np.einsum(
                    "bi,ij,bj->b",
                    flat_deltas,
                    h_block,
                    flat_deltas,
                ).reshape((row_end - row_start, table64.shape[0]))
                best_codes = np.argmin(delta_objective, axis=1)
                changed = best_codes != chunk_codes[:, codeword]
                if not np.any(changed):
                    continue
                chosen_deltas = deltas[chunk_rows, best_codes]
                chosen_candidates = candidates[chunk_rows, best_codes]
                chunk_codes[changed, codeword] = best_codes[changed]
                chunk_weight[changed, dim_start:dim_end] = chosen_candidates[changed].astype(np.float32)
                residual[changed, dim_start:dim_end] += chosen_deltas[changed]
                gradient[changed] += chosen_deltas[changed] @ h[dim_start:dim_end, :]

    current_error = hin_weighted_error(current_weight, selected_source, h)
    new_error = hin_weighted_error(new_weight, selected_source, h)
    changed = int(np.count_nonzero(new_codes != selected_codes))
    code_count = int(new_codes.size)
    stats = BlockLDLQStats(
        group_size=int(group_size),
        code_bits=int(code_bits),
        row_count=int(new_codes.shape[0]),
        codewords_per_row=int(codewords),
        code_count=code_count,
        changed_code_count=changed,
        changed_code_fraction=float(changed / code_count) if code_count else 0.0,
        current_hin_weighted_error=current_error,
        hin_weighted_error=new_error,
        hin_weighted_error_ratio=float(new_error / current_error) if abs(current_error) > 1.0e-12 else 1.0,
        sweeps=int(sweeps),
    )
    return BlockLDLQReassignedWeight(codes=new_codes, weight=new_weight, stats=stats)


def blockldlq_full_reassign_codes(
    source_weight: np.ndarray,
    current_codes: np.ndarray,
    scales: np.ndarray,
    h_in: np.ndarray,
    h_out: np.ndarray,
    *,
    codebook: np.ndarray,
    group_size: int,
    code_bits: int = 8,
    row_indices: np.ndarray | tuple[int, ...] | list[int] | None = None,
    sweeps: int = 1,
) -> FullBlockLDLQReassignedWeight:
    """Reassign 8D codewords using coupled H_out x H_in Kronecker feedback."""

    if group_size <= 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("group_size must be a positive multiple of 8")
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")
    source = np.asarray(source_weight, dtype=np.float32)
    codes = np.asarray(current_codes)
    scale_array = np.asarray(scales, dtype=np.float32)
    h_i = np.asarray(h_in, dtype=np.float64)
    table = np.asarray(codebook, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"source_weight must be 2D [out, in], found {source.shape}")
    if source.shape[1] % CODEWORD_DIM != 0:
        raise ValueError("source input dimension must be divisible by 8")
    if h_i.shape != (source.shape[1], source.shape[1]):
        raise ValueError(f"h_in must have shape ({source.shape[1]}, {source.shape[1]}), found {h_i.shape}")
    if codes.shape != (source.shape[0], source.shape[1] // CODEWORD_DIM):
        raise ValueError("codes shape does not match source weight")
    if scale_array.shape != (source.shape[0], source.shape[1] // group_size):
        raise ValueError("scales shape does not match source weight")
    _validate_codebook_search_table(table, code_bits=code_bits)
    if not np.isfinite(h_i).all():
        raise ValueError("h_in must be finite")

    if row_indices is None:
        selected_rows = np.arange(source.shape[0], dtype=np.int64)
    else:
        selected_rows = np.asarray(row_indices, dtype=np.int64)
    if selected_rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if selected_rows.size and (int(np.min(selected_rows)) < 0 or int(np.max(selected_rows)) >= source.shape[0]):
        raise ValueError("row_indices contain rows outside source_weight")
    if selected_rows.size == 0:
        raise ValueError("row_indices must select at least one row")
    h_o = _selected_output_hessian(h_out, selected_rows=selected_rows, output_dim=source.shape[0])
    if not np.isfinite(h_o).all():
        raise ValueError("h_out must be finite")

    selected_source = source[selected_rows]
    selected_codes = codes[selected_rows].copy()
    selected_scales = scale_array[selected_rows]
    current_weight = _decode_from_full_codebook(
        selected_codes,
        selected_scales,
        table,
        group_size=group_size,
    )
    new_weight = current_weight.astype(np.float64, copy=True)
    new_codes = selected_codes.copy()
    residual = new_weight - selected_source.astype(np.float64)
    gradient = h_o @ residual @ h_i

    codewords = new_codes.shape[1]
    words_per_scale = group_size // CODEWORD_DIM
    table64 = table.astype(np.float64)
    h_blocks = [
        h_i[start : start + CODEWORD_DIM, start : start + CODEWORD_DIM]
        for start in range(0, source.shape[1], CODEWORD_DIM)
    ]

    for _sweep in range(sweeps):
        for row in range(new_codes.shape[0]):
            row_weight = float(h_o[row, row])
            for codeword in range(codewords):
                dim_start = codeword * CODEWORD_DIM
                dim_end = dim_start + CODEWORD_DIM
                scale = float(selected_scales[row, codeword // words_per_scale])
                candidates = table64 * scale
                current_block = new_weight[row, dim_start:dim_end]
                deltas = candidates - current_block[None, :]
                gradient_block = gradient[row, dim_start:dim_end]
                h_block = h_blocks[codeword]
                delta_objective = 2.0 * (deltas @ gradient_block) + row_weight * np.einsum(
                    "bi,ij,bj->b",
                    deltas,
                    h_block,
                    deltas,
                )
                best_code = int(np.argmin(delta_objective))
                if best_code == int(new_codes[row, codeword]):
                    continue
                delta = deltas[best_code]
                new_codes[row, codeword] = best_code
                new_weight[row, dim_start:dim_end] = candidates[best_code]
                residual[row, dim_start:dim_end] += delta
                gradient += h_o[:, row : row + 1] * (delta @ h_i[dim_start:dim_end, :])[None, :]

    current_error = kronecker_weighted_error(current_weight, selected_source, h_i, h_o)
    new_error = kronecker_weighted_error(new_weight, selected_source, h_i, h_o)
    changed = int(np.count_nonzero(new_codes != selected_codes))
    code_count = int(new_codes.size)
    stats = FullBlockLDLQStats(
        group_size=int(group_size),
        code_bits=int(code_bits),
        row_count=int(new_codes.shape[0]),
        codewords_per_row=int(codewords),
        code_count=code_count,
        changed_code_count=changed,
        changed_code_fraction=float(changed / code_count) if code_count else 0.0,
        current_kronecker_weighted_error=current_error,
        kronecker_weighted_error=new_error,
        kronecker_weighted_error_ratio=float(new_error / current_error) if abs(current_error) > 1.0e-12 else 1.0,
        sweeps=int(sweeps),
    )
    return FullBlockLDLQReassignedWeight(codes=new_codes, weight=new_weight.astype(np.float32), stats=stats)


def relative_symlink_target(source_path: Path, target_parent: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=target_parent.resolve())


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stats_dict(stats: BlockLDLQStats | FullBlockLDLQStats) -> dict[str, Any]:
    return asdict(stats)


__all__ = [
    "FullYAQAHessianEntry",
    "BlockLDLQReassignedWeight",
    "BlockLDLQStats",
    "FullBlockLDLQReassignedWeight",
    "FullBlockLDLQStats",
    "LDLFactor",
    "activation_covariance",
    "blockldlq_full_reassign_codes",
    "blockldlq_hin_only_reassign_codes",
    "hin_weighted_error",
    "kronecker_weighted_error",
    "ldl_factor",
    "load_hessian_file",
    "reconstruct_ldl",
    "regularize_hessian",
    "relative_symlink_target",
    "require_full_yaqa_hessian_manifest",
    "save_hessian_factors",
    "stats_dict",
    "write_json",
    "yaqa_sketch_a_hessian_factors",
]
