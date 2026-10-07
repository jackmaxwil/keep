"""MLX/Metal port of the DeepSeek-V4 VQ fit's CPU hot paths.

The Wave 5 pilot profiled one ``[2048, 4096]`` projection at 8 iterations and
found the fit **65% CPU NumPy**: Metal E8P search 35.1%, codebook gather 28.7%,
float64 scale-update sums 30.6%, normalise/broadcast 5.5%. The standing
MLX/Metal directive says that CPU work belongs on the GPU, so this module is a
stage-by-stage port of
:func:`keep.convert.glm52_recovery_materialize.quantize_weight_importance_aware`
whose *only* deliberate difference from the reference is a choice the caller
makes explicitly.

What each stage does and what its parity contract is:

``normalise`` (``grouped / scales``)
    Ported. **Byte-exact.** MLX's Metal float32 divide is IEEE round-to-nearest
    and agrees with NumPy's on every bit; asserted by
    ``test_mlx_normalise_is_byte_exact``.

``search`` (weighted nearest E8P codeword)
    Already Metal, but the shipped entry point round-trips through the host
    every call. This module drives
    :func:`keep.quant.e8p_metal.encode_e8p_diagonal_hessian_fused_mx`
    instead, so the vectors, the per-row metric and the codes all stay
    resident. Same kernel, same bytes.

``gather`` (``table[codes]``)
    Ported to ``mx.take``. **Byte-exact** -- a gather moves codebook entries,
    it does not compute.

``products`` (``h * w * decoded`` and ``h * decoded * decoded``)
    Ported. **Byte-exact**: IEEE float32 multiply, evaluated in the reference's
    own left-to-right order. Asserted by ``test_mlx_products_are_byte_exact``.

``reduce`` (the float64 scale-update sums)
    **Not portable exactly.** Metal has no float64, and an fp32 tree reduction
    is not NumPy's float64 pairwise sum: measured on this geometry, ~30% of the
    per-group scales land on a different final float32 bit (~3e-8 relative).
    Those scales feed the next iteration's assignment, so a differing bit can
    flip an argmin and therefore change the artifact's *bytes*. Two modes,
    named rather than assumed:

    ``scale_reduction="float64"`` (default, and what the sweep runs)
        The fp32 product tensors come back from the GPU and NumPy performs the
        identical ``np.sum(..., dtype=np.float64)`` reduction. Byte-identical
        codes and scales to the NumPy reference, proven by
        ``test_mlx_exact_fit_is_byte_identical_to_the_numpy_reference``.

    ``scale_reduction="float32"``
        ``mx.sum`` on the GPU. Faster, and **not** bit-reproducible. Only for
        experiments that have measured and accepted the drift;
        :func:`compare_fit_backends` is the tool that measures it.

Only ``code_bits=16`` (E8P) is ported. ``code_bits=8`` keeps the NumPy path --
the ship ladder is E8P and porting a codebook nobody will run would be
untested surface.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

__all__ = [
    "SCALE_REDUCTIONS",
    "FitBackendComparison",
    "compare_fit_backends",
    "fit_backend_is_available",
    "quantize_weight_importance_aware_mlx",
    "resolve_fit_backend",
]

#: How the scale-update sums are accumulated. See the module docstring.
SCALE_REDUCTIONS = ("float64", "float32")

#: Backends ``resolve_fit_backend`` understands.
FIT_BACKENDS = ("numpy", "mlx-exact", "mlx-fp32")

CODEWORD_DIM = 8


def fit_backend_is_available(backend: str) -> bool:
    """True when ``backend`` can actually run on this host."""

    if backend == "numpy":
        return True
    if backend not in FIT_BACKENDS:
        raise ValueError(f"unknown fit backend {backend!r}; expected one of {FIT_BACKENDS}")
    try:
        import mlx.core as mx
    except ImportError:
        return False
    metal = getattr(mx, "metal", None)
    return bool(metal is not None and metal.is_available())


def resolve_fit_backend(backend: str, *, code_bits: int) -> str:
    """Pick the backend that will actually run, refusing silent downgrades.

    A caller that asks for an MLX backend on a host without Metal, or for a
    codebook the port does not cover, gets NumPy -- but the *caller* is told,
    because the return value is what the manifest records.
    """

    if backend not in FIT_BACKENDS:
        raise ValueError(f"unknown fit backend {backend!r}; expected one of {FIT_BACKENDS}")
    if backend == "numpy":
        return "numpy"
    if code_bits != 16:
        return "numpy"
    return backend if fit_backend_is_available(backend) else "numpy"


def quantize_weight_importance_aware_mlx(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    group_size: int = 512,
    code_bits: int = 16,
    iterations: int = 3,
    scale_reduction: Literal["float64", "float32"] = "float64",
) -> Any:
    """MLX twin of ``quantize_weight_importance_aware`` for the E8P path.

    Returns the same ``ImportanceAwareQuantizedWeight`` the reference returns.
    With ``scale_reduction="float64"`` the codes and scales are byte-identical
    to the reference's.
    """

    import mlx.core as mx

    from keep.vq.e8 import e8p_full_grid, e8p_packed_abs_grid
    from keep.convert.glm52_recovery_materialize import (
        ImportanceAwareQuantizedWeight,
        _validate_importance,
    )
    from keep.quant.e8p_metal import encode_e8p_diagonal_hessian_fused_mx

    if code_bits != 16:
        raise ValueError("the MLX fit port covers code_bits=16 (E8P) only")
    if scale_reduction not in SCALE_REDUCTIONS:
        raise ValueError(f"scale_reduction must be one of {SCALE_REDUCTIONS}")
    values = np.asarray(weight, dtype=np.float32)
    # Same validation, same coercion of an all-zero importance vector to ones,
    # so the two backends cannot disagree about what they were asked to fit.
    diagonal = _validate_importance(values, importance)
    if group_size <= 0 or group_size % 8 or values.shape[1] % group_size:
        raise ValueError(
            "group_size must be positive, divisible by 8, and divide the input dimension"
        )
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    out_dim, in_dim = values.shape
    group_count = in_dim // group_size
    codewords_per_group = group_size // CODEWORD_DIM

    table = e8p_full_grid().astype(np.float32)
    table_mx = mx.array(table)
    grouped = mx.array(
        np.ascontiguousarray(
            values.reshape(out_dim, group_count, codewords_per_group, CODEWORD_DIM)
        )
    )
    hessian = mx.array(
        np.ascontiguousarray(
            diagonal.reshape(group_count, codewords_per_group, CODEWORD_DIM)
        )
    )
    # The search kernel wants one metric row per codeword row. The metric is
    # shared across output rows, so this is a single [N, 8] materialisation
    # held for the whole fit instead of a fresh 33 MB broadcast per iteration.
    metric_rows = mx.broadcast_to(
        hessian.reshape(1, group_count * codewords_per_group, CODEWORD_DIM),
        (out_dim, group_count * codewords_per_group, CODEWORD_DIM),
    ).reshape(-1, CODEWORD_DIM)
    metric_rows = mx.contiguous(metric_rows)
    mx.eval(metric_rows)

    divisor = float(np.max(np.abs(table)))
    scales_mx = mx.max(mx.abs(grouped), axis=(2, 3)) / mx.array(np.float32(divisor))
    scales_mx = mx.where(scales_mx > 0, scales_mx, mx.array(np.float32(1.0)))
    mx.eval(scales_mx)

    codes_mx: Any = None

    def assign_codes() -> Any:
        normalized = grouped / scales_mx[:, :, None, None]
        flat = mx.contiguous(normalized.reshape(-1, CODEWORD_DIM))
        return encode_e8p_diagonal_hessian_fused_mx(flat, metric_rows)

    for _ in range(iterations):
        codes_mx = assign_codes()
        decoded = mx.take(table_mx, codes_mx, axis=0).reshape(
            out_dim, group_count, codewords_per_group, CODEWORD_DIM
        )
        # Multiplication order matches the reference term for term.
        numerator_terms = hessian[None, ...] * grouped * decoded
        denominator_terms = hessian[None, ...] * decoded * decoded
        if scale_reduction == "float64":
            mx.eval(numerator_terms, denominator_terms)
            numerator = np.sum(
                np.asarray(numerator_terms), axis=(2, 3), dtype=np.float64
            )
            denominator = np.sum(
                np.asarray(denominator_terms), axis=(2, 3), dtype=np.float64
            )
        else:
            summed = mx.sum(numerator_terms, axis=(2, 3))
            summed_denominator = mx.sum(denominator_terms, axis=(2, 3))
            mx.eval(summed, summed_denominator)
            numerator = np.asarray(summed).astype(np.float64)
            denominator = np.asarray(summed_denominator).astype(np.float64)
        scales = np.asarray(scales_mx, dtype=np.float32)
        updates = np.divide(
            numerator,
            denominator,
            out=scales.astype(np.float64),
            where=(denominator > 0) & (numerator > 0),
        )
        scales_mx = mx.array(updates.astype(np.float32))

    codes_mx = assign_codes()
    mx.eval(codes_mx)
    codes = (
        np.asarray(codes_mx)
        .astype(np.uint16)
        .reshape(out_dim, group_count, codewords_per_group)
        .reshape(out_dim, in_dim // CODEWORD_DIM)
    )
    scales = np.asarray(scales_mx, dtype=np.float32)
    return ImportanceAwareQuantizedWeight(
        codes=np.ascontiguousarray(codes),
        scales=scales.astype(np.float16),
        codebook=e8p_packed_abs_grid(),
        group_size=group_size,
        code_bits=code_bits,
        stats={},
    )


class FitBackendComparison(dict):
    """A measured NumPy-vs-MLX comparison, as a plain JSON-able mapping."""


def compare_fit_backends(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    group_size: int = 512,
    code_bits: int = 16,
    iterations: int = 8,
    backends: tuple[str, ...] = ("numpy", "mlx-exact", "mlx-fp32"),
) -> FitBackendComparison:
    """Time every backend on one projection and diff their artifacts.

    The point of this function is that the speed-up and the parity claim come
    from the *same* run on the *same* weight, so a report cannot quote one
    without the other. ``code_disagreement_fraction`` is the fraction of
    codewords whose index differs from the NumPy reference's, and
    ``weighted_relative_mse`` lets a reader see whether a disagreement cost
    anything (near-tie flips do not).
    """

    from keep.convert.dsv4_vq_pilot import projection_error, reconstruct_quantized
    from keep.convert.glm52_recovery_materialize import (
        quantize_weight_importance_aware,
    )

    values = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    diagonal = np.ascontiguousarray(np.asarray(importance, dtype=np.float32))
    results: dict[str, Any] = {}
    reference_codes: np.ndarray | None = None
    reference_scales: np.ndarray | None = None

    for backend in backends:
        started = time.perf_counter()
        if backend == "numpy":
            quantized = quantize_weight_importance_aware(
                values,
                diagonal,
                group_size=group_size,
                code_bits=code_bits,
                iterations=iterations,
                e8p_search_backend="metal",
            )
        else:
            quantized = quantize_weight_importance_aware_mlx(
                values,
                diagonal,
                group_size=group_size,
                code_bits=code_bits,
                iterations=iterations,
                scale_reduction="float64" if backend == "mlx-exact" else "float32",
            )
        elapsed = time.perf_counter() - started
        metrics = projection_error(values, reconstruct_quantized(quantized), diagonal)
        codes = np.asarray(quantized.codes)
        scales = np.asarray(quantized.scales)
        if backend == "numpy":
            reference_codes, reference_scales = codes, scales
        row: dict[str, Any] = {
            "seconds": elapsed,
            "weighted_relative_mse": metrics.weighted_relative_mse,
            "expected_cosine": metrics.expected_cosine,
        }
        if reference_codes is not None and backend != "numpy":
            disagreement = float(np.mean(codes != reference_codes))
            row["codes_byte_identical"] = bool(np.array_equal(codes, reference_codes))
            row["scales_byte_identical"] = bool(
                np.array_equal(
                    scales.view(np.uint16), np.asarray(reference_scales).view(np.uint16)
                )
            )
            row["code_disagreement_fraction"] = disagreement
            row["speedup_vs_numpy"] = results["numpy"]["seconds"] / elapsed
            row["relative_mse_delta"] = (
                metrics.weighted_relative_mse - results["numpy"]["weighted_relative_mse"]
            )
        results[backend] = row

    return FitBackendComparison(
        {
            "shape": list(values.shape),
            "group_size": group_size,
            "code_bits": code_bits,
            "iterations": iterations,
            "backends": results,
        }
    )
