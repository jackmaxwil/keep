from __future__ import annotations

from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from keep.vq.e8 import CODEWORD_DIM, e8_1bit_packed, e8p_packed_abs_grid
from ramp.kernels.vq_qmv import _dtype_name, _qmv_output_dtype


_KERNEL_DIR = Path(__file__).resolve().parent
_GATHER_VQMM_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_M1_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_m1",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_m1.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_M1_DECODED_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_m1_decoded",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_m1_decoded.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_M1_ROWPAIR_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_m1_rowpair",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_m1_rowpair.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_M1_PER_ROUTE_DECODED_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_m1_per_route_decoded",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_m1_per_route_decoded.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_VERIFY_MROWS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_verify_mrows",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_verify_mrows.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_LHS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_lhs",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices", "lhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_lhs.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_TILED_DOWN_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_tiled_down",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices", "lhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_tiled_down.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA_DOWN_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma_down",
    input_names=["x", "codes", "scales", "codebook", "rhs_indices", "lhs_indices"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma_down.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA4X4_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma4x4_blocks",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma4x4_blocks.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA8X4_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma8x4_blocks",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma8x4_blocks.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA_CWDECODE_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma_cwdecode_blocks",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma_cwdecode_blocks.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA_K32_CWDECODE_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma_k32_cwdecode_blocks",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma_k32_cwdecode_blocks.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA_K64_CWDECODE_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma_k64_cwdecode_blocks",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma_k64_cwdecode_blocks.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)
_GATHER_VQMM_MMA_K128_CWDECODE_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="mlx_vq_gather_vqmm_mma_k128_cwdecode_blocks",
    input_names=["x", "codes", "scales", "codebook", "lhs_indices", "tile_experts", "tile_offsets", "tile_counts"],
    output_names=["out"],
    source=(_KERNEL_DIR / "gather_vqmm_mma_k128_cwdecode_blocks.metal").read_text(),
    header=(_KERNEL_DIR / "common_e8.metal").read_text(),
)


def _as_mlx(array: mx.array | np.ndarray) -> mx.array:
    if isinstance(array, mx.array):
        return array
    return mx.array(array)


def _default_codebook(code_bits: Literal[8, 16]) -> np.ndarray:
    if code_bits == 8:
        return e8_1bit_packed()
    return e8p_packed_abs_grid()


def m1_rows_per_threadgroup(input_dims: int, output_dims: int) -> Literal[16, 32, 64]:
    """Shape-tuned row packing for the single-token Air decode kernel."""

    return 32


def m1_use_threadgroup_codebook(input_dims: int, output_dims: int) -> bool:
    """Shape-tuned codebook staging for the single-token Air decode kernel."""

    return False


def m1_use_decoded_codebook(input_dims: int, output_dims: int) -> bool:
    """Shape-tuned codebook decode mode for the single-token Air decode kernel."""

    return True


def gather_vqmm_m1_kernel_unchecked(
    x_mx: mx.array,
    codes_mx: mx.array,
    scales_mx: mx.array,
    codebook_mx: mx.array,
    rhs_mx: mx.array,
    *,
    output_dims: int,
    code_bits: Literal[8, 16] = 8,
    rows_per_threadgroup: Literal[2, 4, 8, 16, 32, 64, 128, 256] = 4,
    use_threadgroup_codebook: bool = True,
    use_decoded_codebook: bool = False,
) -> mx.array:
    """Launch the M=1 direct kernel for already-validated MLX arrays."""

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    _tokens, top_k = rhs_mx.shape
    lanes_per_row = 256 // rows_per_threadgroup
    if lanes_per_row not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("256 / rows_per_threadgroup must be 1, 2, 4, 8, 16, 32, 64, or 128")
    if use_decoded_codebook and lanes_per_row > 32:
        raise ValueError("decoded M=1 kernels require lanes_per_row <= 32")
    row_tiles = (output_dims + rows_per_threadgroup - 1) // rows_per_threadgroup
    kernel = _GATHER_VQMM_M1_KERNEL
    if code_bits == 8 and use_decoded_codebook:
        kernel = _GATHER_VQMM_M1_DECODED_KERNEL
    template = [
        ("ROWS_PER_TG", rows_per_threadgroup),
        ("LANES_PER_ROW", lanes_per_row),
        ("OUT_T", output_dtype),
    ]
    if kernel is _GATHER_VQMM_M1_KERNEL:
        template.extend(
            [
                ("CODE_BITS", code_bits),
                ("USE_THREADGROUP_CODEBOOK", int(use_threadgroup_codebook)),
                ("USE_DECODED_CODEBOOK", int(use_decoded_codebook)),
            ]
        )
    outputs = kernel(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx],
        template=template,
        grid=(256, row_tiles, top_k),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, top_k, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_m1_rowpair_kernel_unchecked(
    x_mx: mx.array,
    codes_mx: mx.array,
    scales_mx: mx.array,
    codebook_mx: mx.array,
    rhs_mx: mx.array,
    *,
    output_dims: int,
) -> mx.array:
    """Benchmark-only M=1 decoded kernel that computes two rows per lane group."""

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    _tokens, top_k = rhs_mx.shape
    row_tiles = (output_dims + 31) // 32
    outputs = _GATHER_VQMM_M1_ROWPAIR_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx],
        template=[("OUT_T", output_dtype)],
        grid=(256, row_tiles, top_k),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, top_k, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_m1_per_route_kernel_unchecked(
    x_mx: mx.array,
    codes_mx: mx.array,
    scales_mx: mx.array,
    codebook_mx: mx.array,
    rhs_mx: mx.array,
    *,
    output_dims: int,
    rows_per_threadgroup: Literal[16, 32, 64] = 32,
) -> mx.array:
    """M=1 per-route decoded kernel for down-projection route activations."""

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    route_count = rhs_mx.shape[0]
    lanes_per_row = 256 // rows_per_threadgroup
    if lanes_per_row not in (4, 8, 16):
        raise ValueError("per-route M=1 decoded kernel supports 4, 8, or 16 lanes per row")
    row_tiles = (output_dims + rows_per_threadgroup - 1) // rows_per_threadgroup
    outputs = _GATHER_VQMM_M1_PER_ROUTE_DECODED_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx],
        template=[
            ("ROWS_PER_TG", rows_per_threadgroup),
            ("LANES_PER_ROW", lanes_per_row),
            ("OUT_T", output_dtype),
        ],
        grid=(256, row_tiles, route_count),
        threadgroup=(256, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]




VERIFY_MROWS_VALUES = (1, 2, 3, 4, 8)

# Row packings whose M=1 decoded reference reduces with the same shuffle-down
# tree this kernel uses, so byte-exactness holds. ROWS_PER_TG=8 is excluded:
# it yields LANES_PER_ROW=32, and at exactly 32 lanes
# gather_vqmm_m1_decoded.metal switches to simd_sum, a different reduction
# order. Measured at the V4 shapes, that divergence flips 46/49152 (gate/up)
# and 74/98304 (down) output elements, max delta ~1e-3.
VERIFY_ROWS_PER_THREADGROUP_VALUES = (16, 32, 64)


def verify_m_rows(route_count: int, top_k: int) -> Literal[1, 2, 3, 4, 8]:
    """Shape-keyed row-tile width for the wide-M verify kernel.

    ``route_count // top_k`` is the number of verify rows in the pass. The tile
    width caps how many same-expert routes one threadgroup folds together, so it
    never needs to exceed the verify width.
    """

    verify_rows = max(1, route_count // max(1, top_k))
    for candidate in VERIFY_MROWS_VALUES:
        if candidate >= verify_rows:
            return candidate
    return 8


def gather_vqmm_verify_mrows_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    m_rows: Literal[1, 2, 3, 4, 8] = 4,
    rows_per_threadgroup: Literal[16, 32, 64] = 32,
    codeword_unroll: Literal[1, 2, 4, 8] = 1,
    validate: bool = True,
) -> mx.array:
    """Wide-M VQ verify kernel: ``m_rows`` verify rows per expert-grouped tile.

    Shares the M=1 decoded kernel's lane discipline and float32 accumulation
    order, so each output element is bit-identical to
    ``gather_vqmm_m1_kernel_unchecked(..., use_decoded_codebook=True)`` at the
    **same** ``rows_per_threadgroup`` evaluated for that verify row. The code
    byte, decoded codebook entry, and group scale are hoisted above the
    verify-row loop and reused by every route in the tile.

    ``rows_per_threadgroup`` must be one of
    ``VERIFY_ROWS_PER_THREADGROUP_VALUES``. Callers should source it from
    ``m1_rows_per_threadgroup(input_dims, output_dims)`` rather than a literal,
    so verify tracks whatever the M=1 decode path is tuned to; passing a value
    the M=1 reference reduces differently raises instead of silently diverging.

    Route tiles use the same ``(tile_experts, tile_offsets, tile_counts)``
    descriptor triple as the simdgroup-MMA block family, so "flat" dispatch
    (one route per tile) and expert-grouped dispatch share one kernel. Returns
    ``[route_count, output_dims]`` in the route order the descriptors describe.

    ``codeword_unroll`` processes several codewords per lane iteration to
    overlap the code-byte -> codebook-lookup dependency chain. It stays
    byte-exact at every setting (accumulation remains ascending in codeword),
    but measured slower than 1 on every V4 shape at M=1..8 on M5 Max: the extra
    live registers cost more occupancy than the ILP recovers. Default 1; the
    knob is retained so the negative result is re-testable rather than folklore.
    """

    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(8)) if codebook is None else _as_mlx(codebook)

    lanes_per_row = 256 // rows_per_threadgroup
    if validate:
        if m_rows not in VERIFY_MROWS_VALUES:
            raise ValueError(f"m_rows must be one of {VERIFY_MROWS_VALUES}, found {m_rows}")
        if rows_per_threadgroup not in VERIFY_ROWS_PER_THREADGROUP_VALUES:
            raise ValueError(
                "rows_per_threadgroup must be one of "
                f"{VERIFY_ROWS_PER_THREADGROUP_VALUES}, found {rows_per_threadgroup}; "
                "256 / rows_per_threadgroup == 32 lanes per row is rejected because "
                "gather_vqmm_m1_decoded.metal reduces with simd_sum at exactly 32 "
                "lanes while this kernel uses a shuffle-down tree, which is a "
                "different accumulation order and breaks byte-exact parity"
            )
        if codeword_unroll not in (1, 2, 4, 8):
            raise ValueError("codeword_unroll must be 1, 2, 4, or 8")
        if lanes_per_row > 32:
            raise ValueError("decoded verify kernels require lanes_per_row <= 32")
        if route_count <= 0:
            raise ValueError("route_count must be positive")
        if input_dims <= 0 or output_dims <= 0:
            raise ValueError("input_dims and output_dims must be positive")
        if input_dims % CODEWORD_DIM != 0:
            raise ValueError("input_dims must be divisible by 8")
        if group_size <= 0 or input_dims % group_size != 0 or group_size % CODEWORD_DIM != 0:
            raise ValueError("input_dims must be divisible by positive 8D-aligned group_size")
        if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
            raise ValueError(f"x must have shape [rows, {input_dims}], found {x_mx.shape}")
        if lhs_mx.shape[0] != route_count:
            raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
        if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
            raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
        if codes_mx.ndim != 3:
            raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
        if scales_mx.ndim != 3:
            raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
        num_experts = codes_mx.shape[0]
        expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
        expected_scales = (num_experts, output_dims, input_dims // group_size)
        if codes_mx.shape != expected_codes:
            raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
        if scales_mx.shape != expected_scales:
            raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
        if codebook_mx.shape != (256,):
            raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
        if _dtype_name(codes_mx.dtype) != "uint8":
            raise ValueError(f"verify mrows kernel requires uint8 codes, found {codes_mx.dtype}")
        if _dtype_name(codebook_mx.dtype) != "uint32":
            raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    row_tiles = (output_dims + rows_per_threadgroup - 1) // rows_per_threadgroup
    tile_count = experts_mx.shape[0]
    outputs = _GATHER_VQMM_VERIFY_MROWS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[
            ("M_ROWS", m_rows),
            ("ROWS_PER_TG", rows_per_threadgroup),
            ("LANES_PER_ROW", lanes_per_row),
            ("CW_UNROLL", codeword_unroll),
            ("X_HALF", int(x_mx.dtype == mx.float16)),
            ("OUT_T", output_dtype),
        ],
        grid=(256, row_tiles, tile_count),
        threadgroup=(256, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def _validate_gather_metadata(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    codebook: mx.array,
    rhs_indices: mx.array,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int,
    code_bits: Literal[8, 16],
    codebook_duplication: Literal[1, 4, 8],
) -> None:
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")
    if x.ndim != 2 or x.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x.shape}")
    if rhs_indices.ndim != 2 or rhs_indices.shape[0] != x.shape[0]:
        raise ValueError(f"rhs_indices must have shape [tokens, top_k], found {rhs_indices.shape}")
    if codes.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes.shape}")
    if scales.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales.shape}")

    num_experts = codes.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes.shape}")
    if scales.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales.shape}")
    if codebook.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook.shape}")
    if codebook_duplication not in (1, 4, 8):
        raise ValueError("codebook_duplication must be 1, 4, or 8")

    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes.dtype}")
    if _dtype_name(codebook.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook.dtype}")


def gather_vqmm_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 1,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    rhs_mx = _as_mlx(rhs_indices)
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    _validate_gather_metadata(
        x_mx,
        codes_mx,
        scales_mx,
        codebook_mx,
        rhs_mx,
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=code_bits,
        codebook_duplication=codebook_duplication,
    )

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    tokens, top_k = rhs_mx.shape
    outputs = _GATHER_VQMM_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx],
        template=[
            ("CODE_BITS", code_bits),
            ("CODEBOOK_DUP", codebook_duplication),
            ("OUT_T", output_dtype),
        ],
        grid=(256, output_dims, tokens * top_k),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, top_k, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_lhs_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    lhs_indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 1,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    rhs_mx = _as_mlx(rhs_indices).reshape((-1,))
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if rhs_mx.shape != lhs_mx.shape:
        raise ValueError("rhs_indices and lhs_indices must have the same flattened shape")
    if rhs_mx.ndim != 1:
        raise ValueError("rhs_indices must flatten to a 1D route vector")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    if codebook_duplication not in (1, 4, 8):
        raise ValueError("codebook_duplication must be 1, 4, or 8")
    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes_mx.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    route_count = rhs_mx.shape[0]
    outputs = _GATHER_VQMM_LHS_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx.astype(mx.int32), lhs_mx.astype(mx.int32)],
        template=[
            ("CODE_BITS", code_bits),
            ("CODEBOOK_DUP", codebook_duplication),
            ("OUT_T", output_dtype),
        ],
        grid=(256, output_dims, route_count),
        threadgroup=(256, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_tiled_down_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    lhs_indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    codebook_duplication: Literal[1, 4, 8] = 8,
    m_tile: Literal[8, 16, 32, 64] = 16,
    n_tile: Literal[4, 8] = 8,
    k_tile_dims: int | None = None,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    rhs_mx = _as_mlx(rhs_indices).reshape((-1,))
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    if k_tile_dims is None:
        k_tile_dims = group_size
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0 or input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by positive group_size")
    if k_tile_dims <= 0 or k_tile_dims % CODEWORD_DIM != 0:
        raise ValueError("k_tile_dims must be positive and divisible by 8")
    if input_dims % k_tile_dims != 0:
        raise ValueError("input_dims must be divisible by k_tile_dims for the provisional tiled kernel")
    if m_tile not in (8, 16, 32, 64):
        raise ValueError("m_tile must be 8, 16, 32, or 64")
    if n_tile not in (4, 8):
        raise ValueError("n_tile must be 4 or 8")
    if m_tile * n_tile > 256:
        raise ValueError("m_tile * n_tile must fit in one 256-lane threadgroup")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if rhs_mx.shape != lhs_mx.shape:
        raise ValueError("rhs_indices and lhs_indices must have the same flattened shape")
    if rhs_mx.ndim != 1:
        raise ValueError("rhs_indices must flatten to a 1D route vector")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    if codebook_duplication not in (1, 4, 8):
        raise ValueError("codebook_duplication must be 1, 4, or 8")
    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes_mx.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    route_count = rhs_mx.shape[0]
    route_tiles = (route_count + m_tile - 1) // m_tile
    row_tiles = (output_dims + n_tile - 1) // n_tile
    outputs = _GATHER_VQMM_TILED_DOWN_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx.astype(mx.int32), lhs_mx.astype(mx.int32)],
        template=[
            ("CODE_BITS", code_bits),
            ("CODEBOOK_DUP", codebook_duplication),
            ("OUT_T", output_dtype),
            ("M_TILE", m_tile),
            ("N_TILE", n_tile),
            ("K_TILE_DIMS", k_tile_dims),
        ],
        grid=(256, row_tiles, route_tiles),
        threadgroup=(256, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma_down_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    lhs_indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    rhs_mx = _as_mlx(rhs_indices).reshape((-1,))
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0 or input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by positive group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if rhs_mx.shape != lhs_mx.shape:
        raise ValueError("rhs_indices and lhs_indices must have the same flattened shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes_mx.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    route_count = rhs_mx.shape[0]
    route_tiles = (route_count + 7) // 8
    row_tiles = (output_dims + 7) // 8
    outputs = _GATHER_VQMM_MMA_DOWN_KERNEL(
        inputs=[x_mx, codes_mx, scales_mx, codebook_mx, rhs_mx.astype(mx.int32), lhs_mx.astype(mx.int32)],
        template=[
            ("CODE_BITS", code_bits),
            ("OUT_T", output_dtype),
        ],
        grid=(32, row_tiles, route_tiles),
        threadgroup=(32, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma4x4_blocks_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if route_count <= 0:
        raise ValueError("route_count must be positive")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0 or input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by positive group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if lhs_mx.shape[0] != route_count:
        raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
    if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
        raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes_mx.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    row_tiles = (output_dims + 31) // 32
    tile_count = experts_mx.shape[0]
    outputs = _GATHER_VQMM_MMA4X4_BLOCKS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[
            ("CODE_BITS", code_bits),
            ("OUT_T", output_dtype),
        ],
        grid=(512, row_tiles, tile_count),
        threadgroup=(512, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma8x4_blocks_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if route_count <= 0:
        raise ValueError("route_count must be positive")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0 or input_dims % group_size != 0:
        raise ValueError("input_dims must be divisible by positive group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if lhs_mx.shape[0] != route_count:
        raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
    if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
        raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    expected_code_dtype = "uint8" if code_bits == 8 else "uint16"
    if _dtype_name(codes_mx.dtype) != expected_code_dtype:
        raise ValueError(f"codes dtype must be {expected_code_dtype}, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    row_tiles = (output_dims + 31) // 32
    tile_count = experts_mx.shape[0]
    outputs = _GATHER_VQMM_MMA8X4_BLOCKS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[
            ("CODE_BITS", code_bits),
            ("OUT_T", output_dtype),
        ],
        grid=(1024, row_tiles, tile_count),
        threadgroup=(1024, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma_cwdecode_blocks_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    route_tile: Literal[32, 64] = 32,
) -> mx.array:
    """Benchmark probe that decodes one full 8D VQ codeword per lane."""

    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(8)) if codebook is None else _as_mlx(codebook)

    if route_tile not in (32, 64):
        raise ValueError("route_tile must be 32 or 64")
    if route_count <= 0:
        raise ValueError("route_count must be positive")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by 8")
    if group_size <= 0 or input_dims % group_size != 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by positive 8D-aligned group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if lhs_mx.shape[0] != route_count:
        raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
    if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
        raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    if _dtype_name(codes_mx.dtype) != "uint8":
        raise ValueError(f"cwdecode block kernel requires uint8 codes, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    row_tiles = (output_dims + 31) // 32
    tile_count = experts_mx.shape[0]
    route_groups = route_tile // 8
    thread_count = route_groups * 128
    outputs = _GATHER_VQMM_MMA_CWDECODE_BLOCKS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[
            ("ROUTE_GROUPS", route_groups),
            ("A_TILE_SIZE", route_tile * 8),
            ("C_TILE_SIZE", route_groups * 4 * 64),
            ("OUT_T", output_dtype),
        ],
        grid=(thread_count, row_tiles, tile_count),
        threadgroup=(thread_count, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma_k32_cwdecode_blocks_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    route_tile: Literal[32, 64] = 32,
) -> mx.array:
    """Benchmark probe that stages 32 K dims per block-MMA barrier."""

    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(8)) if codebook is None else _as_mlx(codebook)

    if route_tile not in (32, 64):
        raise ValueError("route_tile must be 32 or 64")
    if route_count <= 0:
        raise ValueError("route_count must be positive")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % 32 != 0:
        raise ValueError("k32 cwdecode block kernel requires input_dims divisible by 32")
    if group_size <= 0 or input_dims % group_size != 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by positive 8D-aligned group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if lhs_mx.shape[0] != route_count:
        raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
    if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
        raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    if _dtype_name(codes_mx.dtype) != "uint8":
        raise ValueError(f"k32 cwdecode block kernel requires uint8 codes, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    row_tiles = (output_dims + 31) // 32
    tile_count = experts_mx.shape[0]
    route_groups = route_tile // 8
    thread_count = route_groups * 128
    outputs = _GATHER_VQMM_MMA_K32_CWDECODE_BLOCKS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[
            ("ROUTE_GROUPS", route_groups),
            ("A_TILE_SIZE", route_tile * 32),
            ("C_TILE_SIZE", route_groups * 4 * 64),
            ("OUT_T", output_dtype),
        ],
        grid=(thread_count, row_tiles, tile_count),
        threadgroup=(thread_count, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma_k64_cwdecode_blocks_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    route_tile: Literal[32, 64] = 32,
    row_groups: Literal[4, 8] = 4,
    use_device_codebook: bool = False,
) -> mx.array:
    """Benchmark probe that stages 64 K dims per block-MMA barrier."""

    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(8)) if codebook is None else _as_mlx(codebook)

    if route_tile not in (32, 64):
        raise ValueError("route_tile must be 32 or 64")
    if row_groups not in (4, 8):
        raise ValueError("row_groups must be 4 or 8")
    if row_groups == 8 and route_tile != 32:
        raise ValueError("row_groups=8 requires route_tile=32")
    if route_count <= 0:
        raise ValueError("route_count must be positive")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % 64 != 0:
        raise ValueError("k64 cwdecode block kernel requires input_dims divisible by 64")
    if group_size <= 0 or input_dims % group_size != 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by positive 8D-aligned group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if lhs_mx.shape[0] != route_count:
        raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
    if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
        raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    if _dtype_name(codes_mx.dtype) != "uint8":
        raise ValueError(f"k64 cwdecode block kernel requires uint8 codes, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    rows_per_tile = row_groups * 8
    row_tiles = (output_dims + rows_per_tile - 1) // rows_per_tile
    tile_count = experts_mx.shape[0]
    route_groups = route_tile // 8
    thread_count = route_groups * row_groups * 32
    outputs = _GATHER_VQMM_MMA_K64_CWDECODE_BLOCKS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[
            ("ROUTE_GROUPS", route_groups),
            ("ROW_GROUPS", row_groups),
            ("A_TILE_SIZE", route_tile * 64),
            ("B_TILE_SIZE", row_groups * 512),
            ("C_TILE_SIZE", route_groups * row_groups * 64),
            ("USE_DEVICE_CODEBOOK", int(use_device_codebook)),
            ("OUT_T", output_dtype),
        ],
        grid=(thread_count, row_tiles, tile_count),
        threadgroup=(thread_count, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_mma_k128_cwdecode_blocks_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    lhs_indices: mx.array | np.ndarray,
    tile_experts: mx.array | np.ndarray,
    tile_offsets: mx.array | np.ndarray,
    tile_counts: mx.array | np.ndarray,
    *,
    route_count: int,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
) -> mx.array:
    """Benchmark probe that stages 128 K dims for 32-route block tiles."""

    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    lhs_mx = _as_mlx(lhs_indices).reshape((-1,))
    experts_mx = _as_mlx(tile_experts).reshape((-1,))
    offsets_mx = _as_mlx(tile_offsets).reshape((-1,))
    counts_mx = _as_mlx(tile_counts).reshape((-1,))
    codebook_mx = mx.array(_default_codebook(8)) if codebook is None else _as_mlx(codebook)

    if route_count <= 0:
        raise ValueError("route_count must be positive")
    if input_dims <= 0 or output_dims <= 0:
        raise ValueError("input_dims and output_dims must be positive")
    if input_dims % 128 != 0:
        raise ValueError("k128 cwdecode block kernel requires input_dims divisible by 128")
    if group_size <= 0 or input_dims % group_size != 0 or group_size % CODEWORD_DIM != 0:
        raise ValueError("input_dims must be divisible by positive 8D-aligned group_size")
    if x_mx.ndim != 2 or x_mx.shape[1] != input_dims:
        raise ValueError(f"x must have shape [tokens, {input_dims}], found {x_mx.shape}")
    if lhs_mx.shape[0] != route_count:
        raise ValueError(f"lhs_indices must have length {route_count}, found {lhs_mx.shape[0]}")
    if experts_mx.shape != offsets_mx.shape or experts_mx.shape != counts_mx.shape:
        raise ValueError("tile_experts, tile_offsets, and tile_counts must have the same shape")
    if codes_mx.ndim != 3:
        raise ValueError(f"codes must be 3D [experts, out, in/8], found {codes_mx.shape}")
    if scales_mx.ndim != 3:
        raise ValueError(f"scales must be 3D [experts, out, in/group], found {scales_mx.shape}")
    num_experts = codes_mx.shape[0]
    expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
    expected_scales = (num_experts, output_dims, input_dims // group_size)
    if codes_mx.shape != expected_codes:
        raise ValueError(f"codes must have shape {expected_codes}, found {codes_mx.shape}")
    if scales_mx.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, found {scales_mx.shape}")
    if codebook_mx.shape != (256,):
        raise ValueError(f"codebook must have shape (256,), found {codebook_mx.shape}")
    if _dtype_name(codes_mx.dtype) != "uint8":
        raise ValueError(f"k128 cwdecode block kernel requires uint8 codes, found {codes_mx.dtype}")
    if _dtype_name(codebook_mx.dtype) != "uint32":
        raise ValueError(f"codebook dtype must be uint32, found {codebook_mx.dtype}")

    output_dtype = _qmv_output_dtype(x_mx.dtype)
    row_tiles = (output_dims + 31) // 32
    tile_count = experts_mx.shape[0]
    outputs = _GATHER_VQMM_MMA_K128_CWDECODE_BLOCKS_KERNEL(
        inputs=[
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            lhs_mx.astype(mx.int32),
            experts_mx.astype(mx.int32),
            offsets_mx.astype(mx.int32),
            counts_mx.astype(mx.int32),
        ],
        template=[("OUT_T", output_dtype)],
        grid=(512, row_tiles, tile_count),
        threadgroup=(512, 1, 1),
        output_shapes=[(route_count, output_dims)],
        output_dtypes=[output_dtype],
    )
    return outputs[0]


def gather_vqmm_m1_kernel(
    x: mx.array | np.ndarray,
    codes: mx.array | np.ndarray,
    scales: mx.array | np.ndarray,
    codebook: mx.array | np.ndarray | None,
    rhs_indices: mx.array | np.ndarray,
    *,
    input_dims: int,
    output_dims: int,
    group_size: int = 512,
    code_bits: Literal[8, 16] = 8,
    rows_per_threadgroup: Literal[2, 4, 8, 16, 32, 64, 128, 256] = 4,
    use_threadgroup_codebook: bool = True,
    use_decoded_codebook: bool = False,
    validate: bool = True,
) -> mx.array:
    x_mx = _as_mlx(x)
    codes_mx = _as_mlx(codes)
    scales_mx = _as_mlx(scales)
    rhs_mx = _as_mlx(rhs_indices)
    codebook_mx = mx.array(_default_codebook(code_bits)) if codebook is None else _as_mlx(codebook)

    if validate:
        _validate_gather_metadata(
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            rhs_mx,
            input_dims=input_dims,
            output_dims=output_dims,
            group_size=group_size,
            code_bits=code_bits,
            codebook_duplication=1,
        )
        if x_mx.shape[0] != 1:
            raise ValueError("gather_vqmm_m1_kernel requires exactly one token")
        if rhs_mx.ndim != 2:
            raise ValueError("rhs_indices must be 2D [1, top_k]")
        if rows_per_threadgroup not in (2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError("rows_per_threadgroup must be 2, 4, 8, 16, 32, 64, 128, or 256")

    if code_bits == 8 and use_decoded_codebook and rows_per_threadgroup == 32 and input_dims > output_dims:
        return gather_vqmm_m1_rowpair_kernel_unchecked(
            x_mx,
            codes_mx,
            scales_mx,
            codebook_mx,
            rhs_mx,
            output_dims=output_dims,
        )

    return gather_vqmm_m1_kernel_unchecked(
        x_mx,
        codes_mx,
        scales_mx,
        codebook_mx,
        rhs_mx,
        output_dims=output_dims,
        code_bits=code_bits,
        rows_per_threadgroup=rows_per_threadgroup,
        use_threadgroup_codebook=use_threadgroup_codebook,
        use_decoded_codebook=use_decoded_codebook,
    )
