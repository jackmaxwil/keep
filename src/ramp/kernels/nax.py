from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx
import numpy as np

from keep.vq.e8 import e8_1bit_packed, e8p_full_grid, e8p_packed_abs_grid


@cache
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _native_candidates() -> list[Path]:
    root = _repo_root() / "native" / "vq_nax_ext"
    # ``build/_vqnax*.so`` already matches the built extension, and a recursive
    # ``build/**/_vqnax*.so`` re-matched the identical path (``**`` matches zero
    # directories) at the cost of walking the whole build tree.
    patterns = [
        "build/_vqnax*.so",
        "_vqnax*.so",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(root.glob(pattern)))
    return candidates


@cache
def _kernel_dir() -> Path:
    return _repo_root() / "native" / "vq_nax_ext" / "kernels"


@cache
def _e8p_full_grid_array() -> mx.array:
    full_grid = mx.array(e8p_full_grid(), dtype=mx.float16)
    mx.eval(full_grid)
    return full_grid


def _load_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_vqnax", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load native VQ NAX module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Register the module so the ``importlib.import_module`` fast path in
    # ``load_native`` hits on every later call. Without this, each call
    # re-executed the 3.5 MB extension after globbing for it, which dominated
    # M=1 expert-projection time.
    sys.modules.setdefault("_vqnax", module)
    return module


@cache
def load_native() -> ModuleType:
    try:
        return importlib.import_module("_vqnax")
    except ImportError as import_error:
        for candidate in _native_candidates():
            try:
                return _load_from_path(candidate)
            except ImportError:
                continue
        raise ImportError(
            "Native VQ NAX extension is not built. Run: "
            "cmake -S native/vq_nax_ext -B native/vq_nax_ext/build "
            "-DPython_EXECUTABLE=.venv/bin/python3 && "
            "cmake --build native/vq_nax_ext/build"
        ) from import_error


@cache
def is_available() -> bool:
    # Cached separately from ``load_native``: ``functools.cache`` does not cache
    # exceptions, so without this the unbuilt-extension case re-globbed on every
    # dispatch. Caching also pins the dispatch decision for the life of the
    # process instead of letting a transient load failure silently switch
    # kernels mid-run.
    try:
        return bool(load_native().is_available())
    except ImportError:
        return False


def build_info() -> str:
    return str(load_native().build_info())


def predecoded_fp16_matmul(x: mx.array, weight_t: mx.array, *, stream: Any = None) -> mx.array:
    """Native build-smoke matmul for the future predecoded FP16 NAX primitive."""

    if stream is not None:
        raise ValueError("native NAX scaffold smoke does not accept an explicit stream yet")
    out = mx.array(0.0, dtype=mx.float16)
    load_native().predecoded_fp16_matmul_into(x, weight_t, out)
    return out


def predecoded_fp16_gather_mm(
    x: mx.array,
    weight_t: mx.array,
    rhs_indices: mx.array,
    *,
    stream: Any = None,
) -> mx.array:
    """Native build-smoke sorted gather_mm path for the predecoded FP16 NAX probe."""

    if stream is not None:
        raise ValueError("native NAX scaffold smoke does not accept an explicit stream yet")
    out = mx.array(0.0, dtype=mx.float16)
    load_native().predecoded_fp16_gather_mm_into(x, weight_t, rhs_indices, out)
    return out


def nax_fp16_matmul_tile(x: mx.array, weight_t: mx.array, *, stream: Any = None) -> mx.array:
    """Runtime-compiled Metal 4 TensorOps FP16 smoke tile.

    This intentionally supports only ``[32,16] @ [16,16]``. It proves the
    native extension can schedule an MLX primitive backed by an MPP/NAX Metal
    kernel without promoting the smoke tile as a routed prefill implementation.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold smoke does not accept an explicit stream yet")
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_fp16_matmul_tile_into(x, weight_t, str(_kernel_dir()), out)
    return out


def nax_e8_fp16_matmul_tile(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Runtime-compiled fused E8 TensorOps smoke tile.

    This supports only ``x [32,16]`` with ``codes [16,2]``. It decodes E8
    values directly while filling the B cooperative tensor, proving the N4
    no-scratch data path on a small fixed tile.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold smoke does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8_1bit_packed(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8_fp16_matmul_tile_into(
        x,
        codes,
        scales,
        codebook,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8_fp16_matmul(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Runtime-compiled fused E8 TensorOps dense matmul.

    This is the first reusable N4 shape: ``x [M,K]`` times an E8-coded
    ``weight [N,K]``. It is dense and single-expert; routed expert blocking is
    layered on top in a later slice.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8_1bit_packed(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8_fp16_matmul_into(
        x,
        codes,
        scales,
        codebook,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8_fp16_routed_matmul(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    lhs_indices: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Runtime-compiled fused E8 TensorOps sorted-routed matmul.

    Inputs are already sorted into same-expert route blocks. The output is
    `[routes, out]` in the same sorted route order.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8_1bit_packed(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8_fp16_routed_matmul_into(
        x,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8_fp16_routed_steel_matmul(
    x: mx.array,
    codes: mx.array,
    scales: mx.array,
    lhs_indices: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Steel ``NAXTile`` probe for fused E8 TensorOps sorted-routed matmul."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8_1bit_packed(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8_fp16_routed_steel_matmul_into(
        x,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8_fp16_sorted_steel_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Steel ``NAXTile`` probe over already sorted route activations."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8_1bit_packed(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8_fp16_sorted_steel_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Steel ``NAXTile`` probe over already sorted E8P route activations.

    This path mirrors ``nax_e8_fp16_sorted_steel_matmul`` but consumes uint16
    E8P codes and decodes through the packed absolute grid. Resident sorted
    prefill dispatch uses it for all-16-bit GLU layers through ``nax_e8p``.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_packed_rhs_tile_matmul(
    x: mx.array,
    code_tile: mx.array,
    scale_tile: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_count: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first native matmul over one compressed E8P RHS tile.

    ``code_tile`` and ``scale_tile`` are the compressed tiles emitted by
    ``pack_e8p_rhs_tiles``. This is the bridge oracle for the future sorted
    gather-qmm RHS NAX path: it proves the native extension can consume packed
    E8P tile codes and per-codeword scale slots without dense RHS materializing.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_packed_rhs_tile_matmul_into(
        x,
        code_tile,
        scale_tile,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        int(output_count),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_rhs_tile_matmul(
    x: mx.array,
    sign_tile: mx.array,
    abs_index_tile: mx.array,
    parity_tile: mx.array,
    scale_tile: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_count: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first native matmul over one split-byte E8P RHS tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_rhs_tile_matmul_into(
        x,
        sign_tile,
        abs_index_tile,
        parity_tile,
        scale_tile,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        int(output_count),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_nibble_abs_index_rhs_tile_matmul(
    x: mx.array,
    sign_low_nibble_tile: mx.array,
    sign_high_nibble_tile: mx.array,
    abs_index_tile: mx.array,
    parity_tile: mx.array,
    scale_tile: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_count: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first native matmul over one sign-nibble E8P RHS tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_nibble_abs_index_rhs_tile_matmul_into(
        x,
        sign_low_nibble_tile,
        sign_high_nibble_tile,
        abs_index_tile,
        parity_tile,
        scale_tile,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        int(output_count),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_plane_abs_index_rhs_tile_matmul(
    x: mx.array,
    sign_bit_planes: mx.array,
    abs_index_tile: mx.array,
    scale_tile: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_count: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first native matmul over one sign-plane E8P RHS tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_plane_abs_index_rhs_tile_matmul_into(
        x,
        sign_bit_planes,
        abs_index_tile,
        scale_tile,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        int(output_count),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul(
    x: mx.array,
    sign_low_nibble_lut: mx.array,
    sign_low_nibble_slots: mx.array,
    sign_high_nibble_lut: mx.array,
    sign_high_nibble_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tile: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_count: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first native matmul over one sign-nibble micro-LUT RHS tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_nibble_micro_lut_rhs_tile_matmul_into(
        x,
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tile,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        int(output_count),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_factor_reuse_rhs_tile_matmul(
    x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tile: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_count: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first native matmul over one factor-reuse split-byte E8P tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_factor_reuse_rhs_tile_matmul_into(
        x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tile,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        int(output_count),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_packed_rhs_expert_matmul(
    x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Compose native packed-RHS tile matmuls for one expert.

    This is a correctness bridge from the single-tile native oracle toward a
    routed packed-RHS path. It keeps the RHS in compressed tile storage and
    accumulates across bk64 tiles using ``nax_e8p_packed_rhs_tile_matmul``.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if code_tiles.ndim != 4 or scale_tiles.ndim != 4:
        raise ValueError("code_tiles and scale_tiles must be [n_tiles,k_blocks,bn,*]")
    if scale_group_indices.ndim != 2 or codeword_scale_slots.ndim != 2:
        raise ValueError("scale_group_indices and codeword_scale_slots must be [k_blocks,*]")
    n_tiles, k_blocks, bn, codewords = code_tiles.shape
    if x.shape[1] != k_blocks * codewords * 8:
        raise ValueError("x input dimension must match packed K blocks")
    if output_dims <= 0 or output_dims > n_tiles * bn:
        raise ValueError("output_dims must be positive and within packed N tiles")
    if scale_tiles.shape[0] != n_tiles or scale_tiles.shape[1] != k_blocks or scale_tiles.shape[2] != bn:
        raise ValueError("scale_tiles shape must match code_tiles [n_tiles,k_blocks,bn]")
    if scale_group_indices.shape[0] != k_blocks or codeword_scale_slots.shape[0] != k_blocks:
        raise ValueError("scale maps must have one row per K block")
    if codeword_scale_slots.shape[1] != codewords:
        raise ValueError("codeword_scale_slots must have one slot per codeword")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)

    bk = codewords * 8
    outputs: list[mx.array] = []
    for n_tile in range(n_tiles):
        out_start = n_tile * bn
        output_count = min(bn, output_dims - out_start)
        if output_count <= 0:
            break
        tile_accum: mx.array | None = None
        for k_block in range(k_blocks):
            k_start = k_block * bk
            partial = nax_e8p_packed_rhs_tile_matmul(
                mx.contiguous(x[:, k_start : k_start + bk]),
                mx.contiguous(code_tiles[n_tile, k_block]),
                mx.contiguous(scale_tiles[n_tile, k_block]),
                mx.contiguous(scale_group_indices[k_block]),
                mx.contiguous(codeword_scale_slots[k_block]),
                codebook,
                output_count=output_count,
            )
            tile_accum = partial if tile_accum is None else tile_accum + partial
        if tile_accum is None:
            raise ValueError("packed RHS expert matmul requires at least one K block")
        outputs.append(tile_accum)
    if not outputs:
        raise ValueError("packed RHS expert matmul produced no output tiles")
    return mx.concatenate(outputs, axis=1)


def nax_e8p_packed_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Correctness bridge from packed expert tiles to sorted-route output."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if code_tiles.ndim != 5 or scale_tiles.ndim != 5:
        raise ValueError("code_tiles and scale_tiles must be [experts,n_tiles,k_blocks,bn,*]")
    if tile_experts.ndim != 1 or tile_offsets.ndim != 1 or tile_counts.ndim != 1:
        raise ValueError("tile descriptors must be 1D arrays")
    if tile_offsets.shape != tile_experts.shape or tile_counts.shape != tile_experts.shape:
        raise ValueError("tile descriptor arrays must have matching shapes")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)

    experts_np = np.array(tile_experts, dtype=np.int32)
    offsets_np = np.array(tile_offsets, dtype=np.int32)
    counts_np = np.array(tile_counts, dtype=np.int32)
    route_outputs: list[mx.array] = []
    for expert, offset, count in zip(experts_np, offsets_np, counts_np, strict=True):
        if count <= 0:
            continue
        if expert < 0 or expert >= code_tiles.shape[0]:
            raise ValueError("tile_experts contains an out-of-range expert id")
        if offset < 0 or offset + count > sorted_x.shape[0]:
            raise ValueError("tile descriptor offsets/counts exceed sorted_x routes")
        route_outputs.append(
            nax_e8p_packed_rhs_expert_matmul(
                mx.contiguous(sorted_x[int(offset) : int(offset + count)]),
                code_tiles[int(expert)],
                scale_tiles[int(expert)],
                scale_group_indices,
                codeword_scale_slots,
                codebook,
                output_dims=output_dims,
            )
        )
    if not route_outputs:
        return mx.zeros((0, int(output_dims)), dtype=mx.float16)
    return mx.concatenate(route_outputs, axis=0)


def nax_e8p_packed_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over the packed RHS tile layout."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_packed_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    sign_tiles: mx.array,
    abs_index_tiles: mx.array,
    parity_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over the split-byte RHS tile layout."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_rhs_sorted_matmul_into(
        sorted_x,
        sign_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_nibble_abs_index_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    sign_low_nibble_tiles: mx.array,
    sign_high_nibble_tiles: mx.array,
    abs_index_tiles: mx.array,
    parity_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over sign-nibble/abs-index RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_nibble_abs_index_rhs_sorted_matmul_into(
        sorted_x,
        sign_low_nibble_tiles,
        sign_high_nibble_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_plane_abs_index_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    sign_bit_planes: mx.array,
    abs_index_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over sign-plane/abs-index RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_plane_abs_index_rhs_sorted_matmul_into(
        sorted_x,
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul(
    sorted_x: mx.array,
    sign_bit_planes: mx.array,
    abs_index_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native TensorOps traversal over sign-plane/abs-index RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_plane_abs_index_rhs_sorted_tensorops_matmul_into(
        sorted_x,
        sign_bit_planes,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_nibble_micro_lut_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    sign_low_nibble_lut: mx.array,
    sign_low_nibble_slots: mx.array,
    sign_high_nibble_lut: mx.array,
    sign_high_nibble_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over sign-nibble micro-LUT RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_nibble_micro_lut_rhs_sorted_matmul_into(
        sorted_x,
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul(
    sorted_x: mx.array,
    sign_low_nibble_tiles: mx.array,
    sign_high_nibble_tiles: mx.array,
    abs_index_tiles: mx.array,
    parity_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native TensorOps traversal over sign-nibble/abs-index RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_nibble_abs_index_rhs_sorted_tensorops_matmul_into(
        sorted_x,
        sign_low_nibble_tiles,
        sign_high_nibble_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul(
    sorted_x: mx.array,
    sign_low_nibble_lut: mx.array,
    sign_low_nibble_slots: mx.array,
    sign_high_nibble_lut: mx.array,
    sign_high_nibble_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native TensorOps traversal over sign-nibble micro-LUT RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_sign_nibble_micro_lut_rhs_sorted_tensorops_matmul_into(
        sorted_x,
        sign_low_nibble_lut,
        sign_low_nibble_slots,
        sign_high_nibble_lut,
        sign_high_nibble_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_factor_reuse_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over factor-reuse split-byte RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_factor_reuse_rhs_sorted_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_expert_kblock_factor_reuse_rhs_sorted_native_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native sorted-route traversal over expert/K-block factor-reuse RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_expert_kblock_factor_reuse_rhs_sorted_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_component_stream_rhs_sorted_scalar_matmul(
    sorted_x: mx.array,
    sign_component_bits: mx.array,
    abs_index_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    component_scale_slots: mx.array,
    component_codeword_indices: mx.array,
    component_offsets: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native scalar sorted-route traversal over component-stream RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_component_stream_rhs_sorted_scalar_matmul_into(
        sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-slot codeword-stream RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_slot_codeword_stream_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-slot MMA codeword-tile RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_slot_mma_codeword_tile_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    active_route_tiles: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for active-route-tile codeword outer products."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_active_route_tile_codeword_outer_product_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        active_route_tiles,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    expert_cohort_offsets: mx.array,
    expert_cohort_counts: mx.array,
    route_cohort_offsets: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for expert-cohort codeword broadcasts."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_expert_cohort_codeword_broadcast_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        expert_cohort_offsets,
        expert_cohort_counts,
        route_cohort_offsets,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    route_batch_segment_offsets: mx.array,
    route_batch_segment_counts: mx.array,
    route_batch_route_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-batch segmented codeword reduces."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_batch_segmented_codeword_reduce_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_batch_segment_offsets,
        route_batch_segment_counts,
        route_batch_route_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_cohort_offsets: mx.array,
    token_cohort_counts: mx.array,
    token_cohort_active_expert_ids: mx.array,
    token_cohort_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-cohort codeword streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_cohort_codeword_stream_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_cohort_offsets,
        token_cohort_counts,
        token_cohort_active_expert_ids,
        token_cohort_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_cohort_offsets: mx.array,
    token_cohort_counts: mx.array,
    token_cohort_active_expert_ids: mx.array,
    token_cohort_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-cohort MMA codeword tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_cohort_mma_codeword_tile_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_cohort_offsets,
        token_cohort_counts,
        token_cohort_active_expert_ids,
        token_cohort_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    output_stationary_route_batch_offsets: mx.array,
    output_stationary_route_batch_counts: mx.array,
    output_stationary_route_batch_active_expert_ids: mx.array,
    output_stationary_route_batch_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for output-stationary codeword tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_output_stationary_codeword_tile_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        output_stationary_route_batch_offsets,
        output_stationary_route_batch_counts,
        output_stationary_route_batch_active_expert_ids,
        output_stationary_route_batch_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    input_stationary_route_batch_offsets: mx.array,
    input_stationary_route_batch_counts: mx.array,
    input_stationary_route_batch_active_expert_ids: mx.array,
    input_stationary_route_batch_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for input-stationary codeword tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_input_stationary_codeword_tile_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        input_stationary_route_batch_offsets,
        input_stationary_route_batch_counts,
        input_stationary_route_batch_active_expert_ids,
        input_stationary_route_batch_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_factor_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for expert/K-block codeword factor reuse."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_expert_kblock_codeword_factor_reuse_rhs_sorted_matmul_into(
        sorted_x,
        codeword_factor_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul(
    route_local_codeword_dot_lut: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    route_codeword_lut_route_slots: mx.array,
    route_codeword_lut_offsets: mx.array,
    route_codeword_lut_counts: mx.array,
    route_codeword_lut_codeword_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-codeword LUT accumulation."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_codeword_lut_accumulate_rhs_sorted_matmul_into(
        route_local_codeword_dot_lut,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_codeword_lut_route_slots,
        route_codeword_lut_offsets,
        route_codeword_lut_counts,
        route_codeword_lut_codeword_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    rowwise_route_microtile_offsets: mx.array,
    rowwise_route_microtile_counts: mx.array,
    rowwise_route_microtile_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for rowwise codeword-tile accumulation."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_rowwise_codeword_tile_accumulate_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        rowwise_route_microtile_offsets,
        rowwise_route_microtile_counts,
        rowwise_route_microtile_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    output_tile_local_route_microtile_offsets: mx.array,
    output_tile_local_route_microtile_counts: mx.array,
    output_tile_local_route_microtile_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for output-tile-local codeword LUT accumulation."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_output_tile_local_codeword_lut_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        output_tile_local_route_microtile_offsets,
        output_tile_local_route_microtile_counts,
        output_tile_local_route_microtile_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    route_microtile_codeword_block_reduce_offsets: mx.array,
    route_microtile_codeword_block_reduce_counts: mx.array,
    route_microtile_codeword_block_reduce_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-microtile codeword-block reduction."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_microtile_codeword_block_reduce_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_microtile_codeword_block_reduce_offsets,
        route_microtile_codeword_block_reduce_counts,
        route_microtile_codeword_block_reduce_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    kblock_wavefront_codeword_scan_offsets: mx.array,
    kblock_wavefront_codeword_scan_counts: mx.array,
    kblock_wavefront_codeword_scan_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for k-block wavefront codeword scans."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_kblock_wavefront_codeword_scan_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        kblock_wavefront_codeword_scan_offsets,
        kblock_wavefront_codeword_scan_counts,
        kblock_wavefront_codeword_scan_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_route_output_stripe_offsets: mx.array,
    token_route_output_stripe_counts: mx.array,
    token_route_output_stripe_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-route output-stripe pipelines."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_route_output_stripe_pipeline_rhs_sorted_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_route_output_stripe_offsets,
        token_route_output_stripe_counts,
        token_route_output_stripe_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for expert/K-block scale-slot streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_expert_kblock_scale_slot_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    scale_group_route_block_offsets: mx.array,
    scale_group_route_block_counts: mx.array,
    scale_group_route_block_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for scale-group route-block reductions."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_scale_group_route_block_reduce_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        scale_group_route_block_offsets,
        scale_group_route_block_counts,
        scale_group_route_block_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_block_output_group_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    route_block_output_group_offsets: mx.array,
    route_block_output_group_counts: mx.array,
    route_block_output_group_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-block output-group streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_block_output_group_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_block_output_group_offsets,
        route_block_output_group_counts,
        route_block_output_group_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    output_group_pretransposed_route_offsets: mx.array,
    output_group_pretransposed_route_counts: mx.array,
    output_group_pretransposed_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for output-group-pretransposed streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_output_group_pretransposed_codeword_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        output_group_pretransposed_route_offsets,
        output_group_pretransposed_route_counts,
        output_group_pretransposed_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    kblock_route_fused_offsets: mx.array,
    kblock_route_fused_counts: mx.array,
    kblock_route_fused_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for K-block route-fused streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_kblock_output_group_route_fused_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        kblock_route_fused_offsets,
        kblock_route_fused_counts,
        kblock_route_fused_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    route_tile_output_swizzle_offsets: mx.array,
    route_tile_output_swizzle_counts: mx.array,
    route_tile_output_swizzle_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for route-tile output-swizzle streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_route_tile_output_swizzle_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_tile_output_swizzle_offsets,
        route_tile_output_swizzle_counts,
        route_tile_output_swizzle_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_topk_offsets: mx.array,
    token_topk_counts: mx.array,
    token_topk_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token top-k output-tile streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_topk_output_tile_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_topk_offsets,
        token_topk_counts,
        token_topk_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_block_output_group_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_block_offsets: mx.array,
    token_block_counts: mx.array,
    token_block_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-block output-group streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_block_output_group_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_block_offsets,
        token_block_counts,
        token_block_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_output_stripe_offsets: mx.array,
    token_output_stripe_counts: mx.array,
    token_output_stripe_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token output-stripe group streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_output_stripe_group_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_output_stripe_offsets,
        token_output_stripe_counts,
        token_output_stripe_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_expert_output_block_offsets: mx.array,
    token_expert_output_block_counts: mx.array,
    token_expert_output_block_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-expert output-block streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_expert_output_block_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_expert_output_block_offsets,
        token_expert_output_block_counts,
        token_expert_output_block_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_pair_kblock_offsets: mx.array,
    token_pair_kblock_counts: mx.array,
    token_pair_kblock_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair K-block accumulator streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_kblock_accumulator_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_pair_kblock_offsets,
        token_pair_kblock_counts,
        token_pair_kblock_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_pair_output_group_offsets: mx.array,
    token_pair_output_group_counts: mx.array,
    token_pair_output_group_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair output-group streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_output_group_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_pair_output_group_offsets,
        token_pair_output_group_counts,
        token_pair_output_group_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_pair_slot_topk_output_group_offsets: mx.array,
    token_pair_slot_topk_output_group_counts: mx.array,
    token_pair_slot_topk_output_group_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair slot/top-k output-group streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_slot_topk_output_group_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_pair_slot_topk_output_group_offsets,
        token_pair_slot_topk_output_group_counts,
        token_pair_slot_topk_output_group_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    token_pair_slot_topk_codeword_group_pipeline_offsets: mx.array,
    token_pair_slot_topk_codeword_group_pipeline_counts: mx.array,
    token_pair_slot_topk_codeword_group_pipeline_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair slot/top-k codeword-group pipelines."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_slot_topk_codeword_group_pipeline_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        token_pair_slot_topk_codeword_group_pipeline_offsets,
        token_pair_slot_topk_codeword_group_pipeline_counts,
        token_pair_slot_topk_codeword_group_pipeline_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    scale_slot_broadcast_offsets: mx.array,
    scale_slot_broadcast_counts: mx.array,
    scale_slot_broadcast_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair slot/top-k scale-slot broadcast streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_slot_topk_scale_slot_broadcast_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        scale_slot_broadcast_offsets,
        scale_slot_broadcast_counts,
        scale_slot_broadcast_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    route_bucket_offsets: mx.array,
    route_bucket_counts: mx.array,
    route_bucket_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair slot/top-k route-bucket reductions."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_slot_topk_route_bucket_codeword_reduce_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        route_bucket_offsets,
        route_bucket_counts,
        route_bucket_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    kblock_microtile_offsets: mx.array,
    kblock_microtile_counts: mx.array,
    kblock_microtile_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair slot/top-k K-block microtile streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_slot_topk_kblock_microtile_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        kblock_microtile_offsets,
        kblock_microtile_counts,
        kblock_microtile_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul(
    sorted_x: mx.array,
    codeword_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    output_tile_fused_offsets: mx.array,
    output_tile_fused_counts: mx.array,
    output_tile_fused_route_slot_ids: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native source guardrail for token-pair slot/top-k output-tile fused streams."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_token_pair_slot_topk_output_tile_fused_stream_rhs_sorted_matmul_into(
        sorted_x,
        codeword_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        output_tile_fused_offsets,
        output_tile_fused_counts,
        output_tile_fused_route_slot_ids,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_component_stream_rhs_sorted_partial_matmul(
    sorted_x: mx.array,
    sign_component_bits: mx.array,
    abs_index_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    component_scale_slots: mx.array,
    component_codeword_indices: mx.array,
    component_offsets: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native two-pass component-stream RHS partial-reduction scaffold."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_component_stream_rhs_sorted_partial_matmul_into(
        sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_component_stream_rhs_sorted_tensorops_matmul(
    sorted_x: mx.array,
    sign_component_bits: mx.array,
    abs_index_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    component_scale_slots: mx.array,
    component_codeword_indices: mx.array,
    component_offsets: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native TensorOps component-stream RHS sorted-route parity candidate."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_component_stream_rhs_sorted_tensorops_matmul_into(
        sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_component_stream_rhs_sorted_shared_decode_matmul(
    sorted_x: mx.array,
    sign_component_bits: mx.array,
    abs_index_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    component_scale_slots: mx.array,
    component_codeword_indices: mx.array,
    component_offsets: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native component-stream shared-decode cache parity scaffold."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_component_stream_rhs_sorted_shared_decode_matmul_into(
        sorted_x,
        sign_component_bits,
        abs_index_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        component_scale_slots,
        component_codeword_indices,
        component_offsets,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native q2-shaped TensorOps traversal over expert/K-block factor LUTs."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Correctness-first v2 traversal over expert/K-block factor LUTs."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_expert_kblock_factor_reuse_rhs_sorted_tensorops_v2_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_packed_rhs_sorted_tiled_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native bn64/bk64 tiled traversal over sorted packed RHS E8P tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_packed_rhs_sorted_tiled_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_rhs_sorted_tiled_matmul(
    sorted_x: mx.array,
    sign_tiles: mx.array,
    abs_index_tiles: mx.array,
    parity_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native bn64/bk64 tiled traversal over sorted split-byte RHS E8P tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_rhs_sorted_tiled_matmul_into(
        sorted_x,
        sign_tiles,
        abs_index_tiles,
        parity_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native bn64/bk64 tiled traversal over sorted factor-reuse split-byte RHS tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_factor_reuse_rhs_sorted_tiled_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native TensorOps factor-reuse traversal without decoded-B Ws staging."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_decode_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul(
    sorted_x: mx.array,
    sign_byte_lut: mx.array,
    sign_byte_slots: mx.array,
    abs_index_lut: mx.array,
    abs_index_slots: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native TensorOps factor-reuse traversal sharing decoded B by output half-tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_split_byte_factor_reuse_rhs_sorted_shared_n_decode_matmul_into(
        sorted_x,
        sign_byte_lut,
        sign_byte_slots,
        abs_index_lut,
        abs_index_slots,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_packed_rhs_sorted_tiled_m128_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native bn64/bk64 tiled traversal reusing each decoded RHS tile over 128 routes."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_packed_rhs_sorted_tiled_m128_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_packed_rhs_sorted_tiled_k128_matmul(
    sorted_x: mx.array,
    code_tiles: mx.array,
    scale_tiles: mx.array,
    scale_group_indices: mx.array,
    codeword_scale_slots: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    output_dims: int,
    stream: Any = None,
) -> mx.array:
    """Native packed RHS traversal staging two bk64 blocks per barrier."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_packed_rhs_sorted_tiled_k128_matmul_into(
        sorted_x,
        code_tiles,
        scale_tiles,
        scale_group_indices,
        codeword_scale_slots,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(output_dims),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_direct_reduce_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted path with no decoded-B threadgroup staging."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_direct_reduce_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_inline_b_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted TensorOps path with inline B-tile decode."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_inline_b_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_gs352_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path specialized for Air down K=1408."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if group_size != 352:
        raise ValueError("nax_e8p_fp16_sorted_steel_gs352_matmul requires group_size=352")
    if sorted_x.shape[1] != 1408:
        raise ValueError("nax_e8p_fp16_sorted_steel_gs352_matmul requires input_dims=1408")
    if scales.shape[2] != 4:
        raise ValueError("nax_e8p_fp16_sorted_steel_gs352_matmul requires four scale groups")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_gs352_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_lut_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    full_grid: mx.array | None = None,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with a predecoded full-grid LUT."""

    del codebook
    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if full_grid is None:
        full_grid = _e8p_full_grid_array()
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_lut_matmul_into(
        sorted_x,
        codes,
        scales,
        full_grid,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_tgcb_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with a threadgroup codebook."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_tgcb_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_tgscale_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with cached threadgroup scales."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_tgscale_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with cached codebook and scales."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_tgcb_tgscale_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with TG codebook and hoisted decode."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_tgcb_hoist_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_bk128_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with a wider K staging tile."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_bk128_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_m128n32_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with 128-route by 32-column tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_m128n32_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_m64n128_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with 64-route by 128-column tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_m64n128_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_m64n64t64_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P Steel path with 64-route tiles and two compute simdgroups."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_m64n64t64_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_m32n64_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with 32-route by 64-column tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_m32n64_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_m32n64t128_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only M32/N64 E8P Steel path with 128 staging threads."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_m32n64t128_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8p_fp16_sorted_steel_m32n128_matmul(
    sorted_x: mx.array,
    codes: mx.array,
    scales: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Benchmark-only E8P sorted Steel path with 32-route by 128-column tiles."""

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8p_packed_abs_grid(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8p_fp16_sorted_steel_m32n128_matmul_into(
        sorted_x,
        codes,
        scales,
        codebook,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out


def nax_e8_int8_routed_matmul(
    x_q: mx.array,
    x_scales: mx.array,
    codes: mx.array,
    scales: mx.array,
    lhs_indices: mx.array,
    tile_experts: mx.array,
    tile_offsets: mx.array,
    tile_counts: mx.array,
    codebook: mx.array | None = None,
    *,
    group_size: int,
    stream: Any = None,
) -> mx.array:
    """Runtime-compiled fused E8 INT8 TensorOps sorted-routed matmul.

    ``x_q`` is prequantized int8 activation data and ``x_scales`` is the
    matching per-token/per-group dequant scale. This is an N6 probe path; it is
    intentionally explicit about the A8 contract so quality experiments can
    swap quantizers without changing the native primitive.
    """

    if stream is not None:
        raise ValueError("native NAX scaffold does not accept an explicit stream yet")
    if codebook is None:
        codebook = mx.array(e8_1bit_packed(), dtype=mx.uint32)
    out = mx.array(0.0, dtype=mx.float16)
    load_native().nax_e8_int8_routed_matmul_into(
        x_q,
        x_scales,
        codes,
        scales,
        codebook,
        lhs_indices,
        tile_experts,
        tile_offsets,
        tile_counts,
        int(group_size),
        str(_kernel_dir()),
        out,
    )
    return out
