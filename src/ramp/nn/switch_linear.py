from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from keep.vq.e8 import CODEWORD_DIM, e8_1bit_packed, e8p_packed_abs_grid
from ramp.kernels.gather_vqmm import (
    gather_vqmm_m1_kernel,
    gather_vqmm_m1_per_route_kernel_unchecked,
    m1_rows_per_threadgroup,
    m1_use_decoded_codebook,
    m1_use_threadgroup_codebook,
)
from ramp.ops.vq_switch import RouteStrategy
from ramp.ops.vq_switch import gather_vqmm, vq_switch_qmv
from keep.quant.rotation import apply_rotation_mx, apply_rotation_np, validate_rotation_matrix_mx
from keep.quant.rht import apply_rht_mx, apply_rht_np
from keep.quant.rtn import quantize_weight_rtn


class HighPrecisionSwitchLinear(nn.Module):
    """Switch-linear layer backed by stacked dense expert tensors."""

    def __init__(
        self,
        *,
        weight: mx.array,
        bias: mx.array | None = None,
        tier: str = "high",
    ):
        super().__init__()
        if weight.ndim != 3:
            raise ValueError(f"weight must be 3D [experts, out, in], found {weight.shape}")
        num_experts, output_dims, input_dims = weight.shape
        if input_dims <= 0 or output_dims <= 0 or num_experts <= 0:
            raise ValueError("weight dimensions must be positive")
        if bias is not None and bias.shape != (num_experts, output_dims):
            raise ValueError(f"bias must have shape ({num_experts}, {output_dims}), found {bias.shape}")

        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.num_experts = int(num_experts)
        self.group_size = None
        self.code_bits = None
        self.use_gather_vqmm = False
        self.route_strategy = "direct"
        self.gather_codebook_duplication = 1
        self.tier = tier
        self.weight = weight
        if bias is not None:
            self.bias = bias
        self.freeze()

    @property
    def route_backend(self) -> str:
        return "dense_high_precision"

    @property
    def has_continuous_sidecar(self) -> bool:
        return False

    @property
    def has_sparse_residual_rows(self) -> bool:
        return False

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        del sorted_indices
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"x trailing dimension must be {self.input_dims}, found {x.shape[-1]}")
        indices = indices.astype(mx.int32)
        selected_weight = self.weight[indices].astype(mx.float32)
        x_float = x.astype(mx.float32)
        if x.shape[:-1] == indices.shape[:-1]:
            x_float = mx.expand_dims(mx.expand_dims(x_float, axis=-2), axis=-2)
        elif x.shape[:-1] == indices.shape:
            x_float = mx.expand_dims(x_float, axis=-2)
        else:
            raise ValueError(
                "x leading dimensions must match indices for per-route input or "
                "indices leading dimensions for token input"
            )
        y = mx.sum(selected_weight * x_float, axis=-1)
        bias = self.get("bias")
        if bias is not None:
            y = y + bias[indices].astype(y.dtype)
        return y


class QuantizedVQSwitchLinear(nn.Module):
    """Switch-linear layer backed by stacked E8 VQ expert tensors."""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        codes: mx.array,
        scales: mx.array,
        codebook: mx.array | None = None,
        bias: mx.array | None = None,
        group_size: int = 512,
        code_bits: Literal[8, 16] = 8,
        use_gather_vqmm: bool = True,
        route_strategy: RouteStrategy = "auto",
        gather_codebook_duplication: Literal[1, 4, 8] = 8,
        continuous_scale_delta: mx.array | None = None,
        continuous_output_bias: mx.array | None = None,
        continuous_low_rank_left: mx.array | None = None,
        continuous_low_rank_right: mx.array | None = None,
        sparse_residual_expert_indices: mx.array | None = None,
        sparse_residual_output_indices: mx.array | None = None,
        sparse_residual_values: mx.array | None = None,
        rht_signs: mx.array | None = None,
        rotation_matrix: mx.array | None = None,
    ):
        super().__init__()
        if input_dims <= 0 or output_dims <= 0 or num_experts <= 0:
            raise ValueError("input_dims, output_dims, and num_experts must be positive")
        if input_dims % CODEWORD_DIM != 0:
            raise ValueError("input_dims must be divisible by 8")
        if input_dims % group_size != 0:
            raise ValueError("input_dims must be divisible by group_size")
        if code_bits not in (8, 16):
            raise ValueError("code_bits must be 8 or 16")
        if gather_codebook_duplication not in (1, 4, 8):
            raise ValueError("gather_codebook_duplication must be 1, 4, or 8")
        # "per_route_decoded" is accepted and passed straight through to
        # gather_vqmm. The M=1 fast-path guards below deliberately test for
        # ("auto", "direct") only, so this strategy always reaches the
        # per-route decoded kernel -- which is byte-exact with those fast paths
        # anyway. This is the strategy Wave 5's MTP verify should select.
        if route_strategy not in ("auto", "direct", "sorted_tiled", "per_route_decoded"):
            raise ValueError(
                "route_strategy must be 'auto', 'direct', 'sorted_tiled', or 'per_route_decoded'"
            )

        expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
        expected_scales = (num_experts, output_dims, input_dims // group_size)
        if codes.shape != expected_codes:
            raise ValueError(f"codes must have shape {expected_codes}, found {codes.shape}")
        if scales.shape != expected_scales:
            raise ValueError(f"scales must have shape {expected_scales}, found {scales.shape}")
        if bias is not None and bias.shape != (num_experts, output_dims):
            raise ValueError(f"bias must have shape ({num_experts}, {output_dims}), found {bias.shape}")
        if continuous_scale_delta is not None and continuous_scale_delta.shape != expected_scales:
            raise ValueError(
                "continuous_scale_delta must have shape "
                f"{expected_scales}, found {continuous_scale_delta.shape}"
            )
        if continuous_output_bias is not None and continuous_output_bias.shape != (num_experts, output_dims):
            raise ValueError(
                "continuous_output_bias must have shape "
                f"({num_experts}, {output_dims}), found {continuous_output_bias.shape}"
            )
        if (continuous_low_rank_left is None) != (continuous_low_rank_right is None):
            raise ValueError("continuous low-rank sidecar requires both left and right tensors")
        if continuous_low_rank_left is not None and continuous_low_rank_right is not None:
            _validate_low_rank_residual(
                left=continuous_low_rank_left,
                right=continuous_low_rank_right,
                input_dims=input_dims,
                output_dims=output_dims,
                num_experts=num_experts,
            )
        if rht_signs is not None and rotation_matrix is not None:
            raise ValueError("rht_signs and rotation_matrix are mutually exclusive")
        if rht_signs is not None and rht_signs.shape != (input_dims,):
            raise ValueError(f"rht_signs must have shape ({input_dims},), found {rht_signs.shape}")
        if rotation_matrix is not None:
            validate_rotation_matrix_mx(rotation_matrix, dim=input_dims)
        _validate_sparse_residual_rows(
            expert_indices=sparse_residual_expert_indices,
            output_indices=sparse_residual_output_indices,
            values=sparse_residual_values,
            input_dims=input_dims,
            output_dims=output_dims,
            num_experts=num_experts,
        )

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.group_size = group_size
        self.code_bits = code_bits
        self.use_gather_vqmm = use_gather_vqmm
        self.route_strategy = route_strategy
        self.gather_codebook_duplication = gather_codebook_duplication
        self.codes = codes
        self.scales = scales
        if codebook is None:
            codebook = mx.array(e8_1bit_packed() if code_bits == 8 else e8p_packed_abs_grid())
        self.codebook = codebook
        if bias is not None:
            self.bias = bias
        if continuous_scale_delta is not None:
            self.continuous_scale_delta = continuous_scale_delta.astype(mx.float32)
        if continuous_output_bias is not None:
            self.continuous_output_bias = continuous_output_bias.astype(mx.float32)
        if continuous_low_rank_left is not None and continuous_low_rank_right is not None:
            self.continuous_low_rank_left = continuous_low_rank_left.astype(mx.float32)
            self.continuous_low_rank_right = continuous_low_rank_right.astype(mx.float32)
        if sparse_residual_values is not None:
            assert sparse_residual_expert_indices is not None
            assert sparse_residual_output_indices is not None
            self.sparse_residual_expert_indices = sparse_residual_expert_indices.astype(mx.int32)
            self.sparse_residual_output_indices = sparse_residual_output_indices.astype(mx.int32)
            self.sparse_residual_values = sparse_residual_values.astype(mx.float32)
        if rht_signs is not None:
            self.rht_signs = rht_signs.astype(mx.float32)
        if rotation_matrix is not None:
            self.rotation_matrix = rotation_matrix.astype(mx.float32)
        self.freeze()

    @property
    def route_backend(self) -> str:
        if not self.use_gather_vqmm:
            return "scalar_qmv"
        return f"gather_vqmm_{self.route_strategy}"

    @property
    def has_continuous_sidecar(self) -> bool:
        return (
            self.get("continuous_scale_delta") is not None
            or self.get("continuous_output_bias") is not None
            or self.get("continuous_low_rank_left") is not None
        )

    @property
    def has_sparse_residual_rows(self) -> bool:
        return self.get("sparse_residual_values") is not None

    def set_continuous_sidecar(
        self,
        *,
        scale_delta: mx.array | None = None,
        output_bias: mx.array | None = None,
        low_rank_left: mx.array | None = None,
        low_rank_right: mx.array | None = None,
    ) -> None:
        if scale_delta is not None and scale_delta.shape != self.scales.shape:
            raise ValueError(f"scale_delta must have shape {self.scales.shape}, found {scale_delta.shape}")
        if output_bias is not None and output_bias.shape != (self.num_experts, self.output_dims):
            raise ValueError(
                f"output_bias must have shape ({self.num_experts}, {self.output_dims}), "
                f"found {output_bias.shape}"
            )
        if (low_rank_left is None) != (low_rank_right is None):
            raise ValueError("low-rank sidecar requires both low_rank_left and low_rank_right")
        if low_rank_left is not None and low_rank_right is not None:
            _validate_low_rank_residual(
                left=low_rank_left,
                right=low_rank_right,
                input_dims=self.input_dims,
                output_dims=self.output_dims,
                num_experts=self.num_experts,
            )
        if scale_delta is None:
            if self.get("continuous_scale_delta") is not None:
                delattr(self, "continuous_scale_delta")
        else:
            self.continuous_scale_delta = scale_delta.astype(mx.float32)
        if output_bias is None:
            if self.get("continuous_output_bias") is not None:
                delattr(self, "continuous_output_bias")
        else:
            self.continuous_output_bias = output_bias.astype(mx.float32)
        if low_rank_left is None:
            if self.get("continuous_low_rank_left") is not None:
                delattr(self, "continuous_low_rank_left")
            if self.get("continuous_low_rank_right") is not None:
                delattr(self, "continuous_low_rank_right")
        else:
            assert low_rank_right is not None
            self.continuous_low_rank_left = low_rank_left.astype(mx.float32)
            self.continuous_low_rank_right = low_rank_right.astype(mx.float32)

    def effective_scales(self) -> mx.array:
        scale_delta = self.get("continuous_scale_delta")
        if scale_delta is None:
            return self.scales
        return self.scales.astype(mx.float32) * mx.exp(scale_delta.astype(mx.float32))

    def set_sparse_residual_rows(
        self,
        *,
        expert_indices: mx.array | None = None,
        output_indices: mx.array | None = None,
        values: mx.array | None = None,
    ) -> None:
        _validate_sparse_residual_rows(
            expert_indices=expert_indices,
            output_indices=output_indices,
            values=values,
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            num_experts=self.num_experts,
        )
        if values is None:
            for name in (
                "sparse_residual_expert_indices",
                "sparse_residual_output_indices",
                "sparse_residual_values",
            ):
                if self.get(name) is not None:
                    delattr(self, name)
            return
        assert expert_indices is not None
        assert output_indices is not None
        self.sparse_residual_expert_indices = expert_indices.astype(mx.int32)
        self.sparse_residual_output_indices = output_indices.astype(mx.int32)
        self.sparse_residual_values = values.astype(mx.float32)

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        scales = self.effective_scales()
        rht_signs = self.get("rht_signs")
        if rht_signs is not None:
            x = apply_rht_mx(x, rht_signs)
        rotation_matrix = self.get("rotation_matrix")
        if rotation_matrix is not None:
            x = apply_rotation_mx(x, rotation_matrix)
        if (
            self.use_gather_vqmm
            and x.ndim >= 2
            and indices.ndim >= 2
            and x.shape[:-1] == indices.shape[:-1]
        ):
            leading_shape = x.shape[:-1]
            top_k = indices.shape[-1]
            flat_x = x.reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1, top_k))
            if (
                flat_x.shape[0] == 1
                and self.code_bits == 8
                and not sorted_indices
                and self.route_strategy in ("auto", "direct")
            ):
                y = gather_vqmm_m1_kernel(
                    flat_x,
                    self.codes,
                    scales,
                    self.codebook,
                    flat_indices,
                    input_dims=self.input_dims,
                    output_dims=self.output_dims,
                    group_size=self.group_size,
                    code_bits=self.code_bits,
                    rows_per_threadgroup=m1_rows_per_threadgroup(self.input_dims, self.output_dims),
                    use_threadgroup_codebook=m1_use_threadgroup_codebook(self.input_dims, self.output_dims),
                    use_decoded_codebook=m1_use_decoded_codebook(self.input_dims, self.output_dims),
                    validate=False,
                )
            else:
                y = gather_vqmm(
                    flat_x,
                    self.codes,
                    scales,
                    self.codebook,
                    flat_indices,
                    input_dims=self.input_dims,
                    output_dims=self.output_dims,
                    group_size=self.group_size,
                    code_bits=self.code_bits,
                    implementation="metal",
                    sorted_indices=sorted_indices,
                    route_strategy=self.route_strategy,
                    codebook_duplication=self.gather_codebook_duplication,
                )
            y = y.reshape((*leading_shape, top_k, self.output_dims))
        elif self.use_gather_vqmm and x.ndim >= 2 and indices.ndim >= 1 and x.shape[:-1] == indices.shape:
            route_shape = indices.shape
            flat_x = x.reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1,))
            if (
                route_shape[0] == 1
                and flat_x.shape[0] <= 8
                and self.code_bits == 8
                and not sorted_indices
                and self.route_strategy in ("auto", "direct")
            ):
                y = gather_vqmm_m1_per_route_kernel_unchecked(
                    flat_x,
                    self.codes,
                    scales,
                    self.codebook,
                    flat_indices,
                    output_dims=self.output_dims,
                    rows_per_threadgroup=32,
                ).reshape((*route_shape, self.output_dims))
            else:
                lhs_indices = mx.arange(flat_x.shape[0], dtype=mx.int32)
                y = gather_vqmm(
                    flat_x,
                    self.codes,
                    scales,
                    self.codebook,
                    flat_indices,
                    lhs_indices=lhs_indices,
                    input_dims=self.input_dims,
                    output_dims=self.output_dims,
                    group_size=self.group_size,
                    code_bits=self.code_bits,
                    implementation="metal",
                    sorted_indices=sorted_indices,
                    route_strategy=self.route_strategy,
                    codebook_duplication=self.gather_codebook_duplication,
                ).reshape((*route_shape, self.output_dims))
        else:
            y = vq_switch_qmv(
                x,
                self.codes,
                scales,
                self.codebook,
                indices,
                input_dims=self.input_dims,
                output_dims=self.output_dims,
                group_size=self.group_size,
                code_bits=self.code_bits,
                sorted_indices=sorted_indices,
            )
        y = self._apply_low_rank_residual(y, x, indices)
        y = self._apply_sparse_residual_rows(y, x, indices)
        bias = self.get("bias")
        if bias is not None:
            y = y + bias[indices]
        output_bias = self.get("continuous_output_bias")
        if output_bias is not None:
            y = y + output_bias[indices]
        return y

    def _apply_low_rank_residual(self, y: mx.array, x: mx.array, indices: mx.array) -> mx.array:
        left = self.get("continuous_low_rank_left")
        right = self.get("continuous_low_rank_right")
        if left is None or right is None:
            return y
        if x.shape[:-1] == indices.shape[:-1]:
            leading_shape = indices.shape[:-1]
            top_k = indices.shape[-1]
            flat_x_tokens = x.reshape((-1, self.input_dims)).astype(mx.float32)
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            expanded_x = mx.broadcast_to(
                flat_x_tokens[:, None, :],
                (flat_x_tokens.shape[0], top_k, self.input_dims),
            ).reshape((-1, self.input_dims))
            flat_y = y.reshape((-1, self.output_dims))
            result_shape = y.shape
        elif x.shape[:-1] == indices.shape:
            expanded_x = x.reshape((-1, self.input_dims)).astype(mx.float32)
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            flat_y = y.reshape((-1, self.output_dims))
            result_shape = y.shape
        else:
            return y
        selected_right = right[flat_indices].astype(mx.float32)
        selected_left = left[flat_indices].astype(mx.float32)
        latent = mx.sum(selected_right * expanded_x[:, None, :], axis=-1)
        correction = mx.sum(selected_left * latent[:, None, :], axis=-1)
        return (flat_y + correction.astype(flat_y.dtype)).reshape(result_shape)

    def _apply_sparse_residual_rows(self, y: mx.array, x: mx.array, indices: mx.array) -> mx.array:
        residual_values = self.get("sparse_residual_values")
        if residual_values is None:
            return y
        expert_indices = self.sparse_residual_expert_indices
        output_indices = self.sparse_residual_output_indices
        if x.shape[:-1] == indices.shape[:-1]:
            leading_shape = indices.shape[:-1]
            top_k = indices.shape[-1]
            flat_x_tokens = x.reshape((-1, self.input_dims))
            flat_x = mx.broadcast_to(
                flat_x_tokens[:, None, :],
                (flat_x_tokens.shape[0], top_k, self.input_dims),
            ).reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            flat_y = y.reshape((-1, self.output_dims))
            result_shape = y.shape
        elif x.shape[:-1] == indices.shape:
            flat_x = x.reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            flat_y = y.reshape((-1, self.output_dims))
            result_shape = y.shape
        else:
            return y
        row_scores = flat_x.astype(mx.float32) @ residual_values.T
        matches = (flat_indices[:, None] == expert_indices[None, :]).astype(row_scores.dtype)
        row_scores = row_scores * matches
        output_columns = mx.arange(self.output_dims, dtype=mx.int32)
        output_selector = (output_indices[:, None] == output_columns[None, :]).astype(row_scores.dtype)
        correction = row_scores @ output_selector
        return (flat_y + correction.astype(flat_y.dtype)).reshape(result_shape)

    @classmethod
    def from_weights(
        cls,
        weight: mx.array,
        bias: mx.array | None = None,
        *,
        group_size: int = 512,
        code_bits: Literal[8, 16] = 8,
        use_gather_vqmm: bool = True,
        route_strategy: RouteStrategy = "auto",
        gather_codebook_duplication: Literal[1, 4, 8] = 8,
        rht_signs: mx.array | None = None,
        rotation_matrix: mx.array | None = None,
    ) -> "QuantizedVQSwitchLinear":
        weight_np = np.array(weight.astype(mx.float32), copy=False)
        if weight_np.ndim != 3:
            raise ValueError(f"weight must be 3D [experts, out, in], found {weight_np.shape}")
        if rht_signs is not None and rotation_matrix is not None:
            raise ValueError("rht_signs and rotation_matrix are mutually exclusive")
        if rht_signs is not None:
            weight_np = apply_rht_np(weight_np, np.array(rht_signs, copy=False))
        if rotation_matrix is not None:
            weight_np = apply_rotation_np(weight_np, np.array(rotation_matrix, copy=False))
        quantized = [
            quantize_weight_rtn(expert_weight, group_size=group_size, code_bits=code_bits)
            for expert_weight in weight_np
        ]
        codes = np.stack([expert.codes for expert in quantized], axis=0)
        scales = np.stack([expert.scales for expert in quantized], axis=0)
        num_experts, output_dims, input_dims = weight_np.shape
        return cls(
            input_dims=input_dims,
            output_dims=output_dims,
            num_experts=num_experts,
            codes=mx.array(codes),
            scales=mx.array(scales),
            codebook=mx.array(quantized[0].codebook),
            bias=bias,
            group_size=quantized[0].group_size,
            code_bits=quantized[0].code_bits,
            use_gather_vqmm=use_gather_vqmm,
            route_strategy=route_strategy,
            gather_codebook_duplication=gather_codebook_duplication,
            rht_signs=rht_signs,
            rotation_matrix=rotation_matrix,
        )


def _validate_sparse_residual_rows(
    *,
    expert_indices: mx.array | None,
    output_indices: mx.array | None,
    values: mx.array | None,
    input_dims: int,
    output_dims: int,
    num_experts: int,
) -> None:
    if values is None:
        if expert_indices is not None or output_indices is not None:
            raise ValueError("sparse residual indices require sparse_residual_values")
        return
    if expert_indices is None or output_indices is None:
        raise ValueError("sparse residual rows require expert_indices, output_indices, and values")
    if values.ndim != 2 or values.shape[1] != input_dims:
        raise ValueError(f"sparse residual values must have shape [rows, {input_dims}], found {values.shape}")
    row_count = values.shape[0]
    if expert_indices.shape != (row_count,):
        raise ValueError(f"sparse residual expert_indices must have shape ({row_count},), found {expert_indices.shape}")
    if output_indices.shape != (row_count,):
        raise ValueError(f"sparse residual output_indices must have shape ({row_count},), found {output_indices.shape}")
    expert_values = np.asarray(expert_indices)
    output_values = np.asarray(output_indices)
    if expert_values.size and (expert_values.min() < 0 or expert_values.max() >= num_experts):
        raise ValueError("sparse residual expert_indices out of range")
    if output_values.size and (output_values.min() < 0 or output_values.max() >= output_dims):
        raise ValueError("sparse residual output_indices out of range")


def _validate_low_rank_residual(
    *,
    left: mx.array,
    right: mx.array,
    input_dims: int,
    output_dims: int,
    num_experts: int,
) -> None:
    if left.ndim != 3:
        raise ValueError(f"low_rank_left must be 3D [experts, out, rank], found {left.shape}")
    if right.ndim != 3:
        raise ValueError(f"low_rank_right must be 3D [experts, rank, in], found {right.shape}")
    experts, out_dim, rank = left.shape
    if (experts, out_dim) != (num_experts, output_dims):
        raise ValueError(
            f"low_rank_left must have shape ({num_experts}, {output_dims}, rank), found {left.shape}"
        )
    if rank <= 0:
        raise ValueError("low-rank rank must be positive")
    if right.shape != (num_experts, rank, input_dims):
        raise ValueError(
            f"low_rank_right must have shape ({num_experts}, {rank}, {input_dims}), found {right.shape}"
        )
