from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from keep.vq.e8 import CODEWORD_DIM, decode_e8_1bit, decode_e8p
from keep.io.continuous_sidecar import SwitchLinearSidecar
from ramp.nn.switch_linear import QuantizedVQSwitchLinear
from keep.quant.rht import apply_rht_mx


class SwitchLinearSurrogate(nn.Module):
    """Differentiable decoded-weight surrogate for one routed VQ projection."""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        codes: mx.array,
        scales: mx.array,
        codebook: mx.array,
        group_size: int,
        code_bits: int,
        bias: mx.array | None = None,
        rht_signs: mx.array | None = None,
        sidecar: SwitchLinearSidecar | None = None,
    ) -> None:
        super().__init__()
        if input_dims <= 0 or output_dims <= 0 or num_experts <= 0:
            raise ValueError("input_dims, output_dims, and num_experts must be positive")
        if input_dims % CODEWORD_DIM != 0:
            raise ValueError("input_dims must be divisible by 8")
        if input_dims % group_size != 0:
            raise ValueError("input_dims must be divisible by group_size")
        if code_bits not in (8, 16):
            raise ValueError("code_bits must be 8 or 16")
        expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
        expected_scales = (num_experts, output_dims, input_dims // group_size)
        if codes.shape != expected_codes:
            raise ValueError(f"codes must have shape {expected_codes}, found {codes.shape}")
        if scales.shape != expected_scales:
            raise ValueError(f"scales must have shape {expected_scales}, found {scales.shape}")
        if bias is not None and bias.shape != (num_experts, output_dims):
            raise ValueError(f"bias must have shape ({num_experts}, {output_dims}), found {bias.shape}")
        if rht_signs is not None and rht_signs.shape != (input_dims,):
            raise ValueError(f"rht_signs must have shape ({input_dims},), found {rht_signs.shape}")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.group_size = group_size
        self.code_bits = code_bits
        self.codes = codes
        self.base_scales = scales.astype(mx.float32)
        self.decoded_codebook = _decode_codebook_table(codebook, code_bits=code_bits)
        self.scale_delta = _sidecar_array(
            sidecar.scale_delta if sidecar and sidecar.enabled else None,
            expected_scales,
            name="scale_delta",
        )
        self.output_bias = _sidecar_array(
            sidecar.output_bias if sidecar and sidecar.enabled else None,
            (num_experts, output_dims),
            name="output_bias",
        )
        self.low_rank_left = _optional_sidecar_array(
            sidecar.low_rank_left if sidecar and sidecar.enabled else None,
            name="low_rank_left",
        )
        self.low_rank_right = _optional_sidecar_array(
            sidecar.low_rank_right if sidecar and sidecar.enabled else None,
            name="low_rank_right",
        )
        _validate_optional_low_rank(
            self.low_rank_left,
            self.low_rank_right,
            num_experts=num_experts,
            output_dims=output_dims,
            input_dims=input_dims,
        )
        if bias is not None:
            self.bias = bias.astype(mx.float32)
        if rht_signs is not None:
            self.rht_signs = rht_signs.astype(mx.float32)

    @classmethod
    def from_layer(
        cls,
        layer: QuantizedVQSwitchLinear,
        *,
        sidecar: SwitchLinearSidecar | None = None,
    ) -> "SwitchLinearSurrogate":
        return cls(
            input_dims=layer.input_dims,
            output_dims=layer.output_dims,
            num_experts=layer.num_experts,
            codes=layer.codes,
            scales=layer.scales,
            codebook=layer.codebook,
            group_size=layer.group_size,
            code_bits=layer.code_bits,
            bias=layer.get("bias"),
            rht_signs=layer.get("rht_signs"),
            sidecar=sidecar,
        )

    def decoded_weight(self) -> mx.array:
        code_indices = self.codes.astype(mx.int32)
        code_vectors = mx.take(self.decoded_codebook, code_indices, axis=0)
        effective_scales = self.base_scales * mx.exp(self.scale_delta)
        words_per_group = self.group_size // CODEWORD_DIM
        scale_groups = effective_scales[:, :, :, None]
        scale_by_codeword = mx.broadcast_to(
            scale_groups,
            (
                self.num_experts,
                self.output_dims,
                self.input_dims // self.group_size,
                words_per_group,
            ),
        ).reshape((self.num_experts, self.output_dims, self.input_dims // CODEWORD_DIM))
        decoded = code_vectors * scale_by_codeword[:, :, :, None]
        return decoded.reshape((self.num_experts, self.output_dims, self.input_dims))

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"input trailing dimension must be {self.input_dims}, found {x.shape[-1]}")
        rht_signs = self.get("rht_signs")
        if rht_signs is not None:
            x = apply_rht_mx(x, rht_signs)
        if indices.ndim < 1:
            raise ValueError("indices must have at least one dimension")
        if x.shape[:-1] == indices.shape[:-1]:
            return self._call_token_topk(x, indices)
        if x.shape[:-1] == indices.shape:
            return self._call_per_route(x, indices)
        raise ValueError(
            "x leading shape must match indices shape or indices leading shape; "
            f"got x={x.shape}, indices={indices.shape}"
        )

    def _call_token_topk(self, x: mx.array, indices: mx.array) -> mx.array:
        leading_shape = x.shape[:-1]
        top_k = indices.shape[-1]
        flat_x = x.reshape((-1, self.input_dims))
        flat_indices = indices.reshape((-1, top_k)).astype(mx.int32)
        weights = mx.take(self.decoded_weight(), flat_indices, axis=0)
        y = mx.sum(weights * flat_x[:, None, None, :], axis=-1)
        y = self._add_low_rank_residual(y, flat_x, flat_indices)
        y = self._add_bias(y, flat_indices)
        return y.reshape((*leading_shape, top_k, self.output_dims))

    def _call_per_route(self, x: mx.array, indices: mx.array) -> mx.array:
        route_shape = indices.shape
        flat_x = x.reshape((-1, self.input_dims))
        flat_indices = indices.reshape((-1,)).astype(mx.int32)
        weights = mx.take(self.decoded_weight(), flat_indices, axis=0)
        y = mx.sum(weights * flat_x[:, None, :], axis=-1)
        y = self._add_low_rank_residual(y, flat_x, flat_indices)
        y = self._add_bias(y, flat_indices)
        return y.reshape((*route_shape, self.output_dims))

    def _add_low_rank_residual(self, y: mx.array, flat_x: mx.array, indices: mx.array) -> mx.array:
        if self.low_rank_left is None or self.low_rank_right is None:
            return y
        if indices.ndim == 2:
            top_k = indices.shape[-1]
            expanded_x = mx.broadcast_to(
                flat_x[:, None, :],
                (flat_x.shape[0], top_k, self.input_dims),
            ).reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            flat_y = y.reshape((-1, self.output_dims))
            result_shape = y.shape
        else:
            expanded_x = flat_x
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            flat_y = y.reshape((-1, self.output_dims))
            result_shape = y.shape
        selected_right = mx.take(self.low_rank_right, flat_indices, axis=0)
        selected_left = mx.take(self.low_rank_left, flat_indices, axis=0)
        latent = mx.sum(selected_right * expanded_x[:, None, :], axis=-1)
        correction = mx.sum(selected_left * latent[:, None, :], axis=-1)
        return (flat_y + correction.astype(flat_y.dtype)).reshape(result_shape)

    def _add_bias(self, y: mx.array, indices: mx.array) -> mx.array:
        bias = self.get("bias")
        if bias is not None:
            y = y + mx.take(bias, indices, axis=0)
        if self.output_bias is not None:
            y = y + mx.take(self.output_bias, indices, axis=0)
        return y


class RouteLocalSwitchLinearSurrogate(nn.Module):
    """Differentiable routed VQ projection that decodes only the selected routes."""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        codes: mx.array,
        scales: mx.array,
        codebook: mx.array,
        group_size: int,
        code_bits: int,
        bias: mx.array | None = None,
        rht_signs: mx.array | None = None,
        sidecar: SwitchLinearSidecar | None = None,
        output_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if output_chunk_size <= 0:
            raise ValueError("output_chunk_size must be positive")
        if input_dims <= 0 or output_dims <= 0 or num_experts <= 0:
            raise ValueError("input_dims, output_dims, and num_experts must be positive")
        if input_dims % CODEWORD_DIM != 0:
            raise ValueError("input_dims must be divisible by 8")
        if input_dims % group_size != 0:
            raise ValueError("input_dims must be divisible by group_size")
        if code_bits not in (8, 16):
            raise ValueError("code_bits must be 8 or 16")
        expected_codes = (num_experts, output_dims, input_dims // CODEWORD_DIM)
        expected_scales = (num_experts, output_dims, input_dims // group_size)
        if codes.shape != expected_codes:
            raise ValueError(f"codes must have shape {expected_codes}, found {codes.shape}")
        if scales.shape != expected_scales:
            raise ValueError(f"scales must have shape {expected_scales}, found {scales.shape}")
        if bias is not None and bias.shape != (num_experts, output_dims):
            raise ValueError(f"bias must have shape ({num_experts}, {output_dims}), found {bias.shape}")
        if rht_signs is not None and rht_signs.shape != (input_dims,):
            raise ValueError(f"rht_signs must have shape ({input_dims},), found {rht_signs.shape}")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.group_size = group_size
        self.code_bits = code_bits
        self.output_chunk_size = int(output_chunk_size)
        self.codes = codes
        self.base_scales = scales.astype(mx.float32)
        self.decoded_codebook = _decode_codebook_table(codebook, code_bits=code_bits)
        self.scale_delta = _sidecar_array(
            sidecar.scale_delta if sidecar and sidecar.enabled else None,
            expected_scales,
            name="scale_delta",
        )
        self.output_bias = _sidecar_array(
            sidecar.output_bias if sidecar and sidecar.enabled else None,
            (num_experts, output_dims),
            name="output_bias",
        )
        self.low_rank_left = _optional_sidecar_array(
            sidecar.low_rank_left if sidecar and sidecar.enabled else None,
            name="low_rank_left",
        )
        self.low_rank_right = _optional_sidecar_array(
            sidecar.low_rank_right if sidecar and sidecar.enabled else None,
            name="low_rank_right",
        )
        _validate_optional_low_rank(
            self.low_rank_left,
            self.low_rank_right,
            num_experts=num_experts,
            output_dims=output_dims,
            input_dims=input_dims,
        )
        if bias is not None:
            self.bias = bias.astype(mx.float32)
        if rht_signs is not None:
            self.rht_signs = rht_signs.astype(mx.float32)

    @classmethod
    def from_layer(
        cls,
        layer: QuantizedVQSwitchLinear,
        *,
        sidecar: SwitchLinearSidecar | None = None,
        output_chunk_size: int = 256,
    ) -> "RouteLocalSwitchLinearSurrogate":
        if layer.has_sparse_residual_rows:
            raise ValueError("route-local surrogate does not support sparse residual rows")
        return cls(
            input_dims=layer.input_dims,
            output_dims=layer.output_dims,
            num_experts=layer.num_experts,
            codes=layer.codes,
            scales=layer.scales,
            codebook=layer.codebook,
            group_size=layer.group_size,
            code_bits=layer.code_bits,
            bias=layer.get("bias"),
            rht_signs=layer.get("rht_signs"),
            sidecar=sidecar,
            output_chunk_size=output_chunk_size,
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"input trailing dimension must be {self.input_dims}, found {x.shape[-1]}")
        rht_signs = self.get("rht_signs")
        if rht_signs is not None:
            x = apply_rht_mx(x, rht_signs)
        if indices.ndim < 1:
            raise ValueError("indices must have at least one dimension")
        if x.shape[:-1] == indices.shape[:-1]:
            leading_shape = x.shape[:-1]
            top_k = indices.shape[-1]
            flat_x = x.reshape((-1, self.input_dims))
            flat_x = mx.broadcast_to(flat_x[:, None, :], (flat_x.shape[0], top_k, self.input_dims))
            flat_x = flat_x.reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
            route_shape = (*leading_shape, top_k)
        elif x.shape[:-1] == indices.shape:
            route_shape = indices.shape
            flat_x = x.reshape((-1, self.input_dims))
            flat_indices = indices.reshape((-1,)).astype(mx.int32)
        else:
            raise ValueError(
                "x leading shape must match indices shape or indices leading shape; "
                f"got x={x.shape}, indices={indices.shape}"
            )
        y = self._flat_route_projection(flat_x, flat_indices)
        return y.reshape((*route_shape, self.output_dims))

    def _flat_route_projection(self, flat_x: mx.array, flat_indices: mx.array) -> mx.array:
        words_per_group = self.group_size // CODEWORD_DIM
        x_words = flat_x.reshape((flat_x.shape[0], self.input_dims // CODEWORD_DIM, CODEWORD_DIM))
        effective_scales = self.base_scales * mx.exp(self.scale_delta)
        route_chunks = []
        for start in range(0, self.output_dims, self.output_chunk_size):
            stop = min(start + self.output_chunk_size, self.output_dims)
            selected_codes = mx.take(self.codes[:, start:stop, :], flat_indices, axis=0).astype(mx.int32)
            selected_scales = mx.take(effective_scales[:, start:stop, :], flat_indices, axis=0)
            code_vectors = mx.take(self.decoded_codebook, selected_codes, axis=0)
            scale_by_codeword = mx.broadcast_to(
                selected_scales[:, :, :, None],
                (
                    flat_x.shape[0],
                    stop - start,
                    self.input_dims // self.group_size,
                    words_per_group,
                ),
            ).reshape((flat_x.shape[0], stop - start, self.input_dims // CODEWORD_DIM))
            products = code_vectors * scale_by_codeword[:, :, :, None] * x_words[:, None, :, :]
            y_chunk = mx.sum(mx.sum(products, axis=-1), axis=-1)
            bias = self.get("bias")
            if bias is not None:
                y_chunk = y_chunk + mx.take(bias[:, start:stop], flat_indices, axis=0)
            if self.output_bias is not None:
                y_chunk = y_chunk + mx.take(self.output_bias[:, start:stop], flat_indices, axis=0)
            if self.low_rank_left is not None and self.low_rank_right is not None:
                selected_right = mx.take(self.low_rank_right, flat_indices, axis=0)
                selected_left = mx.take(self.low_rank_left[:, start:stop, :], flat_indices, axis=0)
                latent = mx.sum(selected_right * flat_x[:, None, :], axis=-1)
                y_chunk = y_chunk + mx.sum(selected_left * latent[:, None, :], axis=-1)
            route_chunks.append(y_chunk)
        if len(route_chunks) == 1:
            return route_chunks[0]
        return mx.concatenate(route_chunks, axis=-1)


def switch_linear_layer_sidecar(layer: QuantizedVQSwitchLinear) -> SwitchLinearSidecar | None:
    scale_delta = layer.get("continuous_scale_delta")
    output_bias = layer.get("continuous_output_bias")
    low_rank_left = layer.get("continuous_low_rank_left")
    low_rank_right = layer.get("continuous_low_rank_right")
    if scale_delta is None and output_bias is None and low_rank_left is None and low_rank_right is None:
        return None
    return SwitchLinearSidecar(
        scale_delta=scale_delta,
        output_bias=output_bias,
        low_rank_left=low_rank_left,
        low_rank_right=low_rank_right,
    )


def call_switch_linear_with_optional_sidecar(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    indices: mx.array,
    *,
    sidecar: SwitchLinearSidecar | None = None,
) -> mx.array:
    if sidecar is None or not sidecar.enabled:
        return layer(x, indices)
    return SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)(x, indices)


def fit_output_bias_sidecar_least_squares(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    indices: mx.array,
    target_output: mx.array,
) -> dict[str, object]:
    """Fit the best per-expert output-bias sidecar for block-local targets."""

    baseline = SwitchLinearSurrogate.from_layer(layer)(x, indices)
    mx.eval(baseline, target_output)
    baseline_np = np.asarray(baseline, dtype=np.float32)
    target_np = np.asarray(target_output, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.int64)
    if baseline_np.shape != target_np.shape:
        raise ValueError(f"target_output shape {target_np.shape} must match baseline shape {baseline_np.shape}")
    if baseline_np.shape[:-1] != indices_np.shape:
        raise ValueError(
            "target_output leading shape must match indices shape; "
            f"got target={target_np.shape}, indices={indices_np.shape}"
        )

    residual = target_np - baseline_np
    flat_residual = residual.reshape((-1, layer.output_dims))
    flat_indices = indices_np.reshape((-1,))
    output_bias = np.zeros((layer.num_experts, layer.output_dims), dtype=np.float32)
    route_counts = np.bincount(flat_indices, minlength=layer.num_experts).astype(np.int64)
    for expert in np.flatnonzero(route_counts):
        output_bias[int(expert)] = flat_residual[flat_indices == expert].mean(axis=0)

    fitted_np = baseline_np + output_bias[indices_np]
    baseline_mse = float(np.mean(np.square(residual.astype(np.float64))))
    fitted_mse = float(np.mean(np.square((target_np - fitted_np).astype(np.float64))))
    improvement_ratio = 0.0 if baseline_mse == 0.0 else fitted_mse / baseline_mse
    return {
        "ok": True,
        "sidecar": SwitchLinearSidecar(output_bias=mx.array(output_bias)),
        "baseline_mse": baseline_mse,
        "fitted_mse": fitted_mse,
        "improvement_ratio": improvement_ratio,
        "expert_route_counts": tuple(int(count) for count in route_counts.tolist()),
    }


def fit_low_rank_residual_sidecar_least_squares(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    indices: mx.array,
    target_output: mx.array,
    *,
    rank: int,
    ridge_strength: float = 0.0,
) -> dict[str, object]:
    """Fit per-expert low-rank linear residuals for block-local targets."""

    if rank <= 0:
        raise ValueError("rank must be positive")
    if ridge_strength < 0.0:
        raise ValueError("ridge_strength must be non-negative")
    baseline = SwitchLinearSurrogate.from_layer(layer)(x, indices)
    mx.eval(baseline, target_output)
    baseline_np = np.asarray(baseline, dtype=np.float32)
    target_np = np.asarray(target_output, dtype=np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.int64)
    _validate_fit_shapes(layer, x_np, indices_np, baseline_np, target_np)

    residual = target_np - baseline_np
    flat_residual = residual.reshape((-1, layer.output_dims))
    flat_indices = indices_np.reshape((-1,))
    if x_np.shape[:-1] == indices_np.shape[:-1]:
        top_k = indices_np.shape[-1]
        flat_x = np.broadcast_to(
            x_np.reshape((-1, layer.input_dims))[:, None, :],
            (np.prod(indices_np.shape[:-1], dtype=np.int64), top_k, layer.input_dims),
        ).reshape((-1, layer.input_dims))
    elif x_np.shape[:-1] == indices_np.shape:
        flat_x = x_np.reshape((-1, layer.input_dims))
    else:
        raise ValueError(
            "x leading shape must match indices shape or indices leading shape; "
            f"got x={x_np.shape}, indices={indices_np.shape}"
        )

    low_rank_left = np.zeros((layer.num_experts, layer.output_dims, rank), dtype=np.float32)
    low_rank_right = np.zeros((layer.num_experts, rank, layer.input_dims), dtype=np.float32)
    route_counts = np.bincount(flat_indices, minlength=layer.num_experts).astype(np.int64)
    actual_ranks: list[int] = [0 for _ in range(layer.num_experts)]
    for expert in np.flatnonzero(route_counts):
        expert_mask = flat_indices == int(expert)
        design = flat_x[expert_mask].astype(np.float64, copy=False)
        expert_residual = flat_residual[expert_mask].astype(np.float64, copy=False)
        if ridge_strength > 0.0:
            regularizer = np.sqrt(float(ridge_strength)) * np.eye(layer.input_dims, dtype=np.float64)
            design_solve = np.concatenate([design, regularizer], axis=0)
            residual_solve = np.concatenate(
                [expert_residual, np.zeros((layer.input_dims, layer.output_dims), dtype=np.float64)],
                axis=0,
            )
        else:
            design_solve = design
            residual_solve = expert_residual
        solved, *_ = np.linalg.lstsq(design_solve, residual_solve, rcond=None)
        correction = solved.T
        u, singular_values, vt = np.linalg.svd(correction, full_matrices=False)
        actual_rank = min(rank, int(singular_values.shape[0]))
        if actual_rank:
            low_rank_left[int(expert), :, :actual_rank] = (
                u[:, :actual_rank] * singular_values[:actual_rank][None, :]
            ).astype(np.float32, copy=False)
            low_rank_right[int(expert), :actual_rank, :] = vt[:actual_rank].astype(np.float32, copy=False)
        actual_ranks[int(expert)] = actual_rank

    sidecar = SwitchLinearSidecar(
        low_rank_left=mx.array(low_rank_left),
        low_rank_right=mx.array(low_rank_right),
    )
    fitted = SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)(x, indices)
    mx.eval(fitted)
    fitted_np = np.asarray(fitted, dtype=np.float32)
    baseline_mse = float(np.mean(np.square((target_np - baseline_np).astype(np.float64))))
    fitted_mse = float(np.mean(np.square((target_np - fitted_np).astype(np.float64))))
    improvement_ratio = 0.0 if baseline_mse == 0.0 else fitted_mse / baseline_mse
    return {
        "ok": True,
        "sidecar": sidecar,
        "baseline_mse": baseline_mse,
        "fitted_mse": fitted_mse,
        "improvement_ratio": improvement_ratio,
        "expert_route_counts": tuple(int(count) for count in route_counts.tolist()),
        "rank": int(rank),
        "actual_ranks": tuple(int(value) for value in actual_ranks),
        "ridge_strength": float(ridge_strength),
    }


def fit_scale_delta_sidecar_least_squares(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    indices: mx.array,
    target_output: mx.array,
    *,
    min_multiplier: float = 1.0e-4,
    max_multiplier: float = 1.0e4,
    max_abs_log_delta: float = 1.0,
    ridge_strength: float = 0.0,
) -> dict[str, object]:
    """Fit per-group scale multipliers for block-local projection targets."""

    if min_multiplier <= 0.0 or max_multiplier <= min_multiplier:
        raise ValueError("expected 0 < min_multiplier < max_multiplier")
    if max_abs_log_delta <= 0.0:
        raise ValueError("max_abs_log_delta must be positive")
    if ridge_strength < 0.0:
        raise ValueError("ridge_strength must be non-negative")
    baseline = SwitchLinearSurrogate.from_layer(layer)(x, indices)
    mx.eval(baseline, target_output)
    baseline_np = np.asarray(baseline, dtype=np.float32)
    target_np = np.asarray(target_output, dtype=np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.int64)
    _validate_fit_shapes(layer, x_np, indices_np, baseline_np, target_np)

    transformed_x = _rht_transformed_input_np(layer, x_np)
    group_contrib = _scale_group_contributions_np(layer, transformed_x)
    multipliers = np.ones(layer.scales.shape, dtype=np.float32)
    route_counts = np.bincount(indices_np.reshape(-1), minlength=layer.num_experts).astype(np.int64)
    flat_indices = indices_np.reshape(-1)
    flat_token_indices = np.broadcast_to(
        np.arange(indices_np.shape[0])[:, None],
        indices_np.shape,
    ).reshape(-1)
    flat_baseline = baseline_np.reshape((-1, layer.output_dims))
    flat_target = target_np.reshape((-1, layer.output_dims))
    flat_residual = flat_target - flat_baseline
    min_bounded_multiplier = max(min_multiplier, float(np.exp(-max_abs_log_delta)))
    max_bounded_multiplier = min(max_multiplier, float(np.exp(max_abs_log_delta)))
    for expert in np.flatnonzero(route_counts):
        route_mask = flat_indices == int(expert)
        expert_contrib = group_contrib[int(expert), flat_token_indices[route_mask]]
        expert_residual = flat_residual[route_mask]
        for output_idx in range(layer.output_dims):
            design = expert_contrib[:, output_idx, :]
            target_residual = expert_residual[:, output_idx]
            if ridge_strength > 0.0:
                regularizer = np.sqrt(float(ridge_strength)) * np.eye(design.shape[1], dtype=np.float32)
                design_solve = np.concatenate([design, regularizer], axis=0)
                target_solve = np.concatenate(
                    [target_residual, np.zeros(design.shape[1], dtype=np.float32)],
                    axis=0,
                )
            else:
                design_solve = design
                target_solve = target_residual
            solved, *_ = np.linalg.lstsq(design_solve, target_solve, rcond=None)
            multiplier = 1.0 + solved.astype(np.float32, copy=False)
            multipliers[int(expert), output_idx] = np.clip(
                multiplier,
                min_bounded_multiplier,
                max_bounded_multiplier,
            )

    scale_delta = np.log(multipliers).astype(np.float32, copy=False)
    sidecar = SwitchLinearSidecar(scale_delta=mx.array(scale_delta))
    fitted = SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)(x, indices)
    mx.eval(fitted)
    fitted_np = np.asarray(fitted, dtype=np.float32)
    baseline_mse = float(np.mean(np.square((target_np - baseline_np).astype(np.float64))))
    fitted_mse = float(np.mean(np.square((target_np - fitted_np).astype(np.float64))))
    improvement_ratio = 0.0 if baseline_mse == 0.0 else fitted_mse / baseline_mse
    return {
        "ok": True,
        "sidecar": sidecar,
        "baseline_mse": baseline_mse,
        "fitted_mse": fitted_mse,
        "improvement_ratio": improvement_ratio,
        "expert_route_counts": tuple(int(count) for count in route_counts.tolist()),
        "min_multiplier": float(np.min(multipliers)),
        "max_multiplier": float(np.max(multipliers)),
        "max_abs_log_delta": float(max_abs_log_delta),
        "ridge_strength": float(ridge_strength),
    }


def fit_scale_delta_output_bias_sidecar_least_squares(
    layer: QuantizedVQSwitchLinear,
    x: mx.array,
    indices: mx.array,
    target_output: mx.array,
    *,
    min_multiplier: float = 1.0e-4,
    max_multiplier: float = 1.0e4,
    max_abs_log_delta: float = 1.0,
    ridge_strength: float = 0.0,
) -> dict[str, object]:
    """Fit scale multipliers, then per-expert output bias for remaining residual."""

    scale_result = fit_scale_delta_sidecar_least_squares(
        layer,
        x,
        indices,
        target_output,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        max_abs_log_delta=max_abs_log_delta,
        ridge_strength=ridge_strength,
    )
    scale_sidecar = scale_result["sidecar"]
    if not isinstance(scale_sidecar, SwitchLinearSidecar) or scale_sidecar.scale_delta is None:
        raise RuntimeError("scale-delta fit did not return a scale_delta sidecar")

    scaled = SwitchLinearSurrogate.from_layer(layer, sidecar=scale_sidecar)(x, indices)
    mx.eval(scaled, target_output)
    scaled_np = np.asarray(scaled, dtype=np.float32)
    target_np = np.asarray(target_output, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.int64)
    if scaled_np.shape != target_np.shape:
        raise ValueError(f"target_output shape {target_np.shape} must match scaled shape {scaled_np.shape}")
    if scaled_np.shape[:-1] != indices_np.shape:
        raise ValueError(
            "target_output leading shape must match indices shape; "
            f"got target={target_np.shape}, indices={indices_np.shape}"
        )

    residual = target_np - scaled_np
    flat_residual = residual.reshape((-1, layer.output_dims))
    flat_indices = indices_np.reshape((-1,))
    route_counts = np.bincount(flat_indices, minlength=layer.num_experts).astype(np.int64)
    output_bias = np.zeros((layer.num_experts, layer.output_dims), dtype=np.float32)
    for expert in np.flatnonzero(route_counts):
        output_bias[int(expert)] = flat_residual[flat_indices == expert].mean(axis=0)

    sidecar = SwitchLinearSidecar(
        scale_delta=scale_sidecar.scale_delta,
        output_bias=mx.array(output_bias),
    )
    fitted = SwitchLinearSurrogate.from_layer(layer, sidecar=sidecar)(x, indices)
    mx.eval(fitted)
    fitted_np = np.asarray(fitted, dtype=np.float32)
    fitted_mse = float(np.mean(np.square((target_np - fitted_np).astype(np.float64))))
    baseline_mse = float(scale_result["baseline_mse"])
    improvement_ratio = 0.0 if baseline_mse == 0.0 else fitted_mse / baseline_mse
    return {
        "ok": True,
        "sidecar": sidecar,
        "baseline_mse": baseline_mse,
        "scale_delta_mse": float(scale_result["fitted_mse"]),
        "fitted_mse": fitted_mse,
        "improvement_ratio": improvement_ratio,
        "expert_route_counts": tuple(int(count) for count in route_counts.tolist()),
        "min_multiplier": scale_result["min_multiplier"],
        "max_multiplier": scale_result["max_multiplier"],
        "max_abs_log_delta": scale_result["max_abs_log_delta"],
        "ridge_strength": scale_result["ridge_strength"],
    }


def _validate_fit_shapes(
    layer: QuantizedVQSwitchLinear,
    x: np.ndarray,
    indices: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
) -> None:
    if x.shape[-1] != layer.input_dims:
        raise ValueError(f"x trailing shape must be {layer.input_dims}, found {x.shape[-1]}")
    if baseline.shape != target.shape:
        raise ValueError(f"target_output shape {target.shape} must match baseline shape {baseline.shape}")
    if baseline.shape[:-1] != indices.shape:
        raise ValueError(
            "target_output leading shape must match indices shape; "
            f"got target={target.shape}, indices={indices.shape}"
        )


def _rht_transformed_input_np(layer: QuantizedVQSwitchLinear, x: np.ndarray) -> np.ndarray:
    signs = layer.get("rht_signs")
    if signs is None:
        return x
    return np.asarray(apply_rht_mx(mx.array(x), signs).astype(mx.float32), dtype=np.float32)


def _bias_np(layer: QuantizedVQSwitchLinear) -> np.ndarray | None:
    bias = layer.get("bias")
    if bias is None:
        return None
    return np.asarray(bias, dtype=np.float32)


def _scale_group_contributions_np(layer: QuantizedVQSwitchLinear, x: np.ndarray) -> np.ndarray:
    code_indices = np.asarray(layer.codes, dtype=np.int64)
    codebook = _decode_codebook_table(layer.codebook, code_bits=layer.code_bits)
    code_vectors = np.asarray(mx.take(codebook, mx.array(code_indices), axis=0), dtype=np.float32)
    tokens = x.reshape((-1, layer.input_dims))
    groups = layer.input_dims // layer.group_size
    words_per_group = layer.group_size // CODEWORD_DIM
    code_vectors = code_vectors.reshape(
        (layer.num_experts, layer.output_dims, groups, words_per_group, CODEWORD_DIM)
    )
    x_groups = tokens.reshape((tokens.shape[0], groups, words_per_group, CODEWORD_DIM))
    scale_base = np.asarray(layer.scales, dtype=np.float32)
    contrib = np.einsum("eogwq,tgwq,eog->etog", code_vectors, x_groups, scale_base, optimize=True)
    return contrib


def _sidecar_array(value: mx.array | None, expected_shape: tuple[int, ...], *, name: str) -> mx.array:
    if value is None:
        return mx.zeros(expected_shape, dtype=mx.float32)
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, found {value.shape}")
    return value.astype(mx.float32)


def _optional_sidecar_array(value: mx.array | None, *, name: str) -> mx.array | None:
    del name
    if value is None:
        return None
    return value.astype(mx.float32)


def _validate_optional_low_rank(
    left: mx.array | None,
    right: mx.array | None,
    *,
    num_experts: int,
    output_dims: int,
    input_dims: int,
) -> None:
    if left is None and right is None:
        return
    if left is None or right is None:
        raise ValueError("low-rank sidecar requires both low_rank_left and low_rank_right")
    if left.ndim != 3:
        raise ValueError(f"low_rank_left must be 3D [experts, out, rank], found {left.shape}")
    if right.ndim != 3:
        raise ValueError(f"low_rank_right must be 3D [experts, rank, in], found {right.shape}")
    experts, out_dim, rank = left.shape
    if (experts, out_dim) != (num_experts, output_dims):
        raise ValueError(f"low_rank_left must have shape ({num_experts}, {output_dims}, rank), found {left.shape}")
    if right.shape != (num_experts, rank, input_dims):
        raise ValueError(f"low_rank_right must have shape ({num_experts}, {rank}, {input_dims}), found {right.shape}")


def _decode_codebook_table(codebook: mx.array, *, code_bits: int) -> mx.array:
    codebook_np = np.array(codebook, dtype=np.uint32)
    if code_bits == 8:
        decoded = decode_e8_1bit(np.arange(256, dtype=np.uint8), codebook_np)
    elif code_bits == 16:
        decoded = decode_e8p(np.arange(1 << 16, dtype=np.uint16), codebook_np)
    else:
        raise ValueError("code_bits must be 8 or 16")
    return mx.array(decoded.astype(np.float32, copy=False))
