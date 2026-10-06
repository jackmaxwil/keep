from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from mlx_vq.kernels import nax
from mlx_vq.nn.switch_linear import HighPrecisionSwitchLinear, QuantizedVQSwitchLinear
from mlx_vq.ops.vq_switch import (
    _expert_block_tile_descriptors,
    _inverse_permutation,
    auto_selects_nax_e8p,
    gather_vqmm_sorted_routes,
)
from mlx_vq.quant.rotation import apply_rotation_mx
from mlx_vq.quant.rht import apply_rht_mx

SwitchGLUProjection = QuantizedVQSwitchLinear | HighPrecisionSwitchLinear


@dataclass(frozen=True)
class GLM4MoeRoutingConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    scoring_func: Literal["sigmoid", "softmax"] = "sigmoid"
    topk_method: Literal["noaux_tc"] = "noaux_tc"
    n_shared_experts: int | None = None


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    return inds, scores * routed_scaling_factor


class GLM4MoEGate(nn.Module):
    def __init__(
        self,
        config: GLM4MoeRoutingConfig,
        *,
        weight: mx.array | None = None,
        e_score_correction_bias: mx.array | None = None,
    ):
        super().__init__()
        if config.topk_method != "noaux_tc":
            raise ValueError("only GLM4 noaux_tc top-k routing is supported")
        if config.scoring_func != "sigmoid":
            raise ValueError("only GLM4 sigmoid routing is supported")
        if config.n_routed_experts % config.n_group != 0:
            raise ValueError("n_routed_experts must be divisible by n_group")
        if config.topk_group > config.n_group:
            raise ValueError("topk_group cannot exceed n_group")
        self.config = config
        self.weight = (
            mx.zeros((config.n_routed_experts, config.hidden_size))
            if weight is None
            else weight
        )
        self.e_score_correction_bias = (
            mx.zeros((config.n_routed_experts,))
            if e_score_correction_bias is None
            else e_score_correction_bias
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.config.num_experts_per_tok,
            self.config.n_group,
            self.config.topk_group,
            self.config.routed_scaling_factor,
            self.config.norm_topk_prob,
        )


class QuantizedVQSwitchGLU(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: SwitchGLUProjection,
        up_proj: SwitchGLUProjection,
        down_proj: SwitchGLUProjection,
        prefill_engine: Literal["auto", "vq_metal", "nax_e8", "nax_e8p"] = "auto",
        activation: Callable[[mx.array, mx.array], mx.array] | None = None,
    ):
        super().__init__()
        if gate_proj.input_dims != up_proj.input_dims:
            raise ValueError("gate_proj and up_proj input dimensions must match")
        if gate_proj.output_dims != up_proj.output_dims:
            raise ValueError("gate_proj and up_proj output dimensions must match")
        if down_proj.input_dims != gate_proj.output_dims:
            raise ValueError("down_proj input dimension must match routed hidden dimension")
        if down_proj.output_dims != gate_proj.input_dims:
            raise ValueError("down_proj output dimension must match model hidden dimension")
        if len({gate_proj.num_experts, up_proj.num_experts, down_proj.num_experts}) != 1:
            raise ValueError("all switch projections must have the same expert count")
        if prefill_engine not in ("auto", "vq_metal", "nax_e8", "nax_e8p"):
            raise ValueError("prefill_engine must be 'auto', 'vq_metal', 'nax_e8', or 'nax_e8p'")
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.prefill_engine = prefill_engine
        # ``None`` means plain SwiGLU, which is what GLM-4.5-Air, GLM-5.2 and
        # Qwen all use; that branch is left literally untouched so their numerics
        # cannot move. Families whose reference builds ``SwitchGLU`` with a
        # non-default activation -- DeepSeek-V4-Flash passes ``LimitedSwiGLU``
        # with ``swiglu_limit`` -- supply it here instead of silently losing it.
        # Argument order matches mlx-lm's ``SwitchGLU``: ``activation(up, gate)``.
        self.activation = activation

    @property
    def input_dims(self) -> int:
        return self.gate_proj.input_dims

    @property
    def hidden_dims(self) -> int:
        return self.gate_proj.output_dims

    @property
    def num_experts(self) -> int:
        return self.gate_proj.num_experts

    def _can_use_shared_sorted_route_prefill(self, x: mx.array, indices: mx.array) -> bool:
        return (
            indices.size >= 64
            and x.ndim >= 2
            and indices.ndim >= 2
            and x.shape[:-1] == indices.shape[:-1]
            and self.gate_proj.use_gather_vqmm
            and self.up_proj.use_gather_vqmm
            and self.down_proj.use_gather_vqmm
            and self.gate_proj.route_strategy != "direct"
            and self.up_proj.route_strategy != "direct"
            and self.down_proj.route_strategy != "direct"
            and not self.gate_proj.has_sparse_residual_rows
            and not self.up_proj.has_sparse_residual_rows
            and not self.down_proj.has_sparse_residual_rows
        )

    @staticmethod
    def _add_sorted_bias(y: mx.array, projection: SwitchGLUProjection, sorted_rhs: mx.array) -> mx.array:
        bias = projection.get("bias")
        if bias is not None:
            y = y + bias[sorted_rhs]
        output_bias = projection.get("continuous_output_bias")
        if output_bias is not None:
            y = y + output_bias[sorted_rhs]
        return y

    @staticmethod
    def _add_sorted_low_rank_residual(
        y: mx.array,
        projection: SwitchGLUProjection,
        x: mx.array,
        sorted_rhs: mx.array,
        sorted_lhs: mx.array,
    ) -> mx.array:
        left = projection.get("continuous_low_rank_left")
        right = projection.get("continuous_low_rank_right")
        if left is None or right is None:
            return y
        selected_x = x[sorted_lhs].astype(mx.float32)
        selected_right = right[sorted_rhs].astype(mx.float32)
        selected_left = left[sorted_rhs].astype(mx.float32)
        latent = mx.sum(selected_right * selected_x[:, None, :], axis=-1)
        correction = mx.sum(selected_left * latent[:, None, :], axis=-1)
        return y + correction.astype(y.dtype)

    def _sorted_prefill_implementation(
        self,
        *,
        route_count: int,
        activation_dtype: mx.Dtype,
    ) -> Literal["metal", "nax_e8", "nax_e8p", "nax_e8p_m32n64"]:
        if self.prefill_engine in ("nax_e8", "nax_e8p"):
            return self.prefill_engine
        if self.prefill_engine == "auto":
            projections = (self.gate_proj, self.up_proj, self.down_proj)
            if (
                nax.is_available()
                and all(projection.code_bits == 8 for projection in projections)
                and all(projection.group_size % 8 == 0 for projection in projections)
            ):
                return "nax_e8"
            if all(
                auto_selects_nax_e8p(
                    route_count=route_count,
                    implementation="metal",
                    code_bits=projection.code_bits,
                    input_dims=projection.input_dims,
                    output_dims=projection.output_dims,
                    group_size=projection.group_size,
                    activation_dtype=activation_dtype,
                )
                for projection in projections
            ):
                if nax.is_available():
                    return "nax_e8p_m32n64"
                warnings.warn(
                    "native VQ NAX extension is unavailable; the E8P fast path is not being used. "
                    "Rebuild native/vq_nax_ext for this Python environment.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return "metal"

    def _shared_sorted_tile_descriptors(
        self,
        sorted_rhs: mx.array,
        implementation: Literal["metal", "nax_e8", "nax_e8p", "nax_e8p_m32n64"],
    ) -> tuple[mx.array, mx.array, mx.array] | None:
        if implementation not in ("nax_e8", "nax_e8p", "nax_e8p_m32n64"):
            return None
        return _expert_block_tile_descriptors(
            sorted_rhs,
            num_experts=self.gate_proj.codes.shape[0],
            route_tile=32 if implementation == "nax_e8p_m32n64" else 64,
        )

    @staticmethod
    def _rht_input_for_projection(projection: SwitchGLUProjection, x: mx.array) -> mx.array:
        rht_signs = projection.get("rht_signs")
        if rht_signs is not None:
            return apply_rht_mx(x, rht_signs)
        rotation_matrix = projection.get("rotation_matrix")
        if rotation_matrix is not None:
            return apply_rotation_mx(x, rotation_matrix)
        return x

    def _shared_sorted_route_prefill(self, x: mx.array, indices: mx.array) -> mx.array:
        leading_shape = x.shape[:-1]
        top_k = indices.shape[-1]
        flat_x = x.reshape((-1, self.input_dims))
        flat_rhs = indices.reshape((-1,)).astype(mx.int32)
        flat_lhs = (mx.arange(flat_rhs.shape[0], dtype=mx.int32) // top_k).astype(mx.int32)
        order = mx.argsort(flat_rhs)
        inverse_order = _inverse_permutation(order)
        sorted_rhs = flat_rhs[order]
        sorted_lhs = flat_lhs[order]
        implementation = self._sorted_prefill_implementation(
            route_count=flat_rhs.shape[0],
            activation_dtype=flat_x.dtype,
        )
        tile_descriptors = self._shared_sorted_tile_descriptors(sorted_rhs, implementation)

        gate_input = self._rht_input_for_projection(self.gate_proj, flat_x)
        gate = gather_vqmm_sorted_routes(
            gate_input,
            self.gate_proj.codes,
            self.gate_proj.effective_scales(),
            self.gate_proj.codebook,
            sorted_rhs,
            sorted_lhs,
            input_dims=self.gate_proj.input_dims,
            output_dims=self.gate_proj.output_dims,
            group_size=self.gate_proj.group_size,
            code_bits=self.gate_proj.code_bits,
            codebook_duplication=self.gate_proj.gather_codebook_duplication,
            implementation=implementation,
            projection="gate_up",
            tile_descriptors=tile_descriptors,
        )
        gate = self._add_sorted_low_rank_residual(gate, self.gate_proj, gate_input, sorted_rhs, sorted_lhs)
        gate = self._add_sorted_bias(gate, self.gate_proj, sorted_rhs)
        up_input = self._rht_input_for_projection(self.up_proj, flat_x)
        up = gather_vqmm_sorted_routes(
            up_input,
            self.up_proj.codes,
            self.up_proj.effective_scales(),
            self.up_proj.codebook,
            sorted_rhs,
            sorted_lhs,
            input_dims=self.up_proj.input_dims,
            output_dims=self.up_proj.output_dims,
            group_size=self.up_proj.group_size,
            code_bits=self.up_proj.code_bits,
            codebook_duplication=self.up_proj.gather_codebook_duplication,
            implementation=implementation,
            projection="gate_up",
            tile_descriptors=tile_descriptors,
        )
        up = self._add_sorted_low_rank_residual(up, self.up_proj, up_input, sorted_rhs, sorted_lhs)
        up = self._add_sorted_bias(up, self.up_proj, sorted_rhs)
        hidden = self._apply_activation(up, gate)

        down_lhs = mx.arange(flat_rhs.shape[0], dtype=mx.int32)
        down_input = self._rht_input_for_projection(self.down_proj, hidden)
        y = gather_vqmm_sorted_routes(
            down_input,
            self.down_proj.codes,
            self.down_proj.effective_scales(),
            self.down_proj.codebook,
            sorted_rhs,
            down_lhs,
            input_dims=self.down_proj.input_dims,
            output_dims=self.down_proj.output_dims,
            group_size=self.down_proj.group_size,
            code_bits=self.down_proj.code_bits,
            codebook_duplication=self.down_proj.gather_codebook_duplication,
            implementation=implementation,
            projection="down",
            tile_descriptors=tile_descriptors,
            # ``hidden`` is built from sorted gate/up outputs and ``down_lhs`` is an
            # explicit arange, so the activation gather is a no-op and can be skipped.
            x_pre_sorted=True,
        )
        y = self._add_sorted_low_rank_residual(y, self.down_proj, down_input, sorted_rhs, down_lhs)
        y = self._add_sorted_bias(y, self.down_proj, sorted_rhs)
        return y[inverse_order].reshape((*leading_shape, top_k, self.down_proj.output_dims))

    def _apply_activation(self, up: mx.array, gate: mx.array) -> mx.array:
        """Fuse gate and up. ``None`` keeps the historical plain SwiGLU exactly.

        The default branch is byte-for-byte the expression this class used
        before the hook existed, so every family that does not pass an
        activation is numerically unchanged.
        """

        if self.activation is None:
            return nn.silu(gate) * up
        return self.activation(up, gate)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if self._can_use_shared_sorted_route_prefill(x, indices):
            return self._shared_sorted_route_prefill(x, indices)
        gate = self.gate_proj(x, indices)
        up = self.up_proj(x, indices)
        return self.down_proj(self._apply_activation(up, gate), indices)

    @classmethod
    def from_weights(
        cls,
        *,
        gate_weight: mx.array,
        up_weight: mx.array,
        down_weight: mx.array,
        group_size: int = 512,
        code_bits: Literal[8, 16] = 8,
        activation: Callable[[mx.array, mx.array], mx.array] | None = None,
    ) -> "QuantizedVQSwitchGLU":
        return cls(
            gate_proj=QuantizedVQSwitchLinear.from_weights(gate_weight, group_size=group_size, code_bits=code_bits),
            up_proj=QuantizedVQSwitchLinear.from_weights(up_weight, group_size=group_size, code_bits=code_bits),
            down_proj=QuantizedVQSwitchLinear.from_weights(down_weight, group_size=group_size, code_bits=code_bits),
            activation=activation,
        )


class QuantizedVQGLM4MoE(nn.Module):
    def __init__(
        self,
        config: GLM4MoeRoutingConfig,
        *,
        switch_mlp: QuantizedVQSwitchGLU,
        gate_weight: mx.array | None = None,
        e_score_correction_bias: mx.array | None = None,
        shared_experts: nn.Module | None = None,
    ):
        super().__init__()
        if switch_mlp.input_dims != config.hidden_size:
            raise ValueError("switch_mlp input dimension must match config.hidden_size")
        if switch_mlp.hidden_dims != config.moe_intermediate_size:
            raise ValueError("switch_mlp hidden dimension must match config.moe_intermediate_size")
        if switch_mlp.num_experts != config.n_routed_experts:
            raise ValueError("switch_mlp expert count must match config.n_routed_experts")
        self.config = config
        self.switch_mlp = switch_mlp
        self.gate = GLM4MoEGate(
            config,
            weight=gate_weight,
            e_score_correction_bias=e_score_correction_bias,
        )
        if shared_experts is not None:
            self.shared_experts = shared_experts

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        shared_experts = self.get("shared_experts")
        if shared_experts is not None:
            y = y + shared_experts(x)
        return y
