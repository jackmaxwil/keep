from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from keep.vq.e8 import e8_1bit_packed, e8p_packed_abs_grid
from ramp.kernels.vq_qmm import vq_qmm
from ramp.kernels.vq_qmv import vq_qmv
from keep.quant.rotation import apply_rotation_mx, apply_rotation_np, validate_rotation_matrix_mx
from keep.quant.rht import apply_rht_mx, apply_rht_np
from keep.quant.rtn import quantize_weight_rtn


class QuantizedVQLinear(nn.Module):
    """Linear layer backed by E8 VQ codes and the `vq_qmv` kernel."""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        codes: mx.array,
        scales: mx.array,
        codebook: mx.array | None = None,
        bias: mx.array | None = None,
        group_size: int = 512,
        code_bits: Literal[8, 16] = 8,
        rht_signs: mx.array | None = None,
        rotation_matrix: mx.array | None = None,
    ):
        super().__init__()
        if input_dims <= 0 or output_dims <= 0:
            raise ValueError("input_dims and output_dims must be positive")
        if input_dims % 8 != 0:
            raise ValueError("input_dims must be divisible by 8")
        if input_dims % group_size != 0:
            raise ValueError("input_dims must be divisible by group_size")
        if code_bits not in (8, 16):
            raise ValueError("code_bits must be 8 or 16")
        if codes.shape != (output_dims, input_dims // 8):
            raise ValueError(f"codes must have shape ({output_dims}, {input_dims // 8}), found {codes.shape}")
        if scales.shape != (output_dims, input_dims // group_size):
            raise ValueError(f"scales must have shape ({output_dims}, {input_dims // group_size}), found {scales.shape}")
        if bias is not None and bias.shape != (output_dims,):
            raise ValueError(f"bias must have shape ({output_dims},), found {bias.shape}")
        if rht_signs is not None and rotation_matrix is not None:
            raise ValueError("rht_signs and rotation_matrix are mutually exclusive")
        if rht_signs is not None and rht_signs.shape != (input_dims,):
            raise ValueError(f"rht_signs must have shape ({input_dims},), found {rht_signs.shape}")
        if rotation_matrix is not None:
            validate_rotation_matrix_mx(rotation_matrix, dim=input_dims)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size
        self.code_bits = code_bits
        self.codes = codes
        self.scales = scales
        if codebook is None:
            codebook = mx.array(e8_1bit_packed() if code_bits == 8 else e8p_packed_abs_grid())
        self.codebook = codebook
        if bias is not None:
            self.bias = bias
        if rht_signs is not None:
            self.rht_signs = rht_signs.astype(mx.float32)
        if rotation_matrix is not None:
            self.rotation_matrix = rotation_matrix.astype(mx.float32)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"input trailing dimension must be {self.input_dims}, found {x.shape[-1]}")

        rht_signs = self.get("rht_signs")
        if rht_signs is not None:
            x = apply_rht_mx(x, rht_signs)
        rotation_matrix = self.get("rotation_matrix")
        if rotation_matrix is not None:
            x = apply_rotation_mx(x, rotation_matrix)
        flat = x.reshape((-1, self.input_dims))
        if flat.shape[0] == 1:
            y_flat = vq_qmv(
                flat[0],
                self.codes,
                self.scales,
                self.codebook,
                in_dim=self.input_dims,
                out_dim=self.output_dims,
                group_size=self.group_size,
                code_bits=self.code_bits,
            ).reshape((1, self.output_dims))
        else:
            y_flat = vq_qmm(
                flat,
                self.codes,
                self.scales,
                self.codebook,
                in_dim=self.input_dims,
                out_dim=self.output_dims,
                group_size=self.group_size,
                code_bits=self.code_bits,
            )
        y = y_flat.reshape((*x.shape[:-1], self.output_dims))
        bias = self.get("bias")
        if bias is not None:
            y = y + bias
        return y

    def _extra_repr(self) -> str:
        return (
            f"{self.input_dims}, {self.output_dims}, "
            f"group_size={self.group_size}, code_bits={self.code_bits}"
        )

    @classmethod
    def from_weights(
        cls,
        weight: mx.array,
        bias: mx.array | None = None,
        *,
        group_size: int = 512,
        code_bits: Literal[8, 16] = 8,
        rht_signs: mx.array | None = None,
        rotation_matrix: mx.array | None = None,
    ) -> "QuantizedVQLinear":
        weight_np = np.array(weight.astype(mx.float32), copy=False)
        if rht_signs is not None and rotation_matrix is not None:
            raise ValueError("rht_signs and rotation_matrix are mutually exclusive")
        if rht_signs is not None:
            weight_np = apply_rht_np(weight_np, np.array(rht_signs, copy=False))
        if rotation_matrix is not None:
            weight_np = apply_rotation_np(weight_np, np.array(rotation_matrix, copy=False))
        quantized = quantize_weight_rtn(weight_np, group_size=group_size, code_bits=code_bits)
        output_dims, input_dims = weight_np.shape
        return cls(
            input_dims=input_dims,
            output_dims=output_dims,
            codes=mx.array(quantized.codes),
            scales=mx.array(quantized.scales),
            codebook=mx.array(quantized.codebook),
            bias=bias,
            group_size=quantized.group_size,
            code_bits=quantized.code_bits,
            rht_signs=rht_signs,
            rotation_matrix=rotation_matrix,
        )
