"""MLX neural-network modules for VQ inference."""

from mlx_vq.nn.linear import QuantizedVQLinear
from mlx_vq.nn.switch_linear import HighPrecisionSwitchLinear, QuantizedVQSwitchLinear

__all__ = ["HighPrecisionSwitchLinear", "QuantizedVQLinear", "QuantizedVQSwitchLinear"]
