"""MLX neural-network modules for VQ inference."""

from ramp.nn.linear import QuantizedVQLinear
from ramp.nn.switch_linear import HighPrecisionSwitchLinear, QuantizedVQSwitchLinear

__all__ = ["HighPrecisionSwitchLinear", "QuantizedVQLinear", "QuantizedVQSwitchLinear"]
