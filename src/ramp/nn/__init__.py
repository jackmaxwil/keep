"""RAMP quantized neural-network modules."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.nn",
    globals(),
    child_modules=("linear", "switch_linear"),
)
