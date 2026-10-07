"""KEEP quantization helpers."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.quant",
    globals(),
    child_modules=("rht", "rtn"),
)
