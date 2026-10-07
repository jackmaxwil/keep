"""KEEP VQ codebook and vector-quantization helpers."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.codebook",
    globals(),
    child_modules=("e8",),
)
