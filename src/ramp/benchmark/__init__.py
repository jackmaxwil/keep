"""RAMP benchmark and runtime-measurement helpers."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.benchmark",
    globals(),
    child_modules=("glm45_air", "metrics", "nax_audit", "projection_kernels", "quant_compare", "vq2_preflight"),
)
