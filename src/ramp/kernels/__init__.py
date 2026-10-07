"""RAMP MLX/Metal kernel entry points."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.kernels",
    globals(),
    child_modules=("gather_vqmm", "metal_capability", "nax", "vq_qmm", "vq_qmv"),
)
