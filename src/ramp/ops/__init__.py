"""RAMP routed expert operations."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.ops",
    globals(),
    child_modules=("vq_switch",),
)
