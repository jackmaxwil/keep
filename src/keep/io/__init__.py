"""KEEP artifact IO, manifests, and sidecar helpers."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.io",
    globals(),
    child_modules=(
        "continuous_sidecar",
        "load",
        "router_correction",
        "schema",
        "source_safetensors",
        "sparse_residual",
    ),
)
