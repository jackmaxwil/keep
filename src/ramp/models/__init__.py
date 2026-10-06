"""RAMP routed MoE model adapters."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.models",
    globals(),
    child_modules=(
        "glm4_moe_adapter",
        "glm45_air_vq_adapter",
        "glm52_policy",
        "glm52_vq_adapter",
        "profiles",
        "qwen_moe_adapter",
    ),
)

# ``registry.py`` is a real module that lives physically in this package
# (canonical home for the cutover), not an alias of ``mlx_vq.models``. The
# ``install_alias_package`` call above keeps this package's own directory at
# the front of ``__path__`` precisely so this import resolves to the local
# file instead of raising. It is imported eagerly (it is metadata only -- no
# MLX) so ``ramp.models.registry`` is reachable as an attribute.
from ramp.models import registry  # noqa: E402,F401

# ``__all__`` was populated by ``install_alias_package`` from the legacy
# package via ``globals()``, so it is not a module-level binding ruff can see;
# read it back through ``globals()`` rather than referencing the name directly.
__all__ = sorted({*globals().get("__all__", ()), "registry"})
