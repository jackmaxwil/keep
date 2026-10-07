"""KEEP declarative build recipes (alias of ``mlx_vq.build``)."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.build",
    globals(),
    child_modules=(
        "cli",
        "executor",
        "gate_profiles",
        "gates",
        "hashing",
        "highlevel",
        "ledger",
        "ops",
        "plan_next",
        "promote",
        "recipe",
        "runner",
    ),
)
