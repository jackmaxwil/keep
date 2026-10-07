"""KEEP calibration, distillation, and quality-recovery helpers."""

from keep._alias import install_alias_package

__getattr__, __dir__ = install_alias_package(
    __name__,
    "mlx_vq.quality",
    globals(),
    child_modules=(
        "calibration",
        "calibration_importance",
        "dynamic_precision",
        "gates",
        "glm45_air",
        "hessian_rounding",
        "imatrix",
        "imatrix_collection",
        "kronecker_hessian",
        "layer_probe_attribution",
        "mlx_surrogate",
        "plan_next",
        "prompts",
        "rc_gates",
        "rht_materialization",
        "selective_precision",
        "teacher_cache",
        "teacher_cache_attribution",
        "teacher_cache_row_compare",
    ),
)
