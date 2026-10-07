"""KEEP: KL-distilled Expert Encoding and Precision for MLX routed MoE VQ."""

from __future__ import annotations

from mlx_vq.env import EnvironmentReport, verify_environment

PROJECT_NAME = "KEEP"
METHOD_NAME = "KL-distilled Expert Encoding and Precision"
LEGACY_PACKAGE = "mlx_vq"

__all__ = [
    "EnvironmentReport",
    "LEGACY_PACKAGE",
    "METHOD_NAME",
    "PROJECT_NAME",
    "verify_environment",
]
