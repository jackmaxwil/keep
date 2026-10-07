"""KEEP: KL-distilled Expert Encoding and Precision for MLX routed MoE VQ."""

from __future__ import annotations

from keep.env import EnvironmentReport, verify_environment

PROJECT_NAME = "KEEP"
METHOD_NAME = "KL-distilled Expert Encoding and Precision"

__all__ = [
    "EnvironmentReport",
    "METHOD_NAME",
    "PROJECT_NAME",
    "verify_environment",
]
