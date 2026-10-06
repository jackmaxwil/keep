"""Legacy KEEP/RAMP compatibility package.

New code should prefer ``keep`` for method and quality tooling and ``ramp`` for
runtime and kernel surfaces. The ``mlx_vq`` import path remains supported while
the active GLM-4.5-Air ladder is stabilized.
"""

from mlx_vq.env import EnvironmentReport, verify_environment

__all__ = ["EnvironmentReport", "verify_environment"]
