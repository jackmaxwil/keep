"""AWS Lambda entrypoint for SSM worker-drain signaling only.

Generation terminal publication belongs to the later retained TerminalV2
control-plane step after terminality and spend closure are proven.
"""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_worker_drain_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_WORKER_DRAIN",
    coordinator_factory=make_retained_worker_drain_coordinator,
)

__all__ = ["main"]
