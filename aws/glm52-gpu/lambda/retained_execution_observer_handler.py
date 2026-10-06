"""AWS Lambda entrypoint for the retained execution observer."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_execution_observer_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_EXECUTION_OBSERVER",
    coordinator_factory=make_retained_execution_observer_coordinator,
)

__all__ = ["main"]
