"""AWS Lambda entrypoint for the retained finalizer."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_finalizer_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_FINALIZER",
    coordinator_factory=make_retained_finalizer_coordinator,
)

__all__ = ["main"]
