"""AWS Lambda entrypoint for the retained H1G-drained writer."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_h1g_drained_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_H1G_DRAINED",
    coordinator_factory=make_retained_h1g_drained_coordinator,
)

__all__ = ["main"]
