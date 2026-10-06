"""AWS Lambda entrypoint for retained operator disposition."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_operator_disposition_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_OPERATOR_DISPOSITION",
    coordinator_factory=make_retained_operator_disposition_coordinator,
)

__all__ = ["main"]
