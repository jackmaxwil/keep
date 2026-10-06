"""AWS Lambda entrypoint for retained snapshot cleanup."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_snapshot_cleanup_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_SNAPSHOT_CLEANUP",
    coordinator_factory=make_retained_snapshot_cleanup_coordinator,
)

__all__ = ["main"]
