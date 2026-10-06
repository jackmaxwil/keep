"""AWS Lambda entrypoint for the retained orphan auditor."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_orphan_audit_coordinator,
)

main = build_lambda_handler(
    handler_kind="RETAINED_ORPHAN_AUDIT",
    coordinator_factory=make_retained_orphan_audit_coordinator,
)

__all__ = ["main"]
