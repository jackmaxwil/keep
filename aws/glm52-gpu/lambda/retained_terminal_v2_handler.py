"""AWS Lambda entrypoint for the retained terminal-v2 writer."""

from glm52_enforcement.task12_lambda_adapters import (
    build_lambda_handler,
    make_retained_terminal_v2_coordinator,
)
from glm52_enforcement.task12_support_terminal_v2 import (
    main as support_main,
)

_retained_main = build_lambda_handler(
    handler_kind="RETAINED_TERMINAL_V2",
    coordinator_factory=make_retained_terminal_v2_coordinator,
)


def main(event, context, **kwargs):
    """Dispatch the existing published family across two closed envelopes."""

    if (
        type(event) is dict
        and event.get("record_type")
        == "glm52_task12_support_terminal_v2_request_v1"
    ):
        return support_main(event, context, **kwargs)
    return _retained_main(event, context, **kwargs)


__all__ = ["main"]
