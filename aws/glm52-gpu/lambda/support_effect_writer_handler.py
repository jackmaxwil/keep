"""Lambda root shim for the fixed Task 11 effect writers."""

from glm52_enforcement.support_effect_writer_handler import main


def lambda_handler(event, context):
    return main(event, context)


__all__ = ["lambda_handler", "main"]
