"""Lambda root shim for the one-shot Task 11 fence executor."""

from glm52_enforcement.support_fence_handler import main


def lambda_handler(event, context):
    return main(event, context)


__all__ = ["lambda_handler", "main"]
