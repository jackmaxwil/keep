"""Published numeric-version wrapper for retained Task 9 liability watch."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Optional


from glm52_enforcement.task9_liability_watcher_runtime import (
    build_aws_task9_liability_watcher,
    run_task9_liability_watch,
)


def _runtime_watcher(context: object):
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    return build_aws_task9_liability_watcher(
        table_name=os.environ.get("GLM52_TASK9_LEDGER_TABLE"),
        region=os.environ.get("GLM52_TASK9_REGION"),
        remaining_time_in_millis=remaining,
        clock=lambda: datetime.now(timezone.utc),
    )


def main(
    event: object,
    context: object,
    *,
    _watcher: Optional[object] = None,
):
    watcher = _watcher if _watcher is not None else _runtime_watcher(context)
    return run_task9_liability_watch(
        event,
        context,
        watcher=watcher,
    )
