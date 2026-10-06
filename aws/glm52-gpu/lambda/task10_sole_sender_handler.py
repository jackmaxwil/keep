"""Published Lambda adapter for the retained Task 10 sole sender."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task10_sole_sender import (  # noqa: E402
    Task10SoleSenderServices,
    run_task10_sole_sender,
)


def _runtime_services() -> Task10SoleSenderServices:
    """Build runtime adapters lazily; imports remain packaging-safe."""

    from task10_sole_sender_runtime import build_runtime_services

    return build_runtime_services(
        ledger_table=os.environ.get("GLM52_TASK10_LEDGER_TABLE"),
        campaign_bucket=os.environ.get("GLM52_TASK10_CAMPAIGN_BUCKET"),
        region=os.environ.get("GLM52_TASK10_REGION"),
        clock=lambda: datetime.now(timezone.utc),
    )


def main(
    event: object,
    context: object,
    *,
    _services: Task10SoleSenderServices | None = None,
):
    del context
    services = _services if _services is not None else _runtime_services()
    return run_task10_sole_sender(event, services=services)
