"""Append-only build ledger: the source of truth for resume.

One JSONL file per recipe build root. Every step transition appends one
record; a ``started`` record with no terminal event marks a partial run that
gets quarantined, never reused. ``gate_failed`` evidence is kept as an
immutable rejection record — a dirty eval is a gate failure, and its
evidence file is never appended to again.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_EVENTS = ("completed", "gate_failed", "failed")


class Ledger:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event: str, *, step_id: str, step_key: str, **fields: Any) -> dict[str, Any]:
        record = {
            "event": event,
            "step_id": step_id,
            "step_key": step_key,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def latest_terminal(self, step_key: str) -> dict[str, Any] | None:
        latest: dict[str, Any] | None = None
        for record in self.records():
            if record.get("step_key") == step_key and record.get("event") in TERMINAL_EVENTS:
                latest = record
        return latest

    def started_without_terminal(self, step_key: str) -> bool:
        started = False
        for record in self.records():
            if record.get("step_key") != step_key:
                continue
            if record.get("event") == "started":
                started = True
            elif record.get("event") in TERMINAL_EVENTS:
                started = False
        return started
