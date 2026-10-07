from __future__ import annotations

import platform
import re
import resource
import subprocess
import time
from collections.abc import Mapping

import mlx.core as mx


MetricSnapshot = dict[str, int | None]
VMStatCounts = dict[str, int]

_VM_STAT_RE = re.compile(r"^\s*(?P<name>[^:]+):\s+(?P<value>[0-9]+)\.?\s*$")


def _mlx_memory_value(name: str) -> int:
    getter = getattr(mx, name, None)
    if getter is None:
        metal = getattr(mx, "metal", None)
        getter = getattr(metal, name, None) if metal is not None else None
    if getter is None:
        raise RuntimeError(f"MLX memory API {name} is unavailable")
    return int(getter())


def reset_mlx_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        metal = getattr(mx, "metal", None)
        reset = getattr(metal, "reset_peak_memory", None) if metal is not None else None
    if reset is None:
        raise RuntimeError("MLX reset_peak_memory API is unavailable")
    reset()


def current_rss_bytes() -> int:
    ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return ru_maxrss
    return ru_maxrss * 1024


def parse_vm_stat_counts(output: str) -> VMStatCounts:
    counts: VMStatCounts = {}
    for line in output.splitlines():
        match = _VM_STAT_RE.match(line)
        if match is None:
            continue
        key = match.group("name").strip().lower().replace(" ", "_")
        counts[key] = int(match.group("value"))
    return counts


def collect_vm_stat_counts() -> VMStatCounts | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    counts = parse_vm_stat_counts(result.stdout)
    if "pageouts" not in counts or "swapouts" not in counts:
        return None
    return counts


def wait_for_memory_quiet(*, window_seconds: float, max_attempts: int) -> dict[str, int | float | bool | None]:
    if window_seconds <= 0:
        return {"enabled": False}
    attempts = max(1, int(max_attempts))
    last: dict[str, int | float | bool | None] = {
        "enabled": True,
        "available": False,
        "quiet": False,
        "attempts": 0,
        "window_seconds": float(window_seconds),
        "pageouts_delta": None,
        "swapouts_delta": None,
    }
    for attempt in range(1, attempts + 1):
        before = collect_vm_stat_counts()
        time.sleep(float(window_seconds))
        after = collect_vm_stat_counts()
        if before is None or after is None:
            return {
                "enabled": True,
                "available": False,
                "quiet": False,
                "attempts": attempt,
                "window_seconds": float(window_seconds),
                "pageouts_delta": None,
                "swapouts_delta": None,
            }
        pageouts_delta = int(after.get("pageouts", 0)) - int(before.get("pageouts", 0))
        swapouts_delta = int(after.get("swapouts", 0)) - int(before.get("swapouts", 0))
        last = {
            "enabled": True,
            "available": True,
            "quiet": pageouts_delta == 0 and swapouts_delta == 0,
            "attempts": attempt,
            "window_seconds": float(window_seconds),
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
        }
        if last["quiet"]:
            return last
    return last


class MemoryPhaseTracer:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.records: list[dict[str, int | str | bool | None]] = []
        self._previous: VMStatCounts | None = None

    def _memory_fields(self) -> dict[str, int | None]:
        return {
            "mlx_active_bytes": _mlx_memory_value("get_active_memory"),
            "mlx_peak_bytes": _mlx_memory_value("get_peak_memory"),
            "mlx_cache_bytes": _mlx_memory_value("get_cache_memory"),
            "rss_bytes": current_rss_bytes(),
        }

    def mark(self, label: str) -> None:
        if not self.enabled:
            return
        counts = collect_vm_stat_counts()
        if counts is None:
            self.records.append(
                {
                    "label": label,
                    "available": False,
                    "pageouts_total": None,
                    "swapouts_total": None,
                    "pageouts_delta": None,
                    "swapouts_delta": None,
                    "pages_free": None,
                    **self._memory_fields(),
                }
            )
            self._previous = None
            return
        record = {
            "label": label,
            "available": True,
            "pageouts_total": int(counts.get("pageouts", 0)),
            "swapouts_total": int(counts.get("swapouts", 0)),
            "pageouts_delta": _delta(counts, self._previous, "pageouts"),
            "swapouts_delta": _delta(counts, self._previous, "swapouts"),
            "pages_free": counts.get("pages_free"),
            **self._memory_fields(),
        }
        self.records.append(record)
        self._previous = counts


def _delta(
    counts: Mapping[str, int] | None,
    previous: Mapping[str, int] | None,
    key: str,
) -> int | None:
    if counts is None or previous is None:
        return None
    if key not in counts or key not in previous:
        return None
    return int(counts[key]) - int(previous[key])


def collect_metric_snapshot(
    *,
    previous_vm_stat_counts: Mapping[str, int] | None = None,
) -> MetricSnapshot:
    vm_counts = collect_vm_stat_counts()
    return {
        "mlx_active_bytes": _mlx_memory_value("get_active_memory"),
        "mlx_peak_bytes": _mlx_memory_value("get_peak_memory"),
        "mlx_cache_bytes": _mlx_memory_value("get_cache_memory"),
        "rss_bytes": current_rss_bytes(),
        "pageouts_total": None if vm_counts is None else vm_counts.get("pageouts"),
        "swapouts_total": None if vm_counts is None else vm_counts.get("swapouts"),
        "pageouts_delta": _delta(vm_counts, previous_vm_stat_counts, "pageouts"),
        "swapouts_delta": _delta(vm_counts, previous_vm_stat_counts, "swapouts"),
    }
