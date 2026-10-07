from __future__ import annotations

import json

import ramp.benchmark.metrics as metrics_module
from ramp.benchmark.metrics import MemoryPhaseTracer, collect_metric_snapshot, parse_vm_stat_counts


def test_metric_snapshot_has_required_keys() -> None:
    snapshot = collect_metric_snapshot()

    assert snapshot["mlx_active_bytes"] is not None
    assert snapshot["mlx_peak_bytes"] is not None
    assert snapshot["mlx_cache_bytes"] is not None
    assert snapshot["rss_bytes"] > 0
    assert "pageouts_delta" in snapshot
    assert "swapouts_delta" in snapshot
    assert set(snapshot) >= {
        "mlx_active_bytes",
        "mlx_peak_bytes",
        "mlx_cache_bytes",
        "rss_bytes",
        "pageouts_delta",
        "swapouts_delta",
    }


def test_metric_snapshot_is_json_serializable() -> None:
    snapshot = collect_metric_snapshot()

    encoded = json.dumps(snapshot, sort_keys=True)

    assert "mlx_active_bytes" in encoded


def test_parse_vm_stat_counts_extracts_reusable_vm_counters() -> None:
    output = """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pages active:                             22222.
Pages occupied by compressor:             333.
Pageouts:                                67.
Swapouts:                                89.
"""

    counts = parse_vm_stat_counts(output)

    assert counts == {
        "pages_free": 12345,
        "pages_active": 22222,
        "pages_occupied_by_compressor": 333,
        "pageouts": 67,
        "swapouts": 89,
    }


def test_memory_phase_tracer_records_labeled_vm_deltas(monkeypatch) -> None:
    samples = iter(
        [
            {"pageouts": 10, "swapouts": 20, "pages_free": 100},
            {"pageouts": 10, "swapouts": 20, "pages_free": 90},
            {"pageouts": 13, "swapouts": 20, "pages_free": 80},
        ]
    )

    monkeypatch.setattr(metrics_module, "collect_vm_stat_counts", lambda: next(samples))
    monkeypatch.setattr(metrics_module, "_mlx_memory_value", lambda name: {"get_active_memory": 1000, "get_peak_memory": 2000, "get_cache_memory": 3000}[name])
    monkeypatch.setattr(metrics_module, "current_rss_bytes", lambda: 4000)

    tracer = MemoryPhaseTracer(enabled=True)

    tracer.mark("before_load")
    tracer.mark("after_load")
    tracer.mark("after_prefill")

    assert tracer.records == [
        {
            "label": "before_load",
            "available": True,
            "pageouts_total": 10,
            "swapouts_total": 20,
            "pageouts_delta": None,
            "swapouts_delta": None,
            "pages_free": 100,
            "mlx_active_bytes": 1000,
            "mlx_peak_bytes": 2000,
            "mlx_cache_bytes": 3000,
            "rss_bytes": 4000,
        },
        {
            "label": "after_load",
            "available": True,
            "pageouts_total": 10,
            "swapouts_total": 20,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "pages_free": 90,
            "mlx_active_bytes": 1000,
            "mlx_peak_bytes": 2000,
            "mlx_cache_bytes": 3000,
            "rss_bytes": 4000,
        },
        {
            "label": "after_prefill",
            "available": True,
            "pageouts_total": 13,
            "swapouts_total": 20,
            "pageouts_delta": 3,
            "swapouts_delta": 0,
            "pages_free": 80,
            "mlx_active_bytes": 1000,
            "mlx_peak_bytes": 2000,
            "mlx_cache_bytes": 3000,
            "rss_bytes": 4000,
        },
    ]
