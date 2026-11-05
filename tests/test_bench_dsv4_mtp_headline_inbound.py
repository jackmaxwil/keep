"""Version 2 headline rows also gate on memory coming back in.

The roadmap found the quiet gate one-sided: a row could fault gigabytes of its
working set back in (page-ins, swap-ins, compressor decompressions) and still
stamp itself clean on zero pageouts and swapouts. Version 2 rows close that;
version 1 rows (the committed Task 4 evidence) keep their original rule.
"""

from __future__ import annotations

import pytest

import ramp.benchmark.metrics as metrics
from benchmarks import bench_dsv4_mtp_headline as benchmark


def _quiet():
    return {
        "enabled": True,
        "available": True,
        "quiet": True,
        "attempts": 1,
        "window_seconds": 5.0,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }


def _v2_row(**changes):
    row = {
        "schema_version": 2,
        "quiet": _quiet(),
        "pageouts_delta": 0,
        "swapouts_delta": 0,
        "inbound_gate": dict(benchmark.INBOUND_GATE),
        "pageins_delta": 144,
        "swapins_delta": 0,
        "decompressions_delta": 12,
    }
    row.update(changes)
    return row


def test_v2_row_under_the_inbound_ceiling_is_clean():
    assert benchmark._row_clean(_v2_row()) is True


@pytest.mark.parametrize(
    "changes",
    [
        # 245,255 page-ins (3.83 GB) passed the old gate in the post-fix series.
        {"pageins_delta": 245_255},
        {"pageins_delta": benchmark.INBOUND_GATE["max_pageins"] + 1},
        {"swapins_delta": 1},
        {"decompressions_delta": benchmark.INBOUND_GATE["max_decompressions"] + 1},
    ],
)
def test_v2_row_that_faulted_memory_back_in_is_dirty(changes):
    assert benchmark._row_clean(_v2_row(**changes)) is False


def test_the_same_inbound_traffic_does_not_change_a_v1_verdict():
    """Committed version 1 evidence replays under the rule it was gated by."""

    row = _v2_row(schema_version=1, pageins_delta=245_255)
    del row["inbound_gate"], row["swapins_delta"], row["decompressions_delta"]
    assert benchmark._row_clean(row) is True
    del row["schema_version"]  # rows that predate the field are version 1
    assert benchmark._row_clean(row) is True


def test_v2_outbound_failure_still_dirties_the_row():
    assert benchmark._row_clean(_v2_row(pageouts_delta=1)) is False


@pytest.mark.parametrize(
    "changes",
    [
        {"pageins_delta": None},
        {"swapins_delta": -1},
        {"decompressions_delta": 1.0},
        {"inbound_gate": None},
        # A row may not declare a laxer ceiling than the harness enforces.
        {"inbound_gate": {**benchmark.INBOUND_GATE, "max_pageins": 10**9}},
        # Malformed inbound fields are caught even when outbound already failed.
        {"pageouts_delta": 3, "swapins_delta": None},
    ],
)
def test_v2_inbound_record_is_validated_not_trusted(changes):
    with pytest.raises(ValueError, match="inbound|_delta"):
        benchmark._row_clean(_v2_row(**changes))


@pytest.mark.parametrize("version", [0, 3, "2", True])
def test_unknown_row_schema_versions_are_refused(version):
    with pytest.raises(ValueError, match="schema_version"):
        benchmark._row_clean(_v2_row(schema_version=version))


def test_metric_snapshot_reports_inbound_counters(monkeypatch):
    monkeypatch.setattr(
        metrics,
        "collect_vm_stat_counts",
        lambda: {
            "pageouts": 5,
            "swapouts": 6,
            "pageins": 1_000,
            "swapins": 7,
            "decompressions": 50,
        },
    )
    monkeypatch.setattr(metrics, "_mlx_memory_value", lambda _name: 0)
    snapshot = metrics.collect_metric_snapshot(
        previous_vm_stat_counts={
            "pageouts": 5,
            "swapouts": 6,
            "pageins": 856,
            "swapins": 7,
            "decompressions": 38,
        }
    )
    assert snapshot["pageins_delta"] == 144
    assert snapshot["swapins_delta"] == 0
    assert snapshot["decompressions_delta"] == 12
    assert snapshot["pageins_total"] == 1_000
    assert set(metrics.INBOUND_VM_COUNTERS) == {"pageins", "swapins", "decompressions"}


def test_vm_stat_parser_reads_the_inbound_lines():
    counts = metrics.parse_vm_stat_counts(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pageins:                                123.\n"
        "Swapins:                                  4.\n"
        "Decompressions:                          56.\n"
    )
    assert counts == {"pageins": 123, "swapins": 4, "decompressions": 56}
