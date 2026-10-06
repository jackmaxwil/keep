from __future__ import annotations

from benchmarks.bench_e8p_search import acceptance_gate, project_worst8_hours


def test_projection_matches_spike_scale() -> None:
    # Spike measured ~1.04M cw/s on Metal -> ~6.8 h for worst8.
    hours = project_worst8_hours(1_040_000)
    assert 5.0 < hours < 9.0


def test_acceptance_gate_rejects_multiday() -> None:
    assert acceptance_gate(project_worst8_hours(22_900)) is False  # exhaustive ~308 h
    assert acceptance_gate(project_worst8_hours(1_040_000)) is True
