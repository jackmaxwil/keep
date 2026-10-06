from __future__ import annotations

from pathlib import Path

from mlx_vq.convert.recovery_checkpoint import (
    ProgressReport,
    completed_experts,
    record_expert_done,
)


def test_expert_checkpoint_is_durable_and_resumable(tmp_path: Path) -> None:
    key = "layer-00075-gate_proj"
    assert completed_experts(tmp_path, key) == frozenset()
    record_expert_done(tmp_path, key, 0)
    record_expert_done(tmp_path, key, 1)
    assert completed_experts(tmp_path, key) == frozenset({0, 1})
    # A different group is independent.
    assert completed_experts(tmp_path, "layer-00075-up_proj") == frozenset()


def test_progress_report_eta() -> None:
    report = ProgressReport(done=42, total=168, seconds_elapsed=210.0)
    assert abs(report.fraction - 0.25) < 1e-9
    # 42 experts in 210s -> 5s/expert -> 126 remaining -> 630s ETA
    assert abs(report.eta_seconds - 630.0) < 1e-6


def test_progress_report_zero_done_has_no_false_eta() -> None:
    report = ProgressReport(done=0, total=168, seconds_elapsed=10.0)
    assert report.eta_seconds is None
