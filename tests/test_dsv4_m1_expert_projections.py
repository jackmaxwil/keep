from __future__ import annotations

import pytest

from benchmarks.bench_dsv4_m1_expert_projections import _run_row, summarize_m1_rows


def _row(run: int, *, vq_scale: float = 1.0, order: str | None = None) -> dict:
    projections = {
        "gate_proj": {"vq_median_ms": 2.0 * vq_scale, "source_median_ms": 3.0},
        "up_proj": {"vq_median_ms": 2.0 * vq_scale, "source_median_ms": 3.0},
        "down_proj": {"vq_median_ms": 4.0 * vq_scale, "source_median_ms": 6.0},
    }
    return {
        "record_type": "dsv4_m1_expert_projection_row_v1",
        "fresh_process": True,
        "pid": run,
        "run_index": run,
        "order": order or ("vq_first" if run % 2 else "source_first"),
        "identities": {
            "artifact_manifest_sha256": "artifact",
            "checkpoint_index_sha256": "source",
        },
        "kernel_identity": "nax_e8p_fp16_sorted_steel_m32n64",
        "kernel_calls": 1,
        "source_backend": "mx.gather_qmm:mxfp4",
        "pageouts_delta": 0,
        "swapouts_delta": 0,
        "projections": projections,
    }


def test_summary_uses_clean_fresh_process_pairs_and_gates_the_projection_total() -> (
    None
):
    rows = [_row(index) for index in range(1, 6)]
    rows.append({**_row(6, vq_scale=10.0), "pageouts_delta": 1})

    summary = summarize_m1_rows(rows, min_clean_rows=5)

    assert summary["clean_rows"] == 5
    assert summary["dirty_rows"] == 1
    assert summary["orders"] == {"source_first": 2, "vq_first": 3}
    assert summary["projections"]["gate_proj"]["median_paired_source_over_vq"] == 1.5
    assert summary["projection_total"]["median_paired_source_over_vq"] == 1.5
    assert summary["gate"] == "PASS"
    assert summary["scope"] == "component_level_m1_expert_projection_only"


def test_summary_fails_closed_on_identity_drift_or_a_non_faster_total() -> None:
    slower = summarize_m1_rows(
        [_row(index, vq_scale=2.0) for index in range(1, 5)], min_clean_rows=4
    )
    assert slower["projection_total"]["median_paired_source_over_vq"] == 0.75
    assert slower["gate"] == "STOP"

    drifted = [_row(index) for index in range(1, 5)]
    drifted[-1]["identities"] = {
        **drifted[-1]["identities"],
        "checkpoint_index_sha256": "other",
    }
    with pytest.raises(ValueError, match="identity drift"):
        summarize_m1_rows(drifted, min_clean_rows=4)

    missing_dispatch = [_row(index) for index in range(1, 5)]
    missing_dispatch[-1]["kernel_calls"] = 0
    with pytest.raises(ValueError, match="did not call"):
        summarize_m1_rows(missing_dispatch, min_clean_rows=4)


def test_summary_fails_closed_when_a_fresh_process_pid_is_reused() -> None:
    rows = [_row(index) for index in range(1, 5)]
    rows[-1]["pid"] = rows[0]["pid"]

    with pytest.raises(ValueError, match="fresh-process PID"):
        summarize_m1_rows(rows, min_clean_rows=4)


def test_run_rejects_an_empty_custom_wired_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB", raising=False)
    monkeypatch.setenv("GLM_MLX_WIRED_LIMIT_GB", "")

    with pytest.raises(RuntimeError, match="custom wired-limit variables"):
        _run_row(None)  # type: ignore[arg-type]
