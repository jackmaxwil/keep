from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mlx_vq.quality.teacher_cache_row_compare import (
    compare_teacher_cache_rows,
    merge_clean_repaired_teacher_cache_records,
    merge_repaired_teacher_cache_records,
    summarize_clean_repaired_teacher_cache_records,
)


def _row(
    row_index: int,
    *,
    prompt_id: str,
    ppl_ratio: float,
    mean_kld: float,
    token_klds: list[float],
    top1_agreement: float,
    pageouts_delta: int = 0,
    swapouts_delta: int = 0,
) -> dict:
    return {
        "row_index": row_index,
        "prompt_id": prompt_id,
        "ppl_ratio": ppl_ratio,
        "nll_delta": 0.0,
        "mean_kld": mean_kld,
        "token_klds": token_klds,
        "top1_agreement": top1_agreement,
        "kld_mode": "exact_full_logits",
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
    }


def test_row_compare_pairs_by_row_index_and_reports_ppl_kld_cross_pressure() -> None:
    baseline = [
        _row(0, prompt_id="same_family", ppl_ratio=2.0, mean_kld=1.0, token_klds=[1.0], top1_agreement=0.5),
        _row(1, prompt_id="same_family", ppl_ratio=3.0, mean_kld=2.0, token_klds=[2.0], top1_agreement=0.25),
    ]
    candidate = [
        _row(0, prompt_id="same_family", ppl_ratio=2.2, mean_kld=0.8, token_klds=[0.8], top1_agreement=0.75),
        _row(1, prompt_id="same_family", ppl_ratio=2.5, mean_kld=1.5, token_klds=[1.5], top1_agreement=0.25),
    ]

    report = compare_teacher_cache_rows(
        baseline,
        candidate,
        candidate_label="hin_probe",
        top_rows=2,
    )

    assert report["candidate_label"] == "hin_probe"
    assert report["decision"]["label"] == "RECOVER"
    assert report["max_ppl_row"]["row_index"] == 1
    assert report["max_ppl_row"]["delta_ppl_ratio"] == -0.5
    assert report["kld_improved_ppl_worsened_rows"][0]["row_index"] == 0
    assert report["kld_improved_ppl_worsened_rows"][0]["delta_mean_kld"] == -0.19999999999999996
    assert report["top_mean_kld_improved_rows"][0]["row_index"] == 1
    assert report["gap_closure"]["mean_kld"]["closed_fraction"] > 0.0


def test_repaired_records_replace_dirty_rows_by_row_index() -> None:
    original = [
        _row(0, prompt_id="row0", ppl_ratio=2.0, mean_kld=1.0, token_klds=[1.0], top1_agreement=0.5),
        _row(1, prompt_id="row1", ppl_ratio=3.0, mean_kld=2.0, token_klds=[2.0], top1_agreement=0.25),
    ]
    repair = [
        _row(1, prompt_id="row1", ppl_ratio=2.5, mean_kld=1.5, token_klds=[1.5], top1_agreement=0.5),
    ]

    merged = merge_repaired_teacher_cache_records(original, repair)

    assert [row["row_index"] for row in merged] == [0, 1]
    assert merged[1]["ppl_ratio"] == 2.5


def test_clean_repaired_records_ignore_dirty_repairs() -> None:
    original = [
        _row(0, prompt_id="row0", ppl_ratio=2.0, mean_kld=1.0, token_klds=[1.0], top1_agreement=0.5),
        _row(
            1,
            prompt_id="row1",
            ppl_ratio=3.0,
            mean_kld=2.0,
            token_klds=[2.0],
            top1_agreement=0.25,
            pageouts_delta=7,
        ),
        _row(
            2,
            prompt_id="row2",
            ppl_ratio=4.0,
            mean_kld=3.0,
            token_klds=[3.0],
            top1_agreement=0.0,
            pageouts_delta=9,
        ),
    ]
    repairs = [
        _row(
            1,
            prompt_id="row1",
            ppl_ratio=2.5,
            mean_kld=1.5,
            token_klds=[1.5],
            top1_agreement=0.5,
            pageouts_delta=1,
        ),
        _row(2, prompt_id="row2", ppl_ratio=2.75, mean_kld=1.75, token_klds=[1.75], top1_agreement=0.75),
    ]

    merged = merge_clean_repaired_teacher_cache_records(original, repairs)
    report = summarize_clean_repaired_teacher_cache_records(original, repairs)

    assert merged[1]["ppl_ratio"] == 3.0
    assert merged[2]["ppl_ratio"] == 2.75
    assert report["clean_repaired_row_indices"] == [2]
    assert report["clean_record_count"] == 2
    assert report["dirty_row_indices"] == [1]
    assert report["all_memory_clean"] is False


def test_cli_outputs_speed_fail_when_speed_gate_fails(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    speed_gate_path = tmp_path / "speed.json"
    output_path = tmp_path / "report.json"
    baseline_path.write_text(
        json.dumps(_row(0, prompt_id="row0", ppl_ratio=2.0, mean_kld=1.0, token_klds=[1.0], top1_agreement=0.5)) + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_row(0, prompt_id="row0", ppl_ratio=1.8, mean_kld=0.8, token_klds=[0.8], top1_agreement=0.5)) + "\n",
        encoding="utf-8",
    )
    speed_gate_path.write_text(
        json.dumps({"speed_verdict": "FAIL"}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "benchmarks/compare_glm45_air_teacher_cache_rows.py",
            "--baseline-jsonl",
            str(baseline_path),
            "--candidate-jsonl",
            str(candidate_path),
            "--candidate-label",
            "sparse_probe",
            "--speed-gate-json",
            str(speed_gate_path),
            "--output-json",
            str(output_path),
        ],
        check=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["decision"]["label"] == "SPEED_FAIL"
    assert report["decision"]["quality_label_without_speed"] == "RECOVER"


def test_clean_repair_summary_cli_filters_summary_rows_and_writes_report(tmp_path) -> None:
    base_path = tmp_path / "base.jsonl"
    repair_path = tmp_path / "repair.jsonl"
    output_path = tmp_path / "summary.json"
    base_path.write_text(
        "\n".join(
            [
                json.dumps(_row(0, prompt_id="row0", ppl_ratio=2.0, mean_kld=1.0, token_klds=[1.0], top1_agreement=0.5)),
                json.dumps(
                    _row(
                        1,
                        prompt_id="row1",
                        ppl_ratio=3.0,
                        mean_kld=2.0,
                        token_klds=[2.0],
                        top1_agreement=0.25,
                        pageouts_delta=7,
                    )
                ),
                json.dumps({"record_type": "air_vq_teacher_cache_eval_summary", "record_count": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repair_path.write_text(
        json.dumps(_row(1, prompt_id="row1", ppl_ratio=2.5, mean_kld=1.5, token_klds=[1.5], top1_agreement=0.5)) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "benchmarks/summarize_glm45_air_teacher_cache_repairs.py",
            "--base-jsonl",
            str(base_path),
            "--repair-jsonl",
            str(repair_path),
            "--label",
            "unit",
            "--output-json",
            str(output_path),
        ],
        check=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["label"] == "unit"
    assert report["row_count"] == 2
    assert report["clean_record_count"] == 2
    assert report["dirty_row_count"] == 0
    assert report["clean_repaired_row_indices"] == [1]
