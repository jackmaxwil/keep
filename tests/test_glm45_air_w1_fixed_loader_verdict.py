from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "summarize_glm45_air_w1_fixed_loader_bump_verdict.py"
    )
    spec = importlib.util.spec_from_file_location("w1_fixed_loader_verdict_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _eval_row(
    *,
    prompt_id: str,
    top1_agreement: float,
    mean_kld: float,
    ppl_ratio: float,
    pageouts_delta: int = 0,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "row_index": 0,
        "nll_delta": 0.0,
        "ppl_ratio": ppl_ratio,
        "mean_kld": mean_kld,
        "token_klds": [mean_kld],
        "top1_agreement": top1_agreement,
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": 0,
    }


def test_fixed_loader_bump_verdict_summarizes_existing_evidence(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    audit = _write_json(
        tmp_path / "audit.json",
        {
            "artifact_dir": "candidate",
            "counted_resident_total_bytes": 100,
            "resident_byte_reduction_bytes": 0,
            "source_non_expert_delta_bytes": 0,
            "source_skipped_by_surface_total_bytes": 0,
            "source_non_expert_complete": True,
        },
    )
    report = _write_jsonl(
        tmp_path / "report.jsonl",
        [
            _eval_row(
                prompt_id=f"report_route_{index:03d}",
                top1_agreement=0.8,
                mean_kld=0.4,
                ppl_ratio=1.04,
                pageouts_delta=1 if index == 127 else 0,
            )
            for index in range(128)
        ],
    )
    selection = _write_jsonl(
        tmp_path / "selection.jsonl",
        [
            _eval_row(
                prompt_id=f"select_route_{index:03d}",
                top1_agreement=0.82,
                mean_kld=0.41,
                ppl_ratio=1.03,
            )
            for index in range(128)
        ],
    )
    lane_s = _write_jsonl(
        tmp_path / "lane_s.jsonl",
        [
            {
                "invalid_memory_pressure": True,
                "pageouts_delta": 39,
                "swapouts_delta": 0,
                "speed_claim": False,
                "lane_s_claim": False,
            }
        ],
    )

    verdict = cli.build_verdict(
        resident_audit_path=audit,
        report_jsonl_path=report,
        selection_jsonl_path=selection,
        lane_s_jsonl_path=lane_s,
    )

    assert verdict["decision"] == (
        "bump_lm_head_embed_fixed_loader_metric_backed_not_promotable"
    )
    assert verdict["implementation_bug_fixed"] is True
    assert verdict["non_expert_precision_exhausted"] is False
    assert verdict["quality_promotable"] is False
    assert verdict["residency_promotable"] is False
    assert verdict["lane_s_valid"] is False
    assert verdict["heavy_slice_rule_satisfied"] is True
    assert verdict["resident_audit"]["resident_byte_reduction_bytes"] == 0
    assert verdict["full_split_record_count_satisfied"] is True
    assert verdict["report"]["record_count"] == 128
    assert verdict["report"]["clean_record_count"] == 127
    assert verdict["selection"]["mean_top1_agreement"] == pytest.approx(0.82)
    assert verdict["lane_s"]["invalid_memory_pressure_row_count"] == 1
    assert verdict["lane_s"]["pageouts_delta_values"] == [39]
    assert verdict["peer2_used"] is False
    assert verdict["rdma_jaccl_touched"] is False
