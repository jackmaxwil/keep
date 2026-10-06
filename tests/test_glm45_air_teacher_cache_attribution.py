from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mlx_vq.quality.teacher_cache_attribution import (
    TeacherCacheRun,
    build_teacher_cache_attribution_report,
)


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_teacher_cache_attribution.py"
    )
    spec = importlib.util.spec_from_file_location("teacher_cache_attribution_cli_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    prompt_id: str,
    *,
    ppl_ratio: float,
    mean_kld: float,
    token_klds: list[float],
    teacher_top1_ids: list[int],
    vq_top1_ids: list[int],
) -> dict:
    return {
        "prompt_id": prompt_id,
        "row_index": 0,
        "target_token_count": len(token_klds),
        "target_token_ids": list(range(100, 100 + len(token_klds))),
        "positions": list(range(10, 10 + len(token_klds))),
        "nll_delta": float(np.log(ppl_ratio)),
        "ppl_ratio": ppl_ratio,
        "mean_kld": mean_kld,
        "token_klds": token_klds,
        "teacher_top1_ids": teacher_top1_ids,
        "vq_top1_ids": vq_top1_ids,
        "top1_agreement": sum(
            1 for teacher_id, vq_id in zip(teacher_top1_ids, vq_top1_ids)
            if teacher_id == vq_id
        ) / len(token_klds),
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }


def test_attribution_report_ranks_prompts_tokens_and_violations() -> None:
    records = [
        _record(
            "short_math",
            ppl_ratio=2.0,
            mean_kld=1.0,
            token_klds=[0.2, 3.5, 0.4],
            teacher_top1_ids=[1, 2, 3],
            vq_top1_ids=[1, 99, 3],
        ),
        _record(
            "capital_france",
            ppl_ratio=1.2,
            mean_kld=0.3,
            token_klds=[0.1, 0.8],
            teacher_top1_ids=[4, 5],
            vq_top1_ids=[44, 5],
        ),
    ]
    report = build_teacher_cache_attribution_report(
        TeacherCacheRun(label="onebit", jsonl="baseline.jsonl", records=records),
        top_tokens=3,
    )

    assert report["summary"]["record_count"] == 2
    assert report["summary"]["top1_flip_count"] == 2
    assert report["prompt_rankings"][0]["prompt_id"] == "short_math"
    assert report["top_kld_tokens"][0]["prompt_id"] == "short_math"
    assert report["top_kld_tokens"][0]["token_index"] == 1
    assert report["top_kld_tokens"][0]["top1_match"] is False
    assert report["next_probe_targets"][0]["prompt_id"] == "short_math"
    assert {
        (violation["metric"], violation.get("prompt_id"))
        for violation in report["quality_floor_violations"]
    } >= {
        ("mean_ppl_ratio", None),
        ("mean_top1_agreement", None),
        ("per_prompt_ppl_ratio", "short_math"),
        ("robust_control_ppl_ratio", "capital_france"),
    }


def test_candidate_comparison_aligns_prompt_and_token_deltas() -> None:
    baseline_records = [
        _record(
            "short_math",
            ppl_ratio=2.0,
            mean_kld=1.0,
            token_klds=[0.2, 3.5],
            teacher_top1_ids=[1, 2],
            vq_top1_ids=[1, 99],
        )
    ]
    candidate_records = [
        _record(
            "short_math",
            ppl_ratio=3.0,
            mean_kld=1.5,
            token_klds=[0.1, 4.5],
            teacher_top1_ids=[1, 2],
            vq_top1_ids=[1, 88],
        )
    ]

    report = build_teacher_cache_attribution_report(
        TeacherCacheRun(label="baseline", jsonl="baseline.jsonl", records=baseline_records),
        candidates=[
            TeacherCacheRun(label="candidate", jsonl="candidate.jsonl", records=candidate_records)
        ],
        top_tokens=2,
    )

    comparison = report["candidate_comparisons"][0]
    assert comparison["delta_vs_baseline"]["mean_ppl_ratio"] == 1.0
    assert comparison["prompt_deltas"][0]["delta_mean_kld"] == 0.5
    assert comparison["top_worsened_tokens"][0]["token_index"] == 1
    assert comparison["top_worsened_tokens"][0]["delta_token_kld"] == 1.0
    assert comparison["top_improved_tokens"][0]["token_index"] == 0
    assert comparison["top_improved_tokens"][0]["delta_token_kld"] == -0.1


def test_attribution_cli_writes_report(tmp_path) -> None:
    _load_cli_module()
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output_path = tmp_path / "report.json"
    baseline_record = _record(
        "short_math",
        ppl_ratio=2.0,
        mean_kld=1.0,
        token_klds=[2.5],
        teacher_top1_ids=[10],
        vq_top1_ids=[11],
    )
    candidate_record = _record(
        "short_math",
        ppl_ratio=2.5,
        mean_kld=1.2,
        token_klds=[3.0],
        teacher_top1_ids=[10],
        vq_top1_ids=[12],
    )
    baseline_path.write_text(json.dumps(baseline_record) + "\n", encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_record) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/analyze_glm45_air_teacher_cache_attribution.py",
            "--baseline-jsonl",
            str(baseline_path),
            "--candidate-jsonl",
            f"candidate={candidate_path}",
            "--output-json",
            str(output_path),
            "--top-tokens",
            "1",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 1
    assert summary["top_kld_prompt"] == "short_math"
    assert report["candidate_comparisons"][0]["label"] == "candidate"
