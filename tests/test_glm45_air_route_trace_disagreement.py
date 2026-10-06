from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "analyze_glm45_air_route_trace_disagreement.py"
    )
    spec = importlib.util.spec_from_file_location("route_trace_disagreement_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_route_trace_disagreement_reports_layer_and_expert_deltas() -> None:
    cli = _load_cli()
    teacher_rows = [
        {
            "prompt_id": "p0",
            "route_trace": {
                "36": {
                    "top_k": 2,
                    "token_expert_indices": [[7, 3], [4, 5]],
                }
            },
        }
    ]
    student_rows = [
        {
            "prompt_id": "p0",
            "route_trace": {
                "36": {
                    "top_k": 2,
                    "token_expert_indices": [[7, 9], [5, 4]],
                }
            },
        }
    ]

    summary = cli.summarize_route_trace_disagreement(
        teacher_rows=teacher_rows,
        student_rows=student_rows,
    )

    layer = summary["layers"]["36"]
    assert summary["matched_prompt_count"] == 1
    assert layer["token_count"] == 2
    assert layer["top1_agreement"] == 0.5
    assert layer["mean_topk_overlap"] == 0.75
    assert layer["expert_bias_delta"] == [
        {"expert": 4, "delta": 0.5},
        {"expert": 5, "delta": -0.5},
        {"expert": 7, "delta": 0.0},
    ]
    assert layer["hard_tokens"] == [
        {
            "prompt_id": "p0",
            "token_index": 1,
            "teacher_top1": 4,
            "student_top1": 5,
            "topk_overlap": 1.0,
        }
    ]


def test_summarize_route_trace_disagreement_tracks_missing_prompt_and_layer() -> None:
    cli = _load_cli()
    summary = cli.summarize_route_trace_disagreement(
        teacher_rows=[
            {"prompt_id": "p0", "route_trace": {"36": {"top_k": 1, "token_expert_indices": [[1]]}}},
            {"prompt_id": "p1", "route_trace": {"36": {"top_k": 1, "token_expert_indices": [[2]]}}},
        ],
        student_rows=[
            {"prompt_id": "p0", "route_trace": {"41": {"top_k": 1, "token_expert_indices": [[1]]}}},
        ],
    )

    assert summary["matched_prompt_count"] == 1
    assert summary["missing_student_prompt_ids"] == ["p1"]
    assert summary["missing_student_layers"] == [{"layer": "36", "prompt_id": "p0"}]
    assert summary["layers"] == {}
