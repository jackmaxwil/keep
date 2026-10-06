from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"route trace row has invalid prompt_id: {prompt_id!r}")
        if prompt_id in indexed:
            raise ValueError(f"duplicate prompt_id {prompt_id!r}")
        indexed[prompt_id] = row
    return indexed


def _token_routes(layer_trace: dict[str, Any], *, prompt_id: str, layer: str) -> list[list[int]]:
    routes = layer_trace.get("token_expert_indices")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"route_trace for prompt {prompt_id!r} layer {layer} needs token_expert_indices")
    parsed: list[list[int]] = []
    for token_index, token_routes in enumerate(routes):
        if not isinstance(token_routes, list) or not token_routes:
            raise ValueError(
                f"route_trace for prompt {prompt_id!r} layer {layer} token {token_index} needs expert ids"
            )
        parsed.append([int(expert) for expert in token_routes])
    return parsed


def _layer_trace(row: dict[str, Any], *, layer: str) -> dict[str, Any] | None:
    trace = row.get("route_trace")
    if not isinstance(trace, dict):
        return None
    layer_trace = trace.get(layer)
    return layer_trace if isinstance(layer_trace, dict) else None


def _candidate_layers(
    teacher_rows: dict[str, dict[str, Any]],
    student_rows: dict[str, dict[str, Any]],
    *,
    layers: set[str] | None,
) -> list[str]:
    if layers is not None:
        return sorted(layers, key=int)
    found: set[str] = set()
    for row in [*teacher_rows.values(), *student_rows.values()]:
        trace = row.get("route_trace")
        if isinstance(trace, dict):
            found.update(str(layer) for layer in trace)
    return sorted(found, key=int)


def _bias_deltas(
    *,
    teacher_top1_counts: Counter[int],
    student_top1_counts: Counter[int],
    token_count: int,
) -> list[dict[str, Any]]:
    if token_count <= 0:
        return []
    experts = sorted(set(teacher_top1_counts) | set(student_top1_counts))
    return [
        {
            "expert": int(expert),
            "delta": float((teacher_top1_counts[expert] - student_top1_counts[expert]) / token_count),
        }
        for expert in experts
    ]


def summarize_route_trace_disagreement(
    *,
    teacher_rows: list[dict[str, Any]],
    student_rows: list[dict[str, Any]],
    layers: set[str] | None = None,
    hard_token_limit: int = 16,
) -> dict[str, Any]:
    teacher_by_prompt = _index_rows(teacher_rows)
    student_by_prompt = _index_rows(student_rows)
    prompt_ids = sorted(set(teacher_by_prompt) & set(student_by_prompt))
    missing_student_prompt_ids = sorted(set(teacher_by_prompt) - set(student_by_prompt))
    missing_teacher_prompt_ids = sorted(set(student_by_prompt) - set(teacher_by_prompt))
    layer_names = _candidate_layers(teacher_by_prompt, student_by_prompt, layers=layers)

    missing_student_layers: list[dict[str, Any]] = []
    missing_teacher_layers: list[dict[str, Any]] = []
    layer_records: dict[str, dict[str, Any]] = {}
    for layer in layer_names:
        token_count = 0
        top1_matches = 0
        topk_overlap_sum = 0.0
        teacher_top1_counts: Counter[int] = Counter()
        student_top1_counts: Counter[int] = Counter()
        hard_tokens: list[dict[str, Any]] = []

        for prompt_id in prompt_ids:
            teacher_layer = _layer_trace(teacher_by_prompt[prompt_id], layer=layer)
            student_layer = _layer_trace(student_by_prompt[prompt_id], layer=layer)
            if teacher_layer is None:
                missing_teacher_layers.append({"prompt_id": prompt_id, "layer": layer})
                continue
            if student_layer is None:
                missing_student_layers.append({"prompt_id": prompt_id, "layer": layer})
                continue

            teacher_routes = _token_routes(teacher_layer, prompt_id=prompt_id, layer=layer)
            student_routes = _token_routes(student_layer, prompt_id=prompt_id, layer=layer)
            compared = min(len(teacher_routes), len(student_routes))
            for token_index in range(compared):
                teacher_token = teacher_routes[token_index]
                student_token = student_routes[token_index]
                teacher_top1 = int(teacher_token[0])
                student_top1 = int(student_token[0])
                teacher_top1_counts[teacher_top1] += 1
                student_top1_counts[student_top1] += 1
                token_count += 1
                if teacher_top1 == student_top1:
                    top1_matches += 1
                teacher_set = set(teacher_token)
                student_set = set(student_token)
                overlap = len(teacher_set & student_set) / max(1, len(teacher_set))
                topk_overlap_sum += overlap
                if teacher_top1 != student_top1 and len(hard_tokens) < hard_token_limit:
                    hard_tokens.append(
                        {
                            "prompt_id": prompt_id,
                            "token_index": int(token_index),
                            "teacher_top1": teacher_top1,
                            "student_top1": student_top1,
                            "topk_overlap": float(overlap),
                        }
                    )

        if token_count == 0:
            continue
        layer_records[layer] = {
            "token_count": int(token_count),
            "top1_agreement": float(top1_matches / token_count),
            "mean_topk_overlap": float(topk_overlap_sum / token_count),
            "expert_bias_delta": _bias_deltas(
                teacher_top1_counts=teacher_top1_counts,
                student_top1_counts=student_top1_counts,
                token_count=token_count,
            ),
            "hard_tokens": hard_tokens,
        }

    return {
        "schema_version": 1,
        "record_type": "air_route_trace_disagreement_summary",
        "matched_prompt_count": len(prompt_ids),
        "missing_student_prompt_ids": missing_student_prompt_ids,
        "missing_teacher_prompt_ids": missing_teacher_prompt_ids,
        "missing_student_layers": missing_student_layers,
        "missing_teacher_layers": missing_teacher_layers,
        "layers": layer_records,
    }


def _parse_layers(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed: set[str] = set()
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                parsed.add(str(int(stripped)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare teacher and student GLM-4.5-Air token-level route traces."
    )
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--student-jsonl", required=True)
    parser.add_argument("--layer", action="append", help="Layer index or comma-separated layer indices to compare.")
    parser.add_argument("--hard-token-limit", type=int, default=16)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.hard_token_limit < 0:
        parser.error("--hard-token-limit must be zero or greater")

    summary = summarize_route_trace_disagreement(
        teacher_rows=_read_jsonl(Path(args.teacher_jsonl)),
        student_rows=_read_jsonl(Path(args.student_jsonl)),
        layers=_parse_layers(args.layer),
        hard_token_limit=args.hard_token_limit,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
