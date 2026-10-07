from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_jsonl_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and float(value) > 0.0
    )


def _memory_clean(row: dict[str, Any]) -> bool:
    return row.get("pageouts_delta") == 0 and row.get("swapouts_delta") == 0


def _candidate_invariants_pass(row: dict[str, Any]) -> bool:
    return (
        _finite_positive(row.get("effective_bpw"))
        and row.get("dtype_parity") is True
        and row.get("dense_routed_experts") is False
        and row.get("unbound_vq_experts") is False
        and row.get("non_expert_dtype_status") == "verified"
    )


def _run_key(row: dict[str, Any]) -> tuple[str, int] | None:
    scenario = row.get("scenario")
    run_index = row.get("run_index", 0)
    if not isinstance(scenario, str) or isinstance(run_index, bool):
        return None
    try:
        return scenario, int(run_index)
    except (TypeError, ValueError):
        return None


def _invariants_by_key(
    rows: Iterable[dict[str, Any]],
    required_scenarios: set[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _run_key(row)
        if key is None or key[0] not in required_scenarios or key in result:
            continue
        if _candidate_invariants_pass(row):
            result[key] = row
    return result


def _required_counts(
    rows: Iterable[dict[str, Any]],
    required_scenarios: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    counts = {
        scenario: {"candidate": 0, "control": 0}
        for scenario in required_scenarios
    }
    for row in rows:
        scenario = row.get("scenario")
        role = row.get("role")
        if scenario in counts and role in counts[str(scenario)]:
            counts[str(scenario)][str(role)] += 1
    return counts


def _nonzero_missing(counts: dict[str, dict[str, int]]) -> bool:
    return any(
        role_count > 0
        for scenario_counts in counts.values()
        for role_count in scenario_counts.values()
    )


def prepare_qwen_family_benchmark_rows(
    *,
    family_policy: dict[str, Any],
    latency_rows: Iterable[dict[str, Any]] = (),
    candidate_invariant_rows: Iterable[dict[str, Any]] = (),
    output_jsonl: str | Path,
) -> dict[str, Any]:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    required_scenarios = tuple(benchmark_gate.get("required_scenarios") or ())
    required_scenario_set = set(required_scenarios)
    invariants_by_key = _invariants_by_key(
        candidate_invariant_rows,
        required_scenario_set,
    )
    output_rows: list[dict[str, Any]] = []
    missing_latency_counts = {
        scenario: {"candidate": 0, "control": 0}
        for scenario in required_scenarios
    }
    missing_invariant_counts = {
        scenario: {"candidate": 0}
        for scenario in required_scenarios
    }

    for row in latency_rows:
        scenario = row.get("scenario")
        role = row.get("role")
        run_index = row.get("run_index", 0)
        if scenario not in required_scenario_set or role not in {"candidate", "control"}:
            continue
        if not _finite_positive(row.get("latency_ms")) or not _memory_clean(row):
            missing_latency_counts[str(scenario)][str(role)] += 1
            continue
        benchmark_row = {
            "record_type": "qwen_family_benchmark_row",
            "model_id": family_policy.get("model_id"),
            "revision": family_policy.get("revision"),
            "scenario": scenario,
            "role": role,
            "run_index": int(run_index),
            "latency_ms": row["latency_ms"],
            "pageouts_delta": row["pageouts_delta"],
            "swapouts_delta": row["swapouts_delta"],
        }
        if role == "candidate":
            key = _run_key(row)
            invariants = invariants_by_key.get(key) if key is not None else None
            if invariants is None:
                missing_invariant_counts[str(scenario)]["candidate"] += 1
                continue
            benchmark_row.update(
                {
                    "effective_bpw": invariants["effective_bpw"],
                    "dtype_parity": invariants["dtype_parity"],
                    "dense_routed_experts": invariants["dense_routed_experts"],
                    "unbound_vq_experts": invariants["unbound_vq_experts"],
                    "non_expert_dtype_status": invariants[
                        "non_expert_dtype_status"
                    ],
                }
            )
        output_rows.append(benchmark_row)

    _write_jsonl(output_jsonl, output_rows)
    scenario_counts = _required_counts(output_rows, required_scenarios)
    missing_requirements: list[str] = []
    if not output_rows:
        missing_requirements.append("qwen_benchmark_latency_rows")
    if _nonzero_missing(missing_latency_counts):
        missing_requirements.append("qwen_benchmark_rows_clean_and_finite")
    if _nonzero_missing(missing_invariant_counts):
        missing_requirements.append("qwen_candidate_benchmark_artifact_invariants")

    return {
        "record_type": "qwen_family_benchmark_row_probe",
        "model_id": family_policy.get("model_id"),
        "revision": family_policy.get("revision"),
        "output_jsonl": str(Path(output_jsonl)),
        "required_scenarios": list(required_scenarios),
        "benchmark_row_count": len(output_rows),
        "scenario_counts": scenario_counts,
        "missing_latency_counts": missing_latency_counts,
        "missing_candidate_invariant_counts": missing_invariant_counts,
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen family benchmark rows from latency plus invariants."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--latency-jsonl", action="append", default=[])
    parser.add_argument("--candidate-invariant-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_family_benchmark_rows.jsonl"
    payload = prepare_qwen_family_benchmark_rows(
        family_policy=_load_json(args.family_policy_json),
        latency_rows=_read_jsonl_paths(args.latency_jsonl),
        candidate_invariant_rows=_read_jsonl_paths(args.candidate_invariant_jsonl),
        output_jsonl=output_jsonl,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
