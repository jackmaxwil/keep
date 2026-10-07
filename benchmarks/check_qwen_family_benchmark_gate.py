from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

QWEN_PLACEHOLDER_BENCHMARK_BASELINES = {
    "",
    "qwen_source_or_qwen_control_runtime",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _row_memory_clean(row: dict[str, Any]) -> bool:
    return row.get("pageouts_delta") == 0 and row.get("swapouts_delta") == 0


def _latency_ms(row: dict[str, Any]) -> float | None:
    value = row.get("latency_ms", row.get("elapsed_ms", row.get("prefill_ms")))
    if not _finite_number(value) or float(value) <= 0.0:
        return None
    return float(value)


def _candidate_invariants_pass(row: dict[str, Any]) -> bool:
    effective_bpw = row.get("effective_bpw")
    return (
        _finite_number(effective_bpw)
        and float(effective_bpw) > 0.0
        and row.get("dtype_parity") is True
        and row.get("dense_routed_experts") is False
        and row.get("unbound_vq_experts") is False
        and row.get("non_expert_dtype_status") == "verified"
    )


def _probe_missing_requirements(
    probe: dict[str, Any] | None,
    *,
    record_type: str,
) -> list[str]:
    if probe is None or probe.get("record_type") != record_type:
        return []
    missing = probe.get("missing_requirements")
    if not isinstance(missing, list | tuple):
        return []
    return [item for item in missing if isinstance(item, str)]


def _append_missing_once(missing_requirements: list[str], requirement: str) -> None:
    if requirement not in missing_requirements:
        missing_requirements.append(requirement)


def _missing_role_name(scenario: str, role: str) -> str:
    return f"qwen_{scenario}_{role}_benchmark_rows"


def _same_machine_reference(benchmark_gate: dict[str, Any]) -> tuple[str | None, bool]:
    baseline = benchmark_gate.get("comparison_baseline")
    if not isinstance(baseline, str):
        return None, False
    baseline = baseline.strip()
    if baseline in QWEN_PLACEHOLDER_BENCHMARK_BASELINES:
        return baseline, False
    normalized = baseline.replace("-", "_").lower()
    return baseline, bool(benchmark_gate.get("same_machine_reference")) or (
        "same_machine" in normalized
    )


def _positive_finite_float(value: Any) -> float | None:
    if _finite_number(value) and float(value) > 0.0:
        return float(value)
    return None


def check_qwen_family_benchmark_gate(
    *,
    family_policy: dict[str, Any],
    benchmark_jsonl_paths: Iterable[str | Path] = (),
    benchmark_row_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    required_scenarios = tuple(benchmark_gate.get("required_scenarios") or ())
    minimum_repetitions = int(benchmark_gate.get("minimum_repetitions_per_scenario") or 0)
    comparison_baseline, same_machine_reference = _same_machine_reference(benchmark_gate)
    maximum_candidate_to_reference_ratio = _positive_finite_float(
        benchmark_gate.get("maximum_candidate_to_reference_ratio")
    )
    rows: list[dict[str, Any]] = []
    input_paths = [str(Path(path)) for path in benchmark_jsonl_paths]
    for path in benchmark_jsonl_paths:
        rows.extend(_read_jsonl(path))

    clean_latencies: dict[str, dict[str, list[float]]] = {
        scenario: {"candidate": [], "control": []} for scenario in required_scenarios
    }
    row_errors: list[dict[str, Any]] = []
    candidate_invariant_errors: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        scenario = row.get("scenario")
        role = row.get("role")
        latency = _latency_ms(row)
        candidate_invariants_pass = role != "candidate" or _candidate_invariants_pass(row)
        if not _row_memory_clean(row):
            row_errors.append({"row_index": row_index, "error": "row_memory_not_clean"})
        if latency is None:
            row_errors.append({"row_index": row_index, "error": "row_latency_not_finite"})
        if role == "candidate" and not candidate_invariants_pass:
            candidate_invariant_errors.append(
                {
                    "row_index": row_index,
                    "scenario": scenario,
                    "error": "candidate_artifact_invariants_not_verified",
                }
            )
        if (
            scenario in clean_latencies
            and role in ("candidate", "control")
            and latency is not None
            and _row_memory_clean(row)
            and candidate_invariants_pass
        ):
            clean_latencies[str(scenario)][str(role)].append(latency)

    scenario_counts = {
        scenario: {
            role: len(clean_latencies[scenario][role])
            for role in ("candidate", "control")
        }
        for scenario in required_scenarios
    }
    missing_requirements: list[str] = []
    for scenario in required_scenarios:
        for role in ("candidate", "control"):
            if scenario_counts[scenario][role] < minimum_repetitions:
                missing_requirements.append(_missing_role_name(scenario, role))
    if row_errors:
        missing_requirements.append("qwen_benchmark_rows_clean_and_finite")
    if candidate_invariant_errors:
        missing_requirements.append("qwen_candidate_benchmark_artifact_invariants")
    if not same_machine_reference:
        _append_missing_once(
            missing_requirements,
            "qwen_benchmark_same_machine_reference_ratio",
        )
    if maximum_candidate_to_reference_ratio is None:
        _append_missing_once(
            missing_requirements,
            "qwen_benchmark_candidate_reference_ratio",
        )
    benchmark_row_probe_missing_requirements = _probe_missing_requirements(
        benchmark_row_probe,
        record_type="qwen_family_benchmark_row_probe",
    )
    for requirement in benchmark_row_probe_missing_requirements:
        _append_missing_once(missing_requirements, requirement)

    scenario_latency_ratios: dict[str, float | None] = {}
    for scenario in required_scenarios:
        candidate = clean_latencies[scenario]["candidate"]
        control = clean_latencies[scenario]["control"]
        if candidate and control:
            scenario_latency_ratios[scenario] = median(candidate) / median(control)
        else:
            scenario_latency_ratios[scenario] = None
    ratio_errors: list[dict[str, Any]] = []
    if maximum_candidate_to_reference_ratio is not None:
        for scenario, ratio in scenario_latency_ratios.items():
            if ratio is not None and ratio > maximum_candidate_to_reference_ratio:
                ratio_errors.append(
                    {
                        "scenario": scenario,
                        "candidate_to_reference_ratio": ratio,
                        "maximum_candidate_to_reference_ratio": (
                            maximum_candidate_to_reference_ratio
                        ),
                    }
                )
        if ratio_errors:
            _append_missing_once(
                missing_requirements,
                "qwen_benchmark_candidate_reference_ratio",
            )

    family_benchmark_gate_pass = not missing_requirements
    return {
        "record_type": "qwen_family_benchmark_gate_check",
        "model_id": family_policy.get("model_id"),
        "input_paths": input_paths,
        "required_scenarios": list(required_scenarios),
        "minimum_repetitions_per_scenario": minimum_repetitions,
        "comparison_baseline": comparison_baseline,
        "same_machine_reference": same_machine_reference,
        "maximum_candidate_to_reference_ratio": maximum_candidate_to_reference_ratio,
        "scenario_counts": scenario_counts,
        "scenario_latency_ratios": scenario_latency_ratios,
        "ratio_error_count": len(ratio_errors),
        "ratio_errors": ratio_errors,
        "row_error_count": len(row_errors),
        "row_errors": row_errors[:20],
        "benchmark_row_probe_missing_requirements": (
            benchmark_row_probe_missing_requirements
        ),
        "candidate_invariant_error_count": len(candidate_invariant_errors),
        "candidate_invariant_errors": candidate_invariant_errors[:20],
        "family_benchmark_gate_pass": family_benchmark_gate_pass,
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Qwen family benchmark gate evidence."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--benchmark-jsonl", action="append", default=[])
    parser.add_argument("--benchmark-row-probe-json")
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = check_qwen_family_benchmark_gate(
        family_policy=_load_json(args.family_policy_json),
        benchmark_jsonl_paths=args.benchmark_jsonl,
        benchmark_row_probe=(
            _load_json(args.benchmark_row_probe_json)
            if args.benchmark_row_probe_json is not None
            else None
        ),
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
