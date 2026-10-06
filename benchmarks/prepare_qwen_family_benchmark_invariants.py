from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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


def _required_scenarios(family_policy: dict[str, Any]) -> tuple[str, ...]:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    return tuple(
        scenario
        for scenario in benchmark_gate.get("required_scenarios") or ()
        if isinstance(scenario, str)
    )


def _minimum_repetitions(family_policy: dict[str, Any]) -> int:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    return int(benchmark_gate.get("minimum_repetitions_per_scenario") or 1)


def _non_expert_dtype_status(non_expert_bind_probe: dict[str, Any]) -> str:
    loaded_dtypes = non_expert_bind_probe.get("loaded_dtype_names")
    if not isinstance(loaded_dtypes, dict) or not loaded_dtypes:
        return "missing"
    if not all(isinstance(dtype, str) and dtype for dtype in loaded_dtypes.values()):
        return "missing"
    return "verified"


def prepare_qwen_family_benchmark_invariants(
    *,
    family_policy: dict[str, Any],
    artifact_audit: dict[str, Any],
    non_expert_bind_probe: dict[str, Any],
    output_jsonl: str | Path,
    run_index: int = 0,
    repetitions: int | None = None,
) -> dict[str, Any]:
    required_scenarios = _required_scenarios(family_policy)
    repetition_count = _minimum_repetitions(family_policy) if repetitions is None else repetitions
    if repetition_count <= 0:
        raise ValueError("repetitions must be positive")
    effective_bpw = artifact_audit.get("effective_routed_bpw")
    non_expert_dtype_status = _non_expert_dtype_status(non_expert_bind_probe)

    missing_requirements: list[str] = []
    if artifact_audit.get("audit_pass") is not True:
        missing_requirements.append("qwen_candidate_artifact_audit")
    if not _finite_positive(effective_bpw):
        missing_requirements.append("qwen_candidate_effective_bpw")
    if artifact_audit.get("dense_routed_experts") is not False:
        missing_requirements.append("qwen_candidate_dense_routed_experts_absent")
    if artifact_audit.get("unbound_vq_experts") is not False:
        missing_requirements.append("qwen_candidate_unbound_vq_experts_absent")
    if (
        non_expert_bind_probe.get("binding_pass") is not True
        or non_expert_bind_probe.get("missing_model_parameters")
        or not _finite_positive(non_expert_bind_probe.get("loaded_model_parameter_count"))
        or non_expert_dtype_status != "verified"
    ):
        missing_requirements.append("qwen_candidate_non_expert_dtype_verification")

    rows: list[dict[str, Any]] = []
    if not missing_requirements:
        rows = [
            {
                "record_type": "qwen_candidate_benchmark_invariants",
                "model_id": family_policy.get("model_id"),
                "revision": family_policy.get("revision"),
                "scenario": scenario,
                "run_index": run_index + repetition_idx,
                "effective_bpw": float(effective_bpw),
                "dtype_parity": True,
                "dense_routed_experts": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_status": "verified",
            }
            for scenario in required_scenarios
            for repetition_idx in range(repetition_count)
        ]

    _write_jsonl(output_jsonl, rows)
    return {
        "record_type": "qwen_candidate_benchmark_invariant_probe",
        "model_id": family_policy.get("model_id"),
        "revision": family_policy.get("revision"),
        "output_jsonl": str(Path(output_jsonl)),
        "required_scenarios": list(required_scenarios),
        "minimum_repetitions_per_scenario": repetition_count,
        "candidate_invariant_row_count": len(rows),
        "artifact_audit_pass": artifact_audit.get("audit_pass") is True,
        "effective_bpw": float(effective_bpw) if _finite_positive(effective_bpw) else None,
        "dense_routed_experts": artifact_audit.get("dense_routed_experts"),
        "unbound_vq_experts": artifact_audit.get("unbound_vq_experts"),
        "non_expert_binding_pass": non_expert_bind_probe.get("binding_pass") is True,
        "non_expert_loaded_model_parameter_count": non_expert_bind_probe.get(
            "loaded_model_parameter_count"
        ),
        "non_expert_dtype_status": non_expert_dtype_status,
        "missing_requirements": missing_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen benchmark candidate invariant rows from audit evidence."
    )
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--artifact-audit-json", required=True)
    parser.add_argument("--non-expert-bind-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    output_jsonl = Path(args.output_dir) / "qwen_candidate_benchmark_invariants.jsonl"
    payload = prepare_qwen_family_benchmark_invariants(
        family_policy=_load_json(args.family_policy_json),
        artifact_audit=_load_json(args.artifact_audit_json),
        non_expert_bind_probe=_load_json(args.non_expert_bind_json),
        output_jsonl=output_jsonl,
        run_index=args.run_index,
        repetitions=args.repetitions,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
