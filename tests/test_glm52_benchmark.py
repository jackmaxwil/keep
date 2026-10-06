from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path("src/mlx_vq/quality/glm52_benchmark.py")
    spec = importlib.util.spec_from_file_location("glm52_benchmark_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _policy() -> dict[str, object]:
    return {
        "policy_contract_sha256": "a" * 64,
        "benchmark_gate": {
            "candidate_and_control_same_machine": True,
            "comparison_baseline": "same_machine_pinned_fp4_source_streaming_control",
            "maximum_candidate_to_reference_ratio": 1.15,
            "minimum_repetitions_per_scenario": 3,
            "pageouts_and_swapouts_must_be_zero": True,
            "required_scenarios": ["prefill_1k", "decode_128"],
        },
    }


def _machine() -> dict[str, object]:
    return {
        "io_platform_uuid": "B45D5B04-39DF-5F86-B8D7-B9D8FAAD30D4",
        "hw.model": "Mac16,7",
        "hw.memsize": 549755813888,
        "os.build": "25A123",
    }


def _session() -> dict[str, object]:
    return {
        "session_uuid": "0a30f6e2-00d9-49c5-a4d1-d2faf4bc4b01",
        "monotonic_start_ns": 1_000_000,
        "monotonic_end_ns": 10_000_000,
        "wall_start_ns": 1_700_000_000_000_000_000,
        "wall_end_ns": 1_700_000_000_009_000_000,
    }


def _rows(role: str, latency_ms: float) -> list[dict[str, object]]:
    return [
        {
            "record_kind": "repetition",
            "scenario": scenario,
            "role": role,
            "run_index": run_index,
            "sequence_number": scenario_index * 3 + run_index,
            "latency_ms": latency_ms,
            "input_token_count": 1024,
            "output_token_count": 0 if scenario == "prefill_1k" else 128,
            "monotonic_start_ns": 2_000_000 + (scenario_index * 3 + run_index) * 1_000,
            "monotonic_end_ns": 2_000_500 + (scenario_index * 3 + run_index) * 1_000,
            "wall_start_ns": 1_700_000_000_001_000_000 + (scenario_index * 3 + run_index) * 1_000,
            "wall_end_ns": 1_700_000_000_001_000_500 + (scenario_index * 3 + run_index) * 1_000,
            "vm_stat_before": {"pageouts": 100 + run_index, "swapouts": 10 + run_index},
            "vm_stat_after": {"pageouts": 100 + run_index, "swapouts": 10 + run_index},
            "pageouts_delta": 0,
            "swapouts_delta": 0,
            "valid": True,
        }
        for scenario_index, scenario in enumerate(("prefill_1k", "decode_128"))
        for run_index in range(3)
    ]


def _evidence(module, role: str, latency_ms: float, *, session: dict[str, object] | None = None) -> dict[str, object]:
    contract = module.load_frozen_glm52_benchmark_contract(_policy())
    return module.build_measurement_evidence(
        role=role,
        contract=contract,
        machine_identity=_machine(),
        benchmark_input_identity={"token_count": 1024, "token_ids_sha256": "c" * 64},
        identity={
            "identity": role,
            "sha256": (
                "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
                if role == "candidate"
                else "b" * 64
            ),
        },
        benchmark_session=session or _session(),
        warmups=[
            {
                "record_kind": "warmup",
                "scenario": scenario,
                "role": role,
                "warmup_sequence_number": index,
                "excluded_from_gate": True,
                "latency_ms": 1.0,
                "input_token_count": 1024,
                "output_token_count": 0 if scenario == "prefill_1k" else 128,
                "monotonic_start_ns": 1_100_000 + index * 1_000,
                "monotonic_end_ns": 1_100_500 + index * 1_000,
                "wall_start_ns": 1_700_000_000_000_100_000 + index * 1_000,
                "wall_end_ns": 1_700_000_000_000_100_500 + index * 1_000,
                "vm_stat_before": {"pageouts": 10, "swapouts": 1},
                "vm_stat_after": {"pageouts": 10, "swapouts": 1},
                "pageouts_delta": 0,
                "swapouts_delta": 0,
                "valid": True,
            }
            for index, scenario in enumerate(("prefill_1k", "decode_128"))
        ],
        repetitions=_rows(role, latency_ms),
    )


def _paired_evaluation(module, *, candidate_ms: float = 100.0, control_ms: float = 100.0) -> dict[str, object]:
    candidate = _evidence(module, "candidate", candidate_ms)
    control = _evidence(module, "control", control_ms)
    session_manifest = module.build_benchmark_session_manifest(
        benchmark_session=_session(),
        machine_identity=_machine(),
        candidate=candidate,
        control=control,
    )
    return module.evaluate_glm52_same_machine_benchmark(
        policy=_policy(),
        candidate=candidate,
        control=control,
        session_manifest=session_manifest,
    )


@pytest.mark.parametrize(
    ("candidate_ms", "expected_pass"),
    [(115.0, True), (115.001, False)],
)
def test_evaluator_honors_the_frozen_ratio_boundary(
    candidate_ms: float,
    expected_pass: bool,
) -> None:
    module = _load_module()

    result = module.evaluate_glm52_same_machine_benchmark(
        policy=_policy(),
        candidate=_evidence(module, "candidate", candidate_ms),
        control=_evidence(module, "control", 100.0),
        session_manifest=module.build_benchmark_session_manifest(
            benchmark_session=_session(),
            machine_identity=_machine(),
            candidate=_evidence(module, "candidate", candidate_ms),
            control=_evidence(module, "control", 100.0),
        ),
    )

    assert result["frozen_gate_pass"] is expected_pass
    assert result["scenario_latency_ratios"]["prefill_1k"] == pytest.approx(
        candidate_ms / 100.0
    )
    assert result["maximum_candidate_to_reference_ratio"] == 1.15


def test_evaluator_invalidates_dirty_repetitions_without_replacement() -> None:
    module = _load_module()
    candidate = _evidence(module, "candidate", 100.0)
    candidate["repetitions"][0]["pageouts_delta"] = 1  # type: ignore[index]
    candidate["repetitions"][0]["vm_stat_after"]["pageouts"] = 101  # type: ignore[index]

    result = module.evaluate_glm52_same_machine_benchmark(
        policy=_policy(),
        candidate=candidate,
        control=_evidence(module, "control", 100.0),
        session_manifest=module.build_benchmark_session_manifest(
            benchmark_session=_session(),
            machine_identity=_machine(),
            candidate=candidate,
            control=_evidence(module, "control", 100.0),
        ),
    )

    assert result["frozen_gate_pass"] is False
    assert result["scenario_counts"]["prefill_1k"]["candidate"] == 2
    assert "candidate_prefill_1k_requires_3_clean_repetitions" in result["missing_requirements"]
    assert result["invalid_repetition_count"] == 1


def test_evaluator_rejects_machine_identity_mismatch() -> None:
    module = _load_module()
    control = _evidence(module, "control", 100.0)
    control["machine_identity"] = {**_machine(), "os.build": "25B456"}

    with pytest.raises(ValueError, match="machine identity"):
        module.evaluate_glm52_same_machine_benchmark(
            policy=_policy(),
            candidate=_evidence(module, "candidate", 100.0),
            control=control,
            session_manifest=module.build_benchmark_session_manifest(
                benchmark_session=_session(),
                machine_identity=_machine(),
                candidate=_evidence(module, "candidate", 100.0),
                control=control,
            ),
        )


def test_policy_contract_rejects_tuning_overrides() -> None:
    module = _load_module()
    altered = _policy()
    altered["benchmark_gate"] = {
        **altered["benchmark_gate"],  # type: ignore[arg-type]
        "maximum_candidate_to_reference_ratio": 1.16,
    }

    with pytest.raises(ValueError, match="maximum_candidate_to_reference_ratio"):
        module.load_frozen_glm52_benchmark_contract(altered)


def test_cli_exposes_no_threshold_or_repetition_override() -> None:
    path = Path("benchmarks/bench_glm52_same_machine_fp4.py")
    spec = importlib.util.spec_from_file_location("glm52_benchmark_cli_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "compare",
                "--family-policy-json",
                "policy.json",
                "--candidate-json",
                "candidate.json",
                "--control-json",
                "control.json",
                "--output-json",
                "gate.json",
                "--maximum-candidate-to-reference-ratio",
                "1.16",
            ]
        )


def test_release_verifier_rejects_forged_stored_aggregates(tmp_path: Path) -> None:
    module = _load_module()
    evidence = _paired_evaluation(module)
    evidence["scenario_latency_ratios"] = {"prefill_1k": 0.01, "decode_128": 0.01}
    path = tmp_path / "forged.json"
    module.write_sealed_evidence(path, evidence)

    with pytest.raises(ValueError, match="stored aggregate"):
        module.load_glm52_same_machine_benchmark_evidence(path, policy=_policy())


def test_release_verifier_rejects_missing_raw_records(tmp_path: Path) -> None:
    module = _load_module()
    evidence = _paired_evaluation(module)
    evidence.pop("raw_measurements")
    path = tmp_path / "missing-raw.json"
    module.write_sealed_evidence(path, evidence)

    with pytest.raises(ValueError, match="raw measurement"):
        module.load_glm52_same_machine_benchmark_evidence(path, policy=_policy())


def test_evaluator_rejects_different_physical_host_uuid() -> None:
    module = _load_module()
    candidate = _evidence(module, "candidate", 100.0)
    control = _evidence(module, "control", 100.0)
    control["machine_identity"] = {**_machine(), "io_platform_uuid": "other-host"}

    with pytest.raises(ValueError, match="machine identity"):
        module.evaluate_glm52_same_machine_benchmark(
            policy=_policy(),
            candidate=candidate,
            control=control,
            session_manifest=module.build_benchmark_session_manifest(
                benchmark_session=_session(),
                machine_identity=_machine(),
                candidate=candidate,
                control=control,
            ),
        )


def test_evaluator_rejects_different_session_nonce() -> None:
    module = _load_module()
    candidate = _evidence(module, "candidate", 100.0)
    control = _evidence(module, "control", 100.0)
    control["benchmark_session"] = {**_session(), "session_uuid": "other-session"}

    with pytest.raises(ValueError, match="session"):
        module.evaluate_glm52_same_machine_benchmark(
            policy=_policy(),
            candidate=candidate,
            control=control,
            session_manifest=module.build_benchmark_session_manifest(
                benchmark_session=_session(),
                machine_identity=_machine(),
                candidate=candidate,
                control=control,
            ),
        )


def test_evaluator_rejects_non_contiguous_repetition_sequence() -> None:
    module = _load_module()
    candidate = _evidence(module, "candidate", 100.0)
    candidate["repetitions"][1]["sequence_number"] = 4  # type: ignore[index]

    with pytest.raises(ValueError, match="contiguous"):
        module.evaluate_glm52_same_machine_benchmark(
            policy=_policy(),
            candidate=candidate,
            control=_evidence(module, "control", 100.0),
            session_manifest=module.build_benchmark_session_manifest(
                benchmark_session=_session(),
                machine_identity=_machine(),
                candidate=candidate,
                control=_evidence(module, "control", 100.0),
            ),
        )


def test_evaluator_rejects_warmup_counted_as_repetition() -> None:
    module = _load_module()
    candidate = _evidence(module, "candidate", 100.0)
    candidate["repetitions"][0]["record_kind"] = "warmup"  # type: ignore[index]

    with pytest.raises(ValueError, match="warmup"):
        module.evaluate_glm52_same_machine_benchmark(
            policy=_policy(),
            candidate=candidate,
            control=_evidence(module, "control", 100.0),
            session_manifest=module.build_benchmark_session_manifest(
                benchmark_session=_session(),
                machine_identity=_machine(),
                candidate=candidate,
                control=_evidence(module, "control", 100.0),
            ),
        )


def test_release_verifier_accepts_identical_cache_honest_raw_path(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "honest.json"
    module.write_sealed_evidence(path, _paired_evaluation(module))

    loaded = module.load_glm52_same_machine_benchmark_evidence(path, policy=_policy())

    assert loaded["frozen_gate_pass"] is True
