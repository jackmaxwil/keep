from __future__ import annotations

from benchmarks.check_qwen_family_gate import evaluate_qwen_family_gate


def _layers(count: int) -> list[int]:
    return list(range(count))


def _hardened_family_policy() -> dict[str, object]:
    return {
        "record_type": "qwen_family_gate_policy",
        "policy_status": "qwen_family_policy_defined",
        "air_thresholds_reused": False,
        "eval_gate_defined": True,
        "benchmark_gate_defined": True,
        "eval_gate": {
            "minimum_clean_rows_per_split": 22,
            "minimum_total_clean_rows": 64,
            "required_splits": ["report", "selection", "holdout"],
        },
        "benchmark_gate": {
            "minimum_repetitions_per_scenario": 2,
            "required_scenarios": ["prefill_1k", "decode_128"],
            "comparison_baseline": "same_machine_qwen_source_switch_projection_control",
            "same_machine_reference": True,
            "maximum_candidate_to_reference_ratio": 3.0,
        },
    }


def _ready_eval_prompt_pack(prompt_count: int = 22) -> dict[str, object]:
    return {
        "record_type": "qwen_family_eval_prompt_probe",
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "prompt_pack_ready": True,
        "missing_requirements": [],
        "split_counts": {
            "report": prompt_count,
            "selection": prompt_count,
            "holdout": prompt_count,
        },
        "encoded_prompt_count": prompt_count * 3,
    }


def test_qwen_family_gate_keeps_binding_ready_separate_from_public_gate() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
            "language_sparse_layers": 40,
            "conversion_blockers": [
                "qwen3_5_moe_runtime_adapter",
                "family_specific_eval_and_benchmark_gates",
            ],
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
            "effective_routed_bpw": 1.03125,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
            "loaded_model_parameter_count": 610,
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        expected_layer_count=40,
    )

    assert payload["binding_readiness_status"] == "qwen_binding_ready"
    assert payload["family_gate_status"] == "qwen_family_gate_not_ready"
    assert payload["family_gate_pass"] is False
    assert payload["evidence_checks"]["full_moe_binding_pass"] is True
    assert payload["evidence_checks"]["runtime_switch_forward_pass"] is True
    assert payload["evidence_checks"]["bounded_global_logit_pass"] is True
    assert payload["evidence_checks"]["upstream_transformer_integration_pass"] is False
    assert payload["evidence_checks"]["tokenizer_to_logits_forward_pass"] is False
    assert payload["evidence_checks"]["family_eval_gate_pass"] is False
    assert payload["evidence_checks"]["family_benchmark_gate_pass"] is False
    assert payload["missing_requirements"] == [
        "upstream_transformer_integration",
        "qwen_tokenizer_payload",
        "tokenizer_to_logits_forward",
        "qwen_family_eval_gate",
        "qwen_family_benchmark_gate",
    ]


def test_qwen_family_gate_consumes_upstream_transformer_probe() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
            "upstream_model_module": "mlx_lm.models.qwen3_5_moe",
        },
        expected_layer_count=40,
    )

    assert payload["binding_readiness_status"] == "qwen_binding_ready"
    assert payload["family_gate_status"] == "qwen_family_gate_not_ready"
    assert payload["evidence_checks"]["upstream_transformer_integration_pass"] is True
    assert payload["missing_requirements"] == [
        "qwen_tokenizer_payload",
        "tokenizer_to_logits_forward",
        "qwen_family_eval_gate",
        "qwen_family_benchmark_gate",
    ]


def test_qwen_family_gate_tracks_tokenizer_readiness_before_tokenizer_logits() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
        },
        tokenizer_probe={
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_payload_ready": False,
        },
        expected_layer_count=40,
    )

    assert payload["evidence_checks"]["tokenizer_payload_ready_pass"] is False
    assert payload["evidence_checks"]["tokenizer_to_logits_forward_pass"] is False
    assert payload["missing_requirements"] == [
        "qwen_tokenizer_payload",
        "tokenizer_to_logits_forward",
        "qwen_family_eval_gate",
        "qwen_family_benchmark_gate",
    ]


def test_qwen_family_gate_consumes_tokenized_logit_probe() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
        },
        tokenizer_probe={
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_payload_ready": True,
        },
        tokenized_logits_probe={
            "runtime_binding_pass": True,
            "tokenized_logit_probe_pass": True,
            "tokenized_logit_probe": {
                "record_type": "qwen_tokenized_source_logits_probe",
                "finite": True,
                "encoded_token_ids": [760, 6511, 314, 9338, 369],
                "selected_token_id": 369,
                "logit_probe": {"finite": True},
            },
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        family_policy=_hardened_family_policy(),
        family_eval_gate_probe={
            "record_type": "qwen_family_eval_gate_check",
            "family_eval_gate_pass": False,
            "missing_requirements": ["qwen_report_eval_jsonl"],
        },
        family_benchmark_gate_probe={
            "record_type": "qwen_family_benchmark_gate_check",
            "family_benchmark_gate_pass": False,
            "missing_requirements": ["qwen_prefill_1k_candidate_benchmark_rows"],
        },
        expected_layer_count=40,
    )

    assert payload["evidence_checks"]["tokenizer_payload_ready_pass"] is True
    assert payload["evidence_checks"]["tokenizer_to_logits_forward_pass"] is True
    assert payload["missing_requirements"] == [
        "qwen_eval_prompt_pack",
        "qwen_family_eval_gate",
        "qwen_family_benchmark_gate",
    ]
    assert (
        payload["qwen_family_thresholds"]["eval_gate"]["minimum_clean_rows_per_split"]
        == 22
    )
    assert (
        payload["qwen_family_thresholds"]["benchmark_gate"][
            "minimum_repetitions_per_scenario"
        ]
        == 2
    )


def test_qwen_family_gate_consumes_eval_prompt_pack() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
        },
        tokenizer_probe={
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_payload_ready": True,
        },
        tokenized_logits_probe={
            "runtime_binding_pass": True,
            "tokenized_logit_probe_pass": True,
            "tokenized_logit_probe": {
                "record_type": "qwen_tokenized_source_logits_probe",
                "finite": True,
                "encoded_token_ids": [760, 6511, 314, 9338, 369],
                "selected_token_id": 369,
                "logit_probe": {"finite": True},
            },
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        family_policy=_hardened_family_policy(),
        eval_prompt_pack_probe=_ready_eval_prompt_pack(),
        family_eval_gate_probe={
            "record_type": "qwen_family_eval_gate_check",
            "family_eval_gate_pass": False,
            "missing_requirements": ["qwen_report_eval_jsonl"],
        },
        family_benchmark_gate_probe={
            "record_type": "qwen_family_benchmark_gate_check",
            "family_benchmark_gate_pass": False,
            "missing_requirements": ["qwen_prefill_1k_candidate_benchmark_rows"],
        },
        expected_layer_count=40,
    )

    assert payload["evidence_checks"]["qwen_eval_prompt_pack_ready_pass"] is True
    assert "qwen_eval_prompt_pack" not in payload["missing_requirements"]
    assert payload["missing_requirements"] == [
        "qwen_family_eval_gate",
        "qwen_family_benchmark_gate",
    ]


def test_qwen_family_gate_surfaces_child_gate_blocker_details() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
        },
        tokenizer_probe={
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_payload_ready": True,
        },
        tokenized_logits_probe={
            "runtime_binding_pass": True,
            "tokenized_logit_probe_pass": True,
            "tokenized_logit_probe": {
                "record_type": "qwen_tokenized_source_logits_probe",
                "finite": True,
                "encoded_token_ids": [760, 6511, 314, 9338, 369],
                "selected_token_id": 369,
                "logit_probe": {"finite": True},
            },
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        family_policy=_hardened_family_policy(),
        eval_prompt_pack_probe=_ready_eval_prompt_pack(),
        family_eval_gate_probe={
            "record_type": "qwen_family_eval_gate_check",
            "family_eval_gate_pass": False,
            "missing_requirements": [
                "qwen_report_eval_jsonl",
                "qwen_report_eval_prompt_rows",
            ],
            "missing_prompt_ids_by_split": {
                "report": ["qwen_report_001"],
                "selection": ["qwen_selection_001"],
            },
        },
        family_benchmark_gate_probe={
            "record_type": "qwen_family_benchmark_gate_check",
            "family_benchmark_gate_pass": False,
            "missing_requirements": [
                "qwen_prefill_1k_candidate_benchmark_rows",
                "qwen_candidate_benchmark_artifact_invariants",
            ],
            "candidate_invariant_error_count": 2,
            "candidate_invariant_errors": [
                {
                    "row_index": 0,
                    "error": "candidate_artifact_invariants_not_verified",
                }
            ],
        },
        expected_layer_count=40,
    )

    assert payload["missing_requirements"] == [
        "qwen_family_eval_gate",
        "qwen_family_benchmark_gate",
    ]
    assert payload["family_gate_blocker_details"] == {
        "eval_gate_missing_requirements": [
            "qwen_report_eval_jsonl",
            "qwen_report_eval_prompt_rows",
        ],
        "benchmark_gate_missing_requirements": [
            "qwen_prefill_1k_candidate_benchmark_rows",
            "qwen_candidate_benchmark_artifact_invariants",
        ],
        "eval_missing_prompt_ids_by_split": {
            "report": ["qwen_report_001"],
            "selection": ["qwen_selection_001"],
        },
        "benchmark_candidate_invariant_error_count": 2,
        "benchmark_candidate_invariant_errors": [
            {
                "row_index": 0,
                "error": "candidate_artifact_invariants_not_verified",
            }
        ],
    }


def test_qwen_family_gate_consumes_eval_and_benchmark_gate_evidence() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
        },
        tokenizer_probe={
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_payload_ready": True,
        },
        tokenized_logits_probe={
            "runtime_binding_pass": True,
            "tokenized_logit_probe_pass": True,
            "tokenized_logit_probe": {
                "record_type": "qwen_tokenized_source_logits_probe",
                "finite": True,
                "encoded_token_ids": [760, 6511, 314, 9338, 369],
                "selected_token_id": 369,
                "logit_probe": {"finite": True},
            },
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        family_policy=_hardened_family_policy(),
        eval_prompt_pack_probe=_ready_eval_prompt_pack(),
        family_eval_gate_probe={
            "record_type": "qwen_family_eval_gate_check",
            "family_eval_gate_pass": True,
        },
        family_benchmark_gate_probe={
            "record_type": "qwen_family_benchmark_gate_check",
            "family_benchmark_gate_pass": True,
        },
        expected_layer_count=40,
    )

    assert payload["family_gate_pass"] is True
    assert payload["evidence_checks"]["qwen_eval_prompt_pack_ready_pass"] is True
    assert payload["evidence_checks"]["family_eval_gate_pass"] is True
    assert payload["evidence_checks"]["family_benchmark_gate_pass"] is True
    assert payload["missing_requirements"] == []


def test_qwen_family_gate_rejects_weak_policy_even_when_child_gates_pass() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": _layers(40),
            "expected_layers": _layers(40),
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": _layers(40),
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0, 39],
            "forward_results": [
                {"layer": 0, "finite": True},
                {"layer": 39, "finite": True},
            ],
            "bound_layers": _layers(40),
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        upstream_transformer_probe={
            "record_type": "qwen_upstream_transformer_binding_probe",
            "integration_pass": True,
        },
        tokenizer_probe={
            "record_type": "qwen_tokenizer_readiness_probe",
            "tokenizer_payload_ready": True,
        },
        tokenized_logits_probe={
            "runtime_binding_pass": True,
            "tokenized_logit_probe_pass": True,
            "tokenized_logit_probe": {
                "record_type": "qwen_tokenized_source_logits_probe",
                "finite": True,
                "encoded_token_ids": [760, 6511, 314, 9338, 369],
                "selected_token_id": 369,
                "logit_probe": {"finite": True},
            },
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        family_policy={
            "record_type": "qwen_family_gate_policy",
            "policy_status": "qwen_family_policy_defined",
            "air_thresholds_reused": False,
            "eval_gate_defined": True,
            "benchmark_gate_defined": True,
            "eval_gate": {
                "minimum_clean_rows_per_split": 16,
                "required_splits": ["report", "selection", "holdout"],
            },
            "benchmark_gate": {
                "minimum_repetitions_per_scenario": 2,
                "required_scenarios": ["prefill_1k", "decode_128"],
                "comparison_baseline": "qwen_source_or_qwen_control_runtime",
            },
        },
        eval_prompt_pack_probe=_ready_eval_prompt_pack(prompt_count=16),
        family_eval_gate_probe={
            "record_type": "qwen_family_eval_gate_check",
            "family_eval_gate_pass": True,
        },
        family_benchmark_gate_probe={
            "record_type": "qwen_family_benchmark_gate_check",
            "family_benchmark_gate_pass": True,
        },
        expected_layer_count=40,
    )

    assert payload["family_gate_pass"] is False
    assert payload["evidence_checks"]["qwen_family_policy_hardened_pass"] is False
    assert payload["missing_requirements"] == ["qwen_hardened_family_policy"]


def test_qwen_family_gate_rejects_partial_layer_binding() -> None:
    payload = evaluate_qwen_family_gate(
        source_audit={
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_type": "qwen3_5_moe",
            "source_checks_pass": True,
            "language_sparse_layers": 40,
        },
        source_payload_audit={
            "payload_status": "qwen_moe_source_payloads_ready",
            "materialization_blocked": False,
            "planned_groups": 80,
            "missing_shards": [],
        },
        artifact_audit={
            "audit_pass": True,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
            "source_projection_groups": 80,
            "target_projection_groups": 120,
        },
        bind_probe={
            "binding_pass": True,
            "bound_layers": [0, 1],
            "expected_layers": [0, 1],
            "missing_layers": [],
            "unbound_vq_experts": False,
            "dense_routed_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "requested_layers": [0, 1],
            "missing_model_parameters": [],
        },
        runtime_forward_probe={
            "runtime_binding_pass": True,
            "moe_binding_pass": True,
            "non_expert_binding_pass": True,
            "forward_pass": True,
            "forward_layers": [0],
            "forward_results": [{"layer": 0, "finite": True}],
            "bound_layers": [0, 1],
            "missing_layers": [],
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        runtime_global_logit_probe={
            "runtime_binding_pass": True,
            "logit_probe_pass": True,
            "logit_probe": {"finite": True, "token_id": 0},
            "missing_model_parameters": [],
            "unbound_vq_experts": False,
        },
        expected_layer_count=40,
    )

    assert payload["binding_readiness_status"] == "qwen_binding_incomplete"
    assert payload["evidence_checks"]["full_moe_binding_pass"] is False
    assert payload["evidence_checks"]["full_non_expert_binding_pass"] is False
    assert "full_40_layer_moe_binding" in payload["missing_requirements"]
    assert "full_40_layer_non_expert_binding" in payload["missing_requirements"]
