from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QWEN_PLACEHOLDER_BENCHMARK_BASELINES = {
    "",
    "qwen_source_or_qwen_control_runtime",
}


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


def _as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list | tuple):
        return set()
    result: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _expected_layers(count: int) -> set[int]:
    return set(range(count))


def _all_forward_results_finite(runtime_forward_probe: dict[str, Any]) -> bool:
    results = runtime_forward_probe.get("forward_results") or []
    return bool(results) and all(bool(item.get("finite")) for item in results)


def _tokenized_logits_probe_pass(tokenized_logits_probe: dict[str, Any] | None) -> bool:
    if tokenized_logits_probe is None:
        return False
    probe = tokenized_logits_probe.get("tokenized_logit_probe") or {}
    logit_probe = probe.get("logit_probe") or {}
    return (
        bool(tokenized_logits_probe.get("runtime_binding_pass"))
        and bool(tokenized_logits_probe.get("tokenized_logit_probe_pass"))
        and not tokenized_logits_probe.get("tokenized_logit_probe_errors")
        and not tokenized_logits_probe.get("missing_model_parameters")
        and tokenized_logits_probe.get("unbound_vq_experts") is False
        and probe.get("record_type") == "qwen_tokenized_source_logits_probe"
        and bool(probe.get("encoded_token_ids"))
        and probe.get("selected_token_id") is not None
        and bool(probe.get("finite"))
        and bool(logit_probe.get("finite"))
    )


def _family_policy_defined(family_policy: dict[str, Any] | None) -> bool:
    return (
        family_policy is not None
        and family_policy.get("record_type") == "qwen_family_gate_policy"
        and family_policy.get("policy_status") == "qwen_family_policy_defined"
        and family_policy.get("air_thresholds_reused") is False
        and family_policy.get("eval_gate_defined") is True
        and family_policy.get("benchmark_gate_defined") is True
        and isinstance(family_policy.get("eval_gate"), dict)
        and isinstance(family_policy.get("benchmark_gate"), dict)
    )


def _positive_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result > 0.0 and result < float("inf"):
        return result
    return None


def _policy_required_eval_rows(family_policy: dict[str, Any]) -> int:
    eval_gate = family_policy.get("eval_gate") or {}
    minimum_total = _non_bool_int(eval_gate.get("minimum_total_clean_rows"))
    if minimum_total > 0:
        return minimum_total
    required_splits = eval_gate.get("required_splits")
    split_count = len(required_splits) if isinstance(required_splits, list | tuple) else 0
    return _non_bool_int(eval_gate.get("minimum_clean_rows_per_split")) * split_count


def _benchmark_gate_has_same_machine_reference(family_policy: dict[str, Any]) -> bool:
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    baseline = benchmark_gate.get("comparison_baseline")
    if not isinstance(baseline, str):
        return False
    baseline = baseline.strip()
    if baseline in QWEN_PLACEHOLDER_BENCHMARK_BASELINES:
        return False
    normalized = baseline.replace("-", "_").lower()
    return bool(benchmark_gate.get("same_machine_reference")) or (
        "same_machine" in normalized
    )


def _family_policy_hardened(family_policy: dict[str, Any] | None) -> bool:
    if not _family_policy_defined(family_policy):
        return False
    benchmark_gate = family_policy.get("benchmark_gate") or {}
    return (
        _policy_required_eval_rows(family_policy) >= 64
        and _benchmark_gate_has_same_machine_reference(family_policy)
        and _positive_finite_number(
            benchmark_gate.get("maximum_candidate_to_reference_ratio")
        )
        is not None
    )


def _family_eval_gate_pass(family_eval_gate_probe: dict[str, Any] | None) -> bool:
    return (
        family_eval_gate_probe is not None
        and family_eval_gate_probe.get("record_type") == "qwen_family_eval_gate_check"
        and family_eval_gate_probe.get("family_eval_gate_pass") is True
    )


def _eval_prompt_pack_ready(
    eval_prompt_pack_probe: dict[str, Any] | None,
    family_policy: dict[str, Any] | None,
) -> bool:
    if (
        eval_prompt_pack_probe is None
        or eval_prompt_pack_probe.get("record_type") != "qwen_family_eval_prompt_probe"
        or eval_prompt_pack_probe.get("prompt_pack_ready") is not True
        or eval_prompt_pack_probe.get("missing_requirements")
    ):
        return False
    if family_policy is None:
        return True
    eval_gate = family_policy.get("eval_gate") or {}
    required_splits = tuple(eval_gate.get("required_splits") or ())
    min_rows = int(eval_gate.get("minimum_clean_rows_per_split") or 0)
    min_total_rows = _non_bool_int(eval_gate.get("minimum_total_clean_rows"))
    split_counts = eval_prompt_pack_probe.get("split_counts") or {}
    encoded_prompt_count = _non_bool_int(eval_prompt_pack_probe.get("encoded_prompt_count"))
    split_rows_ready = all(
        int(split_counts.get(split) or 0) >= min_rows for split in required_splits
    )
    total_rows_ready = min_total_rows <= 0 or encoded_prompt_count >= min_total_rows
    return split_rows_ready and total_rows_ready


def _family_benchmark_gate_pass(
    family_benchmark_gate_probe: dict[str, Any] | None,
) -> bool:
    return (
        family_benchmark_gate_probe is not None
        and family_benchmark_gate_probe.get("record_type")
        == "qwen_family_benchmark_gate_check"
        and family_benchmark_gate_probe.get("family_benchmark_gate_pass") is True
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def _string_lists_by_split(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for split, items in value.items():
        if not isinstance(split, str):
            continue
        clean_items = _string_list(items)
        if clean_items:
            result[split] = clean_items
    return result


def _non_bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _family_gate_blocker_details(
    family_eval_gate_probe: dict[str, Any] | None,
    family_benchmark_gate_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    eval_probe = family_eval_gate_probe or {}
    benchmark_probe = family_benchmark_gate_probe or {}
    benchmark_errors = benchmark_probe.get("candidate_invariant_errors")

    return {
        "eval_gate_missing_requirements": _string_list(
            eval_probe.get("missing_requirements")
        ),
        "benchmark_gate_missing_requirements": _string_list(
            benchmark_probe.get("missing_requirements")
        ),
        "eval_missing_prompt_ids_by_split": _string_lists_by_split(
            eval_probe.get("missing_prompt_ids_by_split")
        ),
        "benchmark_candidate_invariant_error_count": _non_bool_int(
            benchmark_probe.get("candidate_invariant_error_count")
        ),
        "benchmark_candidate_invariant_errors": (
            benchmark_errors if isinstance(benchmark_errors, list) else []
        ),
    }


def evaluate_qwen_family_gate(
    *,
    source_audit: dict[str, Any],
    source_payload_audit: dict[str, Any],
    artifact_audit: dict[str, Any],
    bind_probe: dict[str, Any],
    non_expert_bind_probe: dict[str, Any],
    runtime_forward_probe: dict[str, Any],
    runtime_global_logit_probe: dict[str, Any],
    upstream_transformer_probe: dict[str, Any] | None = None,
    tokenizer_probe: dict[str, Any] | None = None,
    tokenized_logits_probe: dict[str, Any] | None = None,
    family_policy: dict[str, Any] | None = None,
    eval_prompt_pack_probe: dict[str, Any] | None = None,
    family_eval_gate_probe: dict[str, Any] | None = None,
    family_benchmark_gate_probe: dict[str, Any] | None = None,
    expected_layer_count: int = 40,
    upstream_transformer_integration: bool = False,
    tokenizer_to_logits_forward: bool = False,
    family_eval_gate: bool = False,
    family_benchmark_gate: bool = False,
) -> dict[str, Any]:
    if expected_layer_count <= 0:
        raise ValueError("expected_layer_count must be positive")
    expected_layers = _expected_layers(expected_layer_count)
    bound_layers = _as_int_set(bind_probe.get("bound_layers"))
    bind_expected_layers = _as_int_set(bind_probe.get("expected_layers"))
    non_expert_requested_layers = _as_int_set(
        non_expert_bind_probe.get("requested_layers")
    )
    runtime_bound_layers = _as_int_set(runtime_forward_probe.get("bound_layers"))
    runtime_forward_layers = _as_int_set(runtime_forward_probe.get("forward_layers"))

    source_mapping_pass = bool(source_audit.get("source_checks_pass"))
    source_payload_pass = (
        source_payload_audit.get("payload_status") == "qwen_moe_source_payloads_ready"
        and not source_payload_audit.get("materialization_blocked")
        and not source_payload_audit.get("missing_shards")
        and int(source_payload_audit.get("planned_groups") or 0)
        >= expected_layer_count * 2
    )
    artifact_audit_pass = (
        bool(artifact_audit.get("audit_pass"))
        and artifact_audit.get("dense_routed_experts") is False
        and artifact_audit.get("unbound_vq_experts") is False
        and int(artifact_audit.get("source_projection_groups") or 0)
        >= expected_layer_count * 2
        and int(artifact_audit.get("target_projection_groups") or 0)
        >= expected_layer_count * 3
    )
    full_moe_binding_pass = (
        bool(bind_probe.get("binding_pass"))
        and not bind_probe.get("missing_layers")
        and bind_probe.get("unbound_vq_experts") is False
        and bind_probe.get("dense_routed_experts") is False
        and bound_layers == expected_layers
        and bind_expected_layers == expected_layers
    )
    full_non_expert_binding_pass = (
        bool(non_expert_bind_probe.get("binding_pass"))
        and not non_expert_bind_probe.get("missing_model_parameters")
        and non_expert_requested_layers == expected_layers
    )
    runtime_switch_forward_pass = (
        bool(runtime_forward_probe.get("runtime_binding_pass"))
        and bool(runtime_forward_probe.get("moe_binding_pass"))
        and bool(runtime_forward_probe.get("non_expert_binding_pass"))
        and bool(runtime_forward_probe.get("forward_pass"))
        and not runtime_forward_probe.get("forward_errors")
        and not runtime_forward_probe.get("missing_layers")
        and not runtime_forward_probe.get("missing_model_parameters")
        and runtime_forward_probe.get("unbound_vq_experts") is False
        and runtime_bound_layers == expected_layers
        and {0, expected_layer_count - 1}.issubset(runtime_forward_layers)
        and _all_forward_results_finite(runtime_forward_probe)
    )
    logit_probe = runtime_global_logit_probe.get("logit_probe") or {}
    bounded_global_logit_pass = (
        bool(runtime_global_logit_probe.get("runtime_binding_pass"))
        and bool(runtime_global_logit_probe.get("logit_probe_pass"))
        and not runtime_global_logit_probe.get("logit_probe_errors")
        and not runtime_global_logit_probe.get("missing_model_parameters")
        and runtime_global_logit_probe.get("unbound_vq_experts") is False
        and bool(logit_probe.get("finite"))
    )
    upstream_transformer_integration_pass = bool(
        upstream_transformer_integration
        or (
            upstream_transformer_probe is not None
            and upstream_transformer_probe.get("record_type")
            == "qwen_upstream_transformer_binding_probe"
            and upstream_transformer_probe.get("integration_pass") is True
        )
    )
    tokenizer_payload_ready_pass = bool(
        tokenizer_probe is not None
        and tokenizer_probe.get("record_type") == "qwen_tokenizer_readiness_probe"
        and tokenizer_probe.get("tokenizer_payload_ready") is True
    )
    tokenizer_to_logits_forward_pass = bool(
        tokenizer_to_logits_forward
        or _tokenized_logits_probe_pass(tokenized_logits_probe)
    )
    family_policy_defined_pass = _family_policy_defined(family_policy)
    qwen_family_policy_hardened_pass = _family_policy_hardened(family_policy)
    qwen_eval_prompt_pack_ready_pass = _eval_prompt_pack_ready(
        eval_prompt_pack_probe,
        family_policy if family_policy_defined_pass else None,
    )
    family_eval_gate_pass = bool(
        family_eval_gate or _family_eval_gate_pass(family_eval_gate_probe)
    )
    family_benchmark_gate_pass = bool(
        family_benchmark_gate
        or _family_benchmark_gate_pass(family_benchmark_gate_probe)
    )
    evidence_checks = {
        "source_mapping_pass": source_mapping_pass,
        "source_payload_pass": source_payload_pass,
        "artifact_audit_pass": artifact_audit_pass,
        "full_moe_binding_pass": full_moe_binding_pass,
        "full_non_expert_binding_pass": full_non_expert_binding_pass,
        "runtime_switch_forward_pass": runtime_switch_forward_pass,
        "bounded_global_logit_pass": bounded_global_logit_pass,
        "upstream_transformer_integration_pass": upstream_transformer_integration_pass,
        "tokenizer_payload_ready_pass": tokenizer_payload_ready_pass,
        "tokenizer_to_logits_forward_pass": tokenizer_to_logits_forward_pass,
        "family_policy_defined_pass": family_policy_defined_pass,
        "qwen_family_policy_hardened_pass": qwen_family_policy_hardened_pass,
        "qwen_eval_prompt_pack_ready_pass": qwen_eval_prompt_pack_ready_pass,
        "family_eval_gate_pass": family_eval_gate_pass,
        "family_benchmark_gate_pass": family_benchmark_gate_pass,
    }
    binding_keys = (
        "source_mapping_pass",
        "source_payload_pass",
        "artifact_audit_pass",
        "full_moe_binding_pass",
        "full_non_expert_binding_pass",
        "runtime_switch_forward_pass",
        "bounded_global_logit_pass",
    )
    binding_ready = all(evidence_checks[key] for key in binding_keys)
    public_gate_ready = all(evidence_checks.values())

    missing_requirements: list[str] = []
    if not source_mapping_pass:
        missing_requirements.append("source_mapping_audit")
    if not source_payload_pass:
        missing_requirements.append("source_payload_audit")
    if not artifact_audit_pass:
        missing_requirements.append("artifact_audit")
    if not full_moe_binding_pass:
        missing_requirements.append(f"full_{expected_layer_count}_layer_moe_binding")
    if not full_non_expert_binding_pass:
        missing_requirements.append(
            f"full_{expected_layer_count}_layer_non_expert_binding"
        )
    if not runtime_switch_forward_pass:
        missing_requirements.append("first_and_last_layer_switch_forward")
    if not bounded_global_logit_pass:
        missing_requirements.append("bounded_global_logit_probe")
    if not upstream_transformer_integration_pass:
        missing_requirements.append("upstream_transformer_integration")
    if not tokenizer_payload_ready_pass:
        missing_requirements.append("qwen_tokenizer_payload")
    if not tokenizer_to_logits_forward_pass:
        missing_requirements.append("tokenizer_to_logits_forward")
    if family_policy_defined_pass and not qwen_family_policy_hardened_pass:
        missing_requirements.append("qwen_hardened_family_policy")
    if family_policy_defined_pass and not qwen_eval_prompt_pack_ready_pass:
        missing_requirements.append("qwen_eval_prompt_pack")
    if not family_eval_gate_pass:
        missing_requirements.append("qwen_family_eval_gate")
    if not family_benchmark_gate_pass:
        missing_requirements.append("qwen_family_benchmark_gate")

    eval_gate_threshold = (
        family_policy["eval_gate"]
        if family_policy_defined_pass
        else "not_defined_for_qwen3_5_moe_yet"
    )
    benchmark_gate_threshold = (
        family_policy["benchmark_gate"]
        if family_policy_defined_pass
        else "not_defined_for_qwen3_5_moe_yet"
    )

    return {
        "record_type": "qwen_family_gate_check",
        "model_id": source_audit.get("model_id") or artifact_audit.get("model_id"),
        "model_type": source_audit.get("model_type", "qwen3_5_moe"),
        "expected_layer_count": expected_layer_count,
        "template_source": "docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md",
        "binding_readiness_status": (
            "qwen_binding_ready" if binding_ready else "qwen_binding_incomplete"
        ),
        "family_gate_status": (
            "qwen_family_gate_ready" if public_gate_ready else "qwen_family_gate_not_ready"
        ),
        "binding_ready": binding_ready,
        "family_gate_pass": public_gate_ready,
        "evidence_checks": evidence_checks,
        "missing_requirements": missing_requirements,
        "family_gate_blocker_details": _family_gate_blocker_details(
            family_eval_gate_probe,
            family_benchmark_gate_probe,
        ),
        "qwen_family_thresholds": {
            "switch_forward_layers": [0, expected_layer_count - 1],
            "bounded_global_logit_probe": "finite_logits_required",
            "tokenizer_payload": "local_tokenizer_must_encode_non_empty_probe_prompt",
            "tokenizer_to_logits_forward": (
                "tokenizer_prompt_must_select_token_and_compute_finite_source_logits"
            ),
            "eval_prompt_pack": "required_splits_must_have_tokenized_prompt_rows",
            "eval_gate": eval_gate_threshold,
            "benchmark_gate": benchmark_gate_threshold,
            "family_policy_hardened": (
                "minimum_total_clean_rows>=64_and_same_machine_reference_ratio"
            ),
            "air_thresholds_reused": False,
        },
        "notes": [
            "Existing Qwen VQ/source binding evidence is necessary but not sufficient for a public family gate.",
            "Qwen eval and benchmark gates must be family-specific; GLM-4.5-Air thresholds are not reused.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Qwen family readiness against the new-model-family gate."
    )
    parser.add_argument("--source-audit-json", required=True)
    parser.add_argument("--source-payload-audit-json", required=True)
    parser.add_argument("--artifact-audit-json", required=True)
    parser.add_argument("--bind-probe-json", required=True)
    parser.add_argument("--non-expert-bind-probe-json", required=True)
    parser.add_argument("--runtime-forward-json", required=True)
    parser.add_argument("--runtime-global-logit-json", required=True)
    parser.add_argument("--upstream-transformer-json")
    parser.add_argument("--tokenizer-json")
    parser.add_argument("--tokenized-logits-json")
    parser.add_argument("--family-policy-json")
    parser.add_argument("--eval-prompt-pack-json")
    parser.add_argument("--family-eval-gate-json")
    parser.add_argument("--family-benchmark-gate-json")
    parser.add_argument("--expected-layer-count", type=int, default=40)
    parser.add_argument("--upstream-transformer-integration", action="store_true")
    parser.add_argument("--tokenizer-to-logits-forward", action="store_true")
    parser.add_argument("--family-eval-gate", action="store_true")
    parser.add_argument("--family-benchmark-gate", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = evaluate_qwen_family_gate(
        source_audit=_load_json(args.source_audit_json),
        source_payload_audit=_load_json(args.source_payload_audit_json),
        artifact_audit=_load_json(args.artifact_audit_json),
        bind_probe=_load_json(args.bind_probe_json),
        non_expert_bind_probe=_load_json(args.non_expert_bind_probe_json),
        runtime_forward_probe=_load_json(args.runtime_forward_json),
        runtime_global_logit_probe=_load_json(args.runtime_global_logit_json),
        upstream_transformer_probe=(
            _load_json(args.upstream_transformer_json)
            if args.upstream_transformer_json is not None
            else None
        ),
        tokenizer_probe=(
            _load_json(args.tokenizer_json)
            if args.tokenizer_json is not None
            else None
        ),
        tokenized_logits_probe=(
            _load_json(args.tokenized_logits_json)
            if args.tokenized_logits_json is not None
            else None
        ),
        family_policy=(
            _load_json(args.family_policy_json)
            if args.family_policy_json is not None
            else None
        ),
        eval_prompt_pack_probe=(
            _load_json(args.eval_prompt_pack_json)
            if args.eval_prompt_pack_json is not None
            else None
        ),
        family_eval_gate_probe=(
            _load_json(args.family_eval_gate_json)
            if args.family_eval_gate_json is not None
            else None
        ),
        family_benchmark_gate_probe=(
            _load_json(args.family_benchmark_gate_json)
            if args.family_benchmark_gate_json is not None
            else None
        ),
        expected_layer_count=args.expected_layer_count,
        upstream_transformer_integration=args.upstream_transformer_integration,
        tokenizer_to_logits_forward=args.tokenizer_to_logits_forward,
        family_eval_gate=args.family_eval_gate,
        family_benchmark_gate=args.family_benchmark_gate,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["binding_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
