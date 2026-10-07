from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import importlib.util
import sys
from pathlib import Path

import pytest

from benchmarks import check_glm52_family_gate as gate
from mlx_vq.quality.glm52_family import (
    GLM52_FAMILY_GATE_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V1_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V2_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS,
    GLM52_FAMILY_GATE_V3_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V4_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V5_SCHEMA_VERSION,
    GLM52_FAMILY_GATE_V6_SCHEMA_VERSION,
    _expect_exact_field,
    canonical_sha256,
    sha256_file,
)


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PROMPT_CONTRACT = "prompt-contract"
CONFIG_SHA256 = "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
INDEX_SHA256 = "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
PROFILE_SHA256 = "ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d"
PROFILE_CONTRACT_SHA256 = (
    "28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8"
)
NON_VQ_MANIFEST_SHA256 = (
    "5113750fdaf009b2174a2772dd2e3bccf490cfff354f082e5e0ad8af8771f8c8"
)
NON_VQ_PACKAGE_INDEX_SHA256 = (
    "ff7def155c88006cda458344b5cb8e8a20303049d20405cd4e9b28e7885df097"
)
SOURCE_BLOB_INVENTORY_SHA256 = (
    "ace08e87dcbce3a22249e54196a27c0045992f8d5b8ca07f342899fa7a53fc8d"
)
SOURCE_INVENTORY_SHA256 = (
    "bf42d5bc79e8eb5cef55304601571f1f7edfdda7f4086957ffb44c4a8334adde"
)
ROUTED_MANIFEST_SHA256 = (
    "ba1d3135ef8901f1a69ead28b5f9d330ef40d015ac31fdb4d2dcfe678c41f0a4"
)
ROUTED_GROUP_SET_SHA256 = (
    "bb65cf9a0d4a78310eb46c25e07b6492e6f3a5b18b303a7d68a8f0c73819d3fe"
)
NON_VQ_EVIDENCE_FILE_SHA256 = (
    "ccbedd87f72f032bd86fd9d5a89fb75641595f30cf2a80fd7c84636d594e80d2"
)
ACCEPTED_PARAMETER_COUNT = 494_194_805_304
ACCEPTED_TENSOR_PAYLOAD_BYTES = 98_433_923_808
ACCEPTED_TENSOR_PAYLOAD_BPW = (
    ACCEPTED_TENSOR_PAYLOAD_BYTES * 8 / ACCEPTED_PARAMETER_COUNT
)
GROUP_KEYS = tuple(
    f"{layer}:{projection}"
    for layer in range(3, 78)
    for projection in ("gate_proj", "up_proj", "down_proj")
)
NON_VQ_AUDIT_CHECKS = {
    "accounting": True,
    "index": True,
    "lineage": True,
    "payload_hashes": True,
    "physical_extents": True,
    "shard_hashes": True,
    "strict_tree": True,
    "tensor_inventory": True,
}


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"ready": 1}, {"ready": True}),
        ({"offset": False}, {"offset": 0}),
        ([3.0], [3]),
    ],
    ids=("int-for-bool", "bool-for-int", "float-for-int"),
)
def test_exact_field_rejects_nested_scalar_type_confusion(
    actual: object,
    expected: object,
) -> None:
    with pytest.raises(ValueError, match="nested contract"):
        _expect_exact_field(
            {"nested": actual},
            "nested",
            expected,
            label="nested contract",
        )


def _prompt_rows() -> tuple[dict[str, object], ...]:
    rows = []
    for index in range(66):
        split = ("report", "selection", "holdout")[index // 22]
        split_index = index % 22
        domain = (
            "route"
            if split_index < 8
            else "math"
            if split_index < 15
            else "instruction"
        )
        rows.append(
            {
                "prompt_id": f"{split}-{index:02d}",
                "split": split,
                "domain": domain,
                "token_count": 2,
                "token_ids_sha256": f"token-{index:02d}",
                "tuning_eligible": split == "selection",
            }
        )
    return tuple(rows)


def _policy_body() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "eval_gate": {
            "minimum_clean_rows_per_split": 22,
            "minimum_total_clean_rows": 66,
            "mean_kld_max": 0.30,
            "p999_kld_max": 3.0,
            "top1_min": 0.85,
            "domain_top1_min": 0.80,
            "mean_ppl_ratio_max": 1.05,
        },
        "benchmark_gate": {
            "required_scenarios": ["prefill_1k", "decode_128"],
            "minimum_repetitions_per_scenario": 3,
            "comparison_baseline": (
                "same_machine_pinned_fp4_source_streaming_control"
            ),
            "candidate_and_control_same_machine": True,
            "maximum_candidate_to_reference_ratio": 1.15,
            "pageouts_and_swapouts_must_be_zero": True,
        },
        "artifact_target": {
            "tensor_payload_bytes": 98_433_923_808,
            "whole_main_bpw": 1.5934433,
        },
    }


def _policy() -> dict[str, object]:
    payload = _policy_body()
    payload["policy_contract_sha256"] = canonical_sha256(payload)
    return payload


POLICY_CONTRACT = canonical_sha256(_policy_body())


def _prompt_pack() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "family_policy_contract_sha256": POLICY_CONTRACT,
        "prompt_pack_contract_sha256": PROMPT_CONTRACT,
    }


def _teacher_metadata(tmp_path: Path) -> dict[str, object]:
    rows = [
        {
            **row,
            "schema_version": 1,
            "record_type": "glm52_teacher_source_metadata",
            "teacher_model_id": MODEL_ID,
            "teacher_revision": REVISION,
            "teacher_source_kind": (
                "deterministically_dequantized_pinned_modelopt_nvfp4"
            ),
            "source_weight_encoding": "modelopt_nvfp4",
            "source_decoder": "modelopt_nvfp4_v1",
            "family_policy_contract_sha256": POLICY_CONTRACT,
            "prompt_pack_contract_sha256": PROMPT_CONTRACT,
            "teacher_cache_payload_present": False,
            "teacher_cache_ready": False,
            "full_logits_available": False,
            "intended_logit_scope": "full_vocabulary",
            "intended_kld_mode": "exact_full_logits",
            "expected_vocab_size": 154880,
            "activation_quantization_emulated": False,
            "exact_w4a4_runtime_parity_claimed": False,
            "bf16_teacher_claimed": False,
            "holdout_tuning_forbidden": True,
        }
        for row in _prompt_rows()
    ]
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_family_eval_teacher_metadata_probe",
        "metadata_status": "glm52_teacher_source_metadata_ready",
        "metadata_ready": True,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "teacher_source_kind": (
            "deterministically_dequantized_pinned_modelopt_nvfp4"
        ),
        "family_policy_contract_sha256": POLICY_CONTRACT,
        "prompt_pack_contract_sha256": PROMPT_CONTRACT,
        "source_index_revalidated": True,
        "source_payload_headers_revalidated": True,
        "source_payload_group_count": 225,
        "source_payload_shard_count": 61,
        "header_only_source_validation": True,
        "tensor_payloads_read": False,
        "full_model_constructed": False,
        "candidate_artifact_used": False,
        "metadata_jsonl": str(metadata_path),
        "metadata_jsonl_sha256": sha256_file(metadata_path),
        "metadata_row_count": 66,
        "metadata_counts_by_split": {
            "report": 22,
            "selection": 22,
            "holdout": 22,
        },
        "metadata_counts_by_domain": {
            "route": 24,
            "math": 21,
            "instruction": 21,
        },
        "teacher_cache_payload_present": False,
        "teacher_cache_ready": False,
        "teacher_logits_generated": False,
        "intended_logit_scope": "full_vocabulary",
        "intended_kld_mode": "exact_full_logits",
        "expected_vocab_size": 154880,
        "activation_quantization_emulated": False,
        "exact_w4a4_runtime_parity_claimed": False,
        "bf16_teacher_claimed": False,
        "missing_requirements": ["dequantized_source_teacher_cache_payload"],
    }
    payload["metadata_contract_sha256"] = canonical_sha256(payload)
    return payload


def _full_bind(*, present: int = 6) -> dict[str, object]:
    all_keys = list(GROUP_KEYS)
    missing = 225 - present
    return {
        "record_type": "glm52_full_bind_preflight",
        "profile": "glm52-reap-504b-v2",
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "source_index_sha256": INDEX_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "expected_routed_group_count": 225,
        "expected_routed_group_keys": all_keys,
        "present_routed_group_count": present,
        "present_routed_group_keys": all_keys[:present],
        "missing_routed_group_count": missing,
        "missing_routed_group_keys": all_keys[present:],
        "preflight_status": (
            "glm52_full_bind_preflight_ready"
            if missing == 0
            else "glm52_full_bind_preflight_blocked"
        ),
        "preflight_pass": missing == 0,
        "production_binding_proven": False,
        "production_generation_proven": False,
        "dense_routed_experts": False,
    }


def _indexshare_runtime() -> dict[str, object]:
    return {
        "record_type": "glm52_indexshare_runtime_probe",
        "probe_status": "glm52_indexshare_runtime_probe_ready",
        "probe_pass": True,
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "production_identity_contract": True,
        "static_contract_pass": True,
        "synthetic_runtime_contract": True,
        "full_model_bind_proven": False,
        "generation_proven": False,
        "whole_model_runtime_proven": False,
    }


def _synthetic_generation() -> dict[str, object]:
    return {
        "record_type": "glm52_synthetic_generation_probe",
        "probe_status": "glm52_synthetic_generation_probe_ready",
        "probe_pass": True,
        "synthetic_runtime_contract": True,
        "dense_routed_weight_present": False,
        "production_artifact_used": False,
        "production_generation_proven": False,
        "whole_model_scope": "tiny_fixture",
    }


def _quality_gate() -> dict[str, object]:
    return {
        "record_type": "glm52_family_eval_gate",
        "gate_status": "glm52_family_eval_gate_passed",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "family_policy_contract_sha256": POLICY_CONTRACT,
        "prompt_pack_contract_sha256": PROMPT_CONTRACT,
        "eval_rows_sha256": "a" * 64,
        "teacher_logits_manifest_sha256": "b" * 64,
        "candidate_logits_manifest_sha256": "c" * 64,
        "teacher_cache_ready": True,
        "teacher_full_vocabulary_logits": True,
        "candidate_full_vocabulary_logits": True,
        "compact_topk_kld_used": False,
        "holdout_tuning_used": False,
        "clean_row_count": 66,
        "dirty_row_count": 0,
        "split_counts": {"report": 22, "selection": 22, "holdout": 22},
        "metrics": {
            "mean_kld": 0.20,
            "p999_kld": 2.0,
            "top1_agreement": 0.90,
            "mean_ppl_ratio": 1.02,
        },
        "domain_top1_agreement": {
            "route": 0.88,
            "math": 0.89,
            "instruction": 0.91,
        },
        "family_eval_gate_pass": True,
    }


def _benchmark_gate() -> dict[str, object]:
    return {
        "record_type": "glm52_family_benchmark_gate",
        "gate_status": "glm52_family_benchmark_gate_passed",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "family_policy_contract_sha256": POLICY_CONTRACT,
        "benchmark_rows_sha256": "d" * 64,
        "comparison_baseline": (
            "same_machine_pinned_fp4_source_streaming_control"
        ),
        "candidate_and_control_same_machine": True,
        "machine_fingerprint_sha256": "e" * 64,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
        "scenario_counts": {
            "prefill_1k": {"candidate": 3, "control": 3},
            "decode_128": {"candidate": 3, "control": 3},
        },
        "scenario_latency_ratios": {"prefill_1k": 1.10, "decode_128": 1.12},
        "family_benchmark_gate_pass": True,
    }


def _benchmark_api():
    path = Path("src/mlx_vq/quality/glm52_benchmark.py")
    spec = importlib.util.spec_from_file_location("glm52_benchmark_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sealed_benchmark_evidence(*, dirty_memory: bool = False) -> dict[str, object]:
    """Construct the exact sealed compare output accepted by the benchmark CLI."""

    benchmark = _benchmark_api()
    contract = benchmark.load_frozen_glm52_benchmark_contract(_policy())
    machine = {
        "io_platform_uuid": "B45D5B04-39DF-5F86-B8D7-B9D8FAAD30D4",
        "hw.model": "Mac16,7",
        "hw.memsize": 549_755_813_888,
        "os.build": "25A123",
    }
    session = {
        "session_uuid": "0a30f6e2-00d9-49c5-a4d1-d2faf4bc4b01",
        "monotonic_start_ns": 1_000_000,
        "monotonic_end_ns": 10_000_000,
        "wall_start_ns": 1_700_000_000_000_000_000,
        "wall_end_ns": 1_700_000_000_009_000_000,
    }
    benchmark_input = {"token_count": 1024, "token_ids_sha256": "a" * 64}

    def rows(role: str, latency_ms: float) -> list[dict[str, object]]:
        return [
            {
                "record_kind": "repetition",
                "scenario": scenario,
                "role": role,
                "run_index": run_index,
                "sequence_number": scenario_index * benchmark.FROZEN_REPETITIONS + run_index,
                "latency_ms": latency_ms,
                "input_token_count": 1024,
                "output_token_count": 0 if scenario == "prefill_1k" else 128,
                "monotonic_start_ns": 2_000_000 + (scenario_index * 3 + run_index) * 1_000,
                "monotonic_end_ns": 2_000_500 + (scenario_index * 3 + run_index) * 1_000,
                "wall_start_ns": 1_700_000_000_001_000_000 + (scenario_index * 3 + run_index) * 1_000,
                "wall_end_ns": 1_700_000_000_001_000_500 + (scenario_index * 3 + run_index) * 1_000,
                "vm_stat_before": {"pageouts": 100 + run_index, "swapouts": 10 + run_index},
                "vm_stat_after": {
                    "pageouts": 101 if dirty_memory and role == "candidate" and scenario == "prefill_1k" and run_index == 0 else 100 + run_index,
                    "swapouts": 10 + run_index,
                },
                "pageouts_delta": 1 if dirty_memory and role == "candidate" and scenario == "prefill_1k" and run_index == 0 else 0,
                "swapouts_delta": 0,
                "valid": True,
            }
            for scenario_index, scenario in enumerate(benchmark.FROZEN_SCENARIOS)
            for run_index in range(benchmark.FROZEN_REPETITIONS)
        ]

    def measurement(role: str, latency_ms: float) -> dict[str, object]:
        return benchmark.build_measurement_evidence(
            role=role,
            contract=contract,
            machine_identity=machine,
            benchmark_input_identity=benchmark_input,
            identity={
                "identity": role,
                "sha256": (
                    "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
                    if role == "candidate"
                    else "b" * 64
                ),
            },
            benchmark_session=session,
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
                for index, scenario in enumerate(benchmark.FROZEN_SCENARIOS)
            ],
            repetitions=rows(role, latency_ms),
        )

    candidate = measurement("candidate", 110.0)
    control = measurement("control", 100.0)
    return benchmark.seal_evidence(
        benchmark.evaluate_glm52_same_machine_benchmark(
            policy=_policy(),
            candidate=candidate,
            control=control,
            session_manifest=benchmark.build_benchmark_session_manifest(
                benchmark_session=session,
                machine_identity=machine,
                candidate=candidate,
                control=control,
            ),
        )
    )


def _optimistic_production_summary() -> dict[str, object]:
    return {
        "record_type": "glm52_production_generation_probe",
        "probe_status": "glm52_production_generation_probe_ready",
        "probe_pass": True,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "family_policy_contract_sha256": POLICY_CONTRACT,
        "full_routed_group_count": 225,
        "artifact_audit_pass": True,
        "dense_routed_experts": False,
        "production_artifact_used": True,
        "production_binding_proven": True,
        "production_generation_proven": True,
        "tokenizer_used": True,
        "generated_token_count": 8,
        "logits_finite": True,
        "generation_sha256": "f" * 64,
    }


def _non_vq_inventory() -> dict[str, object]:
    shard_names = [
        f"model-{index:05d}-of-00009.safetensors" for index in range(1, 10)
    ]
    shard_sizes = [14, 140, 139, 155, 150, 150, 155, 155, 136]
    shard_payload_targets = [
        4_285_163_008,
        4_129_697_536,
        4_100_337_408,
        4_132_546_624,
        4_113_802_816,
        4_113_802_816,
        4_132_546_624,
        4_132_546_624,
        3_981_045_152,
    ]
    tensors: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    tensor_index = 0
    source_cursors: dict[str, int] = {}

    for shard_index, (shard_name, shard_size, payload_target) in enumerate(
        zip(shard_names, shard_sizes, shard_payload_targets, strict=True)
    ):
        output_offset = 0
        shard_tensors: list[dict[str, object]] = []
        f32_count = 75 if shard_index == 8 else 0
        bf16_count = shard_size - f32_count
        bf16_payload_target = payload_target - f32_count * 672
        bf16_payloads = [268_435_456, *([2] * (bf16_count - 2))]
        bf16_payloads.append(bf16_payload_target - sum(bf16_payloads))
        payload_specs = [("BF16", value) for value in bf16_payloads]
        payload_specs.extend(("F32", 672) for _ in range(f32_count))
        for dtype, payload_bytes in payload_specs:
            parameter_count = payload_bytes // (4 if dtype == "F32" else 2)
            name = f"model.fixture_tensor_{tensor_index:04d}.weight"
            source_shard = f"model-{tensor_index // 20:05d}.safetensors"
            source_offset = source_cursors.get(source_shard, 0)
            tensor = {
                "name": name,
                "dtype": dtype,
                "shape": [parameter_count],
                "parameter_count": parameter_count,
                "payload_bytes": payload_bytes,
                "source_shard": source_shard,
                "source_offsets": [source_offset, source_offset + payload_bytes],
                "output_shard": shard_name,
                "output_offsets": [output_offset, output_offset + payload_bytes],
                "payload_sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            tensors.append(tensor)
            shard_tensors.append(tensor)
            tensor_index += 1
            output_offset += payload_bytes
            source_cursors[source_shard] = source_offset + payload_bytes

        tensor_plan = [
            {key: value for key, value in tensor.items() if key != "payload_sha256"}
            for tensor in shard_tensors
        ]
        shards.append(
            {
                "filename": shard_name,
                "tensor_count": len(shard_tensors),
                "tensor_inventory_sha256": canonical_sha256(tensor_plan),
                "tensor_payload_bytes": output_offset,
                "file_bytes": output_offset + 1_024,
                "file_sha256": hashlib.sha256(shard_name.encode()).hexdigest(),
            }
        )

    tensor_plan = [
        {key: value for key, value in tensor.items() if key != "payload_sha256"}
        for tensor in tensors
    ]
    plan_sha256 = canonical_sha256(
        {
            "schema_version": 1,
            "copy_mode": "raw_safetensors_byte_ranges_v1",
            "selection_policy": "main_model_non_routed_bf16_f32_v1",
            "profile": "glm52-reap-504b-v2",
            "model_id": MODEL_ID,
            "source_revision": REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "max_shard_payload_bytes": 4_294_967_296,
            "tensors": tensor_plan,
        }
    )
    package_set_sha256 = canonical_sha256(
        {
            "package_index_sha256": NON_VQ_PACKAGE_INDEX_SHA256,
            "shards": [dict(shard) for shard in shards],
        }
    )
    return {
        "tensors": tensors,
        "shards": shards,
        "plan_sha256": plan_sha256,
        "package_set_sha256": package_set_sha256,
    }


def _non_vq_evidence() -> dict[str, object]:
    inventory = _non_vq_inventory()
    tensors = inventory["tensors"]
    shards = inventory["shards"]
    assert isinstance(tensors, list)
    assert isinstance(shards, list)
    return {
        "schema_version": 1,
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "source_authority": "pinned_huggingface_lfs_v1",
        "source_blob_inventory_sha256": SOURCE_BLOB_INVENTORY_SHA256,
        "copy_mode": "raw_safetensors_byte_ranges_v1",
        "selection_policy": "main_model_non_routed_bf16_f32_v1",
        "profile": "glm52-reap-504b-v2",
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
        "plan_sha256": inventory["plan_sha256"],
        "retained_tensor_count": 1_194,
        "parameter_count": 18_560_731_704,
        "tensor_payload_bytes": 37_121_488_608,
        "dtype_tensor_counts": {"BF16": 1_119, "F32": 75},
        "excluded_mtp_tensor_count": 2_039,
        "excluded_routed_tensor_count": 151_200,
        "excluded_runtime_vq_tensor_count": 0,
        "max_shard_payload_bytes": 4_294_967_296,
        "shard_count": 9,
        "index_path": "model.safetensors.index.json",
        "package_index_sha256": NON_VQ_PACKAGE_INDEX_SHA256,
        "package_set_sha256": inventory["package_set_sha256"],
        "manifest_sha256": NON_VQ_MANIFEST_SHA256,
        "written_shards": 0,
        "reused_shards": 9,
        "resume_verified": True,
        "largest_copy_buffer_bytes": 0,
        "package_audit_pass": True,
        "package_audit": {
            "artifact_dir": "artifacts/non-vq",
            "manifest_path": "artifacts/non-vq/non-vq-manifest.json",
            "artifact_tree_bytes": 37_122_403_011,
            "audit_pass": True,
            "checks": dict(NON_VQ_AUDIT_CHECKS),
            "manifest_sha256": NON_VQ_MANIFEST_SHA256,
            "package_set_sha256": inventory["package_set_sha256"],
            "parameter_count": 18_560_731_704,
            "retained_tensor_count": 1_194,
            "shard_count": 9,
            "tensor_payload_bytes": 37_121_488_608,
        },
        "shards": shards,
        "tensors": tensors,
    }


def _composite_artifact_audit() -> dict[str, object]:
    non_vq = _non_vq_evidence()
    return {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_artifact_audit",
        "audit_status": "glm52_modelopt_nvfp4_artifact_audit_ready",
        "audit_pass": True,
        "artifact_integrity_pass": True,
        "audit_blockers": [],
        "materialization_scope": "full",
        "accounting_scope": "full_composite_actual",
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "manifest_sha256": ROUTED_MANIFEST_SHA256,
        "group_set_sha256": ROUTED_GROUP_SET_SHA256,
        "expected_group_keys": list(GROUP_KEYS),
        "ready_group_keys": list(GROUP_KEYS),
        "complete_layer_ids": list(range(3, 78)),
        "partial_layer_ids": [],
        "full_routed_artifact_ready": True,
        "dense_routed_experts": False,
        "resume_verified": True,
        "byte_identity_verified": True,
        "requested_code_bits": 8,
        "requested_group_size": 512,
        "requested_scale_estimator": "max_abs",
        "checks": {
            "artifact_hashes_and_sizes": True,
            "bounded_whole_model_claims_suppressed": True,
            "exact_artifact_files": True,
            "exact_group_selection": True,
            "exact_manifest_contract": True,
            "exact_tensor_contract": True,
            "no_dense_routed_experts": True,
            "no_layer_78": True,
            "pinned_lineage": True,
        },
        "actual_routed_codes_bytes": 59_454_259_200,
        "actual_routed_scales_bytes": 1_857_945_600,
        "actual_routed_codebook_bytes": 230_400,
        "actual_routed_payload_bytes": 61_312_204_800,
        "actual_routed_weight_count": 475_634_073_600,
        "actual_routed_bpw": 1.03125,
        "main_routed_parameter_count": 475_634_073_600,
        "main_routed_tensor_count": 151_200,
        "main_non_routed_parameter_count": 18_560_731_704,
        "main_non_routed_tensor_count": 1_194,
        "main_non_routed_tensor_payload_bytes": 37_121_488_608,
        "main_model_parameter_count_excluding_mtp": ACCEPTED_PARAMETER_COUNT,
        "actual_whole_model_tensor_payload_bytes": (
            ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "actual_whole_model_tensor_payload_bpw": ACCEPTED_TENSOR_PAYLOAD_BPW,
        "whole_model_tensor_payload_values_actual": True,
        "whole_model_physical_values_actual": True,
        "whole_model_values_actual": True,
        "whole_model_artifact_blocker": None,
        "non_vq_package_evidence_authenticated": True,
        "non_vq_evidence_sha256": NON_VQ_EVIDENCE_FILE_SHA256,
        "non_vq_package_audit": {
            "audit_pass": True,
            "checks": dict(NON_VQ_AUDIT_CHECKS),
            "manifest_sha256": non_vq["manifest_sha256"],
            "package_set_sha256": non_vq["package_set_sha256"],
            "parameter_count": 18_560_731_704,
            "retained_tensor_count": 1_194,
            "shard_count": 9,
            "tensor_payload_bytes": 37_121_488_608,
        },
    }


def _common_artifact_identity() -> dict[str, object]:
    non_vq = _non_vq_evidence()
    return {
        "schema_version": 1,
        "identity_kind": "glm52_production_composite_v1",
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "profile": "glm52-reap-504b-v2",
        "profile_sha256": PROFILE_SHA256,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "source_index_sha256": INDEX_SHA256,
        "source_blob_inventory_sha256": SOURCE_BLOB_INVENTORY_SHA256,
        "source_inventory_sha256": non_vq["source_inventory_sha256"],
        "non_vq_manifest_sha256": non_vq["manifest_sha256"],
        "non_vq_package_index_sha256": non_vq["package_index_sha256"],
        "non_vq_package_set_sha256": non_vq["package_set_sha256"],
        "routed_manifest_sha256": ROUTED_MANIFEST_SHA256,
        "routed_group_set_sha256": ROUTED_GROUP_SET_SHA256,
        "routed_group_inventory_sha256": canonical_sha256(list(GROUP_KEYS)),
        "routed_group_count": 225,
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "whole_model_parameter_count": ACCEPTED_PARAMETER_COUNT,
        "whole_model_tensor_payload_bytes": ACCEPTED_TENSOR_PAYLOAD_BYTES,
        "whole_model_tensor_payload_bpw": ACCEPTED_TENSOR_PAYLOAD_BPW,
    }


PRODUCTION_PHASE_LABELS = (
    "before_memory_policy",
    "before_input_validation",
    "after_fresh_payload_audits",
    "after_tokenizer_and_prompt",
    "before_model_construction",
    "after_model_construction",
    "after_non_vq_bind",
    "after_routed_bind",
    "before_residency_barrier",
    "after_residency_quiet",
    "before_generation_warmup",
    "after_generation_warmup",
    "after_generation_warmup_cache_clear",
    "after_generation_warmup_quiet",
    "before_generation",
    "after_generation",
)
PRODUCTION_PHASE_ALIASES = {
    "cold_residency_memory": "after_residency_quiet",
    "generation_warmup_phase_memory": "after_generation_warmup",
    "post_warmup_cache_clear_phase_memory": (
        "after_generation_warmup_cache_clear"
    ),
    "post_warmup_quiet_phase_memory": "after_generation_warmup_quiet",
    "pre_generation_memory": "before_generation",
    "steady_state_generation_memory": "after_generation",
}


def _production_phase_memory() -> list[dict[str, object]]:
    dirty_deltas = {
        "after_fresh_payload_audits": (95, 0),
        "after_non_vq_bind": (7, 0),
        "after_residency_quiet": (159, 411_112),
        "after_generation_warmup": (300, 87_081),
    }
    pageouts_total = 90_289
    swapouts_total = 5_097_656
    rows: list[dict[str, object]] = []
    for index, label in enumerate(PRODUCTION_PHASE_LABELS):
        if index == 0:
            pageouts_delta: int | None = None
            swapouts_delta: int | None = None
        else:
            pageouts_delta, swapouts_delta = dirty_deltas.get(label, (0, 0))
            pageouts_total += pageouts_delta
            swapouts_total += swapouts_delta
        rows.append(
            {
                "available": True,
                "label": label,
                "mlx_active_bytes": ACCEPTED_TENSOR_PAYLOAD_BYTES,
                "mlx_cache_bytes": 0,
                "mlx_peak_bytes": ACCEPTED_TENSOR_PAYLOAD_BYTES,
                "pageouts_delta": pageouts_delta,
                "pageouts_total": pageouts_total,
                "pages_free": 4_096,
                "rss_bytes": 57_266_094_080,
                "swapouts_delta": swapouts_delta,
                "swapouts_total": swapouts_total,
            }
        )
    return rows


def _bind_production_phase_aliases(production: dict[str, object]) -> None:
    phase_memory = production["phase_memory"]
    assert isinstance(phase_memory, list)
    by_label = {
        str(row["label"]): row for row in phase_memory if isinstance(row, dict)
    }
    for field, label in PRODUCTION_PHASE_ALIASES.items():
        production[field] = deepcopy(by_label[label])


def _production_generation() -> dict[str, object]:
    identity = _common_artifact_identity()
    identity_sha256 = canonical_sha256(identity)
    loaded_model_parameters = [f"model.parameter.{index}" for index in range(1_272)]
    transformed_kv_b_tensors = [
        f"model.layers.{layer}.self_attn.kv_b_proj.weight" for layer in range(78)
    ]
    phase_memory = _production_phase_memory()
    production: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_generation_probe",
        "probe_status": "glm52_production_generation_probe_ready",
        "probe_pass": True,
        "full_model_scope": "production_composite",
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "production_artifact_used": True,
        "production_binding_proven": True,
        "production_generation_proven": True,
        "whole_model_runtime_proven": True,
        "accepted_whole_model_tensor_payload_bytes": (
            ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "accepted_whole_model_tensor_payload_bpw": ACCEPTED_TENSOR_PAYLOAD_BPW,
        "full_routed_group_count": 225,
        "artifact_identity": dict(identity),
        "artifact_identity_sha256": identity_sha256,
        "common_artifact_identity": dict(identity),
        "common_artifact_identity_sha256": identity_sha256,
        "input_evidence_file_sha256": {
            "composite_audit_json": (
                "8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a"
            ),
            "family_policy_json": (
                "0975f7dc1117c5fba7532e9166f4767546fd691a6cb52520a40f09874f972ce2"
            ),
            "full_bind_preflight_json": (
                "9078a36c683a2487ef5d99f4c8fecc72f086689b8aa1e249ffaa34639b5946a3"
            ),
            "materialization_runs_jsonl": "5" * 64,
            "non_vq_evidence_json": NON_VQ_EVIDENCE_FILE_SHA256,
            "tokenizer_readiness_json": "2" * 64,
        },
        "input_fingerprint_reverified_before_bind": True,
        "input_fingerprint_reverified_after_residency": True,
        "input_fingerprint_reverified_after_warmup": True,
        "input_fingerprint_reverified_after_warmup_preparation": True,
        "input_fingerprint_reverified_before_generation": True,
        "input_fingerprint_reverified_after_generation": True,
        "parameter_residency_established": True,
        "generation_warmup_proven": True,
        "generation_warmup": {
            "explicit_prompt_cache": True,
            "generated_ids_within_vocab": True,
            "generated_token_count": 1,
            "generated_token_ids": [785],
            "generation_method": "mlx_lm_generate_step",
            "greedy_sampling": True,
            "logits_finite": True,
            "logits_shape": [154_880],
            "lookahead_forward_scheduled": True,
        },
        "phase_memory": phase_memory,
        "cold_residency_mlx_peak_scope": "since_probe_start",
        "steady_state_generation_mlx_peak_scope": "since_pre_generation_reset",
        "generation_warmup_memory_clean": False,
        "measured_generation_after_warmup": True,
        "warmup_measured_token_match": True,
        "measured_generation_avoids_lookahead": True,
        "generation_measured_after_residency": True,
        "mlx_peak_reset_before_generation": True,
        "cold_residency_memory_clean": False,
        "pre_generation_memory_clean": True,
        "steady_state_generation_memory_clean": True,
        "warm_residency_proven": True,
        "production_residency_proven": False,
        "non_vq_bind_report": {
            "loaded_count": 1_272,
            "loaded_model_parameters": loaded_model_parameters,
            "missing_model_parameters": [],
            "skipped_mtp_tensors": [],
            "skipped_routed_expert_tensors": [],
            "skipped_unmatched_tensors": [],
            "transformed_kv_b_tensors": transformed_kv_b_tensors,
        },
        "bound_sparse_layer_ids": list(range(3, 78)),
        "dense_routed_parameter_names": [],
        "dense_routed_experts": False,
        "unbound_vq_experts": False,
        "tokenizer_used": True,
        "chat_template_add_generation_prompt": True,
        "chat_template_enable_thinking": False,
        "explicit_prompt_cache": True,
        "prompt": "The capital of France is",
        "prompt_token_ids": [
            154_822,
            154_824,
            154_827,
            785,
            6_722,
            315,
            9_621,
            374,
            154_828,
            154_841,
            154_842,
        ],
        "prompt_token_count": 11,
        "prefill_token_count": 10,
        "decode_token_count": 1,
        "generated_text": "The",
        "generated_token_ids": [785],
        "generated_token_count": 1,
        "generated_ids_within_vocab": True,
        "logits_shape": [154_880],
        "logits_finite": True,
        "greedy_sampling": True,
        "generation_method": "direct_prefill_single_decode",
        "lookahead_forward_scheduled": False,
        "rss_semantics": "process_peak_ru_maxrss",
        "wired_memory_policy": {
            "environment_overrides": {},
            "iogpu_wired_limit_available": True,
            "iogpu_wired_limit_mb": 0,
            "platform": "Darwin",
            "system_wired_default": True,
        },
        "system_wired_default": True,
        "custom_mlx_wired_limit_used": False,
        "long_context_indexshare_proven": False,
        "quality_claim": False,
        "speed_claim": False,
        "same_machine_benchmark_proven": False,
        "full_vocabulary_eval_proven": False,
    }
    _bind_production_phase_aliases(production)
    return production


@pytest.fixture
def trusted_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "validate_glm52_family_gate_policy", lambda value: value)
    monkeypatch.setattr(
        gate,
        "validate_glm52_family_prompt_pack",
        lambda value: _prompt_rows(),
    )


def _kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "family_policy": _policy(),
        "eval_prompt_pack": _prompt_pack(),
        "teacher_metadata": _teacher_metadata(tmp_path),
        "full_bind_preflight": _full_bind(),
        "indexshare_runtime": _indexshare_runtime(),
        "synthetic_generation": _synthetic_generation(),
    }


def _v2_kwargs(tmp_path: Path) -> dict[str, object]:
    kwargs = _kwargs(tmp_path)
    kwargs.update(
        {
            "full_bind_preflight": _full_bind(present=225),
            "non_vq_evidence": _non_vq_evidence(),
            "composite_artifact_audit": _composite_artifact_audit(),
            "production_generation": _production_generation(),
        }
    )
    return kwargs


def _replace_nested(
    payload: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    current: dict[str, object] = payload
    for field in path[:-1]:
        nested = current[field]
        assert isinstance(nested, dict)
        current = nested
    current[path[-1]] = deepcopy(value)


def _write_v2_cli_inputs(
    tmp_path: Path,
    *,
    composite_audit_sha256: str | None,
) -> tuple[dict[str, Path], dict[str, object], list[str], Path]:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs.pop("production_generation")
    audit = kwargs.pop("composite_artifact_audit")
    non_vq = kwargs.pop("non_vq_evidence")
    assert isinstance(production, dict)
    assert isinstance(audit, dict)

    paths: dict[str, Path] = {}
    for name, payload in {**kwargs, "non_vq_evidence": non_vq}.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths[name] = path

    audit["non_vq_evidence_sha256"] = sha256_file(paths["non_vq_evidence"])
    audit_path = tmp_path / "composite_artifact_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    paths["composite_artifact_audit"] = audit_path

    input_hashes = production["input_evidence_file_sha256"]
    assert isinstance(input_hashes, dict)
    input_hashes.update(
        {
            "family_policy_json": sha256_file(paths["family_policy"]),
            "full_bind_preflight_json": sha256_file(
                paths["full_bind_preflight"]
            ),
            "non_vq_evidence_json": sha256_file(paths["non_vq_evidence"]),
            "composite_audit_json": (
                composite_audit_sha256
                if composite_audit_sha256 is not None
                else sha256_file(audit_path)
            ),
        }
    )
    production_path = tmp_path / "production_generation.json"
    production_path.write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )
    paths["production_generation"] = production_path

    output = tmp_path / "gate-v2.json"
    argv = [
        "--family-policy-json",
        str(paths["family_policy"]),
        "--eval-prompt-pack-json",
        str(paths["eval_prompt_pack"]),
        "--teacher-metadata-json",
        str(paths["teacher_metadata"]),
        "--full-bind-preflight-json",
        str(paths["full_bind_preflight"]),
        "--indexshare-runtime-json",
        str(paths["indexshare_runtime"]),
        "--synthetic-generation-json",
        str(paths["synthetic_generation"]),
        "--non-vq-evidence-json",
        str(paths["non_vq_evidence"]),
        "--composite-artifact-audit-json",
        str(paths["composite_artifact_audit"]),
        "--production-generation-json",
        str(paths["production_generation"]),
        "--output-json",
        str(output),
    ]
    return paths, production, argv, output


def test_current_six_group_state_is_valid_but_fail_loud(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    report = gate.check_glm52_family_gate(**_kwargs(tmp_path))

    assert report["gate_schema_version"] == 1
    assert report["family_gate_pass"] is False
    assert report["expected_routed_group_count"] == 225
    assert report["present_routed_group_count"] == 6
    assert report["missing_routed_group_count"] == 219
    assert report["checks"]["teacher_source_metadata_ready"] is True
    assert report["checks"]["full_225_group_artifact_ready"] is False
    assert report["checks"]["production_binding_and_generation_ready"] is False
    assert report["missing_requirements"] == [
        "dequantized_source_teacher_cache_payload",
        "full_225_group_artifact",
        "production_model_bind_and_generation",
        "full_vocabulary_source_relative_family_eval",
        "route_math_diagnostics",
        "same_machine_pinned_fp4_benchmark",
    ]
    assert set(report["input_evidence_contract_sha256"]) == {
        "family_policy",
        "eval_prompt_pack",
        "teacher_metadata",
        "full_bind_preflight",
        "indexshare_runtime",
        "synthetic_generation",
    }
    assert "common_artifact_identity" not in report
    assert "common_artifact_identity_sha256" not in report
    assert "raw_evidence_validator_readiness" not in report


def test_latest_schema_alias_advances_after_declarative_evaluator_upgrade() -> None:
    assert GLM52_FAMILY_GATE_V1_SCHEMA_VERSION == 1
    assert GLM52_FAMILY_GATE_V2_SCHEMA_VERSION == 2
    assert GLM52_FAMILY_GATE_V3_SCHEMA_VERSION == 3
    assert GLM52_FAMILY_GATE_V4_SCHEMA_VERSION == 4
    assert GLM52_FAMILY_GATE_V5_SCHEMA_VERSION == 5
    assert GLM52_FAMILY_GATE_V6_SCHEMA_VERSION == 6
    assert GLM52_FAMILY_GATE_SCHEMA_VERSION == GLM52_FAMILY_GATE_V6_SCHEMA_VERSION


def test_gate_refuses_synthetic_release_summaries_until_raw_validators_exist(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _kwargs(tmp_path)
    kwargs.update(
        {
            "full_bind_preflight": _full_bind(present=225),
            "family_eval_gate": _quality_gate(),
            "family_benchmark_gate": _benchmark_gate(),
            "production_generation": _optimistic_production_summary(),
        }
    )

    with pytest.raises(ValueError, match="raw release evidence validators"):
        gate.check_glm52_family_gate(**kwargs)


def test_raw_artifact_and_production_evidence_emit_schema_v2_checkpoint(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    report = gate.check_glm52_family_gate(**_v2_kwargs(tmp_path))
    identity = _common_artifact_identity()

    assert report["gate_schema_version"] == 2
    assert report["checks"]["full_225_group_artifact_ready"] is True
    assert report["checks"]["accepted_artifact_bytes_and_bpw_ready"] is True
    assert report["checks"]["production_binding_and_generation_ready"] is True
    assert report["checks"]["teacher_cache_full_vocabulary_ready"] is False
    assert (
        report["checks"]["full_vocabulary_source_relative_family_eval_ready"]
        is False
    )
    assert report["checks"]["route_math_diagnostics_ready"] is False
    assert report["checks"]["same_machine_pinned_fp4_benchmark_ready"] is False
    assert report["missing_requirements"] == [
        "dequantized_source_teacher_cache_payload",
        "full_vocabulary_source_relative_family_eval",
        "route_math_diagnostics",
        "same_machine_pinned_fp4_benchmark",
    ]
    assert report["release_pass_enabled"] is False
    assert report["raw_release_evidence_validators_ready"] is False
    assert report["family_gate_pass"] is False
    assert report["common_artifact_identity"] == identity
    assert report["common_artifact_identity_sha256"] == canonical_sha256(identity)
    assert report["raw_evidence_validator_readiness"] == {
        "artifact": True,
        "production": True,
        "family_eval": False,
        "family_benchmark": False,
    }
    assert set(report["input_evidence_contract_sha256"]) == {
        "family_policy",
        "eval_prompt_pack",
        "teacher_metadata",
        "full_bind_preflight",
        "indexshare_runtime",
        "synthetic_generation",
        "non_vq_evidence",
        "composite_artifact_audit",
        "production_generation",
    }
    authenticated = dict(report)
    embedded = authenticated.pop("gate_contract_sha256")
    assert embedded == canonical_sha256(authenticated)


def _clean_teacher_cache_audit() -> gate.GLM52TeacherCacheAudit:
    return gate.GLM52TeacherCacheAudit(
        valid=True,
        release_eligible=True,
        cache_content_sha256="a" * 64,
        manifest_body_sha256="b" * 64,
        prompt_count=66,
        source_token_count=810,
        predictor_position_count=744,
        fp32_value_count=115_230_720,
        raw_tensor_bytes=460_922_880,
        split_position_counts={
            "report": 238,
            "selection": 255,
            "holdout": 251,
        },
        split_raw_tensor_bytes={
            "report": 147_445_760,
            "selection": 157_977_600,
            "holdout": 155_499_520,
        },
    )


def _raw_teacher_cache_path_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    audit: gate.GLM52TeacherCacheAudit,
) -> dict[str, Path]:
    cache_root = tmp_path / "raw-teacher-cache"
    cache_root.mkdir()
    manifest = cache_root / "glm52-teacher-cache-fp32-manifest.json"
    manifest.write_text("{}\n")
    prompt_authority = tmp_path / "prompt-authority.json"
    prompt_authority.write_text("{}\n")
    monkeypatch.setattr(
        gate,
        "_audit_raw_teacher_cache",
        lambda **_kwargs: audit,
    )
    return {
        "teacher_cache_root": cache_root,
        "teacher_cache_manifest_path": manifest,
        "teacher_cache_prompt_authority_path": prompt_authority,
    }


def test_clean_raw_teacher_cache_advances_complete_v2_trio_to_schema_v3(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_paths = _raw_teacher_cache_path_kwargs(
        tmp_path,
        monkeypatch,
        audit=_clean_teacher_cache_audit(),
    )
    report = gate.check_glm52_family_gate(
        **_v2_kwargs(tmp_path),
        **cache_paths,
    )

    assert report["gate_schema_version"] == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
    assert report["checks"]["teacher_cache_full_vocabulary_ready"] is True
    assert report["teacher_cache_payload_present"] is True
    assert report["missing_requirements"] == list(
        GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS
    )
    assert report["missing_requirements"] == [
        "full_vocabulary_source_relative_family_eval",
        "route_math_diagnostics",
        "same_machine_pinned_fp4_benchmark",
    ]
    identity = report["audited_teacher_cache_identity"]
    assert identity["release_eligible"] is True
    assert identity["manifest_file_sha256"] == sha256_file(
        cache_paths["teacher_cache_manifest_path"]
    )
    assert identity["artifact_sha256"] == report["raw_teacher_cache_authority"][
        "artifact_sha256"
    ]
    assert report["audited_teacher_cache_identity_sha256"] == canonical_sha256(
        identity
    )
    assert report["raw_evidence_validator_readiness"] == {
        "artifact": True,
        "production": True,
        "teacher_cache": True,
        "family_eval": False,
        "family_benchmark": False,
    }


def test_raw_release_evidence_advances_v3_through_v6_only_after_all_validators(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each release schema is driven by raw-path validators, never summaries."""

    cache_paths = _raw_teacher_cache_path_kwargs(
        tmp_path,
        monkeypatch,
        audit=_clean_teacher_cache_audit(),
    )
    candidate_root = tmp_path / "candidate-cache"
    candidate_root.mkdir()
    source_trace_root = tmp_path / "source-traces"
    source_trace_root.mkdir()
    candidate_trace_root = tmp_path / "candidate-traces"
    candidate_trace_root.mkdir()
    source_authority = tmp_path / "source-authority.json"
    candidate_authority = tmp_path / "candidate-authority.json"
    for path in (source_authority, candidate_authority):
        path.write_text("{}\n")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(_sealed_benchmark_evidence(), indent=2, sort_keys=True) + "\n"
    )
    frozen_policy = tmp_path / "family-policy.json"
    frozen_policy.write_text(json.dumps(_policy()) + "\n")

    monkeypatch.setattr(
        gate,
        "compare_glm52_candidate_caches",
        lambda *_args, **_kwargs: {"family_eval_gate_pass": True},
    )
    monkeypatch.setattr(
        gate,
        "compare_route_trace_artifacts",
        lambda *_args, **_kwargs: {
            "evaluation": {"diagnostic_only": True, "release_eligible": True},
            "checks": {"structural_math_pass": True},
        },
    )
    monkeypatch.setattr(
        gate,
        "validate_capture_authority",
        lambda value, **_kwargs: dict(value),
    )

    base = {
        **_v2_kwargs(tmp_path),
        **cache_paths,
        "candidate_cache_root": candidate_root,
        "family_policy_path": frozen_policy,
    }
    v4 = gate.check_glm52_family_gate(**base)
    assert v4["gate_schema_version"] == 4
    assert v4["checks"]["full_vocabulary_source_relative_family_eval_ready"]
    assert v4["missing_requirements"] == [
        "route_math_diagnostics",
        "same_machine_pinned_fp4_benchmark",
    ]

    v5 = gate.check_glm52_family_gate(
        **base,
        source_route_trace_root=source_trace_root,
        candidate_route_trace_root=candidate_trace_root,
        source_route_authority_path=source_authority,
        candidate_route_authority_path=candidate_authority,
    )
    assert v5["gate_schema_version"] == 5
    assert v5["checks"]["route_math_diagnostics_ready"] is True
    assert v5["missing_requirements"] == ["same_machine_pinned_fp4_benchmark"]

    v6 = gate.check_glm52_family_gate(
        **base,
        source_route_trace_root=source_trace_root,
        candidate_route_trace_root=candidate_trace_root,
        source_route_authority_path=source_authority,
        candidate_route_authority_path=candidate_authority,
        benchmark_evidence_path=benchmark,
    )
    assert v6["gate_schema_version"] == 6
    assert v6["family_gate_pass"] is True
    assert v6["release_pass_enabled"] is True


def test_raw_candidate_eval_below_policy_keeps_the_v3_blocked_shape(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_paths = _raw_teacher_cache_path_kwargs(
        tmp_path, monkeypatch, audit=_clean_teacher_cache_audit()
    )
    candidate_root = tmp_path / "candidate-cache"
    candidate_root.mkdir()
    policy_path = tmp_path / "family-policy.json"
    policy_path.write_text(json.dumps(_policy()) + "\n")
    monkeypatch.setattr(
        gate,
        "compare_glm52_candidate_caches",
        lambda *_args, **_kwargs: {"family_eval_gate_pass": False},
    )

    report = gate.check_glm52_family_gate(
        **_v2_kwargs(tmp_path),
        **cache_paths,
        candidate_cache_root=candidate_root,
        family_policy_path=policy_path,
    )

    assert report["gate_schema_version"] == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
    assert report["checks"]["full_vocabulary_source_relative_family_eval_ready"] is False
    assert "full_vocabulary_source_relative_family_eval" in report["missing_requirements"]


def test_forged_eval_route_authority_and_dirty_benchmark_are_invalid(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_paths = _raw_teacher_cache_path_kwargs(
        tmp_path, monkeypatch, audit=_clean_teacher_cache_audit()
    )
    candidate_root = tmp_path / "candidate-cache"
    candidate_root.mkdir()
    policy_path = tmp_path / "family-policy.json"
    policy_path.write_text(json.dumps(_policy()) + "\n")
    base = {
        **_v2_kwargs(tmp_path),
        **cache_paths,
        "candidate_cache_root": candidate_root,
        "family_policy_path": policy_path,
    }
    monkeypatch.setattr(
        gate,
        "compare_glm52_candidate_caches",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forged eval")),
    )
    with pytest.raises(ValueError, match="forged eval"):
        gate.check_glm52_family_gate(**base)

    monkeypatch.setattr(
        gate,
        "compare_glm52_candidate_caches",
        lambda *_args, **_kwargs: {"family_eval_gate_pass": True},
    )
    source_trace = tmp_path / "source-trace"
    candidate_trace = tmp_path / "candidate-trace"
    source_trace.mkdir()
    candidate_trace.mkdir()
    source_authority = tmp_path / "source-authority.json"
    candidate_authority = tmp_path / "candidate-authority.json"
    source_authority.write_text("{}\n")
    candidate_authority.write_text("{}\n")
    monkeypatch.setattr(
        gate,
        "validate_capture_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not release class")),
    )
    with pytest.raises(ValueError, match="not release class"):
        gate.check_glm52_family_gate(
            **base,
            source_route_trace_root=source_trace,
            candidate_route_trace_root=candidate_trace,
            source_route_authority_path=source_authority,
            candidate_route_authority_path=candidate_authority,
        )

    dirty_benchmark = _sealed_benchmark_evidence(dirty_memory=True)
    benchmark_path = tmp_path / "dirty-benchmark.json"
    benchmark_path.write_text(
        json.dumps(dirty_benchmark, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(gate, "validate_capture_authority", lambda value, **_kwargs: dict(value))
    monkeypatch.setattr(
        gate,
        "compare_route_trace_artifacts",
        lambda *_args, **_kwargs: {
            "evaluation": {"diagnostic_only": True, "release_eligible": True},
            "checks": {"structural_math_pass": True},
        },
    )
    with pytest.raises(ValueError, match="dirty memory"):
        gate.check_glm52_family_gate(
            **base,
            source_route_trace_root=source_trace,
            candidate_route_trace_root=candidate_trace,
            source_route_authority_path=source_authority,
            candidate_route_authority_path=candidate_authority,
            benchmark_evidence_path=benchmark_path,
        )


def test_public_checker_cannot_emit_schema_v3_from_prepared_audit_dataclass(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    with pytest.raises(TypeError, match="teacher_cache_audit"):
        gate.check_glm52_family_gate(
            **_v2_kwargs(tmp_path),
            teacher_cache_audit=_clean_teacher_cache_audit(),
            teacher_cache_manifest_sha256="c" * 64,
        )


def test_memory_dirty_teacher_cache_is_invalid_instead_of_schema_v3(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirty = _clean_teacher_cache_audit()
    dirty = gate.GLM52TeacherCacheAudit(
        **{**dirty.__dict__, "release_eligible": False}
    )
    cache_paths = _raw_teacher_cache_path_kwargs(
        tmp_path,
        monkeypatch,
        audit=dirty,
    )

    with pytest.raises(ValueError, match="release eligible"):
        gate.check_glm52_family_gate(
            **_v2_kwargs(tmp_path),
            **cache_paths,
        )


def test_teacher_cache_requires_complete_schema_v2_raw_trio(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    with pytest.raises(ValueError, match="complete schema-v2 raw trio"):
        gate.check_glm52_family_gate(
            **_kwargs(tmp_path),
            teacher_cache_root=tmp_path / "raw-teacher-cache",
            teacher_cache_manifest_path=(
                tmp_path
                / "raw-teacher-cache"
                / "glm52-teacher-cache-fp32-manifest.json"
            ),
            teacher_cache_prompt_authority_path=tmp_path / "prompts.json",
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("tensors", "missing"),
        ("tensors", "empty"),
        ("shards", "missing"),
        ("shards", "empty"),
    ],
)
def test_schema_v2_requires_complete_non_vq_inventories(
    tmp_path: Path,
    trusted_contracts: None,
    field: str,
    mutation: str,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    non_vq = kwargs["non_vq_evidence"]
    assert isinstance(non_vq, dict)
    if mutation == "missing":
        non_vq.pop(field)
    else:
        non_vq[field] = []

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    ["reorder-tensors", "duplicate-tensor", "reorder-shards", "duplicate-shard"],
)
def test_schema_v2_rejects_non_vq_inventory_order_or_duplicates(
    tmp_path: Path,
    trusted_contracts: None,
    mutation: str,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    non_vq = kwargs["non_vq_evidence"]
    assert isinstance(non_vq, dict)
    tensors = non_vq["tensors"]
    shards = non_vq["shards"]
    assert isinstance(tensors, list)
    assert isinstance(shards, list)
    if mutation == "reorder-tensors":
        tensors[0], tensors[1] = tensors[1], tensors[0]
    elif mutation == "duplicate-tensor":
        tensors[1] = deepcopy(tensors[0])
    elif mutation == "reorder-shards":
        shards[0], shards[1] = shards[1], shards[0]
    else:
        shards[1] = deepcopy(shards[0])

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    ["parameter-count", "output-offset", "shard-membership", "shard-count"],
)
def test_schema_v2_rejects_non_vq_inventory_accounting_drift(
    tmp_path: Path,
    trusted_contracts: None,
    mutation: str,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    non_vq = kwargs["non_vq_evidence"]
    assert isinstance(non_vq, dict)
    tensors = non_vq["tensors"]
    shards = non_vq["shards"]
    assert isinstance(tensors, list)
    assert isinstance(shards, list)
    first_tensor = tensors[0]
    first_shard = shards[0]
    assert isinstance(first_tensor, dict)
    assert isinstance(first_shard, dict)
    if mutation == "parameter-count":
        first_tensor["parameter_count"] = int(first_tensor["parameter_count"]) + 1
    elif mutation == "output-offset":
        offsets = first_tensor["output_offsets"]
        assert isinstance(offsets, list)
        offsets[1] = int(offsets[1]) + 1
    elif mutation == "shard-membership":
        first_tensor["output_shard"] = "model-00010-of-00009.safetensors"
    else:
        first_shard["tensor_count"] = int(first_shard["tensor_count"]) - 1

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def _replace_non_vq_package_set(
    kwargs: dict[str, object],
    digest: str,
) -> None:
    non_vq = kwargs["non_vq_evidence"]
    composite = kwargs["composite_artifact_audit"]
    production = kwargs["production_generation"]
    assert isinstance(non_vq, dict)
    assert isinstance(composite, dict)
    assert isinstance(production, dict)
    package_audit = non_vq["package_audit"]
    composite_audit = composite["non_vq_package_audit"]
    identity = production["artifact_identity"]
    assert isinstance(package_audit, dict)
    assert isinstance(composite_audit, dict)
    assert isinstance(identity, dict)
    non_vq["package_set_sha256"] = digest
    package_audit["package_set_sha256"] = digest
    composite_audit["package_set_sha256"] = digest
    identity["non_vq_package_set_sha256"] = digest
    identity_sha256 = canonical_sha256(identity)
    production["artifact_identity_sha256"] = identity_sha256
    production["common_artifact_identity"] = deepcopy(identity)
    production["common_artifact_identity_sha256"] = identity_sha256


@pytest.mark.parametrize(
    "mutation",
    ["shard-inventory-digest", "file-digest-shape", "package-set-digest"],
)
def test_schema_v2_rejects_non_vq_inventory_hash_drift(
    tmp_path: Path,
    trusted_contracts: None,
    mutation: str,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    non_vq = kwargs["non_vq_evidence"]
    assert isinstance(non_vq, dict)
    shards = non_vq["shards"]
    assert isinstance(shards, list)
    first_shard = shards[0]
    assert isinstance(first_shard, dict)
    if mutation == "shard-inventory-digest":
        first_shard["tensor_inventory_sha256"] = "0" * 64
    elif mutation == "file-digest-shape":
        first_shard["file_sha256"] = "not-a-sha256"
    else:
        _replace_non_vq_package_set(kwargs, "0" * 64)

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_schema_v2_rejects_stale_non_vq_plan_digest(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    non_vq = kwargs["non_vq_evidence"]
    assert isinstance(non_vq, dict)
    non_vq["plan_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_schema_v2_rejects_coordinated_non_vq_tensor_plan_rehash(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    non_vq = kwargs["non_vq_evidence"]
    assert isinstance(non_vq, dict)
    tensors = non_vq["tensors"]
    shards = non_vq["shards"]
    assert isinstance(tensors, list)
    assert isinstance(shards, list)
    first_tensor = tensors[0]
    first_shard = shards[0]
    assert isinstance(first_tensor, dict)
    assert isinstance(first_shard, dict)
    first_tensor["source_shard"] = "model-99999.safetensors"
    first_filename = first_shard["filename"]
    members = [
        {key: value for key, value in tensor.items() if key != "payload_sha256"}
        for tensor in tensors
        if isinstance(tensor, dict) and tensor.get("output_shard") == first_filename
    ]
    first_shard["tensor_inventory_sha256"] = canonical_sha256(members)
    package_set_sha256 = canonical_sha256(
        {
            "package_index_sha256": non_vq["package_index_sha256"],
            "shards": shards,
        }
    )
    _replace_non_vq_package_set(kwargs, package_set_sha256)

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "empty", "reordered", "duplicate"],
)
def test_schema_v2_requires_exact_production_phase_memory_inventory(
    tmp_path: Path,
    trusted_contracts: None,
    mutation: str,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs["production_generation"]
    assert isinstance(production, dict)
    if mutation == "missing":
        production.pop("phase_memory")
    elif mutation == "empty":
        production["phase_memory"] = []
    else:
        phase_memory = production["phase_memory"]
        assert isinstance(phase_memory, list)
        if mutation == "reordered":
            phase_memory[0], phase_memory[1] = phase_memory[1], phase_memory[0]
        else:
            phase_memory[1] = deepcopy(phase_memory[0])

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_schema_v2_rejects_non_integer_production_memory_delta(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs["production_generation"]
    assert isinstance(production, dict)
    phase_memory = production["phase_memory"]
    assert isinstance(phase_memory, list)
    steady = phase_memory[-1]
    assert isinstance(steady, dict)
    steady["pageouts_delta"] = False
    _bind_production_phase_aliases(production)

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_schema_v2_rejects_production_memory_alias_mismatch(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs["production_generation"]
    assert isinstance(production, dict)
    production["steady_state_generation_memory"] = deepcopy(
        production["pre_generation_memory"]
    )

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_schema_v2_rejects_production_memory_delta_total_mismatch(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs["production_generation"]
    assert isinstance(production, dict)
    phase_memory = production["phase_memory"]
    assert isinstance(phase_memory, list)
    steady = phase_memory[-1]
    assert isinstance(steady, dict)
    steady["pageouts_delta"] = 1
    _bind_production_phase_aliases(production)

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def _shift_production_phase_delta(
    production: dict[str, object],
    *,
    label: str,
    pageouts_delta: int,
    swapouts_delta: int,
) -> None:
    phase_memory = production["phase_memory"]
    assert isinstance(phase_memory, list)
    phase_index = next(
        index
        for index, row in enumerate(phase_memory)
        if isinstance(row, dict) and row.get("label") == label
    )
    row = phase_memory[phase_index]
    assert isinstance(row, dict)
    old_pageouts_delta = row["pageouts_delta"]
    old_swapouts_delta = row["swapouts_delta"]
    assert isinstance(old_pageouts_delta, int)
    assert isinstance(old_swapouts_delta, int)
    row["pageouts_delta"] = pageouts_delta
    row["swapouts_delta"] = swapouts_delta
    pageouts_shift = pageouts_delta - old_pageouts_delta
    swapouts_shift = swapouts_delta - old_swapouts_delta
    for shifted in phase_memory[phase_index:]:
        assert isinstance(shifted, dict)
        shifted["pageouts_total"] = int(shifted["pageouts_total"]) + pageouts_shift
        shifted["swapouts_total"] = int(shifted["swapouts_total"]) + swapouts_shift
    _bind_production_phase_aliases(production)


@pytest.mark.parametrize(
    ("label", "pageouts_delta", "swapouts_delta"),
    [
        ("after_generation_warmup_cache_clear", 1, 0),
        ("after_generation_warmup_quiet", 0, 1),
        ("after_generation", 1, 0),
        ("after_residency_quiet", 0, 0),
        ("after_generation_warmup", 0, 0),
    ],
)
def test_schema_v2_rejects_contradictory_production_memory_claims(
    tmp_path: Path,
    trusted_contracts: None,
    label: str,
    pageouts_delta: int,
    swapouts_delta: int,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs["production_generation"]
    assert isinstance(production, dict)
    _shift_production_phase_delta(
        production,
        label=label,
        pageouts_delta=pageouts_delta,
        swapouts_delta=swapouts_delta,
    )

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


@pytest.mark.parametrize(
    "missing_input",
    [
        "non_vq_evidence",
        "composite_artifact_audit",
        "production_generation",
    ],
)
def test_schema_v2_raw_evidence_must_be_supplied_as_a_complete_trio(
    tmp_path: Path,
    trusted_contracts: None,
    missing_input: str,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    kwargs.pop(missing_input)

    with pytest.raises(ValueError, match="complete trio"):
        gate.check_glm52_family_gate(**kwargs)


@pytest.mark.parametrize(
    ("target", "path", "value"),
    [
        ("composite_artifact_audit", ("schema_version",), 2),
        ("composite_artifact_audit", ("record_type",), "wrong"),
        ("composite_artifact_audit", ("audit_status",), "blocked"),
        ("composite_artifact_audit", ("audit_pass",), False),
        ("composite_artifact_audit", ("audit_pass",), 1),
        ("composite_artifact_audit", ("source_revision",), "drift"),
        ("composite_artifact_audit", ("config_sha256",), "0" * 64),
        (
            "composite_artifact_audit",
            ("checks", "exact_tensor_contract"),
            False,
        ),
        ("composite_artifact_audit", ("audit_blockers",), ["blocker"]),
        (
            "composite_artifact_audit",
            ("expected_group_keys",),
            list(GROUP_KEYS[:-1]),
        ),
        (
            "composite_artifact_audit",
            ("ready_group_keys",),
            list(GROUP_KEYS[:-1]),
        ),
        (
            "composite_artifact_audit",
            ("expected_group_keys",),
            [GROUP_KEYS[1], GROUP_KEYS[0], *GROUP_KEYS[2:]],
        ),
        (
            "composite_artifact_audit",
            ("complete_layer_ids",),
            [*range(3, 78), 78],
        ),
        ("composite_artifact_audit", ("partial_layer_ids",), [77]),
        ("composite_artifact_audit", ("requested_code_bits",), 4),
        ("composite_artifact_audit", ("requested_group_size",), 256),
        (
            "composite_artifact_audit",
            ("requested_scale_estimator",),
            "mean_abs",
        ),
        (
            "composite_artifact_audit",
            ("actual_routed_codes_bytes",),
            59_454_259_199,
        ),
        (
            "composite_artifact_audit",
            ("actual_routed_scales_bytes",),
            1_857_945_599,
        ),
        (
            "composite_artifact_audit",
            ("actual_routed_codebook_bytes",),
            230_399,
        ),
        (
            "composite_artifact_audit",
            ("actual_routed_payload_bytes",),
            61_312_204_799,
        ),
        (
            "composite_artifact_audit",
            ("main_non_routed_tensor_payload_bytes",),
            37_121_488_607,
        ),
        (
            "composite_artifact_audit",
            ("actual_whole_model_tensor_payload_bytes",),
            ACCEPTED_TENSOR_PAYLOAD_BYTES - 1,
        ),
        (
            "composite_artifact_audit",
            ("actual_whole_model_tensor_payload_bpw",),
            1.5934433,
        ),
        (
            "composite_artifact_audit",
            ("main_model_parameter_count_excluding_mtp",),
            ACCEPTED_PARAMETER_COUNT - 1,
        ),
        (
            "composite_artifact_audit",
            ("whole_model_artifact_blocker",),
            "not_full_composite",
        ),
        (
            "family_policy",
            ("artifact_target", "tensor_payload_bytes"),
            ACCEPTED_TENSOR_PAYLOAD_BYTES - 1,
        ),
        (
            "family_policy",
            ("artifact_target", "whole_main_bpw"),
            1.5934432,
        ),
        ("non_vq_evidence", ("schema_version",), 2),
        ("non_vq_evidence", ("manifest_sha256",), "0" * 64),
        ("non_vq_evidence", ("retained_tensor_count",), 1_193),
        ("non_vq_evidence", ("shard_count",), 8),
        ("non_vq_evidence", ("source_inventory_sha256",), "0" * 64),
        (
            "non_vq_evidence",
            ("package_audit", "checks", "tensor_inventory"),
            False,
        ),
        (
            "production_generation",
            ("common_artifact_identity", "group_size"),
            256,
        ),
        ("production_generation", ("schema_version",), 2),
        ("production_generation", ("full_model_scope",), "tiny_fixture"),
        *[
            ("production_generation", (field,), False)
            for field in (
                "production_artifact_used",
                "production_binding_proven",
                "production_generation_proven",
                "whole_model_runtime_proven",
            )
        ],
        ("production_generation", ("dense_routed_experts",), True),
        (
            "production_generation",
            ("dense_routed_parameter_names",),
            ["model.layers.3.mlp.experts.0.gate_proj.weight"],
        ),
        ("production_generation", ("unbound_vq_experts",), True),
        ("production_generation", ("logits_finite",), False),
        ("production_generation", ("logits_shape",), [154_879]),
        ("production_generation", ("generated_token_count",), 2),
        ("production_generation", ("generated_token_ids",), [154_880]),
        (
            "production_generation",
            ("generation_warmup", "generated_token_ids"),
            [786],
        ),
        ("production_generation", ("generated_ids_within_vocab",), False),
        ("production_generation", ("greedy_sampling",), False),
        (
            "production_generation",
            ("generation_method",),
            "mlx_lm_generate_step",
        ),
        ("production_generation", ("lookahead_forward_scheduled",), True),
        (
            "production_generation",
            ("measured_generation_avoids_lookahead",),
            False,
        ),
        ("production_generation", ("custom_mlx_wired_limit_used",), True),
        ("production_generation", ("system_wired_default",), False),
        ("production_generation", ("warm_residency_proven",), False),
        ("production_generation", ("pre_generation_memory_clean",), False),
        (
            "production_generation",
            ("steady_state_generation_memory_clean",),
            False,
        ),
        (
            "production_generation",
            ("non_vq_bind_report", "loaded_count"),
            1_271,
        ),
        (
            "production_generation",
            ("non_vq_bind_report", "transformed_kv_b_tensors"),
            [
                f"model.layers.{layer}.self_attn.kv_b_proj.weight"
                for layer in range(77)
            ],
        ),
        (
            "production_generation",
            ("wired_memory_policy", "environment_overrides"),
            {"MLX_WIRED_LIMIT_GB": "96"},
        ),
        (
            "production_generation",
            ("wired_memory_policy", "iogpu_wired_limit_mb"),
            96_000,
        ),
        *[
            ("production_generation", (field,), True)
            for field in (
                "quality_claim",
                "speed_claim",
                "long_context_indexshare_proven",
                "same_machine_benchmark_proven",
                "full_vocabulary_eval_proven",
                "cold_residency_memory_clean",
                "generation_warmup_memory_clean",
            )
        ],
        *[
            ("production_generation", (field,), False)
            for field in (
                "input_fingerprint_reverified_before_bind",
                "input_fingerprint_reverified_after_residency",
                "input_fingerprint_reverified_after_warmup",
                "input_fingerprint_reverified_after_warmup_preparation",
                "input_fingerprint_reverified_before_generation",
                "input_fingerprint_reverified_after_generation",
            )
        ],
    ],
    ids=lambda value: str(value)[:80],
)
def test_schema_v2_raw_evidence_mutations_fail_closed(
    tmp_path: Path,
    trusted_contracts: None,
    target: str,
    path: tuple[str, ...],
    value: object,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    payload = kwargs[target]
    assert isinstance(payload, dict)
    _replace_nested(payload, path, value)

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_schema_v2_rejects_rehashed_mismatched_identity_alias(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _v2_kwargs(tmp_path)
    production = kwargs["production_generation"]
    assert isinstance(production, dict)
    alias = production["common_artifact_identity"]
    assert isinstance(alias, dict)
    alias["group_size"] = 256
    production["common_artifact_identity_sha256"] = canonical_sha256(alias)

    with pytest.raises(ValueError):
        gate.check_glm52_family_gate(**kwargs)


def test_gate_rejects_rehashed_teacher_metadata_body_tampering(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _kwargs(tmp_path)
    metadata = dict(kwargs["teacher_metadata"])
    metadata["source_payload_group_count"] = 224
    metadata.pop("metadata_contract_sha256")
    metadata["metadata_contract_sha256"] = canonical_sha256(metadata)
    kwargs["teacher_metadata"] = metadata

    with pytest.raises(ValueError, match="source payload group count"):
        gate.check_glm52_family_gate(**kwargs)


def test_cli_returns_two_and_writes_evidence_for_valid_incomplete_state(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    kwargs = _kwargs(tmp_path)
    paths: dict[str, Path] = {}
    for name, payload in kwargs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths[name] = path
    output = tmp_path / "gate.json"
    argv = [
        "--family-policy-json",
        str(paths["family_policy"]),
        "--eval-prompt-pack-json",
        str(paths["eval_prompt_pack"]),
        "--teacher-metadata-json",
        str(paths["teacher_metadata"]),
        "--full-bind-preflight-json",
        str(paths["full_bind_preflight"]),
        "--indexshare-runtime-json",
        str(paths["indexshare_runtime"]),
        "--synthetic-generation-json",
        str(paths["synthetic_generation"]),
        "--output-json",
        str(output),
    ]

    assert gate.main(argv) == 2
    assert json.loads(output.read_text())["family_gate_pass"] is False

    metadata_jsonl = Path(
        str(kwargs["teacher_metadata"]["metadata_jsonl"])
    )
    before = metadata_jsonl.read_bytes()
    alias_argv = list(argv)
    alias_argv[alias_argv.index("--output-json") + 1] = str(metadata_jsonl)
    assert gate.main(alias_argv) == 1
    assert metadata_jsonl.read_bytes() == before


def test_cli_cross_binds_raw_file_hashes_before_emitting_schema_v2(
    tmp_path: Path,
    trusted_contracts: None,
) -> None:
    paths, production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256="0" * 64,
    )

    assert gate.main(argv) == 1
    assert json.loads(output.read_text())["gate_status"] == (
        "glm52_family_gate_invalid"
    )

    input_hashes = production["input_evidence_file_sha256"]
    assert isinstance(input_hashes, dict)
    input_hashes["composite_audit_json"] = sha256_file(
        paths["composite_artifact_audit"]
    )
    paths["production_generation"].write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )

    assert gate.main(argv) == 2
    report = json.loads(output.read_text())
    assert report["gate_schema_version"] == 2
    assert report["missing_requirements"] == [
        "dequantized_source_teacher_cache_payload",
        "full_vocabulary_source_relative_family_eval",
        "route_math_diagnostics",
        "same_machine_pinned_fp4_benchmark",
    ]


def test_raw_teacher_cache_helper_calls_public_strict_auditor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    manifest = cache_root / "glm52-teacher-cache-fp32-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "producer": {"implementation": "test"},
                "source_evidence": gate.EXPECTED_TEACHER_CACHE_SOURCE_EVIDENCE,
                "non_vq_package": {"package": "test"},
                "audit_pass": True,
            }
        )
        + "\n"
    )
    prompt_pack = tmp_path / "prompts.json"
    prompt_pack.write_text("{}\n")
    contract = object()
    monkeypatch.setattr(
        gate.GLM52TeacherCacheContract,
        "from_frozen_prompt_pack",
        lambda *_args, **_kwargs: contract,
    )
    calls: list[tuple[Path, object]] = []

    def strict_audit(root: str | Path, *, contract: object):
        calls.append((Path(root), contract))
        raise ValueError("strict raw audit rejected cache")

    monkeypatch.setattr(gate, "audit_glm52_teacher_cache", strict_audit)

    with pytest.raises(ValueError, match="strict raw audit rejected cache"):
        gate._audit_raw_teacher_cache(
            cache_root=cache_root,
            manifest_path=manifest,
            prompt_pack_path=prompt_pack,
        )
    assert calls == [(cache_root, contract)]


def _write_cli_teacher_cache_manifest(
    cache_root: Path,
    *,
    non_vq_evidence_path: Path,
) -> Path:
    non_vq = json.loads(non_vq_evidence_path.read_text())
    manifest = cache_root / "glm52-teacher-cache-fp32-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "non_vq_package": {
                    "evidence_path": str(non_vq_evidence_path),
                    "evidence_file_sha256": sha256_file(non_vq_evidence_path),
                    "manifest_sha256": non_vq["manifest_sha256"],
                    "package_set_sha256": non_vq["package_set_sha256"],
                    "retained_tensor_count": non_vq["retained_tensor_count"],
                    "tensor_payload_bytes": non_vq["tensor_payload_bytes"],
                    "bound_package_identity": "d" * 64,
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


@pytest.mark.parametrize(
    ("audit", "expected_exit", "expected_status"),
    [
        (_clean_teacher_cache_audit(), 2, "glm52_family_gate_blocked"),
        (
            gate.GLM52TeacherCacheAudit(
                **{
                    **_clean_teacher_cache_audit().__dict__,
                    "release_eligible": False,
                }
            ),
            1,
            "glm52_family_gate_invalid",
        ),
    ],
    ids=("clean", "memory-dirty"),
)
def test_cli_teacher_cache_exit_semantics(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
    audit: gate.GLM52TeacherCacheAudit,
    expected_exit: int,
    expected_status: str,
) -> None:
    paths, production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256=None,
    )
    input_hashes = production["input_evidence_file_sha256"]
    assert isinstance(input_hashes, dict)
    input_hashes["composite_audit_json"] = sha256_file(
        paths["composite_artifact_audit"]
    )
    paths["production_generation"].write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    manifest = _write_cli_teacher_cache_manifest(
        cache_root,
        non_vq_evidence_path=paths["non_vq_evidence"],
    )
    argv[argv.index("--output-json"):argv.index("--output-json")] = [
        "--teacher-cache-root",
        str(cache_root),
        "--teacher-cache-manifest-json",
        str(manifest),
    ]
    monkeypatch.setattr(gate, "_audit_raw_teacher_cache", lambda **_kwargs: audit)

    assert gate.main(argv) == expected_exit
    report = json.loads(output.read_text())
    assert report["gate_status"] == expected_status
    if expected_exit == 2:
        assert report["gate_schema_version"] == GLM52_FAMILY_GATE_V3_SCHEMA_VERSION
        assert report["missing_requirements"] == list(
            GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS
        )


def test_cli_corrupt_teacher_cache_is_invalid(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256=None,
    )
    input_hashes = production["input_evidence_file_sha256"]
    assert isinstance(input_hashes, dict)
    input_hashes["composite_audit_json"] = sha256_file(
        paths["composite_artifact_audit"]
    )
    paths["production_generation"].write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    manifest = _write_cli_teacher_cache_manifest(
        cache_root,
        non_vq_evidence_path=paths["non_vq_evidence"],
    )
    argv[argv.index("--output-json"):argv.index("--output-json")] = [
        "--teacher-cache-root",
        str(cache_root),
        "--teacher-cache-manifest-json",
        str(manifest),
    ]

    def reject(**_kwargs: object) -> gate.GLM52TeacherCacheAudit:
        raise ValueError("shard digest mismatch")

    monkeypatch.setattr(gate, "_audit_raw_teacher_cache", reject)

    assert gate.main(argv) == 1
    report = json.loads(output.read_text())
    assert report["gate_status"] == "glm52_family_gate_invalid"
    assert report["input_error"]["message"] == "shard digest mismatch"


def test_cli_rejects_audited_cache_not_bound_to_raw_non_vq_evidence(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256=None,
    )
    input_hashes = production["input_evidence_file_sha256"]
    assert isinstance(input_hashes, dict)
    input_hashes["composite_audit_json"] = sha256_file(
        paths["composite_artifact_audit"]
    )
    paths["production_generation"].write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    manifest = _write_cli_teacher_cache_manifest(
        cache_root,
        non_vq_evidence_path=paths["non_vq_evidence"],
    )
    payload = json.loads(manifest.read_text())
    payload["non_vq_package"]["evidence_file_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    argv[argv.index("--output-json"):argv.index("--output-json")] = [
        "--teacher-cache-root",
        str(cache_root),
        "--teacher-cache-manifest-json",
        str(manifest),
    ]
    monkeypatch.setattr(
        gate,
        "_audit_raw_teacher_cache",
        lambda **_kwargs: _clean_teacher_cache_audit(),
    )

    assert gate.main(argv) == 1
    report = json.loads(output.read_text())
    assert report["gate_status"] == "glm52_family_gate_invalid"
    assert "non-VQ evidence_file_sha256" in report["input_error"]["message"]


@pytest.mark.parametrize(
    ("key", "needle", "replacement"),
    [
        (
            "record_type",
            '"record_type": "glm52_production_generation_probe",',
            '"record_type": "glm52_production_generation_probe",\n'
            '  "record_type": "glm52_production_generation_probe",',
        ),
        (
            "group_size",
            '"group_size": 512,',
            '"group_size": 512,\n    "group_size": 512,',
        ),
    ],
)
def test_cli_rejects_duplicate_authority_keys_at_any_object_depth(
    tmp_path: Path,
    trusted_contracts: None,
    key: str,
    needle: str,
    replacement: str,
) -> None:
    paths, _production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256=None,
    )
    authority = paths["production_generation"]
    raw = authority.read_text()
    assert raw.count(needle) >= 1
    authority.write_text(raw.replace(needle, replacement, 1))

    assert gate.main(argv) == 1
    report = json.loads(output.read_text())
    assert report["gate_status"] == "glm52_family_gate_invalid"
    assert report["input_error"] == {
        "type": "ValueError",
        "message": (
            "could not read production_generation: "
            f"duplicate JSON object key {key!r}"
        ),
    }


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_cli_rejects_non_standard_json_constants(
    tmp_path: Path,
    trusted_contracts: None,
    constant: str,
) -> None:
    paths, _production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256=None,
    )
    authority = paths["production_generation"]
    raw = authority.read_text()
    authority.write_text(
        raw.replace(
            "{\n",
            f'{{\n  "non_standard_constant": {constant},\n',
            1,
        )
    )

    assert gate.main(argv) == 1
    report = json.loads(output.read_text())
    assert report["gate_status"] == "glm52_family_gate_invalid"
    assert report["input_error"] == {
        "type": "ValueError",
        "message": (
            "could not read production_generation: "
            f"non-standard JSON constant {constant!r}"
        ),
    }


def test_cli_authenticates_the_same_single_read_used_for_json_validation(
    tmp_path: Path,
    trusted_contracts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _production, argv, output = _write_v2_cli_inputs(
        tmp_path,
        composite_audit_sha256=None,
    )
    authority_bytes = {
        path.resolve(): path.read_bytes() for path in paths.values()
    }
    read_counts = {path: 0 for path in authority_bytes}
    real_read_bytes = Path.read_bytes

    def read_authority_once(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in read_counts:
            read_counts[resolved] += 1
            if read_counts[resolved] > 1:
                raise AssertionError(f"CLI authority reopened: {resolved}")
        return real_read_bytes(path)

    real_sha256_file = gate.sha256_file

    def reject_authority_rehash(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved in read_counts:
            raise AssertionError(f"CLI authority rehashed separately: {resolved}")
        return real_sha256_file(path)

    monkeypatch.setattr(Path, "read_bytes", read_authority_once)
    monkeypatch.setattr(gate, "sha256_file", reject_authority_rehash)

    assert gate.main(argv) == 2
    assert set(read_counts.values()) == {1}
    report = json.loads(output.read_text())
    assert report["input_evidence_file_sha256"] == {
        name: hashlib.sha256(authority_bytes[path.resolve()]).hexdigest()
        for name, path in paths.items()
    }
