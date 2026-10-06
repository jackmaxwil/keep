from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def _load_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "audit_glm45_air_resident_bytes.py"
    spec = importlib.util.spec_from_file_location("audit_glm45_air_resident_bytes_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), tensors)


def test_resident_byte_audit_groups_source_and_artifact_surfaces(tmp_path: Path) -> None:
    audit = _load_audit_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_shard = source_dir / "model-00001-of-00001.safetensors"
    source_tensors = {
        "model.embed_tokens.weight": mx.array(np.zeros((4, 3), dtype=np.float16)),
        "lm_head.weight": mx.array(np.zeros((4, 3), dtype=np.float16)),
        "model.layers.0.self_attn.q_proj.weight": mx.array(np.zeros((3, 3), dtype=np.float16)),
        "model.layers.0.mlp.gate_proj.weight": mx.array(np.zeros((5, 3), dtype=np.float16)),
        "model.layers.1.mlp.gate.weight": mx.array(np.zeros((2, 3), dtype=np.float16)),
        "model.layers.1.mlp.experts.0.gate_proj.weight": mx.array(np.zeros((5, 3), dtype=np.float16)),
        "model.layers.2.self_attn.q_proj.weight": mx.array(np.zeros((3, 3), dtype=np.float16)),
    }
    _write_safetensors(source_shard, source_tensors)
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: source_shard.name for name in source_tensors},
            }
        )
    )
    config_path = source_dir / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 2}))

    artifact_dir = tmp_path / "artifact"
    _write_safetensors(
        artifact_dir / "layer-00001-gate_proj.safetensors",
        {
            "model.layers.1.mlp.switch_mlp.gate_proj.codes": mx.array(
                np.zeros((2, 5, 1), dtype=np.uint8)
            ),
            "model.layers.1.mlp.switch_mlp.gate_proj.scales": mx.array(
                np.zeros((2, 5, 1), dtype=np.float16)
            ),
        },
    )
    _write_safetensors(
        artifact_dir / "continuous_params" / "layer-00001-gate_proj.safetensors",
        {"output_bias": mx.array(np.zeros((2, 5), dtype=np.float16))},
    )

    record = audit.audit_resident_bytes(
        model_id="tiny-air",
        config_path=config_path,
        source_dir=source_dir,
        index_path=index_path,
        artifact_dir=artifact_dir,
    )

    assert record["record_type"] == "glm45_air_resident_byte_audit"
    assert record["source_non_expert_total_bytes"] == 108
    assert record["source_skipped_routed_expert_total_bytes"] == 30
    assert record["source_skipped_mtp_total_bytes"] == 18
    assert record["artifact_total_bytes"] == 50
    assert record["counted_resident_total_bytes"] == 158
    by_surface = {row["surface"]: row for row in record["source_non_expert_by_surface"]}
    assert by_surface["dense_mlp"]["bytes"] == 30
    assert by_surface["embed_tokens"]["bytes"] == 24
    assert by_surface["lm_head"]["bytes"] == 24
    assert by_surface["attention"]["bytes"] == 18
    assert by_surface["router_gates"]["bytes"] == 12
    by_artifact = {row["surface"]: row for row in record["artifact_by_surface"]}
    assert by_artifact["vq_gate_proj"]["bytes"] == 30
    assert by_artifact["continuous_sidecars"]["bytes"] == 20


def test_resident_byte_audit_records_policy_surfaces_without_filtering_runtime_source(
    tmp_path: Path,
) -> None:
    audit = _load_audit_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_shard = source_dir / "model.safetensors"
    source_tensors = {
        "model.embed_tokens.weight": mx.array(np.zeros((4, 3), dtype=np.float16)),
        "lm_head.weight": mx.array(np.zeros((4, 3), dtype=np.float16)),
        "model.layers.0.self_attn.q_proj.weight": mx.array(np.zeros((3, 3), dtype=np.float16)),
    }
    _write_safetensors(source_shard, source_tensors)
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {name: source_shard.name for name in source_tensors}})
    )
    config_path = source_dir / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 1}))
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "non_expert_precision": {
                    "enabled": True,
                    "surfaces": ["lm_head"],
                }
            }
        )
    )

    record = audit.audit_resident_bytes(
        model_id="tiny-air",
        config_path=config_path,
        source_dir=source_dir,
        index_path=index_path,
        artifact_dir=artifact_dir,
    )

    assert record["non_expert_precision_surfaces"] == ["lm_head"]
    assert record["source_non_expert_total_bytes"] == 66
    assert record["source_skipped_by_surface_total_bytes"] == 0
    assert record["source_skipped_by_surface"] == []
    by_surface = {row["surface"]: row for row in record["source_non_expert_by_surface"]}
    assert by_surface == {
        "attention": {"surface": "attention", "bytes": 18, "tensor_count": 1},
        "embed_tokens": {"surface": "embed_tokens", "bytes": 24, "tensor_count": 1},
        "lm_head": {"surface": "lm_head", "bytes": 24, "tensor_count": 1},
    }


def test_resident_byte_audit_compares_candidate_against_baseline() -> None:
    audit = _load_audit_module()
    candidate = {
        "record_type": "glm45_air_resident_byte_audit",
        "counted_resident_total_bytes": 850,
        "source_non_expert_total_bytes": 300,
        "artifact_total_bytes": 550,
    }
    baseline = {
        "record_type": "glm45_air_resident_byte_audit",
        "counted_resident_total_bytes": 1000,
        "source_non_expert_total_bytes": 420,
        "artifact_total_bytes": 580,
    }

    compared = audit.add_resident_byte_comparison(
        candidate,
        baseline_record=baseline,
        min_total_byte_reduction=100,
        max_counted_resident_total_bytes=900,
    )

    assert compared["baseline_counted_resident_total_bytes"] == 1000
    assert compared["resident_byte_delta_bytes"] == -150
    assert compared["resident_byte_reduction_bytes"] == 150
    assert compared["resident_byte_reduction_ratio"] == 0.15
    assert compared["source_non_expert_delta_bytes"] == -120
    assert compared["artifact_delta_bytes"] == -30
    assert compared["resident_byte_check"] == {
        "pass": True,
        "thresholds": {
            "max_counted_resident_total_bytes": 900,
            "min_total_byte_reduction": 100,
        },
        "failures": [],
    }


def test_resident_byte_audit_comparison_reports_threshold_failures() -> None:
    audit = _load_audit_module()
    candidate = {
        "record_type": "glm45_air_resident_byte_audit",
        "counted_resident_total_bytes": 970,
        "source_non_expert_total_bytes": 500,
        "artifact_total_bytes": 470,
    }
    baseline = {
        "record_type": "glm45_air_resident_byte_audit",
        "counted_resident_total_bytes": 1000,
        "source_non_expert_total_bytes": 510,
        "artifact_total_bytes": 490,
    }

    compared = audit.add_resident_byte_comparison(
        candidate,
        baseline_record=baseline,
        min_total_byte_reduction=100,
        max_counted_resident_total_bytes=900,
    )

    assert compared["resident_byte_reduction_bytes"] == 30
    assert compared["resident_byte_check"]["pass"] is False
    assert compared["resident_byte_check"]["failures"] == [
        "counted_resident_total_bytes 970 exceeds max 900",
        "resident_byte_reduction_bytes 30 is below min 100",
    ]
