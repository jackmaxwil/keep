from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "audit_glm45_air_single_host_teacher_source.py"
    )
    spec = importlib.util.spec_from_file_location("single_host_teacher_source_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_index(source_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 46,
                "n_routed_experts": 128,
                "hidden_size": 4096,
                "vocab_size": 151552,
            }
        ),
        encoding="utf-8",
    )


def test_single_host_teacher_source_audit_rejects_missing_shard(tmp_path: Path) -> None:
    cli = _load_cli()
    source_dir = tmp_path / "glm-air"
    source_dir.mkdir()
    _write_index(
        source_dir,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
        total_size=200,
    )
    (source_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 80)

    record = cli.build_audit_record(
        model_path=source_dir,
        revision="fixture",
        physical_memory_bytes=100,
        wired_limit_mb=0,
    )

    assert record["record_type"] == "glm45_air_single_host_teacher_source_audit"
    assert record["decision"] == "single_host_teacher_source_incomplete"
    assert record["source_ready_for_single_host_export"] is False
    assert record["expected_shard_count"] == 2
    assert record["present_shard_count"] == 1
    assert record["missing_shard_count"] == 1
    assert record["missing_shards"] == ["model-00002-of-00002.safetensors"]
    assert record["present_shard_bytes"] == 80
    assert record["metadata_total_size"] == 200
    assert record["memory_observation"]["source_bytes_exceed_physical_memory"] is True
    assert record["peer2_used"] is False
    assert record["rdma_jaccl_touched"] is False
    assert record["cache_rows_generated"] is False


def test_single_host_teacher_source_audit_accepts_runtime_complete_extra_missing_shard(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    source_dir = tmp_path / "glm-air"
    source_dir.mkdir()
    _write_index(
        source_dir,
        {
            "model.embed_tokens.weight": "model-00001-of-00047.safetensors",
            "model.layers.45.mlp.down_proj.weight": (
                "model-00046-of-00047.safetensors"
            ),
            "model.layers.46.mlp.down_proj.weight": (
                "model-00047-of-00047.safetensors"
            ),
            "lm_head.weight": "model-00046-of-00047.safetensors",
        },
        total_size=384,
    )
    (source_dir / "model-00001-of-00047.safetensors").write_bytes(b"x" * 64)
    (source_dir / "model-00046-of-00047.safetensors").write_bytes(b"y" * 128)

    record = cli.build_audit_record(
        model_path=source_dir,
        revision="fixture",
        physical_memory_bytes=256,
        wired_limit_mb=0,
    )

    assert record["decision"] == (
        "single_host_teacher_runtime_source_complete_full_source_incomplete"
    )
    assert record["complete_source"] is False
    assert record["complete_runtime_source"] is True
    assert record["source_ready_for_single_host_export"] is True
    assert record["expected_shard_count"] == 3
    assert record["missing_shards"] == ["model-00047-of-00047.safetensors"]
    assert record["required_runtime_shard_count"] == 2
    assert record["missing_required_runtime_shard_count"] == 0
    assert record["missing_required_runtime_shards"] == []
    assert record["missing_extra_shards"] == ["model-00047-of-00047.safetensors"]
    assert record["missing_required_runtime_tensor_count"] == 0
    assert record["missing_extra_tensor_count"] == 1
    assert record["num_hidden_layers"] == 46


def test_single_host_teacher_source_audit_accepts_complete_snapshot(tmp_path: Path) -> None:
    cli = _load_cli()
    source_dir = tmp_path / "glm-air"
    source_dir.mkdir()
    _write_index(
        source_dir,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
        total_size=128,
    )
    (source_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    (source_dir / "model-00002-of-00002.safetensors").write_bytes(b"y" * 64)

    record = cli.build_audit_record(
        model_path=source_dir,
        revision="fixture",
        physical_memory_bytes=256,
        wired_limit_mb=0,
    )

    assert record["decision"] == "single_host_teacher_source_complete"
    assert record["source_ready_for_single_host_export"] is True
    assert record["expected_shard_count"] == 2
    assert record["present_shard_count"] == 2
    assert record["missing_shards"] == []
    assert record["present_shard_bytes"] == 128
    assert record["memory_observation"]["source_bytes_exceed_physical_memory"] is False
    assert record["config"]["torch_dtype"] == "bfloat16"


def test_single_host_teacher_source_audit_scans_alternate_complete_snapshot(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    pinned_dir = tmp_path / "snapshots" / "pinned"
    pinned_dir.mkdir(parents=True)
    _write_index(
        pinned_dir,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
        total_size=128,
    )
    (pinned_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)

    alternate_dir = tmp_path / "snapshots" / "alternate"
    alternate_dir.mkdir()
    _write_index(
        alternate_dir,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
        total_size=128,
    )
    (alternate_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    (alternate_dir / "model-00002-of-00002.safetensors").write_bytes(b"y" * 64)

    record = cli.build_audit_record(
        model_path=pinned_dir,
        revision="pinned",
        physical_memory_bytes=256,
        wired_limit_mb=0,
        scan_roots=(tmp_path / "snapshots",),
    )

    assert record["decision"] == "single_host_teacher_source_incomplete"
    assert record["source_ready_for_single_host_export"] is False
    assert record["local_export_candidate_ready"] is True
    assert record["recommended_model_path"] == str(alternate_dir)
    assert record["candidate_scan"]["scanned_index_count"] == 2
    assert record["candidate_scan"]["complete_candidate_count"] == 1
    assert record["candidate_scan"]["incomplete_candidate_count"] == 1
    assert {
        candidate["decision"] for candidate in record["candidate_scan"]["candidates"]
    } == {
        "single_host_teacher_source_complete",
        "single_host_teacher_source_incomplete",
    }
