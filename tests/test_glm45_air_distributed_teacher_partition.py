from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def _load_partition_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "plan_glm45_air_distributed_teacher_partition.py"
    )
    spec = importlib.util.spec_from_file_location("teacher_partition_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_index(root: Path, weight_map: dict[str, str]) -> Path:
    index_path = root / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return index_path


def test_partition_plan_counts_present_tensor_bytes_and_recommends_split(tmp_path) -> None:
    cli = _load_partition_cli()
    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": mx.array(np.zeros((2, 4), dtype=np.float16)),
        "model.layers.0.self_attn.q_proj.weight": mx.array(np.zeros((3, 4), dtype=np.float16)),
        "model.layers.1.self_attn.q_proj.weight": mx.array(np.zeros((4, 4), dtype=np.float16)),
        "model.layers.2.self_attn.q_proj.weight": mx.array(np.zeros((5, 4), dtype=np.float16)),
        "model.norm.weight": mx.array(np.zeros((4,), dtype=np.float16)),
        "lm_head.weight": mx.array(np.zeros((4, 2), dtype=np.float16)),
    }
    mx.save_safetensors(str(tmp_path / shard_name), tensors)
    index_path = _write_index(tmp_path, {name: shard_name for name in tensors})

    plan = cli.build_partition_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=100,
    )

    assert plan["complete_source"] is True
    assert plan["complete_runtime_source"] is True
    assert plan["num_hidden_layers"] == 3
    assert plan["present_tensor_count"] == len(tensors)
    assert plan["present_weight_bytes"] == 136
    assert plan["recommended_split_layer"] == 2
    assert plan["rank_plans"][0]["layers"] == [0, 1]
    assert plan["rank_plans"][0]["present_weight_bytes"] == 72
    assert plan["rank_plans"][1]["layers"] == [2]
    assert plan["rank_plans"][1]["present_weight_bytes"] == 64
    assert plan["max_rank_present_weight_bytes"] == 72
    assert plan["existing_exporter_compatible"] is False


def test_partition_plan_reports_missing_shards_as_incomplete(tmp_path) -> None:
    cli = _load_partition_cli()
    shard_name = "model-00001-of-00002.safetensors"
    missing_shard_name = "model-00002-of-00002.safetensors"
    tensors = {
        "model.embed_tokens.weight": mx.array(np.zeros((2, 4), dtype=np.float16)),
        "model.layers.0.self_attn.q_proj.weight": mx.array(np.zeros((3, 4), dtype=np.float16)),
    }
    mx.save_safetensors(str(tmp_path / shard_name), tensors)
    weight_map = {name: shard_name for name in tensors}
    weight_map["model.layers.1.self_attn.q_proj.weight"] = missing_shard_name
    index_path = _write_index(tmp_path, weight_map)

    plan = cli.build_partition_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=100,
    )

    assert plan["complete_source"] is False
    assert plan["complete_runtime_source"] is False
    assert plan["missing_shards"] == [missing_shard_name]
    assert plan["missing_required_shards"] == [missing_shard_name]
    assert plan["missing_tensor_count"] == 1
    assert plan["missing_required_tensor_count"] == 1
    assert plan["present_tensor_count"] == 2


def test_partition_plan_treats_layers_beyond_config_as_extra(tmp_path) -> None:
    cli = _load_partition_cli()
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    shard_name = "model-00001-of-00002.safetensors"
    missing_shard_name = "model-00002-of-00002.safetensors"
    tensors = {
        "model.embed_tokens.weight": mx.array(np.zeros((2, 4), dtype=np.float16)),
        "model.layers.0.self_attn.q_proj.weight": mx.array(np.zeros((3, 4), dtype=np.float16)),
    }
    mx.save_safetensors(str(tmp_path / shard_name), tensors)
    weight_map = {name: shard_name for name in tensors}
    weight_map["model.layers.1.self_attn.q_proj.weight"] = missing_shard_name
    index_path = _write_index(tmp_path, weight_map)

    plan = cli.build_partition_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=100,
    )

    assert plan["num_hidden_layers"] == 1
    assert plan["complete_source"] is False
    assert plan["complete_runtime_source"] is True
    assert plan["missing_shards"] == [missing_shard_name]
    assert plan["missing_required_shards"] == []
    assert plan["missing_tensor_count"] == 1
    assert plan["missing_required_tensor_count"] == 0
    assert plan["layer_count_present"] == 1


def test_partition_cli_rejects_non_positive_rank_budget_before_work(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/plan_glm45_air_distributed_teacher_partition.py",
            "--rank-budget-gb",
            "0",
            "--source-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--rank-budget-gb must be positive" in completed.stderr
