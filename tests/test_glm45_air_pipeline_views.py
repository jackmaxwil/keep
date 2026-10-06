from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def _load_views_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "materialize_glm45_air_pipeline_views.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_views_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_probe_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "probe_glm45_air_pipeline_view_load.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_view_probe_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config(*, num_hidden_layers: int = 4) -> dict:
    return {
        "model_type": "glm4_moe",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "moe_intermediate_size": 8,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "n_group": 1,
        "head_dim": 4,
        "topk_group": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 3,
        "routed_scaling_factor": 1.0,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 1,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "use_qk_norm": False,
        "tie_word_embeddings": False,
        "attention_bias": True,
        "partial_rotary_factor": 1.0,
    }


def _write_synthetic_source(root: Path, *, num_hidden_layers: int = 4) -> Path:
    cli = _load_views_cli()
    config = _tiny_config(num_hidden_layers=num_hidden_layers)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    parameter_keys: list[str] = []
    for rank in range(2):
        parameter_keys.extend(cli._pipeline_parameter_keys(config, rank=rank, pipeline_size=2))

    source_names: list[str] = []
    for parameter_key in parameter_keys:
        source_names.extend(
            cli.source_tensors_for_parameter(
                parameter_key,
                weight_map={},
                n_routed_experts=config["n_routed_experts"],
            )
        )
    source_names = sorted(set(source_names))

    shard_by_name: dict[str, str] = {}
    tensors_by_shard: dict[str, dict[str, mx.array]] = {}
    for name in source_names:
        layer = cli._tensor_layer(name)
        if layer is None:
            shard = (
                "model-00001-of-00003.safetensors"
                if name == "model.embed_tokens.weight"
                else "model-00003-of-00003.safetensors"
            )
        elif layer < 2:
            shard = "model-00001-of-00003.safetensors"
        else:
            shard = "model-00002-of-00003.safetensors"
        shard_by_name[name] = shard
        tensors_by_shard.setdefault(shard, {})[name] = mx.array(
            np.zeros((1,), dtype=np.float16)
        )

    for shard, tensors in tensors_by_shard.items():
        mx.save_safetensors(str(root / shard), tensors)

    index_path = root / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": shard_by_name}),
        encoding="utf-8",
    )
    return index_path


def test_switch_mlp_parameter_expands_to_raw_expert_tensors() -> None:
    cli = _load_views_cli()

    expanded = cli.source_tensors_for_parameter(
        "model.layers.2.mlp.switch_mlp.gate_proj.weight",
        weight_map={},
        n_routed_experts=3,
    )

    assert expanded == [
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.2.mlp.experts.1.gate_proj.weight",
        "model.layers.2.mlp.experts.2.gate_proj.weight",
    ]


def test_pipeline_view_plan_matches_upstream_rank_semantics(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
    )

    assert plan["complete_required_source"] is True
    assert plan["pipeline_size"] == 2
    rank0 = plan["rank_plans"][0]
    rank1 = plan["rank_plans"][1]
    assert rank0["layers"] == [2, 3]
    assert rank1["layers"] == [0, 1]
    for rank_plan in [rank0, rank1]:
        assert "model.embed_tokens.weight" in rank_plan["parameter_keys"]
        assert "model.norm.weight" in rank_plan["parameter_keys"]
        assert "lm_head.weight" in rank_plan["parameter_keys"]
        assert rank_plan["fits_required_tensor_budget"] is True
        assert rank_plan["fits_visible_shard_budget"] is True

    assert (
        "model.layers.2.mlp.switch_mlp.gate_proj.weight"
        in rank0["parameter_weight_map"]
    )
    assert (
        "model.layers.2.mlp.experts.0.gate_proj.weight"
        in rank0["required_source_tensors"]
    )
    assert (
        "model.layers.1.mlp.experts.0.gate_proj.weight"
        in rank1["required_source_tensors"]
    )


def test_pipeline_view_plan_supports_custom_two_rank_layer_split(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=3,
    )

    assert plan["layer_split"] == 3
    rank0 = plan["rank_plans"][0]
    rank1 = plan["rank_plans"][1]
    assert rank0["layer_split"] == 3
    assert rank1["layer_split"] == 3
    assert rank0["layers"] == [3]
    assert rank1["layers"] == [0, 1, 2]
    assert "model.layers.3.input_layernorm.weight" in rank0["parameter_keys"]
    assert "model.layers.2.input_layernorm.weight" in rank1["parameter_keys"]
    assert "model.layers.2.input_layernorm.weight" not in rank0["parameter_keys"]
    assert "model.layers.3.input_layernorm.weight" not in rank1["parameter_keys"]
    for rank_plan in [rank0, rank1]:
        assert "model.embed_tokens.weight" in rank_plan["parameter_keys"]
        assert "model.norm.weight" in rank_plan["parameter_keys"]
        assert "lm_head.weight" in rank_plan["parameter_keys"]
        assert rank_plan["fits_required_tensor_budget"] is True


def test_pipeline_view_plan_can_prune_non_layer_params_for_rank0_only_logits(
    tmp_path,
) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=2,
        rank0_only_logits_views=True,
    )

    rank0 = plan["rank_plans"][0]
    rank1 = plan["rank_plans"][1]
    assert plan["rank0_only_logits_views"] is True
    assert "model.embed_tokens.weight" not in rank0["parameter_keys"]
    assert "model.norm.weight" in rank0["parameter_keys"]
    assert "lm_head.weight" in rank0["parameter_keys"]
    assert "model.embed_tokens.weight" in rank1["parameter_keys"]
    assert "model.norm.weight" not in rank1["parameter_keys"]
    assert "lm_head.weight" not in rank1["parameter_keys"]


def test_pipeline_view_plan_recommends_low_residency_rank0_only_split(
    tmp_path,
) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=1,
        rank0_only_logits_views=True,
    )

    recommendation = plan["layer_split_recommendation"]
    assert recommendation["recommended_layer_split"] == 2
    assert recommendation["optimization_target"] == (
        "minimize_max_required_present_tensor_bytes_then_visible_shard_file_bytes"
    )
    candidates = {
        candidate["layer_split"]: candidate for candidate in recommendation["candidates"]
    }
    assert candidates[2]["max_required_present_tensor_bytes"] == 96
    assert candidates[2]["required_present_tensor_bytes_by_rank"] == [96, 72]
    assert candidates[1]["max_required_present_tensor_bytes"] > candidates[2][
        "max_required_present_tensor_bytes"
    ]
    assert candidates[3]["max_required_present_tensor_bytes"] > candidates[2][
        "max_required_present_tensor_bytes"
    ]


def test_pipeline_view_plan_can_prune_rank0_for_route_trace_stop_layer(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=2,
        rank0_stop_after_layer=2,
        rank0_route_trace_only=True,
    )

    rank0 = plan["rank_plans"][0]
    rank1 = plan["rank_plans"][1]
    assert plan["rank0_stop_after_layer"] == 2
    assert plan["rank0_route_trace_only"] is True
    assert rank0["layers"] == [2]
    assert rank0["last_layer"] == 2
    assert rank0["rank0_stop_after_layer"] == 2
    assert rank0["route_trace_only"] is True
    assert "model.layers.2.input_layernorm.weight" in rank0["parameter_keys"]
    assert "model.layers.3.input_layernorm.weight" not in rank0["parameter_keys"]
    assert "model.embed_tokens.weight" not in rank0["parameter_keys"]
    assert "model.norm.weight" not in rank0["parameter_keys"]
    assert "lm_head.weight" not in rank0["parameter_keys"]
    assert not any(".layers.3." in name for name in rank0["required_source_tensors"])
    assert rank1["layers"] == [0, 1]
    assert "model.embed_tokens.weight" in rank1["parameter_keys"]


def test_pipeline_view_plan_can_prune_rank0_stop_layer_after_route_gate(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=2,
        rank0_stop_after_layer=2,
        rank0_route_trace_only=True,
        rank0_stop_after_route_gate=True,
    )

    rank0 = plan["rank_plans"][0]
    assert plan["rank0_stop_after_route_gate"] is True
    assert rank0["rank0_stop_after_route_gate"] is True
    assert "model.layers.2.mlp.gate.weight" in rank0["parameter_keys"]
    assert "model.layers.2.mlp.gate.e_score_correction_bias" in rank0["parameter_keys"]
    assert "model.layers.2.mlp.switch_mlp.gate_proj.weight" not in rank0["parameter_keys"]
    assert "model.layers.2.mlp.shared_experts.gate_proj.weight" not in rank0["parameter_keys"]
    assert "model.layers.2.mlp.experts.0.gate_proj.weight" not in rank0["required_source_tensors"]
    assert "model.layers.3.input_layernorm.weight" not in rank0["parameter_keys"]


def test_pipeline_view_plan_can_prune_all_route_trace_only_views(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=2,
        rank0_stop_after_layer=2,
        route_trace_only_views=True,
    )

    rank0 = plan["rank_plans"][0]
    rank1 = plan["rank_plans"][1]
    assert plan["route_trace_only_views"] is True
    assert rank0["route_trace_only"] is True
    assert rank1["route_trace_only"] is True
    assert "model.embed_tokens.weight" not in rank0["parameter_keys"]
    assert "model.norm.weight" not in rank0["parameter_keys"]
    assert "lm_head.weight" not in rank0["parameter_keys"]
    assert "model.embed_tokens.weight" in rank1["parameter_keys"]
    assert "model.norm.weight" not in rank1["parameter_keys"]
    assert "lm_head.weight" not in rank1["parameter_keys"]


def test_materialize_writes_rank_local_index_and_symlinks(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)
    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
    )

    out_dir = tmp_path / "views"
    materialized = cli.materialize_pipeline_views(plan, output_dir=out_dir)

    assert materialized == [out_dir / "rank-0", out_dir / "rank-1"]
    rank0_index = json.loads(
        (out_dir / "rank-0" / "model.safetensors.index.json").read_text()
    )
    assert rank0_index["metadata"]["format"] == "glm45_air_pipeline_rank_view"
    assert (
        "model.layers.2.mlp.switch_mlp.gate_proj.weight"
        in rank0_index["weight_map"]
    )
    assert (out_dir / "rank-0" / "config.json").is_symlink()
    assert (out_dir / "rank-0" / "model-00002-of-00003.safetensors").is_symlink()
    assert (out_dir / "rank-1" / "model-00001-of-00003.safetensors").is_symlink()
    assert not (out_dir / "rank-1" / "model-00002-of-00003.safetensors").exists()
    assert (out_dir / "rank-1" / "model-00003-of-00003.safetensors").is_symlink()


def test_materialize_records_custom_layer_split_metadata(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)
    plan = cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=3,
    )

    out_dir = tmp_path / "views"
    cli.materialize_pipeline_views(plan, output_dir=out_dir)

    rank0_index = json.loads(
        (out_dir / "rank-0" / "model.safetensors.index.json").read_text()
    )
    rank1_plan = json.loads((out_dir / "rank-1" / "pipeline_view_plan.json").read_text())
    assert rank0_index["metadata"]["layer_split"] == 3
    assert rank1_plan["layer_split"] == 3
    assert rank1_plan["layers"] == [0, 1, 2]


def test_local_sequential_stage_view_plan_prunes_worker_source_surface(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)

    plan = cli.build_local_sequential_stage_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=2,
        lower_split_layer=1,
        upper_split_layer=3,
        head_process=True,
    )

    assert plan["format"] == "glm45_air_local_sequential_stage_views"
    stage_plans = {stage_plan["stage"]: stage_plan for stage_plan in plan["stage_plans"]}
    assert list(stage_plans) == [
        "lower-pre",
        "lower-final",
        "upper-pre",
        "upper-final",
        "head",
    ]
    assert stage_plans["lower-pre"]["rank"] == 1
    assert stage_plans["lower-pre"]["layers"] == [0]
    assert "model.embed_tokens.weight" in stage_plans["lower-pre"]["parameter_keys"]
    assert "model.norm.weight" not in stage_plans["lower-pre"]["parameter_keys"]
    assert "lm_head.weight" not in stage_plans["lower-pre"]["parameter_keys"]

    assert stage_plans["lower-final"]["rank"] == 1
    assert stage_plans["lower-final"]["layers"] == [1]
    assert "model.embed_tokens.weight" not in stage_plans["lower-final"]["parameter_keys"]
    assert "model.norm.weight" not in stage_plans["lower-final"]["parameter_keys"]
    assert "lm_head.weight" not in stage_plans["lower-final"]["parameter_keys"]

    assert stage_plans["upper-pre"]["rank"] == 0
    assert stage_plans["upper-pre"]["layers"] == [2]
    assert "model.embed_tokens.weight" not in stage_plans["upper-pre"]["parameter_keys"]
    assert "model.norm.weight" not in stage_plans["upper-pre"]["parameter_keys"]
    assert "lm_head.weight" not in stage_plans["upper-pre"]["parameter_keys"]

    assert stage_plans["upper-final"]["rank"] == 0
    assert stage_plans["upper-final"]["layers"] == [3]
    assert "model.embed_tokens.weight" not in stage_plans["upper-final"]["parameter_keys"]
    assert "model.norm.weight" in stage_plans["upper-final"]["parameter_keys"]
    assert "lm_head.weight" not in stage_plans["upper-final"]["parameter_keys"]

    assert stage_plans["head"]["rank"] == 0
    assert stage_plans["head"]["layers"] == []
    assert "model.embed_tokens.weight" not in stage_plans["head"]["parameter_keys"]
    assert "model.norm.weight" not in stage_plans["head"]["parameter_keys"]
    assert "lm_head.weight" in stage_plans["head"]["parameter_keys"]


def test_local_sequential_stage_view_plan_accepts_multi_window_splits(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path, num_hidden_layers=6)

    plan = cli.build_local_sequential_stage_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=3,
        lower_split_layers=[1, 2],
        upper_split_layers=[4, 5],
        head_process=True,
    )

    stage_plans = {stage_plan["stage"]: stage_plan for stage_plan in plan["stage_plans"]}
    assert list(stage_plans) == [
        "lower-embed",
        "lower-window-0",
        "lower-window-1",
        "lower-window-2",
        "upper-window-0",
        "upper-window-1",
        "upper-window-2",
        "head",
    ]
    assert stage_plans["lower-embed"]["layers"] == []
    assert "model.embed_tokens.weight" in stage_plans["lower-embed"]["parameter_keys"]
    assert stage_plans["lower-window-0"]["layers"] == [0]
    assert stage_plans["lower-window-1"]["layers"] == [1]
    assert stage_plans["lower-window-2"]["layers"] == [2]
    assert stage_plans["upper-window-0"]["layers"] == [3]
    assert stage_plans["upper-window-1"]["layers"] == [4]
    assert stage_plans["upper-window-2"]["layers"] == [5]
    assert "model.embed_tokens.weight" not in stage_plans["lower-window-0"]["parameter_keys"]
    assert "model.embed_tokens.weight" not in stage_plans["lower-window-1"]["parameter_keys"]
    assert "model.norm.weight" not in stage_plans["upper-window-1"]["parameter_keys"]
    assert "model.norm.weight" in stage_plans["upper-window-2"]["parameter_keys"]
    assert "lm_head.weight" not in stage_plans["upper-window-2"]["parameter_keys"]
    assert "lm_head.weight" in stage_plans["head"]["parameter_keys"]

    out_dir = tmp_path / "multi-stage-views"
    materialized = cli.materialize_local_sequential_stage_views(plan, output_dir=out_dir)
    assert materialized["stage_view_roots"]["lower-embed"] == {
        "1": str(out_dir / "lower-embed")
    }
    assert materialized["stage_view_roots"]["lower-window-2"] == {
        "1": str(out_dir / "lower-window-2")
    }
    assert materialized["stage_view_roots"]["upper-window-2"] == {
        "0": str(out_dir / "upper-window-2")
    }


def test_materialize_local_sequential_stage_views_writes_stage_roots(tmp_path) -> None:
    cli = _load_views_cli()
    index_path = _write_synthetic_source(tmp_path)
    plan = cli.build_local_sequential_stage_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
        layer_split=2,
        lower_split_layer=1,
        upper_split_layer=3,
        head_process=True,
    )

    out_dir = tmp_path / "stage-views"
    materialized = cli.materialize_local_sequential_stage_views(plan, output_dir=out_dir)

    assert materialized["stage_view_roots"] == {
        "lower-pre": {"1": str(out_dir / "lower-pre")},
        "lower-final": {"1": str(out_dir / "lower-final")},
        "upper-pre": {"0": str(out_dir / "upper-pre")},
        "upper-final": {"0": str(out_dir / "upper-final")},
        "head": {"0": str(out_dir / "head")},
    }
    lower_final_index = json.loads(
        (out_dir / "lower-final" / "model.safetensors.index.json").read_text()
    )
    assert lower_final_index["metadata"]["format"] == "glm45_air_local_sequential_stage_view"
    assert lower_final_index["metadata"]["stage"] == "lower-final"
    assert lower_final_index["metadata"]["rank"] == 1
    assert "model.embed_tokens.weight" not in lower_final_index["weight_map"]
    assert "model.layers.1.input_layernorm.weight" in lower_final_index["weight_map"]
    assert (out_dir / "head" / "model-00003-of-00003.safetensors").is_symlink()


def test_pipeline_view_cli_rejects_non_positive_rank_budget(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/materialize_glm45_air_pipeline_views.py",
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


def test_pipeline_view_lazy_load_probe_matches_materialized_plan(tmp_path) -> None:
    probe = _load_probe_cli()
    index_path = _write_synthetic_source(tmp_path)

    result = probe.probe_pipeline_views(
        source_dir=tmp_path,
        index_path=index_path,
        output_dir=tmp_path / "probe-views",
        rank_budget_bytes=1_000_000,
        overwrite=True,
    )

    assert result["complete_required_source"] is True
    assert result["all_ranks_match_plan"] is True
    assert result["did_eval_weights"] is False
    assert len(result["rank_results"]) == 2
    rank0, rank1 = result["rank_results"]
    assert rank0["first_layer"] == 2
    assert rank0["last_layer"] == 3
    assert rank1["first_layer"] == 0
    assert rank1["last_layer"] == 1
    for rank_result in result["rank_results"]:
        assert rank_result["did_eval_weights"] is False
        assert rank_result["parameter_keys_match_plan"] is True
        assert rank_result["missing_after_key_count"] == 0
        assert rank_result["unexpected_after_key_count"] == 0
        assert rank_result["has_embed"] is True
        assert rank_result["has_norm"] is True
        assert rank_result["has_lm_head"] is True


def test_pipeline_view_probe_cli_rejects_reuse_with_overwrite(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/probe_glm45_air_pipeline_view_load.py",
            "--source-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "views"),
            "--reuse-existing",
            "--overwrite",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--overwrite cannot be used with --reuse-existing" in completed.stderr


def test_pipeline_view_probe_cli_writes_output_json(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index_path = _write_synthetic_source(tmp_path)
    output_json = tmp_path / "probe.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/probe_glm45_air_pipeline_view_load.py",
            "--source-dir",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--output-dir",
            str(tmp_path / "views"),
            "--output-json",
            str(output_json),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["all_ranks_match_plan"] is True
    assert payload["did_eval_weights"] is False
