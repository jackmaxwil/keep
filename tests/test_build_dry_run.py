from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from mlx_vq.build.recipe import RecipeError, load_recipe
from mlx_vq.build.runner import plan_recipe, render_dry_run


def _write_tiny_lineage(tmp_path: Path) -> Path:
    seed = tmp_path / "seed-artifact"
    seed.mkdir()
    (seed / "conversion-manifest.json").write_text(json.dumps({"schema": 1}))
    (seed / "layer-00045-gate_proj.safetensors").write_bytes(b"seed")
    for split in ("report", "select"):
        cache = tmp_path / f"cache-{split}"
        cache.mkdir()
        (cache / "metadata.jsonl").write_text(
            json.dumps({"prompt_id": f"{split}_math_001"}) + "\n"
        )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "tiny-lineage",
                "description": "fixture",
                "external_inputs": {
                    "seed_artifact": {"kind": "artifact_dir", "path": str(seed)},
                    "teacher_report": {
                        "kind": "teacher_cache",
                        "path": str(tmp_path / "cache-report" / "metadata.jsonl"),
                    },
                    "teacher_selection": {
                        "kind": "teacher_cache",
                        "path": str(tmp_path / "cache-select" / "metadata.jsonl"),
                    },
                },
                "steps": [
                    {
                        "id": "train",
                        "op": "train-low-rank",
                        "class": "promotable",
                        "inputs": {
                            "seed_artifact": "external:seed_artifact",
                            "selection_teacher": "external:teacher_selection",
                            "validation_teacher": "external:teacher_report",
                        },
                        "params": {
                            "layer": 45,
                            "projections": ["gate_proj", "up_proj", "down_proj"],
                            "trainable": "low_rank_residual",
                            "low_rank": 4,
                            "steps": 12,
                            "learning_rate": 0.5,
                            "train_row_indices": [44, 45],
                        },
                        "gate": {"profile": "train_sane"},
                    },
                    {
                        "id": "eval_report",
                        "op": "eval",
                        "class": "verify",
                        "inputs": {
                            "artifact": "step:train/artifact",
                            "teacher": "external:teacher_report",
                        },
                        "params": {"engine": "vq_e1_routed_nax_e8p", "max_rows": 128},
                        "gate": {"profile": "balanced_rc_split"},
                    },
                ],
                "promotion": {
                    "artifact_step": "train",
                    "gates": [
                        {"evidence_step": "eval_report", "profile": "community_wow"}
                    ],
                },
            }
        )
    )
    return recipe_path


def test_dry_run_resolves_argv_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_spawn(*args, **kwargs):
        raise AssertionError("dry-run must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "run", _no_spawn)
    monkeypatch.setattr(subprocess, "Popen", _no_spawn)

    recipe = load_recipe(_write_tiny_lineage(tmp_path))
    plan = plan_recipe(recipe, build_root=tmp_path / "build")

    train = plan.step("train")
    argv = train.argv
    assert argv[:3] == ["uv", "run", "python"]
    assert argv[3] == "benchmarks/finetune_glm45_air_vq_continuous.py"
    assert "--seed-artifact-dir" in argv
    assert "--allow-existing" not in argv
    assert "--overwrite" not in argv
    projections_at = argv.index("--projections")
    assert argv[projections_at + 1 : projections_at + 4] == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    row_at = argv.index("--train-row-indices")
    assert argv[row_at + 1] == "44,45"
    cache_root_at = argv.index("--selection-cache-root")
    assert argv[cache_root_at + 1].endswith("cache-select")

    eval_step = plan.step("eval_report")
    assert eval_step.argv[3] == "benchmarks/eval_glm45_air_teacher_cache.py"
    artifact_at = eval_step.argv.index("--artifact-dir")
    assert eval_step.argv[artifact_at + 1] == str(train.out_dir)
    append_at = eval_step.argv.index("--append-jsonl")
    assert eval_step.argv[append_at + 1].startswith(str(eval_step.step_dir))

    rendered = render_dry_run(plan)
    assert "step train [promotable]" in rendered
    assert "promotion: artifact_step=train" in rendered


def test_plan_is_deterministic_and_key_sensitive(tmp_path: Path) -> None:
    recipe_path = _write_tiny_lineage(tmp_path)
    recipe = load_recipe(recipe_path)
    first = plan_recipe(recipe, build_root=tmp_path / "build")
    second = plan_recipe(recipe, build_root=tmp_path / "build")
    assert [step.step_key for step in first.steps] == [
        step.step_key for step in second.steps
    ]

    seed_manifest = Path(recipe.external_inputs["seed_artifact"].path) / "conversion-manifest.json"
    seed_manifest.write_text(json.dumps({"schema": 2}))
    third = plan_recipe(recipe, build_root=tmp_path / "build")
    # Seed identity change propagates through the whole chain.
    assert third.step("train").step_key != first.step("train").step_key
    assert third.step("eval_report").step_key != first.step("eval_report").step_key


def test_no_hash_dry_run_skips_external_hashing(tmp_path: Path) -> None:
    recipe = load_recipe(_write_tiny_lineage(tmp_path))
    seed_dir = Path(recipe.external_inputs["seed_artifact"].path)
    for path in sorted(seed_dir.rglob("*"), reverse=True):
        path.unlink()
    seed_dir.rmdir()
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    assert all(value == "<unhashed>" for value in plan.external_hashes.values())
    assert plan.step("train").argv


def test_missing_track_a_ops_render_deterministic_argv(tmp_path: Path) -> None:
    recipe_path = _write_tiny_lineage(tmp_path)
    disagreement = tmp_path / "route-disagreement.json"
    disagreement.write_text(json.dumps({"record_type": "air_route_trace_disagreement_summary"}))
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["route_disagreement"] = {
        "kind": "file",
        "path": str(disagreement),
    }
    raw["steps"].extend(
        [
            {
                "id": "bump_non_expert",
                "op": "bump-non-expert-precision",
                "class": "promotable",
                "inputs": {"seed_artifact": "step:train/artifact"},
                "params": {
                    "surfaces": ["embed_tokens", "lm_head"],
                    "dtype": "bf16",
                    "reason": "top1 gap attributed to non-expert surfaces",
                },
                "gate": {"profile": "train_sane"},
            },
            {
                "id": "router_kd",
                "op": "train-router-kd",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:bump_non_expert/artifact",
                    "disagreement": "external:route_disagreement",
                },
                "params": {"scale": 1, "max_abs_delta": 0.125, "num_experts": 128},
                "gate": {"profile": "train_sane"},
            },
        ]
    )
    recipe_path.write_text(yaml.safe_dump(raw))

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build", no_hash=True)

    bump = plan.step("bump_non_expert").argv
    assert bump[3] == "benchmarks/materialize_glm45_air_non_expert_precision.py"
    assert bump[bump.index("--seed-artifact-dir") + 1] == str(plan.step("train").out_dir)
    assert bump[bump.index("--surfaces") + 1] == "embed_tokens,lm_head"
    assert bump[bump.index("--dtype") + 1] == "bf16"
    assert "--allow-existing" not in bump
    assert "--overwrite" not in bump

    router = plan.step("router_kd").argv
    assert router[3] == "benchmarks/materialize_glm45_air_router_correction.py"
    assert router[router.index("--seed-artifact-dir") + 1] == str(
        plan.step("bump_non_expert").out_dir
    )
    assert router[router.index("--disagreement-json") + 1] == str(disagreement)
    assert router[router.index("--scale") + 1] == "1"
    assert router[router.index("--max-abs-delta") + 1] == "0.125"
    assert router[router.index("--num-experts") + 1] == "128"
    assert "--allow-existing" not in router
    assert "--overwrite" not in router


def test_isolate_sparse_fp16_renders_high_precision_projection_materializer(
    tmp_path: Path,
) -> None:
    recipe_path = _write_tiny_lineage(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].append(
        {
            "id": "isolate_sparse_fp16",
            "op": "isolate-sparse-fp16",
            "class": "promotable",
            "inputs": {"seed_artifact": "step:train/artifact"},
            "params": {
                "projection": ["41:gate_proj", "41:up_proj"],
                "model_id": "zai-org/GLM-4.5-Air",
                "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                "expert_count": 128,
            },
            "gate": {"profile": "train_sane"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build", no_hash=True)

    isolate = plan.step("isolate_sparse_fp16").argv
    assert isolate[3] == "benchmarks/materialize_glm45_air_high_precision_projection.py"
    assert isolate[isolate.index("--seed-artifact-dir") + 1] == str(
        plan.step("train").out_dir
    )
    projection_flags = [
        isolate[index + 1]
        for index, value in enumerate(isolate)
        if value == "--projection"
    ]
    assert projection_flags == ["41:gate_proj", "41:up_proj"]
    assert isolate[isolate.index("--model-id") + 1] == "zai-org/GLM-4.5-Air"
    assert isolate[isolate.index("--revision") + 1] == (
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert isolate[isolate.index("--expert-count") + 1] == "128"
    assert "--source-dir" not in isolate
    assert "--allow-existing" not in isolate


def test_sparse_residual_renders_plan_backed_materializer(tmp_path: Path) -> None:
    recipe_path = _write_tiny_lineage(tmp_path)
    sparse_plan = tmp_path / "sparse-plan.json"
    sparse_plan.write_text('{"record_type":"layer_probe_sparse_residual_plan"}')
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["sparse_plan"] = {
        "kind": "file",
        "path": str(sparse_plan),
    }
    raw["steps"].append(
        {
            "id": "sparse_residual",
            "op": "sparse-residual",
            "class": "promotable",
            "inputs": {
                "seed_artifact": "step:train/artifact",
                "plan_json": "external:sparse_plan",
            },
            "params": {
                "model_id": "zai-org/GLM-4.5-Air",
                "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                "projection": "down_proj",
                "plan_target": "report_route_000:0",
                "plan_expert": 93,
                "plan_route_rank": 0,
                "plan_max_rows": 4,
                "residual_scale": 0.75,
            },
            "gate": {"profile": "train_sane"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")

    sparse = plan.step("sparse_residual").argv
    assert sparse[3] == "benchmarks/materialize_glm45_air_vq_sparse_residual.py"
    assert sparse[sparse.index("--seed-artifact-dir") + 1] == str(plan.step("train").out_dir)
    assert sparse[sparse.index("--plan-json") + 1] == str(sparse_plan)
    assert sparse[sparse.index("--model-id") + 1] == "zai-org/GLM-4.5-Air"
    assert sparse[sparse.index("--revision") + 1] == (
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert sparse[sparse.index("--projection") + 1] == "down_proj"
    assert sparse[sparse.index("--plan-target") + 1] == "report_route_000:0"
    assert sparse[sparse.index("--plan-expert") + 1] == "93"
    assert sparse[sparse.index("--plan-route-rank") + 1] == "0"
    assert sparse[sparse.index("--plan-max-rows") + 1] == "4"
    assert sparse[sparse.index("--residual-scale") + 1] == "0.75"
    assert "--allow-existing" not in sparse


def test_layer_probe_attribution_op_renders_sparse_plan_source_oracle(
    tmp_path: Path,
) -> None:
    recipe_path = _write_tiny_lineage(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text("{}")
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text("{}\n")
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text("{}\n")
    states = tmp_path / "states.npz"
    states.write_bytes(b"fixture")
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"].update(
        {
            "source": {"kind": "artifact_dir", "path": str(source_dir)},
            "index": {"kind": "file", "path": str(index_path)},
            "teacher_report": {"kind": "teacher_cache", "path": str(teacher)},
            "eval_report": {"kind": "file", "path": str(eval_jsonl)},
            "states": {"kind": "file", "path": str(states)},
        }
    )
    raw["steps"].append(
        {
            "id": "layer_probe_sparse_plan",
            "op": "layer-probe-attribution",
            "class": "diagnostic",
            "inputs": {
                "artifact": "step:train/artifact",
                "source_dir": "external:source",
                "index_path": "external:index",
                "teacher": "external:teacher_report",
                "eval": "external:eval_report",
                "state_npz": "external:states",
            },
            "params": {
                "model_id": "zai-org/GLM-4.5-Air",
                "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                "layers": "41",
                "target": ["report_route_000:0", "report_math_044:0"],
                "no_strict_config": True,
            },
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")

    probe = plan.step("layer_probe_sparse_plan").argv
    assert probe[3] == "benchmarks/probe_glm45_air_layer_attribution.py"
    assert probe[probe.index("--artifact-dir") + 1] == str(plan.step("train").out_dir)
    assert probe[probe.index("--source-dir") + 1] == str(source_dir)
    assert probe[probe.index("--index-path") + 1] == str(index_path)
    assert probe[probe.index("--teacher-jsonl") + 1] == str(teacher)
    assert probe[probe.index("--eval-jsonl") + 1] == str(eval_jsonl)
    assert probe[probe.index("--state-npz") + 1] == str(states)
    assert probe[probe.index("--output-json") + 1].endswith("/plan_json.json")
    target_values = [
        probe[index + 1]
        for index, value in enumerate(probe)
        if value == "--target"
    ]
    assert target_values == ["report_route_000:0", "report_math_044:0"]
    assert probe[probe.index("--layers") + 1] == "41"
    assert "--no-strict-config" in probe


def test_merge_block_local_sidecars_renders_repeated_source_inputs(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    source_l18 = tmp_path / "source-l18"
    source_l41 = tmp_path / "source-l41"
    for path in (seed, source_l18, source_l41):
        path.mkdir()
        (path / "conversion-manifest.json").write_text(json.dumps({"schema": 1}))
    recipe_path = tmp_path / "merge-sidecars.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "merge-sidecars",
                "description": "fixture",
                "external_inputs": {
                    "seed": {"kind": "artifact_dir", "path": str(seed)},
                    "source_l18": {"kind": "artifact_dir", "path": str(source_l18)},
                    "source_l41": {"kind": "artifact_dir", "path": str(source_l41)},
                },
                "steps": [
                    {
                        "id": "merge",
                        "op": "merge-block-local-sidecars",
                        "class": "promotable",
                        "inputs": {
                            "seed_artifact": "external:seed",
                            "source_artifact_1": "external:source_l18",
                            "source_artifact_2": "external:source_l41",
                        },
                        "params": {
                            "train_split": "selection",
                            "train_row_count": 8,
                            "max_positions": 8,
                            "trainable": "output_bias",
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("merge").argv

    assert argv[3] == "benchmarks/materialize_glm45_air_merged_sidecars.py"
    assert argv[argv.index("--seed-artifact-dir") + 1] == str(seed)
    source_flags = [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--source-artifact-dir"
    ]
    assert source_flags == [str(source_l18), str(source_l41)]
    assert argv[argv.index("--train-split") + 1] == "selection"
    assert argv[argv.index("--max-positions") + 1] == "8"
    assert "--allow-existing" not in argv
    assert "--overwrite" not in argv


def test_fit_block_local_sidecar_derives_teacher_cache_root(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    source_dir = tmp_path / "hf-snapshot"
    teacher = tmp_path / "teacher-cache" / "metadata.jsonl"
    seed.mkdir()
    source_dir.mkdir()
    teacher.parent.mkdir()
    (seed / "conversion-manifest.json").write_text(json.dumps({"schema": 1}))
    (source_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    teacher.write_text("{}\n")
    recipe_path = tmp_path / "fit-sidecar.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "fit-sidecar",
                "description": "fixture",
                "external_inputs": {
                    "seed": {"kind": "artifact_dir", "path": str(seed)},
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "teacher": {"kind": "teacher_cache", "path": str(teacher)},
                },
                "steps": [
                    {
                        "id": "fit_l18_gate",
                        "op": "fit-block-local-sidecar",
                        "class": "promotable",
                        "inputs": {
                            "seed_artifact": "external:seed",
                            "source_dir": "external:source_dir",
                            "teacher_1": "external:teacher",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                            "layer": 18,
                            "projection": "gate_proj",
                            "trainable": "output_bias",
                            "max_train_rows": 8,
                            "max_positions": 8,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("fit_l18_gate").argv

    assert argv[3] == "benchmarks/fit_glm45_air_block_local_sidecar.py"
    assert argv[argv.index("--seed-artifact-dir") + 1] == str(seed)
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--teacher-jsonl") + 1] == str(teacher)
    assert argv[argv.index("--teacher-cache-root") + 1] == str(teacher.parent)
    assert argv[argv.index("--layer") + 1] == "18"
    assert argv[argv.index("--projection") + 1] == "gate_proj"
    assert argv[argv.index("--max-train-rows") + 1] == "8"
    assert "--allow-existing" not in argv
    assert "--overwrite" not in argv


def test_target_recipe_reuses_external_q2_control_without_q2_step() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)

    assert "lane_s_q2_control" not in {step.spec.id for step in plan.steps}
    lane_s = plan.step("lane_s_candidate")
    assert lane_s.resolved_inputs["control_evidence"] == Path(
        "artifacts/benchmarks/"
        "glm45-air-target23-nextcycle-r26-q2-control-prefill1k-quiet-20260701.jsonl"
    )
    assert "--memory-quiet-preflight" not in lane_s.argv
    assert lane_s.argv[lane_s.argv.index("--warmup-repetitions") + 1] == "1"
    assert lane_s.argv[lane_s.argv.index("--clean-repetition-max-attempts") + 1] == "4"
    assert "--mlx-cache-limit-gb" not in lane_s.argv
    assert "--mlx-clear-cache-before-run" not in lane_s.argv
    assert lane_s.argv[lane_s.argv.index("--memory-quiet-window-seconds") + 1] == "3"
    assert lane_s.argv[lane_s.argv.index("--memory-quiet-max-attempts") + 1] == "1"
    assert lane_s.argv[lane_s.argv.index("--prefill-memory-quiet-window-seconds") + 1] == "3"
    assert lane_s.argv[lane_s.argv.index("--prefill-memory-quiet-max-attempts") + 1] == "1"
    assert "--parent-vm-stat-diagnostics" in lane_s.argv
    assert "--memory-phase-trace" in lane_s.argv

    rendered = render_dry_run(plan)
    assert "input control_evidence: <unhashed>" in rendered
    assert "mlx_q2_routed_g128" not in rendered


def test_target_recipe_has_tail_cleanup_top1_gap_diagnostic() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    tail_targets = plan.step("top1_gap_tail_cleanup")
    assert tail_targets.spec.step_class == "diagnostic"
    assert step_ids.index("eval_report_repair") < step_ids.index(
        "top1_gap_tail_cleanup"
    ) < step_ids.index("lane_s_candidate")
    assert tail_targets.resolved_inputs["evidence"] == plan.step(
        "eval_report_repair"
    ).evidence_dir / "evidence.jsonl"
    assert tail_targets.argv[3] == "benchmarks/analyze_glm45_air_top1_gap.py"
    assert "--select-tail-cleanup-targets" in tail_targets.argv
    assert tail_targets.argv[tail_targets.argv.index("--teacher-token-id") + 1] == "565"
    assert tail_targets.argv[tail_targets.argv.index("--focus-position") + 1] == "0"
    assert tail_targets.argv[tail_targets.argv.index("--max-train-rows") + 1] == "8"
    assert tail_targets.argv[tail_targets.argv.index("--max-validation-rows") + 1] == "12"
    assert tail_targets.argv[tail_targets.argv.index("--top-tail-records") + 1] == "40"
    assert (
        tail_targets.argv[tail_targets.argv.index("--label") + 1]
        == "route4_pos0_tail_cleanup"
    )


def test_target_recipe_runs_tail_cleanup_train_slice_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    assert step_ids.index("top1_gap_tail_cleanup") < step_ids.index(
        "train_tail_cleanup_1"
    ) < step_ids.index("audit_train_tail_cleanup_1") < step_ids.index(
        "lane_s_candidate"
    )

    train = plan.step("train_tail_cleanup_1")
    assert train.spec.step_class == "promotable"
    assert train.spec.inputs["seed_artifact"].raw == "step:train_l45_gud_math8/artifact"
    assert train.spec.params["layer"] == 41
    assert train.spec.params["trainable"] == "low_rank_residual"
    assert train.spec.params["low_rank"] == 4
    assert train.spec.params["steps"] == 1
    assert train.spec.params["learning_rate"] == 0.00025
    assert train.spec.params["loss_scope"] == "selected_layer"
    assert train.spec.params["train_cache"] == "validation"
    assert train.spec.params["train_row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert train.spec.params["max_positions"] == 1
    assert train.spec.params["aux_loss_position_indices"] == [0]
    assert "--aux-loss-position-indices" in train.argv
    assert train.argv[train.argv.index("--aux-loss-position-indices") + 1] == "0"

    audit = plan.step("audit_train_tail_cleanup_1")
    assert audit.spec.inputs["artifact"].raw == "step:train_tail_cleanup_1/artifact"
    assert audit.spec.gate is not None
    assert audit.spec.gate.profile == "audit_ok"


def test_target_recipe_runs_tail_cleanup_focused_report_eval_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    assert step_ids.index("audit_train_tail_cleanup_1") < step_ids.index(
        "eval_report_tail_cleanup_1_focus"
    ) < step_ids.index("lane_s_candidate")

    focus_eval = plan.step("eval_report_tail_cleanup_1_focus")
    assert focus_eval.spec.step_class == "diagnostic"
    assert focus_eval.spec.inputs["artifact"].raw == "step:train_tail_cleanup_1/artifact"
    assert focus_eval.spec.inputs["teacher"].raw == "external:teacher_report"
    assert focus_eval.spec.params["engine"] == "vq_e1_routed_nax_e8p"
    assert focus_eval.spec.params["row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert focus_eval.spec.params["mlx_cache_limit_gb"] == 0
    assert focus_eval.spec.params["mlx_clear_cache_before_load"] is True
    assert focus_eval.argv[focus_eval.argv.index("--row-indices") + 1] == (
        "120,73,44,0,108,121,75,45"
    )


def test_target_recipe_runs_stronger_tail_cleanup_variant_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    assert step_ids.index("eval_report_tail_cleanup_1_focus") < step_ids.index(
        "train_tail_cleanup_2"
    ) < step_ids.index("audit_train_tail_cleanup_2") < step_ids.index(
        "eval_report_tail_cleanup_2_focus"
    ) < step_ids.index(
        "lane_s_candidate"
    )

    train = plan.step("train_tail_cleanup_2")
    assert train.spec.inputs["seed_artifact"].raw == "step:train_tail_cleanup_1/artifact"
    assert train.spec.params["layer"] == 41
    assert train.spec.params["low_rank"] == 4
    assert train.spec.params["steps"] == 4
    assert train.spec.params["learning_rate"] == 0.0005
    assert train.spec.params["target_nll_weight"] == 4.0
    assert train.spec.params["teacher_top1_margin_weight"] == 4.0
    assert train.spec.params["tail_kld_weight"] == 0.0002
    assert train.spec.params["train_row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert train.spec.params["aux_loss_position_indices"] == [0]

    audit = plan.step("audit_train_tail_cleanup_2")
    assert audit.spec.inputs["artifact"].raw == "step:train_tail_cleanup_2/artifact"
    assert audit.spec.gate is not None
    assert audit.spec.gate.profile == "audit_ok"

    focus_eval = plan.step("eval_report_tail_cleanup_2_focus")
    assert focus_eval.spec.inputs["artifact"].raw == "step:train_tail_cleanup_2/artifact"
    assert focus_eval.spec.inputs["teacher"].raw == "external:teacher_report"
    assert focus_eval.spec.params["row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert focus_eval.spec.params["mlx_cache_limit_gb"] == 0
    assert focus_eval.spec.params["mlx_clear_cache_before_load"] is True


def test_target_recipe_runs_non_expert_precision_candidate_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    assert step_ids.index("eval_report_tail_cleanup_2_focus") < step_ids.index(
        "bump_non_expert_precision_1"
    ) < step_ids.index(
        "eval_report_bump_non_expert_precision_1_focus"
    ) < step_ids.index(
        "lane_s_candidate"
    )

    bump = plan.step("bump_non_expert_precision_1")
    assert bump.spec.op == "bump-non-expert-precision"
    assert bump.spec.step_class == "diagnostic"
    assert bump.spec.inputs["seed_artifact"].raw == "step:train_tail_cleanup_2/artifact"
    assert bump.spec.params["surfaces"] == ["embed_tokens", "lm_head", "router_gates"]
    assert bump.spec.params["dtype"] == "bf16"
    assert (
        bump.argv[bump.argv.index("--surfaces") + 1]
        == "embed_tokens,lm_head,router_gates"
    )

    focus_eval = plan.step("eval_report_bump_non_expert_precision_1_focus")
    assert (
        focus_eval.spec.inputs["artifact"].raw
        == "step:bump_non_expert_precision_1/artifact"
    )
    assert focus_eval.spec.inputs["teacher"].raw == "external:teacher_report"
    assert focus_eval.spec.params["row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert focus_eval.spec.params["mlx_cache_limit_gb"] == 0
    assert focus_eval.spec.params["mlx_clear_cache_before_load"] is True


def test_target_recipe_runs_free_lever_selection_splits_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    bump_resident = plan.step("resident_byte_audit_bump_non_expert_precision_1")
    assert bump_resident.spec.op == "resident-byte-audit"
    assert bump_resident.spec.step_class == "diagnostic"
    assert (
        bump_resident.spec.inputs["artifact"].raw
        == "step:bump_non_expert_precision_1/artifact"
    )
    assert (
        bump_resident.spec.inputs["baseline_evidence"].raw
        == "step:resident_byte_audit/evidence_json"
    )

    free_lever_steps = (
        (
            "bump_non_expert_precision_1",
            "eval_selection_bump_non_expert_precision_1_full",
            "eval_selection_bump_non_expert_precision_1_full_repair",
            "step:bump_non_expert_precision_1/artifact",
        ),
        (
            "train_router_kd_1",
            "eval_selection_train_router_kd_1_full",
            "eval_selection_train_router_kd_1_full_repair",
            "step:train_router_kd_1/artifact",
        ),
    )
    for source_step, raw_step_id, repair_step_id, artifact_input in free_lever_steps:
        raw_step = plan.step(raw_step_id)
        repair_step = plan.step(repair_step_id)
        assert step_ids.index(source_step) < step_ids.index(raw_step_id) < step_ids.index(
            repair_step_id
        ) < step_ids.index("lane_s_candidate")
        assert raw_step.spec.op == "eval"
        assert raw_step.spec.step_class == "diagnostic"
        assert raw_step.spec.inputs["artifact"].raw == artifact_input
        assert raw_step.spec.inputs["teacher"].raw == "external:teacher_selection"
        assert raw_step.spec.params["max_rows"] == 128
        assert raw_step.spec.params["mlx_cache_limit_gb"] == 0
        assert raw_step.spec.params["mlx_clear_cache_before_load"] is True
        assert repair_step.spec.op == "eval-repair"
        assert repair_step.spec.step_class == "diagnostic"
        assert repair_step.spec.inputs["artifact"].raw == artifact_input
        assert repair_step.spec.inputs["teacher"].raw == "external:teacher_selection"
        assert (
            repair_step.spec.inputs["existing_evidence"].raw
            == f"step:{raw_step_id}/evidence"
        )
        assert repair_step.spec.params["max_rows"] == 128
        assert repair_step.spec.params["repair_attempts"] == 2
        assert repair_step.spec.params["mlx_cache_limit_gb"] == 0
        assert repair_step.spec.params["mlx_clear_cache_before_load"] is True


def test_target_recipe_runs_free_lever_holdout_splits_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    free_lever_steps = (
        (
            "bump_non_expert_precision_1",
            "eval_holdout_bump_non_expert_precision_1_full",
            "eval_holdout_bump_non_expert_precision_1_full_repair",
            "step:bump_non_expert_precision_1/artifact",
        ),
        (
            "train_router_kd_1",
            "eval_holdout_train_router_kd_1_full",
            "eval_holdout_train_router_kd_1_full_repair",
            "step:train_router_kd_1/artifact",
        ),
    )
    for source_step, raw_step_id, repair_step_id, artifact_input in free_lever_steps:
        raw_step = plan.step(raw_step_id)
        repair_step = plan.step(repair_step_id)
        assert step_ids.index(source_step) < step_ids.index(raw_step_id) < step_ids.index(
            repair_step_id
        ) < step_ids.index("lane_s_candidate")
        assert raw_step.spec.op == "eval"
        assert raw_step.spec.step_class == "diagnostic"
        assert raw_step.spec.inputs["artifact"].raw == artifact_input
        assert raw_step.spec.inputs["teacher"].raw == "external:teacher_holdout"
        assert raw_step.spec.params["max_rows"] == 128
        assert raw_step.spec.params["mlx_cache_limit_gb"] == 0
        assert raw_step.spec.params["mlx_clear_cache_before_load"] is True
        assert repair_step.spec.op == "eval-repair"
        assert repair_step.spec.step_class == "diagnostic"
        assert repair_step.spec.inputs["artifact"].raw == artifact_input
        assert repair_step.spec.inputs["teacher"].raw == "external:teacher_holdout"
        assert (
            repair_step.spec.inputs["existing_evidence"].raw
            == f"step:{raw_step_id}/evidence"
        )
        assert repair_step.spec.params["max_rows"] == 128
        assert repair_step.spec.params["repair_attempts"] == 2
        assert repair_step.spec.params["mlx_cache_limit_gb"] == 0
        assert repair_step.spec.params["mlx_clear_cache_before_load"] is True


def test_target_recipe_runs_router_kd_diagnostic_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    assert "route_disagreement" in recipe.external_inputs
    assert step_ids.index("eval_report_bump_non_expert_precision_1_focus") < step_ids.index(
        "train_router_kd_1"
    ) < step_ids.index(
        "eval_report_train_router_kd_1_focus"
    ) < step_ids.index(
        "lane_s_candidate"
    )

    router = plan.step("train_router_kd_1")
    assert router.spec.op == "train-router-kd"
    assert router.spec.step_class == "diagnostic"
    assert router.spec.inputs["seed_artifact"].raw == "step:train_tail_cleanup_2/artifact"
    assert router.spec.inputs["disagreement"].raw == "external:route_disagreement"
    assert router.spec.params["scale"] == 1
    assert router.spec.params["max_abs_delta"] == 0.125
    assert router.spec.params["num_experts"] == 128
    assert router.argv[router.argv.index("--scale") + 1] == "1"
    assert router.argv[router.argv.index("--max-abs-delta") + 1] == "0.125"

    focus_eval = plan.step("eval_report_train_router_kd_1_focus")
    assert focus_eval.spec.inputs["artifact"].raw == "step:train_router_kd_1/artifact"
    assert focus_eval.spec.inputs["teacher"].raw == "external:teacher_report"
    assert focus_eval.spec.params["row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert focus_eval.spec.params["mlx_cache_limit_gb"] == 0
    assert focus_eval.spec.params["mlx_clear_cache_before_load"] is True


def test_target_recipe_declares_fresh_sparse_residual_plan_for_plan_next() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )

    assert "sparse_residual_plan" in recipe.external_inputs
    assert recipe.external_inputs["sparse_residual_plan"].path == (
        "artifacts/quality/"
        "glm45-air-train-tail-cleanup-2-report-route000-layer41-sparse-plan-20260702.json"
    )


def test_target_recipe_runs_sparse_residual_candidate_before_lane_s() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)
    step_ids = [step.spec.id for step in plan.steps]

    assert step_ids.index("eval_report_train_router_kd_1_focus") < step_ids.index(
        "sparse_residual_1"
    ) < step_ids.index(
        "audit_sparse_residual_1"
    ) < step_ids.index(
        "eval_report_sparse_residual_1_focus"
    ) < step_ids.index(
        "lane_s_candidate"
    )

    sparse = plan.step("sparse_residual_1")
    assert sparse.spec.op == "sparse-residual"
    assert sparse.spec.step_class == "promotable"
    assert sparse.spec.inputs["seed_artifact"].raw == "step:train_tail_cleanup_2/artifact"
    assert sparse.spec.inputs["plan_json"].raw == "external:sparse_residual_plan"
    assert sparse.spec.gate is not None
    assert sparse.spec.gate.profile == "train_sane"
    assert sparse.spec.params["projection"] == "down_proj"
    assert sparse.spec.params["plan_target"] == "report_route_000:0"
    assert sparse.spec.params["plan_expert"] == 93
    assert sparse.spec.params["plan_route_rank"] == 0
    assert sparse.spec.params["plan_max_rows"] == 4
    assert sparse.spec.params["residual_scale"] == 0.5
    assert sparse.argv[sparse.argv.index("--plan-json") + 1] == (
        "artifacts/quality/"
        "glm45-air-train-tail-cleanup-2-report-route000-layer41-sparse-plan-20260702.json"
    )
    assert sparse.argv[sparse.argv.index("--plan-expert") + 1] == "93"

    audit = plan.step("audit_sparse_residual_1")
    assert audit.spec.inputs["artifact"].raw == "step:sparse_residual_1/artifact"
    assert audit.spec.gate is not None
    assert audit.spec.gate.profile == "audit_ok"

    focus_eval = plan.step("eval_report_sparse_residual_1_focus")
    assert focus_eval.spec.inputs["artifact"].raw == "step:sparse_residual_1/artifact"
    assert focus_eval.spec.inputs["teacher"].raw == "external:teacher_report"
    assert focus_eval.spec.params["row_indices"] == [120, 73, 44, 0, 108, 121, 75, 45]
    assert focus_eval.spec.params["mlx_cache_limit_gb"] == 0
    assert focus_eval.spec.params["mlx_clear_cache_before_load"] is True


def test_benchmark_lane_s_renders_prefill_chunk_size(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    control = tmp_path / "q2.jsonl"
    control.write_text("{}\n")
    recipe_path = tmp_path / "lane-s-chunk.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "lane-s-chunk-fixture",
                "description": "fixture",
                "external_inputs": {
                    "artifact": {"kind": "artifact_dir", "path": str(artifact)},
                    "control": {"kind": "file", "path": str(control)},
                },
                "steps": [
                    {
                        "id": "lane_s",
                        "op": "benchmark-lane-s",
                        "class": "verify",
                        "inputs": {
                            "artifact": "external:artifact",
                            "control_evidence": "external:control",
                        },
                        "params": {
                            "engine": "vq_e1_routed_nax_e8p",
                            "scenario": "prefill_1k",
                            "prefill_chunk_size": 256,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("lane_s").argv

    assert argv[argv.index("--prefill-chunk-size") + 1] == "256"


def test_resident_byte_audit_renders_source_and_artifact_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    baseline_json = tmp_path / "baseline-resident-bytes.json"
    source_dir = tmp_path / "hf-source"
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    artifact.mkdir()
    baseline_json.write_text(json.dumps({"counted_resident_total_bytes": 1000}))
    source_dir.mkdir()
    config_path.write_text(json.dumps({"num_hidden_layers": 1}))
    index_path.write_text(json.dumps({"weight_map": {}}))
    recipe_path = tmp_path / "resident-byte-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "resident-byte-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "artifact": {"kind": "artifact_dir", "path": str(artifact)},
                    "baseline_json": {"kind": "file", "path": str(baseline_json)},
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "resident_bytes",
                        "op": "resident-byte-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "artifact": "external:artifact",
                            "baseline_evidence": "external:baseline_json",
                            "source_dir": "external:source_dir",
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                            "max_counted_resident_total_bytes": 900,
                            "min_total_byte_reduction": 100,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("resident_bytes").argv

    assert argv[3] == "benchmarks/audit_glm45_air_resident_bytes.py"
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact)
    assert argv[argv.index("--baseline-json") + 1] == str(baseline_json)
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--model-id") + 1] == "zai-org/GLM-4.5-Air"
    assert argv[argv.index("--revision") + 1] == "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    assert argv[argv.index("--max-counted-resident-total-bytes") + 1] == "900"
    assert argv[argv.index("--min-total-byte-reduction") + 1] == "100"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--allow-existing" not in argv
    assert "--overwrite" not in argv


def test_qwen_moe_source_audit_op_renders_pinned_source_audit(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    config_path = tmp_path / "config.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    recipe_path = tmp_path / "qwen-source-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-source-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                },
                "steps": [
                    {
                        "id": "source_audit",
                        "op": "qwen-moe-source-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                        },
                        "params": {
                            "model_id": "Qwen/Qwen3.6-35B-A3B",
                            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                            "expected_model_type": "qwen3_5_moe",
                            "expected_language_layers": 40,
                            "expected_expert_tensors": 80,
                            "verify_hf_shapes": True,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("source_audit").argv

    assert argv[3] == "benchmarks/inspect_qwen_moe_source.py"
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--model-id") + 1] == "Qwen/Qwen3.6-35B-A3B"
    assert argv[argv.index("--revision") + 1] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert argv[argv.index("--expected-model-type") + 1] == "qwen3_5_moe"
    assert argv[argv.index("--expected-language-layers") + 1] == "40"
    assert argv[argv.index("--expected-expert-tensors") + 1] == "80"
    assert "--verify-hf-shapes" in argv
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_qwen_moe_source_payload_audit_op_renders_payload_preflight(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "qwen-source"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    recipe_path = tmp_path / "qwen-source-payload-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-source-payload-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                },
                "steps": [
                    {
                        "id": "source_payload_audit",
                        "op": "qwen-moe-source-payload-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                        },
                        "params": {
                            "model_id": "Qwen/Qwen3.6-35B-A3B",
                            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                            "expected_language_layers": 40,
                            "max_groups": 2,
                            "group_size": 512,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("source_payload_audit").argv

    assert argv[3] == "benchmarks/audit_qwen_moe_source_payloads.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--model-id") + 1] == "Qwen/Qwen3.6-35B-A3B"
    assert argv[argv.index("--revision") + 1] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert argv[argv.index("--expected-language-layers") + 1] == "40"
    assert argv[argv.index("--max-groups") + 1] == "2"
    assert argv[argv.index("--group-size") + 1] == "512"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_glm52_source_audit_op_renders_pinned_profile_audit(tmp_path: Path) -> None:
    profile_path = tmp_path / "glm52-reap-504b-v2.yaml"
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    index_path.write_text(json.dumps({"weight_map": {}}))
    recipe_path = tmp_path / "glm52-source-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-source-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "source_audit",
                        "op": "glm52-source-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "profile_path": "external:profile_path",
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "model_id": "0xSero/glm-5.2-reap-504B-v2",
                            "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("source_audit").argv

    assert argv[3] == "benchmarks/inspect_glm52_reap_source.py"
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--profile-path") + 1] == str(profile_path)
    assert argv[argv.index("--model-id") + 1] == "0xSero/glm-5.2-reap-504B-v2"
    assert argv[argv.index("--revision") + 1] == "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--output-dir" not in argv
    assert "--overwrite" not in argv

    first_key = plan.step("source_audit").step_key
    profile_path.write_text("name: glm52-reap-504b-v2\nnotes: changed\n")
    replanned = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    assert replanned.step("source_audit").step_key != first_key


def test_glm52_source_payload_audit_op_renders_header_preflight(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    profile_path = tmp_path / "glm52-reap-504b-v2.yaml"
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    source_audit_evidence = tmp_path / "source-audit.json"
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    index_path.write_text(json.dumps({"weight_map": {}}))
    source_audit_evidence.write_text(json.dumps({"source_checks_pass": True}))
    recipe_path = tmp_path / "glm52-source-payload-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-source-payload-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "source_audit_evidence": {
                        "kind": "file",
                        "path": str(source_audit_evidence),
                    },
                },
                "steps": [
                    {
                        "id": "source_payload_audit",
                        "op": "glm52-source-payload-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "profile_path": "external:profile_path",
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                            "source_audit_evidence": "external:source_audit_evidence",
                        },
                        "params": {
                            "model_id": "0xSero/glm-5.2-reap-504B-v2",
                            "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
                            "max_groups": 2,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("source_payload_audit").argv

    assert argv[3] == "benchmarks/audit_glm52_reap_source_payloads.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--profile-path") + 1] == str(profile_path)
    assert argv[argv.index("--model-id") + 1] == "0xSero/glm-5.2-reap-504B-v2"
    assert argv[argv.index("--revision") + 1] == "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    assert argv[argv.index("--max-groups") + 1] == "2"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--output-dir" not in argv
    assert plan.step("source_payload_audit").op_def.cacheable is False


def test_glm52_vq_validate_op_renders_layer_local_probe(tmp_path: Path) -> None:
    source_dir = tmp_path / "glm52-source"
    artifact = tmp_path / "glm52-artifact"
    source_dir.mkdir()
    artifact.mkdir()
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    index_path.write_text(json.dumps({"weight_map": {}}))
    recipe_path = tmp_path / "glm52-validate.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-validate-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "artifact": {"kind": "artifact_dir", "path": str(artifact)},
                },
                "steps": [
                    {
                        "id": "validate_layer3",
                        "op": "glm52-vq-validate",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                            "artifact": "external:artifact",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-5.2",
                            "revision": "f2263102df303b2faa54a6861a29d1770ce846c0",
                            "layer": 3,
                            "tokens": 2,
                            "seed": 20260702,
                            "no_strict_config": True,
                            "min_artifact_cosine": 0.999,
                            "min_source_weighted_cosine": 0.999,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("validate_layer3").argv

    assert argv[3] == "scripts/validate_glm52_vq.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact)
    assert argv[argv.index("--model-id") + 1] == "zai-org/GLM-5.2"
    assert argv[argv.index("--revision") + 1] == (
        "f2263102df303b2faa54a6861a29d1770ce846c0"
    )
    assert argv[argv.index("--layer") + 1] == "3"
    assert argv[argv.index("--tokens") + 1] == "2"
    assert argv[argv.index("--seed") + 1] == "20260702"
    assert argv[argv.index("--min-artifact-cosine") + 1] == "0.999"
    assert argv[argv.index("--min-source-weighted-cosine") + 1] == "0.999"
    assert "--no-strict-config" in argv


def test_qwen_moe_materialize_group_op_renders_diagnostic_materializer(tmp_path: Path) -> None:
    source_dir = tmp_path / "qwen-source"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    recipe_path = tmp_path / "qwen-materialize.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-materialize-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                },
                "steps": [
                    {
                        "id": "materialize_gate_up_l0",
                        "op": "qwen-moe-materialize-group",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                        },
                        "params": {
                            "model_id": "Qwen/Qwen3.6-35B-A3B",
                            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                            "layer": 0,
                            "source_projection": "gate_up_proj",
                            "expected_language_layers": 40,
                            "group_size": 512,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("materialize_gate_up_l0").argv

    assert argv[3] == "benchmarks/materialize_qwen_moe_group.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--model-id") + 1] == "Qwen/Qwen3.6-35B-A3B"
    assert argv[argv.index("--revision") + 1] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert argv[argv.index("--layer") + 1] == "0"
    assert argv[argv.index("--source-projection") + 1] == "gate_up_proj"
    assert argv[argv.index("--expected-language-layers") + 1] == "40"
    assert argv[argv.index("--group-size") + 1] == "512"
    assert argv[argv.index("--output-dir") + 1].endswith("out")
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_qwen_moe_materialize_groups_op_renders_batch_materializer(tmp_path: Path) -> None:
    source_dir = tmp_path / "qwen-source"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    recipe_path = tmp_path / "qwen-materialize-groups.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-materialize-groups-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                },
                "steps": [
                    {
                        "id": "materialize_groups",
                        "op": "qwen-moe-materialize-groups",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                        },
                        "params": {
                            "model_id": "Qwen/Qwen3.6-35B-A3B",
                            "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                            "expected_language_layers": 40,
                            "max_groups": 2,
                            "skip_existing": True,
                            "group_size": 512,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("materialize_groups").argv

    assert argv[3] == "benchmarks/materialize_qwen_moe_group.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--model-id") + 1] == "Qwen/Qwen3.6-35B-A3B"
    assert argv[argv.index("--revision") + 1] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert "--all-groups" in argv
    assert argv[argv.index("--max-groups") + 1] == "2"
    assert "--skip-existing" in argv
    assert argv[argv.index("--expected-language-layers") + 1] == "40"
    assert argv[argv.index("--group-size") + 1] == "512"
    assert argv[argv.index("--output-dir") + 1].endswith("out")
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_glm52_reap_materialize_groups_op_renders_pinned_resumable_materializer(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "glm52-source"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    profile_path = tmp_path / "glm52-profile.yaml"
    payload_audit_evidence = tmp_path / "payload-audit.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    payload_audit_evidence.write_text(
        json.dumps({"materialization_blocked": False})
    )
    recipe_path = tmp_path / "glm52-materialize-groups.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-materialize-groups-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "payload_audit_evidence": {
                        "kind": "file",
                        "path": str(payload_audit_evidence),
                    },
                },
                "steps": [
                    {
                        "id": "materialize_groups",
                        "op": "glm52-moe-materialize-groups",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                            "profile_path": "external:profile_path",
                            "payload_audit_evidence": "external:payload_audit_evidence",
                        },
                        "params": {
                            "model_id": "0xSero/glm-5.2-reap-504B-v2",
                            "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
                            "groups": ["10:gate_proj", "29:gate_proj"],
                            "code_bits": 16,
                            "group_size": 512,
                            "expert_workers": 2,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("materialize_groups")
    argv = step.argv

    assert argv[3] == "benchmarks/materialize_glm52_reap_groups.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--profile-path") + 1] == str(profile_path)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--model-id") + 1] == "0xSero/glm-5.2-reap-504B-v2"
    assert argv[argv.index("--revision") + 1] == (
        "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    )
    assert argv[argv.index("--groups") + 1] == "10:gate_proj,29:gate_proj"
    assert argv[argv.index("--code-bits") + 1] == "16"
    assert argv[argv.index("--expert-workers") + 1] == "2"
    assert "--skip-existing" in argv
    assert argv[argv.index("--output-dir") + 1].endswith("out")
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert step.op_def.resume_partial is True
    assert step.op_def.cacheable is True

    original_key = step.step_key
    profile_path.write_text("name: glm52-reap-504b-v2\nnotes: changed\n")
    replanned = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    assert replanned.step("materialize_groups").step_key != original_key


def test_glm52_materialization_probe_has_live_audit_dependency_chain() -> None:
    recipe = load_recipe(
        "recipes/glm52_reap_504b_materialization_probe_20260709.yaml"
    )

    payload_ref = recipe.step("source_payload_audit").inputs[
        "source_audit_evidence"
    ]
    materialize_ref = recipe.step("materialize_groups").inputs[
        "payload_audit_evidence"
    ]
    non_vq_ref = recipe.step("non_vq_pack").inputs[
        "payload_audit_evidence"
    ]
    artifact_ref = recipe.step("artifact_audit").inputs["artifact"]
    materialization_runs_ref = recipe.step("artifact_audit").inputs[
        "materialization_runs"
    ]
    assert (payload_ref.kind, payload_ref.name, payload_ref.output) == (
        "step",
        "source_audit",
        "evidence_json",
    )
    assert (materialize_ref.kind, materialize_ref.name, materialize_ref.output) == (
        "step",
        "source_payload_audit",
        "evidence_json",
    )
    assert (non_vq_ref.kind, non_vq_ref.name, non_vq_ref.output) == (
        "step",
        "source_payload_audit",
        "evidence_json",
    )
    assert (artifact_ref.kind, artifact_ref.name, artifact_ref.output) == (
        "step",
        "materialize_groups",
        "artifact",
    )
    assert (
        materialization_runs_ref.kind,
        materialization_runs_ref.name,
        materialization_runs_ref.output,
    ) == ("step", "materialize_groups", "evidence")


def test_glm52_reap_artifact_audit_op_renders_strict_pinned_audit(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "glm52-source"
    artifact_dir = tmp_path / "glm52-artifact"
    source_dir.mkdir()
    artifact_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    profile_path = tmp_path / "glm52-profile.yaml"
    runs_path = tmp_path / "materialization.jsonl"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    runs_path.write_text("{}\n")
    recipe_path = tmp_path / "glm52-artifact-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-artifact-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "artifact": {"kind": "artifact_dir", "path": str(artifact_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "materialization_runs": {"kind": "file", "path": str(runs_path)},
                },
                "steps": [
                    {
                        "id": "artifact_audit",
                        "op": "glm52-moe-artifact-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "artifact": "external:artifact",
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                            "profile_path": "external:profile_path",
                            "materialization_runs": "external:materialization_runs",
                        },
                        "params": {
                            "model_id": "0xSero/glm-5.2-reap-504B-v2",
                            "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
                            "groups": ["10:gate_proj", "29:gate_proj"],
                            "code_bits": 8,
                            "group_size": 512,
                            "scale_estimator": "max_abs",
                            "require_resume_proof": True,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("artifact_audit")
    argv = step.argv

    assert argv[3] == "benchmarks/audit_glm52_reap_materialization.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact_dir)
    assert argv[argv.index("--profile-path") + 1] == str(profile_path)
    assert argv[argv.index("--config-path") + 1] == str(config_path)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--materialization-runs-jsonl") + 1] == str(runs_path)
    assert argv[argv.index("--groups") + 1] == "10:gate_proj,29:gate_proj"
    assert argv[argv.index("--code-bits") + 1] == "8"
    assert argv[argv.index("--group-size") + 1] == "512"
    assert argv[argv.index("--scale-estimator") + 1] == "max_abs"
    assert "--require-resume-proof" in argv
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert step.op_def.cacheable is False


def test_glm52_full_artifact_audit_renders_authenticated_non_vq_composite_inputs(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "glm52-source"
    routed_dir = tmp_path / "glm52-routed"
    non_vq_dir = tmp_path / "glm52-non-vq"
    source_dir.mkdir()
    routed_dir.mkdir()
    non_vq_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    profile_path = tmp_path / "glm52-profile.yaml"
    runs_path = tmp_path / "materialization.jsonl"
    non_vq_evidence = tmp_path / "non-vq-evidence.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    runs_path.write_text("{}\n")
    non_vq_evidence.write_text("{}\n")
    recipe_path = tmp_path / "glm52-full-artifact-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-full-artifact-audit-fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "artifact": {"kind": "artifact_dir", "path": str(routed_dir)},
                    "non_vq_artifact": {
                        "kind": "artifact_dir",
                        "path": str(non_vq_dir),
                    },
                    "non_vq_evidence_json": {
                        "kind": "file",
                        "path": str(non_vq_evidence),
                    },
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "materialization_runs": {"kind": "file", "path": str(runs_path)},
                },
                "steps": [
                    {
                        "id": "artifact_audit",
                        "op": "glm52-moe-artifact-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "artifact": "external:artifact",
                            "non_vq_artifact": "external:non_vq_artifact",
                            "non_vq_evidence_json": "external:non_vq_evidence_json",
                            "index_path": "external:index_path",
                            "config_path": "external:config_path",
                            "profile_path": "external:profile_path",
                            "materialization_runs": "external:materialization_runs",
                        },
                        "params": {
                            "model_id": "0xSero/glm-5.2-reap-504B-v2",
                            "revision": "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
                            "all_groups": True,
                            "code_bits": 8,
                            "group_size": 512,
                            "scale_estimator": "max_abs",
                            "require_resume_proof": True,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("artifact_audit").argv

    assert argv[argv.index("--non-vq-artifact-dir") + 1] == str(non_vq_dir)
    assert argv[argv.index("--non-vq-evidence-json") + 1] == str(non_vq_evidence)
    assert "--all-groups" in argv


def test_qwen_moe_artifact_audit_op_renders_manifest_audit(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "qwen-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "qwen-moe-materialization-manifest.json").write_text(
        json.dumps({"materialization_status": "qwen_moe_groups_materialized"})
    )
    recipe_path = tmp_path / "qwen-artifact-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-artifact-audit-fixture",
                "description": "fixture",
                "external_inputs": {
                    "artifact": {"kind": "artifact_dir", "path": str(artifact_dir)},
                },
                "steps": [
                    {
                        "id": "artifact_audit",
                        "op": "qwen-moe-artifact-audit",
                        "class": "diagnostic",
                        "inputs": {"artifact": "external:artifact"},
                        "params": {
                            "expected_source_projection_groups": 80,
                            "expected_target_projection_groups": 120,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("artifact_audit").argv

    assert argv[3] == "benchmarks/audit_qwen_moe_materialization.py"
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact_dir)
    assert argv[argv.index("--expected-source-projection-groups") + 1] == "80"
    assert argv[argv.index("--expected-target-projection-groups") + 1] == "120"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_qwen_moe_bind_probe_op_renders_binding_probe(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "qwen-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "qwen-moe-materialization-manifest.json").write_text(
        json.dumps({"materialization_status": "qwen_moe_groups_materialized"})
    )
    recipe_path = tmp_path / "qwen-bind-probe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-bind-probe-fixture",
                "description": "fixture",
                "external_inputs": {
                    "artifact": {"kind": "artifact_dir", "path": str(artifact_dir)},
                },
                "steps": [
                    {
                        "id": "bind_probe",
                        "op": "qwen-moe-bind-probe",
                        "class": "diagnostic",
                        "inputs": {"artifact": "external:artifact"},
                        "params": {
                            "expected_layers": [0, 1],
                            "model_layer_count": 2,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("bind_probe").argv

    assert argv[3] == "benchmarks/probe_qwen_moe_binding.py"
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact_dir)
    assert argv[argv.index("--expected-layers") + 1] == "0,1"
    assert argv[argv.index("--model-layer-count") + 1] == "2"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_qwen_non_expert_bind_probe_op_renders_source_binding_probe(tmp_path: Path) -> None:
    source_dir = tmp_path / "qwen-source"
    source_dir.mkdir()
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {}}))
    recipe_path = tmp_path / "qwen-non-expert-bind-probe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen-non-expert-bind-probe-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "non_expert_bind_probe",
                        "op": "qwen-non-expert-bind-probe",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "expected_layers": [0, 1],
                            "max_tensors": 8,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("non_expert_bind_probe").argv

    assert argv[3] == "benchmarks/probe_qwen_non_expert_binding.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--expected-layers") + 1] == "0,1"
    assert argv[argv.index("--max-tensors") + 1] == "8"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_qwen_runtime_bind_forward_probe_op_renders_combined_probe(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    index_path = tmp_path / "source" / "model.safetensors.index.json"
    artifact = tmp_path / "artifact"
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "qwen_runtime_probe_test",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "artifact": {"kind": "artifact_dir", "path": str(artifact)},
                    "tokenizer_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                },
                "steps": [
                    {
                        "id": "runtime_bind_forward_probe",
                        "op": "qwen-runtime-bind-forward-probe",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                            "artifact": "external:artifact",
                            "tokenizer_dir": "external:tokenizer_dir",
                        },
                        "params": {
                            "expected_layers": [0, 1],
                            "forward_layers": [0],
                            "model_layer_count": 2,
                            "forward_top_k": 2,
                            "logit_probe_token_id": 2,
                            "logit_probe_top_k": 3,
                            "tokenizer_prompt": "hello world",
                            "tokenized_logit_position": "last",
                            "tokenized_logit_top_k": 4,
                            "max_tensors": 16,
                            "layer_only": True,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build", no_hash=True)
    argv = plan.step("runtime_bind_forward_probe").argv

    assert argv[3] == "benchmarks/probe_qwen_runtime_binding.py"
    assert argv[argv.index("--artifact-dir") + 1] == str(artifact)
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--tokenizer-dir") + 1] == str(source_dir)
    assert argv[argv.index("--expected-layers") + 1] == "0,1"
    assert argv[argv.index("--forward-layers") + 1] == "0"
    assert argv[argv.index("--model-layer-count") + 1] == "2"
    assert argv[argv.index("--forward-top-k") + 1] == "2"
    assert argv[argv.index("--logit-probe-token-id") + 1] == "2"
    assert argv[argv.index("--logit-probe-top-k") + 1] == "3"
    assert argv[argv.index("--tokenizer-prompt") + 1] == "hello world"
    assert argv[argv.index("--tokenized-logit-position") + 1] == "last"
    assert argv[argv.index("--tokenized-logit-top-k") + 1] == "4"
    assert argv[argv.index("--max-tensors") + 1] == "16"
    assert "--layer-only" in argv
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--append-jsonl") + 1].endswith("evidence.jsonl")
    assert "--overwrite" not in argv


def test_qwen36_materialization_probe_recipe_chains_audit_materialize_and_bind(
    tmp_path: Path,
) -> None:
    recipe = load_recipe(
        Path("recipes/qwen36_35b_a3b_materialization_probe_20260702.yaml")
    )

    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)

    assert [step.spec.id for step in plan.steps] == [
        "source_audit",
        "source_payload_audit",
        "materialize_groups",
        "artifact_audit",
        "bind_probe",
        "non_expert_bind_probe",
        "runtime_bind_forward_probe",
        "runtime_global_logit_probe",
        "upstream_transformer_probe",
        "tokenizer_readiness_probe",
        "tokenized_logits_probe",
        "qwen_family_eval_prompts",
        "qwen_family_policy",
        "qwen_family_eval_teacher_metadata",
        "qwen_family_eval_teacher_logits",
        "qwen_family_eval_candidate_logits",
        "qwen_family_eval_metrics",
        "qwen_family_eval_rows",
        "qwen_family_eval_gate",
        "qwen_family_benchmark_invariants",
        "qwen_family_candidate_benchmark_latency",
        "qwen_family_control_benchmark_latency",
        "qwen_family_benchmark_rows",
        "qwen_family_benchmark_gate",
        "qwen_family_gate",
    ]

    source_audit = plan.step("source_audit").argv
    source_payload_audit = plan.step("source_payload_audit").argv
    materialize = plan.step("materialize_groups").argv
    artifact_audit = plan.step("artifact_audit").argv
    bind_probe = plan.step("bind_probe").argv
    non_expert_bind_probe = plan.step("non_expert_bind_probe").argv
    runtime_probe = plan.step("runtime_bind_forward_probe").argv
    logit_probe = plan.step("runtime_global_logit_probe").argv
    upstream_probe = plan.step("upstream_transformer_probe").argv
    tokenizer_probe = plan.step("tokenizer_readiness_probe").argv
    tokenized_logits_probe = plan.step("tokenized_logits_probe").argv
    family_eval_prompts = plan.step("qwen_family_eval_prompts").argv
    family_policy = plan.step("qwen_family_policy").argv
    family_eval_teacher_metadata = plan.step("qwen_family_eval_teacher_metadata").argv
    family_eval_teacher_logits = plan.step("qwen_family_eval_teacher_logits").argv
    family_eval_candidate_logits = plan.step("qwen_family_eval_candidate_logits").argv
    family_eval_metrics = plan.step("qwen_family_eval_metrics").argv
    family_eval_rows = plan.step("qwen_family_eval_rows").argv
    family_eval_gate_probe = plan.step("qwen_family_eval_gate").argv
    family_benchmark_invariants = plan.step("qwen_family_benchmark_invariants").argv
    family_candidate_benchmark_latency = plan.step(
        "qwen_family_candidate_benchmark_latency"
    ).argv
    family_control_benchmark_latency = plan.step(
        "qwen_family_control_benchmark_latency"
    ).argv
    family_benchmark_rows = plan.step("qwen_family_benchmark_rows").argv
    family_benchmark_gate_probe = plan.step("qwen_family_benchmark_gate").argv
    family_gate = plan.step("qwen_family_gate").argv

    materialized_artifact = plan.step("materialize_groups").output_paths["artifact"]
    assert plan.step("artifact_audit").resolved_inputs["artifact"] == materialized_artifact
    assert plan.step("bind_probe").resolved_inputs["artifact"] == materialized_artifact
    assert plan.step("runtime_bind_forward_probe").resolved_inputs["artifact"] == materialized_artifact
    assert plan.step("runtime_global_logit_probe").resolved_inputs["artifact"] == materialized_artifact
    assert plan.step("upstream_transformer_probe").resolved_inputs["artifact"] == materialized_artifact
    assert plan.step("tokenized_logits_probe").resolved_inputs["artifact"] == materialized_artifact
    assert plan.step("tokenized_logits_probe").resolved_inputs["tokenizer_dir"] == Path(
        recipe.external_inputs["source_dir"].path
    )
    assert plan.step("tokenizer_readiness_probe").resolved_inputs["tokenizer_dir"] == Path(
        recipe.external_inputs["source_dir"].path
    )
    assert plan.step("qwen_family_gate").resolved_inputs["source_audit_json"] == plan.step(
        "source_audit"
    ).output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs["runtime_global_logit_json"] == plan.step(
        "runtime_global_logit_probe"
    ).output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs[
        "upstream_transformer_json"
    ] == plan.step("upstream_transformer_probe").output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs[
        "tokenizer_json"
    ] == plan.step("tokenizer_readiness_probe").output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs[
        "tokenized_logits_json"
    ] == plan.step("tokenized_logits_probe").output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_gate").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_gate").resolved_inputs[
        "eval_prompt_pack_json"
    ] == plan.step("qwen_family_eval_prompts").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_rows").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_rows").resolved_inputs[
        "eval_prompt_pack_json"
    ] == plan.step("qwen_family_eval_prompts").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_teacher_metadata").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_teacher_metadata").resolved_inputs[
        "eval_prompt_pack_json"
    ] == plan.step("qwen_family_eval_prompts").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_teacher_metadata").resolved_inputs[
        "tokenizer_dir"
    ] == Path(recipe.external_inputs["source_dir"].path)
    assert plan.step("qwen_family_eval_teacher_metadata").resolved_inputs[
        "source_dir"
    ] == Path(recipe.external_inputs["source_dir"].path)
    assert plan.step("qwen_family_eval_teacher_logits").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_teacher_logits").resolved_inputs[
        "eval_prompt_pack_json"
    ] == plan.step("qwen_family_eval_prompts").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_teacher_logits").resolved_inputs[
        "source_dir"
    ] == Path(recipe.external_inputs["source_dir"].path)
    assert plan.step("qwen_family_eval_teacher_logits").resolved_inputs[
        "index_path"
    ] == Path(recipe.external_inputs["index_path"].path)
    assert plan.step("qwen_family_eval_candidate_logits").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_candidate_logits").resolved_inputs[
        "eval_prompt_pack_json"
    ] == plan.step("qwen_family_eval_prompts").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_metrics").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_metrics").resolved_inputs[
        "eval_prompt_pack_json"
    ] == plan.step("qwen_family_eval_prompts").output_paths["evidence_json"]
    assert plan.step("qwen_family_eval_metrics").resolved_inputs[
        "candidate_logits_jsonl"
    ] == plan.step("qwen_family_eval_candidate_logits").output_paths[
        "candidate_logits_jsonl"
    ]
    assert plan.step("qwen_family_eval_metrics").resolved_inputs[
        "teacher_logits_jsonl"
    ] == plan.step("qwen_family_eval_teacher_logits").output_paths[
        "teacher_logits_jsonl"
    ]
    assert plan.step("qwen_family_eval_rows").resolved_inputs[
        "eval_metric_jsonl"
    ] == plan.step("qwen_family_eval_metrics").output_paths["metric_jsonl"]
    assert plan.step("qwen_family_eval_rows").resolved_inputs[
        "teacher_cache_metadata_jsonl"
    ] == plan.step("qwen_family_eval_teacher_metadata").output_paths["metadata_jsonl"]
    assert plan.step("qwen_family_eval_gate").resolved_inputs[
        "eval_jsonl"
    ] == plan.step("qwen_family_eval_rows").output_paths["eval_jsonl"]
    assert plan.step("qwen_family_eval_gate").resolved_inputs[
        "eval_row_probe_json"
    ] == plan.step("qwen_family_eval_rows").output_paths["evidence_json"]
    assert plan.step("qwen_family_benchmark_gate").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_benchmark_rows").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_benchmark_invariants").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_benchmark_invariants").resolved_inputs[
        "artifact_audit_json"
    ] == plan.step("artifact_audit").output_paths["evidence_json"]
    assert plan.step("qwen_family_benchmark_invariants").resolved_inputs[
        "non_expert_bind_json"
    ] == plan.step("non_expert_bind_probe").output_paths["evidence_json"]
    assert plan.step("qwen_family_candidate_benchmark_latency").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_candidate_benchmark_latency").resolved_inputs[
        "artifact"
    ] == plan.step("materialize_groups").output_paths["artifact"]
    assert plan.step("qwen_family_control_benchmark_latency").resolved_inputs[
        "family_policy_json"
    ] == plan.step("qwen_family_policy").output_paths["evidence_json"]
    assert plan.step("qwen_family_control_benchmark_latency").resolved_inputs[
        "source_dir"
    ] == Path(recipe.external_inputs["source_dir"].path)
    assert plan.step("qwen_family_control_benchmark_latency").resolved_inputs[
        "index_path"
    ] == Path(recipe.external_inputs["index_path"].path)
    assert plan.step("qwen_family_benchmark_rows").resolved_inputs[
        "candidate_latency_jsonl"
    ] == plan.step("qwen_family_candidate_benchmark_latency").output_paths[
        "latency_jsonl"
    ]
    assert plan.step("qwen_family_benchmark_rows").resolved_inputs[
        "control_latency_jsonl"
    ] == plan.step("qwen_family_control_benchmark_latency").output_paths[
        "latency_jsonl"
    ]
    assert plan.step("qwen_family_benchmark_rows").resolved_inputs[
        "candidate_invariant_jsonl"
    ] == plan.step("qwen_family_benchmark_invariants").output_paths[
        "candidate_invariant_jsonl"
    ]
    assert plan.step("qwen_family_benchmark_gate").resolved_inputs[
        "benchmark_jsonl"
    ] == plan.step("qwen_family_benchmark_rows").output_paths["benchmark_jsonl"]
    assert plan.step("qwen_family_benchmark_gate").resolved_inputs[
        "benchmark_row_probe_json"
    ] == plan.step("qwen_family_benchmark_rows").output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs[
        "family_eval_gate_json"
    ] == plan.step("qwen_family_eval_gate").output_paths["evidence_json"]
    assert plan.step("qwen_family_gate").resolved_inputs[
        "family_benchmark_gate_json"
    ] == plan.step("qwen_family_benchmark_gate").output_paths["evidence_json"]
    assert plan.step("non_expert_bind_probe").resolved_inputs["source_dir"] == Path(
        recipe.external_inputs["source_dir"].path
    )
    assert plan.step("non_expert_bind_probe").resolved_inputs["index_path"] == Path(
        recipe.external_inputs["index_path"].path
    )

    assert source_audit[3] == "benchmarks/inspect_qwen_moe_source.py"
    assert source_audit[source_audit.index("--model-id") + 1] == "Qwen/Qwen3.6-35B-A3B"
    assert "--verify-hf-shapes" in source_audit

    expected_layers = ",".join(str(layer) for layer in range(40))

    assert source_payload_audit[3] == "benchmarks/audit_qwen_moe_source_payloads.py"
    assert source_payload_audit[source_payload_audit.index("--max-groups") + 1] == "80"
    assert non_expert_bind_probe[3] == "benchmarks/probe_qwen_non_expert_binding.py"
    assert (
        non_expert_bind_probe[non_expert_bind_probe.index("--expected-layers") + 1]
        == expected_layers
    )
    assert non_expert_bind_probe[non_expert_bind_probe.index("--max-tensors") + 1] == "700"
    assert runtime_probe[3] == "benchmarks/probe_qwen_runtime_binding.py"
    assert runtime_probe[runtime_probe.index("--expected-layers") + 1] == expected_layers
    assert runtime_probe[runtime_probe.index("--forward-layers") + 1] == "0,39"
    assert runtime_probe[runtime_probe.index("--model-layer-count") + 1] == "40"
    assert runtime_probe[runtime_probe.index("--forward-top-k") + 1] == "8"
    assert runtime_probe[runtime_probe.index("--max-tensors") + 1] == "700"
    assert "--layer-only" in runtime_probe
    assert logit_probe[3] == "benchmarks/probe_qwen_runtime_binding.py"
    assert logit_probe[logit_probe.index("--expected-layers") + 1] == ""
    assert logit_probe[logit_probe.index("--model-layer-count") + 1] == "40"
    assert logit_probe[logit_probe.index("--max-tensors") + 1] == "2"
    assert logit_probe[logit_probe.index("--logit-probe-token-id") + 1] == "0"
    assert logit_probe[logit_probe.index("--logit-probe-top-k") + 1] == "5"
    assert "--layer-only" not in logit_probe
    assert upstream_probe[3] == "benchmarks/probe_qwen_upstream_transformer_binding.py"
    assert upstream_probe[upstream_probe.index("--expected-layers") + 1] == "0,39"
    assert upstream_probe[upstream_probe.index("--config-path") + 1] == str(
        Path(recipe.external_inputs["config_path"].path)
    )
    assert upstream_probe[upstream_probe.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )
    assert tokenizer_probe[3] == "benchmarks/probe_qwen_tokenizer_readiness.py"
    assert tokenizer_probe[tokenizer_probe.index("--tokenizer-dir") + 1] == str(
        Path(recipe.external_inputs["source_dir"].path)
    )
    assert tokenizer_probe[tokenizer_probe.index("--prompt") + 1] == "The capital of France is"
    assert tokenizer_probe[tokenizer_probe.index("--min-token-count") + 1] == "1"
    assert tokenizer_probe[tokenizer_probe.index("--expected-vocab-size") + 1] == "248320"
    assert tokenizer_probe[tokenizer_probe.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )
    assert tokenized_logits_probe[3] == "benchmarks/probe_qwen_runtime_binding.py"
    assert (
        tokenized_logits_probe[tokenized_logits_probe.index("--expected-layers") + 1]
        == ""
    )
    assert tokenized_logits_probe[tokenized_logits_probe.index("--model-layer-count") + 1] == "40"
    assert tokenized_logits_probe[tokenized_logits_probe.index("--max-tensors") + 1] == "2"
    assert tokenized_logits_probe[tokenized_logits_probe.index("--tokenizer-dir") + 1] == str(
        Path(recipe.external_inputs["source_dir"].path)
    )
    assert (
        tokenized_logits_probe[tokenized_logits_probe.index("--tokenizer-prompt") + 1]
        == "The capital of France is"
    )
    assert (
        tokenized_logits_probe[
            tokenized_logits_probe.index("--tokenized-logit-position") + 1
        ]
        == "last"
    )
    assert (
        tokenized_logits_probe[tokenized_logits_probe.index("--tokenized-logit-top-k") + 1]
        == "5"
    )
    assert "--layer-only" not in tokenized_logits_probe
    assert family_eval_prompts[3] == "benchmarks/prepare_qwen_family_eval_prompts.py"
    assert family_eval_prompts[family_eval_prompts.index("--tokenizer-dir") + 1] == str(
        Path(recipe.external_inputs["source_dir"].path)
    )
    assert (
        family_eval_prompts[family_eval_prompts.index("--model-id") + 1]
        == "Qwen/Qwen3.6-35B-A3B"
    )
    assert (
        family_eval_prompts[family_eval_prompts.index("--revision") + 1]
        == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    )
    assert family_eval_prompts[family_eval_prompts.index("--min-prompts-per-split") + 1] == "22"
    assert family_eval_prompts[family_eval_prompts.index("--min-tokens-per-prompt") + 1] == "2"
    assert family_eval_prompts[family_eval_prompts.index("--expected-vocab-size") + 1] == "248320"
    assert family_eval_prompts[family_eval_prompts.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )
    assert family_policy[3] == "benchmarks/define_qwen_family_gate_policy.py"
    assert (
        family_policy[family_policy.index("--model-id") + 1]
        == "Qwen/Qwen3.6-35B-A3B"
    )
    assert (
        family_policy[family_policy.index("--revision") + 1]
        == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    )
    assert family_policy[family_policy.index("--minimum-clean-rows-per-split") + 1] == "22"
    assert family_policy[family_policy.index("--minimum-total-clean-rows") + 1] == "64"
    assert family_policy[family_policy.index("--minimum-repetitions-per-scenario") + 1] == "2"
    assert (
        family_policy[family_policy.index("--comparison-baseline") + 1]
        == "same_machine_qwen_source_switch_projection_control"
    )
    assert (
        family_policy[
            family_policy.index("--maximum-candidate-to-reference-ratio") + 1
        ]
        == "3.0"
    )
    assert (
        family_eval_teacher_metadata[3]
        == "benchmarks/prepare_qwen_family_eval_teacher_metadata.py"
    )
    assert (
        family_eval_teacher_metadata[
            family_eval_teacher_metadata.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_eval_teacher_metadata[
            family_eval_teacher_metadata.index("--eval-prompt-pack-json") + 1
        ]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_eval_teacher_metadata[
            family_eval_teacher_metadata.index("--tokenizer-dir") + 1
        ]
        == str(Path(recipe.external_inputs["source_dir"].path))
    )
    assert (
        family_eval_teacher_metadata[
            family_eval_teacher_metadata.index("--source-dir") + 1
        ]
        == str(Path(recipe.external_inputs["source_dir"].path))
    )
    assert (
        family_eval_teacher_metadata[
            family_eval_teacher_metadata.index("--output-dir") + 1
        ]
        == str(
            plan.step("qwen_family_eval_teacher_metadata")
            .output_paths["metadata_jsonl"]
            .parent
        )
    )
    assert (
        family_eval_teacher_logits[3]
        == "benchmarks/prepare_qwen_family_eval_teacher_logits.py"
    )
    assert (
        family_eval_teacher_logits[
            family_eval_teacher_logits.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_eval_teacher_logits[
            family_eval_teacher_logits.index("--eval-prompt-pack-json") + 1
        ]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_eval_teacher_logits[family_eval_teacher_logits.index("--source-dir") + 1]
        == str(Path(recipe.external_inputs["source_dir"].path))
    )
    assert (
        family_eval_teacher_logits[family_eval_teacher_logits.index("--index-path") + 1]
        == str(Path(recipe.external_inputs["index_path"].path))
    )
    assert (
        family_eval_teacher_logits[
            family_eval_teacher_logits.index("--selected-token-position") + 1
        ]
        == "last"
    )
    assert family_eval_teacher_logits[family_eval_teacher_logits.index("--top-k") + 1] == "32"
    assert (
        family_eval_teacher_logits[family_eval_teacher_logits.index("--output-dir") + 1]
        == str(
            plan.step("qwen_family_eval_teacher_logits")
            .output_paths["teacher_logits_jsonl"]
            .parent
        )
    )
    assert (
        family_eval_candidate_logits[3]
        == "benchmarks/prepare_qwen_family_eval_candidate_logits.py"
    )
    assert (
        family_eval_candidate_logits[
            family_eval_candidate_logits.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_eval_candidate_logits[
            family_eval_candidate_logits.index("--eval-prompt-pack-json") + 1
        ]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_eval_candidate_logits[
            family_eval_candidate_logits.index("--output-dir") + 1
        ]
        == str(
            plan.step("qwen_family_eval_candidate_logits")
            .output_paths["candidate_logits_jsonl"]
            .parent
        )
    )
    assert family_eval_metrics[3] == "benchmarks/prepare_qwen_family_eval_metric_rows.py"
    assert (
        family_eval_metrics[family_eval_metrics.index("--family-policy-json") + 1]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_eval_metrics[family_eval_metrics.index("--eval-prompt-pack-json") + 1]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_eval_metrics[family_eval_metrics.index("--candidate-logits-jsonl") + 1]
        == str(
            plan.step("qwen_family_eval_candidate_logits").output_paths[
                "candidate_logits_jsonl"
            ]
        )
    )
    assert (
        family_eval_metrics[family_eval_metrics.index("--teacher-logits-jsonl") + 1]
        == str(
            plan.step("qwen_family_eval_teacher_logits").output_paths[
                "teacher_logits_jsonl"
            ]
        )
    )
    assert (
        family_eval_metrics[family_eval_metrics.index("--output-dir") + 1]
        == str(plan.step("qwen_family_eval_metrics").output_paths["metric_jsonl"].parent)
    )
    assert family_eval_rows[3] == "benchmarks/prepare_qwen_family_eval_rows.py"
    assert (
        family_eval_rows[family_eval_rows.index("--family-policy-json") + 1]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_eval_rows[family_eval_rows.index("--eval-prompt-pack-json") + 1]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_eval_rows[family_eval_rows.index("--eval-metric-jsonl") + 1]
        == str(plan.step("qwen_family_eval_metrics").output_paths["metric_jsonl"])
    )
    assert (
        family_eval_rows[family_eval_rows.index("--teacher-cache-metadata-jsonl") + 1]
        == str(
            plan.step("qwen_family_eval_teacher_metadata").output_paths[
                "metadata_jsonl"
            ]
        )
    )
    assert (
        family_eval_rows[family_eval_rows.index("--output-dir") + 1]
        == str(plan.step("qwen_family_eval_rows").output_paths["eval_jsonl"].parent)
    )
    assert family_eval_rows[family_eval_rows.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )
    assert family_eval_gate_probe[3] == "benchmarks/check_qwen_family_eval_gate.py"
    assert (
        family_eval_gate_probe[family_eval_gate_probe.index("--family-policy-json") + 1]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_eval_gate_probe[family_eval_gate_probe.index("--eval-prompt-pack-json") + 1]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_eval_gate_probe[family_eval_gate_probe.index("--eval-jsonl") + 1]
        == str(plan.step("qwen_family_eval_rows").output_paths["eval_jsonl"])
    )
    assert (
        family_eval_gate_probe[family_eval_gate_probe.index("--eval-row-probe-json") + 1]
        == str(plan.step("qwen_family_eval_rows").output_paths["evidence_json"])
    )
    assert family_benchmark_gate_probe[3] == "benchmarks/check_qwen_family_benchmark_gate.py"
    assert (
        family_benchmark_invariants[3]
        == "benchmarks/prepare_qwen_family_benchmark_invariants.py"
    )
    assert (
        family_benchmark_invariants[
            family_benchmark_invariants.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_benchmark_invariants[
            family_benchmark_invariants.index("--artifact-audit-json") + 1
        ]
        == str(plan.step("artifact_audit").output_paths["evidence_json"])
    )
    assert (
        family_benchmark_invariants[
            family_benchmark_invariants.index("--non-expert-bind-json") + 1
        ]
        == str(plan.step("non_expert_bind_probe").output_paths["evidence_json"])
    )
    assert (
        family_benchmark_invariants[family_benchmark_invariants.index("--output-dir") + 1]
        == str(
            plan.step("qwen_family_benchmark_invariants")
            .output_paths["candidate_invariant_jsonl"]
            .parent
        )
    )
    assert family_benchmark_invariants[
        family_benchmark_invariants.index("--output-json") + 1
    ].endswith("evidence_json.json")
    assert (
        family_candidate_benchmark_latency[3]
        == "benchmarks/bench_qwen_family_candidate_latency.py"
    )
    assert (
        family_candidate_benchmark_latency[
            family_candidate_benchmark_latency.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_candidate_benchmark_latency[
            family_candidate_benchmark_latency.index("--artifact-dir") + 1
        ]
        == str(plan.step("materialize_groups").output_paths["artifact"])
    )
    assert (
        family_candidate_benchmark_latency[
            family_candidate_benchmark_latency.index("--output-dir") + 1
        ]
        == str(
            plan.step("qwen_family_candidate_benchmark_latency")
            .output_paths["latency_jsonl"]
            .parent
        )
    )
    assert (
        family_candidate_benchmark_latency[
            family_candidate_benchmark_latency.index("--repetitions") + 1
        ]
        == "2"
    )
    assert (
        family_control_benchmark_latency[3]
        == "benchmarks/bench_qwen_family_control_latency.py"
    )
    assert (
        family_control_benchmark_latency[
            family_control_benchmark_latency.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_control_benchmark_latency[
            family_control_benchmark_latency.index("--source-dir") + 1
        ]
        == str(Path(recipe.external_inputs["source_dir"].path))
    )
    assert (
        family_control_benchmark_latency[
            family_control_benchmark_latency.index("--index-path") + 1
        ]
        == str(Path(recipe.external_inputs["index_path"].path))
    )
    assert (
        family_control_benchmark_latency[
            family_control_benchmark_latency.index("--output-dir") + 1
        ]
        == str(
            plan.step("qwen_family_control_benchmark_latency")
            .output_paths["latency_jsonl"]
            .parent
        )
    )
    assert family_benchmark_rows[3] == "benchmarks/prepare_qwen_family_benchmark_rows.py"
    assert (
        family_benchmark_rows[family_benchmark_rows.index("--family-policy-json") + 1]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    latency_jsonl_inputs = [
        family_benchmark_rows[index + 1]
        for index, value in enumerate(family_benchmark_rows)
        if value == "--latency-jsonl"
    ]
    assert latency_jsonl_inputs == [
        str(
            plan.step("qwen_family_candidate_benchmark_latency")
            .output_paths["latency_jsonl"]
        ),
        str(
            plan.step("qwen_family_control_benchmark_latency")
            .output_paths["latency_jsonl"]
        ),
    ]
    assert (
        family_benchmark_rows[family_benchmark_rows.index("--candidate-invariant-jsonl") + 1]
        == str(
            plan.step("qwen_family_benchmark_invariants")
            .output_paths["candidate_invariant_jsonl"]
        )
    )
    assert (
        family_benchmark_rows[family_benchmark_rows.index("--output-dir") + 1]
        == str(
            plan.step("qwen_family_benchmark_rows")
            .output_paths["benchmark_jsonl"]
            .parent
        )
    )
    assert family_benchmark_rows[
        family_benchmark_rows.index("--output-json") + 1
    ].endswith("evidence_json.json")
    assert (
        family_benchmark_gate_probe[
            family_benchmark_gate_probe.index("--family-policy-json") + 1
        ]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_benchmark_gate_probe[family_benchmark_gate_probe.index("--benchmark-jsonl") + 1]
        == str(plan.step("qwen_family_benchmark_rows").output_paths["benchmark_jsonl"])
    )
    assert (
        family_benchmark_gate_probe[
            family_benchmark_gate_probe.index("--benchmark-row-probe-json") + 1
        ]
        == str(plan.step("qwen_family_benchmark_rows").output_paths["evidence_json"])
    )
    assert family_gate[3] == "benchmarks/check_qwen_family_gate.py"
    assert (
        family_gate[family_gate.index("--source-audit-json") + 1]
        == str(plan.step("source_audit").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--source-payload-audit-json") + 1]
        == str(plan.step("source_payload_audit").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--artifact-audit-json") + 1]
        == str(plan.step("artifact_audit").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--runtime-forward-json") + 1]
        == str(plan.step("runtime_bind_forward_probe").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--upstream-transformer-json") + 1]
        == str(plan.step("upstream_transformer_probe").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--tokenizer-json") + 1]
        == str(plan.step("tokenizer_readiness_probe").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--tokenized-logits-json") + 1]
        == str(plan.step("tokenized_logits_probe").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--family-policy-json") + 1]
        == str(plan.step("qwen_family_policy").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--eval-prompt-pack-json") + 1]
        == str(plan.step("qwen_family_eval_prompts").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--family-eval-gate-json") + 1]
        == str(plan.step("qwen_family_eval_gate").output_paths["evidence_json"])
    )
    assert (
        family_gate[family_gate.index("--family-benchmark-gate-json") + 1]
        == str(plan.step("qwen_family_benchmark_gate").output_paths["evidence_json"])
    )
    assert family_gate[family_gate.index("--expected-layer-count") + 1] == "40"
    assert family_gate[family_gate.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )

    assert materialize[3] == "benchmarks/materialize_qwen_moe_group.py"
    assert "--all-groups" in materialize
    assert materialize[materialize.index("--max-groups") + 1] == "80"
    assert "--skip-existing" in materialize

    assert artifact_audit[3] == "benchmarks/audit_qwen_moe_materialization.py"
    assert artifact_audit[artifact_audit.index("--artifact-dir") + 1] == str(
        materialized_artifact
    )
    assert (
        artifact_audit[
            artifact_audit.index("--expected-source-projection-groups") + 1
        ]
        == "80"
    )
    assert (
        artifact_audit[
            artifact_audit.index("--expected-target-projection-groups") + 1
        ]
        == "120"
    )

    assert bind_probe[3] == "benchmarks/probe_qwen_moe_binding.py"
    assert bind_probe[bind_probe.index("--artifact-dir") + 1] == str(
        materialized_artifact
    )
    assert bind_probe[bind_probe.index("--expected-layers") + 1] == expected_layers
    assert bind_probe[bind_probe.index("--model-layer-count") + 1] == "40"


def test_glm52_validation_probe_recipe_renders_layer_local_checks(
    tmp_path: Path,
) -> None:
    recipe = load_recipe(
        Path("recipes/glm52_504b_vq_validation_probe_20260702.yaml")
    )

    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)

    assert [step.spec.id for step in plan.steps] == [
        "validate_layer3",
        "validate_layer77",
    ]
    layer3 = plan.step("validate_layer3").argv
    layer77 = plan.step("validate_layer77").argv

    for argv, layer in ((layer3, "3"), (layer77, "77")):
        assert argv[3] == "scripts/validate_glm52_vq.py"
        assert argv[argv.index("--model-id") + 1] == "zai-org/GLM-5.2"
        assert argv[argv.index("--revision") + 1] == (
            "f2263102df303b2faa54a6861a29d1770ce846c0"
        )
        assert argv[argv.index("--layer") + 1] == layer
        assert argv[argv.index("--tokens") + 1] == "1"
        assert argv[argv.index("--artifact-dir") + 1] == "artifacts/glm-5.2-vq"
        assert "--no-strict-config" not in argv
        assert "--min-source-weighted-cosine" not in argv


def test_pipeline_source_views_op_renders_materialize_command(tmp_path: Path) -> None:
    source_dir = tmp_path / "hf-source"
    index_path = source_dir / "model.safetensors.index.json"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    index_path.write_text(json.dumps({"weight_map": {}}))
    recipe_path = tmp_path / "pipeline-views.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "pipeline-views-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "pipeline_views",
                        "op": "pipeline-source-views",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "pipeline_size": 2,
                            "layer_split": 24,
                            "rank_budget_gb": 115,
                            "rank0_only_logits_views": True,
                            "materialize": True,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("pipeline_views").argv

    assert argv[3] == "benchmarks/materialize_glm45_air_pipeline_views.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--output-dir") + 1].endswith("/out")
    assert argv[argv.index("--pipeline-size") + 1] == "2"
    assert argv[argv.index("--layer-split") + 1] == "24"
    assert argv[argv.index("--rank-budget-gb") + 1] == "115"
    assert "--rank0-only-logits-views" in argv
    assert "--materialize" in argv
    assert "--allow-existing" not in argv


def test_pipeline_view_load_probe_op_renders_evidence_command(tmp_path: Path) -> None:
    source_dir = tmp_path / "hf-source"
    index_path = source_dir / "model.safetensors.index.json"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    index_path.write_text(json.dumps({"weight_map": {}}))
    recipe_path = tmp_path / "pipeline-probe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "pipeline-probe-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "pipeline_views",
                        "op": "pipeline-source-views",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "pipeline_size": 2,
                            "layer_split": 24,
                            "rank_budget_gb": 115,
                            "materialize": True,
                        },
                    },
                    {
                        "id": "pipeline_view_probe",
                        "op": "pipeline-view-load-probe",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "index_path": "external:index_path",
                            "view_dir": "step:pipeline_views/artifact",
                        },
                        "params": {
                            "pipeline_size": 2,
                            "layer_split": 24,
                            "rank_budget_gb": 115,
                            "reuse_existing": True,
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    probe = plan.step("pipeline_view_probe")
    argv = probe.argv

    assert argv[3] == "benchmarks/probe_glm45_air_pipeline_view_load.py"
    assert argv[argv.index("--source-dir") + 1] == str(source_dir)
    assert argv[argv.index("--index-path") + 1] == str(index_path)
    assert argv[argv.index("--output-dir") + 1] == str(plan.step("pipeline_views").out_dir)
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert "--reuse-existing" in argv
    assert "--overwrite" not in argv


def test_jaccl_hostfile_op_renders_two_rank_hostfile_writer(tmp_path: Path) -> None:
    recipe_path = tmp_path / "jaccl-hostfile.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "jaccl-hostfile-fixture",
                "description": "fixture",
                "steps": [
                    {
                        "id": "hostfile",
                        "op": "jaccl-hostfile",
                        "class": "diagnostic",
                        "params": {
                            "local_ip": "198.51.100.1",
                            "local_rdma_device": "rdma_en1",
                            "peer_ssh": "jackmazac@203.0.113.191",
                            "peer_rdma_device": "rdma_en1",
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("hostfile")
    argv = step.argv

    assert step.output_paths["hostfile"] == step.out_dir / "hostfile.json"
    assert argv[3] == "benchmarks/write_jaccl_hostfile.py"
    assert argv[argv.index("--output-dir") + 1] == str(step.out_dir)
    assert argv[argv.index("--local-ip") + 1] == "198.51.100.1"
    assert argv[argv.index("--local-rdma-device") + 1] == "rdma_en1"
    assert argv[argv.index("--peer-ssh") + 1] == "jackmazac@203.0.113.191"
    assert argv[argv.index("--peer-rdma-device") + 1] == "rdma_en1"


def test_rdma_topology_audit_op_renders_non_sudo_diagnostic(tmp_path: Path) -> None:
    recipe_path = tmp_path / "rdma-topology-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "rdma-topology-audit-fixture",
                "description": "fixture",
                "steps": [
                    {
                        "id": "topology",
                        "op": "rdma-topology-audit",
                        "class": "diagnostic",
                        "params": {
                            "local_if": "en1",
                            "peer_if": "en1",
                            "local_ip": "198.51.100.1",
                            "peer_ip": "198.51.100.2",
                            "local_rdma_device": "rdma_en1",
                            "peer_rdma_device": "rdma_en1",
                            "peer_ssh": "jackmazac@203.0.113.191",
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("topology")
    argv = step.argv

    assert argv[3] == "benchmarks/audit_glm45_air_rdma_topology.py"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--local-if") + 1] == "en1"
    assert argv[argv.index("--peer-if") + 1] == "en1"
    assert argv[argv.index("--local-ip") + 1] == "198.51.100.1"
    assert argv[argv.index("--peer-ip") + 1] == "198.51.100.2"
    assert argv[argv.index("--local-rdma-device") + 1] == "rdma_en1"
    assert argv[argv.index("--peer-rdma-device") + 1] == "rdma_en1"
    assert argv[argv.index("--peer-ssh") + 1] == "jackmazac@203.0.113.191"
    assert "--sudo" not in argv
    assert "--allow-existing" not in argv
    assert "--overwrite" not in argv


def test_single_host_teacher_source_audit_op_renders_local_diagnostic(tmp_path: Path) -> None:
    recipe_path = tmp_path / "single-host-source-audit.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "single-host-source-audit-fixture",
                "description": "fixture",
                "steps": [
                    {
                        "id": "source_audit",
                        "op": "single-host-teacher-source-audit",
                        "class": "diagnostic",
                        "params": {
                            "model_path": "/models/glm45-air-bf16",
                            "revision": "local",
                            "teacher_kind": "bf16_source",
                            "scan_root": "/models",
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("source_audit")
    argv = step.argv

    assert argv[3] == "benchmarks/audit_glm45_air_single_host_teacher_source.py"
    assert argv[argv.index("--output-json") + 1].endswith("evidence_json.json")
    assert argv[argv.index("--model-path") + 1] == "/models/glm45-air-bf16"
    assert argv[argv.index("--revision") + 1] == "local"
    assert argv[argv.index("--teacher-kind") + 1] == "bf16_source"
    assert argv[argv.index("--scan-root") + 1] == "/models"
    assert "--peer-ssh" not in argv
    assert "--jaccl-hostfile" not in argv
    assert "--sudo" not in argv


def test_export_teacher_cache_cleanroom_can_consume_pipeline_view_dir(
    tmp_path: Path,
) -> None:
    recipe_path = tmp_path / "cleanroom-cache-from-views.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "cleanroom-cache-from-views-fixture",
                "description": "fixture",
                "steps": [
                    {
                        "id": "pipeline_views",
                        "op": "pipeline-source-views",
                        "class": "promotable",
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                            "pipeline_size": 2,
                            "layer_split": 24,
                            "rank_budget_gb": 115,
                            "materialize": True,
                        },
                    },
                    {
                        "id": "report_cache",
                        "op": "export-teacher-cache-cleanroom",
                        "class": "promotable",
                        "inputs": {
                            "rank_view_dir": "step:pipeline_views/artifact",
                        },
                        "params": {
                            "prompt_set": "air_vq_ladder_report_v1",
                            "prompt_id": ["report_math_001", "report_code_001"],
                            "top_k": 128,
                            "max_positions": 128,
                            "layer_split": 24,
                            "mlx_wired_limit_gb": 1,
                            "required_wired_mb": 0,
                            "skip_direct_link_check": True,
                            "preflight_only": True,
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    views = plan.step("pipeline_views")
    cache = plan.step("report_cache")
    argv = cache.argv

    assert cache.resolved_inputs["rank_view_dir"] == views.out_dir
    assert argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert argv[argv.index("--rank-view-dir") + 1] == str(views.out_dir)
    assert "--rank-view-roots-json" not in argv
    assert "--preflight-only" in argv
    assert "--skip-direct-link-check" in argv


def test_export_teacher_cache_cleanroom_renders_single_host_fallback(
    tmp_path: Path,
) -> None:
    recipe_path = tmp_path / "cleanroom-cache-single-host.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "cleanroom-cache-single-host-fixture",
                "description": "fixture",
                "steps": [
                    {
                        "id": "report_cache",
                        "op": "export-teacher-cache-cleanroom",
                        "class": "diagnostic",
                        "params": {
                            "prompt_set": "air_vq_ladder_report_v1",
                            "prompt_id": ["report_math_001"],
                            "top_k": 128,
                            "max_positions": 128,
                            "preflight_only": True,
                            "single_host_fallback": True,
                            "single_host_model_path": "/models/glm45-air-bf16",
                            "single_host_revision": "local",
                            "single_host_teacher_kind": "bf16_source",
                            "single_host_lazy": True,
                            "single_host_source_memory_guard_ratio": 1.0,
                            "single_host_pipeline_local": True,
                            "single_host_pipeline_local_stage_processes": True,
                            "single_host_pipeline_local_head_process": True,
                            "single_host_pipeline_local_lower_split_layer": 12,
                            "single_host_pipeline_local_upper_split_layer": 34,
                            "mlx_cache_limit_gb": 0,
                            "skip_local_consume": True,
                            "allow_dirty_cache": True,
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    cache = plan.step("report_cache")
    argv = cache.argv

    assert argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert "--single-host-fallback" in argv
    assert argv[argv.index("--single-host-model-path") + 1] == "/models/glm45-air-bf16"
    assert argv[argv.index("--single-host-revision") + 1] == "local"
    assert argv[argv.index("--single-host-teacher-kind") + 1] == "bf16_source"
    assert "--single-host-lazy" in argv
    assert argv[argv.index("--single-host-source-memory-guard-ratio") + 1] == "1.0"
    assert "--single-host-pipeline-local" in argv
    assert "--single-host-pipeline-local-stage-processes" in argv
    assert "--single-host-pipeline-local-head-process" in argv
    assert argv[argv.index("--single-host-pipeline-local-lower-split-layer") + 1] == "12"
    assert argv[argv.index("--single-host-pipeline-local-upper-split-layer") + 1] == "34"
    assert argv[argv.index("--mlx-cache-limit-gb") + 1] == "0"
    assert "--skip-local-consume" in argv
    assert "--allow-dirty-cache" in argv
    assert "--jaccl-hostfile" not in argv
    assert "--rank-view-dir" not in argv
    assert "--preflight-only" in argv


def test_pipeline_view_ops_can_resolve_pinned_source_without_source_dir(
    tmp_path: Path,
) -> None:
    recipe_path = tmp_path / "pipeline-source-resolve.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "pipeline-source-resolve-fixture",
                "description": "fixture",
                "steps": [
                    {
                        "id": "pipeline_views",
                        "op": "pipeline-source-views",
                        "class": "diagnostic",
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                            "pipeline_size": 2,
                            "layer_split": 24,
                            "rank_budget_gb": 115,
                            "materialize": True,
                        },
                    },
                    {
                        "id": "pipeline_view_probe",
                        "op": "pipeline-view-load-probe",
                        "class": "diagnostic",
                        "inputs": {
                            "view_dir": "step:pipeline_views/artifact",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                            "pipeline_size": 2,
                            "layer_split": 24,
                            "rank_budget_gb": 115,
                            "reuse_existing": True,
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    views_argv = plan.step("pipeline_views").argv
    probe_argv = plan.step("pipeline_view_probe").argv

    assert views_argv[3] == "benchmarks/materialize_glm45_air_pipeline_views.py"
    assert views_argv[views_argv.index("--model-id") + 1] == "zai-org/GLM-4.5-Air"
    assert views_argv[views_argv.index("--revision") + 1] == (
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert "--source-dir" not in views_argv
    assert "--index-path" not in views_argv
    assert probe_argv[3] == "benchmarks/probe_glm45_air_pipeline_view_load.py"
    assert probe_argv[probe_argv.index("--output-dir") + 1] == str(
        plan.step("pipeline_views").out_dir
    )
    assert "--source-dir" not in probe_argv
    assert "--index-path" not in probe_argv
    assert "--reuse-existing" in probe_argv


def test_legacy_lineage_recipe_includes_pipeline_view_probe() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__legacy_lineage_20260701.yaml")
    )
    plan = plan_recipe(recipe, build_root=Path("artifacts/build") / recipe.name)
    views = plan.step("legacy_pipeline_source_views")
    probe = plan.step("legacy_pipeline_view_probe")

    views_argv = views.argv
    probe_argv = probe.argv
    assert views.spec.op == "pipeline-source-views"
    assert probe.spec.op == "pipeline-view-load-probe"
    assert views_argv[3] == "benchmarks/materialize_glm45_air_pipeline_views.py"
    assert views_argv[views_argv.index("--model-id") + 1] == "zai-org/GLM-4.5-Air"
    assert views_argv[views_argv.index("--revision") + 1] == (
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert "--source-dir" not in views_argv
    assert "--index-path" not in views_argv
    assert "--materialize" in views_argv
    assert probe_argv[3] == "benchmarks/probe_glm45_air_pipeline_view_load.py"
    assert probe_argv[probe_argv.index("--output-dir") + 1] == str(views.out_dir)
    assert probe_argv[probe_argv.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )
    assert "--reuse-existing" in probe_argv
    assert "--source-dir" not in probe_argv
    assert "--index-path" not in probe_argv


def test_legacy_lineage_recipe_renders_cleanroom_cache_preflight_boundary() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__legacy_lineage_20260701.yaml")
    )
    plan = plan_recipe(recipe, build_root=Path("artifacts/build") / recipe.name)
    views = plan.step("legacy_pipeline_source_views")
    stage_views = plan.step("legacy_local_sequential_stage_views")
    topology = plan.step("legacy_rdma_topology_audit")
    hostfile = plan.step("legacy_jaccl_hostfile")
    distributed_prefix = plan.step("legacy_report_cache_distributed_prefix")
    report = plan.step("legacy_report_cache_preflight")
    selection = plan.step("legacy_selection_cache_preflight")
    report_prefix = plan.step("legacy_report_cache_single_host_prefix")
    report_fallback = plan.step("legacy_report_cache_single_host_preflight")
    selection_fallback = plan.step("legacy_selection_cache_single_host_preflight")
    step_ids = [step.spec.id for step in plan.steps]
    assert step_ids.index("legacy_report_cache_single_host_prefix") < step_ids.index(
        "legacy_report_cache_distributed_prefix"
    )

    assert topology.spec.op == "rdma-topology-audit"
    assert topology.argv[3] == "benchmarks/audit_glm45_air_rdma_topology.py"
    assert topology.argv[topology.argv.index("--local-if") + 1] == "en1"
    assert topology.argv[topology.argv.index("--peer-if") + 1] == "en1"
    assert topology.argv[topology.argv.index("--local-ip") + 1] == "198.51.100.1"
    assert topology.argv[topology.argv.index("--peer-ip") + 1] == "198.51.100.2"
    assert topology.argv[topology.argv.index("--peer-ssh") + 1] == (
        "jackmazac@203.0.113.191"
    )
    assert "--require-ready" in topology.argv

    assert hostfile.spec.op == "jaccl-hostfile"
    assert hostfile.output_paths["hostfile"] == hostfile.out_dir / "hostfile.json"
    assert hostfile.argv[3] == "benchmarks/write_jaccl_hostfile.py"
    assert hostfile.argv[hostfile.argv.index("--local-ip") + 1] == "198.51.100.1"
    assert hostfile.argv[hostfile.argv.index("--peer-ssh") + 1] == (
        "jackmazac@203.0.113.191"
    )

    distributed_prefix_argv = distributed_prefix.argv
    assert distributed_prefix.spec.op == "export-teacher-cache-cleanroom"
    assert distributed_prefix.spec.step_class == "diagnostic"
    assert distributed_prefix.resolved_inputs["rank_view_dir"] == views.out_dir
    assert distributed_prefix.resolved_inputs["jaccl_hostfile"] == (
        hostfile.out_dir / "hostfile.json"
    )
    assert distributed_prefix_argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--rank-view-dir") + 1
    ] == str(views.out_dir)
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--jaccl-hostfile") + 1
    ] == str(hostfile.out_dir / "hostfile.json")
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--prompt-set") + 1
    ] == "base"
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--prompt-id") + 1
    ] == "capital_france"
    assert distributed_prefix_argv[distributed_prefix_argv.index("--top-k") + 1] == "16"
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--max-positions") + 1
    ] == "1"
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--layer-split") + 1
    ] == "23"
    assert "--single-host-fallback" not in distributed_prefix_argv
    assert "--no-full-logits" in distributed_prefix_argv
    assert "--skip-local-consume" in distributed_prefix_argv
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--peer-source-stage-ssh") + 1
    ] == "jackmazac@198.51.100.2"
    assert "--stage-peer-source" in distributed_prefix_argv
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--peer-source-min-free-gb") + 1
    ] == "16"
    assert "--require-memory-quiet-preflight" in distributed_prefix_argv
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--memory-quiet-min-free-gb") + 1
    ] == "24"
    assert "--preflight-only" not in distributed_prefix_argv

    for step, prompt_set in (
        (report, "air_vq_ladder_report_v1"),
        (selection, "air_vq_ladder_select_v1"),
    ):
        argv = step.argv
        assert step.spec.op == "export-teacher-cache-cleanroom"
        assert step.spec.step_class == "diagnostic"
        assert step.resolved_inputs["rank_view_dir"] == views.out_dir
        assert step.resolved_inputs["jaccl_hostfile"] == hostfile.out_dir / "hostfile.json"
        assert argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
        assert argv[argv.index("--rank-view-dir") + 1] == str(views.out_dir)
        assert argv[argv.index("--jaccl-hostfile") + 1] == str(
            hostfile.out_dir / "hostfile.json"
        )
        assert argv[argv.index("--prompt-set") + 1] == prompt_set
        assert argv[argv.index("--required-wired-mb") + 1] == "0"
        assert "--mlx-wired-limit-gb" not in argv
        assert argv[argv.index("--local-direct-if") + 1] == "en1"
        assert argv[argv.index("--peer-direct-if") + 1] == "en1"
        assert "--preflight-only" in argv
        assert "--rank-view-roots-json" not in argv

    prefix_argv = report_prefix.argv
    assert report_prefix.spec.op == "export-teacher-cache-cleanroom"
    assert report_prefix.spec.step_class == "diagnostic"
    assert report_prefix.resolved_inputs["rank_view_dir"] == views.out_dir
    assert report_prefix.resolved_inputs["local_sequential_stage_view_roots_json"] == (
        stage_views.out_dir / "local_sequential_stage_view_roots.json"
    )
    assert prefix_argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert prefix_argv[prefix_argv.index("--rank-view-dir") + 1] == str(views.out_dir)
    assert prefix_argv[
        prefix_argv.index("--local-sequential-stage-view-roots-json") + 1
    ] == str(stage_views.out_dir / "local_sequential_stage_view_roots.json")
    assert prefix_argv[prefix_argv.index("--prompt-set") + 1] == "base"
    assert prefix_argv[prefix_argv.index("--prompt-id") + 1] == "capital_france"
    assert prefix_argv[prefix_argv.index("--top-k") + 1] == "16"
    assert prefix_argv[prefix_argv.index("--max-positions") + 1] == "1"
    assert prefix_argv[prefix_argv.index("--layer-split") + 1] == "23"
    assert "--single-host-fallback" in prefix_argv
    assert "--single-host-pipeline-local" in prefix_argv
    assert "--single-host-pipeline-local-stage-processes" in prefix_argv
    assert "--single-host-pipeline-local-head-process" in prefix_argv
    assert (
        prefix_argv[
            prefix_argv.index("--single-host-pipeline-local-lower-split-layers") + 1
        ]
        == "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22"
    )
    assert (
        prefix_argv[
            prefix_argv.index("--single-host-pipeline-local-upper-split-layers") + 1
        ]
        == "24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45"
    )
    assert "--single-host-pipeline-local-abort-on-dirty-stage" in prefix_argv
    assert "--single-host-pipeline-local-lower-split-layer" not in prefix_argv
    assert "--single-host-pipeline-local-upper-split-layer" not in prefix_argv
    assert "--no-full-logits" in prefix_argv
    assert "--skip-local-consume" in prefix_argv
    assert "--allow-dirty-cache" not in prefix_argv
    assert "--require-memory-quiet-preflight" in prefix_argv
    assert prefix_argv[prefix_argv.index("--memory-quiet-seconds") + 1] == "3"
    assert prefix_argv[prefix_argv.index("--memory-quiet-max-attempts") + 1] == "20"
    assert "--memory-quiet-min-free-gb" not in prefix_argv
    assert prefix_argv[prefix_argv.index("--memory-quiet-stage-view-margin-gb") + 1] == "8"
    assert prefix_argv[prefix_argv.index("--peer-ssh") + 1] == (
        "jackmazac@203.0.113.191"
    )
    assert prefix_argv[prefix_argv.index("--peer-rank0-view-root") + 1] == (
        "glm45-air-pipeline-views/rank-0-p2"
    )
    assert prefix_argv[prefix_argv.index("--peer-source-dir") + 1] == (
        "/Users/jackmazac/.cache/huggingface/hub/"
        "models--zai-org--GLM-4.5-Air/snapshots/"
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert prefix_argv[prefix_argv.index("--peer-source-stage-ssh") + 1] == (
        "jackmazac@198.51.100.2"
    )
    assert "--stage-peer-source" in prefix_argv
    assert prefix_argv[prefix_argv.index("--peer-source-min-free-gb") + 1] == "16"
    assert prefix_argv[prefix_argv.index("--local-sequential-remote-workers") + 1] == (
        "head"
    )
    assert prefix_argv[prefix_argv.index("--local-sequential-remote-tmp-dir") + 1] == (
        "/tmp/glm-p2-remote-workers"
    )
    assert prefix_argv[
        prefix_argv.index("--local-sequential-remote-dirty-retries") + 1
    ] == "1"
    assert prefix_argv[
        prefix_argv.index("--local-sequential-remote-retry-sleep-seconds") + 1
    ] == "30"
    assert "--preflight-only" not in prefix_argv

    for step, prompt_set in (
        (report_fallback, "air_vq_ladder_report_v1"),
        (selection_fallback, "air_vq_ladder_select_v1"),
    ):
        argv = step.argv
        assert step.spec.op == "export-teacher-cache-cleanroom"
        assert step.spec.step_class == "diagnostic"
        assert step.resolved_inputs == {}
        assert argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
        assert argv[argv.index("--prompt-set") + 1] == prompt_set
        assert argv[argv.index("--vq-artifact-dir") + 1] == "artifacts/glm-4.5-air-vq"
        assert "--single-host-fallback" in argv
        assert "--preflight-only" in argv
        assert "--jaccl-hostfile" not in argv
        assert "--peer-ssh" not in argv
        assert "--rank-view-dir" not in argv


def test_target_recipe_has_resident_byte_audit_diagnostic() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)

    resident = plan.step("resident_byte_audit")
    assert resident.spec.step_class == "diagnostic"
    assert resident.resolved_inputs["artifact"] == plan.step("train_l45_gud_math8").out_dir
    assert resident.argv[3] == "benchmarks/audit_glm45_air_resident_bytes.py"
    assert resident.argv[resident.argv.index("--artifact-dir") + 1] == str(
        plan.step("train_l45_gud_math8").out_dir
    )
    assert resident.argv[resident.argv.index("--output-json") + 1].endswith(
        "evidence_json.json"
    )
    assert resident.argv[resident.argv.index("--append-jsonl") + 1].endswith(
        "evidence.jsonl"
    )
    assert "--source-dir" not in resident.argv
    assert "--allow-existing" not in resident.argv
    assert "--overwrite" not in resident.argv


def test_target_recipe_eval_steps_render_lower_residency_policy() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)

    for step_id in ("eval_report", "eval_selection", "eval_holdout"):
        argv = plan.step(step_id).argv
        assert argv[3] == "benchmarks/eval_glm45_air_teacher_cache.py"
        assert argv[argv.index("--mlx-cache-limit-gb") + 1] == "0"
        assert "--mlx-clear-cache-before-load" in argv
        assert "--truncate-input-to-selected-positions" not in argv


def test_target_recipe_repairs_split_evidence_before_promotion() -> None:
    recipe = load_recipe(
        Path("recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml")
    )
    plan = plan_recipe(recipe, no_hash=True)

    for split in ("report", "selection", "holdout"):
        raw_step_id = f"eval_{split}"
        repair_step_id = f"{raw_step_id}_repair"
        raw_step = plan.step(raw_step_id)
        repair_step = plan.step(repair_step_id)

        assert raw_step.spec.gate is None
        assert repair_step.spec.gate is not None
        assert repair_step.spec.gate.profile == "balanced_rc_split"
        assert repair_step.argv[3] == "benchmarks/repair_glm45_air_eval_evidence.py"
        assert repair_step.argv[repair_step.argv.index("--existing-jsonl") + 1] == str(
            raw_step.output_paths["evidence"]
        )
        assert repair_step.argv[repair_step.argv.index("--repair-attempts") + 1] == "2"
        assert repair_step.argv[repair_step.argv.index("--mlx-cache-limit-gb") + 1] == "0"
        assert "--mlx-clear-cache-before-load" in repair_step.argv

    assert [gate.evidence_step for gate in recipe.promotion.gates[:3]] == [
        "eval_report_repair",
        "eval_selection_repair",
        "eval_holdout_repair",
    ]


def test_base_build_producer_ops_render_deterministic_argv(tmp_path: Path) -> None:
    source_dir = tmp_path / "hf-source"
    source_dir.mkdir()
    (source_dir / "model-00001-of-00001.safetensors").write_bytes(b"source")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "glm4_moe"}))
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}))

    recipe_path = tmp_path / "base-build.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "base-build-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "base_vq",
                        "op": "stream-convert-vq",
                        "class": "promotable",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "group_size": 512,
                            "code_bits": 8,
                            "code_bits_policy": ["*:down_proj=16"],
                            "scale_estimator": "max_abs",
                            "expert_workers": 4,
                        },
                    },
                    {
                        "id": "report_cache",
                        "op": "export-teacher-cache-cleanroom",
                        "class": "promotable",
                        "params": {
                            "prompt_set": "air_vq_ladder_report_v1",
                            "prompt_id": ["report_math_001", "report_code_001"],
                            "top_k": 128,
                            "max_positions": 128,
                            "rank_view_roots_json": '{"0":"/rank0","1":"/rank1"}',
                            "include_route_trace": True,
                            "route_trace_layer": [31, 36, 41],
                            "mlx_wired_limit_gb": 120,
                        },
                    },
                    {
                        "id": "seed_train",
                        "op": "train-low-rank",
                        "class": "promotable",
                        "inputs": {
                            "seed_artifact": "step:base_vq/artifact",
                            "selection_teacher": "step:report_cache/teacher_cache",
                            "validation_teacher": "step:report_cache/teacher_cache",
                        },
                        "params": {
                            "layer": 45,
                            "projections": ["gate_proj", "up_proj", "down_proj"],
                            "trainable": "low_rank_residual",
                            "low_rank": 4,
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")

    convert = plan.step("base_vq").argv
    assert convert[:4] == ["uv", "run", "python", "scripts/plan_stream_convert.py"]
    assert "--convert-all" in convert
    assert convert[convert.index("--source-dir") + 1] == str(source_dir)
    assert convert[convert.index("--output-dir") + 1] == str(plan.step("base_vq").out_dir)
    assert convert[convert.index("--config-path") + 1] == str(config_path)
    assert convert[convert.index("--index-path") + 1] == str(index_path)
    assert convert[convert.index("--model-id") + 1] == "zai-org/GLM-4.5-Air"
    assert convert[convert.index("--code-bits-policy") + 1] == "*:down_proj=16"
    assert "--overwrite" not in convert

    cache = plan.step("report_cache")
    cache_argv = cache.argv
    assert cache_argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/export_glm45_air_cleanroom_cache.py",
    ]
    assert cache.output_paths["teacher_cache"] == cache.out_dir / "metadata.jsonl"
    assert cache_argv[cache_argv.index("--output-dir") + 1] == str(cache.out_dir)
    assert cache_argv[cache_argv.index("--validation-jsonl") + 1].startswith(
        str(cache.evidence_dir)
    )
    assert cache_argv[cache_argv.index("--local-jsonl") + 1].startswith(
        str(cache.evidence_dir)
    )
    first_prompt = cache_argv.index("--prompt-id")
    assert cache_argv[first_prompt + 1] == "report_math_001"
    assert cache_argv[first_prompt + 2 : first_prompt + 4] == [
        "--prompt-id",
        "report_code_001",
    ]
    assert cache_argv[cache_argv.index("--include-route-trace") + 1 :].count(
        "--route-trace-layer"
    ) == 3

    train = plan.step("seed_train")
    assert train.resolved_inputs["seed_artifact"] == plan.step("base_vq").out_dir
    assert train.resolved_inputs["selection_teacher"] == cache.out_dir / "metadata.jsonl"
    assert train.resolved_inputs["selection_cache_root"] == cache.out_dir
    assert train.resolved_inputs["validation_cache_root"] == cache.out_dir


def test_stream_convert_vq_renders_without_source_dir_input(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "glm4_moe"}))
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}))

    recipe_path = tmp_path / "hf-default-source.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "hf-default-source",
                "description": "fixture",
                "external_inputs": {
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                },
                "steps": [
                    {
                        "id": "base_vq",
                        "op": "stream-convert-vq",
                        "class": "promotable",
                        "inputs": {
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                            "download_missing_source_shards": True,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")

    convert = plan.step("base_vq").argv
    assert "--source-dir" not in convert
    assert convert[convert.index("--index-path") + 1] == str(index_path)
    assert convert[convert.index("--download-missing-source-shards") :].count(
        "--download-missing-source-shards"
    ) == 1


def test_collect_imatrix_feeds_materialize_sweep_with_manifest_file(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    high_bit = tmp_path / "high-bit"
    baseline.mkdir()
    high_bit.mkdir()

    recipe_path = tmp_path / "collect-imatrix.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "collect-imatrix-fixture",
                "description": "fixture",
                "external_inputs": {
                    "baseline": {"kind": "artifact_dir", "path": str(baseline)},
                    "high_bit": {"kind": "artifact_dir", "path": str(high_bit)},
                },
                "steps": [
                    {
                        "id": "collect",
                        "op": "collect-imatrix",
                        "class": "promotable",
                        "inputs": {"artifact": "external:baseline"},
                        "params": {
                            "prompt_set": "air_imatrix_calib_v1",
                            "prompt_id": [
                                "imatrix_calib_code_debug_000",
                                "imatrix_calib_code_tests_001",
                            ],
                            "layers": [1, 2],
                            "projections": ["gate_proj"],
                        },
                    },
                    {
                        "id": "materialize",
                        "op": "materialize-sweep",
                        "class": "promotable",
                        "inputs": {
                            "imatrix_manifest": "step:collect/imatrix_manifest",
                            "baseline_artifact": "external:baseline",
                            "high_bit_artifact": "external:high_bit",
                        },
                        "params": {"budget": [3.0]},
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")

    collect = plan.step("collect")
    materialize_argv = plan.step("materialize").argv
    manifest_path = collect.out_dir / "imatrix-manifest.json"
    assert collect.output_paths["imatrix_manifest"] == manifest_path
    collect_argv = collect.argv
    prompt_flags = [
        collect_argv[index + 1]
        for index, value in enumerate(collect_argv)
        if value == "--prompt-id"
    ]
    assert prompt_flags == [
        "imatrix_calib_code_debug_000",
        "imatrix_calib_code_tests_001",
    ]
    assert materialize_argv[materialize_argv.index("--imatrix-manifest") + 1] == str(
        manifest_path
    )


def test_ebss_route_records_feed_selection_and_collect_imatrix(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    recipe_path = tmp_path / "ebss-imatrix.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "ebss-imatrix-fixture",
                "description": "fixture",
                "external_inputs": {
                    "baseline": {"kind": "artifact_dir", "path": str(baseline)}
                },
                "steps": [
                    {
                        "id": "route_records",
                        "op": "collect-route-records",
                        "class": "promotable",
                        "inputs": {"artifact": "external:baseline"},
                        "params": {
                            "prompt_set": "air_imatrix_calib_v1",
                            "max_prompts": 8,
                            "layers": [1, 2],
                        },
                    },
                    {
                        "id": "select",
                        "op": "select-ebss-prompts",
                        "class": "promotable",
                        "inputs": {"route_records": "step:route_records/route_records"},
                        "params": {"max_prompts": 4},
                    },
                    {
                        "id": "collect",
                        "op": "collect-imatrix",
                        "class": "promotable",
                        "inputs": {
                            "artifact": "external:baseline",
                            "prompt_selection": "step:select/selection_json",
                        },
                        "params": {
                            "prompt_set": "air_imatrix_calib_v1",
                            "layers": [1, 2],
                            "projections": ["gate_proj"],
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")

    route_records = plan.step("route_records")
    assert route_records.argv[3] == "benchmarks/collect_glm45_air_route_records.py"
    assert route_records.output_paths["route_records"].name == "route_records.jsonl"

    select = plan.step("select")
    assert select.argv[3] == "benchmarks/select_glm45_air_ebss_prompts.py"
    assert select.argv[select.argv.index("--records-jsonl") + 1] == str(
        route_records.output_paths["route_records"]
    )
    assert select.output_paths["selection_json"].name == "selection_json.json"

    collect = plan.step("collect")
    assert collect.argv[collect.argv.index("--prompt-selection-json") + 1] == str(
        select.output_paths["selection_json"]
    )


def test_materialize_sweep_renders_importance_key_for_agq(tmp_path: Path) -> None:
    (tmp_path / "imatrix-manifest.json").write_text(
        json.dumps({"schema_version": 1, "record_type": "air_projection_imatrix_manifest", "entry_count": 0, "entries": []}),
        encoding="utf-8",
    )
    (tmp_path / "baseline").mkdir()
    (tmp_path / "high-bit").mkdir()
    recipe_path = tmp_path / "agq-materialize.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "agq-materialize-fixture",
                "description": "fixture",
                "external_inputs": {
                    "imatrix": {"kind": "file", "path": str(tmp_path / "imatrix-manifest.json")},
                    "baseline": {"kind": "artifact_dir", "path": str(tmp_path / "baseline")},
                    "high_bit": {"kind": "artifact_dir", "path": str(tmp_path / "high-bit")},
                },
                "steps": [
                    {
                        "id": "materialize",
                        "op": "materialize-sweep",
                        "class": "promotable",
                        "inputs": {
                            "imatrix_manifest": "external:imatrix",
                            "baseline_artifact": "external:baseline",
                            "high_bit_artifact": "external:high_bit",
                        },
                        "params": {
                            "budget": 3.0,
                            "candidate_prefix": "air-agq-affinity",
                            "importance_key": "affinity_weighted_importance",
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    argv = plan.step("materialize").argv

    assert argv[3] == "benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py"
    assert "--importance-key" in argv
    assert argv[argv.index("--importance-key") + 1] == "affinity_weighted_importance"


def test_materialize_learned_rotation_renders_linked_artifact_materializer(tmp_path: Path) -> None:
    for name in ("source", "seed"):
        (tmp_path / name).mkdir()
    for name in ("config.json", "model.safetensors.index.json", "rotation-manifest.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    recipe_path = tmp_path / "learned-rotation-materialize.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "learned-rotation-materialize-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source": {"kind": "artifact_dir", "path": str(tmp_path / "source")},
                    "config": {"kind": "file", "path": str(tmp_path / "config.json")},
                    "index": {"kind": "file", "path": str(tmp_path / "model.safetensors.index.json")},
                    "seed": {"kind": "artifact_dir", "path": str(tmp_path / "seed")},
                    "rotation_manifest": {
                        "kind": "file",
                        "path": str(tmp_path / "rotation-manifest.json"),
                    },
                },
                "steps": [
                    {
                        "id": "materialize_rotation",
                        "op": "materialize-learned-rotation",
                        "class": "promotable",
                        "inputs": {
                            "source_dir": "external:source",
                            "config_path": "external:config",
                            "index_path": "external:index",
                            "seed_artifact": "external:seed",
                            "rotation_manifest": "external:rotation_manifest",
                        },
                        "params": {
                            "model_id": "zai-org/GLM-4.5-Air",
                            "group_size": 512,
                            "code_bits": 8,
                            "expert_workers": 2,
                        },
                    }
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("materialize_rotation")
    argv = step.argv

    assert argv[3] == "benchmarks/materialize_glm45_air_learned_rotation.py"
    assert argv[argv.index("--source-dir") + 1] == str(tmp_path / "source")
    assert argv[argv.index("--seed-artifact-dir") + 1] == str(tmp_path / "seed")
    assert argv[argv.index("--rotation-manifest") + 1] == str(tmp_path / "rotation-manifest.json")
    assert argv[argv.index("--expert-workers") + 1] == "2"
    assert step.output_paths["artifact"] == step.out_dir


def test_train_learned_rotation_feeds_materializer_rotation_manifest(tmp_path: Path) -> None:
    for name in ("source", "seed"):
        (tmp_path / name).mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    recipe_path = tmp_path / "learned-rotation-chain.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "learned-rotation-chain-fixture",
                "description": "fixture",
                "external_inputs": {
                    "source": {"kind": "artifact_dir", "path": str(tmp_path / "source")},
                    "config": {"kind": "file", "path": str(tmp_path / "config.json")},
                    "index": {"kind": "file", "path": str(tmp_path / "model.safetensors.index.json")},
                    "seed": {"kind": "artifact_dir", "path": str(tmp_path / "seed")},
                },
                "steps": [
                    {
                        "id": "train_rotation",
                        "op": "train-learned-rotation",
                        "class": "promotable",
                        "inputs": {
                            "source_dir": "external:source",
                            "config_path": "external:config",
                            "index_path": "external:index",
                        },
                        "params": {
                            "target": ["41:gate_proj", "41:up_proj"],
                            "steps": 4,
                            "learning_rate": 0.05,
                            "max_experts": 1,
                            "max_rows_per_expert": 8,
                            "structured_rht_signs": True,
                            "sign_candidates_per_step": 16,
                        },
                    },
                    {
                        "id": "materialize_rotation",
                        "op": "materialize-learned-rotation",
                        "class": "promotable",
                        "inputs": {
                            "source_dir": "external:source",
                            "config_path": "external:config",
                            "index_path": "external:index",
                            "seed_artifact": "external:seed",
                            "rotation_manifest": "step:train_rotation/rotation_manifest",
                        },
                    },
                ],
            }
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    train = plan.step("train_rotation")
    materialize = plan.step("materialize_rotation")

    assert train.argv[3] == "benchmarks/train_glm45_air_learned_rotation.py"
    assert [train.argv[index + 1] for index, flag in enumerate(train.argv) if flag == "--target"] == [
        "41:gate_proj",
        "41:up_proj",
    ]
    assert "--structured-rht-signs" in train.argv
    assert train.argv[train.argv.index("--sign-candidates-per-step") + 1] == "16"
    assert train.output_paths["rotation_manifest"].name == "rotation-manifest.json"
    assert train.output_paths["training_evidence"].name == "training_evidence.jsonl"
    assert materialize.argv[materialize.argv.index("--rotation-manifest") + 1] == str(
        train.output_paths["rotation_manifest"]
    )


def test_expect_hash_mismatch_warns_or_errors(tmp_path: Path) -> None:
    recipe_path = _write_tiny_lineage(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["seed_artifact"]["expect_hash"] = "0" * 64
    recipe_path.write_text(yaml.safe_dump(raw))
    recipe = load_recipe(recipe_path)

    plan = plan_recipe(recipe, build_root=tmp_path / "build")
    assert any("hash mismatch" in warning for warning in plan.warnings)

    from mlx_vq.build.recipe import RecipeError

    with pytest.raises(RecipeError, match="hash mismatch"):
        plan_recipe(recipe, build_root=tmp_path / "build", strict_inputs=True)


def test_recipe_can_require_strict_external_input_hashes(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text("{}\n")
    recipe_path = tmp_path / "strict-inputs.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "strict_inputs": True,
                "name": "strict-inputs-fixture",
                "external_inputs": {
                    "authority": {
                        "kind": "file",
                        "path": str(authority),
                        "expect_hash": "0" * 64,
                    }
                },
                "steps": [],
            }
        )
    )

    recipe = load_recipe(recipe_path)
    assert recipe.strict_inputs is True
    with pytest.raises(RecipeError, match="hash mismatch"):
        plan_recipe(recipe, build_root=tmp_path / "build")
    with pytest.raises(RecipeError, match="does not allow --no-hash"):
        plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)


def test_top1_gap_op_renders_tail_cleanup_target_selection(tmp_path: Path) -> None:
    evidence = tmp_path / "eval.jsonl"
    evidence.write_text(json.dumps({"row_index": 120}) + "\n", encoding="utf-8")
    recipe_path = tmp_path / "top1-gap-tail.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "top1-gap-tail-fixture",
                "external_inputs": {
                    "eval_evidence": {"kind": "file", "path": str(evidence)},
                },
                "steps": [
                    {
                        "id": "tail_targets",
                        "op": "top1-gap",
                        "class": "diagnostic",
                        "inputs": {"evidence": "external:eval_evidence"},
                        "params": {
                            "select_tail_cleanup_targets": True,
                            "teacher_token_id": 565,
                            "focus_position": 0,
                            "max_train_rows": 8,
                            "max_validation_rows": 12,
                            "top_tail_records": 30,
                            "label": "route4_tail_targets",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    argv = (
        plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
        .step("tail_targets")
        .argv
    )

    assert argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/analyze_glm45_air_top1_gap.py",
    ]
    assert "--select-tail-cleanup-targets" in argv
    assert argv[argv.index("--teacher-token-id") + 1] == "565"
    assert argv[argv.index("--max-train-rows") + 1] == "8"
    assert argv[argv.index("--max-validation-rows") + 1] == "12"
    assert argv[argv.index("--top-tail-records") + 1] == "30"
