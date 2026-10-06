from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import save_file

from mlx_vq.build.lineage import build_legacy_lineage_recipe
from mlx_vq.build.ops import REGISTRY
from mlx_vq.build.recipe import parse_recipe, validate_recipe
from mlx_vq.build.runner import plan_recipe


def _write_manifest(
    artifact_dir: Path,
    *,
    seed_artifact_dir: Path,
    run: dict,
) -> None:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "format": "glm45_air_continuous_sidecar_v1",
                    "schema_version": 1,
                    "seed_artifact_dir": str(seed_artifact_dir),
                    "run": run,
                    "sidecars": [],
                }
            }
        )
    )


def _write_scaled_manifest(
    artifact_dir: Path,
    *,
    seed_artifact_dir: Path,
    source_artifact: Path,
    scale_policy: dict[str, float],
) -> None:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "format": "switch_linear_scale_delta_output_bias",
                    "schema_version": 1,
                    "seed_artifact_dir": str(seed_artifact_dir),
                    "run": {
                        "kind": "scaled_block_local_sidecar_alpha_sweep",
                        "seed_artifact_dir": str(seed_artifact_dir),
                        "source_artifact": str(source_artifact),
                        "scale_policy": scale_policy,
                        "trainable": "output_bias",
                    },
                    "sidecars": [
                        {
                            "layer": 18,
                            "projection": "gate_proj",
                            "path": "continuous_params/layer-00018-gate_proj.safetensors",
                            "tensors": ["output_bias"],
                        }
                    ],
                }
            }
        )
    )


def _write_merged_sidecar_manifest(
    artifact_dir: Path,
    *,
    seed_artifact_dir: Path,
    source_artifacts: list[Path],
    train_split: str = "selection",
) -> None:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "format": "switch_linear_scale_delta_output_bias",
                    "schema_version": 1,
                    "seed_artifact_dir": str(seed_artifact_dir),
                    "run": {
                        "kind": "merged_block_local_sidecars",
                        "seed_artifact_dir": str(seed_artifact_dir),
                        "source_artifacts": [str(path) for path in source_artifacts],
                        "layers": [18, 41],
                        "train_split": train_split,
                        "train_row_count": 8,
                        "max_positions": 8,
                        "trainable": "output_bias",
                    },
                    "sidecars": [
                        {
                            "layer": 18,
                            "projection": "gate_proj",
                            "path": "continuous_params/layer-00018-gate_proj.safetensors",
                            "tensors": ["output_bias"],
                        },
                        {
                            "layer": 41,
                            "projection": "up_proj",
                            "path": "continuous_params/layer-00041-up_proj.safetensors",
                            "tensors": ["output_bias"],
                        },
                    ],
                }
            }
        )
    )


def _write_block_local_fit_manifest(
    artifact_dir: Path,
    *,
    seed_artifact_dir: Path,
    layer: int,
    projection: str,
) -> None:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "enabled": True,
                    "format": "switch_linear_scale_delta_output_bias",
                    "schema_version": 1,
                    "seed_artifact_dir": str(seed_artifact_dir),
                    "run": {
                        "kind": "block_local_sidecar_least_squares",
                        "seed_artifact_dir": str(seed_artifact_dir),
                        "layer": layer,
                        "projection": projection,
                        "trainable": "output_bias",
                        "train_row_count": 8,
                    },
                    "sidecars": [
                        {
                            "layer": layer,
                            "projection": projection,
                            "path": f"continuous_params/layer-{layer:05d}-{projection}.safetensors",
                            "tensors": ["output_bias"],
                        }
                    ],
                }
            }
        )
    )


def _write_block_local_fit_record(
    quality_root: Path,
    *,
    output_dir: Path,
    seed_artifact_dir: Path,
    source_dir: Path,
    teacher_jsonl: Path,
    layer: int,
    projection: str,
) -> None:
    quality_root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "record_type": "air_vq_block_local_sidecar_fit",
        "output_dir": str(output_dir),
        "seed_artifact_dir": str(seed_artifact_dir),
        "source_dir": str(source_dir),
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
        "teacher_jsonl": [str(teacher_jsonl)],
        "teacher_cache_root": [str(teacher_jsonl.parent)],
        "layer": layer,
        "projection": projection,
        "trainable": "output_bias",
        "train_row_count": 8,
        "train_row_indices": None,
        "max_positions": 8,
        "min_top_k": 128,
    }
    filename = f"fit-{layer}-{projection}-{output_dir.name}.jsonl"
    (quality_root / filename).write_text(json.dumps(record, sort_keys=True) + "\n")


def _write_dynamic_imatrix_candidate_summary(
    artifact_dir: Path,
    *,
    baseline_artifact_dir: Path,
    high_bit_artifact_dir: Path,
    imatrix_manifest: Path,
    budget: float,
) -> None:
    artifact_dir.mkdir(parents=True)
    candidate_name = f"air-dynamic-imatrix-bpw-{budget:.1f}".replace(".", "p")
    candidate_summary = {
        "schema_version": 1,
        "record_type": "air_dynamic_imatrix_candidate_materialization",
        "candidate_name": candidate_name,
        "candidate_dir": str(artifact_dir),
        "requested_budget_bits_per_weight": budget,
        "effective_bits_per_weight": budget,
        "baseline_artifact_dir": str(baseline_artifact_dir),
        "high_bit_artifact_dir": str(high_bit_artifact_dir),
        "imatrix_manifest": str(imatrix_manifest),
    }
    (artifact_dir / "dynamic-materialization-summary.json").write_text(
        json.dumps(candidate_summary),
        encoding="utf-8",
    )
    (artifact_dir.parent / "dynamic-imatrix-sweep-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "air_dynamic_imatrix_budget_sweep_materialization",
                "imatrix_manifest": str(imatrix_manifest),
                "baseline_artifact_dir": str(baseline_artifact_dir),
                "high_bit_artifact_dir": str(high_bit_artifact_dir),
                "output_root": str(artifact_dir.parent),
                "mid_error_factor": 0.5,
                "reround_top_low_experts": 1,
                "candidates": [candidate_summary],
            }
        ),
        encoding="utf-8",
    )


def _write_stream_convert_manifest(
    artifact_dir: Path,
    *,
    model_id: str = "zai-org/GLM-4.5-Air",
    revision: str = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
    code_bits: int = 16,
    group_size: int = 512,
    scale_estimator: str = "max_abs",
    expert_workers: int = 16,
) -> None:
    artifact_dir.mkdir(parents=True)
    groups = [
        {
            "status": "converted",
            "code_bits": code_bits,
            "group_size": group_size,
            "scale_estimator": scale_estimator,
            "expert_workers": expert_workers,
            "output_path": str(artifact_dir / "layer-00001-gate_proj.safetensors"),
        }
    ]
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "planned_vq_groups": 1,
                "ready_vq_groups": 1,
                "converted_vq_groups": 1,
                "existing_vq_groups": 0,
                "skipped_vq_groups": 0,
                "scale_estimator": scale_estimator,
                "code_bits_policy": {},
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "conversion-run-max1-w16.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "index_path": (
                    "/Users/jack.mazac/.cache/huggingface/hub/"
                    f"models--zai-org--GLM-4.5-Air/snapshots/{revision}/"
                    "model.safetensors.index.json"
                ),
                "scale_estimator": scale_estimator,
                "expert_workers": expert_workers,
                "manifest": {
                    "planned_vq_groups": 1,
                    "ready_vq_groups": 1,
                    "converted_vq_groups": 1,
                    "existing_vq_groups": 0,
                    "skipped_vq_groups": 0,
                    "scale_estimator": scale_estimator,
                    "code_bits_policy": {},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_legacy_existing_air_vq_manifest(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True)
    groups = []
    for layer in range(1, 46):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            group_size = 352 if projection == "down_proj" else 512
            output_path = artifact_dir / f"layer-{layer:05d}-{projection}.safetensors"
            save_file(
                {"fixture": np.zeros((1,), dtype=np.uint8)},
                output_path,
                metadata={
                    "quantization_config": json.dumps(
                        {
                            "quant_method": "mlx_vq_e8",
                            "version": 1,
                            "default_code_bits": 8,
                            "default_group_size": group_size,
                            "policy": {},
                        }
                    )
                },
            )
            groups.append(
                {
                    "status": "existing",
                    "output_path": str(output_path),
                    "codes_name": f"model.layers.{layer}.mlp.switch_mlp.{projection}.codes",
                    "scales_name": f"model.layers.{layer}.mlp.switch_mlp.{projection}.scales",
                    "source_tensors_read": 0,
                    "peak_source_tensor_bytes": 0,
                }
            )
    (artifact_dir / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "model_id": "zai-org/GLM-4.5-Air",
                "planned_vq_groups": 135,
                "ready_vq_groups": 135,
                "converted_vq_groups": 0,
                "existing_vq_groups": 135,
                "skipped_vq_groups": 0,
                "scale_estimator": None,
                "code_bits_policy": None,
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )


def _write_imatrix_manifest(
    path: Path,
    *,
    prompt_set: str = "air_imatrix_calib_v1",
    layers: tuple[int, ...] = (1, 2),
    projections: tuple[str, ...] = ("gate_proj", "up_proj"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for layer in layers:
        for projection in projections:
            entries.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "expert": 0,
                    "input_dim": 16,
                    "path": f"imatrix/layer-{layer:05d}-{projection}-expert-00000.safetensors",
                    "prompt_ids": ["imatrix_calib_code_debug_000"],
                    "route_count": 1,
                    "route_frequency": 1.0,
                    "total_route_count": 1,
                    "tensors": {"importance_sum": "fixture"},
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "air_projection_imatrix_manifest",
                "prompt_set": prompt_set,
                "entry_count": len(entries),
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _run(**overrides) -> dict:
    base = {
        "kind": "rung2_continuous_finetune_probe",
        "layer": 45,
        "trainable": "output_bias",
        "steps": 48,
        "learning_rate": 1.0,
        "target_nll_weight": 0.5,
        "teacher_top1_margin_weight": 2.0,
        "teacher_top1_margin": 1.0,
        "loss_scope": "final_layer_selected",
        "surrogate_projections": ["gate_proj", "up_proj", "down_proj"],
        "surrogate_output_chunk_size": 256,
        "train_cache": "both",
        "max_train_rows": 8,
        "train_row_indices": [3, 5],
        "max_positions": 16,
        "prefill_engine": "auto",
        "final_train_loss": 1.23,
    }
    base.update(overrides)
    return base


def test_legacy_lineage_recipe_reconstructs_continuous_training_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root-seed"
    root.mkdir()
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")

    mid = tmp_path / "artifact-r1"
    target = tmp_path / "artifact-r2"
    _write_manifest(
        mid,
        seed_artifact_dir=root,
        run=_run(projection="gate_proj", train_row_indices=[]),
    )
    _write_manifest(
        target,
        seed_artifact_dir=mid,
        run=_run(
            projection=None,
            projections=["gate_proj", "up_proj", "down_proj"],
            trainable="low_rank_residual",
            low_rank=4,
            low_rank_init_scale=0.05,
            steps=12,
            learning_rate=0.5,
            train_row_indices=[44, 45],
        ),
    )

    raw = build_legacy_lineage_recipe(
        target,
        name="lineage-fixture",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert raw["external_inputs"]["root_artifact"]["path"] == str(root)
    assert raw["lineage"]["target_artifact"] == str(target)
    assert raw["lineage"]["root_artifact"] == str(root)
    assert raw["lineage"]["step_count"] == 2
    assert raw["lineage"]["warnings"] == [
        f"stopped at {root}: no supported legacy continuous manifest"
    ]
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_train_01",
        "legacy_train_02",
    ]
    assert raw["steps"][0]["inputs"]["seed_artifact"] == "external:root_artifact"
    assert raw["steps"][0]["params"]["projection"] == "gate_proj"
    assert "projections" not in raw["steps"][0]["params"]
    assert raw["steps"][1]["inputs"]["seed_artifact"] == "step:legacy_train_01/artifact"
    assert raw["steps"][1]["params"]["projections"] == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert raw["steps"][1]["params"]["low_rank"] == 4

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    first_argv = plan.step("legacy_train_01").argv
    second_argv = plan.step("legacy_train_02").argv
    assert first_argv[first_argv.index("--projection") + 1] == "gate_proj"
    assert "--projections" not in first_argv
    assert second_argv[second_argv.index("--projections") + 1 :][0:3] == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_legacy_lineage_recipe_includes_scaled_block_local_sidecar_bridge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dynamic-imatrix-bpw-3p0"
    root.mkdir()
    source = tmp_path / "l18-l41-gate-up-output-bias-select8x8"
    source.mkdir()
    scaled = tmp_path / "l18-l41-gate-up-output-bias-select8x8-l41alpha0p75"
    target = tmp_path / "artifact-r1"
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")

    _write_scaled_manifest(
        scaled,
        seed_artifact_dir=root,
        source_artifact=source,
        scale_policy={"layer_18": 1.0, "layer_41": 0.75},
    )
    _write_manifest(
        target,
        seed_artifact_dir=scaled,
        run=_run(projection="up_proj", train_row_indices=[1, 2]),
    )

    raw = build_legacy_lineage_recipe(
        target,
        name="lineage-scaled-fixture",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert raw["external_inputs"]["root_artifact"]["path"] == str(root)
    assert raw["external_inputs"]["legacy_scale_source_01"]["path"] == str(source)
    assert raw["lineage"]["root_artifact"] == str(root)
    assert raw["lineage"]["step_count"] == 2
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_scale_sidecars_01",
        "legacy_train_02",
    ]
    assert raw["steps"][0]["op"] == "scale-block-local-sidecars"
    assert raw["steps"][0]["inputs"] == {
        "seed_artifact": "external:root_artifact",
        "source_artifact": "external:legacy_scale_source_01",
    }
    assert raw["steps"][0]["params"]["scale_policy_json"] == json.dumps(
        {"layer_18": 1.0, "layer_41": 0.75},
        separators=(",", ":"),
        sort_keys=True,
    )
    assert raw["steps"][1]["inputs"]["seed_artifact"] == "step:legacy_scale_sidecars_01/artifact"

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    scale_argv = plan.step("legacy_scale_sidecars_01").argv
    assert "benchmarks/materialize_glm45_air_scaled_sidecars.py" in scale_argv
    assert scale_argv[scale_argv.index("--source-artifact-dir") + 1] == str(source)
    assert scale_argv[scale_argv.index("--scale-policy-json") + 1] == json.dumps(
        {"layer_18": 1.0, "layer_41": 0.75},
        separators=(",", ":"),
        sort_keys=True,
    )


def test_legacy_lineage_recipe_rebuilds_merged_block_local_sidecar_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dynamic-imatrix-bpw-3p0"
    root.mkdir()
    source_l18 = tmp_path / "l18-gate-output-bias-select8x8"
    source_l41 = tmp_path / "l41-up-output-bias-select8x8"
    source_l18.mkdir()
    source_l41.mkdir()
    merged_source = tmp_path / "l18-l41-gate-up-output-bias-select8x8"
    scaled = tmp_path / "l18-l41-gate-up-output-bias-select8x8-l41alpha0p75"
    target = tmp_path / "artifact-r1"
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")

    _write_merged_sidecar_manifest(
        merged_source,
        seed_artifact_dir=root,
        source_artifacts=[source_l18, source_l41],
    )
    _write_scaled_manifest(
        scaled,
        seed_artifact_dir=root,
        source_artifact=merged_source,
        scale_policy={"layer_18": 1.0, "layer_41": 0.75},
    )
    _write_manifest(
        target,
        seed_artifact_dir=scaled,
        run=_run(projection="up_proj", train_row_indices=[1, 2]),
    )

    raw = build_legacy_lineage_recipe(
        target,
        name="lineage-merged-source-fixture",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert "legacy_scale_source_01" not in raw["external_inputs"]
    assert raw["external_inputs"]["legacy_merge_source_01_01"]["path"] == str(source_l18)
    assert raw["external_inputs"]["legacy_merge_source_01_02"]["path"] == str(source_l41)
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_merge_sidecars_01",
        "legacy_scale_sidecars_01",
        "legacy_train_02",
    ]
    merge_step = raw["steps"][0]
    assert merge_step["op"] == "merge-block-local-sidecars"
    assert merge_step["inputs"] == {
        "seed_artifact": "external:root_artifact",
        "source_artifact_1": "external:legacy_merge_source_01_01",
        "source_artifact_2": "external:legacy_merge_source_01_02",
    }
    assert merge_step["params"] == {
        "train_split": "selection",
        "train_row_count": 8,
        "max_positions": 8,
        "trainable": "output_bias",
    }
    assert raw["steps"][1]["inputs"]["source_artifact"] == (
        "step:legacy_merge_sidecars_01/artifact"
    )

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    merge_argv = plan.step("legacy_merge_sidecars_01").argv
    source_flags = [
        merge_argv[index + 1]
        for index, value in enumerate(merge_argv)
        if value == "--source-artifact-dir"
    ]
    assert source_flags == [str(source_l18), str(source_l41)]
    scale_argv = plan.step("legacy_scale_sidecars_01").argv
    assert scale_argv[scale_argv.index("--source-artifact-dir") + 1] == str(
        plan.step("legacy_merge_sidecars_01").out_dir
    )


def test_legacy_lineage_recipe_rebuilds_block_local_fit_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("dynamic-imatrix-bpw-3p0")
    root.mkdir()
    revision = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    source_dir = Path("hf") / "snapshots" / revision
    source_dir.mkdir(parents=True)
    source_l18_gate = Path("block-local") / "l18-gate-output-bias-select8x8"
    source_l18_up = Path("block-local") / "l18-up-output-bias-select8x8"
    l18_merged_seed = Path("block-local") / "l18-gate-up-output-bias-select8x8"
    source_l41_gate = Path("block-local") / "l18ob-l41-gate-output-bias-select8x8"
    source_l41_up = Path("block-local") / "l18ob-l41-up-output-bias-select8x8"
    merged_source = Path("block-local") / "l18-l41-gate-up-output-bias-select8x8"
    scaled = Path("block-local") / "l18-l41-gate-up-output-bias-select8x8-l41alpha0p75"
    target = Path("artifact-r1")
    teacher_report = Path("report") / "metadata.jsonl"
    teacher_selection = Path("selection") / "metadata.jsonl"
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")

    _write_block_local_fit_manifest(
        source_l18_gate,
        seed_artifact_dir=root,
        layer=18,
        projection="gate_proj",
    )
    _write_block_local_fit_manifest(
        source_l18_up,
        seed_artifact_dir=root,
        layer=18,
        projection="up_proj",
    )
    _write_merged_sidecar_manifest(
        l18_merged_seed,
        seed_artifact_dir=root,
        source_artifacts=[source_l18_gate, source_l18_up],
    )
    _write_block_local_fit_manifest(
        source_l41_gate,
        seed_artifact_dir=l18_merged_seed,
        layer=41,
        projection="gate_proj",
    )
    _write_block_local_fit_manifest(
        source_l41_up,
        seed_artifact_dir=l18_merged_seed,
        layer=41,
        projection="up_proj",
    )
    _write_merged_sidecar_manifest(
        merged_source,
        seed_artifact_dir=root,
        source_artifacts=[
            source_l18_gate,
            source_l18_up,
            source_l41_gate,
            source_l41_up,
        ],
    )
    _write_scaled_manifest(
        scaled,
        seed_artifact_dir=root,
        source_artifact=merged_source,
        scale_policy={"layer_18": 1.0, "layer_41": 0.75},
    )
    _write_manifest(
        target,
        seed_artifact_dir=scaled,
        run=_run(projection="up_proj", train_row_indices=[1, 2]),
    )
    for artifact, seed, layer, projection in (
        (source_l18_gate, root, 18, "gate_proj"),
        (source_l18_up, root, 18, "up_proj"),
        (source_l41_gate, l18_merged_seed, 41, "gate_proj"),
        (source_l41_up, l18_merged_seed, 41, "up_proj"),
    ):
        _write_block_local_fit_record(
            Path("artifacts") / "quality",
            output_dir=artifact,
            seed_artifact_dir=seed,
            source_dir=source_dir,
            teacher_jsonl=teacher_selection,
            layer=layer,
            projection=projection,
        )

    raw = build_legacy_lineage_recipe(
        target,
        name="lineage-block-local-fits-fixture",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert not any(
        name.startswith("legacy_merge_source_01")
        for name in raw["external_inputs"]
    )
    assert not any(
        name.startswith("legacy_fit_source_dir_01")
        for name in raw["external_inputs"]
    )
    ops = [step["op"] for step in raw["steps"]]
    assert ops[:8] == [
        "fit-block-local-sidecar",
        "fit-block-local-sidecar",
        "merge-block-local-sidecars",
        "fit-block-local-sidecar",
        "fit-block-local-sidecar",
        "merge-block-local-sidecars",
        "scale-block-local-sidecars",
        "train-low-rank",
    ]
    l41_fit = raw["steps"][3]
    assert l41_fit["inputs"]["seed_artifact"] == "step:legacy_merge_sidecars_01_03/artifact"
    assert raw["steps"][5]["inputs"]["source_artifact_4"] == (
        "step:legacy_fit_sidecar_01_05/artifact"
    )
    assert raw["steps"][6]["inputs"]["source_artifact"] == (
        "step:legacy_merge_sidecars_01/artifact"
    )

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=Path("build"), no_hash=True)
    fit_argv = plan.step("legacy_fit_sidecar_01_01").argv
    assert fit_argv[3] == "benchmarks/fit_glm45_air_block_local_sidecar.py"
    assert "--source-dir" not in fit_argv
    assert fit_argv[fit_argv.index("--teacher-jsonl") + 1] == str(teacher_selection)
    assert fit_argv[fit_argv.index("--teacher-cache-root") + 1] == str(
        teacher_selection.parent
    )
    l41_argv = plan.step("legacy_fit_sidecar_01_04").argv
    assert l41_argv[l41_argv.index("--seed-artifact-dir") + 1] == str(
        plan.step("legacy_merge_sidecars_01_03").out_dir
    )


def test_legacy_lineage_recipe_includes_dynamic_imatrix_sweep_bridge(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "glm-4.5-air-vq"
    high_bit = tmp_path / "glm-4.5-air-vq2-e8p"
    sweep_root = tmp_path / "dynamic-sweep"
    imatrix_manifest = tmp_path / "imatrix" / "imatrix-manifest.json"
    source = tmp_path / "l18-l41-gate-up-output-bias-select8x8"
    dynamic = sweep_root / "air-dynamic-imatrix-bpw-3p0"
    scaled = tmp_path / "l18-l41-gate-up-output-bias-select8x8-l41alpha0p75"
    target = tmp_path / "artifact-r1"
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    baseline.mkdir()
    high_bit.mkdir()
    imatrix_manifest.parent.mkdir()
    source.mkdir()
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    imatrix_manifest.write_text('{"entries":[]}\n')
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")

    _write_dynamic_imatrix_candidate_summary(
        dynamic,
        baseline_artifact_dir=baseline,
        high_bit_artifact_dir=high_bit,
        imatrix_manifest=imatrix_manifest,
        budget=3.0,
    )
    _write_scaled_manifest(
        scaled,
        seed_artifact_dir=dynamic,
        source_artifact=source,
        scale_policy={"layer_18": 1.0, "layer_41": 0.75},
    )
    _write_manifest(
        target,
        seed_artifact_dir=scaled,
        run=_run(projection="up_proj", train_row_indices=[1, 2]),
    )

    raw = build_legacy_lineage_recipe(
        target,
        name="lineage-dynamic-fixture",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert raw["external_inputs"]["root_artifact"]["path"] == str(baseline)
    assert raw["external_inputs"]["legacy_sweep_imatrix_01"]["path"] == str(imatrix_manifest)
    assert raw["external_inputs"]["legacy_sweep_high_bit_01"]["path"] == str(high_bit)
    assert raw["lineage"]["root_artifact"] == str(baseline)
    assert raw["lineage"]["step_count"] == 3
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_materialize_sweep_01",
        "legacy_scale_sidecars_02",
        "legacy_train_03",
    ]
    assert raw["steps"][0]["op"] == "materialize-sweep"
    assert raw["steps"][0]["inputs"] == {
        "baseline_artifact": "external:root_artifact",
        "high_bit_artifact": "external:legacy_sweep_high_bit_01",
        "imatrix_manifest": "external:legacy_sweep_imatrix_01",
    }
    assert raw["steps"][0]["params"] == {
        "budget": [3.0],
        "candidate_prefix": "air-dynamic-imatrix",
        "mid_error_factor": 0.5,
        "reround_top_low_experts": 1,
    }
    assert raw["steps"][1]["inputs"]["seed_artifact"] == (
        "step:legacy_materialize_sweep_01/artifact_bpw3p0"
    )
    assert raw["steps"][2]["inputs"]["seed_artifact"] == (
        "step:legacy_scale_sidecars_02/artifact"
    )

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    sweep_argv = plan.step("legacy_materialize_sweep_01").argv
    assert "benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py" in sweep_argv
    assert sweep_argv[sweep_argv.index("--budget") + 1] == "3.0"
    assert sweep_argv[sweep_argv.index("--candidate-prefix") + 1] == "air-dynamic-imatrix"


def test_legacy_lineage_recipe_rebuilds_high_bit_stream_convert_side_input(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "glm-4.5-air-vq"
    high_bit = tmp_path / "glm-4.5-air-vq2-e8p"
    sweep_root = tmp_path / "dynamic-sweep"
    imatrix_manifest = tmp_path / "imatrix" / "imatrix-manifest.json"
    dynamic = sweep_root / "air-dynamic-imatrix-bpw-3p0"
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    baseline.mkdir()
    imatrix_manifest.parent.mkdir()
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    imatrix_manifest.write_text('{"entries":[]}\n')
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")
    _write_stream_convert_manifest(high_bit, code_bits=16, expert_workers=16)

    _write_dynamic_imatrix_candidate_summary(
        dynamic,
        baseline_artifact_dir=baseline,
        high_bit_artifact_dir=high_bit,
        imatrix_manifest=imatrix_manifest,
        budget=3.0,
    )

    raw = build_legacy_lineage_recipe(
        dynamic,
        name="lineage-high-bit-stream-convert",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert "legacy_sweep_high_bit_01" not in raw["external_inputs"]
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_stream_convert_high_bit_01",
        "legacy_materialize_sweep_01",
    ]
    stream_step = raw["steps"][0]
    assert stream_step["op"] == "stream-convert-vq"
    assert stream_step["inputs"] == {}
    assert stream_step["params"] == {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
        "download_missing_source_shards": True,
        "code_bits": 16,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "expert_workers": 16,
    }
    assert raw["steps"][1]["inputs"]["high_bit_artifact"] == (
        "step:legacy_stream_convert_high_bit_01/artifact"
    )

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    stream_argv = plan.step("legacy_stream_convert_high_bit_01").argv
    assert "--source-dir" not in stream_argv
    assert stream_argv[stream_argv.index("--revision") + 1] == (
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert stream_argv[stream_argv.index("--code-bits") + 1] == "16"
    assert "--download-missing-source-shards" in stream_argv


def test_legacy_lineage_recipe_rebuilds_complete_legacy_air_root_with_assumed_revision(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "glm-4.5-air-vq"
    high_bit = tmp_path / "glm-4.5-air-vq2-e8p"
    sweep_root = tmp_path / "dynamic-sweep"
    imatrix_manifest = tmp_path / "imatrix" / "imatrix-manifest.json"
    dynamic = sweep_root / "air-dynamic-imatrix-bpw-3p0"
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    revision = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    _write_legacy_existing_air_vq_manifest(baseline)
    high_bit.mkdir()
    imatrix_manifest.parent.mkdir()
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    imatrix_manifest.write_text('{"entries":[]}\n')
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")

    _write_dynamic_imatrix_candidate_summary(
        dynamic,
        baseline_artifact_dir=baseline,
        high_bit_artifact_dir=high_bit,
        imatrix_manifest=imatrix_manifest,
        budget=3.0,
    )

    raw = build_legacy_lineage_recipe(
        dynamic,
        name="lineage-base-stream-convert",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
        base_revision=revision,
    )

    assert "root_artifact" not in raw["external_inputs"]
    assert raw["lineage"]["root_artifact"] == str(baseline)
    assert raw["lineage"]["root_producer"] == {
        "op": "stream-convert-vq",
        "revision": revision,
    }
    assert raw["lineage"]["warnings"] == []
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_stream_convert_root",
        "legacy_materialize_sweep_01",
    ]
    stream_step = raw["steps"][0]
    assert stream_step["op"] == "stream-convert-vq"
    assert stream_step["inputs"] == {}
    assert stream_step["params"] == {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": revision,
        "download_missing_source_shards": True,
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "expert_workers": 16,
    }
    assert raw["steps"][1]["inputs"]["baseline_artifact"] == (
        "step:legacy_stream_convert_root/artifact"
    )

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    stream_argv = plan.step("legacy_stream_convert_root").argv
    assert stream_argv[stream_argv.index("--revision") + 1] == revision
    assert stream_argv[stream_argv.index("--code-bits") + 1] == "8"
    assert stream_argv[stream_argv.index("--group-size") + 1] == "512"
    assert stream_argv[stream_argv.index("--scale-estimator") + 1] == "max_abs"
    assert stream_argv[stream_argv.index("--expert-workers") + 1] == "16"
    assert "--source-dir" not in stream_argv
    assert "--download-missing-source-shards" in stream_argv


def test_legacy_lineage_recipe_can_emit_cleanroom_cache_preflight_prefix(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "glm-4.5-air-vq"
    target = tmp_path / "artifact-r1"
    revision = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    _write_legacy_existing_air_vq_manifest(baseline)
    _write_manifest(
        target,
        seed_artifact_dir=baseline,
        run=_run(projection="up_proj", train_row_indices=[1, 2]),
    )

    raw = build_legacy_lineage_recipe(
        target,
        name="lineage-cache-preflight",
        base_revision=revision,
        include_cache_preflight=True,
    )

    assert [step["id"] for step in raw["steps"][:12]] == [
        "legacy_pipeline_source_views",
        "legacy_rdma_topology_audit",
        "legacy_jaccl_hostfile",
        "legacy_local_sequential_stage_views",
        "legacy_pipeline_view_probe",
        "legacy_report_cache_single_host_prefix",
        "legacy_report_cache_distributed_prefix",
        "legacy_report_cache_single_host_preflight",
        "legacy_selection_cache_single_host_preflight",
        "legacy_report_cache_preflight",
        "legacy_selection_cache_preflight",
        "legacy_stream_convert_root",
    ]
    views = raw["steps"][0]
    topology = raw["steps"][1]
    stage_views = raw["steps"][3]
    report_prefix = raw["steps"][5]
    distributed_prefix = raw["steps"][6]
    report_fallback = raw["steps"][7]
    report = raw["steps"][9]
    selection = raw["steps"][10]
    assert views["op"] == "pipeline-source-views"
    assert views["class"] == "diagnostic"
    assert views["params"] == {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": revision,
        "pipeline_size": 2,
        "layer_split": 23,
        "rank_budget_gb": 115,
        "rank0_only_logits_views": True,
        "materialize": True,
    }
    assert stage_views["op"] == "pipeline-source-views"
    assert stage_views["class"] == "diagnostic"
    assert stage_views["params"] == {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": revision,
        "pipeline_size": 2,
        "layer_split": 23,
        "rank_budget_gb": 115,
        "local_sequential_stage_views": True,
        "local_sequential_head_process": True,
        "local_sequential_lower_split_layers": list(range(1, 23)),
        "local_sequential_upper_split_layers": list(range(24, 46)),
        "materialize": True,
    }
    assert topology["params"]["require_ready"] is True
    assert distributed_prefix["op"] == "export-teacher-cache-cleanroom"
    assert distributed_prefix["class"] == "diagnostic"
    assert distributed_prefix["inputs"] == {
        "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
        "jaccl_hostfile": "step:legacy_jaccl_hostfile/hostfile",
    }
    assert distributed_prefix["params"] == {
        "top_k": 16,
        "max_positions": 1,
        "prompt_set": "base",
        "prompt_id": ["capital_france"],
        "layer_split": 23,
        "required_wired_mb": 0,
        "local_direct_if": "en1",
        "peer_direct_if": "en1",
        "local_direct_ip": "192.168.10.1",
        "peer_direct_ip": "192.168.10.2",
        "peer_ssh": "jackmazac@192.168.100.191",
        "peer_source_stage_ssh": "jackmazac@192.168.10.2",
        "stage_peer_source": True,
        "peer_source_min_free_gb": 16,
        "mlx_cache_limit_gb": 48,
        "no_full_logits": True,
        "skip_local_consume": True,
        "require_memory_quiet_preflight": True,
        "memory_quiet_seconds": 3,
        "memory_quiet_max_attempts": 20,
        "memory_quiet_min_free_gb": 24,
        "allow_dirty_cache": True,
        "lm_head_chunk_rows": 8192,
        "vq_artifact_dir": "artifacts/glm-4.5-air-vq",
    }
    assert report_prefix["op"] == "export-teacher-cache-cleanroom"
    assert report_prefix["class"] == "diagnostic"
    assert report_prefix["inputs"] == {
        "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
        "local_sequential_stage_view_roots_json": (
            "step:legacy_local_sequential_stage_views/stage_view_roots_json"
        ),
    }
    assert report_prefix["params"] == {
        "top_k": 16,
        "max_positions": 1,
        "prompt_set": "base",
        "prompt_id": ["capital_france"],
        "single_host_fallback": True,
        "single_host_pipeline_local": True,
        "single_host_pipeline_local_stage_processes": True,
        "single_host_pipeline_local_head_process": True,
        "single_host_pipeline_local_lower_split_layers": list(range(1, 23)),
        "single_host_pipeline_local_upper_split_layers": list(range(24, 46)),
        "single_host_pipeline_local_abort_on_dirty_stage": True,
        "single_host_revision": revision,
        "mlx_cache_limit_gb": 48,
        "no_full_logits": True,
        "skip_local_consume": True,
        "require_memory_quiet_preflight": True,
        "memory_quiet_seconds": 3,
        "memory_quiet_max_attempts": 20,
        "memory_quiet_stage_view_margin_gb": 8,
        "peer_ssh": "jackmazac@192.168.100.191",
        "peer_rank0_view_root": "glm45-air-pipeline-views/rank-0-p2",
        "peer_source_dir": (
            "/Users/jackmazac/.cache/huggingface/hub/"
            "models--zai-org--GLM-4.5-Air/snapshots/"
            f"{revision}"
        ),
        "peer_source_stage_ssh": "jackmazac@192.168.10.2",
        "stage_peer_source": True,
        "peer_source_min_free_gb": 16,
        "local_sequential_remote_workers": "head",
        "local_sequential_remote_tmp_dir": "/tmp/glm-p2-remote-workers",
        "local_sequential_remote_dirty_retries": 1,
        "local_sequential_remote_retry_sleep_seconds": 30,
        "layer_split": 23,
        "lm_head_chunk_rows": 8192,
        "vq_artifact_dir": "artifacts/glm-4.5-air-vq",
    }
    assert report["op"] == "export-teacher-cache-cleanroom"
    assert report["class"] == "diagnostic"
    assert report["inputs"] == {
        "rank_view_dir": "step:legacy_pipeline_source_views/artifact",
        "jaccl_hostfile": "step:legacy_jaccl_hostfile/hostfile",
    }
    assert report["params"]["prompt_set"] == "air_vq_ladder_report_v1"
    assert report["params"]["preflight_only"] is True
    assert selection["params"]["prompt_set"] == "air_vq_ladder_select_v1"
    assert report_fallback["op"] == "export-teacher-cache-cleanroom"
    assert report_fallback["class"] == "diagnostic"
    assert report_fallback["inputs"] == {}
    assert report_fallback["params"] == {
        "top_k": 128,
        "max_positions": 128,
        "preflight_only": True,
        "single_host_fallback": True,
        "mlx_cache_limit_gb": 48,
        "single_host_source_memory_guard_ratio": 1.0,
        "vq_artifact_dir": "artifacts/glm-4.5-air-vq",
        "prompt_set": "air_vq_ladder_report_v1",
    }

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    report_argv = plan.step("legacy_report_cache_preflight").argv
    distributed_prefix_argv = plan.step("legacy_report_cache_distributed_prefix").argv
    prefix_argv = plan.step("legacy_report_cache_single_host_prefix").argv
    fallback_argv = plan.step("legacy_report_cache_single_host_preflight").argv
    stage_views_plan = plan.step("legacy_local_sequential_stage_views")
    assert report_argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert report_argv[report_argv.index("--rank-view-dir") + 1] == str(
        plan.step("legacy_pipeline_source_views").out_dir
    )
    assert report_argv[report_argv.index("--jaccl-hostfile") + 1] == str(
        plan.step("legacy_jaccl_hostfile").out_dir / "hostfile.json"
    )
    assert "--preflight-only" in report_argv
    assert "--rank-view-roots-json" not in report_argv
    assert distributed_prefix_argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--rank-view-dir") + 1
    ] == str(plan.step("legacy_pipeline_source_views").out_dir)
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--jaccl-hostfile") + 1
    ] == str(plan.step("legacy_jaccl_hostfile").out_dir / "hostfile.json")
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--prompt-set") + 1
    ] == "base"
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--prompt-id") + 1
    ] == "capital_france"
    assert "--single-host-fallback" not in distributed_prefix_argv
    assert "--no-full-logits" in distributed_prefix_argv
    assert "--skip-local-consume" in distributed_prefix_argv
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--peer-source-stage-ssh") + 1
    ] == "jackmazac@192.168.10.2"
    assert "--stage-peer-source" in distributed_prefix_argv
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--peer-source-min-free-gb") + 1
    ] == "16"
    assert "--require-memory-quiet-preflight" in distributed_prefix_argv
    assert distributed_prefix_argv[
        distributed_prefix_argv.index("--memory-quiet-min-free-gb") + 1
    ] == "24"
    assert "--preflight-only" not in distributed_prefix_argv
    assert prefix_argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert prefix_argv[prefix_argv.index("--rank-view-dir") + 1] == str(
        plan.step("legacy_pipeline_source_views").out_dir
    )
    assert prefix_argv[
        prefix_argv.index("--local-sequential-stage-view-roots-json") + 1
    ] == str(stage_views_plan.out_dir / "local_sequential_stage_view_roots.json")
    assert "--single-host-fallback" in prefix_argv
    assert "--single-host-pipeline-local" in prefix_argv
    assert "--single-host-pipeline-local-stage-processes" in prefix_argv
    assert "--single-host-pipeline-local-head-process" in prefix_argv
    assert "--single-host-pipeline-local-abort-on-dirty-stage" in prefix_argv
    assert "--no-full-logits" in prefix_argv
    assert "--skip-local-consume" in prefix_argv
    assert "--allow-dirty-cache" not in prefix_argv
    assert "--require-memory-quiet-preflight" in prefix_argv
    assert prefix_argv[prefix_argv.index("--memory-quiet-seconds") + 1] == "3"
    assert prefix_argv[prefix_argv.index("--memory-quiet-max-attempts") + 1] == "20"
    assert "--memory-quiet-min-free-gb" not in prefix_argv
    assert prefix_argv[prefix_argv.index("--memory-quiet-stage-view-margin-gb") + 1] == "8"
    assert prefix_argv[prefix_argv.index("--peer-ssh") + 1] == (
        "jackmazac@192.168.100.191"
    )
    assert prefix_argv[prefix_argv.index("--peer-rank0-view-root") + 1] == (
        "glm45-air-pipeline-views/rank-0-p2"
    )
    assert prefix_argv[prefix_argv.index("--peer-source-dir") + 1] == (
        "/Users/jackmazac/.cache/huggingface/hub/"
        "models--zai-org--GLM-4.5-Air/snapshots/"
        f"{revision}"
    )
    assert prefix_argv[prefix_argv.index("--peer-source-stage-ssh") + 1] == (
        "jackmazac@192.168.10.2"
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
    assert prefix_argv[prefix_argv.index("--prompt-id") + 1] == "capital_france"
    assert prefix_argv[prefix_argv.index("--max-positions") + 1] == "1"
    assert prefix_argv[prefix_argv.index("--top-k") + 1] == "16"
    assert prefix_argv[prefix_argv.index("--layer-split") + 1] == "23"
    assert prefix_argv[prefix_argv.index("--lm-head-chunk-rows") + 1] == "8192"
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
    assert "--single-host-pipeline-local-lower-split-layer" not in prefix_argv
    assert "--single-host-pipeline-local-upper-split-layer" not in prefix_argv
    assert fallback_argv[3] == "benchmarks/export_glm45_air_cleanroom_cache.py"
    assert "--single-host-fallback" in fallback_argv
    assert fallback_argv[fallback_argv.index("--mlx-cache-limit-gb") + 1] == "48"
    assert fallback_argv[
        fallback_argv.index("--single-host-source-memory-guard-ratio") + 1
    ] == "1.0"
    assert fallback_argv[fallback_argv.index("--vq-artifact-dir") + 1] == (
        "artifacts/glm-4.5-air-vq"
    )
    assert "--preflight-only" in fallback_argv
    assert "--jaccl-hostfile" not in fallback_argv
    assert "--peer-ssh" not in fallback_argv
    assert "--rank-view-dir" not in fallback_argv


def test_legacy_lineage_recipe_rebuilds_imatrix_collection_side_input(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "glm-4.5-air-vq"
    high_bit = tmp_path / "glm-4.5-air-vq2-e8p"
    sweep_root = tmp_path / "dynamic-sweep"
    imatrix_manifest = tmp_path / "imatrix" / "imatrix-manifest.json"
    dynamic = sweep_root / "air-dynamic-imatrix-bpw-3p0"
    teacher_report = tmp_path / "report" / "metadata.jsonl"
    teacher_selection = tmp_path / "selection" / "metadata.jsonl"
    baseline.mkdir()
    high_bit.mkdir()
    teacher_report.parent.mkdir()
    teacher_selection.parent.mkdir()
    teacher_report.write_text("{}\n")
    teacher_selection.write_text("{}\n")
    _write_imatrix_manifest(imatrix_manifest)

    _write_dynamic_imatrix_candidate_summary(
        dynamic,
        baseline_artifact_dir=baseline,
        high_bit_artifact_dir=high_bit,
        imatrix_manifest=imatrix_manifest,
        budget=3.0,
    )

    raw = build_legacy_lineage_recipe(
        dynamic,
        name="lineage-imatrix-collection",
        teacher_report=teacher_report,
        teacher_selection=teacher_selection,
    )

    assert "legacy_sweep_imatrix_01" not in raw["external_inputs"]
    assert [step["id"] for step in raw["steps"]] == [
        "legacy_collect_imatrix_01",
        "legacy_materialize_sweep_01",
    ]
    collect_step = raw["steps"][0]
    assert collect_step["op"] == "collect-imatrix"
    assert collect_step["inputs"] == {"artifact": "external:root_artifact"}
    assert collect_step["params"] == {
        "prompt_set": "air_imatrix_calib_v1",
        "layers": [1, 2],
        "projections": ["gate_proj", "up_proj"],
    }
    assert raw["steps"][1]["inputs"]["imatrix_manifest"] == (
        "step:legacy_collect_imatrix_01/imatrix_manifest"
    )

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)
    collect_argv = plan.step("legacy_collect_imatrix_01").argv
    sweep_argv = plan.step("legacy_materialize_sweep_01").argv
    assert collect_argv[collect_argv.index("--layers") + 1] == "1,2"
    assert collect_argv[collect_argv.index("--projections") + 1] == "gate_proj,up_proj"
    assert sweep_argv[sweep_argv.index("--imatrix-manifest") + 1] == str(
        plan.step("legacy_collect_imatrix_01").out_dir / "imatrix-manifest.json"
    )


def test_keep_recipe_lineage_cli_writes_yaml(tmp_path: Path) -> None:
    root = tmp_path / "root-seed"
    root.mkdir()
    target = tmp_path / "artifact-r1"
    _write_manifest(
        target,
        seed_artifact_dir=root,
        run=_run(projection="up_proj", train_row_indices=[1, 2]),
    )
    report = tmp_path / "report.jsonl"
    selection = tmp_path / "selection.jsonl"
    report.write_text("{}\n")
    selection.write_text("{}\n")
    out = tmp_path / "lineage.yaml"

    result = subprocess.run(
        [
            "uv",
            "run",
            "keep",
            "recipe-lineage",
            str(target),
            "--name",
            "lineage-cli-fixture",
            "--base-revision",
            "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
            "--include-cache-preflight",
            "--teacher-report",
            str(report),
            "--teacher-selection",
            str(selection),
            "--out",
            str(out),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "lineage recipe written" in result.stdout
    raw = yaml.safe_load(out.read_text())
    assert raw["name"] == "lineage-cli-fixture"
    assert [step["id"] for step in raw["steps"][:11]] == [
        "legacy_pipeline_source_views",
        "legacy_rdma_topology_audit",
        "legacy_jaccl_hostfile",
        "legacy_local_sequential_stage_views",
        "legacy_pipeline_view_probe",
        "legacy_report_cache_single_host_prefix",
        "legacy_report_cache_distributed_prefix",
        "legacy_report_cache_single_host_preflight",
        "legacy_selection_cache_single_host_preflight",
        "legacy_report_cache_preflight",
        "legacy_selection_cache_preflight",
    ]
    train_step = next(step for step in raw["steps"] if step["id"] == "legacy_train_01")
    assert train_step["params"]["projection"] == "up_proj"
