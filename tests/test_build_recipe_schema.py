from __future__ import annotations

import pytest

from keep.build.ops import REGISTRY
from keep.build.recipe import (
    InputRef,
    RecipeError,
    parse_recipe,
    validate_recipe,
)


def _base_recipe(steps: list[dict], promotion: dict | None = None) -> dict:
    raw = {
        "schema_version": 1,
        "name": "fixture",
        "external_inputs": {
            "seed_artifact": {"kind": "artifact_dir", "path": "artifacts/seed"},
            "teacher_report": {"kind": "teacher_cache", "path": "caches/report/metadata.jsonl"},
            "teacher_selection": {"kind": "teacher_cache", "path": "caches/select/metadata.jsonl"},
        },
        "steps": steps,
    }
    if promotion is not None:
        raw["promotion"] = promotion
    return raw


def _train_step(step_id: str = "train", seed_ref: str = "external:seed_artifact") -> dict:
    return {
        "id": step_id,
        "op": "train-low-rank",
        "class": "promotable",
        "inputs": {
            "seed_artifact": seed_ref,
            "selection_teacher": "external:teacher_selection",
            "validation_teacher": "external:teacher_report",
        },
        "params": {"layer": 45, "low_rank": 4},
        "gate": {"profile": "train_sane"},
    }


def _eval_step(step_id: str = "eval_report", artifact_ref: str = "step:train/artifact") -> dict:
    return {
        "id": step_id,
        "op": "eval",
        "class": "verify",
        "inputs": {"artifact": artifact_ref, "teacher": "external:teacher_report"},
        "params": {"engine": "vq_e1_routed_nax_e8p"},
        "gate": {"profile": "balanced_rc_split"},
    }


def test_valid_recipe_passes() -> None:
    recipe = parse_recipe(_base_recipe([_train_step(), _eval_step()]))
    assert validate_recipe(recipe, REGISTRY) == []


def test_teacher_cache_artifact_external_input_kind_is_accepted() -> None:
    raw = _base_recipe([_train_step(), _eval_step()])
    raw["external_inputs"]["glm52_teacher_cache"] = {
        "kind": "teacher_cache_artifact",
        "path": "artifacts/quality/glm52-teacher-cache",
    }

    recipe = parse_recipe(raw)

    assert recipe.external_inputs["glm52_teacher_cache"].kind == (
        "teacher_cache_artifact"
    )


def test_input_ref_parsing() -> None:
    external = InputRef.parse("external:seed")
    assert external.kind == "external" and external.name == "seed"
    step = InputRef.parse("step:train/artifact")
    assert step.kind == "step" and step.name == "train" and step.output == "artifact"
    with pytest.raises(ValueError):
        InputRef.parse("step:train")
    with pytest.raises(ValueError):
        InputRef.parse("seed")


def test_duplicate_step_ids_rejected() -> None:
    recipe = parse_recipe(_base_recipe([_train_step(), _train_step()]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("duplicate step id" in error for error in errors)


def test_unknown_op_rejected() -> None:
    step = _train_step()
    step["op"] = "mystery-op"
    recipe = parse_recipe(_base_recipe([step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("unknown op" in error for error in errors)


def test_forward_reference_rejected() -> None:
    recipe = parse_recipe(_base_recipe([_eval_step(), _train_step()]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("not declared earlier" in error for error in errors)


def test_dangling_external_reference_rejected() -> None:
    step = _train_step(seed_ref="external:missing")
    recipe = parse_recipe(_base_recipe([step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("unknown external input 'missing'" in error for error in errors)


def test_unknown_output_reference_rejected() -> None:
    recipe = parse_recipe(
        _base_recipe([_train_step(), _eval_step(artifact_ref="step:train/mystery")])
    )
    errors = validate_recipe(recipe, REGISTRY)
    assert any("does not declare" in error for error in errors)


def test_taint_rule_rejects_diagnostic_feeding_promotable() -> None:
    bias_step = {
        "id": "logit_bias",
        "op": "materialize-logit-bias",
        "class": "diagnostic",
        "inputs": {"seed_artifact": "external:seed_artifact"},
        "params": {"token_bias": "565:2.0"},
    }
    tainted_train = _train_step(step_id="train_tainted", seed_ref="step:logit_bias/artifact")
    recipe = parse_recipe(_base_recipe([bias_step, tainted_train]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any(
        "train_tainted" in error and "logit_bias" in error and "diagnostic" in error
        for error in errors
    )


def test_taint_rule_rejects_diagnostic_feeding_verify() -> None:
    bias_step = {
        "id": "logit_bias",
        "op": "materialize-logit-bias",
        "class": "diagnostic",
        "inputs": {"seed_artifact": "external:seed_artifact"},
        "params": {"token_bias": "565:2.0"},
    }
    eval_step = _eval_step(step_id="eval_bias", artifact_ref="step:logit_bias/artifact")
    recipe = parse_recipe(_base_recipe([bias_step, eval_step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("may not consume" in error for error in errors)


def test_promotable_class_not_allowed_for_diagnostic_only_op() -> None:
    step = {
        "id": "bias",
        "op": "materialize-logit-bias",
        "class": "promotable",
        "inputs": {"seed_artifact": "external:seed_artifact"},
        "params": {"token_bias": "565:2.0"},
    }
    recipe = parse_recipe(_base_recipe([step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("does not allow class" in error for error in errors)


def test_missing_promotable_track_a_ops_validate() -> None:
    raw = _base_recipe(
        [
            _train_step(),
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
                "params": {
                    "scale": 1,
                    "max_abs_delta": 0.125,
                    "num_experts": 128,
                },
                "gate": {"profile": "train_sane"},
            },
        ]
    )
    raw["external_inputs"]["route_disagreement"] = {
        "kind": "file",
        "path": "artifacts/quality/route-disagreement.json",
    }
    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []


def test_isolate_sparse_fp16_op_validates_with_pinned_source() -> None:
    raw = _base_recipe(
        [
            _train_step(),
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
            },
        ]
    )
    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []


def test_sparse_residual_op_validates_with_pinned_source_and_plan() -> None:
    raw = _base_recipe(
        [
            _train_step(),
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
                    "residual_scale": 1.0,
                },
                "gate": {"profile": "train_sane"},
            },
        ]
    )
    raw["external_inputs"]["sparse_plan"] = {
        "kind": "file",
        "path": "artifacts/quality/sparse-plan.json",
    }

    recipe = parse_recipe(raw)

    assert validate_recipe(recipe, REGISTRY) == []


def test_isolate_sparse_fp16_requires_pinned_revision() -> None:
    raw = _base_recipe(
        [
            _train_step(),
            {
                "id": "isolate_sparse_fp16",
                "op": "isolate-sparse-fp16",
                "class": "promotable",
                "inputs": {"seed_artifact": "step:train/artifact"},
                "params": {
                    "projection": ["41:gate_proj"],
                    "model_id": "zai-org/GLM-4.5-Air",
                },
            },
        ]
    )
    recipe = parse_recipe(raw)
    errors = validate_recipe(recipe, REGISTRY)
    assert any("requires params" in error and "revision" in error for error in errors)


def test_eval_repair_op_validates_against_prior_split_evidence() -> None:
    raw = _base_recipe(
        [
            _train_step(),
            _eval_step(),
            {
                "id": "eval_report_repair",
                "op": "eval-repair",
                "class": "verify",
                "inputs": {
                    "artifact": "step:train/artifact",
                    "teacher": "external:teacher_report",
                    "existing_evidence": "step:eval_report/evidence",
                },
                "params": {
                    "engine": "vq_e1_routed_nax_e8p",
                    "max_rows": 128,
                    "repair_attempts": 2,
                    "mlx_cache_limit_gb": 0,
                    "mlx_clear_cache_before_load": True,
                },
                "gate": {"profile": "balanced_rc_split"},
            },
        ],
        promotion={
            "artifact_step": "train",
            "gates": [{"evidence_step": "eval_report_repair", "profile": "community_wow"}],
        },
    )
    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []


def test_base_build_producer_ops_validate() -> None:
    raw = {
        "schema_version": 1,
        "name": "base-build-fixture",
        "external_inputs": {
            "source_dir": {"kind": "artifact_dir", "path": "hf/glm45-air"},
            "config_path": {"kind": "file", "path": "hf/glm45-air/config.json"},
            "index_path": {
                "kind": "file",
                "path": "hf/glm45-air/model.safetensors.index.json",
            },
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
                    "code_bits": 8,
                    "group_size": 512,
                    "expert_workers": 4,
                },
            },
            {
                "id": "report_cache",
                "op": "export-teacher-cache-cleanroom",
                "class": "promotable",
                "params": {
                    "prompt_set": "air_vq_ladder_report_v1",
                    "top_k": 128,
                    "max_positions": 128,
                    "rank_view_roots_json": '{"0":"/rank0","1":"/rank1"}',
                },
            },
            {
                "id": "selection_cache",
                "op": "export-teacher-cache-cleanroom",
                "class": "promotable",
                "params": {
                    "prompt_set": "air_vq_ladder_select_v1",
                    "top_k": 128,
                    "max_positions": 128,
                    "rank_view_roots_json": '{"0":"/rank0","1":"/rank1"}',
                },
            },
            {
                "id": "seed_train",
                "op": "train-low-rank",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:base_vq/artifact",
                    "selection_teacher": "step:selection_cache/teacher_cache",
                    "validation_teacher": "step:report_cache/teacher_cache",
                },
                "params": {"layer": 45, "low_rank": 4},
                "gate": {"profile": "train_sane"},
            },
        ],
    }

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []


def test_stream_convert_vq_can_use_hf_index_parent_as_source_dir() -> None:
    raw = {
        "schema_version": 1,
        "name": "hf-default-source-fixture",
        "external_inputs": {
            "config_path": {"kind": "file", "path": "hf/glm45-air/config.json"},
            "index_path": {
                "kind": "file",
                "path": "hf/glm45-air/model.safetensors.index.json",
            },
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
                    "code_bits": 8,
                    "group_size": 512,
                    "expert_workers": 4,
                },
            }
        ],
    }

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []


def test_fit_block_local_sidecar_op_accepts_source_and_teacher_inputs() -> None:
    raw = {
        "schema_version": 1,
        "name": "fit-block-local-sidecar-fixture",
        "external_inputs": {
            "seed": {"kind": "artifact_dir", "path": "artifacts/seed"},
            "source_dir": {"kind": "artifact_dir", "path": "hf/glm45-air"},
            "teacher": {
                "kind": "teacher_cache",
                "path": "artifacts/quality/select/metadata.jsonl",
            },
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
                "gate": {"profile": "train_sane"},
            }
        ],
    }

    recipe = parse_recipe(raw)
    assert validate_recipe(recipe, REGISTRY) == []


def test_unknown_param_rejected() -> None:
    step = _train_step()
    step["params"]["mystery_knob"] = 1
    recipe = parse_recipe(_base_recipe([step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("mystery_knob" in error for error in errors)


def test_unknown_gate_profile_rejected() -> None:
    step = _train_step()
    step["gate"] = {"profile": "mystery_gate"}
    recipe = parse_recipe(_base_recipe([step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("unknown gate profile" in error for error in errors)


def test_gate_threshold_override_must_name_existing_threshold() -> None:
    step = _eval_step()
    step["gate"] = {"profile": "balanced_rc_split", "thresholds": {"mystery": 1}}
    recipe = parse_recipe(_base_recipe([_train_step(), step]))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("no threshold" in error for error in errors)


def test_promotion_artifact_step_must_be_promotable() -> None:
    promotion = {
        "artifact_step": "eval_report",
        "gates": [{"evidence_step": "eval_report", "profile": "community_wow"}],
    }
    recipe = parse_recipe(_base_recipe([_train_step(), _eval_step()], promotion))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("must be promotable" in error for error in errors)


def test_promotion_gate_step_must_be_verify() -> None:
    promotion = {
        "artifact_step": "train",
        "gates": [{"evidence_step": "train", "profile": "community_wow"}],
    }
    recipe = parse_recipe(_base_recipe([_train_step(), _eval_step()], promotion))
    errors = validate_recipe(recipe, REGISTRY)
    assert any("must be class verify" in error for error in errors)


def test_schema_version_required() -> None:
    raw = _base_recipe([_train_step()])
    raw["schema_version"] = 99
    with pytest.raises(RecipeError, match="schema_version"):
        parse_recipe(raw)
