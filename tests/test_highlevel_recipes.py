from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from keep.build.cli import main as keep_main
from keep.build.ops import REGISTRY
from keep.build.recipe import RecipeError, load_recipe, parse_recipe, validate_recipe
from keep.build.runner import plan_recipe
from keep.build.highlevel import compile_high_level, is_high_level_recipe


def _write_highlevel(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _compiled_recipe(path: Path):
    return parse_recipe(compile_high_level(path), source_path=str(path))


def test_detection_rule_distinguishes_high_level_from_low_level(tmp_path: Path) -> None:
    high = _write_highlevel(tmp_path, {"model": "qwen36-35b-a3b"})
    low = tmp_path / "low.yaml"
    low.write_text(
        yaml.safe_dump(
            {
                "model": "qwen36-35b-a3b",
                "schema_version": 1,
                "name": "low",
                "steps": [],
            }
        )
    )

    assert is_high_level_recipe(high) is True
    assert is_high_level_recipe(low) is False


@pytest.mark.parametrize(
    "path,expected_step",
    [
        (Path("examples/qwen36-a3b.yaml"), "qwen_family_gate"),
        (Path("examples/glm45-air.yaml"), "lane_s_candidate"),
    ],
)
def test_examples_compile_and_validate(path: Path, expected_step: str) -> None:
    recipe = _compiled_recipe(path)

    assert validate_recipe(recipe, REGISTRY) == []
    assert recipe.step(expected_step).id == expected_step


def test_fast_and_wow_presets_have_different_gate_strictness(tmp_path: Path) -> None:
    fast = _compiled_recipe(
        _write_highlevel(
            tmp_path,
            {"model": "glm45-air", "quality": "fast", "recovery": "auto"},
        )
    )
    wow = _compiled_recipe(
        _write_highlevel(
            tmp_path,
            {"model": "glm45-air", "quality": "wow", "recovery": "auto"},
        )
    )

    fast_gate = fast.step("lane_s_candidate").gate
    wow_gate = wow.step("lane_s_candidate").gate
    assert fast_gate is not None
    assert wow_gate is not None
    assert fast_gate.thresholds["allow_dirty_rows"] is True
    assert "allow_dirty_rows" not in wow_gate.thresholds

    fast_report = fast.promotion.gates[0]
    wow_report = wow.promotion.gates[0]
    assert fast_report.thresholds["allow_dirty_rows"] is True
    assert fast_report.thresholds["clean_rows"] == 0
    assert "allow_dirty_rows" not in wow_report.thresholds


def test_unknown_model_and_quality_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="unknown model profile 'missing'"):
        compile_high_level(_write_highlevel(tmp_path, {"model": "missing"}))

    with pytest.raises(RecipeError, match="unknown quality 'turbo'"):
        compile_high_level(
            _write_highlevel(
                tmp_path,
                {"model": "qwen36-35b-a3b", "quality": "turbo"},
            )
        )


def test_glm52_reap_example_compiles_to_the_real_release_evidence_chain() -> None:
    recipe = _compiled_recipe(Path("examples/glm52-reap.yaml"))

    assert validate_recipe(recipe, REGISTRY) == []
    assert [step.id for step in recipe.steps] == [
        "source_audit",
        "source_payload_audit",
        "materialize_groups",
        "non_vq_pack",
        "composite_audit",
        "indexshare_runtime",
        "tokenizer_readiness",
        "synthetic_generation",
        "full_bind_evidence",
        "family_policy",
        "eval_prompts",
        "teacher_metadata",
        "teacher_cache",
        "teacher_cache_audit",
        "production_generation",
        "candidate_eval_produce",
        "candidate_eval_compare",
        "candidate_route_capture",
        "route_math_compare",
        "same_machine_benchmark_pair",
        "same_machine_benchmark_compare",
        "family_gate",
    ]
    assert (
        recipe.step("candidate_eval_produce").op
        == "glm52-candidate-full-vocab-eval-produce"
    )
    assert recipe.step("route_math_compare").op == "glm52-route-math-compare"
    assert (
        recipe.step("same_machine_benchmark_pair").op
        == "glm52-same-machine-benchmark-pair"
    )
    family_gate = recipe.step("family_gate")
    assert family_gate.op == "glm52-family-gate-check"
    assert family_gate.inputs["candidate_cache_root"].raw == (
        "step:candidate_eval_produce/artifact"
    )
    assert family_gate.inputs["benchmark_evidence_json"].raw == (
        "step:same_machine_benchmark_compare/evidence_json"
    )
    assert family_gate.inputs["source_route_trace_root"].raw == "step:teacher_cache/route_trace_root"
    assert family_gate.inputs["candidate_route_trace_root"].raw == (
        "step:candidate_route_capture/artifact"
    )


def test_glm52_reap_requires_explicit_release_input_paths(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="requires release_inputs"):
        compile_high_level(
            _write_highlevel(
                tmp_path,
                {
                    "model": "glm52-reap-504b-v2",
                    "quality": "fast",
                    "recovery": "off",
                },
            )
        )


def test_glm52_reap_rejects_non_1024_token_benchmark_prompt_at_compile_time(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(Path("examples/glm52-reap.yaml").read_text())
    invalid_prompt = tmp_path / "wrong-shaped-prompt.json"
    invalid_prompt.write_text("[1, 2, 3]\n")
    raw["release_inputs"]["benchmark_prompt_token_ids_json"] = str(invalid_prompt)

    with pytest.raises(RecipeError, match="exactly 1024 JSON integers"):
        compile_high_level(_write_highlevel(tmp_path, raw))


def test_overrides_merge_into_step_params(tmp_path: Path) -> None:
    recipe = _compiled_recipe(
        _write_highlevel(
            tmp_path,
            {
                "model": "qwen36-35b-a3b",
                "quality": "fast",
                "overrides": {"materialize_groups": {"max_groups": 8}},
            },
        )
    )

    assert recipe.step("materialize_groups").params["max_groups"] == 8
    assert validate_recipe(recipe, REGISTRY) == []


def test_compile_cli_emits_low_level_yaml(tmp_path: Path) -> None:
    out = tmp_path / "compiled.yaml"

    assert keep_main(["compile", "examples/qwen36-a3b.yaml", "-o", str(out)]) == 0

    recipe = load_recipe(out)
    assert recipe.name == "qwen36-35b-a3b-fast"
    assert recipe.step("qwen_family_gate").op == "qwen-family-gate-check"


def test_build_accepts_high_level_recipe_without_spawning(tmp_path: Path) -> None:
    raw = compile_high_level("examples/qwen36-a3b.yaml")
    recipe = parse_recipe(raw, source_path="examples/qwen36-a3b.yaml")

    plan = plan_recipe(
        recipe,
        build_root=tmp_path / "build",
        no_hash=True,
    )

    assert plan.step("materialize_groups").argv[:3] == ["uv", "run", "python"]
    assert plan.step("qwen_family_gate").argv[3] == "benchmarks/check_qwen_family_gate.py"


def test_build_cli_auto_compiles_high_level_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = keep_main(
        [
            "build",
            "examples/qwen36-a3b.yaml",
            "--dry-run",
            "--no-hash",
            "--build-root",
            str(tmp_path / "build"),
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "recipe: qwen36-35b-a3b-fast" in output
    assert "step qwen_family_gate [diagnostic]" in output


def test_low_level_yaml_still_loads_and_plans_unchanged(tmp_path: Path) -> None:
    recipe = load_recipe("recipes/qwen36_35b_a3b_materialization_probe_20260702.yaml")
    plan = plan_recipe(recipe, build_root=tmp_path / "build", no_hash=True)

    assert plan.step("qwen_family_gate").spec.op == "qwen-family-gate-check"
