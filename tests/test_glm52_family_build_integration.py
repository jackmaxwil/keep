from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from keep.build.ops import REGISTRY, StepContext
from keep.build.recipe import RecipeError, StepSpec, load_recipe
from keep.build.runner import plan_recipe


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
CHECKED_RECIPE = Path("recipes/glm52_reap_504b_eval_contract_20260709.yaml")
PREFLIGHT_RECIPE = Path(
    "recipes/glm52_reap_504b_family_preflight_20260709.yaml"
)
V2_EVIDENCE_RECIPE = Path(
    "recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml"
)
V2_EXTERNAL_EVIDENCE = {
    "family_policy_json": (
        "artifacts/quality/glm52-family-policy-20260709-v2.json",
        "0975f7dc1117c5fba7532e9166f4767546fd691a6cb52520a40f09874f972ce2",
    ),
    "eval_prompt_pack_json": (
        "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json",
        "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31",
    ),
    "teacher_metadata_json": (
        "artifacts/quality/glm52-teacher-metadata-20260709.json",
        "621a013eb37f617409ec568340976e769610b3b6c8cae346639762d816a7fd7a",
    ),
    "full_bind_preflight_json": (
        "artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json",
        "9078a36c683a2487ef5d99f4c8fecc72f086689b8aa1e249ffaa34639b5946a3",
    ),
    "indexshare_runtime_json": (
        "artifacts/quality/glm52-indexshare-runtime-20260709.json",
        "94785cf0884ce5ac955a09b535c0b2ef00ad3cb742da7769bf504a4c9143ad25",
    ),
    "synthetic_generation_json": (
        "artifacts/quality/glm52-synthetic-generation-contract-20260709.json",
        "baa9c8214553c6af391c6a671a5b0185b8d7eb1746c3e6c52586f512536662a9",
    ),
    "non_vq_evidence_json": (
        "artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/"
        "non_vq_pack-a678bc3c/evidence/evidence_json.json",
        "ccbedd87f72f032bd86fd9d5a89fb75641595f30cf2a80fd7c84636d594e80d2",
    ),
    "composite_artifact_audit_json": (
        "artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json",
        "8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a",
    ),
    "production_generation_json": (
        "artifacts/quality/"
        "glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json",
        "758b5bbebbb365cdf69998a034cf3791a6210d2723d5bec6c022336eceadcc82",
    ),
}


def _write_recipe(tmp_path: Path) -> Path:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    readiness_path = tmp_path / "tokenizer-readiness.json"
    readiness_path.write_text(json.dumps({"probe_pass": True}) + "\n")
    recipe_path = tmp_path / "family-contract.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-family-contract-fixture",
                "description": "fixture",
                "external_inputs": {
                    "tokenizer_dir": {
                        "kind": "artifact_dir",
                        "path": str(tokenizer_dir),
                    },
                    "tokenizer_readiness": {
                        "kind": "file",
                        "path": str(readiness_path),
                    },
                },
                "steps": [
                    {
                        "id": "family_policy",
                        "op": "glm52-family-gate-policy",
                        "class": "diagnostic",
                        "params": {"model_id": MODEL_ID, "revision": REVISION},
                    },
                    {
                        "id": "eval_prompts",
                        "op": "glm52-family-eval-prompt-pack",
                        "class": "diagnostic",
                        "inputs": {
                            "tokenizer_dir": "external:tokenizer_dir",
                            "tokenizer_readiness_json": (
                                "external:tokenizer_readiness"
                            ),
                            "family_policy_json": (
                                "step:family_policy/evidence_json"
                            ),
                        },
                        "params": {"model_id": MODEL_ID, "revision": REVISION},
                    },
                ],
            },
            sort_keys=False,
        )
    )
    return recipe_path


def test_glm52_family_contract_ops_render_exact_commands(tmp_path: Path) -> None:
    plan = plan_recipe(
        load_recipe(_write_recipe(tmp_path)),
        build_root=tmp_path / "build",
    )

    policy = plan.step("family_policy")
    assert policy.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/define_glm52_family_gate_policy.py",
    ]
    assert policy.argv[policy.argv.index("--model-id") + 1] == MODEL_ID
    assert policy.argv[policy.argv.index("--revision") + 1] == REVISION
    assert policy.output_paths == {
        "evidence_json": policy.evidence_dir / "evidence_json.json"
    }

    prompts = plan.step("eval_prompts")
    assert prompts.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/prepare_glm52_family_eval_prompts.py",
    ]
    assert "--tokenizer-dir" in prompts.argv
    assert "--tokenizer-readiness-json" in prompts.argv
    assert "--family-policy-json" in prompts.argv
    assert prompts.op_def.cacheable is False
    assert prompts.output_paths == {
        "evidence_json": prompts.evidence_dir / "evidence_json.json"
    }


def test_checked_eval_contract_recipe_freezes_before_teacher_or_candidate() -> None:
    recipe = load_recipe(CHECKED_RECIPE)
    plan = plan_recipe(recipe)

    assert [step.id for step in recipe.steps] == [
        "tokenizer_readiness",
        "family_policy",
        "eval_prompts",
    ]
    assert {
        name: reference.raw
        for name, reference in recipe.step("eval_prompts").inputs.items()
    } == {
        "tokenizer_dir": "external:tokenizer_snapshot",
        "tokenizer_readiness_json": (
            "step:tokenizer_readiness/evidence_json"
        ),
        "family_policy_json": "step:family_policy/evidence_json",
    }
    assert plan.step("tokenizer_readiness").op_def.cacheable is False
    assert plan.step("eval_prompts").op_def.cacheable is False
    assert all("teacher" not in step.id for step in recipe.steps)
    assert all("candidate" not in step.id for step in recipe.steps)


def test_glm52_teacher_metadata_op_renders_authenticated_header_only_probe(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    input_names = (
        "profile_path",
        "config_path",
        "index_path",
        "source_audit_json",
        "source_payload_audit_json",
        "tokenizer_readiness_json",
        "family_policy_json",
        "eval_prompt_pack_json",
    )
    inputs: dict[str, dict[str, str]] = {
        "source_dir": {"kind": "artifact_dir", "path": str(source_dir)}
    }
    for name in input_names:
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        inputs[name] = {"kind": "file", "path": str(path)}
    recipe_path = tmp_path / "teacher-metadata.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-teacher-metadata-fixture",
                "description": "fixture",
                "external_inputs": inputs,
                "steps": [
                    {
                        "id": "teacher_metadata",
                        "op": "glm52-family-eval-teacher-metadata",
                        "class": "diagnostic",
                        "inputs": {
                            name: f"external:{name}"
                            for name in ("source_dir", *input_names)
                        },
                        "params": {"model_id": MODEL_ID, "revision": REVISION},
                    }
                ],
            },
            sort_keys=False,
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("teacher_metadata")

    assert step.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/prepare_glm52_family_eval_teacher_metadata.py",
    ]
    for flag in (
        "--source-dir",
        "--profile-path",
        "--config-path",
        "--index-path",
        "--source-audit-json",
        "--source-payload-audit-json",
        "--tokenizer-readiness-json",
        "--family-policy-json",
        "--eval-prompt-pack-json",
        "--model-id",
        "--revision",
        "--output-dir",
        "--output-json",
    ):
        assert flag in step.argv
    assert step.output_paths["metadata_jsonl"].name == (
        "glm52_teacher_source_metadata.jsonl"
    )
    assert step.op_def.cacheable is False


def test_glm52_teacher_cache_audit_op_consumes_raw_root_and_manifest(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "teacher-cache"
    cache_root.mkdir()
    manifest = cache_root / "glm52-teacher-cache-fp32-manifest.json"
    manifest.write_text("{}\n")
    prompt_pack = tmp_path / "prompt-pack.json"
    prompt_pack.write_text("{}\n")
    op = REGISTRY["glm52-teacher-cache-audit"]
    context = StepContext(
        spec=StepSpec(
            id="teacher_cache_audit",
            op=op.name,
            step_class="diagnostic",
            inputs={},
            params={},
            gate=None,
        ),
        inputs={
            "teacher_cache_root": cache_root,
            "teacher_cache_manifest_json": manifest,
            "eval_prompt_pack_json": prompt_pack,
        },
        out_dir=tmp_path / "out",
        evidence_dir=tmp_path / "evidence",
    )
    argv = op.build_argv(context)

    assert argv[:5] == [
        "uv",
        "run",
        "python",
        "benchmarks/check_glm52_family_gate.py",
        "--audit-teacher-cache-only",
    ]
    assert [token for token in argv if token.startswith("--")] == [
        "--audit-teacher-cache-only",
        "--teacher-cache-root",
        "--teacher-cache-manifest-json",
        "--eval-prompt-pack-json",
        "--output-json",
    ]
    assert op.required_inputs == (
        "teacher_cache_root",
        "teacher_cache_manifest_json",
        "eval_prompt_pack_json",
    )
    assert op.allowed_classes == frozenset({"diagnostic"})
    assert op.produces_artifact is False
    assert op.cacheable is False


def test_glm52_teacher_cache_produce_op_plans_exact_locked_artifact_argv(
    tmp_path: Path,
) -> None:
    input_names = (
        "snapshot_dir",
        "prompt_pack_json",
        "non_vq_package_dir",
        "artifact_identities_json",
        "profile_path",
        "ledger_path",
        "checkpoint_dir",
    )
    external_inputs = {
        name: {
            "kind": (
                "artifact_dir"
                if name in {"snapshot_dir", "non_vq_package_dir"}
                else "file"
            ),
            "path": str(tmp_path / name),
        }
        for name in input_names
    }
    recipe_path = tmp_path / "teacher-cache-produce.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-teacher-cache-produce-fixture",
                "description": "fixture",
                "external_inputs": external_inputs,
                "steps": [
                    {
                        "id": "teacher_cache",
                        "op": "glm52-teacher-cache-produce",
                        "class": "diagnostic",
                        "inputs": {
                            name: f"external:{name}" for name in input_names
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )

    step = plan_recipe(
        load_recipe(recipe_path),
        build_root=tmp_path / "build",
        no_hash=True,
    ).step("teacher_cache")

    assert step.argv == [
        "uv",
        "run",
        "python",
        "benchmarks/produce_glm52_teacher_cache.py",
        "--snapshot-dir",
        str(tmp_path / "snapshot_dir"),
        "--prompt-pack-json",
        str(tmp_path / "prompt_pack_json"),
        "--non-vq-package-dir",
        str(tmp_path / "non_vq_package_dir"),
        "--artifact-identities-json",
        str(tmp_path / "artifact_identities_json"),
        "--profile-path",
        str(tmp_path / "profile_path"),
        "--ledger-path",
        str(tmp_path / "ledger_path"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint_dir"),
        "--cache-root",
        str(step.out_dir),
        "--route-trace-root",
        str(step.out_dir / "../source-route-traces"),
    ]
    assert step.op_def.required_inputs == input_names
    assert step.op_def.produces_artifact is True
    assert step.op_def.artifact_manifest_name == (
        "glm52-teacher-cache-fp32-manifest.json"
    )
    assert step.op_def.resume_partial is True
    assert step.output_paths == {
        "artifact": step.out_dir,
        "manifest": step.out_dir / "glm52-teacher-cache-fp32-manifest.json",
        "route_trace_root": step.out_dir / "../source-route-traces",
    }


def test_glm52_family_gate_op_has_no_boolean_bypass_flags(tmp_path: Path) -> None:
    required_inputs = (
        "family_policy_json",
        "eval_prompt_pack_json",
        "teacher_metadata_json",
        "full_bind_preflight_json",
        "indexshare_runtime_json",
        "synthetic_generation_json",
    )
    external_inputs: dict[str, dict[str, str]] = {}
    for name in required_inputs:
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        external_inputs[name] = {"kind": "file", "path": str(path)}
    recipe_path = tmp_path / "family-gate.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-family-gate-fixture",
                "description": "fixture",
                "external_inputs": external_inputs,
                "steps": [
                        {
                            "id": "family_gate",
                            "op": "glm52-family-gate-check",
                            "class": "diagnostic",
                            "gate": {"profile": "glm52_family"},
                            "inputs": {
                            name: f"external:{name}" for name in required_inputs
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )

    step = plan_recipe(
        load_recipe(recipe_path), build_root=tmp_path / "build"
    ).step("family_gate")

    assert step.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/check_glm52_family_gate.py",
    ]
    for name in required_inputs:
        assert f"--{name.replace('_', '-')}" in step.argv
    assert "--output-json" in step.argv
    assert not any(
        flag in step.argv
        for flag in (
            "--family-gate",
            "--family-eval-gate",
            "--family-benchmark-gate",
            "--production-generation",
        )
    )
    assert step.op_def.cacheable is False


def test_glm52_family_gate_op_renders_complete_schema_v2_raw_evidence_trio(
    tmp_path: Path,
) -> None:
    required_inputs = tuple(V2_EXTERNAL_EVIDENCE)[:6]
    raw_inputs = tuple(V2_EXTERNAL_EVIDENCE)[6:]
    external_inputs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name in (*required_inputs, *raw_inputs):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        paths[name] = path
        external_inputs[name] = {"kind": "file", "path": str(path)}
    recipe_path = tmp_path / "family-gate-v2.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-family-gate-v2-fixture",
                "description": "fixture",
                "external_inputs": external_inputs,
                "steps": [
                    {
                        "id": "family_gate",
                        "op": "glm52-family-gate-check",
                        "class": "diagnostic",
                        "gate": {"profile": "glm52_family"},
                        "inputs": {
                            name: f"external:{name}"
                            for name in (*required_inputs, *raw_inputs)
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )

    step = plan_recipe(
        load_recipe(recipe_path), build_root=tmp_path / "build"
    ).step("family_gate")

    assert step.op_def.required_inputs == required_inputs
    assert step.op_def.optional_inputs == (
        "family_eval_gate_json",
        "family_benchmark_gate_json",
        *raw_inputs,
        "teacher_cache_root",
        "teacher_cache_manifest_json",
        "candidate_cache_root",
        "source_route_trace_root",
        "candidate_route_trace_root",
        "source_route_authority_json",
        "candidate_route_authority_json",
        "benchmark_evidence_json",
    )
    assert [token for token in step.argv if token.startswith("--")] == [
        "--family-policy-json",
        "--eval-prompt-pack-json",
        "--teacher-metadata-json",
        "--full-bind-preflight-json",
        "--indexshare-runtime-json",
        "--synthetic-generation-json",
        "--non-vq-evidence-json",
        "--composite-artifact-audit-json",
        "--production-generation-json",
        "--output-json",
    ]
    for name, path in paths.items():
        flag = f"--{name.replace('_', '-')}"
        assert step.argv[step.argv.index(flag) + 1] == str(path)
    assert step.spec.params == {}
    assert step.op_def.allowed_params == frozenset()
    assert step.op_def.required_params == ()
    assert step.op_def.cacheable is False
    assert step.op_def.accepted_returncodes == frozenset({0, 2})
    assert step.op_def.required_gate_profile == "glm52_family"
    assert step.op_def.regate_can_complete is False


def test_glm52_family_gate_op_accepts_optional_raw_teacher_cache_inputs(
    tmp_path: Path,
) -> None:
    op = REGISTRY["glm52-family-gate-check"]
    inputs = {
        name: tmp_path / name
        for name in (
            *op.required_inputs,
            "non_vq_evidence_json",
            "composite_artifact_audit_json",
            "production_generation_json",
            "teacher_cache_root",
            "teacher_cache_manifest_json",
        )
    }
    context = StepContext(
        spec=StepSpec(
            id="family_gate",
            op=op.name,
            step_class="diagnostic",
            inputs={},
            params={},
            gate=None,
        ),
        inputs=inputs,
        out_dir=tmp_path / "out",
        evidence_dir=tmp_path / "evidence",
    )

    argv = op.build_argv(context)

    assert argv[argv.index("--teacher-cache-root") + 1] == str(
        inputs["teacher_cache_root"]
    )
    assert argv[argv.index("--teacher-cache-manifest-json") + 1] == str(
        inputs["teacher_cache_manifest_json"]
    )
    assert op.version == 1


def test_glm52_release_evidence_ops_wrap_the_real_cli_subcommands(
    tmp_path: Path,
) -> None:
    candidate = REGISTRY["glm52-candidate-full-vocab-eval-produce"]
    candidate_context = StepContext(
        spec=StepSpec(
            id="candidate_eval",
            op=candidate.name,
            step_class="diagnostic",
            inputs={},
            params={},
            gate=None,
        ),
        inputs={name: tmp_path / name for name in candidate.required_inputs},
        out_dir=tmp_path / "candidate-out",
        evidence_dir=tmp_path / "candidate-evidence",
    )
    candidate_argv = candidate.build_argv(candidate_context)
    assert candidate_argv[:5] == [
        "uv",
        "run",
        "python",
        "benchmarks/eval_glm52_candidate_full_vocab.py",
        "produce",
    ]
    assert "--candidate-cache-root" in candidate_argv
    assert candidate.self_managed_heavy_lock is True

    teacher = REGISTRY["glm52-teacher-cache-produce"]
    teacher_context = StepContext(
        spec=StepSpec(
            id="teacher_cache", op=teacher.name, step_class="diagnostic", inputs={}, params={}, gate=None
        ),
        inputs={name: tmp_path / name for name in teacher.required_inputs},
        out_dir=tmp_path / "teacher-cache",
        evidence_dir=tmp_path / "teacher-evidence",
    )
    teacher_argv = teacher.build_argv(teacher_context)
    assert teacher_argv[teacher_argv.index("--route-trace-root") + 1] == str(
        teacher_context.out_dir / "../source-route-traces"
    )
    assert teacher.output_paths(teacher_context)["route_trace_root"] == (
        teacher_context.out_dir / "../source-route-traces"
    )

    route_capture = REGISTRY["glm52-route-math-capture"]
    route_context = StepContext(
        spec=StepSpec(
            id="source_route",
            op=route_capture.name,
            step_class="diagnostic",
            inputs={},
            params={"side": "source", "expert_count": 168},
            gate=None,
        ),
        inputs={"input_root": tmp_path / "source-npz"},
        out_dir=tmp_path / "source-route",
        evidence_dir=tmp_path / "route-evidence",
    )
    route_argv = route_capture.build_argv(route_context)
    assert route_argv[:5] == [
        "uv",
        "run",
        "python",
        "benchmarks/check_glm52_route_math.py",
        "capture",
    ]
    assert route_argv[route_argv.index("--trace-root") + 1] == str(
        route_context.out_dir
    )
    assert route_capture.self_managed_heavy_lock is False

    pair = REGISTRY["glm52-same-machine-benchmark-pair"]
    pair_context = StepContext(
        spec=StepSpec(
            id="same_machine_pair",
            op=pair.name,
            step_class="diagnostic",
            inputs={},
            params={},
            gate=None,
        ),
        inputs={name: tmp_path / name for name in pair.required_inputs},
        out_dir=tmp_path / "pair-out",
        evidence_dir=tmp_path / "pair-evidence",
    )
    pair_argv = pair.build_argv(pair_context)
    assert pair_argv[:5] == [
        "uv",
        "run",
        "python",
        "benchmarks/bench_glm52_same_machine_fp4.py",
        "pair",
    ]
    assert [token for token in pair_argv if token.startswith("--")][-3:] == [
        "--candidate-output-json",
        "--control-output-json",
        "--session-manifest-json",
    ]
    assert pair.self_managed_heavy_lock is True


def test_checked_schema_v2_family_evidence_recipe_wires_exact_raw_files() -> None:
    assert V2_EVIDENCE_RECIPE.is_file()
    recipe = load_recipe(V2_EVIDENCE_RECIPE)
    plan = plan_recipe(recipe)
    spec = recipe.step("family_gate")
    step = plan.step("family_gate")

    assert [item.id for item in recipe.steps] == ["family_gate"]
    assert recipe.strict_inputs is True
    assert tuple(recipe.external_inputs) == tuple(V2_EXTERNAL_EVIDENCE)
    for name, (path, digest) in V2_EXTERNAL_EVIDENCE.items():
        external = recipe.external_inputs[name]
        assert external.kind == "file"
        assert external.path == path
        assert external.expect_hash == digest
    assert {
        name: reference.raw for name, reference in spec.inputs.items()
    } == {
        name: f"external:{name}" for name in V2_EXTERNAL_EVIDENCE
    }
    expected_hashes = {
        name: digest for name, (_path, digest) in V2_EXTERNAL_EVIDENCE.items()
    }
    assert plan.external_hashes == expected_hashes
    assert step.input_hashes == expected_hashes
    assert [token for token in step.argv if token.startswith("--")] == [
        "--family-policy-json",
        "--eval-prompt-pack-json",
        "--teacher-metadata-json",
        "--full-bind-preflight-json",
        "--indexshare-runtime-json",
        "--synthetic-generation-json",
        "--non-vq-evidence-json",
        "--composite-artifact-audit-json",
        "--production-generation-json",
        "--output-json",
    ]
    for name, (path, _digest) in V2_EXTERNAL_EVIDENCE.items():
        flag = f"--{name.replace('_', '-')}"
        assert step.argv[step.argv.index(flag) + 1] == path
    assert spec.params == {}
    assert step.op_def.allowed_params == frozenset()
    assert step.op_def.required_params == ()
    assert spec.gate is not None
    assert spec.gate.profile == "glm52_family"
    assert step.op_def.cacheable is False
    assert step.op_def.accepted_returncodes == frozenset({0, 2})
    assert step.op_def.required_gate_profile == "glm52_family"
    assert step.op_def.regate_can_complete is False


def test_checked_schema_v2_family_evidence_recipe_rejects_hash_drift(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(V2_EVIDENCE_RECIPE.read_text())
    raw["external_inputs"]["family_policy_json"]["expect_hash"] = "0" * 64
    recipe_path = tmp_path / V2_EVIDENCE_RECIPE.name
    recipe_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(RecipeError, match="hash mismatch"):
        plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")


def test_checked_family_preflight_ends_in_intentional_fail_loud_gate() -> None:
    recipe = load_recipe(PREFLIGHT_RECIPE)
    plan = plan_recipe(recipe)

    assert [step.id for step in recipe.steps] == [
        "source_audit",
        "source_payload_audit",
        "indexshare_runtime",
        "tokenizer_readiness",
        "synthetic_generation",
        "full_bind_evidence",
        "family_policy",
        "eval_prompts",
        "teacher_metadata",
        "family_gate",
    ]
    assert recipe.step("source_payload_audit").inputs[
        "source_audit_evidence"
    ].raw == "step:source_audit/evidence_json"
    assert recipe.step("teacher_metadata").inputs[
        "source_payload_audit_json"
    ].raw == "step:source_payload_audit/evidence_json"
    assert {
        name: reference.raw
        for name, reference in recipe.step("family_gate").inputs.items()
    } == {
        "family_policy_json": "step:family_policy/evidence_json",
        "eval_prompt_pack_json": "step:eval_prompts/evidence_json",
        "teacher_metadata_json": "step:teacher_metadata/evidence_json",
        "full_bind_preflight_json": "step:full_bind_evidence/evidence_json",
        "indexshare_runtime_json": "step:indexshare_runtime/evidence_json",
        "synthetic_generation_json": "step:synthetic_generation/evidence_json",
    }
    assert plan.step("family_gate").op_def.cacheable is False
    assert plan.step("family_gate").op_def.accepted_returncodes == frozenset({0, 2})
    assert plan.step("family_gate").op_def.required_gate_profile == "glm52_family"
    assert plan.step("family_gate").op_def.regate_can_complete is False
    assert recipe.step("family_gate").gate is not None
    assert recipe.step("family_gate").gate.profile == "glm52_family"
    assert recipe.strict_inputs is True
    full_bind = plan.step("full_bind_evidence")
    assert full_bind.spec.op == "glm52-full-bind-evidence"
    assert "--evidence-only" in full_bind.argv
    assert full_bind.input_hashes["non_vq_artifact"] == (
        "42112475c610abe884347ccb75d099cfcab7455d47a7a822b91212daee3b3f99"
    )
    assert full_bind.input_hashes["routed_artifact"] == (
        "74ed26bb0e88d30cfd9d1b23f87d7b14122c5f660dbda42ea1c331a419ae53c8"
    )
    assert "--family-eval-gate-json" not in plan.step("family_gate").argv
    assert "--family-benchmark-gate-json" not in plan.step("family_gate").argv
    assert "--production-generation-json" not in plan.step("family_gate").argv


def test_glm52_family_gate_op_rejects_wrong_declarative_gate_profile(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(PREFLIGHT_RECIPE.read_text())
    family_gate = next(
        step for step in raw["steps"] if step["id"] == "family_gate"
    )
    family_gate["gate"] = {"profile": "audit_ok"}
    recipe_path = tmp_path / "wrong-family-gate.yaml"
    recipe_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(RecipeError, match="requires gate profile 'glm52_family'"):
        plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
