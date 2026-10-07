from __future__ import annotations

import json
from pathlib import Path

import yaml

from keep.build import executor
from keep.build.recipe import load_recipe
from keep.build.runner import plan_recipe


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
RUNTIME_RECIPE = Path("recipes/glm52_reap_504b_runtime_preflight_20260709.yaml")


def _write_runtime_recipe(
    tmp_path: Path,
    *,
    include_production_probe: bool = False,
) -> Path:
    tokenizer_dir = tmp_path / "snapshot"
    tokenizer_dir.mkdir()
    non_vq_artifact = tmp_path / "non-vq"
    routed_artifact = tmp_path / "routed"
    non_vq_artifact.mkdir()
    routed_artifact.mkdir()
    config_path = tokenizer_dir / "config.json"
    index_path = tokenizer_dir / "model.safetensors.index.json"
    profile_path = tmp_path / "profile.yaml"
    family_policy_path = tmp_path / "family-policy.json"
    non_vq_evidence_path = tmp_path / "non-vq-evidence.json"
    composite_audit_path = tmp_path / "composite-audit.json"
    materialization_runs_path = tmp_path / "materialization-runs.jsonl"
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}) + "\n")
    index_path.write_text(json.dumps({"weight_map": {}}) + "\n")
    profile_path.write_text("name: glm52-reap-fixture\n")
    for path in (
        family_policy_path,
        non_vq_evidence_path,
        composite_audit_path,
    ):
        path.write_text("{}\n")
    materialization_runs_path.write_text("{}\n")
    steps = [
        {
            "id": "indexshare_runtime",
            "op": "glm52-indexshare-runtime-probe",
            "class": "diagnostic",
            "inputs": {
                "config_path": "external:config_path",
                "index_path": "external:index_path",
            },
            "params": {"model_id": MODEL_ID, "revision": REVISION},
        },
        {
            "id": "tokenizer_readiness",
            "op": "glm52-tokenizer-readiness-probe",
            "class": "diagnostic",
            "inputs": {
                "tokenizer_dir": "external:tokenizer_dir",
                "config_path": "external:config_path",
            },
            "params": {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "prompt": "The capital of France is",
            },
        },
        {
            "id": "synthetic_generation",
            "op": "glm52-synthetic-generation-probe",
            "class": "diagnostic",
        },
        {
            "id": "full_bind_preflight",
            "op": "glm52-full-bind-preflight",
            "class": "diagnostic",
            "inputs": {
                "profile_path": "external:profile_path",
                "config_path": "external:config_path",
                "source_index_path": "external:index_path",
                "non_vq_artifact": "external:non_vq_artifact",
                "routed_artifact": "external:routed_artifact",
            },
            "params": {"model_id": MODEL_ID, "revision": REVISION},
        },
    ]
    if include_production_probe:
        steps.append(
            {
                "id": "production_generation",
                "op": "glm52-production-generation-probe",
                "class": "diagnostic",
                "inputs": {
                    "profile_path": "external:profile_path",
                    "config_path": "external:config_path",
                    "source_index_path": "external:index_path",
                    "tokenizer_dir": "external:tokenizer_dir",
                    "tokenizer_readiness_json": (
                        "step:tokenizer_readiness/evidence_json"
                    ),
                    "family_policy_json": "external:family_policy_json",
                    "non_vq_artifact": "external:non_vq_artifact",
                    "non_vq_evidence_json": "external:non_vq_evidence_json",
                    "routed_artifact": "external:routed_artifact",
                    "composite_audit_json": "external:composite_audit_json",
                    "materialization_runs": "external:materialization_runs",
                    "full_bind_preflight_json": (
                        "step:full_bind_preflight/evidence_json"
                    ),
                },
                "params": {
                    "model_id": MODEL_ID,
                    "revision": REVISION,
                    "prompt": "The capital of France is",
                },
            }
        )
    recipe_path = tmp_path / "runtime-probes.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-runtime-probe-fixture",
                "description": "fixture",
                "external_inputs": {
                    "tokenizer_dir": {
                        "kind": "artifact_dir",
                        "path": str(tokenizer_dir),
                    },
                    "config_path": {"kind": "file", "path": str(config_path)},
                    "index_path": {"kind": "file", "path": str(index_path)},
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "non_vq_artifact": {
                        "kind": "artifact_dir",
                        "path": str(non_vq_artifact),
                    },
                    "routed_artifact": {
                        "kind": "artifact_dir",
                        "path": str(routed_artifact),
                    },
                    "family_policy_json": {
                        "kind": "file",
                        "path": str(family_policy_path),
                    },
                    "non_vq_evidence_json": {
                        "kind": "file",
                        "path": str(non_vq_evidence_path),
                    },
                    "composite_audit_json": {
                        "kind": "file",
                        "path": str(composite_audit_path),
                    },
                    "materialization_runs": {
                        "kind": "file",
                        "path": str(materialization_runs_path),
                    },
                },
                "steps": steps,
            },
            sort_keys=False,
        )
    )
    return recipe_path


def test_glm52_runtime_probe_ops_render_exact_local_only_commands(tmp_path: Path) -> None:
    plan = plan_recipe(
        load_recipe(_write_runtime_recipe(tmp_path)),
        build_root=tmp_path / "build",
    )

    indexshare = plan.step("indexshare_runtime")
    assert indexshare.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/probe_glm52_indexshare_runtime.py",
    ]
    assert indexshare.argv[indexshare.argv.index("--model-id") + 1] == MODEL_ID
    assert indexshare.argv[indexshare.argv.index("--revision") + 1] == REVISION
    assert indexshare.op_def.cacheable is False
    assert indexshare.output_paths == {
        "evidence_json": indexshare.evidence_dir / "evidence_json.json"
    }

    tokenizer = plan.step("tokenizer_readiness")
    assert tokenizer.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/probe_glm52_tokenizer_readiness.py",
    ]
    assert tokenizer.argv[tokenizer.argv.index("--prompt") + 1] == (
        "The capital of France is"
    )
    assert tokenizer.op_def.cacheable is False
    assert tokenizer.output_paths == {
        "evidence_json": tokenizer.evidence_dir / "evidence_json.json"
    }

    generation = plan.step("synthetic_generation")
    assert generation.argv == [
        "uv",
        "run",
        "python",
        "benchmarks/probe_glm52_synthetic_generation.py",
        "--output-json",
        str(generation.evidence_dir / "evidence_json.json"),
    ]
    assert generation.op_def.cacheable is False
    assert generation.output_paths == {
        "evidence_json": generation.evidence_dir / "evidence_json.json"
    }

    preflight = plan.step("full_bind_preflight")
    assert preflight.argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/probe_glm52_full_bind_preflight.py",
    ]
    assert "--source-index-path" in preflight.argv
    assert "--non-vq-artifact-dir" in preflight.argv
    assert "--routed-artifact-dir" in preflight.argv
    assert preflight.op_def.cacheable is False
    assert preflight.output_paths == {
        "evidence_json": preflight.evidence_dir / "evidence_json.json"
    }


def test_checked_runtime_preflight_recipe_keeps_blocking_full_bind_last() -> None:
    recipe = load_recipe(RUNTIME_RECIPE)
    plan = plan_recipe(recipe)

    assert [step.id for step in recipe.steps] == [
        "indexshare_runtime",
        "tokenizer_readiness",
        "synthetic_generation",
        "full_bind_preflight",
    ]
    assert plan.step("indexshare_runtime").op_def.cacheable is False
    assert plan.step("tokenizer_readiness").op_def.cacheable is False
    assert plan.step("synthetic_generation").op_def.cacheable is False
    assert plan.step("full_bind_preflight").op_def.cacheable is False
    assert recipe.step("tokenizer_readiness").params["prompt"] == (
        "The capital of France is"
    )
    assert recipe.external_inputs["non_vq_artifact"].path.endswith(
        "non_vq_pack-a678bc3c/out"
    )
    assert recipe.external_inputs["routed_artifact"].path.endswith(
        "e8-layers3-4-w1"
    )
    assert all(
        step.op != "glm52-production-generation-probe"
        for step in recipe.steps
    )


def test_glm52_production_probe_requires_all_authorities_and_heavy_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recipe = load_recipe(
        _write_runtime_recipe(tmp_path, include_production_probe=True)
    )
    plan = plan_recipe(recipe, build_root=tmp_path / "build")
    production = plan.step("production_generation")
    tokenizer_readiness = plan.step("tokenizer_readiness")
    full_bind = plan.step("full_bind_preflight")

    assert production.op_def.required_inputs == (
        "profile_path",
        "config_path",
        "source_index_path",
        "tokenizer_dir",
        "tokenizer_readiness_json",
        "family_policy_json",
        "non_vq_artifact",
        "non_vq_evidence_json",
        "routed_artifact",
        "composite_audit_json",
        "materialization_runs",
        "full_bind_preflight_json",
    )
    assert production.argv == [
        "uv",
        "run",
        "python",
        "benchmarks/probe_glm52_production_generation.py",
        "--profile-path",
        str(tmp_path / "profile.yaml"),
        "--config-path",
        str(tmp_path / "snapshot/config.json"),
        "--source-index-path",
        str(tmp_path / "snapshot/model.safetensors.index.json"),
        "--tokenizer-dir",
        str(tmp_path / "snapshot"),
        "--tokenizer-readiness-json",
        str(tokenizer_readiness.evidence_dir / "evidence_json.json"),
        "--family-policy-json",
        str(tmp_path / "family-policy.json"),
        "--non-vq-artifact-dir",
        str(tmp_path / "non-vq"),
        "--non-vq-evidence-json",
        str(tmp_path / "non-vq-evidence.json"),
        "--routed-artifact-dir",
        str(tmp_path / "routed"),
        "--composite-audit-json",
        str(tmp_path / "composite-audit.json"),
        "--materialization-runs-jsonl",
        str(tmp_path / "materialization-runs.jsonl"),
        "--full-bind-preflight-json",
        str(full_bind.evidence_dir / "evidence_json.json"),
        "--output-json",
        str(production.evidence_dir / "evidence_json.json"),
        "--model-id",
        MODEL_ID,
        "--prompt",
        "The capital of France is",
        "--revision",
        REVISION,
    ]
    assert production.op_def.cacheable is False
    assert production.op_def.produces_artifact is False
    assert production.op_def.accepted_returncodes == frozenset({0})
    assert production.output_paths == {
        "evidence_json": production.evidence_dir / "evidence_json.json"
    }

    monkeypatch.delenv("GLM45_DISABLE_HEAVY_LOCK", raising=False)
    assert "glm52-production-generation-probe" in executor.MODEL_LOADING_OPS
    assert executor._uses_heavy_job_lock(production) is True
    lock_path = tmp_path / ".keep-heavy-job.lock"
    monkeypatch.setattr(executor, "HEAVY_JOB_LOCK_PATH", lock_path)
    lock_calls: list[int] = []
    monkeypatch.setattr(
        executor.fcntl,
        "flock",
        lambda _fileno, operation: lock_calls.append(operation),
    )
    with executor._heavy_job_lock(production):
        assert lock_path.exists()
    assert lock_calls == [executor.fcntl.LOCK_EX, executor.fcntl.LOCK_UN]

    monkeypatch.setenv("GLM45_DISABLE_HEAVY_LOCK", "1")
    assert executor._uses_heavy_job_lock(production) is False
