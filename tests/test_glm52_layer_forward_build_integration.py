from __future__ import annotations

import json
from pathlib import Path

import yaml

from mlx_vq.build.ops import REGISTRY
from mlx_vq.build.recipe import load_recipe
from mlx_vq.build.runner import plan_recipe


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
RECIPE_PATH = Path("recipes/glm52_reap_504b_materialization_probe_20260709.yaml")
ROUTED_ARTIFACT_PATH = Path(
    "artifacts/quality/glm52-wave2-materialization-20260709/e8-layers3-4-w1"
)
ROUTED_AUDIT_PATH = Path(
    "artifacts/quality/glm52-wave3-artifact-audit-20260709/"
    "e8-layers3-4-audit.json"
)


def test_glm52_layer_forward_probe_op_renders_exact_live_cli(
    tmp_path: Path,
) -> None:
    non_vq_artifact = tmp_path / "non-vq"
    routed_artifact = tmp_path / "routed"
    non_vq_artifact.mkdir()
    routed_artifact.mkdir()
    non_vq_evidence = tmp_path / "non-vq-evidence.json"
    routed_audit = tmp_path / "routed-audit.json"
    profile_path = tmp_path / "profile.yaml"
    config_path = tmp_path / "config.json"
    non_vq_evidence.write_text(json.dumps({"production_ready": True}) + "\n")
    routed_audit.write_text(json.dumps({"audit_pass": True}) + "\n")
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}) + "\n")

    recipe_path = tmp_path / "layer-forward.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-layer-forward-fixture",
                "description": "fixture",
                "external_inputs": {
                    "non_vq_artifact": {
                        "kind": "artifact_dir",
                        "path": str(non_vq_artifact),
                    },
                    "non_vq_evidence_json": {
                        "kind": "file",
                        "path": str(non_vq_evidence),
                    },
                    "routed_artifact": {
                        "kind": "artifact_dir",
                        "path": str(routed_artifact),
                    },
                    "routed_audit_json": {
                        "kind": "file",
                        "path": str(routed_audit),
                    },
                    "profile_path": {"kind": "file", "path": str(profile_path)},
                    "config_path": {"kind": "file", "path": str(config_path)},
                },
                "steps": [
                    {
                        "id": "layer_forward_probe",
                        "op": "glm52-layer-forward-probe",
                        "class": "diagnostic",
                        "inputs": {
                            "non_vq_artifact": "external:non_vq_artifact",
                            "non_vq_evidence_json": (
                                "external:non_vq_evidence_json"
                            ),
                            "routed_artifact": "external:routed_artifact",
                            "routed_audit_json": "external:routed_audit_json",
                            "profile_path": "external:profile_path",
                            "config_path": "external:config_path",
                        },
                        "params": {
                            "model_id": MODEL_ID,
                            "revision": REVISION,
                            "layer": 3,
                            "tokens": 1,
                            "seed": 20260709,
                            "input_scale": 0.125,
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("layer_forward_probe")
    argv = step.argv

    assert argv[:4] == [
        "uv",
        "run",
        "python",
        "benchmarks/probe_glm52_layer_forward.py",
    ]
    expected_input_flags = {
        "--non-vq-artifact": non_vq_artifact,
        "--non-vq-evidence-json": non_vq_evidence,
        "--routed-artifact": routed_artifact,
        "--routed-audit-json": routed_audit,
        "--profile-path": profile_path,
        "--config-path": config_path,
    }
    for flag, expected_path in expected_input_flags.items():
        assert argv[argv.index(flag) + 1] == str(expected_path)
    assert argv[argv.index("--model-id") + 1] == MODEL_ID
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--layer") + 1] == "3"
    assert argv[argv.index("--tokens") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == "20260709"
    assert argv[argv.index("--input-scale") + 1] == "0.125"
    assert argv[argv.index("--output-json") + 1] == str(
        step.evidence_dir / "evidence_json.json"
    )
    assert step.op_def.cacheable is False
    assert step.op_def.produces_artifact is False
    assert step.output_paths == {
        "evidence_json": step.evidence_dir / "evidence_json.json"
    }


def test_glm52_materialization_recipe_ends_with_external_layer3_forward_probe(
) -> None:
    recipe = load_recipe(RECIPE_PATH)

    assert [step.id for step in recipe.steps][-1] == "layer_forward_probe"
    assert recipe.step("materialize_groups").params["groups"] == [
        "10:gate_proj",
        "29:gate_proj",
    ]
    probe = recipe.step("layer_forward_probe")
    expected_refs = {
        "non_vq_artifact": ("step", "non_vq_pack", "artifact"),
        "non_vq_evidence_json": ("step", "non_vq_pack", "evidence_json"),
        "routed_artifact": ("external", "routed_layer3_artifact", None),
        "routed_audit_json": ("external", "routed_layer3_audit", None),
        "profile_path": ("external", "profile_path", None),
        "config_path": ("external", "config_path", None),
    }
    assert {
        name: (ref.kind, ref.name, ref.output)
        for name, ref in probe.inputs.items()
    } == expected_refs
    assert probe.params == {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "layer": 3,
        "tokens": 1,
        "seed": 20260709,
        "input_scale": 0.125,
    }
    assert recipe.external_inputs["routed_layer3_artifact"].kind == "artifact_dir"
    assert recipe.external_inputs["routed_layer3_artifact"].path == str(
        ROUTED_ARTIFACT_PATH
    )
    assert recipe.external_inputs["routed_layer3_audit"].kind == "file"
    assert recipe.external_inputs["routed_layer3_audit"].path == str(
        ROUTED_AUDIT_PATH
    )
    assert REGISTRY[probe.op].cacheable is False
