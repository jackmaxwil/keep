from __future__ import annotations

import json
from pathlib import Path

import yaml

from mlx_vq.build.ops import OpDef, REGISTRY
from mlx_vq.build.recipe import load_recipe
from mlx_vq.build.runner import plan_recipe


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
CONFIG_SHA256 = "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
INDEX_SHA256 = "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"


def _write_external_inputs(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    profile_path = tmp_path / "profile.yaml"
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    payload_audit_path = tmp_path / "payload-audit.json"
    profile_path.write_text("name: glm52-reap-504b-v2\n")
    config_path.write_text(json.dumps({"model_type": "glm_moe_dsa"}) + "\n")
    index_path.write_text(json.dumps({"weight_map": {}}) + "\n")
    payload_audit_path.write_text(
        json.dumps({"full_payload_ready": True}) + "\n"
    )
    paths = {
        "source_dir": source_dir,
        "profile_path": profile_path,
        "config_path": config_path,
        "index_path": index_path,
        "payload_audit_evidence": payload_audit_path,
    }
    external_inputs = {
        "source_dir": {"kind": "artifact_dir", "path": str(source_dir)},
        "profile_path": {"kind": "file", "path": str(profile_path)},
        "config_path": {"kind": "file", "path": str(config_path)},
        "index_path": {"kind": "file", "path": str(index_path)},
        "payload_audit_evidence": {
            "kind": "file",
            "path": str(payload_audit_path),
        },
    }
    return external_inputs, paths


def _pack_step(*, payload_ref: str) -> dict[str, object]:
    return {
        "id": "non_vq_pack",
        "op": "glm52-non-vq-pack",
        "class": "diagnostic",
        "inputs": {
            "source_dir": "external:source_dir",
            "profile_path": "external:profile_path",
            "config_path": "external:config_path",
            "index_path": "external:index_path",
            "payload_audit_evidence": payload_ref,
        },
        "params": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "max_shard_payload_bytes": 4 * 1024**3,
        },
    }


def test_glm52_non_vq_pack_op_maps_resumable_cli_and_exact_outputs(
    tmp_path: Path,
) -> None:
    external_inputs, paths = _write_external_inputs(tmp_path)
    recipe_path = tmp_path / "pack.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-non-vq-pack-fixture",
                "description": "fixture",
                "external_inputs": external_inputs,
                "steps": [
                    _pack_step(payload_ref="external:payload_audit_evidence")
                ],
            },
            sort_keys=False,
        )
    )

    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    step = plan.step("non_vq_pack")
    argv = step.argv

    assert argv[:4] == [
        "uv",
        "run",
        "python",
        "src/mlx_vq/convert/glm52_non_vq.py",
    ]
    assert "--resume" in argv
    assert argv[argv.index("--source-dir") + 1] == str(paths["source_dir"])
    assert argv[argv.index("--profile-path") + 1] == str(paths["profile_path"])
    assert argv[argv.index("--config-path") + 1] == str(paths["config_path"])
    assert argv[argv.index("--index-path") + 1] == str(paths["index_path"])
    assert argv[argv.index("--model-id") + 1] == MODEL_ID
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--config-sha256") + 1] == CONFIG_SHA256
    assert argv[argv.index("--index-sha256") + 1] == INDEX_SHA256
    assert argv[argv.index("--max-shard-payload-bytes") + 1] == str(
        4 * 1024**3
    )
    assert argv[argv.index("--output-dir") + 1] == str(step.out_dir)
    assert argv[argv.index("--output-json") + 1] == str(
        step.evidence_dir / "evidence_json.json"
    )
    assert str(paths["payload_audit_evidence"]) not in argv

    assert step.op_def.resume_partial is True
    assert step.op_def.manages_partial_files is True
    assert step.op_def.cacheable is True
    assert step.op_def.produces_artifact is True
    assert step.op_def.artifact_manifest_name == "non-vq-manifest.json"
    assert step.output_paths == {
        "artifact": step.out_dir,
        "manifest": step.out_dir / "non-vq-manifest.json",
        "index": step.out_dir / "model.safetensors.index.json",
        "evidence_json": step.evidence_dir / "evidence_json.json",
    }


def test_glm52_non_vq_pack_dag_bridges_audits_to_later_bind_refs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_inputs, _paths = _write_external_inputs(tmp_path)
    source_audit_evidence = tmp_path / "source-audit.json"
    source_audit_evidence.write_text(json.dumps({"audit_blockers": []}) + "\n")
    external_inputs["source_audit_evidence"] = {
        "kind": "file",
        "path": str(source_audit_evidence),
    }
    monkeypatch.setitem(
        REGISTRY,
        "test-glm52-bind-probe",
        OpDef(
            name="test-glm52-bind-probe",
            version=1,
            allowed_classes=frozenset({"diagnostic"}),
            script="tests/fake_glm52_bind_probe.py",
            required_inputs=(
                "non_vq_artifact",
                "non_vq_manifest",
                "non_vq_index",
                "pack_evidence",
            ),
        ),
    )
    recipe_path = tmp_path / "chain.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-non-vq-dag-fixture",
                "description": "fixture",
                "external_inputs": external_inputs,
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
                            "model_id": MODEL_ID,
                            "revision": REVISION,
                        },
                    },
                    {
                        "id": "source_payload_audit",
                        "op": "glm52-source-payload-audit",
                        "class": "diagnostic",
                        "inputs": {
                            "source_dir": "external:source_dir",
                            "profile_path": "external:profile_path",
                            "config_path": "external:config_path",
                            "index_path": "external:index_path",
                            "source_audit_evidence": (
                                "step:source_audit/evidence_json"
                            ),
                        },
                        "params": {
                            "model_id": MODEL_ID,
                            "revision": REVISION,
                        },
                    },
                    _pack_step(
                        payload_ref="step:source_payload_audit/evidence_json"
                    ),
                    {
                        "id": "bind_probe",
                        "op": "test-glm52-bind-probe",
                        "class": "diagnostic",
                        "inputs": {
                            "non_vq_artifact": "step:non_vq_pack/artifact",
                            "non_vq_manifest": "step:non_vq_pack/manifest",
                            "non_vq_index": "step:non_vq_pack/index",
                            "pack_evidence": "step:non_vq_pack/evidence_json",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )

    plan = plan_recipe(
        load_recipe(recipe_path),
        build_root=tmp_path / "build",
        no_hash=True,
    )

    assert [step.spec.id for step in plan.steps] == [
        "source_audit",
        "source_payload_audit",
        "non_vq_pack",
        "bind_probe",
    ]
    payload_input = plan.recipe.step("non_vq_pack").inputs[
        "payload_audit_evidence"
    ]
    assert (payload_input.kind, payload_input.name, payload_input.output) == (
        "step",
        "source_payload_audit",
        "evidence_json",
    )
    pack = plan.step("non_vq_pack")
    bind = plan.step("bind_probe")
    assert bind.resolved_inputs == {
        "non_vq_artifact": pack.output_paths["artifact"],
        "non_vq_manifest": pack.output_paths["manifest"],
        "non_vq_index": pack.output_paths["index"],
        "pack_evidence": pack.output_paths["evidence_json"],
    }
    assert bind.input_hashes == {
        "non_vq_artifact": f"{pack.step_key}/artifact",
        "non_vq_manifest": f"{pack.step_key}/manifest",
        "non_vq_index": f"{pack.step_key}/index",
        "pack_evidence": f"{pack.step_key}/evidence_json",
    }
