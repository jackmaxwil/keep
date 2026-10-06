from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "audit_air_teacher_feasibility.py"
    spec = importlib.util.spec_from_file_location(
        "air_teacher_feasibility_test",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


def test_discover_q8_air_candidates_ignores_lower_bit_artifacts(tmp_path) -> None:
    q8_snapshot = (
        tmp_path
        / "models--mlx-community--GLM-4.5-Air-8bit"
        / "snapshots"
        / "abc123"
    )
    q8_snapshot.mkdir(parents=True)
    (q8_snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"x": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    (q8_snapshot / "model-00001.safetensors").write_bytes(b"q8")

    q2_artifact = tmp_path / "glm-4.5-air-mlx-q2-routed-g128"
    q2_artifact.mkdir()
    (q2_artifact / "model-00001.safetensors").write_bytes(b"q2")

    candidates = audit.discover_q8_air_candidates([tmp_path])

    assert len(candidates) == 1
    assert candidates[0]["path"].endswith("models--mlx-community--GLM-4.5-Air-8bit")
    assert candidates[0]["has_index"] is True
    assert candidates[0]["safetensor_file_count"] == 1


def test_feasibility_record_reports_q8_candidates_without_promoting_q2(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "_device_info",
        lambda: {"max_recommended_working_set_size": 128 * 1024**3},
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "model-00001.safetensors").write_bytes(b"bf16-source")
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": {"x": "model-00001.safetensors"}}),
        encoding="utf-8",
    )

    q8_dir = tmp_path / "GLM-4.5-Air-Q8"
    q8_dir.mkdir()
    (q8_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"x": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    (q8_dir / "model-00001.safetensors").write_bytes(b"q8")

    q2_dir = tmp_path / "glm-4.5-air-mlx-q2-routed-g128"
    q2_dir.mkdir()
    (q2_dir / "model-00001.safetensors").write_bytes(b"q2")

    record = audit.build_feasibility_record(
        model_id="zai-org/GLM-4.5-Air",
        revision="test",
        source_dir=source_dir,
        index_path=index_path,
        vq_artifact_dir=tmp_path / "missing-vq",
        q2_artifact_dir=q2_dir,
        host_budget_bytes=128 * 1024**3,
        q8_search_roots=[tmp_path],
    )

    q8 = record["teacher_candidates"]["q8_full_estimate"]
    q2 = record["teacher_candidates"]["mlx_q2_routed_g128"]
    assert q8["available"] is True
    assert q8["candidate_count"] == 1
    assert q8["decision"] == "candidate_found_needs_quality_provenance_review"
    assert q8["candidates"][0]["path"].endswith("GLM-4.5-Air-Q8")
    assert q2["available"] is True
    assert q2["decision"] == "local_reference_not_high_bit_teacher"
