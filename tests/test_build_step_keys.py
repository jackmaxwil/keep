from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_vq.build import hashing


def test_step_key_is_stable_and_param_order_independent() -> None:
    key_a = hashing.step_key(
        op="train-low-rank",
        op_version=1,
        params={"layer": 45, "low_rank": 4},
        input_keys={"seed_artifact": "abc"},
    )
    key_b = hashing.step_key(
        op="train-low-rank",
        op_version=1,
        params={"low_rank": 4, "layer": 45},
        input_keys={"seed_artifact": "abc"},
    )
    assert key_a == key_b
    assert len(key_a) == hashing.STEP_KEY_LENGTH


def test_step_key_changes_with_params_inputs_and_op_version() -> None:
    base = dict(
        op="train-low-rank",
        op_version=1,
        params={"layer": 45, "learning_rate": 0.5},
        input_keys={"seed_artifact": "abc"},
    )
    key = hashing.step_key(**base)
    assert hashing.step_key(**{**base, "params": {"layer": 45, "learning_rate": 0.6}}) != key
    assert hashing.step_key(**{**base, "input_keys": {"seed_artifact": "def"}}) != key
    assert hashing.step_key(**{**base, "op_version": 2}) != key


def test_gate_key_independent_of_step_key() -> None:
    gate_a = hashing.gate_key(profile="lane_s", thresholds={"lane_s_ratio_max": 1.15})
    gate_b = hashing.gate_key(profile="lane_s", thresholds={"lane_s_ratio_max": 1.10})
    assert gate_a != gate_b


def _write_fake_artifact(root: Path) -> Path:
    artifact = root / "artifact"
    artifact.mkdir()
    (artifact / "conversion-manifest.json").write_text(json.dumps({"schema": 1}))
    (artifact / "layer-00001-gate_proj.safetensors").write_bytes(b"weights-a")
    target = root / "seed-shard.safetensors"
    target.write_bytes(b"seed-bytes")
    (artifact / "layer-00002-up_proj.safetensors").symlink_to(target)
    return artifact


def test_artifact_hash_stable_and_symlink_retarget_changes_identity(tmp_path: Path) -> None:
    artifact = _write_fake_artifact(tmp_path)
    first = hashing.artifact_dir_hash(artifact)
    assert first == hashing.artifact_dir_hash(artifact)

    link = artifact / "layer-00002-up_proj.safetensors"
    other_target = tmp_path / "other-shard.safetensors"
    other_target.write_bytes(b"seed-bytes")
    link.unlink()
    link.symlink_to(other_target)
    assert hashing.artifact_dir_hash(artifact) != first


def test_artifact_hash_does_not_read_symlink_target_bytes(tmp_path: Path) -> None:
    artifact = _write_fake_artifact(tmp_path)
    first = hashing.artifact_dir_hash(artifact)
    # Mutating the link target's bytes must not change identity: symlink
    # targets are protected seeds identified by their link string.
    (tmp_path / "seed-shard.safetensors").write_bytes(b"mutated!!!")
    assert hashing.artifact_dir_hash(artifact) == first


def test_artifact_hash_changes_with_manifest_and_listing(tmp_path: Path) -> None:
    artifact = _write_fake_artifact(tmp_path)
    first = hashing.artifact_dir_hash(artifact)
    (artifact / "conversion-manifest.json").write_text(json.dumps({"schema": 2}))
    second = hashing.artifact_dir_hash(artifact)
    assert second != first
    (artifact / "extra.safetensors").write_bytes(b"x")
    assert hashing.artifact_dir_hash(artifact) != second


def test_external_input_hash_kinds(tmp_path: Path) -> None:
    artifact = _write_fake_artifact(tmp_path)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"prompt_id": "report_math_001"}\n')
    teacher_cache = tmp_path / "teacher-cache"
    logits = teacher_cache / "teacher_logits"
    logits.mkdir(parents=True)
    (teacher_cache / "glm52-teacher-cache-fp32-manifest.json").write_bytes(
        b'{"schema_version":1}\n'
    )
    shard = logits / "report-route-00.safetensors"
    shard.write_bytes(b"same-size-shard")
    assert hashing.external_input_hash("artifact_dir", artifact)
    assert hashing.external_input_hash("teacher_cache", metadata) == hashing.file_hash(metadata)
    initial_cache_hash = hashing.external_input_hash(
        "teacher_cache_artifact", teacher_cache
    )
    shard.write_bytes(b"mutated-content")
    assert shard.stat().st_size == len(b"same-size-shard")
    assert (
        hashing.external_input_hash("teacher_cache_artifact", teacher_cache)
        != initial_cache_hash
    )
    with pytest.raises(ValueError):
        hashing.external_input_hash("mystery", metadata)
    with pytest.raises(FileNotFoundError):
        hashing.external_input_hash("artifact_dir", tmp_path / "missing")
