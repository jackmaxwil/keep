from __future__ import annotations

import hashlib
import importlib
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
import yaml
from mlx.utils import tree_flatten

from mlx_vq.convert.stream_convert import SafetensorsIndex, load_safetensors_index
from mlx_vq.io import source_safetensors
from mlx_vq.io.source_safetensors import read_safetensors_tensor_bytes
from mlx_vq.models.glm52_vq_adapter import (
    GLM52VQModel,
    GLM52VQModelArgs,
    bind_glm52_non_vq_weights,
)
from mlx_vq.models.profiles import ModelProfile, get_profile, profile_to_dict


MODEL_ID = "fixture/glm52-reap"
REVISION = "fixture-revision"
CONFIG_SHA256 = "c" * 64
MANIFEST_NAME = "non-vq-manifest.json"
INDEX_NAME = "model.safetensors.index.json"
RETAINED_NAMES = (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.1.mlp.gate.weight",
    "model.layers.1.mlp.shared_experts.gate_proj.weight",
)


@dataclass(frozen=True)
class _SourceFixture:
    root: Path
    index_path: Path
    index: SafetensorsIndex
    index_sha256: str
    profile: ModelProfile


def _api():
    return importlib.import_module("mlx_vq.convert.glm52_non_vq")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_profile() -> ModelProfile:
    return replace(
        get_profile("glm52-reap-504b-v2"),
        name="glm52-non-vq-fixture",
        hf_model_id=MODEL_ID,
        revision=REVISION,
        num_layers=3,
        num_sparse_layers=2,
        first_sparse_layer=1,
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=2,
        experts_per_tok=1,
        vocab_size=32,
        shared_experts=1,
        group_size_policy={"gate": 8, "up": 8, "down": 8},
        recovery_layer=2,
        recovery_projections=("gate", "up", "down"),
        hard_layers=(),
        imatrix_layers=(1, 2),
    )


def _bf16_bytes(words: list[int]) -> bytes:
    return np.asarray(words, dtype="<u2").tobytes()


def _f32_bytes(words: list[int]) -> bytes:
    return np.asarray(words, dtype="<u4").tobytes()


def _write_raw_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _build_source(tmp_path: Path) -> _SourceFixture:
    root = tmp_path / "source"
    root.mkdir()
    shard_a = "source-a.safetensors"
    shard_b = "source-b.safetensors"
    tensors_a = {
        "model.layers.3.input_layernorm.weight": (
            "BF16",
            (2,),
            _bf16_bytes([0x3F80, 0x4000]),
        ),
        "model.layers.1.mlp.experts.0.gate_proj.weight": (
            "U8",
            (2, 2),
            bytes([1, 2, 3, 4]),
        ),
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale": (
            "F32",
            (1,),
            _f32_bytes([0x3F800000]),
        ),
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale_2": (
            "F32",
            (),
            _f32_bytes([0x3F000000]),
        ),
        "model.layers.1.mlp.experts.0.gate_proj.input_scale": (
            "F32",
            (),
            _f32_bytes([0x3E800000]),
        ),
        "model.layers.1.mlp.gate.weight": (
            "F32",
            (2, 2),
            _f32_bytes([0x80000000, 0x7FC12345, 0x3F800000, 0xBF800000]),
        ),
        "model.embed_tokens.weight": (
            "BF16",
            (4, 2),
            _bf16_bytes(
                [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7FC1, 0x4000, 0x4040, 0x4080]
            ),
        ),
    }
    tensors_b = {
        "model.layers.4.self_attn.q_proj.weight": (
            "BF16",
            (2, 2),
            _bf16_bytes([1, 2, 3, 4]),
        ),
        "model.layers.2.mlp.switch_mlp.gate_proj.codes": (
            "U8",
            (2, 2),
            bytes([5, 6, 7, 8]),
        ),
        "model.layers.1.mlp.experts.0.unknown_companion": (
            "F32",
            (1,),
            _f32_bytes([0x3F800000]),
        ),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": (
            "BF16",
            (2, 2),
            _bf16_bytes([0x3F80, 0x4000, 0x4040, 0x4080]),
        ),
        "model.layers.0.self_attn.q_proj.weight": (
            "BF16",
            (2, 2),
            _bf16_bytes([0x8000, 0x7FC1, 0x3F80, 0xBF80]),
        ),
        "lm_head.weight": (
            "F32",
            (2, 2),
            _f32_bytes([0x00000000, 0x80000000, 0x7FCABCDE, 0x3F000000]),
        ),
    }
    _write_raw_safetensors(root / shard_a, tensors_a)
    _write_raw_safetensors(root / shard_b, tensors_b)
    weight_map = {
        **{name: shard_a for name in tensors_a},
        **{name: shard_b for name in tensors_b},
    }
    index_path = root / INDEX_NAME
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, sort_keys=True) + "\n"
    )
    return _SourceFixture(
        root=root,
        index_path=index_path,
        index=load_safetensors_index(index_path),
        index_sha256=_sha256(index_path),
        profile=_fixture_profile(),
    )


def _pack(
    source: _SourceFixture,
    output_dir: Path,
    *,
    max_shard_payload_bytes: int = 24,
    resume: bool = False,
):
    return _api().pack_glm52_non_vq_safetensors(
        source.root,
        source.index,
        output_dir,
        profile=source.profile,
        model_id=MODEL_ID,
        revision=REVISION,
        config_sha256=CONFIG_SHA256,
        index_sha256=source.index_sha256,
        max_shard_payload_bytes=max_shard_payload_bytes,
        resume=resume,
        enforce_pinned_source=False,
    )


def _audit(source: _SourceFixture, output_dir: Path):
    return _api().audit_glm52_non_vq_package(
        output_dir,
        profile=source.profile,
        expected_model_id=MODEL_ID,
        expected_revision=REVISION,
        expected_config_sha256=CONFIG_SHA256,
        expected_index_sha256=source.index_sha256,
        source_dir=source.root,
        source_index=source.index,
        enforce_pinned_source=False,
    )


def _package_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_non_vq_pack_rejects_unpinned_fixture_by_default(tmp_path: Path) -> None:
    source = _build_source(tmp_path)

    with pytest.raises(ValueError, match="pinned|canonical|fixture"):
        _api().pack_glm52_non_vq_safetensors(
            source.root,
            source.index,
            tmp_path / "default-production-package",
            profile=source.profile,
            model_id=MODEL_ID,
            revision=REVISION,
            config_sha256=CONFIG_SHA256,
            index_sha256=source.index_sha256,
            max_shard_payload_bytes=24,
        )


def test_non_vq_pack_allows_explicit_unpinned_fixture_mode(tmp_path: Path) -> None:
    source = _build_source(tmp_path)

    result = _pack(source, tmp_path / "explicit-fixture-package")

    payload = result.to_dict()
    assert payload["pack_status"] == "glm52_non_vq_fixture_ready"
    assert payload["production_ready"] is False
    assert payload["source_authority"] == "unverified_test_fixture"


def _interrupt_after_first_published_shard(
    source: _SourceFixture,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, bytes, Any]:
    api = _api()
    write_shard = api._write_shard
    published: list[Path] = []

    def interrupting_write(path, *args, **kwargs):
        if published:
            raise RuntimeError("fixture interruption after first published shard")
        result = write_shard(path, *args, **kwargs)
        published.append(Path(path))
        return result

    monkeypatch.setattr(api, "_write_shard", interrupting_write)
    with pytest.raises(
        RuntimeError,
        match="fixture interruption after first published shard",
    ):
        _pack(source, output_dir)
    monkeypatch.setattr(api, "_write_shard", write_shard)

    shard_paths = sorted(output_dir.glob("model-*.safetensors"))
    assert shard_paths == published
    assert len(shard_paths) == 1
    assert source_safetensors.read_safetensors_file_header(shard_paths[0]).tensors
    assert not tuple(output_dir.rglob("*.partial-*"))
    return shard_paths[0].name, shard_paths[0].read_bytes(), write_shard


def test_non_vq_pack_preserves_exact_bf16_f32_bytes_names_and_lineage(
    tmp_path: Path,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"

    result = _pack(source, output)
    payload = result.to_dict()
    output_index = load_safetensors_index(output / INDEX_NAME)

    assert tuple(sorted(output_index.weight_map)) == RETAINED_NAMES
    for name in RETAINED_NAMES:
        source_header, source_raw = read_safetensors_tensor_bytes(
            source.root / source.index.weight_map[name], name
        )
        output_header, output_raw = read_safetensors_tensor_bytes(
            output / output_index.weight_map[name], name
        )
        assert output_header.dtype == source_header.dtype
        assert output_header.shape == source_header.shape
        assert output_raw == source_raw

    assert not any(".mlp.experts." in name for name in output_index.weight_map)
    assert not any(".mlp.switch_mlp." in name for name in output_index.weight_map)
    assert not any("model.layers.3." in name for name in output_index.weight_map)
    assert not any("model.layers.4." in name for name in output_index.weight_map)
    assert payload["pack_status"] == "glm52_non_vq_fixture_ready"
    assert payload["production_ready"] is False
    assert payload["source_authority"] == "unverified_test_fixture"
    assert payload["copy_mode"] == "raw_safetensors_byte_ranges_v1"
    assert payload["retained_tensor_count"] == 5
    assert payload["parameter_count"] == 24
    assert payload["tensor_payload_bytes"] == 64
    assert payload["model_id"] == MODEL_ID
    assert payload["source_revision"] == REVISION
    assert payload["config_sha256"] == CONFIG_SHA256
    assert payload["index_sha256"] == source.index_sha256
    assert _audit(source, output).audit_pass is True


def test_non_vq_pack_uses_deterministic_lexical_sharding_index_and_manifest(
    tmp_path: Path,
) -> None:
    source = _build_source(tmp_path)
    output_a = tmp_path / "package-a"
    output_b = tmp_path / "package-b"

    _pack(source, output_a)
    _pack(source, output_b)

    expected_weight_map = {
        "lm_head.weight": "model-00001-of-00003.safetensors",
        "model.embed_tokens.weight": "model-00002-of-00003.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
        "model.layers.1.mlp.gate.weight": "model-00003-of-00003.safetensors",
        "model.layers.1.mlp.shared_experts.gate_proj.weight": (
            "model-00003-of-00003.safetensors"
        ),
    }
    index = json.loads((output_a / INDEX_NAME).read_text())
    assert index == {
        "metadata": {"total_size": 64},
        "weight_map": expected_weight_map,
    }
    assert _package_files(output_a) == _package_files(output_b)
    assert sorted(_package_files(output_a)) == sorted(
        [
            MANIFEST_NAME,
            INDEX_NAME,
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
    )


def test_non_vq_pack_resume_reuses_only_verified_byte_identical_shards(
    tmp_path: Path,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    first = _pack(source, output)
    before = _package_files(output)

    resumed = _pack(source, output, resume=True)

    assert resumed.to_dict()["written_shards"] == 0
    assert resumed.to_dict()["reused_shards"] == 3
    assert resumed.to_dict()["resume_verified"] is True
    assert _package_files(output) == before
    assert not tuple(output.rglob("*.partial-*"))
    assert first.to_dict()["package_set_sha256"] == resumed.to_dict()[
        "package_set_sha256"
    ]


def test_non_vq_pack_restores_build_executor_resume_manifest(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    manifest_path = output / MANIFEST_NAME
    previous = output / f"{MANIFEST_NAME}.resume-previous.json"
    manifest_path.replace(previous)

    resumed = _pack(source, output, resume=True)

    assert resumed.to_dict()["written_shards"] == 0
    assert resumed.to_dict()["reused_shards"] == 3
    assert resumed.to_dict()["resume_verified"] is True
    assert manifest_path.is_file()
    assert not previous.exists()
    assert _audit(source, output).audit_pass is True


def test_non_vq_pack_does_not_republish_corrupt_executor_resume_manifest(
    tmp_path: Path,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    manifest_path = output / MANIFEST_NAME
    previous = output / f"{MANIFEST_NAME}.resume-previous.json"
    manifest_path.replace(previous)
    shard = output / "model-00001-of-00003.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 1
    shard.write_bytes(raw)

    with pytest.raises(ValueError, match="source|payload|SHA-256|byte identity"):
        _pack(source, output, resume=True)

    assert not manifest_path.exists()
    assert previous.is_file()


def test_non_vq_pack_partial_resume_reuses_verified_published_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    reused_name, reused_bytes, write_shard = _interrupt_after_first_published_shard(
        source,
        output,
        monkeypatch,
    )

    def write_only_missing_shards(path, *args, **kwargs):
        if Path(path).name == reused_name:
            raise AssertionError("verified published shard must not be rewritten")
        return write_shard(path, *args, **kwargs)

    monkeypatch.setattr(_api(), "_write_shard", write_only_missing_shards)
    resumed = _pack(source, output, resume=True)
    payload = resumed.to_dict()

    assert payload["written_shards"] == 2
    assert payload["reused_shards"] == 1
    assert payload["resume_verified"] is True
    assert (output / reused_name).read_bytes() == reused_bytes
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    expected_files = {
        MANIFEST_NAME,
        INDEX_NAME,
        *(str(record["filename"]) for record in manifest["shards"]),
    }
    assert set(_package_files(output)) == expected_files
    assert not tuple(output.rglob("*.partial-*"))
    assert _audit(source, output).audit_pass is True


def test_non_vq_pack_partial_resume_rejects_plan_mismatch_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _interrupt_after_first_published_shard(source, output, monkeypatch)
    before = _package_files(output)

    with pytest.raises(
        ValueError,
        match="max_shard_payload_bytes|plan",
    ):
        _pack(
            source,
            output,
            max_shard_payload_bytes=32,
            resume=True,
        )

    assert _package_files(output) == before


def test_non_vq_pack_partial_resume_rejects_unrelated_partial_named_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _interrupt_after_first_published_shard(source, output, monkeypatch)
    unrelated = output / "notes.partial-backup"
    unrelated.write_text("belongs to the caller")

    with pytest.raises(ValueError, match="unexpected|unmanifested|tree"):
        _pack(source, output, resume=True)

    assert unrelated.read_text() == "belongs to the caller"


@pytest.mark.parametrize(
    ("field", "wrong_value", "error_pattern"),
    [
        ("model_id", "fixture/other-model", "model_id|model ID"),
        ("revision", "other-revision", "revision"),
        ("config_sha256", "d" * 64, "config_sha256|config SHA-256"),
        ("index_sha256", "e" * 64, "index_sha256|index SHA-256"),
    ],
)
def test_non_vq_pack_partial_resume_rejects_lineage_mismatch_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong_value: str,
    error_pattern: str,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _interrupt_after_first_published_shard(source, output, monkeypatch)
    before = _package_files(output)
    kwargs = {
        "profile": source.profile,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": source.index_sha256,
        "max_shard_payload_bytes": 24,
        "resume": True,
        "enforce_pinned_source": False,
    }
    kwargs[field] = wrong_value

    with pytest.raises(ValueError, match=error_pattern):
        _api().pack_glm52_non_vq_safetensors(
            source.root,
            source.index,
            output,
            **kwargs,
        )

    assert _package_files(output) == before


def test_non_vq_pack_resume_rejects_corrupted_existing_shard(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    shard = output / "model-00002-of-00003.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 1
    shard.write_bytes(raw)

    with pytest.raises(ValueError, match="corrupt|SHA-256|sha256|byte identity"):
        _pack(source, output, resume=True)


@pytest.mark.parametrize("intruder", ["extra", "nested", "symlink"])
def test_non_vq_package_audit_rejects_every_unmanifested_tree_entry(
    tmp_path: Path,
    intruder: str,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    if intruder == "extra":
        (output / "notes.txt").write_text("not part of the package")
    elif intruder == "nested":
        nested = output / "nested"
        nested.mkdir()
        (nested / "rogue.safetensors").write_bytes(b"rogue")
    else:
        (output / "alias.safetensors").symlink_to(
            "model-00001-of-00003.safetensors"
        )

    with pytest.raises(ValueError, match="unexpected|unmanifested|symlink|tree"):
        _audit(source, output)


@pytest.mark.parametrize("drift", ["shard", "index", "manifest"])
def test_non_vq_package_audit_recomputes_all_bytes_counts_and_hashes(
    tmp_path: Path,
    drift: str,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    if drift == "shard":
        shard = output / "model-00001-of-00003.safetensors"
        shard.write_bytes(shard.read_bytes() + b"x")
    elif drift == "index":
        index_path = output / INDEX_NAME
        index = json.loads(index_path.read_text())
        index["metadata"]["total_size"] += 1
        index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
    else:
        manifest_path = output / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["tensor_payload_bytes"] += 1
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="bytes|count|hash|SHA-256|sha256|extent"):
        _audit(source, output)


def test_non_vq_package_source_relative_audit_rejects_coordinated_payload_tamper(
    tmp_path: Path,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    api = _api()
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    tensor_record = manifest["tensors"][0]
    shard_record = next(
        record
        for record in manifest["shards"]
        if record["filename"] == tensor_record["output_shard"]
    )
    shard_path = output / tensor_record["output_shard"]
    shard_bytes = bytearray(shard_path.read_bytes())
    header_size = struct.unpack("<Q", shard_bytes[:8])[0]
    payload_start = 8 + header_size + tensor_record["output_offsets"][0]
    shard_bytes[payload_start] ^= 1
    shard_path.write_bytes(shard_bytes)

    _, tampered_payload = read_safetensors_tensor_bytes(
        shard_path,
        tensor_record["name"],
    )
    tensor_record["payload_sha256"] = hashlib.sha256(tampered_payload).hexdigest()
    shard_record["file_sha256"] = _sha256(shard_path)
    manifest["package_set_sha256"] = api._package_set_sha256(
        package_index_sha256=manifest["package_index_sha256"],
        shard_records=manifest["shards"],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="source|payload|byte identity|SHA-256|sha256"):
        _audit(source, output)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "source_inventory_sha256",
        "plan_sha256",
        "dtype_tensor_counts",
        "excluded_routed_tensor_count",
        "excluded_mtp_tensor_count",
        "excluded_runtime_vq_tensor_count",
        "max_shard_payload_bytes",
    ],
)
def test_non_vq_package_audit_rejects_manifest_summary_drift(
    tmp_path: Path,
    field: str,
) -> None:
    source = _build_source(tmp_path)
    output = tmp_path / "package"
    _pack(source, output)
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if field == "schema_version":
        manifest[field] = 2
    elif field in {"source_inventory_sha256", "plan_sha256"}:
        manifest[field] = "0" * 64
    elif field == "dtype_tensor_counts":
        manifest[field] = {"BF16": 999, "F32": 999}
    elif field == "max_shard_payload_bytes":
        manifest[field] = 1
    else:
        manifest[field] = int(manifest[field]) + 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        ValueError,
        match="schema|source inventory|plan|dtype|excluded|shard|summary|mismatch",
    ):
        _audit(source, output)


def test_non_vq_pack_does_not_use_tensor_materialization_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_source(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("non-VQ repacking must copy raw tensor byte ranges")

    monkeypatch.setattr(source_safetensors, "read_safetensors_tensor_mlx", forbidden)
    monkeypatch.setattr(
        source_safetensors,
        "read_indexed_safetensors_tensor_mlx",
        forbidden,
    )
    monkeypatch.setattr(mx, "load", forbidden)

    payload = _pack(source, tmp_path / "package").to_dict()

    assert payload["copy_mode"] == "raw_safetensors_byte_ranges_v1"
    assert payload["largest_copy_buffer_bytes"] <= 1024 * 1024


def test_non_vq_pack_cli_emits_audited_evidence(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_to_dict(source.profile), sort_keys=False))
    output = tmp_path / "package"
    evidence = tmp_path / "pack-evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_vq.convert.glm52_non_vq",
            "--source-dir",
            str(source.root),
            "--index-path",
            str(source.index_path),
            "--output-dir",
            str(output),
            "--profile-path",
            str(profile_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--config-sha256",
            CONFIG_SHA256,
            "--index-sha256",
            source.index_sha256,
            "--max-shard-payload-bytes",
            "24",
            "--allow-unpinned-fixture",
            "--output-json",
            str(evidence),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(evidence.read_text())
    assert payload["pack_status"] == "glm52_non_vq_fixture_ready"
    assert payload["production_ready"] is False
    assert payload["package_audit_pass"] is True
    assert (output / MANIFEST_NAME).is_file()


def _tiny_args() -> GLM52VQModelArgs:
    return GLM52VQModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        v_head_dim=4,
        qk_nope_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=["full", "full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse", "sparse"],
    )


def _kv_b_from_split_weights(embed_q: mx.array, unembed_out: mx.array) -> mx.array:
    nope = embed_q.swapaxes(-1, -2)
    combined = mx.concatenate([nope, unembed_out], axis=1)
    return combined.reshape(-1, combined.shape[-1])


def _write_bindable_source(root: Path, model: GLM52VQModel) -> Path:
    params = dict(tree_flatten(model.parameters()))
    arrays: dict[str, mx.array] = {}
    skip_keys: set[str] = set()
    for layer in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{layer}.self_attn"
        embed_q = f"{prefix}.embed_q.weight"
        unembed_out = f"{prefix}.unembed_out.weight"
        skip_keys.update((embed_q, unembed_out))
        arrays[f"{prefix}.kv_b_proj.weight"] = _kv_b_from_split_weights(
            params[embed_q], params[unembed_out]
        ).astype(mx.bfloat16)
    for name, value in params.items():
        if name in skip_keys or ".mlp.switch_mlp." in name:
            continue
        arrays[name] = (
            value.astype(mx.bfloat16)
            if value.dtype in (mx.float16, mx.float32)
            else value
        )
    arrays["model.layers.1.mlp.experts.0.gate_proj.weight"] = mx.zeros(
        (8, 16), dtype=mx.uint8
    )
    arrays["model.layers.3.input_layernorm.weight"] = mx.zeros(
        (16,), dtype=mx.bfloat16
    )
    shard_name = "source.safetensors"
    mx.save_safetensors(str(root / shard_name), arrays)
    index_path = root / INDEX_NAME
    index_path.write_text(
        json.dumps(
            {"metadata": {}, "weight_map": {name: shard_name for name in arrays}},
            sort_keys=True,
        )
        + "\n"
    )
    return index_path


def test_non_vq_package_is_consumable_by_existing_strict_bind_function(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_model = GLM52VQModel(_tiny_args())
    index_path = _write_bindable_source(source_root, source_model)
    source = _SourceFixture(
        root=source_root,
        index_path=index_path,
        index=load_safetensors_index(index_path),
        index_sha256=_sha256(index_path),
        profile=_fixture_profile(),
    )
    package = tmp_path / "package"
    _pack(source, package, max_shard_payload_bytes=10_000_000)
    target = GLM52VQModel(_tiny_args())

    report = bind_glm52_non_vq_weights(
        target,
        package,
        load_safetensors_index(package / INDEX_NAME),
        strict=True,
    )
    mx.eval(target.parameters())

    assert report.missing_model_parameters == ()
    assert report.skipped_routed_expert_tensors == ()
    assert report.skipped_mtp_tensors == ()
