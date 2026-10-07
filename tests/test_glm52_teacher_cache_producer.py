from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import os
import struct
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

QUALITY_ROOT = Path(__file__).resolve().parents[1] / "src/mlx_vq/quality"
CACHE_PATH = QUALITY_ROOT / "glm52_teacher_cache.py"
CACHE_SPEC = importlib.util.spec_from_file_location(
    "mlx_vq.quality.glm52_teacher_cache", CACHE_PATH
)
assert CACHE_SPEC is not None and CACHE_SPEC.loader is not None
cache_module = importlib.util.module_from_spec(CACHE_SPEC)
sys.modules[CACHE_SPEC.name] = cache_module
CACHE_SPEC.loader.exec_module(cache_module)

PRODUCER_PATH = QUALITY_ROOT / "glm52_teacher_cache_producer.py"
PRODUCER_SPEC = importlib.util.spec_from_file_location(
    "glm52_teacher_cache_producer_under_test", PRODUCER_PATH
)
assert PRODUCER_SPEC is not None and PRODUCER_SPEC.loader is not None
producer_module = importlib.util.module_from_spec(PRODUCER_SPEC)
sys.modules[PRODUCER_SPEC.name] = producer_module
PRODUCER_SPEC.loader.exec_module(producer_module)

GLM52TeacherCacheContract = cache_module.GLM52TeacherCacheContract
GLM52TeacherCachePrompt = cache_module.GLM52TeacherCachePrompt
publish_glm52_teacher_cache_manifest = cache_module.publish_glm52_teacher_cache_manifest
write_glm52_teacher_cache_shard = cache_module.write_glm52_teacher_cache_shard
HEAVY_JOB_LOCK_PATH = producer_module.HEAVY_JOB_LOCK_PATH
ProducerPlan = producer_module.ProducerPlan
SourceRunEvidence = producer_module.SourceRunEvidence
plan_glm52_teacher_cache = producer_module.plan_glm52_teacher_cache
produce_glm52_teacher_cache = producer_module.produce_glm52_teacher_cache


def _is_fake_sensitive_module(name: str) -> bool:
    return name == "mlx" or name.startswith(("mlx.", "mlx_lm", "mlx_vq."))


@pytest.fixture(autouse=True)
def _restore_fake_sensitive_modules() -> Any:
    """Do not let a fake-MLX producer test alter later test-module imports."""

    pristine = {
        name: module
        for name, module in sys.modules.items()
        if _is_fake_sensitive_module(name)
    }
    yield
    for name in tuple(sys.modules):
        if _is_fake_sensitive_module(name) and name not in pristine:
            module_path = getattr(sys.modules[name], "__file__", None)
            if name in {"mlx", "mlx.core"} and module_path is not None:
                continue
            sys.modules.pop(name, None)
    sys.modules.update(pristine)


def _contract() -> GLM52TeacherCacheContract:
    return GLM52TeacherCacheContract.for_testing(
        prompts=(
            GLM52TeacherCachePrompt(
                prompt_id="report-000",
                split="report",
                domain="reasoning",
                tuning_eligible=False,
                encoded_token_ids=(1, 2, 3),
            ),
            GLM52TeacherCachePrompt(
                prompt_id="selection-000",
                split="selection",
                domain="coding",
                tuning_eligible=True,
                encoded_token_ids=(2, 3, 4, 1),
            ),
        ),
        vocab_size=5,
    )


def _route_trace_contract() -> GLM52TeacherCacheContract:
    """Return a production-shaped synthetic contract with 744 valid rows."""

    prompts = tuple(
        GLM52TeacherCachePrompt(
            prompt_id=f"route-{index:03d}",
            split=("report" if index < 22 else "selection" if index < 44 else "holdout"),
            domain="reasoning",
            tuning_eligible=22 <= index < 44,
            encoded_token_ids=(1,) * (13 if index < 18 else 12),
        )
        for index in range(66)
    )
    contract = GLM52TeacherCacheContract.for_testing(prompts=prompts, vocab_size=5)
    return replace(
        contract,
        prompt_authority={
            **contract.prompt_authority,
            "path": "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json",
            "file_sha256": "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31",
        },
    )


def _logits(prompt: GLM52TeacherCachePrompt, vocab_size: int) -> np.ndarray:
    positions = prompt.token_count - 1
    return np.arange(positions * vocab_size, dtype=np.float32).reshape(
        positions, vocab_size
    ) + np.float32(0.25)


class FakeLockProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.heavy_held = False
        self.run_held = False
        self.heavy_path: Path | None = None
        self.run_path: Path | None = None

    @contextmanager
    def heavy_job_lock(self, path: Path):
        assert not self.heavy_held
        self.heavy_path = path
        self.events.append("heavy-enter")
        self.heavy_held = True
        try:
            yield
        finally:
            self.heavy_held = False
            self.events.append("heavy-exit")

    @contextmanager
    def run_lock(self, path: Path):
        assert self.heavy_held
        assert not self.run_held
        self.run_path = path
        self.events.append("run-enter")
        self.run_held = True
        try:
            yield
        finally:
            self.run_held = False
            self.events.append("run-exit")


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "cache_root": tmp_path / "cache",
        "ledger_path": tmp_path / "state" / "shard-ledger.jsonl",
        "checkpoint_dir": tmp_path / "state" / "checkpoints",
    }


def _clean_audit(contract: GLM52TeacherCacheContract) -> SimpleNamespace:
    return SimpleNamespace(
        audit_pass=True,
        manifest_sha256=contract.non_vq_package["manifest_sha256"],
        package_set_sha256=contract.non_vq_package["package_set_sha256"],
        retained_tensor_count=1_194,
        tensor_payload_bytes=37_121_488_608,
    )


def test_hf_snapshot_blob_symlinks_authenticate_source_config_and_index(
    tmp_path: Path,
) -> None:
    hf_model_root = tmp_path / "hf" / "models--example--glm52"
    snapshot = hf_model_root / "snapshots" / "pinned-revision"
    blobs = hf_model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    config_payload = b'{"model_type":"glm"}\n'
    index_payload = b'{"weight_map":{}}\n'
    config_blob = blobs / "config-sha"
    index_blob = blobs / "index-sha"
    config_blob.write_bytes(config_payload)
    index_blob.write_bytes(index_payload)
    (snapshot / "config.json").symlink_to("../../blobs/config-sha")
    (snapshot / "model.safetensors.index.json").symlink_to(
        "../../blobs/index-sha"
    )

    assert producer_module._authenticated_hf_snapshot_file(
        snapshot,
        "config.json",
        expected_sha256=hashlib.sha256(config_payload).hexdigest(),
        label="source config",
    ) == config_blob.resolve()
    assert producer_module._authenticated_hf_snapshot_file(
        snapshot,
        "model.safetensors.index.json",
        expected_sha256=hashlib.sha256(index_payload).hexdigest(),
        label="source index",
    ) == index_blob.resolve()


def _write_source_snapshot(
    tmp_path: Path,
    *,
    shard_names: tuple[str, ...] = (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ),
    shard_payloads: tuple[bytes, ...] = (b"routed-one", b"routed-two", b"non-routed"),
    weight_map: dict[str, str] | None = None,
) -> tuple[Path, bytes, dict[str, Path]]:
    hf_model_root = tmp_path / "hf" / "models--example--glm52"
    snapshot = hf_model_root / "snapshots" / "pinned-revision"
    blobs = hf_model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    paths: dict[str, Path] = {}
    for shard_name, payload in zip(shard_names, shard_payloads, strict=True):
        blob = blobs / hashlib.sha256(payload).hexdigest()
        blob.write_bytes(payload)
        (snapshot / shard_name).symlink_to(Path("../../blobs") / blob.name)
        paths[shard_name] = blob
    if weight_map is None:
        weight_map = {
            "model.layers.3.mlp.experts.0.gate_proj.weight": shard_names[0],
            "model.layers.77.mlp.experts.0.gate_proj.weight": shard_names[1],
            "model.layers.78.mlp.experts.0.gate_proj.weight": shard_names[2],
        }
    index_bytes = json.dumps({"weight_map": weight_map}, sort_keys=True).encode("utf-8")
    return snapshot, index_bytes, paths


def _raw_safetensors(tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> bytes:
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
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def _load_headless_modelopt_expert_resolver(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the real resolver factory without initializing a Metal device."""

    source_path = Path(__file__).resolve().parents[1] / "src/mlx_vq/models/glm52_source_teacher.py"
    module_name = "glm52_teacher_cache_headless_source_teacher_test"
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_nn = ModuleType("mlx.nn")
    mlx.core = mlx_core  # type: ignore[attr-defined]
    mlx.nn = mlx_nn  # type: ignore[attr-defined]
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm_models = ModuleType("mlx_lm.models")
    mlx_lm_base = ModuleType("mlx_lm.models.base")
    mlx_lm_base.create_causal_mask = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    mlx_utils = ModuleType("mlx.utils")
    mlx_utils.tree_flatten = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
    adapter = ModuleType("mlx_vq.models.glm52_vq_adapter")
    adapter.GLM52VQModel = object  # type: ignore[attr-defined]
    adapter.Glm52VQMoE = object  # type: ignore[attr-defined]
    io_load = ModuleType("mlx_vq.io.load")
    io_load.inspect_safetensors = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    io_source = ModuleType("mlx_vq.io.source_safetensors")
    io_source.read_safetensors_file_header = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    io_source.read_safetensors_tensor_bytes = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    io_source.read_safetensors_tensor_mlx = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    for name, module in {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx.nn": mlx_nn,
        "mlx_lm": mlx_lm,
        "mlx_lm.models": mlx_lm_models,
        "mlx_lm.models.base": mlx_lm_base,
        "mlx.utils": mlx_utils,
        "mlx_vq.models.glm52_vq_adapter": adapter,
        "mlx_vq.io.load": io_load,
        "mlx_vq.io.source_safetensors": io_source,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    source_teacher = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, source_teacher)
    spec.loader.exec_module(source_teacher)
    return source_teacher.make_modelopt_nvfp4_expert_weight_resolver


def test_layer_resolver_reads_authenticated_blob_path_after_anchor_reaches_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver_factory = _load_headless_modelopt_expert_resolver(monkeypatch)
    shard_name = "model-00001-of-00001.safetensors"
    weight_map: dict[str, str] = {}
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "padding": ("U8", (1, 1_048_576), b"\0" * 1_048_576),
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.3.mlp.experts.0.{projection}"
        weight_map[f"{base}.weight"] = shard_name
        weight_map[f"{base}.weight_scale"] = shard_name
        weight_map[f"{base}.weight_scale_2"] = shard_name
        tensors[f"{base}.weight"] = ("U8", (1, 8), b"\x11" * 8)
        tensors[f"{base}.weight_scale"] = ("F8_E4M3", (1, 1), b"\x38")
        tensors[f"{base}.weight_scale_2"] = ("F32", (), struct.pack("<f", 1.0))
    snapshot, index_bytes, _ = _write_source_snapshot(
        tmp_path,
        shard_names=(shard_name,),
        shard_payloads=(_raw_safetensors(tensors),),
        weight_map=weight_map,
    )
    authenticated = producer_module._open_authenticated_source_blob_inventory(
        snapshot,
        index_bytes,
    )
    try:
        anchored = authenticated.blobs[0]
        os.lseek(anchored.descriptor, 0, os.SEEK_END)
        layer_resolver = producer_module._make_authenticated_layer_resolver(
            authenticated,
            resolver_factory=resolver_factory,
        )
        with authenticated.modelopt_reader_guard():
            decoded = layer_resolver(3)(0)["gate_proj"]()
    finally:
        authenticated.close()

    np.testing.assert_array_equal(decoded, np.full((1, 16), 0.5, dtype=np.float32))


def test_authenticated_reader_guard_is_run_scoped_not_projection_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver_factory = _load_headless_modelopt_expert_resolver(monkeypatch)
    shard_name = "model-00001-of-00001.safetensors"
    weight_map: dict[str, str] = {}
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.3.mlp.experts.0.{projection}"
        weight_map[f"{base}.weight"] = shard_name
        weight_map[f"{base}.weight_scale"] = shard_name
        weight_map[f"{base}.weight_scale_2"] = shard_name
        tensors[f"{base}.weight"] = ("U8", (1, 8), b"\x11" * 8)
        tensors[f"{base}.weight_scale"] = ("F8_E4M3", (1, 1), b"\x38")
        tensors[f"{base}.weight_scale_2"] = ("F32", (), struct.pack("<f", 1.0))
    snapshot, index_bytes, _ = _write_source_snapshot(
        tmp_path,
        shard_names=(shard_name,),
        shard_payloads=(_raw_safetensors(tensors),),
        weight_map=weight_map,
    )
    authenticated = producer_module._open_authenticated_source_blob_inventory(
        snapshot,
        index_bytes,
    )
    reader_guard_entries = 0
    original_reader_guard = authenticated.modelopt_reader_guard

    @contextmanager
    def counting_reader_guard():
        nonlocal reader_guard_entries
        reader_guard_entries += 1
        with original_reader_guard():
            yield

    authenticated.modelopt_reader_guard = counting_reader_guard
    try:
        layer_resolver = producer_module._make_authenticated_layer_resolver(
            authenticated,
            resolver_factory=resolver_factory,
        )
        projections = layer_resolver(3)(0)
        with authenticated.modelopt_reader_guard():
            decoded = tuple(projections[name]() for name in sorted(projections))
    finally:
        authenticated.close()

    assert all(value.dtype == np.float32 for value in decoded)
    assert reader_guard_entries == 1


def test_verify_glm52_source_blob_inventory_authenticates_indexed_hf_blobs(
    tmp_path: Path,
) -> None:
    snapshot, index_bytes, paths = _write_source_snapshot(
        tmp_path,
        shard_names=(
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ),
        shard_payloads=(b"one", b"two", b"three"),
    )

    report = producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)

    expected_routed_inventory = {
        name: paths[name].name
        for name in (
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
        )
    }
    expected_full_inventory = {
        name: paths[name].name
        for name in (
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        )
    }
    assert report["blob_inventory"] == expected_routed_inventory
    assert report["source_blob_inventory_sha256"] == producer_module.canonical_sha256(
        expected_full_inventory
    )
    assert report["routed_source_blob_inventory_sha256"] == producer_module.canonical_sha256(
        expected_routed_inventory
    )
    assert report["shard_count"] == 2
    assert report["inventory_hash_workers"] == 4
    assert list(report["shard_verify_seconds"]) == sorted(report["shard_verify_seconds"])


def test_full_source_blob_inventory_matches_non_vq_audit_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, index_bytes, _ = _write_source_snapshot(tmp_path)
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx.core = mlx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    non_vq_module = __import__("mlx_vq.convert.glm52_non_vq", fromlist=["*"])
    source_weight_map = producer_module._source_weight_map_from_index_bytes(index_bytes)
    expected_inventory_sha256 = producer_module.canonical_sha256(
        {
            shard_name: (snapshot / shard_name).resolve().name
            for shard_name in sorted(set(source_weight_map.values()))
        }
    )
    monkeypatch.setattr(
        non_vq_module,
        "NON_VQ_PINNED_BLOB_INVENTORY_SHA256",
        expected_inventory_sha256,
    )

    assert non_vq_module._validate_pinned_blob_inventory(
        snapshot,
        index=non_vq_module.SafetensorsIndex(
            metadata={},
            weight_map=source_weight_map,
        ),
        revision="pinned-revision",
    ) == expected_inventory_sha256
    assert producer_module.canonical_sha256(
        producer_module._source_blob_inventory(
            snapshot,
            snapshot.parent.parent / "blobs",
            source_weight_map,
        )
    ) == expected_inventory_sha256


def test_verify_glm52_source_blob_inventory_selects_only_routed_payload_shards(
    tmp_path: Path,
) -> None:
    snapshot, index_bytes, paths = _write_source_snapshot(tmp_path)

    report = producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)

    assert report["blob_inventory"] == {
        "model-00001-of-00003.safetensors": paths[
            "model-00001-of-00003.safetensors"
        ].name,
        "model-00002-of-00003.safetensors": paths[
            "model-00002-of-00003.safetensors"
        ].name,
    }
    assert report["shard_count"] == 2


def test_authenticated_streaming_descriptors_request_nocache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(producer_module.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(
        producer_module.fcntl,
        "fcntl",
        lambda descriptor, command, value: calls.append((descriptor, command, value)),
    )

    descriptor = producer_module._open_regular_no_follow(source)
    try:
        assert calls == [(descriptor, 48, 1)]
    finally:
        os.close(descriptor)


def test_source_blob_report_records_nocache_mitigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, index_bytes, _ = _write_source_snapshot(tmp_path)
    monkeypatch.setattr(producer_module.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(producer_module.fcntl, "fcntl", lambda *_args: 0)

    report = producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)

    assert report["nocache_applied"] is True


def test_verify_glm52_source_blob_inventory_rejects_blob_bytes_not_matching_name(
    tmp_path: Path,
) -> None:
    snapshot, index_bytes, paths = _write_source_snapshot(tmp_path)
    paths["model-00001-of-00003.safetensors"].write_bytes(b"replaced-content")

    with pytest.raises(ValueError, match="content SHA-256"):
        producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)

    assert not any(
        thread.name.startswith("glm52-inventory-hash")
        for thread in threading.enumerate()
    )


def test_verify_glm52_source_blob_inventory_requires_supplied_evidence_match(
    tmp_path: Path,
) -> None:
    snapshot, index_bytes, _ = _write_source_snapshot(tmp_path)

    with pytest.raises(ValueError, match="supplied non-VQ evidence"):
        producer_module.verify_glm52_source_blob_inventory(
            snapshot,
            index_bytes,
            expected_inventory_sha256="0" * 64,
        )


def test_verify_glm52_source_blob_inventory_rejects_symlink_escaping_blobs_root(
    tmp_path: Path,
) -> None:
    snapshot, index_bytes, _ = _write_source_snapshot(tmp_path)
    escaped = tmp_path / "escaped.safetensors"
    escaped.write_bytes(b"outside")
    (snapshot / "model-00001-of-00003.safetensors").unlink()
    (snapshot / "model-00001-of-00003.safetensors").symlink_to(escaped)

    with pytest.raises(ValueError, match="blob store"):
        producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)


def test_verify_glm52_source_blob_inventory_rejects_out_of_snapshot_shard_name(
    tmp_path: Path,
) -> None:
    snapshot, _, _ = _write_source_snapshot(tmp_path)
    index_bytes = json.dumps(
        {
            "weight_map": {
                "model.layers.3.mlp.experts.0.gate_proj.weight": "../escaped.safetensors"
            }
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="shard name"):
        producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)


def test_producer_rejects_authenticated_source_identity_change_after_forward(
    tmp_path: Path,
) -> None:
    contract = _contract()
    verified = SimpleNamespace(
        verify_after_forward=lambda: (_ for _ in ()).throw(
            ValueError("source shard identity changed after forward")
        ),
        close=lambda: None,
    )

    def run_source(
        source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        del source, checkpoint_dir, checkpoint_identities
        phase_boundary()
        for index in pending_prompt_indices:
            sink(index, _logits(prompts[index], contract.vocab_size))
        return SourceRunEvidence(False, "d" * 64)

    source = SimpleNamespace(source_blob_inventory=verified)
    with pytest.raises(ValueError, match="identity changed after forward"):
        produce_glm52_teacher_cache(
            contract=contract,
            **_paths(tmp_path),
            non_vq_package_auditor=lambda: _clean_audit(contract),
            source_loader=lambda: source,
            source_runner=run_source,
            lock_provider=FakeLockProvider([]),
            memory_counter_reader=lambda: {"pageouts": 0, "swapouts": 0},
            wired_policy_probe=lambda: True,
        )


def test_cache_side_symlink_still_fails(tmp_path: Path) -> None:
    cache_target = tmp_path / "real-cache"
    cache_target.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.symlink_to(cache_target, target_is_directory=True)

    with pytest.raises(ValueError, match="cache_root must not be a symlink"):
        plan_glm52_teacher_cache(
            contract=_contract(),
            cache_root=cache_root,
            ledger_path=tmp_path / "state" / "shard-ledger.jsonl",
            checkpoint_dir=tmp_path / "state" / "checkpoints",
        )


def test_same_size_non_vq_payload_mutation_aborts_before_source_load(
    tmp_path: Path,
) -> None:
    contract = _contract()
    events: list[str] = []
    locks = FakeLockProvider(events)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"authentic-payload")
    expected_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    shard.write_bytes(b"Authentic-payload")
    assert shard.stat().st_size == len(b"authentic-payload")

    def audit_non_vq_package() -> Any:
        assert locks.heavy_held and locks.run_held
        events.append("audit")
        if hashlib.sha256(shard.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("tensor payload SHA-256 byte identity failed")
        raise AssertionError("mutated shard unexpectedly passed its byte audit")

    def forbidden_source_load() -> Any:
        events.append("bind")
        raise AssertionError("source construction/binding must follow the package audit")

    with pytest.raises(ValueError, match="payload SHA-256 byte identity failed"):
        produce_glm52_teacher_cache(
            contract=contract,
            **_paths(tmp_path),
            non_vq_package_auditor=audit_non_vq_package,
            source_loader=forbidden_source_load,
            source_runner=lambda *_args, **_kwargs: None,
            lock_provider=locks,
            memory_counter_reader=lambda: {"pageouts": 0, "swapouts": 0},
            wired_policy_probe=lambda: True,
        )

    assert events == [
        "heavy-enter",
        "run-enter",
        "audit",
        "run-exit",
        "heavy-exit",
    ]


def test_production_audit_wrapper_defaults_to_real_byte_auditor() -> None:
    signature = inspect.signature(
        producer_module.audit_glm52_non_vq_package_for_teacher_cache
    )
    assert signature.parameters["audit_callable"].default.__name__ == (
        "audit_glm52_non_vq_package"
    )


def test_source_runner_bounds_and_restores_mlx_memory_and_cache_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_calls: list[int] = []
    memory_calls: list[int] = []

    def set_cache_limit(value: int) -> int:
        cache_calls.append(value)
        return 9_000_000_000

    def set_memory_limit(value: int) -> int:
        memory_calls.append(value)
        return 90_000_000_000

    mlx_core = ModuleType("mlx.core")
    mlx_core.set_cache_limit = set_cache_limit  # type: ignore[attr-defined]
    mlx_core.set_memory_limit = set_memory_limit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setattr(
        producer_module,
        "_run_glm52_source_teacher_to_sink",
        lambda *_args, **_kwargs: SourceRunEvidence(False, "6" * 64),
    )

    evidence = producer_module.run_glm52_source_teacher_to_sink(
        object(),
        (),
        pending_prompt_indices=(),
        checkpoint_dir=Path("unused"),
        checkpoint_identities=object(),
        phase_boundary=lambda: None,
        sink=lambda *_args: None,
    )

    assert cache_calls == [4 * 1024**3, 9_000_000_000]
    assert memory_calls == [48 * 1024**3, 90_000_000_000]
    assert evidence.mlx_cache_limit_bytes == 4 * 1024**3


def test_locks_are_acquired_in_order_before_load_and_held_through_manifest(
    tmp_path: Path,
) -> None:
    contract = _contract()
    paths = _paths(tmp_path)
    events: list[str] = []
    locks = FakeLockProvider(events)

    def load_source() -> object:
        assert locks.heavy_held and locks.run_held
        events.append("load")
        return SimpleNamespace(
            source_blob_inventory=SimpleNamespace(
                report=lambda: {
                    "nocache_applied": True,
                    "inventory_hash_workers": 4,
                },
                verify_after_forward=lambda: None,
                close=lambda: None,
            )
        )

    def run_source(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        assert locks.heavy_held and locks.run_held
        assert checkpoint_dir == paths["checkpoint_dir"]
        assert checkpoint_identities is None
        events.append("run")
        phase_boundary()
        for index in pending_prompt_indices:
            sink(index, _logits(prompts[index], contract.vocab_size))
        return SourceRunEvidence(
            resumed_from_checkpoint=False,
            output_identity_sha256="7" * 64,
            mlx_cache_limit_bytes=4 * 1024**3,
            decode_workers=2,
            decode_queue_depth=2,
            observed_peak_decoded_experts=3,
        )

    def shard_writer(*args: Any, **kwargs: Any):
        assert locks.heavy_held and locks.run_held
        events.append(f"shard:{kwargs['prompt'].prompt_id}")
        return write_glm52_teacher_cache_shard(*args, **kwargs)

    def manifest_publisher(*args: Any, **kwargs: Any):
        assert locks.heavy_held and locks.run_held
        events.append("manifest")
        return publish_glm52_teacher_cache_manifest(*args, **kwargs)

    snapshots = iter(
        (
            {"pageouts": 10, "swapouts": 20},
            {"pageouts": 10, "swapouts": 20},
            {"pageouts": 10, "swapouts": 20},
            {"pageouts": 10, "swapouts": 20},
            {"pageouts": 10, "swapouts": 20},
        )
    )
    result = produce_glm52_teacher_cache(
        contract=contract,
        **paths,
        non_vq_package_auditor=lambda: events.append("audit")
        or _clean_audit(contract),
        source_loader=load_source,
        source_runner=run_source,
        lock_provider=locks,
        memory_counter_reader=lambda: next(snapshots),
        wired_policy_probe=lambda: True,
        shard_writer=shard_writer,
        manifest_publisher=manifest_publisher,
        created_at_factory=lambda: "2026-07-10T12:00:00Z",
    )

    assert result.manifest.release_eligible is True
    assert "nocache_applied=true" in result.manifest.producer_phases[1].contribution_range
    assert (
        "inventory_hash_workers=4"
        in result.manifest.producer_phases[1].contribution_range
    )
    assert (
        "mlx_cache_limit_bytes=4294967296"
        in result.manifest.producer_phases[2].contribution_range
    )
    assert "decode_workers=2" in result.manifest.producer_phases[2].contribution_range
    assert "decode_queue_depth=2" in result.manifest.producer_phases[2].contribution_range
    assert (
        "observed_peak_decoded_experts=3"
        in result.manifest.producer_phases[2].contribution_range
    )
    assert result.manifest_path.is_file()
    assert locks.heavy_path == HEAVY_JOB_LOCK_PATH
    assert locks.run_path == Path(f"{paths['ledger_path']}.producer.lock")
    assert events == [
        "heavy-enter",
        "run-enter",
        "audit",
        "load",
        "run",
        "shard:report-000",
        "shard:selection-000",
        "manifest",
        "run-exit",
        "heavy-exit",
    ]


def test_each_prompt_is_written_and_released_before_next_projection(
    tmp_path: Path,
) -> None:
    contract = _contract()
    events: list[str] = []
    locks = FakeLockProvider(events)
    live_arrays: list[Any] = []

    def run_source(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        del checkpoint_dir, checkpoint_identities
        phase_boundary()
        for index in pending_prompt_indices:
            events.append(f"project:{index}")
            array = _logits(prompts[index], contract.vocab_size)
            live_arrays.append(array)
            sink(index, array)
            del array
            gc.collect()
        return SourceRunEvidence(False, "8" * 64)

    def shard_writer(*args: Any, **kwargs: Any):
        events.append(f"write:{kwargs['prompt'].prompt_id}")
        return write_glm52_teacher_cache_shard(*args, **kwargs)

    def release_buffers() -> None:
        events.append("release")
        live_arrays.clear()
        gc.collect()

    snapshots = iter(
        (
            {"pageouts": 1, "swapouts": 2},
            {"pageouts": 1, "swapouts": 2},
            {"pageouts": 1, "swapouts": 2},
            {"pageouts": 1, "swapouts": 2},
            {"pageouts": 1, "swapouts": 2},
        )
    )
    produce_glm52_teacher_cache(
        contract=contract,
        **_paths(tmp_path),
        non_vq_package_auditor=lambda: _clean_audit(contract),
        source_loader=object,
        source_runner=run_source,
        lock_provider=locks,
        memory_counter_reader=lambda: next(snapshots),
        wired_policy_probe=lambda: True,
        shard_writer=shard_writer,
        buffer_releaser=release_buffers,
    )

    assert events[2:-3] == [
        "project:0",
        "write:report-000",
        "release",
        "project:1",
        "write:selection-000",
        "release",
    ]
    assert live_arrays == []


def test_resume_validates_ledgered_shard_and_skips_its_projection(
    tmp_path: Path,
) -> None:
    contract = _contract()
    paths = _paths(tmp_path)
    first_events: list[str] = []
    first_locks = FakeLockProvider(first_events)

    def interrupted_runner(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        del checkpoint_dir, checkpoint_identities
        phase_boundary()
        sink(pending_prompt_indices[0], _logits(prompts[0], contract.vocab_size))
        raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        produce_glm52_teacher_cache(
            contract=contract,
            **paths,
            non_vq_package_auditor=lambda: _clean_audit(contract),
            source_loader=object,
            source_runner=interrupted_runner,
            lock_provider=first_locks,
            memory_counter_reader=lambda: {"pageouts": 0, "swapouts": 0},
            wired_policy_probe=lambda: True,
        )

    pending_seen: list[tuple[int, ...]] = []

    def resumed_runner(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        del checkpoint_dir, checkpoint_identities
        phase_boundary()
        pending_seen.append(pending_prompt_indices)
        for index in pending_prompt_indices:
            sink(index, _logits(prompts[index], contract.vocab_size))
        return SourceRunEvidence(True, "9" * 64)

    second_locks = FakeLockProvider([])
    result = produce_glm52_teacher_cache(
        contract=contract,
        **paths,
        non_vq_package_auditor=lambda: _clean_audit(contract),
        source_loader=object,
        source_runner=resumed_runner,
        lock_provider=second_locks,
        memory_counter_reader=lambda: {"pageouts": 0, "swapouts": 0},
        wired_policy_probe=lambda: True,
    )

    assert pending_seen == [(1,)]
    assert [shard.prompt_id for shard in result.manifest.shards] == [
        "report-000",
        "selection-000",
    ]
    assert result.resumed_prompt_ids == ("report-000",)
    assert result.manifest.all_producer_memory_counters_known is False
    assert result.manifest.release_eligible is False


def test_unknown_or_boolean_memory_counters_stay_null_and_block_release(
    tmp_path: Path,
) -> None:
    contract = _contract()

    def runner(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        del checkpoint_dir, checkpoint_identities
        phase_boundary()
        for index in pending_prompt_indices:
            sink(index, _logits(prompts[index], contract.vocab_size))
        return SourceRunEvidence(False, "a" * 64)

    snapshots = iter(
        (
            {"pageouts": False, "swapouts": 10},
            {"pageouts": 10, "swapouts": 10},
            {"pageouts": 10, "swapouts": 10},
            {"pageouts": 10, "swapouts": 10},
            {"pageouts": 10, "swapouts": 10},
        )
    )
    result = produce_glm52_teacher_cache(
        contract=contract,
        **_paths(tmp_path),
        non_vq_package_auditor=lambda: _clean_audit(contract),
        source_loader=object,
        source_runner=runner,
        lock_provider=FakeLockProvider([]),
        memory_counter_reader=lambda: next(snapshots),
        wired_policy_probe=lambda: True,
    )

    assert [phase.phase_id for phase in result.manifest.producer_phases] == [
        "non-vq-byte-audit",
        "source-load-and-bind",
        "layer-forward-and-checkpoints",
        "prompt-projection-and-shard-publication",
    ]
    phase = result.manifest.producer_phases[0]
    assert phase.pageouts_delta is None
    assert phase.swapouts_delta == 0
    assert phase.memory_counters_known is False
    assert phase.memory_clean is False
    assert result.manifest.all_producer_memory_counters_known is False
    assert result.manifest.release_eligible is False


def test_complete_valid_cache_returns_without_loading_or_republishing(
    tmp_path: Path,
) -> None:
    contract = _contract()
    paths = _paths(tmp_path)

    def runner(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
    ) -> SourceRunEvidence:
        del checkpoint_dir, checkpoint_identities
        phase_boundary()
        for index in pending_prompt_indices:
            sink(index, _logits(prompts[index], contract.vocab_size))
        return SourceRunEvidence(False, "b" * 64)

    snapshots = lambda: {"pageouts": 0, "swapouts": 0}
    first = produce_glm52_teacher_cache(
        contract=contract,
        **paths,
        non_vq_package_auditor=lambda: _clean_audit(contract),
        source_loader=object,
        source_runner=runner,
        lock_provider=FakeLockProvider([]),
        memory_counter_reader=snapshots,
        wired_policy_probe=lambda: True,
    )

    def forbidden() -> Any:
        raise AssertionError("a complete audited cache must not reload the source")

    second = produce_glm52_teacher_cache(
        contract=contract,
        **paths,
        non_vq_package_auditor=lambda: _clean_audit(contract),
        source_loader=forbidden,
        source_runner=forbidden,
        lock_provider=FakeLockProvider([]),
        memory_counter_reader=snapshots,
        wired_policy_probe=lambda: True,
    )

    assert second.completed is True
    assert second.manifest_path == first.manifest_path
    assert second.resumed_prompt_ids == ("report-000", "selection-000")
    assert second.produced_prompt_ids == ()


def test_route_trace_capture_publishes_frozen_release_sidecar(
    tmp_path: Path,
) -> None:
    contract = _route_trace_contract()
    paths = _paths(tmp_path)
    trace_root = tmp_path / "route-traces"
    captured_logits: list[np.ndarray] = []
    base_ids = np.tile(np.arange(8, dtype=np.int32), (744, 1))
    base_scores = np.full((744, 8), np.float32(2.5 / 8), dtype=np.float32)

    def runner(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
        capture_route_trace: bool,
    ) -> SourceRunEvidence:
        del checkpoint_dir, checkpoint_identities
        assert capture_route_trace is True
        phase_boundary()
        for index in pending_prompt_indices:
            logits = _logits(prompts[index], contract.vocab_size)
            captured_logits.append(logits.copy())
            sink(index, logits)
        return SourceRunEvidence(
            False,
            "c" * 64,
            route_traces=tuple(
                SimpleNamespace(
                    layer_index=layer_index,
                    expert_ids=base_ids.copy(),
                    scores=base_scores.copy(),
                    valid_assignment_count=744 * 8,
                    padded_assignment_count=0,
                )
                for layer_index in range(3, 78)
            ),
        )

    result = produce_glm52_teacher_cache(
        contract=contract,
        **paths,
        route_trace_root=trace_root,
        non_vq_package_auditor=lambda: _clean_audit(contract),
        source_loader=object,
        source_runner=runner,
        lock_provider=FakeLockProvider([]),
        memory_counter_reader=lambda: {"pageouts": 0, "swapouts": 0},
        wired_policy_probe=lambda: True,
    )

    diagnostics_path = QUALITY_ROOT / "glm52_route_diagnostics.py"
    diagnostics_spec = importlib.util.spec_from_file_location(
        "glm52_route_diagnostics_producer_test", diagnostics_path
    )
    assert diagnostics_spec is not None and diagnostics_spec.loader is not None
    diagnostics = importlib.util.module_from_spec(diagnostics_spec)
    sys.modules[diagnostics_spec.name] = diagnostics
    diagnostics_spec.loader.exec_module(diagnostics)
    audited = diagnostics.audit_route_trace_artifact(trace_root, expected_side="source")

    assert [phase.phase_id for phase in result.manifest.producer_phases] == [
        "non-vq-byte-audit",
        "source-load-and-bind",
        "layer-forward-and-checkpoints",
        "prompt-projection-and-shard-publication",
        "route-trace-publication",
    ]
    assert audited.manifest["n_valid"] == 744
    assert audited.manifest["layer_count"] == 75
    assert [layer.layer_index for layer in audited.layers] == list(range(3, 78))
    assert all(layer.expert_ids.dtype == np.dtype("int32") for layer in audited.layers)
    assert all(layer.scores.dtype == np.dtype("float32") for layer in audited.layers)
    assert all(layer.valid_assignment_count == 744 * 8 for layer in audited.layers)
    assert all(layer.padded_assignment_count == 0 for layer in audited.layers)
    assert audited.manifest["authority"] == {
        "teacher_cache_path": str(result.manifest_path),
        "teacher_cache_content_sha256": result.manifest.cache_content_sha256,
        "candidate_composite_path": "artifacts/glm52-reap-504b-v2",
        "candidate_composite_identity_sha256": "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067",
        "prompt_pack_path": "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json",
        "prompt_pack_sha256": "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31",
        "model_id": contract.source["model_id"],
        "model_revision": contract.source["revision"],
        "producer_implementation_id": "mlx_vq.quality.glm52_teacher_cache_producer.route-trace.v1",
        "capture_output_path": str(trace_root),
        "capture_output_sha256": audited.manifest["authority"]["capture_output_sha256"],
        "evidence_class": "release",
    }
    assert all(np.array_equal(value, _logits(prompt, contract.vocab_size)) for value, prompt in zip(captured_logits, contract.prompts, strict=True))


def test_route_trace_writer_failure_rolls_back_cache_state_and_allows_retry(
    tmp_path: Path,
) -> None:
    contract = _route_trace_contract()
    paths = _paths(tmp_path)
    trace_root = tmp_path / "route-traces"
    base_ids = np.tile(np.arange(8, dtype=np.int32), (744, 1))
    base_scores = np.full((744, 8), np.float32(2.5 / 8), dtype=np.float32)

    def runner(
        _source: object,
        prompts: tuple[GLM52TeacherCachePrompt, ...],
        *,
        pending_prompt_indices: tuple[int, ...],
        checkpoint_dir: Path,
        checkpoint_identities: object | None,
        phase_boundary: Any,
        sink: Any,
        capture_route_trace: bool,
    ) -> SourceRunEvidence:
        del checkpoint_identities
        assert capture_route_trace is True
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "run-owned.partial").write_text("partial")
        phase_boundary()
        for index in pending_prompt_indices:
            sink(index, _logits(prompts[index], contract.vocab_size))
        return SourceRunEvidence(
            False,
            "c" * 64,
            route_traces=tuple(
                SimpleNamespace(
                    layer_index=layer_index,
                    expert_ids=base_ids.copy(),
                    scores=base_scores.copy(),
                    valid_assignment_count=744 * 8,
                    padded_assignment_count=0,
                )
                for layer_index in range(3, 78)
            ),
        )

    attempts = 0

    def writer(root: Path, _layers: object, **_kwargs: object) -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        root.mkdir(parents=True, exist_ok=True)
        (root / "partial.bin").write_bytes(b"partial")
        if attempts == 1:
            raise RuntimeError("synthetic trace writer failure")
        (root / "partial.bin").unlink()
        (root / "complete.json").write_text("{}\n")
        return {"published": True}

    kwargs = {
        "contract": contract,
        **paths,
        "route_trace_root": trace_root,
        "non_vq_package_auditor": lambda: _clean_audit(contract),
        "source_loader": object,
        "source_runner": runner,
        "lock_provider": FakeLockProvider([]),
        "memory_counter_reader": lambda: {"pageouts": 0, "swapouts": 0},
        "wired_policy_probe": lambda: True,
        "route_trace_writer": writer,
    }

    with pytest.raises(RuntimeError, match="trace writer failure"):
        produce_glm52_teacher_cache(**kwargs)

    assert not paths["cache_root"].exists()
    assert not paths["ledger_path"].exists()
    assert not paths["checkpoint_dir"].exists()
    assert not trace_root.exists()

    result = produce_glm52_teacher_cache(**kwargs)

    assert result.completed is True
    assert result.manifest_path is not None and result.manifest_path.is_file()
    assert attempts == 2


def test_route_trace_root_inside_cache_root_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(ValueError, match="route trace root must remain outside the final cache root"):
        plan_glm52_teacher_cache(
            contract=_contract(),
            **paths,
            route_trace_root=paths["cache_root"] / "route-traces",
        )


def test_route_trace_capture_rejects_preexisting_transaction_paths_without_mutation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    trace_root = tmp_path / "route-traces"
    trace_root.mkdir()
    sentinel = trace_root / "protected.txt"
    sentinel.write_text("preserve")

    with pytest.raises(ValueError, match="fresh exact-run transaction paths"):
        produce_glm52_teacher_cache(
            contract=_route_trace_contract(),
            **paths,
            route_trace_root=trace_root,
            non_vq_package_auditor=lambda: pytest.fail("audit ran after preflight failure"),
            source_loader=lambda: pytest.fail("source loaded after preflight failure"),
            source_runner=lambda *_args, **_kwargs: pytest.fail("source ran"),
            lock_provider=FakeLockProvider([]),
        )

    assert sentinel.read_text() == "preserve"
    assert not paths["cache_root"].exists()
    assert not paths["ledger_path"].exists()


def test_absent_route_trace_root_preserves_plan_and_phase_identity(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    implicit = plan_glm52_teacher_cache(contract=_contract(), **paths)
    explicit = plan_glm52_teacher_cache(
        contract=_contract(),
        **paths,
        route_trace_root=None,
    )

    assert implicit.to_dict() == explicit.to_dict()
    assert producer_module.PRODUCER_PHASE_IDS == (
        "non-vq-byte-audit",
        "source-load-and-bind",
        "layer-forward-and-checkpoints",
        "prompt-projection-and-shard-publication",
    )


def test_dry_run_plan_is_read_only_and_does_not_acquire_locks(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    plan = plan_glm52_teacher_cache(
        contract=_contract(),
        **paths,
        limit_prompts=1,
    )

    assert isinstance(plan, ProducerPlan)
    assert plan.dry_run is True
    assert plan.prompt_ids == ("report-000",)
    assert plan.manifest_will_publish is False
    assert not paths["cache_root"].exists()
    assert not paths["ledger_path"].exists()
    assert not paths["checkpoint_dir"].exists()


def test_cli_dry_run_prints_plan_without_calling_producer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "benchmarks/produce_glm52_teacher_cache.py"
    spec = importlib.util.spec_from_file_location(
        "produce_glm52_teacher_cache_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    paths = _paths(tmp_path)

    def forbidden_producer(**_kwargs: Any) -> Any:
        raise AssertionError("dry-run must not enter the producer or acquire locks")

    exit_code = script.run_cli(
        [
            "--snapshot-dir",
            str(tmp_path / "snapshot"),
            "--prompt-pack-json",
            str(tmp_path / "prompts.json"),
            "--non-vq-package-dir",
            str(tmp_path / "non-vq"),
            "--artifact-identities-json",
            str(tmp_path / "identities.json"),
            "--profile-path",
            str(tmp_path / "profile.yaml"),
            "--cache-root",
            str(paths["cache_root"]),
            "--route-trace-root",
            str(tmp_path / "route-traces"),
            "--ledger-path",
            str(paths["ledger_path"]),
            "--checkpoint-dir",
            str(paths["checkpoint_dir"]),
            "--limit-prompts",
            "1",
            "--dry-run",
        ],
        contract_loader=lambda _prompt_pack, _identities: _contract(),
        producer=forbidden_producer,
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["locks_acquired"] is False
    assert output["prompt_ids"] == ["report-000"]
    assert output["manifest_will_publish"] is False
    assert not paths["cache_root"].exists()


def test_inventory_hash_pool_is_drained_before_first_mlx_metal_use(
    tmp_path: Path,
) -> None:
    snapshot, index_bytes, _ = _write_source_snapshot(tmp_path)

    producer_module.verify_glm52_source_blob_inventory(snapshot, index_bytes)

    assert not any(
        thread.name.startswith("glm52-inventory-hash")
        for thread in threading.enumerate()
    )
    mx = pytest.importorskip("mlx.core")
    try:
        value = mx.array([1])
        mx.eval(value)
    except RuntimeError as error:
        if "No Metal device available" in str(error):
            pytest.skip("Metal is unavailable in this test environment")
        raise


def test_fake_mlx_tests_do_not_replace_canonical_source_teacher_modules() -> None:
    mx = importlib.import_module("mlx.core")
    try:
        source_teacher = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    except RuntimeError as error:
        if "No Metal device available" in str(error):
            pytest.skip("Metal is unavailable in this test environment")
        raise

    assert hasattr(mx, "array")
    assert source_teacher.mx is mx
