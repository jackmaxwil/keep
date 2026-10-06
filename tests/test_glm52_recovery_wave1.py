from __future__ import annotations

import hashlib
import json
import importlib.util
import sys
import os
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

import mlx_vq.convert.glm52_recovery_materialize as recovery_materialize
from mlx_vq.codebook.e8 import (
    E8P_PACKED_ABS_SHA256,
    decode_weight_matrix,
    e8_1bit_packed,
    e8p_packed_abs_grid,
)
from mlx_vq.convert.glm52_recovery_materialize import (
    build_recovery_conversion_manifest,
    build_mixed_artifact_tree,
    materialize_recovery_group,
    materialize_groups_from_source,
    quantize_weight_importance_aware,
)


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PROFILE = "glm52-reap-504b-v2"
FULL_SOURCE_BLOB_INVENTORY_SHA256 = "7" * 64
ROUTED_SOURCE_BLOB_INVENTORY_SHA256 = "8" * 64


def _load_recovery_module():
    name = "mlx_vq.quality.glm52_recovery"
    path = Path(__file__).resolve().parents[1] / "src/mlx_vq/quality/glm52_recovery.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_wave1_cli_module():
    name = "glm52_recovery_wave1_cli_under_test"
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_glm52_recovery_wave1.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_recovery = _load_recovery_module()
validate_selection_split = _recovery.validate_selection_split


def _load_rtn_module():
    name = "glm52_recovery_test_rtn"
    path = Path(__file__).resolve().parents[1] / "src/mlx_vq/quant/rtn.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_rtn = _load_rtn_module()
quantize_weight_rtn = _rtn.quantize_weight_rtn
dequantize_weight_np = _rtn.dequantize_weight_np


def _weighted_error(source: np.ndarray, reconstructed: np.ndarray, importance: np.ndarray) -> float:
    return float(np.sum((source - reconstructed) ** 2 * importance[None, :]))


def _selection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(22):
        token_count = 12 if index < 21 else 25
        rows.append(
            {
                "prompt_id": f"selection-{index:02d}",
                "split": "selection",
                "tuning_eligible": True,
                "encoded_token_ids": list(range(token_count)),
            }
        )
    return rows


def test_real_capture_path_accumulates_bf16_activation_stats() -> None:
    mx = pytest.importorskip("mlx.core", reason="MLX is unavailable")
    capture_projection_stats = _recovery._capture_projection_stats
    accumulator = _recovery.RecoveryStatsAccumulator(selected_layers=(3,), num_experts=2)
    accumulator.begin_layer(layer=3, route_ids=np.array([[0, 1]], dtype=np.int32))
    try:
        activations = mx.array([[1.5, -2.0], [0.5, 3.0]]).astype(mx.bfloat16)
    except RuntimeError as error:
        if "No Metal device available" not in str(error):
            raise
        pytest.skip("MLX runtime is unavailable: no Metal device")

    capture_projection_stats(
        accumulator=accumulator,
        layer=3,
        projection="gate_proj",
        expert=0,
        values=activations,
        route_scores=np.array([0.25, 0.75], dtype=np.float32),
    )

    moment = accumulator.moments[(3, "gate_proj", 0)]
    np.testing.assert_allclose(moment.sum_x2, np.array([2.5, 13.0], dtype=np.float64))
    np.testing.assert_allclose(
        moment.score_weighted_sum_x2,
        np.array([0.75, 7.75], dtype=np.float64),
    )
    assert moment.sum_x2.dtype == np.float64
    assert moment.score_weighted_sum_x2.dtype == np.float64


def _artifact_group(path: Path, *, layer: int, projection: str, fill: int) -> None:
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    quantization = {
        "quant_method": "mlx_vq_e8",
        "version": 1,
        "default_code_bits": 8,
        "default_group_size": 8,
        "codebook": {"name": "quip_e8", "dtype": "uint32", "entries": 256},
        "policy": {"scale_estimator": "max_abs"},
    }
    save_file(
        {
            f"{prefix}.codes": np.full((2, 1, 1), fill, dtype=np.uint8),
            f"{prefix}.scales": np.ones((2, 1, 1), dtype=np.float16),
            "model.vq_codebook.e8": e8_1bit_packed(),
        },
        path,
        metadata={"quantization_config": json.dumps(quantization, sort_keys=True)},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_accepted_composite_audit(
    path: Path,
    *,
    config_sha256: str,
    index_sha256: str,
    seed_manifest_sha256: str = "0" * 64,
) -> str:
    path.write_text(
        json.dumps(
            {
                "audit_pass": True,
                "artifact_integrity_pass": True,
                "model_id": MODEL_ID,
                "source_revision": REVISION,
                "config_sha256": config_sha256,
                "index_sha256": index_sha256,
                "actual_whole_model_tensor_payload_bytes": 98_433_923_808,
                "actual_routed_payload_bytes": 61_312_204_800,
                "main_non_routed_tensor_payload_bytes": 37_121_488_608,
                "manifest_sha256": seed_manifest_sha256,
                "group_set_sha256": recovery_materialize.PINNED_ACCEPTED_ROUTED_GROUP_SET_SHA256,
                "non_vq_evidence_sha256": recovery_materialize.PINNED_ACCEPTED_NON_VQ_EVIDENCE_SHA256,
                "expected_group_keys": [
                    f"{layer}:{projection}"
                    for layer in range(3, 78)
                    for projection in ("gate_proj", "up_proj", "down_proj")
                ],
                "ready_group_keys": [
                    f"{layer}:{projection}"
                    for layer in range(3, 78)
                    for projection in ("gate_proj", "up_proj", "down_proj")
                ],
                "checks": recovery_materialize.PINNED_ACCEPTED_AUDIT_CHECKS,
                "byte_identity_verified": True,
                "full_routed_artifact_ready": True,
                "main_routed_tensor_count": 151_200,
                "main_non_routed_tensor_count": 1_194,
                "non_vq_package_audit": {
                    "audit_pass": True,
                    "checks": recovery_materialize.PINNED_ACCEPTED_NON_VQ_AUDIT_CHECKS,
                    "manifest_sha256": recovery_materialize.PINNED_ACCEPTED_NON_VQ_MANIFEST_SHA256,
                    "package_set_sha256": recovery_materialize.PINNED_ACCEPTED_NON_VQ_PACKAGE_SET_SHA256,
                    "retained_tensor_count": 1_194,
                    "shard_count": 9,
                    "parameter_count": 18_560_731_704,
                    "tensor_payload_bytes": 37_121_488_608,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return _sha256(path)


def test_lineage_compatible_rehashed_composite_audit_is_not_independent_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forged-composite-audit.json"
    digest = _write_accepted_composite_audit(
        path,
        config_sha256="a" * 64,
        index_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="repository-pinned accepted baseline"):
        recovery_materialize._authenticate_accepted_composite_audit(
            path,
            expected_sha256=digest,
            expected_source_lineage={
                "source_model_id": MODEL_ID,
                "source_revision": REVISION,
                "source_config_sha256": "a" * 64,
                "source_index_sha256": "b" * 64,
                "source_profile": PROFILE,
            },
        )


def test_standalone_materializer_takes_heavy_lock_before_any_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidate"
    events: list[Path] = []

    @contextmanager
    def reject_lock(path: str | Path):
        events.append(Path(path))
        raise RuntimeError("synthetic lock acquisition failure")
        yield

    monkeypatch.setattr(
        recovery_materialize,
        "_cooperating_file_lock",
        reject_lock,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="lock acquisition failure"):
        materialize_groups_from_source(
            source_dir=tmp_path / "missing-source",
            index_path=tmp_path / "missing-index.json",
            seed_artifact_dir=tmp_path / "missing-seed",
            stats_dir=tmp_path / "missing-stats",
            output_dir=output,
            groups=tuple((3, projection) for projection in recovery_materialize.GROUP_PROJECTIONS),
            expected_stats_manifest_sha256="a" * 64,
            expected_seed_manifest_sha256="b" * 64,
            expected_full_source_blob_inventory_sha256="7" * 64,
            expected_routed_source_blob_inventory_sha256="8" * 64,
            accepted_composite_audit_json=tmp_path / "missing-audit.json",
            expected_composite_audit_sha256="c" * 64,
            heavy_lock_path=tmp_path / ".keep-heavy-job.lock",
        )

    assert events == [tmp_path / ".keep-heavy-job.lock"]
    assert not output.exists()


def test_materializer_heavy_lock_rejects_concurrent_mutation_transactions(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".keep-heavy-job.lock"
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    contender_errors: list[BaseException] = []

    def holder() -> None:
        with recovery_materialize._cooperating_file_lock(lock_path):
            first_entered.set()
            assert release_first.wait(2)

    def contender() -> None:
        assert first_entered.wait(2)
        second_started.set()
        try:
            with recovery_materialize._cooperating_file_lock(lock_path):
                second_entered.set()
        except BaseException as error:
            contender_errors.append(error)

    first = threading.Thread(target=holder)
    second = threading.Thread(target=contender)
    first.start()
    second.start()
    assert second_started.wait(2)
    assert not second_entered.wait(0.1)
    release_first.set()
    first.join(2)
    second.join(2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not second_entered.is_set()
    assert len(contender_errors) == 1
    assert isinstance(
        contender_errors[0], recovery_materialize.RecoveryHeavyLockBusy
    )


def _stats_manifest(
    root: Path,
    *,
    layer: int = 3,
    projection: str = "gate_proj",
    experts: int = 2,
    config_sha256: str = "a" * 64,
    index_sha256: str = "b" * 64,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for expert in range(experts):
        filename = f"layer-{layer:05d}-{projection}-expert-{expert:03d}.npz"
        path = root / filename
        np.savez(
            path,
            sum_x2=np.ones(8, dtype=np.float32),
            mean_second_moment=np.ones(8, dtype=np.float32),
            routing_weighted_importance=np.ones(8, dtype=np.float32),
            router_score_weighted_importance=np.ones(8, dtype=np.float32),
        )
        entries.append(
            {
                "layer": layer,
                "projection": projection,
                "expert": expert,
                "input_dim": 8,
                "route_count": 1,
                "total_route_count": 2040,
                "route_frequency": 1 / 2040,
                "router_score_sum": 0.5,
                "prompt_ids": [f"selection-{index:02d}" for index in range(22)],
                "sample_count": 22,
                "prompt_count": 22,
                "position_count": 255,
                "path": filename,
                "sha256": _sha256(path),
            }
        )
    split_evidence = {
        "split": "selection",
        "prompt_count": 22,
        "position_count": 255,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    return {
        "schema_version": 1,
        "record_type": "glm52_recovery_stats_manifest",
        "status": "complete",
        "method": "selection_only_diagonal_hessian_second_moments_v1",
        "selected_layers": [layer],
        "projections": ["gate_proj", "up_proj", "down_proj"],
        "split_evidence": split_evidence,
        "source_authority": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "profile": PROFILE,
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "authenticated_source_teacher": True,
            "layer_major_streaming": True,
            "split_evidence": split_evidence,
        },
        "sample_count": 22,
        "prompt_count": 22,
        "position_count": 255,
        "route_count_by_layer": {str(layer): 2040},
        "entry_count": len(entries),
        "entries": entries,
    }


def _write_stats_manifest(root: Path, manifest: dict[str, object]) -> Path:
    path = root / "glm52-recovery-stats-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _write_seed_manifest(
    root: Path, *, config_sha256: str = "a" * 64, index_sha256: str = "b" * 64
) -> Path:
    records: list[dict[str, object]] = []
    for layer in range(3, 78):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            filename = f"layer-{layer:05d}-{projection}.safetensors"
            path = root / filename
            records.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "artifact_path": filename,
                    "artifact_bytes": path.stat().st_size if path.exists() else 1,
                    "artifact_sha256": _sha256(path) if path.exists() else "0" * 64,
                    "codes_dtype": "uint8",
                    "codes_name": f"model.layers.{layer}.mlp.switch_mlp.{projection}.codes",
                    "codes_shape": [2, 1, 1],
                    "cross_shard_bundle_count": 0,
                    "decoded_expert_bytes": 32,
                    "expert_count": 2,
                    "scales_dtype": "float16",
                    "scales_name": f"model.layers.{layer}.mlp.switch_mlp.{projection}.scales",
                    "scales_shape": [2, 1, 1],
                    "source_bundle_members": 2,
                    "source_bundles": 2,
                    "source_shards": ["model-00001.safetensors"],
                    "status": "ready",
                }
            )
    selected_group_keys = [
        f"{layer}:{projection}"
        for layer in range(3, 78)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    manifest = {
        "artifact_total_bytes": sum(int(record["artifact_bytes"]) for record in records),
        "code_bits": 8,
        "codebook_name": "quip_e8",
        "codebook_sha256": "1" * 64,
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "profile": PROFILE,
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": config_sha256,
        "dense_checkpoint_written": False,
        "full_group_coverage": True,
        "group_size": 512,
        "index_sha256": index_sha256,
        "materialization_blocked": False,
        "materialization_blockers": [],
        "materialization_scope": "full",
        "materialization_status": "glm52_modelopt_nvfp4_groups_ready",
        "planned_vq_groups": 225,
        "peak_decoded_expert_bytes": 32,
        "ready_vq_groups": 225,
        "scale_estimator": "max_abs",
        "selected_group_keys": selected_group_keys,
        "selected_vq_groups": 225,
        "skipped_vq_groups": 0,
        "source_bundle_members": 450,
        "source_bundles": 450,
        "source_decoder": "modelopt_nvfp4_v1",
        "source_weight_encoding": "modelopt_nvfp4",
        "working_set_policy": "one_decoded_expert_per_worker_v1",
        "groups": records,
    }
    path = root / "conversion-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


class _ReadLog(list[str]):
    authenticated_source_opens: int = 0
    verified_after_reads: int = 0
    closed_inventories: int = 0


def _install_materialization_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    full_inventory_sha256: str = FULL_SOURCE_BLOB_INVENTORY_SHA256,
    routed_inventory_sha256: str = ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
    verify_after_error: str | None = None,
) -> _ReadLog:
    reads = _ReadLog()

    def resolve(_weight_map: object, name: str) -> str:
        return name

    def read(_source_root: Path, name: str) -> np.ndarray:
        reads.append(name)
        expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
        sign = 1.0 if expert == 0 else -1.0
        return np.array([[sign * value for value in (0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1)]], dtype=np.float32)

    import mlx_vq.convert.nvfp4 as nvfp4

    monkeypatch.setattr(nvfp4, "resolve_modelopt_nvfp4_weight_bundle", resolve)
    monkeypatch.setattr(nvfp4, "read_modelopt_nvfp4_weight", read)
    monkeypatch.setitem(
        sys.modules,
        "mlx_vq.convert.stream_convert",
        SimpleNamespace(load_safetensors_index=lambda _path: SimpleNamespace(weight_map={})),
    )

    class _AuthenticatedInventory:
        blob_root = Path("/authenticated/blob-root")
        weight_map: dict[str, str] = {}

        def __init__(self) -> None:
            reads.authenticated_source_opens += 1

        def report(self) -> dict[str, object]:
            return {
                "source_blob_inventory_sha256": full_inventory_sha256,
                "routed_source_blob_inventory_sha256": routed_inventory_sha256,
                "shard_count": 1,
            }

        def modelopt_reader_guard(self):
            return nullcontext()

        def verify_after_forward(self) -> None:
            reads.verified_after_reads += 1
            if verify_after_error is not None:
                raise ValueError(verify_after_error)

        def close(self) -> None:
            reads.closed_inventories += 1

    monkeypatch.setattr(
        recovery_materialize,
        "_load_source_auth_module",
        lambda: SimpleNamespace(
            _open_authenticated_source_blob_inventory=lambda _source, _index: _AuthenticatedInventory()
        ),
        raising=False,
    )
    return reads


def _run_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume: bool,
    actual_full_inventory_sha256: str = FULL_SOURCE_BLOB_INVENTORY_SHA256,
    actual_routed_inventory_sha256: str = ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
    tamper_seed_group: bool = False,
) -> tuple[dict[str, object], list[str]]:
    seed_root = tmp_path / "seed"
    stats_root = tmp_path / "stats"
    output_root = tmp_path / "output"
    seed_root.mkdir(exist_ok=True)
    stats_root.mkdir(exist_ok=True)
    seed_path = seed_root / "layer-00003-gate_proj.safetensors"
    for projection in recovery_materialize.GROUP_PROJECTIONS:
        projection_path = seed_root / f"layer-00003-{projection}.safetensors"
        if not projection_path.exists():
            _artifact_group(projection_path, layer=3, projection=projection, fill=1)
    seed_manifest_path = seed_root / "conversion-manifest.json"
    source_root = tmp_path / "hf/models--0xSero--glm-5.2-reap-504B-v2/snapshots" / REVISION
    source_root.mkdir(parents=True, exist_ok=True)
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    if not config_path.exists():
        config_path.write_text('{"model_type":"glm_moe_dsa"}\n')
    if not index_path.exists():
        index_path.write_text('{"metadata":{},"weight_map":{}}\n')
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)
    if not seed_manifest_path.exists():
        _write_seed_manifest(
            seed_root,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
        )
    monkeypatch.setattr(recovery_materialize, "GLM52_REAP_CONFIG_SHA256", config_sha256, raising=False)
    monkeypatch.setattr(recovery_materialize, "GLM52_REAP_INDEX_SHA256", index_sha256, raising=False)
    stats_manifest_path = stats_root / "glm52-recovery-stats-manifest.json"
    if not stats_manifest_path.exists():
        combined_manifest: dict[str, object] | None = None
        combined_entries: list[dict[str, object]] = []
        for projection in recovery_materialize.GROUP_PROJECTIONS:
            projection_manifest = _stats_manifest(
                stats_root,
                projection=projection,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
            )
            combined_entries.extend(projection_manifest["entries"])
            combined_manifest = projection_manifest
        assert combined_manifest is not None
        combined_manifest["entries"] = combined_entries
        combined_manifest["entry_count"] = len(combined_entries)
        _write_stats_manifest(stats_root, combined_manifest)
    reads = _install_materialization_stubs(
        monkeypatch,
        full_inventory_sha256=actual_full_inventory_sha256,
        routed_inventory_sha256=actual_routed_inventory_sha256,
    )
    if tamper_seed_group:
        _artifact_group(seed_path, layer=3, projection="gate_proj", fill=2)
    composite_audit_path = tmp_path / "accepted-composite-audit.json"
    monkeypatch.setattr(
        recovery_materialize,
        "PINNED_ACCEPTED_SEED_MANIFEST_SHA256",
        _sha256(seed_manifest_path),
    )
    composite_audit_sha256 = _write_accepted_composite_audit(
        composite_audit_path,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        seed_manifest_sha256=_sha256(seed_manifest_path),
    )
    manifest = materialize_groups_from_source(
        source_dir=source_root,
        index_path=index_path,
        seed_artifact_dir=seed_root,
        stats_dir=stats_root,
        output_dir=output_root,
        groups=tuple((3, projection) for projection in recovery_materialize.GROUP_PROJECTIONS),
        group_size=8,
        resume=resume,
        expected_stats_manifest_sha256=_sha256(stats_manifest_path),
        expected_seed_manifest_sha256=_sha256(seed_manifest_path),
        expected_full_source_blob_inventory_sha256=FULL_SOURCE_BLOB_INVENTORY_SHA256,
        expected_routed_source_blob_inventory_sha256=ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
        accepted_composite_audit_json=composite_audit_path,
        expected_composite_audit_sha256=composite_audit_sha256,
        heavy_lock_path=tmp_path / "synthetic-heavy.lock",
    )
    return manifest, reads


def _prepare_attribution_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    verify_after_error: str | None = None,
) -> tuple[object, dict[str, object], _ReadLog]:
    module = _load_wave1_cli_module()
    seed_root = tmp_path / "seed"
    stats_root = tmp_path / "stats"
    seed_root.mkdir()
    stats_root.mkdir()
    entries: list[dict[str, object]] = []
    stats_manifest: dict[str, object] | None = None
    for projection in ("gate_proj", "up_proj", "down_proj"):
        _artifact_group(
            seed_root / f"layer-00003-{projection}.safetensors",
            layer=3,
            projection=projection,
            fill=1,
        )
        projection_manifest = _stats_manifest(stats_root, projection=projection)
        entries.extend(projection_manifest["entries"])
        stats_manifest = projection_manifest
    assert stats_manifest is not None
    stats_manifest["entries"] = entries
    stats_manifest["entry_count"] = len(entries)
    stats_manifest_path = _write_stats_manifest(stats_root, stats_manifest)

    source_root = (
        tmp_path / "hf/models--0xSero--glm-5.2-reap-504B-v2/snapshots" / REVISION
    )
    source_root.mkdir(parents=True)
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    config_path.write_text('{"model_type":"glm_moe_dsa"}\n')
    index_path.write_text('{"metadata":{},"weight_map":{}}\n')
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)
    stats_manifest["source_authority"]["config_sha256"] = config_sha256
    stats_manifest["source_authority"]["index_sha256"] = index_sha256
    stats_manifest_path = _write_stats_manifest(stats_root, stats_manifest)
    seed_manifest_path = _write_seed_manifest(
        seed_root,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
    )
    monkeypatch.setattr(recovery_materialize, "GLM52_REAP_CONFIG_SHA256", config_sha256)
    monkeypatch.setattr(recovery_materialize, "GLM52_REAP_INDEX_SHA256", index_sha256)
    reads = _install_materialization_stubs(
        monkeypatch,
        verify_after_error=verify_after_error,
    )
    return module, {
        "source_dir": source_root,
        "index_path": index_path,
        "seed_artifact_dir": seed_root,
        "stats_dir": stats_root,
        "output_json": tmp_path / "attribution.json",
        "expected_stats_manifest_sha256": _sha256(stats_manifest_path),
        "expected_seed_manifest_sha256": _sha256(seed_manifest_path),
        "expected_full_source_blob_inventory_sha256": FULL_SOURCE_BLOB_INVENTORY_SHA256,
        "expected_routed_source_blob_inventory_sha256": ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
    }, reads


def _recovery_policy() -> dict[str, str]:
    return {
        "recovery_lever": (
            "selection_diagonal_hessian_importance_weighted_reround_v1"
        ),
        "scale_estimator": "importance_weighted_least_squares",
        "rounding_objective": (
            "selection_diagonal_hessian_weighted_squared_error"
        ),
    }


def _prepare_recovered_attribution_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict[str, object], SimpleNamespace, list[str]]:
    module, seed_kwargs, _reads = _prepare_attribution_fixture(tmp_path, monkeypatch)
    seed_root = Path(seed_kwargs.pop("seed_artifact_dir"))
    recovery_root = tmp_path / "recovery"
    candidate_root = recovery_root / "artifact"
    candidate_root.mkdir(parents=True)
    manifest_path = recovery_root / "conversion-manifest.json"
    manifest_path.write_text('{"status":"complete"}\n')
    policy_path = tmp_path / "recovery-policy.json"
    policy_path.write_text(json.dumps(_recovery_policy(), sort_keys=True) + "\n")
    composite_path = tmp_path / "accepted-composite-audit.json"
    composite_path.write_text('{"audit_pass":true}\n')
    groups: list[SimpleNamespace] = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        filename = f"layer-00003-{projection}.safetensors"
        source = seed_root / filename
        (candidate_root / filename).symlink_to(source)
        groups.append(
            SimpleNamespace(
                group_key=f"3:{projection}",
                filename=filename,
                classification="replacement",
                artifact_sha256=_sha256(source),
                artifact_bytes=source.stat().st_size,
            )
        )
    groups.extend(
        SimpleNamespace(
            group_key=f"unused-{index}",
            filename=f"unused-{index}.safetensors",
            classification="inherited",
            artifact_sha256="0" * 64,
            artifact_bytes=1,
        )
        for index in range(222)
    )
    events: list[str] = []
    audit = SimpleNamespace(
        audit_pass=True,
        group_count=225,
        replacement_group_count=3,
        inherited_group_count=222,
        groups=tuple(groups),
        recovery_dir=str(recovery_root),
        candidate_identity_sha256="d" * 64,
        accepted_baseline_composite_identity_sha256="e" * 64,
        verify_current_identity=lambda: events.append("verify"),
    )
    monkeypatch.setattr(
        module,
        "_load_recovery_profile",
        lambda _path: SimpleNamespace(name="profile"),
    )
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=(
                lambda *_args, **_kwargs: audit
            )
        ),
    )
    return module, {
        **seed_kwargs,
        "recovery_conversion_dir": recovery_root,
        "profile_path": tmp_path / "profile.yaml",
        "expected_recovery_manifest_sha256": _sha256(manifest_path),
        "expected_recovery_lever": _recovery_policy()["recovery_lever"],
        "expected_recovery_policy_json": policy_path,
        "accepted_composite_audit_json": composite_path,
        "expected_composite_audit_sha256": _sha256(composite_path),
    }, audit, events


def test_materialization_rejects_validly_rehashed_unauthorized_stats_tree_before_entry_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    stats_root = tmp_path / "stats"
    manifest_path = stats_root / "glm52-recovery-stats-manifest.json"
    accepted_sha256 = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["entries"][0]
    stats_path = stats_root / entry["path"]
    np.savez(
        stats_path,
        sum_x2=np.full(8, 9.0, dtype=np.float32),
        mean_second_moment=np.full(8, 9.0, dtype=np.float32),
        routing_weighted_importance=np.full(8, 9.0, dtype=np.float32),
        router_score_weighted_importance=np.full(8, 9.0, dtype=np.float32),
    )
    entry["sha256"] = _sha256(stats_path)
    _write_stats_manifest(stats_root, manifest)
    monkeypatch.setattr(
        recovery_materialize,
        "_load_authenticated_stats",
        lambda **_kwargs: pytest.fail("stats entries loaded before external manifest authority check"),
    )

    seed_manifest = tmp_path / "seed/conversion-manifest.json"
    source_root = tmp_path / "hf/models--0xSero--glm-5.2-reap-504B-v2/snapshots" / REVISION
    with pytest.raises(ValueError, match="stats manifest SHA-256.*external accepted authority"):
        materialize_groups_from_source(
            source_dir=source_root,
            index_path=source_root / "model.safetensors.index.json",
            seed_artifact_dir=tmp_path / "seed",
            stats_dir=stats_root,
            output_dir=tmp_path / "unauthorized-output",
            groups=tuple((3, projection) for projection in recovery_materialize.GROUP_PROJECTIONS),
            group_size=8,
            expected_stats_manifest_sha256=accepted_sha256,
            expected_seed_manifest_sha256=_sha256(seed_manifest),
            expected_full_source_blob_inventory_sha256=FULL_SOURCE_BLOB_INVENTORY_SHA256,
            expected_routed_source_blob_inventory_sha256=ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
            accepted_composite_audit_json=tmp_path / "unused-composite.json",
            expected_composite_audit_sha256="f" * 64,
            heavy_lock_path=tmp_path / "unauthorized-heavy.lock",
        )


@pytest.mark.parametrize("command", ("rematerialize", "run"))
@pytest.mark.parametrize(
    "missing_option",
    (
        "--expected-stats-manifest-sha256",
        "--expected-attribution-sha256",
        "--expected-seed-manifest-sha256",
        "--expected-full-source-blob-inventory-sha256",
        "--expected-routed-source-blob-inventory-sha256",
        "--accepted-composite-audit-json",
        "--expected-composite-audit-sha256",
    ),
)
def test_wave1_materialization_commands_require_external_authority_inputs(
    command: str,
    missing_option: str,
) -> None:
    module = _load_wave1_cli_module()
    shared = [
        "--source-dir", "source",
        "--index-path", "index.json",
        "--seed-artifact-dir", "seed",
        "--expected-stats-manifest-sha256", "a" * 64,
        "--expected-attribution-sha256", "c" * 64,
        "--expected-seed-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", "7" * 64,
        "--expected-routed-source-blob-inventory-sha256", "8" * 64,
        "--accepted-composite-audit-json", "composite.json",
        "--expected-composite-audit-sha256", "d" * 64,
    ]
    if command == "rematerialize":
        argv = [command, *shared, "--stats-dir", "stats", "--attribution-json", "attr.json", "--output-dir", "out"]
    else:
        argv = [
            command,
            "--snapshot-dir", "snapshot",
            "--prompt-pack-json", "prompts.json",
            "--artifact-identities-json", "identities.json",
            "--non-vq-package-dir", "non-vq",
            "--profile-path", "profile.json",
            "--layers", "1",
            *shared,
            "--output-root", "out",
        ]
    option_index = argv.index(missing_option)
    del argv[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        module._build_parser().parse_args(argv)


def test_recovery_clis_remove_raw_accounting_knobs_and_require_pinned_composite() -> None:
    module = _load_wave1_cli_module()
    command_parsers = next(
        action for action in module._build_parser()._actions if getattr(action, "choices", None)
    ).choices
    for command in ("rematerialize", "reevaluate", "run"):
        options = {
            option
            for action in command_parsers[command]._actions
            for option in action.option_strings
        }
        assert "--non-routed-tensor-payload-bytes" not in options
        assert "--logical-payload-limit-bytes" not in options
        assert "--accepted-composite-audit-json" in options
        assert "--expected-composite-audit-sha256" in options

    materializer_options = {
        option
        for action in recovery_materialize._build_parser()._actions
        for option in action.option_strings
    }
    assert "--non-routed-tensor-payload-bytes" not in materializer_options
    assert "--logical-payload-limit-bytes" not in materializer_options
    assert "--accepted-composite-audit-json" in materializer_options
    assert "--expected-composite-audit-sha256" in materializer_options


def test_combined_run_requires_existing_pinned_attribution() -> None:
    module = _load_wave1_cli_module()
    run_parser = next(
        action for action in module._build_parser()._actions if getattr(action, "choices", None)
    ).choices["run"]
    required = {
        option
        for action in run_parser._actions
        if action.required
        for option in action.option_strings
    }
    assert "--attribution-json" in required
    assert "--expected-attribution-sha256" in required


@pytest.mark.parametrize(
    "missing_option",
    (
        "--expected-stats-manifest-sha256",
        "--expected-seed-manifest-sha256",
        "--expected-full-source-blob-inventory-sha256",
        "--expected-routed-source-blob-inventory-sha256",
    ),
)
def test_wave1_attribute_requires_complete_input_authority(missing_option: str) -> None:
    module = _load_wave1_cli_module()
    argv = [
        "attribute",
        "--source-dir", "source",
        "--index-path", "index.json",
        "--seed-artifact-dir", "seed",
        "--stats-dir", "stats",
        "--output-json", "attribution.json",
        "--expected-stats-manifest-sha256", "a" * 64,
        "--expected-seed-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", "7" * 64,
        "--expected-routed-source-blob-inventory-sha256", "8" * 64,
    ]
    option_index = argv.index(missing_option)
    del argv[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        module._build_parser().parse_args(argv)


def _reevaluation_kwargs(tmp_path: Path) -> dict[str, object]:
    policy_path = tmp_path / "recovery-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "recovery_lever": (
                    "selection_diagonal_hessian_importance_weighted_reround_v1"
                ),
                "scale_estimator": "importance_weighted_least_squares",
                "rounding_objective": (
                    "selection_diagonal_hessian_weighted_squared_error"
                ),
            }
        )
    )
    readiness_path = tmp_path / "tokenizer-readiness.json"
    readiness_path.write_text(json.dumps({"prompt": "authenticated prompt"}))
    return {
        "profile_path": tmp_path / "profile.yaml",
        "config_path": tmp_path / "config.json",
        "source_index_path": tmp_path / "model.safetensors.index.json",
        "tokenizer_dir": tmp_path / "tokenizer",
        "tokenizer_readiness_json": readiness_path,
        "family_policy_json": tmp_path / "family-policy.json",
        "prompt_pack_json": tmp_path / "prompt-pack.json",
        "teacher_cache_root": tmp_path / "teacher-cache",
        "non_vq_artifact_dir": tmp_path / "non-vq",
        "non_vq_evidence_json": tmp_path / "non-vq-evidence.json",
        "accepted_routed_artifact_dir": tmp_path / "accepted-routed",
        "accepted_composite_audit_json": tmp_path / "accepted-composite-audit.json",
        "accepted_materialization_runs_jsonl": tmp_path / "materialization-runs.jsonl",
        "accepted_full_bind_preflight_json": tmp_path / "full-bind-preflight.json",
        "recovery_conversion_dir": tmp_path / "recovery",
        "output_json": tmp_path / "reevaluation.json",
        "expected_seed_manifest_sha256": "a" * 64,
        "expected_stats_manifest_sha256": "b" * 64,
        "expected_full_source_blob_inventory_sha256": "7" * 64,
        "expected_routed_source_blob_inventory_sha256": "8" * 64,
        "expected_recovery_lever": (
            "selection_diagonal_hessian_importance_weighted_reround_v1"
        ),
        "expected_recovery_policy_json": policy_path,
        "expected_composite_audit_sha256": "f" * 64,
        "heavy_lock_path": tmp_path / "heavy.lock",
    }


@pytest.mark.parametrize(
    "missing_option",
    (
        "--expected-seed-manifest-sha256",
        "--expected-stats-manifest-sha256",
        "--expected-full-source-blob-inventory-sha256",
        "--expected-routed-source-blob-inventory-sha256",
        "--expected-recovery-lever",
        "--expected-recovery-policy-json",
        "--expected-composite-audit-sha256",
    ),
)
def test_reevaluate_cli_requires_complete_external_audit_authority(
    missing_option: str,
) -> None:
    module = _load_wave1_cli_module()
    argv = [
        "reevaluate",
        "--profile-path", "profile.yaml",
        "--config-path", "config.json",
        "--source-index-path", "index.json",
        "--tokenizer-dir", "tokenizer",
        "--tokenizer-readiness-json", "tokenizer-readiness.json",
        "--family-policy-json", "family-policy.json",
        "--prompt-pack-json", "prompt-pack.json",
        "--teacher-cache-root", "teacher-cache",
        "--non-vq-artifact-dir", "non-vq",
        "--non-vq-evidence-json", "non-vq-evidence.json",
        "--accepted-routed-artifact-dir", "accepted-routed",
        "--accepted-composite-audit-json", "accepted-composite-audit.json",
        "--accepted-materialization-runs-jsonl", "materialization-runs.jsonl",
        "--accepted-full-bind-preflight-json", "full-bind-preflight.json",
        "--recovery-conversion-dir", "recovery",
        "--output-json", "reevaluation.json",
        "--expected-seed-manifest-sha256", "a" * 64,
        "--expected-stats-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", "7" * 64,
        "--expected-routed-source-blob-inventory-sha256", "8" * 64,
        "--expected-recovery-lever",
        "selection_diagonal_hessian_importance_weighted_reround_v1",
        "--expected-recovery-policy-json", "recovery-policy.json",
        "--expected-composite-audit-sha256", "f" * 64,
    ]
    option_index = argv.index(missing_option)
    del argv[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        module._build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "audit_error",
    (
        "recovery manifest was retargeted",
        "artifact symlink was retargeted",
        "recovery group mutated during inspection",
        "external identity mismatch",
        "audit checks did not pass",
    ),
)
def test_reevaluate_rejects_audit_failure_before_model_validation_load_or_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit_error: str,
) -> None:
    module = _load_wave1_cli_module()
    kwargs = _reevaluation_kwargs(tmp_path)
    events: list[str] = []

    def reject_audit(*_args: object, **_kwargs: object) -> object:
        events.append("audit")
        raise ValueError(audit_error)

    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(audit_glm52_recovery_mixed_artifact=reject_audit),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_load_recovery_profile",
        lambda _path: SimpleNamespace(name="profile"),
        raising=False,
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: pytest.fail(f"runtime module imported before audit passed: {name}"),
    )

    with pytest.raises(ValueError, match=audit_error):
        module.reevaluate_recovery_candidate(**kwargs)

    assert events == ["audit"]


@pytest.mark.parametrize(
    ("audit_pass", "group_count", "error"),
    (
        (False, 225, "audit did not pass"),
        (True, 224, "complete 225-group tree"),
    ),
)
def test_reevaluate_requires_passing_full_tree_audit_before_runtime_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit_pass: bool,
    group_count: int,
    error: str,
) -> None:
    module = _load_wave1_cli_module()
    kwargs = _reevaluation_kwargs(tmp_path)
    audit = SimpleNamespace(
        audit_pass=audit_pass,
        group_count=group_count,
        complete_replacement_layer_ids=(77,),
        groups=(),
        recovery_dir=str(kwargs["recovery_conversion_dir"]),
    )
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=lambda *_args, **_kwargs: audit
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_recovery_profile",
        lambda _path: SimpleNamespace(name="profile"),
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: pytest.fail(f"runtime module imported before audit passed: {name}"),
    )

    with pytest.raises(ValueError, match=error):
        module.reevaluate_recovery_candidate(**kwargs)


def test_reevaluate_forwards_audit_authority_and_reports_only_audited_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    kwargs = _reevaluation_kwargs(tmp_path)
    events: list[str] = []
    audit_calls: list[dict[str, object]] = []
    candidate_root = Path(kwargs["recovery_conversion_dir"]) / "artifact"
    group_keys = ("77:gate_proj", "77:up_proj", "77:down_proj")
    audit = SimpleNamespace(
        audit_pass=True,
        recovery_dir=str(kwargs["recovery_conversion_dir"]),
        manifest_path=str(Path(kwargs["recovery_conversion_dir"]) / "conversion-manifest.json"),
        manifest_body_sha256="c" * 64,
        seed_manifest_sha256=kwargs["expected_seed_manifest_sha256"],
        accepted_composite_audit_sha256=kwargs["expected_composite_audit_sha256"],
        group_count=225,
        replacement_group_count=3,
        inherited_group_count=222,
        complete_replacement_layer_ids=(77,),
        routed_tensor_payload_bytes=200,
        non_routed_tensor_payload_bytes=123,
        logical_whole_model_tensor_payload_bytes=323,
        logical_payload_limit_bytes=456,
        incremental_disk_bytes=99,
        candidate_identity_sha256="d" * 64,
        groups=tuple(
            SimpleNamespace(group_key=key, classification="replacement")
            for key in group_keys
        ),
        verify_current_identity=lambda: events.append("recheck"),
    )

    def accept_audit(*args: object, **forwarded: object) -> object:
        events.append("audit")
        assert args == (kwargs["recovery_conversion_dir"],)
        audit_calls.append(forwarded)
        return audit

    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(audit_glm52_recovery_mixed_artifact=accept_audit),
        raising=False,
    )
    profile = SimpleNamespace(name="profile")
    monkeypatch.setattr(module, "_load_recovery_profile", lambda _path: profile, raising=False)

    metrics = {
        "mean_kld": 0.0,
        "mean_ppl_ratio": 1.0,
        "p999_kld": 0.0,
        "top1_agreement": 1.0,
    }
    contract = SimpleNamespace(
        source={"model_id": "model", "revision": "revision"},
        prompts=(),
        vocab_size=1,
    )
    candidate = SimpleNamespace(
        _contract_from_cache_manifest=lambda *_args, **_kwargs: contract,
        _strict_audit=lambda *_args, **_kwargs: None,
        GLM52CandidateWorkload=lambda **forwarded: SimpleNamespace(**forwarded),
        run_glm52_candidate_to_sink=lambda *_args, **_kwargs: None,
        _load_frozen_gate=lambda _path: {
            "required_splits": (),
            "required_domains": (),
            "thresholds": {
                "domain_top1_min": 0.0,
                "mean_kld_max": 0.0,
                "mean_ppl_ratio_max": 1.0,
                "p999_kld_max": 0.0,
                "top1_min": 1.0,
            },
        },
        _summarize_rows=lambda _rows: dict(metrics),
    )
    validated = SimpleNamespace(profile=profile)

    def validate(**_forwarded: object) -> object:
        events.append("validate")
        return validated

    def load(value: object) -> tuple[object, object]:
        events.append("load")
        assert value is validated
        return object(), object()

    candidate_paths = {
        f"layer-00077-{projection}.safetensors": candidate_root
        / f"layer-00077-{projection}.safetensors"
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    monkeypatch.setattr(
        module.recovery_api,
        "authenticated_recovery_candidate_group_paths",
        lambda value: events.append("resolve") or candidate_paths
        if value is audit
        else pytest.fail("unexpected recovery audit"),
    )

    def bind(
        _model: object,
        artifact_paths: dict[str, Path],
        **forwarded: object,
    ) -> tuple[int, ...]:
        events.append("bind")
        assert artifact_paths == candidate_paths
        assert forwarded == {"layers": (77,), "profile": profile, "strict": True}
        return (77,)

    composite = SimpleNamespace(
        validate_glm52_production_inputs=validate,
        load_authenticated_glm52_composite=load,
    )
    adapter = SimpleNamespace(bind_glm52_vq_experts_from_paths=bind)
    producer = SimpleNamespace(
        FileProducerLockProvider=lambda: SimpleNamespace(
            heavy_job_lock=lambda _path: nullcontext(),
            run_lock=lambda _path: nullcontext(),
        )
    )
    runtime_modules = {
        "mlx_vq.quality.glm52_candidate_eval": candidate,
        "mlx_vq.models.glm52_composite_loader": composite,
        "mlx_vq.models.glm52_vq_adapter": adapter,
        "mlx_vq.quality.glm52_teacher_cache_producer": producer,
    }
    monkeypatch.setattr(module.importlib, "import_module", runtime_modules.__getitem__)

    payload = module.reevaluate_recovery_candidate(**kwargs)

    assert events == [
        "audit",
        "validate",
        "load",
        "resolve",
        "bind",
        "recheck",
        "recheck",
    ]
    assert audit_calls == [
        {
            "profile": profile,
            "expected_seed_manifest_sha256": "a" * 64,
            "expected_stats_manifest_sha256": "b" * 64,
            "expected_full_source_blob_inventory_sha256": "7" * 64,
            "expected_routed_source_blob_inventory_sha256": "8" * 64,
            "expected_recovery_lever": (
                "selection_diagonal_hessian_importance_weighted_reround_v1"
            ),
            "expected_recovery_policy": json.loads(
                Path(kwargs["expected_recovery_policy_json"]).read_text()
            ),
            "accepted_composite_audit_json": kwargs["accepted_composite_audit_json"],
            "expected_composite_audit_sha256": "f" * 64,
        }
    ]
    assert payload["recovered_layers"] == [77]
    assert payload["recovered_groups"] == list(group_keys)
    assert payload["recovery_artifact_audit"] == {
        "audit_pass": True,
        "candidate_artifact_root": str(candidate_root),
        "candidate_identity_sha256": "d" * 64,
        "manifest_path": audit.manifest_path,
        "manifest_body_sha256": "c" * 64,
        "seed_manifest_sha256": "a" * 64,
        "accepted_composite_audit_sha256": "f" * 64,
        "group_count": 225,
        "replacement_group_count": 3,
        "inherited_group_count": 222,
        "complete_replacement_layer_ids": [77],
        "routed_tensor_payload_bytes": 200,
        "non_routed_tensor_payload_bytes": 123,
        "logical_whole_model_tensor_payload_bytes": 323,
        "logical_payload_limit_bytes": 456,
        "incremental_disk_bytes": 99,
    }


@pytest.mark.parametrize("command", ("rematerialize", "run"))
def test_wave1_materialization_commands_forward_external_authority_without_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    module = _load_wave1_cli_module()
    calls: list[dict[str, object]] = []
    stats_dir = tmp_path / ("stats" if command == "rematerialize" else "run/stats")
    stats_dir.mkdir(parents=True)
    stats_manifest = stats_dir / "glm52-recovery-stats-manifest.json"
    stats_manifest.write_text('{"accepted":true}\n')
    attribution = {
        "stats_manifest_sha256": _sha256(stats_manifest),
        "ranked_layers": [{"layer": 3}],
    }
    attribution_path = (
        tmp_path / "attribution.json"
        if command == "rematerialize"
        else tmp_path / "run/layer-attribution.json"
    )
    module._write_json_atomic(attribution_path, attribution)
    authorities = {
        "expected_stats_manifest_sha256": _sha256(stats_manifest),
        "expected_attribution_sha256": _sha256(attribution_path),
        "expected_seed_manifest_sha256": "b" * 64,
        "expected_full_source_blob_inventory_sha256": "7" * 64,
        "expected_routed_source_blob_inventory_sha256": "8" * 64,
        "accepted_composite_audit_json": "accepted-composite.json",
        "expected_composite_audit_sha256": "d" * 64,
    }
    monkeypatch.setattr(module, "materialize_groups_from_source", lambda **kwargs: calls.append(kwargs) or {"materialized": True})
    shared = [
        "--source-dir", "source",
        "--index-path", "index.json",
        "--seed-artifact-dir", "seed",
        "--expected-stats-manifest-sha256", authorities["expected_stats_manifest_sha256"],
        "--expected-attribution-sha256", authorities["expected_attribution_sha256"],
        "--expected-seed-manifest-sha256", authorities["expected_seed_manifest_sha256"],
        "--expected-full-source-blob-inventory-sha256", authorities["expected_full_source_blob_inventory_sha256"],
        "--expected-routed-source-blob-inventory-sha256", authorities["expected_routed_source_blob_inventory_sha256"],
        "--accepted-composite-audit-json", authorities["accepted_composite_audit_json"],
        "--expected-composite-audit-sha256", authorities["expected_composite_audit_sha256"],
    ]
    heavy_lock_path = tmp_path / ".keep-heavy-job.lock"
    if command == "rematerialize":
        argv = [command, *shared, "--stats-dir", str(stats_dir), "--attribution-json", str(attribution_path), "--output-dir", "out", "--worst-layer-count", "1", "--heavy-lock-path", str(heavy_lock_path)]
    else:
        monkeypatch.setattr(module, "_collect", lambda _args, _stats: {"collected": True})
        monkeypatch.setattr(
            module,
            "attribute_recovery_layers",
            lambda **_kwargs: pytest.fail(
                "combined run regenerated attribution instead of consuming pinned authority"
            ),
        )
        monkeypatch.setattr(module, "build_wave1_evidence", lambda **_kwargs: {"evidence": True})
        argv = [
            command,
            "--snapshot-dir", "snapshot",
            "--prompt-pack-json", "prompts.json",
            "--artifact-identities-json", "identities.json",
            "--non-vq-package-dir", "non-vq",
            "--profile-path", "profile.json",
            "--layers", "1",
            *shared,
            "--output-root", str(tmp_path / "run"),
            "--attribution-json", str(attribution_path),
            "--worst-layer-count", "1",
            "--heavy-lock-path", str(heavy_lock_path),
        ]

    assert module.run_cli(argv) == 0
    assert len(calls) == 1
    materialization_authorities = {
        name: value
        for name, value in authorities.items()
        if name != "expected_attribution_sha256"
    }
    assert {
        name: calls[0][name] for name in materialization_authorities
    } == materialization_authorities
    assert Path(calls[0]["heavy_lock_path"]) == heavy_lock_path


@pytest.mark.parametrize("command", ("rematerialize", "run"))
def test_operational_cli_rejects_stats_swap_before_attribution_or_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    module = _load_wave1_cli_module()
    expected = "a" * 64
    if command == "rematerialize":
        stats_dir = tmp_path / "stats"
        output = tmp_path / "out"
        attribution = tmp_path / "attribution.json"
        attribution.write_text("not-json")
    else:
        output = tmp_path / "run"
        stats_dir = output / "stats"
        attribution = tmp_path / "accepted-attribution.json"
    stats_dir.mkdir(parents=True)
    (stats_dir / "glm52-recovery-stats-manifest.json").write_text('{"swapped":true}\n')
    monkeypatch.setattr(
        module,
        "_collect",
        lambda *_args, **_kwargs: pytest.fail("collection ran before stats authentication"),
    )
    monkeypatch.setattr(
        module,
        "attribute_recovery_layers",
        lambda **_kwargs: pytest.fail("attribution ran before stats authentication"),
    )
    monkeypatch.setattr(
        module,
        "materialize_groups_from_source",
        lambda **_kwargs: pytest.fail("materialization ran before stats authentication"),
    )
    shared = [
        "--source-dir", "source",
        "--index-path", "index.json",
        "--seed-artifact-dir", "seed",
        "--expected-stats-manifest-sha256", expected,
        "--expected-attribution-sha256", "c" * 64,
        "--expected-seed-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", FULL_SOURCE_BLOB_INVENTORY_SHA256,
        "--expected-routed-source-blob-inventory-sha256", ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
        "--accepted-composite-audit-json", "accepted-composite.json",
        "--expected-composite-audit-sha256", "d" * 64,
    ]
    if command == "rematerialize":
        argv = [
            command, *shared,
            "--stats-dir", str(stats_dir),
            "--attribution-json", str(attribution),
            "--output-dir", str(output),
        ]
    else:
        argv = [
            command,
            "--snapshot-dir", "snapshot",
            "--prompt-pack-json", "prompts.json",
            "--artifact-identities-json", "identities.json",
            "--non-vq-package-dir", "non-vq",
            "--profile-path", "profile.json",
            "--layers", "1",
            *shared,
            "--output-root", str(output),
            "--attribution-json", str(attribution),
        ]

    with pytest.raises(ValueError, match="stats manifest SHA-256.*external accepted authority"):
        module.run_cli(argv)


@pytest.mark.parametrize("command", ("rematerialize", "run"))
def test_operational_cli_rejects_attribution_replay_before_group_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    module = _load_wave1_cli_module()
    output = tmp_path / command
    stats_dir = output / "stats"
    stats_dir.mkdir(parents=True)
    stats_path = stats_dir / "glm52-recovery-stats-manifest.json"
    stats_path.write_text('{"accepted":true}\n')
    accepted = _sha256(stats_path)
    replay = {"stats_manifest_sha256": "f" * 64, "ranked_layers": [{"layer": 3}]}
    monkeypatch.setattr(module, "_collect", lambda *_args, **_kwargs: {"collected": True})
    accepted_attribution = tmp_path / "accepted-attribution.json"
    module._write_json_atomic(accepted_attribution, replay)
    expected_attribution_sha256 = _sha256(accepted_attribution)
    monkeypatch.setattr(
        module,
        "attribute_recovery_layers",
        lambda **kwargs: module._write_json_atomic(kwargs["output_json"], replay) or replay,
    )
    monkeypatch.setattr(
        module,
        "materialize_groups_from_source",
        lambda **_kwargs: pytest.fail("groups selected from replayed attribution"),
    )
    shared = [
        "--source-dir", "source",
        "--index-path", "index.json",
        "--seed-artifact-dir", "seed",
        "--expected-stats-manifest-sha256", accepted,
        "--expected-attribution-sha256", expected_attribution_sha256,
        "--expected-seed-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", FULL_SOURCE_BLOB_INVENTORY_SHA256,
        "--expected-routed-source-blob-inventory-sha256", ROUTED_SOURCE_BLOB_INVENTORY_SHA256,
        "--accepted-composite-audit-json", "accepted-composite.json",
        "--expected-composite-audit-sha256", "d" * 64,
        "--worst-layer-count", "1",
    ]
    if command == "rematerialize":
        attribution_path = tmp_path / "replayed-attribution.json"
        module._write_json_atomic(attribution_path, replay)
        argv = [
            command, *shared,
            "--stats-dir", str(stats_dir),
            "--attribution-json", str(attribution_path),
            "--output-dir", str(output / "candidate"),
        ]
    else:
        argv = [
            command,
            "--snapshot-dir", "snapshot",
            "--prompt-pack-json", "prompts.json",
            "--artifact-identities-json", "identities.json",
            "--non-vq-package-dir", "non-vq",
            "--profile-path", "profile.json",
            "--layers", "1",
            *shared,
            "--output-root", str(output),
            "--attribution-json", str(accepted_attribution),
        ]

    with pytest.raises(ValueError, match="attribution.*stats manifest SHA-256"):
        module.run_cli(argv)


def test_rematerialize_rejects_ranked_layers_edit_with_unchanged_stats_sha(
    tmp_path: Path,
) -> None:
    module = _load_wave1_cli_module()
    stats_sha256 = "a" * 64
    path = tmp_path / "layer-attribution.json"
    path.write_text(
        json.dumps(
            {
                "stats_manifest_sha256": stats_sha256,
                "ranked_layers": [{"rank": 1, "layer": 3}],
            },
            sort_keys=True,
        )
    )
    accepted_attribution_sha256 = _sha256(path)
    path.write_text(
        json.dumps(
            {
                "stats_manifest_sha256": stats_sha256,
                "ranked_layers": [{"rank": 1, "layer": 77}],
            },
            sort_keys=True,
        )
    )

    with pytest.raises(ValueError, match="attribution SHA-256.*external accepted authority"):
        module._load_bound_attribution(
            path,
            expected_attribution_sha256=accepted_attribution_sha256,
            expected_stats_manifest_sha256=stats_sha256,
        )


def test_run_lifecycle_collects_clean_stats_root_then_authenticates(
    tmp_path: Path,
) -> None:
    module = _load_wave1_cli_module()
    stats_dir = tmp_path / "run/stats"
    manifest_payload = b'{"status":"complete"}\n'
    expected = hashlib.sha256(manifest_payload).hexdigest()
    events: list[str] = []

    def collect() -> None:
        events.append("collect")
        assert not stats_dir.exists()
        stats_dir.mkdir(parents=True)
        (stats_dir / "glm52-recovery-stats-manifest.json").write_bytes(manifest_payload)

    result = module._prepare_authenticated_stats_root(
        stats_dir,
        expected_sha256=expected,
        collector=collect,
        event_sink=events.append,
    )

    assert result["status"] == "complete"
    assert events == ["collect", "authenticate"]


def test_run_lifecycle_consumes_accepted_stats_root_without_recollection(
    tmp_path: Path,
) -> None:
    module = _load_wave1_cli_module()
    stats_dir = tmp_path / "run/stats"
    stats_dir.mkdir(parents=True)
    manifest = stats_dir / "glm52-recovery-stats-manifest.json"
    manifest.write_text('{"status":"complete"}\n')
    events: list[str] = []

    result = module._prepare_authenticated_stats_root(
        stats_dir,
        expected_sha256=_sha256(manifest),
        collector=lambda: pytest.fail("accepted stats root was recollected"),
        event_sink=events.append,
    )

    assert result["status"] == "complete"
    assert events == ["authenticate"]


def test_materialization_rejects_selected_seed_group_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="seed group.*SHA-256|seed group.*size"):
        _run_materialization(
            tmp_path,
            monkeypatch,
            resume=False,
            tamper_seed_group=True,
        )


def test_materialization_rejects_seed_manifest_outside_exact_225_group_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    path = tmp_path / "seed/conversion-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["attacker_field"] = True
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="seed manifest.*schema|seed manifest.*keys"):
        _run_materialization(tmp_path, monkeypatch, resume=True)


def test_materialization_rejects_stats_npz_swap_after_manifest_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    path = tmp_path / "stats/layer-00003-gate_proj-expert-000.npz"
    np.savez(
        path,
        sum_x2=np.full(8, 9.0, dtype=np.float32),
        mean_second_moment=np.full(8, 9.0, dtype=np.float32),
        routing_weighted_importance=np.full(8, 9.0, dtype=np.float32),
        router_score_weighted_importance=np.full(8, 9.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="recovery stats file.*SHA-256"):
        _run_materialization(tmp_path, monkeypatch, resume=True)


def test_attribution_uses_only_authenticated_stats_seed_and_source_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, reads = _prepare_attribution_fixture(tmp_path, monkeypatch)
    live_seed_paths = {
        Path(kwargs["seed_artifact_dir"]) / f"layer-00003-{projection}.safetensors"
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    live_stats_paths = set(Path(kwargs["stats_dir"]).glob("*.npz"))
    real_safe_open = module.safe_open
    real_np_load = module.np.load

    def authenticated_safe_open(path: str | Path, *args: object, **options: object):
        if Path(path) in live_seed_paths:
            pytest.fail("attribution opened a mutable live seed group")
        return real_safe_open(path, *args, **options)

    def authenticated_np_load(file: object, *args: object, **options: object):
        if isinstance(file, (str, Path)) and Path(file) in live_stats_paths:
            pytest.fail("attribution opened a mutable live stats NPZ")
        return real_np_load(file, *args, **options)

    monkeypatch.setattr(module, "safe_open", authenticated_safe_open)
    monkeypatch.setattr(module.np, "load", authenticated_np_load)
    monkeypatch.setitem(
        sys.modules,
        "mlx_vq.convert.stream_convert",
        SimpleNamespace(
            load_safetensors_index=lambda _path: pytest.fail(
                "attribution loaded the mutable source index"
            )
        ),
    )

    payload = module.attribute_recovery_layers(**kwargs)

    assert len(payload["ranked_layers"]) == 1
    assert payload["ranked_layers"][0]["rank"] == 1
    assert payload["ranked_layers"][0]["layer"] == 3
    assert payload["ranked_layers"][0]["weighted_reconstruction_error"] > 0
    assert reads.authenticated_source_opens == 1
    assert reads.verified_after_reads == 1
    assert reads.closed_inventories == 1


def test_attribution_rejects_stats_npz_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _reads = _prepare_attribution_fixture(tmp_path, monkeypatch)
    path = Path(kwargs["stats_dir"]) / "layer-00003-gate_proj-expert-000.npz"
    np.savez(
        path,
        sum_x2=np.full(8, 9.0, dtype=np.float32),
        mean_second_moment=np.full(8, 9.0, dtype=np.float32),
        routing_weighted_importance=np.full(8, 9.0, dtype=np.float32),
        router_score_weighted_importance=np.full(8, 9.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="recovery stats file.*SHA-256"):
        module.attribute_recovery_layers(**kwargs)


def test_attribution_rejects_selected_seed_group_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _reads = _prepare_attribution_fixture(tmp_path, monkeypatch)
    _artifact_group(
        Path(kwargs["seed_artifact_dir"]) / "layer-00003-gate_proj.safetensors",
        layer=3,
        projection="gate_proj",
        fill=2,
    )

    with pytest.raises(ValueError, match="seed group.*SHA-256|seed group.*size"):
        module.attribute_recovery_layers(**kwargs)


def test_attribution_rejects_source_shard_tamper_after_authenticated_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, reads = _prepare_attribution_fixture(
        tmp_path,
        monkeypatch,
        verify_after_error="source shard changed during authenticated reads",
    )

    with pytest.raises(ValueError, match="source shard changed"):
        module.attribute_recovery_layers(**kwargs)

    assert reads.verified_after_reads == 1
    assert reads.closed_inventories == 1


def _attribute_cli_base() -> list[str]:
    return [
        "attribute",
        "--source-dir", "source",
        "--index-path", "index.json",
        "--stats-dir", "stats",
        "--output-json", "attribution.json",
        "--expected-stats-manifest-sha256", "a" * 64,
        "--expected-seed-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", "7" * 64,
        "--expected-routed-source-blob-inventory-sha256", "8" * 64,
    ]


def _attribute_mode_argv(mode: str) -> list[str]:
    argv = _attribute_cli_base()
    if mode == "seed":
        return [*argv, "--seed-artifact-dir", "seed"]
    if mode != "recovered":
        raise AssertionError(mode)
    return [
        *argv,
        "--recovery-conversion-dir", "recovery",
        "--profile-path", "profile.yaml",
        "--expected-recovery-manifest-sha256", "c" * 64,
        "--expected-recovery-lever",
        "selection_diagonal_hessian_importance_weighted_reround_v1",
        "--expected-recovery-policy-json", "recovery-policy.json",
        "--accepted-composite-audit-json", "composite.json",
        "--expected-composite-audit-sha256", "d" * 64,
    ]


@pytest.mark.parametrize("mode", ("seed", "recovered"))
@pytest.mark.parametrize("declared", (False, True))
def test_attribute_cli_holds_declared_or_default_heavy_lock_for_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    declared: bool,
) -> None:
    module = _load_wave1_cli_module()
    events: list[str] = []
    expected_lock = (
        tmp_path / "declared-heavy.lock"
        if declared
        else Path(".keep-heavy-job.lock")
    )

    @contextmanager
    def lock(path: str | Path):
        assert Path(path) == expected_lock
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def attribute(**_kwargs: object) -> dict[str, bool]:
        events.append("core")
        return {"complete": True}

    monkeypatch.setattr(module, "_cooperating_file_lock", lock)
    monkeypatch.setattr(module, "attribute_recovery_layers", attribute)
    argv = _attribute_mode_argv(mode)
    if declared:
        argv.extend(["--heavy-lock-path", str(expected_lock)])
    assert module.run_cli(argv) == 0
    assert events == ["enter", "core", "exit"]


@pytest.mark.parametrize("mode", ("seed", "recovered"))
def test_attribute_cli_releases_heavy_lock_when_core_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    module = _load_wave1_cli_module()
    lock_path = tmp_path / "heavy.lock"
    monkeypatch.setattr(
        module,
        "attribute_recovery_layers",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("core failed")),
    )
    with pytest.raises(RuntimeError, match="core failed"):
        module.run_cli(
            [
                *_attribute_mode_argv(mode),
                "--heavy-lock-path", str(lock_path),
            ]
        )
    reacquired = False
    with module._cooperating_file_lock(lock_path):
        reacquired = True
    assert reacquired


def test_attribute_cli_never_enters_core_while_heavy_lock_is_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    lock_path = tmp_path / "heavy.lock"
    core_entered = threading.Event()
    worker_done = threading.Event()
    errors: list[BaseException] = []
    monkeypatch.setattr(
        module,
        "attribute_recovery_layers",
        lambda **_kwargs: core_entered.set() or {"complete": True},
    )

    def worker() -> None:
        try:
            module.run_cli(
                [
                    *_attribute_mode_argv("seed"),
                    "--heavy-lock-path", str(lock_path),
                ]
            )
        except BaseException as error:
            errors.append(error)
        finally:
            worker_done.set()

    with module._cooperating_file_lock(lock_path):
        thread = threading.Thread(target=worker)
        thread.start()
        assert not core_entered.wait(0.1)
        assert not worker_done.is_set()
    assert core_entered.wait(2)
    assert worker_done.wait(2)
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []


def test_attribute_cli_requires_exactly_one_seed_or_recovery_artifact() -> None:
    module = _load_wave1_cli_module()
    base = _attribute_cli_base()
    with pytest.raises(SystemExit):
        module._build_parser().parse_args(base)
    with pytest.raises(SystemExit):
        module._build_parser().parse_args(
            [
                *base,
                "--seed-artifact-dir", "seed",
                "--recovery-conversion-dir", "recovery",
            ]
        )


@pytest.mark.parametrize(
    "missing_option",
    (
        "--profile-path",
        "--expected-recovery-manifest-sha256",
        "--expected-recovery-lever",
        "--expected-recovery-policy-json",
        "--accepted-composite-audit-json",
        "--expected-composite-audit-sha256",
    ),
)
def test_recovered_attribute_cli_requires_complete_audit_authority(
    missing_option: str,
) -> None:
    module = _load_wave1_cli_module()
    argv = [
        *_attribute_cli_base(),
        "--recovery-conversion-dir", "recovery",
        "--profile-path", "profile.yaml",
        "--expected-recovery-manifest-sha256", "c" * 64,
        "--expected-recovery-lever",
        "selection_diagonal_hessian_importance_weighted_reround_v1",
        "--expected-recovery-policy-json", "recovery-policy.json",
        "--accepted-composite-audit-json", "composite.json",
        "--expected-composite-audit-sha256", "d" * 64,
    ]
    index = argv.index(missing_option)
    del argv[index : index + 2]
    with pytest.raises(ValueError, match="recovered attribution requires"):
        module.run_cli(argv)


def test_recovered_attribution_binds_audited_candidate_and_selection_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _audit, events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    real_decode = module.decode_weight_matrix
    real_write = module._write_json_atomic

    def decode(*args: object, **options: object) -> np.ndarray:
        events.append("compare")
        return real_decode(*args, **options)

    def write(path: str | Path, value: dict[str, object]) -> None:
        events.append("write")
        real_write(path, value)

    monkeypatch.setattr(module, "decode_weight_matrix", decode)
    monkeypatch.setattr(module, "_write_json_atomic", write)
    payload = module.attribute_recovery_layers(**kwargs)

    assert payload["source_artifact_kind"] == "recovery_mixed"
    assert payload["raw_recovery_manifest_sha256"] == kwargs[
        "expected_recovery_manifest_sha256"
    ]
    assert payload["candidate_identity_sha256"] == "d" * 64
    assert payload["baseline_composite_identity_sha256"] == "e" * 64
    assert payload["split_evidence"] == {
        "split": "selection",
        "prompt_count": 22,
        "position_count": 255,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    assert events[0] == "verify"
    assert "compare" in events[1:-2]
    assert events[-2:] == ["write", "verify"]


def test_recovered_attribution_rejects_manifest_hash_before_other_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    kwargs["expected_recovery_manifest_sha256"] = "0" * 64
    monkeypatch.setattr(
        module,
        "_authenticate_stats_manifest",
        lambda *_args, **_kwargs: pytest.fail("stats read before raw manifest pin"),
    )
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: pytest.fail("audit loaded before raw manifest pin"),
    )
    with pytest.raises(ValueError, match="raw recovery manifest SHA-256"):
        module.attribute_recovery_layers(**kwargs)


def test_recovered_attribution_rejects_same_body_manifest_swap_after_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    manifest = Path(kwargs["recovery_conversion_dir"]) / "conversion-manifest.json"

    def audit_then_swap(*_args: object, **_options: object) -> object:
        manifest.write_text('{ "status" : "complete" }\n')
        return audit

    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=audit_then_swap
        ),
    )
    monkeypatch.setattr(
        module,
        "_authenticate_stats_manifest",
        lambda *_args, **_kwargs: pytest.fail(
            "stats read after raw manifest substitution"
        ),
    )
    with pytest.raises(ValueError, match="raw recovery manifest SHA-256"):
        module.attribute_recovery_layers(**kwargs)


@pytest.mark.parametrize(
    ("audit_pass", "group_count", "replacement_count", "inherited_count", "error"),
    (
        (False, 225, 3, 222, "audit did not pass"),
        (True, 224, 3, 221, "complete 225-group"),
        (True, 225, 3, 221, "complete candidate"),
    ),
)
def test_recovered_attribution_requires_complete_passing_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit_pass: bool,
    group_count: int,
    replacement_count: int,
    inherited_count: int,
    error: str,
) -> None:
    module, kwargs, audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    audit.audit_pass = audit_pass
    audit.group_count = group_count
    audit.replacement_group_count = replacement_count
    audit.inherited_group_count = inherited_count
    with pytest.raises(ValueError, match=error):
        module.attribute_recovery_layers(**kwargs)


def test_recovered_attribution_rejects_candidate_group_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    audit.groups[0].artifact_sha256 = "0" * 64
    with pytest.raises(ValueError, match="candidate group.*SHA-256"):
        module.attribute_recovery_layers(**kwargs)


def test_recovered_attribution_rejects_symlink_retarget_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    candidate = Path(kwargs["recovery_conversion_dir"]) / "artifact"
    link = candidate / "layer-00003-gate_proj.safetensors"
    alternate = tmp_path / "alternate.safetensors"
    _artifact_group(alternate, layer=3, projection="gate_proj", fill=2)
    real_clone = module.clone_or_copy_authenticated
    changed = False

    def retarget(source: object, destination: Path, **options: object) -> object:
        nonlocal changed
        if not changed:
            changed = True
            link.unlink()
            link.symlink_to(alternate)
        return real_clone(source, destination, **options)

    monkeypatch.setattr(module, "clone_or_copy_authenticated", retarget)
    with pytest.raises(ValueError, match="retargeted|identity changed"):
        module.attribute_recovery_layers(**kwargs)


def test_recovered_attribution_rechecks_identity_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    checks = 0

    def verify() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("candidate changed after attribution publication")

    audit.verify_current_identity = verify
    with pytest.raises(ValueError, match="changed after attribution"):
        module.attribute_recovery_layers(**kwargs)
    assert checks == 2
    assert Path(kwargs["output_json"]).is_file()


def test_recovered_attribution_rejects_same_body_manifest_swap_after_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    manifest = Path(kwargs["recovery_conversion_dir"]) / "conversion-manifest.json"
    real_write = module._write_json_atomic

    def write_then_swap(path: str | Path, payload: dict[str, object]) -> None:
        real_write(path, payload)
        manifest.write_text('{ "status" : "complete" }\n')

    monkeypatch.setattr(module, "_write_json_atomic", write_then_swap)
    with pytest.raises(ValueError, match="raw recovery manifest SHA-256"):
        module.attribute_recovery_layers(**kwargs)
    assert Path(kwargs["output_json"]).is_file()


def test_seed_attribution_retains_legacy_output_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, _reads = _prepare_attribution_fixture(tmp_path, monkeypatch)
    payload = module.attribute_recovery_layers(**kwargs)
    assert set(payload) == {
        "schema_version",
        "record_type",
        "status",
        "importance_kind",
        "split_evidence",
        "stats_manifest_path",
        "stats_manifest_sha256",
        "selected_layers",
        "ranked_layers",
        "groups",
    }
    assert set(payload["groups"][0]) == {
        "layer",
        "projection",
        "weighted_reconstruction_error",
        "routed_expert_count",
        "route_count",
        "seed_group_path",
        "seed_group_sha256",
    }


def test_changed_recovered_bytes_change_layer_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, kwargs, audit, _events = _prepare_recovered_attribution_fixture(
        tmp_path, monkeypatch
    )
    stats_root = Path(kwargs["stats_dir"])
    stats_path = stats_root / "glm52-recovery-stats-manifest.json"
    stats_manifest = json.loads(stats_path.read_text())
    layer_four_entries: list[dict[str, object]] = []
    candidate_root = Path(kwargs["recovery_conversion_dir"]) / "artifact"
    seed_root = tmp_path / "seed"
    records = list(audit.groups)
    records = [record for record in records if not record.group_key.startswith("unused-")]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        source = seed_root / f"layer-00004-{projection}.safetensors"
        _artifact_group(source, layer=4, projection=projection, fill=1)
        (candidate_root / source.name).symlink_to(source)
        projection_manifest = _stats_manifest(
            stats_root,
            layer=4,
            projection=projection,
            config_sha256=stats_manifest["source_authority"]["config_sha256"],
            index_sha256=stats_manifest["source_authority"]["index_sha256"],
        )
        layer_four_entries.extend(projection_manifest["entries"])
        records.append(
            SimpleNamespace(
                group_key=f"4:{projection}",
                filename=source.name,
                classification="replacement",
                artifact_sha256=_sha256(source),
                artifact_bytes=source.stat().st_size,
            )
        )
    records.extend(
        SimpleNamespace(
            group_key=f"unused-two-layer-{index}",
            filename=f"unused-two-layer-{index}.safetensors",
            classification="inherited",
            artifact_sha256="0" * 64,
            artifact_bytes=1,
        )
        for index in range(219)
    )
    audit.groups = tuple(records)
    audit.replacement_group_count = 6
    audit.inherited_group_count = 219
    stats_manifest["selected_layers"] = [3, 4]
    stats_manifest["route_count_by_layer"]["4"] = 2040
    stats_manifest["entries"].extend(layer_four_entries)
    stats_manifest["entry_count"] = len(stats_manifest["entries"])
    _write_stats_manifest(stats_root, stats_manifest)
    kwargs["expected_stats_manifest_sha256"] = _sha256(stats_path)

    first = module.attribute_recovery_layers(**kwargs)
    first_order = [row["layer"] for row in first["ranked_layers"]]

    for projection in ("gate_proj", "up_proj", "down_proj"):
        source = seed_root / f"layer-00004-{projection}.safetensors"
        _artifact_group(source, layer=4, projection=projection, fill=250)
        record = next(
            item for item in audit.groups if item.group_key == f"4:{projection}"
        )
        record.artifact_sha256 = _sha256(source)
        record.artifact_bytes = source.stat().st_size
    second = module.attribute_recovery_layers(**kwargs)
    second_order = [row["layer"] for row in second["ranked_layers"]]
    assert second_order != first_order


def _audit_cli_argv(tmp_path: Path) -> list[str]:
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir(exist_ok=True)
    manifest = recovery_root / "conversion-manifest.json"
    if not manifest.exists():
        manifest.write_text('{"status":"complete"}\n')
    return [
        "audit",
        "--recovery-conversion-dir", str(recovery_root),
        "--output-json", str(tmp_path / "audit.json"),
        "--profile-path", str(tmp_path / "profile.yaml"),
        "--expected-recovery-manifest-sha256", _sha256(manifest),
        "--expected-seed-manifest-sha256", "a" * 64,
        "--expected-stats-manifest-sha256", "b" * 64,
        "--expected-full-source-blob-inventory-sha256", "7" * 64,
        "--expected-routed-source-blob-inventory-sha256", "8" * 64,
        "--expected-recovery-lever",
        "selection_diagonal_hessian_importance_weighted_reround_v1",
        "--expected-recovery-policy-json", str(tmp_path / "policy.json"),
        "--accepted-composite-audit-json", str(tmp_path / "composite.json"),
        "--expected-composite-audit-sha256", "f" * 64,
        "--heavy-lock-path", str(tmp_path / "heavy.lock"),
    ]


@pytest.mark.parametrize(
    "missing_option",
    (
        "--recovery-conversion-dir",
        "--output-json",
        "--profile-path",
        "--expected-recovery-manifest-sha256",
        "--expected-seed-manifest-sha256",
        "--expected-stats-manifest-sha256",
        "--expected-full-source-blob-inventory-sha256",
        "--expected-routed-source-blob-inventory-sha256",
        "--expected-recovery-lever",
        "--expected-recovery-policy-json",
        "--accepted-composite-audit-json",
        "--expected-composite-audit-sha256",
    ),
)
def test_audit_subcommand_requires_complete_authority(
    tmp_path: Path, missing_option: str
) -> None:
    module = _load_wave1_cli_module()
    argv = _audit_cli_argv(tmp_path)
    index = argv.index(missing_option)
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        module._build_parser().parse_args(argv)


def test_audit_subcommand_locks_and_revalidates_around_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(_recovery_policy(), sort_keys=True) + "\n")
    events: list[str] = []
    audit_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @contextmanager
    def lock(path: str | Path):
        events.append(f"lock:{Path(path).name}")
        yield

    audit = SimpleNamespace(
        audit_pass=True,
        group_count=225,
        replacement_group_count=3,
        inherited_group_count=222,
        groups=tuple(range(225)),
        verify_current_identity=lambda: events.append("verify"),
        to_dict=lambda: {
            "audit_pass": True,
            "group_count": 225,
            "candidate_identity_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(module, "_cooperating_file_lock", lock)
    profile = object()
    monkeypatch.setattr(module, "_load_recovery_profile", lambda _path: profile)

    def run_audit(*args: object, **options: object) -> object:
        events.append("audit")
        audit_calls.append((args, options))
        return audit

    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=run_audit
        ),
    )
    real_write = module._write_json_atomic

    def write(path: str | Path, payload: dict[str, object]) -> None:
        events.append("write")
        real_write(path, payload)

    monkeypatch.setattr(module, "_write_json_atomic", write)

    assert module.run_cli(_audit_cli_argv(tmp_path)) == 0
    assert events == [
        "lock:heavy.lock",
        "lock:audit.lock",
        "audit",
        "verify",
        "write",
        "verify",
    ]
    assert audit_calls == [
        (
            (str(tmp_path / "recovery"),),
            {
                "profile": profile,
                "expected_seed_manifest_sha256": "a" * 64,
                "expected_stats_manifest_sha256": "b" * 64,
                "expected_full_source_blob_inventory_sha256": "7" * 64,
                "expected_routed_source_blob_inventory_sha256": "8" * 64,
                "expected_recovery_lever": (
                    "selection_diagonal_hessian_importance_weighted_reround_v1"
                ),
                "expected_recovery_policy": _recovery_policy(),
                "accepted_composite_audit_json": str(
                    tmp_path / "composite.json"
                ),
                "expected_composite_audit_sha256": "f" * 64,
            },
        )
    ]
    assert json.loads((tmp_path / "audit.json").read_text())["audit_pass"] is True


def test_audit_subcommand_rejects_policy_symlink_before_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    policy = tmp_path / "real-policy.json"
    policy.write_text(json.dumps(_recovery_policy(), sort_keys=True) + "\n")
    (tmp_path / "policy.json").symlink_to(policy)
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: pytest.fail("audit ran before policy bytes were authenticated"),
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        module.run_cli(_audit_cli_argv(tmp_path))


def test_audit_subcommand_rejects_same_body_manifest_swap_after_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    (tmp_path / "policy.json").write_text(
        json.dumps(_recovery_policy(), sort_keys=True) + "\n"
    )
    argv = _audit_cli_argv(tmp_path)
    manifest = tmp_path / "recovery/conversion-manifest.json"
    audit = SimpleNamespace(
        audit_pass=True,
        group_count=225,
        replacement_group_count=3,
        inherited_group_count=222,
        groups=tuple(range(225)),
        verify_current_identity=lambda: None,
        to_dict=lambda: {"audit_pass": True, "group_count": 225},
    )

    def audit_then_swap(*_args: object, **_options: object) -> object:
        manifest.write_text('{ "status" : "complete" }\n')
        return audit

    monkeypatch.setattr(module, "_load_recovery_profile", lambda _path: object())
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=audit_then_swap
        ),
    )
    monkeypatch.setattr(
        module,
        "_write_json_atomic",
        lambda *_args, **_kwargs: pytest.fail(
            "audit output written after raw manifest substitution"
        ),
    )
    with pytest.raises(ValueError, match="raw recovery manifest SHA-256"):
        module.run_cli(argv)


def test_audit_subcommand_rejects_same_body_manifest_swap_after_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    (tmp_path / "policy.json").write_text(
        json.dumps(_recovery_policy(), sort_keys=True) + "\n"
    )
    argv = _audit_cli_argv(tmp_path)
    manifest = tmp_path / "recovery/conversion-manifest.json"
    audit = SimpleNamespace(
        audit_pass=True,
        group_count=225,
        replacement_group_count=3,
        inherited_group_count=222,
        groups=tuple(range(225)),
        verify_current_identity=lambda: None,
        to_dict=lambda: {"audit_pass": True, "group_count": 225},
    )
    monkeypatch.setattr(module, "_load_recovery_profile", lambda _path: object())
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=lambda *_args, **_kwargs: audit
        ),
    )
    real_write = module._write_json_atomic

    def write_then_swap(path: str | Path, payload: dict[str, object]) -> None:
        real_write(path, payload)
        manifest.write_text('{ "status" : "complete" }\n')

    monkeypatch.setattr(module, "_write_json_atomic", write_then_swap)
    with pytest.raises(ValueError, match="raw recovery manifest SHA-256"):
        module.run_cli(argv)
    assert (tmp_path / "audit.json").is_file()


@pytest.mark.parametrize(
    ("audit_pass", "group_count", "identity_error", "error"),
    (
        (False, 225, None, "audit did not pass"),
        (True, 224, None, "complete 225-group"),
        (True, 225, "candidate changed", "candidate changed"),
    ),
)
def test_audit_subcommand_fails_closed_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit_pass: bool,
    group_count: int,
    identity_error: str | None,
    error: str,
) -> None:
    module = _load_wave1_cli_module()
    (tmp_path / "policy.json").write_text(
        json.dumps(_recovery_policy(), sort_keys=True) + "\n"
    )

    def verify() -> None:
        if identity_error is not None:
            raise ValueError(identity_error)

    audit = SimpleNamespace(
        audit_pass=audit_pass,
        group_count=group_count,
        replacement_group_count=3,
        inherited_group_count=222,
        groups=tuple(range(225)),
        verify_current_identity=verify,
        to_dict=lambda: {"audit_pass": audit_pass, "group_count": group_count},
    )
    monkeypatch.setattr(module, "_load_recovery_profile", lambda _path: object())
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=lambda *_args, **_kwargs: audit
        ),
    )
    with pytest.raises(ValueError, match=error):
        module.run_cli(_audit_cli_argv(tmp_path))
    assert not (tmp_path / "audit.json").exists()


def test_audit_subcommand_rechecks_identity_after_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_wave1_cli_module()
    (tmp_path / "policy.json").write_text(
        json.dumps(_recovery_policy(), sort_keys=True) + "\n"
    )
    checks = 0

    def verify() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("candidate changed after audit publication")

    audit = SimpleNamespace(
        audit_pass=True,
        group_count=225,
        replacement_group_count=3,
        inherited_group_count=222,
        groups=tuple(range(225)),
        verify_current_identity=verify,
        to_dict=lambda: {"audit_pass": True, "group_count": 225},
    )
    monkeypatch.setattr(module, "_load_recovery_profile", lambda _path: object())
    monkeypatch.setattr(
        module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=lambda *_args, **_kwargs: audit
        ),
    )
    with pytest.raises(ValueError, match="changed after audit publication"):
        module.run_cli(_audit_cli_argv(tmp_path))
    assert checks == 2
    assert (tmp_path / "audit.json").is_file()


def test_materialization_never_reopens_selected_live_seed_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_safe_open = recovery_materialize.safe_open

    def snapshot_only(path: str | Path, *args: object, **options: object):
        candidate = Path(path)
        if candidate.parent.name == "seed" and candidate.name.endswith(".safetensors"):
            pytest.fail("materialization reopened a mutable selected seed group")
        return real_safe_open(path, *args, **options)

    monkeypatch.setattr(recovery_materialize, "safe_open", snapshot_only)

    _run_materialization(tmp_path, monkeypatch, resume=False)


def test_importance_weighted_quantization_beats_max_abs_on_constructed_case() -> None:
    source = np.array(
        [[0.18, -0.21, 0.16, -0.20, 0.19, -0.17, 0.22, -7.5]],
        dtype=np.float32,
    )
    importance = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.001], dtype=np.float32)

    baseline = quantize_weight_rtn(source, group_size=8, code_bits=8)
    recovered = quantize_weight_importance_aware(
        source,
        importance,
        group_size=8,
        code_bits=8,
    )

    baseline_error = _weighted_error(source, dequantize_weight_np(baseline), importance)
    recovered_weight = decode_weight_matrix(
        recovered.codes,
        recovered.scales,
        code_bits=8,
        codebook=recovered.codebook,
    )
    assert _weighted_error(source, recovered_weight, importance) < baseline_error


def test_importance_weighted_e8p_is_canonical_uint16_and_chunk_invariant() -> None:
    source = np.array(
        [[0.18, -0.21, 0.16, -0.20, 0.19, -0.17, 0.22, -7.5]],
        dtype=np.float32,
    )
    importance = np.array(
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.001],
        dtype=np.float32,
    )

    small = quantize_weight_importance_aware(
        source,
        importance,
        group_size=8,
        code_bits=16,
        codeword_chunk_size=17,
    )
    large = quantize_weight_importance_aware(
        source,
        importance,
        group_size=8,
        code_bits=16,
        codeword_chunk_size=8192,
    )
    baseline = quantize_weight_rtn(source, group_size=8, code_bits=16)
    recovered_weight = decode_weight_matrix(
        small.codes,
        small.scales,
        code_bits=16,
        codebook=small.codebook,
    )

    assert small.codes.dtype == np.uint16
    np.testing.assert_array_equal(small.codebook, e8p_packed_abs_grid())
    assert hashlib.sha256(small.codebook.astype("<u4").tobytes()).hexdigest() == E8P_PACKED_ABS_SHA256
    np.testing.assert_array_equal(small.codes, large.codes)
    np.testing.assert_array_equal(small.scales, large.scales)
    assert _weighted_error(source, recovered_weight, importance) < _weighted_error(
        source,
        dequantize_weight_np(baseline),
        importance,
    )


def test_zero_importance_e8_preserves_seed_and_e8p_uses_source_rtn(tmp_path: Path) -> None:
    seed = tmp_path / "seed.safetensors"
    _artifact_group(seed, layer=3, projection="gate_proj", fill=7)
    source = np.array(
        [
            [[0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1]],
            [[-0.4, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    importance = np.zeros((2, 8), dtype=np.float32)

    e8_path = tmp_path / "e8.safetensors"
    e8_record = materialize_recovery_group(
        source_weights=source,
        importance=importance,
        seed_group_path=seed,
        output_path=e8_path,
        layer=3,
        projection="gate_proj",
        group_size=8,
        code_bits=8,
    )
    e8p_path = tmp_path / "e8p.safetensors"
    e8p_record = materialize_recovery_group(
        source_weights=source,
        importance=importance,
        seed_group_path=seed,
        output_path=e8p_path,
        layer=3,
        projection="gate_proj",
        group_size=8,
        code_bits=16,
    )

    codes_name = "model.layers.3.mlp.switch_mlp.gate_proj.codes"
    scales_name = "model.layers.3.mlp.switch_mlp.gate_proj.scales"
    with safe_open(seed, framework="np") as seed_handle, safe_open(
        e8_path, framework="np"
    ) as e8_handle, safe_open(e8p_path, framework="np") as e8p_handle:
        np.testing.assert_array_equal(e8_handle.get_tensor(codes_name), seed_handle.get_tensor(codes_name))
        np.testing.assert_array_equal(e8_handle.get_tensor(scales_name), seed_handle.get_tensor(scales_name))
        expected = [quantize_weight_rtn(expert, group_size=8, code_bits=16) for expert in source]
        np.testing.assert_array_equal(
            e8p_handle.get_tensor(codes_name),
            np.stack([item.codes for item in expected]),
        )
        np.testing.assert_array_equal(
            e8p_handle.get_tensor(scales_name),
            np.stack([item.scales for item in expected]),
        )
        np.testing.assert_array_equal(
            e8p_handle.get_tensor("model.vq_codebook.e8"), e8p_packed_abs_grid()
        )
    assert e8_record["zero_importance_policy"] == "preserve_seed_expert_bytes"
    assert e8p_record["zero_importance_policy"] == "source_rtn_e8p"


def test_explicit_layer_rates_reject_partial_or_out_of_range_before_lock_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = False

    def forbidden(**_kwargs: object) -> dict[str, object]:
        nonlocal entered
        entered = True
        raise AssertionError("materializer entered after invalid rate policy")

    monkeypatch.setattr(recovery_materialize, "_materialize_groups_from_source_locked", forbidden)
    lock_path = tmp_path / "heavy.lock"

    for groups, rates in (
        (((3, "gate_proj"),), {3: 16}),
        (
            tuple((layer, projection) for projection in recovery_materialize.GROUP_PROJECTIONS for layer in (2,)),
            {2: 16},
        ),
    ):
        with pytest.raises(ValueError, match=r"complete|\[3, 77\]"):
            materialize_groups_from_source(
                groups=groups,
                layer_code_bits=rates,
                heavy_lock_path=lock_path,
            )
    assert entered is False
    assert not lock_path.exists()


def test_e8p_worst_layer_map_allows_17_but_campaign_cli_caps_16() -> None:
    module = _load_wave1_cli_module()
    attribution = {
        "ranked_layers": [
            {"rank": rank, "layer": layer, "weighted_reconstruction_error": float(100 - rank)}
            for rank, layer in enumerate(range(3, 23), start=1)
        ]
    }

    assert module._e8p_worst_layer_code_bits(attribution, count=8) == {
        layer: 16 for layer in range(3, 11)
    }
    assert len(module._e8p_worst_layer_code_bits(attribution, count=16)) == 16
    assert len(module._e8p_worst_layer_code_bits(attribution, count=17)) == 17
    with pytest.raises(ValueError, match="at most 16"):
        module._validate_e8p_campaign_count(17)
    with pytest.raises(ValueError, match="selected worst-layer count"):
        module._validate_e8p_campaign_count(9, worst_layer_count=8)


def test_e8_importance_quantization_preserves_frozen_byte_result() -> None:
    source = np.array(
        [[0.18, -0.21, 0.16, -0.20, 0.19, -0.17, 0.22, -7.5]],
        dtype=np.float32,
    )
    importance = np.array(
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.001],
        dtype=np.float32,
    )

    recovered = quantize_weight_importance_aware(
        source, importance, group_size=8, code_bits=8
    )

    np.testing.assert_array_equal(recovered.codes, np.array([[56]], dtype=np.uint8))
    np.testing.assert_array_equal(
        recovered.scales.view(np.uint16), np.array([[17280]], dtype=np.uint16)
    )
    np.testing.assert_array_equal(recovered.codebook, e8_1bit_packed())


def test_schema_v2_manifest_is_published_by_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_publish = recovery_materialize.publish_file_transactionally
    publications: list[tuple[Path, Path, bool]] = []

    def recording_publish(source: str | Path, target: str | Path, **kwargs: object):
        source_path = Path(source)
        target_path = Path(target)
        if target_path.name == "conversion-manifest.json":
            publications.append(
                (
                    source_path,
                    target_path,
                    kwargs.get("expected_destination") is not None,
                )
            )
            assert source_path.name.startswith(".conversion-manifest.json.partial-")
            assert source_path.exists()
        return real_publish(source, target, **kwargs)

    monkeypatch.setattr(
        recovery_materialize, "publish_file_transactionally", recording_publish
    )
    manifest, _reads = _run_materialization(tmp_path, monkeypatch, resume=False)

    assert manifest["schema_version"] == 2
    assert len(publications) == 1
    assert publications[0][2] is False
    assert not list((tmp_path / "output").glob(".conversion-manifest.json.partial-*"))


def test_schema_v2_manifest_binds_exact_rate_policy_and_resume_rejects_drift(
    tmp_path: Path,
) -> None:
    source_lineage = {
        "source_model_id": MODEL_ID,
        "source_revision": REVISION,
        "source_config_sha256": "c" * 64,
        "source_index_sha256": "d" * 64,
        "source_profile": PROFILE,
    }
    source_verification = {
        "source_blob_inventory_sha256": "e" * 64,
        "routed_source_blob_inventory_sha256": "f" * 64,
        "shard_count": 1,
    }
    provenance = recovery_materialize._lever_provenance(
        "a" * 64,
        source_lineage=source_lineage,
        source_verification=source_verification,
    )
    groups = tuple((3, projection) for projection in recovery_materialize.GROUP_PROJECTIONS)
    records = [
        {
            "layer": 3,
            "projection": projection,
            "status": "materialized",
            "artifact_path": f"layer-00003-{projection}.safetensors",
            "artifact_bytes": 1,
            "artifact_sha256": f"{index + 1:x}" * 64,
            "source_lineage": source_lineage,
            "recovery_policy": {"rounding_objective": "selection_diagonal_hessian_weighted_squared_error"},
            "lever_provenance": provenance,
            "code_bits": 16,
            "codes_dtype": "uint16",
            "codebook_name": "quip_e8p",
            "codebook_sha256": E8P_PACKED_ABS_SHA256,
            "tensor_payload_bytes": 8,
            "zero_importance_policy": "source_rtn_e8p",
        }
        for index, projection in enumerate(recovery_materialize.GROUP_PROJECTIONS)
    ]
    manifest = build_recovery_conversion_manifest(
        groups=groups,
        layer_code_bits={3: 16},
        stats_manifest_sha256="a" * 64,
        expected_full_source_blob_inventory_sha256="e" * 64,
        expected_routed_source_blob_inventory_sha256="f" * 64,
        group_records=records,
        mixed_artifact={},
        source_lineage=source_lineage,
        source_verification=source_verification,
        accounting={},
        accepted_composite_audit_sha256="b" * 64,
    )
    assert manifest["schema_version"] == 2
    assert manifest["rate_policy"] == {
        "default_code_bits": 8,
        "layer_code_bits": {"3": 16},
        "complete_layer_rates_required": True,
    }
    manifest_path = tmp_path / "conversion-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="rate|requested recovery run"):
        recovery_materialize._load_prior_conversion_records(
            manifest_path=manifest_path,
            groups=groups,
            layer_code_bits={},
            stats_manifest_sha256="a" * 64,
            accepted_composite_audit_sha256="b" * 64,
        )


def test_worst8_and_worst16_accounting_uses_fresh_tensor_payloads(tmp_path: Path) -> None:
    def write_inventory(root: Path, upgraded_count: int) -> None:
        root.mkdir()
        for layer in range(3, 19):
            for projection in recovery_materialize.GROUP_PROJECTIONS:
                prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
                code_bits = 16 if layer < 3 + upgraded_count else 8
                save_file(
                    {
                        f"{prefix}.codes": np.zeros(
                            (2, 1, 1), dtype=np.uint16 if code_bits == 16 else np.uint8
                        ),
                        f"{prefix}.scales": np.zeros((2, 1, 1), dtype=np.float16),
                        "model.vq_codebook.e8": (
                            e8p_packed_abs_grid() if code_bits == 16 else e8_1bit_packed()
                        ),
                    },
                    root / f"layer-{layer:05d}-{projection}.safetensors",
                )

    worst8 = tmp_path / "worst8"
    worst16 = tmp_path / "worst16"
    write_inventory(worst8, 8)
    write_inventory(worst16, 16)

    accounting8 = recovery_materialize._routed_tensor_payload_accounting(worst8)
    accounting16 = recovery_materialize._routed_tensor_payload_accounting(worst16)
    codebook_bytes = 16 * 3 * 256 * np.dtype(np.uint32).itemsize
    assert accounting8 == {
        "routed_codes_scales_bytes": 8 * 3 * 8 + 8 * 3 * 6,
        "routed_codebook_bytes": codebook_bytes,
    }
    assert accounting16 == {
        "routed_codes_scales_bytes": 16 * 3 * 8,
        "routed_codebook_bytes": codebook_bytes,
    }


def test_selection_split_is_the_only_tuning_authority() -> None:
    evidence = validate_selection_split(_selection_rows())
    assert evidence == {
        "split": "selection",
        "prompt_count": 22,
        "position_count": 255,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }

    invalid = _selection_rows()
    invalid[0] = {**invalid[0], "split": "report", "tuning_eligible": False}
    with pytest.raises(ValueError, match="selection"):
        validate_selection_split(invalid)


def test_recovered_group_is_byte_contract_compatible_in_mixed_tree(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    mixed = tmp_path / "mixed"
    seed.mkdir()
    group3 = seed / "layer-00003-gate_proj.safetensors"
    group4 = seed / "layer-00004-gate_proj.safetensors"
    _artifact_group(group3, layer=3, projection="gate_proj", fill=1)
    _artifact_group(group4, layer=4, projection="gate_proj", fill=2)

    source = np.array(
        [
            [[0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1]],
            [[-0.2, 0.1, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    rewritten = materialize_recovery_group(
        source_weights=source,
        importance=np.ones((2, 8), dtype=np.float32),
        seed_group_path=group3,
        output_path=tmp_path / "rewritten.safetensors",
        layer=3,
        projection="gate_proj",
        group_size=8,
    )
    result = build_mixed_artifact_tree(
        seed_artifact_dir=seed,
        output_dir=mixed,
        replacements={group3.name: rewritten},
    )

    assert (mixed / group4.name).read_bytes() == group4.read_bytes()
    with safe_open(mixed / group3.name, framework="np") as recovered_handle, safe_open(
        group3, framework="np"
    ) as seed_handle:
        assert set(recovered_handle.keys()) == set(seed_handle.keys())
        for name in seed_handle.keys():
            assert recovered_handle.get_tensor(name).shape == seed_handle.get_tensor(name).shape
            assert recovered_handle.get_tensor(name).dtype == seed_handle.get_tensor(name).dtype
    assert result["replacement_count"] == 1
    assert result["inherited_group_count"] == 1


def test_conversion_manifest_records_recovery_lever_provenance(tmp_path: Path) -> None:
    seed = tmp_path / "seed.safetensors"
    _artifact_group(seed, layer=7, projection="down_proj", fill=3)
    source = np.array(
        [
            [[0.1, -0.2, 0.3, -0.4, 0.4, -0.3, 0.2, -0.1]],
            [[-0.1, 0.2, -0.3, 0.4, -0.4, 0.3, -0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    output = tmp_path / "recovered.safetensors"
    record = materialize_recovery_group(
        source_weights=source,
        importance=np.ones((2, 8), dtype=np.float32),
        seed_group_path=seed,
        output_path=output,
        layer=7,
        projection="down_proj",
        group_size=8,
        stats_manifest_sha256="a" * 64,
    )

    assert record["lever_provenance"] == {
        "lever": "selection_diagonal_hessian_importance_weighted_reround_v1",
        "scale_estimator": "importance_weighted_least_squares",
        "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
        "stats_manifest_sha256": "a" * 64,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    with safe_open(output, framework="np") as handle:
        quantization = json.loads((handle.metadata() or {})["quantization_config"])
    assert quantization["policy"]["recovery_lever"] == record["lever_provenance"]["lever"]
    source_lineage = {
        "source_model_id": MODEL_ID,
        "source_revision": REVISION,
        "source_config_sha256": "c" * 64,
        "source_index_sha256": "d" * 64,
        "source_profile": PROFILE,
    }
    source_verification = {
        "source_blob_inventory_sha256": "e" * 64,
        "routed_source_blob_inventory_sha256": "f" * 64,
        "shard_count": 1,
    }
    enriched_provenance = recovery_materialize._lever_provenance(
        "a" * 64,
        source_lineage=source_lineage,
        source_verification=source_verification,
    )
    canonical_record = {
        "layer": 7,
        "projection": "down_proj",
        "status": "materialized",
        "artifact_path": output.name,
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": _sha256(output),
        "source_lineage": source_lineage,
        "recovery_policy": quantization["policy"],
        "lever_provenance": enriched_provenance,
    }
    manifest = build_recovery_conversion_manifest(
        groups=((7, "down_proj"),),
        stats_manifest_sha256="a" * 64,
        expected_full_source_blob_inventory_sha256="e" * 64,
        expected_routed_source_blob_inventory_sha256="f" * 64,
        group_records=(canonical_record,),
        mixed_artifact={
            "seed_artifact_dir": str(tmp_path / "seed"),
            "seed_manifest_sha256": "b" * 64,
            "output_dir": str(tmp_path / "mixed"),
            "recovered_groups_dir": str(tmp_path / "recovered-groups"),
            "replacement_count": 1,
            "inherited_group_count": 224,
        },
        source_lineage=source_lineage,
        source_verification=source_verification,
        accounting={
            "non_routed_tensor_payload_bytes": 0,
            "logical_whole_model_tensor_payload_bytes": 0,
            "incremental_disk_bytes": output.stat().st_size,
            "logical_payload_limit_bytes": 1,
        },
        accepted_composite_audit_sha256="c" * 64,
    )
    assert manifest["record_type"] == "glm52_recovery_conversion_manifest"
    assert manifest["lever_provenance"] == enriched_provenance
    assert manifest["expected_full_source_blob_inventory_sha256"] == "e" * 64
    assert manifest["expected_routed_source_blob_inventory_sha256"] == "f" * 64


@pytest.mark.parametrize(
    "attack",
    (
        "schema",
        "status",
        "method",
        "source_authority",
        "duplicate",
        "missing",
        "extra_requested_expert",
        "path_escape",
        "symlink_escape",
        "noncanonical_filename",
        "sha256",
    ),
)
def test_stats_manifest_authenticates_requested_entries_before_npz_load(
    tmp_path: Path,
    attack: str,
) -> None:
    manifest = _stats_manifest(tmp_path)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    if attack == "schema":
        manifest["schema_version"] = 2
    elif attack == "status":
        manifest["status"] = "partial"
    elif attack == "method":
        manifest["method"] = "untrusted_second_moments"
    elif attack == "source_authority":
        source_authority = manifest["source_authority"]
        assert isinstance(source_authority, dict)
        source_authority["authenticated_source_teacher"] = False
    elif attack == "duplicate":
        entries.append(dict(entries[0]))
        manifest["entry_count"] = len(entries)
    elif attack == "missing":
        entries.pop()
        manifest["entry_count"] = len(entries)
    elif attack == "extra_requested_expert":
        extra_path = tmp_path / "layer-00003-gate_proj-expert-002.npz"
        extra_path.write_bytes((tmp_path / str(entries[0]["path"])).read_bytes())
        extra = dict(entries[0])
        extra.update({"expert": 2, "path": extra_path.name, "sha256": _sha256(extra_path)})
        entries.append(extra)
        manifest["entry_count"] = len(entries)
    elif attack == "path_escape":
        entries[0]["path"] = "../layer-00003-gate_proj-expert-000.npz"
    elif attack == "symlink_escape":
        canonical_path = tmp_path / str(entries[0]["path"])
        escaped_path = tmp_path.parent / f"{tmp_path.name}-escaped.npz"
        canonical_path.replace(escaped_path)
        canonical_path.symlink_to(escaped_path)
    elif attack == "noncanonical_filename":
        entries[0]["path"] = "renamed.npz"
    elif attack == "sha256":
        entries[0]["sha256"] = "0" * 64

    with pytest.raises(ValueError):
        recovery_materialize._load_authenticated_stats(
            stats_dir=tmp_path,
            stats_manifest=manifest,
            group_specs={(3, "gate_proj"): (2, 8)},
        )


def test_stats_authority_requires_exact_pinned_model_profile_and_revision(tmp_path: Path) -> None:
    for field, value in (
        ("model_id", "attacker/model"),
        ("profile", "glm52-lookalike"),
        ("revision", "0" * 40),
    ):
        stats_root = tmp_path / field
        stats_root.mkdir()
        manifest = _stats_manifest(stats_root)
        authority = manifest["source_authority"]
        assert isinstance(authority, dict)
        authority[field] = value
        with pytest.raises(ValueError, match="model|profile|revision|pinned"):
            recovery_materialize._load_authenticated_stats(
                stats_dir=stats_root,
                stats_manifest=manifest,
                group_specs={(3, "gate_proj"): (2, 8)},
            )


@pytest.mark.parametrize("mutated", ("config", "index"))
def test_materialization_binds_actual_config_and_index_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated: str,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    source_root = tmp_path / "hf/models--0xSero--glm-5.2-reap-504B-v2/snapshots" / REVISION
    path = source_root / ("config.json" if mutated == "config" else "model.safetensors.index.json")
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(
        ValueError,
        match=f"{mutated}.*SHA-256|{mutated}.*identity|lineage.*{mutated}",
    ):
        _run_materialization(tmp_path, monkeypatch, resume=False)


def test_materialization_uses_authenticated_source_blob_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest, reads = _run_materialization(tmp_path, monkeypatch, resume=False)

    assert reads.authenticated_source_opens == 1
    assert reads.verified_after_reads == 1
    assert reads.closed_inventories == 1


@pytest.mark.parametrize("inventory", ("full", "routed"))
def test_materialization_rejects_lookalike_source_tree_with_changed_blob_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: str,
) -> None:
    actual_full = (
        "9" * 64 if inventory == "full" else FULL_SOURCE_BLOB_INVENTORY_SHA256
    )
    actual_routed = (
        "9" * 64 if inventory == "routed" else ROUTED_SOURCE_BLOB_INVENTORY_SHA256
    )

    with pytest.raises(ValueError, match=f"{inventory}.*source blob inventory.*external"):
        _run_materialization(
            tmp_path,
            monkeypatch,
            resume=False,
            actual_full_inventory_sha256=actual_full,
            actual_routed_inventory_sha256=actual_routed,
        )


def test_stats_npz_same_directory_symlink_is_rejected(tmp_path: Path) -> None:
    manifest = _stats_manifest(tmp_path)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    path = tmp_path / str(entries[0]["path"])
    target = tmp_path / "alternate.npz"
    path.replace(target)
    path.symlink_to(target.name)

    with pytest.raises(ValueError, match="symlink|no-follow|regular"):
        recovery_materialize._load_authenticated_stats(
            stats_dir=tmp_path,
            stats_manifest=manifest,
            group_specs={(3, "gate_proj"): (2, 8)},
        )


def test_resume_rejects_copied_provenance_without_prior_artifact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_reads = _run_materialization(tmp_path, monkeypatch, resume=False)
    assert len(first_reads) == 6
    prior_sha256 = str(first["groups"][0]["artifact_sha256"])
    target = tmp_path / "output/recovered-groups/layer-00003-gate_proj.safetensors"
    forged_source = np.full((2, 1, 8), 0.45, dtype=np.float32)
    materialize_recovery_group(
        source_weights=forged_source,
        importance=np.ones((2, 8), dtype=np.float32),
        seed_group_path=tmp_path / "seed/layer-00003-gate_proj.safetensors",
        output_path=target,
        layer=3,
        projection="gate_proj",
        group_size=8,
        stats_manifest_sha256=str(first["stats_manifest_sha256"]),
    )
    forged_sha256 = _sha256(target)
    assert forged_sha256 != prior_sha256

    repaired, repair_reads = _run_materialization(tmp_path, monkeypatch, resume=True)
    assert len(repair_reads) == 2
    assert repaired["groups"][0]["status"] == "materialized"
    assert repaired["groups"][0]["artifact_sha256"] != forged_sha256

    reattested, reattest_reads = _run_materialization(tmp_path, monkeypatch, resume=True)
    assert reattest_reads == []
    assert reattested["groups"][0]["status"] == "resumed"
    body = dict(reattested)
    body.pop("manifest_body_sha256")
    assert reattested["manifest_body_sha256"] == recovery_materialize._canonical_sha256(body)


def test_resume_rejects_identical_path_replacement_between_prior_check_and_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    target = tmp_path / "output/recovered-groups/layer-00003-gate_proj.safetensors"
    replacement = tmp_path / "identical-replacement.safetensors"
    replacement.write_bytes(target.read_bytes())
    original = recovery_materialize._validate_resumable_group
    swapped = False

    def replace_then_validate(**kwargs: object) -> dict[str, object]:
        nonlocal swapped
        os.replace(replacement, target)
        swapped = True
        return original(**kwargs)

    monkeypatch.setattr(
        recovery_materialize,
        "_validate_resumable_group",
        replace_then_validate,
    )
    real_read_policy = recovery_materialize._read_recovery_policy

    def require_snapshot_policy(path: Path) -> dict[str, str]:
        if path == target:
            pytest.fail("resume policy was read from the mutable target path")
        return real_read_policy(path)

    monkeypatch.setattr(
        recovery_materialize,
        "_read_recovery_policy",
        require_snapshot_policy,
    )

    with pytest.raises(ValueError, match="retarget|identity|snapshot|replaced"):
        _run_materialization(tmp_path, monkeypatch, resume=True)
    assert swapped is True


def test_resume_rematerializes_interrupted_output_without_completed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    (tmp_path / "output/conversion-manifest.json").unlink()

    manifest, reads = _run_materialization(tmp_path, monkeypatch, resume=True)

    assert len(reads) == 6
    assert manifest["groups"][0]["status"] == "materialized"


def test_overwrite_unpublishes_completed_manifest_before_first_group_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    manifest_path = tmp_path / "output/conversion-manifest.json"
    target = tmp_path / "output/recovered-groups/layer-00003-gate_proj.safetensors"
    prior_target_sha256 = _sha256(target)
    original = recovery_materialize.materialize_recovery_group
    mutations = 0

    def mutate_then_interrupt(**kwargs: object) -> dict[str, object]:
        nonlocal mutations
        record = original(**kwargs)
        output_path = Path(str(kwargs["output_path"]))
        output_path.write_bytes(output_path.read_bytes() + b"interrupted")
        mutations += 1
        raise RuntimeError("injected interruption after first replacement")

    monkeypatch.setattr(
        recovery_materialize,
        "materialize_recovery_group",
        mutate_then_interrupt,
    )

    with pytest.raises(RuntimeError, match="injected interruption"):
        _run_materialization(tmp_path, monkeypatch, resume=False)

    assert mutations == 1
    assert _sha256(target) != prior_target_sha256
    assert not manifest_path.exists()
    forensics = list(
        (tmp_path / "output/.publication-forensics").glob(
            "layer-00003-gate_proj.safetensors-*"
        )
    )
    assert len(forensics) == 1
    assert _sha256(forensics[0]) == prior_target_sha256
    assert not list(
        (tmp_path / "output/recovered-groups").glob(
            ".layer-00003-gate_proj.safetensors.partial-*"
        )
    )


def test_resume_rematerializes_when_authenticated_stats_evidence_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    stats_root = tmp_path / "stats"
    manifest_path = stats_root / "glm52-recovery-stats-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["entries"][0]
    stats_path = stats_root / entry["path"]
    np.savez(
        stats_path,
        sum_x2=np.full(8, 2.0, dtype=np.float32),
        mean_second_moment=np.full(8, 2.0, dtype=np.float32),
        routing_weighted_importance=np.full(8, 2.0, dtype=np.float32),
        router_score_weighted_importance=np.full(8, 2.0, dtype=np.float32),
    )
    entry["sha256"] = _sha256(stats_path)
    _write_stats_manifest(stats_root, manifest)

    rematerialized, reads = _run_materialization(tmp_path, monkeypatch, resume=True)

    assert len(reads) == 6
    assert rematerialized["groups"][0]["status"] == "materialized"


def test_resume_fails_closed_on_corrupt_prior_manifest_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    manifest_path = tmp_path / "output/conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest body"):
        _run_materialization(tmp_path, monkeypatch, resume=True)


@pytest.mark.parametrize("attack", ("codebook", "quantization_policy"))
def test_resume_fails_closed_on_prior_attested_noncanonical_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _run_materialization(tmp_path, monkeypatch, resume=False)
    target = tmp_path / "output/recovered-groups/layer-00003-gate_proj.safetensors"
    with safe_open(target, framework="np") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    if attack == "codebook":
        tensors["model.vq_codebook.e8"] = np.zeros_like(tensors["model.vq_codebook.e8"])
    else:
        quantization = json.loads(metadata["quantization_config"])
        quantization["policy"]["rounding_objective"] = "copied_metadata_only"
        metadata["quantization_config"] = json.dumps(quantization, sort_keys=True)
    save_file(tensors, target, metadata=metadata)

    manifest_path = tmp_path / "output/conversion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["groups"][0]["artifact_sha256"] = _sha256(target)
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = recovery_materialize._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="codebook|quantization"):
        _run_materialization(tmp_path, monkeypatch, resume=True)
