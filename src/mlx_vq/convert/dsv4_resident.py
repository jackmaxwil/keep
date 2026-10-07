"""Byte-exact resident packaging for the pinned DeepSeek-V4-Flash release.

The checkpoint already carries its resident precision policy: block-FP8 affine
weights plus higher-precision embeddings, heads, routers, norms, and scalars.
This module only repacks those bytes; routed experts remain owned by the VQ
artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mlx_vq.convert import glm52_non_vq as raw_pack
from mlx_vq.convert.stream_convert import SafetensorsIndex, load_safetensors_index
from mlx_vq.io.source_safetensors import (
    SafetensorsTensorHeader,
    read_safetensors_file_header,
)
from mlx_vq.models.profiles import get_profile
from ramp.models.deepseek_v4_flash_adapter import _map_checkpoint_name
from ramp.models.deepseek_v4_policy import validate_deepseek_v4_config

MANIFEST_NAME = "resident-manifest.json"
INDEX_NAME = raw_pack.SAFETENSORS_INDEX_NAME
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
PROFILE_NAME = "deepseek-v4-flash-0731"
RECORD_TYPE = "dsv4_resident_package_manifest_v1"
READY_STATUS = "dsv4_resident_package_ready"
COPY_MODE = "raw_safetensors_byte_ranges_v1"
SELECTION_POLICY = "dsv4_all_non_routed_release_precision_v1"
FP8_WEIGHT_DTYPE = "F8_E4M3"
FP8_SCALE_DTYPE = "F8_E8M0"
FP8_BLOCK = (128, 128)
RETAINED_DTYPES = frozenset({"BF16", "F32", FP8_WEIGHT_DTYPE, FP8_SCALE_DTYPE, "I64"})

EXPECTED_SOURCE_TENSORS = 72_317
EXPECTED_RETAINED_TENSORS = 1_661
EXPECTED_LOGICAL_PARAMETER_TENSORS = 1_271
EXPECTED_SCALE_TENSORS = 390
EXPECTED_LOGICAL_PARAMETERS = 7_827_675_070
EXPECTED_STORAGE_ELEMENTS = 7_828_059_838
EXPECTED_PAYLOAD_BYTES = 9_441_141_496
EXPECTED_BACKBONE_TENSORS = 1_564
EXPECTED_MTP_TENSORS = 97
EXPECTED_ROUTED_TENSORS = 70_656
EXPECTED_ROUTED_PAYLOAD_BYTES = 157_437_394_944
EXPECTED_DTYPE_COUNTS = {
    "BF16": 445,
    "F32": 433,
    FP8_WEIGHT_DTYPE: 390,
    FP8_SCALE_DTYPE: 390,
    "I64": 3,
}


@dataclass(frozen=True)
class _ResidentPlan:
    tensors: tuple[raw_pack._PlannedTensor, ...]
    shards: tuple[raw_pack._PlannedShard, ...]
    source_inventory_sha256: str
    plan_sha256: str
    logical_parameter_count: int
    storage_element_count: int
    tensor_payload_bytes: int
    logical_parameter_tensor_count: int
    scale_tensor_count: int
    backbone_tensor_count: int
    mtp_tensor_count: int
    excluded_routed_tensor_count: int
    excluded_routed_payload_bytes: int
    dtype_tensor_counts: Mapping[str, int]


@dataclass(frozen=True)
class Dsv4ResidentPackResult:
    manifest: Mapping[str, object]
    manifest_sha256: str
    written_shards: int
    largest_copy_buffer_bytes: int

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.manifest)
        payload.update(
            {
                "manifest_sha256": self.manifest_sha256,
                "written_shards": self.written_shards,
                "largest_copy_buffer_bytes": self.largest_copy_buffer_bytes,
            }
        )
        return payload


@dataclass(frozen=True)
class Dsv4ResidentPackageAudit:
    artifact_dir: str
    manifest_path: str
    manifest_sha256: str
    retained_tensor_count: int
    logical_parameter_count: int
    tensor_payload_bytes: int
    shard_count: int
    artifact_tree_bytes: int
    package_set_sha256: str
    checks: Mapping[str, bool]

    @property
    def audit_pass(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["audit_pass"] = self.audit_pass
        return payload


def _sha256(path: Path) -> str:
    return raw_pack._sha256_file(path)


def _weight_for_scale(name: str) -> str:
    return f"{name.removesuffix('.scale')}.weight"


def _is_routed(name: str) -> bool:
    return ".ffn.experts." in name


def _source_target(name: str) -> str | None:
    return _map_checkpoint_name(
        _weight_for_scale(name) if name.endswith(".scale") else name
    )


def _validate_authority(
    source_root: Path,
    index: SafetensorsIndex,
    *,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    enforce_pinned_source: bool,
) -> None:
    config_path = source_root / "config.json"
    index_path = source_root / INDEX_NAME
    if _sha256(config_path) != config_sha256:
        raise ValueError("config SHA-256 does not match source bytes")
    if _sha256(index_path) != index_sha256:
        raise ValueError("index SHA-256 does not match source bytes")
    if load_safetensors_index(index_path) != index:
        raise ValueError("source index argument does not match source bytes")
    if not enforce_pinned_source:
        return
    expected = (MODEL_ID, REVISION, CONFIG_SHA256, INDEX_SHA256)
    if (model_id, revision, config_sha256, index_sha256) != expected:
        raise ValueError(
            "production DSV4 resident packaging requires the pinned identity"
        )
    profile = get_profile(PROFILE_NAME)
    if profile.hf_model_id != model_id or profile.revision != revision:
        raise ValueError("DSV4 profile lineage does not match the pinned source")
    completion = json.loads((source_root / "_KEEP_DOWNLOAD_COMPLETE.json").read_text())
    if completion.get("model_id") != model_id or completion.get("revision") != revision:
        raise ValueError(
            "source download completion record does not match the pinned identity"
        )
    config = json.loads(config_path.read_text())
    failures = validate_deepseek_v4_config(config)
    if failures:
        raise ValueError(f"pinned DSV4 config policy failed: {failures}")
    quantization = config["quantization_config"]
    if quantization.get("scale_fmt") != "ue8m0" or quantization.get(
        "weight_block_size"
    ) != [128, 128]:
        raise ValueError("pinned DSV4 resident block-scale contract drifted")


def _validate_fp8_pair(
    name: str,
    header: SafetensorsTensorHeader,
    *,
    index: SafetensorsIndex,
    headers: Mapping[str, Mapping[str, SafetensorsTensorHeader]],
) -> None:
    if header.dtype == FP8_WEIGHT_DTYPE:
        scale_name = f"{name.removesuffix('.weight')}.scale"
        if not name.endswith(".weight") or scale_name not in index.weight_map:
            raise ValueError(
                f"resident FP8 weight {name!r} has no block-scale companion"
            )
        scale = headers[index.weight_map[scale_name]][scale_name]
        expected_shape = tuple(
            (dim + block - 1) // block
            for dim, block in zip(header.shape, FP8_BLOCK, strict=True)
        )
        if (
            len(header.shape) != 2
            or scale.dtype != FP8_SCALE_DTYPE
            or scale.shape != expected_shape
        ):
            raise ValueError(
                f"resident FP8 weight {name!r} has invalid 128x128 scale geometry"
            )
    elif header.dtype == FP8_SCALE_DTYPE:
        weight_name = _weight_for_scale(name)
        if not name.endswith(".scale") or weight_name not in index.weight_map:
            raise ValueError(
                f"resident block scale {name!r} has no FP8 weight companion"
            )
        weight = headers[index.weight_map[weight_name]][weight_name]
        if weight.dtype != FP8_WEIGHT_DTYPE:
            raise ValueError(
                f"resident block scale {name!r} does not belong to an FP8 weight"
            )


def _plan_package(
    source_root: Path,
    index: SafetensorsIndex,
    *,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
) -> tuple[_ResidentPlan, Mapping[str, Path], Mapping[str, int]]:
    if max_shard_payload_bytes <= 0:
        raise ValueError("max_shard_payload_bytes must be positive")
    indexed_by_shard: dict[str, set[str]] = {name: set() for name in index.shards}
    for name, shard in index.weight_map.items():
        if shard not in indexed_by_shard:
            raise ValueError(f"tensor {name!r} points to unknown shard {shard!r}")
        indexed_by_shard[shard].add(name)

    source_paths: dict[str, Path] = {}
    source_offsets: dict[str, int] = {}
    headers: dict[str, Mapping[str, SafetensorsTensorHeader]] = {}
    for shard in index.shards:
        path = raw_pack._safe_source_path(source_root, shard)
        file_header = read_safetensors_file_header(path)
        raw_pack._validate_physical_extent(path, file_header)
        if set(file_header.tensors) != indexed_by_shard[shard]:
            raise ValueError(f"source shard {shard!r} index/header inventory mismatch")
        source_paths[shard] = path
        source_offsets[shard] = file_header.payload_offset
        headers[shard] = file_header.tensors

    retained: list[tuple[str, str, SafetensorsTensorHeader]] = []
    descriptors: list[dict[str, object]] = []
    routed_count = 0
    routed_bytes = 0
    logical_targets: dict[str, str] = {}
    for name, shard in sorted(index.weight_map.items()):
        header = headers[shard][name]
        payload_bytes = raw_pack._tensor_payload_bytes(header)
        if _is_routed(name):
            routed_count += 1
            routed_bytes += payload_bytes
            continue
        if header.dtype not in RETAINED_DTYPES:
            raise ValueError(
                f"resident tensor {name!r} has unsupported dtype {header.dtype!r}"
            )
        target = _source_target(name)
        if target is None:
            raise ValueError(
                f"non-routed source tensor {name!r} has no production binder target"
            )
        _validate_fp8_pair(name, header, index=index, headers=headers)
        if not name.endswith(".scale"):
            previous = logical_targets.setdefault(target, name)
            if previous != name:
                raise ValueError(
                    f"resident sources {previous!r} and {name!r} duplicate target {target!r}"
                )
        retained.append((name, shard, header))
        descriptors.append(
            {
                "name": name,
                "target": target,
                "dtype": header.dtype,
                "shape": list(header.shape),
                "payload_bytes": payload_bytes,
                "source_shard": shard,
                "source_offsets": list(header.data_offsets),
            }
        )
    if not retained:
        raise ValueError("source contains no DSV4 resident tensors")

    groups: list[list[tuple[str, str, SafetensorsTensorHeader]]] = []
    current: list[tuple[str, str, SafetensorsTensorHeader]] = []
    current_bytes = 0
    for item in retained:
        payload_bytes = raw_pack._tensor_payload_bytes(item[2])
        if current and current_bytes + payload_bytes > max_shard_payload_bytes:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(item)
        current_bytes += payload_bytes
    if current:
        groups.append(current)

    planned_tensors: list[raw_pack._PlannedTensor] = []
    planned_shards: list[raw_pack._PlannedShard] = []
    for shard_number, items in enumerate(groups, start=1):
        filename = f"model-{shard_number:05d}-of-{len(groups):05d}.safetensors"
        cursor = 0
        shard_tensors: list[raw_pack._PlannedTensor] = []
        for name, source_shard, header in items:
            payload_bytes = raw_pack._tensor_payload_bytes(header)
            tensor = raw_pack._PlannedTensor(
                name=name,
                dtype=header.dtype,
                shape=header.shape,
                parameter_count=header.element_count,
                payload_bytes=payload_bytes,
                source_shard=source_shard,
                source_offsets=header.data_offsets,
                output_shard=filename,
                output_offsets=(cursor, cursor + payload_bytes),
            )
            cursor += payload_bytes
            planned_tensors.append(tensor)
            shard_tensors.append(tensor)
        planned_shards.append(raw_pack._PlannedShard(filename, tuple(shard_tensors)))

    scale_count = sum(name.endswith(".scale") for name, _, _ in retained)
    logical_count = len(retained) - scale_count
    dtype_counts = Counter(header.dtype for _, _, header in retained)
    storage_elements = sum(header.element_count for _, _, header in retained)
    logical_parameters = sum(
        header.element_count
        for name, _, header in retained
        if not name.endswith(".scale")
    )
    payload_bytes = sum(
        raw_pack._tensor_payload_bytes(header) for _, _, header in retained
    )
    plan_basis = {
        "schema_version": 1,
        "copy_mode": COPY_MODE,
        "selection_policy": SELECTION_POLICY,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "max_shard_payload_bytes": max_shard_payload_bytes,
        "tensors": [tensor.plan_dict() for tensor in planned_tensors],
    }
    return (
        _ResidentPlan(
            tensors=tuple(planned_tensors),
            shards=tuple(planned_shards),
            source_inventory_sha256=raw_pack._canonical_sha256(descriptors),
            plan_sha256=raw_pack._canonical_sha256(plan_basis),
            logical_parameter_count=logical_parameters,
            storage_element_count=storage_elements,
            tensor_payload_bytes=payload_bytes,
            logical_parameter_tensor_count=logical_count,
            scale_tensor_count=scale_count,
            backbone_tensor_count=sum(
                not name.startswith("mtp.") for name, _, _ in retained
            ),
            mtp_tensor_count=sum(name.startswith("mtp.") for name, _, _ in retained),
            excluded_routed_tensor_count=routed_count,
            excluded_routed_payload_bytes=routed_bytes,
            dtype_tensor_counts=dict(sorted(dtype_counts.items())),
        ),
        source_paths,
        source_offsets,
    )


def _validate_pinned_plan(
    plan: _ResidentPlan, index: SafetensorsIndex, *, enforce: bool
) -> None:
    if not enforce:
        return
    actual = {
        "source_tensors": len(index.weight_map),
        "retained_tensors": len(plan.tensors),
        "logical_parameter_tensors": plan.logical_parameter_tensor_count,
        "scale_tensors": plan.scale_tensor_count,
        "logical_parameters": plan.logical_parameter_count,
        "storage_elements": plan.storage_element_count,
        "payload_bytes": plan.tensor_payload_bytes,
        "backbone_tensors": plan.backbone_tensor_count,
        "mtp_tensors": plan.mtp_tensor_count,
        "routed_tensors": plan.excluded_routed_tensor_count,
        "routed_payload_bytes": plan.excluded_routed_payload_bytes,
        "dtype_tensor_counts": dict(plan.dtype_tensor_counts),
    }
    expected = {
        "source_tensors": EXPECTED_SOURCE_TENSORS,
        "retained_tensors": EXPECTED_RETAINED_TENSORS,
        "logical_parameter_tensors": EXPECTED_LOGICAL_PARAMETER_TENSORS,
        "scale_tensors": EXPECTED_SCALE_TENSORS,
        "logical_parameters": EXPECTED_LOGICAL_PARAMETERS,
        "storage_elements": EXPECTED_STORAGE_ELEMENTS,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "backbone_tensors": EXPECTED_BACKBONE_TENSORS,
        "mtp_tensors": EXPECTED_MTP_TENSORS,
        "routed_tensors": EXPECTED_ROUTED_TENSORS,
        "routed_payload_bytes": EXPECTED_ROUTED_PAYLOAD_BYTES,
        "dtype_tensor_counts": EXPECTED_DTYPE_COUNTS,
    }
    if actual != expected:
        raise ValueError(
            f"pinned DSV4 resident inventory drifted; expected={expected} actual={actual}"
        )


def _manifest(
    plan: _ResidentPlan,
    *,
    artifact_dir: Path,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    package_index_sha256: str,
    shard_records: Sequence[Mapping[str, object]],
    payload_hashes: Mapping[str, str],
    production_ready: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": RECORD_TYPE,
        "pack_status": READY_STATUS,
        "production_ready": production_ready,
        "artifact_dir": str(artifact_dir),
        "copy_mode": COPY_MODE,
        "selection_policy": SELECTION_POLICY,
        "profile": PROFILE_NAME if production_ready else None,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source_inventory_sha256": plan.source_inventory_sha256,
        "plan_sha256": plan.plan_sha256,
        "quantization": {
            "affine_weight_dtype": FP8_WEIGHT_DTYPE,
            "block_geometry": list(FP8_BLOCK),
            "block_scale_dtype": FP8_SCALE_DTYPE,
            "high_precision_policy": "preserve_source_dtype",
        },
        "retained_tensor_count": len(plan.tensors),
        "logical_parameter_tensor_count": plan.logical_parameter_tensor_count,
        "scale_tensor_count": plan.scale_tensor_count,
        "logical_parameter_count": plan.logical_parameter_count,
        "storage_element_count": plan.storage_element_count,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "backbone_tensor_count": plan.backbone_tensor_count,
        "mtp_tensor_count": plan.mtp_tensor_count,
        "dtype_tensor_counts": dict(plan.dtype_tensor_counts),
        "excluded_routed_tensor_count": plan.excluded_routed_tensor_count,
        "excluded_routed_payload_bytes": plan.excluded_routed_payload_bytes,
        "deliberate_exclusions": [
            "backbone and mtp routed-expert .weight/.scale tensors owned by the VQ artifact"
        ],
        "duplicate_resident_tensor_count": 0,
        "missing_resident_tensor_count": 0,
        "routed_resident_tensor_count": 0,
        "max_shard_payload_bytes": max_shard_payload_bytes,
        "shard_count": len(plan.shards),
        "index_path": INDEX_NAME,
        "package_index_sha256": package_index_sha256,
        "package_set_sha256": raw_pack._package_set_sha256(
            package_index_sha256=package_index_sha256,
            shard_records=shard_records,
        ),
        "shards": [dict(record) for record in shard_records],
        "tensors": [
            tensor.manifest_dict(payload_sha256=payload_hashes[tensor.name])
            for tensor in plan.tensors
        ],
    }


def pack_dsv4_resident_safetensors(
    source_dir: str | Path,
    index: SafetensorsIndex,
    output_dir: str | Path,
    *,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    enforce_pinned_source: bool = True,
) -> Dsv4ResidentPackResult:
    """Repack every non-routed source tensor without changing its bytes."""

    source_root = Path(source_dir)
    output_root = Path(output_dir)
    _validate_authority(
        source_root,
        index,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        enforce_pinned_source=enforce_pinned_source,
    )
    if output_root.exists():
        raise ValueError(
            f"refusing to overwrite existing resident artifact {output_root}"
        )
    plan, source_paths, source_offsets = _plan_package(
        source_root,
        index,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        max_shard_payload_bytes=max_shard_payload_bytes,
    )
    _validate_pinned_plan(plan, index, enforce=enforce_pinned_source)

    staging = output_root.with_name(f"{output_root.name}.partial-{uuid4().hex}")
    staging.mkdir(parents=True)
    shard_records: list[dict[str, object]] = []
    payload_hashes: dict[str, str] = {}
    largest_buffer = 0
    for shard in plan.shards:
        record, hashes, largest = raw_pack._write_shard(
            staging / shard.filename,
            shard,
            source_paths=source_paths,
            source_payload_offsets=source_offsets,
        )
        shard_records.append(record)
        payload_hashes.update(hashes)
        largest_buffer = max(largest_buffer, largest)
    package_index = {
        "metadata": {"total_size": plan.tensor_payload_bytes},
        "weight_map": {tensor.name: tensor.output_shard for tensor in plan.tensors},
    }
    raw_pack._write_json_atomic(staging / INDEX_NAME, package_index)
    package_index_sha256 = _sha256(staging / INDEX_NAME)
    manifest = _manifest(
        plan,
        artifact_dir=output_root,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        max_shard_payload_bytes=max_shard_payload_bytes,
        package_index_sha256=package_index_sha256,
        shard_records=shard_records,
        payload_hashes=payload_hashes,
        production_ready=enforce_pinned_source,
    )
    raw_pack._write_json_atomic(staging / MANIFEST_NAME, manifest)
    try:
        audit_dsv4_resident_package(
            staging,
            source_dir=source_root,
            source_index=index,
            expected_model_id=model_id,
            expected_revision=revision,
            expected_config_sha256=config_sha256,
            expected_index_sha256=index_sha256,
            enforce_pinned_source=enforce_pinned_source,
            _expected_artifact_dir=output_root,
        )
    except Exception:
        shutil.rmtree(staging)
        raise
    staging.replace(output_root)
    return Dsv4ResidentPackResult(
        manifest=manifest,
        manifest_sha256=_sha256(output_root / MANIFEST_NAME),
        written_shards=len(plan.shards),
        largest_copy_buffer_bytes=largest_buffer,
    )


def audit_dsv4_resident_package(
    output_dir: str | Path,
    *,
    source_dir: str | Path,
    source_index: SafetensorsIndex,
    expected_model_id: str,
    expected_revision: str,
    expected_config_sha256: str,
    expected_index_sha256: str,
    enforce_pinned_source: bool = True,
    _expected_artifact_dir: str | Path | None = None,
) -> Dsv4ResidentPackageAudit:
    """Recompute inventory, hashes, physical extents, and source byte identity."""

    root = Path(output_dir)
    source_root = Path(source_dir)
    _validate_authority(
        source_root,
        source_index,
        model_id=expected_model_id,
        revision=expected_revision,
        config_sha256=expected_config_sha256,
        index_sha256=expected_index_sha256,
        enforce_pinned_source=enforce_pinned_source,
    )
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    max_bytes = manifest.get("max_shard_payload_bytes")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("resident manifest has an invalid max shard size")
    plan, source_paths, source_offsets = _plan_package(
        source_root,
        source_index,
        model_id=expected_model_id,
        revision=expected_revision,
        config_sha256=expected_config_sha256,
        index_sha256=expected_index_sha256,
        max_shard_payload_bytes=max_bytes,
    )
    _validate_pinned_plan(plan, source_index, enforce=enforce_pinned_source)
    expected_manifest = {
        "schema_version": 1,
        "record_type": RECORD_TYPE,
        "pack_status": READY_STATUS,
        "production_ready": enforce_pinned_source,
        "artifact_dir": str(
            root if _expected_artifact_dir is None else Path(_expected_artifact_dir)
        ),
        "copy_mode": COPY_MODE,
        "selection_policy": SELECTION_POLICY,
        "profile": PROFILE_NAME if enforce_pinned_source else None,
        "model_id": expected_model_id,
        "source_revision": expected_revision,
        "config_sha256": expected_config_sha256,
        "index_sha256": expected_index_sha256,
        "source_inventory_sha256": plan.source_inventory_sha256,
        "plan_sha256": plan.plan_sha256,
        "quantization": {
            "affine_weight_dtype": FP8_WEIGHT_DTYPE,
            "block_geometry": list(FP8_BLOCK),
            "block_scale_dtype": FP8_SCALE_DTYPE,
            "high_precision_policy": "preserve_source_dtype",
        },
        "retained_tensor_count": len(plan.tensors),
        "logical_parameter_tensor_count": plan.logical_parameter_tensor_count,
        "scale_tensor_count": plan.scale_tensor_count,
        "logical_parameter_count": plan.logical_parameter_count,
        "storage_element_count": plan.storage_element_count,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "backbone_tensor_count": plan.backbone_tensor_count,
        "mtp_tensor_count": plan.mtp_tensor_count,
        "dtype_tensor_counts": dict(plan.dtype_tensor_counts),
        "excluded_routed_tensor_count": plan.excluded_routed_tensor_count,
        "excluded_routed_payload_bytes": plan.excluded_routed_payload_bytes,
        "deliberate_exclusions": [
            "backbone and mtp routed-expert .weight/.scale tensors owned by the VQ artifact"
        ],
        "duplicate_resident_tensor_count": 0,
        "missing_resident_tensor_count": 0,
        "routed_resident_tensor_count": 0,
        "shard_count": len(plan.shards),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"resident manifest {key} mismatch: expected {expected!r}, found {manifest.get(key)!r}"
            )

    raw_shards = manifest.get("shards")
    raw_tensors = manifest.get("tensors")
    if not isinstance(raw_shards, list) or not isinstance(raw_tensors, list):
        raise TypeError("resident manifest shards/tensors must be arrays")
    shard_names = [
        str(record.get("filename")) for record in raw_shards if isinstance(record, dict)
    ]
    expected_shards = [shard.filename for shard in plan.shards]
    if shard_names != expected_shards or len(set(shard_names)) != len(shard_names):
        raise ValueError(
            "resident manifest shard inventory is missing, duplicated, or reordered"
        )
    raw_pack._strict_package_tree(
        root, expected_files={MANIFEST_NAME, INDEX_NAME, *shard_names}
    )

    output_index = load_safetensors_index(root / INDEX_NAME)
    expected_weight_map = {tensor.name: tensor.output_shard for tensor in plan.tensors}
    if (
        output_index.weight_map != expected_weight_map
        or output_index.metadata.get("total_size") != plan.tensor_payload_bytes
    ):
        raise ValueError("resident output index does not match the planned inventory")
    if _sha256(root / INDEX_NAME) != manifest.get("package_index_sha256"):
        raise ValueError("resident output index hash mismatch")
    expected_tensor_records = [tensor.plan_dict() for tensor in plan.tensors]
    manifest_tensor_records = [
        {key: value for key, value in record.items() if key != "payload_sha256"}
        for record in raw_tensors
        if isinstance(record, dict)
    ]
    if manifest_tensor_records != expected_tensor_records:
        raise ValueError(
            "resident manifest tensor inventory is missing, duplicated, or drifted"
        )
    payload_hashes = {
        str(record["name"]): str(record["payload_sha256"])
        for record in raw_tensors
        if isinstance(record, dict)
    }
    if len(payload_hashes) != len(plan.tensors):
        raise ValueError("resident manifest tensor hashes are missing or duplicated")

    actual_records: list[dict[str, object]] = []
    for shard, recorded in zip(plan.shards, raw_shards, strict=True):
        assert isinstance(recorded, dict)
        path = root / shard.filename
        file_sha256, _ = raw_pack._audit_output_shard(
            path,
            tensors=shard.tensors,
            expected_payload_hashes=payload_hashes,
            source_paths=source_paths,
            source_payload_offsets=source_offsets,
        )
        actual = raw_pack._shard_record(shard, path=path, file_sha256=file_sha256)
        if actual != recorded:
            raise ValueError(
                f"resident shard {shard.filename!r} hash or extent mismatch"
            )
        actual_records.append(actual)
    package_set_sha256 = raw_pack._package_set_sha256(
        package_index_sha256=str(manifest["package_index_sha256"]),
        shard_records=actual_records,
    )
    if package_set_sha256 != manifest.get("package_set_sha256"):
        raise ValueError("resident package-set hash mismatch")
    files = {MANIFEST_NAME, INDEX_NAME, *shard_names}
    return Dsv4ResidentPackageAudit(
        artifact_dir=str(root),
        manifest_path=str(root / MANIFEST_NAME),
        manifest_sha256=_sha256(root / MANIFEST_NAME),
        retained_tensor_count=len(plan.tensors),
        logical_parameter_count=plan.logical_parameter_count,
        tensor_payload_bytes=plan.tensor_payload_bytes,
        shard_count=len(plan.shards),
        artifact_tree_bytes=sum((root / name).stat().st_size for name in files),
        package_set_sha256=package_set_sha256,
        checks={
            "lineage": True,
            "source_inventory": True,
            "selection": True,
            "strict_tree": True,
            "index": True,
            "physical_extents": True,
            "payload_hashes": True,
            "source_byte_identity": True,
            "accounting": True,
        },
    )


@contextmanager
def _heavy_job_lock(path: Path):
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pinned DSV4 non-routed resident package"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-shard-payload-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = datetime.now(UTC).isoformat()
    payload: dict[str, object]
    try:
        forbidden = [
            name
            for name in ("GLM_MLX_WIRED_LIMIT_GB", "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB")
            if os.environ.get(name) is not None
        ]
        if forbidden:
            raise ValueError(
                f"custom wired-limit variables must be absent: {forbidden}"
            )
        source = Path(args.source_dir)
        index = load_safetensors_index(source / INDEX_NAME)
        with _heavy_job_lock(Path(args.heavy_lock_path).resolve()):
            result = pack_dsv4_resident_safetensors(
                source,
                index,
                args.output_dir,
                model_id=MODEL_ID,
                revision=REVISION,
                config_sha256=CONFIG_SHA256,
                index_sha256=INDEX_SHA256,
                max_shard_payload_bytes=args.max_shard_payload_bytes,
            )
            audit = audit_dsv4_resident_package(
                args.output_dir,
                source_dir=source,
                source_index=index,
                expected_model_id=MODEL_ID,
                expected_revision=REVISION,
                expected_config_sha256=CONFIG_SHA256,
                expected_index_sha256=INDEX_SHA256,
            )
        payload = {
            "schema_version": 1,
            "record_type": "dsv4_resident_package_run_v1",
            "status": "complete",
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "heavy_job_lock": str(Path(args.heavy_lock_path).resolve()),
            "wired_limit_variables_absent": True,
            "pack": result.to_dict(),
            "audit": audit.to_dict(),
        }
        code = 0
    except Exception as error:  # noqa: BLE001 - CLI must persist terminal failure evidence.
        payload = {
            "schema_version": 1,
            "record_type": "dsv4_resident_package_run_v1",
            "status": "failed",
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        code = 1
    raw_pack._write_json_atomic(Path(args.output_json), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
