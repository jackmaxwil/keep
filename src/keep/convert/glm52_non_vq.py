"""Deterministic raw-byte packaging for GLM-5.2 non-routed weights.

The source checkpoint uses ModelOpt NVFP4 for routed experts, while the rest of
the main model is already BF16/F32.  This module copies only those BF16/F32
main-model tensors into compact safetensors shards without materializing a
tensor as NumPy or MLX data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from keep.convert.glm52_reap import (
    GLM52_REAP_CONFIG_SHA256,
    GLM52_REAP_EXPECTED_BUNDLES,
    GLM52_REAP_INDEX_SHA256,
    GLM52_REAP_PROFILE_NAME,
    audit_glm52_reap_source_index,
)
from keep.convert.stream_convert import SafetensorsIndex, load_safetensors_index
from keep.io.source_safetensors import (
    SafetensorsFileHeader,
    SafetensorsTensorHeader,
    read_safetensors_file_header,
)
from ramp.models.profiles import ModelProfile, get_profile, load_profile


NON_VQ_MANIFEST_NAME = "non-vq-manifest.json"
SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
NON_VQ_RECORD_TYPE = "glm52_non_vq_package_manifest"
NON_VQ_READY_STATUS = "glm52_non_vq_package_ready"
NON_VQ_FIXTURE_READY_STATUS = "glm52_non_vq_fixture_ready"
NON_VQ_COPY_MODE = "raw_safetensors_byte_ranges_v1"
NON_VQ_SELECTION_POLICY = "main_model_non_routed_bf16_f32_v1"
COPY_BUFFER_BYTES = 1024 * 1024
NON_VQ_PARTIAL_STATUS = "glm52_non_vq_package_partial"
NON_VQ_FIXTURE_PARTIAL_STATUS = "glm52_non_vq_fixture_partial"
NON_VQ_PINNED_SOURCE_AUTHORITY = "pinned_huggingface_lfs_v1"
NON_VQ_FIXTURE_SOURCE_AUTHORITY = "unverified_test_fixture"
NON_VQ_PINNED_BLOB_INVENTORY_SHA256 = (
    "ace08e87dcbce3a22249e54196a27c0045992f8d5b8ca07f342899fa7a53fc8d"
)
NON_VQ_EXPECTED_SOURCE_TENSORS = 154_433
NON_VQ_EXPECTED_RETAINED_TENSORS = 1_194
NON_VQ_EXPECTED_PARAMETERS = 18_560_731_704
NON_VQ_EXPECTED_PAYLOAD_BYTES = 37_121_488_608
NON_VQ_EXPECTED_ROUTED_TENSORS = GLM52_REAP_EXPECTED_BUNDLES * 4
NON_VQ_EXPECTED_MTP_TENSORS = 2_039
NON_VQ_EXPECTED_DTYPE_COUNTS = {"BF16": 1_119, "F32": 75}

_MODEL_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_SHARD_NAME_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
_ROUTED_NAMESPACE = ".mlp.experts."
_VQ_RUNTIME_NAMESPACE = ".mlp.switch_mlp."

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E8M0": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_RETAINED_DTYPES = frozenset({"BF16", "F32"})


@dataclass(frozen=True)
class _PlannedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    parameter_count: int
    payload_bytes: int
    source_shard: str
    source_offsets: tuple[int, int]
    output_shard: str
    output_offsets: tuple[int, int]

    def manifest_dict(self, *, payload_sha256: str) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "parameter_count": self.parameter_count,
            "payload_bytes": self.payload_bytes,
            "source_shard": self.source_shard,
            "source_offsets": list(self.source_offsets),
            "output_shard": self.output_shard,
            "output_offsets": list(self.output_offsets),
            "payload_sha256": payload_sha256,
        }

    def plan_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "parameter_count": self.parameter_count,
            "payload_bytes": self.payload_bytes,
            "source_shard": self.source_shard,
            "source_offsets": list(self.source_offsets),
            "output_shard": self.output_shard,
            "output_offsets": list(self.output_offsets),
        }


@dataclass(frozen=True)
class _PlannedShard:
    filename: str
    tensors: tuple[_PlannedTensor, ...]

    @property
    def tensor_payload_bytes(self) -> int:
        return sum(tensor.payload_bytes for tensor in self.tensors)


@dataclass(frozen=True)
class _PackagePlan:
    tensors: tuple[_PlannedTensor, ...]
    shards: tuple[_PlannedShard, ...]
    source_inventory_sha256: str
    plan_sha256: str
    parameter_count: int
    tensor_payload_bytes: int
    excluded_routed_tensor_count: int
    excluded_mtp_tensor_count: int
    excluded_runtime_vq_tensor_count: int


@dataclass(frozen=True)
class Glm52NonVQPackResult:
    manifest: Mapping[str, object]
    manifest_sha256: str
    written_shards: int
    reused_shards: int
    resume_verified: bool
    largest_copy_buffer_bytes: int

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.manifest)
        payload.update(
            {
                "manifest_sha256": self.manifest_sha256,
                "written_shards": self.written_shards,
                "reused_shards": self.reused_shards,
                "resume_verified": self.resume_verified,
                "largest_copy_buffer_bytes": self.largest_copy_buffer_bytes,
            }
        )
        return payload


@dataclass(frozen=True)
class Glm52NonVQPackageAudit:
    artifact_dir: str
    manifest_path: str
    manifest_sha256: str
    retained_tensor_count: int
    parameter_count: int
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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_file_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{uuid4().hex}")
    try:
        with partial.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_bytes_atomic(path, _json_file_bytes(value))


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_source_path(source_root: Path, shard_name: str) -> Path:
    relative = Path(shard_name)
    if not shard_name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source shard path {shard_name!r}")
    path = source_root / relative
    if not path.is_file():
        raise ValueError(f"source shard is missing: {path}")
    return path


def _tensor_payload_bytes(header: SafetensorsTensorHeader) -> int:
    start, end = header.data_offsets
    if start < 0 or end < start:
        raise ValueError(f"invalid safetensors data offsets: {(start, end)}")
    dtype_bytes = _DTYPE_BYTES.get(header.dtype)
    if dtype_bytes is None:
        raise ValueError(f"unsupported safetensors dtype {header.dtype!r}")
    expected = header.element_count * dtype_bytes
    actual = end - start
    if actual != expected:
        raise ValueError(
            f"safetensors tensor byte extent must be {expected}, found {actual}"
        )
    return actual


def _validate_physical_extent(path: Path, header: SafetensorsFileHeader) -> None:
    expected_file_bytes = header.payload_offset + header.declared_payload_bytes
    actual_file_bytes = path.stat().st_size
    if expected_file_bytes != actual_file_bytes:
        raise ValueError(
            f"{path} physical extent bytes do not match file size "
            f"({expected_file_bytes} != {actual_file_bytes})"
        )
    cursor = 0
    for name, tensor in sorted(
        header.tensors.items(), key=lambda item: item[1].data_offsets
    ):
        start, end = tensor.data_offsets
        if start != cursor:
            raise ValueError(
                f"{path} tensor {name!r} has overlapping or non-contiguous extent "
                f"starting at {start}, expected {cursor}"
            )
        _tensor_payload_bytes(tensor)
        cursor = end
    if cursor != header.declared_payload_bytes:
        raise ValueError(
            f"{path} tensor extents end at {cursor}, expected "
            f"{header.declared_payload_bytes} bytes"
        )


def _is_main_non_routed_tensor(name: str, profile: ModelProfile) -> bool:
    layer_match = _MODEL_LAYER_RE.match(name)
    if layer_match is not None and int(layer_match.group("layer")) >= profile.num_layers:
        return False
    if _ROUTED_NAMESPACE in name or _VQ_RUNTIME_NAMESPACE in name:
        return False
    return True


def _validate_profile_lineage(
    profile: ModelProfile,
    *,
    model_id: str,
    revision: str,
) -> None:
    if profile.hf_model_id != model_id:
        raise ValueError(
            f"profile model_id must be {model_id!r}, found {profile.hf_model_id!r}"
        )
    if profile.revision != revision:
        raise ValueError(
            f"profile revision must be {revision!r}, found {profile.revision!r}"
        )


def _ready_status(enforce_pinned_source: bool) -> str:
    return NON_VQ_READY_STATUS if enforce_pinned_source else NON_VQ_FIXTURE_READY_STATUS


def _partial_status(enforce_pinned_source: bool) -> str:
    return (
        NON_VQ_PARTIAL_STATUS
        if enforce_pinned_source
        else NON_VQ_FIXTURE_PARTIAL_STATUS
    )


def _source_authority(enforce_pinned_source: bool) -> str:
    return (
        NON_VQ_PINNED_SOURCE_AUTHORITY
        if enforce_pinned_source
        else NON_VQ_FIXTURE_SOURCE_AUTHORITY
    )


def _validate_pinned_blob_inventory(
    source_root: Path,
    *,
    index: SafetensorsIndex,
    revision: str,
) -> str:
    if source_root.name != revision or source_root.parent.name != "snapshots":
        raise ValueError(
            "production source_dir must be the exact pinned Hugging Face snapshot"
        )
    model_cache_root = source_root.parent.parent.resolve()
    expected_blob_root = model_cache_root / "blobs"
    blob_inventory: dict[str, str] = {}
    for shard_name in index.shards:
        shard_path = source_root / shard_name
        if not shard_path.is_symlink():
            raise ValueError(
                f"production source shard {shard_name!r} must be a Hugging Face "
                "snapshot symlink"
            )
        blob_path = shard_path.resolve(strict=True)
        if blob_path.parent != expected_blob_root:
            raise ValueError(
                f"production source shard {shard_name!r} does not resolve inside "
                "the pinned model's blob store"
            )
        blob_id = blob_path.name
        if re.fullmatch(r"[0-9a-f]{64}", blob_id) is None:
            raise ValueError(
                f"production source shard {shard_name!r} has an invalid LFS blob identity"
            )
        blob_inventory[shard_name] = blob_id
    inventory_sha256 = _canonical_sha256(blob_inventory)
    if inventory_sha256 != NON_VQ_PINNED_BLOB_INVENTORY_SHA256:
        raise ValueError(
            "production source shard LFS blob inventory does not match the pinned "
            "GLM52 revision oracle"
        )
    return inventory_sha256


def _validate_pinned_authority(
    *,
    source_dir: str | Path,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    config_path: str | Path | None,
    index_path: str | Path | None,
    index: SafetensorsIndex,
    enforce_pinned_source: bool,
) -> None:
    """Make the production-ready status depend on the pinned source oracle."""

    _validate_profile_lineage(profile, model_id=model_id, revision=revision)
    if not enforce_pinned_source:
        return
    canonical_profile = get_profile(GLM52_REAP_PROFILE_NAME)
    if profile != canonical_profile:
        raise ValueError(
            "production GLM52 non-VQ packaging requires the canonical immutable "
            f"profile {GLM52_REAP_PROFILE_NAME!r}"
        )
    if config_sha256 != GLM52_REAP_CONFIG_SHA256:
        raise ValueError(
            "production config SHA-256 does not match the pinned GLM52 source"
        )
    if index_sha256 != GLM52_REAP_INDEX_SHA256:
        raise ValueError(
            "production index SHA-256 does not match the pinned GLM52 source"
        )
    if config_path is None or index_path is None:
        raise ValueError(
            "production GLM52 packaging requires config_path and index_path so "
            "their pinned hashes can be recomputed"
        )
    config_file = Path(config_path)
    index_file = Path(index_path)
    if _sha256_file(config_file) != config_sha256:
        raise ValueError("config SHA-256 does not match config_path bytes")
    if _sha256_file(index_file) != index_sha256:
        raise ValueError("index SHA-256 does not match index_path bytes")
    indexed_file = load_safetensors_index(index_file)
    if indexed_file != index:
        raise ValueError("in-memory source index does not match pinned index_path bytes")
    _validate_pinned_blob_inventory(
        Path(source_dir),
        index=index,
        revision=revision,
    )
    config = _load_json_object(config_file)
    source_audit = audit_glm52_reap_source_index(
        config=config,
        index=index,
        model_id=model_id,
        revision=revision,
        profile=profile,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
    )
    if not source_audit.source_checks_pass:
        blockers = source_audit.to_json_dict().get("audit_blockers", [])
        raise ValueError(
            "pinned GLM52 source contract failed before non-VQ packaging: "
            + ", ".join(str(item) for item in blockers[:10])
        )


def _validate_pinned_plan(
    plan: _PackagePlan,
    *,
    index: SafetensorsIndex,
    enforce_pinned_source: bool,
) -> None:
    if not enforce_pinned_source:
        return
    actual = {
        "source_tensors": len(index.weight_map),
        "retained_tensors": len(plan.tensors),
        "parameter_count": plan.parameter_count,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "excluded_routed_tensors": plan.excluded_routed_tensor_count,
        "excluded_mtp_tensors": plan.excluded_mtp_tensor_count,
        "excluded_runtime_vq_tensors": plan.excluded_runtime_vq_tensor_count,
        "dtype_tensor_counts": {
            dtype: sum(tensor.dtype == dtype for tensor in plan.tensors)
            for dtype in sorted(_RETAINED_DTYPES)
            if any(tensor.dtype == dtype for tensor in plan.tensors)
        },
    }
    expected = {
        "source_tensors": NON_VQ_EXPECTED_SOURCE_TENSORS,
        "retained_tensors": NON_VQ_EXPECTED_RETAINED_TENSORS,
        "parameter_count": NON_VQ_EXPECTED_PARAMETERS,
        "tensor_payload_bytes": NON_VQ_EXPECTED_PAYLOAD_BYTES,
        "excluded_routed_tensors": NON_VQ_EXPECTED_ROUTED_TENSORS,
        "excluded_mtp_tensors": NON_VQ_EXPECTED_MTP_TENSORS,
        "excluded_runtime_vq_tensors": 0,
        "dtype_tensor_counts": NON_VQ_EXPECTED_DTYPE_COUNTS,
    }
    if actual != expected:
        raise ValueError(
            "pinned GLM52 non-VQ inventory is incomplete or drifted; "
            f"expected={expected} actual={actual}"
        )


@dataclass(frozen=True)
class _SourceInventory:
    shard_paths: Mapping[str, Path]
    shard_payload_offsets: Mapping[str, int]
    retained: tuple[tuple[str, str, SafetensorsTensorHeader], ...]
    source_descriptors: tuple[Mapping[str, object], ...]
    excluded_routed_tensor_count: int
    excluded_mtp_tensor_count: int
    excluded_runtime_vq_tensor_count: int


def _read_source_inventory(
    source_root: Path,
    index: SafetensorsIndex,
    *,
    profile: ModelProfile,
) -> _SourceInventory:
    if not source_root.is_dir():
        raise ValueError(f"source directory does not exist: {source_root}")
    shard_paths: dict[str, Path] = {}
    payload_offsets: dict[str, int] = {}
    retained: list[tuple[str, str, SafetensorsTensorHeader]] = []
    source_descriptors: list[dict[str, object]] = []
    excluded_routed = 0
    excluded_mtp = 0
    excluded_runtime_vq = 0
    indexed_names: dict[str, set[str]] = {name: set() for name in index.shards}
    for tensor_name, shard_name in index.weight_map.items():
        if shard_name not in indexed_names:
            raise ValueError(
                f"tensor {tensor_name!r} points to unknown shard {shard_name!r}"
            )
        indexed_names[shard_name].add(tensor_name)
    for shard_name in index.shards:
        path = _safe_source_path(source_root, shard_name)
        header = read_safetensors_file_header(path)
        _validate_physical_extent(path, header)
        expected_names = indexed_names[shard_name]
        actual_names = set(header.tensors)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unindexed = sorted(actual_names - expected_names)
            raise ValueError(
                f"source shard {shard_name} index/header mismatch; "
                f"missing={missing[:10]} unindexed={unindexed[:10]}"
            )
        shard_paths[shard_name] = path
        payload_offsets[shard_name] = header.payload_offset
        for name in sorted(expected_names):
            tensor = header.tensors[name]
            layer_match = _MODEL_LAYER_RE.match(name)
            is_mtp = (
                layer_match is not None
                and int(layer_match.group("layer")) >= profile.num_layers
            )
            if is_mtp:
                excluded_mtp += 1
                continue
            if _ROUTED_NAMESPACE in name:
                excluded_routed += 1
                continue
            if _VQ_RUNTIME_NAMESPACE in name:
                excluded_runtime_vq += 1
                continue
            if tensor.dtype not in _RETAINED_DTYPES:
                raise ValueError(
                    f"main non-routed tensor {name!r} must be BF16 or F32, "
                    f"found {tensor.dtype!r}"
                )
            payload_bytes = _tensor_payload_bytes(tensor)
            retained.append((name, shard_name, tensor))
            source_descriptors.append(
                {
                    "name": name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "payload_bytes": payload_bytes,
                    "source_shard": shard_name,
                    "source_offsets": list(tensor.data_offsets),
                }
            )
        # Do not retain every parsed descriptor from the 154k-tensor source.
        # Only the 1,194 copied descriptors and one payload offset per shard are
        # needed after this physical-integrity pass.
        del header
    retained.sort(key=lambda item: item[0])
    source_descriptors.sort(key=lambda item: str(item["name"]))
    return _SourceInventory(
        shard_paths=shard_paths,
        shard_payload_offsets=payload_offsets,
        retained=tuple(retained),
        source_descriptors=tuple(source_descriptors),
        excluded_routed_tensor_count=excluded_routed,
        excluded_mtp_tensor_count=excluded_mtp,
        excluded_runtime_vq_tensor_count=excluded_runtime_vq,
    )


def _plan_package(
    source_root: Path,
    index: SafetensorsIndex,
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
) -> tuple[_PackagePlan, Mapping[str, Path], Mapping[str, int]]:
    if max_shard_payload_bytes <= 0:
        raise ValueError("max_shard_payload_bytes must be positive")
    _validate_profile_lineage(profile, model_id=model_id, revision=revision)
    inventory = _read_source_inventory(source_root, index, profile=profile)
    retained_source = list(inventory.retained)
    if not retained_source:
        raise ValueError("source index contains no main-model non-routed BF16/F32 tensors")

    grouped: list[list[tuple[str, str, SafetensorsTensorHeader]]] = []
    current: list[tuple[str, str, SafetensorsTensorHeader]] = []
    current_bytes = 0
    for item in retained_source:
        payload_bytes = _tensor_payload_bytes(item[2])
        if current and current_bytes + payload_bytes > max_shard_payload_bytes:
            grouped.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += payload_bytes
    if current:
        grouped.append(current)

    shard_count = len(grouped)
    tensors: list[_PlannedTensor] = []
    shards: list[_PlannedShard] = []
    for shard_index, items in enumerate(grouped, start=1):
        filename = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        cursor = 0
        shard_tensors: list[_PlannedTensor] = []
        for name, source_shard, header in items:
            payload_bytes = _tensor_payload_bytes(header)
            planned = _PlannedTensor(
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
            tensors.append(planned)
            shard_tensors.append(planned)
        shards.append(_PlannedShard(filename=filename, tensors=tuple(shard_tensors)))

    plan_basis = {
        "schema_version": 1,
        "copy_mode": NON_VQ_COPY_MODE,
        "selection_policy": NON_VQ_SELECTION_POLICY,
        "profile": profile.name,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "max_shard_payload_bytes": max_shard_payload_bytes,
        "tensors": [tensor.plan_dict() for tensor in tensors],
    }
    return (
        _PackagePlan(
            tensors=tuple(tensors),
            shards=tuple(shards),
            source_inventory_sha256=_canonical_sha256(inventory.source_descriptors),
            plan_sha256=_canonical_sha256(plan_basis),
            parameter_count=sum(tensor.parameter_count for tensor in tensors),
            tensor_payload_bytes=sum(tensor.payload_bytes for tensor in tensors),
            excluded_routed_tensor_count=inventory.excluded_routed_tensor_count,
            excluded_mtp_tensor_count=inventory.excluded_mtp_tensor_count,
            excluded_runtime_vq_tensor_count=(
                inventory.excluded_runtime_vq_tensor_count
            ),
        ),
        inventory.shard_paths,
        inventory.shard_payload_offsets,
    )


def _safetensors_header_bytes(tensors: Sequence[_PlannedTensor]) -> bytes:
    header: dict[str, object] = {}
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": list(tensor.output_offsets),
        }
    encoded = _canonical_json_bytes(header)
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def _write_shard(
    output_path: Path,
    shard: _PlannedShard,
    *,
    source_paths: Mapping[str, Path],
    source_payload_offsets: Mapping[str, int],
) -> tuple[dict[str, object], dict[str, str], int]:
    prefix = _safetensors_header_bytes(shard.tensors)
    partial = output_path.with_name(f"{output_path.name}.partial-{uuid4().hex}")
    file_digest = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    largest_copy_buffer = 0
    buffer = bytearray(COPY_BUFFER_BYTES)
    try:
        with partial.open("xb") as target:
            target.write(prefix)
            file_digest.update(prefix)
            for tensor in shard.tensors:
                source_path = source_paths[tensor.source_shard]
                payload_offset = source_payload_offsets[tensor.source_shard]
                remaining = tensor.payload_bytes
                tensor_digest = hashlib.sha256()
                with source_path.open("rb") as source:
                    source.seek(payload_offset + tensor.source_offsets[0])
                    while remaining:
                        requested = min(remaining, COPY_BUFFER_BYTES)
                        view = memoryview(buffer)[:requested]
                        read = source.readinto(view)
                        if read is None or read <= 0:
                            raise ValueError(
                                f"source tensor {tensor.name!r} is truncated while copying"
                            )
                        target.write(view[:read])
                        file_digest.update(view[:read])
                        tensor_digest.update(view[:read])
                        largest_copy_buffer = max(largest_copy_buffer, read)
                        remaining -= read
                tensor_digests[tensor.name] = tensor_digest.hexdigest()
            target.flush()
            os.fsync(target.fileno())
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)

    record = _shard_record(
        shard,
        path=output_path,
        file_sha256=file_digest.hexdigest(),
    )
    return record, tensor_digests, largest_copy_buffer


def _shard_record(
    shard: _PlannedShard,
    *,
    path: Path,
    file_sha256: str,
) -> dict[str, object]:
    tensor_inventory = [tensor.plan_dict() for tensor in shard.tensors]
    return {
        "filename": shard.filename,
        "tensor_count": len(shard.tensors),
        "tensor_inventory_sha256": _canonical_sha256(tensor_inventory),
        "tensor_payload_bytes": shard.tensor_payload_bytes,
        "file_bytes": path.stat().st_size,
        "file_sha256": file_sha256,
    }


def _package_set_sha256(
    *,
    package_index_sha256: str,
    shard_records: Sequence[Mapping[str, object]],
) -> str:
    return _canonical_sha256(
        {
            "package_index_sha256": package_index_sha256,
            "shards": [
                {
                    "filename": record["filename"],
                    "tensor_count": record["tensor_count"],
                    "tensor_inventory_sha256": record["tensor_inventory_sha256"],
                    "tensor_payload_bytes": record["tensor_payload_bytes"],
                    "file_bytes": record["file_bytes"],
                    "file_sha256": record["file_sha256"],
                }
                for record in sorted(
                    shard_records, key=lambda item: str(item["filename"])
                )
            ],
        }
    )


def _ready_manifest(
    plan: _PackagePlan,
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    package_index_sha256: str,
    shard_records: Sequence[Mapping[str, object]],
    payload_hashes: Mapping[str, str],
    enforce_pinned_source: bool,
) -> dict[str, object]:
    dtype_counts = {
        dtype: sum(tensor.dtype == dtype for tensor in plan.tensors)
        for dtype in sorted(_RETAINED_DTYPES)
        if any(tensor.dtype == dtype for tensor in plan.tensors)
    }
    return {
        "schema_version": 1,
        "record_type": NON_VQ_RECORD_TYPE,
        "pack_status": _ready_status(enforce_pinned_source),
        "production_ready": enforce_pinned_source,
        "source_authority": _source_authority(enforce_pinned_source),
        "source_blob_inventory_sha256": (
            NON_VQ_PINNED_BLOB_INVENTORY_SHA256
            if enforce_pinned_source
            else None
        ),
        "copy_mode": NON_VQ_COPY_MODE,
        "selection_policy": NON_VQ_SELECTION_POLICY,
        "profile": profile.name,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source_inventory_sha256": plan.source_inventory_sha256,
        "plan_sha256": plan.plan_sha256,
        "retained_tensor_count": len(plan.tensors),
        "parameter_count": plan.parameter_count,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "dtype_tensor_counts": dtype_counts,
        "excluded_routed_tensor_count": plan.excluded_routed_tensor_count,
        "excluded_mtp_tensor_count": plan.excluded_mtp_tensor_count,
        "excluded_runtime_vq_tensor_count": plan.excluded_runtime_vq_tensor_count,
        "max_shard_payload_bytes": max_shard_payload_bytes,
        "shard_count": len(plan.shards),
        "index_path": SAFETENSORS_INDEX_NAME,
        "package_index_sha256": package_index_sha256,
        "package_set_sha256": _package_set_sha256(
            package_index_sha256=package_index_sha256,
            shard_records=shard_records,
        ),
        "shards": [dict(record) for record in shard_records],
        "tensors": [
            tensor.manifest_dict(payload_sha256=payload_hashes[tensor.name])
            for tensor in plan.tensors
        ],
    }


def _partial_manifest(
    plan: _PackagePlan,
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    completed_shards: Sequence[Mapping[str, object]],
    completed_payload_hashes: Mapping[str, str],
    enforce_pinned_source: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": NON_VQ_RECORD_TYPE,
        "pack_status": _partial_status(enforce_pinned_source),
        "production_ready": False,
        "source_authority": _source_authority(enforce_pinned_source),
        "source_blob_inventory_sha256": (
            NON_VQ_PINNED_BLOB_INVENTORY_SHA256
            if enforce_pinned_source
            else None
        ),
        "copy_mode": NON_VQ_COPY_MODE,
        "selection_policy": NON_VQ_SELECTION_POLICY,
        "profile": profile.name,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source_inventory_sha256": plan.source_inventory_sha256,
        "plan_sha256": plan.plan_sha256,
        "retained_tensor_count": len(plan.tensors),
        "parameter_count": plan.parameter_count,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "max_shard_payload_bytes": max_shard_payload_bytes,
        "shard_count": len(plan.shards),
        "planned_shards": [shard.filename for shard in plan.shards],
        "completed_shards": [dict(record) for record in completed_shards],
        "completed_payload_hashes": dict(sorted(completed_payload_hashes.items())),
    }


def _validate_partial_request(
    manifest: Mapping[str, object],
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    enforce_pinned_source: bool,
) -> None:
    expected = {
        "schema_version": 1,
        "record_type": NON_VQ_RECORD_TYPE,
        "pack_status": _partial_status(enforce_pinned_source),
        "production_ready": False,
        "source_authority": _source_authority(enforce_pinned_source),
        "source_blob_inventory_sha256": (
            NON_VQ_PINNED_BLOB_INVENTORY_SHA256
            if enforce_pinned_source
            else None
        ),
        "copy_mode": NON_VQ_COPY_MODE,
        "selection_policy": NON_VQ_SELECTION_POLICY,
        "profile": profile.name,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "max_shard_payload_bytes": max_shard_payload_bytes,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"partial package {key} plan/lineage mismatch: expected {value!r}, "
                f"found {manifest.get(key)!r}"
            )


def _resume_partial_shards(
    root: Path,
    *,
    manifest: Mapping[str, object],
    plan: _PackagePlan,
    source_paths: Mapping[str, Path],
    source_payload_offsets: Mapping[str, int],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    if manifest.get("source_inventory_sha256") != plan.source_inventory_sha256:
        raise ValueError("partial package source inventory plan hash mismatch")
    if manifest.get("plan_sha256") != plan.plan_sha256:
        raise ValueError("partial package plan SHA-256 mismatch")
    if manifest.get("planned_shards") != [shard.filename for shard in plan.shards]:
        raise ValueError("partial package planned shard names mismatch")

    raw_completed = manifest.get("completed_shards")
    raw_hashes = manifest.get("completed_payload_hashes")
    if not isinstance(raw_completed, list) or not isinstance(raw_hashes, dict):
        raise ValueError("partial package completion state is invalid")
    recorded_by_name: dict[str, dict[str, object]] = {}
    for raw in raw_completed:
        if not isinstance(raw, dict):
            raise ValueError("partial package shard completion record is invalid")
        filename = str(raw.get("filename", ""))
        if filename in recorded_by_name:
            raise ValueError("partial package contains duplicate completed shards")
        recorded_by_name[filename] = dict(raw)
    recorded_hashes = {str(key): str(value) for key, value in raw_hashes.items()}

    planned_by_name = {shard.filename: shard for shard in plan.shards}
    atomic_targets = {
        NON_VQ_MANIFEST_NAME,
        SAFETENSORS_INDEX_NAME,
        *planned_by_name,
    }
    removable_partials: list[Path] = []
    existing_shards: dict[str, Path] = {}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink() or entry.is_dir():
            raise ValueError(f"partial package tree contains unexpected entry {relative!r}")
        if relative == NON_VQ_MANIFEST_NAME:
            continue
        if relative == SAFETENSORS_INDEX_NAME:
            removable_partials.append(entry)
            continue
        is_atomic_partial = any(
            re.fullmatch(
                rf"{re.escape(target)}\.partial-[0-9a-f]{{32}}",
                entry.name,
            )
            is not None
            for target in atomic_targets
        )
        if entry.parent == root and is_atomic_partial:
            removable_partials.append(entry)
            continue
        if relative not in planned_by_name:
            raise ValueError(
                f"partial package tree contains unexpected or unmanifested file {relative!r}"
            )
        existing_shards[relative] = entry
    missing_recorded = set(recorded_by_name) - set(existing_shards)
    if missing_recorded:
        raise ValueError(
            f"partial package recorded shard(s) are missing: {sorted(missing_recorded)}"
        )

    completed_records: list[dict[str, object]] = []
    payload_hashes: dict[str, str] = {}
    for shard in plan.shards:
        path = existing_shards.get(shard.filename)
        if path is None:
            continue
        file_sha256, shard_hashes = _audit_output_shard(
            path,
            tensors=shard.tensors,
            expected_payload_hashes=None,
            source_paths=source_paths,
            source_payload_offsets=source_payload_offsets,
        )
        record = _shard_record(shard, path=path, file_sha256=file_sha256)
        recorded = recorded_by_name.get(shard.filename)
        if recorded is not None and record != recorded:
            raise ValueError(
                f"partial package shard {shard.filename} is corrupt or not byte identical"
            )
        for name, digest in shard_hashes.items():
            recorded_digest = recorded_hashes.get(name)
            if recorded is not None and recorded_digest != digest:
                raise ValueError(
                    f"partial package tensor {name!r} SHA-256 byte identity mismatch"
                )
        completed_records.append(record)
        payload_hashes.update(shard_hashes)

    # Mutate only after every persisted lineage, plan, and completed shard has
    # been verified. These are recognized crash remnants, never arbitrary files.
    for path in removable_partials:
        path.unlink(missing_ok=True)
    return completed_records, payload_hashes


def _ensure_clean_output(output_root: Path, *, resume: bool) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output path is not a directory: {output_root}")
    if not output_root.exists():
        output_root.mkdir(parents=True)
        return
    entries = tuple(output_root.iterdir())
    if entries and not resume:
        raise ValueError(
            f"output directory is not empty; use resume only for a verified package: "
            f"{output_root}"
        )


def _restore_executor_resume_manifest(
    output_root: Path,
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    enforce_pinned_source: bool,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Restore the manifest invalidated by the build executor for reruns.

    Resumable artifact ops have their completion manifest moved aside before the
    subprocess starts.  This packer needs that verified state to distinguish a
    complete no-op from an interrupted shard set. A partial marker is safe to
    restore after its shallow lineage check. A ready marker is returned to the
    caller and is not republished until the full source-relative audit passes.
    """

    if not output_root.is_dir():
        return None, None
    manifest_path = output_root / NON_VQ_MANIFEST_NAME
    previous = manifest_path.with_name(f"{manifest_path.name}.resume-previous.json")
    if not previous.exists():
        return None, None
    if previous.is_symlink() or not previous.is_file():
        raise ValueError("executor resume manifest must be a regular local file")
    if manifest_path.exists():
        raise ValueError("both current and executor resume manifests exist")
    state = _load_json_object(previous)
    expected = {
        "schema_version": 1,
        "record_type": NON_VQ_RECORD_TYPE,
        "production_ready": (
            state.get("pack_status") == _ready_status(enforce_pinned_source)
            and enforce_pinned_source
        ),
        "source_authority": _source_authority(enforce_pinned_source),
        "source_blob_inventory_sha256": (
            NON_VQ_PINNED_BLOB_INVENTORY_SHA256
            if enforce_pinned_source
            else None
        ),
        "copy_mode": NON_VQ_COPY_MODE,
        "selection_policy": NON_VQ_SELECTION_POLICY,
        "profile": profile.name,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "max_shard_payload_bytes": max_shard_payload_bytes,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"executor resume manifest {key} plan/lineage mismatch: "
                f"expected {value!r}, found {state.get(key)!r}"
            )
    ready_status = _ready_status(enforce_pinned_source)
    partial_status = _partial_status(enforce_pinned_source)
    if state.get("pack_status") not in {partial_status, ready_status}:
        raise ValueError("executor resume manifest has an invalid pack status")
    if state.get("pack_status") == ready_status:
        return previous, state
    previous.replace(manifest_path)
    return None, state


def pack_glm52_non_vq_safetensors(
    source_dir: str | Path,
    index: SafetensorsIndex,
    output_dir: str | Path,
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    max_shard_payload_bytes: int,
    resume: bool = False,
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
    enforce_pinned_source: bool = True,
) -> Glm52NonVQPackResult:
    """Copy main-model non-routed BF16/F32 payloads into compact shards."""

    source_root = Path(source_dir)
    output_root = Path(output_dir)
    _validate_pinned_authority(
        source_dir=source_root,
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        config_path=config_path,
        index_path=index_path,
        index=index,
        enforce_pinned_source=enforce_pinned_source,
    )
    pending_ready_manifest: Path | None = None
    pending_ready_state: dict[str, Any] | None = None
    if resume:
        pending_ready_manifest, pending_ready_state = _restore_executor_resume_manifest(
            output_root,
            profile=profile,
            model_id=model_id,
            revision=revision,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            max_shard_payload_bytes=max_shard_payload_bytes,
            enforce_pinned_source=enforce_pinned_source,
        )
    manifest_path = output_root / NON_VQ_MANIFEST_NAME
    if pending_ready_manifest is not None:
        assert pending_ready_state is not None
        audit = audit_glm52_non_vq_package(
            output_root,
            source_dir=source_root,
            source_index=index,
            profile=profile,
            expected_model_id=model_id,
            expected_revision=revision,
            expected_config_sha256=config_sha256,
            expected_index_sha256=index_sha256,
            config_path=config_path,
            index_path=index_path,
            enforce_pinned_source=enforce_pinned_source,
            _manifest_path=pending_ready_manifest,
        )
        pending_ready_manifest.replace(manifest_path)
        return Glm52NonVQPackResult(
            manifest=pending_ready_state,
            manifest_sha256=audit.manifest_sha256,
            written_shards=0,
            reused_shards=audit.shard_count,
            resume_verified=True,
            largest_copy_buffer_bytes=0,
        )
    _ensure_clean_output(output_root, resume=resume)
    if resume and manifest_path.is_file():
        existing_manifest = _load_json_object(manifest_path)
        status = existing_manifest.get("pack_status")
        if status == _ready_status(enforce_pinned_source):
            if (
                int(existing_manifest.get("max_shard_payload_bytes", -1))
                != max_shard_payload_bytes
            ):
                raise ValueError(
                    "verified package max_shard_payload_bytes does not match resume request"
                )
            audit = audit_glm52_non_vq_package(
                output_root,
                source_dir=source_root,
                source_index=index,
                profile=profile,
                expected_model_id=model_id,
                expected_revision=revision,
                expected_config_sha256=config_sha256,
                expected_index_sha256=index_sha256,
                config_path=config_path,
                index_path=index_path,
                enforce_pinned_source=enforce_pinned_source,
            )
            return Glm52NonVQPackResult(
                manifest=existing_manifest,
                manifest_sha256=audit.manifest_sha256,
                written_shards=0,
                reused_shards=audit.shard_count,
                resume_verified=True,
                largest_copy_buffer_bytes=0,
            )
        _validate_partial_request(
            existing_manifest,
            profile=profile,
            model_id=model_id,
            revision=revision,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            max_shard_payload_bytes=max_shard_payload_bytes,
            enforce_pinned_source=enforce_pinned_source,
        )
    elif resume and tuple(output_root.iterdir()):
        raise ValueError(
            "resume requires a verified complete or partial non-VQ manifest"
        )

    plan, source_paths, source_payload_offsets = _plan_package(
        source_root,
        index,
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        max_shard_payload_bytes=max_shard_payload_bytes,
    )
    _validate_pinned_plan(
        plan, index=index, enforce_pinned_source=enforce_pinned_source
    )
    partial_resume = (
        resume
        and manifest_path.is_file()
        and _load_json_object(manifest_path).get("pack_status")
        == _partial_status(enforce_pinned_source)
    )
    if partial_resume:
        partial_state = _load_json_object(manifest_path)
        shard_records, payload_hashes = _resume_partial_shards(
            output_root,
            manifest=partial_state,
            plan=plan,
            source_paths=source_paths,
            source_payload_offsets=source_payload_offsets,
        )
    else:
        shard_records = []
        payload_hashes = {}
        _write_json_atomic(
            manifest_path,
            _partial_manifest(
                plan,
                profile=profile,
                model_id=model_id,
                revision=revision,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                max_shard_payload_bytes=max_shard_payload_bytes,
                completed_shards=(),
                completed_payload_hashes={},
                enforce_pinned_source=enforce_pinned_source,
            ),
        )
    reused_shards = len(shard_records)
    completed_names = {str(record["filename"]) for record in shard_records}
    shard_order = {
        shard.filename: position for position, shard in enumerate(plan.shards)
    }
    largest_copy_buffer = 0
    written_shards = 0
    for shard in plan.shards:
        if shard.filename in completed_names:
            continue
        record, shard_hashes, shard_largest = _write_shard(
            output_root / shard.filename,
            shard,
            source_paths=source_paths,
            source_payload_offsets=source_payload_offsets,
        )
        shard_records.append(record)
        shard_records.sort(key=lambda item: shard_order[str(item["filename"])])
        payload_hashes.update(shard_hashes)
        largest_copy_buffer = max(largest_copy_buffer, shard_largest)
        written_shards += 1
        _write_json_atomic(
            manifest_path,
            _partial_manifest(
                plan,
                profile=profile,
                model_id=model_id,
                revision=revision,
                config_sha256=config_sha256,
                index_sha256=index_sha256,
                max_shard_payload_bytes=max_shard_payload_bytes,
                completed_shards=shard_records,
                completed_payload_hashes=payload_hashes,
                enforce_pinned_source=enforce_pinned_source,
            ),
        )

    weight_map = {
        tensor.name: tensor.output_shard for tensor in plan.tensors
    }
    package_index = {
        "metadata": {"total_size": plan.tensor_payload_bytes},
        "weight_map": weight_map,
    }
    index_path = output_root / SAFETENSORS_INDEX_NAME
    _write_json_atomic(index_path, package_index)
    package_index_sha256 = _sha256_file(index_path)
    manifest = _ready_manifest(
        plan,
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        max_shard_payload_bytes=max_shard_payload_bytes,
        package_index_sha256=package_index_sha256,
        shard_records=shard_records,
        payload_hashes=payload_hashes,
        enforce_pinned_source=enforce_pinned_source,
    )
    _write_json_atomic(manifest_path, manifest)
    return Glm52NonVQPackResult(
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        written_shards=written_shards,
        reused_shards=reused_shards,
        resume_verified=partial_resume,
        largest_copy_buffer_bytes=largest_copy_buffer,
    )


def _strict_package_tree(
    root: Path,
    *,
    expected_files: set[str],
) -> None:
    actual_files: set[str] = set()
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise ValueError(f"package tree contains unexpected symlink: {relative}")
        if entry.is_dir():
            raise ValueError(f"package tree contains unexpected nested directory: {relative}")
        if not entry.is_file():
            raise ValueError(f"package tree contains unexpected entry: {relative}")
        actual_files.add(relative)
    if actual_files != expected_files:
        raise ValueError(
            "package tree contains missing or unmanifested files; "
            f"missing={sorted(expected_files - actual_files)} "
            f"unexpected={sorted(actual_files - expected_files)}"
        )


def _manifest_tensor_from_dict(raw: Mapping[str, object]) -> _PlannedTensor:
    try:
        shape_raw = raw["shape"]
        source_offsets_raw = raw["source_offsets"]
        output_offsets_raw = raw["output_offsets"]
        if not isinstance(shape_raw, list):
            raise TypeError("shape")
        if not isinstance(source_offsets_raw, list) or len(source_offsets_raw) != 2:
            raise TypeError("source_offsets")
        if not isinstance(output_offsets_raw, list) or len(output_offsets_raw) != 2:
            raise TypeError("output_offsets")
        return _PlannedTensor(
            name=str(raw["name"]),
            dtype=str(raw["dtype"]),
            shape=tuple(int(value) for value in shape_raw),
            parameter_count=int(raw["parameter_count"]),
            payload_bytes=int(raw["payload_bytes"]),
            source_shard=str(raw["source_shard"]),
            source_offsets=tuple(int(value) for value in source_offsets_raw),
            output_shard=str(raw["output_shard"]),
            output_offsets=tuple(int(value) for value in output_offsets_raw),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid tensor descriptor in non-VQ manifest: {raw!r}") from error


def _audit_output_shard(
    path: Path,
    *,
    tensors: Sequence[_PlannedTensor],
    expected_payload_hashes: Mapping[str, str] | None,
    source_paths: Mapping[str, Path],
    source_payload_offsets: Mapping[str, int],
) -> tuple[str, dict[str, str]]:
    header = read_safetensors_file_header(path)
    _validate_physical_extent(path, header)
    expected_names = [tensor.name for tensor in tensors]
    if set(header.tensors) != set(expected_names):
        raise ValueError(
            f"{path} tensor inventory count/names do not match manifest"
        )
    for tensor in tensors:
        actual = header.tensors[tensor.name]
        if (
            actual.dtype != tensor.dtype
            or actual.shape != tensor.shape
            or actual.data_offsets != tensor.output_offsets
        ):
            raise ValueError(
                f"{path} tensor {tensor.name!r} dtype/shape/extent does not match manifest"
            )

    file_digest = hashlib.sha256()
    payload_hashes: dict[str, str] = {}
    buffer = bytearray(COPY_BUFFER_BYTES)
    source_buffer = bytearray(COPY_BUFFER_BYTES)
    with path.open("rb") as handle:
        prefix = handle.read(header.payload_offset)
        if len(prefix) != header.payload_offset:
            raise ValueError(f"{path} has a truncated safetensors header")
        file_digest.update(prefix)
        for tensor in sorted(tensors, key=lambda item: item.output_offsets):
            remaining = tensor.payload_bytes
            tensor_digest = hashlib.sha256()
            source_path = source_paths[tensor.source_shard]
            source_start = (
                source_payload_offsets[tensor.source_shard]
                + tensor.source_offsets[0]
            )
            with source_path.open("rb") as source:
                source.seek(source_start)
                while remaining:
                    requested = min(remaining, COPY_BUFFER_BYTES)
                    view = memoryview(buffer)[:requested]
                    read = handle.readinto(view)
                    if read is None or read <= 0:
                        raise ValueError(
                            f"{path} tensor {tensor.name!r} has a truncated byte extent"
                        )
                    source_view = memoryview(source_buffer)[:read]
                    source_read = 0
                    while source_read < read:
                        count = source.readinto(source_view[source_read:read])
                        if count is None or count <= 0:
                            raise ValueError(
                                f"source tensor {tensor.name!r} is truncated during audit"
                            )
                        source_read += count
                    if view[:read] != source_view[:read]:
                        raise ValueError(
                            f"tensor {tensor.name!r} output bytes do not match pinned "
                            "source byte identity"
                        )
                    file_digest.update(view[:read])
                    tensor_digest.update(view[:read])
                    remaining -= read
            digest = tensor_digest.hexdigest()
            payload_hashes[tensor.name] = digest
            if (
                expected_payload_hashes is not None
                and digest != expected_payload_hashes[tensor.name]
            ):
                raise ValueError(
                    f"{path} tensor {tensor.name!r} payload SHA-256 byte identity failed"
                )
        if handle.read(1):
            raise ValueError(f"{path} has bytes beyond its declared physical extent")
    return file_digest.hexdigest(), payload_hashes


def audit_glm52_non_vq_package(
    output_dir: str | Path,
    *,
    source_dir: str | Path,
    source_index: SafetensorsIndex,
    profile: ModelProfile,
    expected_model_id: str,
    expected_revision: str,
    expected_config_sha256: str,
    expected_index_sha256: str,
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
    enforce_pinned_source: bool = True,
    _manifest_path: str | Path | None = None,
) -> Glm52NonVQPackageAudit:
    """Recompute package integrity and exact pinned-source byte identity."""

    root = Path(output_dir)
    source_root = Path(source_dir)
    _validate_pinned_authority(
        source_dir=source_root,
        profile=profile,
        model_id=expected_model_id,
        revision=expected_revision,
        config_sha256=expected_config_sha256,
        index_sha256=expected_index_sha256,
        config_path=config_path,
        index_path=index_path,
        index=source_index,
        enforce_pinned_source=enforce_pinned_source,
    )
    if not root.is_dir():
        raise ValueError(f"package directory does not exist: {root}")
    manifest_path = (
        root / NON_VQ_MANIFEST_NAME
        if _manifest_path is None
        else Path(_manifest_path)
    )
    try:
        manifest_relative = manifest_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("package manifest path must remain inside the artifact root") from error
    allowed_manifest_names = {
        NON_VQ_MANIFEST_NAME,
        f"{NON_VQ_MANIFEST_NAME}.resume-previous.json",
    }
    if manifest_relative not in allowed_manifest_names:
        raise ValueError("package manifest path is not a recognized canonical/resume name")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("package manifest must be a regular local file")
    manifest = _load_json_object(manifest_path)
    expected_lineage = {
        "schema_version": 1,
        "record_type": NON_VQ_RECORD_TYPE,
        "pack_status": _ready_status(enforce_pinned_source),
        "production_ready": enforce_pinned_source,
        "source_authority": _source_authority(enforce_pinned_source),
        "source_blob_inventory_sha256": (
            NON_VQ_PINNED_BLOB_INVENTORY_SHA256
            if enforce_pinned_source
            else None
        ),
        "copy_mode": NON_VQ_COPY_MODE,
        "selection_policy": NON_VQ_SELECTION_POLICY,
        "profile": profile.name,
        "model_id": expected_model_id,
        "source_revision": expected_revision,
        "config_sha256": expected_config_sha256,
        "index_sha256": expected_index_sha256,
    }
    for key, expected in expected_lineage.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"non-VQ manifest {key} mismatch: expected {expected!r}, "
                f"found {manifest.get(key)!r}"
            )
    max_shard_payload_bytes = manifest.get("max_shard_payload_bytes")
    if (
        type(max_shard_payload_bytes) is not int
        or max_shard_payload_bytes <= 0
    ):
        raise ValueError("manifest max shard payload bytes must be positive")
    plan, source_paths, source_payload_offsets = _plan_package(
        source_root,
        source_index,
        profile=profile,
        model_id=expected_model_id,
        revision=expected_revision,
        config_sha256=expected_config_sha256,
        index_sha256=expected_index_sha256,
        max_shard_payload_bytes=max_shard_payload_bytes,
    )
    _validate_pinned_plan(
        plan,
        index=source_index,
        enforce_pinned_source=enforce_pinned_source,
    )
    dtype_counts = {
        dtype: sum(tensor.dtype == dtype for tensor in plan.tensors)
        for dtype in sorted(_RETAINED_DTYPES)
        if any(tensor.dtype == dtype for tensor in plan.tensors)
    }
    expected_summary = {
        "source_inventory_sha256": plan.source_inventory_sha256,
        "plan_sha256": plan.plan_sha256,
        "retained_tensor_count": len(plan.tensors),
        "parameter_count": plan.parameter_count,
        "tensor_payload_bytes": plan.tensor_payload_bytes,
        "dtype_tensor_counts": dtype_counts,
        "excluded_routed_tensor_count": plan.excluded_routed_tensor_count,
        "excluded_mtp_tensor_count": plan.excluded_mtp_tensor_count,
        "excluded_runtime_vq_tensor_count": plan.excluded_runtime_vq_tensor_count,
        "shard_count": len(plan.shards),
        "index_path": SAFETENSORS_INDEX_NAME,
    }
    for key, expected in expected_summary.items():
        if manifest.get(key) != expected:
            label = key.replace("_", " ")
            raise ValueError(
                f"non-VQ manifest {label} summary mismatch: expected {expected!r}, "
                f"found {manifest.get(key)!r}"
            )

    raw_shards = manifest.get("shards")
    raw_tensors = manifest.get("tensors")
    if not isinstance(raw_shards, list) or not isinstance(raw_tensors, list):
        raise ValueError("non-VQ manifest shards/tensors must be arrays")
    shard_records: list[dict[str, object]] = []
    shard_names: list[str] = []
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise ValueError("non-VQ manifest shard record must be an object")
        filename = str(raw.get("filename", ""))
        if not _SHARD_NAME_RE.fullmatch(filename) or Path(filename).name != filename:
            raise ValueError(f"unsafe or unexpected package shard name {filename!r}")
        shard_records.append(dict(raw))
        shard_names.append(filename)
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("non-VQ manifest contains duplicate shard names")
    if int(manifest.get("shard_count", -1)) != len(shard_names):
        raise ValueError("non-VQ manifest shard count does not match shard records")
    expected_shard_names = [shard.filename for shard in plan.shards]
    if shard_names != expected_shard_names:
        raise ValueError(
            "non-VQ manifest shard plan/order mismatch: "
            f"expected={expected_shard_names} found={shard_names}"
        )
    for shard in plan.shards:
        if (
            shard.tensor_payload_bytes > max_shard_payload_bytes
            and len(shard.tensors) != 1
        ):
            raise ValueError(
                f"shard {shard.filename} exceeds max shard payload bytes"
            )

    expected_files = {
        manifest_relative,
        SAFETENSORS_INDEX_NAME,
        *shard_names,
    }
    _strict_package_tree(root, expected_files=expected_files)

    index_path = root / SAFETENSORS_INDEX_NAME
    package_index_sha256 = _sha256_file(index_path)
    if package_index_sha256 != manifest.get("package_index_sha256"):
        raise ValueError("package index SHA-256 hash does not match manifest")
    output_index = load_safetensors_index(index_path)

    tensors: list[_PlannedTensor] = []
    raw_tensor_names: list[str] = []
    expected_payload_hashes: dict[str, str] = {}
    seen_names: set[str] = set()
    for raw in raw_tensors:
        if not isinstance(raw, dict):
            raise ValueError("non-VQ manifest tensor record must be an object")
        tensor = _manifest_tensor_from_dict(raw)
        raw_tensor_names.append(tensor.name)
        if tensor.name in seen_names:
            raise ValueError(f"duplicate manifest tensor {tensor.name!r}")
        seen_names.add(tensor.name)
        if not _is_main_non_routed_tensor(tensor.name, profile):
            raise ValueError(
                f"manifest contains routed, runtime-VQ, or MTP tensor {tensor.name!r}"
            )
        if tensor.dtype not in _RETAINED_DTYPES:
            raise ValueError(
                f"manifest non-routed tensor {tensor.name!r} has invalid dtype {tensor.dtype!r}"
            )
        dtype_bytes = _DTYPE_BYTES[tensor.dtype]
        parameter_count = 1
        for dim in tensor.shape:
            if dim < 0:
                raise ValueError(f"tensor {tensor.name!r} has a negative shape dimension")
            parameter_count *= dim
        expected_bytes = parameter_count * dtype_bytes
        if tensor.parameter_count != parameter_count or tensor.payload_bytes != expected_bytes:
            raise ValueError(
                f"tensor {tensor.name!r} parameter count/payload bytes mismatch"
            )
        if tensor.output_shard not in shard_names:
            raise ValueError(
                f"tensor {tensor.name!r} points to unmanifested output shard"
            )
        payload_sha256 = raw.get("payload_sha256")
        if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
            raise ValueError(f"tensor {tensor.name!r} has invalid payload SHA-256")
        expected_payload_hashes[tensor.name] = payload_sha256
        tensors.append(tensor)
    expected_tensor_names = [tensor.name for tensor in plan.tensors]
    if raw_tensor_names != expected_tensor_names:
        raise ValueError("non-VQ tensor inventory plan/order mismatch")
    for tensor, expected_tensor in zip(tensors, plan.tensors, strict=True):
        if tensor.plan_dict() != expected_tensor.plan_dict():
            raise ValueError(
                f"non-VQ tensor {tensor.name!r} source/output plan mismatch"
            )

    expected_weight_map = {
        tensor.name: tensor.output_shard for tensor in tensors
    }
    if output_index.weight_map != expected_weight_map:
        raise ValueError("package index tensor count or weight map does not match manifest")
    parameter_count = sum(tensor.parameter_count for tensor in tensors)
    tensor_payload_bytes = sum(tensor.payload_bytes for tensor in tensors)
    total_size = output_index.metadata.get("total_size")
    if total_size != tensor_payload_bytes:
        raise ValueError(
            f"package index total_size bytes mismatch: {total_size} != {tensor_payload_bytes}"
        )
    if int(manifest.get("retained_tensor_count", -1)) != len(tensors):
        raise ValueError("manifest retained tensor count mismatch")
    if int(manifest.get("parameter_count", -1)) != parameter_count:
        raise ValueError("manifest parameter count mismatch")
    if int(manifest.get("tensor_payload_bytes", -1)) != tensor_payload_bytes:
        raise ValueError("manifest tensor payload bytes mismatch")

    actual_shard_records: list[dict[str, object]] = []
    for record in sorted(shard_records, key=lambda item: str(item["filename"])):
        filename = str(record["filename"])
        shard_tensors = tuple(
            tensor for tensor in tensors if tensor.output_shard == filename
        )
        if int(record.get("tensor_count", -1)) != len(shard_tensors):
            raise ValueError(f"{filename} tensor count mismatch")
        shard_payload_bytes = sum(tensor.payload_bytes for tensor in shard_tensors)
        if int(record.get("tensor_payload_bytes", -1)) != shard_payload_bytes:
            raise ValueError(f"{filename} tensor payload bytes mismatch")
        inventory_sha256 = _canonical_sha256(
            [tensor.plan_dict() for tensor in shard_tensors]
        )
        if record.get("tensor_inventory_sha256") != inventory_sha256:
            raise ValueError(f"{filename} tensor inventory hash mismatch")
        path = root / filename
        file_sha256, _ = _audit_output_shard(
            path,
            tensors=shard_tensors,
            expected_payload_hashes=expected_payload_hashes,
            source_paths=source_paths,
            source_payload_offsets=source_payload_offsets,
        )
        if int(record.get("file_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"{filename} file bytes mismatch")
        if record.get("file_sha256") != file_sha256:
            raise ValueError(f"{filename} file SHA-256 byte identity mismatch")
        actual_shard_records.append(dict(record))

    package_set_sha256 = _package_set_sha256(
        package_index_sha256=package_index_sha256,
        shard_records=actual_shard_records,
    )
    if manifest.get("package_set_sha256") != package_set_sha256:
        raise ValueError("package set hash does not match recomputed bytes")
    artifact_tree_bytes = sum((root / name).stat().st_size for name in expected_files)
    return Glm52NonVQPackageAudit(
        artifact_dir=str(root),
        manifest_path=str(manifest_path),
        manifest_sha256=_sha256_file(manifest_path),
        retained_tensor_count=len(tensors),
        parameter_count=parameter_count,
        tensor_payload_bytes=tensor_payload_bytes,
        shard_count=len(shard_records),
        artifact_tree_bytes=artifact_tree_bytes,
        package_set_sha256=package_set_sha256,
        checks={
            "lineage": True,
            "strict_tree": True,
            "index": True,
            "tensor_inventory": True,
            "physical_extents": True,
            "payload_hashes": True,
            "shard_hashes": True,
            "accounting": True,
        },
    )


def _emit_json(path: str | None, payload: Mapping[str, object]) -> None:
    if path is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _write_json_atomic(Path(path), payload)


def _sha256_path(path: str | Path) -> str:
    return _sha256_file(Path(path))


def _run_cli(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    actual_index_sha256 = _sha256_path(args.index_path)
    if actual_index_sha256 != args.index_sha256:
        raise ValueError(
            f"index SHA-256 mismatch: expected {args.index_sha256}, "
            f"found {actual_index_sha256}"
        )
    profile = load_profile(args.profile_path)
    index = load_safetensors_index(args.index_path)
    result = pack_glm52_non_vq_safetensors(
        args.source_dir,
        index,
        args.output_dir,
        profile=profile,
        model_id=args.model_id,
        revision=args.revision,
        config_sha256=args.config_sha256,
        index_sha256=args.index_sha256,
        max_shard_payload_bytes=args.max_shard_payload_bytes,
        resume=args.resume,
        config_path=args.config_path,
        index_path=args.index_path,
        enforce_pinned_source=not args.allow_unpinned_fixture,
    )
    audit = audit_glm52_non_vq_package(
        args.output_dir,
        source_dir=args.source_dir,
        source_index=index,
        profile=profile,
        expected_model_id=args.model_id,
        expected_revision=args.revision,
        expected_config_sha256=args.config_sha256,
        expected_index_sha256=args.index_sha256,
        config_path=args.config_path,
        index_path=args.index_path,
        enforce_pinned_source=not args.allow_unpinned_fixture,
    )
    payload = result.to_dict()
    payload["package_audit_pass"] = audit.audit_pass
    payload["package_audit"] = audit.to_dict()
    return payload, 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw-range repack of GLM-5.2 main-model non-routed BF16/F32 tensors"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--max-shard-payload-bytes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unpinned-fixture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload, code = _run_cli(args)
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": "glm52_non_vq_package_run",
            "pack_status": "glm52_non_vq_package_failed",
            "package_audit_pass": False,
            "model_id": args.model_id,
            "source_revision": args.revision,
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _emit_json(args.output_json, payload)
        return 1
    _emit_json(args.output_json, payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
