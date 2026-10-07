"""Strict integrity and accounting audit for GLM-5.2 REAP VQ groups."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open

from keep.convert.stream_convert import SafetensorsIndex
from keep.io.load import inspect_safetensors
from keep.io.schema import codebook_metadata_for_bits
from keep.io.source_safetensors import (
    SafetensorsFileHeader,
    SafetensorsTensorHeader,
    read_safetensors_file_header,
)
from ramp.models.profiles import ModelProfile


GLM52_ARTIFACT_MANIFEST_NAME = "conversion-manifest.json"
GLM52_ARTIFACT_RECORD_TYPE = "glm52_modelopt_nvfp4_materialization_manifest"
GLM52_RUN_RECORD_TYPE = "glm52_modelopt_nvfp4_materialization_run"
GLM52_READY_STATUS = "glm52_modelopt_nvfp4_groups_ready"
GLM52_SOURCE_ENCODING = "modelopt_nvfp4"
GLM52_SOURCE_DECODER = "modelopt_nvfp4_v1"
GLM52_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GLM52_CODEBOOK_TENSOR = "model.vq_codebook.e8"

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
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

_MODEL_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_ROUTED_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<member>.+)$"
)


@dataclass(frozen=True)
class GLM52ReapSourceAccounting:
    source_dir: str
    source_tensor_count: int
    source_tensor_payload_bytes: int
    source_shard_file_bytes: int
    main_non_routed_tensor_count: int
    main_non_routed_parameter_count: int
    main_non_routed_tensor_payload_bytes: int
    main_non_routed_source_shard_file_footprint_bytes: int
    main_routed_tensor_count: int
    main_routed_parameter_count: int
    main_routed_source_payload_bytes: int
    excluded_mtp_tensor_count: int
    excluded_mtp_parameter_count: int
    excluded_mtp_tensor_payload_bytes: int
    main_model_parameter_count_excluding_mtp: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GLM52ReapMaterializationAudit:
    artifact_dir: str
    manifest_path: str
    manifest_sha256: str
    materialization_scope: str
    expected_group_keys: tuple[str, ...]
    ready_group_keys: tuple[str, ...]
    complete_layer_ids: tuple[int, ...]
    partial_layer_ids: tuple[int, ...]
    actual_routed_weight_count: int
    actual_routed_codes_bytes: int
    actual_routed_scales_bytes: int
    actual_routed_codebook_bytes: int
    actual_routed_payload_bytes: int
    actual_routed_bpw: float
    actual_artifact_bytes: int
    physical_routed_bpw: float
    artifact_tree_bytes: int
    group_set_sha256: str
    actual_whole_model_artifact_bytes: int | None
    actual_whole_model_bpw: float | None
    whole_model_values_actual: bool
    full_routed_artifact_ready: bool
    dense_routed_experts: bool
    evidence_records_checked: int
    resume_verified: bool
    byte_identity_verified: bool
    checks: dict[str, bool]

    @property
    def audit_pass(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["expected_group_keys"] = list(self.expected_group_keys)
        payload["ready_group_keys"] = list(self.ready_group_keys)
        payload["complete_layer_ids"] = list(self.complete_layer_ids)
        payload["partial_layer_ids"] = list(self.partial_layer_ids)
        payload["audit_pass"] = self.audit_pass
        return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_source_shard_path(source_root: Path, shard_name: str) -> Path:
    relative = Path(shard_name)
    if not shard_name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe safetensors shard path in source index: {shard_name!r}")
    path = source_root / relative
    if not path.is_file():
        raise ValueError(f"source safetensors shard is missing: {path}")
    if path.is_symlink():
        resolved = path.resolve()
        try:
            resolved.relative_to(source_root.resolve())
        except ValueError:
            # Hugging Face snapshots intentionally point into their sibling blob
            # store. The indexed relative name remains the trusted containment
            # boundary; do not reject that cache layout merely for using links.
            pass
    return path


def _tensor_payload_bytes(header: SafetensorsTensorHeader) -> int:
    start, end = header.data_offsets
    if start < 0 or end < start:
        raise ValueError(f"invalid safetensors data offsets: {(start, end)}")
    return end - start


def _validate_safetensors_physical_extent(
    path: Path,
    header: SafetensorsFileHeader,
) -> None:
    expected_file_bytes = header.payload_offset + header.declared_payload_bytes
    actual_file_bytes = path.stat().st_size
    if expected_file_bytes != actual_file_bytes:
        raise ValueError(
            f"{path} physical extent does not match file size "
            f"({expected_file_bytes} != {actual_file_bytes}); payload may be truncated"
        )
    cursor = 0
    for tensor_name, tensor in sorted(
        header.tensors.items(), key=lambda item: item[1].data_offsets
    ):
        start, end = tensor.data_offsets
        if start != cursor:
            raise ValueError(
                f"{path} tensor {tensor_name!r} has a non-contiguous or overlapping "
                f"physical extent starting at {start}, expected {cursor}"
            )
        dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(tensor.dtype)
        if dtype_bytes is None:
            raise ValueError(
                f"{path} tensor {tensor_name!r} has unsupported dtype {tensor.dtype!r}"
            )
        declared_bytes = end - start
        expected_bytes = tensor.element_count * dtype_bytes
        if declared_bytes != expected_bytes:
            raise ValueError(
                f"{path} tensor {tensor_name!r} byte extent must be {expected_bytes}, "
                f"found {declared_bytes}"
            )
        cursor = end
    if cursor != header.declared_payload_bytes:
        raise ValueError(
            f"{path} tensor extents end at {cursor}, expected "
            f"{header.declared_payload_bytes}"
        )


def _logical_parameter_count(
    name: str,
    *,
    header: SafetensorsTensorHeader,
    index: SafetensorsIndex,
) -> int:
    if name.endswith("_scale") or name.endswith("_scale_2") or name.endswith(
        "input_scale"
    ):
        return 0
    elements = int(header.element_count)
    if (
        name.endswith(".weight")
        and header.dtype == "U8"
        and f"{name}_scale" in index.weight_map
    ):
        elements *= 2
    return elements


def audit_glm52_reap_source_accounting(
    source_dir: str | Path,
    *,
    index: SafetensorsIndex,
    profile: ModelProfile,
) -> GLM52ReapSourceAccounting:
    """Inventory tensor payload and logical parameters without reading payloads."""

    source_root = Path(source_dir)
    if not source_root.is_dir():
        raise ValueError(f"source directory does not exist: {source_root}")
    headers = {}
    shard_paths: dict[str, Path] = {}
    for shard_name in index.shards:
        path = _safe_source_shard_path(source_root, shard_name)
        shard_paths[shard_name] = path
        header = read_safetensors_file_header(path)
        _validate_safetensors_physical_extent(path, header)
        headers[shard_name] = header
    indexed_by_shard: dict[str, set[str]] = {name: set() for name in index.shards}
    for tensor_name, shard_name in index.weight_map.items():
        if shard_name not in indexed_by_shard:
            raise ValueError(
                f"tensor {tensor_name!r} points to unknown shard {shard_name!r}"
            )
        indexed_by_shard[shard_name].add(tensor_name)
    for shard_name, header in headers.items():
        unindexed = sorted(set(header.tensors) - indexed_by_shard[shard_name])
        if unindexed:
            raise ValueError(
                f"source shard {shard_name} contains unindexed header tensor(s): "
                f"{unindexed[:10]}"
            )

    source_payload_bytes = 0
    main_non_routed_payload_bytes = 0
    main_non_routed_parameters = 0
    main_non_routed_count = 0
    main_non_routed_shards: set[str] = set()
    main_routed_payload_bytes = 0
    main_routed_parameters = 0
    main_routed_count = 0
    mtp_payload_bytes = 0
    mtp_parameters = 0
    mtp_count = 0

    for tensor_name, shard_name in index.weight_map.items():
        if shard_name not in headers:
            raise ValueError(
                f"tensor {tensor_name!r} points to unknown shard {shard_name!r}"
            )
        header = headers[shard_name].tensors.get(tensor_name)
        if header is None:
            raise ValueError(
                f"tensor {tensor_name!r} is indexed but absent from {shard_name}"
            )
        payload_bytes = _tensor_payload_bytes(header)
        source_payload_bytes += payload_bytes
        logical_parameters = _logical_parameter_count(
            tensor_name,
            header=header,
            index=index,
        )
        layer_match = _MODEL_LAYER_RE.match(tensor_name)
        layer = int(layer_match.group("layer")) if layer_match is not None else None
        routed_match = _ROUTED_EXPERT_RE.match(tensor_name)

        if layer is not None and layer >= profile.num_layers:
            mtp_count += 1
            mtp_payload_bytes += payload_bytes
            mtp_parameters += logical_parameters
        elif (
            routed_match is not None
            and profile.first_sparse_layer
            <= int(routed_match.group("layer"))
            < profile.num_layers
        ):
            main_routed_count += 1
            main_routed_payload_bytes += payload_bytes
            main_routed_parameters += logical_parameters
        else:
            main_non_routed_count += 1
            main_non_routed_payload_bytes += payload_bytes
            main_non_routed_parameters += logical_parameters
            main_non_routed_shards.add(shard_name)

    source_shard_bytes = sum(path.stat().st_size for path in shard_paths.values())
    retained_shard_footprint = sum(
        shard_paths[shard_name].stat().st_size
        for shard_name in main_non_routed_shards
    )
    return GLM52ReapSourceAccounting(
        source_dir=str(source_root),
        source_tensor_count=len(index.weight_map),
        source_tensor_payload_bytes=source_payload_bytes,
        source_shard_file_bytes=source_shard_bytes,
        main_non_routed_tensor_count=main_non_routed_count,
        main_non_routed_parameter_count=main_non_routed_parameters,
        main_non_routed_tensor_payload_bytes=main_non_routed_payload_bytes,
        main_non_routed_source_shard_file_footprint_bytes=retained_shard_footprint,
        main_routed_tensor_count=main_routed_count,
        main_routed_parameter_count=main_routed_parameters,
        main_routed_source_payload_bytes=main_routed_payload_bytes,
        excluded_mtp_tensor_count=mtp_count,
        excluded_mtp_parameter_count=mtp_parameters,
        excluded_mtp_tensor_payload_bytes=mtp_payload_bytes,
        main_model_parameter_count_excluding_mtp=(
            main_non_routed_parameters + main_routed_parameters
        ),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _manifest_root(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if candidate.is_dir():
        root = candidate
        manifest_path = root / GLM52_ARTIFACT_MANIFEST_NAME
    else:
        root = candidate.parent
        manifest_path = candidate
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    if not manifest_path.is_file():
        raise ValueError(f"missing artifact manifest: {manifest_path}")
    if manifest_path.is_symlink():
        raise ValueError("artifact manifest must be a regular local file, not a symlink")
    return root, manifest_path


def _group_key(layer: int, projection: str) -> str:
    return f"{layer}:{projection}"


def _parse_group_key(value: str) -> tuple[int, str]:
    try:
        layer_raw, projection = value.split(":", maxsplit=1)
        layer = int(layer_raw)
    except (AttributeError, ValueError) as error:
        raise ValueError(
            f"expected group keys must use '<layer>:<projection>', found {value!r}"
        ) from error
    if projection not in GLM52_PROJECTIONS:
        raise ValueError(f"expected group key has unsupported projection: {value!r}")
    return layer, projection


def _canonical_profile_group_keys(profile: ModelProfile) -> tuple[str, ...]:
    first_layer = profile.first_sparse_layer
    last_layer = profile.num_layers - 1
    if first_layer < 0 or last_layer < first_layer:
        raise ValueError("profile has an invalid sparse-layer range")
    layer_count = last_layer - first_layer + 1
    if layer_count != profile.num_sparse_layers:
        raise ValueError(
            "profile sparse-layer count does not match first_sparse_layer/num_layers"
        )
    return tuple(
        _group_key(layer, projection)
        for layer in range(first_layer, profile.num_layers)
        for projection in GLM52_PROJECTIONS
    )


def _projection_dims(profile: ModelProfile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    if projection == "down_proj":
        return profile.moe_intermediate_size, profile.hidden_size
    raise ValueError(f"unsupported GLM52 projection {projection!r}")


def _profile_group_size(profile: ModelProfile, projection: str) -> int:
    policy_key = projection.removesuffix("_proj")
    try:
        value = int(profile.group_size_policy[policy_key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"profile is missing a valid group-size policy for {projection}"
        ) from error
    if value <= 0:
        raise ValueError(f"profile group size for {projection} must be positive")
    return value


def _expected_group_filename(layer: int, projection: str) -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _expected_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _expect_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, found {actual!r}")


def _expect_int(name: str, actual: object, expected: int) -> None:
    if type(actual) is not int or actual != expected:
        raise ValueError(f"{name} must be {expected}, found {actual!r}")


def _validate_manifest_contract(
    payload: Mapping[str, object],
    *,
    profile: ModelProfile,
    expected_group_keys: tuple[str, ...],
    all_group_keys: tuple[str, ...],
    expected_code_bits: int,
    expected_config_sha256: str,
    expected_index_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    full = expected_group_keys == all_group_keys
    expected_scope = "full" if full else "bounded"
    expected_group_sizes = {
        _profile_group_size(profile, _parse_group_key(key)[1])
        for key in expected_group_keys
    }
    if len(expected_group_sizes) != 1:
        raise ValueError("selected profile groups must use one group size per artifact run")
    expected_group_size = next(iter(expected_group_sizes))

    _expect_int("manifest schema_version", payload.get("schema_version"), 1)
    _expect_equal("manifest record_type", payload.get("record_type"), GLM52_ARTIFACT_RECORD_TYPE)
    _expect_equal("manifest materialization_status", payload.get("materialization_status"), GLM52_READY_STATUS)
    _expect_equal("manifest materialization_scope", payload.get("materialization_scope"), expected_scope)
    _expect_equal("manifest full_group_coverage", payload.get("full_group_coverage"), full)
    _expect_equal("manifest materialization_blocked", payload.get("materialization_blocked"), False)
    _expect_equal("manifest materialization_blockers", payload.get("materialization_blockers"), [])
    _expect_equal("manifest profile", payload.get("profile"), profile.name)
    _expect_equal("manifest model_id", payload.get("model_id"), profile.hf_model_id)
    _expect_equal("manifest source revision", payload.get("source_revision"), profile.revision)
    _expect_equal("manifest config SHA-256", payload.get("config_sha256"), expected_config_sha256)
    _expect_equal("manifest index SHA-256", payload.get("index_sha256"), expected_index_sha256)
    _expect_equal("manifest source encoding", payload.get("source_weight_encoding"), GLM52_SOURCE_ENCODING)
    _expect_equal("manifest source decoder", payload.get("source_decoder"), GLM52_SOURCE_DECODER)
    _expect_int("manifest planned_vq_groups", payload.get("planned_vq_groups"), len(all_group_keys))
    _expect_int("manifest selected_vq_groups", payload.get("selected_vq_groups"), len(expected_group_keys))
    _expect_int("manifest ready_vq_groups", payload.get("ready_vq_groups"), len(expected_group_keys))
    _expect_int(
        "manifest skipped_vq_groups",
        payload.get("skipped_vq_groups"),
        len(all_group_keys) - len(expected_group_keys),
    )
    _expect_equal("manifest selected group selection", payload.get("selected_group_keys"), list(expected_group_keys))
    _expect_int("manifest code bits budget", payload.get("code_bits"), expected_code_bits)
    _expect_int("manifest group size", payload.get("group_size"), expected_group_size)
    if payload.get("scale_estimator") not in {"max_abs", "percentile_99"}:
        raise ValueError("manifest scale estimator is unsupported")
    expected_codebook_name, expected_codebook_sha256 = codebook_metadata_for_bits(
        expected_code_bits
    )
    _expect_equal("manifest codebook name", payload.get("codebook_name"), expected_codebook_name)
    _expect_equal(
        "manifest codebook SHA-256",
        payload.get("codebook_sha256"),
        expected_codebook_sha256,
    )
    _expect_equal("manifest dense routed experts", payload.get("dense_checkpoint_written"), False)

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("materialization manifest groups must be a list")
    if len(raw_groups) != len(expected_group_keys):
        raise ValueError(
            "materialization manifest group count does not match the exact expected selection"
        )
    groups: list[dict[str, Any]] = []
    for record in raw_groups:
        if not isinstance(record, dict):
            raise ValueError("materialization manifest group records must be objects")
        groups.append(record)
    actual_keys = tuple(
        _group_key(record.get("layer"), record.get("projection"))
        for record in groups
        if type(record.get("layer")) is int and isinstance(record.get("projection"), str)
    )
    if len(actual_keys) != len(groups) or actual_keys != expected_group_keys:
        raise ValueError(
            "materialization manifest group keys do not match the exact expected selection"
        )
    return groups, expected_scope


def _load_quantization_config(path: Path) -> dict[str, Any]:
    inspection = inspect_safetensors(path)
    raw = inspection.metadata.get("quantization_config")
    if raw is None:
        raise ValueError(f"{path.name} is missing quantization_config metadata")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} has invalid quantization_config metadata") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} quantization_config must be an object")
    return payload


def _actual_codebook_sha256(path: Path) -> str:
    with safe_open(path, framework="np") as handle:
        values = handle.get_tensor(GLM52_CODEBOOK_TENSOR)
    words = np.asarray(values, dtype=np.uint32)
    if words.shape != (256,):
        raise ValueError(f"{path.name} codebook tensor must have shape (256,)")
    return hashlib.sha256(words.tobytes(order="C")).hexdigest()


def _validate_group_artifact(
    *,
    root: Path,
    record: Mapping[str, object],
    group_key: str,
    profile: ModelProfile,
    expected_code_bits: int,
    expected_config_sha256: str,
    expected_index_sha256: str,
    scale_estimator: str,
) -> dict[str, object]:
    layer, projection = _parse_group_key(group_key)
    if layer == 78:
        raise ValueError("layer 78 MTP artifacts are forbidden")
    filename = _expected_group_filename(layer, projection)
    path = root / filename
    if not path.is_file():
        raise ValueError(f"missing expected group file {filename}")
    if path.is_symlink():
        raise ValueError(f"group file {filename} must not be a symlink")

    recorded_path = record.get("artifact_path")
    if not isinstance(recorded_path, str) or Path(recorded_path).name != filename:
        raise ValueError(
            f"manifest artifact file name for {group_key} must be {filename!r}"
        )
    actual_bytes = path.stat().st_size
    _expect_int(f"{filename} byte size", record.get("artifact_bytes"), actual_bytes)
    actual_sha256 = _sha256_file(path)
    _expect_equal(f"{filename} SHA-256 hash", record.get("artifact_sha256"), actual_sha256)
    _expect_equal(f"{filename} status", record.get("status"), "ready")
    _expect_int(f"{filename} expert count", record.get("expert_count"), profile.num_experts)

    input_dims, output_dims = _projection_dims(profile, projection)
    group_size = _profile_group_size(profile, projection)
    prefix = _expected_prefix(layer, projection)
    codes_name = f"{prefix}.codes"
    scales_name = f"{prefix}.scales"
    expected_codes_shape = (
        profile.num_experts,
        output_dims,
        input_dims // 8,
    )
    expected_scales_shape = (
        profile.num_experts,
        output_dims,
        input_dims // group_size,
    )
    expected_codes_dtype = "U8" if expected_code_bits == 8 else "U16"
    expected_manifest_codes_dtype = "uint8" if expected_code_bits == 8 else "uint16"

    inspection = inspect_safetensors(path)
    required_names = {codes_name, scales_name, GLM52_CODEBOOK_TENSOR}
    actual_names = set(inspection.tensors)
    missing_names = sorted(required_names - actual_names)
    if missing_names:
        raise ValueError(f"{filename} is missing tensor(s): {missing_names}")
    unexpected_names = sorted(actual_names - required_names)
    if unexpected_names:
        if any(name.endswith(".weight") for name in unexpected_names):
            raise ValueError(
                f"{filename} contains forbidden dense routed weight tensor(s): "
                f"{unexpected_names}"
            )
        raise ValueError(f"{filename} contains unexpected tensor(s): {unexpected_names}")

    codes = inspection.tensors[codes_name]
    scales = inspection.tensors[scales_name]
    codebook = inspection.tensors[GLM52_CODEBOOK_TENSOR]
    if codes.dtype != expected_codes_dtype:
        raise ValueError(
            f"{filename} codes dtype must be {expected_codes_dtype}, found {codes.dtype}"
        )
    if codes.shape != expected_codes_shape:
        raise ValueError(
            f"{filename} codes shape must be {expected_codes_shape}, found {codes.shape}"
        )
    if scales.dtype != "F16":
        raise ValueError(f"{filename} scales dtype must be F16, found {scales.dtype}")
    if scales.shape != expected_scales_shape:
        raise ValueError(
            f"{filename} scales shape must be {expected_scales_shape}, found {scales.shape}"
        )
    if codebook.dtype != "U32" or codebook.shape != (256,):
        raise ValueError(f"{filename} codebook tensor must be U32 with shape (256,)")

    _expect_equal(f"{filename} manifest codes name", record.get("codes_name"), codes_name)
    _expect_equal(
        f"{filename} manifest codes shape",
        record.get("codes_shape"),
        list(expected_codes_shape),
    )
    _expect_equal(
        f"{filename} manifest codes dtype",
        record.get("codes_dtype"),
        expected_manifest_codes_dtype,
    )
    _expect_equal(f"{filename} manifest scales name", record.get("scales_name"), scales_name)
    _expect_equal(
        f"{filename} manifest scales shape",
        record.get("scales_shape"),
        list(expected_scales_shape),
    )
    _expect_equal(
        f"{filename} manifest scales dtype", record.get("scales_dtype"), "float16"
    )

    quantization = _load_quantization_config(path)
    _expect_equal(f"{filename} quant_method", quantization.get("quant_method"), "mlx_vq_e8")
    _expect_int(f"{filename} quantization version", quantization.get("version"), 1)
    _expect_int(
        f"{filename} code bits budget",
        quantization.get("default_code_bits"),
        expected_code_bits,
    )
    _expect_int(
        f"{filename} group size", quantization.get("default_group_size"), group_size
    )
    codebook_metadata = quantization.get("codebook")
    if not isinstance(codebook_metadata, dict):
        raise ValueError(f"{filename} codebook metadata must be an object")
    expected_codebook_name, expected_codebook_sha256 = codebook_metadata_for_bits(
        expected_code_bits
    )
    _expect_equal(
        f"{filename} codebook name",
        codebook_metadata.get("name"),
        expected_codebook_name,
    )
    _expect_equal(f"{filename} codebook dtype", codebook_metadata.get("dtype"), "uint32")
    _expect_int(f"{filename} codebook entries", codebook_metadata.get("entries"), 256)
    _expect_equal(
        f"{filename} codebook metadata SHA-256",
        codebook_metadata.get("sha256"),
        expected_codebook_sha256,
    )
    _expect_equal(
        f"{filename} embedded codebook SHA-256",
        _actual_codebook_sha256(path),
        expected_codebook_sha256,
    )

    policy = quantization.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{filename} quantization policy must be an object")
    expected_policy = {
        "scale_estimator": scale_estimator,
        "source_weight_encoding": GLM52_SOURCE_ENCODING,
        "source_decoder": GLM52_SOURCE_DECODER,
        "source_model_id": profile.hf_model_id,
        "source_revision": profile.revision,
        "source_config_sha256": expected_config_sha256,
        "source_index_sha256": expected_index_sha256,
        "source_profile": profile.name,
        "decoded_expert_working_set": "one_per_worker",
    }
    for name, expected in expected_policy.items():
        _expect_equal(f"{filename} {name.replace('_', ' ')}", policy.get(name), expected)

    raw_header = read_safetensors_file_header(path)
    _validate_safetensors_physical_extent(path, raw_header)

    return {
        "group_key": group_key,
        "artifact_bytes": actual_bytes,
        "artifact_sha256": actual_sha256,
        "routed_weight_count": profile.num_experts * input_dims * output_dims,
        "codes_bytes": codes.nbytes,
        "scales_bytes": scales.nbytes,
        "codebook_bytes": codebook.nbytes,
    }


def _load_evidence_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"materialization evidence path does not exist: {path}")
        if path.suffix == ".jsonl":
            for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{path}:{line_number}: materialization evidence must be an object"
                    )
                records.append(payload)
        else:
            records.append(_load_json_object(path))
    return records


def _evidence_group_identity(record: Mapping[str, object]) -> tuple[tuple[str, int, str], ...]:
    raw_groups = record.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("materialization resume evidence is missing groups")
    identity: list[tuple[str, int, str]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ValueError("materialization resume evidence group must be an object")
        layer = raw_group.get("layer")
        projection = raw_group.get("projection")
        artifact_bytes = raw_group.get("artifact_bytes")
        artifact_sha256 = raw_group.get("artifact_sha256")
        if (
            type(layer) is not int
            or not isinstance(projection, str)
            or type(artifact_bytes) is not int
            or not isinstance(artifact_sha256, str)
        ):
            raise ValueError("materialization resume evidence group identity is invalid")
        identity.append(
            (_group_key(layer, projection), artifact_bytes, artifact_sha256)
        )
    return tuple(identity)


def _validate_evidence(
    records: Sequence[Mapping[str, object]],
    *,
    expected_group_keys: tuple[str, ...],
    expected_identity: tuple[tuple[str, int, str], ...],
    profile: ModelProfile,
    expected_code_bits: int,
    expected_config_sha256: str,
    expected_index_sha256: str,
) -> tuple[bool, bool]:
    if not records:
        return False, False
    initial_seen = False
    resume_seen = False
    run_signatures: set[tuple[object, ...]] = set()
    group_count = len(expected_group_keys)
    for record in records:
        _expect_equal("materialization evidence record type", record.get("record_type"), GLM52_RUN_RECORD_TYPE)
        _expect_equal("materialization evidence model", record.get("model_id"), profile.hf_model_id)
        _expect_equal("materialization evidence revision", record.get("source_revision"), profile.revision)
        _expect_equal(
            "materialization evidence config SHA-256",
            record.get("config_sha256"),
            expected_config_sha256,
        )
        _expect_equal(
            "materialization evidence index SHA-256",
            record.get("index_sha256"),
            expected_index_sha256,
        )
        _expect_int("materialization evidence code bits", record.get("code_bits"), expected_code_bits)
        _expect_equal(
            "materialization evidence selected group keys",
            record.get("selected_group_keys"),
            list(expected_group_keys),
        )
        identity = _evidence_group_identity(record)
        if identity != expected_identity:
            raise ValueError(
                "materialization resume evidence byte identity or SHA-256 hash drifted"
            )
        converted = record.get("converted_vq_groups")
        existing = record.get("existing_vq_groups")
        run_groups = record.get("run_groups")
        statuses = []
        if isinstance(run_groups, list):
            statuses = [
                item.get("run_status")
                for item in run_groups
                if isinstance(item, dict)
            ]
        if converted == group_count and existing == 0 and statuses == ["converted"] * group_count:
            initial_seen = True
        if converted == 0 and existing == group_count and statuses == ["existing"] * group_count:
            resume_seen = True
        run_signatures.add(
            (
                converted,
                existing,
                record.get("expert_workers"),
                record.get("artifact_manifest_path"),
                tuple(statuses),
            )
        )
    return initial_seen and resume_seen, len(run_signatures) >= 2


def audit_glm52_reap_materialization_manifest(
    path: str | Path,
    *,
    profile: ModelProfile,
    expected_group_keys: Sequence[str],
    expected_code_bits: int,
    expected_config_sha256: str,
    expected_index_sha256: str,
    evidence_paths: Sequence[str | Path] = (),
    non_routed_artifact_bytes: int | None = None,
    whole_model_parameter_count: int | None = None,
) -> GLM52ReapMaterializationAudit:
    """Audit exact routed artifacts without inferring unavailable whole-model values."""

    if expected_code_bits not in {8, 16}:
        raise ValueError("expected code bits budget must be 8 or 16")
    expected_keys = tuple(expected_group_keys)
    if not expected_keys:
        raise ValueError("expected group selection must not be empty")
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("expected group selection contains duplicate group keys")
    all_group_keys = _canonical_profile_group_keys(profile)
    all_group_set = set(all_group_keys)
    unknown = tuple(key for key in expected_keys if key not in all_group_set)
    if unknown:
        raise ValueError(f"expected group keys are outside the profile plan: {unknown}")
    canonical_selected = tuple(key for key in all_group_keys if key in set(expected_keys))
    if expected_keys != canonical_selected:
        raise ValueError("expected group selection must use canonical profile order")
    if (non_routed_artifact_bytes is None) != (whole_model_parameter_count is None):
        raise ValueError(
            "non_routed_artifact_bytes and whole_model_parameter_count must be supplied together"
        )
    if non_routed_artifact_bytes is not None and non_routed_artifact_bytes < 0:
        raise ValueError("non_routed_artifact_bytes must be non-negative")
    if whole_model_parameter_count is not None and whole_model_parameter_count <= 0:
        raise ValueError("whole_model_parameter_count must be positive")

    root, manifest_path = _manifest_root(path)
    payload = _load_json_object(manifest_path)
    groups, scope = _validate_manifest_contract(
        payload,
        profile=profile,
        expected_group_keys=expected_keys,
        all_group_keys=all_group_keys,
        expected_code_bits=expected_code_bits,
        expected_config_sha256=expected_config_sha256,
        expected_index_sha256=expected_index_sha256,
    )

    expected_filenames = {
        _expected_group_filename(*_parse_group_key(key)) for key in expected_keys
    }
    allowed_tree_entries = {manifest_path.name, *expected_filenames}
    actual_tree_entries: set[str] = set()
    for candidate in root.rglob("*"):
        relative = str(candidate.relative_to(root))
        if candidate.is_symlink() or candidate.is_file():
            actual_tree_entries.add(relative)
        elif candidate.is_dir():
            actual_tree_entries.add(f"{relative}/")
    missing_files = sorted(allowed_tree_entries - actual_tree_entries)
    unexpected_files = sorted(actual_tree_entries - allowed_tree_entries)
    if missing_files:
        raise ValueError(f"missing expected artifact tree file(s): {missing_files}")
    if unexpected_files:
        raise ValueError(
            "unexpected artifact tree entry or layer 78 payload: "
            f"{unexpected_files}"
        )
    artifact_tree_bytes = sum(
        (root / relative).stat().st_size for relative in allowed_tree_entries
    )

    scale_estimator = str(payload["scale_estimator"])
    audited_groups = [
        _validate_group_artifact(
            root=root,
            record=record,
            group_key=key,
            profile=profile,
            expected_code_bits=expected_code_bits,
            expected_config_sha256=expected_config_sha256,
            expected_index_sha256=expected_index_sha256,
            scale_estimator=scale_estimator,
        )
        for key, record in zip(expected_keys, groups, strict=True)
    ]
    artifact_bytes = sum(int(group["artifact_bytes"]) for group in audited_groups)
    _expect_int("manifest artifact_total_bytes", payload.get("artifact_total_bytes"), artifact_bytes)
    routed_weight_count = sum(
        int(group["routed_weight_count"]) for group in audited_groups
    )
    codes_bytes = sum(int(group["codes_bytes"]) for group in audited_groups)
    scales_bytes = sum(int(group["scales_bytes"]) for group in audited_groups)
    codebook_bytes = sum(int(group["codebook_bytes"]) for group in audited_groups)
    routed_payload_bytes = codes_bytes + scales_bytes
    if routed_weight_count <= 0:
        raise ValueError("audited routed weight count must be positive")
    actual_routed_bpw = routed_payload_bytes * 8.0 / routed_weight_count
    physical_routed_bpw = artifact_bytes * 8.0 / routed_weight_count

    identity = tuple(
        (
            str(group["group_key"]),
            int(group["artifact_bytes"]),
            str(group["artifact_sha256"]),
        )
        for group in audited_groups
    )
    group_set_sha256 = _canonical_sha256(
        [
            {
                "group_key": group_key,
                "artifact_bytes": group_bytes,
                "artifact_sha256": group_sha256,
            }
            for group_key, group_bytes, group_sha256 in identity
        ]
    )
    evidence_records = _load_evidence_records(evidence_paths)
    resume_verified, byte_identity_verified = _validate_evidence(
        evidence_records,
        expected_group_keys=expected_keys,
        expected_identity=identity,
        profile=profile,
        expected_code_bits=expected_code_bits,
        expected_config_sha256=expected_config_sha256,
        expected_index_sha256=expected_index_sha256,
    )

    layer_projections: dict[int, set[str]] = {}
    for key in expected_keys:
        layer, projection = _parse_group_key(key)
        layer_projections.setdefault(layer, set()).add(projection)
    complete_layers = tuple(
        layer
        for layer, projections in sorted(layer_projections.items())
        if projections == set(GLM52_PROJECTIONS)
    )
    partial_layers = tuple(
        layer
        for layer, projections in sorted(layer_projections.items())
        if projections != set(GLM52_PROJECTIONS)
    )

    full = expected_keys == all_group_keys
    whole_values_actual = (
        full
        and non_routed_artifact_bytes is not None
        and whole_model_parameter_count is not None
    )
    whole_bytes = (
        artifact_bytes + non_routed_artifact_bytes
        if whole_values_actual and non_routed_artifact_bytes is not None
        else None
    )
    whole_bpw = (
        whole_bytes * 8.0 / whole_model_parameter_count
        if whole_bytes is not None and whole_model_parameter_count is not None
        else None
    )
    manifest_sha256 = _sha256_file(manifest_path)
    checks = {
        "exact_manifest_contract": True,
        "exact_group_selection": True,
        "exact_artifact_files": True,
        "exact_tensor_contract": True,
        "pinned_lineage": True,
        "no_layer_78": True,
        "no_dense_routed_experts": True,
        "artifact_hashes_and_sizes": True,
        "bounded_whole_model_claims_suppressed": not (
            scope == "bounded" and whole_values_actual
        ),
    }
    return GLM52ReapMaterializationAudit(
        artifact_dir=str(root),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        materialization_scope=scope,
        expected_group_keys=expected_keys,
        ready_group_keys=expected_keys,
        complete_layer_ids=complete_layers,
        partial_layer_ids=partial_layers,
        actual_routed_weight_count=routed_weight_count,
        actual_routed_codes_bytes=codes_bytes,
        actual_routed_scales_bytes=scales_bytes,
        actual_routed_codebook_bytes=codebook_bytes,
        actual_routed_payload_bytes=routed_payload_bytes,
        actual_routed_bpw=actual_routed_bpw,
        actual_artifact_bytes=artifact_bytes,
        physical_routed_bpw=physical_routed_bpw,
        artifact_tree_bytes=artifact_tree_bytes,
        group_set_sha256=group_set_sha256,
        actual_whole_model_artifact_bytes=whole_bytes,
        actual_whole_model_bpw=whole_bpw,
        whole_model_values_actual=whole_values_actual,
        full_routed_artifact_ready=full,
        dense_routed_experts=False,
        evidence_records_checked=len(evidence_records),
        resume_verified=resume_verified,
        byte_identity_verified=byte_identity_verified,
        checks=checks,
    )


__all__ = [
    "GLM52ReapMaterializationAudit",
    "GLM52ReapSourceAccounting",
    "audit_glm52_reap_materialization_manifest",
    "audit_glm52_reap_source_accounting",
]
