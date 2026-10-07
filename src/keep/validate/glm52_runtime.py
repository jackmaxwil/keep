"""Nonresident runtime-contract checks for GLM-5.2 REAP KEEP artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from keep.io.schema import codebook_metadata_for_bits
from keep.io.source_safetensors import (
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsFileHeader,
    SafetensorsTensorHeader,
    read_safetensors_file_header,
)
from ramp.models.profiles import (
    ModelProfile,
    load_profile,
    validate_profile_against_hf_config_data,
)


GLM52_ROUTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GLM52_FULL_BIND_PREFLIGHT_RECORD_TYPE = "glm52_full_bind_preflight"
GLM52_FULL_BIND_PREFLIGHT_READY = "glm52_full_bind_preflight_ready"
GLM52_FULL_BIND_PREFLIGHT_BLOCKED = "glm52_full_bind_preflight_blocked"
GLM52_NON_VQ_MANIFEST_NAME = "non-vq-manifest.json"
GLM52_NON_VQ_INDEX_NAME = "model.safetensors.index.json"
GLM52_ROUTED_MANIFEST_NAME = "conversion-manifest.json"
GLM52_CODEBOOK_TENSOR_NAME = "model.vq_codebook.e8"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U8": 1,
    "U16": 2,
    "U32": 4,
}


@dataclass(frozen=True)
class GLM52TensorContract:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GLM52FullBindPreflightReport:
    record_type: str
    schema_version: int
    preflight_status: str
    preflight_pass: bool
    blockers: tuple[str, ...]
    profile: str
    profile_path: str | None
    profile_sha256: str | None
    profile_contract_sha256: str
    model_id: str
    source_revision: str
    config_path: str
    config_sha256: str
    source_index_path: str
    source_index_sha256: str
    non_vq_artifact_dir: str
    routed_artifact_dir: str
    non_vq_tensor_count: int
    non_vq_runtime_target_count: int
    non_vq_header_file_count: int
    kv_b_source_tensor_count: int
    kv_b_runtime_target_count: int
    expected_routed_group_keys: tuple[str, ...]
    present_routed_group_keys: tuple[str, ...]
    missing_routed_group_keys: tuple[str, ...]
    routed_header_file_count: int
    header_only: bool
    tensor_payloads_read: bool
    payload_hashes_verified: bool
    full_model_constructed: bool
    dense_routed_experts: bool
    production_binding_proven: bool
    production_generation_proven: bool

    @property
    def expected_routed_group_count(self) -> int:
        return len(self.expected_routed_group_keys)

    @property
    def present_routed_group_count(self) -> int:
        return len(self.present_routed_group_keys)

    @property
    def missing_routed_group_count(self) -> int:
        return len(self.missing_routed_group_keys)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        payload["expected_routed_group_keys"] = list(
            self.expected_routed_group_keys
        )
        payload["present_routed_group_keys"] = list(self.present_routed_group_keys)
        payload["missing_routed_group_keys"] = list(self.missing_routed_group_keys)
        payload["expected_routed_group_count"] = self.expected_routed_group_count
        payload["present_routed_group_count"] = self.present_routed_group_count
        payload["missing_routed_group_count"] = self.missing_routed_group_count
        return payload


def _required_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"GLM52 config {name} must be a positive integer")
    return value


def _layer_schedule(
    config: Mapping[str, Any],
    *,
    name: str,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    num_layers = _required_int(config, "num_hidden_layers")
    raw = config.get(name)
    if not isinstance(raw, list) or len(raw) != num_layers:
        raise ValueError(f"GLM52 config {name} must contain {num_layers} entries")
    values = tuple(str(value) for value in raw)
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"GLM52 config {name} has unsupported entries: {invalid}")
    return values


def expected_glm52_non_vq_source_schema(
    config: Mapping[str, Any],
) -> dict[str, GLM52TensorContract]:
    """Return the exact source-side non-routed tensor contract without a model."""

    if config.get("model_type") != "glm_moe_dsa":
        raise ValueError("GLM52 non-VQ schema requires model_type='glm_moe_dsa'")
    if config.get("attention_bias") is not False:
        raise ValueError("GLM52 non-VQ schema currently requires attention_bias=false")

    num_layers = _required_int(config, "num_hidden_layers")
    hidden = _required_int(config, "hidden_size")
    vocab = _required_int(config, "vocab_size")
    q_lora = _required_int(config, "q_lora_rank")
    kv_lora = _required_int(config, "kv_lora_rank")
    qk_rope = _required_int(config, "qk_rope_head_dim")
    qk_nope = _required_int(config, "qk_nope_head_dim")
    v_head = _required_int(config, "v_head_dim")
    attention_heads = _required_int(config, "num_attention_heads")
    index_heads = _required_int(config, "index_n_heads")
    index_head_dim = _required_int(config, "index_head_dim")
    intermediate = _required_int(config, "intermediate_size")
    moe_intermediate = _required_int(config, "moe_intermediate_size")
    routed_experts = _required_int(config, "n_routed_experts")
    shared_experts = _required_int(config, "n_shared_experts")
    mlp_types = _layer_schedule(
        config,
        name="mlp_layer_types",
        allowed=frozenset({"dense", "sparse"}),
    )
    indexer_types = _layer_schedule(
        config,
        name="indexer_types",
        allowed=frozenset({"full", "shared"}),
    )

    schema: dict[str, GLM52TensorContract] = {
        "model.embed_tokens.weight": GLM52TensorContract("BF16", (vocab, hidden)),
        "model.norm.weight": GLM52TensorContract("BF16", (hidden,)),
        "lm_head.weight": GLM52TensorContract("BF16", (vocab, hidden)),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        layer_schema: dict[str, GLM52TensorContract] = {
            f"{prefix}.input_layernorm.weight": GLM52TensorContract("BF16", (hidden,)),
            f"{prefix}.post_attention_layernorm.weight": GLM52TensorContract(
                "BF16", (hidden,)
            ),
            f"{prefix}.self_attn.q_a_proj.weight": GLM52TensorContract(
                "BF16", (q_lora, hidden)
            ),
            f"{prefix}.self_attn.q_a_layernorm.weight": GLM52TensorContract(
                "BF16", (q_lora,)
            ),
            f"{prefix}.self_attn.q_b_proj.weight": GLM52TensorContract(
                "BF16", (attention_heads * (qk_nope + qk_rope), q_lora)
            ),
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": GLM52TensorContract(
                "BF16", (kv_lora + qk_rope, hidden)
            ),
            f"{prefix}.self_attn.kv_a_layernorm.weight": GLM52TensorContract(
                "BF16", (kv_lora,)
            ),
            f"{prefix}.self_attn.kv_b_proj.weight": GLM52TensorContract(
                "BF16", (attention_heads * (qk_nope + v_head), kv_lora)
            ),
            f"{prefix}.self_attn.o_proj.weight": GLM52TensorContract(
                "BF16", (hidden, attention_heads * v_head)
            ),
        }
        if indexer_types[layer] == "full":
            layer_schema.update(
                {
                    f"{prefix}.self_attn.indexer.wq_b.weight": GLM52TensorContract(
                        "BF16", (index_heads * index_head_dim, q_lora)
                    ),
                    f"{prefix}.self_attn.indexer.wk.weight": GLM52TensorContract(
                        "BF16", (index_head_dim, hidden)
                    ),
                    f"{prefix}.self_attn.indexer.k_norm.weight": GLM52TensorContract(
                        "BF16", (index_head_dim,)
                    ),
                    f"{prefix}.self_attn.indexer.k_norm.bias": GLM52TensorContract(
                        "BF16", (index_head_dim,)
                    ),
                    f"{prefix}.self_attn.indexer.weights_proj.weight": GLM52TensorContract(
                        "BF16", (index_heads, hidden)
                    ),
                }
            )
        if mlp_types[layer] == "dense":
            layer_schema.update(
                {
                    f"{prefix}.mlp.gate_proj.weight": GLM52TensorContract(
                        "BF16", (intermediate, hidden)
                    ),
                    f"{prefix}.mlp.up_proj.weight": GLM52TensorContract(
                        "BF16", (intermediate, hidden)
                    ),
                    f"{prefix}.mlp.down_proj.weight": GLM52TensorContract(
                        "BF16", (hidden, intermediate)
                    ),
                }
            )
        else:
            shared_size = shared_experts * moe_intermediate
            layer_schema.update(
                {
                    f"{prefix}.mlp.gate.weight": GLM52TensorContract(
                        "BF16", (routed_experts, hidden)
                    ),
                    f"{prefix}.mlp.gate.e_score_correction_bias": GLM52TensorContract(
                        "F32", (routed_experts,)
                    ),
                    f"{prefix}.mlp.shared_experts.gate_proj.weight": GLM52TensorContract(
                        "BF16", (shared_size, hidden)
                    ),
                    f"{prefix}.mlp.shared_experts.up_proj.weight": GLM52TensorContract(
                        "BF16", (shared_size, hidden)
                    ),
                    f"{prefix}.mlp.shared_experts.down_proj.weight": GLM52TensorContract(
                        "BF16", (hidden, shared_size)
                    ),
                }
            )
        overlap = set(schema).intersection(layer_schema)
        if overlap:
            raise AssertionError(f"duplicate GLM52 schema tensor names: {sorted(overlap)}")
        schema.update(layer_schema)
    return dict(sorted(schema.items()))


def expected_glm52_non_vq_runtime_schema(
    config: Mapping[str, Any],
) -> dict[str, GLM52TensorContract]:
    """Translate source ``kv_b_proj`` tensors into their two runtime targets."""

    runtime = expected_glm52_non_vq_source_schema(config)
    num_layers = _required_int(config, "num_hidden_layers")
    attention_heads = _required_int(config, "num_attention_heads")
    kv_lora = _required_int(config, "kv_lora_rank")
    qk_nope = _required_int(config, "qk_nope_head_dim")
    v_head = _required_int(config, "v_head_dim")
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}.self_attn"
        source_name = f"{prefix}.kv_b_proj.weight"
        if runtime.pop(source_name, None) is None:
            raise ValueError(f"GLM52 source schema is missing {source_name}")
        runtime[f"{prefix}.embed_q.weight"] = GLM52TensorContract(
            "BF16",
            (attention_heads, kv_lora, qk_nope),
        )
        runtime[f"{prefix}.unembed_out.weight"] = GLM52TensorContract(
            "BF16",
            (attention_heads, v_head, kv_lora),
        )
    return dict(sorted(runtime.items()))


def expected_glm52_routed_group_keys(
    profile: ModelProfile,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    mismatches = validate_profile_against_hf_config_data(profile, config)
    if mismatches:
        raise ValueError(f"GLM52 profile/config mismatch: {mismatches}")
    mlp_types = _layer_schedule(
        config,
        name="mlp_layer_types",
        allowed=frozenset({"dense", "sparse"}),
    )
    sparse_layers = tuple(
        layer for layer, layer_type in enumerate(mlp_types) if layer_type == "sparse"
    )
    if len(sparse_layers) != profile.num_sparse_layers:
        raise ValueError(
            "GLM52 sparse-layer count does not match the model profile: "
            f"config={len(sparse_layers)} profile={profile.num_sparse_layers}"
        )
    if sparse_layers and sparse_layers[0] != profile.first_sparse_layer:
        raise ValueError(
            "GLM52 first sparse layer does not match the model profile: "
            f"config={sparse_layers[0]} profile={profile.first_sparse_layer}"
        )
    return tuple(
        f"{layer}:{projection}"
        for layer in sparse_layers
        for projection in GLM52_ROUTED_PROJECTIONS
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return payload


def _metadata_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return resolved


def _artifact_root(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return resolved


def _sha256_metadata_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}")


def _expect_int(label: str, actual: object, expected: int) -> None:
    if type(actual) is not int or actual != expected:
        raise ValueError(f"{label} must be {expected}, found {actual!r}")


def _expect_nonnegative_int(label: str, actual: object) -> int:
    if type(actual) is not int or actual < 0:
        raise ValueError(f"{label} must be a non-negative integer, found {actual!r}")
    return actual


def _expect_sha256(label: str, actual: object) -> str:
    if not isinstance(actual, str) or _SHA256_RE.fullmatch(actual) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return actual


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _tensor_nbytes(tensor: SafetensorsTensorHeader) -> int:
    dtype_bytes = _DTYPE_BYTES.get(tensor.dtype)
    if dtype_bytes is None:
        raise ValueError(f"unsupported safetensors dtype {tensor.dtype!r}")
    return _element_count(tensor.shape) * dtype_bytes


def _validate_header_physical_extent(
    path: Path,
    header: SafetensorsFileHeader,
) -> None:
    expected_file_bytes = header.payload_offset + header.declared_payload_bytes
    actual_file_bytes = path.stat().st_size
    if expected_file_bytes != actual_file_bytes:
        raise ValueError(
            f"{path.name} header-declared physical extent must be "
            f"{expected_file_bytes}, found {actual_file_bytes}"
        )
    cursor = 0
    for name, tensor in sorted(
        header.tensors.items(),
        key=lambda item: item[1].data_offsets,
    ):
        start, end = tensor.data_offsets
        if start != cursor or end < start:
            raise ValueError(
                f"{path.name} tensor {name!r} has non-contiguous data offsets "
                f"{tensor.data_offsets}, expected start {cursor}"
            )
        expected_bytes = _tensor_nbytes(tensor)
        if end - start != expected_bytes:
            raise ValueError(
                f"{path.name} tensor {name!r} header extent must be "
                f"{expected_bytes} bytes, found {end - start}"
            )
        cursor = end


def _safe_artifact_file(root: Path, filename: str, *, label: str) -> Path:
    relative = Path(filename)
    if not filename or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} has unsafe relative path {filename!r}")
    path = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks: {filename!r}")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root: {filename!r}") from error
    return path


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path.name} is not a valid safetensors file")
        header_size = struct.unpack("<Q", raw_size)[0]
        if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"{path.name} has invalid safetensors header size")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"{path.name} has a truncated safetensors header")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} has invalid safetensors header JSON") from error
    if not isinstance(header, dict):
        raise ValueError(f"{path.name} safetensors header must be an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError(f"{path.name} safetensors metadata must be string-valued")
    return dict(metadata)


def _validate_lineage(
    payload: Mapping[str, object],
    *,
    label: str,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    source_index_sha256: str,
) -> None:
    _expect_equal(f"{label} profile", payload.get("profile"), profile.name)
    _expect_equal(f"{label} model_id", payload.get("model_id"), model_id)
    _expect_equal(
        f"{label} source revision",
        payload.get("source_revision"),
        revision,
    )
    _expect_equal(
        f"{label} config SHA-256",
        payload.get("config_sha256"),
        config_sha256,
    )
    _expect_equal(
        f"{label} source index SHA-256",
        payload.get("index_sha256"),
        source_index_sha256,
    )


def _validate_non_vq_package(
    root: Path,
    *,
    schema: Mapping[str, GLM52TensorContract],
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    source_index_sha256: str,
    source_weight_map: Mapping[str, str],
) -> tuple[int, int]:
    manifest_path = _safe_artifact_file(
        root,
        GLM52_NON_VQ_MANIFEST_NAME,
        label="non-VQ manifest",
    )
    package_index_path = _safe_artifact_file(
        root,
        GLM52_NON_VQ_INDEX_NAME,
        label="non-VQ package index",
    )
    manifest = _load_json_object(manifest_path, label="non-VQ manifest")
    package_index = _load_json_object(
        package_index_path,
        label="non-VQ package index",
    )

    _expect_int("non-VQ manifest schema_version", manifest.get("schema_version"), 1)
    _expect_equal(
        "non-VQ manifest record_type",
        manifest.get("record_type"),
        "glm52_non_vq_package_manifest",
    )
    _expect_equal(
        "non-VQ manifest pack_status",
        manifest.get("pack_status"),
        "glm52_non_vq_package_ready",
    )
    _expect_equal(
        "non-VQ manifest production_ready",
        manifest.get("production_ready"),
        True,
    )
    _expect_equal(
        "non-VQ manifest source_authority",
        manifest.get("source_authority"),
        "pinned_huggingface_lfs_v1",
    )
    _expect_equal(
        "non-VQ manifest selection_policy",
        manifest.get("selection_policy"),
        "main_model_non_routed_bf16_f32_v1",
    )
    _expect_equal(
        "non-VQ manifest copy_mode",
        manifest.get("copy_mode"),
        "raw_safetensors_byte_ranges_v1",
    )
    _validate_lineage(
        manifest,
        label="non-VQ manifest",
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
    )
    _expect_equal(
        "non-VQ manifest source index filename",
        manifest.get("index_path"),
        "model.safetensors.index.json",
    )
    for digest_name in (
        "source_blob_inventory_sha256",
        "source_inventory_sha256",
        "plan_sha256",
        "package_set_sha256",
    ):
        _expect_sha256(
            f"non-VQ manifest {digest_name}",
            manifest.get(digest_name),
        )
    _expect_equal(
        "non-VQ manifest package index SHA-256",
        manifest.get("package_index_sha256"),
        _sha256_metadata_file(package_index_path),
    )
    _expect_nonnegative_int(
        "non-VQ excluded routed tensor count",
        manifest.get("excluded_routed_tensor_count"),
    )
    _expect_nonnegative_int(
        "non-VQ excluded MTP tensor count",
        manifest.get("excluded_mtp_tensor_count"),
    )
    _expect_int(
        "non-VQ excluded runtime VQ tensor count",
        manifest.get("excluded_runtime_vq_tensor_count"),
        0,
    )

    weight_map = package_index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError("non-VQ package index weight_map must be string-valued")
    expected_names = set(schema)
    actual_names = set(weight_map)
    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    if missing_names or unexpected_names:
        raise ValueError(
            "non-VQ package index tensor schema mismatch: "
            f"missing_non_vq_tensors={missing_names}; "
            f"unexpected_non_vq_tensors={unexpected_names}"
        )

    raw_records = manifest.get("tensors")
    if not isinstance(raw_records, list) or any(
        not isinstance(record, dict) for record in raw_records
    ):
        raise ValueError("non-VQ manifest tensors must be a list of objects")
    record_names = [record.get("name") for record in raw_records]
    if record_names != sorted(expected_names):
        raise ValueError(
            "non-VQ manifest tensor records must exactly match lexical source schema"
        )
    records = {str(record["name"]): record for record in raw_records}

    dtype_counts: dict[str, int] = {}
    total_parameters = 0
    total_payload_bytes = 0
    shard_expected_names: dict[str, set[str]] = {}
    for name in sorted(schema):
        contract = schema[name]
        record = records[name]
        _expect_equal(f"non-VQ tensor {name} dtype", record.get("dtype"), contract.dtype)
        _expect_equal(
            f"non-VQ tensor {name} shape",
            record.get("shape"),
            list(contract.shape),
        )
        parameters = _element_count(contract.shape)
        payload_bytes = parameters * _DTYPE_BYTES[contract.dtype]
        _expect_int(
            f"non-VQ tensor {name} parameter_count",
            record.get("parameter_count"),
            parameters,
        )
        _expect_int(
            f"non-VQ tensor {name} payload_bytes",
            record.get("payload_bytes"),
            payload_bytes,
        )
        _expect_sha256(
            f"non-VQ tensor {name} payload_sha256",
            record.get("payload_sha256"),
        )
        source_shard = record.get("source_shard")
        if not isinstance(source_shard, str) or not source_shard:
            raise ValueError(f"non-VQ tensor {name} source_shard must be a string")
        if source_weight_map.get(name) != source_shard:
            raise ValueError(
                f"non-VQ tensor {name} source_shard must match pinned source index"
            )
        source_offsets = record.get("source_offsets")
        if (
            not isinstance(source_offsets, list)
            or len(source_offsets) != 2
            or any(type(offset) is not int for offset in source_offsets)
            or source_offsets[0] < 0
            or source_offsets[1] - source_offsets[0] != payload_bytes
        ):
            raise ValueError(f"non-VQ tensor {name} source_offsets are invalid")
        output_shard = record.get("output_shard")
        _expect_equal(
            f"non-VQ tensor {name} output_shard",
            output_shard,
            weight_map[name],
        )
        if not isinstance(output_shard, str):
            raise ValueError(f"non-VQ tensor {name} output_shard must be a string")
        shard_expected_names.setdefault(output_shard, set()).add(name)
        dtype_counts[contract.dtype] = dtype_counts.get(contract.dtype, 0) + 1
        total_parameters += parameters
        total_payload_bytes += payload_bytes

    _expect_int(
        "non-VQ retained_tensor_count",
        manifest.get("retained_tensor_count"),
        len(schema),
    )
    _expect_int(
        "non-VQ parameter_count",
        manifest.get("parameter_count"),
        total_parameters,
    )
    _expect_int(
        "non-VQ tensor_payload_bytes",
        manifest.get("tensor_payload_bytes"),
        total_payload_bytes,
    )
    _expect_equal(
        "non-VQ dtype_tensor_counts",
        manifest.get("dtype_tensor_counts"),
        dict(sorted(dtype_counts.items())),
    )

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or any(
        not isinstance(record, dict) for record in raw_shards
    ):
        raise ValueError("non-VQ manifest shards must be a list of objects")
    shard_records = {
        str(record.get("filename")): record
        for record in raw_shards
        if isinstance(record.get("filename"), str)
    }
    expected_shards = set(shard_expected_names)
    if set(shard_records) != expected_shards or len(shard_records) != len(raw_shards):
        raise ValueError("non-VQ manifest shard inventory is not exact")
    actual_shard_files = {
        str(path.relative_to(root)) for path in root.rglob("*.safetensors")
    }
    unexpected_shard_files = sorted(actual_shard_files - expected_shards)
    if unexpected_shard_files:
        raise ValueError(
            "non-VQ artifact contains unindexed files: "
            f"unexpected_non_vq_safetensors_files={unexpected_shard_files}"
        )
    _expect_int(
        "non-VQ shard_count",
        manifest.get("shard_count"),
        len(expected_shards),
    )
    header_file_count = 0
    for shard_name in sorted(expected_shards):
        shard_path = _safe_artifact_file(
            root,
            shard_name,
            label="non-VQ safetensors shard",
        )
        header = read_safetensors_file_header(shard_path)
        _validate_header_physical_extent(shard_path, header)
        header_file_count += 1
        expected_for_shard = shard_expected_names[shard_name]
        if set(header.tensors) != expected_for_shard:
            raise ValueError(
                f"non-VQ shard {shard_name} header tensor inventory is not exact"
            )
        shard_payload_bytes = 0
        for name in sorted(expected_for_shard):
            tensor = header.tensors[name]
            contract = schema[name]
            if tensor.dtype != contract.dtype:
                raise ValueError(
                    f"non-VQ header tensor {name} dtype must be "
                    f"{contract.dtype}, found {tensor.dtype}"
                )
            if tensor.shape != contract.shape:
                raise ValueError(
                    f"non-VQ header tensor {name} shape must be "
                    f"{contract.shape}, found {tensor.shape}"
                )
            _expect_equal(
                f"non-VQ tensor {name} output_offsets",
                records[name].get("output_offsets"),
                list(tensor.data_offsets),
            )
            shard_payload_bytes += _tensor_nbytes(tensor)
        shard_record = shard_records[shard_name]
        _expect_int(
            f"non-VQ shard {shard_name} file_bytes",
            shard_record.get("file_bytes"),
            shard_path.stat().st_size,
        )
        _expect_sha256(
            f"non-VQ shard {shard_name} file_sha256",
            shard_record.get("file_sha256"),
        )
        _expect_sha256(
            f"non-VQ shard {shard_name} tensor_inventory_sha256",
            shard_record.get("tensor_inventory_sha256"),
        )
        _expect_int(
            f"non-VQ shard {shard_name} tensor_count",
            shard_record.get("tensor_count"),
            len(expected_for_shard),
        )
        _expect_int(
            f"non-VQ shard {shard_name} tensor_payload_bytes",
            shard_record.get("tensor_payload_bytes"),
            shard_payload_bytes,
        )

    metadata = package_index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("non-VQ package index metadata must be an object")
    _expect_int(
        "non-VQ package index total_size",
        metadata.get("total_size"),
        total_payload_bytes,
    )
    return len(schema), header_file_count


def _profile_group_size(profile: ModelProfile, projection: str) -> int:
    key = projection.removesuffix("_proj")
    value = profile.group_size_policy.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"profile group size for {projection} must be positive")
    return value


def _projection_dims(profile: ModelProfile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    if projection == "down_proj":
        return profile.moe_intermediate_size, profile.hidden_size
    raise ValueError(f"unsupported routed projection {projection!r}")


def _parse_group_key(group_key: str) -> tuple[int, str]:
    try:
        raw_layer, projection = group_key.split(":", maxsplit=1)
        layer = int(raw_layer)
    except (AttributeError, ValueError) as error:
        raise ValueError(f"invalid routed group key {group_key!r}") from error
    if projection not in GLM52_ROUTED_PROJECTIONS:
        raise ValueError(f"invalid routed projection in group key {group_key!r}")
    return layer, projection


def _validate_quantization_metadata(
    path: Path,
    *,
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    source_index_sha256: str,
    code_bits: int,
    group_size: int,
    scale_estimator: str,
) -> None:
    metadata = _read_safetensors_metadata(path)
    raw = metadata.get("quantization_config")
    if raw is None:
        raise ValueError(f"{path.name} is missing quantization_config metadata")
    try:
        quantization = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path.name} has invalid quantization_config metadata"
        ) from error
    if not isinstance(quantization, dict):
        raise ValueError(f"{path.name} quantization_config must be an object")
    _expect_equal(
        f"{path.name} quant_method",
        quantization.get("quant_method"),
        "mlx_vq_e8",
    )
    _expect_int(
        f"{path.name} quantization version",
        quantization.get("version"),
        1,
    )
    _expect_int(
        f"{path.name} code bits",
        quantization.get("default_code_bits"),
        code_bits,
    )
    _expect_int(
        f"{path.name} group size",
        quantization.get("default_group_size"),
        group_size,
    )
    codebook = quantization.get("codebook")
    if not isinstance(codebook, dict):
        raise ValueError(f"{path.name} codebook metadata must be an object")
    expected_name, expected_sha256 = codebook_metadata_for_bits(code_bits)
    _expect_equal(f"{path.name} codebook name", codebook.get("name"), expected_name)
    _expect_equal(f"{path.name} codebook dtype", codebook.get("dtype"), "uint32")
    _expect_int(f"{path.name} codebook entries", codebook.get("entries"), 256)
    _expect_equal(
        f"{path.name} codebook SHA-256",
        codebook.get("sha256"),
        expected_sha256,
    )
    policy = quantization.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{path.name} quantization policy must be an object")
    expected_policy = {
        "scale_estimator": scale_estimator,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "source_model_id": model_id,
        "source_revision": revision,
        "source_config_sha256": config_sha256,
        "source_index_sha256": source_index_sha256,
        "source_profile": profile.name,
        "decoded_expert_working_set": "one_per_worker",
    }
    for name, expected in expected_policy.items():
        _expect_equal(
            f"{path.name} policy {name}",
            policy.get(name),
            expected,
        )


def _validate_routed_group(
    root: Path,
    *,
    group_key: str,
    record: Mapping[str, object],
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    source_index_sha256: str,
    code_bits: int,
    scale_estimator: str,
) -> tuple[int, int]:
    layer, projection = _parse_group_key(group_key)
    if layer >= profile.num_layers:
        raise ValueError(f"routed group {group_key} includes forbidden MTP layer")
    filename = f"layer-{layer:05d}-{projection}.safetensors"
    path = _safe_artifact_file(root, filename, label="routed VQ artifact")
    recorded_path = record.get("artifact_path")
    if not isinstance(recorded_path, str) or Path(recorded_path).name != filename:
        raise ValueError(f"routed group {group_key} artifact_path must name {filename}")
    _expect_int(
        f"routed group {group_key} artifact_bytes",
        record.get("artifact_bytes"),
        path.stat().st_size,
    )
    _expect_sha256(
        f"routed group {group_key} artifact_sha256",
        record.get("artifact_sha256"),
    )
    _expect_equal(f"routed group {group_key} status", record.get("status"), "ready")
    _expect_int(
        f"routed group {group_key} expert_count",
        record.get("expert_count"),
        profile.num_experts,
    )

    input_dims, output_dims = _projection_dims(profile, projection)
    group_size = _profile_group_size(profile, projection)
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    codes_name = f"{prefix}.codes"
    scales_name = f"{prefix}.scales"
    codes_shape = (profile.num_experts, output_dims, input_dims // 8)
    scales_shape = (
        profile.num_experts,
        output_dims,
        input_dims // group_size,
    )
    codes_dtype = "U8" if code_bits == 8 else "U16"
    manifest_codes_dtype = "uint8" if code_bits == 8 else "uint16"

    header = read_safetensors_file_header(path)
    _validate_header_physical_extent(path, header)
    expected_names = {codes_name, scales_name, GLM52_CODEBOOK_TENSOR_NAME}
    if set(header.tensors) != expected_names:
        unexpected = sorted(set(header.tensors) - expected_names)
        if any(name.endswith(".weight") for name in unexpected):
            raise ValueError(
                f"{filename} contains forbidden dense routed tensor(s): {unexpected}"
            )
        raise ValueError(f"{filename} header tensor inventory is not exact")
    codes = header.tensors[codes_name]
    scales = header.tensors[scales_name]
    codebook = header.tensors[GLM52_CODEBOOK_TENSOR_NAME]
    if codes.dtype != codes_dtype:
        raise ValueError(
            f"{filename} codes dtype must be {codes_dtype}, found {codes.dtype}"
        )
    if codes.shape != codes_shape:
        raise ValueError(
            f"{filename} codes shape must be {codes_shape}, found {codes.shape}"
        )
    if scales.dtype != "F16":
        raise ValueError(f"{filename} scales dtype must be F16, found {scales.dtype}")
    if scales.shape != scales_shape:
        raise ValueError(
            f"{filename} scales shape must be {scales_shape}, found {scales.shape}"
        )
    if codebook.dtype != "U32" or codebook.shape != (256,):
        raise ValueError(f"{filename} codebook must be U32 with shape (256,)")

    _expect_equal(
        f"routed group {group_key} manifest codes_name",
        record.get("codes_name"),
        codes_name,
    )
    _expect_equal(
        f"routed group {group_key} manifest codes_dtype",
        record.get("codes_dtype"),
        manifest_codes_dtype,
    )
    _expect_equal(
        f"routed group {group_key} manifest codes_shape",
        record.get("codes_shape"),
        list(codes_shape),
    )
    _expect_equal(
        f"routed group {group_key} manifest scales_name",
        record.get("scales_name"),
        scales_name,
    )
    _expect_equal(
        f"routed group {group_key} manifest scales_dtype",
        record.get("scales_dtype"),
        "float16",
    )
    _expect_equal(
        f"routed group {group_key} manifest scales_shape",
        record.get("scales_shape"),
        list(scales_shape),
    )
    decoded_expert_bytes = input_dims * output_dims * 4
    _expect_int(
        f"routed group {group_key} decoded_expert_bytes",
        record.get("decoded_expert_bytes"),
        decoded_expert_bytes,
    )
    _expect_int(
        f"routed group {group_key} source_bundles",
        record.get("source_bundles"),
        profile.num_experts,
    )
    _expect_int(
        f"routed group {group_key} source_bundle_members",
        record.get("source_bundle_members"),
        profile.num_experts * 3,
    )
    source_shards = record.get("source_shards")
    if not isinstance(source_shards, list) or not source_shards or any(
        not isinstance(shard, str) or not shard for shard in source_shards
    ):
        raise ValueError(f"routed group {group_key} source_shards must be non-empty")
    cross_shard = _expect_nonnegative_int(
        f"routed group {group_key} cross_shard_bundle_count",
        record.get("cross_shard_bundle_count"),
    )
    if cross_shard > profile.num_experts:
        raise ValueError(
            f"routed group {group_key} cross_shard_bundle_count exceeds experts"
        )
    _validate_quantization_metadata(
        path,
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
        code_bits=code_bits,
        group_size=group_size,
        scale_estimator=scale_estimator,
    )
    return path.stat().st_size, decoded_expert_bytes


def _validate_routed_package(
    root: Path,
    *,
    expected_group_keys: tuple[str, ...],
    profile: ModelProfile,
    model_id: str,
    revision: str,
    config_sha256: str,
    source_index_sha256: str,
) -> tuple[tuple[str, ...], int]:
    manifest_path = _safe_artifact_file(
        root,
        GLM52_ROUTED_MANIFEST_NAME,
        label="routed manifest",
    )
    manifest = _load_json_object(manifest_path, label="routed manifest")
    _expect_int("routed manifest schema_version", manifest.get("schema_version"), 1)
    _expect_equal(
        "routed manifest record_type",
        manifest.get("record_type"),
        "glm52_modelopt_nvfp4_materialization_manifest",
    )
    _expect_equal(
        "routed manifest materialization_status",
        manifest.get("materialization_status"),
        "glm52_modelopt_nvfp4_groups_ready",
    )
    _validate_lineage(
        manifest,
        label="routed manifest",
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
    )
    _expect_equal(
        "routed manifest source_weight_encoding",
        manifest.get("source_weight_encoding"),
        "modelopt_nvfp4",
    )
    _expect_equal(
        "routed manifest source_decoder",
        manifest.get("source_decoder"),
        "modelopt_nvfp4_v1",
    )
    _expect_equal(
        "routed manifest working_set_policy",
        manifest.get("working_set_policy"),
        "one_decoded_expert_per_worker_v1",
    )
    _expect_equal(
        "routed manifest materialization_blocked",
        manifest.get("materialization_blocked"),
        False,
    )
    _expect_equal(
        "routed manifest materialization_blockers",
        manifest.get("materialization_blockers"),
        [],
    )
    _expect_equal(
        "routed manifest dense_checkpoint_written",
        manifest.get("dense_checkpoint_written"),
        False,
    )

    raw_selected = manifest.get("selected_group_keys")
    if not isinstance(raw_selected, list) or any(
        not isinstance(key, str) for key in raw_selected
    ):
        raise ValueError("routed manifest selected_group_keys must be strings")
    present_keys = tuple(raw_selected)
    if len(set(present_keys)) != len(present_keys):
        raise ValueError("routed manifest selected_group_keys contains duplicates")
    expected_set = set(expected_group_keys)
    unknown = tuple(key for key in present_keys if key not in expected_set)
    if unknown:
        raise ValueError(f"routed manifest contains non-profile groups: {unknown}")
    canonical_present = tuple(
        key for key in expected_group_keys if key in set(present_keys)
    )
    if present_keys != canonical_present:
        raise ValueError("routed manifest group selection is not in canonical order")
    expected_artifact_files = {
        f"layer-{_parse_group_key(key)[0]:05d}-{_parse_group_key(key)[1]}.safetensors"
        for key in present_keys
    }
    actual_artifact_files = {
        str(path.relative_to(root)) for path in root.rglob("*.safetensors")
    }
    unexpected_artifact_files = sorted(
        actual_artifact_files - expected_artifact_files
    )
    if unexpected_artifact_files:
        raise ValueError(
            "routed artifact contains unmanifested files: "
            f"unexpected_routed_safetensors_files={unexpected_artifact_files}"
        )
    missing_count = len(expected_group_keys) - len(present_keys)
    full = missing_count == 0
    _expect_equal(
        "routed manifest materialization_scope",
        manifest.get("materialization_scope"),
        "full" if full else "bounded",
    )
    _expect_equal(
        "routed manifest full_group_coverage",
        manifest.get("full_group_coverage"),
        full,
    )
    _expect_int(
        "routed manifest planned_vq_groups",
        manifest.get("planned_vq_groups"),
        len(expected_group_keys),
    )
    _expect_int(
        "routed manifest selected_vq_groups",
        manifest.get("selected_vq_groups"),
        len(present_keys),
    )
    _expect_int(
        "routed manifest ready_vq_groups",
        manifest.get("ready_vq_groups"),
        len(present_keys),
    )
    _expect_int(
        "routed manifest skipped_vq_groups",
        manifest.get("skipped_vq_groups"),
        missing_count,
    )

    code_bits = profile.default_code_bits
    if code_bits not in {8, 16}:
        raise ValueError("GLM52 routed preflight supports code bits 8 or 16")
    _expect_int("routed manifest code_bits", manifest.get("code_bits"), code_bits)
    group_sizes = {
        _profile_group_size(profile, projection)
        for projection in GLM52_ROUTED_PROJECTIONS
    }
    if len(group_sizes) != 1:
        raise ValueError("routed artifact manifest requires one group size")
    group_size = next(iter(group_sizes))
    _expect_int(
        "routed manifest group_size",
        manifest.get("group_size"),
        group_size,
    )
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(code_bits)
    _expect_equal(
        "routed manifest codebook_name",
        manifest.get("codebook_name"),
        codebook_name,
    )
    _expect_equal(
        "routed manifest codebook_sha256",
        manifest.get("codebook_sha256"),
        codebook_sha256,
    )
    scale_estimator = manifest.get("scale_estimator")
    if scale_estimator not in {"max_abs", "percentile_99"}:
        raise ValueError("routed manifest scale_estimator is unsupported")

    raw_groups = manifest.get("groups")
    if not isinstance(raw_groups, list) or any(
        not isinstance(record, dict) for record in raw_groups
    ):
        raise ValueError("routed manifest groups must be a list of objects")
    group_record_keys = tuple(
        f"{record.get('layer')}:{record.get('projection')}"
        for record in raw_groups
    )
    if group_record_keys != present_keys:
        raise ValueError("routed manifest group records do not match selected keys")
    artifact_total_bytes = 0
    decoded_bytes: list[int] = []
    for group_key, record in zip(present_keys, raw_groups):
        artifact_bytes, decoded_expert_bytes = _validate_routed_group(
            root,
            group_key=group_key,
            record=record,
            profile=profile,
            model_id=model_id,
            revision=revision,
            config_sha256=config_sha256,
            source_index_sha256=source_index_sha256,
            code_bits=code_bits,
            scale_estimator=str(scale_estimator),
        )
        artifact_total_bytes += artifact_bytes
        decoded_bytes.append(decoded_expert_bytes)
    _expect_int(
        "routed manifest artifact_total_bytes",
        manifest.get("artifact_total_bytes"),
        artifact_total_bytes,
    )
    _expect_int(
        "routed manifest peak_decoded_expert_bytes",
        manifest.get("peak_decoded_expert_bytes"),
        max(decoded_bytes, default=0),
    )
    _expect_int(
        "routed manifest source_bundles",
        manifest.get("source_bundles"),
        len(present_keys) * profile.num_experts,
    )
    _expect_int(
        "routed manifest source_bundle_members",
        manifest.get("source_bundle_members"),
        len(present_keys) * profile.num_experts * 3,
    )
    return present_keys, len(present_keys)


def preflight_glm52_full_bind(
    *,
    profile: ModelProfile,
    profile_path: str | Path | None = None,
    config_path: str | Path,
    source_index_path: str | Path,
    non_vq_artifact_dir: str | Path,
    routed_artifact_dir: str | Path,
    model_id: str,
    revision: str,
) -> GLM52FullBindPreflightReport:
    """Prove static full-bind readiness without reading tensor payloads or a model."""

    profile_input = (
        None
        if profile_path is None
        else _metadata_file(profile_path, label="GLM52 profile")
    )
    if profile_input is not None and load_profile(profile_input) != profile:
        raise ValueError("GLM52 profile object does not match profile_path content")
    config_input = _metadata_file(config_path, label="GLM52 config")
    source_index_input = _metadata_file(
        source_index_path,
        label="GLM52 source index",
    )
    non_vq_root = _artifact_root(
        non_vq_artifact_dir,
        label="GLM52 non-VQ artifact",
    )
    routed_root = _artifact_root(
        routed_artifact_dir,
        label="GLM52 routed artifact",
    )
    config = _load_json_object(config_input, label="GLM52 config")
    source_index = _load_json_object(source_index_input, label="GLM52 source index")
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in source_weight_map.items()
    ):
        raise ValueError("GLM52 source index weight_map must be an object")
    if profile.converter != "glm52_vq_groups":
        raise ValueError("GLM52 full-bind preflight requires glm52_vq_groups profile")
    _expect_equal("requested model_id", model_id, profile.hf_model_id)
    _expect_equal("requested revision", revision, profile.revision)
    mismatches = validate_profile_against_hf_config_data(profile, config)
    if mismatches:
        raise ValueError(f"GLM52 profile/config mismatch: {mismatches}")

    config_sha256 = _sha256_metadata_file(config_input)
    source_index_sha256 = _sha256_metadata_file(source_index_input)
    schema = expected_glm52_non_vq_source_schema(config)
    runtime_schema = expected_glm52_non_vq_runtime_schema(config)
    expected_group_keys = expected_glm52_routed_group_keys(profile, config)
    if any(key.startswith(f"{profile.num_layers}:") for key in expected_group_keys):
        raise ValueError("GLM52 routed profile plan includes forbidden MTP layer")

    non_vq_tensor_count, non_vq_header_file_count = _validate_non_vq_package(
        non_vq_root,
        schema=schema,
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
        source_weight_map=source_weight_map,
    )
    present_group_keys, routed_header_file_count = _validate_routed_package(
        routed_root,
        expected_group_keys=expected_group_keys,
        profile=profile,
        model_id=model_id,
        revision=revision,
        config_sha256=config_sha256,
        source_index_sha256=source_index_sha256,
    )
    present_set = set(present_group_keys)
    missing_group_keys = tuple(
        key for key in expected_group_keys if key not in present_set
    )
    blockers = (
        (f"missing_routed_groups={len(missing_group_keys)}",)
        if missing_group_keys
        else ()
    )
    preflight_pass = not blockers
    kv_b_source_count = sum(
        name.endswith(".self_attn.kv_b_proj.weight") for name in schema
    )
    if kv_b_source_count != profile.num_layers:
        raise ValueError(
            "GLM52 non-VQ schema must contain one kv_b_proj source per layer"
        )
    return GLM52FullBindPreflightReport(
        record_type=GLM52_FULL_BIND_PREFLIGHT_RECORD_TYPE,
        schema_version=1,
        preflight_status=(
            GLM52_FULL_BIND_PREFLIGHT_READY
            if preflight_pass
            else GLM52_FULL_BIND_PREFLIGHT_BLOCKED
        ),
        preflight_pass=preflight_pass,
        blockers=blockers,
        profile=profile.name,
        profile_path=None if profile_input is None else str(profile_input),
        profile_sha256=(
            None
            if profile_input is None
            else _sha256_metadata_file(profile_input)
        ),
        profile_contract_sha256=_canonical_sha256(asdict(profile)),
        model_id=model_id,
        source_revision=revision,
        config_path=str(config_input),
        config_sha256=config_sha256,
        source_index_path=str(source_index_input),
        source_index_sha256=source_index_sha256,
        non_vq_artifact_dir=str(non_vq_root),
        routed_artifact_dir=str(routed_root),
        non_vq_tensor_count=non_vq_tensor_count,
        non_vq_runtime_target_count=len(runtime_schema),
        non_vq_header_file_count=non_vq_header_file_count,
        kv_b_source_tensor_count=kv_b_source_count,
        kv_b_runtime_target_count=kv_b_source_count * 2,
        expected_routed_group_keys=expected_group_keys,
        present_routed_group_keys=present_group_keys,
        missing_routed_group_keys=missing_group_keys,
        routed_header_file_count=routed_header_file_count,
        header_only=True,
        tensor_payloads_read=False,
        payload_hashes_verified=False,
        full_model_constructed=False,
        dense_routed_experts=False,
        production_binding_proven=False,
        production_generation_proven=False,
    )
