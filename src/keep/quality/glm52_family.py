"""Immutable source-relative release contract for GLM-5.2 REAP KEEP."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from keep.quality.rc_gates import (
    BALANCED_HARD_TARGETS,
    COMMUNITY_WOW_TARGETS,
)


PINNED_GLM52_MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
PINNED_GLM52_REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
GLM52_FAMILY_POLICY_RECORD_TYPE = "glm52_family_gate_policy"
GLM52_FAMILY_POLICY_STATUS = "glm52_family_policy_frozen"
GLM52_REQUIRED_EVAL_SPLITS = ("report", "selection", "holdout")
GLM52_REQUIRED_EVAL_DOMAINS = ("route", "math", "instruction")
GLM52_TOKENIZER_BASE_VOCAB_SIZE = 154_820
GLM52_TOKENIZER_LENGTH = 154_856
GLM52_MODEL_VOCAB_SIZE = 154_880
GLM52_EOS_TOKEN_IDS = (154_820, 154_827, 154_829)
GLM52_EVAL_PROMPT_TEXT_SHA256 = (
    "3c39f6bb133907cf2fd93260a3cc548c0b186a264230568678cd076f1638eb6a"
)
GLM52_PROMPT_CONTENT_CONTRACT_SHA256 = (
    "cc1951aeeacda74c1315485ca5b477de344ffcd69d3d8f3ff3d44d61f84d1265"
)
GLM52_FAMILY_GATE_V1_SCHEMA_VERSION = 1
GLM52_FAMILY_GATE_V2_SCHEMA_VERSION = 2
GLM52_FAMILY_GATE_V3_SCHEMA_VERSION = 3
GLM52_FAMILY_GATE_V4_SCHEMA_VERSION = 4
GLM52_FAMILY_GATE_V5_SCHEMA_VERSION = 5
GLM52_FAMILY_GATE_V6_SCHEMA_VERSION = 6
GLM52_FAMILY_GATE_SCHEMA_VERSION = GLM52_FAMILY_GATE_V6_SCHEMA_VERSION
GLM52_FAMILY_GATE_V1_CHECK_NAMES = (
    "policy_frozen",
    "prompt_pack_frozen",
    "teacher_source_metadata_ready",
    "teacher_cache_full_vocabulary_ready",
    "indexshare_runtime_contract_ready",
    "synthetic_generation_contract_ready",
    "full_routed_coverage_ready",
    "full_225_group_artifact_ready",
    "accepted_artifact_bytes_and_bpw_ready",
    "production_binding_and_generation_ready",
    "full_vocabulary_source_relative_family_eval_ready",
    "route_math_diagnostics_ready",
    "same_machine_pinned_fp4_benchmark_ready",
    "no_dense_routed_experts",
)
GLM52_FAMILY_GATE_V1_MISSING_REQUIREMENTS = (
    "dequantized_source_teacher_cache_payload",
    "full_225_group_artifact",
    "production_model_bind_and_generation",
    "full_vocabulary_source_relative_family_eval",
    "route_math_diagnostics",
    "same_machine_pinned_fp4_benchmark",
)
GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS = (
    "dequantized_source_teacher_cache_payload",
    "full_vocabulary_source_relative_family_eval",
    "route_math_diagnostics",
    "same_machine_pinned_fp4_benchmark",
)
GLM52_FAMILY_GATE_V3_MISSING_REQUIREMENTS = (
    "full_vocabulary_source_relative_family_eval",
    "route_math_diagnostics",
    "same_machine_pinned_fp4_benchmark",
)
GLM52_FAMILY_GATE_V4_MISSING_REQUIREMENTS = (
    "route_math_diagnostics",
    "same_machine_pinned_fp4_benchmark",
)
GLM52_FAMILY_GATE_V5_MISSING_REQUIREMENTS = (
    "same_machine_pinned_fp4_benchmark",
)
GLM52_FAMILY_GATE_V6_MISSING_REQUIREMENTS: tuple[str, ...] = ()
GLM52_TEACHER_CACHE_SOURCE_EVIDENCE = {
    "source_audit_path": "artifacts/quality/glm52-reap-source-audit-20260709.json",
    "source_audit_file_sha256": (
        "c24a6a3cc7f87b829b07875342e76997e0e955aae5967c5f811562801bf5b448"
    ),
    "source_audit_record_type": "glm52_modelopt_nvfp4_source_audit",
    "source_payload_audit_path": (
        "artifacts/quality/glm52-reap-source-payload-audit-20260709.json"
    ),
    "source_payload_audit_file_sha256": (
        "f7a78c061139c08cfe51347f6dc457f8e7ac5745c4e64a2c0e7814196e41e46d"
    ),
    "source_payload_audit_record_type": (
        "glm52_modelopt_nvfp4_source_payload_audit"
    ),
    "routed_group_count": 225,
    "routed_projection_bundle_count": 37_800,
    "required_payload_shard_count": 61,
}
GLM52_REAP_PROFILE = "glm52-reap-504b-v2"
GLM52_REAP_CONFIG_SHA256 = (
    "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
)
GLM52_REAP_INDEX_SHA256 = (
    "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
)
GLM52_REAP_PROFILE_SHA256 = (
    "ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d"
)
GLM52_REAP_PROFILE_CONTRACT_SHA256 = (
    "28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8"
)
GLM52_REAP_EXPECTED_GROUP_KEYS = tuple(
    f"{layer}:{projection}"
    for layer in range(3, 78)
    for projection in ("gate_proj", "up_proj", "down_proj")
)
GLM52_REAP_EXPECTED_LAYER_IDS = tuple(range(3, 78))
GLM52_NON_VQ_TENSOR_COUNT = 1_194
GLM52_NON_VQ_SHARD_COUNT = 9
GLM52_NON_VQ_RUNTIME_TARGET_COUNT = 1_272
GLM52_NON_VQ_PARAMETER_COUNT = 18_560_731_704
GLM52_NON_VQ_TENSOR_PAYLOAD_BYTES = 37_121_488_608
GLM52_ROUTED_PARAMETER_COUNT = 475_634_073_600
GLM52_ROUTED_TENSOR_COUNT = 151_200
GLM52_ROUTED_CODES_BYTES = 59_454_259_200
GLM52_ROUTED_SCALES_BYTES = 1_857_945_600
GLM52_ROUTED_CODEBOOK_BYTES = 230_400
# Audit `actual_routed_payload_bytes` is codes + scales only; the accepted
# whole-model tensor payload accounts for the codebook separately.
GLM52_ROUTED_CODES_SCALES_BYTES = 61_312_204_800
GLM52_WHOLE_MODEL_PARAMETER_COUNT = 494_194_805_304
GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES = 98_433_923_808
GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW = (
    GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
    * 8
    / GLM52_WHOLE_MODEL_PARAMETER_COUNT
)
GLM52_ACCEPTED_POLICY_BPW = 1.5934433
GLM52_ACCEPTED_PRODUCTION_COMPOSITE_IDENTITY_SHA256 = (
    "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
)
GLM52_PINNED_TOKENIZER_FILES: dict[str, dict[str, int | str]] = {
    "config.json": {
        "size_bytes": 10_854,
        "sha256": "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b",
    },
    "tokenizer.json": {
        "size_bytes": 20_217_442,
        "sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    },
    "tokenizer_config.json": {
        "size_bytes": 761,
        "sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    },
    "chat_template.jinja": {
        "size_bytes": 5_076,
        "sha256": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
    },
    "generation_config.json": {
        "size_bytes": 194,
        "sha256": "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55",
    },
}
GLM52_FORBIDDEN_TOKENIZER_INPUTS = (
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
)


def find_unpinned_glm52_tokenizer_inputs(
    tokenizer_dir: str | Path,
) -> tuple[str, ...]:
    """Return local files that Transformers can consume outside the pinned set."""

    root = Path(tokenizer_dir).expanduser().resolve()
    unexpected = [
        filename
        for filename in GLM52_FORBIDDEN_TOKENIZER_INPUTS
        if (root / filename).exists() or (root / filename).is_symlink()
    ]
    chat_templates = root / "chat_templates"
    if chat_templates.is_dir():
        templates = sorted(chat_templates.glob("*.jinja"))
        unexpected.extend(
            str(path.relative_to(root)) for path in templates
        )
        if not templates:
            unexpected.append("chat_templates/")
    elif chat_templates.exists() or chat_templates.is_symlink():
        unexpected.append("chat_templates")
    return tuple(unexpected)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _glm52_benchmark_api() -> Any:
    """Load the MLX-free benchmark contract without the quality package initializer."""

    name = "glm52_same_machine_fp4_contract"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("glm52_benchmark.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM52 benchmark contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_glm52_same_machine_benchmark_evidence(
    path: str | Path,
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the release verdict from sealed embedded raw measurements."""

    benchmark = _glm52_benchmark_api()
    return benchmark.load_glm52_same_machine_benchmark_evidence(path, policy=policy)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_exact_field(
    payload: Mapping[str, Any],
    field: str,
    expected: object,
    *,
    label: str,
) -> None:
    actual = payload.get(field)
    if not _exact_values_equal(actual, expected):
        raise ValueError(
            f"{label} {field} must be {expected!r}, found {actual!r}"
        )


def _exact_values_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        return set(actual) == set(expected) and all(
            _exact_values_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        return len(actual) == len(expected) and all(
            _exact_values_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return actual == expected


def _expect_exact_fields(
    payload: Mapping[str, Any],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for field, value in expected.items():
        _expect_exact_field(payload, field, value, label=label)


def _require_mapping_field(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} {field} must be an object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256_field(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> str:
    value = payload.get(field)
    if not _is_sha256(value):
        raise ValueError(f"{label} {field} must be a lowercase SHA-256")
    return str(value)


_NON_VQ_AUDIT_CHECKS = {
    "accounting": True,
    "index": True,
    "lineage": True,
    "payload_hashes": True,
    "physical_extents": True,
    "shard_hashes": True,
    "strict_tree": True,
    "tensor_inventory": True,
}

_NON_VQ_MAX_SHARD_PAYLOAD_BYTES = 4_294_967_296
_NON_VQ_TENSOR_PLAN_FIELDS = (
    "name",
    "dtype",
    "shape",
    "parameter_count",
    "payload_bytes",
    "source_shard",
    "source_offsets",
    "output_shard",
    "output_offsets",
)
_NON_VQ_TENSOR_FIELDS = frozenset(
    (*_NON_VQ_TENSOR_PLAN_FIELDS, "payload_sha256")
)
_NON_VQ_SHARD_FIELDS = frozenset(
    {
        "filename",
        "tensor_count",
        "tensor_inventory_sha256",
        "tensor_payload_bytes",
        "file_bytes",
        "file_sha256",
    }
)
_NON_VQ_SHARD_NAMES = tuple(
    f"model-{index:05d}-of-{GLM52_NON_VQ_SHARD_COUNT:05d}.safetensors"
    for index in range(1, GLM52_NON_VQ_SHARD_COUNT + 1)
)
def _require_exact_record_fields(
    value: object,
    expected_fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{label} field inventory is not exact")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_tensor_offsets(
    value: object,
    *,
    payload_bytes: int,
    label: str,
) -> list[int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(offset) is not int for offset in value)
    ):
        raise ValueError(f"{label} must contain exactly two integers")
    start, end = value
    if start < 0 or end - start != payload_bytes:
        raise ValueError(f"{label} extent does not match tensor payload bytes")
    return value


def _validate_glm52_non_vq_inventories(payload: Mapping[str, Any]) -> None:
    raw_tensors = payload.get("tensors")
    raw_shards = payload.get("shards")
    if type(raw_tensors) is not list or len(raw_tensors) != GLM52_NON_VQ_TENSOR_COUNT:
        raise ValueError(
            "non-VQ tensor inventory must contain exactly "
            f"{GLM52_NON_VQ_TENSOR_COUNT} records"
        )
    if type(raw_shards) is not list or len(raw_shards) != GLM52_NON_VQ_SHARD_COUNT:
        raise ValueError(
            "non-VQ shard inventory must contain exactly "
            f"{GLM52_NON_VQ_SHARD_COUNT} records"
        )

    tensor_names: list[str] = []
    tensor_plans: list[dict[str, object]] = []
    shard_tensors: dict[str, list[dict[str, object]]] = {
        name: [] for name in _NON_VQ_SHARD_NAMES
    }
    shard_offsets = {name: 0 for name in _NON_VQ_SHARD_NAMES}
    dtype_counts: Counter[str] = Counter()
    parameter_count = 0
    tensor_payload_bytes = 0
    previous_shard_index = -1

    for index, raw_tensor in enumerate(raw_tensors):
        label = f"non-VQ tensor record {index}"
        tensor = _require_exact_record_fields(
            raw_tensor,
            _NON_VQ_TENSOR_FIELDS,
            label=label,
        )
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} name must be a non-empty string")
        tensor_names.append(name)

        dtype = tensor.get("dtype")
        if dtype not in {"BF16", "F32"} or not isinstance(dtype, str):
            raise ValueError(f"{label} dtype must be BF16 or F32")
        shape = tensor.get("shape")
        if (
            type(shape) is not list
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError(f"{label} shape must contain positive integers")
        derived_parameters = 1
        for dimension in shape:
            derived_parameters *= dimension
        submitted_parameters = _require_exact_int(
            tensor.get("parameter_count"),
            label=f"{label} parameter_count",
            minimum=1,
        )
        if submitted_parameters != derived_parameters:
            raise ValueError(f"{label} parameter count does not match its shape")
        submitted_payload_bytes = _require_exact_int(
            tensor.get("payload_bytes"),
            label=f"{label} payload_bytes",
            minimum=1,
        )
        expected_payload_bytes = submitted_parameters * (2 if dtype == "BF16" else 4)
        if submitted_payload_bytes != expected_payload_bytes:
            raise ValueError(f"{label} payload bytes do not match dtype and shape")

        source_shard = tensor.get("source_shard")
        if (
            not isinstance(source_shard, str)
            or not source_shard.endswith(".safetensors")
            or Path(source_shard).name != source_shard
        ):
            raise ValueError(f"{label} source_shard is not a safe shard filename")
        _require_tensor_offsets(
            tensor.get("source_offsets"),
            payload_bytes=submitted_payload_bytes,
            label=f"{label} source_offsets",
        )

        output_shard = tensor.get("output_shard")
        if output_shard not in _NON_VQ_SHARD_NAMES or not isinstance(
            output_shard, str
        ):
            raise ValueError(f"{label} output_shard is not canonical")
        shard_index = _NON_VQ_SHARD_NAMES.index(output_shard)
        if shard_index < previous_shard_index or shard_index > previous_shard_index + 1:
            raise ValueError("non-VQ tensor shard membership order is not canonical")
        previous_shard_index = shard_index
        output_offset = shard_offsets[output_shard]
        expected_output_offsets = [
            output_offset,
            output_offset + submitted_payload_bytes,
        ]
        _expect_exact_field(
            tensor,
            "output_offsets",
            expected_output_offsets,
            label=label,
        )
        shard_offsets[output_shard] = output_offset + submitted_payload_bytes

        _require_sha256_field(tensor, "payload_sha256", label=label)
        plan = {field: tensor[field] for field in _NON_VQ_TENSOR_PLAN_FIELDS}
        tensor_plans.append(plan)
        shard_tensors[output_shard].append(plan)
        dtype_counts[dtype] += 1
        parameter_count += submitted_parameters
        tensor_payload_bytes += submitted_payload_bytes

    if tensor_names != sorted(tensor_names) or len(set(tensor_names)) != len(tensor_names):
        raise ValueError("non-VQ tensor inventory must be uniquely ordered by name")
    if previous_shard_index + 1 != GLM52_NON_VQ_SHARD_COUNT:
        raise ValueError("non-VQ tensor inventory does not produce exactly nine shards")
    _expect_exact_field(
        payload,
        "dtype_tensor_counts",
        dict(sorted(dtype_counts.items())),
        label="non-VQ evidence",
    )
    _expect_exact_field(
        payload,
        "parameter_count",
        parameter_count,
        label="non-VQ evidence",
    )
    _expect_exact_field(
        payload,
        "tensor_payload_bytes",
        tensor_payload_bytes,
        label="non-VQ evidence",
    )
    plan_basis = {
        "schema_version": 1,
        "copy_mode": "raw_safetensors_byte_ranges_v1",
        "selection_policy": "main_model_non_routed_bf16_f32_v1",
        "profile": GLM52_REAP_PROFILE,
        "model_id": PINNED_GLM52_MODEL_ID,
        "source_revision": PINNED_GLM52_REVISION,
        "config_sha256": GLM52_REAP_CONFIG_SHA256,
        "index_sha256": GLM52_REAP_INDEX_SHA256,
        "max_shard_payload_bytes": _NON_VQ_MAX_SHARD_PAYLOAD_BYTES,
        "tensors": tensor_plans,
    }
    _expect_exact_field(
        payload,
        "plan_sha256",
        canonical_sha256(plan_basis),
        label="non-VQ evidence",
    )
    expected_shard_records: list[dict[str, object]] = []
    for index, raw_shard in enumerate(raw_shards):
        label = f"non-VQ shard record {index}"
        shard = _require_exact_record_fields(
            raw_shard,
            _NON_VQ_SHARD_FIELDS,
            label=label,
        )
        expected_filename = _NON_VQ_SHARD_NAMES[index]
        _expect_exact_field(
            shard,
            "filename",
            expected_filename,
            label=label,
        )
        members = shard_tensors[expected_filename]
        expected_payload_bytes = sum(int(member["payload_bytes"]) for member in members)
        _expect_exact_field(
            shard,
            "tensor_count",
            len(members),
            label=label,
        )
        _expect_exact_field(
            shard,
            "tensor_payload_bytes",
            expected_payload_bytes,
            label=label,
        )
        _expect_exact_field(
            shard,
            "tensor_inventory_sha256",
            canonical_sha256(members),
            label=label,
        )
        _require_exact_int(
            shard.get("file_bytes"),
            label=f"{label} file_bytes",
            minimum=expected_payload_bytes,
        )
        _require_sha256_field(shard, "file_sha256", label=label)
        expected_shard_records.append(dict(shard))

    expected_package_set_sha256 = canonical_sha256(
        {
            "package_index_sha256": payload["package_index_sha256"],
            "shards": expected_shard_records,
        }
    )
    _expect_exact_field(
        payload,
        "package_set_sha256",
        expected_package_set_sha256,
        label="non-VQ evidence",
    )


def validate_glm52_non_vq_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the pinned production non-VQ package summary fail-closed."""

    expected = {
        "schema_version": 1,
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "package_audit_pass": True,
        "resume_verified": True,
        "profile": GLM52_REAP_PROFILE,
        "model_id": PINNED_GLM52_MODEL_ID,
        "source_revision": PINNED_GLM52_REVISION,
        "config_sha256": GLM52_REAP_CONFIG_SHA256,
        "index_sha256": GLM52_REAP_INDEX_SHA256,
        "source_authority": "pinned_huggingface_lfs_v1",
        "selection_policy": "main_model_non_routed_bf16_f32_v1",
        "copy_mode": "raw_safetensors_byte_ranges_v1",
        "retained_tensor_count": GLM52_NON_VQ_TENSOR_COUNT,
        "shard_count": GLM52_NON_VQ_SHARD_COUNT,
        "reused_shards": GLM52_NON_VQ_SHARD_COUNT,
        "written_shards": 0,
        "parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "tensor_payload_bytes": GLM52_NON_VQ_TENSOR_PAYLOAD_BYTES,
        "excluded_mtp_tensor_count": 2_039,
        "excluded_routed_tensor_count": GLM52_ROUTED_TENSOR_COUNT,
        "excluded_runtime_vq_tensor_count": 0,
        "dtype_tensor_counts": {"BF16": 1_119, "F32": 75},
        "max_shard_payload_bytes": _NON_VQ_MAX_SHARD_PAYLOAD_BYTES,
        "index_path": "model.safetensors.index.json",
    }
    _expect_exact_fields(payload, expected, label="non-VQ evidence")
    for field in (
        "manifest_sha256",
        "package_index_sha256",
        "package_set_sha256",
        "source_blob_inventory_sha256",
        "source_inventory_sha256",
        "plan_sha256",
    ):
        _require_sha256_field(payload, field, label="non-VQ evidence")
    _validate_glm52_non_vq_inventories(payload)

    package_audit = _require_mapping_field(
        payload,
        "package_audit",
        label="non-VQ evidence",
    )
    audit_expected = {
        "audit_pass": True,
        "checks": _NON_VQ_AUDIT_CHECKS,
        "manifest_sha256": payload["manifest_sha256"],
        "package_set_sha256": payload["package_set_sha256"],
        "parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "retained_tensor_count": GLM52_NON_VQ_TENSOR_COUNT,
        "shard_count": GLM52_NON_VQ_SHARD_COUNT,
        "tensor_payload_bytes": GLM52_NON_VQ_TENSOR_PAYLOAD_BYTES,
    }
    _expect_exact_fields(
        package_audit,
        audit_expected,
        label="non-VQ package audit",
    )
    return dict(payload)


_FULL_ARTIFACT_AUDIT_CHECKS = {
    "artifact_hashes_and_sizes": True,
    "bounded_whole_model_claims_suppressed": True,
    "exact_artifact_files": True,
    "exact_group_selection": True,
    "exact_manifest_contract": True,
    "exact_tensor_contract": True,
    "no_dense_routed_experts": True,
    "no_layer_78": True,
    "pinned_lineage": True,
}


def validate_glm52_full_artifact_audit(
    payload: Mapping[str, Any],
    *,
    non_vq_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete routed + non-VQ composite audit."""

    expected = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_artifact_audit",
        "audit_status": "glm52_modelopt_nvfp4_artifact_audit_ready",
        "audit_pass": True,
        "artifact_integrity_pass": True,
        "audit_blockers": [],
        "materialization_scope": "full",
        "accounting_scope": "full_composite_actual",
        "model_id": PINNED_GLM52_MODEL_ID,
        "source_revision": PINNED_GLM52_REVISION,
        "config_sha256": GLM52_REAP_CONFIG_SHA256,
        "index_sha256": GLM52_REAP_INDEX_SHA256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "expected_group_keys": list(GLM52_REAP_EXPECTED_GROUP_KEYS),
        "ready_group_keys": list(GLM52_REAP_EXPECTED_GROUP_KEYS),
        "complete_layer_ids": list(GLM52_REAP_EXPECTED_LAYER_IDS),
        "partial_layer_ids": [],
        "full_routed_artifact_ready": True,
        "dense_routed_experts": False,
        "resume_verified": True,
        "byte_identity_verified": True,
        "requested_code_bits": 8,
        "requested_group_size": 512,
        "requested_scale_estimator": "max_abs",
        "checks": _FULL_ARTIFACT_AUDIT_CHECKS,
        "actual_routed_codes_bytes": GLM52_ROUTED_CODES_BYTES,
        "actual_routed_scales_bytes": GLM52_ROUTED_SCALES_BYTES,
        "actual_routed_codebook_bytes": GLM52_ROUTED_CODEBOOK_BYTES,
        "actual_routed_payload_bytes": GLM52_ROUTED_CODES_SCALES_BYTES,
        "actual_routed_weight_count": GLM52_ROUTED_PARAMETER_COUNT,
        "actual_routed_bpw": 1.03125,
        "main_routed_parameter_count": GLM52_ROUTED_PARAMETER_COUNT,
        "main_routed_tensor_count": GLM52_ROUTED_TENSOR_COUNT,
        "main_non_routed_parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "main_non_routed_tensor_count": GLM52_NON_VQ_TENSOR_COUNT,
        "main_non_routed_tensor_payload_bytes": (
            GLM52_NON_VQ_TENSOR_PAYLOAD_BYTES
        ),
        "main_model_parameter_count_excluding_mtp": (
            GLM52_WHOLE_MODEL_PARAMETER_COUNT
        ),
        "actual_whole_model_tensor_payload_bytes": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "actual_whole_model_tensor_payload_bpw": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW
        ),
        "whole_model_tensor_payload_values_actual": True,
        "whole_model_physical_values_actual": True,
        "whole_model_values_actual": True,
        "whole_model_artifact_blocker": None,
        "non_vq_package_evidence_authenticated": True,
    }
    _expect_exact_fields(payload, expected, label="composite audit")
    for field in ("manifest_sha256", "group_set_sha256", "non_vq_evidence_sha256"):
        _require_sha256_field(payload, field, label="composite audit")

    non_vq_audit = _require_mapping_field(
        payload,
        "non_vq_package_audit",
        label="composite audit",
    )
    nested_expected = {
        "audit_pass": True,
        "checks": _NON_VQ_AUDIT_CHECKS,
        "manifest_sha256": non_vq_evidence["manifest_sha256"],
        "package_set_sha256": non_vq_evidence["package_set_sha256"],
        "parameter_count": GLM52_NON_VQ_PARAMETER_COUNT,
        "retained_tensor_count": GLM52_NON_VQ_TENSOR_COUNT,
        "shard_count": GLM52_NON_VQ_SHARD_COUNT,
        "tensor_payload_bytes": GLM52_NON_VQ_TENSOR_PAYLOAD_BYTES,
    }
    _expect_exact_fields(
        non_vq_audit,
        nested_expected,
        label="composite non-VQ audit",
    )
    return dict(payload)


def build_glm52_production_composite_identity(
    *,
    non_vq_evidence: Mapping[str, Any],
    composite_artifact_audit: Mapping[str, Any],
) -> dict[str, object]:
    """Derive the path-free identity shared by every production consumer."""

    expected_groups = composite_artifact_audit["expected_group_keys"]
    if not isinstance(expected_groups, list):
        raise ValueError("composite audit expected group keys must be a list")
    return {
        "schema_version": 1,
        "identity_kind": "glm52_production_composite_v1",
        "model_id": PINNED_GLM52_MODEL_ID,
        "source_revision": PINNED_GLM52_REVISION,
        "profile": GLM52_REAP_PROFILE,
        "profile_sha256": GLM52_REAP_PROFILE_SHA256,
        "profile_contract_sha256": GLM52_REAP_PROFILE_CONTRACT_SHA256,
        "config_sha256": GLM52_REAP_CONFIG_SHA256,
        "source_index_sha256": GLM52_REAP_INDEX_SHA256,
        "source_blob_inventory_sha256": non_vq_evidence[
            "source_blob_inventory_sha256"
        ],
        "source_inventory_sha256": non_vq_evidence[
            "source_inventory_sha256"
        ],
        "non_vq_manifest_sha256": non_vq_evidence["manifest_sha256"],
        "non_vq_package_index_sha256": non_vq_evidence[
            "package_index_sha256"
        ],
        "non_vq_package_set_sha256": non_vq_evidence[
            "package_set_sha256"
        ],
        "routed_manifest_sha256": composite_artifact_audit[
            "manifest_sha256"
        ],
        "routed_group_set_sha256": composite_artifact_audit[
            "group_set_sha256"
        ],
        "routed_group_inventory_sha256": canonical_sha256(expected_groups),
        "routed_group_count": 225,
        "code_bits": 8,
        "group_size": 512,
        "scale_estimator": "max_abs",
        "whole_model_parameter_count": GLM52_WHOLE_MODEL_PARAMETER_COUNT,
        "whole_model_tensor_payload_bytes": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "whole_model_tensor_payload_bpw": GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW,
    }


_PRODUCTION_FINGERPRINT_FLAGS = (
    "input_fingerprint_reverified_before_bind",
    "input_fingerprint_reverified_after_residency",
    "input_fingerprint_reverified_after_warmup",
    "input_fingerprint_reverified_after_warmup_preparation",
    "input_fingerprint_reverified_before_generation",
    "input_fingerprint_reverified_after_generation",
)

_PRODUCTION_PHASE_LABELS = (
    "before_memory_policy",
    "before_input_validation",
    "after_fresh_payload_audits",
    "after_tokenizer_and_prompt",
    "before_model_construction",
    "after_model_construction",
    "after_non_vq_bind",
    "after_routed_bind",
    "before_residency_barrier",
    "after_residency_quiet",
    "before_generation_warmup",
    "after_generation_warmup",
    "after_generation_warmup_cache_clear",
    "after_generation_warmup_quiet",
    "before_generation",
    "after_generation",
)
_PRODUCTION_PHASE_MEMORY_FIELDS = frozenset(
    {
        "available",
        "label",
        "mlx_active_bytes",
        "mlx_cache_bytes",
        "mlx_peak_bytes",
        "pageouts_delta",
        "pageouts_total",
        "pages_free",
        "rss_bytes",
        "swapouts_delta",
        "swapouts_total",
    }
)
_PRODUCTION_PHASE_MEMORY_ALIASES = {
    "cold_residency_memory": "after_residency_quiet",
    "generation_warmup_phase_memory": "after_generation_warmup",
    "post_warmup_cache_clear_phase_memory": (
        "after_generation_warmup_cache_clear"
    ),
    "post_warmup_quiet_phase_memory": "after_generation_warmup_quiet",
    "pre_generation_memory": "before_generation",
    "steady_state_generation_memory": "after_generation",
}


def _production_memory_phase_is_clean(
    row: Mapping[str, Any],
    *,
    label: str,
) -> bool:
    return all(
        (
            row.get("label") == label,
            row.get("available") is True,
            type(row.get("pageouts_delta")) is int,
            row.get("pageouts_delta") == 0,
            type(row.get("swapouts_delta")) is int,
            row.get("swapouts_delta") == 0,
        )
    )


def _validate_production_phase_memory(payload: Mapping[str, Any]) -> None:
    raw_phase_memory = payload.get("phase_memory")
    if (
        type(raw_phase_memory) is not list
        or len(raw_phase_memory) != len(_PRODUCTION_PHASE_LABELS)
    ):
        raise ValueError(
            "production phase_memory must contain the exact 16-phase inventory"
        )

    rows_by_label: dict[str, Mapping[str, Any]] = {}
    previous_pageouts_total: int | None = None
    previous_swapouts_total: int | None = None
    for index, (raw_row, expected_label) in enumerate(
        zip(raw_phase_memory, _PRODUCTION_PHASE_LABELS, strict=True)
    ):
        label = f"production phase_memory record {index}"
        row = _require_exact_record_fields(
            raw_row,
            _PRODUCTION_PHASE_MEMORY_FIELDS,
            label=label,
        )
        _expect_exact_field(row, "label", expected_label, label=label)
        _expect_exact_field(row, "available", True, label=label)
        for field in (
            "mlx_active_bytes",
            "mlx_cache_bytes",
            "mlx_peak_bytes",
            "pageouts_total",
            "pages_free",
            "rss_bytes",
            "swapouts_total",
        ):
            _require_exact_int(row.get(field), label=f"{label} {field}")

        pageouts_total = int(row["pageouts_total"])
        swapouts_total = int(row["swapouts_total"])
        if index == 0:
            _expect_exact_field(row, "pageouts_delta", None, label=label)
            _expect_exact_field(row, "swapouts_delta", None, label=label)
        else:
            pageouts_delta = _require_exact_int(
                row.get("pageouts_delta"),
                label=f"{label} pageouts_delta",
            )
            swapouts_delta = _require_exact_int(
                row.get("swapouts_delta"),
                label=f"{label} swapouts_delta",
            )
            assert previous_pageouts_total is not None
            assert previous_swapouts_total is not None
            if pageouts_total - previous_pageouts_total != pageouts_delta:
                raise ValueError(
                    f"{label} pageouts delta does not reconcile adjacent totals"
                )
            if swapouts_total - previous_swapouts_total != swapouts_delta:
                raise ValueError(
                    f"{label} swapouts delta does not reconcile adjacent totals"
                )
        previous_pageouts_total = pageouts_total
        previous_swapouts_total = swapouts_total
        rows_by_label[expected_label] = row

    for field, phase_label in _PRODUCTION_PHASE_MEMORY_ALIASES.items():
        _expect_exact_field(
            payload,
            field,
            dict(rows_by_label[phase_label]),
            label="production generation",
        )

    cold_clean = _production_memory_phase_is_clean(
        rows_by_label["after_residency_quiet"],
        label="after_residency_quiet",
    )
    warmup_clean = _production_memory_phase_is_clean(
        rows_by_label["after_generation_warmup"],
        label="after_generation_warmup",
    )
    post_warmup_clear_clean = _production_memory_phase_is_clean(
        rows_by_label["after_generation_warmup_cache_clear"],
        label="after_generation_warmup_cache_clear",
    )
    post_warmup_quiet_clean = _production_memory_phase_is_clean(
        rows_by_label["after_generation_warmup_quiet"],
        label="after_generation_warmup_quiet",
    )
    pre_generation_clean = _production_memory_phase_is_clean(
        rows_by_label["before_generation"],
        label="before_generation",
    )
    steady_state_clean = _production_memory_phase_is_clean(
        rows_by_label["after_generation"],
        label="after_generation",
    )
    if not post_warmup_clear_clean or not post_warmup_quiet_clean:
        raise ValueError("production post-warmup clear and quiet phases must be clean")
    warm_residency_proven = bool(
        payload.get("parameter_residency_established") is True
        and payload.get("system_wired_default") is True
        and pre_generation_clean
        and steady_state_clean
    )
    derived_claims = {
        "generation_warmup_memory_clean": warmup_clean,
        "cold_residency_memory_clean": cold_clean,
        "pre_generation_memory_clean": pre_generation_clean,
        "steady_state_generation_memory_clean": steady_state_clean,
        "warm_residency_proven": warm_residency_proven,
        "production_residency_proven": bool(
            cold_clean and warmup_clean and warm_residency_proven
        ),
    }
    pinned_claims = {
        "generation_warmup_memory_clean": False,
        "cold_residency_memory_clean": False,
        "pre_generation_memory_clean": True,
        "steady_state_generation_memory_clean": True,
        "warm_residency_proven": True,
        "production_residency_proven": False,
    }
    if derived_claims != pinned_claims:
        raise ValueError(
            "production memory rows do not prove the pinned warm-only residency state"
        )
    _expect_exact_fields(
        payload,
        derived_claims,
        label="production generation",
    )


def _validate_token_ids(
    value: object,
    *,
    expected_count: int,
    label: str,
) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} IDs")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= GLM52_MODEL_VOCAB_SIZE
        for token_id in value
    ):
        raise ValueError(f"{label} contains an ID outside the model vocabulary")
    return value


def validate_glm52_production_generation(
    payload: Mapping[str, Any],
    *,
    common_artifact_identity: Mapping[str, object],
    composite_artifact_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate detailed production bind and one-token generation evidence."""

    expected = {
        "schema_version": 1,
        "record_type": "glm52_production_generation_probe",
        "probe_status": "glm52_production_generation_probe_ready",
        "probe_pass": True,
        "full_model_scope": "production_composite",
        "model_id": PINNED_GLM52_MODEL_ID,
        "source_revision": PINNED_GLM52_REVISION,
        "production_artifact_used": True,
        "production_binding_proven": True,
        "production_generation_proven": True,
        "whole_model_runtime_proven": True,
        "accepted_whole_model_tensor_payload_bytes": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "accepted_whole_model_tensor_payload_bpw": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW
        ),
        "full_routed_group_count": 225,
        "bound_sparse_layer_ids": list(GLM52_REAP_EXPECTED_LAYER_IDS),
        "dense_routed_parameter_names": [],
        "dense_routed_experts": False,
        "unbound_vq_experts": False,
        "tokenizer_used": True,
        "chat_template_add_generation_prompt": True,
        "chat_template_enable_thinking": False,
        "explicit_prompt_cache": True,
        "prompt_token_count": 11,
        "prefill_token_count": 10,
        "decode_token_count": 1,
        "generated_token_count": 1,
        "generated_ids_within_vocab": True,
        "logits_shape": [GLM52_MODEL_VOCAB_SIZE],
        "logits_finite": True,
        "greedy_sampling": True,
        "generation_method": "direct_prefill_single_decode",
        "lookahead_forward_scheduled": False,
        "generation_warmup_proven": True,
        "measured_generation_after_warmup": True,
        "warmup_measured_token_match": True,
        "measured_generation_avoids_lookahead": True,
        "generation_measured_after_residency": True,
        "mlx_peak_reset_before_generation": True,
        "parameter_residency_established": True,
        "cold_residency_mlx_peak_scope": "since_probe_start",
        "steady_state_generation_mlx_peak_scope": (
            "since_pre_generation_reset"
        ),
        "system_wired_default": True,
        "custom_mlx_wired_limit_used": False,
        "quality_claim": False,
        "speed_claim": False,
        "same_machine_benchmark_proven": False,
        "full_vocabulary_eval_proven": False,
        "long_context_indexshare_proven": False,
    }
    _expect_exact_fields(payload, expected, label="production generation")
    _validate_production_phase_memory(payload)
    for field in _PRODUCTION_FINGERPRINT_FLAGS:
        _expect_exact_field(
            payload,
            field,
            True,
            label="production generation",
        )

    identity = dict(common_artifact_identity)
    identity_digest = canonical_sha256(identity)
    for alias in ("artifact_identity", "common_artifact_identity"):
        submitted = _require_mapping_field(
            payload,
            alias,
            label="production generation",
        )
        if not _exact_values_equal(dict(submitted), identity):
            raise ValueError(
                f"production generation {alias} does not match the derived identity"
            )
    for field in (
        "artifact_identity_sha256",
        "common_artifact_identity_sha256",
    ):
        _expect_exact_field(
            payload,
            field,
            identity_digest,
            label="production generation",
        )

    input_hashes = _require_mapping_field(
        payload,
        "input_evidence_file_sha256",
        label="production generation",
    )
    expected_input_names = {
        "composite_audit_json",
        "family_policy_json",
        "full_bind_preflight_json",
        "materialization_runs_jsonl",
        "non_vq_evidence_json",
        "tokenizer_readiness_json",
    }
    if set(input_hashes) != expected_input_names:
        raise ValueError("production input evidence hash inventory is not exact")
    for field in expected_input_names:
        _require_sha256_field(
            input_hashes,
            field,
            label="production input evidence",
        )
    _expect_exact_field(
        input_hashes,
        "non_vq_evidence_json",
        composite_artifact_audit["non_vq_evidence_sha256"],
        label="production input evidence",
    )

    bind_report = _require_mapping_field(
        payload,
        "non_vq_bind_report",
        label="production generation",
    )
    _expect_exact_field(
        bind_report,
        "loaded_count",
        GLM52_NON_VQ_RUNTIME_TARGET_COUNT,
        label="production non-VQ bind report",
    )
    for field in (
        "missing_model_parameters",
        "skipped_mtp_tensors",
        "skipped_routed_expert_tensors",
        "skipped_unmatched_tensors",
    ):
        _expect_exact_field(
            bind_report,
            field,
            [],
            label="production non-VQ bind report",
        )
    for field, expected_count in (
        ("loaded_model_parameters", GLM52_NON_VQ_RUNTIME_TARGET_COUNT),
        ("transformed_kv_b_tensors", 78),
    ):
        names = bind_report.get(field)
        if (
            not isinstance(names, list)
            or len(names) != expected_count
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != expected_count
        ):
            raise ValueError(
                f"production non-VQ bind report {field} inventory is not exact"
            )

    _validate_token_ids(
        payload.get("prompt_token_ids"),
        expected_count=11,
        label="production prompt token IDs",
    )
    measured_token_ids = _validate_token_ids(
        payload.get("generated_token_ids"),
        expected_count=1,
        label="production generated token IDs",
    )
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("production prompt must be a non-empty string")
    generated_text = payload.get("generated_text")
    if not isinstance(generated_text, str):
        raise ValueError("production generated text must be a string")

    warmup = _require_mapping_field(
        payload,
        "generation_warmup",
        label="production generation",
    )
    warmup_expected = {
        "explicit_prompt_cache": True,
        "generated_ids_within_vocab": True,
        "generated_token_count": 1,
        "generation_method": "mlx_lm_generate_step",
        "greedy_sampling": True,
        "logits_finite": True,
        "logits_shape": [GLM52_MODEL_VOCAB_SIZE],
        "lookahead_forward_scheduled": True,
    }
    _expect_exact_fields(
        warmup,
        warmup_expected,
        label="production generation warmup",
    )
    warmup_token_ids = _validate_token_ids(
        warmup.get("generated_token_ids"),
        expected_count=1,
        label="production warmup generated token IDs",
    )
    if warmup_token_ids != measured_token_ids:
        raise ValueError(
            "production warmup token IDs do not match measured token IDs"
        )

    wired_policy = _require_mapping_field(
        payload,
        "wired_memory_policy",
        label="production generation",
    )
    wired_expected = {
        "environment_overrides": {},
        "iogpu_wired_limit_available": True,
        "iogpu_wired_limit_mb": 0,
        "platform": "Darwin",
        "system_wired_default": True,
    }
    _expect_exact_fields(
        wired_policy,
        wired_expected,
        label="production wired-memory policy",
    )
    return dict(payload)


def define_glm52_family_gate_policy(
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Return the immutable v1 GLM52 source-relative release policy."""

    if model_id != PINNED_GLM52_MODEL_ID:
        raise ValueError(
            "model_id must be the pinned GLM52 source "
            f"{PINNED_GLM52_MODEL_ID!r}"
        )
    if revision != PINNED_GLM52_REVISION:
        raise ValueError(
            "revision must be the pinned GLM52 source "
            f"{PINNED_GLM52_REVISION!r}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "record_type": GLM52_FAMILY_POLICY_RECORD_TYPE,
        "policy_status": GLM52_FAMILY_POLICY_STATUS,
        "model_id": model_id,
        "revision": revision,
        "profile": GLM52_REAP_PROFILE,
        "thresholds_frozen_before_candidate_metrics": True,
        "air_evidence_reused": False,
        # Frozen label: part of the hashed policy contract, so it keeps the
        # module path the policy was frozen under before the package rename.
        "quality_threshold_source": (
            "mlx_vq.quality.rc_gates.COMMUNITY_WOW_TARGETS"
        ),
        "tokenizer_identity": {
            "base_vocab_size": GLM52_TOKENIZER_BASE_VOCAB_SIZE,
            "tokenizer_length": GLM52_TOKENIZER_LENGTH,
            "model_vocab_size": GLM52_MODEL_VOCAB_SIZE,
            "eos_token_ids": list(GLM52_EOS_TOKEN_IDS),
            "files": {
                filename: dict(identity)
                for filename, identity in GLM52_PINNED_TOKENIZER_FILES.items()
            },
        },
        "teacher": {
            "source_kind": (
                "deterministically_dequantized_pinned_modelopt_nvfp4"
            ),
            "source_weight_encoding": "modelopt_nvfp4",
            "source_decoder": "modelopt_nvfp4_v1",
            "activation_quantization_emulated": False,
            "exact_w4a4_runtime_parity_claimed": False,
            "bf16_teacher_claimed": False,
        },
        "eval_gate": {
            "required_splits": list(GLM52_REQUIRED_EVAL_SPLITS),
            "required_domains": list(GLM52_REQUIRED_EVAL_DOMAINS),
            "prompt_text_sha256": GLM52_EVAL_PROMPT_TEXT_SHA256,
            "minimum_clean_rows_per_split": 22,
            "minimum_total_clean_rows": 66,
            "full_vocabulary_logits_required": True,
            "compact_topk_kld_accepted": False,
            "mean_kld_max": COMMUNITY_WOW_TARGETS["mean_kld_max"],
            "p999_kld_max": COMMUNITY_WOW_TARGETS["p999_kld_max"],
            "top1_min": COMMUNITY_WOW_TARGETS["top1_min"],
            "domain_top1_min": COMMUNITY_WOW_TARGETS[
                "domain_top1_min"
            ],
            "mean_ppl_ratio_max": BALANCED_HARD_TARGETS[
                "mean_ppl_ratio_max"
            ],
            "dirty_rows_accepted": False,
            "holdout_tuning_forbidden": True,
        },
        "benchmark_gate": {
            "required_scenarios": ["prefill_1k", "decode_128"],
            "minimum_repetitions_per_scenario": 3,
            "comparison_baseline": (
                "same_machine_pinned_fp4_source_streaming_control"
            ),
            "maximum_candidate_to_reference_ratio": (
                BALANCED_HARD_TARGETS["lane_s_ratio_max"]
            ),
            "candidate_and_control_same_machine": True,
            "pageouts_and_swapouts_must_be_zero": True,
        },
        "artifact_target": {
            "tensor_payload_bytes": 98_433_923_808,
            "whole_main_bpw": 1.5934433,
            "target_accepted_by_user": True,
            "actual_full_artifact_required": True,
        },
        "hard_requirements": [
            "pinned_source_and_decoder_lineage",
            "frozen_prompt_pack_before_candidate_metrics",
            "full_vocabulary_teacher_and_candidate_logits",
            "report_selection_holdout_rows_are_memory_clean",
            "holdout_is_never_used_for_fit_or_selection",
            "per_domain_metrics_recomputed_from_rows",
            "candidate_has_no_dense_routed_experts",
            "full_225_group_artifact_passes_strict_audit",
            "production_model_binds_and_generates",
            "benchmark_uses_frozen_same_machine_control",
        ],
    }
    payload["policy_contract_sha256"] = canonical_sha256(payload)
    return payload


def validate_glm52_family_gate_policy(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected = define_glm52_family_gate_policy(
        model_id=PINNED_GLM52_MODEL_ID,
        revision=PINNED_GLM52_REVISION,
    )
    submitted = dict(payload)
    embedded_digest = submitted.pop("policy_contract_sha256", None)
    if embedded_digest != canonical_sha256(submitted):
        raise ValueError(
            "family policy embedded digest does not authenticate its body"
        )
    if not _exact_values_equal(dict(payload), expected):
        raise ValueError("family policy body does not match the frozen contract")
    return expected


def validate_glm52_tokenizer_readiness(
    payload: Mapping[str, Any],
    *,
    tokenizer_dir: str | Path,
) -> dict[str, dict[str, int | str]]:
    root = Path(tokenizer_dir).expanduser().resolve()
    unexpected_inputs = find_unpinned_glm52_tokenizer_inputs(root)
    if unexpected_inputs:
        raise ValueError(
            "unexpected tokenizer input(s) outside the pinned identity set: "
            f"{list(unexpected_inputs)}"
        )
    expected = {
        "record_type": "glm52_tokenizer_readiness_probe",
        "probe_status": "glm52_tokenizer_readiness_probe_ready",
        "probe_pass": True,
        "tokenizer_payload_ready": True,
        "production_ready": True,
        "model_id": PINNED_GLM52_MODEL_ID,
        "source_revision": PINNED_GLM52_REVISION,
        "source_authority": "pinned_huggingface_snapshot",
        "production_identity_contract": True,
        "immutable_file_identity_pass": True,
        "config_lineage_pass": True,
        "tokenizer_load_pass": True,
        "mlx_lm_loader_pass": True,
        "mlx_lm_wrapper_eos_pass": True,
        "prompt_token_ids_within_model_vocab": True,
        "prompt_roundtrip_pass": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "remote_code_required": False,
        "tokenizer_base_vocab_size": GLM52_TOKENIZER_BASE_VOCAB_SIZE,
        "tokenizer_length": GLM52_TOKENIZER_LENGTH,
        "model_vocab_size": GLM52_MODEL_VOCAB_SIZE,
        "tokenizer_fits_model_vocab": True,
        "pad_token_id": GLM52_EOS_TOKEN_IDS[0],
        "tokenizer_eos_token_id": GLM52_EOS_TOKEN_IDS[0],
        "model_eos_token_ids": list(GLM52_EOS_TOKEN_IDS),
        "mlx_lm_wrapper_eos_token_ids": list(GLM52_EOS_TOKEN_IDS),
        "thinking_disabled_pass": True,
        "assistant_generation_marker_pass": True,
        "missing_requirements": [],
    }
    for name, value in expected.items():
        actual = payload.get(name)
        if not _exact_values_equal(actual, value):
            raise ValueError(
                f"tokenizer readiness {name} must be {value!r}, "
                f"found {actual!r}"
            )
    recorded_dir = payload.get("tokenizer_dir")
    if not isinstance(recorded_dir, str) or Path(recorded_dir).resolve() != root:
        raise ValueError("tokenizer readiness directory does not match tokenizer_dir")
    recorded_identities = payload.get("file_identities")
    if not isinstance(recorded_identities, Mapping):
        raise ValueError("tokenizer readiness file_identities must be a mapping")
    actual_identities: dict[str, dict[str, int | str]] = {}
    for filename, expected_identity in GLM52_PINNED_TOKENIZER_FILES.items():
        path = root / filename
        if not path.is_file():
            raise ValueError(f"pinned tokenizer file is missing: {filename}")
        actual = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if not _exact_values_equal(actual, expected_identity):
            raise ValueError(
                f"pinned tokenizer file identity mismatch for {filename}: "
                f"{actual}"
            )
        recorded = recorded_identities.get(filename)
        if not isinstance(recorded, Mapping):
            raise ValueError(
                f"tokenizer readiness identity is missing for {filename}"
            )
        recorded_subset = {
            "size_bytes": recorded.get("size_bytes"),
            "sha256": recorded.get("sha256"),
            "expected_size_bytes": recorded.get("expected_size_bytes"),
            "expected_sha256": recorded.get("expected_sha256"),
            "identity_pass": recorded.get("identity_pass"),
        }
        expected_recorded = {
            "size_bytes": expected_identity["size_bytes"],
            "sha256": expected_identity["sha256"],
            "expected_size_bytes": expected_identity["size_bytes"],
            "expected_sha256": expected_identity["sha256"],
            "identity_pass": True,
        }
        if not _exact_values_equal(recorded_subset, expected_recorded):
            raise ValueError(
                f"tokenizer readiness identity claim mismatch for {filename}"
            )
        recorded_path = recorded.get("path")
        if (
            not isinstance(recorded_path, str)
            or Path(recorded_path).resolve() != path.resolve()
        ):
            raise ValueError(
                f"tokenizer readiness path mismatch for {filename}"
            )
        actual_identities[filename] = actual
    if set(recorded_identities) != set(GLM52_PINNED_TOKENIZER_FILES):
        raise ValueError("tokenizer readiness file identity inventory is not exact")
    return actual_identities


_PROMPT_PROVENANCE_FIELDS = frozenset(
    {
        "tokenizer_dir",
        "tokenizer_readiness_json",
        "tokenizer_readiness_sha256",
        "family_policy_json",
        "family_policy_sha256",
        "family_policy_contract_sha256",
        "tokenizer_file_identities",
        "local_files_only",
        "trust_remote_code",
    }
)


def validate_glm52_family_prompt_pack(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    final_body = dict(payload)
    final_digest = final_body.pop("prompt_pack_contract_sha256", None)
    if final_digest != canonical_sha256(final_body):
        raise ValueError(
            "prompt pack embedded digest does not authenticate final provenance"
        )

    content_body = dict(final_body)
    for field in _PROMPT_PROVENANCE_FIELDS:
        if field not in content_body:
            raise ValueError(f"prompt pack is missing provenance field {field}")
        content_body.pop(field)
    content_digest = content_body.pop("prompt_content_contract_sha256", None)
    if content_digest != canonical_sha256(content_body):
        raise ValueError(
            "prompt content digest does not authenticate prompts and token IDs"
        )
    if content_digest != GLM52_PROMPT_CONTENT_CONTRACT_SHA256:
        raise ValueError("prompt content contract is not the pinned GLM52 pack")

    expected_scalars = {
        "schema_version": 1,
        "record_type": "glm52_family_eval_prompt_pack",
        "prompt_pack_status": "glm52_family_eval_prompt_pack_frozen",
        "prompt_pack_ready": True,
        "model_id": PINNED_GLM52_MODEL_ID,
        "revision": PINNED_GLM52_REVISION,
        "profile": GLM52_REAP_PROFILE,
        "prompt_pack_frozen_before_candidate_metrics": True,
        "holdout_tuning_forbidden": True,
        "selection_is_only_tuning_eligible_split": True,
        "required_splits": list(GLM52_REQUIRED_EVAL_SPLITS),
        "required_domains": list(GLM52_REQUIRED_EVAL_DOMAINS),
        "split_counts": {
            split: 22 for split in GLM52_REQUIRED_EVAL_SPLITS
        },
        "domain_counts_by_split": {
            split: {"route": 8, "math": 7, "instruction": 7}
            for split in GLM52_REQUIRED_EVAL_SPLITS
        },
        "prompt_row_count": 66,
        "prompt_text_sha256": GLM52_EVAL_PROMPT_TEXT_SHA256,
        "tokenizer_vocab_size": GLM52_TOKENIZER_BASE_VOCAB_SIZE,
        "tokenizer_length": GLM52_TOKENIZER_LENGTH,
        "expected_vocab_size": GLM52_MODEL_VOCAB_SIZE,
        "missing_requirements": [],
        "family_policy_contract_sha256": define_glm52_family_gate_policy(
            model_id=PINNED_GLM52_MODEL_ID,
            revision=PINNED_GLM52_REVISION,
        )["policy_contract_sha256"],
        "tokenizer_file_identities": GLM52_PINNED_TOKENIZER_FILES,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    for field, expected in expected_scalars.items():
        if not _exact_values_equal(payload.get(field), expected):
            raise ValueError(
                f"prompt pack {field} must be {expected!r}, "
                f"found {payload.get(field)!r}"
            )

    raw_rows = payload.get("prompt_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 66:
        raise ValueError("prompt pack must contain exactly 66 rows")
    rows: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    prompts: set[str] = set()
    split_counts: Counter[str] = Counter()
    domain_counts: Counter[tuple[str, str]] = Counter()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("prompt rows must be JSON objects")
        row = dict(raw_row)
        split = row.get("split")
        domain = row.get("domain")
        prompt_id = row.get("prompt_id")
        prompt = row.get("prompt")
        token_ids = row.get("encoded_token_ids")
        if split not in GLM52_REQUIRED_EVAL_SPLITS:
            raise ValueError("prompt row split is not allowed")
        if domain not in GLM52_REQUIRED_EVAL_DOMAINS:
            raise ValueError("prompt row domain is not allowed")
        if not isinstance(prompt_id, str) or prompt_id in prompt_ids:
            raise ValueError("prompt row IDs must be unique strings")
        if not isinstance(prompt, str) or not prompt or prompt in prompts:
            raise ValueError("prompt texts must be non-empty and unique")
        if not isinstance(token_ids, list) or len(token_ids) < 2:
            raise ValueError("prompt token IDs must be a non-empty list")
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= GLM52_MODEL_VOCAB_SIZE
            for token_id in token_ids
        ):
            raise ValueError("prompt token ID is outside the model vocabulary")
        expected_row = {
            "token_count": len(token_ids),
            "token_ids_sha256": canonical_sha256(token_ids),
            "max_token_id": max(token_ids),
            "teacher_logit_scope": "full_vocabulary",
            "candidate_logit_scope": "full_vocabulary",
            "kld_scope": "full_vocabulary",
            "memory_clean_required": True,
            "tuning_eligible": split == "selection",
        }
        for field, expected in expected_row.items():
            if not _exact_values_equal(row.get(field), expected):
                raise ValueError(f"prompt row {prompt_id} has invalid {field}")
        prompt_ids.add(prompt_id)
        prompts.add(prompt)
        split_counts[str(split)] += 1
        domain_counts[(str(split), str(domain))] += 1
        rows.append(row)
    if split_counts != Counter({split: 22 for split in GLM52_REQUIRED_EVAL_SPLITS}):
        raise ValueError("prompt row split counts are not exact")
    for split in GLM52_REQUIRED_EVAL_SPLITS:
        if {
            domain: domain_counts[(split, domain)]
            for domain in GLM52_REQUIRED_EVAL_DOMAINS
        } != {"route": 8, "math": 7, "instruction": 7}:
            raise ValueError(f"prompt row domain counts are not exact for {split}")
    return tuple(rows)
