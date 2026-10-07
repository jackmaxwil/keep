"""Strict GLM-5.2 FP32 source-teacher cache artifact contract.

This module deliberately does not import MLX.  The cache is a filesystem trust
boundary, and its writer/auditor operate on canonical safetensors bytes with
NumPy used only for little-endian FP32 validation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import struct
import tempfile
from contextlib import contextmanager
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np


SCHEMA_VERSION: Final = 1
RECORD_TYPE: Final = "glm52_teacher_cache_fp32_v1"
MANIFEST_FILENAME: Final = "glm52-teacher-cache-fp32-manifest.json"
SHARD_DIRECTORY: Final = "teacher_logits"

PINNED_MODEL_ID: Final = "0xSero/glm-5.2-reap-504B-v2"
PINNED_REVISION: Final = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PINNED_PROFILE: Final = "glm52-reap-504b-v2"
PINNED_PROFILE_SHA256: Final = (
    "ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d"
)
PINNED_PROFILE_CONTRACT_SHA256: Final = (
    "28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8"
)
PINNED_CONFIG_SHA256: Final = (
    "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
)
PINNED_INDEX_SHA256: Final = (
    "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
)
PINNED_PROMPT_PACK_SHA256: Final = (
    "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31"
)
PINNED_PROMPT_CONTENT_SHA256: Final = (
    "cc1951aeeacda74c1315485ca5b477de344ffcd69d3d8f3ff3d44d61f84d1265"
)
PINNED_PROMPT_TEXT_SHA256: Final = (
    "3c39f6bb133907cf2fd93260a3cc548c0b186a264230568678cd076f1638eb6a"
)
PINNED_TOKENIZER_FILES: Final = {
    "config.json": {
        "size_bytes": 10_854,
        "sha256": PINNED_CONFIG_SHA256,
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

_TOP_LEVEL_FIELDS = (
    "schema_version",
    "record_type",
    "created_at",
    "producer",
    "source",
    "source_evidence",
    "non_vq_package",
    "policy",
    "prompt_authority",
    "tokenizer",
    "precision",
    "architecture",
    "totals",
    "producer_phases",
    "shards",
    "payload_integrity_pass",
    "all_producer_memory_counters_known",
    "all_producer_memory_clean",
    "system_wired_default",
    "release_eligible",
    "cache_content_sha256",
    "manifest_body_sha256",
)
_PRODUCER_FIELDS = ("implementation", "version", "mlx_version", "mlx_lm_version")
_SOURCE_FIELDS = (
    "model_id",
    "revision",
    "profile",
    "profile_sha256",
    "profile_contract_sha256",
    "config_sha256",
    "index_sha256",
    "weight_encoding",
    "decoder",
    "decoded_weight_oracle_contract",
)
_SOURCE_EVIDENCE_FIELDS = (
    "source_audit_path",
    "source_audit_file_sha256",
    "source_audit_record_type",
    "source_payload_audit_path",
    "source_payload_audit_file_sha256",
    "source_payload_audit_record_type",
    "routed_group_count",
    "routed_projection_bundle_count",
    "required_payload_shard_count",
)
_NON_VQ_FIELDS = (
    "evidence_path",
    "evidence_file_sha256",
    "manifest_sha256",
    "package_set_sha256",
    "retained_tensor_count",
    "tensor_payload_bytes",
    "bound_package_identity",
)
_POLICY_FIELDS = (
    "path",
    "file_sha256",
    "contract_sha256",
    "activation_quantization_emulated",
    "exact_w4a4_runtime_parity_claimed",
    "bf16_teacher_claimed",
    "holdout_tuning_forbidden",
    "full_vocabulary_logits_required",
)
_PROMPT_AUTHORITY_FIELDS = (
    "path",
    "file_sha256",
    "contract_sha256",
    "content_contract_sha256",
    "prompt_text_sha256",
    "ordered_prompt_ids",
)
_TOKENIZER_FIELDS = (
    "base_vocab_size",
    "tokenizer_length",
    "model_vocab_size",
    "eos_token_ids",
    "files",
)
_TOKENIZER_FILE_FIELDS = ("size_bytes", "sha256")
_PRECISION_FIELDS = (
    "decoded_weight_dtype",
    "routed_matmul_weight_dtype",
    "routed_hidden_dtype",
    "routed_output_dtype",
    "route_score_dtype",
    "non_routed_operand_dtype",
    "lm_head_operand_dtype",
    "stored_logits_dtype",
    "routed_matmul_accumulation_contract",
    "route_reduction_accumulation_contract",
    "lm_head_accumulation_contract",
    "route_reduction_order",
    "padding_mode",
    "right_padding_formula",
)
_ARCHITECTURE_FIELDS = (
    "predictor_batch_shape",
    "hidden_size",
    "experts_per_token",
    "valid_route_assignments_per_sparse_layer",
    "first_main_layer",
    "last_main_layer",
    "excluded_mtp_layer",
    "indexshare_state",
)
_TOTAL_FIELDS = (
    "prompt_count",
    "source_token_count",
    "predictor_position_count",
    "vocab_size",
    "fp32_value_count",
    "raw_tensor_bytes",
    "splits",
)
_SPLIT_FIELDS = ("position_count", "raw_tensor_bytes")
_PHASE_FIELDS = (
    "phase_id",
    "ordinal",
    "start_boundary",
    "end_boundary",
    "contribution_range",
    "pageouts_delta",
    "swapouts_delta",
    "memory_counters_known",
    "memory_clean",
    "system_wired_default",
    "input_identity_sha256",
    "output_identity_sha256",
)
_SHARD_FIELDS = (
    "prompt_id",
    "split",
    "domain",
    "tuning_eligible",
    "token_count",
    "token_ids_sha256",
    "relative_path",
    "tensor_name",
    "dtype",
    "shape",
    "element_count",
    "raw_tensor_bytes",
    "file_sha256",
    "raw_tensor_sha256",
    "producer_phase_ids",
)
_LEDGER_FIELDS = (
    "schema_version",
    "record_type",
    "bound_identity_sha256",
    "shard",
    "previous_ledger_record_sha256",
    "record_sha256",
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical finite JSON: {error}") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return str(value)


def _require_exact_keys(value: object, fields: Sequence[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} fields are not exact: missing={missing}, extra={extra}")
    return value


def _exact_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        return set(actual) == set(expected) and all(
            _exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        return len(actual) == len(expected) and all(
            _exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_exact_value(actual: object, expected: object, *, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}")


def _duplicate_key_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _strict_json_loads(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error


def _validate_logical_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    if "\\" in value:
        raise ValueError(f"{label} contains a backslash")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError(f"{label} must not be absolute")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} contains an empty or dot component")
    return value


def _validate_prompt_id(value: str) -> None:
    _validate_logical_path(value, label="prompt_id")
    if "/" in value:
        raise ValueError("prompt_id must be a canonical basename")
    if value.endswith(".safetensors"):
        raise ValueError("prompt_id must not include the shard suffix")


def _reject_symlink_path_components(path: Path, *, label: str) -> None:
    """Reject symlinks in every existing component, including the leaf."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"{label} contains a symlinked path component: {current}")


@dataclass(frozen=True)
class GLM52TeacherCachePrompt:
    prompt_id: str
    split: str
    domain: str
    tuning_eligible: bool
    encoded_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_prompt_id(self.prompt_id)
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("prompt split must be a non-empty string")
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError("prompt domain must be a non-empty string")
        if type(self.tuning_eligible) is not bool:
            raise ValueError("prompt tuning_eligible must be a boolean")
        if len(self.encoded_token_ids) < 2:
            raise ValueError("prompt encoded_token_ids must contain at least two IDs")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.encoded_token_ids):
            raise ValueError("prompt encoded_token_ids must be non-negative integers")

    @property
    def token_count(self) -> int:
        return len(self.encoded_token_ids)

    @property
    def token_ids_sha256(self) -> str:
        return canonical_sha256(list(self.encoded_token_ids))

    @classmethod
    def from_prompt_pack_row(cls, row: Mapping[str, Any]) -> GLM52TeacherCachePrompt:
        token_ids = row.get("encoded_token_ids")
        if not isinstance(token_ids, list):
            raise ValueError("prompt row encoded_token_ids must be a list")
        return cls(
            prompt_id=str(row.get("prompt_id", "")),
            split=str(row.get("split", "")),
            domain=str(row.get("domain", "")),
            tuning_eligible=row.get("tuning_eligible"),
            encoded_token_ids=tuple(token_ids),
        )


@dataclass(frozen=True)
class GLM52ProducerPhase:
    phase_id: str
    ordinal: int
    start_boundary: str
    end_boundary: str
    contribution_range: str
    pageouts_delta: int | None
    swapouts_delta: int | None
    memory_counters_known: bool
    memory_clean: bool
    system_wired_default: bool
    input_identity_sha256: str
    output_identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        phase_id: str,
        ordinal: int,
        start_boundary: str,
        end_boundary: str,
        contribution_range: str,
        pageouts_delta: int | None,
        swapouts_delta: int | None,
        system_wired_default: bool,
        input_identity_sha256: str,
        output_identity_sha256: str,
    ) -> GLM52ProducerPhase:
        for name, value in (
            ("pageouts_delta", pageouts_delta),
            ("swapouts_delta", swapouts_delta),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        known = pageouts_delta is not None and swapouts_delta is not None
        clean = known and pageouts_delta == 0 and swapouts_delta == 0
        phase = cls(
            phase_id=phase_id,
            ordinal=ordinal,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            contribution_range=contribution_range,
            pageouts_delta=pageouts_delta,
            swapouts_delta=swapouts_delta,
            memory_counters_known=known,
            memory_clean=clean,
            system_wired_default=system_wired_default,
            input_identity_sha256=input_identity_sha256,
            output_identity_sha256=output_identity_sha256,
        )
        phase.validate()
        return phase

    def validate(self) -> None:
        if not isinstance(self.phase_id, str) or not self.phase_id:
            raise ValueError("producer phase_id must be a non-empty string")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("producer phase ordinal must be a non-negative integer")
        for field in ("start_boundary", "end_boundary", "contribution_range"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"producer phase {field} must be a non-empty string")
        for field in ("pageouts_delta", "swapouts_delta"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"producer phase {field} must be a non-negative integer or null")
        known = self.pageouts_delta is not None and self.swapouts_delta is not None
        clean = known and self.pageouts_delta == 0 and self.swapouts_delta == 0
        _require_exact_value(self.memory_counters_known, known, label="producer phase memory_counters_known")
        _require_exact_value(self.memory_clean, clean, label="producer phase memory_clean")
        if type(self.system_wired_default) is not bool:
            raise ValueError("producer phase system_wired_default must be a boolean")
        _require_sha256(self.input_identity_sha256, label="producer phase input identity")
        _require_sha256(self.output_identity_sha256, label="producer phase output identity")

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in _PHASE_FIELDS}

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> GLM52ProducerPhase:
        record = _require_exact_keys(value, _PHASE_FIELDS, label=f"producer_phases[{index}]")
        phase = cls(**record)
        phase.validate()
        return phase


def _default_source() -> dict[str, Any]:
    return {
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "profile": PINNED_PROFILE,
        "profile_sha256": PINNED_PROFILE_SHA256,
        "profile_contract_sha256": PINNED_PROFILE_CONTRACT_SHA256,
        "config_sha256": PINNED_CONFIG_SHA256,
        "index_sha256": PINNED_INDEX_SHA256,
        "weight_encoding": "modelopt_nvfp4",
        "decoder": "modelopt_nvfp4_v1",
        "decoded_weight_oracle_contract": "modelopt_nvfp4_decoded_bytes_v1",
    }


def _default_precision() -> dict[str, Any]:
    return {
        "decoded_weight_dtype": "F32",
        "routed_matmul_weight_dtype": "BF16",
        "routed_hidden_dtype": "BF16",
        "routed_output_dtype": "BF16",
        "route_score_dtype": "F32",
        "non_routed_operand_dtype": "BF16",
        "lm_head_operand_dtype": "BF16",
        "stored_logits_dtype": "F32",
        "routed_matmul_accumulation_contract": "mlx_native_bf16_operands",
        "route_reduction_accumulation_contract": (
            "BF16_route_output_times_FP32_score_sum_axis_-2_then_BF16"
        ),
        "lm_head_accumulation_contract": "mlx_native_bf16_operands",
        "route_reduction_order": "route_rank",
        "padding_mode": "right_padding_pad_count",
        "right_padding_formula": "18-(token_count-1)",
    }


@dataclass(frozen=True)
class GLM52TeacherCacheContract:
    prompts: tuple[GLM52TeacherCachePrompt, ...]
    vocab_size: int
    producer: Mapping[str, Any]
    source: Mapping[str, Any]
    source_evidence: Mapping[str, Any]
    non_vq_package: Mapping[str, Any]
    policy: Mapping[str, Any]
    prompt_authority: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    precision: Mapping[str, Any]
    architecture: Mapping[str, Any]
    bound_identity_sha256: str

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("cache contract must contain at least one prompt")
        if type(self.vocab_size) is not int or self.vocab_size <= 1:
            raise ValueError("cache contract vocab_size must be an integer greater than one")
        prompt_ids = [prompt.prompt_id for prompt in self.prompts]
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("cache contract prompt IDs must be unique")
        if any(max(prompt.encoded_token_ids) >= self.vocab_size for prompt in self.prompts):
            raise ValueError("cache contract prompt token ID is outside the vocabulary")
        for prompt in self.prompts:
            if prompt.tuning_eligible != (prompt.split == "selection"):
                raise ValueError("only selection prompts may be tuning eligible")
        _require_exact_keys(self.producer, _PRODUCER_FIELDS, label="producer")
        _require_exact_keys(self.source, _SOURCE_FIELDS, label="source")
        _require_exact_keys(self.source_evidence, _SOURCE_EVIDENCE_FIELDS, label="source_evidence")
        _require_exact_keys(self.non_vq_package, _NON_VQ_FIELDS, label="non_vq_package")
        _require_exact_keys(self.policy, _POLICY_FIELDS, label="policy")
        _require_exact_keys(self.prompt_authority, _PROMPT_AUTHORITY_FIELDS, label="prompt_authority")
        _require_exact_keys(self.tokenizer, _TOKENIZER_FIELDS, label="tokenizer")
        _require_exact_keys(self.precision, _PRECISION_FIELDS, label="precision")
        _require_exact_keys(self.architecture, _ARCHITECTURE_FIELDS, label="architecture")
        _require_sha256(self.bound_identity_sha256, label="bound cache input identity")
        _require_exact_value(self.source, _default_source(), label="pinned source")
        _require_exact_value(self.precision, _default_precision(), label="precision contract")
        for field, expected in (
            ("routed_group_count", 225),
            ("routed_projection_bundle_count", 37_800),
            ("required_payload_shard_count", 61),
        ):
            _require_exact_value(
                self.source_evidence.get(field),
                expected,
                label=f"source_evidence.{field}",
            )
        _require_exact_value(
            self.non_vq_package.get("retained_tensor_count"),
            1_194,
            label="non_vq_package.retained_tensor_count",
        )
        _require_exact_value(
            self.non_vq_package.get("tensor_payload_bytes"),
            37_121_488_608,
            label="non_vq_package.tensor_payload_bytes",
        )
        for field, expected in (
            ("activation_quantization_emulated", False),
            ("exact_w4a4_runtime_parity_claimed", False),
            ("bf16_teacher_claimed", False),
            ("holdout_tuning_forbidden", True),
            ("full_vocabulary_logits_required", True),
        ):
            _require_exact_value(self.policy.get(field), expected, label=f"policy.{field}")
        _require_exact_value(
            self.prompt_authority.get("ordered_prompt_ids"),
            prompt_ids,
            label="prompt_authority.ordered_prompt_ids",
        )
        _require_exact_value(
            self.tokenizer.get("model_vocab_size"),
            self.vocab_size,
            label="tokenizer.model_vocab_size",
        )
        expected_architecture = {
            "predictor_batch_shape": [
                len(self.prompts),
                max(prompt.token_count - 1 for prompt in self.prompts),
            ],
            "hidden_size": 6_144,
            "experts_per_token": 8,
            "valid_route_assignments_per_sparse_layer": (
                sum(prompt.token_count - 1 for prompt in self.prompts) * 8
            ),
            "first_main_layer": 0,
            "last_main_layer": 77,
            "excluded_mtp_layer": 78,
            "indexshare_state": "absent",
        }
        _require_exact_value(
            self.architecture,
            expected_architecture,
            label="architecture contract",
        )
        for section_name, section in (
            ("source_evidence", self.source_evidence),
            ("non_vq_package", self.non_vq_package),
            ("policy", self.policy),
            ("prompt_authority", self.prompt_authority),
        ):
            for field, value in section.items():
                if field.endswith("sha256"):
                    _require_sha256(value, label=f"{section_name}.{field}")

    @property
    def source_token_count(self) -> int:
        return sum(prompt.token_count for prompt in self.prompts)

    @property
    def predictor_position_count(self) -> int:
        return sum(prompt.token_count - 1 for prompt in self.prompts)

    @property
    def fp32_value_count(self) -> int:
        return self.predictor_position_count * self.vocab_size

    @property
    def raw_tensor_bytes(self) -> int:
        return self.fp32_value_count * 4

    def totals(self) -> dict[str, Any]:
        split_positions: Counter[str] = Counter()
        for prompt in self.prompts:
            split_positions[prompt.split] += prompt.token_count - 1
        return {
            "prompt_count": len(self.prompts),
            "source_token_count": self.source_token_count,
            "predictor_position_count": self.predictor_position_count,
            "vocab_size": self.vocab_size,
            "fp32_value_count": self.fp32_value_count,
            "raw_tensor_bytes": self.raw_tensor_bytes,
            "splits": {
                split: {
                    "position_count": positions,
                    "raw_tensor_bytes": positions * self.vocab_size * 4,
                }
                for split, positions in sorted(split_positions.items())
            },
        }

    @classmethod
    def for_testing(
        cls,
        *,
        prompts: Sequence[GLM52TeacherCachePrompt],
        vocab_size: int,
    ) -> GLM52TeacherCacheContract:
        prompt_tuple = tuple(prompts)
        prompt_rows = [
            {
                "prompt_id": prompt.prompt_id,
                "split": prompt.split,
                "domain": prompt.domain,
                "tuning_eligible": prompt.tuning_eligible,
                "encoded_token_ids": list(prompt.encoded_token_ids),
            }
            for prompt in prompt_tuple
        ]
        prompt_digest = canonical_sha256(prompt_rows)
        dummy = lambda label: hashlib.sha256(label.encode()).hexdigest()
        source_evidence = {
            "source_audit_path": "artifacts/audits/source-audit.json",
            "source_audit_file_sha256": dummy("source-audit"),
            "source_audit_record_type": "glm52_source_audit_v1",
            "source_payload_audit_path": "artifacts/audits/source-payload-audit.json",
            "source_payload_audit_file_sha256": dummy("source-payload-audit"),
            "source_payload_audit_record_type": "glm52_source_payload_audit_v1",
            "routed_group_count": 225,
            "routed_projection_bundle_count": 37_800,
            "required_payload_shard_count": 61,
        }
        non_vq_package = {
            "evidence_path": "artifacts/audits/non-vq-package.json",
            "evidence_file_sha256": dummy("non-vq-evidence"),
            "manifest_sha256": dummy("non-vq-manifest"),
            "package_set_sha256": dummy("non-vq-set"),
            "retained_tensor_count": 1_194,
            "tensor_payload_bytes": 37_121_488_608,
            "bound_package_identity": dummy("bound-non-vq-package"),
        }
        policy = {
            "path": "artifacts/quality/glm52-family-policy.json",
            "file_sha256": dummy("policy-file"),
            "contract_sha256": dummy("policy-contract"),
            "activation_quantization_emulated": False,
            "exact_w4a4_runtime_parity_claimed": False,
            "bf16_teacher_claimed": False,
            "holdout_tuning_forbidden": True,
            "full_vocabulary_logits_required": True,
        }
        prompt_authority = {
            "path": "artifacts/quality/tiny-glm52-prompts.json",
            "file_sha256": dummy("tiny-prompt-file"),
            "contract_sha256": prompt_digest,
            "content_contract_sha256": prompt_digest,
            "prompt_text_sha256": dummy("tiny-prompt-text"),
            "ordered_prompt_ids": [prompt.prompt_id for prompt in prompt_tuple],
        }
        tokenizer = {
            "base_vocab_size": vocab_size,
            "tokenizer_length": vocab_size,
            "model_vocab_size": vocab_size,
            "eos_token_ids": [vocab_size - 1],
            "files": {
                "tokenizer.json": {
                    "size_bytes": 1,
                    "sha256": dummy("tiny-tokenizer"),
                }
            },
        }
        max_positions = max(prompt.token_count - 1 for prompt in prompt_tuple)
        architecture = {
            "predictor_batch_shape": [len(prompt_tuple), max_positions],
            "hidden_size": 6_144,
            "experts_per_token": 8,
            "valid_route_assignments_per_sparse_layer": sum(
                prompt.token_count - 1 for prompt in prompt_tuple
            )
            * 8,
            "first_main_layer": 0,
            "last_main_layer": 77,
            "excluded_mtp_layer": 78,
            "indexshare_state": "absent",
        }
        producer = {
            # Frozen identifier recorded in existing artifacts; keeps its pre-rename module path.
            "implementation": "mlx_vq.quality.glm52_teacher_cache",
            "version": "1",
            "mlx_version": "test",
            "mlx_lm_version": "test",
        }
        bound_identity = canonical_sha256(
            {
                "source": _default_source(),
                "prompt": prompt_digest,
                "non_vq": non_vq_package["bound_package_identity"],
                "policy": policy["contract_sha256"],
            }
        )
        return cls(
            prompts=prompt_tuple,
            vocab_size=vocab_size,
            producer=producer,
            source=_default_source(),
            source_evidence=source_evidence,
            non_vq_package=non_vq_package,
            policy=policy,
            prompt_authority=prompt_authority,
            tokenizer=tokenizer,
            precision=_default_precision(),
            architecture=architecture,
            bound_identity_sha256=bound_identity,
        )

    @classmethod
    def from_frozen_prompt_pack(
        cls,
        prompt_pack_path: str | Path,
        *,
        producer: Mapping[str, Any],
        source_evidence: Mapping[str, Any],
        non_vq_package: Mapping[str, Any],
    ) -> GLM52TeacherCacheContract:
        """Build the production 66-row contract from the authenticated pack."""

        path = Path(prompt_pack_path)
        _reject_symlink_path_components(path, label="prompt pack path")
        raw, _identity = _read_regular_file_stable(path)
        file_sha256 = hashlib.sha256(raw).hexdigest()
        _require_exact_value(
            file_sha256,
            PINNED_PROMPT_PACK_SHA256,
            label="frozen prompt pack file SHA-256",
        )
        payload = _strict_json_loads(raw, label="frozen prompt pack")
        if not isinstance(payload, Mapping):
            raise ValueError("frozen prompt pack must be a JSON object")
        final_body = dict(payload)
        embedded_contract = final_body.pop("prompt_pack_contract_sha256", None)
        _require_exact_value(
            embedded_contract,
            canonical_sha256(final_body),
            label="prompt pack contract SHA-256",
        )
        _require_exact_value(
            payload.get("prompt_content_contract_sha256"),
            PINNED_PROMPT_CONTENT_SHA256,
            label="prompt content contract SHA-256",
        )
        expected_scalars = {
            "schema_version": 1,
            "record_type": "glm52_family_eval_prompt_pack",
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_REVISION,
            "profile": PINNED_PROFILE,
            "prompt_row_count": 66,
            "expected_vocab_size": 154_880,
            "prompt_text_sha256": PINNED_PROMPT_TEXT_SHA256,
            "holdout_tuning_forbidden": True,
        }
        for field, expected in expected_scalars.items():
            _require_exact_value(payload.get(field), expected, label=f"prompt pack {field}")
        raw_rows = payload.get("prompt_rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != 66:
            raise ValueError("frozen prompt pack must contain exactly 66 rows")
        prompts: list[GLM52TeacherCachePrompt] = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"prompt_rows[{index}] must be an object")
            prompt = GLM52TeacherCachePrompt.from_prompt_pack_row(row)
            _require_exact_value(row.get("token_count"), prompt.token_count, label=f"prompt_rows[{index}] token_count")
            _require_exact_value(row.get("token_ids_sha256"), prompt.token_ids_sha256, label=f"prompt_rows[{index}] token_ids_sha256")
            _require_exact_value(row.get("model_id"), PINNED_MODEL_ID, label=f"prompt_rows[{index}] model_id")
            _require_exact_value(row.get("revision"), PINNED_REVISION, label=f"prompt_rows[{index}] revision")
            prompts.append(prompt)
        if len({prompt.prompt_id for prompt in prompts}) != 66:
            raise ValueError("frozen prompt pack prompt IDs must be unique")
        split_counts = Counter(prompt.split for prompt in prompts)
        _require_exact_value(
            dict(split_counts),
            {"report": 22, "selection": 22, "holdout": 22},
            label="frozen prompt split counts",
        )
        split_positions = Counter()
        for prompt in prompts:
            split_positions[prompt.split] += prompt.token_count - 1
        _require_exact_value(
            dict(split_positions),
            {"report": 238, "selection": 255, "holdout": 251},
            label="frozen prompt split position counts",
        )
        _require_exact_value(
            sum(prompt.token_count for prompt in prompts),
            810,
            label="frozen prompt source token count",
        )

        _require_exact_keys(producer, _PRODUCER_FIELDS, label="producer")
        _require_exact_keys(source_evidence, _SOURCE_EVIDENCE_FIELDS, label="source_evidence")
        _require_exact_keys(non_vq_package, _NON_VQ_FIELDS, label="non_vq_package")
        for field, expected in (
            ("routed_group_count", 225),
            ("routed_projection_bundle_count", 37_800),
            ("required_payload_shard_count", 61),
        ):
            _require_exact_value(source_evidence.get(field), expected, label=f"source_evidence.{field}")
        _require_exact_value(non_vq_package.get("retained_tensor_count"), 1_194, label="non_vq_package.retained_tensor_count")
        _require_exact_value(non_vq_package.get("tensor_payload_bytes"), 37_121_488_608, label="non_vq_package.tensor_payload_bytes")

        policy = {
            "path": "artifacts/quality/glm52-family-policy-20260709-v2.json",
            "file_sha256": payload.get("family_policy_sha256"),
            "contract_sha256": payload.get("family_policy_contract_sha256"),
            "activation_quantization_emulated": False,
            "exact_w4a4_runtime_parity_claimed": False,
            "bf16_teacher_claimed": False,
            "holdout_tuning_forbidden": True,
            "full_vocabulary_logits_required": True,
        }
        prompt_authority = {
            "path": "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json",
            "file_sha256": PINNED_PROMPT_PACK_SHA256,
            "contract_sha256": embedded_contract,
            "content_contract_sha256": PINNED_PROMPT_CONTENT_SHA256,
            "prompt_text_sha256": PINNED_PROMPT_TEXT_SHA256,
            "ordered_prompt_ids": [prompt.prompt_id for prompt in prompts],
        }
        tokenizer = {
            "base_vocab_size": 154_820,
            "tokenizer_length": 154_856,
            "model_vocab_size": 154_880,
            "eos_token_ids": [154_820, 154_827, 154_829],
            "files": json.loads(_canonical_json_bytes(PINNED_TOKENIZER_FILES)),
        }
        architecture = {
            "predictor_batch_shape": [66, 18],
            "hidden_size": 6_144,
            "experts_per_token": 8,
            "valid_route_assignments_per_sparse_layer": 5_952,
            "first_main_layer": 0,
            "last_main_layer": 77,
            "excluded_mtp_layer": 78,
            "indexshare_state": "absent",
        }
        bound_identity = canonical_sha256(
            {
                "source": _default_source(),
                "source_evidence": source_evidence,
                "non_vq_package": non_vq_package,
                "policy": policy,
                "prompt_authority": prompt_authority,
                "tokenizer": tokenizer,
                "precision": _default_precision(),
            }
        )
        return cls(
            prompts=tuple(prompts),
            vocab_size=154_880,
            producer=dict(producer),
            source=_default_source(),
            source_evidence=dict(source_evidence),
            non_vq_package=dict(non_vq_package),
            policy=policy,
            prompt_authority=prompt_authority,
            tokenizer=tokenizer,
            precision=_default_precision(),
            architecture=architecture,
            bound_identity_sha256=bound_identity,
        )


@dataclass(frozen=True)
class GLM52TeacherCacheShard:
    prompt_id: str
    split: str
    domain: str
    tuning_eligible: bool
    token_count: int
    token_ids_sha256: str
    relative_path: str
    tensor_name: str
    dtype: str
    shape: tuple[int, int]
    element_count: int
    raw_tensor_bytes: int
    file_sha256: str
    raw_tensor_sha256: str
    producer_phase_ids: tuple[str, ...]

    @classmethod
    def from_logits_bytes(
        cls,
        *,
        prompt: GLM52TeacherCachePrompt,
        vocab_size: int,
        relative_path: str,
        file_bytes: bytes,
        raw_tensor_bytes: bytes,
        producer_phase_ids: Sequence[str],
    ) -> GLM52TeacherCacheShard:
        positions = prompt.token_count - 1
        return cls(
            prompt_id=prompt.prompt_id,
            split=prompt.split,
            domain=prompt.domain,
            tuning_eligible=prompt.tuning_eligible,
            token_count=prompt.token_count,
            token_ids_sha256=prompt.token_ids_sha256,
            relative_path=relative_path,
            tensor_name="logits",
            dtype="F32",
            shape=(positions, vocab_size),
            element_count=positions * vocab_size,
            raw_tensor_bytes=positions * vocab_size * 4,
            file_sha256=hashlib.sha256(file_bytes).hexdigest(),
            raw_tensor_sha256=hashlib.sha256(raw_tensor_bytes).hexdigest(),
            producer_phase_ids=tuple(producer_phase_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in _SHARD_FIELDS}
        result["shape"] = list(self.shape)
        result["producer_phase_ids"] = list(self.producer_phase_ids)
        return result

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> GLM52TeacherCacheShard:
        record = _require_exact_keys(value, _SHARD_FIELDS, label=f"shards[{index}]")
        shape = record["shape"]
        phase_ids = record["producer_phase_ids"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"shards[{index}] shape must be a two-item list")
        if not isinstance(phase_ids, list):
            raise ValueError(f"shards[{index}] producer_phase_ids must be a list")
        return cls(
            **{
                **record,
                "shape": tuple(shape),
                "producer_phase_ids": tuple(phase_ids),
            }
        )


def compute_cache_content_sha256(manifest: Mapping[str, Any]) -> str:
    """Compute the stable content identity without provenance/self digests."""

    basis = {
        field: manifest[field]
        for field in (
            "schema_version",
            "record_type",
            "producer",
            "source",
            "source_evidence",
            "non_vq_package",
            "policy",
            "prompt_authority",
            "tokenizer",
            "precision",
            "architecture",
            "totals",
            "producer_phases",
        )
    }
    basis["shards"] = [
        {
            "prompt_id": shard["prompt_id"],
            "file_sha256": shard["file_sha256"],
            "raw_tensor_sha256": shard["raw_tensor_sha256"],
        }
        for shard in manifest["shards"]
    ]
    return canonical_sha256(basis)


def compute_manifest_body_sha256(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_body_sha256", None)
    return canonical_sha256(body)


@dataclass(frozen=True)
class GLM52TeacherCacheManifest:
    schema_version: int
    record_type: str
    created_at: str
    producer: Mapping[str, Any]
    source: Mapping[str, Any]
    source_evidence: Mapping[str, Any]
    non_vq_package: Mapping[str, Any]
    policy: Mapping[str, Any]
    prompt_authority: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    precision: Mapping[str, Any]
    architecture: Mapping[str, Any]
    totals: Mapping[str, Any]
    producer_phases: tuple[GLM52ProducerPhase, ...]
    shards: tuple[GLM52TeacherCacheShard, ...]
    payload_integrity_pass: bool
    all_producer_memory_counters_known: bool
    all_producer_memory_clean: bool
    system_wired_default: bool
    release_eligible: bool
    cache_content_sha256: str
    manifest_body_sha256: str

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in _TOP_LEVEL_FIELDS:
            value = getattr(self, field)
            if field == "producer_phases":
                value = [phase.to_dict() for phase in self.producer_phases]
            elif field == "shards":
                value = [shard.to_dict() for shard in self.shards]
            else:
                value = json.loads(_canonical_json_bytes(value))
            result[field] = value
        return result


def build_glm52_teacher_cache_manifest(
    *,
    contract: GLM52TeacherCacheContract,
    shards: Sequence[GLM52TeacherCacheShard],
    producer_phases: Sequence[GLM52ProducerPhase],
    created_at: str,
) -> GLM52TeacherCacheManifest:
    shard_tuple = tuple(shards)
    phase_tuple = tuple(producer_phases)
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("manifest created_at must be a non-empty string")
    if [shard.prompt_id for shard in shard_tuple] != [
        prompt.prompt_id for prompt in contract.prompts
    ]:
        raise ValueError("ordered shard inventory must exactly match frozen prompt order")
    if not phase_tuple:
        raise ValueError("producer phase inventory must not be empty")
    if [phase.ordinal for phase in phase_tuple] != list(range(len(phase_tuple))):
        raise ValueError("producer phase ordinals must be contiguous and ordered from zero")
    phase_ids = [phase.phase_id for phase in phase_tuple]
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError("producer phase IDs must be unique")
    for phase in phase_tuple:
        phase.validate()
    for index, (shard, prompt) in enumerate(zip(shard_tuple, contract.prompts, strict=True)):
        _validate_shard_record(shard, prompt=prompt, contract=contract, index=index, phase_ids=set(phase_ids))

    known = all(phase.memory_counters_known for phase in phase_tuple)
    clean = known and all(phase.memory_clean for phase in phase_tuple)
    wired_default = all(phase.system_wired_default for phase in phase_tuple)
    payload_pass = True
    release = payload_pass and known and clean and wired_default
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "created_at": created_at,
        "producer": dict(contract.producer),
        "source": dict(contract.source),
        "source_evidence": dict(contract.source_evidence),
        "non_vq_package": dict(contract.non_vq_package),
        "policy": dict(contract.policy),
        "prompt_authority": dict(contract.prompt_authority),
        "tokenizer": dict(contract.tokenizer),
        "precision": dict(contract.precision),
        "architecture": dict(contract.architecture),
        "totals": contract.totals(),
        "producer_phases": [phase.to_dict() for phase in phase_tuple],
        "shards": [shard.to_dict() for shard in shard_tuple],
        "payload_integrity_pass": payload_pass,
        "all_producer_memory_counters_known": known,
        "all_producer_memory_clean": clean,
        "system_wired_default": wired_default,
        "release_eligible": release,
        "cache_content_sha256": "",
        "manifest_body_sha256": "",
    }
    base["cache_content_sha256"] = compute_cache_content_sha256(base)
    base["manifest_body_sha256"] = compute_manifest_body_sha256(base)
    return GLM52TeacherCacheManifest(
        **{
            **base,
            "producer_phases": phase_tuple,
            "shards": shard_tuple,
        }
    )


def _validate_shard_record(
    shard: GLM52TeacherCacheShard,
    *,
    prompt: GLM52TeacherCachePrompt,
    contract: GLM52TeacherCacheContract,
    index: int,
    phase_ids: set[str],
) -> None:
    _validate_logical_path(shard.relative_path, label=f"shards[{index}] relative_path")
    expected = {
        "prompt_id": prompt.prompt_id,
        "split": prompt.split,
        "domain": prompt.domain,
        "tuning_eligible": prompt.tuning_eligible,
        "token_count": prompt.token_count,
        "token_ids_sha256": prompt.token_ids_sha256,
        "relative_path": f"{SHARD_DIRECTORY}/{prompt.prompt_id}.safetensors",
        "tensor_name": "logits",
        "dtype": "F32",
        "shape": (prompt.token_count - 1, contract.vocab_size),
        "element_count": (prompt.token_count - 1) * contract.vocab_size,
        "raw_tensor_bytes": (prompt.token_count - 1) * contract.vocab_size * 4,
    }
    for field, expected_value in expected.items():
        _require_exact_value(getattr(shard, field), expected_value, label=f"shards[{index}] {field}")
    _require_sha256(shard.file_sha256, label=f"shards[{index}] file_sha256")
    _require_sha256(shard.raw_tensor_sha256, label=f"shards[{index}] raw_tensor_sha256")
    if not shard.producer_phase_ids:
        raise ValueError(f"shards[{index}] producer_phase_ids must not be empty")
    if len(set(shard.producer_phase_ids)) != len(shard.producer_phase_ids):
        raise ValueError(f"shards[{index}] producer_phase_ids must not contain duplicates")
    if any(phase_id not in phase_ids for phase_id in shard.producer_phase_ids):
        raise ValueError(f"shards[{index}] references an unknown producer phase")


def _safetensors_bytes(logits: np.ndarray) -> tuple[bytes, bytes]:
    contiguous = np.ascontiguousarray(logits, dtype="<f4")
    raw = contiguous.tobytes(order="C")
    header = _canonical_json_bytes(
        {
            "logits": {
                "dtype": "F32",
                "shape": list(contiguous.shape),
                "data_offsets": [0, len(raw)],
            }
        }
    )
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header + raw, raw


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_publish_bytes(final_path: Path, payload: bytes) -> None:
    if final_path.exists() or final_path.is_symlink():
        raise ValueError(f"refusing to overwrite existing final file {final_path}")
    parent = final_path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"publication directory is not a real directory: {parent}")
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if final_path.exists() or final_path.is_symlink():
            raise ValueError(f"refusing to overwrite concurrently created final file {final_path}")
        try:
            os.link(temp_path, final_path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite concurrently published final file {final_path}"
            ) from error
        temp_path.unlink()
        _fsync_directory(parent)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class GLM52TeacherCacheShardWrite:
    shard: GLM52TeacherCacheShard
    ledger_record: dict[str, Any]
    reused: bool


def _ledger_for_shard(
    shard: GLM52TeacherCacheShard,
    *,
    bound_identity_sha256: str,
    previous_ledger_record_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_teacher_cache_shard_ledger_v1",
        "bound_identity_sha256": bound_identity_sha256,
        "shard": shard.to_dict(),
        "previous_ledger_record_sha256": previous_ledger_record_sha256,
    }
    return {**body, "record_sha256": canonical_sha256(body)}


def _ledger_genesis(bound_identity_sha256: str) -> str:
    return canonical_sha256(
        {
            "record_type": "glm52_teacher_cache_shard_ledger_genesis_v1",
            "bound_identity_sha256": bound_identity_sha256,
        }
    )


def _external_ledger_path(cache_root: Path, ledger_path: str | Path) -> Path:
    ledger = Path(ledger_path)
    _reject_symlink_path_components(cache_root, label="cache root")
    _reject_symlink_path_components(ledger, label="teacher cache shard ledger")
    root_absolute = Path(os.path.abspath(cache_root))
    ledger_absolute = Path(os.path.abspath(ledger))
    try:
        ledger_absolute.relative_to(root_absolute)
    except ValueError:
        pass
    else:
        raise ValueError("teacher cache shard ledger must remain outside the cache root")
    parent = ledger_absolute.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError("teacher cache shard ledger directory must be a real directory")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(parent, label="teacher cache shard ledger directory")
    if ledger_absolute.exists() and (
        ledger_absolute.is_symlink() or not ledger_absolute.is_file()
    ):
        raise ValueError("teacher cache shard ledger must be a regular non-symlink file")
    return ledger_absolute


@contextmanager
def _exclusive_ledger_lock(ledger_path: Path) -> Iterator[None]:
    lock_path = Path(f"{ledger_path}.lock")
    _reject_symlink_path_components(lock_path, label="teacher cache run lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ValueError(f"cannot open teacher cache run lock: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("teacher cache run lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_shard_ledger(
    ledger_path: Path,
    *,
    bound_identity_sha256: str | None,
) -> tuple[dict[str, Any], ...]:
    if not ledger_path.exists() and not ledger_path.is_symlink():
        return ()
    try:
        raw, _identity = _read_regular_file_stable(ledger_path)
    except ValueError as error:
        raise ValueError(f"teacher cache shard ledger is invalid: {error}") from error
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    expected_bound_identity = bound_identity_sha256
    previous_record_sha256: str | None = None
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or not line.strip():
            raise ValueError(
                f"teacher cache shard ledger line {line_number} is not a complete JSON line"
            )
        value = _strict_json_loads(
            line,
            label=f"teacher cache shard ledger line {line_number}",
        )
        record = dict(
            _require_exact_keys(
                value,
                _LEDGER_FIELDS,
                label=f"teacher cache shard ledger line {line_number}",
            )
        )
        _require_exact_value(
            record["schema_version"],
            1,
            label=f"teacher cache shard ledger line {line_number} schema_version",
        )
        _require_exact_value(
            record["record_type"],
            "glm52_teacher_cache_shard_ledger_v1",
            label=f"teacher cache shard ledger line {line_number} record_type",
        )
        record_bound_identity = _require_sha256(
            record["bound_identity_sha256"],
            label=f"teacher cache shard ledger line {line_number} bound identity",
        )
        if expected_bound_identity is None:
            expected_bound_identity = record_bound_identity
        _require_exact_value(
            record_bound_identity,
            expected_bound_identity,
            label=f"teacher cache shard ledger line {line_number} bound identity",
        )
        shard = GLM52TeacherCacheShard.from_mapping(record["shard"], index=line_number - 1)
        _validate_logical_path(
            shard.relative_path,
            label=f"teacher cache shard ledger line {line_number} relative_path",
        )
        _require_sha256(
            shard.file_sha256,
            label=f"teacher cache shard ledger line {line_number} file SHA-256",
        )
        _require_sha256(
            shard.raw_tensor_sha256,
            label=f"teacher cache shard ledger line {line_number} raw tensor SHA-256",
        )
        if shard.relative_path in seen_paths:
            raise ValueError(
                f"teacher cache shard ledger duplicates {shard.relative_path}"
            )
        seen_paths.add(shard.relative_path)
        if previous_record_sha256 is None:
            assert expected_bound_identity is not None
            previous_record_sha256 = _ledger_genesis(expected_bound_identity)
        _require_exact_value(
            record["previous_ledger_record_sha256"],
            previous_record_sha256,
            label=f"teacher cache shard ledger line {line_number} previous-record chain SHA-256",
        )
        record_sha256 = _require_sha256(
            record["record_sha256"],
            label=f"teacher cache shard ledger line {line_number} record SHA-256",
        )
        body = dict(record)
        body.pop("record_sha256")
        if record_sha256 != canonical_sha256(body):
            raise ValueError(
                f"teacher cache shard ledger line {line_number} record SHA-256 mismatch"
            )
        record["shard"] = shard.to_dict()
        records.append(record)
        previous_record_sha256 = record_sha256
    return tuple(records)


def _append_shard_ledger_record(ledger_path: Path, record: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(record) + b"\n"
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(ledger_path, flags, 0o600)
    except OSError as error:
        raise ValueError(f"cannot append teacher cache shard ledger: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("teacher cache shard ledger is not a regular file")
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short teacher cache shard ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(ledger_path.parent)


def _require_manifest_ledger_inventory(
    records: Sequence[Mapping[str, Any]],
    shards: Sequence[GLM52TeacherCacheShard],
) -> None:
    expected = [shard.to_dict() for shard in shards]
    actual = [dict(record["shard"]) for record in records]
    if actual != expected:
        raise ValueError(
            "cannot publish manifest unless shards and durable ledger records "
            "are one-to-one in canonical prompt order"
        )


def write_glm52_teacher_cache_shard(
    cache_root: str | Path,
    *,
    ledger_path: str | Path,
    prompt: GLM52TeacherCachePrompt,
    logits: np.ndarray,
    vocab_size: int,
    producer_phase_ids: Sequence[str],
    bound_identity_sha256: str,
    existing_ledger_record: Mapping[str, Any] | None = None,
) -> GLM52TeacherCacheShardWrite:
    """Durably publish or strictly reuse one canonical FP32 prompt shard."""

    _require_sha256(bound_identity_sha256, label="bound_identity_sha256")
    _validate_prompt_id(prompt.prompt_id)
    expected_shape = (prompt.token_count - 1, vocab_size)
    array = np.asarray(logits)
    if array.dtype != np.float32:
        raise ValueError(f"logits dtype must be float32, found {array.dtype}")
    if array.shape != expected_shape:
        raise ValueError(f"logits shape must be {expected_shape}, found {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError("logits must be contiguous row-major")
    if not np.isfinite(array).all():
        raise ValueError("logits contain a non-finite value")
    for row_index, row in enumerate(array):
        if row.size and np.all(row == row[0]):
            raise ValueError(f"logits contain constant causal row {row_index}")
    if not producer_phase_ids or any(
        not isinstance(phase_id, str) or not phase_id for phase_id in producer_phase_ids
    ):
        raise ValueError("producer_phase_ids must contain non-empty strings")

    root = Path(cache_root)
    durable_ledger_path = _external_ledger_path(root, ledger_path)
    _reject_symlink_path_components(root, label="cache root")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("cache root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    logits_dir = root / SHARD_DIRECTORY
    if logits_dir.exists() and (logits_dir.is_symlink() or not logits_dir.is_dir()):
        raise ValueError("teacher_logits must be a real directory")
    logits_dir.mkdir(exist_ok=True)
    final_path = logits_dir / f"{prompt.prompt_id}.safetensors"
    relative_path = f"{SHARD_DIRECTORY}/{prompt.prompt_id}.safetensors"
    file_bytes, raw_bytes = _safetensors_bytes(array)
    expected_shard = GLM52TeacherCacheShard.from_logits_bytes(
        prompt=prompt,
        vocab_size=vocab_size,
        relative_path=relative_path,
        file_bytes=file_bytes,
        raw_tensor_bytes=raw_bytes,
        producer_phase_ids=producer_phase_ids,
    )
    with _exclusive_ledger_lock(durable_ledger_path):
        ledger_records = _read_shard_ledger(
            durable_ledger_path,
            bound_identity_sha256=bound_identity_sha256,
        )
        matching_records = [
            record
            for record in ledger_records
            if record["shard"]["relative_path"] == relative_path
        ]

        if final_path.exists() or final_path.is_symlink():
            if not matching_records:
                raise ValueError("existing shard has no authenticated ledger record")
            ledger = matching_records[0]
            try:
                _require_exact_value(
                    ledger["shard"],
                    expected_shard.to_dict(),
                    label="existing shard ledger",
                )
                if existing_ledger_record is not None:
                    record = _require_exact_keys(
                        existing_ledger_record,
                        _LEDGER_FIELDS,
                        label="existing shard ledger assertion",
                    )
                    _require_exact_value(
                        dict(record),
                        ledger,
                        label="existing shard ledger assertion",
                    )
                actual = _audit_shard(
                    final_path,
                    prompt=prompt,
                    vocab_size=vocab_size,
                    expected=expected_shard,
                )
                _require_exact_value(actual.file_sha256, expected_shard.file_sha256, label="existing shard file SHA-256")
                _require_exact_value(actual.raw_tensor_sha256, expected_shard.raw_tensor_sha256, label="existing shard raw tensor SHA-256")
            except ValueError as error:
                raise ValueError(f"existing shard is invalid and will not be overwritten: {error}") from error
            return GLM52TeacherCacheShardWrite(shard=expected_shard, ledger_record=ledger, reused=True)

        if matching_records or existing_ledger_record is not None:
            raise ValueError("existing shard ledger was supplied but the final shard is missing")
        previous_ledger_record_sha256 = (
            str(ledger_records[-1]["record_sha256"])
            if ledger_records
            else _ledger_genesis(bound_identity_sha256)
        )
        ledger = _ledger_for_shard(
            expected_shard,
            bound_identity_sha256=bound_identity_sha256,
            previous_ledger_record_sha256=previous_ledger_record_sha256,
        )
        _durable_publish_bytes(final_path, file_bytes)
        _append_shard_ledger_record(durable_ledger_path, ledger)
        return GLM52TeacherCacheShardWrite(shard=expected_shard, ledger_record=ledger, reused=False)


def publish_glm52_teacher_cache_manifest(
    cache_root: str | Path,
    manifest: GLM52TeacherCacheManifest,
    *,
    ledger_path: str | Path,
    bound_identity_sha256: str,
) -> Path:
    """Publish the manifest last, after checking the exact final shard tree."""

    root = Path(cache_root)
    _require_sha256(bound_identity_sha256, label="bound_identity_sha256")
    durable_ledger_path = _external_ledger_path(root, ledger_path)
    _reject_symlink_path_components(root, label="cache root")
    with _exclusive_ledger_lock(durable_ledger_path):
        ledger_records = _read_shard_ledger(
            durable_ledger_path,
            bound_identity_sha256=bound_identity_sha256,
        )
        _require_manifest_ledger_inventory(ledger_records, manifest.shards)
        expected_names = {f"{prompt.prompt_id}.safetensors" for prompt in manifest.shards}
        logits_dir = root / SHARD_DIRECTORY
        if root.is_symlink() or logits_dir.is_symlink() or not logits_dir.is_dir():
            raise ValueError("cache root and teacher_logits must be real directories")
        actual_names = {entry.name for entry in os.scandir(logits_dir)}
        if actual_names != expected_names:
            raise ValueError(
                "cannot publish manifest before exact shard tree is complete: "
                f"missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
            )
        for entry in os.scandir(logits_dir):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"cannot publish manifest with non-regular shard {entry.name}")
        vocab_size = int(manifest.totals["vocab_size"])
        for shard in manifest.shards:
            prompt_view = GLM52TeacherCachePrompt(
                prompt_id=shard.prompt_id,
                split=shard.split,
                domain=shard.domain,
                tuning_eligible=shard.tuning_eligible,
                encoded_token_ids=tuple(0 for _ in range(shard.token_count)),
            )
            _audit_shard(
                root / Path(*PurePosixPath(shard.relative_path).parts),
                prompt=prompt_view,
                vocab_size=vocab_size,
                expected=shard,
            )
        final_ledger_records = _read_shard_ledger(
            durable_ledger_path,
            bound_identity_sha256=bound_identity_sha256,
        )
        _require_exact_value(
            final_ledger_records,
            ledger_records,
            label="durable ledger identity during manifest publication",
        )
        _require_manifest_ledger_inventory(final_ledger_records, manifest.shards)
        payload = _canonical_json_bytes(manifest.to_dict()) + b"\n"
        final_path = root / MANIFEST_FILENAME
        _durable_publish_bytes(final_path, payload)
        return final_path


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _identity_from_stat(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
    )


def _assert_file_identity_unchanged(
    path: Path, before: _FileIdentity, after: _FileIdentity
) -> None:
    try:
        current = _identity_from_stat(path.lstat())
    except FileNotFoundError as error:
        raise ValueError(f"file changed during audit: {path}") from error
    if before != after or before != current:
        raise ValueError(f"file changed during audit: {path}")


def _open_nofollow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot no-follow open regular file {path}: {error}") from error


def _read_regular_file_stable(path: Path) -> tuple[bytes, _FileIdentity]:
    try:
        link_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing file {path}") from error
    if stat.S_ISLNK(link_stat.st_mode):
        raise ValueError(f"file is a symlink: {path}")
    if not stat.S_ISREG(link_stat.st_mode):
        raise ValueError(f"file is special, not regular: {path}")
    fd = _open_nofollow(path)
    try:
        before = _identity_from_stat(os.fstat(fd))
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = _identity_from_stat(os.fstat(fd))
    finally:
        os.close(fd)
    _assert_file_identity_unchanged(path, before, after)
    return b"".join(chunks), before


@dataclass(frozen=True)
class _AuditedShard:
    file_sha256: str
    raw_tensor_sha256: str
    element_count: int
    raw_tensor_bytes: int


def _audit_shard(
    path: Path,
    *,
    prompt: GLM52TeacherCachePrompt,
    vocab_size: int,
    expected: GLM52TeacherCacheShard,
) -> _AuditedShard:
    link_stat = path.lstat()
    if stat.S_ISLNK(link_stat.st_mode):
        raise ValueError(f"shard is a symlink: {path}")
    if not stat.S_ISREG(link_stat.st_mode):
        raise ValueError(f"shard is a special file: {path}")
    fd = _open_nofollow(path)
    try:
        before = _identity_from_stat(os.fstat(fd))
        prefix = os.read(fd, 8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors shard has a truncated length prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > 1024 * 1024:
            raise ValueError(f"safetensors header size is invalid: {header_size}")
        header_raw = b""
        while len(header_raw) < header_size:
            chunk = os.read(fd, header_size - len(header_raw))
            if not chunk:
                raise ValueError(f"safetensors header is truncated: {path}")
            header_raw += chunk
        stripped_header = header_raw.rstrip(b" ")
        if header_raw[len(stripped_header) :] != b" " * (len(header_raw) - len(stripped_header)):
            raise ValueError("safetensors header has non-space padding")
        header = _strict_json_loads(stripped_header, label=f"safetensors header {path}")
        header_object = _require_exact_keys(header, ("logits",), label="safetensors tensor inventory")
        descriptor = _require_exact_keys(
            header_object["logits"],
            ("dtype", "shape", "data_offsets"),
            label="safetensors logits descriptor",
        )
        positions = prompt.token_count - 1
        expected_raw_bytes = positions * vocab_size * 4
        _require_exact_value(descriptor["dtype"], "F32", label="safetensors logits dtype")
        _require_exact_value(descriptor["shape"], [positions, vocab_size], label="safetensors logits shape")
        _require_exact_value(descriptor["data_offsets"], [0, expected_raw_bytes], label="safetensors logits offsets")
        expected_extent = 8 + header_size + expected_raw_bytes
        if before.size != expected_extent:
            if before.size > expected_extent:
                raise ValueError(
                    f"safetensors physical extent has trailing bytes: expected {expected_extent}, found {before.size}"
                )
            raise ValueError(
                f"safetensors physical extent is truncated: expected {expected_extent}, found {before.size}"
            )

        file_digest = hashlib.sha256(prefix + header_raw)
        raw_digest = hashlib.sha256()
        row_bytes = vocab_size * 4
        for row_index in range(positions):
            raw_row = b""
            while len(raw_row) < row_bytes:
                chunk = os.read(fd, row_bytes - len(raw_row))
                if not chunk:
                    raise ValueError(f"safetensors tensor row {row_index} is truncated")
                raw_row += chunk
            file_digest.update(raw_row)
            raw_digest.update(raw_row)
            values = np.frombuffer(raw_row, dtype="<f4")
            if not np.isfinite(values).all():
                raise ValueError(f"shard {prompt.prompt_id} row {row_index} contains a non-finite value")
            if values.size and np.all(values == values[0]):
                raise ValueError(f"shard {prompt.prompt_id} has constant causal row {row_index}")
        if os.read(fd, 1):
            raise ValueError("safetensors shard contains trailing bytes")
        after = _identity_from_stat(os.fstat(fd))
    finally:
        os.close(fd)
    _assert_file_identity_unchanged(path, before, after)
    actual = _AuditedShard(
        file_sha256=file_digest.hexdigest(),
        raw_tensor_sha256=raw_digest.hexdigest(),
        element_count=positions * vocab_size,
        raw_tensor_bytes=expected_raw_bytes,
    )
    if actual.file_sha256 != expected.file_sha256:
        raise ValueError(
            f"shard {prompt.prompt_id} file SHA-256 mismatch: expected {expected.file_sha256}, found {actual.file_sha256}"
        )
    if actual.raw_tensor_sha256 != expected.raw_tensor_sha256:
        raise ValueError(
            f"shard {prompt.prompt_id} raw tensor SHA-256 mismatch: expected {expected.raw_tensor_sha256}, found {actual.raw_tensor_sha256}"
        )
    return actual


def _validate_nested_path_fields(manifest: Mapping[str, Any]) -> None:
    for section, fields in (
        ("source_evidence", ("source_audit_path", "source_payload_audit_path")),
        ("non_vq_package", ("evidence_path",)),
        ("policy", ("path",)),
        ("prompt_authority", ("path",)),
    ):
        for field in fields:
            _validate_logical_path(manifest[section][field], label=f"{section}.{field}")


def _validate_manifest_mapping(
    value: object, *, contract: GLM52TeacherCacheContract
) -> tuple[dict[str, Any], tuple[GLM52ProducerPhase, ...], tuple[GLM52TeacherCacheShard, ...]]:
    manifest = dict(_require_exact_keys(value, _TOP_LEVEL_FIELDS, label="manifest"))
    _require_exact_value(manifest["schema_version"], SCHEMA_VERSION, label="manifest schema_version")
    _require_exact_value(manifest["record_type"], RECORD_TYPE, label="manifest record_type")
    if not isinstance(manifest["created_at"], str) or not manifest["created_at"]:
        raise ValueError("manifest created_at must be a non-empty string")

    section_fields = (
        ("producer", _PRODUCER_FIELDS),
        ("source", _SOURCE_FIELDS),
        ("source_evidence", _SOURCE_EVIDENCE_FIELDS),
        ("non_vq_package", _NON_VQ_FIELDS),
        ("policy", _POLICY_FIELDS),
        ("prompt_authority", _PROMPT_AUTHORITY_FIELDS),
        ("tokenizer", _TOKENIZER_FIELDS),
        ("precision", _PRECISION_FIELDS),
        ("architecture", _ARCHITECTURE_FIELDS),
        ("totals", _TOTAL_FIELDS),
    )
    for section, fields in section_fields:
        _require_exact_keys(manifest[section], fields, label=section)
    tokenizer_files = manifest["tokenizer"]["files"]
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise ValueError("tokenizer.files must be a non-empty object")
    for filename, identity in tokenizer_files.items():
        _validate_logical_path(filename, label="tokenizer filename")
        file_record = _require_exact_keys(identity, _TOKENIZER_FILE_FIELDS, label=f"tokenizer file {filename}")
        if type(file_record["size_bytes"]) is not int or file_record["size_bytes"] < 0:
            raise ValueError(f"tokenizer file {filename} size_bytes must be non-negative integer")
        _require_sha256(file_record["sha256"], label=f"tokenizer file {filename} sha256")
    splits = manifest["totals"]["splits"]
    expected_splits = contract.totals()["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(expected_splits):
        raise ValueError("totals split inventory does not match frozen prompts")
    for split, record in splits.items():
        _require_exact_keys(record, _SPLIT_FIELDS, label=f"totals.splits.{split}")

    expected_sections = {
        "producer": contract.producer,
        "source": contract.source,
        "source_evidence": contract.source_evidence,
        "non_vq_package": contract.non_vq_package,
        "policy": contract.policy,
        "prompt_authority": contract.prompt_authority,
        "tokenizer": contract.tokenizer,
        "precision": contract.precision,
        "architecture": contract.architecture,
        "totals": contract.totals(),
    }
    for section, expected in expected_sections.items():
        _require_exact_value(manifest[section], expected, label=section)
    _validate_nested_path_fields(manifest)
    for section in ("source_evidence", "non_vq_package", "policy", "prompt_authority"):
        for field, field_value in manifest[section].items():
            if field.endswith("sha256"):
                _require_sha256(field_value, label=f"{section}.{field}")
    for field in (
        "activation_quantization_emulated",
        "exact_w4a4_runtime_parity_claimed",
        "bf16_teacher_claimed",
    ):
        _require_exact_value(manifest["policy"][field], False, label=f"policy.{field}")
    _require_exact_value(manifest["policy"]["holdout_tuning_forbidden"], True, label="policy.holdout_tuning_forbidden")
    _require_exact_value(manifest["policy"]["full_vocabulary_logits_required"], True, label="policy.full_vocabulary_logits_required")

    raw_phases = manifest["producer_phases"]
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("producer_phases must be a non-empty list")
    phases = tuple(
        GLM52ProducerPhase.from_mapping(record, index=index)
        for index, record in enumerate(raw_phases)
    )
    if [phase.ordinal for phase in phases] != list(range(len(phases))):
        raise ValueError("producer phase ordinals must be contiguous and ordered from zero")
    phase_ids = [phase.phase_id for phase in phases]
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError("producer phase IDs must be unique")

    raw_shards = manifest["shards"]
    if not isinstance(raw_shards, list) or len(raw_shards) != len(contract.prompts):
        raise ValueError("ordered shard inventory length does not match frozen prompt count")
    shards = tuple(
        GLM52TeacherCacheShard.from_mapping(record, index=index)
        for index, record in enumerate(raw_shards)
    )
    for index, (shard, prompt) in enumerate(zip(shards, contract.prompts, strict=True)):
        _validate_shard_record(
            shard,
            prompt=prompt,
            contract=contract,
            index=index,
            phase_ids=set(phase_ids),
        )

    known = all(phase.memory_counters_known for phase in phases)
    clean = known and all(phase.memory_clean for phase in phases)
    wired_default = all(phase.system_wired_default for phase in phases)
    expected_booleans = {
        "payload_integrity_pass": True,
        "all_producer_memory_counters_known": known,
        "all_producer_memory_clean": clean,
        "system_wired_default": wired_default,
        "release_eligible": known and clean and wired_default,
    }
    for field, expected in expected_booleans.items():
        _require_exact_value(manifest[field], expected, label=field)
    _require_sha256(manifest["cache_content_sha256"], label="cache_content_sha256")
    _require_sha256(manifest["manifest_body_sha256"], label="manifest_body_sha256")
    expected_content = compute_cache_content_sha256(manifest)
    _require_exact_value(manifest["cache_content_sha256"], expected_content, label="cache content SHA-256")
    expected_body = compute_manifest_body_sha256(manifest)
    _require_exact_value(manifest["manifest_body_sha256"], expected_body, label="manifest body SHA-256")
    return manifest, phases, shards


def _validate_exact_tree(root: Path, contract: GLM52TeacherCacheContract) -> None:
    root_entries = {entry.name: entry for entry in os.scandir(root)}
    expected_root = {MANIFEST_FILENAME, SHARD_DIRECTORY}
    extra_root = sorted(set(root_entries) - expected_root)
    missing_root = sorted(expected_root - set(root_entries))
    if extra_root:
        raise ValueError(f"cache root contains extra entries: {extra_root}")
    if missing_root:
        raise ValueError(f"cache root is missing entries: {missing_root}")
    manifest_entry = root_entries[MANIFEST_FILENAME]
    if manifest_entry.is_symlink():
        raise ValueError("manifest file is a symlink")
    if not manifest_entry.is_file(follow_symlinks=False):
        raise ValueError("manifest file is special, not regular")
    logits_entry = root_entries[SHARD_DIRECTORY]
    if logits_entry.is_symlink():
        raise ValueError("teacher_logits directory is a symlink")
    if not logits_entry.is_dir(follow_symlinks=False):
        raise ValueError("teacher_logits is special, not a directory")

    logits_dir = root / SHARD_DIRECTORY
    entries = {entry.name: entry for entry in os.scandir(logits_dir)}
    expected = {f"{prompt.prompt_id}.safetensors" for prompt in contract.prompts}
    extra = sorted(set(entries) - expected)
    missing = sorted(expected - set(entries))
    if extra:
        special = [name for name in extra if not entries[name].is_file(follow_symlinks=False)]
        if special:
            raise ValueError(f"teacher_logits contains special extra files: {special}")
        raise ValueError(f"teacher_logits contains extra files: {extra}")
    if missing:
        raise ValueError(f"teacher_logits is missing files: {missing}")
    for name, entry in entries.items():
        if entry.is_symlink():
            raise ValueError(f"shard file is a symlink: {name}")
        if not entry.is_file(follow_symlinks=False):
            raise ValueError(f"shard file is special, not regular: {name}")


@dataclass(frozen=True)
class GLM52TeacherCacheAudit:
    valid: bool
    release_eligible: bool
    cache_content_sha256: str
    manifest_body_sha256: str
    prompt_count: int
    source_token_count: int
    predictor_position_count: int
    fp32_value_count: int
    raw_tensor_bytes: int
    split_position_counts: Mapping[str, int]
    split_raw_tensor_bytes: Mapping[str, int]


def audit_glm52_teacher_cache(
    cache_root: str | Path,
    *,
    contract: GLM52TeacherCacheContract,
) -> GLM52TeacherCacheAudit:
    """Strictly audit a raw cache root without trusting its manifest claims."""

    root = Path(cache_root)
    _reject_symlink_path_components(root, label="cache root")
    try:
        root_stat = root.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"cache root is missing: {root}") from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError("cache root is a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("cache root is not a directory")
    root_before = _identity_from_stat(root_stat)
    _validate_exact_tree(root, contract)
    logits_dir = root / SHARD_DIRECTORY
    logits_before = _identity_from_stat(logits_dir.lstat())

    manifest_path = root / MANIFEST_FILENAME
    manifest_raw, manifest_identity = _read_regular_file_stable(manifest_path)
    manifest_value = _strict_json_loads(manifest_raw, label="teacher cache manifest")
    manifest, _phases, shards = _validate_manifest_mapping(manifest_value, contract=contract)

    position_count = 0
    element_count = 0
    raw_byte_count = 0
    split_positions: Counter[str] = Counter()
    split_bytes: Counter[str] = Counter()
    for prompt, shard in zip(contract.prompts, shards, strict=True):
        shard_path = root / Path(*PurePosixPath(shard.relative_path).parts)
        audited = _audit_shard(
            shard_path,
            prompt=prompt,
            vocab_size=contract.vocab_size,
            expected=shard,
        )
        positions = prompt.token_count - 1
        position_count += positions
        element_count += audited.element_count
        raw_byte_count += audited.raw_tensor_bytes
        split_positions[prompt.split] += positions
        split_bytes[prompt.split] += audited.raw_tensor_bytes

    expected_totals = contract.totals()
    reconciled = {
        "prompt_count": len(shards),
        "source_token_count": sum(shard.token_count for shard in shards),
        "predictor_position_count": position_count,
        "vocab_size": contract.vocab_size,
        "fp32_value_count": element_count,
        "raw_tensor_bytes": raw_byte_count,
        "splits": {
            split: {
                "position_count": split_positions[split],
                "raw_tensor_bytes": split_bytes[split],
            }
            for split in expected_totals["splits"]
        },
    }
    _require_exact_value(reconciled, expected_totals, label="recomputed row/split/global totals")

    final_manifest_raw, final_manifest_identity = _read_regular_file_stable(manifest_path)
    if final_manifest_identity != manifest_identity or final_manifest_raw != manifest_raw:
        raise ValueError("manifest changed during audit")
    final_manifest = _strict_json_loads(final_manifest_raw, label="teacher cache manifest recheck")
    _validate_manifest_mapping(final_manifest, contract=contract)
    _validate_exact_tree(root, contract)
    logits_after = _identity_from_stat(logits_dir.lstat())
    if logits_before != logits_after:
        raise ValueError("teacher_logits directory changed during audit")
    root_after = _identity_from_stat(root.lstat())
    if root_before != root_after:
        raise ValueError("cache root changed during audit")

    return GLM52TeacherCacheAudit(
        valid=True,
        release_eligible=bool(manifest["release_eligible"]),
        cache_content_sha256=str(manifest["cache_content_sha256"]),
        manifest_body_sha256=str(manifest["manifest_body_sha256"]),
        prompt_count=len(shards),
        source_token_count=contract.source_token_count,
        predictor_position_count=position_count,
        fp32_value_count=element_count,
        raw_tensor_bytes=raw_byte_count,
        split_position_counts=dict(split_positions),
        split_raw_tensor_bytes=dict(split_bytes),
    )


# Concise aliases for operation registration and callers that already carry a
# GLM52-specific context.
write_teacher_cache_shard = write_glm52_teacher_cache_shard
publish_teacher_cache_manifest = publish_glm52_teacher_cache_manifest
audit_teacher_cache = audit_glm52_teacher_cache


__all__ = [
    "GLM52ProducerPhase",
    "GLM52TeacherCacheAudit",
    "GLM52TeacherCacheContract",
    "GLM52TeacherCacheManifest",
    "GLM52TeacherCachePrompt",
    "GLM52TeacherCacheShard",
    "GLM52TeacherCacheShardWrite",
    "audit_glm52_teacher_cache",
    "audit_teacher_cache",
    "build_glm52_teacher_cache_manifest",
    "canonical_sha256",
    "compute_cache_content_sha256",
    "compute_manifest_body_sha256",
    "publish_glm52_teacher_cache_manifest",
    "publish_teacher_cache_manifest",
    "write_glm52_teacher_cache_shard",
    "write_teacher_cache_shard",
]
