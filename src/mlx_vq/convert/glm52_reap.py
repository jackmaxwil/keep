"""Pinned GLM-5.2 REAP source and local payload contract audits."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from mlx_vq.convert.nvfp4 import (
    ModelOptNvfp4Spec,
    parse_modelopt_nvfp4_weight_spec,
    resolve_modelopt_nvfp4_weight_bundle,
)
from mlx_vq.convert.stream_convert import (
    SafetensorsIndex,
    StreamingConversionPlan,
    plan_streaming_conversion_from_index,
)
from mlx_vq.io.source_safetensors import (
    SafetensorsFileHeader,
    SafetensorsTensorHeader,
    read_safetensors_file_header,
)
from mlx_vq.models.glm52_policy import validate_glm52_config
from mlx_vq.models.profiles import ModelProfile, get_profile


GLM52_REAP_PROFILE_NAME = "glm52-reap-504b-v2"
GLM52_REAP_EXPECTED_MAIN_SHARDS = 61
GLM52_REAP_EXPECTED_CROSS_SHARD_BUNDLES = 53
GLM52_REAP_EXPECTED_GROUPS = 225
GLM52_REAP_EXPECTED_EXPERTS = 168
GLM52_REAP_EXPECTED_BUNDLES = 37_800
GLM52_REAP_EXPECTED_LAYER_IDS = tuple(range(3, 78))
GLM52_REAP_CONFIG_SHA256 = "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
GLM52_REAP_INDEX_SHA256 = "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
GLM52_REAP_SOURCE_DECODER = "modelopt_nvfp4_v1"
GLM52_REAP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MAX_AUDIT_ERROR_EXAMPLES = 100
MAX_GROUP_ERROR_EXAMPLES = 10

_ANY_EXPERT_MEMBER_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>[^.]+)\.(?P<member>[^.]+)$"
)
_ANY_MODEL_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _index_json(index: SafetensorsIndex) -> dict[str, object]:
    return {"metadata": index.metadata, "weight_map": index.weight_map}


def _projection_dims(profile: ModelProfile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    if projection == "down_proj":
        return profile.moe_intermediate_size, profile.hidden_size
    raise ValueError(f"unknown GLM52 projection {projection!r}")


def _profile_source_contract(profile: ModelProfile) -> tuple[object, ...]:
    return (
        profile.hf_model_id,
        profile.revision,
        profile.architecture,
        profile.num_layers,
        profile.num_sparse_layers,
        profile.first_sparse_layer,
        profile.hidden_size,
        profile.moe_intermediate_size,
        profile.num_experts,
        profile.experts_per_tok,
        profile.vocab_size,
        profile.shared_experts,
        profile.converter,
        profile.fused_gate_up,
    )


def _bundle_names(layer: int, expert: int, projection: str) -> tuple[str, str, str]:
    weight = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
    return weight, f"{weight}_scale", f"{weight}_scale_2"


def _valid_indexed_shard(weight_map: Mapping[str, str], tensor_name: str) -> str | None:
    shard = weight_map.get(tensor_name)
    return shard if isinstance(shard, str) and shard else None


def _is_safe_relative_shard_name(shard_name: str) -> bool:
    path = Path(shard_name)
    return bool(shard_name) and not path.is_absolute() and ".." not in path.parts


@dataclass(frozen=True)
class Glm52Nvfp4BundleContract:
    layer: int
    expert: int
    projection: str
    input_dims: int
    output_dims: int
    weight_name: str
    block_scale_name: str
    global_scale_name: str
    weight_shard: str | None
    block_scale_shard: str | None
    global_scale_shard: str | None

    @property
    def resolved(self) -> bool:
        return all(
            shard is not None
            for shard in (self.weight_shard, self.block_scale_shard, self.global_scale_shard)
        )

    @property
    def missing_members(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, shard in (
                (self.weight_name, self.weight_shard),
                (self.block_scale_name, self.block_scale_shard),
                (self.global_scale_name, self.global_scale_shard),
            )
            if shard is None
        )

    @property
    def source_shards(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    shard
                    for shard in (
                        self.weight_shard,
                        self.block_scale_shard,
                        self.global_scale_shard,
                    )
                    if shard is not None
                }
            )
        )

    @property
    def cross_shard(self) -> bool:
        return self.resolved and len(self.source_shards) > 1

    def members(
        self,
    ) -> tuple[tuple[str, str, str | None, str, tuple[int, ...], int], ...]:
        weight_shape = (self.output_dims, self.input_dims // 2)
        block_scale_shape = (self.output_dims, self.input_dims // 16)
        return (
            (
                "weight",
                self.weight_name,
                self.weight_shard,
                "U8",
                weight_shape,
                weight_shape[0] * weight_shape[1],
            ),
            (
                "block_scale",
                self.block_scale_name,
                self.block_scale_shard,
                "F8_E4M3",
                block_scale_shape,
                block_scale_shape[0] * block_scale_shape[1],
            ),
            (
                "global_scale",
                self.global_scale_name,
                self.global_scale_shard,
                "F32",
                (),
                4,
            ),
        )


@dataclass(frozen=True)
class Glm52ReapSourceGroupAudit:
    layer: int
    projection: str
    input_dims: int
    output_dims: int
    bundles: tuple[Glm52Nvfp4BundleContract, ...]

    def to_json_dict(self) -> dict[str, object]:
        missing = tuple(name for bundle in self.bundles for name in bundle.missing_members)
        return {
            "layer": self.layer,
            "projection": self.projection,
            "experts": len(self.bundles),
            "expected_bundles": len(self.bundles),
            "resolved_bundles": sum(bundle.resolved for bundle in self.bundles),
            "required_bundle_members": len(self.bundles) * 3,
            "missing_companion_tensor_count": len(missing),
            "missing_companion_tensors": list(missing[:MAX_GROUP_ERROR_EXAMPLES]),
            "required_shards": sorted(
                {shard for bundle in self.bundles for shard in bundle.source_shards}
            ),
            "cross_shard_bundles": sum(bundle.cross_shard for bundle in self.bundles),
            "decoded_shape_per_expert": [self.output_dims, self.input_dims],
            "packed_weight_shape_per_expert": [self.output_dims, self.input_dims // 2],
            "block_scale_shape_per_expert": [self.output_dims, self.input_dims // 16],
            "global_scale_shape_per_expert": [],
        }


@dataclass(frozen=True)
class Glm52ReapSourceAudit:
    profile: ModelProfile
    model_id: str
    revision: str
    config_sha256: str
    index_sha256: str
    config_summary: Mapping[str, object]
    nvfp4_spec: ModelOptNvfp4Spec | None
    source_encoding_error: str | None
    source_index_tensors: int
    source_index_shards: tuple[str, ...]
    main_layer_ids: tuple[int, ...]
    excluded_mtp_layer_ids: tuple[int, ...]
    excluded_mtp_tensors: int
    main_activation_scale_tensors: int
    groups: tuple[Glm52ReapSourceGroupAudit, ...]
    plan: StreamingConversionPlan | None
    missing_vq_groups: tuple[str, ...]
    unexpected_routed_tensors: tuple[str, ...]
    invalid_source_shards: tuple[str, ...]
    profile_failures: tuple[str, ...]
    checks: Mapping[str, bool]

    @property
    def bundles(self) -> tuple[Glm52Nvfp4BundleContract, ...]:
        return tuple(bundle for group in self.groups for bundle in group.bundles)

    @property
    def missing_companion_tensors(self) -> tuple[str, ...]:
        return tuple(name for bundle in self.bundles for name in bundle.missing_members)

    @property
    def required_main_shards(self) -> tuple[str, ...]:
        return tuple(sorted({shard for bundle in self.bundles for shard in bundle.source_shards}))

    @property
    def cross_shard_bundles(self) -> int:
        return sum(bundle.cross_shard for bundle in self.bundles)

    @property
    def source_checks_pass(self) -> bool:
        return all(self.checks.values())

    def to_json_dict(self) -> dict[str, object]:
        planned_groups = 0 if self.plan is None else len(self.plan.vq_groups)
        planned_bundles = (
            0
            if self.plan is None
            else sum(len(group.experts) for group in self.plan.vq_groups)
        )
        resolved_bundles = sum(bundle.resolved for bundle in self.bundles)
        missing_companions = self.missing_companion_tensors
        failed_checks = [name for name, passed in self.checks.items() if not passed]
        return {
            "schema_version": 1,
            "record_type": "glm52_modelopt_nvfp4_source_audit",
            "profile": self.profile.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "config_summary": dict(self.config_summary),
            "nvfp4_spec": None if self.nvfp4_spec is None else asdict(self.nvfp4_spec),
            "source_weight_encoding": (
                "modelopt_nvfp4" if self.nvfp4_spec is not None else "unsupported"
            ),
            "source_decoder": (
                GLM52_REAP_SOURCE_DECODER if self.nvfp4_spec is not None else None
            ),
            "source_encoding_error": self.source_encoding_error,
            "source_index_tensors": self.source_index_tensors,
            "source_index_shards": len(self.source_index_shards),
            "source_index_shard_names": list(self.source_index_shards),
            "main_layer_ids": list(self.main_layer_ids),
            "excluded_mtp_layer_ids": list(self.excluded_mtp_layer_ids),
            "excluded_mtp_tensors": self.excluded_mtp_tensors,
            "main_activation_scale_tensors": self.main_activation_scale_tensors,
            "expected_groups": len(self.groups),
            "planned_groups": planned_groups,
            "expected_bundles": len(self.bundles),
            "planned_bundles": planned_bundles,
            "resolved_bundles": resolved_bundles,
            "required_bundle_members": len(self.bundles) * 3,
            "resolved_bundle_members": resolved_bundles * 3,
            "required_main_shards": len(self.required_main_shards),
            "required_main_shard_names": list(self.required_main_shards),
            "cross_shard_bundles": self.cross_shard_bundles,
            "missing_vq_groups": list(self.missing_vq_groups),
            "missing_companion_tensor_count": len(missing_companions),
            "missing_companion_tensors": list(
                missing_companions[:MAX_AUDIT_ERROR_EXAMPLES]
            ),
            "unexpected_routed_tensor_count": len(self.unexpected_routed_tensors),
            "unexpected_routed_tensors": list(
                self.unexpected_routed_tensors[:MAX_AUDIT_ERROR_EXAMPLES]
            ),
            "invalid_source_shards": list(self.invalid_source_shards),
            "profile_failures": list(self.profile_failures),
            "checks": dict(self.checks),
            "source_checks_pass": self.source_checks_pass,
            "audit_status": (
                "glm52_modelopt_nvfp4_source_audited"
                if self.source_checks_pass
                else "glm52_modelopt_nvfp4_source_incompatible"
            ),
            "audit_blockers": failed_checks,
            "groups": [group.to_json_dict() for group in self.groups],
        }


def _build_bundle_contract(
    *,
    index: SafetensorsIndex,
    profile: ModelProfile,
    layer: int,
    expert: int,
    projection: str,
) -> Glm52Nvfp4BundleContract:
    input_dims, output_dims = _projection_dims(profile, projection)
    weight_name, block_scale_name, global_scale_name = _bundle_names(
        layer, expert, projection
    )
    try:
        resolved = resolve_modelopt_nvfp4_weight_bundle(index.weight_map, weight_name)
    except (KeyError, ValueError):
        resolved = None
    return Glm52Nvfp4BundleContract(
        layer=layer,
        expert=expert,
        projection=projection,
        input_dims=input_dims,
        output_dims=output_dims,
        weight_name=weight_name,
        block_scale_name=block_scale_name,
        global_scale_name=global_scale_name,
        weight_shard=(
            resolved.weight_shard
            if resolved is not None
            else _valid_indexed_shard(index.weight_map, weight_name)
        ),
        block_scale_shard=(
            resolved.block_scale_shard
            if resolved is not None
            else _valid_indexed_shard(index.weight_map, block_scale_name)
        ),
        global_scale_shard=(
            resolved.global_scale_shard
            if resolved is not None
            else _valid_indexed_shard(index.weight_map, global_scale_name)
        ),
    )


def audit_glm52_reap_source_index(
    *,
    config: Mapping[str, Any],
    index: SafetensorsIndex,
    model_id: str,
    revision: str,
    profile: ModelProfile | None = None,
    config_sha256: str | None = None,
    index_sha256: str | None = None,
    enforce_pinned_profile: bool = True,
) -> Glm52ReapSourceAudit:
    """Audit the complete pinned main-model NVFP4 index without reading payloads."""

    canonical_profile = get_profile(GLM52_REAP_PROFILE_NAME)
    selected_profile = profile or canonical_profile
    contract_profile = canonical_profile if enforce_pinned_profile else selected_profile
    if enforce_pinned_profile and (config_sha256 is None or index_sha256 is None):
        raise ValueError(
            "pinned GLM52 REAP audit requires raw config_sha256 and index_sha256"
        )
    resolved_config_sha256 = config_sha256 or _canonical_json_sha256(config)
    resolved_index_sha256 = index_sha256 or _canonical_json_sha256(_index_json(index))
    profile_failures = tuple(
        validate_glm52_config(
            config,
            model_id=model_id,
            revision=revision,
            profile=contract_profile,
        )
    )
    try:
        nvfp4_spec = parse_modelopt_nvfp4_weight_spec(config)
        source_encoding_error = None
    except ValueError as error:
        nvfp4_spec = None
        source_encoding_error = str(error)

    plan: StreamingConversionPlan | None = None
    plan_error: str | None = None
    if nvfp4_spec is not None:
        try:
            plan = plan_streaming_conversion_from_index(
                dict(config),
                index,
                model_id=model_id,
                revision=revision,
                config_sha256=resolved_config_sha256,
                index_sha256=resolved_index_sha256,
                source_profile=contract_profile.name,
                group_size=512,
                code_bits=contract_profile.default_code_bits,
            )
        except ValueError as error:
            plan_error = str(error)

    first_sparse_layer = contract_profile.first_sparse_layer
    if first_sparse_layer is None:
        raise ValueError("GLM52 REAP profile must declare first_sparse_layer")
    main_layer_ids = tuple(range(first_sparse_layer, contract_profile.num_layers))
    groups = tuple(
        Glm52ReapSourceGroupAudit(
            layer=layer,
            projection=projection,
            input_dims=_projection_dims(contract_profile, projection)[0],
            output_dims=_projection_dims(contract_profile, projection)[1],
            bundles=tuple(
                _build_bundle_contract(
                    index=index,
                    profile=contract_profile,
                    layer=layer,
                    expert=expert,
                    projection=projection,
                )
                for expert in range(contract_profile.num_experts)
            ),
        )
        for layer in main_layer_ids
        for projection in GLM52_REAP_PROJECTIONS
    )

    excluded_mtp_layer_ids: set[int] = set()
    excluded_mtp_tensors = 0
    unexpected_routed_tensors: list[str] = []
    expected_routed_members = {
        name
        for group in groups
        for bundle in group.bundles
        for _, name, *_ in bundle.members()
    }
    expected_routed_members.update(
        f"{bundle.weight_name[:-len('.weight')]}.input_scale"
        for group in groups
        for bundle in group.bundles
    )
    for tensor_name in index.weight_map:
        layer_match = _ANY_MODEL_LAYER_RE.match(tensor_name)
        if layer_match is not None and int(layer_match.group("layer")) == contract_profile.num_layers:
            excluded_mtp_layer_ids.add(contract_profile.num_layers)
            excluded_mtp_tensors += 1
        expert_match = _ANY_EXPERT_MEMBER_RE.match(tensor_name)
        if expert_match is None:
            continue
        layer = int(expert_match.group("layer"))
        if layer == contract_profile.num_layers:
            continue
        if tensor_name not in expected_routed_members:
            unexpected_routed_tensors.append(tensor_name)

    invalid_source_shards = tuple(
        sorted(
            {
                shard
                for shard in index.weight_map.values()
                if not _is_safe_relative_shard_name(shard)
            }
        )
    )

    bundles = tuple(bundle for group in groups for bundle in group.bundles)
    missing_companions = tuple(name for bundle in bundles for name in bundle.missing_members)
    required_main_shards = tuple(
        sorted({shard for bundle in bundles for shard in bundle.source_shards})
    )
    cross_shard_bundles = sum(bundle.cross_shard for bundle in bundles)
    expected_group_contract = tuple(
        (group.layer, group.projection, tuple(range(contract_profile.num_experts)))
        for group in groups
    )
    planned_group_contract = (
        ()
        if plan is None
        else tuple((group.layer, group.projection, group.experts) for group in plan.vq_groups)
    )
    missing_vq_groups = (
        ((plan_error,) if plan_error is not None else ())
        if plan is None
        else plan.missing_vq_groups
    )
    checks: dict[str, bool] = {
        "profile_config": not profile_failures,
        "pinned_model_id": model_id == contract_profile.hf_model_id,
        "pinned_revision": revision == contract_profile.revision,
        "profile_converter": contract_profile.converter == "glm52_vq_groups",
        "native_modelopt_nvfp4": nvfp4_spec is not None,
        "complete_group_plan": (
            plan is not None
            and not plan.missing_vq_groups
            and planned_group_contract == expected_group_contract
        ),
        "complete_bundle_index": not missing_companions,
        "no_unexpected_routed_tensors": not unexpected_routed_tensors,
        "relative_source_shard_names": not invalid_source_shards,
        "mtp_excluded_from_plan": (
            plan is not None
            and all(group.layer < contract_profile.num_layers for group in plan.vq_groups)
        ),
    }
    if enforce_pinned_profile:
        checks["pinned_profile_definition"] = (
            selected_profile.name == GLM52_REAP_PROFILE_NAME
            and
            _profile_source_contract(selected_profile)
            == _profile_source_contract(canonical_profile)
        )
        checks["pinned_layer_ids"] = main_layer_ids == GLM52_REAP_EXPECTED_LAYER_IDS
        checks["pinned_group_count"] = len(groups) == GLM52_REAP_EXPECTED_GROUPS
        checks["pinned_expert_count"] = (
            contract_profile.num_experts == GLM52_REAP_EXPECTED_EXPERTS
            and all(len(group.bundles) == GLM52_REAP_EXPECTED_EXPERTS for group in groups)
        )
        checks["pinned_bundle_count"] = len(bundles) == GLM52_REAP_EXPECTED_BUNDLES
        checks["pinned_config_sha256"] = resolved_config_sha256 == GLM52_REAP_CONFIG_SHA256
        checks["pinned_index_sha256"] = resolved_index_sha256 == GLM52_REAP_INDEX_SHA256
        checks["pinned_required_main_shards"] = (
            len(required_main_shards) == GLM52_REAP_EXPECTED_MAIN_SHARDS
        )
        checks["pinned_cross_shard_bundles"] = (
            cross_shard_bundles == GLM52_REAP_EXPECTED_CROSS_SHARD_BUNDLES
        )

    summary = {} if plan is None else asdict(plan.summary)
    activation_scales = sum(
        f"{bundle.weight_name[:-len('.weight')]}.input_scale" in index.weight_map
        for bundle in bundles
    )
    return Glm52ReapSourceAudit(
        profile=contract_profile,
        model_id=model_id,
        revision=revision,
        config_sha256=resolved_config_sha256,
        index_sha256=resolved_index_sha256,
        config_summary=summary,
        nvfp4_spec=nvfp4_spec,
        source_encoding_error=source_encoding_error,
        source_index_tensors=len(index.weight_map),
        source_index_shards=index.shards,
        main_layer_ids=main_layer_ids,
        excluded_mtp_layer_ids=tuple(sorted(excluded_mtp_layer_ids)),
        excluded_mtp_tensors=excluded_mtp_tensors,
        main_activation_scale_tensors=activation_scales,
        groups=groups,
        plan=plan,
        missing_vq_groups=tuple(missing_vq_groups),
        unexpected_routed_tensors=tuple(sorted(unexpected_routed_tensors)),
        invalid_source_shards=invalid_source_shards,
        profile_failures=profile_failures,
        checks=checks,
    )


def _validate_tensor_header(
    *,
    file_header: SafetensorsFileHeader,
    file_size: int,
    tensor_name: str,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    expected_bytes: int,
) -> tuple[SafetensorsTensorHeader | None, tuple[str, ...]]:
    header = file_header.tensors.get(tensor_name)
    if header is None:
        return None, (f"{tensor_name}: missing tensor header",)
    errors: list[str] = []
    if header.dtype != expected_dtype:
        errors.append(
            f"{tensor_name}: expected dtype {expected_dtype}, found {header.dtype}"
        )
    if header.shape != expected_shape:
        errors.append(
            f"{tensor_name}: expected shape {list(expected_shape)}, found {list(header.shape)}"
        )
    start, end = header.data_offsets
    if start < 0 or end < start:
        errors.append(f"{tensor_name}: invalid data offsets {list(header.data_offsets)}")
    elif end - start != expected_bytes:
        errors.append(
            f"{tensor_name}: expected {expected_bytes} payload bytes, found {end - start}"
        )
    if file_header.payload_offset + end > file_size:
        errors.append(
            f"{tensor_name}: payload extent {file_header.payload_offset + end} exceeds file size {file_size}"
        )
    return header, tuple(errors)


def _validate_safetensors_file_layout(
    *,
    shard_name: str,
    file_header: SafetensorsFileHeader,
    file_size: int,
) -> tuple[int, tuple[str, ...]]:
    """Validate the shard-wide non-overlapping, contiguous safetensors payload layout."""

    errors: list[str] = []
    error_count = 0

    def record(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < MAX_AUDIT_ERROR_EXAMPLES:
            errors.append(message)
    intervals = sorted(
        (
            header.data_offsets[0],
            header.data_offsets[1],
            tensor_name,
        )
        for tensor_name, header in file_header.tensors.items()
    )
    cursor = 0
    for start, end, tensor_name in intervals:
        if start < 0 or end < start:
            record(
                f"{shard_name}: {tensor_name!r} has invalid data offsets {[start, end]}"
            )
            continue
        if start < cursor:
            record(
                f"{shard_name}: {tensor_name!r} has overlapping tensor extent "
                f"starting at {start} before {cursor}"
            )
        elif start > cursor:
            record(
                f"{shard_name}: gap before {tensor_name!r}: expected offset {cursor}, "
                f"found {start}"
            )
        cursor = max(cursor, end)
    declared_file_size = file_header.payload_offset + cursor
    if declared_file_size != file_size:
        record(
            f"{shard_name}: declared file extent {declared_file_size} does not match "
            f"physical file size {file_size}"
        )
    return error_count, tuple(errors)


def audit_glm52_reap_source_payloads(
    *,
    source_dir: str | Path,
    source_audit: Glm52ReapSourceAudit,
    max_groups: int | None = None,
    group_keys: Sequence[tuple[int, str]] | None = None,
) -> dict[str, object]:
    """Validate required local shard headers and extents without decoding weights."""

    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when provided")
    if max_groups is not None and group_keys is not None:
        raise ValueError("max_groups and group_keys are mutually exclusive")
    source_root = Path(source_dir)
    available_groups = {
        (group.layer, group.projection): group for group in source_audit.groups
    }
    requested_keys = tuple(group_keys or ())
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("group_keys contains duplicate layer/projection groups")
    unknown_keys = tuple(key for key in requested_keys if key not in available_groups)
    if unknown_keys:
        raise ValueError(f"group_keys contains unplanned groups: {unknown_keys}")
    requested_group_coverage = max_groups is None or max_groups <= len(source_audit.groups)
    if group_keys is not None:
        selected_key_set = set(requested_keys)
        selected_groups = tuple(
            group
            for group in source_audit.groups
            if (group.layer, group.projection) in selected_key_set
        )
    elif max_groups is not None:
        selected_groups = source_audit.groups[:max_groups]
    else:
        selected_groups = source_audit.groups
    selected_bundles = tuple(bundle for group in selected_groups for bundle in group.bundles)
    required_shards = tuple(
        sorted({shard for bundle in selected_bundles for shard in bundle.source_shards})
    )
    if source_audit.source_checks_pass:
        present_shards = tuple(
            shard for shard in required_shards if (source_root / shard).is_file()
        )
        missing_shards = tuple(
            shard for shard in required_shards if not (source_root / shard).is_file()
        )
    else:
        present_shards = ()
        missing_shards = ()

    members_by_shard: dict[
        str,
        list[
            tuple[
                tuple[int, str],
                tuple[str, str, str | None, str, tuple[int, ...], int],
            ]
        ],
    ] = {}
    for group in selected_groups:
        for bundle in group.bundles:
            for member in bundle.members():
                shard = member[2]
                if shard is not None:
                    members_by_shard.setdefault(shard, []).append(
                        ((group.layer, group.projection), member)
                    )

    valid_shards: set[str] = set()
    invalid_shards: list[str] = []
    invalid_shard_error_count = 0
    tensor_errors: dict[str, tuple[str, ...]] = {}
    tensor_error_count = 0
    invalid_groups: set[tuple[int, str]] = set()
    header_checked_tensors = 0
    for shard in present_shards:
        try:
            file_size = (source_root / shard).stat().st_size
            file_header = read_safetensors_file_header(source_root / shard)
            layout_error_count, layout_errors = _validate_safetensors_file_layout(
                shard_name=shard,
                file_header=file_header,
                file_size=file_size,
            )
            if layout_error_count:
                invalid_shard_error_count += layout_error_count
                invalid_shards.extend(
                    layout_errors[: max(0, MAX_AUDIT_ERROR_EXAMPLES - len(invalid_shards))]
                )
            else:
                valid_shards.add(shard)
                for group_key, member in members_by_shard.get(shard, ()):
                    _, tensor_name, _, dtype, shape, payload_bytes = member
                    header, errors = _validate_tensor_header(
                        file_header=file_header,
                        file_size=file_size,
                        tensor_name=tensor_name,
                        expected_dtype=dtype,
                        expected_shape=shape,
                        expected_bytes=payload_bytes,
                    )
                    if header is not None:
                        header_checked_tensors += 1
                    if errors:
                        tensor_error_count += len(errors)
                        invalid_groups.add(group_key)
                        if len(tensor_errors) < MAX_AUDIT_ERROR_EXAMPLES:
                            tensor_errors[tensor_name] = errors
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid_shard_error_count += 1
            if len(invalid_shards) < MAX_AUDIT_ERROR_EXAMPLES:
                invalid_shards.append(f"{shard}: {error}")

    invalid_tensor_headers = tuple(
        error for tensor_name in sorted(tensor_errors) for error in tensor_errors[tensor_name]
    )
    checks = {
        "source_audit_pass": source_audit.source_checks_pass,
        "requested_group_coverage": requested_group_coverage,
        "all_required_shards_present": source_audit.source_checks_pass and not missing_shards,
        "all_shard_headers_readable": source_audit.source_checks_pass and not invalid_shards,
        "all_bundle_headers_valid": source_audit.source_checks_pass and not invalid_tensor_headers,
    }
    blocked = not all(checks.values())
    full_group_coverage = len(selected_groups) == len(source_audit.groups)
    if not source_audit.source_checks_pass:
        payload_status = "glm52_modelopt_nvfp4_source_audit_failed"
    elif missing_shards:
        payload_status = "glm52_modelopt_nvfp4_source_payloads_missing"
    elif not requested_group_coverage or invalid_shards or invalid_tensor_headers:
        payload_status = "glm52_modelopt_nvfp4_source_payloads_invalid"
    elif not full_group_coverage:
        payload_status = "glm52_modelopt_nvfp4_source_payloads_bounded_ready"
    else:
        payload_status = "glm52_modelopt_nvfp4_source_payloads_ready"

    blockers: list[str] = []
    if not source_audit.source_checks_pass:
        blockers.append("repair_source_index_or_nvfp4_companions")
    if not requested_group_coverage:
        blockers.append("repair_source_audit_group_coverage")
    if missing_shards:
        blockers.append("download_payload_shards")
    if invalid_shards or invalid_tensor_headers:
        blockers.append("repair_source_payload_headers_or_files")
    if blocked:
        blockers.append("rerun_with_payload_bearing_source_dir")

    group_records: list[dict[str, object]] = []
    for group in selected_groups:
        shards = sorted({shard for bundle in group.bundles for shard in bundle.source_shards})
        group_missing = [shard for shard in shards if shard in missing_shards]
        names = {name for bundle in group.bundles for _, name, *_ in bundle.members()}
        group_invalid = [
            error
            for name in sorted(names)
            for error in tensor_errors.get(name, ())
        ][:MAX_GROUP_ERROR_EXAMPLES]
        group_records.append(
            {
                "layer": group.layer,
                "projection": group.projection,
                "bundles": len(group.bundles),
                "bundle_members": len(group.bundles) * 3,
                "required_shards": shards,
                "missing_shards": group_missing,
                "invalid_tensor_headers": group_invalid,
                "payload_ready": (
                    source_audit.source_checks_pass
                    and not group_missing
                    and (group.layer, group.projection) not in invalid_groups
                    and all(shard in valid_shards for shard in shards)
                ),
            }
        )

    return {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_source_payload_audit",
        "profile": source_audit.profile.name,
        "model_id": source_audit.model_id,
        "revision": source_audit.revision,
        "config_sha256": source_audit.config_sha256,
        "index_sha256": source_audit.index_sha256,
        "source_dir": str(source_root),
        "payload_scope": "full" if full_group_coverage else "bounded",
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": GLM52_REAP_SOURCE_DECODER,
        "total_groups": len(source_audit.groups),
        "selected_groups": len(selected_groups),
        "requested_max_groups": max_groups,
        "requested_groups": [f"{layer}:{projection}" for layer, projection in requested_keys],
        "selected_bundles": len(selected_bundles),
        "bundle_members": len(selected_bundles) * 3,
        "cross_shard_bundles": sum(bundle.cross_shard for bundle in selected_bundles),
        "required_shards": list(required_shards),
        "present_shards": list(present_shards),
        "missing_shards": list(missing_shards),
        "invalid_shards": invalid_shards,
        "invalid_shard_error_count": invalid_shard_error_count,
        "header_checked_tensors": header_checked_tensors,
        "invalid_tensor_headers": list(invalid_tensor_headers),
        "invalid_tensor_header_error_count": tensor_error_count,
        "checks": checks,
        "payload_status": payload_status,
        "full_group_coverage": full_group_coverage,
        "full_payload_ready": not blocked and full_group_coverage,
        "materialization_blocked": blocked,
        "materialization_blockers": blockers,
        "groups": group_records,
    }
