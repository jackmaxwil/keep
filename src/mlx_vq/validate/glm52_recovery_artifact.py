"""Headless integrity audit for a GLM-5.2 recovery mixed artifact.

This module deliberately avoids importing :mod:`mlx` or validation modules
whose package initialization requires a Metal device.  It authenticates the
recovery manifest, pins the inherited seed manifest by caller-supplied hash,
and derives all artifact accounting from the resolved safetensor payloads.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

import numpy as np
from safetensors import safe_open

from mlx_vq.codebook.e8 import e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.convert.glm52_recovery_materialize import (
    GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES,
    PINNED_ACCEPTED_SEED_MANIFEST_SHA256,
    _authenticate_accepted_composite_audit,
    _authenticate_recovery_seed,
    _validate_seed_manifest,
)
from mlx_vq.io.schema import codebook_metadata_for_bits
from mlx_vq.io.authenticated_artifacts import (
    AuthenticatedFile,
    authenticate_json,
    clone_or_copy_authenticated,
    open_regular_no_follow,
)


GLM52_RECOVERY_MANIFEST = "conversion-manifest.json"
GLM52_RECOVERY_RECORD_TYPE = "glm52_recovery_conversion_manifest"
GLM52_CODEBOOK_TENSOR = "model.vq_codebook.e8"
GLM52_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GLM52_FIRST_ROUTED_LAYER = 3
GLM52_LAST_ROUTED_LAYER = 77
GLM52_GROUP_COUNT = 225
GLM52_CODEBOOK_BYTES_PER_GROUP = 256 * np.dtype(np.uint32).itemsize
GLM52_ACCEPTED_WHOLE_TENSOR_PAYLOAD_BYTES = 98_433_923_808
GLM52_RECOVERY_LEVER = "selection_diagonal_hessian_importance_weighted_reround_v1"
GLM52_RECOVERY_SCALE_ESTIMATOR = "importance_weighted_least_squares"
GLM52_RECOVERY_ROUNDING_OBJECTIVE = (
    "selection_diagonal_hessian_weighted_squared_error"
)
GLM52_RECOVERY_SEED_KIND = "audited_recovery"
_CLASSIC_MIXED_ARTIFACT_KEYS = {
    "seed_artifact_dir",
    "seed_manifest_sha256",
    "output_dir",
    "recovered_groups_dir",
    "replacement_count",
    "inherited_group_count",
}
_RECOVERY_SEEDED_MIXED_ARTIFACT_KEYS = {
    "seed_artifact_dir",
    "output_dir",
    "recovered_groups_dir",
    "replacement_count",
    "inherited_group_count",
    "seed_kind",
    "seed_recovery_manifest_sha256",
    "seed_recovery_audit_sha256",
    "seed_candidate_identity_sha256",
    "accepted_baseline_seed_manifest_sha256",
    "parent_artifact_root",
}


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True)
class _GroupSnapshot:
    copy_path: Path
    identity: _FileIdentity
    sha256: str


def _open_regular_no_follow(path: Path, *, label: str) -> int:
    return open_regular_no_follow(path, label=label)


@contextmanager
def _authenticated_group_snapshot(path: Path, *, label: str) -> Iterator[_GroupSnapshot]:
    authenticated = AuthenticatedFile.open(path, label=label)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix="glm52-recovery-audit-")
    ).resolve(strict=True)
    copy_path = temporary_dir / "authenticated-group.safetensors"
    try:
        identity = _FileIdentity(
            authenticated.device,
            authenticated.inode,
            authenticated.size,
            authenticated.identity.mtime_ns,
        )
        clone_or_copy_authenticated(
            authenticated,
            copy_path,
            allow_copy_fallback=True,
            required_free_bytes=authenticated.size,
        )
        yield _GroupSnapshot(copy_path, identity, authenticated.sha256)
        authenticated.verify_visible()
    finally:
        authenticated.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)


class _Profile(Protocol):
    name: str
    hf_model_id: str
    revision: str | None
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    group_size_policy: Mapping[str, int]


@dataclass(frozen=True)
class GLM52RecoveryGroupAudit:
    group_key: str
    filename: str
    classification: str
    artifact_sha256: str
    artifact_bytes: int
    tensor_payload_bytes: int
    code_bits: int
    codes_dtype: str
    group_size: int
    codebook_name: str
    codebook_sha256: str


@dataclass(frozen=True)
class _RecoveryVerificationAuthority:
    profile: _Profile
    expected_seed_manifest_sha256: str
    expected_stats_manifest_sha256: str
    expected_full_source_blob_inventory_sha256: str
    expected_routed_source_blob_inventory_sha256: str
    expected_recovery_lever: str
    expected_recovery_policy: Mapping[str, str]
    accepted_composite_audit_json: str
    expected_composite_audit_sha256: str
    seed_kind: str
    parent_recovery_dir: str | None = None
    parent_recovery_audit_json: str | None = None
    expected_parent_manifest_sha256: str | None = None
    expected_parent_audit_sha256: str | None = None
    expected_parent_candidate_identity_sha256: str | None = None


@dataclass(frozen=True)
class GLM52RecoveryArtifactAudit:
    recovery_dir: str
    manifest_path: str
    manifest_body_sha256: str
    seed_manifest_sha256: str
    accepted_composite_audit_sha256: str
    accepted_baseline_composite_identity_sha256: str
    group_count: int
    replacement_group_count: int
    inherited_group_count: int
    complete_replacement_layer_ids: tuple[int, ...]
    routed_codes_scales_bytes: int
    routed_codebook_bytes: int
    routed_tensor_payload_bytes: int
    main_non_routed_tensor_payload_bytes: int
    non_routed_tensor_payload_bytes: int
    logical_whole_model_tensor_payload_bytes: int
    logical_payload_limit_bytes: int
    incremental_disk_bytes: int
    candidate_identity_sha256: str
    groups: tuple[GLM52RecoveryGroupAudit, ...]
    checks: dict[str, bool]
    _verification_authority: _RecoveryVerificationAuthority = field(
        repr=False,
        compare=False,
    )

    @property
    def audit_pass(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("_verification_authority", None)
        payload["complete_replacement_layer_ids"] = list(
            self.complete_replacement_layer_ids
        )
        payload["audit_pass"] = self.audit_pass
        return payload

    def verify_current_identity(self) -> None:
        _verify_recovery_audit_current_identity(self)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _load_authenticated_object(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    authenticated, value = authenticate_json(path, label=label, loads=json.loads)
    try:
        return value, authenticated.sha256
    finally:
        authenticated.close()


def _expect(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}")


def _expect_int(label: str, actual: object, expected: int) -> None:
    if type(actual) is not int or actual != expected:
        raise ValueError(f"{label} must be {expected}, found {actual!r}")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _expect_exact_keys(label: str, value: Mapping[str, object], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must have exact fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_logical_payload_budget(logical_payload_bytes: int) -> None:
    if logical_payload_bytes > GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES:
        raise ValueError(
            f"logical whole-model tensor payload {logical_payload_bytes} exceeds "
            f"the canonical {GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES} byte budget limit"
        )


def _zero_importance_policy_for_bits(code_bits: int) -> str:
    if code_bits == 8:
        return "preserve_seed_expert_bytes"
    if code_bits == 16:
        return "source_rtn_e8p"
    raise ValueError(f"unsupported recovery code bits: {code_bits}")


def _validate_rate_policy(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("recovery manifest rate_policy must be an object")
    _expect_exact_keys(
        "recovery manifest rate_policy",
        raw,
        {
            "default_code_bits",
            "layer_code_bits",
            "complete_layer_rates_required",
        },
    )
    _expect_int(
        "recovery manifest default code bits",
        raw.get("default_code_bits"),
        8,
    )
    if raw.get("complete_layer_rates_required") is not True:
        raise ValueError(
            "recovery manifest complete_layer_rates_required must be true"
        )
    layer_code_bits = raw.get("layer_code_bits")
    if not isinstance(layer_code_bits, dict):
        raise ValueError("recovery manifest layer_code_bits must be an object")
    normalized: dict[str, int] = {}
    for layer_text, code_bits in layer_code_bits.items():
        if (
            not isinstance(layer_text, str)
            or not layer_text.isdecimal()
            or str(int(layer_text)) != layer_text
        ):
            raise ValueError("rate_policy layer_code_bits keys must be canonical strings")
        layer = int(layer_text)
        if layer < GLM52_FIRST_ROUTED_LAYER or layer > GLM52_LAST_ROUTED_LAYER:
            raise ValueError(f"rate_policy layer {layer} is outside routed layers")
        _expect_int(f"rate_policy layer {layer} code bits", code_bits, 16)
        normalized[layer_text] = 16
    return {
        "default_code_bits": 8,
        "layer_code_bits": {
            str(layer): normalized[str(layer)]
            for layer in sorted(int(value) for value in normalized)
        },
        "complete_layer_rates_required": True,
    }


def _expected_provenance(
    *,
    stats_manifest_sha256: str,
    recovery_lever: str,
    source_lineage: Mapping[str, str],
    source_verification: Mapping[str, object],
) -> dict[str, object]:
    return {
        "lever": recovery_lever,
        "scale_estimator": GLM52_RECOVERY_SCALE_ESTIMATOR,
        "rounding_objective": GLM52_RECOVERY_ROUNDING_OBJECTIVE,
        "stats_manifest_sha256": stats_manifest_sha256,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
        "source_lineage": dict(source_lineage),
        "source_verification": dict(source_verification),
    }


def _canonical_groups() -> tuple[tuple[int, str, str, str], ...]:
    return tuple(
        (
            layer,
            projection,
            f"{layer}:{projection}",
            f"layer-{layer:05d}-{projection}.safetensors",
        )
        for layer in range(GLM52_FIRST_ROUTED_LAYER, GLM52_LAST_ROUTED_LAYER + 1)
        for projection in GLM52_PROJECTIONS
    )


def _record_map(
    raw: object,
    *,
    label: str,
    allowed_keys: set[str],
    require_all: bool,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, Mapping[str, object]] = {}
    ordered: list[str] = []
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError(f"{label} records must be objects")
        layer = record.get("layer")
        projection = record.get("projection")
        if type(layer) is not int or projection not in GLM52_PROJECTIONS:
            raise ValueError(f"{label} has an invalid layer/projection record")
        key = f"{layer}:{projection}"
        if key not in allowed_keys:
            if layer == 78:
                raise ValueError("layer 78 MTP groups are forbidden")
            raise ValueError(f"{label} group {key!r} is outside canonical GLM-5.2 groups")
        if key in result:
            raise ValueError(f"{label} contains duplicate group {key!r}")
        result[key] = record
        ordered.append(key)
    canonical_order = [key for _, _, key, _ in _canonical_groups() if key in result]
    if require_all and ordered != canonical_order:
        raise ValueError(f"{label} group records must use canonical order")
    if require_all and set(result) != allowed_keys:
        raise ValueError(f"{label} must contain exactly 225 canonical group records")
    return result


def _validated_declared_root(
    raw: object,
    *,
    label: str,
    expected: Path | None = None,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"recovery manifest {label} must be a non-empty path")
    declared = Path(raw)
    if not declared.is_absolute():
        raise ValueError(f"recovery manifest {label} must be an absolute authenticated root")
    try:
        resolved = declared.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ValueError(f"recovery manifest {label} root is missing or cyclic") from error
    if not resolved.is_dir():
        raise ValueError(f"recovery manifest {label} root must be a directory")
    if expected is not None and resolved != expected.resolve(strict=True):
        raise ValueError(
            f"manifest replay or {label} output root mismatch: {resolved} != {expected}"
        )
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_group_link(
    link: Path,
    *,
    seed_root: Path,
    recovered_root: Path,
    filename: str,
    inherited_resolved: Path | None = None,
    allowed_alias_roots: tuple[Path, ...] = (),
) -> tuple[str, Path]:
    if not link.is_symlink():
        raise ValueError(f"mixed artifact group {filename} must be a relative symlink")
    raw_target = os.readlink(link)
    if os.path.isabs(raw_target):
        raise ValueError(f"mixed artifact group {filename} uses an absolute symlink")

    # Check the lexical target before resolving aliases.  This rejects a link
    # that traverses an untrusted alias outside both authenticated roots even
    # when that alias ultimately resolves back into an allowed root.
    lexical = Path(os.path.abspath(link.parent / raw_target))
    if not lexical.exists() and not lexical.is_symlink():
        raise ValueError(f"mixed artifact group {filename} has a missing target")
    try:
        target = link.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"mixed artifact group {filename} has a missing target") from error
    except RuntimeError as error:
        raise ValueError(f"mixed artifact group {filename} has a symlink cycle or loop") from error
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(
                f"mixed artifact group {filename} has a symlink cycle or loop"
            ) from error
        raise
    authenticated_roots = (seed_root, recovered_root, *allowed_alias_roots)
    if not any(_is_within(lexical, root) for root in authenticated_roots):
        raise ValueError(
            f"mixed artifact group {filename} traverses an alias outside authenticated roots"
        )
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"mixed artifact group {filename} target must be a regular file")
    seed_target = seed_root / filename
    recovered_target = recovered_root / filename
    if inherited_resolved is not None:
        if target == inherited_resolved:
            return "inherited", target
        if lexical == seed_target:
            raise ValueError(
                f"mixed artifact group {filename} parent binding drifted"
            )
    if target == seed_target:
        return "inherited", target
    if target == recovered_target:
        return "replacement", target
    if _is_within(target, seed_root) or _is_within(target, recovered_root):
        raise ValueError(f"mixed artifact group {filename} was retargeted to {target.name}")
    raise ValueError(f"mixed artifact group {filename} resolves outside authenticated roots")


def _parent_alias_roots(
    parent_artifact_root: Path,
    parent_records: Mapping[tuple[int, str], Mapping[str, object]],
) -> tuple[Path, ...]:
    roots = {parent_artifact_root}
    roots.update(
        Path(str(record["resolved_path"])).parent
        for record in parent_records.values()
    )
    return tuple(sorted(roots, key=os.fspath))


def _authenticated_regular_digest(path: Path, *, label: str) -> tuple[int, str]:
    authenticated = AuthenticatedFile.open(path, label=label)
    try:
        return authenticated.size, authenticated.sha256
    finally:
        authenticated.close()


def _find_parent_recovery_audit(
    parent_recovery_root: Path,
    *,
    expected_sha256: str,
) -> Path:
    matches: list[Path] = []
    for candidate in parent_recovery_root.parent.iterdir():
        if candidate.suffix != ".json" or candidate.is_symlink() or not candidate.is_file():
            continue
        _size, candidate_sha256 = _authenticated_regular_digest(
            candidate,
            label="parent recovery audit candidate",
        )
        if candidate_sha256 == expected_sha256:
            matches.append(candidate.resolve(strict=True))
    if len(matches) != 1:
        raise ValueError(
            "parent recovery audit SHA-256 must resolve to exactly one sibling JSON; "
            f"found {len(matches)}"
        )
    return matches[0]


def _verify_recovery_audit_current_identity(
    audit: GLM52RecoveryArtifactAudit,
) -> None:
    root = Path(audit.recovery_dir).resolve(strict=True)
    manifest_path = root / GLM52_RECOVERY_MANIFEST
    manifest, _manifest_file_sha256 = _load_authenticated_object(
        manifest_path,
        label="recovery manifest current identity",
    )
    body = dict(manifest)
    recorded_body_sha256 = body.pop("manifest_body_sha256", None)
    if (
        recorded_body_sha256 != audit.manifest_body_sha256
        or _canonical_sha256(body) != audit.manifest_body_sha256
    ):
        raise ValueError("recovery artifact current manifest identity changed after audit")
    mixed = manifest.get("mixed_artifact")
    if not isinstance(mixed, Mapping):
        raise ValueError("recovery artifact current mixed-artifact claim is missing")
    seed_root = _validated_declared_root(
        mixed.get("seed_artifact_dir"),
        label="seed_artifact_dir",
    )
    authority = audit._verification_authority
    parent_records: Mapping[tuple[int, str], Mapping[str, object]] | None = None
    allowed_alias_roots: tuple[Path, ...] = ()
    if authority.seed_kind == "accepted_baseline":
        _seed_manifest, seed_manifest_sha256 = _load_authenticated_object(
            seed_root / GLM52_RECOVERY_MANIFEST,
            label="accepted seed manifest current identity",
        )
        if seed_manifest_sha256 != audit.seed_manifest_sha256:
            raise ValueError("recovery artifact current seed identity changed after audit")
    else:
        if (
            authority.parent_recovery_dir is None
            or authority.parent_recovery_audit_json is None
            or authority.expected_parent_manifest_sha256 is None
            or authority.expected_parent_audit_sha256 is None
        ):
            raise ValueError("recovery artifact parent verification authority is incomplete")
        parent = _authenticate_recovery_seed(
            recovery_root=authority.parent_recovery_dir,
            audit_path=authority.parent_recovery_audit_json,
            expected_manifest_sha256=authority.expected_parent_manifest_sha256,
            expected_audit_sha256=authority.expected_parent_audit_sha256,
            expected_stats_manifest_sha256=authority.expected_stats_manifest_sha256,
            expected_composite_audit_sha256=authority.expected_composite_audit_sha256,
        )
        if (
            parent.artifact_root != seed_root
            or parent.candidate_identity_sha256
            != authority.expected_parent_candidate_identity_sha256
        ):
            raise ValueError("recovery artifact current parent identity changed after audit")
        parent_records = parent.records
        allowed_alias_roots = _parent_alias_roots(seed_root, parent_records)
    _composite, composite_sha256 = _load_authenticated_object(
        Path(authority.accepted_composite_audit_json),
        label="accepted composite audit current identity",
    )
    if composite_sha256 != audit.accepted_composite_audit_sha256:
        raise ValueError("recovery artifact current composite identity changed after audit")
    artifact_root = root / "artifact"
    recovered_root = root / "recovered-groups"
    expected_groups = {group.filename: group for group in audit.groups}
    actual_entries = {entry.name for entry in artifact_root.iterdir()}
    if actual_entries != set(expected_groups):
        raise ValueError("recovery artifact current group inventory changed after audit")
    for filename, group in expected_groups.items():
        layer_text, projection = group.group_key.split(":", 1)
        parent_record = (
            parent_records[(int(layer_text), projection)]
            if parent_records is not None
            else None
        )
        classification, resolved = _resolve_group_link(
            artifact_root / filename,
            seed_root=seed_root,
            recovered_root=recovered_root,
            filename=filename,
            inherited_resolved=(
                Path(str(parent_record["resolved_path"]))
                if parent_record is not None
                else None
            ),
            allowed_alias_roots=allowed_alias_roots,
        )
        if classification != group.classification:
            raise ValueError(
                f"recovery artifact current classification changed after audit: {filename}"
            )
        artifact_bytes, artifact_sha256 = _authenticated_regular_digest(
            resolved,
            label=f"current recovery group {group.group_key}",
        )
        if (
            artifact_bytes != group.artifact_bytes
            or artifact_sha256 != group.artifact_sha256
        ):
            raise ValueError(
                f"recovery artifact current group identity changed after audit: {filename}"
            )


def _projection_dims(profile: _Profile, projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return profile.hidden_size, profile.moe_intermediate_size
    return profile.moe_intermediate_size, profile.hidden_size


def _group_size(profile: _Profile, projection: str) -> int:
    key = projection.removesuffix("_proj")
    try:
        value = profile.group_size_policy[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"profile has no group-size policy for {projection}") from error
    if type(value) is not int or value <= 0:
        raise ValueError(f"profile group size for {projection} must be positive")
    return value


def _validate_record_binding(
    record: Mapping[str, object],
    *,
    label: str,
    filename: str,
    snapshot: _GroupSnapshot,
) -> tuple[int, str]:
    recorded_path = record.get("artifact_path")
    if recorded_path != filename:
        raise ValueError(f"{label} artifact_path must be canonical filename {filename!r}")
    actual_bytes = snapshot.identity.size
    actual_sha256 = snapshot.sha256
    _expect_int(f"{label} artifact bytes", record.get("artifact_bytes"), actual_bytes)
    _expect(f"{label} artifact SHA-256", record.get("artifact_sha256"), actual_sha256)
    return actual_bytes, actual_sha256


def _load_metadata_object(
    metadata: Mapping[str, str], name: str, *, filename: str
) -> dict[str, Any]:
    raw = metadata.get(name)
    if raw is None:
        raise ValueError(f"{filename} is missing {name} metadata")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{filename} has invalid {name} metadata") from error
    if not isinstance(value, dict):
        raise ValueError(f"{filename} {name} metadata must be an object")
    return value


def _validate_group_payload(
    path: Path,
    *,
    filename: str,
    layer: int,
    projection: str,
    profile: _Profile,
    source_lineage: Mapping[str, str],
    classification: str,
    replacement_record: Mapping[str, object] | None,
    expected_recovery_policy: Mapping[str, str],
    expected_recovery_provenance: Mapping[str, object],
) -> tuple[int, int, int, int, str, int, str, str, dict[str, object]]:
    input_dim, output_dim = _projection_dims(profile, projection)
    group_size = _group_size(profile, projection)
    if input_dim % 8 or input_dim % group_size:
        raise ValueError(f"profile dimensions are incompatible with {filename}")
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    codes_name = f"{prefix}.codes"
    scales_name = f"{prefix}.scales"
    expected_names = {codes_name, scales_name, GLM52_CODEBOOK_TENSOR}
    with safe_open(path, framework="np") as handle:
        names = set(handle.keys())
        if names != expected_names:
            missing = sorted(expected_names - names)
            unexpected = sorted(names - expected_names)
            raise ValueError(
                f"{filename} tensor inventory mismatch; missing={missing}, unexpected={unexpected}"
            )
        codes = handle.get_tensor(codes_name)
        scales = handle.get_tensor(scales_name)
        codebook = handle.get_tensor(GLM52_CODEBOOK_TENSOR)
        metadata = dict(handle.metadata() or {})
    if codes.dtype == np.dtype(np.uint8):
        code_bits = 8
        codes_dtype = "uint8"
        canonical_codebook = e8_1bit_packed()
    elif codes.dtype == np.dtype(np.uint16):
        code_bits = 16
        codes_dtype = "uint16"
        canonical_codebook = e8p_packed_abs_grid()
    else:
        raise ValueError(f"{filename} codes dtype must be uint8 or uint16")
    expected_codes_shape = (profile.num_experts, output_dim, input_dim // 8)
    expected_scales_shape = (
        profile.num_experts,
        output_dim,
        input_dim // group_size,
    )
    if codes.shape != expected_codes_shape:
        raise ValueError(
            f"{filename} codes shape must be {expected_codes_shape}, found {codes.shape}"
        )
    if scales.dtype != np.dtype(np.float16) or scales.shape != expected_scales_shape:
        raise ValueError(
            f"{filename} scales must be float16 with shape {expected_scales_shape}"
        )
    if codebook.dtype != np.dtype(np.uint32) or codebook.shape != (256,):
        raise ValueError(f"{filename} codebook must be uint32 with shape (256,)")
    if not np.array_equal(codebook, canonical_codebook):
        raise ValueError(f"{filename} embedded codebook bytes are not canonical")

    quantization = _load_metadata_object(
        metadata, "quantization_config", filename=filename
    )
    _expect(f"{filename} quant_method", quantization.get("quant_method"), "mlx_vq_e8")
    _expect_int(f"{filename} quantization version", quantization.get("version"), 1)
    _expect_int(f"{filename} code bits", quantization.get("default_code_bits"), code_bits)
    _expect_int(f"{filename} group size", quantization.get("default_group_size"), group_size)
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(code_bits)
    codebook_metadata = quantization.get("codebook")
    if not isinstance(codebook_metadata, dict):
        raise ValueError(f"{filename} codebook metadata must be an object")
    _expect(f"{filename} codebook name", codebook_metadata.get("name"), codebook_name)
    _expect(f"{filename} codebook dtype", codebook_metadata.get("dtype"), "uint32")
    _expect_int(f"{filename} codebook entries", codebook_metadata.get("entries"), 256)
    _expect(
        f"{filename} codebook SHA-256",
        codebook_metadata.get("sha256"),
        codebook_sha256,
    )
    actual_codebook_sha256 = hashlib.sha256(
        np.asarray(codebook, dtype=np.uint32).tobytes(order="C")
    ).hexdigest()
    _expect(
        f"{filename} embedded codebook SHA-256",
        actual_codebook_sha256,
        codebook_sha256,
    )
    policy = quantization.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{filename} quantization policy must be an object")
    for name, expected in source_lineage.items():
        _expect(f"{filename} source lineage {name}", policy.get(name), expected)

    provenance: dict[str, object] = {}
    if classification == "replacement":
        provenance = _load_metadata_object(
            metadata, "glm52_recovery_provenance", filename=filename
        )
        if replacement_record is None:
            raise ValueError(f"{filename} replacement classification has no manifest record")
        manifest_provenance = replacement_record.get("lever_provenance")
        legacy_provenance = {
            key: value
            for key, value in expected_recovery_provenance.items()
            if key not in {"source_lineage", "source_verification"}
        }
        legacy_resumed = (
            replacement_record.get("status") == "resumed"
            and provenance == legacy_provenance
        )
        if provenance != manifest_provenance and not legacy_resumed:
            raise ValueError(f"{filename} recovery provenance does not match manifest record")
        if not legacy_resumed:
            _expect(
                f"{filename} exact recovery provenance",
                provenance,
                expected_recovery_provenance,
            )
        _expect(f"{filename} exact recovery policy", policy, expected_recovery_policy)

    codes_scales_bytes = int(codes.nbytes + scales.nbytes)
    codebook_bytes = int(codebook.nbytes)
    tensor_payload = codes_scales_bytes + codebook_bytes
    resolved_policy = {
        "quant_method": "mlx_vq_e8",
        "version": 1,
        "code_bits": code_bits,
        "codes_dtype": codes_dtype,
        "group_size": group_size,
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "tensor_payload_bytes": tensor_payload,
        "recovery_provenance": provenance,
    }
    return (
        tensor_payload,
        codes_scales_bytes,
        codebook_bytes,
        code_bits,
        codes_dtype,
        group_size,
        codebook_name,
        codebook_sha256,
        resolved_policy,
    )


def audit_glm52_recovery_mixed_artifact(
    recovery_dir: str | Path,
    *,
    profile: _Profile,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy: Mapping[str, str],
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
) -> GLM52RecoveryArtifactAudit:
    """Authenticate and audit one diagnostic mixed-artifact candidate.

    The accepted composite audit is stable-read under ``O_NOFOLLOW`` and must
    match its externally pinned SHA-256 before it can authorize accounting.
    """

    _require_sha256(expected_seed_manifest_sha256, label="expected seed manifest SHA-256")
    if expected_seed_manifest_sha256 != PINNED_ACCEPTED_SEED_MANIFEST_SHA256:
        raise ValueError(
            "expected seed manifest SHA-256 does not match the repository-pinned "
            "accepted baseline"
        )
    _require_sha256(expected_stats_manifest_sha256, label="expected stats manifest SHA-256")
    _require_sha256(
        expected_full_source_blob_inventory_sha256,
        label="expected full source blob inventory SHA-256",
    )
    _require_sha256(
        expected_routed_source_blob_inventory_sha256,
        label="expected routed source blob inventory SHA-256",
    )
    if expected_recovery_lever != GLM52_RECOVERY_LEVER:
        raise ValueError("expected recovery lever must be the canonical GLM52 recovery lever")
    if not isinstance(expected_recovery_policy, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in expected_recovery_policy.items()
    ):
        raise ValueError("expected recovery policy must map strings to strings")
    _expect(
        "expected recovery policy lever",
        expected_recovery_policy.get("recovery_lever"),
        expected_recovery_lever,
    )
    _expect(
        "expected recovery policy scale estimator",
        expected_recovery_policy.get("scale_estimator"),
        GLM52_RECOVERY_SCALE_ESTIMATOR,
    )
    _expect(
        "expected recovery policy rounding objective",
        expected_recovery_policy.get("rounding_objective"),
        GLM52_RECOVERY_ROUNDING_OBJECTIVE,
    )

    root = Path(recovery_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"recovery directory does not exist: {root}")
    manifest_path = root / GLM52_RECOVERY_MANIFEST
    manifest, _manifest_file_sha256 = _load_authenticated_object(
        manifest_path,
        label="recovery manifest",
    )
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError(
            "recovery manifest schema version must be 1 or 2, "
            f"found {schema_version!r}"
        )
    manifest_keys = {
        "schema_version",
        "record_type",
        "status",
        "resumable",
        "selected_groups",
        "source_lineage",
        "source_verification",
        "expected_full_source_blob_inventory_sha256",
        "expected_routed_source_blob_inventory_sha256",
        "stats_manifest_sha256",
        "accepted_composite_audit_sha256",
        "recovery_policy",
        "lever_provenance",
        "groups",
        "mixed_artifact",
        "accounting",
        "manifest_body_sha256",
    }
    if schema_version == 2:
        manifest_keys.add("rate_policy")
    _expect_exact_keys("recovery manifest", manifest, manifest_keys)
    recorded_body_sha256 = manifest.get("manifest_body_sha256")
    body = dict(manifest)
    body.pop("manifest_body_sha256", None)
    actual_body_sha256 = _canonical_sha256(body)
    if recorded_body_sha256 != actual_body_sha256:
        raise ValueError(
            "recovery manifest body SHA-256 does not authenticate its canonical body"
        )
    rate_policy = (
        _validate_rate_policy(manifest.get("rate_policy"))
        if schema_version == 2
        else None
    )
    _expect(
        "recovery manifest record type",
        manifest.get("record_type"),
        GLM52_RECOVERY_RECORD_TYPE,
    )
    _expect("recovery manifest status", manifest.get("status"), "complete")
    _expect("recovery manifest resumable", manifest.get("resumable"), True)
    _expect(
        "recovery manifest accepted stats SHA-256",
        manifest.get("stats_manifest_sha256"),
        expected_stats_manifest_sha256,
    )
    _expect(
        "recovery manifest expected full source blob inventory SHA-256",
        manifest.get("expected_full_source_blob_inventory_sha256"),
        expected_full_source_blob_inventory_sha256,
    )
    _expect(
        "recovery manifest expected routed source blob inventory SHA-256",
        manifest.get("expected_routed_source_blob_inventory_sha256"),
        expected_routed_source_blob_inventory_sha256,
    )

    mixed = manifest.get("mixed_artifact")
    if not isinstance(mixed, dict):
        raise ValueError("recovery manifest mixed_artifact must be an object")
    mixed_keys = set(mixed)
    if mixed_keys == _CLASSIC_MIXED_ARTIFACT_KEYS:
        seed_kind = "accepted_baseline"
    elif mixed_keys == _RECOVERY_SEEDED_MIXED_ARTIFACT_KEYS:
        seed_kind = GLM52_RECOVERY_SEED_KIND
    else:
        expected = (
            _RECOVERY_SEEDED_MIXED_ARTIFACT_KEYS
            if "seed_kind" in mixed_keys
            or any(key.startswith("seed_recovery_") for key in mixed_keys)
            else _CLASSIC_MIXED_ARTIFACT_KEYS
        )
        _expect_exact_keys("recovery manifest mixed_artifact", mixed, expected)
        raise AssertionError("unreachable exact mixed-artifact schema")
    artifact_root = _validated_declared_root(
        mixed.get("output_dir"), label="output_dir", expected=root / "artifact"
    )
    recovered_root = _validated_declared_root(
        mixed.get("recovered_groups_dir"),
        label="recovered_groups_dir",
        expected=root / "recovered-groups",
    )
    seed_root = _validated_declared_root(
        mixed.get("seed_artifact_dir"), label="seed_artifact_dir"
    )
    if seed_root == recovered_root or seed_root == artifact_root:
        raise ValueError("authenticated seed root must be distinct from candidate roots")

    parent_recovery_root: Path | None = None
    parent_audit_path: Path | None = None
    parent_manifest_sha256: str | None = None
    parent_audit_sha256: str | None = None
    parent_candidate_identity_sha256: str | None = None
    parent_records: Mapping[tuple[int, str], Mapping[str, object]] | None = None
    allowed_alias_roots: tuple[Path, ...] = ()
    seed_manifest: Mapping[str, object] | None = None
    if seed_kind == "accepted_baseline":
        _expect(
            "recovery manifest seed manifest SHA-256",
            mixed.get("seed_manifest_sha256"),
            expected_seed_manifest_sha256,
        )
        seed_manifest, actual_seed_manifest_sha256 = _load_authenticated_object(
            seed_root / GLM52_RECOVERY_MANIFEST,
            label="seed manifest",
        )
        _expect(
            "authenticated seed manifest SHA-256",
            actual_seed_manifest_sha256,
            expected_seed_manifest_sha256,
        )
        source_lineage = {
            "source_model_id": str(seed_manifest.get("model_id")),
            "source_revision": str(seed_manifest.get("source_revision")),
            "source_config_sha256": str(seed_manifest.get("config_sha256")),
            "source_index_sha256": str(seed_manifest.get("index_sha256")),
            "source_profile": str(seed_manifest.get("profile")),
        }
    else:
        _expect("recovery manifest seed kind", mixed.get("seed_kind"), seed_kind)
        parent_manifest_sha256 = _require_sha256(
            mixed.get("seed_recovery_manifest_sha256"),
            label="parent recovery manifest SHA-256",
        )
        parent_audit_sha256 = _require_sha256(
            mixed.get("seed_recovery_audit_sha256"),
            label="parent recovery audit SHA-256",
        )
        parent_candidate_identity_sha256 = _require_sha256(
            mixed.get("seed_candidate_identity_sha256"),
            label="parent recovery candidate identity SHA-256",
        )
        _expect(
            "recovery manifest accepted baseline seed manifest SHA-256",
            mixed.get("accepted_baseline_seed_manifest_sha256"),
            expected_seed_manifest_sha256,
        )
        parent_artifact_root = _validated_declared_root(
            mixed.get("parent_artifact_root"),
            label="parent_artifact_root",
            expected=seed_root,
        )
        parent_recovery_root = parent_artifact_root.parent
        _parent_manifest, actual_parent_manifest_sha256 = _load_authenticated_object(
            parent_recovery_root / GLM52_RECOVERY_MANIFEST,
            label="parent recovery manifest",
        )
        _expect(
            "parent recovery manifest SHA-256",
            actual_parent_manifest_sha256,
            parent_manifest_sha256,
        )
        parent_audit_path = _find_parent_recovery_audit(
            parent_recovery_root,
            expected_sha256=parent_audit_sha256,
        )
        parent_seed = _authenticate_recovery_seed(
            recovery_root=parent_recovery_root,
            audit_path=parent_audit_path,
            expected_manifest_sha256=parent_manifest_sha256,
            expected_audit_sha256=parent_audit_sha256,
            expected_stats_manifest_sha256=expected_stats_manifest_sha256,
            expected_composite_audit_sha256=expected_composite_audit_sha256,
        )
        _expect(
            "parent recovery candidate identity",
            parent_seed.candidate_identity_sha256,
            parent_candidate_identity_sha256,
        )
        source_lineage = dict(parent_seed.source_lineage)
        parent_records = parent_seed.records
        allowed_alias_roots = _parent_alias_roots(
            parent_artifact_root,
            parent_records,
        )
        actual_seed_manifest_sha256 = expected_seed_manifest_sha256

    _expect("seed model ID", source_lineage["source_model_id"], profile.hf_model_id)
    _expect("seed revision", source_lineage["source_revision"], str(profile.revision))
    _expect("seed profile", source_lineage["source_profile"], profile.name)
    for name in ("source_config_sha256", "source_index_sha256"):
        value = source_lineage[name]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"seed {name} must be a lowercase SHA-256")
    composite_authority = _authenticate_accepted_composite_audit(
        accepted_composite_audit_json,
        expected_sha256=expected_composite_audit_sha256,
        expected_source_lineage=source_lineage,
    )
    accepted_routed_codebook_bytes = (
        GLM52_GROUP_COUNT * GLM52_CODEBOOK_BYTES_PER_GROUP
    )
    accepted_reconciled_payload = (
        composite_authority.actual_routed_payload_bytes
        + accepted_routed_codebook_bytes
        + composite_authority.non_routed_tensor_payload_bytes
    )
    _expect_int(
        "accepted composite whole-model tensor payload reconciliation",
        composite_authority.actual_whole_model_tensor_payload_bytes,
        accepted_reconciled_payload,
    )
    _expect_int(
        "accepted composite canonical whole-model tensor payload",
        accepted_reconciled_payload,
        GLM52_ACCEPTED_WHOLE_TENSOR_PAYLOAD_BYTES,
    )
    _expect(
        "recovery manifest accepted composite audit SHA-256",
        manifest.get("accepted_composite_audit_sha256"),
        composite_authority.sha256,
    )
    seed_records_by_group = (
        _validate_seed_manifest(
            seed_manifest,
            expected_source_lineage=source_lineage,
        )
        if seed_manifest is not None
        else parent_records
    )
    if seed_records_by_group is None:
        raise ValueError("recovery seed group authority is missing")
    seed_records = {
        f"{layer}:{projection}": record
        for (layer, projection), record in seed_records_by_group.items()
    }
    _expect("recovery manifest source lineage", manifest.get("source_lineage"), source_lineage)
    source_verification = manifest.get("source_verification")
    if not isinstance(source_verification, dict):
        raise ValueError("recovery manifest source_verification must be an object")
    _expect_exact_keys(
        "recovery manifest source_verification",
        source_verification,
        {
            "source_blob_inventory_sha256",
            "routed_source_blob_inventory_sha256",
            "shard_count",
        },
    )
    _expect(
        "full source blob inventory SHA-256",
        source_verification.get("source_blob_inventory_sha256"),
        expected_full_source_blob_inventory_sha256,
    )
    _expect(
        "routed source blob inventory SHA-256",
        source_verification.get("routed_source_blob_inventory_sha256"),
        expected_routed_source_blob_inventory_sha256,
    )
    if type(source_verification.get("shard_count")) is not int or source_verification["shard_count"] <= 0:
        raise ValueError("source verification shard_count must be a positive integer")
    expected_provenance = _expected_provenance(
        stats_manifest_sha256=expected_stats_manifest_sha256,
        recovery_lever=expected_recovery_lever,
        source_lineage=source_lineage,
        source_verification=source_verification,
    )
    _expect(
        "recovery manifest exact recovery provenance",
        manifest.get("lever_provenance"),
        expected_provenance,
    )
    _expect(
        "recovery manifest exact recovery policy",
        manifest.get("recovery_policy"),
        dict(expected_recovery_policy),
    )

    canonical = _canonical_groups()
    canonical_keys = {key for _, _, key, _ in canonical}
    canonical_filenames = {filename for _, _, _, filename in canonical}
    if seed_manifest is not None:
        _expect_int(
            "seed planned group count",
            seed_manifest.get("planned_vq_groups"),
            GLM52_GROUP_COUNT,
        )
        _expect_int(
            "seed ready group count",
            seed_manifest.get("ready_vq_groups"),
            GLM52_GROUP_COUNT,
        )
    replacement_records = _record_map(
        manifest.get("groups"),
        label="recovery manifest groups",
        allowed_keys=canonical_keys,
        require_all=False,
    )
    replacement_record_keys = {
        "layer",
        "projection",
        "status",
        "artifact_path",
        "artifact_bytes",
        "artifact_sha256",
        "source_lineage",
        "recovery_policy",
        "lever_provenance",
    }
    if schema_version == 2:
        replacement_record_keys.update(
            {
                "code_bits",
                "codes_dtype",
                "codebook_name",
                "codebook_sha256",
                "tensor_payload_bytes",
                "zero_importance_policy",
            }
        )
    for key, record in replacement_records.items():
        _expect_exact_keys(f"recovery manifest group {key}", record, replacement_record_keys)
        _expect(f"recovery group {key} source lineage", record.get("source_lineage"), source_lineage)
        _expect(
            f"recovery group {key} exact recovery policy",
            record.get("recovery_policy"),
            dict(expected_recovery_policy),
        )
        _expect(
            f"recovery group {key} exact recovery provenance",
            record.get("lever_provenance"),
            expected_provenance,
        )
    selected_groups = manifest.get("selected_groups")
    expected_selected = list(replacement_records)
    _expect("recovery selected groups", selected_groups, expected_selected)

    actual_artifact_entries = {entry.name for entry in artifact_root.iterdir()}
    if actual_artifact_entries != canonical_filenames:
        missing = sorted(canonical_filenames - actual_artifact_entries)
        unexpected = sorted(actual_artifact_entries - canonical_filenames)
        if any(name.startswith("layer-00078-") for name in unexpected):
            raise ValueError(f"unexpected layer 78 artifact entry: {unexpected}")
        raise ValueError(
            f"mixed artifact must contain exactly 225 canonical files; "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_recovered_names = {
        filename for _, _, key, filename in canonical if key in replacement_records
    }
    actual_recovered_entries = {entry.name for entry in recovered_root.iterdir()}
    if actual_recovered_entries != expected_recovered_names:
        raise ValueError(
            "recovered-groups directory has unexpected, missing, or masquerading files"
        )

    group_audits: list[GLM52RecoveryGroupAudit] = []
    identity_groups: list[dict[str, object]] = []
    layer_rates: dict[int, dict[str, int]] = {}
    replacement_layers: dict[int, set[str]] = {}
    routed_codes_scales_bytes = 0
    routed_codebook_bytes = 0
    incremental_disk_bytes = 0
    for layer, projection, key, filename in canonical:
        parent_record = (
            parent_records[(layer, projection)]
            if parent_records is not None
            else None
        )
        classification, resolved = _resolve_group_link(
            artifact_root / filename,
            seed_root=seed_root,
            recovered_root=recovered_root,
            filename=filename,
            inherited_resolved=(
                Path(str(parent_record["resolved_path"]))
                if parent_record is not None
                else None
            ),
            allowed_alias_roots=allowed_alias_roots,
        )
        replacement_record = replacement_records.get(key)
        expected_classification = (
            "replacement" if replacement_record is not None else "inherited"
        )
        if classification != expected_classification:
            raise ValueError(
                f"{filename} recovered-group masquerade/classification mismatch: "
                f"manifest says {expected_classification}, link says {classification}"
            )
        record = replacement_record or seed_records[key]
        if replacement_record is not None and replacement_record.get("status") not in {
            "materialized",
            "resumed",
        }:
            raise ValueError(f"replacement group {key} has an invalid status")
        with _authenticated_group_snapshot(
            resolved,
            label=f"{classification} group {key}",
        ) as snapshot:
            artifact_bytes, artifact_sha256 = _validate_record_binding(
                record,
                label=f"{classification} group {key}",
                filename=filename,
                snapshot=snapshot,
            )
            (
                tensor_payload,
                codes_scales_bytes,
                codebook_bytes,
                code_bits,
                codes_dtype,
                group_size,
                codebook_name,
                codebook_sha256,
                resolved_policy,
            ) = _validate_group_payload(
                snapshot.copy_path,
                filename=filename,
                layer=layer,
                projection=projection,
                profile=profile,
                source_lineage=source_lineage,
                classification=classification,
                replacement_record=replacement_record,
                expected_recovery_policy=expected_recovery_policy,
                expected_recovery_provenance=expected_provenance,
            )
        if (
            classification == "inherited"
            and seed_kind == "accepted_baseline"
            and code_bits != 8
        ):
            raise ValueError(
                f"{filename} inherited E8P is forbidden against the accepted E8 seed"
            )
        if parent_record is not None and classification == "inherited":
            _expect_int(
                f"parent recovery group {key} code bits",
                parent_record.get("code_bits"),
                code_bits,
            )
            _expect(
                f"parent recovery group {key} codes dtype",
                parent_record.get("codes_dtype"),
                codes_dtype,
            )
            _expect(
                f"parent recovery group {key} codebook name",
                parent_record.get("codebook_name"),
                codebook_name,
            )
            _expect(
                f"parent recovery group {key} codebook SHA-256",
                parent_record.get("codebook_sha256"),
                codebook_sha256,
            )
        if schema_version == 1 and code_bits != 8:
            raise ValueError(
                "schema-v1 recovery candidates must remain uniform E8 and cannot claim E8P"
            )
        if replacement_record is not None and schema_version == 2:
            _expect_int(
                f"replacement group {key} code bits",
                replacement_record.get("code_bits"),
                code_bits,
            )
            _expect(
                f"replacement group {key} codes dtype",
                replacement_record.get("codes_dtype"),
                codes_dtype,
            )
            _expect(
                f"replacement group {key} codebook name",
                replacement_record.get("codebook_name"),
                codebook_name,
            )
            _expect(
                f"replacement group {key} codebook SHA-256",
                replacement_record.get("codebook_sha256"),
                codebook_sha256,
            )
            _expect_int(
                f"replacement group {key} tensor payload bytes",
                replacement_record.get("tensor_payload_bytes"),
                tensor_payload,
            )
            _expect(
                f"replacement group {key} zero importance policy",
                replacement_record.get("zero_importance_policy"),
                _zero_importance_policy_for_bits(code_bits),
            )
        classification_after, resolved_after = _resolve_group_link(
            artifact_root / filename,
            seed_root=seed_root,
            recovered_root=recovered_root,
            filename=filename,
            inherited_resolved=(
                Path(str(parent_record["resolved_path"]))
                if parent_record is not None
                else None
            ),
            allowed_alias_roots=allowed_alias_roots,
        )
        if classification_after != classification or resolved_after != resolved:
            raise ValueError(f"{filename} link was retargeted during authenticated inspection")
        routed_codes_scales_bytes += codes_scales_bytes
        routed_codebook_bytes += codebook_bytes
        layer_rates.setdefault(layer, {})[projection] = code_bits
        if classification == "replacement":
            incremental_disk_bytes += artifact_bytes
            replacement_layers.setdefault(layer, set()).add(projection)
        group_audits.append(
            GLM52RecoveryGroupAudit(
                group_key=key,
                filename=filename,
                classification=classification,
                artifact_sha256=artifact_sha256,
                artifact_bytes=artifact_bytes,
                tensor_payload_bytes=tensor_payload,
                code_bits=code_bits,
                codes_dtype=codes_dtype,
                group_size=group_size,
                codebook_name=codebook_name,
                codebook_sha256=codebook_sha256,
            )
        )
        identity_group = {
            "group_key": key,
            "classification": classification,
            "artifact_sha256": artifact_sha256,
            "tensor_payload_bytes": tensor_payload,
            "resolved_policy": resolved_policy,
        }
        if schema_version == 1:
            identity_group["resolved_policy"] = {
                name: value
                for name, value in resolved_policy.items()
                if name not in {"codes_dtype", "tensor_payload_bytes"}
            }
        else:
            identity_group.update(
                {
                    "code_bits": code_bits,
                    "codes_dtype": codes_dtype,
                    "codebook_name": codebook_name,
                    "codebook_sha256": codebook_sha256,
                }
            )
        identity_groups.append(identity_group)

    inconsistent_layers = {
        layer: rates for layer, rates in layer_rates.items() if len(set(rates.values())) != 1
    }
    if inconsistent_layers:
        raise ValueError(
            "complete layer rate consistency across gate/up/down failed: "
            f"{inconsistent_layers}"
        )
    if schema_version == 2:
        derived_rate_policy = {
            "default_code_bits": 8,
            "layer_code_bits": {
                str(layer): 16
                for layer, rates in sorted(layer_rates.items())
                if set(rates.values()) == {16}
            },
            "complete_layer_rates_required": True,
        }
        _expect(
            "recovery manifest rate policy derived from authenticated tensors",
            rate_policy,
            derived_rate_policy,
        )
    routed_payload_bytes = routed_codes_scales_bytes + routed_codebook_bytes
    main_non_routed_tensor_payload_bytes = (
        composite_authority.non_routed_tensor_payload_bytes
    )
    logical_payload_bytes = (
        main_non_routed_tensor_payload_bytes
        + routed_codes_scales_bytes
        + routed_codebook_bytes
    )
    _validate_logical_payload_budget(logical_payload_bytes)

    replacement_count = len(replacement_records)
    inherited_count = GLM52_GROUP_COUNT - replacement_count
    _expect_int(
        "mixed artifact replacement count", mixed.get("replacement_count"), replacement_count
    )
    _expect_int(
        "mixed artifact inherited count", mixed.get("inherited_group_count"), inherited_count
    )
    accounting = manifest.get("accounting")
    if not isinstance(accounting, dict):
        raise ValueError("recovery manifest accounting must be an object")
    accounting_keys = {
        "logical_whole_model_tensor_payload_bytes",
        "incremental_disk_bytes",
        "logical_payload_limit_bytes",
    }
    if schema_version == 2:
        accounting_keys.update(
            {
                "routed_codes_scales_bytes",
                "routed_codebook_bytes",
                "main_non_routed_tensor_payload_bytes",
            }
        )
    else:
        accounting_keys.add("non_routed_tensor_payload_bytes")
    _expect_exact_keys("recovery manifest accounting", accounting, accounting_keys)
    if schema_version == 2:
        _expect_int(
            "accounting routed codes and scales bytes",
            accounting.get("routed_codes_scales_bytes"),
            routed_codes_scales_bytes,
        )
        _expect_int(
            "accounting routed codebook bytes",
            accounting.get("routed_codebook_bytes"),
            routed_codebook_bytes,
        )
        _expect_int(
            "accounting main non-routed tensor payload",
            accounting.get("main_non_routed_tensor_payload_bytes"),
            main_non_routed_tensor_payload_bytes,
        )
    else:
        _expect_int(
            "accounting non-routed tensor payload",
            accounting.get("non_routed_tensor_payload_bytes"),
            main_non_routed_tensor_payload_bytes,
        )
    _expect_int(
        "accounting logical whole-model tensor payload",
        accounting.get("logical_whole_model_tensor_payload_bytes"),
        logical_payload_bytes,
    )
    _expect_int(
        "accounting incremental disk bytes",
        accounting.get("incremental_disk_bytes"),
        incremental_disk_bytes,
    )
    _expect_int(
        "accounting logical payload limit",
        accounting.get("logical_payload_limit_bytes"),
        GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES,
    )

    identity: dict[str, object] = {
        "schema_version": schema_version,
        "source_lineage": source_lineage,
        "seed_manifest_sha256": expected_seed_manifest_sha256,
        "accepted_baseline_composite_identity_sha256": (
            composite_authority.accepted_composite_identity_sha256
        ),
        "accepted_composite_audit_sha256": composite_authority.sha256,
        "groups": identity_groups,
        "logical_whole_model_tensor_payload_bytes": logical_payload_bytes,
    }
    if schema_version == 1:
        identity["non_routed_tensor_payload_bytes"] = (
            main_non_routed_tensor_payload_bytes
        )
    else:
        identity.update(
            {
                "rate_policy": rate_policy,
                "routed_codes_scales_bytes": routed_codes_scales_bytes,
                "routed_codebook_bytes": routed_codebook_bytes,
                "main_non_routed_tensor_payload_bytes": (
                    main_non_routed_tensor_payload_bytes
                ),
            }
        )
    candidate_identity_sha256 = _canonical_sha256(identity)
    complete_replacement_layers = tuple(
        layer
        for layer, projections in sorted(replacement_layers.items())
        if projections == set(GLM52_PROJECTIONS)
    )
    checks = {
        "canonical_manifest_body_authenticated": True,
        "seed_manifest_authenticated": True,
        "exact_225_canonical_groups": True,
        "safe_relative_symlinks": True,
        "group_classification_and_hash_binding": True,
        "tensor_inventory_shapes_and_dtypes": True,
        "canonical_codebooks": True,
        "complete_layer_rate_consistency": True,
        "logical_payload_budget": True,
        "path_independent_candidate_identity": True,
    }
    return GLM52RecoveryArtifactAudit(
        recovery_dir=str(root),
        manifest_path=str(manifest_path),
        manifest_body_sha256=actual_body_sha256,
        seed_manifest_sha256=actual_seed_manifest_sha256,
        accepted_composite_audit_sha256=composite_authority.sha256,
        accepted_baseline_composite_identity_sha256=(
            composite_authority.accepted_composite_identity_sha256
        ),
        group_count=GLM52_GROUP_COUNT,
        replacement_group_count=replacement_count,
        inherited_group_count=inherited_count,
        complete_replacement_layer_ids=complete_replacement_layers,
        routed_codes_scales_bytes=routed_codes_scales_bytes,
        routed_codebook_bytes=routed_codebook_bytes,
        routed_tensor_payload_bytes=routed_payload_bytes,
        main_non_routed_tensor_payload_bytes=main_non_routed_tensor_payload_bytes,
        non_routed_tensor_payload_bytes=main_non_routed_tensor_payload_bytes,
        logical_whole_model_tensor_payload_bytes=logical_payload_bytes,
        logical_payload_limit_bytes=GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES,
        incremental_disk_bytes=incremental_disk_bytes,
        candidate_identity_sha256=candidate_identity_sha256,
        groups=tuple(group_audits),
        checks=checks,
        _verification_authority=_RecoveryVerificationAuthority(
            profile=profile,
            expected_seed_manifest_sha256=expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=expected_recovery_lever,
            expected_recovery_policy=dict(expected_recovery_policy),
            accepted_composite_audit_json=str(accepted_composite_audit_json),
            expected_composite_audit_sha256=expected_composite_audit_sha256,
            seed_kind=seed_kind,
            parent_recovery_dir=(
                str(parent_recovery_root)
                if parent_recovery_root is not None
                else None
            ),
            parent_recovery_audit_json=(
                str(parent_audit_path) if parent_audit_path is not None else None
            ),
            expected_parent_manifest_sha256=parent_manifest_sha256,
            expected_parent_audit_sha256=parent_audit_sha256,
            expected_parent_candidate_identity_sha256=(
                parent_candidate_identity_sha256
            ),
        ),
    )


__all__ = [
    "GLM52RecoveryArtifactAudit",
    "GLM52RecoveryGroupAudit",
    "audit_glm52_recovery_mixed_artifact",
]
