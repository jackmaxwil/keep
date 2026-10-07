"""Selection-only diagonal-Hessian recovery materialization for GLM-5.2.

The recovery artifact preserves the existing one-file-per-layer/projection
safetensors contract.  Only selected group files are rewritten; every other
group is inherited byte-for-byte from the seed artifact tree.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
from types import ModuleType
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from keep.vq.e8 import (
    e8_1bit_grid,
    e8_1bit_packed,
    e8p_full_grid,
    e8p_packed_abs_grid,
)
from keep.io.schema import QuantizationConfig, codebook_metadata_for_bits
from keep.io.authenticated_artifacts import (
    AuthenticatedFile,
    PublishedArtifact,
    ReplacementAuthority,
    archive_displaced_artifact,
    clone_or_copy_authenticated,
    publish_file_transactionally,
)
from keep.quant.rtn import (
    nearest_e8_codes_diagonal_hessian,
    nearest_e8p_codes_diagonal_hessian,
    quantize_weight_rtn,
)
RECOVERY_LEVER = "selection_diagonal_hessian_importance_weighted_reround_v1"
LDLQ_LEVER = "selection_hin_blockldlq_feedback_fp32_v1"
GLM52_REAP_MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
GLM52_REAP_PROFILE = "glm52-reap-504b-v2"
GLM52_REAP_REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
GLM52_REAP_CONFIG_SHA256 = "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
GLM52_REAP_INDEX_SHA256 = "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
GROUP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_SPLIT_EVIDENCE = {
    "split": "selection",
    "prompt_count": 22,
    "position_count": 255,
    "holdout_used_for_tuning": False,
    "report_used_for_tuning": False,
}
STATS_METHOD = "selection_only_diagonal_hessian_second_moments_v1"
GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES = 112_000_000_000
PINNED_ACCEPTED_COMPOSITE_IDENTITY_SHA256 = (
    "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
)
PINNED_ACCEPTED_SEED_MANIFEST_SHA256 = (
    "ba1d3135ef8901f1a69ead28b5f9d330ef40d015ac31fdb4d2dcfe678c41f0a4"
)
PINNED_ACCEPTED_ROUTED_GROUP_SET_SHA256 = (
    "bb65cf9a0d4a78310eb46c25e07b6492e6f3a5b18b303a7d68a8f0c73819d3fe"
)
PINNED_ACCEPTED_NON_VQ_PACKAGE_SET_SHA256 = (
    "2719f13a66313b5b8acc4c053914cdbfa10fdbf103a62b924495eecddbcea2ed"
)
PINNED_ACCEPTED_NON_VQ_MANIFEST_SHA256 = (
    "5113750fdaf009b2174a2772dd2e3bccf490cfff354f082e5e0ad8af8771f8c8"
)
PINNED_ACCEPTED_NON_VQ_EVIDENCE_SHA256 = (
    "ccbedd87f72f032bd86fd9d5a89fb75641595f30cf2a80fd7c84636d594e80d2"
)
PINNED_ACCEPTED_WHOLE_TENSOR_PAYLOAD_BYTES = 98_433_923_808
PINNED_ACCEPTED_ROUTED_TENSOR_PAYLOAD_BYTES = 61_312_204_800
PINNED_ACCEPTED_NON_ROUTED_TENSOR_PAYLOAD_BYTES = 37_121_719_008
PINNED_ACCEPTED_MAIN_NON_ROUTED_TENSOR_PAYLOAD_BYTES = 37_121_488_608
PINNED_ACCEPTED_NESTED_NON_VQ_TENSOR_PAYLOAD_BYTES = 37_121_488_608
PINNED_ACCEPTED_GROUP_COUNT = 225
PINNED_ACCEPTED_NON_VQ_TENSOR_COUNT = 1_194
PINNED_ACCEPTED_NON_VQ_SHARD_COUNT = 9
PINNED_ACCEPTED_NON_VQ_PARAMETER_COUNT = 18_560_731_704
PINNED_ACCEPTED_ROUTED_TENSOR_COUNT = 151_200
PINNED_ACCEPTED_AUDIT_CHECKS = {
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
PINNED_ACCEPTED_NON_VQ_AUDIT_CHECKS = {
    "accounting": True,
    "index": True,
    "lineage": True,
    "payload_hashes": True,
    "physical_extents": True,
    "shard_hashes": True,
    "strict_tree": True,
    "tensor_inventory": True,
}


def _load_direct_module(name: str, relative_path: str) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load quality lever module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ebss_module() -> Any:
    return _load_direct_module("_glm52_materializer_ebss", "quality/ebss.py")


def _rotation_module() -> Any:
    return _load_direct_module(
        "_glm52_materializer_rotation_search", "quality/rotation_search.py"
    )


def _ldlq_module() -> Any:
    cached = sys.modules.get("_glm52_materializer_ldlq_feedback")
    if cached is not None:
        return cached
    quality_root = Path(__file__).resolve().parents[1] / "quality"
    managed_names = (
        "mlx",
        "mlx.core",
        "keep.quality",
        "keep.quality.kronecker_hessian",
    )
    previous = {name: sys.modules.get(name) for name in managed_names}
    quality_package = ModuleType("keep.quality")
    quality_package.__path__ = [str(quality_root)]
    mlx_module = ModuleType("mlx")
    mlx_core_module = ModuleType("mlx.core")
    mlx_module.core = mlx_core_module
    try:
        sys.modules.update(
            {
                "mlx": mlx_module,
                "mlx.core": mlx_core_module,
                "keep.quality": quality_package,
            }
        )
        _load_direct_module(
            "keep.quality.kronecker_hessian",
            "quality/kronecker_hessian.py",
        )
        return _load_direct_module(
            "_glm52_materializer_ldlq_feedback", "quality/ldlq_feedback.py"
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def canonical_multiplier_string(multipliers: tuple[float, ...]) -> str:
    return str(_ebss_module().canonical_multiplier_string(multipliers))


def search_group_scales(*args: Any, **kwargs: Any) -> Any:
    return _ebss_module().search_group_scales(*args, **kwargs)


def ldlq_reassign_codes(*args: Any, **kwargs: Any) -> Any:
    return _ldlq_module().ldlq_reassign_codes(*args, **kwargs)


def generate_rht_signs(*args: Any, **kwargs: Any) -> Any:
    return _rotation_module().generate_rht_signs(*args, **kwargs)


def rotate_weight_rht(*args: Any, **kwargs: Any) -> Any:
    return _rotation_module().rotate_weight_rht(*args, **kwargs)


def select_projection_rotation(*args: Any, **kwargs: Any) -> Any:
    return _rotation_module().select_projection_rotation(*args, **kwargs)
STATS_ENTRY_KEYS = {
    "layer",
    "projection",
    "expert",
    "input_dim",
    "route_count",
    "total_route_count",
    "route_frequency",
    "router_score_sum",
    "prompt_ids",
    "sample_count",
    "prompt_count",
    "position_count",
    "path",
    "sha256",
}
SEED_MANIFEST_KEYS = {
    "artifact_total_bytes", "code_bits", "codebook_name", "codebook_sha256",
    "config_sha256", "dense_checkpoint_written", "full_group_coverage", "group_size",
    "groups", "index_sha256", "materialization_blocked", "materialization_blockers",
    "materialization_scope", "materialization_status", "model_id",
    "peak_decoded_expert_bytes", "planned_vq_groups", "profile", "ready_vq_groups",
    "record_type", "scale_estimator", "schema_version", "selected_group_keys",
    "selected_vq_groups", "skipped_vq_groups", "source_bundle_members",
    "source_bundles", "source_decoder", "source_revision", "source_weight_encoding",
    "working_set_policy",
}
SEED_GROUP_RECORD_KEYS = {
    "artifact_bytes", "artifact_path", "artifact_sha256", "codes_dtype", "codes_name",
    "codes_shape", "cross_shard_bundle_count", "decoded_expert_bytes", "expert_count",
    "layer", "projection", "scales_dtype", "scales_name", "scales_shape",
    "source_bundle_members", "source_bundles", "source_shards", "status",
}


@dataclass(frozen=True)
class ImportanceAwareQuantizedWeight:
    codes: np.ndarray
    scales: np.ndarray
    codebook: np.ndarray
    group_size: int
    code_bits: int = 8
    stats: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryPolicy:
    recovery_lever: str = RECOVERY_LEVER
    scale_search_multipliers: str | None = None
    rotation_rht_seed: str | None = None

    def __post_init__(self) -> None:
        if self.recovery_lever not in {RECOVERY_LEVER, LDLQ_LEVER}:
            raise ValueError(f"unsupported recovery lever: {self.recovery_lever}")
        if self.scale_search_multipliers is not None:
            try:
                values = tuple(
                    float(value) for value in self.scale_search_multipliers.split(",")
                )
                canonical = canonical_multiplier_string(values)
            except (TypeError, ValueError) as error:
                raise ValueError("scale_search_multipliers must be a canonical string") from error
            if canonical != self.scale_search_multipliers:
                raise ValueError(
                    "scale_search_multipliers must use the canonical ordered string"
                )
        if self.rotation_rht_seed is not None and (
            not isinstance(self.rotation_rht_seed, str) or not self.rotation_rht_seed
        ):
            raise ValueError("rotation_rht_seed must be a nonempty canonical string")

    @property
    def multipliers(self) -> tuple[float, ...] | None:
        if self.scale_search_multipliers is None:
            return None
        return tuple(float(value) for value in self.scale_search_multipliers.split(","))


def _recovery_policy_declarations(
    policy: RecoveryPolicy,
    *,
    scale_refinement_iterations: int = 3,
) -> dict[str, str]:
    if policy.recovery_lever == LDLQ_LEVER:
        scale_estimator = (
            "selection_diagonal_hessian_ebss_v1"
            if policy.scale_search_multipliers is not None
            else "max_abs_fixed_for_ldlq"
        )
        rounding_objective = "selection_hin_weighted_squared_error_with_blockldlq_feedback"
    else:
        scale_estimator = (
            "selection_diagonal_hessian_ebss_v1"
            if policy.scale_search_multipliers is not None
            else "importance_weighted_least_squares"
        )
        rounding_objective = "selection_diagonal_hessian_weighted_squared_error"
    declarations = {
        "scale_estimator": scale_estimator,
        "recovery_lever": policy.recovery_lever,
        "rounding_objective": rounding_objective,
    }
    if policy.scale_search_multipliers is not None:
        declarations.update(
            {
                "scale_search_multipliers": policy.scale_search_multipliers,
                "scale_search_objective": (
                    "selection_diagonal_hessian_weighted_squared_error"
                ),
                "scale_refinement_iterations": str(scale_refinement_iterations),
            }
        )
    if policy.recovery_lever == LDLQ_LEVER:
        declarations["ldlq_refinement_sweeps"] = "1"
    if policy.rotation_rht_seed is not None:
        declarations.update(
            {
                "rotation_rht_seed": policy.rotation_rht_seed,
                "rotation_kind": "selection_guided_rht_signs_v1",
            }
        )
    return declarations


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
class _AuthenticatedBytes:
    payload: bytes
    sha256: str
    identity: _FileIdentity


@dataclass(frozen=True)
class _ResumeTargetSnapshot:
    path: Path
    identity: _FileIdentity
    sha256: str
    size: int


@dataclass(frozen=True)
class _SourceIndex:
    metadata: dict[str, object]
    weight_map: dict[str, str]


@dataclass
class _SeedGroupSnapshots:
    paths: dict[tuple[int, str], Path]
    temporary_dir: tempfile.TemporaryDirectory[str]

    def close(self) -> None:
        self.temporary_dir.cleanup()


@dataclass(frozen=True)
class AuthenticatedSeed:
    kind: str
    root: Path
    artifact_root: Path
    records: Mapping[tuple[int, str], Mapping[str, object]]
    manifest_sha256: str
    candidate_identity_sha256: str | None
    audit_sha256: str | None
    source_lineage: Mapping[str, str]
    layer_code_bits: Mapping[int, int]
    manifest_path: Path
    audit_path: Path | None = None

    def verify_current_identity(
        self, groups: Sequence[tuple[int, str]] | None = None
    ) -> None:
        manifest = _read_stable_bytes(self.manifest_path, label="seed manifest identity")
        if manifest.sha256 != self.manifest_sha256:
            raise ValueError("seed manifest identity changed after authentication")
        if self.audit_path is not None:
            audit = _read_stable_bytes(self.audit_path, label="seed audit identity")
            if audit.sha256 != self.audit_sha256:
                raise ValueError("seed audit identity changed after authentication")
        keys = tuple(self.records) if groups is None else tuple(groups)
        for key in keys:
            record = self.records[key]
            path = Path(str(record["resolved_path"]))
            authenticated = AuthenticatedFile.open(path, label=f"seed group identity {key}")
            try:
                if authenticated.size != record["artifact_bytes"]:
                    raise ValueError(f"seed group {key} size changed after authentication")
                if authenticated.sha256 != record["artifact_sha256"]:
                    raise ValueError(f"seed group {key} SHA-256 changed after authentication")
            finally:
                authenticated.close()


@dataclass(frozen=True)
class AcceptedCompositeAuditAuthority:
    path: str
    sha256: str
    accepted_composite_identity_sha256: str
    actual_whole_model_tensor_payload_bytes: int
    actual_routed_payload_bytes: int
    non_routed_tensor_payload_bytes: int


class RecoveryHeavyLockBusy(RuntimeError):
    """The repository heavy lock is owned; recovery must exit, never queue."""


def _validate_seed_namespace(namespace: argparse.Namespace) -> None:
    if getattr(namespace, "command", None) not in {None, "rematerialize", "run"}:
        return
    if not hasattr(namespace, "seed_artifact_dir"):
        return
    classic = (
        getattr(namespace, "seed_artifact_dir", None),
        getattr(namespace, "expected_seed_manifest_sha256", None),
    )
    recovery = (
        getattr(namespace, "seed_recovery_artifact_dir", None),
        getattr(namespace, "seed_recovery_audit_json", None),
        getattr(namespace, "expected_seed_recovery_manifest_sha256", None),
        getattr(namespace, "expected_seed_recovery_audit_sha256", None),
    )
    if all(value is not None for value in classic) and not any(
        value is not None for value in recovery
    ):
        return
    if all(value is not None for value in recovery) and not any(
        value is not None for value in classic
    ):
        return
    raise ValueError(
        "exactly one complete classic or recovery seed flag family is required"
    )


class _SeedFamilyArgumentParser(argparse.ArgumentParser):
    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        try:
            _validate_seed_namespace(parsed)
        except ValueError as error:
            self.error(str(error))
        return parsed


@contextmanager
def _cooperating_file_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path)
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    try:
        opened = os.fstat(descriptor)
        visible = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or opened.st_nlink == 0
            or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
        ):
            raise ValueError(f"lock path must be a regular file: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN} or isinstance(
                error, BlockingIOError
            ):
                raise RecoveryHeavyLockBusy(
                    f"recovery heavy lock is already held: {lock_path}"
                ) from error
            raise
        opened = os.fstat(descriptor)
        visible = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or opened.st_nlink == 0
            or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
        ):
            raise ValueError(
                f"lock path changed while acquiring authority: {lock_path}"
            )
        acquired = True
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_stable_bytes(
    path: Path,
    *,
    label: str,
    allow_snapshot_symlink: bool = False,
) -> _AuthenticatedBytes:
    authenticated = AuthenticatedFile.open(
        path,
        label=label,
        allow_resolved_symlink=allow_snapshot_symlink,
    )
    try:
        return _AuthenticatedBytes(
            authenticated.bytes,
            authenticated.sha256,
            _FileIdentity(
                authenticated.device,
                authenticated.inode,
                authenticated.size,
                authenticated.identity.mtime_ns,
            ),
        )
    finally:
        authenticated.close()


def _replacement_destination_identity(
    path: Path,
    *,
    label: str,
) -> ReplacementAuthority | None:
    try:
        authenticated = AuthenticatedFile.open(path, label=label)
    except ValueError:
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        raise
    try:
        return ReplacementAuthority.from_authenticated(authenticated)
    finally:
        authenticated.close()


def _archive_displaced_publication(
    published: PublishedArtifact,
    *,
    output_root: Path,
) -> Path | None:
    if published.displaced_path is None:
        return None
    archive_root = output_root / ".publication-forensics"
    archive_root.mkdir(mode=0o700, exist_ok=True)
    return archive_displaced_artifact(published, archive_root)


@contextmanager
def _authenticated_resume_target_snapshot(
    path: Path,
    *,
    label: str,
) -> Iterator[_ResumeTargetSnapshot]:
    authenticated = AuthenticatedFile.open(path, label=label)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix="glm52-recovery-resume-")
    ).resolve(strict=True)
    snapshot_path = temporary_dir / "authenticated-target.safetensors"
    try:
        clone_or_copy_authenticated(
            authenticated,
            snapshot_path,
            allow_copy_fallback=True,
            required_free_bytes=authenticated.size,
        )
        identity = _FileIdentity(
            authenticated.device,
            authenticated.inode,
            authenticated.size,
            authenticated.identity.mtime_ns,
        )
        yield _ResumeTargetSnapshot(
            path=snapshot_path,
            identity=identity,
            sha256=authenticated.sha256,
            size=identity.size,
        )
        authenticated.verify_visible()
    finally:
        authenticated.close()
        for child in temporary_dir.iterdir() if temporary_dir.exists() else ():
            child.unlink(missing_ok=True)
        temporary_dir.rmdir() if temporary_dir.exists() else None


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _load_source_index(payload: bytes) -> _SourceIndex:
    value = _load_json_bytes(payload, label="source index")
    weight_map = value.get("weight_map")
    metadata = value.get("metadata") or {}
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise ValueError("source index must contain weight_map and object metadata")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise ValueError("source index weight_map must map strings to strings")
    return _SourceIndex(dict(metadata), dict(weight_map))


def _load_source_auth_module() -> Any:
    name = "keep.quality.glm52_teacher_cache_producer"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "quality/glm52_teacher_cache_producer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 source authentication module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _authenticate_accepted_composite_audit(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_source_lineage: Mapping[str, str],
) -> AcceptedCompositeAuditAuthority:
    if not _is_sha256(expected_sha256):
        raise ValueError("expected composite audit SHA-256 must be lowercase hexadecimal")
    authenticated = _read_stable_bytes(Path(path), label="accepted composite audit")
    if authenticated.sha256 != expected_sha256:
        raise ValueError(
            "accepted composite audit SHA-256 does not match external pinned authority"
        )
    payload = _load_json_bytes(authenticated.payload, label="accepted composite audit")
    canonical_groups = [
        f"{layer}:{projection}"
        for layer in range(3, 78)
        for projection in GROUP_PROJECTIONS
    ]
    pinned = {
        "manifest_sha256": PINNED_ACCEPTED_SEED_MANIFEST_SHA256,
        "group_set_sha256": PINNED_ACCEPTED_ROUTED_GROUP_SET_SHA256,
        "non_vq_evidence_sha256": PINNED_ACCEPTED_NON_VQ_EVIDENCE_SHA256,
        "expected_group_keys": canonical_groups,
        "ready_group_keys": canonical_groups,
        "checks": PINNED_ACCEPTED_AUDIT_CHECKS,
        "byte_identity_verified": True,
        "full_routed_artifact_ready": True,
        "main_routed_tensor_count": PINNED_ACCEPTED_ROUTED_TENSOR_COUNT,
        "main_non_routed_tensor_count": PINNED_ACCEPTED_NON_VQ_TENSOR_COUNT,
        "actual_whole_model_tensor_payload_bytes": (
            PINNED_ACCEPTED_WHOLE_TENSOR_PAYLOAD_BYTES
        ),
        "actual_routed_payload_bytes": PINNED_ACCEPTED_ROUTED_TENSOR_PAYLOAD_BYTES,
        "main_non_routed_tensor_payload_bytes": (
            PINNED_ACCEPTED_MAIN_NON_ROUTED_TENSOR_PAYLOAD_BYTES
        ),
    }
    for field, expected in pinned.items():
        if payload.get(field) != expected:
            raise ValueError(
                "accepted composite audit does not match the repository-pinned "
                f"accepted baseline/budget: {field}"
            )
    nested = payload.get("non_vq_package_audit")
    nested_pinned = {
        "audit_pass": True,
        "checks": PINNED_ACCEPTED_NON_VQ_AUDIT_CHECKS,
        "manifest_sha256": PINNED_ACCEPTED_NON_VQ_MANIFEST_SHA256,
        "package_set_sha256": PINNED_ACCEPTED_NON_VQ_PACKAGE_SET_SHA256,
        "retained_tensor_count": PINNED_ACCEPTED_NON_VQ_TENSOR_COUNT,
        "shard_count": PINNED_ACCEPTED_NON_VQ_SHARD_COUNT,
        "parameter_count": PINNED_ACCEPTED_NON_VQ_PARAMETER_COUNT,
        "tensor_payload_bytes": PINNED_ACCEPTED_NESTED_NON_VQ_TENSOR_PAYLOAD_BYTES,
    }
    if not isinstance(nested, Mapping) or any(
        nested.get(field) != expected for field, expected in nested_pinned.items()
    ):
        raise ValueError(
            "accepted composite audit does not match the repository-pinned accepted "
            "baseline: non_vq_package_audit"
        )
    for field in ("audit_pass", "artifact_integrity_pass"):
        if payload.get(field) is not True:
            raise ValueError(f"accepted composite audit {field} must be true")
    for audit_field, lineage_field in {
        "model_id": "source_model_id",
        "source_revision": "source_revision",
        "config_sha256": "source_config_sha256",
        "index_sha256": "source_index_sha256",
    }.items():
        if payload.get(audit_field) != expected_source_lineage[lineage_field]:
            raise ValueError(
                f"accepted composite audit lineage {audit_field} does not match "
                "the authenticated recovery source"
            )
    whole = payload.get("actual_whole_model_tensor_payload_bytes")
    routed = payload.get("actual_routed_payload_bytes")
    if type(whole) is not int or whole <= 0:
        raise ValueError(
            "accepted composite audit whole-model tensor payload must be a positive integer"
        )
    if type(routed) is not int or routed <= 0:
        raise ValueError(
            "accepted composite audit routed tensor payload must be a positive integer"
        )
    non_routed = payload.get("main_non_routed_tensor_payload_bytes")
    if type(non_routed) is not int or non_routed <= 0:
        raise ValueError(
            "accepted composite audit main non-routed tensor payload must be positive"
        )
    if whole > GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES:
        raise ValueError(
            "accepted composite audit tensor payload exceeds the canonical "
            f"{GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES} byte budget"
        )
    return AcceptedCompositeAuditAuthority(
        path=str(Path(path)),
        sha256=expected_sha256,
        accepted_composite_identity_sha256=(
            PINNED_ACCEPTED_COMPOSITE_IDENTITY_SHA256
        ),
        actual_whole_model_tensor_payload_bytes=whole,
        actual_routed_payload_bytes=routed,
        non_routed_tensor_payload_bytes=non_routed,
    )


def _canonical_seed_groups() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (
            layer,
            projection,
            f"layer-{layer:05d}-{projection}.safetensors",
        )
        for layer in range(3, 78)
        for projection in GROUP_PROJECTIONS
    )


def _validate_seed_manifest(
    manifest: Mapping[str, object],
    *,
    expected_source_lineage: Mapping[str, str],
) -> dict[tuple[int, str], Mapping[str, object]]:
    if set(manifest) != SEED_MANIFEST_KEYS:
        raise ValueError("seed manifest keys do not match the exact accepted schema")
    expected_header = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "profile": expected_source_lineage["source_profile"],
        "model_id": expected_source_lineage["source_model_id"],
        "source_revision": expected_source_lineage["source_revision"],
        "config_sha256": expected_source_lineage["source_config_sha256"],
        "index_sha256": expected_source_lineage["source_index_sha256"],
        "planned_vq_groups": 225,
        "ready_vq_groups": 225,
    }
    for name, expected in expected_header.items():
        if manifest.get(name) != expected:
            raise ValueError(f"seed manifest {name} does not match accepted authority")
    raw_records = manifest.get("groups")
    canonical = _canonical_seed_groups()
    if not isinstance(raw_records, list) or len(raw_records) != len(canonical):
        raise ValueError("seed manifest groups must contain exactly 225 records")
    records: dict[tuple[int, str], Mapping[str, object]] = {}
    for raw_record, (layer, projection, filename) in zip(raw_records, canonical, strict=True):
        key = (layer, projection)
        if not isinstance(raw_record, Mapping) or set(raw_record) != SEED_GROUP_RECORD_KEYS:
            raise ValueError(f"seed manifest group {key} keys do not match the exact schema")
        if (
            raw_record.get("layer") != layer
            or raw_record.get("projection") != projection
            or Path(str(raw_record.get("artifact_path"))).name != filename
        ):
            raise ValueError(f"seed manifest group {key} is not in canonical order")
        if type(raw_record.get("artifact_bytes")) is not int or raw_record["artifact_bytes"] <= 0:
            raise ValueError(f"seed manifest group {key} has invalid artifact size")
        if not _is_sha256(raw_record.get("artifact_sha256")):
            raise ValueError(f"seed manifest group {key} has invalid artifact SHA-256")
        records[key] = {**raw_record, "artifact_path": filename}
    return records


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_recovery_seed_group_link(
    link: Path,
    *,
    filename: str,
    accepted_seed_root: Path,
    recovered_root: Path,
) -> tuple[str, Path]:
    if not link.is_symlink():
        raise ValueError(f"recovery seed group {filename} must be a relative symlink")
    raw_target = os.readlink(link)
    if os.path.isabs(raw_target):
        raise ValueError(f"recovery seed group {filename} uses an absolute symlink")
    lexical = Path(os.path.abspath(link.parent / raw_target))
    try:
        resolved = link.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ValueError(f"recovery seed group {filename} has an invalid target") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"recovery seed group {filename} target must be a regular file")
    if not (
        _path_is_within(lexical, accepted_seed_root)
        or _path_is_within(lexical, recovered_root)
    ):
        raise ValueError(
            f"recovery seed group {filename} escapes authenticated seed roots"
        )
    accepted_target = accepted_seed_root / filename
    recovered_target = recovered_root / filename
    if resolved == recovered_target:
        return "replacement", resolved
    if resolved == accepted_target:
        return "inherited", resolved
    raise ValueError(f"recovery seed group {filename} was retargeted")


def _authenticate_recovery_seed(
    *,
    recovery_root: str | Path,
    audit_path: str | Path,
    expected_manifest_sha256: str,
    expected_audit_sha256: str,
    expected_source_lineage: Mapping[str, str] | None = None,
    expected_stats_manifest_sha256: str | None = None,
    expected_composite_audit_sha256: str | None = None,
) -> AuthenticatedSeed:
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("expected recovery seed manifest SHA-256 is invalid")
    if not _is_sha256(expected_audit_sha256):
        raise ValueError("expected recovery seed audit SHA-256 is invalid")
    root = Path(recovery_root).resolve(strict=True)
    manifest_path = root / "conversion-manifest.json"
    authenticated_manifest = _read_stable_bytes(
        manifest_path, label="recovery seed manifest"
    )
    if authenticated_manifest.sha256 != expected_manifest_sha256:
        raise ValueError("recovery seed manifest SHA-256 does not match external authority")
    manifest = _load_json_bytes(
        authenticated_manifest.payload, label="recovery seed manifest"
    )
    if (
        manifest.get("schema_version") != 2
        or manifest.get("record_type") != "glm52_recovery_conversion_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("resumable") is not True
    ):
        raise ValueError("recovery seed manifest is not a complete resumable schema-v2 artifact")
    body = dict(manifest)
    recorded_body_sha256 = body.pop("manifest_body_sha256", None)
    if not _is_sha256(recorded_body_sha256) or recorded_body_sha256 != _canonical_sha256(body):
        raise ValueError("recovery seed manifest body SHA-256 mismatch")

    authenticated_audit = _read_stable_bytes(Path(audit_path), label="recovery seed audit")
    if authenticated_audit.sha256 != expected_audit_sha256:
        raise ValueError("recovery seed audit SHA-256 does not match external authority")
    audit = _load_json_bytes(authenticated_audit.payload, label="recovery seed audit")
    if audit.get("audit_pass") is not True or not isinstance(audit.get("checks"), Mapping):
        raise ValueError("recovery seed audit did not pass")
    if not all(value is True for value in audit["checks"].values()):
        raise ValueError("recovery seed audit contains a failed check")
    if audit.get("raw_recovery_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("recovery seed audit does not bind the raw recovery manifest")
    if audit.get("manifest_body_sha256") != recorded_body_sha256:
        raise ValueError("recovery seed audit does not bind the manifest body")
    candidate_identity = audit.get("candidate_identity_sha256")
    if not _is_sha256(candidate_identity):
        raise ValueError("recovery seed audit candidate identity is invalid")
    if Path(str(audit.get("recovery_dir"))).resolve() != root:
        raise ValueError("recovery seed audit recovery_dir does not match the requested artifact")

    mixed = manifest.get("mixed_artifact")
    if not isinstance(mixed, Mapping):
        raise ValueError("recovery seed manifest mixed_artifact is invalid")
    artifact_root = Path(str(mixed.get("output_dir"))).resolve(strict=True)
    recovered_root = Path(str(mixed.get("recovered_groups_dir"))).resolve(strict=True)
    accepted_seed_root = Path(str(mixed.get("seed_artifact_dir"))).resolve()
    if artifact_root != (root / "artifact").resolve(strict=True):
        raise ValueError("recovery seed artifact root does not match the requested artifact")
    raw_audit_groups = audit.get("groups")
    canonical = _canonical_seed_groups()
    if (
        not isinstance(raw_audit_groups, list)
        or audit.get("group_count") != len(canonical)
        or len(raw_audit_groups) != len(canonical)
    ):
        raise ValueError("recovery seed audit does not prove every canonical group")
    audit_groups = {
        raw.get("group_key"): raw
        for raw in raw_audit_groups
        if isinstance(raw, Mapping)
    }
    if len(audit_groups) != len(canonical):
        raise ValueError("recovery seed audit group identities are invalid or duplicated")
    replacement_records = manifest.get("groups")
    if not isinstance(replacement_records, list):
        raise ValueError("recovery seed manifest groups must be a list")
    manifest_by_key = {
        (raw.get("layer"), raw.get("projection")): raw
        for raw in replacement_records
        if isinstance(raw, Mapping)
    }
    records: dict[tuple[int, str], Mapping[str, object]] = {}
    layer_code_bits: dict[int, int] = {}
    identity_groups: list[dict[str, object]] = []
    for layer, projection, filename in canonical:
        key = (layer, projection)
        text_key = f"{layer}:{projection}"
        raw = audit_groups.get(text_key)
        if not isinstance(raw, Mapping) or raw.get("filename") != filename:
            raise ValueError(f"recovery seed audit group {text_key} is missing or noncanonical")
        classification, resolved = _resolve_recovery_seed_group_link(
            artifact_root / filename,
            filename=filename,
            accepted_seed_root=accepted_seed_root,
            recovered_root=recovered_root,
        )
        if raw.get("classification") != classification:
            raise ValueError(f"recovery seed group {text_key} classification drifted")
        authenticated_group = AuthenticatedFile.open(
            resolved, label=f"recovery seed group {text_key}"
        )
        try:
            if authenticated_group.size != raw.get("artifact_bytes"):
                raise ValueError(f"recovery seed group {text_key} size mismatch")
            if authenticated_group.sha256 != raw.get("artifact_sha256"):
                raise ValueError(f"recovery seed group {text_key} SHA-256 mismatch")
        finally:
            authenticated_group.close()
        bits = raw.get("code_bits")
        if bits not in (8, 16):
            raise ValueError(f"recovery seed group {text_key} code_bits are invalid")
        prior_bits = layer_code_bits.setdefault(layer, int(bits))
        if prior_bits != bits:
            raise ValueError(f"recovery seed layer {layer} has mixed projection rates")
        manifest_record = manifest_by_key.get(key)
        if classification == "replacement":
            if not isinstance(manifest_record, Mapping):
                raise ValueError(f"recovery seed replacement group {text_key} is absent from manifest")
            if (
                manifest_record.get("artifact_sha256") != raw.get("artifact_sha256")
                or manifest_record.get("artifact_bytes") != raw.get("artifact_bytes")
            ):
                raise ValueError(f"recovery seed group {text_key} manifest binding drifted")
        elif manifest_record is not None:
            raise ValueError(f"recovery seed inherited group {text_key} masquerades as replacement")
        records[key] = {
            "artifact_path": filename,
            "resolved_path": str(resolved),
            "artifact_bytes": raw["artifact_bytes"],
            "artifact_sha256": raw["artifact_sha256"],
            "classification": classification,
            "code_bits": int(bits),
            "codes_dtype": raw.get("codes_dtype"),
            "codebook_name": raw.get("codebook_name"),
            "codebook_sha256": raw.get("codebook_sha256"),
        }
        identity_groups.append(
            {
                "group_key": text_key,
                "classification": classification,
                "artifact_sha256": raw["artifact_sha256"],
                "tensor_payload_bytes": raw.get("tensor_payload_bytes"),
                "resolved_policy": {
                    "quant_method": "mlx_vq_e8",
                    "version": 1,
                    "code_bits": int(bits),
                    "codes_dtype": raw.get("codes_dtype"),
                    "group_size": raw.get("group_size"),
                    "codebook_name": raw.get("codebook_name"),
                    "codebook_sha256": raw.get("codebook_sha256"),
                    "tensor_payload_bytes": raw.get("tensor_payload_bytes"),
                    "recovery_provenance": (
                        manifest_record.get("lever_provenance", {})
                        if classification == "replacement"
                        else {}
                    ),
                },
                "code_bits": int(bits),
                "codes_dtype": raw.get("codes_dtype"),
                "codebook_name": raw.get("codebook_name"),
                "codebook_sha256": raw.get("codebook_sha256"),
            }
        )
    expected_replacements = sum(
        record["classification"] == "replacement" for record in records.values()
    )
    if (
        audit.get("replacement_group_count") != expected_replacements
        or audit.get("inherited_group_count") != len(canonical) - expected_replacements
    ):
        raise ValueError("recovery seed audit group accounting drifted")
    source_lineage = manifest.get("source_lineage")
    if not isinstance(source_lineage, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in source_lineage.items()
    ):
        raise ValueError("recovery seed source lineage is invalid")
    if expected_source_lineage is not None and dict(source_lineage) != dict(
        expected_source_lineage
    ):
        raise ValueError("recovery seed source lineage is incompatible with the child run")
    if (
        expected_stats_manifest_sha256 is not None
        and manifest.get("stats_manifest_sha256") != expected_stats_manifest_sha256
    ):
        raise ValueError("recovery seed stats authority is incompatible with the child run")
    if (
        expected_composite_audit_sha256 is not None
        and manifest.get("accepted_composite_audit_sha256")
        != expected_composite_audit_sha256
    ):
        raise ValueError("recovery seed composite authority is incompatible with the child run")
    if audit.get("seed_manifest_sha256") != PINNED_ACCEPTED_SEED_MANIFEST_SHA256:
        raise ValueError("recovery seed audit does not inherit the accepted baseline seed")
    if (
        audit.get("accepted_baseline_composite_identity_sha256")
        != PINNED_ACCEPTED_COMPOSITE_IDENTITY_SHA256
    ):
        raise ValueError("recovery seed audit accepted baseline identity drifted")
    candidate_body = {
        "schema_version": 2,
        "source_lineage": dict(source_lineage),
        "seed_manifest_sha256": audit.get("seed_manifest_sha256"),
        "accepted_baseline_composite_identity_sha256": audit.get(
            "accepted_baseline_composite_identity_sha256"
        ),
        "accepted_composite_audit_sha256": audit.get(
            "accepted_composite_audit_sha256"
        ),
        "groups": identity_groups,
        "logical_whole_model_tensor_payload_bytes": audit.get(
            "logical_whole_model_tensor_payload_bytes"
        ),
        "rate_policy": manifest.get("rate_policy"),
        "routed_codes_scales_bytes": audit.get("routed_codes_scales_bytes"),
        "routed_codebook_bytes": audit.get("routed_codebook_bytes"),
        "main_non_routed_tensor_payload_bytes": audit.get(
            "main_non_routed_tensor_payload_bytes"
        ),
    }
    if _canonical_sha256(candidate_body) != candidate_identity:
        raise ValueError("recovery seed audit candidate identity does not match the artifact")
    authenticated_seed = AuthenticatedSeed(
        kind="audited_recovery",
        root=root,
        artifact_root=artifact_root,
        records=records,
        manifest_sha256=expected_manifest_sha256,
        candidate_identity_sha256=str(candidate_identity),
        audit_sha256=expected_audit_sha256,
        source_lineage=dict(source_lineage),
        layer_code_bits=dict(sorted(layer_code_bits.items())),
        manifest_path=manifest_path,
        audit_path=Path(audit_path),
    )
    authenticated_seed.verify_current_identity()
    return authenticated_seed


def _authenticate_baseline_seed(
    *,
    seed_root: str | Path,
    expected_manifest_sha256: str,
    expected_source_lineage: Mapping[str, str],
) -> AuthenticatedSeed:
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("expected seed manifest SHA-256 must be lowercase hexadecimal")
    if expected_manifest_sha256 != PINNED_ACCEPTED_SEED_MANIFEST_SHA256:
        raise ValueError(
            "expected seed manifest SHA-256 does not match the repository-pinned accepted baseline"
        )
    root = Path(seed_root)
    manifest_path = root / "conversion-manifest.json"
    authenticated_manifest = _read_stable_bytes(
        manifest_path, label="accepted seed manifest"
    )
    if authenticated_manifest.sha256 != expected_manifest_sha256:
        raise ValueError("seed manifest SHA-256 does not match external accepted authority")
    manifest = _load_json_bytes(
        authenticated_manifest.payload, label="accepted seed manifest"
    )
    raw_records = _validate_seed_manifest(
        manifest, expected_source_lineage=expected_source_lineage
    )
    records = {
        key: {
            **record,
            "resolved_path": str(root / str(record["artifact_path"])),
            "classification": "accepted_baseline",
            "code_bits": 8,
        }
        for key, record in raw_records.items()
    }
    return AuthenticatedSeed(
        kind="accepted_baseline",
        root=root,
        artifact_root=root,
        records=records,
        manifest_sha256=expected_manifest_sha256,
        candidate_identity_sha256=None,
        audit_sha256=None,
        source_lineage=dict(expected_source_lineage),
        layer_code_bits={layer: 8 for layer in range(3, 78)},
        manifest_path=manifest_path,
    )


def _authenticate_seed(
    *,
    seed_artifact_dir: str | Path | None,
    expected_seed_manifest_sha256: str | None,
    seed_recovery_artifact_dir: str | Path | None,
    seed_recovery_audit_json: str | Path | None,
    expected_seed_recovery_manifest_sha256: str | None,
    expected_seed_recovery_audit_sha256: str | None,
    expected_source_lineage: Mapping[str, str],
    expected_stats_manifest_sha256: str,
    expected_composite_audit_sha256: str,
) -> AuthenticatedSeed:
    classic_values = (seed_artifact_dir, expected_seed_manifest_sha256)
    recovery_values = (
        seed_recovery_artifact_dir,
        seed_recovery_audit_json,
        expected_seed_recovery_manifest_sha256,
        expected_seed_recovery_audit_sha256,
    )
    if any(value is not None for value in classic_values) and any(
        value is not None for value in recovery_values
    ):
        raise ValueError("classic and recovery seed flag families are mutually exclusive")
    if all(value is not None for value in classic_values):
        return _authenticate_baseline_seed(
            seed_root=seed_artifact_dir,
            expected_manifest_sha256=str(expected_seed_manifest_sha256),
            expected_source_lineage=expected_source_lineage,
        )
    if all(value is not None for value in recovery_values):
        return _authenticate_recovery_seed(
            recovery_root=seed_recovery_artifact_dir,
            audit_path=seed_recovery_audit_json,
            expected_manifest_sha256=str(expected_seed_recovery_manifest_sha256),
            expected_audit_sha256=str(expected_seed_recovery_audit_sha256),
            expected_source_lineage=expected_source_lineage,
            expected_stats_manifest_sha256=expected_stats_manifest_sha256,
            expected_composite_audit_sha256=expected_composite_audit_sha256,
        )
    raise ValueError("exactly one complete classic or recovery seed flag family is required")


def _snapshot_seed_groups(
    *,
    seed_root: Path,
    records: Mapping[tuple[int, str], Mapping[str, object]],
    groups: Sequence[tuple[int, str]],
) -> _SeedGroupSnapshots:
    temporary_dir = tempfile.TemporaryDirectory(prefix="glm52-recovery-seed-")
    snapshot_root = Path(temporary_dir.name).resolve(strict=True)
    paths: dict[tuple[int, str], Path] = {}
    remaining_copy_bytes = sum(int(records[key]["artifact_bytes"]) for key in groups)
    try:
        for key in groups:
            record = records[key]
            source = Path(
                str(record.get("resolved_path", seed_root / str(record["artifact_path"])))
            )
            authenticated = AuthenticatedFile.open(source, label=f"seed group {key}")
            try:
                if authenticated.size != record["artifact_bytes"]:
                    raise ValueError(f"seed group {key} size does not match accepted seed manifest")
                if authenticated.sha256 != record["artifact_sha256"]:
                    raise ValueError(f"seed group {key} SHA-256 does not match accepted seed manifest")
                snapshot = snapshot_root / source.name
                clone_or_copy_authenticated(
                    authenticated,
                    snapshot,
                    allow_copy_fallback=True,
                    required_free_bytes=remaining_copy_bytes,
                )
                remaining_copy_bytes -= authenticated.size
                paths[key] = snapshot
            finally:
                authenticated.close()
    except BaseException:
        temporary_dir.cleanup()
        raise
    return _SeedGroupSnapshots(paths, temporary_dir)


def _validate_importance(weight: np.ndarray, importance: np.ndarray) -> np.ndarray:
    values = np.asarray(weight, dtype=np.float32)
    diagonal = np.asarray(importance, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 8:
        raise ValueError("weight must have shape [out, in] with in divisible by 8")
    if diagonal.shape != (values.shape[1],):
        raise ValueError(f"importance must have shape ({values.shape[1]},), found {diagonal.shape}")
    if not np.isfinite(values).all():
        raise ValueError("weight must be finite")
    if not np.isfinite(diagonal).all() or np.any(diagonal < 0):
        raise ValueError("importance must be finite and non-negative")
    if not np.any(diagonal > 0):
        diagonal = np.ones_like(diagonal)
    return diagonal


def _default_recovery_search_backend() -> str:
    """Resolve the E8P recovery search backend.

    Order: explicit ``GLM52_E8P_BACKEND`` env override, then the fused Metal
    kernel when a Metal GPU is available (byte-identical to the exhaustive
    NumPy reference and ~40x faster), else the deterministic NumPy reference.
    """
    override = os.environ.get("GLM52_E8P_BACKEND")
    if override:
        return override
    try:
        import mlx.core as _mx

        metal = getattr(_mx, "metal", None)
        if metal is not None and metal.is_available():
            return "metal"
    except Exception:
        pass
    return "numpy"


def quantize_weight_importance_aware(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    group_size: int = 512,
    code_bits: int = 8,
    iterations: int = 3,
    codeword_chunk_size: int = 8192,
    e8p_search_backend: str = "numpy",
    recovery_policy: RecoveryPolicy | None = None,
    h_in: np.ndarray | None = None,
) -> ImportanceAwareQuantizedWeight:
    """Jointly fit group scales and E8 codes under a diagonal Hessian.

    Each row/group alternates weighted nearest-code assignment with the exact
    weighted least-squares scale update.  The importance is shared across
    output rows but may differ by expert at the caller.
    """

    values = np.asarray(weight, dtype=np.float32)
    diagonal = _validate_importance(values, importance)
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    if group_size <= 0 or group_size % 8 or values.shape[1] % group_size:
        raise ValueError("group_size must be positive, divisible by 8, and divide the input dimension")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    policy = recovery_policy or RecoveryPolicy()
    if not isinstance(policy, RecoveryPolicy):
        raise TypeError("recovery_policy must be a RecoveryPolicy")

    out_dim, in_dim = values.shape
    group_count = in_dim // group_size
    codewords_per_group = group_size // 8
    grouped = values.reshape(out_dim, group_count, codewords_per_group, 8)
    grouped_hessian = diagonal.reshape(group_count, codewords_per_group, 8)
    codes_dtype = np.dtype(np.uint8 if code_bits == 8 else np.uint16)
    codes_by_group = np.empty(
        (out_dim, group_count, codewords_per_group), dtype=codes_dtype
    )
    table = (
        e8_1bit_grid().astype(np.float32)
        if code_bits == 8
        else e8p_full_grid().astype(np.float32)
    )
    divisor = float(np.max(np.abs(table)))
    scales = np.max(np.abs(grouped), axis=(2, 3)) / np.float32(divisor)
    scales = np.where(scales > 0, scales, np.float32(1.0)).astype(np.float32)
    quantization_stats: dict[str, object] = {}
    if policy.multipliers is not None:
        scales, searched_codes, ebss_stats = search_group_scales(
            values,
            diagonal,
            codebook=table,
            group_size=group_size,
            code_bits=code_bits,
            multipliers=policy.multipliers,
        )
        codes_by_group[...] = searched_codes.reshape(codes_by_group.shape)
        quantization_stats.update(ebss_stats)

    if policy.recovery_lever == LDLQ_LEVER:
        if code_bits != 16:
            raise ValueError("the LDLQ recovery lever requires the per-expert 16-bit path")
        if h_in is None:
            raise ValueError("the LDLQ recovery lever requires an authenticated full h_in")
        reassigned = ldlq_reassign_codes(
            values,
            scales,
            h_in,
            codebook=table,
            group_size=group_size,
            code_bits=code_bits,
            sweeps=1,
        )
        quantization_stats["ldlq"] = asdict(reassigned.stats)
        return ImportanceAwareQuantizedWeight(
            codes=reassigned.codes,
            scales=scales.astype(np.float16),
            codebook=e8p_packed_abs_grid(),
            group_size=group_size,
            code_bits=code_bits,
            stats=quantization_stats,
        )

    def assign_codes() -> None:
        if code_bits == 8:
            normalized = grouped / scales[:, :, None, None]
            per_row_diagonals = np.broadcast_to(
                grouped_hessian[None, ...], normalized.shape
            )
            flat_vectors = normalized.reshape(-1, 8)
            flat_diagonals = per_row_diagonals.reshape(-1, 8)
            codes_by_group[...] = nearest_e8_codes_diagonal_hessian(
                flat_vectors,
                flat_diagonals,
                backend=e8p_search_backend,
                index_dtype=np.uint8,
            ).reshape(out_dim, group_count, codewords_per_group).astype(codes_dtype)
            return

        if code_bits == 16:
            normalized = grouped / scales[:, :, None, None]
            per_row_diagonals = np.broadcast_to(
                grouped_hessian[None, ...], normalized.shape
            )
            flat_vectors = normalized.reshape(-1, 8)
            flat_diagonals = per_row_diagonals.reshape(-1, 8)
            codes_by_group[...] = nearest_e8p_codes_diagonal_hessian(
                flat_vectors,
                flat_diagonals,
                backend=e8p_search_backend,
            ).reshape(out_dim, group_count, codewords_per_group).astype(codes_dtype)
            return

    for _ in range(iterations):
        assign_codes()
        decoded = table[codes_by_group]
        numerator = np.sum(
            grouped_hessian[None, ...] * grouped * decoded,
            axis=(2, 3),
            dtype=np.float64,
        )
        denominator = np.sum(
            grouped_hessian[None, ...] * decoded * decoded,
            axis=(2, 3),
            dtype=np.float64,
        )
        updates = np.divide(
            numerator,
            denominator,
            out=scales.astype(np.float64),
            where=(denominator > 0) & (numerator > 0),
        )
        scales = updates.astype(np.float32)
    assign_codes()
    codes = codes_by_group.reshape(out_dim, in_dim // 8)

    return ImportanceAwareQuantizedWeight(
        codes=codes,
        scales=scales.astype(np.float16),
        codebook=e8_1bit_packed() if code_bits == 8 else e8p_packed_abs_grid(),
        group_size=group_size,
        code_bits=code_bits,
        stats=quantization_stats,
    )


def _group_names(layer: int, projection: str) -> tuple[str, str]:
    if layer < 3 or layer > 77:
        raise ValueError("GLM-5.2 routed recovery layer must be in [3, 77]")
    if projection not in GROUP_PROJECTIONS:
        raise ValueError(f"projection must be one of {GROUP_PROJECTIONS}")
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    return f"{prefix}.codes", f"{prefix}.scales"


def _lever_provenance(
    stats_manifest_sha256: str | None,
    *,
    source_lineage: Mapping[str, str] | None = None,
    source_verification: Mapping[str, object] | None = None,
    recovery_policy: RecoveryPolicy | None = None,
) -> dict[str, object]:
    if stats_manifest_sha256 is not None and not _is_sha256(stats_manifest_sha256):
        raise ValueError("stats_manifest_sha256 must be lowercase SHA-256 when provided")
    declarations = _recovery_policy_declarations(
        recovery_policy or RecoveryPolicy()
    )
    provenance: dict[str, object] = {
        "lever": declarations.pop("recovery_lever"),
        **declarations,
        "stats_manifest_sha256": stats_manifest_sha256,
        "holdout_used_for_tuning": False,
        "report_used_for_tuning": False,
    }
    if source_lineage is not None:
        provenance["source_lineage"] = dict(source_lineage)
    if source_verification is not None:
        provenance["source_verification"] = dict(source_verification)
    return provenance


def _recovery_quantization(
    seed_metadata: Mapping[str, str],
    *,
    group_size: int,
    code_bits: int = 8,
    recovery_policy: RecoveryPolicy | None = None,
) -> dict[str, object]:
    try:
        seed_quantization = json.loads(seed_metadata["quantization_config"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("seed group has no valid quantization_config metadata") from error
    seed_policy = seed_quantization.get("policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("seed group quantization policy must be an object")
    policy = {str(key): str(value) for key, value in seed_policy.items()}
    policy.update(_recovery_policy_declarations(recovery_policy or RecoveryPolicy()))
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(code_bits)
    return QuantizationConfig(
        default_code_bits=code_bits,
        default_group_size=group_size,
        codebook_name=codebook_name,
        codebook_sha256=codebook_sha256,
        policy=policy,
    ).to_json_dict()


def materialize_recovery_group(
    *,
    source_weights: np.ndarray,
    importance: np.ndarray,
    seed_group_path: str | Path,
    output_path: str | Path,
    layer: int,
    projection: str,
    group_size: int = 512,
    code_bits: int = 8,
    stats_manifest_sha256: str | None = None,
    source_lineage: Mapping[str, str] | None = None,
    source_verification: Mapping[str, object] | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    h_in: np.ndarray | None = None,
    e8p_search_backend: str | None = None,
) -> dict[str, object]:
    """Rewrite one layer/projection group while preserving its byte contract."""

    codes_name, scales_name = _group_names(layer, projection)
    seed = Path(seed_group_path)
    output = Path(output_path)
    weights = np.asarray(source_weights, dtype=np.float32)
    diagonals = np.asarray(importance, dtype=np.float32)
    if weights.ndim != 3:
        raise ValueError("source_weights must have shape [experts, out, in]")
    if diagonals.shape == (weights.shape[-1],):
        diagonals = np.repeat(diagonals[None, :], weights.shape[0], axis=0)
    if diagonals.shape != (weights.shape[0], weights.shape[-1]):
        raise ValueError("importance must have shape [experts, in] or [in]")
    requested_policy = recovery_policy or RecoveryPolicy()
    if not isinstance(requested_policy, RecoveryPolicy):
        raise TypeError("recovery_policy must be a RecoveryPolicy")

    rht_signs: np.ndarray | None = None
    weights_to_quantize = weights
    diagonals_to_quantize = diagonals
    effective_policy = requested_policy
    rotation_stats: dict[str, object] | None = None
    if requested_policy.rotation_rht_seed is not None:
        signs = generate_rht_signs(
            weights.shape[-1],
            f"{requested_policy.rotation_rht_seed}:{layer}:{projection}",
        )
        selection = select_projection_rotation(
            weights.reshape(-1, weights.shape[-1]),
            np.mean(diagonals, axis=0, dtype=np.float64).astype(np.float32),
            e8_1bit_grid() if code_bits == 8 else e8p_full_grid(),
            group_size,
            code_bits,
            signs,
        )
        if selection.selected:
            if requested_policy.recovery_lever == LDLQ_LEVER:
                raise ValueError("combined rotation and LDLQ require a rotated full-Hessian authority")
            rht_signs = np.asarray(selection.selected_signs, dtype=np.int8)
            weights_to_quantize = rotate_weight_rht(
                weights.reshape(-1, weights.shape[-1]), rht_signs
            ).reshape(weights.shape)
            diagonals_to_quantize = np.repeat(
                np.mean(diagonals, axis=1, dtype=np.float64).astype(np.float32)[:, None],
                weights.shape[-1],
                axis=1,
            )
            rotation_stats = {
                "selected": True,
                "unrotated_objective": selection.unrotated_objective,
                "rotated_objective": selection.rotated_objective,
                "objective_gain": selection.objective_gain,
            }
        else:
            effective_policy = replace(requested_policy, rotation_rht_seed=None)

    with safe_open(seed, framework="np") as handle:
        seed_metadata = dict(handle.metadata() or {})
        seed_keys = set(handle.keys())
        if seed_keys != {codes_name, scales_name, "model.vq_codebook.e8"}:
            raise ValueError("seed group does not have the canonical GLM-5.2 VQ tensor inventory")
        seed_codes = handle.get_tensor(codes_name)
        seed_scales = handle.get_tensor(scales_name)
        seed_codebook = handle.get_tensor("model.vq_codebook.e8")
    if code_bits not in (8, 16):
        raise ValueError("code_bits must be 8 or 16")
    quantization = _recovery_quantization(
        seed_metadata,
        group_size=group_size,
        code_bits=code_bits,
        recovery_policy=effective_policy,
    )
    expected_codes_shape = (weights.shape[0], weights.shape[1], weights.shape[2] // 8)
    expected_scales_shape = (weights.shape[0], weights.shape[1], weights.shape[2] // group_size)
    seed_code_bits = 8 if seed_codes.dtype == np.dtype(np.uint8) else (
        16 if seed_codes.dtype == np.dtype(np.uint16) else 0
    )
    if seed_code_bits == 0 or seed_codes.shape != expected_codes_shape:
        raise ValueError("seed codes are not byte-compatible with the requested source group")
    if seed_scales.dtype != np.dtype(np.float16) or seed_scales.shape != expected_scales_shape:
        raise ValueError("seed scales are not byte-compatible with the requested source group")
    expected_seed_codebook = (
        e8_1bit_packed() if seed_code_bits == 8 else e8p_packed_abs_grid()
    )
    if not np.array_equal(seed_codebook, expected_seed_codebook):
        raise ValueError("seed group does not embed its canonical E8-family codebook")

    recovered_codes: list[np.ndarray] = []
    recovered_scales: list[np.ndarray] = []
    expert_stats: list[Mapping[str, object]] = []
    hessian_values = None if h_in is None else np.asarray(h_in)
    if hessian_values is not None and hessian_values.shape not in {
        (weights.shape[-1], weights.shape[-1]),
        (weights.shape[0], weights.shape[-1], weights.shape[-1]),
    }:
        raise ValueError("h_in must be shared [in, in] or per-expert [experts, in, in]")
    backend = e8p_search_backend or _default_recovery_search_backend()
    for expert in range(weights.shape[0]):
        if np.any(diagonals_to_quantize[expert] > 0):
            expert_h_in = None
            if hessian_values is not None:
                expert_h_in = (
                    hessian_values
                    if hessian_values.ndim == 2
                    else hessian_values[expert]
                )
            quantized = quantize_weight_importance_aware(
                weights_to_quantize[expert],
                diagonals_to_quantize[expert],
                group_size=group_size,
                code_bits=code_bits,
                # Byte-identical fast path. Defaults to the fused Metal kernel on
                # Metal-capable hosts (verified byte-exact vs exhaustive, ~40x
                # faster); NumPy stays the deterministic off-host/reference
                # fallback. GLM52_E8P_BACKEND overrides either way.
                e8p_search_backend=backend,
                recovery_policy=effective_policy,
                h_in=expert_h_in,
            )
            recovered_codes.append(quantized.codes)
            recovered_scales.append(quantized.scales)
            expert_stats.append(quantized.stats)
        elif seed_code_bits == code_bits:
            # The selection split supplied no tuning evidence for this expert;
            # preserve its accepted seed bytes rather than data-free rerounding.
            recovered_codes.append(seed_codes[expert])
            recovered_scales.append(seed_scales[expert])
            expert_stats.append({})
        else:
            quantized = quantize_weight_rtn(
                weights_to_quantize[expert], group_size=group_size, code_bits=code_bits
            )
            recovered_codes.append(quantized.codes)
            recovered_scales.append(quantized.scales)
            expert_stats.append({})
    codes_dtype = np.dtype(np.uint8 if code_bits == 8 else np.uint16)
    codebook = e8_1bit_packed() if code_bits == 8 else e8p_packed_abs_grid()
    zero_importance_policy = (
        "preserve_seed_expert_bytes"
        if seed_code_bits == code_bits
        else "source_rtn_e8p"
    )
    codes = np.stack(recovered_codes).astype(codes_dtype, copy=False)
    scales = np.stack(recovered_scales).astype(np.float16, copy=False)
    provenance = _lever_provenance(
        stats_manifest_sha256,
        source_lineage=source_lineage,
        source_verification=source_verification,
        recovery_policy=effective_policy,
    )
    metadata = dict(seed_metadata)
    metadata["quantization_config"] = json.dumps(quantization, sort_keys=True)
    metadata["glm52_recovery_provenance"] = json.dumps(provenance, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial-{uuid4().hex}")
    try:
        tensors = {
            codes_name: codes,
            scales_name: scales,
            "model.vq_codebook.e8": codebook,
        }
        if rht_signs is not None:
            tensors[f"{codes_name.removesuffix('.codes')}.rht_signs"] = rht_signs
        save_file(
            tensors,
            partial,
            metadata=metadata,
        )
        expected_destination = _replacement_destination_identity(
            output,
            label=f"existing recovery group {output}",
        )
        published = publish_file_transactionally(
            partial,
            output,
            expected_destination=expected_destination,
            mode=0o600,
            label=f"recovery group {output}",
        )
        _archive_displaced_publication(
            published,
            output_root=output.parent.parent,
        )
    finally:
        partial.unlink(missing_ok=True)
    codebook_name, expected_codebook_sha256 = codebook_metadata_for_bits(code_bits)
    codebook_sha256 = hashlib.sha256(
        np.asarray(codebook, dtype="<u4").tobytes()
    ).hexdigest()
    if codebook_sha256 != expected_codebook_sha256:
        raise ValueError(f"recovery group {output} codebook hash is not canonical")
    return {
        "layer": layer,
        "projection": projection,
        "artifact_path": output.name,
        "output_path": str(output),
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": _sha256_file(output),
        "codes_shape": list(codes.shape),
        "scales_shape": list(scales.shape),
        "code_bits": code_bits,
        "codes_dtype": codes_dtype.name,
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "tensor_payload_bytes": (
            codes.nbytes
            + scales.nbytes
            + codebook.nbytes
            + (0 if rht_signs is None else rht_signs.nbytes)
        ),
        "zero_importance_policy": zero_importance_policy,
        "lever_provenance": provenance,
        "lever_stats": {
            "experts": expert_stats,
            **({"rotation": rotation_stats} if rotation_stats is not None else {}),
        },
    }


def build_mixed_artifact_tree(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    replacements: Mapping[str, str | Path],
) -> dict[str, object]:
    """Build a resumable mixed tree with inherited files as relative symlinks."""

    seed = Path(seed_artifact_dir).resolve()
    output = Path(output_dir).resolve()
    if output == seed or seed in output.parents:
        raise ValueError("output_dir must not be the seed artifact directory or its child")
    output.mkdir(parents=True, exist_ok=True)
    seed_groups = sorted(seed.glob("layer-*.safetensors"))
    known = {path.name for path in seed_groups}
    unknown = sorted(set(replacements) - known)
    if unknown:
        raise ValueError(f"replacement group files are absent from seed artifact: {unknown}")
    replaced = 0
    inherited = 0
    for seed_path in seed_groups:
        target = output / seed_path.name
        replacement = replacements.get(seed_path.name, seed_path)
        if isinstance(replacement, Mapping):
            replacement = replacement.get("output_path")
        source = Path(replacement).resolve()
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"resume target {target} does not resolve to the expected source")
        else:
            os.symlink(os.path.relpath(source, start=output), target)
        if seed_path.name in replacements:
            replaced += 1
        else:
            inherited += 1
    return {
        "seed_artifact_dir": str(seed),
        "output_dir": str(output),
        "recovered_groups_dir": str((output.parent / "recovered-groups").resolve()),
        "replacement_count": replaced,
        "inherited_group_count": inherited,
    }


def _load_authenticated_stats(
    *,
    stats_dir: Path,
    stats_manifest: Mapping[str, object],
    group_specs: Mapping[tuple[int, str], tuple[int, int]],
) -> dict[tuple[int, str], np.ndarray]:
    if stats_manifest.get("schema_version") != 1:
        raise ValueError("recovery stats manifest schema_version must be 1")
    if stats_manifest.get("record_type") != "glm52_recovery_stats_manifest":
        raise ValueError("recovery stats manifest record_type drifted")
    if stats_manifest.get("status") != "complete":
        raise ValueError("recovery stats manifest is not complete")
    if stats_manifest.get("method") != STATS_METHOD:
        raise ValueError("recovery stats manifest method drifted")
    if stats_manifest.get("split_evidence") != EXPECTED_SPLIT_EVIDENCE:
        raise ValueError("recovery stats are not frozen selection-only evidence")
    for field, expected in (
        ("sample_count", 22),
        ("prompt_count", 22),
        ("position_count", 255),
    ):
        if stats_manifest.get(field) != expected:
            raise ValueError(f"recovery stats manifest {field} drifted")

    authority = stats_manifest.get("source_authority")
    authority_keys = {
        "model_id",
        "revision",
        "profile",
        "config_sha256",
        "index_sha256",
        "authenticated_source_teacher",
        "layer_major_streaming",
        "split_evidence",
    }
    if not isinstance(authority, Mapping) or set(authority) != authority_keys:
        raise ValueError("recovery stats source_authority does not match the producer format")
    if authority.get("model_id") != GLM52_REAP_MODEL_ID:
        raise ValueError("recovery stats source model authority drifted")
    if authority.get("profile") != GLM52_REAP_PROFILE:
        raise ValueError("recovery stats source profile authority drifted")
    if authority.get("revision") != GLM52_REAP_REVISION:
        raise ValueError("recovery stats source revision does not match the pinned revision")
    if not _is_sha256(authority.get("config_sha256")) or not _is_sha256(
        authority.get("index_sha256")
    ):
        raise ValueError("recovery stats source hashes are invalid")
    if authority.get("authenticated_source_teacher") is not True:
        raise ValueError("recovery stats did not use the authenticated source teacher")
    if authority.get("layer_major_streaming") is not True:
        raise ValueError("recovery stats did not use layer-major streaming")
    if authority.get("split_evidence") != EXPECTED_SPLIT_EVIDENCE:
        raise ValueError("recovery stats source authority split evidence drifted")

    selected_layers = stats_manifest.get("selected_layers")
    if (
        not isinstance(selected_layers, list)
        or any(type(layer) is not int or layer < 3 or layer > 77 for layer in selected_layers)
        or len(selected_layers) != len(set(selected_layers))
    ):
        raise ValueError("recovery stats selected_layers are invalid")
    requested_layers = {layer for layer, _ in group_specs}
    if not requested_layers.issubset(selected_layers):
        raise ValueError("recovery stats do not cover every requested layer")
    if stats_manifest.get("projections") != list(GROUP_PROJECTIONS):
        raise ValueError("recovery stats projection inventory drifted")
    route_counts = stats_manifest.get("route_count_by_layer")
    if not isinstance(route_counts, Mapping) or set(route_counts) != {
        str(layer) for layer in selected_layers
    }:
        raise ValueError("recovery stats route_count_by_layer inventory drifted")
    if any(type(count) is not int or count != 2040 for count in route_counts.values()):
        raise ValueError("recovery stats route counts drifted")

    entries = stats_manifest.get("entries")
    if not isinstance(entries, list) or stats_manifest.get("entry_count") != len(entries):
        raise ValueError("recovery stats entry_count does not match entries")
    entries_by_key: dict[tuple[int, str, int], Mapping[str, object]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != STATS_ENTRY_KEYS:
            raise ValueError("recovery stats entry does not match the producer format")
        layer = raw_entry.get("layer")
        projection = raw_entry.get("projection")
        expert = raw_entry.get("expert")
        if (
            type(layer) is not int
            or layer not in selected_layers
            or projection not in GROUP_PROJECTIONS
            or type(expert) is not int
            or expert < 0
        ):
            raise ValueError("recovery stats entry identity is invalid")
        key = (layer, str(projection), expert)
        if key in entries_by_key:
            raise ValueError(f"duplicate recovery stats entry {key}")
        filename = f"layer-{layer:05d}-{projection}-expert-{expert:03d}.npz"
        if raw_entry.get("path") != filename or Path(filename).name != filename:
            raise ValueError(f"recovery stats entry {key} has a noncanonical path")
        if not _is_sha256(raw_entry.get("sha256")):
            raise ValueError(f"recovery stats entry {key} has an invalid SHA-256")
        if raw_entry.get("sample_count") != 22 or raw_entry.get("prompt_count") != 22:
            raise ValueError(f"recovery stats entry {key} prompt counts drifted")
        if raw_entry.get("position_count") != 255:
            raise ValueError(f"recovery stats entry {key} position count drifted")
        prompt_ids = raw_entry.get("prompt_ids")
        if (
            not isinstance(prompt_ids, list)
            or len(prompt_ids) != 22
            or len(set(prompt_ids)) != 22
            or any(not isinstance(prompt_id, str) or not prompt_id for prompt_id in prompt_ids)
        ):
            raise ValueError(f"recovery stats entry {key} prompt IDs are invalid")
        route_count = raw_entry.get("route_count")
        total_route_count = raw_entry.get("total_route_count")
        if type(route_count) is not int or route_count < 0 or total_route_count != 2040:
            raise ValueError(f"recovery stats entry {key} route counts are invalid")
        entries_by_key[key] = raw_entry

    loaded: dict[tuple[int, str], np.ndarray] = {}
    for (layer, projection), (experts, input_dim) in group_specs.items():
        scoped_keys = {
            key for key in entries_by_key if key[:2] == (layer, projection)
        }
        expected_keys = {(layer, projection, expert) for expert in range(experts)}
        if scoped_keys != expected_keys:
            raise ValueError(
                f"recovery stats entries for {layer}:{projection} do not exactly match requested experts"
            )
        values = np.empty((experts, input_dim), dtype=np.float32)
        for expert in range(experts):
            entry = entries_by_key[(layer, projection, expert)]
            if entry.get("input_dim") != input_dim:
                raise ValueError(
                    f"recovery stats entry {(layer, projection, expert)} input_dim drifted"
                )
            path = stats_dir / str(entry["path"])
            if path.resolve().parent != stats_dir.resolve():
                raise ValueError(f"recovery stats path {path} escapes stats_dir")
            authenticated = _read_stable_bytes(path, label=f"recovery stats file {path}")
            if authenticated.sha256 != entry["sha256"]:
                raise ValueError(f"recovery stats file {path} SHA-256 mismatch")
            with np.load(io.BytesIO(authenticated.payload), allow_pickle=False) as payload:
                if set(payload.files) != {
                    "sum_x2",
                    "mean_second_moment",
                    "routing_weighted_importance",
                    "router_score_weighted_importance",
                }:
                    raise ValueError(f"recovery stats file {path} array inventory drifted")
                for name in payload.files:
                    vector = np.asarray(payload[name], dtype=np.float32)
                    if vector.shape != (input_dim,) or not np.isfinite(vector).all():
                        raise ValueError(
                            f"recovery stats {path} has invalid {name} vector"
                        )
                    if np.any(vector < 0):
                        raise ValueError(f"recovery stats {path} has negative {name} values")
                values[expert] = np.asarray(
                    payload["routing_weighted_importance"], dtype=np.float32
                )
        loaded[(layer, projection)] = values
    return loaded


def _validate_resumable_group(
    *,
    target: Path,
    seed: Path,
    layer: int,
    projection: str,
    stats_manifest_sha256: str,
    group_size: int,
    code_bits: int = 8,
    source_lineage: Mapping[str, str],
    source_verification: Mapping[str, object],
    authenticated_sha256: str,
    authenticated_size: int,
    allow_legacy_provenance: bool = False,
    recovery_policy: RecoveryPolicy | None = None,
    seed_code_bits: int = 8,
) -> dict[str, object]:
    codes_name, scales_name = _group_names(layer, projection)
    rht_name = f"{codes_name.removesuffix('.codes')}.rht_signs"
    with safe_open(seed, framework="np") as seed_handle, safe_open(
        target, framework="np"
    ) as target_handle:
        expected_target_names = {
            codes_name,
            scales_name,
            "model.vq_codebook.e8",
        }
        if recovery_policy is not None and recovery_policy.rotation_rht_seed is not None:
            expected_target_names.add(rht_name)
        if set(target_handle.keys()) != expected_target_names:
            raise ValueError(f"resumed group {target} tensor inventory drifted")
        for name in {codes_name, scales_name, "model.vq_codebook.e8"}:
            seed_tensor = seed_handle.get_tensor(name)
            target_tensor = target_handle.get_tensor(name)
            if seed_tensor.shape != target_tensor.shape:
                raise ValueError(f"resumed group {target} tensor contract drifted for {name}")
        if rht_name in expected_target_names:
            rht = target_handle.get_tensor(rht_name)
            if rht.dtype != np.dtype(np.int8) or rht.shape != (
                target_handle.get_tensor(codes_name).shape[-1] * 8,
            ):
                raise ValueError(f"resumed group {target} RHT tensor contract drifted")
        metadata = dict(target_handle.metadata() or {})
        seed_metadata = dict(seed_handle.metadata() or {})
        expected_codes_dtype = np.dtype(np.uint8 if code_bits == 8 else np.uint16)
        expected_codebook = e8_1bit_packed() if code_bits == 8 else e8p_packed_abs_grid()
        if target_handle.get_tensor(codes_name).dtype != expected_codes_dtype:
            raise ValueError(f"resumed group {target} codes dtype drifted")
        if target_handle.get_tensor(scales_name).dtype != np.dtype(np.float16):
            raise ValueError(f"resumed group {target} does not use float16 scales")
        if not np.array_equal(
            target_handle.get_tensor("model.vq_codebook.e8"), expected_codebook
        ):
            raise ValueError(f"resumed group {target} does not embed the canonical codebook")
    try:
        provenance = json.loads(metadata["glm52_recovery_provenance"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"resumed group {target} has no valid recovery provenance") from error
    expected_provenance = _lever_provenance(
        stats_manifest_sha256,
        source_lineage=source_lineage,
        source_verification=source_verification,
        recovery_policy=recovery_policy,
    )
    legacy_provenance = _lever_provenance(stats_manifest_sha256)
    if provenance != expected_provenance and not (
        allow_legacy_provenance and provenance == legacy_provenance
    ):
        raise ValueError(f"resumed group {target} recovery provenance drifted")
    try:
        quantization = json.loads(metadata["quantization_config"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"resumed group {target} has no valid quantization metadata") from error
    if quantization != _recovery_quantization(
        seed_metadata,
        group_size=group_size,
        code_bits=code_bits,
        recovery_policy=recovery_policy,
    ):
        raise ValueError(f"resumed group {target} quantization policy drifted")
    with safe_open(target, framework="np") as target_handle:
        codes_shape = list(target_handle.get_tensor(codes_name).shape)
        scales_shape = list(target_handle.get_tensor(scales_name).shape)
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(code_bits)
    with safe_open(target, framework="np") as target_handle:
        tensor_payload_bytes = sum(
            target_handle.get_tensor(name).nbytes for name in target_handle.keys()
        )
    return {
        "layer": layer,
        "projection": projection,
        "status": "resumed",
        "output_path": str(target),
        "artifact_bytes": authenticated_size,
        "artifact_sha256": authenticated_sha256,
        "codes_shape": codes_shape,
        "scales_shape": scales_shape,
        "code_bits": code_bits,
        "codes_dtype": expected_codes_dtype.name,
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "tensor_payload_bytes": tensor_payload_bytes,
        "zero_importance_policy": (
            "preserve_seed_expert_bytes"
            if seed_code_bits == code_bits
            else "source_rtn_e8p"
        ),
        "lever_provenance": provenance,
    }


def _parse_groups(raw: str) -> tuple[tuple[int, str], ...]:
    result: list[tuple[int, str]] = []
    for item in raw.split(","):
        layer_text, separator, projection = item.strip().partition(":")
        if not separator:
            raise ValueError("groups must use comma-separated '<layer>:<projection>' entries")
        key = (int(layer_text), projection)
        _group_names(*key)
        result.append(key)
    if not result or len(result) != len(set(result)):
        raise ValueError("groups must be non-empty and unique")
    return tuple(result)


def _validate_complete_layer_rates(
    groups: Sequence[tuple[int, str]],
    layer_code_bits: Mapping[int, int],
) -> dict[int, int]:
    requested = tuple(groups)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("groups must be non-empty and unique")
    by_layer: dict[int, set[str]] = {}
    for layer, projection in requested:
        _group_names(layer, projection)
        by_layer.setdefault(layer, set()).add(projection)
    required = set(GROUP_PROJECTIONS)
    if any(projections != required for projections in by_layer.values()):
        raise ValueError("mixed-rate materialization requires complete gate/up/down layers")
    normalized: dict[int, int] = {}
    for layer, code_bits in layer_code_bits.items():
        if type(layer) is not int or layer < 3 or layer > 77:
            raise ValueError("layer_code_bits layers must be in [3, 77]")
        if type(code_bits) is not int or code_bits not in (8, 16):
            raise ValueError("layer_code_bits values must be 8 or 16")
        if layer not in by_layer:
            raise ValueError("layer_code_bits may only name complete selected layers")
        normalized[layer] = code_bits
    return dict(sorted(normalized.items()))


def _rate_policy(layer_code_bits: Mapping[int, int]) -> dict[str, object]:
    return {
        "default_code_bits": 8,
        "layer_code_bits": {
            str(layer): code_bits
            for layer, code_bits in sorted(layer_code_bits.items())
            if code_bits != 8
        },
        "complete_layer_rates_required": True,
    }


def _source_lineage(authority: Mapping[str, object]) -> dict[str, str]:
    return {
        "source_model_id": str(authority["model_id"]),
        "source_revision": str(authority["revision"]),
        "source_config_sha256": str(authority["config_sha256"]),
        "source_index_sha256": str(authority["index_sha256"]),
        "source_profile": str(authority["profile"]),
    }


def _authenticate_source_inputs(
    *,
    source_root: Path,
    index_path: Path,
    authority: Mapping[str, object],
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
) -> tuple[_SourceIndex, Any, dict[str, str], dict[str, object]]:
    if source_root.name != GLM52_REAP_REVISION or source_root.parent.name != "snapshots":
        raise ValueError("source_dir must be the exact pinned GLM52 Hugging Face snapshot")
    expected_index = source_root / "model.safetensors.index.json"
    if index_path.absolute() != expected_index.absolute():
        raise ValueError("index_path must be the pinned snapshot model.safetensors.index.json")
    config_bytes = _read_stable_bytes(
        source_root / "config.json",
        label="source config",
        allow_snapshot_symlink=True,
    )
    index_bytes = _read_stable_bytes(
        index_path,
        label="source index",
        allow_snapshot_symlink=True,
    )
    if config_bytes.sha256 != authority.get("config_sha256"):
        raise ValueError("source config SHA-256 does not match accepted stats authority")
    if index_bytes.sha256 != authority.get("index_sha256"):
        raise ValueError("source index SHA-256 does not match accepted stats authority")
    if config_bytes.sha256 != GLM52_REAP_CONFIG_SHA256:
        raise ValueError("source config SHA-256 does not match the pinned GLM52 config")
    if index_bytes.sha256 != GLM52_REAP_INDEX_SHA256:
        raise ValueError("source index SHA-256 does not match the pinned GLM52 index")
    _load_json_bytes(config_bytes.payload, label="source config")
    index = _load_source_index(index_bytes.payload)
    source_auth = _load_source_auth_module()
    inventory = source_auth._open_authenticated_source_blob_inventory(
        source_root,
        index_bytes.payload,
    )
    try:
        raw_report = inventory.report()
        verification = {
            "source_blob_inventory_sha256": raw_report.get(
                "source_blob_inventory_sha256"
            ),
            "routed_source_blob_inventory_sha256": raw_report.get(
                "routed_source_blob_inventory_sha256"
            ),
            "shard_count": raw_report.get("shard_count"),
        }
        if not _is_sha256(verification["source_blob_inventory_sha256"]):
            raise ValueError("authenticated source blob inventory SHA-256 is invalid")
        if not _is_sha256(verification["routed_source_blob_inventory_sha256"]):
            raise ValueError("authenticated routed source blob inventory SHA-256 is invalid")
        if (
            verification["source_blob_inventory_sha256"]
            != expected_full_source_blob_inventory_sha256
        ):
            raise ValueError(
                "full source blob inventory SHA-256 does not match external accepted authority"
            )
        if (
            verification["routed_source_blob_inventory_sha256"]
            != expected_routed_source_blob_inventory_sha256
        ):
            raise ValueError(
                "routed source blob inventory SHA-256 does not match external accepted authority"
            )
        if type(verification["shard_count"]) is not int or verification["shard_count"] <= 0:
            raise ValueError("authenticated source shard count must be positive")
    except BaseException:
        inventory.close()
        raise
    return index, inventory, _source_lineage(authority), verification


def _read_recovery_policy(path: Path) -> dict[str, str]:
    with safe_open(path, framework="np") as handle:
        metadata = dict(handle.metadata() or {})
    try:
        quantization = json.loads(metadata["quantization_config"])
        policy = quantization["policy"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"recovery group {path} has no valid quantization policy") from error
    if not isinstance(policy, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in policy.items()
    ):
        raise ValueError(f"recovery group {path} quantization policy must map strings")
    return dict(policy)


def _canonical_group_record(
    record: Mapping[str, object],
    *,
    target: Path,
    status: str,
    source_lineage: Mapping[str, str],
    source_verification: Mapping[str, object],
    stats_manifest_sha256: str,
    inspection_path: Path | None = None,
) -> dict[str, object]:
    payload_path = inspection_path or target
    with safe_open(payload_path, framework="np") as handle:
        names = tuple(handle.keys())
        code_names = [name for name in names if name.endswith(".codes")]
        if len(code_names) != 1 or "model.vq_codebook.e8" not in names:
            raise ValueError(f"recovery group {payload_path} tensor inventory drifted")
        codes = handle.get_tensor(code_names[0])
        codebook = handle.get_tensor("model.vq_codebook.e8")
        tensor_payload_bytes = sum(handle.get_tensor(name).nbytes for name in names)
    if codes.dtype == np.dtype(np.uint8):
        code_bits = 8
        expected_codebook = e8_1bit_packed()
    elif codes.dtype == np.dtype(np.uint16):
        code_bits = 16
        expected_codebook = e8p_packed_abs_grid()
    else:
        raise ValueError(f"recovery group {payload_path} codes dtype is not canonical")
    if not np.array_equal(codebook, expected_codebook):
        raise ValueError(f"recovery group {payload_path} codebook is not canonical")
    codebook_name, expected_codebook_sha256 = codebook_metadata_for_bits(code_bits)
    codebook_sha256 = hashlib.sha256(
        np.asarray(codebook, dtype="<u4").tobytes()
    ).hexdigest()
    if codebook_sha256 != expected_codebook_sha256:
        raise ValueError(f"recovery group {payload_path} codebook hash is not canonical")
    return {
        "layer": record["layer"],
        "projection": record["projection"],
        "status": status,
        "artifact_path": target.name,
        "artifact_bytes": record["artifact_bytes"],
        "artifact_sha256": record["artifact_sha256"],
        "source_lineage": dict(source_lineage),
        "recovery_policy": _read_recovery_policy(inspection_path or target),
        "code_bits": code_bits,
        "codes_dtype": codes.dtype.name,
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "tensor_payload_bytes": tensor_payload_bytes,
        "zero_importance_policy": record["zero_importance_policy"],
        "lever_provenance": record["lever_provenance"],
    }


def _routed_tensor_payload_accounting(artifact_root: Path) -> dict[str, int]:
    codes_scales = 0
    codebooks = 0
    for path in sorted(artifact_root.glob("layer-*.safetensors")):
        with safe_open(path, framework="np") as handle:
            for name in handle.keys():
                size = handle.get_tensor(name).nbytes
                if name == "model.vq_codebook.e8":
                    codebooks += size
                else:
                    codes_scales += size
    return {
        "routed_codes_scales_bytes": codes_scales,
        "routed_codebook_bytes": codebooks,
    }


def _routed_tensor_payload_bytes(artifact_root: Path) -> int:
    accounting = _routed_tensor_payload_accounting(artifact_root)
    return accounting["routed_codes_scales_bytes"] + accounting["routed_codebook_bytes"]


def _load_prior_conversion_records(
    *,
    manifest_path: Path,
    groups: Sequence[tuple[int, str]],
    layer_code_bits: Mapping[int, int] | None = None,
    stats_manifest_sha256: str,
    accepted_composite_audit_sha256: str,
    recovery_policy: RecoveryPolicy | None = None,
    seed_authority: AuthenticatedSeed | None = None,
) -> dict[tuple[int, str], Mapping[str, object]] | None:
    if not manifest_path.exists():
        return None
    authenticated = _read_stable_bytes(
        manifest_path,
        label="prior conversion manifest",
    )
    manifest = _load_json_bytes(
        authenticated.payload,
        label="prior conversion manifest",
    )
    recorded_body_sha256 = manifest.get("manifest_body_sha256")
    body = dict(manifest)
    body.pop("manifest_body_sha256", None)
    if not _is_sha256(recorded_body_sha256) or recorded_body_sha256 != _canonical_sha256(
        body
    ):
        raise ValueError("prior conversion manifest body SHA-256 mismatch")
    expected_selected_groups = [f"{layer}:{projection}" for layer, projection in groups]
    expected_rate_policy = _rate_policy(layer_code_bits or {})
    prior_stats_sha256 = manifest.get("stats_manifest_sha256")
    manifest_provenance = manifest.get("lever_provenance")
    legacy_manifest = manifest.get("schema_version") == 1
    old_legacy_manifest = legacy_manifest and manifest.get("accepted_composite_audit_sha256") is None
    legacy_keys = {
        "groups", "lever_provenance", "manifest_body_sha256", "mixed_artifact",
        "record_type", "resumable", "schema_version", "selected_groups",
        "stats_manifest_path", "stats_manifest_sha256", "status",
    }
    authority_matches = (
        manifest.get("accepted_composite_audit_sha256")
        == accepted_composite_audit_sha256
    ) or (
        old_legacy_manifest
        and set(manifest) == legacy_keys
        and manifest_provenance == _lever_provenance(prior_stats_sha256)
    )
    if (
        manifest.get("schema_version") not in (1, 2)
        or manifest.get("record_type") != "glm52_recovery_conversion_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("resumable") is not True
        or manifest.get("selected_groups") != expected_selected_groups
        or (legacy_manifest and expected_rate_policy["layer_code_bits"] != {})
        or (not legacy_manifest and manifest.get("rate_policy") != expected_rate_policy)
        or not authority_matches
        or not _is_sha256(prior_stats_sha256)
        or not isinstance(manifest_provenance, Mapping)
        or manifest_provenance.get("stats_manifest_sha256") != prior_stats_sha256
    ):
        raise ValueError("prior conversion manifest contradicts the requested recovery run")
    if prior_stats_sha256 != stats_manifest_sha256:
        return None
    expected_provenance = _lever_provenance(
        stats_manifest_sha256,
        source_lineage=(manifest.get("source_lineage") if not legacy_manifest else None),
        source_verification=(
            manifest.get("source_verification") if not legacy_manifest else None
        ),
        recovery_policy=recovery_policy,
    )
    if manifest_provenance != expected_provenance:
        return None
    if seed_authority is not None:
        mixed = manifest.get("mixed_artifact")
        if not isinstance(mixed, Mapping):
            return None
        if seed_authority.kind == "accepted_baseline":
            if mixed.get("seed_manifest_sha256") != seed_authority.manifest_sha256:
                return None
        elif (
            mixed.get("seed_recovery_manifest_sha256")
            != seed_authority.manifest_sha256
            or mixed.get("seed_recovery_audit_sha256") != seed_authority.audit_sha256
            or mixed.get("seed_candidate_identity_sha256")
            != seed_authority.candidate_identity_sha256
        ):
            return None
    records = manifest.get("groups")
    if not isinstance(records, list) or len(records) != len(groups):
        raise ValueError("prior conversion manifest group inventory is incomplete")
    by_group: dict[tuple[int, str], Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("prior conversion manifest group record is invalid")
        layer = record.get("layer")
        projection = record.get("projection")
        if type(layer) is not int or not isinstance(projection, str):
            raise ValueError("prior conversion manifest group identity is invalid")
        key = (layer, projection)
        if key in by_group or key not in groups:
            raise ValueError("prior conversion manifest group inventory contradicts the request")
        if record.get("status") not in {"materialized", "resumed"}:
            raise ValueError(f"prior conversion manifest group {key} is not complete")
        if not _is_sha256(record.get("artifact_sha256")):
            raise ValueError(f"prior conversion manifest group {key} has no valid SHA-256")
        record_provenance = record.get("lever_provenance")
        if record_provenance != manifest_provenance:
            raise ValueError(f"prior conversion manifest group {key} provenance drifted")
        if legacy_manifest:
            legacy_record_keys = {
                "artifact_bytes", "artifact_sha256", "codes_shape", "layer",
                "lever_provenance", "output_path", "projection", "scales_shape",
                "status",
            }
            expected_name = f"layer-{layer:05d}-{projection}.safetensors"
            if old_legacy_manifest:
                if set(record) != legacy_record_keys or Path(
                    str(record.get("output_path"))
                ).name != expected_name:
                    raise ValueError(f"prior legacy conversion group {key} schema drifted")
                by_group[key] = {**record, "_legacy_resume": True}
            else:
                by_group[key] = record
        else:
            expected_bits = int((layer_code_bits or {}).get(layer, 8))
            expected_name, expected_hash = codebook_metadata_for_bits(expected_bits)
            if (
                record.get("code_bits") != expected_bits
                or record.get("codes_dtype") != ("uint8" if expected_bits == 8 else "uint16")
                or record.get("codebook_name") != expected_name
                or record.get("codebook_sha256") != expected_hash
                or type(record.get("tensor_payload_bytes")) is not int
                or record.get("zero_importance_policy")
                != (
                    "preserve_seed_expert_bytes"
                    if seed_authority is not None
                    and seed_authority.layer_code_bits.get(layer) == expected_bits
                    else ("preserve_seed_expert_bytes" if expected_bits == 8 else "source_rtn_e8p")
                )
            ):
                raise ValueError(f"prior conversion manifest group {key} rate contract drifted")
            by_group[key] = record
    if set(by_group) != set(groups):
        raise ValueError("prior conversion manifest group inventory is incomplete")
    return by_group


def build_recovery_conversion_manifest(
    *,
    groups: Sequence[tuple[int, str]],
    layer_code_bits: Mapping[int, int] | None = None,
    stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    group_records: Sequence[Mapping[str, object]],
    mixed_artifact: Mapping[str, object],
    source_lineage: Mapping[str, str],
    source_verification: Mapping[str, object],
    accounting: Mapping[str, object],
    accepted_composite_audit_sha256: str,
) -> dict[str, object]:
    if (
        source_verification.get("source_blob_inventory_sha256")
        != expected_full_source_blob_inventory_sha256
        or source_verification.get("routed_source_blob_inventory_sha256")
        != expected_routed_source_blob_inventory_sha256
    ):
        raise ValueError("source verification does not match external inventory authorities")
    policies = [record.get("recovery_policy") for record in group_records]
    if not policies or any(policy != policies[0] for policy in policies):
        raise ValueError("recovery group policies must be identical")
    provenances = [record.get("lever_provenance") for record in group_records]
    if not provenances or any(value != provenances[0] for value in provenances):
        raise ValueError("recovery group lever provenance must be identical")
    provenance = provenances[0]
    manifest: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_recovery_conversion_manifest",
        "status": "complete",
        "resumable": True,
        "selected_groups": [f"{layer}:{projection}" for layer, projection in groups],
        "rate_policy": _rate_policy(layer_code_bits or {}),
        "source_lineage": dict(source_lineage),
        "source_verification": dict(source_verification),
        "expected_full_source_blob_inventory_sha256": (
            expected_full_source_blob_inventory_sha256
        ),
        "expected_routed_source_blob_inventory_sha256": (
            expected_routed_source_blob_inventory_sha256
        ),
        "stats_manifest_sha256": stats_manifest_sha256,
        "accepted_composite_audit_sha256": accepted_composite_audit_sha256,
        "recovery_policy": policies[0],
        "lever_provenance": provenance,
        "groups": [dict(record) for record in group_records],
        "mixed_artifact": dict(mixed_artifact),
        "accounting": dict(accounting),
    }
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    return manifest


def _materialize_groups_from_source_locked(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    seed_artifact_dir: str | Path | None = None,
    seed_recovery_artifact_dir: str | Path | None = None,
    seed_recovery_audit_json: str | Path | None = None,
    stats_dir: str | Path,
    output_dir: str | Path,
    groups: Sequence[tuple[int, str]],
    layer_code_bits: Mapping[int, int] | None = None,
    group_size: int = 512,
    resume: bool = False,
    expected_stats_manifest_sha256: str,
    expected_seed_manifest_sha256: str | None = None,
    expected_seed_recovery_manifest_sha256: str | None = None,
    expected_seed_recovery_audit_sha256: str | None = None,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
    recovery_policy: RecoveryPolicy | None = None,
) -> dict[str, object]:
    source_root = Path(source_dir)
    stats_root = Path(stats_dir)
    output_root = Path(output_dir)
    rate_map = dict(layer_code_bits or {})
    policy = recovery_policy or RecoveryPolicy()
    if not isinstance(policy, RecoveryPolicy):
        raise TypeError("recovery_policy must be a RecoveryPolicy")
    if policy.recovery_lever == LDLQ_LEVER:
        raise ValueError(
            "the source materialization CLI requires a separately authenticated full h_in authority for LDLQ"
        )
    if not _is_sha256(expected_stats_manifest_sha256):
        raise ValueError("expected stats manifest SHA-256 must be lowercase hexadecimal")
    if not _is_sha256(expected_full_source_blob_inventory_sha256):
        raise ValueError(
            "expected full source blob inventory SHA-256 must be lowercase hexadecimal"
        )
    if not _is_sha256(expected_routed_source_blob_inventory_sha256):
        raise ValueError(
            "expected routed source blob inventory SHA-256 must be lowercase hexadecimal"
        )
    stats_manifest_path = stats_root / "glm52-recovery-stats-manifest.json"
    authenticated_manifest = _read_stable_bytes(
        stats_manifest_path,
        label="recovery stats manifest",
    )
    if authenticated_manifest.sha256 != expected_stats_manifest_sha256:
        raise ValueError(
            "stats manifest SHA-256 does not match external accepted authority"
        )
    stats_manifest_sha256 = expected_stats_manifest_sha256
    stats_manifest = _load_json_bytes(
        authenticated_manifest.payload,
        label="recovery stats manifest",
    )
    authority = stats_manifest.get("source_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("recovery stats source authority must be an object")
    source_lineage = _source_lineage(authority)
    composite_authority = _authenticate_accepted_composite_audit(
        accepted_composite_audit_json,
        expected_sha256=expected_composite_audit_sha256,
        expected_source_lineage=source_lineage,
    )
    authenticated_seed = _authenticate_seed(
        seed_artifact_dir=seed_artifact_dir,
        expected_seed_manifest_sha256=expected_seed_manifest_sha256,
        seed_recovery_artifact_dir=seed_recovery_artifact_dir,
        seed_recovery_audit_json=seed_recovery_audit_json,
        expected_seed_recovery_manifest_sha256=(
            expected_seed_recovery_manifest_sha256
        ),
        expected_seed_recovery_audit_sha256=expected_seed_recovery_audit_sha256,
        expected_source_lineage=source_lineage,
        expected_stats_manifest_sha256=stats_manifest_sha256,
        expected_composite_audit_sha256=composite_authority.sha256,
    )
    seed_root = authenticated_seed.root
    seed_records = authenticated_seed.records
    manifest_path = output_root / "conversion-manifest.json"
    prior_records = (
        _load_prior_conversion_records(
            manifest_path=manifest_path,
            groups=groups,
            layer_code_bits=rate_map,
            stats_manifest_sha256=stats_manifest_sha256,
            accepted_composite_audit_sha256=composite_authority.sha256,
            recovery_policy=policy,
            seed_authority=authenticated_seed,
        )
        if resume
        else None
    )
    seed_snapshots = _snapshot_seed_groups(
        seed_root=seed_root,
        records=seed_records,
        groups=groups,
    )
    group_specs: dict[tuple[int, str], tuple[int, int]] = {}
    try:
        for layer, projection in groups:
            seed_path = seed_snapshots.paths[(layer, projection)]
            with safe_open(seed_path, framework="np") as handle:
                codes_name, _ = _group_names(layer, projection)
                codes_shape = handle.get_tensor(codes_name).shape
            if len(codes_shape) != 3:
                raise ValueError(f"seed group {seed_path} codes must have rank 3")
            experts, _out_dim, codeword_count = codes_shape
            group_specs[(layer, projection)] = (experts, codeword_count * 8)
        authenticated_stats = _load_authenticated_stats(
            stats_dir=stats_root,
            stats_manifest=stats_manifest,
            group_specs=group_specs,
        )
    except BaseException:
        seed_snapshots.close()
        raise
    # Keep the synthetic/core API headless-safe.  The streaming converter owns
    # MLX and is imported only after all file-backed evidence is authenticated.
    try:
        from keep.convert.nvfp4 import (
            read_modelopt_nvfp4_weight,
            resolve_modelopt_nvfp4_weight_bundle,
        )
    except BaseException:
        seed_snapshots.close()
        raise
    try:
        _index, source_inventory, source_lineage, source_verification = _authenticate_source_inputs(
            source_root=source_root,
            index_path=Path(index_path),
            authority=authority,
            expected_full_source_blob_inventory_sha256=(
                expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                expected_routed_source_blob_inventory_sha256
            ),
        )
    except BaseException:
        seed_snapshots.close()
        raise
    replacements: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    rewritten_root = output_root / "recovered-groups"
    try:
        # A prior complete manifest must not remain published while its payload
        # generation is replaced in place.  On interruption, absence is the
        # fail-closed state; a new complete manifest is published only after
        # every group, link, accounting claim, and source recheck succeeds.
        manifest_path.unlink(missing_ok=True)
        with source_inventory.modelopt_reader_guard():
            for layer, projection in groups:
                filename = f"layer-{layer:05d}-{projection}.safetensors"
                seed_path = seed_snapshots.paths[(layer, projection)]
                target = rewritten_root / filename
                prior_record = prior_records.get((layer, projection)) if prior_records else None
                if target.exists() and prior_record is not None:
                    with _authenticated_resume_target_snapshot(
                        target,
                        label=f"resume target {target}",
                    ) as snapshot:
                        if snapshot.sha256 == prior_record["artifact_sha256"]:
                            raw_record = _validate_resumable_group(
                                target=snapshot.path,
                                seed=seed_path,
                                layer=layer,
                                projection=projection,
                                stats_manifest_sha256=stats_manifest_sha256,
                                group_size=group_size,
                                code_bits=rate_map.get(layer, 8),
                                source_lineage=source_lineage,
                                source_verification=source_verification,
                                authenticated_sha256=snapshot.sha256,
                                authenticated_size=snapshot.size,
                                allow_legacy_provenance=(
                                    prior_record.get("_legacy_resume") is True
                                ),
                                recovery_policy=policy,
                                seed_code_bits=authenticated_seed.layer_code_bits[layer],
                            )
                            record = _canonical_group_record(
                                raw_record,
                                target=target,
                                status="resumed",
                                source_lineage=source_lineage,
                                source_verification=source_verification,
                                stats_manifest_sha256=stats_manifest_sha256,
                                inspection_path=snapshot.path,
                            )
                            replacements[filename] = target
                            records.append(record)
                            continue
                with safe_open(seed_path, framework="np") as handle:
                    codes_name, _ = _group_names(layer, projection)
                    codes_shape = handle.get_tensor(codes_name).shape
                experts, out_dim, codeword_count = codes_shape
                in_dim = codeword_count * 8
                weights = np.empty((experts, out_dim, in_dim), dtype=np.float32)
                for expert in range(experts):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    bundle = resolve_modelopt_nvfp4_weight_bundle(
                        source_inventory.weight_map,
                        name,
                    )
                    weights[expert] = read_modelopt_nvfp4_weight(
                        source_inventory.blob_root,
                        bundle,
                    )
                importance = authenticated_stats[(layer, projection)]
                raw_record = materialize_recovery_group(
                    source_weights=weights,
                    importance=importance,
                    seed_group_path=seed_path,
                    output_path=target,
                    layer=layer,
                    projection=projection,
                    group_size=group_size,
                    code_bits=rate_map.get(layer, 8),
                    stats_manifest_sha256=stats_manifest_sha256,
                    source_lineage=source_lineage,
                    source_verification=source_verification,
                    recovery_policy=policy,
                )
                record = _canonical_group_record(
                    raw_record,
                    target=target,
                    status="materialized",
                    source_lineage=source_lineage,
                    source_verification=source_verification,
                    stats_manifest_sha256=stats_manifest_sha256,
                )
                records.append(record)
                replacements[filename] = target
                del weights
        source_inventory.verify_after_forward()
    finally:
        source_inventory.close()
        seed_snapshots.close()
    mixed = build_mixed_artifact_tree(
        seed_artifact_dir=authenticated_seed.artifact_root,
        output_dir=output_root / "artifact",
        replacements=replacements,
    )
    if authenticated_seed.kind == "accepted_baseline":
        mixed["seed_manifest_sha256"] = authenticated_seed.manifest_sha256
    else:
        mixed.update(
            {
                "seed_kind": authenticated_seed.kind,
                "seed_recovery_manifest_sha256": authenticated_seed.manifest_sha256,
                "seed_recovery_audit_sha256": authenticated_seed.audit_sha256,
                "seed_candidate_identity_sha256": (
                    authenticated_seed.candidate_identity_sha256
                ),
                "accepted_baseline_seed_manifest_sha256": (
                    PINNED_ACCEPTED_SEED_MANIFEST_SHA256
                ),
                "parent_artifact_root": str(authenticated_seed.artifact_root),
            }
        )
    routed_accounting = _routed_tensor_payload_accounting(
        Path(str(mixed["output_dir"]))
    )
    routed_payload = sum(routed_accounting.values())
    non_routed_tensor_payload_bytes = composite_authority.non_routed_tensor_payload_bytes
    logical_payload = non_routed_tensor_payload_bytes + routed_payload
    if logical_payload > GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES:
        raise ValueError(
            f"logical whole-model tensor payload {logical_payload} exceeds "
            f"the canonical {GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES} byte budget limit"
        )
    accounting = {
        **routed_accounting,
        "main_non_routed_tensor_payload_bytes": non_routed_tensor_payload_bytes,
        "logical_whole_model_tensor_payload_bytes": logical_payload,
        "incremental_disk_bytes": sum(int(record["artifact_bytes"]) for record in records),
        "logical_payload_limit_bytes": GLM52_LOGICAL_PAYLOAD_LIMIT_BYTES,
    }
    manifest = build_recovery_conversion_manifest(
        groups=groups,
        layer_code_bits=rate_map,
        stats_manifest_sha256=stats_manifest_sha256,
        expected_full_source_blob_inventory_sha256=(
            expected_full_source_blob_inventory_sha256
        ),
        expected_routed_source_blob_inventory_sha256=(
            expected_routed_source_blob_inventory_sha256
        ),
        group_records=records,
        mixed_artifact=mixed,
        source_lineage=source_lineage,
        source_verification=source_verification,
        accounting=accounting,
        accepted_composite_audit_sha256=composite_authority.sha256,
    )
    authenticated_seed.verify_current_identity(groups)
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial-{uuid4().hex}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        expected_destination = _replacement_destination_identity(
            manifest_path,
            label="existing recovery conversion manifest",
        )
        published = publish_file_transactionally(
            temporary,
            manifest_path,
            expected_destination=expected_destination,
            mode=0o600,
            label="recovery conversion manifest",
        )
        _archive_displaced_publication(
            published,
            output_root=manifest_path.parent,
        )
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def materialize_groups_from_source(
    *,
    heavy_lock_path: str | Path = ".keep-heavy-job.lock",
    groups: Sequence[tuple[int, str]],
    layer_code_bits: Mapping[int, int] | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Hold the repository heavy-job lock across every output mutation."""

    normalized_rates = _validate_complete_layer_rates(groups, layer_code_bits or {})
    with _cooperating_file_lock(heavy_lock_path):
        return _materialize_groups_from_source_locked(
            groups=groups,
            layer_code_bits=normalized_rates,
            **kwargs,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = _SeedFamilyArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    seed_mode = parser.add_mutually_exclusive_group(required=True)
    seed_mode.add_argument("--seed-artifact-dir")
    seed_mode.add_argument("--seed-recovery-artifact-dir")
    parser.add_argument("--seed-recovery-audit-json")
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-stats-manifest-sha256", required=True)
    parser.add_argument("--expected-seed-manifest-sha256")
    parser.add_argument("--expected-seed-recovery-manifest-sha256")
    parser.add_argument("--expected-seed-recovery-audit-sha256")
    parser.add_argument(
        "--recovery-lever",
        choices=(RECOVERY_LEVER, LDLQ_LEVER),
        default=RECOVERY_LEVER,
    )
    parser.add_argument("--scale-search-multipliers")
    parser.add_argument("--rotation-rht-seed")
    parser.add_argument("--expected-full-source-blob-inventory-sha256", required=True)
    parser.add_argument("--expected-routed-source-blob-inventory-sha256", required=True)
    parser.add_argument("--accepted-composite-audit-json", required=True)
    parser.add_argument("--expected-composite-audit-sha256", required=True)
    parser.add_argument("--heavy-lock-path", default=".keep-heavy-job.lock")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = materialize_groups_from_source(
        source_dir=args.source_dir,
        index_path=args.index_path,
        seed_artifact_dir=args.seed_artifact_dir,
        seed_recovery_artifact_dir=args.seed_recovery_artifact_dir,
        seed_recovery_audit_json=args.seed_recovery_audit_json,
        stats_dir=args.stats_dir,
        output_dir=args.output_dir,
        groups=_parse_groups(args.groups),
        group_size=args.group_size,
        resume=args.resume,
        expected_stats_manifest_sha256=args.expected_stats_manifest_sha256,
        expected_seed_manifest_sha256=args.expected_seed_manifest_sha256,
        expected_seed_recovery_manifest_sha256=(
            args.expected_seed_recovery_manifest_sha256
        ),
        expected_seed_recovery_audit_sha256=(
            args.expected_seed_recovery_audit_sha256
        ),
        recovery_policy=RecoveryPolicy(
            recovery_lever=args.recovery_lever,
            scale_search_multipliers=args.scale_search_multipliers,
            rotation_rht_seed=args.rotation_rht_seed,
        ),
        expected_full_source_blob_inventory_sha256=(
            args.expected_full_source_blob_inventory_sha256
        ),
        expected_routed_source_blob_inventory_sha256=(
            args.expected_routed_source_blob_inventory_sha256
        ),
        accepted_composite_audit_json=args.accepted_composite_audit_json,
        expected_composite_audit_sha256=args.expected_composite_audit_sha256,
        heavy_lock_path=args.heavy_lock_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
