"""GLM-5.2 composite-candidate cache production and frozen-gate comparison.

The comparison path is deliberately NumPy-only.  MLX and the production model
stack are imported lazily, after the candidate producer has acquired the same
global heavy-job and run-specific locks as the source-teacher producer.

Holdout discipline is part of the API contract: all three frozen splits are
measured, but only ``selection`` may guide recovery or candidate choice.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Any, Final

import numpy as np


def _load_teacher_cache_api() -> Any:
    module_name = "mlx_vq.quality.glm52_teacher_cache"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).with_name("glm52_teacher_cache.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 teacher-cache API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_teacher_producer_api() -> Any:
    module_name = "mlx_vq.quality.glm52_teacher_cache_producer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).with_name("glm52_teacher_cache_producer.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 teacher-cache producer API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_route_diagnostics_api() -> Any:
    module_name = "mlx_vq.quality.glm52_route_diagnostics"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).with_name("glm52_route_diagnostics.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 route-diagnostics API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_recovery_audit_api() -> Any:
    module_name = "_glm52_recovery_artifact_audit"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).parents[1] / "validate/glm52_recovery_artifact.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 recovery-artifact audit API")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cache_api = _load_teacher_cache_api()
GLM52ProducerPhase = cache_api.GLM52ProducerPhase
GLM52TeacherCacheContract = cache_api.GLM52TeacherCacheContract
GLM52TeacherCachePrompt = cache_api.GLM52TeacherCachePrompt
build_glm52_teacher_cache_manifest = cache_api.build_glm52_teacher_cache_manifest
publish_glm52_teacher_cache_manifest = cache_api.publish_glm52_teacher_cache_manifest
write_glm52_teacher_cache_shard = cache_api.write_glm52_teacher_cache_shard
audit_glm52_teacher_cache = cache_api.audit_glm52_teacher_cache
canonical_sha256 = cache_api.canonical_sha256


GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256: Final = (
    "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
)
GLM52_CANDIDATE_KIND: Final = "production_composite"
GLM52_RECOVERED_CANDIDATE_KIND: Final = "recovered_composite"
CANDIDATE_PRODUCER_IMPLEMENTATION: Final = (
    "mlx_vq.quality.glm52_candidate_eval;candidate_kind=production_composite"
)
RECOVERED_CANDIDATE_PRODUCER_IMPLEMENTATION_PREFIX: Final = (
    "mlx_vq.quality.glm52_candidate_eval;candidate_kind=recovered_composite"
)
COMPARISON_RECORD_TYPE: Final = "glm52_family_eval_gate"
TEACHER_CACHE_IDENTITY_RECORD_TYPE: Final = (
    "glm52_source_teacher_cache_identity_v1"
)
CANDIDATE_CACHE_IDENTITY_RECORD_TYPE: Final = (
    "glm52_production_composite_candidate_cache_identity_v1"
)
CANDIDATE_ROUTE_TRACE_PRODUCER_IMPLEMENTATION_ID: Final = (
    "mlx_vq.quality.glm52_candidate_eval.route-trace.v2"
)
FROZEN_POLICY_FILE_SHA256: Final = (
    "0975f7dc1117c5fba7532e9166f4767546fd691a6cb52520a40f09874f972ce2"
)

_GATE_KEYS: Final = (
    "mean_kld_max",
    "p999_kld_max",
    "top1_min",
    "domain_top1_min",
    "mean_ppl_ratio_max",
)
LEGACY_DIAGNOSTIC_CANDIDATE_MANIFEST_FILENAME: Final = (
    "glm52-candidate-diagnostic-manifest.json"
)
DIAGNOSTIC_CANDIDATE_MANIFEST_SUFFIX: Final = "-diagnostic-manifest.json"


@dataclass(frozen=True)
class _DiagnosticCandidateManifest:
    payload: Mapping[str, Any]

    @property
    def release_eligible(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload))


def _duplicate_key_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path)
    try:
        value = json.loads(
            input_path.read_bytes(),
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {input_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {input_path} must contain a JSON object")
    return value


def _diagnostic_candidate_manifest_path(cache_root: str | Path) -> Path:
    root = Path(cache_root)
    return root.parent / f"{root.name}{DIAGNOSTIC_CANDIDATE_MANIFEST_SUFFIX}"


def _migrate_legacy_diagnostic_candidate_manifest(cache_root: str | Path) -> None:
    root = Path(cache_root)
    legacy_path = root / LEGACY_DIAGNOSTIC_CANDIDATE_MANIFEST_FILENAME
    if not legacy_path.exists() and not legacy_path.is_symlink():
        return
    if legacy_path.is_symlink() or not legacy_path.is_file():
        raise ValueError("legacy diagnostic candidate manifest must be a regular file")
    sibling_path = _diagnostic_candidate_manifest_path(root)
    if sibling_path.exists() or sibling_path.is_symlink():
        if sibling_path.is_symlink() or not sibling_path.is_file():
            raise ValueError("diagnostic candidate manifest sibling must be a regular file")
        if legacy_path.read_bytes() != sibling_path.read_bytes():
            raise ValueError("legacy and sibling diagnostic candidate manifests differ")
        legacy_path.unlink()
        return
    legacy_path.replace(sibling_path)


def _load_diagnostic_candidate_manifest(
    candidate_root: Path,
    *,
    teacher_cache_identity_sha256: str,
) -> tuple[dict[str, Any], Path]:
    path = _diagnostic_candidate_manifest_path(candidate_root)
    payload = _load_json_object(path, label="diagnostic candidate manifest")
    canonical = _load_json_object(
        candidate_root / cache_api.MANIFEST_FILENAME,
        label="candidate cache manifest",
    )
    reason = (
        "teacher cache release_eligible=false waived for diagnostic-only "
        "evaluation; "
        f"teacher_cache_identity_sha256={teacher_cache_identity_sha256}"
    )
    canonical.update(
        {
            "evidence_class": "diagnostic_only",
            "release_eligible": False,
            "reason": reason,
            "teacher_cache_identity_sha256": teacher_cache_identity_sha256,
        }
    )
    if payload != canonical:
        raise ValueError(
            "diagnostic candidate manifest does not match the audited candidate cache "
            "and teacher identity"
        )
    return payload, path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_family_policy_api() -> Any:
    """Load the repository validator without importing the MLX package surface."""

    module_name = "_glm52_family_policy_validator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    rc_module_name = "mlx_vq.quality.rc_gates"
    previous_rc_module = sys.modules.get(rc_module_name)
    if previous_rc_module is None:
        rc_stub = ModuleType(rc_module_name)
        rc_stub.BALANCED_HARD_TARGETS = {
            "clean_rows": 128,
            "effective_bpw_max": 2.1,
            "lane_s_ratio_max": 1.15,
            "mean_ppl_ratio_max": 1.05,
        }
        rc_stub.COMMUNITY_WOW_TARGETS = {
            "mean_kld_max": 0.30,
            "p999_kld_max": 3.0,
            "top1_min": 0.85,
            "domain_top1_min": 0.80,
        }
        sys.modules[rc_module_name] = rc_stub
    module_path = Path(__file__).with_name("glm52_family.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 frozen-policy validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if previous_rc_module is None:
            sys.modules.pop(rc_module_name, None)
        else:
            sys.modules[rc_module_name] = previous_rc_module
    return module


def build_glm52_candidate_cache_contract(
    base_contract: Any,
    *,
    recovery_candidate_identity_sha256: str | None = None,
    recovery_manifest_body_sha256: str | None = None,
    accepted_composite_audit_sha256: str | None = None,
) -> Any:
    """Create the strict cache-envelope variant for the bound VQ candidate.

    The committed cache schema fixes the source-lineage and precision field
    inventories.  The candidate variant therefore keeps that frozen comparison
    lineage while identifying the producer kind in ``producer.implementation``
    and binding both the durable ledger and package binding identity to the
    authenticated production-composite SHA-256.
    """

    if not isinstance(base_contract, GLM52TeacherCacheContract):
        raise TypeError("base_contract must be a GLM52TeacherCacheContract")
    recovery_values = (
        recovery_candidate_identity_sha256,
        recovery_manifest_body_sha256,
        accepted_composite_audit_sha256,
    )
    if any(value is not None for value in recovery_values) and not all(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in recovery_values
    ):
        raise ValueError("recovered candidate contract requires three lowercase SHA-256 identities")
    recovered = all(value is not None for value in recovery_values)
    if recovered:
        identity_payload = {
            "schema_version": 1,
            "identity_kind": "glm52_recovered_candidate_cache_contract_v1",
            "accepted_baseline_composite_identity_sha256": (
                GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
            ),
            "recovery_candidate_identity_sha256": recovery_candidate_identity_sha256,
            "recovery_manifest_body_sha256": recovery_manifest_body_sha256,
            "accepted_composite_audit_sha256": accepted_composite_audit_sha256,
        }
        bound_identity = canonical_sha256(identity_payload)
        implementation = (
            f"{RECOVERED_CANDIDATE_PRODUCER_IMPLEMENTATION_PREFIX};"
            f"baseline={GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256};"
            f"recovery_candidate={recovery_candidate_identity_sha256};"
            f"recovery_manifest={recovery_manifest_body_sha256};"
            f"accepted_audit={accepted_composite_audit_sha256}"
        )
    else:
        bound_identity = GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
        implementation = CANDIDATE_PRODUCER_IMPLEMENTATION
    producer = dict(base_contract.producer)
    producer["implementation"] = implementation
    producer["version"] = "1"
    non_vq_package = dict(base_contract.non_vq_package)
    non_vq_package["bound_package_identity"] = bound_identity
    return replace(
        base_contract,
        producer=producer,
        non_vq_package=non_vq_package,
        bound_identity_sha256=bound_identity,
    )


def build_glm52_candidate_contract_from_teacher_cache(
    teacher_cache_root: str | Path,
    *,
    prompt_pack_path: str | Path,
) -> Any:
    """Reuse the source cache's frozen envelope; production re-audits under lock."""

    teacher_contract = _contract_from_cache_manifest(
        teacher_cache_root,
        prompt_pack_path=prompt_pack_path,
        candidate=False,
    )
    return build_glm52_candidate_cache_contract(teacher_contract)


def _contract_from_cache_manifest(
    cache_root: str | Path,
    *,
    prompt_pack_path: str | Path,
    candidate: bool,
) -> Any:
    manifest = _load_json_object(
        Path(cache_root) / cache_api.MANIFEST_FILENAME,
        label="candidate cache manifest" if candidate else "teacher cache manifest",
    )
    required = ("producer", "source_evidence", "non_vq_package")
    if any(not isinstance(manifest.get(name), Mapping) for name in required):
        raise ValueError("cache manifest does not contain contract identity objects")
    base = GLM52TeacherCacheContract.from_frozen_prompt_pack(
        prompt_pack_path,
        producer=dict(manifest["producer"]),
        source_evidence=dict(manifest["source_evidence"]),
        non_vq_package=dict(manifest["non_vq_package"]),
    )
    if not candidate:
        return base
    implementation = str(base.producer.get("implementation", ""))
    if implementation.startswith(RECOVERED_CANDIDATE_PRODUCER_IMPLEMENTATION_PREFIX + ";"):
        parts = dict(
            item.split("=", 1)
            for item in implementation.split(";")[2:]
            if "=" in item
        )
        expected = build_glm52_candidate_cache_contract(
            base,
            recovery_candidate_identity_sha256=parts.get("recovery_candidate"),
            recovery_manifest_body_sha256=parts.get("recovery_manifest"),
            accepted_composite_audit_sha256=parts.get("accepted_audit"),
        )
        if parts.get("baseline") != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256:
            raise ValueError("recovered candidate cache baseline identity drifted")
        if (
            base.non_vq_package.get("bound_package_identity")
            != expected.bound_identity_sha256
        ):
            raise ValueError("recovered candidate cache contract identity drifted")
        return expected
    return build_glm52_candidate_cache_contract(base)


def load_glm52_prompt_pack_accounting(
    prompt_pack_path: str | Path,
) -> dict[str, Any]:
    """Reconcile prompt, predictor-position, split, and domain assignments."""

    payload = _load_json_object(prompt_pack_path, label="prompt pack")
    rows = payload.get("prompt_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt pack must contain non-empty prompt_rows")
    split_prompts: Counter[str] = Counter()
    split_positions: Counter[str] = Counter()
    domain_prompts: Counter[str] = Counter()
    domain_positions: Counter[str] = Counter()
    split_domain_prompts: dict[str, Counter[str]] = defaultdict(Counter)
    source_tokens = 0
    prompt_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"prompt_rows[{index}] must be an object")
        prompt = GLM52TeacherCachePrompt.from_prompt_pack_row(row)
        if row.get("token_count") != prompt.token_count:
            raise ValueError(f"prompt_rows[{index}] token_count mismatch")
        prompt_ids.append(prompt.prompt_id)
        source_tokens += prompt.token_count
        positions = prompt.token_count - 1
        split_prompts[prompt.split] += 1
        split_positions[prompt.split] += positions
        domain_prompts[prompt.domain] += 1
        domain_positions[prompt.domain] += positions
        split_domain_prompts[prompt.split][prompt.domain] += 1
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt pack prompt IDs must be unique")
    embedded_count = payload.get("prompt_row_count")
    if type(embedded_count) is int and embedded_count != len(rows):
        raise ValueError("prompt pack prompt_row_count does not reconcile")
    return {
        "prompt_count": len(rows),
        "source_token_count": source_tokens,
        "predictor_position_count": sum(split_positions.values()),
        "split_prompt_counts": dict(sorted(split_prompts.items())),
        "split_position_counts": dict(sorted(split_positions.items())),
        "domain_prompt_counts": dict(sorted(domain_prompts.items())),
        "domain_position_counts": dict(sorted(domain_positions.items())),
        "split_domain_prompt_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_domain_prompts.items())
        },
    }


def _load_frozen_gate(policy_path: str | Path) -> dict[str, Any]:
    policy_file_sha256 = _sha256_file(policy_path)
    if policy_file_sha256 != FROZEN_POLICY_FILE_SHA256:
        raise ValueError(
            "family policy does not match the pinned frozen policy file SHA-256"
        )
    policy = _load_json_object(policy_path, label="family policy")
    family_api = _load_family_policy_api()
    policy = family_api.validate_glm52_family_gate_policy(policy)
    gate = policy.get("eval_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("family policy eval_gate must be an object")
    missing = [key for key in _GATE_KEYS if key not in gate]
    if missing:
        raise ValueError("family policy eval_gate is missing: " + ", ".join(missing))
    thresholds: dict[str, float] = {}
    for key in _GATE_KEYS:
        value = gate[key]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"family policy eval_gate.{key} must be finite")
        thresholds[key] = float(value)
    required_splits = gate.get("required_splits")
    required_domains = gate.get("required_domains")
    if (
        not isinstance(required_splits, list)
        or not required_splits
        or any(not isinstance(value, str) or not value for value in required_splits)
    ):
        raise ValueError("family policy required_splits must be non-empty strings")
    if (
        not isinstance(required_domains, list)
        or not required_domains
        or any(not isinstance(value, str) or not value for value in required_domains)
    ):
        raise ValueError("family policy required_domains must be non-empty strings")
    if gate.get("holdout_tuning_forbidden") is not True:
        raise ValueError("family policy must forbid holdout tuning")
    if gate.get("full_vocabulary_logits_required") is not True:
        raise ValueError("family policy must require full-vocabulary logits")
    return {
        "policy": policy,
        "gate": dict(gate),
        "thresholds": thresholds,
        "required_splits": tuple(required_splits),
        "required_domains": tuple(required_domains),
    }


def _strict_audit(
    root: str | Path,
    *,
    contract: Any,
    label: str,
    allow_non_release: bool = False,
) -> Any:
    try:
        audit = audit_glm52_teacher_cache(root, contract=contract)
    except Exception as error:
        raise ValueError(f"{label} cache failed strict audit: {error}") from error
    if audit.valid is not True:
        raise ValueError(f"{label} cache failed strict audit")
    if audit.release_eligible is not True and not allow_non_release:
        raise ValueError(
            f"{label} cache passed payload audit but is not release eligible"
        )
    return audit


def _require_distinct_cache_roots(
    teacher_cache_root: str | Path,
    candidate_cache_root: str | Path,
) -> tuple[Path, Path]:
    teacher_root = Path(teacher_cache_root)
    candidate_root = Path(candidate_cache_root)
    if teacher_root.resolve(strict=False) == candidate_root.resolve(strict=False):
        raise ValueError("teacher and candidate must use distinct cache roots")
    try:
        if teacher_root.samefile(candidate_root):
            raise ValueError("teacher and candidate must use distinct cache roots")
    except FileNotFoundError:
        pass
    return teacher_root, candidate_root


def _file_identity(path: Path) -> tuple[int, int]:
    identity = path.stat(follow_symlinks=False)
    return identity.st_dev, identity.st_ino


def _require_distinct_cache_files(
    teacher_root: Path,
    candidate_root: Path,
    *,
    prompts: Sequence[Any],
) -> None:
    teacher_paths = [teacher_root / cache_api.MANIFEST_FILENAME]
    candidate_paths = [candidate_root / cache_api.MANIFEST_FILENAME]
    for prompt in prompts:
        relative = Path(cache_api.SHARD_DIRECTORY) / f"{prompt.prompt_id}.safetensors"
        teacher_paths.append(teacher_root / relative)
        candidate_paths.append(candidate_root / relative)
    teacher_identities = {_file_identity(path): path for path in teacher_paths}
    for candidate_path in candidate_paths:
        identity = _file_identity(candidate_path)
        if identity in teacher_identities:
            raise ValueError(
                "teacher and candidate caches must use distinct file identities: "
                f"{teacher_identities[identity]} and {candidate_path}"
            )


def _require_non_conflatable_contracts(
    teacher_contract: Any,
    candidate_contract: Any,
    *,
    expected_recovery_candidate_identity_sha256: str | None = None,
    expected_recovery_manifest_body_sha256: str | None = None,
    expected_accepted_composite_audit_sha256: str | None = None,
) -> None:
    teacher_implementation = teacher_contract.producer.get("implementation")
    if (
        not isinstance(teacher_implementation, str)
        or "glm52_teacher_cache" not in teacher_implementation
        or "candidate_kind=" in teacher_implementation
        or teacher_implementation == CANDIDATE_PRODUCER_IMPLEMENTATION
    ):
        raise ValueError(
            "teacher cache contract does not identify a source-teacher producer"
        )
    if (
        teacher_contract.bound_identity_sha256
        == GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
        or teacher_contract.non_vq_package.get("bound_package_identity")
        == GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
    ):
        raise ValueError("teacher cache uses the candidate composite identity")
    recovery_authority = (
        expected_recovery_candidate_identity_sha256,
        expected_recovery_manifest_body_sha256,
        expected_accepted_composite_audit_sha256,
    )
    if any(value is not None for value in recovery_authority) and not all(
        value is not None for value in recovery_authority
    ):
        raise ValueError(
            "recovered candidate comparison requires all three external identities"
        )
    recovered = all(value is not None for value in recovery_authority)
    expected_candidate = build_glm52_candidate_cache_contract(
        teacher_contract,
        recovery_candidate_identity_sha256=(
            expected_recovery_candidate_identity_sha256 if recovered else None
        ),
        recovery_manifest_body_sha256=(
            expected_recovery_manifest_body_sha256 if recovered else None
        ),
        accepted_composite_audit_sha256=(
            expected_accepted_composite_audit_sha256 if recovered else None
        ),
    )
    if candidate_contract != expected_candidate:
        if recovered:
            raise ValueError(
                "candidate cache is not the exact externally authorized recovery"
            )
        raise ValueError(
            "candidate cache identity envelope is not derived from the source teacher"
        )
    if recovered and (
        candidate_contract.producer.get("implementation")
        != expected_candidate.producer.get("implementation")
        or f"candidate_kind={GLM52_RECOVERED_CANDIDATE_KIND}"
        not in candidate_contract.producer.get("implementation", "")
    ):
        raise ValueError(
            "recovered candidate cache does not identify the recovered producer kind"
        )
    if (
        teacher_contract.prompt_authority.get("file_sha256")
        == cache_api.PINNED_PROMPT_PACK_SHA256
    ):
        family_api = _load_family_policy_api()
        if dict(teacher_contract.source_evidence) != dict(
            family_api.GLM52_TEACHER_CACHE_SOURCE_EVIDENCE
        ):
            raise ValueError(
                "teacher cache source identity does not match the frozen source audits"
            )


def _cache_identity_envelope(
    *,
    record_type: str,
    contract: Any,
    audit: Any,
    manifest_path: Path,
    teacher_cache_identity_sha256: str | None = None,
) -> dict[str, Any]:
    envelope = {
        "record_type": record_type,
        "producer": dict(contract.producer),
        "source": dict(contract.source),
        "source_evidence": dict(contract.source_evidence),
        "bound_identity_sha256": contract.bound_identity_sha256,
        "cache_content_sha256": audit.cache_content_sha256,
        "manifest_body_sha256": audit.manifest_body_sha256,
        "manifest_file_sha256": _sha256_file(manifest_path),
    }
    if teacher_cache_identity_sha256 is not None:
        implementation = contract.producer.get("implementation")
        candidate_kind = (
            GLM52_RECOVERED_CANDIDATE_KIND
            if isinstance(implementation, str)
            and implementation.startswith(
                RECOVERED_CANDIDATE_PRODUCER_IMPLEMENTATION_PREFIX + ";"
            )
            else GLM52_CANDIDATE_KIND
        )
        envelope.update(
            {
                "candidate_kind": candidate_kind,
                "teacher_cache_identity_sha256": teacher_cache_identity_sha256,
            }
        )
    envelope["identity_sha256"] = canonical_sha256(envelope)
    return envelope


def _read_f32_logits(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors header length")
            header_length = struct.unpack("<Q", prefix)[0]
            header_raw = handle.read(header_length)
    except OSError as error:
        raise ValueError(f"could not read audited shard {path}: {error}") from error
    try:
        header = json.loads(
            header_raw,
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"audited shard header changed: {path}: {error}") from error
    if not isinstance(header, Mapping) or set(header) != {"logits"}:
        raise ValueError(f"audited shard tensor inventory changed: {path}")
    descriptor = header["logits"]
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 4
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("dtype") != "F32"
        or descriptor.get("shape") != list(expected_shape)
        or descriptor.get("data_offsets") != [0, expected_bytes]
    ):
        raise ValueError(f"audited shard descriptor changed: {path}")
    array = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=8 + header_length,
        shape=expected_shape,
        order="C",
    )
    return array


def _log_softmax_float64(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _compare_prompt_logits(
    prompt: Any,
    *,
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
) -> dict[str, Any]:
    teacher_log_probs = _log_softmax_float64(teacher_logits)
    candidate_log_probs = _log_softmax_float64(candidate_logits)
    teacher_probs = np.exp(teacher_log_probs)
    token_klds = np.sum(
        teacher_probs * (teacher_log_probs - candidate_log_probs),
        axis=-1,
    )
    token_klds = np.maximum(token_klds, 0.0)
    teacher_top1 = np.argmax(teacher_logits, axis=-1)
    candidate_top1 = np.argmax(candidate_logits, axis=-1)
    top1_matches = teacher_top1 == candidate_top1
    targets = np.asarray(prompt.encoded_token_ids[1:], dtype=np.int64)
    positions = np.arange(targets.size)
    teacher_nll = -float(np.mean(teacher_log_probs[positions, targets]))
    candidate_nll = -float(np.mean(candidate_log_probs[positions, targets]))
    ppl_ratio = math.exp(candidate_nll - teacher_nll)
    if not math.isfinite(ppl_ratio):
        raise ValueError(f"non-finite PPL ratio for prompt {prompt.prompt_id}")
    return {
        "prompt_id": prompt.prompt_id,
        "split": prompt.split,
        "domain": prompt.domain,
        "tuning_eligible": prompt.tuning_eligible,
        "position_count": int(targets.size),
        "teacher_nll": teacher_nll,
        "candidate_nll": candidate_nll,
        "ppl_ratio": ppl_ratio,
        "mean_kld": float(np.mean(token_klds)),
        "p999_kld": float(np.quantile(token_klds, 0.999)),
        "top1_agreement": float(np.mean(top1_matches)),
        "top1_match_count": int(np.sum(top1_matches)),
        "token_klds": [float(value) for value in token_klds.tolist()],
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty GLM52 eval partition")
    token_klds = np.asarray(
        [float(value) for row in rows for value in row["token_klds"]],
        dtype=np.float64,
    )
    position_count = sum(int(row["position_count"]) for row in rows)
    mean_klds = np.asarray(
        [float(row["mean_kld"]) for row in rows],
        dtype=np.float64,
    )
    top1_agreements = np.asarray(
        [float(row["top1_agreement"]) for row in rows],
        dtype=np.float64,
    )
    ppl_ratios = np.asarray(
        [float(row["ppl_ratio"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "prompt_count": len(rows),
        "position_count": position_count,
        "mean_kld": float(np.mean(mean_klds)),
        "p999_kld": float(np.quantile(token_klds, 0.999)),
        "top1_agreement": float(np.mean(top1_agreements)),
        "mean_ppl_ratio": float(np.mean(ppl_ratios)),
    }


def _summarize_optional_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if rows:
        return _summarize_rows(rows)
    return {
        "prompt_count": 0,
        "position_count": 0,
        "mean_kld": None,
        "p999_kld": None,
        "top1_agreement": None,
        "mean_ppl_ratio": None,
    }


def compare_glm52_candidate_caches(
    teacher_cache_root: str | Path,
    candidate_cache_root: str | Path,
    *,
    policy_path: str | Path,
    prompt_pack_path: str | Path | None = None,
    teacher_contract: Any | None = None,
    candidate_contract: Any | None = None,
    allow_non_release_teacher_cache: bool = False,
    expected_recovery_candidate_identity_sha256: str | None = None,
    expected_recovery_manifest_body_sha256: str | None = None,
    expected_accepted_composite_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Strict-audit then compare every FP32 full-vocabulary predictor row."""

    allow_diagnostic_evidence_chain = allow_non_release_teacher_cache
    teacher_root, candidate_root = _require_distinct_cache_roots(
        teacher_cache_root,
        candidate_cache_root,
    )

    if teacher_contract is None or candidate_contract is None:
        if prompt_pack_path is None:
            raise ValueError(
                "prompt_pack_path is required when cache contracts are not supplied"
            )
        teacher_contract = _contract_from_cache_manifest(
            teacher_cache_root,
            prompt_pack_path=prompt_pack_path,
            candidate=False,
        )
        candidate_contract = _contract_from_cache_manifest(
            candidate_cache_root,
            prompt_pack_path=prompt_pack_path,
            candidate=True,
        )
    if not isinstance(teacher_contract, GLM52TeacherCacheContract):
        raise TypeError("teacher_contract must be a GLM52TeacherCacheContract")
    if not isinstance(candidate_contract, GLM52TeacherCacheContract):
        raise TypeError("candidate_contract must be a GLM52TeacherCacheContract")
    recovery_authority = (
        expected_recovery_candidate_identity_sha256,
        expected_recovery_manifest_body_sha256,
        expected_accepted_composite_audit_sha256,
    )
    if any(value is not None for value in recovery_authority) and not all(
        value is not None for value in recovery_authority
    ):
        raise ValueError(
            "recovered candidate comparison requires all three external identities"
        )
    recovered = all(value is not None for value in recovery_authority)
    _require_non_conflatable_contracts(
        teacher_contract,
        candidate_contract,
        expected_recovery_candidate_identity_sha256=(
            expected_recovery_candidate_identity_sha256 if recovered else None
        ),
        expected_recovery_manifest_body_sha256=(
            expected_recovery_manifest_body_sha256 if recovered else None
        ),
        expected_accepted_composite_audit_sha256=(
            expected_accepted_composite_audit_sha256 if recovered else None
        ),
    )
    if not recovered:
        if candidate_contract.producer.get("implementation") != CANDIDATE_PRODUCER_IMPLEMENTATION:
            raise ValueError("candidate cache contract does not identify candidate_kind")
        if (
            candidate_contract.bound_identity_sha256
            != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
            or candidate_contract.non_vq_package.get("bound_package_identity")
            != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
        ):
            raise ValueError("candidate cache is not bound to the frozen composite artifact")
    if teacher_contract.prompts != candidate_contract.prompts:
        raise ValueError("teacher and candidate cache prompt contracts differ")
    if teacher_contract.vocab_size != candidate_contract.vocab_size:
        raise ValueError("teacher and candidate cache vocabulary widths differ")

    frozen = _load_frozen_gate(policy_path)
    required_splits = frozen["required_splits"]
    required_domains = frozen["required_domains"]
    actual_splits = {prompt.split for prompt in teacher_contract.prompts}
    actual_domains = {prompt.domain for prompt in teacher_contract.prompts}
    if actual_splits != set(required_splits):
        raise ValueError("cache splits do not match the frozen policy")
    if actual_domains != set(required_domains):
        raise ValueError("cache domains do not match the frozen policy")

    teacher_audit = _strict_audit(
        teacher_cache_root,
        contract=teacher_contract,
        label="teacher",
        allow_non_release=allow_diagnostic_evidence_chain,
    )
    candidate_audit = _strict_audit(
        candidate_cache_root,
        contract=candidate_contract,
        label="candidate",
        allow_non_release=allow_diagnostic_evidence_chain,
    )
    _require_distinct_cache_files(
        teacher_root,
        candidate_root,
        prompts=teacher_contract.prompts,
    )

    teacher_cache_identity = _cache_identity_envelope(
        record_type=TEACHER_CACHE_IDENTITY_RECORD_TYPE,
        contract=teacher_contract,
        audit=teacher_audit,
        manifest_path=teacher_root / cache_api.MANIFEST_FILENAME,
    )
    candidate_cache_identity = _cache_identity_envelope(
        record_type=(
            "glm52_recovered_composite_candidate_cache_identity_v1"
            if recovered
            else CANDIDATE_CACHE_IDENTITY_RECORD_TYPE
        ),
        contract=candidate_contract,
        audit=candidate_audit,
        manifest_path=candidate_root / cache_api.MANIFEST_FILENAME,
        teacher_cache_identity_sha256=teacher_cache_identity["identity_sha256"],
    )
    diagnostic_manifest: dict[str, Any] | None = None
    diagnostic_manifest_path: Path | None = None
    if (
        teacher_audit.release_eligible is not True
        or candidate_audit.release_eligible is not True
    ):
        diagnostic_manifest, diagnostic_manifest_path = (
            _load_diagnostic_candidate_manifest(
                candidate_root,
                teacher_cache_identity_sha256=teacher_cache_identity[
                    "identity_sha256"
                ],
            )
        )

    rows: list[dict[str, Any]] = []
    for prompt in teacher_contract.prompts:
        shape = (prompt.token_count - 1, teacher_contract.vocab_size)
        relative = PurePosixPath(
            f"{cache_api.SHARD_DIRECTORY}/{prompt.prompt_id}.safetensors"
        )
        teacher_logits = _read_f32_logits(
            teacher_root / Path(*relative.parts),
            expected_shape=shape,
        )
        candidate_logits = _read_f32_logits(
            candidate_root / Path(*relative.parts),
            expected_shape=shape,
        )
        try:
            row = _compare_prompt_logits(
                prompt,
                teacher_logits=teacher_logits,
                candidate_logits=candidate_logits,
            )
            row["teacher_cache_identity_sha256"] = teacher_cache_identity[
                "identity_sha256"
            ]
            row["candidate_cache_identity_sha256"] = candidate_cache_identity[
                "identity_sha256"
            ]
            rows.append(row)
        finally:
            del teacher_logits, candidate_logits

    metrics = _summarize_rows(rows)
    split_metrics = {
        split: _summarize_rows([row for row in rows if row["split"] == split])
        for split in required_splits
    }
    domain_metrics = {
        domain: _summarize_rows([row for row in rows if row["domain"] == domain])
        for domain in required_domains
    }
    split_domain_metrics = {
        split: {
            domain: _summarize_optional_rows(
                [
                    row
                    for row in rows
                    if row["split"] == split and row["domain"] == domain
                ]
            )
            for domain in required_domains
        }
        for split in required_splits
    }
    thresholds = frozen["thresholds"]
    checks = {
        "domain_top1": all(
            domain_metrics[domain]["top1_agreement"]
            >= thresholds["domain_top1_min"]
            for domain in required_domains
        ),
        "mean_kld": metrics["mean_kld"] <= thresholds["mean_kld_max"],
        "mean_ppl_ratio": (
            metrics["mean_ppl_ratio"] <= thresholds["mean_ppl_ratio_max"]
        ),
        "p999_kld": metrics["p999_kld"] <= thresholds["p999_kld_max"],
        "top1": metrics["top1_agreement"] >= thresholds["top1_min"],
    }
    split_counts = {
        split: split_metrics[split]["prompt_count"] for split in required_splits
    }
    split_position_counts = {
        split: split_metrics[split]["position_count"] for split in required_splits
    }
    quality_gate_pass = all(checks.values())
    waived_reasons = [
        f"{label} cache release_eligible=false"
        for label, audit in (
            ("teacher", teacher_audit),
            ("candidate", candidate_audit),
        )
        if audit.release_eligible is not True
    ]
    diagnostic_only = bool(waived_reasons)
    report = {
        "schema_version": 1,
        "record_type": COMPARISON_RECORD_TYPE,
        "gate_status": (
            "glm52_family_eval_gate_passed"
            if quality_gate_pass and not diagnostic_only
            else "glm52_family_eval_gate_failed"
        ),
        "model_id": frozen["policy"].get("model_id"),
        "revision": frozen["policy"].get("revision"),
        "family_policy_contract_sha256": frozen["policy"].get(
            "policy_contract_sha256"
        ),
        "prompt_pack_contract_sha256": teacher_contract.prompt_authority[
            "contract_sha256"
        ],
        "eval_rows_sha256": canonical_sha256(rows),
        "teacher_logits_manifest_sha256": _sha256_file(
            teacher_root / cache_api.MANIFEST_FILENAME
        ),
        "candidate_logits_manifest_sha256": _sha256_file(
            candidate_root / cache_api.MANIFEST_FILENAME
        ),
        "candidate_kind": (
            GLM52_RECOVERED_CANDIDATE_KIND if recovered else GLM52_CANDIDATE_KIND
        ),
        "candidate_artifact_identity_sha256": (
            expected_recovery_candidate_identity_sha256
            if recovered
            else GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
        ),
        "teacher_cache_identity": teacher_cache_identity,
        "candidate_cache_identity": candidate_cache_identity,
        "teacher_cache_audit": {
            "valid": teacher_audit.valid,
            "release_eligible": teacher_audit.release_eligible,
            "cache_content_sha256": teacher_audit.cache_content_sha256,
            "manifest_body_sha256": teacher_audit.manifest_body_sha256,
        },
        "candidate_cache_audit": {
            "valid": candidate_audit.valid,
            "release_eligible": candidate_audit.release_eligible,
            "cache_content_sha256": candidate_audit.cache_content_sha256,
            "manifest_body_sha256": candidate_audit.manifest_body_sha256,
        },
        "teacher_cache_ready": True,
        "teacher_full_vocabulary_logits": True,
        "candidate_full_vocabulary_logits": True,
        "compact_topk_kld_used": False,
        "holdout_tuning_used": False,
        "clean_row_count": len(rows),
        "dirty_row_count": 0,
        "split_counts": split_counts,
        "split_position_counts": split_position_counts,
        "metrics": {
            key: metrics[key]
            for key in ("mean_kld", "p999_kld", "top1_agreement", "mean_ppl_ratio")
        },
        "split_metrics": split_metrics,
        "domain_metrics": domain_metrics,
        "split_domain_metrics": split_domain_metrics,
        "domain_top1_agreement": {
            domain: domain_metrics[domain]["top1_agreement"]
            for domain in required_domains
        },
        "thresholds": thresholds,
        "checks": checks,
        "holdout_discipline": {
            "selection_is_only_tuning_eligible_split": True,
            "selection_may_guide_recovery": True,
            "holdout_tuning_forbidden": True,
            "holdout_may_guide_recovery": False,
            "all_splits_computed": list(required_splits),
        },
        "rows": rows,
        "family_eval_gate_pass": quality_gate_pass and not diagnostic_only,
    }
    if diagnostic_only:
        assert diagnostic_manifest is not None
        assert diagnostic_manifest_path is not None
        report.update(
            {
                "evidence_class": diagnostic_manifest["evidence_class"],
                "release_eligible": diagnostic_manifest["release_eligible"],
                "reason": (
                    f"{'; '.join(waived_reasons)} waived for diagnostic-only "
                    "evaluation; "
                    f"teacher_cache_identity_sha256={teacher_cache_identity['identity_sha256']}"
                ),
                "waived_reasons": waived_reasons,
                "diagnostic_candidate_manifest_path": str(
                    diagnostic_manifest_path
                ),
            }
        )
    return report


def prepare_glm52_candidate_predictor_batch(
    prompts: Sequence[Any],
    *,
    vocab_size: int,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the frozen right-padded predictor batch and per-row pad counts."""

    if not prompts:
        raise ValueError("at least one prompt is required")
    encoded = [tuple(prompt.encoded_token_ids) for prompt in prompts]
    if any(len(row) < 2 for row in encoded):
        raise ValueError("every prompt must contain at least two token IDs")
    if any(
        type(token) is not int or token < 0 or token >= vocab_size
        for row in encoded
        for token in row
    ):
        raise ValueError("prompt token ID is outside the model vocabulary")
    lengths = np.asarray([len(row) - 1 for row in encoded], dtype=np.int32)
    max_length = int(lengths.max())
    token_batch = np.full(
        (len(encoded), max_length),
        pad_token_id,
        dtype=np.int32,
    )
    valid_mask = np.zeros(token_batch.shape, dtype=np.bool_)
    for index, row in enumerate(encoded):
        predictor = row[:-1]
        token_batch[index, : len(predictor)] = predictor
        valid_mask[index, : len(predictor)] = True
    right_padding = max_length - lengths
    return token_batch, valid_mask, right_padding


def _scatter_valid_candidate_rows(
    valid_output: Any,
    *,
    valid_flat_indices: np.ndarray,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    mx: Any,
) -> Any:
    flat = mx.zeros((batch_size * sequence_length, hidden_size), dtype=valid_output.dtype)
    element_indices = (
        valid_flat_indices[:, None] * hidden_size
        + np.arange(hidden_size, dtype=np.int64)[None, :]
    ).reshape(-1)
    flat = mx.put_along_axis(
        flat,
        mx.array(element_indices),
        valid_output.reshape(-1),
        axis=None,
    )
    return flat.reshape(batch_size, sequence_length, hidden_size)


@dataclass(frozen=True)
class GLM52CandidateWorkload:
    model: Any
    validated_inputs: Any
    load_report: Any


@dataclass(frozen=True)
class LayerRouteTrace:
    layer_index: int
    expert_ids: np.ndarray
    scores: np.ndarray
    valid_assignment_count: int
    padded_assignment_count: int


def run_glm52_candidate_to_sink(
    workload: GLM52CandidateWorkload,
    prompts: Sequence[Any],
    *,
    pending_prompt_indices: tuple[int, ...],
    checkpoint_dir: Path,
    checkpoint_identities: object | None,
    phase_boundary: Callable[[], None],
    sink: Callable[[int, np.ndarray], None],
    capture_route_trace: bool = False,
    route_trace_collector: Callable[[LayerRouteTrace], Any] | None = None,
) -> Any:
    """Run the bound VQ model over the exact right-padded predictor batch."""

    del checkpoint_dir, checkpoint_identities
    if not isinstance(workload, GLM52CandidateWorkload):
        raise TypeError("workload must be a GLM52CandidateWorkload")
    mx = importlib.import_module("mlx.core")
    adapter = importlib.import_module("mlx_vq.models.glm52_vq_adapter")
    composite = importlib.import_module("mlx_vq.models.glm52_composite_loader")
    producer = _load_teacher_producer_api()
    create_causal_mask = importlib.import_module(
        "mlx_lm.models.base"
    ).create_causal_mask

    composite.assert_glm52_production_inputs_unchanged(
        workload.validated_inputs.input_fingerprint
    )
    model = workload.model
    token_batch, valid_mask, right_padding = prepare_glm52_candidate_predictor_batch(
        prompts,
        vocab_size=int(model.args.vocab_size),
    )
    if token_batch.shape != (66, 18):
        raise ValueError(
            f"candidate predictor batch must be [66, 18], found {token_batch.shape}"
        )
    if int(valid_mask.sum()) != 744:
        raise ValueError("candidate predictor batch must contain exactly 744 positions")
    if len(model.model.layers) != 78:
        raise ValueError("candidate model must expose exactly main layers 0 through 77")
    tokens = mx.array(token_batch)
    hidden = mx.contiguous(model.model.embed_tokens(tokens))
    sequence_length = token_batch.shape[1]
    causal_mask = create_causal_mask(
        sequence_length,
        right_padding=mx.array(right_padding),
    )
    mx.eval(hidden, causal_mask)
    valid_flat_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
    batch_size = token_batch.shape[0]
    prev_topk_indices = None
    route_traces: list[LayerRouteTrace] = []
    for layer_number, layer in enumerate(model.model.layers):
        if int(layer.layer_idx) != layer_number:
            raise ValueError("candidate decoder layer identity is not contiguous")
        attention_output, next_topk_indices = layer.self_attn(
            layer.input_layernorm(hidden),
            causal_mask,
            None,
            prev_topk_indices,
        )
        if next_topk_indices is not None:
            raise ValueError("short candidate eval must have absent IndexShare state")
        prev_topk_indices = next_topk_indices
        attention_hidden = hidden + attention_output
        mlp_hidden = layer.post_attention_layernorm(attention_hidden)
        if isinstance(layer.mlp, adapter.Glm52VQMoE):
            if layer.mlp.switch_mlp is None:
                raise ValueError("candidate sparse layer has no bound VQ experts")
            if layer.mlp.sharding_group is not None:
                raise ValueError("candidate full-vocabulary eval requires single-host MoE")
            flat_hidden = mlp_hidden.reshape(-1, model.args.hidden_size)
            valid_hidden = mx.take(flat_hidden, mx.array(valid_flat_indices), axis=0)
            expert_indices, route_scores = layer.mlp.gate(valid_hidden)
            expected_route_shape = (
                len(valid_flat_indices),
                int(model.args.num_experts_per_tok),
            )
            if tuple(expert_indices.shape) != expected_route_shape:
                raise ValueError("candidate sparse route inventory is not 744 by top-k")
            routed = layer.mlp.switch_mlp(valid_hidden, expert_indices)
            valid_output = (
                routed * route_scores[..., None]
            ).sum(axis=-2).astype(routed.dtype)
            shared_experts = layer.mlp.get("shared_experts")
            if shared_experts is not None:
                valid_output = valid_output + shared_experts(valid_hidden)
            sparse_output = _scatter_valid_candidate_rows(
                valid_output,
                valid_flat_indices=valid_flat_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=int(model.args.hidden_size),
                mx=mx,
            )
            hidden = attention_hidden + sparse_output
            if capture_route_trace:
                trace = LayerRouteTrace(
                    layer_index=layer_number,
                    expert_ids=np.ascontiguousarray(
                        np.array(expert_indices).astype(np.int32, copy=False)
                    ),
                    scores=np.ascontiguousarray(
                        np.array(route_scores).astype(np.float32, copy=False)
                    ),
                    valid_assignment_count=int(expert_indices.shape[0])
                    * int(expert_indices.shape[1]),
                    padded_assignment_count=0,
                )
                route_traces.append(trace)
                if route_trace_collector is not None:
                    route_trace_collector(trace)
        else:
            hidden = attention_hidden + layer.mlp(mlp_hidden)
        hidden = mx.contiguous(hidden)
        mx.eval(hidden)

    normalized = model.model.norm(hidden)
    mx.eval(normalized)
    phase_boundary()
    output_digest = hashlib.sha256()
    pending = set(pending_prompt_indices)
    predictor_lengths = valid_mask.sum(axis=1).astype(np.int64)
    for prompt_index, predictor_length in enumerate(predictor_lengths):
        if prompt_index not in pending:
            continue
        prompt_hidden = normalized[
            prompt_index : prompt_index + 1,
            : int(predictor_length),
            :,
        ]
        logits = model.lm_head(prompt_hidden).astype(mx.float32).reshape(
            int(predictor_length),
            int(model.args.vocab_size),
        )
        mx.eval(logits)
        stored = np.ascontiguousarray(np.array(logits).astype(np.float32, copy=False))
        output_digest.update(prompt_index.to_bytes(4, "little"))
        output_digest.update(stored.tobytes(order="C"))
        try:
            sink(prompt_index, stored)
        finally:
            del stored, logits
            mx.clear_cache()
            gc.collect()
    composite.assert_glm52_production_inputs_unchanged(
        workload.validated_inputs.input_fingerprint
    )
    return producer.SourceRunEvidence(
        resumed_from_checkpoint=False,
        output_identity_sha256=output_digest.hexdigest(),
        route_traces=tuple(route_traces) if capture_route_trace else None,
    )


def produce_glm52_candidate_cache(
    *,
    contract: Any,
    profile_path: str | Path,
    config_path: str | Path,
    source_index_path: str | Path,
    tokenizer_dir: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
    non_vq_artifact_dir: str | Path,
    non_vq_evidence_json: str | Path,
    routed_artifact_dir: str | Path,
    composite_audit_json: str | Path,
    materialization_runs_jsonl: str | Path,
    full_bind_preflight_json: str | Path,
    teacher_cache_root: str | Path,
    prompt_pack_path: str | Path,
    cache_root: str | Path,
    ledger_path: str | Path,
    allow_non_release_teacher_cache: bool = False,
    route_trace_root: str | Path | None = None,
    recovery_dir: str | Path | None = None,
    expected_seed_manifest_sha256: str | None = None,
    expected_stats_manifest_sha256: str | None = None,
    expected_full_source_blob_inventory_sha256: str | None = None,
    expected_routed_source_blob_inventory_sha256: str | None = None,
    expected_recovery_lever: str | None = None,
    expected_recovery_policy: Mapping[str, str] | None = None,
    expected_composite_audit_sha256: str | None = None,
    expected_recovery_mixed_artifact_identity_sha256: str | None = None,
    expected_recovery_manifest_body_sha256: str | None = None,
) -> Any:
    """Authenticate, strictly bind, run, and publish under both producer locks."""

    capture_route_trace = route_trace_root is not None
    recovery_authority = {
        "recovery_dir": recovery_dir,
        "expected_seed_manifest_sha256": expected_seed_manifest_sha256,
        "expected_stats_manifest_sha256": expected_stats_manifest_sha256,
        "expected_full_source_blob_inventory_sha256": (
            expected_full_source_blob_inventory_sha256
        ),
        "expected_routed_source_blob_inventory_sha256": (
            expected_routed_source_blob_inventory_sha256
        ),
        "expected_recovery_lever": expected_recovery_lever,
        "expected_recovery_policy": expected_recovery_policy,
        "expected_composite_audit_sha256": expected_composite_audit_sha256,
        "expected_recovery_mixed_artifact_identity_sha256": (
            expected_recovery_mixed_artifact_identity_sha256
        ),
        "expected_recovery_manifest_body_sha256": (
            expected_recovery_manifest_body_sha256
        ),
    }
    supplied_recovery_authority = any(
        value is not None for value in recovery_authority.values()
    )
    if not capture_route_trace and supplied_recovery_authority:
        raise ValueError("recovery audit authority requires route trace capture")
    if capture_route_trace:
        missing = [
            name for name, value in recovery_authority.items() if value is None
        ]
        if missing:
            raise ValueError(
                "candidate route trace capture requires recovery audit authority: "
                + ", ".join(missing)
            )
        if allow_non_release_teacher_cache:
            raise ValueError(
                "candidate route trace capture requires a release-eligible teacher cache"
            )
        if len(contract.prompts) != 66 or contract.predictor_position_count != 744:
            raise ValueError(
                "candidate route trace capture requires the complete frozen 66-prompt, "
                "744-position pack"
            )
        expected_contract = build_glm52_candidate_cache_contract(
            contract,
            recovery_candidate_identity_sha256=(
                expected_recovery_mixed_artifact_identity_sha256
            ),
            recovery_manifest_body_sha256=expected_recovery_manifest_body_sha256,
            accepted_composite_audit_sha256=expected_composite_audit_sha256,
        )
        contract = expected_contract
    else:
        if contract.bound_identity_sha256 != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256:
            raise ValueError("candidate contract is not bound to the frozen composite identity")
        if contract.producer.get("implementation") != CANDIDATE_PRODUCER_IMPLEMENTATION:
            raise ValueError("candidate contract does not identify candidate_kind")
    producer = _load_teacher_producer_api()
    validated_box: dict[str, Any] = {}

    if allow_non_release_teacher_cache:
        _migrate_legacy_diagnostic_candidate_manifest(cache_root)

    def audit_composite_inputs() -> Any:
        teacher_contract = _contract_from_cache_manifest(
            teacher_cache_root,
            prompt_pack_path=prompt_pack_path,
            candidate=False,
        )
        teacher_audit = _strict_audit(
            teacher_cache_root,
            contract=teacher_contract,
            label="teacher",
            allow_non_release=allow_non_release_teacher_cache,
        )
        if build_glm52_candidate_cache_contract(
            teacher_contract,
            recovery_candidate_identity_sha256=(
                expected_recovery_mixed_artifact_identity_sha256
                if capture_route_trace
                else None
            ),
            recovery_manifest_body_sha256=(
                expected_recovery_manifest_body_sha256 if capture_route_trace else None
            ),
            accepted_composite_audit_sha256=(
                expected_composite_audit_sha256 if capture_route_trace else None
            ),
        ) != contract:
            raise ValueError(
                "teacher cache identity changed between planning and locked audit"
            )
        composite = importlib.import_module("mlx_vq.models.glm52_composite_loader")
        readiness = _load_json_object(
            tokenizer_readiness_json,
            label="tokenizer readiness",
        )
        authenticated_prompt = readiness.get("prompt")
        if not isinstance(authenticated_prompt, str) or not authenticated_prompt:
            raise ValueError("tokenizer readiness must contain its authenticated prompt")
        validated = composite.validate_glm52_production_inputs(
            profile_path=profile_path,
            config_path=config_path,
            source_index_path=source_index_path,
            tokenizer_dir=tokenizer_dir,
            tokenizer_readiness_json=tokenizer_readiness_json,
            family_policy_json=family_policy_json,
            non_vq_artifact_dir=non_vq_artifact_dir,
            non_vq_evidence_json=non_vq_evidence_json,
            routed_artifact_dir=routed_artifact_dir,
            composite_audit_json=composite_audit_json,
            materialization_runs_jsonl=materialization_runs_jsonl,
            full_bind_preflight_json=full_bind_preflight_json,
            model_id=contract.source["model_id"],
            revision=contract.source["revision"],
            prompt=authenticated_prompt,
        )
        if validated.artifact_identity.sha256 != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256:
            raise ValueError("authenticated composite artifact identity drifted")
        if capture_route_trace:
            recovery_api = _load_recovery_audit_api()
            recovery_audit = recovery_api.audit_glm52_recovery_mixed_artifact(
                recovery_dir,
                profile=validated.profile,
                expected_seed_manifest_sha256=expected_seed_manifest_sha256,
                expected_stats_manifest_sha256=expected_stats_manifest_sha256,
                expected_full_source_blob_inventory_sha256=(
                    expected_full_source_blob_inventory_sha256
                ),
                expected_routed_source_blob_inventory_sha256=(
                    expected_routed_source_blob_inventory_sha256
                ),
                expected_recovery_lever=expected_recovery_lever,
                expected_recovery_policy=expected_recovery_policy,
                accepted_composite_audit_json=composite_audit_json,
                expected_composite_audit_sha256=expected_composite_audit_sha256,
            )
            if recovery_audit.audit_pass is not True:
                raise ValueError("recovery mixed artifact failed its leaf audit")
            if (
                recovery_audit.candidate_identity_sha256
                != expected_recovery_mixed_artifact_identity_sha256
            ):
                raise ValueError("recovery mixed artifact identity mismatch")
            if (
                recovery_audit.manifest_body_sha256
                != expected_recovery_manifest_body_sha256
            ):
                raise ValueError("recovery manifest body identity mismatch")
            validated = replace(
                validated,
                routed_artifact_dir=Path(recovery_audit.recovery_dir) / "artifact",
            )
            validated_box["recovery_audit"] = recovery_audit
            teacher_manifest_path = (
                Path(teacher_cache_root) / cache_api.MANIFEST_FILENAME
            )
            validated_box["teacher_audit"] = teacher_audit
            validated_box["teacher_identity"] = _cache_identity_envelope(
                record_type=TEACHER_CACHE_IDENTITY_RECORD_TYPE,
                contract=teacher_contract,
                audit=teacher_audit,
                manifest_path=teacher_manifest_path,
            )
        validated_box["composite_api"] = composite
        validated_box["value"] = validated
        return SimpleNamespace(
            package_set_sha256=contract.non_vq_package["package_set_sha256"]
        )

    def load_composite() -> GLM52CandidateWorkload:
        validated = validated_box.get("value")
        composite = validated_box.get("composite_api")
        if validated is None or composite is None:
            raise RuntimeError("composite load attempted before fresh input audit")
        recovery_audit = validated_box.get("recovery_audit")
        if recovery_audit is not None:
            recovery_audit.verify_current_identity()
        model, report = composite.load_authenticated_glm52_composite(validated)
        if report.artifact_identity_sha256 != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256:
            raise ValueError("strict composite load report identity drifted")
        return GLM52CandidateWorkload(
            model=model,
            validated_inputs=validated,
            load_report=report,
        )

    def run_candidate_with_recovery_recheck(*args: Any, **kwargs: Any) -> Any:
        evidence = run_glm52_candidate_to_sink(*args, **kwargs)
        recovery_audit = validated_box.get("recovery_audit")
        if recovery_audit is None:
            raise RuntimeError("recovery forward completed without an authenticated audit")
        recovery_audit.verify_current_identity()
        return evidence

    def write_candidate_route_trace(
        root: str | Path,
        layers: Sequence[Any],
        *,
        side: str,
        authority: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if side != "source":
            raise ValueError("teacher producer supplied an unexpected trace side")
        teacher_audit = validated_box.get("teacher_audit")
        teacher_identity = validated_box.get("teacher_identity")
        recovery_audit = validated_box.get("recovery_audit")
        if teacher_audit is None or teacher_identity is None or recovery_audit is None:
            raise RuntimeError("candidate trace publication lacks audited authority")
        candidate_manifest_path = Path(cache_root) / cache_api.MANIFEST_FILENAME
        candidate_audit = _strict_audit(
            cache_root,
            contract=contract,
            label="candidate",
        )
        candidate_identity = _cache_identity_envelope(
            record_type=(
                "glm52_recovered_composite_candidate_cache_identity_v1"
                if recovery_audit is not None
                else CANDIDATE_CACHE_IDENTITY_RECORD_TYPE
            ),
            contract=contract,
            audit=candidate_audit,
            manifest_path=candidate_manifest_path,
            teacher_cache_identity_sha256=teacher_identity["identity_sha256"],
        )
        capture_output_sha256 = authority.get("capture_output_sha256")
        diagnostics = _load_route_diagnostics_api()
        trace_authority = {
            "teacher_cache_path": str(
                Path(teacher_cache_root) / cache_api.MANIFEST_FILENAME
            ),
            "teacher_cache_content_sha256": teacher_audit.cache_content_sha256,
            "accepted_baseline_composite_path": str(routed_artifact_dir),
            "accepted_baseline_composite_identity_sha256": (
                GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
            ),
            "candidate_cache_identity_sha256": candidate_identity[
                "identity_sha256"
            ],
            "recovery_mixed_artifact_identity_sha256": (
                recovery_audit.candidate_identity_sha256
            ),
            "recovery_manifest_body_sha256": (
                recovery_audit.manifest_body_sha256
            ),
            "accepted_composite_audit_sha256": (
                recovery_audit.accepted_composite_audit_sha256
            ),
            "prompt_pack_path": str(contract.prompt_authority["path"]),
            "prompt_pack_sha256": str(
                contract.prompt_authority["file_sha256"]
            ),
            "model_id": str(contract.source["model_id"]),
            "model_revision": str(contract.source["revision"]),
            "producer_implementation_id": (
                CANDIDATE_ROUTE_TRACE_PRODUCER_IMPLEMENTATION_ID
            ),
            "capture_output_path": str(root),
            "capture_output_sha256": capture_output_sha256,
            "evidence_class": "release",
        }
        recovery_audit.verify_current_identity()
        written = diagnostics.write_route_trace_artifact(
            root,
            layers,
            side="candidate",
            schema_version=diagnostics.RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION,
            authority=trace_authority,
        )
        recovery_audit.verify_current_identity()
        return written

    producer_kwargs: dict[str, Any] = dict(
        contract=contract,
        cache_root=cache_root,
        ledger_path=ledger_path,
        checkpoint_dir=Path(f"{ledger_path}.candidate-state"),
        checkpoint_identities=None,
        non_vq_package_auditor=audit_composite_inputs,
        source_loader=load_composite,
        source_runner=(
            run_candidate_with_recovery_recheck
            if capture_route_trace
            else run_glm52_candidate_to_sink
        ),
    )
    if capture_route_trace:
        producer_kwargs.update(
            route_trace_root=route_trace_root,
            route_trace_writer=write_candidate_route_trace,
        )
    result = producer.produce_glm52_teacher_cache(**producer_kwargs)
    if not allow_non_release_teacher_cache or result.manifest is None:
        return result

    teacher_contract = _contract_from_cache_manifest(
        teacher_cache_root,
        prompt_pack_path=prompt_pack_path,
        candidate=False,
    )
    teacher_audit = _strict_audit(
        teacher_cache_root,
        contract=teacher_contract,
        label="teacher",
        allow_non_release=True,
    )
    if teacher_audit.release_eligible is True:
        return result
    teacher_root = Path(teacher_cache_root)
    teacher_identity = _cache_identity_envelope(
        record_type=TEACHER_CACHE_IDENTITY_RECORD_TYPE,
        contract=teacher_contract,
        audit=teacher_audit,
        manifest_path=teacher_root / cache_api.MANIFEST_FILENAME,
    )["identity_sha256"]
    diagnostic_payload = result.manifest.to_dict()
    diagnostic_payload.update(
        {
            "evidence_class": "diagnostic_only",
            "release_eligible": False,
            "reason": (
                "teacher cache release_eligible=false waived for diagnostic-only "
                f"evaluation; teacher_cache_identity_sha256={teacher_identity}"
            ),
            "teacher_cache_identity_sha256": teacher_identity,
        }
    )
    diagnostic_manifest_path = _diagnostic_candidate_manifest_path(cache_root)
    diagnostic_bytes = (
        json.dumps(
            diagnostic_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if diagnostic_manifest_path.exists() or diagnostic_manifest_path.is_symlink():
        if (
            diagnostic_manifest_path.is_symlink()
            or not diagnostic_manifest_path.is_file()
            or diagnostic_manifest_path.read_bytes() != diagnostic_bytes
        ):
            raise ValueError("existing diagnostic candidate manifest differs")
    else:
        cache_api._durable_publish_bytes(
            diagnostic_manifest_path,
            diagnostic_bytes,
        )
    return replace(
        result,
        manifest=_DiagnosticCandidateManifest(diagnostic_payload),
        manifest_path=diagnostic_manifest_path,
    )


__all__ = [
    "CANDIDATE_PRODUCER_IMPLEMENTATION",
    "CANDIDATE_ROUTE_TRACE_PRODUCER_IMPLEMENTATION_ID",
    "GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256",
    "GLM52_CANDIDATE_KIND",
    "GLM52CandidateWorkload",
    "GLM52ProducerPhase",
    "GLM52TeacherCacheContract",
    "GLM52TeacherCachePrompt",
    "LayerRouteTrace",
    "audit_glm52_teacher_cache",
    "build_glm52_candidate_cache_contract",
    "build_glm52_candidate_contract_from_teacher_cache",
    "build_glm52_teacher_cache_manifest",
    "compare_glm52_candidate_caches",
    "load_glm52_prompt_pack_accounting",
    "prepare_glm52_candidate_predictor_batch",
    "produce_glm52_candidate_cache",
    "publish_glm52_teacher_cache_manifest",
    "run_glm52_candidate_to_sink",
    "write_glm52_teacher_cache_shard",
]
