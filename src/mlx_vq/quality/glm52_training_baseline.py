"""Production validated-baseline providers for GLM-5.2 adapter training."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mlx_vq.io.authenticated_artifacts import AuthenticatedFile, parse_json_bytes


CONFIG_ENVIRONMENT_VARIABLE = "GLM52_TRAINING_BASELINE_JSON"

_PATH_FIELDS = (
    "profile_path",
    "config_path",
    "source_index_path",
    "tokenizer_dir",
    "tokenizer_readiness_json",
    "family_policy_json",
    "non_vq_artifact_dir",
    "non_vq_evidence_json",
    "routed_artifact_dir",
    "composite_audit_json",
    "materialization_runs_jsonl",
    "full_bind_preflight_json",
)
_TEXT_FIELDS = ("model_id", "revision")
_REQUIRED_FIELDS = frozenset(
    (*_PATH_FIELDS, *_TEXT_FIELDS, "expected_composite_audit_sha256")
)
_OPTIONAL_FIELDS = frozenset(("recovery_candidate",))
_RECOVERY_PATH_FIELDS = ("conversion_dir", "expected_recovery_policy_json")
_RECOVERY_SHA256_FIELDS = (
    "expected_composite_audit_sha256",
    "expected_seed_manifest_sha256",
    "expected_stats_manifest_sha256",
    "expected_full_source_blob_inventory_sha256",
    "expected_routed_source_blob_inventory_sha256",
)
_RECOVERY_TEXT_FIELDS = ("expected_recovery_lever",)
_RECOVERY_REQUIRED_FIELDS = frozenset(
    (*_RECOVERY_PATH_FIELDS, *_RECOVERY_SHA256_FIELDS, *_RECOVERY_TEXT_FIELDS)
)


def _require_lowercase_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class _RecoveryCandidateConfig:
    conversion_dir: Path
    expected_composite_audit_sha256: str
    expected_seed_manifest_sha256: str
    expected_stats_manifest_sha256: str
    expected_full_source_blob_inventory_sha256: str
    expected_routed_source_blob_inventory_sha256: str
    expected_recovery_lever: str
    expected_recovery_policy_json: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "_RecoveryCandidateConfig":
        missing = sorted(_RECOVERY_REQUIRED_FIELDS - payload.keys())
        unknown = sorted(payload.keys() - _RECOVERY_REQUIRED_FIELDS)
        if missing:
            raise ValueError(
                "GLM52 recovery candidate config is missing required fields: "
                + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "GLM52 recovery candidate config has unknown fields: "
                + ", ".join(unknown)
            )
        values: dict[str, Any] = {}
        for field in _RECOVERY_PATH_FIELDS:
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"GLM52 recovery candidate config {field} must be a path string"
                )
            values[field] = Path(value).expanduser()
        for field in _RECOVERY_TEXT_FIELDS:
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"GLM52 recovery candidate config {field} must be a non-empty string"
                )
            values[field] = value
        for field in _RECOVERY_SHA256_FIELDS:
            values[field] = _require_lowercase_sha256(
                payload[field],
                label=f"recovery candidate {field}",
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class _TrainingBaselineConfig:
    profile_path: Path
    config_path: Path
    source_index_path: Path
    tokenizer_dir: Path
    tokenizer_readiness_json: Path
    family_policy_json: Path
    non_vq_artifact_dir: Path
    non_vq_evidence_json: Path
    routed_artifact_dir: Path
    composite_audit_json: Path
    expected_composite_audit_sha256: str
    materialization_runs_jsonl: Path
    full_bind_preflight_json: Path
    model_id: str
    revision: str
    recovery_candidate: _RecoveryCandidateConfig | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "_TrainingBaselineConfig":
        missing = sorted(_REQUIRED_FIELDS - payload.keys())
        unknown = sorted(payload.keys() - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
        if missing:
            raise ValueError(
                "GLM52 training baseline config is missing required fields: "
                + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "GLM52 training baseline config has unknown fields: "
                + ", ".join(unknown)
            )
        values: dict[str, Any] = {}
        for field in _PATH_FIELDS:
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"GLM52 training baseline config {field} must be a path string"
                )
            values[field] = Path(value).expanduser()
        for field in _TEXT_FIELDS:
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"GLM52 training baseline config {field} must be a non-empty string"
                )
            values[field] = value
        values["expected_composite_audit_sha256"] = _require_lowercase_sha256(
            payload["expected_composite_audit_sha256"],
            label="expected composite audit SHA-256",
        )
        if "recovery_candidate" in payload:
            recovery_candidate = payload["recovery_candidate"]
            if not isinstance(recovery_candidate, Mapping):
                raise ValueError(
                    "GLM52 training baseline config recovery_candidate must be an object"
                )
            values["recovery_candidate"] = _RecoveryCandidateConfig.from_mapping(
                recovery_candidate
            )
        return cls(**values)


def _config_path_from_environment() -> Path:
    value = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{CONFIG_ENVIRONMENT_VARIABLE} must point at the authenticated "
            "GLM52 training baseline config JSON"
        )
    return Path(value).expanduser()


def _load_config(authenticated: AuthenticatedFile) -> _TrainingBaselineConfig:
    payload = parse_json_bytes(
        authenticated.bytes,
        label="GLM52 training baseline config",
    )
    return _TrainingBaselineConfig.from_mapping(payload)


def _authenticated_prompt(authenticated: AuthenticatedFile) -> str:
    readiness = parse_json_bytes(
        authenticated.bytes,
        label="GLM52 tokenizer readiness",
    )
    prompt = readiness.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("tokenizer readiness has no authenticated prompt")
    return prompt


def _validate_strict_load_report(
    report: Any,
    *,
    candidate_identity_sha256: str,
    expected_sparse_layers: tuple[int, ...],
) -> None:
    if report.artifact_identity_sha256 != candidate_identity_sha256:
        raise ValueError("strict composite load report identity drifted")
    if tuple(report.bound_sparse_layer_ids) != expected_sparse_layers:
        raise ValueError("strict composite load report routed layer inventory drifted")
    if tuple(report.dense_routed_parameter_names):
        raise ValueError("strict composite load report found dense routed parameters")
    if report.unbound_vq_experts is not False:
        raise ValueError("strict composite load report found unbound routed experts")


def audit_glm52_recovery_candidate(
    *,
    conversion_dir: str | Path,
    profile: Any,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy: Mapping[str, Any],
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
    recovery_audit_api: Any = None,
    recovery_api: Any = None,
) -> Any:
    """Run the recovery audit and enforce reevaluation's routed-group contract."""

    if recovery_audit_api is None:
        recovery_audit_api = importlib.import_module(
            "mlx_vq.validate.glm52_recovery_artifact"
        )
    if recovery_api is None:
        recovery_api = importlib.import_module("mlx_vq.quality.glm52_recovery")
    recovery_audit = recovery_audit_api.audit_glm52_recovery_mixed_artifact(
        conversion_dir,
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
        expected_recovery_policy=expected_recovery_policy,
        accepted_composite_audit_json=accepted_composite_audit_json,
        expected_composite_audit_sha256=expected_composite_audit_sha256,
    )
    if recovery_audit.audit_pass is not True:
        raise ValueError("recovery artifact audit did not pass")
    if recovery_audit.group_count != 225:
        raise ValueError(
            "recovery artifact audit did not prove the complete 225-group tree"
        )
    recovered_layers = tuple(recovery_audit.complete_replacement_layer_ids)
    recovered_groups = tuple(
        group.group_key
        for group in recovery_audit.groups
        if group.classification == "replacement"
    )
    expected_groups = {
        f"{layer}:{projection}"
        for layer in recovered_layers
        for projection in recovery_api.PROJECTIONS
    }
    if not recovered_layers or set(recovered_groups) != expected_groups:
        raise ValueError(
            "recovery artifact audit did not prove complete replacement groups per layer"
        )
    return recovery_audit


def bind_audited_glm52_recovery_candidate(
    model: Any,
    recovery_audit: Any,
    *,
    profile: Any,
    recovery_api: Any = None,
    adapter: Any = None,
) -> None:
    """Bind exactly the routed groups authenticated by a recovery audit."""

    if recovery_api is None:
        recovery_api = importlib.import_module("mlx_vq.quality.glm52_recovery")
    if adapter is None:
        adapter = importlib.import_module("mlx_vq.models.glm52_vq_adapter")
    candidate_group_paths = recovery_api.authenticated_recovery_candidate_group_paths(
        recovery_audit
    )
    candidate_layers = tuple(
        sorted(
            {
                int(group.group_key.split(":", 1)[0])
                for group in recovery_audit.groups
            }
        )
    )
    rebound = adapter.bind_glm52_vq_experts_from_paths(
        model,
        candidate_group_paths,
        layers=candidate_layers,
        profile=profile,
        strict=True,
    )
    if rebound != candidate_layers:
        raise ValueError("recovery candidate did not bind every audited routed layer")
    recovery_audit.verify_current_identity()


def _load_accepted_baseline(
    config_path: Path,
) -> tuple[Any, _TrainingBaselineConfig, Any, Any]:
    with AuthenticatedFile.open(
        config_path,
        label="GLM52 training baseline config",
    ) as authenticated_config:
        config = _load_config(authenticated_config)
        with AuthenticatedFile.open(
            config.composite_audit_json,
            label="accepted GLM52 composite audit",
        ) as authenticated_audit, AuthenticatedFile.open(
            config.tokenizer_readiness_json,
            label="GLM52 tokenizer readiness",
        ) as authenticated_readiness:
            if authenticated_audit.sha256 != config.expected_composite_audit_sha256:
                raise ValueError(
                    "accepted composite audit SHA-256 does not match the external "
                    "training baseline authority"
                )
            prompt = _authenticated_prompt(authenticated_readiness)
            composite = importlib.import_module(
                "mlx_vq.models.glm52_composite_loader"
            )
            training = importlib.import_module(
                "mlx_vq.quality.glm52_adapter_training"
            )
            validated = composite.validate_glm52_production_inputs(
                profile_path=config.profile_path,
                config_path=config.config_path,
                source_index_path=config.source_index_path,
                tokenizer_dir=config.tokenizer_dir,
                tokenizer_readiness_json=config.tokenizer_readiness_json,
                family_policy_json=config.family_policy_json,
                non_vq_artifact_dir=config.non_vq_artifact_dir,
                non_vq_evidence_json=config.non_vq_evidence_json,
                routed_artifact_dir=config.routed_artifact_dir,
                composite_audit_json=config.composite_audit_json,
                materialization_runs_jsonl=config.materialization_runs_jsonl,
                full_bind_preflight_json=config.full_bind_preflight_json,
                model_id=config.model_id,
                revision=config.revision,
                prompt=prompt,
            )
            validated_audit_sha256 = validated.input_evidence_file_sha256.get(
                "composite_audit_json"
            )
            if validated_audit_sha256 != config.expected_composite_audit_sha256:
                raise ValueError(
                    "validated composite audit SHA-256 drifted from the external "
                    "training baseline authority"
                )
            candidate_identity_sha256 = validated.artifact_identity.sha256
            model, load_report = composite.load_authenticated_glm52_composite(validated)
            _validate_strict_load_report(
                load_report,
                candidate_identity_sha256=candidate_identity_sha256,
                expected_sparse_layers=tuple(composite.GLM52_EXPECTED_SPARSE_LAYERS),
            )
            authenticated_readiness.verify_visible()
            authenticated_audit.verify_visible()
            authenticated_config.verify_visible()
            baseline = training.ValidatedGLM52TrainingBaseline(
                model=model,
                candidate_identity_sha256=candidate_identity_sha256,
            )
            return baseline, config, validated, training


def accepted_baseline_provider(_args: Any = None) -> Any:
    """Load and bind the accepted production composite named by the env config."""

    config_path = _config_path_from_environment()
    baseline, _config, _validated, _training = _load_accepted_baseline(config_path)
    return baseline


def recovery_candidate_baseline_provider(_args: Any = None) -> Any:
    """Load the accepted composite and optionally bind its recovery candidate."""

    config_path = _config_path_from_environment()
    baseline, config, validated, training = _load_accepted_baseline(config_path)
    recovery = config.recovery_candidate
    if recovery is None:
        return baseline
    if (
        recovery.expected_composite_audit_sha256
        != config.expected_composite_audit_sha256
    ):
        raise ValueError(
            "recovery candidate composite audit SHA-256 does not match the "
            "accepted training baseline authority"
        )
    with AuthenticatedFile.open(
        recovery.expected_recovery_policy_json,
        label="GLM52 expected recovery policy",
    ) as authenticated_policy:
        expected_recovery_policy = parse_json_bytes(
            authenticated_policy.bytes,
            label="GLM52 expected recovery policy",
        )
        recovery_audit = audit_glm52_recovery_candidate(
            conversion_dir=recovery.conversion_dir,
            profile=validated.profile,
            expected_seed_manifest_sha256=recovery.expected_seed_manifest_sha256,
            expected_stats_manifest_sha256=recovery.expected_stats_manifest_sha256,
            expected_full_source_blob_inventory_sha256=(
                recovery.expected_full_source_blob_inventory_sha256
            ),
            expected_routed_source_blob_inventory_sha256=(
                recovery.expected_routed_source_blob_inventory_sha256
            ),
            expected_recovery_lever=recovery.expected_recovery_lever,
            expected_recovery_policy=expected_recovery_policy,
            accepted_composite_audit_json=config.composite_audit_json,
            expected_composite_audit_sha256=(
                recovery.expected_composite_audit_sha256
            ),
        )
        bind_audited_glm52_recovery_candidate(
            baseline.model,
            recovery_audit,
            profile=validated.profile,
        )
        authenticated_policy.verify_visible()
    return training.ValidatedGLM52TrainingBaseline(
        model=baseline.model,
        candidate_identity_sha256=recovery_audit.candidate_identity_sha256,
    )


environment_baseline_provider = accepted_baseline_provider


__all__ = [
    "CONFIG_ENVIRONMENT_VARIABLE",
    "accepted_baseline_provider",
    "audit_glm52_recovery_candidate",
    "bind_audited_glm52_recovery_candidate",
    "environment_baseline_provider",
    "recovery_candidate_baseline_provider",
]
