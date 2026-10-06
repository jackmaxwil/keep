"""Deterministic, local-only Task 13 reviewed source bundle.

This module validates already-authoritative live records and reproduces fixed
repository records.  It never calls AWS, deploys infrastructure, runs a
rehearsal, promotes a model, or invents a review/evaluation result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType

from .approvals import (
    build_gpu_residual_liability_approval,
    build_production_support_plane_approval,
    validate_gpu_residual_liability_approval,
    validate_production_support_plane_approval,
)
from .canonical import canonical_json_bytes, canonical_sha256
from .cloudformation_stacks import validate_bootstrap_template
from .glm52_sky_campaign import (
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from .task10_worker import (
    MountFreeTaskInputs,
    validate_task_inputs,
    worker_bootstrap_descriptor_from_mapping,
)
from .task13_clean_rehearsal import CLEAN_REHEARSAL_KEY_PREFIX

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
MANIFEST_NAME = "source-bundle-manifest.json"

ARTIFACT_KINDS = (
    "TASK11_REVIEW_APPROVAL",
    "TASK12_REVIEW_APPROVAL",
    "BOOTSTRAP_TEMPLATE",
    "ACCEPTED_BASELINE",
    "PROMPT_PACK",
    "TRAINING_CONFIGURATION",
    "GPU_SPEND_APPROVAL",
    "SUPPORT_APPROVAL",
    "RESIDUAL_LIABILITY_APPROVAL",
    "CAMPAIGN_DESCRIPTOR",
    "TASK10_WORKER_DESCRIPTOR",
    "TASK10_TASK_INPUTS",
)
LIVE_SOURCE_KINDS = (
    "TASK11_REVIEW_APPROVAL",
    "TASK12_REVIEW_APPROVAL",
    "GPU_SPEND_APPROVAL",
    "CAMPAIGN_DESCRIPTOR",
    "TASK10_WORKER_DESCRIPTOR",
    "TASK10_TASK_INPUTS",
)
DYNAMIC_OUTPUT_KINDS = (
    "RETAINED_TEMPLATE",
    "FENCE_TEMPLATE",
    "SUPPORT_TEMPLATE",
    "CLEAN_REHEARSAL",
)
DYNAMIC_OUTPUT_KEY_PREFIXES = MappingProxyType(
    {"CLEAN_REHEARSAL": CLEAN_REHEARSAL_KEY_PREFIX}
)
ARTIFACT_FILENAMES = MappingProxyType(
    {
        "TASK11_REVIEW_APPROVAL": "task11-review-approval.json",
        "TASK12_REVIEW_APPROVAL": "task12-review-approval.json",
        "BOOTSTRAP_TEMPLATE": "container-bootstrap-v1.json",
        "ACCEPTED_BASELINE": "accepted-baseline.json",
        "PROMPT_PACK": "prompt-pack.json",
        "TRAINING_CONFIGURATION": "training-configuration.json",
        "GPU_SPEND_APPROVAL": "gpu-spend-approval.json",
        "SUPPORT_APPROVAL": "support-plane-approval.json",
        "RESIDUAL_LIABILITY_APPROVAL": ("residual-liability-approval.json"),
        "CAMPAIGN_DESCRIPTOR": "campaign-descriptor-v2.json",
        "TASK10_WORKER_DESCRIPTOR": "task10-worker-descriptor.json",
        "TASK10_TASK_INPUTS": "task10-task-inputs.json",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OWNER_APPROVAL_RELATIVE = (
    "docs/superpowers/approvals/2026-07-28-glm52-campaign-owner-approvals.md"
)
_OWNER_APPROVAL_SHA256 = (
    "23ccff3454896d7fb8e9d0722b6c76522ebf2dfa3dfea9856a17a3e48b26ebc1"
)
_FIXED_JSON_SOURCES = MappingProxyType(
    {
        "BOOTSTRAP_TEMPLATE": (
            "aws/glm52-gpu/cfn/h1g/container-bootstrap-v1.json",
            ("03e7caa5252a308df80a87674c106f21cb8ad18241f4c1b83b8bea10b6fe9d09"),
        ),
        "ACCEPTED_BASELINE": (
            ("artifacts/quality/glm52-training-baseline-accepted-20260711.json"),
            ("81ab08e143314907985c7e1e5985474ae8ec9b8c6e126098832bb582687bc221"),
        ),
        "PROMPT_PACK": (
            "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json",
            ("697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31"),
        ),
    }
)
_TRAINING_BUILDER_RELATIVE = "src/mlx_vq/quality/glm52_sky_training_config.py"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Task13SourceBundleError(ValueError):
    """A Task 13 local source or output contract was not exact."""


@dataclass(frozen=True)
class PinnedFileSource:
    """One absolute regular file and its independently supplied identity."""

    path: Path
    expected_file_sha256: str


@dataclass(frozen=True)
class PinnedJsonSource:
    """One absolute canonical JSON object and its two supplied identities."""

    path: Path
    expected_file_sha256: str
    expected_body_sha256: str


@dataclass(frozen=True)
class _Artifact:
    kind: str
    payload: dict[str, object]
    source: dict[str, object]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Task13SourceBundleError(label + " must be lowercase SHA-256")
    return value


def _read_absolute_regular(path: Path, label: str) -> bytes:
    if not isinstance(path, Path):
        raise Task13SourceBundleError(label + " path must be a Path")
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise Task13SourceBundleError(
            label + " must be an absolute canonical regular non-symlink file"
        )
    raw = path.read_bytes()
    if not raw:
        raise Task13SourceBundleError(label + " must not be empty")
    return raw


def _parse_json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Task13SourceBundleError(label + " must be UTF-8 JSON") from error
    if type(value) is not dict:
        raise Task13SourceBundleError(label + " must be a JSON object")
    return value


def _read_pinned_file(source: PinnedFileSource, label: str) -> bytes:
    if not isinstance(source, PinnedFileSource):
        raise Task13SourceBundleError(label + " identity must be typed")
    expected = _require_sha256(
        source.expected_file_sha256,
        label + " expected_file_sha256",
    )
    raw = _read_absolute_regular(source.path, label)
    if _sha256(raw) != expected:
        raise Task13SourceBundleError(label + " file SHA-256 mismatch")
    return raw


def _read_pinned_json(
    source: PinnedJsonSource,
    label: str,
) -> tuple[dict[str, object], bytes]:
    if not isinstance(source, PinnedJsonSource):
        raise Task13SourceBundleError(label + " identity must be typed")
    raw = _read_pinned_file(
        PinnedFileSource(
            path=source.path,
            expected_file_sha256=source.expected_file_sha256,
        ),
        label,
    )
    value = _parse_json_object(raw, label)
    canonical = canonical_json_bytes(value)
    if raw != canonical + b"\n":
        raise Task13SourceBundleError(label + " must be canonical JSON with one LF")
    expected_body = _require_sha256(
        source.expected_body_sha256,
        label + " expected_body_sha256",
    )
    if _sha256(canonical) != expected_body:
        raise Task13SourceBundleError(label + " body SHA-256 mismatch")
    return value, raw


def _read_fixed_json(
    kind: str,
) -> tuple[dict[str, object], bytes, str]:
    relative, expected_sha256 = _FIXED_JSON_SOURCES[kind]
    path = _REPO_ROOT / relative
    raw = _read_absolute_regular(path, kind + " repository source")
    if _sha256(raw) != expected_sha256:
        raise Task13SourceBundleError(
            kind + " fixed repository source identity drifted"
        )
    return _parse_json_object(raw, kind), raw, relative


def _load_training_builder() -> tuple[ModuleType, str]:
    path = _REPO_ROOT / _TRAINING_BUILDER_RELATIVE
    raw = _read_absolute_regular(path, "training configuration builder")
    spec = importlib.util.spec_from_file_location(
        "_glm52_task13_training_configuration_builder",
        path,
    )
    if spec is None or spec.loader is None:
        raise Task13SourceBundleError("training configuration builder cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, _sha256(raw)


def _fixed_artifacts() -> dict[str, _Artifact]:
    bootstrap, bootstrap_raw, bootstrap_relative = _read_fixed_json(
        "BOOTSTRAP_TEMPLATE"
    )
    try:
        validate_bootstrap_template(bootstrap)
    except (TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "BOOTSTRAP_TEMPLATE semantic validation failed"
        ) from error

    baseline, baseline_raw, baseline_relative = _read_fixed_json("ACCEPTED_BASELINE")
    required_baseline = {
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
        "expected_composite_audit_sha256",
        "materialization_runs_jsonl",
        "full_bind_preflight_json",
        "model_id",
        "revision",
    }
    if set(baseline) != required_baseline:
        raise Task13SourceBundleError(
            "ACCEPTED_BASELINE fixed repository schema drifted"
        )
    _require_sha256(
        baseline["expected_composite_audit_sha256"],
        "accepted baseline expected_composite_audit_sha256",
    )

    prompt, prompt_raw, prompt_relative = _read_fixed_json("PROMPT_PACK")
    prompt_body = dict(prompt)
    prompt_self_hash = prompt_body.pop("prompt_pack_contract_sha256", None)
    if (
        prompt_self_hash != canonical_sha256(prompt_body)
        or prompt.get("record_type") != "glm52_family_eval_prompt_pack"
        or prompt.get("prompt_pack_status") != "glm52_family_eval_prompt_pack_frozen"
        or prompt.get("prompt_pack_ready") is not True
        or prompt.get("prompt_row_count") != 66
        or prompt.get("prompt_pack_frozen_before_candidate_metrics") is not True
        or prompt.get("holdout_tuning_forbidden") is not True
    ):
        raise Task13SourceBundleError("PROMPT_PACK fixed repository contract drifted")

    training_module, training_builder_sha256 = _load_training_builder()
    try:
        training = training_module.build_sky_training_config()
        training = training_module.validate_sky_training_config(training)
    except (AttributeError, TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "TRAINING_CONFIGURATION builder contract failed"
        ) from error
    if (
        type(training) is not dict
        or training.get("automatic_promotion") is not False
        or training.get("model_upload_authorized") is not False
    ):
        raise Task13SourceBundleError(
            "TRAINING_CONFIGURATION permits automatic publication"
        )

    return {
        "BOOTSTRAP_TEMPLATE": _Artifact(
            kind="BOOTSTRAP_TEMPLATE",
            payload=bootstrap,
            source={
                "authority_class": "FIXED_REPOSITORY_ARTIFACT",
                "source_path": bootstrap_relative,
                "source_file_sha256": _sha256(bootstrap_raw),
                "semantic_validation": "EXACT_INERT_BOOTSTRAP_TEMPLATE",
            },
        ),
        "ACCEPTED_BASELINE": _Artifact(
            kind="ACCEPTED_BASELINE",
            payload=baseline,
            source={
                "authority_class": "FIXED_REPOSITORY_ARTIFACT",
                "source_path": baseline_relative,
                "source_file_sha256": _sha256(baseline_raw),
                "semantic_validation": (
                    "FIXED_REPOSITORY_IDENTITY_AND_BASELINE_SCHEMA"
                ),
            },
        ),
        "PROMPT_PACK": _Artifact(
            kind="PROMPT_PACK",
            payload=prompt,
            source={
                "authority_class": "FIXED_REPOSITORY_ARTIFACT",
                "source_path": prompt_relative,
                "source_file_sha256": _sha256(prompt_raw),
                "semantic_validation": (
                    "FIXED_REPOSITORY_IDENTITY_AND_EMBEDDED_CONTRACT"
                ),
            },
        ),
        "TRAINING_CONFIGURATION": _Artifact(
            kind="TRAINING_CONFIGURATION",
            payload=training,
            source={
                "authority_class": "REPOSITORY_BUILDER",
                "source_path": _TRAINING_BUILDER_RELATIVE,
                "source_file_sha256": training_builder_sha256,
                "semantic_validation": "EXACT_REPOSITORY_BUILDER_VALIDATOR",
            },
        ),
    }


def _live_artifacts(
    live_sources: Mapping[str, PinnedJsonSource],
) -> tuple[dict[str, _Artifact], dict[str, object], object, object]:
    if type(live_sources) is not dict or set(live_sources) != set(LIVE_SOURCE_KINDS):
        raise Task13SourceBundleError(
            "live source kinds must be exact: " + ",".join(LIVE_SOURCE_KINDS)
        )
    values: dict[str, dict[str, object]] = {}
    artifacts: dict[str, _Artifact] = {}
    for kind in LIVE_SOURCE_KINDS:
        source = live_sources[kind]
        value, _raw = _read_pinned_json(source, kind)
        values[kind] = value
        artifacts[kind] = _Artifact(
            kind=kind,
            payload=value,
            source={
                "authority_class": "EXPLICIT_PINNED_LIVE_SOURCE",
                "source_path": str(source.path),
                "source_file_sha256": source.expected_file_sha256,
                "source_body_sha256": source.expected_body_sha256,
                "semantic_validation": (
                    "EXTERNAL_REVIEW_IDENTITY_ONLY"
                    if kind
                    in {
                        "TASK11_REVIEW_APPROVAL",
                        "TASK12_REVIEW_APPROVAL",
                    }
                    else "EXACT_REPOSITORY_CONTRACT"
                ),
            },
        )

    try:
        validate_gpu_spend_approval(values["GPU_SPEND_APPROVAL"])
    except (TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "GPU_SPEND_APPROVAL semantic validation failed"
        ) from error

    campaign = values["CAMPAIGN_DESCRIPTOR"]
    if campaign.get("run_id") != RUN_ID:
        raise Task13SourceBundleError("CAMPAIGN_DESCRIPTOR run_id is not exact")
    try:
        validate_sky_campaign_descriptor(campaign)
    except (TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "CAMPAIGN_DESCRIPTOR semantic validation failed"
        ) from error

    task_inputs_value = values["TASK10_TASK_INPUTS"]
    if set(task_inputs_value) != set(MountFreeTaskInputs.__dataclass_fields__):
        raise Task13SourceBundleError("TASK10_TASK_INPUTS schema drifted")
    try:
        task_inputs = validate_task_inputs(MountFreeTaskInputs(**task_inputs_value))
    except (TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "TASK10_TASK_INPUTS semantic validation failed"
        ) from error
    try:
        worker = worker_bootstrap_descriptor_from_mapping(
            values["TASK10_WORKER_DESCRIPTOR"]
        )
    except (TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "TASK10_WORKER_DESCRIPTOR semantic validation failed"
        ) from error
    return artifacts, campaign, task_inputs, worker


def _validate_task10_bindings(
    *,
    campaign: Mapping[str, object],
    campaign_file_sha256: str,
    worker_file_sha256: str,
    task_inputs: object,
    worker: object,
) -> None:
    expected_worker_uri = (
        "s3://keep-glm52-models-246813579024-us-west-2/"
        "task13/production/task10-worker-descriptor.json"
    )
    expected_campaign_uri = (
        f"s3://{campaign['bucket']}/{campaign['campaign_descriptor_key']}"
    )
    expected = {
        "task input descriptor file SHA-256": (
            task_inputs.descriptor_file_sha256,
            worker_file_sha256,
        ),
        "task input worker descriptor S3 URI": (
            task_inputs.descriptor_s3_uri,
            expected_worker_uri,
        ),
        "worker descriptor file SHA-256": (
            worker.base_descriptor_file_sha256,
            campaign_file_sha256,
        ),
        "worker descriptor body SHA-256": (
            worker.base_descriptor_body_sha256,
            campaign["descriptor_body_sha256"],
        ),
        "worker campaign identity": (
            worker.campaign_identity_sha256,
            campaign["campaign_identity_sha256"],
        ),
        "base campaign descriptor S3 URI": (
            worker.base_descriptor_s3_uri,
            expected_campaign_uri,
        ),
        "archive identity": (
            worker.archive_identity_sha256,
            task_inputs.repository_archive_file_sha256,
        ),
        "archive VersionId": (
            worker.repository_archive_version_id,
            task_inputs.repository_archive_version_id,
        ),
        "approval identity": (
            worker.approval_identity_sha256,
            task_inputs.approval_file_sha256,
        ),
        "approval VersionId": (
            worker.approval_version_id,
            task_inputs.approval_version_id,
        ),
        "intent identity": (
            worker.intent_identity_sha256,
            task_inputs.intent_body_sha256,
        ),
        "intent VersionId": (
            worker.intent_version_id,
            task_inputs.intent_version_id,
        ),
    }
    for label, pair in expected.items():
        if pair[0] != pair[1]:
            raise Task13SourceBundleError(
                "Task 10 " + label + " drifted from campaign descriptor inputs"
            )


def _approval_artifacts(
    owner_approval_source: PinnedFileSource,
) -> dict[str, _Artifact]:
    owner_raw = _read_pinned_file(
        owner_approval_source,
        "owner approval source",
    )
    if (
        owner_approval_source.path != _REPO_ROOT / _OWNER_APPROVAL_RELATIVE
        or owner_approval_source.expected_file_sha256 != _OWNER_APPROVAL_SHA256
    ):
        raise Task13SourceBundleError(
            "owner approval source is not the fixed repository authority"
        )
    support = build_production_support_plane_approval(owner_raw)
    residual = build_gpu_residual_liability_approval(owner_raw)
    try:
        validate_production_support_plane_approval(support, owner_raw)
        validate_gpu_residual_liability_approval(residual, owner_raw)
    except (TypeError, ValueError) as error:
        raise Task13SourceBundleError(
            "owner approval builder contract failed"
        ) from error
    common = {
        "authority_class": "OWNER_APPROVAL_BUILDER",
        "source_path": str(owner_approval_source.path),
        "source_file_sha256": _sha256(owner_raw),
        "semantic_validation": "EXACT_OWNER_APPROVAL_TERMS",
    }
    return {
        "SUPPORT_APPROVAL": _Artifact(
            kind="SUPPORT_APPROVAL",
            payload=support,
            source=dict(common),
        ),
        "RESIDUAL_LIABILITY_APPROVAL": _Artifact(
            kind="RESIDUAL_LIABILITY_APPROVAL",
            payload=residual,
            source=dict(common),
        ),
    }


def _write_once(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _validate_output_directory(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        raise Task13SourceBundleError(
            "output directory must be one new absolute directory"
        )
    if path.exists() or path.is_symlink():
        raise Task13SourceBundleError(
            "output directory must be one new absolute directory"
        )
    parent = path.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
        or path.resolve(strict=False) != path
    ):
        raise Task13SourceBundleError(
            "output directory must be one new absolute canonical directory"
        )


def build_task13_source_bundle(
    *,
    output_directory: Path,
    owner_approval_source: PinnedFileSource,
    live_sources: Mapping[str, PinnedJsonSource],
) -> dict[str, object]:
    """Validate and materialize all non-dynamic reviewed Task 13 sources."""

    _validate_output_directory(output_directory)
    fixed = _fixed_artifacts()
    owner = _approval_artifacts(owner_approval_source)
    live, campaign, task_inputs, worker = _live_artifacts(live_sources)
    all_artifacts = {**fixed, **owner, **live}
    if set(all_artifacts) != set(ARTIFACT_KINDS):
        raise Task13SourceBundleError("source artifact inventory is incomplete")

    campaign_raw = (
        canonical_json_bytes(all_artifacts["CAMPAIGN_DESCRIPTOR"].payload) + b"\n"
    )
    worker_raw = (
        canonical_json_bytes(all_artifacts["TASK10_WORKER_DESCRIPTOR"].payload) + b"\n"
    )
    _validate_task10_bindings(
        campaign=campaign,
        campaign_file_sha256=_sha256(campaign_raw),
        worker_file_sha256=_sha256(worker_raw),
        task_inputs=task_inputs,
        worker=worker,
    )

    artifact_rows: list[dict[str, object]] = []
    encoded: list[tuple[str, bytes]] = []
    for kind in ARTIFACT_KINDS:
        artifact = all_artifacts[kind]
        body = canonical_json_bytes(artifact.payload)
        raw = body + b"\n"
        filename = ARTIFACT_FILENAMES[kind]
        encoded.append((filename, raw))
        artifact_rows.append(
            {
                "artifact_kind": kind,
                "reviewed_artifact_kind": (
                    "PRODUCTION_DESCRIPTOR" if kind == "CAMPAIGN_DESCRIPTOR" else kind
                ),
                "relative_path": filename,
                "file_sha256": _sha256(raw),
                "body_sha256": _sha256(body),
                "source": artifact.source,
            }
        )
    manifest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_local_source_bundle_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "automatic_promotion": False,
        "model_upload_authorized": False,
        "artifacts": artifact_rows,
        "dynamic_outputs": [
            {
                "artifact_kind": kind,
                "classification": (
                    "DYNAMIC_DEPLOYMENT_OR_REHEARSAL_OUTPUT"
                ),
                **(
                    {
                        "key_prefix": DYNAMIC_OUTPUT_KEY_PREFIXES[kind],
                        "materialized_by_source_bundle": False,
                    }
                    if kind in DYNAMIC_OUTPUT_KEY_PREFIXES
                    else {}
                ),
            }
            for kind in DYNAMIC_OUTPUT_KINDS
        ],
        "unclosed_source_semantics": [
            {
                "artifact_kind": "TASK11_REVIEW_APPROVAL",
                "reason": "NO_REPOSITORY_SEMANTIC_SCHEMA",
            },
            {
                "artifact_kind": "TASK12_REVIEW_APPROVAL",
                "reason": "NO_REPOSITORY_SEMANTIC_SCHEMA",
            },
        ],
    }
    manifest = {
        **manifest_body,
        "canonical_identity_sha256": canonical_sha256(manifest_body),
    }
    manifest_raw = canonical_json_bytes(manifest) + b"\n"

    try:
        os.mkdir(output_directory, 0o700)
        os.chmod(output_directory, 0o700)
        for filename, raw in encoded:
            _write_once(output_directory / filename, raw)
        _write_once(output_directory / MANIFEST_NAME, manifest_raw)
        directory = os.open(output_directory, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise Task13SourceBundleError(
            "write-once source bundle materialization failed"
        ) from error
    return manifest
