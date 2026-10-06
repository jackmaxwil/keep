"""Authenticated runtime binding for GLM-5.2 low-rank adapter sidecars."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_vq.quality.glm52_adapter_training import ValidatedGLM52Adapter


def _load_adapter_training_api() -> Any:
    return importlib.import_module("mlx_vq.quality.glm52_adapter_training")


def bind_authenticated_glm52_adapter_sidecars(
    model: Any,
    adapter_sidecar_dir: str | Path,
    *,
    expected_adapter_manifest_sha256: str,
    expected_parent_candidate_identity_sha256: str,
    expected_num_experts: int,
) -> ValidatedGLM52Adapter:
    """Authenticate an adapter overlay and bind its exact routed projections."""

    training = _load_adapter_training_api()
    training._require_sha256(
        expected_adapter_manifest_sha256,
        label="expected adapter manifest",
    )
    training._require_sha256(
        expected_parent_candidate_identity_sha256,
        label="expected adapter parent candidate identity",
    )
    root = Path(adapter_sidecar_dir)
    raw = training._read_regular_file(
        root / training.ADAPTER_MANIFEST,
        label="adapter manifest",
    )
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("adapter manifest must contain valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("adapter manifest must be an object")
    body_sha256 = training._sha256_bytes(
        training._canonical_bytes(training._manifest_body(manifest))
    )
    if (
        manifest.get("manifest_body_sha256") != body_sha256
        or body_sha256 != expected_adapter_manifest_sha256
    ):
        raise ValueError("adapter manifest body SHA-256 mismatch")
    parent_sha256 = manifest.get("parent_candidate_identity_sha256")
    if parent_sha256 != expected_parent_candidate_identity_sha256:
        raise ValueError("adapter parent candidate identity mismatch")
    teacher_sha256 = manifest.get("teacher_manifest_body_sha256")
    candidate_sha256 = manifest.get("candidate_identity_sha256")
    training._require_sha256(teacher_sha256, label="adapter teacher manifest body")
    training._require_sha256(candidate_sha256, label="adapter candidate identity")

    validated = training.load_authenticated_glm52_adapter(
        root,
        expected_parent_candidate_identity_sha256=parent_sha256,
        expected_teacher_manifest_body_sha256=teacher_sha256,
        expected_manifest_body_sha256=body_sha256,
        expected_candidate_identity_sha256=candidate_sha256,
        expected_num_experts=expected_num_experts,
    )
    projection_map = {
        key: training.target_glm52_projection(
            model,
            layer=key[0],
            projection=key[1],
        )
        for key in validated.sidecar_paths
    }
    runtime_inventory = training.attest_glm52_projection_inventory(projection_map)
    declared_inventory = manifest.get("routed_projection_inventory")
    if declared_inventory != {
        "count": runtime_inventory.projection_count,
        "metadata_sha256": runtime_inventory.projection_metadata_sha256,
    }:
        raise ValueError("adapter routed projection inventory does not match runtime")
    training.bind_glm52_low_rank_adapters(projection_map, validated)
    return validated


__all__ = ["bind_authenticated_glm52_adapter_sidecars"]
