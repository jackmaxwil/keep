"""Authenticated GLM-5.2 router traces and source/candidate diagnostics.

This module deliberately has no MLX dependency. Runtime code converts the source
runner's ``LayerRouteTrace`` objects or candidate gate ``(inds, scores)`` tuples
to :class:`RouteTraceLayer`; capture, audit, and comparison stay cheap and can
run on a non-Metal host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np


TRACE_SCHEMA_VERSION: Final = 1
TRACE_RECORD_TYPE: Final = "glm52_route_trace_v1"
RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION: Final = 2
RECOVERY_CANDIDATE_TRACE_RECORD_TYPE: Final = "glm52_recovery_candidate_route_trace_v2"
TRACE_MANIFEST_FILENAME: Final = "glm52-route-trace-manifest.json"
TRACE_LAYER_DIRECTORY: Final = "layers"
EVIDENCE_SCHEMA_VERSION: Final = 1
EVIDENCE_RECORD_TYPE: Final = "glm52_route_math_diagnostics_v1"
RECOVERY_EVIDENCE_SCHEMA_VERSION: Final = 2
RECOVERY_EVIDENCE_RECORD_TYPE: Final = "glm52_recovery_route_math_diagnostics_v2"
TOP_K: Final = 8
FROZEN_N_VALID: Final = 744
FROZEN_SPARSE_LAYERS: Final = tuple(range(3, 78))
DEFAULT_ROUTED_SCALING_FACTOR: Final = 2.5
DEFAULT_EXPERT_COUNT: Final = 168
PINNED_POLICY_FILE_SHA256: Final = "0975f7dc1117c5fba7532e9166f4767546fd691a6cb52520a40f09874f972ce2"
PINNED_PROMPT_PACK_SHA256: Final = "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31"
PINNED_CANDIDATE_COMPOSITE_IDENTITY_SHA256: Final = "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
AUTHORITY_FIELDS: Final = frozenset(
    {
        "teacher_cache_path",
        "teacher_cache_content_sha256",
        "candidate_composite_path",
        "candidate_composite_identity_sha256",
        "prompt_pack_path",
        "prompt_pack_sha256",
        "model_id",
        "model_revision",
        "producer_implementation_id",
        "capture_output_path",
        "capture_output_sha256",
        "evidence_class",
    }
)
RECOVERY_CANDIDATE_AUTHORITY_FIELDS: Final = frozenset(
    (AUTHORITY_FIELDS - {"candidate_composite_path", "candidate_composite_identity_sha256"})
    | {
        "accepted_baseline_composite_path",
        "accepted_baseline_composite_identity_sha256",
        "candidate_cache_identity_sha256",
        "recovery_mixed_artifact_identity_sha256",
        "recovery_manifest_body_sha256",
    }
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {error}") from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_publish_bytes(final_path: Path, payload: bytes) -> None:
    if final_path.exists() or final_path.is_symlink():
        raise ValueError(f"refusing to overwrite existing final file {final_path}")
    parent = final_path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"publication directory is not a real directory: {parent}")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if final_path.exists() or final_path.is_symlink():
            raise ValueError(
                f"refusing to overwrite concurrently created final file {final_path}"
            )
        os.rename(temp_path, final_path)
        _fsync_directory(parent)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class RouteTraceLayer:
    layer_index: int
    expert_ids: np.ndarray
    scores: np.ndarray
    valid_assignment_count: int
    padded_assignment_count: int = 0


@dataclass(frozen=True)
class AuditedRouteTrace:
    root: Path
    manifest: dict[str, Any]
    layers: tuple[RouteTraceLayer, ...]


def candidate_gate_output_to_trace(
    layer_index: int,
    gate_output: tuple[Any, Any],
) -> RouteTraceLayer:
    """Convert ``Glm52VQMoE.gate`` output ``(inds, scores)`` to a trace row."""

    if not isinstance(gate_output, tuple) or len(gate_output) != 2:
        raise ValueError("candidate gate output must be an (inds, scores) tuple")
    ids, scores = gate_output
    ids_array = np.asarray(ids)
    scores_array = np.asarray(scores)
    if ids_array.ndim != 2 or ids_array.shape[1] != TOP_K:
        raise ValueError("candidate expert IDs must have shape [n_valid, 8]")
    return RouteTraceLayer(
        layer_index=layer_index,
        expert_ids=ids_array,
        scores=scores_array,
        valid_assignment_count=int(ids_array.size),
        padded_assignment_count=0,
    )


def source_trace_to_layers(traces: Sequence[Any]) -> tuple[RouteTraceLayer, ...]:
    """Copy source-runner ``LayerRouteTrace`` objects into the stable contract."""

    return tuple(
        RouteTraceLayer(
            layer_index=int(trace.layer_index),
            expert_ids=np.asarray(trace.expert_ids),
            scores=np.asarray(trace.scores),
            valid_assignment_count=int(trace.valid_assignment_count),
            padded_assignment_count=int(trace.padded_assignment_count),
        )
        for trace in traces
    )


def _normalize_layer(layer: RouteTraceLayer) -> RouteTraceLayer:
    if type(layer.layer_index) is not int or layer.layer_index < 0:
        raise ValueError("route trace layer index must be a non-negative integer")
    ids = np.asarray(layer.expert_ids)
    scores = np.asarray(layer.scores)
    if ids.ndim != 2 or ids.shape[1] != TOP_K:
        raise ValueError("expert IDs must have shape [n_valid, 8]")
    if scores.shape != ids.shape:
        raise ValueError("route scores must match expert ID shape")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("expert IDs must use an integer dtype")
    if type(layer.valid_assignment_count) is not int or layer.valid_assignment_count < 0:
        raise ValueError("valid assignment count must be a non-negative integer")
    if type(layer.padded_assignment_count) is not int or layer.padded_assignment_count < 0:
        raise ValueError("padded assignment count must be a non-negative integer")
    return RouteTraceLayer(
        layer_index=layer.layer_index,
        expert_ids=np.ascontiguousarray(ids, dtype="<i4"),
        scores=np.ascontiguousarray(scores, dtype="<f4"),
        valid_assignment_count=layer.valid_assignment_count,
        padded_assignment_count=layer.padded_assignment_count,
    )


def _layer_safetensors_bytes(layer: RouteTraceLayer) -> bytes:
    ids_raw = layer.expert_ids.tobytes(order="C")
    scores_raw = layer.scores.tobytes(order="C")
    header = _canonical_json_bytes(
        {
            "expert_ids": {
                "data_offsets": [0, len(ids_raw)],
                "dtype": "I32",
                "shape": list(layer.expert_ids.shape),
            },
            "scores": {
                "data_offsets": [len(ids_raw), len(ids_raw) + len(scores_raw)],
                "dtype": "F32",
                "shape": list(layer.scores.shape),
            },
        }
    )
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header + ids_raw + scores_raw


def _manifest_content_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "side": manifest["side"],
            "contract": manifest["contract"],
            "authority": manifest["authority"],
            "layers": [
                {
                    "layer_index": row["layer_index"],
                    "relative_path": row["relative_path"],
                    "file_sha256": row["file_sha256"],
                    "expert_ids_sha256": row["expert_ids_sha256"],
                    "scores_sha256": row["scores_sha256"],
                }
                for row in manifest["layers"]
            ],
        }
    )


def _manifest_body_sha256(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_body_sha256", None)
    return canonical_sha256(body)


def validate_capture_authority(value: Any, *, label: str = "route trace authority") -> dict[str, Any]:
    authority = _require_keys(value, set(AUTHORITY_FIELDS), label=label)
    for field in (
        "teacher_cache_path",
        "candidate_composite_path",
        "prompt_pack_path",
        "model_id",
        "model_revision",
        "producer_implementation_id",
        "capture_output_path",
    ):
        if not isinstance(authority[field], str) or not authority[field].strip():
            raise ValueError(f"{label} {field} must be a non-empty string")
    for field in (
        "teacher_cache_content_sha256",
        "candidate_composite_identity_sha256",
        "prompt_pack_sha256",
        "capture_output_sha256",
    ):
        digest = authority[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} {field} must be a lowercase SHA-256")
    if authority["evidence_class"] not in {"release", "fixture_only"}:
        raise ValueError(f"{label} evidence_class must be 'release' or 'fixture_only'")
    if authority["evidence_class"] == "release":
        if authority["candidate_composite_identity_sha256"] != PINNED_CANDIDATE_COMPOSITE_IDENTITY_SHA256:
            raise ValueError(f"{label} candidate composite identity is not release-pinned")
        if authority["prompt_pack_sha256"] != PINNED_PROMPT_PACK_SHA256:
            raise ValueError(f"{label} prompt pack identity is not release-pinned")
    return dict(authority)


def _require_lowercase_sha256(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def validate_recovery_candidate_authority(
    value: Any, *, label: str = "route trace authority"
) -> dict[str, Any]:
    authority = _require_keys(
        value, set(RECOVERY_CANDIDATE_AUTHORITY_FIELDS), label=label
    )
    for field in (
        "teacher_cache_path",
        "accepted_baseline_composite_path",
        "prompt_pack_path",
        "model_id",
        "model_revision",
        "producer_implementation_id",
        "capture_output_path",
    ):
        if not isinstance(authority[field], str) or not authority[field].strip():
            raise ValueError(f"{label} {field} must be a non-empty string")
    for field in (
        "teacher_cache_content_sha256",
        "accepted_baseline_composite_identity_sha256",
        "candidate_cache_identity_sha256",
        "recovery_mixed_artifact_identity_sha256",
        "recovery_manifest_body_sha256",
        "prompt_pack_sha256",
        "capture_output_sha256",
    ):
        _require_lowercase_sha256(authority[field], label=f"{label} {field}")
    if authority["evidence_class"] not in {"release", "fixture_only"}:
        raise ValueError(f"{label} evidence_class must be 'release' or 'fixture_only'")
    if authority["evidence_class"] == "release":
        if (
            authority["accepted_baseline_composite_identity_sha256"]
            != PINNED_CANDIDATE_COMPOSITE_IDENTITY_SHA256
        ):
            raise ValueError(
                f"{label} accepted baseline composite identity is not release-pinned"
            )
        if authority["prompt_pack_sha256"] != PINNED_PROMPT_PACK_SHA256:
            raise ValueError(f"{label} prompt pack identity is not release-pinned")
    return dict(authority)


def _validate_authority_for_trace(
    value: Any, *, schema_version: int, side: str, label: str
) -> dict[str, Any]:
    if schema_version == TRACE_SCHEMA_VERSION:
        return validate_capture_authority(value, label=label)
    if schema_version != RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported route trace schema version")
    if side != "candidate":
        raise ValueError("route trace schema v2 is restricted to the candidate side")
    return validate_recovery_candidate_authority(value, label=label)


def write_route_trace_artifact(
    root: str | Path,
    layers: Sequence[RouteTraceLayer],
    *,
    side: str,
    authority: Mapping[str, Any] | None = None,
    routed_scaling_factor: float = DEFAULT_ROUTED_SCALING_FACTOR,
    expert_count: int = DEFAULT_EXPERT_COUNT,
    created_at: str | None = None,
    schema_version: int = TRACE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Durably publish authenticated per-layer route safetensors and manifest."""

    if side not in {"source", "candidate"}:
        raise ValueError("route trace side must be 'source' or 'candidate'")
    if type(schema_version) is not int or schema_version not in {
        TRACE_SCHEMA_VERSION,
        RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported route trace schema version")
    validated_authority = _validate_authority_for_trace(
        authority or {},
        schema_version=schema_version,
        side=side,
        label="route trace authority",
    )
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError("routed scaling factor must be positive and finite")
    if type(expert_count) is not int or expert_count <= 0:
        raise ValueError("expert count must be a positive integer")
    normalized = tuple(_normalize_layer(layer) for layer in layers)
    if not normalized:
        raise ValueError("route trace must contain at least one sparse layer")
    indexes = [layer.layer_index for layer in normalized]
    if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
        raise ValueError("route trace layer indexes must be unique and increasing")
    n_valid_values = {layer.expert_ids.shape[0] for layer in normalized}
    if len(n_valid_values) != 1:
        raise ValueError("all route trace layers must have the same n_valid")

    output = Path(root)
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to overwrite route trace root {output}")
    output.mkdir(parents=True)
    layer_dir = output / TRACE_LAYER_DIRECTORY
    layer_dir.mkdir()
    _fsync_directory(output)

    records: list[dict[str, Any]] = []
    for layer in normalized:
        filename = f"layer-{layer.layer_index:05d}.safetensors"
        path = layer_dir / filename
        payload = _layer_safetensors_bytes(layer)
        _durable_publish_bytes(path, payload)
        records.append(
            {
                "layer_index": layer.layer_index,
                "relative_path": f"{TRACE_LAYER_DIRECTORY}/{filename}",
                "n_valid": int(layer.expert_ids.shape[0]),
                "top_k": TOP_K,
                "valid_assignment_count": layer.valid_assignment_count,
                "padded_assignment_count": layer.padded_assignment_count,
                "file_size_bytes": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "expert_ids_sha256": hashlib.sha256(
                    layer.expert_ids.tobytes(order="C")
                ).hexdigest(),
                "scores_sha256": hashlib.sha256(
                    layer.scores.tobytes(order="C")
                ).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "record_type": (
            TRACE_RECORD_TYPE
            if schema_version == TRACE_SCHEMA_VERSION
            else RECOVERY_CANDIDATE_TRACE_RECORD_TYPE
        ),
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "side": side,
        "contract": {
            "expert_ids_dtype": "int32",
            "scores_dtype": "float32",
            "top_k": TOP_K,
            "expert_count": expert_count,
            "routed_scaling_factor": routed_scaling_factor,
            "score_normalization": "sum_equals_routed_scaling_factor",
            "rank_order": "nonincreasing_score",
        },
        "authority": validated_authority,
        "layer_count": len(records),
        "n_valid": next(iter(n_valid_values)),
        "layers": records,
        "trace_content_sha256": "",
        "manifest_body_sha256": "",
    }
    manifest["trace_content_sha256"] = _manifest_content_sha256(manifest)
    manifest["manifest_body_sha256"] = _manifest_body_sha256(manifest)
    _durable_publish_bytes(
        output / TRACE_MANIFEST_FILENAME,
        _canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _require_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} key inventory mismatch: {actual}")
    return value


def _read_layer_file(path: Path, record: Mapping[str, Any]) -> RouteTraceLayer:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["file_sha256"]:
        raise ValueError(f"route layer file SHA-256 mismatch: {path}")
    if len(raw) != record["file_size_bytes"] or len(raw) < 8:
        raise ValueError(f"route layer physical size mismatch: {path}")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length <= 0 or 8 + header_length > len(raw):
        raise ValueError(f"invalid safetensors header length: {path}")
    header_raw = raw[8 : 8 + header_length]
    stripped = header_raw.rstrip(b" ")
    if header_raw[len(stripped) :] != b" " * (len(header_raw) - len(stripped)):
        raise ValueError(f"invalid safetensors header padding: {path}")
    header = _strict_json_bytes(stripped, label=f"safetensors header {path}")
    header = _require_keys(header, {"expert_ids", "scores"}, label="tensor inventory")
    n_valid = record["n_valid"]
    ids_size = n_valid * TOP_K * 4
    scores_size = ids_size
    expected = {
        "expert_ids": {
            "data_offsets": [0, ids_size],
            "dtype": "I32",
            "shape": [n_valid, TOP_K],
        },
        "scores": {
            "data_offsets": [ids_size, ids_size + scores_size],
            "dtype": "F32",
            "shape": [n_valid, TOP_K],
        },
    }
    if header != expected:
        raise ValueError(f"route layer safetensors contract mismatch: {path}")
    payload = raw[8 + header_length :]
    if len(payload) != ids_size + scores_size:
        raise ValueError(f"route layer payload extent mismatch: {path}")
    ids_raw, scores_raw = payload[:ids_size], payload[ids_size:]
    if hashlib.sha256(ids_raw).hexdigest() != record["expert_ids_sha256"]:
        raise ValueError(f"expert IDs SHA-256 mismatch: {path}")
    if hashlib.sha256(scores_raw).hexdigest() != record["scores_sha256"]:
        raise ValueError(f"route scores SHA-256 mismatch: {path}")
    return RouteTraceLayer(
        layer_index=record["layer_index"],
        expert_ids=np.frombuffer(ids_raw, dtype="<i4").reshape(n_valid, TOP_K).copy(),
        scores=np.frombuffer(scores_raw, dtype="<f4").reshape(n_valid, TOP_K).copy(),
        valid_assignment_count=record["valid_assignment_count"],
        padded_assignment_count=record["padded_assignment_count"],
    )


def audit_route_trace_artifact(
    root: str | Path,
    *,
    expected_side: str | None = None,
) -> AuditedRouteTrace:
    """Recompute the exact trace tree, manifest, safetensor, and payload hashes."""

    output = Path(root)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("route trace root must be a real directory")
    manifest_path = output / TRACE_MANIFEST_FILENAME
    layer_dir = output / TRACE_LAYER_DIRECTORY
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("route trace manifest must be a regular file")
    if layer_dir.is_symlink() or not layer_dir.is_dir():
        raise ValueError("route trace layers must be a real directory")
    if {entry.name for entry in output.iterdir()} != {
        TRACE_MANIFEST_FILENAME,
        TRACE_LAYER_DIRECTORY,
    }:
        raise ValueError("route trace root contains missing or extra entries")
    raw_manifest = manifest_path.read_bytes()
    manifest = _strict_json_bytes(raw_manifest, label="route trace manifest")
    top_keys = {
        "schema_version", "record_type", "created_at", "side", "contract",
        "authority", "layer_count", "n_valid", "layers",
        "trace_content_sha256", "manifest_body_sha256",
    }
    manifest = _require_keys(manifest, top_keys, label="route trace manifest")
    schema_identity = (manifest["schema_version"], manifest["record_type"])
    if schema_identity not in {
        (TRACE_SCHEMA_VERSION, TRACE_RECORD_TYPE),
        (
            RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION,
            RECOVERY_CANDIDATE_TRACE_RECORD_TYPE,
        ),
    }:
        raise ValueError("route trace contract identity mismatch")
    if manifest["side"] not in {"source", "candidate"}:
        raise ValueError("invalid route trace side")
    if expected_side is not None and manifest["side"] != expected_side:
        raise ValueError("route trace side does not match expected side")
    if (
        manifest["schema_version"] == RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION
        and manifest["side"] != "candidate"
    ):
        raise ValueError("route trace schema v2 is restricted to the candidate side")
    contract = _require_keys(
        manifest["contract"],
        {"expert_ids_dtype", "scores_dtype", "top_k", "expert_count", "routed_scaling_factor", "score_normalization", "rank_order"},
        label="route trace contract",
    )
    if contract["expert_ids_dtype"] != "int32" or contract["scores_dtype"] != "float32" or contract["top_k"] != TOP_K:
        raise ValueError("route trace dtype/top-k contract mismatch")
    if contract["score_normalization"] != "sum_equals_routed_scaling_factor" or contract["rank_order"] != "nonincreasing_score":
        raise ValueError("route trace router math contract mismatch")
    if type(contract["expert_count"]) is not int or contract["expert_count"] <= 0:
        raise ValueError("invalid route trace expert count")
    if not isinstance(contract["routed_scaling_factor"], (int, float)) or isinstance(contract["routed_scaling_factor"], bool) or not math.isfinite(contract["routed_scaling_factor"]) or contract["routed_scaling_factor"] <= 0:
        raise ValueError("invalid routed scaling factor")
    _validate_authority_for_trace(
        manifest["authority"],
        schema_version=manifest["schema_version"],
        side=manifest["side"],
        label="route trace authority",
    )
    if not isinstance(manifest["layers"], list) or not manifest["layers"]:
        raise ValueError("route trace layers must be a non-empty list")
    if manifest["layer_count"] != len(manifest["layers"]):
        raise ValueError("route trace layer count mismatch")
    if type(manifest["n_valid"]) is not int or manifest["n_valid"] <= 0:
        raise ValueError("route trace n_valid must be positive")
    if manifest["trace_content_sha256"] != _manifest_content_sha256(manifest):
        raise ValueError("route trace content SHA-256 mismatch")
    if manifest["manifest_body_sha256"] != _manifest_body_sha256(manifest):
        raise ValueError("route trace manifest body SHA-256 mismatch")

    record_keys = {
        "layer_index", "relative_path", "n_valid", "top_k",
        "valid_assignment_count", "padded_assignment_count", "file_size_bytes",
        "file_sha256", "expert_ids_sha256", "scores_sha256",
    }
    layers: list[RouteTraceLayer] = []
    expected_names: set[str] = set()
    previous_index = -1
    for raw_record in manifest["layers"]:
        record = _require_keys(raw_record, record_keys, label="route layer record")
        layer_index = record["layer_index"]
        if type(layer_index) is not int or layer_index <= previous_index:
            raise ValueError("route layer indexes must be unique and increasing")
        previous_index = layer_index
        filename = f"layer-{layer_index:05d}.safetensors"
        if record["relative_path"] != f"{TRACE_LAYER_DIRECTORY}/{filename}":
            raise ValueError("route layer relative path mismatch")
        if record["n_valid"] != manifest["n_valid"] or record["top_k"] != TOP_K:
            raise ValueError("route layer shape metadata mismatch")
        for field in ("valid_assignment_count", "padded_assignment_count", "file_size_bytes"):
            if type(record[field]) is not int or record[field] < 0:
                raise ValueError(f"invalid route layer {field}")
        for field in ("file_sha256", "expert_ids_sha256", "scores_sha256"):
            if not isinstance(record[field], str) or len(record[field]) != 64:
                raise ValueError(f"invalid route layer {field}")
        path = layer_dir / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"route layer must be a regular file: {path}")
        layers.append(_read_layer_file(path, record))
        expected_names.add(filename)
    actual_names = {entry.name for entry in layer_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("route layer directory contains missing or extra entries")
    if manifest_path.read_bytes() != raw_manifest:
        raise ValueError("route trace manifest changed during audit")
    return AuditedRouteTrace(output, manifest, tuple(layers))


def _side_checks(audit: AuditedRouteTrace) -> dict[str, bool]:
    contract = audit.manifest["contract"]
    scaling = float(contract["routed_scaling_factor"])
    expert_count = int(contract["expert_count"])
    assignment = all(
        layer.valid_assignment_count == layer.expert_ids.shape[0] * TOP_K
        for layer in audit.layers
    )
    padded = all(layer.padded_assignment_count == 0 for layer in audit.layers)
    finite = all(np.isfinite(layer.scores).all() for layer in audit.layers)
    normalized = all(
        np.allclose(
            layer.scores.sum(axis=1, dtype=np.float64),
            scaling,
            rtol=1e-5,
            atol=1e-5,
        )
        for layer in audit.layers
    )
    monotone = all(
        np.all(np.diff(layer.scores.astype(np.float64), axis=1) <= 1e-7)
        for layer in audit.layers
    )
    unique = all(
        all(len(set(row.tolist())) == TOP_K for row in layer.expert_ids)
        for layer in audit.layers
    )
    ids_in_range = all(
        np.all((layer.expert_ids >= 0) & (layer.expert_ids < expert_count))
        for layer in audit.layers
    )
    result = {
        "assignment_count_pass": bool(assignment),
        "zero_padded_assignments_pass": bool(padded),
        "scores_finite_pass": bool(finite),
        "scores_normalized_pass": bool(normalized),
        "monotone_rank_order_pass": bool(monotone),
        "unique_top8_experts_pass": bool(unique),
        "expert_ids_in_range_pass": bool(ids_in_range),
    }
    result["structural_math_pass"] = all(result.values())
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64).reshape(-1)
    right64 = right.astype(np.float64).reshape(-1)
    left_centered = left64 - left64.mean()
    right_centered = right64 - right64.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return 1.0 if np.array_equal(left64, right64) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _entropy(scores: np.ndarray) -> np.ndarray:
    values = scores.astype(np.float64)
    totals = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals != 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
    return -terms.sum(axis=1)


def _layer_metrics(source: RouteTraceLayer, candidate: RouteTraceLayer) -> dict[str, Any]:
    set_matches = [
        len(set(source_row.tolist()) & set(candidate_row.tolist())) / TOP_K
        for source_row, candidate_row in zip(
            source.expert_ids, candidate.expert_ids, strict=True
        )
    ]
    source_entropy = _entropy(source.scores)
    candidate_entropy = _entropy(candidate.scores)
    return {
        "layer_index": source.layer_index,
        "n_valid": int(source.expert_ids.shape[0]),
        "top8_set_agreement": float(np.mean(set_matches)),
        "rank_exact_agreement": float(np.mean(source.expert_ids == candidate.expert_ids)),
        "score_correlation": _pearson(source.scores, candidate.scores),
        "source_route_entropy": float(np.mean(source_entropy)),
        "candidate_route_entropy": float(np.mean(candidate_entropy)),
        "route_entropy_delta": float(np.mean(candidate_entropy - source_entropy)),
    }


def _load_policy(path: str | Path | None) -> tuple[dict[str, Any] | None, None, str | None]:
    if path is None:
        return None, None, None
    policy_path = Path(path)
    raw = policy_path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != PINNED_POLICY_FILE_SHA256:
        raise ValueError("route diagnostics policy does not match pinned file SHA-256")
    policy = _strict_json_bytes(raw, label="route diagnostics policy")
    if not isinstance(policy, dict):
        raise ValueError("route diagnostics policy must be a JSON object")
    policy_body = dict(policy)
    submitted_contract = policy_body.pop("policy_contract_sha256", None)
    if submitted_contract != canonical_sha256(policy_body):
        raise ValueError("route diagnostics policy contract SHA-256 mismatch")
    section = policy.get("route_math_diagnostics")
    if section is None:
        return policy, None, file_sha256
    raise ValueError("pinned family policy does not authenticate route diagnostics thresholds")


def compare_route_trace_artifacts(
    source_root: str | Path,
    candidate_root: str | Path,
    *,
    policy_json: str | Path | None = None,
    require_frozen_batch: bool = False,
    expected_source_authority: Mapping[str, Any],
    expected_candidate_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit and compare source/candidate traces, returning authenticated evidence."""

    source = audit_route_trace_artifact(source_root, expected_side="source")
    candidate = audit_route_trace_artifact(candidate_root, expected_side="candidate")
    source_indexes = [layer.layer_index for layer in source.layers]
    candidate_indexes = [layer.layer_index for layer in candidate.layers]
    inventory_pass = source_indexes == candidate_indexes
    shape_pass = inventory_pass and all(
        left.expert_ids.shape == right.expert_ids.shape
        for left, right in zip(source.layers, candidate.layers, strict=True)
    )
    contract_match = source.manifest["contract"] == candidate.manifest["contract"]
    expected_source = _validate_authority_for_trace(
        expected_source_authority,
        schema_version=source.manifest["schema_version"],
        side="source",
        label="expected source authority",
    )
    expected_candidate = _validate_authority_for_trace(
        expected_candidate_authority,
        schema_version=candidate.manifest["schema_version"],
        side="candidate",
        label="expected candidate authority",
    )
    source_authority_pass = source.manifest["authority"] == expected_source
    candidate_authority_pass = candidate.manifest["authority"] == expected_candidate
    authority_match = source_authority_pass and candidate_authority_pass
    if not inventory_pass or not shape_pass:
        raise ValueError("source and candidate route trace layer inventories/shapes differ")
    metrics_by_layer = [
        _layer_metrics(left, right)
        for left, right in zip(source.layers, candidate.layers, strict=True)
    ]
    source_ids = np.concatenate([layer.expert_ids for layer in source.layers])
    candidate_ids = np.concatenate([layer.expert_ids for layer in candidate.layers])
    source_scores = np.concatenate([layer.scores for layer in source.layers])
    candidate_scores = np.concatenate([layer.scores for layer in candidate.layers])
    global_metrics = _layer_metrics(
        RouteTraceLayer(-1, source_ids, source_scores, source_ids.size),
        RouteTraceLayer(-1, candidate_ids, candidate_scores, candidate_ids.size),
    )
    global_metrics.pop("layer_index")
    global_metrics["sparse_layer_count"] = len(source.layers)
    frozen_batch_pass = (
        source_indexes == list(FROZEN_SPARSE_LAYERS)
        and source.manifest["n_valid"] == FROZEN_N_VALID
        and candidate.manifest["n_valid"] == FROZEN_N_VALID
    )
    source_checks = _side_checks(source)
    candidate_checks = _side_checks(candidate)
    structural = (
        source_checks["structural_math_pass"]
        and candidate_checks["structural_math_pass"]
        and inventory_pass
        and shape_pass
        and contract_match
        and authority_match
        and (frozen_batch_pass or not require_frozen_batch)
    )
    policy, thresholds, policy_file_sha256 = _load_policy(policy_json)
    diagnostic_only = True
    threshold_pass = False
    release_eligible = bool(
        authority_match
        and source.manifest["authority"]["evidence_class"] == "release"
        and candidate.manifest["authority"]["evidence_class"] == "release"
        and frozen_batch_pass
    )
    recovery_candidate = (
        candidate.manifest["schema_version"]
        == RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION
    )
    evidence: dict[str, Any] = {
        "schema_version": (
            RECOVERY_EVIDENCE_SCHEMA_VERSION
            if recovery_candidate
            else EVIDENCE_SCHEMA_VERSION
        ),
        "record_type": (
            RECOVERY_EVIDENCE_RECORD_TYPE
            if recovery_candidate
            else EVIDENCE_RECORD_TYPE
        ),
        "source_trace": {
            "root": str(Path(source_root).absolute()),
            "manifest_body_sha256": source.manifest["manifest_body_sha256"],
            "trace_content_sha256": source.manifest["trace_content_sha256"],
            **(
                {
                    "schema_version": source.manifest["schema_version"],
                    "record_type": source.manifest["record_type"],
                }
                if recovery_candidate
                else {}
            ),
        },
        "candidate_trace": {
            "root": str(Path(candidate_root).absolute()),
            "manifest_body_sha256": candidate.manifest["manifest_body_sha256"],
            "trace_content_sha256": candidate.manifest["trace_content_sha256"],
            **(
                {
                    "schema_version": candidate.manifest["schema_version"],
                    "record_type": candidate.manifest["record_type"],
                    **{
                        field: candidate.manifest["authority"][field]
                        for field in (
                            "accepted_baseline_composite_identity_sha256",
                            "candidate_cache_identity_sha256",
                            "recovery_mixed_artifact_identity_sha256",
                            "recovery_manifest_body_sha256",
                        )
                    },
                }
                if recovery_candidate
                else {}
            ),
        },
        "policy": {
            "path": None if policy_json is None else str(Path(policy_json).absolute()),
            "file_sha256": policy_file_sha256,
            "policy_contract_sha256": None if policy is None else policy.get("policy_contract_sha256"),
            "route_math_thresholds": thresholds,
        },
        "evaluation": {
            "diagnostic_only": diagnostic_only,
            "release_eligible": release_eligible,
            "thresholds_present": thresholds is not None,
            "require_frozen_batch": require_frozen_batch,
            **(
                {
                    "recovery_tuning_scope": "per_layer_only",
                    "global_route_metrics_allowed_for_recovery_tuning": False,
                }
                if recovery_candidate
                else {}
            ),
        },
        "metrics": {"per_layer": metrics_by_layer, "global": global_metrics},
        "checks": {
            "source": source_checks,
            "candidate": candidate_checks,
            "layer_inventory_match_pass": inventory_pass,
            "layer_shapes_match_pass": shape_pass,
            "router_contract_match_pass": contract_match,
            "capture_authority_match_pass": authority_match,
            "source_capture_authority_pass": source_authority_pass,
            "candidate_capture_authority_pass": candidate_authority_pass,
            "frozen_66_prompt_batch_pass": frozen_batch_pass,
            "structural_math_pass": bool(structural),
            "release_thresholds_pass": bool(threshold_pass),
            "diagnostics_pass": bool(structural and (threshold_pass or diagnostic_only)),
            "release_gate_pass": False,
        },
        "evidence_body_sha256": "",
    }
    evidence["evidence_body_sha256"] = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_body_sha256"}
    )
    return evidence


def write_route_math_evidence(path: str | Path, evidence: Mapping[str, Any]) -> None:
    body = dict(evidence)
    submitted = body.pop("evidence_body_sha256", None)
    if submitted != canonical_sha256(body):
        raise ValueError("route math evidence body SHA-256 mismatch")
    _durable_publish_bytes(Path(path), _canonical_json_bytes(dict(evidence)) + b"\n")


def load_npz_trace_root(root: str | Path) -> tuple[RouteTraceLayer, ...]:
    """Load runtime exchange files ``layer-XXXXX.npz`` for CLI capture."""

    input_root = Path(root)
    paths = sorted(input_root.glob("layer-*.npz"))
    if not paths:
        raise ValueError("raw trace input root contains no layer-XXXXX.npz files")
    layers: list[RouteTraceLayer] = []
    for path in paths:
        try:
            layer_index = int(path.stem.removeprefix("layer-"))
        except ValueError as error:
            raise ValueError(f"invalid raw trace layer filename {path.name}") from error
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if not {"expert_ids", "scores"} <= keys:
                raise ValueError(f"raw trace {path} lacks expert_ids or scores")
            ids = np.asarray(payload["expert_ids"])
            valid_count = (
                int(np.asarray(payload["valid_assignment_count"]).item())
                if "valid_assignment_count" in keys
                else int(ids.size)
            )
            padded_count = (
                int(np.asarray(payload["padded_assignment_count"]).item())
                if "padded_assignment_count" in keys
                else 0
            )
            layers.append(
                RouteTraceLayer(
                    layer_index,
                    ids,
                    np.asarray(payload["scores"]),
                    valid_count,
                    padded_count,
                )
            )
    return tuple(layers)


__all__ = [
    "AuditedRouteTrace",
    "RouteTraceLayer",
    "RECOVERY_CANDIDATE_AUTHORITY_FIELDS",
    "audit_route_trace_artifact",
    "candidate_gate_output_to_trace",
    "canonical_sha256",
    "compare_route_trace_artifacts",
    "load_npz_trace_root",
    "source_trace_to_layers",
    "validate_recovery_candidate_authority",
    "write_route_math_evidence",
    "write_route_trace_artifact",
]
