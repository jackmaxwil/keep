"""Finalize and audit the sparse GLM-5.2 Teich teacher-signal cache (v3)."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

MANIFEST_FILENAME = "glm52-teacher-signal-cache-v3-manifest.json"
READY_FILENAME = "TEACHER_CACHE_READY.json"
SHARD_DIRECTORY = "teacher_signal"
SPLIT_SEED = 20260712
ROUTER_LAYERS = tuple(range(70, 78))

_EXPECTED_DTYPES = {
    "positions": np.dtype(np.int32),
    "target_token_ids": np.dtype(np.int32),
    "topk_logit_ids": np.dtype(np.int32),
    "topk_logit_values": np.dtype(np.float16),
    "logsumexp": np.dtype(np.float32),
    "tail_mass": np.dtype(np.float16),
    "layer_77_hidden_probe": np.dtype(np.float16),
    "router_top8_expert_ids": np.dtype(np.int32),
    "router_top8_normalized_weights": np.dtype(np.float16),
}


class TeacherCacheV3Error(ValueError):
    """The sparse cache cannot be authenticated against its authorities."""


@dataclass(frozen=True)
class TeacherCacheV3Audit:
    manifest_sha256: str
    manifest_body_sha256: str
    session_count: int
    supervised_positions: int
    split_supervised_positions: dict[str, int]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    authority = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_sha256": _sha256_bytes(array.tobytes()),
    }
    return _canonical_sha256(authority)


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise TeacherCacheV3Error(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _load_json(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    raw = _read_regular(path, label=label)
    try:
        value = json.loads(raw)
    except Exception as error:
        raise TeacherCacheV3Error(f"{label} is malformed: {error}") from error
    if not isinstance(value, dict):
        raise TeacherCacheV3Error(f"{label} must be a JSON object")
    return value, raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise TeacherCacheV3Error(f"refusing to replace immutable shard {path.name}")
    _write_atomic(path, payload)


def _prompt_rows(payload: Mapping[str, object], *, label: str) -> list[dict[str, object]]:
    rows = payload.get("prompt_rows")
    if not isinstance(rows, list):
        raise TeacherCacheV3Error(f"{label} must contain prompt_rows")
    result: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TeacherCacheV3Error(f"{label} prompt_rows[{index}] must be an object")
        result.append(row)
    return result


def _provider(row: Mapping[str, object]) -> str:
    provenance = row.get("provenance")
    value = provenance.get("provider") if isinstance(provenance, dict) else None
    if not isinstance(value, str) or not value:
        raise TeacherCacheV3Error("every Teich row requires provenance.provider")
    return value


def _source_session_id(row: Mapping[str, object]) -> str:
    provenance = row.get("provenance")
    value = provenance.get("source_session_id") if isinstance(provenance, dict) else None
    if not isinstance(value, str) or not value:
        raise TeacherCacheV3Error("every Teich row requires provenance.source_session_id")
    return value


def assign_provider_balanced_splits(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Deterministic provider-stratified weighted 80/10/10 assignment."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_provider(row), []).append(row)
    assigned: dict[str, str] = {}
    fractions = {"train": 0.8, "validation": 0.1, "holdout": 0.1}
    split_order = ("train", "validation", "holdout")
    for provider, provider_rows in sorted(grouped.items()):
        total = sum(len(row.get("positions", [])) for row in provider_rows)
        if total <= 0:
            raise TeacherCacheV3Error(f"provider {provider!r} has no supervised positions")
        targets = {split: total * fraction for split, fraction in fractions.items()}
        observed = {split: 0 for split in fractions}
        ordered = sorted(
            provider_rows,
            key=lambda row: (
                -len(row.get("positions", [])),
                _sha256_bytes(f"{seed}:{row.get('prompt_id')}".encode()),
            ),
        )
        for row in ordered:
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise TeacherCacheV3Error("every Teich row requires a prompt_id")
            weight = len(row.get("positions", []))

            def cost(split: str) -> tuple[float, str]:
                before = observed[split] - targets[split]
                after = observed[split] + weight - targets[split]
                # Validation and holdout have equal targets.  Resolve their
                # otherwise systematic tie from the pinned seed and session
                # identity instead of always starving one split.
                tie = _sha256_bytes(
                    f"{seed}:{provider}:{prompt_id}:{split}".encode()
                )
                return (after * after - before * before, tie)

            selected = min(split_order, key=cost)
            assigned[prompt_id] = selected
            observed[selected] += weight
    return assigned


def _validate_no_frozen_overlap(
    teich_rows: Sequence[Mapping[str, object]],
    frozen_rows: Sequence[Mapping[str, object]],
) -> None:
    frozen_prompt_ids = {row.get("prompt_id") for row in frozen_rows}
    frozen_token_hashes = {row.get("token_ids_sha256") for row in frozen_rows}
    frozen_sessions = {
        row.get("provenance", {}).get("source_session_id")
        for row in frozen_rows
        if isinstance(row.get("provenance"), dict)
    }
    for row in teich_rows:
        if (
            row.get("prompt_id") in frozen_prompt_ids
            or row.get("token_ids_sha256") in frozen_token_hashes
            or _source_session_id(row) in frozen_sessions
        ):
            raise TeacherCacheV3Error("frozen-66 overlap detected in Teich cache authority")


def _expected_shapes(*, positions: int, top_k: int, hidden_size: int) -> dict[str, tuple[int, ...]]:
    return {
        "positions": (positions,),
        "target_token_ids": (positions,),
        "topk_logit_ids": (positions, top_k),
        "topk_logit_values": (positions, top_k),
        "logsumexp": (positions,),
        "tail_mass": (positions,),
        "layer_77_hidden_probe": (positions, hidden_size),
        "router_top8_expert_ids": (len(ROUTER_LAYERS), positions, 8),
        "router_top8_normalized_weights": (len(ROUTER_LAYERS), positions, 8),
    }


def _validate_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    row: Mapping[str, object],
    top_k: int,
    hidden_size: int,
) -> dict[str, np.ndarray]:
    if set(tensors) != set(_EXPECTED_DTYPES):
        raise TeacherCacheV3Error(f"capture tensor inventory mismatch for {row.get('prompt_id')!r}")
    positions = row.get("positions")
    targets = row.get("target_token_ids")
    if not isinstance(positions, list) or not isinstance(targets, list):
        raise TeacherCacheV3Error("prompt row positions/targets must be lists")
    shapes = _expected_shapes(positions=len(positions), top_k=top_k, hidden_size=hidden_size)
    normalized: dict[str, np.ndarray] = {}
    for name, dtype in _EXPECTED_DTYPES.items():
        value = np.ascontiguousarray(tensors[name])
        if value.dtype != dtype or value.shape != shapes[name]:
            raise TeacherCacheV3Error(
                f"capture {row.get('prompt_id')!r} tensor {name} contract mismatch"
            )
        normalized[name] = value
    if not np.array_equal(normalized["positions"], np.asarray(positions, dtype=np.int32)):
        raise TeacherCacheV3Error("capture sparse positions do not match prompt pack")
    if not np.array_equal(normalized["target_token_ids"], np.asarray(targets, dtype=np.int32)):
        raise TeacherCacheV3Error("capture targets do not match prompt pack")
    tail = normalized["tail_mass"].astype(np.float32)
    if not np.all(np.isfinite(tail)) or np.any(tail < 0) or np.any(tail >= 1):
        raise TeacherCacheV3Error("capture tail mass must be finite and in [0, 1)")
    weights = normalized["router_top8_normalized_weights"].astype(np.float32)
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1.0, atol=1e-3):
        raise TeacherCacheV3Error("capture router weights are not normalized")
    return normalized


def _capture_path(capture_dir: Path, prompt_id: str) -> Path:
    return capture_dir / f"{prompt_id.replace('/', '__')}.npz"


def finalize_glm52_teich_teacher_cache(
    *,
    capture_dir: str | Path,
    prompt_pack_path: str | Path,
    frozen_prompt_pack_path: str | Path,
    cache_dir: str | Path,
    expected_session_count: int = 257,
    expected_supervised_positions: int = 2_500_735,
    top_k: int = 2048,
    hidden_size: int = 7168,
    split_seed: int = SPLIT_SEED,
) -> dict[str, object]:
    capture_root = Path(capture_dir)
    root = Path(cache_dir)
    prompt_payload, prompt_raw = _load_json(Path(prompt_pack_path), label="Teich prompt pack")
    frozen_payload, frozen_raw = _load_json(
        Path(frozen_prompt_pack_path), label="frozen-66 prompt pack"
    )
    rows = _prompt_rows(prompt_payload, label="Teich prompt pack")
    frozen_rows = _prompt_rows(frozen_payload, label="frozen-66 prompt pack")
    if len(rows) != expected_session_count:
        raise TeacherCacheV3Error("Teich session count does not match campaign authority")
    total_positions = sum(len(row.get("positions", [])) for row in rows)
    if total_positions != expected_supervised_positions:
        raise TeacherCacheV3Error("Teich supervised-position count does not match campaign authority")
    _validate_no_frozen_overlap(rows, frozen_rows)
    prompt_ids = [row.get("prompt_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in prompt_ids) or len(set(prompt_ids)) != len(rows):
        raise TeacherCacheV3Error("Teich prompt IDs must be unique non-empty strings")
    source_sessions = [_source_session_id(row) for row in rows]
    if len(set(source_sessions)) != len(source_sessions):
        raise TeacherCacheV3Error("Teich source sessions must be disjoint")
    split_by_prompt = assign_provider_balanced_splits(rows, seed=split_seed)

    validated: list[dict[str, np.ndarray]] = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        path = _capture_path(capture_root, prompt_id)
        if not path.exists():
            raise TeacherCacheV3Error(f"capture is missing for {prompt_id!r}")
        if path.is_symlink() or not path.is_file():
            raise TeacherCacheV3Error(f"capture for {prompt_id!r} must be a regular file")
        try:
            with np.load(path, allow_pickle=False) as archive:
                tensors = {name: archive[name] for name in archive.files}
        except Exception as error:
            raise TeacherCacheV3Error(f"capture for {prompt_id!r} is malformed: {error}") from error
        validated.append(_validate_tensors(tensors, row=row, top_k=top_k, hidden_size=hidden_size))

    shard_entries: list[dict[str, object]] = []
    for index, (row, tensors) in enumerate(zip(rows, validated, strict=True)):
        prompt_id = str(row["prompt_id"])
        relative = f"{SHARD_DIRECTORY}/row-{index:05d}-{_sha256_bytes(prompt_id.encode())[:12]}.safetensors"
        raw_shard = save_safetensors(
            tensors,
            metadata={
                "record_type": "glm52_teacher_signal_cache_shard_v3",
                "prompt_id": prompt_id,
                "token_ids_sha256": str(row.get("token_ids_sha256")),
            },
        )
        _write_immutable(root / relative, raw_shard)
        split = split_by_prompt[prompt_id]
        tensor_meta = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
            for name, value in tensors.items()
        }
        shard_entries.append(
            {
                "row_index": index,
                "prompt_id": prompt_id,
                "source_session_id": _source_session_id(row),
                "provider": _provider(row),
                "split": split,
                "tuning_eligible": split == "train",
                "sparse_positions": True,
                "supervised_position_count": len(row["positions"]),
                "token_ids_sha256": row.get("token_ids_sha256"),
                "relative_path": relative,
                "file_sha256": _sha256_bytes(raw_shard),
                "tensors": tensor_meta,
            }
        )
    split_counts = {
        split: sum(
            int(entry["supervised_position_count"])
            for entry in shard_entries
            if entry["split"] == split
        )
        for split in ("train", "validation", "holdout")
    }
    manifest_body: dict[str, object] = {
        "schema_version": 3,
        "record_type": "glm52_teacher_signal_cache",
        "release_eligible": False,
        "personal_training_data": True,
        "prompt_pack_sha256": _sha256_bytes(prompt_raw),
        "frozen_66_prompt_pack_sha256": _sha256_bytes(frozen_raw),
        "session_count": len(rows),
        "supervised_position_count": total_positions,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "router_layers": list(ROUTER_LAYERS),
        "cka_probe_layers": [77],
        "split_policy": {
            "record_type": "provider_balanced_session_disjoint_weighted_v1",
            "seed": split_seed,
            "target_fractions": {"train": 0.8, "validation": 0.1, "holdout": 0.1},
            "weight": "supervised_positions",
            "split_supervised_positions": split_counts,
        },
        "frozen_66_overlap_count": 0,
        "shards": shard_entries,
    }
    manifest = {
        **manifest_body,
        "manifest_body_sha256": _canonical_sha256(manifest_body),
    }
    manifest_raw = _canonical_bytes(manifest) + b"\n"
    _write_atomic(root / MANIFEST_FILENAME, manifest_raw)
    (root / READY_FILENAME).unlink(missing_ok=True)

    audit = _audit(
        cache_dir=root,
        prompt_pack_path=Path(prompt_pack_path),
        frozen_prompt_pack_path=Path(frozen_prompt_pack_path),
        expected_session_count=expected_session_count,
        expected_supervised_positions=expected_supervised_positions,
        top_k=top_k,
        hidden_size=hidden_size,
        require_ready=False,
    )
    inventory_sha256 = _canonical_sha256(
        [
            {"relative_path": entry["relative_path"], "file_sha256": entry["file_sha256"]}
            for entry in shard_entries
        ]
    )
    ready_body: dict[str, object] = {
        "record_type": "glm52_teacher_cache_ready_v3",
        "manifest_filename": MANIFEST_FILENAME,
        "manifest_sha256": audit.manifest_sha256,
        "manifest_body_sha256": audit.manifest_body_sha256,
        "shard_inventory_sha256": inventory_sha256,
        "session_count": audit.session_count,
        "supervised_position_count": audit.supervised_positions,
    }
    ready = {**ready_body, "ready_record_sha256": _canonical_sha256(ready_body)}
    _write_atomic(root / READY_FILENAME, _canonical_bytes(ready) + b"\n")
    _audit(
        cache_dir=root,
        prompt_pack_path=Path(prompt_pack_path),
        frozen_prompt_pack_path=Path(frozen_prompt_pack_path),
        expected_session_count=expected_session_count,
        expected_supervised_positions=expected_supervised_positions,
        top_k=top_k,
        hidden_size=hidden_size,
        require_ready=True,
    )
    return ready


def audit_glm52_teich_teacher_cache(
    *,
    cache_dir: str | Path,
    prompt_pack_path: str | Path,
    frozen_prompt_pack_path: str | Path,
    expected_session_count: int = 257,
    expected_supervised_positions: int = 2_500_735,
    top_k: int = 2048,
    hidden_size: int = 7168,
) -> TeacherCacheV3Audit:
    return _audit(
        cache_dir=Path(cache_dir),
        prompt_pack_path=Path(prompt_pack_path),
        frozen_prompt_pack_path=Path(frozen_prompt_pack_path),
        expected_session_count=expected_session_count,
        expected_supervised_positions=expected_supervised_positions,
        top_k=top_k,
        hidden_size=hidden_size,
        require_ready=True,
    )


def _audit(
    *,
    cache_dir: Path,
    prompt_pack_path: Path,
    frozen_prompt_pack_path: Path,
    expected_session_count: int,
    expected_supervised_positions: int,
    top_k: int,
    hidden_size: int,
    require_ready: bool,
) -> TeacherCacheV3Audit:
    manifest, manifest_raw = _load_json(cache_dir / MANIFEST_FILENAME, label="teacher cache v3 manifest")
    manifest_sha256 = _sha256_bytes(manifest_raw)
    body = dict(manifest)
    recorded_body_sha = body.pop("manifest_body_sha256", None)
    body_sha = _canonical_sha256(body)
    if recorded_body_sha != body_sha:
        raise TeacherCacheV3Error("teacher cache manifest body SHA-256 mismatch")
    if manifest.get("schema_version") != 3 or manifest.get("record_type") != "glm52_teacher_signal_cache":
        raise TeacherCacheV3Error("teacher cache v3 manifest schema mismatch")
    prompt_payload, prompt_raw = _load_json(prompt_pack_path, label="Teich prompt pack")
    frozen_payload, frozen_raw = _load_json(frozen_prompt_pack_path, label="frozen-66 prompt pack")
    rows = _prompt_rows(prompt_payload, label="Teich prompt pack")
    frozen_rows = _prompt_rows(frozen_payload, label="frozen-66 prompt pack")
    _validate_no_frozen_overlap(rows, frozen_rows)
    if manifest.get("prompt_pack_sha256") != _sha256_bytes(prompt_raw) or manifest.get(
        "frozen_66_prompt_pack_sha256"
    ) != _sha256_bytes(frozen_raw):
        raise TeacherCacheV3Error("teacher cache prompt authority SHA-256 mismatch")
    if len(rows) != expected_session_count or manifest.get("session_count") != expected_session_count:
        raise TeacherCacheV3Error("teacher cache session count mismatch")
    if manifest.get("supervised_position_count") != expected_supervised_positions:
        raise TeacherCacheV3Error("teacher cache supervised-position count mismatch")
    if manifest.get("top_k") != top_k or manifest.get("hidden_size") != hidden_size:
        raise TeacherCacheV3Error("teacher cache signal dimensions mismatch")
    entries = manifest.get("shards")
    if not isinstance(entries, list) or len(entries) != len(rows):
        raise TeacherCacheV3Error("teacher cache ordered shard inventory mismatch")
    expected_splits = assign_provider_balanced_splits(rows, seed=SPLIT_SEED)
    split_counts = {"train": 0, "validation": 0, "holdout": 0}
    for index, (row, entry) in enumerate(zip(rows, entries, strict=True)):
        if not isinstance(entry, dict) or entry.get("row_index") != index or entry.get("prompt_id") != row.get("prompt_id"):
            raise TeacherCacheV3Error("teacher cache shard order does not match prompt pack")
        prompt_id = str(row["prompt_id"])
        split = expected_splits[prompt_id]
        if entry.get("split") != split or entry.get("tuning_eligible") != (split == "train"):
            raise TeacherCacheV3Error("teacher cache split inventory mismatch")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise TeacherCacheV3Error("teacher cache shard relative path is unsafe")
        shard_path = cache_dir / relative
        raw = _read_regular(shard_path, label=f"teacher shard {prompt_id!r}")
        if _sha256_bytes(raw) != entry.get("file_sha256"):
            raise TeacherCacheV3Error(f"teacher shard {prompt_id!r} file SHA-256 mismatch")
        try:
            tensors = load_safetensors(raw)
        except Exception as error:
            raise TeacherCacheV3Error(f"teacher shard {prompt_id!r} is malformed: {error}") from error
        normalized = _validate_tensors(tensors, row=row, top_k=top_k, hidden_size=hidden_size)
        tensor_meta = entry.get("tensors")
        if not isinstance(tensor_meta, dict) or set(tensor_meta) != set(normalized):
            raise TeacherCacheV3Error("teacher shard tensor manifest mismatch")
        for name, value in normalized.items():
            expected = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
            if tensor_meta.get(name) != expected:
                raise TeacherCacheV3Error(f"teacher shard {name} identity mismatch")
        positions = len(row["positions"])
        if entry.get("supervised_position_count") != positions or entry.get("sparse_positions") is not True:
            raise TeacherCacheV3Error("teacher shard sparse-position accounting mismatch")
        split_counts[split] += positions
    if sum(split_counts.values()) != expected_supervised_positions:
        raise TeacherCacheV3Error("teacher cache split accounting does not sum to authority")
    if require_ready:
        ready, _ = _load_json(cache_dir / READY_FILENAME, label="teacher cache ready marker")
        ready_body = dict(ready)
        ready_sha = ready_body.pop("ready_record_sha256", None)
        if ready_sha != _canonical_sha256(ready_body):
            raise TeacherCacheV3Error("teacher cache ready marker SHA-256 mismatch")
        if (
            ready.get("record_type") != "glm52_teacher_cache_ready_v3"
            or ready.get("manifest_sha256") != manifest_sha256
            or ready.get("manifest_body_sha256") != body_sha
            or ready.get("session_count") != expected_session_count
            or ready.get("supervised_position_count") != expected_supervised_positions
        ):
            raise TeacherCacheV3Error("teacher cache ready marker authority mismatch")
        inventory_sha = _canonical_sha256(
            [
                {"relative_path": entry["relative_path"], "file_sha256": entry["file_sha256"]}
                for entry in entries
            ]
        )
        if ready.get("shard_inventory_sha256") != inventory_sha:
            raise TeacherCacheV3Error("teacher cache ready marker inventory mismatch")
    return TeacherCacheV3Audit(
        manifest_sha256=manifest_sha256,
        manifest_body_sha256=body_sha,
        session_count=expected_session_count,
        supervised_positions=expected_supervised_positions,
        split_supervised_positions=split_counts,
    )


__all__ = [
    "MANIFEST_FILENAME",
    "READY_FILENAME",
    "TeacherCacheV3Audit",
    "TeacherCacheV3Error",
    "assign_provider_balanced_splits",
    "audit_glm52_teich_teacher_cache",
    "finalize_glm52_teich_teacher_cache",
]
