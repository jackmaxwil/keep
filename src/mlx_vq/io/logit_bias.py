from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx


LOGIT_BIAS_MANIFEST_KEY = "logit_bias"
LOGIT_BIAS_DIR = "logit_bias"
LOGIT_BIAS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LogitBiasSidecar:
    token_ids: mx.array
    biases: mx.array
    position_indices: tuple[int, ...] | None = None
    token_position_indices: tuple[tuple[int, ...] | None, ...] | None = None


def logit_bias_relpath() -> str:
    return f"{LOGIT_BIAS_DIR}/token-bias.safetensors"


def load_conversion_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return payload


def logit_bias_enabled(artifact_dir: str | Path) -> bool:
    config = load_conversion_manifest(artifact_dir).get(LOGIT_BIAS_MANIFEST_KEY)
    return isinstance(config, dict) and config.get("enabled") is True


def copy_declared_logit_bias_sidecar(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any] | None:
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    config = load_conversion_manifest(seed_root).get(LOGIT_BIAS_MANIFEST_KEY)
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return None
    relpath = config.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY} needs a path")
    source_path = _resolve_artifact_relative_path(seed_root, relpath)
    target_path = output_root / relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return dict(config)


def load_logit_bias_sidecar(
    artifact_dir: str | Path,
    *,
    vocab_size: int,
) -> LogitBiasSidecar | None:
    artifact_root = Path(artifact_dir)
    manifest = load_conversion_manifest(artifact_root)
    config = manifest.get(LOGIT_BIAS_MANIFEST_KEY)
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return None
    if int(config.get("schema_version", 0)) != LOGIT_BIAS_SCHEMA_VERSION:
        raise ValueError(
            f"{LOGIT_BIAS_MANIFEST_KEY}.schema_version must be {LOGIT_BIAS_SCHEMA_VERSION}"
        )
    relpath = config.get("path")
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY} needs a path")
    sidecar_path = _resolve_artifact_relative_path(artifact_root, relpath)
    arrays = mx.load(str(sidecar_path))
    token_ids = arrays.get("token_ids")
    biases = arrays.get("biases")
    _validate_logit_bias_tensors(
        token_ids=token_ids,
        biases=biases,
        vocab_size=vocab_size,
        path=sidecar_path,
    )
    scope = _load_logit_bias_scope(config)
    token_count = int(token_ids.shape[0])
    token_position_indices = scope.get("token_position_indices")
    if token_position_indices is not None and len(token_position_indices) != token_count:
        raise ValueError(
            f"{LOGIT_BIAS_MANIFEST_KEY}.scope.token_position_indices must match token_count={token_count}"
        )
    return LogitBiasSidecar(
        token_ids=token_ids,
        biases=biases,
        position_indices=scope.get("position_indices"),
        token_position_indices=token_position_indices,
    )


def write_logit_bias_sidecar(
    *,
    output_dir: str | Path,
    token_biases: Mapping[int, float],
    position_indices: tuple[int, ...] | None = None,
    token_position_indices: Mapping[int, tuple[int, ...] | int] | None = None,
) -> dict[str, Any]:
    if not token_biases:
        raise ValueError("token_biases must be non-empty")
    token_ids = [int(token_id) for token_id in token_biases]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("token_biases must not contain duplicate token ids")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("token ids must be non-negative")
    bias_values = [float(token_biases[token_id]) for token_id in token_biases]
    scope = _normalize_logit_bias_scope(
        token_ids=tuple(token_ids),
        position_indices=position_indices,
        token_position_indices=token_position_indices,
    )

    output_root = Path(output_dir)
    relpath = logit_bias_relpath()
    sidecar_path = output_root / relpath
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "token_ids": mx.array(token_ids, dtype=mx.int32),
        "biases": mx.array(bias_values, dtype=mx.float32),
    }
    mx.save_safetensors(
        str(sidecar_path),
        tensors,
        metadata={
            "logit_bias_config": json.dumps(
                {
                    "schema_version": LOGIT_BIAS_SCHEMA_VERSION,
                    "format": "sparse_token_bias",
                    "token_count": len(token_ids),
                    "scope": scope,
                },
                sort_keys=True,
            )
        },
    )
    sidecar = {
        "path": relpath,
        "format": "sparse_token_bias",
        "token_count": len(token_ids),
        "tokens": [
            _token_bias_manifest_entry(
                token_id=int(token_id),
                bias=float(bias),
                token_position_indices=scope.get("token_position_indices"),
                token_offset=index,
            )
            for index, (token_id, bias) in enumerate(zip(token_ids, bias_values, strict=True))
        ],
    }
    if scope:
        sidecar["scope"] = scope
    return sidecar


def write_logit_bias_artifact_manifest(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    sidecar: dict[str, Any],
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(load_conversion_manifest(seed_artifact_dir))
    payload = {
        "schema_version": LOGIT_BIAS_SCHEMA_VERSION,
        "enabled": True,
        "format": "sparse_token_bias",
        **sidecar,
    }
    if run_manifest is not None:
        payload["run"] = run_manifest
    manifest[LOGIT_BIAS_MANIFEST_KEY] = payload
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _normalize_logit_bias_scope(
    *,
    token_ids: tuple[int, ...] | None = None,
    position_indices: tuple[int, ...] | None,
    token_position_indices: Mapping[int, tuple[int, ...] | int] | None = None,
) -> dict[str, Any]:
    if position_indices is not None and token_position_indices is not None:
        raise ValueError("position_indices and token_position_indices cannot both be set")
    if position_indices is None and token_position_indices is None:
        return {}
    if position_indices is not None:
        indices = tuple(int(index) for index in position_indices)
        if not indices:
            raise ValueError("position_indices must include at least one position")
        if len(set(indices)) != len(indices):
            raise ValueError("position_indices must be unique")
        if any(index < 0 for index in indices):
            raise ValueError("position_indices must be non-negative")
        return {"position_indices": list(indices)}
    if token_ids is None:
        raise ValueError("token_ids are required when token_position_indices are provided")
    normalized: list[tuple[int, ...] | None] = []
    raw_by_token = {int(token_id): value for token_id, value in (token_position_indices or {}).items()}
    unknown = sorted(set(raw_by_token) - set(token_ids))
    if unknown:
        raise ValueError(f"token_position_indices include unknown token ids: {unknown}")
    if not raw_by_token:
        raise ValueError("token_position_indices must include at least one token")
    for token_id in token_ids:
        raw = raw_by_token.get(int(token_id))
        if raw is None:
            normalized.append(None)
            continue
        if isinstance(raw, int):
            indices = (int(raw),)
        else:
            indices = tuple(int(index) for index in raw)
        if not indices:
            raise ValueError("token_position_indices values must include at least one position")
        if len(set(indices)) != len(indices):
            raise ValueError("token_position_indices values must be unique per token")
        if any(index < 0 for index in indices):
            raise ValueError("token_position_indices values must be non-negative")
        normalized.append(indices)
    return {
        "token_position_indices": [
            None if indices is None else list(indices)
            for indices in normalized
        ]
    }


def _token_bias_manifest_entry(
    *,
    token_id: int,
    bias: float,
    token_position_indices: list[list[int] | None] | None,
    token_offset: int,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"token_id": token_id, "bias": bias}
    if token_position_indices is not None:
        positions = token_position_indices[token_offset]
        if positions is not None:
            entry["position_indices"] = positions
    return entry


def _normalize_position_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    if not indices:
        raise ValueError("position_indices must include at least one position")
    if len(set(indices)) != len(indices):
        raise ValueError("position_indices must be unique")
    if any(index < 0 for index in indices):
        raise ValueError("position_indices must be non-negative")
    return indices


def _load_logit_bias_scope(
    config: dict[str, Any],
) -> dict[str, tuple[int, ...] | tuple[tuple[int, ...] | None, ...] | None]:
    scope = config.get("scope")
    if scope is None:
        return {"position_indices": None, "token_position_indices": None}
    if not isinstance(scope, dict):
        raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.scope must be an object")
    raw_positions = scope.get("position_indices")
    raw_token_positions = scope.get("token_position_indices")
    if raw_positions is not None and raw_token_positions is not None:
        raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.scope cannot define both position_indices and token_position_indices")
    if raw_positions is not None:
        if not isinstance(raw_positions, list):
            raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.scope.position_indices must be a list")
        normalized = _normalize_position_indices(tuple(int(index) for index in raw_positions))
        return {"position_indices": normalized, "token_position_indices": None}
    if raw_token_positions is None:
        tokens = config.get("tokens")
        if isinstance(tokens, list) and any(isinstance(token, dict) and "position_indices" in token for token in tokens):
            loaded: list[tuple[int, ...] | None] = []
            for token in tokens:
                if not isinstance(token, dict):
                    raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.tokens entries must be objects")
                positions = token.get("position_indices")
                if positions is None:
                    loaded.append(None)
                    continue
                if not isinstance(positions, list):
                    raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.tokens.position_indices must be lists")
                loaded.append(_normalize_position_indices(tuple(int(index) for index in positions)))
            return {"position_indices": None, "token_position_indices": tuple(loaded)}
        return {"position_indices": None, "token_position_indices": None}
    if not isinstance(raw_token_positions, list):
        raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.scope.token_position_indices must be a list")
    loaded_positions: list[tuple[int, ...] | None] = []
    for positions in raw_token_positions:
        if positions is None:
            loaded_positions.append(None)
            continue
        if not isinstance(positions, list):
            raise ValueError(f"{LOGIT_BIAS_MANIFEST_KEY}.scope.token_position_indices entries must be lists or null")
        loaded_positions.append(_normalize_position_indices(tuple(int(index) for index in positions)))
    return {"position_indices": None, "token_position_indices": tuple(loaded_positions)}


def _validate_logit_bias_tensors(
    *,
    token_ids: mx.array | None,
    biases: mx.array | None,
    vocab_size: int,
    path: Path,
) -> None:
    if token_ids is None or biases is None:
        raise ValueError(f"{path} must contain token_ids and biases")
    if token_ids.ndim != 1:
        raise ValueError(f"{path} token_ids must be 1D, found {token_ids.shape}")
    if biases.ndim != 1:
        raise ValueError(f"{path} biases must be 1D, found {biases.shape}")
    if token_ids.shape != biases.shape:
        raise ValueError(f"{path} token_ids and biases must have matching shapes")
    if token_ids.shape[0] == 0:
        raise ValueError(f"{path} must contain at least one token bias")
    ids = [int(value) for value in token_ids.tolist()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} token_ids must be unique")
    if any(token_id < 0 or token_id >= vocab_size for token_id in ids):
        raise ValueError(f"{path} token_ids must be within vocab_size={vocab_size}")


def _resolve_artifact_relative_path(artifact_root: Path, relpath: str) -> Path:
    sidecar_path = Path(relpath)
    if sidecar_path.is_absolute():
        raise ValueError(f"logit bias path must be artifact-relative, got {relpath!r}")
    root = artifact_root.resolve(strict=False)
    candidate = (root / sidecar_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"logit bias path escapes artifact root: {relpath!r}") from error
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate
