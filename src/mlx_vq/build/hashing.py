"""Content-addressing for recipe steps and external inputs.

Step keys capture execution identity (op + params + resolved input
identities). Gate thresholds are deliberately excluded so acceptance can be
re-evaluated (``--regate``) without re-running model work; they hash into a
separate gate key. External model artifacts are identified by manifest bytes
plus a symlink-aware file listing — never tensor bytes, which run to tens of
gigabytes. GLM52 teacher-cache artifacts are the deliberate exception: their
content identity streams every manifest and shard byte so a same-size payload
mutation changes the build input hash. Symlink targets in model artifacts are
recorded as raw link strings, not resolved: identity must not depend on where a
protected seed happens to live, and a re-pointed link must change the hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STEP_KEY_LENGTH = 16
UNHASHED = "<unhashed>"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def step_key(
    *,
    op: str,
    op_version: int,
    params: Mapping[str, Any],
    input_keys: Mapping[str, str],
) -> str:
    payload = canonical_json(
        {
            "op": op,
            "op_version": op_version,
            "params": dict(params),
            "inputs": dict(input_keys),
        }
    )
    return _sha256_text(payload)[:STEP_KEY_LENGTH]


def gate_key(*, profile: str, thresholds: Mapping[str, Any]) -> str:
    payload = canonical_json({"profile": profile, "thresholds": dict(thresholds)})
    return _sha256_text(payload)[:STEP_KEY_LENGTH]


def _listing_entry(root: Path, path: Path) -> tuple[str, str, int | None, str | None]:
    relpath = path.relative_to(root).as_posix()
    if path.is_symlink():
        return (relpath, "symlink", None, os.readlink(path))
    if path.is_dir():
        return (relpath, "dir", None, None)
    return (relpath, "file", path.stat().st_size, None)


def artifact_dir_hash(artifact_dir: Path) -> str:
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact dir not found: {artifact_dir}")
    digest = hashlib.sha256()
    manifest_path = artifact_dir / "conversion-manifest.json"
    if manifest_path.exists():
        digest.update(manifest_path.read_bytes())
    entries = sorted(
        _listing_entry(artifact_dir, path)
        for path in artifact_dir.rglob("*")
    )
    digest.update(canonical_json(entries).encode("utf-8"))
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_cache_artifact_hash(cache_root: Path) -> str:
    """Hash every regular file byte in a teacher-cache tree."""

    if not cache_root.is_dir():
        raise FileNotFoundError(f"teacher cache artifact not found: {cache_root}")
    digest = hashlib.sha256()
    file_paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        cache_root, followlinks=False
    ):
        root = Path(directory)
        for name in directory_names:
            path = root / name
            if path.is_symlink():
                raise ValueError(
                    f"teacher cache artifact contains symlinked directory: {path}"
                )
        for name in file_names:
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"teacher cache artifact contains non-regular file: {path}"
                )
            file_paths.append(path)
    for path in sorted(
        file_paths,
        key=lambda item: item.relative_to(cache_root).as_posix(),
    ):
        relative = path.relative_to(cache_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def external_input_hash(kind: str, path: Path) -> str:
    if kind == "artifact_dir":
        return artifact_dir_hash(path)
    if kind == "teacher_cache_artifact":
        return teacher_cache_artifact_hash(path)
    if kind in ("teacher_cache", "file"):
        return file_hash(path)
    raise ValueError(f"unknown external input kind: {kind}")
