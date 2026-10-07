"""Lock-safe, prompt-at-a-time GLM-5.2 teacher-cache production.

The filesystem cache contract deliberately remains MLX-free.  This producer
keeps that property for orchestration and imports the concrete MLX source
runner only inside the production source-loader/runner functions.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np


def _load_cache_api() -> Any:
    module_name = "mlx_vq.quality.glm52_teacher_cache"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).with_name("glm52_teacher_cache.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 teacher-cache contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_route_diagnostics_api() -> Any:
    """Load the writer contract without importing ``mlx_vq.quality.__init__``."""

    module_name = "mlx_vq.quality.glm52_route_diagnostics"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).with_name("glm52_route_diagnostics.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 route diagnostics contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cache_api = _load_cache_api()
GLM52ProducerPhase = cache_api.GLM52ProducerPhase
GLM52TeacherCacheContract = cache_api.GLM52TeacherCacheContract
GLM52TeacherCacheManifest = cache_api.GLM52TeacherCacheManifest
GLM52TeacherCachePrompt = cache_api.GLM52TeacherCachePrompt
GLM52TeacherCacheShard = cache_api.GLM52TeacherCacheShard
GLM52TeacherCacheShardWrite = cache_api.GLM52TeacherCacheShardWrite
build_glm52_teacher_cache_manifest = cache_api.build_glm52_teacher_cache_manifest
canonical_sha256 = cache_api.canonical_sha256
publish_glm52_teacher_cache_manifest = cache_api.publish_glm52_teacher_cache_manifest
write_glm52_teacher_cache_shard = cache_api.write_glm52_teacher_cache_shard


# This is the exact repository-root path used by build.executor.  Importing the
# executor here would eagerly import the model stack and defeat synthetic,
# headless producer tests.
HEAVY_JOB_LOCK_PATH = Path(__file__).resolve().parents[3] / ".keep-heavy-job.lock"
PRODUCER_PHASE_IDS = (
    "non-vq-byte-audit",
    "source-load-and-bind",
    "layer-forward-and-checkpoints",
    "prompt-projection-and-shard-publication",
)
ROUTE_TRACE_PUBLICATION_PHASE_ID = "route-trace-publication"
ROUTE_TRACE_PRODUCER_IMPLEMENTATION_ID = (
    "mlx_vq.quality.glm52_teacher_cache_producer.route-trace.v1"
)
ROUTE_TRACE_CANDIDATE_COMPOSITE_PATH = "artifacts/glm52-reap-504b-v2"
NON_VQ_EXPECTED_RETAINED_TENSORS = 1_194
NON_VQ_EXPECTED_PAYLOAD_BYTES = 37_121_488_608
_SOURCE_TEACHER_MLX_MEMORY_LIMIT_BYTES = 48 * 1024**3
_ROUTED_EXPERT_TENSOR_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\."
)
_MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
_MODEL_OPT_READER_PATCH_LOCK = threading.RLock()
_INVENTORY_HASH_WORKERS = 4


def audit_glm52_non_vq_package(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the canonical byte audit without importing MLX at startup."""

    audit_module = importlib.import_module("mlx_vq.convert.glm52_non_vq")
    return audit_module.audit_glm52_non_vq_package(*args, **kwargs)


class ProducerLockProvider(Protocol):
    def heavy_job_lock(self, path: Path) -> Any: ...

    def run_lock(self, path: Path) -> Any: ...


@dataclass(frozen=True)
class SourceRunEvidence:
    """Small result returned after the source runner has drained its sink."""

    resumed_from_checkpoint: bool
    output_identity_sha256: str
    route_traces: tuple[Any, ...] | None = None
    mlx_cache_limit_bytes: int | None = None
    decode_workers: int | None = None
    decode_queue_depth: int | None = None
    observed_peak_decoded_experts: int | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.output_identity_sha256, label="source output identity")
        if self.route_traces is not None and not isinstance(self.route_traces, tuple):
            raise TypeError("route traces must be a tuple when supplied")
        if self.mlx_cache_limit_bytes is not None and (
            type(self.mlx_cache_limit_bytes) is not int
            or self.mlx_cache_limit_bytes <= 0
        ):
            raise ValueError("MLX cache limit must be a positive integer or null")
        for field_name in (
            "decode_workers",
            "decode_queue_depth",
            "observed_peak_decoded_experts",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or null")


@dataclass(frozen=True)
class GLM52SourceTeacherWorkload:
    """Loaded source model plus the lazy per-layer expert resolver."""

    model: Any
    expert_resolver_for_layer: Callable[[int], Any]
    expected_num_layers: int = 78
    pad_token_id: int = 0
    source_blob_inventory: Any | None = None


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
        )


@dataclass(frozen=True)
class _AuthenticatedSourceBlob:
    shard_name: str
    blob_path: Path
    blob_name: str
    descriptor: int
    identity: _FileIdentity
    verify_seconds: float


@dataclass(frozen=True)
class _AuthenticatedSafetensorsTensorHeader:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass
class _AuthenticatedSourceBlobInventory:
    """Verified blob descriptors kept open for the lifetime of one forward."""

    blob_inventory: dict[str, str]
    source_blob_inventory_sha256: str
    routed_source_blob_inventory_sha256: str
    blobs: tuple[_AuthenticatedSourceBlob, ...]
    blob_root: Path
    weight_map: dict[str, str]
    _closed: bool = False

    def report(self) -> dict[str, Any]:
        return {
            "blob_inventory": dict(self.blob_inventory),
            "source_blob_inventory_sha256": self.source_blob_inventory_sha256,
            "routed_source_blob_inventory_sha256": (
                self.routed_source_blob_inventory_sha256
            ),
            "shard_count": len(self.blobs),
            "shard_verify_seconds": {
                blob.shard_name: blob.verify_seconds for blob in self.blobs
            },
            "total_verify_seconds": sum(blob.verify_seconds for blob in self.blobs),
            "nocache_applied": hasattr(fcntl, "F_NOCACHE"),
            "inventory_hash_workers": _INVENTORY_HASH_WORKERS,
        }

    def verify_after_forward(self) -> None:
        """Reject replacement or mutation of any source blob during the forward."""

        for blob in self.blobs:
            descriptor = _open_regular_no_follow(blob.blob_path)
            try:
                if _FileIdentity.from_stat(os.fstat(descriptor)) != blob.identity:
                    raise ValueError(
                        f"source shard {blob.shard_name!r} identity changed after forward"
                    )
            finally:
                os.close(descriptor)
            if _FileIdentity.from_stat(os.fstat(blob.descriptor)) != blob.identity:
                raise ValueError(
                    f"source shard {blob.shard_name!r} identity changed after forward"
                )

    def _blob_for_path(self, path: str | Path) -> _AuthenticatedSourceBlob:
        requested = Path(path)
        for blob in self.blobs:
            if requested == blob.blob_path:
                return blob
        raise ValueError(f"ModelOpt reader requested unauthenticated source path {requested}")

    @staticmethod
    def _read_exact(descriptor: int, *, offset: int, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        cursor = offset
        while remaining:
            chunk = os.pread(descriptor, remaining, cursor)
            if not chunk:
                raise ValueError("source safetensors file is truncated")
            chunks.append(chunk)
            cursor += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_safetensors_tensor_bytes(
        self,
        path: str | Path,
        tensor_name: str,
    ) -> tuple[_AuthenticatedSafetensorsTensorHeader, bytes]:
        """Read one tensor from a fresh, identity-checked source blob descriptor."""

        blob = self._blob_for_path(path)
        descriptor = _open_regular_no_follow(blob.blob_path)
        try:
            if _FileIdentity.from_stat(os.fstat(descriptor)) != blob.identity:
                raise ValueError(
                    f"source shard {blob.shard_name!r} identity changed before ModelOpt read"
                )
            header_size_raw = self._read_exact(descriptor, offset=0, size=8)
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if header_size <= 0 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(
                    f"{blob.blob_path} safetensors header size must be between 1 and "
                    f"{_MAX_SAFETENSORS_HEADER_BYTES} bytes, got {header_size}"
                )
            header_end = 8 + header_size
            if header_end > blob.identity.size:
                raise ValueError(f"{blob.blob_path} has a truncated safetensors header")
            try:
                header = json.loads(
                    self._read_exact(descriptor, offset=8, size=header_size)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{blob.blob_path} has an invalid safetensors header") from error
            if not isinstance(header, dict):
                raise ValueError(f"{blob.blob_path} safetensors header must be an object")
            tensor = header.get(tensor_name)
            if not isinstance(tensor, dict):
                raise KeyError(f"{tensor_name!r} not found in {blob.blob_path}")
            dtype = tensor.get("dtype")
            shape = tensor.get("shape")
            data_offsets = tensor.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                or not isinstance(data_offsets, list)
                or len(data_offsets) != 2
                or any(type(offset) is not int for offset in data_offsets)
            ):
                raise ValueError(
                    f"{blob.blob_path} has an invalid tensor header for {tensor_name!r}"
                )
            start, end = data_offsets
            if start < 0 or end < start or header_end + end > blob.identity.size:
                raise ValueError(
                    f"{blob.blob_path} has invalid payload offsets for {tensor_name!r}"
                )
            raw = self._read_exact(descriptor, offset=header_end + start, size=end - start)
            if _FileIdentity.from_stat(os.fstat(descriptor)) != blob.identity:
                raise ValueError(
                    f"source shard {blob.shard_name!r} identity changed during ModelOpt read"
                )
            return _AuthenticatedSafetensorsTensorHeader(
                dtype=dtype,
                shape=tuple(shape),
                data_offsets=(start, end),
            ), raw
        finally:
            os.close(descriptor)

    @contextmanager
    def modelopt_reader_guard(self) -> Iterator[None]:
        """Route the unchanged ModelOpt decoder through authenticated fresh reads."""

        nvfp4 = importlib.import_module("mlx_vq.convert.nvfp4")
        with _MODEL_OPT_READER_PATCH_LOCK:
            original_reader = nvfp4.read_safetensors_tensor_bytes
            nvfp4.read_safetensors_tensor_bytes = self.read_safetensors_tensor_bytes
            try:
                yield
            finally:
                nvfp4.read_safetensors_tensor_bytes = original_reader

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for blob in self.blobs:
            os.close(blob.descriptor)


def _make_authenticated_layer_resolver(
    source_blobs: _AuthenticatedSourceBlobInventory,
    *,
    resolver_factory: Callable[..., Any],
) -> Callable[[int], Any]:
    """Build the production layer resolver over blob names, never ``/dev/fd``."""

    def layer_resolver(layer_index: int) -> Any:
        resolver = resolver_factory(
            source_blobs.blob_root,
            source_blobs.weight_map,
            layer_index=layer_index,
        )

        def resolve(expert_index: int) -> Mapping[str, Any]:
            return resolver(expert_index)

        return resolve

    return layer_resolver

@dataclass(frozen=True)
class ProducerPlan:
    cache_root: Path
    ledger_path: Path
    checkpoint_dir: Path
    heavy_job_lock_path: Path
    run_lock_path: Path
    prompt_ids: tuple[str, ...]
    dry_run: bool
    manifest_will_publish: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_root": str(self.cache_root),
            "ledger_path": str(self.ledger_path),
            "checkpoint_dir": str(self.checkpoint_dir),
            "heavy_job_lock_path": str(self.heavy_job_lock_path),
            "run_lock_path": str(self.run_lock_path),
            "prompt_ids": list(self.prompt_ids),
            "prompt_count": len(self.prompt_ids),
            "dry_run": self.dry_run,
            "locks_acquired": False,
            "manifest_will_publish": self.manifest_will_publish,
        }


@dataclass(frozen=True)
class ProducerResult:
    manifest: GLM52TeacherCacheManifest | None
    manifest_path: Path | None
    shards: tuple[GLM52TeacherCacheShard, ...]
    resumed_prompt_ids: tuple[str, ...]
    produced_prompt_ids: tuple[str, ...]
    completed: bool


class FileProducerLockProvider:
    """The executor's advisory ``open('a')`` + exclusive ``flock`` mechanism."""

    @contextmanager
    def heavy_job_lock(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def run_lock(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("teacher-cache producer run lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _require_non_symlink_input(path: str | Path, *, label: str) -> Path:
    """Resolve an input while preserving the caller-visible symlink boundary."""

    input_path = Path(path).expanduser()
    if input_path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    return input_path.resolve(strict=False)


def _authenticated_hf_snapshot_file(
    snapshot_dir: str | Path,
    filename: str,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    """Resolve and authenticate one source file in an HF snapshot/blob tree."""

    path, _ = _authenticated_hf_snapshot_bytes(
        snapshot_dir,
        filename,
        expected_sha256=expected_sha256,
        label=label,
    )
    return path


def _open_regular_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"could not open source blob without following links: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("source blob must be a regular file")
        nocache_command = getattr(fcntl, "F_NOCACHE", None)
        if nocache_command is not None:
            fcntl.fcntl(descriptor, nocache_command, 1)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _sha256_open_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 16 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _authenticated_hf_snapshot_bytes(
    snapshot_dir: str | Path,
    filename: str,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes]:
    """Read authenticated source bytes exactly once through a no-follow descriptor."""

    snapshot = _resolved(snapshot_dir)
    if snapshot.parent.name != "snapshots":
        raise ValueError(f"{label} snapshot must be inside an HF snapshots directory")
    source_path = snapshot / filename
    try:
        resolved = source_path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"could not resolve {label}: {error}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} resolved target must be a regular file")
    hf_cache_tree = snapshot.parent.parent
    try:
        resolved.relative_to(hf_cache_tree)
    except ValueError as error:
        raise ValueError(f"{label} resolved target must remain inside the HF cache tree") from error
    descriptor = _open_regular_no_follow(resolved)
    try:
        payload = bytearray()
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            payload.extend(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} identity does not match the cache contract")
    return resolved, bytes(payload)


def _load_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _source_weight_map_from_index_bytes(index_bytes: bytes) -> dict[str, str]:
    index = _load_json_object_bytes(index_bytes, label="source index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source index must contain a non-empty weight_map object")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise ValueError("source index weight_map must map strings to strings")
    return dict(weight_map)


def _routed_source_weight_map(source_weight_map: Mapping[str, str]) -> dict[str, str]:
    """Return the source-audit scope: routed expert tensors for layers 3..77."""

    return {
        tensor_name: shard_name
        for tensor_name, shard_name in source_weight_map.items()
        if (match := _ROUTED_EXPERT_TENSOR_RE.match(tensor_name)) is not None
        and 3 <= int(match.group("layer")) < 78
    }


def _validated_snapshot_shard_path(
    snapshot: Path,
    blob_root: Path,
    shard_name: str,
) -> tuple[Path, Path]:
    if (
        not shard_name.endswith(".safetensors")
        or Path(shard_name).name != shard_name
        or shard_name in {".", ".."}
    ):
        raise ValueError(f"source index shard name is not a local safetensors filename: {shard_name!r}")
    snapshot_path = snapshot / shard_name
    if not snapshot_path.is_symlink():
        raise ValueError(f"source shard {shard_name!r} must be a snapshot symlink")
    try:
        blob_path = snapshot_path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"could not resolve source shard {shard_name!r}: {error}") from error
    if blob_path.parent != blob_root:
        raise ValueError(
            f"source shard {shard_name!r} does not resolve inside the pinned model blob store"
        )
    if not blob_path.is_file() or blob_path.is_symlink():
        raise ValueError(f"source shard {shard_name!r} blob target must be a regular file")
    if re.fullmatch(r"[0-9a-f]{64}", blob_path.name) is None:
        raise ValueError(f"source shard {shard_name!r} has an invalid LFS blob identity")
    return snapshot_path, blob_path


def _source_blob_inventory(
    snapshot: Path,
    blob_root: Path,
    source_weight_map: Mapping[str, str],
) -> dict[str, str]:
    """Return the full source-index inventory used by the non-VQ audit.

    ``glm52_non_vq._validate_pinned_blob_inventory`` hashes every unique value
    in ``SafetensorsIndex.weight_map`` (its ``index.shards`` property), rather
    than the routed-expert subset needed for this producer's forwards.
    """

    inventory: dict[str, str] = {}
    for shard_name in sorted(set(source_weight_map.values())):
        _, blob_path = _validated_snapshot_shard_path(snapshot, blob_root, shard_name)
        inventory[shard_name] = blob_path.name
    return inventory


def _open_authenticated_source_blob_inventory(
    snapshot_dir: str | Path,
    index_bytes: bytes,
) -> _AuthenticatedSourceBlobInventory:
    snapshot = _resolved(snapshot_dir)
    if snapshot.parent.name != "snapshots" or not snapshot.is_dir():
        raise ValueError("snapshot_dir must be an HF snapshot directory")
    blob_root = snapshot.parent.parent / "blobs"
    if not blob_root.is_dir() or blob_root.is_symlink():
        raise ValueError("HF cache blobs directory must be a real directory")
    full_source_weight_map = _source_weight_map_from_index_bytes(index_bytes)
    source_blob_inventory = _source_blob_inventory(
        snapshot,
        blob_root,
        full_source_weight_map,
    )
    source_weight_map = _routed_source_weight_map(full_source_weight_map)
    blobs: list[_AuthenticatedSourceBlob] = []
    blob_inventory: dict[str, str] = {}
    routed_shards = tuple(sorted(set(source_weight_map.values())))

    def authenticate_shard(shard_name: str) -> _AuthenticatedSourceBlob:
        _, blob_path = _validated_snapshot_shard_path(snapshot, blob_root, shard_name)
        descriptor = _open_regular_no_follow(blob_path)
        identity = _FileIdentity.from_stat(os.fstat(descriptor))
        try:
            started = time.perf_counter()
            if _sha256_open_descriptor(descriptor) != blob_path.name:
                raise ValueError(
                    f"source shard {shard_name!r} content SHA-256 does not match its LFS blob name"
                )
            if _FileIdentity.from_stat(os.fstat(descriptor)) != identity:
                raise ValueError(
                    f"source shard {shard_name!r} identity changed during verification"
                )
            verify_seconds = time.perf_counter() - started
        except BaseException:
            os.close(descriptor)
            raise
        return _AuthenticatedSourceBlob(
            shard_name=shard_name,
            blob_path=blob_path,
            blob_name=blob_path.name,
            descriptor=descriptor,
            identity=identity,
            verify_seconds=verify_seconds,
        )

    futures: list[Future[_AuthenticatedSourceBlob]] = []
    try:
        executor = ThreadPoolExecutor(
            max_workers=_INVENTORY_HASH_WORKERS,
            thread_name_prefix="glm52-inventory-hash",
        )
        try:
            futures = [
                executor.submit(authenticate_shard, name) for name in routed_shards
            ]
            for shard_name, future in zip(routed_shards, futures, strict=True):
                blob = future.result()
                blob_inventory[shard_name] = blob.blob_name
                blobs.append(blob)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    except BaseException:
        for blob in blobs:
            os.close(blob.descriptor)
        for future in futures:
            if future.done() and not future.cancelled() and future.exception() is None:
                blob = future.result()
                if blob not in blobs:
                    os.close(blob.descriptor)
        raise
    return _AuthenticatedSourceBlobInventory(
        blob_inventory=blob_inventory,
        source_blob_inventory_sha256=canonical_sha256(source_blob_inventory),
        routed_source_blob_inventory_sha256=canonical_sha256(blob_inventory),
        blobs=tuple(blobs),
        blob_root=blob_root,
        weight_map={
            tensor_name: blob_inventory[shard_name]
            for tensor_name, shard_name in source_weight_map.items()
        },
    )


def verify_glm52_source_blob_inventory(
    snapshot_dir: str | Path,
    index_bytes: bytes,
    *,
    expected_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Stream-verify routed GLM-5.2 payload blobs without loading model code."""

    verified = _open_authenticated_source_blob_inventory(snapshot_dir, index_bytes)
    try:
        if expected_inventory_sha256 is not None:
            _require_sha256(
                expected_inventory_sha256,
                label="expected source blob inventory",
            )
            if verified.source_blob_inventory_sha256 != expected_inventory_sha256:
                raise ValueError(
                    "source blob inventory does not match the supplied non-VQ evidence"
                )
        return verified.report()
    finally:
        verified.close()


def _outside_cache_root(path: Path, cache_root: Path, *, label: str) -> None:
    try:
        path.relative_to(cache_root)
    except ValueError:
        return
    raise ValueError(f"{label} must remain outside the final cache root")


def _producer_phase_ids(*, capture_route_trace: bool) -> tuple[str, ...]:
    if not capture_route_trace:
        return PRODUCER_PHASE_IDS
    return (*PRODUCER_PHASE_IDS, ROUTE_TRACE_PUBLICATION_PHASE_ID)


def _route_trace_root(
    route_trace_root: str | Path | None,
    *,
    cache_root: Path,
) -> Path | None:
    if route_trace_root is None:
        return None
    root = _require_non_symlink_input(route_trace_root, label="route trace root")
    _outside_cache_root(root, cache_root, label="route trace root")
    _outside_cache_root(cache_root, root, label="cache root")
    return root


def _remove_route_capture_transaction_state(
    *,
    cache_root: Path,
    ledger_path: Path,
    checkpoint_dir: Path,
    trace_root: Path,
) -> None:
    """Remove only the lock-held paths owned by one same-pass capture run."""

    for path in (cache_root, checkpoint_dir, trace_root):
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    if ledger_path.is_symlink() or ledger_path.exists():
        ledger_path.unlink()


def _validated_limit(limit_prompts: int | None, prompt_count: int) -> int:
    if limit_prompts is None:
        return prompt_count
    if type(limit_prompts) is not int or not 1 <= limit_prompts <= prompt_count:
        raise ValueError(f"limit_prompts must be between 1 and {prompt_count}")
    return limit_prompts


def plan_glm52_teacher_cache(
    *,
    contract: GLM52TeacherCacheContract,
    cache_root: str | Path,
    ledger_path: str | Path,
    checkpoint_dir: str | Path,
    limit_prompts: int | None = None,
    route_trace_root: str | Path | None = None,
) -> ProducerPlan:
    """Return a read-only plan.  It intentionally creates no path or lock."""

    root = _require_non_symlink_input(cache_root, label="cache_root")
    ledger = _require_non_symlink_input(ledger_path, label="ledger path")
    checkpoints = _require_non_symlink_input(
        checkpoint_dir,
        label="checkpoint directory",
    )
    run_lock = Path(f"{ledger}.producer.lock")
    _outside_cache_root(ledger, root, label="ledger path")
    _outside_cache_root(checkpoints, root, label="checkpoint directory")
    _outside_cache_root(run_lock, root, label="producer run lock")
    _outside_cache_root(HEAVY_JOB_LOCK_PATH, root, label="heavy-job lock")
    _route_trace_root(route_trace_root, cache_root=root)
    selected_count = _validated_limit(limit_prompts, len(contract.prompts))
    return ProducerPlan(
        cache_root=root,
        ledger_path=ledger,
        checkpoint_dir=checkpoints,
        heavy_job_lock_path=HEAVY_JOB_LOCK_PATH,
        run_lock_path=run_lock,
        prompt_ids=tuple(
            prompt.prompt_id for prompt in contract.prompts[:selected_count]
        ),
        dry_run=True,
        manifest_will_publish=selected_count == len(contract.prompts),
    )


def collect_vm_stat_counts() -> dict[str, int] | None:
    """Use the repository's ``vm_stat`` counter semantics without importing MLX."""

    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    counters: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        name, raw_value = line.split(":", 1)
        raw_value = raw_value.strip().removesuffix(".")
        if raw_value.isdigit():
            counters[name.strip().lower().replace(" ", "_")] = int(raw_value)
    if "pageouts" not in counters or "swapouts" not in counters:
        return None
    return counters


def system_wired_default() -> bool:
    """Conservatively prove the same default wired policy used by GLM52 probes."""

    if any(
        os.environ.get(name)
        for name in ("GLM_MLX_WIRED_LIMIT_GB", "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB")
    ):
        return False
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip()) == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _counter(snapshot: Mapping[str, object] | None, name: str) -> int | None:
    if snapshot is None:
        return None
    value = snapshot.get(name)
    if type(value) is not int or value < 0:
        return None
    return value


def _delta(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    name: str,
) -> int | None:
    start = _counter(before, name)
    end = _counter(after, name)
    if start is None or end is None or end < start:
        return None
    return end - start


def _default_buffer_releaser() -> None:
    gc.collect()
    # Do not import MLX just to release buffers in a synthetic/headless process.
    # A concrete source runner already has MLX imported and clears it per prompt.
    mx = sys.modules.get("mlx.core")
    clear_cache = None if mx is None else getattr(mx, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()


def _validated_resume_shards(
    *,
    contract: GLM52TeacherCacheContract,
    cache_root: Path,
    ledger_path: Path,
    producer_phase_ids: tuple[str, ...] = PRODUCER_PHASE_IDS,
) -> tuple[GLM52TeacherCacheShard, ...]:
    records = cache_api._read_shard_ledger(
        ledger_path,
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    if len(records) > len(contract.prompts):
        raise ValueError("teacher-cache shard ledger exceeds the prompt inventory")
    validated: list[GLM52TeacherCacheShard] = []
    for index, record in enumerate(records):
        prompt = contract.prompts[index]
        shard = cache_api.GLM52TeacherCacheShard.from_mapping(
            record["shard"], index=index
        )
        if shard.prompt_id != prompt.prompt_id:
            raise ValueError("teacher-cache shard ledger is not a canonical prompt prefix")
        if shard.producer_phase_ids != producer_phase_ids:
            raise ValueError("resumed shard has an incompatible producer phase identity")
        final_path = cache_root / Path(*shard.relative_path.split("/"))
        cache_api._audit_shard(
            final_path,
            prompt=prompt,
            vocab_size=contract.vocab_size,
            expected=shard,
        )
        validated.append(shard)
    return tuple(validated)


def _completed_cache_result(
    *,
    contract: GLM52TeacherCacheContract,
    cache_root: Path,
) -> ProducerResult | None:
    manifest_path = cache_root / cache_api.MANIFEST_FILENAME
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    cache_api.audit_glm52_teacher_cache(cache_root, contract=contract)
    manifest_raw, _manifest_identity = cache_api._read_regular_file_stable(
        manifest_path
    )
    raw = cache_api._strict_json_loads(
        manifest_raw,
        label="completed teacher-cache manifest",
    )
    mapping, phases, shards = cache_api._validate_manifest_mapping(
        raw,
        contract=contract,
    )
    manifest = GLM52TeacherCacheManifest(
        **{
            **mapping,
            "producer_phases": phases,
            "shards": shards,
        }
    )
    return ProducerResult(
        manifest=manifest,
        manifest_path=manifest_path,
        shards=shards,
        resumed_prompt_ids=tuple(shard.prompt_id for shard in shards),
        produced_prompt_ids=(),
        completed=True,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def checkpoint_identity_values_for_contract(
    contract: GLM52TeacherCacheContract,
) -> dict[str, str]:
    """Build checkpoint identity values without importing MLX/model code."""

    return {
        "prompt_sha256": canonical_sha256(dict(contract.prompt_authority)),
        "source_sha256": canonical_sha256(
            {
                "source": dict(contract.source),
                "source_evidence": dict(contract.source_evidence),
            }
        ),
        "profile_sha256": str(contract.source["profile_sha256"]),
        "config_sha256": str(contract.source["config_sha256"]),
        "policy_sha256": canonical_sha256(dict(contract.policy)),
        "precision_sha256": canonical_sha256(dict(contract.precision)),
    }


def checkpoint_identities_for_contract(contract: GLM52TeacherCacheContract) -> Any:
    """Build the runner identity object; call only inside both producer locks."""

    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    return source_api.CheckpointIdentities(
        **checkpoint_identity_values_for_contract(contract)
    )


def load_glm52_source_teacher_workload(
    *,
    snapshot_dir: str | Path,
    non_vq_package_dir: str | Path,
    profile_path: str | Path,
    contract: GLM52TeacherCacheContract,
) -> GLM52SourceTeacherWorkload:
    """Authenticate and bind the source workload; call only inside both locks."""

    snapshot = _require_non_symlink_input(snapshot_dir, label="snapshot_dir")
    non_vq_root = _require_non_symlink_input(
        non_vq_package_dir,
        label="non_vq_package_dir",
    )
    profile = _require_non_symlink_input(profile_path, label="profile_path")
    if not snapshot.is_dir():
        raise ValueError("snapshot_dir must be a real directory")
    if not non_vq_root.is_dir():
        raise ValueError("non_vq_package_dir must be a real directory")
    if not profile.is_file():
        raise ValueError("profile_path must be a regular non-symlink file")
    if _sha256_file(profile) != contract.source["profile_sha256"]:
        raise ValueError("profile identity does not match the cache contract")
    config_path, config_bytes = _authenticated_hf_snapshot_bytes(
        snapshot,
        "config.json",
        expected_sha256=contract.source["config_sha256"],
        label="source config",
    )
    _, source_index_bytes = _authenticated_hf_snapshot_bytes(
        snapshot,
        "model.safetensors.index.json",
        expected_sha256=contract.source["index_sha256"],
        label="source index",
    )
    non_vq_manifest_path = non_vq_root / "non-vq-manifest.json"
    non_vq_index_path = non_vq_root / "model.safetensors.index.json"
    for path, label in (
        (non_vq_manifest_path, "non-VQ manifest"),
        (non_vq_index_path, "non-VQ index"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} must be a regular non-symlink file")
    if _sha256_file(non_vq_manifest_path) != contract.non_vq_package["manifest_sha256"]:
        raise ValueError("non-VQ manifest identity does not match the cache contract")
    non_vq_manifest = _load_json_object(non_vq_manifest_path, label="non-VQ manifest")
    for field, expected in (
        ("package_set_sha256", contract.non_vq_package["package_set_sha256"]),
        ("retained_tensor_count", contract.non_vq_package["retained_tensor_count"]),
        ("tensor_payload_bytes", contract.non_vq_package["tensor_payload_bytes"]),
        ("model_id", contract.source["model_id"]),
        ("source_revision", contract.source["revision"]),
        ("profile", contract.source["profile"]),
    ):
        if non_vq_manifest.get(field) != expected:
            raise ValueError(f"non-VQ manifest {field} does not match the cache contract")

    import mlx.core as mx

    from mlx.utils import tree_map_with_path

    from mlx_vq.convert.stream_convert import load_safetensors_index
    from mlx_vq.models.glm52_vq_adapter import (
        GLM52VQModel,
        bind_glm52_non_vq_weights,
        glm52_vq_args_from_config,
    )
    from mlx_vq.models.glm52_source_teacher import (
        make_modelopt_nvfp4_expert_weight_resolver,
    )

    config = _load_json_object_bytes(config_bytes, label="source config")
    model = GLM52VQModel(glm52_vq_args_from_config(config))

    def cast_parameter(path: str, value: Any) -> Any:
        if model.cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(cast_parameter, model.parameters()))
    non_vq_index = load_safetensors_index(non_vq_index_path)
    bind_report = bind_glm52_non_vq_weights(
        model,
        non_vq_root,
        non_vq_index,
        strict=True,
    )
    if bind_report.loaded_count != 1_272:
        raise ValueError("strict non-VQ bind did not load exactly 1272 runtime targets")
    model.eval()
    mx.eval(model.parameters())
    expected_inventory = non_vq_manifest.get("source_blob_inventory_sha256")
    if expected_inventory is not None:
        _require_sha256(
            expected_inventory,
            label="non-VQ source_blob_inventory_sha256",
        )
    source_blobs = _open_authenticated_source_blob_inventory(snapshot, source_index_bytes)
    try:
        if len(source_blobs.blobs) != contract.source_evidence["required_payload_shard_count"]:
            raise ValueError("source index does not name exactly 61 payload shards")
        if (
            expected_inventory is not None
            and source_blobs.source_blob_inventory_sha256 != expected_inventory
        ):
            raise ValueError(
                "source blob inventory does not match the supplied non-VQ evidence"
            )
    except BaseException:
        source_blobs.close()
        raise

    layer_resolver = _make_authenticated_layer_resolver(
        source_blobs,
        resolver_factory=make_modelopt_nvfp4_expert_weight_resolver,
    )

    return GLM52SourceTeacherWorkload(
        model=model,
        expert_resolver_for_layer=layer_resolver,
        expected_num_layers=78,
        source_blob_inventory=source_blobs,
    )


def audit_glm52_non_vq_package_for_teacher_cache(
    *,
    snapshot_dir: str | Path,
    non_vq_package_dir: str | Path,
    profile_path: str | Path,
    contract: GLM52TeacherCacheContract,
    audit_callable: Callable[..., Any] = audit_glm52_non_vq_package,
) -> Any:
    """Run the canonical byte audit and bind its recomputed result to the contract."""

    snapshot = _require_non_symlink_input(snapshot_dir, label="snapshot_dir")
    non_vq_root = _require_non_symlink_input(
        non_vq_package_dir,
        label="non_vq_package_dir",
    )
    profile_input = _require_non_symlink_input(profile_path, label="profile_path")
    if not snapshot.is_dir():
        raise ValueError("snapshot_dir must be a real directory")
    if not non_vq_root.is_dir():
        raise ValueError("non_vq_package_dir must be a real directory")
    if not profile_input.is_file():
        raise ValueError("profile_path must be a regular non-symlink file")
    config_path = _authenticated_hf_snapshot_file(
        snapshot,
        "config.json",
        expected_sha256=contract.source["config_sha256"],
        label="source config",
    )
    index_path = _authenticated_hf_snapshot_file(
        snapshot,
        "model.safetensors.index.json",
        expected_sha256=contract.source["index_sha256"],
        label="source index",
    )
    stream_convert = importlib.import_module("mlx_vq.convert.stream_convert")
    profiles = importlib.import_module("mlx_vq.models.profiles")
    source_index = stream_convert.load_safetensors_index(index_path)
    audit = audit_callable(
        non_vq_root,
        source_dir=snapshot,
        source_index=source_index,
        profile=profiles.load_profile(profile_input),
        expected_model_id=contract.source["model_id"],
        expected_revision=contract.source["revision"],
        expected_config_sha256=contract.source["config_sha256"],
        expected_index_sha256=contract.source["index_sha256"],
        config_path=config_path,
        index_path=index_path,
        enforce_pinned_source=True,
    )
    expected = (
        ("manifest_sha256", contract.non_vq_package["manifest_sha256"]),
        ("package_set_sha256", contract.non_vq_package["package_set_sha256"]),
        ("retained_tensor_count", NON_VQ_EXPECTED_RETAINED_TENSORS),
        ("tensor_payload_bytes", NON_VQ_EXPECTED_PAYLOAD_BYTES),
    )
    if getattr(audit, "audit_pass", False) is not True:
        raise ValueError("non-VQ byte-level package audit did not pass every check")
    for field, expected_value in expected:
        actual = getattr(audit, field, None)
        if actual != expected_value:
            raise ValueError(
                f"non-VQ byte-level package audit {field} mismatch: "
                f"expected {expected_value!r}, found {actual!r}"
            )
    if audit.retained_tensor_count != contract.non_vq_package["retained_tensor_count"]:
        raise ValueError(
            "non-VQ byte-level package audit tensor count disagrees with contract"
        )
    if audit.tensor_payload_bytes != contract.non_vq_package["tensor_payload_bytes"]:
        raise ValueError(
            "non-VQ byte-level package audit payload bytes disagree with contract"
        )
    return audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _route_trace_capture_identity(layers: tuple[Any, ...]) -> str:
    """Hash the observed router outputs before the diagnostics writer copies them."""

    records: list[dict[str, Any]] = []
    for layer in layers:
        ids = np.asarray(layer.expert_ids)
        scores = np.asarray(layer.scores)
        records.append(
            {
                "layer_index": int(layer.layer_index),
                "expert_ids_shape": list(ids.shape),
                "expert_ids_sha256": hashlib.sha256(
                    np.ascontiguousarray(ids).tobytes(order="C")
                ).hexdigest(),
                "scores_shape": list(scores.shape),
                "scores_sha256": hashlib.sha256(
                    np.ascontiguousarray(scores).tobytes(order="C")
                ).hexdigest(),
                "valid_assignment_count": int(layer.valid_assignment_count),
                "padded_assignment_count": int(layer.padded_assignment_count),
            }
        )
    return canonical_sha256(records)


def _validated_route_trace_layers(
    traces: tuple[Any, ...] | None,
    *,
    contract: GLM52TeacherCacheContract,
) -> tuple[Any, ...]:
    """Require the fixed source-side trace coverage before durable publication."""

    if traces is None:
        raise ValueError("source runner did not return route traces for requested capture")
    diagnostics = _load_route_diagnostics_api()
    layers = tuple(diagnostics.source_trace_to_layers(traces))
    expected_indexes = tuple(diagnostics.FROZEN_SPARSE_LAYERS)
    if tuple(layer.layer_index for layer in layers) != expected_indexes:
        raise ValueError("route trace must cover source sparse layers 3 through 77 exactly")
    if contract.predictor_position_count != diagnostics.FROZEN_N_VALID:
        raise ValueError("route trace capture requires the frozen 744-row prompt pack")
    expected_assignments = diagnostics.FROZEN_N_VALID * diagnostics.TOP_K
    for layer in layers:
        ids = np.asarray(layer.expert_ids)
        scores = np.asarray(layer.scores)
        if ids.shape != (diagnostics.FROZEN_N_VALID, diagnostics.TOP_K):
            raise ValueError("route expert IDs must have shape [744, 8]")
        if scores.shape != ids.shape:
            raise ValueError("route scores must align with route expert IDs")
        if ids.dtype != np.dtype("int32"):
            raise ValueError("route expert IDs must use int32")
        if scores.dtype != np.dtype("float32"):
            raise ValueError("route scores must use float32")
        if not np.isfinite(scores).all():
            raise ValueError("route scores must be finite")
        if layer.valid_assignment_count != expected_assignments:
            raise ValueError("route trace valid assignment count must equal 744 * 8")
        if layer.padded_assignment_count != 0:
            raise ValueError("route trace must contain zero padded assignments")
    return layers


def _route_trace_capture_authority(
    *,
    contract: GLM52TeacherCacheContract,
    manifest: GLM52TeacherCacheManifest,
    manifest_path: Path,
    trace_root: Path,
    capture_output_sha256: str,
) -> dict[str, str]:
    """Build the writer's complete release-class source-capture authority."""

    diagnostics = _load_route_diagnostics_api()
    return {
        "teacher_cache_path": str(manifest_path),
        "teacher_cache_content_sha256": manifest.cache_content_sha256,
        "candidate_composite_path": ROUTE_TRACE_CANDIDATE_COMPOSITE_PATH,
        "candidate_composite_identity_sha256": (
            diagnostics.PINNED_CANDIDATE_COMPOSITE_IDENTITY_SHA256
        ),
        "prompt_pack_path": str(contract.prompt_authority["path"]),
        "prompt_pack_sha256": str(contract.prompt_authority["file_sha256"]),
        "model_id": str(contract.source["model_id"]),
        "model_revision": str(contract.source["revision"]),
        "producer_implementation_id": ROUTE_TRACE_PRODUCER_IMPLEMENTATION_ID,
        "capture_output_path": str(trace_root),
        "capture_output_sha256": capture_output_sha256,
        "evidence_class": "release",
    }


def _run_glm52_source_teacher_to_sink(
    source: GLM52SourceTeacherWorkload,
    prompts: tuple[GLM52TeacherCachePrompt, ...],
    *,
    pending_prompt_indices: tuple[int, ...],
    checkpoint_dir: Path,
    checkpoint_identities: object | None,
    phase_boundary: Callable[[], None],
    sink: Callable[[int, np.ndarray], None],
    capture_route_trace: bool = False,
) -> SourceRunEvidence:
    """Run the layer-major teacher and drain each FP32 LM-head row to ``sink``.

    The committed runner has no sink callback.  This adapter reuses its exact
    prompt preparation, precision checks, MoE primitive, scatter primitive,
    checkpoint genesis/resume/write chain, and checkpoint byte identities.  It
    differs only at the final LM-head loop, where each array is published and
    released before the next prompt projection.
    """

    if not isinstance(source, GLM52SourceTeacherWorkload):
        raise TypeError("source must be a GLM52SourceTeacherWorkload")
    if checkpoint_identities is None:
        raise ValueError("checkpoint_identities are required for source production")
    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    if isinstance(checkpoint_identities, Mapping):
        checkpoint_identities = source_api.CheckpointIdentities(
            **dict(checkpoint_identities)
        )
    import mlx.core as mx

    from mlx_lm.models.base import create_causal_mask
    from mlx_vq.models.glm52_vq_adapter import GLM52VQModel, Glm52VQMoE

    model = source.model
    expected_num_layers = source.expected_num_layers
    if not isinstance(model, GLM52VQModel):
        raise TypeError("source workload model must be a GLM52VQModel")
    if model.args.num_hidden_layers != expected_num_layers:
        raise ValueError("source workload main-layer count is not exactly 78")
    if model.model.start_idx != 0 or model.model.end_idx != expected_num_layers:
        raise ValueError("source teacher requires the complete main-layer range")
    if len(model.model.layers) != expected_num_layers:
        raise ValueError("model layer inventory does not match expected_num_layers")
    encoded = tuple(tuple(prompt.encoded_token_ids) for prompt in prompts)
    token_batch, valid_mask, right_padding = source_api._prepare_predictor_batch(
        encoded,
        vocab_size=model.args.vocab_size,
        pad_token_id=source.pad_token_id,
    )
    batch_size, sequence_length = token_batch.shape
    if sequence_length > model.args.index_topk:
        raise ValueError("source-teacher checkpoint requires absent IndexShare state")
    source_api._audit_source_teacher_model_precision(model)

    tokens = mx.array(token_batch)
    hidden = mx.contiguous(model.model.embed_tokens(tokens))
    if hidden.dtype != mx.bfloat16:
        raise ValueError("embedded source-teacher hidden state must use bfloat16")
    mx.eval(hidden)
    causal_mask = create_causal_mask(
        sequence_length,
        right_padding=mx.array(right_padding),
    )
    mx.eval(causal_mask)
    valid_flat_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
    valid_mask_digest = source_api._valid_mask_sha256(valid_mask)
    hidden_shape = (batch_size, sequence_length, model.args.hidden_size)
    genesis_sha256 = source_api._genesis_identity(
        identities=checkpoint_identities,
        encoded_token_ids=encoded,
        valid_mask_sha256=valid_mask_digest,
        embedded_hidden=hidden,
    )
    (
        start_layer,
        resumed_hidden,
        previous_checkpoint_sha256,
        previous_ledger_record_sha256,
    ) = source_api._resume_checkpoint_chain(
        checkpoint_dir,
        num_layers=expected_num_layers,
        hidden_shape=hidden_shape,
        valid_mask_sha256=valid_mask_digest,
        identities=checkpoint_identities,
        genesis_sha256=genesis_sha256,
    )
    if resumed_hidden is not None:
        if capture_route_trace:
            raise ValueError(
                "route traces cannot be reconstructed from hidden-only checkpoints"
            )
        hidden = resumed_hidden

    route_traces: list[Any] = []
    pipeline_enabled = bool(source_api._EXPERT_DECODE_PIPELINE_ENABLED)
    decode_workers = source_api._EXPERT_DECODE_WORKERS if pipeline_enabled else 0
    decode_queue_depth = (
        source_api._EXPERT_DECODE_QUEUE_DEPTH if pipeline_enabled else 0
    )
    observed_peak_decoded_experts = 0
    prev_topk_indices = None
    for layer_number in range(start_layer, expected_num_layers):
        layer = model.model.layers[layer_number]
        if int(layer.layer_idx) != layer_number:
            raise ValueError("decoder layer identity does not match its index")
        attention_output, next_topk_indices = layer.self_attn(
            layer.input_layernorm(hidden),
            causal_mask,
            None,
            prev_topk_indices,
        )
        if next_topk_indices is not None:
            raise ValueError("short source-teacher run must have absent IndexShare state")
        prev_topk_indices = next_topk_indices
        attention_hidden = hidden + attention_output
        mlp_hidden = layer.post_attention_layernorm(attention_hidden)
        if isinstance(layer.mlp, Glm52VQMoE):
            flat_mlp_hidden = mlp_hidden.reshape(-1, model.args.hidden_size)
            valid_hidden = mx.take(
                flat_mlp_hidden,
                mx.array(valid_flat_indices),
                axis=0,
            )
            streamed = source_api.stream_selected_expert_moe(
                valid_hidden,
                gate=layer.mlp.gate,
                expert_weight_resolver=source.expert_resolver_for_layer(
                    layer_number
                ),
                shared_expert=layer.mlp.get("shared_experts"),
                num_routed_experts=model.args.n_routed_experts,
            )
            if (
                decode_workers != streamed.decode_workers
                or decode_queue_depth != streamed.decode_queue_depth
            ):
                raise AssertionError("source sparse layers changed decode pipeline settings")
            observed_peak_decoded_experts = max(
                observed_peak_decoded_experts,
                streamed.observed_peak_decoded_experts,
            )
            expected_assignments = (
                len(valid_flat_indices) * model.args.num_experts_per_tok
            )
            if streamed.valid_assignment_count != expected_assignments:
                raise AssertionError("valid sparse assignments do not equal n_valid * top_k")
            if streamed.padded_assignment_count != 0:
                raise AssertionError("padded rows entered sparse router accounting")
            sparse_output = source_api._scatter_valid_rows(
                streamed.output,
                valid_flat_indices=valid_flat_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=model.args.hidden_size,
            )
            hidden = attention_hidden + sparse_output
            # This is deliberately after ``hidden`` has been computed from the
            # streamed MoE output.  Host copies observe router results only;
            # they never feed a value back into the logits path.
            if capture_route_trace:
                expert_ids = np.ascontiguousarray(
                    np.array(streamed.expert_indices).astype(np.int32, copy=False)
                )
                route_scores = np.ascontiguousarray(
                    np.array(streamed.route_scores).astype(np.float32, copy=False)
                )
                if expert_ids.shape != route_scores.shape or expert_ids.shape[1] != 8:
                    raise AssertionError("captured source route trace must have top-k 8")
                if expert_ids.dtype != np.dtype("int32") or route_scores.dtype != np.dtype(
                    "float32"
                ):
                    raise AssertionError("route capture observation changed its required dtypes")
                route_traces.append(
                    source_api.LayerRouteTrace(
                        layer_index=layer_number,
                        expert_ids=expert_ids,
                        scores=route_scores,
                        valid_assignment_count=streamed.valid_assignment_count,
                        padded_assignment_count=streamed.padded_assignment_count,
                    )
                )
        else:
            hidden = attention_hidden + layer.mlp(mlp_hidden)
        hidden = mx.contiguous(hidden)
        if hidden.dtype != mx.bfloat16:
            raise ValueError("post-layer source-teacher hidden state must use bfloat16")
        mx.eval(hidden)
        (
            previous_checkpoint_sha256,
            previous_ledger_record_sha256,
        ) = source_api._write_layer_checkpoint(
            checkpoint_dir,
            layer_number=layer_number,
            layer_identity=f"model.layers.{layer_number}",
            hidden=hidden,
            valid_mask_sha256=valid_mask_digest,
            identities=checkpoint_identities,
            previous_checkpoint_sha256=previous_checkpoint_sha256,
            previous_ledger_record_sha256=previous_ledger_record_sha256,
        )
        del attention_output
        del attention_hidden
        del mlp_hidden
        if isinstance(layer.mlp, Glm52VQMoE):
            del flat_mlp_hidden
            del valid_hidden
            del streamed
            del sparse_output
        mx.clear_cache()

    phase_boundary()
    normalized = model.model.norm(hidden)
    if normalized.dtype != mx.bfloat16:
        raise ValueError("final normalized source-teacher hidden state must use bfloat16")
    predictor_lengths = valid_mask.sum(axis=1).astype(np.int64)
    for prompt_index in pending_prompt_indices:
        predictor_length = int(predictor_lengths[prompt_index])
        prompt_hidden = normalized[
            prompt_index : prompt_index + 1,
            :predictor_length,
            :,
        ]
        prompt_logits = model.lm_head(prompt_hidden).astype(mx.float32).reshape(
            predictor_length,
            model.args.vocab_size,
        )
        mx.eval(prompt_logits)
        stored_logits = np.ascontiguousarray(
            np.array(prompt_logits).astype(np.float32, copy=False)
        )
        try:
            sink(prompt_index, stored_logits)
        finally:
            del stored_logits
            del prompt_logits
            del prompt_hidden
            mx.clear_cache()
            gc.collect()
    del normalized
    mx.clear_cache()
    return SourceRunEvidence(
        resumed_from_checkpoint=start_layer > 0,
        output_identity_sha256=previous_checkpoint_sha256,
        route_traces=tuple(route_traces) if capture_route_trace else None,
        decode_workers=decode_workers,
        decode_queue_depth=decode_queue_depth,
        observed_peak_decoded_experts=observed_peak_decoded_experts,
    )


def run_glm52_source_teacher_to_sink(
    source: GLM52SourceTeacherWorkload,
    prompts: tuple[GLM52TeacherCachePrompt, ...],
    *,
    pending_prompt_indices: tuple[int, ...],
    checkpoint_dir: Path,
    checkpoint_identities: object | None,
    phase_boundary: Callable[[], None],
    sink: Callable[[int, np.ndarray], None],
    capture_route_trace: bool = False,
) -> SourceRunEvidence:
    """Run the source workload with a bounded, transactionally restored MLX cache."""

    mx = importlib.import_module("mlx.core")

    cache_limit_bytes = 4 * 1024**3
    memory_limit_bytes = _SOURCE_TEACHER_MLX_MEMORY_LIMIT_BYTES
    set_cache_limit = getattr(mx, "set_cache_limit", None)
    set_memory_limit = getattr(mx, "set_memory_limit", None)
    previous_cache_limit: int | None = None
    previous_memory_limit: int | None = None
    if set_memory_limit is not None:
        previous_memory_limit = int(set_memory_limit(memory_limit_bytes))
    try:
        if set_cache_limit is not None:
            previous_cache_limit = int(set_cache_limit(cache_limit_bytes))
        try:
            source_blobs = getattr(source, "source_blob_inventory", None)
            reader_guard = (
                source_blobs.modelopt_reader_guard()
                if source_blobs is not None
                else nullcontext()
            )
            with reader_guard:
                evidence = _run_glm52_source_teacher_to_sink(
                    source,
                    prompts,
                    pending_prompt_indices=pending_prompt_indices,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_identities=checkpoint_identities,
                    phase_boundary=phase_boundary,
                    sink=sink,
                    capture_route_trace=capture_route_trace,
                )
        finally:
            if set_cache_limit is not None and previous_cache_limit is not None:
                set_cache_limit(previous_cache_limit)
    finally:
        if set_memory_limit is not None and previous_memory_limit is not None:
            set_memory_limit(previous_memory_limit)
    return SourceRunEvidence(
        resumed_from_checkpoint=evidence.resumed_from_checkpoint,
        output_identity_sha256=evidence.output_identity_sha256,
        route_traces=evidence.route_traces,
        mlx_cache_limit_bytes=(cache_limit_bytes if set_cache_limit is not None else None),
        decode_workers=evidence.decode_workers,
        decode_queue_depth=evidence.decode_queue_depth,
        observed_peak_decoded_experts=evidence.observed_peak_decoded_experts,
    )


def produce_glm52_teacher_cache(
    *,
    contract: GLM52TeacherCacheContract,
    cache_root: str | Path,
    ledger_path: str | Path,
    checkpoint_dir: str | Path,
    non_vq_package_auditor: Callable[[], Any],
    source_loader: Callable[[], Any],
    source_runner: Callable[..., SourceRunEvidence],
    checkpoint_identities: object | None = None,
    limit_prompts: int | None = None,
    route_trace_root: str | Path | None = None,
    lock_provider: ProducerLockProvider | None = None,
    memory_counter_reader: Callable[[], Mapping[str, object] | None] = (
        collect_vm_stat_counts
    ),
    wired_policy_probe: Callable[[], bool] = system_wired_default,
    shard_writer: Callable[..., GLM52TeacherCacheShardWrite] = (
        write_glm52_teacher_cache_shard
    ),
    manifest_publisher: Callable[..., Path] = publish_glm52_teacher_cache_manifest,
    route_trace_writer: Callable[..., Mapping[str, Any]] | None = None,
    buffer_releaser: Callable[[], None] = _default_buffer_releaser,
    created_at_factory: Callable[[], str] = _utc_now,
) -> ProducerResult:
    """Produce shards under both locks and publish the manifest last.

    ``non_vq_package_auditor`` and ``source_loader`` are deliberately
    zero-argument callables. Accepting an already-audited package or loaded
    model would make it possible to perform either operation before the global
    and run-specific lock boundary.
    """

    plan = plan_glm52_teacher_cache(
        contract=contract,
        cache_root=cache_root,
        ledger_path=ledger_path,
        checkpoint_dir=checkpoint_dir,
        limit_prompts=limit_prompts,
        route_trace_root=route_trace_root,
    )
    trace_root = _route_trace_root(route_trace_root, cache_root=plan.cache_root)
    capture_route_trace = trace_root is not None
    selected_count = len(plan.prompt_ids)
    if capture_route_trace and selected_count != len(contract.prompts):
        raise ValueError("route trace capture requires the complete frozen prompt pack")
    phase_ids = _producer_phase_ids(capture_route_trace=capture_route_trace)
    locks = lock_provider or FileProducerLockProvider()
    with locks.heavy_job_lock(plan.heavy_job_lock_path):
        with locks.run_lock(plan.run_lock_path):
            if capture_route_trace:
                preexisting = [
                    path
                    for path in (
                        plan.cache_root,
                        plan.ledger_path,
                        plan.checkpoint_dir,
                        trace_root,
                    )
                    if path is not None and (path.exists() or path.is_symlink())
                ]
                if preexisting:
                    raise ValueError(
                        "route trace capture requires fresh exact-run transaction paths: "
                        + ", ".join(str(path) for path in preexisting)
                    )
            completed = _completed_cache_result(
                contract=contract,
                cache_root=plan.cache_root,
            )
            if completed is not None:
                if capture_route_trace:
                    raise ValueError(
                        "route trace capture requires cache production in the same run"
                    )
                return completed
            resumed = _validated_resume_shards(
                contract=contract,
                cache_root=plan.cache_root,
                ledger_path=plan.ledger_path,
                producer_phase_ids=phase_ids,
            )
            if len(resumed) > selected_count:
                selected_resumed = resumed[:selected_count]
            else:
                selected_resumed = resumed
            if capture_route_trace and selected_resumed:
                raise ValueError(
                    "route trace capture cannot resume prompt shards from an earlier run"
                )
            pending_indices = tuple(range(len(selected_resumed), selected_count))
            shards_by_index: dict[int, GLM52TeacherCacheShard] = {
                index: shard for index, shard in enumerate(selected_resumed)
            }
            produced_ids: list[str] = []
            wired_default_before = bool(wired_policy_probe())
            before = memory_counter_reader()
            non_vq_audit = non_vq_package_auditor()
            after_audit = memory_counter_reader()
            source_evidence = SourceRunEvidence(
                resumed_from_checkpoint=bool(selected_resumed),
                output_identity_sha256=canonical_sha256(
                    [shard.to_dict() for shard in selected_resumed]
                ),
            )
            after_load: Mapping[str, object] | None = None
            after_layers: Mapping[str, object] | None = None
            source_blob_inventory_sha256: str | None = None
            source_blob_verification: Mapping[str, Any] | None = None
            trace_layers: tuple[Any, ...] | None = None
            trace_capture_identity: str | None = None
            if pending_indices:
                source = source_loader()
                after_load = memory_counter_reader()
                next_pending = iter(pending_indices)
                expected_index = next(next_pending, None)

                def phase_boundary() -> None:
                    nonlocal after_layers
                    if after_layers is not None:
                        raise ValueError("source runner reported the layer boundary twice")
                    after_layers = memory_counter_reader()

                def sink(prompt_index: int, logits: np.ndarray) -> None:
                    nonlocal expected_index
                    if prompt_index != expected_index:
                        raise ValueError(
                            "source runner must drain pending prompts once in canonical order"
                        )
                    prompt = contract.prompts[prompt_index]
                    stored = np.ascontiguousarray(logits, dtype=np.float32)
                    try:
                        written = shard_writer(
                            plan.cache_root,
                            ledger_path=plan.ledger_path,
                            prompt=prompt,
                            logits=stored,
                            vocab_size=contract.vocab_size,
                            producer_phase_ids=PRODUCER_PHASE_IDS,
                            bound_identity_sha256=contract.bound_identity_sha256,
                        )
                        shards_by_index[prompt_index] = written.shard
                        produced_ids.append(prompt.prompt_id)
                    finally:
                        del stored
                        buffer_releaser()
                    expected_index = next(next_pending, None)

                source_blobs = getattr(source, "source_blob_inventory", None)
                if source_blobs is not None:
                    source_blob_inventory_sha256 = getattr(
                        source_blobs,
                        "source_blob_inventory_sha256",
                        None,
                    )
                    report = getattr(source_blobs, "report", None)
                    if callable(report):
                        source_blob_verification = report()
                try:
                    source_runner_kwargs: dict[str, Any] = {
                        "pending_prompt_indices": pending_indices,
                        "checkpoint_dir": plan.checkpoint_dir,
                        "checkpoint_identities": checkpoint_identities,
                        "phase_boundary": phase_boundary,
                        "sink": sink,
                    }
                    if capture_route_trace:
                        source_runner_kwargs["capture_route_trace"] = True
                    source_evidence = source_runner(
                        source,
                        contract.prompts,
                        **source_runner_kwargs,
                    )
                finally:
                    if source_blobs is not None:
                        verify_after_forward = getattr(
                            source_blobs,
                            "verify_after_forward",
                            None,
                        )
                        if callable(verify_after_forward):
                            verify_after_forward()
                        close_source_blobs = getattr(source_blobs, "close", None)
                        if callable(close_source_blobs):
                            close_source_blobs()
                if not isinstance(source_evidence, SourceRunEvidence):
                    raise TypeError("source_runner must return SourceRunEvidence")
                if expected_index is not None:
                    raise ValueError("source runner returned before draining every pending prompt")
                if after_layers is None:
                    raise ValueError("source runner did not report the layer boundary")
                if capture_route_trace:
                    trace_layers = _validated_route_trace_layers(
                        source_evidence.route_traces,
                        contract=contract,
                    )
                    trace_capture_identity = _route_trace_capture_identity(trace_layers)
                del source
                buffer_releaser()
            after = memory_counter_reader()
            selected_shards = tuple(shards_by_index[index] for index in range(selected_count))
            resumed_or_checkpointed = bool(selected_resumed) or bool(
                source_evidence.resumed_from_checkpoint
            )
            wired_default = (
                wired_default_before
                and bool(wired_policy_probe())
                and not resumed_or_checkpointed
            )
            load_output_identity = canonical_sha256(
                {
                    "source": dict(contract.source),
                    "non_vq_bound_package_identity": contract.non_vq_package[
                        "bound_package_identity"
                    ],
                    "source_blob_inventory_sha256": source_blob_inventory_sha256,
                    "source_blob_verification": source_blob_verification,
                }
            )
            shard_output_identity = canonical_sha256(
                [shard.to_dict() for shard in selected_shards]
            )
            nocache_applied = bool(
                source_blob_verification
                and source_blob_verification.get("nocache_applied") is True
            )
            inventory_hash_workers = (
                source_blob_verification.get("inventory_hash_workers")
                if source_blob_verification
                else None
            )
            mlx_cache_limit_bytes = source_evidence.mlx_cache_limit_bytes
            boundaries: tuple[tuple[Mapping[str, object] | None, Mapping[str, object] | None], ...] = (
                (before, after_audit),
                (after_audit, after_load),
                (after_load, after_layers),
                (after_layers, after),
            )
            phase_specs: tuple[tuple[str, str, str, str, str], ...] = (
                (
                    "before-non-vq-byte-audit",
                    "after-non-vq-byte-audit",
                    "all 1194 non-VQ tensor payloads",
                    contract.non_vq_package["manifest_sha256"],
                    non_vq_audit.package_set_sha256,
                ),
                (
                    "after-non-vq-byte-audit",
                    "after-source-load-and-bind",
                    "authenticated source blobs and non-VQ package; "
                    f"nocache_applied={str(nocache_applied).lower()}; "
                    f"inventory_hash_workers={inventory_hash_workers}",
                    contract.bound_identity_sha256,
                    load_output_identity,
                ),
                (
                    "after-source-load-and-bind",
                    "after-layer-forward-and-checkpoints",
                    "main decoder layers 0:77; "
                    f"nocache_applied={str(nocache_applied).lower()}; "
                    f"mlx_cache_limit_bytes={mlx_cache_limit_bytes}; "
                    f"decode_workers={source_evidence.decode_workers}; "
                    f"decode_queue_depth={source_evidence.decode_queue_depth}; "
                    "observed_peak_decoded_experts="
                    f"{source_evidence.observed_peak_decoded_experts}",
                    load_output_identity,
                    source_evidence.output_identity_sha256,
                ),
                (
                    "after-layer-forward-and-checkpoints",
                    "after-durable-prompt-shard-publication",
                    f"prompts[0:{selected_count}]",
                    source_evidence.output_identity_sha256,
                    shard_output_identity,
                ),
            )
            if capture_route_trace:
                assert trace_capture_identity is not None
                # The cache manifest must be immutable before it is bound into
                # the sidecar authority.  The trace phase therefore records
                # the observed capture identity and the publication boundary;
                # route buffers are released immediately after the writer
                # durably publishes the sidecar below.
                boundaries = (*boundaries, (after, after))
                phase_specs = (
                    *phase_specs,
                    (
                        "after-durable-prompt-shard-publication",
                        "after-route-trace-publication",
                        "source sparse layers 3:77 route traces",
                        source_evidence.output_identity_sha256,
                        trace_capture_identity,
                    ),
                )
            phases = tuple(
                GLM52ProducerPhase.create(
                    phase_id=phase_id,
                    ordinal=ordinal,
                    start_boundary=spec[0],
                    end_boundary=spec[1],
                    contribution_range=spec[2],
                    pageouts_delta=(
                        None
                        if resumed_or_checkpointed
                        else _delta(boundaries[ordinal][0], boundaries[ordinal][1], "pageouts")
                    ),
                    swapouts_delta=(
                        None
                        if resumed_or_checkpointed
                        else _delta(boundaries[ordinal][0], boundaries[ordinal][1], "swapouts")
                    ),
                    system_wired_default=wired_default,
                    input_identity_sha256=spec[3],
                    output_identity_sha256=spec[4],
                )
                for ordinal, (phase_id, spec) in enumerate(
                    zip(phase_ids, phase_specs, strict=True)
                )
            )
            if selected_count != len(contract.prompts):
                return ProducerResult(
                    manifest=None,
                    manifest_path=None,
                    shards=selected_shards,
                    resumed_prompt_ids=tuple(
                        shard.prompt_id for shard in selected_resumed
                    ),
                    produced_prompt_ids=tuple(produced_ids),
                    completed=False,
                )
            manifest = build_glm52_teacher_cache_manifest(
                contract=contract,
                shards=selected_shards,
                producer_phases=phases,
                created_at=created_at_factory(),
            )
            manifest_path = manifest_publisher(
                plan.cache_root,
                manifest,
                ledger_path=plan.ledger_path,
                bound_identity_sha256=contract.bound_identity_sha256,
            )
            if capture_route_trace:
                assert trace_root is not None
                assert trace_layers is not None
                assert trace_capture_identity is not None
                diagnostics = _load_route_diagnostics_api()
                writer = route_trace_writer or diagnostics.write_route_trace_artifact
                authority = _route_trace_capture_authority(
                    contract=contract,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    trace_root=trace_root,
                    capture_output_sha256=trace_capture_identity,
                )
                try:
                    writer(
                        trace_root,
                        trace_layers,
                        side="source",
                        authority=authority,
                    )
                except BaseException:
                    _remove_route_capture_transaction_state(
                        cache_root=plan.cache_root,
                        ledger_path=plan.ledger_path,
                        checkpoint_dir=plan.checkpoint_dir,
                        trace_root=trace_root,
                    )
                    raise
                finally:
                    trace_layers = None
                    source_evidence = SourceRunEvidence(
                        resumed_from_checkpoint=source_evidence.resumed_from_checkpoint,
                        output_identity_sha256=source_evidence.output_identity_sha256,
                    )
                    buffer_releaser()
            return ProducerResult(
                manifest=manifest,
                manifest_path=manifest_path,
                shards=selected_shards,
                resumed_prompt_ids=tuple(
                    shard.prompt_id for shard in selected_resumed
                ),
                produced_prompt_ids=tuple(produced_ids),
                completed=True,
            )


__all__ = [
    "FileProducerLockProvider",
    "GLM52SourceTeacherWorkload",
    "HEAVY_JOB_LOCK_PATH",
    "ProducerLockProvider",
    "ProducerPlan",
    "ProducerResult",
    "SourceRunEvidence",
    "checkpoint_identity_values_for_contract",
    "checkpoint_identities_for_contract",
    "collect_vm_stat_counts",
    "load_glm52_source_teacher_workload",
    "plan_glm52_teacher_cache",
    "produce_glm52_teacher_cache",
    "run_glm52_source_teacher_to_sink",
    "system_wired_default",
    "verify_glm52_source_blob_inventory",
]
