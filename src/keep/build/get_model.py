"""Stage HuggingFace model snapshots for KEEP pipeline use.

The config/index/shard download flow is lifted from ``scripts/plan_stream_convert.py``
so the CLI uses the same HuggingFace primitives as the streaming converter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ramp.models.profiles import ModelProfile, ProfileError, get_profile

INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"
_COMMIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True)
class ModelGetRequest:
    model_id: str
    revision: str
    profile_name: str | None
    approx_bf16_gb: float | None
    check_only: bool
    download_workers: int = 8


@dataclass(frozen=True)
class ModelReadiness:
    model_id: str
    revision: str
    snapshot_path: Path
    profile_name: str | None
    check_only: bool
    config_present: bool
    index_present: bool
    shards_present: int
    shards_total: int
    missing_shards: tuple[str, ...]
    total_bytes: int
    approx_bf16_gb: float | None
    machine_ram_gb: float | None
    downloaded: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.config_present
            and self.index_present
            and self.shards_total > 0
            and self.shards_present == self.shards_total
        )


def resolve_model_request(
    name_or_hf_id: str,
    *,
    revision: str | None = None,
    check_only: bool = False,
    download_workers: int = 8,
) -> ModelGetRequest:
    try:
        profile = get_profile(name_or_hf_id)
    except ProfileError:
        profile = None

    if profile is not None:
        return _request_from_profile(
            profile,
            revision=revision or profile.revision or "main",
            check_only=check_only,
            download_workers=download_workers,
        )

    return ModelGetRequest(
        model_id=name_or_hf_id,
        revision=revision or "main",
        profile_name=None,
        approx_bf16_gb=None,
        check_only=check_only,
        download_workers=download_workers,
    )


def stage_model(request: ModelGetRequest) -> ModelReadiness:
    requested_revision_is_commit = bool(
        _COMMIT_REVISION_RE.fullmatch(request.revision)
    )
    resolved_revision = (
        resolve_cached_revision(request.model_id, request.revision)
        if request.check_only or requested_revision_is_commit
        else request.revision
    )
    snapshot_path = expected_snapshot_path(request.model_id, resolved_revision)
    downloaded: list[str] = []

    if not request.check_only:
        download_revision = resolved_revision
        config_path = snapshot_path / CONFIG_FILENAME
        if not requested_revision_is_commit or not config_path.exists():
            config_path = Path(
                hf_hub_download(
                    request.model_id,
                    CONFIG_FILENAME,
                    revision=(
                        request.revision
                        if not requested_revision_is_commit
                        else download_revision
                    ),
                )
            )
            downloaded.append(CONFIG_FILENAME)
        snapshot_path = config_path.parent
        resolved_revision = resolved_snapshot_revision(
            snapshot_path,
            fallback=resolved_revision,
        )
        download_revision = resolved_revision

        index_path = snapshot_path / INDEX_FILENAME
        if not index_path.exists():
            index_path = Path(
                hf_hub_download(
                    request.model_id,
                    INDEX_FILENAME,
                    revision=download_revision,
                )
            )
            downloaded.append(INDEX_FILENAME)

        snapshot_path = index_path.parent
        inventory = inventory_snapshot(snapshot_path)
        if inventory.missing_shards:
            snapshot_path = Path(
                snapshot_download(
                    request.model_id,
                    revision=download_revision,
                    allow_patterns=list(inventory.missing_shards),
                    local_dir=None,
                    max_workers=request.download_workers,
                )
            )
            downloaded.extend(inventory.missing_shards)
        resolved_revision = resolved_snapshot_revision(
            snapshot_path,
            fallback=resolved_revision,
        )

    inventory = inventory_snapshot(snapshot_path)
    return ModelReadiness(
        model_id=request.model_id,
        revision=resolved_revision,
        snapshot_path=snapshot_path,
        profile_name=request.profile_name,
        check_only=request.check_only,
        config_present=inventory.config_present,
        index_present=inventory.index_present,
        shards_present=inventory.shards_present,
        shards_total=inventory.shards_total,
        missing_shards=inventory.missing_shards,
        total_bytes=inventory.total_bytes,
        approx_bf16_gb=request.approx_bf16_gb,
        machine_ram_gb=machine_ram_gb(),
        downloaded=tuple(downloaded),
    )


@dataclass(frozen=True)
class SnapshotInventory:
    config_present: bool
    index_present: bool
    shards_present: int
    shards_total: int
    missing_shards: tuple[str, ...]
    total_bytes: int


def inventory_snapshot(snapshot_path: Path) -> SnapshotInventory:
    config_present = (snapshot_path / CONFIG_FILENAME).exists()
    index_path = snapshot_path / INDEX_FILENAME
    index_present = index_path.exists()
    shard_names = shard_names_from_index(index_path) if index_present else ()
    present = 0
    missing: list[str] = []
    total_bytes = 0
    for shard_name in shard_names:
        path = snapshot_path / shard_name
        if path.exists():
            present += 1
            total_bytes += path.stat().st_size
        else:
            missing.append(shard_name)
    return SnapshotInventory(
        config_present=config_present,
        index_present=index_present,
        shards_present=present,
        shards_total=len(shard_names),
        missing_shards=tuple(missing),
        total_bytes=total_bytes,
    )


def shard_names_from_index(index_path: Path) -> tuple[str, ...]:
    try:
        raw = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ()
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict):
        return ()
    return tuple(sorted({str(value) for value in weight_map.values()}))


def expected_snapshot_path(model_id: str, revision: str) -> Path:
    return model_cache_root(model_id) / "snapshots" / revision


def model_cache_root(model_id: str) -> Path:
    return hf_cache_root() / f"models--{model_id.replace('/', '--')}"


def resolve_cached_revision(model_id: str, revision: str) -> str:
    """Resolve a cached Hugging Face ref without network access."""

    if _COMMIT_REVISION_RE.fullmatch(revision):
        return revision
    relative = Path(revision)
    if relative.is_absolute() or ".." in relative.parts:
        return revision
    ref_path = model_cache_root(model_id) / "refs" / relative
    try:
        resolved = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return revision
    return resolved if _COMMIT_REVISION_RE.fullmatch(resolved) else revision


def resolved_snapshot_revision(snapshot_path: Path, *, fallback: str) -> str:
    if (
        snapshot_path.parent.name == "snapshots"
        and _COMMIT_REVISION_RE.fullmatch(snapshot_path.name)
    ):
        return snapshot_path.name
    return fallback


def hf_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def machine_ram_gb() -> float | None:
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(raw) / 1_000_000_000
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1_000_000_000
    except (OSError, ValueError):
        return None


def format_readiness(readiness: ModelReadiness) -> str:
    model_label = (
        f"{readiness.profile_name} ({readiness.model_id})"
        if readiness.profile_name
        else readiness.model_id
    )
    lines = [
        f"model: {model_label}",
        f"revision: {readiness.revision}",
        f"snapshot: {readiness.snapshot_path}",
        f"{CONFIG_FILENAME}: {'present' if readiness.config_present else 'missing'}",
        f"{INDEX_FILENAME}: {'present' if readiness.index_present else 'missing'}",
        f"shards: {readiness.shards_present}/{readiness.shards_total} present",
        f"total: {readiness.total_bytes / 1_000_000_000:.6f} GB",
    ]
    if readiness.missing_shards:
        lines.append("missing: " + ", ".join(readiness.missing_shards[:10]))
    if readiness.downloaded:
        lines.append("downloaded: " + ", ".join(readiness.downloaded[:10]))
    if readiness.approx_bf16_gb is not None:
        lines.append(_format_fit_line(readiness.approx_bf16_gb, readiness.machine_ram_gb))
    lines.append("ready: " + ("yes" if readiness.ready else "no"))
    return "\n".join(lines) + "\n"


def _format_fit_line(approx_bf16_gb: float, ram_gb: float | None) -> str:
    teacher = _format_gb(approx_bf16_gb)
    if ram_gb is None:
        return f"teacher ~{teacher}GB, machine unknown: fit unknown"
    machine = _format_gb(ram_gb)
    fit = "fits single-host" if ram_gb >= approx_bf16_gb else "needs off-box/distributed"
    return f"teacher ~{teacher}GB, machine {machine}GB: {fit}"


def _format_gb(value: float) -> str:
    return f"{value:g}"


def _request_from_profile(
    profile: ModelProfile,
    *,
    revision: str,
    check_only: bool,
    download_workers: int,
) -> ModelGetRequest:
    return ModelGetRequest(
        model_id=profile.hf_model_id,
        revision=revision,
        profile_name=profile.name,
        approx_bf16_gb=profile.approx_bf16_gb,
        check_only=check_only,
        download_workers=download_workers,
    )
