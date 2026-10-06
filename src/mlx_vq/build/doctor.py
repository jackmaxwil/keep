"""Offline environment and capacity preflight for the ``keep`` CLI."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from mlx_vq.build import get_model

CheckStatus = Literal["pass", "warn", "fail"]
_GIGABYTE = 1_000_000_000
_MIN_FREE_DISK_BYTES = 20 * _GIGABYTE
_LOW_MEMORY_FREE_PERCENT = 10.0
_RUNTIME_HEADROOM_RATIO = 0.10
_PROFILE_ARTIFACT_BYTES = {
    # Accepted whole-model tensor payload for the current GLM52 REAP profile.
    "glm52-reap-504b-v2": 98_433_923_808,
}


@dataclass(frozen=True)
class DoctorCheck:
    id: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def exit_code(self) -> int:
        return 1 if any(check.status == "fail" for check in self.checks) else 0

    def check(self, check_id: str) -> DoctorCheck:
        return next(check for check in self.checks if check.id == check_id)

    def as_dict(self) -> dict[str, object]:
        summary = {status: 0 for status in ("pass", "warn", "fail")}
        for check in self.checks:
            summary[check.status] += 1
        return {
            "schema_version": 1,
            "offline_policy": "offline; no network checks were performed",
            "exit_code": self.exit_code,
            "summary": summary,
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class DoctorProbes:
    system: Callable[[], str]
    machine: Callable[[], str]
    python_version: Callable[[], tuple[int, int, int]]
    command: Callable[[tuple[str, ...]], str]
    find_executable: Callable[[str], str | None]
    disk_usage: Callable[[Path], shutil._ntuple_diskusage]
    mlx_info: Callable[[], tuple[str, str, bool]]
    model_readiness: Callable[[str], get_model.ModelReadiness]


def _command(argv: tuple[str, ...]) -> str:
    return subprocess.check_output(argv, stderr=subprocess.DEVNULL, text=True).strip()


def _mlx_info() -> tuple[str, str, bool]:
    import mlx.core as mx

    available = bool(mx.metal.is_available())
    if available:
        # ``is_available`` can be true in a sandboxed macOS process that cannot
        # actually initialize a Metal device. This query is read-only but forces
        # that initialization so the preflight reports the real condition.
        mx.device_info()
    return mx.__version__, importlib.metadata.version("mlx-lm"), available


def _model_readiness(model: str) -> get_model.ModelReadiness:
    request = get_model.resolve_model_request(model, check_only=True)
    return get_model.stage_model(request)


def default_probes() -> DoctorProbes:
    return DoctorProbes(
        system=platform.system,
        machine=platform.machine,
        python_version=lambda: tuple(sys.version_info[:3]),
        command=_command,
        find_executable=shutil.which,
        disk_usage=lambda path: shutil.disk_usage(path),
        mlx_info=_mlx_info,
        model_readiness=_model_readiness,
    )


def _read_command(
    probes: DoctorProbes, argv: tuple[str, ...]
) -> tuple[str | None, str | None]:
    try:
        return probes.command(argv).strip(), None
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        return None, str(error)


def _disk_check(
    check_id: str, path: Path, probes: DoctorProbes
) -> DoctorCheck:
    try:
        free = probes.disk_usage(path).free
    except OSError as error:
        return DoctorCheck(check_id, "warn", f"could not inspect {path}: {error}")
    status: CheckStatus = "pass" if free >= _MIN_FREE_DISK_BYTES else "warn"
    return DoctorCheck(
        check_id,
        status,
        f"{_format_gb(free)} GB free on {path} (recommended >= {_format_gb(_MIN_FREE_DISK_BYTES)} GB)",
    )


def _format_gb(value: int | float) -> str:
    return f"{float(value) / _GIGABYTE:.1f}"


def _memory_pressure_check(probes: DoctorProbes) -> DoctorCheck:
    raw, error = _read_command(probes, ("memory_pressure", "-Q"))
    if error or raw is None:
        return DoctorCheck("memory_pressure", "warn", "unavailable: memory_pressure query failed")
    match = re.search(r"(\d+(?:\.\d+)?)%", raw)
    if not match:
        return DoctorCheck("memory_pressure", "warn", f"unparseable output: {raw}")
    free_percent = float(match.group(1))
    status: CheckStatus = "pass" if free_percent >= _LOW_MEMORY_FREE_PERCENT else "warn"
    return DoctorCheck(
        "memory_pressure",
        status,
        f"{free_percent:g}% memory free (warn below {_LOW_MEMORY_FREE_PERCENT:g}%)",
    )


def _model_checks(
    model: str, probes: DoctorProbes, ram_bytes: int | None
) -> list[DoctorCheck]:
    try:
        readiness = probes.model_readiness(model)
    except Exception as error:  # A preflight must report an unavailable local cache.
        return [
            DoctorCheck(
                "model_snapshot",
                "fail",
                f"offline cache inspection failed for {model}: {error}",
            )
        ]

    model_label = readiness.profile_name or readiness.model_id
    if readiness.ready:
        snapshot = DoctorCheck(
            "model_snapshot",
            "pass",
            f"{model_label}: {readiness.shards_present}/{readiness.shards_total} shards; offline; complete",
        )
    else:
        snapshot = DoctorCheck(
            "model_snapshot",
            "fail",
            f"{model_label}: {readiness.shards_present}/{readiness.shards_total} shards; offline; incomplete",
        )

    artifact_bytes = _PROFILE_ARTIFACT_BYTES.get(readiness.profile_name or "")
    if artifact_bytes is None and readiness.approx_bf16_gb is not None:
        artifact_bytes = int(readiness.approx_bf16_gb * _GIGABYTE)
        basis = "profile BF16 estimate"
    elif artifact_bytes is not None:
        basis = "profile artifact payload"
    else:
        return [
            snapshot,
            DoctorCheck(
                "model_feasibility",
                "warn",
                f"{model_label}: no profile resident-size estimate; feasibility skipped offline",
            ),
        ]

    headroom_bytes = int(artifact_bytes * _RUNTIME_HEADROOM_RATIO)
    required_bytes = artifact_bytes + headroom_bytes
    required_detail = (
        f"{model_label}: {_format_gb(artifact_bytes)} GB artifact + "
        f"{_format_gb(headroom_bytes)} GB headroom = {_format_gb(required_bytes)} GB required ({basis})"
    )
    if ram_bytes is None:
        feasibility = DoctorCheck("model_feasibility", "warn", required_detail + "; RAM unavailable")
    elif ram_bytes < required_bytes:
        feasibility = DoctorCheck(
            "model_feasibility",
            "warn",
            required_detail + f"; {_format_gb(ram_bytes)} GB RAM available",
        )
    else:
        feasibility = DoctorCheck(
            "model_feasibility",
            "pass",
            required_detail + f"; {_format_gb(ram_bytes)} GB RAM available",
        )
    return [snapshot, feasibility]


def collect_report(
    *,
    repo_root: Path,
    hf_cache_root: Path | None = None,
    model: str | None = None,
    probes: DoctorProbes | None = None,
) -> DoctorReport:
    """Collect local-only readiness checks without changing host or cache state."""

    probes = probes or default_probes()
    hf_cache_root = hf_cache_root or get_model.hf_cache_root()
    checks: list[DoctorCheck] = []

    system = probes.system()
    machine = probes.machine()
    supported = system == "Darwin" and machine.lower() in {"arm64", "aarch64"}
    checks.append(
        DoctorCheck(
            "platform",
            "pass" if supported else "fail",
            f"{system} {machine}" + ("" if supported else "; requires Darwin on Apple Silicon"),
        )
    )

    python_version = probes.python_version()
    python_supported = python_version >= (3, 11, 0)
    checks.append(
        DoctorCheck(
            "python",
            "pass" if python_supported else "fail",
            f"Python {'.'.join(map(str, python_version))}" + ("" if python_supported else "; requires >= 3.11"),
        )
    )

    try:
        mlx_version, mlx_lm_version, metal_available = probes.mlx_info()
        mlx_ok = metal_available
        mlx_detail = f"mlx {mlx_version}; mlx-lm {mlx_lm_version}; Metal {'available' if metal_available else 'unavailable'}"
    except Exception as error:  # Imports can fail before a device probe is possible.
        mlx_ok = False
        mlx_detail = f"MLX/Metal unavailable: {error}"
    checks.append(DoctorCheck("mlx", "pass" if mlx_ok else "fail", mlx_detail))

    uv_path = probes.find_executable("uv")
    checks.append(
        DoctorCheck("uv", "pass" if uv_path else "fail", uv_path or "uv not found on PATH")
    )
    checks.append(_disk_check("disk_repo", repo_root, probes))
    checks.append(_disk_check("disk_hf_cache", hf_cache_root, probes))

    mem_raw, mem_error = _read_command(probes, ("sysctl", "-n", "hw.memsize"))
    try:
        ram_bytes = int(mem_raw) if mem_raw is not None else None
    except ValueError:
        ram_bytes = None
    checks.append(
        DoctorCheck(
            "memory_ram",
            "pass" if ram_bytes is not None else "warn",
            f"{_format_gb(ram_bytes)} GB total RAM" if ram_bytes is not None else f"unavailable: {mem_error or mem_raw}",
        )
    )
    checks.append(_memory_pressure_check(probes))

    if system == "Darwin":
        wired_raw, wired_error = _read_command(probes, ("sysctl", "-n", "iogpu.wired_limit_mb"))
        try:
            wired_limit = int(wired_raw) if wired_raw is not None else None
        except ValueError:
            wired_limit = None
        if wired_limit is None:
            checks.append(DoctorCheck("wired_limit", "warn", f"unavailable: {wired_error or wired_raw}"))
        elif wired_limit == 0:
            checks.append(DoctorCheck("wired_limit", "pass", "iogpu.wired_limit_mb=0 (system default)"))
        else:
            checks.append(DoctorCheck("wired_limit", "warn", f"iogpu.wired_limit_mb={wired_limit} override is active"))
    else:
        checks.append(DoctorCheck("wired_limit", "warn", "skipped: Darwin-only policy"))

    cache_status: CheckStatus = "pass" if hf_cache_root.exists() else "warn"
    cache_detail = f"present: {hf_cache_root}" if hf_cache_root.exists() else f"missing: {hf_cache_root}; offline check only"
    checks.append(DoctorCheck("hf_cache", cache_status, cache_detail))

    if model:
        checks.extend(_model_checks(model, probes, ram_bytes))
    return DoctorReport(tuple(checks))


def format_report(report: DoctorReport, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
    lines = [f"{check.status.upper():4} {check.id}: {check.detail}" for check in report.checks]
    lines.append("offline: no network checks were performed")
    return "\n".join(lines) + "\n"


def run(*, model: str | None, as_json: bool) -> int:
    report = collect_report(repo_root=Path.cwd(), model=model)
    print(format_report(report, as_json=as_json), end="")
    return report.exit_code
