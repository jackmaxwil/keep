from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import pytest

from keep.build import doctor, get_model
from keep.build.cli import main as keep_main


DiskUsage = namedtuple("DiskUsage", "total used free")


def _readiness(*, ready: bool = True) -> get_model.ModelReadiness:
    return get_model.ModelReadiness(
        model_id="0xSero/glm-5.2-reap-504B-v2",
        revision="a" * 40,
        snapshot_path=Path("/cache/snapshot"),
        profile_name="glm52-reap-504b-v2",
        check_only=True,
        config_present=ready,
        index_present=ready,
        shards_present=61 if ready else 60,
        shards_total=61,
        missing_shards=() if ready else ("model-00061-of-00061.safetensors",),
        total_bytes=98_433_923_808,
        approx_bf16_gb=1008.0,
        machine_ram_gb=128.0,
        downloaded=(),
    )


def _probes(
    *,
    system: str = "Darwin",
    machine: str = "arm64",
    python_version: tuple[int, int, int] = (3, 11, 9),
    memsize: str = "137438953472",
    pressure: str = "System-wide memory free percentage: 75%",
    wired_limit: str = "0",
    uv: str | None = "/usr/local/bin/uv",
    free_bytes: int = 500 * 1000**3,
    mlx_info: tuple[str, str, bool] | Exception = ("0.31.2", "0.31.3", True),
    readiness: get_model.ModelReadiness | None = None,
) -> doctor.DoctorProbes:
    outputs = {
        ("sysctl", "-n", "hw.memsize"): memsize,
        ("memory_pressure", "-Q"): pressure,
        ("sysctl", "-n", "iogpu.wired_limit_mb"): wired_limit,
    }

    def command(argv: tuple[str, ...]) -> str:
        return outputs[argv]

    def model_probe(model: str) -> get_model.ModelReadiness:
        assert model == "glm52-reap-504b-v2"
        assert readiness is not None
        return readiness

    return doctor.DoctorProbes(
        system=lambda: system,
        machine=lambda: machine,
        python_version=lambda: python_version,
        command=command,
        find_executable=lambda _name: uv,
        disk_usage=lambda _path: DiskUsage(1000 * 1000**3, 0, free_bytes),
        mlx_info=lambda: mlx_info,
        model_readiness=model_probe,
    )


def test_collect_report_all_passes_for_ready_apple_silicon_host(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    cache.mkdir()

    report = doctor.collect_report(
        repo_root=tmp_path,
        hf_cache_root=cache,
        model="glm52-reap-504b-v2",
        probes=_probes(readiness=_readiness()),
    )

    assert report.exit_code == 0
    assert {check.status for check in report.checks} == {"pass"}
    assert report.check("model_snapshot").detail.endswith("offline; complete")
    assert "98.4 GB artifact + 9.8 GB headroom" in report.check("model_feasibility").detail


def test_collect_report_surfaces_nonfatal_capacity_and_policy_warnings(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    cache.mkdir()

    report = doctor.collect_report(
        repo_root=tmp_path,
        hf_cache_root=cache,
        model="glm52-reap-504b-v2",
        probes=_probes(
            memsize="100000000000",
            pressure="System-wide memory free percentage: 5%",
            wired_limit="122880",
            free_bytes=5 * 1000**3,
            readiness=_readiness(),
        ),
    )

    assert report.exit_code == 0
    assert report.check("disk_repo").status == "warn"
    assert report.check("memory_pressure").status == "warn"
    assert report.check("wired_limit").status == "warn"
    assert report.check("model_feasibility").status == "warn"


def test_collect_report_fails_for_unsupported_or_unavailable_requirements(tmp_path: Path) -> None:
    report = doctor.collect_report(
        repo_root=tmp_path,
        hf_cache_root=tmp_path / "missing-cache",
        model="glm52-reap-504b-v2",
        probes=_probes(
            system="Linux",
            machine="x86_64",
            python_version=(3, 10, 14),
            uv=None,
            mlx_info=RuntimeError("Metal unavailable"),
            readiness=_readiness(ready=False),
        ),
    )

    assert report.exit_code == 1
    assert report.check("platform").status == "fail"
    assert report.check("python").status == "fail"
    assert report.check("mlx").status == "fail"
    assert report.check("uv").status == "fail"
    assert report.check("hf_cache").status == "warn"
    assert report.check("model_snapshot").status == "fail"


def test_json_schema_is_machine_readable_and_complete(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    cache.mkdir()
    report = doctor.collect_report(
        repo_root=tmp_path,
        hf_cache_root=cache,
        probes=_probes(),
    )

    payload = json.loads(doctor.format_report(report, as_json=True))

    assert payload["schema_version"] == 1
    assert payload["summary"] == {"pass": len(report.checks), "warn": 0, "fail": 0}
    assert {"id", "status", "detail"} <= set(payload["checks"][0])
    assert "offline; no network checks were performed" in payload["offline_policy"]


def test_cli_wires_model_and_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}

    def fake_run(*, model: str | None, as_json: bool) -> int:
        captured.update(model=model, as_json=as_json)
        print('{"schema_version": 1}')
        return 0

    monkeypatch.setattr(doctor, "run", fake_run)

    assert keep_main(["doctor", "--model", "glm52-reap-504b-v2", "--json"]) == 0
    assert captured == {"model": "glm52-reap-504b-v2", "as_json": True}
    assert capsys.readouterr().out == '{"schema_version": 1}\n'
