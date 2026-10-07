from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mlx_vq.benchmark.glm45_air import append_jsonl


DEFAULT_CHECKS = (
    ("xcode_select_path", ["xcode-select", "-p"]),
    ("xctrace_version", ["xcrun", "xctrace", "version"]),
    ("metal_path", ["xcrun", "--find", "metal"]),
    ("metal_execution", ["xcrun", "metal", "--help"]),
    ("metallib_path", ["xcrun", "--find", "metallib"]),
    ("metal_shaderconverter_path", ["xcrun", "--find", "metal-shaderconverter"]),
    ("metal_shaderconverter_version", ["metal-shaderconverter", "--version"]),
    ("metal_shaderconverter_pkg", ["pkgutil", "--pkg-info", "com.apple.metal"]),
    ("metal_toolchain_component", ["xcodebuild", "-showComponent", "MetalToolchain"]),
    ("macos_version", ["sw_vers"]),
    ("mlx_cmake_dir", ["uv", "run", "python", "-m", "mlx", "--cmake-dir"]),
    ("nanobind_cmake_dir", ["uv", "run", "python", "-m", "nanobind", "--cmake_dir"]),
)

USER_XCODE_26_4_1 = Path.home() / "Applications" / "Xcode-26.4.1.app" / "Contents" / "Developer"


def _summarize(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return " ".join(stripped.splitlines()[:3])[:400]


def _run_check(
    name: str,
    command: list[str],
    *,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
        )
        output = completed.stdout or completed.stderr
        record = {
            "name": name,
            "command": command,
            "returncode": completed.returncode,
            "summary": _summarize(output),
        }
        if env_overrides:
            record["env_overrides"] = env_overrides
        return record
    except FileNotFoundError as exc:
        record = {
            "name": name,
            "command": command,
            "returncode": 127,
            "summary": str(exc),
        }
        if env_overrides:
            record["env_overrides"] = env_overrides
        return record
    except subprocess.TimeoutExpired:
        record = {
            "name": name,
            "command": command,
            "returncode": 124,
            "summary": f"timed out after {timeout_seconds}s",
        }
        if env_overrides:
            record["env_overrides"] = env_overrides
        return record


def _static_path_check(name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "command": ["path-exists", str(path)],
        "returncode": 0 if path.exists() else 1,
        "summary": str(path) if path.exists() else f"missing: {path}",
    }


def _direct_tool_check(name: str, developer_dir: Path, tool: str) -> dict[str, Any]:
    matches = sorted(developer_dir.rglob(tool)) if developer_dir.exists() else []
    executable_matches = [path for path in matches if path.is_file() and os.access(path, os.X_OK)]
    if executable_matches:
        summary = "; ".join(str(path) for path in executable_matches[:3])
        if len(executable_matches) > 3:
            summary += f"; ... {len(executable_matches)} total"
        return {
            "name": name,
            "command": ["direct-tool-search", str(developer_dir), tool],
            "returncode": 0,
            "summary": summary,
        }
    return {
        "name": name,
        "command": ["direct-tool-search", str(developer_dir), tool],
        "returncode": 1,
        "summary": f"no executable {tool} under {developer_dir}",
    }


def _xcode_26_4_1_checks(*, timeout_seconds: int) -> list[dict[str, Any]]:
    developer_dir = USER_XCODE_26_4_1
    env = {"DEVELOPER_DIR": str(developer_dir)}
    checks = [
        _static_path_check("xcode_26_4_1_developer_dir", developer_dir),
        _direct_tool_check("xcode_26_4_1_direct_xctrace", developer_dir, "xctrace"),
    ]
    if developer_dir.exists():
        checks.extend(
            [
                _run_check(
                    "xcode_26_4_1_license_check",
                    ["xcodebuild", "-license", "check"],
                    timeout_seconds=timeout_seconds,
                    env_overrides=env,
                ),
                _run_check(
                    "xcode_26_4_1_metal_path",
                    ["xcrun", "--find", "metal"],
                    timeout_seconds=timeout_seconds,
                    env_overrides=env,
                ),
                _run_check(
                    "xcode_26_4_1_metal_execution",
                    ["xcrun", "metal", "--help"],
                    timeout_seconds=timeout_seconds,
                    env_overrides=env,
                ),
                _run_check(
                    "xcode_26_4_1_metallib_path",
                    ["xcrun", "--find", "metallib"],
                    timeout_seconds=timeout_seconds,
                    env_overrides=env,
                ),
                _run_check(
                    "xcode_26_4_1_xctrace_version",
                    ["xcrun", "xctrace", "version"],
                    timeout_seconds=timeout_seconds,
                    env_overrides=env,
                ),
            ]
        )
    return checks


def _metal_compile_smoke(*, timeout_seconds: int) -> dict[str, Any]:
    source = "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "kernel void add_one(device float* x [[buffer(0)]], uint id [[thread_position_in_grid]]) {",
            "  x[id] += 1.0f;",
            "}",
            "",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="glm-metal-smoke-") as tmp:
        root = Path(tmp)
        metal_path = root / "smoke.metal"
        air_path = root / "smoke.air"
        metallib_path = root / "smoke.metallib"
        metal_path.write_text(source)

        compile_step = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(metal_path), "-o", str(air_path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if compile_step.returncode != 0:
            return {
                "name": "metal_compile_smoke",
                "command": ["xcrun", "-sdk", "macosx", "metal", "-c", "smoke.metal", "-o", "smoke.air"],
                "returncode": compile_step.returncode,
                "summary": _summarize(compile_step.stdout or compile_step.stderr),
            }

        link_step = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metallib", str(air_path), "-o", str(metallib_path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        returncode = link_step.returncode
        summary_parts = [
            f"metal_rc={compile_step.returncode}",
            f"metallib_rc={link_step.returncode}",
            f"air_exists={air_path.exists()}",
            f"metallib_exists={metallib_path.exists()}",
        ]
        output = _summarize(link_step.stdout or link_step.stderr)
        if output:
            summary_parts.append(output)
        return {
            "name": "metal_compile_smoke",
            "command": [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "smoke.metal",
                "&&",
                "xcrun",
                "-sdk",
                "macosx",
                "metallib",
                "smoke.air",
            ],
            "returncode": returncode,
            "summary": "; ".join(summary_parts),
        }


def run_audit(*, timeout_seconds: int = 30, try_xcode_install: bool = False) -> dict[str, Any]:
    checks = [
        _run_check(name, command, timeout_seconds=timeout_seconds)
        for name, command in DEFAULT_CHECKS
    ]
    checks.extend(_xcode_26_4_1_checks(timeout_seconds=timeout_seconds))
    checks.append(_metal_compile_smoke(timeout_seconds=timeout_seconds))
    if try_xcode_install:
        checks.append(
            _run_check(
                "xcodes_install_26_4_1_probe",
                [
                    "xcodes",
                    "install",
                    "26.4.1",
                    "--directory",
                    "/Users/jack.mazac/Applications",
                    "--no-superuser",
                    "--experimental-unxip",
                ],
                timeout_seconds=timeout_seconds,
            )
        )
    return {
        "record_type": "toolchain_audit",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local Xcode/Metal/MLX/NAX build prerequisites.")
    parser.add_argument("--append-jsonl", default="artifacts/benchmarks/glm45-air-nax-toolchain-audit.jsonl")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--try-xcode-install",
        action="store_true",
        help="Probe the documented xcodes install route; stops if Apple ID interaction is required.",
    )
    args = parser.parse_args()

    record = run_audit(
        timeout_seconds=args.timeout_seconds,
        try_xcode_install=args.try_xcode_install,
    )
    if args.append_jsonl:
        Path(args.append_jsonl).parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(args.append_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
