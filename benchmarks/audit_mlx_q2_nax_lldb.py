from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.benchmark.nax_audit import ProjectionName, q2_nax_runtime_cases


KERNEL_RE = re.compile(r"\b[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+\b")
Q2_NAX_RE = re.compile(r"affine_gather_qmm_(?:rhs|t)_nax_.*_gs_128_b_2_.*bk_?64.*")


def _projections(value: str) -> Iterable[ProjectionName]:
    if value == "all":
        return ("gate_up", "down")
    return (value,)  # type: ignore[return-value]


def _lldb_command_file() -> str:
    return "\n".join(
        [
            'breakpoint set -r "newFunctionWithName:"',
            "breakpoint command add 1",
            "expression -O -- (id)$x2",
            "continue",
            "DONE",
            "run",
            "",
        ]
    )


def _extract_kernel_names(log: str) -> list[str]:
    names: list[str] = []
    for line in log.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("(") or stripped.startswith("Command #"):
            continue
        for match in KERNEL_RE.findall(stripped):
            if match.startswith(("affine_", "gather_", "sort_", "merge_", "partition_", "v_", "vn_", "vs_", "vsn_", "vvn_", "all_reduce_")):
                names.append(match)
    return sorted(set(names))


def _run_case(
    projection: ProjectionName,
    tokens: int,
    *,
    log_dir: Path,
    python_executable: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_projection = projection.replace("_", "-")
    log_path = log_dir / f"glm45-air-nax-q2-lldb-current-{safe_projection}-{tokens}.txt"
    with tempfile.NamedTemporaryFile("w", suffix=".lldb", delete=False) as command_file:
        command_file.write(_lldb_command_file())
        command_path = Path(command_file.name)

    command = [
        "lldb",
        "--batch",
        "-s",
        str(command_path),
        "--",
        python_executable,
        "benchmarks/audit_mlx_q2_nax_runtime.py",
        "--run",
        "--projection",
        projection,
        "--tokens",
        str(tokens),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    finally:
        command_path.unlink(missing_ok=True)

    log = completed.stdout + completed.stderr
    log_path.write_text(log)
    kernel_names = _extract_kernel_names(log)
    q2_nax_kernels = [name for name in kernel_names if Q2_NAX_RE.fullmatch(name)]
    return {
        "record_type": "q2_nax_lldb_current",
        "projection": projection,
        "tokens": tokens,
        "route_count": tokens * 8,
        "command": command,
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "kernel_names": q2_nax_kernels,
        "all_kernel_names": kernel_names,
        "requests_bk32": any(("bk_32" in name or "bk32" in name) for name in kernel_names),
        "passes_bk64_rhs_nax": any(name.startswith("affine_gather_qmm_rhs_nax_") for name in q2_nax_kernels),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MLX q2 NAX runtime-name audit with LLDB.")
    parser.add_argument("--projection", choices=["gate_up", "down", "all"], default="all")
    parser.add_argument("--tokens", type=int, action="append", default=None)
    parser.add_argument("--log-dir", default="artifacts/benchmarks")
    parser.add_argument("--append-jsonl", default="artifacts/benchmarks/glm45-air-nax-q2-lldb-current.jsonl")
    parser.add_argument("--python-executable", default=".venv/bin/python")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    tokens = args.tokens or [1024, 2048, 4096]
    projections = tuple(_projections(args.projection))
    expected_cases = {(case.projection, case.tokens) for case in q2_nax_runtime_cases(tokens=tokens, projections=projections)}
    records = []
    for projection, token_count in sorted(expected_cases):
        record = _run_case(
            projection,
            token_count,
            log_dir=Path(args.log_dir),
            python_executable=args.python_executable,
            timeout_seconds=args.timeout_seconds,
        )
        records.append(record)
        if args.append_jsonl:
            append_jsonl(args.append_jsonl, record)
        print(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
