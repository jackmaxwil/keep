from __future__ import annotations

import plistlib
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import mlx.core as mx
import numpy as np
from mlx_lm.models.switch_layers import _gather_sort


_KERNEL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]*nax[A-Za-z0-9_]*")
_GROUP_SIZE_RE = re.compile(r"(?:^|_)gs_?([0-9]+)(?:_|$)")
_BK_RE = re.compile(r"(?:^|_)bk_?([0-9]+)(?:_|$)")
_TRACE_KERNEL_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]*(?:gather_qmm|qmm)[A-Za-z0-9_./:-]*nax[A-Za-z0-9_./:-]*")
_BK32_RE = re.compile(r"(?:^|_)bk_?32(?:_|$)")
_BK64_RE = re.compile(r"(?:^|_)bk_?64(?:_|$)")

ProjectionName = Literal["gate_up", "down"]
AIR_PROJECTION_DIMS: dict[ProjectionName, tuple[int, int]] = {
    "gate_up": (4096, 1408),
    "down": (1408, 4096),
}


@dataclass(frozen=True)
class Q2NaxRuntimeCase:
    projection: ProjectionName
    tokens: int
    input_dims: int
    output_dims: int
    top_k: int = 8
    experts: int = 128
    group_size: int = 128
    bits: int = 2
    mode: str = "affine"
    seed: int = 20260625

    @property
    def route_count(self) -> int:
        return self.tokens * self.top_k

    @property
    def trace_name(self) -> str:
        return f"mlx-q2-nax-{self.projection}-m{self.tokens}.gputrace"

    @property
    def allowed_kernel_regexes(self) -> list[str]:
        return [
            rf"affine_gather_qmm_rhs_nax_.*_gs_{self.group_size}_b_{self.bits}_.*bk_?64",
            rf"affine_gather_qmm_t_nax_.*_gs_{self.group_size}_b_{self.bits}_.*bk_?64",
        ]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["route_count"] = self.route_count
        record["trace_name"] = self.trace_name
        record["allowed_kernel_regexes"] = self.allowed_kernel_regexes
        return record


def mlx_metallib_path() -> Path:
    core_path = Path(mx.__file__).resolve()
    path = core_path.parent / "lib" / "mlx.metallib"
    if not path.exists():
        raise FileNotFoundError(f"Unable to find MLX metallib at {path}")
    return path


def _kernel_tokens(path: Path) -> list[str]:
    data = path.read_bytes().decode("latin-1", errors="ignore")
    return sorted(set(_KERNEL_TOKEN_RE.findall(data)))


def _group_sizes(names: list[str]) -> list[int]:
    sizes = set()
    for name in names:
        for match in _GROUP_SIZE_RE.finditer(name):
            sizes.add(int(match.group(1)))
    return sorted(sizes)


def _has_bk(names: list[str], value: int) -> bool:
    expected = str(value)
    for name in names:
        if any(match.group(1) == expected for match in _BK_RE.finditer(name)):
            return True
    return False


def mlx_metallib_nax_inventory() -> dict[str, Any]:
    path = mlx_metallib_path()
    names = _kernel_tokens(path)
    gather_qmm = [name for name in names if "gather_qmm" in name]
    qmm = [name for name in names if "qmm_" in name and "gather_qmm" not in name]
    steel_gemm = [name for name in names if name.startswith("steel_gemm_fused_nax")]
    return {
        "mlx_core_path": str(Path(mx.__file__).resolve()),
        "mlx_metallib_path": str(path),
        "nax_kernel_count": len(names),
        "gather_qmm_nax_count": len(gather_qmm),
        "qmm_nax_count": len(qmm),
        "steel_gemm_fused_nax_count": len(steel_gemm),
        "gather_qmm_nax_group_sizes": _group_sizes(gather_qmm),
        "qmm_nax_group_sizes": _group_sizes(qmm),
        "has_gather_qmm_nax_bk32": _has_bk(gather_qmm, 32),
        "has_gather_qmm_nax_bk64": _has_bk(gather_qmm, 64),
        "gather_qmm_nax_samples": gather_qmm[:12],
        "steel_gemm_fused_nax_samples": steel_gemm[:12],
    }


def q2_nax_runtime_cases(
    *,
    tokens: Iterable[int] = (1024, 2048, 4096),
    projections: Iterable[ProjectionName] = ("gate_up", "down"),
    top_k: int = 8,
    experts: int = 128,
    group_size: int = 128,
    seed: int = 20260625,
) -> list[Q2NaxRuntimeCase]:
    cases: list[Q2NaxRuntimeCase] = []
    for projection in projections:
        input_dims, output_dims = AIR_PROJECTION_DIMS[projection]
        for token_count in tokens:
            cases.append(
                Q2NaxRuntimeCase(
                    projection=projection,
                    tokens=int(token_count),
                    input_dims=input_dims,
                    output_dims=output_dims,
                    top_k=top_k,
                    experts=experts,
                    group_size=group_size,
                    seed=seed + int(token_count) + (0 if projection == "gate_up" else 10_000),
                )
            )
    return cases


def q2_nax_capture_plan_records(
    *,
    trace_dir: str | Path = "artifacts/traces/mlx-q2-nax",
    tokens: Iterable[int] = (1024, 2048, 4096),
    projections: Iterable[ProjectionName] = ("gate_up", "down"),
) -> list[dict[str, Any]]:
    trace_root = Path(trace_dir)
    records = []
    for case in q2_nax_runtime_cases(tokens=tokens, projections=projections):
        trace_path = trace_root / case.trace_name
        record = case.to_record()
        record.update(
            {
                "trace_path": str(trace_path),
                "capture_env": {"MTL_CAPTURE_ENABLED": "1"},
                "capture_command": (
                    "MTL_CAPTURE_ENABLED=1 uv run python benchmarks/audit_mlx_q2_nax_runtime.py "
                    f"--capture --projection {case.projection} --tokens {case.tokens} --trace-dir {trace_root}"
                ),
                "manual_xcode_check": (
                    "Open the .gputrace in Xcode and verify the encoded q2 projection kernel is "
                    "gather_qmm_rhs_nax or gather_qmm_t_nax with bk64, not bk32 or non-NAX."
                ),
            }
        )
        records.append(record)
    return records


def _trace_metadata(trace_path: Path) -> dict[str, Any] | None:
    metadata_path = trace_path / "metadata"
    if not metadata_path.exists():
        return None
    try:
        return plistlib.loads(metadata_path.read_bytes())
    except Exception as exc:
        return {"parse_error": str(exc)}


def _extract_trace_symbols(trace_path: Path, *, max_scan_bytes: int) -> tuple[list[str], list[dict[str, Any]]]:
    symbols: set[str] = set()
    files = []
    for path in sorted(trace_path.iterdir()):
        if path.is_dir():
            continue
        stat = path.stat()
        info: dict[str, Any] = {"path": str(path), "bytes": stat.st_size}
        if path.name.startswith("MTLBuffer-"):
            info["scanned"] = False
            info["skip_reason"] = "buffer"
            files.append(info)
            continue
        if stat.st_size > max_scan_bytes:
            info["scanned"] = False
            info["skip_reason"] = "too_large"
            files.append(info)
            continue
        data = path.read_bytes().decode("latin-1", errors="ignore")
        found = sorted(set(_TRACE_KERNEL_TOKEN_RE.findall(data)))
        symbols.update(found)
        info["scanned"] = True
        info["nax_symbol_count"] = len(found)
        files.append(info)
    return sorted(symbols), files


def summarize_gputrace(
    trace_path: str | Path,
    *,
    allowed_kernel_regexes: Iterable[str] = (),
    max_scan_bytes: int = 256 * 1024 * 1024,
    max_symbols: int = 64,
) -> dict[str, Any]:
    trace = Path(trace_path)
    if not trace.exists():
        raise FileNotFoundError(f"Trace path does not exist: {trace}")
    if not trace.is_dir():
        raise ValueError(f"Trace path is not a .gputrace directory: {trace}")

    symbols, files = _extract_trace_symbols(trace, max_scan_bytes=max_scan_bytes)
    regexes = [re.compile(pattern) for pattern in allowed_kernel_regexes]
    allowed_hits = [symbol for symbol in symbols if any(pattern.search(symbol) for pattern in regexes)]
    gather_nax_symbols = [symbol for symbol in symbols if "gather_qmm" in symbol and "nax" in symbol]
    return {
        "trace_path": str(trace),
        "metadata": _trace_metadata(trace),
        "scanned_files": files,
        "trace_nax_symbol_count": len(symbols),
        "trace_nax_symbol_samples": symbols[:max_symbols],
        "allowed_kernel_regexes": list(allowed_kernel_regexes),
        "allowed_kernel_hits": allowed_hits[:max_symbols],
        "has_allowed_gather_qmm_nax_bk64_symbol": bool(allowed_hits),
        "has_gather_qmm_nax_bk32_symbol": any(_BK32_RE.search(symbol) for symbol in gather_nax_symbols),
        "has_gather_qmm_nax_bk64_symbol": any(_BK64_RE.search(symbol) for symbol in gather_nax_symbols),
        "capture_summary_note": (
            "This summary proves the trace resources contain matching kernel symbols. "
            "The Phase N0 gate still requires Xcode inspection, or an equivalent command-level "
            "parser, to prove which kernel was actually encoded."
        ),
    }


def _trace_symbol_summary(
    *,
    trace_path: Path,
    symbols: list[str],
    allowed_kernel_regexes: Iterable[str],
    max_symbols: int,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regexes = [re.compile(pattern) for pattern in allowed_kernel_regexes]
    allowed_hits = [symbol for symbol in symbols if any(pattern.search(symbol) for pattern in regexes)]
    gather_nax_symbols = [symbol for symbol in symbols if "gather_qmm" in symbol and "nax" in symbol]
    record = {
        "trace_path": str(trace_path),
        "trace_summary_source": source,
        "trace_nax_symbol_count": len(symbols),
        "trace_nax_symbol_samples": symbols[:max_symbols],
        "allowed_kernel_regexes": list(allowed_kernel_regexes),
        "allowed_kernel_hits": allowed_hits[:max_symbols],
        "has_allowed_gather_qmm_nax_bk64_symbol": bool(allowed_hits),
        "has_gather_qmm_nax_bk32_symbol": any(_BK32_RE.search(symbol) for symbol in gather_nax_symbols),
        "has_gather_qmm_nax_bk64_symbol": any(_BK64_RE.search(symbol) for symbol in gather_nax_symbols),
    }
    if extra:
        record.update(extra)
    return record


def summarize_xctrace(
    trace_path: str | Path,
    *,
    allowed_kernel_regexes: Iterable[str] = (),
    max_symbols: int = 64,
    xpath: str = '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-shader-list"]',
) -> dict[str, Any]:
    trace = Path(trace_path)
    if not trace.exists():
        raise FileNotFoundError(f"Trace path does not exist: {trace}")
    if not trace.is_dir() or trace.suffix != ".trace":
        raise ValueError(f"Trace path is not an xctrace .trace directory: {trace}")

    completed = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xpath],
        check=False,
        text=True,
        capture_output=True,
    )
    output = completed.stdout or completed.stderr
    symbols = sorted(set(_TRACE_KERNEL_TOKEN_RE.findall(output)))
    return _trace_symbol_summary(
        trace_path=trace,
        symbols=symbols,
        allowed_kernel_regexes=allowed_kernel_regexes,
        max_symbols=max_symbols,
        source="xctrace_export_shader_list",
        extra={
            "xctrace_export_returncode": completed.returncode,
            "xctrace_export_xpath": xpath,
            "xctrace_export_summary": " ".join(output.strip().splitlines()[:3])[:400],
            "capture_summary_note": (
                "This summary is exported from the Metal System Trace shader list and can prove "
                "which compiled shader name appeared in the command-line xctrace capture."
            ),
        },
    )


def _q2_runtime_inputs(case: Q2NaxRuntimeCase) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    if case.input_dims % case.group_size != 0:
        raise ValueError(f"input_dims={case.input_dims} must be divisible by group_size={case.group_size}")
    if 32 % case.bits != 0:
        raise ValueError(f"bits={case.bits} must divide 32")

    rng = np.random.default_rng(case.seed)
    packed_cols = case.input_dims // (32 // case.bits)
    scale_cols = case.input_dims // case.group_size
    x = mx.array(rng.normal(size=(case.tokens, case.input_dims)).astype(np.float32)).astype(mx.bfloat16)
    indices = mx.array(rng.integers(0, case.experts, size=(case.tokens, case.top_k), dtype=np.int32))
    qweight = mx.array(
        rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=(case.experts, case.output_dims, packed_cols),
            dtype=np.uint32,
        )
    )
    scales = mx.array(
        rng.uniform(0.001, 0.05, size=(case.experts, case.output_dims, scale_cols)).astype(np.float32)
    )
    biases = mx.array(
        rng.uniform(-0.5, 0.5, size=(case.experts, case.output_dims, scale_cols)).astype(np.float32)
    )
    mx.eval(x, indices, qweight, scales, biases)
    return x, indices, qweight, scales, biases


def run_sorted_q2_case(
    case: Q2NaxRuntimeCase,
    *,
    capture_path: str | Path | None = None,
) -> dict[str, Any]:
    x, indices, qweight, scales, biases = _q2_runtime_inputs(case)
    expanded = mx.expand_dims(x, (-2, -3))
    sorted_x, sorted_indices, _ = _gather_sort(expanded, indices)

    trace_path = Path(capture_path) if capture_path is not None else None
    capture_started = False
    try:
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            mx.metal.start_capture(str(trace_path))
            capture_started = True
        output = mx.gather_qmm(
            sorted_x,
            qweight,
            scales,
            biases,
            rhs_indices=sorted_indices,
            transpose=True,
            group_size=case.group_size,
            bits=case.bits,
            mode=case.mode,
            sorted_indices=True,
        )
        mx.eval(output)
    finally:
        if capture_started:
            mx.metal.stop_capture()

    output_float = output.astype(mx.float32)
    record = case.to_record()
    record.update(
        {
            "runtime_call": "mx.gather_qmm(sorted_indices=True)",
            "packed_weight_shape": list(qweight.shape),
            "scales_shape": list(scales.shape),
            "biases_shape": list(biases.shape),
            "sorted_indices": True,
            "uses_rhs_indices": True,
            "uses_lhs_indices": False,
            "output_shape": list(output.shape),
            "finite_output": bool(mx.all(mx.isfinite(output_float)).item()),
            "checksum": float(mx.sum(output_float).item()),
            "capture_path": str(trace_path) if trace_path is not None else None,
            "capture_started": capture_started,
        }
    )
    if trace_path is not None and trace_path.exists():
        record["trace_summary"] = summarize_gputrace(
            trace_path,
            allowed_kernel_regexes=case.allowed_kernel_regexes,
        )
    return record
