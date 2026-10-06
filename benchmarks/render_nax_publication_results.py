from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - report generation should not fail outside git.
        return "unknown"


def _format_seconds(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _format_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _resident_table(rows: list[dict[str, Any]]) -> str:
    raw_summaries = [row for row in rows if row.get("record_type") == "publication_summary"]
    if not raw_summaries:
        return "_No resident publication summary rows found yet._"
    latest_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    latest_clean_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_summaries:
        key = (str(row.get("engine_name")), str(row.get("scenario")))
        latest_by_cell[key] = row
        if row.get("accepted_clean_rows"):
            latest_clean_by_cell[key] = row
    summaries = [
        latest_clean_by_cell.get(key, row)
        for key, row in latest_by_cell.items()
    ]
    baselines = {
        str(row["scenario"]): row.get("timing_median_seconds")
        for row in summaries
        if row.get("engine_name") == "mlx_q2_routed_g128"
    }
    lines = [
        "| Engine | Scenario | Valid | Median s | Min s | P90 s | Stddev s | x q2 | Invalid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda item: (str(item.get("scenario")), str(item.get("engine_name")))):
        baseline = baselines.get(str(row.get("scenario")))
        median = row.get("timing_median_seconds")
        ratio = None if baseline in (None, 0) or median is None else float(median) / float(baseline)
        reasons = row.get("invalid_repetition_reasons") or []
        reason_text = "clean" if not reasons else "; ".join(
            f"run {reason.get('run_index')}: p{reason.get('pageouts_delta')}/s{reason.get('swapouts_delta')}"
            for reason in reasons
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("engine_name", "")),
                    str(row.get("scenario", "")),
                    f"{row.get('valid_repetition_count', 0)}/{row.get('repetition_count', 0)}",
                    _format_seconds(median),
                    _format_seconds(row.get("timing_min_seconds")),
                    _format_seconds(row.get("timing_p90_seconds")),
                    _format_seconds(row.get("timing_stddev_seconds")),
                    _format_float(ratio, digits=2),
                    reason_text,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _projection_table(rows: list[dict[str, Any]]) -> str:
    projections = [row for row in rows if row.get("record_type") == "publication_projection"]
    if not projections:
        return "_No projection publication rows found yet._"
    lines = [
        "| Projection | M | Variant | ms/iter | Pageouts | Swapouts |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(projections, key=lambda item: (str(item.get("projection")), int(item.get("tokens", 0)), str(item.get("variant")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("projection", "")),
                    str(row.get("tokens", "")),
                    str(row.get("variant", "")),
                    _format_float(row.get("ms_per_iter"), digits=3),
                    str(row.get("pageouts_delta", "")),
                    str(row.get("swapouts_delta", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _resident_rejection_notes(rows: list[dict[str, Any]]) -> str:
    rejected = [
        row for row in rows
        if row.get("record_type") == "publication_summary" and not row.get("accepted_clean_rows", False)
    ]
    if not rejected:
        return "_No rejected resident summary cells recorded._"
    lines = []
    for row in rejected:
        reasons = row.get("invalid_repetition_reasons") or []
        reason_text = "; ".join(
            f"run {reason.get('run_index')}: pageouts {reason.get('pageouts_delta')}, swapouts {reason.get('swapouts_delta')}"
            for reason in reasons
        )
        lines.append(
            f"- `{row.get('engine_name')}` / `{row.get('scenario')}` rejected: "
            f"{row.get('valid_repetition_count')}/{row.get('repetition_count')} clean reps. {reason_text}"
        )
    return "\n".join(lines)


def _int8_section(rows: list[dict[str, Any]]) -> str:
    probes = [row for row in rows if row.get("benchmark") == "nax_int8_raw_ceiling"]
    if not probes:
        return "_No INT8 raw-ceiling row found yet._"
    row = probes[-1]
    if not row.get("available"):
        return f"INT8 raw-ceiling probe unavailable: {row.get('reason', 'unknown')}."
    return "\n".join(
        [
            "| Tile | FP16 ms | INT8 prequant ms | INT8 full ms | Speedup | Gate | Cosine |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
            "| "
            + " | ".join(
                [
                    str(row.get("tile_geometry", "")),
                    _format_float(row.get("fp16_ms_per_iter"), digits=4),
                    _format_float(row.get("int8_prequantized_ms_per_iter"), digits=4),
                    _format_float(row.get("int8_with_quant_ms_per_iter"), digits=4),
                    _format_float(row.get("int8_prequantized_speedup_vs_fp16"), digits=2),
                    str(row.get("raw_ceiling_gate", "")),
                    _format_float(row.get("cosine_vs_fp16"), digits=5),
                ]
            )
            + " |",
        ]
    )


def _toolchain_section(rows: list[dict[str, Any]]) -> str:
    audits = [row for row in rows if row.get("record_type") == "toolchain_audit"]
    if not audits:
        return "_No toolchain audit JSONL row found yet._"
    row = audits[-1]
    lines = [
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in row.get("checks", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(check.get("name", "")),
                    "ok" if check.get("returncode") == 0 else "failed",
                    str(check.get("summary", "")).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _xctrace_case_label(row: dict[str, Any]) -> str:
    trace_path = str(row.get("trace_path", ""))
    match = re.search(r"xctrace-current-(gate_up|down)-([0-9]+)\.trace", trace_path)
    if match:
        return f"{match.group(1)} M={match.group(2)}"
    return trace_path or "unknown"


def _q2_xctrace_section(rows: list[dict[str, Any]]) -> str:
    summaries = [row for row in rows if row.get("trace_summary_source") == "xctrace_export_shader_list"]
    if not summaries:
        return "_No q2 xtrace shader-list summary rows found yet._"
    latest_by_case: dict[str, dict[str, Any]] = {}
    for row in summaries:
        latest_by_case[_xctrace_case_label(row)] = row
    lines = [
        "| Case | Export | bk64 RHS NAX | bk32 seen | Kernel |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, row in sorted(latest_by_case.items()):
        hits = row.get("allowed_kernel_hits") or row.get("trace_nax_symbol_samples") or []
        kernel = str(hits[0]) if hits else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    "ok" if row.get("xctrace_export_returncode") == 0 else f"failed {row.get('xctrace_export_returncode')}",
                    "yes" if row.get("has_allowed_gather_qmm_nax_bk64_symbol") else "no",
                    "yes" if row.get("has_gather_qmm_nax_bk32_symbol") else "no",
                    kernel,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_report(
    *,
    quant_jsonl: Path,
    projection_jsonl: Path,
    int8_jsonl: Path,
    toolchain_jsonl: Path,
    xctrace_jsonl: Path,
    output: Path,
    commit: str,
) -> None:
    quant_rows = _read_jsonl(quant_jsonl)
    projection_rows = _read_jsonl(projection_jsonl)
    int8_rows = _read_jsonl(int8_jsonl)
    toolchain_rows = _read_jsonl(toolchain_jsonl)
    xctrace_rows = _read_jsonl(xctrace_jsonl)
    content = f"""# NAX Parity Publication Results

Generated from JSONL artifacts, not terminal output.

## Methodology

- Commit: `{commit}`
- Resident artifact: `{quant_jsonl}`
- Projection artifact: `{projection_jsonl}`
- INT8 ceiling artifact: `{int8_jsonl}`
- Toolchain audit artifact: `{toolchain_jsonl}`
- q2 xtrace shader-list artifact: `{xctrace_jsonl}`
- Headline resident rows require fresh-process repetitions and accept only `pageouts_delta == 0` and `swapouts_delta == 0`.
- Baseline is the audited sorted `mlx_q2_routed_g128` path. LLDB and xtrace shader-list export identify `affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2` / aligned suffix variants and no `bk32` request on the local MLX 0.31.2 stack.
- Full Xcode GPU counters are optional; the reproducible fallback is the LLDB kernel-name tap plus zero pageout/swapout resident rows.

## Toolchain Audit

{_toolchain_section(toolchain_rows)}

## q2 Xtrace Kernel Audit

{_q2_xctrace_section(xctrace_rows)}

## Resident Matrix

The table uses the latest fully accepted clean summary row for each `(engine,
scenario)` pair when a JSONL contains targeted reruns after the initial full
matrix; if no clean summary exists for a cell, it falls back to that cell's
latest attempted summary.

{_resident_table(quant_rows)}

## Rejected Resident Rows

{_resident_rejection_notes(quant_rows)}

## Projection Matrix

{_projection_table(projection_rows)}

## INT8 Raw Ceiling

{_int8_section(int8_rows)}

## Reference Notes

- Apple setup references: [installing command-line tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/), [Metal developer tools](https://developer.apple.com/metal/tools/), [command-line Metal compilation](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Dev-Technique/Dev-Technique.html), and [Metal dynamic-library compilation](https://developer.apple.com/documentation/metal/compiling-and-linking-metal-dynamic-libraries).
- Apple ML Research reference: [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- Cider reference: [M5 INT8 TensorOps tutorial](https://github.com/Mininglamp-AI/cider/blob/main/tutorial/how_to_write_efficient_int_gemm_m5_en.md), which describes `matmul2d(16,32,16)` INT8 TensorOps and the C++ primitive requirement for cooperative tensor kernels.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the NAX parity publication report from JSONL artifacts.")
    parser.add_argument("--quant-jsonl", default="artifacts/benchmarks/glm45-air-nax-publication.jsonl")
    parser.add_argument(
        "--projection-jsonl",
        default="artifacts/benchmarks/glm45-air-nax-publication-projections.jsonl",
    )
    parser.add_argument("--int8-jsonl", default="artifacts/benchmarks/glm45-air-nax-int8-raw-ceiling.jsonl")
    parser.add_argument("--toolchain-jsonl", default="artifacts/benchmarks/glm45-air-nax-toolchain-audit.jsonl")
    parser.add_argument("--xctrace-jsonl", default="artifacts/benchmarks/glm45-air-nax-q2-xctrace-summary-full.jsonl")
    parser.add_argument("--output", default="docs/research/NAX_PARITY_RESULTS.md")
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()

    render_report(
        quant_jsonl=Path(args.quant_jsonl),
        projection_jsonl=Path(args.projection_jsonl),
        int8_jsonl=Path(args.int8_jsonl),
        toolchain_jsonl=Path(args.toolchain_jsonl),
        xctrace_jsonl=Path(args.xctrace_jsonl),
        output=Path(args.output),
        commit=args.commit or _git_commit(),
    )


if __name__ == "__main__":
    main()
