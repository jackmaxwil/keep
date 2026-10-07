"""Real layer-0 DeepSeek-V4-Flash M=1 expert-projection gate.

Each ``run`` invocation is one fresh-process paired row. It loads the released
FP4 tensors through the existing native ``mxfp4`` loader and the real E8P
artifact through the production VQ loader, then interleaves both paths. The
summary is deliberately component-scoped; it is not a full-model decode claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from benchmarks.bench_dsv4_flash_layer_slice_prefill import (
    ShardReader,
    load_layer_experts,
)
from ramp.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from keep.io.load import load_quantized_vq_switch_linear
from ramp.ops import vq_switch
from keep.quality.dsv4_teacher_agreement import heavy_job_lock, vq_artifact_identity

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
DEFAULT_CHECKPOINT = Path.home() / "models/DeepSeek-V4-Flash-0731"
DEFAULT_ARTIFACT = Path.home() / "keep-artifacts/dsv4-vq-e8p-g512"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
KERNEL_IDENTITY = "nax_e8p_fp16_sorted_steel_m32n64"
SOURCE_BACKEND = "mx.gather_qmm:mxfp4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "q1": float(np.quantile(ordered, 0.25)),
        "q3": float(np.quantile(ordered, 0.75)),
        "max": ordered[-1],
    }


def summarize_m1_rows(
    rows: Sequence[Mapping[str, Any]], *, min_clean_rows: int = 7
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no M=1 rows")
    for row in rows:
        if row.get("record_type") != "dsv4_m1_expert_projection_row_v1":
            raise ValueError("unexpected M=1 row type")
        if row.get("fresh_process") is not True:
            raise ValueError("every M=1 row must be fresh-process evidence")
        if row.get("kernel_identity") != KERNEL_IDENTITY:
            raise ValueError("M=1 row did not use the production E8P kernel")
        if int(row.get("kernel_calls", 0)) <= 0:
            raise ValueError("M=1 row did not call the production E8P kernel")
        if row.get("source_backend") != SOURCE_BACKEND:
            raise ValueError("M=1 row did not use the native FP4 source backend")
        if set(row.get("projections", {})) != set(PROJECTIONS):
            raise ValueError("M=1 row must contain gate, up, and down projections")

    pids = [row.get("pid") for row in rows]
    if any(not isinstance(pid, int) for pid in pids) or len(set(pids)) != len(pids):
        raise ValueError("every M=1 row must have a distinct fresh-process PID")

    identity_encodings = {
        json.dumps(row["identities"], sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    if len(identity_encodings) != 1:
        raise ValueError("identity drift across M=1 rows")

    clean = [
        row
        for row in rows
        if row.get("pageouts_delta") == 0 and row.get("swapouts_delta") == 0
    ]
    if len(clean) < min_clean_rows:
        raise ValueError(
            f"need at least {min_clean_rows} clean M=1 rows, found {len(clean)}"
        )
    orders = Counter(str(row.get("order")) for row in clean)
    if not orders["vq_first"] or not orders["source_first"]:
        raise ValueError("clean M=1 rows must include both interleaving orders")

    projection_summary: dict[str, Any] = {}
    total_ratios: list[float] = []
    total_vq_ms: list[float] = []
    total_source_ms: list[float] = []
    for projection in PROJECTIONS:
        vq = [float(row["projections"][projection]["vq_median_ms"]) for row in clean]
        source = [
            float(row["projections"][projection]["source_median_ms"]) for row in clean
        ]
        ratios = [source_ms / vq_ms for source_ms, vq_ms in zip(source, vq)]
        projection_summary[projection] = {
            "vq_ms": _distribution(vq),
            "source_ms": _distribution(source),
            "paired_source_over_vq": _distribution(ratios),
            "median_paired_source_over_vq": statistics.median(ratios),
        }

    for row in clean:
        vq_ms = sum(
            float(row["projections"][projection]["vq_median_ms"])
            for projection in PROJECTIONS
        )
        source_ms = sum(
            float(row["projections"][projection]["source_median_ms"])
            for projection in PROJECTIONS
        )
        total_vq_ms.append(vq_ms)
        total_source_ms.append(source_ms)
        total_ratios.append(source_ms / vq_ms)

    total_ratio = statistics.median(total_ratios)
    return {
        "record_type": "dsv4_m1_expert_projection_summary_v1",
        "scope": "component_level_m1_expert_projection_only",
        "comparison_formula": "source_mxfp4_projection_ms / production_vq_e8p_projection_ms",
        "rows": len(rows),
        "clean_rows": len(clean),
        "dirty_rows": len(rows) - len(clean),
        "orders": dict(sorted(orders.items())),
        "identities": dict(clean[0]["identities"]),
        "kernel_identity": KERNEL_IDENTITY,
        "source_backend": SOURCE_BACKEND,
        "projections": projection_summary,
        "projection_total": {
            "vq_ms": _distribution(total_vq_ms),
            "source_ms": _distribution(total_source_ms),
            "paired_source_over_vq": _distribution(total_ratios),
            "median_paired_source_over_vq": total_ratio,
        },
        "faster_clean_rows": sum(ratio > 1.0 for ratio in total_ratios),
        "gate": "PASS" if total_ratio > 1.0 else "STOP",
        "claim_limit": (
            "Layer-0 real-artifact M=1 expert projections only; residents do not "
            "exist yet, so this is not full-model or end-to-end decode evidence."
        ),
    }


def _weight_map(checkpoint: Path) -> dict[str, str]:
    payload = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def _source_shard_identities(
    checkpoint: Path, weight_map: Mapping[str, str]
) -> dict[str, Any]:
    names = [
        f"layers.0.ffn.experts.{expert}.{projection}.{suffix}"
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    ]
    shards = sorted({weight_map[name] for name in names})
    metadata_root = checkpoint / ".cache" / "huggingface" / "download"
    identities: dict[str, Any] = {}
    for shard in shards:
        lines = (metadata_root / f"{shard}.metadata").read_text().splitlines()
        identities[shard] = {
            "revision": lines[0],
            "etag": lines[1],
            "bytes": (checkpoint / shard).stat().st_size,
        }
    return identities


def _load_vq_projections(artifact: Path) -> dict[str, Any]:
    return {
        projection: load_quantized_vq_switch_linear(
            artifact / f"layer-{0:05d}-{projection}.safetensors",
            f"model.layers.0.ffn.switch_mlp.{projection}",
        )
        for projection in PROJECTIONS
    }


def _timed_ms(fn: Callable[[], mx.array], inner_loops: int) -> tuple[float, float]:
    started = time.perf_counter()
    checksum = 0.0
    for _ in range(inner_loops):
        output = fn()
        mx.eval(output)
        checksum = float(mx.sum(output.astype(mx.float32)).item())
    return (time.perf_counter() - started) * 1_000.0 / inner_loops, checksum


def _run_row(args: argparse.Namespace) -> dict[str, Any]:
    forbidden = [
        name
        for name in ("GLM_MLX_WIRED_LIMIT_GB", "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB")
        if name in os.environ
    ]
    if forbidden:
        raise RuntimeError(f"custom wired-limit variables must stay unset: {forbidden}")
    if not vq_switch.nax.is_available():
        raise RuntimeError("native VQ NAX extension is unavailable")

    checkpoint = args.checkpoint.resolve()
    artifact = args.artifact_dir.resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    expected = {
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
    }
    for key, value in expected.items():
        if int(config[key]) != value:
            raise ValueError(f"{key} must be {value}, found {config[key]}")

    weight_map = _weight_map(checkpoint)
    vq_layers = _load_vq_projections(artifact)
    reader = ShardReader(checkpoint, weight_map)
    try:
        source_glu, source_load = load_layer_experts(
            reader,
            0,
            expected["n_routed_experts"],
            float(config["swiglu_limit"]),
            mode="mxfp4",
        )
    finally:
        reader.close()
    source_layers = {
        projection: getattr(source_glu, projection) for projection in PROJECTIONS
    }

    for projection, layer in vq_layers.items():
        if (
            layer.code_bits != 16
            or layer.group_size != 512
            or layer.num_experts != 256
            or layer.route_strategy != "auto"
        ):
            raise ValueError(f"{projection} is not the production E8P auto artifact")

    rng = np.random.default_rng(args.seed)
    indices = mx.array(rng.integers(0, 256, size=(1, 1, 6), dtype=np.int32))
    inputs = {
        projection: mx.array(
            rng.normal(size=(1, 1, layer.input_dims)).astype(np.float32)
        ).astype(mx.bfloat16)
        for projection, layer in vq_layers.items()
    }
    mx.eval(
        indices,
        list(inputs.values()),
        [
            value
            for layer in vq_layers.values()
            for value in layer.parameters().values()
        ],
        source_glu.parameters(),
    )

    identities = vq_artifact_identity(artifact)
    download = json.loads((checkpoint / "_KEEP_DOWNLOAD_COMPLETE.json").read_text())
    row_identities = {
        "checkpoint": str(checkpoint),
        "checkpoint_revision": download["revision"],
        "checkpoint_config_sha256": _sha256(checkpoint / "config.json"),
        "checkpoint_index_sha256": _sha256(checkpoint / "model.safetensors.index.json"),
        "source_layer0_shards": _source_shard_identities(checkpoint, weight_map),
        "artifact": str(artifact),
        "artifact_manifest_sha256": identities["manifest_sha256"],
        "artifact_file_inventory_sha256": identities["file_inventory_sha256"],
        "artifact_codebook_sha256": (identities.get("codebook") or {}).get("sha256"),
    }

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    real_kernel = vq_switch.nax.nax_e8p_fp16_sorted_steel_m32n64_matmul
    kernel_calls = 0

    def observed_kernel(*call_args: Any, **call_kwargs: Any) -> mx.array:
        nonlocal kernel_calls
        kernel_calls += 1
        return real_kernel(*call_args, **call_kwargs)

    vq_switch.nax.nax_e8p_fp16_sorted_steel_m32n64_matmul = observed_kernel
    projection_rows: dict[str, Any] = {}
    try:
        for projection in PROJECTIONS:
            vq_layer = vq_layers[projection]
            source_layer = source_layers[projection]
            x = inputs[projection]
            vq_fn = lambda layer=vq_layer, value=x: layer(value, indices)
            source_fn = lambda layer=source_layer, value=x: layer(value, indices)
            for _ in range(args.warmup):
                mx.eval(vq_fn(), source_fn())

            vq_samples: list[float] = []
            source_samples: list[float] = []
            checksums = {"vq": 0.0, "source": 0.0}
            for sample in range(args.iterations):
                first = args.order == "vq_first"
                if sample % 2:
                    first = not first
                calls = (("vq", vq_fn), ("source", source_fn))
                if not first:
                    calls = tuple(reversed(calls))
                for label, fn in calls:
                    elapsed, checksum = _timed_ms(fn, args.inner_loops)
                    (vq_samples if label == "vq" else source_samples).append(elapsed)
                    checksums[label] = checksum
            projection_rows[projection] = {
                "input_dims": int(vq_layer.input_dims),
                "output_dims": int(vq_layer.output_dims),
                "experts": int(vq_layer.num_experts),
                "top_k": 6,
                "tokens": 1,
                "vq_median_ms": statistics.median(vq_samples),
                "source_median_ms": statistics.median(source_samples),
                "paired_source_over_vq": statistics.median(
                    source_ms / vq_ms
                    for source_ms, vq_ms in zip(source_samples, vq_samples)
                ),
                "vq_samples_ms": vq_samples,
                "source_samples_ms": source_samples,
                "checksums": checksums,
            }
    finally:
        vq_switch.nax.nax_e8p_fp16_sorted_steel_m32n64_matmul = real_kernel

    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    row = {
        "record_type": "dsv4_m1_expert_projection_row_v1",
        "schema_version": 1,
        "scope": "component_level_m1_expert_projection_only",
        "fresh_process": True,
        "pid": os.getpid(),
        "run_index": args.run_index,
        "order": args.order,
        "seed": args.seed,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "inner_loops": args.inner_loops,
        "kernel_identity": KERNEL_IDENTITY,
        "kernel_calls": kernel_calls,
        "source_backend": SOURCE_BACKEND,
        "identities": row_identities,
        "source_load": source_load,
        "projections": projection_rows,
        "argv": sys.argv,
        **metrics,
    }
    args.append_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.append_jsonl.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return row


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()
    ]
    summary = summarize_m1_rows(rows, min_clean_rows=args.min_clean_rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    run.add_argument("--append-jsonl", type=Path, required=True)
    run.add_argument("--run-index", type=int, required=True)
    run.add_argument("--order", choices=("vq_first", "source_first"), required=True)
    run.add_argument("--seed", type=int, default=20260819)
    run.add_argument("--iterations", type=int, default=21)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--inner-loops", type=int, default=5)
    run.set_defaults(func=_run_row, heavy=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("--rows", type=Path, required=True)
    summary.add_argument("--min-clean-rows", type=int, default=7)
    summary.add_argument("--output-json", type=Path)
    summary.set_defaults(func=_summarize, heavy=False)

    args = parser.parse_args()
    holder = f"dsv4-m1-expert-projections:{args.command}:{os.getpid()}"
    with heavy_job_lock(HEAVY_JOB_LOCK, enabled=args.heavy, holder=holder):
        result = args.func(args)
    if args.command == "summarize" and result["gate"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
