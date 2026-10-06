from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.convert.stream_convert import load_safetensors_index

DEFAULT_VQ_ARTIFACT_DIR = "artifacts/glm-4.5-air-vq"
DEFAULT_Q2_ARTIFACT_DIR = "artifacts/glm-4.5-air-mlx-q2-routed-g128"
DEFAULT_Q8_SEARCH_ROOTS = (
    "~/.cache/huggingface/hub",
    "/tmp",
    "artifacts",
)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["model.safetensors.index.json"],
            local_files_only=True,
        )
    )


def _device_info() -> dict[str, Any]:
    try:
        return dict(mx.device_info())
    except Exception:
        return {}


def _max_working_set_bytes(device_info: dict[str, Any]) -> int | None:
    candidates = (
        "max_recommended_working_set_size",
        "max_recommended_working_set",
        "recommendedMaxWorkingSetSize",
    )
    for key in candidates:
        value = device_info.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _artifact_bytes(path: str | Path) -> int | None:
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return root.stat().st_size
    total = 0
    for child in root.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _looks_like_q8_air_path(path: Path) -> bool:
    name = str(path).lower().replace("_", "-")
    has_model_marker = "glm" in name and "air" in name
    has_q8_marker = any(marker in name for marker in ("q8", "8bit", "8-bit", "int8"))
    has_lower_bit_marker = any(
        marker in name
        for marker in (
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "2bit",
            "3bit",
            "4bit",
            "5bit",
            "6bit",
            "7bit",
            "2-bit",
            "3-bit",
            "4-bit",
            "5-bit",
            "6-bit",
            "7-bit",
        )
    )
    return has_model_marker and has_q8_marker and not has_lower_bit_marker


def _q8_candidate_file_summary(path: Path) -> tuple[bool, int, int]:
    if path.is_file():
        return (
            path.name == "model.safetensors.index.json",
            int(path.suffix == ".safetensors"),
            path.stat().st_size if path.suffix == ".safetensors" else 0,
        )

    has_index = (path / "model.safetensors.index.json").exists()
    safetensor_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        if child.suffix == ".safetensors":
            safetensor_count += 1
            total_bytes += child.stat().st_size
        elif child.name == "model.safetensors.index.json":
            has_index = True
    return has_index, safetensor_count, total_bytes


def discover_q8_air_candidates(search_roots: list[str | Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_root in search_roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        paths = [root] if _looks_like_q8_air_path(root) else []
        for marker in ("*q8*", "*Q8*", "*8bit*", "*8Bit*", "*8-bit*", "*int8*", "*INT8*"):
            paths.extend(root.rglob(marker))
        for path in paths:
            if not _looks_like_q8_air_path(path):
                continue
            candidate_root = path if path.is_dir() else path.parent
            try:
                resolved = candidate_root.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            has_index, safetensor_count, total_bytes = _q8_candidate_file_summary(candidate_root)
            if not has_index and safetensor_count == 0:
                continue
            seen.add(resolved)
            candidates.append(
                {
                    "path": str(candidate_root),
                    "has_index": has_index,
                    "safetensor_file_count": safetensor_count,
                    "total_safetensor_bytes": total_bytes,
                    "decision": "candidate_requires_quality_provenance_review",
                }
            )
    return sorted(candidates, key=lambda candidate: candidate["path"])


def _source_shard_bytes(source_dir: Path, index_path: Path) -> tuple[int, int, list[str]]:
    index = load_safetensors_index(index_path)
    shard_names = sorted(set(index.weight_map.values()))
    missing = []
    total = 0
    for shard_name in shard_names:
        path = source_dir / shard_name
        if not path.exists():
            missing.append(shard_name)
            continue
        total += path.stat().st_size
    return total, len(shard_names), missing


def build_feasibility_record(
    *,
    model_id: str,
    revision: str,
    source_dir: str | Path,
    index_path: str | Path,
    vq_artifact_dir: str | Path,
    q2_artifact_dir: str | Path,
    host_budget_bytes: int,
    q8_search_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    source_root = Path(source_dir)
    source_bytes, source_shards, missing_shards = _source_shard_bytes(source_root, Path(index_path))
    device = _device_info()
    max_working_set = _max_working_set_bytes(device)
    vq_bytes = _artifact_bytes(vq_artifact_dir)
    q2_bytes = _artifact_bytes(q2_artifact_dir)
    q8_weight_estimate_bytes = int(source_bytes * 0.5)
    q8_runtime_headroom_bytes = (
        host_budget_bytes - q8_weight_estimate_bytes
        if host_budget_bytes is not None
        else None
    )
    q8_device_headroom_bytes = (
        max_working_set - q8_weight_estimate_bytes
        if max_working_set is not None
        else None
    )

    bf16_fits_host_budget = source_bytes <= host_budget_bytes
    bf16_fits_device_working_set = (
        max_working_set is not None
        and source_bytes <= max_working_set
    )
    q8_estimate_fits_host_budget = q8_weight_estimate_bytes <= host_budget_bytes
    q8_estimate_fits_device_working_set = (
        max_working_set is not None
        and q8_weight_estimate_bytes <= max_working_set
    )
    q8_candidates = discover_q8_air_candidates(
        list(DEFAULT_Q8_SEARCH_ROOTS) if q8_search_roots is None else q8_search_roots
    )
    q8_artifact_present = bool(q8_candidates)

    return {
        "model_id": model_id,
        "revision": revision,
        "source_dir": str(source_root),
        "index_path": str(index_path),
        "host_budget_bytes": host_budget_bytes,
        "device_info": device,
        "max_working_set_bytes": max_working_set,
        "source_shards": source_shards,
        "source_missing_shards": missing_shards,
        "source_shard_bytes": source_bytes,
        "vq_artifact_dir": str(vq_artifact_dir),
        "vq_artifact_bytes": vq_bytes,
        "q2_artifact_dir": str(q2_artifact_dir),
        "q2_artifact_bytes": q2_bytes,
        "teacher_candidates": {
            "bf16_source": {
                "available": len(missing_shards) == 0,
                "weight_bytes": source_bytes,
                "fits_host_budget": bf16_fits_host_budget,
                "fits_device_working_set": bf16_fits_device_working_set,
                "decision": "reject_local" if not bf16_fits_host_budget else "maybe_local",
            },
            "q8_full_estimate": {
                "available": q8_artifact_present,
                "candidate_count": len(q8_candidates),
                "candidates": q8_candidates,
                "estimated_weight_bytes": q8_weight_estimate_bytes,
                "estimated_host_budget_headroom_bytes": q8_runtime_headroom_bytes,
                "estimated_device_working_set_headroom_bytes": q8_device_headroom_bytes,
                "fits_host_budget_estimate": q8_estimate_fits_host_budget,
                "fits_device_working_set_estimate": q8_estimate_fits_device_working_set,
                "decision": (
                    "candidate_found_needs_quality_provenance_review"
                    if q8_artifact_present
                    else "not_available_locally_marginal_even_if_converted"
                ),
                "notes": "Estimate excludes temporary conversion buffers, KV cache, MLX cache, tokenizer/model overhead, and scale metadata.",
            },
            "mlx_q2_routed_g128": {
                "available": q2_bytes is not None,
                "artifact_bytes": q2_bytes,
                "decision": "local_reference_not_high_bit_teacher",
            },
        },
        "recommended_authority": "offbox_bf16_or_q8_teacher_logit_cache",
        "local_item1_verdict_status": "blocked_on_teacher_authority",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether GLM-4.5-Air high-bit teacher eval fits locally.")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--vq-artifact-dir", default=DEFAULT_VQ_ARTIFACT_DIR)
    parser.add_argument("--q2-artifact-dir", default=DEFAULT_Q2_ARTIFACT_DIR)
    parser.add_argument(
        "--q8-search-root",
        action="append",
        default=None,
        help="Root to search for trusted Q8/8bit Air teacher candidates. Repeatable.",
    )
    parser.add_argument("--host-budget-gb", type=float, default=128.0)
    parser.add_argument("--append-jsonl", default="artifacts/quality/glm45-air-teacher-feasibility.jsonl")
    args = parser.parse_args()

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path is not None else source_dir / "model.safetensors.index.json"
    record = build_feasibility_record(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=source_dir,
        index_path=index_path,
        vq_artifact_dir=args.vq_artifact_dir,
        q2_artifact_dir=args.q2_artifact_dir,
        q8_search_roots=args.q8_search_root,
        host_budget_bytes=int(args.host_budget_gb * 1024**3),
    )
    if args.append_jsonl:
        append_jsonl(args.append_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
