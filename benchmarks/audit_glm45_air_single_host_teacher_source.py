from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

LAYER_RE = re.compile(r"^model[.]layers[.](?P<layer>[0-9]+)[.]")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sysctl_int(name: str) -> int | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", name],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    stripped = output.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _shard_record(source_dir: Path, shard: str) -> dict[str, Any]:
    path = source_dir / shard
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return {
            "shard": shard,
            "path": str(path),
            "exists": False,
            "bytes": 0,
        }
    size = int(resolved.stat().st_size)
    return {
        "shard": shard,
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": True,
        "bytes": size,
    }


def _memory_observation(
    *,
    expected_source_bytes: int | None,
    physical_memory_bytes: int | None,
    wired_limit_mb: int | None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "physical_memory_bytes": physical_memory_bytes,
        "iogpu_wired_limit_mb": wired_limit_mb,
        "expected_source_bytes": expected_source_bytes,
        "source_bytes_exceed_physical_memory": None,
    }
    if expected_source_bytes is not None and physical_memory_bytes is not None:
        observation["source_bytes_exceed_physical_memory"] = (
            expected_source_bytes > physical_memory_bytes
        )
        observation["expected_source_to_physical_memory_ratio"] = (
            expected_source_bytes / physical_memory_bytes
            if physical_memory_bytes > 0
            else None
        )
    return observation


def _tensor_layer(tensor_name: str) -> int | None:
    match = LAYER_RE.match(tensor_name)
    if match is None:
        return None
    return int(match.group("layer"))


def _num_hidden_layers(config: dict[str, Any]) -> int | None:
    value = config.get("num_hidden_layers")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _runtime_required_tensor(tensor_name: str, *, num_hidden_layers: int | None) -> bool:
    layer = _tensor_layer(tensor_name)
    if layer is None or num_hidden_layers is None:
        return True
    return layer < num_hidden_layers


def _candidate_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_path": record["model_path"],
        "decision": record["decision"],
        "source_ready_for_single_host_export": record["source_ready_for_single_host_export"],
        "complete_source": record.get("complete_source"),
        "complete_runtime_source": record.get("complete_runtime_source"),
        "expected_shard_count": record["expected_shard_count"],
        "present_shard_count": record["present_shard_count"],
        "missing_shard_count": record["missing_shard_count"],
        "missing_shards": record["missing_shards"],
        "missing_required_runtime_shards": record.get(
            "missing_required_runtime_shards", []
        ),
        "missing_extra_shards": record.get("missing_extra_shards", []),
        "present_shard_bytes": record["present_shard_bytes"],
        "metadata_total_size": record["metadata_total_size"],
        "torch_dtype": record["config"].get("torch_dtype"),
    }


def _scan_local_source_candidates(
    *,
    scan_roots: tuple[Path, ...],
    physical_memory_bytes: int | None,
    wired_limit_mb: int | None,
) -> dict[str, Any]:
    candidate_dirs: list[Path] = []
    seen: set[Path] = set()
    roots: list[str] = []
    for root in scan_roots:
        roots.append(str(root))
        if not root.exists():
            continue
        index_paths = [root / "model.safetensors.index.json"] if root.is_dir() else []
        if root.is_dir():
            index_paths.extend(root.rglob("model.safetensors.index.json"))
        for index_path in sorted(index_paths):
            if not index_path.is_file():
                continue
            source_dir = index_path.parent
            resolved = source_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidate_dirs.append(source_dir)

    candidate_records = [
        build_audit_record(
            model_path=source_dir,
            revision=source_dir.name,
            physical_memory_bytes=physical_memory_bytes,
            wired_limit_mb=wired_limit_mb,
        )
        for source_dir in candidate_dirs
    ]
    summaries = [_candidate_summary(record) for record in candidate_records]
    complete = [item for item in summaries if item["source_ready_for_single_host_export"]]
    return {
        "scan_roots": roots,
        "scanned_index_count": len(candidate_records),
        "complete_candidate_count": len(complete),
        "incomplete_candidate_count": len(summaries) - len(complete),
        "local_export_candidate_ready": bool(complete),
        "recommended_model_path": complete[0]["model_path"] if complete else None,
        "candidates": summaries,
    }


def build_audit_record(
    *,
    model_path: str | Path,
    revision: str | None = None,
    teacher_kind: str = "bf16_source",
    physical_memory_bytes: int | None = None,
    wired_limit_mb: int | None = None,
    scan_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    source_dir = Path(model_path)
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    record: dict[str, Any] = {
        "record_type": "glm45_air_single_host_teacher_source_audit",
        "model_path": str(model_path),
        "revision": revision,
        "teacher_kind": teacher_kind,
        "source_dir": str(source_dir),
        "index_path": str(index_path),
        "config_path": str(config_path),
        "source_dir_exists": source_dir.is_dir(),
        "index_exists": index_path.is_file(),
        "config_exists": config_path.is_file(),
        "peer2_used": False,
        "rdma_jaccl_touched": False,
        "cache_rows_generated": False,
    }

    config: dict[str, Any] = {}
    if config_path.is_file():
        config = _read_json(config_path)
    record["config"] = {
        key: config.get(key)
        for key in (
            "architectures",
            "model_type",
            "torch_dtype",
            "num_hidden_layers",
            "n_routed_experts",
            "hidden_size",
            "vocab_size",
        )
        if key in config
    }

    if not source_dir.is_dir():
        record.update(
            {
                "decision": "single_host_teacher_source_not_local",
                "source_ready_for_single_host_export": False,
                "weight_map_tensor_count": 0,
                "expected_shard_count": 0,
                "present_shard_count": 0,
                "missing_shard_count": 0,
                "missing_shards": [],
                "present_shard_bytes": 0,
                "metadata_total_size": None,
                "memory_observation": _memory_observation(
                    expected_source_bytes=None,
                    physical_memory_bytes=physical_memory_bytes,
                    wired_limit_mb=wired_limit_mb,
                ),
            }
        )
        if scan_roots:
            candidate_scan = _scan_local_source_candidates(
                scan_roots=scan_roots,
                physical_memory_bytes=physical_memory_bytes,
                wired_limit_mb=wired_limit_mb,
            )
            record["candidate_scan"] = candidate_scan
            record["local_export_candidate_ready"] = candidate_scan[
                "local_export_candidate_ready"
            ]
            record["recommended_model_path"] = candidate_scan["recommended_model_path"]
        return record

    if not index_path.is_file():
        record.update(
            {
                "decision": "single_host_teacher_source_index_missing",
                "source_ready_for_single_host_export": False,
                "weight_map_tensor_count": 0,
                "expected_shard_count": 0,
                "present_shard_count": 0,
                "missing_shard_count": 0,
                "missing_shards": [],
                "present_shard_bytes": 0,
                "metadata_total_size": None,
                "memory_observation": _memory_observation(
                    expected_source_bytes=None,
                    physical_memory_bytes=physical_memory_bytes,
                    wired_limit_mb=wired_limit_mb,
                ),
            }
        )
        if scan_roots:
            candidate_scan = _scan_local_source_candidates(
                scan_roots=scan_roots,
                physical_memory_bytes=physical_memory_bytes,
                wired_limit_mb=wired_limit_mb,
            )
            record["candidate_scan"] = candidate_scan
            record["local_export_candidate_ready"] = candidate_scan[
                "local_export_candidate_ready"
            ]
            record["recommended_model_path"] = candidate_scan["recommended_model_path"]
        return record

    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index must contain a weight_map object")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata_total_size = metadata.get("total_size")
    if isinstance(metadata_total_size, str) and metadata_total_size.isdigit():
        metadata_total_size = int(metadata_total_size)
    if not isinstance(metadata_total_size, int):
        metadata_total_size = None

    num_hidden_layers = _num_hidden_layers(config)
    runtime_required_shards = sorted(
        {
            str(shard)
            for tensor_name, shard in weight_map.items()
            if _runtime_required_tensor(
                str(tensor_name), num_hidden_layers=num_hidden_layers
            )
        }
    )
    shards = sorted({str(shard) for shard in weight_map.values()})
    shard_records = [_shard_record(source_dir, shard) for shard in shards]
    missing = [item["shard"] for item in shard_records if not item["exists"]]
    missing_set = set(missing)
    missing_required_runtime_shards = sorted(
        shard for shard in runtime_required_shards if shard in missing_set
    )
    missing_extra_shards = sorted(missing_set - set(missing_required_runtime_shards))
    missing_required_runtime_tensors = sorted(
        str(tensor_name)
        for tensor_name, shard in weight_map.items()
        if str(shard) in missing_set
        and _runtime_required_tensor(
            str(tensor_name), num_hidden_layers=num_hidden_layers
        )
    )
    missing_extra_tensors = sorted(
        str(tensor_name)
        for tensor_name, shard in weight_map.items()
        if str(shard) in missing_set
        and not _runtime_required_tensor(
            str(tensor_name), num_hidden_layers=num_hidden_layers
        )
    )
    present_bytes = sum(int(item["bytes"]) for item in shard_records if item["exists"])
    expected_source_bytes = (
        max(metadata_total_size, present_bytes)
        if metadata_total_size is not None
        else present_bytes
    )
    complete_source = not missing and bool(shards)
    complete_runtime_source = (
        not missing_required_runtime_shards and bool(runtime_required_shards)
    )
    decision = (
        "single_host_teacher_source_complete"
        if complete_source
        else "single_host_teacher_runtime_source_complete_full_source_incomplete"
        if complete_runtime_source
        else "single_host_teacher_source_incomplete"
    )
    present_required_runtime_shard_count = len(
        [
            shard
            for shard in runtime_required_shards
            if (source_dir / shard).exists()
        ]
    )

    record.update(
        {
            "decision": decision,
            "source_ready_for_single_host_export": complete_runtime_source,
            "complete_source": complete_source,
            "complete_runtime_source": complete_runtime_source,
            "num_hidden_layers": num_hidden_layers,
            "weight_map_tensor_count": len(weight_map),
            "expected_shard_count": len(shards),
            "present_shard_count": len(shards) - len(missing),
            "missing_shard_count": len(missing),
            "missing_shards": missing,
            "required_runtime_shard_count": len(runtime_required_shards),
            "present_required_runtime_shard_count": (
                present_required_runtime_shard_count
            ),
            "missing_required_runtime_shard_count": len(
                missing_required_runtime_shards
            ),
            "missing_required_runtime_shards": missing_required_runtime_shards,
            "missing_extra_shards": missing_extra_shards,
            "missing_required_runtime_tensor_count": len(
                missing_required_runtime_tensors
            ),
            "missing_required_runtime_tensor_examples": (
                missing_required_runtime_tensors[:20]
            ),
            "missing_extra_tensor_count": len(missing_extra_tensors),
            "missing_extra_tensor_examples": missing_extra_tensors[:20],
            "present_shard_bytes": int(present_bytes),
            "metadata_total_size": metadata_total_size,
            "memory_observation": _memory_observation(
                expected_source_bytes=expected_source_bytes,
                physical_memory_bytes=physical_memory_bytes,
                wired_limit_mb=wired_limit_mb,
            ),
        }
    )
    if scan_roots:
        candidate_scan = _scan_local_source_candidates(
            scan_roots=scan_roots,
            physical_memory_bytes=physical_memory_bytes,
            wired_limit_mb=wired_limit_mb,
        )
        current_summary = _candidate_summary(record)
        candidates = candidate_scan["candidates"]
        if current_summary["model_path"] not in {
            candidate["model_path"] for candidate in candidates
        }:
            candidates.insert(0, current_summary)
            if current_summary["source_ready_for_single_host_export"]:
                candidate_scan["complete_candidate_count"] += 1
            else:
                candidate_scan["incomplete_candidate_count"] += 1
        complete = [
            item for item in candidates if item["source_ready_for_single_host_export"]
        ]
        candidate_scan["local_export_candidate_ready"] = bool(complete)
        recommended_model_path = complete[0]["model_path"] if complete else None
        candidate_scan["recommended_model_path"] = recommended_model_path
        record["candidate_scan"] = candidate_scan
        record["local_export_candidate_ready"] = candidate_scan[
            "local_export_candidate_ready"
        ]
        record["recommended_model_path"] = candidate_scan["recommended_model_path"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a local GLM-4.5-Air teacher source snapshot is complete "
            "before a single-host cleanroom cache attempt."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--teacher-kind",
        choices=["bf16_source", "q8", "other_high_bit"],
        default="bf16_source",
    )
    parser.add_argument("--output-json")
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help=(
            "Optional local directory to scan recursively for alternate "
            "model.safetensors.index.json snapshots."
        ),
    )
    args = parser.parse_args()

    record = build_audit_record(
        model_path=args.model_path,
        revision=args.revision,
        teacher_kind=args.teacher_kind,
        physical_memory_bytes=_sysctl_int("hw.memsize"),
        wired_limit_mb=_sysctl_int("iogpu.wired_limit_mb"),
        scan_roots=tuple(Path(root) for root in args.scan_root),
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["source_ready_for_single_host_export"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
