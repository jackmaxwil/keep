from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io.load import inspect_safetensors


_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_ROUTED_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_ARTIFACT_LAYER_RE = re.compile(r"^layer-\d+-(?P<projection>gate_proj|up_proj|down_proj)\.safetensors$")


def _load_config(*, model_id: str, revision: str, config_path: str | Path | None) -> dict[str, Any]:
    if config_path is not None:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text(encoding="utf-8"))
    except Exception:
        return fetch_hf_config(model_id, revision=revision)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | Path | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["config.json", "model.safetensors.index.json", "*.safetensors"],
            local_files_only=True,
        )
    )


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.match(name)
    return None if match is None else int(match.group("layer"))


def _is_mtp_tensor(name: str, config: dict[str, Any]) -> bool:
    layer_idx = _layer_index(name)
    if layer_idx is None:
        return False
    num_hidden_layers = config.get("num_hidden_layers")
    return num_hidden_layers is not None and layer_idx >= int(num_hidden_layers)


def _is_routed_expert_weight(name: str) -> bool:
    return _ROUTED_EXPERT_WEIGHT_RE.match(name) is not None


def _surface_for_source_tensor(name: str) -> str:
    if name.startswith("model.embed_tokens."):
        return "embed_tokens"
    if name.startswith("lm_head."):
        return "lm_head"
    if ".mlp.gate." in name:
        return "router_gates"
    if ".mlp.shared_expert" in name or ".mlp.shared_experts" in name:
        return "shared_experts"
    if ".mlp." in name:
        return "dense_mlp"
    if "norm" in name or name.endswith("layernorm.weight"):
        return "norms"
    if ".self_attn." in name:
        return "attention"
    return "other_non_expert"


def _load_non_expert_precision_surfaces(artifact_dir: str | Path) -> tuple[str, ...]:
    manifest_path = Path(artifact_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return ()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest.get("non_expert_precision")
    if not isinstance(policy, dict) or policy.get("enabled") is not True:
        return ()
    surfaces = policy.get("surfaces")
    if not isinstance(surfaces, list) or not all(isinstance(item, str) for item in surfaces):
        raise ValueError("non_expert_precision.surfaces must be a list of strings")
    return tuple(surfaces)


def _add_group(
    groups: dict[str, dict[str, int]],
    surface: str,
    *,
    nbytes: int,
) -> None:
    group = groups.setdefault(surface, {"bytes": 0, "tensor_count": 0})
    group["bytes"] += int(nbytes)
    group["tensor_count"] += 1


def _format_groups(groups: dict[str, dict[str, int]]) -> list[dict[str, int | str]]:
    return [
        {"surface": surface, "bytes": values["bytes"], "tensor_count": values["tensor_count"]}
        for surface, values in sorted(
            groups.items(),
            key=lambda item: (-item[1]["bytes"], item[0]),
        )
    ]


def _summarize_source_bytes(
    *,
    source_dir: Path,
    index,
    config: dict[str, Any],
    requested_surfaces: tuple[str, ...],
) -> dict[str, Any]:
    # Policy surfaces are experiment metadata; runtime binds every source non-expert.
    inspections = {}
    loaded_groups: dict[str, dict[str, int]] = {}
    skipped_routed_groups: dict[str, dict[str, int]] = {}
    skipped_mtp_groups: dict[str, dict[str, int]] = {}
    skipped_by_surface_groups: dict[str, dict[str, int]] = {}
    missing_shards_by_category: dict[str, set[str]] = {
        "non_expert": set(),
        "routed_expert": set(),
        "mtp": set(),
        "skipped_by_surface": set(),
    }
    missing_tensor_counts: dict[str, int] = {
        "non_expert": 0,
        "routed_expert": 0,
        "mtp": 0,
        "skipped_by_surface": 0,
    }

    def tensor_nbytes(tensor_name: str, category: str) -> int | None:
        shard_path = source_dir / index.weight_map[tensor_name]
        inspection = inspections.get(shard_path)
        if inspection is None:
            try:
                inspection = inspect_safetensors(shard_path)
            except FileNotFoundError:
                missing_shards_by_category[category].add(str(shard_path))
                missing_tensor_counts[category] += 1
                return None
            inspections[shard_path] = inspection
        return int(inspection.tensors[tensor_name].nbytes)

    for tensor_name in sorted(index.weight_map):
        if _is_mtp_tensor(tensor_name, config):
            nbytes = tensor_nbytes(tensor_name, "mtp")
            if nbytes is not None:
                _add_group(skipped_mtp_groups, "mtp", nbytes=nbytes)
            continue
        if _is_routed_expert_weight(tensor_name):
            projection = tensor_name.rsplit(".", maxsplit=2)[-2]
            nbytes = tensor_nbytes(tensor_name, "routed_expert")
            if nbytes is not None:
                _add_group(skipped_routed_groups, f"routed_{projection}", nbytes=nbytes)
            continue
        surface = _surface_for_source_tensor(tensor_name)
        nbytes = tensor_nbytes(tensor_name, "non_expert")
        if nbytes is not None:
            _add_group(loaded_groups, surface, nbytes=nbytes)

    loaded_total = sum(group["bytes"] for group in loaded_groups.values())
    skipped_routed_total = sum(group["bytes"] for group in skipped_routed_groups.values())
    skipped_mtp_total = sum(group["bytes"] for group in skipped_mtp_groups.values())
    skipped_by_surface_total = sum(group["bytes"] for group in skipped_by_surface_groups.values())
    return {
        "source_non_expert_total_bytes": loaded_total,
        "source_non_expert_by_surface": _format_groups(loaded_groups),
        "source_skipped_routed_expert_total_bytes": skipped_routed_total,
        "source_skipped_routed_expert_by_surface": _format_groups(skipped_routed_groups),
        "source_skipped_mtp_total_bytes": skipped_mtp_total,
        "source_skipped_mtp_by_surface": _format_groups(skipped_mtp_groups),
        "source_skipped_by_surface_total_bytes": skipped_by_surface_total,
        "source_skipped_by_surface": _format_groups(skipped_by_surface_groups),
        "source_missing_shards_by_category": {
            category: sorted(shards)
            for category, shards in missing_shards_by_category.items()
            if shards
        },
        "source_missing_tensor_counts": {
            category: count for category, count in missing_tensor_counts.items() if count
        },
        "source_non_expert_complete": missing_tensor_counts["non_expert"] == 0,
    }


def _surface_for_artifact_path(path: Path, artifact_dir: Path) -> str:
    relative = path.relative_to(artifact_dir)
    if len(relative.parts) > 1 and relative.parts[0] == "continuous_params":
        return "continuous_sidecars"
    match = _ARTIFACT_LAYER_RE.match(relative.name)
    if match is not None and len(relative.parts) == 1:
        return f"vq_{match.group('projection')}"
    return "other_artifact"


def _summarize_artifact_bytes(artifact_dir: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    seen_real_paths: set[Path] = set()
    for path in sorted(artifact_dir.rglob("*.safetensors")):
        real_path = path.resolve()
        if real_path in seen_real_paths:
            continue
        seen_real_paths.add(real_path)
        inspection = inspect_safetensors(path)
        _add_group(
            groups,
            _surface_for_artifact_path(path, artifact_dir),
            nbytes=inspection.stored_nbytes,
        )
    total = sum(group["bytes"] for group in groups.values())
    return {
        "artifact_total_bytes": total,
        "artifact_by_surface": _format_groups(groups),
    }


def audit_resident_bytes(
    *,
    model_id: str = GLM45_AIR_MODEL_ID,
    revision: str = "main",
    source_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    index_path: str | Path | None = None,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    source_root = _resolve_source_dir(model_id=model_id, revision=revision, source_dir=source_dir)
    config = _load_config(model_id=model_id, revision=revision, config_path=config_path)
    index = load_safetensors_index(
        Path(index_path) if index_path is not None else source_root / "model.safetensors.index.json"
    )
    artifact_root = Path(artifact_dir)
    requested_surfaces = _load_non_expert_precision_surfaces(artifact_root)
    source_summary = _summarize_source_bytes(
        source_dir=source_root,
        index=index,
        config=config,
        requested_surfaces=requested_surfaces,
    )
    artifact_summary = _summarize_artifact_bytes(artifact_root)
    counted_total = (
        int(source_summary["source_non_expert_total_bytes"])
        + int(artifact_summary["artifact_total_bytes"])
    )
    record: dict[str, Any] = {
        "record_type": "glm45_air_resident_byte_audit",
        "model_id": model_id,
        "revision": revision,
        "source_dir": str(source_root),
        "artifact_dir": str(artifact_root),
        "non_expert_precision_surfaces": list(requested_surfaces),
        "counted_resident_total_bytes": counted_total,
        "counted_resident_total_gib": counted_total / (1024**3),
    }
    record.update(source_summary)
    record.update(artifact_summary)
    return record


def _int_field(record: dict[str, Any], name: str) -> int:
    value = record.get(name)
    if value is None:
        return 0
    return int(value)


def add_resident_byte_comparison(
    record: dict[str, Any],
    *,
    baseline_record: dict[str, Any],
    max_counted_resident_total_bytes: int | None = None,
    min_total_byte_reduction: int | None = None,
) -> dict[str, Any]:
    compared = dict(record)
    candidate_total = _int_field(record, "counted_resident_total_bytes")
    baseline_total = _int_field(baseline_record, "counted_resident_total_bytes")
    candidate_source = _int_field(record, "source_non_expert_total_bytes")
    baseline_source = _int_field(baseline_record, "source_non_expert_total_bytes")
    candidate_artifact = _int_field(record, "artifact_total_bytes")
    baseline_artifact = _int_field(baseline_record, "artifact_total_bytes")
    reduction_bytes = baseline_total - candidate_total

    compared.update(
        {
            "baseline_counted_resident_total_bytes": baseline_total,
            "baseline_source_non_expert_total_bytes": baseline_source,
            "baseline_artifact_total_bytes": baseline_artifact,
            "resident_byte_delta_bytes": candidate_total - baseline_total,
            "resident_byte_reduction_bytes": reduction_bytes,
            "resident_byte_reduction_ratio": (
                reduction_bytes / baseline_total if baseline_total else None
            ),
            "source_non_expert_delta_bytes": candidate_source - baseline_source,
            "artifact_delta_bytes": candidate_artifact - baseline_artifact,
        }
    )

    thresholds: dict[str, int] = {}
    failures: list[str] = []
    if max_counted_resident_total_bytes is not None:
        max_total = int(max_counted_resident_total_bytes)
        thresholds["max_counted_resident_total_bytes"] = max_total
        if candidate_total > max_total:
            failures.append(
                f"counted_resident_total_bytes {candidate_total} exceeds max {max_total}"
            )
    if min_total_byte_reduction is not None:
        min_reduction = int(min_total_byte_reduction)
        thresholds["min_total_byte_reduction"] = min_reduction
        if reduction_bytes < min_reduction:
            failures.append(
                f"resident_byte_reduction_bytes {reduction_bytes} is below min {min_reduction}"
            )

    compared["resident_byte_check"] = {
        "pass": not failures,
        "thresholds": thresholds,
        "failures": failures,
    }
    return compared


def _load_json_record(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"empty JSON record: {path}")
    if stripped.startswith("{"):
        payload = json.loads(stripped)
    else:
        payload = json.loads(stripped.splitlines()[-1])
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit GLM-4.5-Air resident source/artifact tensor bytes without loading the model."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--baseline-json")
    parser.add_argument("--max-counted-resident-total-bytes", type=int)
    parser.add_argument("--min-total-byte-reduction", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    record = audit_resident_bytes(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=args.artifact_dir,
    )
    if args.baseline_json:
        record = add_resident_byte_comparison(
            record,
            baseline_record=_load_json_record(args.baseline_json),
            max_counted_resident_total_bytes=args.max_counted_resident_total_bytes,
            min_total_byte_reduction=args.min_total_byte_reduction,
        )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.append_jsonl:
        append_jsonl(args.append_jsonl, record)
    if not args.output_json and not args.append_jsonl:
        print(json.dumps(record, indent=2, sort_keys=True))
    check = record.get("resident_byte_check")
    if isinstance(check, dict) and check.get("pass") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
