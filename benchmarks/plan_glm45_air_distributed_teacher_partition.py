from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.convert.stream_convert import load_safetensors_index


GLM45_AIR_REVISION = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
LAYER_RE = re.compile(r"^model[.]layers[.](?P<layer>[0-9]+)[.]")


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


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"{path} is not a valid safetensors file")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        return json.loads(handle.read(header_size))


def _tensor_bytes(header: dict[str, Any], tensor_name: str) -> int:
    tensor_info = header[tensor_name]
    start, end = (int(offset) for offset in tensor_info["data_offsets"])
    return end - start


def _tensor_layer(tensor_name: str) -> int | None:
    match = LAYER_RE.match(tensor_name)
    if match is None:
        return None
    return int(match.group("layer"))


def _infer_num_hidden_layers(tensor_names: list[str]) -> int:
    layers = [layer for name in tensor_names if (layer := _tensor_layer(name)) is not None]
    return max(layers) + 1 if layers else 0


def _config_num_hidden_layers(source_dir: Path) -> int | None:
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = payload.get("num_hidden_layers")
    if value is None:
        return None
    return int(value)


def _non_layer_owner(tensor_name: str, *, last_rank: int) -> int:
    if tensor_name.startswith("model.embed_tokens."):
        return 0
    if tensor_name.startswith("model.norm.") or tensor_name.startswith("lm_head."):
        return last_rank
    return 0


def _contiguous_rank_for_layer(layer: int, split_layer: int) -> int:
    return 0 if layer < split_layer else 1


def _candidate_rank_bytes(
    *,
    layer_bytes: dict[int, int],
    non_layer_rank_bytes: dict[int, int],
    split_layer: int,
) -> tuple[int, int]:
    rank_bytes = [non_layer_rank_bytes.get(0, 0), non_layer_rank_bytes.get(1, 0)]
    for layer, byte_count in layer_bytes.items():
        rank_bytes[_contiguous_rank_for_layer(layer, split_layer)] += byte_count
    return rank_bytes[0], rank_bytes[1]


def _best_split(layer_bytes: dict[int, int], non_layer_rank_bytes: dict[int, int]) -> int:
    layers = sorted(layer_bytes)
    if len(layers) < 2:
        return layers[-1] + 1 if layers else 0
    first, last = layers[0], layers[-1]
    best_split = first + 1
    best_score: tuple[int, int] | None = None
    for split_layer in range(first + 1, last + 1):
        rank0, rank1 = _candidate_rank_bytes(
            layer_bytes=layer_bytes,
            non_layer_rank_bytes=non_layer_rank_bytes,
            split_layer=split_layer,
        )
        score = (max(rank0, rank1), abs(rank0 - rank1))
        if best_score is None or score < best_score:
            best_score = score
            best_split = split_layer
    return best_split


def build_partition_plan(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    rank_budget_bytes: int,
    num_hidden_layers: int | None = None,
) -> dict[str, Any]:
    source_root = Path(source_dir)
    index = load_safetensors_index(index_path)
    tensor_names = sorted(index.weight_map)
    if num_hidden_layers is None:
        num_hidden_layers = _config_num_hidden_layers(source_root)
    if num_hidden_layers is None:
        num_hidden_layers = _infer_num_hidden_layers(tensor_names)
    header_cache: dict[str, dict[str, Any]] = {}
    missing_shards: set[str] = set()
    missing_required_shards: set[str] = set()
    missing_tensors: list[str] = []
    missing_required_tensors: list[str] = []
    tensor_records: list[dict[str, Any]] = []
    layer_bytes: dict[int, int] = {}
    non_layer_rank_bytes: dict[int, int] = {0: 0, 1: 0}
    extra_layer_bytes: dict[int, int] = {}
    extra_tensor_count = 0

    for tensor_name in tensor_names:
        shard_name = index.weight_map[tensor_name]
        layer = _tensor_layer(tensor_name)
        required_runtime_tensor = layer is None or layer < num_hidden_layers
        shard_path = source_root / shard_name
        if not shard_path.exists():
            missing_shards.add(shard_name)
            missing_tensors.append(tensor_name)
            if required_runtime_tensor:
                missing_required_shards.add(shard_name)
                missing_required_tensors.append(tensor_name)
            continue
        header = header_cache.get(shard_name)
        if header is None:
            header = _read_safetensors_header(shard_path)
            header_cache[shard_name] = header
        byte_count = _tensor_bytes(header, tensor_name)
        if layer is None:
            owner = _non_layer_owner(tensor_name, last_rank=1)
            non_layer_rank_bytes[owner] = non_layer_rank_bytes.get(owner, 0) + byte_count
        elif layer >= num_hidden_layers:
            extra_layer_bytes[layer] = extra_layer_bytes.get(layer, 0) + byte_count
            extra_tensor_count += 1
        else:
            layer_bytes[layer] = layer_bytes.get(layer, 0) + byte_count
        tensor_records.append(
            {
                "name": tensor_name,
                "layer": layer,
                "shard": shard_name,
                "bytes": byte_count,
            }
        )

    split_layer = _best_split(layer_bytes, non_layer_rank_bytes)
    rank0_bytes, rank1_bytes = _candidate_rank_bytes(
        layer_bytes=layer_bytes,
        non_layer_rank_bytes=non_layer_rank_bytes,
        split_layer=split_layer,
    )
    layers = sorted(layer_bytes)
    rank0_layers = [layer for layer in layers if layer < split_layer]
    rank1_layers = [layer for layer in layers if layer >= split_layer]
    total_present_bytes = rank0_bytes + rank1_bytes
    complete = not missing_shards
    complete_runtime_source = not missing_required_shards

    rank_plans = [
        {
            "rank": 0,
            "role": "tokenizer_cache_writer_and_early_layers",
            "layers": rank0_layers,
            "present_weight_bytes": rank0_bytes,
            "fits_rank_budget": rank0_bytes <= rank_budget_bytes,
            "non_layer_assignment": "model.embed_tokens.* plus unclassified tensors",
        },
        {
            "rank": 1,
            "role": "late_layers_final_norm_lm_head",
            "layers": rank1_layers,
            "present_weight_bytes": rank1_bytes,
            "fits_rank_budget": rank1_bytes <= rank_budget_bytes,
            "non_layer_assignment": "model.norm.* and lm_head.*",
        },
    ]

    return {
        "source_dir": str(source_root),
        "index_path": str(index_path),
        "rank_count": 2,
        "rank_budget_bytes": rank_budget_bytes,
        "num_hidden_layers": num_hidden_layers,
        "complete_source": complete,
        "complete_runtime_source": complete_runtime_source,
        "missing_shards": sorted(missing_shards),
        "missing_required_shards": sorted(missing_required_shards),
        "missing_tensor_count": len(missing_tensors),
        "missing_required_tensor_count": len(missing_required_tensors),
        "missing_tensor_examples": missing_tensors[:20],
        "missing_required_tensor_examples": missing_required_tensors[:20],
        "present_tensor_count": len(tensor_records),
        "present_weight_bytes": total_present_bytes,
        "layer_count_present": len(layers),
        "extra_layer_count_present": len(extra_layer_bytes),
        "extra_layer_present_weight_bytes": sum(extra_layer_bytes.values()),
        "extra_tensor_count": extra_tensor_count,
        "recommended_split_layer": split_layer,
        "rank_plans": rank_plans,
        "max_rank_present_weight_bytes": max(rank0_bytes, rank1_bytes),
        "rank_present_weight_imbalance_bytes": abs(rank0_bytes - rank1_bytes),
        "existing_exporter_compatible": False,
        "required_next_component": "distributed_bf16_layer_partition_forward_and_cache_writer",
        "notes": [
            "Byte counts come from safetensors headers and do not load tensor payloads.",
            "Missing required runtime shards make the distributed teacher plan incomplete.",
            "Missing extra layers beyond num_hidden_layers are reported but not assigned to the runtime partition.",
            "The current mlx_lm.load() teacher-cache exporter is single-process and does not use this partition plan.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan a two-rank GLM-4.5-Air BF16 teacher partition from safetensors headers."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default=GLM45_AIR_REVISION)
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--rank-budget-gb", type=float, default=115.0)
    args = parser.parse_args()

    if args.rank_budget_gb <= 0:
        parser.error("--rank-budget-gb must be positive")
    if args.num_hidden_layers is not None and args.num_hidden_layers <= 0:
        parser.error("--num-hidden-layers must be positive")

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    plan = build_partition_plan(
        source_dir=source_dir,
        index_path=index_path,
        rank_budget_bytes=int(args.rank_budget_gb * 1024**3),
        num_hidden_layers=args.num_hidden_layers,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
