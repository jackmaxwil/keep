from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm.models.glm4_moe import Model, ModelArgs

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID


GLM45_AIR_REVISION = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
LAYER_RE = re.compile(r"^model[.]layers[.](?P<layer>[0-9]+)[.]")
SWITCH_MLP_RE = re.compile(
    r"^model[.]layers[.](?P<layer>[0-9]+)[.]mlp[.]switch_mlp[.]"
    r"(?P<projection>gate_proj|up_proj|down_proj)[.]weight$"
)
METADATA_SUFFIXES = {
    ".json",
    ".jsonl",
    ".jinja",
    ".model",
    ".py",
    ".tiktoken",
    ".txt",
}


class _PipelineGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.py",
                "tokenizer.model",
                "*.tiktoken",
                "tiktoken.model",
                "*.txt",
                "*.jsonl",
                "*.jinja",
            ],
            local_files_only=True,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_weight_map(index_path: Path) -> dict[str, str]:
    payload = _read_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map object")
    return {str(name): str(shard) for name, shard in weight_map.items()}


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


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _is_stop_layer_post_gate_parameter(parameter_key: str, *, layer: int) -> bool:
    prefix = f"model.layers.{layer}.mlp."
    if not parameter_key.startswith(prefix):
        return False
    return not parameter_key.startswith(f"{prefix}gate.")


def _keep_rank0_only_logits_parameter(
    parameter_key: str,
    *,
    rank: int,
    pipeline_size: int,
) -> bool:
    if _tensor_layer(parameter_key) is not None:
        return True
    if parameter_key == "model.embed_tokens.weight":
        return rank == pipeline_size - 1
    if parameter_key in {"model.norm.weight", "lm_head.weight"}:
        return rank == 0
    return True


def _custom_layer_bounds(
    *,
    num_hidden_layers: int,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
) -> tuple[int, int] | None:
    if layer_split is None:
        return None
    if pipeline_size != 2:
        raise ValueError("--layer-split is currently supported only with --pipeline-size 2")
    if rank not in {0, 1}:
        raise ValueError("--layer-split expects ranks 0 and 1")
    if layer_split <= 0 or layer_split >= num_hidden_layers:
        raise ValueError(
            f"--layer-split must be between 1 and {num_hidden_layers - 1}, got {layer_split}"
        )
    if rank == 0:
        return layer_split, num_hidden_layers
    return 0, layer_split


def _apply_pipeline_split(
    language_model: Any,
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
) -> None:
    if layer_split is None:
        language_model.pipeline(_PipelineGroup(rank, pipeline_size))
        return

    start_idx, end_idx = _custom_layer_bounds(
        num_hidden_layers=len(language_model.layers),
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    language_model.pipeline_rank = rank
    language_model.pipeline_size = pipeline_size
    language_model.start_idx = start_idx
    language_model.end_idx = end_idx
    language_model.layers = language_model.layers[:end_idx]
    language_model.layers[:start_idx] = [None] * start_idx


def _pipeline_parameter_keys(
    config: dict[str, Any],
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int | None = None,
    rank0_stop_after_layer: int | None = None,
    rank0_route_trace_only: bool = False,
    route_trace_only_views: bool = False,
    rank0_stop_after_route_gate: bool = False,
    rank0_only_logits_views: bool = False,
) -> list[str]:
    model = Model(ModelArgs.from_dict(config))
    _apply_pipeline_split(
        model.model,
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    parameter_keys = [key for key, _ in tree_flatten(model.parameters())]
    if rank0_only_logits_views:
        parameter_keys = [
            key
            for key in parameter_keys
            if _keep_rank0_only_logits_parameter(
                key,
                rank=rank,
                pipeline_size=pipeline_size,
            )
        ]
    route_trace_only = route_trace_only_views or (rank == 0 and rank0_route_trace_only)
    if not route_trace_only:
        return parameter_keys
    filtered: list[str] = []
    for key in parameter_keys:
        layer = _tensor_layer(key)
        if layer is None:
            if key == "model.embed_tokens.weight" and rank == pipeline_size - 1:
                filtered.append(key)
            continue
        if rank == 0 and rank0_stop_after_layer is not None and layer > rank0_stop_after_layer:
            continue
        if (
            rank == 0
            and rank0_stop_after_route_gate
            and rank0_stop_after_layer is not None
            and layer == rank0_stop_after_layer
            and _is_stop_layer_post_gate_parameter(key, layer=layer)
        ):
            continue
        if rank != 0 or rank0_stop_after_layer is None or layer <= rank0_stop_after_layer:
            filtered.append(key)
    return filtered


def _validate_rank0_stop_after_layer(
    *,
    rank0_stop_after_layer: int | None,
    layer_split: int | None,
    pipeline_size: int,
    num_hidden_layers: int,
) -> None:
    if rank0_stop_after_layer is None:
        return
    if pipeline_size != 2:
        raise ValueError("--rank0-stop-after-layer is currently supported only with --pipeline-size 2")
    if layer_split is None:
        raise ValueError("--rank0-stop-after-layer requires --layer-split")
    if rank0_stop_after_layer < layer_split or rank0_stop_after_layer >= num_hidden_layers:
        raise ValueError(
            "--rank0-stop-after-layer must be between layer_split and "
            f"{num_hidden_layers - 1}, got {rank0_stop_after_layer}"
        )


def _validate_rank0_stop_after_route_gate(
    *,
    rank0_stop_after_route_gate: bool,
    rank0_stop_after_layer: int | None,
    rank0_route_trace_only: bool,
    route_trace_only_views: bool,
) -> None:
    if not rank0_stop_after_route_gate:
        return
    if rank0_stop_after_layer is None:
        raise ValueError("--rank0-stop-after-route-gate requires --rank0-stop-after-layer")
    if not (rank0_route_trace_only or route_trace_only_views):
        raise ValueError(
            "--rank0-stop-after-route-gate requires a route-trace-only rank-0 view"
        )


def source_tensors_for_parameter(
    parameter_key: str,
    *,
    weight_map: dict[str, str],
    n_routed_experts: int,
) -> list[str]:
    if parameter_key in weight_map:
        return [parameter_key]
    match = SWITCH_MLP_RE.match(parameter_key)
    if match is None:
        return [parameter_key]
    layer = int(match.group("layer"))
    projection = match.group("projection")
    return [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for expert in range(n_routed_experts)
    ]


def _parameter_shard(
    parameter_key: str,
    *,
    weight_map: dict[str, str],
    n_routed_experts: int,
) -> str | None:
    source_tensors = source_tensors_for_parameter(
        parameter_key,
        weight_map=weight_map,
        n_routed_experts=n_routed_experts,
    )
    for source_tensor in source_tensors:
        shard = weight_map.get(source_tensor)
        if shard is not None:
            return shard
    return None


def _layer_range(source_tensors: list[str]) -> dict[str, Any]:
    layers = sorted(
        {layer for tensor_name in source_tensors if (layer := _tensor_layer(tensor_name)) is not None}
    )
    if not layers:
        return {"layers": [], "first_layer": None, "last_layer": None}
    return {"layers": layers, "first_layer": layers[0], "last_layer": layers[-1]}


def _metadata_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "model.safetensors.index.json":
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        if path.suffix in METADATA_SUFFIXES:
            files.append(path)
    return files


def _build_source_view_plan(
    *,
    source_root: Path,
    weight_map: dict[str, str],
    header_cache: dict[str, dict[str, Any]],
    n_routed_experts: int,
    rank_budget_bytes: int,
    parameter_keys: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    pipeline_weight_map: dict[str, str] = {}
    source_tensors: list[str] = []
    missing_parameter_keys: list[str] = []

    for parameter_key in parameter_keys:
        shard = _parameter_shard(
            parameter_key,
            weight_map=weight_map,
            n_routed_experts=n_routed_experts,
        )
        if shard is None:
            missing_parameter_keys.append(parameter_key)
        else:
            pipeline_weight_map[parameter_key] = shard
        source_tensors.extend(
            source_tensors_for_parameter(
                parameter_key,
                weight_map=weight_map,
                n_routed_experts=n_routed_experts,
            )
        )

    required_source_tensors = _ordered_unique(source_tensors)
    required_shards: set[str] = set()
    present_source_tensors: list[str] = []
    missing_source_tensors: list[str] = []
    missing_shards: set[str] = set()
    required_present_tensor_bytes = 0

    for tensor_name in required_source_tensors:
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            missing_source_tensors.append(tensor_name)
            continue
        required_shards.add(shard_name)
        shard_path = source_root / shard_name
        if not shard_path.exists():
            missing_source_tensors.append(tensor_name)
            missing_shards.add(shard_name)
            continue
        header = header_cache.get(shard_name)
        if header is None:
            header = _read_safetensors_header(shard_path)
            header_cache[shard_name] = header
        if tensor_name not in header:
            missing_source_tensors.append(tensor_name)
            continue
        required_present_tensor_bytes += _tensor_bytes(header, tensor_name)
        present_source_tensors.append(tensor_name)

    visible_shard_file_bytes = sum(
        (source_root / shard_name).stat().st_size
        for shard_name in required_shards
        if (source_root / shard_name).exists()
    )
    return {
        **metadata,
        "parameter_key_count": len(parameter_keys),
        "required_source_tensor_count": len(required_source_tensors),
        "present_source_tensor_count": len(present_source_tensors),
        "missing_parameter_keys": missing_parameter_keys,
        "missing_source_tensors": missing_source_tensors[:50],
        "missing_source_tensor_count": len(missing_source_tensors),
        "missing_shards": sorted(missing_shards),
        "required_shards": sorted(required_shards),
        "required_shard_count": len(required_shards),
        "required_present_tensor_bytes": required_present_tensor_bytes,
        "visible_shard_file_bytes": visible_shard_file_bytes,
        "fits_required_tensor_budget": required_present_tensor_bytes <= rank_budget_bytes,
        "fits_visible_shard_budget": visible_shard_file_bytes <= rank_budget_bytes,
        "parameter_weight_map": dict(sorted(pipeline_weight_map.items())),
        "parameter_keys": parameter_keys,
        "required_source_tensors": required_source_tensors,
        **_layer_range(required_source_tensors),
    }


def _validate_stage_layer_window(
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int,
    num_hidden_layers: int,
    start_layer: int | None,
    stop_after_layer: int | None,
) -> None:
    if start_layer is None and stop_after_layer is None:
        return
    if start_layer is None or stop_after_layer is None:
        raise ValueError("stage layer window requires both start and stop layer")
    if start_layer > stop_after_layer:
        raise ValueError(
            f"stage start layer must be <= stop layer, got {start_layer}>{stop_after_layer}"
        )
    rank_start, rank_end = _custom_layer_bounds(
        num_hidden_layers=num_hidden_layers,
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    if start_layer < rank_start or stop_after_layer >= rank_end:
        raise ValueError(
            "stage layer window must be within rank layer bounds "
            f"{rank_start}..{rank_end - 1}, got {start_layer}..{stop_after_layer}"
        )


def _local_sequential_stage_parameter_keys(
    config: dict[str, Any],
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int,
    start_layer: int | None,
    stop_after_layer: int | None,
    include_embed: bool,
    include_norm: bool,
    include_lm_head: bool,
) -> list[str]:
    _validate_stage_layer_window(
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
        num_hidden_layers=int(config["num_hidden_layers"]),
        start_layer=start_layer,
        stop_after_layer=stop_after_layer,
    )
    parameter_keys = _pipeline_parameter_keys(
        config,
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    filtered: list[str] = []
    for key in parameter_keys:
        layer = _tensor_layer(key)
        if layer is not None:
            if (
                start_layer is not None
                and stop_after_layer is not None
                and start_layer <= layer <= stop_after_layer
            ):
                filtered.append(key)
            continue
        if key == "model.embed_tokens.weight":
            if include_embed:
                filtered.append(key)
        elif key == "model.norm.weight":
            if include_norm:
                filtered.append(key)
        elif key == "lm_head.weight":
            if include_lm_head:
                filtered.append(key)
        else:
            filtered.append(key)
    return filtered


def _normalize_stage_split_layers(
    split_layers: list[int] | None,
    *,
    single_split_layer: int | None,
    lower_bound: int,
    upper_bound: int,
    name: str,
) -> list[int]:
    if split_layers is None:
        split_layers = [] if single_split_layer is None else [single_split_layer]
    elif single_split_layer is not None and split_layers != [single_split_layer]:
        raise ValueError(
            f"{name} cannot combine single split {single_split_layer} with split list {split_layers}"
        )
    normalized = [int(layer) for layer in split_layers]
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{name} split layers must be sorted and unique")
    for layer in normalized:
        if layer <= lower_bound or layer >= upper_bound:
            raise ValueError(f"{name} split layer {layer} must be between {lower_bound + 1} and {upper_bound - 1}")
    return normalized


def _stage_windows(start_layer: int, stop_after_layer: int, split_layers: list[int]) -> list[tuple[int, int]]:
    starts = [start_layer, *split_layers]
    stops = [layer - 1 for layer in split_layers] + [stop_after_layer]
    return list(zip(starts, stops))


def build_local_sequential_stage_view_plan(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    rank_budget_bytes: int,
    layer_split: int,
    lower_split_layer: int | None = None,
    upper_split_layer: int | None = None,
    lower_split_layers: list[int] | None = None,
    upper_split_layers: list[int] | None = None,
    head_process: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_dir)
    index_root = Path(index_path)
    config = _read_json(source_root / "config.json")
    weight_map = _read_weight_map(index_root)
    n_routed_experts = int(config["n_routed_experts"])
    num_hidden_layers = int(config["num_hidden_layers"])
    pipeline_size = 2
    if layer_split <= 0 or layer_split >= num_hidden_layers:
        raise ValueError(
            f"layer_split must be between 1 and {num_hidden_layers - 1}, got {layer_split}"
        )
    lower_splits = _normalize_stage_split_layers(
        lower_split_layers,
        single_split_layer=lower_split_layer,
        lower_bound=0,
        upper_bound=layer_split,
        name="lower",
    )
    upper_splits = _normalize_stage_split_layers(
        upper_split_layers,
        single_split_layer=upper_split_layer,
        lower_bound=layer_split,
        upper_bound=num_hidden_layers,
        name="upper",
    )
    use_lower_window_names = lower_split_layers is not None
    use_upper_window_names = upper_split_layers is not None

    stage_specs: list[dict[str, Any]] = []
    if not lower_splits:
        stage_specs.append(
            {
                "stage": "lower",
                "rank": 1,
                "start_layer": 0,
                "stop_after_layer": layer_split - 1,
                "include_embed": True,
                "include_norm": False,
                "include_lm_head": False,
            }
        )
    elif use_lower_window_names:
        stage_specs.append(
            {
                "stage": "lower-embed",
                "rank": 1,
                "start_layer": None,
                "stop_after_layer": None,
                "include_embed": True,
                "include_norm": False,
                "include_lm_head": False,
            }
        )
        for index, (start_layer, stop_after_layer) in enumerate(
            _stage_windows(0, layer_split - 1, lower_splits)
        ):
            stage_specs.append(
                {
                    "stage": f"lower-window-{index}",
                    "rank": 1,
                    "start_layer": start_layer,
                    "stop_after_layer": stop_after_layer,
                    "include_embed": False,
                    "include_norm": False,
                    "include_lm_head": False,
                }
            )
    else:
        stage_specs.extend(
            [
                {
                    "stage": "lower-pre",
                    "rank": 1,
                    "start_layer": 0,
                    "stop_after_layer": lower_splits[0] - 1,
                    "include_embed": True,
                    "include_norm": False,
                    "include_lm_head": False,
                },
                {
                    "stage": "lower-final",
                    "rank": 1,
                    "start_layer": lower_splits[0],
                    "stop_after_layer": layer_split - 1,
                    "include_embed": False,
                    "include_norm": False,
                    "include_lm_head": False,
                },
            ]
        )

    upper_final_needs_lm_head = not head_process
    if not upper_splits:
        stage_specs.append(
            {
                "stage": "upper",
                "rank": 0,
                "start_layer": layer_split,
                "stop_after_layer": num_hidden_layers - 1,
                "include_embed": False,
                "include_norm": True,
                "include_lm_head": upper_final_needs_lm_head,
            }
        )
    elif use_upper_window_names:
        upper_windows = _stage_windows(layer_split, num_hidden_layers - 1, upper_splits)
        for index, (start_layer, stop_after_layer) in enumerate(upper_windows):
            is_final = index == len(upper_windows) - 1
            stage_specs.append(
                {
                    "stage": f"upper-window-{index}",
                    "rank": 0,
                    "start_layer": start_layer,
                    "stop_after_layer": stop_after_layer,
                    "include_embed": False,
                    "include_norm": is_final,
                    "include_lm_head": bool(is_final and upper_final_needs_lm_head),
                }
            )
    else:
        stage_specs.extend(
            [
                {
                    "stage": "upper-pre",
                    "rank": 0,
                    "start_layer": layer_split,
                    "stop_after_layer": upper_splits[0] - 1,
                    "include_embed": False,
                    "include_norm": False,
                    "include_lm_head": False,
                },
                {
                    "stage": "upper-final",
                    "rank": 0,
                    "start_layer": upper_splits[0],
                    "stop_after_layer": num_hidden_layers - 1,
                    "include_embed": False,
                    "include_norm": True,
                    "include_lm_head": upper_final_needs_lm_head,
                },
            ]
        )
    if head_process:
        stage_specs.append(
            {
                "stage": "head",
                "rank": 0,
                "start_layer": None,
                "stop_after_layer": None,
                "include_embed": False,
                "include_norm": False,
                "include_lm_head": True,
            }
        )

    header_cache: dict[str, dict[str, Any]] = {}
    stage_plans: list[dict[str, Any]] = []
    missing_source_tensors: list[str] = []
    missing_shards: set[str] = set()
    required_shards: set[str] = set()
    for spec in stage_specs:
        parameter_keys = _local_sequential_stage_parameter_keys(
            config,
            rank=int(spec["rank"]),
            pipeline_size=pipeline_size,
            layer_split=layer_split,
            start_layer=spec["start_layer"],
            stop_after_layer=spec["stop_after_layer"],
            include_embed=bool(spec["include_embed"]),
            include_norm=bool(spec["include_norm"]),
            include_lm_head=bool(spec["include_lm_head"]),
        )
        stage_plan = _build_source_view_plan(
            source_root=source_root,
            weight_map=weight_map,
            header_cache=header_cache,
            n_routed_experts=n_routed_experts,
            rank_budget_bytes=rank_budget_bytes,
            parameter_keys=parameter_keys,
            metadata={
                "stage": spec["stage"],
                "rank": spec["rank"],
                "pipeline_size": pipeline_size,
                "layer_split": layer_split,
                "lower_split_layer": lower_split_layer,
                "upper_split_layer": upper_split_layer,
                "head_process": bool(head_process),
                "start_layer": spec["start_layer"],
                "stop_after_layer": spec["stop_after_layer"],
                "include_embed": bool(spec["include_embed"]),
                "include_norm": bool(spec["include_norm"]),
                "include_lm_head": bool(spec["include_lm_head"]),
            },
        )
        missing_source_tensors.extend(stage_plan["missing_source_tensors"])
        missing_shards.update(stage_plan["missing_shards"])
        required_shards.update(stage_plan["required_shards"])
        stage_plans.append(stage_plan)

    return {
        "format": "glm45_air_local_sequential_stage_views",
        "source_dir": str(source_root),
        "index_path": str(index_root),
        "rank_budget_bytes": rank_budget_bytes,
        "pipeline_size": pipeline_size,
        "layer_split": layer_split,
        "lower_split_layer": lower_split_layer,
        "upper_split_layer": upper_split_layer,
        "lower_split_layers": lower_splits if use_lower_window_names else None,
        "upper_split_layers": upper_splits if use_upper_window_names else None,
        "head_process": bool(head_process),
        "num_hidden_layers": num_hidden_layers,
        "n_routed_experts": n_routed_experts,
        "complete_required_source": not missing_source_tensors and not missing_shards,
        "missing_source_tensor_count": len(missing_source_tensors),
        "missing_source_tensor_examples": missing_source_tensors[:50],
        "missing_shards": sorted(missing_shards),
        "required_shards": sorted(required_shards),
        "stage_plans": stage_plans,
        "notes": [
            "Stage views prune source visibility for local-sequential child workers.",
            "Each materialized stage root contains only the source tensors needed by that worker's layer window and non-layer role.",
            "This is a source-view/load preflight, not a completed teacher-cache forward pass.",
        ],
    }


def build_pipeline_view_plan(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    rank_budget_bytes: int,
    pipeline_size: int = 2,
    layer_split: int | None = None,
    rank0_stop_after_layer: int | None = None,
    rank0_route_trace_only: bool = False,
    route_trace_only_views: bool = False,
    rank0_stop_after_route_gate: bool = False,
    rank0_only_logits_views: bool = False,
    _include_layer_split_recommendation: bool = True,
) -> dict[str, Any]:
    if pipeline_size < 2:
        raise ValueError("pipeline_size must be at least 2")
    source_root = Path(source_dir)
    index_root = Path(index_path)
    config = _read_json(source_root / "config.json")
    weight_map = _read_weight_map(index_root)
    n_routed_experts = int(config["n_routed_experts"])
    num_hidden_layers = int(config["num_hidden_layers"])
    _validate_rank0_stop_after_layer(
        rank0_stop_after_layer=rank0_stop_after_layer,
        layer_split=layer_split,
        pipeline_size=pipeline_size,
        num_hidden_layers=num_hidden_layers,
    )
    _validate_rank0_stop_after_route_gate(
        rank0_stop_after_route_gate=rank0_stop_after_route_gate,
        rank0_stop_after_layer=rank0_stop_after_layer,
        rank0_route_trace_only=rank0_route_trace_only,
        route_trace_only_views=route_trace_only_views,
    )
    header_cache: dict[str, dict[str, Any]] = {}

    rank_plans: list[dict[str, Any]] = []
    all_missing_source_tensors: list[str] = []
    all_missing_shards: set[str] = set()
    all_required_shards: set[str] = set()

    for rank in range(pipeline_size):
        parameter_keys = _pipeline_parameter_keys(
            config,
            rank=rank,
            pipeline_size=pipeline_size,
            layer_split=layer_split,
            rank0_stop_after_layer=rank0_stop_after_layer,
            rank0_route_trace_only=rank0_route_trace_only,
            route_trace_only_views=route_trace_only_views,
            rank0_stop_after_route_gate=rank0_stop_after_route_gate,
            rank0_only_logits_views=rank0_only_logits_views,
        )
        pipeline_weight_map: dict[str, str] = {}
        source_tensors: list[str] = []
        missing_parameter_keys: list[str] = []

        for parameter_key in parameter_keys:
            shard = _parameter_shard(
                parameter_key,
                weight_map=weight_map,
                n_routed_experts=n_routed_experts,
            )
            if shard is None:
                missing_parameter_keys.append(parameter_key)
            else:
                pipeline_weight_map[parameter_key] = shard
            source_tensors.extend(
                source_tensors_for_parameter(
                    parameter_key,
                    weight_map=weight_map,
                    n_routed_experts=n_routed_experts,
                )
            )

        required_source_tensors = _ordered_unique(source_tensors)
        required_shards: set[str] = set()
        present_source_tensors: list[str] = []
        missing_source_tensors: list[str] = []
        missing_shards: set[str] = set()
        required_present_tensor_bytes = 0

        for tensor_name in required_source_tensors:
            shard_name = weight_map.get(tensor_name)
            if shard_name is None:
                missing_source_tensors.append(tensor_name)
                continue
            required_shards.add(shard_name)
            all_required_shards.add(shard_name)
            shard_path = source_root / shard_name
            if not shard_path.exists():
                missing_source_tensors.append(tensor_name)
                missing_shards.add(shard_name)
                all_missing_shards.add(shard_name)
                continue
            header = header_cache.get(shard_name)
            if header is None:
                header = _read_safetensors_header(shard_path)
                header_cache[shard_name] = header
            if tensor_name not in header:
                missing_source_tensors.append(tensor_name)
                continue
            required_present_tensor_bytes += _tensor_bytes(header, tensor_name)
            present_source_tensors.append(tensor_name)

        visible_shard_file_bytes = sum(
            (source_root / shard_name).stat().st_size
            for shard_name in required_shards
            if (source_root / shard_name).exists()
        )
        all_missing_source_tensors.extend(missing_source_tensors)
        layer_info = _layer_range(required_source_tensors)

        rank_plans.append(
            {
                "rank": rank,
                "layer_split": layer_split,
                "rank0_stop_after_layer": rank0_stop_after_layer if rank == 0 else None,
                "rank0_stop_after_route_gate": bool(
                    rank == 0 and rank0_stop_after_route_gate
                ),
                "rank0_only_logits_views": bool(rank0_only_logits_views),
                "route_trace_only": bool(route_trace_only_views or (rank == 0 and rank0_route_trace_only)),
                "pipeline_semantics": (
                    "Custom two-rank split: rank 1 owns layers below layer_split and "
                    "rank 0 owns layer_split through the final layer."
                    if layer_split is not None
                    else (
                        "MLX PipelineMixin assigns rank 0 to the last layer partition; "
                        "higher ranks own earlier layers."
                    )
                ),
                "parameter_key_count": len(parameter_keys),
                "required_source_tensor_count": len(required_source_tensors),
                "present_source_tensor_count": len(present_source_tensors),
                "missing_parameter_keys": missing_parameter_keys,
                "missing_source_tensors": missing_source_tensors[:50],
                "missing_source_tensor_count": len(missing_source_tensors),
                "missing_shards": sorted(missing_shards),
                "required_shards": sorted(required_shards),
                "required_shard_count": len(required_shards),
                "required_present_tensor_bytes": required_present_tensor_bytes,
                "visible_shard_file_bytes": visible_shard_file_bytes,
                "fits_required_tensor_budget": required_present_tensor_bytes <= rank_budget_bytes,
                "fits_visible_shard_budget": visible_shard_file_bytes <= rank_budget_bytes,
                "parameter_weight_map": dict(sorted(pipeline_weight_map.items())),
                "parameter_keys": parameter_keys,
                "required_source_tensors": required_source_tensors,
                **layer_info,
            }
        )

    plan = {
        "source_dir": str(source_root),
        "index_path": str(index_root),
        "rank_budget_bytes": rank_budget_bytes,
        "pipeline_size": pipeline_size,
        "layer_split": layer_split,
        "rank0_stop_after_layer": rank0_stop_after_layer,
        "rank0_route_trace_only": bool(rank0_route_trace_only),
        "route_trace_only_views": bool(route_trace_only_views),
        "rank0_stop_after_route_gate": bool(rank0_stop_after_route_gate),
        "rank0_only_logits_views": bool(rank0_only_logits_views),
        "num_hidden_layers": num_hidden_layers,
        "n_routed_experts": n_routed_experts,
        "complete_required_source": not all_missing_source_tensors and not all_missing_shards,
        "missing_source_tensor_count": len(all_missing_source_tensors),
        "missing_source_tensor_examples": all_missing_source_tensors[:50],
        "missing_shards": sorted(all_missing_shards),
        "required_shards": sorted(all_required_shards),
        "rank_plans": rank_plans,
        "materialized_view_compatible_with": "mlx_lm.utils.sharded_load local path and load_model lazy pipeline experiments",
        "existing_teacher_cache_exporter_compatible": False,
        "required_next_component": "distributed_bf16_pipeline_forward_and_teacher_cache_writer",
        "notes": [
            "Dry-run byte counts read safetensors headers only; tensor payloads are not loaded.",
            "required_present_tensor_bytes counts only tensors needed by this rank.",
            "visible_shard_file_bytes counts the whole shard files that a symlink view exposes.",
            "The generated index maps MLX pipeline parameter keys, including switch_mlp stacked keys, to source shard files.",
            "HF source tensors remain raw expert tensors; switch_mlp parameters expand to all mlp.experts.<n> projection tensors for the rank-local source files.",
            "This is a source-view/load preflight, not a completed teacher-cache forward pass.",
        ],
    }
    if _include_layer_split_recommendation:
        recommendation = _build_layer_split_recommendation(
            source_dir=source_root,
            index_path=index_root,
            rank_budget_bytes=rank_budget_bytes,
            pipeline_size=pipeline_size,
            num_hidden_layers=num_hidden_layers,
            rank0_stop_after_layer=rank0_stop_after_layer,
            rank0_route_trace_only=rank0_route_trace_only,
            route_trace_only_views=route_trace_only_views,
            rank0_stop_after_route_gate=rank0_stop_after_route_gate,
            rank0_only_logits_views=rank0_only_logits_views,
        )
        if recommendation is not None:
            plan["layer_split_recommendation"] = recommendation
            plan["notes"].append(
                "layer_split_recommendation scans candidate two-rank split boundaries "
                "using the same header-only source-residency accounting."
            )
    return plan


def _build_layer_split_recommendation(
    *,
    source_dir: Path,
    index_path: Path,
    rank_budget_bytes: int,
    pipeline_size: int,
    num_hidden_layers: int,
    rank0_stop_after_layer: int | None,
    rank0_route_trace_only: bool,
    route_trace_only_views: bool,
    rank0_stop_after_route_gate: bool,
    rank0_only_logits_views: bool,
) -> dict[str, Any] | None:
    if pipeline_size != 2 or num_hidden_layers < 2:
        return None
    if not rank0_only_logits_views:
        return None
    if rank0_stop_after_layer is not None or rank0_stop_after_route_gate:
        return None
    if rank0_route_trace_only or route_trace_only_views:
        return None

    candidates: list[dict[str, Any]] = []
    for candidate_split in range(1, num_hidden_layers):
        candidate_plan = build_pipeline_view_plan(
            source_dir=source_dir,
            index_path=index_path,
            rank_budget_bytes=rank_budget_bytes,
            pipeline_size=pipeline_size,
            layer_split=candidate_split,
            rank0_stop_after_layer=rank0_stop_after_layer,
            rank0_route_trace_only=rank0_route_trace_only,
            route_trace_only_views=route_trace_only_views,
            rank0_stop_after_route_gate=rank0_stop_after_route_gate,
            rank0_only_logits_views=rank0_only_logits_views,
            _include_layer_split_recommendation=False,
        )
        rank_plans = sorted(candidate_plan["rank_plans"], key=lambda item: item["rank"])
        required_bytes = [
            int(rank_plan["required_present_tensor_bytes"]) for rank_plan in rank_plans
        ]
        visible_bytes = [
            int(rank_plan["visible_shard_file_bytes"]) for rank_plan in rank_plans
        ]
        required_shard_counts = [
            int(rank_plan["required_shard_count"]) for rank_plan in rank_plans
        ]
        candidates.append(
            {
                "layer_split": candidate_split,
                "complete_required_source": bool(
                    candidate_plan["complete_required_source"]
                ),
                "fits_required_tensor_budget": all(
                    bool(rank_plan["fits_required_tensor_budget"])
                    for rank_plan in rank_plans
                ),
                "fits_visible_shard_budget": all(
                    bool(rank_plan["fits_visible_shard_budget"])
                    for rank_plan in rank_plans
                ),
                "required_present_tensor_bytes_by_rank": required_bytes,
                "visible_shard_file_bytes_by_rank": visible_bytes,
                "required_shard_count_by_rank": required_shard_counts,
                "max_required_present_tensor_bytes": max(required_bytes),
                "max_visible_shard_file_bytes": max(visible_bytes),
                "max_required_shard_count": max(required_shard_counts),
            }
        )

    recommendation_pool = [
        candidate for candidate in candidates if candidate["complete_required_source"]
    ] or candidates
    budget_fit_pool = [
        candidate
        for candidate in recommendation_pool
        if candidate["fits_required_tensor_budget"] and candidate["fits_visible_shard_budget"]
    ]
    if budget_fit_pool:
        recommendation_pool = budget_fit_pool

    recommended = min(
        recommendation_pool,
        key=lambda candidate: (
            candidate["max_required_present_tensor_bytes"],
            candidate["max_visible_shard_file_bytes"],
            candidate["max_required_shard_count"],
            abs(candidate["layer_split"] - (num_hidden_layers // 2)),
            candidate["layer_split"],
        ),
    )
    return {
        "optimization_target": (
            "minimize_max_required_present_tensor_bytes_then_visible_shard_file_bytes"
        ),
        "recommended_layer_split": recommended["layer_split"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _link_or_copy(source: Path, target: Path, *, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)


def materialize_pipeline_views(
    plan: dict[str, Any],
    *,
    output_dir: str | Path,
    copy_files: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    if not plan["complete_required_source"]:
        raise ValueError("cannot materialize views with missing required source tensors or shards")

    source_root = Path(plan["source_dir"])
    output_root = Path(output_dir)
    materialized: list[Path] = []
    metadata_files = _metadata_files(source_root)

    for rank_plan in plan["rank_plans"]:
        rank_dir = output_root / f"rank-{rank_plan['rank']}"
        if rank_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{rank_dir} already exists; pass --overwrite to replace it")
            shutil.rmtree(rank_dir)
        rank_dir.mkdir(parents=True)

        for metadata_path in metadata_files:
            _link_or_copy(metadata_path, rank_dir / metadata_path.name, copy_files=copy_files)

        index_payload = {
            "metadata": {
                "format": "glm45_air_pipeline_rank_view",
                "source_dir": str(source_root),
                "rank": rank_plan["rank"],
                "pipeline_size": plan["pipeline_size"],
                "layer_split": plan.get("layer_split"),
                "rank0_stop_after_layer": rank_plan.get("rank0_stop_after_layer"),
                "route_trace_only": rank_plan.get("route_trace_only", False),
                "required_present_tensor_bytes": rank_plan["required_present_tensor_bytes"],
                "visible_shard_file_bytes": rank_plan["visible_shard_file_bytes"],
            },
            "weight_map": rank_plan["parameter_weight_map"],
        }
        (rank_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (rank_dir / "pipeline_view_plan.json").write_text(
            json.dumps(rank_plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for shard_name in rank_plan["required_shards"]:
            _link_or_copy(source_root / shard_name, rank_dir / shard_name, copy_files=copy_files)

        materialized.append(rank_dir)

    return materialized


def materialize_local_sequential_stage_views(
    plan: dict[str, Any],
    *,
    output_dir: str | Path,
    copy_files: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not plan["complete_required_source"]:
        raise ValueError("cannot materialize stage views with missing required source tensors or shards")

    source_root = Path(plan["source_dir"])
    output_root = Path(output_dir)
    metadata_files = _metadata_files(source_root)
    materialized_stage_dirs: list[str] = []
    stage_view_roots: dict[str, dict[str, str]] = {}

    for stage_plan in plan["stage_plans"]:
        stage = str(stage_plan["stage"])
        stage_dir = output_root / stage
        if stage_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{stage_dir} already exists; pass --overwrite to replace it")
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)

        for metadata_path in metadata_files:
            _link_or_copy(metadata_path, stage_dir / metadata_path.name, copy_files=copy_files)

        index_payload = {
            "metadata": {
                "format": "glm45_air_local_sequential_stage_view",
                "source_dir": str(source_root),
                "stage": stage,
                "rank": stage_plan["rank"],
                "pipeline_size": plan["pipeline_size"],
                "layer_split": plan["layer_split"],
                "lower_split_layer": plan.get("lower_split_layer"),
                "upper_split_layer": plan.get("upper_split_layer"),
                "head_process": bool(plan.get("head_process", False)),
                "start_layer": stage_plan.get("start_layer"),
                "stop_after_layer": stage_plan.get("stop_after_layer"),
                "include_embed": bool(stage_plan.get("include_embed", False)),
                "include_norm": bool(stage_plan.get("include_norm", False)),
                "include_lm_head": bool(stage_plan.get("include_lm_head", False)),
                "required_present_tensor_bytes": stage_plan["required_present_tensor_bytes"],
                "visible_shard_file_bytes": stage_plan["visible_shard_file_bytes"],
            },
            "weight_map": stage_plan["parameter_weight_map"],
        }
        (stage_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage_dir / "pipeline_stage_view_plan.json").write_text(
            json.dumps(stage_plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for shard_name in stage_plan["required_shards"]:
            _link_or_copy(source_root / shard_name, stage_dir / shard_name, copy_files=copy_files)

        materialized_stage_dirs.append(str(stage_dir))
        stage_view_roots[stage] = {str(stage_plan["rank"]): str(stage_dir)}

    roots_path = output_root / "local_sequential_stage_view_roots.json"
    roots_path.write_text(
        json.dumps(stage_view_roots, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "materialized_stage_dirs": materialized_stage_dirs,
        "stage_view_roots": stage_view_roots,
        "stage_view_roots_json": str(roots_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or materialize rank-local GLM-4.5-Air source views for "
            "upstream MLX pipeline loading."
        )
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default=GLM45_AIR_REVISION)
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--pipeline-size", type=int, default=2)
    parser.add_argument(
        "--layer-split",
        type=int,
        help=(
            "Optional two-rank split boundary. Rank 1 owns layers below this "
            "index and rank 0 owns this layer through the final layer."
        ),
    )
    parser.add_argument(
        "--rank0-stop-after-layer",
        type=int,
        help=(
            "For route-trace-only rank-0 views, omit rank-0 layer tensors after this "
            "absolute layer index. Requires --layer-split and --pipeline-size 2."
        ),
    )
    parser.add_argument(
        "--rank0-route-trace-only",
        action="store_true",
        help=(
            "With --rank0-stop-after-layer, omit rank-0 non-layer tensors such as "
            "embeddings, norm, and lm_head that route-trace-only export does not evaluate."
        ),
    )
    parser.add_argument(
        "--route-trace-only-views",
        action="store_true",
        help=(
            "Omit non-layer tensors unused by route-trace-only distributed export from all "
            "rank views, while keeping embeddings on the input rank."
        ),
    )
    parser.add_argument(
        "--rank0-stop-after-route-gate",
        action="store_true",
        help=(
            "For a route-trace-only rank-0 stop-layer view, keep the stop layer router gate "
            "but omit tensors used only after that gate, such as routed and shared experts."
        ),
    )
    parser.add_argument(
        "--rank0-only-logits-views",
        action="store_true",
        help=(
            "For rank0-only logits export, keep embeddings only on the input rank "
            "and keep norm/lm_head only on rank 0."
        ),
    )
    parser.add_argument(
        "--local-sequential-stage-views",
        action="store_true",
        help=(
            "Build source-pruned view roots for local-sequential stage workers instead "
            "of the two full rank roots."
        ),
    )
    parser.add_argument("--local-sequential-lower-split-layer", type=int)
    parser.add_argument("--local-sequential-upper-split-layer", type=int)
    parser.add_argument(
        "--local-sequential-lower-split-layers",
        type=lambda value: [int(item) for item in value.split(",") if item],
        help=(
            "Comma-separated lower-rank layer split points for multi-window "
            "local-sequential stage views."
        ),
    )
    parser.add_argument(
        "--local-sequential-upper-split-layers",
        type=lambda value: [int(item) for item in value.split(",") if item],
        help=(
            "Comma-separated upper-rank layer split points for multi-window "
            "local-sequential stage views."
        ),
    )
    parser.add_argument("--local-sequential-head-process", action="store_true")
    parser.add_argument("--rank-budget-gb", type=float, default=115.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--copy", action="store_true", help="copy files instead of creating symlinks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rank_budget_gb <= 0:
        parser.error("--rank-budget-gb must be positive")
    if args.pipeline_size < 2:
        parser.error("--pipeline-size must be at least 2")
    if args.layer_split is not None and args.layer_split <= 0:
        parser.error("--layer-split must be positive")
    if args.layer_split is not None and args.pipeline_size != 2:
        parser.error("--layer-split is currently supported only with --pipeline-size 2")
    if args.rank0_stop_after_layer is not None and args.rank0_stop_after_layer < 0:
        parser.error("--rank0-stop-after-layer must be non-negative")
    if args.rank0_route_trace_only and args.rank0_stop_after_layer is None:
        parser.error("--rank0-route-trace-only requires --rank0-stop-after-layer")
    if args.rank0_stop_after_route_gate and args.rank0_stop_after_layer is None:
        parser.error("--rank0-stop-after-route-gate requires --rank0-stop-after-layer")
    if args.rank0_stop_after_route_gate and not (
        args.rank0_route_trace_only or args.route_trace_only_views
    ):
        parser.error("--rank0-stop-after-route-gate requires a route-trace-only rank-0 view")
    if args.copy and not args.materialize:
        parser.error("--copy requires --materialize")
    if args.overwrite and not args.materialize:
        parser.error("--overwrite requires --materialize")
    if args.materialize and not args.output_dir:
        parser.error("--materialize requires --output-dir")
    if args.local_sequential_stage_views and args.layer_split is None:
        parser.error("--local-sequential-stage-views requires --layer-split")
    local_sequential_split_flags = (
        args.local_sequential_lower_split_layer is not None
        or args.local_sequential_upper_split_layer is not None
        or args.local_sequential_lower_split_layers is not None
        or args.local_sequential_upper_split_layers is not None
    )
    if local_sequential_split_flags and not args.local_sequential_stage_views:
        parser.error("local-sequential split layer flags require --local-sequential-stage-views")

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    if args.local_sequential_stage_views:
        plan = build_local_sequential_stage_view_plan(
            source_dir=source_dir,
            index_path=index_path,
            rank_budget_bytes=int(args.rank_budget_gb * 1024**3),
            layer_split=int(args.layer_split),
            lower_split_layer=args.local_sequential_lower_split_layer,
            upper_split_layer=args.local_sequential_upper_split_layer,
            lower_split_layers=args.local_sequential_lower_split_layers,
            upper_split_layers=args.local_sequential_upper_split_layers,
            head_process=args.local_sequential_head_process,
        )
        if args.materialize:
            materialized = materialize_local_sequential_stage_views(
                plan,
                output_dir=args.output_dir,
                copy_files=args.copy,
                overwrite=args.overwrite,
            )
            plan.update(materialized)
    else:
        plan = build_pipeline_view_plan(
            source_dir=source_dir,
            index_path=index_path,
            rank_budget_bytes=int(args.rank_budget_gb * 1024**3),
            pipeline_size=args.pipeline_size,
            layer_split=args.layer_split,
            rank0_stop_after_layer=args.rank0_stop_after_layer,
            rank0_route_trace_only=args.rank0_route_trace_only,
            route_trace_only_views=args.route_trace_only_views,
            rank0_stop_after_route_gate=args.rank0_stop_after_route_gate,
            rank0_only_logits_views=args.rank0_only_logits_views,
        )
        if args.materialize:
            materialized = materialize_pipeline_views(
                plan,
                output_dir=args.output_dir,
                copy_files=args.copy,
                overwrite=args.overwrite,
            )
            plan["materialized_rank_dirs"] = [str(path) for path in materialized]
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
