from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx

from keep.convert.stream_convert import load_safetensors_index
from keep.io.source_safetensors import read_safetensors_tensor_header
from ramp.models.qwen_moe_adapter import bind_qwen_non_expert_weights


_QWEN_LANGUAGE_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")


class _QwenFlatNonExpertProbeModel:
    def __init__(self, parameters: dict[str, mx.array]) -> None:
        self._parameters = parameters

    def parameters(self) -> dict[str, mx.array]:
        return self._parameters

    def load_weights(self, weights, *, strict: bool = False) -> None:
        for name, value in weights:
            if strict and name not in self._parameters:
                raise ValueError(f"unexpected Qwen non-expert probe parameter {name!r}")
            self._parameters[name] = value


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    return tuple(int(item) for item in value.split(",") if item != "")


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _qwen_source_layer(name: str) -> int | None:
    match = _QWEN_LANGUAGE_LAYER_RE.match(name)
    if match is None:
        return None
    return int(match.group("layer"))


def _qwen_target_name(source_name: str) -> str | None:
    if source_name == "lm_head.weight" or source_name.startswith("lm_head."):
        return source_name
    if source_name.startswith("model.language_model."):
        return source_name.removeprefix("model.")
    return None


def _is_routed_or_auxiliary_qwen_tensor(source_name: str) -> bool:
    return (
        ".mlp.experts." in source_name
        or source_name.startswith("mtp.")
        or source_name.startswith("model.visual.")
    )


def _placeholder_dtype(header_dtype: str):
    if header_dtype == "BF16":
        return mx.bfloat16
    if header_dtype == "F16":
        return mx.float16
    return mx.float32


def _dtype_name(value: mx.array) -> str:
    return str(value.dtype).removeprefix("mlx.core.")


def _select_non_expert_source_tensors(
    index,
    *,
    layers: tuple[int, ...] | None = None,
    layer_only: bool = False,
    max_tensors: int | None = None,
) -> tuple[tuple[str, str], ...]:
    selected_layers = None if layers is None else set(layers)
    selected: list[tuple[str, str]] = []
    for source_name in sorted(index.weight_map):
        if _is_routed_or_auxiliary_qwen_tensor(source_name):
            continue
        layer = _qwen_source_layer(source_name)
        if layer_only and layer is None:
            continue
        if selected_layers is not None and layer is not None and layer not in selected_layers:
            continue
        target_name = _qwen_target_name(source_name)
        if target_name is None:
            continue
        selected.append((source_name, target_name))
        if max_tensors is not None and len(selected) >= max_tensors:
            break
    return tuple(selected)


def probe_qwen_non_expert_binding_source(
    source_dir: str | Path,
    *,
    index_path: str | Path,
    layers: tuple[int, ...] | None = None,
    layer_only: bool = False,
    max_tensors: int | None = None,
) -> dict[str, object]:
    source_root = Path(source_dir)
    index = load_safetensors_index(index_path)
    selected = _select_non_expert_source_tensors(
        index,
        layers=layers,
        layer_only=layer_only,
        max_tensors=max_tensors,
    )
    parameters: dict[str, mx.array] = {}
    source_tensors: list[str] = []
    for source_name, target_name in selected:
        shard = index.weight_map[source_name]
        header = read_safetensors_tensor_header(source_root / shard, source_name)
        parameters[target_name] = mx.zeros(header.shape, dtype=_placeholder_dtype(header.dtype))
        source_tensors.append(source_name)

    model = _QwenFlatNonExpertProbeModel(parameters)
    report = bind_qwen_non_expert_weights(
        model,
        source_root,
        index,
        layers=layers,
    )
    loaded = list(report.loaded_model_parameters)
    loaded_dtype_names = {
        name: _dtype_name(model.parameters()[name])
        for name in loaded
        if name in model.parameters()
    }
    binding_pass = not report.missing_model_parameters and set(loaded) == set(parameters)
    return {
        "source_dir": str(source_root),
        "index_path": str(Path(index_path)),
        "requested_layers": None if layers is None else list(layers),
        "layer_only": layer_only,
        "max_tensors": max_tensors,
        "selected_source_tensors": source_tensors,
        "selected_source_tensor_count": len(source_tensors),
        "loaded_model_parameters": loaded,
        "loaded_model_parameter_count": len(loaded),
        "loaded_dtype_names": loaded_dtype_names,
        "missing_model_parameters": list(report.missing_model_parameters),
        "skipped_routed_expert_tensor_count": len(report.skipped_routed_expert_tensors),
        "skipped_mtp_tensor_count": len(report.skipped_mtp_tensors),
        "skipped_visual_tensor_count": len(report.skipped_visual_tensors),
        "skipped_unmatched_tensor_count": len(report.skipped_unmatched_tensors),
        "skipped_by_layer_tensor_count": len(report.skipped_by_layer_tensors),
        "binding_pass": binding_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Qwen non-expert source binding without loading a full model."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--expected-layers")
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = probe_qwen_non_expert_binding_source(
        args.source_dir,
        index_path=args.index_path,
        layers=_parse_layers(args.expected_layers),
        layer_only=args.layer_only,
        max_tensors=args.max_tensors,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["binding_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
