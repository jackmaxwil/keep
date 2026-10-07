from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from keep.convert.stream_convert import load_safetensors_index
from keep.io.source_safetensors import read_safetensors_tensor_header
from ramp.models.qwen_moe_adapter import (
    bind_qwen_moe_vq_experts,
    bind_qwen_non_expert_weights,
    has_unbound_qwen_moe_vq_experts,
)


_QWEN_LANGUAGE_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")


class _QwenRuntimeProbeMlp:
    def __init__(self) -> None:
        self.switch_mlp = None

    def bind_switch_mlp(self, switch_mlp) -> None:
        self.switch_mlp = switch_mlp


class _QwenRuntimeProbeLayer:
    def __init__(self) -> None:
        self.mlp = _QwenRuntimeProbeMlp()


class _QwenRuntimeProbeLanguageModel:
    def __init__(self, layer_count: int) -> None:
        self.layers = [_QwenRuntimeProbeLayer() for _ in range(layer_count)]


class _QwenRuntimeProbeModel:
    def __init__(self, *, layer_count: int, parameters: dict[str, mx.array]) -> None:
        self.language_model = _QwenRuntimeProbeLanguageModel(layer_count)
        self._parameters = parameters

    def parameters(self) -> dict[str, mx.array]:
        return self._parameters

    def load_weights(self, weights, *, strict: bool = False) -> None:
        for name, value in weights:
            if strict and name not in self._parameters:
                raise ValueError(f"unexpected Qwen runtime probe parameter {name!r}")
            self._parameters[name] = value


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if value == "":
        return ()
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


def _qwen_runtime_target_name(source_name: str) -> str | None:
    if source_name == "lm_head.weight" or source_name.startswith("lm_head."):
        return source_name
    if source_name.startswith("model.language_model."):
        return source_name.removeprefix("model.")
    return None


def _qwen_source_layer(name: str) -> int | None:
    match = _QWEN_LANGUAGE_LAYER_RE.match(name)
    if match is None:
        return None
    return int(match.group("layer"))


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
        target_name = _qwen_runtime_target_name(source_name)
        if target_name is None:
            continue
        selected.append((source_name, target_name))
        if max_tensors is not None and len(selected) >= max_tensors:
            break
    return tuple(selected)


def _placeholder_parameters_for_non_experts(
    source_root: Path,
    index,
    *,
    layers: tuple[int, ...] | None,
    layer_only: bool,
    max_tensors: int | None,
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    selected = _select_non_expert_source_tensors(
        index,
        layers=layers,
        layer_only=layer_only,
        max_tensors=max_tensors,
    )
    parameters: dict[str, mx.array] = {}
    source_tensors: list[str] = []
    for source_name, target_name in selected:
        target = _qwen_runtime_target_name(source_name) or target_name
        shard = index.weight_map[source_name]
        header = read_safetensors_tensor_header(source_root / shard, source_name)
        parameters[target] = mx.zeros(header.shape, dtype=_placeholder_dtype(header.dtype))
        source_tensors.append(source_name)
    return parameters, tuple(source_tensors)


def _forward_bound_switch(
    switch_mlp,
    *,
    layer: int,
    top_k: int,
    input_scale: float,
) -> dict[str, object]:
    input_dims = int(switch_mlp.input_dims)
    expert_count = int(switch_mlp.num_experts)
    if top_k <= 0:
        raise ValueError("forward_top_k must be positive")
    if top_k > expert_count:
        raise ValueError(
            f"forward_top_k={top_k} exceeds bound layer {layer} expert count {expert_count}"
        )
    x = mx.array(
        np.linspace(-input_scale, input_scale, input_dims, dtype=np.float32).reshape(
            1,
            input_dims,
        )
    )
    indices = mx.array(np.arange(top_k, dtype=np.int32).reshape(1, top_k))
    output = switch_mlp(x, indices)
    mx.eval(output)
    output_np = np.array(output)
    return {
        "layer": layer,
        "input_shape": list(x.shape),
        "indices_shape": list(indices.shape),
        "output_shape": list(output.shape),
        "finite": bool(np.isfinite(output_np).all()),
    }


def _compute_bounded_source_logits(
    parameters: dict[str, mx.array],
    *,
    token_id: int,
    top_k: int,
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("logit_probe_top_k must be positive")
    if "language_model.embed_tokens.weight" not in parameters:
        raise ValueError("logit probe requires language_model.embed_tokens.weight")
    if "lm_head.weight" not in parameters:
        raise ValueError("logit probe requires lm_head.weight")
    embeddings = parameters["language_model.embed_tokens.weight"]
    lm_head = parameters["lm_head.weight"]
    vocab = int(embeddings.shape[0])
    if token_id < 0 or token_id >= vocab:
        raise ValueError(f"logit_probe_token_id={token_id} outside embedding vocab {vocab}")
    if int(lm_head.shape[1]) != int(embeddings.shape[1]):
        raise ValueError(
            f"lm_head hidden dims {lm_head.shape[1]} do not match embedding dims {embeddings.shape[1]}"
        )
    hidden = embeddings[token_id].astype(mx.float32)
    logits = mx.matmul(lm_head.astype(mx.float32), hidden)
    mx.eval(logits)
    logits_np = np.array(logits)
    bounded_top_k = min(int(top_k), int(logits_np.shape[0]))
    top_indices = np.argsort(-logits_np, kind="stable")[:bounded_top_k]
    return {
        "token_id": int(token_id),
        "embedding_shape": list(hidden.shape),
        "lm_head_shape": list(lm_head.shape),
        "logits_shape": list(logits.shape),
        "top_k": bounded_top_k,
        "top_indices": [int(item) for item in top_indices],
        "finite": bool(np.isfinite(logits_np).all()),
    }


def _select_token_position(token_ids: list[int], position: str) -> int:
    if not token_ids:
        raise ValueError("tokenized logit probe prompt encoded to zero tokens")
    if position == "first":
        return 0
    if position == "last":
        return len(token_ids) - 1
    raise ValueError("tokenized_logit_position must be 'first' or 'last'")


def _compute_tokenized_source_logits(
    parameters: dict[str, mx.array],
    *,
    tokenizer_dir: str | Path,
    prompt: str,
    position: str,
    top_k: int,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    token_ids = [int(item) for item in tokenizer.encode(prompt, add_special_tokens=False)]
    selected_position = _select_token_position(token_ids, position)
    selected_token_id = token_ids[selected_position]
    logit_probe = _compute_bounded_source_logits(
        parameters,
        token_id=selected_token_id,
        top_k=top_k,
    )
    return {
        "record_type": "qwen_tokenized_source_logits_probe",
        "prompt": prompt,
        "encoded_token_ids": token_ids,
        "selected_token_position": selected_position,
        "selected_token_id": selected_token_id,
        "logit_probe": logit_probe,
        "finite": bool(logit_probe["finite"]),
    }


def probe_qwen_runtime_binding(
    *,
    artifact_dir: str | Path,
    source_dir: str | Path,
    index_path: str | Path,
    expected_layers: tuple[int, ...] | None,
    model_layer_count: int,
    forward_layers: tuple[int, ...] = (),
    forward_top_k: int = 8,
    input_scale: float = 0.125,
    layer_only: bool = False,
    max_tensors: int | None = None,
    logit_probe_token_id: int | None = None,
    logit_probe_top_k: int = 5,
    tokenizer_dir: str | Path | None = None,
    tokenizer_prompt: str = "",
    tokenized_logit_position: str = "last",
    tokenized_logit_top_k: int = 5,
) -> dict[str, object]:
    if model_layer_count < 0:
        raise ValueError("model_layer_count must be non-negative")
    source_root = Path(source_dir)
    index = load_safetensors_index(index_path)
    parameters, selected_source_tensors = _placeholder_parameters_for_non_experts(
        source_root,
        index,
        layers=expected_layers,
        layer_only=layer_only,
        max_tensors=max_tensors,
    )
    model = _QwenRuntimeProbeModel(
        layer_count=model_layer_count,
        parameters=parameters,
    )

    moe_report = bind_qwen_moe_vq_experts(
        model,
        artifact_dir,
        layers=expected_layers,
    )
    unbound = has_unbound_qwen_moe_vq_experts(model, layers=expected_layers)
    non_expert_report = bind_qwen_non_expert_weights(
        model,
        source_root,
        index,
        layers=expected_layers,
    )
    loaded = tuple(non_expert_report.loaded_model_parameters)
    loaded_dtype_names = {
        name: str(model.parameters()[name].dtype).removeprefix("mlx.core.")
        for name in loaded
        if name in model.parameters()
    }

    forward_results: list[dict[str, object]] = []
    forward_errors: list[str] = []
    for layer_idx in forward_layers:
        try:
            layer = model.language_model.layers[layer_idx]
            switch_mlp = layer.mlp.switch_mlp
            if switch_mlp is None:
                raise RuntimeError(f"layer {layer_idx} has no bound switch_mlp")
            forward_results.append(
                _forward_bound_switch(
                    switch_mlp,
                    layer=layer_idx,
                    top_k=forward_top_k,
                    input_scale=input_scale,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced in JSON evidence.
            forward_errors.append(f"layer {layer_idx}: {exc}")

    moe_binding_pass = not moe_report.missing_layers and not unbound
    non_expert_binding_pass = (
        not non_expert_report.missing_model_parameters
        and set(loaded) == set(model.parameters())
    )
    forward_pass = not forward_errors and all(
        bool(record["finite"]) for record in forward_results
    )
    logit_probe: dict[str, object] | None = None
    logit_probe_errors: list[str] = []
    if logit_probe_token_id is not None:
        try:
            logit_probe = _compute_bounded_source_logits(
                model.parameters(),
                token_id=logit_probe_token_id,
                top_k=logit_probe_top_k,
            )
        except Exception as exc:  # pragma: no cover - surfaced in JSON evidence.
            logit_probe_errors.append(str(exc))
    logit_probe_pass = not logit_probe_errors and (
        logit_probe_token_id is None or bool(logit_probe and logit_probe["finite"])
    )
    tokenized_logit_probe: dict[str, object] | None = None
    tokenized_logit_probe_errors: list[str] = []
    if tokenizer_dir is not None:
        try:
            tokenized_logit_probe = _compute_tokenized_source_logits(
                model.parameters(),
                tokenizer_dir=tokenizer_dir,
                prompt=tokenizer_prompt,
                position=tokenized_logit_position,
                top_k=tokenized_logit_top_k,
            )
        except Exception as exc:  # pragma: no cover - surfaced in JSON evidence.
            tokenized_logit_probe_errors.append(str(exc))
    tokenized_logit_probe_pass = not tokenized_logit_probe_errors and (
        tokenizer_dir is None
        or bool(tokenized_logit_probe and tokenized_logit_probe["finite"])
    )
    runtime_binding_pass = (
        moe_binding_pass
        and non_expert_binding_pass
        and forward_pass
        and logit_probe_pass
        and tokenized_logit_probe_pass
    )
    return {
        "artifact_dir": str(Path(artifact_dir)),
        "source_dir": str(source_root),
        "index_path": str(Path(index_path)),
        "expected_layers": None if expected_layers is None else list(expected_layers),
        "model_layer_count": model_layer_count,
        "selected_source_tensor_count": len(selected_source_tensors),
        "loaded_model_parameter_count": len(loaded),
        "loaded_dtype_names": loaded_dtype_names,
        "missing_model_parameters": list(non_expert_report.missing_model_parameters),
        "bound_layers": list(moe_report.bound_layers),
        "missing_layers": list(moe_report.missing_layers),
        "skipped_layers": list(moe_report.skipped_layers),
        "unbound_vq_experts": unbound,
        "moe_binding_pass": moe_binding_pass,
        "non_expert_binding_pass": non_expert_binding_pass,
        "forward_layers": list(forward_layers),
        "forward_top_k": forward_top_k,
        "forward_results": forward_results,
        "forward_errors": forward_errors,
        "forward_pass": forward_pass,
        "logit_probe": logit_probe,
        "logit_probe_errors": logit_probe_errors,
        "logit_probe_pass": logit_probe_pass,
        "tokenized_logit_probe": tokenized_logit_probe,
        "tokenized_logit_probe_errors": tokenized_logit_probe_errors,
        "tokenized_logit_probe_pass": tokenized_logit_probe_pass,
        "runtime_binding_pass": runtime_binding_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe combined Qwen source/VQ runtime binding plus switch forward."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--expected-layers")
    parser.add_argument("--model-layer-count", type=int, required=True)
    parser.add_argument("--forward-layers", default="")
    parser.add_argument("--forward-top-k", type=int, default=8)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--logit-probe-token-id", type=int)
    parser.add_argument("--logit-probe-top-k", type=int, default=5)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--tokenizer-prompt", default="")
    parser.add_argument(
        "--tokenized-logit-position",
        choices=("first", "last"),
        default="last",
    )
    parser.add_argument("--tokenized-logit-top-k", type=int, default=5)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = probe_qwen_runtime_binding(
        artifact_dir=args.artifact_dir,
        source_dir=args.source_dir,
        index_path=args.index_path,
        expected_layers=_parse_layers(args.expected_layers),
        model_layer_count=args.model_layer_count,
        forward_layers=_parse_layers(args.forward_layers) or (),
        forward_top_k=args.forward_top_k,
        input_scale=args.input_scale,
        layer_only=args.layer_only,
        max_tensors=args.max_tensors,
        logit_probe_token_id=args.logit_probe_token_id,
        logit_probe_top_k=args.logit_probe_top_k,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_prompt=args.tokenizer_prompt,
        tokenized_logit_position=args.tokenized_logit_position,
        tokenized_logit_top_k=args.tokenized_logit_top_k,
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["runtime_binding_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
