from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx_lm.models.base import create_attention_mask

from mlx_vq.benchmark.glm45_air import load_resident_air
from mlx_vq.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from mlx_vq.models.glm45_air_vq_adapter import GLM45AirVQMoE
from mlx_vq.quality.layer_probe_attribution import (
    LayerProbeTarget,
    load_layer_probe_state_bundle,
    parse_layer_spec,
    parse_probe_target,
    summarize_layer_probe_records,
    write_layer_probe_state_bundle,
)
from mlx_vq.quality.teacher_cache import read_teacher_cache_rows
from mlx_vq.validate.glm45_air_vq import validate_glm45_air_vq


DEFAULT_LAYERS = "1,6,11,16,21,26,31,36,41,45"


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict[str, Any]:
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


def _target_metadata(
    *,
    target: LayerProbeTarget,
    teacher_rows_by_prompt: dict[str, dict[str, Any]],
    eval_rows_by_prompt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        teacher_row = teacher_rows_by_prompt[target.prompt_id]
    except KeyError as error:
        raise ValueError(f"teacher JSONL is missing prompt {target.prompt_id!r}") from error
    positions = [int(position) for position in teacher_row.get("positions") or []]
    if target.position not in positions:
        raise ValueError(
            f"prompt {target.prompt_id!r} has no teacher position {target.position}; "
            f"available positions: {positions[:8]}..."
        )
    token_index = positions.index(target.position)
    input_token_ids = [int(token_id) for token_id in teacher_row["input_token_ids"]]
    if target.position + 1 >= len(input_token_ids):
        raise ValueError(
            f"target {target.key} is outside input token range length {len(input_token_ids)}"
        )
    eval_row = eval_rows_by_prompt.get(target.prompt_id, {})
    def list_item(key: str) -> Any:
        values = eval_row.get(key)
        if isinstance(values, list) and token_index < len(values):
            return values[token_index]
        return None

    return {
        "key": target.key,
        "prompt_id": target.prompt_id,
        "position": target.position,
        "token_index": token_index,
        "input_token_count": len(input_token_ids),
        "target_token_id": int(teacher_row["target_token_ids"][token_index]),
        "teacher_top1_id": list_item("teacher_top1_ids"),
        "vq_top1_id": list_item("vq_top1_ids"),
        "token_kld": list_item("token_klds"),
        "top1_match": (
            list_item("teacher_top1_ids") == list_item("vq_top1_ids")
            if list_item("teacher_top1_ids") is not None
            and list_item("vq_top1_ids") is not None
            else None
        ),
        "input_token_ids": input_token_ids,
    }


def _capture_prompt_states_for_layers(
    model,
    input_token_ids: list[int],
    *,
    layers: tuple[int, ...],
) -> dict[int, np.ndarray]:
    if not input_token_ids:
        raise ValueError("input_token_ids must not be empty")
    requested = set(layers)
    captures: dict[int, np.ndarray] = {}
    inputs = mx.array([input_token_ids], dtype=mx.int32)
    h = model.model.embed_tokens(inputs)
    cache = [None] * len(model.layers)
    mask = create_attention_mask(h, cache[0])

    for layer_idx, layer_module in enumerate(model.layers):
        if layer_idx not in requested:
            h = layer_module(h, mask, cache[layer_idx])
            continue
        if not isinstance(layer_module.mlp, GLM45AirVQMoE):
            raise ValueError(f"layer {layer_idx} is not a sparse GLM-4.5-Air MoE layer")
        attention = layer_module.self_attn(layer_module.input_layernorm(h), mask, cache[layer_idx])
        h_after_attention = h + attention
        moe_input = layer_module.post_attention_layernorm(h_after_attention)
        mx.eval(moe_input)
        captures[layer_idx] = np.asarray(moe_input[0].astype(mx.float32), dtype=np.float32)
        h = h_after_attention + layer_module.mlp(moe_input)

    missing = sorted(requested - set(captures))
    if missing:
        raise ValueError(f"requested layers were not captured: {missing}")
    return captures


def _strip_prompt_tokens(record: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(record)
    prompt_tokens = stripped.get("prompt_token_ids")
    if isinstance(prompt_tokens, list):
        stripped["prompt_token_count"] = len(prompt_tokens)
        stripped.pop("prompt_token_ids", None)
    return stripped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run coarse GLM-4.5-Air source-oracle layer/projection attribution probes."
    )
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument(
        "--capture-artifact-dir",
        help="Resident artifact used only for prompt hidden-state capture; defaults to --artifact-dir.",
    )
    parser.add_argument(
        "--write-state-npz",
        help="Write captured prompt hidden states to this NPZ for later --state-npz validation.",
    )
    parser.add_argument(
        "--state-npz",
        help="Load captured prompt hidden states from this NPZ and skip resident model capture.",
    )
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        type=parse_probe_target,
        help="Probe target as PROMPT_ID:POSITION. May be supplied more than once.",
    )
    parser.add_argument("--strict-config", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.write_state_npz and args.state_npz:
        parser.error("--write-state-npz and --state-npz cannot be used together")

    layers = parse_layer_spec(args.layers)
    teacher_rows = read_teacher_cache_rows(args.teacher_jsonl)
    eval_rows = read_teacher_cache_rows(args.eval_jsonl)
    teacher_rows_by_prompt = {str(row["prompt_id"]): row for row in teacher_rows}
    eval_rows_by_prompt = {str(row["prompt_id"]): row for row in eval_rows}
    targets = [
        _target_metadata(
            target=target,
            teacher_rows_by_prompt=teacher_rows_by_prompt,
            eval_rows_by_prompt=eval_rows_by_prompt,
        )
        for target in args.target
    ]
    targets_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        targets_by_prompt.setdefault(str(target["prompt_id"]), []).append(target)
    if (args.write_state_npz or args.state_npz) and len(targets_by_prompt) != 1:
        parser.error("state NPZ mode currently supports exactly one prompt")

    source_dir = Path(args.source_dir)
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    config = _load_config(
        model_id=args.model_id,
        revision=args.revision,
        config_path=args.config_path,
    )

    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    start = time.perf_counter()
    model = None
    non_expert_report = None
    bound_layers: list[int] = []
    if not args.state_npz:
        model, _, non_expert_report, bound_layers = load_resident_air(
            model_id=args.model_id,
            revision=args.revision,
            source_dir=str(source_dir),
            config_path=args.config_path,
            index_path=str(index_path),
            artifact_dir=args.capture_artifact_dir or args.artifact_dir,
        )
    validation_records: list[dict[str, Any]] = []
    for prompt_id, prompt_targets in targets_by_prompt.items():
        if args.state_npz:
            captures = load_layer_probe_state_bundle(
                args.state_npz,
                prompt_id=prompt_id,
                input_token_ids=prompt_targets[0]["input_token_ids"],
                layers=layers,
            )
        else:
            if model is None:
                raise RuntimeError("resident model was not loaded")
            captures = _capture_prompt_states_for_layers(
                model,
                prompt_targets[0]["input_token_ids"],
                layers=layers,
            )
            if args.write_state_npz:
                write_layer_probe_state_bundle(
                    args.write_state_npz,
                    prompt_id=prompt_id,
                    input_token_ids=prompt_targets[0]["input_token_ids"],
                    captures=captures,
                )
        for target in prompt_targets:
            position = int(target["position"])
            for layer in layers:
                captured = captures[layer]
                if position >= captured.shape[0]:
                    raise ValueError(
                        f"target {target['key']} position {position} is outside captured "
                        f"state length {captured.shape[0]} for layer {layer}"
                    )
                result = validate_glm45_air_vq(
                    config,
                    source_dir=source_dir,
                    index_path=index_path,
                    artifact_dir=args.artifact_dir,
                    model_id=args.model_id,
                    layer=layer,
                    tokens=1,
                    input_states=captured[position : position + 1],
                    input_source=f"teacher_cache:{target['key']}",
                    prompt_token_ids=tuple(int(token_id) for token_id in target["input_token_ids"]),
                    input_state_indices=(position,),
                    strict_config=args.strict_config,
                )
                payload = _strip_prompt_tokens(result.to_dict())
                payload["probe_target"] = {
                    key: value
                    for key, value in target.items()
                    if key != "input_token_ids"
                }
                validation_records.append(payload)

    elapsed_seconds = time.perf_counter() - start
    memory = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    memory_clean = memory.get("pageouts_delta") == 0 and memory.get("swapouts_delta") == 0
    summary = summarize_layer_probe_records(validation_records)
    report = {
        "schema_version": 1,
        "evidence_scope": "source_oracle_attribution_only",
        "memory_clean": memory_clean,
        "capture_state_source": "state_npz" if args.state_npz else "resident_vq_hidden_states",
        "state_npz": args.state_npz,
        "write_state_npz": args.write_state_npz,
        "teacher_jsonl": args.teacher_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "source_dir": str(source_dir),
        "artifact_dir": args.artifact_dir,
        "capture_artifact_dir": (
            None if args.state_npz else args.capture_artifact_dir or args.artifact_dir
        ),
        "layers": list(layers),
        "targets": [
            {key: value for key, value in target.items() if key != "input_token_ids"}
            for target in targets
        ],
        "elapsed_seconds": elapsed_seconds,
        "non_expert_bind_report": (
            non_expert_report.to_dict() if non_expert_report is not None else None
        ),
        "bound_vq_layers": list(bound_layers),
        "memory": memory,
        "summary": summary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_path),
        "record_count": summary["record_count"],
        "layer_count": summary["layer_count"],
        "target_count": summary["target_count"],
        "top_layer": summary["layer_rankings"][0]["layer"] if summary["layer_rankings"] else None,
        "top_layer_mean_source_weighted_rel_l2": (
            summary["layer_rankings"][0]["mean_source_weighted_rel_l2"]
            if summary["layer_rankings"]
            else None
        ),
        "memory_clean": memory_clean,
        "pageouts_delta": memory.get("pageouts_delta"),
        "swapouts_delta": memory.get("swapouts_delta"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
