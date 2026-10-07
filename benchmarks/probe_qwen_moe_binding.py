from __future__ import annotations

import argparse
import json
from pathlib import Path

from ramp.models.qwen_moe_adapter import (
    bind_qwen_moe_vq_experts,
    has_unbound_qwen_moe_vq_experts,
)


class _QwenProbeMlp:
    def __init__(self) -> None:
        self.switch_mlp = None

    def bind_switch_mlp(self, switch_mlp) -> None:
        self.switch_mlp = switch_mlp


class _QwenProbeLayer:
    def __init__(self) -> None:
        self.mlp = _QwenProbeMlp()


class _QwenProbeLanguageModel:
    def __init__(self, layer_count: int) -> None:
        self.layers = [_QwenProbeLayer() for _ in range(layer_count)]


class _QwenProbeModel:
    def __init__(self, layer_count: int) -> None:
        self.language_model = _QwenProbeLanguageModel(layer_count)


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


def probe_qwen_moe_binding_artifact(
    artifact_dir: str | Path,
    *,
    expected_layers: tuple[int, ...] | None = None,
    model_layer_count: int | None = None,
) -> dict[str, object]:
    if model_layer_count is None:
        if expected_layers is None:
            raise ValueError("model_layer_count is required when expected_layers is omitted")
        model_layer_count = max(expected_layers, default=-1) + 1
    if model_layer_count < 0:
        raise ValueError("model_layer_count must be non-negative")

    model = _QwenProbeModel(model_layer_count)
    report = bind_qwen_moe_vq_experts(
        model,
        artifact_dir,
        layers=expected_layers,
    )
    unbound = has_unbound_qwen_moe_vq_experts(model, layers=expected_layers)
    binding_pass = not report.missing_layers and not unbound
    return {
        "artifact_dir": str(Path(artifact_dir)),
        "expected_layers": None if expected_layers is None else list(expected_layers),
        "model_layer_count": model_layer_count,
        "bound_layers": list(report.bound_layers),
        "skipped_layers": list(report.skipped_layers),
        "missing_layers": list(report.missing_layers),
        "unbound_vq_experts": unbound,
        "dense_routed_experts": False,
        "binding_pass": binding_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Qwen MoE materialization binding without loading a full model."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--expected-layers")
    parser.add_argument("--model-layer-count", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = probe_qwen_moe_binding_artifact(
        args.artifact_dir,
        expected_layers=_parse_layers(args.expected_layers),
        model_layer_count=args.model_layer_count,
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
