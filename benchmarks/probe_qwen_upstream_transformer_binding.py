from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models import qwen3_5_moe

from mlx_vq.models.qwen_moe_adapter import (
    bind_qwen_moe_vq_experts,
    has_unbound_qwen_moe_vq_experts,
)


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


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Qwen config must be a JSON object")
    return payload


def _projected_qwen35_moe_config(
    source_config: dict[str, Any],
    *,
    layer_count: int,
) -> dict[str, Any]:
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = source_config
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": bool(text_config.get("attn_output_gate", True)),
            "bos_token_id": int(text_config.get("bos_token_id") or 1),
            "eos_token_id": int(text_config.get("eos_token_id") or 1),
            "full_attention_interval": 1,
            "head_dim": 4,
            "hidden_act": str(text_config.get("hidden_act", "silu")),
            "hidden_size": 8,
            "initializer_range": float(text_config.get("initializer_range", 0.02)),
            "layer_types": ["full_attention"] * layer_count,
            "max_position_embeddings": 128,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 4,
            "num_attention_heads": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "num_hidden_layers": layer_count,
            "num_key_value_heads": 1,
            "output_router_logits": False,
            "partial_rotary_factor": 1.0,
            "rms_norm_eps": float(text_config.get("rms_norm_eps", 1e-6)),
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "router_aux_loss_coef": float(text_config.get("router_aux_loss_coef", 0.001)),
            "shared_expert_intermediate_size": 4,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 16,
        },
    }


def _sanitize_switch_key_probe(model: qwen3_5_moe.Model) -> bool:
    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros(
            (2, 8, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.mlp.experts.down_proj": mx.zeros(
            (2, 8, 4),
            dtype=mx.bfloat16,
        ),
    }
    sanitized = model.sanitize(weights)
    return all(
        key in sanitized
        for key in (
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
        )
    )


def probe_qwen_upstream_transformer_binding(
    *,
    artifact_dir: str | Path,
    config_path: str | Path,
    expected_layers: tuple[int, ...] | None = None,
) -> dict[str, object]:
    source_config = _load_config(config_path)
    full_args = qwen3_5_moe.ModelArgs.from_dict(source_config)
    full_text_config = source_config.get("text_config")
    if not isinstance(full_text_config, dict):
        full_text_config = source_config
    selected_layers = (0,) if expected_layers is None else tuple(expected_layers)
    layer_count = max(selected_layers, default=0) + 1
    projected_config = _projected_qwen35_moe_config(
        source_config,
        layer_count=max(layer_count, 1),
    )
    projected_args = qwen3_5_moe.ModelArgs.from_dict(projected_config)
    model = qwen3_5_moe.Model(projected_args)

    moe_report = bind_qwen_moe_vq_experts(
        model,
        artifact_dir,
        layers=selected_layers,
    )
    unbound = has_unbound_qwen_moe_vq_experts(model, layers=selected_layers)
    sanitized_switch_keys_present = _sanitize_switch_key_probe(model)
    bound_selected_layers = set(moe_report.bound_layers) == set(selected_layers)
    integration_pass = (
        full_args.model_type == "qwen3_5_moe"
        and bool(getattr(model, "layers", None))
        and bound_selected_layers
        and not moe_report.missing_layers
        and not unbound
        and sanitized_switch_keys_present
    )
    return {
        "record_type": "qwen_upstream_transformer_binding_probe",
        "artifact_dir": str(Path(artifact_dir)),
        "config_path": str(Path(config_path)),
        "upstream_model_module": "mlx_lm.models.qwen3_5_moe",
        "source_model_type": source_config.get("model_type"),
        "source_text_model_type": full_text_config.get("model_type"),
        "full_config_model_args_pass": full_args.model_type == "qwen3_5_moe",
        "full_config_layer_count": int(full_text_config.get("num_hidden_layers") or 0),
        "instantiated_full_config": False,
        "projected_layer_count": max(layer_count, 1),
        "expected_layers": list(selected_layers),
        "bound_layers": list(moe_report.bound_layers),
        "missing_layers": list(moe_report.missing_layers),
        "skipped_layers": list(moe_report.skipped_layers),
        "unbound_vq_experts": unbound,
        "sanitized_switch_keys_present": sanitized_switch_keys_present,
        "integration_pass": integration_pass,
        "notes": [
            "The full Qwen3.6 config is parsed through upstream mlx_lm ModelArgs.",
            "The upstream model instance is a tiny projected surface to avoid dense full-model allocation.",
            "Selected materialized VQ switch layers are bound into upstream mlx_lm switch_mlp slots.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Qwen materialized VQ switch binding against upstream mlx_lm."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--expected-layers")
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    payload = probe_qwen_upstream_transformer_binding(
        artifact_dir=args.artifact_dir,
        config_path=args.config_path,
        expected_layers=_parse_layers(args.expected_layers),
    )
    if args.output_json is not None:
        _write_json(args.output_json, payload)
    if args.append_jsonl is not None:
        _append_jsonl(args.append_jsonl, payload)
    if args.output_json is None and args.append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["integration_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
