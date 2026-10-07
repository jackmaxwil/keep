from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lm.models.switch_layers import SwitchLinear

from mlx_vq.benchmark.quant_compare import quantized_parameter_bytes
from mlx_vq.convert.stream_convert import SafetensorsIndex
from mlx_vq.io.source_safetensors import read_indexed_safetensors_tensor_mlx

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _expert_weight_name(*, layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def _stack_expert_projection(
    *,
    source_dir: str | Path,
    index: SafetensorsIndex,
    layer: int,
    projection: str,
    num_experts: int,
) -> mx.array:
    weights = [
        read_indexed_safetensors_tensor_mlx(
            source_dir,
            index,
            _expert_weight_name(layer=layer, expert=expert, projection=projection),
        ).astype(mx.bfloat16)
        for expert in range(num_experts)
    ]
    stacked = mx.stack(weights, axis=0)
    mx.eval(stacked)
    return stacked


def _quantize_switch_projection(
    weight: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str,
):
    num_experts, output_dims, input_dims = weight.shape
    dense = SwitchLinear(input_dims, output_dims, num_experts, bias=False)
    dense.weight = weight
    quantized = dense.to_quantized(group_size=group_size, bits=bits, mode=mode)
    mx.eval(quantized.parameters())
    return quantized


def _save_quantized_projection(
    *,
    output_dir: Path,
    layer: int,
    projection: str,
    quantized,
    group_size: int,
    bits: int,
    mode: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
    arrays = {
        f"{prefix}.weight": quantized.weight,
        f"{prefix}.scales": quantized.scales,
    }
    biases = quantized.get("biases")
    if biases is not None:
        arrays[f"{prefix}.biases"] = biases
    bias = quantized.get("bias")
    if bias is not None:
        arrays[f"{prefix}.bias"] = bias
    output_path = output_dir / f"layer-{layer:05d}-{projection}.safetensors"
    mx.save_safetensors(
        str(output_path),
        arrays,
        metadata={
            "mlx_quantization_config": json.dumps(
                {
                    "group_size": int(group_size),
                    "bits": int(bits),
                    "mode": mode,
                },
                sort_keys=True,
            )
        },
    )
    return output_path


def convert_mlx_routed_quant_layer(
    *,
    source_dir: str | Path,
    index: SafetensorsIndex,
    output_dir: str | Path,
    layer: int,
    num_experts: int,
    group_size: int = 128,
    bits: int = 2,
    mode: str = "affine",
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    output_root = Path(output_dir)
    for projection in PROJECTIONS:
        output_path = output_root / f"layer-{layer:05d}-{projection}.safetensors"
        if skip_existing and output_path.exists():
            records.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "status": "existing",
                    "output_path": str(output_path),
                    "source_tensors_read": 0,
                    "group_size": group_size,
                    "bits": bits,
                    "mode": mode,
                }
            )
            continue
        weight = _stack_expert_projection(
            source_dir=source_dir,
            index=index,
            layer=layer,
            projection=projection,
            num_experts=num_experts,
        )
        quantized = _quantize_switch_projection(
            weight,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        output_path = _save_quantized_projection(
            output_dir=output_root,
            layer=layer,
            projection=projection,
            quantized=quantized,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        byte_counts = quantized_parameter_bytes(dict(quantized.parameters()))
        records.append(
            {
                "layer": layer,
                "projection": projection,
                "status": "converted",
                "output_path": str(output_path),
                "source_tensors_read": num_experts,
                "source_shape": list(weight.shape),
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
                **byte_counts,
            }
        )
    return records


def write_mlx_routed_quant_manifest(
    *,
    output_dir: str | Path,
    model_id: str,
    records: list[dict[str, Any]],
    group_size: int,
    bits: int,
    mode: str,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    manifest = {
        "model_id": model_id,
        "output_dir": str(output_root),
        "quantized_scope": "routed_experts",
        "quant_family": "mlx_affine",
        "group_size": int(group_size),
        "bits": int(bits),
        "mode": mode,
        "projection_records": records,
        "converted_projection_groups": sum(1 for record in records if record["status"] == "converted"),
        "existing_projection_groups": sum(1 for record in records if record["status"] == "existing"),
        "total_projection_groups": len(records),
        "code_bytes": sum(int(record.get("code_bytes", 0)) for record in records),
        "scale_bytes": sum(int(record.get("scale_bytes", 0)) for record in records),
        "bias_bytes": sum(int(record.get("bias_bytes", 0)) for record in records),
    }
    manifest["total_quantized_bytes"] = (
        int(manifest["code_bytes"]) + int(manifest["scale_bytes"]) + int(manifest["bias_bytes"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
