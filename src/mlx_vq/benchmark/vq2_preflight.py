from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import e8p_packed_abs_grid, encode_e8p_rtn
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.convert.stream_convert import (
    _read_tensor_from_shard,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from mlx_vq.kernels.vq_qmm import vq_qmm
from mlx_vq.kernels.vq_qmv import vq_qmv
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear
from mlx_vq.ops.vq_switch import gather_vqmm
from mlx_vq.quant.rtn import quantize_weight_rtn


def _finite(value: mx.array) -> bool:
    mx.eval(value)
    return bool(mx.all(mx.isfinite(value.astype(mx.float32))).item())


def _run_real_source_smoke(
    *,
    source_dir: str | Path,
    index_path: str | Path,
    config: Mapping[str, Any],
    model_id: str,
    layer: int | None = None,
    projection: str = "gate_proj",
    expert: int = 0,
) -> dict[str, Any]:
    source_root = Path(source_dir)
    index = load_safetensors_index(index_path)
    plan = plan_streaming_conversion_from_index(
        dict(config),
        index,
        model_id=model_id,
        code_bits=16,
    )
    group = next(
        (
            candidate
            for candidate in plan.vq_groups
            if candidate.projection == projection and (layer is None or candidate.layer == layer)
        ),
        None,
    )
    if group is None:
        target = f"layer {layer} " if layer is not None else ""
        raise ValueError(f"no VQ-2 group found for {target}{projection}")
    if expert not in group.experts:
        raise ValueError(f"expert {expert} is not present in layer {group.layer} {group.projection}")

    tensor_name = f"model.layers.{group.layer}.mlp.experts.{expert}.{group.projection}.weight"
    shard_name = index.weight_map[tensor_name]
    weight = _read_tensor_from_shard(source_root, shard_name, tensor_name)

    start = time.perf_counter()
    quantized = quantize_weight_rtn(weight, group_size=group.group_size, code_bits=16)
    elapsed = time.perf_counter() - start
    vector_count = int(weight.size // 8)
    vectors_per_second = vector_count / elapsed if elapsed > 0 else None
    total_vectors = sum(
        planned.code_bytes // (planned.code_bits // 8)
        for planned in plan.vq_groups
        if planned.code_bits == 16
    )
    estimated_seconds = None if vectors_per_second in (None, 0) else total_vectors / vectors_per_second

    return {
        "ok": True,
        "scope": "single_real_source_tensor_no_artifact_written",
        "estimate_scope": "encoder_only_linear_extrapolation_from_single_tensor",
        "model_id": model_id,
        "source_dir": str(source_root),
        "source_tensor": tensor_name,
        "source_shard": shard_name,
        "source_shape": list(weight.shape),
        "source_bytes_float32_buffer": int(weight.nbytes),
        "layer": int(group.layer),
        "projection": group.projection,
        "expert": int(expert),
        "group_size": int(group.group_size),
        "code_bits": int(group.code_bits),
        "vectors": vector_count,
        "seconds": elapsed,
        "vectors_per_second": vectors_per_second,
        "codes_shape": list(quantized.codes.shape),
        "scales_shape": list(quantized.scales.shape),
        "codes_dtype": str(quantized.codes.dtype),
        "scales_dtype": str(quantized.scales.dtype),
        "planned_vq_groups": len(plan.vq_groups),
        "planned_vq_code_bytes": int(plan.vq_code_bytes),
        "planned_vq_scale_bytes": int(plan.vq_scale_bytes),
        "planned_vq_total_bytes": int(plan.vq_code_bytes + plan.vq_scale_bytes),
        "linear_estimated_full_routed_encoder_seconds": estimated_seconds,
        "linear_estimated_full_routed_encoder_hours": None if estimated_seconds is None else estimated_seconds / 3600,
    }


def run_vq2_preflight(
    *,
    source_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    model_id: str = GLM45_AIR_MODEL_ID,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260624)
    codebook = mx.array(e8p_packed_abs_grid())
    codes = mx.array(rng.integers(0, 65536, size=(4, 2), dtype=np.uint16))
    scales = mx.array(rng.uniform(0.1, 0.5, size=(4, 2)).astype(np.float16))
    x_vec = mx.array(rng.normal(size=(16,)).astype(np.float16))
    x_mat = mx.array(rng.normal(size=(3, 16)).astype(np.float16))

    encode_ok = False
    qmv_ok = False
    qmm_ok = False
    gather_ok = False
    switch_ok = False
    artifact_rtn_ok = False
    real_source_smoke: dict[str, Any] | str = "skipped"
    errors: dict[str, str] = {}

    try:
        vectors = rng.normal(size=(4, 8)).astype(np.float32)
        encoded = encode_e8p_rtn(vectors, chunk_size=4096)
        encode_ok = encoded.dtype == np.uint16 and encoded.shape == (4,)
    except Exception as exc:  # noqa: BLE001 - diagnostic preflight captures exact failure.
        errors["encode_e8p_rtn"] = str(exc)

    try:
        qmv_ok = _finite(
            vq_qmv(
                x_vec,
                codes,
                scales,
                codebook,
                in_dim=16,
                out_dim=4,
                group_size=8,
                code_bits=16,
            )
        )
    except Exception as exc:  # noqa: BLE001
        errors["qmv_uint16"] = str(exc)

    try:
        qmm_ok = _finite(
            vq_qmm(
                x_mat,
                codes,
                scales,
                codebook,
                in_dim=16,
                out_dim=4,
                group_size=8,
                code_bits=16,
            )
        )
    except Exception as exc:  # noqa: BLE001
        errors["qmm_uint16"] = str(exc)

    try:
        switch_codes = mx.array(rng.integers(0, 65536, size=(3, 4, 2), dtype=np.uint16))
        switch_scales = mx.array(rng.uniform(0.1, 0.5, size=(3, 4, 2)).astype(np.float16))
        indices = mx.array(np.array([[0, 1], [2, 0]], dtype=np.int32))
        x_tokens = mx.array(rng.normal(size=(2, 16)).astype(np.float16))
        gather_ok = _finite(
            gather_vqmm(
                x_tokens,
                switch_codes,
                switch_scales,
                codebook,
                indices,
                input_dims=16,
                output_dims=4,
                group_size=8,
                code_bits=16,
            )
        )
        switch = QuantizedVQSwitchLinear(
            input_dims=16,
            output_dims=4,
            num_experts=3,
            codes=switch_codes,
            scales=switch_scales,
            codebook=codebook,
            group_size=8,
            code_bits=16,
        )
        switch_ok = _finite(switch(x_tokens, indices))
    except Exception as exc:  # noqa: BLE001
        errors["switch_or_gather_uint16"] = str(exc)

    try:
        quantize_weight_rtn(rng.normal(size=(4, 16)).astype(np.float32), group_size=8, code_bits=16)  # type: ignore[arg-type]
        artifact_rtn_ok = True
    except Exception as exc:  # noqa: BLE001
        errors["artifact_rtn_code_bits_16"] = str(exc)

    if source_dir is not None or index_path is not None or config is not None:
        if source_dir is None or index_path is None or config is None:
            errors["real_source_smoke"] = "source_dir, index_path, and config are all required"
            real_source_smoke = "skipped_missing_inputs"
        else:
            try:
                real_source_smoke = _run_real_source_smoke(
                    source_dir=source_dir,
                    index_path=index_path,
                    config=config,
                    model_id=model_id,
                )
            except Exception as exc:  # noqa: BLE001
                errors["real_source_smoke"] = str(exc)
                real_source_smoke = {"ok": False, "error": str(exc)}

    return {
        "encode_e8p_rtn": encode_ok,
        "qmv_uint16": qmv_ok,
        "qmm_uint16": qmm_ok,
        "gather_uint16": gather_ok,
        "switch_uint16": switch_ok,
        "artifact_rtn_code_bits_16": artifact_rtn_ok,
        "artifact_ready": all([encode_ok, qmv_ok, qmm_ok, gather_ok, switch_ok, artifact_rtn_ok]),
        "real_source_smoke": real_source_smoke,
        "errors": errors,
    }
