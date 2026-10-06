from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import e8p_packed_abs_grid
from mlx_vq.kernels import nax
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear


def _write_projection(
    root: Path,
    *,
    layer_index: int,
    projection: str,
    weight_shape: tuple[int, int, int],
    group_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    layer = QuantizedVQSwitchLinear.from_weights(
        mx.array(rng.normal(scale=0.03, size=weight_shape).astype(np.float32)),
        group_size=group_size,
        code_bits=16,
    )
    prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
    mx.save_safetensors(
        str(root / f"layer-{layer_index:05d}-{projection}.safetensors"),
        {
            f"{prefix}.codes": layer.codes,
            f"{prefix}.scales": layer.scales,
            "model.vq_codebook.e8": mx.array(e8p_packed_abs_grid()),
        },
    )


def _write_e8p_switch_glu_artifact(root: Path, *, layer_index: int = 1) -> None:
    num_experts = 4
    input_dims = 16
    hidden_dims = 8
    _write_projection(
        root,
        layer_index=layer_index,
        projection="gate_proj",
        weight_shape=(num_experts, hidden_dims, input_dims),
        group_size=8,
        seed=20260706,
    )
    _write_projection(
        root,
        layer_index=layer_index,
        projection="up_proj",
        weight_shape=(num_experts, hidden_dims, input_dims),
        group_size=8,
        seed=20260707,
    )
    _write_projection(
        root,
        layer_index=layer_index,
        projection="down_proj",
        weight_shape=(num_experts, input_dims, hidden_dims),
        group_size=8,
        seed=20260708,
    )


def test_sorted_prefill_e8p_dispatch_report_hits_native_and_matches_metal(tmp_path: Path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    from benchmarks.prove_glm45_air_e8p_sorted_prefill_dispatch import (
        run_sorted_prefill_dispatch_parity,
    )

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_switch_glu_artifact(artifact)

    report = run_sorted_prefill_dispatch_parity(
        artifact_dir=artifact,
        layer_index=1,
        tokens=32,
        top_k=2,
        seed=20260706,
        min_cosine=0.99998,
        max_abs_diff=2e-5,
    )

    assert report["record_type"] == "glm45_air_e8p_sorted_prefill_dispatch_parity"
    assert report["decision"] == "sorted_prefill_e8p_dispatch_parity_pass"
    assert report["passes_routed_forward_parity"] is True
    assert report["resident_auto_claim"] is True
    assert report["speed_claim"] is False
    assert report["peer2_used"] is False
    assert report["rdma_jaccl_touched"] is False
    assert report["can_use_shared_sorted_prefill"] is True
    assert report["selected_implementation"] == "nax_e8p_m32n64"
    assert report["call_implementations"] == ["nax_e8p_m32n64"] * 3
    assert report["call_projections"] == ["gate_up", "gate_up", "down"]
    assert report["route_count"] == 64
    assert report["output_shape"] == [32, 2, 16]
    assert report["cosine"] >= 0.99998
    assert report["max_abs_diff"] <= 2e-5
