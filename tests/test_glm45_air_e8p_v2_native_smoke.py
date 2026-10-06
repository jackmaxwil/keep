from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import e8p_packed_abs_grid
from mlx_vq.kernels import nax


def _write_e8p_projection_artifact(
    root,
    *,
    layer_index: int = 1,
    projection: str = "gate_proj",
    num_experts: int = 4,
    input_dims: int = 704,
    output_dims: int = 73,
    group_size: int = 352,
    seed: int = 20260714,
):
    prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
    rng = np.random.default_rng(seed)
    codes = rng.integers(
        0,
        65536,
        size=(num_experts, output_dims, input_dims // 8),
        dtype=np.uint16,
    )
    scales = rng.uniform(
        0.01,
        0.05,
        size=(num_experts, output_dims, input_dims // group_size),
    ).astype(np.float16)
    shard = root / f"layer-{layer_index:05d}-{projection}.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            f"{prefix}.codes": mx.array(codes),
            f"{prefix}.scales": mx.array(scales),
            "model.vq_codebook.e8": mx.array(e8p_packed_abs_grid()),
        },
    )


def test_expert_kblock_v2_native_smoke_reports_artifact_parity(tmp_path) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    from benchmarks.smoke_glm45_air_e8p_expert_kblock_v2_native import (
        run_expert_kblock_v2_native_smoke,
    )

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_e8p_projection_artifact(artifact)

    report = run_expert_kblock_v2_native_smoke(
        artifact_dir=artifact,
        layer_index=1,
        projections=("gate_proj",),
        tokens=11,
        top_k=3,
        seed=20260714,
        min_cosine=0.99999,
        max_abs_diff=8e-3,
    )

    assert report["record_type"] == "glm45_air_e8p_expert_kblock_v2_native_smoke"
    assert report["decision"] == "parity_pass_not_tensorops_speed"
    assert report["all_v2_checks_pass"] is True
    assert report["speed_claim"] is False
    assert report["checks"][0]["projection"] == "gate_proj"
    assert report["checks"][0]["group_size"] == 352
    assert report["checks"][0]["cosine"] >= 0.99999
