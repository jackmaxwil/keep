"""GLM-5.3-Flash MLX adapter against the pinned transformers reference.

``tests/fixtures/glm53_reference_tiny.npz`` holds a tiny random GLM-5.3 text
model under its checkpoint tensor names plus the outputs of transformers
v5.16.1's ``Glm5NextTextModel`` on it, in float32, written by
``scripts/make_glm53_reference_fixtures.py``. The tiny config keeps every
structural feature: KDA and DSA layers, a dense layer then MoE, 4-stream mHC,
NoPE MLA and a k-pool indexer whose top-k is shorter than the sequence.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from ramp.models.deepseek_v4_flash_adapter import LimitedSwiGLU
from ramp.models.glm5_next_adapter import (
    Glm5NextMoE,
    Glm5NextVQModel,
    bind_glm5_next_non_vq_weights,
    glm5_next_args_from_config,
    has_unbound_glm5_next_vq_experts,
)

FIXTURE = np.load(Path(__file__).resolve().parent / "fixtures/glm53_reference_tiny.npz")
CONFIG = json.loads(FIXTURE["config_json"].tobytes())
META = json.loads(FIXTURE["meta_json"].tobytes())
WEIGHTS = {k[2:]: FIXTURE[k] for k in FIXTURE.files if k.startswith("w/")}
IDS = mx.array(FIXTURE["input_ids"])
TOL = 1e-4


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _dense_experts(layer: int, args) -> SwitchGLU:
    switch = SwitchGLU(
        args.hidden_size, args.moe_intermediate_size, args.n_routed_experts, activation=LimitedSwiGLU(args.swiglu_limit)
    )
    stem = f"model.language_model.layers.{layer}.mlp.experts"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = np.stack([WEIGHTS[f"{stem}.{e}.{proj}.weight"] for e in range(args.n_routed_experts)])
        getattr(switch, proj).weight = mx.array(stacked)
    return switch


def _model() -> Glm5NextVQModel:
    args = glm5_next_args_from_config(CONFIG)
    model = Glm5NextVQModel(args)
    report = bind_glm5_next_non_vq_weights(model, {k: mx.array(v) for k, v in WEIGHTS.items()})
    assert report["routed_skipped"] == 3 * 8 * 3
    assert has_unbound_glm5_next_vq_experts(model)
    for i, layer in enumerate(model.layers):
        if isinstance(layer.mlp, Glm5NextMoE):
            layer.mlp.bind_switch_mlp(_dense_experts(i, args))
    assert not has_unbound_glm5_next_vq_experts(model)
    return model


def test_fixture_is_the_pinned_reference():
    assert META["transformers"] == "5.16.1"
    assert META["dtype"] == "float32"
    assert CONFIG["layer_types"] == ["linear_attention", "deepseek_sparse_attention"] * 2


def test_full_forward_matches_reference_layer_by_layer():
    model = _model()
    hidden: list = []
    last = model.model(IDS, hidden_states=hidden)
    for i, h in enumerate(hidden):
        assert _rel(h, FIXTURE[f"full_hidden_{i}"]) < TOL, f"input to layer {i}"
    assert _rel(last, FIXTURE["full_last_hidden"]) < TOL
    logits = model.lm_head(last)
    assert _rel(logits, FIXTURE["full_logits"]) < TOL


def test_cached_prefill_then_decode_matches_reference():
    model = _model()
    prefill, decode = META["prefill"], META["decode"]
    cache = model.make_cache()
    steps = [model.model(IDS[:, :prefill], cache)[:, -1]]
    for t in range(prefill, prefill + decode - 1):
        steps.append(model.model(IDS[:, t : t + 1], cache)[:, -1])
    assert _rel(mx.stack(steps, axis=1), FIXTURE["incremental_last_hidden"]) < TOL


def test_fixture_exercises_sparse_selection():
    """With selection disabled the model must miss the reference, or the
    parity above would not be evidence about the indexer."""

    model = _model()
    for layer in model.layers:
        if not layer.is_linear:
            layer.self_attn.indexer.topk = 10_000
    assert _rel(model.model(IDS), FIXTURE["full_last_hidden"]) > 1e-2


def test_unbound_experts_refuse_to_run():
    model = Glm5NextVQModel(glm5_next_args_from_config(CONFIG))
    bind_glm5_next_non_vq_weights(model, {k: mx.array(v) for k, v in WEIGHTS.items()})
    with pytest.raises(RuntimeError, match="routed experts must be bound"):
        model(IDS)


def test_plain_swiglu_experts_are_rejected():
    args = glm5_next_args_from_config(CONFIG)
    moe = Glm5NextMoE(args)
    with pytest.raises(ValueError, match="LimitedSwiGLU"):
        moe.bind_switch_mlp(SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts))


def test_binder_skips_vision_and_mtp_and_rejects_undecoded_fp8():
    model = Glm5NextVQModel(glm5_next_args_from_config(CONFIG))
    extra = {
        "model.visual.blocks.0.attn.qkv.weight": mx.zeros((3, 3)),
        "model.language_model.layers.4.eh_proj.weight": mx.zeros((3, 3)),
    }
    report = bind_glm5_next_non_vq_weights(model, {**{k: mx.array(v) for k, v in WEIGHTS.items()}, **extra})
    assert report["vision_skipped"] == 1 and report["mtp_skipped"] == 1
    stray = {"model.language_model.layers.1.self_attn.o_proj.weight_scale_inv": mx.zeros((1, 1))}
    with pytest.raises(ValueError, match="must be decoded"):
        bind_glm5_next_non_vq_weights(model, stray)


def test_binder_is_strict_about_missing_tensors():
    model = Glm5NextVQModel(glm5_next_args_from_config(CONFIG))
    partial = {k: mx.array(v) for k, v in WEIGHTS.items() if not k.endswith("layers.2.self_attn.A_log")}
    with pytest.raises(ValueError):
        bind_glm5_next_non_vq_weights(model, partial)
