from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def _load_finetune_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "finetune_glm45_air_vq_continuous.py"
    )
    spec = importlib.util.spec_from_file_location("finetune_glm45_air_vq_continuous_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TinyEmbed:
    def __init__(self, weight: np.ndarray) -> None:
        self.weight = mx.array(weight)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.weight[tokens]


class _TinyLinear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        self.weight = mx.array(weight)
        self.bias = None if bias is None else mx.array(bias)

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class _TinyFinalMlp:
    def __init__(self, weight: np.ndarray) -> None:
        self.weight = mx.array(weight)
        self.delta = mx.zeros((weight.shape[0],), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x @ self.weight.T + self.delta)


class _TinyLayer:
    def __init__(self, attention_weight: np.ndarray, mlp) -> None:
        self.input_layernorm = lambda h: h
        self.post_attention_layernorm = lambda h: h
        self._attention = _TinyLinear(attention_weight)
        self.mlp = mlp

    def self_attn(self, h: mx.array, mask, layer_cache) -> mx.array:
        del mask, layer_cache
        return self._attention(h)

    def __call__(self, h: mx.array, mask, layer_cache) -> mx.array:
        h = h + self.self_attn(self.input_layernorm(h), mask, layer_cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class _TinyInnerModel:
    def __init__(self) -> None:
        rng = np.random.default_rng(20260708)
        hidden = 6
        vocab = 11
        self.embed_tokens = _TinyEmbed(rng.normal(scale=0.05, size=(vocab, hidden)).astype(np.float32))
        self.layers = [
            _TinyLayer(
                rng.normal(scale=0.03, size=(hidden, hidden)).astype(np.float32),
                _TinyLinear(rng.normal(scale=0.04, size=(hidden, hidden)).astype(np.float32)),
            ),
            _TinyLayer(
                rng.normal(scale=0.02, size=(hidden, hidden)).astype(np.float32),
                _TinyFinalMlp(rng.normal(scale=0.05, size=(hidden, hidden)).astype(np.float32)),
            ),
        ]
        self.norm = lambda h: h * mx.array(0.75, dtype=mx.float32)


class _TinyModel:
    def __init__(self) -> None:
        rng = np.random.default_rng(20260709)
        self.model = _TinyInnerModel()
        self.layers = self.model.layers
        self.lm_head = _TinyLinear(rng.normal(scale=0.06, size=(13, 6)).astype(np.float32))


def run_prefix_cache_equivalence_probe() -> list[dict[str, float]]:
    finetune = _load_finetune_module()
    assert hasattr(finetune, "_cache_final_layer_training_row")

    model = _TinyModel()
    row = finetune.PreparedTeacherRow(
        row_index=0,
        prompt_id="tiny-prefix-cache",
        input_token_ids=(1, 4, 7, 2),
        positions=(1, 3),
        target_token_ids=(5, 8),
        teacher_logits=mx.array(
            np.random.default_rng(20260710).normal(scale=0.1, size=(2, 13)).astype(np.float32)
        ),
    )
    cached_row = finetune._cache_final_layer_training_row(model, row, layer=1)

    ref_params = {"delta": mx.array([0.02, -0.01, 0.03, 0.0, -0.02, 0.01], dtype=mx.float32)}
    cached_params = {"delta": ref_params["delta"]}
    results: list[dict[str, float]] = []
    for step in range(2):

        def ref_loss(values):
            model.model.layers[1].mlp.delta = values["delta"]
            return finetune._kl_loss(
                model,
                row,
                target_nll_weight=0.2,
                tail_kld_weight=0.1,
                aux_loss_position_indices=(0,),
                loss_scope="final_layer_selected",
                layer=1,
            )

        def cached_loss(values):
            model.model.layers[1].mlp.delta = values["delta"]
            return finetune._kl_loss(
                model,
                row,
                target_nll_weight=0.2,
                tail_kld_weight=0.1,
                aux_loss_position_indices=(0,),
                loss_scope="final_layer_selected",
                layer=1,
                final_layer_prefix_cache=cached_row,
                teacher_log_probs=cached_row.teacher_log_probs,
                teacher_probs=cached_row.teacher_probs,
            )

        ref_loss_value, ref_grads = mx.value_and_grad(ref_loss)(ref_params)
        cached_loss_value, cached_grads = mx.value_and_grad(cached_loss)(cached_params)
        mx.eval(ref_loss_value, cached_loss_value, ref_grads, cached_grads)

        ref_loss_float = float(ref_loss_value.item())
        cached_loss_float = float(cached_loss_value.item())
        grad_delta = float(mx.max(mx.abs(ref_grads["delta"] - cached_grads["delta"])).item())
        results.append(
            {
                "step": float(step),
                "ref_loss": ref_loss_float,
                "cached_loss": cached_loss_float,
                "loss_delta": abs(ref_loss_float - cached_loss_float),
                "grad_delta": grad_delta,
            }
        )
        assert np.isclose(cached_loss_float, ref_loss_float, rtol=1.0e-4, atol=1.0e-5)
        assert grad_delta <= 1.0e-5

        ref_params = {"delta": ref_params["delta"] - 0.05 * ref_grads["delta"]}
        cached_params = {"delta": cached_params["delta"] - 0.05 * cached_grads["delta"]}
        mx.eval(ref_params, cached_params)

    np.testing.assert_allclose(
        np.array(cached_params["delta"]),
        np.array(ref_params["delta"]),
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    return results


def test_final_layer_prefix_cache_matches_uncached_loss_for_two_update_steps() -> None:
    run_prefix_cache_equivalence_probe()
