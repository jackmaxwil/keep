from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import cosine_similarity, decode_weight_matrix, e8_1bit_packed
from mlx_vq.nn.linear import QuantizedVQLinear


def _build_tiny_model(seed: int = 20260623):
    rng = np.random.default_rng(seed)
    vocab_size = 32
    hidden = 64
    out_dim = 48
    group_size = 32
    embeddings = rng.normal(scale=0.1, size=(vocab_size, hidden)).astype(np.float32)
    codes = rng.integers(0, 256, size=(out_dim, hidden // 8), dtype=np.uint8)
    scales = rng.uniform(0.85, 1.15, size=(out_dim, hidden // group_size)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(out_dim,)).astype(np.float32)
    lm_head = rng.normal(scale=0.1, size=(vocab_size, out_dim)).astype(np.float32)
    layer = QuantizedVQLinear(
        input_dims=hidden,
        output_dims=out_dim,
        codes=mx.array(codes),
        scales=mx.array(scales),
        codebook=mx.array(e8_1bit_packed()),
        bias=mx.array(bias),
        group_size=group_size,
        code_bits=8,
    )
    return embeddings, codes, scales, bias, lm_head, layer


def test_tiny_generation_uses_quantized_vq_linear_with_logit_cosine() -> None:
    embeddings, codes, scales, bias, lm_head, layer = _build_tiny_model()
    dense_weight = decode_weight_matrix(codes, scales, codebook=e8_1bit_packed())

    def vq_logits(token: int) -> mx.array:
        hidden = mx.array(embeddings[token])
        projected = mx.tanh(layer(hidden))
        return projected @ mx.array(lm_head.T)

    def reference_logits(token: int) -> np.ndarray:
        projected = np.tanh(dense_weight @ embeddings[token] + bias)
        return projected @ lm_head.T

    prompt = [1, 7, 13]
    for _ in range(4):
        logits = vq_logits(prompt[-1])
        mx.eval(logits)
        prompt.append(int(np.array(mx.argmax(logits))))

    assert len(prompt) == 7
    assert all(0 <= token < embeddings.shape[0] for token in prompt)

    actual = np.array(vq_logits(13))
    expected = reference_logits(13)
    assert cosine_similarity(actual, expected) >= 0.99999
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
