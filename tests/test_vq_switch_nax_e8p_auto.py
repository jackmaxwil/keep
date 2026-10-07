from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import e8_1bit_packed, e8p_packed_abs_grid
from mlx_vq.kernels import nax
import mlx_vq.ops.vq_switch as vq_switch


def test_auto_nax_e8p_selector_pins_every_kernel_boundary() -> None:
    selector = getattr(vq_switch, "auto_selects_nax_e8p", None)
    assert selector is not None, "production auto dispatch needs an E8P NAX eligibility predicate"

    base = dict(
        route_count=1,
        implementation="metal",
        code_bits=16,
        input_dims=8,
        output_dims=1,
        group_size=8,
        activation_dtype=mx.float16,
    )
    assert selector(**base)
    assert selector(**{**base, "activation_dtype": mx.bfloat16})

    rejected = (
        {"route_count": 0},
        {"implementation": "reference"},
        {"code_bits": 8},
        {"input_dims": 0},
        {"input_dims": 7},
        {"output_dims": 0},
        {"group_size": 0},
        {"group_size": 7},
        {"group_size": 16},
        {"activation_dtype": mx.float32},
    )
    for change in rejected:
        assert not selector(**{**base, **change}), change


def _inputs(*, code_bits: int = 16, dtype: mx.Dtype = mx.float16):
    code_dtype = mx.uint16 if code_bits == 16 else mx.uint8
    codebook = e8p_packed_abs_grid() if code_bits == 16 else e8_1bit_packed()
    return (
        mx.ones((1, 8), dtype=dtype),
        mx.zeros((1, 1, 1), dtype=code_dtype),
        mx.ones((1, 1, 1), dtype=mx.float16),
        mx.array(codebook),
        mx.zeros((1, 1), dtype=mx.int32),
    )


def _gather(*, code_bits: int = 16, dtype: mx.Dtype = mx.float16, route_strategy="auto"):
    x, codes, scales, codebook, rhs = _inputs(code_bits=code_bits, dtype=dtype)
    return vq_switch.gather_vqmm(
        x,
        codes,
        scales,
        codebook,
        rhs,
        input_dims=8,
        output_dims=1,
        group_size=8,
        code_bits=code_bits,
        route_strategy=route_strategy,
    )


def test_eligible_auto_warns_and_keeps_scalar_fallback_without_native(monkeypatch) -> None:
    monkeypatch.setattr(vq_switch.nax, "is_available", lambda: False)

    with pytest.warns(
        RuntimeWarning,
        match="native VQ NAX extension is unavailable.*E8P fast path is not being used",
    ):
        actual = _gather()
    mx.eval(actual)

    expected = _gather(route_strategy="direct")
    mx.eval(expected)
    assert np.array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize(
    ("code_bits", "dtype", "route_strategy"),
    [
        (8, mx.float16, "auto"),
        (16, mx.float32, "auto"),
        (16, mx.float16, "direct"),
    ],
)
def test_ineligible_calls_do_not_warn_without_native(
    monkeypatch, code_bits: int, dtype: mx.Dtype, route_strategy: str
) -> None:
    monkeypatch.setattr(vq_switch.nax, "is_available", lambda: False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = _gather(code_bits=code_bits, dtype=dtype, route_strategy=route_strategy)
        mx.eval(actual)


@pytest.mark.parametrize(
    ("input_dims", "output_dims"),
    [(4_096, 2_048), (2_048, 4_096)],
    ids=["dsv4_gate_up", "dsv4_down"],
)
def test_auto_uses_real_native_m32n64_and_stays_close_to_prior_scalar(
    monkeypatch, input_dims: int, output_dims: int
) -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    rng = np.random.default_rng(20260819)
    experts, tokens, top_k = 2, 1, 6
    group_size = 512
    x = mx.array(rng.normal(size=(tokens, input_dims)).astype(np.float16))
    codes = mx.array(
        rng.integers(
            0,
            1 << 16,
            size=(experts, output_dims, input_dims // 8),
            dtype=np.uint16,
        )
    )
    scales = mx.array(
        rng.uniform(
            0.01,
            0.05,
            size=(experts, output_dims, input_dims // group_size),
        ).astype(np.float16)
    )
    codebook = mx.array(e8p_packed_abs_grid())
    rhs = mx.array([[0, 1, 0, 1, 0, 1]], dtype=mx.int32)
    mx.eval(x, codes, scales, codebook, rhs)

    real_m32n64 = vq_switch.nax.nax_e8p_fp16_sorted_steel_m32n64_matmul
    calls = 0

    def observed_m32n64(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_m32n64(*args, **kwargs)

    monkeypatch.setattr(
        vq_switch.nax,
        "nax_e8p_fp16_sorted_steel_m32n64_matmul",
        observed_m32n64,
    )
    kwargs = dict(
        input_dims=input_dims,
        output_dims=output_dims,
        group_size=group_size,
        code_bits=16,
    )
    actual = vq_switch.gather_vqmm(
        x, codes, scales, codebook, rhs, route_strategy="auto", **kwargs
    )
    expected = vq_switch.gather_vqmm(
        x, codes, scales, codebook, rhs, route_strategy="direct", **kwargs
    )
    mx.eval(actual, expected)

    actual_np = np.array(actual, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)
    delta = actual_np.astype(np.float64) - expected_np.astype(np.float64)
    relative_rms = float(
        np.sqrt(np.mean(delta**2))
        / np.sqrt(np.mean(expected_np.astype(np.float64) ** 2))
    )
    assert calls == 1
    # Measured on both production DSV4 projection shapes: max_abs=0.00390625
    # and relative RMS <=3.4e-4. TensorOps reassociation is not byte-exact;
    # 0.004 is the narrowest decimal bound above the observed float16 delta.
    assert float(np.max(np.abs(delta))) <= 0.004
    assert relative_rms <= 5e-4


def test_auto_native_m32n64_honors_custom_e8p_codebook() -> None:
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")

    custom_codebook = np.roll(e8p_packed_abs_grid(), 1)
    x = mx.ones((1, 8), dtype=mx.float16)
    codes = mx.zeros((1, 1, 1), dtype=mx.uint16)
    scales = mx.ones((1, 1, 1), dtype=mx.float16)
    codebook = mx.array(custom_codebook)
    rhs = mx.zeros((1, 1), dtype=mx.int32)
    kwargs = dict(input_dims=8, output_dims=1, group_size=8, code_bits=16)

    actual = vq_switch.gather_vqmm(
        x, codes, scales, codebook, rhs, route_strategy="auto", **kwargs
    )
    expected = vq_switch.gather_vqmm(
        x, codes, scales, codebook, rhs, route_strategy="direct", **kwargs
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=5e-4, atol=0.004)
