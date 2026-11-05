"""Opt-in M=1 kernels for 16-bit E8P codes on GPUs without the NAX kernels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import ramp.nn.switch_linear as switch_linear  # noqa: E402
from ramp.kernels import e8p_m1  # noqa: E402
from ramp.nn.switch_linear import QuantizedVQSwitchLinear  # noqa: E402


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return bool(metal is not None and metal.is_available())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"code_bits": 8}, False),
        ({"tokens": 2}, False),
        ({"sorted_indices": True}, False),
        ({"route_strategy": "sorted_tiled"}, False),
        ({"route_strategy": "direct"}, True),
    ],
)
def test_decision_table_with_the_flag_on(monkeypatch, kwargs, expected):
    monkeypatch.setenv(e8p_m1.E8P_M1_ENV, "1")
    args = {"code_bits": 16, "tokens": 1, "sorted_indices": False, "route_strategy": "auto", **kwargs}
    assert e8p_m1.should_use_e8p_m1(**args) is expected


def test_flag_is_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv(e8p_m1.E8P_M1_ENV, raising=False)
    assert e8p_m1.should_use_e8p_m1(code_bits=16, tokens=1, sorted_indices=False, route_strategy="auto") is False
    monkeypatch.setenv(e8p_m1.E8P_M1_ENV, "maybe")
    with pytest.raises(ValueError, match=e8p_m1.E8P_M1_ENV):
        e8p_m1.e8p_m1_enabled()


def test_per_route_source_only_offsets_the_activation_reads():
    original = (Path(e8p_m1.__file__).parent / "gather_vqmm_m1.metal").read_text()
    rewritten = e8p_m1.per_route_x_source()
    assert rewritten.count("route * (codewords * 8u)") == 2
    # Nothing else changed: undoing the two rewrites restores the file exactly.
    restored = rewritten
    for old, new in e8p_m1._X_INDEX_REWRITES:
        restored = restored.replace(new, old)
    assert restored == original


def _layer(code_bits=16, experts=4, out=32, inp=64, seed=0):
    rng = np.random.default_rng(seed)
    weight = mx.array(rng.normal(size=(experts, out, inp)).astype(np.float32))
    return QuantizedVQSwitchLinear.from_weights(weight, group_size=32, code_bits=code_bits)


def test_switch_linear_routes_m1_16bit_calls_to_the_new_kernels(monkeypatch):
    calls = []

    def fake_token(x, codes, scales, codebook, rhs, *, output_dims):
        calls.append(("token", tuple(x.shape), tuple(rhs.shape)))
        return mx.zeros((1, rhs.shape[1], output_dims))

    def fake_route(x, codes, scales, codebook, rhs, *, output_dims):
        calls.append(("route", tuple(x.shape), tuple(rhs.shape)))
        return mx.zeros((rhs.size, output_dims))

    monkeypatch.setattr(switch_linear, "e8p_m1_token_layout", fake_token)
    monkeypatch.setattr(switch_linear, "e8p_m1_per_route", fake_route)
    monkeypatch.setenv(e8p_m1.E8P_M1_ENV, "1")
    layer = _layer()
    indices = mx.array([[0, 2, 3]], dtype=mx.uint32)
    y = layer(mx.zeros((1, 64)), indices)  # gate/up: one row, three experts
    assert y.shape == (1, 3, 32)
    y = layer(mx.zeros((1, 3, 64)), indices)  # down: one row per route
    assert y.shape == (1, 3, 32)
    assert calls == [("token", (1, 64), (1, 3)), ("route", (3, 64), (3,))]


def test_switch_linear_leaves_8bit_and_flag_off_calls_alone(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("opt-in kernel used without the flag or for 8-bit codes")

    monkeypatch.setattr(switch_linear, "e8p_m1_token_layout", boom)
    monkeypatch.setattr(switch_linear, "e8p_m1_per_route", boom)
    monkeypatch.setattr(switch_linear, "gather_vqmm", lambda *a, **k: mx.zeros((3, 32)))
    monkeypatch.setattr(switch_linear, "gather_vqmm_m1_kernel", lambda *a, **k: mx.zeros((1, 3, 32)))
    indices = mx.array([[0, 2, 3]], dtype=mx.uint32)
    monkeypatch.delenv(e8p_m1.E8P_M1_ENV, raising=False)
    _layer(code_bits=16)(mx.zeros((1, 64)), indices)
    monkeypatch.setenv(e8p_m1.E8P_M1_ENV, "1")
    _layer(code_bits=8)(mx.zeros((1, 64)), indices)


def _dense(layer) -> np.ndarray:
    from keep.vq.e8 import decode_weight_matrix

    codes = np.array(layer.codes)
    scales = np.array(layer.effective_scales().astype(mx.float32))
    return np.stack(
        [decode_weight_matrix(codes[e], scales[e], code_bits=16) for e in range(codes.shape[0])]
    )


@pytest.mark.skipif(not _metal_available(), reason="needs a Metal GPU")
@pytest.mark.parametrize("rows_per_threadgroup", [8, 32])
def test_token_and_per_route_kernels_match_the_dense_decode(rows_per_threadgroup):
    layer = _layer(experts=6, out=96, inp=128, seed=3)
    dense = _dense(layer)
    rng = np.random.default_rng(4)
    rhs = mx.array([[5, 0, 3]], dtype=mx.uint32)

    x = rng.normal(size=(1, 128)).astype(np.float32)
    got = e8p_m1.e8p_m1_token_layout(
        mx.array(x), layer.codes, layer.effective_scales(), layer.codebook, rhs,
        output_dims=96, rows_per_threadgroup=rows_per_threadgroup,
    )
    want = np.stack([x[0] @ dense[e].T for e in (5, 0, 3)])[None]
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), want, rtol=1e-4, atol=1e-3)

    xr = rng.normal(size=(3, 128)).astype(np.float32)
    got = e8p_m1.e8p_m1_per_route(
        mx.array(xr), layer.codes, layer.effective_scales(), layer.codebook, rhs.reshape(-1),
        output_dims=96, rows_per_threadgroup=rows_per_threadgroup,
    )
    want = np.stack([xr[r] @ dense[e].T for r, e in enumerate((5, 0, 3))])
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), want, rtol=1e-4, atol=1e-3)


def test_benchmark_synthetic_layers_match_the_shipped_byte_accounting():
    from benchmarks import bench_e8p_m1 as bench

    gate = bench.synthetic_projection(experts=2, out=2048, inp=4096, group_size=512, seed=0)
    down = bench.synthetic_projection(experts=2, out=4096, inp=2048, group_size=512, seed=0)
    # The roadmap's V4-Flash decode table: 2,129,920 B per projection per expert.
    assert bench.projection_bytes_per_expert(gate) == 2_129_920
    assert bench.projection_bytes_per_expert(down) == 2_129_920
    assert gate.code_bits == 16 and gate.codes.dtype == mx.uint16
