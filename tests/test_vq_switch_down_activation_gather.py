"""The down projection must not skip the activation gather unless x is pre-sorted.

``gather_vqmm_sorted_routes`` used to infer "x is already in sorted-route order"
from ``x.shape[0] == route_count``. That is true both for the shared prefill path
(which pre-sorts and passes an ``arange`` lhs) and for ``gather_vqmm``'s
three-chain path (which passes ``flat_lhs[argsort(rhs)]`` with unsorted x), but
valid only for the first. M=1 decode takes the second, so the down projection was
computed against permuted activations whenever the routed expert ids were not
already ascending.

The defect is latent in production rather than active: the DSV4 router selects
experts with ``mx.argpartition(-biased, kth=top_k - 1)[..., :top_k]``
(``src/ramp/models/deepseek_v4_flash_adapter.py:892``), and argpartition returns
the selected prefix in ascending index order, so ``argsort`` is the identity and
the skipped gather was a no-op. Verified by measurement: post-fix decode emits
byte-identical tokens and the same 11/19 acceptance as the pre-fix run. These
tests pin the contract at the API level, where the defect is reachable by any
caller that supplies unsorted routes.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vq.codebook.e8 import e8p_packed_abs_grid
from mlx_vq.kernels import nax
from mlx_vq.ops import vq_switch

EXPERTS = 256
GROUP = 512
CODE_BITS = 16
IN_DIMS = 2048
OUT_DIMS = 4096  # output >= input, so _infer_air_projection resolves to "down"
ROUTES = 6


def _operands(seed: int = 7):
    rng = np.random.default_rng(seed)
    codes = mx.array(
        rng.integers(
            0, 1 << 16, size=(EXPERTS, OUT_DIMS, IN_DIMS // 8), dtype=np.uint16
        )
    )
    scales = mx.array(
        rng.normal(0.0, 0.02, size=(EXPERTS, OUT_DIMS, IN_DIMS // GROUP)).astype(
            np.float16
        )
    )
    codebook = mx.array(e8p_packed_abs_grid())
    x = mx.array(rng.normal(0.0, 1.0, size=(ROUTES, IN_DIMS)).astype(np.float16))
    mx.eval(codes, scales, codebook, x)
    return codes, scales, codebook, x


def _run(x, codes, scales, codebook, rhs, strategy):
    out = vq_switch.gather_vqmm(
        x,
        codes,
        scales,
        codebook,
        rhs,
        lhs_indices=mx.arange(ROUTES, dtype=mx.int32),
        input_dims=IN_DIMS,
        output_dims=OUT_DIMS,
        group_size=GROUP,
        code_bits=CODE_BITS,
        route_strategy=strategy,
    )
    mx.eval(out)
    return np.asarray(out, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a.ravel() @ b.ravel()) / denom) if denom else float("nan")


@pytest.mark.parametrize(
    "expert_ids",
    [
        pytest.param([3, 11, 40, 77, 150, 201], id="already_ascending"),
        pytest.param([201, 3, 150, 11, 77, 40], id="shuffled"),
        pytest.param([201, 150, 77, 40, 11, 3], id="descending"),
    ],
)
def test_down_projection_matches_reference_for_any_route_order(expert_ids):
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")
    codes, scales, codebook, x = _operands()
    rhs = mx.array(np.asarray(expert_ids, dtype=np.int32))

    reference = _run(x, codes, scales, codebook, rhs, "direct")
    routed = _run(x, codes, scales, codebook, rhs, "auto")

    # Quantization noise only. The pre-fix bug produced cosine ~= -0.01 and a max
    # absolute delta over 8 on exactly these operands.
    assert _cosine(reference, routed) > 0.999
    assert float(np.abs(reference - routed).max()) < 0.5


def test_shortcut_requires_the_caller_to_declare_pre_sorted_input():
    if not nax.is_available():
        pytest.skip("native VQ NAX extension is not built")
    codes, scales, codebook, x = _operands()
    rhs_sorted = mx.array(np.asarray([3, 11, 40, 77, 150, 201], dtype=np.int32))
    identity = mx.arange(ROUTES, dtype=mx.int32)

    def call(**kwargs):
        out = vq_switch.gather_vqmm_sorted_routes(
            x,
            codes,
            scales,
            codebook,
            rhs_sorted,
            identity,
            input_dims=IN_DIMS,
            output_dims=OUT_DIMS,
            group_size=GROUP,
            code_bits=CODE_BITS,
            implementation="nax_e8p_m32n64",
            projection="down",
            **kwargs,
        )
        mx.eval(out)
        return np.asarray(out, dtype=np.float32)

    # With an identity lhs the gather is a no-op, so declaring it changes nothing.
    gathered = call()
    skipped = call(x_pre_sorted=True)
    assert _cosine(gathered, skipped) > 0.9999
    assert float(np.abs(gathered - skipped).max()) == pytest.approx(0.0, abs=1e-6)
