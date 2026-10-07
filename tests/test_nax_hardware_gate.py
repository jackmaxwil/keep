"""The NAX kernels are only trusted on GPUs that have tensor units."""

import pytest

from ramp.kernels.nax import gpu_has_neural_accelerators


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Apple M5 Max", True),
        ("Apple M5", True),
        ("Apple M6 Pro", True),
        ("Apple M4 Max", False),
        ("Apple M1 Ultra", False),
        ("", False),
        ("AMD Radeon Pro", False),
    ],
)
def test_only_m5_and_newer_count_as_nax_capable(name, expected):
    assert gpu_has_neural_accelerators(name) is expected
