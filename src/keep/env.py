from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any


MIN_MLX_VERSION = (0, 31, 2)
HADAMARD_REQUIRED_SIZES = (4096, 6144, 12288)


@dataclass(frozen=True)
class EnvironmentReport:
    mlx_version: str | None
    mlx_lm_available: bool
    metal_kernel_available: bool
    hadamard_sizes_ok: dict[int, bool]
    safetensors_integer_roundtrip: bool
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mlx_version": self.mlx_version,
            "mlx_lm_available": self.mlx_lm_available,
            "metal_kernel_available": self.metal_kernel_available,
            "hadamard_sizes_ok": self.hadamard_sizes_ok,
            "safetensors_integer_roundtrip": self.safetensors_integer_roundtrip,
            "failures": list(self.failures),
        }


def _parse_version(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = []
        for char in raw_part:
            if not char.isdigit():
                break
            digits.append(char)
        if digits:
            parts.append(int("".join(digits)))
    return tuple(parts)


def _version_at_least(version: str | None, minimum: tuple[int, ...]) -> bool:
    parsed = _parse_version(version)
    if not parsed:
        return False
    padded = parsed + (0,) * max(0, len(minimum) - len(parsed))
    return padded[: len(minimum)] >= minimum


def _check_hadamard_sizes(mx: Any) -> dict[int, bool]:
    results: dict[int, bool] = {}
    for size in HADAMARD_REQUIRED_SIZES:
        try:
            x = mx.ones((1, size), dtype=mx.float16)
            y = mx.hadamard_transform(x)
            mx.eval(y)
            results[size] = tuple(y.shape) == (1, size)
        except Exception:
            results[size] = False
    return results


def safetensors_integer_roundtrip_ok() -> bool:
    import mlx.core as mx

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        expected_u8 = [[1, 2, 3, 4]]
        expected_u16 = [[1, 258, 65535]]
        arrays = {
            "codes_u8": mx.array(expected_u8, dtype=mx.uint8),
            "codes_u16": mx.array(expected_u16, dtype=mx.uint16),
            "scale_f16": mx.array([[1.0, 2.0]], dtype=mx.float16),
        }
        mx.save_safetensors(path, arrays)
        loaded = mx.load(path)
        return (
            loaded["codes_u8"].dtype == mx.uint8
            and loaded["codes_u16"].dtype == mx.uint16
            and loaded["codes_u8"].tolist() == expected_u8
            and loaded["codes_u16"].tolist() == expected_u16
        )
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def verify_environment() -> EnvironmentReport:
    failures: list[str] = []
    mlx_version: str | None = None
    metal_kernel_available = False
    hadamard_sizes_ok: dict[int, bool] = {size: False for size in HADAMARD_REQUIRED_SIZES}
    roundtrip_ok = False

    try:
        import mlx.core as mx

        mlx_version = getattr(mx, "__version__", None)
        if not _version_at_least(mlx_version, MIN_MLX_VERSION):
            failures.append(
                f"mlx.core version {mlx_version!r} is below required "
                f"{'.'.join(map(str, MIN_MLX_VERSION))}"
            )
        metal_kernel_available = hasattr(getattr(mx, "fast", None), "metal_kernel")
        if not metal_kernel_available:
            failures.append("mx.fast.metal_kernel is unavailable")
        if not hasattr(mx, "hadamard_transform"):
            failures.append("mx.hadamard_transform is unavailable")
        else:
            hadamard_sizes_ok = _check_hadamard_sizes(mx)
            bad_sizes = [str(size) for size, ok in hadamard_sizes_ok.items() if not ok]
            if bad_sizes:
                failures.append(
                    "mx.hadamard_transform failed required final-axis sizes: "
                    + ", ".join(bad_sizes)
                )
        roundtrip_ok = safetensors_integer_roundtrip_ok()
        if not roundtrip_ok:
            failures.append("safetensors uint8/uint16 roundtrip failed")
    except Exception as exc:
        failures.append(f"MLX environment probe failed: {type(exc).__name__}: {exc}")

    return EnvironmentReport(
        mlx_version=mlx_version,
        mlx_lm_available=importlib.util.find_spec("mlx_lm") is not None,
        metal_kernel_available=metal_kernel_available,
        hadamard_sizes_ok=hadamard_sizes_ok,
        safetensors_integer_roundtrip=roundtrip_ok,
        failures=tuple(failures),
    )
