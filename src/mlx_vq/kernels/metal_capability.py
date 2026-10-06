from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_SDK_PATH = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
MPP_TENSOR_OPS_HEADER = (
    "System/Library/Frameworks/MetalPerformancePrimitives.framework/"
    "Versions/A/Headers/MPPTensorOpsMatMul2d.h"
)
MPP_INCLUDE = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"


def _base_result(level: str, sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    sdk = Path(sdk_path)
    header_path = sdk / MPP_TENSOR_OPS_HEADER
    return {
        "level": level,
        "available": header_path.exists(),
        "compiled": False,
        "ran": False,
        "supported": False,
        "error": None,
        "error_type": None,
        "sdk_path": str(sdk),
        "sdk_path_note": "Used for header existence checks; MLX selects the Metal compiler SDK/include path.",
        "header_path": str(header_path),
    }


def _record_error(result: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)[:12000]
    return result


def _import_mlx() -> Any:
    import mlx.core as mx

    if not hasattr(getattr(mx, "fast", None), "metal_kernel"):
        raise RuntimeError("mx.fast.metal_kernel is unavailable")
    return mx


def probe_mpp_tensor_ops_symbols(sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    """Compile and run a minimal MPP tensor-ops symbol probe through MLX."""

    result = _base_result("mpp_tensor_ops_symbols", sdk_path)
    if not result["available"]:
        result["error"] = "MPPTensorOpsMatMul2d.h was not found under sdk_path"
        return result

    source = """
uint elem = thread_position_in_grid.x;
constexpr auto descriptor =
    mpp::tensor_ops::matmul2d_descriptor(64, 32, 32, false, false, false);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<4>> op;
out[elem] = inp[elem] + float(descriptor.m == 64 && descriptor.n == 32 && descriptor.k == 32);
"""
    try:
        mx = _import_mlx()
        kernel = mx.fast.metal_kernel(
            name="mlx_vq_mpp_tensor_ops_symbol_probe",
            input_names=["inp"],
            output_names=["out"],
            source=source,
            header=MPP_INCLUDE,
        )
        x = mx.array([1.0], dtype=mx.float32)
        y = kernel(
            inputs=[x],
            grid=(1, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
        mx.eval(y)
        result["compiled"] = True
        result["ran"] = True
        result["supported"] = bool(y.tolist() == [2.0])
        result["result"] = y.tolist()
        return result
    except Exception as exc:  # pragma: no cover - exercised on unsupported SDKs.
        return _record_error(result, exc)


def probe_mpp_tensor_ops_cooperative_matmul(sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    """Compile and run a tiny cooperative-tensor matmul2d op through MLX."""

    result = _base_result("mpp_tensor_ops_cooperative_matmul", sdk_path)
    if not result["available"]:
        result["error"] = "MPPTensorOpsMatMul2d.h was not found under sdk_path"
        return result

    source = """
uint lane = thread_position_in_threadgroup.x;
constexpr auto descriptor =
    mpp::tensor_ops::matmul2d_descriptor(32, 16, 16, false, false, false);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmulOp;
auto aT = matmulOp.get_left_input_cooperative_tensor<half, half, float>();
auto bT = matmulOp.get_right_input_cooperative_tensor<half, half, float>();
auto cT = matmulOp.get_destination_cooperative_tensor<decltype(aT), decltype(bT), float>();
for (uint16_t i = 0; i < aT.get_capacity(); ++i) {
    aT[i] = half(1.0h);
}
for (uint16_t i = 0; i < bT.get_capacity(); ++i) {
    bT[i] = half(1.0h);
}
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    cT[i] = 0.0f;
}
matmulOp.run(aT, bT, cT);
float sum = 0.0f;
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    sum += cT[i];
}
out[lane] = sum;
"""
    try:
        mx = _import_mlx()
        kernel = mx.fast.metal_kernel(
            name="mlx_vq_mpp_tensor_ops_cooperative_matmul_probe",
            input_names=["inp"],
            output_names=["out"],
            source=source,
            header=MPP_INCLUDE,
        )
        x = mx.array([0.0], dtype=mx.float32)
        y = kernel(
            inputs=[x],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(32,)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(y)
        values = y.tolist()
        result["compiled"] = True
        result["ran"] = True
        result["supported"] = bool(values and all(value == 256.0 for value in values))
        result["result"] = values
        return result
    except Exception as exc:  # pragma: no cover - exercised on unsupported SDKs.
        return _record_error(result, exc)


def probe_mpp_tensor_ops_device_matmul(sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    """Run MPP cooperative matmul on values loaded from MLX device arrays."""

    result = _base_result("mpp_tensor_ops_device_matmul", sdk_path)
    if not result["available"]:
        result["error"] = "MPPTensorOpsMatMul2d.h was not found under sdk_path"
        return result

    source = """
constexpr auto descriptor =
    mpp::tensor_ops::matmul2d_descriptor(32, 16, 16, false, false, false);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmulOp;
auto aT = matmulOp.get_left_input_cooperative_tensor<half, half, float>();
auto bT = matmulOp.get_right_input_cooperative_tensor<half, half, float>();
auto cT = matmulOp.get_destination_cooperative_tensor<decltype(aT), decltype(bT), float>();

for (uint16_t i = 0; i < aT.get_capacity(); ++i) {
    if (aT.is_valid_element(i)) {
        auto ids = aT.get_multidimensional_index(i);
        uint k = uint(ids[0]);
        uint m = uint(ids[1]);
        aT[i] = half(a[m * 16u + k]);
    }
}
for (uint16_t i = 0; i < bT.get_capacity(); ++i) {
    if (bT.is_valid_element(i)) {
        auto ids = bT.get_multidimensional_index(i);
        uint n = uint(ids[0]);
        uint k = uint(ids[1]);
        bT[i] = half(b[k * 16u + n]);
    }
}
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
        cT[i] = 0.0f;
    }
}
matmulOp.run(aT, bT, cT);
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
        auto ids = cT.get_multidimensional_index(i);
        uint n = uint(ids[0]);
        uint m = uint(ids[1]);
        out[m * 16u + n] = cT[i];
    }
}
"""
    try:
        import numpy as np

        mx = _import_mlx()
        kernel = mx.fast.metal_kernel(
            name="mlx_vq_mpp_tensor_ops_device_matmul_probe",
            input_names=["a", "b"],
            output_names=["out"],
            source=source,
            header=MPP_INCLUDE,
        )
        a_np = (((np.arange(32 * 16, dtype=np.int32).reshape(32, 16) * 3) % 7) <= 2).astype(np.float16)
        b_np = (((np.arange(16 * 16, dtype=np.int32).reshape(16, 16) * 5 + 1) % 11) <= 4).astype(np.float16)
        a = mx.array(a_np, dtype=mx.float16)
        b = mx.array(b_np, dtype=mx.float16)
        y = kernel(
            inputs=[a, b],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(32, 16)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(y)
        values = np.array(y, dtype=np.float32)
        reference = a_np.astype(np.float32) @ b_np.astype(np.float32)
        max_abs_diff = float(np.max(np.abs(values - reference)))
        result["compiled"] = True
        result["ran"] = True
        result["supported"] = bool(max_abs_diff <= 1e-4)
        result["max_abs_diff"] = max_abs_diff
        result["output_shape"] = list(values.shape)
        result["result_sample"] = values[:2, :4].reshape(-1).tolist()
        return result
    except Exception as exc:  # pragma: no cover - exercised on unsupported SDKs.
        return _record_error(result, exc)


def probe_mpp_tensor_ops_device_tensor_matmul(sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    """Probe direct MPP device-tensor operands through an MLX custom kernel."""

    result = _base_result("mpp_tensor_ops_device_tensor_matmul", sdk_path)
    result.update(
        {
            "descriptor": {"m": 64, "n": 32, "k": 32, "execution_simdgroups": 4},
            "input_shapes": {"a": [64, 32], "b": [32, 32]},
            "output_shape": None,
            "max_abs_diff": None,
            "result_sample": None,
        }
    )
    if not result["available"]:
        result["error"] = "MPPTensorOpsMatMul2d.h was not found under sdk_path"
        return result

    source = """
uint2 tgid = uint2(threadgroup_position_in_grid.xy);
constexpr auto descriptor =
    mpp::tensor_ops::matmul2d_descriptor(64, 32, 32, false, false, false);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<4>> matmulOp;

metal::tensor<const device half, metal::dextents<int32_t, 2>> A(
    a,
    metal::dextents<int32_t, 2>(32, 64));
metal::tensor<const device half, metal::dextents<int32_t, 2>> B(
    b,
    metal::dextents<int32_t, 2>(32, 32));
metal::tensor<device float, metal::dextents<int32_t, 2>> C(
    out,
    metal::dextents<int32_t, 2>(32, 64));

auto tA = A.static_slice<32, 64>(0, tgid.y * 64);
auto tB = B.static_slice<32, 32>(tgid.x * 32, 0);
auto tC = C.static_slice<32, 64>(tgid.x * 32, tgid.y * 64);
matmulOp.run(tA, tB, tC);
"""
    try:
        import numpy as np

        mx = _import_mlx()
        kernel = mx.fast.metal_kernel(
            name="mlx_vq_mpp_tensor_ops_device_tensor_matmul_probe",
            input_names=["a", "b"],
            output_names=["out"],
            source=source,
            header=MPP_INCLUDE,
        )
        a_np = (((np.arange(64 * 32, dtype=np.int32).reshape(64, 32) * 3 + 1) % 13) <= 5).astype(np.float16)
        b_np = (((np.arange(32 * 32, dtype=np.int32).reshape(32, 32) * 5 + 2) % 17) <= 6).astype(np.float16)
        a = mx.array(a_np, dtype=mx.float16)
        b = mx.array(b_np, dtype=mx.float16)
        y = kernel(
            inputs=[a, b],
            grid=(1, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(64, 32)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(y)
        values = np.array(y, dtype=np.float32)
        reference = a_np.astype(np.float32) @ b_np.astype(np.float32)
        max_abs_diff = float(np.max(np.abs(values - reference)))
        result["compiled"] = True
        result["ran"] = True
        result["supported"] = bool(max_abs_diff <= 1e-4)
        result["max_abs_diff"] = max_abs_diff
        result["output_shape"] = list(values.shape)
        result["result_sample"] = values[:2, :4].reshape(-1).tolist()
        return result
    except Exception as exc:  # pragma: no cover - exercised on unsupported SDKs.
        return _record_error(result, exc)


def _skipped_device_tensor_matmul_probe(sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    result = _base_result("mpp_tensor_ops_device_tensor_matmul", sdk_path)
    result.update(
        {
            "descriptor": {"m": 64, "n": 32, "k": 32, "execution_simdgroups": 4},
            "input_shapes": {"a": [64, 32], "b": [32, 32]},
            "output_shape": None,
            "max_abs_diff": None,
            "result_sample": None,
            "skipped": True,
            "skip_reason": (
                "Direct device-tensor MPP probing is opt-in because it is a known-failing "
                "MLX custom-kernel ABI check on the current stack."
            ),
        }
    )
    return result


def probe_mpp_tensor_ops_accumulation_semantics(sdk_path: str | Path = DEFAULT_SDK_PATH) -> dict[str, Any]:
    """Probe whether repeated MPP matmul2d runs overwrite or accumulate C."""

    expected_overwrite_value = 16.0
    expected_accumulate_value = 32.0
    result = _base_result("mpp_tensor_ops_accumulation_semantics", sdk_path)
    result.update(
        {
            "observed_value": None,
            "expected_overwrite_value": expected_overwrite_value,
            "expected_accumulate_value": expected_accumulate_value,
            "overwrites_destination": False,
            "accumulates_destination": False,
        }
    )
    if not result["available"]:
        result["error"] = "MPPTensorOpsMatMul2d.h was not found under sdk_path"
        return result

    source = """
constexpr auto descriptor =
    mpp::tensor_ops::matmul2d_descriptor(32, 16, 16, false, false, false);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmulOp;
auto aT = matmulOp.get_left_input_cooperative_tensor<half, half, float>();
auto bT = matmulOp.get_right_input_cooperative_tensor<half, half, float>();
auto cT = matmulOp.get_destination_cooperative_tensor<decltype(aT), decltype(bT), float>();

for (uint16_t i = 0; i < aT.get_capacity(); ++i) {
    aT[i] = half(1.0h);
}
for (uint16_t i = 0; i < bT.get_capacity(); ++i) {
    bT[i] = half(1.0h);
}
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
        cT[i] = 0.0f;
    }
}
matmulOp.run(aT, bT, cT);
matmulOp.run(aT, bT, cT);
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
        auto ids = cT.get_multidimensional_index(i);
        uint n = uint(ids[0]);
        uint m = uint(ids[1]);
        out[m * 16u + n] = cT[i];
    }
}
"""
    try:
        import numpy as np

        mx = _import_mlx()
        kernel = mx.fast.metal_kernel(
            name="mlx_vq_mpp_tensor_ops_accumulation_semantics_probe",
            input_names=["inp"],
            output_names=["out"],
            source=source,
            header=MPP_INCLUDE,
        )
        x = mx.array([0.0], dtype=mx.float32)
        y = kernel(
            inputs=[x],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(32, 16)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(y)
        values = np.array(y, dtype=np.float32)
        observed_value = float(values[0, 0])
        overwrites_destination = bool(np.allclose(values, expected_overwrite_value, rtol=0.0, atol=1e-4))
        accumulates_destination = bool(np.allclose(values, expected_accumulate_value, rtol=0.0, atol=1e-4))
        result["compiled"] = True
        result["ran"] = True
        result["observed_value"] = observed_value
        result["overwrites_destination"] = overwrites_destination
        result["accumulates_destination"] = accumulates_destination
        result["supported"] = overwrites_destination or accumulates_destination
        result["output_shape"] = list(values.shape)
        result["result_sample"] = values[:2, :4].reshape(-1).tolist()
        return result
    except Exception as exc:  # pragma: no cover - exercised on unsupported SDKs.
        return _record_error(result, exc)


def probe_mpp_tensor_ops(
    sdk_path: str | Path = DEFAULT_SDK_PATH,
    *,
    include_device_tensor_matmul: bool = False,
) -> dict[str, Any]:
    """Return a benchmark-only MPP tensor-ops capability report."""

    symbol_probe = probe_mpp_tensor_ops_symbols(sdk_path)
    matmul_probe = probe_mpp_tensor_ops_cooperative_matmul(sdk_path)
    device_matmul_probe = probe_mpp_tensor_ops_device_matmul(sdk_path)
    device_tensor_matmul_probe = (
        probe_mpp_tensor_ops_device_tensor_matmul(sdk_path)
        if include_device_tensor_matmul
        else _skipped_device_tensor_matmul_probe(sdk_path)
    )
    accumulation_semantics_probe = probe_mpp_tensor_ops_accumulation_semantics(sdk_path)
    return {
        "level": "mpp_tensor_ops",
        "available": symbol_probe["available"] or matmul_probe["available"] or device_matmul_probe["available"],
        "compiled": symbol_probe["compiled"] and matmul_probe["compiled"] and device_matmul_probe["compiled"],
        "ran": symbol_probe["ran"] and matmul_probe["ran"] and device_matmul_probe["ran"],
        "supported": symbol_probe["supported"] and matmul_probe["supported"] and device_matmul_probe["supported"],
        "error": device_matmul_probe["error"] or matmul_probe["error"] or symbol_probe["error"],
        "error_type": device_matmul_probe["error_type"] or matmul_probe["error_type"] or symbol_probe["error_type"],
        "sdk_path": str(Path(sdk_path)),
        "sdk_path_note": "Used for header existence checks; MLX selects the Metal compiler SDK/include path.",
        "symbol_probe": symbol_probe,
        "matmul_probe": matmul_probe,
        "device_matmul_probe": device_matmul_probe,
        "device_tensor_matmul_probe": device_tensor_matmul_probe,
        "accumulation_semantics_probe": accumulation_semantics_probe,
    }
