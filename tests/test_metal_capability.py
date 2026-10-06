from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from mlx_vq.kernels.metal_capability import (
    probe_mpp_tensor_ops,
    probe_mpp_tensor_ops_accumulation_semantics,
    probe_mpp_tensor_ops_cooperative_matmul,
    probe_mpp_tensor_ops_device_matmul,
    probe_mpp_tensor_ops_device_tensor_matmul,
    probe_mpp_tensor_ops_symbols,
)


REQUIRED_KEYS = {
    "available",
    "compiled",
    "ran",
    "supported",
    "error",
    "error_type",
    "sdk_path",
}


def _assert_probe_shape(result: dict) -> None:
    assert REQUIRED_KEYS <= result.keys()
    assert isinstance(result["available"], bool)
    assert isinstance(result["compiled"], bool)
    assert isinstance(result["ran"], bool)
    assert isinstance(result["supported"], bool)
    assert result["error"] is None or isinstance(result["error"], str)
    assert result["error_type"] is None or isinstance(result["error_type"], str)
    assert isinstance(result["sdk_path"], str)
    assert isinstance(result["sdk_path_note"], str)


def test_mpp_tensor_ops_symbol_probe_is_structured() -> None:
    result = probe_mpp_tensor_ops_symbols()

    _assert_probe_shape(result)
    assert result["level"] == "mpp_tensor_ops_symbols"
    assert "header_path" in result


def test_mpp_tensor_ops_cooperative_matmul_probe_is_structured() -> None:
    result = probe_mpp_tensor_ops_cooperative_matmul()

    _assert_probe_shape(result)
    assert result["level"] == "mpp_tensor_ops_cooperative_matmul"
    assert "header_path" in result


def test_mpp_tensor_ops_device_matmul_probe_is_structured_and_correct_when_supported() -> None:
    result = probe_mpp_tensor_ops_device_matmul()

    _assert_probe_shape(result)
    assert result["level"] == "mpp_tensor_ops_device_matmul"
    assert "header_path" in result
    if result["supported"]:
        assert result["output_shape"] == [32, 16]
        assert result["max_abs_diff"] <= 1e-4
        assert len(result["result_sample"]) == 8


def test_mpp_tensor_ops_device_tensor_matmul_probe_is_structured_and_correct_when_supported() -> None:
    result = probe_mpp_tensor_ops_device_tensor_matmul()

    _assert_probe_shape(result)
    assert result["level"] == "mpp_tensor_ops_device_tensor_matmul"
    assert "header_path" in result
    assert result["descriptor"] == {"m": 64, "n": 32, "k": 32, "execution_simdgroups": 4}
    assert result["input_shapes"] == {"a": [64, 32], "b": [32, 32]}
    if result["supported"]:
        assert result["output_shape"] == [64, 32]
        assert result["max_abs_diff"] <= 1e-4
        assert len(result["result_sample"]) == 8


def test_mpp_tensor_ops_accumulation_semantics_probe_is_structured_and_understood() -> None:
    result = probe_mpp_tensor_ops_accumulation_semantics()

    _assert_probe_shape(result)
    assert result["level"] == "mpp_tensor_ops_accumulation_semantics"
    assert "header_path" in result
    assert result["expected_overwrite_value"] == 16.0
    assert result["expected_accumulate_value"] == 32.0
    assert isinstance(result["overwrites_destination"], bool)
    assert isinstance(result["accumulates_destination"], bool)
    if result["supported"]:
        assert result["overwrites_destination"] is not result["accumulates_destination"]
        assert min(
            abs(result["observed_value"] - result["expected_overwrite_value"]),
            abs(result["observed_value"] - result["expected_accumulate_value"]),
        ) <= 1e-4


def test_mpp_tensor_ops_report_combines_probe_results() -> None:
    result = probe_mpp_tensor_ops()

    _assert_probe_shape(result)
    assert result["level"] == "mpp_tensor_ops"
    assert "symbol_probe" in result
    assert "matmul_probe" in result
    assert "device_matmul_probe" in result
    assert "device_tensor_matmul_probe" in result
    assert "accumulation_semantics_probe" in result
    _assert_probe_shape(result["symbol_probe"])
    _assert_probe_shape(result["matmul_probe"])
    _assert_probe_shape(result["device_matmul_probe"])
    _assert_probe_shape(result["device_tensor_matmul_probe"])
    _assert_probe_shape(result["accumulation_semantics_probe"])
    assert result["device_tensor_matmul_probe"]["skipped"] is True
    assert result["device_tensor_matmul_probe"]["compiled"] is False
    assert result["device_tensor_matmul_probe"]["ran"] is False
    assert result["compiled"] == (
        result["symbol_probe"]["compiled"]
        and result["matmul_probe"]["compiled"]
        and result["device_matmul_probe"]["compiled"]
    )
    assert result["ran"] == (
        result["symbol_probe"]["ran"] and result["matmul_probe"]["ran"] and result["device_matmul_probe"]["ran"]
    )
    assert result["supported"] == (
        result["symbol_probe"]["supported"]
        and result["matmul_probe"]["supported"]
        and result["device_matmul_probe"]["supported"]
    )


def test_mpp_tensor_ops_probe_reports_missing_sdk_without_compiling(tmp_path: Path) -> None:
    result = probe_mpp_tensor_ops_symbols(tmp_path)

    _assert_probe_shape(result)
    assert result["available"] is False
    assert result["compiled"] is False
    assert result["ran"] is False
    assert "was not found" in result["error"]


def test_importing_capability_module_does_not_import_mlx_core() -> None:
    script = """
import importlib
import json
import sys

before = "mlx.core" in sys.modules
module = importlib.import_module("mlx_vq.kernels.metal_capability")
after_import = "mlx.core" in sys.modules
probe = module.probe_mpp_tensor_ops_symbols()
after_probe = "mlx.core" in sys.modules
print(json.dumps({
    "before": before,
    "after_import": after_import,
    "after_probe": after_probe,
    "probe_available": probe["available"],
    "probe_compiled": probe["compiled"],
    "probe_error": probe["error"],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["before"] is False
    assert result["after_import"] is False
    if result["probe_available"]:
        assert result["after_probe"] is True
