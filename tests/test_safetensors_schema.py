from __future__ import annotations

import pytest

from keep.env import safetensors_integer_roundtrip_ok, verify_environment
from keep.vq.e8 import E8_1BIT_PACKED_SHA256, E8P_PACKED_ABS_SHA256
from keep.io.schema import QuantizationConfig, VQTensorSpec, codebook_metadata_for_bits


def test_environment_api_gate() -> None:
    report = verify_environment()
    assert report.ok, report.failures
    assert report.mlx_version == "0.31.2"
    assert report.metal_kernel_available
    assert all(report.hadamard_sizes_ok.values())
    assert report.safetensors_integer_roundtrip


def test_safetensors_integer_codes_round_trip() -> None:
    assert safetensors_integer_roundtrip_ok()


def test_vq_tensor_spec_shapes_for_linear_and_switch() -> None:
    linear = VQTensorSpec(
        name="model.layers.3.mlp.down_proj",
        in_dim=6144,
        out_dim=2048,
        code_bits=8,
        group_size=512,
    )
    assert linear.codeword_dtype == "uint8"
    assert linear.codes_shape == (2048, 768)
    assert linear.scales_shape == (2048, 12)

    switch = VQTensorSpec(
        name="model.layers.3.mlp.switch_mlp.gate_proj",
        in_dim=4096,
        out_dim=1408,
        code_bits=16,
        group_size=512,
        num_experts=128,
    )
    assert switch.codeword_dtype == "uint16"
    assert switch.codes_shape == (128, 1408, 512)
    assert switch.scales_shape == (128, 1408, 8)


def test_vq_tensor_spec_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        VQTensorSpec(name="bad", in_dim=1025, out_dim=128).codes_shape

    with pytest.raises(ValueError, match="code_bits"):
        VQTensorSpec(name="bad", in_dim=1024, out_dim=128, code_bits=12).codes_shape


def test_quantization_config_shape() -> None:
    config = QuantizationConfig(
        policy={
            "routed_experts": "vq",
            "attention": "affine_6_or_8_bit",
            "router": "fp16_or_affine_8_bit",
        }
    )
    as_json = config.to_json_dict()
    assert as_json["quant_method"] == "mlx_vq_e8"
    assert as_json["default_code_bits"] == 8
    assert as_json["codebook"] == {
        "name": "quip_e8",
        "dtype": "uint32",
        "entries": 256,
        "sha256": E8_1BIT_PACKED_SHA256,
    }


def test_quantization_config_supports_e8p_codebook_hash_for_code_bits_16() -> None:
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(16)
    config = QuantizationConfig(
        default_code_bits=16,
        codebook_name=codebook_name,
        codebook_sha256=codebook_sha256,
    )

    as_json = config.to_json_dict()

    assert as_json["default_code_bits"] == 16
    assert as_json["codebook"]["name"] == "quip_e8p"
    assert as_json["codebook"]["sha256"] == E8P_PACKED_ABS_SHA256
