from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from mlx_vq.build.cli import main as keep_main
from mlx_vq.models.profiles import (
    ModelProfile,
    ProfileError,
    get_profile,
    list_profiles,
    load_profile,
    profile_to_dict,
    validate_profile_against_hf_config,
    validate_profile_against_hf_config_data,
)


def test_registry_loads_known_profiles() -> None:
    assert list_profiles() == (
        "deepseek-v4-flash-0731",
        "glm45-air",
        "glm52-reap-504b-v2",
        "qwen36-35b-a3b",
    )

    air = get_profile("glm45-air")
    glm52 = get_profile("glm52-reap-504b-v2")
    qwen = get_profile("qwen36-35b-a3b")
    deepseek = get_profile("deepseek-v4-flash-0731")

    assert air.hf_model_id == "zai-org/GLM-4.5-Air"
    assert glm52.hf_model_id == "0xSero/glm-5.2-reap-504B-v2"
    assert qwen.hf_model_id == "Qwen/Qwen3.6-35B-A3B"
    assert deepseek.hf_model_id == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert deepseek.default_engine == "vq_e1_routed_nax_e8p"


def test_unknown_profile_error_lists_choices() -> None:
    with pytest.raises(ProfileError, match="unknown model profile 'missing'"):
        get_profile("missing")

    try:
        get_profile("missing")
    except ProfileError as error:
        message = str(error)

    assert "glm45-air" in message
    assert "glm52-reap-504b-v2" in message
    assert "qwen36-35b-a3b" in message
    assert "deepseek-v4-flash-0731" in message


def test_rejects_negative_dimensions() -> None:
    air = get_profile("glm45-air")

    with pytest.raises(ProfileError, match="hidden_size must be positive"):
        ModelProfile(**{**asdict(air), "hidden_size": -1})


def test_rejects_unknown_converter_kind() -> None:
    air = get_profile("glm45-air")

    with pytest.raises(ProfileError, match="unknown converter"):
        ModelProfile(**{**asdict(air), "converter": "other"})


def test_yaml_round_trip(tmp_path: Path) -> None:
    source = get_profile("qwen36-35b-a3b")
    path = tmp_path / "qwen36-35b-a3b.yaml"
    path.write_text(yaml.safe_dump(profile_to_dict(source), sort_keys=False))

    loaded = load_profile(path)

    assert loaded == source


def test_glm45_air_profile_matches_audited_constants() -> None:
    profile = get_profile("glm45-air")

    assert profile.architecture == "glm4_moe"
    assert profile.num_layers == 46
    assert profile.num_sparse_layers == 45
    assert profile.first_sparse_layer == 1
    assert profile.hidden_size == 4096
    assert profile.moe_intermediate_size == 1408
    assert profile.num_experts == 128
    assert profile.experts_per_tok == 8
    assert profile.shared_experts == 1
    assert profile.group_size_policy == {"gate_up": 512, "down": 352}
    assert profile.default_engine == "vq_e1_routed_nax_e8p"
    assert profile.recovery_layer == 45
    assert profile.recovery_projections == ("gate", "up", "down")
    assert profile.imatrix_layers == tuple(range(1, 46))
    assert profile.prompt_sets["imatrix"] == "air_imatrix_calib_v1"
    assert profile.prompt_sets["eval"].startswith("air_vq_ladder_")
    assert profile.lane_s_scenario == "prefill_1k"


def test_qwen_profile_matches_audited_constants() -> None:
    profile = get_profile("qwen36-35b-a3b")

    assert profile.revision == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert profile.architecture == "qwen3_5_moe"
    assert profile.num_layers == 40
    assert profile.hidden_size == 2048
    assert profile.moe_intermediate_size == 512
    assert profile.num_experts == 256
    assert profile.experts_per_tok == 8
    assert profile.shared_experts == 1
    assert profile.vocab_size == 248320
    assert profile.converter == "qwen_moe_groups"
    assert profile.fused_gate_up is True


def test_glm52_reap_profile_matches_cached_config_constants() -> None:
    profile = get_profile("glm52-reap-504b-v2")

    assert profile.revision == "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    assert profile.architecture == "glm_moe_dsa"
    assert profile.num_layers == 78
    assert profile.num_sparse_layers == 75
    assert profile.first_sparse_layer == 3
    assert profile.hidden_size == 6144
    assert profile.moe_intermediate_size == 2048
    assert profile.num_experts == 168
    assert profile.experts_per_tok == 8
    assert profile.shared_experts == 1
    assert profile.vocab_size == 154880
    assert profile.converter == "glm52_vq_groups"
    assert profile.fused_gate_up is False
    assert profile.default_code_bits == 8
    assert profile.group_size_policy == {"gate": 512, "up": 512, "down": 512}
    assert profile.default_engine == "vq_e1_routed_nax_e8"
    assert profile.imatrix_layers == tuple(range(3, 78))
    assert profile.approx_bf16_gb == 1008


def test_hf_config_cross_check_reports_mismatches(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "glm4_moe",
                "num_hidden_layers": 46,
                "mlp_layer_types": ["dense"] + ["sparse"] * 45,
                "hidden_size": 4096,
                "moe_intermediate_size": 1408,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "vocab_size": 151552,
            }
        )
    )

    assert validate_profile_against_hf_config(get_profile("glm45-air"), config_path) == []

    wrong = replace(get_profile("glm45-air"), hidden_size=1)

    assert validate_profile_against_hf_config(wrong, config_path) == [
        "hidden_size: profile=1 config=4096"
    ]


def test_reap_profile_cross_check_accepts_pure_config_data() -> None:
    profile = get_profile("glm52-reap-504b-v2")
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 168,
        "num_experts_per_tok": 8,
        "vocab_size": 154880,
        "n_shared_experts": 1,
    }

    assert validate_profile_against_hf_config_data(profile, config) == []


def test_hf_config_cross_check_uses_cached_configs_when_available() -> None:
    cache_root = Path.home() / ".cache/huggingface/hub"
    paths = {
        "glm45-air": cache_root
        / "models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa/config.json",
        "glm52-reap-504b-v2": cache_root
        / "models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d/config.json",
        "qwen36-35b-a3b": cache_root
        / "models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip("cached HF configs not present: " + ", ".join(missing))

    for name, path in paths.items():
        assert validate_profile_against_hf_config(get_profile(name), path) == []


def test_keep_models_list_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert keep_main(["models"]) == 0

    output = capsys.readouterr().out
    assert "glm45-air" in output
    assert "glm52-reap-504b-v2" in output
    assert "qwen36-35b-a3b" in output
    assert "deepseek-v4-flash-0731" in output


def test_keep_models_show_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert keep_main(["models", "show", "qwen36-35b-a3b"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "qwen36-35b-a3b"
    assert payload["converter"] == "qwen_moe_groups"


def test_converter_registry_is_open():
    from mlx_vq.models import profiles

    assert "glm52_vq_groups" in profiles.converter_kinds()
    assert "deepseek_v4_vq_groups" in profiles.converter_kinds()

    profiles.register_converter("test_family_vq_groups")
    try:
        assert "test_family_vq_groups" in profiles.converter_kinds()
    finally:
        profiles.unregister_converter("test_family_vq_groups")
    assert "test_family_vq_groups" not in profiles.converter_kinds()


def test_register_converter_rejects_blank():
    import pytest
    from mlx_vq.models import profiles

    with pytest.raises(profiles.ProfileError):
        profiles.register_converter("")


def test_deepseek_v4_flash_profile_loads():
    from mlx_vq.models.profiles import get_profile

    profile = get_profile("deepseek-v4-flash-0731")
    assert profile.hf_model_id == "deepseek-ai/DeepSeek-V4-Flash-0731"
    # Pinned to the downloaded, measured snapshot; must match REVISION in
    # scripts/download_dsv4_flash_source.py.
    assert profile.revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert profile.architecture == "deepseek_v4"
    assert profile.num_layers == 43
    assert profile.hidden_size == 4096
    assert profile.vocab_size == 129280
    assert profile.num_experts == 256
    assert profile.experts_per_tok == 6
    assert profile.shared_experts == 1
    assert profile.moe_intermediate_size == 2048
    assert profile.converter == "deepseek_v4_vq_groups"
    # Wave 5 pilot derivation (.superpowers/sdd/wave5-pilot-report.md), not the
    # GLM-5.2 inheritance: E8P rather than E8-1bit, and a uniform group size
    # because group size is a rate dial on this geometry rather than a
    # quality-per-bit lever. Pinned so a silent revert to 8 bits -- which
    # measured a 0.56 block cosine here -- fails loudly.
    assert profile.default_code_bits == 16
    assert profile.group_size_policy == {"gate": 512, "up": 512, "down": 512}
