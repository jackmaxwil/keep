"""DeepSeek-V4-Flash parameter classification and config validation.

Tensor names in ``REAL_TENSOR_PATTERNS`` are OBSERVED: they are the real
released names, taken from the 24 of 48 shards of
``deepseek-ai/DeepSeek-V4-Flash-0731`` that were downloaded at scan time,
revision ``7872f01b`` (48 distinct patterns, ``N`` standing in for a layer or
expert index). The list is frozen into this file on purpose: the checkpoint
scan lived in gitignored scratch
(``.superpowers/sdd/dsv4-real-tensor-patterns.txt``), so pinning the policy to
the real names requires carrying them in the test.

Some names asserted in the individual tests below are INFERRED from the
reference implementation rather than observed in that partial scan, and are
marked as such at their use sites: the ``mtp.*`` drafter subtree,
``head.weight``, and the top-level ``norm.weight``. The scanned shards held
neither the drafter nor the model tail, so these are predictions about the
remaining 24 shards. A Wave-1 full-shard scan either confirms them or replaces
them; until then they pin *intent*, not measurement, and ``REAL_TENSOR_PATTERNS``
is the only measured list.

The second naming form covered below is the post-sanitize MLX module tree
(``model.layers.N.ffn.switch_mlp.{gate,up,down}_proj`` etc.) produced by the
reference implementation's ``sanitize``; the policy must classify both. Those
names are likewise inferred from the reference ``sanitize``, not observed in a
checkpoint -- no checkpoint ships them.
"""

import subprocess
import sys

from ramp.models.deepseek_v4_policy import (
    DeepseekV4Precision,
    REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS,
    classify_deepseek_v4_parameter,
    is_vq_routed_expert,
    validate_deepseek_v4_config,
)

_VQ = DeepseekV4Precision.VQ_ROUTED_EXPERT
_AFFINE = DeepseekV4Precision.AFFINE_OR_FP16
_FP32 = DeepseekV4Precision.FP16_OR_FP32
_ROUTER = DeepseekV4Precision.ROUTER
_SKIP = DeepseekV4Precision.MTP_SKIP

# Frozen ground truth: every tensor-name pattern present in the released
# shards, with the precision bucket it must land in. ``N`` is substituted with
# a real index by the tests below.
REAL_TENSOR_PATTERNS: dict[str, DeepseekV4Precision] = {
    "embed.weight": _FP32,
    "layers.N.attn.attn_sink": _AFFINE,
    "layers.N.attn.compressor.ape": _AFFINE,
    "layers.N.attn.compressor.norm.weight": _AFFINE,
    "layers.N.attn.compressor.wgate.weight": _AFFINE,
    "layers.N.attn.compressor.wkv.weight": _AFFINE,
    "layers.N.attn.indexer.compressor.ape": _AFFINE,
    "layers.N.attn.indexer.compressor.norm.weight": _AFFINE,
    "layers.N.attn.indexer.compressor.wgate.weight": _AFFINE,
    "layers.N.attn.indexer.compressor.wkv.weight": _AFFINE,
    "layers.N.attn.indexer.weights_proj.weight": _AFFINE,
    "layers.N.attn.indexer.wq_b.scale": _AFFINE,
    "layers.N.attn.indexer.wq_b.weight": _AFFINE,
    "layers.N.attn.kv_norm.weight": _AFFINE,
    "layers.N.attn.q_norm.weight": _AFFINE,
    "layers.N.attn.wkv.scale": _AFFINE,
    "layers.N.attn.wkv.weight": _AFFINE,
    "layers.N.attn.wo_a.scale": _AFFINE,
    "layers.N.attn.wo_a.weight": _AFFINE,
    "layers.N.attn.wo_b.scale": _AFFINE,
    "layers.N.attn.wo_b.weight": _AFFINE,
    "layers.N.attn.wq_a.scale": _AFFINE,
    "layers.N.attn.wq_a.weight": _AFFINE,
    "layers.N.attn.wq_b.scale": _AFFINE,
    "layers.N.attn.wq_b.weight": _AFFINE,
    "layers.N.attn_norm.weight": _FP32,
    "layers.N.ffn.experts.N.w1.scale": _VQ,
    "layers.N.ffn.experts.N.w1.weight": _VQ,
    "layers.N.ffn.experts.N.w2.scale": _VQ,
    "layers.N.ffn.experts.N.w2.weight": _VQ,
    "layers.N.ffn.experts.N.w3.scale": _VQ,
    "layers.N.ffn.experts.N.w3.weight": _VQ,
    "layers.N.ffn.gate.bias": _ROUTER,
    "layers.N.ffn.gate.tid2eid": _ROUTER,
    "layers.N.ffn.gate.weight": _ROUTER,
    "layers.N.ffn.shared_experts.w1.scale": _AFFINE,
    "layers.N.ffn.shared_experts.w1.weight": _AFFINE,
    "layers.N.ffn.shared_experts.w2.scale": _AFFINE,
    "layers.N.ffn.shared_experts.w2.weight": _AFFINE,
    "layers.N.ffn.shared_experts.w3.scale": _AFFINE,
    "layers.N.ffn.shared_experts.w3.weight": _AFFINE,
    "layers.N.ffn_norm.weight": _FP32,
    "layers.N.hc_attn_base": _FP32,
    "layers.N.hc_attn_fn": _FP32,
    "layers.N.hc_attn_scale": _FP32,
    "layers.N.hc_ffn_base": _FP32,
    "layers.N.hc_ffn_fn": _FP32,
    "layers.N.hc_ffn_scale": _FP32,
}

# Only these may legitimately classify as FP16_OR_FP32. Anything else landing
# there means the classifier fell through to its default instead of matching.
# ``norm``/``embed``/``head`` are anchored at the top level on purpose: the
# ``norm`` inside ``attn.compressor`` is an attention tensor, not a final norm.
_FP32_TOP_LEVEL = ("embed", "head", "norm")
_FP32_SEGMENTS = ("attn_norm", "ffn_norm")


def _fp32_allowlisted(pattern: str) -> bool:
    segments = pattern.split(".")
    if segments[0] in _FP32_TOP_LEVEL:
        return True
    if any(segment.startswith("hc_") for segment in segments):
        return True
    return any(segment in _FP32_SEGMENTS for segment in segments)


def _real_names(pattern: str, index: int) -> str:
    return pattern.replace(".N.", f".{index}.")


def test_frozen_pattern_list_matches_shard_scan():
    # OBSERVED: 48 distinct patterns across the 24 of 48 shards downloaded at
    # scan time @7872f01b. A full-shard rescan may add patterns (the drafter
    # subtree and model tail were not in the scanned range); it must not
    # contradict these.
    assert len(REAL_TENSOR_PATTERNS) == 48


def test_every_real_pattern_classifies_as_reviewed():
    for index in (0, 42):
        for pattern, expected in REAL_TENSOR_PATTERNS.items():
            path = _real_names(pattern, index)
            assert classify_deepseek_v4_parameter(path) is expected, path


def test_no_real_pattern_falls_through_to_fp32():
    """Pins the policy to the released checkpoint's naming forever."""

    for index in (0, 42):
        for pattern in REAL_TENSOR_PATTERNS:
            path = _real_names(pattern, index)
            if _fp32_allowlisted(pattern):
                continue
            assert classify_deepseek_v4_parameter(path) is not _FP32, path


def test_real_routed_expert_projections_are_vq():
    # w1=gate, w2=down, w3=up per DeepSeek convention; the ``.scale``
    # companion tensor classifies with its ``.weight``.
    for proj in ("w1", "w2", "w3"):
        for suffix in ("weight", "scale"):
            path = f"layers.7.ffn.experts.13.{proj}.{suffix}"
            assert classify_deepseek_v4_parameter(path) is _VQ, path
            assert is_vq_routed_expert(path), path


def test_real_router_tensors_are_router():
    for leaf in ("weight", "bias", "tid2eid"):
        path = f"layers.3.ffn.gate.{leaf}"
        assert classify_deepseek_v4_parameter(path) is _ROUTER, path
        assert not is_vq_routed_expert(path), path


def test_real_attention_and_shared_experts_are_affine():
    for path in (
        "layers.2.attn.wq_b.weight",
        "layers.2.attn.wq_b.scale",
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.compressor.norm.weight",
        "layers.2.attn.compressor.ape",
        "layers.0.attn.attn_sink",
        "layers.0.attn.kv_norm.weight",
        "layers.0.ffn.shared_experts.w2.weight",
    ):
        assert classify_deepseek_v4_parameter(path) is _AFFINE, path


def test_real_norms_hyper_connections_and_embeddings_stay_high_precision():
    for path in (
        "embed.weight",
        # INFERRED from the reference implementation, not observed in the
        # partial scan: the model tail was not in the 24 downloaded shards.
        "head.weight",
        "norm.weight",
        "layers.0.attn_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.hc_attn_fn",
        "layers.0.hc_ffn_scale",
    ):
        assert classify_deepseek_v4_parameter(path) is _FP32, path


def test_post_sanitize_switch_mlp_paths_are_vq():
    # Post-sanitize MLX module tree: experts are stacked into ``switch_mlp``
    # and renamed to gate/up/down. Leaf module paths carry no ``.weight``
    # suffix, mirroring the reference quant predicate's ``endswith("_proj")``.
    for path in (
        "model.layers.7.ffn.switch_mlp.gate_proj.weight",
        "model.layers.7.ffn.switch_mlp.up_proj.scales",
        "model.layers.7.ffn.switch_mlp.down_proj",
    ):
        assert classify_deepseek_v4_parameter(path) is _VQ, path
        assert is_vq_routed_expert(path), path


def test_post_sanitize_non_expert_paths():
    assert (
        classify_deepseek_v4_parameter(
            "model.layers.7.ffn.gate.e_score_correction_bias"
        )
        is _ROUTER
    )
    assert (
        classify_deepseek_v4_parameter(
            "model.layers.7.ffn.shared_experts.gate_proj.weight"
        )
        is _AFFINE
    )
    assert (
        classify_deepseek_v4_parameter("model.layers.7.attn.indexer.wq_b.weight")
        is _AFFINE
    )
    assert classify_deepseek_v4_parameter("model.embed_tokens.weight") is _FP32
    assert classify_deepseek_v4_parameter("lm_head.weight") is _FP32
    assert classify_deepseek_v4_parameter("model.norm.weight") is _FP32
    # Hyper-connection scalars after sanitize's ``hc_attn_* -> attn_hc.*`` swap.
    assert classify_deepseek_v4_parameter("model.layers.7.attn_hc.fn") is _FP32
    assert classify_deepseek_v4_parameter("model.hc_head.base") is _FP32


def test_mtp_prefix_is_skipped_before_module_rules():
    # INFERRED: every ``mtp.*`` name here comes from the reference
    # implementation's drafter module, not from the partial shard scan -- the
    # drafter shards were not among the 24 downloaded. The skip rule is written
    # to be prefix-based precisely so it holds whatever the exact leaf names
    # turn out to be.
    for path in (
        "mtp.0.e_proj.weight",
        "model.mtp.0.h_proj.weight",
        "mtp.0.layers.0.ffn.experts.0.w1.weight",
        "mtp.0.layers.0.ffn.switch_mlp.gate_proj.weight",
        "mtp.0.embed.weight",
    ):
        assert classify_deepseek_v4_parameter(path) is _SKIP, path
        assert not is_vq_routed_expert(path), path


def test_mtp_layer_index_is_skip_for_both_naming_forms():
    # num_hidden_layers=43 means layer index 43 and up are drafter layers, not
    # backbone layers. The unprefixed released form must hit this rule too.
    for path in (
        "layers.43.ffn.experts.0.w1.weight",
        "model.layers.43.ffn.switch_mlp.gate_proj.weight",
        "layers.44.attn.wq_b.weight",
    ):
        assert classify_deepseek_v4_parameter(path, num_hidden_layers=43) is _SKIP, path
        assert not is_vq_routed_expert(path, num_hidden_layers=43), path
    assert (
        classify_deepseek_v4_parameter(
            "layers.43.ffn.experts.0.w1.weight", num_hidden_layers=45
        )
        is _VQ
    )


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------

# Mirrors the real config.json of DeepSeek-V4-Flash-0731 (only the fields the
# gate looks at). ``compress_ratios`` ships 46 entries for 43 hidden layers:
# the trailing entries cover the embedded drafter, and the loader truncates.
REAL_COMPRESS_RATIOS = (
    [0, 0]
    + [4 if i % 2 == 0 else 128 for i in range(41)]
    + [0, 0, 0]
)

VALID_CONFIG = {
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "max_position_embeddings": 1048576,
    "num_nextn_predict_layers": 1,
    "index_topk": 512,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "compress_ratios": REAL_COMPRESS_RATIOS,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "num_hash_layers": 3,
    "sliding_window": 128,
    "o_lora_rank": 1024,
    "o_groups": 8,
    # ``quantization_config`` describes the *resident* tensors only. The routed
    # experts are FP4 -- see ``expert_dtype`` below.
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    },
    "expert_dtype": "fp4",
}


def test_real_config_shape_passes():
    assert len(REAL_COMPRESS_RATIOS) == 46
    assert validate_deepseek_v4_config(VALID_CONFIG) == []


def test_missing_field_reported():
    config = dict(VALID_CONFIG)
    del config["index_topk"]
    del config["dspark_target_layer_ids"]
    failures = validate_deepseek_v4_config(config)
    assert "missing_index_topk" in failures
    assert "missing_dspark_target_layer_ids" in failures


def test_scalar_with_wrong_value_reported():
    config = dict(VALID_CONFIG)
    config["model_type"] = "deepseek_v3"
    config["n_routed_experts"] = 128
    failures = validate_deepseek_v4_config(config)
    assert "model_type" in failures
    assert "n_routed_experts" in failures
    assert "missing_model_type" not in failures


def test_expert_dtype_fp4_is_required():
    """``expert_dtype`` is the routed-expert storage discriminator.

    ``quantization_config`` covers residents only, so a checkpoint claiming
    fp8 experts needs a decoder this repo does not have and must not slip
    through the gate that says the policy was measured against this release.
    """

    fp8_experts = dict(VALID_CONFIG)
    fp8_experts["expert_dtype"] = "fp8"
    failures = validate_deepseek_v4_config(fp8_experts)
    assert "expert_dtype" in failures
    assert "missing_expert_dtype" not in failures

    absent = dict(VALID_CONFIG)
    del absent["expert_dtype"]
    assert "missing_expert_dtype" in validate_deepseek_v4_config(absent)


def test_present_but_null_required_field_is_missing_not_ok():
    """A ``null`` value is no more informative than an absent key.

    Regression: the gate used a bare ``in`` membership test, so ``field: null``
    satisfied it and then skipped every value check below, passing clean.
    """

    for field in ("compress_ratios", "expert_dtype", "quantization_config"):
        config = dict(VALID_CONFIG)
        config[field] = None
        failures = validate_deepseek_v4_config(config)
        assert f"missing_{field}" in failures, field
        # Reported once, as missing -- not also as a bad value.
        assert field not in failures, field


def test_non_fp8_quantization_reported():
    config = dict(VALID_CONFIG)
    config["quantization_config"] = {"quant_method": "awq"}
    assert "quantization_config" in validate_deepseek_v4_config(config)


def test_non_e4m3_quantization_fmt_reported():
    config = dict(VALID_CONFIG)
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e5m2"}
    assert "quantization_config" in validate_deepseek_v4_config(config)


def test_non_mapping_quantization_config_reported():
    config = dict(VALID_CONFIG)
    config["quantization_config"] = "fp8"
    assert "quantization_config" in validate_deepseek_v4_config(config)


def test_compress_ratios_validated():
    short = dict(VALID_CONFIG)
    short["compress_ratios"] = [0, 4, 128]
    assert "compress_ratios" in validate_deepseek_v4_config(short)

    bad_value = dict(VALID_CONFIG)
    bad_value["compress_ratios"] = [64] + list(REAL_COMPRESS_RATIOS[1:])
    assert "compress_ratios" in validate_deepseek_v4_config(bad_value)

    not_a_list = dict(VALID_CONFIG)
    not_a_list["compress_ratios"] = 128
    assert "compress_ratios" in validate_deepseek_v4_config(not_a_list)

    exact = dict(VALID_CONFIG)
    exact["compress_ratios"] = list(REAL_COMPRESS_RATIOS[:43])
    assert validate_deepseek_v4_config(exact) == []


def test_required_fields_cover_moe_indexer_and_v4_discriminators():
    for field in (
        "n_routed_experts",
        "index_topk",
        "num_nextn_predict_layers",
        "compress_ratios",
        "dspark_block_size",
        "dspark_target_layer_ids",
        "num_hash_layers",
        "sliding_window",
        "o_lora_rank",
        "o_groups",
    ):
        assert field in REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS
    # ``kv_lora_rank`` is a V3-ism: the real V4-Flash config has no such field
    # (attention is ``wkv``/``kv_norm`` with ``o_lora_rank``/``o_groups``).
    assert "kv_lora_rank" not in REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS


# ---------------------------------------------------------------------------
# Registry binding and module identity
# ---------------------------------------------------------------------------


def test_family_registered_against_the_real_adapter():
    from ramp.models import registry

    binding = registry.get_family("deepseek_v4_flash")
    assert binding.architecture == "deepseek_v4"
    assert binding.profile_name == "deepseek-v4-flash-0731"
    assert binding.adapter_module == "ramp.models.deepseek_v4_flash_adapter"
    assert binding.model_args_symbol == "DeepseekV4FlashVQModelArgs"
    assert binding.bind_vq_experts_symbol == "bind_deepseek_v4_flash_vq_experts"
    assert (
        binding.bind_non_vq_weights_symbol == "bind_deepseek_v4_flash_non_vq_weights"
    )
    assert binding.has_unbound_symbol == "has_unbound_deepseek_v4_flash_vq_experts"


def test_policy_module_no_longer_carries_adapter_stubs():
    """The Wave-2 adapter landed, so the placeholder binders are gone.

    They existed only so the family binding never pointed a binder symbol at
    the classifier; keeping them now would give callers two answers to "how do
    I bind V4 experts".
    """

    import ramp.models.deepseek_v4_policy as policy

    for name in (
        "DEEPSEEK_V4_ADAPTER_NOT_IMPLEMENTED",
        "bind_deepseek_v4_vq_experts_stub",
        "has_unbound_deepseek_v4_vq_experts_stub",
    ):
        assert not hasattr(policy, name)


def test_policy_shim_is_same_module():
    import ramp.models.deepseek_v4_policy  # noqa: F401
    import ramp.models.deepseek_v4_policy as policy

    assert (
        sys.modules["ramp.models.deepseek_v4_policy"]
        is sys.modules["ramp.models.deepseek_v4_policy"]
    )
    assert sys.modules["ramp.models.deepseek_v4_policy"] is policy


_HEADLESS_PROBE = """
import sys

import ramp.models.deepseek_v4_policy
import ramp.models.deepseek_v4_policy

assert sys.modules["ramp.models.deepseek_v4_policy"] is sys.modules[
    "ramp.models.deepseek_v4_policy"
]

heavy = sorted(
    name
    for name in sys.modules
    if name in ("mlx", "numpy") or name.startswith(("mlx.", "numpy."))
)
assert not heavy, heavy
print("ok")
"""


def test_policy_import_stays_headless():
    """The policy is metadata only: no MLX, no NumPy at import time."""

    result = subprocess.run(
        [sys.executable, "-c", _HEADLESS_PROBE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
