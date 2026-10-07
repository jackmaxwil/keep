"""GLM-5.3-Flash config validation and tensor classification.

Both inputs are measured, not written by hand: the fixture is the real
``config.json`` at revision ``eb9eb208`` (sha256 ``bb8f01c4...``), and the
tensor table is the census of all 62 shard headers at the same revision.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ramp.models.glm5_next_policy import (
    DSA,
    KDA,
    Glm5NextPrecision as P,
    classify_glm5_next_parameter,
    layer_schedule,
    text_config_of,
    validate_glm5_next_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "tests/fixtures/glm53_flash_config_eb9eb208.json").read_text())
CENSUS = json.loads((ROOT / "artifacts/census/glm53-flash-source-census-eb9eb208.json").read_text())

_EXPECTED_BY_LEAF = {
    "A_log": P.SOURCE_PRECISION,
    "dt_bias": P.SOURCE_PRECISION,
    "b_proj.weight": P.SOURCE_PRECISION,
    "f_a_proj.weight": P.SOURCE_PRECISION,
    "f_b_proj.weight": P.SOURCE_PRECISION,
    "g_a_proj.weight": P.SOURCE_PRECISION,
    "g_b_proj.weight": P.SOURCE_PRECISION,
    "q_conv1d.weight": P.SOURCE_PRECISION,
    "k_conv1d.weight": P.SOURCE_PRECISION,
    "v_conv1d.weight": P.SOURCE_PRECISION,
    "o_norm.weight": P.SOURCE_PRECISION,
    "q_proj.weight": P.RESIDENT_QUANT_CANDIDATE,
    "k_proj.weight": P.RESIDENT_QUANT_CANDIDATE,
    "v_proj.weight": P.RESIDENT_QUANT_CANDIDATE,
    "o_proj.weight": P.RESIDENT_QUANT_CANDIDATE,
    "o_proj.weight_scale_inv": P.RESIDENT_QUANT_CANDIDATE,
    "q_a_proj.weight": P.RESIDENT_QUANT_CANDIDATE,
    "q_b_proj.weight_scale_inv": P.RESIDENT_QUANT_CANDIDATE,
    "kv_a_proj_with_mqa.weight": P.RESIDENT_QUANT_CANDIDATE,
    "kv_b_proj.weight": P.RESIDENT_QUANT_CANDIDATE,
    "q_a_layernorm.weight": P.SOURCE_PRECISION,
    "kv_a_layernorm.weight": P.SOURCE_PRECISION,
    "indexer.wq_b.weight": P.SOURCE_PRECISION,
    "indexer.index_kpool_compress_ape": P.SOURCE_PRECISION,
}


def _concrete(pattern: str, layer: int) -> str:
    return pattern.replace("layers.N.", f"layers.{layer}.").replace("experts.E.", "experts.7.")


def test_real_config_validates_and_schedule_matches_census():
    assert validate_glm5_next_config(CONFIG) == []
    schedule = layer_schedule(text_config_of(CONFIG))
    assert len(schedule) == 45
    assert [i for i, (a, _) in enumerate(schedule) if a == DSA] == CENSUS["full_attn_layers"]
    assert [i for i, (a, _) in enumerate(schedule) if a == KDA] == CENSUS["kda_layers"]
    assert [m for _, m in schedule[:4]] == ["dense", "dense", "dense", "sparse"]
    assert sum(m == "sparse" for _, m in schedule) == CENSUS["sparse_moe_layers"] == 42


@pytest.mark.parametrize(
    ("mutate", "tag"),
    [
        (lambda t: t.update(qk_rope_head_dim=64), "qk_rope_head_dim"),
        (lambda t: t.update(n_routed_experts=256), "n_routed_experts"),
        (lambda t: t.update(first_k_dense_replace=1), "first_k_dense_replace"),
        (lambda t: t["layer_types"].__setitem__(0, DSA), "kda_layers"),
        (lambda t: t["indexer_types"].__setitem__(7, "shared"), "indexer_types"),
        (lambda t: t.pop("index_kpool"), "missing_index_kpool"),
    ],
)
def test_validation_rejects_what_the_adapter_cannot_run(mutate, tag):
    config = copy.deepcopy(CONFIG)
    mutate(config["text_config"])
    assert tag in validate_glm5_next_config(config)


def test_validation_requires_fp8_128_blocks():
    config = copy.deepcopy(CONFIG)
    config["quantization_config"]["weight_block_size"] = [1, 128]
    assert validate_glm5_next_config(config) == ["quantization_config"]


def test_every_checkpoint_tensor_classifies_as_measured():
    counts = {cls: 0 for cls in P}
    for row in CENSUS["patterns"]:
        name = row["name"]
        layer = 45 if row["count"] == 1 and "layers.N." in name else 20
        if ".experts.E." in name and row["count"] == 12384:
            # 43 layers x 288: backbone layers and the MTP layer share a pattern.
            assert classify_glm5_next_parameter(_concrete(name, 3)) is P.VQ_ROUTED_EXPERT
            assert classify_glm5_next_parameter(_concrete(name, 44)) is P.VQ_ROUTED_EXPERT
            assert classify_glm5_next_parameter(_concrete(name, 45)) is P.MTP
            counts[P.VQ_ROUTED_EXPERT] += row["count"] * 42 // 43
            continue
        cls = classify_glm5_next_parameter(_concrete(name, layer))
        counts[cls] += row["count"]
        leaf = name.split(".self_attn.")[-1]
        if leaf in _EXPECTED_BY_LEAF and layer != 45:
            assert cls is _EXPECTED_BY_LEAF[leaf], name
    # 42 layers x 288 experts x 3 projections x (weight + scale).
    assert counts[P.VQ_ROUTED_EXPERT] == 42 * 288 * 3 * 2
    assert counts[P.VISION_SKIP] == sum(r["count"] for r in CENSUS["patterns"] if ".visual." in r["name"])
    assert counts[P.ROUTER] == 43 * 2


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("model.language_model.layers.12.mlp.experts.287.down_proj.weight_scale_inv", P.VQ_ROUTED_EXPERT),
        ("model.layers.12.mlp.switch_mlp.down_proj", P.VQ_ROUTED_EXPERT),
        ("model.language_model.layers.12.mlp.shared_experts.up_proj.weight", P.RESIDENT_QUANT_CANDIDATE),
        ("model.language_model.layers.1.mlp.gate_proj.weight", P.RESIDENT_QUANT_CANDIDATE),
        ("model.language_model.layers.12.mlp.gate.e_score_correction_bias", P.ROUTER),
        ("model.language_model.layers.12.hc_attn_fn", P.SOURCE_PRECISION),
        ("model.language_model.embed_tokens.weight", P.SOURCE_PRECISION),
        ("model.language_model.norm.weight", P.SOURCE_PRECISION),
        ("lm_head.weight", P.RESIDENT_QUANT_CANDIDATE),
        ("model.language_model.layers.45.eh_proj.weight", P.MTP),
        ("model.language_model.layers.45.mlp.experts.0.gate_proj.weight", P.MTP),
        ("model.visual.merger.proj.weight", P.VISION_SKIP),
    ],
)
def test_classification_of_named_tensors(path, expected):
    assert classify_glm5_next_parameter(path) is expected


def test_policy_imports_headless():
    code = "import sys, ramp.models.glm5_next_policy; assert 'mlx' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
