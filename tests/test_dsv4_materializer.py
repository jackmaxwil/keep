"""Wave 5 production materializer: headless tests.

No real checkpoint, no calibration run, no 3.4 GB read. Everything here builds
its own tiny synthetic FP4 block and drives the *production* code path over it,
including the bind-and-forward roundtrip that makes ``w1 -> gate_proj`` a
verified mapping rather than a plausible one.

Two things deliberately are *not* asserted from a fixture. The first is the
real-weight fit parity of the MLX port and the second is the real-layer
roundtrip; both need the 163 GB release, so they are run once by the CLI into
the run directory and the tests here assert on that evidence when it is present
and skip when it is not.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from mlx_vq.codebook.e8 import E8P_PACKED_ABS_SHA256, e8p_full_grid
from mlx_vq.convert.dsv4_vq_fit_mlx import (
    FIT_BACKENDS,
    SCALE_REDUCTIONS,
    compare_fit_backends,
    quantize_weight_importance_aware_mlx,
    resolve_fit_backend,
)
from mlx_vq.convert.dsv4_vq_materialize import (
    ARTIFACT_SCHEMA_VERSION,
    CONVERTER_KIND,
    FAMILY,
    PROJECTIONS,
    SOURCE_REVISION,
    TENSOR_MAPPING,
    MaterializePolicy,
    audit_artifact_tree,
    block_importance,
    build_imatrix_cache,
    build_manifest,
    decode_artifact_experts,
    load_imatrix_cache,
    materialize_block,
    plan_blocks,
    read_block_experts,
    record_path,
    verified_block_record,
    verify_bind_roundtrip,
    write_block_record,
)
from mlx_vq.convert.glm52_recovery_materialize import (
    quantize_weight_importance_aware,
)
from mlx_vq.quality.dsv4_teacher_runner import (
    build_dsv4_block_span_index,
    build_dsv4_expert_span_index,
)

RUN_DIR = Path.home() / "keep-artifacts" / "dsv4-vq-e8p-g512"

# The toy geometry: small enough to fit in a test, structured exactly like the
# release (three projections, w2 transposed, FP4 group-32 scales).
TOY_HIDDEN = 64
TOY_MOE = 32
TOY_EXPERTS = 4
TOY_GROUP = 32
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic FP4 checkpoint
# ---------------------------------------------------------------------------


def _save_raw_safetensors(path: Path, tensors: dict[str, tuple[str, tuple, bytes]]) -> None:
    """Write a safetensors file with dtypes numpy cannot express (I8, F8_E8M0)."""

    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        handle.write(bytes(payload))


def _fp4_expert(rng, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random FP4-in-I8 codes plus E8M0 group-32 scales, and the exact weight.

    The release packs two e2m1 codes per byte with a power-of-two group scale, so
    the dense weight is exactly recoverable -- which is what makes the roundtrip
    assertions meaningful instead of tolerance-bound.
    """

    magnitude = rng.integers(0, 8, size=(rows, cols)).astype(np.uint8)
    sign = rng.integers(0, 2, size=(rows, cols)).astype(np.uint8)
    nibbles = (sign << 3) | magnitude
    packed = (nibbles[:, 1::2].astype(np.uint8) << 4) | nibbles[:, 0::2].astype(np.uint8)
    exponents = rng.integers(120, 132, size=(rows, cols // TOY_GROUP)).astype(np.uint8)
    scale = np.exp2(exponents.astype(np.float32) - 127.0)
    values = _E2M1[magnitude] * np.where(sign > 0, -1.0, 1.0).astype(np.float32)
    dense = (
        values.reshape(rows, cols // TOY_GROUP, TOY_GROUP) * scale[:, :, None]
    ).reshape(rows, cols)
    return packed, exponents, dense.astype(np.float32)


def _write_toy_checkpoint(path: Path, blocks=("layers.0", "layers.2", "mtp.0")) -> dict:
    """One shard per block carrying only routed experts, plus the index."""

    rng = np.random.default_rng(20260814)
    shapes = {
        "w1": (TOY_MOE, TOY_HIDDEN),
        "w3": (TOY_MOE, TOY_HIDDEN),
        "w2": (TOY_HIDDEN, TOY_MOE),
    }
    weight_map: dict[str, str] = {}
    dense: dict[tuple[str, str, int], np.ndarray] = {}
    for position, block in enumerate(blocks):
        shard = f"model-{position + 1:05d}-of-{len(blocks):05d}.safetensors"
        tensors: dict[str, tuple[str, tuple, bytes]] = {}
        for expert in range(TOY_EXPERTS):
            for projection, (rows, cols) in shapes.items():
                packed, exponents, values = _fp4_expert(rng, rows, cols)
                dense[(block, projection, expert)] = values
                base = f"{block}.ffn.experts.{expert}.{projection}"
                tensors[f"{base}.weight"] = ("I8", packed.shape, packed.tobytes())
                tensors[f"{base}.scale"] = ("F8_E8M0", exponents.shape, exponents.tobytes())
        _save_raw_safetensors(path / shard, tensors)
        weight_map.update({name: shard for name in tensors})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return dense


def _toy_cache(layers=None) -> dict:
    """An imatrix cache shaped like the real one but at the toy geometry."""

    rng = np.random.default_rng(11)
    rows = list(range(43) if layers is None else layers)
    return {
        "layers": rows,
        "num_experts": TOY_EXPERTS,
        "sessions": 40,
        "route_count": rng.integers(1, 5000, size=(len(rows), TOY_EXPERTS)).astype(np.int64),
        "importance": {
            "hidden": (
                rng.random((len(rows), TOY_EXPERTS, TOY_HIDDEN)).astype(np.float32) * 10 + 0.1
            ),
            "down": (
                rng.random((len(rows), TOY_EXPERTS, TOY_MOE)).astype(np.float32) * 10 + 0.1
            ),
        },
    }


TOY_POLICY = MaterializePolicy(code_bits=16, group_size=TOY_GROUP, iterations=3)


@pytest.fixture
def toy_run(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dense = _write_toy_checkpoint(checkpoint)
    return {
        "checkpoint": checkpoint,
        "output": tmp_path / "artifact",
        "dense": dense,
        "cache": _toy_cache(),
    }


def _materialize(toy_run, block: str = "layers.0", policy: MaterializePolicy = TOY_POLICY):
    spec = plan_blocks(only=[block])[0]
    record = materialize_block(
        spec,
        checkpoint_dir=toy_run["checkpoint"],
        output_dir=toy_run["output"],
        cache=toy_run["cache"],
        policy=policy,
    )
    write_block_record(toy_run["output"], record)
    return spec, record


# ---------------------------------------------------------------------------
# The plan and the naming contract
# ---------------------------------------------------------------------------


def test_the_plan_is_46_moe_blocks_43_backbone_then_3_drafter():
    specs = plan_blocks()
    assert len(specs) == 46
    assert [spec.kind for spec in specs] == ["backbone"] * 43 + ["mtp"] * 3
    assert specs[0].key == "layers.0" and specs[42].key == "layers.42"
    assert [spec.key for spec in specs[43:]] == ["mtp.0", "mtp.1", "mtp.2"]


def test_backbone_names_match_the_adapter_loader_convention():
    """The loader's own format strings, spelled out so a rename breaks a test."""

    spec = plan_blocks(only=["layers.7"])[0]
    assert spec.file_stem == "layer-00007"
    assert spec.artifact_name("gate_proj") == "layer-00007-gate_proj.safetensors"
    assert spec.tensor_names("down_proj") == (
        "model.layers.7.ffn.switch_mlp.down_proj.codes",
        "model.layers.7.ffn.switch_mlp.down_proj.scales",
    )


def test_drafter_blocks_use_the_mtp_parameter_path_not_a_layer_index():
    spec = plan_blocks(only=["mtp.2"])[0]
    assert spec.file_stem == "mtp-00002"
    assert spec.tensor_prefix == "mtp_drafter.blocks.2.ffn.switch_mlp"
    assert spec.calibration_layer is None


def test_tensor_mapping_is_w1_gate_w3_up_w2_down():
    assert TENSOR_MAPPING == {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def test_artifact_name_refuses_an_unknown_projection():
    with pytest.raises(ValueError, match="projection must be one of"):
        plan_blocks(only=["layers.0"])[0].artifact_name("mlp")


def test_plan_refuses_unknown_block_keys():
    with pytest.raises(ValueError, match="unknown block keys"):
        plan_blocks(only=["layers.99"])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_default_policy_is_the_ship_ladder():
    policy = MaterializePolicy()
    assert (policy.code_bits, policy.group_size, policy.iterations) == (16, 512, 8)
    assert policy.bpw == pytest.approx(2.03125)
    assert policy.as_dict()["group_size_policy"] == {"gate": 512, "up": 512, "down": 512}


@pytest.mark.parametrize(
    "overrides",
    [
        {"code_bits": 4},
        {"group_size": 0},
        {"group_size": 12},
        {"iterations": 0},
        {"mtp_importance": "guess"},
    ],
)
def test_policy_refuses_out_of_contract_values(overrides):
    with pytest.raises(ValueError):
        MaterializePolicy(**overrides)


# ---------------------------------------------------------------------------
# Block span index (the MTP generalisation)
# ---------------------------------------------------------------------------


def test_block_span_index_reaches_backbone_and_drafter_blocks(toy_run):
    index = build_dsv4_block_span_index(toy_run["checkpoint"])
    assert sorted(index) == ["layers.0", "layers.2", "mtp.0"]
    assert index["mtp.0"].block == "mtp.0"
    assert index["mtp.0"].num_experts == TOY_EXPERTS


def test_backbone_span_index_keeps_its_int_keyed_contract(toy_run):
    index = build_dsv4_expert_span_index(toy_run["checkpoint"])
    assert list(index) == [0, 2]
    assert index[0].block == "layers.0"
    assert index[2].block == "layers.2"


def test_block_span_index_refuses_a_block_the_checkpoint_lacks(toy_run):
    with pytest.raises(ValueError, match="no routed experts for blocks"):
        build_dsv4_block_span_index(toy_run["checkpoint"], blocks=["layers.9"])


def test_read_block_experts_returns_checkpoint_named_stacks(toy_run):
    weights, scales, seconds, span = read_block_experts(toy_run["checkpoint"], "layers.0")
    assert sorted(weights) == ["w1", "w2", "w3"]
    assert weights["w1"].shape == (TOY_EXPERTS, TOY_MOE, TOY_HIDDEN // 2)
    assert scales["w2"].shape == (TOY_EXPERTS, TOY_HIDDEN, TOY_MOE // TOY_GROUP)
    assert seconds >= 0.0 and span.num_experts == TOY_EXPERTS


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------


def test_imatrix_cache_roundtrips_through_disk(tmp_path):
    sessions = tmp_path / "calibration" / "sessions"
    sessions.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(2):
        np.savez(
            sessions / f"session-{index}.npz",
            record_type="dsv4_teacher_calibration_v1",
            layers=np.arange(3, dtype=np.int64),
            num_experts=np.int64(TOY_EXPERTS),
            prompt_id=f"prompt-{index}",
            importance_sum__hidden=rng.random((3, TOY_EXPERTS, TOY_HIDDEN)).astype(np.float32),
            importance_sum__down=rng.random((3, TOY_EXPERTS, TOY_MOE)).astype(np.float32),
            affinity_weighted_importance__hidden=rng.random(
                (3, TOY_EXPERTS, TOY_HIDDEN)
            ).astype(np.float32),
            affinity_weighted_importance__down=rng.random(
                (3, TOY_EXPERTS, TOY_MOE)
            ).astype(np.float32),
            route_count__hidden=rng.integers(1, 100, size=(3, TOY_EXPERTS)).astype(np.int64),
            route_count__down=rng.integers(1, 100, size=(3, TOY_EXPERTS)).astype(np.int64),
        )
    meta = build_imatrix_cache(
        tmp_path / "calibration", tmp_path / "cache.npz", layers=[0, 1, 2]
    )
    assert meta["sessions"] == 2 and len(meta["sha256"]) == 64
    assert Path(meta["path"]).name == "cache.npz"
    cache = load_imatrix_cache(tmp_path / "cache.npz")
    assert cache["layers"] == [0, 1, 2]
    assert cache["importance"]["hidden"].shape == (3, TOY_EXPERTS, TOY_HIDDEN)


def test_backbone_importance_is_the_measured_row(toy_run):
    spec = plan_blocks(only=["layers.5"])[0]
    vector, source = block_importance(
        toy_run["cache"],
        spec,
        projection="gate_proj",
        expert=2,
        policy=TOY_POLICY,
        mtp_reference_layer=42,
    )
    assert source == "measured"
    assert np.array_equal(vector, toy_run["cache"]["importance"]["hidden"][5, 2])


def test_drafter_importance_falls_back_and_names_its_source(toy_run):
    spec = plan_blocks(only=["mtp.1"])[0]
    vector, source = block_importance(
        toy_run["cache"],
        spec,
        projection="down_proj",
        expert=0,
        policy=TOY_POLICY,
        mtp_reference_layer=42,
    )
    assert source == "backbone_mean_layer_42"
    assert vector.shape == (TOY_MOE,)
    # Normalised to unit mean: a route count is a magnitude, only the per-column
    # shape is the objective, so the fallback must not carry layer 42's scale.
    assert float(vector.mean()) == pytest.approx(1.0, rel=1e-5)


def test_an_unrouted_expert_gets_uniform_importance_and_is_named(toy_run):
    """A measured row can be all zeros, and that is not an error.

    Measured on the real release for exactly one pair in the 43 x 256 grid --
    `layers.40` expert 170, route_count 0 across all 40 calibration sessions,
    weights perfectly ordinary. The expert still has to be in the artifact, so
    it is fitted against uniform importance and *named*.
    """

    cache = toy_run["cache"]
    cache["importance"]["hidden"][3, 2] = 0.0
    cache["route_count"][3, 2] = 0
    spec = plan_blocks(only=["layers.3"])[0]

    vector, source = block_importance(
        cache,
        spec,
        projection="gate_proj",
        expert=2,
        policy=TOY_POLICY,
        mtp_reference_layer=42,
    )
    assert source == "uniform_zero_importance_fallback"
    assert np.all(vector == 1.0) and vector.shape == (TOY_HIDDEN,)
    # Its neighbours are untouched: the fallback is per expert, not per layer.
    _, neighbour = block_importance(
        cache,
        spec,
        projection="gate_proj",
        expert=1,
        policy=TOY_POLICY,
        mtp_reference_layer=42,
    )
    assert neighbour == "measured"


def test_a_block_with_an_unrouted_expert_materializes_and_reports_it(toy_run):
    """The regression: this crashed the 46-block sweep at block 41 of 46.

    The fit already coerced the zero vector to ones internally, so the codes
    were fine; the materializer then scored them against the *raw* zero vector
    and `projection_error` correctly refused a zero weighted reference energy.
    The fix makes the substitution explicit, so the fit and the metrics see the
    same effective importance and the expert is recorded.
    """

    cache = toy_run["cache"]
    cache["importance"]["hidden"][0, 1] = 0.0
    cache["importance"]["down"][0, 1] = 0.0
    cache["route_count"][0, 1] = 0

    spec, record = _materialize(toy_run)

    assert record.importance_fallback_experts == [1]
    assert record.importance_sources["uniform_zero_importance_fallback"] == 3
    assert record.importance_sources["measured"] == (TOY_EXPERTS - 1) * 3
    # All 256-equivalent experts are present: an unrouted expert is still bound.
    assert record.files[0]["codes_shape"][0] == TOY_EXPERTS
    assert np.isfinite(record.metrics["gate_proj"]["weighted_relative_mse_max"])
    assert (
        verified_block_record(
            toy_run["output"], spec, policy=TOY_POLICY, expected_experts=TOY_EXPERTS
        )
        is not None
    )

    manifest = build_manifest(
        output_dir=toy_run["output"],
        checkpoint_dir=toy_run["checkpoint"],
        specs=[spec],
        policy=TOY_POLICY,
        records=[record.as_dict()],
        calibration={},
    )
    assert manifest["audit"]["importance_fallback_experts"] == {"layers.0": [1]}
    assert manifest["audit"]["importance_fallback_expert_count"] == 1
    # A block with no cold experts contributes no key at all.
    assert "layers.1" not in manifest["audit"]["importance_fallback_experts"]


def test_drafter_uniform_policy_returns_flat_ones(toy_run):
    spec = plan_blocks(only=["mtp.0"])[0]
    vector, source = block_importance(
        toy_run["cache"],
        spec,
        projection="gate_proj",
        expert=0,
        policy=MaterializePolicy(mtp_importance="uniform"),
        mtp_reference_layer=42,
    )
    assert source == "uniform_fallback"
    # Real geometry: the uniform fallback does not consult the cache at all.
    assert vector.shape == (4096,) and np.all(vector == 1.0)


# ---------------------------------------------------------------------------
# MLX port parity
# ---------------------------------------------------------------------------


def test_mlx_normalise_is_byte_exact():
    """The ported normalise step: MLX float32 divide agrees bit for bit."""

    rng = np.random.default_rng(0)
    values = rng.standard_normal((32, 4, 8, 8)).astype(np.float32)
    scales = (rng.random((32, 4)).astype(np.float32) + 0.5).astype(np.float32)
    reference = values / scales[:, :, None, None]
    got = np.asarray(mx.array(values) / mx.array(scales)[:, :, None, None])
    assert np.array_equal(reference.view(np.uint32), got.view(np.uint32))


def test_mlx_products_are_byte_exact():
    """The ported scale-update products: same order, same float32 rounding."""

    rng = np.random.default_rng(1)
    values = rng.standard_normal((16, 2, 4, 8)).astype(np.float32)
    hessian = rng.random((2, 4, 8)).astype(np.float32)
    reference = hessian[None, ...] * values * values
    got = np.asarray(mx.array(hessian)[None, ...] * mx.array(values) * mx.array(values))
    assert np.array_equal(reference.view(np.uint32), got.view(np.uint32))


def test_mlx_gather_is_byte_exact():
    """The ported codebook gather: mx.take moves entries, it does not compute."""

    table = e8p_full_grid().astype(np.float32)
    rng = np.random.default_rng(2)
    codes = rng.integers(0, table.shape[0], size=(8, 3, 4)).astype(np.uint16)
    reference = table[codes]
    got = np.asarray(mx.take(mx.array(table), mx.array(codes.astype(np.uint32)), axis=0))
    assert np.array_equal(reference.view(np.uint32), got.view(np.uint32))


@pytest.mark.parametrize("group_size", [128, 512])
def test_mlx_exact_fit_is_byte_identical_to_the_numpy_reference(group_size):
    """The whole default fit path, end to end, against the shipped reference."""

    rng = np.random.default_rng(5)
    weight = (rng.standard_normal((64, 512)) * 0.02).astype(np.float32)
    importance = ((rng.random(512) * 6 + 0.05) ** 2).astype(np.float32)
    reference = quantize_weight_importance_aware(
        weight,
        importance,
        group_size=group_size,
        code_bits=16,
        iterations=4,
        e8p_search_backend="metal",
    )
    got = quantize_weight_importance_aware_mlx(
        weight,
        importance,
        group_size=group_size,
        code_bits=16,
        iterations=4,
        scale_reduction="float64",
    )
    assert np.array_equal(got.codes, reference.codes)
    assert np.array_equal(
        np.asarray(got.scales).view(np.uint16), np.asarray(reference.scales).view(np.uint16)
    )
    assert got.codes.dtype == np.uint16
    assert got.group_size == group_size and got.code_bits == 16


def test_mlx_fp32_reduction_is_close_but_is_not_promised_to_be_identical():
    """The fast mode's contract: measured, bounded, and explicitly not exact.

    An fp32 tree reduction is not NumPy's float64 pairwise sum, so this asserts
    the *shape* of the honest claim -- the artifacts stay within a hair of each
    other -- and never that the bytes match.
    """

    rng = np.random.default_rng(6)
    weight = (rng.standard_normal((64, 512)) * 0.02).astype(np.float32)
    importance = ((rng.random(512) * 6 + 0.05) ** 2).astype(np.float32)
    comparison = compare_fit_backends(
        weight, importance, group_size=128, code_bits=16, iterations=4
    )
    exact = comparison["backends"]["mlx-exact"]
    fast = comparison["backends"]["mlx-fp32"]
    assert exact["codes_byte_identical"] is True
    assert exact["relative_mse_delta"] == 0.0
    assert fast["code_disagreement_fraction"] < 1e-3
    assert abs(fast["relative_mse_delta"]) < 1e-6


def test_mlx_fit_refuses_a_codebook_it_has_not_ported():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="code_bits=16"):
        quantize_weight_importance_aware_mlx(
            rng.standard_normal((8, 64)).astype(np.float32),
            np.ones(64, dtype=np.float32),
            group_size=32,
            code_bits=8,
        )


def test_mlx_fit_refuses_an_unknown_scale_reduction():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="scale_reduction must be one of"):
        quantize_weight_importance_aware_mlx(
            rng.standard_normal((8, 64)).astype(np.float32),
            np.ones(64, dtype=np.float32),
            group_size=32,
            scale_reduction="float16",
        )
    assert SCALE_REDUCTIONS == ("float64", "float32")


def test_resolve_fit_backend_downgrades_loudly_rather_than_silently():
    assert resolve_fit_backend("numpy", code_bits=16) == "numpy"
    assert resolve_fit_backend("mlx-exact", code_bits=16) == "mlx-exact"
    # code_bits=8 has no port, so the resolved backend -- the one the manifest
    # records -- is NumPy, not the requested MLX one.
    assert resolve_fit_backend("mlx-exact", code_bits=8) == "numpy"
    with pytest.raises(ValueError, match="unknown fit backend"):
        resolve_fit_backend("cuda", code_bits=16)
    assert FIT_BACKENDS == ("numpy", "mlx-exact", "mlx-fp32")


# ---------------------------------------------------------------------------
# Materializing a block
# ---------------------------------------------------------------------------


def test_materialized_block_writes_the_three_expected_files(toy_run):
    spec, record = _materialize(toy_run)
    for projection in PROJECTIONS:
        path = toy_run["output"] / spec.artifact_name(projection)
        assert path.is_file()
    assert [entry["projection"] for entry in record.files] == list(PROJECTIONS)
    assert [entry["checkpoint_tensor"] for entry in record.files] == ["w1", "w3", "w2"]
    assert record.importance_sources == {"measured": TOY_EXPERTS * 3}
    assert record.num_experts == TOY_EXPERTS


def test_materialized_tensors_load_through_the_shipped_vq_loader(toy_run):
    from mlx_vq.io.load import load_quantized_vq_switch_linear

    spec, _ = _materialize(toy_run)
    for projection in PROJECTIONS:
        linear = load_quantized_vq_switch_linear(
            toy_run["output"] / spec.artifact_name(projection),
            f"{spec.tensor_prefix}.{projection}",
        )
        assert linear.num_experts == TOY_EXPERTS
        assert linear.code_bits == 16
        assert linear.group_size == TOY_GROUP


def test_artifact_metadata_carries_the_codebook_hash_and_the_mapping(toy_run):
    from mlx_vq.io.load import inspect_safetensors

    spec, _ = _materialize(toy_run)
    inspection = inspect_safetensors(toy_run["output"] / spec.artifact_name("gate_proj"))
    config = json.loads(inspection.metadata["quantization_config"])
    assert config["codebook"]["sha256"] == E8P_PACKED_ABS_SHA256
    assert config["default_group_size"] == TOY_GROUP
    assert config["policy"]["source_revision"] == SOURCE_REVISION
    assert config["policy"]["block"] == "layers.0"
    assert json.loads(inspection.metadata["deepseek_v4_tensor_mapping"]) == TENSOR_MAPPING


def test_decoded_artifact_matches_the_fit_reconstruction_of_the_source(toy_run):
    """The artifact is the fit's own output, and the fit tracks the source."""

    from mlx_vq.convert.dsv4_vq_pilot import projection_error

    spec, _ = _materialize(toy_run)
    decoded = decode_artifact_experts(toy_run["output"], spec, range(TOY_EXPERTS))
    for projection, checkpoint_name in (
        ("gate_proj", "w1"),
        ("up_proj", "w3"),
        ("down_proj", "w2"),
    ):
        for expert in range(TOY_EXPERTS):
            source = toy_run["dense"][("layers.0", checkpoint_name, expert)]
            assert decoded[projection][expert].shape == source.shape
            space = "hidden" if projection != "down_proj" else "down"
            importance = toy_run["cache"]["importance"][space][0, expert].astype(np.float64)
            metrics = projection_error(source, decoded[projection][expert], importance)
            # A 2-bit fit of a random FP4 matrix: the assertion is that it is a
            # fit at all, not that it is tight.
            assert metrics.weighted_relative_mse < 0.5


def test_a_partial_expert_run_never_counts_as_a_complete_block(toy_run):
    spec = plan_blocks(only=["layers.0"])[0]
    record = materialize_block(
        spec,
        checkpoint_dir=toy_run["checkpoint"],
        output_dir=toy_run["output"],
        cache=toy_run["cache"],
        policy=TOY_POLICY,
        experts=[0, 1],
    )
    write_block_record(toy_run["output"], record)
    assert (
        verified_block_record(
            toy_run["output"], spec, policy=TOY_POLICY, expected_experts=TOY_EXPERTS
        )
        is None
    )


def test_drafter_block_materializes_with_the_fallback_importance(toy_run):
    spec, record = _materialize(toy_run, block="mtp.0")
    assert record.importance_sources == {"backbone_mean_layer_42": TOY_EXPERTS * 3}
    assert (toy_run["output"] / "mtp-00000-gate_proj.safetensors").is_file()
    codes_name, _ = spec.tensor_names("gate_proj")
    assert codes_name.startswith("mtp_drafter.blocks.0.")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_a_complete_block_verifies_and_is_skippable(toy_run):
    spec, _ = _materialize(toy_run)
    payload = verified_block_record(
        toy_run["output"], spec, policy=TOY_POLICY, expected_experts=TOY_EXPERTS
    )
    assert payload is not None and payload["key"] == "layers.0"


def test_a_missing_projection_file_invalidates_the_record(toy_run):
    spec, _ = _materialize(toy_run)
    (toy_run["output"] / spec.artifact_name("up_proj")).unlink()
    assert (
        verified_block_record(
            toy_run["output"], spec, policy=TOY_POLICY, expected_experts=TOY_EXPERTS
        )
        is None
    )


def test_a_tampered_projection_file_invalidates_the_record(toy_run):
    spec, _ = _materialize(toy_run)
    path = toy_run["output"] / spec.artifact_name("down_proj")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert (
        verified_block_record(
            toy_run["output"], spec, policy=TOY_POLICY, expected_experts=TOY_EXPERTS
        )
        is None
    )


def test_a_record_written_at_another_rate_invalidates_the_block(toy_run):
    spec, _ = _materialize(toy_run)
    other = MaterializePolicy(code_bits=16, group_size=64, iterations=3)
    assert (
        verified_block_record(
            toy_run["output"], spec, policy=other, expected_experts=TOY_EXPERTS
        )
        is None
    )


def test_a_corrupt_record_file_invalidates_the_block(toy_run):
    spec, _ = _materialize(toy_run)
    record_path(toy_run["output"], spec).write_text("{not json")
    assert (
        verified_block_record(
            toy_run["output"], spec, policy=TOY_POLICY, expected_experts=TOY_EXPERTS
        )
        is None
    )


def test_publication_leaves_no_partial_files_behind(toy_run):
    _materialize(toy_run)
    assert [path.name for path in toy_run["output"].glob(".*partial*")] == []


# ---------------------------------------------------------------------------
# Roundtrip: the naming contract, proven through the adapter's own bind path
# ---------------------------------------------------------------------------


def test_bind_roundtrip_proves_the_mapping_and_discriminates_a_swap(toy_run):
    """The whole verification, on a toy artifact, through the real bind path.

    The gate/up-swapped control is the point: it is shape-legal, it loads, and
    it produces finite output, so only comparing against both references turns
    "the artifact binds" into "the projections are in the right slots".
    """

    spec, _ = _materialize(toy_run)
    payload = verify_bind_roundtrip(
        toy_run["output"],
        spec,
        policy=TOY_POLICY,
        experts=range(TOY_EXPERTS),
        tokens=3,
        input_scale=0.05,
        hidden_size=TOY_HIDDEN,
        moe_intermediate_size=TOY_MOE,
        num_experts=TOY_EXPERTS,
    )
    assert payload["passed"] is True
    assert payload["bind_proof"]["bind_path"] == "binder"
    assert payload["bind_proof"]["file_discovery_exercised"] is True
    assert payload["bind_proof"]["gap"] is None
    assert payload["bind_proof"]["unbound_vq_experts"] is False
    assert payload["bind_proof"]["dense_routed_experts"] is False
    assert payload["bind_proof"]["dense_routed_parameter_names"] == []
    correct = payload["forward_agreement"]["correct_mapping"]
    swapped = payload["forward_agreement"]["gate_up_swapped_control"]
    assert correct["relative_mse"] < 1e-6
    assert swapped["relative_mse"] > correct["relative_mse"] * 100


def test_the_binder_path_refuses_a_drafter_block(toy_run):
    """No MTP binder exists, so the full-chain route must refuse, not improvise."""

    spec, _ = _materialize(toy_run, block="mtp.0")
    with pytest.raises(ValueError, match="binds backbone layers only"):
        verify_bind_roundtrip(
            toy_run["output"],
            spec,
            policy=TOY_POLICY,
            bind_path="binder",
            hidden_size=TOY_HIDDEN,
            moe_intermediate_size=TOY_MOE,
            num_experts=TOY_EXPERTS,
        )


def test_the_loader_path_verifies_a_drafter_block_and_names_its_gap(toy_run):
    """What IS verifiable for a drafter block, and what is not.

    The loader route proves the files load through the adapter's own loader, that
    ``bind_switch_mlp`` accepts their dimensions/expert count/clamp, and that the
    forward numerics match the fit with the projections in the right slots. It
    does not prove file *discovery*, because no MTP binder defines that yet -- so
    the payload says so instead of implying a stronger claim.
    """

    spec, _ = _materialize(toy_run, block="mtp.0")
    payload = verify_bind_roundtrip(
        toy_run["output"],
        spec,
        policy=TOY_POLICY,
        bind_path="loader",
        experts=range(TOY_EXPERTS),
        tokens=3,
        input_scale=0.05,
        hidden_size=TOY_HIDDEN,
        moe_intermediate_size=TOY_MOE,
        num_experts=TOY_EXPERTS,
    )
    assert payload["passed"] is True
    proof = payload["bind_proof"]
    assert proof["bind_path"] == "loader"
    assert proof["file_discovery_exercised"] is False
    assert "MTP binder" in proof["gap"]
    assert proof["unbound_vq_experts"] is False
    assert proof["dense_routed_experts"] is False
    swapped = payload["forward_agreement"]["gate_up_swapped_control"]
    correct = payload["forward_agreement"]["correct_mapping"]
    assert swapped["relative_mse"] > correct["relative_mse"] * 100


def test_the_loader_path_reaches_a_late_backbone_layer(toy_run):
    """A one-layer probe keeps `has_unbound_...` a genuine whole-model check."""

    spec, _ = _materialize(toy_run, block="layers.2")
    payload = verify_bind_roundtrip(
        toy_run["output"],
        spec,
        policy=TOY_POLICY,
        experts=range(TOY_EXPERTS),
        tokens=3,
        input_scale=0.05,
        hidden_size=TOY_HIDDEN,
        moe_intermediate_size=TOY_MOE,
        num_experts=TOY_EXPERTS,
    )
    assert payload["passed"] is True
    # auto resolves to the loader route for any layer but 0.
    assert payload["bind_proof"]["bind_path"] == "loader"
    assert payload["bind_proof"]["probe_layers"] == 1
    assert payload["bind_proof"]["unbound_vq_experts"] is False


# ---------------------------------------------------------------------------
# Manifest and audit
# ---------------------------------------------------------------------------


def test_manifest_carries_the_family_template_fields(toy_run):
    spec, record = _materialize(toy_run)
    manifest = build_manifest(
        output_dir=toy_run["output"],
        checkpoint_dir=toy_run["checkpoint"],
        specs=[spec],
        policy=TOY_POLICY,
        records=[record.as_dict()],
        calibration={"sha256": "0" * 64, "sessions": 40},
    )
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["family"] == FAMILY
    assert manifest["converter"] == CONVERTER_KIND
    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["source"]["index_sha256"] is not None
    assert manifest["seed_artifact"] is None
    assert manifest["seed_artifact_mutated"] is False
    assert manifest["codebook"]["sha256"] == E8P_PACKED_ABS_SHA256
    assert manifest["tensor_mapping"]["checkpoint_to_artifact"] == TENSOR_MAPPING
    assert manifest["policy"]["code_bits"] == 16
    audit = manifest["audit"]
    assert audit["missing_projection_files"] == []
    assert audit["symlinked_projection_count"] == 0
    assert audit["high_precision_routed_projection_count"] == 0
    assert audit["continuous_sidecar_count"] == 0
    assert audit["code_bits_distribution"] == {"16": 3}
    assert audit["group_size_distribution"] == {str(TOY_GROUP): 3}
    assert audit["effective_routed_bpw"] == pytest.approx(TOY_POLICY.bpw)
    # Counted off the codes actually written: 4 experts x (two [32,64] plus one
    # [64,32]) = 24576 logical weights.
    assert audit["routed_weight_count"] == TOY_EXPERTS * 3 * TOY_MOE * TOY_HIDDEN
    assert audit["artifact_payload_bytes"] > 0


def test_audit_reports_a_missing_projection_file(toy_run):
    spec, record = _materialize(toy_run)
    (toy_run["output"] / spec.artifact_name("gate_proj")).unlink()
    audit = audit_artifact_tree(
        toy_run["output"], specs=[spec], policy=TOY_POLICY, records=[record.as_dict()]
    )
    assert audit["missing_projection_files"] == ["layer-00000-gate_proj.safetensors"]
    assert audit["total_routed_projection_files_present"] == 2


def test_audit_cannot_claim_dense_or_unbound_without_a_bind_proof(toy_run):
    spec, record = _materialize(toy_run)
    audit = audit_artifact_tree(
        toy_run["output"], specs=[spec], policy=TOY_POLICY, records=[record.as_dict()]
    )
    # A file audit cannot see a dense fallback; refusing to answer is the
    # correct behaviour, and the manifest fills it from a real bind.
    assert audit["dense_routed_experts"] is None
    assert audit["unbound_vq_experts"] is None


def test_manifest_takes_dense_and_unbound_from_the_bind_proof(toy_run):
    spec, record = _materialize(toy_run)
    payload = verify_bind_roundtrip(
        toy_run["output"],
        spec,
        policy=TOY_POLICY,
        experts=range(TOY_EXPERTS),
        tokens=2,
        hidden_size=TOY_HIDDEN,
        moe_intermediate_size=TOY_MOE,
        num_experts=TOY_EXPERTS,
    )
    manifest = build_manifest(
        output_dir=toy_run["output"],
        checkpoint_dir=toy_run["checkpoint"],
        specs=[spec],
        policy=TOY_POLICY,
        records=[record.as_dict()],
        calibration={},
        bind_proof=payload["bind_proof"],
    )
    assert manifest["audit"]["dense_routed_experts"] is False
    assert manifest["audit"]["unbound_vq_experts"] is False
    assert manifest["audit"]["bind_proof"]["layers_bound"] == [0]


# ---------------------------------------------------------------------------
# Real-run evidence (skipped without the release on disk)
# ---------------------------------------------------------------------------


def _evidence(name: str) -> dict:
    path = RUN_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} not present; run the materializer CLI to produce it")
    return json.loads(path.read_text())


def test_real_layer_roundtrip_evidence_passed():
    payload = _evidence("roundtrip.json")
    assert payload["record_type"] == "dsv4_vq_roundtrip_v1"
    assert payload["passed"] is True
    assert payload["bind_proof"]["unbound_vq_experts"] is False
    assert payload["bind_proof"]["dense_routed_experts"] is False
    assert payload["forward_agreement"]["correct_mapping"]["relative_mse"] < 1e-9
    assert payload["forward_agreement"]["discrimination_ratio"] > 100


def test_real_weight_fit_parity_evidence_says_mlx_exact_is_byte_identical():
    payload = _evidence("fit-parity.json")
    summary = payload["summary"]
    assert summary["mlx-exact"]["all_codes_byte_identical"] is True
    assert summary["mlx-exact"]["all_scales_byte_identical"] is True
    assert summary["mlx-exact"]["max_code_disagreement_fraction"] == 0.0
    assert summary["mlx-exact"]["mean_speedup_vs_numpy"] > 1.5
    # The fast mode is the one that is not reproducible, and the evidence says so.
    assert summary["mlx-fp32"]["all_codes_byte_identical"] is False
