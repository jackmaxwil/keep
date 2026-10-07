from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

from mlx_vq.io.continuous_sidecar import (
    continuous_parameters_enabled,
    copy_declared_continuous_sidecars,
    load_switch_linear_continuous_sidecar,
    write_continuous_artifact_manifest,
    write_continuous_sidecar,
)
from mlx_vq.nn.switch_linear import QuantizedVQSwitchLinear


def _load_finetune_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "finetune_glm45_air_vq_continuous.py"
    )
    spec = importlib.util.spec_from_file_location("continuous_finetune_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loose_continuous_sidecar_is_ignored_without_manifest(tmp_path) -> None:
    output_bias = mx.zeros((2, 3), dtype=mx.float32)
    write_continuous_sidecar(
        output_dir=tmp_path,
        layer=41,
        projection="down_proj",
        output_bias=output_bias,
    )

    assert continuous_parameters_enabled(tmp_path) is False
    assert (
        load_switch_linear_continuous_sidecar(
            tmp_path,
            layer=41,
            projection="down_proj",
            scale_shape=(2, 3, 1),
            output_bias_shape=(2, 3),
        )
        is None
    )


def test_manifest_declared_continuous_sidecar_loads(tmp_path) -> None:
    scale_delta = mx.zeros((2, 3, 1), dtype=mx.float32)
    output_bias = mx.array(np.arange(6, dtype=np.float32).reshape(2, 3))
    sidecar = write_continuous_sidecar(
        output_dir=tmp_path,
        layer=41,
        projection="down_proj",
        scale_delta=scale_delta,
        output_bias=output_bias,
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=tmp_path / "seed",
        output_dir=tmp_path,
        sidecars=[sidecar],
    )

    assert continuous_parameters_enabled(tmp_path) is True
    loaded = load_switch_linear_continuous_sidecar(
        tmp_path,
        layer=41,
        projection="down_proj",
        scale_shape=(2, 3, 1),
        output_bias_shape=(2, 3),
    )

    assert loaded is not None
    assert loaded.scale_delta is not None
    assert loaded.output_bias is not None
    np.testing.assert_array_equal(np.array(loaded.output_bias), np.arange(6, dtype=np.float32).reshape(2, 3))


def test_manifest_declared_low_rank_sidecar_loads(tmp_path) -> None:
    left = mx.ones((2, 3, 2), dtype=mx.float32)
    right = mx.ones((2, 2, 4), dtype=mx.float32)
    sidecar = write_continuous_sidecar(
        output_dir=tmp_path,
        layer=41,
        projection="gate_proj",
        low_rank_left=left,
        low_rank_right=right,
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=tmp_path / "seed",
        output_dir=tmp_path,
        sidecars=[sidecar],
    )

    loaded = load_switch_linear_continuous_sidecar(
        tmp_path,
        layer=41,
        projection="gate_proj",
        scale_shape=(2, 3, 1),
        output_bias_shape=(2, 3),
    )

    assert loaded is not None
    assert loaded.low_rank_left is not None
    assert loaded.low_rank_right is not None
    np.testing.assert_array_equal(np.array(loaded.low_rank_left), np.ones((2, 3, 2), dtype=np.float32))
    assert sidecar["low_rank_rank"] == 2


def test_quantized_switch_linear_applies_low_rank_residual_to_selected_routes() -> None:
    layer = QuantizedVQSwitchLinear(
        input_dims=8,
        output_dims=3,
        num_experts=2,
        codes=mx.zeros((2, 3, 1), dtype=mx.uint8),
        scales=mx.zeros((2, 3, 1), dtype=mx.float32),
        group_size=8,
        code_bits=8,
        use_gather_vqmm=False,
    )
    left = mx.array(
        [
            [[1.0], [2.0], [3.0]],
            [[-1.0], [0.5], [1.5]],
        ],
        dtype=mx.float32,
    )
    right = mx.array(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    layer.set_continuous_sidecar(low_rank_left=left, low_rank_right=right)
    x = mx.array([[4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    indices = mx.array([[0, 1]], dtype=mx.int32)

    actual = layer(x, indices)
    mx.eval(actual)

    expected = np.array([[[4.0, 8.0, 12.0], [-6.0, 3.0, 9.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-6)


def test_manifest_declared_sidecar_rejects_shape_mismatch(tmp_path) -> None:
    sidecar = write_continuous_sidecar(
        output_dir=tmp_path,
        layer=41,
        projection="down_proj",
        output_bias=mx.zeros((2, 3), dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=tmp_path / "seed",
        output_dir=tmp_path,
        sidecars=[sidecar],
    )

    try:
        load_switch_linear_continuous_sidecar(
            tmp_path,
            layer=41,
            projection="down_proj",
            scale_shape=(2, 3, 1),
            output_bias_shape=(2, 4),
        )
    except ValueError as error:
        assert "output_bias must have shape" in str(error)
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def test_copy_declared_continuous_sidecars_preserves_seed_entries(tmp_path) -> None:
    seed = tmp_path / "seed"
    output = tmp_path / "output"
    old_sidecar = write_continuous_sidecar(
        output_dir=seed,
        layer=18,
        projection="gate_proj",
        output_bias=mx.ones((2, 3), dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=tmp_path / "base",
        output_dir=seed,
        sidecars=[old_sidecar],
    )

    preserved = copy_declared_continuous_sidecars(
        seed_artifact_dir=seed,
        output_dir=output,
        exclude={(45, "down_proj")},
    )
    new_sidecar = write_continuous_sidecar(
        output_dir=output,
        layer=45,
        projection="down_proj",
        output_bias=mx.zeros((2, 4), dtype=mx.float32),
    )
    write_continuous_artifact_manifest(
        seed_artifact_dir=seed,
        output_dir=output,
        sidecars=[*preserved, new_sidecar],
    )

    assert len(preserved) == 1
    assert (output / preserved[0]["path"]).exists()
    loaded_old = load_switch_linear_continuous_sidecar(
        output,
        layer=18,
        projection="gate_proj",
        scale_shape=(2, 3, 1),
        output_bias_shape=(2, 3),
    )
    loaded_new = load_switch_linear_continuous_sidecar(
        output,
        layer=45,
        projection="down_proj",
        scale_shape=(2, 4, 1),
        output_bias_shape=(2, 4),
    )

    assert loaded_old is not None and loaded_old.output_bias is not None
    assert loaded_new is not None and loaded_new.output_bias is not None
    np.testing.assert_array_equal(np.array(loaded_old.output_bias), np.ones((2, 3), dtype=np.float32))


def test_teacher_top1_margin_loss_is_zero_when_teacher_token_clears_margin() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[0.1, 3.0, 0.2], [4.0, 1.0, 0.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 2.0, 0.5], [3.0, 1.0, 0.0]], dtype=mx.float32)

    loss = cli._teacher_top1_margin_loss(vq_logits, teacher_logits, margin=0.5)
    mx.eval(loss)

    assert float(loss.item()) == 0.0


def test_teacher_top1_margin_loss_penalizes_hardest_competitor() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[0.1, 3.0, 0.2], [4.0, 1.0, 0.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 1.0, 2.0], [2.0, 3.0, 0.0]], dtype=mx.float32)

    loss = cli._teacher_top1_margin_loss(vq_logits, teacher_logits, margin=0.25)
    mx.eval(loss)

    np.testing.assert_allclose(float(loss.item()), 1.25, rtol=1e-6, atol=1e-6)


def test_teacher_top1_margin_loss_can_target_explicit_competitor_tokens() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[0.0, 4.0, 0.0, 0.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 2.0, 9.0, 5.0]], dtype=mx.float32)

    loss = cli._teacher_top1_margin_loss(
        vq_logits,
        teacher_logits,
        margin=0.25,
        competitor_token_ids=(3,),
    )
    mx.eval(loss)

    np.testing.assert_allclose(float(loss.item()), 3.25, rtol=1e-6, atol=1e-6)


def test_teacher_top1_margin_loss_can_add_explicit_and_hardest_competitor_terms() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[0.0, 4.0, 0.0, 0.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 2.0, 9.0, 5.0]], dtype=mx.float32)

    loss = cli._teacher_top1_margin_loss(
        vq_logits,
        teacher_logits,
        margin=0.25,
        competitor_token_ids=(3,),
        include_hardest_competitor=True,
    )
    mx.eval(loss)

    np.testing.assert_allclose(float(loss.item()), 10.5, rtol=1e-6, atol=1e-6)


def test_teacher_top1_margin_loss_can_target_multiple_explicit_competitors_per_position() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[0.0, 4.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 2.0, 9.0, 5.0, 4.0]], dtype=mx.float32)

    loss = cli._teacher_top1_margin_loss(
        vq_logits,
        teacher_logits,
        margin=0.25,
        competitor_token_ids=((3, 4),),
        include_hardest_competitor=True,
    )
    mx.eval(loss)

    np.testing.assert_allclose(float(loss.item()), 12.75, rtol=1e-6, atol=1e-6)


def test_parse_competitor_token_ids_supports_position_groups() -> None:
    cli = _load_finetune_cli()

    assert cli._parse_competitor_token_ids("220+82+102376,17") == ((220, 82, 102376), (17,))


def test_max_token_kld_loss_targets_tail_token_not_row_mean() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[3.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 3.0], [0.0, 3.0]], dtype=mx.float32)
    teacher_log_probs = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    vq_log_probs = vq_logits - mx.logsumexp(vq_logits, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)

    loss = cli._max_token_kld_loss(vq_log_probs, teacher_log_probs, teacher_probs)
    row_mean_loss = mx.mean(mx.sum(teacher_probs * (teacher_log_probs - vq_log_probs), axis=-1))
    mx.eval(loss, row_mean_loss)

    assert float(loss.item()) > float(row_mean_loss.item())
    np.testing.assert_allclose(float(loss.item()), 2.7154448, rtol=1e-6, atol=1e-6)


def test_auxiliary_loss_position_filter_limits_margin_to_selected_token() -> None:
    cli = _load_finetune_cli()
    teacher_logits = mx.array([[0.1, 3.0, 0.2], [4.0, 1.0, 0.0]], dtype=mx.float32)
    vq_logits = mx.array([[0.0, 2.0, 0.5], [2.0, 3.0, 0.0]], dtype=mx.float32)

    full_loss = cli._teacher_top1_margin_loss(vq_logits, teacher_logits, margin=0.25)
    first_vq, first_teacher = cli._filter_logits_by_position_indices(
        vq_logits,
        teacher_logits,
        (0,),
    )
    first_loss = cli._teacher_top1_margin_loss(first_vq, first_teacher, margin=0.25)
    second_vq, second_teacher = cli._filter_logits_by_position_indices(
        vq_logits,
        teacher_logits,
        (1,),
    )
    second_loss = cli._teacher_top1_margin_loss(second_vq, second_teacher, margin=0.25)
    mx.eval(full_loss, first_loss, second_loss)

    np.testing.assert_allclose(float(full_loss.item()), 0.625, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(first_loss.item()), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(second_loss.item()), 1.25, rtol=1e-6, atol=1e-6)


def test_parse_aux_loss_position_indices_accepts_comma_separated_indices() -> None:
    cli = _load_finetune_cli()

    assert cli._parse_position_indices("0, 2,3") == (0, 2, 3)
    assert cli._parse_position_indices(None) is None


def test_parse_aux_loss_position_indices_rejects_negative_indices() -> None:
    cli = _load_finetune_cli()

    try:
        cli._parse_position_indices("-1")
    except ValueError as error:
        assert "position indices must be non-negative" in str(error)
    else:
        raise AssertionError("negative position index should raise")


def test_auxiliary_loss_position_filter_rejects_out_of_range_indices() -> None:
    cli = _load_finetune_cli()
    logits = mx.zeros((2, 3), dtype=mx.float32)

    try:
        cli._filter_logits_by_position_indices(logits, logits, (2,))
    except ValueError as error:
        assert "position index 2 is out of range" in str(error)
    else:
        raise AssertionError("out-of-range position index should raise")


def test_selected_layer_logits_can_train_non_final_layer_and_continue_downstream() -> None:
    cli = _load_finetune_cli()
    calls: list[str] = []

    class FakeProjection:
        def __init__(self, output_dims: int) -> None:
            self.output_dims = output_dims

        def __call__(self, x, indices):
            route_count = int(indices.shape[-1])
            if x.ndim == 2:
                return mx.ones((x.shape[0], route_count, self.output_dims), dtype=mx.float32)
            return mx.ones((x.shape[0], route_count, self.output_dims), dtype=mx.float32) * 2.0

    class FakeSwitchMLP:
        def __init__(self) -> None:
            self.gate_proj = FakeProjection(2)
            self.up_proj = FakeProjection(2)
            self.down_proj = FakeProjection(2)

    class FakeMoE:
        def __init__(self) -> None:
            self.switch_mlp = FakeSwitchMLP()

        def route(self, x):
            return (
                mx.zeros((x.shape[0], 1), dtype=mx.int32),
                mx.ones((x.shape[0], 1), dtype=mx.float32),
            )

        def get(self, _name):
            return None

    class FakeLayer:
        def __init__(self, name: str, *, target: bool = False) -> None:
            self.name = name
            self.mlp = FakeMoE() if target else None

        def input_layernorm(self, h):
            calls.append(f"{self.name}-input-ln")
            return h

        def self_attn(self, h, _mask, _cache):
            calls.append(f"{self.name}-attn")
            return h + 1

        def post_attention_layernorm(self, h):
            calls.append(f"{self.name}-post-ln")
            return h

        def __call__(self, h, _mask, cache=None):
            calls.append(f"{self.name}-full")
            return h + 10

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = [
                FakeLayer("layer0"),
                FakeLayer("layer1", target=True),
                FakeLayer("layer2", target=True),
            ]

        def embed_tokens(self, tokens):
            return mx.zeros((tokens.shape[0], tokens.shape[1], 2), dtype=mx.float32)

        def norm(self, h):
            calls.append("norm")
            return h

    class FakeModel:
        def __init__(self) -> None:
            self.model = FakeLanguageModel()
            self.layers = self.model.layers

        def lm_head(self, h):
            calls.append("lm_head")
            return h

    original_moe_type = cli.GLM45AirVQMoE
    cli.GLM45AirVQMoE = FakeMoE
    try:
        row = cli.PreparedTeacherRow(
            row_index=0,
            prompt_id="fake",
            input_token_ids=(1, 2, 3),
            positions=(1,),
            target_token_ids=(2,),
            teacher_logits=mx.zeros((1, 2), dtype=mx.float32),
        )
        logits = cli._selected_logits_selected_layer(
            FakeModel(),
            row,
            layer=1,
            surrogate_projections=frozenset(),
            output_chunk_size=8,
        )
        mx.eval(logits)
    finally:
        cli.GLM45AirVQMoE = original_moe_type

    assert calls == [
        "layer0-full",
        "layer1-input-ln",
        "layer1-attn",
        "layer1-post-ln",
        "layer2-input-ln",
        "layer2-attn",
        "layer2-post-ln",
        "norm",
        "lm_head",
    ]
    np.testing.assert_allclose(np.array(logits), [[49.0, 49.0]], rtol=1e-6)


def test_parse_row_indices_accepts_comma_separated_indices() -> None:
    cli = _load_finetune_cli()

    assert cli._parse_row_indices("43, 67,71") == (43, 67, 71)


def test_parse_row_indices_rejects_negative_indices() -> None:
    cli = _load_finetune_cli()

    try:
        cli._parse_row_indices("1,-2")
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative row index should fail")


def test_select_rows_preserves_explicit_indices() -> None:
    cli = _load_finetune_cli()
    rows = [
        {"prompt_id": "first"},
        {"prompt_id": "second"},
        {"prompt_id": "third"},
    ]

    selected = cli._select_rows(rows, max_rows=1, row_indices=(2, 0))

    assert selected == [(2, rows[2]), (0, rows[0])]


def test_resolve_trainable_projections_defaults_to_single_projection() -> None:
    cli = _load_finetune_cli()

    assert cli._resolve_trainable_projections("down_proj", None) == ("down_proj",)


def test_resolve_trainable_projections_uses_multi_projection_override() -> None:
    cli = _load_finetune_cli()

    assert cli._resolve_trainable_projections(
        "down_proj",
        ["gate_proj", "up_proj", "down_proj"],
    ) == ("gate_proj", "up_proj", "down_proj")


def test_resolve_trainable_projections_rejects_duplicates() -> None:
    cli = _load_finetune_cli()

    try:
        cli._resolve_trainable_projections("down_proj", ["gate_proj", "gate_proj"])
    except ValueError as error:
        assert "duplicate projection" in str(error)
    else:
        raise AssertionError("duplicate projections should fail")


def test_apply_updates_returns_per_projection_norms() -> None:
    cli = _load_finetune_cli()
    params = {
        "gate_proj": mx.array([1.0, 2.0], dtype=mx.float32),
        "up_proj": mx.array([3.0], dtype=mx.float32),
    }
    grads = {
        "gate_proj": mx.array([0.5, 0.0], dtype=mx.float32),
        "up_proj": mx.array([1.0], dtype=mx.float32),
    }

    updated, norms = cli._apply_updates(
        params,
        grads,
        learning_rate=0.1,
        grad_clip_norm=None,
    )
    mx.eval(updated, norms)

    np.testing.assert_allclose(np.array(updated["gate_proj"]), [0.95, 2.0], rtol=1e-6)
    np.testing.assert_allclose(np.array(updated["up_proj"]), [2.9], rtol=1e-6)
    np.testing.assert_allclose(float(norms["gate_proj"].item()), 0.5, rtol=1e-6)
    np.testing.assert_allclose(float(norms["up_proj"].item()), 1.0, rtol=1e-6)


def test_low_rank_trainable_params_initialize_and_assign_to_projection() -> None:
    cli = _load_finetune_cli()
    projection = QuantizedVQSwitchLinear(
        input_dims=8,
        output_dims=3,
        num_experts=2,
        codes=mx.zeros((2, 3, 1), dtype=mx.uint8),
        scales=mx.ones((2, 3, 1), dtype=mx.float32),
        group_size=8,
        code_bits=8,
        use_gather_vqmm=False,
    )

    params, initialized = cli._initial_trainable_params(
        {"gate_proj": projection},
        trainable="low_rank_residual",
        low_rank=2,
        low_rank_init_scale=1.0e-3,
    )
    cli._assign_trainable_params(
        {"gate_proj": projection},
        params,
        trainable="low_rank_residual",
    )

    assert set(params) == {"gate_proj.low_rank_left", "gate_proj.low_rank_right"}
    assert initialized == {"gate_proj.low_rank_left": False, "gate_proj.low_rank_right": False}
    assert params["gate_proj.low_rank_left"].shape == (2, 3, 2)
    assert params["gate_proj.low_rank_right"].shape == (2, 2, 8)
    assert projection.get("continuous_low_rank_left") is not None
    assert projection.get("continuous_low_rank_right") is not None


def test_low_rank_trainable_params_can_reinitialize_existing_sidecar() -> None:
    cli = _load_finetune_cli()
    projection = QuantizedVQSwitchLinear(
        input_dims=8,
        output_dims=3,
        num_experts=2,
        codes=mx.zeros((2, 3, 1), dtype=mx.uint8),
        scales=mx.ones((2, 3, 1), dtype=mx.float32),
        group_size=8,
        code_bits=8,
        use_gather_vqmm=False,
    )
    existing_left = mx.ones((2, 3, 4), dtype=mx.float32)
    existing_right = mx.ones((2, 4, 8), dtype=mx.float32)
    projection.set_continuous_sidecar(low_rank_left=existing_left, low_rank_right=existing_right)

    inherited, inherited_initialized = cli._initial_trainable_params(
        {"gate_proj": projection},
        trainable="low_rank_residual",
        low_rank=2,
        low_rank_init_scale=1.0e-3,
    )
    reset, reset_initialized = cli._initial_trainable_params(
        {"gate_proj": projection},
        trainable="low_rank_residual",
        low_rank=2,
        low_rank_init_scale=1.0e-3,
        reinitialize_existing_sidecars=True,
    )

    assert inherited_initialized == {"gate_proj.low_rank_left": True, "gate_proj.low_rank_right": True}
    assert inherited["gate_proj.low_rank_left"].shape == (2, 3, 4)
    assert inherited["gate_proj.low_rank_right"].shape == (2, 4, 8)
    assert reset_initialized == {"gate_proj.low_rank_left": False, "gate_proj.low_rank_right": False}
    assert reset["gate_proj.low_rank_left"].shape == (2, 3, 2)
    assert reset["gate_proj.low_rank_right"].shape == (2, 2, 8)
