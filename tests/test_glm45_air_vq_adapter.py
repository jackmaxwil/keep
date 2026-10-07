from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models.glm4_moe import ModelArgs

from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.io.logit_bias import write_logit_bias_artifact_manifest, write_logit_bias_sidecar
from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from mlx_vq.models.glm45_air_vq_adapter import (
    GLM45AirVQModel,
    GLM45AirVQMoE,
    bind_glm45_air_non_expert_precision_policy,
    bind_glm45_air_non_expert_weights,
    bind_glm45_air_vq_experts,
    glm45_air_args_from_config,
    glm45_air_non_expert_dtype_report,
    has_dense_glm45_air_routed_expert_parameters,
    has_unbound_glm45_air_vq_experts,
    set_glm45_air_switch_gather_vqmm,
)


def _tiny_args() -> ModelArgs:
    return ModelArgs(
        model_type="glm4_moe",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=32,
        moe_intermediate_size=8,
        norm_topk_prob=True,
        num_attention_heads=2,
        n_group=1,
        head_dim=8,
        topk_group=1,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        num_hidden_layers=3,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        use_qk_norm=False,
        tie_word_embeddings=False,
        attention_bias=False,
        partial_rotary_factor=0.25,
    )


def _write_tiny_vq_switch_artifacts(path: Path, layers: tuple[int, ...]) -> None:
    rng = np.random.default_rng(4501)
    path.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        gate = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
        up = mx.array(rng.normal(scale=0.04, size=(2, 8, 16)).astype(np.float32))
        down = mx.array(rng.normal(scale=0.04, size=(2, 16, 8)).astype(np.float32))
        switch = QuantizedVQSwitchGLU.from_weights(
            gate_weight=gate,
            up_weight=up,
            down_weight=down,
            group_size=8,
        )
        prefix = f"model.layers.{layer}.mlp.switch_mlp"
        for projection, linear in {
            "gate_proj": switch.gate_proj,
            "up_proj": switch.up_proj,
            "down_proj": switch.down_proj,
        }.items():
            mx.save_safetensors(
                str(path / f"layer-{layer:05d}-{projection}.safetensors"),
                {
                    f"{prefix}.{projection}.codes": linear.codes,
                    f"{prefix}.{projection}.scales": linear.scales,
                    "model.vq_codebook.e8": linear.codebook,
                },
            )


def _write_tiny_non_expert_source_checkpoint(path: Path, model: GLM45AirVQModel) -> Path:
    params = dict(tree_flatten(model.parameters()))
    arrays = {}
    weight_map = {}
    shard_name = "model-00001-of-00001.safetensors"

    for key, value in params.items():
        if ".mlp.switch_mlp." in key:
            continue
        arrays[key] = value.astype(mx.bfloat16) if value.dtype in (mx.float32, mx.float16) else value
        weight_map[key] = shard_name

    for layer_idx in (1, 2):
        for expert in range(model.args.n_routed_experts):
            for projection, shape in {
                "gate_proj": (8, 16),
                "up_proj": (8, 16),
                "down_proj": (16, 8),
            }.items():
                key = f"model.layers.{layer_idx}.mlp.experts.{expert}.{projection}.weight"
                arrays[key] = mx.zeros(shape, dtype=mx.bfloat16)
                weight_map[key] = shard_name

    mtp_key = "model.layers.3.mlp.experts.0.gate_proj.weight"
    arrays[mtp_key] = mx.zeros((8, 16), dtype=mx.bfloat16)
    weight_map[mtp_key] = shard_name

    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return index_path


def _load_non_expert_precision_materializer_module():
    module_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "materialize_glm45_air_non_expert_precision.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_glm45_air_non_expert_precision", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tiny_resident_air(
    *,
    source_dir: Path,
    index_path: Path,
    artifact_dir: Path,
) -> tuple[GLM45AirVQModel, object]:
    model = GLM45AirVQModel(_tiny_args())
    report = bind_glm45_air_non_expert_precision_policy(
        model,
        artifact_dir,
        source_dir,
        load_safetensors_index(index_path),
    )
    bind_glm45_air_vq_experts(model, artifact_dir)
    set_glm45_air_switch_gather_vqmm(model, False)
    mx.eval(model.parameters())
    return model, report


def test_glm45_air_vq_model_omits_dense_routed_expert_parameters() -> None:
    model = GLM45AirVQModel(_tiny_args())

    assert not has_dense_glm45_air_routed_expert_parameters(model)
    assert has_unbound_glm45_air_vq_experts(model)
    assert isinstance(model.model.layers[1].mlp, GLM45AirVQMoE)

    params = dict(tree_flatten(model.parameters()))
    assert not any(".mlp.experts." in key for key in params)
    assert not any(".mlp.switch_mlp." in key for key in params)


def test_glm45_air_args_from_config_filters_extra_hf_fields() -> None:
    args = _tiny_args()
    config = dict(args.__dict__)
    config["torch_dtype"] = "bfloat16"
    config["architectures"] = ["Glm4MoeForCausalLM"]

    rebuilt = glm45_air_args_from_config(config)

    assert rebuilt.model_type == "glm4_moe"
    assert rebuilt.num_hidden_layers == 3


def test_glm45_air_non_expert_bind_skips_experts_and_mtp(tmp_path) -> None:
    source_model = GLM45AirVQModel(_tiny_args())
    index_path = _write_tiny_non_expert_source_checkpoint(tmp_path, source_model)
    target_model = GLM45AirVQModel(_tiny_args())

    report = bind_glm45_air_non_expert_weights(
        target_model,
        tmp_path,
        load_safetensors_index(index_path),
    )
    mx.eval(target_model.parameters())

    params = dict(tree_flatten(target_model.parameters()))
    assert report.missing_model_parameters == ()
    assert len(report.skipped_routed_expert_tensors) == 12
    assert report.skipped_mtp_tensors == ("model.layers.3.mlp.experts.0.gate_proj.weight",)
    assert params["model.layers.0.self_attn.q_proj.weight"].dtype == mx.bfloat16


def test_glm45_air_non_expert_precision_policy_binds_declared_surfaces(tmp_path) -> None:
    source_model = GLM45AirVQModel(_tiny_args())
    index_path = _write_tiny_non_expert_source_checkpoint(tmp_path, source_model)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "non_expert_precision": {
                    "schema_version": 1,
                    "enabled": True,
                    "surfaces": ["embed_tokens", "lm_head", "router_gates"],
                    "dtype": "bf16",
                }
            }
        ),
        encoding="utf-8",
    )
    target_model = GLM45AirVQModel(_tiny_args())

    report = bind_glm45_air_non_expert_precision_policy(
        target_model,
        artifact,
        tmp_path,
        load_safetensors_index(index_path),
    )
    mx.eval(target_model.parameters())

    params = dict(tree_flatten(target_model.parameters()))
    assert report.requested_surfaces == ("embed_tokens", "lm_head", "router_gates")
    assert report.missing_model_parameters == ()
    assert params["model.embed_tokens.weight"].dtype == mx.bfloat16
    assert params["lm_head.weight"].dtype == mx.bfloat16
    assert params["model.layers.1.mlp.gate.weight"].dtype == mx.bfloat16
    assert params["model.layers.1.self_attn.q_proj.weight"].dtype == mx.bfloat16
    assert "model.layers.1.self_attn.q_proj.weight" in report.loaded_model_parameters
    assert report.skipped_by_surface_tensors == ()


def test_glm45_air_non_expert_precision_policy_preserves_unrequested_source_weights(
    tmp_path,
) -> None:
    source_model = GLM45AirVQModel(_tiny_args())
    index_path = _write_tiny_non_expert_source_checkpoint(tmp_path, source_model)
    source_arrays = mx.load(str(tmp_path / "model-00001-of-00001.safetensors"))
    source_arrays["model.layers.1.self_attn.q_proj.weight"] = mx.full(
        source_arrays["model.layers.1.self_attn.q_proj.weight"].shape,
        0.75,
        dtype=mx.bfloat16,
    )
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), source_arrays)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "non_expert_precision": {
                    "schema_version": 1,
                    "enabled": True,
                    "surfaces": ["lm_head"],
                    "dtype": "bf16",
                }
            }
        ),
        encoding="utf-8",
    )
    target_model = GLM45AirVQModel(_tiny_args())

    report = bind_glm45_air_non_expert_precision_policy(
        target_model,
        artifact,
        tmp_path,
        load_safetensors_index(index_path),
    )
    mx.eval(target_model.parameters())

    params = dict(tree_flatten(target_model.parameters()))
    loaded = params["model.layers.1.self_attn.q_proj.weight"]
    assert report.requested_surfaces == ("lm_head",)
    assert "model.layers.1.self_attn.q_proj.weight" in report.loaded_model_parameters
    assert "model.layers.1.self_attn.q_proj.weight" not in report.skipped_by_surface_tensors
    assert loaded.dtype == mx.bfloat16
    assert float(mx.mean(loaded).item()) == 0.75


def test_glm45_air_non_expert_precision_materialized_policy_preserves_one_row_logits(
    tmp_path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    index_path = _write_tiny_non_expert_source_checkpoint(
        source_dir,
        GLM45AirVQModel(_tiny_args()),
    )
    seed_artifact = tmp_path / "seed-artifact"
    _write_tiny_vq_switch_artifacts(seed_artifact, layers=(1, 2))
    (seed_artifact / "conversion-manifest.json").write_text("{}", encoding="utf-8")
    candidate_artifact = tmp_path / "candidate-artifact"

    materializer = _load_non_expert_precision_materializer_module()
    materializer.materialize_non_expert_precision_candidate(
        seed_artifact_dir=seed_artifact,
        output_dir=candidate_artifact,
        surfaces=["lm_head"],
        dtype="bf16",
        reason="unit-test one-row parity",
    )

    baseline_model, baseline_report = _load_tiny_resident_air(
        source_dir=source_dir,
        index_path=index_path,
        artifact_dir=seed_artifact,
    )
    candidate_model, candidate_report = _load_tiny_resident_air(
        source_dir=source_dir,
        index_path=index_path,
        artifact_dir=candidate_artifact,
    )
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    baseline_logits = baseline_model(tokens).astype(mx.float32)
    candidate_logits = candidate_model(tokens).astype(mx.float32)
    mx.eval(baseline_logits, candidate_logits)

    assert baseline_report.requested_surfaces == ()
    assert candidate_report.requested_surfaces == ("lm_head",)
    assert set(candidate_report.loaded_model_parameters) == set(
        baseline_report.loaded_model_parameters
    )
    np.testing.assert_allclose(
        np.array(candidate_logits),
        np.array(baseline_logits),
        rtol=0,
        atol=0,
    )


def test_glm45_air_non_expert_dtype_report_verifies_bound_bfloat16(tmp_path) -> None:
    source_model = GLM45AirVQModel(_tiny_args())
    index_path = _write_tiny_non_expert_source_checkpoint(tmp_path, source_model)
    target_model = GLM45AirVQModel(_tiny_args())
    bind_glm45_air_non_expert_weights(target_model, tmp_path, load_safetensors_index(index_path))

    report = glm45_air_non_expert_dtype_report(target_model, expected_dtype_name="bfloat16")

    assert report["verified"] is True
    assert report["expected_dtype_name"] == "bfloat16"
    assert report["unexpected"] == []


def test_glm45_air_non_expert_dtype_report_allows_source_router_float32_bias() -> None:
    class FakeModel:
        def parameters(self):
            return {
                "model": {
                    "embed_tokens": {"weight": mx.zeros((2, 2), dtype=mx.bfloat16)},
                    "layers": {
                        "1": {
                            "mlp": {
                                "gate": {
                                    "e_score_correction_bias": mx.zeros((2,), dtype=mx.float32)
                                }
                            }
                        }
                    },
                }
            }

    report = glm45_air_non_expert_dtype_report(FakeModel(), expected_dtype_name="bfloat16")

    assert report["verified"] is True
    assert report["unexpected"] == []
    assert report["allowed_non_expected"] == [
        {"name": "model.layers.1.mlp.gate.e_score_correction_bias", "dtype": "float32"}
    ]


def test_glm45_air_vq_model_binds_artifact_switches_and_runs_tiny_forward(tmp_path) -> None:
    model = GLM45AirVQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))

    bound = bind_glm45_air_vq_experts(model, tmp_path)
    assert bound == (1, 2)
    assert not has_unbound_glm45_air_vq_experts(model)

    logits = model(mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(logits)

    assert logits.shape == (1, 2, 32)
    assert bool(mx.all(mx.isfinite(logits)).item())


def test_glm45_air_vq_model_applies_declared_logit_bias_sidecar(tmp_path) -> None:
    inputs = mx.array([[1, 2]], dtype=mx.int32)
    model = GLM45AirVQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    bind_glm45_air_vq_experts(model, tmp_path)
    baseline_logits = model(inputs)
    mx.eval(baseline_logits)

    sidecar = write_logit_bias_sidecar(
        output_dir=tmp_path,
        token_biases={7: 3.0, 11: -0.5},
    )
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=tmp_path,
        output_dir=tmp_path,
        sidecar=sidecar,
        run_manifest={"kind": "tiny-test"},
    )
    bind_glm45_air_vq_experts(model, tmp_path)
    biased_logits = model(inputs)
    mx.eval(biased_logits)

    delta = np.array(biased_logits - baseline_logits)
    np.testing.assert_allclose(delta[:, :, 7], 3.0, atol=1e-6)
    np.testing.assert_allclose(delta[:, :, 11], -0.5, atol=1e-6)
    delta[:, :, [7, 11]] = 0.0
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_glm45_air_vq_model_applies_position_scoped_logit_bias(tmp_path) -> None:
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    model = GLM45AirVQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    bind_glm45_air_vq_experts(model, tmp_path)
    baseline_logits = model(inputs)
    mx.eval(baseline_logits)

    sidecar = write_logit_bias_sidecar(
        output_dir=tmp_path,
        token_biases={7: 3.0},
        position_indices=(0,),
    )
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=tmp_path,
        output_dir=tmp_path,
        sidecar=sidecar,
        run_manifest={"kind": "tiny-test", "scope": {"position_indices": [0]}},
    )
    bind_glm45_air_vq_experts(model, tmp_path)
    biased_logits = model(inputs)
    mx.eval(biased_logits)

    delta = np.array(biased_logits - baseline_logits)
    np.testing.assert_allclose(delta[:, 0, 7], 3.0, atol=1e-6)
    np.testing.assert_allclose(delta[:, 1:, 7], 0.0, atol=1e-6)
    delta[:, 0, 7] = 0.0
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_glm45_air_vq_model_applies_token_position_scoped_logit_bias(tmp_path) -> None:
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    model = GLM45AirVQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    bind_glm45_air_vq_experts(model, tmp_path)
    baseline_logits = model(inputs)
    mx.eval(baseline_logits)

    sidecar = write_logit_bias_sidecar(
        output_dir=tmp_path,
        token_biases={7: 3.0, 11: -0.5},
        token_position_indices={7: (0,), 11: (2, 3)},
    )
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=tmp_path,
        output_dir=tmp_path,
        sidecar=sidecar,
        run_manifest={
            "kind": "tiny-test",
            "scope": {"token_position_indices": [[0], [2, 3]]},
        },
    )
    bind_glm45_air_vq_experts(model, tmp_path)
    biased_logits = model(inputs)
    mx.eval(biased_logits)

    delta = np.array(biased_logits - baseline_logits)
    np.testing.assert_allclose(delta[:, 0, 7], 3.0, atol=1e-6)
    np.testing.assert_allclose(delta[:, 1:, 7], 0.0, atol=1e-6)
    np.testing.assert_allclose(delta[:, :2, 11], 0.0, atol=1e-6)
    np.testing.assert_allclose(delta[:, 2:, 11], -0.5, atol=1e-6)
    delta[:, 0, 7] = 0.0
    delta[:, 2:, 11] = 0.0
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_glm45_air_vq_model_keeps_logit_bias_in_lm_head_dtype(tmp_path) -> None:
    model = GLM45AirVQModel(_tiny_args())
    model.lm_head.weight = model.lm_head.weight.astype(mx.bfloat16)
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    sidecar = write_logit_bias_sidecar(
        output_dir=tmp_path,
        token_biases={7: 3.0},
    )
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=tmp_path,
        output_dir=tmp_path,
        sidecar=sidecar,
        run_manifest={"kind": "tiny-test"},
    )

    bind_glm45_air_vq_experts(model, tmp_path)

    assert model.final_logit_bias_token_ids is not None
    assert model.final_logit_bias_values is not None
    assert model.final_logit_bias_token_ids.shape == (1,)
    assert model.final_logit_bias_values.dtype == mx.bfloat16


def test_glm45_air_switch_gather_flag_can_roll_back_to_scalar(tmp_path) -> None:
    model = GLM45AirVQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    bind_glm45_air_vq_experts(model, tmp_path)

    sparse_mlp = model.model.layers[1].mlp
    assert isinstance(sparse_mlp, GLM45AirVQMoE)
    assert sparse_mlp.switch_mlp is not None
    assert sparse_mlp.switch_mlp.gate_proj.route_backend == "gather_vqmm_auto"

    set_glm45_air_switch_gather_vqmm(model, False)

    assert sparse_mlp.switch_mlp.gate_proj.route_backend == "scalar_qmv"
    assert sparse_mlp.switch_mlp.up_proj.route_backend == "scalar_qmv"
    assert sparse_mlp.switch_mlp.down_proj.route_backend == "scalar_qmv"

    set_glm45_air_switch_gather_vqmm(model, True)

    assert sparse_mlp.switch_mlp.gate_proj.route_backend == "gather_vqmm_auto"


class _FakeProfileRecorder:
    eval_outputs = True

    def __init__(self):
        self.names: list[str] = []

    def time(self, name: str):
        recorder = self

        class _Timer:
            def __enter__(self):
                recorder.names.append(name)

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Timer()


def test_glm45_air_vq_profile_recorder_sees_attention_and_moe_sections(tmp_path) -> None:
    model = GLM45AirVQModel(_tiny_args())
    _write_tiny_vq_switch_artifacts(tmp_path, layers=(1, 2))
    bind_glm45_air_vq_experts(model, tmp_path)
    recorder = _FakeProfileRecorder()

    model.set_profile_recorder(recorder)
    logits = model(mx.array([[1, 2]], dtype=mx.int32))
    mx.eval(logits)

    assert "layer.0.attention" in recorder.names
    assert "layer.1.moe.router" in recorder.names
    assert "layer.1.moe.gate_proj" in recorder.names
    assert "layer.1.moe.up_proj" in recorder.names
    assert "layer.1.moe.down_proj" in recorder.names
    assert "layer.1.moe.weighted_reduce" in recorder.names
