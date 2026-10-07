from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5_moe

from benchmarks.probe_qwen_moe_binding import probe_qwen_moe_binding_artifact
from benchmarks.probe_qwen_non_expert_binding import probe_qwen_non_expert_binding_source
from keep.vq.e8 import cosine_similarity
from keep.convert.stream_convert import load_safetensors_index
from keep.convert.qwen_moe import (
    QWEN36_35B_A3B_MODEL_ID,
    QwenMoeConversionGroup,
    QwenMoeConversionPlan,
    convert_qwen_moe_group_from_safetensors,
    convert_qwen_moe_groups_from_safetensors,
    qwen_moe_group_output_filename,
)
from ramp.models.qwen_moe_adapter import (
    bind_qwen_non_expert_weights,
    bind_qwen_moe_vq_experts,
    has_unbound_qwen_moe_vq_experts,
    load_qwen_moe_switch_glu,
)
from keep.validate.qwen_vq import (
    DenseDequantLinear,
    audit_qwen_moe_materialization_manifest,
    get_module_by_path,
    load_qwen_moe_switch_projection,
    make_vq_and_dense_reference,
    qwen_moe_switch_prefix,
    select_linear_paths,
    set_module_by_path,
    validate_qwen_moe_switch_projection,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = {"q_proj": nn.Linear(16, 8, bias=True)}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = {"layers": [TinyBlock()]}


class TinyQwenMlp:
    def __init__(self) -> None:
        self.switch_mlp = None

    def bind_switch_mlp(self, switch_mlp) -> None:
        self.switch_mlp = switch_mlp


class TinyQwenLayer:
    def __init__(self) -> None:
        self.mlp = TinyQwenMlp()


class TinyQwenLanguageModel:
    def __init__(self, layer_count: int) -> None:
        self.layers = [TinyQwenLayer() for _ in range(layer_count)]


class TinyQwenMoEModel:
    def __init__(self, layer_count: int) -> None:
        self.language_model = TinyQwenLanguageModel(layer_count)


class TinyQwenNorm(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.weight = mx.zeros((dims,), dtype=mx.float32)


class TinyQwenLinearAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(3, 2, bias=False)


class TinyQwenDenseMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(3, 1, bias=False)


class TinyQwenConv1dContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            groups=2,
            bias=False,
        )


class TinyQwenRuntimeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = TinyQwenNorm(3)
        self.linear_attn = TinyQwenLinearAttention()
        self.mlp = TinyQwenDenseMlp()


class TinyQwenRuntimeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)
        self.layers = [TinyQwenRuntimeLayer()]
        self.norm = TinyQwenNorm(3)


class TinyQwenRuntimeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = TinyQwenRuntimeLanguageModel()
        self.lm_head = nn.Linear(3, 4, bias=False)


class TinyQwenSanitizingRuntimeModel(TinyQwenRuntimeModel):
    def __init__(self) -> None:
        super().__init__()
        self.language_model.layers[0].linear_attn = TinyQwenConv1dContainer()

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model", "language_model")
            if key.endswith("conv1d.weight") and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[key] = value
        return sanitized


def _write_tiny_qwen_moe_artifact(tmp_path) -> tuple[Path, QwenMoeConversionPlan]:
    gate_up_tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down_tensor = "model.language_model.layers.0.mlp.experts.down_proj"
    shard_name = "model-00001-of-00001.safetensors"
    gate_up = (
        np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8) - 64.0
    ) / 32.0
    down = (np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8) - 16.0) / 16.0
    mx.save_safetensors(
        str(tmp_path / shard_name),
        {gate_up_tensor: mx.array(gate_up), down_tensor: mx.array(down)},
    )
    plan = QwenMoeConversionPlan(
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        groups=(
            QwenMoeConversionGroup(
                layer=0,
                source_projection="gate_up_proj",
                source_tensor=gate_up_tensor,
                source_shards=(shard_name,),
                target_projections=("gate_proj", "up_proj"),
                experts=2,
                input_dims=8,
                output_dims=16,
                logical_output_dims=(8, 8),
                expected_source_shape=(2, 16, 8),
                source_shape=(2, 16, 8),
                source_dtype="F32",
                source_parameter_count=256,
                source_shape_matches=True,
                code_bits=8,
                group_size=8,
            ),
            QwenMoeConversionGroup(
                layer=0,
                source_projection="down_proj",
                source_tensor=down_tensor,
                source_shards=(shard_name,),
                target_projections=("down_proj",),
                experts=2,
                input_dims=8,
                output_dims=8,
                logical_output_dims=(8,),
                expected_source_shape=(2, 8, 8),
                source_shape=(2, 8, 8),
                source_dtype="F32",
                source_parameter_count=128,
                source_shape_matches=True,
                code_bits=8,
                group_size=8,
            ),
        ),
        missing_language_projection_pairs=(),
        source_shape_checks_present=True,
        conversion_status="qwen_moe_plan_ready__materializer_available",
        conversion_blockers=("family_specific_eval_and_benchmark_gates",),
    )
    output_dir = tmp_path / "qwen-artifact"
    convert_qwen_moe_groups_from_safetensors(
        source_dir=tmp_path,
        plan=plan,
        output_dir=output_dir,
    )
    return output_dir, plan


def _write_tiny_qwen_non_expert_source_checkpoint(path: Path) -> Path:
    shard_name = "model-00001-of-00001.safetensors"
    arrays = {
        "model.language_model.embed_tokens.weight": mx.array(
            np.arange(12, dtype=np.float32).reshape(4, 3)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": mx.array(
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.linear_attn.q_proj.weight": mx.array(
            np.arange(6, dtype=np.float32).reshape(2, 3)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.mlp.gate.weight": mx.array(
            np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        ).astype(mx.bfloat16),
        "model.language_model.norm.weight": mx.array(
            np.array([7.0, 8.0, 9.0], dtype=np.float32)
        ).astype(mx.bfloat16),
        "lm_head.weight": mx.array(
            (np.arange(12, dtype=np.float32).reshape(4, 3) + 100.0)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros(
            (2, 2, 3),
            dtype=mx.bfloat16,
        ),
        "mtp.layers.0.input_layernorm.weight": mx.zeros((3,), dtype=mx.bfloat16),
        "model.visual.blocks.0.attn.proj.weight": mx.zeros((3, 3), dtype=mx.bfloat16),
    }
    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in arrays},
            }
        )
    )
    return index_path


def _write_tiny_qwen_sanitized_non_expert_source_checkpoint(path: Path) -> Path:
    shard_name = "model-00001-of-00001.safetensors"
    arrays = {
        "model.language_model.embed_tokens.weight": mx.array(
            np.arange(12, dtype=np.float32).reshape(4, 3)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": mx.array(
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.array(
            np.arange(2 * 1 * 3, dtype=np.float32).reshape(2, 1, 3)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.mlp.gate.weight": mx.array(
            np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        ).astype(mx.bfloat16),
        "model.language_model.norm.weight": mx.array(
            np.array([7.0, 8.0, 9.0], dtype=np.float32)
        ).astype(mx.bfloat16),
        "lm_head.weight": mx.array(
            (np.arange(12, dtype=np.float32).reshape(4, 3) + 100.0)
        ).astype(mx.bfloat16),
    }
    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in arrays},
            }
        )
    )
    return index_path


def _write_tiny_tokenizer(path: Path) -> Path:
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<unk>",
        pad_token="<unk>",
    )
    tokenizer_dir = path / "tokenizer"
    fast.save_pretrained(tokenizer_dir)
    return tokenizer_dir


def _tiny_qwen35_moe_upstream_model() -> qwen3_5_moe.Model:
    args = qwen3_5_moe.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "attn_output_gate": True,
                "bos_token_id": 1,
                "eos_token_id": 1,
                "full_attention_interval": 1,
                "head_dim": 4,
                "hidden_act": "silu",
                "hidden_size": 8,
                "initializer_range": 0.02,
                "layer_types": ["full_attention"],
                "max_position_embeddings": 128,
                "model_type": "qwen3_5_moe_text",
                "moe_intermediate_size": 4,
                "num_attention_heads": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "output_router_logits": False,
                "partial_rotary_factor": 1.0,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
                "router_aux_loss_coef": 0.001,
                "shared_expert_intermediate_size": 4,
                "tie_word_embeddings": False,
                "use_cache": True,
                "vocab_size": 16,
            },
        }
    )
    return qwen3_5_moe.Model(args)


def _write_tiny_qwen35_moe_source_checkpoint(path: Path) -> Path:
    shard_name = "model-00001-of-00001.safetensors"
    arrays = {
        "model.language_model.embed_tokens.weight": mx.array(
            np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": mx.ones(
            (8,),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.array(
            np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.self_attn.k_proj.weight": mx.zeros(
            (4, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.self_attn.v_proj.weight": mx.zeros(
            (4, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.self_attn.o_proj.weight": mx.zeros(
            (8, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.self_attn.q_norm.weight": mx.ones(
            (4,),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.self_attn.k_norm.weight": mx.ones(
            (4,),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.post_attention_layernorm.weight": mx.ones(
            (8,),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.mlp.gate.weight": mx.zeros(
            (2, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": mx.zeros(
            (4, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": mx.zeros(
            (4, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": mx.zeros(
            (8, 4),
            dtype=mx.bfloat16,
        ),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": mx.zeros(
            (1, 8),
            dtype=mx.bfloat16,
        ),
        "model.language_model.norm.weight": mx.ones((8,), dtype=mx.bfloat16),
        "lm_head.weight": mx.array(
            np.arange(16 * 8, dtype=np.float32).reshape(16, 8) + 200.0
        ).astype(mx.bfloat16),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros(
            (2, 8, 8),
            dtype=mx.bfloat16,
        ),
    }
    mx.save_safetensors(str(path / shard_name), arrays)
    index_path = path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name in arrays},
            }
        )
    )
    return index_path


def _write_tiny_qwen35_moe_config(path: Path) -> Path:
    config_path = path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "attn_output_gate": True,
                    "bos_token_id": 1,
                    "eos_token_id": 1,
                    "full_attention_interval": 1,
                    "head_dim": 4,
                    "hidden_act": "silu",
                    "hidden_size": 8,
                    "initializer_range": 0.02,
                    "layer_types": ["full_attention"],
                    "max_position_embeddings": 128,
                    "model_type": "qwen3_5_moe_text",
                    "moe_intermediate_size": 4,
                    "num_attention_heads": 2,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 1,
                    "output_router_logits": False,
                    "partial_rotary_factor": 1.0,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "partial_rotary_factor": 1.0,
                        "rope_theta": 10000.0,
                        "rope_type": "default",
                    },
                    "router_aux_loss_coef": 0.001,
                    "shared_expert_intermediate_size": 4,
                    "tie_word_embeddings": False,
                    "use_cache": True,
                    "vocab_size": 16,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return config_path


def test_select_and_replace_nested_linear_path() -> None:
    model = TinyModel()
    path = select_linear_paths(
        model,
        target="model.layers.0.self_attn.q_proj",
        max_linears=1,
        require_group_size=8,
    )[0]
    assert path == "model.layers.0.self_attn.q_proj"

    original = get_module_by_path(model, path)
    replacement = DenseDequantLinear(weight=np.zeros_like(np.array(original.weight)))
    set_module_by_path(model, path, replacement)
    assert get_module_by_path(model, path) is replacement


def test_qwen_harness_vq_and_dense_reference_match_for_one_linear() -> None:
    rng = np.random.default_rng(1234)
    linear = nn.Linear(16, 8, bias=True)
    linear.weight = mx.array(rng.normal(scale=0.05, size=(8, 16)).astype(np.float32))
    linear.bias = mx.array(rng.normal(scale=0.01, size=(8,)).astype(np.float32))
    x = mx.array(rng.normal(size=(16,)).astype(np.float32))

    vq_layer, dense_layer, info = make_vq_and_dense_reference(linear, group_size=8)
    actual = vq_layer(x)
    expected = dense_layer(x)
    mx.eval(actual, expected)

    assert info.input_dims == 16
    assert info.output_dims == 8
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
    assert cosine_similarity(np.array(actual), np.array(expected)) >= 0.99999


def test_qwen_non_expert_bind_loads_source_weights_and_skips_auxiliary_tensors(tmp_path) -> None:
    index_path = _write_tiny_qwen_non_expert_source_checkpoint(tmp_path)
    target_model = TinyQwenRuntimeModel()

    report = bind_qwen_non_expert_weights(
        target_model,
        tmp_path,
        load_safetensors_index(index_path),
        layers=(0,),
    )
    mx.eval(target_model.parameters())

    params = dict(tree_flatten(target_model.parameters()))
    assert report.loaded_model_parameters == (
        "language_model.embed_tokens.weight",
        "language_model.layers.0.input_layernorm.weight",
        "language_model.layers.0.linear_attn.q_proj.weight",
        "language_model.layers.0.mlp.gate.weight",
        "language_model.norm.weight",
        "lm_head.weight",
    )
    assert report.missing_model_parameters == ()
    assert report.skipped_routed_expert_tensors == (
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    )
    assert report.skipped_mtp_tensors == ("mtp.layers.0.input_layernorm.weight",)
    assert report.skipped_visual_tensors == ("model.visual.blocks.0.attn.proj.weight",)
    assert params["language_model.embed_tokens.weight"].dtype == mx.bfloat16
    assert params["lm_head.weight"].dtype == mx.bfloat16
    np.testing.assert_allclose(
        np.array(params["language_model.layers.0.linear_attn.q_proj.weight"].astype(mx.float32)),
        np.arange(6, dtype=np.float32).reshape(2, 3),
        rtol=0,
        atol=0,
    )


def test_qwen_non_expert_bind_uses_upstream_sanitize_for_layout_transforms(tmp_path) -> None:
    index_path = _write_tiny_qwen_sanitized_non_expert_source_checkpoint(tmp_path)
    target_model = TinyQwenSanitizingRuntimeModel()

    report = bind_qwen_non_expert_weights(
        target_model,
        tmp_path,
        load_safetensors_index(index_path),
        layers=(0,),
    )
    mx.eval(target_model.parameters())

    params = dict(tree_flatten(target_model.parameters()))
    assert report.loaded_model_parameters == (
        "language_model.embed_tokens.weight",
        "language_model.layers.0.input_layernorm.weight",
        "language_model.layers.0.linear_attn.conv1d.weight",
        "language_model.layers.0.mlp.gate.weight",
        "language_model.norm.weight",
        "lm_head.weight",
    )
    assert report.missing_model_parameters == ()
    assert params["language_model.layers.0.linear_attn.conv1d.weight"].shape == (2, 3, 1)
    np.testing.assert_allclose(
        np.array(
            params["language_model.layers.0.linear_attn.conv1d.weight"].astype(mx.float32)
        ),
        np.arange(2 * 1 * 3, dtype=np.float32).reshape(2, 1, 3).swapaxes(1, 2),
        rtol=0,
        atol=0,
    )


def test_qwen_bind_helpers_support_upstream_qwen35_moe_transformer_surface(
    tmp_path,
) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    index_path = _write_tiny_qwen35_moe_source_checkpoint(tmp_path)
    model = _tiny_qwen35_moe_upstream_model()

    moe_report = bind_qwen_moe_vq_experts(model, output_dir, layers=(0,))
    non_expert_report = bind_qwen_non_expert_weights(
        model,
        tmp_path,
        load_safetensors_index(index_path),
        layers=(0,),
    )
    mx.eval(model.parameters())

    params = dict(tree_flatten(model.parameters()))
    assert moe_report.bound_layers == (0,)
    assert moe_report.missing_layers == ()
    assert has_unbound_qwen_moe_vq_experts(model, layers=(0,)) is False
    assert non_expert_report.loaded_model_parameters == (
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.mlp.gate.weight",
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.layers.0.self_attn.k_norm.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.norm.weight",
    )
    assert non_expert_report.missing_model_parameters == ()
    assert params["language_model.model.embed_tokens.weight"].dtype == mx.bfloat16
    assert params["language_model.lm_head.weight"].dtype == mx.bfloat16


def test_qwen_bind_helpers_run_upstream_qwen35_moe_forward_with_finite_logits(
    tmp_path,
) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    index_path = _write_tiny_qwen35_moe_source_checkpoint(tmp_path)
    model = _tiny_qwen35_moe_upstream_model()

    moe_report = bind_qwen_moe_vq_experts(model, output_dir, layers=(0,))
    non_expert_report = bind_qwen_non_expert_weights(
        model,
        tmp_path,
        load_safetensors_index(index_path),
        layers=(0,),
    )
    tokens = mx.array([[0, 1]], dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    logits_np = np.array(logits.astype(mx.float32))

    assert moe_report.bound_layers == (0,)
    assert non_expert_report.missing_model_parameters == ()
    assert tuple(logits.shape) == (1, 2, 16)
    assert np.isfinite(logits_np).all()


def test_qwen_upstream_transformer_probe_binds_artifact_to_mlx_lm_surface(
    tmp_path,
) -> None:
    from benchmarks.probe_qwen_upstream_transformer_binding import (
        probe_qwen_upstream_transformer_binding,
    )

    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    config_path = _write_tiny_qwen35_moe_config(tmp_path)

    payload = probe_qwen_upstream_transformer_binding(
        artifact_dir=output_dir,
        config_path=config_path,
        expected_layers=(0,),
    )

    assert payload["record_type"] == "qwen_upstream_transformer_binding_probe"
    assert payload["integration_pass"] is True
    assert payload["upstream_model_module"] == "mlx_lm.models.qwen3_5_moe"
    assert payload["bound_layers"] == [0]
    assert payload["missing_layers"] == []
    assert payload["unbound_vq_experts"] is False
    assert payload["sanitized_switch_keys_present"] is True
    assert payload["instantiated_full_config"] is False


def test_qwen_non_expert_binding_probe_reports_loaded_source_tensors(tmp_path) -> None:
    index_path = _write_tiny_qwen_non_expert_source_checkpoint(tmp_path)

    payload = probe_qwen_non_expert_binding_source(
        tmp_path,
        index_path=index_path,
        layers=(0,),
    )

    assert payload["binding_pass"] is True
    assert payload["requested_layers"] == [0]
    assert payload["loaded_model_parameters"] == [
        "language_model.embed_tokens.weight",
        "language_model.layers.0.input_layernorm.weight",
        "language_model.layers.0.linear_attn.q_proj.weight",
        "language_model.layers.0.mlp.gate.weight",
        "language_model.norm.weight",
        "lm_head.weight",
    ]
    assert payload["missing_model_parameters"] == []
    assert payload["skipped_routed_expert_tensor_count"] == 1
    assert payload["skipped_mtp_tensor_count"] == 1
    assert payload["skipped_visual_tensor_count"] == 1
    assert payload["loaded_dtype_names"] == {
        "language_model.embed_tokens.weight": "bfloat16",
        "language_model.layers.0.input_layernorm.weight": "bfloat16",
        "language_model.layers.0.linear_attn.q_proj.weight": "bfloat16",
        "language_model.layers.0.mlp.gate.weight": "bfloat16",
        "language_model.norm.weight": "bfloat16",
        "lm_head.weight": "bfloat16",
    }


def test_qwen_moe_materialized_projection_loads_and_runs_switch_runtime(tmp_path) -> None:
    source_tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    shard_name = "model-00001-of-00001.safetensors"
    fused = (np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8) - 64.0) / 32.0
    mx.save_safetensors(str(tmp_path / shard_name), {source_tensor: mx.array(fused)})
    output = tmp_path / "qwen-layer0-gate-up.safetensors"
    group = QwenMoeConversionGroup(
        layer=0,
        source_projection="gate_up_proj",
        source_tensor=source_tensor,
        source_shards=(shard_name,),
        target_projections=("gate_proj", "up_proj"),
        experts=2,
        input_dims=8,
        output_dims=16,
        logical_output_dims=(8, 8),
        expected_source_shape=(2, 16, 8),
        source_shape=(2, 16, 8),
        source_dtype="F32",
        source_parameter_count=256,
        source_shape_matches=True,
        code_bits=8,
        group_size=8,
    )
    convert_qwen_moe_group_from_safetensors(
        source_dir=tmp_path,
        group=group,
        output_path=output,
    )

    prefix = qwen_moe_switch_prefix(layer=0, projection="gate_proj")
    layer = load_qwen_moe_switch_projection(output, layer=0, projection="gate_proj")
    result = validate_qwen_moe_switch_projection(output, layer=0, projection="gate_proj")

    assert prefix == "model.language_model.layers.0.mlp.switch_mlp.gate_proj"
    assert layer.num_experts == 2
    assert layer.input_dims == 8
    assert layer.output_dims == 8
    assert result.projection == "gate_proj"
    assert result.num_experts == 2
    assert result.output_shape == (2, 2, 8)
    assert result.max_abs_diff < 2e-4
    assert result.cosine >= 0.99999


def test_qwen_moe_materialization_audit_proves_manifest_is_loadable(tmp_path) -> None:
    gate_up_tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down_tensor = "model.language_model.layers.0.mlp.experts.down_proj"
    shard_name = "model-00001-of-00001.safetensors"
    gate_up = (
        np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8) - 64.0
    ) / 32.0
    down = (np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8) - 16.0) / 16.0
    mx.save_safetensors(
        str(tmp_path / shard_name),
        {gate_up_tensor: mx.array(gate_up), down_tensor: mx.array(down)},
    )
    plan = QwenMoeConversionPlan(
        model_id=QWEN36_35B_A3B_MODEL_ID,
        revision="fixture",
        groups=(
            QwenMoeConversionGroup(
                layer=0,
                source_projection="gate_up_proj",
                source_tensor=gate_up_tensor,
                source_shards=(shard_name,),
                target_projections=("gate_proj", "up_proj"),
                experts=2,
                input_dims=8,
                output_dims=16,
                logical_output_dims=(8, 8),
                expected_source_shape=(2, 16, 8),
                source_shape=(2, 16, 8),
                source_dtype="F32",
                source_parameter_count=256,
                source_shape_matches=True,
                code_bits=8,
                group_size=8,
            ),
            QwenMoeConversionGroup(
                layer=0,
                source_projection="down_proj",
                source_tensor=down_tensor,
                source_shards=(shard_name,),
                target_projections=("down_proj",),
                experts=2,
                input_dims=8,
                output_dims=8,
                logical_output_dims=(8,),
                expected_source_shape=(2, 8, 8),
                source_shape=(2, 8, 8),
                source_dtype="F32",
                source_parameter_count=128,
                source_shape_matches=True,
                code_bits=8,
                group_size=8,
            ),
        ),
        missing_language_projection_pairs=(),
        source_shape_checks_present=True,
        conversion_status="qwen_moe_plan_ready__materializer_available",
        conversion_blockers=("qwen3_5_moe_runtime_adapter",),
    )
    output_dir = tmp_path / "qwen-artifact"
    convert_qwen_moe_groups_from_safetensors(
        source_dir=tmp_path,
        plan=plan,
        output_dir=output_dir,
    )

    audit = audit_qwen_moe_materialization_manifest(
        output_dir,
        expected_source_projection_groups=2,
        expected_target_projection_groups=3,
    )

    assert audit.audit_pass is True
    assert audit.model_id == QWEN36_35B_A3B_MODEL_ID
    assert audit.source_projection_groups == 2
    assert audit.target_projection_groups == 3
    assert audit.loadable_projection_groups == 3
    assert audit.missing_projection_files == ()
    assert audit.missing_projection_tensors == ()
    assert audit.load_errors == ()
    assert audit.dense_routed_experts is False
    assert audit.unbound_vq_experts is False
    assert audit.effective_routed_bpw == 3.0
    assert audit.projection_files == (
        str(output_dir / qwen_moe_group_output_filename(plan.groups[0])),
        str(output_dir / qwen_moe_group_output_filename(plan.groups[1])),
    )


def test_qwen_moe_materialization_audit_accepts_cwd_relative_manifest_paths(
    tmp_path, monkeypatch
) -> None:
    output_dir, plan = _write_tiny_qwen_moe_artifact(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    relative_output_dir = output_dir.relative_to(tmp_path.parent)
    manifest_path = relative_output_dir / "qwen-moe-materialization-manifest.json"
    payload = json.loads(manifest_path.read_text())
    for group in payload["groups"]:
        group["output_path"] = str(Path(group["output_path"]).relative_to(tmp_path.parent))
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    audit = audit_qwen_moe_materialization_manifest(
        relative_output_dir,
        expected_source_projection_groups=2,
        expected_target_projection_groups=3,
    )

    assert audit.audit_pass is True
    assert audit.loadable_projection_groups == 3
    assert audit.missing_projection_files == ()
    assert audit.projection_files == (
        str(relative_output_dir / qwen_moe_group_output_filename(plan.groups[0])),
        str(relative_output_dir / qwen_moe_group_output_filename(plan.groups[1])),
    )


def test_qwen_moe_manifest_binds_complete_layer_switch_glu(tmp_path) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)

    switch_mlp = load_qwen_moe_switch_glu(output_dir, layer=0)
    x = mx.array(np.linspace(-0.4, 0.4, 2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8))
    indices = mx.array(np.array([[[0, 1], [1, 0], [0, 0]], [[1, 1], [0, 1], [1, 0]]]))

    actual = switch_mlp(x, indices)
    expected = switch_mlp.down_proj(
        nn.silu(switch_mlp.gate_proj(x, indices)) * switch_mlp.up_proj(x, indices),
        indices,
    )
    mx.eval(actual, expected)

    assert switch_mlp.num_experts == 2
    assert switch_mlp.input_dims == 8
    assert switch_mlp.hidden_dims == 8
    assert tuple(actual.shape) == (2, 3, 2, 8)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)


def test_qwen_moe_manifest_binds_selected_model_layers(tmp_path) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    model = TinyQwenMoEModel(layer_count=2)

    report = bind_qwen_moe_vq_experts(model, output_dir, layers=(0,))

    assert report.bound_layers == (0,)
    assert report.skipped_layers == (1,)
    assert report.missing_layers == ()
    assert model.language_model.layers[0].mlp.switch_mlp is not None
    assert model.language_model.layers[1].mlp.switch_mlp is None
    assert has_unbound_qwen_moe_vq_experts(model, layers=(0,)) is False
    assert has_unbound_qwen_moe_vq_experts(model) is True


def test_qwen_moe_manifest_binding_accepts_cwd_relative_manifest_paths(
    tmp_path, monkeypatch
) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    relative_output_dir = output_dir.relative_to(tmp_path.parent)
    manifest_path = relative_output_dir / "qwen-moe-materialization-manifest.json"
    payload = json.loads(manifest_path.read_text())
    for group in payload["groups"]:
        group["output_path"] = str(Path(group["output_path"]).relative_to(tmp_path.parent))
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    model = TinyQwenMoEModel(layer_count=2)

    report = bind_qwen_moe_vq_experts(model, relative_output_dir, layers=(0,))

    assert report.bound_layers == (0,)
    assert report.missing_layers == ()
    assert model.language_model.layers[0].mlp.switch_mlp is not None
    assert has_unbound_qwen_moe_vq_experts(model, layers=(0,)) is False


def test_qwen_moe_manifest_reports_missing_bindable_layers(tmp_path) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    model = TinyQwenMoEModel(layer_count=2)

    report = bind_qwen_moe_vq_experts(model, output_dir, layers=(1,))

    assert report.bound_layers == ()
    assert report.skipped_layers == (0,)
    assert report.missing_layers == (1,)
    assert model.language_model.layers[0].mlp.switch_mlp is None
    assert model.language_model.layers[1].mlp.switch_mlp is None
    assert has_unbound_qwen_moe_vq_experts(model, layers=(1,)) is True


def test_qwen_moe_binding_probe_reports_bound_and_missing_layers(tmp_path) -> None:
    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)

    payload = probe_qwen_moe_binding_artifact(
        output_dir,
        expected_layers=(0, 1),
        model_layer_count=2,
    )

    assert payload["binding_pass"] is False
    assert payload["bound_layers"] == [0]
    assert payload["missing_layers"] == [1]
    assert payload["unbound_vq_experts"] is True
    assert payload["dense_routed_experts"] is False


def test_qwen_runtime_binding_probe_combines_source_and_vq_forward(tmp_path) -> None:
    from benchmarks.probe_qwen_runtime_binding import probe_qwen_runtime_binding

    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    index_path = _write_tiny_qwen_non_expert_source_checkpoint(tmp_path)

    payload = probe_qwen_runtime_binding(
        artifact_dir=output_dir,
        source_dir=tmp_path,
        index_path=index_path,
        expected_layers=(0,),
        model_layer_count=1,
        forward_layers=(0,),
        forward_top_k=2,
        input_scale=0.125,
    )

    assert payload["runtime_binding_pass"] is True
    assert payload["moe_binding_pass"] is True
    assert payload["non_expert_binding_pass"] is True
    assert payload["bound_layers"] == [0]
    assert payload["loaded_model_parameter_count"] == 6
    assert payload["forward_pass"] is True
    assert payload["forward_results"] == [
        {
            "layer": 0,
            "input_shape": [1, 8],
            "indices_shape": [1, 2],
            "output_shape": [1, 2, 8],
            "finite": True,
        }
    ]


def test_qwen_runtime_binding_probe_can_compute_bounded_source_logits(tmp_path) -> None:
    from benchmarks.probe_qwen_runtime_binding import probe_qwen_runtime_binding

    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    index_path = _write_tiny_qwen_non_expert_source_checkpoint(tmp_path)

    payload = probe_qwen_runtime_binding(
        artifact_dir=output_dir,
        source_dir=tmp_path,
        index_path=index_path,
        expected_layers=(),
        model_layer_count=1,
        max_tensors=2,
        logit_probe_token_id=2,
        logit_probe_top_k=2,
    )

    assert payload["runtime_binding_pass"] is True
    assert payload["moe_binding_pass"] is True
    assert payload["non_expert_binding_pass"] is True
    assert payload["loaded_model_parameter_count"] == 2
    assert payload["logit_probe_pass"] is True
    assert payload["logit_probe"] == {
        "token_id": 2,
        "embedding_shape": [3],
        "lm_head_shape": [4, 3],
        "logits_shape": [4],
        "top_k": 2,
        "top_indices": [3, 2],
        "finite": True,
    }


def test_qwen_runtime_binding_probe_can_compute_tokenized_source_logits(tmp_path) -> None:
    from benchmarks.probe_qwen_runtime_binding import probe_qwen_runtime_binding

    output_dir, _plan = _write_tiny_qwen_moe_artifact(tmp_path)
    index_path = _write_tiny_qwen_non_expert_source_checkpoint(tmp_path)
    tokenizer_dir = _write_tiny_tokenizer(tmp_path)

    payload = probe_qwen_runtime_binding(
        artifact_dir=output_dir,
        source_dir=tmp_path,
        index_path=index_path,
        expected_layers=(),
        model_layer_count=1,
        max_tensors=2,
        tokenizer_dir=tokenizer_dir,
        tokenizer_prompt="hello world",
        tokenized_logit_position="last",
        tokenized_logit_top_k=2,
    )

    assert payload["runtime_binding_pass"] is True
    assert payload["tokenized_logit_probe_pass"] is True
    assert payload["tokenized_logit_probe"] == {
        "record_type": "qwen_tokenized_source_logits_probe",
        "prompt": "hello world",
        "encoded_token_ids": [1, 2],
        "selected_token_position": 1,
        "selected_token_id": 2,
        "logit_probe": {
            "token_id": 2,
            "embedding_shape": [3],
            "lm_head_shape": [4, 3],
            "logits_shape": [4],
            "top_k": 2,
            "top_indices": [3, 2],
            "finite": True,
        },
        "finite": True,
    }


def test_qwen_runtime_binding_cli_empty_expected_layers_means_no_layers() -> None:
    from benchmarks.probe_qwen_runtime_binding import _parse_layers

    assert _parse_layers(None) is None
    assert _parse_layers("") == ()
    assert _parse_layers("0,39") == (0, 39)
