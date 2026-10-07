#!/usr/bin/env python3
"""Generate GLM-5.3-Flash parity fixtures from the pinned reference model.

Runs ``transformers`` v5.16.1's ``Glm5NextTextModel`` (the first release
that ships ``glm5_next``) on a tiny random config in float32 on CPU and saves
the weights under their *checkpoint* names, the inputs, and the outputs. The
RAMP tests load this file and never need torch.

torch is not a project dependency, so this runs in a throwaway environment:

    uv run --no-project --python 3.12 \
        --with torch --with transformers==5.16.1 --with numpy \
        python scripts/make_glm53_reference_fixtures.py \
        --out tests/fixtures/glm53_reference_tiny.npz

The tiny config keeps every structural feature of the release: KDA and DSA
layers interleaved, a dense first layer then MoE, mHC with 4 streams, NoPE
MLA, and a k-pool indexer whose top-k is smaller than the sequence so the
sparse path and the tail are exercised. KDA head dim is 32 because the MLX
gated delta kernel needs a multiple of 32.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import transformers
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

PREFILL = 40
DECODE = 6
SEED = 20261006

TINY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "deepseek_sparse_attention"] * 2,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full"] * 4,
    "first_k_dense_replace": 1,
    "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 2],
        "full_attn_layers": [1, 3],
    },
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "q_lora_rank": 48,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 0,
    "v_head_dim": 32,
    "index_topk": 8,
    # Eight heads make an all-ReLU-zero pool score rare; tie_free() rejects
    # any seed that still leaves a tie at a top-k cut.
    "index_n_heads": 8,
    "index_head_dim": 32,
    "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "index_kpool_compress": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    # Small enough that the clamps actually bind at these activation scales.
    "swiglu_limit": 1.0,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "mhc": True,
    "rms_norm_eps": 1e-5,
    "num_nextn_predict_layers": 0,
    "attention_bias": False,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "pad_token_id": 0,
}


def randomize(model: torch.nn.Module, generator: torch.Generator) -> None:
    """Replace the near-identity init with weights that exercise every path."""

    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            if name.endswith(("norm.weight", "layernorm.weight", "o_norm.weight")):
                param.copy_(1.0 + 0.2 * noise)
            elif name.endswith(("hc.scale",)):
                param.copy_(1.0 + 0.3 * noise)
            elif param.ndim >= 2:
                param.copy_(noise * (2.0 / param.shape[-1]) ** 0.5)
            else:
                param.copy_(0.5 * noise)
        for name, buf in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buf.copy_(0.1 * torch.randn(buf.shape, generator=generator))


def tie_free(model: torch.nn.Module, ids: torch.Tensor) -> bool:
    """True when no top-k call in the forward has a tie at its cut.

    The indexer's ReLU makes exact-zero scores common, and torch, CUDA kernels
    and MLX break such ties differently: the order is implementation-defined,
    not part of the model. A fixture with a tie at a cut would test tie order,
    not the architecture, so such seeds are rejected.
    """

    calls: list[tuple[torch.Tensor, int]] = []
    method, function = torch.Tensor.topk, torch.topk

    def record(scores, k, *args, **kwargs):
        calls.append((scores.detach().float().clone(), int(k)))
        return method(scores, k, *args, **kwargs)

    torch.Tensor.topk = record
    torch.topk = record
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        torch.Tensor.topk, torch.topk = method, function
    for scores, k in calls:
        if k >= scores.shape[-1]:
            continue
        ordered = scores.sort(dim=-1, descending=True).values
        live = ordered[..., k] > torch.finfo(torch.float32).min / 2
        if ((ordered[..., k - 1] - ordered[..., k] < 1e-5) & live).any():
            return False
    return True


def checkpoint_names(state: dict[str, torch.Tensor], config) -> dict[str, np.ndarray]:
    """Invert transformers' ``glm5_next`` load-time conversion mapping."""

    out: dict[str, np.ndarray] = {}
    qkv = config.linear_num_heads * config.linear_head_dim
    for name, value in state.items():
        array = value.detach().float().numpy()
        key = "model.language_model." + name
        key = key.replace("self_attn.forget_gate.", "self_attn.")
        for site in ("attn", "ffn"):
            for leaf in ("fn", "base", "scale"):
                key = key.replace(f"{site}_hc.{leaf}", f"hc_{site}_{leaf}")
        if key.endswith("mlp.experts.gate_up_proj"):
            gate, up = np.split(array, 2, axis=1)
            stem = key[: -len("gate_up_proj")]
            for e in range(array.shape[0]):
                out[f"{stem}{e}.gate_proj.weight"] = gate[e]
                out[f"{stem}{e}.up_proj.weight"] = up[e]
        elif key.endswith("mlp.experts.down_proj"):
            stem = key[: -len("down_proj")]
            for e in range(array.shape[0]):
                out[f"{stem}{e}.down_proj.weight"] = array[e]
        elif key.endswith("self_attn.conv1d.weight"):
            stem = key[: -len("conv1d.weight")]
            for part, chunk in zip("qkv", np.split(array, [qkv, 2 * qkv], axis=0)):
                out[f"{stem}{part}_conv1d.weight"] = chunk
        else:
            out[key] = array
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert transformers.__version__ == "5.16.1", transformers.__version__

    config = Glm5NextTextConfig(**TINY)
    config._attn_implementation = "eager"
    for seed in range(SEED, SEED + 100):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = Glm5NextTextModel(config).float().eval()
        randomize(model, generator)
        lm_head = torch.randn(TINY["vocab_size"], TINY["hidden_size"], generator=generator) * 0.125
        ids = torch.randint(0, TINY["vocab_size"], (1, PREFILL + DECODE), generator=generator)
        if tie_free(model, ids):
            break
    else:
        raise SystemExit("no tie-free seed in 100 tries")
    arrays: dict[str, np.ndarray] = {"input_ids": ids.numpy().astype(np.int32)}
    with torch.no_grad():
        full = model(ids, output_hidden_states=True, use_cache=False)
        arrays["full_last_hidden"] = full.last_hidden_state.numpy()
        arrays["full_logits"] = (full.last_hidden_state @ lm_head.T).numpy()
        for i, h in enumerate(full.hidden_states):
            arrays[f"full_hidden_{i}"] = h.numpy()

        prefix = model(ids[:, :PREFILL], use_cache=True)
        cache = prefix.past_key_values
        steps = [prefix.last_hidden_state[:, -1]]
        for t in range(PREFILL, PREFILL + DECODE - 1):
            step = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            steps.append(step.last_hidden_state[:, -1])
        arrays["incremental_last_hidden"] = torch.stack(steps, dim=1).numpy()

    weights = checkpoint_names(model.state_dict(), config)
    weights["lm_head.weight"] = lm_head.numpy()
    for name, value in weights.items():
        arrays["w/" + name] = value
    arrays["config_json"] = np.frombuffer(json.dumps(TINY).encode(), dtype=np.uint8)
    arrays["meta_json"] = np.frombuffer(
        json.dumps(
            {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "seed": seed,
                "prefill": PREFILL,
                "decode": DECODE,
                "dtype": "float32",
                "attn_implementation": "eager",
            }
        ).encode(),
        dtype=np.uint8,
    )
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}: {len(weights)} weights, {len(arrays)} arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
