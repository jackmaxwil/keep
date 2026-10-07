#!/usr/bin/env python3
"""Real-weight parity: the first N GLM-5.3-Flash layers, RAMP against the reference.

Fetches the embedding, the final norm and every tensor of layers ``0..N-1``
from the pinned shards by byte range, then:

1. decodes the FP8 tensors with ``keep.convert.fp8_block`` and checks the
   result is bit-identical to torch's native ``float8_e4m3fn`` times the F32
   128x128 ``weight_scale_inv``, which is what transformers does on CPU;
2. runs the same real prompt through transformers v5.16.1's
   ``Glm5NextTextModel`` truncated to N layers and through
   ``ramp.models.glm5_next_adapter``, both in float32, and compares the
   hidden streams entering each layer and leaving the last one;
3. repeats the RAMP run in bfloat16 to record the drift against float32.

N=3 covers three KDA layers with dense FP8 MLPs (about 2.5 GB fetched, under
10 GB resident). N=4 adds layer 3 with all 288 experts and projects about
70 GB resident, so the script refuses it unless ``--max-projected-gb`` is
raised, which needs the machine owner's OK. A memory-bounded layer-3 check
should stream experts instead of holding all 288 in float32.

Runs in a throwaway env so torch never becomes a project dependency:

    uv run --no-project --python 3.12 --with-editable . \
        --with torch --with transformers==5.16.1 \
        python scripts/check_glm53_real_layer_parity.py --layers 3 \
        --out artifacts/quality/glm53-real-layer-parity-l3-20261006.json
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import HfFileSystem, get_safetensors_metadata, hf_hub_download
from tokenizers import Tokenizer
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

from keep.convert.fp8_block import dequantize_fp8_block
from ramp.models.glm5_next_policy import GLM5_NEXT_MODEL_ID as MODEL, GLM5_NEXT_REVISION as REV

PREFIX = "model.language_model."
PROMPT = (
    "def merge_intervals(intervals):\n"
    "    # Sort by start, then sweep and merge overlapping ranges.\n"
    "    intervals.sort(key=lambda pair: pair[0])\n"
    "    merged = []\n"
    "    for start, end in intervals:\n"
    "        if merged and start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return merged\n"
)


def fetch(names: list[str]) -> dict[str, tuple[str, list[int], bytes]]:
    fs = HfFileSystem()
    meta = get_safetensors_metadata(MODEL, revision=REV)
    header: dict[str, int] = {}

    def cat(shard: str, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                return fs.cat_file(f"{MODEL}@{REV}/{shard}", start=start, end=end)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def one(name: str):
        shard = meta.weight_map[name]
        if shard not in header:
            header[shard] = int.from_bytes(cat(shard, 0, 8), "little")
        info = meta.files_metadata[shard].tensors[name]
        base = 8 + header[shard]
        data = cat(shard, base + info.data_offsets[0], base + info.data_offsets[1])
        if len(data) != info.data_offsets[1] - info.data_offsets[0]:
            raise SystemExit(f"short read on {name}")
        return name, (info.dtype, list(info.shape), data)

    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(one, names))


_EXPERT = re.compile(r"(.*mlp\.experts\.)(\d+)\.(gate|up|down)_proj\.weight$")


def decode(raw: dict, n_experts: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], int]:
    """Decode to float32, FP8 pairs through KEEP, checked bit-exact against torch.

    Routed experts go straight into per-layer stacks in the reference layout,
    ``gate_up [E, 2I, H]`` and ``down [E, H, I]``, so torch can use them without
    a copy. Raw bytes are released as they are consumed. Every non-expert FP8
    tensor and the first four experts of each projection are checked against
    torch; the rest share the same code path.
    """

    out: dict[str, np.ndarray] = {}
    stacks: dict[str, dict[str, np.ndarray]] = {}
    checked = 0
    for name in [k for k in raw if not k.endswith("weight_scale_inv")]:
        dtype, shape, data = raw.pop(name)
        if dtype == "F8_E4M3":
            codes = np.frombuffer(data, dtype=np.uint8).reshape(shape)
            _, s_shape, s_data = raw.pop(name + "_scale_inv")
            scales = np.frombuffer(s_data, dtype=np.float32).reshape(s_shape)
            value = dequantize_fp8_block(codes, scales)
            expert = _EXPERT.match(name)
            if expert is None or int(expert.group(2)) < 4:
                t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(shape).float()
                s = torch.from_numpy(scales.copy()).repeat_interleave(128, 0).repeat_interleave(128, 1)
                if not np.array_equal(value, (t * s[: shape[0], : shape[1]]).numpy()):
                    raise SystemExit(f"FP8 decode differs from torch on {name}")
                checked += 1
            if expert is not None:
                stem, e, proj = expert.group(1), int(expert.group(2)), expert.group(3)
                inter, hidden = (shape[0], shape[1]) if proj != "down" else (shape[1], shape[0])
                stack = stacks.setdefault(stem, {
                    "gate_up": np.empty((n_experts, 2 * inter, hidden), dtype=np.float32),
                    "down": np.empty((n_experts, hidden, inter), dtype=np.float32),
                })
                if proj == "down":
                    stack["down"][e] = value
                else:
                    offset = 0 if proj == "gate" else inter
                    stack["gate_up"][e, offset : offset + inter] = value
                continue
            out[name] = value
        elif dtype == "BF16":
            out[name] = (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif dtype == "F32":
            out[name] = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
        else:
            raise SystemExit(f"unexpected dtype {dtype} for {name}")
    return out, stacks, checked


def reference_state(weights: dict[str, np.ndarray], stacks: dict) -> dict[str, torch.Tensor]:
    """Checkpoint names to transformers module names (its conversion_mapping)."""

    qkv: dict[str, dict[str, np.ndarray]] = {}
    state: dict[str, torch.Tensor] = {}
    for name, value in weights.items():
        key = name.removeprefix(PREFIX)
        key = re.sub(r"self_attn\.(f_a_proj|f_b_proj|dt_bias|A_log)", r"self_attn.forget_gate.\1", key)
        key = re.sub(r"hc_(attn|ffn)_(fn|base|scale)", r"\1_hc.\2", key)
        if m := re.match(r"(.*self_attn\.)([qkv])_conv1d\.weight$", key):
            qkv.setdefault(m.group(1), {})[m.group(2)] = value
            continue
        state[key] = torch.from_numpy(value)
    for stem, parts in qkv.items():
        state[stem + "conv1d.weight"] = torch.from_numpy(np.concatenate([parts[p] for p in "qkv"]))
    for stem, stack in stacks.items():
        key = stem.removeprefix(PREFIX)
        state[key + "gate_up_proj"] = torch.from_numpy(stack["gate_up"])
        state[key + "down_proj"] = torch.from_numpy(stack["down"])
    return state


def ramp_hidden(text_config: dict, weights: dict[str, np.ndarray], stacks: dict, ids: np.ndarray, dtype) -> list[np.ndarray]:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import SwitchGLU

    from ramp.models.deepseek_v4_flash_adapter import LimitedSwiGLU
    from ramp.models.glm5_next_adapter import (
        Glm5NextMoE,
        Glm5NextVQModel,
        bind_glm5_next_non_vq_weights,
        glm5_next_args_from_config,
    )

    args = glm5_next_args_from_config(text_config)
    model = Glm5NextVQModel(args)
    arrays = {k: mx.array(v).astype(dtype) for k, v in weights.items()}
    arrays["lm_head.weight"] = mx.zeros((args.vocab_size, args.hidden_size), dtype=dtype)
    bind_glm5_next_non_vq_weights(model, arrays)
    for i, layer in enumerate(model.layers):
        if isinstance(layer.mlp, Glm5NextMoE):
            switch = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts, activation=LimitedSwiGLU(args.swiglu_limit))
            stack = stacks[f"{PREFIX}layers.{i}.mlp.experts."]
            inter = args.moe_intermediate_size
            switch.gate_proj.weight = mx.array(stack["gate_up"][:, :inter]).astype(dtype)
            switch.up_proj.weight = mx.array(stack["gate_up"][:, inter:]).astype(dtype)
            switch.down_proj.weight = mx.array(stack["down"]).astype(dtype)
            layer.mlp.bind_switch_mlp(switch)
    hidden: list = []
    model.model(mx.array(ids), hidden_states=hidden)
    # The stream leaving the last layer, before the mean and norm.
    last = model.layers[-1](hidden[-1])
    return [np.array(x.astype(mx.float32)) for x in hidden + [last]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-projected-gb", type=float, default=24.0,
        help="refuse to start when the projected peak exceeds this (default 24 GB)",
    )
    args = parser.parse_args()
    # Projected peak: every fetched byte decoded to float32 (about 4x the FP8
    # experts), held once by numpy and torch together and once more by MLX.
    # A layers=4 run projects about 70 GB and was killed on 2026-10-06 at
    # 57 GB resident. Above the budget, run only with the owner's OK.
    sparse = max(0, args.layers - 3)
    projected_gb = 3.0 + 1.3 * min(args.layers, 3) + sparse * 2 * 29.0
    if projected_gb > args.max_projected_gb:
        raise SystemExit(
            f"projected peak {projected_gb:.0f} GB exceeds --max-projected-gb "
            f"{args.max_projected_gb:.0f}. Ask before raising it."
        )
    assert transformers.__version__ == "5.16.1", transformers.__version__
    # Heavy jobs serialize on the repo lock: two model-sized processes at once
    # is how a 128 GB machine runs out of memory.
    lock = open(Path(__file__).resolve().parents[1] / ".keep-heavy-job.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    config = json.loads(Path(hf_hub_download(MODEL, "config.json", revision=REV)).read_text())
    text = dict(config["text_config"])
    n = args.layers
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        text[key] = text[key][:n]
    text["num_hidden_layers"] = n
    lin = dict(text["linear_attn_config"])
    lin["kda_layers"] = [i for i in lin["kda_layers"] if i < n]
    lin["full_attn_layers"] = [i for i in lin["full_attn_layers"] if i < n]
    text["linear_attn_config"] = lin
    text["num_nextn_predict_layers"] = 0

    meta = get_safetensors_metadata(MODEL, revision=REV)
    wanted = [
        k for k in meta.weight_map
        if k in (PREFIX + "embed_tokens.weight", PREFIX + "norm.weight")
        or (re.match(rf"{re.escape(PREFIX)}layers\.(\d+)\.", k) and int(k.split(".")[3]) < n)
    ]
    started = time.perf_counter()
    raw = fetch(wanted)
    fetched_bytes = sum(len(v[2]) for v in raw.values())
    fetch_seconds = time.perf_counter() - started
    weights, stacks, fp8_checked = decode(raw, text["n_routed_experts"])

    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json", revision=REV))
    ids = np.array([tokenizer.encode(PROMPT).ids], dtype=np.int32)

    torch_config = Glm5NextTextConfig(**text)
    torch_config._attn_implementation = "eager"
    reference = Glm5NextTextModel(torch_config).float().eval()
    reference.load_state_dict(reference_state(weights, stacks), strict=True)
    with torch.no_grad():
        out = reference(torch.from_numpy(ids.astype(np.int64)), output_hidden_states=True, use_cache=False)
    # hidden_states = (input to layer 0, ..., input to layer n-1, final normed).
    ref_inputs = [h.numpy() for h in out.hidden_states[:n]]
    with torch.no_grad():
        visible = torch.ones(ids.shape, dtype=torch.bool)
        ref_last = reference.layers[-1](out.hidden_states[n - 1], attention_mask=visible)[0].numpy()

    del reference, out
    import gc

    gc.collect()
    import mlx.core as mx

    rows = {}
    for label, dtype in (("float32", mx.float32), ("bfloat16", mx.bfloat16)):
        ours = ramp_hidden(text, weights, stacks, ids, dtype)
        gc.collect()
        rel = []
        for got, want in zip(ours, ref_inputs + [ref_last]):
            rel.append(float(np.abs(got - want).max() / np.abs(want).max()))
        rows[label] = {"max_relative_error_per_stream": rel}
    report = {
        "record_type": "glm53_real_layer_parity_v1",
        "model_id": MODEL,
        "revision": REV,
        "layers": n,
        "layer_types": text["layer_types"],
        "mlp_layer_types": text["mlp_layer_types"],
        "prompt_tokens": int(ids.shape[1]),
        "tensors_fetched": len(wanted),
        "bytes_fetched": fetched_bytes,
        "fetch_seconds": round(fetch_seconds, 1),
        "fp8_tensors_checked_bit_exact_vs_torch": fp8_checked,
        "reference": f"transformers {transformers.__version__} Glm5NextTextModel, float32, eager, torch {torch.__version__}",
        "streams_compared": [f"input to layer {i}" for i in range(n)] + [f"output of layer {n - 1}"],
        "ramp": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
