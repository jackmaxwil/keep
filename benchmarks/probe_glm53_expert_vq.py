#!/usr/bin/env python3
"""Phase 3 probe: real GLM-5.3-Flash experts, fetched by byte range, through KEEP VQ.

For each chosen (layer, expert), reads the three FP8 projections and their F32
128x128 ``weight_scale_inv`` straight out of the pinned shards with HTTP range
requests (about 25 MB per expert, nothing else is downloaded), decodes them
with ``keep.convert.fp8_block``, and fits each projection at every point of a
rate ladder with KEEP's importance-aware lattice VQ.

Importance is **uniform** here: no calibration traffic exists yet, so these
rows measure the codebook and group-size trade on the real weight
distribution, not the final imatrix-weighted error. The whole-expert proxy
pushes unit-variance synthetic inputs through the clamped SwiGLU, so its scale
per channel is not the model's either. Both are labelled in every row.

Usage:
    uv run python benchmarks/probe_glm53_expert_vq.py \
        --out artifacts/quality/glm53-expert-vq-probe-uniform-20261006.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from huggingface_hub import HfFileSystem, get_safetensors_metadata

from keep.convert.fp8_block import dequantize_fp8_block
from keep.convert.dsv4_vq_pilot import expert_block_proxy, fit_projection, rate_bpw, reconstruct_quantized
from ramp.models.glm5_next_policy import GLM5_NEXT_MODEL_ID, GLM5_NEXT_REVISION

PROJECTIONS = ("gate", "up", "down")
LADDER = ((8, 512), (8, 32), (16, 512), (16, 256), (16, 128), (16, 32))


def fetch_expert(fs, metadata, layer: int, expert: int) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    weights, digests = {}, {}
    for proj in PROJECTIONS:
        stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
        raw = {}
        for suffix in ("weight", "weight_scale_inv"):
            name = f"{stem}.{suffix}"
            shard = metadata.weight_map[name]
            info = metadata.files_metadata[shard].tensors[name]
            base = 8 + header_length(fs, shard)
            data = fs.cat_file(_path(shard), start=base + info.data_offsets[0], end=base + info.data_offsets[1])
            digests[name] = hashlib.sha256(data).hexdigest()
            dtype = np.uint8 if info.dtype == "F8_E4M3" else np.float32
            raw[suffix] = np.frombuffer(data, dtype=dtype).reshape(info.shape)
        weights[proj] = dequantize_fp8_block(raw["weight"], raw["weight_scale_inv"])
    return weights, digests


_HEADER_LENGTH: dict[str, int] = {}


def _path(shard: str) -> str:
    return f"{GLM5_NEXT_MODEL_ID}@{GLM5_NEXT_REVISION}/{shard}"


def header_length(fs, shard: str) -> int:
    if shard not in _HEADER_LENGTH:
        _HEADER_LENGTH[shard] = int.from_bytes(fs.cat_file(_path(shard), start=0, end=8), "little")
    return _HEADER_LENGTH[shard]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 23, 44])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 143, 287])
    parser.add_argument("--backend", default="metal")
    args = parser.parse_args(argv)

    fs = HfFileSystem()
    metadata = get_safetensors_metadata(GLM5_NEXT_MODEL_ID, revision=GLM5_NEXT_REVISION)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        for layer in args.layers:
            for expert in args.experts:
                started = time.perf_counter()
                weights, digests = fetch_expert(fs, metadata, layer, expert)
                fetch_seconds = time.perf_counter() - started
                uniform = np.ones(weights["gate"].shape[1], dtype=np.float32)
                for code_bits, group in LADDER:
                    recon, fits = {}, {}
                    for proj in PROJECTIONS:
                        importance = np.ones(weights[proj].shape[1], dtype=np.float32)
                        quantized, metrics, seconds = fit_projection(
                            weights[proj], importance, group_size=group, code_bits=code_bits, backend=args.backend
                        )
                        recon[proj] = reconstruct_quantized(quantized)
                        fits[proj] = {**metrics.as_dict(), "fit_seconds": seconds}
                    block = expert_block_proxy(weights, recon, column_sigma=uniform, swiglu_limit=10.0)
                    row = {
                        "record_type": "glm53_expert_vq_probe_v1",
                        "model_id": GLM5_NEXT_MODEL_ID,
                        "revision": GLM5_NEXT_REVISION,
                        "layer": layer,
                        "expert": expert,
                        "code_bits": code_bits,
                        "group_size": group,
                        "bpw": rate_bpw(code_bits, group),
                        "importance": "uniform",
                        "block_proxy_inputs": "unit-variance gaussian, 512 tokens",
                        "projections": fits,
                        **block,
                        "fetch_seconds": fetch_seconds,
                        "source_tensor_sha256": digests,
                        "backend": args.backend,
                        "chip": chip,
                        "python": platform.python_version(),
                    }
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
                    sink.flush()
                    print(
                        f"L{layer} E{expert} {code_bits}b g{group} {row['bpw']:.3f} bpw "
                        f"block cos {block['block_cosine']:.4f} rel {block['block_relative_mse']:.3e}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
