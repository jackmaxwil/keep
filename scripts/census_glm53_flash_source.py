#!/usr/bin/env python3
"""Header-only census of the GLM-5.3-Flash source checkpoint.

Reads ``config.json``, the safetensors index and every shard's safetensors
header at a pinned revision. Hugging Face serves headers through HTTP range
requests, so this costs about 20 MB of transfer and never downloads weights.

Writes one JSON with measured parameter counts and on-disk bytes per tensor
class, dtype and shape patterns, the per-layer expert byte layout across
shards, and a ``derived`` block whose numbers are arithmetic on the measured
counts (labelled projections, not measurements).

Usage:
    uv run python scripts/census_glm53_flash_source.py \
        --out artifacts/census/glm53-flash-source-census-eb9eb208.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download

MODEL_ID = "zai-org/GLM-5.3-Flash"
# Must stay in sync with ``revision:`` in models/glm-5.3-flash.yaml.
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def tensor_class(name: str, mtp_layer: int) -> str:
    if ".visual." in name:
        return "vision"
    match = _LAYER.search(name)
    if match and int(match.group(1)) == mtp_layer:
        return "mtp_routed" if ".mlp.experts." in name else "mtp_resident"
    return "routed" if ".mlp.experts." in name else "resident"


def pattern(name: str) -> str:
    name = re.sub(r"\.experts\.\d+\.", ".experts.E.", name)
    return re.sub(r"\.(layers|blocks)\.\d+\.", r".\1.N.", name)


def census(revision: str) -> dict:
    config = json.loads(Path(hf_hub_download(MODEL_ID, "config.json", revision=revision)).read_text())
    text = config["text_config"]
    mtp_layer = text["num_hidden_layers"]
    metadata = get_safetensors_metadata(MODEL_ID, revision=revision)
    shards = sorted(metadata.files_metadata)

    params = Counter()
    nbytes = Counter()
    patterns: dict[tuple, int] = Counter()
    layout: dict[int, dict[str, int]] = defaultdict(Counter)
    for shard in shards:
        for name, info in metadata.files_metadata[shard].tensors.items():
            cls = tensor_class(name, mtp_layer)
            size = info.data_offsets[1] - info.data_offsets[0]
            nbytes[cls] += size
            if not name.endswith("weight_scale_inv"):
                params[cls] += math.prod(info.shape)
            patterns[(pattern(name), info.dtype, tuple(info.shape))] += 1
            if ".mlp.experts." in name:
                layout[int(_LAYER.search(name).group(1))][shard] += size

    n_sparse = text["mlp_layer_types"].count("sparse")
    experts, top_k = text["n_routed_experts"], text["num_experts_per_tok"]
    per_expert = 3 * text["hidden_size"] * text["moe_intermediate_size"]
    assert params["routed"] == n_sparse * experts * per_expert, "routed census disagrees with config"
    # Decode reads every resident each token except the embedding (a row gather).
    non_embed_resident_bytes = nbytes["resident"] - text["vocab_size"] * text["hidden_size"] * 2

    def artifact_gb(bpw: float, count: int) -> float:
        return round(count * bpw / 8 / 1e9, 2)

    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "architecture": config["architectures"],
        "text_model_type": text["model_type"],
        "shards": len(shards),
        "tensors": sum(patterns.values()),
        "params": dict(params),
        "params_total": sum(params.values()),
        "bytes": dict(nbytes),
        "bytes_total": sum(nbytes.values()),
        "layer_types": Counter(text["layer_types"]),
        "kda_layers": text["linear_attn_config"]["kda_layers"],
        "full_attn_layers": text["linear_attn_config"]["full_attn_layers"],
        "sparse_moe_layers": n_sparse,
        "mtp_layer": mtp_layer,
        "expert_layout_bytes_by_shard": {str(k): dict(v) for k, v in sorted(layout.items())},
        "layers_with_experts_in_more_than_one_shard": sum(len(v) > 1 for v in layout.values()),
        "patterns": [
            {"name": p, "dtype": d, "shape": list(s), "count": n}
            for (p, d, s), n in sorted(patterns.items())
        ],
        "derived": {
            "_note": "projections: arithmetic on the measured counts above, not measurements",
            "active_routed_params_per_token": n_sparse * top_k * per_expert,
            "routed_artifact_gb_at_bpw": {
                str(b): artifact_gb(b, params["routed"]) for b in (2.031, 2.062, 2.125, 2.5, 3.0)
            },
            "mtp_routed_artifact_gb_at_2.031": artifact_gb(2.031, params["mtp_routed"]),
            "decode_resident_bytes_per_token_source_precision": non_embed_resident_bytes,
            "decode_routed_bytes_per_token_at_2.031": int(n_sparse * top_k * per_expert * 2.031 / 8),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = census(args.revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("params", "bytes", "params_total", "bytes_total")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
