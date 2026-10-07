from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


GLM52_MODEL_ID = "zai-org/GLM-5.2"
GLM45_AIR_MODEL_ID = "zai-org/GLM-4.5-Air"


@dataclass(frozen=True)
class ConfigSummary:
    model_id: str
    model_type: str
    num_hidden_layers: int
    dense_layers: int
    sparse_layers: int
    n_routed_experts: int
    n_shared_experts: int | None
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int
    intermediate_size: int
    max_position_embeddings: int
    indexer_full_layers: int | None = None
    indexer_shared_layers: int | None = None


def hf_config_url(model_id: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{model_id}/raw/{revision}/config.json"


def fetch_hf_config(model_id: str, revision: str = "main", timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(hf_config_url(model_id, revision), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_config(config: dict[str, Any], *, model_id: str) -> ConfigSummary:
    num_hidden_layers = int(config["num_hidden_layers"])
    if "mlp_layer_types" in config:
        mlp_layer_types = list(config["mlp_layer_types"])
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError("mlp_layer_types length does not match num_hidden_layers")
        dense_layers = mlp_layer_types.count("dense")
        sparse_layers = mlp_layer_types.count("sparse")
    else:
        dense_layers = int(config.get("first_k_dense_replace", 0))
        sparse_layers = num_hidden_layers - dense_layers

    indexer_full_layers: int | None = None
    indexer_shared_layers: int | None = None
    if "indexer_types" in config:
        indexer_types = list(config["indexer_types"])
        if len(indexer_types) != num_hidden_layers:
            raise ValueError("indexer_types length does not match num_hidden_layers")
        indexer_full_layers = indexer_types.count("full")
        indexer_shared_layers = indexer_types.count("shared")

    return ConfigSummary(
        model_id=model_id,
        model_type=str(config["model_type"]),
        num_hidden_layers=num_hidden_layers,
        dense_layers=dense_layers,
        sparse_layers=sparse_layers,
        n_routed_experts=int(config["n_routed_experts"]),
        n_shared_experts=(
            None if config.get("n_shared_experts") is None else int(config["n_shared_experts"])
        ),
        num_experts_per_tok=int(config["num_experts_per_tok"]),
        hidden_size=int(config["hidden_size"]),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        intermediate_size=int(config["intermediate_size"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        indexer_full_layers=indexer_full_layers,
        indexer_shared_layers=indexer_shared_layers,
    )


def validate_glm52(
    summary: ConfigSummary,
    *,
    expected_n_routed_experts: int = 256,
) -> list[str]:
    checks = {
        "model_type": summary.model_type == "glm_moe_dsa",
        "num_hidden_layers": summary.num_hidden_layers == 78,
        "dense_layers": summary.dense_layers == 3,
        "sparse_layers": summary.sparse_layers == 75,
        "n_routed_experts": summary.n_routed_experts == expected_n_routed_experts,
        "num_experts_per_tok": summary.num_experts_per_tok == 8,
        "indexer_full_layers": summary.indexer_full_layers == 21,
        "indexer_shared_layers": summary.indexer_shared_layers == 57,
        "hidden_size": summary.hidden_size == 6144,
        "moe_intermediate_size": summary.moe_intermediate_size == 2048,
    }
    return [name for name, ok in checks.items() if not ok]


def validate_glm45_air(summary: ConfigSummary) -> list[str]:
    checks = {
        "model_type": summary.model_type == "glm4_moe",
        "num_hidden_layers": summary.num_hidden_layers == 46,
        "dense_layers": summary.dense_layers == 1,
        "sparse_layers": summary.sparse_layers == 45,
        "n_routed_experts": summary.n_routed_experts == 128,
        "num_experts_per_tok": summary.num_experts_per_tok == 8,
        "hidden_size": summary.hidden_size == 4096,
        "moe_intermediate_size": summary.moe_intermediate_size == 1408,
    }
    return [name for name, ok in checks.items() if not ok]


def inspect_required_configs(fetch_live: bool = True) -> dict[str, ConfigSummary]:
    if not fetch_live:
        raise ValueError("inspect_required_configs(fetch_live=False) requires caller-provided fixtures")
    return {
        GLM52_MODEL_ID: summarize_config(fetch_hf_config(GLM52_MODEL_ID), model_id=GLM52_MODEL_ID),
        GLM45_AIR_MODEL_ID: summarize_config(
            fetch_hf_config(GLM45_AIR_MODEL_ID), model_id=GLM45_AIR_MODEL_ID
        ),
    }
