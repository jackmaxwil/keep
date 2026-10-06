from __future__ import annotations

import argparse
import json
import resource
import socket
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model


class _PipelineGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = int(rank)
        self._size = int(size)

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _custom_layer_bounds(
    *,
    num_hidden_layers: int,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
) -> tuple[int, int] | None:
    if layer_split is None:
        return None
    if pipeline_size != 2:
        raise ValueError("--layer-split is currently supported only with --pipeline-size 2")
    if rank not in {0, 1}:
        raise ValueError("--layer-split expects ranks 0 and 1")
    if layer_split <= 0 or layer_split >= num_hidden_layers:
        raise ValueError(
            f"--layer-split must be between 1 and {num_hidden_layers - 1}, got {layer_split}"
        )
    if rank == 0:
        return layer_split, num_hidden_layers
    return 0, layer_split


def _apply_pipeline_split(
    language_model: Any,
    *,
    rank: int,
    pipeline_size: int,
    layer_split: int | None,
) -> None:
    if layer_split is None:
        language_model.pipeline(_PipelineGroup(rank, pipeline_size))
        return

    start_idx, end_idx = _custom_layer_bounds(
        num_hidden_layers=len(language_model.layers),
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    language_model.pipeline_rank = rank
    language_model.pipeline_size = pipeline_size
    language_model.start_idx = start_idx
    language_model.end_idx = end_idx
    language_model.layers = language_model.layers[:end_idx]
    language_model.layers[:start_idx] = [None] * start_idx


def _parse_rank_view_roots(raw: str) -> dict[int, Path]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--rank-view-roots-json must be a JSON object")

    roots: dict[int, Path] = {}
    for key, value in payload.items():
        try:
            rank = int(key)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(f"rank key {key!r} is not an integer") from error
        if rank < 0:
            raise argparse.ArgumentTypeError(f"rank key {key!r} must be non-negative")
        if not isinstance(value, str) or not value:
            raise argparse.ArgumentTypeError(f"rank {rank} root must be a non-empty string")
        roots[rank] = Path(value)
    if not roots:
        raise argparse.ArgumentTypeError("--rank-view-roots-json must contain at least one rank")
    return roots


def _gb_to_bytes(value: float | None, *, name: str, allow_zero: bool = False) -> int | None:
    if value is None:
        return None
    if value < 0 or (value == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"{name} must be {requirement}")
    return int(value * 1024**3)


def _apply_mlx_memory_settings(
    *,
    memory_limit_bytes: int | None,
    cache_limit_bytes: int | None,
    wired_limit_bytes: int | None,
) -> dict[str, int | None]:
    settings: dict[str, int | None] = {
        "mlx_memory_limit_bytes": memory_limit_bytes,
        "mlx_cache_limit_bytes": cache_limit_bytes,
        "mlx_wired_limit_bytes": wired_limit_bytes,
        "previous_mlx_memory_limit_bytes": None,
        "previous_mlx_cache_limit_bytes": None,
        "previous_mlx_wired_limit_bytes": None,
    }
    if memory_limit_bytes is not None:
        settings["previous_mlx_memory_limit_bytes"] = int(mx.set_memory_limit(memory_limit_bytes))
    if cache_limit_bytes is not None:
        settings["previous_mlx_cache_limit_bytes"] = int(mx.set_cache_limit(cache_limit_bytes))
    if wired_limit_bytes is not None:
        settings["previous_mlx_wired_limit_bytes"] = int(mx.set_wired_limit(wired_limit_bytes))
    if cache_limit_bytes == 0:
        mx.clear_cache()
    return settings


def _parse_token_ids(raw: str) -> list[int]:
    if not raw.strip():
        raise argparse.ArgumentTypeError("--input-token-ids must not be empty")
    token_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("--input-token-ids contains an empty entry")
        try:
            token_id = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"token id {item!r} is not an integer") from error
        if token_id < 0:
            raise argparse.ArgumentTypeError(f"token id {token_id} must be non-negative")
        token_ids.append(token_id)
    return token_ids


def _read_rank_plan(rank_dir: Path) -> dict[str, Any] | None:
    path = rank_dir / "pipeline_view_plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_summary(parameter_keys: list[str]) -> dict[str, Any]:
    layers = sorted(
        {int(key.split(".")[2]) for key in parameter_keys if key.startswith("model.layers.")}
    )
    if not layers:
        return {"layers": [], "first_layer": None, "last_layer": None, "layer_count": 0}
    return {
        "layers": layers,
        "first_layer": layers[0],
        "last_layer": layers[-1],
        "layer_count": len(layers),
    }


def _key_match_summary(
    *,
    after_keys: list[str],
    rank_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if rank_plan is None:
        return {
            "expected_parameter_key_count": None,
            "parameter_keys_match_plan": None,
            "missing_after_key_count": None,
            "unexpected_after_key_count": None,
            "missing_after_key_examples": [],
            "unexpected_after_key_examples": [],
        }
    expected_keys = set(str(key) for key in rank_plan.get("parameter_keys", []))
    after_key_set = set(after_keys)
    missing_after_keys = sorted(expected_keys - after_key_set)
    unexpected_after_keys = sorted(after_key_set - expected_keys)
    return {
        "expected_parameter_key_count": len(expected_keys),
        "parameter_keys_match_plan": not missing_after_keys and not unexpected_after_keys,
        "missing_after_key_count": len(missing_after_keys),
        "unexpected_after_key_count": len(unexpected_after_keys),
        "missing_after_key_examples": missing_after_keys[:20],
        "unexpected_after_key_examples": unexpected_after_keys[:20],
    }


def _sentinel_sums(
    parameters: dict[str, mx.array],
    *,
    max_elements: int,
) -> list[dict[str, Any]]:
    layernorm_keys = [
        key
        for key in sorted(parameters)
        if key.endswith(".input_layernorm.weight")
        or key.endswith(".post_attention_layernorm.weight")
    ]
    fallback_keys = [key for key in ["model.norm.weight", "lm_head.weight"] if key in parameters]
    sentinel_keys = layernorm_keys + fallback_keys
    results: list[dict[str, Any]] = []
    for key in sentinel_keys[:2]:
        value = parameters[key]
        flat = value.reshape((-1,))
        sample = flat[: min(max_elements, flat.size)]
        sample_sum = mx.sum(sample.astype(mx.float32))
        mx.eval(sample_sum)
        results.append(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sampled_elements": int(sample.size),
                "sample_sum": float(sample_sum.item()),
            }
        )
    return results


def _all_sum_barrier(rank: int) -> float:
    value = mx.array([float(rank + 1)], dtype=mx.float32)
    summed = mx.distributed.all_sum(value)
    mx.eval(summed)
    return float(summed[0].item())


def _forward_probe(
    model: Any,
    *,
    token_ids: list[int],
    logit_sample_count: int,
) -> dict[str, Any]:
    tokens = mx.array([token_ids], dtype=mx.int32)
    start = time.perf_counter()
    logits = model(tokens)
    final_logits = logits[0, -1]
    sample_count = min(logit_sample_count, final_logits.shape[-1])
    sample = final_logits[:sample_count].astype(mx.float32)
    sample_sum = mx.sum(sample)
    top1 = mx.argmax(final_logits)
    mx.eval(sample_sum, top1)
    elapsed = time.perf_counter() - start
    return {
        "forward_attempted": True,
        "forward_ok": True,
        "forward_seconds": elapsed,
        "input_token_count": len(token_ids),
        "logits_shape": list(logits.shape),
        "logit_sample_count": int(sample_count),
        "logit_sample_sum": float(sample_sum.item()),
        "top1_id": int(top1.item()),
    }


def probe_rank(
    *,
    rank_dir: Path,
    rank: int,
    distributed_size: int,
    pipeline_size: int,
    layer_split: int | None,
    mlx_memory_settings: dict[str, int | None],
    evaluate_sentinels: bool,
    sentinel_max_elements: int,
    attempt_forward: bool,
    input_token_ids: list[int],
    logit_sample_count: int,
) -> dict[str, Any]:
    if not rank_dir.exists():
        raise FileNotFoundError(f"rank {rank} view root does not exist: {rank_dir}")

    mx.reset_peak_memory()
    rank_plan = _read_rank_plan(rank_dir)
    start = time.perf_counter()
    model, _config = load_model(rank_dir, lazy=True, strict=False)
    load_seconds = time.perf_counter() - start
    before_keys = [key for key, _ in tree_flatten(model.parameters())]
    _apply_pipeline_split(
        model.model,
        rank=rank,
        pipeline_size=pipeline_size,
        layer_split=layer_split,
    )
    parameters = dict(tree_flatten(model.parameters()))
    after_keys = list(parameters)

    result: dict[str, Any] = {
        "host": socket.gethostname(),
        "rank": rank,
        "distributed_size": distributed_size,
        "pipeline_size": pipeline_size,
        "layer_split": layer_split,
        "mlx_memory_settings": mlx_memory_settings,
        "rank_dir": str(rank_dir),
        "lazy_load_seconds": load_seconds,
        "did_eval_full_weights": False,
        "before_pipeline_parameter_key_count": len(before_keys),
        "after_pipeline_parameter_key_count": len(after_keys),
        "has_embed": "model.embed_tokens.weight" in parameters,
        "has_norm": "model.norm.weight" in parameters,
        "has_lm_head": "lm_head.weight" in parameters,
        **_layer_summary(after_keys),
        **_key_match_summary(after_keys=after_keys, rank_plan=rank_plan),
    }

    if rank_plan is not None:
        result.update(
            {
                "required_present_tensor_bytes": rank_plan.get("required_present_tensor_bytes"),
                "visible_shard_file_bytes": rank_plan.get("visible_shard_file_bytes"),
            }
        )

    if evaluate_sentinels:
        result["sentinel_results"] = _sentinel_sums(
            parameters,
            max_elements=sentinel_max_elements,
        )
        result["did_eval_sentinels"] = True
    else:
        result["sentinel_results"] = []
        result["did_eval_sentinels"] = False

    result["pre_forward_all_sum"] = _all_sum_barrier(rank)

    if attempt_forward:
        result.update(
            _forward_probe(
                model,
                token_ids=input_token_ids,
                logit_sample_count=logit_sample_count,
            )
        )
        result["post_forward_all_sum"] = _all_sum_barrier(rank)
        result["did_eval_full_weights"] = True
    else:
        result["forward_attempted"] = False
        result["forward_ok"] = None

    result.update(
        {
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
            "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe upstream GLM-4.5-Air rank-local pipeline views inside an MLX "
            "distributed launch, with an opt-in real forward pass."
        )
    )
    parser.add_argument("--backend", default="jaccl", choices=["any", "ring", "jaccl"])
    parser.add_argument("--rank-view-roots-json", required=True, type=_parse_rank_view_roots)
    parser.add_argument("--pipeline-size", type=int)
    parser.add_argument(
        "--layer-split",
        type=int,
        help=(
            "Optional two-rank split boundary. Rank 1 owns layers below this "
            "index and rank 0 owns this layer through the final layer."
        ),
    )
    parser.add_argument("--sentinel-max-elements", type=int, default=65536)
    parser.add_argument("--skip-sentinels", action="store_true")
    parser.add_argument("--attempt-forward", action="store_true")
    parser.add_argument("--input-token-ids", type=_parse_token_ids, default=[1, 2])
    parser.add_argument("--logit-sample-count", type=int, default=16)
    parser.add_argument(
        "--mlx-memory-limit-gb",
        type=float,
        help="Optional MLX graph-evaluation memory limit in GiB.",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        help="Optional MLX free-cache limit in GiB. Use 0 to disable the cache.",
    )
    parser.add_argument(
        "--mlx-wired-limit-gb",
        type=float,
        help="Optional MLX wired-memory limit in GiB.",
    )
    parser.add_argument(
        "--require-two-ranks",
        action="store_true",
        help="Exit nonzero unless the launcher created at least two ranks.",
    )
    args = parser.parse_args()

    if args.sentinel_max_elements <= 0:
        parser.error("--sentinel-max-elements must be positive")
    if args.logit_sample_count <= 0:
        parser.error("--logit-sample-count must be positive")
    if args.layer_split is not None and args.layer_split <= 0:
        parser.error("--layer-split must be positive")
    try:
        memory_limit_bytes = _gb_to_bytes(args.mlx_memory_limit_gb, name="--mlx-memory-limit-gb")
        cache_limit_bytes = _gb_to_bytes(
            args.mlx_cache_limit_gb,
            name="--mlx-cache-limit-gb",
            allow_zero=True,
        )
        wired_limit_bytes = _gb_to_bytes(args.mlx_wired_limit_gb, name="--mlx-wired-limit-gb")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    group = mx.distributed.init(strict=args.require_two_ranks, backend=args.backend)
    rank = int(group.rank())
    distributed_size = int(group.size())
    if args.require_two_ranks and distributed_size < 2:
        raise SystemExit("--require-two-ranks needs at least two distributed ranks")
    pipeline_size = args.pipeline_size or distributed_size
    if pipeline_size <= 0:
        parser.error("--pipeline-size must be positive")
    if args.layer_split is not None and pipeline_size != 2:
        parser.error("--layer-split is currently supported only with --pipeline-size 2")
    if args.require_two_ranks and pipeline_size != distributed_size:
        parser.error("--pipeline-size must match distributed size when --require-two-ranks is set")
    mlx_memory_settings = _apply_mlx_memory_settings(
        memory_limit_bytes=memory_limit_bytes,
        cache_limit_bytes=cache_limit_bytes,
        wired_limit_bytes=wired_limit_bytes,
    )
    if rank not in args.rank_view_roots_json:
        available = ", ".join(str(key) for key in sorted(args.rank_view_roots_json))
        raise SystemExit(f"no rank view root for rank {rank}; available ranks: {available}")

    result = probe_rank(
        rank_dir=args.rank_view_roots_json[rank],
        rank=rank,
        distributed_size=distributed_size,
        pipeline_size=pipeline_size,
        layer_split=args.layer_split,
        mlx_memory_settings=mlx_memory_settings,
        evaluate_sentinels=not args.skip_sentinels,
        sentinel_max_elements=args.sentinel_max_elements,
        attempt_forward=args.attempt_forward,
        input_token_ids=args.input_token_ids,
        logit_sample_count=args.logit_sample_count,
    )
    result["backend"] = args.backend
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
