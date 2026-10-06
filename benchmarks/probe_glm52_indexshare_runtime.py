import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask

from mlx_vq.convert.glm52_reap import (
    GLM52_REAP_CONFIG_SHA256,
    GLM52_REAP_INDEX_SHA256,
    GLM52_REAP_PROFILE_NAME,
)
from mlx_vq.models.glm4_moe_adapter import QuantizedVQSwitchGLU
from mlx_vq.models.glm52_policy import validate_glm52_config
from mlx_vq.models.glm52_vq_adapter import (
    GLM52VQModel,
    GLM52VQModelArgs,
    Glm52VQMoE,
    audit_glm52_indexshare_static_contract,
)
from mlx_vq.models.profiles import get_profile


PROBE_RECORD_TYPE = "glm52_indexshare_runtime_probe"
PROBE_READY_STATUS = "glm52_indexshare_runtime_probe_ready"
PROBE_FAILED_STATUS = "glm52_indexshare_runtime_probe_failed"
PINNED_MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
PINNED_REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"


@dataclass(frozen=True, slots=True)
class GLM52IndexShareIdentityContract:
    model_id: str
    revision: str
    profile_name: str
    config_sha256: str
    index_sha256: str


PINNED_GLM52_INDEXSHARE_CONTRACT = GLM52IndexShareIdentityContract(
    model_id=PINNED_MODEL_ID,
    revision=PINNED_REVISION,
    profile_name=GLM52_REAP_PROFILE_NAME,
    config_sha256=GLM52_REAP_CONFIG_SHA256,
    index_sha256=GLM52_REAP_INDEX_SHA256,
)


class GLM52IndexShareProbeError(ValueError):
    """A staged, evidence-safe IndexShare probe failure."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GLM52IndexShareProbeError(
            "input_json",
            f"could not read {label} {path}: {error}",
        ) from error
    if not isinstance(value, dict):
        raise GLM52IndexShareProbeError(
            "input_json",
            f"{label} {path} must contain a JSON object",
        )
    return value


def _base_payload(
    *,
    config_path: Path,
    index_path: Path,
    model_id: str,
    revision: str,
    contract: GLM52IndexShareIdentityContract,
) -> dict[str, Any]:
    production_identity_contract = contract == PINNED_GLM52_INDEXSHARE_CONTRACT
    return {
        "schema_version": 1,
        "record_type": PROBE_RECORD_TYPE,
        "probe_status": PROBE_FAILED_STATUS,
        "probe_pass": False,
        "readiness_scope": "indexshare_static_and_synthetic_runtime",
        "source_authority": (
            "pinned_huggingface_snapshot"
            if production_identity_contract
            else "test_fixture_contract"
        ),
        "production_identity_contract": production_identity_contract,
        "model_id": model_id,
        "source_revision": revision,
        "expected_model_id": contract.model_id,
        "expected_source_revision": contract.revision,
        "profile_name": contract.profile_name,
        "config_path": str(config_path),
        "index_path": str(index_path),
        "config_sha256": None,
        "expected_config_sha256": contract.config_sha256,
        "index_sha256": None,
        "expected_index_sha256": contract.index_sha256,
        "immutable_file_identity_pass": False,
        "profile_validation_pass": False,
        "profile_validation_failures": [],
        "static_contract_pass": False,
        "production_static_contract": False,
        "full_layer_count": 0,
        "shared_layer_count": 0,
        "main_indexer_tensor_count": 0,
        "mtp_indexer_tensor_count": 0,
        "synthetic_runtime_contract": False,
        "synthetic_runtime": {},
        "shared_missing_topk_guard": {},
        "production_long_context_proven": False,
        "whole_model_runtime_proven": False,
        "full_model_bind_proven": False,
        "generation_proven": False,
    }


def _tiny_indexshare_args() -> GLM52VQModelArgs:
    return GLM52VQModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=32,
        hidden_size=16,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        v_head_dim=4,
        qk_nope_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=["full", "shared"],
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        mlp_layer_types=["dense", "sparse"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )


def _cache_offsets(cache: Any) -> list[int]:
    return [int(member.offset) for member in cache.caches]


def _run_synthetic_runtime_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    mx.random.seed(5202)
    args = _tiny_indexshare_args()
    model = GLM52VQModel(args)
    full_layer, shared_layer = model.layers
    if not isinstance(shared_layer.mlp, Glm52VQMoE):
        raise GLM52IndexShareProbeError(
            "synthetic_runtime_contract",
            "synthetic shared layer did not construct the routed VQ MoE seam",
        )
    shared_layer.mlp.bind_switch_mlp(
        QuantizedVQSwitchGLU.from_weights(
            gate_weight=mx.full((2, 8, 16), 0.03125, dtype=mx.float32),
            up_weight=mx.full((2, 8, 16), -0.015625, dtype=mx.float32),
            down_weight=mx.full((2, 16, 8), 0.0078125, dtype=mx.float32),
            group_size=8,
        )
    )
    caches = model.make_cache()
    full_cache, shared_cache = caches

    prefill = mx.arange(3 * args.hidden_size, dtype=mx.float32).reshape(
        1, 3, args.hidden_size
    ) / 64.0
    prefill_mask = create_attention_mask(
        prefill,
        full_cache[0],
        return_array=True,
    )
    full_output, full_topk = full_layer(prefill, prefill_mask, full_cache, None)
    shared_output, shared_topk = shared_layer(
        full_output,
        prefill_mask,
        shared_cache,
        full_topk,
    )
    if full_topk is None or shared_topk is None:
        raise GLM52IndexShareProbeError(
            "synthetic_runtime_contract",
            "synthetic prefill did not produce and reuse full-layer top-k indices",
        )
    mx.eval(shared_output, full_topk, shared_topk)

    prefill_topk_reuse_exact = np.array_equal(
        np.asarray(shared_topk),
        np.asarray(full_topk),
    )
    prefill_full_cache_offsets = _cache_offsets(full_cache)
    prefill_shared_cache_offsets = _cache_offsets(shared_cache)
    prefill_output_finite = bool(mx.all(mx.isfinite(shared_output)).item())

    decode = mx.full((1, 1, args.hidden_size), 0.125, dtype=mx.float32)
    decode_mask = create_attention_mask(
        decode,
        full_cache[0],
        return_array=True,
    )
    full_decode, full_decode_topk = full_layer(
        decode,
        decode_mask,
        full_cache,
        None,
    )
    shared_decode, shared_decode_topk = shared_layer(
        full_decode,
        decode_mask,
        shared_cache,
        full_decode_topk,
    )
    if full_decode_topk is None or shared_decode_topk is None:
        raise GLM52IndexShareProbeError(
            "synthetic_runtime_contract",
            "synthetic decode did not produce and reuse full-layer top-k indices",
        )
    mx.eval(shared_decode, full_decode_topk, shared_decode_topk)

    decode_topk_reuse_exact = np.array_equal(
        np.asarray(shared_decode_topk),
        np.asarray(full_decode_topk),
    )
    decode_full_cache_offsets = _cache_offsets(full_cache)
    decode_shared_cache_offsets = _cache_offsets(shared_cache)
    decode_output_finite = bool(mx.all(mx.isfinite(shared_decode)).item())

    parameter_names = {name for name, _value in tree_flatten(model.parameters())}
    dense_routed_weight_present = any(
        ".mlp.experts." in name
        or (
            ".mlp.switch_mlp." in name
            and name.endswith(".weight")
        )
        for name in parameter_names
    )

    guard_model = GLM52VQModel(args)
    guard_cache = guard_model.make_cache()[1]
    guard_attention = guard_model.layers[1].self_attn
    guard_hidden = mx.ones(
        (1, args.index_topk + 1, args.hidden_size),
        dtype=mx.float32,
    )
    cache_offset_before = int(guard_cache[0].offset)
    guard_error: ValueError | None = None
    try:
        guard_attention(
            guard_hidden,
            cache=guard_cache,
            prev_topk_indices=None,
        )
    except ValueError as error:
        guard_error = error
    cache_offset_after = int(guard_cache[0].offset)
    guard_message = "" if guard_error is None else str(guard_error)
    guard_pass = bool(
        guard_error is not None
        and "requires top-k indices from a previous full layer" in guard_message
        and cache_offset_before == 0
        and cache_offset_after == 0
    )
    guard_record = {
        "guard_pass": guard_pass,
        "error_type": None if guard_error is None else type(guard_error).__name__,
        "message": guard_message,
        "cache_offset_before": cache_offset_before,
        "cache_offset_after": cache_offset_after,
    }

    runtime = {
        "index_topk": args.index_topk,
        "prefill_tokens": int(prefill.shape[1]),
        "prefill_topk_shape": list(full_topk.shape),
        "decode_topk_shape": list(full_decode_topk.shape),
        "prefill_topk_reuse_exact": prefill_topk_reuse_exact,
        "decode_topk_reuse_exact": decode_topk_reuse_exact,
        "prefill_full_cache_offsets": prefill_full_cache_offsets,
        "prefill_shared_cache_offsets": prefill_shared_cache_offsets,
        "decode_full_cache_offsets": decode_full_cache_offsets,
        "decode_shared_cache_offsets": decode_shared_cache_offsets,
        "prefill_output_finite": prefill_output_finite,
        "decode_output_finite": decode_output_finite,
        "dense_routed_weight_present": dense_routed_weight_present,
    }
    expected_checks = {
        "prefill_topk_shape": runtime["prefill_topk_shape"] == [1, 1, 3, 2],
        "decode_topk_shape": runtime["decode_topk_shape"] == [1, 1, 1, 2],
        "prefill_topk_reuse_exact": prefill_topk_reuse_exact,
        "decode_topk_reuse_exact": decode_topk_reuse_exact,
        "prefill_full_cache_offsets": prefill_full_cache_offsets == [3, 3],
        "prefill_shared_cache_offsets": prefill_shared_cache_offsets == [3],
        "decode_full_cache_offsets": decode_full_cache_offsets == [4, 4],
        "decode_shared_cache_offsets": decode_shared_cache_offsets == [4],
        "prefill_output_finite": prefill_output_finite,
        "decode_output_finite": decode_output_finite,
        "no_dense_routed_weight": not dense_routed_weight_present,
        "shared_missing_topk_guard": guard_pass,
    }
    failed_checks = [name for name, passed in expected_checks.items() if not passed]
    if failed_checks:
        raise GLM52IndexShareProbeError(
            "synthetic_runtime_contract",
            f"synthetic IndexShare runtime checks failed: {failed_checks}",
        )
    return runtime, guard_record


def probe_glm52_indexshare_runtime(
    *,
    config_path: str | Path,
    index_path: str | Path,
    model_id: str,
    revision: str,
    identity_contract: GLM52IndexShareIdentityContract = PINNED_GLM52_INDEXSHARE_CONTRACT,
) -> dict[str, Any]:
    resolved_config_path = _absolute(config_path)
    resolved_index_path = _absolute(index_path)
    payload = _base_payload(
        config_path=resolved_config_path,
        index_path=resolved_index_path,
        model_id=model_id,
        revision=revision,
        contract=identity_contract,
    )

    lineage_failures = []
    if model_id != identity_contract.model_id:
        lineage_failures.append("model_id")
    if revision != identity_contract.revision:
        lineage_failures.append("revision")
    if lineage_failures:
        raise GLM52IndexShareProbeError(
            "source_lineage",
            f"source lineage failed: {lineage_failures}",
        )

    try:
        config_sha256 = _sha256_file(resolved_config_path)
        index_sha256 = _sha256_file(resolved_index_path)
    except OSError as error:
        raise GLM52IndexShareProbeError(
            "immutable_file_identity",
            f"could not hash immutable probe input: {error}",
        ) from error
    payload["config_sha256"] = config_sha256
    payload["index_sha256"] = index_sha256
    identity_failures = []
    if config_sha256 != identity_contract.config_sha256:
        identity_failures.append(
            "config SHA-256 "
            f"{config_sha256} != {identity_contract.config_sha256}"
        )
    if index_sha256 != identity_contract.index_sha256:
        identity_failures.append(
            "index SHA-256 "
            f"{index_sha256} != {identity_contract.index_sha256}"
        )
    if identity_failures:
        raise GLM52IndexShareProbeError(
            "immutable_file_identity",
            "; ".join(identity_failures),
        )
    payload["immutable_file_identity_pass"] = True

    config = _load_json_object(resolved_config_path, label="GLM52 config")
    index = _load_json_object(resolved_index_path, label="GLM52 index")
    try:
        profile = get_profile(identity_contract.profile_name)
        profile_failures = validate_glm52_config(
            config,
            model_id=model_id,
            revision=revision,
            profile=profile,
        )
    except Exception as error:
        raise GLM52IndexShareProbeError(
            "profile_validation",
            f"could not validate profile {identity_contract.profile_name!r}: {error}",
        ) from error
    payload["profile_validation_failures"] = profile_failures
    if profile_failures:
        raise GLM52IndexShareProbeError(
            "profile_validation",
            f"GLM52 profile validation failed: {profile_failures}",
        )
    payload["profile_validation_pass"] = True

    try:
        static_report = audit_glm52_indexshare_static_contract(config, index)
    except ValueError as error:
        raise GLM52IndexShareProbeError(
            "production_static_contract",
            str(error),
        ) from error
    payload.update(
        {
            "static_contract_pass": static_report.audit_pass,
            "full_layers": list(static_report.full_layers),
            "shared_layers": list(static_report.shared_layers),
            "full_layer_count": len(static_report.full_layers),
            "shared_layer_count": len(static_report.shared_layers),
            "main_indexer_tensor_count": static_report.main_indexer_tensor_count,
            "mtp_indexer_tensor_count": len(static_report.mtp_indexer_tensors),
            "production_long_context_proven": (
                static_report.production_long_context_proven
            ),
        }
    )
    expected_static_counts = (
        payload["full_layer_count"] == 21
        and payload["shared_layer_count"] == 57
        and payload["main_indexer_tensor_count"] == 105
        and payload["mtp_indexer_tensor_count"] == 5
    )
    if not static_report.audit_pass or not expected_static_counts:
        raise GLM52IndexShareProbeError(
            "production_static_contract",
            "GLM52 IndexShare static report did not prove 21/57 layers and 105/5 tensors",
        )
    payload["production_static_contract"] = bool(
        payload["production_identity_contract"]
    )

    runtime, guard = _run_synthetic_runtime_contract()
    payload.update(
        {
            "synthetic_runtime": runtime,
            "shared_missing_topk_guard": guard,
            "synthetic_runtime_contract": True,
            "probe_status": PROBE_READY_STATUS,
            "probe_pass": True,
        }
    )
    return payload


def _failure_payload(
    *,
    config_path: str | Path,
    index_path: str | Path,
    model_id: str,
    revision: str,
    contract: GLM52IndexShareIdentityContract,
    error: BaseException,
) -> dict[str, Any]:
    payload = _base_payload(
        config_path=_absolute(config_path),
        index_path=_absolute(index_path),
        model_id=model_id,
        revision=revision,
        contract=contract,
    )
    payload["failure_stage"] = (
        error.stage
        if isinstance(error, GLM52IndexShareProbeError)
        else "unexpected_error"
    )
    payload["input_error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    return payload


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = _absolute(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pinned GLM-5.2 IndexShare production contract and run a "
            "tiny deterministic full-to-shared runtime/caching proof."
        )
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    identity_contract: GLM52IndexShareIdentityContract = PINNED_GLM52_INDEXSHARE_CONTRACT,
) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = probe_glm52_indexshare_runtime(
            config_path=args.config_path,
            index_path=args.index_path,
            model_id=args.model_id,
            revision=args.revision,
            identity_contract=identity_contract,
        )
    except Exception as error:  # Preserve durable failure evidence for every stage.
        payload = _failure_payload(
            config_path=args.config_path,
            index_path=args.index_path,
            model_id=args.model_id,
            revision=args.revision,
            contract=identity_contract,
            error=error,
        )
    _write_json_atomic(args.output_json, payload)
    return 0 if payload.get("probe_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
