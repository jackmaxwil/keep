#!/usr/bin/env python3
"""Diagnose GLM-5.2 candidate-cache capture against a plain model forward."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


def _candidate_api() -> Any:
    module_name = "mlx_vq.quality.glm52_candidate_eval"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/mlx_vq/quality/glm52_candidate_eval.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 candidate-eval module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _duplicate_key_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path)
    try:
        payload = json.loads(
            input_path.read_bytes(),
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {input_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {input_path} must contain a JSON object")
    return payload


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().resolve(strict=False)
    if output.suffix != ".json":
        raise ValueError("diagnostic output must use the .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial-{uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_f32_safetensors_logits(
    path: str | Path,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    shard_path = Path(path)
    if shard_path.is_symlink() or not shard_path.is_file():
        raise ValueError(f"{label} shard must be a regular non-symlink file: {shard_path}")
    try:
        file_bytes = shard_path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read {label} shard {shard_path}: {error}") from error
    if len(file_bytes) < 8:
        raise ValueError(f"{label} shard has a truncated safetensors prefix")
    header_length = struct.unpack("<Q", file_bytes[:8])[0]
    if header_length == 0 or header_length > 1024 * 1024:
        raise ValueError(f"{label} shard has invalid safetensors header length")
    data_offset = 8 + header_length
    if data_offset > len(file_bytes):
        raise ValueError(f"{label} shard has a truncated safetensors header")
    header_bytes = file_bytes[8:data_offset]
    stripped = header_bytes.rstrip(b" ")
    if header_bytes[len(stripped) :] != b" " * (len(header_bytes) - len(stripped)):
        raise ValueError(f"{label} shard header has non-space padding")
    try:
        header = json.loads(
            stripped,
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} shard has invalid safetensors JSON: {error}") from error
    if not isinstance(header, Mapping) or set(header) != {"logits"}:
        raise ValueError(f"{label} shard tensor inventory must contain only logits")
    descriptor = header["logits"]
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 4
    if not isinstance(descriptor, Mapping) or descriptor.get("dtype") != "F32":
        raise ValueError(f"{label} logits dtype must be F32")
    if descriptor.get("shape") != list(expected_shape):
        raise ValueError(
            f"{label} logits shape must be {expected_shape}, "
            f"found {descriptor.get('shape')}"
        )
    if descriptor.get("data_offsets") != [0, expected_bytes]:
        raise ValueError(f"{label} logits data offsets do not match the expected shape")
    raw = file_bytes[data_offset:]
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{label} shard physical extent mismatch: expected {expected_bytes} "
            f"logit bytes, found {len(raw)}"
        )
    logits = np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()
    if logits.dtype != np.dtype(np.float32):
        raise ValueError(f"{label} logits dtype must materialize as float32")
    if not np.isfinite(logits).all():
        raise ValueError(f"{label} logits contain a non-finite value")
    return logits


def _require_f32_logits(
    value: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    logits = np.asarray(value)
    if logits.dtype != np.dtype(np.float32):
        raise ValueError(f"{label} logits dtype must be float32, found {logits.dtype}")
    if logits.shape != expected_shape:
        raise ValueError(
            f"{label} logits shape must be {expected_shape}, found {logits.shape}"
        )
    if not logits.flags.c_contiguous:
        raise ValueError(f"{label} logits must be contiguous row-major")
    if not np.isfinite(logits).all():
        raise ValueError(f"{label} logits contain a non-finite value")
    return logits


def _sha256_f32(logits: np.ndarray) -> str:
    canonical = np.asarray(logits, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _mean_true_token_logprob(
    logits: np.ndarray,
    true_token_ids: np.ndarray,
) -> float:
    values = np.asarray(logits, dtype=np.float64)
    row_max = values.max(axis=-1, keepdims=True)
    shifted = values - row_max
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1)) + row_max[:, 0]
    selected = values[np.arange(values.shape[0]), true_token_ids]
    return float(np.mean(selected - log_normalizer))


def diagnose_capture_consistency(
    *,
    model: Any,
    prompt: Any,
    prompt_index: int,
    candidate_logits: np.ndarray,
    teacher_logits: np.ndarray,
    mx: Any,
) -> dict[str, Any]:
    """Run the trusted plain path and compare it with injected cache shards."""

    encoded = tuple(prompt.encoded_token_ids)
    if len(encoded) < 2 or prompt.token_count != len(encoded):
        raise ValueError("selected prompt has inconsistent or insufficient token IDs")
    vocab_size = int(model.args.vocab_size)
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in encoded):
        raise ValueError("selected prompt token ID is outside the model vocabulary")
    predictor_tokens = np.asarray(encoded[:-1], dtype=np.int32)[None, :]
    model_output = model(mx.array(predictor_tokens, dtype=mx.int32))
    plain_mx = model_output.astype(mx.float32)
    mx.eval(plain_mx)
    plain_raw = np.asarray(plain_mx)
    expected_3d = (1, len(encoded) - 1, vocab_size)
    if plain_raw.shape != expected_3d:
        raise ValueError(
            f"plain single-prompt logits shape must be {expected_3d}, "
            f"found {plain_raw.shape}"
        )
    plain_logits = np.ascontiguousarray(plain_raw[0], dtype=np.float32)
    expected_shape = (len(encoded) - 1, vocab_size)
    plain_logits = _require_f32_logits(
        plain_logits,
        expected_shape=expected_shape,
        label="plain single-prompt",
    )
    candidate = _require_f32_logits(
        candidate_logits,
        expected_shape=expected_shape,
        label="candidate cache",
    )
    teacher = _require_f32_logits(
        teacher_logits,
        expected_shape=expected_shape,
        label="teacher cache",
    )
    true_tokens = np.asarray(encoded[1:], dtype=np.int64)
    plain_top1 = np.argmax(plain_logits, axis=-1).astype(np.int64)
    candidate_top1 = np.argmax(candidate, axis=-1).astype(np.int64)
    teacher_top1 = np.argmax(teacher, axis=-1).astype(np.int64)
    positions: list[dict[str, Any]] = []
    for position in range(expected_shape[0]):
        plain_row = plain_logits[position]
        candidate_row = candidate[position]
        positions.append(
            {
                "position": position,
                "max_abs_diff": float(
                    np.max(
                        np.abs(
                            plain_row.astype(np.float64)
                            - candidate_row.astype(np.float64)
                        )
                    )
                ),
                "exact_equal": bool(np.array_equal(plain_row, candidate_row)),
                "plain_top1_token_id": int(plain_top1[position]),
                "candidate_cache_top1_token_id": int(candidate_top1[position]),
                "top1_match": bool(plain_top1[position] == candidate_top1[position]),
            }
        )
    exact_count = sum(row["exact_equal"] for row in positions)
    top1_match_count = sum(row["top1_match"] for row in positions)
    passed = exact_count == len(positions)
    return {
        "schema_version": 1,
        "record_type": "glm52_candidate_capture_consistency_diagnostic",
        "status": "capture_exact_match" if passed else "capture_mismatch",
        "capture_consistency_pass": passed,
        "prompt_index": prompt_index,
        "prompt_id": prompt.prompt_id,
        "split": prompt.split,
        "domain": prompt.domain,
        "predictor_token_ids": list(encoded[:-1]),
        "true_next_token_ids": true_tokens.tolist(),
        "position_count": len(positions),
        "exact_position_count": exact_count,
        "top1_match_position_count": top1_match_count,
        "plain_single_prompt": {
            "shape": list(plain_logits.shape),
            "dtype": "float32",
            "sha256": _sha256_f32(plain_logits),
            "top1_token_ids": plain_top1.tolist(),
            "mean_true_token_logprob": _mean_true_token_logprob(
                plain_logits, true_tokens
            ),
        },
        "candidate_cache": {
            "shape": list(candidate.shape),
            "dtype": "float32",
            "sha256": _sha256_f32(candidate),
            "top1_token_ids": candidate_top1.tolist(),
            "mean_true_token_logprob": _mean_true_token_logprob(
                candidate, true_tokens
            ),
        },
        "teacher_cache": {
            "shape": list(teacher.shape),
            "dtype": "float32",
            "sha256": _sha256_f32(teacher),
            "top1_token_ids": teacher_top1.tolist(),
            "mean_true_token_logprob": _mean_true_token_logprob(
                teacher, true_tokens
            ),
        },
        "positions": positions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind the production GLM-5.2 composite and compare one trusted plain "
            "single-prompt forward with its candidate-cache capture."
        )
    )
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-index-path", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--non-vq-evidence-json", required=True)
    parser.add_argument("--routed-artifact-dir", required=True)
    parser.add_argument("--composite-audit-json", required=True)
    parser.add_argument("--materialization-runs-jsonl", required=True)
    parser.add_argument("--full-bind-preflight-json", required=True)
    parser.add_argument("--candidate-cache-root", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidate_api = _candidate_api()
    contract = candidate_api.build_glm52_candidate_contract_from_teacher_cache(
        args.teacher_cache_root,
        prompt_pack_path=args.prompt_pack_json,
    )
    if args.prompt_index < 0 or args.prompt_index >= len(contract.prompts):
        raise ValueError(
            f"prompt index must be in [0, {len(contract.prompts) - 1}], "
            f"found {args.prompt_index}"
        )
    prompt = contract.prompts[args.prompt_index]
    composite = importlib.import_module("mlx_vq.models.glm52_composite_loader")
    readiness = _load_json_object(
        args.tokenizer_readiness_json,
        label="tokenizer readiness",
    )
    authenticated_prompt = readiness.get("prompt")
    if not isinstance(authenticated_prompt, str) or not authenticated_prompt:
        raise ValueError("tokenizer readiness must contain its authenticated prompt")
    validated = composite.validate_glm52_production_inputs(
        profile_path=args.profile_path,
        config_path=args.config_path,
        source_index_path=args.source_index_path,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_readiness_json=args.tokenizer_readiness_json,
        family_policy_json=args.family_policy_json,
        non_vq_artifact_dir=args.non_vq_artifact_dir,
        non_vq_evidence_json=args.non_vq_evidence_json,
        routed_artifact_dir=args.routed_artifact_dir,
        composite_audit_json=args.composite_audit_json,
        materialization_runs_jsonl=args.materialization_runs_jsonl,
        full_bind_preflight_json=args.full_bind_preflight_json,
        model_id=contract.source["model_id"],
        revision=contract.source["revision"],
        prompt=authenticated_prompt,
    )
    expected_identity = candidate_api.GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
    if validated.artifact_identity.sha256 != expected_identity:
        raise ValueError("authenticated composite artifact identity drifted")
    model, load_report = composite.load_authenticated_glm52_composite(validated)
    if load_report.artifact_identity_sha256 != expected_identity:
        raise ValueError("strict composite load report identity drifted")
    composite.assert_glm52_production_inputs_unchanged(validated.input_fingerprint)
    expected_shape = (prompt.token_count - 1, int(model.args.vocab_size))
    relative = Path(candidate_api.cache_api.SHARD_DIRECTORY) / (
        f"{prompt.prompt_id}.safetensors"
    )
    candidate_logits = _read_f32_safetensors_logits(
        Path(args.candidate_cache_root) / relative,
        expected_shape=expected_shape,
        label="candidate cache",
    )
    teacher_logits = _read_f32_safetensors_logits(
        Path(args.teacher_cache_root) / relative,
        expected_shape=expected_shape,
        label="teacher cache",
    )
    mx = importlib.import_module("mlx.core")
    evidence = diagnose_capture_consistency(
        model=model,
        prompt=prompt,
        prompt_index=args.prompt_index,
        candidate_logits=candidate_logits,
        teacher_logits=teacher_logits,
        mx=mx,
    )
    composite.assert_glm52_production_inputs_unchanged(validated.input_fingerprint)
    evidence["candidate_artifact_identity_sha256"] = expected_identity
    evidence["candidate_cache_root"] = str(Path(args.candidate_cache_root))
    evidence["teacher_cache_root"] = str(Path(args.teacher_cache_root))
    evidence["prompt_pack_path"] = str(Path(args.prompt_pack_json))
    _write_json_atomic(args.output_json, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["capture_consistency_pass"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except Exception as error:
        print(
            json.dumps(
                {
                    "completed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
