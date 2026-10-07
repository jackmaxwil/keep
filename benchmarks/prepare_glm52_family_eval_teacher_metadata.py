from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from keep.convert.glm52_reap import (
    GLM52_REAP_CONFIG_SHA256,
    GLM52_REAP_INDEX_SHA256,
    audit_glm52_reap_source_index,
    audit_glm52_reap_source_payloads,
)
from keep.convert.stream_convert import load_safetensors_index
from ramp.models.profiles import load_profile
from keep.quality.glm52_family import (
    GLM52_MODEL_VOCAB_SIZE,
    GLM52_PINNED_TOKENIZER_FILES,
    PINNED_GLM52_MODEL_ID,
    PINNED_GLM52_REVISION,
    canonical_sha256,
    sha256_file,
    validate_glm52_family_gate_policy,
    validate_glm52_family_prompt_pack,
    validate_glm52_tokenizer_readiness,
)


METADATA_FILENAME = "glm52_teacher_source_metadata.jsonl"
METADATA_RECORD_TYPE = "glm52_family_eval_teacher_metadata_probe"
METADATA_READY_STATUS = "glm52_teacher_source_metadata_ready"
TEACHER_SOURCE_KIND = "deterministically_dequantized_pinned_modelopt_nvfp4"


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise ValueError(f"teacher metadata output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_identity(model_id: str, revision: str) -> None:
    if model_id != PINNED_GLM52_MODEL_ID:
        raise ValueError(f"model_id must be {PINNED_GLM52_MODEL_ID!r}")
    if revision != PINNED_GLM52_REVISION:
        raise ValueError(f"revision must be {PINNED_GLM52_REVISION!r}")


def _validate_source_payload_audit_evidence(
    *,
    recorded: Mapping[str, Any],
    fresh: Mapping[str, Any],
) -> None:
    """Compare payload evidence while accepting one known additive schema field.

    The checked 20260709 full-payload evidence predates ``requested_groups``.  A
    full audit has no explicit group selection, so the only equivalent legacy
    representation is an omitted field versus the current empty list.  Every
    other field remains exact.
    """

    normalized = dict(recorded)
    if "requested_groups" not in normalized and fresh.get("requested_groups") == []:
        normalized["requested_groups"] = []
    if normalized != dict(fresh):
        raise ValueError(
            "source payload audit evidence does not match live pinned headers"
        )


def _validate_prompt_pack_provenance(
    *,
    prompt_pack: Mapping[str, Any],
    tokenizer_dir: Path,
    tokenizer_readiness_path: Path,
    tokenizer_readiness_sha256: str,
    tokenizer_identities: Mapping[str, Mapping[str, int | str]],
    family_policy_path: Path,
    family_policy_sha256: str,
    family_policy_contract_sha256: str,
) -> None:
    expected_scalars = {
        "tokenizer_readiness_sha256": tokenizer_readiness_sha256,
        "family_policy_sha256": family_policy_sha256,
        "family_policy_contract_sha256": family_policy_contract_sha256,
        "tokenizer_file_identities": dict(tokenizer_identities),
    }
    for field, expected in expected_scalars.items():
        if prompt_pack.get(field) != expected:
            raise ValueError(
                f"prompt pack provenance {field} does not match supplied evidence"
            )
    expected_paths = {
        "tokenizer_dir": tokenizer_dir,
        "tokenizer_readiness_json": tokenizer_readiness_path,
        "family_policy_json": family_policy_path,
    }
    for field, expected in expected_paths.items():
        claimed = prompt_pack.get(field)
        if (
            not isinstance(claimed, str)
            or Path(claimed).expanduser().resolve() != expected
        ):
            raise ValueError(
                f"prompt pack provenance {field} does not match supplied evidence"
            )


def prepare_glm52_family_eval_teacher_metadata(
    *,
    source_dir: str | Path,
    profile_path: str | Path,
    config_path: str | Path,
    index_path: str | Path,
    source_audit_json: str | Path,
    source_payload_audit_json: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
    eval_prompt_pack_json: str | Path,
    model_id: str,
    revision: str,
    output_dir: str | Path,
) -> dict[str, object]:
    _validate_identity(model_id, revision)
    source_root = Path(source_dir).expanduser().resolve()
    profile_input = Path(profile_path).expanduser().resolve()
    config_input = Path(config_path).expanduser().resolve()
    index_input = Path(index_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().absolute()

    config_sha256 = sha256_file(config_input)
    index_sha256 = sha256_file(index_input)
    if config_sha256 != GLM52_REAP_CONFIG_SHA256:
        raise ValueError("config SHA-256 does not match the pinned GLM52 source")
    if index_sha256 != GLM52_REAP_INDEX_SHA256:
        raise ValueError("index SHA-256 does not match the pinned GLM52 source")

    policy_input = Path(family_policy_json).expanduser().resolve()
    prompt_input = Path(eval_prompt_pack_json).expanduser().resolve()
    tokenizer_input = Path(tokenizer_readiness_json).expanduser().resolve()
    source_audit_input = Path(source_audit_json).expanduser().resolve()
    source_payload_input = Path(source_payload_audit_json).expanduser().resolve()
    policy = _load_json_object(policy_input, label="family policy")
    validate_glm52_family_gate_policy(policy)
    prompt_pack = _load_json_object(prompt_input, label="eval prompt pack")
    prompt_rows = validate_glm52_family_prompt_pack(prompt_pack)
    tokenizer_readiness = _load_json_object(
        tokenizer_input,
        label="tokenizer readiness",
    )
    tokenizer_identities = validate_glm52_tokenizer_readiness(
        tokenizer_readiness,
        tokenizer_dir=source_root,
    )
    policy_sha256 = sha256_file(policy_input)
    tokenizer_readiness_sha256 = sha256_file(tokenizer_input)
    _validate_prompt_pack_provenance(
        prompt_pack=prompt_pack,
        tokenizer_dir=source_root,
        tokenizer_readiness_path=tokenizer_input,
        tokenizer_readiness_sha256=tokenizer_readiness_sha256,
        tokenizer_identities=tokenizer_identities,
        family_policy_path=policy_input,
        family_policy_sha256=policy_sha256,
        family_policy_contract_sha256=str(policy["policy_contract_sha256"]),
    )

    config = _load_json_object(config_input, label="config")
    fresh_source_audit = audit_glm52_reap_source_index(
        config=config,
        index=load_safetensors_index(index_input),
        model_id=model_id,
        revision=revision,
        profile=load_profile(profile_input),
        config_sha256=config_sha256,
        index_sha256=index_sha256,
    )
    fresh_source_json = fresh_source_audit.to_json_dict()
    recorded_source_json = _load_json_object(
        source_audit_input,
        label="source audit",
    )
    if recorded_source_json != fresh_source_json:
        raise ValueError("source audit evidence does not match live pinned metadata")
    fresh_payload_json = audit_glm52_reap_source_payloads(
        source_dir=source_root,
        source_audit=fresh_source_audit,
    )
    recorded_payload_json = _load_json_object(
        source_payload_input,
        label="source payload audit",
    )
    _validate_source_payload_audit_evidence(
        recorded=recorded_payload_json,
        fresh=fresh_payload_json,
    )
    if fresh_payload_json.get("full_payload_ready") is not True:
        raise ValueError("full pinned source payload is not header-ready")

    policy_contract = str(policy["policy_contract_sha256"])
    prompt_contract = str(prompt_pack["prompt_pack_contract_sha256"])
    common = {
        "teacher_model_id": model_id,
        "teacher_revision": revision,
        "teacher_source_kind": TEACHER_SOURCE_KIND,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "family_policy_contract_sha256": policy_contract,
        "prompt_pack_contract_sha256": prompt_contract,
        "prompt_text_sha256": prompt_pack["prompt_text_sha256"],
        "teacher_cache_payload_present": False,
        "teacher_cache_ready": False,
        "full_logits_available": False,
        "intended_logit_scope": "full_vocabulary",
        "intended_kld_mode": "exact_full_logits",
        "expected_vocab_size": GLM52_MODEL_VOCAB_SIZE,
        "activation_quantization_emulated": False,
        "exact_w4a4_runtime_parity_claimed": False,
        "bf16_teacher_claimed": False,
        "holdout_tuning_forbidden": True,
    }
    metadata_rows = tuple(
        {
            "schema_version": 1,
            "record_type": "glm52_teacher_source_metadata",
            "prompt_id": row["prompt_id"],
            "split": row["split"],
            "domain": row["domain"],
            "token_count": row["token_count"],
            "token_ids_sha256": row["token_ids_sha256"],
            "tuning_eligible": row["tuning_eligible"],
            **common,
        }
        for row in prompt_rows
    )
    metadata_path = output_root / METADATA_FILENAME
    _write_jsonl_atomic(metadata_path, metadata_rows)
    split_counts = Counter(str(row["split"]) for row in metadata_rows)
    domain_counts = Counter(str(row["domain"]) for row in metadata_rows)
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": METADATA_RECORD_TYPE,
        "metadata_status": METADATA_READY_STATUS,
        "metadata_ready": True,
        "profile": "glm52-reap-504b-v2",
        "model_id": model_id,
        "revision": revision,
        "source_dir": str(source_root),
        "profile_path": str(profile_input),
        "config_path": str(config_input),
        "index_path": str(index_input),
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "teacher_source_kind": TEACHER_SOURCE_KIND,
        "source_audit_json": str(source_audit_input),
        "source_audit_sha256": sha256_file(source_audit_input),
        "source_payload_audit_json": str(source_payload_input),
        "source_payload_audit_sha256": sha256_file(source_payload_input),
        "tokenizer_readiness_json": str(tokenizer_input),
        "tokenizer_readiness_sha256": tokenizer_readiness_sha256,
        "family_policy_json": str(policy_input),
        "family_policy_sha256": policy_sha256,
        "family_policy_contract_sha256": policy_contract,
        "eval_prompt_pack_json": str(prompt_input),
        "eval_prompt_pack_sha256": sha256_file(prompt_input),
        "prompt_pack_contract_sha256": prompt_contract,
        "prompt_text_sha256": prompt_pack["prompt_text_sha256"],
        "tokenizer_file_identities": tokenizer_identities,
        "source_index_revalidated": True,
        "source_payload_headers_revalidated": True,
        "source_payload_group_count": fresh_payload_json["selected_groups"],
        "source_payload_bundle_count": fresh_payload_json["selected_bundles"],
        "source_payload_shard_count": len(fresh_payload_json["required_shards"]),
        "header_only_source_validation": True,
        "tensor_payloads_read": False,
        "full_model_constructed": False,
        "candidate_artifact_used": False,
        "metadata_jsonl": str(metadata_path),
        "metadata_jsonl_sha256": sha256_file(metadata_path),
        "metadata_row_count": len(metadata_rows),
        "metadata_counts_by_split": dict(split_counts),
        "metadata_counts_by_domain": dict(domain_counts),
        "teacher_cache_payload_present": False,
        "teacher_cache_ready": False,
        "teacher_logits_generated": False,
        "intended_cache_format": "full_logits_safetensors_v1",
        "intended_logit_scope": "full_vocabulary",
        "intended_kld_mode": "exact_full_logits",
        "expected_vocab_size": GLM52_MODEL_VOCAB_SIZE,
        "activation_quantization_emulated": False,
        "exact_w4a4_runtime_parity_claimed": False,
        "bf16_teacher_claimed": False,
        "missing_requirements": ["dequantized_source_teacher_cache_payload"],
    }
    payload["metadata_contract_sha256"] = canonical_sha256(payload)
    return payload


def _protected_roots(source_dir: Path) -> set[Path]:
    lexical = source_dir.expanduser().absolute()
    resolved = lexical.resolve(strict=False)
    roots = {lexical, resolved}
    for root in (lexical, resolved):
        if root.parent.name == "snapshots":
            repository = root.parent.parent
            roots.add(repository)
            roots.add(repository.resolve(strict=False))
    return roots


def _validate_outputs(
    *,
    output_dir: str | Path,
    output_json: str | Path,
    source_dir: str | Path,
    protected_files: Sequence[str | Path],
) -> None:
    output_root = Path(output_dir).expanduser().absolute()
    summary_output = Path(output_json).expanduser().absolute()
    metadata_output = output_root / METADATA_FILENAME

    def identities(path: Path) -> set[Path]:
        return {path, path.resolve(strict=False)}

    if identities(summary_output) & identities(metadata_output):
        raise ValueError(
            "teacher metadata summary must not alias the metadata JSONL"
        )
    outputs = set().union(
        identities(output_root),
        identities(summary_output),
        identities(metadata_output),
    )
    protected = {Path(path).expanduser().absolute() for path in protected_files}
    protected |= {path.resolve(strict=False) for path in tuple(protected)}
    if outputs & protected:
        raise ValueError("teacher metadata outputs must not alias input files")
    for output in outputs:
        for root in _protected_roots(Path(source_dir)):
            try:
                output.relative_to(root)
            except ValueError:
                continue
            raise ValueError("teacher metadata outputs must not enter the source repository")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate GLM52 dequantized-source teacher metadata without "
            "claiming or generating a teacher cache payload."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--source-audit-json", required=True)
    parser.add_argument("--source-payload-audit-json", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--eval-prompt-pack-json", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protected_files = (
        args.profile_path,
        args.config_path,
        args.index_path,
        args.source_audit_json,
        args.source_payload_audit_json,
        args.tokenizer_readiness_json,
        args.family_policy_json,
        args.eval_prompt_pack_json,
    )
    try:
        _validate_outputs(
            output_dir=args.output_dir,
            output_json=args.output_json,
            source_dir=args.source_dir,
            protected_files=protected_files,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        payload = prepare_glm52_family_eval_teacher_metadata(
            source_dir=args.source_dir,
            profile_path=args.profile_path,
            config_path=args.config_path,
            index_path=args.index_path,
            source_audit_json=args.source_audit_json,
            source_payload_audit_json=args.source_payload_audit_json,
            tokenizer_readiness_json=args.tokenizer_readiness_json,
            family_policy_json=args.family_policy_json,
            eval_prompt_pack_json=args.eval_prompt_pack_json,
            model_id=args.model_id,
            revision=args.revision,
            output_dir=args.output_dir,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": METADATA_RECORD_TYPE,
            "metadata_status": "glm52_teacher_source_metadata_failed",
            "metadata_ready": False,
            "teacher_cache_payload_present": False,
            "teacher_cache_ready": False,
            "teacher_logits_generated": False,
            "tensor_payloads_read": False,
            "full_model_constructed": False,
            "missing_requirements": ["authenticated_teacher_source_metadata"],
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    _write_json_atomic(args.output_json, payload)
    return 0 if payload.get("metadata_ready") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
