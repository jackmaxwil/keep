from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mlx_lm.utils import load_tokenizer

from keep.quality.glm52_family import find_unpinned_glm52_tokenizer_inputs


PROBE_RECORD_TYPE = "glm52_tokenizer_readiness_probe"
PROBE_READY_STATUS = "glm52_tokenizer_readiness_probe_ready"
PROBE_FAILED_STATUS = "glm52_tokenizer_readiness_probe_failed"
SOURCE_AUTHORITY = "pinned_huggingface_snapshot"
ASSISTANT_GENERATION_MARKER = "<|assistant|>"
MLX_LM_TOKENIZER_CONFIG_EXTRA = MappingProxyType(
    {"local_files_only": True, "trust_remote_code": False}
)


@dataclass(frozen=True, slots=True)
class PinnedFileIdentity:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GLM52TokenizerIdentityContract:
    model_id: str
    revision: str
    model_type: str
    model_vocab_size: int
    tokenizer_base_vocab_size: int
    tokenizer_length: int
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    files: tuple[PinnedFileIdentity, ...]
    mlx_lm_detokenizer_class: str = "NaiveStreamingDetokenizer"
    mlx_lm_tool_parser_module: str | None = None
    mlx_lm_has_chat_template: bool = True
    mlx_lm_has_thinking: bool = False


PINNED_GLM52_TOKENIZER_CONTRACT = GLM52TokenizerIdentityContract(
    model_id="0xSero/glm-5.2-reap-504B-v2",
    revision="6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
    model_type="glm_moe_dsa",
    model_vocab_size=154_880,
    tokenizer_base_vocab_size=154_820,
    tokenizer_length=154_856,
    pad_token_id=154_820,
    eos_token_ids=(154_820, 154_827, 154_829),
    files=(
        PinnedFileIdentity(
            filename="config.json",
            size_bytes=10_854,
            sha256="5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b",
        ),
        PinnedFileIdentity(
            filename="tokenizer.json",
            size_bytes=20_217_442,
            sha256="19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        ),
        PinnedFileIdentity(
            filename="tokenizer_config.json",
            size_bytes=761,
            sha256="98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
        ),
        PinnedFileIdentity(
            filename="chat_template.jinja",
            size_bytes=5_076,
            sha256="172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
        ),
        PinnedFileIdentity(
            filename="generation_config.json",
            size_bytes=194,
            sha256="ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55",
        ),
    ),
    mlx_lm_detokenizer_class="BPEStreamingDetokenizer",
    mlx_lm_tool_parser_module="mlx_lm.tool_parsers.glm47",
    mlx_lm_has_chat_template=True,
    mlx_lm_has_thinking=True,
)


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
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return value


def _base_payload(
    *,
    tokenizer_dir: Path,
    config_path: Path,
    model_id: str,
    revision: str,
    prompt: str,
    contract: GLM52TokenizerIdentityContract,
) -> dict[str, Any]:
    production_contract = contract == PINNED_GLM52_TOKENIZER_CONTRACT
    return {
        "schema_version": 1,
        "record_type": PROBE_RECORD_TYPE,
        "probe_status": PROBE_FAILED_STATUS,
        "probe_pass": False,
        "readiness_scope": "tokenizer_payload",
        "tokenizer_payload_ready": False,
        "production_ready": False,
        "source_authority": (
            SOURCE_AUTHORITY if production_contract else "test_fixture_contract"
        ),
        "production_identity_contract": production_contract,
        "model_id": model_id,
        "source_revision": revision,
        "expected_model_id": contract.model_id,
        "expected_source_revision": contract.revision,
        "tokenizer_dir": str(tokenizer_dir),
        "config_path": str(config_path),
        "prompt": prompt,
        "local_files_only": MLX_LM_TOKENIZER_CONFIG_EXTRA["local_files_only"],
        "trust_remote_code": MLX_LM_TOKENIZER_CONFIG_EXTRA["trust_remote_code"],
        "remote_code_required": False,
        "immutable_file_identity_pass": False,
        "file_identities": {},
        "config_lineage_pass": False,
        "tokenizer_load_pass": False,
        "mlx_lm_loader_pass": False,
        "mlx_lm_detokenizer_class": None,
        "mlx_lm_tool_parser_module": None,
        "mlx_lm_has_chat_template": False,
        "mlx_lm_has_thinking": False,
        "tokenizer_base_vocab_size": None,
        "tokenizer_length": None,
        "model_vocab_size": contract.model_vocab_size,
        "tokenizer_fits_model_vocab": False,
        "pad_token_id": None,
        "tokenizer_eos_token_id": None,
        "model_eos_token_ids": list(contract.eos_token_ids),
        "mlx_lm_wrapper_eos_token_ids": [],
        "mlx_lm_wrapper_eos_pass": False,
        "prompt_token_count": 0,
        "prompt_encoded_token_ids": [],
        "prompt_token_ids_within_model_vocab": False,
        "prompt_decoded_text": "",
        "prompt_roundtrip_pass": False,
        "thinking_enabled_pass": False,
        "thinking_disabled_pass": False,
        "assistant_generation_marker": ASSISTANT_GENERATION_MARKER,
        "assistant_generation_marker_pass": False,
        "whole_model_runtime_proven": False,
        "full_model_bind_proven": False,
        "generation_proven": False,
        "missing_requirements": [],
    }


def _audit_file_identities(
    tokenizer_dir: Path,
    contract: GLM52TokenizerIdentityContract,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    records: dict[str, dict[str, object]] = {}
    missing_requirements: list[str] = []
    for expected in contract.files:
        path = tokenizer_dir / expected.filename
        exists = path.is_file()
        size_bytes: int | None = None
        sha256: str | None = None
        read_error: str | None = None
        if exists:
            try:
                size_bytes = path.stat().st_size
                sha256 = _sha256_file(path)
            except OSError as error:
                read_error = str(error)
        identity_pass = bool(
            exists
            and read_error is None
            and size_bytes == expected.size_bytes
            and sha256 == expected.sha256
        )
        record: dict[str, object] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": size_bytes,
            "expected_size_bytes": expected.size_bytes,
            "sha256": sha256,
            "expected_sha256": expected.sha256,
            "identity_pass": identity_pass,
        }
        if read_error is not None:
            record["read_error"] = read_error
        records[expected.filename] = record
        if not identity_pass:
            missing_requirements.append(
                f"immutable_file_identity:{expected.filename}"
            )
    return records, missing_requirements


def _config_requirements(
    *,
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    contract: GLM52TokenizerIdentityContract,
) -> tuple[dict[str, bool], list[str]]:
    expected_eos = list(contract.eos_token_ids)
    checks = {
        "config_model_type": config.get("model_type") == contract.model_type,
        "config_vocab_size": config.get("vocab_size") == contract.model_vocab_size,
        "config_pad_token_id": config.get("pad_token_id") == contract.pad_token_id,
        "config_eos_token_ids": config.get("eos_token_id") == expected_eos,
        "generation_pad_token_id": (
            generation_config.get("pad_token_id") == contract.pad_token_id
        ),
        "generation_eos_token_ids": (
            generation_config.get("eos_token_id") == expected_eos
        ),
    }
    return checks, [name for name, passed in checks.items() if not passed]


def _failure(
    payload: dict[str, Any],
    *,
    stage: str,
    missing_requirements: Sequence[str],
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload["failure_stage"] = stage
    payload["missing_requirements"] = list(missing_requirements)
    if error is not None:
        payload["input_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def probe_glm52_tokenizer_readiness(
    *,
    tokenizer_dir: str | Path,
    config_path: str | Path,
    model_id: str,
    revision: str,
    prompt: str,
    identity_contract: GLM52TokenizerIdentityContract = PINNED_GLM52_TOKENIZER_CONTRACT,
) -> dict[str, Any]:
    tokenizer_root = _absolute(tokenizer_dir)
    model_config_path = _absolute(config_path)
    payload = _base_payload(
        tokenizer_dir=tokenizer_root,
        config_path=model_config_path,
        model_id=model_id,
        revision=revision,
        prompt=prompt,
        contract=identity_contract,
    )

    lineage_requirements: list[str] = []
    if model_id != identity_contract.model_id:
        lineage_requirements.append("pinned_model_id")
    if revision != identity_contract.revision:
        lineage_requirements.append("pinned_revision")
    if lineage_requirements:
        return _failure(
            payload,
            stage="source_lineage",
            missing_requirements=lineage_requirements,
        )

    file_identities, identity_requirements = _audit_file_identities(
        tokenizer_root,
        identity_contract,
    )
    payload["file_identities"] = file_identities
    payload["immutable_file_identity_pass"] = not identity_requirements
    unexpected_inputs = find_unpinned_glm52_tokenizer_inputs(tokenizer_root)
    identity_requirements.extend(
        f"unexpected_tokenizer_input:{relative_path}"
        for relative_path in unexpected_inputs
    )
    payload["immutable_file_identity_pass"] = not identity_requirements
    if identity_requirements:
        return _failure(
            payload,
            stage="immutable_file_identity",
            missing_requirements=identity_requirements,
        )

    expected_config_path = tokenizer_root / "config.json"
    try:
        config_path_matches = (
            model_config_path.resolve(strict=True)
            == expected_config_path.resolve(strict=True)
        )
    except OSError as error:
        return _failure(
            payload,
            stage="config_lineage",
            missing_requirements=["config_path_matches_tokenizer_snapshot"],
            error=error,
        )
    if not config_path_matches:
        return _failure(
            payload,
            stage="config_lineage",
            missing_requirements=["config_path_matches_tokenizer_snapshot"],
        )

    try:
        config = _load_json_object(model_config_path, label="model config")
        generation_config = _load_json_object(
            tokenizer_root / "generation_config.json",
            label="generation config",
        )
    except ValueError as error:
        return _failure(
            payload,
            stage="config_lineage",
            missing_requirements=["readable_config_lineage"],
            error=error,
        )
    config_checks, config_requirements = _config_requirements(
        config=config,
        generation_config=generation_config,
        contract=identity_contract,
    )
    payload["config_checks"] = config_checks
    payload["config_lineage_pass"] = not config_requirements
    if config_requirements:
        return _failure(
            payload,
            stage="config_lineage",
            missing_requirements=config_requirements,
        )

    try:
        wrapper = load_tokenizer(
            tokenizer_root,
            dict(MLX_LM_TOKENIZER_CONFIG_EXTRA),
            eos_token_ids=list(identity_contract.eos_token_ids),
        )
    except Exception as error:
        return _failure(
            payload,
            stage="tokenizer_load",
            missing_requirements=["local_mlx_lm_tokenizer_load_without_remote_code"],
            error=error,
        )
    payload["tokenizer_load_pass"] = True
    payload["mlx_lm_loader_pass"] = True
    tokenizer = wrapper._tokenizer
    payload["tokenizer_class"] = type(tokenizer).__name__

    try:
        base_vocab_size = int(tokenizer.vocab_size)
        tokenizer_length = int(len(tokenizer))
        pad_token_id = int(tokenizer.pad_token_id)
        tokenizer_eos_token_id = int(tokenizer.eos_token_id)
        encoded_ids = [
            int(token_id)
            for token_id in tokenizer.encode(prompt, add_special_tokens=False)
        ]
        decoded_text = tokenizer.decode(
            encoded_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        messages = [{"role": "user", "content": prompt}]
        thinking_enabled = wrapper.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        thinking_disabled = wrapper.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        wrapper_eos_ids = sorted(int(token_id) for token_id in wrapper.eos_token_ids)
        detokenizer_class = type(wrapper.detokenizer).__name__
        tool_parser_module = (
            wrapper.tool_parser.__module__ if wrapper.tool_parser is not None else None
        )
        wrapper_has_chat_template = bool(wrapper.has_chat_template)
        wrapper_has_thinking = bool(wrapper.has_thinking)
    except Exception as error:
        return _failure(
            payload,
            stage="tokenizer_capabilities",
            missing_requirements=["tokenizer_capability_evaluation"],
            error=error,
        )

    thinking_enabled_text = str(thinking_enabled)
    thinking_disabled_text = str(thinking_disabled)
    thinking_enabled_tail = thinking_enabled_text.rstrip()
    thinking_disabled_tail = thinking_disabled_text.rstrip()
    thinking_enabled_pass = bool(
        thinking_enabled_text != thinking_disabled_text
        and thinking_enabled_tail.endswith(
            f"{ASSISTANT_GENERATION_MARKER}<think>"
        )
        and not thinking_enabled_tail.endswith("<think></think>")
    )
    thinking_disabled_pass = thinking_disabled_tail.endswith(
        f"{ASSISTANT_GENERATION_MARKER}<think></think>"
    )
    assistant_marker_pass = bool(
        ASSISTANT_GENERATION_MARKER in thinking_enabled_text
        and ASSISTANT_GENERATION_MARKER in thinking_disabled_text
    )
    tokenizer_fits_model_vocab = tokenizer_length <= identity_contract.model_vocab_size
    prompt_ids_within_model_vocab = all(
        0 <= token_id < identity_contract.model_vocab_size for token_id in encoded_ids
    )
    expected_eos_ids = list(identity_contract.eos_token_ids)
    checks = {
        "tokenizer_base_vocab_size": (
            base_vocab_size == identity_contract.tokenizer_base_vocab_size
        ),
        "tokenizer_length": tokenizer_length == identity_contract.tokenizer_length,
        "tokenizer_fits_model_vocab": tokenizer_fits_model_vocab,
        "tokenizer_pad_token_id": pad_token_id == identity_contract.pad_token_id,
        "tokenizer_eos_token_id": (
            tokenizer_eos_token_id == identity_contract.eos_token_ids[0]
        ),
        "prompt_encodes_tokens": bool(encoded_ids),
        "prompt_token_ids_within_model_vocab": prompt_ids_within_model_vocab,
        "prompt_decodes_exactly": decoded_text == prompt,
        "chat_template_thinking_enabled": thinking_enabled_pass,
        "chat_template_thinking_disabled": thinking_disabled_pass,
        "assistant_generation_marker": assistant_marker_pass,
        "mlx_lm_wrapper_full_eos_set": wrapper_eos_ids == expected_eos_ids,
        "mlx_lm_detokenizer_class": (
            detokenizer_class == identity_contract.mlx_lm_detokenizer_class
        ),
        "mlx_lm_tool_parser_module": (
            tool_parser_module == identity_contract.mlx_lm_tool_parser_module
        ),
        "mlx_lm_has_chat_template": (
            wrapper_has_chat_template == identity_contract.mlx_lm_has_chat_template
        ),
        "mlx_lm_has_thinking": (
            wrapper_has_thinking == identity_contract.mlx_lm_has_thinking
        ),
    }
    missing_requirements = [name for name, passed in checks.items() if not passed]
    payload.update(
        {
            "capability_checks": checks,
            "tokenizer_base_vocab_size": base_vocab_size,
            "tokenizer_length": tokenizer_length,
            "tokenizer_fits_model_vocab": tokenizer_fits_model_vocab,
            "pad_token_id": pad_token_id,
            "tokenizer_eos_token_id": tokenizer_eos_token_id,
            "mlx_lm_wrapper_eos_token_ids": wrapper_eos_ids,
            "mlx_lm_wrapper_eos_pass": wrapper_eos_ids == expected_eos_ids,
            "mlx_lm_detokenizer_class": detokenizer_class,
            "mlx_lm_tool_parser_module": tool_parser_module,
            "mlx_lm_has_chat_template": wrapper_has_chat_template,
            "mlx_lm_has_thinking": wrapper_has_thinking,
            "prompt_token_count": len(encoded_ids),
            "prompt_encoded_token_ids": encoded_ids,
            "prompt_token_ids_within_model_vocab": prompt_ids_within_model_vocab,
            "prompt_decoded_text": decoded_text,
            "prompt_roundtrip_pass": decoded_text == prompt,
            "thinking_enabled_render": thinking_enabled_text,
            "thinking_disabled_render": thinking_disabled_text,
            "thinking_enabled_pass": thinking_enabled_pass,
            "thinking_disabled_pass": thinking_disabled_pass,
            "assistant_generation_marker_pass": assistant_marker_pass,
        }
    )
    if missing_requirements:
        return _failure(
            payload,
            stage="tokenizer_capabilities",
            missing_requirements=missing_requirements,
        )

    payload.update(
        {
            "probe_status": PROBE_READY_STATUS,
            "probe_pass": True,
            "tokenizer_payload_ready": True,
            "production_ready": payload["production_identity_contract"],
            "missing_requirements": [],
        }
    )
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
            "Verify the immutable local tokenizer and chat payload for the pinned "
            "GLM-5.2 REAP source without remote code or model weight loading."
        )
    )
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    identity_contract: GLM52TokenizerIdentityContract = PINNED_GLM52_TOKENIZER_CONTRACT,
) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = probe_glm52_tokenizer_readiness(
            tokenizer_dir=args.tokenizer_dir,
            config_path=args.config_path,
            model_id=args.model_id,
            revision=args.revision,
            prompt=args.prompt,
            identity_contract=identity_contract,
        )
    except Exception as error:  # Defensive: preserve durable evidence for unexpected errors.
        payload = _base_payload(
            tokenizer_dir=_absolute(args.tokenizer_dir),
            config_path=_absolute(args.config_path),
            model_id=args.model_id,
            revision=args.revision,
            prompt=args.prompt,
            contract=identity_contract,
        )
        _failure(
            payload,
            stage="unexpected_error",
            missing_requirements=["tokenizer_probe_completed"],
            error=error,
        )
    _write_json_atomic(args.output_json, payload)
    return 0 if payload.get("probe_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
