from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mlx_lm.utils import load_tokenizer

from mlx_vq.quality.glm52_family import (
    GLM52_EOS_TOKEN_IDS,
    GLM52_EVAL_PROMPT_TEXT_SHA256,
    GLM52_MODEL_VOCAB_SIZE,
    GLM52_PINNED_TOKENIZER_FILES,
    GLM52_REQUIRED_EVAL_DOMAINS as REQUIRED_DOMAINS,
    GLM52_REQUIRED_EVAL_SPLITS as REQUIRED_SPLITS,
    GLM52_TOKENIZER_BASE_VOCAB_SIZE,
    GLM52_TOKENIZER_LENGTH,
    PINNED_GLM52_MODEL_ID as PINNED_MODEL_ID,
    PINNED_GLM52_REVISION as PINNED_REVISION,
    canonical_sha256,
    define_glm52_family_gate_policy,
    validate_glm52_tokenizer_readiness,
)


PROMPT_PACK_RECORD_TYPE = "glm52_family_eval_prompt_pack"
PROMPT_PACK_READY_STATUS = "glm52_family_eval_prompt_pack_frozen"
PROMPT_PACK_FAILED_STATUS = "glm52_family_eval_prompt_pack_failed"
TOKENIZER_RECORD_TYPE = "glm52_tokenizer_readiness_probe"
TOKENIZER_READY_STATUS = "glm52_tokenizer_readiness_probe_ready"
MLX_LM_TOKENIZER_CONFIG_EXTRA = {
    "local_files_only": True,
    "trust_remote_code": False,
}
Domain = Literal["route", "math", "instruction"]


@dataclass(frozen=True)
class GLM52EvalPrompt:
    domain: Domain
    prompt: str


def _prompts(domain: Domain, *values: str) -> tuple[GLM52EvalPrompt, ...]:
    return tuple(GLM52EvalPrompt(domain=domain, prompt=value) for value in values)


DEFAULT_GLM52_EVAL_PROMPTS: dict[str, tuple[GLM52EvalPrompt, ...]] = {
    "report": (
        *_prompts(
            "route",
            "Explain how a mixture of experts router selects its top k experts.",
            "Describe how router correction bias can change expert selection.",
            "Compare a full IndexShare layer with a shared IndexShare layer.",
            "Explain why cached top k indices must stay aligned during decoding.",
            "Describe one failure caused by a missing routed expert artifact.",
            "Explain how quantizing routed weights can change model logits.",
            "Compare shared experts with token selected routed experts.",
            "Explain why the layer seventy eight MTP weights are excluded.",
        ),
        *_prompts(
            "math",
            "Calculate forty seven plus sixty eight and show the steps.",
            "A box holds twelve rows of nine batteries. How many batteries are there?",
            "Solve three x plus seven equals twenty eight.",
            "Find the next two numbers in the sequence two six twelve twenty.",
            "A train travels one hundred eighty miles in three hours. Find its average speed.",
            "Compute fifteen percent of three hundred forty.",
            "Explain why the square root of eighty one is nine.",
        ),
        *_prompts(
            "instruction",
            "Write a short Python function that returns the larger of two numbers.",
            "Summarize in two sentences why reproducible benchmarks need fixed inputs.",
            "Translate good morning and welcome into Spanish.",
            "Give three concise safety rules for handling lithium batteries.",
            "Rewrite this request politely: send the report today.",
            "Write a SQL query that selects active users ordered by creation date.",
            "Explain caching to a beginner using one concrete example.",
        ),
    ),
    "selection": (
        *_prompts(
            "route",
            "Explain why routing scores should be normalized after top k selection.",
            "Describe what happens when a shared index layer receives no prior top k indices.",
            "Explain why expert counts must come from the model profile rather than a stock default.",
            "Compare dense expert fallback with fail loud routed artifact validation.",
            "Describe how a cross shard expert bundle should be audited.",
            "Explain why route traces help diagnose quality regressions.",
            "Describe the difference between routed projection codes and their scale tensors.",
            "Explain why an unexpected layer seventy eight routed group is unsafe.",
        ),
        *_prompts(
            "math",
            "Calculate ninety six divided by eight and explain the result.",
            "A rectangle is thirteen meters long and seven meters wide. Find its area.",
            "Solve five y minus four equals thirty one.",
            "Find the missing number in one four nine sixteen blank thirty six.",
            "A recipe uses three cups for twelve servings. How many cups serve twenty people?",
            "Compute seven eighths minus three eighths.",
            "Explain whether one hundred one is divisible by three.",
        ),
        *_prompts(
            "instruction",
            "Write pseudocode for binary search over a sorted list.",
            "Summarize the difference between validation data and holdout data.",
            "Translate thank you for your help into French.",
            "Draft a concise status update that names one blocker and one next action.",
            "Rewrite this sentence in active voice: the test was run by the engineer.",
            "Write a shell command that lists JSON files recursively.",
            "Explain the purpose of a tokenizer without using technical jargon.",
        ),
    ),
    "holdout": (
        *_prompts(
            "route",
            "Explain why a router can choose different experts for neighboring tokens.",
            "Describe how stale cache offsets could corrupt sparse attention selection.",
            "Explain why routed artifact names and tensor shapes must be exact.",
            "Compare source relative quality evaluation with task accuracy evaluation.",
            "Describe one reason expert weights should be decoded one working set at a time.",
            "Explain why a model can bind correctly yet still fail generation.",
            "Describe how per domain route metrics expose a hidden regression.",
            "Explain why full vocabulary logits are required for an exact KLD measurement.",
        ),
        *_prompts(
            "math",
            "Calculate two hundred fifteen minus eighty seven and show the steps.",
            "A circle has radius five. Express its area using pi.",
            "Solve nine z plus six equals sixty nine.",
            "Find the next value in the sequence three six eleven eighteen twenty seven.",
            "A store discounts an eighty dollar item by twenty five percent. Find the sale price.",
            "Compute the average of twelve eighteen twenty four and thirty.",
            "Explain why zero point one plus zero point two can be inexact in binary floating point.",
        ),
        *_prompts(
            "instruction",
            "Write a JavaScript function that removes duplicate strings from an array.",
            "Summarize why a holdout set must not guide recovery tuning.",
            "Translate the system is ready into German.",
            "Draft a short troubleshooting step for a missing model shard.",
            "Rewrite this paragraph as a three item checklist.",
            "Write a regular expression that matches a string of decimal digits.",
            "Explain why publication claims need immutable provenance.",
        ),
    ),
}


def _canonical_sha256(value: object) -> str:
    return canonical_sha256(value)


def _validate_identity(model_id: str, revision: str) -> None:
    if model_id != PINNED_MODEL_ID:
        raise ValueError(f"model_id must be {PINNED_MODEL_ID!r}")
    if revision != PINNED_REVISION:
        raise ValueError(f"revision must be {PINNED_REVISION!r}")


def build_glm52_family_eval_prompt_pack(
    tokenizer: Any,
    *,
    model_id: str,
    revision: str,
    expected_vocab_size: int,
    prompts_by_split: Mapping[str, Sequence[GLM52EvalPrompt]] | None = None,
) -> dict[str, Any]:
    _validate_identity(model_id, revision)
    if expected_vocab_size != GLM52_MODEL_VOCAB_SIZE:
        raise ValueError("expected_vocab_size must be the pinned value 154880")
    prompt_authority = prompts_by_split or DEFAULT_GLM52_EVAL_PROMPTS
    if tuple(prompt_authority) != REQUIRED_SPLITS:
        raise ValueError(
            f"prompt splits must be exactly ordered as {REQUIRED_SPLITS!r}"
        )

    rows: list[dict[str, Any]] = []
    prompt_texts: set[str] = set()
    domain_counts_by_split: dict[str, dict[str, int]] = {}
    for split in REQUIRED_SPLITS:
        specs = tuple(prompt_authority[split])
        if len(specs) != 22:
            raise ValueError(f"{split} must contain exactly 22 prompts")
        counts = Counter(spec.domain for spec in specs)
        expected_counts = {"route": 8, "math": 7, "instruction": 7}
        if dict(counts) != expected_counts:
            raise ValueError(
                f"{split} domain quota must be {expected_counts}, found {dict(counts)}"
            )
        domain_counts_by_split[split] = expected_counts
        per_domain_index: Counter[str] = Counter()
        for spec in specs:
            prompt = spec.prompt.strip()
            if not prompt or prompt in prompt_texts:
                raise ValueError("prompt text must be non-empty and globally unique")
            prompt_texts.add(prompt)
            prompt_id = (
                f"glm52_{split}_{spec.domain}_{per_domain_index[spec.domain]:03d}"
            )
            per_domain_index[spec.domain] += 1
            raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, str):
                raise ValueError(f"{prompt_id} tokenizer output must be a sequence")
            token_ids = [int(token_id) for token_id in raw_ids]
            if len(token_ids) < 2:
                raise ValueError(f"{prompt_id} must encode to at least two tokens")
            if any(token_id < 0 or token_id >= expected_vocab_size for token_id in token_ids):
                raise ValueError(
                    f"{prompt_id} contains token outside expected model vocabulary"
                )
            rows.append(
                {
                    "record_type": "glm52_family_eval_prompt",
                    "model_id": model_id,
                    "revision": revision,
                    "split": split,
                    "domain": spec.domain,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "token_count": len(token_ids),
                    "encoded_token_ids": token_ids,
                    "token_ids_sha256": _canonical_sha256(token_ids),
                    "max_token_id": max(token_ids),
                    "teacher_logit_scope": "full_vocabulary",
                    "candidate_logit_scope": "full_vocabulary",
                    "kld_scope": "full_vocabulary",
                    "memory_clean_required": True,
                    "tuning_eligible": split == "selection",
                }
            )

    canonical_prompts = [
        {
            "prompt_id": row["prompt_id"],
            "split": row["split"],
            "domain": row["domain"],
            "prompt": row["prompt"],
        }
        for row in rows
    ]
    tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size"))
    tokenizer_length = int(len(tokenizer))
    if (
        tokenizer_vocab_size != GLM52_TOKENIZER_BASE_VOCAB_SIZE
        or tokenizer_length != GLM52_TOKENIZER_LENGTH
    ):
        raise ValueError(
            "tokenizer identity must have base vocab 154820 and length 154856"
        )
    prompt_text_sha256 = _canonical_sha256(canonical_prompts)
    if prompt_text_sha256 != GLM52_EVAL_PROMPT_TEXT_SHA256:
        raise ValueError(
            "prompt authority does not match the pinned GLM52 corpus: "
            f"{prompt_text_sha256}"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "record_type": PROMPT_PACK_RECORD_TYPE,
        "prompt_pack_status": PROMPT_PACK_READY_STATUS,
        "prompt_pack_ready": True,
        "model_id": model_id,
        "revision": revision,
        "profile": "glm52-reap-504b-v2",
        "prompt_pack_frozen_before_candidate_metrics": True,
        "holdout_tuning_forbidden": True,
        "selection_is_only_tuning_eligible_split": True,
        "required_splits": list(REQUIRED_SPLITS),
        "required_domains": list(REQUIRED_DOMAINS),
        "split_counts": {split: 22 for split in REQUIRED_SPLITS},
        "domain_counts_by_split": domain_counts_by_split,
        "prompt_row_count": len(rows),
        "prompt_text_sha256": prompt_text_sha256,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_length": tokenizer_length,
        "expected_vocab_size": expected_vocab_size,
        "prompt_rows": rows,
        "missing_requirements": [],
    }
    payload["prompt_content_contract_sha256"] = _canonical_sha256(payload)
    return payload


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_family_policy(payload: Mapping[str, Any]) -> None:
    expected = define_glm52_family_gate_policy(
        model_id=PINNED_MODEL_ID,
        revision=PINNED_REVISION,
    )
    submitted = dict(payload)
    embedded_digest = submitted.pop("policy_contract_sha256", None)
    recomputed_digest = _canonical_sha256(submitted)
    if embedded_digest != recomputed_digest:
        raise ValueError(
            "family policy embedded digest does not authenticate its body"
        )
    if dict(payload) != expected:
        raise ValueError("family policy body does not match the frozen contract")


def probe_glm52_family_eval_prompts(
    *,
    tokenizer_dir: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    root = Path(tokenizer_dir).expanduser().resolve()
    readiness_path = Path(tokenizer_readiness_json).expanduser().resolve()
    readiness = _load_json_object(readiness_path, label="tokenizer readiness")
    tokenizer_file_identities = validate_glm52_tokenizer_readiness(
        readiness,
        tokenizer_dir=root,
    )
    policy_path = Path(family_policy_json).expanduser().resolve()
    policy = _load_json_object(policy_path, label="family policy")
    _validate_family_policy(policy)
    wrapper = load_tokenizer(
        root,
        dict(MLX_LM_TOKENIZER_CONFIG_EXTRA),
        eos_token_ids=list(GLM52_EOS_TOKEN_IDS),
    )
    payload = build_glm52_family_eval_prompt_pack(
        wrapper._tokenizer,
        model_id=model_id,
        revision=revision,
        expected_vocab_size=GLM52_MODEL_VOCAB_SIZE,
    )
    payload.update(
        {
            "tokenizer_dir": str(root),
            "tokenizer_readiness_json": str(readiness_path),
            "tokenizer_readiness_sha256": _sha256_file(readiness_path),
            "family_policy_json": str(policy_path),
            "family_policy_sha256": _sha256_file(policy_path),
            "family_policy_contract_sha256": policy[
                "policy_contract_sha256"
            ],
            "tokenizer_file_identities": tokenizer_file_identities,
            "local_files_only": True,
            "trust_remote_code": False,
        }
    )
    payload["prompt_pack_contract_sha256"] = _canonical_sha256(payload)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and tokenize the pinned GLM52 report/selection/holdout pack."
        )
    )
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _validate_output_does_not_alias_inputs(
    *,
    output_json: str | Path,
    tokenizer_dir: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
) -> None:
    output_lexical = Path(output_json).expanduser().absolute()
    root_lexical = Path(tokenizer_dir).expanduser().absolute()
    output_resolved = output_lexical.resolve(strict=False)
    root_resolved = root_lexical.resolve(strict=False)
    protected_lexical = {
        Path(tokenizer_readiness_json).expanduser().absolute(),
        Path(family_policy_json).expanduser().absolute(),
    }
    protected_resolved = {
        path.resolve(strict=False) for path in protected_lexical
    }
    if (
        output_lexical in protected_lexical
        or output_resolved in protected_resolved
    ):
        raise ValueError("--output-json must not alias an input evidence file")
    protected_roots = {root_lexical, root_resolved}
    for candidate_root in (root_lexical, root_resolved):
        if candidate_root.parent.name != "snapshots":
            continue
        repository_root = candidate_root.parent.parent
        protected_roots.add(repository_root)
        protected_roots.add(repository_root.resolve(strict=False))
    for filename in GLM52_PINNED_TOKENIZER_FILES:
        tokenizer_file = root_lexical / filename
        protected_lexical.add(tokenizer_file)
        protected_resolved.add(tokenizer_file.resolve(strict=False))
    if (
        output_lexical in protected_lexical
        or output_resolved in protected_resolved
    ):
        raise ValueError("--output-json must not alias a pinned tokenizer file")
    for output in (output_lexical, output_resolved):
        for root in protected_roots:
            try:
                output.relative_to(root)
            except ValueError:
                continue
            raise ValueError(
                "--output-json must not be inside the pinned tokenizer repository"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_output_does_not_alias_inputs(
            output_json=args.output_json,
            tokenizer_dir=args.tokenizer_dir,
            tokenizer_readiness_json=args.tokenizer_readiness_json,
            family_policy_json=args.family_policy_json,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        payload = probe_glm52_family_eval_prompts(
            tokenizer_dir=args.tokenizer_dir,
            tokenizer_readiness_json=args.tokenizer_readiness_json,
            family_policy_json=args.family_policy_json,
            model_id=args.model_id,
            revision=args.revision,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": PROMPT_PACK_RECORD_TYPE,
            "prompt_pack_status": PROMPT_PACK_FAILED_STATUS,
            "prompt_pack_ready": False,
            "model_id": args.model_id,
            "revision": args.revision,
            "prompt_pack_frozen_before_candidate_metrics": False,
            "holdout_tuning_forbidden": True,
            "missing_requirements": ["valid_pinned_tokenizer_and_prompt_pack"],
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    _write_json_atomic(args.output_json, payload)
    return 0 if payload.get("prompt_pack_ready") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
