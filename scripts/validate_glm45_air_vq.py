from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_vq.benchmark.glm45_air import load_resident_air
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from mlx_vq.quality.prompts import QualityPrompt, get_quality_prompts
from mlx_vq.validate.glm45_air_vq import (
    capture_glm45_air_moe_input_states,
    select_input_state_rows,
    validate_glm45_air_vq,
)


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict:
    if config_path is not None:
        return json.loads(Path(config_path).read_text())
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text())
    except Exception:
        return fetch_hf_config(model_id, revision=revision)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["model.safetensors.index.json"],
            local_files_only=True,
        )
    )


def _quality_prompt(prompt_id: str) -> QualityPrompt:
    for prompt in get_quality_prompts():
        if prompt.prompt_id == prompt_id:
            return prompt
    choices = ", ".join(prompt.prompt_id for prompt in get_quality_prompts())
    raise ValueError(f"unknown quality prompt {prompt_id!r}; choices: {choices}")


def _prompt_tokens(tokenizer, prompt: QualityPrompt) -> list[int]:
    tokens = list(tokenizer.encode(prompt.text, add_special_tokens=False))
    if prompt.context_tokens is None:
        return tokens
    if not tokens:
        raise ValueError(f"quality prompt {prompt.prompt_id!r} tokenized to zero tokens")
    expanded = list(tokens)
    while len(expanded) < prompt.context_tokens:
        expanded.extend(tokens)
    return expanded[: prompt.context_tokens]


def _custom_prompt(prompt_text: str) -> QualityPrompt:
    return QualityPrompt(
        prompt_id="custom",
        text=prompt_text,
        max_new_tokens=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the GLM-4.5-Air routed VQ artifact against source tensors.")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--source-dir")
    parser.add_argument("--artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument(
        "--capture-artifact-dir",
        help="Resident artifact used only for prompt hidden-state capture; defaults to --artifact-dir.",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--quality-prompt-id")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-state-position", choices=("head", "tail"), default="tail")
    parser.add_argument("--strict-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-artifact-cosine", type=float, default=0.999)
    parser.add_argument("--min-source-weighted-cosine", type=float)
    args = parser.parse_args()

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path is not None else source_dir / "model.safetensors.index.json"
    config = _load_config(
        model_id=args.model_id,
        revision=args.revision,
        config_path=args.config_path,
    )
    input_states = None
    input_source = None
    prompt_text = None
    prompt_token_ids: tuple[int, ...] = ()
    input_state_indices: tuple[int, ...] | None = None
    if args.quality_prompt_id is not None or args.prompt_text is not None:
        if args.quality_prompt_id is not None and args.prompt_text is not None:
            raise SystemExit("pass only one of --quality-prompt-id or --prompt-text")
        prompt = (
            _quality_prompt(args.quality_prompt_id)
            if args.quality_prompt_id is not None
            else _custom_prompt(args.prompt_text)
        )
        model, tokenizer, _, _ = load_resident_air(
            model_id=args.model_id,
            revision=args.revision,
            source_dir=str(source_dir),
            config_path=args.config_path,
            index_path=str(index_path),
            artifact_dir=args.capture_artifact_dir or args.artifact_dir,
        )
        token_ids = _prompt_tokens(tokenizer, prompt)
        captured = capture_glm45_air_moe_input_states(model, token_ids, layer=args.layer)
        input_states, input_state_indices = select_input_state_rows(
            captured,
            tokens=args.tokens,
            position=args.prompt_state_position,
        )
        input_source = (
            f"quality_prompt:{prompt.prompt_id}:{args.prompt_state_position}"
            if args.quality_prompt_id is not None
            else f"prompt_text:{args.prompt_state_position}"
        )
        prompt_text = prompt.text
        prompt_token_ids = tuple(int(token_id) for token_id in token_ids)

    result = validate_glm45_air_vq(
        config,
        source_dir=source_dir,
        index_path=index_path,
        artifact_dir=args.artifact_dir,
        model_id=args.model_id,
        layer=args.layer,
        tokens=args.tokens,
        seed=args.seed,
        input_scale=args.input_scale,
        input_states=input_states,
        input_source=input_source,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        input_state_indices=input_state_indices,
        strict_config=args.strict_config,
    )
    payload = result.to_dict()
    payload["thresholds"] = {
        "min_artifact_cosine": args.min_artifact_cosine,
        "min_source_weighted_cosine": args.min_source_weighted_cosine,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    failures = []
    for metric_name in (
        "artifact_gate_proj",
        "artifact_up_proj",
        "artifact_routed_glu",
        "artifact_weighted_routed",
    ):
        cosine = result.metrics[metric_name].cosine
        if cosine < args.min_artifact_cosine:
            failures.append(f"{metric_name} cosine {cosine:.6f} < {args.min_artifact_cosine:.6f}")
    if args.min_source_weighted_cosine is not None:
        cosine = result.metrics["source_weighted_routed"].cosine
        if cosine < args.min_source_weighted_cosine:
            failures.append(
                "source_weighted_routed cosine "
                f"{cosine:.6f} < {args.min_source_weighted_cosine:.6f}"
            )
    if failures:
        raise SystemExit("validation threshold failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
