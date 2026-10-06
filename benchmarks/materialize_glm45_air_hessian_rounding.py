from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from mlx_vq.benchmark.glm45_air import load_resident_air
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from mlx_vq.quality.hessian_rounding import materialize_hessian_rounding_candidate
from mlx_vq.quality.prompts import QualityPrompt, get_quality_prompts
from mlx_vq.validate.glm45_air_vq import capture_glm45_air_moe_input_states, select_input_state_rows


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
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a bounded GLM-4.5-Air diagonal-Hessian rounding candidate. "
            "This rewrites selected 8-bit gate/up shards in a new artifact directory."
        )
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--source-dir")
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--capture-artifact-dir",
        help="Resident artifact used only for prompt hidden-state capture; defaults to --seed-artifact-dir.",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--tokens-per-prompt", type=int)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--quality-prompt-id", action="append", default=[])
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-state-position", choices=("head", "tail"), default="tail")
    parser.add_argument("--projection", action="append", choices=("gate_proj", "up_proj"), default=[])
    parser.add_argument("--max-experts", type=int, default=8)
    parser.add_argument(
        "--max-output-rows",
        type=int,
        help="Optional output-row cap per rewritten projection; omitted means all output rows.",
    )
    parser.add_argument("--strict-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    if args.prompt_text is not None and args.quality_prompt_id:
        raise SystemExit("pass --prompt-text or one or more --quality-prompt-id values, not both")

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
    input_state_indices: tuple[int, ...] | None = None
    prompt_captures: list[dict[str, object]] = []

    requested_prompts: list[QualityPrompt] = []
    if args.quality_prompt_id:
        requested_prompts = [_quality_prompt(prompt_id) for prompt_id in args.quality_prompt_id]
    elif args.prompt_text is not None:
        requested_prompts = [_custom_prompt(args.prompt_text)]

    if requested_prompts:
        model, tokenizer, _, _ = load_resident_air(
            model_id=args.model_id,
            revision=args.revision,
            source_dir=str(source_dir),
            config_path=args.config_path,
            index_path=str(index_path),
            artifact_dir=args.capture_artifact_dir or args.seed_artifact_dir,
        )
        tokens_per_prompt = args.tokens_per_prompt or args.tokens
        selected_states: list[np.ndarray] = []
        selected_indices: list[int] = []
        for prompt in requested_prompts:
            token_ids = _prompt_tokens(tokenizer, prompt)
            captured = capture_glm45_air_moe_input_states(model, token_ids, layer=args.layer)
            prompt_states, prompt_indices = select_input_state_rows(
                captured,
                tokens=tokens_per_prompt,
                position=args.prompt_state_position,
            )
            offset = len(selected_indices)
            selected_states.append(prompt_states)
            selected_indices.extend(range(offset, offset + prompt_states.shape[0]))
            prompt_captures.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "token_count": len(token_ids),
                    "selected_state_indices": list(prompt_indices),
                    "selected_state_count": int(prompt_states.shape[0]),
                }
            )

        input_states = np.concatenate(selected_states, axis=0).astype(np.float32, copy=False)
        input_state_indices = tuple(selected_indices)
        input_source = (
            f"quality_prompts:{args.prompt_state_position}:{','.join(args.quality_prompt_id)}"
            if args.quality_prompt_id
            else f"prompt_text:{args.prompt_state_position}"
        )

    manifest = materialize_hessian_rounding_candidate(
        config,
        source_dir=source_dir,
        index_path=index_path,
        seed_artifact_dir=args.seed_artifact_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        layer=args.layer,
        tokens=args.tokens,
        seed=args.seed,
        input_scale=args.input_scale,
        input_states=input_states,
        input_source=input_source,
        input_state_indices=input_state_indices,
        projections=tuple(args.projection or ["gate_proj", "up_proj"]),
        max_experts=args.max_experts,
        max_output_rows=args.max_output_rows,
        strict_config=args.strict_config,
        allow_existing=args.allow_existing,
    )
    if prompt_captures:
        manifest["prompt_captures"] = prompt_captures
        manifest_path = Path(args.output_dir) / "hessian-rounding-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
