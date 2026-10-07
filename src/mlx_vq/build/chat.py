"""Interactive chat runtime for a built KEEP artifact.

The public CLI is stitched by ``configure_chat_parser`` and ``run_chat``.  The
REPL itself deliberately has no MLX dependency: ``ChatSession`` accepts an
injected token generator so command handling remains cheap and deterministic to
test.  MLX imports only happen once a real runtime has passed preflight.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


ChatMessage = dict[str, str]
TokenGenerator = Callable[[tuple[ChatMessage, ...]], Iterator[str]]
InputFn = Callable[[str], str]
OutputFn = Callable[[str], Any]

_GLM52_AUTHENTICATION_PROMPT = "The capital of France is"
_GLM52_REQUIRED_ARGUMENTS = (
    "glm52_profile_path",
    "glm52_config_path",
    "glm52_source_index_path",
    "glm52_tokenizer_dir",
    "glm52_tokenizer_readiness_json",
    "glm52_family_policy_json",
    "glm52_non_vq_artifact_dir",
    "glm52_non_vq_evidence_json",
    "glm52_routed_artifact_dir",
    "glm52_composite_audit_json",
    "glm52_materialization_runs_jsonl",
    "glm52_full_bind_preflight_json",
)


class ChatError(RuntimeError):
    """A user-facing failure while preparing or running a chat session."""


class RuntimeLoader(Protocol):
    def __call__(self, config: "ChatConfig") -> "ChatRuntime": ...


@dataclass(frozen=True)
class ChatConfig:
    """Arguments needed by the CLI stitching layer and runtime loader."""

    artifact: str
    max_tokens: int = 256
    glm52_profile_path: str | None = None
    glm52_config_path: str | None = None
    glm52_source_index_path: str | None = None
    glm52_tokenizer_dir: str | None = None
    glm52_tokenizer_readiness_json: str | None = None
    glm52_family_policy_json: str | None = None
    glm52_non_vq_artifact_dir: str | None = None
    glm52_non_vq_evidence_json: str | None = None
    glm52_routed_artifact_dir: str | None = None
    glm52_composite_audit_json: str | None = None
    glm52_materialization_runs_jsonl: str | None = None
    glm52_full_bind_preflight_json: str | None = None
    glm52_model_id: str | None = None
    glm52_revision: str | None = None

    @property
    def is_glm52_composite(self) -> bool:
        return any(
            getattr(self, name) is not None for name in _GLM52_REQUIRED_ARGUMENTS
        )

    def require_complete_glm52_composite(self) -> None:
        missing = [
            name.replace("glm52_", "--glm52-").replace("_", "-")
            for name in _GLM52_REQUIRED_ARGUMENTS
            if getattr(self, name) is None
        ]
        if missing:
            raise ChatError(
                "GLM52 composite chat requires strict binding inputs: "
                + ", ".join(missing)
            )

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ChatConfig":
        """Discard parser bookkeeping before constructing the typed config."""

        field_names = cls.__dataclass_fields__
        return cls(**{name: getattr(args, name) for name in field_names})


@dataclass(frozen=True)
class ChatRuntime:
    model: Any
    tokenizer: Any
    artifact_label: str
    max_tokens: int


class ChatSession:
    """Pure conversation state and streamed assistant-turn assembly."""

    def __init__(self, generator: TokenGenerator) -> None:
        self._generator = generator
        self._messages: list[ChatMessage] = []

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        return tuple(dict(message) for message in self._messages)

    def reset(self) -> None:
        self._messages.clear()

    def stats(self) -> tuple[int, int]:
        return (
            sum(message["role"] == "user" for message in self._messages),
            sum(message["role"] == "assistant" for message in self._messages),
        )

    def stream(self, prompt: str) -> Iterator[str]:
        if not prompt.strip():
            raise ChatError("message must not be empty")
        self._messages.append({"role": "user", "content": prompt})
        pieces: list[str] = []
        generator = self._generator(self.messages)
        try:
            for piece in generator:
                if not isinstance(piece, str):
                    raise ChatError("generator yielded a non-text token")
                pieces.append(piece)
                yield piece
        except KeyboardInterrupt:
            raise
        except Exception:
            raise
        else:
            response = "".join(pieces)
            if not response:
                raise ChatError("generation produced no text")
            self._messages.append({"role": "assistant", "content": response})
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()


def validate_chat_template(
    tokenizer: Any,
    *,
    prompt: str,
    strict_glm52: bool,
) -> None:
    """Validate the exact disabled-thinking template used for every turn."""

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise ChatError("chat template produced invalid token IDs")
    if strict_glm52:
        if any(token_id >= 154_880 for token_id in token_ids):
            raise ChatError(
                "chat template produced token IDs outside the GLM52 vocabulary"
            )
        if not str(rendered).rstrip().endswith("<|assistant|><think></think>"):
            raise ChatError("chat template did not use disabled-thinking assistant form")


def run_repl(
    session: ChatSession,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    """Run the command loop. Ctrl-C cancels only the current generation."""

    while True:
        try:
            prompt = input_fn("you> ")
        except EOFError:
            output_fn("bye")
            return 0
        except KeyboardInterrupt:
            output_fn("use /exit to leave chat")
            continue

        command = prompt.strip()
        if command == "/exit":
            output_fn("bye")
            return 0
        if command == "/reset":
            session.reset()
            output_fn("conversation reset")
            continue
        if command == "/stats":
            users, assistants = session.stats()
            output_fn(f"stats: {users} user turns, {assistants} assistant turns")
            continue
        if command.startswith("/"):
            output_fn(
                f"error: unknown command {command}; use /exit, /reset, or /stats"
            )
            continue
        try:
            for piece in session.stream(prompt):
                output_fn(f"assistant: {piece}")
        except KeyboardInterrupt:
            output_fn("generation cancelled")
            continue
        except Exception as error:
            output_fn(f"error: {error}")
            continue


def _stream_mlx_tokens(
    runtime: ChatRuntime,
    messages: tuple[ChatMessage, ...],
) -> Iterator[str]:
    """Render the source chat template and stream ``mlx_lm.generate_step`` tokens."""

    # Deliberately lazy: importing MLX can initialize Metal even before loading.
    import mlx.core as mx
    from mlx_lm.generate import generate_step, generation_stream

    token_ids = runtime.tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise ChatError("chat template produced invalid token IDs")
    generator = generate_step(
        mx.array(token_ids, dtype=mx.int32),
        runtime.model,
        max_tokens=runtime.max_tokens,
        prompt_cache=runtime.model.make_cache(),
    )
    try:
        for token, _logprobs in generator:
            token_id = int(token.item()) if hasattr(token, "item") else int(token)
            if token_id in getattr(runtime.tokenizer, "eos_token_ids", ()):
                break
            text = runtime.tokenizer.decode([token_id])
            if text:
                yield str(text)
    finally:
        try:
            generator.close()
        finally:
            mx.synchronize(generation_stream)


def _load_glm52_composite(config: ChatConfig) -> ChatRuntime:
    """Use the authenticated production bind path for a GLM52 composite."""

    config.require_complete_glm52_composite()
    from mlx_lm.utils import load_tokenizer

    from mlx_vq.models.glm52_composite_loader import (
        load_authenticated_glm52_composite,
        validate_glm52_production_inputs,
    )
    from mlx_vq.quality.glm52_family import (
        GLM52_EOS_TOKEN_IDS,
        PINNED_GLM52_MODEL_ID,
        PINNED_GLM52_REVISION,
        validate_glm52_tokenizer_readiness,
    )

    model_id = config.glm52_model_id or PINNED_GLM52_MODEL_ID
    revision = config.glm52_revision or PINNED_GLM52_REVISION
    validated = validate_glm52_production_inputs(
        profile_path=config.glm52_profile_path,
        config_path=config.glm52_config_path,
        source_index_path=config.glm52_source_index_path,
        tokenizer_dir=config.glm52_tokenizer_dir,
        tokenizer_readiness_json=config.glm52_tokenizer_readiness_json,
        family_policy_json=config.glm52_family_policy_json,
        non_vq_artifact_dir=config.glm52_non_vq_artifact_dir,
        non_vq_evidence_json=config.glm52_non_vq_evidence_json,
        routed_artifact_dir=config.glm52_routed_artifact_dir,
        composite_audit_json=config.glm52_composite_audit_json,
        materialization_runs_jsonl=config.glm52_materialization_runs_jsonl,
        full_bind_preflight_json=config.glm52_full_bind_preflight_json,
        model_id=model_id,
        revision=revision,
        prompt=_GLM52_AUTHENTICATION_PROMPT,
    )
    validate_glm52_tokenizer_readiness(
        validated.tokenizer_readiness,
        tokenizer_dir=validated.tokenizer_dir,
    )
    tokenizer = load_tokenizer(
        validated.tokenizer_dir,
        {"local_files_only": True, "trust_remote_code": False},
        eos_token_ids=list(GLM52_EOS_TOKEN_IDS),
    )
    validate_chat_template(
        tokenizer,
        prompt=_GLM52_AUTHENTICATION_PROMPT,
        strict_glm52=True,
    )
    model, _report = load_authenticated_glm52_composite(validated)
    return ChatRuntime(model, tokenizer, config.artifact, config.max_tokens)


def load_chat_runtime(config: ChatConfig) -> ChatRuntime:
    """Load a supported MLX artifact after selecting the GLM52 strict path."""

    if config.is_glm52_composite:
        return _load_glm52_composite(config)
    from mlx_lm.utils import load

    model, tokenizer = load(config.artifact, lazy=True)
    validate_chat_template(
        tokenizer,
        prompt=_GLM52_AUTHENTICATION_PROMPT,
        strict_glm52=False,
    )
    return ChatRuntime(model, tokenizer, config.artifact, config.max_tokens)


def _doctor_preflight(config: ChatConfig) -> Any:
    from mlx_vq.build import doctor

    model = "glm52-reap-504b-v2" if config.is_glm52_composite else config.artifact
    return doctor.collect_report(repo_root=Path.cwd(), model=model)


def run_chat_config(
    config: ChatConfig,
    *,
    runtime_loader: RuntimeLoader = load_chat_runtime,
    doctor_preflight: Callable[[ChatConfig], Any] = _doctor_preflight,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    """Run read-only feasibility checks, then load and enter the interactive REPL."""

    if config.max_tokens <= 0:
        raise ChatError("--max-tokens must be positive")
    report = doctor_preflight(config)
    if getattr(report, "exit_code", 0) != 0:
        raise ChatError(
            "preflight failed; run `keep doctor --model " + config.artifact + "`"
        )
    runtime = runtime_loader(config)
    output_fn(f"chatting with {runtime.artifact_label}; /exit, /reset, /stats")
    return run_repl(
        ChatSession(lambda messages: _stream_mlx_tokens(runtime, messages)),
        input_fn=input_fn,
        output_fn=output_fn,
    )


def configure_chat_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register ``keep chat`` without importing or editing the shared CLI module."""

    parser = subparsers.add_parser(
        "chat",
        help="chat interactively with a built artifact or MLX model",
    )
    parser.add_argument(
        "artifact",
        help="local MLX artifact directory or Hugging Face model id",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--glm52-profile-path")
    parser.add_argument("--glm52-config-path")
    parser.add_argument("--glm52-source-index-path")
    parser.add_argument("--glm52-tokenizer-dir")
    parser.add_argument("--glm52-tokenizer-readiness-json")
    parser.add_argument("--glm52-family-policy-json")
    parser.add_argument("--glm52-non-vq-artifact-dir")
    parser.add_argument("--glm52-non-vq-evidence-json")
    parser.add_argument("--glm52-routed-artifact-dir")
    parser.add_argument("--glm52-composite-audit-json")
    parser.add_argument("--glm52-materialization-runs-jsonl")
    parser.add_argument("--glm52-full-bind-preflight-json")
    parser.add_argument("--glm52-model-id")
    parser.add_argument("--glm52-revision")
    parser.set_defaults(func=run_chat)
    return parser


def run_chat(args: argparse.Namespace) -> int:
    """CLI-ready command handler intended for ``set_defaults(func=...)``."""

    try:
        return run_chat_config(ChatConfig.from_namespace(args))
    except ChatError as error:
        print(f"error: {error}")
        return 1


__all__ = [
    "ChatConfig",
    "ChatError",
    "ChatRuntime",
    "ChatSession",
    "configure_chat_parser",
    "load_chat_runtime",
    "run_chat",
    "run_chat_config",
    "run_repl",
    "validate_chat_template",
]
