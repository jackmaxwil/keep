from __future__ import annotations

import argparse
from collections.abc import Iterator

import pytest

from keep.build.chat import (
    ChatConfig,
    ChatError,
    ChatSession,
    ChatRuntime,
    configure_chat_parser,
    run_chat_config,
    run_repl,
    validate_chat_template,
)


class FakeGenerator:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[dict[str, str], ...]] = []

    def __call__(self, messages: tuple[dict[str, str], ...]) -> Iterator[str]:
        self.calls.append(messages)
        response = self.responses[messages[-1]["content"]]
        if isinstance(response, BaseException):
            raise response
        yield from response  # type: ignore[misc]


def test_repl_handles_commands_streams_turns_and_resets_history() -> None:
    generator = FakeGenerator({"hello": ("Hel", "lo")})
    session = ChatSession(generator)
    inputs = iter(("/stats", "hello", "/stats", "/reset", "/stats", "/exit"))
    output: list[str] = []

    result = run_repl(
        session,
        input_fn=lambda _prompt: next(inputs),
        output_fn=output.append,
    )

    assert result == 0
    assert output == [
        "stats: 0 user turns, 0 assistant turns",
        "assistant: Hel",
        "assistant: lo",
        "stats: 1 user turns, 1 assistant turns",
        "conversation reset",
        "stats: 0 user turns, 0 assistant turns",
        "bye",
    ]
    assert generator.calls == [({"role": "user", "content": "hello"},)]


def test_repl_cancels_generation_without_ending_the_session() -> None:
    def generator(messages: tuple[dict[str, str], ...]) -> Iterator[str]:
        if messages[-1]["content"] == "interrupt":
            yield "first"
            raise KeyboardInterrupt
        yield "second"

    session = ChatSession(generator)
    inputs = iter(("interrupt", "continue", "/exit"))
    output: list[str] = []

    assert (
        run_repl(
            session,
            input_fn=lambda _prompt: next(inputs),
            output_fn=output.append,
        )
        == 0
    )

    assert output == [
        "assistant: first",
        "generation cancelled",
        "assistant: second",
        "bye",
    ]
    assert session.messages == (
        {"role": "user", "content": "interrupt"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "second"},
    )


def test_repl_reports_generation_errors_and_keeps_the_session_usable() -> None:
    generator = FakeGenerator(
        {"broken": RuntimeError("bad artifact"), "okay": ("ok",)}
    )
    session = ChatSession(generator)
    inputs = iter(("broken", "okay", "/exit"))
    output: list[str] = []

    assert (
        run_repl(
            session,
            input_fn=lambda _prompt: next(inputs),
            output_fn=output.append,
        )
        == 0
    )

    assert output == [
        "error: bad artifact",
        "assistant: ok",
        "bye",
    ]
    assert session.messages == (
        {"role": "user", "content": "broken"},
        {"role": "user", "content": "okay"},
        {"role": "assistant", "content": "ok"},
    )


class FakeTokenizer:
    eos_token_ids = (2,)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        assert messages == [{"role": "user", "content": "hello"}]
        assert add_generation_prompt is True
        assert enable_thinking is False
        return [1, 2] if tokenize else "hello<|assistant|><think></think>"


def test_glm52_config_requires_the_complete_strict_binding_input_set() -> None:
    config = ChatConfig(artifact="glm52", glm52_profile_path="models/glm52.yaml")

    with pytest.raises(ChatError, match="--glm52-config-path"):
        config.require_complete_glm52_composite()


def test_chat_template_validation_requires_disabled_thinking_and_token_ids() -> None:
    validate_chat_template(FakeTokenizer(), prompt="hello", strict_glm52=True)


def test_configure_parser_exports_a_register_ready_chat_command() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    configure_chat_parser(subparsers)

    args = parser.parse_args(["chat", "./artifact", "--max-tokens", "12"])

    config = ChatConfig.from_namespace(args)
    assert config == ChatConfig(artifact="./artifact", max_tokens=12)
    assert args.func.__name__ == "run_chat"


def test_preflight_runs_before_loading_and_can_block_without_a_model() -> None:
    events: list[str] = []

    class FailedReport:
        exit_code = 1

    def loader(_config: ChatConfig) -> ChatRuntime:
        events.append("load")
        raise AssertionError("the loader must not run after a failed preflight")

    with pytest.raises(ChatError, match="preflight failed"):
        run_chat_config(
            ChatConfig(artifact="./artifact"),
            doctor_preflight=lambda _config: (events.append("doctor") or FailedReport()),
            runtime_loader=loader,
        )

    assert events == ["doctor"]
