from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlx.core as mx
from mlx_lm.generate import generate_step, generation_stream
from mlx_lm.utils import load_tokenizer

from ramp.benchmark.metrics import (
    MemoryPhaseTracer,
    collect_metric_snapshot,
    reset_mlx_peak_memory,
    wait_for_memory_quiet,
)
from ramp.models.glm52_composite_loader import (
    GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW,
    GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES,
    _canonical_sha256,
    assert_glm52_production_inputs_unchanged,
    build_glm52_composite_artifact_identity,
    load_authenticated_glm52_composite,
    snapshot_glm52_production_inputs,
    validate_glm52_composite_claims,
    validate_glm52_full_bind_claims,
    validate_glm52_non_vq_evidence_claims,
    validate_glm52_production_inputs,
    validate_glm52_routed_manifest_policy,
)
from ramp.models.glm52_vq_adapter import GLM52VQModel
from keep.quality.glm52_family import (
    GLM52_EOS_TOKEN_IDS,
    GLM52_TOKENIZER_BASE_VOCAB_SIZE,
    GLM52_TOKENIZER_LENGTH,
    validate_glm52_tokenizer_readiness,
)


PROBE_RECORD_TYPE = "glm52_production_generation_probe"
PROBE_READY_STATUS = "glm52_production_generation_probe_ready"
PROBE_FAILED_STATUS = "glm52_production_generation_probe_failed"
MODEL_VOCAB_SIZE = 154_880


class GLM52ProductionProbeError(RuntimeError):
    def __init__(
        self,
        stage: str,
        error: Exception,
        phase_memory: Sequence[Mapping[str, Any]],
    ) -> None:
        self.stage = stage
        self.error_type = type(error).__name__
        self.phase_memory = [dict(row) for row in phase_memory]
        super().__init__(str(error))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_output_path(
    output_json: str | Path,
    *,
    authority_roots: Sequence[str | Path],
    protected_files: Sequence[str | Path],
) -> Path:
    output = Path(output_json).expanduser().resolve(strict=False)
    if output.suffix != ".json":
        raise ValueError("production output must use the .json suffix")
    if output.is_dir():
        raise ValueError("production output must be a JSON file")
    roots = tuple(Path(root).expanduser().resolve() for root in authority_roots)
    if any(_is_within(output, root) for root in roots):
        raise ValueError("production output must remain outside authority/artifact roots")
    protected = tuple(
        Path(path).expanduser().resolve(strict=False) for path in protected_files
    )
    if output in protected:
        raise ValueError("production output aliases a protected authority input")
    if output.exists() and any(
        path.exists() and output.samefile(path) for path in protected
    ):
        raise ValueError("production output aliases a protected authority input")
    if output.exists():
        for root in roots:
            if any(
                candidate.is_file() and output.samefile(candidate)
                for candidate in root.rglob("*")
            ):
                raise ValueError(
                    "production output aliases a file inside an authority/artifact root"
                )
    return output


def _directory_identity(path: str | Path) -> tuple[int, int]:
    directory = Path(path)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"output parent is not a real directory: {directory}")
    stat = directory.stat()
    return stat.st_dev, stat.st_ino


def _write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    output = Path(path)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(output.parent, parent_flags)
    temporary_name = f".{output.name}.partial-{uuid4().hex}"
    try:
        parent_stat = os.fstat(parent_fd)
        actual_parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        if (
            expected_parent_identity is not None
            and actual_parent_identity != expected_parent_identity
        ):
            raise ValueError("output parent changed after path validation")
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            file_flags,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def _greedy_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def run_one_greedy_token(
    model: Any,
    token_ids: Sequence[int],
    *,
    generation_factory: Callable[..., Any] = generate_step,
    synchronize: Callable[[Any], Any] = mx.synchronize,
    generation_stream_value: Any = generation_stream,
) -> dict[str, Any]:
    if not token_ids:
        raise ValueError("production prompt token IDs must not be empty")
    prompt = mx.array(list(token_ids), dtype=mx.int32)
    prompt_cache = model.make_cache()
    generator = generation_factory(
        prompt,
        model,
        max_tokens=1,
        sampler=_greedy_sampler,
        prompt_cache=prompt_cache,
    )
    try:
        token, logprobs = next(generator)
        try:
            next(generator)
        except StopIteration:
            pass
        else:
            raise ValueError("generate_step exceeded the one-token limit")
    except StopIteration as error:
        raise ValueError("generate_step produced no token") from error
    finally:
        try:
            generator.close()
        finally:
            synchronize(generation_stream_value)
    mx.eval(token, logprobs)
    token_id = int(token.item()) if hasattr(token, "item") else int(token)
    finite = bool(mx.all(mx.isfinite(logprobs)).item())
    shape = [int(dim) for dim in logprobs.shape]
    return {
        "generated_token_ids": [token_id],
        "generated_token_count": 1,
        "generated_ids_within_vocab": 0 <= token_id < MODEL_VOCAB_SIZE,
        "logits_shape": shape,
        "logits_finite": finite,
        "greedy_sampling": True,
        "explicit_prompt_cache": True,
        "generation_method": "mlx_lm_generate_step",
        "lookahead_forward_scheduled": True,
    }


def run_one_direct_greedy_token(
    model: Any,
    token_ids: Sequence[int],
    *,
    stream_context: Callable[[Any], Any] = mx.stream,
    clear_cache: Callable[[], Any] = mx.clear_cache,
    synchronize: Callable[[Any], Any] = mx.synchronize,
    generation_stream_value: Any = generation_stream,
) -> dict[str, Any]:
    if not token_ids:
        raise ValueError("production prompt token IDs must not be empty")
    prompt = mx.array(list(token_ids), dtype=mx.int32)
    prompt_cache = model.make_cache()
    try:
        with stream_context(generation_stream_value):
            if len(prompt) > 1:
                model(prompt[:-1][None], cache=prompt_cache)
                mx.eval([cache.state for cache in prompt_cache])
                clear_cache()
            logits = model(prompt[-1:][None], cache=prompt_cache)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            token = _greedy_sampler(logprobs)
            mx.async_eval(token, logprobs)
    finally:
        synchronize(generation_stream_value)
    mx.eval(token, logprobs)
    token_id = int(token.item()) if hasattr(token, "item") else int(token)
    logprobs = logprobs.squeeze(0)
    finite = bool(mx.all(mx.isfinite(logprobs)).item())
    shape = [int(dim) for dim in logprobs.shape]
    return {
        "generated_token_ids": [token_id],
        "generated_token_count": 1,
        "generated_ids_within_vocab": 0 <= token_id < MODEL_VOCAB_SIZE,
        "logits_shape": shape,
        "logits_finite": finite,
        "greedy_sampling": True,
        "explicit_prompt_cache": True,
        "generation_method": "direct_prefill_single_decode",
        "lookahead_forward_scheduled": False,
        "prefill_token_count": max(0, len(token_ids) - 1),
        "decode_token_count": 1,
    }


def clear_glm52_mlx_cache(
    *,
    synchronize: Callable[[], Any] = mx.synchronize,
    clear_cache: Callable[[], Any] = mx.clear_cache,
    metric_snapshot: Callable[[], Mapping[str, Any]] = collect_metric_snapshot,
) -> dict[str, Any]:
    before_cache_clear = dict(metric_snapshot())
    clear_cache()
    synchronize()
    after_cache_clear = dict(metric_snapshot())
    before_cache_bytes = before_cache_clear.get("mlx_cache_bytes")
    after_cache_bytes = after_cache_clear.get("mlx_cache_bytes")
    if (
        type(before_cache_bytes) is not int
        or type(after_cache_bytes) is not int
        or before_cache_bytes < 0
        or after_cache_bytes < 0
        or after_cache_bytes > before_cache_bytes
        or after_cache_bytes != 0
    ):
        raise ValueError(
            "residency cache clear returned invalid MLX cache accounting: "
            f"before={before_cache_bytes!r} after={after_cache_bytes!r}"
        )
    return {
        "before_memory": before_cache_clear,
        "after_memory": after_cache_clear,
        "released_bytes": before_cache_bytes - after_cache_bytes,
    }


def require_glm52_memory_quiet(
    *,
    quiet_probe: Callable[..., Mapping[str, Any]] = wait_for_memory_quiet,
    quiet_window_seconds: float = 15.0,
    quiet_max_attempts: int = 3,
) -> dict[str, Any]:
    memory_quiet = dict(
        quiet_probe(
            window_seconds=quiet_window_seconds,
            max_attempts=quiet_max_attempts,
        )
    )
    if not all(
        (
            memory_quiet.get("enabled") is True,
            memory_quiet.get("available") is True,
            memory_quiet.get("quiet") is True,
            type(memory_quiet.get("pageouts_delta")) is int,
            memory_quiet.get("pageouts_delta") == 0,
            type(memory_quiet.get("swapouts_delta")) is int,
            memory_quiet.get("swapouts_delta") == 0,
        )
    ):
        raise ValueError(
            f"resident generation requires a quiet memory window: {memory_quiet}"
        )
    return memory_quiet


def establish_glm52_residency(
    model: Any,
    *,
    evaluate_parameters: Callable[[Any], Any] = mx.eval,
    synchronize: Callable[[], Any] = mx.synchronize,
    clear_cache: Callable[[], Any] = mx.clear_cache,
    quiet_probe: Callable[..., Mapping[str, Any]] = wait_for_memory_quiet,
    metric_snapshot: Callable[[], Mapping[str, Any]] = collect_metric_snapshot,
    quiet_window_seconds: float = 15.0,
    quiet_max_attempts: int = 3,
) -> dict[str, Any]:
    evaluate_parameters(model.parameters())
    synchronize()
    cache_clear = clear_glm52_mlx_cache(
        synchronize=synchronize,
        clear_cache=clear_cache,
        metric_snapshot=metric_snapshot,
    )
    memory_quiet = require_glm52_memory_quiet(
        quiet_probe=quiet_probe,
        quiet_window_seconds=quiet_window_seconds,
        quiet_max_attempts=quiet_max_attempts,
    )
    resident_memory = dict(metric_snapshot())
    active_bytes = resident_memory.get("mlx_active_bytes")
    if type(active_bytes) is not int or active_bytes < GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES:
        raise ValueError(
            "resident MLX active bytes do not cover the authenticated tensor payload: "
            f"active={active_bytes!r} required={GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES}"
        )
    return {
        "parameter_residency_established": True,
        "cache_clear": cache_clear,
        "memory_quiet": memory_quiet,
        "resident_memory": resident_memory,
    }


def glm52_memory_phase_is_clean(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> bool:
    return all(
        (
            payload.get("label") == label,
            payload.get("available") is True,
            type(payload.get("pageouts_delta")) is int,
            payload.get("pageouts_delta") == 0,
            type(payload.get("swapouts_delta")) is int,
            payload.get("swapouts_delta") == 0,
        )
    )


def classify_glm52_residency_evidence(
    *,
    parameter_residency_established: Any,
    system_wired_default: Any,
    generation_warmup_memory_clean: Any,
    cold_residency_memory: Mapping[str, Any],
    pre_generation_memory: Mapping[str, Any],
    steady_state_generation_memory: Mapping[str, Any],
) -> dict[str, bool]:
    cold_clean = glm52_memory_phase_is_clean(
        cold_residency_memory,
        label="after_residency_quiet",
    )
    pre_generation_clean = glm52_memory_phase_is_clean(
        pre_generation_memory,
        label="before_generation",
    )
    generation_clean = glm52_memory_phase_is_clean(
        steady_state_generation_memory,
        label="after_generation",
    )
    warm_residency_proven = bool(
        parameter_residency_established is True
        and system_wired_default is True
        and pre_generation_clean
        and generation_clean
    )
    return {
        "cold_residency_memory_clean": cold_clean,
        "pre_generation_memory_clean": pre_generation_clean,
        "steady_state_generation_memory_clean": generation_clean,
        "warm_residency_proven": warm_residency_proven,
        "production_residency_proven": bool(
            cold_clean
            and generation_warmup_memory_clean is True
            and warm_residency_proven
        ),
    }


def validate_glm52_generation_record(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("generated_token_count") != 1:
        raise ValueError("production probe must generate exactly one token")
    if payload.get("logits_shape") != [MODEL_VOCAB_SIZE]:
        raise ValueError("production generation must return one full-vocabulary row")
    if payload.get("greedy_sampling") is not True:
        raise ValueError("production probe must use greedy sampling")
    if payload.get("logits_finite") is not True:
        raise ValueError("production generation logits must be finite")
    generated_ids = payload.get("generated_token_ids")
    if (
        not isinstance(generated_ids, list)
        or len(generated_ids) != 1
        or type(generated_ids[0]) is not int
        or not 0 <= generated_ids[0] < MODEL_VOCAB_SIZE
    ):
        raise ValueError("production generated token is outside the model vocabulary")
    return dict(payload)


def validate_glm52_generation_consistency(
    warmup: Mapping[str, Any],
    measured: Mapping[str, Any],
) -> bool:
    if warmup.get("generated_token_ids") != measured.get("generated_token_ids"):
        raise ValueError("warmup and measured greedy generation token IDs differ")
    return True


def validate_glm52_measured_generation_record(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_glm52_generation_record(payload)
    if validated.get("generation_method") != "direct_prefill_single_decode":
        raise ValueError(
            "measured generation must use direct prefill and a single decode"
        )
    if validated.get("lookahead_forward_scheduled") is not False:
        raise ValueError("measured generation must not schedule a lookahead forward")
    return validated


def _load_production_tokenizer(tokenizer_dir: str | Path):
    return load_tokenizer(
        tokenizer_dir,
        {"local_files_only": True, "trust_remote_code": False},
        eos_token_ids=list(GLM52_EOS_TOKEN_IDS),
    )


def validate_glm52_live_tokenizer(
    wrapper: Any,
    *,
    authenticated_readiness: Mapping[str, Any],
    prompt: str,
    rendered_prompt: str,
) -> None:
    """Recheck the exact tokenizer instance used by production generation."""

    tokenizer = getattr(wrapper, "_tokenizer", None)
    if tokenizer is None:
        raise ValueError("production tokenizer wrapper has no underlying tokenizer")
    checks = (
        (
            "base vocabulary",
            getattr(tokenizer, "vocab_size", None),
            GLM52_TOKENIZER_BASE_VOCAB_SIZE,
        ),
        ("tokenizer length", len(tokenizer), GLM52_TOKENIZER_LENGTH),
        ("pad token", getattr(tokenizer, "pad_token_id", None), GLM52_EOS_TOKEN_IDS[0]),
        ("EOS token", getattr(tokenizer, "eos_token_id", None), GLM52_EOS_TOKEN_IDS[0]),
    )
    for label, actual, expected in checks:
        if type(actual) is not int or actual != expected:
            raise ValueError(
                f"production tokenizer {label} must be {expected}, found {actual!r}"
            )
    wrapper_eos = tuple(sorted(int(value) for value in wrapper.eos_token_ids))
    if wrapper_eos != tuple(GLM52_EOS_TOKEN_IDS):
        raise ValueError("production tokenizer wrapper EOS identity drifted")
    if authenticated_readiness.get("prompt") != prompt:
        raise ValueError("production prompt does not match authenticated tokenizer evidence")
    if authenticated_readiness.get("thinking_disabled_render") != rendered_prompt:
        raise ValueError("production chat render does not match authenticated evidence")


def audit_default_wired_policy() -> dict[str, Any]:
    override_names = (
        "GLM_MLX_WIRED_LIMIT_GB",
        "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB",
    )
    overrides = {
        name: os.environ[name]
        for name in override_names
        if os.environ.get(name)
    }
    system = platform.system()
    wired_limit_mb: int | None = None
    sysctl_available = False
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "iogpu.wired_limit_mb"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            wired_limit_mb = int(result.stdout.strip())
            sysctl_available = True
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {
        "platform": system,
        "environment_overrides": overrides,
        "iogpu_wired_limit_mb": wired_limit_mb,
        "iogpu_wired_limit_available": sysctl_available,
        "system_wired_default": bool(
            system == "Darwin"
            and sysctl_available
            and wired_limit_mb == 0
            and not overrides
        ),
    }


def _phase_recorder(
    memory_snapshot: Callable[[str], Mapping[str, Any]] | None,
) -> tuple[Callable[[str], None], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    if memory_snapshot is None:
        reset_mlx_peak_memory()
        tracer = MemoryPhaseTracer(enabled=True)

        def snapshot(label: str) -> Mapping[str, Any]:
            tracer.mark(label)
            return tracer.records[-1]

    else:
        snapshot = memory_snapshot

    def mark(label: str) -> None:
        record = dict(snapshot(label))
        record["label"] = label
        records.append(record)

    return mark, records


def probe_glm52_production_generation(
    *,
    profile_path: str | Path,
    config_path: str | Path,
    source_index_path: str | Path,
    tokenizer_dir: str | Path,
    tokenizer_readiness_json: str | Path,
    family_policy_json: str | Path,
    non_vq_artifact_dir: str | Path,
    non_vq_evidence_json: str | Path,
    routed_artifact_dir: str | Path,
    composite_audit_json: str | Path,
    materialization_runs_jsonl: str | Path,
    full_bind_preflight_json: str | Path,
    model_id: str,
    revision: str,
    prompt: str,
    input_validator: Callable[..., Any] = validate_glm52_production_inputs,
    tokenizer_factory: Callable[..., Any] = _load_production_tokenizer,
    composite_loader: Callable[..., Any] = load_authenticated_glm52_composite,
    residency_preparer: Callable[..., Mapping[str, Any]] = (
        establish_glm52_residency
    ),
    generation_runner: Callable[..., Mapping[str, Any]] = (
        run_one_direct_greedy_token
    ),
    generation_warmup_runner: Callable[..., Mapping[str, Any]] = (
        run_one_greedy_token
    ),
    generation_warmup_cache_clearer: Callable[..., Mapping[str, Any]] = (
        clear_glm52_mlx_cache
    ),
    generation_warmup_quiet_preparer: Callable[..., Mapping[str, Any]] = (
        require_glm52_memory_quiet
    ),
    model_factory: Callable[..., Any] = GLM52VQModel,
    memory_snapshot: Callable[[str], Mapping[str, Any]] | None = None,
    generation_peak_resetter: Callable[[], Any] = reset_mlx_peak_memory,
    wired_policy_probe: Callable[[], Mapping[str, Any]] = (
        audit_default_wired_policy
    ),
    tokenizer_reauthenticator: Callable[..., Any] = (
        validate_glm52_tokenizer_readiness
    ),
    input_integrity_checker: Callable[..., Any] = (
        assert_glm52_production_inputs_unchanged
    ),
) -> dict[str, Any]:
    mark, phase_memory = _phase_recorder(memory_snapshot)
    stage = "memory_policy"
    try:
        mark("before_memory_policy")
        wired_policy = dict(wired_policy_probe())
        if wired_policy.get("system_wired_default") is not True:
            raise ValueError(
                "system wired memory defaults are required; custom wired limits "
                f"are forbidden for this production proof: {wired_policy}"
            )
        stage = "input_authentication"
        mark("before_input_validation")
        validated = input_validator(
            profile_path=profile_path,
            config_path=config_path,
            source_index_path=source_index_path,
            tokenizer_dir=tokenizer_dir,
            tokenizer_readiness_json=tokenizer_readiness_json,
            family_policy_json=family_policy_json,
            non_vq_artifact_dir=non_vq_artifact_dir,
            non_vq_evidence_json=non_vq_evidence_json,
            routed_artifact_dir=routed_artifact_dir,
            composite_audit_json=composite_audit_json,
            materialization_runs_jsonl=materialization_runs_jsonl,
            full_bind_preflight_json=full_bind_preflight_json,
            model_id=model_id,
            revision=revision,
            prompt=prompt,
        )
        mark("after_fresh_payload_audits")

        stage = "tokenizer_and_prompt"
        input_integrity_checker(validated.input_fingerprint)
        tokenizer_reauthenticator(
            validated.tokenizer_readiness,
            tokenizer_dir=validated.tokenizer_dir,
        )
        tokenizer = tokenizer_factory(validated.tokenizer_dir)
        tokenizer_reauthenticator(
            validated.tokenizer_readiness,
            tokenizer_dir=validated.tokenizer_dir,
        )
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = tokenizer.apply_chat_template(
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
            or any(type(token_id) is not int for token_id in token_ids)
            or any(not 0 <= token_id < MODEL_VOCAB_SIZE for token_id in token_ids)
        ):
            raise ValueError("chat template produced invalid production token IDs")
        rendered_text = str(rendered_prompt)
        if not rendered_text.rstrip().endswith("<|assistant|><think></think>"):
            raise ValueError("chat template did not use the disabled-thinking assistant form")
        validate_glm52_live_tokenizer(
            tokenizer,
            authenticated_readiness=validated.tokenizer_readiness,
            prompt=prompt,
            rendered_prompt=rendered_text,
        )
        input_integrity_checker(validated.input_fingerprint)
        mark("after_tokenizer_and_prompt")

        stage = "model_binding"
        model, load_report = composite_loader(
            validated,
            model_factory=model_factory,
            integrity_checker=input_integrity_checker,
            phase_marker=mark,
        )
        stage = "residency"
        input_integrity_checker(validated.input_fingerprint)
        mark("before_residency_barrier")
        residency = dict(residency_preparer(model))
        mark("after_residency_quiet")
        input_integrity_checker(validated.input_fingerprint)
        stage = "generation_warmup"
        mark("before_generation_warmup")
        generation_warmup = validate_glm52_generation_record(
            generation_warmup_runner(model, token_ids)
        )
        mark("after_generation_warmup")
        input_integrity_checker(validated.input_fingerprint)
        stage = "generation_warmup_cache_clear"
        generation_warmup_cache_clear = dict(
            generation_warmup_cache_clearer()
        )
        mark("after_generation_warmup_cache_clear")
        stage = "generation_warmup_memory_quiet"
        generation_warmup_memory_quiet = dict(
            generation_warmup_quiet_preparer()
        )
        mark("after_generation_warmup_quiet")
        input_integrity_checker(validated.input_fingerprint)
        stage = "generation_peak_reset"
        generation_peak_resetter()
        mark("before_generation")
        stage = "generation"
        generation = validate_glm52_measured_generation_record(
            generation_runner(model, token_ids)
        )
        mark("after_generation")
        input_integrity_checker(validated.input_fingerprint)
        stage = "generation_consistency"
        validate_glm52_generation_consistency(
            generation_warmup,
            generation,
        )
        generated_ids = generation.get("generated_token_ids")
        assert isinstance(generated_ids, list)
        decoded = tokenizer.decode(generated_ids)
    except GLM52ProductionProbeError:
        raise
    except Exception as error:
        raise GLM52ProductionProbeError(stage, error, phase_memory) from error

    after_residency_memory = next(
        (
            dict(record)
            for record in phase_memory
            if record.get("label") == "after_residency_quiet"
        ),
        {},
    )
    after_generation_warmup_memory = next(
        (
            dict(record)
            for record in phase_memory
            if record.get("label") == "after_generation_warmup"
        ),
        {},
    )
    generation_warmup_memory_clean = glm52_memory_phase_is_clean(
        after_generation_warmup_memory,
        label="after_generation_warmup",
    )
    after_generation_warmup_cache_clear_memory = next(
        (
            dict(record)
            for record in phase_memory
            if record.get("label") == "after_generation_warmup_cache_clear"
        ),
        {},
    )
    after_generation_warmup_quiet_memory = next(
        (
            dict(record)
            for record in phase_memory
            if record.get("label") == "after_generation_warmup_quiet"
        ),
        {},
    )
    before_generation_memory = next(
        (
            dict(record)
            for record in phase_memory
            if record.get("label") == "before_generation"
        ),
        {},
    )
    after_generation_memory = next(
        (
            dict(record)
            for record in phase_memory
            if record.get("label") == "after_generation"
        ),
        {},
    )
    residency_claims = classify_glm52_residency_evidence(
        parameter_residency_established=residency[
            "parameter_residency_established"
        ],
        system_wired_default=wired_policy["system_wired_default"],
        generation_warmup_memory_clean=generation_warmup_memory_clean,
        cold_residency_memory=after_residency_memory,
        pre_generation_memory=before_generation_memory,
        steady_state_generation_memory=after_generation_memory,
    )

    return {
        "schema_version": 1,
        "record_type": PROBE_RECORD_TYPE,
        "probe_status": PROBE_READY_STATUS,
        "probe_pass": True,
        "full_model_scope": "production_composite",
        "model_id": model_id,
        "source_revision": revision,
        "production_artifact_used": True,
        "production_binding_proven": True,
        "production_generation_proven": True,
        "whole_model_runtime_proven": True,
        "accepted_whole_model_tensor_payload_bytes": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BYTES
        ),
        "accepted_whole_model_tensor_payload_bpw": (
            GLM52_ACCEPTED_TENSOR_PAYLOAD_BPW
        ),
        "full_routed_group_count": 225,
        "artifact_identity": dict(validated.artifact_identity.body),
        "artifact_identity_sha256": validated.artifact_identity.sha256,
        "common_artifact_identity": dict(validated.artifact_identity.body),
        "common_artifact_identity_sha256": validated.artifact_identity.sha256,
        "input_evidence_file_sha256": dict(
            validated.input_evidence_file_sha256
        ),
        "input_fingerprint_reverified_before_bind": True,
        "input_fingerprint_reverified_after_residency": True,
        "input_fingerprint_reverified_after_warmup": True,
        "input_fingerprint_reverified_after_warmup_preparation": True,
        "input_fingerprint_reverified_before_generation": True,
        "input_fingerprint_reverified_after_generation": True,
        "parameter_residency_established": residency[
            "parameter_residency_established"
        ],
        "residency_cache_clear": dict(residency["cache_clear"]),
        "resident_memory_quiet": dict(residency["memory_quiet"]),
        "resident_memory": dict(residency["resident_memory"]),
        "generation_warmup_proven": True,
        "generation_warmup": dict(generation_warmup),
        "generation_warmup_phase_memory": after_generation_warmup_memory,
        "generation_warmup_memory_clean": generation_warmup_memory_clean,
        "generation_warmup_cache_clear": generation_warmup_cache_clear,
        "post_warmup_cache_clear_phase_memory": (
            after_generation_warmup_cache_clear_memory
        ),
        "generation_warmup_memory_quiet": generation_warmup_memory_quiet,
        "post_warmup_quiet_phase_memory": (
            after_generation_warmup_quiet_memory
        ),
        "measured_generation_after_warmup": True,
        "warmup_measured_token_match": True,
        "measured_generation_avoids_lookahead": bool(
            generation.get("generation_method")
            == "direct_prefill_single_decode"
            and generation.get("lookahead_forward_scheduled") is False
        ),
        "generation_measured_after_residency": True,
        "mlx_peak_reset_before_generation": True,
        "cold_residency_mlx_peak_scope": "since_probe_start",
        "steady_state_generation_mlx_peak_scope": (
            "since_pre_generation_reset"
        ),
        "cold_residency_memory": after_residency_memory,
        "cold_residency_memory_clean": residency_claims[
            "cold_residency_memory_clean"
        ],
        "pre_generation_memory": before_generation_memory,
        "pre_generation_memory_clean": residency_claims[
            "pre_generation_memory_clean"
        ],
        "steady_state_generation_memory": after_generation_memory,
        "steady_state_generation_memory_clean": (
            residency_claims["steady_state_generation_memory_clean"]
        ),
        "warm_residency_proven": residency_claims["warm_residency_proven"],
        "production_residency_proven": residency_claims[
            "production_residency_proven"
        ],
        "non_vq_bind_report": load_report.non_vq_bind_report.to_dict(),
        "bound_sparse_layer_ids": list(load_report.bound_sparse_layer_ids),
        "dense_routed_parameter_names": list(
            load_report.dense_routed_parameter_names
        ),
        "dense_routed_experts": bool(load_report.dense_routed_parameter_names),
        "unbound_vq_experts": load_report.unbound_vq_experts,
        "tokenizer_used": True,
        "chat_template_add_generation_prompt": True,
        "chat_template_enable_thinking": False,
        "prompt": prompt,
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "generated_text": str(decoded),
        **generation,
        "phase_memory": phase_memory,
        "rss_semantics": "process_peak_ru_maxrss",
        "wired_memory_policy": wired_policy,
        "system_wired_default": wired_policy["system_wired_default"],
        "custom_mlx_wired_limit_used": bool(
            wired_policy.get("environment_overrides")
        ),
        "long_context_indexshare_proven": False,
        "quality_claim": False,
        "speed_claim": False,
        "same_machine_benchmark_proven": False,
        "full_vocabulary_eval_proven": False,
    }


def _failure_payload(
    *,
    model_id: str,
    revision: str,
    prompt: str,
    error: Exception,
) -> dict[str, Any]:
    if isinstance(error, GLM52ProductionProbeError):
        stage = error.stage
        error_type = error.error_type
        phase_memory = error.phase_memory
    else:
        stage = "unexpected_error"
        error_type = type(error).__name__
        phase_memory = []
    return {
        "schema_version": 1,
        "record_type": PROBE_RECORD_TYPE,
        "probe_status": PROBE_FAILED_STATUS,
        "probe_pass": False,
        "model_id": model_id,
        "source_revision": revision,
        "prompt": prompt,
        "production_artifact_used": False,
        "production_binding_proven": False,
        "production_generation_proven": False,
        "whole_model_runtime_proven": False,
        "generation_warmup_proven": False,
        "generation_warmup_memory_clean": False,
        "measured_generation_after_warmup": False,
        "parameter_residency_established": False,
        "steady_state_generation_memory_clean": False,
        "cold_residency_memory_clean": False,
        "pre_generation_memory_clean": False,
        "warm_residency_proven": False,
        "production_residency_proven": False,
        "long_context_indexshare_proven": False,
        "quality_claim": False,
        "speed_claim": False,
        "same_machine_benchmark_proven": False,
        "full_vocabulary_eval_proven": False,
        "phase_memory": phase_memory,
        "failure": {
            "stage": stage,
            "type": error_type,
            "message": str(error),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate, bind, and generate one token with the full GLM52 composite."
    )
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-index-path", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--non-vq-evidence-json", required=True)
    parser.add_argument("--routed-artifact-dir", required=True)
    parser.add_argument("--composite-audit-json", required=True)
    parser.add_argument("--materialization-runs-jsonl", required=True)
    parser.add_argument("--full-bind-preflight-json", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = (
        args.tokenizer_dir,
        args.non_vq_artifact_dir,
        args.routed_artifact_dir,
    )
    protected = (
        args.profile_path,
        args.config_path,
        args.source_index_path,
        args.tokenizer_readiness_json,
        args.family_policy_json,
        args.non_vq_evidence_json,
        args.composite_audit_json,
        args.materialization_runs_jsonl,
        args.full_bind_preflight_json,
    )
    try:
        output = _validated_output_path(
            args.output_json,
            authority_roots=roots,
            protected_files=protected,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output = _validated_output_path(
            output,
            authority_roots=roots,
            protected_files=protected,
        )
        output_parent_identity = _directory_identity(output.parent)
    except (OSError, ValueError) as error:
        print(f"invalid production output path: {error}", file=sys.stderr)
        return 1
    try:
        payload = probe_glm52_production_generation(
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
            model_id=args.model_id,
            revision=args.revision,
            prompt=args.prompt,
        )
        return_code = 0
    except Exception as error:
        payload = _failure_payload(
            model_id=args.model_id,
            revision=args.revision,
            prompt=args.prompt,
            error=error,
        )
        return_code = 1
    try:
        revalidated_output = _validated_output_path(
            args.output_json,
            authority_roots=roots,
            protected_files=protected,
        )
        if revalidated_output != output:
            raise ValueError("production output path changed during the probe")
        _write_json_atomic(
            output,
            payload,
            expected_parent_identity=output_parent_identity,
        )
    except (OSError, ValueError) as error:
        print(f"could not write production evidence safely: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
