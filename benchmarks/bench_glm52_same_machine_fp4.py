#!/usr/bin/env python3
"""Run the frozen GLM52 same-machine pinned-FP4 benchmark.

The source control is deliberately explicit: upstream mlx-lm cannot load the
pinned ModelOpt NVFP4 snapshot, so this invokes the existing source-teacher
path (non-VQ package + streamed ModelOpt expert decode).  Its comparison is a
source-streaming reference, not a native-FP4 latency claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

def _benchmark_api() -> Any:
    """Load the isolated contract without executing ``keep.quality.__init__``.

    The package initializer currently imports MLX through unrelated GLM45
    surfaces.  Keeping this command's contract path MLX-free is necessary for
    ``--help`` and sealed-evidence comparison on non-Metal hosts.
    """

    name = "glm52_same_machine_fp4_contract"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "src/keep/quality/glm52_benchmark.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM52 benchmark contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_api = _benchmark_api()
CONTROL_IMPLEMENTATION = benchmark_api.CONTROL_IMPLEMENTATION
FROZEN_REPETITIONS = benchmark_api.FROZEN_REPETITIONS
FROZEN_SCENARIOS = benchmark_api.FROZEN_SCENARIOS
build_measurement_evidence = benchmark_api.build_measurement_evidence
build_benchmark_session_manifest = benchmark_api.build_benchmark_session_manifest
canonical_sha256 = benchmark_api.canonical_sha256
capture_machine_identity = benchmark_api.capture_machine_identity
evaluate_glm52_same_machine_benchmark = benchmark_api.evaluate_glm52_same_machine_benchmark
load_frozen_glm52_benchmark_contract = benchmark_api.load_frozen_glm52_benchmark_contract
measurement_locks = benchmark_api.measurement_locks
read_sealed_evidence = benchmark_api.read_sealed_evidence
write_sealed_evidence = benchmark_api.write_sealed_evidence


def _json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _prompt_ids(path: str | Path) -> list[int]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list) or len(value) != 1024:
        raise ValueError("--prompt-token-ids-json must contain exactly 1024 token IDs")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError("prompt token IDs must be non-negative integers")
    return list(value)


def _measure(
    *,
    role: str,
    scenario: str,
    operation: Callable[[], None],
    metric_snapshot: Callable[..., Mapping[str, Any]],
    vm_counters: Callable[[], Mapping[str, int] | None],
) -> dict[str, Any]:
    before = vm_counters()
    if before is None:
        raise RuntimeError("macOS vm_stat counters are required for release benchmark evidence")
    monotonic_start = time.monotonic_ns()
    wall_start = time.time_ns()
    start = time.perf_counter()
    operation()
    latency_ms = (time.perf_counter() - start) * 1000.0
    monotonic_end = time.monotonic_ns()
    wall_end = time.time_ns()
    after = vm_counters()
    if after is None:
        raise RuntimeError("macOS vm_stat counters are required after each release repetition")
    metrics = dict(metric_snapshot(previous_vm_stat_counts=before))
    pageouts_delta = int(after["pageouts"]) - int(before["pageouts"])
    swapouts_delta = int(after["swapouts"]) - int(before["swapouts"])
    valid = (
        math.isfinite(latency_ms)
        and latency_ms > 0
        and pageouts_delta == 0
        and swapouts_delta == 0
    )
    return {
        "role": role,
        "scenario": scenario,
        "latency_ms": latency_ms,
        "input_token_count": 1024,
        "output_token_count": 0 if scenario == "prefill_1k" else 128,
        "monotonic_start_ns": monotonic_start,
        "monotonic_end_ns": monotonic_end,
        "wall_start_ns": wall_start,
        "wall_end_ns": wall_end,
        "vm_stat_before": dict(before),
        "vm_stat_after": dict(after),
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
        "mlx_active_bytes": metrics.get("mlx_active_bytes"),
        "mlx_peak_bytes": metrics.get("mlx_peak_bytes"),
        "mlx_cache_bytes": metrics.get("mlx_cache_bytes"),
        "rss_bytes": metrics.get("rss_bytes"),
        "valid": valid,
    }


def _run_protocol(
    *,
    role: str,
    operations: Mapping[str, Callable[[], None]],
    metric_snapshot: Callable[..., Mapping[str, Any]],
    vm_counters: Callable[[], Mapping[str, int] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One excluded warmup then exactly three recorded attempts per scenario.

    A dirty attempt is retained and marked invalid.  It is never replaced,
    preventing a clean-only retry from concealing memory pressure.
    """

    warmups: list[dict[str, Any]] = []
    repetitions: list[dict[str, Any]] = []
    for scenario in FROZEN_SCENARIOS:
        warmup = _measure(
            role=role,
            scenario=scenario,
            operation=operations[scenario],
            metric_snapshot=metric_snapshot,
            vm_counters=vm_counters,
        )
        warmups.append(
            {
                **warmup,
                "record_kind": "warmup",
                "warmup_sequence_number": len(warmups),
                "excluded_from_gate": True,
            }
        )
        for run_index in range(FROZEN_REPETITIONS):
            measured = _measure(
                role=role,
                scenario=scenario,
                operation=operations[scenario],
                metric_snapshot=metric_snapshot,
                vm_counters=vm_counters,
            )
            repetitions.append(
                {
                    **measured,
                    "record_kind": "repetition",
                    "run_index": run_index,
                    "sequence_number": len(repetitions),
                }
            )
    return warmups, repetitions


def _candidate_operations(args: argparse.Namespace, policy: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Callable[[], None]]]:
    """Authenticate and bind the composite before timing a warm-resident model."""

    import mlx.core as mx

    from ramp.models.glm52_composite_loader import (
        assert_glm52_production_inputs_unchanged,
        load_authenticated_glm52_composite,
        validate_glm52_production_inputs,
    )
    from ramp.models.glm52_vq_adapter import GLM52VQModel

    token_ids = _prompt_ids(args.prompt_token_ids_json)
    validated = validate_glm52_production_inputs(
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
        model_id=str(policy["model_id"]),
        revision=str(policy["revision"]),
        prompt="same-machine pinned-FP4 benchmark",
    )
    model, _report = load_authenticated_glm52_composite(
        validated, model_factory=GLM52VQModel
    )
    mx.eval(model.parameters())
    mx.synchronize()
    mx.clear_cache()

    def prefill() -> None:
        cache = model.make_cache()
        logits = model(mx.array(token_ids, dtype=mx.int32)[None], cache=cache)
        mx.eval(logits)
        mx.synchronize()

    def decode() -> None:
        cache = model.make_cache()
        prompt = mx.array(token_ids, dtype=mx.int32)
        warm_logits = model(prompt[None], cache=cache)
        mx.eval(warm_logits)
        token = mx.argmax(warm_logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(token)
        mx.synchronize()
        for _ in range(128):
            logits = model(token[:, None], cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(token)
        mx.synchronize()

    return (
        {**dict(validated.artifact_identity.body), "sha256": validated.artifact_identity.sha256},
        {"prefill_1k": prefill, "decode_128": decode},
    )


def _load_teacher_contract(prompt_pack_json: str, source_contract_json: str) -> Any:
    path = Path(__file__).with_name("produce_glm52_teacher_cache.py")
    spec = importlib.util.spec_from_file_location("glm52_benchmark_teacher_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM52 source-teacher contract loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_contract(prompt_pack_json, source_contract_json)


def _cached_resolver(source: Any) -> Callable[[int], Any]:
    """Cache decoded projections within one control measurement process.

    The cache is intentionally process-local and discarded after this command;
    it only avoids re-decoding the same routed projection during the warmup and
    three frozen repetitions.  It never converts the source into a native FP4
    runtime.
    """

    cache: dict[tuple[int, int, str], Any] = {}

    def for_layer(layer_index: int) -> Any:
        base = source.expert_resolver_for_layer(layer_index)

        def resolve(expert_index: int) -> Mapping[str, Callable[[], Any]]:
            original = base(expert_index)
            resolved: dict[str, Callable[[], Any]] = {}
            for projection, loader in original.items():
                key = (layer_index, expert_index, projection)

                def cached(*, _key: tuple[int, int, str] = key, _loader: Callable[[], Any] = loader) -> Any:
                    if _key not in cache:
                        cache[_key] = _loader()
                    return cache[_key]

                resolved[projection] = cached
            return resolved

        return resolve

    return for_layer


def _control_operations(args: argparse.Namespace, policy: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Callable[[], None]]]:
    """Bind non-VQ tensors and run the existing ModelOpt streamed source path."""

    import mlx.core as mx

    from ramp.models.glm52_source_teacher import run_glm52_source_teacher
    from keep.quality.glm52_teacher_cache_producer import load_glm52_source_teacher_workload

    token_ids = _prompt_ids(args.prompt_token_ids_json)
    contract = _load_teacher_contract(args.prompt_pack_json, args.source_contract_json)
    source = load_glm52_source_teacher_workload(
        snapshot_dir=args.snapshot_dir,
        non_vq_package_dir=args.non_vq_artifact_dir,
        profile_path=args.profile_path,
        contract=contract,
    )
    resolver = _cached_resolver(source)

    def source_forward(ids: list[int]) -> list[int]:
        logits, _ = run_glm52_source_teacher(
            source.model,
            (tuple(ids),),
            expert_resolver_for_layer=resolver,
            expected_num_layers=source.expected_num_layers,
            pad_token_id=source.pad_token_id,
        )
        return [int(logits[0][-1].argmax())]

    def prefill() -> None:
        source_forward(token_ids)
        mx.synchronize()

    def decode() -> None:
        # There is no native ModelOpt KV-cache implementation.  This measures
        # 128 source-streamed next-token forwards after one warm prompt prefill.
        context = list(token_ids)
        next_token = source_forward(context)[0]
        for _ in range(128):
            context.append(next_token)
            next_token = source_forward(context)[0]
        mx.synchronize()

    control_identity = {
        "kind": CONTROL_IMPLEMENTATION["kind"],
        "model_id": policy.get("model_id"),
        "source_revision": policy.get("revision"),
        "source": dict(contract.source),
        "non_vq_package": dict(contract.non_vq_package),
    }
    return ({"identity": control_identity, "sha256": canonical_sha256(control_identity)}, {"prefill_1k": prefill, "decode_128": decode})


def _session_window(*, session_uuid: str, monotonic_start_ns: int, wall_start_ns: int) -> dict[str, object]:
    return {
        "session_uuid": session_uuid,
        "monotonic_start_ns": monotonic_start_ns,
        "monotonic_end_ns": time.monotonic_ns(),
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": time.time_ns(),
    }


def _measure_role(
    args: argparse.Namespace,
    *,
    role: str,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if role == "candidate":
        identity, operations = _candidate_operations(args, policy)
    else:
        identity, operations = _control_operations(args, policy)
    from ramp.benchmark.metrics import collect_metric_snapshot, collect_vm_stat_counts

    warmups, repetitions = _run_protocol(
        role=role,
        operations=operations,
        metric_snapshot=collect_metric_snapshot,
        vm_counters=collect_vm_stat_counts,
    )
    return identity, warmups, repetitions


def _run_role(args: argparse.Namespace, *, role: str) -> int:
    policy = _json_object(args.family_policy_json)
    contract = load_frozen_glm52_benchmark_contract(policy)
    with measurement_locks(args.output_json):
        machine = capture_machine_identity()
        monotonic_start_ns = time.monotonic_ns()
        wall_start_ns = time.time_ns()
        session_uuid = str(uuid.uuid4())
        prompt_token_ids = _prompt_ids(args.prompt_token_ids_json)
        benchmark_input_identity = {
            "token_count": len(prompt_token_ids),
            "token_ids_sha256": canonical_sha256(prompt_token_ids),
        }
        identity, warmups, repetitions = _measure_role(args, role=role, policy=policy)
        payload = build_measurement_evidence(
            role=role,
            contract=contract,
            machine_identity=machine,
            benchmark_input_identity=benchmark_input_identity,
            identity=identity,
            benchmark_session=_session_window(
                session_uuid=session_uuid,
                monotonic_start_ns=monotonic_start_ns,
                wall_start_ns=wall_start_ns,
            ),
            warmups=warmups,
            repetitions=repetitions,
        )
        write_sealed_evidence(args.output_json, payload)
    return 0


def _run_pair(args: argparse.Namespace) -> int:
    """Run candidate and source control under one host/session nonce and lock."""

    policy = _json_object(args.family_policy_json)
    contract = load_frozen_glm52_benchmark_contract(policy)
    with measurement_locks(args.candidate_output_json):
        machine = capture_machine_identity()
        monotonic_start_ns = time.monotonic_ns()
        wall_start_ns = time.time_ns()
        session_uuid = str(uuid.uuid4())
        prompt_token_ids = _prompt_ids(args.prompt_token_ids_json)
        benchmark_input_identity = {
            "token_count": len(prompt_token_ids),
            "token_ids_sha256": canonical_sha256(prompt_token_ids),
        }
        candidate_identity, candidate_warmups, candidate_repetitions = _measure_role(
            args, role="candidate", policy=policy
        )
        control_identity, control_warmups, control_repetitions = _measure_role(
            args, role="control", policy=policy
        )
        session = _session_window(
            session_uuid=session_uuid,
            monotonic_start_ns=monotonic_start_ns,
            wall_start_ns=wall_start_ns,
        )
        candidate = build_measurement_evidence(
            role="candidate",
            contract=contract,
            machine_identity=machine,
            benchmark_input_identity=benchmark_input_identity,
            identity=candidate_identity,
            benchmark_session=session,
            warmups=candidate_warmups,
            repetitions=candidate_repetitions,
        )
        control = build_measurement_evidence(
            role="control",
            contract=contract,
            machine_identity=machine,
            benchmark_input_identity=benchmark_input_identity,
            identity=control_identity,
            benchmark_session=session,
            warmups=control_warmups,
            repetitions=control_repetitions,
        )
        write_sealed_evidence(args.candidate_output_json, candidate)
        write_sealed_evidence(args.control_output_json, control)
        manifest = build_benchmark_session_manifest(
            benchmark_session=session,
            machine_identity=machine,
            candidate=candidate,
            control=control,
        )
        write_sealed_evidence(args.session_manifest_json, manifest)
    return 0


def _candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-index-path", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--non-vq-evidence-json", required=True)
    parser.add_argument("--routed-artifact-dir", required=True)
    parser.add_argument("--composite-audit-json", required=True)
    parser.add_argument("--materialization-runs-jsonl", required=True)
    parser.add_argument("--full-bind-preflight-json", required=True)
    parser.add_argument("--prompt-token-ids-json", required=True)
    parser.add_argument("--output-json", required=True)


def _control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--source-contract-json", required=True)
    parser.add_argument("--prompt-token-ids-json", required=True)
    parser.add_argument("--output-json", required=True)


def _pair_arguments(parser: argparse.ArgumentParser) -> None:
    """The paired form owns one process, one host identity, and one session nonce."""

    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--prompt-token-ids-json", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-index-path", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--non-vq-evidence-json", required=True)
    parser.add_argument("--routed-artifact-dir", required=True)
    parser.add_argument("--composite-audit-json", required=True)
    parser.add_argument("--materialization-runs-jsonl", required=True)
    parser.add_argument("--full-bind-preflight-json", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--source-contract-json", required=True)
    parser.add_argument("--candidate-output-json", required=True)
    parser.add_argument("--control-output-json", required=True)
    parser.add_argument("--session-manifest-json", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate", help="measure the authenticated composite")
    _candidate_arguments(candidate)
    control = commands.add_parser("control", help="measure the ModelOpt streamed source control")
    _control_arguments(control)
    pair = commands.add_parser(
        "pair",
        help="run both roles under one physical-host identity and session nonce",
    )
    _pair_arguments(pair)
    compare = commands.add_parser("compare", aliases=["evaluate"], help="seal the frozen gate evaluation")
    compare.add_argument("--family-policy-json", required=True)
    compare.add_argument("--candidate-json", required=True)
    compare.add_argument("--control-json", required=True)
    compare.add_argument("--session-manifest-json", required=True)
    compare.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "candidate":
        return _run_role(args, role="candidate")
    if args.command == "control":
        return _run_role(args, role="control")
    if args.command == "pair":
        return _run_pair(args)
    policy = _json_object(args.family_policy_json)
    result = evaluate_glm52_same_machine_benchmark(
        policy=policy,
        candidate=read_sealed_evidence(args.candidate_json),
        control=read_sealed_evidence(args.control_json),
        session_manifest=read_sealed_evidence(args.session_manifest_json),
    )
    write_sealed_evidence(args.output_json, result)
    return 0 if result["frozen_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
