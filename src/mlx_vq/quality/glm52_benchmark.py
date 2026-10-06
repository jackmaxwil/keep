"""Frozen GLM-5.2 same-machine pinned-FP4 benchmark contract.

This module intentionally has no MLX imports.  It owns the release contract,
evidence sealing, identity checks, and lock protocol; the benchmark CLI imports
MLX only after it has acquired both locks on a Metal-capable host.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


RECORD_TYPE = "glm52_same_machine_fp4_measurement"
EVALUATION_RECORD_TYPE = "glm52_same_machine_fp4_benchmark_gate"
FROZEN_SCENARIOS = ("prefill_1k", "decode_128")
FROZEN_REPETITIONS = 3
FROZEN_RATIO = 1.15
FROZEN_BASELINE = "same_machine_pinned_fp4_source_streaming_control"
FROZEN_RATIO_POLICY_KEY = "benchmark_gate.maximum_candidate_to_reference_ratio"
FROZEN_RATIO_STATISTIC = "per_scenario_median_candidate_latency_ms_divided_by_median_control_latency_ms_v1"
EXPECTED_CANDIDATE_COMPOSITE_IDENTITY_SHA256 = (
    "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
)
EVIDENCE_PROVENANCE = "local_filesystem_hash_referenced_evidence_v1"
MAX_SESSION_DURATION_NS = 6 * 60 * 60 * 1_000_000_000
MAX_CLOCK_SKEW_NS = 5 * 60 * 1_000_000_000
HEAVY_JOB_LOCK_PATH = Path(__file__).resolve().parents[3] / ".keep-heavy-job.lock"
SCENARIO_DEFINITIONS = {
    "prefill_1k": {
        "prompt_token_count": 1024,
        "latency_scope": "one 1024-token prompt prefill",
    },
    "decode_128": {
        "prompt_token_count": 1024,
        "decode_token_count": 128,
        "latency_scope": "128 decode tokens after an untimed warm prompt prefill",
    },
}

CONTROL_IMPLEMENTATION = {
    "kind": "modelopt_nvfp4_stream_dequantized_source_teacher",
    "upstream_mlx_lm_modelopt_nvfp4_supported": False,
    "native_fp4_runtime": False,
    "measures": (
        "full GLM52 source forward work after binding the authenticated non-VQ "
        "package, while routed experts are lazily decoded with "
        "read_modelopt_nvfp4_weight and reused by the control cache"
    ),
    "does_not_measure": (
        "native ModelOpt NVFP4 kernels or a latency-realistic upstream FP4 runtime"
    ),
}


@dataclass(frozen=True)
class GLM52BenchmarkContract:
    policy_contract_sha256: str
    required_scenarios: tuple[str, ...]
    minimum_repetitions_per_scenario: int
    maximum_candidate_to_reference_ratio: float
    comparison_baseline: str


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def load_frozen_glm52_benchmark_contract(
    policy: Mapping[str, Any],
) -> GLM52BenchmarkContract:
    """Accept only the immutable benchmark keys from the frozen policy."""

    gate = _mapping(policy.get("benchmark_gate"), label="benchmark_gate")
    expected: dict[str, object] = {
        "candidate_and_control_same_machine": True,
        "comparison_baseline": FROZEN_BASELINE,
        "maximum_candidate_to_reference_ratio": FROZEN_RATIO,
        "minimum_repetitions_per_scenario": FROZEN_REPETITIONS,
        "pageouts_and_swapouts_must_be_zero": True,
        "required_scenarios": list(FROZEN_SCENARIOS),
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(
                f"frozen benchmark_gate {key} must be {value!r}, found {gate.get(key)!r}"
            )
    return GLM52BenchmarkContract(
        policy_contract_sha256=_sha256(
            policy.get("policy_contract_sha256"), label="policy_contract_sha256"
        ),
        required_scenarios=FROZEN_SCENARIOS,
        minimum_repetitions_per_scenario=FROZEN_REPETITIONS,
        maximum_candidate_to_reference_ratio=FROZEN_RATIO,
        comparison_baseline=FROZEN_BASELINE,
    )


def capture_machine_identity(
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Capture the stable physical-host fields required by paired evidence."""

    if platform.system() != "Darwin":
        raise RuntimeError("GLM52 same-machine evidence must be captured on macOS")

    def command(argv: list[str]) -> str:
        result = command_runner(argv, check=True, capture_output=True, text=True, timeout=5)
        value = result.stdout.strip()
        if not value:
            raise RuntimeError(f"machine identity command returned no output: {argv!r}")
        return value

    try:
        platform_expert = command(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
        platform_uuid_match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', platform_expert)
        if platform_uuid_match is None:
            raise ValueError("IOPlatformUUID is absent from IOPlatformExpertDevice")
        platform_uuid = platform_uuid_match.group(1)
        model = command(["sysctl", "-n", "hw.model"])
        memsize = int(command(["sysctl", "-n", "hw.memsize"]))
        os_build = command(["sw_vers", "-buildVersion"])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"could not capture authenticated machine identity: {error}") from error
    if memsize <= 0:
        raise RuntimeError("hw.memsize must be positive")
    return {
        "io_platform_uuid": platform_uuid,
        "hw.model": model,
        "hw.memsize": memsize,
        "os.build": os_build,
    }


def seal_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("evidence_sha256", None)
    return {**body, "evidence_sha256": canonical_sha256(body)}


def verify_evidence_seal(payload: Mapping[str, Any]) -> None:
    recorded = _sha256(payload.get("evidence_sha256"), label="evidence_sha256")
    body = dict(payload)
    body.pop("evidence_sha256", None)
    if canonical_sha256(body) != recorded:
        raise ValueError("measurement evidence SHA-256 does not match its payload")


def write_sealed_evidence(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_evidence(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sealed


def read_sealed_evidence(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read benchmark evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("benchmark evidence must be an object")
    verify_evidence_seal(payload)
    return payload


class FileBenchmarkLockProvider:
    """The shared KEEP heavy-job lock plus a regular-file measurement lock."""

    @contextmanager
    def heavy_job_lock(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def run_lock(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("benchmark run lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def measurement_locks(
    output_json: str | Path,
    *,
    lock_provider: FileBenchmarkLockProvider | None = None,
) -> Iterator[None]:
    provider = lock_provider or FileBenchmarkLockProvider()
    run_lock = Path(f"{Path(output_json).resolve()}.lock")
    with provider.heavy_job_lock(HEAVY_JOB_LOCK_PATH):
        with provider.run_lock(run_lock):
            yield


def _valid_latency(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_session(session: Mapping[str, Any]) -> None:
    nonce = session.get("session_uuid")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", nonce):
        raise ValueError("benchmark session UUID is invalid")
    starts_ends = (
        ("monotonic_start_ns", "monotonic_end_ns"),
        ("wall_start_ns", "wall_end_ns"),
    )
    durations: list[int] = []
    for start_key, end_key in starts_ends:
        start = _positive_int(session.get(start_key), label=f"benchmark_session.{start_key}")
        end = _positive_int(session.get(end_key), label=f"benchmark_session.{end_key}")
        if end < start:
            raise ValueError("benchmark session time window is inverted")
        duration = end - start
        if duration > MAX_SESSION_DURATION_NS:
            raise ValueError("benchmark session time window exceeds the frozen bound")
        durations.append(duration)
    if abs(durations[0] - durations[1]) > MAX_CLOCK_SKEW_NS:
        raise ValueError("benchmark session monotonic and wall durations disagree")


def _validate_machine_identity(machine: Mapping[str, Any]) -> None:
    required = {"io_platform_uuid", "hw.model", "hw.memsize", "os.build"}
    if set(machine) != required:
        raise ValueError("machine identity must contain only the stable physical-host fields")
    if not isinstance(machine.get("io_platform_uuid"), str) or not machine["io_platform_uuid"]:
        raise ValueError("machine identity IOPlatformUUID is invalid")
    if not isinstance(machine.get("hw.model"), str) or not machine["hw.model"]:
        raise ValueError("machine identity hw.model is invalid")
    _positive_int(machine.get("hw.memsize"), label="machine_identity.hw.memsize")
    if not isinstance(machine.get("os.build"), str) or not machine["os.build"]:
        raise ValueError("machine identity os.build is invalid")


def _expected_tokens(scenario: str) -> tuple[int, int]:
    definition = SCENARIO_DEFINITIONS[scenario]
    return int(definition["prompt_token_count"]), int(definition.get("decode_token_count", 0))


def _validate_counter_pair(row: Mapping[str, Any], *, label: str) -> tuple[int, int]:
    before = _mapping(row.get("vm_stat_before"), label=f"{label}.vm_stat_before")
    after = _mapping(row.get("vm_stat_after"), label=f"{label}.vm_stat_after")
    values: list[int] = []
    for counter in ("pageouts", "swapouts"):
        before_value = _nonnegative_int(before.get(counter), label=f"{label}.vm_stat_before.{counter}")
        after_value = _nonnegative_int(after.get(counter), label=f"{label}.vm_stat_after.{counter}")
        if after_value < before_value:
            raise ValueError(f"{label} {counter} counter is not monotonic")
        delta = after_value - before_value
        if row.get(f"{counter}_delta") != delta:
            raise ValueError(f"{label} {counter}_delta does not match raw counters")
        values.append(delta)
    return values[0], values[1]


def _validate_record(
    row: Mapping[str, Any],
    *,
    role: str,
    scenario: str,
    session: Mapping[str, Any],
    record_kind: str,
    label: str,
) -> tuple[int, int]:
    if row.get("record_kind") != record_kind:
        if record_kind == "repetition" and row.get("record_kind") == "warmup":
            raise ValueError("warmup record must not be counted as a repetition")
        raise ValueError(f"{label} record_kind is invalid")
    if row.get("role") != role or row.get("scenario") != scenario:
        raise ValueError(f"{label} role or scenario is invalid")
    if _valid_latency(row.get("latency_ms")) is None:
        raise ValueError(f"{label} latency is not positive and finite")
    input_tokens, output_tokens = _expected_tokens(scenario)
    if row.get("input_token_count") != input_tokens or row.get("output_token_count") != output_tokens:
        raise ValueError(f"{label} token counts do not match the frozen scenario")
    monotonic_start = _positive_int(row.get("monotonic_start_ns"), label=f"{label}.monotonic_start_ns")
    monotonic_end = _positive_int(row.get("monotonic_end_ns"), label=f"{label}.monotonic_end_ns")
    wall_start = _positive_int(row.get("wall_start_ns"), label=f"{label}.wall_start_ns")
    wall_end = _positive_int(row.get("wall_end_ns"), label=f"{label}.wall_end_ns")
    if monotonic_end < monotonic_start or wall_end < wall_start:
        raise ValueError(f"{label} timestamp window is inverted")
    if not (
        session["monotonic_start_ns"] <= monotonic_start <= monotonic_end <= session["monotonic_end_ns"]
        and session["wall_start_ns"] <= wall_start <= wall_end <= session["wall_end_ns"]
    ):
        raise ValueError(f"{label} timestamp is outside the paired session window")
    if abs((monotonic_end - monotonic_start) - (wall_end - wall_start)) > MAX_CLOCK_SKEW_NS:
        raise ValueError(f"{label} monotonic and wall durations disagree")
    if type(row.get("valid")) is not bool:
        raise ValueError(f"{label}.valid must be a boolean")
    return _validate_counter_pair(row, label=label)


def _valid_repetition(row: Mapping[str, Any], *, role: str, scenario: str) -> tuple[bool, str | None]:
    if row.get("role") != role or row.get("scenario") != scenario:
        return False, "role_or_scenario_mismatch"
    if row.get("record_kind") != "repetition":
        return False, "warmup_counted_as_repetition"
    if row.get("valid") is not True:
        return False, "runner_marked_invalid"
    if _valid_latency(row.get("latency_ms")) is None:
        return False, "latency_not_positive_finite"
    if row.get("pageouts_delta") != 0 or row.get("swapouts_delta") != 0:
        return False, "memory_not_clean"
    return True, None


def _validate_measurement_evidence(
    evidence: Mapping[str, Any],
    *,
    role: str,
    contract: GLM52BenchmarkContract,
) -> None:
    if evidence.get("record_type") != RECORD_TYPE:
        raise ValueError("measurement record_type is invalid")
    if evidence.get("role") != role:
        raise ValueError(f"measurement role must be {role!r}")
    if evidence.get("policy_contract_sha256") != contract.policy_contract_sha256:
        raise ValueError("measurement policy identity does not match the frozen policy")
    if evidence.get("schema_version") != 2:
        raise ValueError("measurement evidence schema_version is invalid")
    if evidence.get("evidence_provenance") != EVIDENCE_PROVENANCE:
        raise ValueError("measurement evidence provenance is invalid")
    machine = _mapping(evidence.get("machine_identity"), label="machine_identity")
    _validate_machine_identity(machine)
    session = _mapping(evidence.get("benchmark_session"), label="benchmark_session")
    _validate_session(session)
    identity_key = "candidate_composite_identity" if role == "candidate" else "control_source_identity"
    identity = _mapping(evidence.get(identity_key), label=identity_key)
    _sha256(identity.get("sha256"), label=f"{identity_key}.sha256")
    if evidence.get("scenario_definitions") != SCENARIO_DEFINITIONS:
        raise ValueError("measurement scenario definitions drift from the frozen contract")
    benchmark_input = _mapping(
        evidence.get("benchmark_input_identity"), label="benchmark_input_identity"
    )
    if benchmark_input.get("token_count") != 1024:
        raise ValueError("benchmark input must identify an exact 1024-token prompt")
    _sha256(benchmark_input.get("token_ids_sha256"), label="benchmark_input_identity.token_ids_sha256")
    warmups = evidence.get("warmups")
    if not isinstance(warmups, list) or len(warmups) != len(contract.required_scenarios):
        raise ValueError("each scenario requires one separately recorded warmup")
    for index, scenario in enumerate(contract.required_scenarios):
        row = warmups[index]
        if not isinstance(row, Mapping):
            raise ValueError("warmup record must be an object")
        if row.get("warmup_sequence_number") != index or row.get("excluded_from_gate") is not True:
            raise ValueError("warmup sequence or exclusion marker is invalid")
        _validate_record(row, role=role, scenario=scenario, session=session, record_kind="warmup", label=f"{role}.{scenario}.warmup")
    repetitions = evidence.get("repetitions")
    if not isinstance(repetitions, list):
        raise ValueError("measurement repetitions must be a list")
    expected_count = len(contract.required_scenarios) * contract.minimum_repetitions_per_scenario
    if len(repetitions) != expected_count:
        raise ValueError("measurement must contain exactly the planned repetitions")
    for sequence_number, row in enumerate(repetitions):
        if not isinstance(row, Mapping):
            raise ValueError("repetition record must be an object")
        if row.get("sequence_number") != sequence_number:
            raise ValueError("repetition sequence numbers must be contiguous from zero")
        scenario_index, run_index = divmod(sequence_number, contract.minimum_repetitions_per_scenario)
        scenario = contract.required_scenarios[scenario_index]
        if row.get("run_index") != run_index:
            raise ValueError("repetition run indexes must match the planned sequence")
        _validate_record(row, role=role, scenario=scenario, session=session, record_kind="repetition", label=f"{role}.{scenario}.repetition[{run_index}]")


def build_benchmark_session_manifest(
    *,
    benchmark_session: Mapping[str, Any],
    machine_identity: Mapping[str, Any],
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the two role records emitted by one local paired invocation."""

    _validate_session(benchmark_session)
    _validate_machine_identity(machine_identity)
    candidate_digest = _sha256(candidate.get("evidence_sha256"), label="candidate evidence_sha256")
    control_digest = _sha256(control.get("evidence_sha256"), label="control evidence_sha256")
    return seal_evidence(
        {
            "schema_version": 1,
            "record_type": "glm52_same_machine_fp4_paired_session",
            "evidence_provenance": EVIDENCE_PROVENANCE,
            "machine_identity": dict(machine_identity),
            "benchmark_session": dict(benchmark_session),
            "candidate_measurement_evidence_sha256": candidate_digest,
            "control_measurement_evidence_sha256": control_digest,
        }
    )


def _validate_session_manifest(
    manifest: Mapping[str, Any], *, candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> None:
    verify_evidence_seal(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("record_type") != "glm52_same_machine_fp4_paired_session":
        raise ValueError("paired session manifest is invalid")
    if manifest.get("evidence_provenance") != EVIDENCE_PROVENANCE:
        raise ValueError("paired session manifest provenance is invalid")
    machine = _mapping(manifest.get("machine_identity"), label="paired session machine_identity")
    session = _mapping(manifest.get("benchmark_session"), label="paired session benchmark_session")
    _validate_machine_identity(machine)
    _validate_session(session)
    if dict(machine) != dict(candidate["machine_identity"]) or dict(machine) != dict(control["machine_identity"]):
        raise ValueError("paired session manifest machine identity does not bind both roles")
    if dict(session) != dict(candidate["benchmark_session"]) or dict(session) != dict(control["benchmark_session"]):
        raise ValueError("paired session manifest nonce or time window does not bind both roles")
    if manifest.get("candidate_measurement_evidence_sha256") != candidate.get("evidence_sha256"):
        raise ValueError("paired session manifest does not bind candidate evidence")
    if manifest.get("control_measurement_evidence_sha256") != control.get("evidence_sha256"):
        raise ValueError("paired session manifest does not bind control evidence")


def evaluate_glm52_same_machine_benchmark(
    *,
    policy: Mapping[str, Any],
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    session_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the frozen gate; no caller can tune its thresholds."""

    contract = load_frozen_glm52_benchmark_contract(policy)
    _validate_measurement_evidence(candidate, role="candidate", contract=contract)
    _validate_measurement_evidence(control, role="control", contract=contract)
    if candidate.get("machine_identity") != control.get("machine_identity"):
        raise ValueError("candidate and control machine identity must match exactly")
    if candidate.get("benchmark_session") != control.get("benchmark_session"):
        raise ValueError("candidate and control benchmark session must match exactly")
    if candidate.get("benchmark_input_identity") != control.get("benchmark_input_identity"):
        raise ValueError("candidate and control benchmark input identity must match exactly")
    if candidate["candidate_composite_identity"].get("sha256") != EXPECTED_CANDIDATE_COMPOSITE_IDENTITY_SHA256:
        raise ValueError("candidate composite identity does not match the frozen release identity")
    _validate_session_manifest(session_manifest, candidate=candidate, control=control)

    clean: dict[str, dict[str, list[float]]] = {
        scenario: {"candidate": [], "control": []} for scenario in contract.required_scenarios
    }
    invalid_rows: list[dict[str, object]] = []
    for role, evidence in (("candidate", candidate), ("control", control)):
        rows = evidence["repetitions"]
        assert isinstance(rows, list)
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                invalid_rows.append({"role": role, "row_index": row_index, "error": "row_not_object"})
                continue
            scenario = row.get("scenario")
            run_index = row.get("run_index")
            assert isinstance(scenario, str) and type(run_index) is int
            valid, error = _valid_repetition(row, role=role, scenario=str(scenario))
            if not valid:
                invalid_rows.append({"role": role, "scenario": scenario, "run_index": run_index, "error": error})
                continue
            latency = _valid_latency(row.get("latency_ms"))
            assert latency is not None
            clean[str(scenario)][role].append(latency)

    scenario_counts = {
        scenario: {role: len(clean[scenario][role]) for role in ("candidate", "control")}
        for scenario in contract.required_scenarios
    }
    missing: list[str] = []
    for scenario in contract.required_scenarios:
        for role in ("candidate", "control"):
            if scenario_counts[scenario][role] != contract.minimum_repetitions_per_scenario:
                missing.append(
                    f"{role}_{scenario}_requires_{contract.minimum_repetitions_per_scenario}_clean_repetitions"
                )
    if invalid_rows:
        missing.append("benchmark_repetitions_require_zero_pageouts_and_swapouts")

    ratios: dict[str, float | None] = {}
    ratio_errors: list[dict[str, object]] = []
    for scenario in contract.required_scenarios:
        candidate_rows = clean[scenario]["candidate"]
        control_rows = clean[scenario]["control"]
        if len(candidate_rows) != FROZEN_REPETITIONS or len(control_rows) != FROZEN_REPETITIONS:
            ratios[scenario] = None
            continue
        ratio = median(candidate_rows) / median(control_rows)
        ratios[scenario] = ratio
        if ratio > contract.maximum_candidate_to_reference_ratio:
            ratio_errors.append({"scenario": scenario, "candidate_to_reference_ratio": ratio})
    if ratio_errors:
        missing.append("candidate_to_reference_ratio_at_most_1_15")

    return {
        "schema_version": 2,
        "record_type": EVALUATION_RECORD_TYPE,
        "evidence_provenance": EVIDENCE_PROVENANCE,
        "policy_contract_sha256": contract.policy_contract_sha256,
        "comparison_baseline": contract.comparison_baseline,
        "machine_identity": dict(candidate["machine_identity"]),
        "benchmark_session": dict(candidate["benchmark_session"]),
        "paired_session_manifest": dict(session_manifest),
        "benchmark_input_identity": dict(candidate["benchmark_input_identity"]),
        "candidate_composite_identity": dict(candidate["candidate_composite_identity"]),
        "control_source_identity": dict(control["control_source_identity"]),
        "control_implementation": dict(CONTROL_IMPLEMENTATION),
        "required_scenarios": list(contract.required_scenarios),
        "minimum_repetitions_per_scenario": contract.minimum_repetitions_per_scenario,
        "maximum_candidate_to_reference_ratio": contract.maximum_candidate_to_reference_ratio,
        "ratio_statistic_policy_key": FROZEN_RATIO_POLICY_KEY,
        "ratio_statistic": FROZEN_RATIO_STATISTIC,
        "scenario_counts": scenario_counts,
        "scenario_latency_ratios": ratios,
        "invalid_repetition_count": len(invalid_rows),
        "invalid_repetitions": invalid_rows,
        "ratio_errors": ratio_errors,
        "frozen_gate_pass": not missing,
        "missing_requirements": missing,
        "raw_measurements": {"candidate": dict(candidate), "control": dict(control)},
    }


def load_glm52_same_machine_benchmark_evidence(
    path: str | Path,
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a release gate by recomputing every aggregate from embedded raw records.

    The trust boundary is deliberately local-filesystem hash-referenced evidence:
    hashes bind the submitted files and records, while this loader derives all
    release quantities instead of accepting an authored summary.
    """

    evidence = read_sealed_evidence(path)
    if evidence.get("schema_version") != 2 or evidence.get("record_type") != EVALUATION_RECORD_TYPE:
        raise ValueError("benchmark evidence schema or record type is invalid")
    if evidence.get("evidence_provenance") != EVIDENCE_PROVENANCE:
        raise ValueError("benchmark evidence provenance is not local hash-referenced evidence")
    raw_measurements = _mapping(evidence.get("raw_measurements"), label="raw measurement records")
    if set(raw_measurements) != {"candidate", "control"}:
        raise ValueError("raw measurement records must contain candidate and control")
    candidate = _mapping(raw_measurements["candidate"], label="raw candidate measurement")
    control = _mapping(raw_measurements["control"], label="raw control measurement")
    verify_evidence_seal(candidate)
    verify_evidence_seal(control)
    manifest = _mapping(evidence.get("paired_session_manifest"), label="paired_session_manifest")
    recomputed = evaluate_glm52_same_machine_benchmark(
        policy=policy,
        candidate=candidate,
        control=control,
        session_manifest=manifest,
    )
    stored_body = dict(evidence)
    stored_body.pop("evidence_sha256", None)
    if stored_body != recomputed:
        raise ValueError("stored aggregate does not match raw recomputation")
    if any(
        isinstance(row, Mapping) and row.get("error") == "memory_not_clean"
        for row in recomputed["invalid_repetitions"]
    ):
        raise ValueError("benchmark evidence contains dirty memory repetitions")
    if recomputed["invalid_repetitions"]:
        raise ValueError("benchmark evidence contains invalid repetitions")
    if recomputed["ratio_errors"]:
        raise ValueError("benchmark evidence ratio does not pass the frozen policy")
    if recomputed["missing_requirements"] or recomputed["frozen_gate_pass"] is not True:
        raise ValueError("benchmark evidence frozen gate does not pass")
    return evidence


def build_measurement_evidence(
    *,
    role: str,
    contract: GLM52BenchmarkContract,
    machine_identity: Mapping[str, Any],
    benchmark_input_identity: Mapping[str, Any],
    identity: Mapping[str, Any],
    benchmark_session: Mapping[str, Any],
    warmups: Sequence[Mapping[str, Any]],
    repetitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if role not in ("candidate", "control"):
        raise ValueError("role must be candidate or control")
    identity_key = "candidate_composite_identity" if role == "candidate" else "control_source_identity"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "record_type": RECORD_TYPE,
        "evidence_provenance": EVIDENCE_PROVENANCE,
        "role": role,
        "policy_contract_sha256": contract.policy_contract_sha256,
        "machine_identity": dict(machine_identity),
        "benchmark_session": dict(benchmark_session),
        "benchmark_input_identity": dict(benchmark_input_identity),
        identity_key: dict(identity),
        "scenario_definitions": SCENARIO_DEFINITIONS,
        "warmups": [dict(row) for row in warmups],
        "repetitions": [dict(row) for row in repetitions],
    }
    if role == "control":
        payload["control_implementation"] = dict(CONTROL_IMPLEMENTATION)
    _validate_measurement_evidence(payload, role=role, contract=contract)
    return seal_evidence(payload)
