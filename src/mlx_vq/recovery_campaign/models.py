"""Immutable domain values for GLM-5.2 recovery campaigns."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHELL_EXECUTABLES = frozenset(
    {"ash", "bash", "csh", "dash", "fish", "ksh", "sh", "tcsh", "zsh"}
)
_DIRECT_EXECUTABLES = frozenset({"pytest", "keep", "uv", "git", "rg", "node"})
_PYTHON_EXECUTABLE_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASCII_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_ADVANCE_ACTIONS = frozenset(
    {
        "wait",
        "resume",
        "audit",
        "reevaluate",
        "attribute",
        "ready",
        "blocked",
        "complete",
    }
)


class FrozenJsonList(Sequence[object]):
    """Tuple-backed immutable JSON array with list-compatible equality."""

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[object]) -> None:
        object.__setattr__(
            self, "_values", tuple(_freeze_json(item) for item in values)
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("ledger JSON arrays are immutable")

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            return FrozenJsonList(self._values[index])
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[object]:
        return iter(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(
            other, (str, bytes, bytearray)
        ):
            return tuple(self) == tuple(other)
        return False

    def __repr__(self) -> str:
        return repr(list(self._values))

    def append(self, _value: object) -> None:
        raise TypeError("ledger JSON arrays are immutable")


class FrozenJsonDict(Mapping[str, object]):
    """Tuple-backed immutable JSON object with dict-compatible equality."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, object]) -> None:
        items: list[tuple[str, object]] = []
        for key, value in values.items():
            items.append((key, _freeze_json(value)))
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("ledger JSON objects are immutable")

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        try:
            return all(self[key] == other[key] for key in self)
        except KeyError:
            return False

    def __repr__(self) -> str:
        return repr(dict(self._items))


def _freeze_json(value: object) -> object:
    if isinstance(value, FrozenJsonDict | FrozenJsonList):
        return value
    if isinstance(value, Mapping):
        return FrozenJsonDict(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return FrozenJsonList(value)
    return value


def _require_safe_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    if _ASCII_CONTROL_RE.search(value) is not None:
        raise ValueError(f"{label} must not contain ASCII control characters")
    return value


def _require_sha256(value: object, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _require_argv(value: object, label: str = "argv") -> None:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{label} must be a nonempty argv tuple")
    for item in value:
        _require_safe_text(item, label)
    command_index = 0
    executable = os.path.basename(value[command_index])
    while executable == "env":
        command_index = _env_command_index(value, label, command_index)
        executable = os.path.basename(value[command_index])
    if executable in _SHELL_EXECUTABLES:
        raise ValueError(f"{label} must not invoke a shell")
    if (
        executable not in _DIRECT_EXECUTABLES
        and _PYTHON_EXECUTABLE_RE.fullmatch(executable) is None
    ):
        raise ValueError(f"{label} executable is not in the supported allowlist")


def _env_command_index(
    value: tuple[str, ...], label: str, wrapper_index: int
) -> int:
    index = wrapper_index + 1
    while index < len(value):
        item = value[index]
        if item == "--":
            index += 1
            break
        if item in {"-i", "--ignore-environment"}:
            index += 1
            continue
        if item in {"-u", "--unset"}:
            if index + 1 >= len(value) or _ENV_NAME_RE.fullmatch(value[index + 1]) is None:
                raise ValueError(f"{label} env --unset requires a safe variable name")
            index += 2
            continue
        if item.startswith("--unset="):
            name = item.partition("=")[2]
            if _ENV_NAME_RE.fullmatch(name) is None:
                raise ValueError(f"{label} env --unset requires a safe variable name")
            index += 1
            continue
        if item.startswith("-"):
            raise ValueError(f"{label} uses an unsupported env option")
        if "=" in item:
            name, _separator, _assigned = item.partition("=")
            if _ENV_NAME_RE.fullmatch(name) is None:
                raise ValueError(f"{label} contains an invalid env assignment")
            index += 1
            continue
        break
    if index >= len(value):
        raise ValueError(f"{label} env wrapper must name a direct command")
    return index


def _require_safe_artifact_path(value: object) -> None:
    value = _require_safe_text(value, "artifact_path")
    if "\\" in value or value.startswith("/"):
        raise ValueError("artifact_path must be a safe repo-relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact_path must be a safe repo-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise ValueError("artifact_path must be a safe repo-relative POSIX path")


@dataclass(frozen=True, slots=True)
class Authority:
    name: str
    path: str
    sha256: str
    role: str
    split: str | None


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    split: str
    prompt_count: int
    position_count: int
    holdout_used_for_tuning: bool
    report_used_for_tuning: bool


@dataclass(frozen=True, slots=True)
class PayloadBudget:
    accepted_payload_bytes: int
    e8p_increment_per_projection_bytes: int
    payload_limit_bytes: int


@dataclass(frozen=True, slots=True)
class ProjectionRates:
    layer: int
    gate: int
    up: int
    down: int

    @property
    def values(self) -> tuple[int, int, int]:
        return (self.gate, self.up, self.down)


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    kind: str
    depends_on: tuple[str, ...]
    tuning_authorities: tuple[str, ...]
    output_path: str
    recovered_layers: tuple[int, ...]
    projection_rates: tuple[ProjectionRates, ...]
    recovery_levers: tuple[str, ...]
    expected_payload_bytes: int
    transition_name: str | None


@dataclass(frozen=True, slots=True)
class Transition:
    name: str
    experiment_name: str
    action: str
    heavy: bool
    argv: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    reason: str = ""
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise TypeError("transition exit_code must be an exact integer or None")


@dataclass(frozen=True, slots=True)
class LauncherRecord:
    """Detached-process identity recorded by the campaign controller."""

    pid: int
    command_sha256: str
    started_at: str
    stdout_path: str
    stderr_path: str
    experiment_name: str
    transition_name: str
    supervisor_pid: int
    launch_token: str
    terminal_path: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise TypeError("launcher pid must be a positive integer")
        if type(self.supervisor_pid) is not int or self.supervisor_pid <= 0:
            raise TypeError("launcher supervisor_pid must be a positive integer")
        _require_sha256(self.command_sha256, "launcher command_sha256")
        _require_sha256(self.launch_token, "launcher launch_token")
        if (
            not isinstance(self.started_at, str)
            or _RFC3339_UTC_RE.fullmatch(self.started_at) is None
        ):
            raise ValueError("launcher started_at must be canonical RFC3339 UTC with Z")
        try:
            parsed = datetime.fromisoformat(self.started_at[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError(
                "launcher started_at must be a valid UTC instant"
            ) from error
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("launcher started_at must be UTC")
        _require_safe_artifact_path(self.stdout_path)
        _require_safe_artifact_path(self.stderr_path)
        _require_safe_artifact_path(self.terminal_path)
        _require_safe_text(self.experiment_name, "launcher experiment_name")
        _require_safe_text(self.transition_name, "launcher transition_name")


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    """One deterministic advance decision and optional launched process."""

    action: str
    experiment_name: str
    transition_name: str | None
    reason: str
    argv: tuple[str, ...]
    heavy: bool
    dry_run: bool
    launcher_record: LauncherRecord | None
    exit_code: int = 0

    def __post_init__(self) -> None:
        if self.action not in _ADVANCE_ACTIONS:
            raise ValueError(f"invalid advance action {self.action!r}")
        _require_safe_text(self.experiment_name, "advance experiment_name")
        _require_safe_text(self.reason, "advance reason")
        if self.transition_name is not None:
            _require_safe_text(self.transition_name, "advance transition_name")
        if not isinstance(self.argv, tuple) or any(
            not isinstance(item, str) or not item for item in self.argv
        ):
            raise TypeError("advance argv must be a tuple of nonempty strings")
        if not isinstance(self.heavy, bool) or not isinstance(self.dry_run, bool):
            raise TypeError("advance heavy and dry_run must be booleans")
        if type(self.exit_code) is not int or self.exit_code < 0:
            raise TypeError("advance exit_code must be a nonnegative integer")
        if self.action == "blocked" and self.exit_code == 0:
            raise ValueError("blocked advance results require a nonzero exit_code")
        if self.action in {
            "audit",
            "reevaluate",
            "attribute",
            "ready",
            "complete",
        } and (
            self.transition_name is not None
            or self.argv
            or self.heavy
            or self.launcher_record is not None
            or self.exit_code != 0
        ):
            raise ValueError(
                f"{self.action} is a non-launching successful controller state"
            )
        if self.action in {"wait", "resume"} and (
            self.transition_name is None or not self.argv
        ):
            raise ValueError("wait/resume results require a transition and exact argv")
        if self.launcher_record is not None:
            if self.dry_run or self.action not in {"wait", "resume"}:
                raise ValueError(
                    "launcher records are valid only for mutating wait/resume"
                )
            if (
                self.launcher_record.experiment_name != self.experiment_name
                or self.launcher_record.transition_name != self.transition_name
            ):
                raise ValueError(
                    "launcher record identity does not match advance result"
                )
        if (
            self.action == "resume"
            and not self.dry_run
            and self.launcher_record is None
        ):
            raise ValueError("mutating resume result requires a launcher record")


@dataclass(frozen=True, slots=True)
class ProcessObservation:
    pid: int
    command: tuple[str, ...]
    elapsed_seconds: float | None
    experiment_name: str | None = None
    transition_name: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentObservation:
    """Authenticated physical snapshot for one declared experiment."""

    experiment_name: str
    recovered_groups: int
    expected_groups: int
    recovered_groups_state: str
    artifact_links: int
    expected_artifact_links: int
    artifact_links_state: str
    manifest_state: str
    manifest_file_sha256: str | None
    manifest_body_sha256: str | None
    contradictions: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_safe_text(self.experiment_name, "experiment observation name")
        for field_name in (
            "recovered_groups",
            "expected_groups",
            "artifact_links",
            "expected_artifact_links",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{field_name} must be a nonnegative integer")
        if self.expected_groups <= 0 or self.expected_artifact_links <= 0:
            raise ValueError("physical expected counts must be positive")
        if self.recovered_groups_state not in {"absent", "present", "invalid"}:
            raise ValueError("invalid recovered_groups_state")
        if self.artifact_links_state not in {"absent", "present", "invalid"}:
            raise ValueError("invalid artifact_links_state")
        if self.manifest_state not in {"absent", "invalid", "valid"}:
            raise ValueError("invalid manifest_state")
        _require_sha256(
            self.manifest_file_sha256,
            "manifest_file_sha256",
            optional=True,
        )
        _require_sha256(
            self.manifest_body_sha256,
            "manifest_body_sha256",
            optional=True,
        )
        if self.manifest_state == "valid" and (
            self.manifest_file_sha256 is None
            or self.manifest_body_sha256 is None
        ):
            raise ValueError("valid manifest snapshot requires file and body hashes")
        if self.manifest_state == "absent" and (
            self.manifest_file_sha256 is not None
            or self.manifest_body_sha256 is not None
        ):
            raise ValueError("absent manifest snapshot cannot carry hashes")
        if not isinstance(self.contradictions, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.contradictions
        ):
            raise TypeError("snapshot contradictions must be nonempty string tuples")

    @property
    def physically_complete(self) -> bool:
        return (
            self.manifest_state == "valid"
            and self.manifest_file_sha256 is not None
            and self.manifest_body_sha256 is not None
            and self.recovered_groups_state == "present"
            and self.recovered_groups == self.expected_groups
            and self.artifact_links_state == "present"
            and self.artifact_links == self.expected_artifact_links
            and not self.contradictions
        )


@dataclass(frozen=True, slots=True)
class CampaignObservation:
    lock_held: bool
    lock_owner_known: bool
    active_processes: tuple[ProcessObservation, ...]
    recovered_groups: int
    expected_groups: int
    manifest_sha256: str | None
    contradictions: tuple[str, ...]
    campaign: str = ""
    experiment_name: str | None = None
    lock_state: str = "absent"
    artifact_links: int = 0
    expected_artifact_links: int = 225
    manifest_state: str = "absent"
    manifest_file_sha256: str | None = None
    manifest_body_sha256: str | None = None
    throughput_groups_per_second: float | None = None
    eta_seconds: float | None = None
    recovered_groups_state: str = "absent"
    artifact_links_state: str = "absent"
    experiment_observations: tuple[ExperimentObservation, ...] = ()

    def snapshot(self, experiment_name: str) -> ExperimentObservation:
        for observation in self.experiment_observations:
            if observation.experiment_name == experiment_name:
                return observation
        raise KeyError(experiment_name)


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    schema_version: int
    sequence: int
    timestamp: str
    event_kind: str
    experiment: str
    payload: FrozenJsonDict
    previous_event_sha256: str | None
    event_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("LedgerEvent payload must be a JSON object")
        object.__setattr__(self, "payload", FrozenJsonDict(self.payload))


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Exact command result attached to one evidence registration."""

    name: str
    argv: tuple[str, ...]
    exit_code: int
    stdout_sha256: str
    stderr_sha256: str

    def __post_init__(self) -> None:
        _require_safe_text(self.name, "verification result name")
        _require_argv(self.argv, "verification result argv")
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("verification result exit_code must be an integer")
        _require_sha256(self.stdout_sha256, "verification result stdout_sha256")
        _require_sha256(self.stderr_sha256, "verification result stderr_sha256")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Immutable identity record; the recorded content hash is authoritative."""

    artifact_path: str
    content_sha256: str
    evidence_class: str
    release_eligible: bool
    recovery_levers: tuple[str, ...]
    argv: tuple[str, ...]
    verification_results: tuple[VerificationResult, ...]
    manifest_identity_sha256: str | None = None
    candidate_identity_sha256: str | None = None
    parent_identity_sha256: str | None = None
    baseline_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_safe_artifact_path(self.artifact_path)
        _require_sha256(self.content_sha256, "content_sha256")
        for field_name in (
            "manifest_identity_sha256",
            "candidate_identity_sha256",
            "parent_identity_sha256",
            "baseline_identity_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name, optional=True)
        if self.evidence_class not in {"diagnostic_only", "release"}:
            raise ValueError(
                "evidence_class must be 'diagnostic_only' or 'release'"
            )
        if not isinstance(self.release_eligible, bool):
            raise TypeError("release_eligible must be a boolean")
        if self.evidence_class == "diagnostic_only" and self.release_eligible:
            raise ValueError(
                "diagnostic_only evidence must have release_eligible=false"
            )
        if not isinstance(self.recovery_levers, tuple) or not self.recovery_levers:
            raise TypeError("recovery_levers must be a nonempty tuple")
        if any(
            not isinstance(lever, str) or not lever.strip()
            for lever in self.recovery_levers
        ):
            raise ValueError("recovery_levers must contain nonempty strings")
        for lever in self.recovery_levers:
            _require_safe_text(lever, "recovery lever")
        _require_argv(self.argv)
        if (
            not isinstance(self.verification_results, tuple)
            or not self.verification_results
        ):
            raise TypeError("verification_results must be a nonempty tuple")
        if any(
            not isinstance(result, VerificationResult)
            for result in self.verification_results
        ):
            raise TypeError(
                "verification_results must contain VerificationResult records"
            )
        candidate = self.candidate_identity_sha256
        for label, related in (
            ("parent", self.parent_identity_sha256),
            ("baseline", self.baseline_identity_sha256),
        ):
            if candidate is not None and related is not None and candidate == related:
                raise ValueError(
                    f"candidate identity must be distinct from {label} identity"
                )
        if self.release_eligible:
            if self.evidence_class != "release":
                raise ValueError(
                    "release_eligible evidence must use evidence_class='release'"
                )
            if (
                self.candidate_identity_sha256 is None
                or self.manifest_identity_sha256 is None
                or self.baseline_identity_sha256 is None
            ):
                raise ValueError(
                    "release_eligible evidence requires candidate, manifest, and "
                    "baseline identities"
                )
            if any(result.exit_code != 0 for result in self.verification_results):
                raise ValueError(
                    "release_eligible evidence requires every verification exit_code=0"
                )


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    schema_version: int
    campaign: str
    model_id: str
    model_revision: str
    source_dir: str
    index_path: str
    config_sha256: str
    index_sha256: str
    full_source_blob_inventory_sha256: str
    routed_source_blob_inventory_sha256: str
    tuning: SelectionPolicy
    budget: PayloadBudget
    authorities: tuple[Authority, ...]
    experiments: tuple[Experiment, ...]
    transitions: tuple[Transition, ...]
    campaign_config_sha256: str
    campaign_config_path: str

    def experiment(self, name: str) -> Experiment:
        for experiment in self.experiments:
            if experiment.name == name:
                return experiment
        raise KeyError(name)

    def transition(self, name: str) -> Transition:
        for transition in self.transitions:
            if transition.name == name:
                return transition
        raise KeyError(name)

    def canonical_payload_bytes(self, experiment: Experiment) -> int:
        projection_increments = sum(
            (bits // 8) - 1
            for rates in experiment.projection_rates
            for bits in rates.values
        )
        return (
            self.budget.accepted_payload_bytes
            + projection_increments
            * self.budget.e8p_increment_per_projection_bytes
        )


__all__ = [
    "Authority",
    "AdvanceResult",
    "CampaignConfig",
    "CampaignObservation",
    "EvidenceRecord",
    "Experiment",
    "ExperimentObservation",
    "LedgerEvent",
    "LauncherRecord",
    "PayloadBudget",
    "ProcessObservation",
    "ProjectionRates",
    "SelectionPolicy",
    "Transition",
    "VerificationResult",
]
