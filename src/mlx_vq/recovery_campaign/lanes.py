"""Pure companion-lane declarations, planning, and adapter validation."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Mapping, Protocol, Sequence


class LaneError(ValueError):
    """A lane declaration, plan, request, or result is unsafe or stale."""


_LANE_KEYS = {
    "schema_version",
    "name",
    "mode",
    "owned_paths",
    "protected_paths",
    "out_of_scope_paths",
    "frozen_paths",
    "red_argv",
    "green_argv",
    "prerequisites",
    "heavy_lock_policy",
}
_MODES = {"read-only", "write"}
_PREREQUISITE_KINDS = {"verification", "review"}
_HEAVY_LOCK_POLICIES = {"forbidden", "wait-if-owned"}
_LOCK_STATES = {"absent", "free", "held", "unobserved"}
_RESULT_STATUSES = {"passed", "failed", "blocked"}
_PINNED_PYTEST_PREFIX = (".venv/bin/python", "-m", "pytest")
_SAFE_PYTEST_FLAGS = {"-q"}
_SHELL_PROGRAMS = {"bash", "dash", "env", "fish", "sh", "zsh"}
_SHELL_TOKENS = (";", "&", "|", "<", ">", "`", "$(", "\\")
_UTC_RFC3339_SECONDS = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(body).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LaneError(f"{label} must contain non-empty relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise LaneError(f"{label} contains unsafe path {value!r}")
    return value


def _contains(parent: str, child: str) -> bool:
    return parent == child or child.startswith(parent + "/")


def _overlaps(first: str, second: str) -> bool:
    return _contains(first, second) or _contains(second, first)


def _parse_paths(value: object, *, label: str, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LaneError(f"{label} must be a list of paths")
    paths = tuple(_safe_path(item, label=label) for item in value)
    if nonempty and not paths:
        raise LaneError(f"{label} must not be empty")
    if len(set(paths)) != len(paths):
        raise LaneError(f"{label} contains duplicate paths")
    return paths


def validate_direct_argv(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(argument, str) or not argument for argument in value
    ):
        raise LaneError(f"{label} must be a non-empty direct argv list")
    argv = tuple(value)
    executable_path = PurePosixPath(argv[0])
    executable = executable_path.name
    if (
        executable in _SHELL_PROGRAMS
        or executable_path.is_absolute()
        or any(part in {"", ".", ".."} for part in executable_path.parts)
        or any(
            character.isspace()
            or ord(character) < 32
            or ord(character) == 127
            for character in argv[0]
        )
    ):
        raise LaneError(f"{label} must be direct and shell-free")
    for argument in argv:
        if any(ord(character) < 32 or ord(character) == 127 for character in argument):
            raise LaneError(f"{label} contains control characters")
        if any(token in argument for token in _SHELL_TOKENS):
            raise LaneError(f"{label} must be direct and shell-free")
        candidate = argument.split("=", 1)[-1]
        if ".." in PurePosixPath(candidate).parts:
            raise LaneError(f"{label} contains path traversal")
    if argv[:3] != _PINNED_PYTEST_PREFIX:
        raise LaneError(
            f"{label} executable must be pinned `.venv/bin/python -m pytest`"
        )
    flags: set[str] = set()
    targets: list[str] = []
    target_seen = False
    for argument in argv[3:]:
        if argument.startswith("-"):
            if (
                target_seen
                or argument not in _SAFE_PYTEST_FLAGS
                or argument in flags
            ):
                raise LaneError(f"{label} contains an unauthorized pytest option")
            flags.add(argument)
            continue
        target_seen = True
        path_text, separator, node_text = argument.partition("::")
        path = PurePosixPath(path_text)
        if (
            path.is_absolute()
            or path.as_posix() != path_text
            or not path.parts
            or path.parts[0] != "tests"
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix != ".py"
        ):
            raise LaneError(f"{label} target must be a safe relative tests/**/*.py path")
        if separator and (
            not node_text
            or any(not part or "/" in part for part in node_text.split("::"))
        ):
            raise LaneError(f"{label} pytest node selector is invalid")
        targets.append(argument)
    if not targets:
        raise LaneError(f"{label} must declare at least one in-repo pytest target")
    return argv


@dataclass(frozen=True, slots=True)
class LanePrerequisite:
    kind: str
    sha256: str

    def canonical_body(self) -> dict[str, str]:
        return {"kind": self.kind, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class LaneDeclaration:
    schema_version: int
    name: str
    mode: str
    owned_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    out_of_scope_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]
    red_argv: tuple[str, ...]
    green_argv: tuple[str, ...]
    prerequisites: tuple[LanePrerequisite, ...]
    heavy_lock_policy: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise LaneError("lane declaration schema_version must be exact integer 1")
        for name in (
            "owned_paths",
            "protected_paths",
            "out_of_scope_paths",
            "frozen_paths",
            "red_argv",
            "green_argv",
            "prerequisites",
        ):
            if type(getattr(self, name)) is not tuple:
                raise LaneError(f"lane declaration {name} must be an immutable tuple")

    def canonical_body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "mode": self.mode,
            "owned_paths": list(self.owned_paths),
            "protected_paths": list(self.protected_paths),
            "out_of_scope_paths": list(self.out_of_scope_paths),
            "frozen_paths": list(self.frozen_paths),
            "red_argv": list(self.red_argv),
            "green_argv": list(self.green_argv),
            "prerequisites": [item.canonical_body() for item in self.prerequisites],
            "heavy_lock_policy": self.heavy_lock_policy,
        }


@dataclass(frozen=True, slots=True)
class LanePlan:
    lane: LaneDeclaration
    available_prerequisites: tuple[LanePrerequisite, ...]
    lock_state: str
    owner_known: bool
    status: str
    reason: str
    red_argv: tuple[str, ...]
    green_argv: tuple[str, ...]
    acquire_heavy_lock: bool
    stuck: bool
    dry_run: bool = True

    def __post_init__(self) -> None:
        for name in ("available_prerequisites", "red_argv", "green_argv"):
            if type(getattr(self, name)) is not tuple:
                raise LaneError(f"lane plan {name} must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class LaneResult:
    lane_name: str
    lane_sha256: str
    started_at: str
    finished_at: str
    status: str
    red_evidence_sha256: str | None
    green_evidence_sha256: str | None
    final_diff_paths: tuple[str, ...]
    exit_code: int
    finding_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.final_diff_paths) is not tuple:
            raise LaneError("lane result final diff paths must be an immutable tuple")
        if type(self.finding_fingerprints) is not tuple:
            raise LaneError("lane result finding fingerprints must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class CompanionRequest:
    lane_name: str
    lane_sha256: str
    mode: str
    owned_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    out_of_scope_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]
    red_argv: tuple[str, ...]
    green_argv: tuple[str, ...]
    acquire_heavy_lock: bool

    def __post_init__(self) -> None:
        for name in (
            "owned_paths",
            "protected_paths",
            "out_of_scope_paths",
            "frozen_paths",
            "red_argv",
            "green_argv",
        ):
            if type(getattr(self, name)) is not tuple:
                raise LaneError(f"companion request {name} must be an immutable tuple")

    def canonical_body(self) -> dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "lane_sha256": self.lane_sha256,
            "mode": self.mode,
            "owned_paths": list(self.owned_paths),
            "protected_paths": list(self.protected_paths),
            "out_of_scope_paths": list(self.out_of_scope_paths),
            "frozen_paths": list(self.frozen_paths),
            "red_argv": list(self.red_argv),
            "green_argv": list(self.green_argv),
            "acquire_heavy_lock": self.acquire_heavy_lock,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_body())


@dataclass(frozen=True, slots=True)
class CompanionResult:
    request_sha256: str
    result: LaneResult


class CompanionAdapter(Protocol):
    def run(self, request: CompanionRequest) -> CompanionResult: ...


def _parse_prerequisites(value: object) -> tuple[LanePrerequisite, ...]:
    if not isinstance(value, list) or not value:
        raise LaneError("prerequisites must not be empty")
    parsed: list[LanePrerequisite] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"kind", "sha256"}:
            raise LaneError("prerequisite keys do not match schema")
        kind = item["kind"]
        digest = item["sha256"]
        if kind not in _PREREQUISITE_KINDS or not _is_sha256(digest):
            raise LaneError("prerequisite identity is invalid")
        parsed.append(LanePrerequisite(str(kind), str(digest)))
    identities = tuple((item.kind, item.sha256) for item in parsed)
    if len(set(identities)) != len(identities):
        raise LaneError("duplicate prerequisite identity")
    return tuple(parsed)


def parse_lane_declaration(payload: Mapping[str, object]) -> LaneDeclaration:
    if set(payload) != _LANE_KEYS:
        raise LaneError("lane declaration keys do not match the exact schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise LaneError("lane schema_version must be exact integer 1")
    name = payload["name"]
    if not isinstance(name, str) or not name:
        raise LaneError("lane name must be non-empty")
    mode = payload["mode"]
    if mode not in _MODES:
        raise LaneError("lane mode must be read-only or write")
    policy = payload["heavy_lock_policy"]
    if policy not in _HEAVY_LOCK_POLICIES:
        raise LaneError("lane heavy-lock policy is invalid")
    if mode == "read-only" and policy != "forbidden":
        raise LaneError("read-only lane cannot wait on or acquire the heavy lock")

    owned = _parse_paths(payload["owned_paths"], label="owned_paths")
    protected = _parse_paths(payload["protected_paths"], label="protected_paths")
    out_of_scope = _parse_paths(
        payload["out_of_scope_paths"], label="out_of_scope_paths"
    )
    frozen = _parse_paths(payload["frozen_paths"], label="frozen_paths")
    for boundary_name, boundaries in (
        ("protected", protected),
        ("out-of-scope", out_of_scope),
        ("frozen", frozen),
    ):
        # A broad ownership fence may contain an explicitly excluded child.  The
        # exclusion is enforced again against every result diff path.  What is
        # contradictory here is placing ownership wholly inside an exclusion.
        if any(_contains(boundary, path) for path in owned for boundary in boundaries):
            raise LaneError(f"owned paths overlap {boundary_name} boundary")

    return LaneDeclaration(
        schema_version=1,
        name=name,
        mode=str(mode),
        owned_paths=owned,
        protected_paths=protected,
        out_of_scope_paths=out_of_scope,
        frozen_paths=frozen,
        red_argv=validate_direct_argv(payload["red_argv"], label="RED argv"),
        green_argv=validate_direct_argv(payload["green_argv"], label="GREEN argv"),
        prerequisites=_parse_prerequisites(payload["prerequisites"]),
        heavy_lock_policy=str(policy),
    )


def lane_fingerprint(lane: LaneDeclaration) -> str:
    _validate_lane_declaration_value(lane)
    return _canonical_sha256(lane.canonical_body())


def _validate_lane_declaration_value(lane: LaneDeclaration) -> LaneDeclaration:
    if not isinstance(lane, LaneDeclaration):
        raise LaneError("lane declaration has invalid type")
    parsed = parse_lane_declaration(lane.canonical_body())
    if lane != parsed:
        raise LaneError("lane declaration is not an exact immutable parsed value")
    return parsed


def plan_lanes(
    lanes: Sequence[LaneDeclaration],
    *,
    available_prerequisites: Sequence[LanePrerequisite],
    lock_state: str,
    owner_known: bool,
) -> tuple[LanePlan, ...]:
    lanes = tuple(_validate_lane_declaration_value(lane) for lane in lanes)
    if lock_state not in _LOCK_STATES:
        raise LaneError(f"unknown heavy-lock state {lock_state!r}")
    if len({lane.name for lane in lanes}) != len(lanes):
        raise LaneError("lane names must be unique")
    writes = tuple(lane for lane in lanes if lane.mode == "write")
    for index, first in enumerate(writes):
        for second in writes[index + 1 :]:
            if any(
                _overlaps(left, right)
                for left in first.owned_paths
                for right in second.owned_paths
            ):
                raise LaneError(
                    f"simultaneous write lane ownership overlap: "
                    f"{first.name!r} and {second.name!r}"
                )

    for writer in writes:
        for other in lanes:
            if other is writer:
                continue
            for boundary_name, boundaries in (
                ("protected", other.protected_paths),
                ("out-of-scope", other.out_of_scope_paths),
                ("frozen", other.frozen_paths),
            ):
                if any(
                    _overlaps(owned, boundary)
                    for owned in writer.owned_paths
                    for boundary in boundaries
                ):
                    raise LaneError(
                        f"write lane {writer.name!r} overlaps {boundary_name} "
                        f"boundary declared by lane {other.name!r}"
                    )

    if any(not isinstance(item, LanePrerequisite) for item in available_prerequisites):
        raise LaneError("available prerequisites must preserve kind and digest identity")
    if any(
        item.kind not in _PREREQUISITE_KINDS or not _is_sha256(item.sha256)
        for item in available_prerequisites
    ):
        raise LaneError("available prerequisite identity is invalid")
    available = {(item.kind, item.sha256) for item in available_prerequisites}
    plans: list[LanePlan] = []
    for lane in lanes:
        missing = tuple(
            prerequisite
            for prerequisite in lane.prerequisites
            if (prerequisite.kind, prerequisite.sha256) not in available
        )
        if missing:
            identities = ", ".join(
                f"{item.kind}:{item.sha256}" for item in missing
            )
            status = "blocked"
            reason = f"missing or stale prerequisite: {identities}"
            stuck = False
            acquire = False
        elif lane.mode == "read-only" or lane.heavy_lock_policy == "forbidden":
            status = "ready"
            reason = "all prerequisites satisfied; heavy work forbidden"
            stuck = False
            acquire = False
        elif lock_state == "held" and owner_known:
            status = "wait-policy"
            reason = "known live heavy-lock owner; wait by policy"
            stuck = False
            acquire = False
        elif lock_state == "held":
            status = "blocked"
            reason = "heavy lock held with unknown owner; manual diagnosis required"
            stuck = True
            acquire = False
        elif lock_state == "unobserved":
            status = "blocked"
            reason = "heavy lock state was not observed"
            stuck = False
            acquire = False
        else:
            status = "ready"
            reason = "all prerequisites satisfied; heavy lock available"
            stuck = False
            acquire = True
        plans.append(
            LanePlan(
                lane=lane,
                available_prerequisites=tuple(available_prerequisites),
                lock_state=lock_state,
                owner_known=owner_known,
                status=status,
                reason=reason,
                red_argv=lane.red_argv,
                green_argv=lane.green_argv,
                acquire_heavy_lock=acquire,
                stuck=stuck,
            )
        )
    return tuple(plans)


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or _UTC_RFC3339_SECONDS.fullmatch(value) is None:
        raise LaneError("lane result timestamp must be UTC RFC3339")
    try:
        timestamp = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise LaneError("lane result timestamp is invalid") from error
    return timestamp


def validate_lane_result(lane: LaneDeclaration, result: LaneResult) -> LaneResult:
    if result.lane_name != lane.name or result.lane_sha256 != lane_fingerprint(lane):
        raise LaneError("lane result has stale lane identity")
    if result.status not in _RESULT_STATUSES:
        raise LaneError("lane result status is invalid")
    started = _parse_timestamp(result.started_at)
    finished = _parse_timestamp(result.finished_at)
    if finished < started:
        raise LaneError("lane result timestamp order is invalid")
    if not _is_sha256(result.red_evidence_sha256):
        raise LaneError("lane result is missing valid RED evidence")
    if not _is_sha256(result.green_evidence_sha256):
        raise LaneError("lane result is missing valid GREEN evidence")
    if not isinstance(result.exit_code, int) or isinstance(result.exit_code, bool):
        raise LaneError("lane result exit code is invalid")
    if result.status == "passed" and result.exit_code != 0:
        raise LaneError("passed lane result must have exit code zero")
    if result.status in {"failed", "blocked"} and result.exit_code == 0:
        raise LaneError(f"{result.status} lane result must have a nonzero exit code")

    if type(result.final_diff_paths) is not tuple:
        raise LaneError("lane result final diff paths must be an immutable tuple")
    diff_paths = tuple(
        _safe_path(path, label="final diff paths") for path in result.final_diff_paths
    )
    if len(set(diff_paths)) != len(diff_paths):
        raise LaneError("lane result final diff paths contain duplicates")
    if lane.mode == "read-only" and diff_paths:
        raise LaneError("read-only lane result must have an empty diff")
    for path in diff_paths:
        if any(_overlaps(boundary, path) for boundary in lane.protected_paths):
            raise LaneError(f"lane result touches protected path {path!r}")
        if any(_overlaps(boundary, path) for boundary in lane.frozen_paths):
            raise LaneError(f"lane result expands frozen boundary at {path!r}")
        if any(_overlaps(boundary, path) for boundary in lane.out_of_scope_paths):
            raise LaneError(f"lane result touches out-of-scope path {path!r}")
        if not any(_contains(boundary, path) for boundary in lane.owned_paths):
            raise LaneError(f"lane result path is outside ownership fence: {path!r}")
    if type(result.finding_fingerprints) is not tuple or not all(
        _is_sha256(fingerprint) for fingerprint in result.finding_fingerprints
    ):
        raise LaneError(
            "lane result finding fingerprints must be an immutable tuple of SHA-256 values"
        )
    if len(set(result.finding_fingerprints)) != len(result.finding_fingerprints):
        raise LaneError("lane result finding fingerprints contain duplicates")
    return LaneResult(
        lane_name=result.lane_name,
        lane_sha256=result.lane_sha256,
        started_at=result.started_at,
        finished_at=result.finished_at,
        status=result.status,
        red_evidence_sha256=result.red_evidence_sha256,
        green_evidence_sha256=result.green_evidence_sha256,
        final_diff_paths=diff_paths,
        exit_code=result.exit_code,
        finding_fingerprints=tuple(result.finding_fingerprints),
    )


def _validate_plan(
    plan: LanePlan,
    *,
    available_prerequisites: Sequence[LanePrerequisite],
    lock_state: str,
    owner_known: bool,
) -> LanePlan:
    if (
        type(plan.available_prerequisites) is not tuple
        or type(plan.owner_known) is not bool
    ):
        raise LaneError("lane plan inputs are not exact immutable values")
    expected = plan_lanes(
        (plan.lane,),
        available_prerequisites=available_prerequisites,
        lock_state=lock_state,
        owner_known=owner_known,
    )[0]
    if plan != expected:
        raise LaneError("lane plan fields do not match authenticated planning inputs")
    return expected


def _request_for_plan(plan: LanePlan) -> CompanionRequest:
    lane = plan.lane
    return CompanionRequest(
        lane_name=lane.name,
        lane_sha256=lane_fingerprint(lane),
        mode=lane.mode,
        owned_paths=lane.owned_paths,
        protected_paths=lane.protected_paths,
        out_of_scope_paths=lane.out_of_scope_paths,
        frozen_paths=lane.frozen_paths,
        red_argv=plan.red_argv,
        green_argv=plan.green_argv,
        acquire_heavy_lock=plan.acquire_heavy_lock,
    )


def execute_lane(
    plan: LanePlan,
    *,
    adapter: CompanionAdapter | None,
    dry_run: bool,
    available_prerequisites: Sequence[LanePrerequisite],
    lock_state: str,
    owner_known: bool,
) -> LaneResult | None:
    plan = _validate_plan(
        plan,
        available_prerequisites=available_prerequisites,
        lock_state=lock_state,
        owner_known=owner_known,
    )
    if dry_run:
        return None
    if plan.status != "ready":
        raise LaneError(f"lane is not executable: {plan.status}: {plan.reason}")
    if adapter is None:
        raise LaneError("lane execution requires an injected companion adapter")
    request = _request_for_plan(plan)
    response = adapter.run(request)
    if not isinstance(response, CompanionResult) or response.request_sha256 != request.sha256:
        raise LaneError("companion result request identity mismatch")
    return validate_lane_result(plan.lane, response.result)


_TASK6_VERIFICATION_SHA256 = "0b12d95605e06e9928a280c16c40c276466f389931ebc9cfbff843d1ea1f060d"
_TASK6_REVIEW_SHA256 = "95ea3520eed4228d1c13fdca273d8ca0b22566f9eb4f1188d462556be90e120c"

_BUILTIN_PAYLOADS: dict[str, dict[str, object]] = {
    "recovery-review-audit": {
        "schema_version": 1,
        "name": "recovery-review-audit",
        "mode": "read-only",
        "owned_paths": [
            "src/mlx_vq/recovery_campaign/review.py",
            "src/mlx_vq/recovery_campaign/lanes.py",
            "tests/test_recovery_campaign_review.py",
            "tests/test_recovery_campaign_lanes.py",
        ],
        "protected_paths": [".keep-heavy-job.lock", "runs"],
        "out_of_scope_paths": [
            "artifacts",
            "glm52-community-wow-section-handoff-20260710.md",
            "glm52-recovery-campaign-handoff-20260710.md",
        ],
        "frozen_paths": [
            "src/mlx_vq/recovery_campaign/controller.py",
            "src/mlx_vq/recovery_campaign/ledger.py",
            "src/mlx_vq/recovery_campaign/materializer.py",
        ],
        "red_argv": [
            ".venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests/test_recovery_campaign_review.py",
            "tests/test_recovery_campaign_lanes.py",
        ],
        "green_argv": [
            ".venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests/test_recovery_campaign_review.py",
            "tests/test_recovery_campaign_lanes.py",
            "tests/test_recovery_campaign_cli.py",
        ],
        "prerequisites": [
            {"kind": "verification", "sha256": _TASK6_VERIFICATION_SHA256},
            {"kind": "review", "sha256": _TASK6_REVIEW_SHA256},
        ],
        "heavy_lock_policy": "forbidden",
    }
}

AVAILABLE_LANES = tuple(sorted(_BUILTIN_PAYLOADS))


def get_lane(name: str) -> LaneDeclaration:
    try:
        payload = _BUILTIN_PAYLOADS[name]
    except KeyError as error:
        available = ", ".join(AVAILABLE_LANES)
        raise LaneError(f"unknown lane {name!r}; available: {available}") from error
    return parse_lane_declaration(payload)


def render_lane_plan(
    plan: LanePlan,
    *,
    as_json: bool,
    available_prerequisites: Sequence[LanePrerequisite],
    lock_state: str,
    owner_known: bool,
) -> str:
    plan = _validate_plan(
        plan,
        available_prerequisites=available_prerequisites,
        lock_state=lock_state,
        owner_known=owner_known,
    )
    body = {
        "acquire_heavy_lock": plan.acquire_heavy_lock,
        "dry_run": True,
        "green_argv": list(plan.green_argv),
        "lane": plan.lane.name,
        "lane_sha256": lane_fingerprint(plan.lane),
        "mode": plan.lane.mode,
        "reason": plan.reason,
        "red_argv": list(plan.red_argv),
        "status": plan.status,
        "stuck": plan.stuck,
        "worker_launched": False,
    }
    if as_json:
        return json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
    red = shlex.join(plan.red_argv)
    green = shlex.join(plan.green_argv)
    return (
        f"Lane: {plan.lane.name}\n"
        f"Mode: {plan.lane.mode}\n"
        f"Status: {plan.status}\n"
        f"Reason: {plan.reason}\n"
        f"RED argv: {red}\n"
        f"GREEN argv: {green}\n"
        f"Acquire heavy lock: {'yes' if plan.acquire_heavy_lock else 'no'}\n"
        f"Stuck: {'yes' if plan.stuck else 'no'}\n"
        "Worker launched: no\n"
    )


__all__ = [
    "AVAILABLE_LANES",
    "CompanionAdapter",
    "CompanionRequest",
    "CompanionResult",
    "LaneDeclaration",
    "LaneError",
    "LanePlan",
    "LanePrerequisite",
    "LaneResult",
    "execute_lane",
    "get_lane",
    "lane_fingerprint",
    "parse_lane_declaration",
    "plan_lanes",
    "render_lane_plan",
    "validate_lane_result",
    "validate_direct_argv",
]
