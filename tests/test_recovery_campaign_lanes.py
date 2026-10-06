from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mlx_vq.recovery_campaign.lanes import (
    AVAILABLE_LANES,
    CompanionResult,
    LaneError,
    LanePrerequisite,
    LaneResult,
    execute_lane,
    get_lane,
    lane_fingerprint,
    parse_lane_declaration,
    plan_lanes,
    render_lane_plan,
    validate_lane_result,
)


V = "a" * 64
R = "b" * 64


def _payload(name: str = "write-review") -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": name,
        "mode": "write",
        "owned_paths": ["src/mlx_vq/recovery_campaign/review.py"],
        "protected_paths": [".keep-heavy-job.lock", "runs"],
        "out_of_scope_paths": ["docs"],
        "frozen_paths": ["src/mlx_vq/recovery_campaign/controller.py"],
        "red_argv": [
            ".venv/bin/python", "-m", "pytest", "-q", "tests/test_review.py"
        ],
        "green_argv": [
            ".venv/bin/python", "-m", "pytest", "-q", "tests/test_review.py"
        ],
        "prerequisites": [
            {"kind": "verification", "sha256": V},
            {"kind": "review", "sha256": R},
        ],
        "heavy_lock_policy": "forbidden",
    }


def _lane(name: str = "write-review"):
    return parse_lane_declaration(_payload(name))


def _prerequisites(lane=None):
    return (lane or _lane()).prerequisites


def _authority(plan):
    return {
        "available_prerequisites": plan.available_prerequisites,
        "lock_state": plan.lock_state,
        "owner_known": plan.owner_known,
    }


def test_lane_declaration_is_strict_immutable_and_fingerprinted() -> None:
    lane = _lane()
    assert len(lane_fingerprint(lane)) == 64
    with pytest.raises(Exception):
        lane.mode = "read-only"  # type: ignore[misc]
    changed = _payload()
    changed["green_argv"] = [
        ".venv/bin/python", "-m", "pytest", "tests/other.py"
    ]
    assert lane_fingerprint(parse_lane_declaration(changed)) != lane_fingerprint(lane)


def test_lane_schema_version_rejects_boolean_alias_for_one() -> None:
    payload = _payload()
    payload["schema_version"] = True
    with pytest.raises(LaneError, match="schema"):
        parse_lane_declaration(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown",
        "unsafe-path",
        "unsafe-argv",
        "env-wrapper",
        "argv-traversal",
        "argv-control",
        "absolute-executable",
        "protected-owned",
        "frozen-owned",
        "missing-red",
        "missing-green",
        "duplicate-prereq",
        "read-only-heavy",
    ),
)
def test_lane_declaration_rejects_unsafe_or_contradictory_scope(mutation: str) -> None:
    payload = _payload()
    if mutation == "unknown":
        payload["extra"] = True
    elif mutation == "unsafe-path":
        payload["owned_paths"] = ["../escape"]
    elif mutation == "unsafe-argv":
        payload["red_argv"] = ["/bin/sh", "-c", "pytest; rm -rf /tmp/x"]
    elif mutation == "env-wrapper":
        payload["green_argv"] = ["/usr/bin/env", "bash", "-c", "pytest"]
    elif mutation == "argv-traversal":
        payload["red_argv"] = [
            ".venv/bin/python", "-m", "pytest", "../tests/test.py"
        ]
    elif mutation == "argv-control":
        payload["red_argv"] = [
            ".venv/bin/python", "-m", "pytest", "tests/test.py\nnext"
        ]
    elif mutation == "absolute-executable":
        payload["red_argv"] = ["/tmp/python", "-m", "pytest", "tests/test.py"]
    elif mutation == "protected-owned":
        payload["owned_paths"] = ["runs"]
    elif mutation == "frozen-owned":
        payload["frozen_paths"] = ["src/mlx_vq"]
    elif mutation == "missing-red":
        payload["red_argv"] = []
    elif mutation == "missing-green":
        payload["green_argv"] = []
    elif mutation == "duplicate-prereq":
        payload["prerequisites"] = [
            {"kind": "review", "sha256": R},
            {"kind": "review", "sha256": R},
        ]
    else:
        payload["mode"] = "read-only"
        payload["heavy_lock_policy"] = "wait-if-owned"
    with pytest.raises(LaneError):
        parse_lane_declaration(payload)


@pytest.mark.parametrize(
    "argv",
    (
        ["python", "-m", "pytest", "tests/test_review.py"],
        ["pytest", "tests/test_review.py"],
        [".venv/bin/python", "-m", "pytest", "/tmp/attacker_test.py"],
        [".venv/bin/python", "-m", "pytest", "../tests/test_review.py"],
        [
            ".venv/bin/python", "-m", "pytest", "--rootdir=/tmp/attacker",
            "tests/test_review.py",
        ],
        [
            ".venv/bin/python", "-m", "pytest", "--confcutdir", "/tmp",
            "tests/test_review.py",
        ],
        [
            ".venv/bin/python", "-m", "pytest", "-p", "attacker",
            "tests/test_review.py",
        ],
        [".venv/bin/python", "-m", "pytest", "--pyargs", "attacker"],
        [
            ".venv/bin/python", "-m", "pytest", "-c", "/tmp/pytest.ini",
            "tests/test_review.py",
        ],
        [
            ".venv/bin/python", "-m", "pytest",
            "--override-ini=pythonpath=/tmp", "tests/test_review.py",
        ],
        [
            ".venv/bin/python", "-m", "pytest", "--basetemp=/tmp/x",
            "tests/test_review.py",
        ],
        [
            ".venv/bin/python", "-m", "pytest", "--junitxml=/tmp/x.xml",
            "tests/test_review.py",
        ],
    ),
)
def test_lane_argv_rejects_path_and_pytest_authority_injection(argv) -> None:
    payload = _payload()
    payload["red_argv"] = argv
    with pytest.raises(LaneError, match="argv|pytest|executable|target"):
        parse_lane_declaration(payload)


def test_simultaneous_write_lanes_cannot_overlap_ownership() -> None:
    first = _lane("first")
    payload = _payload("second")
    payload["owned_paths"] = ["src/mlx_vq/recovery_campaign"]
    second = parse_lane_declaration(payload)
    with pytest.raises(LaneError, match="overlap"):
        plan_lanes(
            (first, second),
            available_prerequisites=_prerequisites(first),
            lock_state="free",
            owner_known=False,
        )


@pytest.mark.parametrize("boundary", ("protected_paths", "out_of_scope_paths", "frozen_paths"))
def test_write_ownership_cannot_overlap_another_lane_boundary(boundary: str) -> None:
    first = _lane("first")
    payload = _payload("second")
    payload["owned_paths"] = ["src/other.py"]
    payload[boundary] = ["src/mlx_vq/recovery_campaign/review.py"]
    second = parse_lane_declaration(payload)
    with pytest.raises(LaneError, match="boundary"):
        plan_lanes(
            (first, second),
            available_prerequisites=(*first.prerequisites, *second.prerequisites),
            lock_state="free",
            owner_known=False,
        )


def test_stale_or_missing_prerequisite_blocks_lane() -> None:
    plan = plan_lanes(
        (_lane(),),
        available_prerequisites=(LanePrerequisite("verification", V),),
        lock_state="free",
        owner_known=False,
    )[0]
    assert plan.status == "blocked"
    assert "prerequisite" in plan.reason


def test_prerequisite_kind_is_part_of_exact_identity() -> None:
    lane = _lane()
    plan = plan_lanes(
        (lane,),
        available_prerequisites=(
            LanePrerequisite("review", V),
            LanePrerequisite("verification", R),
        ),
        lock_state="free",
        owner_known=False,
    )[0]
    assert plan.status == "blocked"
    assert "prerequisite" in plan.reason


def test_known_live_heavy_owner_is_policy_wait_not_stuck() -> None:
    payload = _payload("heavy-write")
    payload["heavy_lock_policy"] = "wait-if-owned"
    heavy = parse_lane_declaration(payload)
    waiting = plan_lanes(
        (heavy,),
        available_prerequisites=_prerequisites(heavy),
        lock_state="held",
        owner_known=True,
    )[0]
    assert waiting.status == "wait-policy"
    assert waiting.stuck is False
    blocked = plan_lanes(
        (heavy,),
        available_prerequisites=_prerequisites(heavy),
        lock_state="held",
        owner_known=False,
    )[0]
    assert blocked.status == "blocked"
    assert blocked.stuck is True


def test_unobserved_lock_blocks_heavy_lane_without_calling_it_stuck() -> None:
    payload = _payload("heavy-write")
    payload["heavy_lock_policy"] = "wait-if-owned"
    heavy = parse_lane_declaration(payload)
    plan = plan_lanes(
        (heavy,),
        available_prerequisites=_prerequisites(heavy),
        lock_state="unobserved",
        owner_known=False,
    )[0]
    assert plan.status == "blocked"
    assert plan.acquire_heavy_lock is False
    assert plan.stuck is False


def test_read_only_lane_never_waits_on_heavy_lock() -> None:
    payload = _payload("readonly")
    payload["mode"] = "read-only"
    readonly = parse_lane_declaration(payload)
    plan = plan_lanes(
        (readonly,),
        available_prerequisites=_prerequisites(readonly),
        lock_state="held",
        owner_known=True,
    )[0]
    assert plan.status == "ready"
    assert plan.acquire_heavy_lock is False


def test_plan_preserves_exact_direct_red_and_green_argv() -> None:
    lane = _lane()
    plan = plan_lanes(
        (lane,),
        available_prerequisites=_prerequisites(lane),
        lock_state="free",
        owner_known=False,
    )[0]
    assert plan.red_argv == lane.red_argv
    assert plan.green_argv == lane.green_argv
    assert all(isinstance(arg, str) for arg in plan.red_argv)
    assert plan.status == "ready"


def _result(lane=None, **changes: object) -> LaneResult:
    lane = lane or _lane()
    values: dict[str, object] = {
        "lane_name": lane.name,
        "lane_sha256": lane_fingerprint(lane),
        "started_at": "2026-07-11T12:00:00Z",
        "finished_at": "2026-07-11T12:01:00Z",
        "status": "passed",
        "red_evidence_sha256": "c" * 64,
        "green_evidence_sha256": "d" * 64,
        "final_diff_paths": lane.owned_paths,
        "exit_code": 0,
        "finding_fingerprints": ("e" * 64,),
    }
    values.update(changes)
    return LaneResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"lane_sha256": "f" * 64}, "stale"),
        ({"red_evidence_sha256": None}, "RED"),
        ({"green_evidence_sha256": None}, "GREEN"),
        ({"final_diff_paths": ("docs/outside.md",)}, "outside"),
        ({"final_diff_paths": ("runs/private",)}, "protected"),
        ({"finished_at": "2026-07-11T11:59:00Z"}, "timestamp"),
        ({"started_at": "2026-07-11Z"}, "timestamp"),
        ({"started_at": "2026-07-11T12:00Z"}, "timestamp"),
        ({"status": "failed", "exit_code": 0}, "exit code"),
        ({"status": "blocked", "exit_code": 0}, "exit code"),
    ),
)
def test_lane_result_revalidation_rejects_stale_or_out_of_scope_evidence(
    changes: dict[str, object], match: str
) -> None:
    lane = _lane()
    with pytest.raises(LaneError, match=match):
        validate_lane_result(lane, _result(lane, **changes))


def test_read_only_result_must_have_empty_diff() -> None:
    payload = _payload("readonly")
    payload["mode"] = "read-only"
    lane = parse_lane_declaration(payload)
    with pytest.raises(LaneError, match="read-only"):
        validate_lane_result(lane, _result(lane))


def test_lane_result_sequences_must_be_exact_immutable_tuples() -> None:
    lane = _lane()
    with pytest.raises(LaneError, match="tuple|immutable"):
        validate_lane_result(
            lane,
            _result(lane, final_diff_paths=list(lane.owned_paths)),
        )
    with pytest.raises(LaneError, match="tuple|immutable"):
        validate_lane_result(
            lane,
            _result(lane, finding_fingerprints=["e" * 64]),
        )


def test_directory_diff_cannot_hide_a_frozen_child_boundary() -> None:
    lane = _lane()
    with pytest.raises(LaneError, match="frozen"):
        validate_lane_result(
            lane,
            _result(lane, final_diff_paths=("src/mlx_vq/recovery_campaign",)),
        )


def test_injected_adapter_is_never_called_by_dry_run_and_result_is_revalidated() -> None:
    lane = _lane()
    plan = plan_lanes(
        (lane,),
        available_prerequisites=_prerequisites(lane),
        lock_state="free",
        owner_known=False,
    )[0]

    class Adapter:
        calls = 0

        def run(self, request):
            self.calls += 1
            return CompanionResult(request_sha256=request.sha256, result=_result(lane))

    adapter = Adapter()
    assert execute_lane(plan, adapter=adapter, dry_run=True, **_authority(plan)) is None
    assert adapter.calls == 0
    result = execute_lane(plan, adapter=adapter, dry_run=False, **_authority(plan))
    assert result is not None and result.status == "passed"
    assert adapter.calls == 1


def test_forged_plan_is_rejected_before_render_or_adapter_call() -> None:
    lane = _lane()
    plan = plan_lanes(
        (lane,),
        available_prerequisites=lane.prerequisites,
        lock_state="free",
        owner_known=False,
    )[0]
    forged = replace(
        plan,
        red_argv=("/bin/sh", "-c", "pytest"),
        green_argv=("/usr/bin/env", "bash", "-c", "pytest"),
        acquire_heavy_lock=True,
        status="ready",
        reason="forged",
        stuck=True,
        dry_run=False,
    )

    class Adapter:
        calls = 0

        def run(self, request):
            self.calls += 1
            return CompanionResult(request_sha256=request.sha256, result=_result(lane))

    adapter = Adapter()
    with pytest.raises(LaneError, match="plan"):
        render_lane_plan(forged, as_json=True, **_authority(plan))
    with pytest.raises(LaneError, match="plan"):
        execute_lane(
            forged,
            adapter=adapter,
            dry_run=False,
            **_authority(plan),
        )
    assert adapter.calls == 0


def test_coherent_forged_ready_plan_cannot_replace_external_lock_authority() -> None:
    payload = _payload("heavy")
    payload["heavy_lock_policy"] = "wait-if-owned"
    lane = parse_lane_declaration(payload)
    held = plan_lanes(
        (lane,),
        available_prerequisites=lane.prerequisites,
        lock_state="held",
        owner_known=True,
    )[0]
    forged_ready = plan_lanes(
        (lane,),
        available_prerequisites=lane.prerequisites,
        lock_state="free",
        owner_known=False,
    )[0]

    class Adapter:
        calls = 0

        def run(self, request):
            self.calls += 1
            return CompanionResult(request_sha256=request.sha256, result=_result(lane))

    adapter = Adapter()
    authority = {
        "available_prerequisites": held.available_prerequisites,
        "lock_state": held.lock_state,
        "owner_known": held.owner_known,
    }
    with pytest.raises(LaneError, match="plan|authorit"):
        render_lane_plan(forged_ready, as_json=True, **authority)
    with pytest.raises(LaneError, match="plan|authorit"):
        execute_lane(
            forged_ready,
            adapter=adapter,
            dry_run=False,
            **authority,
        )
    assert adapter.calls == 0


def test_forged_lane_declaration_is_reparsed_before_planning() -> None:
    lane = _lane()
    forged = replace(lane, red_argv=("/bin/sh", "-c", "pytest"))
    with pytest.raises(LaneError, match="argv|declaration"):
        plan_lanes(
            (forged,),
            available_prerequisites=lane.prerequisites,
            lock_state="free",
            owner_known=False,
        )


def test_adapter_request_identity_mismatch_is_rejected() -> None:
    lane = _lane()
    plan = plan_lanes(
        (lane,),
        available_prerequisites=_prerequisites(lane),
        lock_state="free",
        owner_known=False,
    )[0]

    class Adapter:
        def run(self, request):
            return CompanionResult(request_sha256="f" * 64, result=_result(lane))

    with pytest.raises(LaneError, match="request"):
        execute_lane(
            plan,
            adapter=Adapter(),
            dry_run=False,
            **_authority(plan),
        )


def test_builtin_lane_and_both_renderers_are_deterministic() -> None:
    assert "recovery-review-audit" in AVAILABLE_LANES
    lane = get_lane("recovery-review-audit")
    assert "glm52-community-wow-section-handoff-20260710.md" in lane.out_of_scope_paths
    assert "glm52-recovery-campaign-handoff-20260710.md" in lane.out_of_scope_paths
    plan = plan_lanes(
        (lane,),
        available_prerequisites=lane.prerequisites,
        lock_state="held",
        owner_known=True,
    )[0]
    first = render_lane_plan(plan, as_json=True, **_authority(plan))
    assert first == render_lane_plan(plan, as_json=True, **_authority(plan))
    payload = json.loads(first)
    assert payload["dry_run"] is True
    assert payload["status"] == "ready"
    human = render_lane_plan(plan, as_json=False, **_authority(plan))
    assert "Lane: recovery-review-audit" in human
    assert "Worker launched: no" in human


def test_human_renderer_losslessly_quotes_argv_with_spaces() -> None:
    payload = _payload("quoted")
    payload["red_argv"] = [
        ".venv/bin/python", "-m", "pytest", "tests/path with spaces.py"
    ]
    lane = parse_lane_declaration(payload)
    plan = plan_lanes(
        (lane,),
        available_prerequisites=lane.prerequisites,
        lock_state="free",
        owner_known=False,
    )[0]
    assert "'tests/path with spaces.py'" in render_lane_plan(
        plan, as_json=False, **_authority(plan)
    )


def test_unknown_builtin_lane_is_configuration_error() -> None:
    with pytest.raises(LaneError, match="unknown lane"):
        get_lane("missing")
