from __future__ import annotations

import re
from pathlib import Path

from mlx_vq.models.profiles import list_profiles


REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPO_ROOT / "docs" / "QUICKSTART.md"
HISTORICAL_AIR_GOAL = REPO_ROOT / "codex-goal-community-wow-keep-rc.md"


def test_quickstart_lists_every_registered_model_profile() -> None:
    text = QUICKSTART.read_text()
    profiles_section = text.split("Current profiles:", 1)[1].split(
        "Inspect a profile:", 1
    )[0]
    documented_profiles = re.findall(
        r"^- `([^`]+)`:", profiles_section, flags=re.MULTILINE
    )

    assert documented_profiles == list(list_profiles())


def test_quickstart_documents_the_honest_glm52_frontier() -> None:
    text = QUICKSTART.read_text()
    raw_frontier = text.split("## GLM-5.2-REAP Current Frontier", 1)[1].split(
        "\n## ", 1
    )[0]
    frontier = " ".join(raw_frontier.split())

    required_literals = (
        "uv run keep models show glm52-reap-504b-v2",
        "uv run keep get glm52-reap-504b-v2 --check-only",
        "uv run keep compile examples/glm52-reap.yaml",
        "6c9241aa05fb243a0edb7c804c213ec1cf5c920d",
        "63/63 shards",
        "about 308.829 GB",
        "all 225 routed groups",
        "98,433,923,808",
        "`1.03125` routed bits per weight (bpw)",
        "1.593443277857996",
        "one-token generation are proven without dense routed experts",
        "warm steady-state interval is memory-clean",
        "intentionally exits `1`",
        "profile's embedded `notes` field is frozen",
        "authenticated artifact identity",
        "There is no BF16 REAP teacher",
        "deterministically dequantized pinned FP4 source",
        "rough BF16-equivalent unquantized-weight memory estimate",
        "KEEP does not yet expose working `doctor`, `report`, or `chat` release surfaces",
        "`uv run keep build examples/glm52-reap.yaml` is not a runnable release command",
        "[`docs/GLM52_PIPELINE_READINESS.md`](GLM52_PIPELINE_READINESS.md)",
    )
    for literal in required_literals:
        assert literal in frontier

    blocker_text = raw_frontier.split(
        "four release-evidence blockers remain:", 1
    )[1].lstrip().split("\n\n", 1)[0]
    blockers = [
        item.rstrip(";.")
        for item in re.findall(r"^- (.+)$", blocker_text, flags=re.MULTILINE)
    ]
    assert blockers == [
        "dequantized-source teacher cache",
        "full-vocabulary source-relative eval",
        "route/math diagnostics",
        "same-machine pinned-FP4 benchmark",
    ]


def test_quickstart_local_links_resolve() -> None:
    text = QUICKSTART.read_text()
    relative_targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)

    assert "GLM52_PIPELINE_READINESS.md" in relative_targets
    for target in relative_targets:
        path = target.split("#", 1)[0]
        assert (QUICKSTART.parent / path).exists(), target


def test_obsolete_air_goal_is_marked_historical_and_superseded() -> None:
    text = HISTORICAL_AIR_GOAL.read_text()

    assert text.startswith("# Historical Codex Goal")
    assert "SUPERSEDED" in text
    assert "Do not execute this file as the active objective" in text
    assert "GLM-5.2-REAP-KEEP-504B" in text
    assert "docs/COMMUNITY_PRODUCT_PLAN.md" in text
