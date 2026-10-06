"""Thin argparse wiring for recovery campaign and verification commands."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from .config import CampaignConfigError, load_campaign_config
from .controller import (
    advance_campaign,
    authorize_retry,
    default_ledger_path,
    render_advance,
)
from .ledger import LedgerError, load_ledger
from .lanes import (
    AVAILABLE_LANES,
    LaneDeclaration,
    LaneError,
    LanePrerequisite,
    execute_lane,
    get_lane,
    plan_lanes,
    render_lane_plan,
)
from .observer import observe_campaign
from .render import (
    CampaignHandoffError,
    build_campaign_handoff,
    handoff_payload,
    observe_repository_identity,
    publish_handoff,
    render_handoff_json,
    render_handoff_markdown,
    render_handoff_terminal,
    render_publication_receipt,
    render_status,
)
from .review import ReviewRecordError, _read_regular_bytes_no_follow
from .verification import (
    AVAILABLE_VERIFICATION_PROFILES,
    VerificationProfileError,
    render_verification_result,
    run_verification_profile,
)


DEFAULT_CONFIG = Path("recipes/glm52_recovery_campaign_v1_20260711.yaml")
_TASK6_AUTHORITY_COMMIT = "f033debb6c89c648a9579ba1471b135900332d95"
_TASK6_REVIEW_PACKAGE = Path(
    ".superpowers/sdd/recovery-campaign-task-6-review.diff"
)
_GIT = "/usr/bin/git"


def _authenticate_regular_sha256(path: Path) -> str | None:
    """Hash one no-follow, nonblocking regular leaf and revalidate identity."""

    try:
        payload = _read_regular_bytes_no_follow(path, label="lane prerequisite")
    except ReviewRecordError:
        return None
    return hashlib.sha256(payload).hexdigest()


def _available_lane_prerequisites(
    lane: LaneDeclaration, repo_root: Path
) -> tuple[LanePrerequisite, ...]:
    """Authenticate prerequisite bytes independently from the declaration."""

    expected = {(item.kind, item.sha256) for item in lane.prerequisites}
    available: list[LanePrerequisite] = []
    review_digest = _authenticate_regular_sha256(repo_root / _TASK6_REVIEW_PACKAGE)
    if review_digest is not None and ("review", review_digest) in expected:
        available.append(LanePrerequisite("review", review_digest))

    try:
        ancestry = subprocess.run(
            [
                _GIT,
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                _TASK6_AUTHORITY_COMMIT,
                "HEAD",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        ancestry = None
    if ancestry is None or ancestry.returncode != 0:
        return tuple(available)
    try:
        completed = subprocess.run(
            [
                _GIT,
                "-C",
                str(repo_root),
                "show",
                "--no-ext-diff",
                "--no-textconv",
                "--format=",
                "--binary",
                _TASK6_AUTHORITY_COMMIT,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0:
        verification_digest = hashlib.sha256(completed.stdout).hexdigest()
        if ("verification", verification_digest) in expected:
            available.append(LanePrerequisite("verification", verification_digest))
    return tuple(available)


def _cmd_status(args: argparse.Namespace) -> int:
    try:
        config = load_campaign_config(args.config)
    except CampaignConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    observation = observe_campaign(config, repo_root=Path.cwd())
    print(render_status(observation, as_json=args.json), end="")
    return 3 if observation.contradictions else 0


def _cmd_advance(args: argparse.Namespace) -> int:
    try:
        config = load_campaign_config(args.config)
        result = advance_campaign(
            config,
            repo_root=Path.cwd(),
            ledger_path=args.ledger,
            dry_run=args.dry_run,
        )
    except (CampaignConfigError, LedgerError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render_advance(result, as_json=args.json), end="")
    if result.exit_code:
        return result.exit_code
    if result.action == "blocked":
        return 3
    if not args.dry_run and result.action == "wait":
        return 4
    return 0


def _cmd_authorize_retry(args: argparse.Namespace) -> int:
    try:
        config = load_campaign_config(args.config)
        event = authorize_retry(
            config,
            repo_root=Path.cwd(),
            ledger_path=args.ledger,
            transition_name=args.transition,
            reason=args.reason,
        )
    except (CampaignConfigError, LedgerError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(event.event_sha256)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        result = run_verification_profile(args.profile, Path.cwd())
    except (OSError, VerificationProfileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render_verification_result(result, as_json=args.json), end="")
    return 0 if result.commit_ready else 1


def _cmd_lane_run(args: argparse.Namespace) -> int:
    try:
        lane = get_lane(args.name)
    except LaneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        available_prerequisites = _available_lane_prerequisites(lane, Path.cwd())
        lock_state = "unobserved"
        owner_known = False
        plan = plan_lanes(
            (lane,),
            available_prerequisites=available_prerequisites,
            # This thin command does not inspect or acquire the live lock.  A
            # read-only lane remains ready; a heavy lane would conservatively
            # block until an injected observer supplied an exact lock state.
            lock_state=lock_state,
            owner_known=owner_known,
        )[0]
    except LaneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if plan.status != "ready":
        print(f"error: {plan.reason}", file=sys.stderr)
        return 3
    try:
        if not args.dry_run:
            execute_lane(
                plan,
                adapter=None,
                dry_run=False,
                available_prerequisites=available_prerequisites,
                lock_state=lock_state,
                owner_known=owner_known,
            )
    except LaneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    try:
        output = render_lane_plan(
            plan,
            as_json=args.json,
            available_prerequisites=available_prerequisites,
            lock_state=lock_state,
            owner_known=owner_known,
        )
    except LaneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output, end="")
    return 0


def _cmd_handoff(args: argparse.Namespace) -> int:
    try:
        config = load_campaign_config(args.config)
        repo_root = Path.cwd()
        ledger_path = (
            args.ledger
            if args.ledger is not None
            else default_ledger_path(config, repo_root)
        )
        if (args.ledger_head_sha256 is None) != (
            args.ledger_event_count is None
        ):
            raise CampaignHandoffError(
                "ledger head and event-count anchors must be supplied together"
            )
        events = load_ledger(
            ledger_path,
            expected_head_sha256=args.ledger_head_sha256,
            expected_event_count=args.ledger_event_count,
        )
        observation = observe_campaign(config, repo_root=repo_root)
        repository = observe_repository_identity(repo_root)
        handoff = build_campaign_handoff(
            config,
            events,
            observation,
            repository,
            ledger_head_sha256=args.ledger_head_sha256,
            ledger_event_count=args.ledger_event_count,
        )
        if args.out is not None:
            with publish_handoff(
                args.out,
                render_handoff_markdown(handoff),
                repo_root=repo_root,
            ) as receipt:
                output = render_publication_receipt(receipt, as_json=args.json)
        else:
            output = (
                render_handoff_json(handoff)
                if args.json
                else render_handoff_terminal(handoff)
            )
    except (CampaignConfigError, CampaignHandoffError, LedgerError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output, end="")
    return 3 if handoff_payload(handoff)["contradictions"] else 0


def _configure_verify_parser(parser: argparse.ArgumentParser) -> None:
    inventory = ", ".join(AVAILABLE_VERIFICATION_PROFILES)
    parser.add_argument(
        "profile",
        metavar="PROFILE",
        help=f"verification profile; available: {inventory}",
    )
    parser.add_argument("--json", action="store_true", help="emit deterministic JSON")
    parser.set_defaults(func=_cmd_verify)


def _wide_help_formatter(prog: str) -> argparse.HelpFormatter:
    return argparse.HelpFormatter(prog, max_help_position=30, width=160)


def configure_recovery_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach recovery campaign commands and the top-level verification alias."""

    verify = subparsers.add_parser(
        "verify",
        help="run a named, scope-authenticated verification profile",
        formatter_class=_wide_help_formatter,
    )
    _configure_verify_parser(verify)

    recovery = subparsers.add_parser(
        "recovery", help="observe and control recovery campaigns"
    )
    recovery_subparsers = recovery.add_subparsers(
        dest="recovery_command", required=True
    )
    campaign = recovery_subparsers.add_parser(
        "campaign", help="recovery campaign operations"
    )
    campaign_subparsers = campaign.add_subparsers(
        dest="campaign_command", required=True
    )
    status = campaign_subparsers.add_parser(
        "status", help="observe campaign state without mutation"
    )
    status.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"campaign matrix (default: {DEFAULT_CONFIG})",
    )
    status.add_argument("--json", action="store_true", help="emit deterministic JSON")
    status.set_defaults(func=_cmd_status)
    advance = campaign_subparsers.add_parser(
        "advance", help="plan or launch one guarded campaign transition"
    )
    advance.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"campaign matrix (default: {DEFAULT_CONFIG})",
    )
    advance.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help=(
            "campaign ledger for status/dry-run; mutating advance requires the "
            "canonical ledger under artifacts/quality"
        ),
    )
    advance.add_argument(
        "--dry-run",
        action="store_true",
        help="observe and print the next state without filesystem mutation",
    )
    advance.add_argument("--json", action="store_true", help="emit deterministic JSON")
    advance.set_defaults(func=_cmd_advance)
    authorize_retry_parser = campaign_subparsers.add_parser(
        "authorize-retry",
        help="authorize one retry of the latest failed transition terminal",
    )
    authorize_retry_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"campaign matrix (default: {DEFAULT_CONFIG})",
    )
    authorize_retry_parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="campaign ledger (default: canonical ignored campaign ledger)",
    )
    authorize_retry_parser.add_argument(
        "--transition",
        required=True,
        help="declared transition whose latest failed terminal is authorized",
    )
    authorize_retry_parser.add_argument(
        "--reason",
        required=True,
        help="operator reason recorded in the append-only ledger",
    )
    authorize_retry_parser.set_defaults(func=_cmd_authorize_retry)
    campaign_verify = campaign_subparsers.add_parser(
        "verify",
        help="run a named, scope-authenticated verification profile",
        formatter_class=_wide_help_formatter,
    )
    _configure_verify_parser(campaign_verify)
    handoff = campaign_subparsers.add_parser(
        "handoff", help="render an evidence-bound recovery campaign handoff"
    )
    handoff.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"campaign matrix (default: {DEFAULT_CONFIG})",
    )
    handoff.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="verified campaign ledger (default: canonical ignored campaign ledger)",
    )
    handoff.add_argument(
        "--ledger-head-sha256",
        default=None,
        help="externally retained exact ledger head SHA-256",
    )
    handoff.add_argument(
        "--ledger-event-count",
        type=int,
        default=None,
        help="externally retained exact ledger event count",
    )
    handoff.add_argument(
        "--out",
        type=Path,
        default=None,
        help="atomically publish Markdown under artifacts/quality",
    )
    handoff.add_argument("--json", action="store_true", help="emit deterministic JSON")
    handoff.set_defaults(func=_cmd_handoff)

    lanes = recovery_subparsers.add_parser(
        "lanes", help="plan strict review and companion lanes"
    )
    lanes_subparsers = lanes.add_subparsers(dest="lanes_command", required=True)
    lane_run = lanes_subparsers.add_parser(
        "run",
        help="render or execute one declared lane",
        formatter_class=_wide_help_formatter,
    )
    lane_run.add_argument(
        "name",
        metavar="NAME",
        help=f"lane declaration; available: {', '.join(AVAILABLE_LANES)}",
    )
    lane_run.add_argument(
        "--dry-run",
        action="store_true",
        help="render the exact plan without lock, filesystem, Git, or worker mutation",
    )
    lane_run.add_argument("--json", action="store_true", help="emit deterministic JSON")
    lane_run.set_defaults(func=_cmd_lane_run)


__all__ = ["configure_recovery_parser"]
